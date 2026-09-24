// The DSP side of the DDR read probe: the implementation of ddrbw.idl.
//
// The library runs in the unsigned protection domain (PD) of the compute DSP, as the llama.cpp
// backend does. It measures how fast the DSP reads a large buffer of the host from the DDR:
//
//   HVX loads   Each thread adds the 32-bit words of its part with vector loads through the L2
//               cache. An l2fetch box of the next chunks runs ahead of the loads.
//   DMA         Each thread moves its part chunk by chunk from the DDR into its slots of the VTCM
//               with the user DMA engine, and keeps "depth" descriptors in flight. The descriptor
//               code is dma-queue.h of the backend, thus the descriptors are the descriptors of the
//               GEMV kernels. A thread adds the first 128 bytes of each chunk, or all of its bytes
//               with DDRBW_FLAG_SUM.
//
// Each run starts all threads at one time (a spin barrier) and stops them at one deadline. The
// checksum of each thread lets the host prove that each byte that the run counts was read.
//
// The library also answers the latency tests of the host: an empty FastRPC call (nop), a dspqueue
// round trip in three wait modes (queue_start, queue_stop), and a fence round trip in shared
// memory (fence), with the fence code of htp/htp-fence.h.
//
// The power votes (setup) are the votes of htp/main.c, plus the optional votes that the backend
// does not send. Refer to ddrbw_common.h for the mask.

#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_dcvs.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <dspqueue.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <qurt_memory.h>
#include <qurt_thread.h>

#include "hexagon_protos.h"
#include "hexagon_types.h"

#include "dma-queue.h"
#include "ddrbw.h"
#include "ddrbw_common.h"

#define DDRBW_MAX_THREADS  8u
#define DDRBW_MAX_MAPS     8u
#define DDRBW_STACK_BYTES  (32u * 1024u)
#define DDRBW_ACQUIRE_US   2000000u
#define DDRBW_QTIMER_MHZ10 192u  // the qtimer runs at 19.2 MHz: ticks = us * 192 / 10
#define DDRBW_BARRIER_US   1000000u

// One mapping of a host buffer.
struct ddrbw_map {
    int      fd;
    uint8_t *va;
    uint32_t size;
};

// The state of the library. One handle at a time uses it.
static struct {
    int              setup_done;
    unsigned int     rctx;
    uint8_t *        vtcm;
    uint32_t         vtcm_size;
    uint32_t         n_hvx;
    struct ddrbw_map maps[DDRBW_MAX_MAPS];

    // The queue test
    dspqueue_t    queue;
    qurt_thread_t qthread;
    void *        qstack;
    uint32_t      qmode;
    atomic_uint   qserved;
    atomic_int    qerr;
} g;

// The addresses are the identities of the power clients. Each family of votes has its own client,
// thus a vote of one family does not replace a vote of the other.
static int g_client_backend;
static int g_client_ceng;
static int g_client_bw;

// The parameters of one read run that all threads share.
struct job {
    uint32_t    chunk;
    uint32_t    row;
    uint32_t    dst_row;
    uint32_t    rows;       // rows of one chunk
    uint32_t    slot;       // the VTCM bytes of one chunk
    uint32_t    depth;
    uint32_t    pf;
    uint32_t    flags;
    uint32_t    box_w;      // the l2fetch box of one chunk: box_w bytes, box_h rows
    uint32_t    box_h;
    uint64_t    deadline;   // the qtimer count at which the threads stop
    atomic_uint ready;
    atomic_uint go;
};

// One thread of a read run.
struct worker {
    struct job *            job;
    uint32_t                kind;
    const uint8_t *         part;
    uint32_t                part_start;
    uint32_t                part_len;
    uint32_t                nchunks;
    uint8_t *               vtcm;
    void *                  qmem;
    dma_queue *             q;
    struct htp_thread_trace trace;
    void *                  stack;
    qurt_thread_t           tid;
    int                     started;

    uint64_t t_start;
    uint64_t t_end;
    uint64_t chunks;
    uint64_t pcycles;
    uint32_t sum;
};

// ---- helpers

static inline uint64_t us_to_ticks(uint64_t us) {
    return us * DDRBW_QTIMER_MHZ10 / 10u;
}

static inline void l2fetch_box(const void * p, uint32_t width, uint32_t height) {
    const uint64_t control = Q6_P_combine_RR(width, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((void *) p, control);
}

// The 32-bit sum of the words of four accumulators. O(1).
static uint32_t reduce4(HVX_Vector a0, HVX_Vector a1, HVX_Vector a2, HVX_Vector a3) {
    HVX_Vector __attribute__((aligned(128))) v = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(a0, a1), Q6_Vw_vadd_VwVw(a2, a3));
    const uint32_t * w = (const uint32_t *) &v;
    uint32_t         s = 0;
    for (int i = 0; i < 32; i++) {
        s += w[i];
    }
    return s;
}

static struct ddrbw_map * find_map(int fd) {
    for (uint32_t i = 0; i < DDRBW_MAX_MAPS; i++) {
        if (g.maps[i].va != NULL && g.maps[i].fd == fd) {
            return &g.maps[i];
        }
    }
    return NULL;
}

// The fence code of htp/htp-fence.h: a store, a barrier, then a clean and invalidate of the line.
static inline void fence_write(volatile uint32_t * slot, uint32_t seq) {
    atomic_store((atomic_uint *) slot, seq);
    asm volatile("syncht" : : : "memory");
    Q6_dccleaninva_A((void *) slot);
}

static inline uint32_t fence_read(volatile uint32_t * slot) {
    Q6_dccleaninva_A((void *) slot);
    asm volatile("syncht" : : : "memory");
    return atomic_load((atomic_uint *) slot);
}

// ---- the handle

AEEResult ddrbw_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    void * h = calloc(1, 8);
    if (h == NULL) {
        return AEE_ENOMEMORY;
    }
    *handle = (remote_handle64) h;
    return AEE_SUCCESS;
}

static void queue_shutdown(void);

AEEResult ddrbw_close(remote_handle64 handle) {
    queue_shutdown();
    for (uint32_t i = 0; i < DDRBW_MAX_MAPS; i++) {
        if (g.maps[i].va != NULL) {
            HAP_munmap2(g.maps[i].va, g.maps[i].size);
            g.maps[i].va = NULL;
        }
    }
    if (g.rctx != 0) {
        HAP_compute_res_release(g.rctx);
        g.rctx = 0;
        g.vtcm = NULL;
    }
    g.setup_done = 0;
    free((void *) handle);
    return AEE_SUCCESS;
}

// ---- setup and facts

static void get_facts(uint64 * facts, int n) {
    if (n < DDRBW_FACT_COUNT) {
        return;
    }
    memset(facts, 0, sizeof(uint64) * (size_t) n);
    facts[DDRBW_FACT_ARCH] = __HVX_ARCH__;
    facts[DDRBW_FACT_HVX]  = g.n_hvx;
    facts[DDRBW_FACT_VTCM] = g.vtcm_size;

    HAP_power_response_t r;
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_get_max_mips;
    if (HAP_power_get(NULL, &r) == 0) {
        facts[DDRBW_FACT_MAX_MIPS] = r.max_mips;
    }
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_get_max_bus_bw;
    if (HAP_power_get(NULL, &r) == 0) {
        facts[DDRBW_FACT_MAX_BUS_BW] = r.max_bus_bw;
    }
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_get_clk_Freq;
    if (HAP_power_get(NULL, &r) == 0) {
        facts[DDRBW_FACT_CORE_HZ] = r.clkFreqHz;
    }
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_get_hmx_core_clk_Freq;
    if (HAP_power_get(NULL, &r) == 0) {
        facts[DDRBW_FACT_HMX_HZ] = r.clkFreqHz;
    }
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_get_dcvsEnabled;
    if (HAP_power_get(NULL, &r) == 0) {
        facts[DDRBW_FACT_DCVS] = r.dcvsEnabled;
    }
    facts[DDRBW_FACT_PRIO] = (uint64) qurt_thread_get_priority(qurt_thread_get_id());
}

// The DCVS v3 request of htp/main.c, plus the optional fields of the mask. Each helper writes its
// return code. The request goes to the power manager only after all helpers.
static void vote_dcvs(uint32_t votes, int32 * rc) {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type                              = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable           = TRUE;
    req.dcvs_v3.dcvs_enable               = FALSE;
    req.dcvs_v3.set_bus_params            = TRUE;
    req.dcvs_v3.bus_params.min_corner     = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner     = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner  = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params           = TRUE;
    req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable         = TRUE;
    req.dcvs_v3.sleep_disable             = TRUE;
#if __HVX_ARCH__ >= 79
    rc[DDRBW_RC_PROTECTED] = HAP_set_dcvs_v3_protected_bus_corners(&req, 1);
#endif
    if (votes & DDRBW_VOTE_DDRPERF) {
        rc[DDRBW_RC_DDRPERF] = HAP_set_ddr_perf_mode(&req, 1);
    }
    if (votes & DDRBW_VOTE_BUSPERF) {
        rc[DDRBW_RC_COREPERF] = HAP_set_dcvs_v3_core_perf_mode(&req, HAP_DCVS_CLK_PERF_HIGH);
        rc[DDRBW_RC_BUSPERF]  = HAP_set_dcvs_v3_bus_perf_mode(&req, HAP_DCVS_CLK_PERF_HIGH);
    }
    if (votes & DDRBW_VOTE_EXPV) {
        HAP_dcvs_request_t d;
        memset(&d, 0, sizeof(d));
        d.type                                    = HAP_DCVS_SET_EXP_VCORNERS;
        d.exp_vcorners.set_core_vcorner           = 1;
        d.exp_vcorners.core_params.target_corner  = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.core_params.min_corner     = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.core_params.max_corner     = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.set_bus_vcorner            = 1;
        d.exp_vcorners.bus_params.target_corner   = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.bus_params.min_corner      = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.bus_params.max_corner      = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.set_ceng_vcorner           = 1;
        d.exp_vcorners.ceng_params.target_corner  = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.ceng_params.min_corner     = HAP_DCVS_EXP_VCORNER_MAX;
        d.exp_vcorners.ceng_params.max_corner     = HAP_DCVS_EXP_VCORNER_MAX;
        rc[DDRBW_RC_EXPV] = HAP_dcvs_config((void *) &req, &d);
    }
    rc[DDRBW_RC_DCVS] = HAP_power_set((void *) &g_client_backend, &req);
}

// The HVX and HMX requests of htp/main.c.
static void vote_units(int32 * rc) {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type         = HAP_power_set_HVX;
    req.hvx.power_up = TRUE;
    rc[DDRBW_RC_HVX] = HAP_power_set((void *) &g_client_backend, &req);

    memset(&req, 0, sizeof(req));
    req.type                 = HAP_power_set_HMX_v2;
    req.hmx_v2.set_power     = TRUE;
    req.hmx_v2.power_up      = TRUE;
    req.hmx_v2.set_clock     = TRUE;
    req.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.min_corner    = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.max_corner    = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.perf_mode     = HAP_CLK_PERF_HIGH;
    rc[DDRBW_RC_HMX]         = HAP_power_set((void *) &g_client_backend, &req);
}

static void vote_ceng(int32 * rc) {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type                   = HAP_power_set_CENG_bus;
    req.ceng_bus.target_corner = HAP_DCVS_VCORNER_MAX;
    req.ceng_bus.min_corner    = HAP_DCVS_VCORNER_MAX;
    req.ceng_bus.max_corner    = HAP_DCVS_VCORNER_MAX;
    req.ceng_bus.perf_mode     = HAP_CLK_PERF_HIGH;
    rc[DDRBW_RC_CENG]          = HAP_power_set((void *) &g_client_ceng, &req);
}

// A MIPS and bus bandwidth vote at the maximum values that the DSP gives. When the DSP gives no
// maximum bandwidth, the vote asks for 85 GB/s, the peak of LPDDR5X-10667 on 64 bits.
static void vote_bw(const uint64 * facts, int32 * rc) {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type                         = HAP_power_set_mips_bw;
    req.mips_bw.set_mips             = TRUE;
    req.mips_bw.mipsTotal            = facts[DDRBW_FACT_MAX_MIPS] ? (unsigned int) facts[DDRBW_FACT_MAX_MIPS] : 0xFFFFu;
    req.mips_bw.mipsPerThread        = g.n_hvx ? req.mips_bw.mipsTotal / g.n_hvx : req.mips_bw.mipsTotal;
    req.mips_bw.set_bus_bw           = TRUE;
    req.mips_bw.bwBytePerSec         = facts[DDRBW_FACT_MAX_BUS_BW] ? facts[DDRBW_FACT_MAX_BUS_BW] : 85000000000ull;
    req.mips_bw.busbwUsagePercentage = 100;
    req.mips_bw.set_latency          = TRUE;
    req.mips_bw.latency              = 1;
    rc[DDRBW_RC_MIPSBW]              = HAP_power_set((void *) &g_client_bw, &req);
}

// Hold the full VTCM. Returns the return code (0 or an error).
static int hold_vtcm(void) {
    unsigned int size = 0;
    int          err  = HAP_compute_res_query_VTCM(0, &size, NULL, NULL, NULL);
    if (err != 0 || size == 0) {
        return err != 0 ? err : AEE_ENOMEMORY;
    }
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, size, size, size);
    const unsigned int rctx = HAP_compute_res_acquire(&attr, DDRBW_ACQUIRE_US);
    if (rctx == 0) {
        return AEE_ERESOURCENOTFOUND;
    }
    void * ptr = NULL;
    err        = HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &ptr, &size);
    if (err != 0 || ptr == NULL) {
        HAP_compute_res_release(rctx);
        return err != 0 ? err : AEE_ENOMEMORY;
    }
    g.rctx      = rctx;
    g.vtcm      = (uint8_t *) ptr;
    g.vtcm_size = size;
    return 0;
}

AEEResult ddrbw_setup(remote_handle64 handle, uint32 votes, int32 * rc, int rcLen, uint64 * facts, int factsLen) {
    (void) handle;
    if (rcLen < DDRBW_RC_COUNT || factsLen < DDRBW_FACT_COUNT) {
        return AEE_EBADPARM;
    }
    if (g.setup_done) {
        return AEE_EITEMBUSY;
    }
    for (int i = 0; i < rcLen; i++) {
        rc[i] = DDRBW_RC_NOT_SENT;
    }
    g.n_hvx = (uint32_t) ((qurt_hvx_get_units() >> 8) & 0xFF);
    get_facts(facts, factsLen);

    if (votes & DDRBW_VOTE_BACKEND) {
        HAP_power_request_t req;
        memset(&req, 0, sizeof(req));
        req.type            = HAP_power_set_apptype;
        req.apptype         = HAP_POWER_COMPUTE_CLIENT_CLASS;
        rc[DDRBW_RC_APPTYPE] = HAP_power_set((void *) &g_client_backend, &req);
        vote_dcvs(votes, rc);
        vote_units(rc);
    }
    if (votes & DDRBW_VOTE_CENG) {
        vote_ceng(rc);
    }
    if (votes & DDRBW_VOTE_BW) {
        vote_bw(facts, rc);
    }

    HAP_dcvs_request_t caps;
    memset(&caps, 0, sizeof(caps));
    caps.type          = HAP_DCVS_CAPABILITY_QUERY;
    rc[DDRBW_RC_CAPS]  = HAP_dcvs_config(NULL, &caps);
    rc[DDRBW_RC_VTCM]  = hold_vtcm();
    g.setup_done       = 1;

    get_facts(facts, factsLen);
    facts[DDRBW_FACT_CAPS] = rc[DDRBW_RC_CAPS] == 0 ? caps.capability.capability_info : 0;
    FARF(ALWAYS, "ddrbw: setup votes 0x%x vtcm %u hvx %u", (unsigned) votes, (unsigned) g.vtcm_size, (unsigned) g.n_hvx);
    return AEE_SUCCESS;
}

AEEResult ddrbw_query(remote_handle64 handle, uint64 * facts, int factsLen) {
    (void) handle;
    if (factsLen < DDRBW_FACT_COUNT) {
        return AEE_EBADPARM;
    }
    get_facts(facts, factsLen);
    return AEE_SUCCESS;
}

// ---- mappings

AEEResult ddrbw_map(remote_handle64 handle, int32 fd, uint32 size, uint64 * ticks) {
    (void) handle;
    *ticks = 0;
    if (find_map(fd) != NULL) {
        return AEE_SUCCESS;
    }
    for (uint32_t i = 0; i < DDRBW_MAX_MAPS; i++) {
        if (g.maps[i].va != NULL) {
            continue;
        }
        void *         va = (void *) -1;
        const uint64_t t0 = HAP_perf_get_qtimer_count();
        for (int retry = 0; retry < 2 && (va == (void *) -1 || va == NULL); retry++) {
            va = HAP_mmap2(NULL, size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
        }
        *ticks = HAP_perf_get_qtimer_count() - t0;
        if (va == (void *) -1 || va == NULL) {
            FARF(ERROR, "ddrbw: HAP_mmap2 of fd %d size %u failed", (int) fd, (unsigned) size);
            return AEE_EFAILED;
        }
        g.maps[i].fd   = fd;
        g.maps[i].va   = (uint8_t *) va;
        g.maps[i].size = size;
        return AEE_SUCCESS;
    }
    return AEE_ENOMEMORY;
}

AEEResult ddrbw_unmap(remote_handle64 handle, int32 fd, uint64 * ticks) {
    (void) handle;
    *ticks                 = 0;
    struct ddrbw_map * m = find_map(fd);
    if (m == NULL) {
        return AEE_EBADPARM;
    }
    const uint64_t t0 = HAP_perf_get_qtimer_count();
    HAP_munmap2(m->va, m->size);
    *ticks = HAP_perf_get_qtimer_count() - t0;
    m->va  = NULL;
    return AEE_SUCCESS;
}

// ---- the read run

static void wait_go(struct job * job) {
    atomic_fetch_add_explicit(&job->ready, 1, memory_order_release);
    while (!atomic_load_explicit(&job->go, memory_order_acquire)) {
        hex_pause();
    }
}

// HVX loads: add the words of each chunk, and keep an l2fetch box of the next "pf" chunks in
// flight. The box starts at the next chunk and stops at the end of the part. O(bytes read).
static void run_hvx(struct worker * w) {
    const struct job * j       = w->job;
    const uint32_t     chunk   = j->chunk;
    const uint32_t     nchunks = w->nchunks;
    const uint32_t     pf      = j->pf;
    const uint32_t     nvec    = chunk / 128u;
    HVX_Vector         a0      = Q6_V_vzero();
    HVX_Vector         a1      = a0;
    HVX_Vector         a2      = a0;
    HVX_Vector         a3      = a0;
    uint32_t           c       = 0;
    uint64_t           n       = 0;

    wait_go(w->job);
    if (pf) {
        const uint32_t h = (pf < nchunks ? pf : nchunks) * j->box_h;
        l2fetch_box(w->part, j->box_w, h);
    }
    const uint64_t t0 = HAP_perf_get_qtimer_count();
    const uint64_t c0 = HAP_perf_get_pcycles();
    for (;;) {
        if (pf) {
            const uint32_t next = c + 1u;
            if (next < nchunks) {
                const uint32_t left = nchunks - next;
                l2fetch_box(w->part + (size_t) next * chunk, j->box_w, (pf < left ? pf : left) * j->box_h);
            } else {
                l2fetch_box(w->part, j->box_w, (pf < nchunks ? pf : nchunks) * j->box_h);
            }
        }
        const HVX_Vector * v = (const HVX_Vector *) (w->part + (size_t) c * chunk);
        for (uint32_t i = 0; i < nvec; i += 4) {
            a0 = Q6_Vw_vadd_VwVw(a0, v[i + 0]);
            a1 = Q6_Vw_vadd_VwVw(a1, v[i + 1]);
            a2 = Q6_Vw_vadd_VwVw(a2, v[i + 2]);
            a3 = Q6_Vw_vadd_VwVw(a3, v[i + 3]);
        }
        n++;
        if (++c == nchunks) {
            c = 0;
        }
        if (HAP_perf_get_qtimer_count() >= j->deadline) {
            break;
        }
    }
    w->t_start = t0;
    w->t_end   = HAP_perf_get_qtimer_count();
    w->pcycles = HAP_perf_get_pcycles() - c0;
    w->chunks  = n;
    w->sum     = reduce4(a0, a1, a2, a3);
}

// Add all bytes of the rows of one VTCM slot. A row that is not a multiple of 128 bytes adds its
// last vector through a mask. O(slot bytes).
static inline void sum_slot(const uint8_t * slot, uint32_t rows, uint32_t row, uint32_t pitch, HVX_Vector * a0,
                            HVX_Vector * a1) {
    const uint32_t     full = row / 128u;
    const uint32_t     tail = row % 128u;
    const HVX_VectorPred m  = Q6_Q_vsetq_R(tail);
    for (uint32_t r = 0; r < rows; r++) {
        const HVX_Vector * v = (const HVX_Vector *) (slot + (size_t) r * pitch);
        uint32_t           i = 0;
        for (; i + 1 < full; i += 2) {
            *a0 = Q6_Vw_vadd_VwVw(*a0, v[i]);
            *a1 = Q6_Vw_vadd_VwVw(*a1, v[i + 1]);
        }
        if (i < full) {
            *a0 = Q6_Vw_vadd_VwVw(*a0, v[i]);
        }
        if (tail) {
            *a1 = Q6_Vw_vadd_VwVw(*a1, Q6_V_vmux_QVV(m, v[full], Q6_V_vzero()));
        }
    }
}

// The DMA engine: keep "depth" descriptors in flight, add each chunk that arrives, then use its
// slot for the next chunk. After the deadline the thread pushes no descriptor and waits for the
// descriptors in flight. O(bytes read).
static void run_dma(struct worker * w) {
    const struct job * j       = w->job;
    dma_queue *        q       = w->q;
    const uint32_t     chunk   = j->chunk;
    const uint32_t     nchunks = w->nchunks;
    const int          full    = (j->flags & DDRBW_FLAG_SUM) != 0;
    HVX_Vector         a0      = Q6_V_vzero();
    HVX_Vector         a1      = a0;
    uint32_t           c       = 0;
    uint64_t           pushed  = 0;
    uint64_t           popped  = 0;
    int                stop    = 0;

    wait_go(w->job);
    const uint64_t t0 = HAP_perf_get_qtimer_count();
    const uint64_t c0 = HAP_perf_get_pcycles();
    for (uint32_t d = 0; d < j->depth; d++) {
        dma_queue_push(q, dma_make_ptr(w->vtcm + (size_t) d * j->slot, w->part + (size_t) c * chunk), j->dst_row, j->row,
                       j->row, j->rows);
        pushed++;
        if (++c == nchunks) {
            c = 0;
        }
    }
    while (popped < pushed) {
        uint8_t * dst = (uint8_t *) dma_queue_pop(q).dst;
        popped++;
        if (full) {
            sum_slot(dst, j->rows, j->row, j->dst_row, &a0, &a1);
        } else {
            a0 = Q6_Vw_vadd_VwVw(a0, *(const HVX_Vector *) dst);
        }
        if (!stop && HAP_perf_get_qtimer_count() >= j->deadline) {
            stop = 1;
        }
        if (!stop) {
            dma_queue_push(q, dma_make_ptr(dst, w->part + (size_t) c * chunk), j->dst_row, j->row, j->row, j->rows);
            pushed++;
            if (++c == nchunks) {
                c = 0;
            }
        }
    }
    w->t_start = t0;
    w->t_end   = HAP_perf_get_qtimer_count();
    w->pcycles = HAP_perf_get_pcycles() - c0;
    w->chunks  = popped;
    w->sum     = reduce4(a0, a1, Q6_V_vzero(), Q6_V_vzero());
}

static void worker_main(void * arg) {
    struct worker * w = (struct worker *) arg;
    if (w->kind == DDRBW_KIND_DMA) {
        run_dma(w);
    } else {
        run_hvx(w);
    }
    qurt_thread_exit(0);
}

// Check the parameters of a run. Returns DDRBW_OK or DDRBW_E_PARAM.
static int check_run(const ddrbw_run * run, const struct ddrbw_map * m, int threadsLen) {
    const uint32_t row  = run->row ? run->row : run->chunk;
    const uint32_t dst  = run->dst_row ? run->dst_row : row;
    const int      hvx  = run->n_dma < run->n_threads;
    const int      dma  = run->n_dma > 0;
    if (run->n_threads == 0 || run->n_threads > DDRBW_MAX_THREADS || run->n_threads > g.n_hvx ||
        (int) run->n_threads > threadsLen || run->n_dma > run->n_threads) {
        return DDRBW_E_PARAM;
    }
    if (run->chunk < 512 || run->chunk % 128 != 0 || run->region > m->size || run->budget_us == 0) {
        return DDRBW_E_PARAM;
    }
    if (run->region / run->n_threads < run->chunk) {
        return DDRBW_E_PARAM;
    }
    if (hvx && (run->chunk % 512 != 0 || run->pf > 16)) {
        return DDRBW_E_PARAM;
    }
    if (dma && (run->depth == 0 || run->chunk % row != 0 || row < 128 || row % 4 != 0 || row > 0xFFFFFFu ||
                dst < row || dst % 128 != 0 || run->chunk / row > 0xFFFFu)) {
        return DDRBW_E_PARAM;
    }
    return DDRBW_OK;
}

static void free_workers(struct worker * w, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        free(w[i].stack);
        free(w[i].qmem);
    }
}

AEEResult ddrbw_read(remote_handle64 handle, const ddrbw_run * run, ddrbw_thread * threads, int threadsLen,
                     uint64 * t_go, int32 * status) {
    (void) handle;
    *t_go   = 0;
    *status = DDRBW_OK;
    memset(threads, 0, sizeof(ddrbw_thread) * (size_t) threadsLen);

    const struct ddrbw_map * m = find_map(run->fd);
    if (m == NULL) {
        *status = DDRBW_E_NOMAP;
        return AEE_SUCCESS;
    }
    if ((*status = check_run(run, m, threadsLen)) != DDRBW_OK) {
        return AEE_SUCCESS;
    }

    static struct job job;
    memset(&job, 0, sizeof(job));
    job.chunk   = run->chunk;
    job.row     = run->row ? run->row : run->chunk;
    job.dst_row = run->dst_row ? run->dst_row : job.row;
    job.rows    = run->chunk / job.row;
    job.slot    = job.rows * job.dst_row;
    job.depth   = run->depth;
    job.pf      = run->pf;
    job.flags   = run->flags;
    job.box_w   = run->chunk % 4096u == 0 ? 4096u : 512u;
    job.box_h   = run->chunk / job.box_w;
    atomic_init(&job.ready, 0);
    atomic_init(&job.go, 0);

    const uint32_t n         = run->n_threads;
    const uint32_t part_len  = run->region / n / run->chunk * run->chunk;
    const uint32_t vtcm_part = g.vtcm_size / n / 2048u * 2048u;
    if (run->n_dma > 0 && (g.vtcm == NULL || (uint64_t) job.depth * job.slot > vtcm_part)) {
        *status = DDRBW_E_VTCM;
        return AEE_SUCCESS;
    }

    // Static, because the stack of the FastRPC thread is small. One run at a time uses it.
    static struct worker w[DDRBW_MAX_THREADS];
    memset(w, 0, sizeof(w));
    const int prio = qurt_thread_get_priority(qurt_thread_get_id());
    for (uint32_t i = 0; i < n; i++) {
        w[i].job        = &job;
        w[i].kind       = i < run->n_dma ? DDRBW_KIND_DMA : DDRBW_KIND_HVX;
        w[i].part_start = i * part_len;
        w[i].part_len   = part_len;
        w[i].part       = m->va + w[i].part_start;
        w[i].nchunks    = part_len / run->chunk;
        w[i].stack      = memalign(128, DDRBW_STACK_BYTES);
        if (w[i].stack == NULL) {
            free_workers(w, n);
            *status = DDRBW_E_NOMEM;
            return AEE_SUCCESS;
        }
        if (w[i].kind == DDRBW_KIND_DMA) {
            const uint32_t cap = job.depth + 1u;
            w[i].vtcm          = g.vtcm + (size_t) i * vtcm_part;
            w[i].qmem          = memalign(dma_queue_alignof(), dma_queue_sizeof(cap));
            if (w[i].qmem == NULL) {
                free_workers(w, n);
                *status = DDRBW_E_NOMEM;
                return AEE_SUCCESS;
            }
            w[i].q          = dma_queue_init(w[i].qmem, cap, (uintptr_t) g.vtcm, g.vtcm_size, &w[i].trace);
            w[i].q->nocache = (run->flags & DDRBW_FLAG_CACHED) ? 0 : 1;
        }
    }

    uint32_t started = 0;
    for (uint32_t i = 0; i < n; i++) {
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, w[i].stack);
        qurt_thread_attr_set_stack_size(&attr, DDRBW_STACK_BYTES);
        qurt_thread_attr_set_priority(&attr, (unsigned short) prio);
        qurt_thread_attr_set_name(&attr, "ddrbw");
        if (qurt_thread_create(&w[i].tid, &attr, worker_main, &w[i]) != QURT_EOK) {
            break;
        }
        w[i].started = 1;
        started++;
    }

    // Start the threads at one time. This thread sleeps while it waits, thus it gives its hardware
    // thread to the workers. When a thread did not start, the others read for 1 us and the status
    // tells the host.
    const uint64_t limit = HAP_perf_get_qtimer_count() + us_to_ticks(DDRBW_BARRIER_US);
    while (atomic_load_explicit(&job.ready, memory_order_acquire) < started && HAP_perf_get_qtimer_count() < limit) {
        qurt_sleep(20);
    }
    const uint64_t go = HAP_perf_get_qtimer_count();
    job.deadline      = go + us_to_ticks(started == n ? run->budget_us : 1u);
    atomic_store_explicit(&job.go, 1, memory_order_release);

    for (uint32_t i = 0; i < n; i++) {
        if (w[i].started) {
            int st;
            (void) qurt_thread_join(w[i].tid, &st);
        }
    }
    free_workers(w, n);

    *t_go = go;
    if (started != n) {
        *status = DDRBW_E_THREAD;
    }
    for (uint32_t i = 0; i < n; i++) {
        threads[i].t_start    = w[i].t_start;
        threads[i].t_end      = w[i].t_end;
        threads[i].chunks     = w[i].chunks;
        threads[i].pcycles    = w[i].pcycles;
        threads[i].part_start = w[i].part_start;
        threads[i].part_len   = w[i].part_len;
        threads[i].sum        = w[i].sum;
        threads[i].kind       = w[i].kind;
    }
    return AEE_SUCCESS;
}

// ---- the latency tests

AEEResult ddrbw_nop(remote_handle64 handle) {
    (void) handle;
    return AEE_SUCCESS;
}

#define Q_SERVED 0
#define Q_EMPTY  1
#define Q_STOP   2
#define Q_ERROR  3

// Read one packet if there is one, and answer it with the same buffers, as htp/main.c answers a
// batch. Returns Q_SERVED, Q_EMPTY, Q_STOP or Q_ERROR.
static int serve_one(void) {
    struct ddrbw_qmsg      msg;
    struct dspqueue_buffer buf;
    uint32_t               flags = 0;
    uint32_t               nbufs = 0;
    uint32_t               len   = 0;
    memset(&msg, 0, sizeof(msg));
    memset(&buf, 0, sizeof(buf));
    int err = dspqueue_read_noblock(g.queue, &flags, 1, &nbufs, &buf, sizeof(msg), &len, (uint8_t *) &msg);
    if (err == AEE_EWOULDBLOCK) {
        return Q_EMPTY;
    }
    if (err != 0) {
        atomic_store(&g.qerr, err);
        return Q_ERROR;
    }
    msg.t_dsp = HAP_perf_get_qtimer_count();
    if (nbufs) {
        buf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    }
    err = dspqueue_write(g.queue, 0, nbufs, nbufs ? &buf : NULL, sizeof(msg), (const uint8_t *) &msg,
                         DSPQUEUE_TIMEOUT_NONE);
    if (err != 0) {
        atomic_store(&g.qerr, err);
        return Q_ERROR;
    }
    atomic_fetch_add(&g.qserved, 1);
    return msg.cmd == DDRBW_QCMD_STOP ? Q_STOP : Q_SERVED;
}

// Wait in dspqueue_peek until a packet is there. Returns 0, or 1 when the thread must stop.
static int peek_wait(void) {
    for (;;) {
        uint32_t flags = 0;
        uint32_t nbufs = 0;
        uint32_t len   = 0;
        const int err  = dspqueue_peek(g.queue, &flags, &nbufs, &len, 50000);
        if (err == 0) {
            return 0;
        }
        if (err != AEE_EWOULDBLOCK && err != AEE_EEXPIRED) {
            atomic_store(&g.qerr, err);
            return 1;
        }
    }
}

static void queue_main(void * arg) {
    (void) arg;
    const uint32_t mode = g.qmode;
    for (;;) {
        int r;
        if (mode == DDRBW_QMODE_SPIN) {
            r = serve_one();
            if (r == Q_EMPTY) {
                hex_pause();
                continue;
            }
        } else {
            if (peek_wait()) {
                break;
            }
            r = serve_one();
            // The backend: after one batch, poll 100 times with a sleep of 100 us between polls.
            for (uint32_t poll = 100; mode == DDRBW_QMODE_BACKEND && (r == Q_SERVED || r == Q_EMPTY);) {
                if (r == Q_SERVED) {
                    poll = 100;
                } else if (--poll == 0) {
                    break;
                } else {
                    qurt_sleep(100);
                }
                r = serve_one();
            }
        }
        if (r == Q_STOP || r == Q_ERROR) {
            break;
        }
    }
    qurt_thread_exit(0);
}

AEEResult ddrbw_queue_start(remote_handle64 handle, uint64 queue_id, uint32 mode) {
    (void) handle;
    if (g.queue != NULL || (mode & DDRBW_QMODE_MASK) > DDRBW_QMODE_SPIN) {
        return AEE_EBADPARM;
    }
    int err = dspqueue_import(queue_id, NULL, NULL, NULL, &g.queue);
    if (err != 0) {
        g.queue = NULL;
        return err;
    }
    g.qmode = mode & DDRBW_QMODE_MASK;
    atomic_store(&g.qserved, 0);
    atomic_store(&g.qerr, 0);
    g.qstack = memalign(128, DDRBW_STACK_BYTES);
    if (g.qstack == NULL) {
        dspqueue_close(g.queue);
        g.queue = NULL;
        return AEE_ENOMEMORY;
    }
    // The priority of the main thread of htp/main.c: 10 above the FastRPC thread.
    int prio = qurt_thread_get_priority(qurt_thread_get_id()) - 10;
    if (prio < 1) {
        prio = 1;
    }
    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, g.qstack);
    qurt_thread_attr_set_stack_size(&attr, DDRBW_STACK_BYTES);
    qurt_thread_attr_set_priority(&attr, (unsigned short) prio);
    qurt_thread_attr_set_name(&attr, "ddrbw-q");
    if (qurt_thread_create(&g.qthread, &attr, queue_main, NULL) != QURT_EOK) {
        free(g.qstack);
        g.qstack = NULL;
        dspqueue_close(g.queue);
        g.queue = NULL;
        return AEE_EFAILED;
    }
    return AEE_SUCCESS;
}

static void queue_shutdown(void) {
    if (g.queue == NULL) {
        return;
    }
    int st;
    (void) qurt_thread_join(g.qthread, &st);
    dspqueue_close(g.queue);
    g.queue = NULL;
    free(g.qstack);
    g.qstack = NULL;
}

AEEResult ddrbw_queue_stop(remote_handle64 handle, uint32 * served) {
    (void) handle;
    if (g.queue == NULL) {
        return AEE_EBADPARM;
    }
    queue_shutdown();
    *served = atomic_load(&g.qserved);
    return atomic_load(&g.qerr);
}

AEEResult ddrbw_fence(remote_handle64 handle, int32 fd, uint32 iters, uint32 timeout_us, uint32 * done, uint64 * t_first,
                      uint64 * t_last) {
    (void) handle;
    *done    = 0;
    *t_first = 0;
    *t_last  = 0;
    const struct ddrbw_map * m = find_map(fd);
    if (m == NULL || m->size < DDRBW_FENCE_BYTES) {
        return AEE_EBADPARM;
    }
    volatile uint32_t * host = (volatile uint32_t *) m->va;
    volatile uint32_t * dsp  = (volatile uint32_t *) (m->va + DDRBW_FENCE_DSP);
    const uint64_t      to   = us_to_ticks(timeout_us);
    for (uint32_t i = 1; i <= iters; i++) {
        const uint64_t t0 = HAP_perf_get_qtimer_count();
        while (fence_read(host) != i) {
            if (HAP_perf_get_qtimer_count() - t0 > to) {
                return AEE_SUCCESS;
            }
        }
        fence_write(dsp, i);
        const uint64_t t = HAP_perf_get_qtimer_count();
        if (i == 1) {
            *t_first = t;
        }
        *t_last = t;
        *done   = i;
    }
    return AEE_SUCCESS;
}
