// ddrbw: the host program of the DDR read probe. It runs on the ARM CPU of an Android phone and
// drives the DSP library libddrbw_skel.so in the unsigned PD of the compute DSP through FastRPC.
//
// Usage (from adb shell, with ADSP_LIBRARY_PATH set to the directory of libddrbw_skel.so):
//   ddrbw info --votes NAME
//   ddrbw nsp  --votes NAME [--plan full|short|ws] [--mb N] [--reps N] [--budget-ms N]
//   ddrbw cpu  [--sets LIST] [--mb N] [--reps N] [--budget-ms N]
//   ddrbw both --votes NAME [--cfgs LIST] [--sets LIST] [--mb N] [--reps N] [--budget-ms N]
//   ddrbw rtt  --votes NAME [--iters N]
//   ddrbw mapcost --votes NAME [--mb N] [--iters N]
//   ddrbw freqlist
//   ddrbw freq --out FILE [--hz N] [--ms N]
//   ddrbw now
//
// The power votes (--votes) of the DSP library: none, backend (the votes of htp/main.c of the
// llama.cpp backend), ddrperf, busperf, expv, ceng, bw (each one is backend plus one more vote),
// and max (all votes). Refer to ddrbw_common.h.
//
// "nsp" reads a buffer of --mb MiB (512 by default) of rpcmem, the memory of the weights of the
// backend, from the DSP. Each configuration of the plan (k_cfgs) sets the thread count, the HVX
// loads or the DMA engine, the chunk, the DMA descriptor form and depth, and the l2fetch distance.
// Each run reads for --budget-ms, and the plan runs --reps times, in the reverse order in each
// second repetition.
//
// "cpu" reads a buffer of anonymous memory with NEON loads from the threads of each CPU set, each
// thread pinned to one core. A set is a list of cores with ranges, and "/" divides the sets, for
// example "7/6,7/0-5/0-7".
//
// "both" measures, for each NSP configuration of --cfgs and each CPU set of --sets: the NSP alone,
// the CPU alone, and the two at the same time. The rates of the two readers at the same time are
// the bytes of each reader in the time window of the NSP run: the DSP gives its start and end on
// the global qtimer, and the CPU threads record their progress on CNTVCT_EL0, the same counter.
//
// "rtt" measures the time of one CPU-DSP synchronization: an empty FastRPC call, a dspqueue round
// trip for each of the three waits of the DSP thread (peek, the poll loop of htp/main.c, spin),
// each host wait (dspqueue_read, a poll of dspqueue_read_noblock) and with or without a buffer
// reference, and a round trip of two fence words in shared memory (the fence of htp-fence.h), with
// and without a cache clean on the host.
//
// "mapcost" measures the mapping of one model chunk of the backend (1 GiB of rpcmem): the host
// fastrpc_mmap, and the HAP_mmap2 and HAP_munmap2 on the DSP that prep_op_bufs of htp/main.c does
// for a batch that does not reuse a mapping.
//
// "freqlist" prints the frequency nodes of the shell (devfreq, bus_dcvs, the GPU and the CPU
// clusters) and their values. "freq" samples the cur_freq nodes with pread (no process start) and
// writes a line on CNTVCT_EL0 and on CLOCK_MONOTONIC when a value changes, thus a run of the DSP
// (its qtimer counts) and the log of memprobe --log-ts go on the same time axis. "now" prints the
// two clocks at one time.
//
// Each result is one line on stdout that starts with "ddrbw: " and holds "key=value" fields.
// build/bw/stage.py reads these lines. Each checksum that does not agree gives "check=FAIL".
//
// Exit codes: 0 success, 1 a usage error, 2 a FastRPC, memory or setup error, 4 a checksum or a run
// status was not correct.

#include <arm_neon.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "AEEStdErr.h"
#include "ddrbw.h"
#include "ddrbw_common.h"
#include "dspqueue.h"
#include "remote.h"
#include "rpc.h"
#include "rpcmem.h"

#define EXIT_USAGE 1
#define EXIT_RPC   2
#define EXIT_CHECK 4

#define MAX_THREADS  8
#define MAX_SETS     16
#define CPU_CHUNK    65536u
#define CPU_LOG_STEP 16u         // one progress sample for each 16 chunks (1 MiB)
#define CPU_LOG_MAX  (1u << 16)  // progress samples for each thread
#define BLOCK        128u        // the granularity of the prefix sums

// ---- time

static uint64_t g_cntfrq;

// The global counter of the SoC. On Snapdragon it is the qtimer of the DSP (19.2 MHz).
static inline uint64_t cntvct(void) {
    uint64_t v;
    __asm__ volatile("isb\n mrs %0, cntvct_el0" : "=r"(v) : : "memory");
    return v;
}

static inline uint64_t read_cntfrq(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}

static inline double ticks_to_s(uint64_t t) {
    return (double) t / (double) g_cntfrq;
}

static void sleep_ms(unsigned ms) {
    struct timespec ts = { (time_t) (ms / 1000u), (long) (ms % 1000u) * 1000000L };
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

// ---- the buffer and its prefix sums

// A buffer that a reader reads, and the prefix sums of its 128-byte blocks. The 32-bit sums wrap,
// thus the sum of the words of [a, b) is pre[b / 128] - pre[a / 128] for a and b on a block edge.
struct buffer {
    uint8_t *  data;
    size_t     size;
    uint32_t * pre;  // size / 128 + 1 values
    int        fd;   // the rpcmem fd, or -1 for anonymous memory
    int        mapped;
};

static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct fill_job {
    struct buffer * b;
    size_t          first;  // the first block
    size_t          last;   // one after the last block
    uint32_t        seed;
};

static void * fill_main(void * arg) {
    struct fill_job * j = (struct fill_job *) arg;
    uint32_t *        w = (uint32_t *) j->b->data;
    for (size_t blk = j->first; blk < j->last; blk++) {
        uint32_t s = 0;
        for (size_t i = blk * 32u; i < blk * 32u + 32u; i++) {
            const uint32_t v = mix32((uint32_t) i * 2u + j->seed);
            w[i]             = v;
            s += v;
        }
        j->b->pre[blk + 1] = s;
    }
    return NULL;
}

// Fill the buffer with a hash of the word index and make the prefix sums. O(size), 8 threads.
static int fill_buffer(struct buffer * b, uint32_t seed) {
    const size_t nblk = b->size / BLOCK;
    b->pre            = (uint32_t *) malloc((nblk + 1) * sizeof(uint32_t));
    if (b->pre == NULL) {
        fprintf(stderr, "ddrbw: error: no memory for the prefix sums of %zu MiB\n", b->size >> 20);
        return -1;
    }
    pthread_t       tid[8];
    struct fill_job jobs[8];
    for (int t = 0; t < 8; t++) {
        jobs[t] = (struct fill_job) { b, nblk * (size_t) t / 8u, nblk * (size_t) (t + 1) / 8u, seed };
        if (pthread_create(&tid[t], NULL, fill_main, &jobs[t]) != 0) {
            fill_main(&jobs[t]);
            tid[t] = 0;
        }
    }
    for (int t = 0; t < 8; t++) {
        if (tid[t]) {
            pthread_join(tid[t], NULL);
        }
    }
    b->pre[0] = 0;
    for (size_t i = 0; i < nblk; i++) {
        b->pre[i + 1] += b->pre[i];
    }
    return 0;
}

static inline uint32_t sum_range(const struct buffer * b, size_t a, size_t e) {
    return b->pre[e / BLOCK] - b->pre[a / BLOCK];
}

// 1 when the kernel lets EL0 clean a cache line by address (SCTLR_EL1.UCI), 0 when the instruction
// traps. The first call runs one "dc cvac" under a SIGILL handler.
static sigjmp_buf g_ill_jmp;

static void on_ill(int sig) {
    (void) sig;
    siglongjmp(g_ill_jmp, 1);
}

static int cmo_permitted(void) {
    static int state = -1;
    if (state < 0) {
        struct sigaction sa;
        struct sigaction old;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = on_ill;
        sigaction(SIGILL, &sa, &old);
        volatile uint64_t probe = 0;
        if (sigsetjmp(g_ill_jmp, 1) == 0) {
            __asm__ volatile("dc cvac, %0\n dsb ish" : : "r"(&probe) : "memory");
            state = 1;
        } else {
            state = 0;
            fprintf(stderr, "ddrbw: note: the kernel does not let EL0 clean a cache line; the probe skips it\n");
        }
        sigaction(SIGILL, &old, NULL);
    }
    return state;
}

// Clean the lines of the buffer from the CPU caches to the point of coherence. The DSP maps the
// buffer with FASTRPC_MAP_FD_DELAYED, and then the user owns the cache maintenance. O(size).
static void clean_dcache(const uint8_t * p, size_t n) {
    if (!cmo_permitted()) {
        return;
    }
    for (size_t i = 0; i < n; i += 64) {
        __asm__ volatile("dc cvac, %0" : : "r"(p + i) : "memory");
    }
    __asm__ volatile("dsb ish" : : : "memory");
}

// The checksum that a thread must give for "chunks" chunks of its part: each full pass adds the
// part, and the last pass adds its first chunks. With "sample", a chunk adds its first 128 bytes
// only. O(chunks of the part) for a sample, O(1) else.
static uint32_t expected_sum(const struct buffer * b, uint32_t part_start, uint32_t part_len, uint32_t chunk,
                             uint64_t chunks, int sample) {
    const uint64_t per    = part_len / chunk;
    const uint64_t passes = chunks / per;
    const uint64_t rem    = chunks % per;
    if (!sample) {
        const uint32_t whole = sum_range(b, part_start, (size_t) part_start + part_len);
        return (uint32_t) passes * whole + sum_range(b, part_start, (size_t) part_start + rem * chunk);
    }
    uint32_t whole = 0;
    uint32_t head  = 0;
    for (uint64_t j = 0; j < per; j++) {
        const size_t   a = (size_t) part_start + j * chunk;
        const uint32_t s = sum_range(b, a, a + BLOCK);
        whole += s;
        if (j < rem) {
            head += s;
        }
    }
    return (uint32_t) passes * whole + head;
}

// ---- the DSP session

static remote_handle64 g_h;
static int             g_session;
static uint32_t        g_votes;
static const char *    g_votes_name = "none";

static const char * rpc_hint(int err) {
    return err == AEE_ENOSUCH ? " (the DSP has no such library: check ADSP_LIBRARY_PATH)" : "";
}

static int dsp_open(void) {
    if (rpc_load() != 0) {
        return EXIT_RPC;
    }
    struct remote_rpc_control_unsigned_module u = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u));
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "ddrbw: error: remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE) failed with 0x%08x. "
                        "This phone does not permit an unsigned PD on the compute DSP.\n", (unsigned) err);
        return EXIT_RPC;
    }
    err = ddrbw_open(ddrbw_URI CDSP_DOMAIN, &g_h);
    if (err != AEE_SUCCESS) {
        const char * path = getenv("ADSP_LIBRARY_PATH");
        fprintf(stderr, "ddrbw: error: ddrbw_open failed with 0x%08x%s. ADSP_LIBRARY_PATH=%s\n", (unsigned) err,
                rpc_hint(err), path != NULL ? path : "(not set)");
        return EXIT_RPC;
    }
    g_session = 1;
    // The FastRPC QoS mode of the backend (ggml-hexagon.cpp): PM QoS on the CPU during calls.
    struct remote_rpc_control_latency l = { .enable = RPC_PM_QOS, .latency = 100 };
    err = remote_handle64_control(g_h, DSPRPC_CONTROL_LATENCY, &l, sizeof(l));
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "ddrbw: note: the FastRPC QoS mode failed with 0x%08x\n", (unsigned) err);
    }
    return 0;
}

static void print_facts(const char * what, const uint64 * f) {
    printf("ddrbw: %s votes=%s arch=%" PRIu64 " hvx=%" PRIu64 " vtcm=%" PRIu64 " max_mips=%" PRIu64
           " max_bus_bw=%" PRIu64 " core_hz=%" PRIu64 " hmx_hz=%" PRIu64 " dcvs=%" PRIu64 " caps=0x%" PRIx64
           " prio=%" PRIu64 "\n",
           what, g_votes_name, (uint64_t) f[DDRBW_FACT_ARCH], (uint64_t) f[DDRBW_FACT_HVX],
           (uint64_t) f[DDRBW_FACT_VTCM], (uint64_t) f[DDRBW_FACT_MAX_MIPS], (uint64_t) f[DDRBW_FACT_MAX_BUS_BW],
           (uint64_t) f[DDRBW_FACT_CORE_HZ], (uint64_t) f[DDRBW_FACT_HMX_HZ], (uint64_t) f[DDRBW_FACT_DCVS],
           (uint64_t) f[DDRBW_FACT_CAPS], (uint64_t) f[DDRBW_FACT_PRIO]);
    fflush(stdout);
}

static int dsp_setup(void) {
    static const char * const names[DDRBW_RC_COUNT] = { "apptype", "protected", "ddrperf", "coreperf", "busperf",
                                                        "expv",    "dcvs",      "hvx",     "hmx",      "ceng",
                                                        "mipsbw",  "caps",      "vtcm" };
    int32  rc[DDRBW_RC_COUNT];
    uint64 facts[DDRBW_FACT_COUNT];
    int    err = ddrbw_setup(g_h, g_votes, rc, DDRBW_RC_COUNT, facts, DDRBW_FACT_COUNT);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "ddrbw: error: ddrbw_setup failed with 0x%08x%s\n", (unsigned) err, rpc_hint(err));
        return EXIT_RPC;
    }
    printf("ddrbw: votes name=%s mask=0x%02x", g_votes_name, (unsigned) g_votes);
    for (int i = 0; i < DDRBW_RC_COUNT; i++) {
        if (rc[i] == DDRBW_RC_NOT_SENT) {
            printf(" %s=-", names[i]);
        } else {
            printf(" %s=0x%x", names[i], (unsigned) rc[i]);
        }
    }
    printf("\n");
    print_facts("facts", facts);
    if (rc[DDRBW_RC_VTCM] != 0) {
        fprintf(stderr, "ddrbw: note: the DSP holds no VTCM (0x%x). The DMA runs stop with status %d.\n",
                (unsigned) rc[DDRBW_RC_VTCM], DDRBW_E_VTCM);
    }
    return 0;
}

static void dsp_close(void) {
    if (g_session) {
        uint64 facts[DDRBW_FACT_COUNT];
        if (ddrbw_query(g_h, facts, DDRBW_FACT_COUNT) == AEE_SUCCESS) {
            print_facts("facts-end", facts);
        }
        ddrbw_close(g_h);
        g_session = 0;
    }
}

// Allocate rpcmem as the backend does for its weights (the system heap, cached), fill it, and map
// it on the DSP with FASTRPC_MAP_FD_DELAYED and HAP_mmap2. Returns 0 or EXIT_RPC.
static int nsp_buffer(struct buffer * b, size_t size) {
    memset(b, 0, sizeof(*b));
    b->fd   = -1;
    b->size = size;
    b->data = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
    if (b->data == NULL) {
        fprintf(stderr, "ddrbw: error: rpcmem_alloc2 of %zu MiB failed. Close apps and run again.\n", size >> 20);
        return EXIT_RPC;
    }
    b->fd = rpcmem_to_fd(b->data);
    if (b->fd < 0 || fill_buffer(b, 0x5eedu) != 0) {
        return EXIT_RPC;
    }
    clean_dcache(b->data, size);
    int err = fastrpc_mmap(CDSP_DOMAIN_ID, b->fd, b->data, 0, size, FASTRPC_MAP_FD_DELAYED);
    if (err != 0) {
        fprintf(stderr, "ddrbw: error: fastrpc_mmap of %zu MiB failed with 0x%08x\n", size >> 20, (unsigned) err);
        return EXIT_RPC;
    }
    b->mapped = 1;
    uint64 ticks = 0;
    err          = ddrbw_map(g_h, b->fd, (uint32_t) size, &ticks);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "ddrbw: error: ddrbw_map of %zu MiB failed with 0x%08x\n", size >> 20, (unsigned) err);
        return EXIT_RPC;
    }
    return 0;
}

static void nsp_buffer_free(struct buffer * b) {
    if (b->data == NULL) {
        return;
    }
    if (b->mapped) {
        uint64 ticks = 0;
        ddrbw_unmap(g_h, b->fd, &ticks);
        fastrpc_munmap(CDSP_DOMAIN_ID, b->fd, b->data, b->size);
    }
    rpcmem_free(b->data);
    free(b->pre);
    memset(b, 0, sizeof(*b));
}

// ---- the NSP configurations

struct nsp_cfg {
    const char * name;
    uint32_t     n_threads;
    uint32_t     n_dma;
    uint32_t     chunk;
    uint32_t     row;
    uint32_t     dst_row;
    uint32_t     depth;
    uint32_t     pf;
    uint32_t     flags;
    uint32_t     region_kib;  // 0 for the full buffer
};

// The GEMV form: the Q8_0 tile rows of hvx_mm_2d_repacked_q8_0 for k = 2560. One descriptor moves
// 80 rows of 1088 bytes (32 weight rows) to a VTCM pitch of 1152 bytes, 8 descriptors in flight.
static const struct nsp_cfg k_cfgs[] = {
    { "hvx-t1", 1, 0, 65536, 0, 0, 0, 2, 0, 0 },
    { "hvx-t2", 2, 0, 65536, 0, 0, 0, 2, 0, 0 },
    { "hvx-t4", 4, 0, 65536, 0, 0, 0, 2, 0, 0 },
    { "hvx-t6", 6, 0, 65536, 0, 0, 0, 2, 0, 0 },
    { "hvx-nopf-t1", 1, 0, 65536, 0, 0, 0, 0, 0, 0 },
    { "hvx-nopf-t6", 6, 0, 65536, 0, 0, 0, 0, 0, 0 },
    { "hvx-pf4-t6", 6, 0, 65536, 0, 0, 0, 4, 0, 0 },
    { "dma-t1", 1, 1, 65536, 0, 0, 8, 0, 0, 0 },
    { "dma-t2", 2, 2, 65536, 0, 0, 8, 0, 0, 0 },
    { "dma-t4", 4, 4, 65536, 0, 0, 8, 0, 0, 0 },
    { "dma-t6", 6, 6, 65536, 0, 0, 8, 0, 0, 0 },
    { "gemv-t1", 1, 1, 87040, 1088, 1152, 8, 0, 0, 0 },
    { "gemv-t2", 2, 2, 87040, 1088, 1152, 8, 0, 0, 0 },
    { "gemv-t4", 4, 4, 87040, 1088, 1152, 8, 0, 0, 0 },
    { "gemv-t6", 6, 6, 87040, 1088, 1152, 8, 0, 0, 0 },
    { "gemv-sum-t6", 6, 6, 87040, 1088, 1152, 8, 0, DDRBW_FLAG_SUM, 0 },
    { "gemv-d2-t6", 6, 6, 87040, 1088, 1152, 2, 0, 0, 0 },
    { "gemv-d4-t6", 6, 6, 87040, 1088, 1152, 4, 0, 0, 0 },
    { "lin87k-t6", 6, 6, 87040, 0, 0, 8, 0, 0, 0 },
    { "dma-sum-t6", 6, 6, 65536, 0, 0, 8, 0, DDRBW_FLAG_SUM, 0 },
    { "dma-cached-t6", 6, 6, 65536, 0, 0, 8, 0, DDRBW_FLAG_CACHED, 0 },
    { "dma16k-t6", 6, 6, 16384, 0, 0, 16, 0, 0, 0 },
    { "dma256k-t6", 6, 6, 262144, 0, 0, 4, 0, 0, 0 },
    { "mix-3d3h", 6, 3, 65536, 0, 0, 8, 2, 0, 0 },
    { "mix-4d2h", 6, 4, 65536, 0, 0, 8, 2, 0, 0 },
    { "ws-256k", 6, 6, 16384, 0, 0, 8, 0, 0, 256 },
    { "ws-1m", 6, 6, 16384, 0, 0, 8, 0, 0, 1024 },
    { "ws-4m", 6, 6, 16384, 0, 0, 8, 0, 0, 4096 },
    { "ws-8m", 6, 6, 16384, 0, 0, 8, 0, 0, 8192 },
    { "ws-16m", 6, 6, 16384, 0, 0, 8, 0, 0, 16384 },
    { "ws-64m", 6, 6, 16384, 0, 0, 8, 0, 0, 65536 },
    { "ws-full", 6, 6, 16384, 0, 0, 8, 0, 0, 0 },
};
#define N_CFGS ((int) (sizeof(k_cfgs) / sizeof(k_cfgs[0])))

static const char * const k_plan_full =
    "hvx-t1,hvx-t2,hvx-t4,hvx-t6,hvx-nopf-t1,hvx-nopf-t6,hvx-pf4-t6,dma-t1,dma-t2,dma-t4,dma-t6,gemv-t1,gemv-t2,"
    "gemv-t4,gemv-t6,gemv-sum-t6,gemv-d2-t6,gemv-d4-t6,lin87k-t6,dma-sum-t6,dma-cached-t6,dma16k-t6,dma256k-t6,"
    "mix-3d3h,mix-4d2h,ws-256k,ws-1m,ws-4m,ws-8m,ws-16m,ws-64m,ws-full";
static const char * const k_plan_short = "hvx-t6,dma-t6,gemv-t6,gemv-sum-t6,mix-3d3h";
static const char * const k_plan_ws    = "ws-256k,ws-1m,ws-4m,ws-8m,ws-16m,ws-64m,ws-full";

static const struct nsp_cfg * find_cfg(const char * name, size_t len) {
    for (int i = 0; i < N_CFGS; i++) {
        if (strlen(k_cfgs[i].name) == len && strncmp(k_cfgs[i].name, name, len) == 0) {
            return &k_cfgs[i];
        }
    }
    return NULL;
}

// Parse a comma list of configuration names. Returns the count, or -1 after a message.
static int parse_cfgs(const char * list, const struct nsp_cfg ** out, int max) {
    int          n = 0;
    const char * p = list;
    while (*p) {
        const char * e = strchr(p, ',');
        const size_t l = e ? (size_t) (e - p) : strlen(p);
        const struct nsp_cfg * c = find_cfg(p, l);
        if (c == NULL || n == max) {
            fprintf(stderr, "ddrbw: error: \"%.*s\" is not a configuration of k_cfgs, or the list is too long\n", (int) l, p);
            return -1;
        }
        out[n++] = c;
        p += l;
        if (*p == ',') {
            p++;
        }
    }
    return n;
}

// The result of one NSP run.
struct nsp_result {
    int      status;
    int      check_ok;
    uint64_t t_go;
    uint64_t t_end;  // the latest end of a thread
    uint64_t bytes;
    double   gbs;
    double   tmin;  // the lowest rate of one thread in GB/s
    double   tmax;
    double   ghz;   // the processor clock of thread 0 in the loop
};

static int nsp_run(const struct buffer * b, const struct nsp_cfg * c, uint32_t budget_ms, struct nsp_result * r) {
    ddrbw_run run;
    memset(&run, 0, sizeof(run));
    run.fd        = b->fd;
    run.region    = c->region_kib ? c->region_kib * 1024u : (uint32_t) b->size;
    run.n_threads = c->n_threads;
    run.n_dma     = c->n_dma;
    run.chunk     = c->chunk;
    run.row       = c->row;
    run.dst_row   = c->dst_row;
    run.depth     = c->depth;
    run.pf        = c->pf;
    run.flags     = c->flags;
    run.budget_us = budget_ms * 1000u;

    ddrbw_thread th[MAX_THREADS];
    uint64       t_go   = 0;
    int32        status = 0;
    memset(r, 0, sizeof(*r));
    const int err = ddrbw_read(g_h, &run, th, MAX_THREADS, &t_go, &status);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "ddrbw: error: ddrbw_read(%s) failed with 0x%08x%s\n", c->name, (unsigned) err, rpc_hint(err));
        return EXIT_RPC;
    }
    r->status   = status;
    r->t_go     = t_go;
    r->check_ok = status == DDRBW_OK;
    r->tmin     = 1e30;
    if (status != DDRBW_OK) {
        return 0;
    }
    for (uint32_t i = 0; i < c->n_threads; i++) {
        const uint64_t bytes = th[i].chunks * c->chunk;
        const int      sample = th[i].kind == DDRBW_KIND_DMA && !(c->flags & DDRBW_FLAG_SUM);
        const uint32_t want   = expected_sum(b, th[i].part_start, th[i].part_len, c->chunk, th[i].chunks, sample);
        if (want != th[i].sum) {
            r->check_ok = 0;
            fprintf(stderr, "ddrbw: error: %s thread %u: the checksum is 0x%08x, the host expects 0x%08x (%" PRIu64
                            " chunks)\n", c->name, i, (unsigned) th[i].sum, (unsigned) want, (uint64_t) th[i].chunks);
        }
        r->bytes += bytes;
        if (th[i].t_end > r->t_end) {
            r->t_end = th[i].t_end;
        }
        const double s    = ticks_to_s(th[i].t_end - th[i].t_start);
        const double rate = s > 0 ? (double) bytes / s / 1e9 : 0;
        r->tmin           = rate < r->tmin ? rate : r->tmin;
        r->tmax           = rate > r->tmax ? rate : r->tmax;
        if (i == 0 && s > 0) {
            r->ghz = (double) th[i].pcycles / s / 1e9;
        }
    }
    const double s = ticks_to_s(r->t_end - r->t_go);
    r->gbs         = s > 0 ? (double) r->bytes / s / 1e9 : 0;
    return 0;
}

static void print_nsp(const char * what, const struct nsp_cfg * c, int rep, const struct nsp_result * r) {
    printf("ddrbw: %s votes=%s cfg=%s rep=%d threads=%u dma=%u chunk=%u row=%u depth=%u pf=%u flags=%u region_kib=%u "
           "bytes=%" PRIu64 " ms=%.2f gbs=%.2f tmin=%.2f tmax=%.2f ghz=%.3f t_go=%" PRIu64 " t_end=%" PRIu64
           " status=%d check=%s\n",
           what, g_votes_name, c->name, rep, c->n_threads, c->n_dma, c->chunk, c->row, c->depth, c->pf, c->flags,
           c->region_kib, r->bytes, ticks_to_s(r->t_end - r->t_go) * 1e3, r->gbs, r->status ? 0 : r->tmin,
           r->tmax, r->ghz, r->t_go, r->t_end, r->status, r->check_ok ? "ok" : "FAIL");
    fflush(stdout);
}

// ---- the CPU reader

struct core_set {
    int  n;
    int  cpu[MAX_THREADS];
    char text[48];
};

// Parse "7/6,7/0-5" into sets. Returns the count, or -1 after a message.
static int parse_sets(const char * list, struct core_set * sets, int max) {
    int          n = 0;
    const char * p = list;
    while (*p) {
        if (n == max) {
            fprintf(stderr, "ddrbw: error: more than %d CPU sets\n", max);
            return -1;
        }
        struct core_set * s = &sets[n];
        memset(s, 0, sizeof(*s));
        const char * end = strchr(p, '/');
        const size_t len = end ? (size_t) (end - p) : strlen(p);
        snprintf(s->text, sizeof(s->text), "%.*s", (int) len, p);
        const char * q = s->text;
        while (*q) {
            char *     e;
            const long a = strtol(q, &e, 10);
            long       z = a;
            if (e == q) {
                fprintf(stderr, "ddrbw: error: the CPU set \"%s\" is not a list such as 0-3,6\n", s->text);
                return -1;
            }
            if (*e == '-') {
                q = e + 1;
                z = strtol(q, &e, 10);
            }
            for (long c = a; c <= z; c++) {
                if (c < 0 || c > 63 || s->n == MAX_THREADS) {
                    fprintf(stderr, "ddrbw: error: the CPU set \"%s\" has a core that is not valid, or more than %d cores\n",
                            s->text, MAX_THREADS);
                    return -1;
                }
                s->cpu[s->n++] = (int) c;
            }
            q = *e == ',' ? e + 1 : e;
            if (*e != ',' && *e != '\0') {
                fprintf(stderr, "ddrbw: error: the CPU set \"%s\" is not a list such as 0-3,6\n", s->text);
                return -1;
            }
        }
        n++;
        p += len;
        if (*p == '/') {
            p++;
        }
    }
    return n;
}

struct cpu_log {
    uint64_t t;
    uint64_t chunks;
};

struct cpu_worker {
    const struct buffer * b;
    int                   cpu;
    uint32_t              part_start;
    uint32_t              part_len;
    uint64_t              deadline;  // cntvct; the thread also stops at *stop
    atomic_int *          stop;
    atomic_int *          ready;
    atomic_int *          go;
    struct cpu_log *      log;       // NULL for no log
    uint32_t              n_log;
    uint64_t              t_start;
    uint64_t              t_end;
    uint64_t              chunks;
    uint32_t              sum;
    int                   pinned;    // 1 when the thread ran on its core at the start and at the end
    pthread_t             tid;
};

// Add the words of n bytes with NEON loads, 128 bytes per step into 8 accumulators. O(n).
static inline void neon_add(const uint8_t * p, size_t n, uint32x4_t acc[8]) {
    uint32x4_t a0 = acc[0], a1 = acc[1], a2 = acc[2], a3 = acc[3];
    uint32x4_t a4 = acc[4], a5 = acc[5], a6 = acc[6], a7 = acc[7];
    for (size_t i = 0; i < n; i += 128) {
        const uint32x4x4_t x = vld1q_u32_x4((const uint32_t *) (p + i));
        const uint32x4x4_t y = vld1q_u32_x4((const uint32_t *) (p + i + 64));
        a0                   = vaddq_u32(a0, x.val[0]);
        a1                   = vaddq_u32(a1, x.val[1]);
        a2                   = vaddq_u32(a2, x.val[2]);
        a3                   = vaddq_u32(a3, x.val[3]);
        a4                   = vaddq_u32(a4, y.val[0]);
        a5                   = vaddq_u32(a5, y.val[1]);
        a6                   = vaddq_u32(a6, y.val[2]);
        a7                   = vaddq_u32(a7, y.val[3]);
    }
    acc[0] = a0, acc[1] = a1, acc[2] = a2, acc[3] = a3;
    acc[4] = a4, acc[5] = a5, acc[6] = a6, acc[7] = a7;
}

static void * cpu_main(void * arg) {
    struct cpu_worker * w = (struct cpu_worker *) arg;
    cpu_set_t           set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    const int aff = sched_setaffinity(0, sizeof(set), &set);
    int       on  = aff == 0 && sched_getcpu() == w->cpu;

    uint32x4_t acc[8];
    for (int i = 0; i < 8; i++) {
        acc[i] = vdupq_n_u32(0);
    }
    const uint32_t  nchunks = w->part_len / CPU_CHUNK;
    const uint8_t * part    = w->b->data + w->part_start;
    uint32_t        c       = 0;
    uint64_t        n       = 0;

    atomic_fetch_add(w->ready, 1);
    while (!atomic_load_explicit(w->go, memory_order_acquire)) {
    }
    w->t_start = cntvct();
    for (;;) {
        neon_add(part + (size_t) c * CPU_CHUNK, CPU_CHUNK, acc);
        n++;
        if (++c == nchunks) {
            c = 0;
        }
        if (n % CPU_LOG_STEP == 0) {
            const uint64_t t = cntvct();
            if (w->log != NULL && w->n_log < CPU_LOG_MAX) {
                w->log[w->n_log++] = (struct cpu_log) { t, n };
            }
            if (t >= w->deadline || atomic_load_explicit(w->stop, memory_order_relaxed)) {
                break;
            }
        }
    }
    w->t_end  = cntvct();
    w->chunks = n;
    uint32x4_t s = vaddq_u32(vaddq_u32(vaddq_u32(acc[0], acc[1]), vaddq_u32(acc[2], acc[3])),
                             vaddq_u32(vaddq_u32(acc[4], acc[5]), vaddq_u32(acc[6], acc[7])));
    w->sum    = vaddvq_u32(s);
    w->pinned = on && sched_getcpu() == w->cpu;
    return NULL;
}

// A group of CPU threads that read one buffer.
struct cpu_group {
    struct cpu_worker w[MAX_THREADS];
    int               n;
    atomic_int        stop;
    atomic_int        ready;
    atomic_int        go;
};

// Start the threads of one set. They start to read at once. Returns 0, or -1 after a message.
static int cpu_start(struct cpu_group * g, const struct buffer * b, const struct core_set * s, uint64_t deadline,
                     struct cpu_log * logs) {
    memset(g, 0, sizeof(*g));
    g->n                     = s->n;
    const uint32_t part_len = (uint32_t) (b->size / (size_t) s->n / CPU_CHUNK * CPU_CHUNK);
    for (int i = 0; i < s->n; i++) {
        struct cpu_worker * w = &g->w[i];
        w->b                  = b;
        w->cpu                = s->cpu[i];
        w->part_start         = (uint32_t) i * part_len;
        w->part_len           = part_len;
        w->deadline           = deadline;
        w->stop               = &g->stop;
        w->ready              = &g->ready;
        w->go                 = &g->go;
        w->log                = logs ? logs + (size_t) i * CPU_LOG_MAX : NULL;
        if (pthread_create(&w->tid, NULL, cpu_main, w) != 0) {
            fprintf(stderr, "ddrbw: error: pthread_create failed\n");
            atomic_store(&g->stop, 1);
            atomic_store(&g->go, 1);
            for (int k = 0; k < i; k++) {
                pthread_join(g->w[k].tid, NULL);
            }
            return -1;
        }
    }
    while (atomic_load(&g->ready) < s->n) {
    }
    atomic_store_explicit(&g->go, 1, memory_order_release);
    return 0;
}

// Wait for the threads, and check their checksums. Returns 1 when each checksum is correct.
static int cpu_join(struct cpu_group * g) {
    int ok = 1;
    for (int i = 0; i < g->n; i++) {
        struct cpu_worker * w = &g->w[i];
        pthread_join(w->tid, NULL);
        const uint32_t want = expected_sum(w->b, w->part_start, w->part_len, CPU_CHUNK, w->chunks, 0);
        if (want != w->sum) {
            ok = 0;
            fprintf(stderr, "ddrbw: error: CPU %d: the checksum is 0x%08x, the host expects 0x%08x\n", w->cpu,
                    (unsigned) w->sum, (unsigned) want);
        }
    }
    return ok;
}

struct cpu_result {
    uint64_t bytes;
    double   gbs;
    double   tmin;
    double   tmax;
    int      pinned;
    int      check_ok;
};

static void cpu_summary(const struct cpu_group * g, struct cpu_result * r) {
    uint64_t t0 = UINT64_MAX;
    uint64_t t1 = 0;
    r->bytes    = 0;
    r->tmin     = 1e30;
    r->tmax     = 0;
    r->pinned   = 1;
    for (int i = 0; i < g->n; i++) {
        const struct cpu_worker * w = &g->w[i];
        const uint64_t            b = w->chunks * CPU_CHUNK;
        const double              s = ticks_to_s(w->t_end - w->t_start);
        const double              v = s > 0 ? (double) b / s / 1e9 : 0;
        r->bytes += b;
        t0        = w->t_start < t0 ? w->t_start : t0;
        t1        = w->t_end > t1 ? w->t_end : t1;
        r->tmin   = v < r->tmin ? v : r->tmin;
        r->tmax   = v > r->tmax ? v : r->tmax;
        r->pinned = r->pinned && w->pinned;
    }
    const double s = ticks_to_s(t1 - t0);
    r->gbs         = s > 0 ? (double) r->bytes / s / 1e9 : 0;
}

// The chunks that a thread read at time t, from its progress samples, by linear interpolation.
// Returns -1 when t is outside the samples. O(log samples).
static double chunks_at(const struct cpu_worker * w, uint64_t t) {
    if (w->n_log < 2 || t < w->log[0].t || t > w->log[w->n_log - 1].t) {
        return -1;
    }
    uint32_t lo = 0;
    uint32_t hi = w->n_log - 1;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        if (w->log[mid].t <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const struct cpu_log * a = &w->log[lo];
    const struct cpu_log * z = &w->log[hi];
    const double           f = z->t > a->t ? (double) (t - a->t) / (double) (z->t - a->t) : 0;
    return (double) a->chunks + f * (double) (z->chunks - a->chunks);
}

static int cpu_buffer(struct buffer * b, size_t size) {
    memset(b, 0, sizeof(*b));
    b->fd   = -1;
    b->size = size;
    void * p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) {
        fprintf(stderr, "ddrbw: error: mmap of %zu MiB failed: %s\n", size >> 20, strerror(errno));
        return EXIT_RPC;
    }
    b->data = (uint8_t *) p;
    return fill_buffer(b, 0xc0ffeeu) == 0 ? 0 : EXIT_RPC;
}

static void cpu_buffer_free(struct buffer * b) {
    if (b->data != NULL) {
        munmap(b->data, b->size);
        free(b->pre);
        memset(b, 0, sizeof(*b));
    }
}

// ---- the latency tests

static int cmp_double(const void * a, const void * b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return x < y ? -1 : x > y;
}

// Print the distribution of n samples in microseconds on one line of the kind "what". The samples
// get sorted. O(n log n).
static void print_dist_kind(const char * what, const char * fields, double * us, size_t n) {
    if (n == 0) {
        printf("ddrbw: %s votes=%s %s n=0\n", what, g_votes_name, fields);
        fflush(stdout);
        return;
    }
    qsort(us, n, sizeof(double), cmp_double);
    double mean = 0;
    for (size_t i = 0; i < n; i++) {
        mean += us[i];
    }
    mean /= (double) n;
    printf("ddrbw: %s votes=%s %s n=%zu min=%.2f p10=%.2f p50=%.2f p90=%.2f p99=%.2f max=%.2f mean=%.2f\n", what,
           g_votes_name, fields, n, us[0], us[n / 10], us[n / 2], us[n * 9 / 10], us[n * 99 / 100], us[n - 1], mean);
    fflush(stdout);
}

static void print_dist(const char * fields, double * us, size_t n) {
    print_dist_kind("rtt", fields, us, n);
}

static void rtt_fastrpc(uint32_t iters, double * us) {
    size_t n = 0;
    for (uint32_t i = 0; i < iters + 10; i++) {
        const uint64_t t0 = cntvct();
        const int      e  = ddrbw_nop(g_h);
        const uint64_t t1 = cntvct();
        if (e != AEE_SUCCESS) {
            fprintf(stderr, "ddrbw: error: ddrbw_nop failed with 0x%08x\n", (unsigned) e);
            break;
        }
        if (i >= 10) {
            us[n++] = ticks_to_s(t1 - t0) * 1e6;
        }
    }
    print_dist("kind=fastrpc", us, n);
}

// One dspqueue test: the DSP thread waits in "dsp_mode", the host waits with dspqueue_read (poll 0)
// or with a loop of dspqueue_read_noblock (poll 1), and each packet refers to the buffer or not.
static void rtt_queue(uint32_t dsp_mode, int poll, int use_buf, const struct buffer * qbuf, uint32_t iters, double * us) {
    static const char * const dsp_names[] = { "peek", "backend", "spin" };
    char                      fields[128];
    snprintf(fields, sizeof(fields), "kind=queue dsp=%s host=%s buf=%d", dsp_names[dsp_mode], poll ? "poll" : "block",
             use_buf);

    dspqueue_t q  = NULL;
    uint64_t   id = 0;
    int        e  = dspqueue_create(CDSP_DOMAIN_ID, 0, 4096, 4096, NULL, NULL, NULL, &q);
    if (e == 0) {
        e = dspqueue_export(q, &id);
    }
    if (e == 0) {
        e = ddrbw_queue_start(g_h, id, dsp_mode);
    }
    if (e != 0) {
        printf("ddrbw: rtt votes=%s %s error=0x%08x\n", g_votes_name, fields, (unsigned) e);
        fflush(stdout);
        if (q != NULL) {
            dspqueue_close(q);
        }
        return;
    }
    size_t                 n = 0;
    struct dspqueue_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.fd    = (uint32_t) qbuf->fd;
    buf.ptr   = qbuf->data;
    buf.size  = 4096;
    buf.flags = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    int failed = 0;
    for (uint32_t i = 0; i <= iters + 10 && !failed; i++) {
        struct ddrbw_qmsg m = { i, i == iters + 10 ? DDRBW_QCMD_STOP : DDRBW_QCMD_ECHO, 0 };
        const uint64_t    t0 = cntvct();
        e = dspqueue_write(q, 0, use_buf ? 1 : 0, use_buf ? &buf : NULL, sizeof(m), (const uint8_t *) &m, 1000000);
        if (e != 0) {
            fprintf(stderr, "ddrbw: error: dspqueue_write failed with 0x%08x\n", (unsigned) e);
            failed = 1;
            break;
        }
        struct ddrbw_qmsg      r;
        struct dspqueue_buffer rb;
        uint32_t               flags = 0, nb = 0, len = 0;
        if (poll) {
            do {
                e = dspqueue_read_noblock(q, &flags, 1, &nb, &rb, sizeof(r), &len, (uint8_t *) &r);
            } while (e == AEE_EWOULDBLOCK && ticks_to_s(cntvct() - t0) < 1.0);
        } else {
            e = dspqueue_read(q, &flags, 1, &nb, &rb, sizeof(r), &len, (uint8_t *) &r, 1000000);
        }
        const uint64_t t1 = cntvct();
        if (e != 0 || len != sizeof(r) || r.seq != i) {
            fprintf(stderr, "ddrbw: error: dspqueue read of packet %u: 0x%08x (length %u, seq %u)\n", i, (unsigned) e,
                    len, e == 0 ? r.seq : 0);
            failed = 1;
            break;
        }
        if (i >= 10 && i < iters + 10) {
            us[n++] = ticks_to_s(t1 - t0) * 1e6;
        }
    }
    uint32_t served = 0;
    if (failed) {
        // The DSP thread waits for a stop packet. After a failed write or read it continues to wait,
        // thus the exit of the process stops it.
        printf("ddrbw: rtt votes=%s %s error=io\n", g_votes_name, fields);
        fflush(stdout);
        return;
    }
    e = ddrbw_queue_stop(g_h, &served);
    dspqueue_close(q);
    char f2[192];
    snprintf(f2, sizeof(f2), "%s served=%u dsp_err=0x%x", fields, served, (unsigned) e);
    print_dist(f2, us, n);
}

struct fence_host {
    volatile uint32_t * host;
    volatile uint32_t * dsp;
    uint32_t            iters;
    int                 cmo;
    double *            us;
    size_t              n;
    uint32_t            done;
};

static inline void cmo_civac(volatile void * p) {
    if (cmo_permitted()) {
        __asm__ volatile("dc civac, %0\n dsb ish" : : "r"(p) : "memory");
    }
}

static void * fence_host_main(void * arg) {
    struct fence_host * f = (struct fence_host *) arg;
    for (uint32_t i = 1; i <= f->iters; i++) {
        const uint64_t t0 = cntvct();
        __atomic_store_n(f->host, i, __ATOMIC_RELEASE);
        if (f->cmo) {
            cmo_civac(f->host);
        }
        for (;;) {
            if (f->cmo) {
                cmo_civac(f->dsp);
            }
            if (__atomic_load_n(f->dsp, __ATOMIC_ACQUIRE) == i) {
                break;
            }
            if (ticks_to_s(cntvct() - t0) > 0.2) {
                return NULL;
            }
        }
        const uint64_t t1 = cntvct();
        f->done           = i;
        if (i > 16) {
            f->us[f->n++] = ticks_to_s(t1 - t0) * 1e6;
        }
    }
    return NULL;
}

static void rtt_fence(const struct buffer * fb, int cmo, uint32_t iters, double * us) {
    volatile uint32_t * host = (volatile uint32_t *) fb->data;
    volatile uint32_t * dsp  = (volatile uint32_t *) (fb->data + DDRBW_FENCE_DSP);
    __atomic_store_n(host, 0, __ATOMIC_RELEASE);
    __atomic_store_n(dsp, 0, __ATOMIC_RELEASE);
    cmo_civac(host);
    cmo_civac(dsp);

    struct fence_host f = { host, dsp, iters, cmo, us, 0, 0 };
    pthread_t         tid;
    if (pthread_create(&tid, NULL, fence_host_main, &f) != 0) {
        fprintf(stderr, "ddrbw: error: pthread_create failed\n");
        return;
    }
    uint32 done = 0;
    uint64 tf = 0, tl = 0;
    const int e = ddrbw_fence(g_h, fb->fd, iters, 100000, &done, &tf, &tl);
    pthread_join(tid, NULL);
    char fields[128];
    snprintf(fields, sizeof(fields), "kind=fence host=%s done_host=%u done_dsp=%u err=0x%x",
             cmo ? (cmo_permitted() ? "cmo" : "cmo-trapped") : "plain",
             f.done, (unsigned) done, (unsigned) e);
    print_dist(fields, us, f.n);
}

// ---- the modes

struct options {
    const char * mode;
    const char * votes;
    const char * plan;
    const char * cfgs;
    const char * sets;
    uint32_t     mb;
    uint32_t     reps;
    uint32_t     budget_ms;
    uint32_t     iters;
    uint32_t     hz;
    uint32_t     ms;
    const char * out;
};

static int mode_nsp(const struct options * o) {
    const char * list = o->plan;
    if (strcmp(list, "full") == 0) {
        list = k_plan_full;
    } else if (strcmp(list, "short") == 0) {
        list = k_plan_short;
    } else if (strcmp(list, "ws") == 0) {
        list = k_plan_ws;
    }
    const struct nsp_cfg * cfgs[64];
    const int              n = parse_cfgs(list, cfgs, 64);
    if (n <= 0) {
        return EXIT_USAGE;
    }
    struct buffer b;
    int           rc = nsp_buffer(&b, (size_t) o->mb << 20);
    if (rc != 0) {
        nsp_buffer_free(&b);
        return rc;
    }
    int bad = 0;
    for (uint32_t rep = 1; rep <= o->reps && rc == 0; rep++) {
        for (int k = 0; k < n && rc == 0; k++) {
            const struct nsp_cfg * c = cfgs[rep % 2 ? k : n - 1 - k];
            struct nsp_result      r;
            rc = nsp_run(&b, c, o->budget_ms, &r);
            if (rc == 0) {
                print_nsp("nsp", c, (int) rep, &r);
                bad |= !r.check_ok;
            }
        }
    }
    nsp_buffer_free(&b);
    return rc ? rc : (bad ? EXIT_CHECK : 0);
}

static int mode_cpu(const struct options * o) {
    struct core_set sets[MAX_SETS];
    const int      ns = parse_sets(o->sets, sets, MAX_SETS);
    if (ns <= 0) {
        return EXIT_USAGE;
    }
    struct buffer b;
    int           rc = cpu_buffer(&b, (size_t) o->mb << 20);
    if (rc != 0) {
        cpu_buffer_free(&b);
        return rc;
    }
    int bad = 0;
    for (uint32_t rep = 1; rep <= o->reps && rc == 0; rep++) {
        for (int k = 0; k < ns && rc == 0; k++) {
            const struct core_set * s = &sets[rep % 2 ? k : ns - 1 - k];
            struct cpu_group       g;
            const uint64_t deadline = cntvct() + (uint64_t) ((double) o->budget_ms * 1e-3 * (double) g_cntfrq);
            if (cpu_start(&g, &b, s, deadline, NULL) != 0) {
                rc = EXIT_RPC;
                break;
            }
            const int ok = cpu_join(&g);
            struct cpu_result r;
            cpu_summary(&g, &r);
            printf("ddrbw: cpu set=%s rep=%u threads=%d bytes=%" PRIu64 " gbs=%.2f tmin=%.2f tmax=%.2f pinned=%d check=%s\n",
                   s->text, rep, s->n, r.bytes, r.gbs, r.tmin, r.tmax, r.pinned, ok ? "ok" : "FAIL");
            fflush(stdout);
            bad |= !ok;
            sleep_ms(50);
        }
    }
    cpu_buffer_free(&b);
    return rc ? rc : (bad ? EXIT_CHECK : 0);
}

static int mode_both(const struct options * o) {
    const struct nsp_cfg * cfgs[64];
    const int              nc = parse_cfgs(o->cfgs, cfgs, 64);
    struct core_set         sets[MAX_SETS];
    const int              ns = parse_sets(o->sets, sets, MAX_SETS);
    if (nc <= 0 || ns <= 0) {
        return EXIT_USAGE;
    }
    struct buffer nb;
    struct buffer cb;
    memset(&cb, 0, sizeof(cb));
    int rc = nsp_buffer(&nb, (size_t) o->mb << 20);
    if (rc == 0) {
        rc = cpu_buffer(&cb, (size_t) o->mb << 20);
    }
    struct cpu_log * logs = (struct cpu_log *) malloc(sizeof(struct cpu_log) * CPU_LOG_MAX * MAX_THREADS);
    if (logs == NULL && rc == 0) {
        fprintf(stderr, "ddrbw: error: no memory for the progress logs\n");
        rc = EXIT_RPC;
    }
    int bad = 0;
    for (uint32_t rep = 1; rep <= o->reps && rc == 0; rep++) {
        for (int kc = 0; kc < nc && rc == 0; kc++) {
            for (int ks = 0; ks < ns && rc == 0; ks++) {
                const struct nsp_cfg * c = cfgs[rep % 2 ? kc : nc - 1 - kc];
                const struct core_set * s = &sets[rep % 2 ? ks : ns - 1 - ks];
                const uint64_t         budget_ticks = (uint64_t) ((double) o->budget_ms * 1e-3 * (double) g_cntfrq);

                // 1. The NSP alone.
                struct nsp_result na;
                if ((rc = nsp_run(&nb, c, o->budget_ms, &na)) != 0) {
                    break;
                }
                sleep_ms(30);

                // 2. The CPU alone.
                struct cpu_group g;
                if (cpu_start(&g, &cb, s, cntvct() + budget_ticks, NULL) != 0) {
                    rc = EXIT_RPC;
                    break;
                }
                int               ok = cpu_join(&g);
                struct cpu_result ca;
                cpu_summary(&g, &ca);
                sleep_ms(30);

                // 3. The two at the same time. The CPU threads start first and stop after the NSP run,
                // and their bytes count only in the window of the NSP run.
                if (cpu_start(&g, &cb, s, cntvct() + 4 * budget_ticks + g_cntfrq, logs) != 0) {
                    rc = EXIT_RPC;
                    break;
                }
                sleep_ms(30);
                struct nsp_result nt;
                const uint64_t    h0 = cntvct();
                rc                   = nsp_run(&nb, c, o->budget_ms, &nt);
                const uint64_t h1    = cntvct();
                sleep_ms(10);
                atomic_store(&g.stop, 1);
                ok = cpu_join(&g) && ok;
                if (rc != 0) {
                    break;
                }
                const int clock_ok = nt.status == DDRBW_OK && h0 <= nt.t_go && nt.t_go < nt.t_end && nt.t_end <= h1;
                double    cpu_chunks = 0;
                int       window_ok  = 1;
                for (int i = 0; i < g.n; i++) {
                    const double a = chunks_at(&g.w[i], nt.t_go);
                    const double z = chunks_at(&g.w[i], nt.t_end);
                    if (a < 0 || z < 0) {
                        window_ok = 0;
                    } else {
                        cpu_chunks += z - a;
                    }
                }
                const double win    = ticks_to_s(nt.t_end - nt.t_go);
                const double cpu_bt = window_ok && win > 0 ? cpu_chunks * CPU_CHUNK / win / 1e9 : 0;
                printf("ddrbw: both votes=%s cfg=%s set=%s rep=%u nsp_alone=%.2f cpu_alone=%.2f nsp_both=%.2f "
                       "cpu_both=%.2f sum_both=%.2f window_ms=%.1f t_go=%" PRIu64 " t_end=%" PRIu64
                       " clock=%s window=%s pinned=%d check=%s\n",
                       g_votes_name, c->name, s->text, rep, na.gbs, ca.gbs, nt.gbs, cpu_bt, nt.gbs + cpu_bt, win * 1e3,
                       nt.t_go, nt.t_end,
                       clock_ok ? "ok" : "FAIL", window_ok ? "ok" : "FAIL", ca.pinned,
                       ok && na.check_ok && nt.check_ok ? "ok" : "FAIL");
                fflush(stdout);
                bad |= !(ok && na.check_ok && nt.check_ok);
                sleep_ms(50);
            }
        }
    }
    free(logs);
    cpu_buffer_free(&cb);
    nsp_buffer_free(&nb);
    return rc ? rc : (bad ? EXIT_CHECK : 0);
}

static int mode_rtt(const struct options * o) {
    // The fence test takes 10 times the iterations of the other tests.
    double * us = (double *) malloc(sizeof(double) * ((size_t) o->iters * 10u + 64u));
    if (us == NULL) {
        return EXIT_RPC;
    }
    rtt_fastrpc(o->iters, us);

    // The buffer of the queue packets: pinned rpcmem as the op batch buffer of the backend.
    struct buffer qb;
    memset(&qb, 0, sizeof(qb));
    qb.fd   = -1;
    qb.size = DDRBW_FENCE_BYTES;
    qb.data = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, qb.size);
    int rc  = 0;
    if (qb.data == NULL || (qb.fd = rpcmem_to_fd(qb.data)) < 0 ||
        fastrpc_mmap(CDSP_DOMAIN_ID, qb.fd, qb.data, 0, qb.size, FASTRPC_MAP_FD) != 0) {
        fprintf(stderr, "ddrbw: error: the queue buffer failed\n");
        rc = EXIT_RPC;
    } else {
        memset(qb.data, 0, qb.size);
        for (uint32_t mode = 0; mode <= DDRBW_QMODE_SPIN; mode++) {
            for (int poll = 0; poll <= 1; poll++) {
                for (int use_buf = 0; use_buf <= 1; use_buf++) {
                    rtt_queue(mode, poll, use_buf, &qb, o->iters, us);
                }
            }
        }
        fastrpc_munmap(CDSP_DOMAIN_ID, qb.fd, qb.data, qb.size);
    }
    if (qb.data != NULL) {
        rpcmem_free(qb.data);
    }

    // The fence buffer: rpcmem mapped as the fence buffer of the backend (not pinned, HAP_mmap2).
    struct buffer fb;
    uint64        ticks = 0;
    memset(&fb, 0, sizeof(fb));
    fb.fd   = -1;
    fb.size = DDRBW_FENCE_BYTES;
    fb.data = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, fb.size);
    if (rc == 0 && (fb.data == NULL || (fb.fd = rpcmem_to_fd(fb.data)) < 0 ||
                    fastrpc_mmap(CDSP_DOMAIN_ID, fb.fd, fb.data, 0, fb.size, FASTRPC_MAP_FD_DELAYED) != 0 ||
                    ddrbw_map(g_h, fb.fd, (uint32_t) fb.size, &ticks) != AEE_SUCCESS)) {
        fprintf(stderr, "ddrbw: error: the fence buffer failed\n");
        rc = EXIT_RPC;
    }
    if (rc == 0) {
        memset(fb.data, 0, fb.size);
        clean_dcache(fb.data, fb.size);
        rtt_fence(&fb, 0, o->iters * 10u, us);
        rtt_fence(&fb, 1, o->iters * 10u, us);
        ddrbw_unmap(g_h, fb.fd, &ticks);
        fastrpc_munmap(CDSP_DOMAIN_ID, fb.fd, fb.data, fb.size);
    }
    if (fb.data != NULL) {
        rpcmem_free(fb.data);
    }
    free(us);
    return rc;
}

// ---- the cost of a mapping

// The cost of the mapping of one model chunk of the backend (opt_mbuf, 1 GiB): the host
// fastrpc_mmap with FASTRPC_MAP_FD_DELAYED one time, then "iters" times the HAP_mmap2 and the
// HAP_munmap2 on the DSP, as prep_op_bufs of htp/main.c does for a batch that does not reuse a
// mapping. The first map is on its own line, because it can also fill the page tables.
static int mode_mapcost(const struct options * o) {
    const size_t size = (size_t) o->mb << 20;
    double *     us_map   = (double *) malloc(sizeof(double) * o->iters);
    double *     us_unmap = (double *) malloc(sizeof(double) * o->iters);
    double *     us_call  = (double *) malloc(sizeof(double) * o->iters);
    if (us_map == NULL || us_unmap == NULL || us_call == NULL) {
        free(us_map);
        free(us_unmap);
        free(us_call);
        return EXIT_RPC;
    }
    int            rc   = 0;
    const uint64_t t0   = cntvct();
    uint8_t *      data = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
    const uint64_t t1   = cntvct();
    const int      fd   = data != NULL ? rpcmem_to_fd(data) : -1;
    if (fd < 0) {
        fprintf(stderr, "ddrbw: error: rpcmem_alloc2 of %zu MiB failed. Close apps and run again.\n", size >> 20);
        rc = EXIT_RPC;
    }
    uint64_t t2 = t1;
    uint64_t t3 = t1;
    if (rc == 0) {
        t2 = cntvct();
        const int e = fastrpc_mmap(CDSP_DOMAIN_ID, fd, data, 0, size, FASTRPC_MAP_FD_DELAYED);
        t3          = cntvct();
        if (e != 0) {
            fprintf(stderr, "ddrbw: error: fastrpc_mmap of %zu MiB failed with 0x%08x\n", size >> 20, (unsigned) e);
            rc = EXIT_RPC;
        }
    }
    size_t n = 0;
    double first_map = 0, first_unmap = 0, first_call = 0;
    for (uint32_t i = 0; rc == 0 && i <= o->iters; i++) {
        uint64         tk_map = 0, tk_unmap = 0;
        const uint64_t h0     = cntvct();
        int            e      = ddrbw_map(g_h, fd, (uint32_t) size, &tk_map);
        const uint64_t h1     = cntvct();
        if (e == AEE_SUCCESS) {
            e = ddrbw_unmap(g_h, fd, &tk_unmap);
        }
        if (e != AEE_SUCCESS) {
            fprintf(stderr, "ddrbw: error: map %u of %zu MiB failed with 0x%08x\n", i, size >> 20, (unsigned) e);
            rc = EXIT_RPC;
            break;
        }
        const double m = (double) tk_map / 19.2;
        const double u = (double) tk_unmap / 19.2;
        const double c = ticks_to_s(h1 - h0) * 1e6;
        if (i == 0) {
            first_map = m, first_unmap = u, first_call = c;
        } else {
            us_map[n] = m, us_unmap[n] = u, us_call[n] = c;
            n++;
        }
    }
    uint64_t t4 = cntvct();
    if (fd >= 0 && t3 != t1) {
        fastrpc_munmap(CDSP_DOMAIN_ID, fd, data, size);
    }
    const uint64_t t5 = cntvct();
    if (data != NULL) {
        rpcmem_free(data);
    }
    if (rc == 0) {
        printf("ddrbw: mapcost votes=%s mb=%zu alloc_ms=%.2f fastrpc_mmap_ms=%.3f fastrpc_munmap_ms=%.3f "
               "first_hap_mmap_us=%.1f first_hap_munmap_us=%.1f first_call_us=%.1f\n",
               g_votes_name, size >> 20, ticks_to_s(t1 - t0) * 1e3, ticks_to_s(t3 - t2) * 1e3,
               ticks_to_s(t5 - t4) * 1e3, first_map, first_unmap, first_call);
        char f[64];
        snprintf(f, sizeof(f), "kind=hap_mmap2 mb=%zu", size >> 20);
        print_dist_kind("mapcost", f, us_map, n);
        snprintf(f, sizeof(f), "kind=hap_munmap2 mb=%zu", size >> 20);
        print_dist_kind("mapcost", f, us_unmap, n);
        snprintf(f, sizeof(f), "kind=map_call mb=%zu", size >> 20);
        print_dist_kind("mapcost", f, us_call, n);
    }
    free(us_map);
    free(us_unmap);
    free(us_call);
    return rc;
}

// ---- the clock nodes of the shell

#define MAX_NODES 48

// The frequency nodes that a sample reads: cur_freq of each devfreq device and of each bus_dcvs
// device, the clock of the GPU, and the clocks of the two CPU clusters.
static int find_nodes(char paths[][160], int max) {
    static const char * const roots[] = { "/sys/class/devfreq", "/sys/devices/system/cpu/bus_dcvs" };
    static const char * const fixed[] = { "/sys/class/kgsl/kgsl-3d0/gpuclk",
                                          "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq",
                                          "/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq" };
    int n = 0;
    for (size_t r = 0; r < sizeof(roots) / sizeof(roots[0]); r++) {
        DIR * d = opendir(roots[r]);
        if (d == NULL) {
            continue;
        }
        struct dirent * e;
        while ((e = readdir(d)) != NULL && n < max) {
            if (e->d_name[0] == '.') {
                continue;
            }
            snprintf(paths[n], 160, "%s/%s/cur_freq", roots[r], e->d_name);
            if (access(paths[n], R_OK) == 0) {
                n++;
            }
        }
        closedir(d);
    }
    for (size_t i = 0; i < sizeof(fixed) / sizeof(fixed[0]) && n < max; i++) {
        if (access(fixed[i], R_OK) == 0) {
            snprintf(paths[n++], 160, "%s", fixed[i]);
        }
    }
    return n;
}

// Print the first line of a file (at most 200 characters), or its error.
static void print_file(const char * path) {
    FILE * f = fopen(path, "r");
    if (f == NULL) {
        printf("ddrbw: node path=%s readable=0 error=%s\n", path, strerror(errno));
        return;
    }
    char line[256] = "";
    if (fgets(line, sizeof(line), f) == NULL) {
        line[0] = '\0';
    }
    fclose(f);
    line[strcspn(line, "\n")] = '\0';
    line[200]                 = '\0';
    printf("ddrbw: node path=%s readable=1 value=%s\n", path, line);
}

// List the nodes of each devfreq device and each bus_dcvs device (two levels) that hold "freq",
// "governor" or "trans_stat" in the name, with the first line of each one.
static void list_dir(const char * dir, int depth) {
    DIR * d = opendir(dir);
    if (d == NULL) {
        printf("ddrbw: node path=%s readable=0 error=%s\n", dir, strerror(errno));
        return;
    }
    struct dirent * e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') {
            continue;
        }
        char path[512];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(path, &st) != 0) {
            continue;
        }
        if (S_ISDIR(st.st_mode)) {
            if (depth > 0) {
                list_dir(path, depth - 1);
            }
        } else if (strstr(e->d_name, "freq") != NULL || strstr(e->d_name, "governor") != NULL ||
                   strstr(e->d_name, "trans_stat") != NULL || strstr(e->d_name, "bw") != NULL) {
            print_file(path);
        }
    }
    closedir(d);
}

static int mode_freqlist(void) {
    list_dir("/sys/class/devfreq", 1);
    list_dir("/sys/devices/system/cpu/bus_dcvs", 2);
    print_file("/sys/class/kgsl/kgsl-3d0/gpuclk");
    print_file("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq");
    print_file("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq");
    char paths[MAX_NODES][160];
    const int n = find_nodes(paths, MAX_NODES);
    printf("ddrbw: freqnodes n=%d", n);
    for (int i = 0; i < n; i++) {
        printf(" %s", paths[i]);
    }
    printf("\n");
    return 0;
}

static volatile sig_atomic_t g_stop;

static void on_term(int sig) {
    (void) sig;
    g_stop = 1;
}

// The CLOCK_MONOTONIC time in microseconds, the clock of the log time stamps of memprobe --log-ts.
static int64_t mono_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

// Print the two clocks of the phone at one time, to put the log of another tool on the time axis.
static int mode_now(void) {
    const uint64_t t = cntvct();
    const int64_t  m = mono_us();
    printf("ddrbw: now cntvct=%" PRIu64 " mono_us=%" PRId64 "\n", t, m);
    return 0;
}

// Sample the frequency nodes "hz" times per second for "ms" milliseconds or until SIGTERM, and
// write one line "cntvct mono_us v1 v2 ..." when a value changes, and at least each 100 ms. The
// file starts with the paths. A sample reads each node with pread, thus it starts no process.
static int mode_freq(const struct options * o) {
    char      paths[MAX_NODES][160];
    const int n = find_nodes(paths, MAX_NODES);
    FILE *    out = fopen(o->out, "w");
    if (out == NULL) {
        fprintf(stderr, "ddrbw: error: cannot write %s: %s\n", o->out, strerror(errno));
        return EXIT_USAGE;
    }
    int fds[MAX_NODES];
    fprintf(out, "# cntfrq %" PRIu64 " hz %u nodes %d columns cntvct mono_us node0..\n", g_cntfrq, o->hz, n);
    for (int i = 0; i < n; i++) {
        fds[i] = open(paths[i], O_RDONLY);
        fprintf(out, "# node %d %s%s\n", i, paths[i], fds[i] < 0 ? " (not readable)" : "");
    }
    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);
    long long      last[MAX_NODES];
    long long      cur[MAX_NODES];
    const uint64_t period = g_cntfrq / o->hz;
    const uint64_t t_end  = cntvct() + (uint64_t) o->ms * (g_cntfrq / 1000u);
    uint64_t       t_last = 0;
    uint64_t       next   = cntvct();
    uint64_t       lines  = 0;
    memset(last, 0xff, sizeof(last));
    while (!g_stop && cntvct() < t_end) {
        const uint64_t t       = cntvct();
        int            changed = 0;
        for (int i = 0; i < n; i++) {
            char          buf[32];
            const ssize_t r = fds[i] >= 0 ? pread(fds[i], buf, sizeof(buf) - 1, 0) : -1;
            if (r > 0) {
                buf[r] = '\0';
                cur[i] = strtoll(buf, NULL, 10);
            } else {
                cur[i] = -1;
            }
            changed |= cur[i] != last[i];
        }
        if (changed || t - t_last >= g_cntfrq / 10u) {
            fprintf(out, "%" PRIu64 " %" PRId64, t, mono_us());
            for (int i = 0; i < n; i++) {
                fprintf(out, " %lld", cur[i]);
                last[i] = cur[i];
            }
            fprintf(out, "\n");
            t_last = t;
            lines++;
        }
        next += period;
        const uint64_t now = cntvct();
        if (next > now) {
            const uint64_t  ns = (next - now) * 1000000000ull / g_cntfrq;
            struct timespec ts = { (time_t) (ns / 1000000000ull), (long) (ns % 1000000000ull) };
            nanosleep(&ts, NULL);
        } else {
            next = now;
        }
    }
    fprintf(out, "# end lines %" PRIu64 "\n", lines);
    fclose(out);
    for (int i = 0; i < n; i++) {
        if (fds[i] >= 0) {
            close(fds[i]);
        }
    }
    printf("ddrbw: freq out=%s nodes=%d lines=%" PRIu64 "\n", o->out, n, lines);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
            "usage: ddrbw info --votes NAME\n"
            "       ddrbw nsp  --votes NAME [--plan full|short|ws|LIST] [--mb N] [--reps N] [--budget-ms N]\n"
            "       ddrbw cpu  [--sets LIST] [--mb N] [--reps N] [--budget-ms N]\n"
            "       ddrbw both --votes NAME [--cfgs LIST] [--sets LIST] [--mb N] [--reps N] [--budget-ms N]\n"
            "       ddrbw rtt  --votes NAME [--iters N]\n"
            "       ddrbw mapcost --votes NAME [--mb N] [--iters N]\n"
            "       ddrbw freqlist\n"
            "       ddrbw now\n"
            "       ddrbw freq --out FILE [--hz N] [--ms N]\n"
            "votes: none backend ddrperf busperf expv ceng bw max\n");
}

static int parse_votes(const char * name, uint32_t * mask) {
    static const struct {
        const char * name;
        uint32_t     mask;
    } k[] = {
        { "none", 0 },
        { "backend", DDRBW_VOTE_BACKEND },
        { "ddrperf", DDRBW_VOTE_BACKEND | DDRBW_VOTE_DDRPERF },
        { "busperf", DDRBW_VOTE_BACKEND | DDRBW_VOTE_BUSPERF },
        { "expv", DDRBW_VOTE_BACKEND | DDRBW_VOTE_EXPV },
        { "ceng", DDRBW_VOTE_BACKEND | DDRBW_VOTE_CENG },
        { "bw", DDRBW_VOTE_BACKEND | DDRBW_VOTE_BW },
        { "max", DDRBW_VOTE_ALL },
    };
    for (size_t i = 0; i < sizeof(k) / sizeof(k[0]); i++) {
        if (strcmp(name, k[i].name) == 0) {
            *mask = k[i].mask;
            return 0;
        }
    }
    fprintf(stderr, "ddrbw: error: \"%s\" is not a vote name\n", name);
    return -1;
}

static int parse_u32(const char * s, uint32_t * v, uint32_t lo, uint32_t hi) {
    char *              e;
    const unsigned long x = strtoul(s, &e, 10);
    if (*s == '\0' || *e != '\0' || x < lo || x > hi) {
        fprintf(stderr, "ddrbw: error: \"%s\" is not a number from %u to %u\n", s, lo, hi);
        return -1;
    }
    *v = (uint32_t) x;
    return 0;
}

int main(int argc, char ** argv) {
    struct options o = { NULL, "backend", "full", "gemv-t6,dma-t6", "6,7/0-5/0-7", 512, 3, 300, 2000, 10, 30000, NULL };
    if (argc < 2) {
        usage();
        return EXIT_USAGE;
    }
    o.mode = argv[1];
    for (int i = 2; i < argc; i += 2) {
        const char * a = argv[i];
        const char * v = i + 1 < argc ? argv[i + 1] : NULL;
        int          bad = v == NULL;
        if (!bad && strcmp(a, "--votes") == 0) {
            o.votes = v;
        } else if (!bad && strcmp(a, "--plan") == 0) {
            o.plan = v;
        } else if (!bad && strcmp(a, "--cfgs") == 0) {
            o.cfgs = v;
        } else if (!bad && strcmp(a, "--sets") == 0) {
            o.sets = v;
        } else if (!bad && strcmp(a, "--mb") == 0) {
            bad = parse_u32(v, &o.mb, 64, 2048) != 0;
        } else if (!bad && strcmp(a, "--reps") == 0) {
            bad = parse_u32(v, &o.reps, 1, 100) != 0;
        } else if (!bad && strcmp(a, "--budget-ms") == 0) {
            bad = parse_u32(v, &o.budget_ms, 10, 5000) != 0;
        } else if (!bad && strcmp(a, "--iters") == 0) {
            bad = parse_u32(v, &o.iters, 10, 100000) != 0;
        } else if (!bad && strcmp(a, "--hz") == 0) {
            bad = parse_u32(v, &o.hz, 1, 5000) != 0;
        } else if (!bad && strcmp(a, "--ms") == 0) {
            bad = parse_u32(v, &o.ms, 10, 600000) != 0;
        } else if (!bad && strcmp(a, "--out") == 0) {
            o.out = v;
        } else {
            bad = 1;
        }
        if (bad) {
            usage();
            return EXIT_USAGE;
        }
    }
    g_cntfrq = read_cntfrq();
    printf("ddrbw: host mode=%s cntfrq=%" PRIu64 " cpus=%ld mb=%u reps=%u budget_ms=%u\n", o.mode, g_cntfrq,
           sysconf(_SC_NPROCESSORS_CONF), o.mb, o.reps, o.budget_ms);
    fflush(stdout);
    if (g_cntfrq == 0) {
        fprintf(stderr, "ddrbw: error: CNTFRQ_EL0 is 0\n");
        return EXIT_RPC;
    }
    if (strcmp(o.mode, "cpu") == 0) {
        return mode_cpu(&o);
    }
    if (strcmp(o.mode, "freqlist") == 0) {
        return mode_freqlist();
    }
    if (strcmp(o.mode, "now") == 0) {
        return mode_now();
    }
    if (strcmp(o.mode, "freq") == 0) {
        if (o.out == NULL) {
            usage();
            return EXIT_USAGE;
        }
        return mode_freq(&o);
    }
    if (strcmp(o.mode, "info") != 0 && strcmp(o.mode, "nsp") != 0 && strcmp(o.mode, "both") != 0 &&
        strcmp(o.mode, "rtt") != 0 && strcmp(o.mode, "mapcost") != 0) {
        usage();
        return EXIT_USAGE;
    }
    if (parse_votes(o.votes, &g_votes) != 0) {
        return EXIT_USAGE;
    }
    g_votes_name = o.votes;
    int rc       = dsp_open();
    if (rc == 0) {
        rc = dsp_setup();
    }
    if (rc == 0) {
        if (strcmp(o.mode, "nsp") == 0) {
            rc = mode_nsp(&o);
        } else if (strcmp(o.mode, "both") == 0) {
            rc = mode_both(&o);
        } else if (strcmp(o.mode, "rtt") == 0) {
            rc = mode_rtt(&o);
        } else if (strcmp(o.mode, "mapcost") == 0) {
            rc = mode_mapcost(&o);
        }
    }
    dsp_close();
    return rc;
}
