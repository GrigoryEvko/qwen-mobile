// Target fa2: the tile-layout softmax of the HMX flash attention (flash-attn-ops.c).
//
// The program includes the kernel file, thus it calls the functions of the kernel and not copies
// of them. It has three modes:
//
//   --mode exp   hvx_vec_exp2_neg_f16 against 2^-u in double for each f16 value u in [0, 32] and
//                for the special values, and against hvx_vec_exp2_f16 (the old softmax), plus the
//                cycles for each vector of the two functions.
//   --mode smx   one softmax step of a KV block (fa2_softmax_thread) against a reference in double
//                that uses the same f16 maximum. It also runs the old step (fa_softmax_impl) on the
//                same scores for the cycles. Arguments: --rows (the Q rows times the GQA factor,
//                a multiple of 2), --bc, --kv (the KV rows of the block, at most --bc), --g (the
//                GQA factor), --mask 0 (none), 1 (0 and -inf, a causal band) or 2 (values in
//                [-1, 0] and -inf), --threads, --iters, --old 0 (no old step), --check 0 (no
//                reference, for the timing runs).
//   --mode full  hmx_flash_attn_ext against a reference in double, for the old kernel and for the
//                fa2 kernel, in the functional mode of the simulator (the timing model does not
//                retire HMX instructions). Arguments: --tokens, --nkv, --heads, --kvheads, --dk,
//                --q8 0|1, --mask 0|1|2, --sinks 0|1 (--sink-lo, --sink-hi: the range of the sink
//                logits), --variant (the bits of HTP_FA_OPT_*), --check-stride, --vtcm-kb.
//
// The DMA of the kernel goes through a synchronous shim (refer to target_fa.c): a push copies at
// once and a pop gives the destinations in the order of the pushes. The HMX queue of the full mode
// is a synchronous shim too: a push runs the job on the calling thread.
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include "lab.h"

#include <math.h>
#include <hexagon_standalone.h>
#include <stdio.h>
#include <string.h>

// ---- The DMA shim (the interface of dma-queue.h) ----
#define HTP_DMA_H

typedef struct {
    void *       dst;
    const void * src;
} dma_ptr;

#define LAB_DMA_CAPACITY 256

typedef struct dma_queue_s {
    void *   dst[LAB_DMA_CAPACITY];
    uint32_t push_idx;
    uint32_t pop_idx;
} dma_queue;
typedef dma_queue * dma_queue_t;

static inline dma_ptr dma_make_ptr(void * dst, const void * src) {
    dma_ptr p = { dst, src };
    return p;
}

// The DMA engine of a hardware thread follows one descriptor chain: a push links its descriptor to the
// tail of its queue (dmlink in dma-queue.h). On the chip, a push to a queue while a different queue has
// transfers of the same thread in flight links to a chain that the engine does not read, thus the
// transfer never starts and its pop waits forever. The shim copies at once, thus it cannot hang. It
// counts each such push instead (lab_dma_ring_violations), for the calling hardware thread.
#define LAB_DMA_THREADS 32

static struct {
    unsigned int tid;
    dma_queue *  q;  // the last queue that this thread pushed to
} lab_dma_active[LAB_DMA_THREADS];
static uint32_t lab_dma_ring_violations;
static int      lab_dma_mutex;

static inline uint32_t lab_dma_in_flight(const dma_queue * q);

static void lab_dma_note_push(dma_queue * q) {
    const unsigned int tid = (unsigned int) thread_get_tnum();
    lockMutex(&lab_dma_mutex);
    int slot = -1;
    for (int i = 0; i < LAB_DMA_THREADS; i++) {
        if (lab_dma_active[i].q && lab_dma_active[i].tid == tid) {
            slot = i;
            break;
        }
        if (!lab_dma_active[i].q && slot < 0) {
            slot = i;
        }
    }
    if (slot >= 0) {
        dma_queue * last = lab_dma_active[slot].q;
        if (last && last != q && lab_dma_in_flight(last) > 0) {
            if (lab_dma_ring_violations++ == 0) {
                printf("lab: fa2 dma: thread %u pushes to queue %p while queue %p has %u transfers in flight\n", tid,
                       (void *) q, (void *) last, lab_dma_in_flight(last));
            }
        }
        lab_dma_active[slot].tid = tid;
        lab_dma_active[slot].q   = q;
    }
    unlockMutex(&lab_dma_mutex);
}

static inline bool dma_queue_push(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride, size_t row_size,
                                  size_t nrows) {
    lab_dma_note_push(q);
    for (size_t r = 0; r < nrows; r++) {
        memcpy((uint8_t *) p.dst + r * dst_stride, (const uint8_t *) p.src + r * src_stride, row_size);
    }
    q->dst[q->push_idx & (LAB_DMA_CAPACITY - 1)] = p.dst;
    q->push_idx++;
    return true;
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr p = { NULL, NULL };
    if (q->pop_idx == q->push_idx) {
        return p;
    }
    p.dst = q->dst[q->pop_idx & (LAB_DMA_CAPACITY - 1)];
    q->pop_idx++;
    return p;
}

static inline uint32_t lab_dma_in_flight(const dma_queue * q) {
    return q->push_idx - q->pop_idx;
}

#define DMA_CACHE_MAX_SIZE 128

// The line cache of the mask, with the replacement rule of dma-queue.h.
typedef struct {
    uint8_t * base;
    uint32_t  line_size;
    uint32_t  capacity;
    uint32_t  src[DMA_CACHE_MAX_SIZE];
    uint16_t  age[DMA_CACHE_MAX_SIZE];
} dma_cache;

static inline void dma_cache_init(dma_cache * c, uint8_t * base, uint32_t line_size, uint32_t capacity) {
    c->capacity  = (capacity > DMA_CACHE_MAX_SIZE) ? DMA_CACHE_MAX_SIZE : capacity;
    c->base      = base;
    c->line_size = line_size;
    for (unsigned i = 0; i < c->capacity; i++) {
        c->src[i] = 0;
        c->age[i] = 0;
    }
}

static inline bool dma_cache_push(dma_queue * q, dma_cache * c, const uint8_t * src, uint32_t dst_stride,
                                  uint32_t src_stride, uint32_t row_size, uint32_t nrows) {
    uint32_t  o_idx = 0;
    uint16_t  o_age = 0;
    uint8_t * dst   = 0;
    for (unsigned i = 0; i < c->capacity; i++) {
        if (c->src[i] == (uint32_t) (uintptr_t) src) {
            c->age[i] = 0;
            dst       = c->base + (i * c->line_size);
            nrows     = 0;
        } else {
            c->age[i]++;
            if (c->age[i] > o_age) {
                o_age = c->age[i];
                o_idx = i;
            }
        }
    }
    if (!dst) {
        c->age[o_idx] = 0;
        c->src[o_idx] = (uint32_t) (uintptr_t) src;
        dst           = c->base + o_idx * c->line_size;
    }
    return dma_queue_push(q, dma_make_ptr(dst, src), dst_stride, src_stride, row_size, nrows);
}

// ---- The HMX queue shim (the interface of hmx-queue.h): a push runs the job at once ----
#define HMX_QUEUE_H

typedef void (*hmx_queue_func)(void *);

struct hmx_queue_desc {
    hmx_queue_func func;
    void *         data;
};

struct hmx_queue_s {
    uint32_t pushed;
    uint32_t popped;
};

typedef struct hmx_queue_s * hmx_queue_t;

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    d.func(d.data);
    q->pushed++;
    return true;
}

static inline struct hmx_queue_desc hmx_queue_pop(hmx_queue_t q) {
    struct hmx_queue_desc d = { NULL, NULL };
    if (q->popped == q->pushed) {
        printf("lab: error: an HMX queue pop without a job\n");
        return d;
    }
    q->popped++;
    d.func = (hmx_queue_func) 1;
    return d;
}

#include "flash-attn-ops.c"

#define TARGET "fa2"

// ---- Helpers ----

static double hf_bits_to_double(uint16_t h) {
    return (double) lab_hf_to_f32(h);
}

// ---- Mode exp ----

#define EXP_NVEC 64  // vectors in the timed loop, thus 4096 values

static __attribute__((noinline)) void time_new_exp(HVX_Vector * restrict out, const HVX_Vector * restrict in) {
#pragma clang loop unroll_count(4)
    for (int i = 0; i < EXP_NVEC; i++) {
        out[i] = hvx_vec_exp2_neg_f16(in[i]);
    }
}

static __attribute__((noinline)) void time_old_exp(HVX_Vector * restrict out, const HVX_Vector * restrict in) {
    const HVX_Vector sign = Q6_Vh_vsplat_R(0x8000);
#pragma clang loop unroll_count(4)
    for (int i = 0; i < EXP_NVEC; i++) {
        out[i] = hvx_vec_exp2_f16(Q6_V_vxor_VV(in[i], sign));
    }
}

// The error classes of one exp function over all inputs. ulp is the f16 unit in the last place of
// the exact result, for results in the normal range.
struct exp_err {
    double   max_ulp;       // the largest error in ulp of the exact result, normal results
    uint16_t max_ulp_in;    // the input of that error
    uint32_t n_over_1ulp;   // normal results with an error of more than 1 ulp
    double   max_abs_small; // the largest absolute error for exact results below 2^-14
    uint32_t n_normal;
};

static void exp_account(struct exp_err * e, uint16_t in, double got, double want) {
    if (want >= ldexp(1.0, -14)) {
        const int    ex  = ilogb(want);
        const double ulp = ldexp(1.0, ex - 10);
        const double d   = fabs(got - want) / ulp;
        e->n_normal++;
        if (d > e->max_ulp) {
            e->max_ulp    = d;
            e->max_ulp_in = in;
        }
        if (d > 1.0) {
            e->n_over_1ulp++;
        }
    } else {
        const double d = fabs(got - want);
        if (d > e->max_abs_small) {
            e->max_abs_small = d;
        }
    }
}

static int mode_exp(int argc, char ** argv) {
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 5);

    HVX_Vector * in  = lab_vtcm_alloc(EXP_NVEC * 128, 128);
    HVX_Vector * out = lab_vtcm_alloc(EXP_NVEC * 128, 128);
    HVX_Vector * alt = lab_vtcm_alloc(EXP_NVEC * 128, 128);

    // The semantics of the two conversions that the new function uses.
    {
        static const float probe[8] = { 0.25f, 0.5f, 0.75f, 1.5f, 2.5f, 14.99f, 29.9f, 30.0f };
        __fp16 *           pi       = (__fp16 *) in;
        for (int i = 0; i < 64; i++) {
            pi[i] = (__fp16) probe[i % 8];
        }
        const HVX_Vector k = Q6_Vh_equals_Vhf(in[0]);
        int16_t          kk[64];
        memcpy(kk, &k, 128);
        printf("lab: fa2 Vh_equals_Vhf:");
        for (int i = 0; i < 8; i++) {
            printf(" %g->%d", probe[i], kk[i]);
        }
        printf("\n");
    }

    // Every f16 bit pattern from 0 to 32.0 (0x5000), then +Inf, -0, a large value and NaN.
    struct exp_err e_new = { 0 }, e_old = { 0 };
    uint32_t       n_diff = 0, n_vals = 0, n_special_bad = 0;
    uint16_t       u      = 0;
    bool           done   = false;
    while (!done) {
        uint16_t * pi = (uint16_t *) in;
        for (int i = 0; i < 64; i++) {
            pi[i] = u;
            if (u < 0x5000) {
                u++;
            } else {
                done = true;
            }
        }
        const HVX_Vector vn = hvx_vec_exp2_neg_f16(in[0]);
        const HVX_Vector vo = hvx_vec_exp2_f16(Q6_V_vxor_VV(in[0], Q6_Vh_vsplat_R(0x8000)));
        uint16_t         rn[64], ro[64];
        memcpy(rn, &vn, 128);
        memcpy(ro, &vo, 128);
        for (int i = 0; i < 64; i++) {
            const double x    = hf_bits_to_double(pi[i]);
            const double want = exp2(-x);
            exp_account(&e_new, pi[i], hf_bits_to_double(rn[i]), want);
            exp_account(&e_old, pi[i], hf_bits_to_double(ro[i]), want);
            n_diff += rn[i] != ro[i];
            n_vals++;
        }
    }
    {
        static const uint16_t special[6] = { 0x7c00, 0x8000, 0x7bff, 0x5800, 0x4f80, 0x4f7f };
        uint16_t *            pi         = (uint16_t *) in;
        for (int i = 0; i < 64; i++) {
            pi[i] = special[i % 6];
        }
        const HVX_Vector vn = hvx_vec_exp2_neg_f16(in[0]);
        uint16_t         rn[64];
        memcpy(rn, &vn, 128);
        for (int i = 0; i < 6; i++) {
            printf("lab: fa2 exp2_neg(0x%04x) = 0x%04x\n", special[i], rn[i]);
            // +Inf, the largest f16, 128 and 30 must give 0. -0 must give 1.0.
            if (i == 1 ? rn[i] != 0x3c00 : (i != 5 && rn[i] != 0)) {
                n_special_bad++;
            }
        }
    }

    lab_report(TARGET, "exp_values", n_vals, "");
    lab_report(TARGET, "exp_new_max_ulp", e_new.max_ulp, "ulp");
    lab_report(TARGET, "exp_new_max_ulp_input", hf_bits_to_double(e_new.max_ulp_in), "");
    lab_report(TARGET, "exp_new_over_1ulp", e_new.n_over_1ulp, "");
    lab_report(TARGET, "exp_new_max_abs_below_2^-14", e_new.max_abs_small, "");
    lab_report(TARGET, "exp_old_max_ulp", e_old.max_ulp, "ulp");
    lab_report(TARGET, "exp_old_max_ulp_input", hf_bits_to_double(e_old.max_ulp_in), "");
    lab_report(TARGET, "exp_old_over_1ulp", e_old.n_over_1ulp, "");
    lab_report(TARGET, "exp_old_max_abs_below_2^-14", e_old.max_abs_small, "");
    lab_report(TARGET, "exp_normal_results", e_new.n_normal, "");
    lab_report(TARGET, "exp_new_old_different", n_diff, "");
    lab_report(TARGET, "exp_special_wrong", n_special_bad, "");

    // The cycles for each vector, inputs in [0, 16).
    for (int i = 0; i < EXP_NVEC * 64; i++) {
        ((__fp16 *) in)[i] = (__fp16) lab_rand_f32(0.0f, 16.0f);
    }

    uint64_t best_new = UINT64_MAX, best_old = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        time_new_exp(out, in);
        uint64_t t1 = lab_cycles();
        best_new    = t1 - t0 < best_new ? t1 - t0 : best_new;
        LAB_BARRIER();
        t0 = lab_cycles();
        time_old_exp(alt, in);
        t1       = lab_cycles();
        best_old = t1 - t0 < best_old ? t1 - t0 : best_old;
    }
    lab_report(TARGET, "exp_new_cycles_per_vector", (double) best_new / EXP_NVEC, "cycles");
    lab_report(TARGET, "exp_old_cycles_per_vector", (double) best_old / EXP_NVEC, "cycles");
    return 0;
}

// ---- Mode smx ----

// The S tile value of row r and column c of a block with n_tiles_per_bc column tiles. A tile
// vector holds the rows 2v and 2v + 1, interleaved for each column.
static inline size_t tile_index(uint32_t r, uint32_t c, uint32_t n_tiles_per_bc) {
    const uint32_t t = r / 32, rr = r % 32, ct = c / 32, cc = c % 32;
    return ((size_t) t * n_tiles_per_bc + ct) * 1024 + (rr / 2) * 64 + cc * 2 + (rr % 2);
}

// A mask row for the query q of a block: 0 up to the causal limit, then -inf (mode 1), or a value
// in [-1, 0] up to the limit (mode 2). The limit moves with q, thus the rows differ.
static void fill_mask(__fp16 * m, uint32_t n_rows_q, uint32_t cols, size_t line, int mode, uint32_t kv_total,
                      uint32_t n_tokens) {
    for (uint32_t q = 0; q < n_rows_q; q++) {
        const uint32_t limit = kv_total - n_tokens + q;  // the last visible column
        for (uint32_t c = 0; c < cols; c++) {
            float v = c <= limit ? 0.0f : -INFINITY;
            if (mode == 2 && c <= limit) {
                v = lab_rand_f32(-1.0f, 0.0f);
            }
            m[q * line + c] = (__fp16) v;
        }
    }
}

static int mode_smx(int argc, char ** argv) {
    const uint32_t rows    = (uint32_t) lab_arg_long(argc, argv, "--rows", 320);
    const uint32_t Bc      = (uint32_t) lab_arg_long(argc, argv, "--bc", 1024);
    const uint32_t kv_rows = (uint32_t) lab_arg_long(argc, argv, "--kv", Bc);
    const uint32_t G       = (uint32_t) lab_arg_long(argc, argv, "--g", 4);
    const int      mmode   = (int) lab_arg_long(argc, argv, "--mask", 1);
    const uint32_t n_thr   = (uint32_t) lab_arg_long(argc, argv, "--threads", 1);
    const uint32_t iters   = (uint32_t) lab_arg_long(argc, argv, "--iters", 3);
    const int      old     = (int) lab_arg_long(argc, argv, "--old", 1);
    const int      check   = (int) lab_arg_long(argc, argv, "--check", 1);

    if (rows % 2 || Bc % 64 || kv_rows > Bc || kv_rows == 0 || rows % G) {
        printf("lab: fa2 smx: --rows even and a multiple of --g, --bc a multiple of 64, 0 < --kv <= --bc\n");
        return 2;
    }
    const uint32_t g_br        = (rows + 31) / 32 * 32;
    const uint32_t n_row_tiles = g_br / 32;
    const uint32_t n_tpb       = Bc / 32;
    const uint32_t n_rows_q    = rows / G;
    const size_t   tiles_elms  = (size_t) n_row_tiles * n_tpb * 1024;
    const size_t   line        = (Bc * 2 + 127) / 128 * 64;  // f16 elements of one mask row

    static struct htp_context     ctx;
    static struct htp_ops_context octx;
    static struct hmx_fa_context  factx;
    memset(&ctx, 0, sizeof(ctx));
    memset(&octx, 0, sizeof(octx));
    memset(&factx, 0, sizeof(factx));
    ctx.n_threads     = n_thr;
    ctx.n_threads_div = init_fastdiv_values(n_thr);
    octx.ctx          = &ctx;
    factx.octx        = &octx;
    factx.n_threads   = n_thr;
    factx.G           = G;
    factx.div_G       = init_fastdiv_values(G);
    factx.Bc          = Bc;
    factx.g_br        = g_br;
    factx.mask_broadcast      = true;
    factx.mask_buf_row_stride = line;
    factx.row_buf_stride      = (Bc * 2 + 255) / 256 * 2;

    __fp16 * s0     = lab_vtcm_alloc(tiles_elms * 2, 2048);  // the scores, kept unchanged
    __fp16 * s_old  = lab_vtcm_alloc(tiles_elms * 2, 2048);
    __fp16 * s_new  = lab_vtcm_alloc(tiles_elms * 2, 2048);
    __fp16 * p_old  = lab_vtcm_alloc(tiles_elms * 2, 2048);
    __fp16 * p_new  = lab_vtcm_alloc(tiles_elms * 2, 2048);
    __fp16 * d_old  = lab_vtcm_alloc((size_t) n_row_tiles * 2048, 2048);
    __fp16 * d_new  = lab_vtcm_alloc((size_t) n_row_tiles * 2048, 2048);
    __fp16 * mask   = mmode ? lab_vtcm_alloc((size_t) n_rows_q * line * 2, 128) : NULL;
    HVX_Vector * m_old = lab_vtcm_alloc((size_t) g_br * 4 + 256, 256);
    HVX_Vector * l_old = lab_vtcm_alloc((size_t) g_br * 4 + 256, 256);
    HVX_Vector * m_new = lab_vtcm_alloc((size_t) n_row_tiles * 128, 128);
    HVX_Vector * rbufs = lab_vtcm_alloc((size_t) factx.row_buf_stride * 128 * 2 * n_thr, 128);

    // Scores in log2 units, as the HMX gives them (scale times log2(e) included): a normal
    // spread with a few large values, thus the maximum of a row is not always at the start.
    for (uint32_t r = 0; r < g_br; r++) {
        for (uint32_t c = 0; c < Bc; c++) {
            float s = lab_rand_f32(-6.0f, 6.0f);
            if ((lab_rand_u32() & 63) == 0) {
                s += 8.0f;
            }
            s0[tile_index(r, c, n_tpb)] = (__fp16) s;
        }
    }
    if (mask) {
        fill_mask(mask, n_rows_q, Bc, line, mmode, kv_rows + 64, kv_rows + 64);
    }

    fa2_softmax_args_t na;
    memset(&na, 0, sizeof(na));
    na.factx          = &factx;
    na.s_tiles        = s_new;
    na.p_tiles        = p_new;
    na.d_tiles        = d_new;
    na.mask           = mask;
    na.mask_rows      = n_rows_q;
    na.kv_rows        = kv_rows;
    na.n_tiles_per_bc = n_tpb;
    na.n_rows_g       = rows;

    fa_softmax_args_t oa;
    memset(&oa, 0, sizeof(oa));
    oa.factx                = &factx;
    oa.buf_idx              = 0;
    oa.kv_rows              = kv_rows;
    oa.n_rows_g             = rows;
    oa.n_col_tiles          = (kv_rows + 31) / 32;
    oa.n_tiles_per_bc       = n_tpb;
    oa.n_row_tiles          = n_row_tiles;
    oa.n_row_tiles_g_br     = n_row_tiles;
    oa.Bc                   = Bc;
    oa.G                    = G;
    oa.mask                 = NULL;
    oa.mask_vtcm            = mask;
    oa.mask_vtcm_row_stride = line;

    // fa_softmax_impl reads a mask tensor only for its ne[3]; one broadcast mask row block.
    static struct htp_tensor tmask;
    memset(&tmask, 0, sizeof(tmask));
    tmask.ne[0] = Bc;
    tmask.ne[1] = n_rows_q;
    tmask.ne[2] = 1;
    tmask.ne[3] = 1;
    factx.src3_div3 = init_fastdiv_values(1);
    if (mask) {
        oa.mask = &tmask;
    }

    if (mask) {
        printf("lab: fa2 smx: the mask is binary for all rows: %d\n", (int) fa2_mask_is_binary(&na, 0, rows / 2));
    }

    uint64_t best_new = UINT64_MAX, best_old = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        // the new softmax
        memcpy(s_new, s0, tiles_elms * 2);
        hvx_splat_u8_a(d_new, 0, n_row_tiles * 2048);
        for (uint32_t t = 0; t < n_row_tiles; t++) {
            m_new[t] = hvx_vec_splat_f16(HTP_FA_M_INITIAL_VAL);
        }
        factx.vtcm_m_vec = m_new;
        LAB_BARRIER();
        uint64_t t0 = lab_cycles();
        fa2_phase_softmax(&factx, &na);
        uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        best_new = t1 - t0 < best_new ? t1 - t0 : best_new;

        if (!old) {
            continue;
        }
        memcpy(s_old, s0, tiles_elms * 2);
        hvx_splat_u8_a(d_old, 0, n_row_tiles * 2048);
        hvx_splat_f32_a(m_old, HTP_FA_M_INITIAL_VAL, g_br);
        hvx_splat_u8_a(l_old, 0, g_br * 4);
        factx.vtcm_s_tiles[0] = s_old;
        factx.vtcm_p_tiles[0] = p_old;
        factx.vtcm_d_tiles[0] = d_old;
        factx.vtcm_m_vec      = m_old;
        factx.vtcm_l_vec      = l_old;
        factx.vtcm_row_bufs   = rbufs;
        LAB_BARRIER();
        t0 = lab_cycles();
        fa_phase_softmax_and_build_d(&factx, &oa, n_row_tiles, n_row_tiles);
        t1 = lab_cycles();
        LAB_BARRIER();
        best_old = t1 - t0 < best_old ? t1 - t0 : best_old;
    }

    if (!check) {
        const double vecs = (double) rows / 2 * ((kv_rows + 31) / 32);
        lab_report(TARGET, "smx_threads", n_thr, "");
        lab_report(TARGET, "smx_new_cycles", (double) best_new, "cycles");
        lab_report(TARGET, "smx_new_cycles_per_tile_vector", (double) best_new * n_thr / vecs, "cycles");
        if (old) {
            lab_report(TARGET, "smx_old_cycles", (double) best_old, "cycles");
            lab_report(TARGET, "smx_old_over_new", (double) best_old / (double) best_new, "x");
        }
        return 0;
    }

    // The check of P against 2^(s' - max s') in double, with s' the masked score. The maximum of
    // the new kernel is the f16 maximum of the row, which is the exact maximum of the f16 scores.
    const double log2e = 1.4426950408889634;
    double   max_rel = 0.0, sum_rel = 0.0, max_abs_small = 0.0, max_rel_old = 0.0;
    uint32_t n_rel = 0, n_bad = 0, n_nan = 0, m_bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t q = r / G;
        double         mx = -1e30;
        for (uint32_t c = 0; c < kv_rows; c++) {
            double s = (double) s0[tile_index(r, c, n_tpb)];
            if (mask) {
                const double mv = (double) mask[q * line + c];
                s = mv > -16.0 ? s + (double) (__fp16) ((__fp16) mv * (__fp16) log2e) : -INFINITY;
            }
            if (s > mx) {
                mx = s;
            }
        }
        const double m_got = (double) ((const __fp16 *) m_new)[(r / 32) * 64 + r % 32];
        if (fabs(m_got - mx) > 0.02 * (fabs(mx) + 1.0) && mx > -1e29) {
            if (m_bad < 5) {
                printf("lab: fa2 smx row %u: m %g, expected %g\n", r, m_got, mx);
            }
            m_bad++;
        }
        for (uint32_t c = 0; c < Bc; c++) {
            double s = (double) s0[tile_index(r, c, n_tpb)];
            bool   masked = c >= kv_rows;
            if (mask && !masked) {
                const double mv = (double) mask[q * line + c];
                if (mv > -16.0) {
                    s += (double) (__fp16) ((__fp16) mv * (__fp16) log2e);
                } else {
                    masked = true;
                }
            }
            const double want = masked ? 0.0 : exp2(s - m_got);
            const double got  = (double) p_new[tile_index(r, c, n_tpb)];
            const double gold = (double) p_old[tile_index(r, c, n_tpb)];
            if (got != got) {
                n_nan++;
                continue;
            }
            if (c >= ((kv_rows + 31) / 32) * 32) {
                continue;  // the kernel does not compute the tiles after the last column tile
            }
            if (want >= ldexp(1.0, -14)) {
                const double rel = fabs(got - want) / want;
                sum_rel += rel;
                n_rel++;
                if (rel > max_rel) {
                    max_rel = rel;
                }
                // The rounding of u = m - s to f16 gives half an f16 unit of u in the exponent,
                // and the qf16 polynomial about 4 units of the result.
                const double u   = m_got - s;
                const double tol = (u >= 1.0 ? ldexp(1.0, ilogb(u) - 11) : ldexp(1.0, -12)) * 0.6931 + 2.5e-3;
                if (rel > tol) {
                    if (n_bad < 5) {
                        printf("lab: fa2 smx P row %u col %u: %g, expected %g\n", r, c, got, want);
                    }
                    n_bad++;
                }
                if (old && !masked) {
                    const double rel_old = fabs(gold - exp2(s - mx)) / exp2(s - mx);
                    if (rel_old > max_rel_old) {
                        max_rel_old = rel_old;
                    }
                }
            } else if (fabs(got - want) > max_abs_small) {
                max_abs_small = fabs(got - want);
            }
        }
    }

    // D: 2^(m_prev - m) with m_prev = -10000, thus 0 for each row.
    uint32_t d_bad = 0;
    for (uint32_t r = 0; r < rows; r++) {
        const __fp16 d = d_new[(r / 32) * 1024 + ((r % 32) / 2) * 64 + (r % 32) * 2 + (r % 2)];
        d_bad += d != 0;
    }

    const double vecs = (double) rows / 2 * ((kv_rows + 31) / 32);
    lab_report(TARGET, "smx_rows", rows, "");
    lab_report(TARGET, "smx_kv_rows", kv_rows, "");
    lab_report(TARGET, "smx_threads", n_thr, "");
    lab_report(TARGET, "smx_new_cycles", (double) best_new, "cycles");
    lab_report(TARGET, "smx_new_cycles_per_tile_vector", (double) best_new * n_thr / vecs, "cycles");
    if (old) {
        lab_report(TARGET, "smx_old_cycles", (double) best_old, "cycles");
        lab_report(TARGET, "smx_old_over_new", (double) best_old / (double) best_new, "x");
        lab_report(TARGET, "smx_old_max_rel", max_rel_old, "");
    }
    lab_report(TARGET, "smx_new_max_rel", max_rel, "");
    lab_report(TARGET, "smx_new_mean_rel", n_rel ? sum_rel / n_rel : 0.0, "");
    lab_report(TARGET, "smx_new_max_abs_small", max_abs_small, "");
    lab_report(TARGET, "smx_new_p_bad", n_bad, "");
    lab_report(TARGET, "smx_new_p_nan", n_nan, "");
    lab_report(TARGET, "smx_new_m_bad", m_bad, "");
    lab_report(TARGET, "smx_new_d_bad", d_bad, "");
    return (n_bad || n_nan || m_bad || d_bad) ? 1 : 0;
}

// ---- Mode full ----

// Quantize one row to Q8_0 as quantize_row_q8_0_ref does: d = amax / 127, q = round(x / d).
static void quantize_q8_0_row(const float * x, block_q8_0 * y, uint32_t n) {
    for (uint32_t b = 0; b < n / 32; b++) {
        float amax = 0.0f;
        for (uint32_t j = 0; j < 32; j++) {
            const float a = fabsf(x[b * 32 + j]);
            amax          = a > amax ? a : amax;
        }
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[b].d         = lab_f32_to_hf(d);  // ggml_half holds the f16 bits
        for (uint32_t j = 0; j < 32; j++) {
            y[b].qs[j] = (int8_t) roundf(x[b * 32 + j] * id);
        }
    }
}

// The value of element j of a K or V row as the kernel reads it.
static double kv_value(const uint8_t * row, uint32_t j, bool q8) {
    if (q8) {
        const block_q8_0 * b = (const block_q8_0 *) row + j / 32;
        return (double) lab_hf_to_f32(b->d) * (double) b->qs[j % 32];
    }
    return (double) ((const __fp16 *) row)[j];
}

// Build the kernel params as ggml_hexagon_precompute_flash_attn_params does for the HMX kernels.
// variant (the bits of HTP_FA_OPT_*): 0 the kernel HTP_FA_KERNEL_HMX with hmx_fa_find_chunk_size, 1 the
// kernel HTP_FA_KERNEL_HMX2 for the prefill shapes, 2 (with at most 32 Q rows times the GQA factor)
// HTP_FA_KERNEL_HMX2 in KV spans, 3 both. --vtcm-kb makes the spans smaller.
static bool full_params(struct htp_fa_kernel_params * kp, int variant, uint32_t G, uint32_t DK, uint32_t n_tokens,
                        uint32_t n_kv, uint32_t n_thr, size_t vtcm, bool q8, bool has_mask, float scale) {
    memset(kp, 0, sizeof(*kp));
    kp->scale       = scale;
    kp->is_q_fp32   = 1;
    kp->is_dst_fp32 = 1;
    kp->G           = G;
    kp->n_head_log2 = 1;
    kp->m0 = kp->m1 = 1.0f;
    size_t Br = 0, Bc = 0;
    if ((variant & HTP_FA_OPT_DECODE) && n_tokens * G <= 32) {
        if (hmx_fa2_find_span_size(&Br, &Bc, G, DK, DK, n_tokens, n_kv, vtcm, n_thr, true) != 0) {
            return false;
        }
        kp->kernel_type = HTP_FA_KERNEL_HMX2;
        kp->n_threads   = n_thr;
        kp->u.hmx.spans = 1;
        kp->vtcm_size   = hmx_fa2_compute_vtcm_usage(G, DK, DK, Br, Bc, n_thr, true, true);
    } else if ((variant & HTP_FA_OPT_TILE_SMX) && n_tokens * G >= 64) {
        if (hmx_fa2_find_chunk_size(&Br, &Bc, G, DK, DK, n_tokens, n_kv, vtcm, n_thr, true, q8) != 0) {
            return false;
        }
        kp->kernel_type = HTP_FA_KERNEL_HMX2;
        kp->n_threads   = n_thr;
        kp->vtcm_size   = hmx_fa2_compute_vtcm_usage(G, DK, DK, Br, Bc, n_thr, false, true);
    } else {
        if (hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DK, n_tokens, n_kv, vtcm, n_thr, true) != 0) {
            return false;
        }
        kp->kernel_type    = HTP_FA_KERNEL_HMX;
        const uint32_t nkb = (n_kv + Bc - 1) / Bc;
        kp->n_threads      = (nkb >= 3 && n_thr >= 2) ? n_thr : 1;
        kp->u.hmx.pipeline = (nkb >= 3 && n_thr >= 2) ? 1 : 0;
        kp->vtcm_size      = hmx_fa_compute_vtcm_usage(G, DK, DK, Br, Bc, kp->n_threads, kp->u.hmx.pipeline, true);
    }
    kp->Br                   = Br;
    kp->Bc                   = Bc;
    kp->n_kv_blocks          = (n_kv + Bc - 1) / Bc;
    kp->u.hmx.g_br           = (G * Br + 31) / 32 * 32;
    kp->u.hmx.mask_broadcast = has_mask;
    kp->u.hmx.div_G          = init_fastdiv_values(G);
    kp->src3_div2            = init_fastdiv_values(1);
    kp->src3_div3            = init_fastdiv_values(1);
    kp->broadcast_rk3        = init_fastdiv_values(1);
    kp->broadcast_rv3        = init_fastdiv_values(1);
    return true;
}

static int mode_full(int argc, char ** argv) {
    const uint32_t n_tokens = (uint32_t) lab_arg_long(argc, argv, "--tokens", 64);
    const uint32_t n_kv     = (uint32_t) lab_arg_long(argc, argv, "--nkv", 300);
    const uint32_t n_heads  = (uint32_t) lab_arg_long(argc, argv, "--heads", 16);
    const uint32_t n_kvh    = (uint32_t) lab_arg_long(argc, argv, "--kvheads", 4);
    const uint32_t DK       = (uint32_t) lab_arg_long(argc, argv, "--dk", 256);
    const bool     q8       = lab_arg_long(argc, argv, "--q8", 1) != 0;
    const int      mmode    = (int) lab_arg_long(argc, argv, "--mask", 1);
    const bool     use_snk  = lab_arg_long(argc, argv, "--sinks", 0) != 0;
    const int      variant  = (int) lab_arg_long(argc, argv, "--variant", 3);
    const uint32_t n_thr    = (uint32_t) lab_arg_long(argc, argv, "--threads", 6);
    const uint32_t stride   = (uint32_t) lab_arg_long(argc, argv, "--check-stride", 1);
    const size_t   vtcm_cap = (size_t) lab_arg_long(argc, argv, "--vtcm-kb", 8192) * 1024;
    const float    sink_lo  = (float) lab_arg_long(argc, argv, "--sink-lo", -2);
    const float    sink_hi  = (float) lab_arg_long(argc, argv, "--sink-hi", 6);
    const uint32_t G        = n_heads / n_kvh;
    const float    scale    = 1.0f / sqrtf((float) DK);

    if (n_heads % n_kvh || DK % 64) {
        printf("lab: fa2 full: --heads a multiple of --kvheads, --dk a multiple of 64\n");
        return 2;
    }

    // The layouts of the model: Q [DK, tokens, heads] f32 with the heads of one token together,
    // K and V [DK, n_kv, kv heads] with the heads of one KV row together, the mask [n_kv, tokens]
    // f16, dst [DK, heads, tokens] f32.
    const size_t k_row  = q8 ? DK / 32 * sizeof(block_q8_0) : DK * sizeof(__fp16);
    const size_t line   = (n_kv * 2 + 63) / 64 * 64;  // mask bytes of one row, padded as the model pads
    float *      qd     = lab_ddr_alloc((size_t) n_tokens * n_heads * DK * 4, 128);
    uint8_t *    kd     = lab_ddr_alloc((size_t) n_kv * n_kvh * k_row, 128);
    uint8_t *    vd     = lab_ddr_alloc((size_t) n_kv * n_kvh * k_row, 128);
    __fp16 *     md     = mmode ? lab_ddr_alloc((size_t) n_tokens * line, 128) : NULL;
    float *      sd     = lab_ddr_alloc((size_t) n_heads * 4, 128);
    float *      od     = lab_ddr_alloc((size_t) n_tokens * n_heads * DK * 4, 128);
    float *      rowbuf = lab_ddr_alloc((size_t) DK * 4, 128);
    double *     sc     = lab_ddr_alloc((size_t) n_kv * 8, 128);

    lab_fill_f32(qd, (size_t) n_tokens * n_heads * DK, -1.0f, 1.0f);
    for (uint32_t r = 0; r < n_kv * n_kvh; r++) {
        for (int kv = 0; kv < 2; kv++) {
            lab_fill_f32(rowbuf, DK, -2.0f, 2.0f);
            uint8_t * dst = (kv ? vd : kd) + (size_t) r * k_row;
            if (q8) {
                quantize_q8_0_row(rowbuf, (block_q8_0 *) dst, DK);
            } else {
                for (uint32_t j = 0; j < DK; j++) {
                    ((__fp16 *) dst)[j] = (__fp16) rowbuf[j];
                }
            }
        }
    }
    if (md) {
        fill_mask(md, n_tokens, n_kv, line / 2, mmode, n_kv, n_tokens);
    }
    for (uint32_t h = 0; h < n_heads; h++) {
        sd[h] = lab_rand_f32(sink_lo, sink_hi);
    }

    static struct htp_tensor tq, tk, tv, tm, ts, td;
    memset(&tq, 0, sizeof(tq)); memset(&tk, 0, sizeof(tk)); memset(&tv, 0, sizeof(tv));
    memset(&tm, 0, sizeof(tm)); memset(&ts, 0, sizeof(ts)); memset(&td, 0, sizeof(td));
    tq.data = (uint32_t) (uintptr_t) qd; tq.type = HTP_TYPE_F32;
    tq.ne[0] = DK; tq.ne[1] = n_tokens; tq.ne[2] = n_heads; tq.ne[3] = 1;
    tq.nb[0] = 4; tq.nb[1] = DK * 4 * n_heads; tq.nb[2] = DK * 4; tq.nb[3] = tq.nb[1] * n_tokens;
    tk.data = (uint32_t) (uintptr_t) kd; tk.type = q8 ? HTP_TYPE_Q8_0 : HTP_TYPE_F16;
    tk.ne[0] = DK; tk.ne[1] = n_kv; tk.ne[2] = n_kvh; tk.ne[3] = 1;
    tk.nb[0] = q8 ? sizeof(block_q8_0) : 2; tk.nb[1] = k_row * n_kvh; tk.nb[2] = k_row; tk.nb[3] = tk.nb[1] * n_kv;
    tv      = tk;
    tv.data = (uint32_t) (uintptr_t) vd;
    if (md) {
        tm.data = (uint32_t) (uintptr_t) md; tm.type = HTP_TYPE_F16;
        tm.ne[0] = n_kv; tm.ne[1] = n_tokens; tm.ne[2] = 1; tm.ne[3] = 1;
        tm.nb[0] = 2; tm.nb[1] = line; tm.nb[2] = line * n_tokens; tm.nb[3] = tm.nb[2];
    }
    ts.data = (uint32_t) (uintptr_t) sd; ts.type = HTP_TYPE_F32;
    ts.ne[0] = n_heads; ts.ne[1] = ts.ne[2] = ts.ne[3] = 1; ts.nb[0] = 4;
    td.data = (uint32_t) (uintptr_t) od; td.type = HTP_TYPE_F32;
    td.ne[0] = DK; td.ne[1] = n_heads; td.ne[2] = n_tokens; td.ne[3] = 1;
    td.nb[0] = 4; td.nb[1] = DK * 4; td.nb[2] = DK * 4 * n_heads; td.nb[3] = td.nb[2] * n_tokens;

    static struct htp_context     ctx;
    static struct htp_ops_context octx;
    static dma_queue              queues[HTP_MAX_NTHREADS];
    static struct hmx_queue_s     hq;
    memset(&ctx, 0, sizeof(ctx));
    memset(&octx, 0, sizeof(octx));
    memset(&hq, 0, sizeof(hq));
    ctx.vtcm_base     = lab_vtcm_base();
    ctx.vtcm_size     = vtcm_cap < lab_vtcm_size() ? vtcm_cap : lab_vtcm_size();
    ctx.n_threads     = n_thr;
    ctx.n_threads_div = init_fastdiv_values(n_thr);
    ctx.hmx_enabled   = true;
    ctx.hmx_queue     = &hq;
    for (uint32_t i = 0; i < n_thr; i++) {
        memset(&queues[i], 0, sizeof(queues[i]));
        ctx.dma[i] = &queues[i];
    }
    octx.ctx           = &ctx;
    octx.n_threads     = n_thr;
    octx.n_threads_div = ctx.n_threads_div;
    octx.src[0] = &tq; octx.src[1] = &tk; octx.src[2] = &tv;
    octx.src[3] = md ? &tm : NULL;
    octx.src[4] = use_snk ? &ts : NULL;
    octx.dst    = &td;

    struct htp_fa_kernel_params * kp = (struct htp_fa_kernel_params *) octx.kernel_params;
    if (!full_params(kp, variant, G, DK, n_tokens, n_kv, n_thr, ctx.vtcm_size, q8, md != NULL, scale)) {
        printf("lab: fa2 full: no plan fits\n");
        return 2;
    }
    printf("lab: fa2 full: tokens %u n_kv %u heads %u/%u DK %u %s mask %d sinks %d variant %d: kernel %u Br %u Bc %u "
           "kv blocks %u threads %u spans %u vtcm %u\n",
           n_tokens, n_kv, n_heads, n_kvh, DK, q8 ? "q8_0" : "f16", mmode, (int) use_snk, variant, kp->kernel_type, kp->Br,
           kp->Bc, kp->n_kv_blocks, kp->n_threads, kp->u.hmx.spans, kp->vtcm_size);

    for (size_t i = 0; i < (size_t) n_tokens * n_heads * DK; i++) {
        od[i] = NAN;
    }
    lab_dma_mutex = 0;
    memset(lab_dma_active, 0, sizeof(lab_dma_active));
    lab_dma_ring_violations = 0;
    const int st = op_flash_attn_ext(&octx);
    if (st != HTP_STATUS_OK) {
        printf("lab: fa2 full: the kernel gave the status %d\n", st);
        return 1;
    }
    if (hq.pushed != hq.popped) {
        printf("lab: fa2 full: %u HMX jobs pushed, %u popped\n", hq.pushed, hq.popped);
    }
    for (uint32_t i = 0; i < n_thr; i++) {
        if (queues[i].push_idx != queues[i].pop_idx) {
            printf("lab: fa2 full: DMA queue %u: %u pushed, %u popped\n", i, queues[i].push_idx, queues[i].pop_idx);
        }
    }
    lab_report(TARGET, "full_dma_ring_violations", lab_dma_ring_violations, "");
    // --check-stride 0 runs the kernel and the queue checks only, without the reference.
    if (stride == 0) {
        return (lab_dma_ring_violations || hq.pushed != hq.popped) ? 1 : 0;
    }

    // The reference in double: Q in f32, K and V as the kernel reads them, the mask with the
    // threshold of the kernel (a value of -16 or less masks the score).
    double   se = 0.0, sr = 0.0, max_err = 0.0;
    uint32_t n_nan = 0, n_rows = 0, n_bad = 0;
    for (uint32_t t = 0; t < n_tokens; t += stride) {
        for (uint32_t h = 0; h < n_heads; h++) {
            const uint32_t kh = h / G;
            const float *  qr = qd + ((size_t) t * n_heads + h) * DK;
            double         mx = use_snk ? (double) sd[h] : -1e300;
            for (uint32_t j = 0; j < n_kv; j++) {
                const uint8_t * kr = kd + ((size_t) j * n_kvh + kh) * k_row;
                double          s  = 0.0;
                for (uint32_t d = 0; d < DK; d++) {
                    s += (double) qr[d] * kv_value(kr, d, q8);
                }
                s *= scale;
                if (md) {
                    const double m = (double) md[(size_t) t * (line / 2) + j];
                    s              = m > -16.0 ? s + m : -INFINITY;
                }
                sc[j] = s;
                mx    = s > mx ? s : mx;
            }
            double l = use_snk ? exp((double) sd[h] - mx) : 0.0;
            for (uint32_t j = 0; j < n_kv; j++) {
                sc[j] = exp(sc[j] - mx);
                l += sc[j];
            }
            const float * o = od + ((size_t) t * n_heads + h) * DK;
            for (uint32_t d = 0; d < DK; d++) {
                double acc = 0.0;
                for (uint32_t j = 0; j < n_kv; j++) {
                    acc += sc[j] * kv_value(vd + ((size_t) j * n_kvh + kh) * k_row, d, q8);
                }
                const double want = l > 0.0 ? acc / l : 0.0;
                const double got  = (double) o[d];
                if (got != got) {
                    n_nan++;
                    continue;
                }
                const double e = fabs(got - want);
                if (t == 0 && h == 0 && d < 4) {
                    printf("lab: fa2 full: token 0 head 0 d %u: got %.6f want %.6f\n", d, got, want);
                }
                se += e * e;
                sr += want * want;
                max_err = e > max_err ? e : max_err;
                if (e > 0.02 + 0.02 * fabs(want)) {
                    if (n_bad < 5) {
                        printf("lab: fa2 full: token %u head %u d %u: %g, expected %g\n", t, h, d, got, want);
                    }
                    n_bad++;
                }
            }
            n_rows++;
        }
    }
    lab_report(TARGET, "full_rows_checked", n_rows, "");
    lab_report(TARGET, "full_nmse", sr > 0.0 ? se / sr : 0.0, "");
    lab_report(TARGET, "full_max_abs_err", max_err, "");
    lab_report(TARGET, "full_nan", n_nan, "");
    lab_report(TARGET, "full_bad", n_bad, "");
    lab_report(TARGET, "full_hmx_jobs", hq.pushed, "");
    return (n_nan || n_bad || lab_dma_ring_violations || hq.pushed != hq.popped) ? 1 : 0;
}

int main(int argc, char ** argv) {
    lab_init();
    const char * mode = "exp";
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0) {
            mode = argv[i + 1];
        }
    }
    if (strcmp(mode, "exp") == 0) {
        return mode_exp(argc, argv);
    }
    if (strcmp(mode, "smx") == 0) {
        return mode_smx(argc, argv);
    }
    if (strcmp(mode, "full") == 0) {
        return mode_full(argc, argv);
    }
    printf("lab: %s: the mode %s is not known\n", TARGET, mode);
    return 2;
}
