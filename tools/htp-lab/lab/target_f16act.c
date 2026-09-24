// Target: the F16 activation of the Hexagon backend (htp-mm-fusion.h of the host): the GLU op SWIGLU
// with an F16 dst (act-ops.c), and MUL_MAT and MUL_MAT_ADD on the HMX 2D path with an F16 src1
// (matmul-ops.c), bit for bit against the F32 forms.
//
// The program includes matmul-ops.c and act-ops.c verbatim and calls the op entry points with the kernel
// params that the host computes for the HMX 2D path. Two shims replace the engines that the standalone
// runtime of the simulator does not give:
//   - The DMA: a push copies at once and a pop returns the destinations in the order of the pushes
//     (the shim of target_fa.c). The order is the contract that the kernels depend on.
//   - The HMX queue: a push runs the job at once on the thread of the push, and a pop returns.
// Thus the lab finds errors of the layout and of the lane order, but not of the timing of the DMA.
// Run it in the functional mode of the simulator (MODE=functional): the timing model never retires an
// HMX instruction.
//
// The cases: the GLU op with rows that are not a multiple of 64 values, MUL_MAT and MUL_MAT_ADD with
// Q8_0 tiles and with F16 weights, a partial last row pair, several chunks of rows and of columns, and
// the ffn_down shape of the 4B (--big 1).
//
// Arguments: --threads 4 --vtcm 1048576 --glu-vtcm 8388608 --big 0. The thread count must not be more
// than the HVX contexts of the simulator (6 on v79, 4 on v73, v75 and v81), and --glu-vtcm not more than
// its VTCM (2 MB on v81). On v73, v75 and v81 the GLU op of this program does not give the scalar SwiGLU
// also with an F32 dst, thus only v79 (the phone) gives a result for the GLU cases.
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The MUL_MAT_ID path of matmul-ops.c calls memalign. The standalone runtime has no malloc.h.
void * memalign(size_t alignment, size_t size);

// --- the synchronous DMA shim (refer to target_fa.c) ---
#define HTP_DMA_H

typedef struct {
    void *       dst;
    const void * src;
} dma_ptr;

#define LAB_DMA_CAPACITY 256

typedef struct dma_queue_s {
    void *       dst[LAB_DMA_CAPACITY];
    const void * src[LAB_DMA_CAPACITY];
    uint32_t     push_idx;
    uint32_t     pop_idx;
} dma_queue;
typedef dma_queue * dma_queue_t;

static inline dma_ptr dma_make_ptr(void * dst, const void * src) {
    dma_ptr p = { dst, src };
    return p;
}

static inline bool dma_queue_push(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride, size_t row_size,
                                  size_t nrows) {
    for (size_t r = 0; r < nrows; r++) {
        memcpy((uint8_t *) p.dst + r * dst_stride, (const uint8_t *) p.src + r * src_stride, row_size);
    }
    q->dst[q->push_idx & (LAB_DMA_CAPACITY - 1)] = p.dst;
    q->src[q->push_idx & (LAB_DMA_CAPACITY - 1)] = p.src;
    q->push_idx++;
    return true;
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr p = { NULL, NULL };
    if (q->pop_idx == q->push_idx) {
        return p;
    }
    p.dst = q->dst[q->pop_idx & (LAB_DMA_CAPACITY - 1)];
    p.src = q->src[q->pop_idx & (LAB_DMA_CAPACITY - 1)];
    q->pop_idx++;
    return p;
}

static inline bool dma_queue_push_vtcm_to_ddr(dma_queue * q, dma_ptr p, size_t dst_row_size, size_t src_row_size,
                                              size_t nrows) {
    return dma_queue_push(q, p, dst_row_size, src_row_size, dst_row_size, nrows);
}

static inline void dma_queue_flush(dma_queue * q) {
    while (dma_queue_pop(q).dst != NULL) {
    }
}

// --- the synchronous HMX queue shim ---
#define HMX_QUEUE_H

#include "hex-profile.h"

typedef void (*hmx_queue_func)(void *);

struct hmx_queue_desc {
    hmx_queue_func func;
    void *         data;
};

struct hmx_queue_s {
    int pending;
};
typedef struct hmx_queue_s * hmx_queue_t;

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    d.func(d.data);
    q->pending++;
    return true;
}

static inline struct hmx_queue_desc hmx_queue_pop(hmx_queue_t q) {
    struct hmx_queue_desc d = { NULL, NULL };
    if (q->pending <= 0) {
        printf("lab: error: hmx_queue_pop with no job\n");
    }
    q->pending--;
    return d;
}

#include "matmul-ops.c"
#include "act-ops.c"

#define TARGET "f16act"

static struct htp_context  g_ctx;
static dma_queue           g_dma[HTP_MAX_NTHREADS];
static struct hmx_queue_s  g_hmx;
static size_t              g_fail  = 0;
static size_t              g_cases = 0;

static void fail(const char * what, uint32_t r, uint32_t c, uint32_t got, uint32_t want) {
    if (g_fail < 16) {
        printf("lab: %s mismatch at row %u col %u: got 0x%08x want 0x%08x\n", what, r, c, got, want);
    }
    g_fail++;
}

static void set_tensor(struct htp_tensor * t, void * data, uint32_t type, uint32_t ne0, uint32_t ne1, uint32_t nb0,
                       uint32_t nb1) {
    memset(t, 0, sizeof(*t));
    t->data  = (uint32_t) (uintptr_t) data;
    t->type  = type;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = 1;
    t->ne[3] = 1;
    t->nb[0] = nb0;
    t->nb[1] = nb1;
    t->nb[2] = nb1 * ne1;
    t->nb[3] = nb1 * ne1;
    t->size  = nb1 * ne1;
}

// The scalar SiLU. The lab flags let the compiler assume that no value is infinite, thus expf of a large
// argument (the overflow above 88) gives no defined result. The function uses the limits of the SiLU there.
static float scalar_silu(float g) {
    if (g < -80.0f) {
        return 0.0f;
    }
    if (g > 80.0f) {
        return g;
    }
    return g / (1.0f + expf(-g));
}

// The kernel params of the HMX 2D path for a weight of n rows: the solver calls of
// ggml_hexagon_precompute_hmx_mm_params of the host. Returns false when no layout fits.
static bool hmx_params(int wtype, uint32_t k, uint32_t n, uint32_t m, struct htp_mm_kernel_params * kp) {
    memset(kp, 0, sizeof(*kp));
    const int    nt     = (int) g_ctx.n_threads;
    const int    ats    = (int) htp_mm_get_weight_aligned_tile_size(wtype);
    const size_t budget = g_ctx.vtcm_size;
    bool         pipe   = htp_mm_hmx_pipeline(m);
    size_t       mc = 0, nc = 0, vtcm = 0;
    int          act_threads = 0;
    if (!htp_mm_hmx_solve_2d_params(wtype, k, 0, n, hex_round_up(m, 32), m, nt, pipe, false, ats, budget, &mc, &nc,
                                    &act_threads, &vtcm)) {
        pipe = false;
        if (!htp_mm_hmx_solve_2d_params(wtype, k, 0, n, hex_round_up(m, 32), m, nt, pipe, false, ats, budget, &mc, &nc,
                                        &act_threads, &vtcm)) {
            return false;
        }
    }
    kp->kernel_type       = HTP_MM_KERNEL_HMX_2D;
    kp->n_hmx             = 1;
    kp->pipeline          = pipe ? 1 : 0;
    kp->m_chunk           = (int32_t) mc;
    kp->n_chunk           = (int32_t) nc;
    kp->n_threads         = nt;
    kp->n_act_threads     = act_threads;
    kp->tile_size         = (int32_t) htp_mm_get_weight_tile_size(wtype);
    kp->aligned_tile_size = ats;
    kp->vtcm_size         = (int32_t) vtcm;
    kp->n_weights         = 1;
    kp->div_n_act_threads = init_fastdiv_values(act_threads);
    kp->div_ne00_padded   = init_fastdiv_values(k);
    printf("lab: hmx params m %u k %u n %u: mc %zu nc %zu act_threads %d pipeline %d vtcm %zu\n", m, k, n, mc, nc,
           act_threads, (int) pipe, vtcm);
    return vtcm <= budget;
}

static int run_op(uint32_t op, const struct htp_mm_kernel_params * kp, const struct htp_tensor * src,
                  unsigned n_src, const struct htp_tensor * dst, unsigned n_dst) {
    struct htp_ops_context * octx = &g_ctx.octx;
    memset(octx, 0, sizeof(*octx));
    octx->ctx           = &g_ctx;
    octx->op            = (enum htp_op_code) op;
    octx->n_threads     = g_ctx.n_threads;
    octx->n_threads_div = g_ctx.n_threads_div;
    memcpy(octx->kernel_params, kp, sizeof(*kp));
    for (unsigned i = 0; i < n_src; i++) {
        octx->src[i] = &src[i];
    }
    for (unsigned i = 0; i < n_dst; i++) {
        octx->dsts[i] = &dst[i];
    }
    const int s = op == HTP_OP_GLU_SWIGLU ? op_activations(octx) : op_matmul(octx);
    if (s != HTP_STATUS_OK) {
        printf("lab: error: op %u returned status %d\n", op, s);
        g_fail++;
    }
    return s;
}

// Fills weight bytes: Q8_0 tiles with random quants and small scales, or F16 values. O(bytes).
static void fill_weight(uint8_t * w, size_t w_bytes, int wtype, size_t n_values) {
    if (wtype == HTP_TYPE_Q8_0) {
        for (size_t t = 0; t < w_bytes / HTP_MM_WEIGHT_TILE_SIZE_Q8_0; t++) {
            uint8_t * tile = w + t * HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
            lab_fill_u8(tile, 1024);
            uint16_t * sc = (uint16_t *) (tile + 1024);
            for (int j = 0; j < 32; j++) {
                sc[j] = lab_f32_to_hf(lab_rand_f32(-1.0f / 64.0f, 1.0f / 64.0f));
            }
        }
    } else {
        uint16_t * h = (uint16_t *) w;
        for (size_t j = 0; j < n_values; j++) {
            h[j] = lab_f32_to_hf(lab_rand_f32(-0.25f, 0.25f));
        }
    }
}

// The F16 rows of an F32 matrix with the conversion of the SwiGLU op for an F16 activation
// (htp_act_rows_f32_to_f16 of act-ops.c). O(rows * k).
static void to_f16_rows(uint16_t * dst, const float * src, uint32_t rows, uint32_t k) {
    const uint32_t nvec = (k + 31) / 32;
    HVX_Vector *   tmp  = lab_ddr_alloc((size_t) (nvec + 1) * 128, 128);
    for (uint32_t r = 0; r < rows; r++) {
        memset(tmp, 0, (size_t) (nvec + 1) * 128);
        memcpy(tmp, src + (size_t) r * k, (size_t) k * 4);
        for (uint32_t j = 0; 2 * j < nvec; j++) {
            HVX_Vector h = hvx_vec_f32_to_f16(tmp[2 * j], tmp[2 * j + 1]);
            const uint32_t n = MIN(64, k - 64 * j);
            memcpy(dst + (size_t) r * k + 64 * j, &h, (size_t) n * 2);
        }
    }
    free(tmp);
}

// MUL_MAT (with src2: MUL_MAT_ADD) on the HMX path with an F32 activation and with its F16 rows. The
// two outputs must be bit-identical. The activation range lo..hi can go outside the F16 range.
static void run_f16_act_case(int wtype, uint32_t k, uint32_t n, uint32_t m, bool with_src2, float x_range) {
    const uint32_t n_pad   = wtype == HTP_TYPE_Q8_0 ? hex_round_up(n, 32) : n;
    const uint32_t nkt     = k / 32;
    const size_t   w_bytes = wtype == HTP_TYPE_Q8_0 ? (size_t) (n_pad / 32) * nkt * HTP_MM_WEIGHT_TILE_SIZE_Q8_0
                                                    : (size_t) n_pad * k * 2;
    g_cases++;
    printf("lab: case f16-act %s k %u n %u m %u src2 %d range %g\n", wtype == HTP_TYPE_Q8_0 ? "q8_0" : "f16", k, n,
           m, (int) with_src2, x_range);

    uint8_t * w = lab_ddr_alloc(w_bytes, 128);
    fill_weight(w, w_bytes, wtype, (size_t) n_pad * k);
    float *    x   = lab_ddr_alloc((size_t) m * k * 4, 128);
    uint16_t * x16 = lab_ddr_alloc((size_t) m * k * 2 + 128, 128);
    float *    s2  = lab_ddr_alloc((size_t) m * n * 4, 128);
    float *    o32 = lab_ddr_alloc((size_t) m * n * 4, 128);
    float *    o16 = lab_ddr_alloc((size_t) m * n * 4, 128);
    lab_fill_f32(x, (size_t) m * k, -x_range, x_range);
    lab_fill_f32(s2, (size_t) m * n, -1.0f, 1.0f);
    to_f16_rows(x16, x, m, k);

    struct htp_mm_kernel_params kp;
    if (!hmx_params(wtype, k, n_pad, m, &kp)) {
        printf("lab: error: no layout\n");
        g_fail++;
    } else {
        struct htp_tensor src[3], dst;
        set_tensor(&src[0], w, wtype, k, n_pad, wtype == HTP_TYPE_Q8_0 ? sizeof(block_q8_0) : 2,
                   wtype == HTP_TYPE_Q8_0 ? nkt * sizeof(block_q8_0) : 2 * k);
        set_tensor(&src[2], s2, HTP_TYPE_F32, n, m, 4, 4 * n);
        const uint32_t op = with_src2 ? HTP_OP_MUL_MAT_ADD : HTP_OP_MUL_MAT;

        set_tensor(&src[1], x, HTP_TYPE_F32, k, m, 4, 4 * k);
        set_tensor(&dst, o32, HTP_TYPE_F32, n, m, 4, 4 * n);
        run_op(op, &kp, src, with_src2 ? 3 : 2, &dst, 1);

        set_tensor(&src[1], x16, HTP_TYPE_F16, k, m, 2, 2 * k);
        set_tensor(&dst, o16, HTP_TYPE_F32, n, m, 4, 4 * n);
        run_op(op, &kp, src, with_src2 ? 3 : 2, &dst, 1);

        for (uint32_t r = 0; r < m; r++) {
            for (uint32_t c = 0; c < n; c++) {
                uint32_t a, b;
                memcpy(&a, o16 + (size_t) r * n + c, 4);
                memcpy(&b, o32 + (size_t) r * n + c, 4);
                if (a != b) {
                    fail("f16-act", r, c, a, b);
                }
            }
        }
    }
    void * bufs[] = { w, x, x16, s2, o32, o16 };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        free(bufs[i]);
    }
}

// The GLU op SWIGLU with an F32 dst and with an F16 dst (the view of the plan). The F16 value of each
// lane must be the value that the activation load of a MUL_MAT gives for the F32 value.
static void run_glu_case(uint32_t n, uint32_t m, float range) {
    g_cases++;
    printf("lab: case glu-f16 n %u m %u range %g\n", n, m, range);
    float *    g   = lab_ddr_alloc((size_t) m * n * 4, 128);
    float *    u   = lab_ddr_alloc((size_t) m * n * 4, 128);
    float *    o32 = lab_ddr_alloc((size_t) m * n * 4, 128);
    uint16_t * o16 = lab_ddr_alloc((size_t) m * n * 2 + 256, 128);
    uint16_t * r16 = lab_ddr_alloc((size_t) m * n * 2 + 256, 128);
    lab_fill_f32(g, (size_t) m * n, -2.0f * range, 2.0f * range);
    lab_fill_f32(u, (size_t) m * n, -range, range);
    memset(o16, 0x5a, (size_t) m * n * 2 + 256);

    struct htp_mm_kernel_params none;
    memset(&none, 0, sizeof(none));
    struct htp_tensor src[2], dst;
    set_tensor(&src[0], g, HTP_TYPE_F32, n, m, 4, 4 * n);
    set_tensor(&src[1], u, HTP_TYPE_F32, n, m, 4, 4 * n);
    set_tensor(&dst, o32, HTP_TYPE_F32, n, m, 4, 4 * n);
    run_op(HTP_OP_GLU_SWIGLU, &none, src, 2, &dst, 1);
    set_tensor(&dst, o16, HTP_TYPE_F16, n, m, 2, 2 * n);
    run_op(HTP_OP_GLU_SWIGLU, &none, src, 2, &dst, 1);

    // The F32 op against the scalar SwiGLU (a check of the reference, not of the F16 path). For a gate far
    // below 0 the SiLU of the op gives -2^-15 (3.05e-5) where the SiLU is smaller, thus the bound also
    // allows 3.1e-5 times |up|.
    size_t bad32 = 0;
    for (size_t i = 0; i < (size_t) m * n; i++) {
        const float want = scalar_silu(g[i]) * u[i];
        if (!(fabsf(o32[i] - want) <= 1e-2f * (1.0f + fabsf(want)) + 3.1e-5f * fabsf(u[i]))) {
            if (bad32 < 4) {
                printf("lab: glu-f32 differs from the scalar SwiGLU at %zu: %g against %g\n", i, o32[i], want);
            }
            bad32++;
        }
    }
    if (bad32) {
        printf("lab: glu-f32 differs from the scalar SwiGLU in %zu of %zu values\n", bad32, (size_t) m * n);
        g_fail += bad32;
    }

    to_f16_rows(r16, o32, m, n);
    for (uint32_t r = 0; r < m; r++) {
        for (uint32_t c = 0; c < n; c++) {
            if (o16[(size_t) r * n + c] != r16[(size_t) r * n + c]) {
                fail("glu-f16", r, c, o16[(size_t) r * n + c], r16[(size_t) r * n + c]);
            }
        }
    }
    const uint8_t * tail = (const uint8_t *) (o16 + (size_t) m * n);
    for (int b = 0; b < 256; b++) {
        if (tail[b] != 0x5a) {
            fail("glu-f16-tail", m, (uint32_t) b, tail[b], 0x5a);
            break;
        }
    }
    void * bufs[] = { g, u, o32, o16, r16 };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        free(bufs[i]);
    }
}

// The SiLU of the GLU op SWIGLU (up = 1) against the scalar SiLU for gate values from -limit to limit in
// steps of limit / 4096. It prints the smallest gate magnitude with an error above 1e-3 * max(1, |SiLU|)
// on each side, and the first such values. A measure of the SwiGLU of HEAD, not a check of the F16 path.
// O(8192).
static void run_glu_sweep(float limit) {
    const uint32_t n = 8192;
    float *        g = lab_ddr_alloc(n * 4, 128);
    float *        u = lab_ddr_alloc(n * 4, 128);
    float *        o = lab_ddr_alloc(n * 4, 128);
    for (uint32_t i = 0; i < n; i++) {
        g[i] = -limit + 2.0f * limit * (float) i / (float) (n - 1);
        u[i] = 1.0f;
    }
    struct htp_mm_kernel_params none;
    memset(&none, 0, sizeof(none));
    struct htp_tensor src[2], dst;
    set_tensor(&src[0], g, HTP_TYPE_F32, n, 1, 4, 4 * n);
    set_tensor(&src[1], u, HTP_TYPE_F32, n, 1, 4, 4 * n);
    set_tensor(&dst, o, HTP_TYPE_F32, n, 1, 4, 4 * n);
    run_op(HTP_OP_GLU_SWIGLU, &none, src, 2, &dst, 1);

    float  first_neg = -INFINITY, first_pos = INFINITY;
    size_t n_bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float want = scalar_silu(g[i]);
        const float err  = fabsf(o[i] - want) / fmaxf(fabsf(want), 1.0f);
        if (!(err <= 1e-3f)) {
            if (n_bad < 8) {
                printf("lab: glu-sweep gate %g: silu %g, scalar %g\n", g[i], o[i], want);
            }
            n_bad++;
            if (g[i] < 0.0f) {
                first_neg = fmaxf(first_neg, g[i]);
            } else {
                first_pos = fminf(first_pos, g[i]);
            }
        }
    }
    printf("lab: glu-sweep limit %g: %zu of %u values with an error above 1e-3 * max(1, |SiLU|), the smallest such "
           "gates %g and %g\n", limit, n_bad, n, first_neg, first_pos);
    free(g);
    free(u);
    free(o);
}

int main(int argc, char ** argv) {
    lab_init();

    const uint32_t nt = (uint32_t) lab_arg_long(argc, argv, "--threads", 4);

    // The HMX model of the simulator stops with the exception 0x26 (coprocessor VMEM address error)
    // when the range of one deep tile load crosses a multiple of 1 MB above the VTCM base, thus the ops
    // get the first 1 MB. The phone runs the same code with 8 MB.
    g_ctx.vtcm_base     = lab_vtcm_base();
    g_ctx.vtcm_size     = (size_t) lab_arg_long(argc, argv, "--vtcm", 1 << 20);
    g_ctx.n_threads     = nt;
    g_ctx.n_threads_div = init_fastdiv_values(nt);
    g_ctx.work_queue    = (work_queue_t) &g_hmx;  // the lab stub does not read it
    g_ctx.hmx_queue     = &g_hmx;
    g_ctx.mdev.count    = 1;
    for (uint32_t i = 0; i < HTP_MAX_NTHREADS; i++) {
        g_ctx.dma[i] = &g_dma[i];
    }

    printf("lab: %s threads %u vtcm %zu\n", TARGET, nt, g_ctx.vtcm_size);

    // --glu-sweep 1: only the measure of the SiLU of the GLU op for large gate values
    if (lab_arg_long(argc, argv, "--glu-sweep", 0)) {
        g_ctx.vtcm_size      = 8 << 20;
        const float limits[] = { 32.0f, 128.0f, 1024.0f };
        for (size_t i = 0; i < sizeof(limits) / sizeof(limits[0]); i++) {
            run_glu_sweep(limits[i]);
        }
        return 0;
    }

    // The GLU op: rows that are not a multiple of 64 values, one row, and values outside the F16 range.
    // The GLU op uses no HMX, thus it gets the VTCM of the phone (a row of 9216 values needs 108 KB for
    // each thread).
    const size_t hmx_vtcm = g_ctx.vtcm_size;
    g_ctx.vtcm_size       = (size_t) lab_arg_long(argc, argv, "--glu-vtcm", 8 << 20);
    run_glu_case(200, 7, 4.0f);
    run_glu_case(9216, 3, 4.0f);
    run_glu_case(96, 1, 400.0f);
    g_ctx.vtcm_size = hmx_vtcm;
    // MUL_MAT and MUL_MAT_ADD with an F16 src1: Q8_0 tiles and F16 weights, a partial last row pair,
    // several row chunks, and activations outside the F16 range
    run_f16_act_case(HTP_TYPE_Q8_0, 512, 256, 37, true, 2.0f);
    run_f16_act_case(HTP_TYPE_Q8_0, 96, 64, 64, false, 2.0f);
    run_f16_act_case(HTP_TYPE_Q8_0, 2048, 96, 100, true, 2.0f);
    run_f16_act_case(HTP_TYPE_F16, 256, 128, 45, true, 2.0f);
    run_f16_act_case(HTP_TYPE_Q8_0, 256, 64, 40, true, 1.0e5f);
    if (lab_arg_long(argc, argv, "--big", 0)) {
        run_f16_act_case(HTP_TYPE_Q8_0, 9216, 64, 64, true, 2.0f);
    }

    lab_report(TARGET, "cases", (double) g_cases, "cases");
    lab_report(TARGET, "mismatches", (double) g_fail, "values");
    printf("lab: check %s %s\n", TARGET, g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
