// Target: the chunks of the HMX 2D matmul, bit for bit. The ops MUL_MAT, MUL_MAT with src2 (the MUL_MAT_ADD form)
// and MUL_MAT_NX of matmul-ops.c run with the chunks of the old solver (htp_mm_hmx_solve_2d_params), of the cost
// model of the kernel (htp_mm_hmx_solve_2d_cost) and of a list of requests (htp_mm_hmx_fit_2d_chunks, as
// GGML_HEXAGON_MM_CHUNKS of the host gives them). Each output tile of the kernel is one accumulator chain over all
// of k for each selection of the chunks. Thus each run must give the bytes of the run with the old chunks, and the
// bytes after dst must not change.
//
// The program includes matmul-ops.c verbatim and calls the op entry points op_matmul and op_matmul_nx with the
// kernel params that the host computes. The shims of target_mmswiglu_op.c replace the engines that the standalone
// runtime of the simulator does not give:
//   - The DMA: a push copies at once and a pop returns the destinations in the order of the pushes.
//   - The HMX queue: a push runs the job at once on the calling thread, and a pop returns.
// Run it in the functional mode of the simulator (MODE=functional): the timing model never retires an HMX
// instruction. The HMX model of the simulator stops with the exception 0x26 when one deep tile load crosses a
// multiple of 1 MB above the VTCM base, thus the ops get 1 MB of VTCM. The shapes are small so that the old model
// takes more than one pass in 1 MB, as it does in the 8 MB of the phone at 512 to 1024 tokens.
//
// The cases: Q8_0 tiles and F16 weights, token counts that give a partial last token chunk, weight row counts that
// give a partial last weight chunk, the narrowest n chunk (32), one n chunk and two n chunks (the prologue of the
// pipelined loop), and a request with a value of 0. A Q8_0 run on the pipelined layout also runs with the weight
// streams PACKED and RINGS (hmx_wstream of the kernel params), which must give the same bytes. The DMA shim copies at
// the push, and the work queue of the lab runs the workers one after the other, thus a worker that moves bytes into
// the tiles of a different worker before that worker reads them changes the output.
//
// With --dq_tiles N the program only measures the Q8_0 dequantization task of one thread on N tiles, aligned against
// packed (dq_bench), in the timing mode.
//
// Arguments: --threads 6 --vtcm 1048576 [--dq_tiles 480]
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
    void *   dst[LAB_DMA_CAPACITY];
    uint32_t push_idx;
    uint32_t pop_idx;
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

#define TARGET "mmchunks"

static struct htp_context g_ctx;
static dma_queue          g_dma[HTP_MAX_NTHREADS];
static struct hmx_queue_s g_hmx;
static size_t             g_fail  = 0;
static size_t             g_runs  = 0;
static size_t             g_cases = 0;

enum form { FORM_MM, FORM_ADD, FORM_NX };

static const char * form_name(enum form f) {
    return f == FORM_MM ? "mm" : f == FORM_ADD ? "add" : "nx";
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

// The kernel params of the HMX 2D path, as ggml_hexagon_precompute_hmx_mm_params and ggml_hexagon_solve_hmx_2d of
// the host give them. sel 0: the old solver, 1: the cost model of the kernel, 2: the request mc, nc on top of the
// cost model. The pipelined layout first, then the serial layout. Returns false when no layout is in the budget.
static bool hmx_params(int sel, int wtype, uint32_t k, uint32_t n, uint32_t m, size_t mc_req, size_t nc_req,
                       uint32_t n_weights, struct htp_mm_kernel_params * kp) {
    memset(kp, 0, sizeof(*kp));
    const int    nt     = (int) g_ctx.n_threads;
    const int    ats    = (int) htp_mm_get_weight_aligned_tile_size(wtype);
    const size_t budget = g_ctx.vtcm_size;
    const uint32_t m_pad = hex_round_up(m, 32);
    size_t       mc = 0, nc = 0, vtcm = 0;
    int          act_threads = 0;
    bool         ok = false;
    bool         pipe = htp_mm_hmx_pipeline(m);
    for (int attempt = 0; attempt < 2 && !ok; attempt++) {
        if (attempt == 1) {
            if (!pipe) {
                break;
            }
            pipe = false;
        }
        ok = sel == 0 ? htp_mm_hmx_solve_2d_params(wtype, k, 0, n, m_pad, m, nt, pipe, false, ats, budget, &mc, &nc,
                                                   &act_threads, &vtcm)
                      : htp_mm_hmx_solve_2d_cost(wtype, k, n, m_pad, m, nt, pipe, ats, budget, &mc, &nc, &act_threads,
                                                 &vtcm);
    }
    if (!ok) {
        return false;
    }
    if (sel == 2) {
        size_t rm = 0, rn = 0, rv = 0;
        int    ra = 0;
        if (!htp_mm_hmx_fit_2d_chunks(wtype, k, n, m_pad, nt, pipe, ats, budget, mc_req, nc_req, &rm, &rn, &ra, &rv)) {
            return false;
        }
        mc = rm;
        nc = rn;
        act_threads = ra;
        vtcm = rv;
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
    kp->n_weights         = (int32_t) n_weights;
    kp->div_n_act_threads = init_fastdiv_values(act_threads);
    kp->div_ne00_padded   = init_fastdiv_values(k);
    return vtcm <= budget;
}

static int run_op(uint32_t op, const struct htp_mm_kernel_params * kp, const struct htp_tensor * src, unsigned n_src,
                  const struct htp_tensor * dst, unsigned n_dst) {
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
    return op == HTP_OP_MUL_MAT_NX ? op_matmul_nx(octx) : op_matmul(octx);
}

// The output of the old chunks is the reference of the other runs, thus it must be a matrix product and not, for
// example, all zeros. Each value must be finite and at most 1 % of the values can be 0. For an F16 weight, 256 sampled
// values must be within 2e-2 x (1 + |value|) of a float product of the f16 inputs (the HMX rounds the activation to
// f16 and accumulates in f16 groups). O(n x m + 256 x k).
static void check_reference(enum form f, int wtype, uint32_t k, uint32_t n, uint32_t m, const struct htp_tensor * w,
                            const float * x, const float * r, const float * out) {
    size_t zeros = 0, nonfinite = 0;
    for (size_t i = 0; i < (size_t) n * m; i++) {
        zeros += out[i] == 0.0f;
        nonfinite += !isfinite(out[i]);
    }
    size_t bad = 0;
    if (wtype == HTP_TYPE_F16) {
        const uint16_t * h = (const uint16_t *) (uintptr_t) w->data;
        for (int s = 0; s < 256; s++) {
            const uint32_t row = lab_rand_u32() % m;
            const uint32_t col = lab_rand_u32() % n;
            float          acc = r ? r[(size_t) row * n + col] : 0.0f;
            for (uint32_t kk = 0; kk < k; kk++) {
                acc += lab_hf_to_f32(h[(size_t) col * k + kk]) * lab_hf_to_f32(lab_f32_to_hf(x[(size_t) row * k + kk]));
            }
            const float got = out[(size_t) row * n + col];
            if (!(fabsf(got - acc) <= 2e-2f * (1.0f + fabsf(acc)))) {
                if (bad < 4) {
                    printf("lab:   reference row %u col %u: got %g want %g\n", row, col, got, acc);
                }
                bad++;
            }
        }
    }
    const bool ok = nonfinite == 0 && zeros * 100 <= (size_t) n * m && bad == 0;
    printf("lab:   reference (%s): %zu values 0, %zu not finite%s: %s\n", form_name(f), zeros, nonfinite,
           wtype == HTP_TYPE_F16 ? ", 256 sampled values against a float product" : "", ok ? "ok" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

// A request of chunks: mc and nc for htp_mm_hmx_fit_2d_chunks (0 gets the largest value for the other one)
struct request {
    uint32_t mc;
    uint32_t nc;
};

// One case: the op with the old chunks gives the reference bytes, then the op with the new chunks and with each
// request must give the same bytes. The bytes after each dst must stay 0x5a.
static void run_case(enum form f, int wtype, uint32_t k, uint32_t n, uint32_t m, const struct request * req,
                     size_t n_req) {
    const uint32_t nkt       = k / 32;
    const uint32_t n_weights = f == FORM_NX ? 2 : 1;
    size_t         w_bytes;
    uint32_t       w_nb0, w_nb1;
    if (wtype == HTP_TYPE_Q8_0) {
        w_nb0   = sizeof(block_q8_0);
        w_nb1   = nkt * sizeof(block_q8_0);
        w_bytes = (size_t) (n / 32) * nkt * HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    } else {
        w_nb0   = 2;
        w_nb1   = 2 * k;
        w_bytes = (size_t) n * k * 2;
    }
    g_cases++;
    printf("lab: case %s %s k %u n %u m %u\n", form_name(f), wtype == HTP_TYPE_Q8_0 ? "q8_0" : "f16", k, n, m);

    struct htp_tensor src[3], dst[2];
    for (uint32_t i = 0; i < n_weights; i++) {
        uint8_t * w = lab_ddr_alloc(w_bytes, 128);
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
            for (size_t j = 0; j < (size_t) n * k; j++) {
                h[j] = lab_f32_to_hf(lab_rand_f32(-0.25f, 0.25f));
            }
        }
        set_tensor(&src[i], w, wtype, k, n, w_nb0, w_nb1);
    }
    float * x = lab_ddr_alloc((size_t) m * k * 4, 128);
    lab_fill_f32(x, (size_t) m * k, -1.0f, 1.0f);
    set_tensor(&src[n_weights], x, HTP_TYPE_F32, k, m, 4, 4 * k);
    unsigned n_src = n_weights + 1;
    if (f == FORM_ADD) {
        float * r = lab_ddr_alloc((size_t) m * n * 4, 128);
        lab_fill_f32(r, (size_t) m * n, -1.0f, 1.0f);
        set_tensor(&src[2], r, HTP_TYPE_F32, n, m, 4, 4 * n);
        n_src = 3;
    }

    const size_t out_bytes = (size_t) m * n * 4;
    uint8_t *    ref[2];
    uint8_t *    out[2];
    for (uint32_t i = 0; i < n_weights; i++) {
        ref[i] = lab_ddr_alloc(out_bytes, 128);
        out[i] = lab_ddr_alloc(out_bytes + 256, 128);
        set_tensor(&dst[i], out[i], HTP_TYPE_F32, n, m, 4, 4 * n);
    }
    const uint32_t op = f == FORM_NX ? HTP_OP_MUL_MAT_NX : HTP_OP_MUL_MAT;

    const size_t n_sel = 2 + n_req;
    for (size_t s = 0; s < n_sel; s++) {
        const int    sel = s < 2 ? (int) s : 2;
        const size_t mcr = s < 2 ? 0 : req[s - 2].mc;
        const size_t ncr = s < 2 ? 0 : req[s - 2].nc;
        struct htp_mm_kernel_params kp;
        char what[64];
        if (s < 2) {
            snprintf(what, sizeof(what), "%s", s == 0 ? "old" : "new");
        } else {
            snprintf(what, sizeof(what), "request %zu,%zu", mcr, ncr);
        }
        if (!hmx_params(sel, wtype, k, n, m, mcr, ncr, n_weights, &kp)) {
            printf("lab:   %-18s no layout in the budget\n", what);
            if (s < 2) {
                g_fail++;
            }
            continue;
        }
        // Each weight stream of the pipelined Q8_0 loop (hmx_wstream) must give the bytes of the reference. A tree
        // without the field runs the one form of the loop.
#ifdef HTP_MM_WSTREAM_RINGS
        const int n_ws = wtype == HTP_TYPE_Q8_0 && kp.pipeline ? 3 : 1;
#else
        const int n_ws = 1;
#endif
        for (int ws = 0; ws < n_ws; ws++) {
#ifdef HTP_MM_WSTREAM_RINGS
            kp.hmx_wstream = ws;
#endif
            for (uint32_t i = 0; i < n_weights; i++) {
                memset(out[i], 0x5a, out_bytes + 256);
            }
            const int status = run_op(op, &kp, src, n_src, dst, n_weights);
            g_runs++;
            size_t diff = 0, tail = 0;
            for (uint32_t i = 0; i < n_weights; i++) {
                if (s == 0 && ws == 0) {
                    memcpy(ref[i], out[i], out_bytes);
                    check_reference(f, wtype, k, n, m, &src[i], x,
                                    f == FORM_ADD ? (const float *) (uintptr_t) src[2].data : NULL, (const float *) ref[i]);
                } else {
                    for (size_t b = 0; b < out_bytes; b += 4) {
                        diff += memcmp(out[i] + b, ref[i] + b, 4) != 0;
                    }
                }
                for (size_t b = 0; b < 256; b++) {
                    tail += out[i][out_bytes + b] != 0x5a;
                }
            }
            const uint32_t passes = (m + (uint32_t) kp.m_chunk - 1) / (uint32_t) kp.m_chunk;
            const uint32_t chunks = passes * ((n + (uint32_t) kp.n_chunk - 1) / (uint32_t) kp.n_chunk);
            printf("lab:   %-18s ws %d mc %4d nc %4d act %d %s passes %2u chunks %3u vtcm %7d: status %d, %zu values "
                   "differ from old, %zu tail bytes changed%s\n",
                   what, ws, kp.m_chunk, kp.n_chunk, kp.n_act_threads, kp.pipeline ? "pipe  " : "serial", passes, chunks,
                   kp.vtcm_size, status, diff, tail, (status != HTP_STATUS_OK || diff || tail) ? "  FAIL" : "");
            if (status != HTP_STATUS_OK || diff || tail) {
                g_fail++;
            }
        }
    }
}

#ifdef HTP_MM_WSTREAM_RINGS
// The cycles of the Q8_0 dequantization task of one thread on n_tiles tiles in VTCM: the aligned task on tiles at the
// pitch aligned_tile_size (the streams TILES), and the packed task on tiles at the pitch tile_size (the streams PACKED
// and RINGS, half of the tiles start 64 bytes after a vector edge). The two tasks must give the same f16 bytes. Run it
// in the timing mode: it uses no HMX. O(n_tiles * iters).
static void dq_bench(uint32_t n_tiles, int iters) {
    const uint32_t ts  = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const uint32_t ats = HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0;
    uint8_t *      raw_a = (uint8_t *) lab_vtcm_alloc((size_t) n_tiles * ats, 128);
    uint8_t *      raw_p = (uint8_t *) lab_vtcm_alloc((size_t) n_tiles * ats, 128);
    __fp16 *       out_a = (__fp16 *) lab_vtcm_alloc((size_t) n_tiles * HTP_MM_HMX_TILE_N_ELMS * sizeof(__fp16), 128);
    __fp16 *       out_p = (__fp16 *) lab_vtcm_alloc((size_t) n_tiles * HTP_MM_HMX_TILE_N_ELMS * sizeof(__fp16), 128);
    uint8_t        tile[HTP_MM_WEIGHT_TILE_SIZE_Q8_0];

    for (uint32_t t = 0; t < n_tiles; t++) {
        lab_fill_u8(tile, 1024);
        for (int s = 0; s < 32; s++) {  // finite f16 scales in [0.5, 1)
            const uint16_t h = lab_f32_to_hf(lab_rand_f32(0.5f, 1.0f));
            memcpy(tile + 1024 + 2 * s, &h, 2);
        }
        memcpy(raw_a + (size_t) t * ats, tile, ts);
        memcpy(raw_p + (size_t) t * ts, tile, ts);
    }
    tiled_dequantize_state_t st;
    memset(&st, 0, sizeof(st));
    st.tile_size         = ts;
    st.aligned_tile_size = ats;
    uint64_t best_a = UINT64_MAX, best_p = UINT64_MAX;
    for (int it = 0; it < iters; it++) {
        st.src = raw_a;
        st.dst = out_a;
        LAB_BARRIER();
        uint64_t c0 = lab_cycles();
        dequantize_tiled_weight_to_fp16_task_q8_0(&st, 0, n_tiles);
        LAB_BARRIER();
        uint64_t c1 = lab_cycles();
        best_a      = c1 - c0 < best_a ? c1 - c0 : best_a;
        st.src      = raw_p;
        st.dst      = out_p;
        LAB_BARRIER();
        c0 = lab_cycles();
        dequantize_tiled_weight_to_fp16_task_q8_0_packed(&st, 0, n_tiles);
        LAB_BARRIER();
        c1     = lab_cycles();
        best_p = c1 - c0 < best_p ? c1 - c0 : best_p;
    }
    bool same = memcmp(out_a, out_p, (size_t) n_tiles * HTP_MM_HMX_TILE_N_ELMS * sizeof(__fp16)) == 0;
    // The ranges of the workers: a range can start at an odd tile and end at an even tile
    if (n_tiles > 8) {
        memset(out_p, 0, (size_t) n_tiles * HTP_MM_HMX_TILE_N_ELMS * sizeof(__fp16));
        const uint32_t cut[] = { 0, 3, 8, n_tiles - 1, n_tiles };
        for (int r = 0; r + 1 < (int) (sizeof(cut) / sizeof(cut[0])); r++) {
            dequantize_tiled_weight_to_fp16_task_q8_0_packed(&st, cut[r], cut[r + 1]);
        }
        same = same && memcmp(out_a, out_p, (size_t) n_tiles * HTP_MM_HMX_TILE_N_ELMS * sizeof(__fp16)) == 0;
    }
    printf("lab: dq %u tiles: aligned %.1f cycles per tile, packed %.1f cycles per tile (%+.1f %%), f16 bytes %s\n",
           n_tiles, (double) best_a / n_tiles, (double) best_p / n_tiles,
           100.0 * ((double) best_p - (double) best_a) / (double) best_a, same ? "the same" : "DIFFERENT");
    lab_report(TARGET, "dq_aligned", (double) best_a / n_tiles, "cycles/tile");
    lab_report(TARGET, "dq_packed", (double) best_p / n_tiles, "cycles/tile");
    g_fail += same ? 0 : 1;
}
#endif

int main(int argc, char ** argv) {
    lab_init();

    const uint32_t nt = (uint32_t) lab_arg_long(argc, argv, "--threads", 6);
    const uint32_t dq = (uint32_t) lab_arg_long(argc, argv, "--dq_tiles", 0);
    if (dq > 0) {
#ifdef HTP_MM_WSTREAM_RINGS
        dq_bench(dq, 3);
        printf("lab: check %s %s\n", TARGET, g_fail == 0 ? "PASS" : "FAIL");
        return g_fail == 0 ? 0 : 1;
#else
        printf("lab: --dq_tiles needs the tree with the weight streams (HTP_MM_WSTREAM_RINGS)\n");
        return 1;
#endif
    }

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

    // The old model takes 2 or 3 passes in these shapes, the new model 1 or 2 (tools/stages/mmsolve/plan.cpp pair
    // with PLAN_VTCM=1048576). The requests add the narrowest n chunk, one or two n chunks, partial chunks, and a 0.
    const struct request r_a[] = { { 32, 32 }, { 0, 32 }, { 512, 0 }, { 160, 96 }, { 96, 1024 } };
    run_case(FORM_MM, HTP_TYPE_Q8_0, 512, 1024, 512, r_a, 5);
    const struct request r_b[] = { { 1024, 32 }, { 0, 64 }, { 224, 128 } };
    run_case(FORM_ADD, HTP_TYPE_Q8_0, 256, 1024, 1000, r_b, 3);
    const struct request r_c[] = { { 0, 32 }, { 288, 0 }, { 64, 512 } };
    run_case(FORM_NX, HTP_TYPE_Q8_0, 768, 512, 520, r_c, 3);
    const struct request r_d[] = { { 0, 32 }, { 256, 96 } };
    run_case(FORM_MM, HTP_TYPE_F16, 512, 768, 700, r_d, 2);
    const struct request r_e[] = { { 0, 32 }, { 64, 64 } };
    run_case(FORM_ADD, HTP_TYPE_Q8_0, 1024, 384, 480, r_e, 2);
    // A small row count (the verify batch of the MTP draft): one token chunk, the two models agree
    const struct request r_f[] = { { 32, 32 }, { 0, 480 } };
    run_case(FORM_MM, HTP_TYPE_Q8_0, 512, 1024, 5, r_f, 2);
    // One n chunk and two n chunks: the prologue of the pipelined loop pushes one or two weight chunks
    const struct request r_g[] = { { 0, 64 }, { 0, 32 }, { 32, 64 } };
    run_case(FORM_ADD, HTP_TYPE_Q8_0, 256, 64, 100, r_g, 3);
    // The row counts of a short prefill call on the pipelined path (5 to 64 rows), with a partial last weight chunk
    // (n 1056 is 33 column tiles) and the NX form: the weight streams of the Q8_0 loop
    const struct request r_h[] = { { 0, 96 }, { 0, 160 } };
    run_case(FORM_MM, HTP_TYPE_Q8_0, 2560, 1056, 22, r_h, 2);
    run_case(FORM_ADD, HTP_TYPE_Q8_0, 2560, 1056, 8, r_h, 2);
    run_case(FORM_NX, HTP_TYPE_Q8_0, 2560, 1056, 5, r_h, 2);
    run_case(FORM_MM, HTP_TYPE_Q8_0, 1024, 1056, 64, r_h, 2);
    run_case(FORM_NX, HTP_TYPE_Q8_0, 1024, 544, 32, r_h, 2);

    lab_report(TARGET, "cases", (double) g_cases, "cases");
    lab_report(TARGET, "runs", (double) g_runs, "runs");
    lab_report(TARGET, "failures", (double) g_fail, "runs");
    printf("lab: check %s %s\n", TARGET, g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
