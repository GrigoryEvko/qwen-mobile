// Target: the whole GATED_DELTA_NET op of a batch of tokens (op_gated_delta_net of
// gated-delta-net-ops.c, which sends the batch to the chunked kernel of gdn-chunk-ops.c).
//
// The program includes the two kernel files verbatim and calls the op, thus the run covers the
// group loop, the phases of the worker threads, the HMX jobs and the tail. Two shims replace the
// runtime services that the standalone simulator does not give:
//   - The DMA queue is a FIFO that copies at once (the engine raises "No Access" in this runtime,
//     refer to target_fa.c). The copies are ordinary loads and stores, thus the cycle numbers hold
//     the transfer cost inside the kernel.
//   - The HMX queue runs each job at once on the op thread. With --hmx 1 the job executes its HMX
//     instructions, which the simulator runs in the functional mode only. With --hmx 0 the job
//     returns at once, thus the timing mode measures the HVX part of the op (the HMX output tiles
//     keep their old contents and the numbers are not checked).
//
// The inputs have the layout of the Qwen3.5 graph: q, k and v are views of one conv output row of
// 2 * Hk * D + H * D floats for each token, the gate and beta are [1, H, T] rows. q and k have
// unit length unless --qknorm 1 gives the op raw rows and the parameters of the L2 norm.
//
// The check (functional mode, --hmx 1): a float64 sequential reference of the gated delta rule
// gives the attention output and the state after each of the last K tokens. The report gives the
// NMSE of each against the bound of test-backend-ops for the chunked path (2e-6).
//
// Arguments: --h 32 --hk 16 --d 128 --tokens 256 --k 1 --threads 6 --iters 1 --hmx 1 --vtcm_kb 0
//            --ver 0 --qknorm 0 --gate_min_milli -500 (the lowest log decay, in 1/1000) --check 1
//            --defer 0
// --ver sets kernel_params[1], the version of the chunked kernel (2 selects version 2, another value
// version 1). --defer 1 runs each HMX job of version 2 at its wait (refer to the shim below).
#include "lab.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// ---- the synchronous DMA shim, as in target_fa.c
#define HTP_DMA_H

typedef struct {
    void *       dst;
    const void * src;
} dma_ptr;

#define LAB_DMA_CAPACITY 1024

typedef struct dma_queue_s {
    dma_ptr  ptr[LAB_DMA_CAPACITY];
    uint32_t push_idx;
    uint32_t pop_idx;
} dma_queue;
typedef dma_queue * dma_queue_t;

static inline dma_ptr dma_make_ptr(void * dst, const void * src) {
    dma_ptr p = { dst, src };
    return p;
}

// Copies nrows rows of row_size bytes at once, then records the pointers for the pop.
static inline bool dma_queue_push(dma_queue * q, dma_ptr p, size_t dst_stride, size_t src_stride,
                                  size_t row_size, size_t nrows) {
    if (q->push_idx - q->pop_idx >= LAB_DMA_CAPACITY) {
        printf("lab: error: the DMA shim queue is full\n");
        return false;
    }
    for (size_t r = 0; r < nrows; r++) {
        memcpy((uint8_t *) p.dst + r * dst_stride, (const uint8_t *) p.src + r * src_stride, row_size);
    }
    q->ptr[q->push_idx & (LAB_DMA_CAPACITY - 1)] = p;
    q->push_idx++;
    return true;
}

static inline dma_ptr dma_queue_pop(dma_queue * q) {
    dma_ptr p = { NULL, NULL };
    if (q->pop_idx == q->push_idx) {
        return p;
    }
    p = q->ptr[q->pop_idx & (LAB_DMA_CAPACITY - 1)];
    q->pop_idx++;
    return p;
}

static inline void dma_queue_flush(dma_queue * q) {
    q->pop_idx = q->push_idx;
}

static inline bool dma_queue_empty(dma_queue * q) {
    return q->pop_idx == q->push_idx;
}

// ---- the synchronous HMX queue shim, with the fields that gdn_ch_hmx_submit reads
#define HMX_QUEUE_H

#include "hex-profile.h"

typedef void (*hmx_queue_func)(void *);

struct hmx_queue_desc {
    hmx_queue_func func;
    void *         data;
    atomic_uint    done;
};

struct hmx_queue_s {
    struct hmx_queue_desc * desc;
    atomic_uint             idx_write;
    unsigned int            idx_pop;
    uint32_t                idx_mask;
    uint32_t                capacity;
};
typedef struct hmx_queue_s * hmx_queue_t;

static int g_hmx_run = 1;

// With --defer 1 a job of version 2 runs at its wait (the hook GDN_C2_HMX_WAIT_HOOK of the kernel),
// after the HVX phase that the op thread runs beside it. Without it the job runs at the push, before
// that phase. The phone runs the two at the same time, thus a phase that writes an input of the job
// or reads an output of it gives a different result in one of the two orders.
static int            g_hmx_defer = 0;
static hmx_queue_func g_deferred_fn;
static void *         g_deferred_data;

static inline void lab_hmx_run_deferred(void) {
    if (g_deferred_fn) {
        hmx_queue_func fn = g_deferred_fn;
        g_deferred_fn = NULL;
        if (g_hmx_run) {
            fn(g_deferred_data);
        }
    }
}
#define GDN_C2_HMX_WAIT_HOOK() lab_hmx_run_deferred()

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    const unsigned int iw = atomic_load(&q->idx_write);
    q->desc[iw].func = d.func;
    q->desc[iw].data = d.data;
    if (g_hmx_defer) {
        g_deferred_fn   = d.func;
        g_deferred_data = d.data;
    } else if (g_hmx_run) {
        d.func(d.data);
    }
    atomic_store(&q->desc[iw].done, 1);
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    return true;
}

#include "gdn-chunk-ops.c"
#include "gated-delta-net-ops.c"

#define TARGET "gdnk"

static struct htp_context     g_ctx;
static dma_queue              g_dma[HTP_MAX_NTHREADS];
static struct hmx_queue_desc  g_hmx_desc[16];
static struct hmx_queue_s     g_hmx;

static void set_tensor4(struct htp_tensor * t, void * data, uint32_t ne0, uint32_t ne1, uint32_t ne2, uint32_t ne3,
                        uint32_t nb1, uint32_t nb2, uint32_t nb3) {
    memset(t, 0, sizeof(*t));
    t->data  = (uint32_t) (uintptr_t) data;
    t->type  = HTP_TYPE_F32;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = ne2;
    t->ne[3] = ne3;
    t->nb[0] = sizeof(float);
    t->nb[1] = nb1;
    t->nb[2] = nb2;
    t->nb[3] = nb3;
}

// The float64 reference of one (head) row over all tokens. state[j * D + i] = S[i][j]. The state
// after token t goes to snap[(T - 1 - t)] for the last K tokens. O(T * D * D).
static void ref_row(double * state, const float * qkv, uint32_t row_floats, uint32_t q_off, uint32_t k_off,
                    uint32_t v_off, const float * g, const float * beta, uint32_t H, uint32_t h, uint32_t T,
                    uint32_t D, uint32_t K, double scale, const double * qk_r, double * attn, double * snaps) {
    double d[128];
    for (uint32_t t = 0; t < T; t++) {
        const float * q = qkv + (size_t) t * row_floats + q_off;
        const float * k = qkv + (size_t) t * row_floats + k_off;
        const float * v = qkv + (size_t) t * row_floats + v_off;
        const double rq = qk_r[2 * t + 0];
        const double rk = qk_r[2 * t + 1];
        const double a  = exp((double) g[(size_t) t * H + h]);
        const double b  = (double) beta[(size_t) t * H + h];
        for (uint32_t j = 0; j < D; j++) {
            double * row = state + (size_t) j * D;
            double   sum = 0.0;
            for (uint32_t i = 0; i < D; i++) {
                row[i] *= a;
                sum += row[i] * (double) k[i] * rk;
            }
            d[j] = ((double) v[j] - sum) * b;
        }
        for (uint32_t j = 0; j < D; j++) {
            double * row = state + (size_t) j * D;
            double   acc = 0.0;
            for (uint32_t i = 0; i < D; i++) {
                row[i] += (double) k[i] * rk * d[j];
                acc += row[i] * (double) q[i] * rq;
            }
            attn[(size_t) t * D + j] = acc * scale;
        }
        const int64_t slot = (int64_t) T - 1 - (int64_t) t;
        if (slot >= 0 && slot < (int64_t) K) {
            memcpy(snaps + (size_t) slot * D * D, state, (size_t) D * D * sizeof(double));
        }
    }
}

int main(int argc, char ** argv) {
    const uint32_t H        = (uint32_t) lab_arg_long(argc, argv, "--h", 32);
    const uint32_t Hk       = (uint32_t) lab_arg_long(argc, argv, "--hk", 16);
    const uint32_t D        = (uint32_t) lab_arg_long(argc, argv, "--d", 128);
    const uint32_t T        = (uint32_t) lab_arg_long(argc, argv, "--tokens", 256);
    const uint32_t K        = (uint32_t) lab_arg_long(argc, argv, "--k", 1);
    const uint32_t nth      = (uint32_t) lab_arg_long(argc, argv, "--threads", 6);
    const uint32_t iters    = (uint32_t) lab_arg_long(argc, argv, "--iters", 1);
    const uint32_t ver      = (uint32_t) lab_arg_long(argc, argv, "--ver", 0);
    const uint32_t qknorm   = (uint32_t) lab_arg_long(argc, argv, "--qknorm", 0);
    const float    gate_min = (float) lab_arg_long(argc, argv, "--gate_min_milli", -500) / 1000.0f;
    const uint32_t check    = (uint32_t) lab_arg_long(argc, argv, "--check", 1);
    // The VTCM that the op gets, in KB. The standalone runtime maps the VTCM in pages of 1 MB, and an
    // HMX tile run that crosses a page raises "Coprocessor VMEM address error" (the phone maps the
    // VTCM as one page). Thus a functional run with the HMX gives the op 1 MB at a page start, and a
    // timing run without the HMX (--hmx 0) gives it the full VTCM.
    const uint32_t vtcm_kb  = (uint32_t) lab_arg_long(argc, argv, "--vtcm_kb", 0);
    g_hmx_run               = (int) lab_arg_long(argc, argv, "--hmx", 1);
    g_hmx_defer             = (int) lab_arg_long(argc, argv, "--defer", 0);

    if (D > 128 || D % 32 != 0 || H % Hk != 0 || nth == 0 || nth > HTP_MAX_NTHREADS) {
        printf("lab: %s the shape is not supported\n", TARGET);
        return 2;
    }

    lab_init();
    printf("lab: %s H %u Hk %u D %u T %u K %u threads %u ver %u qknorm %u hmx %d gate_min %.3f\n", TARGET, H, Hk, D,
           T, K, nth, ver, qknorm, g_hmx_run, (double) gate_min);

    // ---- the inputs, in the layout of the model graph
    const uint32_t row_floats = 2 * Hk * D + H * D;
    const uint32_t q_off = 0, k_off = Hk * D, v_off = 2 * Hk * D;
    float * qkv   = lab_ddr_alloc((size_t) T * row_floats * sizeof(float) + 256, 128);
    float * g     = lab_ddr_alloc((size_t) T * H * sizeof(float) + 256, 128);
    float * beta  = lab_ddr_alloc((size_t) T * H * sizeof(float) + 256, 128);
    float * s0    = lab_ddr_alloc((size_t) H * D * D * sizeof(float) + 256, 128);
    const size_t n_attn = (size_t) T * H * D;
    const size_t n_snap = (size_t) K * H * D * D;
    float * dst   = lab_ddr_alloc((n_attn + n_snap) * sizeof(float) + 256, 128);

    lab_fill_f32(qkv, (size_t) T * row_floats, -1.0f, 1.0f);
    for (uint32_t t = 0; t < T; t++) {
        float * v = qkv + (size_t) t * row_floats + v_off;
        for (uint32_t i = 0; i < H * D; i++) {
            v[i] = lab_rand_f32(-0.3f, 5.0f);
        }
    }
    lab_fill_f32(g, (size_t) T * H, gate_min, -1e-4f);
    lab_fill_f32(beta, (size_t) T * H, 0.0f, 1.0f);
    lab_fill_f32(s0, (size_t) H * D * D, -0.5f, 0.5f);

    // The per-token factors of the L2 norm of each q and k head, for the reference. Without
    // --qknorm the rows are normalized here and the factor is 1.
    const float l2_eps = 1e-6f;
    double * qk_r = lab_ddr_alloc((size_t) T * Hk * 2 * sizeof(double), 128);
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t hk = 0; hk < Hk; hk++) {
            for (int w = 0; w < 2; w++) {
                float * x = qkv + (size_t) t * row_floats + (w ? k_off : q_off) + hk * D;
                double  s = 0.0;
                for (uint32_t i = 0; i < D; i++) {
                    s += (double) x[i] * x[i];
                }
                const double r = 1.0 / fmax(sqrt(s), (double) l2_eps);
                if (qknorm) {
                    qk_r[((size_t) t * Hk + hk) * 2 + w] = r;
                } else {
                    for (uint32_t i = 0; i < D; i++) {
                        x[i] = (float) ((double) x[i] * r);
                    }
                    qk_r[((size_t) t * Hk + hk) * 2 + w] = 1.0;
                }
            }
        }
    }

    // ---- the op context
    struct htp_tensor tq, tk, tv, tg, tb, ts, td;
    const uint32_t nb2 = row_floats * sizeof(float);
    set_tensor4(&tq, qkv + q_off, D, Hk, T, 1, D * sizeof(float), nb2, nb2 * T);
    set_tensor4(&tk, qkv + k_off, D, Hk, T, 1, D * sizeof(float), nb2, nb2 * T);
    set_tensor4(&tv, qkv + v_off, D, H, T, 1, D * sizeof(float), nb2, nb2 * T);
    set_tensor4(&tg, g, 1, H, T, 1, sizeof(float), H * sizeof(float), H * T * sizeof(float));
    set_tensor4(&tb, beta, 1, H, T, 1, sizeof(float), H * sizeof(float), H * T * sizeof(float));
    set_tensor4(&ts, s0, D, D, H, 1, D * sizeof(float), D * D * sizeof(float), H * D * D * sizeof(float));
    set_tensor4(&td, dst, D * H, T + K * D, 1, 1, D * H * sizeof(float), (T + K * D) * D * H * sizeof(float),
                (T + K * D) * D * H * sizeof(float));

    memset(&g_ctx, 0, sizeof(g_ctx));
    for (uint32_t i = 0; i < HTP_MAX_NTHREADS; i++) {
        g_ctx.dma[i]        = &g_dma[i];
        g_ctx.dma_cached[i] = &g_dma[i];
    }
    memset(g_hmx_desc, 0, sizeof(g_hmx_desc));
    g_hmx.desc     = g_hmx_desc;
    g_hmx.idx_mask = 15;
    g_hmx.capacity = 16;
    g_ctx.hmx_queue   = &g_hmx;
    g_ctx.hmx_enabled = true;
    g_ctx.n_threads   = nth;
    g_ctx.n_threads_div = init_fastdiv_values(nth);
    if (vtcm_kb > 0) {
        g_ctx.vtcm_size = (size_t) vtcm_kb * 1024;
        g_ctx.vtcm_base = lab_vtcm_alloc(g_ctx.vtcm_size, 1024 * 1024);
    } else {
        g_ctx.vtcm_size = lab_vtcm_size() - 4096;
        g_ctx.vtcm_base = lab_vtcm_alloc(g_ctx.vtcm_size, 2048);
    }
    g_ctx.mdev.count  = 1;

    struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx           = &g_ctx;
    octx.op            = HTP_OP_GATED_DELTA_NET;
    octx.n_threads     = nth;
    octx.n_threads_div = g_ctx.n_threads_div;
    octx.op_params[0]  = (int32_t) K;
    octx.src[0] = &tq;
    octx.src[1] = &tk;
    octx.src[2] = &tv;
    octx.src[3] = &tg;
    octx.src[4] = &tb;
    octx.src[5] = &ts;
    octx.dst    = &td;
    octx.kernel_params[0] = 0;
    octx.kernel_params[1] = (int32_t) ver;
    if (qknorm) {
        // the L2_NORM form: r = 1 / sqrt(max(sum, eps^2))
        const float a = 1.0f, b = 0.0f, c = l2_eps * l2_eps, s = 1.0f;
        octx.kernel_params[2] = 1;
        memcpy(&octx.kernel_params[3], &a, 4);
        memcpy(&octx.kernel_params[4], &b, 4);
        memcpy(&octx.kernel_params[5], &c, 4);
        memcpy(&octx.kernel_params[6], &s, 4);
    }

    uint64_t best = UINT64_MAX;
    int status = 0;
    for (uint32_t it = 0; it < iters + 1; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        status = op_gated_delta_net(&octx);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (it > 0 || iters == 0) {
            best = (t1 - t0) < best ? (t1 - t0) : best;
        }
        if (status != HTP_STATUS_OK) {
            printf("lab: %s the op returned %d (detail %u)\n", TARGET, status, octx.err_detail);
            return 1;
        }
    }
    const double row_chunks = (double) H * ((T - (K > 1 ? K : 0) + 63) / 64);
    lab_report(TARGET, "cycles_per_op", (double) best, "cycles");
    lab_report(TARGET, "us_per_op_at_2112_mhz", (double) best / 2112.0, "us");
    lab_report(TARGET, "cycles_per_row_chunk", (double) best / row_chunks, "cycles");
    lab_report(TARGET, "ms_per_1024_tokens_24_layers", (double) best / 2112.0 / 1000.0 * 24.0 * 1024.0 / T, "ms");

    if (!check || !g_hmx_run) {
        lab_report(TARGET, "checked", 0, "");
        return 0;
    }

    // ---- the float64 check
    const double scale = 1.0 / sqrt((double) D);
    double * st    = lab_ddr_alloc((size_t) D * D * sizeof(double), 128);
    double * attn  = lab_ddr_alloc((size_t) T * D * sizeof(double), 128);
    double * snaps = lab_ddr_alloc((size_t) K * D * D * sizeof(double), 128);
    double * rr    = lab_ddr_alloc((size_t) T * 2 * sizeof(double), 128);
    double se_a = 0.0, sr_a = 0.0, se_s = 0.0, sr_s = 0.0, se_k = 0.0, sr_k = 0.0;
    size_t n_bad = 0;
    for (uint32_t h = 0; h < H; h++) {
        const uint32_t hk = h % Hk;
        for (uint32_t t = 0; t < T; t++) {
            rr[2 * t + 0] = qk_r[((size_t) t * Hk + hk) * 2 + 0];
            rr[2 * t + 1] = qk_r[((size_t) t * Hk + hk) * 2 + 1];
        }
        for (size_t i = 0; i < (size_t) D * D; i++) {
            st[i] = (double) s0[(size_t) h * D * D + i];
        }
        ref_row(st, qkv, row_floats, q_off + hk * D, k_off + hk * D, v_off + h * D, g, beta, H, h, T, D, K, scale, rr,
                attn, snaps);
        for (uint32_t t = 0; t < T; t++) {
            for (uint32_t j = 0; j < D; j++) {
                const double got = (double) dst[((size_t) t * H + h) * D + j];
                const double ref = attn[(size_t) t * D + j];
                if (!isfinite(got)) {
                    n_bad++;
                }
                se_a += (got - ref) * (got - ref);
                sr_a += ref * ref;
            }
        }
        const uint32_t n_slots = K < T ? K : T;
        for (uint32_t s = 0; s < n_slots; s++) {
            const float * got = dst + n_attn + (size_t) s * H * D * D + (size_t) h * D * D;
            for (size_t i = 0; i < (size_t) D * D; i++) {
                const double ref = snaps[(size_t) s * D * D + i];
                const double d   = (double) got[i] - ref;
                if (!isfinite((double) got[i])) {
                    n_bad++;
                }
                if (s == 0) {
                    se_s += d * d;
                    sr_s += ref * ref;
                } else {
                    se_k += d * d;
                    sr_k += ref * ref;
                }
            }
        }
    }
    const double nmse_a = sr_a > 0 ? se_a / sr_a : 0.0;
    const double nmse_s = sr_s > 0 ? se_s / sr_s : 0.0;
    const double nmse_k = sr_k > 0 ? se_k / sr_k : 0.0;
    lab_report(TARGET, "nmse_attn", nmse_a, "");
    lab_report(TARGET, "nmse_state", nmse_s, "");
    lab_report(TARGET, "nmse_slots_above_0", nmse_k, "");
    lab_report(TARGET, "not_finite", (double) n_bad, "");
    const bool pass = n_bad == 0 && nmse_a < 2e-6 && nmse_s < 2e-6 && nmse_k < 2e-6;
    printf("lab: %s check %s (bound 2e-6)\n", TARGET, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
