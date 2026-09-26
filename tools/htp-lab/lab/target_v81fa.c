// Target v81fa: the whole HMX flash attention op (op_flash_attn_ext of flash-attn-ops.c with the kernel params of
// the host) for the shapes of Qwen3.5 4B and of its vision encoder, with an FNV-1a hash of each output.
//
// The program includes flash-attn-ops.c verbatim and calls op_flash_attn_ext with the params that
// ggml_hexagon_precompute_flash_attn_params gives for the HMX path (hmx_fa_find_chunk_size and the fields of
// that function). Two shims replace the engines that the standalone runtime of the simulator does not give:
//   - The DMA (lab-dma.h): a push copies at once, a pop returns the destinations in the order of the pushes, and
//     the line cache of the mask keeps the replacement rule of dma-queue.h.
//   - The HMX queue (lab-hmx.h): a push runs the job at once on the thread of the push.
// The ops get the first 1 MB of the VTCM: the HMX model of the simulator stops with the exception 0x26 when one
// deep tile load crosses a multiple of 1 MB above the VTCM base. Thus the blocks (Br, Bc) are those of a 1 MB
// budget, the same on each core. Run it in the functional mode (MODE=functional).
//
// The cases:
//   dec256q8   decode: 1 token, 16 heads on 4 KV heads, head size 256, 512 KV rows of Q8_0, F32 Q, a mask
//   dec256f16  the same with F16 K and V
//   pre256q8   prefill: 64 tokens, the same heads, 256 KV rows of Q8_0, a causal mask
//   vis64      vision: 64 tokens, 4 heads on 4 KV heads, head size 64, 64 KV rows of F16, no mask
// Each case prints "hash = 0x..." of its output bytes and the largest error against a float64 reference of the
// attention with the dequantized K and V and the F32 Q, relative to the largest magnitude of the reference. The
// reference is not the naive CPU oracle (that one converts Q to F16 for an F16 K and to Q8_0 for a Q8_0 K), thus
// the error of a correct kernel is the F16 rounding of Q and of the probabilities, about 1e-3.
//
// Arguments: --threads 4 --vtcm 1048576 --cases all
//   --cases  the names of the cases to run, separated by commas, or "all". The functional simulator runs the
//            decode cases in minutes and the prefill case pre256q8 in hours, thus the registry runs the others.
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
// lab-run: mode=functional args=--cases dec256q8,dec256f16,vis64

#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lab-dma.h"
#include "lab-hmx.h"

#include "hex-profile.h"

#include "flash-attn-ops.c"

#define TARGET "v81fa"

static struct htp_context g_ctx;
static dma_queue          g_dma[HTP_MAX_NTHREADS];
static struct hmx_queue_s g_hmx;
static size_t             g_fail = 0;

// The FNV-1a hash of n bytes. O(n).
static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = p;
    uint64_t        h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 0x100000001B3ull;
    }
    return h;
}

static void set_tensor(struct htp_tensor * t, void * data, uint32_t type, uint32_t ne0, uint32_t ne1, uint32_t ne2,
                       uint32_t nb1, uint32_t nb2) {
    memset(t, 0, sizeof(*t));
    t->data  = (uint32_t) (uintptr_t) data;
    t->type  = type;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = ne2;
    t->ne[3] = 1;
    t->nb[0] = type == HTP_TYPE_F32 ? 4 : (type == HTP_TYPE_Q8_0 ? sizeof(block_q8_0) : 2);
    t->nb[1] = nb1;
    t->nb[2] = nb2;
    t->nb[3] = nb2 * ne2;
    t->size  = nb2 * ne2;
}

// The kernel params of the HMX path, as ggml_hexagon_precompute_flash_attn_params of the host gives them.
// Returns false when no block fits the VTCM budget.
static bool fa_params(uint32_t G, uint32_t DK, uint32_t DV, uint32_t neq1, uint32_t nek1, uint32_t n_head,
                      bool has_mask, uint32_t mask_ne2, float scale, struct htp_fa_kernel_params * kp) {
    memset(kp, 0, sizeof(*kp));
    kp->scale         = scale;
    kp->max_bias      = 0.0f;
    kp->logit_softcap = 0.0f;
    kp->is_q_fp32     = 1;
    kp->is_dst_fp32   = 1;
    kp->G             = G;
    kp->n_head_log2   = 1u << (uint32_t) floorf(log2f((float) n_head));
    kp->m0            = 1.0f;
    kp->m1            = 1.0f;
    size_t     Br = 0, Bc = 0;
    const int  r  = hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, nek1, g_ctx.vtcm_size, g_ctx.n_threads, true);
    if (r != 0) {
        return false;
    }
    kp->kernel_type        = HTP_FA_KERNEL_HMX;
    kp->Br                 = Br;
    kp->Bc                 = Bc;
    kp->n_kv_blocks        = (nek1 + Bc - 1) / Bc;
    kp->n_threads          = (kp->n_kv_blocks >= 3 && g_ctx.n_threads >= 2) ? g_ctx.n_threads : 1;
    kp->u.hmx.g_br         = hex_align_up(G * Br, 32);
    kp->u.hmx.pipeline     = (kp->n_kv_blocks >= 3 && g_ctx.n_threads >= 2) ? 1 : 0;
    kp->vtcm_size          = hmx_fa_compute_vtcm_usage(G, DK, DV, Br, Bc, kp->n_threads, kp->u.hmx.pipeline != 0, true);
    kp->u.hmx.row_buf_stride      = hex_align_up(Bc * sizeof(uint16_t), 256) / 128;
    kp->u.hmx.mask_buf_row_stride = hex_align_up(Bc * sizeof(uint16_t), 128) / sizeof(uint16_t);
    kp->u.hmx.mask_broadcast      = (has_mask && mask_ne2 == 1) ? 1 : 0;
    kp->u.hmx.div_G               = init_fastdiv_values(G);
    if (has_mask) {
        kp->src3_div2 = init_fastdiv_values(mask_ne2);
        kp->src3_div3 = init_fastdiv_values(1);
    }
    printf("lab: %s params G %u DK %u tokens %u kv %u: Br %zu Bc %zu blocks %u threads %u pipeline %d vtcm %u\n", TARGET,
           G, DK, neq1, nek1, Br, Bc, (unsigned) kp->n_kv_blocks, (unsigned) kp->n_threads, kp->u.hmx.pipeline,
           (unsigned) kp->vtcm_size);
    return kp->vtcm_size <= g_ctx.vtcm_size;
}

// The value of element d of row j of a K or V head in the buffer of the case (F16 or Q8_0)
static double kv_at(const uint8_t * base, bool q8, uint32_t D, size_t row_bytes, uint32_t j, uint32_t d) {
    const uint8_t * row = base + (size_t) j * row_bytes;
    if (!q8) {
        return (double) ((const __fp16 *) row)[d];
    }
    const block_q8_0 * b = (const block_q8_0 *) row + d / QK8_0;
    return (double) b->qs[d % QK8_0] * (double) lab_hf_to_f32(b->d);
}

// One case: fills the tensors, runs the op, prints the hash and the error against the float64 reference.
// O(tokens * heads * kv * D) for the reference.
static void run_case(const char * name, uint32_t n_tokens, uint32_t n_heads, uint32_t n_kv_heads, uint32_t D,
                     uint32_t n_kv, bool q8, bool mask_on, bool causal) {
    const uint32_t G         = n_heads / n_kv_heads;
    const size_t   row_bytes = q8 ? (size_t) (D / QK8_0) * sizeof(block_q8_0) : (size_t) D * 2;
    const float    scale     = 1.0f / sqrtf((float) D);

    float *   q    = lab_ddr_alloc((size_t) D * n_tokens * n_heads * 4, 128);
    uint8_t * k    = lab_ddr_alloc(row_bytes * n_kv * n_kv_heads + 128, 128);
    uint8_t * v    = lab_ddr_alloc(row_bytes * n_kv * n_kv_heads + 128, 128);
    __fp16 *  mask = lab_ddr_alloc((size_t) n_kv * n_tokens * 2 + 128, 128);
    float *   out  = lab_ddr_alloc((size_t) D * n_heads * n_tokens * 4 + 128, 128);

    lab_fill_f32(q, (size_t) D * n_tokens * n_heads, -1.0f, 1.0f);
    for (size_t r = 0; r < (size_t) n_kv * n_kv_heads; r++) {
        uint8_t * kr = k + r * row_bytes;
        uint8_t * vr = v + r * row_bytes;
        if (q8) {
            for (uint32_t b = 0; b < D / QK8_0; b++) {
                block_q8_0 * kb = (block_q8_0 *) kr + b;
                block_q8_0 * vb = (block_q8_0 *) vr + b;
                kb->d           = lab_f32_to_hf(lab_rand_f32(0.004f, 0.02f));
                vb->d           = lab_f32_to_hf(lab_rand_f32(0.004f, 0.02f));
                for (int i = 0; i < QK8_0; i++) {
                    kb->qs[i] = (int8_t) (lab_rand_u32() % 255 - 127);
                    vb->qs[i] = (int8_t) (lab_rand_u32() % 255 - 127);
                }
            }
        } else {
            for (uint32_t d = 0; d < D; d++) {
                ((uint16_t *) kr)[d] = lab_f32_to_hf(lab_rand_f32(-1.0f, 1.0f));
                ((uint16_t *) vr)[d] = lab_f32_to_hf(lab_rand_f32(-1.0f, 1.0f));
            }
        }
    }
    // The mask of the last tokens of a sequence: token t sees the KV rows up to n_kv - n_tokens + t
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t j = 0; j < n_kv; j++) {
            const bool visible = !causal || j <= n_kv - n_tokens + t;
            mask[(size_t) t * n_kv + j] = visible ? (__fp16) 0.0f : (__fp16) -INFINITY;
        }
    }
    memset(out, 0, (size_t) D * n_heads * n_tokens * 4);

    struct htp_fa_kernel_params kp;
    if (!fa_params(G, D, D, n_tokens, n_kv, n_heads, mask_on, 1, scale, &kp)) {
        printf("lab: %s %s: no block fits the VTCM\n", TARGET, name);
        g_fail++;
        return;
    }

    struct htp_tensor tq, tk, tv, tm, td;
    set_tensor(&tq, q, HTP_TYPE_F32, D, n_tokens, n_heads, D * 4, D * 4 * n_tokens);
    set_tensor(&tk, k, q8 ? HTP_TYPE_Q8_0 : HTP_TYPE_F16, D, n_kv, n_kv_heads, row_bytes, row_bytes * n_kv);
    set_tensor(&tv, v, q8 ? HTP_TYPE_Q8_0 : HTP_TYPE_F16, D, n_kv, n_kv_heads, row_bytes, row_bytes * n_kv);
    set_tensor(&tm, mask, HTP_TYPE_F16, n_kv, n_tokens, 1, n_kv * 2, n_kv * 2 * n_tokens);
    set_tensor(&td, out, HTP_TYPE_F32, D, n_heads, n_tokens, D * 4, D * 4 * n_heads);

    struct htp_ops_context * octx = &g_ctx.octx;
    memset(octx, 0, sizeof(*octx));
    octx->ctx           = &g_ctx;
    octx->op            = HTP_OP_FLASH_ATTN_EXT;
    octx->n_threads     = g_ctx.n_threads;
    octx->n_threads_div = g_ctx.n_threads_div;
    memcpy(octx->kernel_params, &kp, sizeof(kp));
    octx->src[0] = &tq;
    octx->src[1] = &tk;
    octx->src[2] = &tv;
    octx->src[3] = mask_on ? &tm : NULL;
    octx->dst    = &td;
    const int st = op_flash_attn_ext(octx);
    if (st != HTP_STATUS_OK) {
        printf("lab: %s %s: the op gave the status %d\n", TARGET, name, st);
        g_fail++;
    }

    // The float64 reference: softmax(scale * q k^T + mask) v, for each token and head
    double   max_ref = 0.0, max_err = 0.0;
    size_t   n_bad   = 0;
    double * s       = malloc((size_t) n_kv * sizeof(double));
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_heads; h++) {
            const float *   qr  = q + ((size_t) h * n_tokens + t) * D;
            const uint8_t * kh  = k + (size_t) (h / G) * n_kv * row_bytes;
            const uint8_t * vh  = v + (size_t) (h / G) * n_kv * row_bytes;
            double          m   = -INFINITY;
            for (uint32_t j = 0; j < n_kv; j++) {
                double dot = 0.0;
                for (uint32_t d = 0; d < D; d++) {
                    dot += (double) qr[d] * kv_at(kh, q8, D, row_bytes, j, d);
                }
                s[j] = dot * scale + (mask_on ? (double) mask[(size_t) t * n_kv + j] : 0.0);
                m    = s[j] > m ? s[j] : m;
            }
            double sum = 0.0;
            for (uint32_t j = 0; j < n_kv; j++) {
                s[j] = exp(s[j] - m);
                sum += s[j];
            }
            const float * o = out + ((size_t) t * n_heads + h) * D;
            for (uint32_t d = 0; d < D; d++) {
                double acc = 0.0;
                for (uint32_t j = 0; j < n_kv; j++) {
                    acc += s[j] * kv_at(vh, q8, D, row_bytes, j, d);
                }
                const double ref = acc / sum;
                max_ref          = fabs(ref) > max_ref ? fabs(ref) : max_ref;
                const double err = isfinite(o[d]) ? fabs((double) o[d] - ref) : INFINITY;
                max_err          = err > max_err ? err : max_err;
                n_bad += isfinite(o[d]) ? 0 : 1;
            }
        }
    }
    free(s);
    const double rel = max_ref > 0.0 ? max_err / max_ref : max_err;
    printf("lab: %s %s hash = 0x%016llx fnv1a\n", TARGET, name,
           (unsigned long long) fnv1a(out, (size_t) D * n_heads * n_tokens * 4));
    printf("lab: %s %s max_err %.6e of max_ref %.6e (relative %.3e), not finite %zu\n", TARGET, name, max_err, max_ref,
           rel, n_bad);
    if (!(rel < 2e-2) || n_bad) {
        g_fail++;
    }
    free(q);
    free(k);
    free(v);
    free(mask);
    free(out);
}

// Returns true when the list of case names holds the name, or when the list is "all". The names in the list
// are separated by commas. Complexity O(length of the list).
static bool case_selected(const char * list, const char * name) {
    if (strcmp(list, "all") == 0) {
        return true;
    }
    const size_t n = strlen(name);
    for (const char * p = list; *p != '\0';) {
        const char * end = strchr(p, ',');
        const size_t len = end ? (size_t) (end - p) : strlen(p);
        if (len == n && strncmp(p, name, n) == 0) {
            return true;
        }
        p += len + (end ? 1 : 0);
    }
    return false;
}

int main(int argc, char ** argv) {
    lab_init();
    const uint32_t     nt    = (uint32_t) lab_arg_long(argc, argv, "--threads", 4);
    const char * const cases = lab_arg_str(argc, argv, "--cases", "all");

    g_ctx.vtcm_base     = lab_vtcm_base();
    g_ctx.vtcm_size     = (size_t) lab_arg_long(argc, argv, "--vtcm", 1 << 20);
    lab_args_done(argc, argv);
    g_ctx.n_threads     = nt;
    g_ctx.n_threads_div = init_fastdiv_values(nt);
    g_ctx.work_queue    = (work_queue_t) &g_hmx;  // the lab stub does not read it
    g_ctx.hmx_queue     = &g_hmx;
    g_ctx.hmx_enabled   = true;
    g_ctx.mdev.count    = 1;
    for (uint32_t i = 0; i < HTP_MAX_NTHREADS; i++) {
        g_ctx.dma[i]        = &g_dma[i];
        g_ctx.dma_cached[i] = &g_dma[i];
    }
    printf("lab: %s threads %u vtcm %zu\n", TARGET, nt, g_ctx.vtcm_size);

    if (case_selected(cases, "dec256q8")) {
        run_case("dec256q8", 1, 16, 4, 256, 512, true, true, true);
    }
    if (case_selected(cases, "dec256f16")) {
        run_case("dec256f16", 1, 16, 4, 256, 512, false, true, true);
    }
    if (case_selected(cases, "pre256q8")) {
        run_case("pre256q8", 64, 16, 4, 256, 256, true, true, true);
    }
    if (case_selected(cases, "vis64")) {
        run_case("vis64", 64, 4, 4, 64, 64, false, false, false);
    }
    if (strcmp(cases, "all") != 0) {
        lab_limit("--cases selects a part of the cases of this target, thus the other cases did not run.");
    }

    lab_report(TARGET, "failures", (double) g_fail, "cases");
    // On the chip a push to a second DMA ring of one thread stops the op. A job that no pop took is
    // an error of the kernel. Thus the two counts go into the failure count of the run.
    g_fail += lab_dma_report(TARGET);
    g_fail += lab_hmx_report(TARGET, &g_hmx);
    printf("lab: check %s %s\n", TARGET, g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
