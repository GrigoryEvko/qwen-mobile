// Target: the HVX decode ops of matmul-ops.c with Q8_0 weights, run as a full op, with an FNV-1a hash
// of each output.
//
// The program includes matmul-ops.c verbatim and calls the op entry points op_matmul, op_matmul_nx and
// op_matmul_id with the kernel params that the host computes for the HVX path
// (ggml_hexagon_precompute_hvx_mm_params). Two shims replace the engines that the standalone runtime of
// the simulator does not give (lab-dma.h and lab-hmx.h):
//   - The DMA: a push copies the rows of the descriptor at once (dst stride, src stride, row size, row
//     count), and a pop returns the destinations in the order of the pushes. Thus a descriptor with a
//     wrong geometry puts wrong bytes into VTCM, and the output changes.
//   - The HMX queue: the HVX paths do not use it. The shim lets matmul-ops.c link.
//
// Each case prints "hash = 0x..." of all its output bytes. The inputs come from a fixed seed, thus two
// kernel trees give the same bits when they print the same hashes. Each case also checks the output
// against a float64 reference with a loose bound, which finds a broken case setup.
//
// The cases: the 4B decode shapes of MUL_MAT (with and without the fused ADD), the 2 to 4 rows of the
// multi-row path and of the row-pair path, MUL_MAT_NX with two and three weights, MUL_MAT_ID, weight row
// counts that are not a multiple of 32, and k-tile counts that are odd. Run it in the functional mode
// of the simulator (MODE=functional).
//
// The heap of the simulator holds about 25 MB of weights for one case, thus the MUL_MAT_NX cases use
// 2048 weight rows in place of the 9216 rows of the gate and the up weight of the 4B. The phone tool
// tools/gemv/gemvcheck runs the full 4B shapes.
//
// Arguments: --threads 4. With 6 threads the lab runtime does not start the sixth HVX thread, and
// the op waits for it.
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
// lab-run: mode=functional

#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The MUL_MAT_ID path of matmul-ops.c calls memalign. The standalone runtime has no malloc.h.
void * memalign(size_t alignment, size_t size);

#include "lab-dma.h"
#include "lab-hmx.h"

#include "hex-profile.h"

#include "matmul-ops.c"

#define TARGET "gemvop"
#define TILE   HTP_MM_WEIGHT_TILE_SIZE_Q8_0

static struct htp_context g_ctx;
static dma_queue          g_dma[HTP_MAX_NTHREADS];
static struct hmx_queue_s g_hmx;
static size_t             g_fail  = 0;
static size_t             g_cases = 0;

static void set_tensor(struct htp_tensor * t, void * data, uint32_t type, uint32_t ne0, uint32_t ne1, uint32_t ne2,
                       uint32_t nb0, uint32_t nb1) {
    memset(t, 0, sizeof(*t));
    t->data  = (uint32_t) (uintptr_t) data;
    t->type  = type;
    t->ne[0] = ne0;
    t->ne[1] = ne1;
    t->ne[2] = ne2;
    t->ne[3] = 1;
    t->nb[0] = nb0;
    t->nb[1] = nb1;
    t->nb[2] = nb1 * ne1;
    t->nb[3] = nb1 * ne1 * ne2;
    t->size  = nb1 * ne1 * ne2;
}

// The FNV-1a hash of n bytes, continued from h. O(n).
static uint64_t fnv1a(const void * p, size_t n, uint64_t h) {
    const uint8_t * b = p;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 0x100000001B3ull;
    }
    return h;
}

// A repacked Q8_0 weight of n_pad rows (a multiple of 32) and k columns: n_pad / 32 column tiles of
// k / 32 tiles of 1088 bytes. Random quants and scales in [-1/64, 1/64]. O(bytes).
static uint8_t * make_weight(uint32_t k, uint32_t n_pad) {
    const size_t bytes = (size_t) (n_pad / 32) * (k / 32) * TILE;
    uint8_t *    w     = lab_ddr_alloc(bytes, 128);
    for (size_t t = 0; t < bytes / TILE; t++) {
        uint8_t * tile = w + t * TILE;
        lab_fill_u8(tile, 1024);
        uint16_t * sc = (uint16_t *) (tile + 1024);
        for (int j = 0; j < 32; j++) {
            sc[j] = lab_f32_to_hf(lab_rand_f32(-1.0f / 64.0f, 1.0f / 64.0f));
        }
    }
    return w;
}

// The value of weight row r, column c of a repacked Q8_0 weight (the layout of target_q8.c)
static double weight_at(const uint8_t * w, uint32_t k, uint32_t r, uint32_t c) {
    const uint32_t  nkt  = k / 32;
    const uint8_t * tile = w + ((size_t) (r / 32) * nkt + c / 32) * TILE;
    const uint32_t  row  = r % 32;
    const uint32_t  kk   = c % 32;
    const int8_t    q    = (int8_t) tile[(kk / 2) * 64 + 2 * row + (kk % 2)];
    uint16_t        d;
    memcpy(&d, tile + 1024 + 2 * row, 2);
    return (double) q * (double) lab_hf_to_f32(d);
}

// Checks out (m rows of n values, row stride n) against the float64 reference x * w^T (+ add), with a
// bound of 3 % of the largest reference value: the activation is quantized to 8 bits. The check reads
// the first 64 and the last 32 values of each row, thus the first column tiles and the last partial
// one. The hash covers all values. O(m * 96 * k).
static void check_ref(const char * what, const float * out, const uint8_t * w, const float * x, const float * add,
                      uint32_t k, uint32_t n, uint32_t m) {
    double max_ref = 0.0, max_err = 0.0;
    for (uint32_t r = 0; r < m; r++) {
        for (uint32_t c = 0; c < n; c++) {
            if (c >= 64 && c + 32 < n) {
                c = n - 32;
            }
            double ref = add ? (double) add[(size_t) r * n + c] : 0.0;
            for (uint32_t j = 0; j < k; j++) {
                ref += weight_at(w, k, c, j) * (double) x[(size_t) r * k + j];
            }
            const double err = fabs((double) out[(size_t) r * n + c] - ref);
            max_ref = fabs(ref) > max_ref ? fabs(ref) : max_ref;
            max_err = err > max_err ? err : max_err;
        }
    }
    const bool ok = max_err <= 0.03 * max_ref;
    printf("lab: %s %s reference max_err %.4g max_ref %.4g %s\n", TARGET, what, max_err, max_ref, ok ? "ok" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

// The kernel params of the HVX path: the kernel type, the largest prefetch depth (16 down to 2) whose
// layout fits the VTCM, and the VTCM size, as ggml_hexagon_precompute_hvx_mm_params and
// ggml_hexagon_precompute_fused_mmnx_params give them. Returns false when no layout fits.
static bool hvx_params(uint32_t k, uint32_t n_pad, uint32_t m, int kernel_type, uint32_t n_weights, bool is_id,
                       bool with_add, struct htp_mm_kernel_params * kp) {
    memset(kp, 0, sizeof(*kp));
    const uint32_t nt       = g_ctx.n_threads;
    const size_t   act_row  = htp_mm_q8_0_tiled_row_size(k);
    const size_t   src0_row = (size_t) (k / 32) * sizeof(block_q8_0);
    const bool     is_nx    = n_weights > 0;
    struct htp_mm_hvx_vtcm_layout L;
    uint32_t pf = 0;
    for (uint32_t d = (m > HTP_MM_HMX_MIN_NROWS) ? 2 : 16; d >= 2; d /= 2) {
        htp_mm_hvx_vtcm_layout_build(&L, kernel_type, HTP_TYPE_Q8_0, k, m, nt, is_nx || is_id ? 0 : (size_t) n_pad * 4,
                                     src0_row, act_row, with_add ? (size_t) n_pad * 4 : 0, d, is_id, is_nx);
        if (L.total_bytes <= g_ctx.vtcm_size) {
            pf = d;
            break;
        }
    }
    if (pf == 0) {
        return false;
    }
    kp->kernel_type       = kernel_type;
    kp->n_threads         = (int32_t) nt;
    kp->n_prefetch        = (int32_t) pf;
    kp->tile_size         = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    kp->aligned_tile_size = HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0;
    kp->src1_row_size     = (int32_t) act_row;
    kp->vtcm_size         = (int32_t) L.total_bytes;
    kp->vtcm_src0_size    = (int32_t) L.src0_bytes;
    kp->vtcm_src1_size    = (int32_t) L.src1_bytes;
    kp->vtcm_dst_size     = (int32_t) L.dst_bytes;
    kp->n_weights         = (int32_t) n_weights;
    kp->div_ne11          = init_fastdiv_values(m);
    return true;
}

static int run_op(uint32_t op, const struct htp_mm_kernel_params * kp, struct htp_tensor * src, unsigned n_src,
                  struct htp_tensor * dst, unsigned n_dst) {
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
    octx->dst = &dst[0];
    for (unsigned i = 0; i < n_dst; i++) {
        octx->dsts[i] = &dst[i];
    }
    int s;
    switch (op) {
        case HTP_OP_MUL_MAT_NX: s = op_matmul_nx(octx); break;
        case HTP_OP_MUL_MAT_ID: s = op_matmul_id(octx); break;
        default:                s = op_matmul(octx); break;
    }
    if (s != HTP_STATUS_OK) {
        printf("lab: error: op %u returned status %d\n", op, s);
        g_fail++;
    }
    return s;
}

static const char * kernel_name(int kernel_type) {
    switch (kernel_type) {
        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:    return "block";
        case HTP_MM_KERNEL_HVX_QUANT_ROW:      return "row";
        case HTP_MM_KERNEL_HVX_QUANT_MULTIROW: return "multirow";
        default:                               return "other";
    }
}

static void report_hash(const char * what, const void * out, size_t bytes) {
    printf("lab: %s %s hash = 0x%016llx fnv1a\n", TARGET, what,
           (unsigned long long) fnv1a(out, bytes, 0xCBF29CE484222325ull));
}

// MUL_MAT (MUL_MAT_ADD with add) of a Q8_0 weight of n rows and k columns with m activation rows
static void run_mm(uint32_t k, uint32_t n, uint32_t m, int kernel_type, bool with_add) {
    const uint32_t n_pad = hex_round_up(n, 32);
    char what[96];
    snprintf(what, sizeof(what), "mm%s_k%u_n%u_m%u_%s", with_add ? "add" : "", k, n, m, kernel_name(kernel_type));
    g_cases++;

    uint8_t * w   = make_weight(k, n_pad);
    float *   x   = lab_ddr_alloc((size_t) m * k * 4, 128);
    float *   add = lab_ddr_alloc((size_t) m * n * 4, 128);
    float *   out = lab_ddr_alloc((size_t) m * n * 4 + 256, 128);
    lab_fill_f32(x, (size_t) m * k, -1.0f, 1.0f);
    lab_fill_f32(add, (size_t) m * n, -1.0f, 1.0f);

    struct htp_mm_kernel_params kp;
    if (!hvx_params(k, n_pad, m, kernel_type, 0, false, with_add, &kp)) {
        printf("lab: error: %s: no layout\n", what);
        g_fail++;
        return;
    }
    struct htp_tensor src[3], dst;
    set_tensor(&src[0], w, HTP_TYPE_Q8_0, k, n_pad, 1, sizeof(block_q8_0), (k / 32) * sizeof(block_q8_0));
    set_tensor(&src[1], x, HTP_TYPE_F32, k, m, 1, 4, 4 * k);
    set_tensor(&src[2], add, HTP_TYPE_F32, n, m, 1, 4, 4 * n);
    set_tensor(&dst, out, HTP_TYPE_F32, n, m, 1, 4, 4 * n);
    const uint64_t pushes = lab_dma_pushes();
    const uint64_t rows   = lab_dma_rows();
    run_op(with_add ? HTP_OP_MUL_MAT_ADD : HTP_OP_MUL_MAT, &kp, src, with_add ? 3 : 2, &dst, 1);
    printf("lab: %s %s prefetch %d dma descriptors %llu rows %llu\n", TARGET, what, kp.n_prefetch,
           (unsigned long long) (lab_dma_pushes() - pushes), (unsigned long long) (lab_dma_rows() - rows));
    report_hash(what, out, (size_t) m * n * 4);
    check_ref(what, out, w, x, with_add ? add : NULL, k, n, m);
    free(w);
    free(x);
    free(add);
    free(out);
}

// MUL_MAT_NX of n_w Q8_0 weights of the rows n[i] and k columns with m activation rows
static void run_nx(uint32_t k, const uint32_t * n, uint32_t n_w, uint32_t m) {
    char what[96];
    snprintf(what, sizeof(what), "nx%u_k%u_n%u_m%u", n_w, k, n[0], m);
    g_cases++;

    uint8_t * w[4];
    float *   out[4];
    for (uint32_t i = 0; i < n_w; i++) {
        w[i]   = make_weight(k, hex_round_up(n[i], 32));
        out[i] = lab_ddr_alloc((size_t) m * n[i] * 4 + 256, 128);
    }
    float * x = lab_ddr_alloc((size_t) m * k * 4, 128);
    lab_fill_f32(x, (size_t) m * k, -1.0f, 1.0f);

    const int kernel_type = m < g_ctx.n_threads ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
    struct htp_mm_kernel_params kp;
    if (!hvx_params(k, hex_round_up(n[0], 32), m, kernel_type, n_w, false, false, &kp)) {
        printf("lab: error: %s: no layout\n", what);
        g_fail++;
        return;
    }
    struct htp_tensor src[5], dst[4];
    for (uint32_t i = 0; i < n_w; i++) {
        set_tensor(&src[i], w[i], HTP_TYPE_Q8_0, k, hex_round_up(n[i], 32), 1, sizeof(block_q8_0),
                   (k / 32) * sizeof(block_q8_0));
        set_tensor(&dst[i], out[i], HTP_TYPE_F32, n[i], m, 1, 4, 4 * n[i]);
    }
    set_tensor(&src[n_w], x, HTP_TYPE_F32, k, m, 1, 4, 4 * k);
    run_op(HTP_OP_MUL_MAT_NX, &kp, src, n_w + 1, dst, n_w);
    uint64_t h = 0xCBF29CE484222325ull;
    for (uint32_t i = 0; i < n_w; i++) {
        h = fnv1a(out[i], (size_t) m * n[i] * 4, h);
        char sub[112];
        snprintf(sub, sizeof(sub), "%s_w%u", what, i);
        check_ref(sub, out[i], w[i], x, NULL, k, n[i], m);
    }
    printf("lab: %s %s hash = 0x%016llx fnv1a\n", TARGET, what, (unsigned long long) h);
    for (uint32_t i = 0; i < n_w; i++) {
        free(w[i]);
        free(out[i]);
    }
    free(x);
}

// MUL_MAT_ID: n_as Q8_0 experts of n rows and k columns, n_used experts for each of the t tokens
static void run_id(uint32_t k, uint32_t n, uint32_t n_as, uint32_t n_used, uint32_t t) {
    const uint32_t n_pad = hex_round_up(n, 32);
    char what[96];
    snprintf(what, sizeof(what), "id_k%u_n%u_experts%u_used%u_tokens%u", k, n, n_as, n_used, t);
    g_cases++;

    const size_t expert_bytes = (size_t) (n_pad / 32) * (k / 32) * TILE;
    uint8_t *    w            = lab_ddr_alloc(expert_bytes * n_as, 128);
    for (uint32_t e = 0; e < n_as; e++) {
        uint8_t * we = make_weight(k, n_pad);
        memcpy(w + e * expert_bytes, we, expert_bytes);
        free(we);
    }
    // The activation has one row for each token (broadcast to the experts), and ids has n_used ids per token
    float *   x   = lab_ddr_alloc((size_t) t * k * 4, 128);
    int32_t * ids = lab_ddr_alloc((size_t) t * n_used * 4, 128);
    float *   out = lab_ddr_alloc((size_t) t * n_used * n * 4 + 256, 128);
    lab_fill_f32(x, (size_t) t * k, -1.0f, 1.0f);
    for (uint32_t i = 0; i < t; i++) {
        for (uint32_t j = 0; j < n_used; j++) {
            ids[i * n_used + j] = (int32_t) ((i * 3 + j * 5 + 1) % n_as);
        }
    }

    const int kernel_type = t < g_ctx.n_threads ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
    struct htp_mm_kernel_params kp;
    if (!hvx_params(k, n_pad, t, kernel_type, 0, true, false, &kp)) {
        printf("lab: error: %s: no layout\n", what);
        g_fail++;
        return;
    }
    kp.div_ne11 = init_fastdiv_values(1);
    struct htp_tensor src[3], dst;
    set_tensor(&src[0], w, HTP_TYPE_Q8_0, k, n_pad, n_as, sizeof(block_q8_0), (k / 32) * sizeof(block_q8_0));
    // src1: [k, 1, t]: one activation row for each token, broadcast over the ids of the token
    set_tensor(&src[1], x, HTP_TYPE_F32, k, 1, t, 4, 4 * k);
    set_tensor(&src[2], ids, HTP_TYPE_I32, n_used, t, 1, 4, 4 * n_used);
    // dst: [n, n_used, t]
    set_tensor(&dst, out, HTP_TYPE_F32, n, n_used, t, 4, 4 * n);
    run_op(HTP_OP_MUL_MAT_ID, &kp, src, 3, &dst, 1);
    report_hash(what, out, (size_t) t * n_used * n * 4);
    for (uint32_t i = 0; i < t; i++) {
        for (uint32_t j = 0; j < n_used; j++) {
            char sub[128];
            snprintf(sub, sizeof(sub), "%s_t%u_e%u", what, i, j);
            check_ref(sub, out + ((size_t) i * n_used + j) * n, w + (size_t) ids[i * n_used + j] * expert_bytes,
                      x + (size_t) i * k, NULL, k, n, 1);
        }
    }
    free(w);
    free(x);
    free(ids);
    free(out);
}

int main(int argc, char ** argv) {
    // Each result line goes out at once, thus a run that stops early keeps the lines of its cases
    setvbuf(stdout, NULL, _IONBF, 0);
    lab_init();

    const uint32_t nt = (uint32_t) lab_arg_long(argc, argv, "--threads", 4);

    g_ctx.vtcm_base     = lab_vtcm_base();
    g_ctx.vtcm_size     = lab_vtcm_size();
    g_ctx.n_threads     = nt;
    g_ctx.n_threads_div = init_fastdiv_values(nt);
    g_ctx.work_queue    = (work_queue_t) &g_hmx;  // the lab stub does not read it
    g_ctx.hmx_queue     = &g_hmx;
    g_ctx.mdev.count    = 1;
    g_ctx.ddr_spad_size = 1 << 20;
    g_ctx.ddr_spad_base = lab_ddr_alloc(g_ctx.ddr_spad_size, 128);
    for (uint32_t i = 0; i < HTP_MAX_NTHREADS; i++) {
        g_ctx.dma[i] = &g_dma[i];
    }
    printf("lab: %s threads %u vtcm %zu\n", TARGET, nt, g_ctx.vtcm_size);

    // One activation row: the decode GEMV of the 4B shapes, the fused residual ADD, a weight row count
    // that is not a multiple of 32, and odd k-tile counts (65 and 3)
    run_mm(2560, 1000, 1, HTP_MM_KERNEL_HVX_QUANT_BLOCK, false);
    run_mm(9216, 2560, 1, HTP_MM_KERNEL_HVX_QUANT_BLOCK, true);
    run_mm(4096, 2560, 1, HTP_MM_KERNEL_HVX_QUANT_BLOCK, true);
    run_mm(2080, 200, 1, HTP_MM_KERNEL_HVX_QUANT_BLOCK, false);
    run_mm(96, 100, 1, HTP_MM_KERNEL_HVX_QUANT_BLOCK, false);

    // 2 to 4 rows: the multi-row path (compact dots) and the row-pair path (32x2 and 32x1 dots)
    for (uint32_t m = 2; m <= 4; m++) {
        run_mm(2560, 1000, m, HTP_MM_KERNEL_HVX_QUANT_MULTIROW, false);
        run_mm(2560, 1000, m, HTP_MM_KERNEL_HVX_QUANT_BLOCK, false);
        run_mm(2080, 200, m, HTP_MM_KERNEL_HVX_QUANT_BLOCK, m == 3);
    }
    run_mm(4096, 2560, 3, HTP_MM_KERNEL_HVX_QUANT_MULTIROW, true);
    run_mm(2560, 512, 7, HTP_MM_KERNEL_HVX_QUANT_ROW, false);

    // MUL_MAT_NX: two weights (the gate and the up weight) and three weights (q, v and k), 1 to 4 rows
    const uint32_t gate_up[2] = { 2048, 2048 };
    const uint32_t qkv[3]     = { 2048, 256, 256 };
    const uint32_t small[2]   = { 200, 100 };
    for (uint32_t m = 1; m <= 4; m++) {
        run_nx(2560, gate_up, 2, m);
    }
    run_nx(2560, qkv, 3, 1);
    run_nx(2080, small, 2, 1);
    run_nx(2080, small, 2, 3);

    // MUL_MAT_ID: one token (hvx_mv_id) and three tokens (hvx_mm_id)
    run_id(256, 96, 4, 2, 1);
    run_id(2080, 100, 4, 2, 3);

    lab_report(TARGET, "cases", (double) g_cases, "cases");
    lab_report(TARGET, "mismatches", (double) g_fail, "values");
    // On the chip a push to a second DMA ring of one thread stops the op. A job that no pop took is
    // an error of the kernel. Thus the two counts go into the failure count of the run.
    g_fail += lab_dma_report(TARGET);
    g_fail += lab_hmx_report(TARGET, &g_hmx);
    printf("lab: check %s %s\n", TARGET, g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
