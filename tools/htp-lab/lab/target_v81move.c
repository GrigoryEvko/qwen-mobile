// Target v81move: the whole ops CPY, SET_ROWS, GET_ROWS, ROPE, SOFT_MAX and the UNARY op GELU of the Qwen3.5 4B
// text path and of its vision encoder, with an FNV-1a hash of each output.
//
// The program includes the op files verbatim and calls the op entry points with the kernel params that the host
// computes (ggml_hexagon_precompute_get_rows_params, _set_rows_params, _rope_params and _unary_params). The DMA is
// the shim of lab-dma.h: a push copies at once, a pop returns the destinations in the order of the pushes. Run it
// in the functional mode (MODE=functional).
//
// The cases (the shapes of the 4B and of its vision encoder):
//   cpy_f16     CPY of 3 rows of 2560 f32 values to f16, plus 1 row of 2500 values (a tail): the special values
//               of the f16 conversion (ties, subnormals, the overflow edge, the signed zeros, the infinities)
//   cont_t      CPY of a transposed 64 x 96 f32 matrix (the CONT of the vision encoder)
//   set_rows    SET_ROWS of 4 f32 rows of 1024 values into an f16 cache of 64 rows (the K and V cache write)
//   get_rows    GET_ROWS of 3 f32 rows of 2560 values (the embedding lookup of the MTP draft)
//   rope_neox   ROPE NEOX of 16 heads of 256 values, 64 rotated dimensions, 5 tokens at positions 0 to 40000
//   rope_imrope ROPE IMROPE with the sections 11, 11, 10, 0 (the text layers of Qwen3.5)
//   soft_max    SOFT_MAX of 16 rows of 1000 values with a scale
//   gelu        the UNARY op GELU of 4 rows of 4304 values (the vision MLP)
// Each case prints "hash = 0x..." of its output bytes and the largest error against a float64 reference of the
// op. For the f16 outputs the reference is the round to nearest even of the float64 value, thus the error of a
// correct conversion is 0. The ROPE reference makes the angles in f32 in the order of the CPU op.
//
// cont_t checks the transpose tile path of cpy-ops.c: a row of the original matrix is at the element stride nb00
// of the transposed view, and nb01 of the view is one element.
//
// Arguments: --threads 4
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

#include "lab-dma.h"

#include "cpy-ops.c"
#include "get-rows-ops.c"
#include "set-rows-ops.c"
#include "rope-ops.c"
#include "softmax-ops.c"
#include "unary-ops.c"

#define TARGET "v81move"

// The CPY_FENCE path of op_cpy flushes the dirty cache ranges of main.c. The cases run CPY only.
void htp_flush_dirty_ranges(struct htp_context * ctx) {
    (void) ctx;
}

static struct htp_context g_ctx;
static dma_queue          g_dma[HTP_MAX_NTHREADS];
static size_t             g_fail = 0;

// A tensor with 4 dimensions and the given strides
static struct htp_tensor mk(void * data, uint32_t type, uint32_t ne0, uint32_t ne1, uint32_t ne2, uint32_t ne3,
                            uint32_t nb0, uint32_t nb1, uint32_t nb2, uint32_t nb3) {
    struct htp_tensor t;
    memset(&t, 0, sizeof(t));
    t.data  = (uint32_t) (uintptr_t) data;
    t.type  = type;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = ne2;
    t.ne[3] = ne3;
    t.nb[0] = nb0;
    t.nb[1] = nb1;
    t.nb[2] = nb2;
    t.nb[3] = nb3;
    t.size  = (ne3 - 1) * nb3 + (ne2 - 1) * nb2 + (ne1 - 1) * nb1 + ne0 * nb0;
    return t;
}

// A contiguous tensor with 4 dimensions of elements of size es
static struct htp_tensor mkc(void * data, uint32_t type, uint32_t es, uint32_t ne0, uint32_t ne1, uint32_t ne2,
                             uint32_t ne3) {
    return mk(data, type, ne0, ne1, ne2, ne3, es, es * ne0, es * ne0 * ne1, es * ne0 * ne1 * ne2);
}

// Runs one op. src ends at the first NULL. O(1) plus the op.
static void run_op(const char * name, uint32_t op, const struct htp_tensor * const * src, const struct htp_tensor * dst,
                   const void * kp, size_t kp_size, uint32_t n_threads, const int32_t * op_params) {
    static struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx           = &g_ctx;
    octx.op            = (enum htp_op_code) op;
    octx.status        = HTP_STATUS_OK;  // as the batch loop of main.c sets it before each op
    octx.n_threads     = n_threads;
    octx.n_threads_div = init_fastdiv_values(n_threads);
    for (int i = 0; i < HTP_OP_MAX_INPUTS && src[i]; i++) {
        octx.src[i] = src[i];
    }
    octx.dst     = dst;
    octx.dsts[0] = dst;
    if (kp) {
        memcpy(octx.kernel_params, kp, kp_size);
    }
    if (op_params) {
        memcpy(octx.op_params, op_params, sizeof(octx.op_params));
    }
    int st;
    switch (op) {
        case HTP_OP_CPY:      st = op_cpy(&octx); break;
        case HTP_OP_GET_ROWS: st = op_get_rows(&octx); break;
        case HTP_OP_SET_ROWS: st = op_set_rows(&octx); break;
        case HTP_OP_ROPE:     st = op_rope(&octx); break;
        case HTP_OP_SOFTMAX:  st = op_softmax(&octx); break;
        default:              st = op_unary(&octx); break;
    }
    if (st != HTP_STATUS_OK) {
        printf("lab: %s %s: the op gave the status %d (detail %u)\n", TARGET, name, st, octx.err_detail);
        g_fail++;
    }
}

// The round to nearest even of a double to the f16 bits (the CPU conversion of a float value that the double
// holds exactly)
static uint16_t f16_of(double x) {
    const _Float16 h = (_Float16) (float) x;
    uint16_t       b;
    memcpy(&b, &h, 2);
    return b;
}

static void report(const char * name, const void * out, size_t bytes, double max_err, double max_ref, size_t n_bad,
                   double bound) {
    const double rel = max_ref > 0.0 ? max_err / max_ref : max_err;
    printf("lab: %s %s hash = 0x%016llx fnv1a\n", TARGET, name, (unsigned long long) lab_fnv1a(out, bytes));
    printf("lab: %s %s max_err %.6e of max_ref %.6e (relative %.3e), bad %zu\n", TARGET, name, max_err, max_ref, rel,
           n_bad);
    if (n_bad || !(rel <= bound)) {
        g_fail++;
    }
}

// The f32 bits of the special values of the f16 conversion, then random values in the f16 range
static void fill_f16_specials(float * x, size_t n) {
    static const uint32_t sp[] = {
        0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x3f801000u, 0x3f803000u, 0xbf801000u, 0x3f800fffu,
        0x3f801001u, 0x477fe000u, 0x477fefffu, 0x477ff000u, 0xc77ff000u, 0x477ff001u, 0x7f7fffffu, 0xff7fffffu,
        0x33000000u, 0xb3000000u, 0x33000001u, 0x32ffffffu, 0x33800000u, 0x33c00000u, 0x00000001u, 0x80000001u,
        0x38800000u, 0x387fe000u, 0x387ff000u,
    };
    for (size_t i = 0; i < n; i++) {
        if (i < sizeof(sp) / sizeof(sp[0])) {
            memcpy(&x[i], &sp[i], 4);
        } else if (i % 7 == 0) {
            const uint32_t w = ((uint32_t) (102 + lab_rand_u32() % 42) << 23) | ((lab_rand_u32() >> 22) << 13) | 0x1000u |
                               ((lab_rand_u32() & 1u) << 31);
            memcpy(&x[i], &w, 4);
        } else {
            x[i] = lab_rand_f32(-300.0f, 300.0f);
        }
    }
}

// CPY f32 -> f16 of rows of ne0 values: each output must be the round to nearest even of its input
static void case_cpy_f16(uint32_t ne0, uint32_t ne1) {
    float *    x = lab_ddr_alloc((size_t) ne0 * ne1 * 4, 128);
    uint16_t * y = lab_ddr_alloc((size_t) ne0 * ne1 * 2 + 128, 128);
    fill_f16_specials(x, (size_t) ne0 * ne1);
    struct htp_tensor         s = mkc(x, HTP_TYPE_F32, 4, ne0, ne1, 1, 1);
    struct htp_tensor         d = mkc(y, HTP_TYPE_F16, 2, ne0, ne1, 1, 1);
    const struct htp_tensor * src[] = { &s, NULL };
    run_op(ne0 == 2560 ? "cpy_f16" : "cpy_f16_tail", HTP_OP_CPY, src, &d, NULL, 0, g_ctx.n_threads, NULL);
    size_t n_bad = 0;
    for (size_t i = 0; i < (size_t) ne0 * ne1; i++) {
        const uint16_t want  = f16_of((double) x[i]);
        const bool     nan_w = (want & 0x7c00u) == 0x7c00u && (want & 0x3ffu);
        n_bad += nan_w ? !((y[i] & 0x7c00u) == 0x7c00u && (y[i] & 0x3ffu)) : (y[i] != want);
    }
    report(ne0 == 2560 ? "cpy_f16" : "cpy_f16_tail", y, (size_t) ne0 * ne1 * 2, (double) n_bad, 1.0, n_bad, 0.0);
    free(x);
    free(y);
}

// CPY of the transpose of an r x c f32 matrix into a contiguous c x r matrix
static void case_cont_t(uint32_t r, uint32_t c) {
    float * x = lab_ddr_alloc((size_t) r * c * 4, 128);
    float * y = lab_ddr_alloc((size_t) r * c * 4 + 128, 128);
    lab_fill_f32(x, (size_t) r * c, -4.0f, 4.0f);
    // src: the view with ne0 = r, ne1 = c of the matrix x of c columns (element (i0, i1) at x[i0 * c + i1])
    struct htp_tensor         s = mk(x, HTP_TYPE_F32, r, c, 1, 1, 4 * c, 4, 4 * r * c, 4 * r * c);
    struct htp_tensor         d = mkc(y, HTP_TYPE_F32, 4, r, c, 1, 1);
    const struct htp_tensor * src[] = { &s, NULL };
    run_op("cont_t", HTP_OP_CPY, src, &d, NULL, 0, g_ctx.n_threads, NULL);
    size_t n_bad = 0;
    for (uint32_t i1 = 0; i1 < c; i1++) {
        for (uint32_t i0 = 0; i0 < r; i0++) {
            if (y[(size_t) i1 * r + i0] != x[(size_t) i0 * c + i1]) {
                if (n_bad < 4) {
                    printf("lab: %s cont_t (%u, %u): %g, the source %g\n", TARGET, i0, i1, y[(size_t) i1 * r + i0],
                           x[(size_t) i0 * c + i1]);
                }
                n_bad++;
            }
        }
    }
    report("cont_t", y, (size_t) r * c * 4, (double) n_bad, 1.0, n_bad, 0.0);
    free(x);
    free(y);
}

// SET_ROWS of n f32 rows of ne0 values into an f16 cache of rows rows at the int64 indices idx
static void case_set_rows(uint32_t ne0, uint32_t n, uint32_t rows) {
    float *    x   = lab_ddr_alloc((size_t) ne0 * n * 4, 128);
    int64_t *  idx = lab_ddr_alloc(128, 128);
    uint16_t * c   = lab_ddr_alloc((size_t) ne0 * rows * 2 + 128, 128);
    fill_f16_specials(x, (size_t) ne0 * n);
    memset(c, 0x11, (size_t) ne0 * rows * 2);
    for (uint32_t i = 0; i < n; i++) {
        idx[i] = (int64_t) ((i * 17 + 5) % rows);
    }
    struct htp_tensor s  = mkc(x, HTP_TYPE_F32, 4, ne0, n, 1, 1);
    struct htp_tensor ix = mkc(idx, HTP_TYPE_I64, 8, n, 1, 1, 1);
    struct htp_tensor d  = mkc(c, HTP_TYPE_F16, 2, ne0, rows, 1, 1);

    struct htp_set_rows_kernel_params kp;
    memset(&kp, 0, sizeof(kp));
    kp.n_threads            = (int32_t) MIN(g_ctx.n_threads, n);
    kp.tasks_per_thread     = (int32_t) ((n + kp.n_threads - 1) / kp.n_threads);
    kp.total_tasks          = (int32_t) n;
    kp.div_ne11             = init_fastdiv_values(1);
    kp.div_ne12             = init_fastdiv_values(1);
    kp.div_tasks_per_thread = init_fastdiv_values(kp.tasks_per_thread);
    kp.div_ne02             = init_fastdiv_values(1);
    struct htp_set_rows_vtcm_layout L;
    htp_set_rows_vtcm_layout_build(&L, HTP_TYPE_F16, ne0, kp.n_threads);
    kp.vtcm_size = (int32_t) L.total_bytes;

    const struct htp_tensor * src[] = { &s, &ix, NULL };
    run_op("set_rows", HTP_OP_SET_ROWS, src, &d, &kp, sizeof(kp), kp.n_threads, NULL);
    size_t n_bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < ne0; j++) {
            const uint16_t want  = f16_of((double) x[(size_t) i * ne0 + j]);
            const uint16_t got   = c[(size_t) idx[i] * ne0 + j];
            const bool     nan_w = (want & 0x7c00u) == 0x7c00u && (want & 0x3ffu);
            const bool     bad   = nan_w ? !((got & 0x7c00u) == 0x7c00u && (got & 0x3ffu)) : (got != want);
            if (bad && n_bad < 4) {
                uint32_t w;
                memcpy(&w, &x[(size_t) i * ne0 + j], 4);
                printf("lab: %s set_rows row %u col %u: f32 0x%08x gives 0x%04x, the reference 0x%04x\n", TARGET, i, j,
                       (unsigned) w, (unsigned) got, (unsigned) want);
            }
            n_bad += bad;
        }
    }
    report("set_rows", c, (size_t) ne0 * rows * 2, (double) n_bad, 1.0, n_bad, 0.0);
    free(x);
    free(idx);
    free(c);
}

// GET_ROWS of n f32 rows of ne0 values from a table of rows rows at the int32 indices idx
static void case_get_rows(uint32_t ne0, uint32_t n, uint32_t rows) {
    float *   t   = lab_ddr_alloc((size_t) ne0 * rows * 4, 128);
    int32_t * idx = lab_ddr_alloc(128, 128);
    float *   y   = lab_ddr_alloc((size_t) ne0 * n * 4 + 128, 128);
    lab_fill_f32(t, (size_t) ne0 * rows, -2.0f, 2.0f);
    for (uint32_t i = 0; i < n; i++) {
        idx[i] = (int32_t) ((i * 7 + 3) % rows);
    }
    struct htp_tensor s  = mkc(t, HTP_TYPE_F32, 4, ne0, rows, 1, 1);
    struct htp_tensor ix = mkc(idx, HTP_TYPE_I32, 4, n, 1, 1, 1);
    struct htp_tensor d  = mkc(y, HTP_TYPE_F32, 4, ne0, n, 1, 1);

    struct htp_get_rows_kernel_params kp;
    memset(&kp, 0, sizeof(kp));
    const bool use_dma = ne0 >= 2048;
    kp.use_dma         = use_dma ? 1 : 0;
    uint32_t chunks = 1, chunk = ne0, tasks = n;
    if (!use_dma && n < g_ctx.n_threads) {
        uint32_t max_chunks = ne0 / 1024;
        max_chunks          = max_chunks ? max_chunks : 1;
        chunks              = MIN((g_ctx.n_threads + n - 1) / n, max_chunks);
        chunk               = (ne0 + chunks - 1) / chunks;
        tasks               = n * chunks;
    }
    kp.n_threads          = (int32_t) MIN(tasks, g_ctx.n_threads);
    kp.tasks_per_thread   = (int32_t) ((tasks + kp.n_threads - 1) / kp.n_threads);
    kp.chunks_per_row     = (int32_t) chunks;
    kp.chunk_size         = (int32_t) chunk;
    kp.total_tasks        = (int32_t) tasks;
    kp.div_ne10           = init_fastdiv_values(n);
    kp.div_ne10_ne11      = init_fastdiv_values(n);
    kp.div_chunks_per_row = init_fastdiv_values(chunks);
    kp.div_ne02           = init_fastdiv_values(1);
    kp.div_ne03           = init_fastdiv_values(1);
    struct htp_get_rows_vtcm_layout L;
    htp_get_rows_vtcm_layout_build(&L, HTP_TYPE_F32, ne0, kp.n_threads);
    kp.vtcm_size = (int32_t) L.total_bytes;

    const struct htp_tensor * src[] = { &s, &ix, NULL };
    run_op("get_rows", HTP_OP_GET_ROWS, src, &d, &kp, sizeof(kp), kp.n_threads, NULL);
    size_t n_bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        n_bad += memcmp(y + (size_t) i * ne0, t + (size_t) idx[i] * ne0, (size_t) ne0 * 4) != 0;
    }
    report("get_rows", y, (size_t) ne0 * n * 4, (double) n_bad, 1.0, n_bad, 0.0);
    free(t);
    free(idx);
    free(y);
}

// ROPE of heads x tokens rows of hd values, n_dims rotated, in the mode mode (NEOX or IMROPE with sections)
static void case_rope(const char * name, int32_t mode, uint32_t hd, uint32_t heads, uint32_t tokens, int32_t n_dims) {
    const float base   = 10000000.0f;
    float *     x      = lab_ddr_alloc((size_t) hd * heads * tokens * 4, 128);
    float *     y      = lab_ddr_alloc((size_t) hd * heads * tokens * 4 + 128, 128);
    int32_t *   pos    = lab_ddr_alloc((size_t) tokens * 4 * 4 + 128, 128);
    lab_fill_f32(x, (size_t) hd * heads * tokens, -1.0f, 1.0f);
    for (uint32_t t = 0; t < tokens; t++) {
        const int32_t p = (int32_t) (t * 10000 + 7);
        for (int k = 0; k < 4; k++) {
            pos[k * tokens + t] = k < 3 ? p : 0;
        }
    }
    struct htp_tensor s  = mkc(x, HTP_TYPE_F32, 4, hd, heads, tokens, 1);
    struct htp_tensor pp = mkc(pos, HTP_TYPE_I32, 4, tokens * (mode == HTP_ROPE_TYPE_NEOX ? 1 : 4), 1, 1, 1);
    struct htp_tensor d  = mkc(y, HTP_TYPE_F32, 4, hd, heads, tokens, 1);

    int32_t op_params[HTP_OP_MAX_PARAMS];
    memset(op_params, 0, sizeof(op_params));
    op_params[1]           = n_dims;
    op_params[2]           = mode;
    op_params[4]           = 262144;
    const float fl[6]      = { base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f };
    memcpy(&op_params[5], fl, sizeof(fl));
    const int32_t sections[4] = { 11, 11, 10, 0 };
    memcpy(&op_params[11], sections, sizeof(sections));

    struct htp_rope_kernel_params kp;
    memset(&kp, 0, sizeof(kp));
    const uint32_t nrows = heads * tokens;
    kp.n_threads         = MIN(g_ctx.n_threads, nrows);
    struct htp_rope_vtcm_layout L;
    htp_rope_vtcm_layout_build(&L, hd, kp.n_threads);
    kp.src0_nrows            = nrows;
    kp.src0_nrows_per_thread = (nrows + kp.n_threads - 1) / kp.n_threads;
    kp.vtcm_size             = (uint32_t) L.total_bytes;
    kp.spad_per_thread       = (uint32_t) L.bytes_per_thread;
    kp.theta_cache_offset    = (uint32_t) L.theta_cache_size_aligned;
    kp.src0_row_size_aligned = (uint32_t) L.src0_row_size_aligned;
    kp.div_ne2_ne1           = init_fastdiv_values(tokens * heads);
    kp.div_ne1               = init_fastdiv_values(heads);

    const struct htp_tensor * src[] = { &s, &pp, NULL };
    run_op(name, HTP_OP_ROPE, src, &d, &kp, sizeof(kp), kp.n_threads, op_params);

    // The reference: pairs (i, i + n_dims / 2) turn by theta_i. As in the CPU op, theta_i is the f32 product
    // pos * theta_scale^i in the order of the cache (one f32 multiply for each step), theta_scale = powf(base,
    // -2 / n_dims), and the sine and the cosine of that f32 angle are in float64. IMROPE takes the position of the
    // component of the sector: sector i of the rotated pairs uses the component i % 3 (all three positions are equal
    // in these cases).
    const float theta_scale = powf(base, -2.0f / (float) n_dims);
    double max_err = 0.0, max_ref = 0.0;
    size_t n_bad   = 0;
    for (uint32_t t = 0; t < tokens; t++) {
        for (uint32_t h = 0; h < heads; h++) {
            const float * xr = x + ((size_t) t * heads + h) * hd;
            const float * yr = y + ((size_t) t * heads + h) * hd;
            float theta_f[3] = { (float) pos[t], (float) pos[tokens + t], (float) pos[2 * tokens + t] };
            for (int32_t i = 0; i < n_dims / 2; i++) {
                int comp = 0;
                if (mode == HTP_ROPE_TYPE_IMROPE) {
                    comp = i % 3;
                }
                const double theta = (double) theta_f[comp];
                for (int k = 0; k < 3; k++) {
                    theta_f[k] *= theta_scale;
                }
                const double c = cos(theta), sn = sin(theta);
                const double a = xr[i], b = xr[i + n_dims / 2];
                const double r0 = a * c - b * sn, r1 = a * sn + b * c;
                const double e0 = fabs(yr[i] - r0), e1 = fabs(yr[i + n_dims / 2] - r1);
                max_err = fmax(max_err, fmax(e0, e1));
                max_ref = fmax(max_ref, fmax(fabs(r0), fabs(r1)));
            }
            for (uint32_t i = (uint32_t) n_dims; i < hd; i++) {
                n_bad += yr[i] != xr[i];
            }
        }
    }
    // The NEOX path of the kernel makes theta_i as pos times the f32 power theta_scale^i, one rounding where the
    // CPU makes i roundings, thus its angle differs from the CPU angle at a large position (8.8e-4 of the output
    // at the position 40007). The IMROPE path of the text layers of Qwen3.5 follows the order of the CPU.
    report(name, y, (size_t) hd * heads * tokens * 4, max_err, max_ref, n_bad, mode == HTP_ROPE_TYPE_NEOX ? 2e-3 : 1e-5);
    free(x);
    free(y);
    free(pos);
}

// SOFT_MAX of rows of n values with a scale and no mask
static void case_softmax(uint32_t n, uint32_t rows) {
    float * x = lab_ddr_alloc((size_t) n * rows * 4, 128);
    float * y = lab_ddr_alloc((size_t) n * rows * 4 + 128, 128);
    lab_fill_f32(x, (size_t) n * rows, -12.0f, 12.0f);
    struct htp_tensor s = mkc(x, HTP_TYPE_F32, 4, n, rows, 1, 1);
    struct htp_tensor d = mkc(y, HTP_TYPE_F32, 4, n, rows, 1, 1);
    int32_t           op_params[HTP_OP_MAX_PARAMS];
    memset(op_params, 0, sizeof(op_params));
    const float scale = 0.125f, max_bias = 0.0f;
    memcpy(&op_params[0], &scale, 4);
    memcpy(&op_params[1], &max_bias, 4);
    const struct htp_tensor * src[] = { &s, NULL };
    run_op("soft_max", HTP_OP_SOFTMAX, src, &d, NULL, 0, MIN(g_ctx.n_threads, rows), op_params);
    double max_err = 0.0, max_ref = 0.0;
    for (uint32_t r = 0; r < rows; r++) {
        double m = -INFINITY, sum = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            m = fmax(m, scale * (double) x[(size_t) r * n + i]);
        }
        for (uint32_t i = 0; i < n; i++) {
            sum += exp(scale * (double) x[(size_t) r * n + i] - m);
        }
        for (uint32_t i = 0; i < n; i++) {
            const double ref = exp(scale * (double) x[(size_t) r * n + i] - m) / sum;
            max_err          = fmax(max_err, fabs(y[(size_t) r * n + i] - ref));
            max_ref          = fmax(max_ref, ref);
        }
    }
    report("soft_max", y, (size_t) n * rows * 4, max_err, max_ref, 0, 1e-4);
    free(x);
    free(y);
}

// The UNARY op GELU (the tanh form of the CPU) of rows of n values
static void case_gelu(uint32_t n, uint32_t rows) {
    float * x = lab_ddr_alloc((size_t) n * rows * 4, 128);
    float * y = lab_ddr_alloc((size_t) n * rows * 4 + 128, 128);
    lab_fill_f32(x, (size_t) n * rows, -6.0f, 6.0f);
    struct htp_tensor s = mkc(x, HTP_TYPE_F32, 4, n, rows, 1, 1);
    struct htp_tensor d = mkc(y, HTP_TYPE_F32, 4, n, rows, 1, 1);

    // The kernel params of ggml_hexagon_precompute_unary_params (the lab_unary_params of target_gate.c)
    struct htp_unary_kernel_params kp;
    memset(&kp, 0, sizeof(kp));
    const uint32_t nt          = MIN(g_ctx.n_threads, rows);
    kp.n_threads               = nt;
    kp.src0_row_size_aligned   = hex_round_up(n * 4, 128);
    kp.dst_row_size_aligned    = hex_round_up(n * 4, 128);
    struct htp_unary_vtcm_layout L;
    uint32_t                     col_tile = 0, vrows = 0;
    htp_unary_vtcm_layout_build(&L, HTP_OP_UNARY_GELU, n, n, 0, false, nt, g_ctx.vtcm_size, 4, &col_tile, &vrows);
    kp.col_tile                  = col_tile;
    kp.vtcm_row_per_thread       = vrows;
    kp.vtcm_size                 = L.total_bytes;
    kp.vtcm_src0_size_per_thread = L.src0_bytes;
    kp.vtcm_dst_size_per_thread  = L.dst_bytes;
    kp.vtcm_src0_size            = L.src0_bytes * nt;
    kp.vtcm_dst_size             = L.dst_bytes * nt;
    kp.block                     = col_tile ? 0 : ((L.src0_bytes / 2) / kp.src0_row_size_aligned);
    const uint32_t tiles         = col_tile > 0 ? (n + col_tile - 1) / col_tile : 1;
    kp.div_ne01                  = init_fastdiv_values(rows);
    kp.div_ne02                  = init_fastdiv_values(1);
    kp.div_ne012                 = init_fastdiv_values(rows);
    kp.div_tpr                   = init_fastdiv_values(tiles);

    const struct htp_tensor * src[] = { &s, NULL };
    run_op("gelu", HTP_OP_UNARY_GELU, src, &d, &kp, sizeof(kp), nt, NULL);
    double max_err = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < (size_t) n * rows; i++) {
        const double v   = x[i];
        const double ref = 0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
        max_err          = fmax(max_err, fabs(y[i] - ref));
        max_ref          = fmax(max_ref, fabs(ref));
    }
    report("gelu", y, (size_t) n * rows * 4, max_err, max_ref, 0, 1e-3);
    free(x);
    free(y);
}

int main(int argc, char ** argv) {
    lab_init();
    const uint32_t nt = (uint32_t) lab_arg_long(argc, argv, "--threads", 4);
    g_ctx.vtcm_base     = lab_vtcm_base();
    g_ctx.vtcm_size     = lab_vtcm_size();
    g_ctx.n_threads     = nt;
    g_ctx.n_threads_div = init_fastdiv_values(nt);
    g_ctx.mdev.count    = 1;
    for (uint32_t i = 0; i < HTP_MAX_NTHREADS; i++) {
        g_ctx.dma[i]        = &g_dma[i];
        g_ctx.dma_cached[i] = &g_dma[i];
    }
    printf("lab: %s threads %u vtcm %zu\n", TARGET, nt, g_ctx.vtcm_size);

    case_cpy_f16(2560, 3);
    case_cpy_f16(2500, 1);
    case_cont_t(64, 96);
    case_set_rows(1024, 4, 64);
    case_get_rows(2560, 3, 16);
    case_rope("rope_neox", HTP_ROPE_TYPE_NEOX, 256, 16, 5, 64);
    case_rope("rope_imrope", HTP_ROPE_TYPE_IMROPE, 256, 16, 5, 64);
    case_softmax(1000, 16);
    case_gelu(4304, 4);

    g_fail += lab_dma_report(TARGET);
    lab_report(TARGET, "failures", (double) g_fail, "cases");
    printf("lab: check %s %s\n", TARGET, g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
