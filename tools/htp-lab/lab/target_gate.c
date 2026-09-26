// Target gate: the gate ops of gate-ops.c (HTP_OP_RMS_NORM_GATE, HTP_OP_SIGMOID_GATE) against the unfused ops.
//
// The program includes unary-ops.c, binary-ops.c and gate-ops.c verbatim. For each case it runs the unfused
// ops in the form that the host sends them, and the fused op on the same inputs:
//   norm     RMS_NORM_MUL (x, w), SILU (z), MUL (norm, silu)          against RMS_NORM_GATE (x, w, z)
//   norm2    RMS_NORM (x), MUL (weight row), SILU (z), MUL            against RMS_NORM_GATE (x, w, z)
//            (the host sends RMS_NORM and MUL as two ops when x is not contiguous)
//   sigmoid  a copy of the gate half (CONT), SIGMOID, MUL (a, sigmoid) against SIGMOID_GATE (a, the gate half)
// It compares the two outputs bit for bit, and the fused output with a float64 reference, to show that the
// inputs reach the ranges of the model. A mismatch of one bit fails the case.
//
// The inputs have special values at a rate of 1 in 61 when --specials 1: +0 and -0, a subnormal, 1e30,
// +Inf, -Inf, a NaN, and values in [14.49, 15.73] (the band where a sign error of the int16 SiLU gives
// +Inf). The reference comparison skips the rows that hold a special value.
//
// DMA: the standalone runtime of the simulator has no user DMA (refer to target_fa.c). The shim below copies
// at the push and returns the destination and the source of each push in the order of the pushes. The unary,
// binary and gate ops take the output slot from the source of a pop, thus the shim keeps both.
//
// Arguments: --kind 0 (norm) | 1 (norm2) | 2 (sigmoid)  --d 128 --heads 32 --tokens 64 --threads 4
//            --layout 0 (x and a are tensors of their own) | 1 (x is the head of a larger tensor, a gap after
//            each token)  --swap 0|1  --block 0 (the largest block that fits) or N rows  --specials 0|1
//            --iters 1
//            --compute 1 --rows 64: the cycles of the compute of one block on one thread (timing mode)
// lab-run: mode=functional
#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "lab-dma.h"

#include "unary-ops.c"
#include "binary-ops.c"
#include "gate-ops.c"

#define TARGET "gate"

static struct htp_context g_ctx;
static dma_queue          g_queues[HTP_MAX_NTHREADS];

// A tensor descriptor in DDR, as prep_tensor of main.c leaves it: data is the address
static struct htp_tensor lab_tensor(void * data, uint32_t ne0, uint32_t ne1, uint32_t ne2, uint32_t ne3, uint32_t nb1,
                                    uint32_t nb2, uint32_t nb3) {
    struct htp_tensor t;
    memset(&t, 0, sizeof(t));
    t.data  = (uint32_t) (uintptr_t) data;
    t.type  = HTP_TYPE_F32;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = ne2;
    t.ne[3] = ne3;
    t.nb[0] = sizeof(float);
    t.nb[1] = nb1;
    t.nb[2] = nb2;
    t.nb[3] = nb3;
    t.size  = (ne3 - 1) * nb3 + (ne2 - 1) * nb2 + (ne1 - 1) * nb1 + ne0 * sizeof(float);
    return t;
}

static struct htp_tensor lab_tensor_contig(void * data, uint32_t ne0, uint32_t ne1, uint32_t ne2) {
    const uint32_t nb1 = ne0 * sizeof(float);
    return lab_tensor(data, ne0, ne1, ne2, 1, nb1, nb1 * ne1, nb1 * ne1 * ne2);
}

// The kernel params of a unary op, as ggml_hexagon_precompute_unary_params of the host computes them
static void lab_unary_params(uint32_t op, const struct htp_tensor * src0, const struct htp_tensor * src1,
                             const struct htp_tensor * dst, uint32_t n_threads_max, struct htp_unary_kernel_params * kp) {
    memset(kp, 0, sizeof(*kp));
    const uint32_t nrows     = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads = MIN(n_threads_max, nrows);
    kp->n_threads            = n_threads;

    const size_t elem         = sizeof(float);
    const size_t src0_aligned = hex_round_up(src0->ne[0] * elem, 128);
    const size_t dst_aligned  = hex_round_up(dst->ne[0] * elem, 128);
    kp->src0_row_size_aligned = src0_aligned;
    kp->dst_row_size_aligned  = dst_aligned;

    bool broadcast = false;
    if (op == HTP_OP_RMS_NORM_MUL) {
        kp->src1_row_size_aligned = hex_round_up(src1->ne[0] * elem, 128);
        broadcast                 = src1->ne[1] * src1->ne[2] * src1->ne[3] == 1;
    }
    kp->broadcast_weight = broadcast;

    struct htp_unary_vtcm_layout L;
    uint32_t                     col_tile = 0;
    uint32_t                     rows     = 0;
    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0], op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0, broadcast,
                                n_threads, g_ctx.vtcm_size, elem, &col_tile, &rows);
    kp->col_tile                  = col_tile;
    kp->vtcm_row_per_thread       = rows;
    kp->vtcm_size                 = L.total_bytes;
    kp->vtcm_src0_size_per_thread = L.src0_bytes;
    kp->vtcm_src1_size_per_thread = L.src1_bytes;
    kp->vtcm_dst_size_per_thread  = L.dst_bytes;
    kp->vtcm_src0_size            = L.src0_bytes * n_threads;
    kp->vtcm_src1_size            = L.src1_bytes * n_threads;
    kp->vtcm_dst_size             = L.dst_bytes * n_threads;
    kp->block                     = col_tile ? 0 : ((L.src0_bytes / 2) / src0_aligned);
    const uint32_t tiles          = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kp->div_ne01                  = init_fastdiv_values(src0->ne[1]);
    kp->div_ne02                  = init_fastdiv_values(src0->ne[2]);
    kp->div_ne012                 = init_fastdiv_values(src0->ne[1] * src0->ne[2]);
    kp->div_tpr                   = init_fastdiv_values(tiles);
}

// Runs one op on the lab context. src[] ends at the first NULL.
static int lab_run_op(uint32_t op, const struct htp_tensor * const * src, const struct htp_tensor * dst,
                      const void * kparams, size_t kparams_size, uint32_t n_threads, const int32_t * op_params) {
    static struct htp_ops_context octx;
    memset(&octx, 0, sizeof(octx));
    octx.ctx           = &g_ctx;
    octx.op            = (enum htp_op_code) op;
    octx.n_threads     = n_threads;
    octx.n_threads_div = init_fastdiv_values(n_threads);
    for (int i = 0; i < HTP_OP_MAX_INPUTS && src[i]; i++) {
        octx.src[i] = src[i];
    }
    octx.dst = dst;
    if (kparams) {
        memcpy(octx.kernel_params, kparams, kparams_size);
    }
    if (op_params) {
        memcpy(octx.op_params, op_params, sizeof(octx.op_params));
    }
    int st;
    switch (op) {
        case HTP_OP_MUL:
            st = op_binary(&octx);
            break;
        case HTP_OP_RMS_NORM_GATE:
        case HTP_OP_SIGMOID_GATE:
            st = op_gate(&octx);
            break;
        default:
            st = op_unary(&octx);
            break;
    }
    if (st != HTP_STATUS_OK) {
        printf("lab: %s op %u gave the status %d (detail %u)\n", "gate", op, st, octx.err_detail);
    }
    return st;
}

// Fills n floats, with a special value at a rate of 1 in 61 when specials is set
static void lab_fill(float * p, size_t n, float lo, float hi, bool specials) {
    static const float band[] = { 14.49f, 14.8f, 15.1f, 15.4f, 15.73f };
    for (size_t i = 0; i < n; i++) {
        p[i] = lab_rand_f32(lo, hi);
        if (specials && lab_rand_u32() % 61 == 0) {
            switch (lab_rand_u32() % 9) {
                case 0: p[i] = 0.0f; break;
                case 1: p[i] = -0.0f; break;
                case 2: p[i] = 1e-40f; break;
                case 3: p[i] = 1e30f; break;
                case 4: p[i] = INFINITY; break;
                case 5: p[i] = -INFINITY; break;
                case 6: p[i] = NAN; break;
                default: p[i] = (lab_rand_u32() & 1 ? 1.0f : -1.0f) * band[lab_rand_u32() % 5]; break;
            }
        }
    }
}

static bool lab_is_special(float v) {
    return !isfinite(v) || fabsf(v) > 1e20f || (v != 0.0f && fabsf(v) < 1e-30f) || v == 0.0f;
}

// --compute 1: the cycles of gate_compute alone on one thread, with the rows already in the VTCM. The DDR side
// of an op moves 3 vectors for each output vector at about 4.55 cycles each for all threads together
// (target_binary), thus the op stays bound by the DDR while this number is below 3 * 4.55 * the threads.
static int lab_compute_only(uint32_t kind, uint32_t D, uint32_t n, uint32_t iters, uint32_t part) {
    const bool   norm = kind != 2;
    const size_t Ra   = hex_round_up(D * sizeof(float), 128);
    uint8_t *    a    = lab_vtcm_alloc(n * Ra, 128);
    uint8_t *    b    = lab_vtcm_alloc(n * Ra, 128);
    uint8_t *    o    = lab_vtcm_alloc(n * Ra, 128);
    uint8_t *    act  = lab_vtcm_alloc(n * Ra, 128);
    uint8_t *    w    = lab_vtcm_alloc(Ra, 128);
    lab_fill((float *) a, n * Ra / sizeof(float), -10.0f, 10.0f, false);
    lab_fill((float *) b, n * Ra / sizeof(float), -20.0f, 20.0f, false);
    lab_fill((float *) w, Ra / sizeof(float), -2.0f, 2.0f, false);

    struct htp_gate_context g;
    memset(&g, 0, sizeof(g));
    g.D           = D;
    g.row_bytes   = D * sizeof(float);
    g.row_aligned = Ra;
    g.has_weight  = norm;
    g.packed      = g.row_bytes == g.row_aligned;
    g.eps         = 1e-6f;

    // --part: 0 the whole block, 2 the norm (hvx_fast_rms_norm_mul_f32 for each row), 3 the SiLU, 4 the
    // product
#define LAB_PART()                                                                                        \
    do {                                                                                                  \
        switch (part) {                                                                                   \
            case 2: for (uint32_t r = 0; r < n; r++) {                                                    \
                        hvx_fast_rms_norm_mul_f32(a + r * Ra, w, o + r * Ra, (int) D, g.eps);             \
                    }                                                                                     \
                    break;                                                                                \
            case 3: gate_silu(act, b, n * D); break;                                                      \
            case 4: hvx_mul_f32_aaa(o, o, act, n * D); break;                                             \
            default: gate_compute(&g, o, a, b, act, w, n); break;                                         \
        }                                                                                                 \
    } while (0)
    LAB_PART();
    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        LAB_PART();
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }
    const double vecs = (double) n * D / 32.0;
    printf("lab: %s compute kind %u d %u rows %u part %u\n", TARGET, kind, D, n, part);
    lab_report(TARGET, "compute_cycles_min", (double) best, "cycles");
    lab_report(TARGET, "compute_cycles_per_output_vector", (double) best / vecs, "cycles");
    return 0;
}

int main(int argc, char ** argv) {
    const uint32_t kind      = (uint32_t) lab_arg_long(argc, argv, "--kind", 0);
    const uint32_t D         = (uint32_t) lab_arg_long(argc, argv, "--d", 128);
    const uint32_t H         = (uint32_t) lab_arg_long(argc, argv, "--heads", 32);
    const uint32_t T         = (uint32_t) lab_arg_long(argc, argv, "--tokens", 64);
    const uint32_t n_threads = (uint32_t) lab_arg_long(argc, argv, "--threads", 4);
    const uint32_t kind_arg  = (uint32_t) lab_arg_long(argc, argv, "--kind", 0);
    // the sigmoid gate reads a as one contiguous tensor, thus layout 1 is for the norm kinds only
    const uint32_t layout    = kind_arg == 2 ? 0 : (uint32_t) lab_arg_long(argc, argv, "--layout", 0);
    const bool     swap      = lab_arg_long(argc, argv, "--swap", 0) != 0;
    const uint32_t block_arg = (uint32_t) lab_arg_long(argc, argv, "--block", 0);
    const bool     specials  = lab_arg_long(argc, argv, "--specials", 0) != 0;
    const uint32_t iters     = (uint32_t) lab_arg_long(argc, argv, "--iters", 1);

    lab_init();
    if (lab_arg_long(argc, argv, "--compute", 0)) {
        return lab_compute_only(kind, D, (uint32_t) lab_arg_long(argc, argv, "--rows", 64), iters,
                                (uint32_t) lab_arg_long(argc, argv, "--part", 0));
    }
    memset(&g_ctx, 0, sizeof(g_ctx));
    g_ctx.vtcm_base     = lab_vtcm_base();
    g_ctx.vtcm_size     = lab_vtcm_size();
    g_ctx.n_threads     = n_threads;
    g_ctx.n_threads_div = init_fastdiv_values(n_threads);
    for (uint32_t i = 0; i < n_threads && i < HTP_MAX_NTHREADS; i++) {
        memset(&g_queues[i], 0, sizeof(g_queues[i]));
        g_ctx.dma[i] = &g_queues[i];
    }

    const bool     norm  = kind != 2;
    const uint32_t nrows = H * T;
    const size_t   vals  = (size_t) D * nrows;
    const size_t   bytes = vals * sizeof(float);
    const float    eps   = 1e-6f;

    printf("lab: %s kind %u d %u heads %u tokens %u threads %u layout %u swap %d specials %d\n", TARGET, kind, D, H, T,
           n_threads, layout, (int) swap, (int) specials);

    // in0: x (norm) or a (sigmoid). layout 1 gives x one more head of gap in each token.
    const uint32_t H0     = layout == 1 ? H + 1 : H;
    float *        in0    = lab_ddr_alloc((size_t) D * H0 * T * sizeof(float) + 256, 128);
    float *        w      = lab_ddr_alloc(D * sizeof(float) + 256, 128);
    // in1: z (norm), or the joint q and gate projection [2 D H, T] (sigmoid)
    float *        in1    = lab_ddr_alloc(2 * bytes + 256, 128);
    float *        mid0   = lab_ddr_alloc(bytes + 256, 128);
    float *        mid1   = lab_ddr_alloc(bytes + 256, 128);
    float *        out_u  = lab_ddr_alloc(bytes + 256, 128);
    float *        out_f  = lab_ddr_alloc(bytes + 256, 128);

    lab_fill(in0, (size_t) D * H0 * T, norm ? -10.0f : -4.0f, norm ? 10.0f : 4.0f, specials);
    lab_fill(w, D, -2.0f, 2.0f, false);
    lab_fill(in1, 2 * vals, -20.0f, 20.0f, specials);

    struct htp_tensor t_in0 = layout == 1 ? lab_tensor(in0, D, H, T, 1, D * 4, D * H0 * 4, D * H0 * T * 4)
                                          : lab_tensor_contig(in0, D, H, T);
    struct htp_tensor t_w   = lab_tensor_contig(w, D, 1, 1);
    struct htp_tensor t_z   = lab_tensor_contig(in1, D, H, T);
    // the gate half of each head: [D, H, T] with the strides of the joint projection, 1 row of D after the q row
    struct htp_tensor t_g   = lab_tensor((uint8_t *) in1 + D * 4, D, H, T, 1, 2 * D * 4, 2 * D * H * 4, 2 * D * H * T * 4);
    struct htp_tensor t_a2  = lab_tensor_contig(in0, D * H, T, 1);
    struct htp_tensor t_m0  = lab_tensor_contig(mid0, D, H, T);
    struct htp_tensor t_m1  = lab_tensor_contig(mid1, D, H, T);
    struct htp_tensor t_ou  = lab_tensor_contig(out_u, D, H, T);
    struct htp_tensor t_of  = norm ? lab_tensor_contig(out_f, D, H, T) : lab_tensor_contig(out_f, D * H, T, 1);

    int32_t op_params[HTP_OP_MAX_PARAMS];
    memset(op_params, 0, sizeof(op_params));
    memcpy(&op_params[0], &eps, sizeof(float));

    // The unfused ops
    struct htp_unary_kernel_params ukp;
    int                            st = 0;  // 1 when an unfused op fails
    if (kind == 0) {
        const struct htp_tensor * s_n[] = { &t_in0, &t_w, NULL };
        lab_unary_params(HTP_OP_RMS_NORM_MUL, &t_in0, &t_w, &t_m0, n_threads, &ukp);
        st |= lab_run_op(HTP_OP_RMS_NORM_MUL, s_n, &t_m0, &ukp, sizeof(ukp), n_threads, op_params) != HTP_STATUS_OK;
    } else if (kind == 1) {
        const struct htp_tensor * s_r[] = { &t_in0, NULL };
        lab_unary_params(HTP_OP_RMS_NORM, &t_in0, NULL, &t_m0, n_threads, &ukp);
        st |= lab_run_op(HTP_OP_RMS_NORM, s_r, &t_m0, &ukp, sizeof(ukp), n_threads, op_params) != HTP_STATUS_OK;
        const struct htp_tensor * s_m[] = { &t_m0, &t_w, NULL };
        st |= lab_run_op(HTP_OP_MUL, s_m, &t_m0, NULL, 0, n_threads, NULL) != HTP_STATUS_OK;
    }
    if (norm) {
        const struct htp_tensor * s_s[] = { &t_z, NULL };
        lab_unary_params(HTP_OP_UNARY_SILU, &t_z, NULL, &t_m1, n_threads, &ukp);
        st |= lab_run_op(HTP_OP_UNARY_SILU, s_s, &t_m1, &ukp, sizeof(ukp), n_threads, NULL) != HTP_STATUS_OK;
        const struct htp_tensor * s_p[] = { swap ? &t_m1 : &t_m0, swap ? &t_m0 : &t_m1, NULL };
        st |= lab_run_op(HTP_OP_MUL, s_p, &t_ou, NULL, 0, n_threads, NULL) != HTP_STATUS_OK;
    } else {
        // CONT of the gate half, as the copy op writes it
        for (uint32_t r = 0; r < nrows; r++) {
            memcpy(mid0 + (size_t) r * D, in1 + (size_t) r * 2 * D + D, D * sizeof(float));
        }
        const struct htp_tensor * s_s[] = { &t_m0, NULL };
        lab_unary_params(HTP_OP_UNARY_SIGMOID, &t_m0, NULL, &t_m1, n_threads, &ukp);
        st |= lab_run_op(HTP_OP_UNARY_SIGMOID, s_s, &t_m1, &ukp, sizeof(ukp), n_threads, NULL) != HTP_STATUS_OK;
        struct htp_tensor         t_a = lab_tensor_contig(in0, D, H, T);
        const struct htp_tensor * s_p[] = { swap ? &t_m1 : &t_a, swap ? &t_a : &t_m1, NULL };
        st |= lab_run_op(HTP_OP_MUL, s_p, &t_ou, NULL, 0, n_threads, NULL) != HTP_STATUS_OK;
    }
    if (st) {
        printf("lab: %s an unfused op failed\n", TARGET);
        return 1;
    }

    // The fused op
    struct htp_gate_kernel_params gkp;
    memset(&gkp, 0, sizeof(gkp));
    const size_t   row_aligned = hex_round_up(D * sizeof(float), 128);
    const uint32_t per_thread  = (nrows + n_threads - 1) / n_threads;
    gkp.n_threads              = MIN(n_threads, nrows);
    gkp.block = block_arg ? block_arg : htp_gate_max_block(norm, row_aligned, gkp.n_threads, g_ctx.vtcm_size, per_thread);
    memcpy(&gkp.eps_bits, &eps, sizeof(float));
    gkp.flags = swap ? HTP_GATE_FLAG_SWAP : 0;

    const struct htp_tensor * s_fn[] = { &t_in0, &t_w, &t_z, NULL };
    const struct htp_tensor * s_fs[] = { &t_a2, &t_g, NULL };
    const uint32_t            op     = norm ? HTP_OP_RMS_NORM_GATE : HTP_OP_SIGMOID_GATE;

    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters; it++) {
        memset(out_f, 0xa5, bytes);
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        st = lab_run_op(op, norm ? s_fn : s_fs, &t_of, &gkp, sizeof(gkp), gkp.n_threads, NULL);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        if (st != HTP_STATUS_OK) {
            printf("lab: %s the fused op gave the status %d\n", TARGET, st);
            return 1;
        }
        if (t1 - t0 < best) {
            best = t1 - t0;
        }
    }

    // Bit for bit against the unfused ops
    size_t n_diff = 0;
    for (size_t i = 0; i < vals; i++) {
        uint32_t a, b;
        memcpy(&a, &out_u[i], 4);
        memcpy(&b, &out_f[i], 4);
        if (a != b) {
            if (n_diff < 5) {
                printf("lab: %s differ at %zu: unfused 0x%08x (%g) fused 0x%08x (%g)\n", TARGET, i, a, (double) out_u[i],
                       b, (double) out_f[i]);
            }
            n_diff++;
        }
    }

    // Against a float64 reference, on the rows with no special value
    double   max_abs = 0.0;
    uint32_t n_rows_checked = 0;
    for (uint32_t r = 0; r < nrows; r++) {
        const uint32_t h  = r % H;
        const uint32_t t  = r / H;
        const float *  xr = in0 + (size_t) t * D * H0 + (size_t) h * D;
        const float *  gr = norm ? in1 + (size_t) r * D : in1 + (size_t) r * 2 * D + D;
        bool           sp = false;
        for (uint32_t k = 0; k < D; k++) {
            sp |= lab_is_special(xr[k]) || lab_is_special(gr[k]);
        }
        if (sp) {
            continue;
        }
        n_rows_checked++;
        double ss = 0.0;
        for (uint32_t k = 0; k < D; k++) {
            ss += (double) xr[k] * xr[k];
        }
        const double scale = 1.0 / sqrt(ss / D + eps);
        for (uint32_t k = 0; k < D; k++) {
            const double gv  = gr[k];
            const double sig = 1.0 / (1.0 + exp(-gv));
            const double ref = norm ? (double) xr[k] * scale * w[k] * gv * sig : (double) xr[k] * sig;
            const double d   = fabs(ref - (double) out_f[(size_t) r * D + k]);
            if (d > max_abs) {
                max_abs = d;
            }
        }
    }

    lab_report(TARGET, "rows", nrows, "");
    lab_report(TARGET, "block", gkp.block, "rows");
    lab_report(TARGET, "cycles_min", (double) best, "cycles");
    lab_report(TARGET, "cycles_per_vector", (double) best / (double) (vals / 32 ? vals / 32 : 1), "cycles");
    lab_report(TARGET, "bits_differ", (double) n_diff, "values");
    lab_report(TARGET, "ref_rows_checked", n_rows_checked, "rows");
    lab_report(TARGET, "ref_max_abs_error", max_abs, "");
    const uint32_t dma_faults = lab_dma_report(TARGET);
    return (n_diff || dma_faults) ? 1 : 0;
}
