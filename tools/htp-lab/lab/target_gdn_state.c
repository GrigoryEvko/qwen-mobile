// Target 8: the HVX helpers of the chunked gated delta net kernel, gdn-chunk-ops.c.
//
// On the phone the op GATED_DELTA_NET takes 78 ms of the 536 ms of DSP op time of a 512-token
// prefill of the 4B (15 %, 1.83 instructions per packet). Qwen3.5 gives the gate the shape
// [1, H, tokens, seqs], thus g->ne[0] is 1, and with K = 1 and one sequence
// op_gated_delta_net_chunked accepts the batch: the chunked kernel is what runs, and the
// sequential thread of gated-delta-net-ops.c is the fallback for K > 1.
//
// The chunked kernel is part HMX and part HVX. The timing model of this SDK never retires an HMX
// instruction, thus the whole op cannot be timed here. This target times the HVX helpers on their
// own, which is the half that a kernel change can reach:
//   decay_matrix  E[t][s] = exp(gamma_t - gamma_s) below the diagonal, C x C
//   solve         T = (I + A')^-1 by forward substitution, the sequential core
//   scale_p       P' = P * E * scale, C x C
//   transpose_f32 the state transpose, D x D
//   f32_to_f16_rows and scatter_transpose_f16, the activation staging of C x D
//
// Each helper runs on its own, thus the numbers are throughput at a warm cache and no DMA.
//
// Arguments: --d 128 --iters 8
#include "lab.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "gdn-chunk-ops.c"

#define TARGET "gdn_state"

// Time one helper over iters calls and report cycles, packets of work and the derived rate.
// The caller gives the count of one unit of work, for example one 32-lane block.
#define TIME_HELPER(name, units, call)                                          \
    do {                                                                        \
        call;                          /* warm the cache and the branch */      \
        uint64_t best = UINT64_MAX;                                             \
        for (uint32_t it = 0; it < iters; it++) {                               \
            LAB_BARRIER();                                                      \
            const uint64_t t0 = lab_cycles();                                   \
            call;                                                               \
            const uint64_t t1 = lab_cycles();                                   \
            LAB_BARRIER();                                                      \
            if (t1 - t0 < best) { best = t1 - t0; }                             \
        }                                                                       \
        printf("lab: %s %-22s %8llu cycles  %8.2f cycles per unit (%u units)\n",\
               TARGET, name, (unsigned long long) best,                         \
               (double) best / (double) (units), (unsigned) (units));           \
        total_cycles += best;                                                   \
    } while (0)

int main(int argc, char ** argv) {
    const uint32_t D     = (uint32_t) lab_arg_long(argc, argv, "--d", 128);
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 8);
    const uint32_t C     = GDN_CH_C;

    lab_init();
    printf("lab: %s chunk C %u head size D %u\n", TARGET, C, D);

    static struct gdn_ch_gates ga;
    for (uint32_t t = 0; t < C; t++) {
        ga.gam[t]    = -0.02f * (float) (t + 1);          // the cumulative log decay falls
        ga.bet[t]    = 0.5f;
        ga.egc[t]    = expf(ga.gam[C - 1] - ga.gam[t]);
        ga.sc_qg[t]  = 0.1f;
        ga.sc_kgb[t] = 0.1f;
    }
    ga.e_gc = expf(ga.gam[C - 1]);

    struct gdn_ch_row rw;
    memset(&rw, 0, sizeof(rw));
    rw.e32 = lab_vtcm_alloc((size_t) C * C * sizeof(float), 128);
    rw.a32 = lab_vtcm_alloc((size_t) C * C * sizeof(float), 128);
    rw.t32 = lab_vtcm_alloc((size_t) C * C * sizeof(float), 128);
    rw.p32 = lab_vtcm_alloc((size_t) C * C * sizeof(float), 128);
    rw.s32 = lab_vtcm_alloc((size_t) D * D * sizeof(float), 128);
    rw.f16_stage = lab_vtcm_alloc((size_t) C * D * sizeof(__fp16), 128);
    rw.act_q     = lab_vtcm_alloc((size_t) C * D * sizeof(__fp16), 128);
    rw.wt_kt     = lab_vtcm_alloc((size_t) D * C * sizeof(__fp16), 128);

    float * f32_src = lab_vtcm_alloc((size_t) C * D * sizeof(float), 128);
    float * f32_dst = lab_vtcm_alloc((size_t) D * D * sizeof(float), 128);
    float * scales  = lab_vtcm_alloc((size_t) C * sizeof(float), 128);

    lab_fill_f32(rw.a32, (size_t) C * C, -0.2f, 0.2f);
    lab_fill_f32(rw.p32, (size_t) C * C, -1.0f, 1.0f);
    lab_fill_f32(rw.s32, (size_t) D * D, -0.5f, 0.5f);
    lab_fill_f32(f32_src, (size_t) C * D, -1.0f, 1.0f);
    for (uint32_t i = 0; i < C; i++) {
        scales[i] = 0.5f;
    }

    uint64_t total_cycles = 0;
    const uint32_t cc_blocks = C * C / VLEN_FP32;      // 32-lane blocks of a C x C matrix
    const uint32_t dd_blocks = D * D / VLEN_FP32;
    const uint32_t cd_blocks = C * D / VLEN_FP32;

    TIME_HELPER("decay_matrix", cc_blocks, gdn_ch_decay_matrix(&rw, &ga));
    TIME_HELPER("solve",        cc_blocks, gdn_ch_solve(&rw));
    TIME_HELPER("scale_p",      cc_blocks, gdn_ch_scale_p(&rw, 0.125f));
    TIME_HELPER("transpose_f32", dd_blocks, gdn_ch_transpose_f32(f32_dst, rw.s32, D));
    TIME_HELPER("f32_to_f16_rows", cd_blocks,
                gdn_ch_f32_to_f16_rows(rw.f16_stage, f32_src, C, D, D, scales));
    printf("lab: %s the five helpers together %llu cycles, %.2f us at 2112 MHz\n", TARGET,
           (unsigned long long) total_cycles, (double) total_cycles / 2112.0);

    // E[t][s] = exp(gamma_t - gamma_s) for s <= t, and 0 above the diagonal. The check runs after
    // the timing, thus it never enters a measured window.
    gdn_ch_decay_matrix(&rw, &ga);
    double se = 0.0, sr = 0.0, worst = 0.0;
    for (uint32_t t = 0; t < C; t++) {
        for (uint32_t s = 0; s < C; s++) {
            const double ref = (s <= t) ? exp((double) ga.gam[t] - (double) ga.gam[s]) : 0.0;
            const double got = (double) rw.e32[(size_t) t * C + s];
            const double d   = got - ref;
            se += d * d;
            sr += ref * ref;
            if (fabs(d) > worst) { worst = fabs(d); }
        }
    }
    printf("lab: %s decay_matrix nmse %.3g worst abs %.3g\n", TARGET, sr > 0 ? se / sr : 0.0, worst);
    lab_report(TARGET, "nmse_decay", sr > 0 ? se / sr : 0.0, "");

    // The transpose must be exact: dst[c][r] = src[r][c] for every element.
    gdn_ch_transpose_f32(f32_dst, rw.s32, D);
    size_t bad = 0;
    for (uint32_t r = 0; r < D && bad < 4; r++) {
        for (uint32_t c = 0; c < D; c++) {
            if (f32_dst[(size_t) c * D + r] != rw.s32[(size_t) r * D + c]) {
                if (bad < 4) {
                    printf("lab: %s transpose bad at r %u c %u: got %g want %g\n", TARGET, r, c,
                           (double) f32_dst[(size_t) c * D + r], (double) rw.s32[(size_t) r * D + c]);
                }
                bad++;
            }
        }
    }
    printf("lab: %s transpose_f32 %s\n", TARGET, bad ? "WRONG" : "exact");
    lab_report(TARGET, "transpose_bad", (double) bad, "");

    lab_report(TARGET, "chunk_C", C, "");
    lab_report(TARGET, "head_D", D, "");
    lab_report(TARGET, "hvx_helpers_cycles", (double) total_cycles, "cycles");
    return 0;
}
