// Target: the forward substitution of version 2 of the chunked gated delta net kernel
// (gdn_c2_solve of gdn-chunk-ops.c) against a float64 reference.
//
// T = (I + A')^-1 with A'[i][s] = beta_i * a[i][s] for s < i, where a = E * (K K^T) of one chunk. The
// program builds a from unit keys and the decay of a gate, runs the solve, and reports the worst error
// and the count of values that are not finite. A fast gate (--gate_min_milli -20000) gives an E whose
// entries far from the diagonal are 0 or below the f16 range, the case of the fallback of E.
//
// Arguments: --gate_min_milli -500 --d 128 --iters 1
// lab-run: mode=functional
#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "gdn-chunk-ops.c"
#include "gated-delta-net-ops.c"

#define TARGET "gdnsolve"

int main(int argc, char ** argv) {
    const float    gate_min = (float) lab_arg_long(argc, argv, "--gate_min_milli", -500) / 1000.0f;
    const uint32_t D        = (uint32_t) lab_arg_long(argc, argv, "--d", 128);
    const uint32_t iters    = (uint32_t) lab_arg_long(argc, argv, "--iters", 1);
    const uint32_t C        = GDN_CH_C;

    lab_init();
    printf("lab: %s gate_min %.3f D %u\n", TARGET, (double) gate_min, D);

    float * a32 = lab_vtcm_alloc((size_t) C * C * sizeof(float), 128);
    float * t32 = lab_vtcm_alloc((size_t) C * C * sizeof(float), 128);
    static float  bet[GDN_CH_C] __attribute__((aligned(128)));
    static double k[GDN_CH_C][128];
    static double gam[GDN_CH_C];
    static double ref[GDN_CH_C][GDN_CH_C];

    double acc = 0.0;
    for (uint32_t t = 0; t < C; t++) {
        double n = 0.0;
        for (uint32_t i = 0; i < D; i++) {
            k[t][i] = lab_rand_f32(-1.0f, 1.0f);
            n += k[t][i] * k[t][i];
        }
        for (uint32_t i = 0; i < D; i++) {
            k[t][i] /= sqrt(n);
        }
        acc += lab_rand_f32(gate_min, -1e-4f);
        gam[t] = acc;
        bet[t] = lab_rand_f32(0.0f, 1.0f);
    }
    // a[t][s] = E[t][s] * (k_t . k_s) for s <= t, rounded as the kernel sees it: the dot product is an
    // f16 HMX output and E is f32
    for (uint32_t t = 0; t < C; t++) {
        for (uint32_t s = 0; s < C; s++) {
            double dot = 0.0;
            for (uint32_t i = 0; i < D; i++) {
                dot += k[t][i] * k[s][i];
            }
            const double e = s <= t ? exp(gam[t] - gam[s]) : 0.0;
            float a = (float) (e * (double) lab_hf_to_f32(lab_f32_to_hf((float) dot)));
            // the kernel flushes A' below 2^-50 before the solve (gdn_c2_ftz)
            a32[(size_t) t * C + s] = fabsf(a) < 0x1p-50f ? 0.0f : a;
        }
    }

    uint64_t best = UINT64_MAX;
    for (uint32_t it = 0; it < iters + 1; it++) {
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        gdn_c2_solve(t32, a32, bet);
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        best = (t1 - t0) < best ? (t1 - t0) : best;
    }

    // the reference: row i of T = e_i - beta_i sum_{s<i} a[i][s] T[s]
    for (uint32_t i = 0; i < C; i++) {
        for (uint32_t j = 0; j < C; j++) {
            double v = i == j ? 1.0 : 0.0;
            for (uint32_t s = 0; s < i; s++) {
                v -= (double) bet[i] * (double) a32[(size_t) i * C + s] * ref[s][j];
            }
            ref[i][j] = v;
        }
    }
    size_t bad = 0;
    double worst = 0.0;
    uint32_t wi = 0, wj = 0;
    for (uint32_t i = 0; i < C; i++) {
        for (uint32_t j = 0; j < C; j++) {
            const double g = (double) t32[(size_t) i * C + j];
            if (!isfinite(g)) {
                if (bad < 8) {
                    printf("lab: %s not finite at row %u lane %u: %g (ref %g)\n", TARGET, i, j, g, ref[i][j]);
                }
                bad++;
                continue;
            }
            const double d = fabs(g - ref[i][j]);
            if (d > worst) {
                worst = d;
                wi = i;
                wj = j;
            }
        }
    }
    printf("lab: %s worst abs error %.3g at row %u lane %u (got %g ref %g)\n", TARGET, worst, wi, wj,
           (double) t32[(size_t) wi * C + wj], ref[wi][wj]);
    // the first row, in the order of the solve, with an error above 1e-5, and its lanes
    for (uint32_t i = 0; i < C; i++) {
        for (uint32_t j = 0; j < C; j++) {
            const double g = (double) t32[(size_t) i * C + j];
            if (!(fabs(g - ref[i][j]) <= 1e-5)) {
                printf("lab: %s first bad row %u lane %u got %g ref %g (a[%u][%u] = %g)\n", TARGET, i, j, g, ref[i][j],
                       i, j, (double) a32[(size_t) i * C + j]);
                for (uint32_t s = 0; s < i; s++) {
                    printf("lab: %s   a[%u][%u] = %g  T[%u][%u] = %g ref %g\n", TARGET, i, s,
                           (double) a32[(size_t) i * C + s], s, j, (double) t32[(size_t) s * C + j], ref[s][j]);
                }
                i = C;
                break;
            }
        }
    }
    lab_report(TARGET, "cycles", (double) best, "cycles");
    lab_report(TARGET, "not_finite", (double) bad, "");
    lab_report(TARGET, "worst_abs", worst, "");
    return (bad == 0 && worst < 1e-5) ? 0 : 1;
}
