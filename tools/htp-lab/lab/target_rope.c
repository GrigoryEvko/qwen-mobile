// Target: the sin and cos of the HTP rope cache (hvx-sin-cos.h) at the angles of the Qwen3.5 rope.
//
// The rope of the text layers (IMROPE, n_dims 64, base 1e7) turns pair k of the token at position p
// by theta_k = p * base^(-2k/64). The kernel makes theta_k with f32 products, as the CPU does, and
// then calls hvx_vec_sincos_f32. The CPU and the oracle compute sinf and cosf of the same f32
// angle, thus their error does not grow with the position. A range reduction that is not exact
// (for example y = x - n pi with pi and the product rounded to f32) gives an error that grows with
// n, thus with the position: 2.8e-3 at position 32768.
//
// For each position bucket (256 positions from p0, up to the end of a 262144-token context), the
// program compares the kernel sin and cos with the float64 sin and cos of the same f32 angle. It
// stops with the status 1 when an error is more than 2e-6. It also counts the cycles of one call
// for one vector of 32 angles.
//
// Arguments: --iters 200
#include "lab.h"

#include <math.h>
#include <stdio.h>

#include "hvx-utils.h"
#include "hvx-sin-cos.h"

#define TARGET "rope"

static float angles[32] __attribute__((aligned(128)));
static float out_c[32] __attribute__((aligned(128)));
static float out_s[32] __attribute__((aligned(128)));

// The 32 angles of one token, made as the kernel makes them (theta *= theta_scale for each pair).
static void model_angles(int32_t p, float theta_scale) {
    float theta = (float) p;
    for (int k = 0; k < 32; k++) {
        angles[k] = theta;
        theta *= theta_scale;
    }
}

// Update the largest errors of the kernel sin and cos over the angles, against float64.
static void measure(double * e_cos, double * e_sin) {
    HVX_Vector vc, vs;
    hvx_vec_sincos_f32(*(const HVX_Vector *) angles, &vc, &vs);
    *(HVX_Vector *) out_c = vc;
    *(HVX_Vector *) out_s = vs;
    for (int k = 0; k < 32; k++) {
        const double a  = (double) angles[k];
        const double ec = fabs((double) out_c[k] - cos(a));
        const double es = fabs((double) out_s[k] - sin(a));
        *e_cos = ec > *e_cos ? ec : *e_cos;
        *e_sin = es > *e_sin ? es : *e_sin;
    }
}

int main(int argc, char ** argv) {
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 200);
    lab_init();

    const float theta_scale = powf(1e7f, -2.0f / 64.0f);
    static const int32_t p0s[] = { 0, 512, 4096, 8192, 32768, 65536, 131072, 262144 - 256 };
    int bad = 0;
    for (size_t i = 0; i < sizeof(p0s) / sizeof(p0s[0]); i++) {
        double ec = 0.0, es = 0.0;
        for (int32_t p = p0s[i]; p < p0s[i] + 256; p++) {
            model_angles(p, theta_scale);
            measure(&ec, &es);
        }
        printf("lab: %s pos %6d..%6d  max|dcos| %.3g max|dsin| %.3g\n", TARGET, p0s[i], p0s[i] + 255, ec, es);
        bad += ec > 2e-6 || es > 2e-6;
    }

    // the cycles of one call for one vector of 32 angles, the best of 3 runs of `iters` calls
    model_angles(32768, theta_scale);
    HVX_Vector x    = *(const HVX_Vector *) angles;
    uint64_t   best = UINT64_MAX;
    for (int rep = 0; rep < 3; rep++) {
        HVX_Vector vc, vs;
        LAB_BARRIER();
        const uint64_t t0 = lab_cycles();
        for (uint32_t it = 0; it < iters; it++) {
            hvx_vec_sincos_f32(x, &vc, &vs);
            x = Q6_V_vor_VV(x, Q6_V_vand_VV(vc, Q6_V_vzero()));
        }
        const uint64_t t1 = lab_cycles();
        LAB_BARRIER();
        *(HVX_Vector *) out_s = vs;
        best = t1 - t0 < best ? t1 - t0 : best;
    }
    printf("lab: %s cycles %.1f per vector of 32 angles\n", TARGET, (double) best / iters);
    printf("lab: %s status %s\n", TARGET, bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
