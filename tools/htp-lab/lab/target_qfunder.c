// Target: the qf32 values near and below the smallest normal f32, through the conversions that the
// gated delta net kernels use: Vsf_equals_Vqf32 (to f32) and Vhf_equals_Wqf32 (to f16), from a
// product vmpy(sf, sf), from a sum of such products, and from a product with an exact zero.
//
// The program prints, for each magnitude, the value that each conversion gives and the value that it
// must give (the IEEE product, or 0 below the f32 subnormal range). A conversion that wraps the
// exponent gives a large number in place of a small one.
#include "lab.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hvx-utils.h"

#define TARGET "qfunder"

int main(void) {
    lab_init();
    static float a[32] __attribute__((aligned(128)));
    static float b[32] __attribute__((aligned(128)));
    static float o[32] __attribute__((aligned(128)));
    static __fp16 h[64] __attribute__((aligned(128)));

    // lane i: a product of about 10^(-30 - i / 2)
    for (int i = 0; i < 32; i++) {
        a[i] = powf(10.0f, -15.0f - (float) i / 4.0f);
        b[i] = powf(10.0f, -15.0f - (float) i / 4.0f);
    }
    a[31] = 0.0f;
    b[31] = 1.5f;
    const HVX_Vector va = hvx_vmem(a);
    const HVX_Vector vb = hvx_vmem(b);
    const HVX_Vector p  = Q6_Vqf32_vmpy_VsfVsf(va, vb);
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(p);
    hvx_vmem(h) = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(p, p));
    size_t bad = 0;
    for (int i = 0; i < 32; i++) {
        const double want = (double) a[i] * (double) b[i];
        printf("lab: %s mul lane %2d want %.3e sf %.3e hf %.3e\n", TARGET, i, want, (double) o[i],
               (double) (float) h[2 * i]);
        if (!(fabs((double) o[i]) < 1e-30) && want < 1e-30) {
            bad++;
        }
    }
    // a sum of two tiny products
    const HVX_Vector s = Q6_Vqf32_vadd_Vqf32Vqf32(p, p);
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(s);
    for (int i = 20; i < 32; i++) {
        printf("lab: %s sum lane %2d want %.3e sf %.3e\n", TARGET, i, 2.0 * (double) a[i] * (double) b[i],
               (double) o[i]);
    }
    // a subnormal f32 as an input of the multiply and of the add
    static const float sub[8] = { 1e-40f, 1e-39f, 5e-39f, 1.1e-38f, -1e-39f, 1.4e-45f, 3e-39f, 1e-38f };
    static const float mul[4] = { 1.0f, 2.0f, 1e10f, -0.5f };
    for (int i = 0; i < 32; i++) {
        a[i] = sub[i % 8];
        b[i] = mul[i / 8];
    }
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(a), hvx_vmem(b)));
    size_t wrong = 0;
    for (int i = 0; i < 32; i++) {
        const double want = (double) a[i] * (double) b[i];
        const bool   ok   = fabs((double) o[i] - want) <= 1e-3 * fabs(want) + 1e-44;
        wrong += ok ? 0 : 1;
        printf("lab: %s subnormal mul lane %2d a %.3e b %.3e want %.3e got %.3e %s\n", TARGET, i, (double) a[i],
               (double) b[i], want, (double) o[i], ok ? "" : "WRONG");
    }
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(hvx_vmem(a), Q6_V_vzero()));
    for (int i = 0; i < 8; i++) {
        printf("lab: %s subnormal add lane %d a %.3e got %.3e\n", TARGET, i, (double) a[i], (double) o[i]);
    }
    // zeros: -0 and +0 times a normal value, the qf32 of an all-zero vector plus a product, and the
    // f32 of the sum of two all-zero vectors
    static const float zc[8] = { 1.0f, 0.8f, 3.5f, 1e-5f, 7.9e-5f, 1e10f, -2.0f, 0.3f };
    for (int i = 0; i < 32; i++) {
        a[i] = (i < 16) ? -0.0f : 0.0f;
        b[i] = zc[i % 8];
    }
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(a), hvx_vmem(b)));
    for (int i = 0; i < 32; i += 3) {
        printf("lab: %s zero mul lane %2d a %s0 b %.3e got %.3e\n", TARGET, i, i < 16 ? "-" : "+", (double) b[i],
               (double) o[i]);
    }
    const HVX_Vector z = Q6_V_vzero();
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(z, z));
    printf("lab: %s sf(qf32 zero + qf32 zero) lane 0 got %.3e\n", TARGET, (double) o[0]);
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(z, z)),
                                                           hvx_vec_splat_f32(-0.8f)));
    printf("lab: %s sf(sf(zero + zero) * -0.8) lane 0 got %.3e\n", TARGET, (double) o[0]);
    const HVX_Vector pz = Q6_Vqf32_vmpy_VsfVsf(hvx_vmem(a), hvx_vmem(b));
    hvx_vmem(o) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(pz, z));
    for (int i = 0; i < 32; i += 3) {
        printf("lab: %s sf(zero product + qf32 zero) lane %2d a %s0 b %.3e got %.3e\n", TARGET, i, i < 16 ? "-" : "+",
               (double) b[i], (double) o[i]);
    }
    lab_report(TARGET, "wrapped_lanes", (double) bad, "");
    lab_report(TARGET, "subnormal_mul_wrong", (double) wrong, "");
    return 0;
}
