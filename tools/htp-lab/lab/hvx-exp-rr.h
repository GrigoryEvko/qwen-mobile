// THE VERDICT, MEASURED, AND WHY THIS FILE IS IN THE LAB AND NOT IN THE KERNEL.
// In the streaming form that the row loop of the softmax actually runs, over 64 independent
// vectors with nothing carried between groups:
//
//     the exponential of softmax-ops.c, six Taylor multiplies   34.47 cycles for each vector
//     this routine, three multiplies and one table lookup       43.47 cycles for each vector
//
// Half the multiplies and 26 % slower. vlut16 is CVI_VP_VS, thus the permute unit takes one of
// them per packet: the two lookups and the three operations that build the index cost more
// packets than the three qfloat multiplies they remove. Accuracy is not the reason to prefer
// either, the worst relative error over [-20, 20] being 9.775e-07 here against 1.062e-06 there.
//
// The lesson generalises, and it is the opposite of the int16 tables elsewhere in this tree: a
// vlut16 lookup pays when it replaces a long dependent chain, as it does for a sigmoid in
// int16, and it loses when it replaces a handful of multiplies that already pipeline.
//
// Kept here, buildable and measurable, so that nobody spends the same day on it again.

// A second range-reduction stage for the f32 exponential of softmax-ops.c.
//
// WHAT IS ALREADY DONE. hvx_vec_exp_f32_xn of softmax-ops.c reduces with the forced mantissa
// and rounds to nearest, which removed hvx_vec_floor_f32 and hvx_vec_truncate_f32 (about 27
// instructions and 18 chain steps) and halved the reduced range. Six Taylor terms remain, thus
// six qfloat multiplies, and a v79 packet holds one of those.
//
// WHAT THIS ADDS. Reduce a second time, in sixteenths of an octave, and read the middle stage
// from a table:
//
//   T = x * 16*log2(e),  n = round(T),  w = T - n in [-0.5, 0.5]
//   k = n >> 4,  i = n & 15
//   exp(x) = 2^k * 2^(i/16) * 2^(w/16)
//
// 2^(w/16) needs only degree 3, because w/16 lies in [-1/32, 1/32]: the first term left out is
// (ln2/16)^4/24 * w^4, at most 9.2e-09, below the resolution of an f32. That is three qfloat
// multiplies for the series and one for the table value, against six. 2^(i/16) is one of
// sixteen f32 constants read by vlut16, and 2^k is the same exponent insertion as before.
//
// The routine stays in f32 throughout. The table holds f32 constants and not int16, thus the
// precision policy of the softmax is intact.
//
// THE LANE MAPPING OF vlut16 in the 128-byte mode, measured on the v79 simulator over all 32
// values of Rt and all 256 byte values: an index byte matches when its high nibble is equal to
// Rt & 15, it then reads the table halfword 2*(byte mod 32) + ((Rt >> 1) & 1), and a byte that
// does not match gives 0. The output pair splits by byte position: the low vector holds the
// results of the even byte positions. Entry i of an f32 table is the halfword pair (2i, 2i+1):
//   - byte 4i of the index, the low byte of word i, with Rt = 0 reads the low half into the low
//     vector at halfword 2i, which is where the low half of word i belongs, and
//   - byte 4i+2, holding 32+i, with Rt = 2 reads the high half into halfword 2i+1.
// Each lookup leaves the other halfword zero, thus one vor of the two low vectors is the whole
// f32 result and no shuffle is needed.

#ifndef HVX_EXP_RR_H
#define HVX_EXP_RR_H

#include <stdint.h>

#include "hvx-base.h"

#define EXP_RR_LOG2E_16 0x41b8aa3b  // 16*log2(e)
#define EXP_RR_MAGIC    0x4b400000  // 1.5 * 2^23, the round-to-nearest magic number
#define EXP_RR_A1       0x3d317218  // ln(2)/16
#define EXP_RR_A2       0x3a75fdf0  // (ln2/16)^2/2
#define EXP_RR_A3       0x37635847  // (ln2/16)^3/6
#define EXP_RR_ONE      0x3f800000
#define EXP_RR_RANGE_R  0x42b17218  // ln(FLT_MAX), 88.7228
#define EXP_RR_RANGE_L  0xc2b00000  // -88.0
#define EXP_RR_IDX_HI   0x00200000  // the value 32 in byte 2 of each word
#define EXP_RR_IDX_M    0x0000000f

// 2^(i/16) for i = 0 to 15 as f32, in the first 64 bytes. The rest is never read.
static const uint32_t hvx_exp_rr_tab[32] __attribute__((aligned(128))) = {
    0x3f800000, 0x3f85aac3, 0x3f8b95c2, 0x3f91c3d3, 0x3f9837f0, 0x3f9ef532, 0x3fa5fed7, 0x3fad583f, 0x3fb504f3, 0x3fbd08a4, 0x3fc5672a, 0x3fce248c, 0x3fd744fd, 0x3fe0ccdf, 0x3feac0c7, 0x3ff5257d,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// 2^(i/16) for each lane, i from the low four bits of each word of idx.
static inline __attribute__((always_inline)) HVX_Vector hvx_exp_rr_pow2_frac(HVX_Vector idx) {
    const HVX_Vector hi_tag = Q6_V_vsplat_R(EXP_RR_IDX_HI);
    const HVX_Vector tbl    = *(const HVX_Vector *) hvx_exp_rr_tab;

    const HVX_Vector ib = Q6_V_vor_VV(idx, Q6_V_vor_VV(Q6_Vw_vasl_VwR(idx, 16), hi_tag));
    const HVX_Vector lo = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(ib, tbl, 0));
    const HVX_Vector hi = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(ib, tbl, 2));
    return Q6_V_vor_VV(lo, hi);
}

// exp(x) for n vectors of 32 f32 lanes. n is a constant at the call. Each step runs for every
// vector before the next step starts, thus no packet waits for its own producer.
static inline __attribute__((always_inline))
void hvx_vec_exp_rr_f32_xn(const HVX_Vector * restrict in, HVX_Vector * restrict out, const int n) {
    const HVX_Vector zero = Q6_V_vzero();
    HVX_Vector x[4], t[4], m[4], nf[4], ni[4], w[4], y[4], p[4];

    {   // clamp to (-88, 88), thus the exponent insertion cannot overflow
        const HVX_Vector rr = Q6_V_vsplat_R(EXP_RR_RANGE_R);
        const HVX_Vector rl = Q6_V_vsplat_R(EXP_RR_RANGE_L);
        for (int r = 0; r < n; r++) {
            const HVX_VectorPred hi = Q6_Q_vcmp_gt_VsfVsf(in[r], rr);
            const HVX_VectorPred lo = Q6_Q_vcmp_gt_VsfVsf(rl, in[r]);
            x[r] = Q6_V_vmux_QVV(lo, rl, Q6_V_vmux_QVV(hi, rr, in[r]));
        }
    }
    {   // T = x * 16*log2(e)
        const HVX_Vector c = Q6_V_vsplat_R(EXP_RR_LOG2E_16);
        for (int r = 0; r < n; r++) {
            t[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x[r], c));
        }
    }
    {   // the magic number gives round(T) as a float and as an integer
        const HVX_Vector magic = Q6_V_vsplat_R(EXP_RR_MAGIC);
        for (int r = 0; r < n; r++) {
            m[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(t[r], magic));
        }
        for (int r = 0; r < n; r++) { ni[r] = Q6_Vw_vsub_VwVw(m[r], magic); }
        for (int r = 0; r < n; r++) {
            nf[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(m[r], magic));
        }
    }
    for (int r = 0; r < n; r++) {   // w = T - n, in [-0.5, 0.5]
        w[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(t[r], nf[r]));
    }
    {   // 2^(w/16) = 1 + a1 w + a2 w^2 + a3 w^3, by Horner
        const HVX_Vector c3 = Q6_V_vsplat_R(EXP_RR_A3);
        const HVX_Vector c2 = Q6_V_vsplat_R(EXP_RR_A2);
        for (int r = 0; r < n; r++) {
            y[r] = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(w[r], c3), c2);
        }
    }
    {
        const HVX_Vector c1 = Q6_V_vsplat_R(EXP_RR_A1);
        for (int r = 0; r < n; r++) {
            y[r] = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(y[r], Q6_Vqf32_vadd_VsfVsf(w[r], zero)), c1);
        }
    }
    {
        const HVX_Vector one = Q6_V_vsplat_R(EXP_RR_ONE);
        for (int r = 0; r < n; r++) {
            y[r] = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_Vqf32Vqf32(y[r], Q6_Vqf32_vadd_VsfVsf(w[r], zero)), one);
        }
    }
    {   // the sixteenth of an octave, from the table
        const HVX_Vector msk = Q6_V_vsplat_R(EXP_RR_IDX_M);
        for (int r = 0; r < n; r++) {
            p[r] = hvx_exp_rr_pow2_frac(Q6_V_vand_VV(ni[r], msk));
        }
    }
    for (int r = 0; r < n; r++) {   // y = 2^(w/16) * 2^(i/16)
        y[r] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(y[r], Q6_Vqf32_vadd_VsfVsf(p[r], zero)));
    }
    {   // multiply by 2^k, k = n >> 4, by adding k to the exponent field
        for (int r = 0; r < n; r++) { ni[r] = Q6_Vw_vasr_VwR(ni[r], 4); }
        for (int r = 0; r < n; r++) {
            HVX_Vector e = Q6_Vuw_vlsr_VuwR(Q6_Vw_vasl_VwR(y[r], 1), 24);
            e = Q6_Vw_vadd_VwVw(ni[r], e);
            const HVX_VectorPred under = Q6_Q_vcmp_gt_VwVw(zero, e);
            out[r] = Q6_V_vmux_QVV(under, zero, Q6_Vw_vaslacc_VwVwR(y[r], ni[r], 23));
        }
    }
}

#endif /* HVX_EXP_RR_H */
