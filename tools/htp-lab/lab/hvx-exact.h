// Integer-only IEEE f32 and f16 arithmetic for the HVX.
//
// The qf32 and qf16 instructions of the HVX do not round the same way on v73/v75 and on v79/v81,
// and the IEEE-form float opcodes give inf on v79 silicon. A kernel that uses either gives
// different output on different chips. These routines use only integer instructions, which have
// one definition from v60, thus each chip gives the same bits. Each result is the IEEE value with
// round to nearest even, with these limits:
//   - An f32 subnormal input is zero, and an f32 result below 2^-126 before the rounding is a
//     zero with the sign of the exact result (flush to zero). The f16 conversions keep the f16
//     subnormals.
//   - A NaN result is the quiet NaN 0x7fc00000 (f32) or 0x7e00 (f16).
//
// One lane is one bit pattern in a 32-bit word, thus each routine processes 32 values. An f16 value
// is in the low halfword of its word, and the high halfword is zero.

#ifndef HVX_EXACT_H
#define HVX_EXACT_H

#include <stdint.h>

#include "hvx-types.h"

#define HVX_EXACT_SPLAT(x) Q6_V_vsplat_R((int32_t) (x))

// The f32 value of an f16 value. The conversion is exact for every f16 value.
static inline HVX_Vector hvx_exact_hf_to_sf(HVX_Vector h) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector sign = Q6_Vw_vasl_VwR(Q6_V_vand_VV(h, HVX_EXACT_SPLAT(0x8000)), 16);
    const HVX_Vector em   = Q6_V_vand_VV(h, HVX_EXACT_SPLAT(0x7fff));

    // A normal value, Inf or NaN: move the fields up by 13 bits and add the exponent bias
    // difference, 112 for a finite value and 224 for Inf and NaN (exponent 31 goes to 255).
    const HVX_VectorPred q_infnan = Q6_Q_vcmp_gt_VwVw(em, HVX_EXACT_SPLAT(0x7bff));
    const HVX_Vector     bias     = Q6_V_vmux_QVV(q_infnan, HVX_EXACT_SPLAT(0x70000000), HVX_EXACT_SPLAT(0x38000000));
    const HVX_Vector     r_norm   = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(em, 13), bias);

    // A subnormal value em * 2^-24: the leading one of em is at bit 31 - lz. A shift by lz - 8
    // puts it at bit 23, and the exponent field is 133 - lz, because the leading one adds one.
    const HVX_Vector lz     = Q6_Vuw_vcl0_Vuw(em);
    const HVX_Vector r_sub  = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwVw(em, Q6_Vw_vsub_VwVw(lz, HVX_EXACT_SPLAT(8))),
                                              Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(HVX_EXACT_SPLAT(133), lz), 23));
    const HVX_VectorPred q_sub = Q6_Q_vcmp_gt_VwVw(HVX_EXACT_SPLAT(0x0400), em);

    HVX_Vector r = Q6_V_vmux_QVV(q_sub, r_sub, r_norm);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(em, zero), zero, r);
    return Q6_V_vor_VV(r, sign);
}

// The f16 value of an f32 value, with round to nearest even. A value of 65520 or more becomes Inf,
// and a value below 2^-14 becomes an f16 subnormal or zero. The result is in the low halfword.
static inline HVX_Vector hvx_exact_sf_to_hf(HVX_Vector x) {
    const HVX_Vector one  = HVX_EXACT_SPLAT(1);
    const HVX_Vector sign = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(x, 16), HVX_EXACT_SPLAT(0x8000));
    const HVX_Vector a    = Q6_V_vand_VV(x, HVX_EXACT_SPLAT(0x7fffffff));

    // A normal result: take off the bias difference 112 << 23, add the half ulp 0xfff and the odd
    // bit for the tie, and shift. A carry moves the value to the next exponent, and an f32 value
    // of 65520 or more gives 0x7c00 or more, thus the minimum with 0x7c00 gives Inf.
    const HVX_Vector odd    = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(a, 13), one);
    HVX_Vector       r_norm = Q6_Vw_vadd_VwVw(a, Q6_Vw_vadd_VwVw(odd, HVX_EXACT_SPLAT(0x0fff - 0x38000000)));
    r_norm = Q6_Vw_vmin_VwVw(Q6_Vuw_vlsr_VuwR(r_norm, 13), HVX_EXACT_SPLAT(0x7c00));

    // A subnormal result: the value m * 2^(e - 150) becomes m >> (126 - e) units of 2^-24. The
    // dropped bits, moved to the top of the word, are more than half when they are above
    // 0x80000000, and the odd bit of the quotient decides a tie. A shift of 31 gives zero, because
    // m is below 2^24.
    const HVX_Vector     e     = Q6_Vuw_vlsr_VuwR(a, 23);
    const HVX_Vector     m     = Q6_V_vor_VV(Q6_V_vand_VV(a, HVX_EXACT_SPLAT(0x007fffff)), HVX_EXACT_SPLAT(0x00800000));
    const HVX_Vector     sh    = Q6_Vw_vmin_VwVw(Q6_Vw_vsub_VwVw(HVX_EXACT_SPLAT(126), e), HVX_EXACT_SPLAT(31));
    HVX_Vector           q     = Q6_Vw_vlsr_VwVw(m, sh);
    const HVX_Vector     rest  = Q6_Vw_vasl_VwVw(m, Q6_Vw_vsub_VwVw(HVX_EXACT_SPLAT(32), sh));
    const HVX_VectorPred q_up  = Q6_Q_vcmp_gt_VuwVuw(Q6_V_vor_VV(rest, Q6_V_vand_VV(q, one)), HVX_EXACT_SPLAT(0x80000000));
    q = Q6_Vw_condacc_QVwVw(q_up, q, one);

    HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(HVX_EXACT_SPLAT(0x38800000), a), q, r_norm);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(a, HVX_EXACT_SPLAT(0x7f800000)), HVX_EXACT_SPLAT(0x7e00), r);
    return Q6_V_vor_VV(r, sign);
}

// a * b with round to nearest even. The product of the two 24-bit significands is exact in 48 bits
// (vmpye and vmpyoacc give the 64-bit product), and one rounding makes the f32 result.
static inline HVX_Vector hvx_exact_sf_mul(HVX_Vector a, HVX_Vector b) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = HVX_EXACT_SPLAT(1);
    const HVX_Vector k_ff = HVX_EXACT_SPLAT(0xff);
    const HVX_Vector k_m  = HVX_EXACT_SPLAT(0x007fffff);
    const HVX_Vector k_h  = HVX_EXACT_SPLAT(0x00800000);

    const HVX_Vector sign = Q6_V_vand_VV(Q6_V_vxor_VV(a, b), HVX_EXACT_SPLAT(0x80000000));
    const HVX_Vector ea   = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(a, 23), k_ff);
    const HVX_Vector eb   = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(b, 23), k_ff);
    const HVX_Vector ma   = Q6_V_vor_VV(Q6_V_vand_VV(a, k_m), k_h);
    const HVX_Vector mb   = Q6_V_vor_VV(Q6_V_vand_VV(b, k_m), k_h);

    HVX_VectorPair   p  = Q6_W_vmpye_VwVuh(ma, mb);
    p                   = Q6_W_vmpyoacc_WVwVh(p, ma, mb);
    const HVX_Vector hi = Q6_V_hi_W(p);
    const HVX_Vector lo = Q6_V_lo_W(p);

    // t is 1 when the product is in [2^47, 2^48). The significand is then bits 47..24, else bits
    // 46..23. The dropped bits of lo, moved to the top of the word, decide the rounding.
    const HVX_Vector     t    = Q6_Vuw_vlsr_VuwR(hi, 15);
    const HVX_Vector     s9   = Q6_Vw_vsub_VwVw(HVX_EXACT_SPLAT(9), t);
    HVX_Vector           m    = Q6_V_vor_VV(Q6_Vw_vasl_VwVw(hi, s9), Q6_Vw_vlsr_VwVw(lo, Q6_Vw_vadd_VwVw(t, HVX_EXACT_SPLAT(23))));
    const HVX_Vector     rest = Q6_Vw_vasl_VwVw(lo, s9);
    const HVX_VectorPred q_up = Q6_Q_vcmp_gt_VuwVuw(Q6_V_vor_VV(rest, Q6_V_vand_VV(m, one)), HVX_EXACT_SPLAT(0x80000000));
    m = Q6_Vw_condacc_QVwVw(q_up, m, one);

    // e2 + 1 is the biased exponent before a carry of the rounding. m holds the hidden one, which
    // adds 1 to the exponent field, and a carry to 2^24 adds one more.
    const HVX_Vector e2 = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(ea, eb), Q6_Vw_vsub_VwVw(t, HVX_EXACT_SPLAT(128)));
    HVX_Vector       r  = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(e2, 23), m);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(zero, e2), zero, r);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e2, HVX_EXACT_SPLAT(253)), HVX_EXACT_SPLAT(0x7f800000), r);

    // A zero or subnormal input gives zero. Inf gives Inf, and NaN or Inf * 0 gives NaN.
    const HVX_VectorPred q_z   = Q6_Q_or_QQ(Q6_Q_vcmp_eq_VwVw(ea, zero), Q6_Q_vcmp_eq_VwVw(eb, zero));
    const HVX_VectorPred q_s   = Q6_Q_or_QQ(Q6_Q_vcmp_eq_VwVw(ea, k_ff), Q6_Q_vcmp_eq_VwVw(eb, k_ff));
    const HVX_VectorPred q_nan = Q6_Q_or_QQ(q_z, Q6_Q_or_QQ(Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(a, HVX_EXACT_SPLAT(0x7fffffff)), HVX_EXACT_SPLAT(0x7f800000)),
                                                             Q6_Q_vcmp_gt_VuwVuw(Q6_V_vand_VV(b, HVX_EXACT_SPLAT(0x7fffffff)), HVX_EXACT_SPLAT(0x7f800000))));
    r = Q6_V_vmux_QVV(q_z, zero, r);
    r = Q6_V_vmux_QVV(q_s, Q6_V_vmux_QVV(q_nan, HVX_EXACT_SPLAT(0x7fc00000), HVX_EXACT_SPLAT(0x7f800000)), r);
    return Q6_V_vor_VV(r, Q6_V_vmux_QVV(Q6_Q_and_QQn(q_s, q_nan), sign, Q6_V_vmux_QVV(q_s, zero, sign)));
}

// a + b with round to nearest even. The operand with the larger magnitude is A. The significands
// get 7 guard bits, the shift of B keeps a sticky bit, and one rounding makes the f32 result.
static inline HVX_Vector hvx_exact_sf_add(HVX_Vector a, HVX_Vector b) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = HVX_EXACT_SPLAT(1);
    const HVX_Vector k_ab = HVX_EXACT_SPLAT(0x7fffffff);
    const HVX_Vector k_m  = HVX_EXACT_SPLAT(0x007fffff);
    const HVX_Vector k_h  = HVX_EXACT_SPLAT(0x00800000);

    const HVX_Vector     aa  = Q6_V_vand_VV(a, k_ab);
    const HVX_Vector     ab  = Q6_V_vand_VV(b, k_ab);
    const HVX_VectorPred q_x = Q6_Q_vcmp_gt_VwVw(ab, aa);
    const HVX_Vector     A   = Q6_V_vmux_QVV(q_x, b, a);
    const HVX_Vector     AA  = Q6_V_vmux_QVV(q_x, ab, aa);
    const HVX_Vector     BB  = Q6_V_vmux_QVV(q_x, aa, ab);
    const HVX_Vector     eA  = Q6_Vuw_vlsr_VuwR(AA, 23);
    const HVX_Vector     eB  = Q6_Vuw_vlsr_VuwR(BB, 23);

    // A subnormal B is zero. When A is subnormal, B is too, and the result is zero.
    const HVX_Vector mA = Q6_Vw_vasl_VwR(Q6_V_vor_VV(Q6_V_vand_VV(AA, k_m), k_h), 7);
    HVX_Vector       mB = Q6_Vw_vasl_VwR(Q6_V_vor_VV(Q6_V_vand_VV(BB, k_m), k_h), 7);
    mB = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(eB, zero), zero, mB);

    // Align B. A shift of 31 or more leaves zero, because mB is below 2^31. The sticky bit is one
    // when the shift drops a one.
    const HVX_Vector     d    = Q6_Vw_vmin_VwVw(Q6_Vw_vsub_VwVw(eA, eB), HVX_EXACT_SPLAT(31));
    HVX_Vector           mBs  = Q6_Vw_vlsr_VwVw(mB, d);
    const HVX_VectorPred q_st = Q6_Q_vcmp_eq_VwVw(Q6_Vw_vasl_VwVw(mBs, d), mB);
    mBs = Q6_V_vmux_QVV(q_st, mBs, Q6_V_vor_VV(mBs, one));

    const HVX_VectorPred q_sub = Q6_Q_vcmp_gt_VwVw(zero, Q6_V_vxor_VV(a, b));
    HVX_Vector           M     = Q6_V_vmux_QVV(q_sub, Q6_Vw_vsub_VwVw(mA, mBs), Q6_Vw_vadd_VwVw(mA, mBs));

    // Put the leading one at bit 30. A carry (lz 0) shifts right by one and keeps the sticky bit.
    // The exponent is eA + 1 - lz in both cases.
    const HVX_Vector lz = Q6_Vuw_vcl0_Vuw(M);
    const HVX_Vector Mc = Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(M, 1), Q6_V_vand_VV(M, one));
    const HVX_Vector Mn = Q6_Vw_vasl_VwVw(M, Q6_Vw_vsub_VwVw(lz, one));
    M                   = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(lz, zero), Mc, Mn);
    const HVX_Vector e  = Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(eA, one), lz);

    // Round to nearest even at bit 7.
    HVX_Vector           m    = Q6_Vuw_vlsr_VuwR(M, 7);
    const HVX_Vector     rest = Q6_Vw_vasl_VwR(M, 25);
    const HVX_VectorPred q_up = Q6_Q_vcmp_gt_VuwVuw(Q6_V_vor_VV(rest, Q6_V_vand_VV(m, one)), HVX_EXACT_SPLAT(0x80000000));
    m = Q6_Vw_condacc_QVwVw(q_up, m, one);

    HVX_Vector r = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(e, one), 23), m);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(one, e), zero, r);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e, HVX_EXACT_SPLAT(254)), HVX_EXACT_SPLAT(0x7f800000), r);
    r = Q6_V_vor_VV(r, Q6_V_vand_VV(A, HVX_EXACT_SPLAT(0x80000000)));
    // An exact cancellation gives +0. Two zero inputs give -0 only when both are negative.
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(M, zero), zero, r);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(eA, zero), Q6_V_vand_VV(Q6_V_vand_VV(a, b), HVX_EXACT_SPLAT(0x80000000)), r);

    // Inf and NaN: A has the larger magnitude, thus A is Inf or NaN when either is. NaN, and Inf
    // minus Inf, give NaN. Otherwise the result is A.
    const HVX_VectorPred q_s   = Q6_Q_vcmp_gt_VwVw(AA, HVX_EXACT_SPLAT(0x7f7fffff));
    const HVX_VectorPred q_nan = Q6_Q_or_QQ(Q6_Q_vcmp_gt_VwVw(AA, HVX_EXACT_SPLAT(0x7f800000)),
                                            Q6_Q_and_QQ(q_sub, Q6_Q_vcmp_eq_VwVw(BB, HVX_EXACT_SPLAT(0x7f800000))));
    return Q6_V_vmux_QVV(q_s, Q6_V_vmux_QVV(q_nan, HVX_EXACT_SPLAT(0x7fc00000), A), r);
}

#endif /* HVX_EXACT_H */
