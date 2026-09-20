#!/usr/bin/env python3
"""Generate hvx-act-i16.h: piecewise int16 evaluators for the activation family of the HTP.

An activation of this backend is a smooth function of one variable. Evaluated as a chain of
qfloat operations it costs about 4 cycles for each operation, because a v79 packet holds one
qfloat multiply and a result is ready two packets after its producer. Evaluated as a piecewise
polynomial in int16 it costs about 1 cycle for each operation: a packet holds two int16
multiplies or four integer adds, and `vlut16` reads the coefficients of every lane at once.

This file generates one table for each function of FUNCS and one shared evaluator. A new
function needs one entry of FUNCS and no new code, thus the same machine serves any activation.

The shape of each function is the same: a bump or a step that this file writes as
`f(u)` for u >= 0, with the sign and the linear part kept in the float domain by the caller.

    sigmoid(x) = x < 0 ? g(|x|) : 1 - g(|x|)   with g(u) = 1 / (1 + e^u)
    silu(x)    = relu(x) - h(|x|)              with h(u) = u / (1 + e^u)
    swiglu     = silu(x0) * x1

Each form gives a small result directly and never as the difference of two near-equal numbers,
thus the tails keep their value. The error of all three is 1.1e-4 at worst, and the f16 rounding
of the input is the whole of it: the polynomial and the step back to f32 are both smaller.

One polynomial piece covers one octave [2^e, 2^(e+1)) of u. The piece index is the exponent
field of the f16 value of u and the local coordinate is its 10-bit mantissa, thus the evaluator
has no float to integer conversion and no permute. All coefficients of one function share the
scale 2^16, thus Horner needs no shift between its steps.

Usage:
    tools/htp-lab/gen/act_i16.py > hvx-act-i16.h
    tools/htp-lab/gen/act_i16.py --check      # the error of each function and of each activation
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

import numpy as np

DEGREE = 4
E_LO = -12              # the first octave is [2^-12, 2^-11)
E_HI = 4                # the input is limited to values below 2^4
SCALE = 16              # a value as an integer is value * 2^SCALE
PIECES = E_HI - E_LO    # 16, one page of vlut16


def f_h(u: np.ndarray) -> np.ndarray:
    """The SiLU bump u / (1 + e^u) for u >= 0. It is 0 at 0 and 0 at infinity, peak 0.2785."""
    return u / (1.0 + np.exp(u))


def f_g(u: np.ndarray) -> np.ndarray:
    """The sigmoid of a negative argument, 1 / (1 + e^u) for u >= 0. It is 0.5 at 0 and 0 at
    infinity, thus a large |x| gives a small value directly and no subtraction cancels."""
    return 1.0 / (1.0 + np.exp(u))


# name -> (the function, one line that says what it is, the input needs a clamp from below)
#
# A function that is 0 at u = 0 needs no clamp: a u below the first octave misses the lookup and
# the evaluator gives 0, which is the limit. A function that is not 0 at u = 0 needs the clamp,
# thus every lane reads the first octave instead of the miss.
FUNCS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str, bool]] = {
    "h": (f_h, "u / (1 + e^u), the bump of the SiLU", False),
    "g": (f_g, "1 / (1 + e^u), the sigmoid of -u", True),
}


def table(fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Fit one polynomial for each octave at Chebyshev nodes. O(PIECES).

    Args:
        fn: The function of u, evaluated on an array

    Returns:
        The integer coefficients [PIECES][DEGREE + 1], the constant term first. A coefficient of
        32768, which g reaches at u = 0 because its value is exactly 0.5, becomes 32767. That
        costs 7.6e-6 of the value and the saturating add of the HVX would do the same.
    """
    nodes = (np.cos(np.pi * (np.arange(96) + 0.5) / 96) + 1) / 2
    rows = [np.polynomial.polynomial.polyfit(nodes, fn(2.0 ** e * (1 + nodes)), DEGREE)
            for e in range(E_LO, E_HI)]
    c = np.round(np.array(rows) * 2.0 ** SCALE)
    return np.clip(c, -32767, 32767).astype(np.int64)


def evaluate(u16: np.ndarray, q: np.ndarray, low_clamp: bool = False) -> np.ndarray:
    """The integer path of the HVX evaluator, for an array of f16 values of u. O(len(u16)).

    The saturating add of the HVX bounds every step to an int16, thus this model saturates too.

    Args:
        u16: The non-negative inputs as float16
        q: The coefficient table of table()
        low_clamp: True to raise a u below the first octave to the first octave

    Returns:
        The value times 2^SCALE, as int64
    """
    bits = u16.view(np.uint16).astype(np.int64)
    bits = np.minimum(bits, 0x4BFF)                      # the largest f16 below 16
    if low_clamp:
        bits = np.maximum(bits, (E_LO + 15) << 10)       # the smallest u of the first octave
    idx = (bits >> 10) - (E_LO + 15)
    v = (bits & 0x3FF) << 5                              # the mantissa as Q15
    ok = idx >= 0
    idx = np.clip(idx, 0, PIECES - 1)
    acc = q[idx, DEGREE]
    for d in range(DEGREE - 1, -1, -1):
        acc = np.clip(((acc * v * 2 + 0x8000) >> 16) + q[idx, d], -32768, 32767)
    return np.where(ok, acc, 0)


def check() -> int:
    """Print the error of each function and of each activation that uses it.

    Returns:
        The exit status
    """
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(-18, 18, 400000), rng.normal(0, 1.5, 400000),
                        rng.normal(0, 0.05, 100000)]).astype(np.float32)
    x64 = x.astype(np.float64)
    u16 = np.abs(x).astype(np.float16)
    q_h = table(f_h)
    q_g = table(f_g)

    print(f"{'function':<10} {'peak |coeff|':>12} {'max abs err':>12} {'rms abs err':>12}")
    for name, (fn, _, clamp) in FUNCS.items():
        q = table(fn)
        got = evaluate(u16, q, clamp) / 2.0 ** SCALE
        err = got - fn(np.abs(x64))
        print(f"{name:<10} {int(np.max(np.abs(q))):12d} {np.max(np.abs(err)):12.3e}"
              f" {np.sqrt(np.mean(err ** 2)):12.3e}")

    # The activations. The int16 value goes to f32 through a sign extend and a word to float
    # convert, which is exact, thus the only losses are the f16 rounding of |x| and the
    # polynomial. The f16 rounding is the larger of the two.
    h = evaluate(u16, q_h) / 2.0 ** SCALE
    g = evaluate(u16, q_g, True) / 2.0 ** SCALE
    x1 = rng.uniform(-2, 2, x.size).astype(np.float32).astype(np.float64)

    cases = {
        "sigmoid": (np.where(x64 < 0, g, 1.0 - g), 1.0 / (1.0 + np.exp(-x64))),
        "silu": (np.maximum(x64, 0) - h, x64 / (1.0 + np.exp(-x64))),
        "swiglu": ((np.maximum(x64, 0) - h) * x1, x1 * x64 / (1.0 + np.exp(-x64))),
    }
    print()
    print(f"{'activation':<10} {'nmse':>11} {'max abs err':>12} {'rms abs err':>12}")
    for name, (got, ref) in cases.items():
        err = got - ref
        print(f"{name:<10} {np.sum(err ** 2) / np.sum(ref ** 2):11.3e}"
              f" {np.max(np.abs(err)):12.3e} {np.sqrt(np.mean(err ** 2)):12.3e}")
    return 0


HEADER_TOP = '''// The activation family in int16 for the HVX. tools/htp-lab/gen/act_i16.py of the qwen-mobile
// repository generates this file, do not edit it.
//
// A chain of qfloat operations costs about 4 cycles for each operation, because a v79 packet
// holds one qfloat multiply and a result is ready two packets after its producer. A piecewise
// polynomial in int16 costs about 1 cycle for each operation: a packet holds two int16
// multiplies or four integer adds, and one vlut16 reads the coefficients of 128 lanes.
//
// Each function is written for u >= 0. The caller keeps the sign and the linear part in the
// float domain, where they are exact:
//   sigmoid(x) = x < 0 ? g(|x|) : 1 - g(|x|)   with g(u) = 1 / (1 + e^u)
//   silu(x)    = relu(x) - h(|x|)              with h(u) = u / (1 + e^u)
//   swiglu     = silu(x0) * x1
//
// Each form gives a small result directly and never as the difference of two near-equal
// numbers. The measured error of all three against a double reference is 1.1e-4 at worst and
// the f16 rounding of the input is the whole of it.
//
// One polynomial piece of degree {degree} covers one octave [2^e, 2^(e+1)) of u, for e = {e_lo} to {e_hi_m1}.
// The piece index is the exponent field of the f16 value of u and the local coordinate is its
// 10-bit mantissa, thus the evaluator has no float to integer conversion and no permute. All
// coefficients share the scale 2^{scale}, thus Horner has no shift between its steps.
//
// vlut16 in the 128-byte mode, from the full map that tools/htp-lab measures (target f16math):
//   - An index byte matches when its high nibble is equal to Rt & 15. The entry is the table word
//     (byte % 32), and the bit 1 of Rt selects the odd halfword of that word. A byte that does
//     not match gives 0, thus a u below the first octave gives 0, which is the limit of each
//     function at 0.
//   - The output pair is divided by the byte position: lo holds the results of the even bytes
//     and hi the results of the odd bytes. Thus one 16-bit lane carries the index of one vector
//     in its low byte and of a second vector in its high byte, and one lookup serves two
//     vectors. The coefficient of the octave k is the halfword 2 * k.
//
// The int16 result goes to f32 through a sign extend and a word to float convert, which is
// exact. An f16 step would be 4 times cheaper and would cost 2.4e-4 of absolute error, which is
// more than the polynomial itself, thus this file does not take it.

#ifndef HVX_ACT_I16_H
#define HVX_ACT_I16_H

#include <stdint.h>

#include "hvx-base.h"

#define HVX_ACT_I16_DEGREE {degree}
#define HVX_ACT_I16_SCALE  {scale}
#define HVX_ACT_I16_EBASE  {e_base}

'''

HEADER_BODY = '''
// The evaluator. `a` holds n f16 vectors of the input, `out` takes n int16 vectors of the value
// times 2^16, in the lane order of `a`. n is 1, 2 or 4 and a constant at the call site, thus the
// loops unroll and every vector stays in a register. The routine does each step for all n
// vectors before the next step, thus no packet waits for its own producer.
//
// It must be inline: the table pointer and n are then constants.
static inline __attribute__((always_inline))
void hvx_act_i16_eval(const HVX_Vector * a, HVX_Vector * out,
                      const int16_t tab[HVX_ACT_I16_DEGREE + 1][64], const int n,
                      const int low_clamp) {
    const HVX_Vector m_abs  = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector u_max  = Q6_Vh_vsplat_R(0x4bff);   // the largest f16 below 16
    const HVX_Vector u_min  = Q6_Vh_vsplat_R(HVX_ACT_I16_EBASE << 10);  // 2^E_LO
    const HVX_Vector e_base = Q6_Vh_vsplat_R(HVX_ACT_I16_EBASE);
    const HVX_Vector m_mant = Q6_Vh_vsplat_R(0x03ff);
    const HVX_Vector m_byte = Q6_Vh_vsplat_R(0x00ff);

    HVX_Vector u[4], idx[4], v[4], acc[4], ib[2];
    const int np = (n + 1) / 2;

    for (int r = 0; r < n; r++) {
        u[r] = Q6_Vh_vmin_VhVh(Q6_V_vand_VV(a[r], m_abs), u_max);
    }
    // A function that is not 0 at u = 0 needs every lane inside the table, thus the clamp. A
    // function that is 0 at u = 0 skips it and lets the miss give 0.
    if (low_clamp) {
        for (int r = 0; r < n; r++) {
            u[r] = Q6_Vh_vmax_VhVh(u[r], u_min);
        }
    }
    // k is the octave number, 0 to 15. Below the first octave k is negative, its index byte is
    // 0xfd to 0xff, that byte does not match the page 0, and every coefficient comes back 0.
    for (int r = 0; r < n; r++) {
        idx[r] = Q6_Vh_vsub_VhVh(Q6_Vuh_vlsr_VuhR(u[r], 10), e_base);
    }
    for (int r = 0; r < n; r++) {
        v[r] = Q6_Vh_vasl_VhR(Q6_V_vand_VV(u[r], m_mant), 5);   // the mantissa as Q15
    }
    // The low byte of a lane is an even byte and the high byte is an odd byte, thus the first
    // vector of a pair comes back in lo and the second in hi.
    for (int p = 0; p < np; p++) {
        const HVX_Vector second = idx[(2 * p + 1 < n) ? 2 * p + 1 : 2 * p];
        ib[p] = Q6_V_vor_VV(Q6_V_vand_VV(idx[2 * p], m_byte), Q6_Vh_vasl_VhR(second, 8));
    }
    for (int p = 0; p < np; p++) {
        const HVX_VectorPair c = Q6_Wh_vlut16_VbVhR(ib[p], hvx_vmem(tab[HVX_ACT_I16_DEGREE]), 0);
        acc[2 * p] = Q6_V_lo_W(c);
        if (2 * p + 1 < n) {
            acc[2 * p + 1] = Q6_V_hi_W(c);
        }
    }
    for (int d = HVX_ACT_I16_DEGREE - 1; d >= 0; d--) {
        HVX_Vector c[4];
        for (int p = 0; p < np; p++) {
            const HVX_VectorPair cp = Q6_Wh_vlut16_VbVhR(ib[p], hvx_vmem(tab[d]), 0);
            c[2 * p] = Q6_V_lo_W(cp);
            if (2 * p + 1 < n) {
                c[2 * p + 1] = Q6_V_hi_W(cp);
            }
        }
        for (int r = 0; r < n; r++) {
            acc[r] = Q6_Vh_vmpy_VhVh_s1_rnd_sat(acc[r], v[r]);
        }
        for (int r = 0; r < n; r++) {
            acc[r] = Q6_Vh_vadd_VhVh_sat(acc[r], c[r]);
        }
    }
    for (int r = 0; r < n; r++) {
        out[r] = acc[r];
    }
}

// One int16 vector of a value times 2^16 becomes two qf32 vectors of the value times k * 2^16.
// The sign extend and the word to float convert are exact, thus this step adds no error. lo
// holds the even lanes of v16 and hi the odd lanes, which is the order that a pair of f16
// vectors came from.
static inline __attribute__((always_inline))
HVX_VectorPair hvx_act_i16_to_qf32(HVX_Vector v16, HVX_Vector k_sf) {
    const HVX_VectorPair w = Q6_Ww_vsxt_Vh(v16);
    const HVX_Vector lo = Q6_Vsf_equals_Vw(Q6_V_lo_W(w));
    const HVX_Vector hi = Q6_Vsf_equals_Vw(Q6_V_hi_W(w));
    return Q6_W_vcombine_VV(Q6_Vqf32_vmpy_VsfVsf(hi, k_sf), Q6_Vqf32_vmpy_VsfVsf(lo, k_sf));
}

// Two f32 vectors become one f16 vector in the interleaved lane order of the conversion, which
// is the order that hvx_act_i16_to_qf32 gives back.
static inline __attribute__((always_inline))
HVX_Vector hvx_act_f32_pair_to_f16(HVX_Vector a, HVX_Vector b) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector qa = Q6_Vqf32_vadd_VsfVsf(a, zero);
    const HVX_Vector qb = Q6_Vqf32_vadd_VsfVsf(b, zero);
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(qb, qa));
}

// The three row kernels. Each reads its inputs once and writes the output once, thus a row costs
// one pass and not the two that the sigmoid pass plus the multiply pass of the checkout cost.
//
// A block is `ns` f16 streams, which is 2 * ns f32 vectors. The number of streams decides the
// speed: the chain from the input to the output is about 18 dependent steps and a result is ready
// two packets after its producer, thus one stream leaves three of every four slots empty and the
// loop waits on itself. Four streams fill the packets. The block must not hold the input vectors
// across the evaluator: eight of them plus the working set do not fit the 32 vector registers,
// the compiler spills them to the stack, and the spill costs both a store and the reload. The
// blocks read the input again from the scratch pad instead, which costs the same load and no
// store, thus `vsrc` is indexed a second time and never kept.
#define HVX_ACT_I16_STREAMS 4
#define HVX_ACT_I16_UNROLL  (2 * HVX_ACT_I16_STREAMS)

// silu(x) = relu(x) - h(|x|). The scale of the conversion is negative, thus the combine is one
// add and the subtract costs nothing.
static inline __attribute__((always_inline))
void hvx_silu_i16_block(const HVX_Vector * restrict vsrc, HVX_Vector * restrict vdst, uint32_t i,
                        HVX_Vector k_sf, HVX_Vector zero, const int ns) {
    HVX_Vector a[HVX_ACT_I16_STREAMS], h16[HVX_ACT_I16_STREAMS];

    for (int s = 0; s < ns; s++) {
        a[s] = hvx_act_f32_pair_to_f16(vsrc[i + 2 * s], vsrc[i + 2 * s + 1]);
    }
    hvx_act_i16_eval(a, h16, hvx_act_i16_tab_h, ns, 0);
    for (int s = 0; s < ns; s++) {
        const HVX_VectorPair nh = hvx_act_i16_to_qf32(h16[s], k_sf);
        // a negative f32 is a negative int32, thus the signed max is the relu
        const HVX_Vector r0 = Q6_Vw_vmax_VwVw(vsrc[i + 2 * s], zero);
        const HVX_Vector r1 = Q6_Vw_vmax_VwVw(vsrc[i + 2 * s + 1], zero);
        vdst[i + 2 * s]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(nh), r0));
        vdst[i + 2 * s + 1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(nh), r1));
    }
}

static inline void hvx_silu_i16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src,
                                       uint32_t n) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const HVX_Vector k_sf = hvx_vec_splat_f32(-1.0f / 65536.0f);
    const HVX_Vector zero = Q6_V_vzero();

    const uint32_t nvec = n / 32;
    uint32_t i = 0;

    for (; i + HVX_ACT_I16_UNROLL <= nvec; i += HVX_ACT_I16_UNROLL) {
        hvx_silu_i16_block(vsrc, vdst, i, k_sf, zero, HVX_ACT_I16_STREAMS);
    }
    for (; i + 2 <= nvec; i += 2) {
        hvx_silu_i16_block(vsrc, vdst, i, k_sf, zero, 1);
    }
    if (i < nvec) {
        // one vector left: the second half of the stream repeats the first and its result is dropped
        HVX_Vector a = hvx_act_f32_pair_to_f16(vsrc[i], vsrc[i]);
        HVX_Vector h16;
        hvx_act_i16_eval(&a, &h16, hvx_act_i16_tab_h, 1, 0);
        const HVX_VectorPair nh = hvx_act_i16_to_qf32(h16, k_sf);
        vdst[i] = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(nh), Q6_Vw_vmax_VwVw(vsrc[i], zero)));
    }
}

// swiglu(x0, x1) = silu(x0) * x1, in one pass over the two rows.
static inline __attribute__((always_inline))
void hvx_swiglu_i16_block(const HVX_Vector * restrict vsrc0, const HVX_Vector * restrict vsrc1,
                          HVX_Vector * restrict vdst, uint32_t i,
                          HVX_Vector k_sf, HVX_Vector zero, const int ns) {
    HVX_Vector a[HVX_ACT_I16_STREAMS], h16[HVX_ACT_I16_STREAMS];

    for (int s = 0; s < ns; s++) {
        a[s] = hvx_act_f32_pair_to_f16(vsrc0[i + 2 * s], vsrc0[i + 2 * s + 1]);
    }
    hvx_act_i16_eval(a, h16, hvx_act_i16_tab_h, ns, 0);
    for (int s = 0; s < ns; s++) {
        const HVX_VectorPair nh = hvx_act_i16_to_qf32(h16[s], k_sf);
        const uint32_t j0 = i + 2 * s;
        const uint32_t j1 = j0 + 1;
        const HVX_Vector s0 = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(nh), Q6_Vw_vmax_VwVw(vsrc0[j0], zero)));
        const HVX_Vector s1 = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_hi_W(nh), Q6_Vw_vmax_VwVw(vsrc0[j1], zero)));
        vdst[j0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s0, vsrc1[j0]));
        vdst[j1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s1, vsrc1[j1]));
    }
}

static inline void hvx_swiglu_i16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src0,
                                         const uint8_t * restrict src1, uint32_t n) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc0 = (const HVX_Vector *) src0;
    const HVX_Vector * restrict vsrc1 = (const HVX_Vector *) src1;

    const HVX_Vector k_sf = hvx_vec_splat_f32(-1.0f / 65536.0f);
    const HVX_Vector zero = Q6_V_vzero();

    const uint32_t nvec = n / 32;
    uint32_t i = 0;

    for (; i + HVX_ACT_I16_UNROLL <= nvec; i += HVX_ACT_I16_UNROLL) {
        hvx_swiglu_i16_block(vsrc0, vsrc1, vdst, i, k_sf, zero, HVX_ACT_I16_STREAMS);
    }
    for (; i + 2 <= nvec; i += 2) {
        hvx_swiglu_i16_block(vsrc0, vsrc1, vdst, i, k_sf, zero, 1);
    }
    if (i < nvec) {
        HVX_Vector a = hvx_act_f32_pair_to_f16(vsrc0[i], vsrc0[i]);
        HVX_Vector h16;
        hvx_act_i16_eval(&a, &h16, hvx_act_i16_tab_h, 1, 0);
        const HVX_VectorPair nh = hvx_act_i16_to_qf32(h16, k_sf);
        const HVX_Vector sv = Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vadd_Vqf32Vsf(Q6_V_lo_W(nh), Q6_Vw_vmax_VwVw(vsrc0[i], zero)));
        vdst[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(sv, vsrc1[i]));
    }
}

// sigmoid(x) = x < 0 ? g(|x|) : 1 - g(|x|). Both branches give their result directly, thus a
// large |x| keeps its value instead of coming out of the difference of two numbers near 0.5.
static inline __attribute__((always_inline))
HVX_Vector hvx_sigmoid_i16_combine(HVX_Vector g_qf32, HVX_Vector x, HVX_Vector one) {
    const HVX_Vector     g_sf = Q6_Vsf_equals_Vqf32(g_qf32);
    const HVX_Vector     comp = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(one, g_sf));
    const HVX_VectorPred neg  = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(), x);
    return Q6_V_vmux_QVV(neg, g_sf, comp);
}

static inline __attribute__((always_inline))
void hvx_sigmoid_i16_block(const HVX_Vector * restrict vsrc, HVX_Vector * restrict vdst, uint32_t i,
                           HVX_Vector k_sf, HVX_Vector one, const int ns) {
    HVX_Vector a[HVX_ACT_I16_STREAMS], g16[HVX_ACT_I16_STREAMS];

    for (int s = 0; s < ns; s++) {
        a[s] = hvx_act_f32_pair_to_f16(vsrc[i + 2 * s], vsrc[i + 2 * s + 1]);
    }
    hvx_act_i16_eval(a, g16, hvx_act_i16_tab_g, ns, 1);
    for (int s = 0; s < ns; s++) {
        const HVX_VectorPair g32 = hvx_act_i16_to_qf32(g16[s], k_sf);
        vdst[i + 2 * s]     = hvx_sigmoid_i16_combine(Q6_V_lo_W(g32), vsrc[i + 2 * s], one);
        vdst[i + 2 * s + 1] = hvx_sigmoid_i16_combine(Q6_V_hi_W(g32), vsrc[i + 2 * s + 1], one);
    }
}

static inline void hvx_sigmoid_i16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src,
                                          uint32_t n) {
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;
    const HVX_Vector * restrict vsrc = (const HVX_Vector *) src;

    const HVX_Vector k_sf = hvx_vec_splat_f32(1.0f / 65536.0f);
    const HVX_Vector one  = hvx_vec_splat_f32(1.0f);

    const uint32_t nvec = n / 32;
    uint32_t i = 0;

    for (; i + HVX_ACT_I16_UNROLL <= nvec; i += HVX_ACT_I16_UNROLL) {
        hvx_sigmoid_i16_block(vsrc, vdst, i, k_sf, one, HVX_ACT_I16_STREAMS);
    }
    for (; i + 2 <= nvec; i += 2) {
        hvx_sigmoid_i16_block(vsrc, vdst, i, k_sf, one, 1);
    }
    if (i < nvec) {
        HVX_Vector a = hvx_act_f32_pair_to_f16(vsrc[i], vsrc[i]);
        HVX_Vector g16;
        hvx_act_i16_eval(&a, &g16, hvx_act_i16_tab_g, 1, 1);
        const HVX_VectorPair g32 = hvx_act_i16_to_qf32(g16, k_sf);
        vdst[i] = hvx_sigmoid_i16_combine(Q6_V_lo_W(g32), vsrc[i], one);
    }
}

#endif /* HVX_ACT_I16_H */
'''


def emit_table(name: str, doc: str, q: np.ndarray) -> str:
    """Write one coefficient table as C.

    Args:
        name: The name of the function, the table is hvx_act_i16_tab_<name>
        doc: One line that says what the function is
        q: The integer coefficients of table()

    Returns:
        The C text of the table
    """
    rows = []
    for d in range(DEGREE + 1):
        cells = ["0"] * 64
        for k in range(PIECES):
            cells[2 * k] = str(int(q[k, d]))
        rows.append(f"    // the coefficient of v^{d}\n    {{ " + ", ".join(cells) + " },")
    return (f"// {name}(u) = {doc}\n"
            f"static const int16_t hvx_act_i16_tab_{name}[HVX_ACT_I16_DEGREE + 1][64]"
            f" __attribute__((aligned(128))) = {{\n" + "\n".join(rows) + "\n};\n")


def main(argv: list[str] | None = None) -> int:
    """Write the header to stdout, or print the error with --check.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the error of the integer path")
    a = ap.parse_args(argv)
    if a.check:
        return check()

    out = [HEADER_TOP.format(degree=DEGREE, scale=SCALE, e_lo=E_LO, e_hi_m1=E_HI - 1,
                             e_base=E_LO + 15)]
    for name, (fn, doc, _) in FUNCS.items():
        out.append(emit_table(name, doc, table(fn)))
        out.append("\n")
    out.append(HEADER_BODY)
    sys.stdout.write("".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
