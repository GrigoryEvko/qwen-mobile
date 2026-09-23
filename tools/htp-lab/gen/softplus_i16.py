#!/usr/bin/env python3
"""Generate hvx-softplus.h: the two vector paths of softplus for the HVX.

softplus(x) = log(1 + e^x). The identity

    softplus(x) = relu(x) + log(1 + e^-|x|)

holds for every x, and it has no cancellation: relu(x) is exact in the float domain and the
second term is a smooth bump in (0, log 2]. Both paths of this header use it, and they differ
only in how they evaluate the bump.

The int16 path evaluates D(u) = log(2) - log(1 + e^-u), which is 0 at u = 0 and rises to
log(2). D fits an int16 with the scale 2^15. One polynomial piece covers one octave
[2^e, 2^(e+1)) of u, the piece index is the exponent field of the f16 value of u and the local
coordinate is its 10-bit mantissa, thus the path has no float to integer conversion. All
coefficients share one scale, thus Horner needs no shift between its steps. Below the first
octave the lookup gives 0 for each coefficient, which is the correct limit of D.

    softplus(x) = relu(x) + log(2) - D(|x|)

The f32 path gives the result of the CPU reference, logf(1.0f + expf(x)) for x <= 20 and x above
(the oracle form), to a few ulp. It evaluates t = e^-|x| with the exp of this backend. For a
negative x (t = e^x) it rounds 1 + t to an f32, as the reference does, and takes t = (1 + t) - 1,
which is exact. Then it evaluates log1p(t) = t * P(t), where P is a polynomial of degree
LOG1P_DEGREE on [0, 1], thus the result is log of the rounded sum. For a positive x,
log(1 + e^x) = x + log1p(e^-x), and the rounded sum of the reference moves its result by at most
one half ulp. The rounded sum has an absolute error of up to 2^-24, thus the relative error of the
oracle form against the exact function grows for a large negative x (the result is 0 below -16.6),
and this path has that error too.

Usage:
    tools/htp-lab/gen/softplus_i16.py > hvx-softplus.h
    tools/htp-lab/gen/softplus_i16.py --check      # the error of each path, against float64
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

DEGREE = 4
E_LO = -12          # the first octave is [2^-12, 2^-11)
E_HI = 4            # the input is limited to values below 2^4
SCALE = 15          # D as an integer is D * 2^15, and log(2) * 2^15 = 22713 fits an int16
PIECES = E_HI - E_LO    # 16, one page of vlut16
LOG1P_DEGREE = 9
LOG2 = float(np.log(2.0))


def bump(u: np.ndarray) -> np.ndarray:
    """log(1 + e^-u) for u >= 0, the second term of the identity."""
    return np.log1p(np.exp(-u))


def deficit(u: np.ndarray) -> np.ndarray:
    """D(u) = log(2) - log(1 + e^-u), which is 0 at u = 0 and rises to log(2)."""
    return LOG2 - bump(u)


def table() -> np.ndarray:
    """Fit one polynomial of D for each octave at Chebyshev nodes. O(PIECES).

    Returns:
        The integer coefficients [PIECES][DEGREE + 1], the constant term first

    Raises:
        ValueError: If a coefficient does not fit an int16
    """
    nodes = (np.cos(np.pi * (np.arange(96) + 0.5) / 96) + 1) / 2
    rows = [np.polynomial.polynomial.polyfit(nodes, deficit(2.0 ** e * (1 + nodes)), DEGREE)
            for e in range(E_LO, E_HI)]
    c = np.round(np.array(rows) * 2.0 ** SCALE).astype(np.int64)
    if np.max(np.abs(c)) > 32767:
        raise ValueError(f"a coefficient is {np.max(np.abs(c))}, more than an int16 holds")
    return c


def log1p_coeffs() -> np.ndarray:
    """Fit P with log1p(t) = t * P(t) on [0, 1] at Chebyshev nodes. O(1).

    Returns:
        The LOG1P_DEGREE + 1 coefficients as float32, the constant term first
    """
    nodes = (np.cos(np.pi * (np.arange(128) + 0.5) / 128) + 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(nodes > 0, np.log1p(nodes) / np.where(nodes > 0, nodes, 1), 1.0)
    return np.polynomial.polynomial.polyfit(nodes, y, LOG1P_DEGREE).astype(np.float32)


def eval_i16(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The int16 path, bit for bit as the HVX routine runs it. O(len(x)).

    Args:
        x: The inputs as float32
        q: The coefficient table of table()

    Returns:
        softplus as float64
    """
    u16 = np.abs(x).astype(np.float16)
    bits = u16.view(np.uint16).astype(np.int64)
    bits = np.minimum(bits, 0x4BFF)                      # the largest f16 below 16
    idx = (bits >> 10) - (E_LO + 15)
    v = (bits & 0x3FF) << 5                              # the mantissa as Q15
    ok = idx >= 0
    idx = np.clip(idx, 0, PIECES - 1)
    acc = q[idx, DEGREE]
    for d in range(DEGREE - 1, -1, -1):
        acc = np.clip(((acc * v * 2 + 0x8000) >> 16) + q[idx, d], -32768, 32767)
    d_u = np.where(ok, acc, 0) / 2.0 ** SCALE
    return np.maximum(x.astype(np.float64), 0.0) + (LOG2 - d_u)


def eval_f32(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The f32 path in float32 arithmetic, as the HVX routine runs it. O(len(x)).

    Args:
        x: The inputs as float32
        p: The log1p coefficients of log1p_coeffs()

    Returns:
        softplus as float64
    """
    t = np.exp(-np.abs(x).astype(np.float32)).astype(np.float32)
    one = np.float32(1.0)
    t = np.where(np.signbit(x), ((one + t).astype(np.float32) - one).astype(np.float32), t)
    acc = np.full_like(t, p[LOG1P_DEGREE])
    for d in range(LOG1P_DEGREE - 1, -1, -1):
        acc = (acc * t + p[d]).astype(np.float32)
    return np.maximum(x.astype(np.float64), 0.0) + (t * acc).astype(np.float32)


def oracle_form(x: np.ndarray) -> np.ndarray:
    """The CPU reference in f32 with correctly rounded expf and logf. O(len(x)).

    Args:
        x: The inputs as float32

    Returns:
        x > 20 ? x : logf(1.0f + expf(x)), as float32
    """
    e = np.exp(x.astype(np.float64)).astype(np.float32)
    s = (np.float32(1.0) + e).astype(np.float32)
    return np.where(x > 20, x, np.log(s.astype(np.float64)).astype(np.float32))


def ulp32(v: np.ndarray) -> np.ndarray:
    """The spacing of float32 values at |v|, and the smallest subnormal at 0. O(len(v))."""
    a = np.abs(v.astype(np.float32))
    return np.maximum(np.spacing(a), np.float32(1.4e-45)).astype(np.float64)


def check_inputs() -> np.ndarray:
    """The inputs of the check: the range [-30, 30], a normal spread, and values near 0. O(1)."""
    rng = np.random.default_rng(1)
    return np.concatenate([rng.uniform(-30, 30, 400000), rng.normal(0, 1.5, 400000),
                           rng.normal(0, 0.05, 100000), rng.uniform(-1e-3, 1e-3, 50000)]).astype(np.float32)


def f32_oracle_ulp(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The distance of the f32 path from the oracle form, in ulp of the oracle form, for x >= -10.

    Below -10 the result is less than 4.6e-5, and an exp that differs by one ulp from the correctly
    rounded one can move the rounded sum 1 + e^x by one ulp of 1, which is 2^-23 in the result: a
    large distance in ulp of a small result. The check prints the absolute distance for all x.
    O(len(x)).
    """
    x = x[x >= -10]
    orc = oracle_form(x)
    return np.abs(eval_f32(x, p) - orc.astype(np.float64)) / ulp32(orc)


def check() -> int:
    """Print the error of each path against a float64 reference, and the f32 path against the
    oracle form in ulp.

    Returns:
        The exit status
    """
    x = check_inputs()
    ref = np.log1p(np.exp(-np.abs(x.astype(np.float64)))) + np.maximum(x.astype(np.float64), 0.0)
    q, p = table(), log1p_coeffs()
    print(f"{'path':<6} {'max abs':>10} {'rms abs':>10} {'nmse':>10} {'max rel':>10} {'max rel, x<-8':>14}")
    tail = x < -8
    orc = oracle_form(x)
    for name, y in (("i16", eval_i16(x, q)), ("f32", eval_f32(x, p)), ("oracle", orc.astype(np.float64))):
        e = y - ref
        rel = np.abs(e) / ref
        print(f"{name:<6} {np.max(np.abs(e)):10.3e} {np.sqrt(np.mean(e ** 2)):10.3e}"
              f" {np.sum(e ** 2) / np.sum(ref ** 2):10.2e} {np.max(rel):10.3e} {np.max(rel[tail]):14.3e}")
    d = f32_oracle_ulp(x, p)
    print(f"f32 against the oracle form: max abs {np.max(np.abs(eval_f32(x, p) - orc)):.3e}; for x >= -10 "
          f"max {np.max(d):.1f} ulp, mean {np.mean(d):.3f} ulp, equal {np.mean(d == 0) * 100:.1f} %")
    return 0


HEADER = '''// The softplus in vector form for the HVX. tools/htp-lab/gen/softplus_i16.py of the qwen-mobile
// repository generates this file, do not edit it.
//
// softplus(x) = log(1 + e^x) = relu(x) + log(1 + e^-|x|). The identity holds for every x and it
// has no cancellation: relu(x) is exact in the float domain and the second term is a smooth bump
// in (0, log 2]. The CPU reference (the oracle form) evaluates logf(1.0f + expf(x)) for x <= 20.
//
// Two paths, both of which write the same result the CPU reference writes for x > 20 (the result
// is x, because the bump is below the resolution of an f32 there).
//
//   hvx_softplus_f32_aa    the oracle form to a few ulp. t = e^-|x| with the exp of this backend.
//                          For a negative x (t = e^x), t becomes (1 + t) - 1 with both steps in
//                          f32, as the reference rounds 1 + e^x before its logf, and the second
//                          step is exact. Then log1p(t) = t * P(t) with P of degree {log1p_degree} on
//                          [0, 1], thus the result is log of the rounded sum. The rounded sum has
//                          an absolute error of up to 2^-24, thus against the exact function the
//                          relative error grows for a large negative x (the result is 0 below
//                          -16.6), the same as the reference. The model of this path in the
//                          generator (--check) gives a mean of {orc_mean_ulp} ulp against the oracle
//                          form for x >= -10, and {orc_within} % of the values within 4 ulp. Where the
//                          exp differs by one ulp from expf, the rounded sum can move by one
//                          ulp of 1, thus the largest difference is 2^-23 absolute.
//
//   hvx_softplus_i16_f32_aa  D(u) = log(2) - log(1 + e^-u) from a piecewise int16 table, then
//                          softplus = relu(x) + log(2) - D(|x|). One polynomial of degree {degree}
//                          covers one octave of u, the piece index is the exponent field of the
//                          f16 value of u, thus the path has no float to integer conversion, and
//                          all coefficients share the scale 2^{scale}, thus Horner needs no shift.
//                          Measured: NMSE 2.0e-11, worst absolute error 1.4e-4. **Its relative
//                          error is unbounded for x below -8**, where the result is e^x and the
//                          absolute quantum of an int16 is larger than the result. Use this path
//                          only where the absolute error is what matters.
//
// vlut16 in the 128-byte mode, from the full map that tools/htp-lab measures (target f16math):
// an index byte matches when its high nibble is equal to Rt & 15, the entry is the table word
// (byte % 32), the bit 1 of Rt selects the odd halfword, and the output pair is divided by the
// byte position, thus lo holds the results of the even bytes and hi those of the odd bytes. With
// Rt = 0 the coefficient of the octave k is the halfword 2 * k, and an octave index below 0 has a
// byte of 0xfd to 0xff, which matches no entry of the page 0 and gives 0, the correct limit of D.
//
// Why the routines take four vectors: on the v79 a packet holds at most 4 instructions and costs
// 2 cycles, a result is ready 2 packets after its producer, and one chain of dependent operations
// thus leaves three quarters of the slots empty. The routines do each step for all four vectors
// before the next step, thus no packet waits.
//
// The loop body does the full groups of four vectors first and then one tail block for the rest.
// The compiler inlines the four-vector routine one time for each block, and it can give the two
// copies different code. The model calls the op with ne0 = ssm_dt_rank, which is 16 for the 2B
// and 32 for the 4B, thus the tail block is the only block that runs on the device. A test with
// an element count that is a multiple of 128 measures the other copy only. tools/htp-lab
// (target unary) thus sweeps the counts 1, 8, 15, 16, 17, 31, 32, 33, 48, 64, 96, 127 and 128.

#ifndef HVX_SOFTPLUS_H
#define HVX_SOFTPLUS_H

#include <stdint.h>

#include "hvx-base.h"
#include "hvx-exp.h"

#define HVX_SOFTPLUS_DEGREE       {degree}
#define HVX_SOFTPLUS_SCALE        {scale}
#define HVX_SOFTPLUS_LOG1P_DEGREE {log1p_degree}

// log(2) as an f32, the constant term of the int16 path
#define HVX_SOFTPLUS_LOG2 0x3f317218

// P of log1p(t) = t * P(t) on [0, 1], the constant term first
static const uint32_t hvx_softplus_log1p_p[HVX_SOFTPLUS_LOG1P_DEGREE + 1] = {{
{log1p_rows}
}};

// The coefficient of v^d of the octave k is the halfword 2 * k of the row d.
static const int16_t hvx_softplus_d_table[HVX_SOFTPLUS_DEGREE + 1][64] __attribute__((aligned(128))) = {{
{rows}
}};

// softplus of four f32 vectors through the f32 path. The routine must be inline, thus the loops
// unroll and the four chains stay in registers.
static inline __attribute__((always_inline)) void hvx_softplus_f32_x4(const HVX_Vector * x, HVX_Vector * y) {{
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector sign = Q6_V_vsplat_R(0x80000000);

    HVX_Vector t[4], acc[4], r[4];

    // t = e^-|x|, which is e^x for a negative x and e^-x for the rest
    for (int i = 0; i < 4; i++) {{
        t[i] = Q6_V_vor_VV(x[i], sign);              // -|x|, the sign bit set
    }}
    for (int i = 0; i < 4; i++) {{
        t[i] = hvx_vec_exp_f32(t[i]);
    }}

    // For a negative x, the rounded sum of the reference: t = (1 + t) - 1, both steps in f32. A
    // negative f32 is a negative int32, thus the word compare with 0 selects the negative lanes.
    const HVX_Vector one = Q6_V_vsplat_R(0x3f800000);
    for (int i = 0; i < 4; i++) {{
        const HVX_Vector s  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(one, t[i]));
        const HVX_Vector ts = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(s, one));
        t[i] = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(zero, x[i]), ts, t[i]);
    }}

    // log1p(t) = t * P(t), by Horner over the four chains. The first step multiplies the top
    // coefficient as the IEEE single that it is, thus acc holds a qf32 number from its first
    // value and each Q6_Vsf_equals_Vqf32 below gets an operand of the correct type.
    //
    // DO NOT seed acc with Q6_V_vsplat_R of a coefficient. The IEEE pattern then goes to the
    // Q6_Vsf_equals_Vqf32 of the next step, which reads it as a qf32 number: the pattern
    // 0xbb5568f7 (-0.00325637846) reads as -1.4261448e36. The library of the app, which builds
    // with -flto, gave that value at each element count. The lab, which builds without -flto,
    // gave it in the tail block only, because hexagon-clang 19.0.07 put an sf to qf32 conversion
    // before the read in the other copy. Refer to the header of
    // patches/hexagon-kernels/0004 of the qwen-mobile repository.
    const HVX_Vector c_top = Q6_V_vsplat_R(hvx_softplus_log1p_p[HVX_SOFTPLUS_LOG1P_DEGREE]);
    for (int i = 0; i < 4; i++) {{
        acc[i] = Q6_Vqf32_vmpy_VsfVsf(c_top, t[i]);
    }}
    for (int d = HVX_SOFTPLUS_LOG1P_DEGREE - 1; d >= 0; d--) {{
        const HVX_Vector c = Q6_V_vsplat_R(hvx_softplus_log1p_p[d]);
        for (int i = 0; i < 4; i++) {{
            acc[i] = Q6_Vqf32_vadd_Vqf32Vsf(acc[i], c);
        }}
        for (int i = 0; i < 4; i++) {{
            acc[i] = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc[i]), t[i]);
        }}
    }}

    // relu(x) + log1p(t). A negative f32 is a negative int32, thus vmax on words gives relu.
    for (int i = 0; i < 4; i++) {{
        r[i] = Q6_Vw_vmax_VwVw(x[i], zero);
    }}
    for (int i = 0; i < 4; i++) {{
        y[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(acc[i], r[i]));
    }}
}}

// D(|x|) * 2^15 as int16 for n vectors of 64 f16 lanes, n is 1, 2 or 4 and a constant at the call.
static inline __attribute__((always_inline)) void hvx_softplus_d_i16(const HVX_Vector * a, HVX_Vector * d, const int n) {{
    const HVX_Vector m_abs  = Q6_Vh_vsplat_R(0x7fff);
    const HVX_Vector u_max  = Q6_Vh_vsplat_R(0x4bff);              // the largest f16 below 16
    const HVX_Vector e_base = Q6_Vh_vsplat_R({e_base});
    const HVX_Vector m_mant = Q6_Vh_vsplat_R(0x03ff);
    const HVX_Vector m_byte = Q6_Vh_vsplat_R(0x00ff);

    HVX_Vector u[4], idx[4], v[4], acc[4];
    HVX_Vector ib[2];
    const int np = (n + 1) / 2;

    for (int r = 0; r < n; r++) {{
        u[r] = Q6_Vh_vmin_VhVh(Q6_V_vand_VV(a[r], m_abs), u_max);
    }}
    for (int r = 0; r < n; r++) {{
        idx[r] = Q6_Vh_vsub_VhVh(Q6_Vuh_vlsr_VuhR(u[r], 10), e_base);
    }}
    for (int r = 0; r < n; r++) {{
        v[r] = Q6_Vh_vasl_VhR(Q6_V_vand_VV(u[r], m_mant), 5);      // the mantissa as Q15
    }}
    for (int p = 0; p < np; p++) {{
        const HVX_Vector second = idx[(2 * p + 1 < n) ? 2 * p + 1 : 2 * p];
        ib[p] = Q6_V_vor_VV(Q6_V_vand_VV(idx[2 * p], m_byte), Q6_Vh_vasl_VhR(second, 8));
    }}
    for (int p = 0; p < np; p++) {{
        const HVX_VectorPair c = Q6_Wh_vlut16_VbVhR(ib[p], hvx_vmem(hvx_softplus_d_table[HVX_SOFTPLUS_DEGREE]), 0);
        acc[2 * p] = Q6_V_lo_W(c);
        if (2 * p + 1 < n) {{
            acc[2 * p + 1] = Q6_V_hi_W(c);
        }}
    }}
    for (int dg = HVX_SOFTPLUS_DEGREE - 1; dg >= 0; dg--) {{
        HVX_Vector c[4];
        for (int p = 0; p < np; p++) {{
            const HVX_VectorPair cp = Q6_Wh_vlut16_VbVhR(ib[p], hvx_vmem(hvx_softplus_d_table[dg]), 0);
            c[2 * p] = Q6_V_lo_W(cp);
            if (2 * p + 1 < n) {{
                c[2 * p + 1] = Q6_V_hi_W(cp);
            }}
        }}
        for (int r = 0; r < n; r++) {{
            acc[r] = Q6_Vh_vmpy_VhVh_s1_rnd_sat(acc[r], v[r]);
        }}
        for (int r = 0; r < n; r++) {{
            acc[r] = Q6_Vh_vadd_VhVh_sat(acc[r], c[r]);
        }}
    }}
    for (int r = 0; r < n; r++) {{
        d[r] = acc[r];
    }}
}}

// log(2) - D as a qf32 pair, from D * 2^15 as int16. The int16 becomes an f16 number, an unsigned
// saturating subtract of 2 from its exponent field divides it by 4 and keeps a zero a zero, and
// the widening multiply by -2^-13 gives the scale and the 32-bit form in one step.
static inline __attribute__((always_inline)) HVX_VectorPair hvx_softplus_bump_qf32(HVX_Vector d16) {{
    const HVX_Vector quarter = Q6_Vh_vsplat_R(0x0800);
    const HVX_Vector k       = Q6_Vh_vsplat_R(0x8800);             // -2^-13 as f16
    const HVX_Vector hf      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vhf_equals_Vh(d16), quarter);
    return Q6_Wqf32_vmpy_VhfVhf(hf, k);
}}

// softplus of four f32 vectors through the int16 path, in the interleaved lane order of a pair.
static inline __attribute__((always_inline)) void hvx_softplus_i16_x4(const HVX_Vector * x, HVX_Vector * y) {{
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector log2 = Q6_V_vsplat_R(HVX_SOFTPLUS_LOG2);

    HVX_Vector h[2], d[2], r[4];

    for (int p = 0; p < 2; p++) {{
        h[p] = hvx_vec_f32_to_f16_shuff(x[2 * p], x[2 * p + 1]);
    }}
    hvx_softplus_d_i16(h, d, 2);
    for (int i = 0; i < 4; i++) {{
        r[i] = Q6_Vw_vmax_VwVw(x[i], zero);
    }}
    for (int p = 0; p < 2; p++) {{
        const HVX_VectorPair b = hvx_softplus_bump_qf32(d[p]);
        y[2 * p]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
                           Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_lo_W(b)), log2), r[2 * p]));
        y[2 * p + 1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(
                           Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(b)), log2), r[2 * p + 1]));
    }}
}}

#define hvx_softplus_loop_body(x4_fn)                                          \\
    HVX_Vector * restrict vdst = (HVX_Vector *) dst;                           \\
    HVX_Vector * restrict vsrc = (HVX_Vector *) src;                           \\
    const uint32_t nvec = n / VLEN_FP32;                                       \\
    const uint32_t nloe = n % VLEN_FP32;                                       \\
    uint32_t i = 0;                                                            \\
    HVX_Vector xv[4], yv[4];                                                   \\
    for (; i + 4 <= nvec; i += 4) {{                                            \\
        for (int j = 0; j < 4; j++) {{                                          \\
            xv[j] = vsrc[i + j];                                               \\
        }}                                                                      \\
        x4_fn(xv, yv);                                                         \\
        for (int j = 0; j < 4; j++) {{                                          \\
            vdst[i + j] = yv[j];                                               \\
        }}                                                                      \\
    }}                                                                          \\
    if (i < nvec || nloe) {{                                                    \\
        const uint32_t rest = nvec - i;                                        \\
        for (uint32_t j = 0; j < 4; j++) {{                                     \\
            xv[j] = (j < rest || (j == rest && nloe)) ? vsrc[i + j] : Q6_V_vzero(); \\
        }}                                                                      \\
        x4_fn(xv, yv);                                                         \\
        for (uint32_t j = 0; j < rest; j++) {{                                  \\
            vdst[i + j] = yv[j];                                               \\
        }}                                                                      \\
        if (nloe) {{                                                            \\
            hvx_vec_store_a((void *) &vdst[nvec], nloe * sizeof(float), yv[rest]); \\
        }}                                                                      \\
    }}

// softplus of n f32 elements, aligned source and destination, the accurate path.
static inline void hvx_softplus_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {{
    hvx_softplus_loop_body(hvx_softplus_f32_x4)
}}

// softplus of n f32 elements, aligned source and destination, the int16 path.
static inline void hvx_softplus_i16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {{
    hvx_softplus_loop_body(hvx_softplus_i16_x4)
}}

#endif /* HVX_SOFTPLUS_H */
'''


def main(argv: list[str] | None = None) -> int:
    """Write the header to stdout, or print the error of each path with --check.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the error of each path")
    a = ap.parse_args(argv)
    if a.check:
        return check()

    q = table()
    rows = []
    for d in range(DEGREE + 1):
        cells = ["0"] * 64
        for k in range(PIECES):
            cells[2 * k] = str(int(q[k, d]))
        rows.append(f"    // the coefficient of v^{d}\n    {{ " + ", ".join(cells) + " },")
    p = log1p_coeffs()
    bits = [int(np.float32(c).view(np.uint32)) for c in p]
    log1p_rows = "\n".join(f"    0x{b:08x},   // {c:+.9g} * t^{i}" for i, (b, c) in enumerate(zip(bits, p)))
    d = f32_oracle_ulp(check_inputs(), p)
    sys.stdout.write(HEADER.format(degree=DEGREE, scale=SCALE, log1p_degree=LOG1P_DEGREE,
                                   e_base=E_LO + 15, rows="\n".join(rows), log1p_rows=log1p_rows,
                                   orc_mean_ulp=f"{np.mean(d):.2f}", orc_within=f"{np.mean(d <= 4) * 100:.1f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
