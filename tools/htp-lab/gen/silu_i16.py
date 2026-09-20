#!/usr/bin/env python3
"""Generate hvx-silu-i16.h: the table and the HVX routine of a piecewise int16 SiLU.

SiLU(x) = relu(x) - h(|x|) with h(u) = u / (1 + e^u). h is a smooth bump in [0, 0.2785], thus
it fits an int16 with the scale 2^16, and relu(x) stays exact in the float domain.

One polynomial piece covers one octave [2^e, 2^(e+1)) of u. The piece index is the exponent
field of the f16 value of u and the local coordinate is its 10-bit mantissa, thus the routine
needs no float to integer conversion. All coefficients share the scale 2^16, thus Horner needs
no shift between its steps. Below the first octave the lookup gives zero for each coefficient
(the index byte does not match the table page), thus h is 0 there, with an error below 1.3e-4.

Usage:
    tools/htp-lab/gen/silu_i16.py > hvx-silu-i16.h
    tools/htp-lab/gen/silu_i16.py --check      # print the error of the integer path
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

DEGREE = 4
E_LO = -12          # the first octave is [2^-12, 2^-11)
E_HI = 4            # the input is limited to values below 2^4
SCALE = 16          # h as an integer is h * 2^16
PIECES = E_HI - E_LO    # 16, one page of vlut16


def h(u: np.ndarray) -> np.ndarray:
    """The bump u / (1 + e^u) for u >= 0."""
    return u / (1.0 + np.exp(u))


def table() -> np.ndarray:
    """Fit one polynomial for each octave at Chebyshev nodes. O(PIECES).

    Returns:
        The integer coefficients [PIECES][DEGREE + 1], the constant term first

    Raises:
        ValueError: If a coefficient does not fit an int16
    """
    nodes = (np.cos(np.pi * (np.arange(96) + 0.5) / 96) + 1) / 2
    rows = [np.polynomial.polynomial.polyfit(nodes, h(2.0 ** e * (1 + nodes)), DEGREE)
            for e in range(E_LO, E_HI)]
    c = np.round(np.array(rows) * 2.0 ** SCALE).astype(np.int64)
    if np.max(np.abs(c)) > 32767:
        raise ValueError(f"a coefficient is {np.max(np.abs(c))}, more than an int16 holds")
    return c


def evaluate(u16: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The integer path of the HVX routine, for an array of f16 values of u. O(len(u16)).

    Args:
        u16: The non-negative inputs as float16
        q: The coefficient table of table()

    Returns:
        h as float64
    """
    bits = u16.view(np.uint16).astype(np.int64)
    bits = np.minimum(bits, 0x4BFF)                      # the largest f16 below 16
    idx = (bits >> 10) - (E_LO + 15)
    v = (bits & 0x3FF) << 5                              # the mantissa as Q15
    ok = idx >= 0
    idx = np.clip(idx, 0, PIECES - 1)
    acc = q[idx, DEGREE]
    for d in range(DEGREE - 1, -1, -1):
        acc = np.clip(((acc * v * 2 + 0x8000) >> 16) + q[idx, d], -32768, 32767)
    return np.where(ok, acc, 0) / 2.0 ** SCALE


def integer_h(q: np.ndarray) -> np.ndarray:
    """h * 2^SCALE for every f16 input, before the clamp. O(65536).

    Args:
        q: The coefficient table of table()

    Returns:
        The integer results, one for each of the 65536 f16 bit patterns
    """
    bits = np.arange(0x10000, dtype=np.uint16).astype(np.int64) & 0x7FFF
    u = np.minimum(bits, 0x4BFF)
    idx = (u >> 10) - (E_LO + 15)
    v = ((u << 6) & 0xFFFF) >> 1
    ok = idx >= 0
    k = np.clip(idx, 0, PIECES - 1)
    acc = q[k, DEGREE]
    for d in range(DEGREE - 1, -1, -1):
        acc = np.clip((acc * v * 2 + 0x8000) >> 16, -32768, 32767)
        acc = np.clip(acc + q[k, d], -32768, 32767)
    return np.where(ok, acc, 0)


def h_bound(q: np.ndarray) -> int:
    """The largest integer result of the routine. O(65536)."""
    return int(np.max(integer_h(q)))


def n_negative(q: np.ndarray) -> int:
    """The number of f16 inputs where the integer Horner is negative. O(65536).

    The clamp of the routine covers these inputs. The count goes into the comment of the
    generated file, thus a change of the fit shows there.
    """
    return int(np.sum(integer_h(q) < 0))


def check() -> int:
    """Print the error of relu(x) - h(|x|) against the exact SiLU.

    Returns:
        The exit status
    """
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(-18, 18, 400000), rng.normal(0, 1.5, 400000),
                        rng.normal(0, 0.05, 100000)]).astype(np.float32)
    x64 = x.astype(np.float64)
    ref = x64 / (1 + np.exp(-x64))
    q = table()
    y = np.maximum(x64, 0) - np.maximum(evaluate(np.abs(x).astype(np.float16), q), 0.0)
    print(f"integer h: max {h_bound(q)}, negative for {n_negative(q)} of 65536 f16 inputs"
          f" (the routine clamps these to 0)")
    err = y - ref
    print(f"max abs {np.max(np.abs(err)):.3e}  rms {np.sqrt(np.mean(err ** 2)):.3e}"
          f"  nmse {np.sum(err ** 2) / np.sum(ref ** 2):.2e}")
    return 0


HEADER = '''// The SiLU in int16 for the HVX. tools/htp-lab/gen/silu_i16.py of the qwen-mobile repository
// generates this file, do not edit it.
//
// SiLU(x) = relu(x) - h(|x|) with h(u) = u / (1 + e^u). h is a smooth bump in [0, 0.2785], thus it
// fits an int16 with the scale 2^16, and relu(x) stays exact in the float domain.
//
// One polynomial piece of degree {degree} covers one octave [2^e, 2^(e+1)) of u, for e = {e_lo} to {e_hi_m1}.
// The piece index is the exponent field of the f16 value of u, and the local coordinate is its
// 10-bit mantissa, thus the routine has no float to integer conversion. All coefficients have the
// scale 2^16, thus Horner has no shift between its steps. vlut16 reads the coefficients: an index
// byte out of the page 0 gives 0, thus h is 0 below 2^{e_lo} (the error is below 1.3e-4). The
// input is limited to values below 16, where h is 1.8e-6.
//
// The measured error of relu(x) - h(|x|) against the exact SiLU: max 1.6e-4, NMSE 4e-11. Most
// of that is the f16 rounding of |x|, not the polynomial.
//
// The contract of hvx_silu_h_i16: the result is in [0, {h_max}]. The routine clamps, because the
// integer Horner of the last octave gives -1 or -2 for {n_neg} of the 65536 f16 inputs, all with
// |x| in [14.4922, 15.7266]. The fitted polynomial of that octave crosses zero near the top of
// the octave, where h itself is 1e-5. A caller that divides h by 2^{scale} with an unsigned saturating
// subtract of {scale} steps from the f16 exponent field then reads -1 as +Inf: the bit pattern of the
// f16 -1.0 is 0xBC00, and 0xBC00 - 0x4000 is 0x7C00. Refer to the check of this range in
// tools/htp-lab/lab/target_gdn_chunk.c, phase silu.
//
// Why int16: a v79 packet holds 2 int16 multiplies or 1 qfloat multiply, 4 selects or integer adds,
// and each packet costs 2 cycles. A multiply result is ready 2 packets after its packet. Thus
// the caller must give the routine 2 or 4 independent vectors, and the routine does each step
// for all vectors before the next step.

#ifndef HVX_SILU_I16_H
#define HVX_SILU_I16_H

#include <stdint.h>

#include "hvx-base.h"

#define HVX_SILU_I16_DEGREE {degree}
#define HVX_SILU_I16_SCALE  {scale}

// vlut16 in the 128-byte mode, from the full map that tools/htp-lab measures (target f16math, all
// 32 values of Rt and all 256 byte values, equal to the documented form at each point):
//   - An index byte matches when its high nibble is equal to Rt & 15. The entry is the table word
//     (byte % 32), and the bit 1 of Rt selects the odd halfword of that word. A byte that does not
//     match gives 0.
//   - The output pair is divided by the byte position: lo holds the results of the even bytes and
//     hi the results of the odd bytes.
// Thus with Rt = 0 the coefficient of the octave k (0 to 15) is the halfword 2 * k. One 16-bit lane
// holds the index of one vector in its low byte and the index of a second vector in its high
// byte, and one lookup gives the coefficients of the two vectors. The row d holds the coefficient
// of v^d.
static const int16_t hvx_silu_i16_table[HVX_SILU_I16_DEGREE + 1][64] __attribute__((aligned(128))) = {{
{rows}
}};

// h(|a|) * 2^16 as int16 for n vectors of 64 f16 lanes. n is 1, 2 or 4 and a constant at the call.
// The result is in [0, {h_max}] and the lane order of h is the lane order of a.
// The routine must be inline: n is then a constant, the loops unroll, and the vectors stay in registers.
static inline __attribute__((always_inline)) void hvx_silu_h_i16(const HVX_Vector * a, HVX_Vector * h, const int n) {{
    const HVX_Vector u_max  = Q6_Vh_vsplat_R(0x4bff);              // the largest f16 below 16
    const HVX_Vector e_base = Q6_Vh_vsplat_R({e_base});                  // the exponent field of 2^{e_lo}
    const HVX_Vector m_byte = Q6_Vh_vsplat_R(0x00ff);
    const HVX_Vector zero   = Q6_V_vzero();

    // The table address is opaque to the compiler, thus it reloads a coefficient vector where it
    // needs one instead of holding all 5 in registers across the loop of the caller. Loads are the
    // cheap resource here and registers are the scarce one.
    const int16_t (* tbl)[64] = hvx_silu_i16_table;
    __asm__("" : "+r"(tbl));

    HVX_Vector u[4], idx[4], v[4], acc[4];
    HVX_Vector ib[2];
    const int np = (n + 1) / 2;

    for (int r = 0; r < n; r++) {{
        u[r] = Q6_Vh_vmin_VhVh(Q6_Vhf_vabs_Vhf(a[r]), u_max);
    }}
    // k = the octave number, 0 to 15. Below the table k is -1 to -3, its byte is 0xfd to 0xff, and
    // that byte does not match the page 0, thus each coefficient and h are 0.
    for (int r = 0; r < n; r++) {{
        idx[r] = Q6_Vh_vsub_VhVh(Q6_Vuh_vlsr_VuhR(u[r], 10), e_base);
    }}
    // The mantissa as Q15. A left shift of 6 drops the sign and the exponent, and the logical
    // right shift of 1 puts the 10 mantissa bits at the bits 14 to 5. No mask register is needed.
    for (int r = 0; r < n; r++) {{
        v[r] = Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(u[r], 6), 1);
    }}
    // The low byte of a lane is an even byte and the high byte is an odd byte, thus the first vector
    // of a pair comes out in lo and the second in hi.
    for (int p = 0; p < np; p++) {{
        const HVX_Vector second = idx[(2 * p + 1 < n) ? 2 * p + 1 : 2 * p];
        ib[p] = Q6_V_vor_VV(Q6_V_vand_VV(idx[2 * p], m_byte), Q6_Vh_vasl_VhR(second, 8));
    }}
    for (int p = 0; p < np; p++) {{
        const HVX_VectorPair c = Q6_Wh_vlut16_VbVhR(ib[p], hvx_vmem(tbl[HVX_SILU_I16_DEGREE]), 0);
        acc[2 * p] = Q6_V_lo_W(c);
        if (2 * p + 1 < n) {{
            acc[2 * p + 1] = Q6_V_hi_W(c);
        }}
    }}
    _Pragma("clang loop unroll(full)")
    for (int d = HVX_SILU_I16_DEGREE - 1; d >= 0; d--) {{
        HVX_Vector c[4];
        for (int p = 0; p < np; p++) {{
            const HVX_VectorPair cp = Q6_Wh_vlut16_VbVhR(ib[p], hvx_vmem(tbl[d]), 0);
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
    // h is a value of [0, 0.2785] and the polynomial of the last octave crosses zero, thus the
    // clamp holds the contract of the routine. It is one integer ALU operation for each vector,
    // and a packet holds 4 of them, thus the measured cost of the whole op is 1.8 %.
    for (int r = 0; r < n; r++) {{
        h[r] = Q6_Vh_vmax_VhVh(acc[r], zero);
    }}
}}

// -h as a qf32 pair, from h * 2^16 as int16. The int16 becomes an f16 number, an unsigned
// saturating subtract of 2 from its exponent field divides it by 4 and keeps a zero a zero,
// and the widening multiply by -2^-14 gives the scale and the 32-bit form in one step.
// A subtract of 2 steps cannot cross the sign bit of the f16, thus this form is correct for a
// negative h16 as well. A subtract of 16 steps is not.
static inline __attribute__((always_inline)) HVX_VectorPair hvx_silu_neg_h_qf32(HVX_Vector h16) {{
    const HVX_Vector quarter = Q6_Vh_vsplat_R(0x0800);
    const HVX_Vector k       = Q6_Vh_vsplat_R(0x8400);             // -2^-14 as f16
    const HVX_Vector hf      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vhf_equals_Vh(h16), quarter);
    return Q6_Wqf32_vmpy_VhfVhf(hf, k);
}}

#endif /* HVX_SILU_I16_H */
'''


def main(argv: list[str] | None = None) -> int:
    """Write the header to stdout, or print the error with --check.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="print the error of the integer path")
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
    sys.stdout.write(HEADER.format(degree=DEGREE, scale=SCALE, e_lo=E_LO, e_hi_m1=E_HI - 1,
                                   e_base=E_LO + 15, rows="\n".join(rows),
                                   h_max=h_bound(q), n_neg=n_negative(q)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
