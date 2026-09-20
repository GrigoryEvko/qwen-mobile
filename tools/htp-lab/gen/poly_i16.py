#!/usr/bin/env python3
"""Generate hvx-poly-i16.h: piecewise int16 polynomial evaluators for the HVX.

WHY this exists. On the Hexagon v79 a packet holds one qfloat multiply but two int16
multiplies, and a result is ready two packets after its producer. A transcendental written as
one chain of qfloat operations thus costs about four cycles for each of its 25 or more steps.
The same function as a piecewise polynomial in int16, evaluated over four independent vectors,
costs about one cycle for each step. That is the whole reason for this file.

THE SCHEME. Each function is approximated on the magnitude u = |x| of an IEEE half.
One polynomial piece covers one octave [2^e, 2^(e+1)), thus:
  - The piece index is the exponent field of the f16 value, which needs no conversion.
  - The local coordinate is the 10-bit mantissa, read as Q15 in [0, 1).
  - vlut16 reads the coefficients. All coefficients of one function share one scale, thus
    Horner needs no shift between its steps.
  - Below the first octave every index byte fails to match the page and the lookup gives 0,
    thus the value is 0 there. Select e_lo so that the function is 0 to the working precision.

vlut16 IN THE 128-BYTE MODE, from the full map that tools/htp-lab measures (target f16math,
all 32 values of Rt and all 256 byte values, equal to the documented form at every point):
  - An index byte matches when its high nibble is equal to Rt & 15. The entry is the table word
    (byte % 32), and the bit 1 of Rt selects the odd halfword of that word. A byte that does not
    match gives 0.
  - The output pair is divided by the byte position: lo holds the results of the even bytes and
    hi the results of the odd bytes.
Thus with Rt = 0 the coefficient of the octave k is the halfword 2 * k, and one 16-bit lane
carries the index of one vector in its low byte and of a second vector in its high byte, so one
lookup serves two vectors.

THE SCOPE. This fits a smooth bounded function of |x|: a sigmoid-family bump, tanh, erf, a
reciprocal over a limited range. It does not fit exp or log, whose natural decomposition is the
exponent and the mantissa of the result rather than of the input.

TO ADD A FUNCTION: append one entry at the end of FUNCS. Give the mathematical function of u,
the octave range, the degree and the output scale, then regenerate and run --check.

Usage:
    tools/htp-lab/gen/poly_i16.py emit > hvx-poly-i16.h     # every function
    tools/htp-lab/gen/poly_i16.py emit silu_bump            # named functions only
    tools/htp-lab/gen/poly_i16.py check                     # the error of each
    tools/htp-lab/gen/poly_i16.py list
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

PIECES = 16          # one page of vlut16, thus 16 octaves
TABLE_LANES = 64     # halfwords of one HVX vector


@dataclass(frozen=True)
class Func:
    """One function to approximate.

    Attributes:
        name: The identifier that the generated names carry
        f: The function of the magnitude u, for u >= 0, as a numpy expression
        e_lo: The exponent of the first octave. The value is 0 below 2^e_lo
        degree: The degree of each piece
        scale: The output is the value times 2^scale, as an int16
        at_zero: The value of f at u = 0. The table holds g(u) = at_zero - f(u), which is 0 at
            u = 0, thus the region below the table gives f = at_zero and not 0. For a function
            that already vanishes at 0 this is 0.0 and g is f
        doc: One line that says what the function is and where it is used
    """

    name: str
    f: Callable[[np.ndarray], np.ndarray]
    e_lo: int
    degree: int
    scale: int
    doc: str
    at_zero: float = 0.0

    def g(self, u: np.ndarray) -> np.ndarray:
        """The function that the table holds, which is 0 at u = 0.

        Args:
            u: The magnitudes, u >= 0

        Returns:
            at_zero - f(u), or f(u) when at_zero is 0
        """
        return (self.at_zero - self.f(u)) if self.at_zero else self.f(u)


FUNCS: list[Func] = [
    Func("silu_bump", lambda u: u / (1.0 + np.exp(u)), -12, 4, 16,
         "h(u) = u / (1 + e^u), the bump of SiLU: silu(x) = relu(x) - h(|x|). Range [0, 0.2785]"),
    Func("sigmoid_drop", lambda u: 1.0 / (1.0 + np.exp(u)), -12, 4, 16,
         "the table holds 0.5 - 1/(1+e^u), thus sigmoid(x) = 0.5 -+ table(|x|), the sign of the term being that of x. Range [0, 0.5]",
         at_zero=0.5),
    Func("tanh_half", lambda u: np.tanh(u), -12, 4, 14,
         "tanh(u) for u >= 0, odd: tanh(x) = sign(x) * tanh(|x|). Range [0, 1]"),
    Func("exp_drop", lambda u: np.exp(-u), -12, 5, 15,
         "the table holds 1 - e^-u, thus e^-u = 1 - table(u) for u >= 0, the decay of a gate. Range [0, 1]",
         at_zero=1.0),
]


def _e_hi(fn: Func) -> int:
    """The exponent after the last octave.

    Args:
        fn: The function

    Returns:
        The exponent of the upper limit of the table
    """
    return fn.e_lo + PIECES


def table(fn: Func) -> np.ndarray:
    """Fit one polynomial for each octave at Chebyshev nodes. O(PIECES).

    Args:
        fn: The function

    Returns:
        The integer coefficients [PIECES][degree + 1], the constant term first

    Raises:
        ValueError: If a coefficient does not fit an int16
    """
    nodes = (np.cos(np.pi * (np.arange(96) + 0.5) / 96) + 1) / 2
    rows = [np.polynomial.polynomial.polyfit(nodes, fn.g(2.0 ** e * (1 + nodes)), fn.degree)
            for e in range(fn.e_lo, _e_hi(fn))]
    c = np.round(np.array(rows) * 2.0 ** fn.scale).astype(np.int64)
    worst = int(np.max(np.abs(c)))
    if worst > 32767:
        raise ValueError(f"{fn.name}: a coefficient is {worst}, more than an int16 holds. "
                         f"Decrease scale below {fn.scale}.")
    return c


def evaluate(fn: Func, u16: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The integer path of the generated routine, bit for bit. O(len(u16)).

    Args:
        fn: The function
        u16: The non-negative inputs as float16
        q: The coefficient table of table()

    Returns:
        The value as float64
    """
    bits = np.minimum(u16.view(np.uint16).astype(np.int64), u_max_bits(fn))
    idx = (bits >> 10) - (fn.e_lo + 15)
    v = (bits & 0x3FF) << 5
    ok = idx >= 0
    acc = q[np.clip(idx, 0, PIECES - 1), fn.degree]
    for d in range(fn.degree - 1, -1, -1):
        acc = np.clip(((acc * v * 2 + 0x8000) >> 16) + q[np.clip(idx, 0, PIECES - 1), d], -32768, 32767)
    return np.where(ok, acc, 0) / 2.0 ** fn.scale


def u_max_bits(fn: Func) -> int:
    """The bit pattern of the largest f16 that the table covers.

    Args:
        fn: The function

    Returns:
        The f16 bits of the value just below 2^e_hi
    """
    return int(np.float16(2.0 ** _e_hi(fn) * (1 - 2.0 ** -11)).view(np.uint16))


def cmd_check(names: list[str]) -> int:
    """Print the error of each function against float64. O(1e6) per function.

    Args:
        names: The function names, or an empty list for all

    Returns:
        The exit status
    """
    rng = np.random.default_rng(1)
    print(f"{'function':<14} {'pieces':>6} {'deg':>4} {'scale':>6} {'max abs':>10} {'rms abs':>10}   note")
    for fn in FUNCS:
        if names and fn.name not in names:
            continue
        q = table(fn)
        lim = 2.0 ** _e_hi(fn)
        u = np.abs(np.concatenate([rng.uniform(-lim, lim, 400000), rng.normal(0, 1.5, 400000),
                                   rng.normal(0, 0.05, 200000)])).astype(np.float32)
        u = np.minimum(u, lim * (1 - 2.0 ** -11))
        err = evaluate(fn, u.astype(np.float16), q) - fn.g(u.astype(np.float64))
        print(f"{fn.name:<14} {PIECES:6d} {fn.degree:4d} {fn.scale:6d}"
              f" {np.max(np.abs(err)):10.3e} {np.sqrt(np.mean(err ** 2)):10.3e}   {fn.doc}")
    return 0


CORE = '''// Piecewise int16 evaluators for the HVX. tools/htp-lab/gen/poly_i16.py of the qwen-mobile
// repository generates this file, do not edit it. Refer to that file for the scheme, for the
// measured map of vlut16, and for how to add a function.
//
// WHY: on the v79 a packet holds one qfloat multiply and two int16 multiplies, and a result is
// ready two packets after its producer. A transcendental as one chain of qfloat operations costs
// about four cycles for each of its steps. The same function as a piecewise polynomial in int16
// over four independent vectors costs about one cycle for each step.
//
// HOW TO CALL: give the routine 1, 2 or 4 vectors of non-negative f16 magnitudes and a constant
// count. It gives the value times 2^scale as an int16 in the lane order of the input. Four
// vectors is the number that fills the packets.

#ifndef HVX_POLY_I16_H
#define HVX_POLY_I16_H

#include <stdint.h>

#include "hvx-base.h"

// The Horner core. table[d] is the coefficient vector of v^d, e_base is e_lo + 15 (the exponent
// field of the first octave), u_max_bits is the f16 bit pattern of the largest covered value.
// degree and n are constants at the call, thus the loops unroll and the vectors stay in registers.
#define HVX_POLY_I16_EVAL(table, degree, e_base, u_max_bits, a, out, n)                            \\
    do {                                                                                           \\
        const HVX_Vector _m_abs  = Q6_Vh_vsplat_R(0x7fff);                                         \\
        const HVX_Vector _u_max  = Q6_Vh_vsplat_R(u_max_bits);                                     \\
        const HVX_Vector _e_base = Q6_Vh_vsplat_R(e_base);                                         \\
        const HVX_Vector _m_mant = Q6_Vh_vsplat_R(0x03ff);                                         \\
        const HVX_Vector _m_byte = Q6_Vh_vsplat_R(0x00ff);                                         \\
        HVX_Vector _u[4], _idx[4], _v[4], _acc[4], _ib[2];                                         \\
        const int _np = ((n) + 1) / 2;                                                             \\
        for (int _r = 0; _r < (n); _r++) {                                                         \\
            _u[_r] = Q6_Vh_vmin_VhVh(Q6_V_vand_VV((a)[_r], _m_abs), _u_max);                       \\
        }                                                                                          \\
        /* the octave number. Below the table it is negative, its byte does not match the page, */ \\
        /* and every coefficient of the lookup is 0, thus the value is 0.                       */ \\
        for (int _r = 0; _r < (n); _r++) {                                                         \\
            _idx[_r] = Q6_Vh_vsub_VhVh(Q6_Vuh_vlsr_VuhR(_u[_r], 10), _e_base);                     \\
        }                                                                                          \\
        for (int _r = 0; _r < (n); _r++) {                                                         \\
            _v[_r] = Q6_Vh_vasl_VhR(Q6_V_vand_VV(_u[_r], _m_mant), 5);                             \\
        }                                                                                          \\
        /* the low byte of a lane is an even byte and the high byte an odd byte, thus the first */ \\
        /* vector of a pair comes back in lo and the second in hi                              */ \\
        for (int _p = 0; _p < _np; _p++) {                                                         \\
            const HVX_Vector _b = _idx[(2 * _p + 1 < (n)) ? 2 * _p + 1 : 2 * _p];                  \\
            _ib[_p] = Q6_V_vor_VV(Q6_V_vand_VV(_idx[2 * _p], _m_byte), Q6_Vh_vasl_VhR(_b, 8));     \\
        }                                                                                          \\
        for (int _p = 0; _p < _np; _p++) {                                                         \\
            const HVX_VectorPair _c = Q6_Wh_vlut16_VbVhR(_ib[_p], hvx_vmem((table)[degree]), 0);   \\
            _acc[2 * _p] = Q6_V_lo_W(_c);                                                          \\
            if (2 * _p + 1 < (n)) {                                                                \\
                _acc[2 * _p + 1] = Q6_V_hi_W(_c);                                                  \\
            }                                                                                      \\
        }                                                                                          \\
        for (int _d = (degree) - 1; _d >= 0; _d--) {                                               \\
            HVX_Vector _co[4];                                                                     \\
            for (int _p = 0; _p < _np; _p++) {                                                     \\
                const HVX_VectorPair _cp = Q6_Wh_vlut16_VbVhR(_ib[_p], hvx_vmem((table)[_d]), 0);  \\
                _co[2 * _p] = Q6_V_lo_W(_cp);                                                      \\
                if (2 * _p + 1 < (n)) {                                                            \\
                    _co[2 * _p + 1] = Q6_V_hi_W(_cp);                                              \\
                }                                                                                  \\
            }                                                                                      \\
            for (int _r = 0; _r < (n); _r++) {                                                     \\
                _acc[_r] = Q6_Vh_vmpy_VhVh_s1_rnd_sat(_acc[_r], _v[_r]);                           \\
            }                                                                                      \\
            for (int _r = 0; _r < (n); _r++) {                                                     \\
                _acc[_r] = Q6_Vh_vadd_VhVh_sat(_acc[_r], _co[_r]);                                 \\
            }                                                                                      \\
        }                                                                                          \\
        for (int _r = 0; _r < (n); _r++) {                                                         \\
            (out)[_r] = _acc[_r];                                                                  \\
        }                                                                                          \\
    } while (0)

// A value of the scale 2^s as a qf32 pair, multiplied by k. The int16 becomes an f16 number, an
// unsigned saturating subtract of (s - 10) from its exponent field divides it by 2^(s-10) and
// keeps a zero a zero, and the widening multiply gives the rest of the scale, the factor k and
// the 32-bit form in one step. k is an IEEE half, thus 0x8400 is -2^-14 and 0x0400 is 2^-14.
static inline __attribute__((always_inline))
HVX_VectorPair hvx_poly_i16_to_qf32(HVX_Vector value, int exp_drop, int k_hf) {
    const HVX_Vector drop = Q6_Vh_vsplat_R(exp_drop << 10);
    const HVX_Vector k    = Q6_Vh_vsplat_R(k_hf);
    return Q6_Wqf32_vmpy_VhfVhf(Q6_Vuh_vsub_VuhVuh_sat(Q6_Vhf_equals_Vh(value), drop), k);
}

'''

WRAPPER = '''
// {doc}
// The table is {pieces} octaves from 2^{e_lo}, degree {degree}, the value times 2^{scale}.
// Below 2^{e_lo} the table gives 0, thus the reconstruction is exact there as well.
// Measured against float64: max {max_err:.3e}, rms {rms_err:.3e}.
#define HVX_POLY_{upper}_DEGREE {degree}
#define HVX_POLY_{upper}_SCALE  {scale}
#define HVX_POLY_{upper}_E_BASE {e_base}
#define HVX_POLY_{upper}_U_MAX  0x{u_max:04x}

static const int16_t hvx_poly_{name}_table[{degree} + 1][64] __attribute__((aligned(128))) = {{
{rows}
}};

// {name}(|a[r]|) times 2^{scale} as an int16, for n vectors of 64 f16 lanes. n is 1, 2 or 4 and a
// constant at the call. The lane order of the output is the lane order of the input.
static inline __attribute__((always_inline))
void hvx_poly_{name}(const HVX_Vector * a, HVX_Vector * out, const int n) {{
    HVX_POLY_I16_EVAL(hvx_poly_{name}_table, {degree}, HVX_POLY_{upper}_E_BASE,
                      HVX_POLY_{upper}_U_MAX, a, out, n);
}}
'''


def cmd_emit(names: list[str]) -> int:
    """Write the header to stdout.

    Args:
        names: The function names, or an empty list for all

    Returns:
        The exit status
    """
    rng = np.random.default_rng(1)
    out = [CORE]
    for fn in FUNCS:
        if names and fn.name not in names:
            continue
        q = table(fn)
        lim = 2.0 ** _e_hi(fn)
        u = np.abs(np.concatenate([rng.uniform(-lim, lim, 200000),
                                   rng.normal(0, 1.5, 200000)])).astype(np.float32)
        u = np.minimum(u, lim * (1 - 2.0 ** -11))
        err = evaluate(fn, u.astype(np.float16), q) - fn.g(u.astype(np.float64))
        rows = []
        for d in range(fn.degree + 1):
            cells = ["0"] * TABLE_LANES
            for k in range(PIECES):
                cells[2 * k] = str(int(q[k, d]))
            rows.append(f"    // the coefficient of v^{d}\n    {{ " + ", ".join(cells) + " },")
        out.append(WRAPPER.format(name=fn.name, upper=fn.name.upper(), doc=fn.doc,
                                  pieces=PIECES, e_lo=fn.e_lo, degree=fn.degree, scale=fn.scale,
                                  e_base=fn.e_lo + 15, u_max=u_max_bits(fn), rows="\n".join(rows),
                                  max_err=np.max(np.abs(err)), rms_err=np.sqrt(np.mean(err ** 2))))
    out.append("\n#endif /* HVX_POLY_I16_H */\n")
    sys.stdout.write("".join(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_text in [("emit", "write the header to stdout"),
                            ("check", "print the error of each function"),
                            ("list", "print the functions")]:
        p = sub.add_parser(name, help=help_text)
        if name != "list":
            p.add_argument("names", nargs="*", help="the functions, or none for all")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for fn in FUNCS:
            print(f"{fn.name:<14} octaves 2^{fn.e_lo} to 2^{_e_hi(fn)}, degree {fn.degree}, scale 2^{fn.scale}   {fn.doc}")
        return 0
    return cmd_check(a.names) if a.cmd == "check" else cmd_emit(a.names)


if __name__ == "__main__":
    sys.exit(main())
