#!/usr/bin/env python3
"""The piecewise int16 table machine of the HVX activation generators.

Every int16 activation of the HTP kernels has the same shape. One polynomial piece covers one
octave [2^e, 2^(e+1)) of a non-negative input u. The piece index is the exponent field of the f16
value of u, and the local coordinate is its 10-bit mantissa, thus the routine needs no float to
integer conversion. All coefficients of one table share one scale, thus the Horner loop needs no
shift between its steps. vlut16 reads the coefficients: an index byte outside the page gives 0,
thus the value is 0 below the first octave.

This module holds the parts that every generator of that family shares:

    fit_table     the Chebyshev fit of one polynomial for each octave, with the integer scale
    evaluate      the Horner loop of the HVX routine, with the saturation of the vector unit
    sweep_inputs  the input corpus of a --check mode
    emit_rows     the C text of one coefficient table
    cli           the --check parser and the dispatch of a generator

The callers are tools/htp-lab/gen/act_i16.py, silu_i16.py and softplus_i16.py. Each one keeps its
own functions, its own header text and its own check text. A change of this module changes the
bytes of three shipped headers, thus verify a change with tools/htp-lab/gen/check_tree.py: it runs
each generator and compares the output with the header of the kernel tree, byte for byte.
"run.sh all" runs that check.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

# The Chebyshev nodes of the fit of one octave. The value comes from the generators.
FIT_NODES = 96

# The lanes of one vlut16 page. A table row holds one coefficient of each octave in every second
# lane, because vlut16 reads a pair of bytes for each lane.
TABLE_LANES = 64

# The largest f16 bit pattern below 16. The routines limit the input to it.
U_MAX_BITS = 0x4BFF

# The exponent bias of an f16
F16_BIAS = 15


@dataclass(frozen=True)
class Piecewise:
    """One piecewise int16 function of a non-negative input u.

    Attributes:
        f: The function of u, evaluated on an array of float64
        e_lo: The exponent of the first octave, thus the first octave is [2^e_lo, 2^(e_lo + 1))
        e_hi: The exponent past the last octave, thus the input is limited to values below 2^e_hi
        degree: The degree of the polynomial of one octave
        scale: The integer scale, thus the integer value is the value times 2^scale
        clip_coeffs: True to clip a coefficient to the int16 range, False to raise a ValueError.
            A function whose value is exactly 0.5 at u = 0 gives the coefficient 32768, which an
            int16 does not hold. A clip to 32767 costs 7.6e-6 of the value, and the saturating add
            of the HVX would do the same.
    """

    f: Callable[[np.ndarray], np.ndarray]
    e_lo: int
    e_hi: int
    degree: int
    scale: int
    clip_coeffs: bool = False

    @property
    def pieces(self) -> int:
        """The octaves of the table. One vlut16 page holds 16 of them."""
        return self.e_hi - self.e_lo

    @property
    def e_base(self) -> int:
        """The exponent field of the f16 value of the first octave."""
        return self.e_lo + F16_BIAS


def fit_table(spec: Piecewise, nodes: int = FIT_NODES) -> np.ndarray:
    """Fits one polynomial for each octave at Chebyshev nodes. Complexity O(pieces).

    Args:
        spec: The function and the parameters of the table
        nodes: The Chebyshev nodes of one fit

    Returns:
        The integer coefficients [pieces][degree + 1], the constant term first

    Raises:
        ValueError: If a coefficient does not fit an int16 and spec.clip_coeffs is False
    """
    t = (np.cos(np.pi * (np.arange(nodes) + 0.5) / nodes) + 1) / 2
    rows = [np.polynomial.polynomial.polyfit(t, spec.f(2.0 ** e * (1 + t)), spec.degree)
            for e in range(spec.e_lo, spec.e_hi)]
    c = np.round(np.array(rows) * 2.0 ** spec.scale)
    if spec.clip_coeffs:
        return np.clip(c, -32767, 32767).astype(np.int64)
    c = c.astype(np.int64)
    if np.max(np.abs(c)) > 32767:
        raise ValueError(f"a coefficient is {np.max(np.abs(c))}, more than an int16 holds")
    return c


def evaluate(spec: Piecewise, u16: np.ndarray, q: np.ndarray, low_clamp: bool = False,
             split_clip: bool = False) -> np.ndarray:
    """The integer path of the HVX evaluator, for an array of f16 values of u.

    The saturating add of the HVX bounds every step to an int16, thus this model saturates too.
    Complexity O(len(u16)).

    Args:
        spec: The function and the parameters of the table
        u16: The non-negative inputs as float16, or their bit patterns as int64
        q: The coefficient table of fit_table
        low_clamp: True to raise a u below the first octave to the first octave. A function that is
            not 0 at u = 0 needs this clamp, thus every lane reads the first octave and no lane
            misses the lookup.
        split_clip: True to saturate the product before the add of the constant term, and False to
            saturate the sum only. The HVX multiply and the HVX add each saturate, thus True is the
            model of the two instructions and False is the model of one fused step.

    Returns:
        The value times 2^scale, as int64
    """
    bits = u16 if u16.dtype == np.int64 else u16.view(np.uint16).astype(np.int64)
    bits = np.minimum(bits, U_MAX_BITS)
    if low_clamp:
        bits = np.maximum(bits, spec.e_base << 10)
    idx = (bits >> 10) - spec.e_base
    v = (bits & 0x3FF) << 5                              # the mantissa as Q15
    ok = idx >= 0
    idx = np.clip(idx, 0, spec.pieces - 1)
    acc = q[idx, spec.degree]
    for d in range(spec.degree - 1, -1, -1):
        if split_clip:
            acc = np.clip((acc * v * 2 + 0x8000) >> 16, -32768, 32767)
            acc = np.clip(acc + q[idx, d], -32768, 32767)
        else:
            acc = np.clip(((acc * v * 2 + 0x8000) >> 16) + q[idx, d], -32768, 32767)
    return np.where(ok, acc, 0)


def sweep_inputs(lo: float, hi: float, seed: int = 1, near_zero: bool = False,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """The input corpus of a --check mode: a uniform range, a normal spread, and values near 0.

    The three parts together reach the large magnitudes, the magnitudes of an activation of the
    model, and the values where a relative error is largest. Complexity O(1).

    Args:
        lo: The low end of the uniform part
        hi: The high end of the uniform part
        seed: The seed of a generator of its own, thus the corpus is the same in each run
        near_zero: True to add 50000 values in [-1e-3, 1e-3]
        rng: The generator to draw from, or None for a generator of the seed. A caller that needs
            more values after the corpus gives its own generator, thus the corpus and the values
            after it come from one stream.

    Returns:
        The corpus as float32
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    parts = [rng.uniform(lo, hi, 400000), rng.normal(0, 1.5, 400000), rng.normal(0, 0.05, 100000)]
    if near_zero:
        parts.append(rng.uniform(-1e-3, 1e-3, 50000))
    return np.concatenate(parts).astype(np.float32)


def emit_rows(q: np.ndarray, degree: int, pieces: int, lanes: int = TABLE_LANES) -> str:
    """The C text of one coefficient table, one row for each degree.

    vlut16 reads a pair of bytes for each lane, thus the coefficient of octave k sits in lane 2k and
    the lane between two of them holds 0. Complexity O(degree * lanes).

    Args:
        q: The coefficient table of fit_table
        degree: The degree of the polynomial of one octave
        pieces: The octaves of the table
        lanes: The lanes of one vlut16 page

    Returns:
        The rows of the table, joined by a newline and with no newline at the end
    """
    rows = []
    for d in range(degree + 1):
        cells = ["0"] * lanes
        for k in range(pieces):
            cells[2 * k] = str(int(q[k, d]))
        rows.append(f"    // the coefficient of v^{d}\n    {{ " + ", ".join(cells) + " },")
    return "\n".join(rows)


def cli(doc: str, emit: Callable[[], str], check: Callable[[], int],
        check_help: str = "print the error of the integer path",
        argv: list[str] | None = None) -> int:
    """The command line of a generator: write the header to stdout, or do a check with --check.

    Args:
        doc: The docstring of the generator, for the help text
        emit: The function that gives the text of the header
        check: The function that prints the error and gives the exit status
        check_help: The help text of the option --check
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help=check_help)
    a = ap.parse_args(argv)
    if a.check:
        return check()
    sys.stdout.write(emit())
    return 0
