"""The Hypothesis strategies of the fuzzers of the quantization pipeline.

A strategy draws the structure of a matrix (the shape, a profile and a
magnitude per block of 32) and one integer seed. A numpy generator with
that seed then fills the values. Thus the shrinker makes the structure
simple (fewer rows, fewer blocks, plain profiles, seed 0), and a drawn
example stays small in the database.

The profiles of a block:

- gauss: normal values
- zero: all zeros
- const: one value 32 times
- spike: one large value, the others 2^-12 of it
- sparse: zeros and 1 to 4 normal values
- spread: magnitudes spread over 2^-20 .. 1 in one block
- ties: odd multiples of a half step, the rounding midpoints of a grid
- opposite: the block maximum and its negative in the same block
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st

from qfz_common import F16_MAX

PROFILES = ("gauss", "zero", "const", "spike", "sparse", "spread", "ties", "opposite")
BLOCK = 32


@dataclass(frozen=True)
class MatrixSpec:
    """The structure of a drawn matrix.

    Attributes:
        rows: The number of rows
        nblocks: The number of blocks of 32 in a row
        profiles: The profile of each block, rows x nblocks names
        exponents: The base-2 magnitude of each block, rows x nblocks integers
        seed: The seed of the values
        f16: True when the values are F16 values, as in a converter F16 GGUF
    """

    rows: int
    nblocks: int
    profiles: tuple[tuple[str, ...], ...]
    exponents: tuple[tuple[int, ...], ...]
    seed: int
    f16: bool

    def build(self) -> np.ndarray:
        """Give the float32 matrix [rows, nblocks * 32] of the spec. Complexity is O(rows * cols)."""
        gen = np.random.default_rng(self.seed)
        out = np.zeros((self.rows, self.nblocks * BLOCK), dtype=np.float64)
        for r in range(self.rows):
            for b in range(self.nblocks):
                out[r, b * BLOCK:(b + 1) * BLOCK] = _block(self.profiles[r][b], 2.0 ** self.exponents[r][b], gen)
        if self.f16:
            out = np.clip(out, -F16_MAX, F16_MAX).astype(np.float16)
        return out.astype(np.float32)


def _block(profile: str, scale: float, gen: np.random.Generator) -> np.ndarray:
    """Give the 32 float64 values of one block of a profile and a magnitude."""
    if profile == "gauss":
        return gen.standard_normal(BLOCK) * scale
    if profile == "zero":
        return np.zeros(BLOCK)
    if profile == "const":
        return np.full(BLOCK, scale * (1.0 if gen.random() < 0.5 else -1.0))
    if profile == "spike":
        v = gen.standard_normal(BLOCK) * scale * 2.0 ** -12
        v[gen.integers(BLOCK)] = scale * (1.0 if gen.random() < 0.5 else -1.0)
        return v
    if profile == "sparse":
        v = np.zeros(BLOCK)
        where = gen.choice(BLOCK, size=int(gen.integers(1, 5)), replace=False)
        v[where] = gen.standard_normal(where.size) * scale
        return v
    if profile == "spread":
        return np.sign(gen.standard_normal(BLOCK)) * scale * 2.0 ** gen.uniform(-20.0, 0.0, BLOCK)
    if profile == "ties":
        return (2 * gen.integers(-8, 8, BLOCK) + 1) * scale / 16.0
    if profile == "opposite":
        v = gen.standard_normal(BLOCK) * scale * 0.25
        i, j = gen.choice(BLOCK, size=2, replace=False)
        v[i], v[j] = scale, -scale
        return v
    raise ValueError(f"the block profile {profile} is not one of {PROFILES}")


@st.composite
def matrices(draw: st.DrawFn, max_rows: int = 6, max_blocks: int = 6, f16: bool = True,
             min_exp: int = -24, max_exp: int = 4) -> MatrixSpec:
    """Draw a matrix spec.

    Args:
        draw: The Hypothesis draw function
        max_rows: The largest number of rows
        max_blocks: The largest number of blocks of 32 in a row
        f16: True to give F16 values, as a converter F16 GGUF holds them
        min_exp: The smallest base-2 magnitude of a block
        max_exp: The largest base-2 magnitude of a block

    Returns:
        The spec. Call ``build()`` for the values.
    """
    rows = draw(st.integers(1, max_rows))
    nblocks = draw(st.integers(1, max_blocks))
    profile = st.sampled_from(PROFILES)
    exponent = st.integers(min_exp, max_exp)
    profiles = tuple(tuple(draw(profile) for _ in range(nblocks)) for _ in range(rows))
    exponents = tuple(tuple(draw(exponent) for _ in range(nblocks)) for _ in range(rows))
    seed = draw(st.integers(0, 2**32 - 1))
    return MatrixSpec(rows, nblocks, profiles, exponents, seed, f16)


@st.composite
def raw_float_matrices(draw: st.DrawFn, max_rows: int = 4, max_blocks: int = 4) -> np.ndarray:
    """Draw a float32 matrix element by element, NaN, the infinities and the extremes included.

    This strategy gives the shrinker full control of each value, thus it
    finds the smallest value that breaks a bound. It is slow, thus the
    matrices are small.
    """
    rows = draw(st.integers(1, max_rows))
    nblocks = draw(st.integers(1, max_blocks))
    element = st.floats(width=32, allow_nan=True, allow_infinity=True)
    values = draw(st.lists(element, min_size=rows * nblocks * BLOCK, max_size=rows * nblocks * BLOCK))
    return np.asarray(values, dtype=np.float32).reshape(rows, nblocks * BLOCK)
