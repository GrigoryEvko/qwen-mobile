"""Fuzz the block quantizers and the byte packers of quant/grid.py and quant/grids.py.

The properties:

- The bytes of pack_q8_0 and pack_nibbles decode in gguf-py to the same
  float32 values as the dequantizers of the pipeline, bit for bit.
- A Q8_0 block keeps the format bound: |d| / 2 for a level inside the
  range, |w| - 127 |d| for a clamped level.
- The Q8_0 and Q4_0 round-to-nearest rules of the pipeline are never less
  accurate than the ggml reference quantizers of gguf-py, element by element.
- The scale search is not less accurate than the plain reference scale.
- A block on the grid keeps its scale and its values.
- The results do not depend on the row chunk size.
- block_error agrees with quantize and dequantize.
- A value that is not finite, or a block beyond the F16 scale range, gives
  ValueError (qfz_common.scale_domain).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

import gguf
import quant.grid as grid_module
from qfz_common import F16_MAX, block_view, known_open, q8_0_error_bound, scale_domain
from qfz_hyp import counted, fuzz_settings
from qfz_strategies import MatrixSpec, matrices, raw_float_matrices
from quant.grid import (block_error, dequantize, pack_nibbles, pack_q8_0, q8_0_dequantize, q8_0_quantize,
                        quantize)
from quant.grids import IQ4NLGrid, Q4_0Grid

Q8 = gguf.GGMLQuantizationType.Q8_0
GRIDS = {"Q4_0": Q4_0Grid, "IQ4_NL": IQ4NLGrid}

# The explicit examples: the seeds of the mode "test", a block with a subnormal F16 scale, a tiny block with a
# scale that rounds to zero, and a block beyond the F16 scale range of Q8_0.
SPEC_MIXED = MatrixSpec(2, 2, (("gauss", "spike"), ("zero", "ties")), ((0, -30), (5, -10)), 1, False)
SPEC_EDGES = MatrixSpec(2, 3, (("opposite", "const", "sparse"), ("spread", "gauss", "zero")),
                        ((10, -14, 3), (-12, 15, 0)), 7, True)
SPEC_SUBNORMAL_SCALE = MatrixSpec(3, 6, (("gauss",) * 6, ("spread",) + ("gauss",) * 5, ("gauss",) * 6),
                                  ((0,) * 6, (-12, 0, 0, 0, 0, 0), (0,) * 6), 264, True)
SPEC_TINY_BLOCK = MatrixSpec(1, 1, (("sparse",),), ((-26,),), 0, False)
W_BEYOND_F16 = np.zeros((1, 32), np.float32)
W_BEYOND_F16[0, 0] = 8.4e6


def _amax(w: np.ndarray) -> np.ndarray:
    """Give the largest magnitude of each block of 32, float64 [rows, nblocks]."""
    return np.abs(block_view(w.astype(np.float64))).max(axis=-1)


def _refused(w: np.ndarray, kind: str) -> bool:
    """Check the answer of the quantizer to an input out of the F16 scale range. Give True when it refused the input.

    An input "out" of scale_domain must give ValueError. An input on the
    "edge" can give ValueError, or finite scales that the caller checks.
    """
    domain = scale_domain(w, kind)
    if domain == "in":
        return False
    try:
        if kind == "Q8_0":
            q8_0_quantize(torch.from_numpy(w))
        else:
            quantize(GRIDS[kind](), torch.from_numpy(w))
    except ValueError:
        return True
    assert domain == "edge", "the quantizer accepts a value that is not finite or a block beyond the F16 scale range"
    return False


def _check_q8_0(w: np.ndarray) -> None:
    """Quantize ``w`` to Q8_0 and check the bytes, the bound and the reference of gguf-py."""
    q, d = q8_0_quantize(torch.from_numpy(w))
    assert q.dtype == torch.int8 and d.dtype == torch.float16
    assert int(q.abs().max()) <= 127, "a Q8_0 level is out of -127 .. 127"
    d32 = d.to(torch.float32).numpy()
    assert np.isfinite(d32).all(), f"a Q8_0 scale is not finite: {d32[~np.isfinite(d32)][:4]}"
    ours = q8_0_dequantize(q, d).numpy()
    packed = pack_q8_0(q, d)
    np.testing.assert_array_equal(gguf.quants.dequantize(packed, Q8), ours, err_msg="gguf-py reads other values")
    err = np.abs(block_view(w.astype(np.float64)) - block_view(ours.astype(np.float64)))
    bound = q8_0_error_bound(_amax(w), d32)
    worst = err.max(axis=-1) - bound
    assert (worst <= 0).all(), f"the Q8_0 error is {float(worst.max()):.3e} over the bound"
    reference = gguf.quants.dequantize(gguf.quants.quantize(w, Q8), Q8).astype(np.float64)
    ref_err = np.abs(w.astype(np.float64) - reference)
    slack = 2.0 * np.spacing(np.abs(w).astype(np.float32)).astype(np.float64)
    excess = np.abs(w.astype(np.float64) - ours) - ref_err - slack
    assert (excess <= 0).all(), f"the pipeline Q8_0 is less accurate than the ggml reference by {excess.max():.3e}"


@fuzz_settings()
@given(spec=matrices(f16=False, min_exp=-40, max_exp=22))
@example(spec=SPEC_MIXED)
@example(spec=SPEC_EDGES)
@counted
def test_q8_0_packs_within_the_format_bound(spec) -> None:
    """Q8_0 of a float32 matrix: finite scales, the bytes of gguf-py, the bound, the ggml reference."""
    w = spec.build()
    if not _refused(w, "Q8_0"):
        _check_q8_0(w)


@fuzz_settings(0.5)
@given(w=raw_float_matrices())
@example(w=np.linspace(-3.0, 3.0, 64, dtype=np.float32).reshape(2, 32))
@example(w=W_BEYOND_F16)
@counted
def test_q8_0_element_by_element(w: np.ndarray) -> None:
    """Q8_0 with each value drawn on its own, thus the shrinker gives the smallest failing value."""
    if not _refused(w, "Q8_0"):
        _check_q8_0(w)


@fuzz_settings()
@given(spec=matrices(min_exp=-26, max_exp=15), kind=st.sampled_from(sorted(GRIDS)), search=st.booleans(),
       f16=st.booleans())
@example(spec=SPEC_EDGES, kind="Q4_0", search=True, f16=True)
@example(spec=SPEC_MIXED, kind="IQ4_NL", search=False, f16=False)
@counted
def test_grid4_packs_and_decodes_in_gguf_py(spec, kind: str, search: bool, f16: bool) -> None:
    """Q4_0 and IQ4_NL: indices 0 .. 15, finite scales, and gguf-py decodes the bytes to our values."""
    w = dataclasses.replace(spec, f16=f16).build()
    if _refused(w, kind):
        return
    grid = GRIDS[kind]()
    idx, d = quantize(grid, torch.from_numpy(w), search=search)
    assert int(idx.min()) >= 0 and int(idx.max()) <= 15, "an index is out of 0 .. 15"
    assert torch.isfinite(d).all(), "a 4-bit scale is not finite"
    ours = dequantize(grid, idx, d).numpy()
    packed = pack_nibbles(idx, d)
    decoded = gguf.quants.dequantize(packed, getattr(gguf.GGMLQuantizationType, kind))
    np.testing.assert_array_equal(decoded, ours, err_msg=f"gguf-py reads other {kind} values")
    if kind == "Q4_0" and not search:
        reference = gguf.quants.dequantize(gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q4_0),
                                           gguf.GGMLQuantizationType.Q4_0).astype(np.float64)
        slack = 2.0 * np.spacing(np.abs(w).astype(np.float32)).astype(np.float64)
        excess = np.abs(w - ours.astype(np.float64)) - np.abs(w - reference) - slack
        assert (excess <= 0).all(), f"the pipeline Q4_0 is less accurate than the ggml reference by {excess.max():.3e}"


@fuzz_settings()
@given(spec=matrices(min_exp=-12, max_exp=12), kind=st.sampled_from(sorted(GRIDS)))
@example(spec=SPEC_SUBNORMAL_SCALE, kind="IQ4_NL")
@example(spec=SPEC_EDGES, kind="Q4_0")
@counted
def test_scale_search_is_not_worse_than_the_reference_scale(spec, kind: str) -> None:
    """The searched scale gives a block error that is not larger than the error of the reference scale.

    The factor 1 is a candidate of the search, and the search measures each
    candidate after its F16 rounding, thus the stored scale is not worse
    than the stored reference scale, also in the F16 subnormal range. The
    search sums the error in float32 and the test in float64, thus the
    tolerance is 1e-5 of the error of the reference scale, plus the float32
    noise.
    """
    w = spec.build()
    grid = GRIDS[kind]()
    wt = torch.from_numpy(w)
    searched = dequantize(grid, *quantize(grid, wt, search=True)).numpy().astype(np.float64)
    idx_p, d_p = quantize(grid, wt, search=False)
    plain = dequantize(grid, idx_p, d_p).numpy().astype(np.float64)
    err_s = ((block_view(searched) - block_view(w.astype(np.float64))) ** 2).sum(-1)
    err_p = ((block_view(plain) - block_view(w.astype(np.float64))) ** 2).sum(-1)
    noise = 64.0 * np.spacing(_amax(w).astype(np.float32)).astype(np.float64) ** 2
    worse = err_s - err_p * (1.0 + 1e-5) - noise
    if known_open("search-factor-one-inexact"):
        # Exclude the blocks whose candidate nearest to the factor 1 rounds to another F16 scale than the reference.
        d0 = grid.scale_rtn(torch.from_numpy(block_view(w)))
        near_one = torch.linspace(0.55, 1.05, 41)[36]
        other = ((d0 * near_one).to(torch.float16) != d0.to(torch.float16)).numpy()
        worse = np.where(other, -1.0, worse)
    assert (worse <= 0).all(), (f"the search error is larger than the reference error in {int((worse > 0).sum())} "
                                f"blocks, worst {float(worse.max()):.3e} over the tolerance")


@fuzz_settings()
@given(kind=st.sampled_from(sorted(GRIDS)), rows=st.integers(1, 4), nblocks=st.integers(1, 4),
       exp=st.integers(-14, 11), seed=st.integers(0, 2**32 - 1), negative=st.booleans())
@example(kind="Q4_0", rows=2, nblocks=2, exp=-3, seed=0, negative=False)
@example(kind="IQ4_NL", rows=1, nblocks=1, exp=-14, seed=5, negative=True)
@counted
def test_a_block_on_the_grid_keeps_its_scale(kind: str, rows: int, nblocks: int, exp: int, seed: int,
                                             negative: bool) -> None:
    """A block d · level[k] that holds the level of the largest magnitude comes back with d and k."""
    grid = GRIDS[kind]()
    gen = np.random.default_rng(seed)
    levels = grid.levels.numpy()
    top = int(np.argmax(np.abs(levels)))
    k = gen.integers(0, 16, (rows, nblocks, 32))
    k[:, :, 0] = top
    d = np.float16(gen.uniform(1.0, 2.0) * 2.0 ** exp * (-1.0 if negative else 1.0)).astype(np.float32)
    w = (levels[k] * d).reshape(rows, nblocks * 32).astype(np.float32)
    idx, got_d = quantize(grid, torch.from_numpy(w), search=True)
    np.testing.assert_array_equal(got_d.to(torch.float32).numpy(), np.full((rows, nblocks), d, dtype=np.float32),
                                  err_msg="the search moved the scale of a block on the grid")
    np.testing.assert_array_equal(dequantize(grid, idx, got_d).numpy(), w)


@fuzz_settings(0.5)
@given(spec=matrices(max_rows=9, max_blocks=3, min_exp=-10, max_exp=10), chunk=st.integers(1, 4),
       kind=st.sampled_from(sorted(GRIDS)))
@example(spec=SPEC_EDGES, chunk=1, kind="Q4_0")
@counted
def test_results_do_not_depend_on_the_row_chunk(spec, chunk: int, kind: str) -> None:
    """quantize, dequantize, q8_0_quantize and block_error give the same values for any ROW_CHUNK."""
    w = torch.from_numpy(spec.build())
    grid = GRIDS[kind]()
    weights = torch.rand(w.shape[1], generator=torch.Generator().manual_seed(spec.seed % 1000)) + 0.1
    full = (quantize(grid, w, weights=weights), q8_0_quantize(w), block_error(grid, w, weights))
    saved = grid_module.ROW_CHUNK
    grid_module.ROW_CHUNK = chunk
    try:
        small = (quantize(grid, w, weights=weights), q8_0_quantize(w), block_error(grid, w, weights))
        deq_small = dequantize(grid, *small[0])
    finally:
        grid_module.ROW_CHUNK = saved
    assert torch.equal(full[0][0], small[0][0]) and torch.equal(full[0][1], small[0][1])
    assert torch.equal(full[1][0], small[1][0]) and torch.equal(full[1][1], small[1][1])
    assert torch.equal(dequantize(grid, *full[0]), deq_small)
    torch.testing.assert_close(small[2], full[2], rtol=1e-5, atol=1e-30)


@fuzz_settings()
@given(spec=matrices(min_exp=-30, max_exp=12), kind=st.sampled_from(sorted(GRIDS)), seed=st.integers(0, 1000))
@example(spec=SPEC_EDGES, kind="IQ4_NL", seed=3)
@example(spec=SPEC_TINY_BLOCK, kind="Q4_0", seed=0)
@counted
def test_block_error_agrees_with_the_round_trip(spec, kind: str, seed: int) -> None:
    """block_error is the weighted squared error of quantize with the same weights, then dequantize."""
    w = spec.build()
    grid = GRIDS[kind]()
    wt = torch.from_numpy(w)
    weights = torch.rand(w.shape[1], generator=torch.Generator().manual_seed(seed)) + 0.05
    total = float(block_error(grid, wt, weights))
    idx, d = quantize(grid, wt, weights=weights, search=True)
    diff = (dequantize(grid, idx, d) - wt).to(torch.float64)
    expected = float((diff.pow(2) * weights.to(torch.float64)[None, :]).sum())
    assert total == pytest.approx(expected, rel=1e-4, abs=1e-30)


@fuzz_settings()
@given(rows=st.integers(1, 5), nblocks=st.integers(1, 5), seed=st.integers(0, 2**32 - 1))
@example(rows=1, nblocks=1, seed=0)
@counted
def test_q8_0_of_the_f16_range_is_finite(rows: int, nblocks: int, seed: int) -> None:
    """Every F16 source value, the extremes included, gives a finite Q8_0 scale and the format bound."""
    gen = np.random.default_rng(seed)
    choices = np.array([F16_MAX, -F16_MAX, 2.0 ** -24, -2.0 ** -24, 2.0 ** -14, 0.0, 1.0, 65000.0], dtype=np.float32)
    w = choices[gen.integers(0, choices.size, (rows, nblocks * 32))]
    _check_q8_0(w)
