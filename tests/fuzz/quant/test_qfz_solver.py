"""Fuzz the solvers of the calibration: the column rounding, the block optimization, the column scales, the trellis.

The properties:

- With a diagonal Hessian the Cholesky rounding moves no error to the
  next columns, thus solve_grid gives the indices and the scales of
  quantize with the diagonal as the column weights.
- solve_grid gives the same result for each row chunk.
- A dead column (zero on the diagonal of the Hessian) decodes to the
  level nearest to zero.
- The STE module exports the weight of its forward pass when the scales
  are F16 values and the levels are fixed.
- The rank-r correction lowers the Hessian-weighted error as r grows, and
  the full rank removes the error.
- The column scales are positive, finite, with the geometric mean one,
  and the columns of one group take one scale.
- The Viterbi path of the trellis costs no more than a random path, and
  its decode is the value that the encoder costed.
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from qfz_hyp import counted, fuzz_settings
from qfz_strategies import MatrixSpec, matrices
from quant.blockopt import STELinear, weighted_low_rank
from quant.grid import quantize
from quant.grids import IQ4NLGrid, Q4_0Grid
from quant.scale import group_mean, kv_group_rows, kv_group_share, search_column_scales
from quant.solver import solve_grid
from quant.trellis import Trellis

GRIDS = {"Q4_0": Q4_0Grid, "IQ4_NL": IQ4NLGrid}
# The explicit example of the matrix tests: the seed of the mode "test".
SPEC = MatrixSpec(3, 2, (("gauss", "spike"), ("opposite", "zero"), ("ties", "sparse")), ((0, -2), (3, 0), (-4, 1)),
                  11, True)


@fuzz_settings()
@given(spec=matrices(max_rows=5, max_blocks=4, min_exp=-8, max_exp=8), kind=st.sampled_from(sorted(GRIDS)),
       seed=st.integers(0, 2**31 - 1), rows_per_chunk=st.integers(1, 6))
@example(spec=SPEC, kind="Q4_0", seed=0, rows_per_chunk=1)
@counted
def test_diagonal_hessian_gives_the_weighted_round_to_nearest(spec, kind: str, seed: int, rows_per_chunk: int) -> None:
    """GPTQ on a diagonal Hessian is the weighted scale search and the plain rounding, for each row chunk."""
    w = torch.from_numpy(spec.build())
    grid = GRIDS[kind]()
    gen = torch.Generator().manual_seed(seed)
    diag = torch.rand(w.shape[1], generator=gen) * 4.0 + 0.05
    idx, d, err, change = solve_grid(w, torch.diag(diag), grid, rows_per_chunk=rows_per_chunk)
    idx_r, d_r = quantize(grid, w, weights=diag, search=True)
    assert torch.equal(d, d_r), "the scales differ from the weighted search"
    assert torch.equal(idx, idx_r), "the indices differ from the plain rounding"
    assert change == 0.0 and np.isfinite(err) and err >= 0.0


@fuzz_settings()
@given(spec=matrices(max_rows=5, max_blocks=3, min_exp=-6, max_exp=6), kind=st.sampled_from(sorted(GRIDS)),
       seed=st.integers(0, 2**31 - 1), qronos=st.booleans(), chunk=st.integers(1, 5))
@example(spec=SPEC, kind="IQ4_NL", seed=1, qronos=True, chunk=2)
@counted
def test_solve_grid_is_row_chunk_invariant(spec, kind: str, seed: int, qronos: bool, chunk: int) -> None:
    """Rows are independent given the Hessian, thus each row chunk gives the same result, up to the BLAS rounding.

    A matrix product of 1 row and of 2 rows can use different BLAS kernels,
    thus a value near a rounding boundary can move by one level, and a scale
    by some F16 steps (seen with the Qronos refit on 2026-09-23). A slicing
    defect moves most values. The test thus permits at most 1/16 of the
    blocks with a different scale and 1/16 of the values with a different
    level, each by a small step.
    """
    w = torch.from_numpy(spec.build())
    grid = GRIDS[kind]()
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(4 * w.shape[1], w.shape[1], generator=gen)
    h = x.T @ x / x.shape[0]
    g = h + 0.05 * torch.randn(h.shape, generator=gen) if qronos else None
    full = solve_grid(w, h, grid, g)
    part = solve_grid(w, h, grid, g, rows_per_chunk=chunk)
    assert int(full[0].min()) >= 0 and int(full[0].max()) <= 15
    assert torch.isfinite(full[1].to(torch.float32)).all()
    d_full, d_part = full[1].to(torch.float32), part[1].to(torch.float32)
    d_diff = (d_full - d_part).abs() > 0
    assert int(d_diff.sum()) <= max(1, d_full.numel() // 16), f"{int(d_diff.sum())} scales differ"
    assert bool(((d_full - d_part).abs() <= 2.0 ** -6 * d_full.abs()).all()), "a scale moved by more than 2^-6"
    i_diff = full[0] != part[0]
    assert int(i_diff.sum()) <= max(1, full[0].numel() // 16), f"{int(i_diff.sum())} levels differ"
    assert int((full[0].to(torch.int32) - part[0].to(torch.int32)).abs().max()) <= 1, "a level moved by more than one"


@fuzz_settings()
@given(spec=matrices(max_rows=4, max_blocks=3, min_exp=-4, max_exp=4), kind=st.sampled_from(sorted(GRIDS)),
       seed=st.integers(0, 2**31 - 1))
@example(spec=SPEC, kind="Q4_0", seed=4)
@counted
def test_dead_columns_decode_to_the_level_nearest_zero(spec, kind: str, seed: int) -> None:
    """A column with a zero on the Hessian diagonal holds the level nearest to zero in each row."""
    w = torch.from_numpy(spec.build())
    grid = GRIDS[kind]()
    gen = torch.Generator().manual_seed(seed)
    diag = torch.rand(w.shape[1], generator=gen) + 0.1
    dead = torch.rand(w.shape[1], generator=gen) < 0.3
    diag[dead] = 0.0
    idx, d, _, _ = solve_grid(w, torch.diag(diag), grid)
    nearest = int(torch.argmin(grid.levels.abs()))
    assert (idx[:, dead] == nearest).all(), "a dead column holds a level that is not the nearest to zero"


@fuzz_settings()
@given(spec=matrices(max_rows=5, max_blocks=3, min_exp=-6, max_exp=6), kind=st.sampled_from(sorted(GRIDS)),
       seed=st.integers(0, 2**31 - 1))
@example(spec=SPEC, kind="IQ4_NL", seed=0)
@counted
def test_ste_exports_the_weight_of_its_forward(spec, kind: str, seed: int) -> None:
    """With F16 scales and fixed levels, finalize gives the quantized weight of the forward pass.

    The module stores log |d|, thus exp(log |d|) can differ from d by one
    float32 step. The straight-through sum latent + (w_q − latent) also
    rounds at the step of the latent value. The tolerance accepts these two
    steps. A latent value within 1e-5 of a rounding midpoint can also go to
    the next level, because of the step of the scale (seen on 2026-09-23 in
    1 of 20451 examples). Any other different index fails the test.
    """
    w = torch.from_numpy(spec.build())
    grid = GRIDS[kind]()
    _, d = quantize(grid, w, search=True)
    gen = torch.Generator().manual_seed(seed)
    latent = w + 0.3 * torch.randn(w.shape, generator=gen) * w.abs().amax().clamp_min(1e-30)
    ste = STELinear(latent, d.to(torch.float32), grid, learn_levels=False)
    with torch.no_grad():
        forward = ste.quantized_weight()
    final = ste.finalize().dequantized()
    step = torch.maximum(latent.abs(), final.abs()).numpy()
    limit = 4.0 * np.spacing(step) + 1e-6 * final.abs().numpy()
    gap = (final - forward).abs().numpy()
    t = (latent.reshape(latent.shape[0], -1, 32) / d.to(torch.float32)[..., None]).reshape(latent.shape).numpy()
    near_mid = (np.abs(t[..., None] - grid.mid.numpy()) <= 1e-5 * np.maximum(1.0, np.abs(t))[..., None]).any(-1)
    bad = (gap > limit) & ~near_mid
    over = float((gap - limit)[bad].max()) if bad.any() else 0.0
    assert not bad.any(), f"the export differs from the forward by {over:.3e} over the limit at {int(bad.sum())} values"


@fuzz_settings(0.5)
@given(rows=st.integers(2, 12), cols=st.integers(2, 12), seed=st.integers(0, 2**31 - 1))
@example(rows=6, cols=4, seed=0)
@counted
def test_low_rank_correction_improves_with_rank(rows: int, cols: int, seed: int) -> None:
    """The Hessian-weighted error falls as the rank grows, and the full rank leaves only the damping error."""
    gen = torch.Generator().manual_seed(seed)
    e = torch.randn(rows, cols, generator=gen, dtype=torch.float64).to(torch.float32)
    x = torch.randn(3 * cols, cols, generator=gen)
    h = x.T @ x / x.shape[0]
    damped = h.to(torch.float64) + 0.01 * torch.diag(h).mean() * torch.eye(cols, dtype=torch.float64)
    low = torch.linalg.cholesky(damped)
    errs = []
    for r in range(0, min(rows, cols) + 1):
        if r == 0:
            errs.append(float((e.to(torch.float64) @ low).norm()))
            continue
        a, b = weighted_low_rank(e, h, r)
        assert a.shape == (r, cols) and b.shape == (rows, r)
        errs.append(float(((e - b @ a).to(torch.float64) @ low).norm()))
    for before, after in zip(errs, errs[1:], strict=False):
        assert after <= before * (1 + 1e-4) + 1e-6, f"the weighted error rose with the rank: {errs}"
    assert errs[-1] <= 1e-3 * max(errs[0], 1e-12), f"the full rank leaves the error {errs[-1]:.3e} of {errs[0]:.3e}"


@fuzz_settings(0.5)
@given(n_mats=st.integers(1, 3), rows=st.integers(1, 4), nblocks=st.integers(1, 4), seed=st.integers(0, 2**31 - 1),
       grouped=st.booleans(), kind=st.sampled_from(sorted(GRIDS)))
@example(n_mats=2, rows=2, nblocks=2, seed=0, grouped=True, kind="Q4_0")
@counted
def test_column_scales_are_positive_with_geometric_mean_one(n_mats: int, rows: int, nblocks: int, seed: int,
                                                           grouped: bool, kind: str) -> None:
    """search_column_scales gives positive finite scales, geometric mean one, one value per group."""
    gen = torch.Generator().manual_seed(seed)
    cols = nblocks * 32
    ws = [torch.randn(rows, cols, generator=gen) * torch.exp(torch.randn(cols, generator=gen)) for _ in range(n_mats)]
    h_diag = torch.exp(2.0 * torch.randn(cols, generator=gen))
    h_diag[torch.rand(cols, generator=gen) < 0.1] = 0.0
    share = torch.arange(cols) % 16 if grouped else None
    t = search_column_scales(GRIDS[kind](), ws, h_diag, share, exponents=(0.0, 0.5, 1.0))
    assert torch.isfinite(t).all() and (t > 0).all()
    assert abs(float(t.log().mean())) < 1e-4
    if share is not None:
        torch.testing.assert_close(group_mean(t.log(), share).exp(), t, rtol=1e-5, atol=0)


@fuzz_settings()
@given(kv_heads=st.integers(1, 4), group=st.integers(1, 4), dim=st.integers(1, 16), seed=st.integers(0, 2**31 - 1))
@example(kv_heads=2, group=4, dim=8, seed=0)
@counted
def test_kv_group_rows_take_the_scale_of_their_group(kv_heads: int, group: int, dim: int, seed: int) -> None:
    """The v_proj row divisor of KV head h and channel c is the shared o_proj scale of that group and channel."""
    heads = kv_heads * group
    share = kv_group_share(heads * dim, heads, kv_heads, dim, torch.device("cpu"))
    t = group_mean(torch.rand(heads * dim, generator=torch.Generator().manual_seed(seed)) + 0.5, share)
    rows = kv_group_rows(t, heads, kv_heads, dim)
    assert rows.shape == (kv_heads * dim,)
    for j in range(heads * dim):
        assert float(rows[int(share[j])]) == float(t[j])


@fuzz_settings(0.3)
@given(n=st.integers(1, 3), t_len=st.integers(1, 12), seed=st.integers(0, 2**31 - 1), weighted=st.booleans())
@example(n=2, t_len=8, seed=0, weighted=True)
@counted
def test_trellis_viterbi_path_is_optimal(n: int, t_len: int, seed: int, weighted: bool) -> None:
    """The Viterbi path costs no more than random paths, and its decode is what the encoder costed."""
    gen = torch.Generator().manual_seed(seed)
    trellis = Trellis.gaussian(8, k=4, seed=seed % 97)
    x = torch.randn(n, t_len, generator=gen) * 40.0
    wts = torch.rand(n, t_len, generator=gen) + 0.1 if weighted else None
    codes, heads = trellis.encode(x, wts)
    assert int(codes.min()) >= 0 and int(codes.max()) < 16

    def cost(c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        err = (trellis.decode(c, h) - x).pow(2)
        return (err if wts is None else err * wts).sum(-1)

    best = cost(codes, heads)
    for _ in range(20):
        rc = torch.randint(0, 16, codes.shape, generator=gen, dtype=torch.int8)
        rh = torch.randint(0, 16, heads.shape, generator=gen, dtype=torch.int32)
        assert (best <= cost(rc, rh) * (1 + 1e-5) + 1e-3).all(), "a random path is cheaper than the Viterbi path"
