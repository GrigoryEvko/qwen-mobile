"""The straight-through quantizer starts from the solved rounding and keeps the scale signs."""

from __future__ import annotations

import torch
from torch import nn

from quant.blockopt import STELinear, Target, make_ste
from quant.grid import dequantize
from quant.grids import CodebookGrid, IQ4NLGrid, Q4_0Grid
from quant.solver import solve_grid


def test_ste_starts_from_the_solved_rounding() -> None:
    """With the solved scales the STE value is the solver output, and finalize returns the same indices."""
    torch.manual_seed(0)
    rows, cols = 16, 128
    w = torch.randn(rows, cols) * 0.1
    x = torch.randn(400, cols) * (torch.rand(cols) * 3 + 0.1)
    h = x.T @ x / 400
    for grid in (Q4_0Grid(), IQ4NLGrid()):
        idx, d, _, _ = solve_grid(w, h, grid, None, damp=0.01)
        lin = nn.Linear(cols, rows, bias=False)
        lin.weight.data.copy_(dequantize(grid, idx, d))
        ste = make_ste(lin, Target(grid, h, w, d), rank=0)
        torch.testing.assert_close(ste.quantized_weight().detach(), lin.weight.data, atol=1e-6, rtol=1e-6)
        solved = ste.finalize()
        assert torch.equal(solved.idx, idx) and torch.equal(solved.d, d)
        assert (ste.sign == torch.where(d < 0, -1.0, 1.0)).all()
        assert (ste.scale().sign() == d.float().sign()).all(), "a negative Q4_0 scale keeps its sign"


def test_ste_search_path_reproduces_an_on_grid_weight() -> None:
    """Without the solved scales the search runs, and an on-grid weight still quantizes to itself."""
    torch.manual_seed(1)
    grid = Q4_0Grid()
    w = torch.randn(8, 64)
    lin = nn.Linear(64, 8, bias=False)
    from quant.grid import quantize

    idx, d = quantize(grid, w, search=True)
    lin.weight.data.copy_(dequantize(grid, idx, d))
    ste = make_ste(lin, Target(grid, torch.eye(64), w), rank=0)
    assert torch.equal(ste.finalize().idx, idx)


def test_gradients_reach_the_latent_weight_and_the_scales() -> None:
    torch.manual_seed(2)
    grid = Q4_0Grid()
    w = torch.randn(4, 32)
    ste = STELinear(w, grid.scale_rtn(w.view(4, 1, 32)), grid, learn_levels=False)
    y = ste(torch.randn(3, 32)).pow(2).sum()
    y.backward()
    assert ste.weight.grad is not None and ste.weight.grad.abs().sum() > 0
    assert ste.log_d.grad is not None and ste.log_d.grad.abs().sum() > 0


def test_learned_levels_round_and_look_up_in_the_same_order() -> None:
    """A codebook whose levels crossed still gives d · level of the nearest level."""
    torch.manual_seed(3)
    levels = torch.tensor([-127.0, -90, -60, -35, -20, -10, -4, -1, 1, 4, 10, 20, 35, 60, 90, 127])
    grid = CodebookGrid(levels)
    w = torch.randn(2, 32) * 30
    ste = STELinear(w, torch.ones(2, 1), grid, learn_levels=True)
    with torch.no_grad():
        ste.levels[8], ste.levels[9] = 4.0, 1.0  # the two smallest positive levels swap places
    got = ste.quantized_weight().detach()
    sorted_levels = ste.levels.detach().sort().values
    nearest = sorted_levels[(w[..., None] - sorted_levels).abs().argmin(-1)]
    torch.testing.assert_close(got, nearest)
