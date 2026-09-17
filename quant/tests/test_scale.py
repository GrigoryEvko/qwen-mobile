"""The folds of the column scales keep the function: the norm, the KV group, the gated norm, the SwiGLU."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from quant import grid as grid_module
from quant.grids import Q4_0Grid
from quant.scale import (column_rms, group_mean, head_channel_share, kv_group_rows, kv_group_share,
                         permuted_moments, scaled_moments, search_column_scales)

EPS = 1e-6


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w)


def test_zero_centered_norm_fold() -> None:
    """(1 + w) / t − 1 in the norm and W · diag(t) in the consumer keep y."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(7, 16, generator=gen, dtype=torch.float64)
    w_norm = 0.2 * torch.randn(16, generator=gen, dtype=torch.float64)
    lin = torch.randn(5, 16, generator=gen, dtype=torch.float64)
    t = torch.exp(torch.randn(16, generator=gen, dtype=torch.float64))
    before = rmsnorm(x, w_norm) @ lin.T
    after = rmsnorm(x, (1.0 + w_norm) / t - 1.0) @ (lin * t[None, :]).T
    torch.testing.assert_close(after, before, atol=1e-12, rtol=0)


def test_kv_group_fold_of_gqa_attention() -> None:
    """The column scales of o_proj move into the v_proj rows of the KV group, through attention and the gate."""
    heads, kv_heads, dim, n = 4, 2, 8, 6
    group = heads // kv_heads
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(n, 16, generator=gen, dtype=torch.float64)
    wq = torch.randn(heads * dim, 16, generator=gen, dtype=torch.float64)
    wg = torch.randn(heads * dim, 16, generator=gen, dtype=torch.float64)
    wk = torch.randn(kv_heads * dim, 16, generator=gen, dtype=torch.float64)
    wv = torch.randn(kv_heads * dim, 16, generator=gen, dtype=torch.float64)
    wo = torch.randn(16, heads * dim, generator=gen, dtype=torch.float64)

    def attention(wv_: torch.Tensor, wo_: torch.Tensor) -> torch.Tensor:
        q = (x @ wq.T).view(n, heads, dim)
        k = (x @ wk.T).view(n, kv_heads, dim).repeat_interleave(group, dim=1)
        v = (x @ wv_.T).view(n, kv_heads, dim).repeat_interleave(group, dim=1)
        p = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / dim ** 0.5, -1)
        out = torch.einsum("hqk,khd->qhd", p, v).reshape(n, heads * dim)
        return (out * torch.sigmoid(x @ wg.T)) @ wo_.T

    share = kv_group_share(heads * dim, heads, kv_heads, dim, torch.device("cpu"))
    assert share.tolist() == [(j // dim // group) * dim + j % dim for j in range(heads * dim)]
    t = group_mean(torch.randn(heads * dim, generator=gen, dtype=torch.float64), share).exp()
    rows = kv_group_rows(t, heads, kv_heads, dim)
    assert rows.shape == (kv_heads * dim,)
    before = attention(wv, wo)
    after = attention(wv / rows[:, None], wo * t[None, :])
    torch.testing.assert_close(after, before, atol=1e-12, rtol=0)
    # A scale that differs inside a KV group has no exact fold: the test would fail without the share.
    bad = torch.exp(torch.randn(heads * dim, generator=gen, dtype=torch.float64))
    assert not torch.allclose(attention(wv / kv_group_rows(bad, heads, kv_heads, dim)[:, None], wo * bad[None, :]), before)


def test_gated_norm_fold_of_gdn_output() -> None:
    """The column scales of out_proj move into the shared weight of the gated norm of the value heads."""
    v_heads, v_dim, n = 4, 8, 5
    gen = torch.Generator().manual_seed(2)
    core = torch.randn(n, v_heads, v_dim, generator=gen, dtype=torch.float64)
    z = torch.randn(n, v_heads, v_dim, generator=gen, dtype=torch.float64)
    weight = torch.rand(v_dim, generator=gen, dtype=torch.float64) + 0.5
    wo = torch.randn(16, v_heads * v_dim, generator=gen, dtype=torch.float64)

    def gdn_out(weight_: torch.Tensor, wo_: torch.Tensor) -> torch.Tensor:
        normed = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + EPS) * weight_ * F.silu(z)
        return normed.reshape(n, -1) @ wo_.T

    share = head_channel_share(v_heads * v_dim, v_dim, torch.device("cpu"))
    assert share.tolist() == [j % v_dim for j in range(v_heads * v_dim)]
    t = group_mean(torch.randn(v_heads * v_dim, generator=gen, dtype=torch.float64), share).exp()
    torch.testing.assert_close(gdn_out(weight / t[:v_dim], wo * t[None, :]), gdn_out(weight, wo), atol=1e-12, rtol=0)


def test_swiglu_folds_and_permutation() -> None:
    """The down_proj column scales move into the up_proj rows, and the channel permutation keeps the MLP."""
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(6, 16, generator=gen, dtype=torch.float64)
    wg = torch.randn(24, 16, generator=gen, dtype=torch.float64)
    wu = torch.randn(24, 16, generator=gen, dtype=torch.float64)
    wd = torch.randn(16, 24, generator=gen, dtype=torch.float64)

    def mlp(wg_: torch.Tensor, wu_: torch.Tensor, wd_: torch.Tensor) -> torch.Tensor:
        return (F.silu(x @ wg_.T) * (x @ wu_.T)) @ wd_.T

    before = mlp(wg, wu, wd)
    t = torch.exp(torch.randn(24, generator=gen, dtype=torch.float64))
    torch.testing.assert_close(mlp(wg, wu / t[:, None], wd * t[None, :]), before, atol=1e-12, rtol=0)
    perm = torch.randperm(24, generator=gen)
    torch.testing.assert_close(mlp(wg[perm], wu[perm], wd[:, perm]), before, atol=1e-12, rtol=0)


def test_moments_follow_the_folds() -> None:
    """D⁻¹HD⁻¹ is the Hessian of x/t, and H[perm][:, perm] is the Hessian of x[perm]."""
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(50, 8, generator=gen)
    y = torch.randn(50, 8, generator=gen)
    h, g = x.T @ x / 50, x.T @ y / 50
    t = torch.exp(torch.randn(8, generator=gen))
    xs, ys = x / t, y / t
    hs, gs = scaled_moments(h, g, t)
    torch.testing.assert_close(hs, xs.T @ xs / 50, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gs, xs.T @ ys / 50, atol=1e-5, rtol=1e-5)
    perm = torch.randperm(8, generator=gen)
    hp, gp = permuted_moments(h, g, perm)
    torch.testing.assert_close(hp, x[:, perm].T @ x[:, perm] / 50, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gp, x[:, perm].T @ y[:, perm] / 50, atol=1e-5, rtol=1e-5)


def test_column_scale_search_is_chunk_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    """The search without full-size temporaries gives the scales of the direct formula, geometric mean 1."""
    gen = torch.Generator().manual_seed(5)
    ws = [torch.randn(20, 64, generator=gen) * 0.1, torch.randn(12, 64, generator=gen) * 0.05]
    h_diag = torch.rand(64, generator=gen) + 0.05
    direct_rms = torch.cat(ws, 0).pow(2).mean(0).clamp_min(1e-12).sqrt()
    torch.testing.assert_close(column_rms(ws), direct_rms, atol=1e-6, rtol=1e-6)
    grid = Q4_0Grid()
    t = search_column_scales(grid, ws, h_diag)
    assert abs(t.log().mean().item()) < 1e-5
    monkeypatch.setattr(grid_module, "ROW_CHUNK", 5)
    torch.testing.assert_close(search_column_scales(grid, ws, h_diag), t, atol=1e-6, rtol=1e-6)
