"""Column scales that fold into the graph without an online operation.

A linear that reads x·W ᵀ can read (x/t)·(W·diag(t))ᵀ when its producer
supplies x/t. The producers here are exact: the zero-centered RMSNorm
weight, the rows of v_proj (attention is linear in V, the output gate is
per channel), the rows of up_proj (the SwiGLU product is linear in up),
and the shared weight of the gated norm of the GDN block. The scale gives
the weights of a column with strong activations more of the block range,
and the block-32 scales pay for it. This is the diagonal part of WUSH and
the input-side scale of SINQ, chosen on the true block error.
"""

from __future__ import annotations

import torch

from .grid import ROW_CHUNK, block_error
from .grids import Grid

EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def column_rms(ws: list[torch.Tensor]) -> torch.Tensor:
    """The root mean square of each column over all the rows of all the matrices, in row chunks."""
    total: torch.Tensor | None = None
    rows = 0
    for w in ws:
        for r in range(0, w.shape[0], ROW_CHUNK):
            part = w[r:r + ROW_CHUNK].to(torch.float32).pow(2).sum(0)
            total = part if total is None else total + part
        rows += w.shape[0]
    assert total is not None, "no matrices"
    return (total / rows).clamp_min(1e-12).sqrt()


def search_column_scales(grid: Grid, ws: list[torch.Tensor], h_diag: torch.Tensor,
                         share: torch.Tensor | None = None,
                         exponents: tuple[float, ...] = EXPONENTS) -> torch.Tensor:
    """Column scales t for matrices that read the same input, geometric mean 1.

    Candidates are t_j = a_j^α / r_j^γ, with a the activation rms of column j
    (from the Hessian diagonal), r the weight rms of column j over all the
    matrices, and α, γ from ``exponents``. The winner minimizes the weighted
    block-RTN error summed over the matrices. ``share`` [cols] gives a group
    id per column, and the columns of a group take one scale (one value per
    within-head channel). Complexity is O(|exponents|² · Σ rows · cols · 41).
    """
    a = h_diag.to(torch.float32).clamp_min(1e-12).sqrt()
    log_a, log_r = a.log(), column_rms(ws).log()
    best_t = torch.ones_like(a)
    best_err = None
    for alpha in exponents:
        for gamma in exponents:
            log_t = alpha * log_a - gamma * log_r
            if share is not None:
                log_t = group_mean(log_t, share)
            t = (log_t - log_t.mean()).exp()
            err = sum(block_error(grid, w, h_diag / t.pow(2), col_scale=t) for w in ws)
            if best_err is None or err < best_err:
                best_err, best_t = err, t
    return best_t


def group_mean(x: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """Replace each value by the mean of its group."""
    n = int(group.max().item()) + 1
    total = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, group, x)
    count = torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, group, torch.ones_like(x))
    return (total / count.clamp_min(1))[group]


def kv_group_share(cols: int, heads: int, kv_heads: int, dim: int, device: torch.device) -> torch.Tensor:
    """The group id of each o_proj column: the KV head that supplies it and the channel in the head.

    The attention heads h·group … h·group + group − 1 read KV head h, thus
    the columns of those heads must take one scale per channel, because
    one v_proj row supplies them all.
    """
    group = heads // kv_heads
    j = torch.arange(cols, device=device)
    return (j // dim // group) * dim + j % dim


def kv_group_rows(t: torch.Tensor, heads: int, kv_heads: int, dim: int) -> torch.Tensor:
    """The v_proj row divisor [kv_heads · dim] of the o_proj column scales t [heads · dim], one head per group."""
    group = heads // kv_heads
    return t.view(heads, dim)[::group].reshape(-1)


def head_channel_share(cols: int, v_dim: int, device: torch.device) -> torch.Tensor:
    """The group id of each GDN out_proj column: the channel in the value head, shared by all the heads."""
    return torch.arange(cols, device=device) % v_dim


def scaled_moments(h: torch.Tensor, g: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The moments of x/t from the moments of x: D⁻¹HD⁻¹ and D⁻¹GD⁻¹."""
    inv = (1.0 / t).to(h.dtype)
    return h * inv[:, None] * inv[None, :], g * inv[:, None] * inv[None, :]


def permuted_moments(h: torch.Tensor, g: torch.Tensor, perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The moments of x[perm]."""
    return h[perm][:, perm], g[perm][:, perm]
