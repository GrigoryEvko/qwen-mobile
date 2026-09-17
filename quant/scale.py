"""Column scales that fold into the graph without an online operation.

A linear that reads x·W ᵀ can read (x/t)·(W·diag(t))ᵀ when its producer
supplies x/t. The producers here are exact: the zero-centered RMSNorm
weight, the rows of v_proj (attention is linear in V, the output gate is
per channel), the rows of up_proj (the SwiGLU product is linear in up),
and the shared weight of the gated norm of the GDN block. The scale gives
the weights of a column with strong activations more of the block range,
and the block-32 scales of Q4_0 pay for it. This is the diagonal part of
WUSH and the input-side scale of SINQ, chosen on the true block error.
"""

from __future__ import annotations

import torch

from .grid import q4_0_dequantize, q4_0_quantize

EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def weighted_block_error(w: torch.Tensor, col_weights: torch.Tensor) -> torch.Tensor:
    """The diagonal-Hessian error of the block-RTN quantization of ``w`` with the scale search."""
    q, d = q4_0_quantize(w, weights=col_weights, search=True)
    return ((q4_0_dequantize(q, d) - w).pow(2) * col_weights[None, :]).sum()


def search_column_scales(ws: list[torch.Tensor], h_diag: torch.Tensor, share: int | None = None,
                         exponents: tuple[float, ...] = EXPONENTS) -> torch.Tensor:
    """Column scales t for matrices that read the same input, geometric mean 1.

    Candidates are t_j = a_j^α / r_j^γ, with a the activation rms of column j
    (from the Hessian diagonal), r the weight rms of column j over all the
    matrices, and α, γ from ``exponents``. The winner minimizes the weighted
    block-RTN error summed over the matrices. With ``share``, the scale is
    the same for the columns j with the same j mod share (one value per
    within-head channel). Complexity is O(|exponents|² · Σ rows · cols · 40).
    """
    cols = h_diag.shape[0]
    a = h_diag.to(torch.float32).clamp_min(1e-12).sqrt()
    r = torch.cat([w.to(torch.float32) for w in ws], 0).pow(2).mean(0).clamp_min(1e-12).sqrt()
    log_a, log_r = a.log(), r.log()
    best_t = torch.ones_like(a)
    best_err = None
    for alpha in exponents:
        for gamma in exponents:
            log_t = alpha * log_a - gamma * log_r
            if share is not None:
                log_t = log_t.view(-1, share).mean(0).repeat(cols // share)
            t = (log_t - log_t.mean()).exp()
            err = sum(weighted_block_error(w.to(torch.float32) * t[None, :], h_diag / t.pow(2)) for w in ws)
            if best_err is None or err < best_err:
                best_err, best_t = err, t
    return best_t


def scaled_moments(h: torch.Tensor, g: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The moments of x/t from the moments of x: D⁻¹HD⁻¹ and D⁻¹GD⁻¹."""
    inv = (1.0 / t).to(h.dtype)
    return h * inv[:, None] * inv[None, :], g * inv[:, None] * inv[None, :]


def permuted_moments(h: torch.Tensor, g: torch.Tensor, perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The moments of x[perm]."""
    return h[perm][:, perm], g[perm][:, perm]
