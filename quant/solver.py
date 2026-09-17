"""Rounding into the Q4_0 grid, with blocks of 32 in input order.

Two moments of the layer input drive the solver. ``hessian`` is
H̃ = mean(x̃ᵀx̃) over the calibration tokens, with x̃ the input from the
quantized flow. ``cross`` is G = mean(x̃ᵀx), with x the same tokens from the
FP flow of the reference model.

With ``cross`` the solver is Qronos (Zhang et al., ICLR 2026) in its
efficient form: a least-squares refit W' = W·Gᵀ·H̃⁻¹ aligns the layer to the
FP-flow output of the original weights, then the Cholesky rounding on H̃
(Frantar et al.) diffuses each rounding error into the columns that are
not rounded yet. Without ``cross`` the solver is GPTQ.

Each block of 32 gets its scale when the solver reaches it, from the
compensated weights, by the weighted scale search of the grid.
Complexity is O(rows · cols²) per matrix.
"""

from __future__ import annotations

import torch

from .grid import BLOCK, Q4_MAX, Q4_MIN, q4_0_scale_search


def _damped(h: torch.Tensor, damp: float, relative_to: str) -> torch.Tensor:
    """H plus λI. λ is ``damp`` times the mean diagonal, or times the largest eigenvalue."""
    if relative_to == "sigma":
        lam = damp * torch.linalg.eigvalsh(h)[-1]
    else:
        lam = damp * torch.mean(torch.diag(h))
    return h + lam * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)


def gptq_q4_0(w: torch.Tensor, hessian: torch.Tensor, cross: torch.Tensor | None = None,
              damp: float = 0.01, refit_damp: float = 1e-6, chunk: int = 128,
              scale_search: bool = True):
    """Return (q int8 [rows, cols], d float16 [rows, cols // 32], weighted error, refit change).

    ``w`` is [rows, cols] float32 on the device. ``hessian`` and ``cross``
    are [cols, cols] float32. The refit change is ‖W' − W‖ / ‖W‖, zero
    without ``cross``.
    """
    w = w.to(torch.float32).clone()
    rows, cols = w.shape
    h = hessian.to(torch.float32).clone()

    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    refit_change = 0.0
    if cross is not None:
        # Qronos: the refit uses a much lighter damping than the rounding, thus
        # the least-squares solution stays close to the exact one.
        w_ref = w
        w = torch.linalg.solve(_damped(h, refit_damp, "sigma"), cross.to(torch.float32) @ w_ref.T).T.contiguous()
        refit_change = ((w - w_ref).norm() / w_ref.norm().clamp_min(1e-12)).item()

    hinv = torch.linalg.cholesky(_damped(h, damp, "mean"))
    hinv = torch.cholesky_inverse(hinv)
    hinv = torch.linalg.cholesky(hinv, upper=True)
    diag_h = torch.diag(hessian).to(torch.float32)

    q_all = torch.zeros(rows, cols, dtype=torch.int8, device=w.device)
    d_all = torch.zeros(rows, cols // BLOCK, dtype=torch.float32, device=w.device)
    total_err = torch.zeros(rows, device=w.device)

    for i1 in range(0, cols, chunk):
        i2 = min(i1 + chunk, cols)
        count = i2 - i1
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        d_cur = None
        for i in range(count):
            col = i1 + i
            if col % BLOCK == 0:
                blk = w1[:, i:i + BLOCK].reshape(rows, 1, BLOCK)
                if scale_search:
                    d_cur = q4_0_scale_search(blk, diag_h[col:col + BLOCK])
                else:
                    d_cur = q4_0_scale_search(blk, None, n_candidates=1, lo=1.0, hi=1.0)
                d_cur = d_cur.to(torch.float16).to(torch.float32).reshape(rows)
                d_all[:, col // BLOCK] = d_cur
            assert d_cur is not None
            wc = w1[:, i]
            dd = hinv1[i, i]
            q = torch.clamp(torch.round(wc / d_cur), Q4_MIN, Q4_MAX)
            deq = q * d_cur
            q1[:, i] = q
            resid = wc - deq
            total_err += resid.pow(2)
            err = resid / dd
            w1[:, i:] -= err[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = err
        q_all[:, i1:i2] = q1.to(torch.int8)
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return q_all, d_all.to(torch.float16), total_err.mean().item(), refit_change
