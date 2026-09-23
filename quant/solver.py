"""Rounding onto a block grid, with blocks of 32 in input order.

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
compensated weights, by the weighted scale search on the grid.
Complexity is O(rows · cols²) per matrix. The rows are independent given
the Hessian, thus the solver works in row chunks: the head of 248320 rows
runs on a device that cannot hold it in float32.
"""

from __future__ import annotations

import torch

from .grid import BLOCK, ROW_CHUNK, scale_search
from .grids import Grid, divisor


def _damped(h: torch.Tensor, damp: float, relative_to: str) -> torch.Tensor:
    """H plus λI. λ is ``damp`` times the mean diagonal, or times the largest eigenvalue."""
    if relative_to == "sigma":
        lam = damp * torch.linalg.eigvalsh(h)[-1]
    else:
        lam = damp * torch.mean(torch.diag(h))
    return h + lam * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)


def solve_grid(w: torch.Tensor, hessian: torch.Tensor, grid: Grid, cross: torch.Tensor | None = None,
               damp: float = 0.01, refit_damp: float = 1e-6, chunk: int = 128, search: bool = True,
               rows_per_chunk: int = ROW_CHUNK) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Return (indices int8 [rows, cols], d float16 [rows, cols // 32], weighted error, refit change).

    ``w`` is [rows, cols] on any device. ``hessian`` and ``cross`` are
    [cols, cols] float32 on the compute device, and the results come back
    on that device. The work runs in chunks of ``rows_per_chunk`` rows,
    thus the device holds one chunk of ``w`` at a time. The weighted error
    is the mean over the rows of the squared residual of a row. The refit
    change is ‖W' − W‖ / ‖W‖ over all the rows, zero without ``cross``.
    """
    device = hessian.device
    rows, cols = w.shape
    h = hessian.to(torch.float32).clone()

    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0

    # Qronos: W' = W · Aᵀ with A = H̃⁻¹ · G. The refit uses a much lighter damping than the
    # rounding, thus the least-squares solution stays close to the exact one.
    a_t = None
    if cross is not None:
        a_t = torch.linalg.solve(_damped(h, refit_damp, "sigma"), cross.to(device, torch.float32)).T.contiguous()

    hinv = torch.linalg.cholesky(_damped(h, damp, "mean"))
    hinv = torch.cholesky_inverse(hinv)
    hinv = torch.linalg.cholesky(hinv, upper=True)
    diag_h = torch.diag(hessian).to(device, torch.float32)

    idx_all = torch.empty(rows, cols, dtype=torch.int8, device=device)
    d_all = torch.empty(rows, cols // BLOCK, dtype=torch.float16, device=device)
    err_sum = torch.zeros((), dtype=torch.float32, device=device)
    delta_sq = torch.zeros((), dtype=torch.float32, device=device)
    ref_sq = torch.zeros((), dtype=torch.float32, device=device)
    for r in range(0, rows, rows_per_chunk):
        w_r = w[r:r + rows_per_chunk].to(device, torch.float32).clone()
        w_r[:, dead] = 0.0
        if a_t is not None:
            refit = w_r @ a_t
            delta_sq += (refit - w_r).pow(2).sum()
            ref_sq += w_r.pow(2).sum()
            w_r = refit
        idx_r, d_r, err_r = _solve_rows(w_r, hinv, diag_h, grid, chunk, search)
        idx_all[r:r + rows_per_chunk] = idx_r
        d_all[r:r + rows_per_chunk] = d_r
        err_sum += err_r
    refit_change = (delta_sq.sqrt() / ref_sq.sqrt().clamp_min(1e-12)).item() if a_t is not None else 0.0
    return idx_all, d_all, err_sum.item() / rows, refit_change


def _solve_rows(w: torch.Tensor, hinv: torch.Tensor, diag_h: torch.Tensor, grid: Grid, chunk: int,
                search: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Cholesky rounding of the rows of ``w`` (float32 on the device, changed in place).

    Returns (indices int8, d float16, the sum of the squared residuals).
    """
    rows, cols = w.shape
    idx_all = torch.zeros(rows, cols, dtype=torch.int8, device=w.device)
    d_all = torch.zeros(rows, cols // BLOCK, dtype=torch.float32, device=w.device)
    total_err = torch.zeros(rows, device=w.device)

    for i1 in range(0, cols, chunk):
        i2 = min(i1 + chunk, cols)
        count = i2 - i1
        w1 = w[:, i1:i2].clone()
        idx1 = torch.zeros(rows, count, dtype=torch.long, device=w.device)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        d_cur = None
        for i in range(count):
            col = i1 + i
            if col % BLOCK == 0:
                blk = w1[:, i:i + BLOCK].reshape(rows, 1, BLOCK)
                if search:
                    d_cur = scale_search(grid, blk, diag_h[col:col + BLOCK])
                else:
                    d_cur = grid.scale_rtn(blk)
                d_cur = d_cur.to(torch.float16).to(torch.float32).reshape(rows)
                d_all[:, col // BLOCK] = d_cur
            assert d_cur is not None
            wc = w1[:, i]
            dd = hinv1[i, i]
            idx = grid.round(wc / divisor(d_cur))
            deq = grid.value(idx) * d_cur
            idx1[:, i] = idx
            resid = wc - deq
            total_err += resid.pow(2)
            err = resid / dd
            w1[:, i:] -= err[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = err
        idx_all[:, i1:i2] = idx1.to(torch.int8)
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return idx_all, d_all.to(torch.float16), total_err.sum()
