"""GPTQ into the Q4_0 grid, with blocks of 32 in input order.

The solver rounds the columns of W in order and diffuses the rounding error
of each column into the columns that are not rounded yet, with the inverse
Hessian of the layer input (Frantar et al.). Each block of 32 gets its scale
at the moment the solver reaches it, from the compensated weights, by the
weighted scale search of the grid. Complexity is O(rows · cols²) per matrix.
"""

from __future__ import annotations

import torch

from .grid import BLOCK, q4_0_round, q4_0_scale_search


def gptq_q4_0(w: torch.Tensor, hessian: torch.Tensor, damp: float = 0.01, chunk: int = 128,
              scale_search: bool = True):
    """Return (q int8 [rows, cols], d float16 [rows, cols // 32], weighted error).

    ``w`` is [rows, cols] float32 on the device. ``hessian`` is [cols, cols]
    float32, the mean of xᵀx over the calibration tokens.
    """
    w = w.to(torch.float32).clone()
    rows, cols = w.shape
    h = hessian.to(torch.float32).clone()

    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h += damp * torch.mean(torch.diag(h)) * torch.eye(cols, device=w.device)
    hinv = torch.linalg.cholesky(h)
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
                wgt = diag_h[col:col + BLOCK] if scale_search else None
                d_cur = (q4_0_scale_search(blk, wgt) if scale_search else
                         q4_0_scale_search(blk, None, n_candidates=1, span=0.0))
                d_cur = d_cur.to(torch.float16).to(torch.float32).reshape(rows)
                d_all[:, col // BLOCK] = d_cur
            assert d_cur is not None
            wc = w1[:, i]
            dd = hinv1[i, i]
            q = q4_0_round(wc[:, None], d_cur[:, None]).reshape(rows)
            deq = q * d_cur
            q1[:, i] = q
            err = (wc - deq) / dd
            w1[:, i:] -= err[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = err
            total_err += (wc - deq).pow(2)
        q_all[:, i1:i2] = q1.to(torch.int8)
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]

    return q_all, d_all.to(torch.float16), total_err.mean().item()
