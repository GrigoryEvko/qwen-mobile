"""The Q4_0 and Q8_0 grids of ggml, with scale search and byte packing.

Q4_0: blocks of 32 along the input dimension, one F16 scale d per block,
levels q in [-8, 7], value d · q. ggml stores the nibble q + 8, the low
nibble of byte j holds element j and the high nibble holds element j + 16.

Q8_0: blocks of 32, one F16 scale, levels in [-127, 127].
"""

from __future__ import annotations

import numpy as np
import torch

BLOCK = 32
Q4_MIN, Q4_MAX = -8, 7


def _blocks(w: torch.Tensor) -> torch.Tensor:
    """View [rows, cols] as [rows, cols // 32, 32]."""
    rows, cols = w.shape
    if cols % BLOCK:
        raise ValueError(f"columns {cols} are not a multiple of {BLOCK}")
    return w.reshape(rows, cols // BLOCK, BLOCK)


def q4_0_scale_rtn(blocks: torch.Tensor) -> torch.Tensor:
    """The ggml reference scale: d = max / -8, with max the signed value of largest magnitude."""
    idx = blocks.abs().argmax(dim=-1, keepdim=True)
    m = torch.gather(blocks, -1, idx).squeeze(-1)
    d = m / -8.0
    return torch.where(d == 0, torch.ones_like(d), d)


def q4_0_round(blocks: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Round blocks to the grid for the scales d [rows, nblocks]."""
    return torch.clamp(torch.round(blocks / d[..., None]), Q4_MIN, Q4_MAX)


def q4_0_scale_search(blocks: torch.Tensor, weights: torch.Tensor | None = None,
                      n_candidates: int = 40, lo: float = 0.55, hi: float = 1.05) -> torch.Tensor:
    """Pick the scale per block that minimizes the (weighted) squared error.

    Candidates are the reference scale times factors in [lo, hi]. This is the
    diagonal-Hessian scale search of NeUQI for a grid without zero point.
    ``weights`` [cols] weights the error per input column.
    Complexity is O(rows · cols · n_candidates).
    """
    d0 = q4_0_scale_rtn(blocks)
    factors = torch.linspace(lo, hi, n_candidates, device=blocks.device, dtype=blocks.dtype)
    best_d = d0.clone()
    best_err = torch.full_like(d0, float("inf"))
    wgt = None
    if weights is not None:
        wgt = weights.reshape(1, -1, BLOCK).to(blocks.dtype)
    for f in factors:
        d = d0 * f
        q = q4_0_round(blocks, d)
        err = (blocks - q * d[..., None]).pow(2)
        if wgt is not None:
            err = err * wgt
        err = err.sum(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d, best_d)
    return best_d


def q4_0_quantize(w: torch.Tensor, weights: torch.Tensor | None = None, search: bool = True):
    """Quantize [rows, cols] to (q int8 [rows, cols], d float16 [rows, nblocks])."""
    blocks = _blocks(w.to(torch.float32))
    d = q4_0_scale_search(blocks, weights) if search else q4_0_scale_rtn(blocks)
    d = d.to(torch.float16).to(torch.float32)
    q = q4_0_round(blocks, d)
    return q.reshape(w.shape).to(torch.int8), d.to(torch.float16)


def q4_0_dequantize(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The float32 values d · q of a quantized matrix."""
    rows, cols = q.shape
    return (q.reshape(rows, cols // BLOCK, BLOCK).to(torch.float32) * d.to(torch.float32)[..., None]).reshape(rows, cols)


def pack_q4_0(q: torch.Tensor, d: torch.Tensor) -> np.ndarray:
    """Pack to the ggml byte layout: uint8 [rows, nblocks · 18]."""
    rows, cols = q.shape
    nb = cols // BLOCK
    qb = (q.reshape(rows, nb, BLOCK).to(torch.int16) + 8).to(torch.uint8).cpu().numpy()
    lo, hi = qb[:, :, :16], qb[:, :, 16:]
    nibbles = (lo | (hi << 4)).astype(np.uint8)
    scales = d.reshape(rows, nb).to(torch.float16).cpu().numpy().view(np.uint8).reshape(rows, nb, 2)
    return np.concatenate([scales, nibbles], axis=2).reshape(rows, nb * 18)


def q8_0_quantize(w: torch.Tensor):
    """Round-to-nearest Q8_0: (q int8 [rows, cols], d float16 [rows, nblocks])."""
    blocks = _blocks(w.to(torch.float32))
    amax = blocks.abs().amax(dim=-1)
    d = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0).to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(blocks / d[..., None]), -127, 127)
    return q.reshape(w.shape).to(torch.int8), d.to(torch.float16)


def pack_q8_0(q: torch.Tensor, d: torch.Tensor) -> np.ndarray:
    """Pack to the ggml byte layout: uint8 [rows, nblocks · 34]."""
    rows, cols = q.shape
    nb = cols // BLOCK
    qb = q.reshape(rows, nb, BLOCK).cpu().numpy().view(np.uint8)
    scales = d.reshape(rows, nb).to(torch.float16).cpu().numpy().view(np.uint8).reshape(rows, nb, 2)
    return np.concatenate([scales, qb], axis=2).reshape(rows, nb * 34)
