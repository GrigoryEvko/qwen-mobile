"""Block quantization on a grid, the scale search, and the ggml byte packing.

A 4-bit grid (``grids.Grid``) stores one F16 scale per block of 32 and one
nibble per weight. Q4_0 and IQ4_NL share the byte layout of ggml: the
scale, then 16 bytes with element j in the low nibble of byte j and
element j + 16 in the high nibble.

Q8_0: blocks of 32, one F16 scale, levels in [-127, 127], 34 bytes.
"""

from __future__ import annotations

import numpy as np
import torch

from .grids import Grid

BLOCK = 32


def blocks_of(w: torch.Tensor) -> torch.Tensor:
    """View [rows, cols] as [rows, cols // 32, 32]."""
    rows, cols = w.shape
    if cols % BLOCK:
        raise ValueError(f"columns {cols} are not a multiple of {BLOCK}")
    return w.reshape(rows, cols // BLOCK, BLOCK)


def scale_search(grid: Grid, blocks: torch.Tensor, weights: torch.Tensor | None = None,
                 n_candidates: int = 40, lo: float = 0.55, hi: float = 1.05) -> torch.Tensor:
    """Pick the scale per block that minimizes the (weighted) squared error on the grid.

    Candidates are the reference scale times factors in [lo, hi]. This is the
    diagonal-Hessian scale search of NeUQI for a grid without zero point.
    ``weights`` [cols] weights the error per input column.
    Complexity is O(rows · cols · n_candidates).
    """
    d0 = grid.scale_rtn(blocks)
    factors = torch.linspace(lo, hi, n_candidates, device=blocks.device, dtype=blocks.dtype)
    best_d = d0.clone()
    best_err = torch.full_like(d0, float("inf"))
    wgt = None
    if weights is not None:
        wgt = weights.reshape(1, -1, BLOCK).to(blocks.dtype)
    for f in factors:
        d = d0 * f
        err = (blocks - grid.value(grid.quantize_blocks(blocks, d)) * d[..., None]).pow(2)
        if wgt is not None:
            err = err * wgt
        err = err.sum(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d, best_d)
    return best_d


ROW_CHUNK = 16384


def quantize(grid: Grid, w: torch.Tensor, weights: torch.Tensor | None = None, search: bool = True):
    """Quantize [rows, cols] to (indices int8 [rows, cols], d float16 [rows, nblocks]).

    The work runs in row chunks, thus the head (248320 x 2048) needs no
    multi-gigabyte temporaries.
    """
    rows, cols = w.shape
    idx_out = torch.empty(rows, cols, dtype=torch.int8, device=w.device)
    d_out = torch.empty(rows, cols // BLOCK, dtype=torch.float16, device=w.device)
    for r in range(0, rows, ROW_CHUNK):
        blocks = blocks_of(w[r:r + ROW_CHUNK].to(torch.float32))
        d = scale_search(grid, blocks, weights) if search else grid.scale_rtn(blocks)
        d = d.to(torch.float16).to(torch.float32)
        idx_out[r:r + ROW_CHUNK] = grid.quantize_blocks(blocks, d).reshape(blocks.shape[0], cols).to(torch.int8)
        d_out[r:r + ROW_CHUNK] = d.to(torch.float16)
    return idx_out, d_out


def dequantize(grid: Grid, idx: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The float32 values d · level[idx] of a quantized matrix, in row chunks."""
    rows, cols = idx.shape
    out = torch.empty(rows, cols, dtype=torch.float32, device=idx.device)
    for r in range(0, rows, ROW_CHUNK):
        values = grid.value(idx[r:r + ROW_CHUNK].reshape(-1, cols // BLOCK, BLOCK))
        out[r:r + ROW_CHUNK] = (values * d[r:r + ROW_CHUNK].to(torch.float32)[..., None]).reshape(-1, cols)
    return out


def dequantize_pack(z, device: torch.device) -> torch.Tensor:
    """The values of a saved pack. A pack without ``levels`` holds Q4_0 levels as indices minus 8."""
    from .grids import Grid, Q4_0Grid

    idx = torch.from_numpy(z["q"]).to(device)
    d = torch.from_numpy(z["d"].view(np.float16)).to(device)
    if "levels" in z:
        grid = Grid(torch.from_numpy(z["levels"])).to(device)
    else:
        grid = Q4_0Grid().to(device)
        idx = idx.to(torch.int16) + 8
    return dequantize(grid, idx, d)


def pack_nibbles(idx: torch.Tensor, d: torch.Tensor) -> np.ndarray:
    """Pack indices 0 … 15 to the ggml byte layout: uint8 [rows, nblocks · 18]."""
    rows, cols = idx.shape
    nb = cols // BLOCK
    qb = idx.reshape(rows, nb, BLOCK).to(torch.uint8).cpu().numpy()
    lo, hi = qb[:, :, :16], qb[:, :, 16:]
    nibbles = (lo | (hi << 4)).astype(np.uint8)
    scales = d.reshape(rows, nb).to(torch.float16).cpu().numpy().view(np.uint8).reshape(rows, nb, 2)
    return np.concatenate([scales, nibbles], axis=2).reshape(rows, nb * 18)


def q8_0_quantize(w: torch.Tensor):
    """Round-to-nearest Q8_0: (q int8 [rows, cols], d float16 [rows, nblocks])."""
    blocks = blocks_of(w.to(torch.float32))
    amax = blocks.abs().amax(dim=-1)
    d = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0).to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(blocks / d[..., None]), -127, 127)
    return q.reshape(w.shape).to(torch.int8), d.to(torch.float16)


def q8_0_dequantize(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The float32 values d · q of a Q8_0 matrix."""
    rows, cols = q.shape
    return (q.reshape(rows, cols // BLOCK, BLOCK).to(torch.float32) * d.to(torch.float32)[..., None]).reshape(rows, cols)


def pack_q8_0(q: torch.Tensor, d: torch.Tensor) -> np.ndarray:
    """Pack to the ggml byte layout: uint8 [rows, nblocks · 34]."""
    rows, cols = q.shape
    nb = cols // BLOCK
    qb = q.reshape(rows, nb, BLOCK).cpu().numpy().view(np.uint8)
    scales = d.reshape(rows, nb).to(torch.float16).cpu().numpy().view(np.uint8).reshape(rows, nb, 2)
    return np.concatenate([scales, qb], axis=2).reshape(rows, nb * 34)
