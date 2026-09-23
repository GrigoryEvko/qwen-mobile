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
                 n_candidates: int = 41, lo: float = 0.55, hi: float = 1.05) -> torch.Tensor:
    """Pick the scale per block that minimizes the (weighted) squared error on the grid.

    Candidates are the reference scale times factors in [lo, hi]. The 41
    points have the step 0.0125. linspace in float32 gives 0.99999994 near
    1, thus the factor nearest to 1 is set to exactly 1: the reference scale
    is a candidate, and a block that is already on the grid keeps its scale. This
    is the diagonal-Hessian scale search of NeUQI for a grid without zero
    point. ``weights`` [cols] weights the error per input column. Each
    candidate is rounded to F16 before its error is measured, thus the
    winner is the scale that the file stores, also in the F16 subnormal
    range. A candidate that rounds to zero decodes its block to zero
    (Grid.quantize_blocks). Complexity is O(rows · cols · n_candidates).
    """
    d0 = grid.scale_rtn(blocks)
    factors = torch.linspace(lo, hi, n_candidates, device=blocks.device, dtype=blocks.dtype)
    factors[(factors - 1).abs().argmin()] = 1.0
    best_d = d0.clone()
    best_err = torch.full_like(d0, float("inf"))
    wgt = None
    if weights is not None:
        wgt = weights.reshape(1, -1, BLOCK).to(blocks.dtype)
    for f in factors:
        d = (d0 * f).to(torch.float16).to(blocks.dtype)
        err = (blocks - grid.value(grid.quantize_blocks(blocks, d)) * d[..., None]).pow(2)
        if wgt is not None:
            err = err * wgt
        err = err.sum(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d, best_d)
    return best_d


ROW_CHUNK = 16384


def _check_scales(d: torch.Tensor, blocks: torch.Tensor, row0: int, largest: float) -> None:
    """Raise ValueError when an F16 block scale is not finite. Complexity is O(blocks).

    The block maximum of torch keeps NaN and infinity, thus a value that is
    not finite gives a scale that is not finite. This check finds that value
    too, with no pass over the elements. ``largest`` is the largest block
    maximum that the format permits, for the error text.
    """
    bad = ~torch.isfinite(d)
    if not bad.any():
        return
    r, b = (int(x) for x in bad.nonzero()[0])
    block = blocks[r, b]
    odd = (~torch.isfinite(block)).nonzero()
    if odd.numel():
        j = int(odd[0])
        raise ValueError(f"the weight at row {row0 + r}, column {b * BLOCK + j} is {float(block[j])}: "
                         "a block format needs finite values")
    raise ValueError(f"the block of 32 at row {row0 + r}, column {b * BLOCK} has the maximum "
                     f"{float(block.abs().max()):.6g}: its scale is beyond the F16 range, which permits a block "
                     f"maximum of about {largest:.6g}")


def quantize(grid: Grid, w: torch.Tensor, weights: torch.Tensor | None = None, search: bool = True):
    """Quantize [rows, cols] to (indices int8 [rows, cols], d float16 [rows, nblocks]).

    The work runs in row chunks on the device of the grid, and the results
    live there. ``w`` can be on another device, thus the head (248320 x 2048)
    needs no multi-gigabyte temporaries on the device. Raises ValueError for
    a value that is not finite and for a block whose scale is beyond F16.
    """
    rows, cols = w.shape
    dev = grid.levels.device
    idx_out = torch.empty(rows, cols, dtype=torch.int8, device=dev)
    d_out = torch.empty(rows, cols // BLOCK, dtype=torch.float16, device=dev)
    if weights is not None:
        weights = weights.to(dev)
    for r in range(0, rows, ROW_CHUNK):
        blocks = blocks_of(w[r:r + ROW_CHUNK].to(dev, torch.float32))
        largest = 65504.0 * float(grid.top.abs())
        d0 = grid.scale_rtn(blocks)
        # The reference scale must fit F16. The search could clamp such a block with a smaller finite scale.
        _check_scales(d0.to(torch.float16).to(torch.float32), blocks, r, largest)
        d = scale_search(grid, blocks, weights) if search else d0
        d = d.to(torch.float16).to(torch.float32)
        _check_scales(d, blocks, r, largest)
        idx_out[r:r + ROW_CHUNK] = grid.quantize_blocks(blocks, d).reshape(blocks.shape[0], cols).to(torch.int8)
        d_out[r:r + ROW_CHUNK] = d.to(torch.float16)
    return idx_out, d_out


def dequantize(grid: Grid, idx: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The float32 values d · level[idx] of a quantized matrix, in row chunks, on the device of the grid."""
    rows, cols = idx.shape
    dev = grid.levels.device
    out = torch.empty(rows, cols, dtype=torch.float32, device=dev)
    for r in range(0, rows, ROW_CHUNK):
        values = grid.value(idx[r:r + ROW_CHUNK].to(dev).reshape(-1, cols // BLOCK, BLOCK))
        out[r:r + ROW_CHUNK] = (values * d[r:r + ROW_CHUNK].to(dev, torch.float32)[..., None]).reshape(-1, cols)
    return out


def block_error(grid: Grid, w: torch.Tensor, col_weights: torch.Tensor,
                col_scale: torch.Tensor | None = None) -> torch.Tensor:
    """The weighted squared error of the block quantization of w · diag(col_scale), with the scale search.

    The error is Σ col_weights[j] · (ŵ[i, j] − w[i, j])² with ŵ the
    round trip through the grid and the F16 scales. The work runs in row
    chunks, thus the head needs no full-size temporaries. Complexity is
    O(rows · cols · n_candidates).
    """
    rows, cols = w.shape
    dev = grid.levels.device
    col_weights = col_weights.to(dev, torch.float32)
    wgt = col_weights.reshape(1, cols // BLOCK, BLOCK)
    total = torch.zeros((), dtype=torch.float32, device=dev)
    for r in range(0, rows, ROW_CHUNK):
        chunk = w[r:r + ROW_CHUNK].to(dev, torch.float32)
        if col_scale is not None:
            chunk = chunk * col_scale.to(dev)[None, :]
        blocks = blocks_of(chunk)
        d = scale_search(grid, blocks, col_weights).to(torch.float16).to(torch.float32)
        deq = grid.value(grid.quantize_blocks(blocks, d)) * d[..., None]
        total += ((deq - blocks).pow(2) * wgt).sum()
    return total


def block_permutation(cols: torch.Tensor) -> torch.Tensor:
    """The permutation of the blocks of 32 that a column permutation of whole blocks makes."""
    runs = cols.view(-1, BLOCK)
    if not torch.equal(runs, runs[:, :1] + torch.arange(BLOCK)) or int((runs[:, 0] % BLOCK).max()) != 0:
        raise ValueError("the column permutation splits a block of 32, the packed scales cannot follow it")
    return runs[:, 0] // BLOCK


def dequantize_pack(z, device: torch.device, rows: torch.Tensor | None = None,
                    cols: torch.Tensor | None = None) -> torch.Tensor:
    """The values of a saved pack, the low-rank correction included.

    A pack without ``levels`` holds Q4_0 levels as indices minus 8. ``rows``
    and ``cols`` are index permutations: the result holds pack row rows[i]
    at row i, and pack column cols[j] at column j. The column permutation
    must move whole blocks of 32.
    """
    from .grids import Grid, Q4_0Grid

    idx = torch.from_numpy(z["q"]).to(device)
    d = torch.from_numpy(z["d"].view(np.float16)).to(device)
    if "levels" in z:
        grid = Grid(torch.from_numpy(z["levels"])).to(device)
    else:
        grid = Q4_0Grid().to(device)
        idx = idx.to(torch.int16) + 8
    if rows is not None:
        idx, d = idx[rows.to(device)], d[rows.to(device)]
    if cols is not None:
        idx, d = idx[:, cols.to(device)], d[:, block_permutation(cols).to(device)]
    w = dequantize(grid, idx, d)
    if "lora_a" in z.files:
        a, b = torch.from_numpy(z["lora_a"]).to(device), torch.from_numpy(z["lora_b"]).to(device)
        if rows is not None:
            b = b[rows.to(device)]
        if cols is not None:
            a = a[:, cols.to(device)]
        w = w + b @ a
    return w


def _check_pack(idx: torch.Tensor, d: torch.Tensor, lo: int, hi: int) -> None:
    """Raise ValueError when the levels are out of lo … hi, or the scales do not fit the levels or are not finite."""
    rows, cols = idx.shape
    if cols % BLOCK:
        raise ValueError(f"the levels have {cols} columns, which is not a multiple of {BLOCK}")
    if tuple(d.shape) != (rows, cols // BLOCK):
        raise ValueError(f"the scales have the shape {tuple(d.shape)}, the levels [{rows}, {cols}] need "
                         f"[{rows}, {cols // BLOCK}]")
    if idx.numel():
        least, most = (int(x) for x in torch.aminmax(idx))
        if least < lo or most > hi:
            raise ValueError(f"a level is out of {lo} … {hi}: the range of the levels is {least} … {most}")
    bad = ~torch.isfinite(d.to(torch.float32))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} block scales are not finite")


def pack_nibbles(idx: torch.Tensor, d: torch.Tensor) -> np.ndarray:
    """Pack indices 0 … 15 to the ggml byte layout: uint8 [rows, nblocks · 18].

    Raises ValueError for an index out of 0 … 15, for scales that do not
    fit the indices, and for a scale that is not finite.
    """
    _check_pack(idx, d, 0, 15)
    rows, cols = idx.shape
    nb = cols // BLOCK
    qb = idx.reshape(rows, nb, BLOCK).to(torch.uint8).cpu().numpy()
    lo, hi = qb[:, :, :16], qb[:, :, 16:]
    nibbles = (lo | (hi << 4)).astype(np.uint8)
    scales = d.reshape(rows, nb).to(torch.float16).cpu().numpy().view(np.uint8).reshape(rows, nb, 2)
    return np.concatenate([scales, nibbles], axis=2).reshape(rows, nb * 18)


def q8_0_quantize(w: torch.Tensor):
    """Round-to-nearest Q8_0: (q int8 [rows, cols], d float16 [rows, nblocks]), in row chunks.

    Raises ValueError for a value that is not finite and for a block whose
    maximum is beyond 127 times the largest F16 value.
    """
    rows, cols = w.shape
    q_out = torch.empty(rows, cols, dtype=torch.int8, device=w.device)
    d_out = torch.empty(rows, cols // BLOCK, dtype=torch.float16, device=w.device)
    for r in range(0, rows, ROW_CHUNK):
        blocks = blocks_of(w[r:r + ROW_CHUNK].to(torch.float32))
        amax = blocks.abs().amax(dim=-1)
        d = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0).to(torch.float16).to(torch.float32)
        _check_scales(d, blocks, r, 65504.0 * 127.0)
        q_out[r:r + ROW_CHUNK] = torch.clamp(torch.round(blocks / d[..., None]), -127, 127).reshape(-1, cols).to(torch.int8)
        d_out[r:r + ROW_CHUNK] = d.to(torch.float16)
    return q_out, d_out


def q8_0_dequantize(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """The float32 values d · q of a Q8_0 matrix, in row chunks."""
    rows, cols = q.shape
    out = torch.empty(rows, cols, dtype=torch.float32, device=q.device)
    for r in range(0, rows, ROW_CHUNK):
        blocks = q[r:r + ROW_CHUNK].reshape(-1, cols // BLOCK, BLOCK).to(torch.float32)
        out[r:r + ROW_CHUNK] = (blocks * d[r:r + ROW_CHUNK].to(torch.float32)[..., None]).reshape(-1, cols)
    return out


def pack_q8_0(q: torch.Tensor, d: torch.Tensor) -> np.ndarray:
    """Pack to the ggml byte layout: uint8 [rows, nblocks · 34].

    Raises ValueError for a level out of -127 … 127, for scales that do not
    fit the levels, and for a scale that is not finite.
    """
    _check_pack(q, d, -127, 127)
    rows, cols = q.shape
    nb = cols // BLOCK
    qb = q.reshape(rows, nb, BLOCK).cpu().numpy().view(np.uint8)
    scales = d.reshape(rows, nb).to(torch.float16).cpu().numpy().view(np.uint8).reshape(rows, nb, 2)
    return np.concatenate([scales, qb], axis=2).reshape(rows, nb * 34)
