"""The 4-bit grids: one F16 scale per block of 32, sixteen levels in units of the scale.

Every grid stores 18 bytes per block of 32 weights (4.5 bits per weight):
the scale and one nibble per weight. They differ in the sixteen levels.

- Q4_0: the uniform levels −8 … 7, the ggml format that the phone kernels read.
- IQ4_NL: the non-uniform integer table of ggml, close to the Lloyd–Max
  levels of a Gaussian. llama.cpp reads it on the CPU and on CUDA.
- CB4: a table per matrix, fit by weighted Lloyd–Max on the rotated
  weights of that matrix, with integer levels in −127 … 127. It needs our
  kernel: a 16-entry lookup on the nibble before the int8 dot product.

A grid rounds x/d to the nearest level and returns the level index. The
scale search of the solver evaluates each candidate scale through the grid.
"""

from __future__ import annotations

import torch

BLOCK = 32

# The kvalues_iq4nl table of ggml-common.h. The two halves are not symmetric.
IQ4_NL_TABLE = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def divisor(d: torch.Tensor) -> torch.Tensor:
    """The divisor that gives the level indices of blocks with the scales ``d``: 1 in place of a zero scale.

    A zero scale decodes its block to zero with each index, also on a grid
    with no zero level. The division by 1 gives a defined index in place of
    0/0 = NaN, whose cast to int32 has no defined result.
    """
    return torch.where(d == 0, torch.ones_like(d), d)


class Grid:
    """Sixteen levels in units of the block scale, sorted."""

    name: str = "grid"

    def __init__(self, levels: torch.Tensor) -> None:
        self.levels = levels.to(torch.float32).sort().values
        self.mid = (self.levels[1:] + self.levels[:-1]) / 2
        # The level of the largest magnitude. The reference scale maps the signed maximum of a block onto it.
        self.top = self.levels[0] if abs(self.levels[0]) >= abs(self.levels[-1]) else self.levels[-1]

    def to(self, device: torch.device) -> "Grid":
        out = self.__class__.__new__(self.__class__)
        out.name = self.name
        out.levels, out.mid, out.top = self.levels.to(device), self.mid.to(device), self.top.to(device)
        return out

    def scale_rtn(self, blocks: torch.Tensor) -> torch.Tensor:
        """The reference scale per block [rows, nblocks]: the signed maximum maps to the level of largest magnitude.

        The scale carries the sign of the maximum. Every ggml block format
        stores a signed F16 scale, and the ggml quantizers select the sign
        in the same way, thus an asymmetric grid (Q4_0: −8 … 7, IQ4_NL:
        −127 … 113) keeps its full range on the side of the maximum.

        An all-zero block gets the scale 0, as in the ggml reference
        quantizers. Thus it decodes to zero also on a grid with no zero level
        (IQ4_NL, a symmetric codebook), where a non-zero scale decodes each
        element to the smallest level times the scale. The maximum of such a
        block is +0 also when the block holds -0, as in quantize_row_q4_0_ref,
        thus the Q4_0 scale is +0 / -8 = -0 with the bytes of that function.
        """
        idx = blocks.abs().argmax(dim=-1, keepdim=True)
        m = torch.gather(blocks, -1, idx).squeeze(-1)
        return torch.where(m == 0, torch.zeros_like(m), m) / self.top

    def round(self, x: torch.Tensor) -> torch.Tensor:
        """The index (int32) of the nearest level of x, which is in units of the scale."""
        return torch.bucketize(x, self.mid, out_int32=True)

    def value(self, idx: torch.Tensor) -> torch.Tensor:
        """The level of each index, in units of the scale. int32 indices index without a copy to int64."""
        if idx.dtype not in (torch.int32, torch.int64):
            idx = idx.to(torch.int32)
        return self.levels[idx]

    def quantize_blocks(self, blocks: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Indices [rows, nblocks, 32] for the scales d [rows, nblocks].

        An all-zero block, and the F16 rounding of a very small scale, give a
        zero scale, which decodes its block to zero (refer to ``divisor``).
        Complexity is O(elements).
        """
        return self.round(blocks / divisor(d)[..., None])


class Q4_0Grid(Grid):
    """The uniform ggml grid: levels −8 … 7, the signed maximum maps to −8."""

    name = "Q4_0"

    def __init__(self) -> None:
        super().__init__(torch.arange(-8, 8, dtype=torch.float32))

    def round(self, x: torch.Tensor) -> torch.Tensor:
        """The uniform grid rounds directly, without the search over the midpoints."""
        return (torch.clamp(torch.round(x), -8, 7) + 8).to(torch.int32)


class IQ4NLGrid(Grid):
    """The non-uniform integer table of ggml."""

    name = "IQ4_NL"

    def __init__(self) -> None:
        super().__init__(torch.tensor(IQ4_NL_TABLE, dtype=torch.float32))


class CodebookGrid(Grid):
    """A table per matrix with integer levels in −127 … 127."""

    name = "CB4"


def fit_codebook(w: torch.Tensor, col_weights: torch.Tensor | None = None, iters: int = 30) -> CodebookGrid:
    """Weighted Lloyd–Max levels for the blocks of ``w`` [rows, cols], scaled by their reference scale.

    The weights are normalized by the IQ4_NL reference scale of their block,
    thus the levels live in the integer range of the table. The iteration
    keeps the levels symmetric and rounds them to integers at the end.
    Complexity is O(iters · rows · cols).
    """
    base = IQ4NLGrid().to(w.device)
    blocks = w.to(torch.float32).reshape(w.shape[0], -1, BLOCK)
    x = (blocks / divisor(base.scale_rtn(blocks))[..., None]).reshape(-1)
    wt = None
    if col_weights is not None:
        wt = col_weights.to(torch.float32).repeat(w.shape[0]).reshape(-1)
        wt = wt / wt.mean().clamp_min(1e-30)
    levels = base.levels.clone()
    for _ in range(iters):
        mid = (levels[1:] + levels[:-1]) / 2
        cell = torch.bucketize(x, mid)
        m = torch.ones_like(x) if wt is None else wt
        mass = torch.zeros(16, device=w.device).index_add_(0, cell, m)
        first = torch.zeros(16, device=w.device).index_add_(0, cell, m * x)
        new = first / mass.clamp_min(1e-30)
        keep = mass > 0
        levels = torch.where(keep, new, levels)
        levels = (levels - levels.flip(0)) / 2
    levels = torch.clamp(torch.round(levels), -127, 127)
    return CodebookGrid(levels)


def make_grid(name: str) -> Grid:
    """The fixed grid of a plan type name."""
    if name == "Q4_0":
        return Q4_0Grid()
    if name == "IQ4_NL":
        return IQ4NLGrid()
    raise ValueError(f"no fixed grid for {name}")
