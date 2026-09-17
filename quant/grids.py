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

IQ4_NL_TABLE = (-127, -104, -83, -65, -49, -35, -22, -10, 6, 17, 29, 44, 58, 77, 97, 127)


class Grid:
    """Sixteen levels in units of the block scale, sorted."""

    name: str = "grid"

    def __init__(self, levels: torch.Tensor) -> None:
        self.levels = levels.to(torch.float32).sort().values
        self.mid = (self.levels[1:] + self.levels[:-1]) / 2
        self.amax = self.levels.abs().max()

    def to(self, device: torch.device) -> "Grid":
        out = self.__class__.__new__(self.__class__)
        out.name = self.name
        out.levels, out.mid, out.amax = self.levels.to(device), self.mid.to(device), self.amax.to(device)
        return out

    def scale_rtn(self, blocks: torch.Tensor) -> torch.Tensor:
        """The reference scale per block [rows, nblocks]: the largest magnitude maps to the largest level."""
        d = blocks.abs().amax(dim=-1) / self.amax
        return torch.where(d == 0, torch.ones_like(d), d)

    def round(self, x: torch.Tensor) -> torch.Tensor:
        """The index of the nearest level of x, which is in units of the scale."""
        return torch.bucketize(x, self.mid)

    def value(self, idx: torch.Tensor) -> torch.Tensor:
        """The level of each index, in units of the scale."""
        return self.levels[idx.to(torch.long)]

    def quantize_blocks(self, blocks: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Indices [rows, nblocks, 32] for the scales d [rows, nblocks]."""
        return self.round(blocks / d[..., None])


class Q4_0Grid(Grid):
    """The uniform ggml grid: levels −8 … 7, the signed maximum maps to −8."""

    name = "Q4_0"

    def __init__(self) -> None:
        super().__init__(torch.arange(-8, 8, dtype=torch.float32))

    def scale_rtn(self, blocks: torch.Tensor) -> torch.Tensor:
        idx = blocks.abs().argmax(dim=-1, keepdim=True)
        m = torch.gather(blocks, -1, idx).squeeze(-1)
        d = m / -8.0
        return torch.where(d == 0, torch.ones_like(d), d)

    def round(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.clamp(torch.round(x), -8, 7) + 8).to(torch.long)


class IQ4NLGrid(Grid):
    """The non-uniform integer table of ggml."""

    name = "IQ4_NL"

    def __init__(self) -> None:
        super().__init__(torch.tensor(IQ4_NL_TABLE, dtype=torch.float32))


class CodebookGrid(Grid):
    """A table per matrix with integer levels in −127 … 127."""

    name = "CB4"


def gaussian_lloyd_max(n: int = 16, iters: int = 200) -> torch.Tensor:
    """The Lloyd–Max levels of a standard Gaussian, from a fine sample."""
    x = torch.linspace(-4.5, 4.5, 90_001, dtype=torch.float64)
    p = torch.exp(-0.5 * x * x)
    levels = torch.linspace(-2.5, 2.5, n, dtype=torch.float64)
    for _ in range(iters):
        mid = (levels[1:] + levels[:-1]) / 2
        cell = torch.bucketize(x, mid)
        mass = torch.zeros(n, dtype=torch.float64).index_add_(0, cell, p)
        first = torch.zeros(n, dtype=torch.float64).index_add_(0, cell, p * x)
        levels = first / mass.clamp_min(1e-30)
    return levels.to(torch.float32)


def fit_codebook(w: torch.Tensor, col_weights: torch.Tensor | None = None, iters: int = 30) -> CodebookGrid:
    """Weighted Lloyd–Max levels for the blocks of ``w`` [rows, cols], scaled by their reference scale.

    The weights are normalized by the IQ4_NL reference scale of their block,
    thus the levels live in the integer range of the table. The iteration
    keeps the levels symmetric and rounds them to integers at the end.
    Complexity is O(iters · rows · cols).
    """
    base = IQ4NLGrid().to(w.device)
    blocks = w.to(torch.float32).reshape(w.shape[0], -1, BLOCK)
    x = (blocks / base.scale_rtn(blocks)[..., None]).reshape(-1)
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
