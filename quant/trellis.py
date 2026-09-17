"""Trellis-coded quantization at 4 bits per weight (QTIP, Tseng et al. 2024).

A row of weights is a path through a bit-shift trellis. The bit stream holds
k = 4 bits per weight. The state at position i is the window of the last L
bits of the stream, thus the value of a weight is lut[state], a table with
2^L entries, and the code constrains each value by its L/k − 1 neighbors.
That constraint is what buys the shaping gain over a 16-level grid at the
same bit rate: the decoder still reads one nibble per weight, one table
lookup, and one F16 scale per block of 32.

The encoder is the Viterbi algorithm on the Hessian-weighted squared error,
batched over rows on the GPU. The table starts as Gaussian samples, then
Lloyd steps move each entry to the weighted mean of the weights it codes.
The first L − k bits of each row are free and travel with the row.

Complexity of one encode: O(weights · 2^L · 2^k).
"""

from __future__ import annotations

import torch

BLOCK = 32


class Trellis:
    """The bit-shift trellis with 2^L states, k bits per step, and an integer table of 2^L values."""

    def __init__(self, lut: torch.Tensor, k: int = 4) -> None:
        self.lut = lut.to(torch.float32)
        self.n_states = lut.shape[0]
        self.L = self.n_states.bit_length() - 1
        assert 1 << self.L == self.n_states, "the table size must be a power of two"
        self.k = k
        self.branches = 1 << k
        self.mask = self.n_states - 1
        device = lut.device
        # pred[s', t] = the predecessor state that reaches s' with the phantom high bits t.
        s = torch.arange(self.n_states, device=device)
        t = torch.arange(self.branches, device=device)
        self.pred = (s[:, None] >> k) | (t[None, :] << (self.L - k))

    @staticmethod
    def gaussian(L: int, k: int = 4, seed: int = 0, device: torch.device | None = None) -> "Trellis":
        """A table of 2^L Gaussian samples, integer levels in −127 … 127, sorted by state bits for no reason."""
        gen = torch.Generator().manual_seed(seed)
        z = torch.randn(1 << L, generator=gen)
        lut = torch.clamp(torch.round(z / z.abs().max() * 127), -127, 127)
        return Trellis(lut.to(device or "cpu"), k)

    def encode(self, x: torch.Tensor, weights: torch.Tensor | None = None, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        """Viterbi over each row of x [N, T] (in table units). Returns (codes int8 [N, T], head int32 [N]).

        ``weights`` [N, T] weights the squared error per position. ``head``
        holds the L − k free bits before the first code. The decode of
        position i uses the state window over head and codes[:i + 1].
        """
        n, t_len = x.shape
        codes = torch.empty(n, t_len, dtype=torch.int8, device=x.device)
        heads = torch.empty(n, dtype=torch.int32, device=x.device)
        for start in range(0, n, chunk):
            c, h = self._encode_chunk(x[start:start + chunk], None if weights is None else weights[start:start + chunk])
            codes[start:start + chunk] = c
            heads[start:start + chunk] = h
        return codes, heads

    def _encode_chunk(self, x: torch.Tensor, weights: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        n, t_len = x.shape
        device = x.device
        # cost[r, s]: the best path cost that ends in state s after the current position.
        cost = torch.zeros(n, self.n_states, device=device)
        back = torch.empty(t_len, n, self.n_states, dtype=torch.uint8, device=device)
        lut = self.lut[None, :]
        for i in range(t_len):
            err = (x[:, i:i + 1] - lut).pow(2)
            if weights is not None:
                err = err * weights[:, i:i + 1]
            # cand[r, s', t] = cost[r, pred[s', t]]
            cand = cost[:, self.pred]
            best, arg = cand.min(dim=2)
            cost = best + err
            back[i] = arg.to(torch.uint8)
        state = cost.argmin(dim=1)
        codes = torch.empty(n, t_len, dtype=torch.int8, device=device)
        rows = torch.arange(n, device=device)
        for i in range(t_len - 1, -1, -1):
            codes[:, i] = (state & (self.branches - 1)).to(torch.int8)
            t = back[i][rows, state].to(torch.long)
            state = self.pred[state, t]
        return codes, (state & ((1 << (self.L - self.k)) - 1)).to(torch.int32)

    def states(self, codes: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """The state window at each position, from the free bits and the codes. Complexity O(N · T)."""
        n, t_len = codes.shape
        state = heads.to(torch.long).clone()
        out = torch.empty(n, t_len, dtype=torch.long, device=codes.device)
        for i in range(t_len):
            state = ((state << self.k) | codes[:, i].to(torch.long)) & self.mask
            out[:, i] = state
        return out

    def decode(self, codes: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """The values of the codes, in table units."""
        return self.lut[self.states(codes, heads)]

    def lloyd_step(self, x: torch.Tensor, codes: torch.Tensor, heads: torch.Tensor,
                   weights: torch.Tensor | None = None) -> None:
        """Move each table entry to the weighted mean of the values it codes, integer levels."""
        st = self.states(codes, heads).reshape(-1)
        xf = x.reshape(-1)
        w = torch.ones_like(xf) if weights is None else weights.reshape(-1)
        total = torch.zeros(self.n_states, device=x.device).index_add_(0, st, w * xf)
        mass = torch.zeros(self.n_states, device=x.device).index_add_(0, st, w)
        new = total / mass.clamp_min(1e-30)
        self.lut = torch.where(mass > 0, torch.clamp(torch.round(new), -127, 127), self.lut)


def quantize_trellis(w: torch.Tensor, d: torch.Tensor, trellis: Trellis, col_weights: torch.Tensor | None = None,
                     lloyd_iters: int = 2) -> tuple[torch.Tensor, torch.Tensor, Trellis]:
    """Encode [rows, cols] with the block scales d [rows, cols // 32], refine the table, encode again.

    Returns (codes, heads, trellis). The values are d[block] · lut[state].
    """
    rows, cols = w.shape
    x = (w.to(torch.float32).reshape(rows, cols // BLOCK, BLOCK) / d.to(torch.float32)[..., None]).reshape(rows, cols)
    wt = None
    if col_weights is not None:
        wt = (col_weights.to(torch.float32)[None, :] * d.to(torch.float32).repeat_interleave(BLOCK, dim=1).pow(2))
    codes, heads = trellis.encode(x, wt)
    for _ in range(lloyd_iters):
        trellis.lloyd_step(x, codes, heads, wt)
        codes, heads = trellis.encode(x, wt)
    return codes, heads, trellis


def dequantize_trellis(codes: torch.Tensor, heads: torch.Tensor, d: torch.Tensor, trellis: Trellis) -> torch.Tensor:
    """The float32 weights of the codes."""
    rows, cols = codes.shape
    values = trellis.decode(codes, heads).reshape(rows, cols // BLOCK, BLOCK)
    return (values * d.to(torch.float32)[..., None]).reshape(rows, cols)


def self_test(rows: int = 256, cols: int = 512, L: int = 10, device: str = "cpu") -> None:
    """Compare the uniform grid, the Lloyd–Max grid, and the trellis on Gaussian weights at 4.5 bits."""
    from .grid import dequantize, quantize
    from .grids import IQ4NLGrid, Q4_0Grid, fit_codebook

    torch.manual_seed(0)
    w = torch.randn(rows, cols, device=device) * 0.02
    energy = w.pow(2).mean()
    for name, grid in (("Q4_0", Q4_0Grid()), ("IQ4_NL", IQ4NLGrid()), ("CB4", fit_codebook(w))):
        idx, d = quantize(grid.to(w.device), w, search=True)
        err = (dequantize(grid.to(w.device), idx, d) - w).pow(2).mean()
        print(f"{name:8s} relative mse {err / energy:.5f}  ({10 * torch.log10(energy / err):.2f} dB)")
    base = IQ4NLGrid().to(w.device)
    _, d = quantize(base, w, search=True)
    tr = Trellis.gaussian(L, device=w.device)
    codes, heads, tr = quantize_trellis(w, d.to(torch.float32), tr)
    err = (dequantize_trellis(codes, heads, d, tr) - w).pow(2).mean()
    print(f"TQ4 L={L:<3d} relative mse {err / energy:.5f}  ({10 * torch.log10(energy / err):.2f} dB)")


if __name__ == "__main__":
    import sys

    self_test(L=int(sys.argv[1]) if len(sys.argv) > 1 else 10, device=sys.argv[2] if len(sys.argv) > 2 else "cpu")
