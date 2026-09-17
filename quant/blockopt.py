"""Block reconstruction with the quantizer in the loop.

For one decoder layer, gradient descent moves every parameter of the
working copy at once: the latent weights of the 4-bit matrices, their block
scales, the codebook levels, the norms, and the small F32 tensors. The
objective is the layer output on the quantized-flow input against the FP
layer output on the FP-flow input. The 4-bit matrices go through a
straight-through quantizer on their grid in the forward pass, thus the
rounding is inside the objective. The greedy column rounding is only an
initialization (Block-AP of EfficientQAT, BRECQ).

The head gets the same treatment with the KL divergence of the logits as
the objective, on random token subsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .grid import BLOCK, dequantize, quantize
from .grids import CodebookGrid, Grid

# The small tensors that stay fixed: their GGUF form is not a plain copy of the checkpoint value.
FROZEN = ("conv1d", "A_log", "dt_bias")


@dataclass
class OptOptions:
    """The knobs of the block optimization."""

    epochs: int = 8
    batch: int = 4
    lr_weight: float = 1e-5
    lr_scale: float = 1e-4
    lr_other: float = 1e-4
    lr_levels: float = 1e-2
    head_tokens: int = 1024
    head_steps: int = 300


class STELinear(nn.Module):
    """A linear whose weight is quantized on the grid in the forward pass.

    The forward value is the exact quantized weight d · level[idx]. The
    gradient reaches the latent weight as the identity (straight-through),
    the log-scales and the levels through the quantized value with the
    assignment held fixed.
    """

    def __init__(self, weight: torch.Tensor, d: torch.Tensor, grid: Grid, learn_levels: bool) -> None:
        super().__init__()
        self.rows, self.cols = weight.shape
        self.weight = nn.Parameter(weight.detach().clone().to(torch.float32))
        self.log_d = nn.Parameter(d.detach().to(torch.float32).clamp_min(1e-12).log())
        levels = grid.levels.detach().clone()
        self.levels = nn.Parameter(levels) if learn_levels else levels
        self.grid = grid

    def current_grid(self) -> Grid:
        """The grid with the current levels, integer for a codebook."""
        levels = self.levels.detach()
        if isinstance(self.grid, CodebookGrid):
            levels = torch.clamp(torch.round(levels), -127, 127)
        return _grid_like(self.grid, levels)

    def quantized_weight(self) -> torch.Tensor:
        d = self.log_d.exp()
        blocks = self.weight.view(self.rows, -1, BLOCK)
        idx = _grid_like(self.grid, self.levels.detach()).round(blocks.detach() / d.detach()[..., None])
        w_q = d[..., None] * self.levels[idx]
        w_ste = blocks + (w_q - blocks).detach()
        return (w_ste + (w_q - w_q.detach())).view(self.rows, self.cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight())

    @torch.no_grad()
    def finalize(self) -> tuple[Grid, torch.Tensor, torch.Tensor]:
        """The final grid, indices and F16 scales, with integer levels for a codebook."""
        grid = self.current_grid()
        d = self.log_d.exp().to(torch.float16).to(torch.float32)
        idx = grid.round(self.weight.view(self.rows, -1, BLOCK) / d[..., None]).view(self.rows, self.cols)
        return grid, idx.to(torch.int8), d.to(torch.float16)


def _grid_like(grid: Grid, levels: torch.Tensor) -> Grid:
    """A grid of the same class with other levels."""
    out = grid.__class__.__new__(grid.__class__)
    Grid.__init__(out, levels)
    out.name = grid.name
    return out


def _split(module: nn.Module, rel: str) -> tuple[nn.Module, str]:
    """The parent module and the attribute name of a relative path."""
    parts = rel.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def wrap_layer(layer: nn.Module, targets: dict[str, tuple[Grid, torch.Tensor]]) -> dict[str, tuple[nn.Module, str, nn.Linear, STELinear]]:
    """Replace the target linears of the layer by STE linears. Returns what to restore."""
    wrapped = {}
    for rel, (grid, h_diag) in targets.items():
        parent, attr = _split(layer, rel)
        lin: nn.Linear = getattr(parent, attr)
        _, d = quantize(grid, lin.weight.data, weights=h_diag, search=True)
        ste = STELinear(lin.weight.data, d.to(torch.float32), grid, isinstance(grid, CodebookGrid))
        setattr(parent, attr, ste)
        wrapped[rel] = (parent, attr, lin, ste)
    return wrapped


def unwrap_layer(wrapped: dict[str, tuple[nn.Module, str, nn.Linear, STELinear]]) -> dict[str, tuple[Grid, torch.Tensor, torch.Tensor]]:
    """Put the original linears back with the final quantized weights. Returns (grid, idx, d) per target."""
    out = {}
    for rel, (parent, attr, lin, ste) in wrapped.items():
        grid, idx, d = ste.finalize()
        lin.weight.data.copy_(dequantize(grid, idx, d))
        setattr(parent, attr, lin)
        out[rel] = (grid, idx, d)
    return out


def param_groups(layer: nn.Module, wrapped: dict, opts: OptOptions) -> list[dict]:
    """Adam groups: latent weights, log-scales, levels, everything else in the layer."""
    weights, scales, levels, others = [], [], [], []
    ste_ids = set()
    for _, _, _, ste in wrapped.values():
        weights.append(ste.weight)
        scales.append(ste.log_d)
        if isinstance(ste.levels, nn.Parameter):
            levels.append(ste.levels)
        ste_ids.update(id(p) for p in ste.parameters())
    for name, p in layer.named_parameters():
        if id(p) not in ste_ids and not any(f in name for f in FROZEN):
            others.append(p)
    groups = [{"params": weights, "lr": opts.lr_weight}, {"params": scales, "lr": opts.lr_scale},
              {"params": others, "lr": opts.lr_other}]
    if levels:
        groups.append({"params": levels, "lr": opts.lr_levels})
    return groups


def cosine(step: int, total: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * step / max(total, 1)))


def optimize_layer(step, li: int, targets: dict[str, tuple[Grid, torch.Tensor]], opts: OptOptions) -> dict:
    """Optimize layer li of the working copy toward the reference outputs in ``step.ref_in``.

    ``step`` is a Lockstep whose reference cache already holds the outputs
    of layer li (advance the reference first) and whose working cache holds
    the inputs. Returns (grid, idx, d) per target and prints the loss.
    """
    layer = step.work.model.layers[li]
    wrapped = wrap_layer(layer, targets)
    for name, p in layer.named_parameters():
        p.requires_grad_(not any(f in name for f in FROZEN))
    opt = torch.optim.Adam(param_groups(layer, wrapped, opts), betas=(0.9, 0.99))
    base_lrs = [g["lr"] for g in opt.param_groups]
    n = step.n_seq
    order = torch.randperm(n, generator=torch.Generator().manual_seed(li))
    total = opts.epochs * math.ceil(n / opts.batch)
    it = 0
    first = last = 0.0
    for _ in range(opts.epochs):
        for s in range(0, n, opts.batch):
            rows = order[s:s + opts.batch]
            x = step.work_in[rows]
            target = step.ref_in[rows]
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * cosine(it, total)
            with torch.enable_grad():
                out = step.run_layer(layer, li, x)
                loss = (out - target).pow(2).mean() / target.pow(2).mean().clamp_min(1e-12)
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if it == 0:
                first = loss.item()
            last = loss.item()
            it += 1
    for p in layer.parameters():
        p.requires_grad_(False)
    result = unwrap_layer(wrapped)
    print(f"layer {li:2d} block optimization: relative loss {first:.5f} -> {last:.5f} in {it} steps", flush=True)
    return result


def optimize_head(step, grid: Grid, h_diag: torch.Tensor, opts: OptOptions) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """Optimize the head and the final norm of the working copy on the KL of the logits.

    The caches hold the final-layer outputs of both copies. Each step takes
    ``opts.head_tokens`` random tokens.
    """
    work, ref = step.work, step.ref
    d_model = step.cfg.hidden_size
    ref_h = ref.model.norm(step.ref_in.view(-1, d_model))
    lin = work.lm_head
    _, d = quantize(grid, lin.weight.data, weights=h_diag, search=True)
    ste = STELinear(lin.weight.data, d.to(torch.float32), grid, isinstance(grid, CodebookGrid))
    norm = work.model.norm
    norm.weight.requires_grad_(True)
    groups = [{"params": [ste.weight], "lr": opts.lr_weight}, {"params": [ste.log_d], "lr": opts.lr_scale},
              {"params": [norm.weight], "lr": opts.lr_other}]
    if isinstance(ste.levels, nn.Parameter):
        groups.append({"params": [ste.levels], "lr": opts.lr_levels})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_tok = step.n_seq * step.seq_len
    gen = torch.Generator(device="cpu").manual_seed(0)
    first = last = 0.0
    work_flat = step.work_in.view(-1, d_model)
    for it in range(opts.head_steps):
        rows = torch.randint(0, n_tok, (opts.head_tokens,), generator=gen).to(step.device)
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * cosine(it, opts.head_steps)
        with torch.enable_grad():
            logits_q = ste(norm(work_flat[rows]))
            with torch.no_grad():
                log_p = F.log_softmax(ref.lm_head(ref_h[rows]), -1)
            log_q = F.log_softmax(logits_q, -1)
            loss = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if it == 0:
            first = loss.item()
        last = loss.item()
    norm.weight.requires_grad_(False)
    grid_f, idx, d_f = ste.finalize()
    lin.weight.data.copy_(dequantize(grid_f, idx, d_f))
    print(f"head block optimization: KL {first:.5f} -> {last:.5f} in {opts.head_steps} steps", flush=True)
    return grid_f, idx, d_f
