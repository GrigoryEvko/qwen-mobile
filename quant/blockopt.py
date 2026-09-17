"""Block reconstruction with the quantizer in the loop.

For one decoder layer, gradient descent moves every parameter of the
working copy at once: the latent weights of the 4-bit matrices, their block
scales, the codebook levels, the low-rank correction factors, the norms,
and the small F32 tensors. The objective is the layer output on the
quantized-flow input against the FP layer output on the FP-flow input. The
4-bit matrices go through a straight-through quantizer on their grid in the
forward pass, thus the rounding is inside the objective. The greedy column
rounding is only an initialization (Block-AP of EfficientQAT, BRECQ).

The low-rank correction W_q + B·A starts from the rank-r truncation of the
residual in the Hessian-weighted norm (CALDERA, LQER): with H = L·Lᵀ and
E = W_ref − W_q, the SVD of E·L gives the factors. The factors then train
with the rest. They export as a GGUF LoRA adapter with alpha = rank.

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
    freeze_weights: bool = False
    lr_weight: float = 1e-5
    lr_scale: float = 1e-4
    lr_other: float = 1e-4
    lr_levels: float = 1e-2
    rank: int = 0
    head_rank: int = 0
    head_tokens: int = 512
    head_steps: int = 400


def _grid_like(grid: Grid, levels: torch.Tensor) -> Grid:
    """A grid of the same class with other levels."""
    out = grid.__class__.__new__(grid.__class__)
    Grid.__init__(out, levels)
    out.name = grid.name
    return out


def weighted_low_rank(residual: torch.Tensor, hessian: torch.Tensor, rank: int,
                      damp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """The rank-r factors (a [r, cols], b [rows, r]) of the residual in the Hessian norm.

    Minimizes ‖(E − b·a)·L‖ with H = L·Lᵀ, thus the correction spends its
    rank on the input directions with energy. Complexity is O(rows · cols²).
    """
    h = hessian.to(torch.float32)
    h = h + damp * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], device=h.device)
    low = torch.linalg.cholesky(h)
    u, s, vh = torch.linalg.svd(residual.to(torch.float32) @ low, full_matrices=False)
    b = u[:, :rank] * s[:rank]
    # a·L = vh_r, thus Lᵀ·aᵀ = vh_rᵀ.
    a = torch.linalg.solve_triangular(low.T, vh[:rank].T, upper=True).T
    return a.contiguous(), b.contiguous()


class STELinear(nn.Module):
    """A linear whose weight is quantized on the grid in the forward pass.

    The forward value is the exact quantized weight d · level[idx], plus the
    low-rank term when a rank is set. The gradient reaches the latent
    weight as the identity (straight-through), the log-scales and the
    levels through the quantized value with the assignment held fixed.
    """

    def __init__(self, weight: torch.Tensor, d: torch.Tensor, grid: Grid, learn_levels: bool,
                 low_rank: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.rows, self.cols = weight.shape
        self.weight = nn.Parameter(weight.detach().clone().to(torch.float32))
        # A Q4_0 scale carries the sign of the signed maximum. The sign stays, the magnitude learns.
        d = d.detach().to(torch.float32)
        self.sign = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
        self.log_d = nn.Parameter(d.abs().clamp_min(1e-12).log())
        levels = grid.levels.detach().clone()
        self.levels = nn.Parameter(levels) if learn_levels else levels
        self.grid = grid
        self.lora_a = nn.Parameter(low_rank[0].clone()) if low_rank else None
        self.lora_b = nn.Parameter(low_rank[1].clone()) if low_rank else None

    def scale(self) -> torch.Tensor:
        return self.sign * self.log_d.exp()

    def current_grid(self) -> Grid:
        """The grid with the current levels, integer for a codebook."""
        levels = self.levels.detach()
        if isinstance(self.grid, CodebookGrid):
            levels = torch.clamp(torch.round(levels), -127, 127)
        return _grid_like(self.grid, levels)

    def quantized_weight(self) -> torch.Tensor:
        d = self.scale()
        blocks = self.weight.view(self.rows, -1, BLOCK)
        # The grid rounds on the sorted levels, thus the lookup must use the same order when the levels learn.
        levels = self.levels.sort().values if isinstance(self.levels, nn.Parameter) else self.levels
        idx = _grid_like(self.grid, levels.detach()).round(blocks.detach() / d.detach()[..., None])
        w_q = d[..., None] * levels[idx]
        w_ste = blocks + (w_q - blocks).detach()
        return (w_ste + (w_q - w_q.detach())).view(self.rows, self.cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.quantized_weight())
        if self.lora_a is not None:
            y = y + F.linear(F.linear(x, self.lora_a), self.lora_b)
        return y

    @torch.no_grad()
    def finalize(self) -> "Solved":
        """The final grid, indices, F16 scales, and the low-rank factors."""
        grid = self.current_grid()
        d = self.scale().to(torch.float16).to(torch.float32)
        idx = grid.round(self.weight.view(self.rows, -1, BLOCK) / d[..., None]).view(self.rows, self.cols)
        low_rank = None
        if self.lora_a is not None:
            low_rank = (self.lora_a.detach().clone(), self.lora_b.detach().clone())
        return Solved(grid, idx.to(torch.int8), d.to(torch.float16), low_rank)


@dataclass
class Solved:
    """One solved matrix: the grid, the indices, the F16 scales, the optional factors (a, b)."""

    grid: Grid
    idx: torch.Tensor
    d: torch.Tensor
    low_rank: tuple[torch.Tensor, torch.Tensor] | None = None

    def dequantized(self) -> torch.Tensor:
        """The float32 weight that the runtime computes, low-rank term included."""
        w = dequantize(self.grid, self.idx, self.d)
        if self.low_rank is not None:
            w = w + self.low_rank[1] @ self.low_rank[0]
        return w


@dataclass
class Target:
    """One matrix to optimize: its grid, the Hessian of its input, the reference weight, the solved scales.

    ``d`` [rows, cols // 32] are the block scales of the initial rounding.
    The working weight is exactly on the grid with them, thus the STE starts
    from the solved rounding. Without ``d`` the scale search runs again.
    """

    grid: Grid
    hessian: torch.Tensor
    w_ref: torch.Tensor
    d: torch.Tensor | None = None


def _split(module: nn.Module, rel: str) -> tuple[nn.Module, str]:
    """The parent module and the attribute name of a relative path."""
    parts = rel.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def make_ste(lin: nn.Linear, target: Target, rank: int) -> STELinear:
    """The STE module of a linear, with the solved scales and the factors from the weighted SVD.

    With the solved scales the weight is already on the grid: a new search
    could not give the scale back, because its candidates are discrete, and
    the rounding would move. Without them the search picks the scales.
    """
    w = lin.weight.data
    if target.d is not None:
        d = target.d.to(torch.float32)
        w_q = w
    else:
        idx, d = quantize(target.grid, w, weights=torch.diag(target.hessian), search=True)
        w_q = dequantize(target.grid, idx, d)
    low_rank = None
    if rank > 0:
        low_rank = weighted_low_rank(target.w_ref.to(torch.float32) - w_q, target.hessian, rank)
    return STELinear(w, d, target.grid, isinstance(target.grid, CodebookGrid), low_rank)


def wrap_layer(layer: nn.Module, targets: dict[str, Target], rank: int) -> dict[str, tuple[nn.Module, str, nn.Linear, STELinear]]:
    """Replace the target linears of the layer by STE linears. Returns what to restore."""
    wrapped = {}
    for rel, target in targets.items():
        parent, attr = _split(layer, rel)
        lin: nn.Linear = getattr(parent, attr)
        ste = make_ste(lin, target, rank)
        setattr(parent, attr, ste)
        wrapped[rel] = (parent, attr, lin, ste)
    return wrapped


def unwrap_layer(wrapped: dict[str, tuple[nn.Module, str, nn.Linear, STELinear]]) -> dict[str, Solved]:
    """Put the original linears back with the final weights (low-rank term folded in). Returns the solutions."""
    out = {}
    for rel, (parent, attr, lin, ste) in wrapped.items():
        solved = ste.finalize()
        lin.weight.data.copy_(solved.dequantized())
        setattr(parent, attr, lin)
        out[rel] = solved
    return out


def param_groups(layer: nn.Module, wrapped: dict, opts: OptOptions) -> list[dict]:
    """Adam groups: latent weights, log-scales, levels, low-rank factors, everything else in the layer."""
    weights, scales, levels, factors, others = [], [], [], [], []
    ste_ids = set()
    for _, _, _, ste in wrapped.values():
        weights.append(ste.weight)
        scales.append(ste.log_d)
        if isinstance(ste.levels, nn.Parameter):
            levels.append(ste.levels)
        if ste.lora_a is not None:
            factors += [ste.lora_a, ste.lora_b]
        ste_ids.update(id(p) for p in ste.parameters())
    for name, p in layer.named_parameters():
        if id(p) not in ste_ids and not any(f in name for f in FROZEN):
            others.append(p)
    groups = [{"params": scales, "lr": opts.lr_scale}, {"params": others, "lr": opts.lr_other}]
    if opts.freeze_weights:
        for w in weights:
            w.requires_grad_(False)
    else:
        groups.append({"params": weights, "lr": opts.lr_weight})
    if levels:
        groups.append({"params": levels, "lr": opts.lr_levels})
    if factors:
        groups.append({"params": factors, "lr": opts.lr_other})
    return groups


def cosine(step: int, total: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * step / max(total, 1)))


def optimize_layer(step, li: int, targets: dict[str, Target], opts: OptOptions) -> dict[str, Solved]:
    """Optimize layer li of the working copy toward the reference outputs in ``step.ref_in``.

    ``step`` is a Lockstep whose reference cache already holds the outputs
    of layer li (advance the reference first) and whose working cache holds
    the inputs. Returns the solution per target and prints the loss.
    """
    layer = step.work.model.layers[li]
    wrapped = wrap_layer(layer, targets, opts.rank)
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
    print(f"layer {li:2d} block optimization: relative loss {first:.5f} -> {last:.5f} in {it} steps"
          f"{f', rank {opts.rank}' if opts.rank else ''}", flush=True)
    return result


def optimize_head(step, target: Target, opts: OptOptions) -> Solved:
    """Optimize the head and the final norm of the working copy on the KL of the logits.

    The caches hold the final-layer outputs of both copies. Each step takes
    ``opts.head_tokens`` random tokens.
    """
    work, ref = step.work, step.ref
    d_model = step.cfg.hidden_size
    ref_flat = step.ref_in.view(-1, d_model)
    lin = work.lm_head
    torch.cuda.empty_cache()
    ste = make_ste(lin, target, opts.head_rank)
    norm = work.model.norm
    norm.weight.requires_grad_(True)
    groups = [{"params": [ste.log_d], "lr": opts.lr_scale}, {"params": [norm.weight], "lr": opts.lr_other}]
    if opts.freeze_weights:
        ste.weight.requires_grad_(False)
    else:
        groups.append({"params": [ste.weight], "lr": opts.lr_weight})
    if isinstance(ste.levels, nn.Parameter):
        groups.append({"params": [ste.levels], "lr": opts.lr_levels})
    if ste.lora_a is not None:
        groups.append({"params": [ste.lora_a, ste.lora_b], "lr": opts.lr_other})
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
                log_p = F.log_softmax(ref.lm_head(ref.model.norm(ref_flat[rows])), -1)
            log_q = F.log_softmax(logits_q, -1)
            loss = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if it == 0:
            first = loss.item()
        last = loss.item()
    norm.weight.requires_grad_(False)
    solved = ste.finalize()
    lin.weight.data.copy_(solved.dequantized())
    print(f"head block optimization: KL {first:.5f} -> {last:.5f} in {opts.head_steps} steps", flush=True)
    return solved
