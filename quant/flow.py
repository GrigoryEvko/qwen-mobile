"""The quantized-flow driver, one decoder layer at a time, in lockstep with the FP model.

Two copies of the transformed checkpoint sit on the store device in float32:
the reference, whose function never changes, and the working copy that
receives the quantized weights. The driver caches the inputs of the current
layer for the whole calibration set from both copies, thus each sub-pass
runs one decoder layer and not the model. With the copies on the CPU and a
compute device, each layer moves to the device for its passes and back,
thus the 2B calibrates on a GPU with 6 GB.

A tied head (``tie_head``) reads the final norm through the dense map M of
the transform in both copies: logits = E'·M·norm(h). Its pack is
``token_embd.weight``, and the working embedding holds the rounded rows,
thus the layers see the lookup of the file.

Every fold is a re-parametrization that keeps the function, and the driver
applies it to both copies. Thus the two copies always share coordinates, the
cross moment G = mean(x̃ᵀx) is direct, and the reference at the end is the
folded FP model that the export reads for the unsolved tensors.

Per layer, in this order:
1. The mixer input group (GDN: qkv, z, a, b. Attention: q, k, v). Moments,
   column scales folded into the input norm, solve.
2. The mixer output projection (GDN out_proj through the shared gated norm,
   attention o_proj through the v_proj rows). Moments, scales, solve.
3. gate/up. Moments of the gate/up input and of the down_proj input. A
   permutation of the intermediate channels by input energy, their scales
   through the up_proj rows, the gate/up scales through the post-attention
   norm. Solve gate and up.
4. down_proj: moments on the quantized flow, solve.
Then the head on the final-norm outputs, with its scales in the final norm.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from transformers.masking_utils import create_causal_mask

from .blockopt import FROZEN, OptOptions, Solved, Target, optimize_head, optimize_layer
from .grid import ROW_CHUNK, dequantize, q8_0_dequantize, q8_0_quantize, quantize
from .grids import Grid, IQ4NLGrid, Q4_0Grid, fit_codebook
from .names import to_gguf
from .plan import Plan
from .scale import (head_channel_share, kv_group_rows, kv_group_share, permuted_moments, scaled_moments,
                    search_column_scales)
from .solver import solve_grid

GDN_IN = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b")
ATTN_IN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
ZERO_CENTERED = ("attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                 "output_norm.weight")


@dataclass
class Options:
    """The knobs of one calibration run.

    ``method`` is ``blockopt`` (block reconstruction by gradient with the
    quantizer in the loop) or ``solve`` (the column rounding only).
    ``init`` is the rounding before the optimization: ``rtn`` with the scale
    search, ``qronos``, or ``gptq``.
    """

    method: str = "blockopt"
    init: str = "rtn"
    scale: bool = True
    permute_mlp: bool = True
    mismatch: str = "model"
    damp: float = 0.01
    refit_damp: float = 1e-6
    batch: int = 8
    opt: OptOptions = field(default_factory=OptOptions)


def q8_round_trip_(w: torch.Tensor) -> None:
    """Replace ``w`` by its Q8_0 round trip, in place."""
    q, d = q8_0_quantize(w)
    w.copy_(q8_0_dequantize(q, d))


def grid_round_trip_(w: torch.Tensor, grid: Grid) -> None:
    """Replace ``w`` by its round trip through ``grid`` with the scale search, in place.

    The work runs in row chunks on the device of the grid, thus the
    embedding of 248320 rows needs no full-size temporary there.
    """
    for r in range(0, w.shape[0], ROW_CHUNK):
        idx, d = quantize(grid, w[r:r + ROW_CHUNK], search=True)
        w[r:r + ROW_CHUNK].copy_(dequantize(grid, idx, d))


def tie_head(model: nn.Module, rot: torch.Tensor) -> None:
    """Make a tied HF model read the final norm through M: lm_head becomes Sequential(Linear(M), lm_head).

    The forward of transformers calls ``lm_head`` on the normed hidden
    state, thus the logits become E'·M·norm(h). Raises ValueError when the
    head of the model does not share the embedding.
    """
    lin = model.lm_head
    if not isinstance(lin, nn.Linear) or lin.weight is not model.model.embed_tokens.weight:
        raise ValueError("tie_head needs a model whose lm_head shares the embedding (tie_word_embeddings)")
    d = lin.weight.shape[1]
    m = nn.Linear(d, d, bias=False)
    m.weight = nn.Parameter(rot.to(lin.weight.device, torch.float32).clone(), requires_grad=False)
    model.lm_head = nn.Sequential(m, lin)


def head_linear(model: nn.Module) -> nn.Linear:
    """The big matrix of the head: the last module of a tied head, or the head."""
    return model.lm_head[-1] if isinstance(model.lm_head, nn.Sequential) else model.lm_head


def head_rot(model: nn.Module) -> torch.Tensor | None:
    """The dense map M of a tied head, or None for an untied head."""
    return model.lm_head[0].weight.data if isinstance(model.lm_head, nn.Sequential) else None


def is_tied(model: nn.Module) -> bool:
    """True when the head of the model is the embedding tensor."""
    return head_linear(model).weight is model.model.embed_tokens.weight


def recurrent_mask(text_model: nn.Module, kw: dict) -> torch.Tensor | None:
    """The mask of the linear-attention layers, from the transformers version at hand.

    Some transformers versions give ``create_recurrent_attention_mask`` in
    the Qwen3.5 module, other versions a method of the text model. Without
    padding the mask is None in the two.
    """
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import create_recurrent_attention_mask
    except ImportError:
        return text_model._update_linear_attn_mask(kw["attention_mask"], kw["past_key_values"])
    return create_recurrent_attention_mask(**kw)


class PairedMoments:
    """H = mean(x̃ᵀx̃) and G = mean(x̃ᵀx) of one linear.

    x̃ comes from the working copy and x from the reference. The reference
    runs first on each batch, thus its input waits for the working hook.
    """

    def __init__(self, cols: int, device: torch.device) -> None:
        self.cols = cols
        self.h = torch.zeros(cols, cols, dtype=torch.float32, device=device)
        self.g = torch.zeros(cols, cols, dtype=torch.float32, device=device)
        self.n = 0
        self.ref_x: torch.Tensor | None = None

    def take_ref(self, module: nn.Module, inputs: tuple, output) -> None:
        self.ref_x = inputs[0].detach().reshape(-1, self.cols).to(torch.float32)

    def take_work(self, module: nn.Module, inputs: tuple, output) -> None:
        x = inputs[0].detach().reshape(-1, self.cols).to(torch.float32)
        assert self.ref_x is not None and self.ref_x.shape == x.shape, "the reference input is missing"
        self.h.addmm_(x.T, x)
        self.g.addmm_(x.T, self.ref_x)
        self.n += x.shape[0]
        self.ref_x = None

    def mean(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.h / max(self.n, 1), self.g / max(self.n, 1)


class Lockstep:
    """The two copies, the token ids, and the cached inputs of the current layer.

    The copies and the caches live on the store device, the device of the
    models. The compute runs on ``device``: a layer moves there for its
    passes through ``on_device`` and back after them. With one device the
    moves are no-ops. The caches are [n_seq, seq_len, hidden] float32, one
    per copy. ``advance`` moves them through one layer.
    """

    def __init__(self, ref, work, ids: torch.Tensor, batch: int, device: torch.device | None = None) -> None:
        self.ref, self.work = ref, work
        self.ids = ids
        self.batch = batch
        self.store = ref.device
        self.device = device or self.store
        self.cfg = ref.config
        self.n_seq, self.seq_len = ids.shape
        shape = (self.n_seq, self.seq_len, self.cfg.hidden_size)
        self.ref_in = torch.empty(shape, dtype=torch.float32, device=self.store)
        self.work_in = torch.empty(shape, dtype=torch.float32, device=self.store)
        self._context: dict[int, tuple] = {}
        # The rotary module holds inv_freq on its device, and the positions live on the compute device.
        ref.model.rotary_emb.to(self.device)
        self.embed(ref, self.ref_in)
        self.embed_work()

    @torch.no_grad()
    def embed(self, model: nn.Module, cache: torch.Tensor) -> None:
        """Fill ``cache`` with the embedding of the tokens by ``model``."""
        for i, tok in self.batches():
            cache[i:i + tok.shape[0]] = model.model.embed_tokens(tok)

    def embed_work(self) -> None:
        """Reset the working cache to the lookup of the working embedding, which can hold rounded rows."""
        self.embed(self.work, self.work_in)

    def batches(self):
        """(start, ids) of each batch, on the store device."""
        for i in range(0, self.n_seq, self.batch):
            yield i, self.ids[i:i + self.batch].to(self.store)

    @contextmanager
    def on_device(self, module: nn.Module) -> Iterator[nn.Module]:
        """``module`` on the compute device inside the block, back on the store after it.

        A module that is on the compute device already stays there, thus
        the blocks nest.
        """
        if self.device == self.store or next(module.parameters()).device == self.device:
            yield module
            return
        module.to(self.device)
        try:
            yield module
        finally:
            module.to(self.store)

    def context(self, n: int) -> tuple:
        """The rotary embeddings, the two masks, and the text positions for n sequences."""
        if n not in self._context:
            length = self.seq_len
            pos = torch.arange(length, device=self.device).view(1, 1, -1).expand(4, n, -1)
            hidden = self.ref_in[:n].to(self.device)
            rotary = self.ref.model.rotary_emb(hidden, pos[1:])
            kw = dict(config=self.cfg, inputs_embeds=hidden, attention_mask=None, past_key_values=None,
                      position_ids=pos[0])
            masks = {"full_attention": create_causal_mask(**kw), "linear_attention": recurrent_mask(self.ref.model, kw)}
            self._context[n] = (rotary, masks, pos[0])
        return self._context[n]

    def run_layer(self, layer: nn.Module, li: int, hidden: torch.Tensor) -> torch.Tensor:
        """One decoder layer on [n, seq_len, hidden]. The input moves to the compute device, the output stays there."""
        rotary, masks, text_pos = self.context(hidden.shape[0])
        return layer(hidden.to(self.device), position_embeddings=rotary,
                     attention_mask=masks[self.cfg.layer_types[li]], position_ids=text_pos)

    def layers(self, li: int) -> tuple[nn.Module, nn.Module]:
        return self.ref.model.layers[li], self.work.model.layers[li]

    @torch.no_grad()
    def collect(self, li: int, rels: list[str]) -> dict[str, PairedMoments]:
        """The paired moments of the inputs of the linears ``rels`` of layer li, over the calibration set."""
        ref_layer, work_layer = self.layers(li)
        moments: dict[str, PairedMoments] = {}
        handles = []
        for rel in rels:
            lin_ref, lin_work = ref_layer.get_submodule(rel), work_layer.get_submodule(rel)
            m = PairedMoments(lin_work.in_features, self.device)
            handles += [lin_ref.register_forward_hook(m.take_ref), lin_work.register_forward_hook(m.take_work)]
            moments[rel] = m
        with self.on_device(ref_layer), self.on_device(work_layer):
            for i, tok in self.batches():
                n = tok.shape[0]
                self.run_layer(ref_layer, li, self.ref_in[i:i + n])
                self.run_layer(work_layer, li, self.work_in[i:i + n])
        for h in handles:
            h.remove()
        return moments

    @torch.no_grad()
    def advance_ref(self, li: int) -> None:
        """Replace the cached reference inputs by the reference outputs of layer li."""
        layer = self.ref.model.layers[li]
        with self.on_device(layer):
            for i, tok in self.batches():
                n = tok.shape[0]
                self.ref_in[i:i + n].copy_(self.run_layer(layer, li, self.ref_in[i:i + n]))

    @torch.no_grad()
    def advance_work(self, li: int) -> None:
        """Replace the cached working inputs by the working outputs of layer li."""
        layer = self.work.model.layers[li]
        with self.on_device(layer):
            for i, tok in self.batches():
                n = tok.shape[0]
                self.work_in[i:i + n].copy_(self.run_layer(layer, li, self.work_in[i:i + n]))

    def advance(self, li: int) -> None:
        """Move both caches through layer li."""
        self.advance_ref(li)
        self.advance_work(li)

    @torch.no_grad()
    def final_norm(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The final-norm outputs of both copies, [n_seq · seq_len, hidden]."""
        d = self.cfg.hidden_size
        ref_h = torch.empty(self.n_seq * self.seq_len, d, dtype=torch.float32, device=self.device)
        work_h = torch.empty_like(ref_h)
        for i, tok in self.batches():
            n = tok.shape[0]
            ref_h[i * self.seq_len:(i + n) * self.seq_len] = self.ref.model.norm(self.ref_in[i:i + n]).reshape(-1, d)
            work_h[i * self.seq_len:(i + n) * self.seq_len] = self.work.model.norm(self.work_in[i:i + n]).reshape(-1, d)
        return ref_h, work_h


class Quantizer:
    """Solve the plan on the working copy of a Lockstep, layer by layer."""

    def __init__(self, step: Lockstep, plan: Plan, out_dir: Path, opts: Options) -> None:
        self.step = step
        self.plan = plan
        self.out_dir = out_dir
        self.opts = opts
        self.cfg = step.cfg
        self.tied = is_tied(step.work)
        self.fixed = {"Q4_0": Q4_0Grid().to(step.device), "IQ4_NL": IQ4NLGrid().to(step.device)}
        # The grid, the input Hessian and the block scales of each solved matrix, by (layer, relative name).
        # The head is layer -1. The Hessian and the scales live until the block optimization of the layer.
        self.grids: dict[tuple[int, str], Grid] = {}
        self.hess: dict[tuple[int, str], torch.Tensor] = {}
        self.scales: dict[tuple[int, str], torch.Tensor] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- grids ---

    def search_grid(self, kind: str) -> Grid:
        """The grid of the scale search. A codebook is not fit yet at that time, the IQ4_NL table stands in."""
        return self.fixed["Q4_0" if kind == "Q4_0" else "IQ4_NL"]

    def grid_for(self, kind: str, w: torch.Tensor, h_diag: torch.Tensor) -> Grid:
        """The grid of one matrix: fixed by type, or a codebook fit on the matrix."""
        if kind == "CB4":
            return fit_codebook(w, h_diag)
        return self.fixed[kind]

    # --- names and modules ---

    def name(self, li: int, rel: str) -> str:
        gguf = to_gguf(f"model.layers.{li}.{rel}.weight")
        if gguf is None:
            raise KeyError(f"no GGUF name for model.layers.{li}.{rel}.weight")
        return gguf

    def kind(self, li: int, rel: str) -> str:
        return self.plan.type_of(self.name(li, rel))

    def both(self, li: int, rel: str) -> tuple[nn.Module, nn.Module]:
        ref_layer, work_layer = self.step.layers(li)
        return ref_layer.get_submodule(rel), work_layer.get_submodule(rel)

    # --- folds, applied to both copies ---

    def scale_columns(self, li: int, rels: tuple[str, ...] | list[str], t: torch.Tensor) -> None:
        for rel in rels:
            for lin in self.both(li, rel):
                lin.weight.data.mul_(t[None, :])

    def scale_rows(self, li: int, rel: str, t: torch.Tensor) -> None:
        for lin in self.both(li, rel):
            lin.weight.data.div_(t[:, None])

    def scale_norm(self, li: int, rel: str, t: torch.Tensor) -> None:
        """The zero-centered norm supplies x/t: (1 + w) becomes (1 + w)/t."""
        for norm in self.both(li, rel):
            norm.weight.data.copy_((1.0 + norm.weight.data) / t - 1.0)

    def column_scales(self, li: int, rels: list[str], h: torch.Tensor,
                      share: torch.Tensor | None = None) -> torch.Tensor | None:
        """The column scales of the matrices ``rels``, or None. A class that is not on a 4-bit grid takes none.

        A Q8_0 or F32 matrix needs no column scale: its block scales carry
        256 levels. A scale for it would also move the rows of its
        producer, thus the search runs on the solved members only.
        """
        rels = [rel for rel in rels if self.kind(li, rel) in self.plan.solved_types()]
        if not self.opts.scale or not rels:
            return None
        ws = [self.both(li, rel)[1].weight.data for rel in rels]
        return search_column_scales(self.search_grid(self.kind(li, rels[0])), ws, torch.diag(h), share)

    # --- solving ---

    def save_pack(self, name: str, kind: str, grid: Grid, idx: torch.Tensor, d: torch.Tensor,
                  low_rank: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        extra = {}
        if low_rank is not None:
            extra = {"lora_a": low_rank[0].cpu().numpy(), "lora_b": low_rank[1].cpu().numpy()}
        np.savez(self.out_dir / f"{name}.npz", q=idx.cpu().numpy(), d=d.cpu().numpy().view(np.uint16),
                 levels=grid.levels.cpu().numpy(), kind=np.array(kind), **extra)

    def save_solved(self, name: str, kind: str, solved: Solved) -> None:
        self.save_pack(name, kind, solved.grid, solved.idx, solved.d, solved.low_rank)

    def solve(self, key: tuple[int, str], name: str, kind: str, lin: nn.Linear, h: torch.Tensor, g: torch.Tensor,
              tag: str, w_src: torch.Tensor | None = None) -> None:
        """The initial rounding of one matrix onto its grid. The block optimization moves it later.

        The solver reads ``w_src``, or the weight of ``lin``, and writes the
        rounded values into ``lin``. A tied head gives the reference
        embedding as ``w_src``, because the working embedding is on the
        grid already.
        """
        t0 = time.time()
        w = lin.weight.data if w_src is None else w_src
        grid = self.grid_for(kind, w, torch.diag(h))
        self.grids[key], self.hess[key] = grid, h.clone()
        if self.opts.init == "rtn":
            idx, d = quantize(grid, w, weights=torch.diag(h), search=True)
            err, change = float("nan"), 0.0
        else:
            cross = g if self.opts.init == "qronos" else None
            idx, d, err, change = solve_grid(w, h, grid, cross, damp=self.opts.damp, refit_damp=self.opts.refit_damp)
        self.scales[key] = d
        lin.weight.data.copy_(dequantize(grid, idx, d))
        self.save_pack(name, kind, grid, idx, d)
        print(f"{tag} {name:30s} {kind:6s} {tuple(lin.weight.shape)} {self.opts.init} err {err:.3e} "
              f"refit {change:.3f} in {time.time() - t0:4.1f}s", flush=True)

    def solve_or_round(self, li: int, rel: str, h: torch.Tensor, g: torch.Tensor) -> None:
        kind = self.kind(li, rel)
        lin = self.both(li, rel)[1]
        if kind in self.plan.solved_types():
            self.solve((li, rel), self.name(li, rel), kind, lin, h, g, f"layer {li:2d}")
        elif kind == "Q8_0":
            q8_round_trip_(lin.weight.data)

    def optimize(self, li: int) -> None:
        """Block reconstruction of layer li, then the Q8 members back on their grid."""
        self.step.advance_ref(li)
        targets = {rel: Target(self.grids[(l, rel)], self.hess[(l, rel)], self.both(li, rel)[0].weight.data,
                               self.scales[(l, rel)])
                   for (l, rel) in list(self.grids) if l == li}
        result = optimize_layer(self.step, li, targets, self.opts.opt)
        for rel, solved in result.items():
            self.grids[(li, rel)] = solved.grid
            self.save_solved(self.name(li, rel), self.kind(li, rel), solved)
        work_layer = self.step.layers(li)[1]
        for rel, lin in work_layer.named_modules():
            if isinstance(lin, nn.Linear) and self.kind(li, rel) == "Q8_0":
                q8_round_trip_(lin.weight.data)
        self.step.advance_work(li)
        for key in [k for k in self.hess if k[0] == li]:
            del self.hess[key]
            del self.scales[key]
        torch.cuda.empty_cache()

    def save_folds(self) -> None:
        """The GGUF-space values of the small tensors of the working copy, for the export.

        The optimization moves the norms, the gate projections and the Q8
        matrices. The zero-centered norms get their +1. The plan goes with
        them: the export refuses a different plan on the unfolded F16 GGUF,
        because the folds moved the coordinates of the solved classes. A
        tied head adds its dense map ``output_rot.weight``.
        """
        folds: dict[str, np.ndarray] = {}
        for full, p in self.step.work.named_parameters():
            gguf = to_gguf(full)
            if gguf is None or gguf == "token_embd.weight" or any(f in full for f in FROZEN):
                continue
            if self.plan.type_of(gguf) not in ("F32", "Q8_0"):
                continue
            v = p.detach().to("cpu", torch.float32)
            if gguf.endswith(ZERO_CENTERED):
                v = v + 1.0
            folds[gguf] = v.numpy()
        rot = head_rot(self.step.work)
        if rot is not None:
            folds["output_rot.weight"] = rot.to("cpu", torch.float32).numpy()
        np.savez(self.out_dir / "folds.npz", plan=np.array(json.dumps(asdict(self.plan))), **folds)
        print(f"wrote {len(folds)} small tensors to folds.npz", flush=True)

    # --- groups ---

    def mixer_input(self, li: int, members: tuple[str, ...], norm_rel: str,
                    defer: tuple[str, ...] = ()) -> tuple[torch.Tensor, torch.Tensor]:
        """The projections that read the input norm. ``defer`` waits for a later fold.

        Returns the moments of the (scaled) input, for the deferred members.
        """
        h, g = self.step.collect(li, [members[0]])[members[0]].mean()
        solved = [rel for rel in members if self.kind(li, rel) in self.plan.solved_types()]
        t = self.column_scales(li, solved, h)
        if t is not None:
            self.scale_columns(li, members, t)
            self.scale_norm(li, norm_rel, t)
            h, g = scaled_moments(h, g, t)
        for rel in members:
            if rel not in defer:
                self.solve_or_round(li, rel, h, g)
        return h, g

    def gdn_output(self, li: int) -> None:
        """out_proj, with its column scales in the shared weight of the gated norm."""
        rel = "linear_attn.out_proj"
        h, g = self.step.collect(li, [rel])[rel].mean()
        v_dim = self.cfg.linear_value_head_dim
        t = self.column_scales(li, [rel], h, head_channel_share(h.shape[0], v_dim, h.device))
        if t is not None:
            self.scale_columns(li, [rel], t)
            for norm in self.both(li, "linear_attn.norm"):
                norm.weight.data.div_(t[:v_dim])
            h, g = scaled_moments(h, g, t)
        self.solve_or_round(li, rel, h, g)

    def attention_output(self, li: int, h_in: torch.Tensor, g_in: torch.Tensor) -> None:
        """o_proj, with its column scales in the v_proj rows of the KV group of each head.

        ``h_in``, ``g_in`` are the moments of the mixer input: v_proj reads
        them, and a row scale does not change them.
        """
        rel = "self_attn.o_proj"
        h, g = self.step.collect(li, [rel])[rel].mean()
        heads, kv_heads, dim = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        t = self.column_scales(li, [rel], h, kv_group_share(h.shape[0], heads, kv_heads, dim, h.device))
        if t is not None:
            self.scale_columns(li, [rel], t)
            self.scale_rows(li, "self_attn.v_proj", kv_group_rows(t, heads, kv_heads, dim))
            h, g = scaled_moments(h, g, t)
        self.solve_or_round(li, "self_attn.v_proj", h_in, g_in)
        self.solve_or_round(li, rel, h, g)

    def mlp(self, li: int) -> None:
        gate, up, down = "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
        moments = self.step.collect(li, [gate, down])
        h_in, g_in = moments[gate].mean()
        h_dn, g_dn = moments[down].mean()
        if self.opts.permute_mlp:
            perm = torch.argsort(torch.diag(h_dn), descending=True)
            for rel in (gate, up):
                for lin in self.both(li, rel):
                    lin.weight.data.copy_(lin.weight.data[perm])
            for lin in self.both(li, down):
                lin.weight.data.copy_(lin.weight.data[:, perm])
            h_dn, g_dn = permuted_moments(h_dn, g_dn, perm)
        t_dn = self.column_scales(li, [down], h_dn)
        if t_dn is not None:
            self.scale_columns(li, [down], t_dn)
            self.scale_rows(li, up, t_dn)
        t_in = self.column_scales(li, [gate, up], h_in)
        if t_in is not None:
            self.scale_columns(li, (gate, up), t_in)
            self.scale_norm(li, "post_attention_layernorm", t_in)
            h_in, g_in = scaled_moments(h_in, g_in, t_in)
        self.solve_or_round(li, gate, h_in, g_in)
        self.solve_or_round(li, up, h_in, g_in)
        h, g = self.step.collect(li, [down])[down].mean()
        self.solve_or_round(li, down, h, g)

    # --- the head ---

    def head_name(self) -> str:
        """The GGUF name of the head pack: the embedding for a tied head."""
        return "token_embd.weight" if self.tied else "output.weight"

    @torch.no_grad()
    def head_moments(self, rot: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """H and G of the head input over the calibration set: the final-norm output, through M when tied."""
        step = self.step
        d = self.cfg.hidden_size
        h = torch.zeros(d, d, dtype=torch.float32, device=step.device)
        g = torch.zeros(d, d, dtype=torch.float32, device=step.device)
        norm_ref, norm_work = step.ref.model.norm, step.work.model.norm
        rot = None if rot is None else rot.to(step.device, torch.float32)
        with step.on_device(norm_ref), step.on_device(norm_work):
            for i, tok in step.batches():
                n = tok.shape[0]
                r = norm_ref(step.ref_in[i:i + n].to(step.device)).reshape(-1, d)
                w = norm_work(step.work_in[i:i + n].to(step.device)).reshape(-1, d)
                if rot is not None:
                    r, w = r @ rot.T, w @ rot.T
                h.addmm_(w.T, w)
                g.addmm_(w.T, r)
        h /= step.n_seq * step.seq_len
        g /= step.n_seq * step.seq_len
        return h, g

    def head(self, optimize: bool = True) -> None:
        """The head on the final-norm outputs of both copies.

        Untied: the column scales of the head go into the final norm. Tied:
        the head is the embedding E' and its input is M·norm(h). No column
        scale is possible then, because the rows must stay the rows of the
        lookup. With ``optimize`` the block optimization on the KL follows
        the solve, and the working embedding holds the result.
        """
        name = self.head_name()
        kind = self.plan.type_of(name)
        if kind not in self.plan.solved_types():
            return
        step = self.step
        rot = head_rot(step.work)
        h, g = self.head_moments(rot)
        heads = (head_linear(step.ref), head_linear(step.work))
        if self.opts.scale and not self.tied:
            t = search_column_scales(self.search_grid(kind), [heads[1].weight.data], torch.diag(h))
            for lin in heads:
                lin.weight.data.mul_(t.to(lin.weight.device)[None, :])
            for norm in (step.ref.model.norm, step.work.model.norm):
                norm.weight.data.copy_((1.0 + norm.weight.data) / t.to(norm.weight.device) - 1.0)
            h, g = scaled_moments(h, g, t)
        key = (-1, "lm_head")
        self.solve(key, name, kind, heads[1], h, g, "head    ", w_src=heads[0].weight.data if self.tied else None)
        if optimize and self.opts.method == "blockopt":
            # The head optimization needs only the norm and the head of the reference. With the copies
            # on the compute device, the reference decoder and embedding wait on the CPU meanwhile.
            ref = step.ref.model
            offload = step.store.type == "cuda"
            if offload:
                ref.layers.to("cpu")
                ref.embed_tokens.to("cpu")
                torch.cuda.empty_cache()
            target = Target(self.grids[key], self.hess[key], heads[0].weight.data, self.scales[key])
            solved = optimize_head(step, heads[1], heads[0].weight.data, target, self.opts.opt, rot=rot)
            self.save_solved(name, kind, solved)
            if offload:
                ref.layers.to(step.store)
                ref.embed_tokens.to(step.store)

    def round_embedding(self) -> None:
        """Put the working embedding on its grid, thus the layers see the lookup of the file.

        Untied: the Q8_0 round trip. Tied: the 4-bit round trip with the
        scale search, a first rounding that the head solve replaces.
        """
        kind = self.plan.type_of("token_embd.weight")
        w = self.step.work.model.embed_tokens.weight.data
        if kind == "Q8_0":
            q8_round_trip_(w)
        elif kind in self.fixed:
            grid_round_trip_(w, self.fixed[kind])
        self.step.embed_work()

    # --- the run ---

    @torch.no_grad()
    def run(self) -> None:
        step = self.step
        self.round_embedding()
        for li in range(self.cfg.num_hidden_layers):
            t0 = time.time()
            if self.opts.mismatch == "layer":
                step.ref_in.copy_(step.work_in)
            ref_layer, work_layer = step.layers(li)
            with step.on_device(ref_layer), step.on_device(work_layer):
                if self.cfg.layer_types[li] == "linear_attention":
                    self.mixer_input(li, GDN_IN, "input_layernorm")
                    self.gdn_output(li)
                else:
                    h_in, g_in = self.mixer_input(li, ATTN_IN, "input_layernorm", defer=("self_attn.v_proj",))
                    self.attention_output(li, h_in, g_in)
                self.mlp(li)
                if self.opts.method == "blockopt":
                    self.optimize(li)
                else:
                    step.advance(li)
            print(f"layer {li:2d} done in {time.time() - t0:5.1f}s", flush=True)
        self.head()
        self.save_folds()

    @torch.no_grad()
    def run_head_only(self) -> None:
        """Solve and optimize only the head, on layers that hold the packs of a previous run.

        The caches move through all the layers, then the head follows as in
        ``run``. A tied head takes two passes: the first solves the head on
        the flow of the round-to-nearest embedding, the second on the flow
        of the solved embedding, which is what the file holds.
        """
        step = self.step
        self.round_embedding()
        passes = 2 if self.tied else 1
        for p in range(passes):
            t0 = time.time()
            for li in range(self.cfg.num_hidden_layers):
                if p == 0:
                    step.advance(li)
                else:
                    step.advance_work(li)
            print(f"pass {p}: the caches went through the layers in {time.time() - t0:.0f}s", flush=True)
            self.head(optimize=p == passes - 1)
            if p < passes - 1:
                step.embed_work()
        self.save_folds()


def state_as_checkpoint(model, src_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors of a checkpoint with the language model replaced by ``model``.

    The CausalLM names its text model ``model``; the checkpoint names it
    ``model.language_model``. The vision and MTP tensors stay as they are.
    The embedding carries a tied head, and its dense map passes through
    from the source.
    """
    out = dict(src_tensors)
    tied = is_tied(model)
    for key, value in model.state_dict().items():
        if tied and key.startswith("lm_head."):
            continue
        ck = "model.language_model." + key[len("model."):] if key.startswith("model.") else key
        if ck not in out:
            raise KeyError(f"{key} has no tensor {ck} in the source checkpoint")
        out[ck] = value.detach().to("cpu", torch.float32)
    return out
