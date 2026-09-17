"""The quantized-flow driver, one decoder layer at a time, in lockstep with the FP model.

Two copies of the transformed checkpoint sit on the GPU in float32: the
reference, whose function never changes, and the working copy that receives
the quantized weights. The driver caches the inputs of the current layer for
the whole calibration set from both copies, thus each sub-pass runs one
decoder layer and not the model.

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

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask, create_recurrent_attention_mask

from .grid import q4_0_dequantize, q8_0_quantize
from .names import to_gguf
from .plan import Plan
from .scale import permuted_moments, scaled_moments, search_column_scales
from .solver import gptq_q4_0

GDN_IN = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b")
ATTN_IN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")


@dataclass
class Options:
    """The knobs of one calibration run."""

    solver: str = "qronos"
    scale: bool = True
    permute_mlp: bool = True
    mismatch: str = "model"
    damp: float = 0.01
    refit_damp: float = 1e-6
    batch: int = 8


def q8_round_trip_(w: torch.Tensor) -> None:
    """Replace ``w`` by its Q8_0 round trip, in place."""
    q, d = q8_0_quantize(w)
    w.copy_(q4_0_dequantize(q, d))


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

    The caches are [n_seq, seq_len, hidden] float32 on the device, one per
    copy. ``advance`` moves them through one layer.
    """

    def __init__(self, ref, work, ids: torch.Tensor, batch: int) -> None:
        self.ref, self.work = ref, work
        self.ids = ids
        self.batch = batch
        self.device = ref.device
        self.cfg = ref.config
        self.n_seq, self.seq_len = ids.shape
        shape = (self.n_seq, self.seq_len, self.cfg.hidden_size)
        self.ref_in = torch.empty(shape, dtype=torch.float32, device=self.device)
        self.work_in = torch.empty(shape, dtype=torch.float32, device=self.device)
        self._context: dict[int, tuple] = {}
        with torch.no_grad():
            for i, tok in self.batches():
                self.ref_in[i:i + tok.shape[0]] = ref.model.embed_tokens(tok)
                self.work_in[i:i + tok.shape[0]] = work.model.embed_tokens(tok)

    def batches(self):
        """(start, ids) of each batch, on the device."""
        for i in range(0, self.n_seq, self.batch):
            yield i, self.ids[i:i + self.batch].to(self.device)

    def context(self, n: int) -> tuple:
        """The rotary embeddings, the two masks, and the text positions for n sequences."""
        if n not in self._context:
            length = self.seq_len
            pos = torch.arange(length, device=self.device).view(1, 1, -1).expand(4, n, -1)
            hidden = self.ref_in[:n]
            rotary = self.ref.model.rotary_emb(hidden, pos[1:])
            kw = dict(config=self.cfg, input_embeds=hidden, attention_mask=None,
                      cache_position=torch.arange(length, device=self.device), past_key_values=None,
                      position_ids=pos[0])
            masks = {"full_attention": create_causal_mask(**kw), "linear_attention": create_recurrent_attention_mask(**kw)}
            self._context[n] = (rotary, masks, pos[0])
        return self._context[n]

    def run_layer(self, layer: nn.Module, li: int, hidden: torch.Tensor) -> torch.Tensor:
        """One decoder layer on [n, seq_len, hidden]."""
        rotary, masks, text_pos = self.context(hidden.shape[0])
        return layer(hidden, position_embeddings=rotary, attention_mask=masks[self.cfg.layer_types[li]],
                     position_ids=text_pos)

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
        for i, tok in self.batches():
            n = tok.shape[0]
            self.run_layer(ref_layer, li, self.ref_in[i:i + n])
            self.run_layer(work_layer, li, self.work_in[i:i + n])
        for h in handles:
            h.remove()
        return moments

    @torch.no_grad()
    def advance(self, li: int) -> None:
        """Replace the cached inputs by the outputs of layer li."""
        ref_layer, work_layer = self.layers(li)
        for i, tok in self.batches():
            n = tok.shape[0]
            self.ref_in[i:i + n] = self.run_layer(ref_layer, li, self.ref_in[i:i + n])
            self.work_in[i:i + n] = self.run_layer(work_layer, li, self.work_in[i:i + n])

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
        out_dir.mkdir(parents=True, exist_ok=True)

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
        if not self.opts.scale or not rels:
            return None
        ws = [self.both(li, rel)[1].weight.data for rel in rels]
        return search_column_scales(ws, torch.diag(h), share)

    # --- solving ---

    def solve(self, name: str, lin: nn.Linear, h: torch.Tensor, g: torch.Tensor, tag: str) -> None:
        t0 = time.time()
        cross = g if self.opts.solver == "qronos" else None
        q, d, err, change = gptq_q4_0(lin.weight.data, h, cross, damp=self.opts.damp, refit_damp=self.opts.refit_damp)
        lin.weight.data.copy_(q4_0_dequantize(q, d))
        np.savez(self.out_dir / f"{name}.npz", q=q.cpu().numpy(), d=d.cpu().numpy().view(np.uint16))
        print(f"{tag} {name:30s} {tuple(lin.weight.shape)} err {err:.3e} refit {change:.3f} "
              f"solve {time.time() - t0:4.1f}s", flush=True)

    def solve_or_round(self, li: int, rel: str, h: torch.Tensor, g: torch.Tensor) -> None:
        kind = self.kind(li, rel)
        lin = self.both(li, rel)[1]
        if kind == "Q4_0":
            self.solve(self.name(li, rel), lin, h, g, f"layer {li:2d}")
        elif kind == "Q8_0":
            q8_round_trip_(lin.weight.data)

    # --- groups ---

    def mixer_input(self, li: int, members: tuple[str, ...], norm_rel: str, defer: tuple[str, ...] = ()) -> None:
        """The projections that read the input norm. ``defer`` waits for a later fold."""
        h, g = self.step.collect(li, [members[0]])[members[0]].mean()
        q4 = [rel for rel in members if self.kind(li, rel) == "Q4_0"]
        t = self.column_scales(li, q4, h)
        if t is not None:
            self.scale_columns(li, members, t)
            self.scale_norm(li, norm_rel, t)
            h, g = scaled_moments(h, g, t)
        for rel in members:
            if rel not in defer:
                self.solve_or_round(li, rel, h, g)

    def gdn_output(self, li: int) -> None:
        """out_proj, with its column scales in the shared weight of the gated norm."""
        rel = "linear_attn.out_proj"
        h, g = self.step.collect(li, [rel])[rel].mean()
        v_dim = self.cfg.linear_value_head_dim
        share = torch.arange(h.shape[0], device=h.device) % v_dim
        t = self.column_scales(li, [rel], h, share)
        if t is not None:
            self.scale_columns(li, [rel], t)
            for norm in self.both(li, "linear_attn.norm"):
                norm.weight.data.div_(t[:v_dim])
            h, g = scaled_moments(h, g, t)
        self.solve_or_round(li, rel, h, g)

    def attention_output(self, li: int) -> None:
        """o_proj, with its column scales in the v_proj rows of the KV group of each head."""
        rel = "self_attn.o_proj"
        h, g = self.step.collect(li, [rel])[rel].mean()
        heads, kv_heads, dim = self.cfg.num_attention_heads, self.cfg.num_key_value_heads, self.cfg.head_dim
        group = heads // kv_heads
        j = torch.arange(h.shape[0], device=h.device)
        share = (j // dim // group) * dim + j % dim
        t = self.column_scales(li, [rel], h, share)
        if t is not None:
            self.scale_columns(li, [rel], t)
            self.scale_rows(li, "self_attn.v_proj", t.view(heads, dim)[::group].reshape(-1))
            h, g = scaled_moments(h, g, t)
        self.solve_or_round(li, "self_attn.v_proj", h, g)
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

    def head(self) -> None:
        """The untied head on the final-norm outputs, with its column scales in the final norm."""
        if self.plan.type_of("output.weight") != "Q4_0":
            return
        ref_h, work_h = self.step.final_norm()
        n = work_h.shape[0]
        h = work_h.T @ work_h / n
        g = work_h.T @ ref_h / n
        del ref_h, work_h
        heads = (self.step.ref.lm_head, self.step.work.lm_head)
        if self.opts.scale:
            t = search_column_scales([heads[1].weight.data], torch.diag(h))
            for lin in heads:
                lin.weight.data.mul_(t[None, :])
            for norm in (self.step.ref.model.norm, self.step.work.model.norm):
                norm.weight.data.copy_((1.0 + norm.weight.data) / t - 1.0)
            h, g = scaled_moments(h, g, t)
        self.solve("output.weight", heads[1], h, g, "head    ")

    # --- the run ---

    @torch.no_grad()
    def run(self) -> None:
        step = self.step
        if self.plan.type_of("token_embd.weight") == "Q8_0":
            q8_round_trip_(step.work.model.embed_tokens.weight.data)
        for li in range(self.cfg.num_hidden_layers):
            t0 = time.time()
            if self.opts.mismatch == "layer":
                step.ref_in.copy_(step.work_in)
            if self.cfg.layer_types[li] == "linear_attention":
                self.mixer_input(li, GDN_IN, "input_layernorm")
                self.gdn_output(li)
            else:
                self.mixer_input(li, ATTN_IN, "input_layernorm", defer=("self_attn.v_proj",))
                self.attention_output(li)
            self.mlp(li)
            step.advance(li)
            print(f"layer {li:2d} done in {time.time() - t0:5.1f}s", flush=True)
        self.head()


def state_as_checkpoint(model, src_tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The tensors of a checkpoint with the language model replaced by ``model``.

    The CausalLM names its text model ``model``; the checkpoint names it
    ``model.language_model``. The vision and MTP tensors stay as they are.
    """
    out = dict(src_tensors)
    for key, value in model.state_dict().items():
        ck = "model.language_model." + key[len("model."):] if key.startswith("model.") else key
        if ck not in out:
            raise KeyError(f"{key} has no tensor {ck} in the source checkpoint")
        out[ck] = value.detach().to("cpu", torch.float32)
    return out
