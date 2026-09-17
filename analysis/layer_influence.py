"""Measure how much each sub-block of Qwen3.5 contributes, layer by layer.

For every decoder layer, two sub-blocks write into the residual stream: the
token mixer (GDN or attention) and the MLP. The script reports, per sub-block:

- cos: the mean cosine between the residual before and after the sub-block
  (1 means the sub-block changes the direction little, the ShortGPT signal)
- rel: the mean ratio ||update|| / ||residual before||
- skip KL: the mean per-token KL of the final logits against the full model
  when the sub-block output is set to zero, and the top-1 agreement

Usage: python layer_influence.py <checkpoint dir> <output file> [--n-seq 32] [--seq-len 1024]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


def load_text_ids(tokenizer, path: Path, n_seq: int, seq_len: int) -> torch.Tensor:
    """Consecutive windows of the text file as token ids [n_seq, seq_len]."""
    ids = tokenizer(path.read_text()).input_ids
    rows = [torch.tensor(ids[i:i + seq_len]) for i in range(0, min(len(ids), n_seq * seq_len), seq_len)]
    return torch.stack(rows[:n_seq])


@torch.no_grad()
def logits_of(model, ids: torch.Tensor, batch: int) -> torch.Tensor:
    """Log-probabilities [n, T, V] in float32 on the CPU, batch by batch."""
    out = []
    for i in range(0, ids.shape[0], batch):
        lg = model(ids[i:i + batch].to(model.device)).logits.float()
        out.append(F.log_softmax(lg, dim=-1).cpu())
    return torch.cat(out)


def kl_and_top1(ref: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    """Mean KL(ref || other) per token and the top-1 agreement."""
    kl = (ref.exp() * (ref - other)).sum(-1).mean().item()
    top1 = (ref.argmax(-1) == other.argmax(-1)).float().mean().item()
    return kl, top1


class Stats:
    """Cosine and relative update size of one sub-block, accumulated over tokens."""

    def __init__(self) -> None:
        self.cos = 0.0
        self.rel = 0.0
        self.n = 0
        self.residual: torch.Tensor | None = None

    def pre(self, module, args, kwargs) -> None:
        self.residual = args[0].detach().float().reshape(-1, args[0].shape[-1])

    def post(self, module, args, output) -> None:
        upd = (output[0] if isinstance(output, tuple) else output).detach().float().reshape(-1, args[0].shape[-1])
        h0 = self.residual
        h1 = h0 + upd
        self.cos += F.cosine_similarity(h0, h1, dim=-1).sum().item()
        self.rel += (upd.norm(dim=-1) / h0.norm(dim=-1).clamp_min(1e-6)).sum().item()
        self.n += h0.shape[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--text", type=Path, default=Path("data/wiki.test.raw"))
    p.add_argument("--n-seq", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = Qwen3_5ForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = load_text_ids(tok, args.text, args.n_seq, args.seq_len)
    layers = model.model.layers
    types = model.config.layer_types

    # Pass 1: the reference logits and the cosine statistics of every sub-block.
    stats: dict[tuple[int, str], Stats] = {}
    handles = []
    for i, layer in enumerate(layers):
        mixer = layer.linear_attn if types[i] == "linear_attention" else layer.self_attn
        for name, mod in (("mixer", mixer), ("mlp", layer.mlp)):
            st = Stats()
            stats[(i, name)] = st
            handles.append(mod.register_forward_pre_hook(st.pre, with_kwargs=True))
            handles.append(mod.register_forward_hook(st.post))
    ref = logits_of(model, ids, args.batch)
    for h in handles:
        h.remove()

    # Pass 2: skip one sub-block at a time by zeroing its output.
    def zero_output(module, args, output):
        return (torch.zeros_like(output[0]),) + tuple(output[1:]) if isinstance(output, tuple) else torch.zeros_like(output)

    lines = [f"# Layer influence of {args.checkpoint.name}, {args.n_seq} x {args.seq_len} tokens of {args.text.name}", "",
             f"{'layer':>5s} {'type':18s} {'block':6s} {'cos':>7s} {'rel':>7s} {'skip KL':>9s} {'skip top-1':>10s}"]
    for i, layer in enumerate(layers):
        mixer = layer.linear_attn if types[i] == "linear_attention" else layer.self_attn
        for name, mod in (("mixer", mixer), ("mlp", layer.mlp)):
            h = mod.register_forward_hook(zero_output)
            kl, top1 = kl_and_top1(ref, logits_of(model, ids, args.batch))
            h.remove()
            st = stats[(i, name)]
            lines.append(f"{i:5d} {types[i]:18s} {name:6s} {st.cos / st.n:7.4f} {st.rel / st.n:7.4f} {kl:9.4f} {100 * top1:9.2f}%")
            print(lines[-1], flush=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
