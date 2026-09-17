"""The anatomy of the conversion error: where the quantized model drifts from the FP model.

Both copies read the same tokens in lockstep (``Lockstep``). For every
layer the report gives, for the residual stream at the layer input and for
the two sub-block updates (mixer and MLP):

- rel: ‖a_q − a_r‖ / ‖a_r‖ over all tokens, and per position quarter
- cos: the mean cosine per token
- drift: the shift of the channel means, ‖mean(a_q) − mean(a_r)‖ / rms(a_r)
- rms: rms(a_q) / rms(a_r) − 1
- the moves of the |a| quantiles p50, p99, p99.9 and of the maximum, q/r − 1
- the mean per-channel skewness and excess kurtosis of r, and the same of q
- top 1 %: the share of the error energy in the 1 % channels with most of it
- injected: the error of the quantized sub-block on the reference input,
  total: the error with the propagated input error included.

The logits give the KL, the top-1 agreement, the KL quantiles, and the KL
by position bin. The weights give the relative error and kurtosis by class.

    python -m quant.drift --model Qwen3.5-2B --source t --packs quant-out/Qwen3.5-2B --out analysis/x.drift.md
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .flow import Lockstep
from .grid import dequantize_pack
from .names import to_gguf

QUANTILES = (0.5, 0.99, 0.999)
SAMPLE = 4_000_000


def load_text_ids(tokenizer, path: Path, n_seq: int, seq_len: int) -> torch.Tensor:
    """Consecutive windows of the text file as token ids [n_seq, seq_len]."""
    ids = tokenizer(path.read_text()).input_ids
    rows = [torch.tensor(ids[i:i + seq_len]) for i in range(0, min(len(ids), n_seq * seq_len), seq_len)]
    return torch.stack(rows[:n_seq])


def apply_packs(model, packs: Path) -> dict[str, tuple[float, float]]:
    """Put the solved blocks into the model. Returns (relative error, kurtosis of W) per GGUF name."""
    stats: dict[str, tuple[float, float]] = {}
    for full_name, lin in model.named_modules():
        if not isinstance(lin, nn.Linear):
            continue
        gguf_name = to_gguf(f"{full_name}.weight")
        if not gguf_name or not (packs / f"{gguf_name}.npz").exists():
            continue
        w = lin.weight.data
        deq = dequantize_pack(np.load(packs / f"{gguf_name}.npz"), w.device)
        stats[gguf_name] = (((deq - w).norm() / w.norm()).item(), kurtosis(w.reshape(-1)))
        w.copy_(deq.to(w.dtype))
    return stats


def kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis of all the values."""
    x = x.to(torch.float32)
    c = x - x.mean()
    return (c.pow(4).mean() / c.pow(2).mean().pow(2).clamp_min(1e-30) - 3.0).item()


def channel_shape(a: torch.Tensor) -> tuple[float, float]:
    """The mean over channels of the per-channel |skewness| and excess kurtosis, a is [N, D]."""
    c = a - a.mean(0, keepdim=True)
    m2 = c.pow(2).mean(0).clamp_min(1e-30)
    skew = (c.pow(3).mean(0) / m2.pow(1.5)).abs().mean().item()
    kurt = (c.pow(4).mean(0) / m2.pow(2) - 3.0).mean().item()
    return skew, kurt


def abs_quantiles(a: torch.Tensor) -> list[float]:
    """|a| at QUANTILES from a random sample, then the exact maximum."""
    flat = a.abs().reshape(-1)
    if flat.numel() > SAMPLE:
        flat = flat[torch.randint(0, flat.numel(), (SAMPLE,), device=flat.device)]
    q = torch.quantile(flat, torch.tensor(QUANTILES, device=flat.device)).tolist()
    return q + [a.abs().max().item()]


def compare(r: torch.Tensor, q: torch.Tensor, seq_len: int) -> dict[str, Any]:
    """The drift statistics of q against r, both [N, D] with N a multiple of seq_len."""
    r, q = r.to(torch.float32), q.to(torch.float32)
    err = q - r
    r_norm = r.norm().clamp_min(1e-12)
    out: dict[str, Any] = {
        "rel": (err.norm() / r_norm).item(),
        "cos": F.cosine_similarity(r, q, dim=-1).mean().item(),
        "drift": ((q.mean(0) - r.mean(0)).norm() / (r_norm / r.shape[0] ** 0.5)).item(),
        "rms": (q.pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-12) - 1.0).item(),
    }
    qr, qq = abs_quantiles(r), abs_quantiles(q)
    out["quantiles"] = [b / max(a, 1e-12) - 1.0 for a, b in zip(qr, qq)]
    out["shape_r"] = channel_shape(r)
    out["shape_q"] = channel_shape(q)
    energy = err.pow(2).sum(0)
    top = max(1, energy.shape[0] // 100)
    out["top1pct"] = (energy.topk(top).values.sum() / energy.sum().clamp_min(1e-30)).item()
    by_pos = err.reshape(-1, seq_len, err.shape[-1]).pow(2).sum((0, 2))
    ref_pos = r.reshape(-1, seq_len, r.shape[-1]).pow(2).sum((0, 2))
    n4 = seq_len // 4
    out["by_quarter"] = [
        (by_pos[i * n4:(i + 1) * n4].sum() / ref_pos[i * n4:(i + 1) * n4].sum().clamp_min(1e-12)).sqrt().item()
        for i in range(4)]
    return out


class Capture:
    """Collect the outputs of a module over the batches of one pass."""

    def __init__(self) -> None:
        self.parts: list[torch.Tensor] = []

    def __call__(self, module, inputs, output) -> None:
        out = output[0] if isinstance(output, tuple) else output
        self.parts.append(out.detach().reshape(-1, out.shape[-1]).to(torch.float32))

    def take(self) -> torch.Tensor:
        out = torch.cat(self.parts, 0)
        self.parts = []
        return out


class CaptureInput(Capture):
    """Collect the first input of a module."""

    def __call__(self, module, inputs) -> None:  # type: ignore[override]
        x = inputs[0]
        self.parts.append(x.detach().reshape(-1, x.shape[-1]).to(torch.float32))


def fmt(x: float) -> str:
    return f"{x:+.3f}" if abs(x) < 10 else f"{x:+.1e}"


@torch.no_grad()
def drift_report(step: Lockstep, title: str, weight_stats: dict[str, tuple[float, float]] | None = None) -> str:
    """The Markdown report of the drift between the two copies of ``step`` on its tokens."""
    cfg = step.cfg
    seq_len = step.seq_len
    residual_rows = []
    update_rows = []
    for li in range(cfg.num_hidden_layers):
        ref_layer, work_layer = step.layers(li)
        gdn = cfg.layer_types[li] == "linear_attention"
        mixer_name = "linear_attn" if gdn else "self_attn"
        res = compare(step.ref_in.reshape(-1, cfg.hidden_size), step.work_in.reshape(-1, cfg.hidden_size), seq_len)
        residual_rows.append((str(li), res))

        cap = {k: Capture() for k in ("u_r", "m_r", "u_q", "m_q", "u_qr")}
        x_r = CaptureInput()
        layer_in_r = step.ref_in.clone()

        # The reference layer: its mixer update, its MLP input and update. The cache moves on.
        handles = [
            ref_layer.get_submodule(mixer_name).register_forward_hook(cap["u_r"]),
            ref_layer.mlp.register_forward_hook(cap["m_r"]),
            ref_layer.mlp.register_forward_pre_hook(x_r),
        ]
        for i, tok in step.batches():
            n = tok.shape[0]
            step.ref_in[i:i + n] = step.run_layer(ref_layer, li, step.ref_in[i:i + n])
        for h in handles:
            h.remove()

        # The quantized layer on the quantized flow: the total error of each update.
        handles = [
            work_layer.get_submodule(mixer_name).register_forward_hook(cap["u_q"]),
            work_layer.mlp.register_forward_hook(cap["m_q"]),
        ]
        for i, tok in step.batches():
            n = tok.shape[0]
            step.work_in[i:i + n] = step.run_layer(work_layer, li, step.work_in[i:i + n])
        for h in handles:
            h.remove()

        # The quantized sub-blocks on the reference inputs: the injected error.
        handle = work_layer.get_submodule(mixer_name).register_forward_hook(cap["u_qr"])
        for i, tok in step.batches():
            n = tok.shape[0]
            step.run_layer(work_layer, li, layer_in_r[i:i + n])
        handle.remove()
        x_ref = x_r.take()
        per = step.batch * seq_len
        m_qr = torch.cat([work_layer.mlp(x_ref[s:s + per]) for s in range(0, x_ref.shape[0], per)], 0)

        u_r, u_q, u_qr = cap["u_r"].take(), cap["u_q"].take(), cap["u_qr"].take()
        m_r, m_q = cap["m_r"].take(), cap["m_q"].take()
        update_rows.append((li, "GDN" if gdn else "attn",
                            compare(u_r, u_qr, seq_len)["rel"], compare(u_r, u_q, seq_len),
                            compare(m_r, m_qr, seq_len)["rel"], compare(m_r, m_q, seq_len)))
        del cap, u_r, u_q, u_qr, m_r, m_q, m_qr, layer_in_r, x_ref
        print(f"drift layer {li:2d}: residual rel {res['rel']:.4f}", flush=True)

    ref_h, work_h = step.final_norm()
    residual_rows.append(("out", compare(ref_h, work_h, seq_len)))

    # Logits, two sequences at a time.
    kl_all, agree = [], []
    for start in range(0, ref_h.shape[0], 2 * seq_len):
        lr = F.log_softmax(step.ref.lm_head(ref_h[start:start + 2 * seq_len]), -1)
        lq = F.log_softmax(step.work.lm_head(work_h[start:start + 2 * seq_len]), -1)
        kl_all.append((lr.exp() * (lr - lq)).sum(-1))
        agree.append((lr.argmax(-1) == lq.argmax(-1)).float())
        del lr, lq
    kl = torch.cat(kl_all)
    top1 = torch.cat(agree).mean().item()
    kl_q = torch.quantile(kl, torch.tensor((0.5, 0.9, 0.99, 0.999), device=kl.device)).tolist()
    bins = kl.reshape(-1, seq_len).mean(0).reshape(8, -1).mean(1).tolist()

    lines = [f"# Drift anatomy: {title}", "",
             f"{step.n_seq} x {seq_len} tokens. r = reference (FP), q = quantized. Δ = q/r − 1.", "",
             "## Residual stream at the layer input", "",
             "| layer | rel | cos | drift | rms Δ | p50 Δ | p99 Δ | p99.9 Δ | max Δ | skew r→q | kurt r→q | err top 1 % | rel by quarter |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in residual_rows:
        qd = s["quantiles"]
        lines.append(f"| {name} | {s['rel']:.4f} | {s['cos']:.5f} | {s['drift']:.4f} | {fmt(s['rms'])} | "
                     f"{fmt(qd[0])} | {fmt(qd[1])} | {fmt(qd[2])} | {fmt(qd[3])} | "
                     f"{s['shape_r'][0]:.2f}→{s['shape_q'][0]:.2f} | {s['shape_r'][1]:.1f}→{s['shape_q'][1]:.1f} | "
                     f"{100 * s['top1pct']:.0f} % | " + " ".join(f"{x:.4f}" for x in s["by_quarter"]) + " |")
    lines += ["", "## Sub-block updates", "",
              "injected = the quantized sub-block on the reference input. total = with the propagated input error.", "",
              "| layer | type | mixer injected | mixer total | mixer cos | mixer kurt r→q | MLP injected | MLP total | MLP cos | MLP kurt r→q |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for li, kind, u_inj, u, m_inj, m in update_rows:
        lines.append(f"| {li} | {kind} | {u_inj:.4f} | {u['rel']:.4f} | {u['cos']:.5f} | "
                     f"{u['shape_r'][1]:.1f}→{u['shape_q'][1]:.1f} | {m_inj:.4f} | {m['rel']:.4f} | {m['cos']:.5f} | "
                     f"{m['shape_r'][1]:.1f}→{m['shape_q'][1]:.1f} |")
    lines += ["", "## Logits", "",
              f"Mean KL {kl.mean().item():.5f}, top-1 agreement {100 * top1:.2f} %, KL p50 {kl_q[0]:.5f}, "
              f"p90 {kl_q[1]:.5f}, p99 {kl_q[2]:.4f}, p99.9 {kl_q[3]:.3f}, max {kl.max().item():.3f}.", "",
              "KL by position bin (8 bins): " + ", ".join(f"{x:.4f}" for x in bins)]
    if weight_stats:
        by_class: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for name, st in weight_stats.items():
            by_class[name.split(".", 2)[-1] if name.startswith("blk.") else name].append(st)
        lines += ["", "## Weights", "", "| class | n | mean rel error | mean kurtosis of W |", "|---|---|---|---|"]
        for cls, sts in sorted(by_class.items()):
            lines.append(f"| {cls} | {len(sts)} | {np.mean([s[0] for s in sts]):.4f} | {np.mean([s[1] for s in sts]):.1f} |")
    return "\n".join(lines) + "\n"
