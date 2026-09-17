"""Calibration data and the layer-wise quantized-flow driver.

The driver keeps the whole model on the GPU in float32. For each decoder
layer in order, it registers hooks on the linears of that layer, runs the
calibration set through the full model to accumulate H = mean(xᵀx) for each
linear, solves the matrices of the plan, and writes the dequantized weights
back. The next layer thus calibrates on the outputs of the quantized layers
before it (the quantized flow).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .grid import q4_0_dequantize
from .names import to_gguf
from .plan import Plan
from .solver import gptq_q4_0


def build_calibration(tokenizer, n_seq: int, seq_len: int, seed: int, out: Path) -> torch.Tensor:
    """Token ids [n_seq, seq_len] from C4 English, saved to ``out``."""
    if out.exists():
        return torch.load(out)
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    rows: list[torch.Tensor] = []
    buffer: list[int] = []
    for sample in ds:
        buffer.extend(tokenizer(sample["text"]).input_ids)
        buffer.append(tokenizer.eos_token_id or 0)
        while len(buffer) >= seq_len and len(rows) < n_seq:
            rows.append(torch.tensor(buffer[:seq_len]))
            buffer = buffer[seq_len:]
        if len(rows) >= n_seq:
            break
    ids = torch.stack(rows)
    torch.save(ids, out)
    return ids


class HessianHook:
    """Accumulate mean(xᵀx) of the inputs of one linear in float32."""

    def __init__(self, cols: int, device: torch.device) -> None:
        self.h = torch.zeros(cols, cols, dtype=torch.float32, device=device)
        self.n = 0

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
        self.h.addmm_(x.T, x)
        self.n += x.shape[0]

    def mean(self) -> torch.Tensor:
        return self.h / max(self.n, 1)


def linears_of_layer(layer: nn.Module) -> dict[str, nn.Linear]:
    """The linears of a decoder layer by their relative name."""
    return {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def run_calibration(model, ids: torch.Tensor, batch: int) -> None:
    """One pass of the calibration set through the model."""
    for i in range(0, ids.shape[0], batch):
        model(ids[i:i + batch].to(model.device))


@torch.no_grad()
def quantize_layers(model, ids: torch.Tensor, plan: Plan, out_dir: Path, batch: int = 8,
                    damp: float = 0.01, layers: range | None = None) -> None:
    """Solve each Q4_0 matrix of the plan, layer by layer, in the quantized flow.

    Each solved matrix goes to ``out_dir/<gguf name>.npz`` with the levels and
    the scales, and its dequantized value replaces the weight in the model.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder = model.model.layers
    for li in layers if layers is not None else range(len(decoder)):
        layer = decoder[li]
        targets: dict[str, nn.Linear] = {}
        for rel, lin in linears_of_layer(layer).items():
            gguf_name = to_gguf(f"model.layers.{li}.{rel}.weight")
            if gguf_name and plan.type_of(gguf_name) in plan.solved_types():
                targets[gguf_name] = lin
        if not targets:
            continue
        hooks = {name: HessianHook(lin.in_features, lin.weight.device) for name, lin in targets.items()}
        handles = [lin.register_forward_hook(hooks[name]) for name, lin in targets.items()]
        t0 = time.time()
        run_calibration(model, ids, batch)
        for h in handles:
            h.remove()
        t_h = time.time() - t0
        for name, lin in targets.items():
            t1 = time.time()
            q, d, err = gptq_q4_0(lin.weight.data, hooks[name].mean(), damp=damp)
            lin.weight.data.copy_(q4_0_dequantize(q, d).to(lin.weight.dtype))
            np.savez(out_dir / f"{name}.npz", q=q.cpu().numpy(), d=d.cpu().numpy().view(np.uint16))
            print(f"layer {li:2d} {name:28s} {tuple(lin.weight.shape)} err {err:.3e} "
                  f"hessian {t_h:5.1f}s solve {time.time() - t1:5.1f}s", flush=True)
        del hooks
        torch.cuda.empty_cache()
