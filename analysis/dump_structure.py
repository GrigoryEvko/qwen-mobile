"""Write the structure of a Qwen3.5 checkpoint to a text file.

The report has four parts: the configuration that matters for inference,
the module tree from a meta-device model (no weights are loaded), the
parameter table from the safetensors headers (the stored dtypes and
shapes), and the byte budget of one decode token at F16 and at Q4_0.

Usage: python dump_structure.py <checkpoint dir> <output file>
"""

from __future__ import annotations

import json
import struct
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

# Q4_0 stores 32 weights in 18 bytes: one F16 scale and 32 nibbles.
Q4_0_BITS_PER_WEIGHT = 18 * 8 / 32
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1, "I32": 4, "I64": 8}


def read_safetensors_headers(directory: Path) -> "OrderedDict[str, dict]":
    """Read the tensor headers of every safetensors file, without the data."""
    tensors: "OrderedDict[str, dict]" = OrderedDict()
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            tensors[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "file": path.name}
    return OrderedDict(sorted(tensors.items()))


def numel(shape: list[int]) -> int:
    """Number of elements of a shape."""
    n = 1
    for d in shape:
        n *= d
    return n


def category(name: str) -> str:
    """Group a parameter name into one budget category."""
    if name.startswith("model.visual") or name.startswith("visual"):
        return "vision"
    if "embed_tokens" in name:
        return "embed_tokens (tied lm_head)"
    if "lm_head" in name:
        return "lm_head"
    if ".linear_attn." in name:
        return "gdn." + name.split(".linear_attn.")[1].split(".")[0]
    if ".self_attn." in name:
        return "attn." + name.split(".self_attn.")[1].split(".")[0]
    if ".mlp." in name:
        return "mlp." + name.split(".mlp.")[1].split(".")[0]
    if "norm" in name:
        return "norms"
    return "other"


def is_quantizable(name: str, shape: list[int]) -> bool:
    """A 2-D linear weight of the language model goes to Q4_0. Others stay F16."""
    if len(shape) != 2 or category(name) == "vision":
        return False
    small = ("in_proj_a", "in_proj_b", "conv1d", "A_log", "dt_bias", "norm")
    return not any(s in name for s in small)


def format_config(cfg) -> list[str]:
    """The configuration fields that decide the graph and the byte budget."""
    tc = getattr(cfg, "text_config", cfg)
    lines = ["== Text config =="]
    keys = [
        "model_type", "hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
        "tie_word_embeddings", "num_attention_heads", "num_key_value_heads", "head_dim",
        "attn_output_gate", "full_attention_interval", "linear_num_value_heads",
        "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim",
        "linear_conv_kernel_dim", "rms_norm_eps", "rope_theta", "rope_parameters",
        "max_position_embeddings", "hidden_act", "attention_bias", "mlp_only_layers",
    ]
    for k in keys:
        if hasattr(tc, k):
            lines.append(f"  {k}: {getattr(tc, k)}")
    layer_types = getattr(tc, "layer_types", None)
    if layer_types:
        counts = defaultdict(int)
        for t in layer_types:
            counts[t] += 1
        lines.append(f"  layer_types: {dict(counts)}")
        lines.append("  layer pattern: " + " ".join("A" if "full" in t else "G" for t in layer_types))
    vc = getattr(cfg, "vision_config", None)
    if vc is not None:
        lines.append("== Vision config ==")
        for k in ["model_type", "depth", "hidden_size", "num_heads", "patch_size", "out_hidden_size",
                  "spatial_merge_size", "temporal_patch_size", "intermediate_size"]:
            if hasattr(vc, k):
                lines.append(f"  {k}: {getattr(vc, k)}")
    return lines


def main() -> None:
    """Write the report."""
    ckpt = Path(sys.argv[1])
    out = Path(sys.argv[2])
    cfg = AutoConfig.from_pretrained(ckpt)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg, dtype=torch.bfloat16)
    tensors = read_safetensors_headers(ckpt)

    lines: list[str] = [f"# Structure of {ckpt.name}", ""]
    lines += format_config(cfg)

    lines += ["", "== Module tree (meta device) ==", repr(model), ""]

    lines += ["== Parameters as stored in safetensors ==",
              f"{'name':70s} {'dtype':5s} {'shape':24s} {'numel':>12s} {'bytes':>12s}"]
    total_bytes = 0
    for name, meta in tensors.items():
        n = numel(meta["shape"])
        b = n * DTYPE_BYTES.get(meta["dtype"], 2)
        total_bytes += b
        lines.append(f"{name:70s} {meta['dtype']:5s} {str(meta['shape']):24s} {n:12,d} {b:12,d}")
    lines.append(f"{'TOTAL':70s} {'':5s} {'':24s} {sum(numel(m['shape']) for m in tensors.values()):12,d} {total_bytes:12,d}")

    lines += ["", "== Byte budget of one decode token, language model only ==",
              "Category: parameters, bytes at F16, bytes at Q4_0 for the 2-D linear weights (others F16), share at Q4_0",
              f"{'category':28s} {'params':>14s} {'F16 bytes':>14s} {'Q4_0 bytes':>14s} {'share':>7s}"]
    per_cat = defaultdict(lambda: [0, 0, 0])
    for name, meta in tensors.items():
        cat = category(name)
        n = numel(meta["shape"])
        f16 = n * 2
        q4 = int(n * Q4_0_BITS_PER_WEIGHT / 8) if is_quantizable(name, meta["shape"]) else f16
        per_cat[cat][0] += n
        per_cat[cat][1] += f16
        per_cat[cat][2] += q4
    lm_cats = {k: v for k, v in per_cat.items() if k != "vision"}
    total_q4 = sum(v[2] for v in lm_cats.values())
    for cat, (n, f16, q4) in sorted(lm_cats.items(), key=lambda kv: -kv[1][2]):
        lines.append(f"{cat:28s} {n:14,d} {f16:14,d} {q4:14,d} {100 * q4 / total_q4:6.1f}%")
    lines.append(f"{'TOTAL language model':28s} {sum(v[0] for v in lm_cats.values()):14,d} "
                 f"{sum(v[1] for v in lm_cats.values()):14,d} {total_q4:14,d} {100.0:6.1f}%")
    v = per_cat.get("vision")
    if v:
        lines.append(f"{'vision (not read in decode)':28s} {v[0]:14,d} {v[1]:14,d} {'':>14s}")
    lines += [
        "",
        "Decode reads every language-model weight once per token. At ~60 GB/s DRAM:",
        f"  F16:  {sum(v[1] for v in lm_cats.values()) / 60e9 * 1e3:6.1f} ms per token of weight reads",
        f"  Q4_0: {total_q4 / 60e9 * 1e3:6.1f} ms per token of weight reads",
        "",
        "== Full config JSON ==",
        json.dumps(cfg.to_dict(), indent=2, default=str),
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
