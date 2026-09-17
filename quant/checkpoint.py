"""Read and write a Qwen3.5 checkpoint as a dictionary of tensors.

The language model tensors carry the prefix ``model.language_model.``.
This module keeps that prefix, thus the transformed checkpoint loads with
the same classes as the original one.
"""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LM = "model.language_model."
# The dense map of a tied head, M = Qᵀ·diag(γ_f)·Q. The graph applies it after the final norm.
OUTPUT_ROT = LM + "output_rot.weight"


def load_checkpoint(directory: Path) -> "OrderedDict[str, torch.Tensor]":
    """Load every tensor of every safetensors file into CPU memory."""
    tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            for name in f.keys():
                tensors[name] = f.get_tensor(name)
    return tensors


def load_tensor(directory: Path, name: str) -> torch.Tensor | None:
    """One tensor of the checkpoint, or None when no shard holds it. Reads only that tensor."""
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            if name in f.keys():
                return f.get_tensor(name)
    return None


def save_checkpoint(tensors: "OrderedDict[str, torch.Tensor]", src: Path, dst: Path,
                    tie_word_embeddings: bool, shard_bytes: int = 4 << 30) -> None:
    """Write the tensors as safetensors shards plus the config and tokenizer files.

    The config copies from ``src`` with ``tie_word_embeddings`` set as given.
    Complexity is O(total bytes).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copy(path, dst / path.name)
    config = json.loads((dst / "config.json").read_text())
    config["tie_word_embeddings"] = tie_word_embeddings
    if "text_config" in config:
        config["text_config"]["tie_word_embeddings"] = tie_word_embeddings
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        if size + nbytes > shard_bytes and shards[-1]:
            shards.append({})
            size = 0
        shards[-1][name] = tensor.contiguous()
        size += nbytes
    weight_map: dict[str, str] = {}
    total = 0
    for i, shard in enumerate(shards, start=1):
        filename = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, dst / filename, metadata={"format": "pt"})
        for name, tensor in shard.items():
            weight_map[name] = filename
            total += tensor.numel() * tensor.element_size()
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def layer_types(directory: Path) -> list[str]:
    """The layer types of the text model, ``linear_attention`` or ``full_attention``."""
    config = json.loads((directory / "config.json").read_text())
    text = config.get("text_config", config)
    return list(text["layer_types"])


def num_layers(directory: Path) -> int:
    """The number of decoder layers of the text model."""
    return len(layer_types(directory))
