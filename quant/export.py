"""Assemble the final GGUF from the converter's F16 GGUF and the solved blocks.

The F16 GGUF of the transformed checkpoint supplies the metadata, the
tokenizer, and every tensor. This module copies it and replaces each tensor
according to the plan: packed Q4_0 blocks from the solver where they exist,
round-to-nearest Q4_0 or Q8_0 otherwise, F32 for the sensitive small tensors.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

from .grid import pack_nibbles, pack_q8_0, q8_0_quantize, quantize
from .grids import make_grid
from .plan import Plan

GGUF_4BIT = {"Q4_0": "Q4_0", "IQ4_NL": "IQ4_NL"}


def _load_gguf_module(llama_dir: Path):
    sys.path.insert(0, str(llama_dir / "gguf-py"))
    import gguf  # noqa: E402

    return gguf


def _f32_of(reader_tensor) -> np.ndarray:
    """The float32 numpy array of an F16 or F32 tensor, in numpy row order."""
    data = np.asarray(reader_tensor.data)
    shape = [int(x) for x in reversed(reader_tensor.shape)]
    return data.astype(np.float32).reshape(shape)


def export(f16_gguf: Path, out_gguf: Path, packs: Path, plan: Plan, llama_dir: Path,
           device: torch.device, only: str | None = None, invert: bool = False) -> None:
    """Write ``out_gguf``. Complexity is O(total bytes).

    ``only`` is a regular expression on the GGUF tensor name. The tensors
    that match keep their plan type and every other tensor stays F16, thus
    the file isolates the error of one class. ``invert`` swaps the two sets.
    """
    gguf = _load_gguf_module(llama_dir)
    selector = re.compile(only) if only else None
    # The small tensors that the calibration moved (norms, gates, Q8 matrices), in GGUF space.
    folds = np.load(packs / "folds.npz") if (packs / "folds.npz").exists() else None

    def source(t) -> np.ndarray:
        if folds is not None and t.name in folds.files:
            return folds[t.name].astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
        return _f32_of(t)
    reader = gguf.GGUFReader(str(f16_gguf))
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    writer = gguf.GGUFWriter(str(out_gguf), arch)

    skip = {"general.architecture", "general.file_type", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)

    counts: dict[str, int] = {}
    adapter: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in reader.tensors:
        name = t.name
        kind = plan.type_of(name)
        if selector is not None and kind in ("Q4_0", "Q8_0") and (selector.search(name) is None) != invert:
            kind = "keep"
        shape = [int(x) for x in reversed(t.shape)]
        pack = packs / f"{name}.npz"
        if kind in plan.solved_types() and kind not in GGUF_4BIT:
            raise ValueError(f"{name}: {kind} has no GGUF type, measure it with the drift report")
        if kind in GGUF_4BIT and pack.exists():
            z = np.load(pack)
            idx = torch.from_numpy(z["q"])
            if "levels" not in z:
                idx = idx.to(torch.int16) + 8
            if "kind" in z and str(z["kind"]) != kind:
                raise ValueError(f"{name}: the solved blocks are {z['kind']}, the plan says {kind}")
            d = torch.from_numpy(z["d"].view(np.float16))
            writer.add_tensor(name, pack_nibbles(idx, d), raw_dtype=getattr(gguf.GGMLQuantizationType, GGUF_4BIT[kind]))
            if "lora_a" in z.files:
                adapter[name] = (z["lora_a"], z["lora_b"])
            kind = f"{kind} (solved)"
        elif kind in GGUF_4BIT:
            w = torch.from_numpy(_f32_of(t)).to(device)
            idx, d = quantize(make_grid(kind).to(device), w, search=True)
            writer.add_tensor(name, pack_nibbles(idx, d), raw_dtype=getattr(gguf.GGMLQuantizationType, GGUF_4BIT[kind]))
            kind = f"{kind} (rtn)"
        elif kind == "Q8_0":
            w = torch.from_numpy(source(t)).to(device)
            q, d = q8_0_quantize(w)
            writer.add_tensor(name, pack_q8_0(q, d), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif kind == "F32":
            writer.add_tensor(name, source(t).astype(np.float32))
        else:
            if t.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                raise ValueError(f"{name}: the F16 GGUF holds an unexpected type {t.tensor_type.name}")
            writer.add_tensor(name, np.asarray(t.data).reshape(shape))
            kind = f"keep {t.tensor_type.name}"
        counts[kind] = counts.get(kind, 0) + 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    for kind, n in sorted(counts.items()):
        print(f"  {kind:16s} {n:4d} tensors")
    print(f"wrote {out_gguf} ({out_gguf.stat().st_size / 2**30:.2f} GiB)")
    if adapter:
        write_adapter(gguf, arch, out_gguf.with_name(out_gguf.stem + "-lora.gguf"), adapter)


def write_adapter(gguf, arch: str, path: Path, adapter: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Write the low-rank corrections as a GGUF LoRA adapter, alpha = rank thus scale 1.

    llama.cpp expects ``<name>.lora_a`` as [rank, in] and ``<name>.lora_b``
    as [out, rank], and computes W·x + b·(a·x).
    """
    rank = next(iter(adapter.values()))[0].shape[0]
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_type("adapter")
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(rank))
    total = 0
    for name, (a, b) in adapter.items():
        writer.add_tensor(f"{name}.lora_a", a.astype(np.float16))
        writer.add_tensor(f"{name}.lora_b", b.astype(np.float16))
        total += (a.size + b.size) * 2
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"wrote {path} ({len(adapter)} corrections of rank {rank}, {total / 2**20:.1f} MiB)")
