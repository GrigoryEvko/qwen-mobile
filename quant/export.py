"""Assemble the final GGUF from the converter's F16 GGUF and the solved blocks.

The F16 GGUF of the transformed checkpoint supplies the metadata, the
tokenizer, and every tensor. This module copies it and replaces each tensor
according to the plan: packed Q4_0 blocks from the solver where they exist,
round-to-nearest Q4_0 or Q8_0 otherwise, F32 for the sensitive small tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .grid import pack_q4_0, pack_q8_0, q4_0_quantize, q8_0_quantize
from .plan import Plan


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
           device: torch.device) -> None:
    """Write ``out_gguf``. Complexity is O(total bytes)."""
    gguf = _load_gguf_module(llama_dir)
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
    for t in reader.tensors:
        name = t.name
        kind = plan.type_of(name)
        shape = [int(x) for x in reversed(t.shape)]
        pack = packs / f"{name}.npz"
        if kind == "Q4_0" and pack.exists():
            z = np.load(pack)
            q = torch.from_numpy(z["q"])
            d = torch.from_numpy(z["d"].view(np.float16))
            writer.add_tensor(name, pack_q4_0(q, d), raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
            kind = "Q4_0 (solved)"
        elif kind == "Q4_0":
            w = torch.from_numpy(_f32_of(t)).to(device)
            q, d = q4_0_quantize(w, search=True)
            writer.add_tensor(name, pack_q4_0(q, d), raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
            kind = "Q4_0 (rtn)"
        elif kind == "Q8_0":
            w = torch.from_numpy(_f32_of(t)).to(device)
            q, d = q8_0_quantize(w)
            writer.add_tensor(name, pack_q8_0(q, d), raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif kind == "F32":
            writer.add_tensor(name, _f32_of(t).astype(np.float32))
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
