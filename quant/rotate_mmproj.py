"""The vision projector of a rotated model.

The transform folds the Hadamard rotation into the token embedding, thus
the image features must enter the residual stream in the same basis. The
last linear of the merger (``mm.2`` in the projector GGUF) gets W' = Qᵀ·W
and b' = Qᵀ·b, the same rotation that the checkpoint transform applies to
``model.visual.merger.linear_fc2``. Every other tensor and all the
metadata copy as they are.

    python -m quant.rotate_mmproj weights/gguf/Qwen3.5-2B-mmproj-F16.gguf weights/gguf/Qwen3.5-2B-gptq-Q4_0.mmproj.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .transform import rotation_matrix

ROOT = Path(__file__).resolve().parent.parent
MERGER_OUT = "mm.2"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--seed", type=int, default=0, help="the seed of the rotation, as in the transform")
    p.add_argument("--block", type=int, default=None, help="the block of the rotation, as in the transform")
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
    import gguf  # noqa: E402

    reader = gguf.GGUFReader(str(args.src))
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    writer = gguf.GGUFWriter(str(args.dst), arch)
    skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)

    q: torch.Tensor | None = None
    for t in reader.tensors:
        shape = [int(x) for x in reversed(t.shape)]
        data = np.asarray(t.data).reshape(shape)
        if t.name in (MERGER_OUT + ".weight", MERGER_OUT + ".bias"):
            n = shape[0]
            if q is None:
                q = rotation_matrix(n, args.block, args.seed, torch.device("cpu"))
            x = torch.from_numpy(data.astype(np.float32)).to(torch.float64)
            rotated = (q.T @ x) if x.dim() == 2 else (q.T @ x)
            data = rotated.to(torch.float32).numpy().astype(data.dtype)
            print(f"rotated {t.name} {shape}")
        writer.add_tensor(t.name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"wrote {args.dst} ({args.dst.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
