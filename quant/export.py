"""Assemble the final GGUF from the converter's F16 GGUF and the solved blocks.

The F16 GGUF of the transformed checkpoint supplies the metadata, the
tokenizer, and every tensor. This module copies it and replaces each tensor
according to the plan: packed Q4_0 blocks from the solver where they exist,
round-to-nearest Q4_0 or Q8_0 otherwise, F32 for the sensitive small tensors.

The solved blocks and the folds are in the order of the checkpoint. The
converter writes the value heads of the linear attention in a different
order when a model has fewer key heads than value heads (the 4B: 16 key
heads, 32 value heads), thus the export permutes those tensors the same way.

A tied head (``tie_head``) has no ``output.weight``: llama.cpp reads the
head from ``token_embd.weight``, through the dense map ``output_rot.weight``
that the graph applies after the final norm. The file then holds one tensor
for the lookup and the head, which removes the separate embedding tensor
from the RAM of the phone.

The MTP block passes through with the type ``mtp`` of the plan. A rotated
source holds its two dense maps, which the transform wrote for an output
norm of one. The calibration can move the output norm to s, and the main
graph then supplies s ⊙ norm(h'). Thus the export divides the columns of
``hnorm_rot`` by s and multiplies the rows of ``shared_head_rot`` by s.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .grid import block_permutation, dequantize_pack, pack_nibbles, pack_q8_0, q8_0_quantize, quantize
from .grids import make_grid
from .plan import MTP_MAPS, Plan
from .refold import Geometry, Refold

GGUF_4BIT = {"Q4_0": "Q4_0", "IQ4_NL": "IQ4_NL"}
FILE_TYPES = {"Q4_0": "MOSTLY_Q4_0", "IQ4_NL": "MOSTLY_IQ4_NL", "Q8_0": "MOSTLY_Q8_0"}
# The type names that a plan field can hold. CB4 has no GGUF type, and the export refuses it per tensor.
PLAN_TYPES = set(GGUF_4BIT) | {"Q8_0", "F16", "CB4"}
# The quantized plan types. The filter ``only`` keeps each such tensor out of its match in F16.
QUANTIZED_TYPES = PLAN_TYPES - {"F16"}
PLAN_TYPE_FIELDS = ("bulk", "head", "embedding", "kv_proj", "gdn_gate", "ssm_out", "ffn_down", "edge_type", "mtp")


def _check_plan_types(plan: Plan) -> None:
    """Raise ValueError when a plan field holds a type name that the export cannot write."""
    for field_name in PLAN_TYPE_FIELDS:
        value = getattr(plan, field_name)
        if value is not None and value not in PLAN_TYPES:
            raise ValueError(f"the plan field {field_name} is {value!r}: the export writes {sorted(PLAN_TYPES)}")


def _f16(name: str, a: np.ndarray) -> np.ndarray:
    """Give ``a`` in F16, or raise ValueError when a value is beyond the F16 range."""
    out = np.asarray(a).astype(np.float16)
    bad = ~np.isfinite(out)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} values are not finite in F16, the largest magnitude is "
                         f"{float(np.nanmax(np.abs(a))):.6g} and F16 permits 65504")
    return out


def _load_gguf_module(llama_dir: Path):
    path = str(llama_dir / "gguf-py")
    # One entry for each process: an insertion for each call would grow sys.path with each export.
    if path not in sys.path:
        sys.path.insert(0, path)
    import gguf  # noqa: E402

    return gguf


def _f32_of(reader_tensor) -> np.ndarray:
    """The float32 numpy array of an F16 or F32 tensor, in numpy row order."""
    data = np.asarray(reader_tensor.data)
    shape = [int(x) for x in reversed(reader_tensor.shape)]
    return data.astype(np.float32).reshape(shape)


@dataclass(frozen=True)
class LinearAttentionLayout:
    """The order of the value heads of the GDN tensors in the GGUF.

    The checkpoint groups the value heads by key head: [K0 v0 … v(r−1),
    K1 v0 … v(r−1), …]. The converter writes them tiled: [K0 v0, K1 v0, …,
    K0 v1, K1 v1, …], thus ggml can repeat the key heads with one tiled
    broadcast. With one value head per key head the two orders are equal.
    """

    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int

    @classmethod
    def from_gguf(cls, reader, arch: str) -> "LinearAttentionLayout | None":
        """The layout from the ssm metadata of the GGUF, or None for a model without linear attention."""
        keys = [f"{arch}.ssm.group_count", f"{arch}.ssm.time_step_rank", f"{arch}.ssm.state_size",
                f"{arch}.ssm.inner_size"]
        if any(k not in reader.fields for k in keys):
            return None
        num_k, num_v, head_k, inner = (int(reader.fields[k].contents()) for k in keys)
        return cls(num_k, num_v, head_k, inner // num_v)

    def _tiled(self, head_dim: int) -> torch.Tensor | None:
        """The index (grouped order) of each position in the tiled order, over num_v_heads · head_dim, or None."""
        if self.num_k_heads == self.num_v_heads:
            return None
        per_k = self.num_v_heads // self.num_k_heads
        order = torch.arange(self.num_v_heads * head_dim).view(self.num_k_heads, per_k, head_dim)
        return order.permute(1, 0, 2).reshape(-1)

    def rows(self, name: str) -> torch.Tensor | None:
        """The row permutation of a GGUF tensor (new row i holds checkpoint row rows[i]), or None."""
        tail = name.split(".", 2)[-1] if name.startswith("blk.") else name
        if tail == "attn_qkv.weight":
            v = self._tiled(self.head_v_dim)
            if v is None:
                return None
            qk = 2 * self.num_k_heads * self.head_k_dim
            return torch.cat([torch.arange(qk), qk + v])
        if tail == "attn_gate.weight":
            return self._tiled(self.head_v_dim)
        if tail in ("ssm_alpha.weight", "ssm_beta.weight"):
            return self._tiled(1)
        return None

    def cols(self, name: str) -> torch.Tensor | None:
        """The column permutation of a GGUF tensor, or None."""
        tail = name.split(".", 2)[-1] if name.startswith("blk.") else name
        if tail == "ssm_out.weight":
            return self._tiled(self.head_v_dim)
        return None

    def array(self, name: str, a: np.ndarray) -> np.ndarray:
        """A float tensor [rows, cols] or [rows] of the checkpoint order in the GGUF order."""
        rows, cols = self.rows(name), self.cols(name)
        if rows is not None:
            a = a[rows.numpy()]
        if cols is not None:
            a = a[:, cols.numpy()]
        return a

    def pack(self, name: str, idx: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The indices [rows, cols] and the block scales [rows, cols // 32] in the GGUF order."""
        rows, cols = self.rows(name), self.cols(name)
        if rows is not None:
            idx, d = idx[rows], d[rows]
        if cols is not None:
            idx, d = idx[:, cols], d[:, block_permutation(cols)]
        return idx, d

    def factors(self, name: str, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The low-rank factors a [rank, cols], b [rows, rank] in the GGUF order."""
        rows, cols = self.rows(name), self.cols(name)
        if rows is not None:
            b = b[rows.numpy()]
        if cols is not None:
            a = a[:, cols.numpy()]
        return a, b


def plan_of(folds) -> dict | None:
    """The plan that the calibration used, from folds.npz, or None for an old file."""
    if folds is None or "plan" not in folds.files:
        return None
    return json.loads(str(folds["plan"]))


def output_rot(rot: Path | None, folds, reader) -> np.ndarray:
    """The dense map M of a tied head as a numpy [out, in] float32 array.

    The sources, in this order: the ``.npy`` file ``rot``, the tensor
    ``output_rot.weight`` of folds.npz, the tensor of a tied source GGUF.
    The GGUF stores a 2-D tensor with its numpy shape reversed, thus a
    numpy [out, in] array is what ``ggml_mul_mat(output_rot, v)`` reads as
    M·v, the same as the head ``output.weight`` [vocab, hidden].
    """
    if rot is not None:
        return np.load(rot).astype(np.float32)
    if folds is not None and "output_rot.weight" in folds.files:
        return folds["output_rot.weight"].astype(np.float32)
    for t in reader.tensors:
        if t.name == "output_rot.weight":
            return _f32_of(t)
    raise ValueError("--tie-head needs the dense map M: --rot <file.npy>, output_rot.weight in folds.npz, "
                     "or a tied source GGUF")


def mtp_map(name: str, a: np.ndarray, out_norm: np.ndarray) -> np.ndarray:
    """A dense map of the MTP block, [out, in] float32, for the exported output norm ``out_norm``.

    The transform wrote the maps for an output norm of one. With the norm s
    the main graph supplies s ⊙ norm(h') in place of norm(h'). Thus
    ``hnorm_rot`` takes diag(s)⁻¹ on its input side, and ``shared_head_rot``
    takes diag(s) on its output side, which keeps the state of the block in
    the convention of the main graph.
    """
    if a.shape != (out_norm.shape[0], out_norm.shape[0]):
        raise ValueError(f"{name} has the shape {a.shape}, the hidden size is {out_norm.shape[0]}")
    if name.endswith("nextn.hnorm_rot.weight"):
        return a / out_norm[None, :]
    if name.endswith("nextn.shared_head_rot.weight"):
        return a * out_norm[:, None]
    raise ValueError(f"{name} is not a dense map of the MTP block")


def make_refold(reader, folds, packs: Path, layout: LinearAttentionLayout | None, arch: str,
                device: torch.device) -> Refold:
    """The refold of the calibration ``folds`` and ``packs`` on the unfolded source ``reader``, in GGUF order."""
    by_name = {t.name: t for t in reader.tensors}

    def fold(name: str) -> np.ndarray:
        a = folds[name].astype(np.float32).reshape([int(x) for x in reversed(by_name[name].shape)])
        return layout.array(name, a) if layout is not None else a

    def pack(name: str) -> torch.Tensor | None:
        path = packs / f"{name}.npz"
        if not path.exists():
            return None
        rows = layout.rows(name) if layout is not None else None
        cols = layout.cols(name) if layout is not None else None
        return dequantize_pack(np.load(path), device, rows, cols)

    geometry = Geometry.from_gguf(reader, arch, layout.head_v_dim if layout is not None else 0)
    return Refold(lambda name: _f32_of(by_name[name]), fold, pack, geometry, device)


def export(f16_gguf: Path, out_gguf: Path, packs: Path, plan: Plan, llama_dir: Path,
           device: torch.device, only: str | None = None, invert: bool = False,
           source_folded: bool = False, tie_head: bool = False, rot: Path | None = None,
           promote: str | None = None, promote_type: str = "Q8_0") -> None:
    """Write ``out_gguf``. Complexity is O(total bytes).

    ``only`` is a regular expression on the GGUF tensor name. The tensors
    that match keep their plan type and every other tensor stays F16, thus
    the file isolates the error of one class. ``invert`` swaps the two sets.

    ``promote`` is a regular expression on the GGUF tensor name. The
    tensors of the 4-bit classes that match take ``promote_type`` (Q8_0 by
    round-to-nearest, or F16) of their folded weight in place of their
    pack. The folded weight comes from the folded reference, or from the
    unfolded source through ``Refold`` (refer to that module for the
    approximation), thus a class moves to 8 bits with no new calibration.

    A plan with the bulk Q8_0 and the F16 GGUF of the original checkpoint
    gives a round-to-nearest Q8_0 file with no transform: ``q8_0_quantize``
    packs each 2-D weight of the plan, one tensor at a time.

    ``source_folded`` says that the F16 GGUF comes from the folded
    reference (``--source tf``). The calibration folds move the columns of
    the solved classes, the MLP channel order and the norms. A tensor that
    the F16 GGUF supplies must be in those coordinates when the plan of the
    export differs from the plan of the calibration, or when ``only`` keeps
    some solved tensors in F16.

    ``tie_head`` drops ``output.weight``, writes ``output_rot.weight`` in
    F16 after the output norm (see ``output_rot`` for its sources), and
    sets the output norm to the identity unless folds.npz holds it: a tied
    head has no column scales. ``token_embd.weight`` then follows the plan,
    from the pack ``token_embd.weight.npz`` when the calibration solved the
    tied head.

    The dense maps of the MTP block must be those of the transform (a
    converter F16 as the source), because the export scales them by the
    exported output norm (refer to ``mtp_map``).
    """
    gguf = _load_gguf_module(llama_dir)
    _check_plan_types(plan)
    selector = re.compile(only) if only else None
    promoter = re.compile(promote) if promote else None
    if promote_type not in ("Q8_0", "F16"):
        raise ValueError(f"a promoted class takes Q8_0 or F16, not {promote_type}")
    # The small tensors that the calibration moved (norms, gates, Q8 matrices), in GGUF space.
    folds = np.load(packs / "folds.npz") if (packs / "folds.npz").exists() else None
    saved_plan = plan_of(folds)
    if folds is not None and not source_folded:
        if only is not None:
            raise ValueError("--only keeps solved tensors in F16: export from the folded reference (--source tf)")
        # The embedding and the MTP block are never calibrated, thus their types can change on the unfolded source.
        wanted = json.loads(json.dumps(plan.calibrated()))
        if saved_plan is not None and Plan(**saved_plan).calibrated() != wanted:
            raise ValueError(f"the plan differs from the calibration plan {saved_plan}: "
                             "export from the folded reference (--source tf)")

    reader = gguf.GGUFReader(str(f16_gguf))
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    layout = LinearAttentionLayout.from_gguf(reader, arch)
    rot_m = output_rot(rot, folds, reader) if tie_head else None
    # A promoted tensor of an unfolded source with a calibration needs its folded coordinates.
    refold = make_refold(reader, folds, packs, layout, arch, device) \
        if promoter is not None and folds is not None and not source_folded else None

    def source(t) -> np.ndarray:
        if folds is not None and t.name in folds.files:
            a = folds[t.name].astype(np.float32).reshape([int(x) for x in reversed(t.shape)])
            return layout.array(t.name, a) if layout is not None else a
        return _f32_of(t)

    def promoted(t) -> torch.Tensor:
        """The folded weight of a promoted tensor on the device."""
        if refold is None:
            return torch.from_numpy(source(t)).to(device)
        return refold.weight(t.name, _f32_of(t))

    # The exported output norm: the identity for a tied head without folds, else the source or the folds.
    t_norm = next((t for t in reader.tensors if t.name == "output_norm.weight"), None)
    if t_norm is None:
        raise ValueError(f"{f16_gguf} has no output_norm.weight")
    out_norm = source(t_norm).astype(np.float32)
    if tie_head and (folds is None or "output_norm.weight" not in folds.files):
        out_norm = np.ones_like(out_norm)

    # The writer writes a temporary file, and the export renames it at the end, thus a failure leaves no partial file.
    partial = out_gguf.with_name(out_gguf.name + ".partial")
    writer = gguf.GGUFWriter(str(partial), arch)
    # general.alignment goes through add_custom_alignment: a plain copy of the key does not move the data.
    skip = {"general.architecture", "general.file_type", "general.alignment", "GGUF.version", "GGUF.tensor_count",
            "GGUF.kv_count"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY and len(field.types) == 1:
            raise ValueError(f"the metadata field {key} is an empty array, which gguf-py cannot write")
        if vtype == gguf.GGUFValueType.ARRAY and field.types[1] == gguf.GGUFValueType.ARRAY:
            raise ValueError(f"the metadata field {key} is an array of arrays, which the export cannot copy")
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), vtype, sub_type=sub_type)
    if "general.alignment" in reader.fields:
        writer.add_custom_alignment(int(reader.fields["general.alignment"].contents()))
    writer.add_file_type(getattr(gguf.LlamaFileType, FILE_TYPES.get(plan.bulk, "MOSTLY_Q4_0")))

    counts: dict[str, int] = {}
    adapter: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in reader.tensors:
        name = t.name
        if name == "output_rot.weight":
            if not tie_head:
                raise ValueError("the source GGUF holds a tied head (output_rot.weight): export with --tie-head")
            continue
        if tie_head and name == "output.weight":
            counts["dropped (tied)"] = counts.get("dropped (tied)", 0) + 1
            continue
        kind = plan.type_of(name)
        promote_this = promoter is not None and kind in GGUF_4BIT and promoter.search(name) is not None
        if promote_this:
            kind = promote_type
        if selector is not None and kind in QUANTIZED_TYPES and (selector.search(name) is None) != invert:
            kind = "keep"
        shape = [int(x) for x in reversed(t.shape)]
        pack = packs / f"{name}.npz"
        if kind in plan.solved_types() and kind not in GGUF_4BIT:
            raise ValueError(f"{name}: {kind} has no GGUF type, measure it with the drift report")
        if (not promote_this and kind not in GGUF_4BIT and pack.exists()
                and folds is not None and not source_folded):
            # The calibration solved this tensor, thus the F16 GGUF holds it in the coordinates
            # before the folds. A plan of an old folds.npz passes the guard above, thus this
            # tensor would take the unfolded weight and the file would be wrong.
            raise ValueError(f"{name}: the plan gives {kind} to a class that the calibration solved, and the "
                             f"source is not folded. Export from the folded reference (--source tf), or move "
                             f"the class with --promote on this source")
        if promote_this and kind == "Q8_0":
            q, d = q8_0_quantize(promoted(t))
            writer.add_tensor(name, pack_q8_0(q, d), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            kind = "Q8_0 (promoted)"
        elif promote_this and kind == "F16":
            writer.add_tensor(name, promoted(t).to("cpu", torch.float16).numpy())
            kind = "F16 (promoted)"
        elif kind in GGUF_4BIT and pack.exists():
            z = np.load(pack)
            idx = torch.from_numpy(z["q"])
            if "levels" not in z:
                idx = idx.to(torch.int16) + 8
            if "kind" in z and str(z["kind"]) != kind:
                raise ValueError(f"{name}: the solved blocks are {z['kind']}, the plan says {kind}")
            d = torch.from_numpy(z["d"].view(np.float16))
            if tuple(idx.shape) != tuple(shape):
                raise ValueError(f"{name}: the pack {pack} has the shape {tuple(idx.shape)}, the source tensor "
                                 f"{tuple(shape)}")
            if layout is not None:
                idx, d = layout.pack(name, idx, d)
            writer.add_tensor(name, pack_nibbles(idx, d), raw_dtype=getattr(gguf.GGMLQuantizationType, GGUF_4BIT[kind]))
            if "lora_a" in z.files:
                a, b = z["lora_a"], z["lora_b"]
                adapter[name] = layout.factors(name, a, b) if layout is not None else (a, b)
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
            a = out_norm if name == "output_norm.weight" else source(t).astype(np.float32)
            writer.add_tensor(name, a)
            if tie_head and name == "output_norm.weight":
                if rot_m.shape != (a.shape[0], a.shape[0]):
                    raise ValueError(f"output_rot has the shape {rot_m.shape}, the hidden size is {a.shape[0]}")
                writer.add_tensor("output_rot.weight", _f16("output_rot.weight", rot_m))
                counts["F16 (output_rot)"] = counts.get("F16 (output_rot)", 0) + 1
        elif name.endswith(MTP_MAPS):
            writer.add_tensor(name, _f16(name, mtp_map(name, _f32_of(t), out_norm)))
            kind = "F16 (MTP maps)"
        else:
            if t.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
                raise ValueError(f"{name}: the F16 GGUF holds an unexpected type {t.tensor_type.name}")
            writer.add_tensor(name, np.asarray(t.data).reshape(shape))
            kind = f"keep {t.tensor_type.name}"
        counts[kind] = counts.get(kind, 0) + 1

    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        partial.replace(out_gguf)
    finally:
        partial.unlink(missing_ok=True)
    if refold is not None:
        for note in refold.notes:
            print(f"  refold {note}")
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
