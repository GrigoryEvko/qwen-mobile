"""Build the reduced head that the MTP drafter reads, and put it in the GGUF.

The 4B ties the head to ``token_embd.weight``: 248320 rows of 2560 columns.
Every MTP draft step reads all of it, 675 MB at Q8_0, which is 86 % of the
bytes of a draft step. The drafter does not need the whole vocabulary. A
subset of the rows covers almost every token that the drafter proposes, and
the reduced head is a small fraction of the full one.

The change is lossless. The subset decides only what the drafter can
propose. The target model verifies each proposed token against the full
head, thus a token outside the subset is never accepted wrongly. It is
simply never proposed, and the verify step produces it as usual.

The module has two commands:

- ``select``: blend one or more row-frequency tables and keep the top rows as
  a candidate set of ids.
- ``inject``: gather those rows and write the three draft tensors into a copy
  of the GGUF.

A row-frequency table is a JSON file with the keys ``tokens`` (the number of
tokens that the count read) and ``counts`` (the count of each token id). The
harvest of ``tools/mtp-calib/calib_server.py`` writes those tables, and
``analysis/quant-results.md`` describes the alternative that reads the
proposals of the drafter itself.

The three tensors that ``inject`` adds, all optional and read only by the
MTP graph:

- ``blk.N.nextn.draft_head.weight``, the gathered rows, in the type of the
  source head
- ``blk.N.nextn.draft_ids``, the vocabulary index of each gathered row, as
  I32, which the graph gives to ``ggml_set_rows`` as the scatter index
- ``blk.N.nextn.draft_fill``, one F32 that holds the logit of every token
  outside the subset.

An example, a calibration table blended with a corpus table:

    python -m quant.draftset select --model M.gguf \
        --counts calib.json=3 --counts corpus.json=1 --count 32768 --out ids.json
    python -m quant.draftset inject --src M.gguf --ids ids.json --dst M-draft.gguf

Sources: FR-Spec (arXiv 2502.14856), a static frequency-ranked subset for
the drafter of EAGLE-2, and its calibration refinement that ranks by the
drafter's own proposals.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .paths import gguf_module

# The head that the drafter shares with the main graph. A tied checkpoint has
# no output.weight and llama.cpp reads the head from the embedding table.
HEAD_NAMES = ("output.weight", "token_embd.weight")

# The token types of the GGUF vocabulary that the subset always holds. A
# drafter that cannot propose the end of a turn ends every draft one token
# early, thus these carry far more weight than their frequency shows.
FORCED_TOKEN_TYPES = (2, 3, 4, 5, 6)  # CONTROL, USER_DEFINED, UNUSED, BYTE, UNKNOWN


def forced_ids(reader: Any, n_vocab: int) -> set[int]:
    """Give the ids that the subset must hold whatever their frequency.

    Args:
        reader: An open GGUFReader of the model
        n_vocab: The number of rows of the head

    Returns:
        The set of ids
    """
    out: set[int] = set()
    types = reader.fields.get("tokenizer.ggml.token_type")
    if types is not None:
        for i, t in enumerate(types.contents()):
            if int(t) in FORCED_TOKEN_TYPES and i < n_vocab:
                out.add(i)
    for key in (
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.sep_token_id",
        "tokenizer.ggml.unknown_token_id",
    ):
        f = reader.fields.get(key)
        if f is not None:
            v = int(f.contents())
            if 0 <= v < n_vocab:
                out.add(v)
    return out


def mtp_layer(reader: Any) -> int:
    """Give the index of the MTP block of the model.

    Args:
        reader: An open GGUFReader of the model

    Returns:
        The block index

    Raises:
        ValueError: If the model holds no MTP block
    """
    best = -1
    for t in reader.tensors:
        parts = t.name.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[2] == "nextn":
            best = max(best, int(parts[1]))
    if best < 0:
        raise ValueError("the model holds no blk.N.nextn tensor, thus it has no MTP block")
    return best


def inject(src: Path, dst: Path, ids: list[int], fill: float) -> dict[str, int]:
    """Write a copy of the GGUF that holds the three draft-head tensors.

    The reduced head is a row gather of the full head, thus it holds the same
    quantized bytes and the drafter reads the same numbers it reads today.
    Complexity is O(F) in the bytes of the file, because every tensor is
    copied once.

    Args:
        src: The source GGUF
        dst: The GGUF to write
        ids: The vocabulary indices of the subset, in ascending order
        fill: The logit of every token outside the subset

    Returns:
        The byte count of each tensor that this function adds

    Raises:
        ValueError: If the head is missing or its rows do not divide its bytes
    """
    gguf = gguf_module()
    reader = gguf.GGUFReader(str(src))
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    il = mtp_layer(reader)

    head = next((t for name in HEAD_NAMES for t in reader.tensors if t.name == name), None)
    if head is None:
        raise ValueError(f"{src} holds none of {HEAD_NAMES}")
    n_embd, n_vocab = int(head.shape[0]), int(head.shape[1])
    raw = np.asarray(head.data)
    if raw.nbytes % n_vocab:
        raise ValueError(f"{head.name}: {raw.nbytes} bytes do not divide into {n_vocab} rows")
    row_bytes = raw.nbytes // n_vocab
    index = np.asarray(ids, dtype=np.int64)
    if index.min() < 0 or index.max() >= n_vocab:
        raise ValueError(f"an id is outside 0..{n_vocab - 1}")
    gathered = raw.reshape(n_vocab, row_bytes)[index].copy()

    writer = gguf.GGUFWriter(str(dst), arch)
    skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    for key, f in reader.fields.items():
        if key in skip:
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, f.contents(), vtype, sub_type=sub)

    for t in reader.tensors:
        data = np.asarray(t.data)
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            writer.add_tensor(t.name, data.reshape([int(x) for x in reversed(t.shape)]))
        else:
            rows = int(np.prod([int(x) for x in t.shape[1:]])) or 1
            writer.add_tensor(t.name, data.reshape(rows, data.nbytes // rows), raw_dtype=t.tensor_type)

    added = {
        f"blk.{il}.nextn.draft_head.weight": gathered.nbytes,
        f"blk.{il}.nextn.draft_ids": index.size * 4,
        f"blk.{il}.nextn.draft_fill": 4,
    }
    writer.add_tensor(f"blk.{il}.nextn.draft_head.weight", gathered, raw_dtype=head.tensor_type)
    writer.add_tensor(f"blk.{il}.nextn.draft_ids", index.astype(np.int32))
    writer.add_tensor(f"blk.{il}.nextn.draft_fill", np.array([fill], dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"draftset: head {n_embd}x{n_vocab} -> {n_embd}x{len(ids)} in blk.{il}")
    return added


def cmd_select(a: argparse.Namespace) -> int:
    """Turn one or more frequency tables into a candidate set.

    Each table carries a weight, thus a corpus that is smaller than the
    traffic it stands for can still hold its share of the subset.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    gguf = gguf_module()
    reader = gguf.GGUFReader(str(a.model))
    head = next((t for name in HEAD_NAMES for t in reader.tensors if t.name == name), None)
    if head is None:
        raise ValueError(f"{a.model} holds none of {HEAD_NAMES}")
    n_vocab = int(head.shape[1])

    share: Counter[int] = Counter()
    parts = []
    for spec in a.counts:
        path, _, w = spec.partition("=")
        weight = float(w) if w else 1.0
        d = json.loads(Path(path).read_text())
        tot = d["tokens"] or 1
        for k, v in d["counts"].items():
            share[int(k)] += weight * v / tot
        parts.append(f"{Path(path).name} x{weight:g} ({d['tokens'] / 1e9:.3f} G)")

    forced = forced_ids(reader, n_vocab)
    if len(forced) > a.count:
        raise ValueError(f"{len(forced)} forced ids do not fit a subset of {a.count}")
    chosen = set(forced)
    for tok, _ in sorted(share.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(chosen) >= a.count:
            break
        if tok < n_vocab:
            chosen.add(tok)
    ids = sorted(chosen)
    covered = sum(share[i] for i in ids) / (sum(share.values()) or 1.0)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "model": str(a.model), "n_vocab": n_vocab, "count": len(ids),
        "forced": len(forced), "coverage": covered, "sources": parts, "ids": ids,
    }, indent=1))
    print(f"draftset: {len(ids)} of {n_vocab} rows, {len(forced)} forced, "
          f"weighted coverage {100 * covered:.3f} %")
    for p in parts:
        print(f"draftset:   {p}")
    print(f"draftset: {a.out}")
    return 0


def cmd_inject(a: argparse.Namespace) -> int:
    """Write a GGUF that holds the reduced head.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    spec = json.loads(a.ids.read_text())
    added = inject(a.src, a.dst, spec["ids"], a.fill)
    for name, n in added.items():
        print(f"draftset: {name:40s} {n / 2**20:8.2f} MiB")
    print(f"draftset: {a.dst} ({a.dst.stat().st_size / 2**20:.0f} MiB)")
    return 0


def main() -> int:
    """Run the command line.

    Returns:
        The exit status
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s2 = sub.add_parser("select", help="turn frequency tables into a candidate set")
    s2.add_argument("--model", type=Path, required=True)
    s2.add_argument("--counts", action="append", required=True,
                    help="path, or path=weight to give a corpus a share of the subset")
    s2.add_argument("--count", type=int, default=16384)
    s2.add_argument("--out", type=Path, required=True)
    s2.set_defaults(fn=cmd_select)

    i = sub.add_parser("inject", help="write a GGUF that holds the reduced head")
    i.add_argument("--src", type=Path, required=True)
    i.add_argument("--ids", type=Path, required=True)
    i.add_argument("--dst", type=Path, required=True)
    i.add_argument("--fill", type=float, default=-math.inf)
    i.set_defaults(fn=cmd_inject)

    a = p.parse_args()
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
