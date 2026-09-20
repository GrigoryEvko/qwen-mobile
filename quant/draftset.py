"""Build the reduced head that the MTP drafter reads, and put it in the GGUF.

Our 4B ties the head to ``token_embd.weight``: 248320 rows of 2560 columns.
Every MTP draft step reads all of it, 675 MB at Q8_0, which is 86 % of the
bytes of a draft step. The drafter does not need the whole vocabulary. A
frequency-ranked subset of about 3072 rows covers almost every token that a
draft proposes, and the reduced head is 8 MB.

The change is lossless. The subset decides only what the drafter can
propose. The target model verifies each proposed token against the full
head, thus a token outside the subset is never accepted wrongly. It is
simply never proposed, and the verify step produces it as usual.

This module writes three tensors into the GGUF of the model:

- ``blk.N.nextn.draft_head.weight``, the gathered rows, in the type of the
  source head
- ``blk.N.nextn.draft_ids``, the vocabulary index of each gathered row, as
  I32, which the graph gives to ``ggml_set_rows`` as the scatter index
- ``blk.N.nextn.draft_fill``, one F32 that holds the logit of every token
  outside the subset.

The two commands:

    python -m quant.draftset rank  --model M.gguf --corpus data/wiki.test.raw \
        --count 3072 --out build/draftset/qwen35-4b.json
    python -m quant.draftset inject --src M.gguf --ids build/draftset/qwen35-4b.json \
        --dst M-draft.gguf

Source: FR-Spec (arXiv 2502.14856), which restricts the drafter of EAGLE-2
to a static frequency-ranked subset of the vocabulary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"
TOKENIZE_BIN = ROOT / "build" / "host-llama" / "bin" / "llama-tokenize"

# The head that the drafter shares with the main graph. A tied checkpoint has
# no output.weight and llama.cpp reads the head from the embedding table.
HEAD_NAMES = ("output.weight", "token_embd.weight")

# The token types of the GGUF vocabulary that the subset always holds. A
# drafter that cannot propose the end of a turn ends every draft one token
# early, thus these carry far more weight than their frequency shows.
FORCED_TOKEN_TYPES = (2, 3, 4, 5, 6)  # CONTROL, USER_DEFINED, UNUSED, BYTE, UNKNOWN


def _import_gguf() -> Any:
    """Import the gguf package of the pinned llama.cpp.

    Returns:
        The gguf module
    """
    sys.path.insert(0, str(LLAMA_DIR / "gguf-py"))
    import gguf  # noqa: PLC0415

    return gguf


@dataclass
class CandidateSet:
    """The subset of the vocabulary that the drafter can propose.

    Attributes:
        ids: The vocabulary indices, in ascending order
        coverage: The share of the corpus occurrences that the subset holds
        n_vocab: The number of rows of the full head
        forced: The number of ids that a token type forced into the subset
        corpus_tokens: The number of tokens of the corpus
        sources: The corpus files, for the record
    """

    ids: list[int]
    coverage: float
    n_vocab: int
    forced: int
    corpus_tokens: int
    sources: list[str] = field(default_factory=list)


def tokenize(model: Path, corpus: Path, binary: Path) -> list[int]:
    """Give the token ids of one corpus file, from the tokenizer of the model.

    The tokenizer of the model is the only correct one. A different tokenizer
    gives a frequency table over a different vocabulary.

    Args:
        model: The GGUF that holds the vocabulary
        corpus: The text file
        binary: The llama-tokenize of the same tree

    Returns:
        The token ids, in the order of the corpus

    Raises:
        RuntimeError: If the tokenizer fails or gives no ids
    """
    out = subprocess.run(
        [str(binary), "-m", str(model), "-f", str(corpus), "--ids", "--no-bos", "--log-disable"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed on {corpus}: {out.stderr.strip()[-400:]}")
    text = out.stdout.strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise RuntimeError(f"no id list in the output of llama-tokenize on {corpus}")
    ids = [int(x) for x in text[start + 1 : end].replace("\n", " ").split(",") if x.strip()]
    if not ids:
        raise RuntimeError(f"llama-tokenize gave no ids for {corpus}")
    return ids


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


def count_parquet(job: tuple[str, str, str | None, int]) -> tuple[dict[int, int], int, int]:
    """Count the token ids of one parquet file.

    The file streams in batches, thus the memory of a worker stays flat
    whatever the size of the file. Complexity is O(C) in the characters of
    the file.

    Args:
        job: The parquet path, the tokenizer path, the column name or None to
            detect it, and the maximum number of rows to read

    Returns:
        The counts, the number of tokens, and the number of rows
    """
    import pyarrow.parquet as pq  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    path, tok_path, column, max_rows = job
    tok = Tokenizer.from_file(tok_path)
    tok.no_truncation()
    tok.no_padding()

    pf = pq.ParquetFile(path)
    if column is None:
        names = set(pf.schema_arrow.names)
        column = next((c for c in ("text", "content", "code") if c in names), None)
        if column is None:
            return {}, 0, 0

    counts: Counter[int] = Counter()
    n_tok = n_row = 0
    for batch in pf.iter_batches(batch_size=512, columns=[column]):
        rows = [s for s in batch.column(0).to_pylist() if s]
        if not rows:
            continue
        for enc in tok.encode_batch_fast(rows, add_special_tokens=False):
            counts.update(enc.ids)
            n_tok += len(enc.ids)
        n_row += len(rows)
        if max_rows and n_row >= max_rows:
            break
    return dict(counts), n_tok, n_row


def count_corpus(files: list[Path], tokenizer: Path, column: str | None,
                 workers: int, max_rows: int) -> tuple[Counter[int], int, int]:
    """Count the token ids of a set of parquet files, one process per file.

    Args:
        files: The parquet files
        tokenizer: The tokenizer.json of the model
        column: The text column, or None to detect it per file
        workers: The number of processes
        max_rows: The maximum number of rows per file, or 0 for all of them

    Returns:
        The counts, the number of tokens, and the number of rows
    """
    import multiprocessing as mp  # noqa: PLC0415

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    jobs = [(str(f), str(tokenizer), column, max_rows) for f in files]
    total: Counter[int] = Counter()
    n_tok = n_row = 0
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for i, (c, t, r) in enumerate(pool.imap_unordered(count_parquet, jobs), 1):
            total.update(c)
            n_tok += t
            n_row += r
            print(f"draftset: {i}/{len(jobs)} files, {n_tok / 1e9:.3f} G tokens", flush=True)
    return total, n_tok, n_row


def rank(model: Path, corpora: list[Path], count: int, binary: Path) -> CandidateSet:
    """Build the frequency-ranked subset of the vocabulary.

    The forced ids come first, then the most frequent of the remaining ids,
    until the subset holds ``count`` entries. Complexity is O(T + V log V)
    for T corpus tokens and V vocabulary entries.

    Args:
        model: The GGUF that holds the vocabulary and the head
        corpora: The text files of the frequency table
        count: The number of rows of the reduced head
        binary: The llama-tokenize of the same tree

    Returns:
        The subset

    Raises:
        ValueError: If count does not fit the vocabulary
    """
    gguf = _import_gguf()
    reader = gguf.GGUFReader(str(model))
    head = next((t for name in HEAD_NAMES for t in reader.tensors if t.name == name), None)
    if head is None:
        raise ValueError(f"{model} holds none of {HEAD_NAMES}")
    n_vocab = int(head.shape[1])
    if not 0 < count <= n_vocab:
        raise ValueError(f"count {count} is not in 1..{n_vocab}")

    counts: Counter[int] = Counter()
    for c in corpora:
        counts.update(tokenize(model, c, binary))
    total = sum(counts.values())

    forced = forced_ids(reader, n_vocab)
    if len(forced) > count:
        raise ValueError(f"{len(forced)} forced ids do not fit a subset of {count}")
    chosen = set(forced)
    # A tie on the count keeps the lower id, thus the subset is reproducible.
    for tok, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(chosen) >= count:
            break
        chosen.add(tok)
    ids = sorted(chosen)
    covered = sum(counts[i] for i in ids)
    return CandidateSet(
        ids=ids,
        coverage=covered / total if total else 0.0,
        n_vocab=n_vocab,
        forced=len(forced),
        corpus_tokens=total,
        sources=[str(c) for c in corpora],
    )


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
    gguf = _import_gguf()
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


def cmd_rank(a: argparse.Namespace) -> int:
    """Write the candidate set of a model as JSON.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    cs = rank(a.model, a.corpus, a.count, a.tokenizer)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "model": str(a.model),
                "n_vocab": cs.n_vocab,
                "count": len(cs.ids),
                "forced": cs.forced,
                "corpus_tokens": cs.corpus_tokens,
                "coverage": cs.coverage,
                "sources": cs.sources,
                "ids": cs.ids,
            },
            indent=1,
        )
    )
    print(
        f"draftset: {len(cs.ids)} of {cs.n_vocab} rows ({100 * len(cs.ids) / cs.n_vocab:.2f} %), "
        f"{cs.forced} forced, coverage {100 * cs.coverage:.3f} % of {cs.corpus_tokens} tokens"
    )
    print(f"draftset: {a.out}")
    return 0


def cmd_count(a: argparse.Namespace) -> int:
    """Write the frequency table of a parquet corpus as JSON.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    files = sorted(f for d in a.dir for f in Path(d).rglob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet file below {a.dir}")
    print(f"draftset: {len(files)} files, {a.workers} workers", flush=True)
    counts, n_tok, n_row = count_corpus(files, a.tokenizer, a.column, a.workers, a.max_rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "tokenizer": str(a.tokenizer), "files": len(files), "rows": n_row,
        "tokens": n_tok, "distinct": len(counts),
        "counts": {str(k): v for k, v in counts.items()},
    }))
    print(f"draftset: {n_tok} tokens, {len(counts)} distinct, {a.out}")
    return 0


def cmd_select(a: argparse.Namespace) -> int:
    """Turn one or more frequency tables into a candidate set.

    Each table carries a weight, thus a corpus that is smaller than the
    traffic it stands for can still hold its share of the subset.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    gguf = _import_gguf()
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

    r = sub.add_parser("rank", help="build the candidate set from a corpus")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--corpus", type=Path, action="append", required=True)
    r.add_argument("--count", type=int, default=3072)
    r.add_argument("--tokenizer", type=Path, default=TOKENIZE_BIN)
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(fn=cmd_rank)

    c = sub.add_parser("count", help="write the frequency table of a parquet corpus")
    c.add_argument("--dir", action="append", required=True)
    c.add_argument("--tokenizer", type=Path, required=True)
    c.add_argument("--column", default=None, help="the text column, detected per file when absent")
    c.add_argument("--workers", type=int, default=32)
    c.add_argument("--max-rows", type=int, default=0, help="per file, 0 for all of them")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(fn=cmd_count)

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
