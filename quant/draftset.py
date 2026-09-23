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

The pipeline has four commands. A row-frequency table comes from one of two
sources, or from both blended:

- ``count``: the token frequency of a corpus, from a fast parquet pass. It
  approximates the output distribution of the target model. Cheap.
- ``calibrate``: the tokens that the full MTP head actually proposes, from a
  run of the drafter over calibration prompts. It is the acceptance signal
  itself, because a greedy drafter proposes the argmax of its own logits.

Then:

- ``select``: blend one or more tables and keep the top rows as a candidate
  set of ids.
- ``inject``: gather those rows and write the three draft tensors into a copy
  of the GGUF.

The three tensors that ``inject`` adds, all optional and read only by the
MTP graph:

- ``blk.N.nextn.draft_head.weight``, the gathered rows, in the type of the
  source head
- ``blk.N.nextn.draft_ids``, the vocabulary index of each gathered row, as
  I32, which the graph gives to ``ggml_set_rows`` as the scatter index
- ``blk.N.nextn.draft_fill``, one F32 that holds the logit of every token
  outside the subset.

An example, calibration blended with corpus frequency:

    python -m quant.draftset prompts --dir corpus --tokenizer tok.json \
        --count 400 --out prompts.txt
    python -m quant.draftset calibrate --model M.gguf --prompts prompts.txt \
        --bin build/host-llama/bin/llama-speculative-simple \
        --libdir build/host-llama/bin --gen 256 --out calib.json
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
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"

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


_TOKENIZER_CACHE: dict[str, Any] = {}


def _worker_tokenizer(path: str) -> Any:
    """Give the tokenizer of a worker process, loaded one time.

    The file is 12.8 MB of JSON, thus a load for every row group would cost
    more than the tokenization itself.

    Args:
        path: The tokenizer.json

    Returns:
        The tokenizer
    """
    tok = _TOKENIZER_CACHE.get(path)
    if tok is None:
        from tokenizers import Tokenizer  # noqa: PLC0415

        tok = Tokenizer.from_file(path)
        tok.no_truncation()
        tok.no_padding()
        _TOKENIZER_CACHE[path] = tok
    return tok


def count_parquet(job: tuple[str, str, str | None, int]) -> tuple[dict[int, int], int, int]:
    """Count the token ids of one row group of one parquet file.

    A shard of a web corpus is several gigabytes, thus one process per file
    would leave most cores idle. The unit of work is one row group, which
    gives hundreds of jobs from a handful of files. Complexity is O(C) in the
    characters of the row group.

    Args:
        job: The parquet path, the tokenizer path, the column name or None to
            detect it, and the index of the row group

    Returns:
        The counts, the number of tokens, and the number of rows
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    path, tok_path, column, group = job
    tok = _worker_tokenizer(tok_path)

    pf = pq.ParquetFile(path)
    if column is None:
        names = set(pf.schema_arrow.names)
        column = next((c for c in ("text", "content", "code") if c in names), None)
        if column is None:
            return {}, 0, 0

    table = pf.read_row_group(group, columns=[column])
    rows = [s for s in table.column(0).to_pylist() if s]
    counts: Counter[int] = Counter()
    n_tok = 0
    for i in range(0, len(rows), 1024):
        for enc in tok.encode_batch_fast(rows[i : i + 1024], add_special_tokens=False):
            counts.update(enc.ids)
            n_tok += len(enc.ids)
    return dict(counts), n_tok, len(rows)


def parquet_jobs(files: list[Path], tokenizer: Path, column: str | None) -> list[tuple]:
    """Give one job per row group of every file.

    Args:
        files: The parquet files
        tokenizer: The tokenizer.json of the model
        column: The text column, or None to detect it per file

    Returns:
        The jobs, in the order of the files
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    per_file = [(str(f), pq.ParquetFile(f).num_row_groups) for f in files]
    jobs: list[tuple] = []
    for g in range(max((n for _, n in per_file), default=0)):
        jobs.extend((path, str(tokenizer), column, g) for path, n in per_file if g < n)
    return jobs


def count_corpus(files: list[Path], tokenizer: Path, column: str | None,
                 workers: int, max_rows: int) -> tuple[Counter[int], int, int]:
    """Count the token ids of a set of parquet files, one process per row group.

    Args:
        files: The parquet files
        tokenizer: The tokenizer.json of the model
        column: The text column, or None to detect it per file
        workers: The number of processes
        max_rows: The maximum number of row groups, or 0 for all of them

    Returns:
        The counts, the number of tokens, and the number of rows
    """
    import multiprocessing as mp  # noqa: PLC0415

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    jobs = parquet_jobs(files, tokenizer, column)
    if max_rows:
        jobs = jobs[:max_rows]
    print(f"draftset: {len(files)} files, {len(jobs)} row groups, {workers} workers", flush=True)
    total: Counter[int] = Counter()
    n_tok = n_row = 0
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for i, (c, t, r) in enumerate(pool.imap_unordered(count_parquet, jobs, chunksize=1), 1):
            total.update(c)
            n_tok += t
            n_row += r
            if i % 20 == 0 or i == len(jobs):
                print(f"draftset: {i}/{len(jobs)} groups, {n_tok / 1e9:.3f} G tokens", flush=True)
    return total, n_tok, n_row


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


# The debug line of the drafter: one per candidate per position. Candidate 0
# is the token that the greedy drafter proposes, thus the ranks 1 and above
# only matter for a stochastic drafter. The prefix and the piece do not enter.
DRAFT_LINE = re.compile(
    r"draft candidate\s+(?P<rank>\d+), pos\s+\d+:\s+(?P<id>\d+)\s+\(\s*(?P<prob>[\d.]+)\)"
)


def sample_prompts(files: list[Path], tokenizer: Path, count: int,
                   prompt_tokens: int, seed: int) -> list[str]:
    """Give short prompts from a parquet corpus for the calibration run.

    Each prompt is the first ``prompt_tokens`` tokens of one document,
    decoded back to text. A short prompt seeds a generation, thus the
    drafter proposes over the model's own continuation.

    Args:
        files: The parquet files
        tokenizer: The tokenizer.json of the model
        count: The number of prompts
        prompt_tokens: The token length of each prompt
        seed: The seed of the document choice

    Returns:
        The prompts, at most ``count`` of them
    """
    import random  # noqa: PLC0415

    import pyarrow.parquet as pq  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415

    tok = Tokenizer.from_file(str(tokenizer))
    rng = random.Random(seed)
    out: list[str] = []
    per_file = max(1, count // max(1, len(files)))
    for f in files:
        pf = pq.ParquetFile(f)
        col = next((c for c in ("text", "content", "code") if c in set(pf.schema_arrow.names)), None)
        if col is None:
            continue
        rows = pf.read_row_group(0, columns=[col]).column(0).to_pylist()
        rows = [s for s in rows if s and len(s) > 200]
        rng.shuffle(rows)
        for s in rows[:per_file]:
            ids = tok.encode(s, add_special_tokens=False).ids[:prompt_tokens]
            text = tok.decode(ids).strip()
            if text:
                out.append(text)
            if len(out) >= count:
                return out
    return out


def run_draft(model: Path, prompt: str, binary: Path, libdir: Path | None,
              gen: int, draft_max: int, ctx: int, ngl: int, seed: int) -> list[tuple[int, int, float]]:
    """Run the full-head MTP drafter over one prompt and give its proposals.

    The drafter runs at debug verbosity, thus it prints the candidate tokens
    of every draft position. This function parses those lines. Complexity is
    O(G) in the generated tokens.

    Args:
        model: The GGUF with the full MTP head
        prompt: The seed prompt
        binary: The llama-speculative-simple of a build that supports draft-mtp
        libdir: The directory of the shared libraries, or None for the system path
        gen: The number of tokens to generate
        draft_max: The draft depth
        ctx: The context length
        ngl: The number of layers on the GPU, 0 for the CPU
        seed: The seed of the generation

    Returns:
        The proposals as (rank, token id, probability)

    Raises:
        RuntimeError: If the binary fails
    """
    env = dict(os.environ)
    if libdir is not None:
        env["LD_LIBRARY_PATH"] = f"{libdir}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [str(binary), "-m", str(model), "--spec-type", "draft-mtp",
           "-p", prompt, "-n", str(gen), "--temp", "0", "--seed", str(seed),
           "-c", str(ctx), "--spec-draft-n-max", str(draft_max), "-lv", "5"]
    if ngl > 0:
        cmd += ["-ngl", str(ngl)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"the drafter failed: {proc.stderr.strip()[-400:]}")
    out: list[tuple[int, int, float]] = []
    for line in proc.stderr.splitlines():
        m = DRAFT_LINE.search(line)
        if m:
            out.append((int(m["rank"]), int(m["id"]), float(m["prob"])))
    return out


def calibrate(model: Path, prompts: list[str], binary: Path, libdir: Path | None,
              gen: int, draft_max: int, ctx: int, ngl: int,
              weight_prob: bool, topk: int) -> tuple[dict[int, float], float, int]:
    """Rank the vocabulary by what the full MTP head proposes.

    The default counts one occurrence for the greedy proposal at each draft
    position, because a greedy drafter only ever draws the argmax. The
    ``weight_prob`` option adds the top-k candidates weighted by probability,
    which is the signal a stochastic drafter needs.

    Args:
        model: The GGUF with the full MTP head
        prompts: The seed prompts
        binary: The llama-speculative-simple of a build that supports draft-mtp
        libdir: The directory of the shared libraries, or None
        gen: The number of tokens to generate per prompt
        draft_max: The draft depth
        ctx: The context length
        ngl: The number of layers on the GPU
        weight_prob: True to weight the top-k candidates by probability
        topk: The number of candidates to count when weight_prob is true

    Returns:
        The weighted counts, the total weight, and the number of positions
    """
    counts: dict[int, float] = {}
    total = 0.0
    positions = 0
    for i, prompt in enumerate(prompts, 1):
        props = run_draft(model, prompt, binary, libdir, gen, draft_max, ctx, ngl, seed=i)
        for rank, tok, prob in props:
            if weight_prob:
                if rank < topk:
                    counts[tok] = counts.get(tok, 0.0) + prob
                    total += prob
            elif rank == 0:
                counts[tok] = counts.get(tok, 0.0) + 1.0
                total += 1.0
                positions += 1
        if i % 10 == 0 or i == len(prompts):
            print(f"draftset: {i}/{len(prompts)} prompts, {int(total)} proposals, "
                  f"{len(counts)} distinct", flush=True)
    return counts, total, positions


def cmd_prompts(a: argparse.Namespace) -> int:
    """Write calibration prompts from a parquet corpus, one JSON string per line.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    files = sorted(f for d in a.dir for f in Path(d).rglob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet file below {a.dir}")
    prompts = sample_prompts(files, a.tokenizer, a.count, a.prompt_tokens, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in prompts) + "\n")
    print(f"draftset: {len(prompts)} prompts of {a.prompt_tokens} tokens, {a.out}")
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    """Write the proposal table of the full MTP head as JSON.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    prompts = [json.loads(line) for line in a.prompts.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompt in {a.prompts}")
    print(f"draftset: {len(prompts)} prompts, gen {a.gen}, draft {a.draft_max}, ngl {a.ngl}",
          flush=True)
    counts, total, positions = calibrate(
        a.model, prompts, a.bin, a.libdir, a.gen, a.draft_max, a.ctx, a.ngl,
        a.weight_prob, a.topk)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "model": str(a.model), "prompts": len(prompts), "positions": positions,
        "weight_prob": a.weight_prob, "tokens": total, "distinct": len(counts),
        "counts": {str(k): v for k, v in counts.items()},
    }))
    print(f"draftset: {int(total)} proposals, {len(counts)} distinct, {a.out}")
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

    c = sub.add_parser("count", help="write the frequency table of a parquet corpus")
    c.add_argument("--dir", action="append", required=True)
    c.add_argument("--tokenizer", type=Path, required=True)
    c.add_argument("--column", default=None, help="the text column, detected per file when absent")
    c.add_argument("--workers", type=int, default=32)
    c.add_argument("--max-rows", type=int, default=0, help="row groups in total, 0 for all of them")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(fn=cmd_count)

    pr = sub.add_parser("prompts", help="write calibration prompts from a parquet corpus")
    pr.add_argument("--dir", action="append", required=True)
    pr.add_argument("--tokenizer", type=Path, required=True)
    pr.add_argument("--count", type=int, default=400)
    pr.add_argument("--prompt-tokens", type=int, default=64)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(fn=cmd_prompts)

    cal = sub.add_parser("calibrate", help="rank the vocabulary by the proposals of the full MTP head")
    cal.add_argument("--model", type=Path, required=True)
    cal.add_argument("--prompts", type=Path, required=True)
    cal.add_argument("--bin", type=Path, required=True, help="llama-speculative-simple of a draft-mtp build")
    cal.add_argument("--libdir", type=Path, default=None, help="the shared-library directory of that build")
    cal.add_argument("--gen", type=int, default=256, help="tokens generated per prompt")
    cal.add_argument("--draft-max", type=int, default=3)
    cal.add_argument("--ctx", type=int, default=2048)
    cal.add_argument("--ngl", type=int, default=0, help="layers on the GPU, 0 for the CPU")
    cal.add_argument("--weight-prob", action="store_true",
                     help="count the top-k candidates by probability, not the greedy pick alone")
    cal.add_argument("--topk", type=int, default=3)
    cal.add_argument("--out", type=Path, required=True)
    cal.set_defaults(fn=cmd_calibrate)

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
