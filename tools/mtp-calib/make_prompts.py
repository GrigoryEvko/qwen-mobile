"""Sample balanced calibration prompts from the corpora on the calibration box.

A calibration run at a temperature above zero still needs many distinct seeds,
because the same seed gives nearly the same continuation and the proposal table
would count it again and again. This script takes an equal share of prompts from
each of the four buckets, and walks the row groups of every file so that the
sample is not limited to the head of one shard.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


def buckets(base: Path) -> dict[str, list[Path]]:
    """Give the four buckets of the mix under the corpus directory.

    Russian reads the complete fineweb-2 corpus, which the box holds next to
    the higher quality shards of the mix.

    Args:
        base: The corpus directory

    Returns:
        The bucket name to its parquet directories
    """
    return {
        "en": [base / "fineweb-edu"],
        "ru": [base / "fineweb2-ru"],
        "code": [base / "github-code"],
        "rest": [base / "mix" / "rest"],
    }


def sample_bucket(dirs: list[Path], count: int, tok: Tokenizer,
                  prompt_tokens: int, seed: int) -> list[str]:
    """Take prompts from one bucket, spread over the files and the row groups.

    Complexity is O(count) in the documents that the function decodes.

    Args:
        dirs: The directories to search for parquet files
        count: The number of prompts to return
        tok: The tokenizer of the model
        prompt_tokens: The token length of each prompt
        seed: The seed of the document choice

    Returns:
        The prompts, at most ``count`` of them
    """
    files = sorted(f for d in dirs for f in Path(d).rglob("*.parquet"))
    if not files:
        return []
    rng = random.Random(seed)
    readers = []
    for f in files:
        pf = pq.ParquetFile(f)
        names = set(pf.schema_arrow.names)
        col = next((c for c in ("text", "content", "code") if c in names), None)
        if col is not None:
            readers.append((pf, col))

    out: list[str] = []
    group = 0
    # Round-robin over the files, one row group at a time, so a short bucket
    # still spreads over every file it has.
    while readers and len(out) < count:
        progressed = False
        for pf, col in readers:
            if group >= pf.num_row_groups or len(out) >= count:
                continue
            progressed = True
            rows = [s for s in pf.read_row_group(group, columns=[col]).column(0).to_pylist()
                    if s and len(s) > 400]
            rng.shuffle(rows)
            for doc in rows:
                # 4000 characters hold far more than prompt_tokens tokens in every
                # script, thus the slice prevents the tokenization of whole documents
                ids = tok.encode(doc[:4000], add_special_tokens=False).ids[:prompt_tokens]
                if len(ids) < prompt_tokens // 2:
                    continue
                text = tok.decode(ids).strip()
                if text:
                    out.append(text)
                if len(out) >= count:
                    break
        if not progressed:
            break
        group += 1
    return out


def main() -> None:
    """Write the balanced prompt file, one JSON string per line, shuffled."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--base", type=Path, default=Path.cwd(),
                    help="the corpus directory, which holds tokenizer.json and the buckets")
    ap.add_argument("--per-bucket", type=int, default=20000)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None, help="default base/prompts-calib.txt")
    a = ap.parse_args()
    base = a.base.resolve()
    tok = Tokenizer.from_file(str(base / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()

    everything: list[str] = []
    for i, (name, dirs) in enumerate(buckets(base).items()):
        got = sample_bucket(dirs, a.per_bucket, tok, a.prompt_tokens, seed=i)
        print(f"{name}: {len(got)} prompts", flush=True)
        everything.extend(got)

    random.Random(0).shuffle(everything)
    out = a.out or base / "prompts-calib.txt"
    out.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in everything) + "\n")
    print(f"wrote {len(everything)} prompts to {out}", flush=True)


if __name__ == "__main__":
    main()
