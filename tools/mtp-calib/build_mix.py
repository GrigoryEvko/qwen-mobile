"""Stream a balanced calibration corpus from Hugging Face into local parquet.

The corpus is four buckets of about equal token counts: English, Russian,
code, and the other human languages. The last bucket spreads its budget over
19 languages of ``epfml/fineweb2-hq``. Each worker reads only the ``text``
column of a shard, thus the large ``embeddings`` column never crosses the
network. Each worker stops at its token budget and writes both a text-only
parquet, for the prompt sampler, and a frequency table, for the selector.

The English and the code buckets stay on the local corpus that the box
already holds (fineweb-edu and github-code), thus this script fetches only
the Russian and the other-language buckets.

A worker writes its frequency table only after it closes its parquet writer,
thus the table is a safe marker of a finished job. A job whose table exists
is skipped, and the script can continue a run that stopped in the middle.
Russian is split over several shard ranges, because one worker for the whole
Russian budget is the slowest part of the run.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tokenizers import Tokenizer

REPO = "epfml/fineweb2-hq"

# A job: the language directory of the repository, the token budget, the
# bucket name, the first shard to read, the tag that names the output files,
# and the corpus directory (with tokenizer.json) as a string, because the
# workers start in fresh processes.
Job = tuple[str, int, str, int, str, str]

# The 19 languages of fineweb2-hq that are not Russian, for the "rest" bucket.
REST_LANGS = [
    "arb_Arab", "ces_Latn", "cmn_Hani", "dan_Latn", "deu_Latn", "ell_Grek",
    "fas_Arab", "fra_Latn", "hun_Latn", "ind_Latn", "ita_Latn", "jpn_Jpan",
    "nld_Latn", "pol_Latn", "por_Latn", "spa_Latn", "swe_Latn", "tur_Latn",
    "vie_Latn",
]

# One shard of fineweb2-hq holds about 79 to 91 M tokens, thus a stride of 3
# shards keeps the Russian ranges apart at a budget of 200 M tokens each.
RU_PARTS = 5
RU_STRIDE = 3
RU_BUDGET = 200_000_000
REST_BUDGET = 55_000_000


def build_lang(job: Job) -> tuple[str, int, int, int]:
    """Stream one language range until its token budget and write its table.

    The function returns at once when the table of the job already exists,
    thus a run that stopped in the middle does the remaining jobs only.
    Complexity is O(T) in the tokens that the worker reads.

    Args:
        job: The language directory, the token budget, the bucket name, the
            first shard to read, the tag that names the output files, and
            the corpus directory

    Returns:
        The tag, the token count, the document count, and the distinct ids
    """
    lang, budget, bucket, start_shard, tag, base = job
    out_dir = Path(base) / "mix" / bucket
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"counts-{tag}.json"
    if table_path.exists():
        done = json.loads(table_path.read_text())
        print(f"{tag}: already complete, {done['tokens'] / 1e6:.1f}M tokens", flush=True)
        return tag, int(done["tokens"]), int(done["docs"]), int(done["distinct"])

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fs = HfFileSystem()
    tok = Tokenizer.from_file(str(Path(base) / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()

    # A parquet left by a killed run has no footer and cannot be read, thus
    # the job starts it again from the beginning.
    parquet_path = out_dir / f"{tag}.parquet"
    if parquet_path.exists():
        parquet_path.unlink()

    writer = pq.ParquetWriter(parquet_path, pa.schema([("text", pa.string())]), compression="zstd")
    counts: Counter[int] = Counter()
    n_tok = n_doc = 0
    shard = start_shard
    try:
        while n_tok < budget:
            path = f"datasets/{REPO}/{lang}/000_{shard:05d}.parquet"
            if not fs.exists(path):
                break
            pf = pq.ParquetFile(path, filesystem=fs)
            for group in range(pf.num_row_groups):
                texts = [s for s in pf.read_row_group(group, columns=["text"]).column(0).to_pylist() if s]
                for i in range(0, len(texts), 1024):
                    batch = texts[i : i + 1024]
                    for enc in tok.encode_batch_fast(batch, add_special_tokens=False):
                        counts.update(enc.ids)
                        n_tok += len(enc.ids)
                    writer.write_table(pa.table({"text": batch}))
                    n_doc += len(batch)
                if n_tok >= budget:
                    break
            print(f"{tag}: shard {shard}, {n_tok / 1e6:.1f}M tokens, {n_doc} docs", flush=True)
            shard += 1
    finally:
        writer.close()

    table_path.write_text(json.dumps({
        "lang": lang, "tag": tag, "tokens": n_tok, "docs": n_doc, "distinct": len(counts),
        "counts": {str(k): v for k, v in counts.items()},
    }))
    return tag, n_tok, n_doc, len(counts)


def safe_build(job: Job) -> tuple[str, int, int, int]:
    """Run one job and turn a failure into an empty result.

    One failed language must not stop the other workers of the pool.

    Args:
        job: The job tuple that build_lang takes

    Returns:
        The result of build_lang, or zeros when the job failed
    """
    try:
        return build_lang(job)
    except Exception as exc:  # noqa: BLE001 - one bad shard must not stop the run
        print(f"{job[4]}: FAILED {type(exc).__name__}: {exc}", flush=True)
        return job[4], 0, 0, 0


def merge(base: Path, bucket: str, tags: list[str], out: Path) -> None:
    """Sum the tables of a bucket into one table.

    Args:
        base: The corpus directory
        bucket: The bucket directory under base/mix
        tags: The tags to merge
        out: The path of the merged table
    """
    total: Counter[int] = Counter()
    n_tok = 0
    missing = []
    for tag in tags:
        p = base / "mix" / bucket / f"counts-{tag}.json"
        if not p.exists():
            missing.append(tag)
            continue
        d = json.loads(p.read_text())
        n_tok += d["tokens"]
        for k, v in d["counts"].items():
            total[int(k)] += v
    out.write_text(json.dumps({
        "tokenizer": str(base / "tokenizer.json"), "bucket": bucket, "tags": tags,
        "tokens": n_tok, "distinct": len(total),
        "counts": {str(k): v for k, v in total.items()},
    }))
    print(f"merge: {bucket} -> {out.name}: {n_tok / 1e9:.3f} G tokens, "
          f"{len(total)} distinct, missing={missing}", flush=True)


def main() -> None:
    """Fetch the buckets that are not complete, then merge the tables."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--base", type=Path, default=Path.cwd(),
                    help="the corpus directory, which holds tokenizer.json and takes mix/")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    base = a.base.resolve()

    ru_tags = [f"rus_Cyrl_p{i}" for i in range(RU_PARTS)]
    jobs: list[Job] = [("rus_Cyrl", RU_BUDGET, "ru", i * RU_STRIDE, ru_tags[i], str(base))
                       for i in range(RU_PARTS)]
    jobs += [(lang, REST_BUDGET, "rest", 0, lang, str(base)) for lang in REST_LANGS]

    todo = [j for j in jobs if not (base / "mix" / j[2] / f"counts-{j[4]}.json").exists()]
    print(f"build_mix: {len(jobs)} jobs, {len(todo)} still to do", flush=True)

    if todo:
        with mp.get_context("spawn").Pool(processes=min(len(todo), a.workers)) as pool:
            for tag, n_tok, n_doc, distinct in pool.imap_unordered(safe_build, todo):
                print(f"done {tag}: {n_tok / 1e6:.1f}M tokens, {n_doc} docs, {distinct} distinct",
                      flush=True)

    merge(base, "ru", ru_tags, base / "counts-ru-hq.json")
    merge(base, "rest", REST_LANGS, base / "counts-rest.json")
    print("BUILD_MIX_OK", flush=True)


if __name__ == "__main__":
    main()
