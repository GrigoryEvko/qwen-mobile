"""Fuzz the gguf-py reader with truncated and corrupted files, and compare it with the ggml loader.

The seeds are files of the pipeline: toy models that export() wrote, in
Q8_0, Q4_0 and IQ4_NL, tied and untied. A drawn example applies 1 to 4
mutations to a seed: a cut, a bit flip, an interesting integer in a field
with a role (array count, string length, dimension, tensor offset), an
insertion or a deletion of bytes.

The properties of the reader, which runs in an isolated worker process:

- It ends in 5 s with less than 2 GB of address space.
- It gives a result or an exception of the types in ALLOWED.
- When it gives a result, the data of each tensor is inside the data
  section of the file.

The ggml loader check (metadata mode, the one sanitizer of the run) reads
the same bytes. A sanitizer report keeps one input per distinct report in
build/fuzz/quant-<config>/ggml-loader-reports. run.sh counts each kept
report as a finding of the target loader-check (rule R8), and the test
fails at once only with QFZ_GGML_STRICT=1, thus the fuzzer goes on to the
properties of gguf-py. A file that gguf-py reads and ggml refuses is only
counted, because gguf-py checks less by design.
"""

from __future__ import annotations

import logging
import os
from collections import Counter

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from qfz_checks import check_available, ggml_loader_status, record_ggml_report
from qfz_common import LLAMA_DIR, REGRESS_DIR, SEED_DIR, san_dir, scratch
from qfz_hyp import counted, fuzz_settings
from qfz_rawgguf import INTERESTING_U32, INTERESTING_U64, IntField, put_int, walk
from qfz_reader_worker import ReaderPool
from qfz_toy import SMALL, output_rot_for, write_source
from quant.export import export
from quant.plan import Plan

LOG = logging.getLogger("qfz")
ALLOWED = ("ValueError", "KeyError", "UnicodeDecodeError", "OverflowError", "QuantError")
GGML_REPORTS = san_dir() / "ggml-loader-reports"
SEED_DIRS = (SEED_DIR / "reader", REGRESS_DIR / "reader")
STATS: Counter[str] = Counter()


class Seeds(list):
    """The seed files. The short repr keeps the bytes out of a failure report."""

    def __repr__(self) -> str:
        """Give the count and the sizes of the seeds."""
        return f"Seeds({[len(s) for s in self]} bytes)"


@pytest.fixture(scope="module")
def seeds() -> Seeds:
    """Give the seed files: toy exports of the pipeline in three types, tied and untied, then the committed seeds.

    The committed seeds are tests/fuzz/quant/seeds/reader (two toy exports)
    and tests/fuzz/quant/regress/reader (the minimal files of QR1, QR2 and QR4),
    thus the mode "test" replays them.
    """
    out = Seeds()
    with scratch() as tmp:
        for i, (kind, tie) in enumerate((("Q8_0", False), ("Q4_0", True), ("IQ4_NL", False))):
            geo = SMALL
            src = write_source(tmp / f"src{i}.gguf", geo, "normal", i, tied=tie)
            rot = None
            if tie:
                rot = tmp / "rot.npy"
                np.save(rot, output_rot_for(geo, i))
            plan = Plan(bulk=kind, head=kind, embedding="Q8_0", n_layers=geo.n_layer)
            export(src.path, tmp / f"out{i}.gguf", tmp / "no-packs", plan, LLAMA_DIR, torch.device("cpu"),
                   tie_head=tie, rot=rot)
            out.append((tmp / f"out{i}.gguf").read_bytes())
    out.extend(p.read_bytes() for d in SEED_DIRS for p in sorted(d.glob("*.seed")))
    return out


@pytest.fixture(scope="module")
def pool() -> ReaderPool:
    """Give one reader worker for the module, and stop it after the module."""
    p = ReaderPool()
    yield p
    p.close()
    LOG.info("reader outcomes: %s", dict(STATS))


@st.composite
def mutations(draw: st.DrawFn) -> tuple[int, list[tuple]]:
    """Draw a seed index (taken modulo the seed count) and 0 to 4 mutations, as data that apply() reads."""
    seed = draw(st.integers(0, 63))
    ops = []
    for _ in range(draw(st.integers(0, 4))):
        op = draw(st.sampled_from(("cut", "flip", "field", "field", "field", "insert", "delete")))
        if op == "cut":
            ops.append(("cut", draw(st.floats(0.0, 1.0))))
        elif op == "flip":
            ops.append(("flip", draw(st.floats(0.0, 1.0)), draw(st.integers(0, 7))))
        elif op == "field":
            ops.append(("field", draw(st.integers(0, 10**6)), draw(st.integers(0, 10**6)),
                        draw(st.integers(-3, 3))))
        elif op == "insert":
            ops.append(("insert", draw(st.floats(0.0, 1.0)), draw(st.binary(min_size=1, max_size=16))))
        else:
            ops.append(("delete", draw(st.floats(0.0, 1.0)), draw(st.integers(1, 64))))
    return seed, ops


def apply(data: bytes, ops: list[tuple]) -> bytes:
    """Apply the mutations to a copy of the bytes and give the result. Complexity is O(len(data)) per mutation."""
    out = bytearray(data)
    for op in ops:
        if not out:
            break
        if op[0] == "cut":
            del out[int(op[1] * len(out)):]
        elif op[0] == "flip":
            pos = min(int(op[1] * len(out)), len(out) - 1)
            out[pos] ^= 1 << op[2]
        elif op[0] == "field":
            fields = walk(bytes(out))
            if fields:
                field: IntField = fields[op[1] % len(fields)]
                table = INTERESTING_U64 if field.width == 8 else INTERESTING_U32
                put_int(out, field, table[op[2] % len(table)] + op[3])
        elif op[0] == "insert":
            pos = int(op[1] * len(out))
            out[pos:pos] = op[2]
        else:
            pos = int(op[1] * len(out))
            del out[pos:pos + op[2]]
    return bytes(out)


def _classify(result: dict) -> str | None:
    """Give None for an accepted outcome, or the reason of a failure."""
    status = result["status"]
    if status in ("timeout", "memory", "crash"):
        return f"the reader gave '{status}'"
    if status == "error":
        if result["type"] in ALLOWED:
            STATS[f"error {result['type']}"] += 1
            return None
        return f"the reader raised {result['type']}: {result['message']}"
    STATS["ok"] += 1
    for t in result["tensors"]:
        if not (result["data_offset"] <= t["offset"] and t["offset"] + t["nbytes"] <= result["size"]):
            return f"the reader accepts tensor {t['name']} at {t['offset']} + {t['nbytes']}, out of the data " \
                   f"section {result['data_offset']} .. {result['size']}"
    return None


@fuzz_settings()
@given(case=mutations())
@example(case=(0, []))
@example(case=(1, [("cut", 0.5)]))
@example(case=(2, [("field", 2, 0, 2)]))
@example(case=(0, [("field", 1416, 0, -1)]))
@example(case=(1, [("insert", 0.078125, b"\x80")]))
@example(case=(3, []))
@example(case=(4, []))
@counted
def test_reader_survives_corrupted_pipeline_files(seeds: list[bytes], pool: ReaderPool,
                                                  case: tuple[int, list[tuple]]) -> None:
    """A corrupted file of the pipeline gives a clean exception or a sound result, in the time and memory limits."""
    index, ops = case
    seed = index % len(seeds)
    data = apply(seeds[seed], ops)
    with scratch() as tmp:
        path = tmp / "case.gguf"
        path.write_bytes(data)
        if not data:
            return
        reason = _classify(pool.read(path))
        assert reason is None, f"{reason} (mutations {ops} of seed {seed}, {len(data)} bytes)"
        if check_available():
            status, summary = ggml_loader_status(path)
            if status == 3:
                STATS["ggml refused"] += 1
            elif status != 0:
                # The ggml loader belongs to the core fuzz area: keep one input per report, fail only when asked.
                kept = record_ggml_report(data, summary, GGML_REPORTS)
                STATS[f"ggml report {summary[:90]}"] += 1
                assert os.environ.get("QFZ_GGML_STRICT") != "1", f"the ggml loader check gave {status}: {summary} " \
                                                                  f"(input kept in {kept})"
