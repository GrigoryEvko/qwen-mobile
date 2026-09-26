#!/usr/bin/env python3
"""Write the seed inputs of the hexhost target fuzz_dirty into tests/fuzz/hexhost/corpus/dirty.

fuzz_dirty reads its input with FuzzedDataProvider. Each integral value comes from the end of the
input, the most significant byte first, with the number of bytes that its range needs (no byte for a
range of one value). Thus this script writes the values of a seed in the sequence in which the target
reads them, and it reverses the bytes at the end:
  - the number of threads (1 to HTP_MAX_NTHREADS)
  - for each tensor of the pool: big (0 gives a size up to 9 MiB, else up to 64 KiB), the size, the
    offset in the arena (a multiple of 4), the flags (0 gives a weight tensor)
  - the number of ops, then for each op: the number of inputs, the number of outputs, the input
    indices, the output indices and the event (0 a fence, 1 the end of the batch, else none)
The constants below are the constants of tests/fuzz/hexhost/fuzz_dirty.cpp and of the stubs. A change
of the input format of the target needs the same change here, and new seeds.

Each seed reaches one group of branches of htp/htp-tensor.c. Its name tells which.

Usage:
    uv run --frozen python tests/fuzz/hexhost/tools/make_dirty_seeds.py [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

HTP_MAX_NTHREADS = 10  # htp-ops.h
HTP_MAX_DIRTY_RANGES = 32  # stubs/dsp/htp-ctx.h
POOL = HTP_MAX_DIRTY_RANGES + 8  # fuzz_dirty.cpp: more tensors than ranges, thus the tracker can evict
SPAN = 24 << 20  # fuzz_dirty.cpp: the span of the arena
BIG_MAX = 9 << 20
SMALL_MAX = 64 << 10
KIB = 1 << 10
MIB = 1 << 20
# The offset of the tensors that a seed does not name: 256 bytes each, 64 KiB apart, above 16 MiB. They
# touch no other tensor, thus each one gets a range of its own in the tracker.
DEFAULT_BASE = 16 * MIB
DEFAULT_STEP = 64 * KIB
DEFAULT_SIZE = 256


@dataclasses.dataclass
class Tensor:
    """One tensor of the pool: its offset in the arena, its size, and its flags."""

    offset: int
    size: int
    weight: bool = False


@dataclasses.dataclass
class Op:
    """One op: the indices of its inputs and outputs, and the event after it."""

    srcs: list[int]
    dsts: list[int]
    event: str = "none"


class Seed:
    """The values of one input in the sequence of the reads of the target."""

    def __init__(self) -> None:
        self.stream = bytearray()

    def integral(self, lo: int, hi: int, value: int) -> None:
        """Add the bytes that ConsumeIntegralInRange(lo, hi) reads to give value."""
        if not lo <= value <= hi:
            raise ValueError(f"{value} is not in [{lo}, {hi}]")
        span = hi - lo
        n = (span.bit_length() + 7) // 8
        self.stream += (value - lo).to_bytes(n, "big") if n else b""

    def data(self) -> bytes:
        """Give the input: the target reads it from the end."""
        return bytes(reversed(self.stream))


def encode(threads: int, tensors: dict[int, Tensor], ops: list[Op]) -> bytes:
    """Give the input of one seed. A tensor that the dict does not name gets the default place."""
    seed = Seed()
    seed.integral(1, HTP_MAX_NTHREADS, threads)
    for i in range(POOL):
        t = tensors.get(i, Tensor(DEFAULT_BASE + i * DEFAULT_STEP, DEFAULT_SIZE))
        if t.offset % 4 or t.offset + t.size > SPAN:
            raise ValueError(f"tensor {i}: the offset {t.offset:#x} is not a multiple of 4 inside the arena")
        big = t.size > SMALL_MAX
        seed.integral(0, 7, 0 if big else 1)
        seed.integral(1, BIG_MAX if big else SMALL_MAX, t.size)
        seed.integral(0, SPAN - t.size, t.offset)
        seed.integral(0, 15, 0 if t.weight else 1)
    seed.integral(1, 64, len(ops))
    for op in ops:
        seed.integral(0, 4, len(op.srcs))
        seed.integral(1, 2, len(op.dsts))
        for s in op.srcs:
            seed.integral(0, POOL - 1, s)
        for d in op.dsts:
            seed.integral(0, POOL - 1, d)
        seed.integral(0, 9, {"fence": 0, "batch": 1, "none": 2}[op.event])
    return seed.data()


def fill(first: int, count: int) -> list[Op]:
    """Give ops that write the default tensors first .. first + count - 1, two for each op: each gets a range."""
    return [Op([], list(range(i, min(i + 2, first + count)))) for i in range(first, first + count, 2)]


A = 0x10000  # the place of the tensors of the seeds that make ranges touch


def seeds() -> dict[str, bytes]:
    """Give each seed by its name."""
    out: dict[str, bytes] = {}
    # htp_tensor_flush_all: a dirty input of 4 KiB, one thread: the flush of each range in a loop, and
    # make_tensor_clean of a range that the input covers fully.
    out["input-serial-flush"] = encode(1, {0: Tensor(0x1000, 4 * KIB), 1: Tensor(0x4000, 4 * KIB)},
                                       [Op([], [0]), Op([0], [1])])
    # The same with 4 threads and 8 KiB: the flush goes to the work queue (4 KiB or more).
    out["input-queue-flush"] = encode(4, {0: Tensor(0x1000, 8 * KIB), 1: Tensor(0x8000, 4 * KIB)},
                                      [Op([], [0]), Op([0], [1])])
    # A dirty input above 4 MiB: the flush of the whole data cache.
    out["input-flush-all"] = encode(2, {0: Tensor(0x100000, 5 * MIB), 1: Tensor(0x1000, 4 * KIB)},
                                    [Op([], [0]), Op([0], [1])])
    # Four dirty inputs of 1300 bytes (11 lines, 2.75 blocks of 4 lines) on 10 threads: the work queue
    # with more threads than blocks, the clamp of the last block of each range, and a thread with no block.
    small4 = {i: Tensor(0x1000 + i * 0x2000, 1300) for i in range(4)}
    out["input-queue-10-threads"] = encode(10, {**small4, 4: Tensor(0x20000, 4 * KIB)},
                                           [Op([], [0, 1]), Op([], [2, 3]), Op([0, 1, 2, 3], [4])])
    # htp_flush_dirty_ranges (a fence): 1 KiB on one thread, then 8 KiB on 4 threads (the work queue),
    # then 5 MiB (the whole data cache), then a fence with no dirty range (the output is a weight).
    out["fence-serial"] = encode(1, {0: Tensor(0x1000, 1 * KIB)}, [Op([], [0], "fence")])
    out["fence-paths"] = encode(4, {0: Tensor(0x1000, 1 * KIB), 1: Tensor(0x4000, 8 * KIB),
                                    2: Tensor(0x100000, 5 * MIB), 3: Tensor(0x20000, 4 * KIB, weight=True)},
                                [Op([], [0], "fence"), Op([], [1], "fence"), Op([], [2], "fence"),
                                 Op([3], [3], "fence")])
    # The end of a batch: the full flush clears the ranges, thus the next read finds no dirty input.
    out["batch-end"] = encode(1, {0: Tensor(0x1000, 4 * KIB), 1: Tensor(0x4000, 4 * KIB)},
                              [Op([], [0], "batch"), Op([0], [1])])
    # make_tensor_clean of a range that the input covers at its end, then at its start.
    trim = {0: Tensor(A, 8 * KIB), 1: Tensor(A + 4 * KIB, 8 * KIB), 2: Tensor(0x100000, 1 * KIB)}
    out["trim-end"] = encode(1, trim, [Op([], [0]), Op([1], [2])])
    out["trim-start"] = encode(1, trim, [Op([], [1]), Op([0], [2])])
    # An input strictly inside a range: the tracker keeps the range.
    out["input-inside-range"] = encode(1, {0: Tensor(A, 16 * KIB), 1: Tensor(A + 4 * KIB, 4 * KIB),
                                           2: Tensor(0x100000, 1 * KIB)},
                                       [Op([], [0]), Op([1], [2], "fence")])
    # htp_tensor_dirty_all and merge_dirty_ranges: an output next to a range joins it, two new outputs
    # that touch each other take two free ranges and join, and an output across the gap of two ranges
    # joins the two. Then a range above a new output (the output below it gets a range of its own),
    # and an output across the two: the later range has the lower start.
    merge = {0: Tensor(A, 4 * KIB), 1: Tensor(A + 4 * KIB, 4 * KIB), 2: Tensor(A + 16 * KIB, 4 * KIB),
             3: Tensor(A + 18 * KIB, 4 * KIB), 4: Tensor(A + 8 * KIB, 8 * KIB),
             5: Tensor(A + 56 * KIB, 4 * KIB), 6: Tensor(A + 64 * KIB, 4 * KIB), 7: Tensor(A + 59 * KIB, 6 * KIB)}
    out["merge-ranges"] = encode(1, merge, [Op([], [0]), Op([], [1]), Op([], [2, 3]), Op([], [4]),
                                            Op([], [6]), Op([], [5]), Op([], [7]), Op([4, 7], [0], "fence")])
    # A weight output and a weight input: the tracker skips them.
    out["weights"] = encode(1, {0: Tensor(0x1000, 4 * KIB, weight=True), 1: Tensor(0x4000, 4 * KIB)},
                            [Op([0], [0, 1]), Op([0, 1], [0])])
    # The eviction of htp_tensor_dirty_all: 32 outputs fill each range, and a new output evicts the
    # range of slot 0. One thread and 256 bytes, and one thread and 8 KiB: the flush loop. 4 threads and
    # 8 KiB in slot 0: the work queue. 5 MiB in slot 0: the whole data cache. Two new outputs with one free
    # range: an eviction of one range and the use of the free range.
    out["evict-serial"] = encode(1, {}, fill(0, 32) + [Op([], [32])])
    out["evict-serial-8k"] = encode(1, {0: Tensor(0x1000, 8 * KIB)}, fill(0, 32) + [Op([], [32])])
    out["evict-queue"] = encode(4, {0: Tensor(0x1000, 8 * KIB)}, fill(0, 32) + [Op([], [32, 33])])
    out["evict-flush-all"] = encode(2, {0: Tensor(0x100000, 5 * MIB)}, fill(0, 32) + [Op([], [32, 33])])
    out["evict-one-free"] = encode(1, {}, fill(0, 31) + [Op([], [32, 33]), Op([32, 33], [34], "fence")])
    return out


def main() -> int:
    """Write each seed to its directory."""
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=root / "corpus" / "dirty", help="the directory of the seeds")
    a = p.parse_args()
    for directory, inputs in ((a.out, seeds()),):
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in inputs.items():
            (directory / f"{name}.bin").write_bytes(data)
            print(f"{directory / name}.bin: {len(data)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
