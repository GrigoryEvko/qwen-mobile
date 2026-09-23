#!/usr/bin/env python3
"""Make a problem file for the candidate-kernel harness ("isaprobe kernel").

    mv_case.py make <problem.bin> --rows R --cols K [--seed S] [--name LABEL] [--outliers F]

The problem is the Q8_0 matrix-vector product of the ggml CPU path: W has R rows of K / 32
block_q8_0 blocks (an fp16 scale d, then 32 int8 codes), and x has K f32 values. The file format
is that of mv_format.h: a 128-byte header with the magic "IPMV", then W, then x.

The weights look like the blocks of a quantized model: in each block one code is +127 or -127
and the others are uniform in [-127, 127], and d is uniform in [1e-4, 2e-2]. The activation is
normal with mean 0 and deviation 1, and a fraction F of the values (default 0.01) is 20 times
larger, as the outliers of real activations are.

The script uses only the standard library, thus it also runs in the build container.
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
from pathlib import Path

QK8_0 = 32
HEADER = struct.Struct("<4s7IQQiI64s8s")
OP_MATVEC_Q8_0 = 1


def make_weights(rng: random.Random, rows: int, cols: int) -> bytes:
    """Return rows x cols / 32 block_q8_0 blocks (34 bytes each). O(rows x cols)."""
    out = bytearray()
    for _ in range(rows * (cols // QK8_0)):
        d = rng.uniform(1e-4, 2e-2)
        codes = [rng.randint(-127, 127) for _ in range(QK8_0)]
        codes[rng.randrange(QK8_0)] = rng.choice((-127, 127))
        out += struct.pack("<e", d) + struct.pack(f"<{QK8_0}b", *codes)
    return bytes(out)


def make_activation(rng: random.Random, cols: int, outliers: float) -> bytes:
    """Return cols f32 values: normal values, and a fraction of outliers 20 times larger. O(cols)."""
    xs = [rng.gauss(0.0, 1.0) * (20.0 if rng.random() < outliers else 1.0) for _ in range(cols)]
    return struct.pack(f"<{cols}f", *xs)


def make(path: Path, rows: int, cols: int, seed: int, name: str, outliers: float) -> None:
    """Write one problem file. O(rows x cols)."""
    if rows <= 0 or cols <= 0 or cols % QK8_0 != 0:
        raise ValueError(f"rows must be positive and cols a positive multiple of {QK8_0} (rows {rows}, cols {cols})")
    rng = random.Random(seed)
    w = make_weights(rng, rows, cols)
    x = make_activation(rng, cols, outliers)
    header = HEADER.pack(b"IPMV", 1, OP_MATVEC_Q8_0, rows, cols, 0, 0, 0, 0, 0, 0, seed,
                         name.encode()[:63].ljust(64, b"\0"), bytes(8))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + w + x)
    print(f"mv_case: {path}: {rows} x {cols} Q8_0 matvec, seed {seed}, {len(w) + len(x) + HEADER.size} bytes")


def main(argv: list[str]) -> int:
    """Parse the command line and run the "make" command. Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_make = sub.add_parser("make", help="write a problem file")
    p_make.add_argument("path", type=Path)
    p_make.add_argument("--rows", type=int, required=True)
    p_make.add_argument("--cols", type=int, required=True)
    p_make.add_argument("--seed", type=int, default=1)
    p_make.add_argument("--name", default="")
    p_make.add_argument("--outliers", type=float, default=0.01)
    args = parser.parse_args(argv)
    make(args.path, args.rows, args.cols, args.seed, args.name or f"q8_0_{args.rows}x{args.cols}", args.outliers)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
