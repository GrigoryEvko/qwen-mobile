#!/usr/bin/env python3
"""Generate a shape sweep for ``test-backend-ops --test-file``, and read the result.

``test-backend-ops perf`` has a fixed list of built-in shapes. Its ``--test-file``
option reads operations from a plain text file instead, thus an arbitrary M, K, N
sweep needs no rebuild of the binary.

One line per operation, whitespace separated:

    op type ne0 ne1 ne2 ne3 n_params [params...] n_src (type ne0..3 nb0..3)*N name

For a matrix multiply, ``ggml_mul_mat(a, b)`` takes ``a`` of shape [k, m] and
``b`` of shape [k, n] and gives [m, n]. Thus k is the reduction, m the output
rows and n the batch, which is the number of tokens in flight.

Usage:
    tools/prof/gemm.py gen --shape 2560x9216 --type q8_0 > sweep.txt
    tools/prof/gemm.py gen --model 4b --type q8_0 --out sweep.txt
    tools/prof/gemm.py parse run.log
"""

from __future__ import annotations

import argparse
import re
import sys

GGML_OP_MUL_MAT = 29

# name -> (enum value, block size, bytes per block)
TYPES: dict[str, tuple[int, int, int]] = {
    "f32": (0, 1, 4),
    "f16": (1, 1, 2),
    "q4_0": (2, 32, 18),
    "q8_0": (8, 32, 34),
}

# The matrices that dominate one decode step, as (k, m, label). k is the
# reduction and m the output rows.
MODELS: dict[str, list[tuple[int, int, str]]] = {
    "4b": [
        (2560, 9216, "ffn_gate_up"),
        (9216, 2560, "ffn_down"),
        (2560, 2560, "attn_gdn_proj"),
        (2560, 248320, "output_head"),
    ],
    "2b": [
        (2048, 6144, "ffn_gate_up"),
        (6144, 2048, "ffn_down"),
        (2048, 2048, "attn_gdn_proj"),
        (2048, 248320, "output_head"),
    ],
    # The built-in shape, kept so a run can be checked against the numbers the
    # binary produces on its own.
    "builtin": [(14336, 4096, "builtin_4096x14336")],
}

# Row counts. The dense low end is the speculative verify width, where one
# extra row may be free. The high end reaches the compute-bound regime.
ROWS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128, 256, 512]


def strides(ne: list[int], blck: int, tsize: int) -> list[int]:
    """The byte strides of a contiguous tensor.

    Args:
        ne: The four element counts
        blck: The elements per quantisation block, 1 for a plain type
        tsize: The bytes per block

    Returns:
        The four byte strides
    """
    nb0 = tsize
    nb1 = nb0 * (ne[0] // blck)
    nb2 = nb1 * ne[1]
    nb3 = nb2 * ne[2]
    return [nb0, nb1, nb2, nb3]


def case(k: int, m: int, n: int, type_a: str) -> str:
    """One test-file line for a matrix multiply.

    Args:
        k: The reduction dimension
        m: The output rows, the weight matrix has k by m
        n: The batch, the number of tokens in flight
        type_a: The weight type, a key of TYPES

    Returns:
        The line, without a trailing newline

    Raises:
        KeyError: If type_a is unknown
        ValueError: If k is not a multiple of the block size
    """
    ta, blck, tsize = TYPES[type_a]
    if k % blck:
        raise ValueError(f"k={k} is not a multiple of the block size {blck} of {type_a}")
    tb, bblck, bsize = TYPES["f32"]

    a_ne = [k, m, 1, 1]
    b_ne = [k, n, 1, 1]
    d_ne = [m, n, 1, 1]
    a_nb = strides(a_ne, blck, tsize)
    b_nb = strides(b_ne, bblck, bsize)

    parts: list[str] = [str(GGML_OP_MUL_MAT), "0"]          # op, dst type f32
    parts += [str(x) for x in d_ne]
    parts.append("0")                                        # no op params
    parts.append("2")                                        # two sources
    parts += [str(ta)] + [str(x) for x in a_ne] + [str(x) for x in a_nb]
    parts += [str(tb)] + [str(x) for x in b_ne] + [str(x) for x in b_nb]
    parts.append("-")                                        # no name
    return " ".join(parts)


def cmd_gen(a: argparse.Namespace) -> int:
    """Write the sweep file.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    if a.shape:
        k, m = (int(x) for x in a.shape.split("x"))
        shapes = [(k, m, a.shape)]
    else:
        shapes = MODELS[a.model]
    rows = [int(x) for x in a.rows.split(",")] if a.rows else ROWS

    lines = [case(k, m, n, a.type) for k, m, _ in shapes for n in rows]
    out = "\n".join(lines) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
        print(f"{len(lines)} cases -> {a.out}", file=sys.stderr)
        for k, m, label in shapes:
            print(f"  {label:<18} k={k:<6} m={m:<7} n in {rows[0]}..{rows[-1]}", file=sys.stderr)
    else:
        sys.stdout.write(out)
    return 0


# The test-file printer names the sources rather than m, n, k:
# MUL_MAT(type=f32,ne=[4096,1,1,1],...,sources=q8_0[14336,4096,1,1],f32[14336,1,1,1]):
#   1102 runs - 1125.45 us/run - 61000 kB/run - 51.69 GB/s
SHAPE = re.compile(
    r"MUL_MAT\(type=\w+,ne=\[(?P<m>\d+),(?P<n>\d+),.*?"
    r"sources=(?P<ta>\w+)\[(?P<k>\d+),\d+")
TIMED = re.compile(r"(?P<runs>\d+) runs -\s*(?P<us>[\d.]+) us/run")


def cmd_parse(a: argparse.Namespace) -> int:
    """Turn a run log into a table of rate against row count.

    Two derived columns matter more than the raw time. ``us/row`` shows whether
    an extra row is free, which decides the width of a speculative verify.
    ``GB/s`` shows whether the shape sits on the memory roofline.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    rows: list[tuple] = []
    for line in open(a.log, errors="replace"):
        sh = SHAPE.search(line)
        if not sh:
            continue
        k, mm, n = int(sh["k"]), int(sh["m"]), int(sh["n"])
        t = TIMED.search(line)
        if not t:
            rows.append((sh["ta"], k, mm, n, None, None, None))
            continue
        us = float(t["us"])
        _, blck, tsize = TYPES.get(sh["ta"], (0, 1, 4))
        wbytes = k * mm // blck * tsize
        gflops = 2 * k * mm * n / us / 1e3
        rows.append((sh["ta"], k, mm, n, us, gflops, wbytes / us / 1e3))

    if not rows:
        print(f"no results in {a.log}", file=sys.stderr)
        return 1

    print(f"{'type':>6} {'k':>6} {'m':>7} {'n':>5} {'us':>10} {'us/row':>9}"
          f" {'GFLOPS':>9} {'GB/s':>7} {'vs n=1':>7}")
    base: dict[tuple, float] = {}
    for ta, k, mm, n, us, gflops, gbs in rows:
        key = (ta, k, mm)
        if us is None:
            print(f"{ta:>6} {k:6d} {mm:7d} {n:5d} {'NOT SUPPORTED':>10}")
            continue
        if n == 1:
            base[key] = us
        rel = us / base[key] if key in base else float("nan")
        print(f"{ta:>6} {k:6d} {mm:7d} {n:5d} {us:10.1f} {us/n:9.1f}"
              f" {gflops:9.1f} {gbs:7.1f} {rel:7.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="write a sweep file")
    g.add_argument("--model", default="4b", choices=sorted(MODELS))
    g.add_argument("--shape", default=None, help="one shape as KxM, overrides --model")
    g.add_argument("--type", default="q8_0", choices=sorted(TYPES))
    g.add_argument("--rows", default=None, help="comma-separated row counts")
    g.add_argument("--out", default=None)
    g.set_defaults(fn=cmd_gen)

    p = sub.add_parser("parse", help="turn a run log into a table")
    p.add_argument("log")
    p.set_defaults(fn=cmd_parse)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
