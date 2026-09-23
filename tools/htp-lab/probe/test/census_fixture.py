#!/usr/bin/env python3
"""The census fixture of the host test: a mock op table, a corpus, and the check of the outputs.

The host test (host-test.sh) runs isaprobe against test/mock_rpc.c instead of a DSP. This script
writes the files that the mock and the program read, and checks the files that the program writes:

    make <dir> --ops N --streams S
        <dir>/table.txt            The mock op table: the line "# streams S", then one op on each line:
                                   <name> <available> <s0> <b0> <s1> <b1> <s2> <b2> <out_bytes> <n_vectors>
                                   <out_type> <in_type0> <in_type1> <in_type2>
                                   The types are enum isa_type values. Op ORACLE_OP has the name and the
                                   types of an op of host/oracle.c, thus the host writes its oracle.
        <dir>/corpus/corpus_s<k>.bin   One corpus stream for each stream id k in [1, S), with the
                                   header of isa_kernels.h and CORPUS_VECTORS vectors
    check <dir> <out_dir> --arch A [--only TEXT] [--skip ID...]
        Compares each out_<name>.bin with the output that mock_rpc.c computes, and each header
        field with the format of isa_kernels.h. The ops in --skip must have no output file.

The mock op (the same rule as mock_rpc.c): for iteration i, the first 128 output bytes are
in0[i] ^ in1[i] ^ in2[i] ^ (op_id & 0xff), byte by byte, with a missing input as zeros. A 256-byte
output has the complement of those bytes in its second half.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

VEC = 128
CORPUS_VECTORS = 64
HEADER = struct.Struct("<4s7I64s32s")
ORACLE_OP = 7
ORACLE_NAME = "cc.Q6_Vsf_vadd_VsfVsf"
T_SF = 1  # enum isa_type
MAGIC_CORPUS = b"HVXC"
MAGIC_OUTPUT = b"HVXO"
FILE_VERSION = 1


def isa_hash(data: bytes) -> int:
    """Return isa_hash() of isa_kernels.h: FNV-1a over 32 lanes of 32-bit words, then over the lanes.

    The length must be a multiple of 128. O(len(data)).
    """
    if len(data) % VEC != 0:
        raise ValueError(f"isa_hash needs a multiple of {VEC} bytes, not {len(data)}")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    lanes = [2166136261] * 32
    for i in range(0, len(words), 32):
        for k in range(32):
            lanes[k] = ((lanes[k] ^ words[i + k]) * 16777619) & 0xFFFFFFFF
    r = 2166136261
    for h in lanes:
        r = ((r ^ h) * 16777619) & 0xFFFFFFFF
    return r


def op_row(k: int, n_streams: int) -> tuple[str, int, list[tuple[int, int]], int, int]:
    """Return the mock op k: name, available, three (stream, bytes) inputs, out_bytes, n_vectors.

    The rows cover one, two and three inputs, 256-byte inputs and outputs, and unavailable ops.
    Op ORACLE_OP has two 128-byte inputs and a 128-byte output, as its oracle needs.
    """
    n_inputs = 1 + k % 3
    in_bytes = 256 if k % 11 == 4 else VEC
    inputs = [(1 + (k + 5 * j) % (n_streams - 1), in_bytes if j == 0 else VEC) if j < n_inputs else (0, 0)
              for j in range(3)]
    out_bytes = 256 if k % 7 == 3 else VEC
    n_vectors = 8 if k % 2 else 16
    available = 0 if k % 10 == 9 else 1
    name = ORACLE_NAME if k == ORACLE_OP else f"mock.op{k}"
    return name, available, inputs, out_bytes, n_vectors


def op_types(k: int) -> list[int]:
    """Return the out_type and the three in_types of mock op k: 0 (none) except for ORACLE_OP."""
    return [T_SF, T_SF, T_SF, 0] if k == ORACLE_OP else [0, 0, 0, 0]


def stream_bytes(k: int) -> bytes:
    """Return the vectors of the mock stream k: a byte pattern that differs between the streams."""
    return bytes(((i * 37) ^ (k * 101) ^ (i >> 7)) & 0xFF for i in range(CORPUS_VECTORS * VEC))


def header(magic: bytes, ident: int, n_vectors: int, bpv: int, arch: int, data: bytes, source: int, name: str) -> bytes:
    """Return the 128-byte file header of isa_kernels.h for the given fields and data."""
    return HEADER.pack(magic, FILE_VERSION, ident, n_vectors, bpv, arch, isa_hash(data), source,
                       name.encode().ljust(64, b"\0"), bytes(32))


def make(root: Path, n_ops: int, n_streams: int) -> None:
    """Write the mock op table and the corpus streams into root. O(n_ops + n_streams)."""
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    lines = [f"# streams {n_streams}"]
    for k in range(n_ops):
        name, available, inputs, out_bytes, n_vectors = op_row(k, n_streams)
        fields = [name, str(available)] + [str(v) for pair in inputs for v in pair] + [str(out_bytes), str(n_vectors)]
        fields += [str(t) for t in op_types(k)]
        lines.append(" ".join(fields))
    (root / "table.txt").write_text("\n".join(lines) + "\n")
    for k in range(1, n_streams):
        data = stream_bytes(k)
        hdr = header(MAGIC_CORPUS, k, CORPUS_VECTORS, VEC, 0, data, 0, f"s{k}")
        (corpus / f"corpus_s{k}.bin").write_bytes(hdr + data)


def expected_output(k: int, inputs: list[tuple[int, int]], out_bytes: int, n_vectors: int) -> bytes:
    """Return the output of mock op k, by the rule of mock_rpc.c. O(n_vectors * out_bytes)."""
    streams = [stream_bytes(s) if s else b"" for s, _ in inputs]
    out = bytearray()
    for i in range(n_vectors):
        block = bytearray(VEC)
        for j, (s, b) in enumerate(inputs):
            if s:
                src = streams[j][i * b:i * b + VEC]
                block = bytearray(x ^ y for x, y in zip(block, src))
        block = bytearray(x ^ (k & 0xFF) for x in block)
        out += block
        if out_bytes == 256:
            out += bytes(x ^ 0xFF for x in block)
    return bytes(out)


def check(root: Path, out_dir: Path, arch: int, only: str, skip: set[int], want_oracle: bool) -> int:
    """Check the output files of isaprobe against the mock ops. Returns 0 if all agree, else 1.

    want_oracle tells if the oracle file of ORACLE_OP must exist (with its header) or be absent.
    """
    errors = []
    lines = (root / "table.txt").read_text().splitlines()
    n_streams = int(lines[0].split()[2])
    for k in range(len(lines) - 1):
        name, available, inputs, out_bytes, n_vectors = op_row(k, n_streams)
        path = out_dir / f"out_{name}.bin"
        want_file = available and only in name and k not in skip
        if not want_file:
            if path.exists():
                errors.append(f"{path.name} exists, and the op must have no output")
            continue
        if not path.exists():
            errors.append(f"{path.name} is missing")
            continue
        blob = path.read_bytes()
        data = expected_output(k, inputs, out_bytes, n_vectors)
        want = header(MAGIC_OUTPUT, k, n_vectors, out_bytes, arch, data, 1, name)
        if blob[:HEADER.size] != want:
            errors.append(f"{path.name}: header {HEADER.unpack(blob[:HEADER.size])} is not {HEADER.unpack(want)}")
        if blob[HEADER.size:] != data:
            errors.append(f"{path.name}: the output bytes differ from the mock rule")
    oracle = out_dir / f"out_{ORACLE_NAME}.oracle.bin"
    if want_oracle:
        _, available, _, out_bytes, n_vectors = op_row(ORACLE_OP, n_streams)
        if not oracle.exists():
            errors.append(f"{oracle.name} is missing")
        else:
            blob = oracle.read_bytes()
            f = HEADER.unpack(blob[:HEADER.size])
            want = (b"HVXO", FILE_VERSION, ORACLE_OP, n_vectors, out_bytes, 0, isa_hash(blob[HEADER.size:]), 2,
                    ORACLE_NAME.encode().ljust(64, b"\0"), bytes(32))
            if f != want or len(blob) != HEADER.size + n_vectors * out_bytes:
                errors.append(f"{oracle.name}: header {f} is not {want}")
    elif oracle.exists():
        errors.append(f"{oracle.name} exists, and the run had no oracle")
    for e in errors[:10]:
        print("census_fixture:", e)
    print(f"census_fixture: {len(errors)} errors")
    return 1 if errors else 0


def main(argv: list[str]) -> int:
    """Parse the command line and run "make" or "check". Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_make = sub.add_parser("make")
    p_make.add_argument("root", type=Path)
    p_make.add_argument("--ops", type=int, required=True)
    p_make.add_argument("--streams", type=int, required=True)
    p_check = sub.add_parser("check")
    p_check.add_argument("root", type=Path)
    p_check.add_argument("out", type=Path)
    p_check.add_argument("--arch", type=int, required=True)
    p_check.add_argument("--only", default="")
    p_check.add_argument("--skip", type=int, nargs="*", default=[])
    p_check.add_argument("--no-oracle", action="store_true", help="the oracle file of ORACLE_OP must be absent")
    args = parser.parse_args(argv)
    if args.command == "make":
        make(args.root, args.ops, args.streams)
        return 0
    want_oracle = not args.no_oracle and args.only in ORACLE_NAME
    return check(args.root, args.out, args.arch, args.only, set(args.skip), want_oracle)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
