#!/usr/bin/env python3
"""Print the NaN and Inf positions of the f32 output of one case in ops_replay result files.

Usage: nanrows.py INDEX NE0 FILE [FILE...]
The output is read as rows of NE0 floats. For each file the script prints the rows with a
non-finite value, the count in each row, and the columns when a row is not fully non-finite.
Record layout: tests/fuzz/ops/src/wire.cpp (read_results, decode_result).
"""
import math
import struct
import sys
from pathlib import Path


def records(data: bytes):
    """Yield (pack index, list of output byte strings) for each record. O(file size)."""
    off = 0
    while off + 8 <= len(data):
        (n,) = struct.unpack_from("<Q", data, off)
        rec = data[off + 8:off + 8 + n]
        off += 8 + n
        i = 0
        _magic, idx = struct.unpack_from("<II", rec, i)
        i += 8
        (tl,) = struct.unpack_from("<I", rec, i)
        i += 4 + tl
        _status, _flags = struct.unpack_from("<II", rec, i)
        i += 8
        (dl,) = struct.unpack_from("<I", rec, i)
        i += 4 + dl + 8 + 8  # detail, ms, input_hash
        (nout,) = struct.unpack_from("<I", rec, i)
        i += 4
        outs = []
        for _ in range(nout):
            (bl,) = struct.unpack_from("<Q", rec, i)
            i += 8
            outs.append(rec[i:i + bl])
            i += bl
        yield idx, outs


def main(argv: list[str]) -> int:
    """Print the non-finite rows of case argv[1] in each file."""
    want, ne0 = int(argv[1]), int(argv[2])
    for path in argv[3:]:
        for idx, outs in records(Path(path).read_bytes()):
            if idx != want or not outs:
                continue
            vals = struct.unpack(f"<{len(outs[0]) // 4}f", outs[0])
            rows = {}
            for k, v in enumerate(vals):
                if not math.isfinite(v):
                    rows.setdefault(k // ne0, []).append(k % ne0)
            total = sum(len(c) for c in rows.values())
            desc = ", ".join(f"{r}:{len(c)}" + ("" if len(c) == ne0 else f"{c}") for r, c in sorted(rows.items()))
            print(f"{Path(path).name}: {total} non-finite in {len(rows)} rows: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
