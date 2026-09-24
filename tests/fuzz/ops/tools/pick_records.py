#!/usr/bin/env python3
"""Copy the records of some pack indices from an ops_replay result file.

Usage: pick_records.py IN OUT INDEX [INDEX...]
A record is a u64 length, then the payload: u32 magic, u32 pack index, ... (tests/fuzz/ops/src/wire.cpp).
"""
import struct
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    """Write the records of the given indices of argv[1] to argv[2]. O(file size)."""
    src, dst, want = Path(argv[1]), Path(argv[2]), {int(x) for x in argv[3:]}
    data = src.read_bytes()
    out = bytearray()
    off = 0
    n_kept = 0
    while off + 8 <= len(data):
        (n,) = struct.unpack_from("<Q", data, off)
        if off + 8 + n > len(data):
            break
        _magic, idx = struct.unpack_from("<II", data, off + 8)
        if idx in want:
            out += data[off:off + 8 + n]
            n_kept += 1
        off += 8 + n
    dst.write_bytes(bytes(out))
    print(f"{src.name}: {n_kept} records of {sorted(want)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
