#!/usr/bin/env python3
"""Check the outputs of the lab target "q8oracle" against quantize_row_q8_0_ref, and across versions.

Usage:
    tools/htp-lab/exact/check_q8.py <run dir> [<run dir> ...]

The reference is quantize_row_q8_0_ref (ggml-quants.c:276) in numpy f32, whose division and product
round to nearest even as the x86 CPU does: d = amax / 127, id = 1 / d, q = roundf(x * id) with a tie
away from zero, and y.d = fp16(d) with round to nearest even. The limits of htp/hvx-q8-ref.h change
the reference only for a block with amax below 2^-117: a subnormal d becomes 0, and a subnormal x
gives q = 0. The script applies them, and it also counts the blocks where the plain reference differs.

For a scales run (am.bin, d.bin, id.bin) it compares d and id with the f32 division.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TILE = 1152
F32_MIN_NORMAL = np.float32(2.0**-126)


def roundf(p: np.ndarray) -> np.ndarray:
    """roundf of f32 values: a tie rounds away from zero. The f64 sum |p| + 0.5 is exact. O(n).

    Args:
        p: The f32 values

    Returns:
        The rounded values as int32
    """
    a = np.floor(np.abs(p.astype(np.float64)) + 0.5)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.sign(p) * a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32)


def ref_blocks(x: np.ndarray, limits: bool) -> tuple[np.ndarray, np.ndarray]:
    """Quantize blocks of 32 values as quantize_row_q8_0_ref does. O(n).

    Args:
        x: The f32 values, shape (n_blocks, 32)
        limits: Apply the limits of htp/hvx-q8-ref.h for a block with amax below 2^-117

    Returns:
        The quants (n_blocks, 32) as int8 and the f16 scale bits (n_blocks,) as uint16
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        amax = np.max(np.abs(x), axis=1)
        d = amax / np.float32(127)
        if limits:
            d = np.where(d < F32_MIN_NORMAL, np.float32(0), d)
            x = np.where(np.abs(x) < F32_MIN_NORMAL, np.float32(0), x)
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
        p = (x * idv[:, None]).astype(np.float32)
    q = roundf(p).astype(np.int8)
    dh = d.astype(np.float16).view(np.uint16)
    return q, dh


def expected_tiles(q: np.ndarray, dh: np.ndarray) -> np.ndarray:
    """The tiled layout: vector i (i < 8) holds quants 4i..4i+3 repeated, vector 8 the scale.

    Args:
        q: The quants (n_blocks, 32)
        dh: The f16 scale bits (n_blocks,)

    Returns:
        The tile bytes (n_blocks, 1152)
    """
    n = q.shape[0]
    t = np.zeros((n, TILE), dtype=np.uint8)
    qb = q.view(np.uint8)
    for i in range(8):
        t[:, 128 * i : 128 * (i + 1)] = np.tile(qb[:, 4 * i : 4 * i + 4], 32)
    t[:, 1024:1152] = np.tile(dh.view(np.uint8).reshape(n, 2), 64)
    return t


def check_rows(d: Path, k: int) -> int:
    """Check the four files of one row length.

    Args:
        d: The run directory
        k: The row length

    Returns:
        The number of blocks that differ
    """
    x = np.fromfile(d / f"x_{k}.bin", dtype="<f4").reshape(-1, 32)
    tiles = np.fromfile(d / f"tiles_{k}.bin", dtype=np.uint8).reshape(-1, TILE)
    comp = np.fromfile(d / f"compact_{k}.bin", dtype=np.int8).reshape(-1, 32)
    sums = np.fromfile(d / f"sums_{k}.bin", dtype="<i4")
    scales = np.fromfile(d / f"scales_{k}.bin", dtype="<u2")

    q, dh = ref_blocks(x, limits=True)
    q_plain, dh_plain = ref_blocks(x, limits=False)
    exp = expected_tiles(q, dh)

    bad_tile = np.nonzero(np.any(tiles != exp, axis=1))[0]
    bad_comp = np.nonzero(np.any(comp != q, axis=1))[0]
    bad_sum = np.nonzero(sums != q.astype(np.int32).sum(axis=1))[0]
    bad_scale = np.nonzero(scales != dh)[0]
    amax = np.max(np.abs(x), axis=1)
    plain_diff = np.any(q != q_plain, axis=1) | (dh != dh_plain)
    small = amax < np.float32(2.0**-117)
    print(
        f"  k {k}: {len(x)} blocks, tiles differ {len(bad_tile)}, compact differ {len(bad_comp)}, "
        f"sums differ {len(bad_sum)}, flat scales differ {len(bad_scale)}; limits change {int(plain_diff.sum())} blocks, "
        f"of which {int((plain_diff & ~small).sum())} have amax >= 2^-117"
    )
    for b in bad_tile[:4]:
        got_q = tiles[b, [128 * (j // 4) + j % 4 for j in range(32)]].view(np.int8)
        got_d = tiles[b, 1024:1026].view(np.uint16)[0]
        diff = np.nonzero(got_q != q[b])[0]
        print(f"    block {b}: amax {amax[b]:.9g} scale got {got_d:04x} ref {dh[b]:04x}; quants differ at {diff[:8].tolist()}")
        for j in diff[:3]:
            print(f"      x {x[b, j]!r} ({x[b, j].view(np.uint32):08x}) got {got_q[j]} ref {q[b, j]}")
    return len(bad_tile) + len(bad_comp) + len(bad_sum) + len(bad_scale) + int((plain_diff & ~small).sum())


def check_scales(d: Path) -> int:
    """Check d = amax / 127 and id = 1 / d against the f32 division.

    Args:
        d: The run directory

    Returns:
        The number of lanes that differ
    """
    am = np.fromfile(d / "am.bin", dtype="<u4")
    got_d = np.fromfile(d / "d.bin", dtype="<u4")
    got_id = np.fromfile(d / "id.bin", dtype="<u4")
    a = am.view(np.float32)
    with np.errstate(divide="ignore", over="ignore"):
        ref_d = a / np.float32(127)
        ref_d = np.where(ref_d < F32_MIN_NORMAL, np.float32(0), ref_d).astype(np.float32)
        ref_id = np.where(ref_d != 0, np.float32(1) / ref_d, np.float32(0)).astype(np.float32)
    bd = np.nonzero(got_d != ref_d.view(np.uint32))[0]
    bi = np.nonzero(got_id != ref_id.view(np.uint32))[0]
    print(f"  scales: {len(am)} lanes, d differs {len(bd)}, id differs {len(bi)}")
    for i in list(bd[:4]) + list(bi[:4]):
        print(f"    am {am[i]:08x} d got {got_d[i]:08x} ref {ref_d.view(np.uint32)[i]:08x} id got {got_id[i]:08x} ref {ref_id.view(np.uint32)[i]:08x}")
    return len(bd) + len(bi)


def main() -> int:
    """Check each run directory, then compare the output files across the directories.

    Returns:
        The exit code: 0 when every block agrees with the reference and every directory agrees
    """
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    bad = 0
    names: list[str] = []
    for d in dirs:
        print(f"{d}:")
        if (d / "am.bin").exists():
            bad += check_scales(d)
            names = ["d.bin", "id.bin"]
        else:
            for k in (4096, 2688, 128):
                bad += check_rows(d, k)
            names = [f"{p}_{k}.bin" for k in (4096, 2688, 128) for p in ("tiles", "compact", "sums", "scales")]
    if len(dirs) > 1:
        print("cross-version:")
        for f in names:
            blobs = [(d / f).read_bytes() for d in dirs]
            same = all(b == blobs[0] for b in blobs[1:])
            print(f"  {f}: {'identical' if same else 'DIFFERENT'} in {len(dirs)} directories")
            bad += 0 if same else 1
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
