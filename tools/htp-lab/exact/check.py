#!/usr/bin/env python3
"""Check the outputs of the lab target "exact" against an exact reference, and across versions.

Usage:
    tools/htp-lab/exact/check.py <run dir> [<run dir> ...]

Each run directory holds the files that lab/target_exact.c writes. For each routine the script
counts the lanes that differ from the reference and prints the first ones. With more than one
directory it also compares the output files byte for byte, thus it shows if two Hexagon versions
give the same bits.

The reference follows the definition in lab/hvx-exact.h: round to nearest even, an f32 subnormal
input is zero, an f32 result with a biased exponent of 0 or less before the rounding is a zero with
the sign of the exact result, a NaN result is any NaN, an exact cancellation gives +0, and two zero
inputs give -0 only when both are negative.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F32_NAN = 0x7FC00000
F32_INF = 0x7F800000


def load_u32(path: Path) -> np.ndarray:
    """Read a file of little-endian 32-bit words.

    Args:
        path: The file

    Returns:
        The words as a uint32 array
    """
    return np.fromfile(path, dtype="<u4")


def is_nan32(x: np.ndarray) -> np.ndarray:
    """Tell which f32 bit patterns are NaN.

    Args:
        x: The bit patterns

    Returns:
        A bool array
    """
    return (x & 0x7FFFFFFF) > F32_INF


def ref_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The reference f32 product of two arrays of bit patterns. O(n).

    Args:
        a: The first operands
        b: The second operands

    Returns:
        The result bit patterns
    """
    a = a.astype(np.uint64)
    b = b.astype(np.uint64)
    sign = (a ^ b) & 0x80000000
    ea = (a >> 23) & 0xFF
    eb = (b >> 23) & 0xFF
    ma = (a & 0x7FFFFF) | 0x800000
    mb = (b & 0x7FFFFF) | 0x800000
    p = ma * mb
    t = (p >> 47) & 1
    shift = 23 + t
    m = p >> shift
    rest = p & ((np.uint64(1) << shift) - 1)
    half = np.uint64(1) << (shift - 1)
    up = (rest > half) | ((rest == half) & ((m & 1) == 1))
    m = m + up.astype(np.uint64)
    e = ea.astype(np.int64) + eb.astype(np.int64) - 127 + t.astype(np.int64)
    carry = m == (1 << 24)
    m = np.where(carry, np.uint64(1 << 23), m)
    e_after = e + carry.astype(np.int64)
    r = (np.clip(e_after, 0, 255).astype(np.uint64) << 23) | (m - (1 << 23))
    r = np.where(e <= 0, np.uint64(0), r)
    r = np.where(e_after >= 255, np.uint64(F32_INF), r)
    zero_in = (ea == 0) | (eb == 0)
    r = np.where(zero_in, np.uint64(0), r)
    special = (ea == 255) | (eb == 255)
    nan = zero_in | is_nan32(a.astype(np.uint32)) | is_nan32(b.astype(np.uint32))
    r = np.where(special, np.where(nan, np.uint64(F32_NAN), np.uint64(F32_INF)), r)
    r = r | np.where(special & nan, np.uint64(0), sign)
    return r.astype(np.uint32)


def ref_add_one(a: int, b: int) -> int:
    """The reference f32 sum of two bit patterns, with exact Python integers.

    Args:
        a: The first operand
        b: The second operand

    Returns:
        The result bit pattern
    """
    aa, ab = a & 0x7FFFFFFF, b & 0x7FFFFFFF
    big, small = (b, a) if ab > aa else (a, b)
    if (big & 0x7FFFFFFF) > F32_INF:
        return F32_NAN
    if (big & 0x7FFFFFFF) == F32_INF:
        if (small & 0x7FFFFFFF) == F32_INF and (big ^ small) & 0x80000000:
            return F32_NAN
        return big

    def value(x: int) -> tuple[int, int]:
        """The signed significand and the exponent of an f32 pattern, with zero for a subnormal."""
        e = (x >> 23) & 0xFF
        if e == 0:
            return 0, 1
        m = (x & 0x7FFFFF) | 0x800000
        return (-m if x & 0x80000000 else m), e

    if (a >> 23) & 0xFF == 0 and (b >> 23) & 0xFF == 0:
        return a & b & 0x80000000
    ma, ea = value(a)
    mb, eb = value(b)
    emin = min(ea, eb)
    s = (ma << (ea - emin)) + (mb << (eb - emin))
    if s == 0:
        return 0
    sign = 0x80000000 if s < 0 else 0
    s = abs(s)
    n = s.bit_length()
    e = n + emin - 24
    if n > 24:
        shift = n - 24
        m = s >> shift
        rest = s & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rest > half or (rest == half and m & 1):
            m += 1
    else:
        m = s << (24 - n)
    if e <= 0:
        return sign
    if m == 1 << 24:
        m = 1 << 23
        e += 1
    if e >= 255:
        return sign | F32_INF
    return sign | (e << 23) | (m - (1 << 23))


def ref_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The reference f32 sums of two arrays of bit patterns. O(n) Python operations.

    Args:
        a: The first operands
        b: The second operands

    Returns:
        The result bit patterns
    """
    return np.array([ref_add_one(int(x), int(y)) for x, y in zip(a, b)], dtype=np.uint32)


def compare(name: str, got: np.ndarray, ref: np.ndarray, nan_class: np.ndarray, inputs: list[np.ndarray]) -> int:
    """Count and print the lanes where the result differs from the reference.

    Args:
        name: The routine name
        got: The results of the DSP
        ref: The reference results
        nan_class: True where both are NaN, which counts as equal
        inputs: The input arrays, for the printout of the first differences

    Returns:
        The number of differing lanes
    """
    bad = np.nonzero((got != ref) & ~nan_class)[0]
    print(f"  {name}: {len(bad)} of {len(got)} lanes differ from the reference")
    for i in bad[:6]:
        ins = " ".join(f"{int(x[i]):08x}" for x in inputs)
        print(f"    lane {i}: in {ins} got {int(got[i]):08x} ref {int(ref[i]):08x}")
    return len(bad)


def check_dir(d: Path) -> int:
    """Check all the routines of one run directory.

    Args:
        d: The run directory

    Returns:
        The total number of differing lanes
    """
    print(f"{d}:")
    total = 0

    a, b, got = load_u32(d / "mul_a.bin"), load_u32(d / "mul_b.bin"), load_u32(d / "mul_out.bin")
    ref = ref_mul(a, b)
    total += compare("sf_mul", got, ref, is_nan32(got) & is_nan32(ref), [a, b])

    a, b, got = load_u32(d / "add_a.bin"), load_u32(d / "add_b.bin"), load_u32(d / "add_out.bin")
    ref = ref_add(a, b)
    total += compare("sf_add", got, ref, is_nan32(got) & is_nan32(ref), [a, b])

    got = load_u32(d / "hf2sf_out.bin")
    h = np.arange(65536, dtype=np.uint16)
    ref = h.view(np.float16).astype(np.float32).view(np.uint32)
    total += compare("hf_to_sf", got, ref, is_nan32(got) & is_nan32(ref), [h.astype(np.uint32)])

    x, got = load_u32(d / "sf2hf_in.bin"), load_u32(d / "sf2hf_out.bin")
    with np.errstate(over="ignore", invalid="ignore"):
        ref = x.view(np.float32).astype(np.float16).view(np.uint16).astype(np.uint32)
    nan16 = ((got & 0x7FFF) > 0x7C00) & ((ref & 0x7FFF) > 0x7C00)
    total += compare("sf_to_hf", got, ref, nan16, [x])
    return total


def main() -> int:
    """Check each run directory, then compare the output files across the directories.

    Returns:
        The exit code: 0 when every lane agrees with the reference and every directory agrees
    """
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 2
    bad = sum(check_dir(d) for d in dirs)
    if len(dirs) > 1:
        print("cross-version:")
        for f in ("mul_out.bin", "add_out.bin", "hf2sf_out.bin", "sf2hf_out.bin"):
            blobs = [(d / f).read_bytes() for d in dirs]
            same = all(x == blobs[0] for x in blobs[1:])
            print(f"  {f}: {'identical' if same else 'DIFFERENT'} in {len(dirs)} directories")
            bad += 0 if same else 1
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
