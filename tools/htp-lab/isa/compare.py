#!/usr/bin/env python3
"""Compare the HVX instruction census outputs of the Hexagon versions.

For each op of isa_ops.csv the tool reads out_<op>.bin of each version and reports:
- the versions where the op exists, and the groups of versions with bit-identical output,
- for each pair of groups: the fraction of lanes that differ and the kind of difference. For an
  IEEE result (sf, hf, bf) the kind is one of: nan, inf, zero_sign, subnormal, tie, round (one
  unit in the last place, not a tie) and large (more than one unit). For a qf result the tool
  decodes the values and tells if the values agree when the bits differ,
- for each IEEE-result op with a known semantic: per version the fraction of lanes equal to the
  correctly rounded (round to nearest even) result and the maximum error in units in the last
  place (ulp). For a qf result: the maximum and mean error in ulp of the IEEE format of the same
  width (sf for qf32, hf for qf16),
- the verdict: SAME (bit-identical on each version), FINITE-SAME (the differences are only in lanes
  with a NaN or infinite input or output, or for qf a value outside the sf range) or DIFF,
- the instructions that the compiler emitted for the op on each version (objdump-<arch>.txt).

The reference arithmetic is exact: the inputs decode to float64 without error, the float64 sum or
product of two inputs is exact or its error cannot change the final rounding, and a vectorized
integer step rounds float64 to the target format. The three-term dot products (vdmpy) use exact
Python integers. The tool writes census.csv and exp2.txt to the output directory and prints a
compact table.

Usage: python3 tools/htp-lab/isa/compare.py [--tag c] [--archs v73 v75 v79 v81]
"""
from __future__ import annotations

import argparse
import csv
import re
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ISA_DIR = Path(__file__).resolve().parent
REPO_DIR = ISA_DIR.parents[2]
OUT_DIR = REPO_DIR / "tools/htp-lab/out-isa"

HEADER = struct.Struct("<4s7I64s32s")

# (stored mantissa bits, exponent bits, bias)
FMT = {"sf": (23, 8, 127), "hf": (10, 5, 15), "bf": (7, 8, 127)}
LANE_BYTES = {"sf": 4, "hf": 2, "bf": 2, "qf32": 4, "qf16": 2, "f8": 1, "x": 4, "b": 1, "ub": 1, "h": 2,
              "uh": 2, "w": 4, "uw": 4, "pred": 1, "": 4}
NP_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32}
QF_TARGET = {"qf32": "sf", "qf16": "hf"}


# ---------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------


def run_dir(arch: str, tag: str) -> Path:
    """Return the run directory of one version (run.sh names the v79 directory without the version)."""
    return OUT_DIR / (f"isa-{tag}" if arch == "v79" else f"isa-{arch}-{tag}")


def read_bin(path: Path) -> tuple[dict, np.ndarray] | None:
    """Read one census file. Returns (header fields, data as uint8), or None if it does not exist."""
    if not path.exists():
        return None
    raw = path.read_bytes()
    magic, version, ident, n, bpv, arch, h, source, name, _ = HEADER.unpack_from(raw, 0)
    hdr = dict(magic=magic.decode(), version=version, id=ident, n_vectors=n, bytes_per_vector=bpv, arch=arch,
               hash=h, source=source, name=name.rstrip(b"\0").decode())
    data = np.frombuffer(raw, dtype=np.uint8, offset=128, count=n * bpv)
    return hdr, data


def isa_hash(data: np.ndarray) -> int:
    """The hash of isa_kernels.h: FNV-1a in 32 lanes of 32-bit words, then FNV-1a over the lanes. O(n)."""
    w = data.view(np.uint32).reshape(-1, 32)
    h = np.full(32, 2166136261, dtype=np.uint32)
    prime = np.uint32(16777619)
    with np.errstate(over="ignore"):
        for row in w:
            h = (h ^ row) * prime
    r = 2166136261
    for k in range(32):
        r = ((r ^ int(h[k])) * 16777619) & 0xFFFFFFFF
    return r


# ---------------------------------------------------------------------------------------------
# Decoders: element bits to float64 values (exact)
# ---------------------------------------------------------------------------------------------


def ieee_to_f64(bits: np.ndarray, fmt: str) -> np.ndarray:
    """Decode IEEE bits of the format to float64 (exact, NaN and infinity kept)."""
    mb, eb, bias = FMT[fmt]
    b = bits.astype(np.int64)
    sign = (b >> (mb + eb)) & 1
    e = (b >> mb) & ((1 << eb) - 1)
    m = b & ((1 << mb) - 1)
    emax = (1 << eb) - 1
    sig = np.where(e == 0, m, m | (1 << mb)).astype(np.float64)
    ex = np.where(e == 0, 1 - bias - mb, e - bias - mb)
    val = np.ldexp(sig, ex.astype(np.int64))
    val = np.where(e == emax, np.where(m == 0, np.inf, np.nan), val)
    return np.where(sign == 1, -val, val)


def qf_to_f64(bits: np.ndarray, qf: str) -> np.ndarray:
    """Decode qf32 ((2M+1) * 2^(E-150), M = bits 31:8) or qf16 ((2M+1) * 2^(E-25), M = bits 15:5)."""
    b = bits.astype(np.int64)
    if qf == "qf32":
        m = b >> 8
        m = np.where(m >= (1 << 23), m - (1 << 24), m)
        e = b & 0xFF
        return np.ldexp((2 * m + 1).astype(np.float64), (e - 150).astype(np.int64))
    m = b >> 5
    m = np.where(m >= (1 << 10), m - (1 << 11), m)
    e = b & 0x1F
    return np.ldexp((2 * m + 1).astype(np.float64), (e - 25).astype(np.int64))


def int_to_f64(bits: np.ndarray, et: str) -> np.ndarray:
    """Decode integer lanes (b, ub, h, uh, w, uw) to float64."""
    signed = {"b": np.int8, "h": np.int16, "w": np.int32}
    if et in signed:
        return bits.view(signed[et]).astype(np.float64)
    return bits.astype(np.float64)


def decode(bits: np.ndarray, et: str) -> np.ndarray:
    """Decode lanes of any census element type to float64."""
    if et in FMT:
        return ieee_to_f64(bits, et)
    if et in ("qf32", "qf16"):
        return qf_to_f64(bits, et)
    return int_to_f64(bits, et)


def is_subnormal(bits: np.ndarray, et: str) -> np.ndarray:
    """True where the IEEE lane is subnormal. False for any other type."""
    if et not in FMT:
        return np.zeros(bits.shape, dtype=bool)
    mb, eb, _ = FMT[et]
    b = bits.astype(np.int64)
    return (((b >> mb) & ((1 << eb) - 1)) == 0) & ((b & ((1 << mb) - 1)) != 0)


# ---------------------------------------------------------------------------------------------
# Correct rounding of float64 to a format (vectorized, integer only)
# ---------------------------------------------------------------------------------------------


def round_f64(x: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round float64 values to the format with round to nearest even.

    Returns (bits, tie, sub): tie is True where the value was exactly halfway between two
    neighbors, sub is True where the value is less than the smallest normal in magnitude (and not
    zero). NaN gives the default quiet NaN, infinity and overflow give infinity. O(n).
    """
    mb, eb, bias = FMT[fmt]
    p = mb + 1
    emin, emax = 1 - bias, (1 << eb) - 2 - bias
    signbit = 1 << (mb + eb)
    u = x.view(np.uint64).astype(np.uint64)
    sign = ((u >> np.uint64(63)) & np.uint64(1)).astype(np.int64)
    ex = ((u >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    man = (u & np.uint64((1 << 52) - 1)).astype(np.int64)
    finite = ex != 0x7FF
    zero = (ex == 0) & (man == 0)
    # Every value of the census is a normal float64 or zero: the smallest is about 2^-310
    sig = man | (1 << 52)
    e = ex - 1023
    keep = np.where(e >= emin, p, p - (emin - e))
    drop = np.clip(53 - keep, 1, 62)
    r = sig >> drop
    rem = sig & ((np.int64(1) << drop) - 1)
    half = np.int64(1) << (drop - 1)
    up = (rem > half) | ((rem == half) & ((r & 1) == 1))
    tie = (rem == half) & finite & ~zero
    r = r + up.astype(np.int64)
    normal = e >= emin
    # renormalize a round up to 2^p
    carry = normal & (r >> p != 0)
    r = np.where(carry, r >> 1, r)
    e2 = np.where(carry, e + 1, e)
    out_normal = (sign * signbit) | ((e2 + bias) << mb) | (r & ((1 << mb) - 1))
    out_normal = np.where(e2 > emax, sign * signbit | (((1 << eb) - 1) << mb), out_normal)
    out_sub = (sign * signbit) | r  # a round up to 2^mb gives the smallest normal by itself
    bits = np.where(normal, out_normal, out_sub)
    bits = np.where(zero, sign * signbit, bits)
    inf_bits = sign * signbit | (((1 << eb) - 1) << mb)
    nan_bits = (((1 << eb) - 1) << mb) | (1 << (mb - 1))
    bits = np.where(finite, bits, np.where(np.isnan(x), nan_bits, inf_bits))
    sub = finite & ~zero & (e < emin)
    return bits.astype(np.uint32), tie, sub


def ieee_class(bits: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (is_nan, is_inf, is_zero) of IEEE lanes."""
    mb, eb, _ = FMT[fmt]
    b = bits.astype(np.int64)
    e = (b >> mb) & ((1 << eb) - 1)
    m = b & ((1 << mb) - 1)
    emax = (1 << eb) - 1
    return (e == emax) & (m != 0), (e == emax) & (m == 0), (e == 0) & (m == 0)


def ordered(bits: np.ndarray, fmt: str) -> np.ndarray:
    """Map IEEE bits to integers with the order of the values (+0 and -0 map to 0)."""
    mb, eb, _ = FMT[fmt]
    signbit = 1 << (mb + eb)
    b = bits.astype(np.int64)
    mag = b & (signbit - 1)
    return np.where(b & signbit, -mag, mag)


def classify_ieee(got: np.ndarray, ref: np.ndarray, fmt: str, tie: np.ndarray, sub: np.ndarray) -> dict[str, int]:
    """Count the kinds of difference between two IEEE lane arrays (only the lanes that differ).

    ``tie`` and ``sub`` flag the lanes where the exact value is a tie, or where the exact value or
    an input is subnormal. Two NaNs with different payloads count as "nan_payload".
    """
    gn, gi, gz = ieee_class(got, fmt)
    rn, ri, rz = ieee_class(ref, fmt)
    diff = got != ref
    kinds: dict[str, int] = {}

    def add(name: str, mask: np.ndarray) -> np.ndarray:
        """Count the lanes of the mask under the kind ``name`` and return the mask."""
        n = int(mask.sum())
        if n:
            kinds[name] = kinds.get(name, 0) + n
        return mask

    left = diff.copy()
    left &= ~add("nan_payload", left & gn & rn)
    left &= ~add("nan_vs_inf", left & ((gn & ri) | (gi & rn)))
    left &= ~add("nan", left & (gn | rn))
    left &= ~add("inf", left & (gi | ri))
    left &= ~add("zero_sign", left & gz & rz)
    d = np.abs(ordered(got, fmt) - ordered(ref, fmt))
    left &= ~add("subnormal", left & sub)
    left &= ~add("tie", left & tie & (d == 1))
    left &= ~add("round", left & (d == 1))
    add("large", left)
    return kinds


# ---------------------------------------------------------------------------------------------
# Op semantics
# ---------------------------------------------------------------------------------------------


@dataclass
class Sem:
    """The semantic of an op: the operation, the result format and the candidate lane layouts."""

    op: str
    out: str
    layouts: list[str]
    domain: str = ""  # "lt2p22": only the lanes with |value| < 2^22 count (the magic-number conversions)


def sem_of(name: str, source: str) -> Sem | None:
    """Return the semantic of an op from its name and source, or None if the tool has no model."""
    s = source
    if name.startswith("seq."):
        n = name[4:]
        table = {
            "hvx_vec_add_f32_f32": Sem("add", "sf", ["elem"]), "hvx_vec_sub_f32_f32": Sem("sub", "sf", ["elem"]),
            "hvx_vec_mul_f32_f32": Sem("mul", "sf", ["elem"]), "hvx_vec_add_f16_f16": Sem("add", "hf", ["elem"]),
            "hvx_vec_sub_f16_f16": Sem("sub", "hf", ["elem"]), "hvx_vec_mul_f16_f16": Sem("mul", "hf", ["elem"]),
            "hvx_vec_f32_to_f16": Sem("cvt", "hf", ["narrow_contig"]),
            "hvx_vec_f32_to_f16_shuff": Sem("cvt", "hf", ["narrow_uv", "narrow_vu"]),
            "hvx_vec_f16_to_f32": Sem("cvt", "sf", ["widen_contig"]),
            "hvx_vec_f16_to_f32_shuff": Sem("cvt", "sf", ["widen_eo", "widen_contig"]),
            "hvx_vec_mpyacc_f32_f16": Sem("fma", "sf", ["widen_eo"]),
            "hvx_vec_i16_from_hf_rnd_sat": Sem("toint", "h", ["elem"]),
            "qfpair.mul_f32": Sem("mul", "sf", ["elem"]), "qfpair.add_f32": Sem("add", "sf", ["elem"]),
            "qfpair.sub_f32": Sem("sub", "sf", ["elem"]), "qfpair.sf_roundtrip": Sem("cvt", "sf", ["elem"]),
            "qfpair.mul_f16_wqf32": Sem("mul", "hf", ["elem"]), "qfpair.mul_f16_qf16": Sem("mul", "hf", ["elem"]),
            "qfpair.add_f16_qf16": Sem("add", "hf", ["elem"]), "qfpair.sub_f16_qf16": Sem("sub", "hf", ["elem"]),
            "qfpair.hf_roundtrip_qf16": Sem("cvt", "hf", ["elem"]),
            "qfpair.f16_to_f32": Sem("cvt", "sf", ["widen_eo", "widen_contig"]),
            "qfpair.f32_to_f16_vadd0": Sem("cvt", "hf", ["narrow_uv", "narrow_vu", "narrow_contig"]),
            "qfpair.f32_to_f16_equals": Sem("cvt", "hf", ["narrow_uv", "narrow_vu", "narrow_contig"]),
            "qfpair.f32_to_f16_mpy1": Sem("cvt", "hf", ["narrow_uv", "narrow_vu", "narrow_contig"]),
            "qfext.mul_f32_via_mem": Sem("mul", "sf", ["elem"]), "qfext.add_f32_via_mem": Sem("add", "sf", ["elem"]),
            "qfext.mul_f16_via_mem": Sem("mul", "hf", ["elem"]), "qfext.sub_f16_via_mem": Sem("sub", "hf", ["elem"]),
            "chain.mul_add_f32": Sem("mul_add", "sf", ["elem"]), "chain.add_add_f32": Sem("add_add", "sf", ["elem"]),
            "chain.mul_mul_f32": Sem("mul_mul", "sf", ["elem"]), "chain.mul_add_f16": Sem("mul_add", "hf", ["elem"]),
            "chain.qf_mul_cvt_add_f32": Sem("mul_add", "sf", ["elem"]),
            "chain.qf_mul_cvt_add_f32_barrier": Sem("mul_add", "sf", ["elem"]),
            "chain.dot4_f32": Sem("dot4", "sf", ["elem"]), "chain.dot4_qf_acc": Sem("dot4", "sf", ["elem"]),
            "chain.mul_add_mul_f32": Sem("mul_add_mul", "sf", ["elem"]),
            "chain.dot4_hf_sf": Sem("dot4w", "sf", ["widen_eo", "widen_contig"]),
            "cvt.sf_to_w_rne_int": Sem("toint", "w", ["elem"]),
            "cvt.sf_to_w_magic_qf": Sem("toint", "w", ["elem"], "lt2p22"),
            "cvt.sf_to_w_magic_ieee": Sem("toint", "w", ["elem"], "lt2p22"),
            "q8.mul_rne_int": Sem("mul_toint", "w", ["elem"]), "q8.exact_mul_rne_int": Sem("mul_toint", "w", ["elem"]),
            "q8.mul_magic_ieee": Sem("mul_toint", "w", ["elem"], "lt2p22"),
            "q8.qf_mul_magic": Sem("mul_toint", "w", ["elem"], "lt2p22"),
            "exact.sf_mul": Sem("mul", "sf", ["elem"]), "exact.sf_add": Sem("add", "sf", ["elem"]),
            "exact.hf_to_sf": Sem("cvt", "sf", ["widen_eo", "widen_contig"]),
            "exact.sf_to_hf": Sem("cvt", "hf", ["narrow_contig", "narrow_uv"]),
        }
        return table.get(n)
    m = re.match(r"Q6_([VW])(\w+?)_(\w+?)_(\w+)$", s)
    if not m:
        return None
    ret_vw, rt, op, args = m.groups()
    wide = ret_vw == "W"
    base = {"vadd": "add", "vsub": "sub", "vmpy": "mul", "vmpyacc": "fma", "vdmpy": "dmpy", "vdmpyacc": "dmpyacc",
            "vabs": "abs", "vfneg": "neg", "vneg": "neg", "vmax": "max", "vmin": "min", "vfmax": "max",
            "vfmin": "min", "vcvt": "cvt", "equals": "cvt"}.get(op)
    if base is None or "R" in args or rt in ("", "b", "ub") and base != "cvt":
        return None
    if rt in ("sf", "hf", "bf", "qf32", "qf16"):
        if base == "cvt":
            if wide:
                if "Vb" in args or "Vub" in args:
                    return Sem("cvt", rt, ["widen8_h", "widen8_eo", "widen8_contig"])
                return Sem("cvt", rt, ["widen_eo", "widen_contig"])
            if args.startswith("W"):
                return Sem("cvt", rt, ["narroww_lohi", "narroww_hilo", "narroww_contig"])
            if args.count("V") == 2:
                return Sem("cvt", rt, ["narrow_uv", "narrow_vu", "narrow_contig"])
            if "f8" in args or args == "V":
                return None
            return Sem("cvt", rt, ["elem"])
        if base in ("dmpy", "dmpyacc"):
            return Sem(base, rt, ["dmpy"])
        if "V" == args[-1:] or args.endswith("VV"):
            return None  # the f8 forms: the tool has no f8 model
        return Sem(base, rt, ["widen_eo", "widen_contig"] if wide else ["elem"])
    if rt in ("h", "uh", "w", "b", "ub") and base == "cvt":
        if args.count("V") == 2:
            return Sem("toint", rt, ["narrow_uv", "narrow_vu", "narrow_contig"])
        return Sem("toint", rt, ["elem"])
    return None


def layout_index(layout: str, n_out: int, out_per: int, in_per: int) -> tuple[np.ndarray, np.ndarray]:
    """For each output lane, return (the input slot of its source, the lane index in that stream).

    ``out_per`` and ``in_per`` are the lanes per iteration of the output and of one input. The
    layouts: elem (lane o reads lane o), widen_eo (the low vector holds the even input lanes),
    widen_contig, widen8_h (vdeal.h then vunpack), narrow_uv (output lane 2j is input u lane j,
    2j + 1 is v lane j), narrow_vu, narrow_contig, narroww_* (from one vector pair), dmpy (lane j
    reads input lanes 2j and 2j + 1).
    """
    o = np.arange(n_out)
    i, w = o // out_per, o % out_per
    zero = np.zeros(n_out, dtype=np.int64)
    if layout == "elem":
        return zero, o
    if layout == "widen_eo":
        half = out_per // 2
        return zero, in_per * i + 2 * (w % half) + (w // half)
    if layout == "widen_contig":
        return zero, in_per * i + w
    if layout == "widen8_h":
        half = out_per // 2
        t = w % half
        return zero, in_per * i + 4 * (t // 2) + (t % 2) + 2 * (w // half)
    if layout == "widen8_eo":
        half = out_per // 2
        return zero, in_per * i + 2 * (w % half) + (w // half)
    if layout == "widen8_contig":
        return zero, in_per * i + w
    if layout == "narrow_uv":
        return w % 2, in_per * i + w // 2
    if layout == "narrow_vu":
        return 1 - w % 2, in_per * i + w // 2
    if layout == "narrow_contig":
        return w // in_per, in_per * i + w % in_per
    if layout == "narroww_lohi":
        return zero, in_per * i + (w % 2) * (in_per // 2) + w // 2
    if layout == "narroww_hilo":
        return zero, in_per * i + (1 - w % 2) * (in_per // 2) + w // 2
    if layout == "narroww_contig":
        return zero, in_per * i + w
    if layout == "dmpy":
        return zero, in_per * i + 2 * w
    raise ValueError(f"unknown layout {layout}")


@dataclass
class Ref:
    """The reference of one op for one layout: the exact float64 value (qf, int) or the RNE bits."""

    layout: str
    exact: np.ndarray
    bits: np.ndarray | None = None
    tie: np.ndarray | None = None
    sub: np.ndarray | None = None
    alt: dict = field(default_factory=dict)
    valid: np.ndarray | None = None  # the lanes that count, None for all


def rne_step(x: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round float64 values to the format and return (the rounded float64 values, tie, subnormal)."""
    bits, tie, sub = round_f64(x, fmt)
    return ieee_to_f64(bits, fmt), tie, sub


def build_ref_steps(sem: Sem, layout: str, op: dict, streams: dict[str, np.ndarray], n_out: int,
                    lanes: "Callable[[int], tuple[np.ndarray, str, int]]") -> Ref:
    """The references of the chains with one IEEE rounding after each step (the CPU oracle).

    dot4: s = c, then four times s = rne(s + rne(a_k * b_k)), where a_k and b_k are the inputs
    rotated by k lanes within their vector (vror). dot4w: the same with f16 inputs and an f32 vector
    pair accumulator (the product of two f16 values is exact in f32). mul_add_mul: rne(rne(a * b) +
    rne(c * b_1)), b_1 = b rotated by one lane. mul_toint: rint(rne_f32(a * b)) with saturation.
    """
    o = np.arange(n_out)
    (arr0, et0, _), (arr1, et1, _) = lanes(0), lanes(1)
    sub = np.zeros(n_out, dtype=bool)
    tie = np.zeros(n_out, dtype=bool)

    def val(arr: np.ndarray, et: str, idx: np.ndarray) -> np.ndarray:
        """Decode the lanes idx of a stream and add their subnormal flags to ``sub``."""
        nonlocal sub
        b = arr[idx]
        sub |= is_subnormal(b, et)
        return decode(b, et)

    with np.errstate(all="ignore"):
        if sem.op in ("dot4", "mul_add_mul", "mul_toint"):
            v, i = o // 32, o % 32
            if sem.op == "mul_toint":
                p, _, sp = rne_step(val(arr0, et0, o) * val(arr1, et1, o), "sf")
                ref = Ref(layout, p)
                ref.sub = sub | sp
                nanv = np.isnan(p)
                ref.bits = np.where(nanv, 0, np.clip(np.rint(p), -2.0 ** 31, 2.0 ** 31 - 1)).astype(np.int64) & 0xFFFFFFFF
                if sem.domain == "lt2p22":
                    ref.valid = np.abs(np.where(nanv, np.inf, p)) < 2.0 ** 22
                return ref
            arr2, et2, _ = lanes(2)
            c = val(arr2, et2, o)
            if sem.op == "mul_add_mul":
                p1, _, s1 = rne_step(val(arr0, et0, o) * val(arr1, et1, o), "sf")
                p2, _, s2 = rne_step(c * val(arr1, et1, 32 * v + (i + 1) % 32), "sf")
                r, tie, s3 = rne_step(p1 + p2, "sf")
                sub |= s1 | s2 | s3
            else:
                r = c
                for k in range(4):
                    idx = 32 * v + (i + k) % 32
                    pk, _, s1 = rne_step(val(arr0, et0, idx) * val(arr1, et1, idx), "sf")
                    r, tie, s2 = rne_step(r + pk, "sf")
                    sub |= s1 | s2
        else:  # dot4w
            t, w = o // 64, o % 64
            j = 2 * (w % 32) + w // 32 if layout == "widen_eo" else w
            arr2, et2, _ = lanes(2)
            r = val(arr2, et2, o)
            for k in range(4):
                idx = 64 * t + (j + k) % 64
                r, tie, s2 = rne_step(r + val(arr0, et0, idx) * val(arr1, et1, idx), "sf")
                sub |= s2
    ref = Ref(layout, r)
    ref.bits, _, s4 = round_f64(r, "sf")
    ref.tie, ref.sub = tie, sub | s4
    return ref


def exact_dot(a0: np.ndarray, b0: np.ndarray, a1: np.ndarray, b1: np.ndarray, acc: np.ndarray | None,
              fmt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Round a0*b0 + a1*b1 (+ acc) to the format with exact Python integers. O(n) Python steps."""
    from fractions import Fraction
    vals = np.empty(a0.shape, dtype=np.float64)
    bits = np.empty(a0.shape, dtype=np.uint32)
    tie = np.zeros(a0.shape, dtype=bool)
    sub = np.zeros(a0.shape, dtype=bool)
    mb, eb, bias = FMT[fmt]
    p, emin, emax = mb + 1, 1 - bias, (1 << eb) - 2 - bias
    for k in range(a0.size):
        xs = [a0[k], b0[k], a1[k], b1[k]] + ([acc[k]] if acc is not None else [])
        if not all(np.isfinite(xs)):
            with np.errstate(all="ignore"):
                v = a0[k] * b0[k] + a1[k] * b1[k] + (acc[k] if acc is not None else 0.0)
            vals[k] = v
            bb, _, _ = round_f64(np.array([v]), fmt)
            bits[k] = bb[0]
            continue
        f = Fraction(a0[k]) * Fraction(b0[k]) + Fraction(a1[k]) * Fraction(b1[k])
        if acc is not None:
            f += Fraction(acc[k])
        vals[k] = float(f)
        if f == 0:
            # IEEE: an exact zero sum of nonzero terms is +0 in round to nearest
            terms = [a0[k] * b0[k], a1[k] * b1[k]] + ([acc[k]] if acc is not None else [])
            neg = all(np.signbit(t) for t in terms)
            bits[k] = (1 << (mb + eb)) if neg else 0
            continue
        s = f < 0
        f = abs(f)
        num, den = f.numerator, f.denominator
        e = num.bit_length() - den.bit_length()
        if Fraction(num, den) < Fraction(2) ** e:
            e -= 1
        sub[k] = e < emin
        q = max(e, emin) - (p - 1)
        scaled = f / (Fraction(2) ** q)
        r = scaled.numerator // scaled.denominator
        rem = scaled - r
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and r % 2 == 1):
            r += 1
        tie[k] = rem == Fraction(1, 2)
        if r >> p:
            r >>= 1
            q += 1
        ee = q + p - 1
        if r >> (p - 1):
            if ee > emax:
                bits[k] = (s << (mb + eb)) | (((1 << eb) - 1) << mb)
            else:
                bits[k] = (s << (mb + eb)) | ((ee + bias) << mb) | (r & ((1 << mb) - 1))
        else:
            bits[k] = (s << (mb + eb)) | r
    return bits, tie, sub


def build_ref(sem: Sem, layout: str, op: dict, streams: dict[str, np.ndarray], n_out: int) -> Ref | None:
    """Compute the reference of one op for one layout from the corpus streams."""
    out_t = op["out_type"]
    out_lb = LANE_BYTES[out_t]
    out_per = op["out_bytes"] // out_lb
    in_types = op["in_types"]
    in_streams = op["in_streams"]
    in_bytes = op["in_bytes"]

    def lanes(slot: int) -> tuple[np.ndarray, str, int]:
        """Return (the stream of the slot as lanes, the element type, the lanes per iteration)."""
        et = in_types[slot]
        lb = LANE_BYTES[et]
        arr = streams[in_streams[slot]].view(NP_UINT[lb])
        return arr, et, in_bytes[slot] // lb

    if sem.op in ("add", "sub", "mul", "fma", "max", "min"):
        arr0, et0, per0 = lanes(0)
        arr1, et1, _ = lanes(1)
        src, idx = layout_index(layout, n_out, out_per, per0)
        a_bits, b_bits = arr0[idx], arr1[idx]
        a, b = decode(a_bits, et0), decode(b_bits, et1)
        subin = is_subnormal(a_bits, et0) | is_subnormal(b_bits, et1)
        with np.errstate(all="ignore"):
            if sem.op == "add":
                v = a + b
            elif sem.op == "sub":
                v = a - b
            elif sem.op == "mul":
                v = a * b
            elif sem.op == "max":
                v = np.maximum(a, b)
            elif sem.op == "min":
                v = np.minimum(a, b)
            else:
                arr2, et2, per2 = lanes(2)
                acc_idx = np.arange(n_out)
                c_bits = arr2[acc_idx]
                c = decode(c_bits, et2)
                subin |= is_subnormal(c_bits, et2)
                v = a * b + c
        ref = Ref(layout, v)
        if sem.out in FMT:
            ref.bits, ref.tie, ref.sub = round_f64(v, sem.out)
            ref.sub = ref.sub | subin
            if sem.op == "fma":
                with np.errstate(all="ignore"):
                    pb, _, _ = round_f64(a * b, sem.out)
                    v2 = ieee_to_f64(pb, sem.out) + c
                ref.alt["double_round"] = round_f64(v2, sem.out)[0]
            if sem.op in ("max", "min"):
                # the value, with the sign of zero from the input that the op selects
                pick_a = (a >= b) if sem.op == "max" else (a <= b)
                ref.bits = np.where(pick_a, a_bits, b_bits).astype(np.uint32)
                ref.tie = np.zeros(n_out, dtype=bool)
        return ref
    if sem.op in ("mul_add", "add_add", "mul_mul"):
        # Two IEEE operations in sequence: the main reference rounds after each one, the "fused"
        # alternative rounds once (float64 is exact for a*b+c here, not for a*b*c)
        arrs = [lanes(k) for k in range(3)]
        vals, subin = [], np.zeros(n_out, dtype=bool)
        for arr, et, _ in arrs:
            vals.append(decode(arr[:n_out], et))
            subin |= is_subnormal(arr[:n_out], et)
        a, b, c = vals
        f1 = (lambda x, y: x * y) if sem.op in ("mul_add", "mul_mul") else (lambda x, y: x + y)
        f2 = (lambda x, y: x * y) if sem.op == "mul_mul" else (lambda x, y: x + y)
        with np.errstate(all="ignore"):
            fb, _, fsub = round_f64(f1(a, b), sem.out)
            first = ieee_to_f64(fb, sem.out)
            subin |= fsub
            v = f2(first, c)
            fused = f2(f1(a, b), c)
        ref = Ref(layout, v)
        ref.bits, ref.tie, ref.sub = round_f64(v, sem.out)
        ref.sub = ref.sub | subin
        if sem.op != "mul_mul":
            ref.alt["fused"] = round_f64(fused, sem.out)[0]
        return ref
    if sem.op in ("dot4", "dot4w", "mul_add_mul", "mul_toint"):
        return build_ref_steps(sem, layout, op, streams, n_out, lanes)
    if sem.op in ("dmpy", "dmpyacc"):
        arr0, et0, per0 = lanes(0)
        arr1, et1, _ = lanes(1)
        _, idx = layout_index("dmpy", n_out, out_per, per0)
        a0, b0 = decode(arr0[idx], et0), decode(arr1[idx], et1)
        a1, b1 = decode(arr0[idx + 1], et0), decode(arr1[idx + 1], et1)
        acc = None
        if sem.op == "dmpyacc":
            arr2, et2, _ = lanes(2)
            acc = decode(arr2[np.arange(n_out)], et2)
        bits, tie, sub = exact_dot(a0, b0, a1, b1, acc, sem.out)
        with np.errstate(all="ignore"):
            v = a0 * b0 + a1 * b1 + (acc if acc is not None else 0.0)
        return Ref(layout, v, bits, tie, sub)
    if sem.op in ("cvt", "abs", "neg", "toint"):
        per = [lanes(k)[2] if in_types[k] else 0 for k in range(3)]
        src, idx = layout_index(layout, n_out, out_per, per[0])
        a_bits = np.zeros(n_out, dtype=np.uint32)
        a = np.zeros(n_out, dtype=np.float64)
        subin = np.zeros(n_out, dtype=bool)
        for slot in (0, 1):
            if not in_types[slot]:
                continue
            arr, et, _ = lanes(slot)
            sel = src == slot
            if not sel.any():
                continue
            a_bits[sel] = arr[idx[sel]]
            a[sel] = decode(arr[idx[sel]], et)
            subin[sel] = is_subnormal(arr[idx[sel]], et)
        et0 = in_types[0]
        if sem.op == "toint":
            info = {"h": (-32768, 32767), "uh": (0, 65535), "w": (-2**31, 2**31 - 1), "b": (-128, 127),
                    "ub": (0, 255)}[sem.out]
            with np.errstate(all="ignore"):
                rne = np.clip(np.rint(a), info[0], info[1])
                trunc = np.clip(np.trunc(a), info[0], info[1])
            mask = (1 << (8 * LANE_BYTES[sem.out])) - 1
            ref = Ref(layout, a)
            nanv = np.isnan(a)
            if sem.domain == "lt2p22":
                ref.valid = np.abs(np.where(nanv, np.inf, a)) < 2.0 ** 22
            ref.bits = np.where(nanv, 0, rne).astype(np.int64) & mask
            ref.alt["trunc"] = np.where(nanv, 0, trunc).astype(np.int64) & mask
            ref.alt["floor"] = np.where(nanv, 0, np.clip(np.floor(a), info[0], info[1])).astype(np.int64) & mask
            return ref
        v = a
        if sem.op == "abs":
            v = np.abs(a)
        elif sem.op == "neg":
            v = -a
        ref = Ref(layout, v)
        if sem.out in FMT:
            ref.bits, ref.tie, ref.sub = round_f64(v, sem.out)
            ref.sub = ref.sub | subin
            if sem.op in ("abs", "neg") and et0 == sem.out:
                mb, eb, _ = FMT[sem.out]
                sb = 1 << (mb + eb)
                ref.bits = (a_bits & (sb - 1)) if sem.op == "abs" else (a_bits ^ sb)
                ref.bits = ref.bits.astype(np.uint32)
        return ref
    return None


# ---------------------------------------------------------------------------------------------
# Disassembly
# ---------------------------------------------------------------------------------------------


def parse_objdump(path: Path) -> dict[int, str]:
    """Return {op id: "insn ; insn | insn"} for the isa_core_<k> functions of one objdump file.

    Packets are separated by " | ", the instructions of one packet by " ; ". The return
    instruction, nop and the predicate glue of the call ABI are not listed.
    """
    glue = re.compile(r"^(r\d+ = #-0x1|q\d = vand\(v\d+,r\d+\)|v\d+ = vand\(q\d,r\d+\)|jumpr r31|nop)$")
    out: dict[int, str] = {}
    if not path.exists():
        return out
    cur_id = None
    packets: list[list[str]] = []
    pkt: list[str] = []
    for line in path.read_text().splitlines():
        m = re.match(r"^[0-9a-f]+ <(\w+)>:", line)
        if m:
            if cur_id is not None:
                out[cur_id] = " | ".join(" ; ".join(p) for p in packets if p)
            cur_id = None
            packets, pkt = [], []
            fm = re.match(r"isa_core_(\d+)$", m.group(1))
            if fm:
                cur_id = int(fm.group(1))
            continue
        if cur_id is None:
            continue
        lm = re.match(r"^\s*[0-9a-f]+:\s*(.*)$", line)
        if not lm:
            continue
        s = lm.group(1)
        if "R_HEX" in s:
            call = re.search(r"R_HEX_\w+\s+(\S+)", s)
            if call and packets:
                packets[-1] = [x if not x.startswith("call") else f"call {call.group(1)}" for x in packets[-1]]
            continue
        start, end = "{" in s, "}" in s
        s = " ".join(s.replace("{", "").replace("}", "").split())
        s = re.sub(r"\s*//.*$", "", s)
        if start and pkt:
            packets.append(pkt)
            pkt = []
        if s and not glue.match(s):
            pkt.append(s)
        if end:
            packets.append(pkt)
            pkt = []
    if cur_id is not None:
        out[cur_id] = " | ".join(" ; ".join(p) for p in packets if p)
    return out


# ---------------------------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------------------------


def load_bench(label: str) -> dict[str, tuple[float, int]]:
    """Read the cost bench of one version: {op: (cycles per iteration, output vectors per iteration)}."""
    out: dict[str, tuple[float, int]] = {}
    f = run_dir(label, "bench") / "stdout.txt"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        m = re.match(r"lab: isa-bench (\S+) ([0-9.]+) (\d+)$", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    return out


def verdict_of(v: dict) -> str:
    """The primary verdict of one version against the oracle: EXACT, EXACT-NORMAL (exact for the
    lanes without a subnormal input, intermediate or result), EXACT-FINITE (exact also where the
    oracle result is finite, normal and not zero: the differences are only in the sign of zero, in
    overflow, NaN and subnormal lanes) or the fraction of exact lanes."""
    if "rne_exact" not in v:
        return ""
    if v["rne_exact"] >= 1.0:
        return "EXACT"
    if v.get("rne_exact_normal", 0.0) >= 1.0:
        return "EXACT-NORMAL"
    if v.get("rne_exact_finite", 0.0) >= 1.0:
        return "EXACT-FINITE"
    return f"{v['rne_exact']:.4f}"


# The candidates of the cheapest CPU-exact way for each task
TASKS = {
    "sf mul": ["ieee.Q6_Vsf_vmpy_VsfVsf", "cc.Q6_Vsf_vmpy_VsfVsf", "seq.qfpair.mul_f32", "seq.hvx_vec_mul_f32_f32",
               "seq.exact.sf_mul"],
    "sf add": ["ieee.Q6_Vsf_vadd_VsfVsf", "cc.Q6_Vsf_vadd_VsfVsf", "seq.qfpair.add_f32", "seq.hvx_vec_add_f32_f32",
               "seq.exact.sf_add"],
    "hf->sf (64 values)": ["ieee.Q6_Wsf_vcvt_Vhf", "cc.Q6_Wsf_vcvt_Vhf", "seq.hvx_vec_f16_to_f32",
                           "seq.hvx_vec_f16_to_f32_shuff", "seq.qfpair.f16_to_f32", "seq.exact.hf_to_sf"],
    "sf->hf (64 values)": ["ieee.Q6_Vhf_vcvt_VsfVsf", "cc.Q6_Vhf_vcvt_VsfVsf", "seq.hvx_vec_f32_to_f16",
                           "seq.hvx_vec_f32_to_f16_shuff", "seq.qfpair.f32_to_f16_vadd0", "seq.qfpair.f32_to_f16_mpy1",
                           "seq.qfpair.f32_to_f16_equals", "seq.exact.sf_to_hf"],
    "sf->int32 rne": ["Q6_Vw_equals_Vsf", "seq.cvt.sf_to_w_rne_int", "seq.cvt.sf_to_w_magic_qf",
                      "seq.cvt.sf_to_w_magic_ieee"],
    "q8 quant rne(x*id)": ["seq.q8.mul_rne_int", "seq.q8.mul_magic_ieee", "seq.q8.qf_mul_magic",
                           "seq.q8.exact_mul_rne_int"],
}


# An IEEE-form opcode: an IEEE arithmetic instruction on sf, hf, bf or f8 operands with a result
# that is not qf, and each vcvt and vcvt2. vmax and vmin of sf and hf and the "v.hf = v.h" and
# "v.sf = v.w" conversions are native on each version and are not in the list.
IEEE_OPCODE_RE = re.compile(
    r"v\d+(?::\d+)?\.(?:sf|hf|bf|f8|b|ub|h|uh)\s*\+?=\s*(?:vadd|vsub|vmpy|vdmpy|vabs|vfneg|vfmax|vfmin)"
    r"\(v\d+(?::\d+)?\.(?:sf|hf|bf|f8)\b"
    r"|=\s*vcvt2?\("
    r"|v\d+\.bf\s*=\s*(?:vmax|vmin)\(v\d+\.bf\b")


def silicon_note(disasm: str, label: str) -> str:
    """The silicon status of an op whose compiled code holds an IEEE-form opcode: the v79 chip gives
    inf for them, the v81 simulator gives 0, and no v73 or v75 chip has run them."""
    if not IEEE_OPCODE_RE.search(disasm):
        return ""
    return {"v79": "IEEE opcode: inf on v79 silicon", "v81": "IEEE opcode: 0 on the v81 simulator"}.get(
        label[:3], "IEEE opcode: not verified on silicon")


def load_ops() -> list[dict]:
    """Read isa_ops.csv into a list of op dicts with parsed fields."""
    ops = []
    with open(ISA_DIR / "isa_ops.csv") as f:
        for r in csv.DictReader(f):
            r["id"] = int(r["id"])
            r["in_types"] = [("pred" if t == "pred" else t) for t in r["in_types"].split("/")]
            r["in_streams"] = r["in_streams"].split("/")
            r["in_bytes"] = [int(b) for b in r["in_bytes"].split("/")]
            r["out_bytes"] = int(r["out_bytes"])
            r["n_vectors"] = int(r["n_vectors"])
            if r["out_ctype"] == "Q":
                r["out_type"] = "pred"
            ops.append(r)
    return ops


def lanes_of(data: np.ndarray, et: str) -> np.ndarray:
    """View output bytes as lanes of the element type."""
    return data.view(NP_UINT[LANE_BYTES[et]])


def special_inputs(op: dict, streams: dict[str, np.ndarray], n_out: int, layout: str | None,
                   sem: Sem | None) -> np.ndarray:
    """Return True for each output lane with an IEEE input that is NaN or infinite.

    Without a layout the lane mapping is the byte position, which is correct for elementwise ops
    and for the compares. An input with a different chunk size than the output is then ignored.
    """
    et = op["out_type"]
    out_lb = LANE_BYTES[et]
    out_per = op["out_bytes"] // out_lb
    mask = np.zeros(n_out, dtype=bool)
    o = np.arange(n_out)
    for slot in range(3):
        t = op["in_types"][slot]
        if t not in FMT:
            continue
        lb = LANE_BYTES[t]
        per_in = op["in_bytes"][slot] // lb
        arr = streams[op["in_streams"][slot]].view(NP_UINT[lb])
        idxs: list[np.ndarray] = []
        if layout and sem is not None:
            if slot == 2 and sem.op in ("fma", "dmpyacc"):
                idxs = [o]
            else:
                src, idx = layout_index(layout, n_out, out_per, per_in)
                sel = (src == slot) if layout.startswith("narrow_") else np.ones(n_out, dtype=bool)
                idx = np.where(sel, idx, -1)
                idxs = [idx] + ([np.where(idx >= 0, idx + 1, -1)] if layout == "dmpy" else [])
        elif op["in_bytes"][slot] == op["out_bytes"]:
            idxs = [(o * out_lb) // lb]
        for idx in idxs:
            ok = (idx >= 0) & (idx < arr.size)
            v = np.zeros(n_out, dtype=arr.dtype)
            v[ok] = arr[idx[ok]]
            n_, i_, _ = ieee_class(v, t)
            mask |= ok & (n_ | i_)
    return mask


def pair_diff(a: np.ndarray, b: np.ndarray, et: str, ref: Ref | None, special: np.ndarray | None = None) -> dict:
    """Describe the difference of the outputs of two versions.

    ``special`` flags the lanes with a NaN or infinite input. The result "finite" is the fraction
    of lanes that differ although no input and neither output is NaN or infinite (for a qf result:
    although neither value is outside the sf range).
    """
    la, lb = lanes_of(a, et), lanes_of(b, et)
    diff = la != lb
    n = int(diff.sum())
    res = {"frac": n / la.size, "kinds": {}, "max_ulp": 0, "mean_ulp": 0.0, "value_same": None, "finite": 0.0}
    if n == 0:
        return res
    sp = special.copy() if special is not None else np.zeros(la.size, dtype=bool)
    if et in FMT:
        an, ai, _ = ieee_class(la, et)
        bn, bi, _ = ieee_class(lb, et)
        sp |= an | ai | bn | bi
        if ref is not None and ref.bits is not None:
            rn, ri, _ = ieee_class(ref.bits.astype(np.uint32), et)
            sp |= rn | ri
    elif et in ("qf32", "qf16"):
        va, vb = qf_to_f64(la, et), qf_to_f64(lb, et)
        sp |= (np.abs(va) >= 2.0 ** 128) | (np.abs(vb) >= 2.0 ** 128)
    res["finite"] = float((diff & ~sp).sum()) / la.size
    if et in FMT:
        tie = ref.tie if ref is not None and ref.tie is not None else np.zeros(la.size, dtype=bool)
        sub = ref.sub if ref is not None and ref.sub is not None else (is_subnormal(la, et) | is_subnormal(lb, et))
        res["kinds"] = classify_ieee(la.astype(np.uint32), lb.astype(np.uint32), et, tie, sub)
        gn, _, _ = ieee_class(la, et)
        rn, _, _ = ieee_class(lb, et)
        ok = diff & ~gn & ~rn
        if ok.any():
            d = np.abs(ordered(la, et) - ordered(lb, et))[ok]
            res["max_ulp"] = int(d.max())
            res["mean_ulp"] = float(d.mean())
    elif et in ("qf32", "qf16"):
        va, vb = qf_to_f64(la, et), qf_to_f64(lb, et)
        same_val = diff & (va == vb)
        res["value_same"] = int(same_val.sum())
        vd = diff & (va != vb)
        res["kinds"] = {"bits_only": int(same_val.sum()), "value": int(vd.sum())}
        big = vd & ((np.abs(va) >= 2.0 ** 127) | (np.abs(vb) >= 2.0 ** 127) |
                    (np.abs(va) < 2.0 ** -126) | (np.abs(vb) < 2.0 ** -126))
        if big.any():
            res["kinds"]["range_edge"] = int(big.sum())
            res["kinds"]["value"] = int((vd & ~big).sum())
        vd = vd & ~big
        if vd.any():
            tgt = QF_TARGET[et]
            ulp = ulp_of(np.maximum(np.abs(va), np.abs(vb)), tgt)
            d = np.abs(va - vb)[vd] / ulp[vd]
            res["max_ulp"] = float(d.max())
            res["mean_ulp"] = float(d.mean())
    else:
        res["kinds"] = {"bits": n}
        if et in ("b", "ub", "h", "uh", "w", "uw"):
            da = np.abs(int_to_f64(la, et) - int_to_f64(lb, et))[diff]
            res["max_ulp"] = float(da.max())
            res["mean_ulp"] = float(da.mean())
    return res


def ulp_of(x: np.ndarray, fmt: str) -> np.ndarray:
    """The ulp of the IEEE format at the magnitude x (the subnormal ulp below the smallest normal)."""
    mb, eb, bias = FMT[fmt]
    emin = 1 - bias
    with np.errstate(all="ignore"):
        e = np.floor(np.log2(np.where(np.isfinite(x) & (x > 0), x, 1.0)))
    e = np.maximum(e, emin)
    return np.ldexp(1.0, (e - mb).astype(np.int64))


def vs_ref(got: np.ndarray, et: str, ref: Ref) -> dict:
    """Compare the output of one version with the reference."""
    lg = lanes_of(got, et)
    res: dict = {}
    if et in FMT and ref.bits is not None:
        rb = ref.bits.astype(np.uint32)
        g = lg.astype(np.uint32)
        gn, _, _ = ieee_class(g, et)
        rn, _, _ = ieee_class(rb, et)
        eq = (g == rb) | (gn & rn)
        res["rne_exact"] = float(eq.mean())
        if ref.sub is not None:
            nrm = ~ref.sub
            res["rne_exact_normal"] = float(eq[nrm].mean()) if nrm.any() else 1.0
            _, ri, rz = ieee_class(rb, et)
            fin = nrm & ~rn & ~ri & ~rz
            res["rne_exact_finite"] = float(eq[fin].mean()) if fin.any() else 1.0
        ok = ~eq & ~gn & ~rn
        d = np.abs(ordered(g, et) - ordered(rb, et))
        res["max_ulp"] = int(d[ok].max()) if ok.any() else 0
        res["kinds"] = classify_ieee(np.where(gn & rn, rb, g), rb, et, ref.tie, ref.sub)
        for name, alt in ref.alt.items():
            an, _, _ = ieee_class(alt.astype(np.uint32), et)
            res["match_" + name] = float(((g == alt) | (gn & an)).mean())
    elif et in ("qf32", "qf16"):
        v = qf_to_f64(lg, et)
        tgt = QF_TARGET[et]
        mb_, eb_, bias_ = FMT[tgt]
        lo_, hi_ = 2.0 ** (1 - bias_), 2.0 ** ((1 << eb_) - 1 - bias_)
        mag = np.abs(ref.exact)
        fin = np.isfinite(ref.exact) & ((mag == 0) | ((mag >= lo_) & (mag < hi_)))
        res["out_of_range"] = int((~fin).sum())
        ulp = ulp_of(np.abs(ref.exact), tgt)
        with np.errstate(all="ignore"):
            err = np.abs(v - ref.exact) / ulp
        err = err[fin]
        res["exact"] = float((v[fin] == ref.exact[fin]).mean()) if fin.any() else 0.0
        res["max_ulp"] = float(err.max()) if err.size else 0.0
        res["mean_ulp"] = float(err.mean()) if err.size else 0.0
        rb, _, _ = round_f64(v, tgt)
        eb_, _, _ = round_f64(ref.exact, tgt)
        res["rne_exact"] = float((rb[fin] == eb_[fin]).mean()) if fin.any() else 0.0
    elif ref.bits is not None:
        rb = ref.bits
        g = lg.astype(np.int64)
        ok = ref.valid if ref.valid is not None else np.ones(g.size, dtype=bool)
        res["rne_exact"] = float((g == rb)[ok].mean()) if ok.any() else 0.0
        if ref.sub is not None:
            nrm = ok & ~ref.sub
            res["rne_exact_normal"] = float((g == rb)[nrm].mean()) if nrm.any() else 1.0
        if ref.valid is not None:
            res["valid_frac"] = float(ok.mean())
        if not (g == rb)[ok].all():
            d = np.abs(g.astype(np.int64) - rb.astype(np.int64))
            d = np.minimum(d, (1 << (8 * LANE_BYTES[et])) - d)
            res["max_ulp"] = float(d[ok & (g != rb)].max())
        for name, alt in ref.alt.items():
            res["match_" + name] = float((g == alt)[ok].mean())
    return res


def show_value(bits: int, et: str) -> str:
    """Format one lane as hex and value."""
    b = np.array([bits], dtype=np.int64)
    if et in FMT or et in ("qf32", "qf16"):
        v = decode(b.astype(NP_UINT[LANE_BYTES[et]]), et)[0]
        return f"{bits:0{2 * LANE_BYTES[et]}x}({v:.9g})"
    return f"{bits:0{2 * LANE_BYTES.get(et, 4)}x}"


def detail(op: dict, per: dict[str, np.ndarray], ref: Ref | None, streams: dict[str, np.ndarray], n: int) -> None:
    """Print up to n example lanes where the versions differ or differ from the reference."""
    et = op["out_type"]
    archs = list(per)
    lanes = {a: lanes_of(per[a], et) for a in archs}
    base = lanes[archs[0]]
    bad = np.zeros(base.size, dtype=bool)
    for a in archs[1:]:
        bad |= lanes[a] != base
    if ref is not None and ref.bits is not None:
        for a in archs:
            bad |= lanes[a].astype(np.int64) != ref.bits.astype(np.int64)
    idx = np.nonzero(bad)[0]
    print(f"{op['name']}: {idx.size} lanes differ (layout {ref.layout if ref else '-'}), first {min(n, idx.size)}:")
    step = max(1, idx.size // n) if idx.size > n else 1
    for o in idx[::step][:n]:
        ins = []
        sem = sem_of(op["name"], op["source"])
        if ref is not None:
            out_per = op["out_bytes"] // LANE_BYTES[et]
            for slot in range(3):
                t = op["in_types"][slot]
                if not t:
                    continue
                lb = LANE_BYTES[t]
                per_in = op["in_bytes"][slot] // lb
                src, ii = layout_index(ref.layout, base.size, out_per, per_in)
                if sem and sem.op in ("fma", "dmpyacc") and slot == 2:
                    ii = np.arange(base.size)
                arr = streams[op["in_streams"][slot]].view(NP_UINT[lb])
                if src[o] != slot and ref.layout.startswith("narrow_"):
                    continue
                ins.append(show_value(int(arr[ii[o]]), t))
        outs = " ".join(f"{a}={show_value(int(lanes[a][o]), et)}" for a in archs)
        r = f" ref={show_value(int(ref.bits[o]), et)}" if ref is not None and ref.bits is not None else ""
        print(f"  lane {o}: in {' '.join(ins)} -> {outs}{r}")


def fmt_kinds(k: dict) -> str:
    """Format a kind count dict as "tie:12 round:3"."""
    return " ".join(f"{a}:{b}" for a, b in sorted(k.items(), key=lambda t: -t[1]))


def exp2_report(results: dict[str, dict[str, np.ndarray]], streams: dict[str, np.ndarray], archs: list[str]) -> str:
    """Report the exp2 f16 accuracy per version, as target_f16math measures it, and the first stage
    of hvx_vec_exp2_f16 where the versions differ."""
    lines = []
    x = ieee_to_f64(streams["hf_a"].view(np.uint16)[:65536], "hf")
    for lo, hi in ((-12.0, 0.0), (-1.0, 0.0), (-24.0, 15.0)):
        row = [f"exp2_f16 over the hf values in [{lo}, {hi}]:"]
        sel = (x >= lo) & (x <= hi)
        for a in archs:
            d = results.get("seq.hvx_vec_exp2_f16", {}).get(a)
            if d is None:
                continue
            g = ieee_to_f64(d.view(np.uint16)[:65536], "hf")
            r = np.exp2(x)
            use = sel & (np.abs(r) > 1e-4) & (np.abs(r) < 6e4)
            e = np.abs(g[use] - r[use]) / np.abs(r[use])
            row.append(f"{a} rms rel err {np.sqrt(np.mean(e * e)):.3g} max {e.max():.3g}")
        lines.append("  ".join(row))
    stage_names = sorted(n for n in results if n.startswith("seq.exp2_f16.s"))
    lines.append("stage differences (fraction of the 65536 hf inputs, first 65536 lanes):")
    for n in stage_names:
        per = results[n]
        base = per.get(archs[0])
        cells = []
        for a in archs[1:]:
            if base is None or per.get(a) is None:
                continue
            et = "h" if n.endswith("_k") else "hf"
            la = base.view(np.uint16)[:65536]
            lb = per[a].view(np.uint16)[:65536]
            cells.append(f"{archs[0]}~{a} {np.mean(la != lb):.4f}")
        lines.append(f"  {n:34s} " + "  ".join(cells))
    return "\n".join(lines)


def build_summary(rows: list[dict], archs: list[str]) -> str:
    """Per version: the cheapest candidate of each task at each exactness tier (EXACT, EXACT-NORMAL,
    EXACT-FINITE), without the ops that hold an IEEE-form opcode, and the list of the exact ops with
    their cost. The IEEE-opcode candidates are listed with their note."""
    tiers = {"EXACT": 0, "EXACT-NORMAL": 1, "EXACT-FINITE": 2}
    by = {r["name"]: r for r in rows}
    out = ["== the cheapest CPU-exact way per version (cycles per iteration; a W output is two vectors)",
           "   tiers: EXACT; EXACT-NORMAL (not exact only with a subnormal input, intermediate or result);",
           "   EXACT-FINITE (not exact also for the sign of zero, overflow and NaN)"]
    for task, cands in TASKS.items():
        out.append(f"-- {task}")
        for a in archs:
            cells, best = [], {}
            for c in cands:
                r = by.get(c)
                if r is None or r[f"verdict_{a}"] in ("n/a", ""):
                    continue
                vd, cy, note = r[f"verdict_{a}"], r[f"cycles_{a}"], r[f"note_{a}"]
                short = c.split(".", 1)[-1] if c.startswith("seq.") else c
                cells.append(f"{short}={vd}@{cy or '?'}{' [' + note + ']' if note else ''}")
                if note or vd not in tiers:
                    continue
                cyv = float(cy) if cy else 1e9
                for t, rank in tiers.items():
                    if tiers[vd] <= rank and (t not in best or cyv < best[t][1]):
                        best[t] = (short, cyv)
            picks = [f"{t}: {best[t][0]} {best[t][1]:.1f}c" for t in tiers if t in best]
            dedup = []
            for pk in picks:
                if not dedup or pk.split(": ", 1)[1] != dedup[-1].split(": ", 1)[1]:
                    dedup.append(pk)
            out.append(f"  {a:5s} {' | '.join(dedup) if dedup else 'none without an IEEE opcode'}")
            out.append(f"        candidates: {'  '.join(cells)}")
    out.append("\n== the float ops that are exact against the CPU oracle, per version (cycles per iteration),")
    out.append("   without the ops that hold an IEEE-form opcode (their count is at the end of each line)")
    for a in archs:
        good = [r for r in rows if r[f"verdict_{a}"] in tiers and r["class"] != "int_round" and not r[f"note_{a}"]]
        n_ieee = sum(1 for r in rows if r[f"verdict_{a}"] in tiers and r[f"note_{a}"])
        items = [f"{r['name']}({r[f'verdict_{a}'].replace('EXACT', 'E')},{r[f'cycles_{a}'] or '?'}c)" for r in good]
        out.append(f"-- {a}: {len(good)} ops, plus {n_ieee} exact ops with an IEEE-form opcode")
        out.append("   " + "  ".join(items))
    return "\n".join(out)


def main() -> int:
    """Compare the versions and write census.csv, then print the compact table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="c")
    ap.add_argument("--archs", nargs="+", default=["v73", "v75", "v79", "v81"])
    ap.add_argument("--only", default="", help="a regular expression of the op names to compare")
    ap.add_argument("--detail", type=int, default=0, help="print this many example lanes for each op")
    args = ap.parse_args()
    archs = args.archs
    ops = load_ops()

    # The corpus must be the same on each version
    streams: dict[str, np.ndarray] = {}
    corpus_ok = True
    for a in archs:
        d = run_dir(a, args.tag)
        for f in sorted(d.glob("corpus_*.bin")):
            hdr, data = read_bin(f)
            name = hdr["name"]
            if name not in streams:
                streams[name] = data
            elif not np.array_equal(streams[name], data):
                print(f"error: the corpus stream {name} of {a} is not the same as that of {archs[0]}", file=sys.stderr)
                corpus_ok = False
    if not streams:
        print(f"error: no corpus files in {run_dir(archs[0], args.tag)}. Run run_census.sh first.", file=sys.stderr)
        return 1
    streams["none"] = streams["int_a"]
    h0 = isa_hash(streams["sf_a"])
    hdr0, _ = read_bin(run_dir(archs[0], args.tag) / "corpus_sf_a.bin")
    if h0 != hdr0["hash"]:
        print(f"error: isa_hash in compare.py ({h0:08x}) does not agree with the file ({hdr0['hash']:08x})", file=sys.stderr)
        corpus_ok = False

    disasm = {a: parse_objdump(OUT_DIR / f"objdump-{a}.txt") for a in archs}
    bench = {a: load_bench(a) for a in archs}
    only = re.compile(args.only) if args.only else None

    rows = []
    results: dict[str, dict[str, np.ndarray]] = {}
    for op in ops:
        if only and not only.search(op["name"]):
            continue
        per: dict[str, np.ndarray] = {}
        for a in archs:
            r = read_bin(run_dir(a, args.tag) / f"out_{op['name']}.bin")
            if r is not None:
                per[a] = r[1]
        results[op["name"]] = per
        et = op["out_type"]
        present = [a for a in archs if a in per]
        groups: list[list[str]] = []
        for a in present:
            for g in groups:
                if np.array_equal(per[g[0]], per[a]):
                    g.append(a)
                    break
            else:
                groups.append([a])
        sem = sem_of(op["name"], op["source"])
        n_out = (op["n_vectors"] * op["out_bytes"]) // LANE_BYTES[et] if et in LANE_BYTES else 0
        refs: dict[str, Ref] = {}
        if sem is not None and present:
            for lay in sem.layouts:
                try:
                    rr = build_ref(sem, lay, op, streams, n_out)
                except (IndexError, KeyError, ValueError) as ex:
                    rr = None
                if rr is not None:
                    refs[lay] = rr
        per_arch_ref: dict[str, dict] = {}
        chosen: dict[str, str] = {}
        for a in present:
            best, best_score = None, -1.0
            for lay, rr in refs.items():
                v = vs_ref(per[a], et, rr)
                score = v.get("rne_exact", v.get("exact", 0.0))
                if score > best_score:
                    best, best_score = (lay, v), score
            if best:
                chosen[a] = best[0]
                per_arch_ref[a] = best[1]
        ref_for_diff = refs.get(next(iter(chosen.values()))) if chosen else None
        if args.detail and per:
            detail(op, per, ref_for_diff, streams, args.detail)
        pairs = []
        special = None
        if len(groups) > 1:
            lay = next(iter(chosen.values())) if chosen else None
            special = special_inputs(op, streams, n_out, lay, sem)
        for gi in range(1, len(groups)):
            pd = pair_diff(per[groups[0][0]], per[groups[gi][0]], et, ref_for_diff, special)
            pairs.append((groups[0][0], groups[gi][0], pd))
        if len(groups) == 1:
            verdict = "SAME"
        elif not groups:
            verdict = "n/a"
        elif all(p[2]["finite"] == 0.0 for p in pairs):
            verdict = "FINITE-SAME"
        else:
            verdict = "DIFF"
        row = {
            "id": op["id"], "name": op["name"], "kind": op["kind"], "class": op["class"], "out_type": et,
            "versions": " ".join(present), "missing": " ".join(a for a in archs if a not in present),
            "identical": "yes" if len(groups) == 1 else ("n/a" if not groups else "no"),
            "verdict": verdict,
            "diff_finite_frac": " ".join(f"{p[0]}~{p[1]}:{p[2]['finite']:.4g}" for p in pairs),
            "groups": " | ".join("=".join(g) for g in groups),
            "diff_frac": " ".join(f"{p[0]}~{p[1]}:{p[2]['frac']:.4g}" for p in pairs),
            "diff_kinds": " ; ".join(f"{p[0]}~{p[1]}: {fmt_kinds(p[2]['kinds'])}" for p in pairs),
            "diff_max_ulp": " ".join(f"{p[2]['max_ulp']:.4g}" for p in pairs),
            "diff_mean_ulp": " ".join(f"{p[2]['mean_ulp']:.3g}" for p in pairs),
            "layout": " ".join(sorted(set(chosen.values()))),
        }
        for a in archs:
            v = per_arch_ref.get(a, {})
            row[f"rne_exact_{a}"] = f"{v['rne_exact']:.4f}" if "rne_exact" in v else ""
            row[f"max_ulp_{a}"] = f"{v['max_ulp']:.4g}" if "max_ulp" in v else ""
            extra = []
            if v.get("out_of_range"):
                extra.append(f"out_of_range:{v['out_of_range']}")
            if "mean_ulp" in v:
                extra.append(f"mean_ulp:{v['mean_ulp']:.3g}")
            if "exact" in v:
                extra.append(f"value_exact:{v['exact']:.4f}")
            for k2, v2 in v.items():
                if k2.startswith("match_"):
                    extra.append(f"{k2[6:]}:{v2:.4f}")
            if v.get("kinds"):
                extra.append("vs_rne " + fmt_kinds(v["kinds"]))
            row[f"ref_{a}"] = " ".join(extra)
        for a in archs:
            v = per_arch_ref.get(a, {})
            row[f"verdict_{a}"] = verdict_of(v) if a in present else "n/a"
            row[f"exact_normal_{a}"] = f"{v['rne_exact_normal']:.4f}" if "rne_exact_normal" in v else ""
            row[f"exact_finite_{a}"] = f"{v['rne_exact_finite']:.4f}" if "rne_exact_finite" in v else ""
            b = bench[a].get(op["name"])
            row[f"cycles_{a}"] = f"{b[0]:.2f}" if b else ""
            d = disasm[a].get(op["id"], "")
            row[f"packets_{a}"] = str(len(d.split(" | "))) if d else ""
            row[f"note_{a}"] = silicon_note(d, a) if a in present else ""
        for a in archs:
            row[f"insns_{a}"] = disasm[a].get(op["id"], "")
        rows.append(row)

    fields = list(rows[0].keys()) if rows else []
    with open(OUT_DIR / "census.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    exp2 = exp2_report(results, streams, archs) if "seq.hvx_vec_exp2_f16" in results else ""
    if exp2:
        (OUT_DIR / "exp2.txt").write_text(exp2 + "\n")

    # The compact table: per version the verdict against the oracle and the cost, then the bit
    # identity between the versions (secondary)
    lines = [f"corpus identical on {', '.join(archs)}: {'yes' if corpus_ok else 'NO'}",
             "per version: verdict against the CPU oracle (EXACT, EXACT-NORMAL or the exact fraction) / "
             "max error (ulp) / cycles per iteration in the bench; then the bit groups"]
    cur = None
    for r in sorted(rows, key=lambda r: (r["class"], r["id"])):
        if r["class"] != cur:
            cur = r["class"]
            lines.append(f"\n== {cur}")
        cells = []
        for a in archs:
            vd = r[f"verdict_{a}"]
            if vd == "n/a":
                cells.append(f"{a}:-")
                continue
            mx = r[f"max_ulp_{a}"]
            err = f"/{mx}" if vd not in ("EXACT", "") and mx else ""
            cells.append(f"{a}:{vd or '.'}{err}/{r[f'cycles_{a}'] or '?'}c")
        lines.append(f"{r['name'][:44]:44s} {' '.join(cells)}   bits {r['groups']}")
    table = "\n".join(lines)
    print(table)
    if exp2:
        print("\n" + exp2)
    summary = build_summary(rows, archs)
    (OUT_DIR / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nwrote {OUT_DIR / 'census.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
