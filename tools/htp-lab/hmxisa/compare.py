#!/usr/bin/env python3
"""Compare the HMX census of tools/htp-lab/lab/target_hmxisa.c across Hexagon versions.

The lab target writes one file "<case>.bin" for each case in the run directory of each version.
This tool reads the files of all versions and reports for each case:

- if the output bytes are identical on all versions, and else which versions agree;
- for the f16 cases, the difference to the exact rational dot product, rounded to f16;
- for the f16 cases, the difference to the software model of the HMX accumulator (xfp_model), with
  the parameters of each version;
- for the int cases, the difference to the exact integer product.

Then it prints the characterization of each family of cases (the handling of special values, the
rounding at the store, the alignment depth, the effect of the bias area and of USR, the int path),
the HMX paths of an F16 and of a Q8_0 matmul against the CPU oracle (the scalar reference code of
ggml with IEEE arithmetic) and against the exact dot, and a ranking of the paths per version.

Usage:
    compare.py [--run LABEL=DIR ...] [--case PREFIX] [--no-extra]

Without --run the tool reads the four default runs of tools/htp-lab/hmxisa/run.sh and the other
cores that run.sh --extra wrote.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "tools/htp-lab/out-hmxisa"

# The default runs: label, run directory (refer to run.sh)
DEFAULT_RUNS = [
    ("v73", OUT / "hmxisa-v73-c"),
    ("v75", OUT / "hmxisa-v75-c"),
    ("v79", OUT / "hmxisa-c"),
    ("v81", OUT / "hmxisa-v81-c"),
]

KIND_F16, KIND_I8, KIND_I4, KIND_Q8_SRC = 1, 2, 3, 4
M_RETAIN, M_BLOCK, M_I8_REPEAT, M_I8_BLOCK = 5, 7, 12, 14  # enum method of target_hmxisa.c

HDR = struct.Struct("<8s14I")  # magic, kind, method, m, k, n, bias, a, w, out, n_snap, param[3], usr

F16_MAX_BITS = 0x7BFF  # 65504, the IEEE maximum
F16_EXT_MAX_BITS = 0x7FFF  # 131008, the maximum when exponent 31 is an ordinary exponent


# ---------------------------------------------------------------------------------------------
# file parsing
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Case:
    """One case file: the header fields, the bias area and the A, W and output sections."""

    name: str
    kind: int
    method: int
    m: int
    k: int
    n: int
    n_snap: int
    param: tuple[int, int, int]
    usr: int
    bias: bytes
    a: np.ndarray
    w: np.ndarray
    out: np.ndarray
    raw_out: bytes

    @property
    def is_f16(self) -> bool:
        """True for the f16 cases."""
        return self.kind == KIND_F16


def load_case(path: Path) -> Case:
    """Reads one case file. Raises ValueError if the file is not a complete census file."""
    data = path.read_bytes()
    if len(data) < HDR.size:
        raise ValueError(f"{path}: the file is shorter than the header ({len(data)} bytes)")
    (magic, kind, method, m, k, n, nb, na, nw, no, n_snap, p0, p1, p2, usr) = HDR.unpack_from(data)
    if magic != b"HMXISA1\0":
        raise ValueError(f"{path}: the magic is {magic!r}, not HMXISA1")
    if HDR.size + nb + na + nw + no != len(data):
        raise ValueError(f"{path}: the section sizes do not agree with the file size {len(data)}")
    off = HDR.size
    bias = data[off:off + nb]
    off += nb
    a_raw = data[off:off + na]
    off += na
    w_raw = data[off:off + nw]
    off += nw
    out_raw = data[off:off + no]
    if kind == KIND_F16:
        a = np.frombuffer(a_raw, "<u2").reshape(m, k)
        w = np.frombuffer(w_raw, "<u2").reshape(k, n)
        out = np.frombuffer(out_raw, "<u2").reshape(n_snap, m, n)
    elif kind == KIND_Q8_SRC:
        a = np.frombuffer(a_raw, "<f4").reshape(m, k)  # the f32 activations
        w = np.frombuffer(w_raw, np.uint8)  # q_w then d_w, refer to q8_source
        out = np.frombuffer(out_raw, np.uint8).reshape(1, -1)  # q_x then d_x of the DSP
    else:
        a = np.frombuffer(a_raw, np.uint8)
        w = np.frombuffer(w_raw, np.uint8)
        out = np.frombuffer(out_raw, np.uint8).reshape(n_snap, -1)
    return Case(path.stem, kind, method, m, k, n, n_snap, (p0, p1, p2), usr, bias, a, w, out, out_raw)


def load_run(run_dir: Path) -> dict[str, Case]:
    """Reads all case files of one run directory. O(total file size)."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"the run directory {run_dir} does not exist; run tools/htp-lab/hmxisa/run.sh")
    cases = {}
    for p in sorted(run_dir.glob("*.bin")):
        c = load_case(p)
        cases[c.name] = c
    return cases


def run_core(run_dir: Path) -> str:
    """Returns the simulated core of a run from its stdout (the rev_id line), or '?'."""
    p = run_dir / "stdout.txt"
    if p.exists():
        m = re.search(r"rev_id used in the simulation is \S+ \((\w+)\)", p.read_text(errors="replace"))
        if m:
            return m.group(1)
    return "?"


# ---------------------------------------------------------------------------------------------
# f16 helpers
# ---------------------------------------------------------------------------------------------

def f16_fields(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits f16 bit patterns into sign, integer significand and exponent (the HMX interpretation).

    The value is (-1)^sign * sig * 2^(exp - 10). Subnormals get exp = -14. Exponent code 31 is an
    ordinary exponent (exp = 16): the HMX has no Inf and no NaN while USR bit 20 is clear.
    """
    b = bits.astype(np.int64)
    sign = (b >> 15) & 1
    e = (b >> 10) & 31
    f = b & 1023
    sig = np.where(e == 0, f, f | 1024)
    exp = np.where(e == 0, -14, e - 15)
    return sign, sig, exp


def f16_to_float(bits: np.ndarray) -> np.ndarray:
    """IEEE interpretation of f16 bits as float64."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float64)


def f16_ext_to_float(bits: np.ndarray) -> np.ndarray:
    """The HMX interpretation of f16 bits: exponent code 31 is an ordinary exponent (no Inf, no NaN)."""
    s, sig, e = f16_fields(np.asarray(bits))
    return np.where(s == 1, -1.0, 1.0) * sig.astype(np.float64) * np.exp2(e.astype(np.float64) - 10)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in units of the last place between f16 bit patterns (sign-magnitude order)."""
    def ordinal(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.int64)
        return np.where(x & 0x8000, -(x & 0x7FFF), x & 0x7FFF)
    return np.abs(ordinal(a) - ordinal(b))


def round_int_to_f16(num: int, exp2: int, mode: str = "rne", ext: bool = False, sat: bool = False) -> int:
    """Rounds the exact value num * 2^exp2 to f16 bits.

    mode is rne, rtz, rdn, rup, rna or rto (nearest even, toward zero, down, up, nearest away, odd).
    ext treats exponent code 31 as an ordinary exponent (the HMX output range, maximum 131008).
    sat gives the largest finite value on overflow, else the result is Inf (or the extended maximum).
    A zero result is +0; the caller decides the sign of zero.
    """
    if num == 0:
        return 0
    neg = num < 0
    mag = -num if neg else num
    # value = mag * 2^exp2 = q * 2^(e - 10) with q in [1024, 2048) for normal results
    e = mag.bit_length() - 1 + exp2  # floor(log2(value))
    e_min = -14
    e_q = max(e, e_min) - 10  # the exponent of the last significand bit
    shift = e_q - exp2
    if shift > 0:
        q = mag >> shift
        rem = mag - (q << shift)
        half = 1 << (shift - 1)
        up = False
        if mode == "rne":
            up = rem > half or (rem == half and (q & 1))
        elif mode == "rna":
            up = rem >= half
        elif mode == "rtz":
            up = False
        elif mode == "rdn":
            up = neg and rem != 0
        elif mode == "rup":
            up = (not neg) and rem != 0
        elif mode == "rto":
            if rem != 0:
                q |= 1
        else:
            raise ValueError(f"unknown rounding mode {mode}")
        if up:
            q += 1
    else:
        q = mag << (-shift)
    if q >= 2048:  # the rounding carried into the next binade
        q >>= 1
        e_q += 1
    biased = e_q + 10 + 15 if q >= 1024 else 0
    max_biased = 31 if ext else 30
    if biased > max_biased:
        if ext or sat:
            bits = F16_EXT_MAX_BITS if ext else F16_MAX_BITS
        else:
            bits = 0x7C00
    elif biased == 0:
        bits = q  # subnormal, or zero after the rounding
    else:
        bits = (biased << 10) | (q & 1023)
    return bits | (0x8000 if neg else 0)


# ---------------------------------------------------------------------------------------------
# exact dot products
# ---------------------------------------------------------------------------------------------

def to_scaled_int(bits: np.ndarray) -> np.ndarray:
    """f16 bits to Python ints in units of 2^-24 (exact for all finite values, object array).

    Exponent code 31 is an ordinary exponent here (the HMX interpretation).
    """
    s, sig, e = f16_fields(bits)
    shift = (e + 14).astype(np.int64)  # >= 0
    out = np.empty(bits.shape, dtype=object)
    flat_s, flat_sig, flat_sh = s.ravel(), sig.ravel(), shift.ravel()
    vals = [(-1 if flat_s[i] else 1) * (int(flat_sig[i]) << int(flat_sh[i])) for i in range(flat_s.size)]
    out.ravel()[:] = vals
    return out


def exact_dot(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The exact dot products of f16 matrices, as Python ints in units of 2^-48. O(m k n)."""
    return to_scaled_int(a).dot(to_scaled_int(w))


def round_matrix(num: np.ndarray, exp2: int, mode: str = "rne", ext: bool = False, sat: bool = False) -> np.ndarray:
    """Applies round_int_to_f16 to each element of an object array of ints. O(elements)."""
    flat = [round_int_to_f16(int(v), exp2, mode, ext, sat) for v in num.ravel()]
    return np.array(flat, dtype=np.uint16).reshape(num.shape)


# ---------------------------------------------------------------------------------------------
# the model of the HMX f16 accumulator
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class XfpParams:
    """The parameters of the block floating point accumulator of one Hexagon version.

    group:  products that stage 1 adds after the alignment to their largest exponent (v73: 4,
            v75 to v81: 8). The groups do not cross a 32-channel block.
    width:  the bit width of the stage 2 adder, for the leading zero anticipation (v73: 30 = Q8.22,
            v75 to v81: 31 = Q9.22)
    frac:   the fraction bits of the aligned significands (22 on all versions)
    norm_max: the largest left normalization of the accumulator in one feedback step (2)
    """

    group: int
    width: int
    frac: int = 22
    norm_max: int = 2


XFP_PARAMS = {
    "v73": XfpParams(group=4, width=30),
    "v75": XfpParams(group=8, width=31),
    "v79": XfpParams(group=8, width=31),
    "v81": XfpParams(group=8, width=31),
}

# Cores whose accumulator differs from the phone core of their version (fp_rate 1, Q6.22 adder)
CORE_PARAMS = {
    "v81nd_1": XfpParams(group=1, width=28),
}


def params_for(label: str, core: str) -> XfpParams | None:
    """The model parameters of a core: the core entry, else the version entry."""
    return CORE_PARAMS.get(core, XFP_PARAMS.get(label))


def jam_shift(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Arithmetic right shift by s with the shifted-out bits ORed into the result LSB.

    This is the alignment of the simulator ("hmx_xfp_shift_with inexact"): floor, then set bit 0 if
    the shift discarded a nonzero bit. A nonzero value never becomes zero.
    """
    s = np.clip(s, 0, 62).astype(np.int64)
    fl = x >> s
    sticky = (x - (fl << s)) != 0
    return fl | sticky.astype(np.int64)


def lza(a: np.ndarray, b: np.ndarray, width: int) -> np.ndarray:
    """The leading zero anticipation of a + b in a width-bit two's complement adder.

    This is the indicator of Schmookler and Nowka: the count of leading zeros of f over the bits
    width-2 to 0. It is the true count of redundant sign bits of the sum or one more.
    """
    mask = (1 << width) - 1
    a = a & mask
    b = b & mask
    p = a ^ b
    g = a & b
    z = ~(a | b) & mask
    pn = p >> 1
    gp = (g << 1) & mask
    zp = (z << 1) & mask
    f = (pn & ((g & ~zp) | (z & ~gp))) | (~pn & ((z & ~zp) | (g & ~gp)))
    f &= (1 << (width - 1)) - 1
    n = np.full(a.shape, width - 1, dtype=np.int64)
    for bit in range(width - 1):
        n = np.where((f >> bit) & 1, width - 2 - bit, n)
    return n


def xfp_model(a: np.ndarray, w: np.ndarray, p: XfpParams, skip_zero_groups: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs the accumulator model of the simulator on f16 inputs A [m][k] and W [k][n].

    Stage 1: each group of p.group products is aligned to its largest exponent with p.frac fraction
    bits (jam_shift) and added exactly. Stage 2: the accumulator and the group sum are aligned to the
    larger exponent (jam_shift) and added. Feedback: bit 0 of the sum is jammed into bit 1, and the
    sum is normalized by lza - 1 bits (a left shift is exact, a right shift jams). A group with only
    zero products is skipped. The normalization step is min(lza - 1, p.norm_max) bits, thus a sum
    that cancels loses its low bits at the next alignment. The jam into bit 1 changes a sum of 1 to 2,
    thus a tiny term that survives only as its jam bit doubles. The product of significands a, b is a*b at exponent ea + eb - 2 in
    Q5.18, thus the value of an accumulator (e, s) is s * 2^(e - 22).

    Returns the final accumulator: exponent (m, n), significand (m, n) and a zero flag (m, n).
    O(m n k).
    """
    m, k = a.shape
    n = w.shape[1]
    sa, ma, ea = f16_fields(a)
    sw, mw, ew = f16_fields(w)
    acc_e = np.zeros((m, n), dtype=np.int64)
    acc_s = np.zeros((m, n), dtype=np.int64)
    acc_z = np.ones((m, n), dtype=bool)
    neg_inf = -(1 << 20)
    for k0 in range(0, k, p.group):
        ks = slice(k0, k0 + p.group)
        sig = ma[:, None, ks] * mw[ks, :].T[None, :, :]
        sgn = sa[:, None, ks] ^ sw[ks, :].T[None, :, :]
        exp = ea[:, None, ks] + ew[ks, :].T[None, :, :] - 2
        nz = sig != 0
        sig22 = np.where(sgn == 1, -sig, sig) << (p.frac - 18)
        g_e = np.max(np.where(nz, exp, neg_inf), axis=2)
        g_z = ~nz.any(axis=2)
        aligned = np.where(nz, jam_shift(sig22, np.where(nz, g_e[..., None] - exp, 0)), 0)
        g_s = aligned.sum(axis=2)
        # stage 2
        e = np.where(acc_z, g_e, np.where(g_z, acc_e, np.maximum(acc_e, g_e)))
        a_al = np.where(acc_z, 0, jam_shift(acc_s, e - acc_e))
        b_al = np.where(g_z, 0, jam_shift(g_s, e - g_e))
        s = a_al + b_al
        d = np.minimum(lza(a_al, b_al, p.width) - 1, p.norm_max)
        s1 = ((s >> 1) | (s & 1)) << 1
        s_new = np.where(d >= 0, s1 << np.clip(d, 0, 62), jam_shift(s1, -d))
        e_new = e - d
        z_new = acc_z & g_z  # a sum that cancels to zero keeps its exponent
        upd = ~g_z if skip_zero_groups else np.ones_like(g_z)
        acc_e = np.where(upd, np.where(z_new, 0, e_new), acc_e)
        acc_s = np.where(upd, np.where(z_new, 0, s_new), acc_s)
        acc_z = np.where(upd, z_new, acc_z)
    return acc_e, acc_s, acc_z


CVT_BITS = 15  # the significant bits of the accumulator at the output conversion


def f16_int_parts(bits: int) -> tuple[int, int]:
    """One f16 value (HMX interpretation) as (signed integer significand, exponent of its LSB)."""
    s, sig, e = f16_fields(np.array([bits], dtype=np.uint16))
    return (-1 if s[0] else 1) * int(sig[0]), int(e[0]) - 10


def jam_to_bits(x: int, e: int, bits: int) -> tuple[int, int]:
    """Keeps the top `bits` significant bits of x * 2^e: floor shift with the lost bits jammed."""
    if x == 0:
        return 0, e
    s = abs(x).bit_length() - bits
    if s <= 0:
        return x, e
    q = x >> s
    if x - (q << s):
        q |= 1
    return q, e + s


def cvt_f16(acc_s: int, acc_e: int, lo: int, hi: int, frac: int = 22, mode: str = "rne") -> int:
    """The output conversion of the HMX store ":after.hf" for one element.

    The accumulator value acc_s * 2^(acc_e - frac) keeps CVT_BITS significant bits (jam), is
    multiplied exactly by the scale lo (the low half of the bias word), the bias hi (the high half)
    is added exactly, and the sum is rounded to f16. Exponent code 31 is an ordinary exponent, and
    an overflow gives the largest magnitude 0x7fff.
    """
    t, te = jam_to_bits(acc_s, acc_e - frac, CVT_BITS)
    ls, le = f16_int_parts(lo)
    hs, he = f16_int_parts(hi)
    u, ue = t * ls, te + le
    if hs:
        base = min(ue, he)
        u = (u << (ue - base)) + (hs << (he - base))
        ue = base
    return round_int_to_f16(u, ue, mode, ext=True)


def bias_words(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """The scale (lo) and bias (hi) of each of the 32 columns of an f16 bias area."""
    words = np.frombuffer(case.bias[:128], "<u4")
    return (words & 0xFFFF).astype(np.int64), (words >> 16).astype(np.int64)


def acc_to_f16(acc_e: np.ndarray, acc_s: np.ndarray, acc_z: np.ndarray, lo: np.ndarray | None = None,
               hi: np.ndarray | None = None, frac: int = 22, mode: str = "rne") -> np.ndarray:
    """Converts the model accumulator (m, n) to f16 bits with the column words lo and hi (cvt_f16).

    Without lo and hi the conversion is the identity scale of the kernels (lo = 1.0, hi = 0).
    A zero accumulator (zero flag) gives +0 * lo + hi. O(m n).
    """
    m, n = acc_e.shape
    lo = np.full(32, 0x3C00) if lo is None else lo
    hi = np.zeros(32, dtype=np.int64) if hi is None else hi
    out = np.zeros(acc_e.shape, dtype=np.uint16)
    for i in range(m):
        for j in range(n):
            s = 0 if acc_z[i, j] else int(acc_s[i, j])
            out[i, j] = cvt_f16(s, int(acc_e[i, j]), int(lo[j % 32]), int(hi[j % 32]), frac, mode)
    return out


# ---------------------------------------------------------------------------------------------
# the int path
# ---------------------------------------------------------------------------------------------

def i8_act_logical(case: Case) -> np.ndarray:
    """The u8 activation [64][k]: tile t holds 64 rows x 32 channels row major (byte r * 32 + k)."""
    n_kt = case.a.size // 2048
    return case.a.reshape(n_kt, 64, 32).transpose(1, 0, 2).reshape(64, n_kt * 32).astype(np.int64)


def i8_wgt_logical(case: Case) -> np.ndarray:
    """The s8 weight [k][32]: tile t holds byte (k / 4) * 128 + c * 4 + k % 4."""
    n_kt = case.w.size // 1024
    raw = case.w.view(np.int8).reshape(n_kt, 8, 32, 4)
    return raw.transpose(0, 1, 3, 2).reshape(n_kt * 32, 32).astype(np.int64)


def i4_wgt_logical(case: Case) -> np.ndarray:
    """The s4 weight [k][32]: tile t holds channel k of column c in byte (k / 8) * 128 + c * 4 + k % 4,
    low nibble for k % 8 < 4 and high nibble else, two's complement."""
    n_kt = case.w.size // 512
    raw = case.w.reshape(n_kt, 4, 32, 4).astype(np.int64)  # [t][k/8][c][k%4]
    lo = raw & 15
    hi = raw >> 4
    nib = np.concatenate([lo, hi], axis=3)  # [t][k/8][c][k%8]
    nib = np.where(nib >= 8, nib - 16, nib)
    return nib.transpose(0, 1, 3, 2).reshape(n_kt * 32, 32)


def planes_to_i32(case: Case, read: int = 0) -> np.ndarray:
    """Joins the four u8 planes of one int32 read (plane p is byte p, row major r * 32 + c)."""
    p = case.out[4 * read:4 * read + 4].astype(np.int64)
    v = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24)
    return np.where(v >= 1 << 31, v - (1 << 32), v).reshape(64, 32)


def int_reads(case: Case) -> int:
    """The number of int32 reads in an int case (one per 32-k block for M_I8_BLOCK)."""
    return case.n_snap // 4


def int_exact(case: Case, read: int = 0) -> np.ndarray:
    """The exact int product that int32 read `read` of an int case must hold (int64 is exact here)."""
    a = i8_act_logical(case)
    w = i4_wgt_logical(case) if case.kind == KIND_I4 else i8_wgt_logical(case)
    if case.method == M_I8_REPEAT:  # tile pair 0 param0 times, tile pair 1 param1 times
        n0, n1, _ = case.param
        return n0 * (a[:, :32] @ w[:32]) + n1 * (a[:, 32:64] @ w[32:64])
    if case.method == M_I8_BLOCK:  # read t holds k tile t alone
        ks = slice(32 * read, 32 * read + 32)
        return a[:, ks] @ w[ks]
    return a @ w


def wrap32(x: np.ndarray) -> np.ndarray:
    """Two's complement wrap of int64 values to 32 bits."""
    return ((x + (1 << 31)) % (1 << 32)) - (1 << 31)


# ---------------------------------------------------------------------------------------------
# runs and families
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Run:
    """The census of one core: label, simulated core, directory and cases."""

    label: str
    core: str
    path: Path
    cases: dict[str, Case]


def family(name: str) -> str:
    """The family of a case: the first word of the name (bias_v1 and usr are families of their own)."""
    if name.startswith("bias_v1"):
        return "bias_v1"
    if name.startswith(("i8_sat", "i8_badrange", "i8_overrange")):
        return name.split("_")[0] + "_" + name.split("_")[1]
    return name.split("_")[0]


def discover_extra(main: list[Run]) -> list[Run]:
    """The other census directories of out-hmxisa (the --extra cores of run.sh)."""
    used = {r.path.resolve() for r in main}
    extra = []
    for d in sorted(OUT.glob("hmxisa-*")):
        if d.resolve() in used or not d.is_dir() or not any(d.glob("*.bin")):
            continue
        m = re.match(r"hmxisa-(v\d\d)-(\w+)$", d.name)
        arch = m.group(1) if m else "v79"
        extra.append(Run(arch, run_core(d), d, load_run(d)))
    return extra


def elem_diff(a: Case, b: Case) -> tuple[int, int]:
    """Number of output elements that differ, and the largest f16 ulp distance (0 for int cases)."""
    if a.is_f16:
        ne = a.out != b.out
        return int(ne.sum()), int(ulp_distance(a.out, b.out).max()) if ne.any() else 0
    return int((a.out != b.out).sum()), 0


def fmt_groups(groups: list[list[str]]) -> str:
    """'v73 | v75=v79=v81' style text of the version groups of one case."""
    return " | ".join("=".join(g) for g in groups)


def version_groups(runs: list[Run], name: str) -> list[list[str]]:
    """Groups the runs by identical output bytes of one case."""
    groups: dict[bytes, list[str]] = {}
    for r in runs:
        c = r.cases.get(name)
        key = c.raw_out if c is not None else b"<missing>"
        groups.setdefault(key, []).append(r.label if c is not None else r.label + "(missing)")
    return list(groups.values())


# ---------------------------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------------------------

_exact_cache: dict[tuple[str, int, int], np.ndarray] = {}


def snap_channels(case: Case, snap: int) -> slice:
    """The channels that snapshot `snap` of an f16 case accumulates."""
    if case.method == M_RETAIN:
        return slice(0, min(snap + 1, case.k // 32) * 32)
    if case.method == M_BLOCK:
        return slice(32 * snap, 32 * snap + 32)
    return slice(0, case.k)


def exact_acc(case: Case, ks: slice | None = None) -> np.ndarray:
    """The exact dot products (units of 2^-48, object array) over the channels ks. Cached."""
    ks = slice(0, case.k) if ks is None else ks
    key = (case.name, ks.start, ks.stop)
    if key not in _exact_cache:
        _exact_cache[key] = exact_dot(case.a[:, ks].astype(np.uint16), case.w[ks, :].astype(np.uint16))
    return _exact_cache[key]


def exact_f16(case: Case, snap: int = -1, mode: str = "rne") -> np.ndarray:
    """RNE (or mode) of the exact result, with the bias words, in the HMX output range."""
    snap = case.n_snap - 1 if snap < 0 else snap
    acc = exact_acc(case, snap_channels(case, snap))
    lo, hi = bias_words(case)
    out = np.zeros(acc.shape, dtype=np.uint16)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            ls, le = f16_int_parts(int(lo[j % 32]))
            hs, he = f16_int_parts(int(hi[j % 32]))
            u, ue = int(acc[i, j]) * ls, le - 48
            if hs:
                base = min(ue, he)
                u = (u << (ue - base)) + (hs << (he - base))
                ue = base
            out[i, j] = round_int_to_f16(u, ue, mode, ext=True)
    return out


def model_f16(case: Case, p: XfpParams, snap: int = -1) -> np.ndarray:
    """The model output (xfp_model and cvt_f16) of one snapshot of an f16 case."""
    snap = case.n_snap - 1 if snap < 0 else snap
    ks = snap_channels(case, snap)
    e, s, z = xfp_model(case.a[:, ks].astype(np.uint16), case.w[ks, :].astype(np.uint16), p)
    lo, hi = bias_words(case)
    return acc_to_f16(e, s, z, lo, hi)


USR_STICKY = 0x3F  # OVF and the five sticky IEEE flags: status bits, not control bits


def default_usr(cases: dict[str, Case]) -> int:
    """The USR control bits of the cases that do not set USR: the most common value of the run."""
    counts = Counter(c.usr & ~USR_STICKY for c in cases.values())
    return counts.most_common(1)[0][0] if counts else 0


def modeled(case: Case, usr0: int) -> bool:
    """True for the f16 cases that the model covers: control word zero and the default USR."""
    return case.is_f16 and not any(case.bias[128:]) and (case.usr & ~USR_STICKY) == usr0


def signed_ulp(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """got - ref in ulps along the f16 number line (positive: got is larger)."""
    def ordinal(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.int64)
        return np.where(x & 0x8000, -(x & 0x7FFF), x & 0x7FFF)
    return ordinal(got) - ordinal(ref)


# ---------------------------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------------------------

def section(title: str) -> None:
    """Prints a section title."""
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def report_identity(runs: list[Run], names: list[str]) -> dict[str, list[list[str]]]:
    """Section 1: the cases that are byte-identical on all versions, and the differences."""
    section("1. Cross-version identity of the output bytes")
    groups = {n: version_groups(runs, n) for n in names}
    fam: dict[str, list[str]] = {}
    for n in names:
        fam.setdefault(family(n), []).append(n)
    print(f"{'family':14s} {'cases':>5s} {'identical':>9s}  version groups of the other cases")
    for f, ns in fam.items():
        same = [n for n in ns if len(groups[n]) == 1]
        others = Counter(fmt_groups(groups[n]) for n in ns if len(groups[n]) > 1)
        print(f"{f:14s} {len(ns):5d} {len(same):9d}  " + "; ".join(f"{g} ({c})" for g, c in others.items()))
    diff = [n for n in names if len(groups[n]) > 1]
    if diff:
        print()
        print("cases that differ, element count and largest ulp distance against the last group:")
        for n in diff:
            ref_label = groups[n][-1][0]
            ref = next(r for r in runs if r.label == ref_label).cases[n]
            parts = []
            for g in groups[n][:-1]:
                c = next(r for r in runs if r.label == g[0]).cases[n]
                nd, mu = elem_diff(c, ref)
                parts.append(f"{'='.join(g)}: {nd}/{c.out.size} elements, max {mu} ulp")
            print(f"  {n:24s} {fmt_groups(groups[n]):22s} " + "; ".join(parts))
    return groups


def report_model(runs: list[Run], names: list[str]) -> None:
    """Section 2: the model of each version against every modeled f16 output."""
    section("2. The accumulator model (xfp_model + cvt_f16) against the simulator, bit for bit")
    for r in runs:
        p = params_for(r.label, r.core)
        usr0 = default_usr(r.cases)
        tot = bad = n_cases = 0
        worst = []
        for n in names:
            c = r.cases.get(n)
            if c is None or not modeled(c, usr0) or p is None:
                continue
            n_cases += 1
            for t in range(c.n_snap):
                pred = model_f16(c, p, t)
                nb = int((pred != c.out[t]).sum())
                tot += pred.size
                bad += nb
                if nb:
                    worst.append(f"{n}[{t}]:{nb}")
        print(f"  {r.label} ({r.core}, {p}): {n_cases} cases, {tot - bad} of {tot} elements equal"
              + (f"; mismatches in {', '.join(worst[:8])}" if worst else ""))


def report_accuracy(runs: list[Run], names: list[str]) -> None:
    """Section 3: the output against the exact result rounded once (RNE, HMX output range)."""
    section("3. Accuracy: HMX output against RNE(exact dot product x lo + hi), in ulps of f16")
    print(f"  {'case':24s} " + " ".join(f"{r.label + ' exact/n  max  mean':>30s}" for r in runs))
    for n in names:
        c0 = runs[0].cases.get(n)
        if c0 is None or not c0.is_f16 or family(n) in ("bias_v1", "usr", "pass", "special", "zero"):
            continue
        if not modeled(c0, default_usr(runs[0].cases)):
            continue
        ref = exact_f16(c0)
        cols = []
        for r in runs:
            c = r.cases[n]
            d = signed_ulp(c.out[-1], ref)
            cols.append(f"{int((d == 0).sum()):6d}/{d.size:<6d} {int(np.abs(d).max()):6d} {d.mean():+8.4f}")
        print(f"  {n:24s} " + " ".join(f"{x:>30s}" for x in cols))


def notable_specials(case: Case) -> list[str]:
    """Lists the entries of a special table where the HMX differs from IEEE f16 arithmetic."""
    lines = []
    a0 = case.a[:, 0].astype(np.uint16)
    add = case.name == "special_add"
    w0 = case.w[1 if add else 0, :].astype(np.uint16)
    fa = f16_to_float(a0)
    fw = f16_to_float(w0)
    with np.errstate(all="ignore"):
        ieee = (fa[:, None] + fw[None, :]) if add else (fa[:, None] * fw[None, :])
        ieee16 = ieee.astype(np.float16).view(np.uint16)
    got = case.out[0]
    op = "+" if add else "x"
    shown = set()
    for i in range(32):
        for j in range(32):
            g, e = int(got[i, j]), int(ieee16[i, j])
            both_nan = np.isnan(f16_to_float(np.array([g]))[0]) and np.isnan(f16_to_float(np.array([e]))[0])
            if g == e or both_nan:
                continue
            key = (int(a0[i]), int(w0[j]))
            if key in shown or (int(w0[j]), int(a0[i])) in shown:
                continue
            shown.add(key)
            lines.append(f"{a0[i]:#06x} {op} {w0[j]:#06x} = {g:#06x} (IEEE {e:#06x}, HMX value {f16_ext_to_float(np.array([g]))[0]:.6g})")
    return lines


def report_characterization(runs: list[Run], names: list[str]) -> None:
    """Section 4: the properties of each family, from the first run of each version group."""
    section("4. Characterization")
    r = next(x for x in runs if x.label == "v79") if any(x.label == "v79" for x in runs) else runs[0]
    cs = r.cases

    print("4.1 Inputs, specials and signed zero (all versions give the same bytes for these cases):")
    if "pass_s0" in cs:
        c = cs["pass_s0"]
        a = c.a.ravel()
        same = int((c.out[0].ravel() == a).sum())
        print(f"  pass_s0: {same} of {a.size} finite f16 values (subnormals included) pass x1.0 unchanged")
    if "pass_sp12" in cs:
        c = cs["pass_sp12"]
        o = c.out[0].ravel()
        e31 = int(((o >> 10) & 31 == 31).sum())
        sat = int(((o & 0x7FFF) == 0x7FFF).sum())
        ex = exact_f16(c).ravel()
        print(f"  pass_sp12 (x 2^12): {e31} outputs have exponent code 31 (the HMX value up to 131008; IEEE reads them"
              f" as Inf/NaN), {sat} of them saturate at 0x7fff/0xffff; {int((o == ex).sum())} of {o.size} equal"
              " RNE in the extended range")
    if "pass_sm12" in cs:
        c = cs["pass_sm12"]
        o = c.out[0].ravel()
        res = {m: int((exact_f16(c, mode=m).ravel() == o).sum()) for m in ("rne", "rtz", "rna", "rto")}
        negz = int((o == 0x8000).sum())
        print(f"  pass_sm12 (x 2^-12, subnormal outputs): matches of the rounding modes {res} of {o.size};"
              f" {negz} outputs are -0 (negative underflow)")
    for n in ("special_mul", "special_add"):
        if n in cs:
            lines = notable_specials(cs[n])
            print(f"  {n}: {len(lines)} distinct entries differ from IEEE, for example:")
            for ln in lines[:14]:
                print("    " + ln)
    if "zero_sign" in cs:
        o = cs["zero_sign"].out[0]
        print(f"  zero_sign: {int((o == 0x8000).sum())} outputs are -0 and {int((o == 0).sum())} are +0 of {o.size}"
              " (all products -0 give +0: the zero result has no sign)")

    print()
    print("4.2 Rounding at the store and the sticky width (round_*: x + half ulp + tiny at depth d):")
    for n in ("round_near", "round_far", "round_k64"):
        if n not in cs:
            continue
        c = cs[n]
        got = c.out[0]
        ref = exact_f16(c)
        mdl = {m: exact_f16(c, mode=m) for m in ("rtz",)}
        tiles = ("x+h+2^-d", "x+h-2^-d", "x+2^-d", "x-2^-d")
        parts = []
        for t in range(4):
            ok_d = [d for d in range(1, 32) if (got[:, t * 32 + d] == ref[:, t * 32 + d]).all()]
            bad_d = [d for d in range(1, 32) if d not in ok_d]
            tie_ok = bool((got[:, t * 32] == ref[:, t * 32]).all())
            first_bad = bad_d[0] if bad_d else None
            parts.append(f"{tiles[t]}: RNE-exact for d={ok_d[0] if ok_d else '-'}..{ok_d[-1] if ok_d else '-'}"
                         f"{'' if first_bad is None else f', first miss d={first_bad}'}; tie/plain column exact={tie_ok}")
        print(f"  {n}: " + " | ".join(parts))
        miss = got != ref
        if miss.any():
            d = signed_ulp(got, ref)[miss]
            print(f"    misses: {int(miss.sum())} elements, error {Counter(d.tolist()).most_common(4)} ulps")

    print()
    print("4.3 Cancellation (cancel_*: +big, tiny, -big; exact result = the tiny product at depth d below big):")
    for n in [x for x in names if x.startswith("cancel")]:
        c = cs[n]
        got = c.out[-1].astype(np.int64)
        ref = exact_f16(c).astype(np.int64)
        vg = f16_ext_to_float(got)
        vr = f16_ext_to_float(ref)
        cls = Counter()
        for g, e in zip(vg.ravel(), vr.ravel()):
            if g == e:
                cls["exact"] += 1
            elif e != 0 and g == 2 * e:
                cls["2x exact"] += 1
            elif g == 0:
                cls["zero"] += 1
            else:
                cls["other"] += 1
        print(f"  {n:20s} {dict(cls)}")

    print()
    print("4.4 Growth (growth_*: the tiny product before, at or after a partial sum peak of (K/2-2) 2^30):")
    for n in [x for x in names if x.startswith("growth")]:
        c = cs[n]
        got = c.out[0]
        ref = exact_f16(c)
        per_pos = []
        for q, lbl in enumerate(("first", "peak", "last")):
            rows = [i for i in range(32) if i % 3 == q]
            per_pos.append(f"{lbl}: {int((got[rows] == ref[rows]).sum())}/{len(rows) * 32}")
        print(f"  {n:14s} exact: " + ", ".join(per_pos))

    print()
    print("4.5 Accumulation order and instruction boundaries (meth_*, cancel_k128_*):")
    for fam_prefix in ("meth_wide", "cancel_k128"):
        members = [x for x in names if x.startswith(fam_prefix) and not x.endswith(("retain", "noclear"))]
        if not members:
            continue
        for rr in runs:
            outs = {x: rr.cases[x].out[-1] for x in members if x in rr.cases}
            base = outs.get(members[0])
            same = [x for x in members if x in outs and (outs[x] == base).all()]
            print(f"  {rr.label} {fam_prefix}: {len(same)} of {len(members)} methods give identical outputs ({', '.join(members)})")
    for n in ("meth_wide_retain", "meth_wide_noclear", "cancel_k128_retain"):
        if n in cs and n.replace("retain", "deep").replace("noclear", "deep") in cs:
            base = cs[n.replace("retain", "deep").replace("noclear", "deep")].out[0]
            print(f"  {n}: final output equals the deep result: {bool((cs[n].out[-1] == base).all())}")

    print()
    print("4.6 The bias area (out = RNE(jam15(acc) x lo + hi), lo and hi the halves of word c of the first 128 bytes):")
    if "bias_qk128" in cs and "bias_lo_qk128" in cs:
        a = f16_ext_to_float(cs["bias_qk128"].out[0])
        b = f16_ext_to_float(cs["bias_lo_qk128"].out[0])
        lo = f16_ext_to_float(np.array([0x2DA8]))[0]
        print(f"  bias_qk128 (flash-attn-ops.c: lo = hi = scale {lo:.6f}) minus bias_lo_qk128 (hi = 0):"
              f" mean {np.mean(a - b):+.6f}, min {np.min(a - b):+.6f}, max {np.max(a - b):+.6f}")
    usr0 = default_usr(cs)
    ctrl = [x for x in names if x.startswith("bias_v1")]
    if ctrl and "bias_id" in cs:
        base = cs["bias_id"].out[0]
        eff = []
        for x in ctrl:
            o = cs[x].out[0]
            if (o == base).all():
                continue
            if (o == (base ^ 0x8000)).all():
                what = "negates"
            elif (o == np.where(base & 0x8000, 0, base)).all():
                what = "ReLU"
            else:
                what = f"changes {int((o != base).sum())} elements"
            eff.append(f"bit {int(x[-2:])}: {what}")
        print(f"  the second 128 bytes (control word, zero in the kernels): " + "; ".join(eff) + "; other bits: no effect")

    print()
    print("4.7 USR during the HMX instructions (the kernels do not write USR):")
    for x in [y for y in names if y.startswith("usr")]:
        base_name = "round_near" if x.endswith("round") else "special_mul"
        if base_name not in cs:
            continue
        o, b = cs[x].out[0], cs[base_name].out[0]
        print(f"  {x:22s} usr=0x{cs[x].usr:08x}: {'no effect' if (o == b).all() else f'{int((o != b).sum())} elements change'}")

    print()
    print("4.8 Drift of the accumulator exponent (drift_*: K-32 channels of zero-sum groups, then 32 random channels):")
    for n in [x for x in names if x.startswith("drift")]:
        parts = []
        for rr in runs:
            c = rr.cases[n]
            ref = exact_f16(c)
            parts.append(f"{rr.label} {int((c.out[0] == ref).sum())}/{ref.size} RNE-exact,"
                         f" max {int(ulp_distance(c.out[0], ref).max())} ulp")
        print(f"  {n:12s} " + "; ".join(parts))


def report_int(runs: list[Run], names: list[str]) -> None:
    """Section 5: the int path, decoded with the documented layouts."""
    section("5. The int path (u8 activation x s8/s4 weight, int32 accumulator, read as four u8 planes)")
    for r in runs:
        rows = []
        for n in names:
            c = r.cases.get(n)
            if c is None or c.kind not in (KIND_I8, KIND_I4):
                continue
            n_exact = n_wrap = total = 0
            for t in range(int_reads(c)):
                got = planes_to_i32(c, t)
                ex = int_exact(c, t)
                n_exact += int((got == ex).sum())
                n_wrap += int((got == wrap32(ex)).sum())
                total += got.size
            rows.append(f"{n}: exact {n_exact}/{total}" + (f", wrap32 {n_wrap}" if n_wrap != n_exact else ""))
        print(f"  {r.label}: " + "; ".join(rows))


# ---------------------------------------------------------------------------------------------
# the CPU oracle: the scalar reference code of ggml, executed with IEEE arithmetic
# ---------------------------------------------------------------------------------------------

def oracle_f16(a16: np.ndarray, w16: np.ndarray) -> np.ndarray:
    """The CPU reference of an F16 matmul (the generic ggml_vec_dot_f16).

    Each product f32(a) * f32(w) is exact in f32, the products are added in double (ggml_float) in
    the order of k, and the sum is rounded to f32. The activation is already f16 here (the CPU
    converts f32 to f16 with round to nearest even first). O(m k n) memory and time.
    """
    fa = f16_to_float(a16).astype(np.float32)
    fw = f16_to_float(w16).astype(np.float32)
    prod = (fa[:, :, None] * fw[None, :, :]).astype(np.float64)
    return np.cumsum(prod, axis=1)[:, -1, :].astype(np.float32)


def acc_to_float64(acc: np.ndarray) -> np.ndarray:
    """Exact dot products in units of 2^-48 (object array of ints) as float64, one rounding each."""
    return np.array([math.ldexp(float(v), -48) for v in acc.ravel()], dtype=np.float64).reshape(acc.shape)


def quantize_q8_0_ref(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """quantize_row_q8_0_ref of ggml along the last axis (f32 IEEE arithmetic).

    d = amax / 127 (f32 division), id = 1 / d (f32), the scale is fp16(d) with round to nearest
    even, and q = roundf(x * id) with ties away from zero. Returns q (int8) and the f16 bits of d.
    """
    xb = x.astype(np.float32).reshape(*x.shape[:-1], -1, 32)
    amax = np.max(np.abs(xb), axis=-1)
    d = (amax / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    v = (xb * inv[..., None]).astype(np.float32).astype(np.float64)
    q = np.where(v >= 0, np.floor(v + 0.5), -np.floor(-v + 0.5)).astype(np.int8)
    return q.reshape(x.shape), d.astype(np.float16).view(np.uint16)


def q8_scaled_sum(block_sums: np.ndarray, dx16: np.ndarray, dw16: np.ndarray) -> np.ndarray:
    """The f32 block loop of the generic ggml_vec_dot_q8_0_q8_0: sumf += sumi * (d_x * d_w).

    block_sums [t][m][n] are the block sums as f32 values (exact integers for the CPU), dx16 [m][t]
    and dw16 [t][n] the f16 scales. Every operation is one f32 rounding, in the order of the blocks.
    """
    dx = f16_to_float(dx16).astype(np.float32)
    dw = f16_to_float(dw16).astype(np.float32)
    sumf = np.zeros(block_sums.shape[1:], dtype=np.float32)
    for t in range(block_sums.shape[0]):
        scale = (dx[:, t][:, None] * dw[t][None, :]).astype(np.float32)
        sumf = (sumf + (block_sums[t].astype(np.float32) * scale).astype(np.float32)).astype(np.float32)
    return sumf


@dataclasses.dataclass
class Q8Source:
    """The source data of the Q8_0 cases and its reference quantization."""

    x: np.ndarray    # f32 activations [m][k]
    qw: np.ndarray   # int8 weight quants [k][n]
    dw: np.ndarray   # f16 bits of the weight scales [k/32][n]
    qx: np.ndarray   # int8 activation quants of the reference [m][k]
    dx: np.ndarray   # f16 bits of the activation scales [m][k/32]
    dsp_ok: bool     # the quants that the DSP computed equal the reference


def q8_source(case: Case) -> Q8Source:
    """Decodes a q8src case and quantizes its activations with quantize_q8_0_ref."""
    m, k, n = case.m, case.k, case.n
    qw = case.w[:k * n].view(np.int8).reshape(k, n)
    dw = case.w[k * n:].view("<u2").reshape(k // 32, n)
    raw = case.out[0]
    qx_dsp = raw[:m * k].view(np.int8).reshape(m, k)
    dx_dsp = raw[m * k:].view("<u2").reshape(m, k // 32)
    x = np.array(case.a, dtype=np.float32)
    qx, dx = quantize_q8_0_ref(x)
    # the weight quants come from the same function on the f32 weights, which the file does not hold
    ok = bool((qx == qx_dsp).all() and (dx == dx_dsp).all())
    return Q8Source(x, qw, dw, qx, dx, ok)


def err_stats(p: np.ndarray, ref: np.ndarray) -> tuple[float, float, float, int]:
    """(share of f32-identical results, max and mean of |p - ref| / rms(ref), count of non-finite p).

    The statistics cover the outputs where p is finite: an HMX output above 65504 is an exponent-31
    pattern that the IEEE conversion of the kernel reads as Inf or NaN, and the count reports them.
    """
    p64 = np.asarray(p, dtype=np.float64)
    r64 = np.asarray(ref, dtype=np.float64)
    fin = np.isfinite(p64)
    with np.errstate(invalid="ignore", over="ignore"):
        p32 = p64.astype(np.float32)
        r32 = r64.astype(np.float32)
    same = float((p32.view(np.uint32) == r32.view(np.uint32)).mean())
    rms = float(np.sqrt(np.mean(np.square(r64)))) or 1.0
    d = np.abs(p64[fin] - r64[fin]) / rms
    return same, float(d.max()) if d.size else 0.0, float(d.mean()) if d.size else 0.0, int((~fin).sum())


def fmt_err(s: tuple[float, float, float, int]) -> str:
    """Text of err_stats."""
    return f"same {100 * s[0]:5.1f}%  max {s[1]:.2e}  mean {s[2]:.2e}" + (f"  non-finite {s[3]}" if s[3] else "")


def report_oracle(runs: list[Run], names: list[str]) -> dict[str, dict[str, tuple]]:
    """Section 7: the HMX paths against the CPU oracle (the scalar reference of ggml) and the exact dot.

    Returns, per version, the error of each path against the oracle for the ranking.
    """
    section("7. The HMX paths against the CPU oracle (scalar ggml reference, IEEE) and against the exact dot")
    print("  Errors are |path - reference| / rms(reference) over all outputs of the case; 'same' is the share of")
    print("  outputs whose f32 bits equal the reference. The HMX f16 paths store f16, the oracle stores f32.")
    ranking: dict[str, dict[str, tuple]] = {r.label: {} for r in runs}

    print()
    print("7.1 F16 weights: HMX f16 (deep MAC, store .hf) against the oracle (f32 products summed in double)")
    f16_cases = [n for n in names if n.startswith(("f16w", "rand_"))]
    for n in f16_cases:
        c0 = runs[0].cases.get(n)
        if c0 is None or not c0.is_f16:
            continue
        a16, w16 = c0.a.astype(np.uint16), c0.w.astype(np.uint16)
        orc = oracle_f16(a16, w16)
        ex = acc_to_float64(exact_acc(c0))
        parts = [f"oracle vs exact: {fmt_err(err_stats(orc, ex))}"]
        with np.errstate(over="ignore"):
            ideal_bits = orc.astype(np.float16).view(np.uint16)  # Inf where the oracle is above 65504
        for r in runs:
            out = r.cases[n].out[-1]
            hmx = f16_to_float(out)
            s = err_stats(hmx, orc)
            parts.append(f"{r.label}: {fmt_err(s)}, = f16(oracle) {100 * float((out == ideal_bits).mean()):5.1f}%")
            if n.startswith("f16w"):
                ranking[r.label].setdefault("F16 weights: HMX f16 deep (today)", []).append(s)
        print(f"  {n:18s} " + " | ".join(parts))
        if n.startswith("f16w"):
            ideal = f16_to_float(orc.astype(np.float16).view(np.uint16))
            print(f"  {'':18s} the best possible f16 output, f16(oracle), against the oracle: {fmt_err(err_stats(ideal, orc))}")

    print()
    print("7.2 Q8_0 weights: the three HMX paths on the same data (q8src_k<K>)")
    srcs = [n for n in names if n.startswith("q8src")]
    for sn in srcs:
        k = int(sn.split("_k")[1])
        src = q8_source(runs[0].cases[sn])
        m, n_ = src.x.shape[0], src.qw.shape[1]
        n_kt = k // 32
        blocks = np.stack([src.qx[:, 32 * t:32 * t + 32].astype(np.int64) @ src.qw[32 * t:32 * t + 32].astype(np.int64)
                           for t in range(n_kt)])
        orc = q8_scaled_sum(blocks, src.dx, src.dw)
        dq = src.qw.astype(np.float64) * np.repeat(f16_to_float(src.dw), 32, axis=0)
        exact = np.array([[math.fsum(src.x[i].astype(np.float64) * dq[:, j]) for j in range(n_)] for i in range(m)])
        same_dsp = all(q8_source(r.cases[sn]).dsp_ok for r in runs)
        print(f"  {sn}: the DSP quants of the activations equal quantize_row_q8_0_ref on all versions: {same_dsp};"
              f" oracle vs exact (f32 x times dequantized Q8_0): {fmt_err(err_stats(orc, exact))}")
        for r in runs:
            ca, cb, cc = r.cases.get(f"q8a_k{k}"), r.cases.get(f"q8b_k{k}"), r.cases.get(f"q8c_k{k}")
            res = {}
            if ca is not None:
                res["A f16 deep (today)"] = f16_to_float(ca.out[-1]).astype(np.float32)
            if cb is not None:
                colsum = np.stack([src.qw[32 * t:32 * t + 32].astype(np.int64).sum(axis=0) for t in range(n_kt)])
                sumi = np.stack([planes_to_i32(cb, t) - 128 * colsum[t][None, :] for t in range(n_kt)])
                exact_ints = bool((sumi == blocks).all())
                res["B int8 per block"] = q8_scaled_sum(sumi, src.dx, src.dw)
            if cc is not None:
                lo, _ = bias_words(cc)  # the output scale of the store (2^-8), a power of two
                lo_f = f16_ext_to_float(lo.astype(np.uint16))[None, :]
                bsum = np.stack([f16_ext_to_float(cc.out[t]) / lo_f for t in range(n_kt)])
                res["C f16 per block"] = q8_scaled_sum(bsum, src.dx, src.dw)
            parts = []
            for key, val in res.items():
                so, se = err_stats(val, orc), err_stats(val, exact)
                ranking[r.label].setdefault(f"Q8_0: {key}", []).append(so)
                parts.append(f"{key}: vs oracle {fmt_err(so)}; vs exact {fmt_err(se)}")
            extra = f" (int sums exact: {exact_ints})" if cb is not None else ""
            print(f"    {r.label}{extra}")
            for ptxt in parts:
                print(f"      {ptxt}")
    return ranking


# HMX instructions and accumulator reads for one 64 x 32 output block and K = 1024, from the loops
# of the census (the kernel loops for path A). The cycle numbers are the v79 silicon measurement of
# 2026-09-20 (tools/hmx-bench): 34.7 pcycles per f16 tile issue, 33.8 per int8 64x32x32 issue, and
# about 711 pcycles for one int32 read (four u8 plane stores) of a 64x32 tile. The simulator times
# no HMX instruction.
PATH_COST = {
    "F16 weights: HMX f16 deep (today)": "K=1024: 6 HMX instructions, 64 f16 tile MACs, 2 f16 reads; v79 ~0.6k pcycles"
                                         " at the kernel rate (9 per tile)",
    "Q8_0: A f16 deep (today)": "K=1024: 6 HMX instructions, 64 f16 tile MACs, 2 f16 reads; plus the HVX dequantization"
                                " of the weights to f16",
    "Q8_0: C f16 per block": "K=1024: 192 HMX instructions, 64 f16 tile MACs, 64 f16 reads (not measured);"
                             " HVX f16->f32 and a scale-add of 2048 values per 32 k",
    "Q8_0: B int8 per block": "K=1024: 320 HMX instructions, 32 int8 tile MACs, 32 int32 reads = 128 u8 plane stores;"
                              " v79 ~24k pcycles (711 per read); 3 planes suffice (|sum| < 2^21), -25%;"
                              " HVX plane join, -128 sum(q_w), f32 scale-add",
}


def report_ranking(ranking: dict[str, dict[str, list]]) -> None:
    """Section 8: per version, the paths ranked by the error against the oracle, with their cost."""
    section("8. Ranking per version: error against the CPU oracle (worst case over K), then cost")
    for label, paths in ranking.items():
        print(f"  {label}:")
        rows = []
        for key, stats in paths.items():
            worst_max = max(s[1] for s in stats)
            worst_mean = max(s[2] for s in stats)
            same = min(s[0] for s in stats)
            rows.append((worst_max, worst_mean, same, key))
        for group in ("F16", "Q8_0"):
            for worst_max, worst_mean, same, key in sorted(r for r in rows if r[3].startswith(group)):
                print(f"    {key:36s} max {worst_max:.2e}  mean {worst_mean:.2e}  f32-identical >= {100 * same:5.1f}%"
                      f"  | cost: {PATH_COST.get(key, '?')}")


def report_verdict(runs: list[Run], groups: dict[str, list[list[str]]], extra: list[Run]) -> None:
    """Section 6: the verdict per HMX mode, from the version groups of its cases."""
    section("6. Verdict per HMX mode (byte identity of all cases of the mode)")
    modes = [
        ("f16 MAC :deep + store .hf, identity bias (matmul, gdn-chunk)",
         lambda n: family(n) in ("rand", "cancel", "growth", "round", "drift", "meth", "pass", "zero") and n.find("flat") < 0),
        ("f16 MAC without :deep + store .hf (flash-attn)", lambda n: n.endswith("flat")),
        ("f16 special values and overflow", lambda n: family(n) == "special" or n == "pass_sp12"),
        ("f16 store with scale/bias words", lambda n: family(n) == "bias"),
        ("f16 control word (second 128 bytes)", lambda n: family(n) == "bias_v1"),
        ("f16 under USR bits", lambda n: family(n) == "usr"),
        ("u8 x s8 MAC + four u8 plane reads", lambda n: n.startswith(("i8_rand", "i8_extreme", "i8_single", "i8_deep"))),
        ("u8 x s8 int32 overflow", lambda n: n.startswith("i8_sat")),
        ("u8 x s4 MAC", lambda n: n.startswith("i4")),
        ("F16 weights, realistic data (f16w)", lambda n: n.startswith("f16w")),
        ("Q8_0 path A: f16 dequantized, deep MAC (q8a)", lambda n: n.startswith("q8a")),
        ("Q8_0 path C: integer quants as f16, sum per block (q8c)", lambda n: n.startswith("q8c")),
        ("Q8_0 path B: int8 quants, int32 read per block (q8b)", lambda n: n.startswith("q8b")),
        ("Q8_0 source quantization on the DSP scalar unit (q8src)", lambda n: n.startswith("q8src")),
    ]
    labels = [r.label for r in runs]
    print(f"  {'mode':62s} {'cases':>5s}  groups")
    for title, pred in modes:
        ns = [n for n in groups if pred(n)]
        if not ns:
            continue
        agg = Counter(fmt_groups(groups[n]) for n in ns)
        print(f"  {title:62s} {len(ns):5d}  " + "; ".join(f"{g}: {c}" for g, c in agg.most_common()))
    if extra:
        print()
        print("  other cores against the default core of the same version (cases identical / total, f16 zeros):")
        for e in extra:
            main = next((r for r in runs if r.label == e.label), None)
            if main is None:
                continue
            common = [n for n in e.cases if n in main.cases]
            same = sum(1 for n in common if e.cases[n].raw_out == main.cases[n].raw_out)
            f16_zero = sum(1 for n in common if e.cases[n].is_f16 and not e.cases[n].out.any())
            print(f"    {e.label} {e.core:10s} {e.path.name:20s} identical {same}/{len(common)};"
                  f" f16 cases with all-zero output {f16_zero}")


def main(argv: list[str] | None = None) -> int:
    """Parses the arguments, loads the runs and prints the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                    help="a census directory and its version label (v73, v75, v79 or v81); repeat for each version")
    ap.add_argument("--case", default="", help="only the cases whose names start with this prefix")
    ap.add_argument("--no-extra", action="store_true", help="do not compare the other cores of out-hmxisa")
    args = ap.parse_args(argv)

    specs = [tuple(s.split("=", 1)) for s in args.run] if args.run else [(l, str(p)) for l, p in DEFAULT_RUNS]
    runs = [Run(label, run_core(Path(path)), Path(path), load_run(Path(path))) for label, path in specs]
    extra = [] if args.no_extra or args.run else discover_extra(runs)
    names = [n for n in runs[0].cases if n.startswith(args.case)]
    missing = {r.label: [n for n in names if n not in r.cases] for r in runs}
    print("HMX census:", ", ".join(f"{r.label}={r.core} ({len(r.cases)} cases, {r.path.relative_to(REPO) if r.path.is_relative_to(REPO) else r.path})" for r in runs))
    for label, ms in missing.items():
        if ms:
            print(f"warning: {label} has no file for {len(ms)} cases: {', '.join(ms[:6])}")
    groups = report_identity(runs, names)
    report_model(runs + [e for e in extra if has_f16_path(e)], names)
    report_accuracy(runs, names)
    report_characterization(runs, names)
    report_int(runs, names)
    ranking = report_oracle(runs, names)
    report_verdict(runs, groups, extra)
    report_ranking(ranking)
    return 0


def has_f16_path(run: Run) -> bool:
    """False for a core whose f16 MAC gives zeros (an HMX without the f16 path)."""
    c = run.cases.get("rand_unit_k32")
    return c is None or bool(c.out.any())


if __name__ == "__main__":
    sys.exit(main())
