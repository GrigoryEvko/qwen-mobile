#!/usr/bin/env python3
"""Compare DSP results of the silicon probe with the ARM CPU oracle.

Two commands:

    oracle_compare.py census <dsp_dir> [--oracle-dir DIR] [--ops-csv FILE] [--only REGEX] [--csv FILE]
        For each out_<op>.oracle.bin (the naive IEEE result of host/oracle.c on the ARM CPU),
        read out_<op>.bin (the DSP result) and report the error of the DSP result against the
        oracle, per op. The oracle directory is the DSP directory unless --oracle-dir is given.

    oracle_compare.py matvec <result_dir>
        For each <kernel>.bin of "isaprobe kernel" in the directory, report the error against
        ggml-ref.bin (the ggml scalar reference on the ARM CPU) and the time of the kernel.

The error of an IEEE result (sf, hf, bf) is the distance in units in the last place between the
two bit patterns (the values in order, +0 and -0 at the same place). A qf32 or qf16 result is
decoded to its value (2M + 1) 2^(E - bias), the model of tools/htp-lab/isa/compare.py, and its
error is |dsp - oracle| / ulp(oracle) in the IEEE format of the same width. A qf lane with M = 0
has no clear value (the qf to IEEE conversion gives 0 for some such lanes), thus the column
"qf_m0" counts these lanes and the ulp figures leave them out. The qfpair ops (qf, then the
conversion to IEEE) do not have this problem. An integer result gives the integer difference, and a predicate result counts the
bytes that differ. "zero+-" counts the lanes that differ only in the sign of a zero. Two NaNs agree. A NaN or an infinity on one side only counts as "nonfinite",
and the ulp figures leave such lanes out.

The script needs numpy. The census file format is that of tools/htp-lab/isa/isa_kernels.h, and
the op types come from tools/htp-lab/isa/isa_ops.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OPS_CSV = REPO / "tools/htp-lab/isa/isa_ops.csv"
CENSUS_HEADER = struct.Struct("<4s7I64s32s")
MV_HEADER = struct.Struct("<4s7IQQiI64s8s")

# (mantissa bits, exponent bits) of the IEEE formats
IEEE = {"sf": (23, 8), "hf": (10, 5), "bf": (7, 8)}
LANE_BYTES = {"sf": 4, "hf": 2, "bf": 2, "qf32": 4, "qf16": 2, "b": 1, "ub": 1, "h": 2, "uh": 2, "w": 4, "uw": 4,
              "pred": 1}
INT_DTYPE = {"b": np.int8, "ub": np.uint8, "h": np.int16, "uh": np.uint16, "w": np.int32, "uw": np.uint32,
             "pred": np.uint8}
UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32}
ORACLE_TYPE = {"qf32": "sf", "qf16": "hf"}


def read_census(path: Path) -> tuple[dict, np.ndarray]:
    """Read a census file: (header fields, data bytes). O(size of the file)."""
    raw = path.read_bytes()
    magic, version, ident, n, bpv, arch, h, source, name, _ = CENSUS_HEADER.unpack_from(raw, 0)
    hdr = {"magic": magic, "id": ident, "n_vectors": n, "bytes_per_vector": bpv, "arch": arch, "source": source,
           "name": name.rstrip(b"\0").decode()}
    return hdr, np.frombuffer(raw, dtype=np.uint8, offset=CENSUS_HEADER.size, count=n * bpv)


def ieee_value(bits: np.ndarray, fmt: str) -> np.ndarray:
    """Decode IEEE bits of sf, hf or bf to float64 (exact, NaN and infinity kept)."""
    with np.errstate(invalid="ignore"):
        if fmt == "sf":
            return bits.astype(np.uint32).view(np.float32).astype(np.float64)
        if fmt == "hf":
            return bits.astype(np.uint16).view(np.float16).astype(np.float64)
        return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def qf_value(bits: np.ndarray, qf: str) -> np.ndarray:
    """Decode qf32 ((2M + 1) 2^(E - 150), M = bits 31:8) or qf16 ((2M + 1) 2^(E - 25), M = bits 15:5)."""
    b = bits.astype(np.int64)
    if qf == "qf32":
        m = b >> 8
        m = np.where(m >= (1 << 23), m - (1 << 24), m)
        return np.ldexp((2 * m + 1).astype(np.float64), ((b & 0xFF) - 150).astype(np.int64))
    m = b >> 5
    m = np.where(m >= (1 << 10), m - (1 << 11), m)
    return np.ldexp((2 * m + 1).astype(np.float64), ((b & 0x1F) - 25).astype(np.int64))


def ordered(bits: np.ndarray, fmt: str) -> np.ndarray:
    """Map IEEE bits to integers in the order of the values (+0 and -0 map to 0)."""
    mb, eb = IEEE[fmt]
    sign = 1 << (mb + eb)
    b = bits.astype(np.int64)
    mag = b & (sign - 1)
    return np.where(b & sign, -mag, mag)


def ulp_of(x: np.ndarray, fmt: str) -> np.ndarray:
    """The ulp of the IEEE format at |x| (the subnormal ulp below the smallest normal)."""
    mb, eb = IEEE[fmt]
    emin = 2 - (1 << (eb - 1))
    with np.errstate(all="ignore"):
        e = np.floor(np.log2(np.where(np.abs(x) > 0, np.abs(x), 1.0)))
    return np.ldexp(1.0, (np.maximum(e, emin) - mb).astype(np.int64))


@dataclass
class Result:
    """The comparison of one DSP result with its oracle."""

    lanes: int
    exact: int         # lanes with the same bits (IEEE, integer), or the same value (qf)
    nonfinite: int     # lanes with a NaN or an infinity on one side only
    zero_sign: int     # lanes that differ only in the sign of a zero (0 ulp)
    qf_m0: int         # qf lanes with M = 0 (not in the ulp figures)
    max_err: float     # ulp (float) or integer units
    mean_err: float
    over_1: int        # lanes with an error of more than one unit
    worst: int         # the lane of max_err, -1 if none
    worst_dsp: str
    worst_ref: str


def compare_lanes(dsp: np.ndarray, ref: np.ndarray, dsp_type: str, ref_type: str) -> Result:
    """Compare DSP lanes of dsp_type with oracle lanes of ref_type. O(lanes)."""
    lb = LANE_BYTES[dsp_type]
    qf_m0 = 0
    d = dsp.view(UINT[lb])
    r = ref.view(UINT[LANE_BYTES[ref_type]])
    n = d.size
    if ref_type in IEEE:
        rv = ieee_value(r, ref_type)
        if dsp_type in IEEE:
            dv = ieee_value(d, dsp_type)
            same = (d == r) | (np.isnan(dv) & np.isnan(rv))
            err = np.abs(ordered(d, dsp_type) - ordered(r, ref_type)).astype(np.float64)
            zero_sign = int((~same & (dv == 0) & (rv == 0)).sum())
        else:
            dv = qf_value(d, dsp_type)
            same = dv == rv
            zero_sign = 0
            with np.errstate(all="ignore"):
                err = np.abs(dv - rv) / ulp_of(rv, ref_type)
            m0 = (d.astype(np.int64) >> (8 if dsp_type == "qf32" else 5)) == 0
        finite = np.isfinite(dv) & np.isfinite(rv)
        nonfinite = int((~finite & ~same).sum())
        use = finite & ~same
        if dsp_type not in IEEE:
            qf_m0 = int((m0 & ~same).sum())
            use &= ~m0
    else:
        dv = d.view(INT_DTYPE[dsp_type]).astype(np.int64)
        rv = r.view(INT_DTYPE[ref_type]).astype(np.int64)
        same = dv == rv
        err = np.abs(dv - rv).astype(np.float64)
        nonfinite = 0
        zero_sign = 0
        qf_m0 = 0
        use = ~same
    errs = np.where(use, err, 0.0)
    worst = int(np.argmax(errs)) if use.any() else -1
    worst_dsp = worst_ref = ""
    if worst >= 0:
        worst_dsp = f"0x{int(d[worst]):0{2 * lb}x}" if dsp_type in IEEE else f"{dv[worst]:.9g}"
        worst_ref = f"{rv[worst]:.9g}"
    return Result(
        lanes=n,
        exact=int(same.sum()),
        nonfinite=nonfinite,
        zero_sign=zero_sign,
        qf_m0=qf_m0,
        max_err=float(errs.max()) if use.any() else 0.0,
        mean_err=float(errs.mean()) if n else 0.0,
        over_1=int((errs > 1.0).sum()),
        worst=worst,
        worst_dsp=worst_dsp,
        worst_ref=worst_ref,
    )


def load_types(ops_csv: Path) -> dict[str, str]:
    """Return {op name: output element type} from isa_ops.csv (a predicate output is "pred")."""
    types: dict[str, str] = {}
    with ops_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            types[row["name"]] = "pred" if row["out_ctype"] == "Q" else row["out_type"]
    return types


def census(dsp_dir: Path, oracle_dir: Path, ops_csv: Path, only: str, csv_out: Path | None) -> int:
    """Report the DSP error against the oracle for each op with an oracle file. Returns the exit code."""
    types = load_types(ops_csv)
    pattern = re.compile(only) if only else None
    rows = []
    for ofile in sorted(oracle_dir.glob("out_*.oracle.bin")):
        name = ofile.name[len("out_"):-len(".oracle.bin")]
        if pattern and not pattern.search(name):
            continue
        dfile = dsp_dir / f"out_{name}.bin"
        if name not in types:
            print(f"oracle_compare: note: {name} is not in {ops_csv}")
            continue
        if not dfile.exists():
            rows.append({"name": name, "status": "no DSP output"})
            continue
        dh, dd = read_census(dfile)
        oh, od = read_census(ofile)
        if dd.size != od.size or dh["id"] != oh["id"]:
            rows.append({"name": name, "status": "the files do not agree in size or id"})
            continue
        dsp_type = types[name]
        ref_type = ORACLE_TYPE.get(dsp_type, dsp_type)
        res = compare_lanes(dd, od, dsp_type, ref_type)
        rows.append({
            "name": name, "status": "ok", "type": dsp_type, "lanes": res.lanes,
            "exact": f"{res.exact / res.lanes:.6f}", "nonfinite": res.nonfinite, "zero_sign": res.zero_sign,
            "qf_m0": res.qf_m0,
            "max_err": f"{res.max_err:.4g}", "mean_err": f"{res.mean_err:.3g}", "over_1": res.over_1,
            "worst_lane": res.worst, "worst_dsp": res.worst_dsp, "worst_ref": res.worst_ref,
            "unit": "ulp" if ref_type in IEEE else "int",
        })
    if not rows:
        print(f"oracle_compare: no out_*.oracle.bin in {oracle_dir}")
        return 1
    print(f"{'op':48s} {'type':5s} {'exact':>9s} {'max':>9s} {'mean':>9s} {'>1':>8s} {'nonfin':>7s} {'zero+-':>7s}"
          f" {'qf_m0':>6s}  worst lane: dsp / oracle")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['name'][:48]:48s} {r['status']}")
            continue
        print(f"{r['name'][:48]:48s} {r['type']:5s} {r['exact']:>9s} {r['max_err']:>9s} {r['mean_err']:>9s} "
              f"{r['over_1']:>8d} {r['nonfinite']:>7d} {r['zero_sign']:>7d} {r['qf_m0']:>6d}  {r['worst_lane']}: "
              f"{r['worst_dsp']} / "
              f"{r['worst_ref']} {r['unit']}")
    if csv_out:
        fields = ["name", "status", "type", "unit", "lanes", "exact", "max_err", "mean_err", "over_1", "nonfinite",
                  "zero_sign", "qf_m0", "worst_lane", "worst_dsp", "worst_ref"]
        with csv_out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        print(f"oracle_compare: wrote {csv_out}")
    return 0


def read_mv(path: Path) -> tuple[dict, np.ndarray]:
    """Read a harness result file of mv_format.h: (header fields, f32 values)."""
    raw = path.read_bytes()
    f = MV_HEADER.unpack_from(raw, 0)
    hdr = {"magic": f[0], "op": f[2], "rows": f[3], "cols": f[4], "kernel_id": f[5], "source": f[6], "iters": f[7],
           "pcycles": f[8], "usecs": f[9], "status": f[10], "name": f[12].rstrip(b"\0").decode()}
    if hdr["magic"] != b"IPMR":
        raise ValueError(f"{path} is not a result file of the harness")
    return hdr, np.frombuffer(raw, dtype=np.float32, offset=MV_HEADER.size, count=hdr["rows"])


def matvec(result_dir: Path) -> int:
    """Report each kernel result of the directory against ggml-ref.bin. Returns the exit code."""
    ref_path = result_dir / "ggml-ref.bin"
    if not ref_path.exists():
        print(f"oracle_compare: no ggml-ref.bin in {result_dir}")
        return 1
    rh, rv = read_mv(ref_path)
    print(f"ggml-ref: {rh['rows']} x {rh['cols']}, cpu {rh['usecs']} us")
    print(f"{'kernel':28s} {'status':>6s} {'exact':>8s} {'max ulp':>9s} {'mean ulp':>9s} {'max rel':>10s} "
          f"{'cycles':>12s} {'us':>8s}")
    for path in sorted(result_dir.glob("*.bin")):
        if path.name == "ggml-ref.bin":
            continue
        try:
            h, v = read_mv(path)
        except (ValueError, struct.error):
            continue
        if h["rows"] != rh["rows"] or h["cols"] != rh["cols"]:
            print(f"{h['name']:28s} the shape is not that of ggml-ref.bin")
            continue
        dbits = v.view(np.uint32)
        rbits = rv.view(np.uint32)
        err = np.abs(ordered(dbits, "sf") - ordered(rbits, "sf"))
        with np.errstate(all="ignore"):
            rel = np.abs(v.astype(np.float64) - rv.astype(np.float64)) / np.maximum(np.abs(rv.astype(np.float64)), 1e-30)
        print(f"{h['name']:28s} {h['status']:>6d} {np.mean(err == 0):>8.4f} {int(err.max()):>9d} {err.mean():>9.3g} "
              f"{np.nanmax(rel):>10.3g} {h['pcycles']:>12d} {h['usecs']:>8d}")
    return 0


def main(argv: list[str]) -> int:
    """Parse the command line and run "census" or "matvec". Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_c = sub.add_parser("census", help="the census ops against the oracle")
    p_c.add_argument("dsp_dir", type=Path)
    p_c.add_argument("--oracle-dir", type=Path, default=None)
    p_c.add_argument("--ops-csv", type=Path, default=OPS_CSV)
    p_c.add_argument("--only", default="", help="a regular expression of the op names")
    p_c.add_argument("--csv", type=Path, default=None, help="write the table to this CSV file")
    p_m = sub.add_parser("matvec", help="the candidate kernels against the ggml reference")
    p_m.add_argument("result_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "census":
        return census(args.dsp_dir, args.oracle_dir or args.dsp_dir, args.ops_csv, args.only, args.csv)
    return matvec(args.result_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
