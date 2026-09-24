#!/usr/bin/env python3
"""Compare the float code of two DSP libraries, function by function, for three defect classes.

The script reads the disassembly of two libraries, for example libggml-htp-v79.so (the base, which
the phone verified) and libggml-htp-v81.so. For each function it counts the instructions of three
classes of defect that passed the simulator and failed on silicon:

- chain: a qf32 add or subtract with an operand that another qf32 add or subtract made, with no
  conversion to IEEE between the two. On the v79 chip such chains gave other bits than the
  simulator.
- narrow: each conversion of a qf32 pair to f16 (V.hf = W.qf32), by the producer of its input: a
  multiply (the rounding to f16 is the one rounding), an add or subtract, the direct conversion
  V.qf32 = V.sf (v81), or a value that the analysis does not see (a load, a copy after a merge).
- wrap: a conversion to IEEE (V.sf = V.qf32 or V.hf = W.qf32) of a qf32 product. A product below
  the normal f32 range can wrap its exponent on silicon and give a large value.

It also counts the IEEE-form HVX float opcodes, which must be 0, and the direct conversions
V.qf32 = V.sf. A function is flagged when a count or its set of dataflow signatures differs between
the two libraries. The dataflow signatures come from inventory.chain_signatures.

Usage:
    tools/htp-lab/inventory/archdiff.py --base v79.s --other v81.s --out diff.csv [--qwen-only]

The disassembly files come from hexagon-llvm-objdump (refer to inventory.py). The Qwen3.5 functions
are the closure of the op roots of inventory.OP_MAP in the base library.

Complexity is O(n) in the instructions of the two files, plus the chain pass of each function.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inventory as inv  # noqa: E402  (the path insert must come first)

RE_QF_ADDSUB = re.compile(r"^V\.qf32 = v(add|sub)\(")
RE_QF_PRODUCER = re.compile(r"^V\.qf32 = v(add|sub|mpy)\(")
RE_QF_MPY = re.compile(r"^[VW]\.qf32 = vmpy\(")
RE_NARROW = "V.hf = W.qf32"
RE_TO_SF = "V.sf = V.qf32"
CVT_SF_QF = "V.qf32 = V.sf"

METRICS: tuple[str, ...] = (
    "chain", "narrow_mpy", "narrow_addsub", "narrow_cvt", "narrow_other", "wrap", "cvt_sf_qf32", "ieee_hvx_float",
    "qf32_qq_addsub",
)


def split_signature(sig: str) -> tuple[str, list[list[str]]]:
    """Split a dataflow signature "form <= p1 | p2" into the form and the producer forms of each operand."""
    form, _, rest = sig.partition(" <= ")
    operands = [op.split("/") if op != "x" else [] for op in rest.split(" | ")] if rest else []
    return form, operands


def function_metrics(forms: Counter[str], sigs: Counter[str], classes: Counter[str]) -> Counter[str]:
    """Count the defect classes of one function from its forms, signatures and classes."""
    m: Counter[str] = Counter()
    m["ieee_hvx_float"] = classes.get("ieee_hvx_float", 0)
    m["cvt_sf_qf32"] = forms.get(CVT_SF_QF, 0)
    m["qf32_qq_addsub"] = sum(n for f, n in forms.items() if re.match(r"^V\.qf32 = v(add|sub)\(V\.qf32,V\.qf32\)", f))
    for sig, n in sigs.items():
        form, operands = split_signature(sig)
        producers = [p for op in operands for p in op]
        if RE_QF_ADDSUB.match(form) and any(RE_QF_ADDSUB.match(p) for p in producers):
            m["chain"] += n
        if form == RE_NARROW:
            if any(p == CVT_SF_QF for p in producers):
                m["narrow_cvt"] += n
            elif any(RE_QF_MPY.match(p) for p in producers):
                m["narrow_mpy"] += n
            elif any(RE_QF_ADDSUB.match(p) for p in producers):
                m["narrow_addsub"] += n
            else:
                m["narrow_other"] += n
        if form in (RE_NARROW, RE_TO_SF) and any(RE_QF_MPY.match(p) for p in producers):
            m["wrap"] += n
    return m


def qwen_functions(lib: inv.Library) -> set[str]:
    """Give the functions that the Qwen3.5 op variants of inventory.OP_MAP reach in a library."""
    known = set(lib.classes)
    out: set[str] = set()
    for _op, _variant, use, roots in inv.OP_MAP:
        if use.startswith("-"):
            continue
        out |= set(inv.resolve_roots(roots, lib.edges, known))
    return out


def main(argv: list[str]) -> int:
    """Parse the arguments, compare the two libraries and write the CSV file and the summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True, help="the disassembly of the base library")
    ap.add_argument("--other", type=Path, required=True, help="the disassembly of the other library")
    ap.add_argument("--out", type=Path, required=True, help="the CSV file to write")
    ap.add_argument("--qwen-only", action="store_true", help="print only the Qwen3.5 functions in the summary")
    args = ap.parse_args(argv)
    for p in (args.base, args.other):
        if not p.is_file():
            print(f"error: {p} does not exist. Make it with hexagon-llvm-objdump (refer to inventory.py).",
                  file=sys.stderr)
            return 1

    base = inv.analyze_library("base", args.base)
    other = inv.analyze_library("other", args.other)
    q35 = qwen_functions(base) | qwen_functions(other)
    names = sorted(set(base.classes) | set(other.classes))

    rows: list[list[object]] = []
    flagged: list[tuple[str, list[str]]] = []
    totals = {"base": Counter(), "other": Counter()}
    for name in names:
        mb = function_metrics(base.forms.get(name, Counter()), base.sigs.get(name, Counter()),
                              base.classes.get(name, Counter()))
        mo = function_metrics(other.forms.get(name, Counter()), other.sigs.get(name, Counter()),
                              other.classes.get(name, Counter()))
        totals["base"] += mb
        totals["other"] += mo
        forms_b = base.forms.get(name, Counter())
        forms_o = other.forms.get(name, Counter())
        sig_b = set(base.sigs.get(name, Counter()))
        sig_o = set(other.sigs.get(name, Counter()))
        diff_metrics = [k for k in METRICS if mb[k] != mo[k]]
        form_diff = sorted(f for f in set(forms_b) | set(forms_o) if forms_b.get(f, 0) != forms_o.get(f, 0))
        sig_diff = sorted(sig_b ^ sig_o)
        flag = "yes" if (diff_metrics or form_diff or sig_diff) else "no"
        rows.append([name, "yes" if name in q35 else "no", flag]
                    + [f"{mb[k]}/{mo[k]}" for k in METRICS]
                    + [" || ".join(f"{f} {forms_b.get(f, 0)}/{forms_o.get(f, 0)}" for f in form_diff),
                       " || ".join(f"{s} {base.sigs.get(name, Counter()).get(s, 0)}/"
                                   f"{other.sigs.get(name, Counter()).get(s, 0)}" for s in sig_diff[:16])])
        if flag == "yes" and (name in q35 or not args.qwen_only):
            flagged.append((name, diff_metrics))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "qwen35", "flag"] + [f"{k} base/other" for k in METRICS]
                   + ["forms that differ base/other", "signatures that differ base/other"])
        w.writerows(rows)

    print("totals base/other: " + ", ".join(f"{k} {totals['base'][k]}/{totals['other'][k]}" for k in METRICS))
    print(f"functions: {len(names)}, flagged: {sum(1 for r in rows if r[2] == 'yes')}, "
          f"Qwen3.5 flagged: {sum(1 for r in rows if r[2] == 'yes' and r[1] == 'yes')}")
    for name, dm in flagged:
        print(f"  {name}{' (qwen35)' if name in q35 else ''}: {', '.join(dm) if dm else 'signatures or forms only'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
