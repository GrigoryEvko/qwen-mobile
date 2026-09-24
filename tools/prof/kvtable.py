#!/usr/bin/env python3
"""The table of the Q8_0 KV decision: the DSP time per decode token of four KV forms at four
depths for the 2B and the 4B, and the KL of each form against the naive oracle.

Usage: kvtable.py [--root build/memory]

The DSP times come from the profile runs (GGML_HEXAGON_PROFILE=1, 8 decode tokens) of the stages
g (d4096: F16 g04/g02, Q8_0 with the MUL_MAT rotation g05/g03), n (d16, d1024, d16384: F16,
Q8_0 MUL_MAT, Q8_0 without rotation) and o (Q8_0 with the FWHT rotation at all four depths). The
KL values come from stage p (kvkl, the "all" row and the "decode" row). The tool prints the median
per token of the DSP batch time, and a dash where a stage has no run. It only reads the files.

The logs are the pulled outputs of the memory stages in build/memory (phone-g-out, phone-n-out,
phone-o-out, phone-p-out and phone-q-out). The parser of the profile lines is faprof.parse.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from faprof import parse  # noqa: E402

PER = {"2B": 1, "4B": 4}
FORMS = ["F16", "Q8_0 MUL_MAT", "Q8_0 FWHT", "Q8_0 no rot"]


def token_ms(log: Path, per: int) -> float | None:
    """The median DSP batch time per token of the last 8 tokens, in ms, or None without the log."""
    if not log.exists():
        return None
    batches = parse(log)[-8 * per:]
    if len(batches) < per:
        return None
    sums = [sum(b["usec"] for b in batches[i:i + per]) for i in range(0, len(batches) - per + 1, per)]
    return statistics.median(sums) / 1000.0


def parts(log: Path, per: int) -> str:
    """The medians per token of the FA, rotation and SET_ROWS time of one log, as one text."""
    batches = parse(log)[-8 * per:]
    rows = []
    for i in range(0, len(batches) - per + 1, per):
        fa = rot = sr = 0
        for b in batches[i:i + per]:
            rot += b.get("rot", 0)
            for name, usec in b["ops"]:
                fa += usec if name == "FLASH_ATTN_EXT" else 0
                sr += usec if name == "SET_ROWS" else 0
        rows.append((fa, rot, sr))
    if not rows:
        return "-"
    return "/".join(f"{statistics.median(r[k] for r in rows):.0f}" for k in range(3))


def runs(root: Path) -> dict[tuple[str, int, str], Path]:
    """The profile log of each (model, depth, form)."""
    out: dict[tuple[str, int, str], Path] = {}
    g = root / "phone-g-out"
    out[("2B", 4096, "F16")] = g / "g04.log"
    out[("2B", 4096, "Q8_0 MUL_MAT")] = g / "g05.log"
    out[("4B", 4096, "F16")] = g / "g02.log"
    out[("4B", 4096, "Q8_0 MUL_MAT")] = g / "g03.log"
    n = root / "phone-n-out"
    i = 1
    for model in ("2B", "4B"):
        for depth in (16, 1024, 16384):
            for form in ("F16", "Q8_0 MUL_MAT", "Q8_0 no rot"):
                out[(model, depth, form)] = n / f"n{i:02d}.log"
                i += 1
    q = root / "phone-q-out"
    out[("2B", 4096, "Q8_0 no rot")] = q / "q01.log"
    out[("4B", 4096, "Q8_0 no rot")] = q / "q02.log"
    o = root / "phone-o-out"
    i = 5
    for model in ("2B", "4B"):
        for depth in (16, 1024, 4096, 16384):
            out[(model, depth, "Q8_0 FWHT")] = o / f"o{i:02d}.log"
            i += 1
    return out


def kl(root: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    """The (all, decode) mean KL of each (model, depth name, form) of stage p."""
    out = {}
    p = root / "phone-p-out"
    i = 1
    for model in ("2B", "4B"):
        for dname in ("4k", "16k"):
            for form in ("F16", "Q8_0 MUL_MAT", "Q8_0 FWHT", "Q8_0 no rot"):
                log = p / f"p{i:02d}.log"
                i += 1
                if not log.exists():
                    continue
                text = log.read_text(errors="replace")
                a = re.search(r"^all\s+\d+\s+([\d.]+)", text, re.M)
                d = re.search(r"^decode\S*\s+\d+\s+([\d.]+)", text, re.M)
                if a and d:
                    out[(model, dname, form)] = (float(a.group(1)), float(d.group(1)))
    return out


def main() -> int:
    """Print the two tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("build/memory"))
    a = ap.parse_args()
    r = runs(a.root)
    print("DSP time per decode token, ms (median of 8 tokens); in parentheses the difference to F16")
    for model in ("2B", "4B"):
        print(f"{model}: depth | " + " | ".join(FORMS))
        for depth in (16, 1024, 4096, 16384):
            vals = [token_ms(r[(model, depth, f)], PER[model]) if (model, depth, f) in r else None for f in FORMS]
            base = vals[0]
            cells = []
            for v in vals:
                if v is None:
                    cells.append("-")
                elif base is None or v is vals[0]:
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:.2f} ({v - base:+.2f})")
            print(f"  d{depth:<6d} | " + " | ".join(cells))
    print("The parts of the DSP time per token, us: FA / rotation (MUL_MAT or FWHT) / SET_ROWS")
    for model in ("2B", "4B"):
        for depth in (16, 1024, 4096, 16384):
            cells = []
            for f in FORMS:
                log = r.get((model, depth, f))
                cells.append(parts(log, PER[model]) if log is not None and log.exists() else "-")
            print(f"  {model} d{depth:<6d} | " + " | ".join(cells))
    k = kl(a.root)
    print("KL against the naive oracle: all rows / the 64 decode rows")
    for model in ("2B", "4B"):
        for dname in ("4k", "16k"):
            cells = [f"{k[(model, dname, f)][0]:.6f} / {k[(model, dname, f)][1]:.6f}" if (model, dname, f) in k
                     else "-" for f in FORMS]
            print(f"  {model} {dname:4s} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
