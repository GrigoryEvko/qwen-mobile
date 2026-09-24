#!/usr/bin/env python3
"""Summarize the table of tests/fuzz/ops/run.sh compare.

Reads the compare log, sums the columns of each backend, and lists the
kind/path rows where two backends differ in crash, defect, loose or nonfin.
run_classes.py reads the RUN lines of the same compare, and not its table.

Usage: compare_summary.py LOG [BACKEND_A BACKEND_B]
Complexity: O(rows).
"""

import sys
from collections import defaultdict

COLS = ["runs", "inval", "unsup", "crash", "defect", "pass", "subn", "strict", "loose", "nonfin"]


def parse(path: str) -> dict[str, dict[str, list[int]]]:
    """Return {backend: {"kind path": [runs, inval, ..., nonfin]}} from the tables of the log."""
    rows: dict[str, dict[str, list[int]]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3 + len(COLS) or not (parts[0].startswith("CPU-") or parts[0].startswith("HTP0-")):
                continue
            try:
                vals = [int(x) for x in parts[3:3 + len(COLS)]]
            except ValueError:
                continue
            rows[parts[0]][f"{parts[1]} {parts[2]}"] = vals
    return rows


def main() -> int:
    """Print the sums of each backend, then the differences of two backends."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = parse(sys.argv[1])
    print(f"{'backend':28s} " + " ".join(f"{c:>6s}" for c in COLS))
    for b in sorted(rows):
        s = [sum(v[i] for v in rows[b].values()) for i in range(len(COLS))]
        print(f"{b:28s} " + " ".join(f"{x:6d}" for x in s))
    if len(sys.argv) == 4:
        a, b = sys.argv[2], sys.argv[3]
        print(f"\nkind/path rows where {a} and {b} differ in crash, defect, loose or nonfin:")
        for key in sorted(set(rows[a]) | set(rows[b])):
            va, vb = rows[a].get(key), rows[b].get(key)
            if va is None or vb is None:
                print(f"  {key}: only in {'A' if vb is None else 'B'}")
                continue
            idx = [COLS.index(c) for c in ("crash", "defect", "strict", "loose", "nonfin")]
            if [va[i] for i in idx] != [vb[i] for i in idx]:
                print(f"  {key:40s} A crash/defect/strict/loose/nonfin {[va[i] for i in idx]}  B {[vb[i] for i in idx]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
