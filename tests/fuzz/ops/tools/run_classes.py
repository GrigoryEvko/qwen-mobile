#!/usr/bin/env python3
"""Sort the RUN lines of an op compare into error classes, per kind and path or per kind and backend.

With FUZZ_OPS_COMPARE_LIST=1, the compare of the op fuzzer (tests/fuzz/ops/run.sh compare, or
ops_oracle compare) prints one line for each run:
    RUN <index> <tag> <kind> <path> verdict=<v> special=<0|1> ratio=<r> ulp=<u> err=<e> ref=<f> | <case>
err is the largest finite error of the run, and ref is the largest finite |reference| of the case.
The tag starts with the backend, for example HTP0-release-none.

Two tables:
  --by path     The preset. The runs without special values and with a verdict other than pass or
                subnormal, per kind, path and verdict. Each run gets one class: nonfinite for the
                verdict nonfinite, garbage for err > ref, wrong for err > 1 % of ref, and small for
                the other runs.
  --by backend  All runs per kind, backend and special flag (sp or nsp): the count of each verdict,
                of ratio > 1e30, of err > 1 % of ref and of err > ref. Here a garbage run is also
                in the column err>1%.
Each row also gives the worst relative error err / ref and its case.

Usage: run_classes.py [--by path|backend] COMPARE_TXT...

The tool only reads the files. O(lines) time, O(rows of the table) memory.
"""

import argparse
import re
import sys
from collections import defaultdict

RUN = re.compile(r"^RUN (\d+) (\S+) (\S+) (\S+) verdict=(\S+) special=(\d) ratio=(\S+) ulp=(\S+) "
                 r"err=(\S+) ref=(\S+) \| (.*)$")


def run_class(err: float, ref: float, verdict: str) -> str:
    """Return the class of one run of the table --by path.

    Args:
        err: The largest finite error of the run
        ref: The largest finite |reference| of the case
        verdict: The verdict of the run

    Returns:
        "nonfinite", "garbage", "wrong" or "small"
    """
    if verdict == "nonfinite":
        return "nonfinite"
    if ref == 0.0:
        return "garbage" if err > 0.0 else "small"
    if err > ref:
        return "garbage"
    if err > 0.01 * ref:
        return "wrong"
    return "small"


def runs(paths: list[str]):
    """Yield the groups of each RUN line of the files, in the order of the files. O(lines).

    Args:
        paths: The compare outputs

    Yields:
        (index, tag, kind, path, verdict, special, ratio, ulp, err, ref, case) as texts
    """
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                m = RUN.match(line)
                if m is not None:
                    yield m.groups()


def by_path(paths: list[str]) -> None:
    """Print the classes of the runs without special values, per kind, path and verdict.

    Args:
        paths: The compare outputs
    """
    table: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    worst: dict[tuple[str, str, str], tuple[float, str]] = {}
    for _, _, kind, cpath, verdict, special, _, _, err_text, ref_text, case in runs(paths):
        if special != "0" or verdict in ("pass", "subnormal"):
            continue
        err, ref = float(err_text), float(ref_text)
        key = (kind, cpath, verdict)
        table[key][run_class(err, ref, verdict)] += 1
        rel = err / ref if ref > 0 else float("inf")
        if key not in worst or rel > worst[key][0]:
            worst[key] = (rel, case)
    print(f"{'kind':16} {'path':24} {'verdict':13} garbage wrong small nonfin  max err/ref  worst case")
    for key in sorted(table):
        c = table[key]
        rel, case = worst[key]
        print(f"{key[0]:16} {key[1]:24} {key[2]:13} {c['garbage']:7} {c['wrong']:5} {c['small']:5} "
              f"{c['nonfinite']:6}  {rel:11.3g}  {case[:90]}")


def by_backend(paths: list[str]) -> None:
    """Print the counts of all runs per kind, backend and special flag.

    Args:
        paths: The compare outputs
    """
    agg: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    worst: dict[tuple[str, str, str], tuple[float, str]] = {}
    for idx, tag, kind, _, verdict, special, ratio, _, err_text, ref_text, case in runs(paths):
        key = (kind, tag.split("-")[0], "sp" if special == "1" else "nsp")
        a = agg[key]
        err, ref, q = float(err_text), float(ref_text), float(ratio)
        rel = err / ref if ref > 0 else (float("inf") if err > 0 else 0.0)
        a["runs"] += 1
        a[verdict] += 1
        a["wrong"] += rel > 0.01
        a["garbage"] += rel > 1.0
        a["r30"] += q > 1e30
        if rel > worst.get(key, (0.0, ""))[0]:
            worst[key] = (rel, f"{idx} {case} err={err_text} ref={ref_text}")
    print(f"{'kind':14} {'backend':9} {'sp':3} {'runs':>5} {'strict':>6} {'loose':>5} {'nonfin':>6} "
          f"{'r>1e30':>6} {'err>1%':>6} {'err>ref':>7}  worst relative error")
    for key, a in sorted(agg.items()):
        rel, text = worst.get(key, (0.0, ""))
        print(f"{key[0]:14} {key[1]:9} {key[2]:3} {a['runs']:5d} {a['above-strict']:6d} {a['above-loose']:5d} "
              f"{a['nonfinite']:6d} {a['r30']:6d} {a['wrong']:6d} {a['garbage']:7d}  {rel:.3g} {text[:100]}")


def main() -> int:
    """Print the table of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--by", choices=("path", "backend"), default="path", help="the rows of the table")
    ap.add_argument("files", nargs="+", help="the compare outputs with FUZZ_OPS_COMPARE_LIST=1")
    a = ap.parse_args()
    (by_path if a.by == "path" else by_backend)(a.files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
