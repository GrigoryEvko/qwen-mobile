#!/usr/bin/env python3
"""Find where a stalled decode token loses its time, from one log with LLAMA_HOSTPROF=1 and
GGML_HEXAGON_PROFILE=1.

Usage: stallprof.py [--last N] [--stall-ms T] LOG [LOG ...]

For each of the last N session tokens (one "hostprof: HTP0" line each) the tool pairs the host
wait of the token with the DSP batches that the backend printed since the previous token: the
batch time, and the start stamp of the first batch against the start stamp of the first batch of
the previous token. For a token whose wait is more than T ms, it prints the ops whose time is
more than 3 times the median time of the same op in the other tokens. It only reads the files.
O(lines) time and memory.

tools/prof/stalls.py is a different tool: it reads the PMU stall counters of each op. This tool
pairs the host wait of each token (patches/hexagon-host/0001) with the DSP batches of that token.
"""

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BATCH = re.compile(r"profile-op OPBATCH\|.*\|usec (\d+) cycles \d+ start (\d+)")
OP = re.compile(r"profile-op (\w+)\|([^|]*)\|.*\|usec (\d+)")
WAIT = re.compile(r"hostprof: HTP0 .*\| pack (\d+) submit (\d+) wait (\d+) pop")


def tokens(path: Path) -> list[dict]:
    """The session tokens of one log: the host wait, the batches and the ops of each."""
    out: list[dict] = []
    cur = {"batches": [], "ops": []}
    for line in path.read_text(errors="replace").splitlines():
        m = BATCH.search(line)
        if m:
            cur["batches"].append((int(m.group(1)), int(m.group(2))))
            continue
        m = OP.search(line)
        if m:
            cur["ops"].append((m.group(1), m.group(2)[:60], int(m.group(3))))
            continue
        m = WAIT.search(line)
        if m:
            cur["wait"] = int(m.group(3))
            out.append(cur)
            cur = {"batches": [], "ops": []}
    return out


def summarize(path: Path, last: int, stall_ms: float) -> str:
    """The table of the last tokens of one log, and the slow ops of each stalled token."""
    toks = [t for t in tokens(path) if t["batches"]][-last:]
    out = [f"== {path.name}: {len(toks)} tokens (ms: host wait, DSP batches, start of the first batch after the "
           f"start of the previous token)"]
    per_op: dict[str, list[int]] = defaultdict(list)
    for t in toks:
        for name, _, usec in t["ops"]:
            per_op[name].append(usec)
    med = {k: statistics.median(v) for k, v in per_op.items()}
    prev_start = None
    for i, t in enumerate(toks):
        batch = sum(u for u, _ in t["batches"]) / 1000.0
        start = t["batches"][0][1]
        delta = (start - prev_start) / 1e6 if prev_start is not None else float("nan")
        prev_start = start
        wait = t["wait"] / 1000.0
        flag = "  STALL" if wait > stall_ms else ""
        out.append(f"  {i:3d} wait {wait:7.1f} batch {batch:7.1f} start+{delta:7.1f}{flag}")
        if flag:
            slow = [(u, n, d) for n, d, u in t["ops"] if u > 3 * med.get(n, u) and u > 1000]
            for u, n, d in sorted(slow, reverse=True)[:5]:
                out.append(f"        {n} {d} {u} us (median {med[n]:.0f})")
    return "\n".join(out)


def main() -> int:
    """Print the table of each log that the command line names."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--last", type=int, default=33, help="the number of session tokens at the end of the log")
    ap.add_argument("--stall-ms", type=float, default=80.0, help="the host wait that marks a stalled token")
    ap.add_argument("logs", nargs="+", type=Path)
    a = ap.parse_args()
    for log in a.logs:
        print(summarize(log, a.last, a.stall_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
