#!/usr/bin/env python3
"""Put the "hostprof-qt" stamps of each batch on one time axis and name where a slow batch waited.

Usage: qtprof.py [--last N] [--slow-ms T] LOG [LOG ...]

The instrumented build prints one line for each batch response (LLAMA_HOSTPROF=1):
    hostprof-qt: HTP0 seq S entry E read R start A stop B rsp H acq X rel Y
with E, R, A, B and H in microseconds after the host submit of batch S. E is the entry of the DSP
thread into process_ops (a new burst when E > 0), R the read of the request, A and B the batch, H
the read of the response by the host, X the time of the VTCM acquire of the burst in us, Y the
count of the VTCM release callbacks. For each batch the tool gives the phases: to the read (R),
the read to the start (A - R), the batch (B - A), and the stop to the response (H - B). A batch
whose H is more than T ms gets the name of its largest phase. It only reads the files. O(lines).

No patch of patches/series prints the "hostprof-qt" line. The instrumented build of the memory
measurements (the tree build/memory/src5 on the build box) prints it. It is not a build of the app.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r"hostprof-qt: (\S+) seq (\d+) entry (-?[\d.]+) read (-?[\d.]+) start (-?[\d.]+) "
                  r"stop (-?[\d.]+) rsp (-?[\d.]+) acq (\d+) rel (\d+)")


def summarize(path: Path, last: int, slow_ms: float) -> str:
    """The table of the last batches of one log and the count of each stall cause."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m:
            e, r, a, b, h = (float(m.group(i)) / 1000.0 for i in range(3, 8))
            rows.append((int(m.group(2)), e, r, a, b, h, int(m.group(8)) / 1000.0, int(m.group(9))))
    rows = rows[-last:]
    out = [f"== {path.name}: {len(rows)} batches (ms after the submit: entry, read, read->start, batch, "
           f"stop->rsp, rsp; acq ms, release count)"]
    causes: dict[str, int] = {}
    for seq, e, r, a, b, h, acq, rel in rows:
        phases = {"before the read": r, "read to start": a - r, "batch": b - a, "stop to response": h - b}
        name = ""
        if h > slow_ms:
            name = max(phases, key=phases.get)
            if acq > slow_ms / 2:
                name = "VTCM acquire"
            causes[name] = causes.get(name, 0) + 1
        burst = "new burst" if e > 0 else ""
        out.append(f"  seq {seq:5d} entry {e:8.1f} read {r:8.1f} start {a - r:6.1f} batch {b - a:6.1f} "
                   f"rsp {h - b:8.1f} total {h:8.1f} acq {acq:6.1f} rel {rel:3d} {burst:9s} {name}")
    if rows:
        out.append(f"  median total {statistics.median(r[5] for r in rows):.1f} ms, slow batches by cause: {causes}")
    return "\n".join(out)


def main() -> int:
    """Print the table of each log that the command line names."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--last", type=int, default=34, help="the number of batches at the end of the log")
    ap.add_argument("--slow-ms", type=float, default=80.0, help="the total time that marks a slow batch")
    ap.add_argument("logs", nargs="+", type=Path)
    a = ap.parse_args()
    for log in a.logs:
        print(summarize(log, a.last, a.slow_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
