#!/usr/bin/env python3
"""Summarize the LLAMA_HOSTPROF lines of the decode tokens in a log.

Usage: hostprof.py [--skip N] LOG [LOG ...]

With LLAMA_HOSTPROF=1, llama_decode prints one "hostprof: decode #N ..." line for each call
(src/llama-context.cpp), and the Hexagon session prints one "hostprof: HTP0 graphs ..." line for
each token (ggml-hexagon.cpp, hostprof_print). The tool takes the calls with one token (the
decode steps), skips the first N of them, and prints the median, the 90th percentile and the
maximum of each time field of the two lines, in microseconds. It only reads the files.
O(lines) time and memory.

The two lines come from patches/hexagon-host/0001 (the Hexagon session) and patches/hexagon-host/0002
(llama_decode). tools/prof/stallprof.py pairs the same session line with the DSP batches.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

DECODE = re.compile(r"hostprof: decode #(\d+) tokens (\d+) .*?\| (.*?) \| since_last_return (-?\d+) us")
SESSION = re.compile(r"hostprof: (HTP\d+) graphs .*?\| (.*?) \| turnaround (-?\d+) us")
FIELD = re.compile(r"([a-z_]+) (-?\d+)(?: \((\d+) B\))?")


def fields(text: str) -> dict[str, int]:
    """The "name value" pairs of one part of a hostprof line, as a dict of integers."""
    return {m.group(1): int(m.group(2)) for m in FIELD.finditer(text)}


def stats(values: list[int]) -> str:
    """The median, the 90th percentile and the maximum of a list, as one text."""
    v = sorted(values)
    p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
    return f"{statistics.median(v):8.0f} {p90:8.0f} {v[-1]:8.0f}"


def summarize(path: Path, skip: int) -> str:
    """The summary text of one log."""
    decode: list[dict[str, int]] = []
    session: list[dict[str, int]] = []
    pending: dict[str, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = SESSION.search(line)
        if m:
            pending = fields(m.group(2))
            pending["turnaround"] = int(m.group(3))
            continue
        m = DECODE.search(line)
        if m is None:
            continue
        # The session line of a token comes before the decode line of the same call.
        if int(m.group(2)) == 1:
            d = fields(m.group(3))
            d["since_last_return"] = int(m.group(4))
            decode.append(d)
            session.append(pending)
        pending = {}
    decode, session = decode[skip:], session[skip:]
    out = [f"== {path.name}: {len(decode)} one-token decodes after {skip} skipped (us: median p90 max)"]
    if not decode:
        return "\n".join(out)
    for key in decode[0]:
        out.append(f"  decode  {key:18s} {stats([d.get(key, 0) for d in decode])}")
    keys = [k for k in (session[0] if session and session[0] else {})]
    for key in keys:
        out.append(f"  session {key:18s} {stats([s.get(key, 0) for s in session if s])}")
    return "\n".join(out)


def main() -> int:
    """Print the summary of each log that the command line names."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip", type=int, default=1, help="the number of first one-token decodes to skip")
    ap.add_argument("logs", nargs="+", type=Path)
    a = ap.parse_args()
    for log in a.logs:
        print(summarize(log, a.skip))
    return 0


if __name__ == "__main__":
    sys.exit(main())
