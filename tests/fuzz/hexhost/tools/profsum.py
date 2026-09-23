#!/usr/bin/env python3
"""Summarize the DSP op time of a profile log of the Hexagon backend (GGML_HEXAGON_PROFILE=1).

Usage: profsum.py [--decode N] LOG [LOG ...]

The backend prints one "profile-op OPBATCH" line at the start of each DSP batch, then one
"profile-op <OP>" line for each op of the batch. A token can take more than one DSP batch (one for
each split of its graph).

  --decode N  The log holds one prompt, then N decode tokens (hexhost_logits_hash decode N), and each
              has the same count of DSP batches. The tool skips the batches of the prompt.
  (none)      All the batches of the log are one step (hexhost_logits_hash prefill N).

For each log the tool prints, for one step, the sum of the op times, the sum of the batch times and
the op count, and the mean DSP clock. The op time does not include the host time, thus it compares
two DSP libraries when the CPU clock of the phone changes. O(lines).
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

BATCH = re.compile(r"profile-op OPBATCH\|.*\|n-ops (\d+)\|.*\|usec (\d+) cycles (\d+) start (\d+) mhz ([\d.]+)")
OP = re.compile(r"profile-op (\w+)\|.*\|usec (\d+)")


def batches(path: Path) -> list[dict]:
    """The DSP batches of one log, each with its time, clock, op count and op-time sum."""
    out: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        m = BATCH.search(line)
        if m:
            out.append({"n_ops": int(m.group(1)), "usec": int(m.group(2)), "mhz": float(m.group(5)), "op_sum": 0})
            continue
        m = OP.search(line)
        if m and out and m.group(1) != "OPBATCH":
            out[-1]["op_sum"] += int(m.group(2))
    return out


def main() -> int:
    """Print one line for each log. Gives 1 when a log has no batch or a batch count that does not fit."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode", type=int, default=0, help="the count of decode tokens after the prompt")
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args()
    code = 0
    for path in args.logs:
        bs = batches(path)
        steps = args.decode if args.decode > 0 else 1
        per = len(bs) // (steps + 1) if args.decode > 0 else len(bs)
        if not bs or (args.decode > 0 and len(bs) % (steps + 1)):
            print(f"{path.name}: {len(bs)} batches, not a prompt and {steps} tokens of the same batch count")
            code = 1
            continue
        sel = bs[per:] if args.decode > 0 else bs
        print(f"{path.name}: {len(sel)} batches, for each step: op-sum {sum(b['op_sum'] for b in sel) / steps:9.0f} us, "
              f"batch {sum(b['usec'] for b in sel) / steps:9.0f} us, ops {sum(b['n_ops'] for b in sel) / steps:6.0f}, "
              f"{statistics.mean(b['mhz'] for b in sel):7.1f} MHz")
    return code


if __name__ == "__main__":
    sys.exit(main())
