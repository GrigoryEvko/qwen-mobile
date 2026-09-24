#!/usr/bin/env python3
"""Summarize the HTP0 op profile of the decode graphs in a GGML_HEXAGON_PROFILE=1 log.

Usage: faprof.py [--last N] [--top K] LOG [LOG ...]

The backend prints one "profile-op OPBATCH" line at the start of each DSP batch, then one
"profile-op <OP>" line for each op of that batch. For each of the last N batches (the decode
tokens) the tool prints the batch time, the DSP clock, the op count, and the time of the
FLASH_ATTN_EXT ops (count, sum, median). Then, over the same batches, the K op classes with the
largest time. It only reads the files. O(lines) time and memory.

tools/prof/optable.py and tests/fuzz/hexhost/tools/profsum.py read the same lines. optable.py gives
the per-op tables of the graphs, and profsum.py gives the op time of one step. This tool gives the
flash attention and rotation times of each decode batch and token, and kvtable.py uses its parser.
"""

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BATCH = re.compile(r"profile-op OPBATCH\|.*\|n-ops (\d+)\|.*\|usec (\d+) cycles (\d+) start (\d+) mhz ([\d.]+)")
OP = re.compile(r"profile-op (\w+)\|.*\|usec (\d+)")
KV_DIMS = re.compile(r"profile-op FLASH_ATTN_EXT\|[^|]*\|[^|]* x (\d+):(\d+):(\d+):\d+ x")


def parse(path: Path) -> list[dict]:
    """The batches of one log: a dict per batch with its header values and its ops."""
    batches: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        m = BATCH.search(line)
        if m:
            batches.append({"n_ops": int(m.group(1)), "usec": int(m.group(2)), "mhz": float(m.group(5)),
                            "ops": [], "fa": [], "n_kv": None})
            continue
        m = OP.search(line)
        if m is None or not batches:
            continue
        name, usec = m.group(1), int(m.group(2))
        batches[-1]["ops"].append((name, usec))
        if (name == "MUL_MAT" and "_rot#" in line) or name == "FWHT":
            batches[-1]["rot"] = batches[-1].get("rot", 0) + usec
        if name == "FLASH_ATTN_EXT":
            batches[-1]["fa"].append(usec)
            kv = KV_DIMS.search(line)
            if kv:
                batches[-1]["n_kv"] = int(kv.group(2))
    return batches


def summarize(path: Path, last: int, top: int) -> str:
    """The summary text of the last decode batches of one log."""
    batches = parse(path)
    if not batches:
        return f"== {path.name}: no OPBATCH line"
    sel = batches[-last:]
    out = [f"== {path.name}: {len(batches)} batches, the last {len(sel)}"]
    for b in sel:
        fa = b["fa"]
        fa_txt = (f"FA {len(fa)} ops sum {sum(fa)} us median {statistics.median(fa):.0f} us n_kv {b['n_kv']}"
                  if fa else "FA none")
        out.append(f"  batch {b['usec']:7d} us  {b['mhz']:.1f} MHz  {b['n_ops']:4d} ops  op-sum "
                   f"{sum(u for _, u in b['ops']):7d} us  {fa_txt}")
    classes: dict[str, int] = defaultdict(int)
    for b in sel:
        for name, usec in b["ops"]:
            classes[name] += usec
    ranked = sorted(classes.items(), key=lambda kv: -kv[1])[:top]
    out.append("  per batch: " + ", ".join(f"{name} {usec / len(sel):.0f}" for name, usec in ranked))
    return "\n".join(out)


def per_token(path: Path, per: int, tokens: int) -> str:
    """The medians over the last tokens of one log: the DSP batch time, the op sum, and the time of the
    FLASH_ATTN_EXT, rotation MUL_MAT (a weight named *_rot) and SET_ROWS ops, in us per token. A token has
    per batches."""
    batches = parse(path)[-per * tokens:]
    rows = []
    for i in range(0, len(batches) - per + 1, per):
        grp = batches[i:i + per]
        row = {"batch": sum(b["usec"] for b in grp), "opsum": 0, "fa": 0, "rot": 0, "setrows": 0}
        for b in grp:
            for name, usec in b["ops"]:
                row["opsum"] += usec
                if name == "FLASH_ATTN_EXT":
                    row["fa"] += usec
                elif name == "SET_ROWS":
                    row["setrows"] += usec
            row["rot"] += b.get("rot", 0)
        rows.append(row)
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]} if rows else {}
    return (f"{path.name}: tokens {len(rows)} batch {med.get('batch', 0):.0f} opsum {med.get('opsum', 0):.0f} "
            f"fa {med.get('fa', 0):.0f} rot {med.get('rot', 0):.0f} setrows {med.get('setrows', 0):.0f} us per token")


def main() -> int:
    """Print the summary of each log that the command line names."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--last", type=int, default=4, help="the number of batches at the end of the log")
    ap.add_argument("--top", type=int, default=8, help="the number of op classes to print")
    ap.add_argument("--per", type=int, default=0,
                    help="with N > 0, print the medians per token over the last 8 tokens of N batches each")
    ap.add_argument("logs", nargs="+", type=Path)
    a = ap.parse_args()
    for log in a.logs:
        print(per_token(log, a.per, 8) if a.per > 0 else summarize(log, a.last, a.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
