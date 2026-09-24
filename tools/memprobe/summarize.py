#!/usr/bin/env python3
"""Summarize the memprobe logs of a phone batch as one block per run.

Usage: summarize.py LOG [LOG ...]

For each log: the llama.cpp buffer lines, the flash attention decision, the graph splits, the
TIME lines, the DMABUF total and the process memory (VmRSS, RssAnon, RssFile, MemAvailable)
at each stage, in MiB. It only reads the files.
"""

import re
import sys
from pathlib import Path

BUFFER = re.compile(r"(model buffer size|KV buffer size|RS buffer size|compute buffer size =|output buffer size|"
                    r"Flash Attention|graph splits|load_mode|did not initialize|quantized V)")
MEM = re.compile(r"^MEM (\S+) (.*)$")
DMABUF = re.compile(r"^DMABUF (\S+) (\S+) count=(\d+) mib=([\d.]+)")


def kib_to_mib(value: str) -> str:
    """The KiB value of a MEM field as MiB with one decimal."""
    return f"{int(value) / 1024:.1f}"


def summarize(path: Path) -> str:
    """The summary block of one log file."""
    lines = path.read_text(errors="replace").splitlines()
    out = [f"== {path.name}"]
    seen = set()
    for line in lines:
        if BUFFER.search(line) and not line.startswith("~llama_context") and line not in seen:
            seen.add(line)
            out.append("  " + line.strip())
    for line in lines:
        if line.startswith(("TIME ", "HASH ", "LOADMEM ", "DROPCACHE ")):
            out.append("  " + line)
    dmabuf = {}
    for line in lines:
        m = DMABUF.match(line)
        if m:
            dmabuf[m.group(1)] = f"{m.group(4)} MiB in {m.group(3)}"
    for line in lines:
        m = MEM.match(line)
        if not m:
            continue
        fields = dict(kv.split("=", 1) for kv in m.group(2).split())
        stage = m.group(1)
        out.append(f"  {stage:14s} rss {kib_to_mib(fields['VmRSS'])} anon {kib_to_mib(fields['RssAnon'])} "
                   f"file {kib_to_mib(fields['RssFile'])} avail {kib_to_mib(fields['MemAvailable'])} "
                   f"dmabuf {dmabuf.get(stage, '-')}")
    return "\n".join(out)


def main() -> int:
    """Print the summary of each log that the command line names."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for name in sys.argv[1:]:
        print(summarize(Path(name)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
