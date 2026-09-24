#!/usr/bin/env python3
"""The phone stage "quick-hash": which change of the stage quick moves the logits hashes.

Usage:
    stage.py commands [--out PATH]   write the phone command file (build/quick-hash/phone-commands.txt)
    stage.py table [--root DIR]      print the results from the pulled logs (build/quick-hash/phone-out)

build/quick-hash/stage.py is a link to this file. The stage uses the phone files of the stage quick: the link
build/quick-hash/phone points to build/quick/phone. The run lines come from tools/stages/quick/stage.py.

The variants, each in the app configuration of the stage quick:
    a  the HEAD libraries (lib-base)
    b  the new libraries with the four patches (lib-new)
    l  b with the previous eviction of the DSP (GGML_HEXAGON_MMAP_LRU=0)
    m  b with the replay of a graph of one batch only (GGML_HEXAGON_BATCHCACHE_MULTI=0)
    n  b with the two switches of l and m
"""

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("quick_stage", Path(__file__).resolve().parents[1] / "quick" / "stage.py")
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

q.PHONE = "/data/local/tmp/qwen/quick-hash"
q.LAPTOP_STAGE = "build/quick-hash"
q.BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/quick-hash"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / q.LAPTOP_STAGE))

q.VARIANTS = {v.key: v for v in (
    q.Variant("a", "lib-base", "", "HEAD"),
    q.Variant("b", "lib-new", "", "new, preset values"),
    q.Variant("l", "lib-new", "GGML_HEXAGON_MMAP_LRU=0", "new, previous eviction"),
    q.Variant("m", "lib-new", "GGML_HEXAGON_BATCHCACHE_MULTI=0", "new, replay of one batch only"),
    q.Variant("n", "lib-new", "GGML_HEXAGON_MMAP_LRU=0 GGML_HEXAGON_BATCHCACHE_MULTI=0",
              "new, previous eviction and replay of one batch only"),
)}
q.BLOCKS = [
    q.Block("h", "memprobe", "--hash -p 1024 -n 16", "ablmn", 2, 90, 8388608, "",
            "memprobe --hash, a prompt of 1024 tokens and 16 decode tokens: the logits hashes"),
    q.Block("r", "memprobe", "--hash --reps 2 -p 1024 -n 4", "ab", 1, 90, 8388608, "",
            "memprobe --hash --reps 2: the prompt again into a cleared memory, then 4 decode tokens"),
    q.Block("v", "memprobe", "-p 64 -n 4", "ab", 1, 90, 8388608, "GGML_HEXAGON_VMEM=0",
            "memprobe with the VA limit that the backend measures (GGML_HEXAGON_VMEM=0)"),
]

q.HEADER = """\
# Phone stage "quick-hash": which change of the stage quick moves the logits hashes of the 4B Q8_0, in the app
# configuration (Q8_0 K and V with the FWHT rotation, GGML_HEXAGON_OPFUSION=1, OPFUSION_STATE=1).
#
# The files are the files of the stage quick (build/quick/phone, the link build/quick-hash/phone on the box): lib-base
# is the tree of HEAD, lib-new has the four patches. The stage quick gave the same hashes for B and C, but other hashes
# for A, from the prompt on. Two switches of lib-new turn off two changes: GGML_HEXAGON_MMAP_LRU=0 (the previous
# eviction of the DSP) and GGML_HEXAGON_BATCHCACHE_MULTI=0 (the replay of a graph of one batch only). The row change of
# the unary ops has no switch. The change of the profile lines does nothing without GGML_HEXAGON_PROFILE.
#
# The variants: A HEAD, B new, L B with MMAP_LRU=0, M B with BATCHCACHE_MULTI=0, N B with the two switches.
#
# The runs, 14:
#   h  memprobe --hash -p 1024 -n 16: A B L M N, 2 rounds (A B L M N, N M L B A). Decides: each variant gives the
#      same hashes in its two rounds. If N has the hashes of A, the unary rows do not move the values, and L and M
#      show which of the other two changes moves them. The replay acts on the decode tokens only, thus a different
#      prompt hash does not come from the replay. If N has the hashes of B, the unary rows move the values.
#   r  memprobe --hash --reps 2 -p 1024 -n 4: A B. The second pass decodes the prompt again into a cleared memory,
#      thus the host clears a used recurrent state first. In B the second pass is a replay of the prompt graph.
#      Decides: the prompt hash and the 4 decode hashes are equal to those of h for the same variant.
#   v  GGML_HEXAGON_VMEM=0, memprobe -p 64 -n 4: A B. Decides: the log gives the VA limit that the backend measures
#      ("measured max vmem"), or the error of the session start, for HEAD and for the new libraries. llama-bench of
#      the stage quick gave rc=1 in this configuration and its log had no line of the backend after the load.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB, and it prints the caps), the tool under timeout -s KILL (90 s), the exit code and the conditions after the run,
# then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger), although only
# the hashes decide. Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 5 minutes of tool
# time plus about 8 s of gate and checks for each run. The push is about 140 MB, the pull less than 5 MB. Then on
# the box: python3 build/quick-hash/stage.py table
"""


def hashes(root: Path, name: str) -> list[tuple[str, str]]:
    """The HASH lines of one run as (what, hex) pairs, in order."""
    return re.findall(r"^HASH (.*) ([0-9a-f]{16})$", q.read(root, name)[1], re.M)


def table(root: Path) -> int:
    """Print the results of each block."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    for b, rnd, v in q.runs():
        gate, _, _ = q.read(root, f"{b.key}-{rnd}-{v.key}")
        print(f"{b.key}-{rnd}-{v.key}: {q.conditions(gate) if gate else 'no gate file'}")
    print()

    h = {(k, r): hashes(root, f"h-{r}-{k}") for k in "ablmn" for r in (1, 2)}
    ref_a, ref_b = h[("a", 1)], h[("b", 1)]
    for k in "ablmn":
        one, two = h[(k, 1)], h[(k, 2)]
        same_a = sum(x == y for x, y in zip(one, ref_a))
        same_b = sum(x == y for x, y in zip(one, ref_b))
        first = next((w for (w, x), (_, y) in zip(one, ref_a) if x != y), "none")
        print(f"h {k.upper()}: {len(one)} and {len(two)} HASH lines, rounds {'the same' if one == two and one else 'DIFFERENT'}, "
              f"equal to A in {same_a}, equal to B in {same_b}, the first line that differs from A: {first}")
    print()

    for k in "ab":
        rr = hashes(root, f"r-1-{k}")
        hh = h[(k, 1)]
        pre = dict(rr).get("prefill")
        print(f"r {k.upper()}: prompt hash of the second pass {pre}, h {dict(hh).get('prefill')}: "
              f"{'the same' if pre and pre == dict(hh).get('prefill') else 'DIFFERENT'}; decode lines "
              f"{'the same' if rr[1:] and rr[1:] == hh[1:len(rr)] else 'DIFFERENT'}")
    print()

    for k in "ab":
        log = q.read(root, f"v-1-{k}")[2]
        # The query of the domains fails in each run of the phone, thus its line is not news
        lines = [ln for ln in log.splitlines()
                 if re.search(r"measured max vmem|failed|error|Error", ln) and "FASTRPC_GET_DOMAINS" not in ln]
        print(f"v {k.upper()}: {q.conditions(q.read(root, f'v-1-{k}')[0])}")
        for ln in lines[:12]:
            print(f"    {ln[:200]}")
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the results from the pulled logs")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    a = ap.parse_args()
    if a.cmd == "commands":
        n = q.write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(q.runs())} runs")
        return 0
    return table(a.root)


if __name__ == "__main__":
    sys.exit(main())
