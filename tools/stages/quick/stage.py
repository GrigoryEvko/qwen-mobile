#!/usr/bin/env python3
"""The phone stage "quick": the HEAD libraries against the libraries with four changes of the Hexagon backend.

Usage:
    stage.py commands [--out PATH]   write the phone command file (build/quick/phone-commands.txt)
    stage.py table [--root DIR]      print the results from the pulled logs (build/quick/phone-out)

build/quick/stage.py is a link to this file. The files of the stage go to build/quick.

The changes (the patches of build/quick/patches/final, the tree build/quick/src):
    P1  a pointwise unary op on contiguous tensors gets rows of up to 16 KiB (the beta SIGMOID, the SOFTPLUS
        of the gate and the SCALE that sets a recurrent state to zero)
    P2  the DSP removes only the mappings that a batch needs to free (GGML_HEXAGON_MMAP_LRU, preset 1)
    P3  the batch cache replays a graph of more than one DSP batch (GGML_HEXAGON_BATCHCACHE_MULTI, preset 1)
    P4  the host makes the text of the profile line of an op when it packs the batch (only the output of
        GGML_HEXAGON_PROFILE changes)

The variants. Each one runs the app configuration: Q8_0 K and V, the rotation as FWHT, fusion and state fusion on.
    a  the HEAD libraries (lib-base)
    b  the new libraries with the preset values (lib-new)
    c  b with chunks of 640 MiB for the buffers (GGML_HEXAGON_MBUF=640): 2 batches for a decode token
    l  b with the previous eviction of the DSP (GGML_HEXAGON_MMAP_LRU=0)
    m  b with the replay of a graph of one batch only (GGML_HEXAGON_BATCHCACHE_MULTI=0)
    v  b with the VA limit that the backend measures (GGML_HEXAGON_VMEM=0)
    p  b with the host that polls the queue (GGML_HEXAGON_OPPOLL=1)

A run name is <block>-<round>-<variant>, for example t-2-c. Each run writes <name>-gate.txt, <name>.out and
<name>.log to the phone directory out/. The table only reads files. O(size of the logs) time.
"""

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/quick"
MODEL = "/data/local/tmp/qwen/models/Qwen3.5-4B-Q8_0.gguf"
LAPTOP_STAGE = "build/quick"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/quick"
# The link build/quick/stage.py and the file in tools/stages/quick find the same stage directory
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
APP_ENV = "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -o jsonl -ctk q8_0 -ctv q8_0"
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"
LIBS = ("libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-htp-v79.so", "libggml-opencl.so",
        "libggml.so", "libllama-bench-impl.so", "libllama-common.so", "libllama.so", "libmtmd.so")
NEW_LIBS = ("libggml-hexagon.so", "libggml-htp-v79.so")
BINS = ("gate.sh", "llama-bench", "memprobe", "test-backend-ops")
UNARY_OPS = "SCALE,CLAMP,LEAKY_RELU,SQR,SQRT,NEG,EXP,SIGMOID,SILU,GELU,GELU_QUICK,SOFTPLUS,TANH,ABS,LOG,RELU"
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x llama-bench; pgrep -x memprobe; pgrep -x test-backend-op; echo pgrep-done'"
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')


@dataclass(frozen=True)
class Variant:
    """One library set and its environment switch."""
    key: str
    lib: str
    env: str
    text: str


@dataclass(frozen=True)
class Block:
    """One kind of run: the tool, its arguments, the variants of an odd round, the rounds, the time limit in
    seconds, the MemAvailable (KiB) of the gate, the extra environment and the text."""
    key: str
    tool: str
    args: str
    variants: str
    rounds: int
    limit: int
    gate_kb: int
    env: str
    text: str


VARIANTS = {v.key: v for v in (
    Variant("a", "lib-base", "", "HEAD"),
    Variant("b", "lib-new", "", "new, preset values"),
    Variant("c", "lib-new", "GGML_HEXAGON_MBUF=640", "new, chunks of 640 MiB"),
    Variant("l", "lib-new", "GGML_HEXAGON_MMAP_LRU=0", "new, previous eviction"),
    Variant("m", "lib-new", "GGML_HEXAGON_BATCHCACHE_MULTI=0", "new, replay of one batch only"),
    Variant("v", "lib-new", "GGML_HEXAGON_VMEM=0", "new, measured VA limit"),
    Variant("p", "lib-new", "GGML_HEXAGON_OPPOLL=1", "new, the host polls the queue"),
)}
BLOCKS = [
    Block("k", "test-backend-ops", f"-o {UNARY_OPS} -b HTP0", "ab", 1, 100, 2097152, "",
          "test-backend-ops of the pointwise unary ops on HTP0 against the CPU"),
    Block("h", "memprobe", "--hash -p 1024 -n 16", "abc", 1, 90, 8388608, "",
          "memprobe --hash, a prompt of 1024 tokens and 16 decode tokens: the logits hashes"),
    Block("p", "llama-bench", "-p 512 -n 0 -d 0 -r 3", "abc", 3, 60, 8388608, "",
          "llama-bench pp512 at the depth 0, 3 repetitions"),
    Block("t", "llama-bench", "-p 0 -n 32 -d 0,4096 -r 2", "abcp", 3, 90, 8388608, "",
          "llama-bench tg32 at the depths 0 and 4096, 2 repetitions"),
    Block("f", "memprobe", "-p 1024 -n 8", "abcl", 1, 90, 8388608, "GGML_HEXAGON_PROFILE=1",
          "op profile, a prompt of 1024 tokens and 8 decode tokens"),
    Block("q", "memprobe", "-p 64 -n 24", "abmcp", 1, 60, 8388608, "LLAMA_HOSTPROF=1",
          "host times (LLAMA_HOSTPROF), a prompt of 64 tokens and 24 decode tokens"),
    Block("v", "llama-bench", "-p 0 -n 8 -r 1", "v", 1, 60, 8388608, "",
          "llama-bench tg8 with the VA limit that the backend measures"),
]


def runs() -> list[tuple[Block, int, Variant]]:
    """The runs in the order of the stage: each block, round by round, odd rounds in the order of the block
    and even rounds in the reverse order. O(runs)."""
    out = []
    for b in BLOCKS:
        for rnd in range(1, b.rounds + 1):
            order = b.variants if rnd % 2 else b.variants[::-1]
            out.extend((b, rnd, VARIANTS[k]) for k in order)
    return out


def run_lines(b: Block, rnd: int, v: Variant) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    name = f"{b.key}-{rnd}-{v.key}"
    stem = f"{PHONE}/out/{name}"
    # lib-new holds only the two libraries that differ (the Hexagon backend and the DSP library)
    lib = f"{PHONE}/{v.lib}"
    ld = lib if v.lib == "lib-base" else f"{lib}:{PHONE}/lib-base"
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={ld} ADSP_LIBRARY_PATH={lib}", APP_ENV, v.env, b.env) if x)
    if b.tool == "test-backend-ops":
        tool = f"{PHONE}/bin/test-backend-ops {b.args}"
        title = f"# KERNEL: {name}, {b.text}, {v.key.upper()}: {v.text}"
    else:
        fixed = BENCH_ARGS if b.tool == "llama-bench" else PROBE_ARGS
        tool = f"{PHONE}/bin/{b.tool} -m {MODEL} {fixed} {b.args}"
        title = f"# REAL-MODEL Qwen3.5-4B-Q8_0: {name}, {b.text}, {v.key.upper()}: {v.text}"
    cmd = (f"sh {PHONE}/bin/gate.sh {b.gate_kb} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"timeout -s KILL {b.limit} env {env} {tool} > {stem}.out 2> {stem}.log; "
           f"echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt")
    return ["#", title, THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "quick": the HEAD libraries against the libraries with the patches of build/quick/patches, the 4B Q8_0
# in the app configuration (Q8_0 K and V with the FWHT rotation, GGML_HEXAGON_OPFUSION=1, OPFUSION_STATE=1).
#
# The changes: P1 a pointwise unary op on contiguous tensors gets rows of up to 16 KiB (the beta SIGMOID of each gated
# delta net layer, the SOFTPLUS of its gate, the SCALE that sets a recurrent state to zero), P2 the DSP removes only the
# mappings that a batch needs to free (GGML_HEXAGON_MMAP_LRU), P3 the batch cache replays a graph of more than one DSP
# batch (GGML_HEXAGON_BATCHCACHE_MULTI), P4 the host makes the text of the profile line of an op when it packs the
# batch. No change gives other logits.
#
# The libraries (build/quick/build.sh): lib-base is the tree of HEAD (tests/sanitizers/llama-copy.sh), lib-new the same
# tree with the four patches, both with the preset, the flags, the LTO and the build number of scripts/build-native.sh.
# lib-new holds the two libraries that differ (libggml-hexagon.so, libggml-htp-v79.so), the tools come from the base
# build.
#
# The variants: A HEAD, B new with the preset values, C B with GGML_HEXAGON_MBUF=640 (chunks of 640 MiB, 2 batches for
# a decode token), L B with GGML_HEXAGON_MMAP_LRU=0, M B with GGML_HEXAGON_BATCHCACHE_MULTI=0, V B with
# GGML_HEXAGON_VMEM=0 (the backend measures the VA limit and prints it), P B with GGML_HEXAGON_OPPOLL=1 (the host polls
# the queue and does not block in dspqueue_read).
#
# The runs, 36:
#   k  test-backend-ops -o <the 16 pointwise unary ops> -b HTP0: A B. Decides: the new unary descriptors pass on the
#      DSP (compare the FAIL counts of A and B).
#   h  memprobe --hash -p 1024 -n 16: A B C. Decides: the logits hashes after the prompt and after each decode token
#      are the same for A, B and C (bit-exact: the 1024 prompt has the new unary rows and the state clear, the decode
#      tokens 3 to 16 are multi-batch replays in B and C).
#   p  llama-bench pp512 at d0, -r 3: A B C, 3 rounds (A B C, C B A, A B C). Decides: the prefill speed of P1 to P3
#      (B against A) and of the chunk size (C against B).
#   t  llama-bench tg32 at d0 and d4096, -r 2: A B C P, 3 rounds. Decides: the decode speed of P1 to P3 (B against A),
#      of the chunk size (C against B) and of the poll (P against B). The gate lines give the battery and the NPU heat
#      before and after each run, thus the cost of the poll.
#   f  GGML_HEXAGON_PROFILE=1, memprobe -p 1024 -n 8: A B C L. Decides: the op split (SIGMOID, SOFTPLUS, SCALE), the
#      batch count of a decode token and the prologue of each batch (tools/prof/optable.py batches). The dims on the
#      profile lines of B, C and L agree with the ops (P4). A can show the dims of the next ubatch.
#   q  LLAMA_HOSTPROF=1, memprobe -p 64 -n 24: A B M C P. Decides: the pack time, the replays and the wait of a decode
#      token.
#   v  GGML_HEXAGON_VMEM=0, llama-bench tg8: V. Decides: the VA limit that the DSP gives ("measured max vmem").
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB, or 2 GB for test-backend-ops, and it prints the caps), the tool under timeout -s KILL (110 s or less), the exit
# code and the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 15 minutes of tool time (t 7, p 2.3,
# f 2, h 1.2, q 1.3, k 0.7, v 0.3) plus about 8 s of gate and checks for each run, plus the waits for thermal status 0 and a
# battery of 38 C or less. The push is about 140 MB, the pull about 15 MB. Then on the box: python3 build/quick/stage.py table
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/bin/{f}" for f in BINS)
    base = " ".join(f"{LAPTOP_STAGE}/phone/lib-base/{f}" for f in LIBS)
    new = " ".join(f"{LAPTOP_STAGE}/phone/lib-new/{f}" for f in NEW_LIBS)
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        f"{ADB} shell 'ls -l /data/local/tmp/qwen/models | grep -E \"Qwen3.5-4B-Q8_0.gguf\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib-base {PHONE}/lib-new {PHONE}/out'",
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {base} {PHONE}/lib-base/",
        f"{ADB} push {new} {PHONE}/lib-new/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
    ]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = HEADER.rstrip("\n").split("\n") + setup_lines()
    lines += ["#", f"# ==== Qwen3.5-4B-Q8_0: {len(runs())} runs ===="]
    for b, rnd, v in runs():
        lines += run_lines(b, rnd, v)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The results ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*)")
HOSTPROF_RE = re.compile(r"hostprof: HTP0 graphs \d+ ops (\d+) hits (\d+) replays (\d+) verified \d+ batches (\d+) \| "
                         r"pack (\d+) submit (\d+) wait (\d+) .*turnaround (-?\d+) us")


def read(root: Path, name: str) -> tuple[str, str, str]:
    """The gate file, the stdout and the stderr of one run, or empty strings."""
    return tuple((root / f"{name}{s}").read_text(errors="replace") if (root / f"{name}{s}").exists() else ""
                 for s in ("-gate.txt", ".out", ".log"))


def conditions(gate: str) -> str:
    """The exit code, the caps before and after, and the flags of one run."""
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    flags = [] if "gate: OK" in gate else ["gate stopped the run"]
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        flags.append(f"caps changed to {after.group(2)}/{after.group(3)}")
    if after and after.group(1) not in ("", "0"):
        flags.append(f"thermal {after.group(1)}")
    return f"rc={rc.group(1) if rc else '?'} caps {caps}" + (" " + ", ".join(flags) if flags else "")


def table(root: Path) -> int:
    """Print the results of each block."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    for b, rnd, v in runs():
        gate, _, _ = read(root, f"{b.key}-{rnd}-{v.key}")
        print(f"{b.key}-{rnd}-{v.key}: {conditions(gate) if gate else 'no gate file'}")
    print()

    for k in "ab":
        _, out, log = read(root, f"k-1-{k}")
        summary = re.findall(r"(\d+)/(\d+) tests passed", out + log)
        fails = [ln.strip() for ln in (out + log).splitlines() if "FAIL" in ln]
        print(f"k {k.upper()}: tests passed {summary}, FAIL lines {len(fails)}")
        for ln in fails[:10]:
            print(f"    {ln[:160]}")
    print()

    hashes = {k: re.findall(r"^HASH (.*)$", read(root, f"h-1-{k}")[1], re.M) for k in "abc"}
    for k in "bc":
        same = hashes[k] == hashes["a"] and hashes["a"]
        print(f"h {k.upper()} against A: {len(hashes[k])} and {len(hashes['a'])} HASH lines, "
              f"{'the same' if same else 'DIFFERENT'}")
    print()

    tests: dict[tuple, dict[str, dict[int, float]]] = {}
    for b, rnd, v in runs():
        if b.key not in ("p", "t"):
            continue
        for line in read(root, f"{b.key}-{rnd}-{v.key}")[1].splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                key = (rec["n_prompt"], rec["n_gen"], rec["n_depth"])
                tests.setdefault(key, {}).setdefault(v.key, {})[rnd] = statistics.median(rec["samples_ts"])
    for key, per in sorted(tests.items()):
        cells = []
        for k in "abcp":
            vals = per.get(k, {})
            if not vals:
                continue
            cell = f"{k.upper()} {statistics.median(vals.values()):.2f}"
            # A change against A, and the poll and the chunk size against B, the median of the paired rounds
            ref = "b" if k in "cp" else "a"
            if k != "a" and ref in per:
                ratios = [vals[r] / per[ref][r] for r in vals if r in per[ref]]
                if ratios:
                    cell += (f" ({100 * (statistics.median(ratios) - 1):+.2f}% vs {ref.upper()} "
                             f"[{100 * (min(ratios) - 1):+.2f}, {100 * (max(ratios) - 1):+.2f}])")
            cells.append(cell)
        print(f"pp{key[0]} tg{key[1]} d{key[2]}: " + " | ".join(cells))
    print()

    for k in "abmcp":
        rows = [m for m in HOSTPROF_RE.finditer(read(root, f"q-1-{k}")[2])]
        dec = [m for m in rows if int(m.group(1)) < 2000][4:]
        if dec:
            med = lambda i: statistics.median(int(m.group(i)) for m in dec)  # noqa: E731
            print(f"q {k.upper()}: {len(dec)} decode tokens: replays {sum(int(m.group(3)) for m in dec)}, "
                  f"batches {med(4):.0f}, pack {med(5):.0f} us, submit {med(6):.0f} us, wait {med(7):.0f} us, "
                  f"turnaround {med(8):.0f} us")
    print()
    for k in "abcl":
        print(f"f {k.upper()}: python3 tools/prof/optable.py graphs {root}/f-1-{k}.log")
    vm = re.findall(r"measured max vmem (\d+)", read(root, "v-1-v")[2])
    print(f"\nv: measured max vmem {vm}")
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
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(runs())} runs")
        return 0
    return table(a.root)


if __name__ == "__main__":
    sys.exit(main())
