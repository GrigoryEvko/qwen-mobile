#!/usr/bin/env python3
"""The phone stage "unary-kl": the KL check, the logits hashes and the speed of the row change of the pointwise
unary ops (the switch GGML_HEXAGON_UNARY_FLAT).

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/unary-kl/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled logs (build/unary-kl/phone-out)

build/unary-kl/stage.py is a link to this file, and tools/stages/unary-kl/build.sh builds the files.

The stage unary-rows found that the new rows of the SOFTPLUS move the logits hashes: on the NPU, 167 of 32768
elements of the prompt shape differ by 1 ulp, with the same error against the CPU for the old and the new rows.
The SIGMOID and the SCALE did not move the hashes of the 4B. This stage decides the preset of the switch:
    - If the KL of the value 1 against the naive base stays at the floor (the KL of HEAD, with the error of the mean
      and the spread of the two runs of each), the preset stays 1 and each kind of op gets new rows.
    - If the KL moves, the preset is 26 (SIGMOID, SCALE and the other pointwise ops, not SOFTPLUS), if the hashes of
      26 equal the hashes of HEAD on the 4B and on the 2B.

The variants, each in the app configuration (Q8_0 K and V with the FWHT rotation, op fusion and state fusion on):
    a  the HEAD libraries (lib-base)
    m  the new libraries with GGML_HEXAGON_UNARY_FLAT=26: not SOFTPLUS
    f  the new libraries with the preset value 1: each kind of pointwise op

The blocks, in the order of the stage:
    h   memprobe --hash -p 1024 -n 16 on the 4B: A M
    i   memprobe --hash -p 1024 -n 16 on the 2B: A M
    kp  llama-perplexity KL of the prefill path (-b 512, 4 chunks of 512) on the 4B against naive-4B-q8.kld: A F,
        2 rounds (A F, F A)
    kd  llama-perplexity KL of the decode path (-b 1 -ub 1, 1 chunk of 512) on the 4B against the same base: A F,
        2 rounds
    p   llama-bench pp512 and pp1024 at depth 0, -r 3: A M F, 3 rounds (A M F, F M A, A M F)
    g   llama-bench tg32 at depth 0, -r 3: A M F, 3 rounds

A run name is <block>-<round>-<variant>, for example kp-2-f. Each run writes <name>-gate.txt, <name>.out and
<name>.log to the phone directory out/. The rate table uses a run when its gate passed, its exit code is 0, the CPU
caps after the run are the caps before it, and the thermal status after it is 0. --all also uses the runs with
changed caps or heat. The table only reads files. O(size of the logs) time.
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
PHONE = "/data/local/tmp/qwen/unary-kl"
MODELS = "/data/local/tmp/qwen/models"
MODEL_4B = f"{MODELS}/Qwen3.5-4B-Q8_0.gguf"
MODEL_2B = f"{MODELS}/Qwen3.5-2B-Q8_0.gguf"
EVAL = "/data/local/tmp/qwen/eval"
MODEL_KB = 8388608  # the MemAvailable (KiB) that the gate requires
LAPTOP_STAGE = "build/unary-kl"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/unary-kl"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
APP_ENV = "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"
PPL_ARGS = (f"-dev HTP0 -ngl 99 -t 4 -fa on -ctk q8_0 -ctv q8_0 -f {EVAL}/wiki.test.raw -c 512 "
            f"--kl-divergence-base {EVAL}/naive-4B-q8.kld --kl-divergence")
TOOLS = ("llama-bench", "llama-perplexit", "memprobe")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell '" + "; ".join(f"pgrep -x {t}" for t in TOOLS) + "; echo pgrep-done'"
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
    """One kind of run: the tool with its arguments, the variants of an odd round, the rounds, the time limit in
    seconds and the text. The gate and the lines after the tool take about 6 s, thus each limit is 110 s or less."""
    key: str
    tool: str
    args: str
    variants: str
    rounds: int
    limit: int
    text: str


VARIANTS = {v.key: v for v in (
    Variant("a", "lib-base", "", "HEAD"),
    Variant("m", "lib-new", "GGML_HEXAGON_UNARY_FLAT=26", "new, value 26: not SOFTPLUS"),
    Variant("f", "lib-new", "", "new, preset value 1: each kind"),
)}
BLOCKS = [
    Block("h", "memprobe", f"-m {MODEL_4B} {PROBE_ARGS} --hash -p 1024 -n 16", "am", 1, 90,
          "memprobe --hash on the 4B, a prompt of 1024 tokens and 16 decode tokens"),
    Block("i", "memprobe", f"-m {MODEL_2B} {PROBE_ARGS} --hash -p 1024 -n 16", "am", 1, 90,
          "memprobe --hash on the 2B, a prompt of 1024 tokens and 16 decode tokens"),
    Block("kp", "llama-perplexity", f"-m {MODEL_4B} {PPL_ARGS} --chunks 4 -b 512", "af", 2, 100,
          "KL of the prefill path on the 4B, 4 chunks of 512, -b 512"),
    Block("kd", "llama-perplexity", f"-m {MODEL_4B} {PPL_ARGS} --chunks 1 -b 1 -ub 1", "af", 2, 110,
          "KL of the decode path on the 4B, 1 chunk of 512, -b 1 -ub 1"),
    Block("p", "llama-bench", f"-m {MODEL_4B} {BENCH_ARGS} -p 512,1024 -n 0 -d 0 -r 3", "amf", 3, 90,
          "llama-bench pp512 and pp1024 at depth 0, 3 repetitions"),
    Block("g", "llama-bench", f"-m {MODEL_4B} {BENCH_ARGS} -p 0 -n 32 -d 0 -r 3", "amf", 3, 90,
          "llama-bench tg32 at depth 0, 3 repetitions"),
]


def runs() -> list[tuple[Block, int, Variant]]:
    """The runs in the order of the stage: each block, round by round, odd rounds in the order of the block and even
    rounds in the reverse order. O(runs)."""
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
    lib = f"{PHONE}/{v.lib}"
    ld = lib if v.lib == "lib-base" else f"{lib}:{PHONE}/lib-base"
    # The row change does not change the DSP code, thus each run loads the DSP library of lib-base
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={ld} ADSP_LIBRARY_PATH={PHONE}/lib-base", APP_ENV, v.env) if x)
    cmd = (f"sh {PHONE}/bin/gate.sh {MODEL_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"timeout -s KILL {b.limit} env {env} {PHONE}/bin/{b.tool} {b.args} > {stem}.out 2> {stem}.log; "
           f"echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt")
    model = "Qwen3.5-2B-Q8_0" if MODEL_2B in b.args else "Qwen3.5-4B-Q8_0"
    return ["#", f"# REAL-MODEL {model}: {name}, {b.text}, {v.key.upper()}: {v.text}", THERMAL,
            f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "unary-kl": the KL check, the logits hashes and the speed of the row change of the pointwise unary ops
# (the switch GGML_HEXAGON_UNARY_FLAT), in the app configuration (Q8_0 K and V with the FWHT rotation,
# GGML_HEXAGON_OPFUSION=1, OPFUSION_STATE=1).
#
# The libraries (tools/stages/unary-kl/build.sh): lib-base is the tree of HEAD, lib-new the same tree with the row
# change. lib-new holds only the libraries that differ, and each run loads the DSP library of lib-base.
#
# The variants: A HEAD, M new with GGML_HEXAGON_UNARY_FLAT=26 (not SOFTPLUS), F new with the preset value 1.
#
# The runs, 30:
#   h   memprobe --hash -p 1024 -n 16 on the 4B: A M. Decides: M has the hashes of A.
#   i   memprobe --hash -p 1024 -n 16 on the 2B: A M. Decides: M has the hashes of A.
#   kp  KL of the prefill path on the 4B against the naive base (4 chunks, -b 512): A F, 2 rounds.
#   kd  KL of the decode path on the 4B against the naive base (1 chunk, -b 1 -ub 1): A F, 2 rounds.
#       Decides: the mean KL of F is inside the KL of A, with the error of the mean and the spread of the two runs.
#       Then the preset stays 1. If the KL of F moves, the preset is 26, if h and i give the hashes of A.
#   p   llama-bench pp512 and pp1024 at depth 0, -r 3: A M F, 3 rounds. Decides: the prefill gain of M and of F.
#   g   llama-bench tg32 at depth 0, -r 3: A M F, 3 rounds. Decides: the decode speed of M and F against A.
# The variants run in the order of the block in an odd round and in the reverse order in an even round.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB, and it prints the caps), the tool under timeout -s KILL (110 s or less), the exit code and the conditions after
# the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 21 minutes of tool time plus about
# 4 minutes of gates and checks, plus the waits for thermal status 0. The push is about 170 MB, the pull less than
# 2 MB. Then on the box: python3 build/unary-kl/stage.py table
"""


def stage_files() -> list[str]:
    """The files of the stage: each line of phone/SHA256SUMS that the build wrote. The build recipe checks the needed
    libraries of each program and library, thus the push takes each of these files."""
    sums = STAGE_DIR / "phone" / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"stage.py: {sums} does not exist. Run tools/stages/unary-kl/build.sh first.")
    return [line.split()[1] for line in sums.read_text().splitlines() if line.strip()]


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    files = stage_files()
    lines = [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODELS} | grep -E \"(2B|4B)-Q8_0.gguf\"; ls -l {EVAL} | grep -E \"naive-4B-q8|wiki.test\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib-base {PHONE}/lib-new {PHONE}/out'",
    ]
    for d in ("bin", "lib-base", "lib-new"):
        part = [f for f in files if f.startswith(d + "/")]
        if part:
            lines.append(f"{ADB} push " + " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in part) + f" {PHONE}/{d}/")
    lines += [
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK; echo {len(files)} files; "
        f"chmod 755 {PHONE}/bin/*'",
    ]
    return lines


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone directory goes
    only when the pull has each of its files."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell '" + "; ".join(f"pgrep -x {t}" for t in TOOLS) + f"; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
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
    lines += ["#", f"# ==== {len(runs())} runs ===="]
    for b, rnd, v in runs():
        lines += run_lines(b, rnd, v)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The table ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*)")
KLD_RE = re.compile(r"Mean\s+KLD:\s+([\d.]+) ±\s+([\d.]+)")
TOP_RE = re.compile(r"Same top p:\s+([\d.]+) ±\s+([\d.]+)")
MAXKL_RE = re.compile(r"Maximum KLD:\s+([\d.]+)")
PPL_RE = re.compile(r"Mean PPL\(Q\)\s+:\s+([\d.]+)")


def read(root: Path, name: str) -> tuple[str, str, str]:
    """The gate file, the stdout and the stderr of one run, or empty strings."""
    return tuple((root / f"{name}{s}").read_text(errors="replace") if (root / f"{name}{s}").exists() else ""
                 for s in ("-gate.txt", ".out", ".log"))


def flags(gate: str) -> tuple[bool, list[str]]:
    """True when the run ran with rc 0, and each condition that makes the run not comparable."""
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    ok = "gate: OK" in gate and rc is not None and rc.group(1) == "0"
    out = []
    if not gate:
        out.append("no gate file")
    elif "gate: OK" not in gate:
        out.append("gate stopped the run")
    elif not ok:
        out.append(f"rc={rc.group(1) if rc else '?'}")
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        out.append(f"caps {before.group(3)}/{before.group(4)} -> {after.group(2)}/{after.group(3)}")
    if after and after.group(1) not in ("", "0"):
        out.append(f"thermal {after.group(1)}")
    return ok, out


def hashes(root: Path, name: str) -> list[tuple[str, str]]:
    """The HASH lines of one run as (what, hex) pairs, in order."""
    return re.findall(r"^HASH (.*) ([0-9a-f]{16})$", read(root, name)[1], re.M)


def table(root: Path, include_all: bool) -> int:
    """Print the results of each block."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    for b, rnd, v in runs():
        ok, fl = flags(read(root, f"{b.key}-{rnd}-{v.key}")[0])
        print(f"{b.key}-{rnd}-{v.key}: {'ok' if ok else 'NOT OK'}" + (f" [{', '.join(fl)}]" if fl else ""))
    print()

    for key, model in (("h", "4B"), ("i", "2B")):
        a, m = hashes(root, f"{key}-1-a"), hashes(root, f"{key}-1-m")
        same = sum(x == y for x, y in zip(a, m))
        print(f"{key} {model}: A {len(a)} and M {len(m)} HASH lines, {same} equal"
              + (" (the same hashes)" if a and a == m else " (DIFFERENT)"))
    print()

    print("KL against the naive base naive-4B-q8.kld: mean ± error of the mean, maximum, same top-1 ± error, PPL")
    for key in ("kp", "kd"):
        block = next(b for b in BLOCKS if b.key == key)
        means: dict[str, list[float]] = {}
        for k in block.variants:
            for rnd in range(1, block.rounds + 1):
                _, out, log = read(root, f"{key}-{rnd}-{k}")
                text = out + log
                m, t, mx, ppl = KLD_RE.search(text), TOP_RE.search(text), MAXKL_RE.search(text), PPL_RE.search(text)
                if m:
                    means.setdefault(k, []).append(float(m.group(1)))
                cell = (f"{m.group(1)} ± {m.group(2)}, max {mx.group(1) if mx else '?'}, "
                        f"top-1 {t.group(1) + ' ± ' + t.group(2) if t else '?'} %, PPL {ppl.group(1) if ppl else '?'}"
                        if m else "no KLD line")
                print(f"  {key}-{rnd}-{k} {VARIANTS[k].text:32s}: {cell}")
        if means.get("a") and means.get("f"):
            lo, hi = min(means["a"]), max(means["a"])
            fm = statistics.mean(means["f"])
            print(f"  {key}: A {lo:.6f} to {hi:.6f}, F mean {fm:.6f}, F - A {fm - statistics.mean(means['a']):+.6f}")
    print()

    rates: dict[tuple, dict[str, dict[int, float]]] = {}
    for b, rnd, v in runs():
        if b.tool != "llama-bench":
            continue
        gate, out, _ = read(root, f"{b.key}-{rnd}-{v.key}")
        ok, fl = flags(gate)
        if not ok or (fl and not include_all):
            continue
        for line in out.splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                key = (rec["n_prompt"], rec["n_gen"], rec["n_depth"])
                rates.setdefault(key, {}).setdefault(v.key, {})[rnd] = statistics.median(rec["samples_ts"])
    print("llama-bench t/s: the median of the rounds, and the median change to A over the paired rounds [range]"
          + ("" if include_all else " (--all also uses the runs with changed caps or heat)"))
    for key, per in sorted(rates.items()):
        cells = []
        for k in "amf":
            vals = per.get(k, {})
            if not vals:
                continue
            cell = f"{k.upper()} {statistics.median(vals.values()):.2f}"
            if k != "a" and "a" in per:
                ratios = [vals[r] / per["a"][r] for r in vals if r in per["a"]]
                if ratios:
                    cell += (f" ({100 * (statistics.median(ratios) - 1):+.2f}% "
                             f"[{100 * (min(ratios) - 1):+.2f}, {100 * (max(ratios) - 1):+.2f}], {len(ratios)} rounds)")
            cells.append(cell)
        print(f"  pp{key[0]} tg{key[1]} d{key[2]}: " + " | ".join(cells))
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the results from the pulled logs")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    t.add_argument("--all", action="store_true", help="also use the runs with changed caps or heat")
    a = ap.parse_args()
    if a.cmd == "commands":
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(runs())} runs")
        return 0
    return table(a.root, a.all)


if __name__ == "__main__":
    sys.exit(main())
