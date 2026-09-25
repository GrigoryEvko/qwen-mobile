#!/usr/bin/env python3
"""The phone stages "vit": the vision encoder of the 4B projector on HTP0, its speed, its op profile and its embeddings
against the naive x86 oracle.

Usage:
    stage.py commands STAGE [--out PATH]   write the phone command file (build/vit/STAGE/phone-commands.txt)
    stage.py table STAGE [--all]           print the times and the op profile from build/vit/STAGE/phone-out
    stage.py oracle STAGE [-t N]           encode each RGB file of the phone outputs with the x86 oracle (box only)
    stage.py cmp STAGE                     compare each embedding file of the phone outputs with the oracle

The tool of each run is tools/vit/vitprobe.cpp. The phone files of a set come from tools/stages/vit/build.sh, one
directory for each set in build/vit (for example phone-head, the series of HEAD). A run names its set, thus one stage
can hold an A/B pair of two library sets.

A run name is <set-key>-<variant>-<round>, for example h-t768-1. Each run writes <name>-gate.txt, <name>.out (stdout),
<name>.log (stderr) and, for the variants that write them, <name>.rgb (the input bytes) and <name>.f32 (the
embeddings) to the phone directory out/. A run goes into the time tables when its gate passed, its exit code is 0,
the thermal status after it is 0 and no CPU cap before or after it is less than CAP_MIN_KHZ. --all also uses the
other runs.

The oracle: build/vit/x86/vitprobe-oracle (tools/stages/vit/build.sh oracle) encodes the RGB file of each run on the
CPU of the box, in strict IEEE with no SIMD. The files go to build/vit/oracle/<rgb hash>-<W>x<H>.f32 and are made
once for each input. cmp gives, for each run: the NMSE (the sum of the squared errors over the sum of the squared
oracle values), the cosine of all values, the smallest cosine of one token row, and the largest error over the RMS of
the oracle.

This file is tools/stages/vit/stage.py. The files of the stages stay in build/vit. The time of the tables is
O(size of the files).
"""

import argparse
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/vit"
MODEL_DIR = "/data/local/tmp/qwen/models"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
MMPROJ = "Qwen3.5-4B-Q8_0.mmproj.gguf"
APP_IMAGE = "files/images/a2200d1a726ec0a8576b4a18dc2ef1aa4e4c797d.jpg"
OLD_IMAGE = "/sdcard/qwen/user.jpg"
GATE_KB = 3145728
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build"
REPO = Path(__file__).resolve().parents[3]
STAGE_ROOT = Path(os.path.relpath(REPO / "build/vit"))
BASE_ENV = "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
PROFILE_ENV = "GGML_HEXAGON_PROFILE=1 LLAMA_HOSTPROF=1"
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x vitprobe; echo pgrep-done'"
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')
CAP_MIN_KHZ = 3000000
# The tensors of the dump runs: the output of layers 0, 1, 5, 11, 17 and 23, and the stages of layer 0.
DUMP_RE = "(layer_out-(0|1|5|11|17|23)|ln1-0|QKcur_rope-0|kqv_out-0|attn_out-0|ffn_inp-0|ffn_out-0)"
STAGE_FILES = ("bin/gate.sh", "bin/vitprobe", "bin/test-backend-ops", "lib/libggml-base.so", "lib/libggml-cpu.so",
               "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so",
               "lib/libllama-common.so", "lib/libllama.so", "lib/libmtmd.so")


@dataclass(frozen=True)
class Variant:
    """One kind of run: the image, the token budget, the device, the reps, the extra arguments and environment."""
    key: str
    text: str
    image: str
    tokens: int
    dev: str = "HTP0"
    reps: int = 5
    env: str = ""
    args: str = ""
    embd: bool = True
    dump: bool = False
    timing: bool = True
    limit: int = 100


@dataclass(frozen=True)
class Stage:
    """A phone stage: its header text, its library sets (key -> directory in build/vit) and its runs in order."""
    name: str
    text: str
    sets: dict
    runs: list = field(default_factory=list)


V = {v.key: v for v in (
    Variant("t768", "the photo of the user, 768 tokens (Detailed), 5 encodes", "photo", 768),
    Variant("t256", "the photo of the user, 256 tokens (Fast), 5 encodes", "photo", 256),
    Variant("p768", "the op profile, 768 tokens, 2 encodes", "photo", 768, reps=2, env=PROFILE_ENV, args="--log-ts",
            embd=False, timing=False),
    Variant("p256", "the op profile, 256 tokens, 2 encodes", "photo", 256, reps=2, env=PROFILE_ENV, args="--log-ts",
            embd=False, timing=False),
    Variant("c768", "the phone CPU, 768 tokens, 1 encode", "photo", 768, dev="none", reps=1, args="-t 8",
            timing=False, limit=110),
    Variant("d256", "the tensor dump, 256 tokens, 1 encode", "photo", 256, reps=1, dump=True, embd=False,
            timing=False),
    Variant("u768", "the older test photo, 768 tokens, 2 encodes", "user", 768, reps=2, timing=False),
)}


def runs_ab(set_keys: str, variants: list[str], rounds: int) -> list[tuple[str, str, int]]:
    """The runs of an A/B block: in round r each variant runs once with each set, the sets in their order in an odd
    round and reversed in an even round. O(rounds * sets * variants)."""
    out = []
    for r in range(1, rounds + 1):
        for v in variants:
            order = set_keys if r % 2 else set_keys[::-1]
            out.extend((s, v, r) for s in order)
    return out


STAGES = {s.name: s for s in (
    Stage("vit1", "the encoder of HEAD: its speed at 768 and 256 tokens, its op profile, its embeddings, the phone CPU "
          "embeddings, and a tensor dump", {"h": "phone-head"},
          runs_ab("h", ["t768", "t256"], 2) + [("h", "p768", 1), ("h", "p256", 1), ("h", "u768", 1),
                                              ("h", "d256", 1), ("h", "c768", 1)]),
)}


def run_name(set_key: str, vk: str, rnd: int) -> str:
    """The run name, which is also the stem of its output files."""
    return f"{set_key}-{vk}-{rnd}"


def image_path(key: str) -> str:
    """The phone path of an input image."""
    return f"{PHONE}/in/{key}.jpg"


def run_lines(stage: Stage, set_key: str, vk: str, rnd: int) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    v = V[vk]
    name = run_name(set_key, vk, rnd)
    stem = f"{PHONE}/out/{name}"
    sdir = f"{PHONE}/{stage.sets[set_key]}"
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={sdir}/lib ADSP_LIBRARY_PATH={sdir}/lib", BASE_ENV, v.env) if x)
    args = (f"-m {MODEL_DIR}/{MODEL} --mmproj $P --image {image_path(v.image)} --image-tokens {v.tokens} "
            f"--dev {v.dev} --reps {v.reps} --rgb-out {stem}.rgb")
    if v.embd:
        args += f" --embd-out {stem}.f32"
    if v.dump:
        args += f" --dump {stem}-dump --dump-re \"{DUMP_RE}\""
    if v.args:
        args += f" {v.args}"
    pre = f"mkdir -p {stem}-dump && " if v.dump else ""
    cmd = (f"{pre}sh {sdir}/bin/gate.sh {GATE_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"P={MODEL_DIR}/{MMPROJ}; [ -f $P ] || P=/sdcard/qwen/models/{MMPROJ}; echo \"mmproj: $P\" >> {stem}-gate.txt; "
           f"timeout -s KILL {v.limit} env {env} {sdir}/bin/vitprobe {args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt {stem}.out")
    return ["#", f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {name}, set {stage.sets[set_key]}, {v.text}",
            THERMAL, f"{ADB} shell '{cmd}'", PGREP]


def header(stage: Stage) -> list[str]:
    """The header comment of the command file."""
    n_timing = sum(1 for _, vk, _ in stage.runs if V[vk].timing)
    runs = Counter(vk for _, vk, _ in stage.runs)
    lines = [
        f"Phone stage \"{stage.name}\": {stage.text}.",
        "",
        "The tool: build/vit/<set>/bin/vitprobe (tools/vit/vitprobe.cpp) with the libraries of its set. It loads the",
        "vocabulary of the 4B and the projector Qwen3.5-4B-Q8_0.mmproj.gguf on the device of the run, with the",
        "parameters of the app (no warmup, flash attention AUTO, 4 threads, the fusion switches of the app), and",
        "encodes the image. The image of the user comes from the app with run-as; the older test photo is",
        f"{OLD_IMAGE}. The tool resizes the image to the target size with an exact integer filter and writes the",
        "RGB bytes, thus the box encodes the same bytes with the x86 oracle.",
        "",
        "The sets: " + ", ".join(f"{k} = build/vit/{d}" for k, d in stage.sets.items()) + ".",
        f"The runs, {len(stage.runs)}: " + ", ".join(f"{vk} x{n} ({V[vk].text})" for vk, n in runs.items()) + ".",
        "Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,",
        f"MemAvailable {GATE_KB // 1048576} GB, the caps), the tool under timeout -s KILL (at most 110 s), the exit",
        "code and the conditions after the run, then the pgrep line.",
        "",
    ]
    if n_timing:
        lines += ["This is a TIMING stage: put this file into /tmp/phone-timing-stages.txt (unlocked phone, screen on,",
                  "no charger)."]
    lines += [
        "Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Tool time: about 5 minutes, plus about 8 s of",
        "gate and checks for each run, plus the waits for thermal status 0.",
        f"Then on the box: tools/stages/vit/stage.py table {stage.name}; stage.py oracle {stage.name}; "
        f"stage.py cmp {stage.name}.",
    ]
    return ["# " + x if x else "#" for x in lines]


def setup_lines(stage: Stage) -> list[str]:
    """The lines that copy the phone files of each set and the two images to the phone and check them."""
    out = []
    for d in stage.sets.values():
        local = f"build/vit/{d}"
        out += [f"mkdir -p {local} && rsync -a --delete {BOX}/vit/{d}/ {local}/",
                f"(cd {local} && sha256sum -c SHA256SUMS)"]
    # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
    out += [f"{ADB} shell 'ls -l {MODEL_DIR} /sdcard/qwen/models | grep -E \"Qwen3.5-4B-Q8_0(.mmproj)?.gguf\"'",
            f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/in {PHONE}/out'",
            # run-as writes the file of the app to stdout, and the shell of adb writes it into the stage directory.
            f"{ADB} shell 'run-as ai.airi.qwenmobile cat {APP_IMAGE} > {image_path('photo')} && "
            f"cp {OLD_IMAGE} {image_path('user')} && ls -l {PHONE}/in && sha1sum {PHONE}/in/*'"]
    for d in stage.sets.values():
        local = f"build/vit/{d}"
        bins = " ".join(f"{local}/{f}" for f in STAGE_FILES if f.startswith("bin/"))
        libs = " ".join(f"{local}/{f}" for f in STAGE_FILES if f.startswith("lib/"))
        out += [f"{ADB} shell 'mkdir -p {PHONE}/{d}/bin {PHONE}/{d}/lib'",
                f"{ADB} push {bins} {PHONE}/{d}/bin/",
                f"{ADB} push {libs} {PHONE}/{d}/lib/",
                f"{ADB} push {local}/SHA256SUMS {PHONE}/{d}/",
                f"{ADB} shell 'cd {PHONE}/{d} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/{d}/bin/*'"]
    return out


def output_lines(stage: Stage) -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone directory goes
    only when the pull has each of its files."""
    local = f"build/vit/{stage.name}/phone-out"
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'pgrep -x vitprobe; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {local} && mkdir -p build/vit/{stage.name}",
        f"{ADB} pull {PHONE}/out {local}",
        f"rsync -a --delete {local}/ {BOX}/vit/{stage.name}/phone-out/",
        f"test \"$(find {local} -type f | wc -l)\" -eq \"$({ADB} shell 'find {PHONE}/out -type f | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_commands(stage: Stage, path: Path) -> int:
    """Write the command file and return its line count."""
    lines = header(stage) + setup_lines(stage)
    lines += ["#", f"# ==== {len(stage.runs)} runs ===="]
    for s, vk, r in stage.runs:
        lines += run_lines(stage, s, vk, r)
    lines += output_lines(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The parser of the files ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
TIME_RE = re.compile(r"^TIME encode rep=(\d+) ms=([\d.]+)", re.M)
STAMP_RE = re.compile(r"vitprobe: STAMP (\S+) rep=(\d+)")
# One op of the profile of the Hexagon backend: the op name, the names, the dims, the types, then the time.
OP_RE = re.compile(r"profile-op ([A-Z0-9_+]+)\|([^|]*)\|([^|]*)\|([^|]*)\|.*\|usec (\d+) cycles")
OPBATCH_RE = re.compile(r"profile-op OPBATCH\|.*\|n-ops (\d+)\|.*\|usec (\d+) cycles")
SCHED_RE = re.compile(r"hostprof: sched splits (\d+)(.*) us")
SPLIT_RE = re.compile(r"\[(\S+) nodes (\d+) inputs \d+ copy (\d+) compute (\d+)\]")
EMBD_RE = re.compile(r"^EMBD rows=(\d+) cols=(\d+) hash=(\w+) nonfinite=(\d+)", re.M)
RGB_RE = re.compile(r"^RGB size=(\d+)x(\d+) hash=(\w+)", re.M)


@dataclass
class Result:
    """The files of one run."""
    name: str
    ok: bool
    removed: list
    caps: str
    out: str
    log: str


def read_result(root: Path, name: str) -> Result:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    gate_path, out_path, log_path = (root / f"{name}{s}" for s in ("-gate.txt", ".out", ".log"))
    gate = gate_path.read_text(errors="replace") if gate_path.exists() else ""
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    ok = "gate: OK" in gate and rc is not None and rc.group(1) == "0"
    removed = []
    if not gate:
        removed.append("no gate file")
    elif "gate: OK" not in gate:
        removed.append("the gate stopped the run")
    elif not ok:
        removed.append(f"exit code {rc.group(1) if rc else '?'}")
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    cap_values = [int(v) for v in ((before.group(3), before.group(4)) if before else ()) +
                  ((after.group(2), after.group(3)) if after else ()) if v]
    if cap_values and min(cap_values) < CAP_MIN_KHZ:
        removed.append(f"a cap of {min(cap_values)} kHz")
    if after and after.group(1) not in ("", "0"):
        removed.append(f"thermal {after.group(1)} after the run")
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    return Result(name, ok, removed, caps, out, log)


def fmt(x, digits: int = 1) -> str:
    """A number, or a dash for None."""
    return "-" if x is None else f"{x:.{digits}f}"


def time_table(stage: Stage, results: dict[str, Result], include_all: bool) -> list[str]:
    """The encode times of each set and timing variant: the first encode (the reserve and the first use) and the
    median of the other encodes of each run, then the median over the runs."""
    out = ["Encode times, ms (rep 1: the first encode of the process; reps 2+: the median of the later encodes)"]
    for vk in dict.fromkeys(vk for _, vk, _ in stage.runs):
        if not V[vk].timing:
            continue
        for s in stage.sets:
            rs = [results[run_name(s, vk, r)] for (s2, v2, r) in stage.runs if s2 == s and v2 == vk
                  and run_name(s, vk, r) in results]
            rs = [r for r in rs if r.ok and (include_all or not r.removed)]
            firsts, warms, per_run = [], [], []
            for r in rs:
                t = {int(a): float(b) for a, b in TIME_RE.findall(r.out)}
                if 1 in t:
                    firsts.append(t[1])
                w = [v for k, v in t.items() if k > 1]
                if w:
                    warms.append(statistics.median(w))
                    per_run.append(f"{statistics.median(w):.1f}")
            first = statistics.median(firsts) if firsts else None
            warm = statistics.median(warms) if warms else None
            out.append(f"  {vk:5s} {stage.sets[s]:12s} runs {len(rs)}  first {fmt(first):>8}  warm {fmt(warm):>8}"
                       f"  (runs: {', '.join(per_run)})")
    return out


def profile_table(results: dict[str, Result], name: str) -> list[str]:
    """The op profile of the last encode of one profile run: the ops by name with their count, time and share, the
    ten largest single op forms, the DSP batches and the scheduler splits."""
    res = results.get(name)
    if res is None or not res.ok:
        return [f"{name}: no profile run"]
    lines = res.log.splitlines()
    windows, start = {}, None
    for i, line in enumerate(lines):
        m = STAMP_RE.search(line)
        if not m:
            continue
        if m.group(1) == "encode-begin":
            start = i
        elif m.group(1) == "encode-end" and start is not None:
            windows[int(m.group(2))] = (start, i)
            start = None
    out = []
    for rep in sorted(windows):
        a, b = windows[rep]
        ops_t, ops_n, forms, busy, nbatch, nops_b, sched = Counter(), Counter(), Counter(), 0, 0, 0, Counter()
        for line in lines[a:b + 1]:
            ms = SCHED_RE.search(line)
            if ms:
                sched["graphs"] += 1
                sched["splits"] += int(ms.group(1))
                for sp in SPLIT_RE.finditer(ms.group(2)):
                    side = "cpu" if sp.group(1) == "CPU" else "dev"
                    sched[f"{side}_nodes"] += int(sp.group(2))
                    sched[f"{side}_us"] += int(sp.group(3)) + int(sp.group(4))
                continue
            mb = OPBATCH_RE.search(line)
            if mb:
                nbatch += 1
                nops_b += int(mb.group(1))
                busy += int(mb.group(2))
                continue
            mo = OP_RE.search(line)
            if mo:
                op, dims, types, us = mo.group(1), mo.group(3), mo.group(4), int(mo.group(5))
                ops_t[op] += us
                ops_n[op] += 1
                forms[f"{op} {dims} {types}"] += us
        total = sum(ops_t.values())
        out.append(f"{name}, encode {rep}: DSP busy {busy / 1000:.1f} ms in {nbatch} batches of {nops_b} ops, "
                   f"op sum {total / 1000:.1f} ms")
        out.append(f"  {'op':24s} {'count':>6} {'ms':>9} {'share':>7} {'us/op':>8}")
        for op, us in ops_t.most_common():
            out.append(f"  {op:24s} {ops_n[op]:6d} {us / 1000:9.2f} {100.0 * us / max(total, 1):6.1f}% "
                       f"{us / ops_n[op]:8.0f}")
        out.append("  the largest op forms (op, dims, types): ms")
        for form, us in forms.most_common(12):
            out.append(f"    {us / 1000:8.2f}  {form}")
        out.append(f"  scheduler: {sched['graphs']} graphs, {sched['splits']} splits, CPU {sched['cpu_nodes']} nodes "
                   f"{sched['cpu_us'] / 1000:.1f} ms, device {sched['dev_nodes']} nodes {sched['dev_us'] / 1000:.1f} ms")
    return out


def checks(stage: Stage, results: dict[str, Result]) -> list[str]:
    """The conditions of the runs."""
    names = [run_name(*r) for r in stage.runs]
    got = [results[n] for n in names if n in results]
    out = [f"{stage.name}: {len(got)} of {len(names)} runs have a gate file, {sum(r.ok for r in got)} ran with exit "
           f"code 0, {sum(r.ok and not r.removed for r in got)} have no mark"]
    caps = Counter(r.caps for r in got if r.ok)
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.most_common()))
    for r in got:
        if r.removed:
            out.append(f"  {r.name}: {', '.join(r.removed)}")
        for m in EMBD_RE.finditer(r.out):
            if int(m.group(4)) != 0:
                out.append(f"  {r.name}: {m.group(4)} embedding values are not finite")
        rep = [float(x) for x in re.findall(r"^REPDIFF rep=\d+ maxabs=(\S+)", r.out, re.M)]
        if rep and max(rep) != 0.0:
            out.append(f"  {r.name}: the encodes of one process differ, max {max(rep):.3g}")
    return out


def load_results(stage: Stage) -> tuple[Path, dict[str, Result]]:
    """The results of the runs that have a gate file."""
    root = STAGE_ROOT / stage.name / "phone-out"
    if not root.is_dir():
        sys.exit(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.")
    names = [run_name(*r) for r in stage.runs]
    return root, {n: read_result(root, n) for n in names if (root / f"{n}-gate.txt").exists()}


def table(stage: Stage, include_all: bool) -> int:
    """Print the tables."""
    _, results = load_results(stage)
    parts = [checks(stage, results), time_table(stage, results, include_all)]
    for s, vk, r in stage.runs:
        if V[vk].env == PROFILE_ENV:
            parts.append(profile_table(results, run_name(s, vk, r)))
    for part in parts:
        print("\n".join(part))
        print()
    return 0


def rgb_inputs(stage: Stage, results: dict[str, Result]) -> dict[str, tuple[int, int, str]]:
    """For each run: the size and the hash of its RGB input."""
    out = {}
    for name, r in results.items():
        m = RGB_RE.search(r.out)
        if m:
            out[name] = (int(m.group(1)), int(m.group(2)), m.group(3))
    return out


def oracle_path(w: int, h: int, hsh: str) -> Path:
    """The oracle embeddings of one input."""
    return STAGE_ROOT / "oracle" / f"{hsh}-{w}x{h}.f32"


def oracle(stage: Stage, threads: int) -> int:
    """Encode the RGB input of each run with the x86 oracle, once for each distinct input. O(inputs * encode)."""
    root, results = load_results(stage)
    tool = STAGE_ROOT / "x86" / "vitprobe-oracle"
    if not tool.exists():
        sys.exit(f"stage.py: {tool} does not exist. Run tools/stages/vit/build.sh oracle.")
    done = set()
    for name, (w, h, hsh) in sorted(rgb_inputs(stage, results).items()):
        dst = oracle_path(w, h, hsh)
        if hsh in done or dst.exists():
            done.add(hsh)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        cmd = [str(tool), "-m", str(REPO / "weights/gguf" / MODEL), "--mmproj", str(REPO / "weights/gguf" / MMPROJ),
               "--rgb", str(root / f"{name}.rgb"), "--size", f"{w}x{h}", "--dev", "none", "-t", str(threads),
               "--reps", "1", "--embd-out", str(tmp)]
        print(f"oracle: {name} {w}x{h} {hsh}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        (dst.parent / f"{hsh}-{w}x{h}.out").write_text(p.stdout)
        (dst.parent / f"{hsh}-{w}x{h}.log").write_text(p.stderr)
        if p.returncode != 0:
            print(f"oracle: {name} failed with {p.returncode}, refer to {dst.parent}/{hsh}-{w}x{h}.log")
            continue
        m = RGB_RE.search(p.stdout)
        if m is None or m.group(3) != hsh:
            print(f"oracle: {name}: the oracle read other bytes than the phone ({m.group(3) if m else '?'})")
            continue
        tmp.rename(dst)
        done.add(hsh)
    return 0


def compare(ref: np.ndarray, x: np.ndarray, cols: int) -> dict:
    """The error measures of x against ref: NMSE, the cosine of all values, the smallest cosine of one row, the
    largest error over the RMS of ref, and the count of values that are not finite. O(n)."""
    ref64, x64 = ref.astype(np.float64), x.astype(np.float64)
    bad = int(np.count_nonzero(~np.isfinite(x64)))
    x64 = np.where(np.isfinite(x64), x64, 0.0)
    err = x64 - ref64
    nmse = float(np.sum(err * err) / np.sum(ref64 * ref64))
    cos = float(np.dot(x64, ref64) / (np.linalg.norm(x64) * np.linalg.norm(ref64)))
    r2, x2 = ref64.reshape(-1, cols), x64.reshape(-1, cols)
    row_cos = np.sum(r2 * x2, axis=1) / (np.linalg.norm(r2, axis=1) * np.linalg.norm(x2, axis=1) + 1e-30)
    rms = math.sqrt(float(np.mean(ref64 * ref64)))
    return {"nmse": nmse, "cos": cos, "rowcos_min": float(np.min(row_cos)), "rowcos_mean": float(np.mean(row_cos)),
            "maxerr_rms": float(np.max(np.abs(err))) / rms, "nonfinite": bad}


def cmp(stage: Stage) -> int:
    """Compare the embeddings of each run with the oracle embeddings of its input."""
    root, results = load_results(stage)
    inputs = rgb_inputs(stage, results)
    print(f"{'run':14s} {'input':26s} {'NMSE':>10} {'cos':>12} {'rowcos min':>11} {'rowcos mean':>12} "
          f"{'maxerr/rms':>10} {'nonfinite':>9}")
    for name in sorted(inputs):
        f = root / f"{name}.f32"
        if not f.exists():
            continue
        w, h, hsh = inputs[name]
        ref_path = oracle_path(w, h, hsh)
        if not ref_path.exists():
            print(f"{name:14s} no oracle file {ref_path}")
            continue
        m = EMBD_RE.search(results[name].out)
        cols = int(m.group(2)) if m else 2560
        ref, x = np.fromfile(ref_path, dtype=np.float32), np.fromfile(f, dtype=np.float32)
        if ref.size != x.size:
            print(f"{name:14s} {x.size} values against {ref.size} of the oracle")
            continue
        c = compare(ref, x, cols)
        print(f"{name:14s} {hsh[:12]}-{w}x{h:<9} {c['nmse']:10.3e} {c['cos']:12.9f} {c['rowcos_min']:11.7f} "
              f"{c['rowcos_mean']:12.9f} {c['maxerr_rms']:10.4f} {c['nonfinite']:9d}")
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("commands", "table", "oracle", "cmp"):
        p = sub.add_parser(c)
        p.add_argument("stage", choices=sorted(STAGES))
        if c == "commands":
            p.add_argument("--out", type=Path)
        if c == "table":
            p.add_argument("--all", action="store_true", help="also use the runs with a mark")
        if c == "oracle":
            p.add_argument("-t", type=int, default=64, help="the CPU threads of the oracle")
    a = ap.parse_args()
    stage = STAGES[a.stage]
    if a.cmd == "commands":
        path = a.out or STAGE_ROOT / stage.name / "phone-commands.txt"
        n = write_commands(stage, path)
        print(f"{path}: {n} lines, {len(stage.runs)} runs")
        return 0
    if a.cmd == "table":
        return table(stage, a.all)
    if a.cmd == "oracle":
        return oracle(stage, a.t)
    return cmp(stage)


if __name__ == "__main__":
    sys.exit(main())
