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
# The kernel keeps 15 characters of a process name: test-backend-ops is test-backend-op
PGREP = f"{ADB} shell 'pgrep -x vitprobe; pgrep -x llama-bench; pgrep -x test-backend-op; echo pgrep-done'"
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
# The llama-bench runs of the text model: the flags of the app (4 threads, the Q8_0 cache, flash attention)
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -ctk q8_0 -ctv q8_0 -fa 1 -p 512 -n 32 -r 3"
BENCH_GATE_KB = 8388608
# The switch of the plans of the vision encoder (htp-vit-fusion.h). The HEAD libraries ignore it.
VIT_OFF = "GGML_HEXAGON_FUSE_VIT=0"


@dataclass(frozen=True)
class Variant:
    """One kind of run: the tool (vit: vitprobe, tbo: test-backend-ops, bench: llama-bench), the image, the token
    budget, the device, the reps, the extra arguments and environment."""
    key: str
    text: str
    image: str = "photo"
    tokens: int = 0
    dev: str = "HTP0"
    reps: int = 5
    env: str = ""
    args: str = ""
    embd: bool = True
    dump: bool = False
    timing: bool = True
    limit: int = 100
    tool: str = "vit"


@dataclass(frozen=True)
class Stage:
    """A phone stage: its header text, its library sets (key -> directory in build/vit), its runs in order, and the
    set whose bin/ holds llama-bench (the bench of each set runs that binary with the libraries of its set)."""
    name: str
    text: str
    sets: dict
    runs: list = field(default_factory=list)
    tools_set: str = ""
    minutes: int = 5   # the tool time of the runs, for the header


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
    Variant("x768", "768 tokens with the vision plans off (GGML_HEXAGON_FUSE_VIT=0)", "photo", 768, env=VIT_OFF),
    Variant("x256", "256 tokens with the vision plans off", "photo", 256, env=VIT_OFF),
    Variant("ov", "test-backend-ops test -o VIT_BLOCK (the fused chains of the encoder)", tool="tbo",
            args="-o VIT_BLOCK", timing=False, embd=False),
    Variant("ov0", "test-backend-ops test -o VIT_BLOCK with the vision plans off", tool="tbo", args="-o VIT_BLOCK",
            env=VIT_OFF, timing=False, embd=False),
    Variant("on", "test-backend-ops test -o NORM,GELU,FFN_SWIGLU", tool="tbo", args="-o NORM,GELU,FFN_SWIGLU",
            timing=False, embd=False),
    Variant("or", "test-backend-ops test -o ROPE", tool="tbo", args="-o ROPE", timing=False, embd=False),
    Variant("oc", "test-backend-ops test -o CPY", tool="tbo", args="-o CPY", timing=False, embd=False),
    Variant("b", "llama-bench of the 4B Q8_0 text model, pp512 and tg32, 3 reps", tool="bench", timing=True,
            embd=False, limit=110),
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
    Stage("vit2", "the fused LayerNorm, the fused attention input and the F16 activations of the encoder "
          "(candidate a) against HEAD: the op tests, the encoder speed and embeddings, the op profile, the text bench",
          {"h": "phone-head", "a": "phone-a"},
          [("a", "ov", 1), ("a", "ov0", 1), ("a", "on", 1), ("a", "or", 1), ("a", "oc", 1)] +
          runs_ab("ha", ["t768", "t256"], 2) + [("a", "x768", 1), ("a", "x256", 1)] +
          [("a", "p768", 1), ("a", "u768", 1)] + runs_ab("ha", ["b"], 2),
          tools_set="a", minutes=15),
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
    libs = f"{sdir}/lib"
    gate_kb = GATE_KB
    pre = ""
    if v.tool == "vit":
        args = (f"-m {MODEL_DIR}/{MODEL} --mmproj $P --image {image_path(v.image)} --image-tokens {v.tokens} "
                f"--dev {v.dev} --reps {v.reps} --rgb-out {stem}.rgb")
        if v.embd:
            args += f" --embd-out {stem}.f32"
        if v.dump:
            args += f" --dump {stem}-dump --dump-re \"{DUMP_RE}\""
            pre = f"mkdir -p {stem}-dump && "
        if v.args:
            args += f" {v.args}"
        tool = f"{sdir}/bin/vitprobe {args}"
    elif v.tool == "tbo":
        tool = f"{sdir}/bin/test-backend-ops test -b HTP0 {v.args}"
    else:
        # llama-bench of the tools set, with the libraries of this set first in the search path
        tdir = f"{PHONE}/{stage.sets[stage.tools_set]}"
        libs = f"{sdir}/lib:{tdir}/lib"
        gate_kb = BENCH_GATE_KB
        tool = f"{tdir}/bin/llama-bench -m {MODEL_DIR}/{MODEL} {BENCH_ARGS}"
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={libs} ADSP_LIBRARY_PATH={sdir}/lib", BASE_ENV, v.env) if x)
    cmd = (f"{pre}sh {sdir}/bin/gate.sh {gate_kb} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"P={MODEL_DIR}/{MMPROJ}; [ -f $P ] || P=/sdcard/qwen/models/{MMPROJ}; echo \"mmproj: $P\" >> {stem}-gate.txt; "
           f"timeout -s KILL {v.limit} env {env} {tool} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt; tail -n 12 {stem}.out")
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
        f"Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Tool time: about {stage.minutes} minutes, plus "
        "about 8 s of",
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
    # Each file of the bin/ and lib/ directories of a set (the SHA256SUMS of the set names them all)
    for d in stage.sets.values():
        local = f"build/vit/{d}"
        out += [f"{ADB} shell 'mkdir -p {PHONE}/{d}/bin {PHONE}/{d}/lib'",
                f"{ADB} push {local}/bin/. {PHONE}/{d}/bin/",
                f"{ADB} push {local}/lib/. {PHONE}/{d}/lib/",
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
        if not V[vk].timing or V[vk].tool != "vit":
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


def runner_marks(stage: Stage) -> dict[str, list[str]]:
    """The marks of the runner log build/vit/STAGE/runner.log (the copy of the log of the laptop runner): a run whose
    CAPS line says CAPS-CHANGED. O(size of the log)."""
    path = STAGE_ROOT / stage.name / "runner.log"
    marks: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        return marks
    current = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"# REAL-MODEL [^:]*: (\S+),", line)
        if m:
            current = m.group(1)
        elif current and line.startswith("CAPS ") and "CAPS-CHANGED" in line:
            marks[current].append("the runner marks CAPS-CHANGED")
    return marks


TBO_CASE_RE = re.compile(r"^\s+([A-Z_0-9]+)\((.*)\): (OK|FAIL|not supported)", re.M)
TBO_SUM_RE = re.compile(r"(\d+)/(\d+) tests passed")
BENCH_RE = re.compile(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|")


def tbo_table(stage: Stage, results: dict[str, Result]) -> list[str]:
    """The op test runs: the passed, failed and unsupported cases of each run, and each failed case."""
    out = ["test-backend-ops (HTP0 against the phone CPU):"]
    for s, vk, r in stage.runs:
        if V[vk].tool != "tbo":
            continue
        name = run_name(s, vk, r)
        res = results.get(name)
        if res is None:
            out.append(f"  {name}: no files")
            continue
        cases = TBO_CASE_RE.findall(res.out)
        cnt = Counter(c[2] for c in cases)
        summ = TBO_SUM_RE.findall(res.out)
        mark = "" if res.ok else f"  ({', '.join(res.removed) or 'not ok'})"
        out.append(f"  {name:10s} {V[vk].text}: OK {cnt['OK']}, FAIL {cnt['FAIL']}, not supported {cnt['not supported']}"
                   f"{', summary ' + '/'.join(summ[-1]) if summ else ', no summary line'}{mark}")
        for op, params, st in cases:
            if st == "FAIL":
                out.append(f"      FAIL {op}({params[:150]})")
    return out


def bench_table(stage: Stage, results: dict[str, Result], include_all: bool) -> list[str]:
    """The llama-bench runs: the median of the rounds of each test for each set, and the ratio to the first set."""
    rates: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s, vk, r in stage.runs:
        if V[vk].tool != "bench":
            continue
        res = results.get(run_name(s, vk, r))
        if res is None or not res.ok or (res.removed and not include_all):
            continue
        for test, mean, _sd in BENCH_RE.findall(res.out):
            rates[test][s].append(float(mean))
    if not rates:
        return ["llama-bench: no usable run"]
    keys = list(stage.sets)
    out = ["llama-bench of the 4B Q8_0 (t/s, the median of the rounds): " + ", ".join(f"{k} = {stage.sets[k]}" for k in keys)]
    for test, by_set in rates.items():
        cells = []
        base = statistics.median(by_set[keys[0]]) if by_set.get(keys[0]) else None
        for k in keys:
            if by_set.get(k):
                m = statistics.median(by_set[k])
                ratio = f" ({m / base:.3f})" if base and k != keys[0] else ""
                cells.append(f"{k} {m:.2f}{ratio} [{', '.join(f'{x:.1f}' for x in by_set[k])}]")
        out.append(f"  {test:6s} " + "   ".join(cells))
    return out


def hash_table(stage: Stage, results: dict[str, Result]) -> list[str]:
    """The embedding hash of each encoder run, grouped by the input: equal hashes are equal bits."""
    groups: dict[str, list[str]] = defaultdict(list)
    for name, res in results.items():
        m, g = EMBD_RE.search(res.out), RGB_RE.search(res.out)
        if m and g:
            groups[f"{g.group(3)}-{g.group(1)}x{g.group(2)}"].append(f"{name}={m.group(3)}")
    out = ["The embedding hashes of each input (runs with equal hashes have equal bits):"]
    for key, runs in sorted(groups.items()):
        hashes = Counter(x.split("=")[1] for x in runs)
        out.append(f"  {key}: {len(hashes)} distinct hash(es): " + ", ".join(sorted(runs)))
    return out


def load_results(stage: Stage) -> tuple[Path, dict[str, Result]]:
    """The results of the runs that have a gate file, with the marks of the runner log."""
    root = STAGE_ROOT / stage.name / "phone-out"
    if not root.is_dir():
        sys.exit(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.")
    names = [run_name(*r) for r in stage.runs]
    results = {n: read_result(root, n) for n in names if (root / f"{n}-gate.txt").exists()}
    for n, marks in runner_marks(stage).items():
        if n in results:
            results[n].removed.extend(marks)
    return root, results


def table(stage: Stage, include_all: bool) -> int:
    """Print the tables."""
    _, results = load_results(stage)
    parts = [checks(stage, results), time_table(stage, results, include_all), hash_table(stage, results)]
    if any(V[vk].tool == "tbo" for _, vk, _ in stage.runs):
        parts.append(tbo_table(stage, results))
    if any(V[vk].tool == "bench" for _, vk, _ in stage.runs):
        parts.append(bench_table(stage, results, include_all))
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


# The two oracles. "on" is the graph of the phone (flash attention); the CPU flash attention accumulates P x V in
# F16 when V is F16 (ops.cpp, ggml_vec_mad_f16), thus its error grows with the key count. "off" is the attention of
# two F32 matmuls and an F32 softmax, the more exact reference.
ORACLE_FA = ("off", "on")


def oracle_path(w: int, h: int, hsh: str, fa: str, suffix: str = ".f32") -> Path:
    """The oracle file of one input and one attention form."""
    return STAGE_ROOT / "oracle" / f"{hsh}-{w}x{h}-fa{fa}{suffix}"


def oracle_run(name: str, rgb: Path, w: int, h: int, hsh: str, fa: str, threads: int, dump: bool) -> bool:
    """Encode one RGB file with the x86 oracle: the embeddings, or with dump the tensors of DUMP_RE into a directory.
    Returns False on a failure."""
    tool = STAGE_ROOT / "x86" / "vitprobe-oracle"
    if not tool.exists():
        sys.exit(f"stage.py: {tool} does not exist. Run tools/stages/vit/build.sh oracle.")
    dst = oracle_path(w, h, hsh, fa, "-dump" if dump else ".f32")
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    cmd = [str(tool), "-m", str(REPO / "weights/gguf" / MODEL), "--mmproj", str(REPO / "weights/gguf" / MMPROJ),
           "--rgb", str(rgb), "--size", f"{w}x{h}", "--dev", "none", "-t", str(threads), "--reps", "1", "--fa", fa]
    if dump:
        tmp.mkdir(parents=True, exist_ok=True)
        cmd += ["--dump", str(tmp), "--dump-re", DUMP_RE]
    else:
        cmd += ["--embd-out", str(tmp)]
    print(f"oracle: {name} {w}x{h} {hsh} fa {fa}{' dump' if dump else ''}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = oracle_path(w, h, hsh, fa, "-dump.log" if dump else ".log")
    log.write_text(p.stdout + p.stderr)
    m = RGB_RE.search(p.stdout)
    if p.returncode != 0 or m is None or m.group(3) != hsh:
        print(f"oracle: {name} failed (exit {p.returncode}, input {m.group(3) if m else '?'}), refer to {log}")
        return False
    tmp.rename(dst)
    return True


def oracle(stage: Stage, threads: int) -> int:
    """Encode the RGB input of each run with the two x86 oracles, once for each distinct input and form, and the
    tensors of each dump run. O(inputs * encode)."""
    root, results = load_results(stage)
    for name, (w, h, hsh) in sorted(rgb_inputs(stage, results).items()):
        for fa in ORACLE_FA:
            oracle_run(name, root / f"{name}.rgb", w, h, hsh, fa, threads, False)
            if (root / f"{name}-dump").is_dir():
                oracle_run(name, root / f"{name}.rgb", w, h, hsh, fa, threads, True)
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


CMP_HEAD = (f"{'run':14s} {'ref':6s} {'input':26s} {'NMSE':>10} {'cos':>12} {'rowcos min':>11} {'rowcos mean':>12} "
            f"{'maxerr/rms':>10} {'nonfinite':>9}")


def cmp_line(name: str, ref_name: str, label: str, ref: np.ndarray, x: np.ndarray, cols: int) -> str:
    """One line of the comparison of x with ref."""
    if ref.size != x.size:
        return f"{name:14s} {ref_name:6s} {x.size} values against {ref.size} of the reference"
    c = compare(ref, x, cols)
    return (f"{name:14s} {ref_name:6s} {label:26s} {c['nmse']:10.3e} {c['cos']:12.9f} {c['rowcos_min']:11.7f} "
            f"{c['rowcos_mean']:12.9f} {c['maxerr_rms']:10.4f} {c['nonfinite']:9d}")


def cmp(stage: Stage) -> int:
    """Compare the embeddings of each run with the two oracles of its input, then the two oracles with each other,
    then each tensor of a dump run with the tensors of the two oracles."""
    root, results = load_results(stage)
    inputs = rgb_inputs(stage, results)
    print("The embeddings against the oracle with the F32 attention (fa off) and with the flash attention (fa on):")
    print(CMP_HEAD)
    seen = set()
    for name in sorted(inputs):
        w, h, hsh = inputs[name]
        label = f"{hsh[:12]}-{w}x{h}"
        refs = {fa: oracle_path(w, h, hsh, fa) for fa in ORACLE_FA}
        f = root / f"{name}.f32"
        m = EMBD_RE.search(results[name].out)
        cols = int(m.group(2)) if m else 2560
        if f.exists():
            x = np.fromfile(f, dtype=np.float32)
            for fa, p in refs.items():
                if p.exists():
                    print(cmp_line(name, f"fa{fa}", label, np.fromfile(p, dtype=np.float32), x, cols))
                else:
                    print(f"{name:14s} fa{fa:4s} no oracle file {p}")
        if hsh not in seen and all(p.exists() for p in refs.values()):
            seen.add(hsh)
            print(cmp_line("oracle-faon", "faoff", label, np.fromfile(refs["off"], dtype=np.float32),
                           np.fromfile(refs["on"], dtype=np.float32), cols))
    for name in sorted(inputs):
        ddir = root / f"{name}-dump"
        if not ddir.is_dir():
            continue
        w, h, hsh = inputs[name]
        print(f"\nThe tensors of {name} against the two oracles (NMSE, smallest row cosine, max error / rms):")
        print(f"  {'tensor':16s} {'shape':22s} {'NMSE faoff':>11} {'rowcos':>10} {'max/rms':>8} "
              f"{'NMSE faon':>11} {'oracle on/off':>13}")
        index = [ln.split() for ln in (ddir / "index.txt").read_text().splitlines() if ln.strip()]
        for tname, _type, *ne in index:
            x = np.fromfile(ddir / f"{tname}.f32", dtype=np.float32)
            refs = {fa: oracle_path(w, h, hsh, fa, "-dump") / f"{tname}.f32" for fa in ORACLE_FA}
            if not all(p.exists() for p in refs.values()):
                print(f"  {tname:16s} no oracle tensor")
                continue
            r_off, r_on = (np.fromfile(refs[fa], dtype=np.float32) for fa in ("off", "on"))
            if r_off.size != x.size or r_on.size != x.size:
                print(f"  {tname:16s} the sizes differ: {x.size}, {r_off.size}, {r_on.size}")
                continue
            c_off, c_on, c_oo = compare(r_off, x, int(ne[0])), compare(r_on, x, int(ne[0])), compare(r_off, r_on, int(ne[0]))
            print(f"  {tname:16s} {'x'.join(ne):22s} {c_off['nmse']:11.3e} {c_off['rowcos_min']:10.6f} "
                  f"{c_off['maxerr_rms']:8.3f} {c_on['nmse']:11.3e} {c_oo['nmse']:13.3e}")
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
