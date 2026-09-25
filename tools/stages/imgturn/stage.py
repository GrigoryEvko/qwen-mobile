#!/usr/bin/env python3
"""The phone stage "imgturn": where the time of an image turn of the 4B Q8_0 on HTP0 goes, in the engine of the app.

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/imgturn/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled files (build/imgturn/phone-out)

The tool of each run is memprobe --image-turn (tools/memprobe/memprobe.cpp) in the context of the app (n_ctx 8192,
4 threads, 5 output rows, the lazy token embedding, Q8_0 K and V, the fused state step), with the projector
Qwen3.5-4B-Q8_0.mmproj.gguf on HTP0 and the photo of the chat of the user (the app file
files/images/a2200d1a726ec0a8576b4a18dc2ef1aa4e4c797d.jpg, 130 KB). The memory holds 1647 positions before each image
turn, as in that chat. Each run makes three image turns in one process:

    turn 1  the first image of the engine: the load of the projector, the decode, the encoder, the prefill
    turn 2  the same image again: the encoder without its first use (the app would find the image in its cache)
    turn 3  the image in the cache of the app: no decode and no encoder, only the prefill of the image tokens

The files are the phone files of tools/stages/ttft/build.sh phone: the libraries of the series of HEAD, and memprobe.

A run name is 4b-<block>-<round>-<variant>, for example 4b-it-2-s. Each run writes three files to the phone directory
out/: <name>-gate.txt, <name>.out (the stdout of memprobe) and <name>.log (its stderr with the STAMP lines).

A run goes into the tables when its gate passed, its exit code is 0, the thermal status after it is 0, no CPU cap before
or after it is less than 3.0 GHz (CAP_MIN_KHZ), and its log has no failure line. --all also uses the removed runs. The
table only reads files. O(size of the files) time.

This file is tools/stages/imgturn/stage.py. The files of the stage stay in build/imgturn.
"""

import argparse
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/imgturn"
MODEL_DIR = "/data/local/tmp/qwen/models"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
MMPROJ = "Qwen3.5-4B-Q8_0.mmproj.gguf"
APP_IMAGE = "files/images/a2200d1a726ec0a8576b4a18dc2ef1aa4e4c797d.jpg"
PHOTO = f"{PHONE}/in/photo.jpg"
DEPTH = 1647
GATE_KB = 8388608
LAPTOP_STAGE = "build/imgturn"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
LIB_ENV = f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"
STAGE_FILES = ("bin/gate.sh", "bin/memprobe", "lib/libggml-base.so", "lib/libggml-cpu.so", "lib/libggml-hexagon.so",
               "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so", "lib/libllama-common.so",
               "lib/libllama.so", "lib/libmtmd.so")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x memprobe; echo pgrep-done'"
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')
CAP_MIN_KHZ = 3000000
DRAFT = "--spec --draft-file auto"


@dataclass(frozen=True)
class Variant:
    """One configuration: its key, its text, its memprobe arguments and its extra environment."""
    key: str
    name: str
    args: str
    env: str = ""


@dataclass(frozen=True)
class Block:
    """One kind of run: the variants, the round count, the image turns of a run, and the time limit in seconds."""
    key: str
    variants: str
    rounds: int
    reps: int
    limit: int
    text: str


VARIANTS = {v.key: v for v in (
    Variant("s", "the app: draft on, 768 image tokens (Detailed)", f"{DRAFT} --image-tokens 768"),
    Variant("n", "draft off, 768 image tokens", "--image-tokens 768"),
    Variant("m", "draft on, 576 image tokens (Standard)", f"{DRAFT} --image-tokens 576"),
    Variant("f", "draft on, 256 image tokens (Fast)", f"{DRAFT} --image-tokens 256"),
    Variant("p", "the app with the op profile of the DSP and the scheduler splits", f"{DRAFT} --image-tokens 768",
            "GGML_HEXAGON_PROFILE=1 LLAMA_HOSTPROF=1"),
)}
BLOCKS = {b.key: b for b in (
    Block("it", "snmf", 2, 3, 100, "three image turns after 1647 positions of memory"),
    Block("ip", "p", 1, 2, 100, "two image turns with the op profile"),
)}
GROUPS = (("it",), ("ip",))


@dataclass(frozen=True)
class Run:
    """One phone run of one block, round and variant."""
    block: Block
    round: int
    variant: Variant

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"4b-{self.block.key}-{self.round}-{self.variant.key}"


def all_runs() -> list[Run]:
    """The runs of the stage in their order: the variants in their order in an odd round, reversed in an even
    round. O(runs)."""
    out = []
    for group in GROUPS:
        for rnd in range(1, max(BLOCKS[k].rounds for k in group) + 1):
            for key in group:
                block = BLOCKS[key]
                if rnd > block.rounds:
                    continue
                order = block.variants if rnd % 2 else block.variants[::-1]
                out.extend(Run(block, rnd, VARIANTS[v]) for v in order)
    return out


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line. The projector is the file in
    MODEL_DIR, or the one next to the model of the app in /sdcard/qwen/models."""
    b, v = run.block, run.variant
    stem = f"{PHONE}/out/{run.name}"
    env = " ".join(x for x in (LIB_ENV, v.env) if x)
    args = (f"{PROBE_ARGS} --mmproj $P --vision-dev HTP0 --image-turn {PHOTO} --image-depth {DEPTH} "
            f"--image-reps {b.reps} {v.args}")
    cmd = (f"sh {PHONE}/bin/gate.sh {GATE_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"P={MODEL_DIR}/{MMPROJ}; [ -f $P ] || P=/sdcard/qwen/models/{MMPROJ}; echo \"mmproj: $P\" >> {stem}-gate.txt; "
           f"timeout -s KILL {b.limit} env {env} {PHONE}/bin/memprobe -m {MODEL_DIR}/{MODEL} {args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    return ["#", f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {run.name}, {b.text}, {v.key}: {v.name}",
            THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "imgturn": where the time of an image turn of the 4B Q8_0 on HTP0 goes, in the engine of the app.
#
# The chat of the user (APK 4be6073b) gave "prefill 766 tok in 1615 ms" for a turn with one photo of 744 tokens. That
# time holds the vision encoder and the prefill of the image tokens. Before it come the load of the projector (the
# first image of the engine), the decode of the JPEG and the preprocessor.
#
# The questions:
#   1. The split of the first image turn: the projector load, the SHA-256, the JPEG decode at the target size (the NDK
#      decoder of the platform), the chunks of mtmd, the text before the image, the encoder, the prefill of the image
#      tokens, the text after the image, the first sample.
#   2. The same for a second image (the encoder without its first use) and for an image in the cache of the app.
#   3. The image token budget: 768 (the user), 576 and 256 tokens.
#   4. The draft on (the app) against off: the rollback slots of the draft make each prefill call slower.
#   5. The op profile of the encoder and of the image prefill on the DSP, and the split of the graph of the encoder
#      between the CPU and HTP0 (one run, not a timing run).
#
# The files: build/ttft/phone of the box, the phone files of tools/stages/ttft/build.sh phone (the libraries of the
# series of HEAD and memprobe with --image-turn). The phone must hold Qwen3.5-4B-Q8_0.gguf and
# Qwen3.5-4B-Q8_0-draft32k.gguf in /data/local/tmp/qwen/models, and Qwen3.5-4B-Q8_0.mmproj.gguf there or in
# /sdcard/qwen/models. The photo comes from the app with run-as.
#
# The runs, 9:
#   it  x8   three image turns after 1647 positions of memory: s n m f / f m n s
#   ip  x1   two image turns with the op profile: p
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB, and it prints the caps), the tool under timeout -s KILL (100 s), the exit code and the conditions after the run
# (thermal, caps, battery, NPU zone), then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, screen on, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 4 minutes of tool time plus about 8 s of
# gate and checks for each run, thus about 6 minutes, plus the waits for thermal status 0 and a battery of 38 C or
# less. The pull is about 30 MB (the op profile log). Then: tools/stages/imgturn/stage.py table (on the box).
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage files and the photo to the phone and check them."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/ttft/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODEL_DIR} /sdcard/qwen/models | grep -E \"Qwen3.5-4B-Q8_0(-draft32k|.mmproj)?.gguf\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out {PHONE}/in'",
        # run-as writes the file of the app to stdout, and the shell of adb writes it into the stage directory.
        f"{ADB} shell 'run-as ai.airi.qwenmobile cat {APP_IMAGE} > {PHOTO} && ls -l {PHOTO} && sha1sum {PHOTO}'",
        f"{ADB} pull {PHOTO} {LAPTOP_STAGE}/photo.jpg",
        f"rsync -a {LAPTOP_STAGE}/photo.jpg {BOX}/imgturn/photo.jpg",
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {libs} {PHONE}/lib/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
    ]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone directory goes
    only when the pull has each of its files."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'pgrep -x memprobe; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/imgturn/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    runs = all_runs()
    lines = HEADER.rstrip("\n").split("\n") + setup_lines()
    lines += ["#", f"# ==== {MODEL.removesuffix('.gguf')}: {len(runs)} runs ===="]
    for run in runs:
        lines += run_lines(run)
    lines += output_lines()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The parser of the files ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
KV_RE = re.compile(r"([a-z_]+)=([-\w./]+)")
STAMP_RE = re.compile(r"memprobe: STAMP (\S+)(.*)$")
# One op of the profile of the Hexagon backend: the op name, and its time in microseconds. An OPBATCH line is one batch
# of ops on the DSP, and its time is the busy time of the DSP.
OP_RE = re.compile(r"profile-op ([A-Z0-9_+]+)\|.*\|usec (\d+) cycles")
OPBATCH_RE = re.compile(r"profile-op OPBATCH\|.*\|usec (\d+) cycles")
# One graph of a scheduler (LLAMA_HOSTPROF): the split count, then each split with its backend, its node count and its
# host time. The compute time of an HTP0 split is the time to send it, because the DSP runs it asynchronously.
SCHED_RE = re.compile(r"hostprof: sched splits (\d+)(.*) us")
SPLIT_RE = re.compile(r"\[(\S+) nodes (\d+) inputs \d+ copy (\d+) compute (\d+)\]")


@dataclass
class Result:
    """The files of one run."""
    run: Run
    ok: bool
    flags: list
    removed: list
    caps: str
    out: str
    log: str


def read_result(root: Path, run: Run) -> Result:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    gate_path, out_path, log_path = (root / f"{run.name}{s}" for s in ("-gate.txt", ".out", ".log"))
    gate = gate_path.read_text(errors="replace") if gate_path.exists() else ""
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    flags = []
    ok = "gate: OK" in gate and rc is not None and rc.group(1) == "0"
    if not gate:
        flags.append("no gate file")
    elif "gate: OK" not in gate:
        flags.append("gate stopped the run")
    elif not ok:
        flags.append(f"exit code {rc.group(1) if rc else '?'}")
    removed = []
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        flags.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    cap_values = [int(v) for v in ((before.group(3), before.group(4)) if before else ()) +
                  ((after.group(2), after.group(3)) if after else ()) if v]
    if cap_values and min(cap_values) < CAP_MIN_KHZ:
        removed.append(f"a cap of {min(cap_values)} kHz")
    if after and after.group(1) not in ("", "0"):
        removed.append(f"thermal {after.group(1)} after the run")
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    if ok and re.search(r"GGML_ASSERT|dspqueue_read failed|did not load|did not decode|did not open|did not read|mtmd_tokenize gave", log):
        removed.append("the log has a failure line")
    if ok and re.search(r"image-turn .* rc=[1-9-]", out):
        removed.append("a turn failed")
    return Result(run, ok, flags, removed, caps, out, log)


def med(values) -> float | None:
    """The median of the values that are not None, or None."""
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def fmt(x: float | None, digits: int = 1) -> str:
    """A number, or a dash for None."""
    return "-" if x is None else f"{x:.{digits}f}"


def turns_of(res: Result) -> list[dict]:
    """The TIME image-turn lines of one run."""
    return [dict(KV_RE.findall(line[len("TIME image-turn "):])) for line in res.out.splitlines()
            if line.startswith("TIME image-turn ")]


FIELDS = ("ttft_ms", "vision_load_ms", "sha_ms", "decode_ms", "tokenize_ms", "pre_ms", "encode_ms", "image_tokens",
          "image_ms", "post_tokens", "post_ms", "sample_ms", "nx", "ny")
COUNT_FIELDS = ("image_tokens", "post_tokens", "nx", "ny")


def turn_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The parts of the three image turns of each variant (the median over its runs), then the time to the first
    token of each proposal."""
    out = []
    summary: dict[str, dict[int, dict[str, float | None]]] = {}
    for vk in BLOCKS["it"].variants + BLOCKS["ip"].variants:
        block = "ip" if vk == "p" else "it"
        rs = [results[r.name] for r in all_runs() if r.variant.key == vk and r.name in results and
              results[r.name].ok and (include_all or not results[r.name].removed)]
        if not rs:
            out.append(f"{block} {vk}: no usable run")
            continue
        by_rep: dict[int, list[dict]] = defaultdict(list)
        for res in rs:
            for t in turns_of(res):
                by_rep[int(t.get("rep", "0"))].append(t)
        reps = sorted(by_rep)
        out.append(f"{block} {vk} ({VARIANTS[vk].name}), {len(rs)} runs: ms, the median of the runs for each turn")
        out.append(f"  {'':16s} " + " ".join(f"{'turn ' + str(r) + (' cached' if r >= 3 else ''):>14}" for r in reps))
        summary[vk] = {}
        for key in FIELDS:
            cells = []
            for r in reps:
                v = med(float(t[key]) for t in by_rep[r] if key in t)
                summary[vk].setdefault(r, {})[key] = v
                cells.append(f"{fmt(v, 0 if key in COUNT_FIELDS else 1):>14}")
            out.append(f"  {key:16s} " + " ".join(cells))
        out.append("")
    out += proposals(summary)
    return out


def proposals(summary: dict[str, dict[int, dict[str, float | None]]]) -> list[str]:
    """The time to the first token of the image turn of the user for each proposal, from the medians of the turns.
    An encode at attach time leaves the turn of an image in the cache (turn 3). A prefill at attach time also puts the
    text before the image and the image tokens into the memory, thus the send decodes only the text after the image:
    the tokenize of turn 3, its post_ms and its sample_ms."""
    s = summary.get("s", {})
    if 1 not in s or 2 not in s:
        return ["The proposals need turns 1 and 2 of the variant s."]
    cold, warm, cached = s[1], s[2], s.get(3, {})

    def part_sum(turn: dict, keys: tuple[str, ...]) -> float | None:
        vals = [turn.get(k) for k in keys]
        return None if any(v is None for v in vals) else sum(vals)

    rows = [
        ("today, the first image of the engine", cold.get("ttft_ms")),
        ("today, a later image", warm.get("ttft_ms")),
        ("encode at attach time (the image in the cache)", cached.get("ttft_ms")),
        ("prefill at attach time (the image in the memory)", part_sum(cached, ("tokenize_ms", "post_ms", "sample_ms"))),
    ]
    for vk, label in (("m", "576 tokens (Standard)"), ("f", "256 tokens (Fast)")):
        if 2 in summary.get(vk, {}):
            rows.append((f"a later image with {label}", summary[vk][2].get("ttft_ms")))
    if 2 in summary.get("n", {}):
        rows.append(("a later image with the draft off", summary["n"][2].get("ttft_ms")))
    out = ["The time to the first token of the image turn of the user (draft on, 768 tokens), ms:"]
    out += [f"  {label:50s} {fmt(v):>8}" for label, v in rows]
    return out


def profile_table(results: dict[str, Result]) -> list[str]:
    """The op classes of the DSP in the windows of the encoder and of the image prefill of the profile run: the sum of
    the op times by op name, the ten largest, and the DSP busy time (the OPBATCH lines)."""
    res = results.get("4b-ip-1-p")
    if res is None or not res.ok:
        return ["ip p: no run"]
    out = []
    lines = res.log.splitlines()
    for begin, end, label in (("image-encode-begin", "image-encode-end", "the encoder"),
                              ("image-decode-begin", "image-decode-end", "the prefill of the image tokens")):
        windows, start = [], None
        for i, line in enumerate(lines):
            m = STAMP_RE.search(line)
            if not m:
                continue
            if m.group(1) == begin:
                start = i
            elif m.group(1) == end and start is not None:
                windows.append((start, i))
                start = None
        for n, (a, b) in enumerate(windows):
            ops, busy, sched = Counter(), 0, Counter()
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
                    busy += int(mb.group(1))
                    continue
                mo = OP_RE.search(line)
                if mo:
                    ops[mo.group(1)] += int(mo.group(2))
            top = ", ".join(f"{k} {v / 1000:.1f}" for k, v in ops.most_common(10))
            out.append(f"ip p, {label}, turn {n + 1}: DSP busy {busy / 1000:.1f} ms, ops {sum(ops.values()) / 1000:.1f} ms: "
                       f"{top}")
            out.append(f"    scheduler: {sched['graphs']} graphs, {sched['splits']} splits, CPU {sched['cpu_nodes']} nodes "
                       f"{sched['cpu_us'] / 1000:.1f} ms, device {sched['dev_nodes']} nodes {sched['dev_us'] / 1000:.1f} ms "
                       "to copy and send")
    return out


def checks(results: dict[str, Result]) -> list[str]:
    """The conditions of the runs."""
    runs = all_runs()
    got = [results[r.name] for r in runs if r.name in results]
    out = [f"{MODEL}: {len(got)} of {len(runs)} runs have a gate file, {sum(r.ok for r in got)} ran with exit code 0, "
           f"{sum(r.ok and not r.removed for r in got)} go into the tables"]
    caps = Counter(r.caps for r in got if r.ok)
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.most_common()))
    for r in got:
        if r.removed:
            out.append(f"  {r.run.name}: REMOVED ({', '.join(r.removed)})" + (f", marks: {', '.join(r.flags)}" if r.flags else ""))
        elif r.flags:
            out.append(f"  {r.run.name}: marks: " + ", ".join(r.flags))
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {r.name: read_result(root, r) for r in all_runs() if (root / f"{r.name}-gate.txt").exists()}
    for part in (checks(results), turn_table(results, include_all), profile_table(results)):
        print("\n".join(part))
        print()
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the tables from the pulled files")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    t.add_argument("--all", action="store_true", help="also use the runs with changed caps or heat")
    a = ap.parse_args()
    if a.cmd == "commands":
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(all_runs())} runs")
        return 0
    return table(a.root, a.all)


if __name__ == "__main__":
    sys.exit(main())
