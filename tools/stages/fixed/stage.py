#!/usr/bin/env python3
"""The phone stage "fixed": the fixed costs of a chat turn of the 4B Q8_0 on HTP0 in the engine of the app.

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/fixed/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled files (build/fixed/phone-out)

The tool of each run is memprobe (tools/memprobe/memprobe.cpp) in the context of the app (n_ctx 8192, 4 threads, 5 output
rows, the lazy token embedding, Q8_0 K and V, flash attention AUTO, the fused state step) with LLAMA_HOSTPROF=1.
Its app modes (--cold, --sweep, --turns) make the thread pool of the app and, with --spec, the MTP draft driver
of the app, and they write STAMP lines around each step on the clock of the log lines. The table assigns the
LLAMA_HOSTPROF lines and the GGML_HEXAGON_PROFILE lines of the log to the steps with these stamps.

One run matrix (BLOCKS, GROUPS) gives the command file and the parser, thus the two agree on each run name. A
run name is 4b-<block>-<round>-<variant>, for example 4b-sw-2-s. Each run writes three files to the phone
directory out/: <name>-gate.txt (the conditions before and after the run and the exit code), <name>.out (the
stdout of memprobe) and <name>.log (its stderr).

A run goes into the tables when its gate passed, its exit code is 0, the thermal status after it is 0, no CPU cap
before or after it is less than 3.0 GHz (CAP_MIN_KHZ), and its log has no failure line. A change of the caps that
stays at 3.0 GHz or more only marks the run in the list of conditions: the cap of cpu7 falls from 4320000 to
4089600 or 4204800 kHz after almost each run. --all also uses the removed runs. The table only reads files.
O(size of the files) time.

This file is tools/stages/fixed/stage.py, and build/fixed/stage.py is a link to it. The files of the stage
stay in build/fixed.
"""

import argparse
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/fixed"
MODEL_DIR = "/data/local/tmp/qwen/models"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
GATE_KB = 8388608
LAPTOP_STAGE = "build/fixed"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/fixed"
# The stage directory of the repository (tools/stages/fixed/stage.py is three levels below the root),
# relative to the working directory, for the default paths of the command file and of the outputs.
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
# The environment of the app (init_impl in llama_jni.cpp), the libraries of the stage and the host timers.
LIB_ENV = (f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib "
           "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1 LLAMA_HOSTPROF=1")
# The context of the app (load_impl in llama_jni.cpp): the KV cache of the NPU engine is Q8_0, flash attention
# stays AUTO, the output limit is the draft limit plus 1.
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"
STAGE_FILES = ("bin/gate.sh", "bin/memprobe", "lib/libggml-base.so", "lib/libggml-cpu.so", "lib/libggml-hexagon.so",
               "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so", "lib/libllama-bench-impl.so",
               "lib/libllama-common.so", "lib/libllama.so", "lib/libmtmd.so")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x memprobe; echo pgrep-done'"
# The highest temperature of the NPU thermal zones (type nsp*) in millidegrees, or nothing.
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')
# A run with a CPU cap of less than this value (kHz) before or after it stays out of the tables.
CAP_MIN_KHZ = 3000000
SWEEP_SIZES = (1, 8, 32, 40, 64, 128, 256, 512, 1024)
SWEEP_DEPTHS = (0, 512)
PROF_SIZES = (1, 8, 40, 128, 512)


@dataclass(frozen=True)
class Variant:
    """One engine form: its key, its text, and its memprobe arguments."""
    key: str
    name: str
    args: str


@dataclass(frozen=True)
class Block:
    """One kind of run: the memprobe arguments, the variants, the round count, the time limit in seconds, the
    extra environment, whether the run needs an empty snapshot directory, and its text. The gate and the lines
    after the tool take about 8 s, thus each limit is 110 s or less and a phone command stays under 120 s."""
    key: str
    args: str
    variants: str
    rounds: int
    limit: int
    env: str
    state_dir: bool
    text: str


VARIANTS = {v.key: v for v in (
    Variant("n", "draft off (the preset of the app)", ""),
    Variant("s", "draft on (MTP, n_rs_seq 4, the draft follows each decode)", "--spec"),
    Variant("r", "a rest of 12 s before each pass after the first", "--rest-ms 12000"),
    Variant("c", "no rest between the passes", ""),
    Variant("t", "the token embedding read into memory at the load (--embd-advise touch)", "--embd-advise touch"),
    Variant("d", "the model file dropped from the page cache before the load (--drop-cache)", "--drop-cache"),
)}
_SIZES = ",".join(map(str, SWEEP_SIZES))
_DEPTHS = ",".join(map(str, SWEEP_DEPTHS))
_PROF = ",".join(map(str, PROF_SIZES))
_TURNS = "--turn-first 500 --turn-message 40 --turn-answer 32"
BLOCKS = {b.key: b for b in (
    Block("sw", f"--cold 40 --sweep {_SIZES} --sweep-depths {_DEPTHS} --sweep-calls 4 --therm", "ns", 2, 100, "",
          False, f"the first decode of 40 tokens, then the sizes {_SIZES} at the depths {_DEPTHS}, 4 calls each"),
    Block("tn", f"--turns 6 {_TURNS} --therm", "ns", 2, 90, "", True,
          "a chat of 6 turns: 500 tokens, then 5 messages of 40 tokens, answers of up to 32 tokens"),
    Block("fu", "--cold 512 --sweep 40,512 --sweep-calls 2", "ntd", 2, 70, "", False,
          "the first decode of 512 tokens, then 40 and 512 tokens 2 times each"),
    Block("ti", f"--turns 4 {_TURNS} --turn-idle-ms 6000 --therm", "ns", 1, 100, "", True,
          "a chat of 4 turns with 6 s of idle time before each turn, as a user who reads and writes"),
    Block("pf", f"--cold 40 --sweep {_PROF} --sweep-depths {_DEPTHS} --sweep-calls 2", "ns", 1, 80,
          "GGML_HEXAGON_PROFILE=1", False, f"op profile: the first decode, the sizes {_PROF} at {_DEPTHS}, 2 calls"),
    Block("pt", f"--turns 3 {_TURNS}", "ns", 1, 90, "GGML_HEXAGON_PROFILE=1", True,
          "op profile: a chat of 3 turns"),
    Block("pd", "-p 4096 -n 16 --log-ts", "nt", 1, 60, "", False,
          "the host and DSP parts of the 4 prompt ubatches and of 16 decode tokens at the depth 4096"),
    Block("pp", "-p 4096 -n 16 --log-ts", "n", 1, 70, "GGML_HEXAGON_PROFILE=1", False,
          "op profile: 16 decode tokens at the depth 4096"),
    Block("hp", "-p 2048 --reps 4 --therm --log-ts", "rc", 2, 100, "", False,
          "a prompt of 2048 tokens 4 times, with and without a rest between the passes"),
    Block("hq", "-p 2048 --reps 3 --therm --log-ts", "c", 1, 60, "GGML_HEXAGON_PROFILE=1", False,
          "op profile: a prompt of 2048 tokens 3 times"),
)}
# The blocks of one group run round by round. The variants run in their order in an odd round and in the
# reverse order in an even round, thus a slow drift of the clocks or the heat goes equally to each variant.
GROUPS = (("sw", "tn"), ("fu",), ("ti",), ("hp",), ("pf", "pt", "pd", "pp", "hq"))


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
    """The runs of the stage in their order. O(runs)."""
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
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    b, v = run.block, run.variant
    stem = f"{PHONE}/out/{run.name}"
    env = " ".join(x for x in (LIB_ENV, b.env) if x)
    state = f"{PHONE}/state"
    args = " ".join(x for x in (PROBE_ARGS, v.args, b.args, f"--state-dir {state}" if b.state_dir else "") if x)
    prep = f"rm -rf {state} && mkdir -p {state} && " if b.state_dir else ""
    clean = f"rm -rf {state}; " if b.state_dir else ""
    cmd = (f"sh {PHONE}/bin/gate.sh {GATE_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && {prep}"
           f"timeout -s KILL {b.limit} env {env} {PHONE}/bin/memprobe -m {MODEL_DIR}/{MODEL} {args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {clean}{AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    return ["#", f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {run.name}, {b.text}, {v.key}: {v.name}",
            THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "fixed": the fixed costs that a chat turn of the 4B Q8_0 pays on HTP0 outside the steady prefill and
# decode rates, in the engine of the app, with the draft off (n) and on (s).
#
# The questions:
#   1. The prefill time against the prompt length (1 to 1024 tokens) at the depths 0 and 512, for the first call of a
#      size (a new graph and a new packed batch, as each prompt batch of the app) and for the calls after it (the
#      graph reused): the fixed part and the slope.
#   2. The time to the first token of a chat turn and its parts: the template, the tokens, the snapshot restore, the
#      prompt batch, the snapshot, the generation prompt, the first sample and the first step. Also after 6 s of idle.
#   3. Where the fixed part goes: LLAMA_HOSTPROF (graph build, scheduler, pack, submit, DSP wait, copies) and the
#      op profile (GGML_HEXAGON_PROFILE=1) on the same clock through the STAMP lines of memprobe.
#   4. The host and DSP parts of one decode token at the depth 4096.
#   5. Why the first pass of 2048 tokens is faster than the passes after it: a rest of 12 s between the passes
#      against no rest, the NPU zone temperature before and after each pass, and the DSP clock of each batch.
#   6. Where the host time of the first decode of the process goes (the first 512-token pass is 66 to 88 ms slower at
#      the same DSP time): a new graph, the first use of the buffers, or the pages of the lazy token embedding.
#
# The files (build/fixed/build.sh phone): tools/memprobe/memprobe.cpp with the app modes, built with the NDK against
# the libraries of the stage bench-kv (patches tree f473cad, the code of the app at HEAD). The libraries are the bytes
# of build/bench-kv/phone/lib, which ran 116 runs of bench-kv.
#
# The runs, 28:
#   sw  x4  the first decode of 40 tokens, then the sweep 1..1024 tokens at the depths 0 and 512, 4 calls each, n s / s n
#   tn  x4  a chat of 6 turns (500 tokens, then 40-token messages, answers of up to 32 tokens), n s / s n
#   fu  x6  the first decode of 512 tokens, then 40 and 512 tokens 2 times each: the embedding pages as the earlier
#           runs left them (n), read into memory at the load (t), and dropped with the file before the load (d)
#   ti  x2  a chat of 4 turns with 6 s of idle before each turn, n s
#   hp  x4  -p 2048 --reps 4: r (12 s rest before passes 2 to 4) and c (no rest), r c / c r
#   pf  x2  op profile of the first decode and of the sweep 1,8,40,128,512 at the depths 0 and 512, n s
#   pt  x2  op profile of a chat of 3 turns, n s
#   pd  x2  -p 4096 -n 16: LLAMA_HOSTPROF of the 4 prompt ubatches and of 16 decode tokens at the depth 4096, n t
#   pp  x1  the same with the op profile
#   hq  x1  op profile of -p 2048 --reps 3
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB, and it prints the caps), the tool under timeout -s KILL (less than 120 s), the exit code and the conditions after
# the run (thermal, caps, battery, NPU zone), then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, screen on, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 12 minutes of tool time plus about 8 s of
# gate and checks for each run, thus about 16 minutes, plus the waits for thermal status 0 and a battery of 38 C or less.
# The pull is about 60 MB (the profile logs). Then: build/fixed/stage.py table
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODEL_DIR} | grep -E \"{MODEL}\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out'",
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {libs} {PHONE}/lib/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
    ]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone
    directory goes only when the pull has each of its files."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'pgrep -x memprobe; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/phone-out/",
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
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The parser of the files ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
BEFORE_RE = re.compile(r"before: nsp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
# A log line with the time stamp of --log-ts: minutes, seconds, milliseconds, microseconds, the level letter.
TS_RE = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) [A-Z] (.*)$")
STAMP_RE = re.compile(r"memprobe: STAMP (\S+)(.*)$")
KV_RE = re.compile(r"([a-z_]+)=([-\w.]+)")
DECODE_RE = re.compile(r"hostprof: decode #\d+ tokens (\d+) ubatches (\d+) reused (\d+) embd_host \d+ \| (.*?) \| "
                       r"since_last_return (-?\d+) us")
SESSION_RE = re.compile(r"hostprof: HTP\d+ graphs (\d+) ops (\d+) hits (\d+) replays (\d+) verified \d+ batches (\d+) \| "
                        r"pack (\d+) submit (\d+) wait (\d+) pop (\d+) get (\d+) \((\d+) B\) set (\d+) \((\d+) B\) \| "
                        r"turnaround (-?\d+) us")
SCHED_RE = re.compile(r"hostprof: sched splits \d+ (.*) us")
SPLIT_RE = re.compile(r"\[(\S+) nodes (\d+) inputs (\d+) copy (\d+) compute (\d+)\]")
OPBATCH_RE = re.compile(r"profile-op OPBATCH\|.*?\|n-ops (\d+)\|.*\|usec (\d+) cycles (\d+) start \d+ mhz ([\d.]+)")
OP_RE = re.compile(r"profile-op ([A-Z0-9_+]+)\|([^|]*)\|.*\|usec (\d+) cycles \d+")
FIELD_RE = re.compile(r"([a-z_]+) (-?\d+)")
# The op classes of the profile tables.
CLASSES = ("HEAD", "W MUL_MAT", "FA", "GDN", "rest")


# The src0 names of a matmul of the output head: the head of the model, the tied head (the 4B Q8_0 reads the token
# embedding), and the reduced draft head of patches/draft-head/0001. blk.N.attn_output.weight is not a head.
HEAD_SRC0 = re.compile(r"^(output\.weight|token_embd\.weight|blk\.\d+\.nextn\.draft_head\.weight)\b")


def op_class(name: str, names: str) -> str:
    """The class of one profile-op line from its op name and its tensor names. HEAD is a matmul whose src0 (the
    first name) is a head. GDN is the recurrent-state path."""
    parts = name.split("+")
    if any(p.startswith("MUL_MAT") for p in parts) and HEAD_SRC0.match(names.split(" x ")[0].strip()):
        return "HEAD"
    if "FLASH_ATTN_EXT" in parts:
        return "FA"
    if any(p.startswith(("GATED_DELTA_NET", "GDN_", "SSM_CONV")) for p in parts):
        return "GDN"
    if any(p.startswith("MUL_MAT") for p in parts) and ".weight" in names:
        return "W MUL_MAT"
    return "rest"


@dataclass
class Window:
    """The log lines between two STAMP lines, summed: the target and draft decode lines of LLAMA_HOSTPROF, the
    session and scheduler lines, and the DSP batches and op classes of the profile."""
    t0: float = 0.0
    t1: float = 0.0
    tgt: Counter = field(default_factory=Counter)
    dft: Counter = field(default_factory=Counter)
    draft_decodes: int = 0  # the decode lines of the draft context inside a draft window (not the follows)
    session: Counter = field(default_factory=Counter)
    cpu_split_us: int = 0
    htp_split_us: int = 0
    dsp_us: int = 0
    batches: int = 0
    mhz: list = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)

    @property
    def wall_us(self) -> float:
        return self.t1 - self.t0


@dataclass
class Event:
    """One STAMP line: its time in us, its name, its fields, and the index of the log line."""
    t: float
    name: str
    kv: dict
    line: int


def parse_log(text: str) -> tuple[list[Event], list[tuple[float | None, str]]]:
    """The STAMP events and all lines (time, text) of one stderr file. O(lines).

    The Hexagon session prints the hostprof line of a graph at the next synchronize, or at the start of the next
    graph compute when no synchronize came. The last llama.cpp decode line before it is thus the decode that owns
    it. The parser moves each session line to the position after that decode line, thus a window that holds the
    decode also holds its session line. A session line with no decode line before it stays where it is."""
    raw_lines: list[tuple[float | None, str]] = []
    for raw in text.splitlines():
        m = TS_RE.match(raw)
        t, body = (None, raw)
        if m:
            t = ((int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(m.group(3))) * 1000.0 + int(m.group(4))
            body = m.group(5)
        raw_lines.append((t, body))
    moved: dict[int, list[tuple[float | None, str]]] = defaultdict(list)
    keep = [True] * len(raw_lines)
    last_decode = -1
    for i, (_, body) in enumerate(raw_lines):
        if DECODE_RE.search(body):
            last_decode = i
        elif last_decode >= 0 and SESSION_RE.search(body):
            moved[last_decode].append(raw_lines[i])
            keep[i] = False
    lines: list[tuple[float | None, str]] = []
    for i, item in enumerate(raw_lines):
        if keep[i]:
            lines.append(item)
        lines.extend(moved.get(i, ()))
    events: list[Event] = []
    for i, (t, body) in enumerate(lines):
        s = STAMP_RE.search(body)
        if s and t is not None:
            events.append(Event(t, s.group(1), dict(KV_RE.findall(s.group(2))), i))
    return events, lines


def window(lines: list[tuple[float | None, str]], events: list[Event], begin: Event, end: Event) -> Window:
    """Sum the lines between two events. A decode line inside a follow or draft sub-window belongs to the draft
    context. O(lines of the window)."""
    w = Window(begin.t, end.t)
    in_dft = False
    in_draft = False
    for t, body in lines[begin.line:end.line + 1]:
        s = STAMP_RE.search(body)
        if s:
            name = s.group(1)
            if name in ("follow-begin", "draft-begin"):
                in_dft = True
                in_draft = name == "draft-begin"
            elif name in ("follow-end", "draft-end"):
                in_dft = False
                in_draft = False
            continue
        m = DECODE_RE.search(body)
        if m:
            c = w.dft if in_dft else w.tgt
            w.draft_decodes += 1 if in_draft else 0
            c["calls"] += 1
            c["reused"] += int(m.group(3))
            for k, v in FIELD_RE.findall(m.group(4)):
                c[k] += int(v)
            continue
        m = SESSION_RE.search(body)
        if m:
            keys = ("graphs", "ops", "hits", "replays", "batches", "pack", "submit", "wait", "pop", "get", "get_b",
                    "set", "set_b", "turnaround")
            for k, v in zip(keys, m.groups()):
                w.session[k] += int(v)
            continue
        m = SCHED_RE.search(body)
        if m:
            for sm in SPLIT_RE.finditer(m.group(1)):
                if sm.group(1) == "CPU":
                    w.cpu_split_us += int(sm.group(5))
                else:
                    w.htp_split_us += int(sm.group(4)) + int(sm.group(5))
            continue
        m = OPBATCH_RE.search(body)
        if m:
            w.dsp_us += int(m.group(2))
            w.batches += 1
            w.mhz.append(float(m.group(4)))
            continue
        m = OP_RE.search(body)
        if m:
            w.classes[op_class(m.group(1), m.group(2))] += int(m.group(3))
    return w


def pairs(events: list[Event], begin: str, end: str) -> list[tuple[Event, Event]]:
    """The (begin, end) event pairs of one kind, in order. An end matches the last open begin. O(events)."""
    out, open_ev = [], None
    for ev in events:
        if ev.name == begin:
            open_ev = ev
        elif ev.name == end and open_ev is not None:
            out.append((open_ev, ev))
            open_ev = None
    return out


def kv_lines(text: str, tag: str) -> list[dict]:
    """The key=value fields of each stdout line that starts with the tag, for example "TIME sweep "."""
    return [dict(KV_RE.findall(line[len(tag):])) for line in text.splitlines() if line.startswith(tag)]


@dataclass
class Result:
    """The files of one run. ok is False when the run did not run or failed. flags names each condition that
    marks the run, and removed names each condition that keeps it out of the tables (without --all)."""
    run: Run
    ok: bool
    flags: list
    removed: list
    caps: str
    nsp: tuple
    out: str
    events: list
    lines: list


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
    nsp_b = BEFORE_RE.search(gate)
    nsp = (int(nsp_b.group(1)) / 1000 if nsp_b and nsp_b.group(1) else None,
           int(after.group(6)) / 1000 if after and after.group(6) else None)
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    events, lines = parse_log(log)
    if ok and "llama_kv_cache: size" in log and "K (q8_0)" not in log:
        removed.append("the KV cache is not Q8_0")
    if ok and re.search(r"follow-failed|AddressSanitizer|GGML_ASSERT|dspqueue_read failed", log):
        removed.append("the log has a failure line")
    return Result(run, ok, flags, removed, caps, nsp, out, events, lines)


def usable(res: Result, include_all: bool) -> bool:
    """True when the run goes into the tables: it ran, and no condition removes it (or --all)."""
    return res.ok and (include_all or not res.removed)


def med(values) -> float | None:
    """The median of the values that are not None, or None."""
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def fmt(x: float | None, digits: int = 1) -> str:
    """A number, or a dash for None."""
    return "-" if x is None else f"{x:.{digits}f}"


def fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """The least-squares line y = a + b x of the points, as (a, b), or None with fewer than two x values."""
    if len({x for x, _ in points}) < 2:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return my - b * mx, b


# ---- The tables ----

def checks(results: dict[str, Result]) -> list[str]:
    """The conditions of the runs."""
    runs = all_runs()
    got = [results[r.name] for r in runs if r.name in results]
    out = [f"{MODEL}: {len(got)} of {len(runs)} runs have a gate file, {sum(r.ok for r in got)} ran with exit code 0, "
           f"{sum(r.ok and not r.removed for r in got)} go into the tables, {sum(r.ok and not r.flags for r in got)} "
           f"have no mark"]
    caps = Counter(r.caps for r in got if r.ok)
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.most_common()))
    for i, when in enumerate(("before", "after")):
        temps = [r.nsp[i] for r in got if r.ok and r.nsp[i] is not None]
        if temps:
            out.append(f"  NPU zone {when} the runs: {min(temps):.1f} to {max(temps):.1f} C, median "
                       f"{statistics.median(temps):.1f} C")
    for r in got:
        if r.removed:
            out.append(f"  {r.run.name}: REMOVED ({', '.join(r.removed)})" + (f", marks: {', '.join(r.flags)}" if r.flags else ""))
        elif r.flags:
            out.append(f"  {r.run.name}: marks: " + ", ".join(r.flags))
    return out


def sweep_calls(res: Result) -> list[tuple[dict, Window | None]]:
    """The TIME sweep and TIME cold lines of one run with the window of each call. The k-th call-begin of the log
    belongs to the k-th such line (the cold call comes first). O(lines)."""
    times = kv_lines(res.out, "TIME cold ") + kv_lines(res.out, "TIME sweep ")
    for t in times[:1]:
        t.setdefault("depth", "0")
        t.setdefault("call", "0")
    if not kv_lines(res.out, "TIME cold "):
        times = kv_lines(res.out, "TIME sweep ")
    else:
        times[0]["cold"] = "1"
    wins = [window(res.lines, res.events, b, e) for b, e in pairs(res.events, "call-begin", "call-end")]
    return [(t, wins[i] if i < len(wins) else None) for i, t in enumerate(times)]


def sweep_table(results: dict[str, Result], block: str, include_all: bool) -> list[str]:
    """The prefill time against the size, the first call and the calls after it, for each variant and depth, with
    the host parts of LLAMA_HOSTPROF (medians over the calls and rounds), then the fits."""
    out = []
    for vk in BLOCKS[block].variants:
        rows: dict[tuple[int, int, bool], list[tuple[dict, Window | None]]] = defaultdict(list)
        cold: list[tuple[dict, Window | None]] = []
        for rnd in range(1, BLOCKS[block].rounds + 1):
            res = results.get(f"4b-{block}-{rnd}-{vk}")
            if res is None or not usable(res, include_all):
                continue
            for t, w in sweep_calls(res):
                if t.get("rc", "0") != "0":
                    continue
                if t.get("cold") == "1":
                    cold.append((t, w))
                    continue
                rows[(int(t["depth"]), int(t["tokens"]), t["call"] == "0")].append((t, w))
        if not rows:
            continue
        out.append(f"{block} {vk} ({VARIANTS[vk].name}): ms per prefill call, the median over the calls and rounds")
        out.append(f"  {'depth':>5} {'tokens':>6} | {'first':>7} {'later':>7} {'diff':>6} | later: {'decode':>7} "
                   f"{'follow':>6} {'sync':>6} {'restore':>7} | host us, first/later: {'build':>11} {'alloc':>11} "
                   f"{'pack':>11} {'cpu-split':>11} {'wait':>13} {'get':>9}")

        def host(items: list, key: str, where: str = "tgt") -> float | None:
            vals = []
            for _, w in items:
                if w is None:
                    continue
                if where == "tgt":
                    vals.append(w.tgt.get(key, 0))
                elif where == "session":
                    vals.append(w.session.get(key, 0))
                elif where == "cpu":
                    vals.append(w.cpu_split_us)
            return med(vals)

        fits: dict[tuple[int, bool], list[tuple[float, float]]] = defaultdict(list)
        for depth in sorted({d for d, _, _ in rows}):
            for size in sorted({s for d, s, _ in rows if d == depth}):
                first, later = rows.get((depth, size, True), []), rows.get((depth, size, False), [])
                f_ms = med(float(t["ms"]) for t, _ in first)
                l_ms = med(float(t["ms"]) for t, _ in later)
                if f_ms is not None:
                    fits[(depth, True)].append((size, f_ms))
                if l_ms is not None:
                    fits[(depth, False)].append((size, l_ms))
                diff = f_ms - l_ms if f_ms is not None and l_ms is not None else None

                def pair(key: str, where: str = "tgt") -> str:
                    return f"{fmt(host(first, key, where), 0)}/{fmt(host(later, key, where), 0)}"
                out.append(
                    f"  {depth:>5} {size:>6} | {fmt(f_ms):>7} {fmt(l_ms):>7} {fmt(diff):>6} | later: "
                    f"{fmt(med(float(t['decode_ms']) for t, _ in later)):>7} "
                    f"{fmt(med(float(t['follow_ms']) for t, _ in later)):>6} "
                    f"{fmt(med(float(t['sync_ms']) for t, _ in later)):>6} "
                    f"{fmt(med(float(t['restore_ms']) for t, _ in later)):>7} | "
                    f"{pair('build'):>21} {pair('alloc'):>11} {pair('pack', 'session'):>11} "
                    f"{pair('', 'cpu'):>11} {pair('wait', 'session'):>13} {pair('get', 'session'):>9}")
        if cold:
            t_ms = med(float(t["ms"]) for t, _ in cold)
            first40 = med(float(t["ms"]) for t, _ in rows.get((0, 40, True), []))
            out.append(f"  the first decode of the process, 40 tokens: {fmt(t_ms)} ms, against {fmt(first40)} ms for the "
                       f"first call of 40 tokens later (the first use costs {fmt(t_ms - first40 if t_ms and first40 else None)} "
                       f"ms). host us: build {fmt(host(cold, 'build'), 0)}, alloc {fmt(host(cold, 'alloc'), 0)}, pack "
                       f"{fmt(host(cold, 'pack', 'session'), 0)}, cpu split {fmt(host(cold, '', 'cpu'), 0)}, DSP wait "
                       f"{fmt(host(cold, 'wait', 'session'), 0)}")
        for (depth, first), pts in sorted(fits.items()):
            small = fit([p for p in pts if p[0] <= 64])
            large = fit([p for p in pts if p[0] >= 256])
            which = "first" if first else "later"
            out.append(f"  fit depth {depth} {which}: sizes 1..64 T = {fmt(small[0]) if small else '-'} + "
                       f"{fmt(small[1], 3) if small else '-'} x N ms; sizes 256..1024 T = {fmt(large[0]) if large else '-'} "
                       f"+ {fmt(large[1], 3) if large else '-'} x N ms")
        out.append("")
    return out


TURN_FIELDS = ("template_ms", "tokenize_ms", "restore_ms", "clear_ms", "base_ms", "snapshot_ms", "tail_ms", "prompt_ms",
               "first_sample_ms", "first_step_ms", "ttft_ms")


def turn_rows(res: Result) -> list[tuple[dict, dict[str, Window]]]:
    """The TIME turn lines of one run, each with the windows of its parts. O(lines)."""
    times = kv_lines(res.out, "TIME turn ")
    parts = {name: [window(res.lines, res.events, b, e) for b, e in pairs(res.events, f"{name}-begin", f"{name}-end")]
             for name in ("base", "tail", "snapshot", "restore")}
    turns = pairs(res.events, "turn-begin", "turn-end")
    steps = pairs(res.events, "step-begin", "step-end")
    out = []
    for i, t in enumerate(times):
        wins = {name: ws[i] for name, ws in parts.items() if i < len(ws)}
        if i < len(turns):
            first = [s for s in steps if s[0].t >= turns[i][0].t and s[1].t <= turns[i][1].t][:1]
            if first:
                wins["step0"] = window(res.lines, res.events, *first[0])
        out.append((t, wins))
    return out


def turn_table(results: dict[str, Result], block: str, include_all: bool) -> list[str]:
    """The time to the first token of each turn and its parts: turn 1 (the first message after the load) and the
    median of the later turns, for each variant, with the host parts of the prompt batch, the generation prompt
    and the first step."""
    out = []
    for vk in BLOCKS[block].variants:
        first, later = [], []
        for rnd in range(1, BLOCKS[block].rounds + 1):
            res = results.get(f"4b-{block}-{rnd}-{vk}")
            if res is None or not usable(res, include_all):
                continue
            for t, wins in turn_rows(res):
                (first if t.get("k") == "1" else later).append((t, wins))
        if not first and not later:
            continue
        out.append(f"{block} {vk} ({VARIANTS[vk].name}): ms, turn 1 | the median of turns 2 and later "
                   f"({len(later)} turns)")
        for key in ("base_tokens", "tail_tokens", "items", "reuse") + TURN_FIELDS + (
                "snapshot_mib", "gen_tokens", "gen_ms", "steps", "drafted", "accepted"):
            def val(rows: list, k: str = key):
                vals = [t.get(k) for t, _ in rows]
                if k == "reuse":
                    return ",".join(sorted(set(v for v in vals if v))) or "-"
                nums = [float(v) for v in vals if v not in (None, "-1.0")]
                return fmt(med(nums)) if nums else "-"
            out.append(f"  {key:16s} {val(first):>10} | {val(later):>10}")

        def part(rows: list, name: str, get) -> str:
            return fmt(med(get(w[name]) for _, w in rows if name in w), 0)
        for name in ("base", "tail", "step0"):
            for label, get in (("target build us", lambda w: w.tgt.get("build", 0)),
                               ("target alloc us", lambda w: w.tgt.get("alloc", 0)),
                               ("target reused", lambda w: w.tgt.get("reused", 0)),
                               ("draft decodes", lambda w: w.dft.get("calls", 0)),
                               ("draft build us", lambda w: w.dft.get("build", 0)),
                               ("pack us", lambda w: w.session.get("pack", 0)),
                               ("cpu split us", lambda w: w.cpu_split_us),
                               ("DSP wait us", lambda w: w.session.get("wait", 0)),
                               ("wall us", lambda w: w.wall_us)):
                out.append(f"  {name + ' ' + label:28s} {part(first, name, get):>10} | {part(later, name, get):>10}")
        out.append("")
    return out


def profile_table(results: dict[str, Result]) -> list[str]:
    """The DSP busy time against the wall time for each call of the profile runs: the calls of the sweep (pf), the
    parts of the turns (pt), and the decode tokens and passes (pp, hq)."""
    out = ["The op profile: DSP busy (the sum of the OPBATCH usec), batches, the DSP clock, the op classes in ms, and "
           "wall minus DSP busy (the host and queue part)"]
    head = f"  {'call':34s} {'wall':>7} {'DSP':>7} {'gap':>6} {'bat':>4} {'mhz':>11} " + " ".join(f"{c:>9}" for c in CLASSES)
    for vk in "ns":
        res = results.get(f"4b-pf-1-{vk}")
        if res is None or not res.ok:
            continue
        out += [f"pf {vk} ({VARIANTS[vk].name})", head]
        for t, w in sweep_calls(res):
            if w is None:
                continue
            label = "cold 40" if t.get("cold") == "1" else f"d{t['depth']} n{t['tokens']} call {t['call']}"
            out.append(prof_row(label, w))
    for vk in "ns":
        res = results.get(f"4b-pt-1-{vk}")
        if res is None or not res.ok:
            continue
        out += [f"pt {vk} ({VARIANTS[vk].name})", head]
        for t, wins in turn_rows(res):
            for name in ("restore", "base", "snapshot", "tail", "step0"):
                if name in wins:
                    out.append(prof_row(f"turn {t.get('k')} {name}", wins[name]))
        steps = [window(res.lines, res.events, b, e) for b, e in pairs(res.events, "step-begin", "step-end")]
        if steps:
            out.append(prof_median("median of all steps", steps))
        out += draft_summary(res)
    for key, name, begin, end in (("pp", "decode token", "step-begin", "step-end"), ("hq", "pass", "pass-begin", "pass-end")):
        res = results.get(f"4b-{key}-1-{'n' if key == 'pp' else 'c'}")
        if res is None or not res.ok:
            continue
        out += [f"{key} ({BLOCKS[key].text})", head]
        wins = [window(res.lines, res.events, b, e) for b, e in pairs(res.events, begin, end)]
        for i, w in enumerate(wins):
            out.append(prof_row(f"{name} {i}", w))
        if key == "pp" and len(wins) > 1:
            out.append(prof_median("median of tokens 1 and later", wins[1:]))
    return out


def draft_summary(res: Result) -> list[str]:
    """The draft steps of a run with the draft on: the draft length that the policy asked for (draft-begin n=)
    against the MTP passes that the driver ran (the draft decodes inside the draft window), and the head time of
    a step. The MTP drafter of common/speculative.cpp stops at its configured maximum and not at the length of the
    call, thus the two numbers differ when the driver ignores the length. O(lines)."""
    asked, ran, heads = Counter(), [], []
    for b, e in pairs(res.events, "step-begin", "step-end"):
        drafts = [ev for ev in res.events if b.t <= ev.t <= e.t and ev.name == "draft-begin"]
        if not drafts:
            continue
        w = window(res.lines, res.events, b, e)
        asked[drafts[0].kv.get("n", "?")] += 1
        ran.append(w.draft_decodes)
        heads.append(w.classes.get("HEAD", 0) / 1000)
    if not ran:
        return []
    return [f"  draft steps: {len(ran)}, asked length: " + ", ".join(f"{k} x{n}" for k, n in sorted(asked.items())) +
            f"; MTP passes per step: median {med(ran):.0f}, min {min(ran)}, max {max(ran)}" +
            (f"; HEAD ms per step: median {med(heads):.1f}" if any(heads) else "")]


def prof_row(label: str, w: Window) -> str:
    """One row of the profile table."""
    mhz = f"{min(w.mhz):.0f}-{max(w.mhz):.0f}" if w.mhz else "-"
    return (f"  {label:34s} {w.wall_us / 1000:7.1f} {w.dsp_us / 1000:7.1f} {(w.wall_us - w.dsp_us) / 1000:6.1f} "
            f"{w.batches:4d} {mhz:>11} " + " ".join(f"{w.classes.get(c, 0) / 1000:9.1f}" for c in CLASSES))


def prof_median(label: str, wins: list) -> str:
    """The row of the medians of several windows."""
    mw = Window(0.0, med(w.wall_us for w in wins))
    mw.dsp_us = med(w.dsp_us for w in wins)
    mw.batches = int(med(w.batches for w in wins))
    mw.mhz = [x for w in wins for x in w.mhz]
    mw.classes = Counter({c: med(w.classes.get(c, 0) for w in wins) for c in CLASSES})
    return prof_row(label, mw)


FU_KEYS = (("wall ms", None), ("build", "tgt"), ("alloc", "tgt"), ("inputs", "tgt"), ("compute", "tgt"),
           ("logits_get", "tgt"), ("pack", "session"), ("submit", "session"), ("set", "session"), ("wait", "session"),
           ("cpu split", "cpu"))


def fu_value(items: list, key: str, where: str | None) -> str:
    """The median of one field over the calls: the wall ms, a field of the target decode line, of the session line,
    or the CPU split, in us."""
    if where is None:
        return fmt(med(float(t["ms"]) for t, _ in items))
    if where == "tgt":
        return fmt(med(w.tgt.get(key, 0) for _, w in items if w), 0)
    if where == "session":
        return fmt(med(w.session.get(key, 0) for _, w in items if w), 0)
    return fmt(med(w.cpu_split_us for _, w in items if w), 0)


def first_use_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The first decode of 512 tokens of the process against the first call of 512 tokens after a call of 40 tokens
    (a new graph, no first use) and the call after it (the graph reused), for three states of the pages of the token
    embedding: as the earlier runs left them (n), all read at the load (t), and dropped with the file (d)."""
    out = ["fu: the size of the cold call (512) at depth 0, the median of the rounds. cold = the first decode of the "
           "process, first = the first call of that size after a call of 40 (a new graph), later = the call after it (the "
           "graph reused). us, but ms for the wall",
           f"  {'variant':<11} {'which':6} " + " ".join(f"{k:>10}" for k, _ in FU_KEYS)]
    for vk in "ntd":
        buckets: dict[str, list] = {"cold": [], "first": [], "later": []}
        loads = []
        for rnd in range(1, BLOCKS["fu"].rounds + 1):
            res = results.get(f"4b-fu-{rnd}-{vk}")
            if res is None or not usable(res, include_all):
                continue
            load = re.search(r"^TIME model-load ([\d.]+)", res.out, re.M)
            advise = re.search(r"^TIME embd-advise ([\d.]+)", res.out, re.M)
            loads.append(f"load {load.group(1) if load else '?'} ms" + (f", touch {advise.group(1)} ms" if advise else ""))
            calls = sweep_calls(res)
            size = next((t["tokens"] for t, _ in calls if t.get("cold") == "1"), None)
            for t, w in calls:
                if t.get("cold") == "1":
                    buckets["cold"].append((t, w))
                elif t.get("tokens") == size:
                    buckets["first" if t["call"] == "0" else "later"].append((t, w))
        for which, items in buckets.items():
            if items:
                out.append(f"  {vk + ' ' + ('embd as left' if vk == 'n' else 'embd touched' if vk == 't' else 'file dropped'):<11} "
                           f"{which:6} " + " ".join(f"{fu_value(items, k, where):>10}" for k, where in FU_KEYS))
        if loads:
            out.append(f"  {vk}: " + "; ".join(loads))
    return out


def decode_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The host and DSP parts of the prompt of 4096 tokens (4 ubatches) and of the decode tokens at the depth 4096
    (pd, without the profile), with the embedding pages as the earlier runs left them (n) and all read at the load
    (t): the LLAMA_HOSTPROF fields of each prompt ubatch, and the medians of tokens 1 and later."""
    out = []
    for vk in BLOCKS["pd"].variants:
        res = results.get(f"4b-pd-1-{vk}")
        if res is None or not usable(res, include_all):
            out.append(f"pd {vk}: no usable run")
            continue
        out.append(f"pd {vk} ({VARIANTS[vk].name})")
        passes = pairs(res.events, "pass-begin", "pass-end")
        if passes:
            pb, pe = passes[0]
            pw = window(res.lines, res.events, pb, pe)
            out.append(f"  prompt of 4096 tokens: wall {pw.wall_us / 1000:.1f} ms, DSP wait {pw.session.get('wait', 0) / 1000:.1f} "
                       f"ms, host us of the decode calls {pw.tgt.get('total', 0) / 1000:.1f} ms, cpu split {pw.cpu_split_us / 1000:.1f} ms, "
                       f"set {pw.session.get('set', 0) / 1000:.1f} ms ({pw.session.get('set_b', 0) / 1048576:.1f} MiB)")
            keys = ("build", "alloc", "inputs", "compute", "logits_get", "total")
            out.append(f"    {'ubatch':>6} " + " ".join(f"{k:>10}" for k in keys) + f" {'cpu split':>10} {'wall':>10}")
            ubs = [window(res.lines, res.events, b, e) for b, e in pairs(res.events, "decode-begin", "decode-end")
                   if b.t >= pb.t and e.t <= pe.t]
            for i, w in enumerate(ubs):
                out.append(f"    {i:>6} " + " ".join(f"{w.tgt.get(k, 0):>10}" for k in keys) +
                           f" {w.cpu_split_us:>10} {w.wall_us:>10.0f}")
        wins = [window(res.lines, res.events, b, e) for b, e in pairs(res.events, "step-begin", "step-end")][1:]
        if not wins:
            out.append("  no step stamps")
            continue
        out.append(f"  one decode token at the depth 4096, the median of {len(wins)} tokens (us): wall "
                   f"{med(w.wall_us for w in wins):.0f}, DSP wait {med(w.session.get('wait', 0) for w in wins):.0f}, "
                   f"host = wall - wait {med(w.wall_us - w.session.get('wait', 0) for w in wins):.0f}")
        out.append("    decode: " + ", ".join(f"{k} {med(w.tgt.get(k, 0) for w in wins):.0f}" for k in
                   ("prologue", "apply", "reuse_check", "build", "alloc", "inputs", "compute", "logits_get", "loop_other",
                    "epilogue", "total", "reused")))
        out.append("    session: " + ", ".join(f"{k} {med(w.session.get(k, 0) for w in wins):.0f}" for k in
                   ("batches", "pack", "submit", "wait", "pop", "get", "set", "turnaround")))
        out.append(f"    cpu split (the embedding row) {med(w.cpu_split_us for w in wins):.0f}, max "
                   f"{max(w.cpu_split_us for w in wins)}")
    return out


def pass_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The 2048-token passes with a rest (r) and without (c): the time of each pass and the NPU zone temperature
    before and after it."""
    out = ["hp: -p 2048 --reps 4, ms of each pass [NPU zone C before / after the pass]"]
    for rnd in (1, 2):
        for vk in "rc":
            res = results.get(f"4b-hp-{rnd}-{vk}")
            if res is None or not usable(res, include_all):
                continue
            passes = [float(m.group(1)) for m in re.finditer(r"^TIME prefill ([\d.]+) tokens", res.out, re.M)]
            therm = {k: int(v) / 1000 for k, v in re.findall(r"^THERM (pass-\d+-\w+) nsp=(\d+)", res.out, re.M)}

            def temp(key: str) -> str:
                return f"{therm[key]:.1f}" if key in therm else "-"
            cells = [f"{p:7.1f} [{temp(f'pass-{i}-before')}/{temp(f'pass-{i}-after')}]" for i, p in enumerate(passes)]
            out.append(f"  round {rnd} {vk} ({VARIANTS[vk].name}): " + "  ".join(cells))
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {r.name: read_result(root, r) for r in all_runs() if (root / f"{r.name}-gate.txt").exists()}
    parts = [checks(results), sweep_table(results, "sw", include_all), first_use_table(results, include_all),
             turn_table(results, "tn", include_all), turn_table(results, "ti", include_all),
             decode_table(results, include_all), pass_table(results, include_all), profile_table(results)]
    for part in parts:
        print("\n".join(part))
        print()
    print("The timelines and the op tables of a profile log (pf, pt, pp, hq):\n"
          f"  tools/trace/htp_trace.py summary {root}/4b-pp-1-n.log --batches 4: --show-batches   (the decode batches)\n"
          f"  tools/trace/htp_trace.py convert {root}/4b-pt-1-n.log -o /tmp/pt.json               (ui.perfetto.dev)\n"
          f"  tools/prof/optable.py graphs {root}/4b-pf-1-n.log                                  (each forward pass)\n"
          f"  tools/prof/optable.py ops {root}/4b-pp-1-n.log --graphs 4-19                       (the decode tokens)")
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
