#!/usr/bin/env python3
"""The phone stage "spf": the time of a short prefill call of the 4B Q8_0 on HTP0 against its token count.

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/spf/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled files (build/spf/phone-out)

The engine of each prefill run is memprobe (tools/memprobe/memprobe.cpp) in the context of the app: n_ctx 8192,
4 threads in the thread pool of the app (low priority, no polling), Q8_0 K and V, flash attention AUTO, the
lazy token embedding, the output limit 5, op fusion and the fused state step on (the environment of the app),
and with --spec the MTP draft context with its 32768-row head, which follows each decode as in the app. The
mode --sweep restores the state of a depth before each call, and the first call of a size builds a new graph
(as each prompt call of the app); the later calls of the size reuse the graph. LLAMA_HOSTPROF=1 is on in each
run, and memprobe writes STAMP lines around each call.

Two library sets: b is the tree of HEAD (the table), n is HEAD with the candidate patches of
tools/stages/spf/patches (the switch GGML_HEXAGON_GDN_CHUNK_MIN with the preset 2, and the row copy of CONCAT).

A run name is <block>-<lib>-<draft>, for example sw-b-s. Each run writes <name>-gate.txt (the conditions before
and after the run and the exit code), <name>.out and <name>.log to the phone directory out/. A run goes into the
tables when its gate passed, its exit code is 0, the thermal status after it is 0 and no CPU cap before or after
it is less than 3.0 GHz (--all also uses the other runs). The table only reads files. O(size of the files).

This file is tools/stages/spf/stage.py, and build/spf/stage.py is a link to it. The files of the stage stay in
build/spf. The log parser reuses the patterns of tools/stages/fixed/stage.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_FIXED_PATH = Path(__file__).resolve().parents[1] / "fixed" / "stage.py"
_spec = importlib.util.spec_from_file_location("fixed_stage", _FIXED_PATH)
fixed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixed)

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/spf"
MODELS = "/data/local/tmp/qwen/models"
EVAL = "/data/local/tmp/qwen/eval"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
DRAFT = "Qwen3.5-4B-Q8_0-draft32k.gguf"
GATE_KB = 8388608          # the MemAvailable (KiB) that a run of the 4B needs
GATE_KB_OPS = 2097152      # the MemAvailable (KiB) of a test-backend-ops run
LAPTOP_STAGE = "build/spf"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/spf"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
# The environment of the app (init_impl in llama_jni.cpp) and the host timers
APP_ENV = "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1 LLAMA_HOSTPROF=1"
# The context of the app (load_impl in llama_jni.cpp)
PROBE_ARGS = f"-m {MODELS}/{MODEL} -dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"
BENCH_ARGS = f"-m {MODELS}/{MODEL} -dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
PPL_ARGS = (f"-m {MODELS}/{MODEL} -dev HTP0 -ngl 99 -t 4 -fa on -ctk q8_0 -ctv q8_0 -f {EVAL}/wiki.test.raw -c 512 "
            f"--kl-divergence-base {EVAL}/naive-4B-q8.kld --kl-divergence")
SIZES = (1, 2, 4, 8, 16, 22, 32, 48, 64)
DEPTHS = (700, 2300)
_SIZES = ",".join(map(str, SIZES))
_DEPTHS = ",".join(map(str, DEPTHS))
# The PMU set dma-wait of tools/prof/pmu.py, in its order
DMA_WAIT = ("UDMA_ACTIVE", "UDMA_DMPOLL", "UDMA_NONCOH_RD", "UDMA_RDBUF_FULL", "L2_UDMA_BYPASS_RD", "SYSTEM_BUSY",
            "DU_MISS", "PKT_ANY")
DMA_WAIT_ENV = "GGML_HEXAGON_PROFILE=0x240,0x245,0x262,0x269,0x256,0xef,0xe9,0x3"
# The process names that pgrep -x sees: the kernel keeps 15 characters of a name
TOOLS = ("memprobe", "llama-bench", "llama-perplexit", "test-backend-op")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell '" + "; ".join(f"pgrep -x {t}" for t in TOOLS) + "; echo pgrep-done'"
BEFORE = f'echo "before: nsp={fixed.NSP}"'
CAP_MIN_KHZ = 3000000


@dataclass(frozen=True)
class Lib:
    """One library set: its key, its directories, and its text."""
    key: str
    ld: str
    adsp: str
    text: str


LIBS = {
    "b": Lib("b", f"{PHONE}/lib-base", f"{PHONE}/lib-base", "HEAD"),
    "n": Lib("n", f"{PHONE}/lib-new:{PHONE}/lib-base", f"{PHONE}/lib-new", "HEAD plus the candidates"),
}


@dataclass(frozen=True)
class Run:
    """One phone run: the block, the library set, the draft (s on, n off, - no model context), the tool, its
    arguments, the extra environment, the time limit in seconds, the MemAvailable gate and the text."""
    block: str
    lib: str
    draft: str
    tool: str
    args: str
    env: str
    limit: int
    gate_kb: int
    text: str

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"{self.block}-{self.lib}-{self.draft}"


def _probe(block: str, lib: str, draft: str, args: str, env: str, limit: int, text: str) -> Run:
    spec = " --spec" if draft == "s" else ""
    return Run(block, lib, draft, "memprobe", f"{PROBE_ARGS}{spec} {args}", env, limit, GATE_KB, text)


_SW = f"--sweep {_SIZES} --sweep-depths {_DEPTHS} --sweep-calls 3 --therm"
_PF = f"--sweep {_SIZES} --sweep-depths {_DEPTHS} --sweep-calls 2"
RUNS: list[Run] = [
    # The table (b) and the candidates (n), without the op profile: the wall time and the host timers
    _probe("sw", "b", "n", _SW, "", 100, f"the sweep {_SIZES} at the depths {_DEPTHS}, 3 calls each, draft off"),
    _probe("sw", "n", "n", _SW, "", 100, "the same, draft off"),
    _probe("sw", "n", "s", _SW, "", 100, "the same, draft on"),
    _probe("sw", "b", "s", _SW, "", 100, "the same, draft on"),
    # The op profile of the same calls, 2 calls of each size
    _probe("pf", "b", "n", _PF, "GGML_HEXAGON_PROFILE=1", 100, "op profile of the sweep, draft off"),
    _probe("pf", "n", "n", _PF, "GGML_HEXAGON_PROFILE=1", 100, "op profile of the sweep, draft off"),
    _probe("pf", "n", "s", _PF, "GGML_HEXAGON_PROFILE=1", 100, "op profile of the sweep, draft on"),
    _probe("pf", "b", "s", _PF, "GGML_HEXAGON_PROFILE=1", 100, "op profile of the sweep, draft on"),
    # Where the time of the HMX matmuls of a short call goes: the phases of each DSP thread (level 3), and the
    # DMA counters of each op (the PMU set dma-wait)
    _probe("tr", "b", "n", "--sweep 1,22 --sweep-depths 700 --sweep-calls 2", "GGML_HEXAGON_PROFILE=3", 90,
           "the phase trace of each DSP thread, 1 and 22 tokens at the depth 700, draft off"),
    _probe("pm", "b", "n", "--sweep 1,8,22 --sweep-depths 700 --sweep-calls 2", DMA_WAIT_ENV, 90,
           "the DMA counters of each op (PMU set dma-wait), 1, 8 and 22 tokens at the depth 700, draft off"),
    # A long prefill with the draft on: the MTP concat, and no loss at 512 and 1024 tokens
    _probe("lg", "b", "s", "--sweep 512,1024 --sweep-depths 0 --sweep-calls 2", "", 90,
           "512 and 1024 tokens at the depth 0, 2 calls each, draft on"),
    _probe("lg", "n", "s", "--sweep 512,1024 --sweep-depths 0 --sweep-calls 2", "", 90,
           "512 and 1024 tokens at the depth 0, 2 calls each, draft on"),
    # The rule of the series: pp512 and tg32 do not become slower
    Run("bn", "b", "-", "llama-bench", f"{BENCH_ARGS} -p 512 -n 32 -d 0 -r 3", "", 90, GATE_KB,
        "llama-bench pp512 and tg32 at the depth 0, 3 repetitions"),
    Run("bn", "n", "-", "llama-bench", f"{BENCH_ARGS} -p 512 -n 32 -d 0 -r 3", "", 90, GATE_KB,
        "llama-bench pp512 and tg32 at the depth 0, 3 repetitions"),
    # The KL of the prefill path with small ubatches: n runs the chunked gated delta net at 16 and 5 tokens
    Run("kl", "b", "-", "llama-perplexity", f"{PPL_ARGS} --chunks 4 -b 512 -ub 16", "", 110, GATE_KB,
        "KL of the prefill path in ubatches of 16 tokens, 4 chunks of 512"),
    Run("kl", "n", "-", "llama-perplexity", f"{PPL_ARGS} --chunks 4 -b 512 -ub 16", "", 110, GATE_KB,
        "KL of the prefill path in ubatches of 16 tokens, 4 chunks of 512"),
    Run("ks", "n", "-", "llama-perplexity", f"{PPL_ARGS} --chunks 2 -b 512 -ub 5", "", 110, GATE_KB,
        "KL of the prefill path in ubatches of 5 tokens, 2 chunks of 512"),
    # The op tests against the CPU backend: the chunked gated delta net at small batches (the bound follows the
    # switch) and the row copy of CONCAT
    Run("tb", "b", "-", "test-backend-ops", "-o CONCAT,GATED_DELTA_NET -b HTP0", "", 100, GATE_KB_OPS,
        "test-backend-ops of CONCAT and GATED_DELTA_NET on HTP0"),
    Run("tb", "n", "-", "test-backend-ops-new", "-o CONCAT,GATED_DELTA_NET -b HTP0", "", 100, GATE_KB_OPS,
        "test-backend-ops of CONCAT and GATED_DELTA_NET on HTP0"),
]
DRAFT_TEXT = {"n": "draft off", "s": "draft on (MTP, n_rs_seq 4)", "-": ""}


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    lib = LIBS[run.lib]
    stem = f"{PHONE}/out/{run.name}"
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={lib.ld} ADSP_LIBRARY_PATH={lib.adsp}", APP_ENV, run.env) if x)
    cmd = (f"sh {PHONE}/bin/gate.sh {run.gate_kb} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"timeout -s KILL {run.limit} env {env} {PHONE}/bin/{run.tool} {run.args} > {stem}.out 2> {stem}.log; "
           f"echo \"rc=$?\" >> {stem}-gate.txt; {fixed.AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt")
    title = f"{run.name}, {run.text}" + (f", {DRAFT_TEXT[run.draft]}" if run.draft != "-" and "draft" not in run.text
                                         else "") + f", {lib.text}"
    head = f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {title}" if run.tool != "test-backend-ops" and \
        not run.tool.startswith("test-backend-ops") else f"# {title}"
    return ["#", head, THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "spf": the time of a short prefill call (1 to 64 tokens) of the 4B Q8_0 on HTP0 at the depths 700 and
# 2300, with the draft off and on, in the engine of the app, on the libraries of HEAD, and the first candidate fixes.
#
# The questions:
#   1. The table: for each token count, depth and draft state, the time of the call (the first call of a size, which
#      builds a new graph as each prompt call of the app, and the later calls), the host part, the DSP busy time and
#      the DSP time of each op class. The fixed part and the slope of a fit against the token count.
#   2. The chunked gated delta net at 2 to 31 tokens (GGML_HEXAGON_GDN_CHUNK_MIN=2 in the libraries n) against the
#      sequential kernel of HEAD (the limit 32): the time of the op at each token count, the KL of the prefill path
#      in ubatches of 16 and 5 tokens, and the op tests against the CPU.
#   3. The row copy of CONCAT (the MTP concat of the draft context): its time in the calls with the draft on and in
#      the prefill of 512 and 1024 tokens, and the op tests.
#   4. Why the HMX matmuls of a call of 5 to 64 tokens read the weights at 50 to 54 GB/s, while the decode matvec
#      reads them at about 60 GB/s: the phases of each DSP thread (level 3), and the DMA counters of each op.
#   5. pp512 and tg32 of the libraries n against HEAD.
#
# The files (tools/stages/spf/build.sh): lib-base is the tree of HEAD, lib-new holds the libraries of HEAD plus
# tools/stages/spf/patches that differ (the host library and the DSP library), bin holds memprobe, llama-bench,
# llama-perplexity, test-backend-ops (HEAD) and test-backend-ops-new (the bound of the candidates).
#
# The runs, {n_runs}:
{run_list}
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,
# MemAvailable 8 GB for a model run and 2 GB for test-backend-ops, and it prints the caps), the tool under
# timeout -s KILL (110 s or less), the exit code and the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, screen on, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 14 minutes of tool time plus about
# 8 s of gate and checks for each run, plus the waits for thermal status 0. The push is about 190 MB, the pull
# about 150 MB (the profile logs). Then on the box: python3 build/spf/stage.py table
"""


def stage_files() -> list[str]:
    """The files of the stage: each line of phone/SHA256SUMS that the build wrote."""
    sums = STAGE_DIR / "phone" / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"stage.py: {sums} does not exist. Run tools/stages/spf/build.sh first.")
    return [line.split()[1] for line in sums.read_text().splitlines() if line.strip()]


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    files = stage_files()
    lines = [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODELS} | grep -E \"4B-Q8_0(-draft32k)?.gguf\"; ls -l {EVAL} | grep -E \"naive-4B-q8|wiki.test\"'",
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
    run_list = "\n".join(f"#   {r.name:9s} {r.text}, {LIBS[r.lib].text}" +
                         (f", {DRAFT_TEXT[r.draft]}" if r.draft != "-" and "draft" not in r.text else "")
                         for r in RUNS)
    header = HEADER.format(n_runs=len(RUNS), run_list=run_list)
    lines = header.rstrip("\n").split("\n") + setup_lines()
    lines += ["#", f"# ==== {len(RUNS)} runs ===="]
    for run in RUNS:
        lines += run_lines(run)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The parser of the files ----

# The op classes of the tables
CLASSES = ("W MM", "HEAD", "GDN", "CONV", "STATE", "FA", "DRAFT", "rest")
_HEAD = re.compile(r"^(output\.weight|token_embd\.weight)\b")
_DRAFT = re.compile(r"nextn|mtp_")
_STATE = re.compile(r"cache_s_l|cache_r_l")


def op_class(op: str, names: str) -> str:
    """The class of one profile-op line from its op name and its tensor names. DRAFT holds each op of the MTP draft
    graph (its names hold nextn or mtp_). STATE holds the copies of the recurrent state (GET_ROWS and CPY of the
    cache_s and cache_r tensors) outside the fused ops. O(length of the names)."""
    parts = op.split("+")
    src0 = names.split(" x ")[0].strip()
    if _DRAFT.search(names):
        return "DRAFT"
    if any(p.startswith("MUL_MAT") for p in parts):
        if _HEAD.match(src0):
            return "HEAD"
        if ".weight" in names:
            return "W MM"
        return "rest"
    if "FLASH_ATTN_EXT" in parts:
        return "FA"
    if any(p.startswith(("GATED_DELTA_NET", "GDN_STATE_STEP")) for p in parts):
        return "GDN"
    if any(p.startswith(("GDN_CONV", "SSM_CONV")) for p in parts):
        return "CONV"
    if parts[0] in ("GET_ROWS", "CPY") and _STATE.search(names):
        return "STATE"
    return "rest"


PMU_RE = re.compile(r"\|usec (\d+) cycles (\d+).*? pmu \[([\d,]+)\]")


@dataclass
class Call:
    """One call of a sweep: its TIME line fields and the sums of the log lines between its STAMP lines."""
    depth: int
    tokens: int
    call: int
    ms: float
    wait_us: int = 0
    pack_us: int = 0
    build_us: int = 0
    alloc_us: int = 0
    dsp_us: int = 0
    batches: int = 0
    classes: Counter = field(default_factory=Counter)
    ops: list = field(default_factory=list)  # (op, names, usec, cycles, pmu list or None)

    @property
    def host_ms(self) -> float:
        """The wall time minus the time the host waited for the DSP."""
        return self.ms - self.wait_us / 1000.0


def read_calls(root: Path, name: str, keep_ops: bool = False) -> tuple[list[Call], str, str, str]:
    """The calls of one sweep run, and its gate, stdout and stderr text. The k-th call-begin STAMP line belongs to
    the k-th TIME sweep line. O(lines of the log)."""
    texts = [(root / f"{name}{s}").read_text(errors="replace") if (root / f"{name}{s}").exists() else ""
             for s in ("-gate.txt", ".out", ".log")]
    gate, out, log = texts
    events, lines = fixed.parse_log(log)
    times = fixed.kv_lines(out, "TIME sweep ")
    # A run with --cold writes the span of its cold call first
    spans = fixed.pairs(events, "call-begin", "call-end")[len(fixed.kv_lines(out, "TIME cold ")):]
    calls = []
    for i, t in enumerate(times):
        c = Call(int(t["depth"]), int(t["tokens"]), int(t["call"]), float(t["ms"]))
        if i < len(spans):
            b, e = spans[i]
            for _, body in lines[b.line:e.line + 1]:
                m = fixed.DECODE_RE.search(body)
                if m:
                    for k, v in fixed.FIELD_RE.findall(m.group(4)):
                        if k == "build":
                            c.build_us += int(v)
                        elif k == "alloc":
                            c.alloc_us += int(v)
                    continue
                m = fixed.SESSION_RE.search(body)
                if m:
                    c.pack_us += int(m.group(6))
                    c.wait_us += int(m.group(8))
                    continue
                m = fixed.OPBATCH_RE.search(body)
                if m:
                    c.dsp_us += int(m.group(2))
                    c.batches += 1
                    continue
                m = fixed.OP_RE.search(body)
                if m:
                    c.classes[op_class(m.group(1), m.group(2))] += int(m.group(3))
                    if keep_ops:
                        p = PMU_RE.search(body)
                        c.ops.append((m.group(1), m.group(2), int(m.group(3)), int(p.group(2)) if p else 0,
                                      [int(x) for x in p.group(3).split(",")] if p else None))
        calls.append(c)
    return calls, gate, out, log


def conditions(gate: str, log: str) -> tuple[bool, list[str], list[str]]:
    """True when the run ran with the exit code 0, the marks of the run, and the conditions that keep it out of the
    tables."""
    before, after = fixed.GATE_RE.search(gate), fixed.AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    ok = "gate: OK" in gate and rc is not None and rc.group(1) == "0"
    marks, removed = [], []
    if not gate:
        marks.append("no gate file")
    elif "gate: OK" not in gate:
        marks.append("the gate stopped the run")
    elif not ok:
        marks.append(f"exit code {rc.group(1) if rc else '?'}")
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        marks.append(f"caps {before.group(3)}/{before.group(4)} -> {after.group(2)}/{after.group(3)}")
    caps = [int(v) for v in ((before.group(3), before.group(4)) if before else ()) +
            ((after.group(2), after.group(3)) if after else ()) if v]
    if caps and min(caps) < CAP_MIN_KHZ:
        removed.append(f"a cap of {min(caps)} kHz")
    if after and after.group(1) not in ("", "0"):
        removed.append(f"thermal {after.group(1)} after the run")
    if re.search(r"follow-failed|GGML_ASSERT|dspqueue_read failed|AddressSanitizer", log):
        removed.append("the log has a failure line")
    return ok, marks, removed


med = fixed.med
fmt = fixed.fmt


def usable(root: Path, name: str, include_all: bool) -> bool:
    """True when the run goes into the tables."""
    gate = (root / f"{name}-gate.txt").read_text(errors="replace") if (root / f"{name}-gate.txt").exists() else ""
    log = (root / f"{name}.log").read_text(errors="replace") if (root / f"{name}.log").exists() else ""
    ok, _, removed = conditions(gate, log)
    return ok and (include_all or not removed)


def checks(root: Path) -> list[str]:
    """The conditions of each run."""
    out = []
    for r in RUNS:
        gate = (root / f"{r.name}-gate.txt").read_text(errors="replace") if (root / f"{r.name}-gate.txt").exists() else ""
        log = (root / f"{r.name}.log").read_text(errors="replace") if (root / f"{r.name}.log").exists() else ""
        ok, marks, removed = conditions(gate, log)
        g = fixed.GATE_RE.search(gate)
        cond = f"caps {g.group(3)}/{g.group(4)} battery {g.group(5)}% temp {g.group(6)}" if g else "no gate line"
        state = "ok" if ok and not removed else "REMOVED" if ok else "NOT OK"
        out.append(f"  {r.name:9s} {state:7s} {cond}" + (f" | {', '.join(marks + removed)}" if marks or removed else ""))
    return out


def sweep_table(root: Path, lib: str, draft: str, include_all: bool) -> list[str]:
    """The main table of one library set and draft state: the calls of sw (the wall time and the host part) and of
    pf (the DSP busy time and the op classes), the medians over the calls."""
    sw, pf = f"sw-{lib}-{draft}", f"pf-{lib}-{draft}"
    rows: dict[tuple[int, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    if usable(root, sw, include_all):
        for c in read_calls(root, sw)[0]:
            key = "first" if c.call == 0 else "later"
            rows[(c.depth, c.tokens)][key].append(c)
    if usable(root, pf, include_all):
        for c in read_calls(root, pf)[0]:
            rows[(c.depth, c.tokens)]["prof"].append(c)
    if not rows:
        return [f"{sw} / {pf}: no usable run"]
    out = [f"{LIBS[lib].text}, {DRAFT_TEXT[draft]} ({sw}, {pf}): ms, the median over the calls. first = the first call "
           f"of a size (a new graph), later = the later calls (the graph reused). host = wall minus the DSP wait of the "
           f"host. DSP = the sum of the DSP batches (op profile). The classes are the DSP ms of the op profile.",
           f"  {'depth':>5} {'n':>3} | {'first':>6} {'later':>6} | {'host1':>5} {'hostL':>5} {'build1':>6} {'alloc1':>6} "
           f"{'pack1':>6} {'packL':>6} | {'DSP':>6} {'bat':>3} | " + " ".join(f"{c:>6}" for c in CLASSES)]
    fits: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for (depth, n) in sorted(rows):
        r = rows[(depth, n)]
        first, later, prof = r["first"], r["later"], r["prof"]
        f_ms, l_ms = med(c.ms for c in first), med(c.ms for c in later)
        if f_ms is not None:
            fits[(depth, "first")].append((n, f_ms))
        if l_ms is not None:
            fits[(depth, "later")].append((n, l_ms))
        cls = " ".join(f"{fmt(med(c.classes.get(k, 0) / 1000 for c in prof)):>6}" for k in CLASSES)
        out.append(
            f"  {depth:>5} {n:>3} | {fmt(f_ms):>6} {fmt(l_ms):>6} | {fmt(med(c.host_ms for c in first)):>5} "
            f"{fmt(med(c.host_ms for c in later)):>5} {fmt(med(c.build_us / 1000 for c in first), 2):>6} "
            f"{fmt(med(c.alloc_us / 1000 for c in first), 2):>6} {fmt(med(c.pack_us / 1000 for c in first), 2):>6} "
            f"{fmt(med(c.pack_us / 1000 for c in later), 2):>6} | {fmt(med(c.dsp_us / 1000 for c in prof)):>6} "
            f"{fmt(med(c.batches for c in prof), 0):>3} | {cls}")
    for (depth, which), pts in sorted(fits.items()):
        f = fixed.fit(pts)
        if f:
            out.append(f"  fit depth {depth} {which}: T = {f[0]:.1f} + {f[1]:.3f} x n ms (n = 1 to 64)")
    return out


def compare_table(root: Path, draft: str, include_all: bool) -> list[str]:
    """The candidates against HEAD for each size and depth: the wall time of the later calls, and the DSP ms of the
    GDN class and of the DRAFT class."""
    data = {}
    for lib in "bn":
        sw = read_calls(root, f"sw-{lib}-{draft}")[0] if usable(root, f"sw-{lib}-{draft}", include_all) else []
        pf = read_calls(root, f"pf-{lib}-{draft}")[0] if usable(root, f"pf-{lib}-{draft}", include_all) else []
        data[lib] = (sw, pf)
    out = [f"The candidates (n) against HEAD (b), {DRAFT_TEXT[draft]}: the wall ms of the later calls, and the DSP ms "
           f"of the classes GDN and DRAFT (op profile)",
           f"  {'depth':>5} {'n':>3} | {'wall b':>7} {'wall n':>7} {'diff':>6} | {'GDN b':>6} {'GDN n':>6} | "
           f"{'DRAFT b':>7} {'DRAFT n':>7}"]
    keys = sorted({(c.depth, c.tokens) for lib in "bn" for c in data[lib][0] + data[lib][1]})
    for depth, n in keys:
        def val(lib: str, i: int, get) -> float | None:
            return med(get(c) for c in data[lib][i] if c.depth == depth and c.tokens == n and (i == 1 or c.call > 0))
        wb, wn = val("b", 0, lambda c: c.ms), val("n", 0, lambda c: c.ms)
        diff = wn - wb if wb is not None and wn is not None else None
        out.append(f"  {depth:>5} {n:>3} | {fmt(wb):>7} {fmt(wn):>7} {fmt(diff):>6} | "
                   f"{fmt(val('b', 1, lambda c: c.classes.get('GDN', 0) / 1000), 2):>6} "
                   f"{fmt(val('n', 1, lambda c: c.classes.get('GDN', 0) / 1000), 2):>6} | "
                   f"{fmt(val('b', 1, lambda c: c.classes.get('DRAFT', 0) / 1000), 2):>7} "
                   f"{fmt(val('n', 1, lambda c: c.classes.get('DRAFT', 0) / 1000), 2):>7}")
    return out


_LAYER = re.compile(r"blk\.(\d+)\.(\w+)\.weight")


def dma_table(root: Path, include_all: bool) -> list[str]:
    """The weight matmuls of the run pm (the PMU set dma-wait) at each size: for each weight kind, the time, the
    weight rate and the shares of the op cycles with the DMA active and with the DMA read buffer full, the median
    over the layers and the calls, and the slowest layer."""
    name = "pm-b-n"
    if not usable(root, name, include_all):
        return [f"{name}: no usable run"]
    calls = read_calls(root, name, keep_ops=True)[0]
    per: dict[tuple[int, str], list] = defaultdict(list)
    for c in calls:
        for op, names, usec, cycles, pmu in c.ops:
            m = _LAYER.search(names)
            if not m or op_class(op, names) != "W MM" or pmu is None or cycles == 0:
                continue
            per[(c.tokens, m.group(2))].append((usec, int(m.group(1)), pmu[0] / cycles, pmu[3] / cycles,
                                                pmu[1] / cycles))
    out = ["pm-b-n: the weight matmuls at the depth 700, draft off, the PMU set dma-wait. us = the median op time, "
           "active = UDMA_ACTIVE / op cycles, full = UDMA_RD_BUFFER_LEVEL_FULL / op cycles, poll = UDMA_DMPOLL / op "
           "cycles (medians), slowest = the layer and the time of the slowest op",
           f"  {'n':>3} {'weight':>12} | {'ops':>4} {'us':>6} {'min':>6} | {'active':>6} {'full':>6} {'poll':>6} | slowest"]
    for (n, kind) in sorted(per):
        vals = per[(n, kind)]
        slow = max(vals, key=lambda v: v[0])
        out.append(f"  {n:>3} {kind:>12} | {len(vals):>4} {med(v[0] for v in vals):>6.0f} {min(v[0] for v in vals):>6} | "
                   f"{med(v[2] for v in vals):>6.2f} {med(v[3] for v in vals):>6.2f} {med(v[4] for v in vals):>6.2f} | "
                   f"blk.{slow[1]} {slow[0]} us (active {slow[2]:.2f}, full {slow[3]:.2f})")
    return out


def layer_table(root: Path, include_all: bool) -> list[str]:
    """The time of each weight matmul of each layer at 22 tokens against 1 token (pf-b-n, the depth 2300, the median
    of the 2 calls): a layer that is slow in both calls at 22 tokens and not at 1 token points at the HMX path, not
    at the weight."""
    name = "pf-b-n"
    if not usable(root, name, include_all):
        return [f"{name}: no usable run"]
    calls = read_calls(root, name, keep_ops=True)[0]
    t: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for c in calls:
        if c.depth != 2300 or c.tokens not in (1, 22):
            continue
        for op, names, usec, _, _ in c.ops:
            m = _LAYER.search(names)
            if m and op_class(op, names) == "W MM" and m.group(2) in ("attn_qkv", "ffn_gate", "ffn_down"):
                t[(c.tokens, m.group(2), int(m.group(1)))].append(usec)
    out = ["pf-b-n: us of each layer, 1 token | 22 tokens (the median of the 2 calls at the depth 2300)"]
    for kind in ("attn_qkv", "ffn_gate", "ffn_down"):
        layers = sorted({k[2] for k in t if k[1] == kind})
        cells = [f"{l}:{fmt(med(t.get((1, kind, l), [])), 0)}|{fmt(med(t.get((22, kind, l), [])), 0)}" for l in layers]
        out.append(f"  {kind}: " + " ".join(cells))
    return out


def long_table(root: Path, include_all: bool) -> list[str]:
    """The prefill of 512 and 1024 tokens with the draft on, HEAD against the candidates (the MTP concat)."""
    out = ["lg: ms of 512 and 1024 tokens at the depth 0 with the draft on, call 0 (a new graph) and call 1"]
    for lib in "bn":
        name = f"lg-{lib}-s"
        if not usable(root, name, include_all):
            out.append(f"  {name}: no usable run")
            continue
        calls = read_calls(root, name)[0]
        out.append(f"  {name} ({LIBS[lib].text}): " + ", ".join(f"n{c.tokens} call {c.call} {c.ms:.1f}" for c in calls))
    return out


def bench_table(root: Path, include_all: bool) -> list[str]:
    """pp512 and tg32 of llama-bench, HEAD against the candidates."""
    out = ["bn: llama-bench t/s (the median of the 3 repetitions)"]
    for lib in "bn":
        name = f"bn-{lib}--"
        if not usable(root, name, include_all):
            out.append(f"  {name}: no usable run")
            continue
        text = (root / f"{name}.out").read_text(errors="replace")
        cells = []
        for line in text.splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                cells.append(f"pp{rec['n_prompt']} tg{rec['n_gen']} {statistics.median(rec['samples_ts']):.2f}")
        out.append(f"  {name} ({LIBS[lib].text}): " + ", ".join(cells))
    return out


KLD_RE = re.compile(r"Mean\s+KLD:\s+([\d.]+) ±\s+([\d.]+)")
TOP_RE = re.compile(r"Same top p:\s+([\d.]+) ±\s+([\d.]+)")
MAXKL_RE = re.compile(r"Maximum KLD:\s+([\d.]+)")
PPL_RE = re.compile(r"Mean PPL\(Q\)\s+:\s+([\d.]+)")


def kl_table(root: Path) -> list[str]:
    """The KL of the prefill path against the naive base, for each KL run."""
    out = ["KL against naive-4B-q8.kld: mean ± error, maximum, same top p ± error, PPL"]
    for r in RUNS:
        if r.tool != "llama-perplexity":
            continue
        p = root / f"{r.name}.log"
        text = ((root / f"{r.name}.out").read_text(errors="replace") if (root / f"{r.name}.out").exists() else "") + \
            (p.read_text(errors="replace") if p.exists() else "")
        m, t, mx, ppl = KLD_RE.search(text), TOP_RE.search(text), MAXKL_RE.search(text), PPL_RE.search(text)
        cell = (f"{m.group(1)} ± {m.group(2)}, max {mx.group(1) if mx else '?'}, top-1 "
                f"{t.group(1) + ' ± ' + t.group(2) if t else '?'} %, PPL {ppl.group(1) if ppl else '?'}") if m else "no KLD line"
        out.append(f"  {r.name:9s} {r.text:62s}: {cell}")
    return out


def ops_table(root: Path) -> list[str]:
    """The result lines of the op tests."""
    out = ["tb: test-backend-ops of CONCAT and GATED_DELTA_NET on HTP0 against the CPU"]
    for lib in "bn":
        name = f"tb-{lib}--"
        p = root / f"{name}.out"
        text = p.read_text(errors="replace") if p.exists() else ""
        passed = re.findall(r"(\d+)/(\d+) tests passed", text)
        fails = [line.strip() for line in text.splitlines() if "FAIL" in line][:12]
        out.append(f"  {name} ({LIBS[lib].text}): " + (", ".join(f"{a}/{b} passed" for a, b in passed) or "no result line"))
        out += [f"    {line[:200]}" for line in fails]
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    parts = [["The runs:"] + checks(root)]
    for lib in "bn":
        for draft in "ns":
            parts.append(sweep_table(root, lib, draft, include_all))
    parts += [compare_table(root, "n", include_all), compare_table(root, "s", include_all),
              layer_table(root, include_all), dma_table(root, include_all), long_table(root, include_all),
              bench_table(root, include_all), kl_table(root), ops_table(root)]
    for part in parts:
        print("\n".join(part))
        print()
    print("The phase timeline of the HMX matmuls (tr-b-n, level 3):\n"
          f"  tools/trace/htp_trace.py summary {root}/tr-b-n.log\n"
          f"  tools/trace/htp_trace.py convert {root}/tr-b-n.log -o /tmp/spf-tr.json   (ui.perfetto.dev)")
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
        print(f"{a.out}: {n} lines, {len(RUNS)} runs")
        return 0
    return table(a.root, a.all)


if __name__ == "__main__":
    sys.exit(main())
