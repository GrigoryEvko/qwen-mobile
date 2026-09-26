#!/usr/bin/env python3
"""The phone stage "mmsolve": the chunks of the HMX 2D matmul, the old cost model against the cost of the kernel.

Usage (tools/stages/mmsolve/build.sh makes the libraries and the tools first, then runs "files"):
    stage.py files [--only bx]            write phone/tests, the checksum file and the command file
                                          (phone-commands.txt, or phone-commands-bx.txt for the runs bx-*)
    stage.py table [--only bx] [--root DIR]   print the tables from the pulled outputs (phone-out or phone-out-bx)

One library set holds the two models. GGML_HEXAGON_MM_SOLVER=0 gives the old model (the activation conversion
is a cost of each weight chunk) and 1 the cost of the kernel (the patch default). GGML_HEXAGON_MM_CHUNKS=mc,nc
sets the chunks of each HMX 2D matmul when their layout is in the VTCM budget. Each profile line of an HMX
matmul gives "mc M nc N", thus the table reads the chunks that each op got.

The questions and their runs:
    1. Is the output the same, bit for bit? mmcheck hashes each 4B shape (the forms MUL_MAT, MUL_MAT_NX and
       MUL_MAT+ADD of the model) at 5 to 1024 tokens, with the old model, the new model and the request 0,32
       (the narrowest n chunk with the largest m chunk), and compares each with the CPU backend.
    2. Does test-backend-ops pass the MUL_MAT cases of the 4B shapes with the new model?
    3. What does each 4B op cost, old against new? test-backend-ops perf of each shape at 512 and 1024
       tokens, 2 rounds, with the DSP time of each op copy (fresh and sustained).
    4. What does a chunk cost? A sweep of the chunks of ffn_down and of 4096x2560 at 1024 tokens, and a probe
       where the chunks are small and the DMA is fast: 2560x9216 at 32 tokens with n chunks of 32 to 480.
    5. Is the model faster? llama-bench pp512 and pp1024 at d0 and d3072, and tg32 at d0, for the 4B with the
       flags of the stage bench-kv (variant b: Q8_0 K and V), 3 rounds in alternated order.
    6. Where does the time go? One GGML_HEXAGON_PROFILE=1 llama-bench pp1024 of each model: the op split.

A run name is the stem of its output files on the phone: <run>-gate.txt (the conditions and the exit code),
<run>.out (stdout) and <run>.log or <run>.log.z (stderr, gzip when the phone has it). The table only reads
files. A run with a flag (changed caps, heat, screen) stays in the tables with a "*" after its name.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cli, commands, device, gate, logs, tables  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def _load_sweep():
    """The module of tools/stages/sweep/stage.py: its test-file lines, its case type and its parsers."""
    spec = importlib.util.spec_from_file_location("sweep_stage", REPO / "tools/stages/sweep/stage.py")
    mod = importlib.util.module_from_spec(spec)
    # The dataclasses of the module read their module from sys.modules while it runs
    sys.modules["sweep_stage"] = mod
    spec.loader.exec_module(mod)
    return mod


sweep = _load_sweep()
mm = sweep.mm

# ---- The stage paths and the phone lines ----

ADB = device.ADB
PATHS = device.stage_paths("mmsolve", __file__)
PHONE, LAPTOP_STAGE, BOX, STAGE_DIR = PATHS
MODEL_DIR = device.MODEL_DIR
MODEL = device.MODEL_4B
MODEL_GATE_KB = device.GATE_4B_KB
# The ls of this file makes the laptop runner treat a line as a model run: it waits for the unlocked phone,
# stops the Qwen app, wakes the screen and checks MemAvailable. The runs with this marker load no model.
MARKER = sweep.MARKER
TOOL_GATE_KB = device.GATE_TOOL_KB
# The environment of the app, as the stage bench-kv has it
APP_ENV = device.APP_ENV
# The llama-bench flags of the stage bench-kv, variant b (Q8_0 K and V, the rotation as FWHT)
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
THERMAL = device.THERMAL
TOOLS = ("llama-bench", "test-backend-op", "mmcheck")
PGREP = device.pgrep(*TOOLS)
# The before line and the after line of the stage sweep: they also record the screen state and the keyguard
# state, thus the two stages read a gate file with the same parser.
BEFORE, AFTER = sweep.BEFORE, sweep.AFTER
PHONE_FILES = ("bin/gate.sh", "bin/llama-bench", "bin/test-backend-ops", "bin/mmcheck", "lib/libggml-base.so",
               "lib/libggml-cpu.so", "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so", "lib/libggml-opencl.so",
               "lib/libggml.so", "lib/libllama-bench-impl.so", "lib/libllama-common.so", "lib/libllama.so",
               "lib/libmtmd.so")

MODELS = {"old": "GGML_HEXAGON_MM_SOLVER=0", "new": "GGML_HEXAGON_MM_SOLVER=1"}

# ---- The cases ----

# The Q8_0 weights of the 4B (k, n) and the F16 weights of its MTP block (blk.32)
Q8_SHAPES = ((2560, 9216), (9216, 2560), (2560, 8192), (2560, 4096), (4096, 2560), (2560, 1024))
DOWN, OUT, GATE, KV = (9216, 2560), (4096, 2560), (2560, 9216), (2560, 1024)


def op_cases() -> tuple:
    """The cases of the per-op A/B: each 4B shape at 1024 tokens, the shapes whose chunks change at 512 tokens,
    two F16 shapes of the MTP block at 1024 tokens, and the probe shape at 32 tokens (the solver chunks)."""
    return (*(mm("q8_0", k, n, 1024) for k, n in Q8_SHAPES),
            mm("q8_0", *DOWN, 512), mm("q8_0", *OUT, 512),
            mm("f16", *DOWN, 1024), mm("f16", 5120, 2560, 1024),
            mm("q8_0", *GATE, 32))


def tbo_cases() -> tuple:
    """The MUL_MAT cases of test-backend-ops test mode: each 4B Q8_0 shape at 5, 64, 512 and 1024 tokens, and
    the counts with a partial last token chunk of the new model."""
    out = [mm("q8_0", k, n, m) for k, n in Q8_SHAPES for m in (5, 64, 512, 1024)]
    out += [mm("q8_0", *DOWN, 1000), mm("q8_0", *OUT, 896), mm("q8_0", *GATE, 900), mm("f16", *DOWN, 768),
            mm("f16", 5120, 2560, 640)]
    return tuple(out)


@dataclass(frozen=True)
class Point:
    """One point of the chunk sweep: a shape, a token count and the request of GGML_HEXAGON_MM_CHUNKS."""
    k: int
    n: int
    m: int
    mc: int
    nc: int
    t: str = "q8_0"

    @property
    def case(self):
        """The test-backend-ops case. The tag keeps the name unique in the stage."""
        return mm(self.t, self.k, self.n, self.m, tag=f"c{self.mc}x{self.nc}")


# The sweep points. The old and the new chunks of each shape come from the op runs.
#   ffn_down at 1024: old 128x96 (8 passes), new 352x32 (3 passes). 128x32 has the passes of the old model and
#   three times its chunks. 224x64 and 256x32 have 5 and 4 passes.
#   4096x2560 at 1024: old 384x192 (3 passes), new 576x128 (2 passes). 512x128, 768x64 and 896x32 have 2
#   passes and 40, 80 and 160 chunks.
#   The probe 2560x9216 at 32 tokens: 1 pass, the DMA of the weights takes about 410 us, and the solver takes
#   nc 480 (20 chunks). n chunks of 128, 64 and 32 give 72, 144 and 288 chunks.
POINTS = (
    Point(*DOWN, 1024, 128, 32), Point(*DOWN, 1024, 224, 64), Point(*DOWN, 1024, 256, 32),
    Point(*OUT, 1024, 512, 128), Point(*OUT, 1024, 768, 64), Point(*OUT, 1024, 896, 32),
    Point(*GATE, 32, 32, 128), Point(*GATE, 32, 32, 64), Point(*GATE, 32, 32, 32),
)

# ---- The runs ----


@dataclass(frozen=True)
class Run:
    """One phone run. kind: "mmcheck", "tbo-test", "tbo-perf", "bench" or "prof". env holds the switches of the
    run. args: the arguments of the tool. cases: the test-backend-ops cases of the run (its test file)."""
    key: str
    kind: str
    text: str
    env: str
    args: str = ""
    cases: tuple = ()
    limit: int = 110
    model: str = ""   # "old" or "new" for the A/B runs, else ""


def runs() -> list[Run]:
    """The runs of the stage in the order of the command file."""
    out: list[Run] = []
    # 1. The bit-exact check with the CPU reference, and test-backend-ops test mode. bx-off has the fusion off:
    # each MUL_MAT and each ADD is an op of its own.
    for key, env in (("old", f"{APP_ENV} {MODELS['old']}"), ("new", f"{APP_ENV} {MODELS['new']}"),
                     ("nc32", f"{APP_ENV} {MODELS['new']} GGML_HEXAGON_MM_CHUNKS=0,32"),
                     ("off", f"GGML_HEXAGON_OPFUSION=0 {MODELS['new']}")):
        out.append(Run(f"bx-{key}", "mmcheck", f"mmcheck --cpu, {env}", f"{env} GGML_HEXAGON_PROFILE=1",
                       "--cpu --threads 4", limit=100, model=key))
    out.append(Run("tbo-new", "tbo-test", "test-backend-ops test mode, the MUL_MAT cases of the 4B shapes, new model",
                   f"GGML_HEXAGON_OPFUSION=0 {MODELS['new']}", cases=tbo_cases(), limit=100, model="new"))
    # 2. The per-op A/B: 2 rounds, old then new, then new then old
    for rnd, order in ((1, ("old", "new")), (2, ("new", "old"))):
        for model in order:
            out.append(Run(f"op-{rnd}-{model}", "tbo-perf", f"test-backend-ops perf of the 4B shapes, {model} model",
                           f"GGML_HEXAGON_OPFUSION=0 {MODELS[model]} GGML_HEXAGON_PROFILE=1", cases=op_cases(),
                           limit=100, model=model))
    # 3. The chunk sweep: one request for each run
    for p in POINTS:
        out.append(Run(f"cs-{p.k}x{p.n}-m{p.m}-{p.mc}x{p.nc}", "tbo-perf",
                       f"chunk sweep, {p.k}x{p.n} at {p.m} tokens, request {p.mc},{p.nc}",
                       f"GGML_HEXAGON_OPFUSION=0 {MODELS['new']} GGML_HEXAGON_MM_CHUNKS={p.mc},{p.nc} "
                       f"GGML_HEXAGON_PROFILE=1", cases=(p.case,), limit=60))
    # 4. llama-bench, 3 rounds: old new, new old, old new. The prefill run and the decode run of one model
    # go one after the other.
    for rnd, order in ((1, ("old", "new")), (2, ("new", "old")), (3, ("old", "new"))):
        for model in order:
            out.append(Run(f"pp-{rnd}-{model}", "bench", f"llama-bench pp512 and pp1024 at d0 and d3072, {model} model",
                           f"{APP_ENV} {MODELS[model]}", "-p 512,1024 -n 0 -d 0,3072 -r 3", limit=110, model=model))
            out.append(Run(f"tg-{rnd}-{model}", "bench", f"llama-bench tg32 at d0, {model} model",
                           f"{APP_ENV} {MODELS[model]}", "-p 0 -n 32 -d 0 -r 3", limit=80, model=model))
    # 5. The op split of one pp1024 graph (the warmup graph and the timed graph). Without -v, llama-bench sets a log
    # callback that drops the debug lines of ggml, thus it writes no profile line.
    for model in ("old", "new"):
        out.append(Run(f"prof-{model}", "prof", f"GGML_HEXAGON_PROFILE=1, llama-bench pp1024 at d0, {model} model",
                       f"{APP_ENV} {MODELS[model]} GGML_HEXAGON_PROFILE=1", "-p 1024 -n 0 -d 0 -r 1 -v", limit=90,
                       model=model))
    return out


# The seconds of each kind of run without the gate, for the time of the stage (estimates, not results)
EST_S = {"mmcheck": 45, "tbo-test": 50, "tbo-perf": 35, "bench-pp": 55, "bench-tg": 22, "prof": 25}


def est_s(r: Run) -> float:
    """The estimated tool time of one run in seconds."""
    if r.kind == "bench":
        return EST_S["bench-pp" if r.key.startswith("pp") else "bench-tg"]
    if r.kind == "tbo-perf" and r.key.startswith("cs-"):
        return 8
    return EST_S[r.kind]


# The binary of each kind of run
TOOL_OF = {"mmcheck": "bin/mmcheck", "tbo-test": "bin/test-backend-ops", "tbo-perf": "bin/test-backend-ops",
           "bench": "bin/llama-bench", "prof": "bin/llama-bench"}


@dataclass(frozen=True)
class Target:
    """The runs of one command file and their paths: the full stage (suffix "", each run), or the correctness
    stage (suffix "-bx", the runs bx-*). Each target has its own phone directory, output directory, command file
    and checksum file, thus the outputs of one target do not replace the outputs of the other."""
    suffix: str
    prefix: str

    @property
    def phone(self) -> str:
        """The work directory on the phone."""
        return PHONE + self.suffix

    @property
    def paths(self) -> device.StagePaths:
        """The directories of the target: the directories of the stage with the phone directory of the
        target."""
        return PATHS._replace(phone=self.phone)

    @property
    def out(self) -> str:
        """The output directory below build/mmsolve, on the laptop and on the box."""
        return f"phone-out{self.suffix}"

    @property
    def commands(self) -> str:
        """The command file below build/mmsolve."""
        return f"phone-commands{self.suffix}.txt"

    @property
    def sums(self) -> str:
        """The checksum file below build/mmsolve/phone, for the files that this target pushes."""
        return f"SHA256SUMS{self.suffix}"

    def runs(self) -> list[Run]:
        """The runs of the target, in the order of the stage."""
        return [r for r in runs() if r.key.startswith(self.prefix)]

    def files(self) -> tuple[str, ...]:
        """The files below build/mmsolve/phone that the runs of the target use: the gate, their binaries and
        each library."""
        tools = {TOOL_OF[r.kind] for r in self.runs()}
        return tuple(f for f in PHONE_FILES if f == "bin/gate.sh" or f in tools or f.startswith("lib/"))


TARGETS = {"all": Target("", ""), "bx": Target("-bx", "bx-")}


# ---- The command file ----

HEADER = """\
# Phone stage "mmsolve": the chunks of the HMX 2D matmul, the old cost model (GGML_HEXAGON_MM_SOLVER=0) against the
# cost of the kernel (GGML_HEXAGON_MM_SOLVER=1, the patch default), in one library set.
#
# The libraries: the patched llama.cpp tree of HEAD plus the patch {patch} (sha256 {patch_sha}), built with the preset,
# the flags and the LTO of scripts/build-native.sh (tools/stages/mmsolve/build.sh). test-backend-ops, llama-bench and
# mmcheck come from the same tree.
#
# The runs ({n_runs}):
#   bx-old, bx-new, bx-nc32  mmcheck --cpu: a hash of the HTP0 output of each 4B shape (MUL_MAT, MUL_MAT_NX, MUL_MAT+ADD)
#                            at 5 to 1024 tokens, and the NMSE against the CPU. Old model, new model, request 0,32
#   bx-off                   the same with the fusion off (each MUL_MAT and ADD alone), new model
#   tbo-new                  test-backend-ops test mode, the MUL_MAT cases of the 4B shapes, new model
#   op-R-old, op-R-new       test-backend-ops perf of each 4B shape at 512 and 1024 tokens, 2 rounds, profile on
#   cs-...                   the chunk sweep (GGML_HEXAGON_MM_CHUNKS): ffn_down and 4096x2560 at 1024 tokens, and the
#                            probe 2560x9216 at 32 tokens with n chunks of 128, 64 and 32
#   pp-R-M, tg-R-M           llama-bench 4B Q8_0, the flags of bench-kv variant b: pp512 and pp1024 at d0 and d3072 (-r 3),
#                            then tg32 at d0 (-r 3). 3 rounds: old new, new old, old new
#   prof-old, prof-new       GGML_HEXAGON_PROFILE=1, llama-bench pp1024 at d0 -r 1: the op split
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB for a model run and 2 GB for the others), the tool under timeout -s KILL (110 s or less), the exit code and the
# conditions after the run, then the pgrep line. The runs of mmcheck and test-backend-ops name the 2B model file in an
# ls (they load no model), thus the runner treats them as model runs.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about {tool_min:.0f} minutes of tool time,
# about {total_min:.0f} minutes with the gates, plus the waits for thermal status 0.
# Then, on the box: build/mmsolve/stage.py table
"""

HEADER_BX = """\
# Phone stage "mmsolve-bx": the bit-exact check of the chunks of the HMX 2D matmul on HTP0 (the runs bx-* of the stage
# mmsolve). The libraries are the files of build/mmsolve/phone: the patched llama.cpp tree of HEAD plus the patch {patch}
# (sha256 {patch_sha}), with the preset, the flags and the LTO of scripts/build-native.sh (tools/stages/mmsolve/build.sh).
#
# The runs ({n_runs}): mmcheck --cpu gives a hash of the HTP0 output of each 4B shape at 5 to 1024 tokens (Q8_0 and the
# F16 MTP block), and the NMSE and the largest error against the CPU backend of the phone. mmcheck puts the weights in a
# buffer with GGML_BACKEND_BUFFER_USAGE_WEIGHTS, thus the backend repacks a Q8_0 weight as the model loader has it.
#   bx-old    the old cost model (GGML_HEXAGON_MM_SOLVER=0), fusion on (MUL_MAT, MUL_MAT_NX, MUL_MAT+ADD)
#   bx-new    the cost model of the kernel (GGML_HEXAGON_MM_SOLVER=1), fusion on
#   bx-nc32   the request GGML_HEXAGON_MM_CHUNKS=0,32 (the narrowest n chunk with the largest m chunk), fusion on
#   bx-off    the cost model of the kernel, fusion off (each MUL_MAT and ADD alone)
# Each case of each run must give 0 values that are not finite and an NMSE of 1e-4 or less (the exit code 0), and the
# hashes of bx-old, bx-new and bx-nc32 must be the same.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 2 GB), mmcheck under timeout -s KILL (100 s), the exit code and the conditions after the run, then the pgrep line. Each
# run line names the 2B model file in an ls (no run loads a model), thus the runner treats it as a model run.
#
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. It is not a timing stage. Time: about
# {tool_min:.0f} minutes of tool time, about {total_min:.0f} minutes with the gates.
# Then, on the box: build/mmsolve/stage.py table --only bx
"""


def setup_lines(t: Target) -> list[str]:
    """The lines that copy the files of a target to the phone and check them. Each target has its own
    checksum file (SHA256SUMS-bx for the correctness stage), and the runs of a target can have no test
    file."""
    files = t.files()
    pushes = {
        "bin": [f"{LAPTOP_STAGE}/phone/{f}" for f in files if f.startswith("bin/")],
        "lib": [f"{LAPTOP_STAGE}/phone/{f}" for f in files if f.startswith("lib/")],
        "tests": [f"{LAPTOP_STAGE}/phone/tests/{r.key}.txt" for r in t.runs() if r.cases],
    }
    # No "models/Qwen3.5" in the model line: the runner gates each line with that text as a model run.
    return commands.setup_lines(
        t.paths, pushes, sums=t.sums, model_check=f"{ADB} shell 'ls -l {MODEL_DIR} | grep {MODEL}'",
        extra=[f"{ADB} shell 'echo gzip: $(command -v gzip) timeout: $(command -v timeout)'"])


def run_lines(r: Run, t: Target) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    p = t.phone
    stem = f"{p}/out/{r.key}"
    model_run = r.kind in ("bench", "prof")
    gate_kb = MODEL_GATE_KB if model_run else TOOL_GATE_KB
    prefix = "" if model_run else f"{MARKER}; "
    head = commands.gate_head(stem, gate_kb, p, prefix=prefix, before=BEFORE)
    tail = commands.gate_tail(stem, after=AFTER)
    env = f"{device.lib_env(p)} {r.env}"
    if r.kind == "mmcheck":
        cmd = f"{p}/bin/mmcheck {r.args}"
    elif r.kind == "tbo-test":
        cmd = f"{p}/bin/test-backend-ops test -b HTP0 --test-file {p}/tests/{r.key}.txt"
    elif r.kind == "tbo-perf":
        cmd = f"{p}/bin/test-backend-ops perf -b HTP0 --test-file {p}/tests/{r.key}.txt"
    else:
        cmd = f"{p}/bin/llama-bench -m {MODEL_DIR}/{MODEL} {BENCH_ARGS} {r.args}"
    full = commands.timeout_cmd(r.limit, env, cmd)
    # The tool part is one group: a gate that fails skips all of it, and $? after the group is the exit code
    # of the tool.
    if "GGML_HEXAGON_PROFILE" in r.env:
        tool = commands.gzip_group(stem, full)
    else:
        tool = f"{{ {commands.redirect(stem, full)}; }}; "
    title = f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: " if model_run else "# MMSOLVE "
    return commands.run_lines(f"{title}{r.key}: {r.text} (estimate {est_s(r):.0f} s, limit {r.limit} s)",
                              f"{head}{tool}{tail}", PGREP)


def output_lines(t: Target) -> list[str]:
    """The lines that pull the outputs of a target, copy them to the box and remove its phone directory. The phone
    directory goes only when the pull has each of its files."""
    return commands.output_lines(t.paths, tools=TOOLS, out_dir=t.out)


def check_points() -> None:
    """Make sure that each sweep point has a layout in the budget (build/mmsolve/bin/plan fit).

    Raises:
        RuntimeError: If a request has no layout in the budget, or plan is not built
    """
    plan = STAGE_DIR / "bin/plan"
    if not plan.exists():
        raise RuntimeError(f"{plan} is not there: run tools/stages/mmsolve/build.sh")
    for p in POINTS:
        res = subprocess.run([str(plan), "fit", str(p.k), str(p.n), str(p.m), p.t, str(p.mc), str(p.nc)],
                             capture_output=True, text=True)
        if res.returncode != 0 or f"mc {p.mc:4d} nc {p.nc:3d}" not in res.stdout:
            raise RuntimeError(f"the sweep point {p} does not get its chunks: {res.stdout.strip()}")
        print("  " + res.stdout.strip())


def write_files(t: Target) -> int:
    """Write phone/tests, the checksum file of the target and its command file. The binaries and the libraries are
    in phone/ from build.sh. The test files of each run are written for each target, thus the full stage and the
    correctness stage use one phone directory of the box."""
    ops = ggml_ops(REPO / "build/mmsolve/stage-src/ggml/include/ggml.h")
    phone = STAGE_DIR / "phone"
    missing = [f for f in PHONE_FILES if not (phone / f).exists()]
    if missing:
        raise FileNotFoundError(f"{phone} has no {missing}: run tools/stages/mmsolve/build.sh")
    check_points()
    tests = phone / "tests"
    if tests.exists():
        shutil.rmtree(tests)
    tests.mkdir(parents=True)
    for r in runs():
        names = [c.name for c in r.cases]
        if len(set(names)) != len(names):
            raise ValueError(f"the run {r.key} has a case name two times")
        if r.cases:
            (tests / f"{r.key}.txt").write_text("".join(sweep.line(c, ops) + "\n" for c in r.cases))
    t_runs = t.runs()
    files = list(t.files()) + [f"tests/{r.key}.txt" for r in t_runs if r.cases]
    sums = [f"{hashlib.sha256((phone / f).read_bytes()).hexdigest()}  {f}" for f in files]
    (phone / t.sums).write_text("\n".join(sums) + "\n")
    patch = STAGE_DIR / "patch.sha256"
    patch_line = patch.read_text().split() if patch.exists() else ["?", "?"]
    tool = sum(est_s(r) for r in t_runs)
    head = (HEADER_BX if t.suffix else HEADER).format(
        patch=patch_line[1], patch_sha=patch_line[0][:16], n_runs=len(t_runs), tool_min=tool / 60,
        total_min=(tool + 12 * len(t_runs)) / 60)
    lines = commands.header_lines(head) + setup_lines(t)
    for r in t_runs:
        lines += run_lines(r, t)
    lines += output_lines(t)
    n = commands.write_commands(STAGE_DIR / t.commands, lines)
    for r in t_runs:
        print(f"  {r.key:28s} {r.kind:9s} {len(r.cases):3d} cases  estimate {est_s(r):4.0f} s  limit {r.limit:3d} s")
    print(f"{t.commands}: {n} lines, {len(t_runs)} runs, tool time about {tool / 60:.1f} min")
    return 0


def ggml_ops(header: Path) -> dict[str, int]:
    """The value of each name of the ggml_op enum of a ggml.h."""
    src = header.read_text()
    body = src[src.index("enum ggml_op {"):]
    body = body[:body.index("};")]
    return {n: i for i, n in enumerate(re.findall(r"^\s*(GGML_OP_\w+)", body, re.M))}


# ---- The outputs ----

CHUNK_RE = re.compile(r"mc (\d+) nc (\d+)")
PROF_RE = re.compile(r"profile-op ([A-Z_0-9+]+)\|([^|]*)\|([^|]*)\|([^|]*)\|[^|]*\|([^|]*)\|usec (\d+) cycles (\d+) "
                     r"start \d+ mhz ([\d.]+)")
MMCHECK_RE = re.compile(r"^mmcheck case=(\S+) hash=(\S+) nonfinite=(\d+)(?: nmse=(\S+) maxerr=(\S+))? us=(\d+)( FAILED)?",
                        re.M)
TEST_RE = re.compile(r"^\s+[A-Z_]+\(name=([A-Za-z0-9_x]+),[^\n]*?\):\s*(?:\S+\s+)?(OK|FAIL|not supported)", re.M)


@dataclass
class RunOut:
    """The parsed files of one run."""
    run: Run
    ok: bool
    flags: list[str]
    caps: str
    out: str = ""
    log: str = ""
    us: dict[str, float] = field(default_factory=dict)          # test-backend-ops perf: case -> us/run
    series: dict[str, list[float]] = field(default_factory=dict)  # case -> DSP us of each op copy
    chunks: dict[str, set] = field(default_factory=dict)          # case -> {(mc, nc)} of its profile lines
    bench: dict[tuple, list[float]] = field(default_factory=dict)  # (n_prompt, n_gen, n_depth) -> t/s samples


def read_run(root: Path, r: Run) -> RunOut:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    text = logs.read_text(root / f"{r.key}-gate.txt")
    cond = gate.read(text)
    flags = cond.flags
    before, after = sweep.BEFORE_RE.search(text), sweep.AFTER_RE.search(text)
    for label, m, si, ki in (("before", before, 2, 3), ("after", after, 7, 8)):
        if m and (m.group(si) != "Awake" or m.group(ki) != "false"):
            flags.append(f"screen {m.group(si)} keyguard {m.group(ki)} {label} the run")
    res = RunOut(r, cond.ok, flags, cond.caps)
    # test-backend-ops writes its OK and FAIL with ANSI color codes
    res.out = logs.read_text(root / f"{r.key}.out", strip_ansi=True)
    res.log = logs.read_text(root / f"{r.key}.log")
    both = res.out + "\n" + res.log
    if r.kind == "tbo-perf":
        names = list(sweep.NAME_RE.finditer(res.out))
        for i, m in enumerate(names):
            end = names[i + 1].start() if i + 1 < len(names) else len(res.out)
            t = sweep.TIMING_RE.search(res.out, m.end(), end)
            if t and not m.group(2):
                res.us[m.group(1)] = float(t.group(2))
        missing = [c.name for c in r.cases if c.name not in res.us]
        if cond.ok and missing:
            flags.append(f"{len(missing)} cases without a result, the first {missing[0]}")
    if r.kind in ("tbo-perf", "mmcheck", "prof"):
        for m in PROF_RE.finditer(both):
            op, dims, kp = m.group(1), m.group(3), m.group(5)
            us = int(m.group(7)) / (float(m.group(8)) or sweep.CLOCK_MHZ)
            ch = CHUNK_RE.search(kp)
            for c in r.cases:
                if sweep.match_case([c], op, dims, m.group(4)):
                    res.series.setdefault(c.name, []).append(us)
                    if ch:
                        res.chunks.setdefault(c.name, set()).add((int(ch.group(1)), int(ch.group(2))))
    if r.kind in ("bench", "prof"):
        for ln in res.out.splitlines():
            if ln.startswith("{"):
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    flags.append("a llama-bench line is not complete JSON")
                    continue
                res.bench[(rec["n_prompt"], rec["n_gen"], rec["n_depth"])] = rec["samples_ts"]
                if rec.get("type_k") != "q8_0" or rec.get("flash_attn") != 1:
                    flags.append(f"llama-bench ran type_k {rec.get('type_k')} flash_attn {rec.get('flash_attn')}")
    want = "GGML_HEXAGON_MM_SOLVER=0" in r.env
    if cond.ok and want and "old cost model (GGML_HEXAGON_MM_SOLVER=0)" not in both:
        flags.append("the log has no line of GGML_HEXAGON_MM_SOLVER=0")
    if cond.ok and "GGML_HEXAGON_MM_CHUNKS=" in r.env and "(GGML_HEXAGON_MM_CHUNKS)" not in both:
        flags.append("the log has no line of GGML_HEXAGON_MM_CHUNKS")
    return res


def mark(outs: dict[str, RunOut], key: str) -> str:
    """The run key, with "*" when the run has a flag."""
    o = outs.get(key)
    return f"{key}*" if o is not None and o.flags else key


def conditions(outs: dict[str, RunOut]) -> list[str]:
    """The conditions and the flags of each run."""
    lines = ["== 0. The conditions (measured) =="]
    for o in outs.values():
        lines.append(f"  {o.run.key:28s} {'ok' if o.ok else 'FAILED':6s} caps {o.caps}"
                     + (f"  flags: {'; '.join(o.flags)}" if o.flags else ""))
    return lines


def mmcheck_rows(o: RunOut) -> dict[str, dict]:
    """The case lines of one mmcheck run."""
    rows = {}
    for m in MMCHECK_RE.finditer(o.out):
        rows[m.group(1)] = {"hash": m.group(2), "nonfinite": int(m.group(3)), "nmse": m.group(4),
                            "maxerr": m.group(5), "us": int(m.group(6)), "failed": bool(m.group(7))}
    return rows


def mmcheck_chunks(o: RunOut, name: str) -> str:
    """The chunks of the HMX op of one mmcheck case, from the profile lines of its run: "mc x nc" or "-". An op
    line matches by the weight type, the weight shape and the activation shape. The activation of a fused op
    (MUL_MAT_NX, MUL_MAT+ADD) has more sources after it."""
    m = re.match(r"\w+?_(q8_0|f16)_(\d+)x(\d+)_m(\d+)_", name)
    if not m:
        return "-"
    wtype, k, n, tokens = m.group(1), m.group(2), m.group(3), m.group(4)
    act = re.compile(rf" x {k}:{tokens}( x | -> )")
    got = set()
    for p in PROF_RE.finditer(o.log):
        dims, types = p.group(3), p.group(4)
        if dims.startswith(f"{k}:{n} x ") and act.search(dims) and types.startswith(f"{wtype} x "):
            ch = CHUNK_RE.search(p.group(5))
            got.add(f"{ch.group(1)}x{ch.group(2)}" if ch else p.group(5).split()[0] if p.group(5) else "?")
    return ",".join(sorted(got)) or "-"


BX_KEYS = ("bx-old", "bx-new", "bx-nc32", "bx-off")


def bitexact_table(outs: dict[str, RunOut]) -> list[str]:
    """The hashes of the mmcheck runs, the values that are not finite, the NMSE against the CPU, and the chunks of
    each run. "same" compares bx-old, bx-new and bx-nc32 (the fusion on). "off" compares bx-off with bx-new: the
    unfused ops can round differently from the fused ops, thus a difference there is information, not a failure.
    A run with an exit code other than 0 stays in the table, thus the failed cases show."""
    lines = ["", "== 1. Bit-exact: the hash of the HTP0 output of each case (measured), the chunks of each run "
             "(measured, from the profile lines), the NMSE against the CPU of the new run =="]
    present = [k for k in BX_KEYS if k in outs and outs[k].out]
    if not all(k in present for k in BX_KEYS[:3]):
        return lines + ["  bx-old, bx-new and bx-nc32 did not all give an output"]
    rows = {k: mmcheck_rows(outs[k]) for k in present}
    equal = differ = off_equal = off_differ = failed = 0
    lines.append(f"  {'case':46s} {'same':5s} {'off':4s} {'old chunks':>11s} {'new chunks':>11s} {'nc32 chunks':>11s} "
                 f"{'off chunks':>11s} {'nonfin':>6s} {'nmse':>10s} {'maxerr':>9s}")
    for name in rows["bx-new"]:
        hashes = [rows[k].get(name, {}).get("hash") for k in BX_KEYS[:3]]
        same = all(h is not None and h == hashes[0] for h in hashes)
        equal += same
        differ += not same
        off = "-"
        if "bx-off" in rows:
            off = "yes" if rows["bx-off"].get(name, {}).get("hash") == hashes[1] else "no"
            off_equal += off == "yes"
            off_differ += off == "no"
        bad = any(rows[k].get(name, {}).get("failed", True) for k in present)
        failed += bad
        nonfin = max(rows[k].get(name, {}).get("nonfinite", 0) for k in present)
        new = rows["bx-new"][name]
        chunks = " ".join(f"{mmcheck_chunks(outs[k], name) if k in present else '-':>11s}" for k in BX_KEYS)
        lines.append(f"  {name:46s} {'yes' if same else 'NO':5s} {off:4s} {chunks} {nonfin:6d} "
                     f"{new['nmse'] or '-':>10s} {new['maxerr'] or '-':>9s}" + ("  FAILED" if bad else ""))
    lines.append(f"  {equal} cases with the same hash in bx-old, bx-new and bx-nc32, {differ} with a different hash; "
                 f"{failed} cases fail in one run or more")
    if "bx-off" in rows:
        lines.append(f"  bx-off against bx-new: {off_equal} cases with the same hash, {off_differ} with a different hash")
    return lines


def test_table(outs: dict[str, RunOut]) -> list[str]:
    """The results of test-backend-ops test mode."""
    lines = ["", "== 2. test-backend-ops test mode, new model (measured) =="]
    o = outs.get("tbo-new")
    if not o or not o.ok and not o.out:
        return lines + ["  no tbo-new run"]
    res = {m.group(1): m.group(2) for m in TEST_RE.finditer(o.out)}
    for c in o.run.cases:
        lines.append(f"  {c.name:34s} {res.get(c.name, 'no result')}")
    tail = re.search(r"(\d+)/(\d+) tests passed", o.out)
    lines.append(f"  {tail.group(0) if tail else 'no summary line'}; exit code ok: {o.ok}")
    return lines


def passes(chunks: set, m: int) -> str:
    """The passes of the chunks of one case: ceil(m / mc)."""
    return ",".join(str(-(-m // mc)) for mc, _ in sorted(chunks)) or "-"


def op_table(outs: dict[str, RunOut]) -> list[str]:
    """The per-op A/B: the DSP time of the fresh and of the sustained copies and the loop time, old against new,
    the median over the rounds, and the chunks of each model."""
    lines = ["", "== 3. Each 4B op, old against new (measured: DSP us of the op copies 1-3 (fresh) and of the last "
             "30 % (sustained), host us/run of the loop; the median of the rounds; the chunks from the profile lines) =="]
    lines.append(f"  {'case':28s} {'old chunks':>11s} {'p':>2s} {'new chunks':>11s} {'p':>2s} {'fresh old':>10s} "
                 f"{'fresh new':>10s} {'d':>7s} {'sust old':>9s} {'sust new':>9s} {'d':>7s} {'loop old':>9s} "
                 f"{'loop new':>9s} {'d':>7s}")
    for c in op_cases():
        # The fresh, the sustained and the loop time of each model, and apart from them its chunks
        stats: dict[str, tuple[float | None, float | None, float | None]] = {}
        chunks: dict[str, set] = {}
        for model in ("old", "new"):
            fr, su, lp, ch = [], [], [], set()
            for rnd in (1, 2):
                o = outs.get(f"op-{rnd}-{model}")
                if not o or not o.ok:
                    continue
                st = sweep.series_of(o.series.get(c.name, []))
                if st:
                    fr.append(st.fresh)
                    su.append(st.sustained)
                if c.name in o.us:
                    lp.append(o.us[c.name])
                ch |= o.chunks.get(c.name, set())
            stats[model] = (tables.med(fr), tables.med(su), tables.med(lp))
            chunks[model] = ch
        old, new = stats["old"], stats["new"]
        d = [f"{tables.change(new[i], old[i]):>7s}" for i in range(3)]
        och, nch = chunks["old"], chunks["new"]
        lines.append(f"  {c.name:28s} {','.join(f'{a}x{b}' for a, b in sorted(och)) or '-':>11s} {passes(och, c.n):>2s} "
                     f"{','.join(f'{a}x{b}' for a, b in sorted(nch)) or '-':>11s} {passes(nch, c.n):>2s} "
                     f"{tables.fmt(old[0]):>10s} {tables.fmt(new[0]):>10s} {d[0]} "
                     f"{tables.fmt(old[1]):>9s} {tables.fmt(new[1]):>9s} {d[1]} "
                     f"{tables.fmt(old[2]):>9s} {tables.fmt(new[2]):>9s} {d[2]}")
    return lines


def sweep_table(outs: dict[str, RunOut]) -> list[str]:
    """The chunk sweep: each point with its chunks, passes and chunk count, next to the old and the new chunks of
    the op runs. The probe gives the time of one chunk where the chunks are small."""
    lines = ["", "== 4. The chunk sweep (measured: DSP fresh and sustained us, host us/run; the chunks from the profile "
             "lines; passes and chunks computed from them) =="]
    groups = defaultdict(list)
    for p in POINTS:
        groups[(p.k, p.n, p.m)].append(p)
    for (k, n, m), pts in groups.items():
        lines.append(f"  q8_0 {k}x{n} at {m} tokens")
        lines.append(f"    {'source':34s} {'chunks':>9s} {'passes':>6s} {'chunks#':>7s} {'fresh':>9s} {'sust':>9s} "
                     f"{'loop':>9s}")
        rows = []
        base = mm("q8_0", k, n, m)
        for model in ("old", "new"):
            for rnd in (1, 2):
                key = f"op-{rnd}-{model}"
                o = outs.get(key)
                if o and o.ok and base.name in o.series:
                    rows.append((mark(outs, key), o.chunks.get(base.name, set()), sweep.series_of(o.series[base.name]),
                                 o.us.get(base.name)))
        for p in pts:
            key = f"cs-{p.k}x{p.n}-m{p.m}-{p.mc}x{p.nc}"
            o = outs.get(key)
            if o and o.ok:
                rows.append((mark(outs, key), o.chunks.get(p.case.name, set()),
                             sweep.series_of(o.series.get(p.case.name, [])), o.us.get(p.case.name)))
        for src, ch, st, loop in rows:
            mc, nc = sorted(ch)[0] if ch else (0, 0)
            n_pass = -(-m // mc) if mc else 0
            n_chunks = n_pass * -(-n // nc) if nc else 0
            lines.append(f"    {src:34s} {f'{mc}x{nc}' if mc else '-':>9s} {n_pass:6d} {n_chunks:7d} "
                         f"{st.fresh if st else float('nan'):9.1f} {st.sustained if st else float('nan'):9.1f} "
                         f"{loop if loop else float('nan'):9.1f}")
        if (k, n, m) == (2560, 9216, 32):
            by_count: dict[int, list[float]] = defaultdict(list)
            for src, ch, st, _ in rows:
                if ch and st:
                    by_count[-(-n // sorted(ch)[0][1])].append(st.fresh)
            pts_t = sorted((c, statistics.median(v)) for c, v in by_count.items())
            if len(pts_t) >= 2:
                (c0, t0), (c1, t1) = pts_t[-2], pts_t[-1]
                lines.append(f"    the time of one chunk from the two points with the most chunks (computed): "
                             f"({t1:.1f} - {t0:.1f}) / ({c1} - {c0}) = {(t1 - t0) / (c1 - c0):.2f} us")
    return lines


BENCH_ROWS = (("pp512 d0", (512, 0, 0)), ("pp1024 d0", (1024, 0, 0)), ("pp512 d3072", (512, 0, 3072)),
              ("pp1024 d3072", (1024, 0, 3072)), ("tg32 d0", (0, 32, 0)))


def bench_table(outs: dict[str, RunOut]) -> list[str]:
    """The llama-bench A/B: the median t/s of the rounds, and the median over the rounds of the ratio new / old
    of one round."""
    lines = ["", "== 5. llama-bench 4B Q8_0, old against new (measured: t/s, the median of the -r 3 samples of a run; "
             "the median of the rounds; d: the median of the ratios of the runs of one round) =="]
    lines.append(f"  {'test':14s} {'old':>9s} {'new':>9s} {'d':>8s}  rounds (old / new)")
    for label, key in BENCH_ROWS:
        per = {"old": {}, "new": {}}
        for rnd in (1, 2, 3):
            for model in ("old", "new"):
                run_key = f"{'tg' if key[1] else 'pp'}-{rnd}-{model}"
                o = outs.get(run_key)
                if o and o.ok and key in o.bench:
                    per[model][rnd] = statistics.median(o.bench[key])
        if not per["old"] or not per["new"]:
            lines.append(f"  {label:14s} no result")
            continue
        ratios = [per["new"][r] / per["old"][r] for r in per["new"] if r in per["old"]]
        rounds = "  ".join(f"{per['old'].get(r, float('nan')):.1f}/{per['new'].get(r, float('nan')):.1f}"
                           for r in (1, 2, 3))
        lines.append(f"  {label:14s} {statistics.median(per['old'].values()):9.2f} "
                     f"{statistics.median(per['new'].values()):9.2f} "
                     f"{100 * (statistics.median(ratios) - 1) if ratios else float('nan'):+7.1f}%  {rounds}")
    return lines


def role(names: str) -> str:
    """The weight role of an op line: the name of its first weight without the layer number."""
    first = names.split(" x ")[0].split(" -> ")[0]
    return re.sub(r"blk\.\d+\.", "", first) if first.endswith(".weight") else ""


def split_table(outs: dict[str, RunOut]) -> list[str]:
    """The op split of the timed pp1024 graph of each profile run: the DSP us of the HMX matmuls by weight, the
    other ops, and the chunks of each weight."""
    lines = ["", "== 6. The op split of one pp1024 graph at d0 (measured: DSP us, the second graph of the run) =="]
    per: dict[str, dict] = {}
    for model in ("old", "new"):
        o = outs.get(f"prof-{model}")
        if not o or not o.ok:
            lines.append(f"  prof-{model}: no usable run")
            continue
        graphs, cur = [], None
        for m in PROF_RE.finditer(o.log):
            names = m.group(2)
            if "attn_norm-0" in names.split(" -> ")[-1] or cur is None:
                cur = defaultdict(float)
                cur["_chunks"] = {}
                graphs.append(cur)
            us = int(m.group(7)) / (float(m.group(8)) or sweep.CLOCK_MHZ)
            r = role(names)
            key = r if r and m.group(1).startswith("MUL_MAT") else "other ops"
            cur[key] += us
            ch = CHUNK_RE.search(m.group(5))
            if r and ch:
                cur["_chunks"][r] = f"{ch.group(1)}x{ch.group(2)}"
        per[model] = graphs[-1] if graphs else {}
        lines.append(f"  prof-{model}: {len(graphs)} graphs, the table uses the last")
    if len(per) < 2:
        return lines
    keys = sorted(k for k in set(per["old"]) | set(per["new"]) if not k.startswith("_"))
    lines.append(f"  {'weight':28s} {'old us':>9s} {'new us':>9s} {'d us':>9s} {'old chunks':>11s} {'new chunks':>11s}")
    for k in keys:
        a, b = per["old"].get(k, 0.0), per["new"].get(k, 0.0)
        lines.append(f"  {k:28s} {a:9.0f} {b:9.0f} {b - a:+9.0f} {per['old']['_chunks'].get(k, '-'):>11s} "
                     f"{per['new']['_chunks'].get(k, '-'):>11s}")
    ta = sum(v for k, v in per["old"].items() if not k.startswith("_"))
    tb = sum(v for k, v in per["new"].items() if not k.startswith("_"))
    lines.append(f"  {'sum of the ops':28s} {ta:9.0f} {tb:9.0f} {tb - ta:+9.0f}")
    return lines


def table(root: Path, t: Target) -> int:
    """Print the tables of the runs of a target. The correctness stage has only the sections 0 and 1."""
    if not root.is_dir():
        return cli.missing_root(root)
    outs = {r.key: read_run(root, r) for r in t.runs() if (root / f"{r.key}-gate.txt").exists()}
    parts = [conditions(outs), bitexact_table(outs)]
    if not t.suffix:
        parts += [test_table(outs), op_table(outs), sweep_table(outs), bench_table(outs), split_table(outs)]
    for part in parts:
        print("\n".join(part))
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("files", help="write phone/tests, the checksum file and the command file of a target")
    t = sub.add_parser("table", help="print the tables from the pulled outputs of a target")
    for p in (f, t):
        p.add_argument("--only", choices=sorted(k for k in TARGETS if k != "all"), default=None,
                       help="the correctness stage: only the runs bx-* (phone-commands-bx.txt, phone-out-bx)")
    t.add_argument("--root", type=Path, default=None, help="the output directory (preset: build/mmsolve/phone-out "
                                                          "or phone-out-<only>)")
    a = ap.parse_args()
    target = TARGETS[a.only or "all"]
    if a.cmd == "files":
        return write_files(target)
    return table(a.root or STAGE_DIR / target.out, target)


if __name__ == "__main__":
    sys.exit(main())
