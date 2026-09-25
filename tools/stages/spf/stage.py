#!/usr/bin/env python3
"""The phone stages "spf", "spf2" and "spf3": the time of a short prefill call of the 4B Q8_0 on HTP0 against its token
count (spf), and the checks of the candidate fixes for it (spf2, spf3).

Usage:
    stage.py [--stage S] commands [--out PATH]         write the phone command file (build/S/phone-commands.txt)
    stage.py [--stage S] table [--root DIR] [--all]    print the tables from the pulled files (build/S/phone-out)

The stage S is spf, spf2 or spf3. Without --stage, the directory of the invoked file names the stage when it is one
(build/spf2/stage.py is a link to this file and gives spf2), else the stage is spf.

The engine of each prefill run is memprobe (tools/memprobe/memprobe.cpp) in the context of the app: n_ctx 8192,
4 threads in the thread pool of the app (low priority, no polling), Q8_0 K and V, flash attention AUTO, the
lazy token embedding, the output limit 5, op fusion and the fused state step on (the environment of the app),
and with --spec the MTP draft context with its 32768-row head, which follows each decode as in the app. The
mode --sweep restores the state of a depth before each call, and the first call of a size builds a new graph
(as each prompt call of the app); the later calls of the size reuse the graph. LLAMA_HOSTPROF=1 is on in each
run, and memprobe writes STAMP lines around each call.

The library sets: b is the tree of HEAD (the table), n is HEAD with the candidate patches of
tools/stages/spf/patches, and g (spf3 only) is HEAD with the first candidate. In spf the candidates were the switch
GGML_HEXAGON_GDN_CHUNK_MIN with the preset 2 and the row copy of CONCAT. spf2 and spf3 have the files of the patch
directory at their builds (refer to HEADER_SPF2 and HEADER_SPF3).

A run name is <block>-<lib>-<variant>, for example sw-b-s or ws-n-w0. The variant is the draft key (n off, s on,
- no model context) or a key of the switch or of the round. Each run writes <name>-gate.txt (the conditions before
and after the run and the exit code), <name>.out and <name>.log to the phone directory out/. A run goes into the
tables when its gate passed, its exit code is 0, the thermal status after it is 0 and no CPU cap before or after
it is less than 3.0 GHz (--all also uses the other runs). The table only reads files. O(size of the files).

This file is tools/stages/spf/stage.py, and build/S/stage.py is a link to it. The files of a stage stay in build/S.
The log parser reuses the patterns of tools/stages/fixed/stage.py.
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
# The stage of the run: select_stage sets PHONE, LAPTOP_STAGE, BOX, STAGE_DIR, LIBS, RUNS, HEADER and TABLE
STAGE = "spf"
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
COOL_STEP_S = 2
COOL_MAX_S = 90


def cool_line(limit: int) -> str:
    """The phone shell text that waits until the hottest NPU thermal zone is at limit millidegrees or less, at most
    COOL_MAX_S seconds, and prints the wait. The NPU clock falls as the NPU gets hotter: in the stage spf2 the NPU was
    at 26 to 43 degrees at the start of the runs, and pp1024 of HEAD fell from 1201 to 1085 t/s over the rounds."""
    steps = COOL_MAX_S // COOL_STEP_S
    return (f'{{ n=0; while [ "{fixed.NSP}" -gt {limit} ] && [ $n -lt {steps} ]; do sleep {COOL_STEP_S}; '
            f'n=$((n+1)); done; echo "cool: $((n*{COOL_STEP_S})) s to {limit}"; }}')
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
    arguments, the extra environment, the time limit in seconds, the MemAvailable gate, the text, and the variant
    key of the name (the draft key when it is empty)."""
    block: str
    lib: str
    draft: str
    tool: str
    args: str
    env: str
    limit: int
    gate_kb: int
    text: str
    var: str = ""
    cool: int = 0  # before the run, wait until the NPU is at this temperature (millidegrees) or less, 0 for no wait

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"{self.block}-{self.lib}-{self.var or self.draft}"


def _probe(block: str, lib: str, draft: str, args: str, env: str, limit: int, text: str, var: str = "",
           cool: int = 0) -> Run:
    """A memprobe run in the context of the app, with the MTP draft when draft is s."""
    spec = " --spec" if draft == "s" else ""
    return Run(block, lib, draft, "memprobe", f"{PROBE_ARGS}{spec} {args}", env, limit, GATE_KB, text, var, cool)


_SW = f"--sweep {_SIZES} --sweep-depths {_DEPTHS} --sweep-calls 3 --therm"
_PF = f"--sweep {_SIZES} --sweep-depths {_DEPTHS} --sweep-calls 2"
RUNS_SPF: list[Run] = [
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

# The stage spf2: the tg check of the GDN limit, the three candidates of tools/stages/spf/patches (the GDN limit 9,
# the row copy of CONCAT for one device, the weight stream of the HMX matmul), and their switches on one library
_BN2 = f"{BENCH_ARGS} -p 512,1024 -n 32 -d 0 -r 3"
_BN2_TEXT = "llama-bench pp512, pp1024 and tg32 at the depth 0, 3 repetitions"
_WS = "--sweep 1,8,22,64 --sweep-depths 2300 --sweep-calls 3"
_WP = "--sweep 1,8,22,64 --sweep-depths 2300 --sweep-calls 2"
_GP = "--sweep 8,9,10,12,16 --sweep-depths 2300 --sweep-calls 2"
_CP = "--sweep 22,64 --sweep-depths 2300 --sweep-calls 2"
_LG = "--sweep 512,1024 --sweep-depths 0 --sweep-calls 2"
_HS = "--hash -p 512 -n 8"
_PROF1 = "GGML_HEXAGON_PROFILE=1"
_W0 = "GGML_HEXAGON_MM_WSTREAM=0"
_W1 = "GGML_HEXAGON_MM_WSTREAM=1"
_G0 = "GGML_HEXAGON_GDN_CHUNK_MIN=0"
RUNS_SPF2: list[Run] = [
    # pp512, pp1024 and tg32 of HEAD against the candidates, 3 rounds in alternated order
    Run("bn", "b", "-", "llama-bench", _BN2, "", 100, GATE_KB, _BN2_TEXT, "1"),
    Run("bn", "n", "-", "llama-bench", _BN2, "", 100, GATE_KB, _BN2_TEXT, "1"),
    Run("bn", "n", "-", "llama-bench", _BN2, "", 100, GATE_KB, _BN2_TEXT, "2"),
    Run("bn", "b", "-", "llama-bench", _BN2, "", 100, GATE_KB, _BN2_TEXT, "2"),
    Run("bn", "b", "-", "llama-bench", _BN2, "", 100, GATE_KB, _BN2_TEXT, "3"),
    Run("bn", "n", "-", "llama-bench", _BN2, "", 100, GATE_KB, _BN2_TEXT, "3"),
    # The weight stream of the HMX matmul: the wall time of each stream on the same library, and HEAD
    _probe("ws", "n", "n", _WS, "", 90, "the sweep 1,8,22,64 at the depth 2300, 3 calls each, draft off, the stream 2 "
           "(the preset)", "w2"),
    _probe("ws", "n", "n", _WS, _W0, 90, "the same, draft off, the stream 0 (GGML_HEXAGON_MM_WSTREAM=0)", "w0"),
    _probe("ws", "n", "n", _WS, _W1, 90, "the same, draft off, the stream 1 (GGML_HEXAGON_MM_WSTREAM=1)", "w1"),
    _probe("ws", "b", "n", _WS, "", 90, "the same, draft off"),
    # The op profile of the streams 2 and 0, and the DMA counters of the stream 2
    _probe("wp", "n", "n", _WP, _PROF1, 90, "op profile of the sweep 1,8,22,64 at the depth 2300, draft off, the "
           "stream 2", "w2"),
    _probe("wp", "n", "n", _WP, f"{_PROF1} {_W0}", 90, "the same, draft off, the stream 0", "w0"),
    _probe("wm", "n", "n", "--sweep 8,22 --sweep-depths 2300 --sweep-calls 2", DMA_WAIT_ENV, 90,
           "the DMA counters of each op (PMU set dma-wait), 8 and 22 tokens at the depth 2300, draft off, the stream 2",
           "w2"),
    # The crossover of the two GDN kernels: the limit 9 against the limit 32 on the same library
    _probe("gp", "n", "n", _GP, _PROF1, 90, "op profile of 8, 9, 10, 12 and 16 tokens at the depth 2300, draft off, "
           "the GDN limit 9 (the preset)", "g9"),
    _probe("gp", "n", "n", _GP, f"{_PROF1} {_G0}", 90, "the same, draft off, the GDN limit 32 "
           "(GGML_HEXAGON_GDN_CHUNK_MIN=0)", "g0"),
    # The MTP concat with the draft on, in short calls and in the prefill of 512 and 1024 tokens
    _probe("cp", "b", "s", _CP, _PROF1, 90, "op profile of 22 and 64 tokens at the depth 2300, draft on"),
    _probe("cp", "n", "s", _CP, _PROF1, 90, "the same, draft on"),
    _probe("lg", "b", "s", _LG, "", 90, "512 and 1024 tokens at the depth 0, 2 calls each, draft on"),
    _probe("lg", "n", "s", _LG, "", 90, "the same, draft on"),
    # The bits: the logits of a prompt of 512 tokens (the HMX matmuls) and of 8 decode tokens
    _probe("hs", "b", "n", _HS, "", 90, "the logits hashes of a prompt of 512 tokens and 8 decode tokens, draft off"),
    _probe("hs", "n", "n", _HS, "", 90, "the same, draft off, the stream 2", "w2"),
    _probe("hs", "n", "n", _HS, _W0, 90, "the same, draft off, the stream 0", "w0"),
    # The op tests against the CPU backend: the Q8_0 matmuls (the stream 2 on the HMX path from 5 rows), and CONCAT
    # and GATED_DELTA_NET with the GDN limit 9
    Run("tb", "b", "-", "test-backend-ops", "-o MUL_MAT -p type_a=q8_0 -b HTP0", "", 100, GATE_KB_OPS,
        "test-backend-ops of the Q8_0 MUL_MAT cases on HTP0", "m"),
    Run("tb", "n", "-", "test-backend-ops-new", "-o MUL_MAT -p type_a=q8_0 -b HTP0", "", 100, GATE_KB_OPS,
        "test-backend-ops of the Q8_0 MUL_MAT cases on HTP0", "m"),
    Run("tb", "n", "-", "test-backend-ops-new", "-o CONCAT,GATED_DELTA_NET -b HTP0", "", 100, GATE_KB_OPS,
        "test-backend-ops of CONCAT and GATED_DELTA_NET on HTP0", "g"),
]

# The stage spf3: the check of the candidates after the limit of the weight stream (0003 applies the stream 2 to 32
# rows or fewer, and its packed task reads aligned vectors), on three library sets: HEAD, HEAD plus 0001, and HEAD plus
# the four candidates. f is n with the stream 2 on each matmul of 1024 rows or fewer.
_BN3 = f"{BENCH_ARGS} -p 512,1024 -n 32 -d 0 -r 5"
_BN3_TEXT = "llama-bench pp512, pp1024 and tg32 at the depth 0, 5 repetitions"
_ROWS_ALL = "GGML_HEXAGON_MM_WSTREAM_ROWS=1024"
_PT3 = "--sweep 8,22,32,48,64 --sweep-depths 2300 --sweep-calls 3"
_PP3 = "--sweep 8,22,32,48,64 --sweep-depths 2300 --sweep-calls 2"
COOL_MDEG = 40000
# Each round runs each variant one time, in a different order in each round
ROUNDS_SPF3 = ("b g n f", "g f b n", "n b f g")
VARIANT_TEXT = {"b": "", "g": "", "n": "", "f": ", the stream 2 on each matmul (GGML_HEXAGON_MM_WSTREAM_ROWS=1024)"}


def _bench3(r: int, v: str) -> Run:
    """The llama-bench run of variant v in round r."""
    lib = "n" if v == "f" else v
    return Run("bn", lib, "-", "llama-bench", _BN3, _ROWS_ALL if v == "f" else "", 100, GATE_KB,
               _BN3_TEXT + VARIANT_TEXT[v], f"f{r}" if v == "f" else str(r), COOL_MDEG)


RUNS_SPF3: list[Run] = [_bench3(r, v) for r, seq in enumerate(ROUNDS_SPF3, 1) for v in seq.split()] + [
    # The short calls: the wall time on each library set, and the stream 2 against the stream 0 on the same library
    _probe("ws", "b", "n", _PT3, "", 90, "the sweep 8,22,32,48,64 at the depth 2300, 3 calls each, draft off",
           cool=COOL_MDEG),
    _probe("ws", "n", "n", _PT3, "", 90, "the same, draft off, the preset limit of 32 rows", cool=COOL_MDEG),
    _probe("ws", "g", "n", _PT3, "", 90, "the same, draft off", cool=COOL_MDEG),
    _probe("ws", "n", "n", _PT3, _ROWS_ALL, 90, "the same, draft off, the stream 2 on each matmul", "f", COOL_MDEG),
    _probe("ws", "n", "n", _PT3, "GGML_HEXAGON_MM_WSTREAM=0", 90, "the same, draft off, the stream 0 "
           "(GGML_HEXAGON_MM_WSTREAM=0)", "w0", COOL_MDEG),
    _probe("wp", "n", "n", _PP3, f"{_PROF1} {_ROWS_ALL}", 90, "op profile of the sweep 8,22,32,48,64 at the depth "
           "2300, 2 calls each, draft off, the stream 2 on each matmul", "f", COOL_MDEG),
    _probe("wp", "n", "n", _PP3, f"{_PROF1} GGML_HEXAGON_MM_WSTREAM=0", 90, "the same, draft off, the stream 0", "w0",
           COOL_MDEG),
    # The bits of the packed task at 512 rows: the logits hashes with the stream 2 on each matmul against HEAD
    _probe("hs", "b", "n", _HS, "", 90, "the logits hashes of a prompt of 512 tokens and 8 decode tokens, draft off"),
    _probe("hs", "n", "n", _HS, _ROWS_ALL, 90, "the same, draft off, the stream 2 on each matmul", "f"),
    # The Q8_0 matmuls against the CPU backend with the stream 2 on each matmul
    Run("tb", "n", "-", "test-backend-ops-new", "-o MUL_MAT -p type_a=q8_0 -b HTP0",
        "GGML_HEXAGON_MM_WSTREAM_ROWS=100000", 100, GATE_KB_OPS,
        "test-backend-ops of the Q8_0 MUL_MAT cases on HTP0, the stream 2 on each matmul", "m"),
]


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    lib = LIBS[run.lib]
    stem = f"{PHONE}/out/{run.name}"
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={lib.ld} ADSP_LIBRARY_PATH={lib.adsp}", APP_ENV, run.env) if x)
    cool = f"{cool_line(run.cool)} >> {stem}-gate.txt && " if run.cool else ""
    cmd = (f"sh {PHONE}/bin/gate.sh {run.gate_kb} > {stem}-gate.txt && {cool}{BEFORE} >> {stem}-gate.txt && "
           f"timeout -s KILL {run.limit} env {env} {PHONE}/bin/{run.tool} {run.args} > {stem}.out 2> {stem}.log; "
           f"echo \"rc=$?\" >> {stem}-gate.txt; {fixed.AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt")
    title = f"{run.name}, {run.text}" + (f", {DRAFT_TEXT[run.draft]}" if run.draft != "-" and "draft" not in run.text
                                         else "") + f", {lib.text}"
    head = f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {title}" if run.tool != "test-backend-ops" and \
        not run.tool.startswith("test-backend-ops") else f"# {title}"
    return ["#", head, THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER_SPF = """\
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

HEADER_SPF2 = """\
# Phone stage "spf2": the checks of four candidate patches for the short prefill call of the 4B Q8_0 on HTP0, in the
# engine of the app, and pp512, pp1024 and tg32 in alternated rounds.
#
# The candidates (tools/stages/spf/patches, all in the libraries n):
#   0001  the chunked gated delta net from 9 tokens (GGML_HEXAGON_GDN_CHUNK_MIN, preset 9, and 0 gives the limit 32)
#   0002  the row copy of CONCAT, with the condition of one device corrected (the DSP gives one device the count 0,
#         thus the row copy did not run in the stage spf)
#   0003  the weight stream of the HMX matmul of Q8_0 weights (GGML_HEXAGON_MM_WSTREAM): 0 = one ring and rows of one
#         tile (HEAD), 1 = one ring and one row for each chunk, 2 = each dequantization worker moves its own tiles on
#         its own ring (the preset). The output has the same bits in each stream.
#   0004  an index of the readers of each root tensor for the checks of the fused chains: the host part of a call
#         with a new graph (the pack and the scheduler allocation). The fused ops are the same.
#
# The questions:
#   1. tg32, pp512 and pp1024 of the libraries n against HEAD, 3 rounds in the order b n / n b / b n.
#   2. The time of the weight matmuls at 1, 8, 22 and 64 tokens (depth 2300, draft off) with each weight stream, the
#      wall time of the calls, the host part of the first call of each size (a new graph, 0004), and the DMA
#      counters of the stream 2.
#   3. The GDN time at 8, 9, 10, 12 and 16 tokens with the limit 9 against the limit 32 on the same library.
#   4. The MTP concat with the draft on: its op time at 22 and 64 tokens, and the prefill of 512 and 1024 tokens.
#   5. The bits: the logits hashes of a prompt of 512 tokens and 8 decode tokens on HEAD and with the streams 2 and 0,
#      and the op tests of the Q8_0 MUL_MAT cases, of CONCAT and of GATED_DELTA_NET on HTP0.
#
# The files (tools/stages/spf/build.sh with STAGE=spf2): lib-base is the tree of HEAD, lib-new holds the libraries of
# HEAD plus tools/stages/spf/patches that differ, bin holds memprobe, llama-bench, llama-perplexity, test-backend-ops
# (HEAD) and test-backend-ops-new (the bound of the candidates).
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
# about 40 MB (the profile logs). Then on the box: python3 build/spf2/stage.py table
"""

HEADER_SPF3 = """\
# Phone stage "spf3": the check of the candidates for the short prefill call of the 4B Q8_0 on HTP0 after the limit of
# the weight stream: pp512, pp1024 and tg32 in 3 rounds of 4 variants, the short calls of 8 to 64 tokens, and the bits.
#
# In the stage spf2 the four candidates together gave pp512 -3.4 % [-5.6, -2.1] and tg32 -0.5 % [-1.3, -0.1]. The
# stream 2 of 0003 caused the pp512 loss: its packed dequantization task used 144 cycles for each tile against 72 for
# the aligned task (the kernel lab), and with many rows the dequantization sets the time of the op.
#
# The candidates (tools/stages/spf/patches):
#   0001  the chunked gated delta net from 9 tokens (GGML_HEXAGON_GDN_CHUNK_MIN, preset 9)
#   0002  the row copy of CONCAT for one device
#   0003  the weight stream 2 of the HMX matmul of Q8_0 weights, only for a matmul of 32 rows or fewer
#         (GGML_HEXAGON_MM_WSTREAM_ROWS, preset 32). Its packed task reads aligned vectors: 74 cycles for each tile.
#   0004  an index of the readers of each root tensor for the checks of the fused chains (the host part of a new graph)
#
# The variants: b = HEAD, g = HEAD plus 0001, n = HEAD plus the four candidates, f = n with the stream 2 on each matmul
# of 1024 rows or fewer (GGML_HEXAGON_MM_WSTREAM_ROWS=1024).
#
# The questions:
#   1. pp512, pp1024 and tg32 of g, n and f against b: 3 rounds, each variant one time in each round, in the orders
#      b g n f / g f b n / n b f g, 5 repetitions in each run. Before each model run the phone waits until the NPU is
#      at 40 degrees or less (90 s or less): the NPU clock falls as the NPU gets hotter. 0001 lands when g shows no pp
#      and no tg loss outside the noise, and the other candidates land when n shows none.
#   2. The short calls of 8, 22, 32, 48 and 64 tokens at the depth 2300, draft off: the wall time on b, g and n, and the
#      stream 2 against the stream 0 on n (the wall time and the W MM class of the op profile). These points give the
#      limit of the stream 2.
#   3. The bits: the logits hashes of a prompt of 512 tokens and 8 decode tokens with the stream 2 on each matmul against
#      HEAD, and the Q8_0 MUL_MAT cases of test-backend-ops with the stream 2 on each matmul.
#
# The files (VARIANTS="g:0001-* new:*" STAGE=spf3 tools/stages/spf/build.sh): lib-base is the tree of HEAD, lib-g and
# lib-new hold the libraries of HEAD plus 0001 and plus the four candidates that differ, bin holds memprobe,
# llama-bench, llama-perplexity, test-backend-ops (HEAD), test-backend-ops-g and test-backend-ops-new.
#
# The runs, {n_runs}:
{run_list}
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,
# MemAvailable 8 GB for a model run and 2 GB for test-backend-ops, and it prints the caps), for a model run the wait
# for the NPU temperature, the tool under timeout -s KILL (110 s or less), the exit code and the conditions after the
# run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, screen on, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 13 minutes of tool time plus about
# 8 s of gate and checks for each run, the waits for the NPU temperature (90 s or less for each of the 21 model runs)
# and the waits for thermal status 0. The push is about 200 MB, the pull about 20 MB. Then on the box:
# python3 build/spf3/stage.py table
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
    ]
    dirs = sorted({f.split("/")[0] for f in files}, key=lambda d: (d != "bin", d != "lib-base", d))
    lines.append(f"{ADB} shell 'rm -rf {PHONE} && mkdir -p " + " ".join(f"{PHONE}/{d}" for d in dirs) + f" {PHONE}/out'")
    for d in dirs:
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


def dma_table(root: Path, include_all: bool, name: str = "pm-b-n") -> list[str]:
    """The weight matmuls of a run with the PMU set dma-wait at each size: for each weight kind, the time, the
    weight rate and the shares of the op cycles with the DMA active and with the DMA read buffer full, the median
    over the layers and the calls, and the slowest layer."""
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
    out = [f"{name}: the weight matmuls, draft off, the PMU set dma-wait. us = the median op time, "
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


def table_spf(root: Path, include_all: bool) -> int:
    """Print the tables of the stage spf."""
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


# ---- The tables of the stage spf2 ----

def _median_ms(calls: list, n: int, later: bool) -> float | None:
    """The median wall ms of the calls of n tokens: the later calls (the graph reused) or the first call."""
    return med(c.ms for c in calls if c.tokens == n and ((c.call > 0) if later else (c.call == 0)))


def bench_rounds_table(root: Path, include_all: bool) -> list[str]:
    """pp512, pp1024 and tg32 of each round of bn, and the median of the paired ratios n / b over the rounds."""
    rates: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in RUNS:
        if r.block != "bn" or not usable(root, r.name, include_all):
            continue
        for line in (root / f"{r.name}.out").read_text(errors="replace").splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                rates[(rec["n_prompt"], rec["n_gen"])][r.lib][r.var] = statistics.median(rec["samples_ts"])
    out = ["bn: llama-bench t/s of each round (the median of 3 repetitions), b = HEAD, n = the candidates, and the median "
           "change n / b over the paired rounds [range]" + ("" if include_all else " (--all also uses removed runs)")]
    for key in sorted(rates):
        per = rates[key]
        rounds = sorted(set(per.get("b", {})) | set(per.get("n", {})))
        cells = [f"r{k}: b {fmt(per['b'].get(k), 2)} n {fmt(per['n'].get(k), 2)}" for k in rounds]
        ratios = [per["n"][k] / per["b"][k] for k in rounds if k in per.get("b", {}) and k in per.get("n", {})]
        change = (f"{100 * (statistics.median(ratios) - 1):+.2f}% [{100 * (min(ratios) - 1):+.2f}, "
                  f"{100 * (max(ratios) - 1):+.2f}], {len(ratios)} rounds") if ratios else "no pair"
        label = f"pp{key[0]}" if key[1] == 0 else f"tg{key[1]}"
        out.append(f"  {label:7s} " + " | ".join(cells) + f" | change {change}")
    return out


def stream_table(root: Path, include_all: bool) -> list[str]:
    """The weight streams at 1, 8, 22 and 64 tokens (depth 2300, draft off): the wall ms of the first call and of the
    later calls (ws), and the DSP ms of the weight matmuls (wp)."""
    runs = [("ws-b-n", "HEAD"), ("ws-n-w0", "stream 0"), ("ws-n-w1", "stream 1"), ("ws-n-w2", "stream 2")]
    data = {name: read_calls(root, name)[0] if usable(root, name, include_all) else [] for name, _ in runs}
    prof = {name: read_calls(root, name)[0] if usable(root, name, include_all) else [] for name in ("wp-n-w0", "wp-n-w2")}
    out = ["ws, wp: the weight streams, depth 2300, draft off. ms of the first call / the later calls (the median), and "
           "the DSP ms of the class W MM and of the DSP batches (op profile, streams 0 and 2)",
           f"  {'n':>3} | " + " | ".join(f"{label:>13}" for _, label in runs) + " | W MM w0  W MM w2 | DSP w0  DSP w2"]
    for n in (1, 8, 22, 64):
        cells = [f"{fmt(_median_ms(data[name], n, False)):>6}/{fmt(_median_ms(data[name], n, True)):>6}" for name, _ in runs]
        wm = [fmt(med(c.classes.get("W MM", 0) / 1000 for c in prof[p] if c.tokens == n)) for p in ("wp-n-w0", "wp-n-w2")]
        dsp = [fmt(med(c.dsp_us / 1000 for c in prof[p] if c.tokens == n)) for p in ("wp-n-w0", "wp-n-w2")]
        out.append(f"  {n:>3} | " + " | ".join(cells) + f" | {wm[0]:>7} {wm[1]:>7} | {dsp[0]:>6} {dsp[1]:>6}")
    out.append("  the host part of the first call of each size (a new graph): host = wall minus the DSP wait, pack and alloc "
               "(ms), HEAD (ws-b-n) against the candidates (ws-n-w2)")
    for n in (1, 8, 22, 64):
        cells = []
        for name in ("ws-b-n", "ws-n-w2"):
            first = [c for c in data[name] if c.tokens == n and c.call == 0]
            cells.append(f"{name}: host {fmt(med(c.host_ms for c in first))} pack {fmt(med(c.pack_us / 1000 for c in first), 2)} "
                         f"alloc {fmt(med(c.alloc_us / 1000 for c in first), 2)}")
        out.append(f"  {n:>3} | " + " | ".join(cells))
    return out


def gdn_cross_table(root: Path, include_all: bool) -> list[str]:
    """The GDN class at 8, 9, 10, 12 and 16 tokens with the limit 9 (gp-n-g9) against the limit 32 (gp-n-g0)."""
    data = {name: read_calls(root, name)[0] if usable(root, name, include_all) else [] for name in ("gp-n-g9", "gp-n-g0")}
    out = ["gp: the DSP ms of the class GDN and of the DSP batches, depth 2300, draft off, the limit 9 against the limit 32",
           f"  {'n':>3} | {'GDN 9':>6} {'GDN 32':>6} | {'DSP 9':>6} {'DSP 32':>6}"]
    for n in (8, 9, 10, 12, 16):
        g = [fmt(med(c.classes.get("GDN", 0) / 1000 for c in data[p] if c.tokens == n), 2) for p in ("gp-n-g9", "gp-n-g0")]
        d = [fmt(med(c.dsp_us / 1000 for c in data[p] if c.tokens == n)) for p in ("gp-n-g9", "gp-n-g0")]
        out.append(f"  {n:>3} | {g[0]:>6} {g[1]:>6} | {d[0]:>6} {d[1]:>6}")
    return out


def concat_table(root: Path, include_all: bool) -> list[str]:
    """The MTP concat with the draft on: its op time and the DRAFT class at 22 and 64 tokens (cp), and the prefill of
    512 and 1024 tokens (lg), HEAD against the candidates."""
    out = ["cp, lg: the MTP concat, draft on. CONCAT us (the op mtp_concat) and DRAFT ms of the op profile at the depth "
           "2300, and the ms of 512 and 1024 tokens at the depth 0 (call 0 / call 1)"]
    for lib in "bn":
        name = f"cp-{lib}-s"
        calls = read_calls(root, name, keep_ops=True)[0] if usable(root, name, include_all) else []
        cells = []
        for n in (22, 64):
            cc = [c for c in calls if c.tokens == n]
            cat = med(sum(u for op, names, u, _, _ in c.ops if op == "CONCAT" and "mtp_concat" in names) for c in cc)
            cells.append(f"n{n}: CONCAT {fmt(cat, 0)} us, DRAFT {fmt(med(c.classes.get('DRAFT', 0) / 1000 for c in cc))} ms")
        lg = read_calls(root, f"lg-{lib}-s")[0] if usable(root, f"lg-{lib}-s", include_all) else []
        long = ", ".join(f"n{n} {fmt(_median_ms(lg, n, False))}/{fmt(_median_ms(lg, n, True))}" for n in (512, 1024))
        out.append(f"  {LIBS[lib].text:24s}: " + "; ".join(cells) + f" | {long}")
    return out


def hash_table(root: Path) -> list[str]:
    """The HASH lines of the runs hs: each run must give the lines of hs-b-n."""
    runs = [r.name for r in RUNS if r.block == "hs"]
    got = {name: re.findall(r"^HASH (.*) ([0-9a-f]{16})$", (root / f"{name}.out").read_text(errors="replace"), re.M)
           if (root / f"{name}.out").exists() else [] for name in runs}
    ref = got.get("hs-b-n", [])
    out = [f"hs: the logits hashes of a prompt of 512 tokens and 8 decode tokens against hs-b-n ({len(ref)} lines)"]
    for name in runs:
        same = sum(a == b for a, b in zip(got[name], ref))
        out.append(f"  {name}: {len(got[name])} lines, {same} equal to hs-b-n" +
                   (" (the same)" if got[name] and got[name] == ref else " (DIFFERENT)"))
    return out


def ops_table_spf2(root: Path) -> list[str]:
    """The result lines of the op tests of spf2."""
    out = ["tb: test-backend-ops on HTP0 against the CPU"]
    for r in RUNS:
        if r.block != "tb":
            continue
        p = root / f"{r.name}.out"
        text = p.read_text(errors="replace") if p.exists() else ""
        passed = re.findall(r"(\d+)/(\d+) tests passed", text)
        fails = [line.strip() for line in text.splitlines() if "FAIL" in line][:12]
        out.append(f"  {r.name} ({r.text}, {LIBS[r.lib].text}): " +
                   (", ".join(f"{a}/{b} passed" for a, b in passed) or "no result line"))
        out += [f"    {line[:200]}" for line in fails]
    return out


def table_spf2(root: Path, include_all: bool) -> int:
    """Print the tables of the stage spf2."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    parts = [["The runs:"] + checks(root), bench_rounds_table(root, include_all), stream_table(root, include_all),
             dma_table(root, include_all, "wm-n-w2"), gdn_cross_table(root, include_all), concat_table(root, include_all),
             hash_table(root), ops_table_spf2(root)]
    for part in parts:
        print("\n".join(part))
        print()
    return 0


# ---- The tables of the stage spf3 ----

def _nsp_before(root: Path, name: str) -> float | None:
    """The NPU temperature (degrees) at the start of a run, from the before line of its gate file."""
    p = root / f"{name}-gate.txt"
    m = fixed.BEFORE_RE.search(p.read_text(errors="replace")) if p.exists() else None
    return int(m.group(1)) / 1000 if m and m.group(1) else None


def bench_rounds_table3(root: Path, include_all: bool) -> list[str]:
    """pp512, pp1024 and tg32 of each variant in each round (the median of the repetitions), the NPU temperature at
    the start of each run, and for g, n and f the median of the paired ratios against b over the rounds [range].
    O(runs)."""
    rates: dict[tuple, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    temps: dict[str, dict[int, float | None]] = defaultdict(dict)
    for r in RUNS:
        if r.block != "bn" or not usable(root, r.name, include_all):
            continue
        variant = "f" if r.var.startswith("f") else r.lib
        rnd = int(r.var.lstrip("f"))
        temps[variant][rnd] = _nsp_before(root, r.name)
        for line in (root / f"{r.name}.out").read_text(errors="replace").splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                rates[(rec["n_prompt"], rec["n_gen"])][variant][rnd] = statistics.median(rec["samples_ts"])
    variants = ("b", "g", "n", "f")
    out = ["bn: llama-bench t/s of each variant in each round (the median of 5 repetitions): b = HEAD, g = HEAD plus 0001, "
           "n = HEAD plus the four candidates, f = n with the stream 2 on each matmul. change = the median of the "
           "paired ratios against b over the rounds [range]" + ("" if include_all else " (--all also uses removed runs)")]
    rounds = sorted({k for v in temps.values() for k in v})
    out.append("  NPU at the start (degrees): " + " | ".join(
        f"r{k}: " + " ".join(f"{v} {fmt(temps[v].get(k), 1)}" for v in variants) for k in rounds))
    for key in sorted(rates):
        per = rates[key]
        label = f"pp{key[0]}" if key[1] == 0 else f"tg{key[1]}"
        cells = [f"r{k}: " + " ".join(f"{v} {fmt(per[v].get(k), 2)}" for v in variants) for k in rounds]
        out.append(f"  {label:7s} " + " | ".join(cells))
        b_vals = list(per["b"].values())
        spread = f"{100 * (max(b_vals) / min(b_vals) - 1):.2f}%" if len(b_vals) > 1 else "-"
        changes = []
        for v in ("g", "n", "f"):
            ratios = [per[v][k] / per["b"][k] for k in rounds if k in per[v] and k in per["b"]]
            changes.append(f"{v} {100 * (statistics.median(ratios) - 1):+.2f}% [{100 * (min(ratios) - 1):+.2f}, "
                           f"{100 * (max(ratios) - 1):+.2f}]" if ratios else f"{v} no pair")
        out.append(f"  {'':7s} change: " + ", ".join(changes) + f" | the spread of b over the rounds {spread}")
    return out


def points_table3(root: Path, include_all: bool) -> list[str]:
    """The short calls at 8 to 64 tokens (depth 2300, draft off): the wall ms of the first call and of the later calls
    on each variant, the DSP ms of the class W MM with the stream 2 on each matmul against the stream 0, and the host
    part of the first call of b against n. O(lines of the logs)."""
    runs = [("ws-b-n", "HEAD"), ("ws-g-n", "0001"), ("ws-n-n", "n, limit 32"), ("ws-n-f", "n, stream 2"),
            ("ws-n-w0", "n, stream 0")]
    data = {name: read_calls(root, name)[0] if usable(root, name, include_all) else [] for name, _ in runs}
    prof = {name: read_calls(root, name)[0] if usable(root, name, include_all) else [] for name in ("wp-n-f", "wp-n-w0")}
    out = ["ws, wp: the short calls, depth 2300, draft off. ms of the first call / the later calls (the median), and the "
           "DSP ms of the class W MM and of the DSP batches (op profile) with the stream 2 on each matmul (f) and the "
           "stream 0 (w0)",
           f"  {'n':>3} | " + " | ".join(f"{label:>13}" for _, label in runs) + " | W MM f  W MM w0 | DSP f  DSP w0"]
    for n in (8, 22, 32, 48, 64):
        cells = [f"{fmt(_median_ms(data[name], n, False)):>6}/{fmt(_median_ms(data[name], n, True)):>6}" for name, _ in runs]
        wm = [fmt(med(c.classes.get("W MM", 0) / 1000 for c in prof[p] if c.tokens == n)) for p in ("wp-n-f", "wp-n-w0")]
        dsp = [fmt(med(c.dsp_us / 1000 for c in prof[p] if c.tokens == n)) for p in ("wp-n-f", "wp-n-w0")]
        out.append(f"  {n:>3} | " + " | ".join(cells) + f" | {wm[0]:>6} {wm[1]:>7} | {dsp[0]:>5} {dsp[1]:>6}")
    out.append("  the host part of the first call of each size (a new graph): host = wall minus the DSP wait, pack and alloc "
               "(ms), HEAD (ws-b-n) against the four candidates (ws-n-n)")
    for n in (8, 22, 32, 48, 64):
        cells = []
        for name in ("ws-b-n", "ws-n-n"):
            first = [c for c in data[name] if c.tokens == n and c.call == 0]
            cells.append(f"{name}: host {fmt(med(c.host_ms for c in first))} pack {fmt(med(c.pack_us / 1000 for c in first), 2)} "
                         f"alloc {fmt(med(c.alloc_us / 1000 for c in first), 2)}")
        out.append(f"  {n:>3} | " + " | ".join(cells))
    return out


def table_spf3(root: Path, include_all: bool) -> int:
    """Print the tables of the stage spf3."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    parts = [["The runs:"] + checks(root), bench_rounds_table3(root, include_all), points_table3(root, include_all),
             hash_table(root), ops_table_spf2(root)]
    for part in parts:
        print("\n".join(part))
        print()
    return 0


@dataclass(frozen=True)
class Stage:
    """One phone stage of this file: its name, its runs, the header of its command file and its tables."""
    name: str
    runs: list
    header: str
    table: object


STAGES = {
    "spf": Stage("spf", RUNS_SPF, HEADER_SPF, table_spf),
    "spf2": Stage("spf2", RUNS_SPF2, HEADER_SPF2, table_spf2),
    "spf3": Stage("spf3", RUNS_SPF3, HEADER_SPF3, table_spf3),
}


def select_stage(name: str) -> None:
    """Set the values of the stage name: the phone directory, the laptop and box directories, the library sets, the
    runs, the header and the tables."""
    global STAGE, PHONE, LAPTOP_STAGE, BOX, STAGE_DIR, LIBS, RUNS, HEADER, TABLE
    st = STAGES[name]
    STAGE        = name
    PHONE        = f"/data/local/tmp/qwen/{name}"
    LAPTOP_STAGE = f"build/{name}"
    BOX          = f"grigory@10.10.20.200:airi/qwen-mobile/build/{name}"
    STAGE_DIR    = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
    LIBS         = {
        "b": Lib("b", f"{PHONE}/lib-base", f"{PHONE}/lib-base", "HEAD"),
        "n": Lib("n", f"{PHONE}/lib-new:{PHONE}/lib-base", f"{PHONE}/lib-new", "HEAD plus the candidates"),
        "g": Lib("g", f"{PHONE}/lib-g:{PHONE}/lib-base", f"{PHONE}/lib-g", "HEAD plus 0001"),
    }
    RUNS   = st.runs
    HEADER = st.header
    TABLE  = st.table


select_stage(STAGE)


def main() -> int:
    """Run the subcommand of the command line. The stage is --stage, or the name of the directory of the invoked file
    when that is a stage (build/spf2/stage.py gives spf2), or spf."""
    invoked = Path(sys.argv[0]).parent.name
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=sorted(STAGES), default=invoked if invoked in STAGES else "spf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=None)
    t = sub.add_parser("table", help="print the tables from the pulled files")
    t.add_argument("--root", type=Path, default=None)
    t.add_argument("--all", action="store_true", help="also use the runs with changed caps or heat")
    a = ap.parse_args()
    select_stage(a.stage)
    if a.cmd == "commands":
        out = a.out or STAGE_DIR / "phone-commands.txt"
        n = write_commands(out)
        print(f"{out}: {n} lines, {len(RUNS)} runs")
        return 0
    return TABLE(a.root or STAGE_DIR / "phone-out", a.all)


if __name__ == "__main__":
    sys.exit(main())
