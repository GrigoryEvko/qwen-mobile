#!/usr/bin/env python3
"""The phone stage gdnk: the gated delta net kernels of prefill and decode, old against new, on HTP0.

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/gdnk/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled logs (build/gdnk/phone-out)

The three changes of the stage patches each have an environment switch, thus one library set gives
four cumulative variants. The difference of two neighbour variants is the effect of one change:
    A  the shipped paths: GDN_CONV_CHUNK reads x with vector loads, version 1 of the chunked kernel,
       the q/k norm as its own ops (GGML_HEXAGON_GDN_CONV_DMA=0 GGML_HEXAGON_GDN_CHUNK=1
       GGML_HEXAGON_GDN_QKNORM=0)
    B  A plus the DMA path of GDN_CONV_CHUNK (GGML_HEXAGON_GDN_CONV_DMA=1)
    C  B plus version 2 of the chunked kernel (GGML_HEXAGON_GDN_CHUNK=2)
    D  C plus the q/k norm inside the ops (GGML_HEXAGON_GDN_QKNORM=1), the preset of the patches

The blocks, in the order of the stage:
    t    test-backend-ops test of GATED_DELTA_NET, GDN_CONV_STATE_FUSION and GDN_STATE_FUSION, A and D
    kp   llama-perplexity KL of the prefill path (-b 512, 4 chunks of 512) against the naive base, A and D
    kd   llama-perplexity KL of the decode path (-b 1 -ub 1, 1 chunk) against the naive base, A and D
    f    test-backend-ops perf of GATED_DELTA_NET (PERF_CASES: one token and 1024 tokens), A B C D, 2
         rounds. The perf mode repeats one node, thus it cannot time GDN_CONV_STATE_FUSION: the op
         split of block r gives GDN_CONV_CHUNK
    p    llama-bench pp512 and pp1024 at depth 0, -r 3, A B C D, 3 rounds
    g    llama-bench tg32 at the depths 0 and 4096, -r 2, A B C D, 3 rounds
    r    GGML_HEXAGON_PROFILE=1 llama-bench -p 1024 -n 8 -d 4096 -r 1, the op split, A B C D

A run name is <block>-<round>-<variant>, for example p-2-c. Each run writes three files to the phone
directory out/: <name>-gate.txt (the conditions before and after the run and the exit code),
<name>.out (the stdout of the tool) and <name>.log (its stderr).

The rate table uses a run when its gate passed, its exit code is 0, the CPU caps after the run are the
caps before it, and the thermal status after it is 0. --all also uses the runs with changed caps or
heat. A t/s value is the median of the rounds, and the difference to A is the median over the rounds
of the ratio of the two runs of one round. The table only reads files. O(size of the logs) time.

This file is tools/stages/gdnk/stage.py, and build/gdnk/stage.py is a link to it. build.sh in the same
directory builds the files of the stage into build/gdnk/phone.
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/gdnk"
MODEL = "/data/local/tmp/qwen/models/Qwen3.5-4B-Q8_0.gguf"
MODEL_KB = 8388608      # the MemAvailable (KiB) that the gate requires for the 4B
TEST_KB = 2097152       # the MemAvailable for test-backend-ops, which loads no model
EVAL = "/data/local/tmp/qwen/eval"
LAPTOP_STAGE = "build/gdnk"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/gdnk"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
# The environment of the app (init_impl in llama_jni.cpp) and the stage libraries
LIB_ENV = (f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib "
           "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1")
# The flags of variant b of the stage bench-kv: the context of the app on HTP0 with a Q8_0 KV cache
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
PPL_ARGS = (f"-dev HTP0 -ngl 99 -t 4 -fa on -ctk q8_0 -ctv q8_0 -f {EVAL}/wiki.test.raw -c 512 "
            f"--kl-divergence-base {EVAL}/naive-4B-q8.kld --kl-divergence")
# The perf cases of GATED_DELTA_NET: 32 heads of 128 at one token and at 1024 tokens with the L2_NORM of
# q and k, and the 4B shape (16 k heads, 32 v heads, 1024 tokens) with its SCALE(RMS_NORM) norm. The perf
# mode repeats the last node of the graph only, thus the norm ops of variants A to C run one time for
# each graph and the time of a run is the time of the op.
PERF_CASES = ("head_count=32,head_size=128,n_seq_tokens=(1|1024),n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1"
              "|head_count=16,head_size=128,n_seq_tokens=1024,n_seqs=1,v_repeat=2,permuted=0,kda=0,K=1,qk_form=1")
STAGE_FILES = ("bin/gate.sh", "bin/llama-bench", "bin/llama-perplexity", "bin/test-backend-ops",
               "lib/libggml-base.so", "lib/libggml-cpu.so", "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so",
               "lib/libggml-opencl.so", "lib/libggml.so", "lib/libllama-bench-impl.so", "lib/libllama-common.so",
               "lib/libllama-perplexity-impl.so", "lib/libllama.so", "lib/libmtmd.so")
TOOLS = ("llama-bench", "llama-perplexit", "test-backend-o")
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
    """One set of the three switches."""
    key: str
    label: str
    env: str


VARIANTS = {v.key: v for v in (
    Variant("a", "shipped", "GGML_HEXAGON_GDN_CONV_DMA=0 GGML_HEXAGON_GDN_CHUNK=1 GGML_HEXAGON_GDN_QKNORM=0"),
    Variant("b", "+conv DMA", "GGML_HEXAGON_GDN_CONV_DMA=1 GGML_HEXAGON_GDN_CHUNK=1 GGML_HEXAGON_GDN_QKNORM=0"),
    Variant("c", "+chunk v2", "GGML_HEXAGON_GDN_CONV_DMA=1 GGML_HEXAGON_GDN_CHUNK=2 GGML_HEXAGON_GDN_QKNORM=0"),
    Variant("d", "+qk norm", "GGML_HEXAGON_GDN_CONV_DMA=1 GGML_HEXAGON_GDN_CHUNK=2 GGML_HEXAGON_GDN_QKNORM=1"),
)}


@dataclass(frozen=True)
class Block:
    """One kind of run: the tool, its arguments, the variants, the round count, the time limit in
    seconds, the MemAvailable of the gate, and the extra environment. The gate and the lines after the
    tool take about 6 s, thus each limit is 110 s or less and a phone command stays under 120 s."""
    key: str
    tool: str
    args: str
    variants: str
    rounds: int
    limit: int
    gate_kb: int
    env: str
    text: str


BLOCKS = {b.key: b for b in (
    Block("t", "test-backend-ops", "test -b HTP0 -o GATED_DELTA_NET,GDN_CONV_STATE_FUSION,GDN_STATE_FUSION",
          "ad", 1, 110, TEST_KB, "", "test-backend-ops test of the gated delta net ops"),
    Block("kp", "llama-perplexity", f"-m {MODEL} {PPL_ARGS} --chunks 4 -b 512", "ad", 1, 100, MODEL_KB, "",
          "KL of the prefill path, 4 chunks of 512, -b 512"),
    Block("kd", "llama-perplexity", f"-m {MODEL} {PPL_ARGS} --chunks 1 -b 1 -ub 1", "ad", 1, 110, MODEL_KB, "",
          "KL of the decode path, 1 chunk of 512, -b 1 -ub 1"),
    # double quotes: the command runs inside the single quotes of adb shell
    Block("f", "test-backend-ops", "perf -b HTP0 -o GATED_DELTA_NET -p \"" + PERF_CASES + "\"", "abcd", 2, 100,
          TEST_KB, "", "test-backend-ops perf of GATED_DELTA_NET"),
    Block("p", "llama-bench", f"-m {MODEL} {BENCH_ARGS} -p 512,1024 -n 0 -d 0 -r 3", "abcd", 3, 90, MODEL_KB, "",
          "llama-bench pp512 and pp1024 at depth 0, 3 repetitions"),
    Block("g", "llama-bench", f"-m {MODEL} {BENCH_ARGS} -p 0 -n 32 -d 0,4096 -r 2", "abcd", 3, 105, MODEL_KB, "",
          "llama-bench tg32 at the depths 0 and 4096, 2 repetitions"),
    Block("r", "llama-bench", f"-m {MODEL} {BENCH_ARGS} -p 1024 -n 8 -d 4096 -r 1", "abcd", 1, 105, MODEL_KB,
          "GGML_HEXAGON_PROFILE=1", "op profile: 4 prefill ubatches to depth 4096, pp1024 and tg8 at depth 4096"),
)}
# The blocks of one group run round by round. The variants run in the order of the block in an odd
# round and in the opposite order in an even round, thus a slow drift of the clocks or the heat goes
# equally to each variant over two rounds.
GROUPS = (("t",), ("kp", "kd"), ("f",), ("p", "g"), ("r",))


@dataclass(frozen=True)
class Run:
    """One phone run of one block, round and variant."""
    block: Block
    round: int
    variant: Variant

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"{self.block.key}-{self.round}-{self.variant.key}"


def all_runs() -> list[Run]:
    """The runs in the order of the stage. O(runs)."""
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
    env = " ".join(x for x in (LIB_ENV, v.env, b.env) if x)
    cmd = (f"sh {PHONE}/bin/gate.sh {b.gate_kb} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"timeout -s KILL {b.limit} env {env} {PHONE}/bin/{b.tool} {b.args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    # The runner gates each line with "models/Qwen3.5" as a model run
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if b.gate_kb == MODEL_KB else "NO-MODEL"
    return ["#", f"# {title}: {run.name}, {b.text}, {v.key.upper()} {v.label}", THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "gdnk": the gated delta net kernels of the 4B Q8_0 on HTP0, old against new. The patches of the
# stage (build/gdnk/build.sh) change three things, each behind an environment switch:
#   GGML_HEXAGON_GDN_CONV_DMA=1  GDN_CONV_CHUNK moves its rows through VTCM with the DMA
#   GGML_HEXAGON_GDN_CHUNK=2     version 2 of the chunked gated delta net kernel
#   GGML_HEXAGON_GDN_QKNORM=1    the L2 norm of q and k runs inside GATED_DELTA_NET and GDN_STATE_STEP
# The four variants are cumulative: A shipped, B +conv DMA, C +chunk v2, D +qk norm (the preset).
#
# The runs, 42 in all (tools/stages/gdnk/stage.py):
#   t   test-backend-ops test of GATED_DELTA_NET, GDN_CONV_STATE_FUSION, GDN_STATE_FUSION, A and D
#   kp  KL of the prefill path against the naive base (4 chunks, -b 512), A and D
#   kd  KL of the decode path against the naive base (1 chunk, -b 1 -ub 1), A and D
#   f   test-backend-ops perf of GATED_DELTA_NET at one token and at 1024 tokens, A B C D, 2 rounds
#   p   llama-bench pp512 and pp1024 at depth 0, -r 3, A B C D, 3 rounds
#   g   llama-bench tg32 at the depths 0 and 4096, -r 2, A B C D, 3 rounds
#   r   GGML_HEXAGON_PROFILE=1 llama-bench -p 1024 -n 8 -d 4096 -r 1, the op split, A B C D
# The variants run A B C D in an odd round and D C B A in an even round. The flags are those of variant b of the
# stage bench-kv: HTP0, flash attention, Q8_0 K and V, -b 1024 -ub 1024, 4 threads, op fusion and state fusion on.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, the
# MemAvailable of the run), the tool under timeout -s KILL (110 s or less), the exit code and the conditions after
# the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 23 minutes of tool time plus about
# 5 minutes of gates and checks, plus the waits for thermal status 0. Then: build/gdnk/stage.py table
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    local = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l /data/local/tmp/qwen/models | grep 4B-Q8_0.gguf; ls -l {EVAL} | grep -E \"naive-4B-q8|wiki.test\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out'",
        f"{ADB} push {local} {PHONE}/bin/",
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
    for run in all_runs():
        lines += run_lines(run)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The table ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
PASSED_RE = re.compile(r"(\d+)/(\d+) tests passed")
CASE_RE = re.compile(r"^\s+([A-Z_]+\(.*\)): (?:\x1b\[[0-9;]*m)?(OK|FAIL|not supported)", re.M)
PERF_RE = re.compile(r"^\s+([A-Z_]+\(.*\)):\s+(\d+) runs -\s+([\d.]+) us/run", re.M)
KLD_RE = re.compile(r"Mean\s+KLD:\s+([\d.]+) ±\s+([\d.]+)")
TOP_RE = re.compile(r"Same top p:\s+([\d.]+) ±")
MAXKL_RE = re.compile(r"Maximum KLD:\s+([\d.]+)")
OP_RE = re.compile(r"profile-op ([A-Z0-9_+]+)\|")
USEC_RE = re.compile(r"\|usec (\d+) cycles")
OPBATCH_RE = re.compile(r"profile-op OPBATCH\|.*\|usec (\d+) cycles")
GRAPH_START = "-> attn_norm-0|"
# The op classes of the split: the gated delta net ops, the norms, and the rest
SPLIT = ("GATED_DELTA_NET", "GDN_CONV_CHUNK", "GDN_STATE_STEP", "GDN_CONV_STEP", "RMS_NORM", "SCALE", "rest")


@dataclass
class Result:
    """The parsed files of one run. ok is False when the run did not run or failed. flags names each
    condition that makes the run not comparable (changed caps, heat)."""
    run: Run
    ok: bool
    flags: list[str]
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
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        flags.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    if after and after.group(1) not in ("", "0"):
        flags.append(f"thermal {after.group(1)} after the run")
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    return Result(run, ok, flags, caps, out, log)


def usable(res: Result | None, include_all: bool) -> bool:
    """True when the run goes into the medians."""
    return res is not None and res.ok and (include_all or not res.flags)


def tests_table(results: dict[str, Result]) -> list[str]:
    """The pass counts of test-backend-ops and each failed case."""
    out = ["test-backend-ops test of the gated delta net ops (the exit code is 0 only when every case passes)"]
    for k in BLOCKS["t"].variants:
        res = results.get(f"t-1-{k}")
        if res is None:
            out.append(f"  {k.upper()}: no run")
            continue
        text = res.out + res.log
        passed = PASSED_RE.findall(text)
        cases = CASE_RE.findall(text)
        n_fail = sum(1 for _, s in cases if s == "FAIL")
        n_ok = sum(1 for _, s in cases if s == "OK")
        out.append(f"  {k.upper()} {VARIANTS[k].label}: exit ok {res.ok}, cases OK {n_ok} FAIL {n_fail}, "
                   f"summary {', '.join('/'.join(p) for p in passed) or '?'}")
        for name, s in cases:
            if s == "FAIL":
                out.append(f"    FAIL {name}")
    return out


def kl_table(results: dict[str, Result]) -> list[str]:
    """The KL divergence of each path and variant against the naive base."""
    out = ["KL against the naive oracle base (naive-4B-q8.kld): mean ± error, maximum, same top-1"]
    for block in ("kp", "kd"):
        for k in BLOCKS[block].variants:
            res = results.get(f"{block}-1-{k}")
            text = (res.out + res.log) if res else ""
            m, t, mx = KLD_RE.search(text), TOP_RE.search(text), MAXKL_RE.search(text)
            cell = (f"{m.group(1)} ± {m.group(2)}, max {mx.group(1) if mx else '?'}, top-1 {t.group(1) if t else '?'} %"
                    if m else "no KLD line")
            out.append(f"  {BLOCKS[block].text:48s} {k.upper()} {VARIANTS[k].label:12s}: {cell}"
                       + (f"  [{', '.join(res.flags)}]" if res and res.flags else ""))
    return out


def perf_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The us/run of each perf case: the median of the rounds per variant, the difference to A."""
    block = BLOCKS["f"]
    vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rnd in range(1, block.rounds + 1):
        for k in block.variants:
            res = results.get(f"f-{rnd}-{k}")
            if not usable(res, include_all):
                continue
            for name, _, us in PERF_RE.findall(res.out + res.log):
                vals[name][k].append(float(us))
    out = ["test-backend-ops perf, us per run: the median of the rounds, and the change to A"]
    for name, per in vals.items():
        base = statistics.median(per["a"]) if per.get("a") else None
        cells = []
        for k in block.variants:
            if not per.get(k):
                cells.append(f"{k.upper()} -")
                continue
            med = statistics.median(per[k])
            diff = f" {100 * (med / base - 1):+.1f}%" if base and k != "a" else ""
            cells.append(f"{k.upper()} {med:.0f}{diff}")
        out.append(f"  {name[:110]}")
        out.append("    " + " | ".join(cells))
    return out


def bench_values(res: Result) -> dict[tuple[int, int, int], float]:
    """The median t/s of each llama-bench test of one run, by (n_prompt, n_gen, n_depth)."""
    vals = {}
    for line in res.out.splitlines():
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        vals[(rec["n_prompt"], rec["n_gen"], rec["n_depth"])] = statistics.median(rec["samples_ts"])
    return vals


ROWS = (
    ("pp512 d0", "p", (512, 0, 0)),
    ("pp1024 d0", "p", (1024, 0, 0)),
    ("tg32 d0", "g", (0, 32, 0)),
    ("tg32 d4096", "g", (0, 32, 4096)),
)


def rate_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The t/s of each row and variant: the median of the rounds, the paired difference to A, the lowest
    and the highest round, and the count of rounds. O(runs)."""
    out = ["llama-bench t/s: the median of the rounds, the difference to A (the median of the ratios of the runs "
           "of one round), [the lowest and the highest round], n",
           f"  {'measurement':12s}| " + " | ".join(f"{k.upper()} {VARIANTS[k].label:28s}" for k in "abcd")]
    for text, block_key, key in ROWS:
        block = BLOCKS[block_key]
        per_round: dict[str, dict[int, float]] = {k: {} for k in block.variants}
        for rnd in range(1, block.rounds + 1):
            for k in block.variants:
                res = results.get(f"{block_key}-{rnd}-{k}")
                if usable(res, include_all):
                    v = bench_values(res).get(key)
                    if v is not None:
                        per_round[k][rnd] = v
        cells = []
        for k in "abcd":
            vals = per_round.get(k, {})
            if not vals:
                cells.append(f"{'-':30s}")
                continue
            cell = f"{statistics.median(vals.values()):.2f}"
            if k != "a":
                ratios = [vals[r] / per_round["a"][r] for r in vals if r in per_round.get("a", {})]
                cell += f" {100 * (statistics.median(ratios) - 1):+.1f}%" if ratios else " ?"
            cell += f" [{min(vals.values()):.1f}-{max(vals.values()):.1f}] n{len(vals)}"
            cells.append(f"{cell:30s}")
        out.append(f"  {text:12s}| " + " | ".join(cells))
    return out


def op_split(log: str) -> tuple[list[Counter], list[Counter]]:
    """The DSP op time (us) per class of each prefill graph and of each decode graph of a profile log. A
    graph starts at GRAPH_START. A graph with GDN_STATE_STEP or GDN_CONV_STEP, or with one token, is a
    decode graph. O(lines)."""
    graphs: list[Counter] = []
    for line in log.splitlines():
        if "profile-op " not in line or OPBATCH_RE.search(line):
            continue
        op, us = OP_RE.search(line), USEC_RE.search(line)
        if op is None or us is None:
            continue
        if not graphs or GRAPH_START in line:
            graphs.append(Counter())
        name = op.group(1)
        cls = next((c for c in SPLIT[:-1] if name == c or name.startswith(c + "+")), "rest")
        graphs[-1][cls] += int(us.group(1))
        graphs[-1]["ops"] += int(us.group(1))
    decode = [g for g in graphs if g["GDN_STATE_STEP"] or g["GDN_CONV_STEP"]]
    prefill = [g for g in graphs if g["GATED_DELTA_NET"] or g["GDN_CONV_CHUNK"]]
    return prefill, decode


def op_tables(results: dict[str, Result]) -> list[str]:
    """The op split of the profile runs: the median per graph over the prefill ubatches and over the
    decode tokens. For the full per-op table: tools/prof/optable.py ops LOG."""
    out = ["the op split of the profile runs, us of DSP op time per graph (the median over the graphs of a kind)"]
    head = "".join(f"{c:>17s}" for c in SPLIT) + f"{'op sum':>10s}{'graphs':>8s}"
    for kind in ("prefill ubatch of 1024 tokens", "decode token"):
        out.append(f"  {kind}")
        out.append(f"    {'':14s}{head}")
        for k in BLOCKS["r"].variants:
            res = results.get(f"r-1-{k}")
            if res is None or not res.ok:
                out.append(f"    {k.upper()} {VARIANTS[k].label:12s} no usable run")
                continue
            pre, dec = op_split(res.log)
            gs = pre if kind.startswith("prefill") else dec
            if not gs:
                out.append(f"    {k.upper()} {VARIANTS[k].label:12s} no graph of this kind")
                continue
            cells = "".join(f"{statistics.median(g[c] for g in gs):>17.0f}" for c in SPLIT)
            out.append(f"    {k.upper()} {VARIANTS[k].label:12s}{cells}{statistics.median(g['ops'] for g in gs):>10.0f}"
                       f"{len(gs):>8d}")
    return out


def checks(results: dict[str, Result]) -> list[str]:
    """The conditions of the runs."""
    runs = all_runs()
    got = [results[r.name] for r in runs if r.name in results]
    out = [f"{len(got)} of {len(runs)} runs have a gate file, {sum(r.ok for r in got)} ran with exit code 0, "
           f"{sum(r.ok and not r.flags for r in got)} have no flag"]
    caps = Counter(r.caps for r in got if r.ok)
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.most_common()))
    for r in got:
        if r.flags:
            out.append(f"  {r.run.name}: " + ", ".join(r.flags))
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {r.name: read_result(root, r) for r in all_runs() if (root / f"{r.name}-gate.txt").exists()}
    for part in (checks(results), tests_table(results), kl_table(results), perf_table(results, include_all),
                 rate_table(results, include_all), op_tables(results)):
        print("\n".join(part))
        print()
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the tables from the pulled logs")
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
