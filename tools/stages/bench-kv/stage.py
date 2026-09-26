#!/usr/bin/env python3
"""The phone stage bench-kv: the prefill and the decode of four KV cache forms on HTP0.

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/bench-kv/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled logs (build/bench-kv/phone-out)

The four forms. Each one has flash attention on HTP0 and the one library set of build/bench-kv/build.sh:
    A  F16 K and V
    B  Q8_0 K and V, the rotation as the DSP op FWHT where patches/hexagon-fwht/0001 sends it (the app)
    C  Q8_0 K and V, each rotation as MUL_MAT (GGML_HEXAGON_FWHT=0, patches/hexagon-fwht/0001)
    D  Q8_0 K and V without the rotation (LLAMA_ATTN_ROT_DISABLE=1)

One run matrix (BLOCKS, GROUPS) gives the command file and the parser, thus the two agree on each run
name. A run name is <model>-<block>-<round>-<variant>, for example 4b-p-2-c. Each run writes three files
to the phone directory out/: <name>-gate.txt (the conditions before and after the run and the exit code),
<name>.out (the stdout of the tool) and <name>.log (its stderr).

The table uses a run when its gate passed, its exit code is 0, the CPU caps after the run are the caps
before it, and the thermal status after it is 0. --all also uses the runs with changed caps or heat. A
t/s value is the median of the rounds. The difference to A is the median over the rounds of the ratio of
the two runs of one round, thus a slow change of the clocks between rounds does not go into it. The
table only reads files. O(size of the logs) time.

This file is tools/stages/bench-kv/stage.py, and build/bench-kv/stage.py is a link to it. The files of
the stage stay in build/bench-kv. A new stage can start from a copy of this file and of build.sh.
"""

import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cli, commands, device, gate, logs, parse, tables  # noqa: E402

ADB = device.ADB
PATHS = device.stage_paths("bench-kv", __file__)
PHONE, LAPTOP_STAGE, BOX, STAGE_DIR = PATHS
MODEL_DIR = device.MODEL_DIR
# The environment of the app (init_impl in llama_jni.cpp) and the stage libraries.
LIB_ENV = f"{device.lib_env(PHONE)} {device.APP_ENV}"
# The context of the app (load_impl in llama_jni.cpp): n_batch = n_ubatch = 1024, 4 threads, flash
# attention on HTP0. llama-bench gives n_ctx = depth + tokens and no MTP draft. memprobe gives the context
# of the app: n_ctx 8192, 5 output rows, the lazy token embedding, and with --spec n_rs_seq 4 and the
# MTP draft context. PROBE_ARGS names no cache type, because each run adds -ctk and -ctv of its variant.
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -o jsonl"
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on"
STAGE_FILES = ("bin/gate.sh", "bin/llama-bench", "bin/memprobe", "lib/libggml-base.so", "lib/libggml-cpu.so",
               "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so",
               "lib/libllama-bench-impl.so", "lib/libllama-common.so", "lib/libllama.so", "lib/libmtmd.so")
TOOLS = ("llama-bench", "memprobe")
PGREP = device.pgrep(*TOOLS)


@dataclass(frozen=True)
class Variant:
    """One KV cache form: its short label, its description, its cache type and its environment switch."""
    key: str
    label: str
    name: str
    kv: str
    env: str


@dataclass(frozen=True)
class Model:
    """One model file and the MemAvailable (KiB) that the gate requires for it."""
    key: str
    file: str
    gate_kb: int


@dataclass(frozen=True)
class Block:
    """One kind of run: the tool, its arguments, the variants, the round count and the time limit
    in seconds of the 4B and the 2B. The gate and the lines after the tool take about 6 s, thus each
    limit is 110 s or less and a phone command stays under 120 s."""
    key: str
    tool: str
    args: str
    variants: str
    rounds: int
    limit_4b: int
    limit_2b: int
    profile: bool
    text: str


VARIANTS = {v.key: v for v in (
    Variant("a", "F16", "F16 K and V", "f16", ""),
    Variant("b", "Q8_0 FWHT", "Q8_0 K and V, rotation FWHT (the app)", "q8_0", ""),
    Variant("c", "Q8_0 MUL_MAT", "Q8_0 K and V, rotation MUL_MAT", "q8_0", "GGML_HEXAGON_FWHT=0"),
    Variant("d", "Q8_0 no rot", "Q8_0 K and V, no rotation", "q8_0", "LLAMA_ATTN_ROT_DISABLE=1"),
)}
MODELS = (
    Model("4b", device.MODEL_4B, device.GATE_4B_KB),
    Model("2b", device.MODEL_2B, device.GATE_2B_KB),
)
BLOCKS = {b.key: b for b in (
    Block("p", "llama-bench", "-p 512 -n 0 -d 0,4096 -r 3", "abcd", 3, 90, 60, False,
          "llama-bench pp512 at the depths 0 and 4096, 3 repetitions"),
    Block("ms512", "memprobe", "--spec -p 512 --reps 4", "ab", 3, 70, 50, False,
          "memprobe, the app context with the MTP draft, a prompt of 512 tokens 4 times"),
    Block("mn512", "memprobe", "-p 512 --reps 4", "ab", 3, 70, 50, False,
          "memprobe, the app context without the MTP draft, a prompt of 512 tokens 4 times"),
    Block("t", "llama-bench", "-p 0 -n 32 -d 0,1024,4096 -r 2", "abcd", 3, 105, 80, False,
          "llama-bench tg32 at the depths 0, 1024 and 4096, 2 repetitions"),
    Block("ms2048", "memprobe", "--spec -p 2048 --reps 4", "ab", 2, 80, 60, False,
          "memprobe, the app context with the MTP draft, a prompt of 2048 tokens 4 times"),
    Block("mn2048", "memprobe", "-p 2048 --reps 4", "ab", 2, 80, 60, False,
          "memprobe, the app context without the MTP draft, a prompt of 2048 tokens 4 times"),
    Block("l", "llama-bench", "-p 0 -n 32 -d 16384 -r 2", "abcd", 2, 108, 90, False,
          "llama-bench tg32 at the depth 16384, 2 repetitions"),
    Block("fpp", "memprobe", "--spec -p 512 --reps 2", "ab", 1, 70, 50, True,
          "op profile, the app context with the MTP draft, a prompt of 512 tokens 2 times"),
    Block("fdec", "memprobe", "-p 4096 -n 8", "abcd", 1, 100, 70, True,
          "op profile, a prompt of 4096 tokens and 8 decode tokens"),
)}
# The blocks of one group run round by round: round 1 of each block, then round 2 of each block. The
# variants run in the order A B C D in an odd round and D C B A in an even round, thus a slow drift of
# the clocks or the heat goes equally to each variant over two rounds.
GROUPS = (("p", "ms512", "mn512", "t"), ("ms2048", "mn2048", "l"), ("fpp", "fdec"))


@dataclass(frozen=True)
class Run:
    """One phone run of one model, block, round and variant."""
    model: Model
    block: Block
    round: int
    variant: Variant

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"{self.model.key}-{self.block.key}-{self.round}-{self.variant.key}"


def runs_of(model: Model) -> list[Run]:
    """The runs of one model in the order of the stage. O(runs)."""
    out = []
    for group in GROUPS:
        for rnd in range(1, max(BLOCKS[k].rounds for k in group) + 1):
            for key in group:
                block = BLOCKS[key]
                if rnd > block.rounds:
                    continue
                order = block.variants if rnd % 2 else block.variants[::-1]
                out.extend(Run(model, block, rnd, VARIANTS[v]) for v in order)
    return out


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    m, b, v = run.model, run.block, run.variant
    env = " ".join(x for x in (LIB_ENV, v.env, "GGML_HEXAGON_PROFILE=1" if b.profile else "") if x)
    fixed = BENCH_ARGS if b.tool == "llama-bench" else PROBE_ARGS
    limit = b.limit_4b if m.key == "4b" else b.limit_2b
    tool = f"{PHONE}/bin/{b.tool} -m {MODEL_DIR}/{m.file} {fixed} -ctk {v.kv} -ctv {v.kv} {b.args}"
    cmd = commands.gated_run(f"{PHONE}/out/{run.name}", m.gate_kb, limit, env, tool, stage=PHONE)
    title = (f"# REAL-MODEL {m.file.removesuffix('.gguf')}: {run.name}, {b.text}, "
             f"{v.key.upper()}: {v.name}")
    return commands.run_lines(title, cmd, PGREP)


HEADER = """\
# Phone stage "bench-kv": the prefill and the decode of four KV cache forms of the app on HTP0, the 4B Q8_0 first,
# then the 2B Q8_0. The question: which part of the prefill drop of the app (about 920 to 780 t/s after the APK of
# 56ec1eb) comes from the Q8_0 K and V cache, and which ops cost it.
#
# The four forms, all with flash attention on HTP0:
#   A  F16 K and V
#   B  Q8_0 K and V, the rotation as the DSP op FWHT where patches/hexagon-fwht/0001 sends it (HEAD, the app)
#   C  Q8_0 K and V, each rotation as MUL_MAT (the environment switch GGML_HEXAGON_FWHT=0)
#   D  Q8_0 K and V without the rotation (LLAMA_ATTN_ROT_DISABLE=1)
#
# The libraries (build/bench-kv/build.sh): the patched llama.cpp tree of HEAD e8a3a07 (tests/sanitizers/llama-copy.sh)
# plus build/bench-kv/fwht-switch.patch (the switch GGML_HEXAGON_FWHT, preset 1), built with the preset, the flags,
# the LTO and the build number of scripts/build-native.sh. libllama, libllama-common, libmtmd, libggml, libggml-cpu and
# libggml-opencl have the bytes of the APK libraries. libggml-base has another ggml commit string, libggml-hexagon has
# the switch, and libggml-htp-v79 has hexagon-mm/0008 (the MUL_MAT_ID store, which the dense models do not call).
#
# The runs, 58 for each model. The variants run A B C D in an odd round and D C B A in an even round:
#   p       llama-bench pp512 at d0 and d4096, -r 3, A B C D, 3 rounds
#   ms512   memprobe, the app context with the MTP draft (n_rs_seq 4), 512 tokens 4 times, A B, 3 rounds
#   mn512   memprobe, the app context without the draft (the app preset), 512 tokens 4 times, A B, 3 rounds
#   t       llama-bench tg32 at d0, d1024 and d4096, -r 2, A B C D, 3 rounds
#   ms2048  as ms512 with 2048 tokens, 2 rounds
#   mn2048  as mn512 with 2048 tokens, 2 rounds
#   l       llama-bench tg32 at d16384, -r 2, A B C D, 2 rounds
#   fpp     GGML_HEXAGON_PROFILE=1, the app context with the draft, 512 tokens 2 times, A B
#   fdec    GGML_HEXAGON_PROFILE=1, 4096 prompt tokens and 8 decode tokens, A B C D
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,
# MemAvailable 8 GB for the 4B and 6 GB for the 2B, and it prints the caps), the tool under timeout -s KILL
# (less than 120 s), the exit code and the conditions after the run (thermal, caps, battery), then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 40 minutes for the 4B and 25 minutes
# for the 2B (the tools about 23 and 10 minutes, the gates and the checks about 13 s for each run), plus the waits
# for thermal status 0 and a battery of 38 C or less. Then: build/bench-kv/stage.py table
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    files = {d: [f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith(f"{d}/")]
             for d in ("bin", "lib")}
    # No "models/Qwen3.5" in the model line: the runner gates each line with that text as a model run.
    return commands.setup_lines(
        PATHS, files,
        model_check=f"{ADB} shell 'ls -l {MODEL_DIR} | grep -E \"{MODELS[0].file}|{MODELS[1].file}\"'")


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = commands.header_lines(HEADER) + setup_lines()
    for model in MODELS:
        lines += ["#", f"# ==== {model.file.removesuffix('.gguf')}: {len(runs_of(model))} runs ===="]
        for run in runs_of(model):
            lines += run_lines(run)
    lines += commands.output_lines(PATHS, tools=TOOLS)
    return commands.write_commands(path, lines)


# ---- The table ----

PREFILL_RE = re.compile(r"^TIME prefill ([\d.]+) tokens=(\d+) rate=([\d.]+) rc=(-?\d+)(?: rep=(\d+))?", re.M)
# W MUL_MAT is each MUL_MAT of a weight (also fused, for example MUL_MAT_NX or MUL_MAT+ADD). The KV form
# does not change it, thus a difference there between two variants shows a change of the clocks or the heat.
CLASSES = ("FA", "SET_ROWS", "ROT MUL_MAT", "FWHT", "CPY", "W MUL_MAT", "rest")


@dataclass
class Result:
    """The parsed files of one run. cond holds the conditions of the run: its exit code, the caps, the
    battery and the NPU zone temperature. flags names each condition of cond that makes the run not
    comparable, and each check of the output that failed."""
    run: Run
    cond: gate.Conditions
    flags: list[str]
    bench: dict[tuple[int, int, int], float]
    prefill: list[float]
    log: str


def read_result(root: Path, run: Run) -> Result:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    text, out, log = logs.read_run(root, run.name)
    c = gate.read(text)
    flags = list(c.flags)
    bench = parse.bench_values(out)
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            flags.append("a llama-bench line is not complete JSON")
            continue
        if rec["type_k"] != run.variant.kv or rec["flash_attn"] != 1:
            flags.append(f"llama-bench ran type_k {rec['type_k']} flash_attn {rec['flash_attn']}")
    prefill = [float(m.group(3)) for m in PREFILL_RE.finditer(out)]
    if run.block.tool == "memprobe" and c.ok:
        kv = re.search(r"llama_kv_cache: size = .*?K \((\w+)\)", log)
        if kv is None or kv.group(1) != run.variant.kv:
            flags.append(f"memprobe KV type {kv.group(1) if kv else '?'}")
    return Result(run, c, flags, bench, prefill, log)


def usable(res: Result, include_all: bool) -> bool:
    """True when the run goes into the medians."""
    return res.cond.ok and (include_all or not res.flags)


def value(res: Result, row: tuple) -> float | None:
    """The t/s of one run for one table row, or None."""
    kind = row[0]
    if kind == "bench":
        return res.bench.get(row[1])
    if not res.prefill:
        return None
    if kind == "first":
        return res.prefill[0]
    rest = res.prefill[1:]
    return statistics.median(rest) if rest else None


# The rows of the rate table: the text, the block, and how a run gives its value.
ROWS = (
    ("pp512 d0 (llama-bench)", "p", ("bench", (512, 0, 0))),
    ("pp512 d4096 (llama-bench)", "p", ("bench", (512, 0, 4096))),
    ("app 512, draft on, passes 2-4", "ms512", ("warm",)),
    ("app 512, draft on, pass 1", "ms512", ("first",)),
    ("app 512, draft off, passes 2-4", "mn512", ("warm",)),
    ("app 512, draft off, pass 1", "mn512", ("first",)),
    ("app 2048, draft on, passes 2-4", "ms2048", ("warm",)),
    ("app 2048, draft on, pass 1", "ms2048", ("first",)),
    ("app 2048, draft off, passes 2-4", "mn2048", ("warm",)),
    ("app 2048, draft off, pass 1", "mn2048", ("first",)),
    ("tg32 d0 (llama-bench)", "t", ("bench", (0, 32, 0))),
    ("tg32 d1024 (llama-bench)", "t", ("bench", (0, 32, 1024))),
    ("tg32 d4096 (llama-bench)", "t", ("bench", (0, 32, 4096))),
    ("tg32 d16384 (llama-bench)", "l", ("bench", (0, 32, 16384))),
)


def rate_table(model: Model, results: dict[str, Result], include_all: bool) -> list[str]:
    """The t/s table of one model: per row and variant the median of the rounds, the paired difference
    to A, the lowest and the highest round, and the count of rounds. O(runs)."""
    out = [f"{model.file}: t/s as the median of the rounds, the difference to A (the median of the ratios of "
           f"the runs of one round), [the lowest and the highest round], n and the count of rounds",
           f"  {'measurement':32s}| " + " | ".join(f"{k.upper()} {VARIANTS[k].label:34s}" for k in "abcd")]
    for text, block_key, row in ROWS:
        block = BLOCKS[block_key]
        per_round: dict[str, dict[int, float]] = {k: {} for k in block.variants}
        for rnd in range(1, block.rounds + 1):
            for k in block.variants:
                res = results.get(f"{model.key}-{block_key}-{rnd}-{k}")
                val = value(res, row) if res and usable(res, include_all) else None
                if val is not None:
                    per_round[k][rnd] = val
        cells = []
        for k in "abcd":
            vals = per_round.get(k, {})
            if not vals:
                cells.append(f"{'-':36s}")
                continue
            med = statistics.median(vals.values())
            cell = tables.rate(med)
            if k != "a":
                ratios = [vals[r] / per_round["a"][r] for r in vals if r in per_round.get("a", {})]
                cell += f" {100 * (statistics.median(ratios) - 1):+.1f}%" if ratios else " ?"
            cell += f" [{tables.rate(min(vals.values()))}-{tables.rate(max(vals.values()))}] n{len(vals)}"
            cells.append(f"{cell:36s}")
        out.append(f"  {text:32s}| " + " | ".join(cells))
    return out


@dataclass
class Graph:
    """The DSP work of one graph: the batch time and the op time per class, in us, and the op counts."""
    batch: int
    us: Counter
    count: Counter


def op_class(line: str, name: str) -> str:
    """The class of one profile-op line. A rotation names the Hadamard matrix (attn_inp_k_rot or
    attn_inp_v_rot) as src0. The DSP op FWHT prints the name MUL_MAT and no kernel ("----")."""
    fields = line.split("|")
    parts = name.split("+")
    rot = len(fields) > 1 and "_rot#" in fields[1]
    kernel = fields[5] if len(fields) > 5 else ""
    if "FLASH_ATTN_EXT" in parts:
        return "FA"
    if "SET_ROWS" in parts:
        return "SET_ROWS"
    if name == "FWHT" or (name == "MUL_MAT" and rot and kernel.startswith("----")):
        return "FWHT"
    if name == "MUL_MAT" and rot:
        return "ROT MUL_MAT"
    if "CPY" in parts:
        return "CPY"
    if any(p.startswith("MUL_MAT") for p in parts):
        return "W MUL_MAT"
    return "rest"


def graphs_of(log: str) -> list[Graph]:
    """Divide the profile of one run into graphs. A graph starts at the op parse.GRAPH_START, the norm of
    layer 0 on the input embedding: a prefill ubatch that gives no logits has no output MUL_MAT, thus the
    first op is the only marker. An OPBATCH line comes before the ops of its DSP batch, thus its time goes
    to the graph of the op after it. O(lines)."""
    graphs: list[Graph] = []
    cur: Graph | None = None
    pending = 0
    for line in log.splitlines():
        if "profile-op " not in line:
            continue
        m = parse.OPBATCH_RE.search(line)
        if m:
            pending += int(m.group(1))
            continue
        op, us = parse.OP_RE.search(line), parse.USEC_RE.search(line)
        if op is None or us is None:
            continue
        if cur is None or parse.GRAPH_START in line:
            if cur is not None:
                graphs.append(cur)
            cur = Graph(0, Counter(), Counter())
        cur.batch += pending
        pending = 0
        cls = op_class(line, op.group(1))
        cur.us[cls] += int(us.group(1))
        cur.count[cls] += 1
    if cur is not None:
        graphs.append(cur)
    return graphs


def median_graph(graphs: list[Graph]) -> Graph:
    """The median of each value over the graphs."""
    keys = set(CLASSES)
    return Graph(int(statistics.median(g.batch for g in graphs)),
                 Counter({k: statistics.median(g.us[k] for g in graphs) for k in keys}),
                 Counter({k: statistics.median(g.count[k] for g in graphs) for k in keys}))


def split_lines(title: str, graphs: dict[str, Graph]) -> list[str]:
    """The op split rows of one graph kind: one row per variant, then its difference to A."""
    head = "".join(f"{c:>12s}" for c in CLASSES) + f"{'op sum':>10s}{'batch':>10s}{'FA n':>6s}{'rot n mm/fwht':>15s}"
    out = [f"  {title}", f"    {'':14s}{head}"]

    def row(label: str, g: Graph, base: Graph | None) -> str:
        cells = ""
        for c in CLASSES:
            cells += f"{g.us[c] - (base.us[c] if base else 0):>+12.0f}" if base else f"{g.us[c]:>12.0f}"
        total = sum(g.us[c] for c in CLASSES) - (sum(base.us[c] for c in CLASSES) if base else 0)
        batch = g.batch - (base.batch if base else 0)
        fmt = "+" if base else ""
        tail = "" if base else f"{g.count['FA']:>6.0f}{g.count['ROT MUL_MAT']:>9.0f}/{g.count['FWHT']:<5.0f}"
        return f"    {label:14s}{cells}{total:>{fmt}10.0f}{batch:>{fmt}10.0f}{tail}"

    base = graphs.get("a")
    for k in "abcd":
        if k in graphs:
            out.append(row(f"{k.upper()} {VARIANTS[k].label}", graphs[k], None))
    for k in "bcd":
        if k in graphs and base is not None:
            out.append(row(f"{k.upper()} - A", graphs[k], base))
    return out


def op_tables(model: Model, results: dict[str, Result]) -> list[str]:
    """The op split of the profile runs of one model, in us of DSP time per graph."""
    out = [f"{model.file}: the op split on the DSP, us per graph. FWHT is a rotation that runs as the DSP op "
           f"FWHT. ROT MUL_MAT is a rotation that runs as a MUL_MAT."]
    pp, pre0, pre3, dec = {}, {}, {}, {}
    for k in "ab":
        res = results.get(f"{model.key}-fpp-1-{k}")
        g = graphs_of(res.log) if res and res.cond.ok else []
        if len(g) == 2:
            pp[k] = g[1]
        elif res and res.cond.ok:
            out.append(f"  {res.run.name}: {len(g)} graphs, 2 expected")
    for k in "abcd":
        res = results.get(f"{model.key}-fdec-1-{k}")
        g = graphs_of(res.log) if res and res.cond.ok else []
        if len(g) == 12:
            pre0[k], pre3[k], dec[k] = g[0], g[3], median_graph(g[4:])
        elif res and res.cond.ok:
            out.append(f"  {res.run.name}: {len(g)} graphs, 12 expected (4 prefill ubatches, 8 decode tokens)")
    out += split_lines("prefill of 512 tokens in the app context with the draft, pass 2 (fpp)", pp)
    out += split_lines("prefill ubatch of 1024 tokens at depth 0, without the draft (fdec graph 1)", pre0)
    out += split_lines("prefill ubatch of 1024 tokens at depth 3072, without the draft (fdec graph 4)", pre3)
    out += split_lines("one decode token at depth 4096, the median of 8 tokens (fdec)", dec)
    return out


def checks(model: Model, results: dict[str, Result]) -> list[str]:
    """The conditions of the runs and the checks of the forms."""
    runs = runs_of(model)
    got = [results[r.name] for r in runs if r.name in results]
    out = [f"{model.file}: {len(got)} of {len(runs)} runs have a gate file, "
           f"{sum(r.cond.ok for r in got)} ran with exit code 0, "
           f"{sum(r.cond.ok and not r.flags for r in got)} have no flag"]
    caps = Counter(r.cond.caps for r in got if r.cond.ok)
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.most_common()))
    batt = [r.cond.battery for r in got if r.cond.ok]
    if batt:
        out.append(f"  battery before the first and the last run: {batt[0]}, {batt[-1]}")
    for when in (0, 1):
        line = gate.nsp_range([r.cond for r in got], when)
        if line:
            out.append(line)
    for r in got:
        if r.flags:
            out.append(f"  {r.run.name}: " + ", ".join(r.flags))
    splits = Counter()
    for r in got:
        if r.run.block.tool == "memprobe" and r.cond.ok:
            m = re.search(r"graph splits = (\d+)", r.log)
            splits[(r.run.variant.key, m.group(1) if m else "?")] += 1
            if r.run.variant.key == "c" and "rotation stays MUL_MAT" not in r.log:
                out.append(f"  {r.run.name}: the log has no line of GGML_HEXAGON_FWHT=0")
            if r.run.variant.key == "d" and "attention rotation force disabled" not in r.log:
                out.append(f"  {r.run.name}: the log has no line of LLAMA_ATTN_ROT_DISABLE")
    out.append("  graph splits of the main context (memprobe runs): "
               + ", ".join(f"{k.upper()} {s} x{n}" for (k, s), n in sorted(splits.items())))
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables of each model."""
    if not root.is_dir():
        return cli.missing_root(root)
    for model in MODELS:
        results = {r.name: read_result(root, r) for r in runs_of(model)
                   if (root / f"{r.name}-gate.txt").exists()}
        for part in (checks(model, results), rate_table(model, results, include_all), op_tables(model, results)):
            print("\n".join(part))
            print()
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    return cli.run(STAGE_DIR, __doc__, write=write_commands,
                   count=lambda: sum(len(runs_of(m)) for m in MODELS), table=table)


if __name__ == "__main__":
    sys.exit(main())
