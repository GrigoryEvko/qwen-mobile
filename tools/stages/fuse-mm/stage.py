#!/usr/bin/env python3
"""The phone stage fuse-mm: the correctness and the speed of the matmul fusions of HTP0 on the 4B Q8_0.

Usage:
    stage.py commands [--set NAME] [--out PATH]   write the phone command file of a run set
    stage.py table [--root DIR] [--all]           print the tables from the pulled logs

The run sets (SETS):
    full   the first stage: the op tests, ffncheck, the KL runs, the timing (p, t) and the op profile.
           The command file is build/fuse-mm/phone-commands.txt, the outputs go to build/fuse-mm/phone-out.
    check  the correctness only: test-backend-ops FFN_SWIGLU, ffncheck, ffncheck --cpu and the KL runs.
           The command file is build/fuse-mm/phone-commands-check.txt, the outputs go to phone-out-check.
    prof   the op profile only (GGML_HEXAGON_PROFILE=1 and -v), pp512, pp1024 and tg4, on and off.
           The command file is build/fuse-mm/phone-commands-prof.txt, the outputs go to phone-out-prof.

The fusions (htp-mm-fusion.h of the llama.cpp patches in build/fuse-mm/patches) have environment switches,
thus one library set of tools/stages/fuse-mm/build.sh gives each variant:
    on     the preset values: MUL_MAT_NX_SWIGLU on the HMX and the HVX path, the F16 SwiGLU output
    sw     MUL_MAT_NX_SWIGLU only (GGML_HEXAGON_FUSE_F16_ACT=0)
    f16    the F16 SwiGLU output only, from the unfused GLU op (GGML_HEXAGON_FUSE_SWIGLU=0 and _DECODE=0)
    off    the three switches 0: the ops of HEAD

One run matrix (BLOCKS) gives the command file and the parser, thus the two agree on each run name. A run
name is <block>-<round>-<variant>, for example p-2-on. Each run writes three files to the phone directory
out/: <name>-gate.txt (the conditions before and after the run and the exit code), <name>.out (the stdout of
the tool) and <name>.log (its stderr).

The table uses a timing run when its gate passed, its exit code is 0, the CPU caps after the run are the caps
before it, and the thermal status after it is 0. --all also uses the runs with changed caps or heat. A t/s
value is the median of the rounds. The difference to off is the median over the rounds of the ratio of the two
runs of one round. The table only reads files. O(size of the logs) time.
"""

import argparse
import re
import statistics
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cli, commands, device, gate, logs, parse  # noqa: E402

ADB = device.ADB
PATHS = device.stage_paths("fuse-mm", __file__)
PHONE, LAPTOP_STAGE, BOX, STAGE_DIR = PATHS
MODEL = f"{device.MODEL_DIR}/{device.MODEL_4B}"
EVAL = device.EVAL_DIR
GATE_KB = device.GATE_4B_KB
# The environment of the app (init_impl in llama_jni.cpp) and the stage libraries
LIB_ENV = f"{device.lib_env(PHONE)} {device.APP_ENV}"
# The context of the app (load_impl in llama_jni.cpp): n_batch = n_ubatch = 1024, 4 threads, flash attention
# on HTP0, Q8_0 K and V (the variant b of the stage bench-kv)
BENCH_ARGS = f"-m {MODEL} -dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
KL_ARGS = (f"-m {MODEL} -dev HTP0 -ngl 99 -t 4 -fa on -ctk q8_0 -ctv q8_0 -f {EVAL}/wiki.test.raw -c 512 "
           f"--chunks 1 --kl-divergence-base {EVAL}/naive-4B-q8.kld --kl-divergence")
# The kernel keeps 15 characters of a process name, and pgrep -x matches that name
TOOLS = ("llama-bench", "llama-perplexit", "test-backend-op", "ffncheck")
PGREP = device.pgrep(*TOOLS)

VARIANTS = {
    "on": "",
    "sw": "GGML_HEXAGON_FUSE_F16_ACT=0",
    "f16": "GGML_HEXAGON_FUSE_SWIGLU=0 GGML_HEXAGON_FUSE_SWIGLU_DECODE=0",
    "off": "GGML_HEXAGON_FUSE_SWIGLU=0 GGML_HEXAGON_FUSE_SWIGLU_DECODE=0 GGML_HEXAGON_FUSE_F16_ACT=0",
}


@dataclass(frozen=True)
class Block:
    """One kind of run: the tool, its arguments, the variants in the order of an odd round, the round count,
    the time limit in seconds (the gate and the lines after the tool take about 6 s, thus a limit of 110 s
    keeps a phone command under 120 s), the timing flag and the text. A tail block runs after the timing
    blocks (the op profiles)."""
    key: str
    tool: str
    args: str
    variants: tuple
    rounds: int
    limit: int
    timing: bool
    text: str
    extra_env: str = ""
    tail: bool = False


BLOCKS = (
    Block("tbo", "test-backend-ops", "test -b HTP0 -o FFN_SWIGLU", ("on", "f16", "off"), 1, 110, False,
          "test-backend-ops FFN_SWIGLU on HTP0 against the CPU"),
    Block("reg", "test-backend-ops", "test -b HTP0 -o MUL_MAT_VEC_FUSION,SWIGLU,GEGLU", ("on",), 1, 110, False,
          "test-backend-ops MUL_MAT_VEC_FUSION, SWIGLU and GEGLU on HTP0 (the NX and GLU code of the patches)"),
    Block("mm", "test-backend-ops", "test -b HTP0 -o MUL_MAT", ("on",), 1, 110, False,
          "test-backend-ops MUL_MAT on HTP0 (the HMX activation load)"),
    Block("ffn", "ffncheck", "", ("on", "sw", "f16", "off"), 1, 100, False,
          "ffncheck: the hash of the FFN block of the 4B for each token count"),
    Block("ffncpu", "ffncheck", "--cpu", ("on",), 1, 110, False,
          "ffncheck --cpu: the FFN block of the 4B against the CPU backend"),
    Block("klp", "llama-perplexity", f"{KL_ARGS} -b 512", ("on", "off"), 1, 90, False,
          "llama-perplexity KL against the naive base, 1 chunk of 512, -b 512 (the HMX prefill path)"),
    Block("kld", "llama-perplexity", f"{KL_ARGS} -b 1 -ub 1", ("on", "off"), 1, 108, False,
          "llama-perplexity KL against the naive base, 1 chunk of 512, -b 1 (the HVX decode path)"),
    Block("p", "llama-bench", f"{BENCH_ARGS} -p 512,1024 -n 0 -d 0,3072 -r 3", ("on", "off", "sw"), 3, 100, True,
          "llama-bench pp512 and pp1024 at the depths 0 and 3072, 3 repetitions"),
    Block("t", "llama-bench", f"{BENCH_ARGS} -p 0 -n 32 -d 0,4096 -r 3", ("on", "off"), 3, 100, True,
          "llama-bench tg32 at the depths 0 and 4096, 3 repetitions"),
    # -v: the backend writes the profile-op lines of tools/prof/optable.py only in the verbose mode.
    # LLAMA_HOSTPROF=1: the host times of each token, with the graph cache hits and the batch replays.
    Block("prof", "llama-bench", f"{BENCH_ARGS} -p 512,1024 -n 4 -r 1 -v", ("on", "off"), 1, 90, False,
          "op profile (GGML_HEXAGON_PROFILE=1, LLAMA_HOSTPROF=1, -v): ubatches of 512 and 1024 tokens and 4 decode "
          "tokens", "GGML_HEXAGON_PROFILE=1 LLAMA_HOSTPROF=1", True),
    # The run set f16: the libraries of HEAD plus the patch of the F16 SwiGLU output (no fused op)
    Block("ftbo", "test-backend-ops", "test -b HTP0 -o FFN_SWIGLU", ("f16",), 1, 110, False,
          "test-backend-ops FFN_SWIGLU on HTP0 against the CPU"),
    Block("fffn", "ffncheck", "", ("f16", "off"), 1, 100, False,
          "ffncheck: the hash of the FFN block of the 4B for each token count"),
    Block("fkld", "llama-perplexity", f"{KL_ARGS} -b 1 -ub 1", ("off", "f16"), 2, 108, False,
          "llama-perplexity KL against the naive base, 1 chunk of 512, -b 1 (the decode path), 2 runs of each "
          "variant for the determinism"),
    Block("fp", "llama-bench", f"{BENCH_ARGS} -p 512,1024 -n 0 -d 0,3072 -r 3", ("f16", "off"), 3, 100, True,
          "llama-bench pp512 and pp1024 at the depths 0 and 3072, 3 repetitions"),
    Block("ft", "llama-bench", f"{BENCH_ARGS} -p 0 -n 32 -d 0,4096 -r 3", ("f16", "off"), 3, 100, True,
          "llama-bench tg32 at the depths 0 and 4096, 3 repetitions"),
    Block("fprof", "llama-bench", f"{BENCH_ARGS} -p 512,1024 -n 4 -r 1 -v", ("f16", "off"), 1, 90, False,
          "op profile (GGML_HEXAGON_PROFILE=1, LLAMA_HOSTPROF=1, -v): ubatches of 512 and 1024 tokens and 4 decode "
          "tokens", "GGML_HEXAGON_PROFILE=1 LLAMA_HOSTPROF=1", True),
    # The run set repack: the libraries of HEAD plus the patch that refuses a weight in the plain layout
    Block("plain", "ffncheck", "--plain --only ffn_", ("on",), 1, 100, False,
          "ffncheck --plain: the weights in a buffer with no WEIGHTS usage, the plain layout"),
    Block("rffn", "ffncheck", "", ("on",), 1, 100, False,
          "ffncheck: the weights in a buffer of the WEIGHTS usage, the tiled layout"),
    Block("rmm", "test-backend-ops", "test -b HTP0 -o MUL_MAT", ("on",), 1, 110, False,
          "test-backend-ops MUL_MAT on HTP0 (each tiled weight type)"),
    Block("rmmid", "test-backend-ops", "test -b HTP0 -o MUL_MAT_ID", ("on",), 1, 110, False,
          "test-backend-ops MUL_MAT_ID on HTP0 (each tiled weight type)"),
    Block("rklp", "llama-perplexity", f"{KL_ARGS} -b 512", ("on",), 1, 90, False,
          "llama-perplexity KL against the naive base, 1 chunk of 512, -b 512"),
)
BLOCK = {b.key: b for b in BLOCKS}


@dataclass(frozen=True)
class RunSet:
    """The blocks of one phone stage, its command file, the directory of its outputs, the directory of its
    stage files (build.sh OUT) and the text about its libraries."""
    blocks: tuple
    commands: str
    out_dir: str
    text: str
    minutes: str
    phone: str = "phone"
    libs: str = ("the patched llama.cpp tree of HEAD (tests/sanitizers/llama-copy.sh) plus build/fuse-mm/patches/0001 "
                 "(MUL_MAT_NX_SWIGLU) and 0002 (the F16 SwiGLU output for the HMX MUL_MAT)")


FUSION_BLOCKS = ("tbo", "reg", "mm", "ffn", "ffncpu", "klp", "kld", "p", "t", "prof")
SETS = {
    "full": RunSet(FUSION_BLOCKS, "phone-commands.txt", "phone-out",
                   "the op tests, ffncheck, the KL runs, the timing and the op profile",
                   "about 23 minutes of tools"),
    "check": RunSet(("tbo", "ffn", "ffncpu", "klp", "kld"), "phone-commands-check.txt", "phone-out-check",
                    "the correctness only: test-backend-ops FFN_SWIGLU, ffncheck (with the weights in a buffer of "
                    "the WEIGHTS usage) and the KL runs",
                    "about 8 minutes of tools"),
    "prof": RunSet(("prof",), "phone-commands-prof.txt", "phone-out-prof",
                   "the op profile only: pp512, pp1024 and tg4, on and off", "about 2 minutes of tools",
                   "phone-prof"),
    "repack": RunSet(("plain", "rffn", "rmm", "rmmid", "rklp"), "phone-commands-repack.txt", "phone-out-repack",
                     "the patch that refuses a MUL_MAT weight of a tiled type in the plain layout",
                     "about 5 minutes of tools", "phone-repack",
                     "the patched llama.cpp tree of HEAD plus the patch build/fuse-mm/repack/*.patch and no fusion "
                     "patch (OUT=phone-repack)"),
    "f16": RunSet(("ftbo", "fffn", "fkld", "fp", "ft", "fprof"), "phone-commands-f16.txt", "phone-out-f16",
                  "the F16 SwiGLU output for ffn_down with no fused op: the timing against off, the op profile, "
                  "and the decode KL two times for each variant", "about 25 minutes of tools", "phone-f16",
                  "the patched llama.cpp tree of HEAD plus the patch build/fuse-mm/f16/*.patch only (OUT=phone-f16)"),
}


@dataclass(frozen=True)
class Run:
    """One phone run of one block, round and variant."""
    block: Block
    round: int
    variant: str

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"{self.block.key}-{self.round}-{self.variant}"


def all_runs(keys: tuple = tuple(BLOCK)) -> list:
    """The runs of the blocks keys in the order of the stage: the other blocks (each round of a block, the
    variants in their order), then the timing blocks round by round (round 1 of each timing block, then round
    2, ...), then the tail blocks. A timing round runs the variants in the order of the block in an odd round
    and in the reverse order in an even round. O(runs)."""
    sel = [b for b in BLOCKS if b.key in keys]
    out = []
    for b in sel:
        if not b.timing and not b.tail:
            out += [Run(b, rnd, v) for rnd in range(1, b.rounds + 1) for v in b.variants]
    timing_blocks = [b for b in sel if b.timing]
    for rnd in range(1, max((b.rounds for b in timing_blocks), default=0) + 1):
        for b in timing_blocks:
            if rnd <= b.rounds:
                order = b.variants if rnd % 2 else b.variants[::-1]
                out += [Run(b, rnd, v) for v in order]
    for b in sel:
        if b.tail:
            out += [Run(b, rnd, v) for rnd in range(1, b.rounds + 1) for v in b.variants]
    return out


def run_lines(run: Run) -> list:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    b = run.block
    env = " ".join(x for x in (LIB_ENV, VARIANTS[run.variant], b.extra_env) if x)
    # A block with no argument keeps the space of the empty args, thus each command line stays the same.
    cmd = commands.gated_run(f"{PHONE}/out/{run.name}", GATE_KB, b.limit, env,
                             f"{PHONE}/bin/{b.tool} {b.args}", stage=PHONE)
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if b.tool in ("llama-bench", "llama-perplexity") else "OP-TEST"
    return commands.run_lines(f"# {title}: {run.name}, {b.text}, variant {run.variant}", cmd, PGREP)


HEADER = """\
# Phone stage "fuse-mm", run set "{name}": {text}.
{scope}
#
{libs}
#
{variants}# All runs have GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1 (the app) unless the variant changes it.
#
# The runs:
{runs}
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,
# MemAvailable 8 GB, and it prints the caps), the tool under timeout -s KILL (110 s or less), the exit code and the
# conditions after the run (thermal, caps, battery), then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: the gate needs an unlocked phone and no charger.
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: {minutes} plus about 8 s of gate and
# checks for each of the {n_runs} runs, plus the waits for thermal status 0. Then, on the box:
#   tools/stages/fuse-mm/stage.py table --root build/fuse-mm/{out_dir}
"""

VARIANTS_TEXT = """\
# The variants: on (the preset values), sw (GGML_HEXAGON_FUSE_F16_ACT=0), f16 (GGML_HEXAGON_FUSE_SWIGLU=0 and
# GGML_HEXAGON_FUSE_SWIGLU_DECODE=0), off (the three switches 0, the ops of HEAD).
"""

RUN_TEXT = {
    "tbo": "test-backend-ops -o FFN_SWIGLU, on f16 off: each case must pass",
    "reg": "test-backend-ops -o MUL_MAT_VEC_FUSION,SWIGLU,GEGLU, on: each case must pass",
    "mm": "test-backend-ops -o MUL_MAT, on: each case must pass",
    "ffn": ("ffncheck, on sw f16 off: the hashes of each case must be equal (bit-identical outputs). The cases "
            "bigh and bigy have values outside the F16 range: their hashes and non-finite counts must also be equal"),
    "ffncpu": "ffncheck --cpu, on: the NMSE against the CPU backend of the phone, and hmax and ymax of the CPU",
    "klp": "llama-perplexity -b 512, 1 chunk, KL against naive-4B-q8.kld, on off: the numbers must be equal",
    "kld": "llama-perplexity -b 1 -ub 1, 1 chunk, the same, on off: the numbers must be equal",
    "p": "llama-bench pp512 and pp1024 at d0 and d3072, -r 3, on off sw, 3 rounds (the timing)",
    "t": "llama-bench tg32 at d0 and d4096, -r 3, on off, 3 rounds (the timing)",
    "prof": ("GGML_HEXAGON_PROFILE=1 llama-bench -v pp512, pp1024 and tg4, on off (the op split, "
             "tools/prof/optable.py)"),
    "plain": ("ffncheck --plain --only ffn_: supports_op must refuse a MUL_MAT and graph_compute must fail for "
              "each case (supports=no compute=failed)"),
    "rffn": "ffncheck with the WEIGHTS usage: each case must be finite",
    "rmm": "test-backend-ops -o MUL_MAT: each case must pass, with the pass count of mm in phone-out (673)",
    "rmmid": "test-backend-ops -o MUL_MAT_ID: each case must pass",
    "rklp": ("llama-perplexity -b 512, 1 chunk, KL against naive-4B-q8.kld: the numbers of klp off of the set "
             "check (the same ops)"),
    "ftbo": "test-backend-ops -o FFN_SWIGLU, f16: each case must pass",
    "fffn": "ffncheck, f16 off: the hashes and the non-finite counts of each case must be equal",
    "fkld": ("llama-perplexity -b 1 -ub 1, 1 chunk, KL against naive-4B-q8.kld, off f16 off f16: the two runs of "
             "one variant tell if the decode path is deterministic, and f16 must equal off"),
    "fp": "llama-bench pp512 and pp1024 at d0 and d3072, -r 3, f16 off, 3 rounds (the timing)",
    "ft": "llama-bench tg32 at d0 and d4096, -r 3, f16 off, 3 rounds (the timing)",
    "fprof": ("GGML_HEXAGON_PROFILE=1 llama-bench -v pp512, pp1024 and tg4, f16 off (the op split, "
              "tools/prof/optable.py)"),
}

F16_VARIANTS_TEXT = """\
# The variants: f16 (the preset values: the F16 SwiGLU output for ffn_down), off (GGML_HEXAGON_FUSE_F16_ACT=0, the
# ops of HEAD). The variant f16 also sets GGML_HEXAGON_FUSE_SWIGLU=0 and _DECODE=0, which this build does not read.
"""


def setup_lines(phone: str) -> list:
    """The lines that copy the stage files of build/fuse-mm/<phone> to the phone and check them. The push lines
    take each file of bin/ and lib/, and sha256sum -c on the phone makes sure that each file of SHA256SUMS is
    there.

    `grep -c OK` gives the exit code 0 also when one file of the stage is missing. Thus the last line runs
    sha256sum -c a second time for its exit code, and it replaces the last line of commands.setup_lines.
    """
    local = f"{LAPTOP_STAGE}/{phone}"
    # No "models/Qwen3.5" in the model line: the runner gates each line with that text as a model run.
    lines = commands.setup_lines(
        PATHS, {d: [f"{local}/{d}/*"] for d in ("bin", "lib")}, sub=phone,
        model_check=f"{ADB} shell 'ls -l {device.MODEL_DIR} | grep -E \"{device.MODEL_4B}\"; "
                    f"ls -l {EVAL}/wiki.test.raw {EVAL}/naive-4B-q8.kld'")
    lines[-1] = (f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && "
                 f"sha256sum -c SHA256SUMS > /dev/null "
                 f"&& echo \"stage files: all present\"; chmod 755 {PHONE}/bin/*'")
    return lines


def write_commands(name: str, path: Path) -> int:
    """Write the command file of the run set name and return its line count."""
    rs = SETS[name]
    runs = all_runs(rs.blocks)
    libs = textwrap.wrap(f"The libraries (tools/stages/fuse-mm/build.sh): {rs.libs}, built with the preset, the "
                         "flags and the LTO of scripts/build-native.sh. "
                         f"build/fuse-mm/{rs.phone}/patches.sha256 names the patches.", 114, break_on_hyphens=False)
    fusion = any(k in FUSION_BLOCKS for k in rs.blocks) or name == "f16"
    variants = F16_VARIANTS_TEXT if name == "f16" else (VARIANTS_TEXT if fusion else "")
    header = HEADER.format(name=name, text=rs.text, minutes=rs.minutes, n_runs=len(runs), out_dir=rs.out_dir,
                           libs="\n".join(f"# {x}" for x in libs), variants=variants,
                           scope=("# The matmul fusions of HTP0 (the 4B Q8_0 only), with one library set and the "
                                  "environment switches of the fusions." if fusion else
                                  "# The tiled layout of the MUL_MAT weights of HTP0 (the 4B Q8_0 and the op tests)."),
                           runs="\n".join(f"#   {k:7s} {RUN_TEXT[k]}" for k in rs.blocks))
    lines = commands.header_lines(header) + setup_lines(rs.phone)
    lines += ["#", f"# ==== Qwen3.5-4B-Q8_0: {len(runs)} runs ===="]
    for run in runs:
        lines += run_lines(run)
    lines += commands.output_lines(PATHS, tools=TOOLS, out_dir=rs.out_dir)
    return commands.write_commands(path, lines)


# ---- The table ----

TBO_RE = re.compile(r"(\d+)/(\d+) tests passed")
FFN_RE = re.compile(r"^ffncheck case=(\S+) hash=(\w+) nonfinite=(\d+)"
                    r"(?: nmse=(\S+) maxerr=(\S+)(?: hmax=(\S+) ymax=(\S+))?)?", re.M)
PLAIN_RE = re.compile(r"^ffncheck case=(\S+) plain supports=(yes|no\(.*?\)) compute=(\w+)", re.M)
KL_KEYS = ("Mean    KLD", "Maximum KLD", "Same top p", "Mean PPL(Q)", "RMS Δp")


@dataclass
class Result:
    """The parsed files of one run. ok is False when the run did not run or failed. flags names each
    condition that makes a timing run not comparable (changed caps, heat)."""
    run: Run
    ok: bool
    flags: list
    out: str
    log: str


def read_result(root: Path, run: Run) -> Result:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    text, out, log = logs.read_run(root, run.name)
    c = gate.read(text)
    return Result(run, c.ok, c.flags, out, log)


def correctness(results: dict) -> list:
    """The pass counts of the op tests, the hash comparison of ffncheck and the KL comparison."""
    out = ["correctness"]
    for key in ("tbo", "reg", "mm", "rmm", "rmmid", "ftbo"):
        for v in BLOCK[key].variants:
            res = results.get(f"{key}-1-{v}")
            if res is None:
                continue
            m = TBO_RE.findall(res.out + res.log)
            passed = f"{m[-1][0]}/{m[-1][1]} passed" if m else "no pass count"
            fails = len(re.findall(r"\[(?:FAIL|ERR)\]|  FAIL", res.out))
            out.append(f"  {res.run.name}: {passed}, {fails} FAIL lines, {'ok' if res.ok else ', '.join(res.flags)}")
    for key in ("ffn", "fffn"):
        runs = {}
        for v in BLOCK[key].variants:
            res = results.get(f"{key}-1-{v}")
            if res is not None:
                runs[v] = {m[0]: (m[1], m[2]) for m in FFN_RE.findall(res.out)}
                if not res.ok:
                    out.append(f"  {key}-1-{v}: " + ", ".join(res.flags))
        cases = sorted({c for r in runs.values() for c in r})
        for c in cases:
            row = {v: runs[v].get(c, ("-", "-")) for v in runs}
            same = len(set(row.values())) == 1 and ("-", "-") not in row.values()
            out.append(f"  ffncheck {c:13s} " + " ".join(f"{v}={h}/{n}" for v, (h, n) in row.items())
                       + ("  equal" if same else "  DIFFERENT"))
    res = results.get("ffncpu-1-on")
    if res is not None:
        # The HMX path of HEAD gives an NMSE of about 1e-4 against the CPU for the FFN block (F16 activations
        # and F16 output tiles), thus the 1e-4 bound of ffncheck --cpu is a check of the tool, and a case
        # above it is not a defect of a patch when the hashes of the variants are equal.
        out.append("  ffncheck --cpu (the HMX path of HEAD has an NMSE of about 1e-4 for the ffn cases; compare "
                   "the hashes of the variants, not this bound)")
        for m in FFN_RE.findall(res.out):
            out.append(f"  ffncheck --cpu {m[0]:13s} nmse={m[3]} maxerr={m[4]} nonfinite={m[2]} hmax={m[5]} "
                       f"ymax={m[6]}")
    res = results.get("plain-1-on")
    if res is not None:
        lines = PLAIN_RE.findall(res.out)
        good = sum(1 for m in lines if m[1].startswith("no") and m[2] == "failed")
        out.append(f"  ffncheck --plain: {good}/{len(lines)} cases refused (supports=no, compute=failed), "
                   f"{'ok' if res.ok else ', '.join(res.flags)}")
        out += [f"    {m[0]} supports={m[1]} compute={m[2]}" for m in lines if not (m[1].startswith("no") and
                                                                                   m[2] == "failed")]
    res = results.get("rffn-1-on")
    if res is not None:
        lines = FFN_RE.findall(res.out)
        bad = [m[0] for m in lines if m[2] != "0" and not m[0].startswith("big")]
        out.append(f"  ffncheck (repack set): {len(lines)} cases, non-finite in {bad or 'none'}, "
                   f"{'ok' if res.ok else ', '.join(res.flags)}")
    for key in ("klp", "kld", "rklp", "fkld"):
        b = BLOCK[key]
        rows = {}
        for rnd in range(1, b.rounds + 1):
            for v in b.variants:
                res = results.get(f"{key}-{rnd}-{v}")
                if res is None:
                    continue
                text = res.out + res.log
                label = v if b.rounds == 1 else f"{v}#{rnd}"
                rows[label] = {k: (re.search(re.escape(k) + r"\s*:\s*(.*)", text) or [None, "-"])[1].strip()
                               for k in KL_KEYS}
        for k in KL_KEYS if rows else ():
            vals = [rows[v][k] for v in rows]
            out.append(f"  {key} {k:12s} " + " | ".join(f"{v} {rows[v][k]}" for v in rows)
                       + ("  equal" if len(set(vals)) == 1 and "-" not in vals else "  DIFFERENT"))
    return out


def timing(results: dict, include_all: bool) -> list:
    """The t/s of each llama-bench test: per variant the median of the rounds, the paired difference to off,
    the lowest and the highest round, and the count of rounds. O(runs)."""
    out = ["timing: t/s as the median of the rounds, the difference to off (the median of the ratios of the runs "
           "of one round), [the lowest and the highest round], n and the count of rounds"]
    for b in BLOCKS:
        if not b.timing:
            continue
        key = b.key
        per: dict = {v: {} for v in b.variants}
        for rnd in range(1, b.rounds + 1):
            for v in b.variants:
                res = results.get(f"{key}-{rnd}-{v}")
                if res and res.ok and (include_all or not res.flags):
                    per[v][rnd] = parse.bench_values(res.out)
        tests = sorted({t for v in per for r in per[v] for t in per[v][r]})
        if tests:
            out.append(f"  {key}:")
        for t in tests:
            label = f"pp{t[0]}" if t[0] else f"tg{t[1]}"
            cells = []
            for v in b.variants:
                vals = {r: per[v][r][t] for r in per[v] if t in per[v][r]}
                if not vals:
                    cells.append(f"{v} -")
                    continue
                cell = f"{v} {statistics.median(vals.values()):.2f}"
                if v != "off":
                    ratios = [vals[r] / per["off"][r][t] for r in vals if r in per["off"] and t in per["off"][r]]
                    cell += f" {100 * (statistics.median(ratios) - 1):+.2f}%" if ratios else " ?"
                cell += f" [{min(vals.values()):.2f}-{max(vals.values()):.2f}] n{len(vals)}"
                cells.append(cell)
            out.append(f"  {label:7s} d{t[2]:<5d} | " + " | ".join(cells))
    for r in results.values():
        if r.run.block.timing and r.flags:
            out.append(f"  {r.run.name}: " + ", ".join(r.flags))
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        return cli.missing_root(root)
    runs = all_runs()
    results = {r.name: read_result(root, r) for r in runs if (root / f"{r.name}-gate.txt").exists()}
    print(f"{len(results)} runs have a gate file, {sum(r.ok for r in results.values())} ran with exit code 0")
    for part in (correctness(results), timing(results, include_all)):
        print("\n".join(part))
        print()
    for r in results.values():
        if r.run.block.tail:
            print(f"the op split: tools/prof/optable.py ops {root}/{r.run.name}.log --graphs 0 --by shape")
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file of a run set")
    c.add_argument("--set", choices=sorted(SETS), default="check", help="the run set (preset: check)")
    c.add_argument("--out", type=Path, help="the command file (preset: build/fuse-mm/<the file of the set>)")
    t = sub.add_parser("table", help="print the tables from the pulled logs")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    t.add_argument("--all", action="store_true", help="also use the runs with changed caps or heat")
    a = ap.parse_args()
    if a.cmd == "commands":
        path = a.out or STAGE_DIR / SETS[a.set].commands
        n = write_commands(a.set, path)
        print(f"{path}: {n} lines, {len(all_runs(SETS[a.set].blocks))} runs")
        return 0
    return table(a.root, a.all)


if __name__ == "__main__":
    sys.exit(main())
