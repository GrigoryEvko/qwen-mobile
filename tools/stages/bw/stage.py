#!/usr/bin/env python3
"""The phone stage bw: the DDR read ceiling of the NSP and of the CPU, and the rate of the GEMV kernels.

Usage:
    stage.py tests [--out DIR]                write the test files of test-backend-ops (build/bw/phone/tests)
    stage.py commands [--out PATH]            write the phone command file (build/bw/phone-commands.txt)
    stage.py table [--root DIR] [--strict]    print the tables from the pulled files (build/bw/phone-out)

The file is tools/stages/bw/stage.py, and build/bw/stage.py is a symbolic link to it.

The questions of the stage and the runs that answer them:
    1. The read ceiling of the NSP: tools/ddrbw (ddrbw nsp) reads 512 MiB of rpcmem with HVX loads
       and l2fetch, with the DMA engine into the VTCM (the descriptors of the GEMV kernels), and with
       the two together, at 1, 2, 4 and 6 threads, with the votes of the backend and with more votes.
    2. The read ceiling of the CPU: ddrbw cpu, NEON loads from pinned threads.
    3. The NSP and the CPU at the same time (ddrbw both), and the cost of one CPU-DSP synchronization
       (ddrbw rtt), and the cost of the mapping of one 1 GiB model chunk (ddrbw mapcost).
    4. The GEMV rate of the backend at the 4B shapes: test-backend-ops perf and test with test files.
    5. The cost of an op boundary: one GEMV against 8, 32 and 128 GEMVs of the same bytes.
    6. The frequency nodes that the shell can read, sampled during the NSP runs and during a 4B decode.

One run list (RUNS) gives the command file and the parser, thus the two agree on each run name. Each
run writes to the phone directory out/: <name>-gate.txt (the conditions before and after the run and
the exit code), <name>.out (stdout), <name>.log (stderr), and <name>-freq.txt when a frequency
sampler runs next to the tool. The table uses a run when its gate passed and its exit code is 0 (4
for a checksum error is shown). A value with "*" comes from a run whose CPU caps changed or whose
thermal status after it was not 0: the table keeps it and marks it, and --strict drops it. The table
only reads files. O(size of the files) time.
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path



def find_repo() -> Path:
    """The repository root: the first directory above this file (as called, without the resolution of
    the symbolic link) that holds both build/ and tools/. The box calls build/bw/stage.py or
    tools/stages/bw/stage.py, and the laptop calls its copy build/bw/stage.py."""
    here = Path(__file__).absolute().parent
    for d in (here, *here.parents):
        if (d / "build").is_dir() and (d / "tools").is_dir():
            return d
    # A checkout with no tools/ (the laptop): the copy build/bw/stage.py gives the root two levels up.
    return here.parent.parent if here.name == "bw" and here.parent.name == "build" else here


REPO = find_repo()
HERE = REPO / "build" / "bw"

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/bw"
MODEL_DIR = "/data/local/tmp/qwen/models"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
LAPTOP_STAGE = "build/bw"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/bw"
# The environment of the app (init_impl in llama_jni.cpp) and the stage libraries.
LIB_ENV = (f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib "
           "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1")
DDRBW_ENV = f"ADSP_LIBRARY_PATH={PHONE}/lib"
LIBS = ("libddrbw_skel.so", "libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-htp-v79.so",
        "libggml-opencl.so", "libggml.so", "libllama-bench-impl.so", "libllama-common.so", "libllama.so",
        "libmtmd.so")
BINS = ("ddrbw", "gate.sh", "llama-bench", "memprobe", "test-backend-ops")
TEST_FILES = ("chain-q8_0.txt", "gemv-f16.txt", "gemv-q4_0.txt", "gemv-q8_0.txt", "head-f16.txt", "head-q4_0.txt",
              "head-q8_0.txt", "kern-f16.txt", "kern-q4_0.txt", "kern-q8_0.txt")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = (f"{ADB} shell 'pgrep -x llama-bench; pgrep -x memprobe; pgrep -x ddrbw; pgrep -x test-backend-op; "
         "echo pgrep-done'")
# The highest temperature of the NPU thermal zones (type nsp*) in millidegrees.
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')
# The CPU sets of ddrbw cpu. Cores 0 to 5 are the six cores of the first cluster, cores 6 and 7 the
# two cores of the second cluster (the higher clock).
CPU_SETS = "7/0/6,7/0,1/0-3/4-7/0,1,6,7/0-5/0-7"
BOTH_SETS = "6,7/0-5/0-7"
BOTH_CFGS = "gemv-t6,dma-t6,hvx-t6"
GB = 1e9


@dataclass(frozen=True)
class Run:
    """One phone run: its name, the table that reads it, the tool and its arguments, the environment,
    the time limit in seconds, the MemAvailable (KiB) of the gate, and the sampler rate (0: no sampler)."""
    name: str
    block: str
    tool: str
    args: str
    env: str
    limit: int
    gate_kb: int
    freq_hz: int
    text: str


GATE_SMALL = 2097152
GATE_BIG = 5242880
GATE_MODEL = 8388608

# The 4B shapes (k, m, label) of one decode token: bytes.py budget 4b and the op profile of bench-kv.
SHAPES_4B = (
    (2560, 9216, "ffn gate, ffn up"),
    (9216, 2560, "ffn down"),
    (2560, 8192, "GDN qkv, attn q+gate"),
    (2560, 4096, "GDN z"),
    (4096, 2560, "GDN out, attn out"),
    (2560, 1024, "attn k, attn v"),
)
HEAD_4B = (2560, 248320, "output head")
ROWS = (1, 2, 4)
# The op boundary: the same bytes as one GEMV and as 8, 32 and 128 GEMVs of m / N rows.
CHAIN = ((2560, 8192), (9216, 4096))
CHAIN_N = (1, 8, 32, 128)


def test_files() -> dict[str, list[str]]:
    """The test files of test-backend-ops: file name -> lines. The line format comes from
    tools/prof/gemm.py of the box repository, thus the import is here: the table needs no gemm.py.
    O(cases)."""
    sys.path.insert(0, str(REPO / "tools" / "prof"))
    import gemm  # noqa: PLC0415  (tools/prof/gemm.py: the line format of --test-file)
    files: dict[str, list[str]] = {}
    for t in ("q8_0", "f16", "q4_0"):
        files[f"gemv-{t}.txt"] = [gemm.case(k, m, n, t) for k, m, _ in SHAPES_4B for n in ROWS]
        files[f"head-{t}.txt"] = [gemm.case(HEAD_4B[0], HEAD_4B[1], n, t) for n in ROWS]
    # The f32 GEMVs of the GDN layers (ssm_alpha, ssm_beta: 2560 x 32) of the other ops of a token.
    files["gemv-q8_0.txt"].append(gemm.case(2560, 32, 1, "f32"))
    files["chain-q8_0.txt"] = [gemm.case(k, m // n, 1, "q8_0") for k, m in CHAIN for n in CHAIN_N]
    for t in ("q8_0", "f16", "q4_0"):
        files[f"kern-{t}.txt"] = files[f"gemv-{t}.txt"] + files[f"head-{t}.txt"]
    files["kern-q8_0.txt"] += files["chain-q8_0.txt"]
    return files


def runs() -> list[Run]:
    """The runs of the stage in their order. O(runs)."""
    r: list[Run] = [
        Run("info-1-backend", "info", "ddrbw", "info --votes backend", DDRBW_ENV, 30, GATE_SMALL, 0,
            "ddrbw info: the library loads, the votes of the backend"),
        Run("freqlist-1", "freq", "ddrbw", "freqlist", "", 20, GATE_SMALL, 0,
            "the frequency nodes that the shell can read"),
    ]
    for rnd, votes in ((1, "backend"), (1, "max"), (2, "max"), (2, "backend")):
        r.append(Run(f"nsp-{rnd}-{votes}", "nsp", "ddrbw",
                     f"nsp --votes {votes} --plan full --reps 3 --budget-ms 300", DDRBW_ENV, 100, GATE_SMALL, 10,
                     f"the NSP read ceiling, all configurations, votes {votes}"))
    for votes in ("none", "ddrperf", "busperf", "expv", "ceng", "bw"):
        r.append(Run(f"nsp-1-{votes}", "nsp", "ddrbw",
                     f"nsp --votes {votes} --plan short --reps 2 --budget-ms 300", DDRBW_ENV, 60, GATE_SMALL, 10,
                     f"the NSP read ceiling, five configurations, votes {votes}"))
    r.append(Run("cpu-1", "cpu", "ddrbw", f"cpu --sets {CPU_SETS} --reps 3 --budget-ms 300", "", 60, GATE_SMALL, 0,
                 "the CPU read ceiling, NEON loads from pinned threads"))
    for votes in ("backend", "max"):
        r.append(Run(f"both-1-{votes}", "both", "ddrbw",
                     f"both --votes {votes} --cfgs {BOTH_CFGS} --sets {BOTH_SETS} --reps 2 --budget-ms 400",
                     DDRBW_ENV, 100, GATE_SMALL, 10, f"the NSP and the CPU at the same time, votes {votes}"))
    r.append(Run("rtt-1-backend", "rtt", "ddrbw", "rtt --votes backend --iters 2000", DDRBW_ENV, 60, GATE_SMALL, 0,
                 "the time of one CPU-DSP synchronization"))
    r.append(Run("mapcost-1-backend", "mapcost", "ddrbw", "mapcost --votes backend --mb 1024 --iters 20", DDRBW_ENV,
                 60, GATE_SMALL, 0, "the map and unmap of one 1 GiB model chunk"))
    for t in ("q8_0", "f16", "q4_0"):
        r.append(Run(f"kern-1-{t}", "kern", "test-backend-ops",
                     f"test -o MUL_MAT -b HTP0 --test-file {PHONE}/tests/kern-{t}.txt",
                     f"{LIB_ENV} GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1", 100, GATE_BIG, 0,
                     f"test-backend-ops test of the {t} GEMV shapes: the kernel and the result"))
    for f in ("gemv-q8_0", "head-q8_0", "gemv-f16", "head-f16", "gemv-q4_0", "head-q4_0", "chain-q8_0"):
        r.append(Run(f"perf-1-{f}", "perf", "test-backend-ops",
                     f"perf -o MUL_MAT -b HTP0 --test-file {PHONE}/tests/{f}.txt", LIB_ENV, 100, GATE_BIG, 0,
                     f"test-backend-ops perf of {f}"))
    r.append(Run("perf-2-gemv-q8_0", "perf", "test-backend-ops",
                 f"perf -o MUL_MAT -b HTP0 --test-file {PHONE}/tests/gemv-q8_0.txt", LIB_ENV, 100, GATE_BIG, 0,
                 "test-backend-ops perf of gemv-q8_0, round 2"))
    r.append(Run("cpu-2", "cpu", "ddrbw", f"cpu --sets {CPU_SETS} --reps 3 --budget-ms 300", "", 60, GATE_SMALL, 0,
                 "the CPU read ceiling, round 2"))
    r.append(Run("dec-1", "dec", "llama-bench",
                 f"-m {MODEL_DIR}/{MODEL} -dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -o jsonl -ctk q8_0 -ctv q8_0 "
                 "-p 0 -n 64 -r 2", LIB_ENV, 100, GATE_MODEL, 10,
                 "llama-bench tg64 of the 4B (variant b of bench-kv) with the sampler at 10 Hz"))
    r.append(Run("decprof-1", "decprof", "memprobe",
                 f"-m {MODEL_DIR}/{MODEL} -dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0 "
                 "-p 4096 -n 64", f"{LIB_ENV} GGML_HEXAGON_PROFILE=1", 110, GATE_MODEL, 1000,
                 "memprobe 4B op profile, 4096 prompt tokens and 64 decode tokens, the sampler at 1 kHz. The "
                 "memprobe of bench-kv has no --log-ts (the run of 2026-09-24 stopped at that argument)"))
    return r


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line. Each run line holds
    the text models/Qwen3.5 (a test of the model file), thus the runner applies to it the unlock wait,
    the screen wake, the memory gate and the CAPS line of a model run."""
    stem = f"{PHONE}/out/{run.name}"
    gate = f"{stem}-gate.txt"
    env = f"env {run.env} " if run.env else ""
    pre = ""
    post = ""
    if run.freq_hz:
        pre = (f"{PHONE}/bin/ddrbw freq --out {stem}-freq.txt --hz {run.freq_hz} --ms {run.limit * 1000 + 5000} "
               f"> /dev/null 2>&1 & fp=$!; {PHONE}/bin/ddrbw now > {stem}-now.txt; ")
        post = "kill $fp 2>/dev/null; wait $fp 2>/dev/null; "
    # The brace group keeps the background sampler ("&") out of the && list of the gate: without it,
    # "gate && ... && sampler &" would put the gate itself in the background.
    cmd = (f"test -r {MODEL_DIR}/{MODEL} && sh {PHONE}/bin/gate.sh {run.gate_kb} > {gate} && {BEFORE} >> {gate} && "
           f"{{ {pre}timeout -s KILL {run.limit} {env}{PHONE}/bin/{run.tool} {run.args} > {stem}.out 2> {stem}.log; "
           f"echo \"rc=$?\" >> {gate}; {post}}}; {AFTER} >> {gate}; cat {gate}")
    return ["#", f"# {run.name}: {run.text} (limit {run.limit} s)", THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "bw": the DDR read ceiling of the NSP and of the CPU, and the GEMV rate of the backend on HTP0.
# The question: can the NSP read the weights faster than the 52 to 54 GB/s of the GEMV kernels of the 4B Q8_0?
#
# The runs (tools/stages/bw/stage.py gives each one):
#   info, freqlist   the probe loads on the DSP; the frequency nodes of the shell
#   nsp              tools/ddrbw: 512 MiB of rpcmem, HVX loads with l2fetch, the DMA engine (the descriptors of
#                    the GEMV kernels), the two together, 1 to 6 threads, 300 ms per run, 3 repetitions; the votes
#                    of the backend and all votes (ABBA), then five configurations with each optional vote
#   cpu              NEON loads from pinned threads (1, 2, 4, 6, 8 cores), 3 repetitions, two rounds
#   both             the NSP alone, the CPU alone and the two together (paired), votes backend and max
#   rtt, mapcost     one CPU-DSP synchronization (FastRPC, dspqueue, shared-memory fence); the map of 1 GiB
#   kern, perf       test-backend-ops test (the kernel, GGML_HEXAGON_VERBOSE and PROFILE) and perf of the 4B
#                    GEMV shapes, Q8_0, F16 and Q4_0, n = 1, 2, 4; the chain of 8, 32 and 128 small GEMVs
#   dec, decprof     llama-bench tg64 of the 4B (variant b of bench-kv) with the sampler at 10 Hz; the memprobe
#                    op profile of 64 decode tokens with the sampler at 1 kHz
# The libraries: build/bench-kv/phone (HEAD e8a3a07 plus the switch GGML_HEXAGON_FWHT, the app libraries), and
# test-backend-ops of build/perf-tbo (the same build tree). The probe: tools/ddrbw (tools/stages/bw/build.sh).
#
# Each run: the thermal line, then the test of the model file (the text models/Qwen3.5 makes the runner treat the
# line as a model run: the unlock wait, the screen wake, the memory gate and the CAPS line), bin/gate.sh (the Qwen
# app stopped, the screen on, thermal 0, no charger, MemAvailable), the tool under timeout -s KILL (110 s or
# less), the exit code and the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 12 minutes of tool time and about
# 7 minutes of gates and checks (31 runs), plus the waits for thermal status 0.
# Then: python3 build/bw/stage.py table (on the laptop or on the box).
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/bin/{b}" for b in BINS)
    libs = " ".join(f"{LAPTOP_STAGE}/phone/lib/{lib}" for lib in LIBS)
    tests = " ".join(f"{LAPTOP_STAGE}/phone/tests/{name}" for name in TEST_FILES)
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No model path in this line: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODEL_DIR} | grep -E \"{MODEL}\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/tests {PHONE}/out'",
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {libs} {PHONE}/lib/",
        f"{ADB} push {tests} {PHONE}/tests/",
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
        f"{ADB} shell 'pgrep -x ddrbw; pgrep -x test-backend-op; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
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
    for run in runs():
        lines += run_lines(run)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The table ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
BEFORE_RE = re.compile(r"before: nsp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
KV_RE = re.compile(r"(\w+)=(\S+)")


@dataclass
class Result:
    """The files of one run. ok is True when the gate passed and the tool ran to its end (exit code 0,
    or 4 for a checksum error, which the tables show). flags names each condition of the run."""
    run: Run
    ok: bool
    rc: int | None
    flags: list[str]
    caps: str
    battery: str
    nsp: tuple[float | None, float | None]
    out: str
    log: str
    freq: str
    now: str


def read_result(root: Path, run: Run) -> Result:
    """Read the files of one run. O(size of the files)."""
    def text(suffix: str) -> str:
        p = root / f"{run.name}{suffix}"
        return p.read_text(errors="replace") if p.exists() else ""

    gate = text("-gate.txt")
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    m = re.search(r"^rc=(\d+)", gate, re.M)
    rc = int(m.group(1)) if m else None
    flags = []
    ok = "gate: OK" in gate and rc in (0, 4)
    if not gate:
        flags.append("no gate file")
    elif "gate: OK" not in gate:
        flags.append("gate stopped the run")
    elif rc != 0:
        flags.append(f"exit code {rc}")
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        flags.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    if after and after.group(1) not in ("", "0"):
        flags.append(f"thermal {after.group(1)} after the run")
    battery = f"{before.group(5)}% {int(before.group(6)) / 10:.1f} C" if before and before.group(6) else "?"
    nb = BEFORE_RE.search(gate)
    nsp = (int(nb.group(1)) / 1000 if nb and nb.group(1) else None,
           int(after.group(6)) / 1000 if after and after.group(6) else None)
    return Result(run, ok, rc, flags, caps, battery, nsp, text(".out"), text(".log"), text("-freq.txt"),
                  text("-now.txt"))


def ddrbw_lines(text: str, kind: str) -> list[dict[str, str]]:
    """The "ddrbw: <kind> key=value ..." lines of one output. O(lines)."""
    out = []
    prefix = f"ddrbw: {kind} "
    for line in text.splitlines():
        if line.startswith(prefix):
            out.append(dict(KV_RE.findall(line[len(prefix):])))
    return out


def med(vals: list[float]) -> float | None:
    """The median, or None for no value."""
    return statistics.median(vals) if vals else None


def fmt(x: float | None, nd: int = 2) -> str:
    """A number with nd decimals, or "-"."""
    return "-" if x is None else f"{x:.{nd}f}"


def conditions(results: list[Result]) -> list[str]:
    """The conditions of the runs: caps, battery, NPU temperature and each flag."""
    out = [f"conditions: {len(results)} runs have a gate file, {sum(r.ok for r in results)} ran to the end, "
           f"{sum(r.ok and not r.flags for r in results)} have no flag"]
    caps = defaultdict(int)
    for r in results:
        if r.ok:
            caps[r.caps] += 1
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.items()))
    temps = [r.nsp[1] for r in results if r.ok and r.nsp[1] is not None]
    if temps:
        out.append(f"  NPU zone after the runs: {min(temps):.1f} to {max(temps):.1f} C")
    for r in results:
        if r.flags:
            why = ""
            if r.rc not in (None, 0, 4):
                first = next((ln for ln in r.log.splitlines() if ln.strip()), "")
                why = f" (stderr: {first[:120]})" if first else ""
            out.append(f"  {r.run.name}: " + ", ".join(r.flags) + why)
    return out


# ---- the frequency samples

@dataclass
class Freq:
    """The samples of one sampler file: the node paths, and per line (cntvct, mono_us, values)."""
    nodes: list[str]
    rows: list[tuple[int, int, list[int]]]


def parse_freq(text: str) -> Freq:
    """Read a sampler file. O(lines)."""
    nodes: list[str] = []
    rows = []
    for line in text.splitlines():
        if line.startswith("# node "):
            nodes.append(line.split(" ", 3)[3])
        elif line and line[0].isdigit():
            f = line.split()
            rows.append((int(f[0]), int(f[1]), [int(x) for x in f[2:]]))
    return Freq(nodes, rows)


def freq_shares(fr: Freq, t0: int | None = None, t1: int | None = None) -> dict[str, dict[int, float]]:
    """For each node, the share of the time in [t0, t1] (cntvct) at each value. The value of a line
    holds until the next line. O(lines)."""
    rows = fr.rows
    if not rows:
        return {}
    t0 = rows[0][0] if t0 is None else t0
    t1 = rows[-1][0] if t1 is None else t1
    shares: dict[str, dict[int, float]] = {n: defaultdict(float) for n in fr.nodes}
    for i, (t, _, vals) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else t1
        a, z = max(t, t0), min(nxt, t1)
        if z <= a:
            continue
        for node, v in zip(fr.nodes, vals):
            shares[node][v] += z - a
    total = max(t1 - t0, 1)
    return {n: {v: d / total for v, d in s.items()} for n, s in shares.items() if s}


def short_node(path: str) -> str:
    """The short name of a node path."""
    parts = path.split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else path


def fmt_shares(sh: dict[int, float]) -> str:
    """The values of one node with their shares of the time, the largest first."""
    items = sorted(sh.items(), key=lambda kv: -kv[1])
    return " ".join(f"{v}:{100 * s:.0f}%" for v, s in items[:4])


def ddr_like(node: str) -> bool:
    """True for a node whose name suggests the DDR, the LLCC or a bus."""
    n = node.lower()
    return any(k in n for k in ("ddr", "llcc", "bw", "bus", "mem", "cdsp", "npu", "l3"))


# ---- the tables of ddrbw

MARK_NOTE = ("* = a value of the cell comes from a run whose CPU caps changed or whose thermal status was not 0 "
             "after the run (the conditions list names the runs)")


def use(r: Result, block: str, strict: bool) -> bool:
    """True when the run belongs to the block and goes into the tables. A run with a flag goes in unless
    strict is True."""
    return r.run.block == block and r.ok and not (strict and r.flags)


def mark(flagged: bool) -> str:
    """The mark of a cell with a value from a flagged run."""
    return "*" if flagged else ""


VOTE_ORDER = ("backend", "max", "none", "ddrperf", "busperf", "expv", "ceng", "bw")


def nsp_tables(results: list[Result], strict: bool) -> tuple[list[str], dict[str, float]]:
    """The NSP read table: GB/s per configuration and vote set. Returns the lines and the ceilings.
    O(lines)."""
    by = defaultdict(list)
    fails = defaultdict(int)
    tmins = defaultdict(list)
    windows = []
    cfg_order: list[str] = []
    facts = []
    flagged = set()
    for r in results:
        if not use(r, "nsp", strict):
            continue
        for d in ddrbw_lines(r.out, "votes"):
            facts.append((r.run.name, "votes", d))
        for d in ddrbw_lines(r.out, "facts") + ddrbw_lines(r.out, "facts-end"):
            facts.append((r.run.name, "facts", d))
        fr = parse_freq(r.freq)
        for d in ddrbw_lines(r.out, "nsp"):
            key = (d["cfg"], d["votes"])
            if d["cfg"] not in cfg_order:
                cfg_order.append(d["cfg"])
            if d.get("check") != "ok" or d.get("status") != "0":
                fails[key] += 1
                continue
            by[key].append(float(d["gbs"]))
            tmins[key].append(float(d["tmin"]))
            if r.flags:
                flagged.add(key)
            if fr.rows and "t_go" in d:
                windows.append((d["cfg"], d["votes"], freq_shares(fr, int(d["t_go"]), int(d["t_end"]))))
    votes = [v for v in VOTE_ORDER if any(k[1] == v for k in by)]
    out = ["NSP read (ddrbw nsp): GB/s of all threads = bytes / (last end - start); the median over the "
           "repetitions and rounds, [lowest-highest], n; tmin = the median of the slowest thread",
           f"  {'configuration':15s}" + "".join(f"| {v:24s}" for v in votes)]
    for cfg in cfg_order:
        cells = []
        for v in votes:
            vals = by.get((cfg, v), [])
            f = fails.get((cfg, v), 0)
            if not vals and not f:
                cells.append(f"| {'':24s}")
                continue
            c = f"{fmt(med(vals))} [{fmt(min(vals) if vals else None, 1)}-{fmt(max(vals) if vals else None, 1)}] n{len(vals)}"
            c += f" F{f}" if f else ""
            c += mark((cfg, v) in flagged)
            cells.append(f"| {c:24s}")
        out.append(f"  {cfg:15s}" + "".join(cells))
    ceil = {}
    for v in votes:
        for kind, pred in (("dma", lambda c: c.startswith(("dma", "gemv", "lin")) and "-t6" in c),
                           ("hvx", lambda c: c.startswith("hvx")),
                           ("gemv", lambda c: c.startswith("gemv")),
                           ("any", lambda c: not c.startswith("ws-") or c == "ws-full")):
            vals = [med(by[(c, v)]) for c in cfg_order if pred(c) and by.get((c, v))]
            if vals:
                ceil[f"{kind}-{v}"] = max(vals)
    out.append("  ceilings (the best median of a group): " +
               ", ".join(f"{k} {fmt(x)}" for k, x in sorted(ceil.items())))
    out.append("  the working-set rows (ws-*) show whether a re-read of a small region comes from a cache (the "
               "rate above the ws-full row)")
    if flagged:
        out.append("  " + MARK_NOTE)
    out.append("")
    out.append("NSP setup per vote run: the return codes of the votes, and the facts of the DSP")
    seen = set()
    for name, kind, d in facts:
        key = (name, kind, d.get("name", ""), d.get("core_hz", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(f"  {name:18s} {kind:5s} " + " ".join(f"{k}={v}" for k, v in d.items() if k != "votes"))
    if windows:
        out.append("")
        out.append("NSP runs and the frequency nodes (the share of the run time at each value, the DDR-like nodes)")
        # The mean share over all windows of one configuration and vote set: a window without a value
        # adds 0 to that value.
        agg: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        n_win: dict[tuple[str, str, str], int] = defaultdict(int)
        for cfg, v, sh in windows:
            for node, s in sh.items():
                if ddr_like(node):
                    n_win[(cfg, v, node)] += 1
                    for val, share in s.items():
                        agg[(cfg, v, node)][val].append(share)
        for (cfg, v, node), vals in sorted(agg.items()):
            mean = {val: sum(x) / n_win[(cfg, v, node)] for val, x in vals.items()}
            out.append(f"  {cfg:15s} {v:8s} {short_node(node):40s} {fmt_shares(mean)}")
    return out, ceil


def cpu_table(results: list[Result], strict: bool) -> list[str]:
    """The CPU read table. O(lines)."""
    by = defaultdict(list)
    tmin = defaultdict(list)
    tmax = defaultdict(list)
    pinned = defaultdict(list)
    fails = defaultdict(int)
    order: list[str] = []
    flagged = set()
    for r in results:
        if not use(r, "cpu", strict):
            continue
        for d in ddrbw_lines(r.out, "cpu"):
            s = d["set"]
            if s not in order:
                order.append(s)
            if d.get("check") != "ok":
                fails[s] += 1
                continue
            by[s].append(float(d["gbs"]))
            tmin[s].append(float(d["tmin"]))
            if r.flags:
                flagged.add(s)
            tmax[s].append(float(d["tmax"]))
            pinned[s].append(int(d["pinned"]))
    out = ["CPU read (ddrbw cpu, NEON, pinned): GB/s, the median over the repetitions and rounds",
           f"  {'cores':14s} {'GB/s':>7s} {'low':>7s} {'high':>7s} {'n':>3s} {'thread min':>11s} {'thread max':>11s} pinned"]
    for s in order:
        v = by.get(s, [])
        out.append(f"  {s:14s} {fmt(med(v)):>7s} {fmt(min(v) if v else None):>7s} {fmt(max(v) if v else None):>7s} "
                   f"{len(v):3d} {fmt(med(tmin[s])):>11s} {fmt(med(tmax[s])):>11s} "
                   f"{sum(pinned[s])}/{len(pinned[s])}" + (f" checksum FAIL x{fails[s]}" if fails.get(s) else "")
                   + (" " + mark(True) if s in flagged else ""))
    if flagged:
        out.append("  " + MARK_NOTE)
    return out


def both_table(results: list[Result], strict: bool) -> list[str]:
    """The table of the NSP and the CPU at the same time. O(lines)."""
    by = defaultdict(lambda: defaultdict(list))
    order = []
    bad = defaultdict(int)
    flagged = set()
    for r in results:
        if not use(r, "both", strict):
            continue
        for d in ddrbw_lines(r.out, "both"):
            key = (d["votes"], d["cfg"], d["set"])
            if key not in order:
                order.append(key)
            if d.get("check") != "ok" or d.get("clock") != "ok" or d.get("window") != "ok":
                bad[key] += 1
                continue
            for k in ("nsp_alone", "cpu_alone", "nsp_both", "cpu_both", "sum_both"):
                by[key][k].append(float(d[k]))
            if r.flags:
                flagged.add(key)
    out = ["NSP and CPU at the same time (ddrbw both): GB/s, the median over the repetitions. gain = sum_both / "
           "nsp_alone",
           f"  {'votes':8s} {'NSP':8s} {'CPU':8s} {'nsp alone':>10s} {'cpu alone':>10s} {'nsp both':>9s} "
           f"{'cpu both':>9s} {'sum both':>9s} {'gain':>6s}"]
    for key in order:
        m = {k: med(v) for k, v in by[key].items()}
        gain = m["sum_both"] / m["nsp_alone"] if m.get("sum_both") and m.get("nsp_alone") else None
        out.append(f"  {key[0]:8s} {key[1]:8s} {key[2]:8s} {fmt(m.get('nsp_alone')):>10s} {fmt(m.get('cpu_alone')):>10s} "
                   f"{fmt(m.get('nsp_both')):>9s} {fmt(m.get('cpu_both')):>9s} {fmt(m.get('sum_both')):>9s} "
                   f"{fmt(gain):>6s}" + (f"  (rejected {bad[key]}: checksum, clock or window)" if bad.get(key) else "")
                   + (" " + mark(True) if key in flagged else ""))
    if flagged:
        out.append("  " + MARK_NOTE)
    return out


def rtt_table(results: list[Result]) -> list[str]:
    """The table of the synchronization times and of the map cost. O(lines)."""
    out = ["CPU-DSP synchronization (ddrbw rtt), round trip in us: p10 p50 p90 p99 max"]
    for r in results:
        if r.run.block != "rtt" or not r.ok:
            continue
        for d in ddrbw_lines(r.out, "rtt"):
            name = " ".join(f"{k}={d[k]}" for k in ("kind", "dsp", "host", "buf") if k in d)
            if "error" in d or d.get("n") == "0":
                out.append(f"  {name:45s} error={d.get('error', 'no samples')} " +
                           " ".join(f"{k}={d[k]}" for k in ("done_host", "done_dsp") if k in d))
                continue
            out.append(f"  {name:45s} {d['p10']:>8s} {d['p50']:>8s} {d['p90']:>8s} {d['p99']:>8s} {d['max']:>9s}"
                       f"  n={d['n']}" + (f" served={d['served']}" if "served" in d else "") + mark(bool(r.flags)))
    out.append("")
    out.append("The map of one 1 GiB model chunk (ddrbw mapcost), in us: p10 p50 p90 max")
    for r in results:
        if r.run.block != "mapcost" or not r.ok:
            continue
        for d in ddrbw_lines(r.out, "mapcost"):
            if "kind" in d:
                out.append(f"  {d['kind']:12s} {d.get('p10', '-'):>8s} {d.get('p50', '-'):>8s} {d.get('p90', '-'):>8s} "
                           f"{d.get('max', '-'):>9s}  n={d['n']}" + mark(bool(r.flags)))
            else:
                out.append("  " + " ".join(f"{k}={v}" for k, v in d.items() if k != "votes") + mark(bool(r.flags)))
    return out


# ---- the tables of test-backend-ops

CASE_RE = re.compile(r"MUL_MAT\(type=\w+,ne=\[(?P<m>\d+),(?P<n>\d+),\d+,\d+\].*?sources=(?P<t>\w+)\[(?P<k>\d+),")
PERF_RE = re.compile(r"(?P<runs>\d+) runs -\s*(?P<us>[\d.]+) us/run")
PROF_RE = re.compile(r"profile-op (?P<op>[A-Z_0-9+]+)\|(?P<rest>.*?)\|usec (?P<usec>\d+) cycles")
BPE = {"q8_0": 34 / 32, "q4_0": 18 / 32, "f16": 2.0, "f32": 4.0}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def perf_rows(results: list[Result], strict: bool) -> tuple[dict[tuple, list[float]], set]:
    """us/run of each case (type, k, m, n) over the perf runs, and the cases with a value from a flagged
    run. O(lines)."""
    rows = defaultdict(list)
    flagged = set()
    for r in results:
        if not use(r, "perf", strict):
            continue
        for line in ANSI_RE.sub("", r.out).splitlines():
            c, p = CASE_RE.search(line), PERF_RE.search(line)
            if c and p:
                key = (c["t"], int(c["k"]), int(c["m"]), int(c["n"]))
                rows[key].append(float(p["us"]))
                if r.flags:
                    flagged.add(key)
            elif c and "not supported" in line:
                rows[(c["t"], int(c["k"]), int(c["m"]), int(c["n"]))]
    return rows, flagged


def kern_rows(results: list[Result]) -> dict[tuple, dict[str, str]]:
    """The kernel, the DSP time and the test result of each case of the test runs. O(lines)."""
    info: dict[tuple, dict[str, str]] = {}
    for r in results:
        if r.run.block != "kern" or not r.ok:
            continue
        for line in ANSI_RE.sub("", r.out).splitlines():
            c = CASE_RE.search(line)
            if c:
                key = (c["t"], int(c["k"]), int(c["m"]), int(c["n"]))
                res = "OK" if line.rstrip().endswith("OK") else ("not supported" if "not supported" in line
                                                                  else "FAIL" if "FAIL" in line else "?")
                info.setdefault(key, {})["test"] = res
        for m in PROF_RE.finditer(r.log):
            if not m["op"].startswith("MUL_MAT"):
                continue
            f = m["rest"].split("|")
            if len(f) < 5:
                continue
            srcs = f[1].split(" -> ")[0].split(" x ")
            t = f[2].split(" -> ")[0].split(" x ")[0].strip()
            if len(srcs) < 2:
                continue
            k, mm = (int(x) for x in srcs[0].split(":")[:2])
            n = int(srcs[1].split(":")[1])
            d = info.setdefault((t, k, mm, n), {})
            kernel = f[4].split(" vtcm")[0].strip()
            if kernel == "unknown" and 2 <= n <= 4 and t in ("q8_0", "q4_0"):
                # format_kernel_params of htp-opnode.h has no name for HTP_MM_KERNEL_HVX_QUANT_MULTIROW.
                kernel = "unknown=multirow"
            d["kernel"] = kernel + ("*" if r.flags else "")
            d["usec"] = m["usec"]
    return info


def gemv_tables(results: list[Result], ceil: dict[str, float], strict: bool) -> list[str]:
    """The GEMV table (weight GB/s against the NSP ceiling), the kernel table and the chain. O(cases)."""
    rows, flagged = perf_rows(results, strict)
    kern = kern_rows(results)
    ref = ceil.get("any-backend") or ceil.get("dma-backend")
    labels = {(k, m): lab for k, m, lab in SHAPES_4B + (HEAD_4B,)}
    out = [f"GEMV rate (test-backend-ops perf): weight bytes / us/run, decimal GB/s; % of the NSP ceiling "
           f"{fmt(ref)} GB/s (ddrbw, votes backend); kernel and single-op DSP us from the test run",
           f"  {'type':5s} {'k':>6s} {'m':>7s} {'n':>2s} {'us/run':>10s} {'GB/s':>7s} {'% ceil':>7s} {'n runs':>6s} "
           f"{'kernel':22s} {'1-op us':>8s} {'test':>6s}  shape"]
    for key in sorted(rows, key=lambda k: (k[0], k[1], -k[2], k[3])):
        t, k, m, n = key
        us = med(rows[key])
        wb = k * m * BPE.get(t, 0)
        gbs = wb / us / 1e3 if us else None
        kr = kern.get(key, {})
        out.append(f"  {t:5s} {k:6d} {m:7d} {n:2d} {fmt(us, 1):>10s} {fmt(gbs):>7s} "
                   f"{fmt(100 * gbs / ref if gbs and ref else None, 0):>7s} {len(rows[key]):6d} "
                   f"{kr.get('kernel', '-'):22s} {kr.get('usec', '-'):>8s} {kr.get('test', '-'):>6s}  "
                   f"{labels.get((k, m), '')}" + ("" if us else " not supported or no result") +
                   (" " + mark(True) if key in flagged else ""))
    if flagged or any(d.get("kernel", "").endswith("*") for d in kern.values()):
        out.append("  " + MARK_NOTE)
    missing = [k for k in kern if k not in rows]
    for key in missing:
        out.append(f"  {key[0]:5s} {key[1]:6d} {key[2]:7d} {key[3]:2d} (test run only) kernel {kern[key].get('kernel', '-')} "
                   f"usec {kern[key].get('usec', '-')} test {kern[key].get('test', '-')}")
    # The op boundary: N GEMVs of m / N rows against one GEMV of m rows, the same weight bytes.
    out += ["", "Op boundary (chain-q8_0): one GEMV of m rows against N GEMVs of m / N rows in one graph "
            "(test-backend-ops repeats one node). extra = (N x t(m/N) - t(m)) / (N - 1) is the cost of one more op",
            f"  {'k':>6s} {'m':>6s} {'N':>4s} {'t(m/N) us':>10s} {'N x t':>10s} {'t(m) us':>9s} {'extra us/op':>12s}"]
    fit_x, fit_y = [], []
    for k, m in CHAIN:
        big = med(rows.get(("q8_0", k, m, 1), []))
        for n in CHAIN_N[1:]:
            small = med(rows.get(("q8_0", k, m // n, 1), []))
            extra = (n * small - big) / (n - 1) if small and big else None
            out.append(f"  {k:6d} {m:6d} {n:4d} {fmt(small, 1):>10s} {fmt(n * small if small else None, 1):>10s} "
                       f"{fmt(big, 1):>9s} {fmt(extra, 2):>12s}")
    for (t, k, m, n), v in rows.items():
        if t == "q8_0" and n == 1 and v:
            fit_x.append(k * m * BPE[t])
            fit_y.append(med(v))
    if len(fit_x) >= 3:
        mx, my = statistics.fmean(fit_x), statistics.fmean(fit_y)
        sxx = sum((x - mx) ** 2 for x in fit_x)
        slope = sum((x - mx) * (y - my) for x, y in zip(fit_x, fit_y)) / sxx
        t0 = my - slope * mx
        out.append(f"  least-squares fit over the {len(fit_x)} Q8_0 n=1 cases: t = {t0:.1f} us + bytes / "
                   f"{1 / slope / 1e3:.1f} GB/s")
    return out


# ---- the decode runs

GRAPH_START = "-> attn_norm-0|"
TS_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) \w ")
OPLINE_RE = re.compile(r"profile-op (?P<op>[A-Z_0-9+]+)\|(?P<rest>.*?)\|usec (?P<usec>\d+) cycles")


def dec_tables(results: list[Result]) -> list[str]:
    """The decode rate with the frequency nodes, and the stall events of the op profile against the
    frequency samples at 1 kHz. O(lines)."""
    out = ["Decode of the 4B (llama-bench tg64, variant b) and the frequency nodes during the run"]
    for r in results:
        if r.run.block != "dec" or not r.ok:
            continue
        for line in r.out.splitlines():
            if line.startswith("{"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append(f"  {r.run.name}: tg{rec['n_gen']} {statistics.median(rec['samples_ts']):.2f} t/s "
                           f"(samples {', '.join(f'{x:.2f}' for x in rec['samples_ts'])}) caps {r.caps}"
                           + (" " + mark(True) + " " + ", ".join(r.flags) if r.flags else ""))
        fr = parse_freq(r.freq)
        for node, sh in freq_shares(fr).items():
            out.append(f"    {short_node(node):45s} {fmt_shares(sh)}")
    for r in results:
        if r.run.block != "decprof" or not r.ok:
            continue
        out += ["", f"Stall events of the op profile ({r.run.name}, memprobe, 64 decode tokens)"]
        graphs: list[list[tuple[str, int, float]]] = []
        cur = None
        for line in r.log.splitlines():
            m = OPLINE_RE.search(line)
            if not m or m["op"] == "OPBATCH":
                continue
            ts = TS_RE.match(line)
            t_us = ((int(ts[1]) * 60 + int(ts[2])) * 1000000 + int(ts[3]) * 1000 + int(ts[4])) if ts else -1
            if cur is None or GRAPH_START in line:
                cur = []
                graphs.append(cur)
            f = m["rest"].split("|")
            key = f"{m['op']}|{f[1] if len(f) > 1 else ''}"
            cur.append((key, int(m["usec"]), t_us))
        dec = graphs[4:] if len(graphs) > 4 else []
        out.append(f"  graphs {len(graphs)} (4 prefill ubatches expected), decode graphs {len(dec)}")
        per_key = defaultdict(list)
        for g in dec:
            for key, us, _ in g:
                per_key[key].append(us)
        medians = {k: statistics.median(v) for k, v in per_key.items()}
        events = []
        for gi, g in enumerate(dec):
            for key, us, t in g:
                if us - medians[key] > 1000:
                    events.append((gi, key, us, medians[key], t))
        tok = [sum(us for _, us, _ in g) / 1000 for g in dec]
        if tok:
            out.append(f"  op time per decode token: median {statistics.median(tok):.1f} ms, "
                       f"min {min(tok):.1f}, max {max(tok):.1f}")
        out.append(f"  events (an op more than 1 ms above the median of its op and shape): {len(events)}")
        for gi, key, us, mu, t in events[:40]:
            out.append(f"    token {gi:3d} {key[:70]:70s} {us:7d} us (median {mu:.0f}) log t {t / 1e6:.3f} s")
        # The frequency samples: the changes of each node during the decode window. The log clock of
        # memprobe starts at the tool start, a few 10 ms after the "now" line of ddrbw.
        fr = parse_freq(r.freq)
        now = dict(KV_RE.findall(r.now))
        if fr.rows and "mono_us" in now and dec and dec[0] and dec[-1] and dec[0][0][2] >= 0:
            base = int(now["mono_us"])
            d0 = base + dec[0][0][2]
            d1 = base + dec[-1][-1][2]
            rows = [row for row in fr.rows if d0 - 100000 <= row[1] <= d1 + 100000]
            out.append(f"  frequency lines in the decode window (+-100 ms, the clock offset has an error of some "
                       f"10 ms): {len(rows)}")
            for i, node in enumerate(fr.nodes):
                vals = [row[2][i] for row in rows]
                changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
                out.append(f"    {short_node(node):45s} changes {changes:4d} values "
                           f"{' '.join(str(v) for v in sorted(set(vals))[:6])}")
        elif fr.rows:
            # No log time stamps: the changes of each node over the whole run.
            out.append("  no log time stamps: the node changes over the whole run (load, prefill and decode)")
            for i, node in enumerate(fr.nodes):
                vals = [row[2][i] for row in fr.rows]
                changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
                out.append(f"    {short_node(node):45s} changes {changes:4d} values "
                           f"{' '.join(str(v) for v in sorted(set(vals))[:6])}")
    return out


def freq_table(results: list[Result]) -> list[str]:
    """The frequency nodes of freqlist. O(lines)."""
    out = ["Frequency nodes of the shell (ddrbw freqlist): the readable ones and their values"]
    for r in results:
        if r.run.block != "freq" or not r.ok:
            continue
        for d in ddrbw_lines(r.out, "node"):
            if d.get("readable") == "1":
                out.append(f"  {d['path']}: {d.get('value', '')}")
        unread = [d["path"] for d in ddrbw_lines(r.out, "node") if d.get("readable") == "0"]
        if unread:
            out.append(f"  not readable: {len(unread)} nodes, for example {', '.join(unread[:4])}")
        for line in r.out.splitlines():
            if line.startswith("ddrbw: freqnodes"):
                out.append("  sampled: " + line[len("ddrbw: freqnodes "):])
    return out


def table(root: Path, strict: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = [read_result(root, run) for run in runs() if (root / f"{run.name}-gate.txt").exists()]
    parts = [conditions(results)]
    nsp_lines, ceil = nsp_tables(results, strict)
    parts += [nsp_lines, cpu_table(results, strict), both_table(results, strict), rtt_table(results),
              gemv_tables(results, ceil, strict), freq_table(results), dec_tables(results)]
    for p in parts:
        print("\n".join(p))
        print()
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tests", help="write the test files of test-backend-ops")
    t.add_argument("--out", type=Path, default=HERE / "phone" / "tests")
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=HERE / "phone-commands.txt")
    tb = sub.add_parser("table", help="print the tables from the pulled files")
    tb.add_argument("--root", type=Path, default=HERE / "phone-out")
    tb.add_argument("--strict", action="store_true", help="drop the runs with changed caps or heat")
    a = ap.parse_args()
    if a.cmd == "tests":
        a.out.mkdir(parents=True, exist_ok=True)
        files = test_files()
        if sorted(files) != sorted(TEST_FILES):
            print(f"stage.py: TEST_FILES does not list the files of test_files(): {sorted(files)}", file=sys.stderr)
            return 1
        for name, lines in files.items():
            (a.out / name).write_text("\n".join(lines) + "\n")
        print(f"{a.out}: {len(test_files())} test files")
        return 0
    if a.cmd == "commands":
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(runs())} runs")
        return 0
    return table(a.root, a.strict)


if __name__ == "__main__":
    sys.exit(main())
