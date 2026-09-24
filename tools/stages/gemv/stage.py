#!/usr/bin/env python3
"""The phone stage gemv: the packed Q8_0 tiles of the HVX GEMV, the new DSP library against the base one.

Usage:
    stage.py tests [--out DIR]                write the test files of test-backend-ops (build/gemv/phone/tests)
    stage.py commands [--check-only] [--out PATH]
                                              write the phone command file (build/gemv/phone-commands.txt, or
                                              build/gemv-check/phone-commands.txt)
    stage.py table [--check-only] [--root DIR] [--all]
                                              print the tables from the pulled files (build/gemv/phone-out, or
                                              build/gemv-check/phone-out)
    stage.py libs [--check-only]              print the host libraries of the stage, one on each line

The file is tools/stages/gemv/stage.py, and build/gemv/stage.py and build/gemv-check/stage.py are symbolic links
to it. The stage files come from tools/stages/gemv/build.sh: one set of host libraries, and two DSP library
directories, dsp/new (HEAD plus the patch) and dsp/base (HEAD). A run selects its DSP library with
ADSP_LIBRARY_PATH.

With --check-only the file gives the check stage gemv-check (tools/stages/gemv/build-check.sh): only the check
runs of question 1, with its own phone directory. It uses the libraries of build/gemv/phone.

The questions of the stage and the runs that answer them:
    1. Does the new library give the bits of the base library? gemvcheck check (tools/gemv/gemvcheck.cpp)
       prints a hash of each output of the 4B decode matmuls (MUL_MAT, MUL_MAT_ADD, MUL_MAT_NX, the head,
       1 to 4 rows, and two shapes with an odd k-tile count) for each library: v79, and the v75 and v73
       libraries on the v79 DSP (GGML_HEXAGON_ARCH). The v73 library uses the 1D DMA form of a long row.
    2. Is the new library correct against the CPU? test-backend-ops test of MUL_MAT at the 4B shapes (with
       the kernel names of the profile), of all Q8_0 MUL_MAT and MUL_MAT_ID cases, and of the MUL_MAT cases
       of Q4_1, Q4_K and Q6_K (these types get one DMA row for each column tile, with the same bytes).
    3. The GEMV rate at the 4B shapes, base against new: test-backend-ops perf, and gemvcheck perf (it also
       times MUL_MAT_NX and MUL_MAT_ADD).
    4. Decode and prefill of the 4B, base against new, in 3 alternated rounds: llama-bench tg32 at the depths
       0, 4096 and 16384, and pp512 at the depth 0, with the flags of the variant b of bench-kv.
    5. One decode profile (GGML_HEXAGON_PROFILE=1) of each library: the op time and the weight rate.

One run list (RUNS) gives the command file and the parser, thus the two agree on each run name. Each run writes
to the phone directory out/: <name>-gate.txt (the conditions before and after the run and the exit code),
<name>.out (stdout) and <name>.log (stderr). The tables use a run when its gate passed and its exit code is 0;
a run whose CPU caps changed or whose thermal status after it was not 0 is marked with "*", and without --all
the timing tables drop it. The tables only read files. O(size of the files) time.
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
    """The repository root: the first directory above this file (as called, without the resolution of the
    symbolic link) that holds both build/ and tools/. The laptop copy build/gemv/stage.py gives the root two
    levels up."""
    here = Path(__file__).absolute().parent
    for d in (here, *here.parents):
        if (d / "build").is_dir() and (d / "tools").is_dir():
            return d
    return here.parent.parent if here.name == "gemv" and here.parent.name == "build" else here


REPO = find_repo()

ADB = "adb -s 192.168.14.130:5555"
MODEL_DIR = "/data/local/tmp/qwen/models"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
BOX_REPO = "grigory@10.10.20.200:airi/qwen-mobile"
DSP_LIBS = ("libggml-htp-v73.so", "libggml-htp-v75.so", "libggml-htp-v79.so")


@dataclass(frozen=True)
class Layout:
    """The places and the files of one stage: the phone directory, the stage directory (relative to the
    repository root, the same on the laptop and on the box), the programs of bin/, the host libraries of lib/
    and the test files of tests/."""
    phone: str
    stage: str
    bins: tuple[str, ...]
    libs: tuple[str, ...]
    tests: tuple[str, ...]

    @property
    def here(self) -> Path:
        """The stage directory in this repository."""
        return REPO / self.stage

    @property
    def box(self) -> str:
        """The stage directory on the box, for rsync."""
        return f"{BOX_REPO}/{self.stage}"


FULL = Layout("/data/local/tmp/qwen/gemv", "build/gemv", ("gate.sh", "gemvcheck", "llama-bench", "test-backend-ops"),
              ("libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-opencl.so", "libggml.so",
               "libllama-bench-impl.so", "libllama-common.so", "libllama.so", "libmtmd.so"),
              ("gemv-q8_0.txt", "kern-q8_0.txt"))
# The NEEDED entries of gemvcheck: libggml.so, libggml-cpu.so and libggml-base.so. libggml.so has entries for the two
# other backends.
CHECK = Layout("/data/local/tmp/qwen/gemvchk", "build/gemv-check", ("gate.sh", "gemvcheck"),
               ("libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-opencl.so", "libggml.so"), ())
HERE = FULL.here


def lib_env(lay: Layout) -> str:
    """The environment of the app (init_impl in llama_jni.cpp). ADSP_LIBRARY_PATH comes from the variant."""
    return f"LD_LIBRARY_PATH={lay.phone}/lib GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"


# The flags of the variant b of bench-kv (the app): flash attention on HTP0, a Q8_0 K and V cache.
BENCH_ARGS = f"-m {MODEL_DIR}/{MODEL} -dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -o jsonl -ctk q8_0 -ctv q8_0"
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = (f"{ADB} shell 'pgrep -x llama-bench; pgrep -x gemvcheck; pgrep -x test-backend-op; echo pgrep-done'")
# The highest temperature of the NPU thermal zones (type nsp*) in millidegrees.
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')

GATE_SMALL = 2097152
GATE_CHECK = 4194304
GATE_MODEL = 8388608

# The 4B shapes (k, m, label) of one decode token, and the output head
SHAPES_4B = (
    (2560, 9216, "ffn gate, ffn up"),
    (9216, 2560, "ffn down"),
    (2560, 8192, "GDN qkv, attn q+gate"),
    (2560, 4096, "GDN z"),
    (4096, 2560, "GDN out, attn out"),
    (2560, 1024, "attn k, attn v"),
)
HEAD_4B = (2560, 248320, "output head")
# The gemvcheck cases without the head (the check runs of the v75 and v73 libraries), and the cases of the
# gemvcheck perf runs. The names come from all_cases() of tools/gemv/gemvcheck.cpp.
CHECK_SHAPES = ("qkv_2560x8192", "gate_2560x4096", "kv_2560x1024", "down_add_9216x2560", "out_add_4096x2560",
                "nx_gate_up_2560x9216", "nx_q_v_k_2560x8192", "odd_2080x200", "nx_odd_2080x200")
NO_HEAD = ",".join(f"{s}_n{n}" for n in (1, 2, 3, 4) for s in CHECK_SHAPES)
PERF_CASES = ",".join(f"{s}_n{n}" for n in (1, 4) for s in CHECK_SHAPES[:7]) + ",head_2560x248320_n1"
BPE = {"q8_0": 34 / 32}


@dataclass(frozen=True)
class Run:
    """One phone run: its name, the table that reads it, the DSP variant (new or base), the tool and its
    arguments, more environment, the time limit in seconds, the MemAvailable (KiB) of the gate, and a text."""
    name: str
    block: str
    variant: str
    tool: str
    args: str
    env: str
    limit: int
    gate_kb: int
    text: str


def bench_runs() -> list[Run]:
    """The llama-bench A/B runs: 3 rounds, base then new in an odd round and new then base in an even round,
    thus a slow drift of the clocks or the heat goes equally to the two variants."""
    blocks = (("t", "-p 512 -n 32 -d 0 -r 3", 60, "pp512 and tg32 at the depth 0, 3 repetitions"),
              ("m", "-p 0 -n 32 -d 4096 -r 2", 80, "tg32 at the depth 4096, 2 repetitions"),
              ("l", "-p 0 -n 32 -d 16384 -r 1", 100, "tg32 at the depth 16384, 1 repetition"))
    out = []
    for rnd in (1, 2, 3):
        order = ("base", "new") if rnd % 2 else ("new", "base")
        for key, args, limit, text in blocks:
            for v in order:
                out.append(Run(f"bench-{rnd}-{key}-{v}", "bench", v, "llama-bench", f"{BENCH_ARGS} {args}", "", limit,
                               GATE_MODEL, f"llama-bench {text}, round {rnd}"))
    return out


def check_runs() -> list[Run]:
    """The bit check runs of base and new for the v79, v75 and v73 library. O(runs)."""
    r: list[Run] = []
    for v in ("base", "new"):
        r.append(Run(f"check-v79-{v}", "check", v, "gemvcheck", "check --cases all", "", 110, GATE_CHECK,
                     "gemvcheck check of the 40 cases with the v79 library"))
    for arch in ("v75", "v73"):
        for v in ("base", "new"):
            r.append(Run(f"check-{arch}-{v}", "check", v, "gemvcheck", f"check --cases {NO_HEAD}",
                         f"GGML_HEXAGON_ARCH={arch}", 90, GATE_CHECK,
                         f"gemvcheck check of the 36 cases without the head with the {arch} library"))
    return r


def runs(lay: Layout = FULL) -> list[Run]:
    """The runs of the stage in their order: the check runs only for CHECK, all runs for FULL. O(runs)."""
    r = check_runs()
    if lay == CHECK:
        return r
    phone = lay.phone
    r += [
        Run("tbo-4b-new", "tbo", "new", "test-backend-ops", f"test -o MUL_MAT -b HTP0 --test-file {phone}/tests/kern-q8_0.txt",
            "GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1", 90, GATE_SMALL,
            "test-backend-ops test of the 4B Q8_0 shapes, 1 to 4 rows, with the kernel of each op"),
        Run("tbo-q8-new", "tbo", "new", "test-backend-ops", "test -o MUL_MAT -b HTP0 -p type_a=q8_0", "", 110, GATE_SMALL,
            "test-backend-ops test of the Q8_0 MUL_MAT cases"),
        Run("tbo-id-new", "tbo", "new", "test-backend-ops", "test -o MUL_MAT_ID -b HTP0 -p type_a=q8_0", "", 110,
            GATE_SMALL, "test-backend-ops test of the Q8_0 MUL_MAT_ID cases"),
        Run("tbo-other-new", "tbo", "new", "test-backend-ops", 'test -o MUL_MAT -b HTP0 -p "type_a=(q4_1|q4_K|q6_K)"', "",
            110, GATE_SMALL, "test-backend-ops test of the Q4_1, Q4_K and Q6_K MUL_MAT cases"),
    ]
    # The GEMV rates: test-backend-ops perf base, new, then gemvcheck perf new, base (ABBA over the two tools)
    for v in ("base", "new"):
        r.append(Run(f"perf-{v}", "perf", v, "test-backend-ops", f"perf -o MUL_MAT -b HTP0 --test-file {phone}/tests/gemv-q8_0.txt",
                     "", 100, GATE_SMALL, "test-backend-ops perf of the 4B Q8_0 shapes, 1 and 4 rows, and the head"))
    for v in ("new", "base"):
        r.append(Run(f"gperf-{v}", "gperf", v, "gemvcheck", f"perf --cases {PERF_CASES} --reps 16 --runs 5", "", 100,
                     GATE_CHECK, "gemvcheck perf of MUL_MAT, MUL_MAT_ADD, MUL_MAT_NX and the head, 1 and 4 rows"))
    r += bench_runs()
    for v in ("base", "new"):
        r.append(Run(f"prof-{v}", "prof", v, "llama-bench", f"{BENCH_ARGS} -p 0 -n 8 -d 4096 -r 1",
                     "GGML_HEXAGON_PROFILE=1", 90, GATE_MODEL,
                     "llama-bench tg8 at the depth 4096 with the op profile"))
    return r


def test_files() -> dict[str, list[str]]:
    """The test files of test-backend-ops: file name -> lines. The line format comes from tools/prof/gemm.py.
    O(cases)."""
    sys.path.insert(0, str(REPO / "tools" / "prof"))
    import gemm  # noqa: PLC0415  (tools/prof/gemm.py: the line format of --test-file)
    return {
        "gemv-q8_0.txt": [gemm.case(k, m, n, "q8_0") for k, m, _ in SHAPES_4B for n in (1, 4)]
                         + [gemm.case(HEAD_4B[0], HEAD_4B[1], 1, "q8_0")],
        "kern-q8_0.txt": [gemm.case(k, m, n, "q8_0") for k, m, _ in SHAPES_4B for n in (1, 2, 3, 4)]
                         + [gemm.case(HEAD_4B[0], HEAD_4B[1], n, "q8_0") for n in (1, 4)]
                         + [gemm.case(2080, 200, n, "q8_0") for n in (1, 2, 3, 4)],
    }


def run_lines(run: Run, lay: Layout) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line. A run line of the model
    holds the text models/Qwen3.5 (a test of the model file), thus the runner applies to it the unlock wait,
    the screen wake, the memory gate and the CAPS line of a model run."""
    p = lay.phone
    stem = f"{p}/out/{run.name}"
    gate = f"{stem}-gate.txt"
    env = " ".join(x for x in (lib_env(lay), f"ADSP_LIBRARY_PATH={p}/dsp/{run.variant}", run.env) if x)
    cmd = (f"test -r {MODEL_DIR}/{MODEL} && sh {p}/bin/gate.sh {run.gate_kb} > {gate} && {BEFORE} >> {gate} && "
           f"timeout -s KILL {run.limit} env {env} {p}/bin/{run.tool} {run.args} > {stem}.out 2> {stem}.log; "
           f"echo \"rc=$?\" >> {gate}; {AFTER} >> {gate}; cat {gate}")
    return ["#", f"# {run.name}: {run.text}, DSP {run.variant} (limit {run.limit} s)", THERMAL, f"{ADB} shell '{cmd}'",
            PGREP]


HEADER = """\
# Phone stage "gemv": the packed Q8_0 tiles of the HVX GEMV (one DMA row for each 32-row column tile), the new DSP
# library against the base library of HEAD.
# The question: does the new library give the same bits, and does the 4B decode read its weights faster?
#
# The runs (tools/stages/gemv/stage.py gives each one):
#   check   gemvcheck check (tools/gemv/gemvcheck.cpp): the hash of each output of the 4B decode matmuls, base and
#           new, with the v79 library, and without the head with the v75 and the v73 library on the v79 DSP
#   tbo     test-backend-ops test with the new library: the 4B Q8_0 shapes (with the kernel names), the Q8_0
#           MUL_MAT and MUL_MAT_ID cases, and the MUL_MAT cases of Q4_1, Q4_K and Q6_K
#   perf    test-backend-ops perf of the 4B Q8_0 shapes, base then new
#   gperf   gemvcheck perf (MUL_MAT, MUL_MAT_ADD, MUL_MAT_NX, the head), new then base
#   bench   llama-bench of the 4B, the flags of the variant b of bench-kv: pp512 + tg32 at d0, tg32 at d4096, tg32 at
#           d16384; 3 rounds, base then new in rounds 1 and 3, new then base in round 2
#   prof    llama-bench tg8 at d4096 with GGML_HEXAGON_PROFILE=1, base and new
# The libraries: build/gemv/phone (tools/stages/gemv/build.sh): the host libraries of HEAD plus the patch, and the
# DSP libraries dsp/new (HEAD plus the patch) and dsp/base (HEAD). patch.sha256 names the patch.
#
# Each run: the thermal line, then the test of the model file (the text models/Qwen3.5 makes the runner treat the
# line as a model run: the unlock wait, the screen wake, the memory gate and the CAPS line), bin/gate.sh (the Qwen
# app stopped, the screen on, thermal 0, no charger, MemAvailable), the tool under timeout -s KILL (110 s or
# less), the exit code and the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 19 minutes of tool time and about
# 7 minutes of gates and checks (34 runs), plus the waits for thermal status 0.
# Then: python3 build/gemv/stage.py table (on the laptop or on the box).
"""

CHECK_HEADER = """\
# Phone stage "gemv-check": the bit check of the stage gemv. The packed Q8_0 tiles of the HVX GEMV (one DMA row for
# each 32-row column tile), the new DSP library against the base library of HEAD.
# The question: does the new library give the same bits as the base library for the 4B decode matmuls?
#
# The runs (tools/stages/gemv/stage.py --check-only gives each one):
#   check   gemvcheck check (tools/gemv/gemvcheck.cpp): the hash of each output of the 4B decode matmuls (MUL_MAT,
#           MUL_MAT_ADD, MUL_MAT_NX, the head, 1 to 4 rows, two shapes with an odd k-tile count), and the NMSE
#           against the CPU backend of the phone. v79: 40 cases. v75 and v73 on the v79 DSP (GGML_HEXAGON_ARCH):
#           the 36 cases without the head. gemvcheck puts the weights in a buffer with the usage
#           GGML_BACKEND_BUFFER_USAGE_WEIGHTS, thus the backend repacks each Q8_0 weight as the model loader has it.
# The result is satisfactory when each run has the exit code 0 (each output is finite, and each NMSE is 5e-4 or
# less), and for each library each output of new has the hash of base.
# The libraries: the files of build/gemv/phone (the stage gemv), thus the same bytes as the runs of that stage.
# patch.sha256 names the patch.
#
# Each run: the thermal line, then the test of the model file (the text models/Qwen3.5 makes the runner treat the
# line as a model run; no run loads a model), bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no
# charger, MemAvailable 4 GB), gemvcheck under timeout -s KILL (110 s or less), the exit code and the conditions
# after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: the gate stops a run on a locked phone or with a charger. It is
# not a timing stage: only the hashes and the NMSE give the result. Run from /home/grigory/airi/qwen-mobile on the
# laptop, in order. Time: at most 10 minutes of tool time (6 runs), plus the gates. The push is about 45 MB.
# Then: python3 build/gemv-check/stage.py table --check-only (on the laptop or on the box).
"""


def setup_lines(lay: Layout) -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    local, p = lay.stage, lay.phone
    bins = " ".join(f"{local}/phone/bin/{b}" for b in lay.bins)
    libs = " ".join(f"{local}/phone/lib/{lib}" for lib in lay.libs)
    tests = " ".join(f"{local}/phone/tests/{name}" for name in lay.tests)
    lines = [
        f"mkdir -p {local} && rsync -a --delete {lay.box}/phone/ {local}/phone/",
        f"(cd {local}/phone && sha256sum -c SHA256SUMS)",
        # No model path in this line: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODEL_DIR} | grep -E \"{MODEL}\"'",
        f"{ADB} shell 'rm -rf {p} && mkdir -p {p}/bin {p}/lib {p}/dsp/new {p}/dsp/base "
        + (f"{p}/tests " if lay.tests else "") + f"{p}/out'",
        f"{ADB} push {bins} {p}/bin/",
        f"{ADB} push {libs} {p}/lib/",
    ]
    for v in ("new", "base"):
        dsp = " ".join(f"{local}/phone/dsp/{v}/{lib}" for lib in DSP_LIBS)
        lines.append(f"{ADB} push {dsp} {p}/dsp/{v}/")
    if lay.tests:
        lines.append(f"{ADB} push {tests} {p}/tests/")
    lines += [
        f"{ADB} push {local}/phone/SHA256SUMS {p}/",
        f"{ADB} shell 'cd {p} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {p}/bin/*'",
    ]
    return lines


def output_lines(lay: Layout) -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone directory
    goes only when the pull has each of its files."""
    local, p = lay.stage, lay.phone
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'pgrep -x llama-bench; pgrep -x gemvcheck; ls {p}/out | wc -l; du -sh {p}/out'",
        f"rm -rf {local}/phone-out",
        f"{ADB} pull {p}/out {local}/phone-out",
        f"rsync -a --delete {local}/phone-out/ {lay.box}/phone-out/",
        f"test \"$(ls {local}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {p}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {p}' && echo removed {p}",
    ]


def write_commands(path: Path, lay: Layout) -> int:
    """Write the command file of the stage and return its line count."""
    header = CHECK_HEADER if lay == CHECK else HEADER
    lines = header.rstrip("\n").split("\n") + setup_lines(lay)
    for run in runs(lay):
        lines += run_lines(run, lay)
    lines += output_lines(lay)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The tables ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CHECK_RE = re.compile(r"^gemvcheck: check (\S+) out (\d+) hash (0x[0-9a-f]+) nmse (\S+)(.*)$", re.M)
GPERF_RE = re.compile(r"^gemvcheck: perf (\S+) reps (\d+) us ([\d.]+) min ([\d.]+) max ([\d.]+) gbps ([\d.]+)", re.M)
TBO_CASE_RE = re.compile(r"MUL_MAT\(type=\w+,ne=\[(?P<m>\d+),(?P<n>\d+),\d+,\d+\].*?sources=(?P<t>\w+)\[(?P<k>\d+),")
PERF_RE = re.compile(r"(?P<runs>\d+) runs -\s*(?P<us>[\d.]+) us/run")
PROF_RE = re.compile(r"profile-op (?P<op>[A-Z_0-9+]+)\|(?P<rest>.*?)\|usec (?P<usec>\d+) cycles")
GRAPH_START = "-> attn_norm-0|"


@dataclass
class Result:
    """The files of one run. ok is True when the gate passed and the tool ran to its end (exit code 0).
    flags names each condition that makes a timing value not comparable."""
    run: Run
    ok: bool
    rc: int | None
    flags: list[str]
    caps: str
    out: str
    log: str


def read_result(root: Path, run: Run) -> Result:
    """Read the files of one run. O(size of the files)."""
    def text(suffix: str) -> str:
        p = root / f"{run.name}{suffix}"
        return ANSI_RE.sub("", p.read_text(errors="replace")) if p.exists() else ""

    gate = text("-gate.txt")
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    m = re.search(r"^rc=(\d+)", gate, re.M)
    rc = int(m.group(1)) if m else None
    flags = []
    ok = "gate: OK" in gate and rc == 0
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
    return Result(run, ok, rc, flags, caps, text(".out"), text(".log"))


def med(vals: list[float]) -> float | None:
    """The median, or None for no value."""
    return statistics.median(vals) if vals else None


def fmt(x: float | None, nd: int = 2) -> str:
    """A number with nd decimals, or "-"."""
    return "-" if x is None else f"{x:.{nd}f}"


def pct(new: float | None, base: float | None) -> str:
    """The change of new against base in percent, or "-"."""
    return "-" if not new or not base else f"{100 * (new / base - 1):+.1f}%"


def conditions(results: dict[str, Result], lay: Layout) -> list[str]:
    """The conditions of the runs: the counts, the caps and each flag."""
    got = list(results.values())
    out = [f"conditions: {len(got)} of {len(runs(lay))} runs have a gate file, {sum(r.ok for r in got)} ran to the end "
           f"with exit code 0, {sum(r.ok and not r.flags for r in got)} have no flag"]
    caps = defaultdict(int)
    for r in got:
        if r.ok:
            caps[r.caps] += 1
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.items()))
    for r in got:
        if r.flags:
            first = next((ln for ln in r.log.splitlines() if ln.strip()), "") if r.rc not in (None, 0) else ""
            out.append(f"  {r.run.name}: " + ", ".join(r.flags) + (f" (stderr: {first[:120]})" if first else ""))
    return out


def check_table(results: dict[str, Result]) -> list[str]:
    """The bit check: for each library and each output, the hash of base against new, and the NMSE against the
    CPU. The timing conditions do not apply to the check runs. O(lines)."""
    out = ["Bit check (gemvcheck check): the hash of each output of base against new; nmse = the largest NMSE of the "
           "outputs against the CPU backend"]
    all_equal = True
    for arch in ("v79", "v75", "v73"):
        base, new = results.get(f"check-{arch}-base"), results.get(f"check-{arch}-new")
        if not base or not new:
            out.append(f"  {arch}: a run is missing")
            all_equal = False
            continue
        hb = {(m.group(1), m.group(2)): (m.group(3), float(m.group(4)), m.group(5).strip()) for m in CHECK_RE.finditer(base.out)}
        hn = {(m.group(1), m.group(2)): (m.group(3), float(m.group(4)), m.group(5).strip()) for m in CHECK_RE.finditer(new.out)}
        same = sum(1 for k in hn if k in hb and hb[k][0] == hn[k][0])
        diff = [k for k in hn if k in hb and hb[k][0] != hn[k][0]]
        missing = sorted(set(hb) ^ set(hn))
        worst_b = max((v[1] for v in hb.values()), default=None)
        worst_n = max((v[1] for v in hn.values()), default=None)
        fails = [k for k, v in hn.items() if v[2]] + [k for k, v in hb.items() if v[2]]
        all_equal = all_equal and not diff and not missing and bool(hn) and base.ok and new.ok
        out.append(f"  {arch}: {same} of {len(hn)} outputs have the same bits, {len(diff)} differ, {len(missing)} are in "
                   f"one run only; worst nmse base {fmt(worst_b, 8)} new {fmt(worst_n, 8)}; exit codes "
                   f"{base.rc}/{new.rc}" + (f"; FAIL or NONFINITE: {fails[:4]}" if fails else ""))
        for k in diff[:8]:
            out.append(f"    differs: {k[0]} out {k[1]} base {hb[k][0]} new {hn[k][0]}")
    out.append(f"  verdict: {'the new library gives the bits of the base library' if all_equal else 'NOT PROVEN'}")
    return out


def tbo_table(results: dict[str, Result]) -> list[str]:
    """The test-backend-ops test runs: the counts of OK, FAIL and not supported, each FAIL line, and the kernel
    of each case of the 4B run from its profile lines. O(lines)."""
    out = ["test-backend-ops test with the new library"]
    for name in ("tbo-4b-new", "tbo-q8-new", "tbo-id-new", "tbo-other-new"):
        r = results.get(name)
        if not r:
            out.append(f"  {name}: no result")
            continue
        lines = [ln for ln in r.out.splitlines() if "MUL_MAT" in ln]
        n_ok = sum(1 for ln in lines if ln.rstrip().endswith("OK"))
        n_fail = [ln.strip() for ln in lines if "FAIL" in ln]
        n_ns = sum(1 for ln in lines if "not supported" in ln)
        passed = re.search(r"(\d+)/(\d+) tests passed", r.out)
        out.append(f"  {name}: OK {n_ok}, FAIL {len(n_fail)}, not supported {n_ns}, "
                   f"{passed.group(0) if passed else 'no summary line'}, exit code {r.rc}")
        for ln in n_fail[:6]:
            out.append(f"    {ln[:160]}")
    r = results.get("tbo-4b-new")
    if r:
        kernels = defaultdict(set)
        for m in PROF_RE.finditer(r.log):
            f = m["rest"].split("|")
            if len(f) >= 5 and m["op"].startswith("MUL_MAT"):
                srcs = f[1].split(" -> ")[0].split(" x ")
                if len(srcs) >= 2:
                    n = srcs[1].split(":")[1]
                    kernels[(srcs[0], n)].add(f[4].split(" vtcm")[0].strip())
        out.append("  kernels of the 4B run (weight k:m, rows): " +
                   ", ".join(f"{k}/n{n} {'+'.join(sorted(v))}" for (k, n), v in sorted(kernels.items())))
    return out


def perf_tables(results: dict[str, Result], include_all: bool) -> list[str]:
    """The GEMV rates of base and new: test-backend-ops perf and gemvcheck perf. O(lines)."""
    out = ["GEMV rate (test-backend-ops perf): weight bytes / us/run, decimal GB/s; base and new, and the change",
           f"  {'k':>6s} {'m':>7s} {'n':>2s} {'base GB/s':>10s} {'new GB/s':>10s} {'change':>8s}  shape"]
    rates: dict[str, dict[tuple, float]] = {}
    for v in ("base", "new"):
        r = results.get(f"perf-{v}")
        rates[v] = {}
        if not r or not (r.ok and (include_all or not r.flags)):
            continue
        for line in r.out.splitlines():
            c, p = TBO_CASE_RE.search(line), PERF_RE.search(line)
            if c and p:
                k, m, n = int(c["k"]), int(c["m"]), int(c["n"])
                rates[v][(k, m, n)] = k * m * BPE["q8_0"] / float(p["us"]) / 1e3
    labels = {(k, m): lab for k, m, lab in SHAPES_4B + (HEAD_4B,)}
    for key in sorted(set(rates["base"]) | set(rates["new"]), key=lambda x: (x[0], -x[1], x[2])):
        b, n = rates["base"].get(key), rates["new"].get(key)
        out.append(f"  {key[0]:6d} {key[1]:7d} {key[2]:2d} {fmt(b):>10s} {fmt(n):>10s} {pct(n, b):>8s}  "
                   f"{labels.get((key[0], key[1]), '')}")
    out += ["", "GEMV rate (gemvcheck perf, 16 copies of the case in one graph, the median of 5 computes): GB/s",
            f"  {'case':32s} {'base GB/s':>10s} {'new GB/s':>10s} {'change':>8s} {'base us':>10s} {'new us':>10s}"]
    g: dict[str, dict[str, tuple[float, float]]] = {}
    for v in ("base", "new"):
        r = results.get(f"gperf-{v}")
        g[v] = {}
        if not r or not (r.ok and (include_all or not r.flags)):
            continue
        for m in GPERF_RE.finditer(r.out):
            g[v][m.group(1)] = (float(m.group(6)), float(m.group(3)))
    for case in sorted(set(g["base"]) | set(g["new"])):
        b, n = g["base"].get(case), g["new"].get(case)
        out.append(f"  {case:32s} {fmt(b[0] if b else None):>10s} {fmt(n[0] if n else None):>10s} "
                   f"{pct(n[0] if n else None, b[0] if b else None):>8s} {fmt(b[1] if b else None, 1):>10s} "
                   f"{fmt(n[1] if n else None, 1):>10s}")
    return out


BENCH_ROWS = (("pp512 d0", "t", (512, 0, 0)), ("tg32 d0", "t", (0, 32, 0)), ("tg32 d4096", "m", (0, 32, 4096)),
              ("tg32 d16384", "l", (0, 32, 16384)))


def bench_values(r: Result) -> dict[tuple[int, int, int], float]:
    """The t/s of each llama-bench test of one run: (n_prompt, n_gen, n_depth) -> the median of its samples."""
    vals = {}
    for line in r.out.splitlines():
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vals[(rec["n_prompt"], rec["n_gen"], rec["n_depth"])] = statistics.median(rec["samples_ts"])
    return vals


def bench_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The A/B of decode and prefill: per row the median over the rounds of base and new, the median of the paired
    ratios of one round, and the range. O(runs)."""
    out = ["Decode and prefill of the 4B (llama-bench, variant b of bench-kv): t/s, the median of the rounds [lowest-"
           "highest] n, and the change = the median of the ratios new/base of the runs of one round",
           f"  {'measurement':14s} {'base':>24s} {'new':>24s} {'change':>8s}"]
    for text, key, test in BENCH_ROWS:
        per = {"base": {}, "new": {}}
        for rnd in (1, 2, 3):
            for v in ("base", "new"):
                r = results.get(f"bench-{rnd}-{key}-{v}")
                if r and r.ok and (include_all or not r.flags):
                    val = bench_values(r).get(test)
                    if val is not None:
                        per[v][rnd] = val
        cells = []
        for v in ("base", "new"):
            vals = list(per[v].values())
            cells.append(f"{fmt(med(vals))} [{fmt(min(vals) if vals else None)}-{fmt(max(vals) if vals else None)}] "
                         f"n{len(vals)}")
        ratios = [per["new"][k] / per["base"][k] for k in per["new"] if k in per["base"]]
        change = f"{100 * (statistics.median(ratios) - 1):+.1f}%" if ratios else "-"
        out.append(f"  {text:14s} {cells[0]:>24s} {cells[1]:>24s} {change:>8s}")
    return out


def weight_mm(rest: str) -> tuple[float, str, int] | None:
    """For the fields after the op name of a profile line of a matmul with Q8_0 weights (MUL_MAT, MUL_MAT+ADD,
    MUL_MAT_NX): the weight bytes, the kernel and the activation rows. None for another op. The sources are the
    weights first, then the activation (and for MUL_MAT+ADD the added tensor)."""
    f = rest.split("|")
    if len(f) < 5:
        return None
    srcs = f[1].split(" -> ")[0].split(" x ")
    types = [t.strip() for t in f[2].split(" -> ")[0].split(" x ")]
    n_w = 0
    while n_w < len(types) and types[n_w] == "q8_0":
        n_w += 1
    if n_w == 0 or n_w >= len(srcs):
        return None
    wb = 0.0
    for s in srcs[:n_w]:
        k, m = (int(x) for x in s.split(":")[:2])
        wb += k * m * BPE["q8_0"]
    return wb, f[4].split(" vtcm")[0].strip(), int(srcs[n_w].split(":")[1])


def prof_table(results: dict[str, Result]) -> list[str]:
    """The decode profile of each library: for the decode graphs (the graphs after the depth prefill), the median
    op time, the time and the rate of the weight MUL_MATs, and their kernels. O(lines)."""
    out = ["Decode profile (llama-bench tg8 at d4096, GGML_HEXAGON_PROFILE=1): the median over the decode graphs"]
    for v in ("base", "new"):
        r = results.get(f"prof-{v}")
        if not r or not r.ok:
            out.append(f"  {v}: no result")
            continue
        graphs: list[list[tuple[str, str, int]]] = []
        cur: list[tuple[str, str, int]] | None = None
        for line in r.log.splitlines():
            m = PROF_RE.search(line)
            if not m or m["op"] == "OPBATCH":
                continue
            if cur is None or GRAPH_START in line:
                cur = []
                graphs.append(cur)
            cur.append((m["op"], m["rest"], int(m["usec"])))
        dec = [g for g in graphs if (rows := [weight_mm(rest)[2] for op, rest, _ in g if weight_mm(rest)])
               and all(n == 1 for n in rows)]
        tok_us, mm_us, mm_bytes, kern = [], [], [], defaultdict(int)
        for g in dec:
            tot = sum(us for _, _, us in g)
            wus = wb = 0
            for op, rest, us in g:
                w = weight_mm(rest)
                if not w:
                    continue
                wus += us
                wb += w[0]
                kern[w[1]] += 1
            tok_us.append(tot)
            mm_us.append(wus)
            mm_bytes.append(wb)
        if not dec:
            out.append(f"  {v}: no decode graph in the log")
            continue
        gbs = [b / u / 1e3 for b, u in zip(mm_bytes, mm_us) if u]
        out.append(f"  {v}: {len(dec)} decode graphs, op time {fmt(med(tok_us) / 1e3)} ms, Q8_0 weight MUL_MATs "
                   f"{fmt(med(mm_us) / 1e3)} ms at {fmt(med(gbs))} GB/s ({fmt(med(mm_bytes) / 1e9, 3)} GB); kernels "
                   + ", ".join(f"{k} x{n}" for k, n in sorted(kern.items())))
    return out


def table(root: Path, include_all: bool, lay: Layout) -> int:
    """Print the tables of the stage. The check stage has only the conditions and the bit check."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {run.name: read_result(root, run) for run in runs(lay) if (root / f"{run.name}-gate.txt").exists()}
    parts = [conditions(results, lay), check_table(results)]
    if lay == FULL:
        parts += [tbo_table(results), perf_tables(results, include_all), bench_table(results, include_all),
                  prof_table(results)]
    for part in parts:
        print("\n".join(part))
        print()
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tests", help="write the test files of test-backend-ops")
    t.add_argument("--out", type=Path, default=HERE / "phone" / "tests")
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=None, help="the default is phone-commands.txt of the stage directory")
    tb = sub.add_parser("table", help="print the tables from the pulled files")
    tb.add_argument("--root", type=Path, default=None, help="the default is phone-out of the stage directory")
    tb.add_argument("--all", action="store_true", help="also use the timing runs with changed caps or heat")
    lb = sub.add_parser("libs", help="print the host libraries of the stage, one on each line")
    for p in (c, tb, lb):
        p.add_argument("--check-only", action="store_true", help="the check stage gemv-check")
    a = ap.parse_args()
    lay = CHECK if getattr(a, "check_only", False) else FULL
    if a.cmd == "libs":
        print("\n".join(lay.libs))
        return 0
    if a.cmd == "tests":
        a.out.mkdir(parents=True, exist_ok=True)
        files = test_files()
        if sorted(files) != sorted(FULL.tests):
            print(f"stage.py: FULL.tests does not list the files of test_files(): {sorted(files)}", file=sys.stderr)
            return 1
        for name, lines in files.items():
            (a.out / name).write_text("\n".join(lines) + "\n")
        print(f"{a.out}: {len(files)} test files")
        return 0
    if a.cmd == "commands":
        out = a.out or lay.here / "phone-commands.txt"
        n = write_commands(out, lay)
        print(f"{out}: {n} lines, {len(runs(lay))} runs")
        return 0
    return table(a.root or lay.here / "phone-out", a.all, lay)


if __name__ == "__main__":
    sys.exit(main())
