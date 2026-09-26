#!/usr/bin/env python3
"""The phone stage v81 (the v79 phone) and the v81 silicon kit (a phone with the Hexagon v81 DSP).

Usage:
    stage.py commands [--set phone|kit|mini] [--out PATH]   write the command file of a run set
    stage.py table [--root DIR]                        print the results of the pulled outputs

The run sets:
    phone  The v79 phone (the OnePlus 13) with the libraries of tools/stages/v81/build.sh (build/v81/phone): the
           start-up self-test (canary) of the Hexagon backend, its load time with GGML_HEXAGON_CANARY 0 and 1 in
           alternated rounds, the path with no device (GGML_HEXAGON_CANARY=2), the op tests of the Qwen3.5 ops on
           HTP0, the 4B KL against the naive x86 oracle, and one bench. The command file is
           build/v81/phone-commands.txt, the outputs go to build/v81/phone-out.
    kit    A v81 phone of a device farm (build/v81/kit): the ISA probe (the chip, the library, the census of each
           HVX op against the ARM CPU oracle and the simulator), the canary, the same op tests, the 4B KL against
           the naive x86 oracle, and llama-bench pp512 and tg32. The phone always charges there, thus the gate has
           ALLOW_CHARGER=1. adb has no -s: the session gives one device. The command file is
           build/v81/kit-commands.txt, the outputs go to build/v81/kit-out.
    mini   The v79 phone, a check of the canary on the libraries of HEAD before an APK build: two runs of the canary
           (GGML_HEXAGON_CANARY=1) and test-backend-ops MUL_MAT. The command file is build/v81/mini-commands.txt,
           the outputs go to build/v81/mini-out. Time: less than 3 minutes.

Each run writes three files to the out/ directory of the phone: <name>-gate.txt (the conditions before and after
the run and the exit code), <name>.out (the stdout of the tool) and <name>.log (its stderr). The table only reads
files. O(size of the logs) time.
"""

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import commands, device, gate, logs  # noqa: E402

PATHS = device.stage_paths("v81", __file__)
LAPTOP_STAGE, BOX, STAGE_DIR = PATHS.local, PATHS.box, PATHS.dir
# The kernel keeps 15 characters of a process name, and pgrep -x matches that name
TOOLS = ("llama-bench", "llama-perplexit", "test-backend-op", "canarytime", "isaprobe")


@dataclass(frozen=True)
class Target:
    """The phone of a run set: the adb command, the phone directory, the model and eval paths, the gate."""
    adb: str
    phone: str
    model: str
    eval_dir: str
    base: str
    gate_env: str
    stage: str      # the directory of the stage files in build/v81 (build.sh)
    out_dir: str    # the directory of the pulled outputs in build/v81
    commands: str   # the command file in build/v81

    @property
    def thermal(self) -> str:
        """The line that prints the thermal status. device.THERMAL gives this line for the phone of the
        project, and the kit target has its own adb command."""
        return f"{self.adb} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"

    @property
    def pgrep(self) -> str:
        """The line that prints the process id of each tool of the stage that still runs. device.pgrep gives
        this line for the phone of the project, and the kit target has its own adb command."""
        return f"{self.adb} shell '" + "; ".join(f"pgrep -x {x}" for x in TOOLS) + "; echo pgrep-done'"


PHONE = Target(adb=device.ADB, phone=PATHS.phone, model=f"{device.MODEL_DIR}/{device.MODEL_4B}",
               eval_dir=device.EVAL_DIR, base="naive-4B-q8.kld", gate_env="", stage="phone",
               out_dir="phone-out", commands="phone-commands.txt")
MINI = Target(adb=PHONE.adb, phone=PHONE.phone, model=PHONE.model, eval_dir=PHONE.eval_dir, base=PHONE.base,
              gate_env="", stage="phone", out_dir="mini-out", commands="mini-commands.txt")
KIT = Target(adb="adb", phone="/data/local/tmp/qwen/v81kit", model="/data/local/tmp/qwen/v81kit/models/Qwen3.5-4B-Q8_0.gguf",
             eval_dir="/data/local/tmp/qwen/v81kit/eval", base="naive-4B-q8-c1.kld", gate_env="ALLOW_CHARGER=1",
             stage="kit", out_dir="kit-out", commands="kit-commands.txt")


@dataclass(frozen=True)
class Run:
    """One phone run: the name (the stem of its files), the tool, its arguments, the extra environment, the
    time limit in seconds, the MemAvailable of the gate in kB, the text and the DSP library directory (lib: the
    libraries of the stage, dsp-base: the DSP libraries of HEAD)."""
    name: str
    tool: str
    args: str
    env: str
    limit: int
    gate_kb: int
    text: str
    dsp: str = "lib"


def lib_env(t: Target, dsp: str = "lib") -> str:
    """The environment of the app (init_impl in llama_jni.cpp) and the stage libraries. The host loads the
    libraries of lib, and the DSP loads those of `dsp`, thus device.lib_env does not give this text."""
    return f"LD_LIBRARY_PATH={t.phone}/lib ADSP_LIBRARY_PATH={t.phone}/{dsp} {device.APP_ENV}"


def mini_runs() -> list[Run]:
    """The runs of the set mini: two canary runs and the MUL_MAT op tests."""
    out = [Run(f"can-{i}-c1", "canarytime", "--rounds 3", "GGML_HEXAGON_CANARY=1", 60, device.GATE_TOOL_KB,
               "canarytime, GGML_HEXAGON_CANARY=1: the canary result, the register and open times, 3 work graphs")
           for i in (1, 2)]
    out.append(Run("tbo-mm", "test-backend-ops", "test -b HTP0 -o MUL_MAT", "", 110, device.GATE_TOOL_KB,
                   "test-backend-ops MUL_MAT on HTP0 against the CPU (the Q8_0 GEMV, the HMX GEMM)"))
    return out


def runs(t: Target, kit: bool) -> list[Run]:
    """The runs of a run set, in their order."""
    if t is MINI:
        return mini_runs()
    kl = (f"-m {t.model} -dev HTP0 -ngl 99 -t 4 -fa on -ctk q8_0 -ctv q8_0 -f {t.eval_dir}/wiki.test.raw -c 512 "
          f"--chunks 1 --kl-divergence-base {t.eval_dir}/{t.base} --kl-divergence")
    bench = f"-m {t.model} -dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
    out = []
    # The canary: its load time in alternated rounds, then the path with no device
    for i, c in enumerate(("0", "1", "1", "0", "0", "1")):
        out.append(Run(f"can-{i + 1}-c{c}", "canarytime", "--rounds 3", f"GGML_HEXAGON_CANARY={c}", 60,
                       device.GATE_TOOL_KB,
                       f"canarytime, GGML_HEXAGON_CANARY={c}: the register and open times, 3 work graphs"))
    out.append(Run("can-fail", "canarytime", "--rounds 0", "GGML_HEXAGON_CANARY=2", 60, device.GATE_TOOL_KB,
                   "canarytime, GGML_HEXAGON_CANARY=2: the backend must register no device (exit 3)"))
    ops = (("tbo-mm", "MUL_MAT", "MUL_MAT (the Q8_0 GEMV and the HMX GEMM, the f16 HMX path)"),
           ("tbo-gdn", "GATED_DELTA_NET,SSM_CONV", "GATED_DELTA_NET and SSM_CONV (the state step, the chunked GDN, "
            "the conv)"),
           ("tbo-mov", "CPY,SET_ROWS,GET_ROWS,CONT", "CPY, SET_ROWS, GET_ROWS and CONT (the f32 and f16 moves)"),
           ("tbo-elt", "RMS_NORM,SOFT_MAX,ROPE,SWIGLU,SILU,SIGMOID,SOFTPLUS,GELU,ADD,MUL,SCALE",
            "RMS_NORM, SOFT_MAX, ROPE, SWIGLU, SILU, SIGMOID, SOFTPLUS, GELU (vision), ADD, MUL and SCALE"),
           ("tbo-fa", "FLASH_ATTN_EXT", "FLASH_ATTN_EXT (the cases with sinks fail at the tolerance edge on v79 too)"))
    for name, o, text in ops:
        out.append(Run(name, "test-backend-ops", f"test -b HTP0 -o {o}", "", 110, device.GATE_TOOL_KB,
                       f"test-backend-ops {text} on HTP0 against the CPU"))
    out.append(Run("klp", "llama-perplexity", f"{kl} -b 512", "", 90, device.GATE_4B_KB,
                   f"llama-perplexity KL against {t.base}, 1 chunk of 512, -b 512 (the HMX prefill path)"))
    out.append(Run("kld", "llama-perplexity", f"{kl} -b 1 -ub 1", "", 108, device.GATE_4B_KB,
                   f"llama-perplexity KL against {t.base}, 1 chunk of 512, -b 1 (the HVX decode path)"))
    if kit:
        out.append(Run("bench", "llama-bench", f"{bench} -p 512 -n 32 -r 3", "", 100, device.GATE_4B_KB,
                       "llama-bench pp512 and tg32 at the depth 0, -r 3"))
        return out
    # The A/B timing of the DSP libraries: new (lib) against HEAD (dsp-base), 3 rounds, the order alternated
    for rnd in range(1, 4):
        order = ("new", "base") if rnd % 2 else ("base", "new")
        for v in order:
            out.append(Run(f"p-{rnd}-{v}", "llama-bench", f"{bench} -p 512 -n 0 -r 3", "", 100, device.GATE_4B_KB,
                           "llama-bench pp512 at the depth 0, -r 3", "lib" if v == "new" else "dsp-base"))
        for v in order[::-1]:
            out.append(Run(f"t-{rnd}-{v}", "llama-bench", f"{bench} -p 0 -n 32 -d 0,4096 -r 3", "", 100,
                           device.GATE_4B_KB, "llama-bench tg32 at the depths 0 and 4096, -r 3",
                           "lib" if v == "new" else "dsp-base"))
    return out


def run_lines(t: Target, r: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    env = " ".join(x for x in (lib_env(t, r.dsp), r.env) if x)
    cmd = commands.gated_run(f"{t.phone}/out/{r.name}", r.gate_kb, r.limit, env,
                             f"{t.phone}/bin/{r.tool} {r.args}", stage=t.phone,
                             prefix=f"{t.gate_env} " if t.gate_env else "")
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if r.tool in ("llama-bench", "llama-perplexity") else "OP-TEST"
    return commands.run_lines(f"# {title}: {r.name}, {r.text}", cmd, t.pgrep, adb=t.adb, thermal=t.thermal)


def setup_lines(t: Target, kit: bool) -> list[str]:
    """The lines that copy the stage files to the phone and check them.

    Each run set has its own groups of files, its own order and its own check line. Thus
    commands.setup_lines does not give the same text."""
    local = f"{LAPTOP_STAGE}/{t.stage}"
    lines = [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/{t.stage}/ {local}/",
        f"(cd {local} && sha256sum -c SHA256SUMS)",
        f"{t.adb} shell 'rm -rf {t.phone} && mkdir -p {t.phone}/bin {t.phone}/lib {t.phone}/out'",
        f"{t.adb} push {local}/bin/* {t.phone}/bin/",
        f"{t.adb} push {local}/lib/* {t.phone}/lib/",
        f"{t.adb} push {local}/SHA256SUMS {t.phone}/",
    ]
    if kit:
        lines += [
            f"{t.adb} shell 'mkdir -p {t.phone}/isaprobe/v81 {t.phone}/corpus {t.phone}/models {t.phone}/eval'",
            f"{t.adb} push {local}/isaprobe/v81/libisaprobe_skel.so {t.phone}/isaprobe/v81/",
            f"{t.adb} push {local}/corpus/ {t.phone}/",
            "# The model (4.7 GB) and the KL base of the naive x86 oracle (1 chunk). A farm with slow uploads can "
            "download them on the phone instead.",
            f"rsync -a {BOX}/kit-data/ {LAPTOP_STAGE}/kit-data/",
            f"(cd {LAPTOP_STAGE}/kit-data && sha256sum -c SHA256SUMS)",
            f"{t.adb} push {LAPTOP_STAGE}/kit-data/Qwen3.5-4B-Q8_0.gguf {t.phone}/models/",
            f"{t.adb} push {LAPTOP_STAGE}/kit-data/{t.base} {LAPTOP_STAGE}/kit-data/wiki.test.raw {t.eval_dir}/",
        ]
    elif t is MINI:
        # The mini set pushes bin and lib only, thus its check skips the dsp-base lines of SHA256SUMS
        lines.append(f"{t.adb} shell 'cd {t.phone} && grep -v \" dsp-base/\" SHA256SUMS | sha256sum -c - > /dev/null "
                     f"&& echo \"stage files: all present\"; chmod 755 {t.phone}/bin/*'")
        return lines
    else:
        lines.insert(3, f"{t.adb} shell 'mkdir -p {t.phone}/dsp-base'")
        lines.insert(6, f"{t.adb} push {local}/dsp-base/* {t.phone}/dsp-base/")
        # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
        lines.append(f"{t.adb} shell 'ls -l /data/local/tmp/qwen/models | grep -E \"Qwen3.5-4B-Q8_0.gguf\"; "
                     f"ls -l {t.eval_dir}/wiki.test.raw {t.eval_dir}/{t.base}'")
    lines.append(f"{t.adb} shell 'cd {t.phone} && sha256sum -c SHA256SUMS | grep -c OK && sha256sum -c SHA256SUMS "
                 f"> /dev/null && echo \"stage files: all present\"; chmod 755 {t.phone}/bin/*'")
    return lines


def kit_probe_lines(t: Target) -> list[str]:
    """The ISA probe on the v81 phone: the chip facts, then the census of each HVX op with the ARM CPU oracle."""
    env = f"ADSP_LIBRARY_PATH={t.phone}/isaprobe/v81 LD_LIBRARY_PATH={t.phone}/isaprobe/v81"
    return [
        "#",
        "# ---- The chip: the SoC, the build, then the ISA probe (the DSP version, the HVX and HMX units, the VTCM) ----",
        f"{t.adb} shell 'getprop ro.soc.model; getprop ro.soc.manufacturer; getprop ro.board.platform; "
        "getprop ro.build.type; getprop ro.build.fingerprint'",
        f"{t.adb} shell 'cd {t.phone} && mkdir -p out/probe-info && {env} timeout -s KILL 100 ./bin/isaprobe info "
        "> out/probe-info/stdout.txt 2>&1; echo \"exit $?\" >> out/probe-info/stdout.txt; cat out/probe-info/stdout.txt'",
        "# The census of each HVX op on the chip (about 225 MB of outputs). exit 0 is success, 3 the version gate.",
        f"{t.adb} shell 'cd {t.phone} && mkdir -p out/probe-census && {env} timeout -s KILL 100 ./bin/isaprobe census "
        "corpus out/probe-census > out/probe-census/stdout.txt 2>&1; echo \"exit $?\" >> out/probe-census/stdout.txt; "
        "tail -n 20 out/probe-census/stdout.txt'",
    ]


def output_lines(t: Target, kit: bool, rs: list[Run]) -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory.

    This block has no thermal line, thus commands.output_lines does not give the same text. The phone
    directory goes only when the laptop has each file of the pull and each file of each run of rs, and the
    check names each missing file (commands.pull_check)."""
    local = f"{LAPTOP_STAGE}/{t.out_dir}"
    names = [r.name for r in rs]
    probe = ("probe-info/stdout.txt", "probe-census/stdout.txt") if kit else ()
    lines = [
        "#",
        "# ---- The outputs ----",
        "#",
        f"{t.adb} shell '" + "; ".join(f"pgrep -x {x}" for x in TOOLS) + f"; ls {t.phone}/out | wc -l; "
        f"du -sh {t.phone}/out'",
        f"rm -rf {local}",
        f"{t.adb} pull {t.phone}/out {local}",
        f"rsync -a --delete {local}/ {BOX}/{t.out_dir}/",
        commands.pull_check(local, t.phone, names, adb=t.adb, expected=probe),
        "# Then, on the box:",
        f"#   tools/stages/v81/stage.py table --root build/v81/{t.out_dir}",
    ]
    if kit:
        lines += [
            "#   python3 tools/htp-lab/probe/oracle_compare.py census build/v81/kit-out/probe-census "
            "--csv build/v81/kit-out/probe-census/oracle.csv",
            "#   cp tools/htp-lab/out-isa/isa-c/corpus_*.bin build/v81/kit-out/probe-census/ && ln -sfn "
            "$PWD/build/v81/kit-out/probe-census tools/htp-lab/out-isa/isa-chip81-c && "
            "python3 tools/htp-lab/isa/compare.py --tag c --archs v81 chip81",
        ]
    return lines


HEADER = """\
# {title}
#
# The libraries: tools/stages/v81/build.sh (build/v81/{stage}): the patched llama.cpp tree of HEAD plus the
# patches of build/v81/patches (build/v81/{stage}/patches.sha256 lists them). The series has the three changes
# for v81: v81 uses the f32 to f16 conversion of v79 (patches/hexagon-arch/0003), the start-up self-test of the
# DSP, the canary (patches/hexagon-arch/0004), and the stride of the transpose tile copy
# (patches/hexagon-kernels/0006).{base_text}
#
# The runs and the decision of each:
{runs}
#
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, {charger},
# MemAvailable, and it prints the caps), the tool under timeout -s KILL (110 s or less), the exit code and the
# conditions after the run, then the pgrep line.
#
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about {minutes} minutes of tools plus
# about 8 s of gate and checks for each of the {n_runs} runs, plus the waits for thermal status 0.
"""

DECISIONS = {
    "can": ("canarytime GGML_HEXAGON_CANARY=0 and 1, 3 rounds each: the log must have \"canary pass\" with 1; the "
            "load time (register + open) of 1 minus that of 0 must be less than 50 ms; each work graph must pass"),
    "can-fail": "canarytime GGML_HEXAGON_CANARY=2: exit 3, no device HTP0, and the log line \"canary FAIL\"",
    "tbo": ("test-backend-ops: each case must pass (FLASH_ATTN_EXT: the FAIL count and family of HEAD, the sinks "
            "cases). tbo-mov has the four new CONT cases of a transposed 32-multiple matrix (patch 0003)"),
    "klp": "llama-perplexity -b 512: the KL of HEAD on this phone (mean near 0.00045 on the chunk 1)",
    "kld": "llama-perplexity -b 1: the KL of HEAD on this phone (mean near 0.0005 on the chunk 1)",
    "p, t": ("llama-bench pp512 at d0 and tg32 at d0 and d4096, the DSP library of the stage (new) against HEAD "
             "(base), 3 alternated rounds: new must be equal to base within the spread (the Qwen graph does not "
             "send the shape of patch 0003)"),
}


KIT_DECISIONS = {
    "probe": ("isaprobe info: the DSP version 0x..81, the HVX contexts, the HMX count and the VTCM size. isaprobe census: "
              "compare.py must give the bits of the v81na_2 simulator for each qf op and each conversion to IEEE"),
    "can": ("canarytime GGML_HEXAGON_CANARY=1: the log must have \"canary pass\"; a \"canary FAIL\" line names the first "
            "value outside its bound, and the backend then registers no HTP device (the app uses the GPU)"),
    "can-fail": "canarytime GGML_HEXAGON_CANARY=2: exit 3 and no device HTP0",
    "tbo": "test-backend-ops: each case must pass (FLASH_ATTN_EXT: the FAIL count and family of the v79 phone)",
    "klp": ("llama-perplexity -b 512, KL against naive-4B-q8-c1.kld (1 chunk of the naive x86 oracle): the class of v79 "
            "(mean 0.00045 on the chunk 1, top 1 near 99 %). A mean 10 times larger is a systematic error"),
    "kld": "llama-perplexity -b 1: the class of v79 (mean near 0.0005 on the chunk 1)",
    "bench": "llama-bench pp512 and tg32 (the phone charges, thus the numbers are not those of a phone on its battery)",
}


MINI_DECISIONS = {
    "can-1, can-2": ("canarytime GGML_HEXAGON_CANARY=1: the log line \"canary pass\" and exit 0 in each run. A line "
                     "\"canary FAIL\" names the first value outside its bound: then do not build the APK"),
    "tbo-mm": "test-backend-ops MUL_MAT: each case must pass (673 of 673 on the libraries before hexagon-mm/0012)",
}


def write_commands(set_name: str, path: Path) -> None:
    """Write the command file of one run set."""
    kit = set_name == "kit"
    mini = set_name == "mini"
    t = KIT if kit else (MINI if mini else PHONE)
    rs = runs(t, kit)
    if mini:
        title = ("Phone stage \"v81\", run set mini, on the v79 phone: the canary on the libraries of HEAD before an "
                 "APK build.")
        decisions = MINI_DECISIONS
    elif kit:
        title = ("The v81 silicon kit: a phone with the Hexagon v81 DSP (Snapdragon 8 Elite Gen 5, SM8850) of a "
                 "device farm.")
        decisions = KIT_DECISIONS
    else:
        title = ("Phone stage \"v81\" on the v79 phone: the canary, its load time, the op tests, the 4B KL and the "
                 "A/B timing of the DSP library.")
        decisions = DECISIONS
    base_text = "" if (kit or mini) else ("\n# dsp-base holds the DSP libraries of HEAD (build/v81/dsp-compare.txt "
                                          "gives the hashes of the two sets).")
    text = HEADER.format(title=title, stage=t.stage, base_text=base_text,
                         runs="\n".join(f"#   {k}: {v}" for k, v in decisions.items()),
                         charger="a charger permitted (ALLOW_CHARGER=1)" if kit else "no charger",
                         minutes=20 if kit else (2 if mini else 18), n_runs=len(rs))
    lines = commands.header_lines(text) + ["#", "# ---- Setup ----"] + setup_lines(t, kit)
    if kit:
        lines += kit_probe_lines(t)
    lines += ["#", "# ---- The runs ----"]
    for r in rs:
        lines += run_lines(t, r)
    lines += output_lines(t, kit, rs)
    n = commands.write_commands(path, lines)
    print(f"{path}: {n} lines, {len(rs)} runs")


def bench_rows(path: Path) -> list[tuple[str, float, float]]:
    """The (key, t/s, stddev) of each result line of a llama-bench jsonl output."""
    rows = []
    for ln in logs.read_text(path).splitlines():
        try:
            j = json.loads(ln)
        except json.JSONDecodeError:
            continue
        key = f"pp{j.get('n_prompt')}" if j.get("n_gen", 0) == 0 else f"tg{j.get('n_gen')}"
        rows.append((f"{key}@d{j.get('n_depth', 0)}", j.get("avg_ts", 0.0), j.get("stddev_ts", 0.0)))
    return rows


def table(root: Path) -> None:
    """Print the results of the pulled outputs."""
    print(f"# {root}")
    load: dict[str, list[float]] = {"0": [], "1": []}
    for gate_file in sorted(root.glob("can-*-gate.txt")):
        name = gate_file.name[:-len("-gate.txt")]
        out = logs.read_text(root / f"{name}.out")
        log = logs.read_text(root / f"{name}.log")
        rc = gate.read(logs.read_text(gate_file)).rc
        m = re.search(r"load: ([0-9.]+) ms", out)
        canary = [ln for ln in log.splitlines() if "canary" in ln]
        works = re.findall(r"work \d+: .* (pass|FAIL)", out)
        print(f"{name}: rc {rc if rc is not None else '?'}, load {m.group(1) if m else '-'} ms, work {works}")
        for ln in canary:
            print(f"    {ln.strip()[:400]}")
        c = re.search(r"-c(\d)$", name)
        if c and m:
            load[c.group(1)].append(float(m.group(1)))
    if load["0"] and load["1"]:
        d = statistics.median(load["1"]) - statistics.median(load["0"])
        print(f"canary load cost: median load {statistics.median(load['1']):.1f} ms (1) against "
              f"{statistics.median(load['0']):.1f} ms (0), difference {d:.1f} ms (limit 50 ms)")
    for out in sorted(root.glob("tbo-*.out")):
        txt = logs.read_text(out)
        m = re.findall(r"(\d+)/(\d+) tests passed", txt)
        fails = [ln.strip() for ln in txt.splitlines() if "FAIL" in ln][:6]
        print(f"{out.stem}: {m[-1][0] + '/' + m[-1][1] if m else 'no summary'} passed; " + "; ".join(fails)[:600])
    for name in ("klp", "kld"):
        txt = logs.read_text(root / f"{name}.out") + logs.read_text(root / f"{name}.log")
        vals = {}
        for key, rx in (("mean", r"Mean\s+KLD:\s+([0-9.eE+-]+)"), ("max", r"Maximum KLD:\s+([0-9.eE+-]+)"),
                        ("top1", r"Same top p:\s+([0-9.]+)")):
            m = re.search(rx, txt)
            vals[key] = m.group(1) if m else "-"
        print(f"{name}: KL mean {vals['mean']}, max {vals['max']}, same top 1 {vals['top1']} %")
    for key, ts, sd in bench_rows(root / "bench.out"):
        print(f"bench: {key}: {ts:.2f} +- {sd:.2f} t/s")
    # The A/B timing: a run counts when its gate passed, its exit code is 0 and the thermal status after it is 0
    ab: dict[str, dict[str, dict[str, float]]] = {}
    for out in sorted(root.glob("[pt]-*-*.out")):
        name = out.stem
        text = logs.read_text(root / f"{name}-gate.txt")
        after = gate.AFTER_RE.search(text)
        ok = gate.read(text).ok and after is not None and after.group(1) == "0"
        m = re.match(r"([pt])-(\d+)-(new|base)$", name)
        if not m or not ok:
            print(f"{name}: not counted (gate, exit code or thermal status)")
            continue
        for key, ts, _sd in bench_rows(out):
            ab.setdefault(key, {}).setdefault(m.group(2), {})[m.group(3)] = ts
    for key, rounds in sorted(ab.items()):
        ratios = [r["new"] / r["base"] for r in rounds.values() if "new" in r and "base" in r and r["base"] > 0]
        news = [r["new"] for r in rounds.values() if "new" in r]
        bases = [r["base"] for r in rounds.values() if "base" in r]
        if news and bases:
            print(f"A/B {key}: new {statistics.median(news):.2f} t/s, base {statistics.median(bases):.2f} t/s, "
                  f"median ratio of the rounds {statistics.median(ratios) if ratios else float('nan'):.4f} "
                  f"({len(ratios)} rounds)")
    probe = logs.read_text(root / "probe-info" / "stdout.txt")
    if probe:
        print("probe info:\n" + "\n".join("    " + ln for ln in probe.splitlines()[:40]))


def main(argv: list[str]) -> int:
    """Parse the arguments and run the command."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands")
    c.add_argument("--set", default="phone", choices=("phone", "kit", "mini"))
    c.add_argument("--out", type=Path)
    t = sub.add_parser("table")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    args = ap.parse_args(argv)
    if args.cmd == "commands":
        target = {"kit": KIT, "mini": MINI}.get(args.set, PHONE)
        write_commands(args.set, args.out or STAGE_DIR / target.commands)
        return 0
    if not args.root.is_dir():
        print(f"error: {args.root} does not exist. Pull the outputs first (the end of the command file).",
              file=sys.stderr)
        return 1
    table(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
