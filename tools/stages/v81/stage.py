#!/usr/bin/env python3
"""The phone stage v81 (the v79 phone) and the v81 silicon kit (a phone with the Hexagon v81 DSP).

Usage:
    stage.py commands [--set phone|kit] [--out PATH]   write the command file of a run set
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

Each run writes three files to the out/ directory of the phone: <name>-gate.txt (the conditions before and after
the run and the exit code), <name>.out (the stdout of the tool) and <name>.log (its stderr). The table only reads
files. O(size of the logs) time.
"""

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

LAPTOP_STAGE = "build/v81"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/v81"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')
TOOLS = ("llama-bench", "llama-perplexity", "test-backend-ops", "canarytime", "isaprobe")


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


PHONE = Target(adb="adb -s 192.168.14.130:5555", phone="/data/local/tmp/qwen/v81",
               model="/data/local/tmp/qwen/models/Qwen3.5-4B-Q8_0.gguf", eval_dir="/data/local/tmp/qwen/eval",
               base="naive-4B-q8.kld", gate_env="", stage="phone", out_dir="phone-out",
               commands="phone-commands.txt")
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
    """The environment of the app (init_impl in llama_jni.cpp) and the stage libraries."""
    return (f"LD_LIBRARY_PATH={t.phone}/lib ADSP_LIBRARY_PATH={t.phone}/{dsp} "
            "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1")


def runs(t: Target, kit: bool) -> list:
    """The runs of a run set, in their order."""
    kl = (f"-m {t.model} -dev HTP0 -ngl 99 -t 4 -fa on -ctk q8_0 -ctv q8_0 -f {t.eval_dir}/wiki.test.raw -c 512 "
          f"--chunks 1 --kl-divergence-base {t.eval_dir}/{t.base} --kl-divergence")
    bench = f"-m {t.model} -dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
    out = []
    # The canary: its load time in alternated rounds, then the path with no device
    for i, c in enumerate(("0", "1", "1", "0", "0", "1")):
        out.append(Run(f"can-{i + 1}-c{c}", "canarytime", "--rounds 3", f"GGML_HEXAGON_CANARY={c}", 60, 2097152,
                       f"canarytime, GGML_HEXAGON_CANARY={c}: the register and open times, 3 work graphs"))
    out.append(Run("can-fail", "canarytime", "--rounds 0", "GGML_HEXAGON_CANARY=2", 60, 2097152,
                   "canarytime, GGML_HEXAGON_CANARY=2: the backend must register no device (exit 3)"))
    ops = (("tbo-mm", "MUL_MAT", "MUL_MAT (the Q8_0 GEMV and the HMX GEMM, the f16 HMX path)"),
           ("tbo-gdn", "GATED_DELTA_NET,SSM_CONV", "GATED_DELTA_NET and SSM_CONV (the state step, the chunked GDN, "
            "the conv)"),
           ("tbo-mov", "CPY,SET_ROWS,GET_ROWS,CONT", "CPY, SET_ROWS, GET_ROWS and CONT (the f32 and f16 moves)"),
           ("tbo-elt", "RMS_NORM,SOFT_MAX,ROPE,SWIGLU,SILU,SIGMOID,SOFTPLUS,GELU,ADD,MUL,SCALE",
            "RMS_NORM, SOFT_MAX, ROPE, SWIGLU, SILU, SIGMOID, SOFTPLUS, GELU (vision), ADD, MUL and SCALE"),
           ("tbo-fa", "FLASH_ATTN_EXT", "FLASH_ATTN_EXT (the cases with sinks fail at the tolerance edge on v79 too)"))
    for name, o, text in ops:
        out.append(Run(name, "test-backend-ops", f"test -b HTP0 -o {o}", "", 110, 2097152,
                       f"test-backend-ops {text} on HTP0 against the CPU"))
    out.append(Run("klp", "llama-perplexity", f"{kl} -b 512", "", 90, 8388608,
                   f"llama-perplexity KL against {t.base}, 1 chunk of 512, -b 512 (the HMX prefill path)"))
    out.append(Run("kld", "llama-perplexity", f"{kl} -b 1 -ub 1", "", 108, 8388608,
                   f"llama-perplexity KL against {t.base}, 1 chunk of 512, -b 1 (the HVX decode path)"))
    if kit:
        out.append(Run("bench", "llama-bench", f"{bench} -p 512 -n 32 -r 3", "", 100, 8388608,
                       "llama-bench pp512 and tg32 at the depth 0, -r 3"))
        return out
    # The A/B timing of the DSP libraries: new (lib) against HEAD (dsp-base), 3 rounds, the order alternated
    for rnd in range(1, 4):
        order = ("new", "base") if rnd % 2 else ("base", "new")
        for v in order:
            out.append(Run(f"p-{rnd}-{v}", "llama-bench", f"{bench} -p 512 -n 0 -r 3", "", 100, 8388608,
                           "llama-bench pp512 at the depth 0, -r 3", "lib" if v == "new" else "dsp-base"))
        for v in order[::-1]:
            out.append(Run(f"t-{rnd}-{v}", "llama-bench", f"{bench} -p 0 -n 32 -d 0,4096 -r 3", "", 100, 8388608,
                           "llama-bench tg32 at the depths 0 and 4096, -r 3", "lib" if v == "new" else "dsp-base"))
    return out


def run_lines(t: Target, r: Run) -> list:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{t.phone}/out/{r.name}"
    env = " ".join(x for x in (lib_env(t, r.dsp), r.env) if x)
    gate = f"{t.gate_env} sh {t.phone}/bin/gate.sh {r.gate_kb}".strip()
    cmd = (f"{gate} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"timeout -s KILL {r.limit} env {env} {t.phone}/bin/{r.tool} {r.args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if r.tool in ("llama-bench", "llama-perplexity") else "OP-TEST"
    thermal = f"{t.adb} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
    pgrep = f"{t.adb} shell '" + "; ".join(f"pgrep -x {x}" for x in TOOLS) + "; echo pgrep-done'"
    return ["#", f"# {title}: {r.name}, {r.text}", thermal, f"{t.adb} shell '{cmd}'", pgrep]


def setup_lines(t: Target, kit: bool) -> list:
    """The lines that copy the stage files to the phone and check them."""
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
    else:
        lines.insert(3, f"{t.adb} shell 'mkdir -p {t.phone}/dsp-base'")
        lines.insert(6, f"{t.adb} push {local}/dsp-base/* {t.phone}/dsp-base/")
        # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
        lines.append(f"{t.adb} shell 'ls -l /data/local/tmp/qwen/models | grep -E \"Qwen3.5-4B-Q8_0.gguf\"; "
                     f"ls -l {t.eval_dir}/wiki.test.raw {t.eval_dir}/{t.base}'")
    lines.append(f"{t.adb} shell 'cd {t.phone} && sha256sum -c SHA256SUMS | grep -c OK && sha256sum -c SHA256SUMS "
                 f"> /dev/null && echo \"stage files: all present\"; chmod 755 {t.phone}/bin/*'")
    return lines


def kit_probe_lines(t: Target) -> list:
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


def output_lines(t: Target, kit: bool) -> list:
    """The lines that pull the outputs, copy them to the box and remove the phone directory."""
    lines = [
        "#",
        "# ---- The outputs ----",
        "#",
        f"{t.adb} shell '" + "; ".join(f"pgrep -x {x}" for x in TOOLS) + f"; ls {t.phone}/out | wc -l; "
        f"du -sh {t.phone}/out'",
        f"rm -rf {LAPTOP_STAGE}/{t.out_dir}",
        f"{t.adb} pull {t.phone}/out {LAPTOP_STAGE}/{t.out_dir}",
        f"rsync -a --delete {LAPTOP_STAGE}/{t.out_dir}/ {BOX}/{t.out_dir}/",
        f"test $(ls {LAPTOP_STAGE}/{t.out_dir} | wc -l) -gt 10 && {t.adb} shell 'rm -rf {t.phone}' && echo removed",
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
# The libraries: tools/stages/v81/build.sh (build/v81/{stage}): the patched llama.cpp tree of HEAD (with
# hexagon-v81/0001, v81 uses the f32 to f16 conversion of v79) plus the patches of build/v81/patches: 0002 the
# start-up self-test of the DSP (the host library), 0003 the stride of the transpose tile copy (cpy-ops.c of the
# DSP library) with four CONT cases of test-backend-ops.{base_text}
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


def write_commands(set_name: str, path: Path) -> None:
    """Write the command file of one run set."""
    kit = set_name == "kit"
    t = KIT if kit else PHONE
    rs = runs(t, kit)
    title = ("Phone stage \"v81\" on the v79 phone: the canary, its load time, the op tests, the 4B KL and the A/B "
             "timing of the DSP library." if not kit else
             "The v81 silicon kit: a phone with the Hexagon v81 DSP (Snapdragon 8 Elite Gen 5, SM8850) of a device farm.")
    decisions = KIT_DECISIONS if kit else DECISIONS
    base_text = "" if kit else ("\n# dsp-base holds the DSP libraries of HEAD (build/v81/dsp-compare.txt gives the "
                                "hashes of the two sets).")
    text = HEADER.format(title=title, stage=t.stage, base_text=base_text,
                         runs="\n".join(f"#   {k}: {v}" for k, v in decisions.items()),
                         charger="a charger permitted (ALLOW_CHARGER=1)" if kit else "no charger",
                         minutes=20 if kit else 18, n_runs=len(rs))
    lines = text.rstrip("\n").split("\n") + ["#", "# ---- Setup ----"] + setup_lines(t, kit)
    if kit:
        lines += kit_probe_lines(t)
    lines += ["#", "# ---- The runs ----"]
    for r in rs:
        lines += run_lines(t, r)
    lines += output_lines(t, kit)
    path.write_text("\n".join(lines) + "\n")
    print(f"{path}: {len(lines)} lines, {len(rs)} runs")


def read(path: Path) -> str:
    """The text of a file, or an empty text."""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def table(root: Path) -> None:
    """Print the results of the pulled outputs."""
    print(f"# {root}")
    load = {"0": [], "1": []}
    for gate in sorted(root.glob("can-*-gate.txt")):
        name = gate.name[:-len("-gate.txt")]
        out = read(root / f"{name}.out")
        log = read(root / f"{name}.log")
        rc = re.search(r"rc=(\d+)", read(gate))
        m = re.search(r"load: ([0-9.]+) ms", out)
        canary = [ln for ln in log.splitlines() if "canary" in ln]
        works = re.findall(r"work \d+: .* (pass|FAIL)", out)
        print(f"{name}: rc {rc.group(1) if rc else '?'}, load {m.group(1) if m else '-'} ms, work {works}")
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
        txt = read(out)
        m = re.findall(r"(\d+)/(\d+) tests passed", txt)
        fails = [ln.strip() for ln in txt.splitlines() if "FAIL" in ln][:6]
        print(f"{out.stem}: {m[-1][0] + '/' + m[-1][1] if m else 'no summary'} passed; " + "; ".join(fails)[:600])
    for name in ("klp", "kld"):
        txt = read(root / f"{name}.out") + read(root / f"{name}.log")
        vals = {}
        for key, rx in (("mean", r"Mean\s+KLD:\s+([0-9.eE+-]+)"), ("max", r"Maximum KLD:\s+([0-9.eE+-]+)"),
                        ("top1", r"Same top p:\s+([0-9.]+)")):
            m = re.search(rx, txt)
            vals[key] = m.group(1) if m else "-"
        print(f"{name}: KL mean {vals['mean']}, max {vals['max']}, same top 1 {vals['top1']} %")
    def bench_rows(path: Path) -> list:
        """The (key, t/s, stddev) of each result line of a llama-bench jsonl output."""
        rows = []
        for ln in read(path).splitlines():
            try:
                j = json.loads(ln)
            except json.JSONDecodeError:
                continue
            key = f"pp{j.get('n_prompt')}" if j.get("n_gen", 0) == 0 else f"tg{j.get('n_gen')}"
            rows.append((f"{key}@d{j.get('n_depth', 0)}", j.get("avg_ts", 0.0), j.get("stddev_ts", 0.0)))
        return rows

    for key, ts, sd in bench_rows(root / "bench.out"):
        print(f"bench: {key}: {ts:.2f} +- {sd:.2f} t/s")
    # The A/B timing: a run counts when its gate passed, its exit code is 0 and the thermal status after it is 0
    ab = {}
    for out in sorted(root.glob("[pt]-*-*.out")):
        name = out.stem
        gate = read(root / f"{name}-gate.txt")
        ok = "gate: OK" in gate and re.search(r"rc=0\b", gate) and re.search(r"after: thermal=0\b", gate)
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
    probe = read(root / "probe-info" / "stdout.txt")
    if probe:
        print("probe info:\n" + "\n".join("    " + ln for ln in probe.splitlines()[:40]))


def main(argv: list) -> int:
    """Parse the arguments and run the command."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands")
    c.add_argument("--set", default="phone", choices=("phone", "kit"))
    c.add_argument("--out", type=Path)
    t = sub.add_parser("table")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    args = ap.parse_args(argv)
    if args.cmd == "commands":
        target = KIT if args.set == "kit" else PHONE
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
