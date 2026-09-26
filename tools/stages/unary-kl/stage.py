#!/usr/bin/env python3
"""The phone stage "unary-kl": the KL check, the logits hashes and the speed of the row change of the pointwise
unary ops (the switch GGML_HEXAGON_UNARY_FLAT).

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/unary-kl/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled logs (build/unary-kl/phone-out)

build/unary-kl/stage.py is a link to this file, and tools/stages/unary-kl/build.sh builds the files.

The stage unary-rows found that the new rows of the SOFTPLUS move the logits hashes: on the NPU, 167 of 32768
elements of the prompt shape differ by 1 ulp, with the same error against the CPU for the old and the new rows.
The SIGMOID and the SCALE did not move the hashes of the 4B. This stage decides the preset of the switch:
    - The value 1 passes a KL block when the difference of its mean KLD against HEAD is inside the ± error of HEAD
      that llama-perplexity prints, its same top p agrees inside the error of HEAD, and its maximum KLD is at most
      2 times the maximum KLD of HEAD. The NPU repeats each run bit for bit, thus the two runs of a variant give
      the same values and the error of the mean is the spread. If the value 1 passes kp and kd, the preset stays 1
      and each kind of op gets new rows.
    - Else the preset is 26 (SIGMOID, SCALE and the other pointwise ops, not SOFTPLUS), if the hashes of 26 equal
      the hashes of HEAD on the 4B and on the 2B.

The variants, each in the app configuration (Q8_0 K and V with the FWHT rotation, op fusion and state fusion on):
    a  the HEAD libraries (lib-base)
    m  the new libraries with GGML_HEXAGON_UNARY_FLAT=26: not SOFTPLUS
    f  the new libraries with the preset value 1: each kind of pointwise op

The blocks, in the order of the stage:
    h   memprobe --hash -p 1024 -n 16 on the 4B: A M
    i   memprobe --hash -p 1024 -n 16 on the 2B: A M
    kp  llama-perplexity KL of the prefill path (-b 512, 4 chunks of 512) on the 4B against naive-4B-q8.kld: A F,
        2 rounds (A F, F A)
    kd  llama-perplexity KL of the decode path (-b 1 -ub 1, 1 chunk of 512) on the 4B against the same base: A F,
        2 rounds
    p   llama-bench pp512 and pp1024 at depth 0, -r 3: A M F, 3 rounds (A M F, F M A, A M F)
    g   llama-bench tg32 at depth 0, -r 3: A M F, 3 rounds

A run name is <block>-<round>-<variant>, for example kp-2-f. Each run writes <name>-gate.txt, <name>.out and
<name>.log to the phone directory out/. The rate table uses a run when its gate passed, its exit code is 0, the CPU
caps after the run are the caps before it, and the thermal status after it is 0. --all also uses the runs with
changed caps or heat. The table only reads files. O(size of the logs) time.
"""

import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cli, commands, device, gate, logs, parse  # noqa: E402

ADB = device.ADB
PATHS = device.stage_paths("unary-kl", __file__)
PHONE, LAPTOP_STAGE, BOX, STAGE_DIR = PATHS
MODEL_4B = f"{device.MODEL_DIR}/{device.MODEL_4B}"
MODEL_2B = f"{device.MODEL_DIR}/{device.MODEL_2B}"
EVAL = device.EVAL_DIR
MODEL_KB = device.GATE_4B_KB
APP_ENV = device.APP_ENV
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
PROBE_ARGS = device.PROBE_ARGS
PPL_ARGS = (f"-dev HTP0 -ngl 99 -t 4 -fa on -ctk q8_0 -ctv q8_0 -f {EVAL}/wiki.test.raw -c 512 "
            f"--kl-divergence-base {EVAL}/naive-4B-q8.kld --kl-divergence")
# The kernel keeps 15 characters of a process name, thus llama-perplexity goes in cut to 15 characters
TOOLS = ("llama-bench", "llama-perplexit", "memprobe")
THERMAL = device.THERMAL
PGREP = device.pgrep(*TOOLS)


@dataclass(frozen=True)
class Variant:
    """One library set and its environment switch."""
    key: str
    lib: str
    env: str
    text: str


@dataclass(frozen=True)
class Block:
    """One kind of run: the tool with its arguments, the variants of an odd round, the rounds, the time limit in
    seconds and the text. The gate and the lines after the tool take about 6 s, thus each limit is 110 s or less."""
    key: str
    tool: str
    args: str
    variants: str
    rounds: int
    limit: int
    text: str


VARIANTS = {v.key: v for v in (
    Variant("a", "lib-base", "", "HEAD"),
    Variant("m", "lib-new", "GGML_HEXAGON_UNARY_FLAT=26", "new, value 26: not SOFTPLUS"),
    Variant("f", "lib-new", "", "new, preset value 1: each kind"),
)}
BLOCKS = [
    Block("h", "memprobe", f"-m {MODEL_4B} {PROBE_ARGS} --hash -p 1024 -n 16", "am", 1, 90,
          "memprobe --hash on the 4B, a prompt of 1024 tokens and 16 decode tokens"),
    Block("i", "memprobe", f"-m {MODEL_2B} {PROBE_ARGS} --hash -p 1024 -n 16", "am", 1, 90,
          "memprobe --hash on the 2B, a prompt of 1024 tokens and 16 decode tokens"),
    Block("kp", "llama-perplexity", f"-m {MODEL_4B} {PPL_ARGS} --chunks 4 -b 512", "af", 2, 100,
          "KL of the prefill path on the 4B, 4 chunks of 512, -b 512"),
    Block("kd", "llama-perplexity", f"-m {MODEL_4B} {PPL_ARGS} --chunks 1 -b 1 -ub 1", "af", 2, 110,
          "KL of the decode path on the 4B, 1 chunk of 512, -b 1 -ub 1"),
    Block("p", "llama-bench", f"-m {MODEL_4B} {BENCH_ARGS} -p 512,1024 -n 0 -d 0 -r 3", "amf", 3, 90,
          "llama-bench pp512 and pp1024 at depth 0, 3 repetitions"),
    Block("g", "llama-bench", f"-m {MODEL_4B} {BENCH_ARGS} -p 0 -n 32 -d 0 -r 3", "amf", 3, 90,
          "llama-bench tg32 at depth 0, 3 repetitions"),
]


def runs() -> list[tuple[Block, int, Variant]]:
    """The runs in the order of the stage: each block, round by round, odd rounds in the order of the block and even
    rounds in the reverse order. O(runs)."""
    out = []
    for b in BLOCKS:
        for rnd in range(1, b.rounds + 1):
            order = b.variants if rnd % 2 else b.variants[::-1]
            out.extend((b, rnd, VARIANTS[k]) for k in order)
    return out


def run_lines(b: Block, rnd: int, v: Variant) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    name = f"{b.key}-{rnd}-{v.key}"
    lib = f"{PHONE}/{v.lib}"
    ld = lib if v.lib == "lib-base" else f"{lib}:{PHONE}/lib-base"
    # The row change does not change the DSP code, thus each run loads the DSP library of lib-base
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={ld} ADSP_LIBRARY_PATH={PHONE}/lib-base", APP_ENV, v.env) if x)
    cmd = commands.gated_run(f"{PHONE}/out/{name}", MODEL_KB, b.limit, env,
                             f"{PHONE}/bin/{b.tool} {b.args}", stage=PHONE)
    model = "Qwen3.5-2B-Q8_0" if MODEL_2B in b.args else "Qwen3.5-4B-Q8_0"
    return commands.run_lines(f"# REAL-MODEL {model}: {name}, {b.text}, {v.key.upper()}: {v.text}", cmd, PGREP)


HEADER = """\
# Phone stage "unary-kl": the KL check, the logits hashes and the speed of the row change of the pointwise unary ops
# (the switch GGML_HEXAGON_UNARY_FLAT), in the app configuration (Q8_0 K and V with the FWHT rotation,
# GGML_HEXAGON_OPFUSION=1, OPFUSION_STATE=1).
#
# The libraries (tools/stages/unary-kl/build.sh): lib-base is the tree of HEAD, lib-new the same tree with the row
# change. lib-new holds only the libraries that differ, and each run loads the DSP library of lib-base.
#
# The variants: A HEAD, M new with GGML_HEXAGON_UNARY_FLAT=26 (not SOFTPLUS), F new with the preset value 1.
#
# The runs, 30:
#   h   memprobe --hash -p 1024 -n 16 on the 4B: A M. Decides: M has the hashes of A.
#   i   memprobe --hash -p 1024 -n 16 on the 2B: A M. Decides: M has the hashes of A.
#   kp  KL of the prefill path on the 4B against the naive base (4 chunks, -b 512): A F, 2 rounds.
#   kd  KL of the decode path on the 4B against the naive base (1 chunk, -b 1 -ub 1): A F, 2 rounds.
#       Decides: F passes when its mean KLD minus the mean KLD of A is inside the ± error of A, its same top p
#       agrees with A inside the error of A, and its maximum KLD is at most 2 times that of A. If F passes kp and kd,
#       the preset stays 1. Else the preset is 26, if h and i give the hashes of A.
#   p   llama-bench pp512 and pp1024 at depth 0, -r 3: A M F, 3 rounds. Decides: the prefill gain of M and of F.
#   g   llama-bench tg32 at depth 0, -r 3: A M F, 3 rounds. Decides: the decode speed of M and F against A.
# The variants run in the order of the block in an odd round and in the reverse order in an even round.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB, and it prints the caps), the tool under timeout -s KILL (110 s or less), the exit code and the conditions after
# the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 21 minutes of tool time plus about
# 4 minutes of gates and checks, plus the waits for thermal status 0. The push is about 145 MB, the pull less than
# 2 MB. Then on the box: python3 build/unary-kl/stage.py table
"""


def stage_files() -> list[str]:
    """The files of the stage: each line of phone/SHA256SUMS that the build wrote. The build recipe checks the needed
    libraries of each program and library, thus the push takes each of these files."""
    sums = STAGE_DIR / "phone" / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"stage.py: {sums} does not exist. Run tools/stages/unary-kl/build.sh first.")
    return [line.split()[1] for line in sums.read_text().splitlines() if line.strip()]


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    files = stage_files()
    pushes = {d: [f"{LAPTOP_STAGE}/phone/{f}" for f in files if f.startswith(f"{d}/")]
              for d in ("bin", "lib-base", "lib-new")}
    # No "models/Qwen3.5" in the model line: the runner gates each line with that text as a model run.
    return commands.setup_lines(
        PATHS, pushes, count=len(files),
        model_check=f"{ADB} shell 'ls -l {device.MODEL_DIR} | grep -E \"(2B|4B)-Q8_0.gguf\"; "
                    f"ls -l {EVAL} | grep -E \"naive-4B-q8|wiki.test\"'")


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = commands.header_lines(HEADER) + setup_lines()
    lines += ["#", f"# ==== {len(runs())} runs ===="]
    for b, rnd, v in runs():
        lines += run_lines(b, rnd, v)
    lines += commands.output_lines(PATHS, tools=TOOLS)
    return commands.write_commands(path, lines)


# ---- The table ----

PPL_RE = re.compile(r"Mean PPL\(Q\)\s+:\s+([\d.]+)")


def hashes(root: Path, name: str) -> list[tuple[str, str]]:
    """The HASH lines of one run as (what, hex) pairs, in order."""
    return re.findall(r"^HASH (.*) ([0-9a-f]{16})$", logs.read_text(root / f"{name}.out"), re.M)


def kl_checks(key: str, vals: dict[str, list[tuple[float, float, float, float, float]]]) -> str:
    """The three checks of the value 1 (F) against HEAD (A) of one KL block, each value the mean of the rounds:
    1. the difference of the mean KLD is inside the ± error of HEAD,
    2. the difference of the same top p is inside the ± error of HEAD,
    3. the maximum KLD of F is at most 2 times the maximum KLD of HEAD.
    F passes when the three checks pass."""
    if not vals.get("a") or not vals.get("f"):
        return f"  {key}: no check, a run of A or F has no KLD, maximum or top p line"
    a = [statistics.mean(col) for col in zip(*vals["a"])]
    f = [statistics.mean(col) for col in zip(*vals["f"])]
    # llama-perplexity prints 6 decimals of a KLD and 3 of a top p, thus the checks round the differences to them
    d_kld, d_top = round(f[0] - a[0], 6), round(f[3] - a[3], 3)
    checks = [
        ("mean KLD", abs(d_kld) <= a[1] + 1e-12, f"F - A {d_kld:+.6f}, the error of A ± {a[1]:.6f}"),
        ("same top p", abs(d_top) <= a[4] + 1e-9, f"F - A {d_top:+.3f} %, the error of A ± {a[4]:.3f} %"),
        ("maximum KLD", f[2] <= 2.0 * a[2], f"F {f[2]:.6f}, 2 times A {2.0 * a[2]:.6f}"),
    ]
    lines = [f"  {key} checks of F against A:"]
    lines += [f"    {'PASS' if ok else 'FAIL'} {name}: {text}" for name, ok, text in checks]
    lines.append(f"    {key}: F {'PASSES' if all(ok for _, ok, _ in checks) else 'DOES NOT PASS'}")
    return "\n".join(lines)


def table(root: Path, include_all: bool) -> int:
    """Print the results of each block."""
    if not root.is_dir():
        return cli.missing_root(root)
    for b, rnd, v in runs():
        name = f"{b.key}-{rnd}-{v.key}"
        c = gate.read(logs.read_text(root / f"{name}-gate.txt"))
        print(f"{name}: {'ok' if c.ok else 'NOT OK'}" + (f" [{', '.join(c.flags)}]" if c.flags else ""))
    print()

    for key, model in (("h", "4B"), ("i", "2B")):
        a, m = hashes(root, f"{key}-1-a"), hashes(root, f"{key}-1-m")
        same = sum(x == y for x, y in zip(a, m))
        print(f"{key} {model}: A {len(a)} and M {len(m)} HASH lines, {same} equal"
              + (" (the same hashes)" if a and a == m else " (DIFFERENT)"))
    print()

    print("KL against the naive base naive-4B-q8.kld: mean ± error of the mean, maximum, same top-1 ± error, PPL")
    for key in ("kp", "kd"):
        block = next(b for b in BLOCKS if b.key == key)
        # For each variant, the values of each round: (mean KLD, its error, maximum KLD, same top p, its error)
        vals: dict[str, list[tuple[float, float, float, float, float]]] = {}
        for k in block.variants:
            for rnd in range(1, block.rounds + 1):
                _, out, log = logs.read_run(root, f"{key}-{rnd}-{k}")
                text = out + log
                m, t = parse.KLD_RE.search(text), parse.TOP_RE.search(text)
                mx, ppl = parse.MAXKL_RE.search(text), PPL_RE.search(text)
                if m and t and mx:
                    vals.setdefault(k, []).append((float(m.group(1)), float(m.group(2)), float(mx.group(1)),
                                                   float(t.group(1)), float(t.group(2))))
                cell = (f"{m.group(1)} ± {m.group(2)}, max {mx.group(1) if mx else '?'}, "
                        f"top-1 {t.group(1) + ' ± ' + t.group(2) if t else '?'} %, PPL {ppl.group(1) if ppl else '?'}"
                        if m else "no KLD line")
                print(f"  {key}-{rnd}-{k} {VARIANTS[k].text:32s}: {cell}")
        print(kl_checks(key, vals))
    print()

    rates: dict[tuple, dict[str, dict[int, float]]] = {}
    for b, rnd, v in runs():
        if b.tool != "llama-bench":
            continue
        text, out, _ = logs.read_run(root, f"{b.key}-{rnd}-{v.key}")
        c = gate.read(text)
        if not c.ok or (c.flags and not include_all):
            continue
        for key, ts in parse.bench_values(out).items():
            rates.setdefault(key, {}).setdefault(v.key, {})[rnd] = ts
    print("llama-bench t/s: the median of the rounds, and the median change to A over the paired rounds [range]"
          + ("" if include_all else " (--all also uses the runs with changed caps or heat)"))
    for key, per in sorted(rates.items()):
        cells = []
        for k in "amf":
            vals = per.get(k, {})
            if not vals:
                continue
            cell = f"{k.upper()} {statistics.median(vals.values()):.2f}"
            if k != "a" and "a" in per:
                ratios = [vals[r] / per["a"][r] for r in vals if r in per["a"]]
                if ratios:
                    cell += (f" ({100 * (statistics.median(ratios) - 1):+.2f}% "
                             f"[{100 * (min(ratios) - 1):+.2f}, {100 * (max(ratios) - 1):+.2f}], {len(ratios)} rounds)")
            cells.append(cell)
        print(f"  pp{key[0]} tg{key[1]} d{key[2]}: " + " | ".join(cells))
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    return cli.run(STAGE_DIR, __doc__, write=write_commands, count=lambda: len(runs()), table=table,
                   table_help="print the results from the pulled logs")


if __name__ == "__main__":
    sys.exit(main())
