#!/usr/bin/env python3
"""The phone stage "unary-tg": the decode speed of the row change of the pointwise unary ops
(patches/hexagon-host/0013), with GGML_HEXAGON_UNARY_FLAT=0 against the preset 1 on the same library set.

Usage:
    stage.py commands [--out PATH]                  write the phone command file (build/unary-tg/phone-commands.txt)
    stage.py table [--root DIR] [--runner-log F]    print the table from the pulled logs (build/unary-tg/phone-out)

build/unary-tg/stage.py is a link to this file, and tools/stages/unary-tg/build.sh builds the files.

The stage unary-kl measured tg32 at the depth 0 at -0.25 % for the preset 1 against HEAD, with each of three rounds
below HEAD. This stage decides with longer runs:
    o  GGML_HEXAGON_UNARY_FLAT=0: each op keeps its rows (the rows of the series before the patch)
    f  the preset 1: the pointwise ops get new rows
Block t: llama-bench tg128 at the depths 0 and 4096, -r 2, on the 4B Q8_0, O and F in 5 alternated rounds (O F, F O,
O F, F O, O F).

The rule: for each depth, if the median of F is lower than the median of O by more than the spread of the O runs (the
largest minus the smallest O run), the default of the switch changes to 0 in a new patch. Else the preset stays 1.

A run is valid when its gate passed, its exit code is 0, and the runner log does not mark it SCREEN-OFF or
CAPS-CHANGED. The runner does not stop a run when the screen goes off. It writes the two marks after the run, and
the table rejects a run with a mark. The last line of the command file copies the runner log /tmp/unary-tg.log to
build/unary-tg/runner.log on the box. Without a runner log, the table uses the gate files only and says so. The
table only reads files. O(size of the logs) time.
"""

import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cli, commands, device, gate, logs, parse  # noqa: E402

ADB = device.ADB
PATHS = device.stage_paths("unary-tg", __file__)
PHONE, LAPTOP_STAGE, BOX, STAGE_DIR = PATHS
MODEL = f"{device.MODEL_DIR}/{device.MODEL_4B}"
MODEL_KB = device.GATE_4B_KB
RUNNER_LOG = "/tmp/unary-tg.log"
APP_ENV = device.APP_ENV
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -ctk q8_0 -ctv q8_0 -o jsonl"
LIMIT = 110  # seconds; one run takes about 70 s (load 6 s, 2 x tg128 at d0, 2 x (depth 4096 + tg128))
ROUNDS = 5
THERMAL = device.THERMAL
TOOLS = ("llama-bench",)
PGREP = device.pgrep(*TOOLS)


@dataclass(frozen=True)
class Variant:
    """One value of the switch."""
    key: str
    env: str
    text: str


VARIANTS = {v.key: v for v in (
    Variant("o", "GGML_HEXAGON_UNARY_FLAT=0", "switch off, the old rows"),
    Variant("f", "", "the preset 1, the new rows"),
)}


def runs() -> list[tuple[int, Variant]]:
    """The runs in the order of the stage: O F in an odd round and F O in an even round. O(runs)."""
    out = []
    for rnd in range(1, ROUNDS + 1):
        out.extend((rnd, VARIANTS[k]) for k in ("of" if rnd % 2 else "fo"))
    return out


def run_lines(rnd: int, v: Variant) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    name = f"t-{rnd}-{v.key}"
    env = " ".join(x for x in (device.lib_env(PHONE), APP_ENV, v.env) if x)
    tool = f"{PHONE}/bin/llama-bench -m {MODEL} {BENCH_ARGS} -p 0 -n 128 -d 0,4096 -r 2"
    cmd = commands.gated_run(f"{PHONE}/out/{name}", MODEL_KB, LIMIT, env, tool, stage=PHONE)
    title = (f"# REAL-MODEL Qwen3.5-4B-Q8_0: {name}, llama-bench tg128 at the depths 0 and 4096, -r 2, "
             f"{v.key.upper()}: {v.text}")
    return commands.run_lines(title, cmd, PGREP)


HEADER = f"""\
# Phone stage "unary-tg": the decode speed of the row change of the pointwise unary ops (patches/hexagon-host/0013,
# landed), with GGML_HEXAGON_UNARY_FLAT=0 (O, the old rows) against the preset 1 (F, the new rows) on one library set:
# the tree of HEAD (tools/stages/unary-tg/build.sh). The app configuration: Q8_0 K and V with the FWHT rotation,
# GGML_HEXAGON_OPFUSION=1, OPFUSION_STATE=1.
#
# The runs, 10: llama-bench -p 0 -n 128 -d 0,4096 -r 2 on the 4B Q8_0, O and F in {ROUNDS} alternated rounds (O F, F O, ...).
# Decides: for each depth, if the median of F is lower than the median of O by more than the spread of the O runs,
# the default of the switch changes to 0 in a new patch. Else the preset stays 1.
# A run with the runner mark SCREEN-OFF or CAPS-CHANGED is not valid, and the table rejects it. The last line copies the runner log
# {RUNNER_LOG} to {LAPTOP_STAGE}/runner.log on the box, for the table. If the runner writes a different file, copy
# that file to {LAPTOP_STAGE}/runner.log on the box.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,
# MemAvailable 8 GB, and it prints the caps), llama-bench under timeout -s KILL ({LIMIT} s), the exit code and the
# conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 12 minutes of tool time (10 runs of
# about 70 s) plus about 2 minutes of gates and checks, plus the waits for thermal status 0. The push is about 130 MB,
# the pull less than 1 MB. Then on the box: python3 build/unary-tg/stage.py table
"""


def stage_files() -> list[str]:
    """The files of the stage: each line of phone/SHA256SUMS that the build wrote."""
    sums = STAGE_DIR / "phone" / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"stage.py: {sums} does not exist. Run tools/stages/unary-tg/build.sh first.")
    return [line.split()[1] for line in sums.read_text().splitlines() if line.strip()]


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    files = stage_files()
    pushes = {d: [f"{LAPTOP_STAGE}/phone/{f}" for f in files if f.startswith(f"{d}/")] for d in ("bin", "lib")}
    # No "models/Qwen3.5" in the model line: the runner gates each line with that text as a model run.
    return commands.setup_lines(PATHS, pushes, count=len(files),
                                model_check=f"{ADB} shell 'ls -l {device.MODEL_DIR} | grep 4B-Q8_0.gguf'")


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them and the runner log to the box, and remove the phone directory. The
    phone directory goes only when the pull has each of its files."""
    return commands.output_lines(PATHS, tools=TOOLS, extra=[
        f"(cp {RUNNER_LOG} {LAPTOP_STAGE}/runner.log && rsync -a {LAPTOP_STAGE}/runner.log {BOX}/runner.log "
        f"&& echo copied the runner log) || echo 'no runner log {RUNNER_LOG}: copy the runner log to {BOX}/runner.log'",
    ])


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = commands.header_lines(HEADER) + setup_lines()
    lines += ["#", f"# ==== {len(runs())} runs ===="]
    for rnd, v in runs():
        lines += run_lines(rnd, v)
    lines += output_lines()
    return commands.write_commands(path, lines)


# ---- The table ----

TITLE_RE = re.compile(r"^# REAL-MODEL [^:]*: (t-\d+-[of]),", re.M)


def runner_marks(path: Path) -> dict[str, list[str]] | None:
    """The marks SCREEN-OFF and CAPS-CHANGED of the runner for each run name, or None without a runner log. The marks
    of a run are the lines between its title and the next title or the outputs."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    marks: dict[str, list[str]] = {}
    titles = list(TITLE_RE.finditer(text))
    for i, m in enumerate(titles):
        end = titles[i + 1].start() if i + 1 < len(titles) else text.find("# ---- The outputs", m.end())
        section = text[m.end():end if end > 0 else len(text)]
        marks[m.group(1)] = [w for w in ("SCREEN-OFF", "CAPS-CHANGED") if w in section]
    return marks


def table(root: Path, log: Path) -> int:
    """Print the validity of each run, the rates of each round and the decision for each depth."""
    if not root.is_dir():
        return cli.missing_root(root)
    marks = runner_marks(log)
    if marks is None:
        print(f"NOTE: no runner log {log}. The table cannot see SCREEN-OFF and CAPS-CHANGED, thus it uses the gate "
              f"files only.\n")
    rates: dict[int, dict[str, dict[int, float]]] = {}
    for rnd, v in runs():
        name = f"t-{rnd}-{v.key}"
        text, out = logs.read_run(root, name, ("-gate.txt", ".out"))
        removed = list(gate.read(text).faults)
        if marks is not None:
            if name not in marks:
                removed.append("not in the runner log")
            else:
                removed += marks[name]
        values = {depth: ts for (_, _, depth), ts in parse.bench_values(out).items()}
        cells = " ".join(f"d{d} {values[d]:.3f}" for d in sorted(values)) or "no rate"
        print(f"{name} {v.text:26s}: {cells}  {'VALID' if not removed else 'NOT VALID: ' + ', '.join(removed)}")
        if not removed:
            for d, ts in values.items():
                rates.setdefault(d, {}).setdefault(v.key, {})[rnd] = ts
    print()
    for d in (0, 4096):
        per = rates.get(d, {})
        o, f = per.get("o", {}), per.get("f", {})
        if len(o) < 3 or len(f) < 3:
            print(f"d{d}: not sufficient valid runs (O {len(o)}, F {len(f)}, 3 of each necessary): no decision")
            continue
        med_o, med_f = statistics.median(o.values()), statistics.median(f.values())
        spread = max(o.values()) - min(o.values())
        ratios = [100 * (f[r] / o[r] - 1) for r in sorted(f) if r in o]
        slower = med_o - med_f > spread
        print(f"d{d}: O median {med_o:.3f} t/s (spread {spread:.3f}: {min(o.values()):.3f} to {max(o.values()):.3f}, "
              f"{len(o)} runs), F median {med_f:.3f} t/s ({len(f)} runs), F - O {med_f - med_o:+.3f} t/s "
              f"({100 * (med_f / med_o - 1):+.2f} %)")
        print("     F against O in each round with two valid runs: "
              + (", ".join(f"{x:+.2f} %" for x in ratios) if ratios else "none"))
        print(f"     {'F IS SLOWER BY MORE THAN THE SPREAD OF O: change the default to 0' if slower else 'F is inside the spread of O: the preset stays 1'}")
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    return cli.run(STAGE_DIR, __doc__, write=write_commands, count=lambda: len(runs()), table=table,
                   all_flag=False, table_help="print the table from the pulled logs",
                   runner_log="the log of the laptop runner, for the marks SCREEN-OFF and CAPS-CHANGED",
                   runner_log_default=True)


if __name__ == "__main__":
    sys.exit(main())
