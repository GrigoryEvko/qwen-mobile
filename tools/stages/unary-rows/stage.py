#!/usr/bin/env python3
"""The phone stage "unary-rows": which op of the row change of the pointwise unary ops moves the logits, and which
rows give the correct values on the NPU.

Usage:
    stage.py commands [--out PATH]   write the phone command file (build/unary-rows/phone-commands.txt)
    stage.py table [--root DIR]      print the results from the pulled logs (build/unary-rows/phone-out)

build/unary-rows/stage.py is a link to this file, and tools/stages/unary-rows/build.sh builds the files. The gate,
the conditions and the lines that copy the files come from tools/stages/quick/stage.py.

The variants, each in the app configuration of the stage quick:
    a  the HEAD libraries (lib-base)
    z  the new libraries with GGML_HEXAGON_UNARY_FLAT=0: no op gets new rows
    b  the new libraries with the preset value 1: each kind of pointwise op gets new rows
    s  GGML_HEXAGON_UNARY_FLAT=2: only SIGMOID
    p  GGML_HEXAGON_UNARY_FLAT=4: only SOFTPLUS
    c  GGML_HEXAGON_UNARY_FLAT=8: only SCALE (the clear of a recurrent state)
"""

import argparse
import importlib.util
import os
import re
import struct
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("quick_stage", Path(__file__).resolve().parents[1] / "quick" / "stage.py")
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

q.PHONE = "/data/local/tmp/qwen/unary-rows"
q.LAPTOP_STAGE = "build/unary-rows"
q.BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/unary-rows"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / q.LAPTOP_STAGE))
q.BINS = ("gate.sh", "memprobe", "test-backend-ops", "unarycheck")
q.LIBS = ("libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-htp-v79.so", "libggml-opencl.so",
          "libggml.so", "libllama-common.so", "libllama.so", "libmtmd.so")
# build.sh keeps in lib-new only the libraries that differ from lib-base, thus the push takes the list of the box
_lib_new = STAGE_DIR / "phone" / "lib-new"
q.NEW_LIBS = tuple(sorted(p.name for p in _lib_new.glob("*.so"))) if _lib_new.is_dir() else ("libggml-hexagon.so",)

q.VARIANTS = {v.key: v for v in (
    q.Variant("a", "lib-base", "", "HEAD"),
    q.Variant("z", "lib-new", "GGML_HEXAGON_UNARY_FLAT=0", "new, no op gets new rows"),
    q.Variant("b", "lib-new", "", "new, each kind of pointwise op gets new rows"),
    q.Variant("s", "lib-new", "GGML_HEXAGON_UNARY_FLAT=2", "new, only SIGMOID gets new rows"),
    q.Variant("p", "lib-new", "GGML_HEXAGON_UNARY_FLAT=4", "new, only SOFTPLUS gets new rows"),
    q.Variant("c", "lib-new", "GGML_HEXAGON_UNARY_FLAT=8", "new, only SCALE gets new rows"),
)}
q.BLOCKS = [
    q.Block("h", "memprobe", "--hash -p 1024 -n 16", "azbspc", 1, 90, 8388608, "",
            "memprobe --hash, a prompt of 1024 tokens and 16 decode tokens: the logits hashes"),
    q.Block("u", "unarycheck", "--cpu", "azb", 1, 60, 2097152, "",
            "unarycheck --cpu: the ops of the model shapes on HTP0 against the CPU of the phone"),
    q.Block("k", "test-backend-ops", "-o SIGMOID,SOFTPLUS,SCALE -b HTP0", "azb", 1, 100, 2097152, "",
            "test-backend-ops of SIGMOID, SOFTPLUS and SCALE on HTP0 against the CPU"),
]
CHECK_CASES = ("sigmoid_beta_t1024", "sigmoid_beta_t1", "softplus_gate_t1024", "softplus_gate_t1")


def run_lines(b: q.Block, rnd: int, v: q.Variant) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    name = f"{b.key}-{rnd}-{v.key}"
    stem = f"{q.PHONE}/out/{name}"
    # lib-new holds only the libraries that differ (the Hexagon backend)
    lib = f"{q.PHONE}/{v.lib}"
    ld = lib if v.lib == "lib-base" else f"{lib}:{q.PHONE}/lib-base"
    # The row change does not change the DSP code, thus each run loads the DSP library of lib-base
    env = " ".join(x for x in (f"LD_LIBRARY_PATH={ld} ADSP_LIBRARY_PATH={q.PHONE}/lib-base", q.APP_ENV, v.env, b.env) if x)
    pre = ""
    if b.tool == "memprobe":
        tool = f"{q.PHONE}/bin/memprobe -m {q.MODEL} {q.PROBE_ARGS} {b.args}"
        title = f"# REAL-MODEL Qwen3.5-4B-Q8_0: {name}, {b.text}, {v.key.upper()}: {v.text}"
    elif b.tool == "unarycheck":
        pre = f"mkdir -p {stem}-dump && "
        tool = f"{q.PHONE}/bin/unarycheck {b.args} --dump {stem}-dump"
        title = f"# KERNEL: {name}, {b.text}, {v.key.upper()}: {v.text}"
    else:
        tool = f"{q.PHONE}/bin/{b.tool} {b.args}"
        title = f"# KERNEL: {name}, {b.text}, {v.key.upper()}: {v.text}"
    cmd = (f"sh {q.PHONE}/bin/gate.sh {b.gate_kb} > {stem}-gate.txt && {q.BEFORE} >> {stem}-gate.txt && {pre}"
           f"timeout -s KILL {b.limit} env {env} {tool} > {stem}.out 2> {stem}.log; "
           f"echo \"rc=$?\" >> {stem}-gate.txt; {q.AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt")
    return ["#", title, q.THERMAL, f"{q.ADB} shell '{cmd}'", q.PGREP]


q.run_lines = run_lines

q.HEADER = """\
# Phone stage "unary-rows": which op of the row change of the pointwise unary ops moves the logits of the 4B Q8_0,
# and which rows give the correct values on the NPU. The app configuration: Q8_0 K and V with the FWHT rotation,
# GGML_HEXAGON_OPFUSION=1, OPFUSION_STATE=1.
#
# The libraries (tools/stages/unary-rows/build.sh): lib-base is the tree of HEAD, lib-new the same tree with the
# row change and its switch GGML_HEXAGON_UNARY_FLAT (1 each kind, 0 none, a mask: 2 SIGMOID, 4 SOFTPLUS, 8 SCALE).
# lib-new holds only the libraries that differ from lib-base. The row change does not change the DSP code, thus each
# run loads the DSP library of lib-base. The stage quick found that the row change moves the logits hashes, and the
# simulator gives the same bits for the old and the new rows.
#
# The variants: A HEAD, Z new with the value 0, B new with the value 1, S only SIGMOID, P only SOFTPLUS, C only SCALE.
#
# The runs, 12:
#   h  memprobe --hash -p 1024 -n 16: A Z B S P C. Decides: Z has the hashes of A (the value 0 is HEAD), and B has
#      the hashes of B of the stage quick. Each of S, P and C that has other hashes than A names a kind of op whose
#      new rows move the values.
#   u  unarycheck --cpu --dump: A Z B. The ops of the model shapes on HTP0 against the CPU of the phone: SIGMOID of
#      [1, 32, T], SOFTPLUS of [32, T], the SCALE by 0 of a state slot, and chains of 24 layers in one graph, for
#      T = 1024 and 1. Decides: for each case, the error against the CPU (nmse, maxerr and differ, the count of
#      elements with other bits) of the old rows (A, Z) and of the new rows (B). A case with an error far above the
#      error of the kernel (NMSE 2.5e-9 at most for the int16 SIGMOID) shows the rows that the NPU computes
#      incorrectly. The dumps give the elements that differ between the old and the new rows.
#   k  test-backend-ops -o SIGMOID,SOFTPLUS,SCALE -b HTP0: A Z B. Decides: the test shapes pass with the old and the
#      new rows.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB for memprobe and 2 GB for the kernel runs, and it prints the caps), the tool under timeout -s KILL (100 s or
# less), the exit code and the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger), although only
# the values decide. Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 5 minutes of tool
# time plus about 8 s of gate and checks for each run. The push is about 160 MB, the pull less than 5 MB. Then on
# the box: python3 build/unary-rows/stage.py table
"""

CHECK_RE = re.compile(r"^unarycheck case=(\S+) flat=\S+ hash=([0-9a-f]{16}) nonfinite=(\d+)(.*?) us=", re.M)


def hashes(root: Path, name: str) -> list[tuple[str, str]]:
    """The HASH lines of one run as (what, hex) pairs, in order."""
    return re.findall(r"^HASH (.*) ([0-9a-f]{16})$", q.read(root, name)[1], re.M)


def dump(root: Path, run: str, case: str) -> list[float]:
    """The f32 values of one dumped case of one run, or an empty list."""
    p = root / f"{run}-dump" / f"{case}.f32"
    if not p.exists():
        return []
    data = p.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def table(root: Path) -> int:
    """Print the results of each block."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    for b, rnd, v in q.runs():
        gate, _, _ = q.read(root, f"{b.key}-{rnd}-{v.key}")
        print(f"{b.key}-{rnd}-{v.key}: {q.conditions(gate) if gate else 'no gate file'}")
    print()

    h = {k: hashes(root, f"h-1-{k}") for k in "azbspc"}
    for k in "azbspc":
        one = h[k]
        same_a = sum(x == y for x, y in zip(one, h["a"]))
        same_b = sum(x == y for x, y in zip(one, h["b"]))
        first = next((w for (w, x), (_, y) in zip(one, h["a"]) if x != y), "none")
        print(f"h {k.upper()}: {len(one)} HASH lines, equal to A in {same_a}, equal to B in {same_b}, "
              f"the first line that differs from A: {first}")
    print()

    checks = {k: {m.group(1): m for m in CHECK_RE.finditer(q.read(root, f"u-1-{k}")[1])} for k in "azb"}
    cases = list(dict.fromkeys(c for k in "azb" for c in checks[k]))
    for case in cases:
        print(f"u {case}:")
        for k in "azb":
            m = checks[k].get(case)
            print(f"    {k.upper()}: " + (f"hash {m.group(2)} nonfinite {m.group(3)}{m.group(4)}" if m else "no line"))
    for case in CHECK_CASES:
        old, new = dump(root, "u-1-z", case), dump(root, "u-1-b", case)
        base = dump(root, "u-1-a", case)
        if not old or len(old) != len(new):
            print(f"u dump {case}: missing")
            continue
        differ = [i for i, (x, y) in enumerate(zip(old, new)) if struct.pack("<f", x) != struct.pack("<f", y)]
        maxd = max((abs(old[i] - new[i]) for i in differ), default=0.0)
        same_az = base == old
        print(f"u dump {case}: Z against B {len(differ)} of {len(old)} elements differ, max abs diff {maxd:.3g}, "
              f"first {differ[:5]}; A equal to Z: {same_az}")
    print()

    for k in "azb":
        _, out, log = q.read(root, f"k-1-{k}")
        summary = re.findall(r"(\d+)/(\d+) tests passed", out + log)
        fails = [ln.strip() for ln in (out + log).splitlines() if "FAIL" in ln]
        print(f"k {k.upper()}: tests passed {summary}, FAIL lines {len(fails)}")
        for ln in fails[:10]:
            print(f"    {ln[:160]}")
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the results from the pulled logs")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    a = ap.parse_args()
    if a.cmd == "commands":
        n = q.write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(q.runs())} runs")
        return 0
    return table(a.root)


if __name__ == "__main__":
    sys.exit(main())
