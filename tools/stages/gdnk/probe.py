#!/usr/bin/env python3
"""The phone stage gdnk2: the variants of the stage gdnk one by one, and the probes of the state path of
version 2 of the chunked gated delta net kernel.

Usage:
    probe.py commands [--out PATH]    write the phone command file (build/gdnk2/phone-commands.txt)
    probe.py table [--root DIR]       print the results from the pulled logs (build/gdnk2/phone-out)

The library set of this stage (GDNK_STAGE=build/gdnk2 tools/stages/gdnk/build.sh) holds the patches of
the stage gdnk and one more diagnostic patch. That patch adds the host switch GGML_HEXAGON_GDN_PROBE,
which the host gives to GATED_DELTA_NET in kernel_params[7]. Version 2 of the chunked kernel then does:
    1  logs the count of values that are not finite, and the largest magnitude, of each buffer of row 0
       at each phase (FARF, thus a .farf mask file goes next to the libraries), with the gate scalars
    2  writes the final state with vector stores through the cache, not with the DMA
    4  skips the update of the state by the last chunk (the output state is then wrong, but the run
       shows if the update makes the values that are not finite)
    8  waits 2 M cycles before that update (a late HMX store)
   16  makes each f16 tile of a scaled row (gdn_c2_pair_f16) through f32 and the add of zero, not from
       the qfloat product directly
   32  as 16, and a value below 2^-25 goes to zero before the f16 conversion

The variants (all with the DMA path of GDN_CONV_CHUNK):
    B  version 1 of the chunked kernel, the q/k norm as its own ops
    C  version 2, the q/k norm as its own ops
    S  the sequential kernel for each batch, the q/k norm inside the ops
    D  version 2, the q/k norm inside the ops (the preset of the patches)

The runs, in the order of the stage:
    tB tC     test-backend-ops test of GATED_DELTA_NET, GDN_CONV_STATE_FUSION, GDN_STATE_FUSION
    tS tD     test-backend-ops test of GATED_DELTA_NET and GDN_STATE_FUSION
    x1D x1C   probe 1 on the case of one chunk (4 heads of 64, 64 tokens, gates from -20), the log
    xmD       probe 1 on the case of four chunks (4 heads of 64, 256 tokens), the log
    x2D x4D x8D  probes 2, 4 and 8 on the two cases
    x16D x32D x16C  probes 16 and 32 on the two cases

A run name is the stem of its output files in the phone directory out/: <name>-gate.txt,
<name>.out, <name>.log, and <name>-logcat.txt for the runs with the log. This file is
tools/stages/gdnk/probe.py, and build/gdnk2/probe.py is a link to it.
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage as S  # noqa: E402

PHONE = "/data/local/tmp/qwen/gdnk2"
LAPTOP_STAGE = "build/gdnk2"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/gdnk2"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
LIB_ENV = (f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib "
           "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1")
VARIANTS = {
    "B": "GGML_HEXAGON_GDN_CONV_DMA=1 GGML_HEXAGON_GDN_CHUNK=1 GGML_HEXAGON_GDN_QKNORM=0",
    "C": "GGML_HEXAGON_GDN_CONV_DMA=1 GGML_HEXAGON_GDN_CHUNK=2 GGML_HEXAGON_GDN_QKNORM=0",
    "S": "GGML_HEXAGON_GDN_CONV_DMA=1 GGML_HEXAGON_GDN_CHUNK=0 GGML_HEXAGON_GDN_QKNORM=1",
    "D": "GGML_HEXAGON_GDN_CONV_DMA=1 GGML_HEXAGON_GDN_CHUNK=2 GGML_HEXAGON_GDN_QKNORM=1",
}
# The two GATED_DELTA_NET cases of the probes (vars of test_gated_delta_net, gates from -20)
ONE = "head_count=4,head_size=64,n_seq_tokens=64,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1$"
MULTI = "head_count=4,head_size=64,n_seq_tokens=256,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1$"
T_ALL = "test -b HTP0 -o GATED_DELTA_NET,GDN_CONV_STATE_FUSION,GDN_STATE_FUSION"
T_GDN = "test -b HTP0 -o GATED_DELTA_NET,GDN_STATE_FUSION"
T_ONE = f"test -b HTP0 -o GATED_DELTA_NET -p \"{ONE}\""
T_MULTI = f"test -b HTP0 -o GATED_DELTA_NET -p \"{MULTI}\""
T_BOTH = f"test -b HTP0 -o GATED_DELTA_NET -p \"{ONE}|{MULTI}\""
# name, variant, probe bits, arguments, log capture, text
RUNS = (
    ("tB", "B", 0, T_ALL, False, "the tests of the gated delta net ops, version 1"),
    ("tC", "C", 0, T_ALL, False, "the tests of the gated delta net ops, version 2"),
    ("tS", "S", 0, T_GDN, False, "GATED_DELTA_NET and the state step, the sequential kernel with the norm"),
    ("tD", "D", 0, T_GDN, False, "GATED_DELTA_NET and the state step, version 2 with the norm"),
    ("x1D", "D", 1, T_ONE, True, "the log of each phase, one chunk, version 2 with the norm"),
    ("x1C", "C", 1, T_ONE, True, "the log of each phase, one chunk, version 2"),
    ("xmD", "D", 1, T_MULTI, True, "the log of each phase, four chunks, version 2 with the norm"),
    ("x2D", "D", 2, T_BOTH, False, "the final state through the cache"),
    ("x4D", "D", 4, T_BOTH, False, "no update of the state by the last chunk"),
    ("x8D", "D", 8, T_BOTH, False, "a wait of 2 M cycles before that update"),
    ("x16D", "D", 16, T_BOTH, False, "the scaled f16 tiles through f32 and the add of zero"),
    ("x32D", "D", 32, T_BOTH, False, "as 16, and values below 2^-25 to zero first"),
    ("x16C", "C", 16, T_BOTH, False, "the scaled f16 tiles through f32 and the add of zero"),
)
LIMIT = 100
STAGE_FILES = ("bin/gate.sh", "bin/test-backend-ops", "lib/libggml-base.so", "lib/libggml-cpu.so",
               "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so")

HEADER = """\
# Phone stage "gdnk2": the variants of the stage gdnk one by one, and the probes of the state path of version 2
# of the chunked gated delta net kernel (tools/stages/gdnk/probe.py). One library set: the patches of the stage
# gdnk plus a diagnostic patch (the switch GGML_HEXAGON_GDN_PROBE). 13 runs of test-backend-ops, NO-MODEL:
#   tB tC tS tD     the tests of the gated delta net ops for the variants B (version 1), C (version 2),
#                   S (the sequential kernel with the q/k norm inside) and D (version 2 with the norm)
#   x1D x1C xmD     probe 1: the DSP log of each phase of version 2 (logcat, a .farf mask file in lib/)
#   x2D x4D x8D     probes 2, 4 and 8 of the final state
#   x16D x32D x16C  probes 16 and 32: the f16 conversion of the scaled tiles
# Each run: the thermal line, then bin/gate.sh, test-backend-ops under timeout -s KILL 100, the exit code and the
# conditions after the run, then the pgrep line. It is a check stage, not a timing stage.
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 5 minutes of tool time plus the
# gates. Then: build/gdnk2/probe.py table
"""


def run_lines(name: str, variant: str, probe: int, args: str, logcat: bool, text: str) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{PHONE}/out/{name}"
    env = " ".join(x for x in (LIB_ENV, VARIANTS[variant], f"GGML_HEXAGON_GDN_PROBE={probe}" if probe else "") if x)
    pre = "logcat -c && " if logcat else ""
    post = f"logcat -d -t 20000 | grep gdn-probe > {stem}-logcat.txt; " if logcat else ""
    cmd = (f"sh {PHONE}/bin/gate.sh {S.TEST_KB} > {stem}-gate.txt && {S.BEFORE} >> {stem}-gate.txt && {pre}"
           f"timeout -s KILL {LIMIT} env {env} {PHONE}/bin/test-backend-ops {args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {post}{S.AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    return ["#", f"# NO-MODEL: {name}, {text}, variant {variant}", S.THERMAL, f"{S.ADB} shell '{cmd}'", S.PGREP]


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files, with the .farf mask file."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        f"{S.ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out'",
        f"{S.ADB} push {bins} {PHONE}/bin/",
        f"{S.ADB} push {libs} {PHONE}/lib/",
        f"{S.ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{S.ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS 2>/dev/null | grep -c OK; chmod 755 {PHONE}/bin/*; "
        f"echo 0x1f > {PHONE}/lib/test-backend-ops.farf; echo 0x1f > {PHONE}/bin/test-backend-ops.farf'",
    ]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        f"{S.ADB} shell 'ls {PHONE}/out | wc -l'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{S.ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({S.ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {S.ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = HEADER.rstrip("\n").split("\n") + setup_lines()
    for run in RUNS:
        lines += run_lines(*run)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


FAIL_RE = re.compile(r"^\s+([A-Z_]+\(.*\)): .*FAIL", re.M)
DETAIL_RE = re.compile(r"(NaN at index .*|ERR = [^\n]*|Inf at index .*)")


def table(root: Path) -> int:
    """Print the exit code, the pass counts, the failed cases with their first detail line, and the log
    lines of each run. O(size of the logs)."""
    if not root.is_dir():
        print(f"probe.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    for name, variant, probe, _, logcat, text in RUNS:
        gate = root / f"{name}-gate.txt"
        if not gate.exists():
            print(f"{name}: no run")
            continue
        g = gate.read_text(errors="replace")
        rc = re.search(r"^rc=(\d+)", g, re.M)
        out = ""
        for suffix in (".out", ".log"):
            p = root / f"{name}{suffix}"
            out += p.read_text(errors="replace") if p.exists() else ""
        passed = S.PASSED_RE.findall(out)
        print(f"{name} ({variant}, probe {probe}, {text}): exit {rc.group(1) if rc else '?'}, "
              f"passed {', '.join('/'.join(x) for x in passed) or '?'}")
        lines = out.splitlines()
        for k, line in enumerate(lines):
            m = FAIL_RE.match(line)
            if not m:
                continue
            detail = next((d.group(1) for d in (DETAIL_RE.search(x) for x in lines[max(0, k - 3):k + 1]) if d), "")
            print(f"    FAIL {m.group(1)[:160]}  {detail}")
        if logcat:
            lc = root / f"{name}-logcat.txt"
            for line in (lc.read_text(errors="replace").splitlines() if lc.exists() else ["no logcat file"]):
                pos = line.find("gdn-probe:")
                print(f"    {line[pos:] if pos >= 0 else line}")
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
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(RUNS)} runs")
        return 0
    return table(a.root)


if __name__ == "__main__":
    sys.exit(main())
