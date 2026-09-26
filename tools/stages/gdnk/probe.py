#!/usr/bin/env python3
"""The phone stage gdnk2: the check of a correction of version 2 of the chunked gated delta net kernel,
with the probes of its state path.

Usage:
    probe.py commands [--out PATH]    write the phone command file (build/gdnk2/phone-commands.txt)
    probe.py table [--root DIR]       print the results from the pulled logs (build/gdnk2/phone-out)

The library set of this stage (GDNK_STAGE=build/gdnk2 tools/stages/gdnk/build.sh) holds the patches of
the stage gdnk and one more diagnostic patch. That patch adds the host switch GGML_HEXAGON_GDN_PROBE,
which the host gives to GATED_DELTA_NET in kernel_params[7]. Version 2 of the chunked kernel then does:
    1  logs the count of values that are not finite, and the largest magnitude, of each buffer of row 0
       at each phase (FARF, thus a .farf mask file goes next to the libraries), with the gate scalars
    2  writes the final state with vector stores through the cache, not with the DMA
    4  skips the update of the state by the last chunk
    8  waits 2 M cycles before that update (a late HMX store)
   64  logs the qfloat forms around a zero operand on this core (the sum of a product by zero, and
       the conversions to f32 and f16), one time for each op

The variants:
    A  the shipped paths (no DMA path of GDN_CONV_CHUNK, version 1, the q/k norm as its own ops)
    B  version 1 of the chunked kernel, the q/k norm as its own ops
    C  version 2, the q/k norm as its own ops
    S  the sequential kernel for each batch, the q/k norm inside the ops
    D  version 2, the q/k norm inside the ops (the preset of the patches)
B to D have the DMA path of GDN_CONV_CHUNK.

The runs (RUNS), in the order of the stage: the tests of the gated delta net ops for C and D, the log of
each phase for one chunk (with the qfloat forms of probe 64) and for four chunks, the KL of the prefill
path (A, C, D) and of the decode path (D) against the naive base, and the test-backend-ops perf of
GATED_DELTA_NET (stage.PERF_CASES) for B, C and D.

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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage as S  # noqa: E402
from common import device, parse  # noqa: E402

PHONE = "/data/local/tmp/qwen/gdnk2"
LAPTOP_STAGE = "build/gdnk2"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/gdnk2"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
LIB_ENV = f"{device.lib_env(PHONE)} GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
VARIANTS = {
    "A": "GGML_HEXAGON_GDN_CONV_DMA=0 GGML_HEXAGON_GDN_CHUNK=1 GGML_HEXAGON_GDN_QKNORM=0",
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
KL_PRE = f"-m {S.MODEL} {S.PPL_ARGS} --chunks 4 -b 512"
KL_DEC = f"-m {S.MODEL} {S.PPL_ARGS} --chunks 1 -b 1 -ub 1"
PERF = f"perf -b HTP0 -o GATED_DELTA_NET -p \"{S.PERF_CASES}\""
# name, variant, probe bits, tool, arguments, log capture, text
RUNS = (
    ("tC", "C", 0, "test-backend-ops", T_ALL, False, "the tests of the gated delta net ops, version 2"),
    ("tD", "D", 0, "test-backend-ops", T_GDN, False, "GATED_DELTA_NET and the state step, version 2 with the norm"),
    ("x1D", "D", 65, "test-backend-ops", T_ONE, True,
     "the log of each phase and the qfloat forms, one chunk, version 2 with the norm"),
    ("xmD", "D", 1, "test-backend-ops", T_MULTI, True, "the log of each phase, four chunks, version 2 with the norm"),
    ("kpA", "A", 0, "llama-perplexity", KL_PRE, False, "KL of the prefill path, 4 chunks of 512, -b 512"),
    ("kpC", "C", 0, "llama-perplexity", KL_PRE, False, "KL of the prefill path, 4 chunks of 512, -b 512"),
    ("kpD", "D", 0, "llama-perplexity", KL_PRE, False, "KL of the prefill path, 4 chunks of 512, -b 512"),
    ("kdD", "D", 0, "llama-perplexity", KL_DEC, False, "KL of the decode path, 1 chunk of 512, -b 1 -ub 1"),
    ("fB", "B", 0, "test-backend-ops", PERF, False, "the perf of GATED_DELTA_NET, version 1"),
    ("fC", "C", 0, "test-backend-ops", PERF, False, "the perf of GATED_DELTA_NET, version 2"),
    ("fD", "D", 0, "test-backend-ops", PERF, False, "the perf of GATED_DELTA_NET, version 2 with the norm"),
)
LIMIT = 100


def stage_files() -> list[str]:
    """The files of the stage: each line of phone/SHA256SUMS that the build wrote. The build recipe checks
    that phone/lib holds each llama, ggml and mtmd library that a program needs, thus the stage pushes
    all of them. O(files)."""
    sums = STAGE_DIR / "phone" / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"probe.py: {sums} does not exist. Build the stage first (GDNK_STAGE={LAPTOP_STAGE} "
                 "tools/stages/gdnk/build.sh).")
    files = [line.split()[1] for line in sums.read_text().splitlines() if line.strip()]
    if "bin/test-backend-ops" not in files or not any(f.startswith("lib/") for f in files):
        sys.exit(f"probe.py: {sums} has no test-backend-ops or no library")
    return files

HEADER = """\
# Phone stage "gdnk2": the check of version 2 of the chunked gated delta net kernel, with the solve of version 1
# arithmetic for a chunk with a zero coefficient (tools/stages/gdnk/probe.py). One library set: the patches of the
# stage gdnk plus a diagnostic patch (the switch GGML_HEXAGON_GDN_PROBE). 11 runs:
#   tC tD           NO-MODEL, the tests of the gated delta net ops for the variants C (version 2) and D (version 2
#                   with the q/k norm inside)
#   x1D xmD         NO-MODEL, probe 1: the DSP log of each phase of version 2 (logcat, a .farf mask file in lib/);
#                   x1D also logs the qfloat forms around a zero operand (probe 64)
#   kpA kpC kpD     REAL-MODEL, the KL of the prefill path (4 chunks of 512, -b 512) against the naive base
#   kdD             REAL-MODEL, the KL of the decode path (1 chunk, -b 1 -ub 1) against the naive base
#   fB fC fD        NO-MODEL, test-backend-ops perf of GATED_DELTA_NET (one token, 1024 tokens with gates from -20,
#                   and the 4B shape with gates from -0.5)
# Each run: the thermal line, then bin/gate.sh, the tool under timeout -s KILL (110 s or less), the exit code and
# the conditions after the run, then the pgrep line. It is a check stage; the perf runs are one round each.
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 9 minutes of tool time plus the
# gates. Then: build/gdnk2/probe.py table
"""


def run_lines(name: str, variant: str, probe: int, tool: str, args: str, logcat: bool, text: str) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{PHONE}/out/{name}"
    model = tool == "llama-perplexity"
    env = " ".join(x for x in (LIB_ENV, VARIANTS[variant], f"GGML_HEXAGON_GDN_PROBE={probe}" if probe else "") if x)
    pre = "logcat -c && " if logcat else ""
    post = f"logcat -d -t 20000 | grep gdn-probe > {stem}-logcat.txt; " if logcat else ""
    cmd = (f"sh {PHONE}/bin/gate.sh {S.MODEL_KB if model else S.TEST_KB} > {stem}-gate.txt && "
           f"{device.BEFORE} >> {stem}-gate.txt && {pre}"
           f"timeout -s KILL {110 if '-ub 1' in args else LIMIT} env {env} {PHONE}/bin/{tool} {args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {post}{device.AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    # The runner gates each line with "models/Qwen3.5" as a model run
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if model else "NO-MODEL"
    return ["#", f"# {title}: {name}, {text}, variant {variant}", device.THERMAL, f"{S.ADB} shell '{cmd}'", S.PGREP]


def setup_lines() -> list[str]:
    """The lines that copy each file of the stage to the phone and check it, with the .farf mask file."""
    files = stage_files()
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in files if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in files if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
        f"{S.ADB} shell 'ls -l /data/local/tmp/qwen/models | grep 4B-Q8_0.gguf; ls -l {S.EVAL} | grep -E \"naive-4B-q8|wiki.test\"'",
        f"{S.ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out'",
        f"{S.ADB} push {bins} {PHONE}/bin/",
        f"{S.ADB} push {libs} {PHONE}/lib/",
        f"{S.ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        # each of the files must check, thus the count of OK lines must be the count of the files
        f"{S.ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK; echo {len(files)} files; "
        f"chmod 755 {PHONE}/bin/*; echo 0x1f > {PHONE}/lib/test-backend-ops.farf; "
        f"echo 0x1f > {PHONE}/bin/test-backend-ops.farf'",
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


# test-backend-ops writes "[OP] detail   CASE(vars): FAIL" on one line, the FAIL with color codes
FAIL_RE = re.compile(r"([A-Z_]+\([^)]*\)): \S*FAIL")
DETAIL_RE = re.compile(r"(NaN at index \S+ \([^)]*\)|ERR = \S+ > \S+|Inf at index \S+)")


def table(root: Path) -> int:
    """Print the exit code, the pass counts, the failed cases with their first detail line, and the log
    lines of each run. O(size of the logs)."""
    if not root.is_dir():
        print(f"probe.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    for name, variant, probe, _, _, logcat, text in RUNS:
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
        kl, top, mx = parse.KLD_RE.search(out), parse.TOP_RE.search(out), parse.MAXKL_RE.search(out)
        kl_text = (f", KL {kl.group(1)} ± {kl.group(2)}, max {mx.group(1) if mx else '?'}, "
                   f"top-1 {top.group(1) if top else '?'} %") if kl else ""
        print(f"{name} ({variant}, probe {probe}, {text}): exit {rc.group(1) if rc else '?'}, "
              f"passed {', '.join('/'.join(x) for x in passed) or '-'}{kl_text}")
        for line in out.splitlines():
            m = FAIL_RE.search(line)
            if m:
                detail = DETAIL_RE.search(line)
                print(f"    FAIL {m.group(1)[:160]}  {detail.group(1)[:80] if detail else ''}")
        for case, runs, us in S.PERF_RE.findall(out):
            print(f"    perf {case[:150]}: {us} us per run, {runs} runs")
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
