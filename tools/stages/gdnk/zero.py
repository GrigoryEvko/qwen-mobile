#!/usr/bin/env python3
"""The phone stage gdnk3: patch 0005 of the chunked gated delta net kernel against HEAD, with the qfloat
probe of a zero operand.

Usage:
    zero.py commands [--out PATH]                 write the phone command file (build/gdnk3/phone-commands.txt)
    zero.py table [--root DIR] [--reject NAMES]   print the results from the pulled logs (build/gdnk3/phone-out)

Patch 0005 (wip/gdnk/0005-*.patch) makes the paired qfloat forward substitution of version 2 skip each
term with a zero coefficient, thus each chunk takes that solve, and the solve with the arithmetic of
version 1 goes. It also flushes the input state one time in phase begin, not at each update. It changes
only the DSP code (gdn-chunk-ops.c).

The three library sets (build/gdnk3/build-stage.sh, which runs tools/stages/gdnk/build.sh):
    b  HEAD plus patch 0005, each library and program: bin/ and lib/ on the phone
    a  HEAD, libggml-hexagon.so and libggml-htp-v79.so only: a/lib/ on the phone
    p  set b plus a diagnostic patch (build/gdnk3/probe.patch), the same two libraries: p/lib/ on the phone
The patch of set p adds the host switch GGML_HEXAGON_GDN_PROBE, which goes to GATED_DELTA_NET in
kernel_params[7]. Version 2 of the chunked kernel then does:
     1  logs the count of values that are not finite, and the largest magnitude, of each buffer of row 0
        at each phase, and the count of zero coefficients of the solve (FARF: a .farf mask file goes next
        to the DSP library)
    64  logs the qfloat forms around a zero operand: products with a zero operand and with a result near the
        end of the f32 range, sums and differences of such products, and sums of the form of the solve with
        zero coefficients and with the coefficient 2^-60 (lanes 0 to 7, with a float64 reference)

The variants:
    A  set a before set b: the backend of HEAD with the other libraries of set b. Patch 0005 does not change
       them, thus A is HEAD.
    B  set b: HEAD plus patch 0005, the preset switches
    C  set b with GGML_HEXAGON_GDN_QKNORM=0: version 2 without the q/k norm inside
    P  set p with the probe bits

A run name is the stem of its output files in the phone directory out/: <name>-gate.txt, <name>.out,
<name>.log, and <name>-logcat.txt for the runs with the log. The rule of the landing: tB and tC pass, the
KL of B is at the floor of A, and pp512 and pp1024 of B are not below A beyond the spread of A. This file
is tools/stages/gdnk/zero.py, and build/gdnk3/zero.py is a link to it.
"""

import argparse
import os
import re
import statistics
import sys
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage as S  # noqa: E402

PHONE = "/data/local/tmp/qwen/gdnk3"
LAPTOP_STAGE = "build/gdnk3"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/gdnk3"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
SETS = ("b", "a", "p")
FUSION = "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
VARIANTS = {
    "A": f"LD_LIBRARY_PATH={PHONE}/a/lib:{PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/a/lib {FUSION}",
    "B": f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib {FUSION}",
    "C": f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib {FUSION} GGML_HEXAGON_GDN_QKNORM=0",
    "P": f"LD_LIBRARY_PATH={PHONE}/p/lib:{PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/p/lib {FUSION}",
}
# The two GATED_DELTA_NET cases of the probes (vars of test_gated_delta_net, gates from -20)
ONE = "head_count=4,head_size=64,n_seq_tokens=64,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1$"
MULTI = "head_count=4,head_size=64,n_seq_tokens=256,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1$"
T_ALL = "test -b HTP0 -o GATED_DELTA_NET,GDN_CONV_STATE_FUSION,GDN_STATE_FUSION"
T_ONE = f"test -b HTP0 -o GATED_DELTA_NET -p \"{ONE}\""
T_MULTI = f"test -b HTP0 -o GATED_DELTA_NET -p \"{MULTI}\""
KL_PRE = f"-m {S.MODEL} {S.PPL_ARGS} --chunks 4 -b 512"
KL_DEC = f"-m {S.MODEL} {S.PPL_ARGS} --chunks 1 -b 1 -ub 1"
PERF = S.BLOCKS["f"].args
BENCH = S.BLOCKS["p"].args

Run = namedtuple("Run", "name variant probe tool args limit logcat text")


def runs() -> list[Run]:
    """The runs in the order of the stage: the probes, the tests, the KL, then the perf runs and the
    llama-bench runs. A and B alternate in their order from round to round, thus a slow drift of the clocks
    or the heat goes equally to each variant. O(runs)."""
    out = [
        Run("q1P", "P", 65, "test-backend-ops", T_ONE, 100, True,
            "probe 64 (the qfloat forms) and the log of each phase, one chunk"),
        Run("qmP", "P", 1, "test-backend-ops", T_MULTI, 100, True, "the log of each phase, four chunks"),
        Run("tC", "C", 0, "test-backend-ops", T_ALL, 110, False, "the tests of the gated delta net ops, no norm inside"),
        Run("tB", "B", 0, "test-backend-ops", T_ALL, 110, False, "the tests of the gated delta net ops, the preset"),
        Run("kpA", "A", 0, "llama-perplexity", KL_PRE, 100, False, "KL of the prefill path, 4 chunks of 512, -b 512"),
        Run("kpB", "B", 0, "llama-perplexity", KL_PRE, 100, False, "KL of the prefill path, 4 chunks of 512, -b 512"),
        Run("kdB", "B", 0, "llama-perplexity", KL_DEC, 110, False, "KL of the decode path, 1 chunk of 512, -b 1 -ub 1"),
    ]
    for rnd, order in ((1, "AB"), (2, "BA")):
        out += [Run(f"f-{rnd}-{v.lower()}", v, 0, "test-backend-ops", PERF, 100, False,
                    "test-backend-ops perf of GATED_DELTA_NET") for v in order]
    for rnd, order in ((1, "AB"), (2, "BA"), (3, "AB")):
        out += [Run(f"p-{rnd}-{v.lower()}", v, 0, "llama-bench", BENCH, 90, False,
                    "llama-bench pp512 and pp1024 at depth 0, 3 repetitions") for v in order]
    return out


def stage_files(name: str) -> list[str]:
    """The files of one library set: each line of <set>/phone/SHA256SUMS. O(files)."""
    sums = STAGE_DIR / name / "phone" / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"zero.py: {sums} does not exist. Build the stage first (build/gdnk3/build-stage.sh).")
    files = [line.split()[1] for line in sums.read_text().splitlines() if line.strip()]
    if not any(f.startswith("lib/") for f in files):
        sys.exit(f"zero.py: {sums} has no library")
    if name == "b" and "bin/test-backend-ops" not in files:
        sys.exit(f"zero.py: {sums} has no test-backend-ops")
    return files


HEADER = """\
# Phone stage "gdnk3": patch 0005 of the chunked gated delta net kernel (the forward substitution of version 2
# skips the zero terms, the input state is flushed one time) against HEAD, with the qfloat probe of a zero
# operand (tools/stages/gdnk/zero.py). Three library sets: b (HEAD plus 0005), a (the Hexagon backend of HEAD),
# p (set b plus the probe switch GGML_HEXAGON_GDN_PROBE). {n} runs:
#   q1P qmP         NO-MODEL, the probe: the qfloat forms around a zero operand (probe 64) and the DSP log of each
#                   phase (probe 1, logcat, a .farf mask file next to the DSP library), one chunk and four chunks
#   tC tB           NO-MODEL, the tests of the gated delta net ops: B (0005, the preset) and C (0005, no norm inside)
#   kpA kpB         REAL-MODEL, the KL of the prefill path (4 chunks of 512, -b 512) against the naive base
#   kdB             REAL-MODEL, the KL of the decode path (1 chunk, -b 1 -ub 1) against the naive base
#   f-R-a f-R-b     NO-MODEL, test-backend-ops perf of GATED_DELTA_NET, A and B, 2 rounds
#   p-R-a p-R-b     REAL-MODEL, llama-bench pp512 and pp1024 at depth 0, -r 3, A and B, 3 rounds (A B, B A, A B)
# The flags are those of the stage gdnk: HTP0, flash attention, Q8_0 K and V, -b 1024 -ub 1024, 4 threads, op
# fusion and state fusion on. Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on,
# thermal 0, no charger, the MemAvailable of the run), the tool under timeout -s KILL (110 s or less), the exit
# code and the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 15 minutes of tool time plus about
# 4 minutes of gates and checks, plus the waits for thermal status 0. Then: build/gdnk3/zero.py table
"""


def setup_lines() -> list[str]:
    """The lines that copy the three library sets to the phone and check each file, with the .farf mask
    files of the probe."""
    lines = []
    for name in SETS:
        lines.append(f"mkdir -p {LAPTOP_STAGE}/{name} && rsync -a --delete {BOX}/{name}/phone/ "
                     f"{LAPTOP_STAGE}/{name}/phone/")
        lines.append(f"(cd {LAPTOP_STAGE}/{name}/phone && sha256sum -c SHA256SUMS)")
    # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
    lines.append(f"{S.ADB} shell 'ls -l /data/local/tmp/qwen/models | grep 4B-Q8_0.gguf; "
                 f"ls -l {S.EVAL} | grep -E \"naive-4B-q8|wiki.test\"'")
    lines.append(f"{S.ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/a/lib {PHONE}/p/lib "
                 f"{PHONE}/out'")
    count = 0
    for name in SETS:
        files = stage_files(name)
        count += len(files)
        root = PHONE if name == "b" else f"{PHONE}/{name}"
        bins = " ".join(f"{LAPTOP_STAGE}/{name}/phone/{f}" for f in files if f.startswith("bin/"))
        libs = " ".join(f"{LAPTOP_STAGE}/{name}/phone/{f}" for f in files if f.startswith("lib/"))
        if bins:
            lines.append(f"{S.ADB} push {bins} {root}/bin/")
        lines.append(f"{S.ADB} push {libs} {root}/lib/")
        lines.append(f"{S.ADB} push {LAPTOP_STAGE}/{name}/phone/SHA256SUMS {root}/")
    # each file must check, thus the count of OK lines must be the count of the files
    lines.append(f"{S.ADB} shell '(cd {PHONE} && sha256sum -c SHA256SUMS; cd {PHONE}/a && sha256sum -c SHA256SUMS; "
                 f"cd {PHONE}/p && sha256sum -c SHA256SUMS) | grep -c OK; echo {count} files; chmod 755 {PHONE}/bin/*; "
                 f"echo 0x1f > {PHONE}/p/lib/test-backend-ops.farf; echo 0x1f > {PHONE}/bin/test-backend-ops.farf'")
    return lines


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{PHONE}/out/{run.name}"
    model = run.tool != "test-backend-ops"
    env = VARIANTS[run.variant] + (f" GGML_HEXAGON_GDN_PROBE={run.probe}" if run.probe else "")
    pre = "logcat -c && " if run.logcat else ""
    post = f"logcat -d -t 20000 | grep gdn-probe > {stem}-logcat.txt; " if run.logcat else ""
    cmd = (f"sh {PHONE}/bin/gate.sh {S.MODEL_KB if model else S.TEST_KB} > {stem}-gate.txt && "
           f"{S.BEFORE} >> {stem}-gate.txt && {pre}"
           f"timeout -s KILL {run.limit} env {env} {PHONE}/bin/{run.tool} {run.args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {post}{S.AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    # The runner gates each line with "models/Qwen3.5" as a model run
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if model else "NO-MODEL"
    return ["#", f"# {title}: {run.name}, {run.text}, variant {run.variant}", S.THERMAL,
            f"{S.ADB} shell '{cmd}'", S.PGREP]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone
    directory goes only when the pull has each of its files."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        S.THERMAL,
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
    all_runs = runs()
    lines = HEADER.format(n=len(all_runs)).rstrip("\n").split("\n") + setup_lines()
    for run in all_runs:
        lines += run_lines(run)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The table ----

# test-backend-ops writes "[OP] detail   CASE(vars): FAIL" on one line, the FAIL with color codes
FAIL_RE = re.compile(r"([A-Z_]+\([^)]*\)): \S*FAIL")
DETAIL_RE = re.compile(r"(NaN at index \S+ \([^)]*\)|ERR = \S+ > \S+|Inf at index \S+)")
Named = namedtuple("Named", "name")


def rate_rows(results: dict, rejected: set[str]) -> list[str]:
    """The t/s of pp512 and pp1024 for A and B: the median of the rounds, the paired difference, the
    lowest and the highest round, and the rule: the median of B is not below the lowest round of A. A run
    counts when its exit code is 0 and its name is not in rejected. O(runs)."""
    out = ["llama-bench t/s: the median of the rounds, [the lowest and the highest round], n, and the paired "
           "difference of B to A"]
    for text, key in (("pp512", (512, 0, 0)), ("pp1024", (1024, 0, 0))):
        per = {"a": {}, "b": {}}
        for rnd in (1, 2, 3):
            for v in "ab":
                res = results.get(f"p-{rnd}-{v}")
                if res is None or not res.ok or res.run.name in rejected:
                    continue
                val = S.bench_values(res).get(key)
                if val is not None:
                    per[v][rnd] = val
        cells = []
        for v in "ab":
            vals = per[v]
            cells.append(f"{v.upper()} " + (f"{statistics.median(vals.values()):.2f} [{min(vals.values()):.1f}-"
                                            f"{max(vals.values()):.1f}] n{len(vals)}" if vals else "-"))
        ratios = [per["b"][r] / per["a"][r] for r in per["b"] if r in per["a"]]
        diff = f"{100 * (statistics.median(ratios) - 1):+.1f}%" if ratios else "?"
        rule = "?"
        if per["a"] and per["b"]:
            rule = "PASS" if statistics.median(per["b"].values()) >= min(per["a"].values()) else "FAIL"
        out.append(f"  {text:7s} {cells[0]:32s} {cells[1]:32s} B/A {diff:7s} rule {rule}")
    return out


def table(root: Path, rejected: set[str]) -> int:
    """Print the exit code, the pass counts, the failed cases, the probe lines, the KL, the perf and the
    rates of the stage. O(size of the logs)."""
    if not root.is_dir():
        print(f"zero.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {}
    perf: dict[str, dict[str, list[float]]] = {}
    for run in runs():
        if not (root / f"{run.name}-gate.txt").exists():
            print(f"{run.name}: no run")
            continue
        res = S.read_result(root, Named(run.name))
        results[run.name] = res
        text = res.out + res.log
        passed = S.PASSED_RE.findall(text)
        kl, top, mx = S.KLD_RE.search(text), S.TOP_RE.search(text), S.MAXKL_RE.search(text)
        kl_text = (f", KL {kl.group(1)} ± {kl.group(2)}, max {mx.group(1) if mx else '?'}, "
                   f"top-1 {top.group(1) if top else '?'} %") if kl else ""
        mark = " [rejected]" if run.name in rejected else ""
        flags = f" [{', '.join(res.flags)}]" if res.flags else ""
        print(f"{run.name} ({run.variant}, {run.text}): ok {res.ok}, passed "
              f"{', '.join('/'.join(x) for x in passed) or '-'}{kl_text}{flags}{mark}")
        for line in text.splitlines():
            m = FAIL_RE.search(line)
            if m:
                detail = DETAIL_RE.search(line)
                print(f"    FAIL {m.group(1)[:160]}  {detail.group(1)[:80] if detail else ''}")
        if run.name.startswith("f-") and res.ok and run.name not in rejected:
            for case, _, us in S.PERF_RE.findall(text):
                perf.setdefault(case, {}).setdefault(run.variant, []).append(float(us))
        if run.logcat:
            lc = root / f"{run.name}-logcat.txt"
            for line in (lc.read_text(errors="replace").splitlines() if lc.exists() else ["no logcat file"]):
                pos = line.find("gdn-probe:")
                print(f"    {line[pos:] if pos >= 0 else line}")
    print()
    print("test-backend-ops perf, us per run: the median of the rounds")
    for case, per in perf.items():
        a = statistics.median(per["A"]) if per.get("A") else None
        b = statistics.median(per["B"]) if per.get("B") else None
        diff = f" {100 * (b / a - 1):+.1f}%" if a and b else ""
        print(f"  {case[:120]}")
        print(f"    A {a if a is None else round(a)} | B {b if b is None else round(b)}{diff}")
    print()
    print("\n".join(rate_rows(results, rejected)))
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the results from the pulled logs")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    t.add_argument("--reject", default="", help="comma-separated run names that the runner marked "
                   "CAPS-CHANGED or SCREEN-OFF: the perf and rate tables do not use them")
    a = ap.parse_args()
    if a.cmd == "commands":
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(runs())} runs")
        return 0
    return table(a.root, {x for x in a.reject.split(",") if x})


if __name__ == "__main__":
    sys.exit(main())
