#!/usr/bin/env python3
"""The phone stages gdnk3 and gdnk3p: patch 0005 of the chunked gated delta net kernel against HEAD.

Usage:
    zero.py [--stage full|rates] commands [--out PATH]
    zero.py [--stage full|rates] table [--root DIR] [--reject NAMES]

The stage "full" (gdnk3) has the qfloat probe of a zero operand, the tests, the KL, the perf of the op and
3 rounds of prefill rates. The stage "rates" (gdnk3p) has only the prefill rates, 5 rounds. commands writes
the phone command file (build/gdnk3/phone-commands.txt or phone-commands-p.txt). table prints the results
from the pulled logs (build/gdnk3/phone-out or phone-out-p). --reject takes the comma-separated names of the
runs that the runner marked CAPS-CHANGED or SCREEN-OFF: the perf and rate tables do not use them.

Patch 0005 of the stage, which patches/hexagon-gdn/0013 holds, makes the paired qfloat forward
substitution of version 2 skip each term with a zero coefficient, thus each chunk takes that solve, and
the solve with the arithmetic of version 1 goes. It also flushes the input state one time in phase begin,
not at each update. It changes only the DSP code (gdn-chunk-ops.c).

The three library sets (build/gdnk3/build-stage.sh, which runs tools/stages/gdnk/build.sh):
    b  HEAD plus patch 0005, each library and program: bin/ and lib/ on the phone
    a  HEAD, libggml-hexagon.so and libggml-htp-v79.so only: a/lib/ on the phone
    p  set b plus a diagnostic patch (build/gdnk3/probe.patch), the same two libraries: p/lib/ on the phone.
       Only the stage "full" uses it.
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
KL of B is at the floor of A, and the median pp512 and pp1024 of B are not below the lowest round of A.
This file is tools/stages/gdnk/zero.py, and build/gdnk3/zero.py is a link to it.
"""

import argparse
import os
import re
import statistics
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage as S  # noqa: E402
from common import device, parse  # noqa: E402

LAPTOP_STAGE = "build/gdnk3"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/gdnk3"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
FUSION = "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
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
BENCH_TEXT = "llama-bench pp512 and pp1024 at depth 0, 3 repetitions"

Run = namedtuple("Run", "name variant probe tool args limit logcat text")


def rate_runs(rounds: int, limit: int) -> list[Run]:
    """The llama-bench runs: A B in an odd round and B A in an even round, thus a slow drift of the clocks
    or the heat goes equally to each variant. O(rounds)."""
    return [Run(f"p-{rnd}-{v.lower()}", v, 0, "llama-bench", BENCH, limit, False, BENCH_TEXT)
            for rnd in range(1, rounds + 1) for v in ("AB" if rnd % 2 else "BA")]


def full_runs() -> list[Run]:
    """The runs of the stage "full", in order: the probes, the tests, the KL, then the perf runs and the
    llama-bench runs. O(runs)."""
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
    return out + rate_runs(3, 90)


FULL_HEADER = """\
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

RATES_HEADER = """\
# Phone stage "gdnk3p": the prefill rates of patch 0005 of the chunked gated delta net kernel (the forward
# substitution of version 2 skips the zero terms, the input state is flushed one time) against HEAD
# (tools/stages/gdnk/zero.py --stage rates). The library sets b (HEAD plus 0005) and a (the Hexagon backend of
# HEAD) of the stage gdnk3. {n} runs:
#   p-R-a p-R-b     REAL-MODEL, llama-bench pp512 and pp1024 at depth 0, -r 3, A and B, 5 rounds (A B, B A, A B,
#                   B A, A B)
# The flags are those of the stage gdnk: HTP0, flash attention, Q8_0 K and V, -b 1024 -ub 1024, 4 threads, op
# fusion and state fusion on. Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on,
# thermal 0, no charger, the MemAvailable of the run), the tool under timeout -s KILL (60 s), the exit code and
# the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: at most 10 minutes of tool time (each run
# has a limit of 60 s; the whole stage gdnk3, 17 runs with 6 runs of this form, took about 5 minutes), plus about
# 2 minutes of gates and checks, plus the waits for thermal status 0. Then: build/gdnk3/zero.py --stage rates table
"""


@dataclass(frozen=True)
class Stage:
    """One phone stage: its phone directory, library sets, runs, header and file names."""
    phone: str
    sets: tuple
    runs: tuple
    header: str
    commands: str
    out: str

    def env(self, variant: str) -> str:
        """The environment of one variant."""
        p = self.phone
        return {
            "A": f"LD_LIBRARY_PATH={p}/a/lib:{p}/lib ADSP_LIBRARY_PATH={p}/a/lib {FUSION}",
            "B": f"LD_LIBRARY_PATH={p}/lib ADSP_LIBRARY_PATH={p}/lib {FUSION}",
            "C": f"LD_LIBRARY_PATH={p}/lib ADSP_LIBRARY_PATH={p}/lib {FUSION} GGML_HEXAGON_GDN_QKNORM=0",
            "P": f"LD_LIBRARY_PATH={p}/p/lib:{p}/lib ADSP_LIBRARY_PATH={p}/p/lib {FUSION}",
        }[variant]


STAGES = {
    "full": Stage("/data/local/tmp/qwen/gdnk3", ("b", "a", "p"), tuple(full_runs()), FULL_HEADER,
                  "phone-commands.txt", "phone-out"),
    "rates": Stage("/data/local/tmp/qwen/gdnk3p", ("b", "a"), tuple(rate_runs(5, 60)), RATES_HEADER,
                   "phone-commands-p.txt", "phone-out-p"),
}


def stage_files(name: str) -> list[str]:
    """The files of one library set: each line of <set>/phone/SHA256SUMS. O(files)."""
    sums = STAGE_DIR / name / "phone" / "SHA256SUMS"
    if not sums.exists():
        sys.exit(f"zero.py: {sums} does not exist. Build the stage first (build/gdnk3/build-stage.sh).")
    files = [line.split()[1] for line in sums.read_text().splitlines() if line.strip()]
    if not any(f.startswith("lib/") for f in files):
        sys.exit(f"zero.py: {sums} has no library")
    if name == "b" and "bin/llama-bench" not in files:
        sys.exit(f"zero.py: {sums} has no llama-bench")
    return files


def setup_lines(st: Stage) -> list[str]:
    """The lines that copy the library sets of the stage to the phone and check each file, with the .farf
    mask files of the probe when the stage has set p."""
    phone = st.phone
    lines = []
    for name in st.sets:
        lines.append(f"mkdir -p {LAPTOP_STAGE}/{name} && rsync -a --delete {BOX}/{name}/phone/ "
                     f"{LAPTOP_STAGE}/{name}/phone/")
        lines.append(f"(cd {LAPTOP_STAGE}/{name}/phone && sha256sum -c SHA256SUMS)")
    # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
    lines.append(f"{S.ADB} shell 'ls -l /data/local/tmp/qwen/models | grep 4B-Q8_0.gguf; "
                 f"ls -l {S.EVAL} | grep -E \"naive-4B-q8|wiki.test\"'")
    dirs = " ".join([f"{phone}/bin", f"{phone}/lib", f"{phone}/out"] +
                    [f"{phone}/{name}/lib" for name in st.sets if name != "b"])
    lines.append(f"{S.ADB} shell 'rm -rf {phone} && mkdir -p {dirs}'")
    count = 0
    for name in st.sets:
        files = stage_files(name)
        count += len(files)
        root = phone if name == "b" else f"{phone}/{name}"
        bins = " ".join(f"{LAPTOP_STAGE}/{name}/phone/{f}" for f in files if f.startswith("bin/"))
        libs = " ".join(f"{LAPTOP_STAGE}/{name}/phone/{f}" for f in files if f.startswith("lib/"))
        if bins:
            lines.append(f"{S.ADB} push {bins} {root}/bin/")
        lines.append(f"{S.ADB} push {libs} {root}/lib/")
        lines.append(f"{S.ADB} push {LAPTOP_STAGE}/{name}/phone/SHA256SUMS {root}/")
    # each file must check, thus the count of OK lines must be the count of the files
    checks = "; ".join(f"cd {phone if name == 'b' else f'{phone}/{name}'} && sha256sum -c SHA256SUMS"
                       for name in st.sets)
    farf = (f"; echo 0x1f > {phone}/p/lib/test-backend-ops.farf; echo 0x1f > {phone}/bin/test-backend-ops.farf"
            if "p" in st.sets else "")
    lines.append(f"{S.ADB} shell '({checks}) | grep -c OK; echo {count} files; chmod 755 {phone}/bin/*{farf}'")
    return lines


def run_lines(st: Stage, run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{st.phone}/out/{run.name}"
    model = run.tool != "test-backend-ops"
    env = st.env(run.variant) + (f" GGML_HEXAGON_GDN_PROBE={run.probe}" if run.probe else "")
    pre = "logcat -c && " if run.logcat else ""
    post = f"logcat -d -t 20000 | grep gdn-probe > {stem}-logcat.txt; " if run.logcat else ""
    cmd = (f"sh {st.phone}/bin/gate.sh {S.MODEL_KB if model else S.TEST_KB} > {stem}-gate.txt && "
           f"{device.BEFORE} >> {stem}-gate.txt && {pre}"
           f"timeout -s KILL {run.limit} env {env} {st.phone}/bin/{run.tool} {run.args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {post}{device.AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    # The runner gates each line with "models/Qwen3.5" as a model run
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if model else "NO-MODEL"
    return ["#", f"# {title}: {run.name}, {run.text}, variant {run.variant}", device.THERMAL,
            f"{S.ADB} shell '{cmd}'", S.PGREP]


def output_lines(st: Stage) -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone
    directory goes only when the pull has each of its files."""
    phone, out = st.phone, f"{LAPTOP_STAGE}/{st.out}"
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        device.THERMAL,
        f"{S.ADB} shell 'ls {phone}/out | wc -l'",
        f"rm -rf {out}",
        f"{S.ADB} pull {phone}/out {out}",
        f"rsync -a --delete {out}/ {BOX}/{st.out}/",
        f"test \"$(ls {out} | wc -l)\" -eq \"$({S.ADB} shell 'ls {phone}/out | wc -l' | tr -d '\\r')\" "
        f"&& {S.ADB} shell 'rm -rf {phone}' && echo removed {phone}",
    ]


def write_commands(st: Stage, path: Path) -> int:
    """Write the command file and return its line count."""
    lines = st.header.format(n=len(st.runs)).rstrip("\n").split("\n") + setup_lines(st)
    for run in st.runs:
        lines += run_lines(st, run)
    lines += output_lines(st)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The table ----

# test-backend-ops writes "[OP] detail   CASE(vars): FAIL" on one line, the FAIL with color codes
FAIL_RE = re.compile(r"([A-Z_]+\([^)]*\)): \S*FAIL")
DETAIL_RE = re.compile(r"(NaN at index \S+ \([^)]*\)|ERR = \S+ > \S+|Inf at index \S+)")
P_RE = re.compile(r"^p-(\d+)-([ab])$")
Named = namedtuple("Named", "name")


def rate_rows(results: dict, rejected: set[str]) -> list[str]:
    """The t/s of pp512 and pp1024 for A and B: the value of each round, the median of the rounds, the
    paired difference, the lowest and the highest round, and the rule: the median of B is not below the
    lowest round of A. A run counts when its exit code is 0 and its name is not in rejected. O(runs)."""
    out = ["llama-bench t/s: the median of the rounds, [the lowest and the highest round], n, and the paired "
           "difference of B to A"]
    for text, key in (("pp512", (512, 0, 0)), ("pp1024", (1024, 0, 0))):
        per = {"a": {}, "b": {}}
        for name, res in results.items():
            m = P_RE.match(name)
            if m is None or not res.ok or name in rejected:
                continue
            val = parse.bench_values(res.out).get(key)
            if val is not None:
                per[m.group(2)][int(m.group(1))] = val
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
        rounds = sorted(set(per["a"]) | set(per["b"]))
        out.append("          rounds: " + ", ".join(
            f"{r}: A {per['a'].get(r, float('nan')):.1f} B {per['b'].get(r, float('nan')):.1f}" for r in rounds))
    return out


def table(st: Stage, root: Path, rejected: set[str]) -> int:
    """Print the exit code, the pass counts, the failed cases, the probe lines, the KL, the perf and the
    rates of the stage. O(size of the logs)."""
    if not root.is_dir():
        print(f"zero.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {}
    perf: dict[str, dict[str, list[float]]] = {}
    for run in st.runs:
        if not (root / f"{run.name}-gate.txt").exists():
            print(f"{run.name}: no run")
            continue
        res = S.read_result(root, Named(run.name))
        results[run.name] = res
        text = res.out + res.log
        passed = S.PASSED_RE.findall(text)
        kl, top, mx = parse.KLD_RE.search(text), parse.TOP_RE.search(text), parse.MAXKL_RE.search(text)
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
    if perf:
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
    ap.add_argument("--stage", choices=sorted(STAGES), default="full", help="the stage (preset: full)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, help="the command file (preset: build/gdnk3/<file of the stage>)")
    t = sub.add_parser("table", help="print the results from the pulled logs")
    t.add_argument("--root", type=Path, help="the pulled outputs (preset: build/gdnk3/<directory of the stage>)")
    t.add_argument("--reject", default="", help="comma-separated run names that the runner marked "
                   "CAPS-CHANGED or SCREEN-OFF: the perf and rate tables do not use them")
    a = ap.parse_args()
    st = STAGES[a.stage]
    if a.cmd == "commands":
        path = a.out or STAGE_DIR / st.commands
        n = write_commands(st, path)
        print(f"{path}: {n} lines, {len(st.runs)} runs")
        return 0
    return table(st, a.root or STAGE_DIR / st.out, {x for x in a.reject.split(",") if x})


if __name__ == "__main__":
    sys.exit(main())
