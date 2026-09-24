#!/usr/bin/env python3
"""The phone stage "ttft": the time to the first token of a chat turn of the 4B Q8_0 on HTP0, old engine against new.

Usage:
    stage.py commands [--out PATH]         write the phone command file (build/ttft/phone-commands.txt)
    stage.py table [--root DIR] [--all]    print the tables from the pulled files (build/ttft/phone-out)

The tool of each run is memprobe (tools/memprobe/memprobe.cpp) in the context of the app (n_ctx 8192, 4 threads, 5 output
rows, the lazy token embedding, Q8_0 K and V, flash attention AUTO, the fused state step). Its switches select the
engine of the app before the changes of the chat turn (old) or after them (new):

    old: --render template --first-token late --embd-warm off --draft-file off --draft-passes all
    new: --render live --first-token early --embd-warm on --draft-file auto --draft-passes call

The template render with its snapshot before the generation prompt against the live form (chat_prompt.h), the first
token after its decode against before it, no read of the token embedding at the load against a read on a background
thread, the standard file against Qwen3.5-4B-Q8_0-draft32k.gguf next to it (the 32768-row draft head), and a drafter
that runs 4 MTP passes for each draft against one that stops at the draft length of the step.

A run name is 4b-<block>-<round>-<variant>, for example 4b-tn-2-s. Each run writes three files to the phone directory
out/: <name>-gate.txt (the conditions before and after the run and the exit code), <name>.out (the stdout of memprobe)
and <name>.log (its stderr with the STAMP lines).

A run goes into the tables when its gate passed, its exit code is 0, the thermal status after it is 0, no CPU cap before
or after it is less than 3.0 GHz (CAP_MIN_KHZ), and its log has no failure line. --all also uses the removed runs. The
table only reads files. O(size of the files) time.

This file is tools/stages/ttft/stage.py. The files of the stage stay in build/ttft.
"""

import argparse
import os
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/ttft"
MODEL_DIR = "/data/local/tmp/qwen/models"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
DRAFT_MODEL = "Qwen3.5-4B-Q8_0-draft32k.gguf"
# The sha256 of the draft-head file on the box (/home/grigory/airi/qwen-mobile-calib/Qwen3.5-4B-Q8_0-draft32k.gguf).
DRAFT_SHA256 = "f6526095f501bfb70e0877da528b69741d695b6b6cc41d9deb9102c72b54063b"
GATE_KB = 8388608
LAPTOP_STAGE = "build/ttft"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/ttft"
# The stage directory of the repository (tools/stages/ttft/stage.py is three levels below the root), relative to the
# working directory, for the default paths of the command file and of the outputs.
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
# The environment of the app (init_impl in llama_jni.cpp) and the libraries of the stage.
LIB_ENV = f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
# The context of the app (load_impl in llama_jni.cpp).
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"
STAGE_FILES = ("bin/gate.sh", "bin/memprobe", "lib/libggml-base.so", "lib/libggml-cpu.so", "lib/libggml-hexagon.so",
               "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so", "lib/libllama-common.so",
               "lib/libllama.so", "lib/libmtmd.so")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x memprobe; echo pgrep-done'"
# The highest temperature of the NPU thermal zones (type nsp*) in millidegrees, or nothing.
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')
# A run with a CPU cap of less than this value (kHz) before or after it stays out of the tables.
CAP_MIN_KHZ = 3000000

OLD = "--render template --first-token late --embd-warm off"
NEW = "--render live --first-token early --embd-warm on"


@dataclass(frozen=True)
class Variant:
    """One engine form: its key, its text, and its memprobe arguments."""
    key: str
    name: str
    args: str


@dataclass(frozen=True)
class Block:
    """One kind of run: the memprobe arguments, the variants, the round count, the time limit in seconds, whether
    the run needs an empty snapshot directory, and its text. The gate and the lines after the tool take about 8 s,
    thus each limit is 110 s or less."""
    key: str
    args: str
    variants: str
    rounds: int
    limit: int
    state_dir: bool
    text: str


VARIANTS = {v.key: v for v in (
    Variant("o", "old engine, draft off", OLD),
    Variant("n", "new engine, draft off", NEW),
    Variant("p", "old engine, draft on (the standard file, 4 MTP passes for each draft)",
            f"--spec {OLD} --draft-file off --draft-passes all"),
    Variant("s", "new engine, draft on (the draft-head file, the passes of the draft length)",
            f"--spec {NEW} --draft-file auto --draft-passes call"),
    Variant("k", "no read of the embedding, 3 s after the load", f"{NEW.replace('--embd-warm on', '--embd-warm off')} "
            "--turn-idle-ms 3000"),
    Variant("m", "the embedding read on a background thread, 3 s after the load", f"{NEW} --turn-idle-ms 3000"),
    Variant("i", "the embedding read on a background thread, at once after the load", f"{NEW} --turn-idle-ms 0"),
    Variant("d", "the draft graph of the draft-head file", f"--spec {NEW} --draft-file auto"),
    Variant("f", "the draft graph of the standard file", f"--spec {NEW} --draft-file off"),
)}
BLOCKS = {b.key: b for b in (
    Block("tn", "--turns 6 --turn-first 500 --turn-message 40 --turn-answer 64 --turn-idle-ms 1000 --therm", "onps", 3,
          100, True, "a chat of 6 turns: 500 tokens, then 5 messages of 40 tokens, answers of up to 64 tokens, "
          "1 s of idle before each turn"),
    Block("fm", "--drop-cache --turns 1 --turn-first 512 --turn-answer 8", "kmi", 2, 90, False,
          "the model file dropped from the page cache before the load, then a first message of 512 tokens"),
    Block("dh", "--draft-ops --turns 2 --turn-first 100 --turn-message 20 --turn-answer 32", "df", 1, 90, False,
          "a check, not a timing run: the matrix products that the MTP draft graph reads (DRAFTOP)"),
)}
# The blocks of one group run round by round. The variants run in their order in an odd round and in the reverse
# order in an even round, thus a slow drift of the clocks or the heat goes equally to each variant.
GROUPS = (("tn",), ("fm",), ("dh",))


@dataclass(frozen=True)
class Run:
    """One phone run of one block, round and variant."""
    block: Block
    round: int
    variant: Variant

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"4b-{self.block.key}-{self.round}-{self.variant.key}"


def all_runs() -> list[Run]:
    """The runs of the stage in their order. O(runs)."""
    out = []
    for group in GROUPS:
        for rnd in range(1, max(BLOCKS[k].rounds for k in group) + 1):
            for key in group:
                block = BLOCKS[key]
                if rnd > block.rounds:
                    continue
                order = block.variants if rnd % 2 else block.variants[::-1]
                out.extend(Run(block, rnd, VARIANTS[v]) for v in order)
    return out


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    b, v = run.block, run.variant
    stem = f"{PHONE}/out/{run.name}"
    state = f"{PHONE}/state"
    args = " ".join(x for x in (PROBE_ARGS, b.args, v.args, f"--state-dir {state}" if b.state_dir else "") if x)
    prep = f"rm -rf {state} && mkdir -p {state} && " if b.state_dir else ""
    clean = f"rm -rf {state}; " if b.state_dir else ""
    cmd = (f"sh {PHONE}/bin/gate.sh {GATE_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && {prep}"
           f"timeout -s KILL {b.limit} env {LIB_ENV} {PHONE}/bin/memprobe -m {MODEL_DIR}/{MODEL} {args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {clean}{AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    return ["#", f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {run.name}, {b.text}, {v.key}: {v.name}",
            THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "ttft": the time to the first token of a chat turn and the speculation step of the 4B Q8_0 on HTP0, in
# the engine of the app before the changes of the chat turn (old) and after them (new).
#
# The questions:
#   1. The time to the first token of a 40-token message (turns 2 to 6) and its parts, old against new, with the draft
#      off (o, n) and on (p, s). The stage fixed measured 512 ms (draft off) and 658 ms (draft on) for the old engine.
#   2. The speculation step: the time of one MTP pass with the full head (p) against the 32768-row draft head (s), the
#      passes of a draft against the draft length that the policy asked for, and the decode rate with the draft on
#      against off (s against n).
#   3. The first message of 512 tokens after a load with a cold page cache: without the read of the token embedding
#      (k), with it and 3 s of idle (m), with it and no idle (i). The load time of the three.
#   4. A check: the head that the draft pass reads on HTP0 (DRAFTOP), for the draft-head file (d) and the standard
#      file (f).
#
# The files (tools/stages/ttft/build.sh phone): the libraries of the app from the llama.cpp tree of HEAD with
# patches/spec-mtp/0001 (the MTP drafter stops at the draft length of the call), the DSP library v79, and memprobe
# with the sources of the app (chat_prompt.cpp, engine_tasks.cpp, state_cache.cpp, cache_io.cpp, spec_policy.cpp).
# The phone must hold Qwen3.5-4B-Q8_0.gguf and Qwen3.5-4B-Q8_0-draft32k.gguf in /data/local/tmp/qwen/models.
#
# The runs, 20:
#   tn  x12  a chat of 6 turns (500 tokens, then 40-token messages, answers of up to 64 tokens, 1 s of idle before each
#            turn), o n p s / s p n o / o n p s
#   fm  x6   --drop-cache, then a first message of 512 tokens: k m i / i m k
#   dh  x2   --draft-ops: d f
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB, and it prints the caps), the tool under timeout -s KILL (100 s or less), the exit code and the conditions after
# the run (thermal, caps, battery, NPU zone), then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, screen on, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 11 minutes of tool time plus about 8 s of
# gate and checks for each run, thus about 14 minutes, plus the waits for thermal status 0 and a battery of 38 C or
# less. The pull is about 2 MB. Then: tools/stages/ttft/stage.py table (on the box, or on the laptop with --root).
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files and the two models."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODEL_DIR} | grep -E \"Qwen3.5-4B-Q8_0(-draft32k)?.gguf\"'",
        f"{ADB} shell 'cd {MODEL_DIR} && echo \"{DRAFT_SHA256}  {DRAFT_MODEL}\" | timeout -s KILL 100 sha256sum -c'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out'",
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {libs} {PHONE}/lib/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
    ]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone directory goes
    only when the pull has each of its files."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'pgrep -x memprobe; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    runs = all_runs()
    lines = HEADER.rstrip("\n").split("\n") + setup_lines()
    lines += ["#", f"# ==== {MODEL.removesuffix('.gguf')}: {len(runs)} runs ===="]
    for run in runs:
        lines += run_lines(run)
    lines += output_lines()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The parser of the files ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
# A log line with the time stamp of --log-ts: minutes, seconds, milliseconds, microseconds, the level letter.
TS_RE = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) [A-Z] memprobe: STAMP (\S+)(.*)$")
KV_RE = re.compile(r"([a-z_]+)=([-\w./]+)")


@dataclass
class Result:
    """The files of one run. ok is False when the run did not run or failed. flags names each condition that marks
    the run, and removed names each condition that keeps it out of the tables (without --all)."""
    run: Run
    ok: bool
    flags: list
    removed: list
    caps: str
    out: str
    stamps: list


def parse_stamps(log: str) -> list[tuple[float, str, dict]]:
    """The STAMP lines of one stderr file: time in ms, name, fields. O(lines)."""
    out = []
    for line in log.splitlines():
        m = TS_RE.match(line)
        if m:
            t = ((int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(m.group(3))) + int(m.group(4)) / 1000.0
            out.append((t, m.group(5), dict(KV_RE.findall(m.group(6)))))
    return out


def read_result(root: Path, run: Run) -> Result:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    gate_path, out_path, log_path = (root / f"{run.name}{s}" for s in ("-gate.txt", ".out", ".log"))
    gate = gate_path.read_text(errors="replace") if gate_path.exists() else ""
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    flags = []
    ok = "gate: OK" in gate and rc is not None and rc.group(1) == "0"
    if not gate:
        flags.append("no gate file")
    elif "gate: OK" not in gate:
        flags.append("gate stopped the run")
    elif not ok:
        flags.append(f"exit code {rc.group(1) if rc else '?'}")
    removed = []
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        flags.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    cap_values = [int(v) for v in ((before.group(3), before.group(4)) if before else ()) +
                  ((after.group(2), after.group(3)) if after else ()) if v]
    if cap_values and min(cap_values) < CAP_MIN_KHZ:
        removed.append(f"a cap of {min(cap_values)} kHz")
    if after and after.group(1) not in ("", "0"):
        removed.append(f"thermal {after.group(1)} after the run")
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    if ok and "llama_kv_cache: size" in log and "K (q8_0)" not in log:
        removed.append("the KV cache is not Q8_0")
    if ok and re.search(r"follow-failed|GGML_ASSERT|dspqueue_read failed|did not copy out|cannot extend", log):
        removed.append("the log has a failure line")
    if ok and re.search(r"failed=1", out):
        removed.append("a turn failed")
    return Result(run, ok, flags, removed, caps, out, parse_stamps(log))


def usable(res: Result, include_all: bool) -> bool:
    """True when the run goes into the tables: it ran, and no condition removes it (or --all)."""
    return res.ok and (include_all or not res.removed)


def med(values) -> float | None:
    """The median of the values that are not None, or None."""
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def fmt(x: float | None, digits: int = 1) -> str:
    """A number, or a dash for None."""
    return "-" if x is None else f"{x:.{digits}f}"


def kv_lines(text: str, tag: str) -> list[dict]:
    """The key=value fields of each stdout line that starts with the tag, for example "TIME turn "."""
    return [dict(KV_RE.findall(line[len(tag):])) for line in text.splitlines() if line.startswith(tag)]


def runs_of(results: dict[str, Result], block: str, vk: str, include_all: bool) -> list[Result]:
    """The usable results of one block and variant, in round order."""
    out = []
    for rnd in range(1, BLOCKS[block].rounds + 1):
        res = results.get(f"4b-{block}-{rnd}-{vk}")
        if res is not None and usable(res, include_all):
            out.append(res)
    return out


# ---- The tables ----

def checks(results: dict[str, Result]) -> list[str]:
    """The conditions of the runs."""
    runs = all_runs()
    got = [results[r.name] for r in runs if r.name in results]
    out = [f"{MODEL}: {len(got)} of {len(runs)} runs have a gate file, {sum(r.ok for r in got)} ran with exit code 0, "
           f"{sum(r.ok and not r.removed for r in got)} go into the tables, {sum(r.ok and not r.flags for r in got)} "
           f"have no mark"]
    caps = Counter(r.caps for r in got if r.ok)
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.most_common()))
    for r in got:
        if r.removed:
            out.append(f"  {r.run.name}: REMOVED ({', '.join(r.removed)})" + (f", marks: {', '.join(r.flags)}" if r.flags else ""))
        elif r.flags:
            out.append(f"  {r.run.name}: marks: " + ", ".join(r.flags))
    return out


TURN_FIELDS = ("ttft_ms", "prompt_ms", "lock_ms", "template_ms", "tokenize_ms", "restore_ms", "base_tokens", "base_ms",
               "snapshot_ms", "tail_tokens", "tail_ms", "first_sample_ms", "first_step_ms")


def draft_passes(res: Result) -> tuple[list[float], Counter]:
    """The ms of one MTP pass for each draft of a run (the draft window over its passes), and the counts of (asked
    length, passes). O(stamps)."""
    per, counts, begin = [], Counter(), None
    for t, name, kv in res.stamps:
        if name == "draft-begin":
            begin = (t, kv.get("n", "?"))
        elif name == "draft-end" and begin is not None:
            passes = int(kv.get("passes", "0"))
            counts[(begin[1], passes)] += 1
            if passes > 0:
                per.append((t - begin[0]) / passes)
            begin = None
    return per, counts


def step_times(res: Result) -> list[float]:
    """The ms of each generation step that decoded (a step window with a decode), without the calls that only
    sampled. O(stamps)."""
    out, begin, decoded = [], None, False
    for t, name, _ in res.stamps:
        if name == "step-begin":
            begin, decoded = t, False
        elif name == "decode-begin" and begin is not None:
            decoded = True
        elif name == "step-end" and begin is not None:
            if decoded:
                out.append(t - begin)
            begin = None
    return out


def turn_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The time to the first token and its parts, turn 1 and the median of turns 2 and later, for each variant; then
    the decode rate, the draft counts, the MTP pass time and the snapshot task after the answers."""
    out = []
    summary: dict[str, dict[str, float | None]] = {}
    for vk in BLOCKS["tn"].variants:
        rs = runs_of(results, "tn", vk, include_all)
        if not rs:
            out.append(f"tn {vk}: no usable run")
            continue
        first, later = [], []
        for res in rs:
            for t in kv_lines(res.out, "TIME turn "):
                (first if t.get("k") == "1" else later).append(t)
        out.append(f"tn {vk} ({VARIANTS[vk].name}), {len(rs)} runs: ms, turn 1 | the median of turns 2 and later "
                   f"({len(later)} turns)")

        def val(rows: list, key: str) -> float | None:
            return med(float(t[key]) for t in rows if key in t and t[key] != "-1.0")
        for key in TURN_FIELDS:
            out.append(f"  {key:16s} {fmt(val(first, key)):>9} | {fmt(val(later, key)):>9}")
        cont = sum(t.get("cont") == "1" for t in later)
        reuse = Counter(t.get("reuse", "?") for t in later)
        out.append(f"  continued        {cont} of {len(later)} later turns, reuse " +
                   ", ".join(f"{k} x{n}" for k, n in sorted(reuse.items())))
        ended = [t for t in first + later if t.get("eog") == "1"]
        out.append(f"  answers          {len(ended)} of {len(first + later)} ended with the end token, "
                   f"{sum(t.get('canon') == '1' for t in ended)} of them have the tokens of their text, "
                   f"{sum(t.get('edge_ws') == '1' for t in ended)} have white space at an end")
        gen_tokens = sum(int(t.get("gen_tokens", 0)) for t in first + later)
        gen_ms = sum(float(t.get("gen_ms", 0)) for t in first + later)
        drafted = sum(int(t.get("drafted", 0)) for t in first + later)
        accepted = sum(int(t.get("accepted", 0)) for t in first + later)
        steps = sum(int(t.get("steps", 0)) for t in first + later)
        rate = gen_tokens * 1000.0 / gen_ms if gen_ms > 0 else None
        out.append(f"  generation       {gen_tokens} tokens in {gen_ms / 1000:.1f} s = {fmt(rate, 2)} t/s, {steps} steps, "
                   f"drafted {drafted}, accepted {accepted}" +
                   (f" ({100.0 * accepted / drafted:.0f} %), mean draft {drafted / steps:.2f}" if drafted and steps else ""))
        per, counts = [], Counter()
        stimes = []
        for res in rs:
            p, c = draft_passes(res)
            per += p
            counts += c
            stimes += step_times(res)
        if per:
            out.append(f"  MTP pass         median {fmt(med(per), 2)} ms over {len(per)} drafts; (asked, passes): " +
                       ", ".join(f"({a}, {p}) x{n}" for (a, p), n in sorted(counts.items())))
        if stimes:
            out.append(f"  step             median {fmt(med(stimes))} ms over {len(stimes)} steps")
        snaps = [s for res in rs for s in kv_lines(res.out, "TIME answer-snapshot ")]
        done = [float(s["ms"]) for s in snaps if s.get("skipped") == "0" and "ms" in s]
        if snaps:
            out.append(f"  answer snapshot  {len(done)} copied in the background (median {fmt(med(done))} ms), "
                       f"{sum(s.get('skipped') == '1' for s in snaps)} skipped")
        summary[vk] = {"ttft": val(later, "ttft_ms"), "ttft1": val(first, "ttft_ms"), "rate": rate,
                       "pass": med(per), "step": med(stimes)}
        out.append("")
    for old, new, what in (("o", "n", "draft off"), ("p", "s", "draft on")):
        a, b = summary.get(old), summary.get(new)
        if a and b and a["ttft"] and b["ttft"]:
            out.append(f"The time to the first token, {what}, turns 2 and later: old {fmt(a['ttft'])} ms, new "
                       f"{fmt(b['ttft'])} ms ({fmt(a['ttft'] - b['ttft'])} ms less, {100.0 * (1 - b['ttft'] / a['ttft']):.0f} %); "
                       f"turn 1: old {fmt(a['ttft1'])} ms, new {fmt(b['ttft1'])} ms")
    if "n" in summary and "s" in summary and summary["n"]["rate"] and summary["s"]["rate"]:
        out.append(f"The decode rate of the new engine: draft on {fmt(summary['s']['rate'], 2)} t/s against draft off "
                   f"{fmt(summary['n']['rate'], 2)} t/s ({100.0 * (summary['s']['rate'] / summary['n']['rate'] - 1):+.1f} %)")
    if "o" in summary and "p" in summary and summary["o"]["rate"] and summary["p"]["rate"]:
        out.append(f"The decode rate of the old engine: draft on {fmt(summary['p']['rate'], 2)} t/s against draft off "
                   f"{fmt(summary['o']['rate'], 2)} t/s ({100.0 * (summary['p']['rate'] / summary['o']['rate'] - 1):+.1f} %)")
    if "p" in summary and "s" in summary:
        out.append(f"One MTP pass: full head {fmt(summary['p']['pass'], 2)} ms, draft head {fmt(summary['s']['pass'], 2)} ms")
    return out


def first_message_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The load time and the first message of 512 tokens after a cold page cache, for each variant."""
    out = ["fm: --drop-cache, then the first message of 512 tokens. ms, the median of the rounds [each round]"]
    for vk in BLOCKS["fm"].variants:
        rs = runs_of(results, "fm", vk, include_all)
        if not rs:
            out.append(f"  {vk}: no usable run")
            continue
        loads = [float(m.group(1)) for r in rs if (m := re.search(r"^TIME model-load ([\d.]+)", r.out, re.M))]
        turns = [t for r in rs for t in kv_lines(r.out, "TIME turn ")]
        warms = [w for r in rs for w in kv_lines(r.out, "TIME embd-warm ")]

        def cell(values: list[float]) -> str:
            return f"{fmt(med(values))} [{', '.join(fmt(v) for v in values)}]"
        out.append(f"  {vk} ({VARIANTS[vk].name})")
        out.append(f"    model load       {cell(loads)}")
        out.append(f"    prompt batch     {cell([float(t['base_ms']) for t in turns if 'base_ms' in t])}")
        out.append(f"    first token      {cell([float(t['ttft_ms']) for t in turns if 'ttft_ms' in t])}")
        if warms:
            out.append(f"    embedding read   {cell([float(w['ms']) for w in warms if 'ms' in w])} ms of the call, "
                       f"{fmt(med(float(w['mib']) for w in warms if 'mib' in w))} MiB, ok " +
                       ",".join(w.get("ok", "?") for w in warms))
    return out


def draft_ops_table(results: dict[str, Result]) -> list[str]:
    """The matrix products of the MTP draft graph by weight (the check runs dh)."""
    out = ["dh: the matrix products of the MTP draft graph that read a head (inputs = activation rows)"]
    for vk in BLOCKS["dh"].variants:
        res = results.get(f"4b-dh-1-{vk}")
        if res is None or not res.ok:
            out.append(f"  {vk}: no run")
            continue
        model = re.search(r"^MODEL (\S+)", res.out, re.M)
        out.append(f"  {vk} ({VARIANTS[vk].name}): {model.group(1) if model else '?'}")
        for line in res.out.splitlines():
            if re.match(r"DRAFTOP weight=(token_embd\.weight|output\.weight|blk\.\d+\.nextn\.draft_head\.weight) ", line):
                out.append(f"    {line}")
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {r.name: read_result(root, r) for r in all_runs() if (root / f"{r.name}-gate.txt").exists()}
    for part in (checks(results), turn_table(results, include_all), first_message_table(results, include_all),
                 draft_ops_table(results)):
        print("\n".join(part))
        print()
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the tables from the pulled files")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    t.add_argument("--all", action="store_true", help="also use the runs with changed caps or heat")
    a = ap.parse_args()
    if a.cmd == "commands":
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(all_runs())} runs")
        return 0
    return table(a.root, a.all)


if __name__ == "__main__":
    sys.exit(main())
