#!/usr/bin/env python3
"""The phone stage "imgattach": the stage of an attached image against the send alone, and the time to the first token.

Usage:
    stage.py commands [--out PATH]                     write the phone command file (build/imgattach/phone-commands.txt)
    stage.py table [--root DIR] [--runner-log PATH]    print the tables from the pulled files (build/imgattach/phone-out)

The program of each run is app_fuzz_driver --scenario image-stage (tests/fuzz/app/harness/app_harness.cpp, built by
tools/stages/imgattach/build.sh). It calls the JNI entry points of the app (llama_jni.cpp of the working tree) with the
4B Q8_0 on HTP0, its projector on HTP0, n_ctx 8192, 4 threads, 768 image tokens, the draft file, and the photo of the
chat of the user. Each flow starts from an empty memory, sends a first message of about 1520 tokens and puts in its
answer (a first turn), then makes the second turn:

    vision      not a flow: prepareVision only (the user opens the photo menu)
    warm        chatStart with a second image B: the first image of the engine, which no other flow compares with
    send        chatStart with the photo: the reference
    stage-send  stageImage with the photo and another text (the attach), then the send of "send"
    send-b      chatStart with the image B
    stage-other stageImage with the photo, then the send of "send-b" (the user changed the photo)
    send-first  chatStart with the photo as the first message of a chat
    stage-first stageImage, then the send of "send-first"

The engine decodes the text of the message before the image in the batch of the image (QWEN_IMAGE_TEXT_ROWS=0 turns
that off, the run m1). The sampler is the one of the app (temperature 0.7, top-p 0.8) with a fixed seed
(QWEN_SAMPLER_SEED), 32 tokens for each flow. The kernels of HTP0 round differently for other batch shapes, thus the
scenario runs with FUZZ_APP_STAGE_STRICT=0: it does not stop on other tokens, and each STAGE line reports against the
first flow of its reference the KL divergence and the top token of the logits after the prompt, and the equal leading
tokens. The logits of each flow go to a file, thus the table also compares runs (with and without the merged batch).

A run goes into the tables when its gate passed, its exit code is 0, the thermal status after it is 0, and the runner
log (--runner-log) does not mark it CAPS-CHANGED. The comparisons of the answers use each run. O(size of the files).

This file is tools/stages/imgattach/stage.py. The files of the stage stay in build/imgattach.
"""

import argparse
import array
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/imgattach"
MODEL_DIR = "/data/local/tmp/qwen/models"
MODEL = "Qwen3.5-4B-Q8_0.gguf"
MMPROJ = "Qwen3.5-4B-Q8_0.mmproj.gguf"
# The saved copy of the photo of the chat of the user. A new chat of the app deletes its photos, thus the stage pushes it.
PHOTO_COPY = "build/imgturn/photo.jpg"
PHOTO_SHA1 = "a2200d1a726ec0a8576b4a18dc2ef1aa4e4c797d"
PHOTO = f"{PHONE}/in/photo.jpg"
GATE_KB = 8388608
LAPTOP_STAGE = "build/imgattach"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build"
STAGE_DIR = Path(os.path.relpath(Path(__file__).resolve().parents[3] / LAPTOP_STAGE))
STAGE_FILES = ("bin/app_fuzz_driver", "bin/gate.sh", "lib/libggml-base.so", "lib/libggml-cpu.so",
               "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so",
               "lib/libllama-common.so", "lib/libllama.so", "lib/libmtmd.so")
ENV = (f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib FUZZ_APP_LIBDIR={PHONE}/lib "
       f"FUZZ_APP_WORK={PHONE}/work FUZZ_APP_MODEL_DIR={PHONE}/work FUZZ_APP_DEVICE=HTP0 "
       f"FUZZ_APP_REAL_MODEL={MODEL_DIR}/{MODEL} FUZZ_APP_REAL_MMPROJ={MODEL_DIR}/{MMPROJ} FUZZ_APP_IMAGE={PHOTO} "
       f"FUZZ_APP_IMAGE_TOKENS=768 FUZZ_APP_STAGE_TOKENS=32 FUZZ_APP_STAGE_STRICT=0")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x app_fuzz_driver; echo pgrep-done'"
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')
LIMIT_S = 100


@dataclass(frozen=True)
class Run:
    """One phone run: its name, the draft switch, the merged batch switch, the flows in order, and its question."""
    name: str
    draft: bool
    merged: bool
    flows: str
    text: str


RUNS = (
    Run("d1", False, True, "send,send,send", "the first image of the engine: the first send against the two after it"),
    Run("d2", False, True, "vision,send,send", "the projector before the first turn: the first send against the second"),
    Run("n1", False, True, "warm,send,stage-send,send,stage-send", "draft off: stage-send against send, and the floor"),
    Run("a1", True, True, "warm,send,stage-send,send,stage-send", "the app (draft on): the same"),
    Run("m1", True, False, "warm,send,stage-send,send,stage-send", "draft on, the text before the image in its own batch"),
    Run("a3", True, True, "warm,send-b,stage-other,send-first,stage-first", "a changed photo, the first message of a chat"),
)


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{PHONE}/out/{run.name}"
    env = (f"{ENV} FUZZ_APP_SPECULATIVE={1 if run.draft else 0} QWEN_IMAGE_TEXT_ROWS={1 if run.merged else 0} "
           f"FUZZ_APP_STAGE_DUMP={stem}-logits FUZZ_APP_STAGE_FLOWS={run.flows}")
    cmd = (f"sh {PHONE}/bin/gate.sh {GATE_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"rm -rf {PHONE}/work && mkdir -p {PHONE}/work && "
           f"timeout -s KILL {LIMIT_S} env {env} {PHONE}/bin/app_fuzz_driver --scenario image-stage "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    return ["#", f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {run.name}, {run.text}: {run.flows}",
            THERMAL, f"{ADB} shell '{cmd}'", PGREP]


HEADER = """\
# Phone stage "imgattach" 2: the stage of an attached image (stageImage) against the send alone, the text before the
# image in the batch of the image, and the first image of the engine.
#
# The first stage imgattach: the TTFT of stage-send was 156 to 187 ms against 1663 to 2021 ms for send. With the draft
# off, a send against a send of the same run gave logits 0.155 apart: the first flow of a run (the first image of the
# engine) differs from the later ones. On the box CPU every flow gives the same bits, the first one too, thus the cause
# is on HTP0: the load of the projector, or the first image graph.
#
# The program is app_fuzz_driver --scenario image-stage: the JNI code of the app (llama_jni.cpp of the box working
# tree) through the fake Java VM of the harness, with the libraries of the series of HEAD. The 4B Q8_0 with its draft
# file on HTP0, the projector on HTP0, n_ctx 8192, 768 image tokens, the photo of the chat of the user. Each answer is
# 32 tokens of the sampler of the app with a fixed seed. The run reports and does not stop on other tokens
# (FUZZ_APP_STAGE_STRICT=0). The logits after each prompt go to a file.
#
# The files: build/imgattach/phone of the box (tools/stages/imgattach/build.sh). The phone must hold
# Qwen3.5-4B-Q8_0.gguf, Qwen3.5-4B-Q8_0-draft32k.gguf and Qwen3.5-4B-Q8_0.mmproj.gguf in /data/local/tmp/qwen/models.
# The photo is the saved copy build/imgturn/photo.jpg (sha1 a2200d1a...), and adb push copies it.
#
# The runs, 6:
#   d1  draft off: send, send, send (the first image of the engine against the later ones)
#   d2  draft off: vision, send, send (the projector loads before the first turn)
#   n1  draft off: warm, send, stage-send, send, stage-send
#   a1  draft on (the app): warm, send, stage-send, send, stage-send
#   m1  draft on, the text before the image in its own batch (QWEN_IMAGE_TEXT_ROWS=0): the same flows
#   a3  draft on: warm, send-b, stage-other, send-first, stage-first
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger, MemAvailable
# 8 GB), the program under timeout -s KILL (100 s), the exit code and the conditions after the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, screen on, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 30 s for a run of three flows and 45 s
# for a run of five, thus about 4 minutes of program time plus about 10 s of gate and checks for each run, plus the
# waits for thermal status 0. The pull is about 25 MB (1 MB of logits for each flow). Then copy the runner log to
# build/imgattach/runner.log on the box and run: tools/stages/imgattach/stage.py table --runner-log build/imgattach/runner.log
"""


def photo_lines() -> list[str]:
    """The lines that copy the saved photo from the box, check its sha1, and push it into the stage directory."""
    return [
        f"mkdir -p {Path(PHOTO_COPY).parent} && rsync -a {BOX}/imgturn/photo.jpg {PHOTO_COPY}",
        f"echo '{PHOTO_SHA1}  {PHOTO_COPY}' | sha1sum -c",
        f"{ADB} push {PHOTO_COPY} {PHOTO}",
        f"{ADB} shell 'ls -l {PHOTO} && sha1sum {PHOTO}'",
    ]


def setup_lines() -> list[str]:
    """The lines that copy the stage files and the photo to the phone and check them."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/imgattach/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODEL_DIR} | grep -E \"Qwen3.5-4B-Q8_0(-draft32k|.mmproj)?.gguf\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out {PHONE}/in {PHONE}/work'",
        *photo_lines(),
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {libs} {PHONE}/lib/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
    ]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'pgrep -x app_fuzz_driver; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/imgattach/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = HEADER.rstrip("\n").split("\n") + setup_lines()
    lines += ["#", f"# ==== {MODEL.removesuffix('.gguf')}: {len(RUNS)} runs ===="]
    for run in RUNS:
        lines += run_lines(run)
    lines += output_lines()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


# ---- The parser of the files ----

AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*)")
STAGE_RE = re.compile(r"^STAGE flow=(\S+) ref=(\S+) first_image=(\d) stage_ms=([\d.]+) staged=(-?\d+) chat_ms=([\d.]+) "
                      r"ttft_ms=([\d.]+) tokens=(\d+) same=(\S+) dlogit=(\S+) kl=(\S+) top1=(-?\d+) lead=(\d+) "
                      r"compute0_mib=([\d.]+) compute1_mib=([\d.]+) part=(\S+) next=(\S+) stats=(.*)$")
VISION_RE = re.compile(r"^STAGE-VISION ms=([\d.]+) ready=(\d)")
RUNNER_TITLE_RE = re.compile(r"^# REAL-MODEL \S+: (\S+),")
RUNNER_CAPS_RE = re.compile(r"^CAPS .*CAPS-CHANGED")


def runner_rejects(path: Path | None) -> set[str]:
    """The names of the runs that the runner log marks CAPS-CHANGED. O(lines of the log)."""
    if path is None:
        return set()
    rejected, name = set(), None
    for line in path.read_text(errors="replace").splitlines():
        m = RUNNER_TITLE_RE.match(line)
        if m:
            name = m.group(1)
        elif name is not None and RUNNER_CAPS_RE.match(line):
            rejected.add(name)
    return rejected


@dataclass
class Flow:
    """One STAGE line, with its place in the run."""
    run: str
    index: int
    name: str
    ref: str
    first_image: bool
    stage_ms: float
    staged: int
    chat_ms: float
    ttft_ms: float
    tokens: int
    same: str
    kl: float
    top1: int
    lead: int
    compute0: float
    compute1: float
    part: str
    next: str
    stats: str


def read_run(root: Path, run: Run, rejected: set[str]) -> tuple[list[Flow], list[str], list[str]]:
    """The flows of one run, the reasons to remove it from the times (empty when it stays), and its vision lines."""
    gate_path, out_path, log_path = (root / f"{run.name}{s}" for s in ("-gate.txt", ".out", ".log"))
    gate = gate_path.read_text(errors="replace") if gate_path.exists() else ""
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    removed = []
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    if "gate: OK" not in gate:
        removed.append("no gate file" if not gate else "the gate stopped the run")
    elif rc is None or rc.group(1) != "0":
        tail = [ln for ln in log.splitlines() if "image-stage" in ln or "FAKEJNI" in ln][-1:]
        removed.append(f"exit code {rc.group(1) if rc else '?'}" + (f": {tail[0][:200]}" if tail else ""))
    after = AFTER_RE.search(gate)
    if after and after.group(1) not in ("", "0"):
        removed.append(f"thermal {after.group(1)} after the run")
    if run.name in rejected:
        removed.append("the runner marked CAPS-CHANGED")
    flows, vision, index = [], [], 0
    for line in out.splitlines():
        v = VISION_RE.match(line)
        if v:
            vision.append(f"prepareVision {float(v.group(1)):.0f} ms, ready {v.group(2)}")
            index += 1
            continue
        m = STAGE_RE.match(line)
        if m:
            g = m.groups()
            flows.append(Flow(run.name, index, g[0], g[1], g[2] == "1", float(g[3]), int(g[4]), float(g[5]),
                              float(g[6]), int(g[7]), g[8], float(g[10]), int(g[11]), int(g[12]), float(g[13]),
                              float(g[14]), g[15], g[16], g[17]))
            index += 1
    return flows, removed, vision


def logits_of(root: Path, flow: Flow) -> array.array | None:
    """The logits after the prompt of a flow from its dump file, or None."""
    path = root / f"{flow.run}-logits-{flow.index}-{flow.name}.f32"
    if not path.exists():
        return None
    values = array.array("f")
    values.frombytes(path.read_bytes())
    return values


def kl(p: array.array, q: array.array) -> float:
    """KL(softmax(p) || softmax(q)) in double. O(vocabulary)."""
    mp, mq = max(p), max(q)
    lzp = mp + math.log(math.fsum(math.exp(x - mp) for x in p))
    lzq = mq + math.log(math.fsum(math.exp(x - mq) for x in q))
    return max(0.0, math.fsum(math.exp(a - lzp) * ((a - lzp) - (b - lzq)) for a, b in zip(p, q, strict=True)))


def top(values: array.array) -> int:
    """The index of the largest value."""
    return max(range(len(values)), key=values.__getitem__)


def med(values: list[float]) -> str:
    """The median as text, or a dash."""
    return f"{statistics.median(values):.0f}" if values else "-"


def table(root: Path, runner_log: Path | None) -> int:
    """Print the checks, the answers, the times and the comparisons across runs."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    rejected = runner_rejects(runner_log)
    timed: list[Flow] = []
    by_run: dict[str, list[Flow]] = {}
    print("The runs:")
    for run in RUNS:
        flows, removed, vision = read_run(root, run, rejected)
        by_run[run.name] = flows
        if not removed:
            timed += flows
        state = "REMOVED from the times (" + ", ".join(removed) + ")" if removed else "in the times"
        extra = f", {'; '.join(vision)}" if vision else ""
        print(f"  {run.name} ({'draft on' if run.draft else 'draft off'}, "
              f"{'merged' if run.merged else 'separate'} batch): {len(flows)} flows, {state}{extra}")
    print()
    print("The answers, each flow against the first flow of its reference in its run (every run):")
    print(f"  {'run':3s} {'#':>1s} {'flow':12s} {'ref':10s} {'image':5s} {'KL':>10s} {'top1':>4s} {'lead':>5s} "
          f"{'compute MiB':>12s}")
    for run in RUNS:
        for f in by_run[run.name]:
            kl_text = f"{f.kl:.3g}" if f.kl >= 0 else "-"
            top1 = {1: "yes", 0: "no"}.get(f.top1, "-")
            lead = f"{f.lead}/{f.tokens}" if f.same != "-" else "-"
            print(f"  {run.name:3s} {f.index:1d} {f.name:12s} {f.ref:10s} {'first' if f.first_image else 'later':5s} "
                  f"{kl_text:>10s} {top1:>4s} {lead:>5s} {f.compute0:5.1f}>{f.compute1:<5.1f}")
    print()
    print("stage-send against send, next to the floor (send against send) of the same run, KL of the prompt logits:")
    for run in RUNS:
        flows = by_run[run.name]
        floor = [f.kl for f in flows if f.name == "send" and f.kl >= 0]
        stage = [f.kl for f in flows if f.name.startswith("stage-") and f.kl >= 0]
        if floor or stage:
            print(f"  {run.name}: floor {', '.join(f'{x:.3g}' for x in floor) or '-'}; "
                  f"stage flows {', '.join(f'{x:.3g}' for x in stage) or '-'}")
    print()
    print("Across runs, the logits after the prompt of the first later send of each run (KL, the same top token):")
    sends = {}
    for run in RUNS:
        for f in by_run[run.name]:
            if f.name == "send" and not f.first_image and run.name not in sends:
                values = logits_of(root, f)
                if values is not None:
                    sends[run.name] = values
    pairs = (("a1", "m1", "merged against separate batch, draft on"), ("n1", "a1", "draft off against draft on"),
             ("d1", "n1", "two runs with the draft off"))
    for x, y, text in pairs:
        if x in sends and y in sends:
            print(f"  {x} against {y} ({text}): KL {kl(sends[x], sends[y]):.3g}, top token "
                  f"{'the same' if top(sends[x]) == top(sends[y]) else 'different'}")
    print()
    print("The times of the runs in the times, ms (the median; n is the count of flows):")
    print(f"  {'run':3s} {'flow':12s} {'image':6s} {'n':>2s} {'ttft':>6s} {'chat':>6s} {'stage':>6s} {'staged':>6s}")
    groups = defaultdict(list)
    for f in timed:
        groups[(f.run, f.name, f.first_image)].append(f)
    for (run, name, first_image), fs in sorted(groups.items(), key=lambda kv: ([r.name for r in RUNS].index(kv[0][0]), kv[0][1])):
        print(f"  {run:3s} {name:12s} {'first' if first_image else 'later':6s} {len(fs):2d} "
              f"{med([f.ttft_ms for f in fs]):>6s} {med([f.chat_ms for f in fs]):>6s} "
              f"{med([f.stage_ms for f in fs if f.stage_ms > 0]):>6s} {med([f.staged for f in fs if f.staged > 0]):>6s}")
    print()
    print("The staged part and the first item of the send after it (a flow of each kind):")
    seen = set()
    for f in timed + [f for fs in by_run.values() for f in fs]:
        if f.part != "-" and f.name not in seen:
            seen.add(f.name)
            print(f"  {f.name}: {f.part} then {f.next}")
    print()
    print("The stats line of the first later flow of each name in the times:")
    seen = set()
    for f in timed:
        if f.name not in seen and not f.first_image:
            seen.add(f.name)
            print(f"  {f.run} {f.name}: {f.stats}")
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    t = sub.add_parser("table", help="print the tables from the pulled files")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    t.add_argument("--runner-log", type=Path, help="the log of the laptop runner: its CAPS-CHANGED runs leave the times")
    a = ap.parse_args()
    if a.cmd == "commands":
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(RUNS)} runs")
        return 0
    if a.runner_log is not None and not a.runner_log.is_file():
        print(f"stage.py: the runner log {a.runner_log} does not exist.", file=sys.stderr)
        return 1
    return table(a.root, a.runner_log)


if __name__ == "__main__":
    sys.exit(main())
