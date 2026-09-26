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

import array
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cli, commands, device, gate, logs, tables  # noqa: E402

ADB = device.ADB
PATHS = device.stage_paths("imgattach", __file__)
PHONE, LAPTOP_STAGE, BOX, STAGE_DIR = PATHS
MODEL_DIR = device.MODEL_DIR
MODEL = device.MODEL_4B
MMPROJ = device.MMPROJ_4B
# The saved copy of the photo of the chat of the user. A new chat of the app deletes its photos, thus the stage pushes it.
PHOTO = f"{PHONE}/in/photo.jpg"
GATE_KB = device.GATE_4B_KB
STAGE_FILES = ("bin/app_fuzz_driver", "bin/gate.sh", "lib/libggml-base.so", "lib/libggml-cpu.so",
               "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so",
               "lib/libllama-common.so", "lib/libllama.so", "lib/libmtmd.so")
ENV = (f"{device.lib_env(PHONE)} FUZZ_APP_LIBDIR={PHONE}/lib "
       f"FUZZ_APP_WORK={PHONE}/work FUZZ_APP_MODEL_DIR={PHONE}/work FUZZ_APP_DEVICE=HTP0 "
       f"FUZZ_APP_REAL_MODEL={MODEL_DIR}/{MODEL} FUZZ_APP_REAL_MMPROJ={MODEL_DIR}/{MMPROJ} FUZZ_APP_IMAGE={PHOTO} "
       f"FUZZ_APP_IMAGE_TOKENS=768 FUZZ_APP_STAGE_TOKENS=32 FUZZ_APP_STAGE_STRICT=0")
TOOLS = ("app_fuzz_driver",)
PGREP = device.pgrep(*TOOLS)
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
    # Each flow of the scenario writes into the work directory, thus each run starts with an empty one.
    cmd = commands.gated_run(stem, GATE_KB, LIMIT_S, env, f"{PHONE}/bin/app_fuzz_driver --scenario image-stage",
                             stage=PHONE, pre=f"rm -rf {PHONE}/work && mkdir -p {PHONE}/work && ")
    return commands.run_lines(f"# REAL-MODEL {MODEL.removesuffix('.gguf')}: {run.name}, {run.text}: {run.flows}",
                              cmd, PGREP)


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


def setup_lines() -> list[str]:
    """The lines that copy the stage files and the photo to the phone and check them. The phone directory holds the
    photo in in/ and the files of the harness in work/, thus the stage makes those two directories also."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in these lines: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {MODEL_DIR} | grep -E \"Qwen3.5-4B-Q8_0(-draft32k|.mmproj)?.gguf\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out {PHONE}/in {PHONE}/work'",
        *commands.photo_lines(device.BOX_BUILD, PHOTO),
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {libs} {PHONE}/lib/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
    ]


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = commands.header_lines(HEADER) + setup_lines()
    lines += ["#", f"# ==== {MODEL.removesuffix('.gguf')}: {len(RUNS)} runs ===="]
    for run in RUNS:
        lines += run_lines(run)
    lines += commands.output_lines(PATHS, tools=TOOLS)
    return commands.write_commands(path, lines)


# ---- The parser of the files ----

STAGE_RE = re.compile(r"^STAGE flow=(\S+) ref=(\S+) first_image=(\d) stage_ms=([\d.]+) staged=(-?\d+) chat_ms=([\d.]+) "
                      r"ttft_ms=([\d.]+) tokens=(\d+) same=(\S+) dlogit=(\S+) kl=(\S+) top1=(-?\d+) lead=(\d+) "
                      r"compute0_mib=([\d.]+) compute1_mib=([\d.]+) part=(\S+) next=(\S+) stats=(.*)$")
VISION_RE = re.compile(r"^STAGE-VISION ms=([\d.]+) ready=(\d)")


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
    gate_text, out, log = logs.read_run(root, run.name)
    conditions = gate.read(gate_text)
    removed = []
    if not conditions.ok:
        if "gate: OK" not in gate_text:
            removed.append("no gate file" if not gate_text else "the gate stopped the run")
        else:
            tail = [ln for ln in log.splitlines() if "image-stage" in ln or "FAKEJNI" in ln][-1:]
            removed.append(f"exit code {conditions.rc if conditions.rc is not None else '?'}"
                           + (f": {tail[0][:200]}" if tail else ""))
    # A changed cap does not remove a run here, thus the thermal status comes from the gate file itself.
    after = gate.AFTER_RE.search(gate_text)
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


def table(root: Path, runner_log: Path | None) -> int:
    """Print the checks, the answers, the times and the comparisons across runs."""
    if runner_log is not None and not runner_log.is_file():
        print(f"stage.py: the runner log {runner_log} does not exist.", file=sys.stderr)
        return 1
    if not root.is_dir():
        return cli.missing_root(root)
    # The runner reads the caps some seconds after a run, thus its CAPS-CHANGED mark catches a change that the
    # "after:" line of the gate file does not show.
    rejected = set(logs.runner_marks(runner_log))
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
              f"{tables.fmt(tables.med(f.ttft_ms for f in fs), 0):>6s} "
              f"{tables.fmt(tables.med(f.chat_ms for f in fs), 0):>6s} "
              f"{tables.fmt(tables.med(f.stage_ms for f in fs if f.stage_ms > 0), 0):>6s} "
              f"{tables.fmt(tables.med(f.staged for f in fs if f.staged > 0), 0):>6s}")
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
    return cli.run(STAGE_DIR, __doc__, write=write_commands, count=lambda: len(RUNS), table=table, all_flag=False,
                   table_help="print the tables from the pulled files",
                   runner_log="the log of the laptop runner: its CAPS-CHANGED runs leave the times")


if __name__ == "__main__":
    sys.exit(main())
