#!/usr/bin/env python3
"""The phone stage fak: the flash attention of HTP0 against the flash attention of HEAD, on one library set.

Usage (tools/stages/fak/build.sh makes the binaries first and then runs "files"):
    stage.py files                        write the test files, SHA256SUMS and phone-commands.txt
    stage.py table [--root DIR] [--all]   print the tables from the pulled outputs (phone-out)

The switch GGML_HEXAGON_FA_OPT holds the bits HTP_FA_OPT_* of htp/flash-attn-ops.h. One variant for
each part (the patches land one at a time in this order: the chunk cost model, the tile softmax, the
resident K and V, the decode spans), and the sums:
    A  0   the kernel HTP_FA_KERNEL_HMX with hmx_fa_find_chunk_size (the flash attention of HEAD)
    B  1   the chunk cost model alone: hmx_fa_find_chunk_size_v2 for HTP_FA_KERNEL_HMX
    T  2   the tile softmax alone: HTP_FA_KERNEL_HMX2 streaming for the prefill shapes
    R  6   the resident K and V with the tile softmax (bit 4 alone selects nothing)
    S  8   the decode spans alone: HTP_FA_KERNEL_HMX2 for the decode shapes
    C  3   B plus T
    D  7   C plus the resident K and V
    E  15  D plus the decode spans (the preset value of the library)
The op tests run each part alone (A B T R S E), thus a failure on the chip names its part. A prefill
ubatch has no decode shape, thus D stands for E in the prefill blocks, and a decode token has no prefill
shape, thus B stands for C and D in the decode blocks. The evidence of each patch: B against A (chunk
model), C against B (tile softmax), D against C (resident K and V, prefill at depth 0 only, where the plan
picks the resident form), E against B (decode spans). The earlier stages give the KL of A
(oracle-kl-floor: -b 512 0.000574, -b 1 0.000464) and the FLASH_ATTN_EXT failures of HEAD
(htp-kernel-patch-verification: sinks cases of the HMX path). At 512 KV rows or less, B has the plans of A
for the decode shapes and for 512 queries (the plan tool of build/fak), thus the KL runs at -c 512 give
B the KL of A: the prefill KL runs C and D, and the decode KL runs E.

The vision encoder of the 4B (tools/mtmd/models/qwen3vl.cpp) calls the flash attention with head size 64,
16 heads, no GQA, no mask and F16 K and V. HTP_FA_KERNEL_HMX2 takes these shapes too. The op tests have two
cases of that layout, and the block vit gives the time of one op (test-backend-ops perf) for A B C E.

The questions:
    1. Do the FLASH_ATTN_EXT ops of the 4B shape pass test-backend-ops (HTP0 against the CPU) with E, and
       how do their errors compare with A? Does the whole FLASH_ATTN_EXT suite give the same failures?
    2. Is the KL of the 4B at the floor with E: the prefill (-b 512), the decode (-b 1) and 16384 tokens?
    3. The prefill and the decode rates of A to E, in alternated rounds, and the F16 cache decode rate
       with E (the release rule: a Q8_0 cache decodes at least as fast as an F16 cache at every depth).
    4. The op split of one 1024-token ubatch at depth 3072 and of the decode tokens at depth 3072 for
       A B C E (GGML_HEXAGON_PROFILE=1).
    5. The time of the flash attention of the vision encoder (1024 and 2688 patches) for A B C E.

A run name is the stem of its three output files on the phone: <run>-gate.txt (the conditions before and
after the run and the exit code), <run>.out (the stdout of the tool) and <run>.log (its stderr). The
table uses a run when its gate passed, its exit code is 0, the CPU caps after the run are the caps before
it, and the thermal status after it is 0. --all also uses the runs with changed caps or heat. A rate is
the median of the rounds, and the difference to A is the median over the rounds of the ratio of the two
runs of one round. The table only reads files: O(size of the outputs).

This file is tools/stages/fak/stage.py, and build/fak/stage.py is a link to it. The files of the stage
stay in build/fak.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = REPO / "build/fak"

# ---- The stage paths and the phone lines ----

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/fak"
LAPTOP_STAGE = "build/fak"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/fak"
MODEL = "/data/local/tmp/qwen/models/Qwen3.5-4B-Q8_0.gguf"
MODEL_KB = 8388608
# The ls of a model file makes the laptop runner treat a line as a model run: it waits for the unlocked
# phone, stops the Qwen app and wakes the screen. The op runs load no model.
MARKER = "ls /data/local/tmp/qwen/models/Qwen3.5-2B-Q8_0.gguf > /dev/null"
OPS_KB = 2097152
# The KL bases: the naive x86 oracle (oracle-kl-floor), the text, and the 16k base with f32
# activations (the prefill rows run with f16 activations on HTP0).
KLD_BASE = "/data/local/tmp/qwen/eval/naive-4B-q8.kld"
WIKI = "/data/local/tmp/qwen/eval/wiki.test.raw"
KVKL_DIR = "/data/local/tmp/qwen/memory/b/bases"
KVKL_16K = f"{KVKL_DIR}/naive-4B-deqf32-c16384-s32-t64.kvb"
KVKL_TEXT = f"{KVKL_DIR}/wiki.test.raw"
# The environment of the app (init_impl in llama_jni.cpp) and the stage libraries.
LIB_ENV = (f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib "
           "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1")
# The context of the app (load_impl in llama_jni.cpp): n_batch = n_ubatch = 1024, 4 threads, flash
# attention on HTP0, the Q8_0 cache with the FWHT rotation (the flags of the variant b of bench-kv).
BENCH_ARGS = "-dev HTP0 -ngl 99 -t 4 -fa on -b 1024 -ub 1024 -o jsonl"
STAGE_FILES = ("bin/gate.sh", "bin/llama-bench", "bin/llama-perplexity", "bin/test-backend-ops", "bin/kvkl",
               "lib/libggml-base.so", "lib/libggml-cpu.so", "lib/libggml-hexagon.so", "lib/libggml-htp-v79.so",
               "lib/libggml-opencl.so", "lib/libggml.so", "lib/libllama-bench-impl.so",
               "lib/libllama-perplexity-impl.so", "lib/libllama-common.so", "lib/libllama.so", "lib/libmtmd.so")
# The test files of test-backend-ops (phone/tests/<name>.txt, written by write_files)
TEST_FILES = ("fa4b", "vit")
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = (f"{ADB} shell 'pgrep -x llama-bench; pgrep -x llama-perplexi; pgrep -x test-backend-op; pgrep -x kvkl; "
         "echo pgrep-done'")
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')


@dataclass(frozen=True)
class Variant:
    """One value of GGML_HEXAGON_FA_OPT: its key, its value and its description."""
    key: str
    opt: int
    text: str


VARIANTS = {v.key: v for v in (
    Variant("a", 0, "HEAD: HTP_FA_KERNEL_HMX, hmx_fa_find_chunk_size"),
    Variant("b", 1, "the chunk cost model alone"),
    Variant("t", 2, "the tile softmax alone (HMX2 streaming for the prefill shapes, the old chunk model)"),
    Variant("r", 6, "the tile softmax with the resident K and V (the resident form needs the tile softmax)"),
    Variant("s", 8, "the decode spans alone"),
    Variant("c", 3, "B plus the tile softmax, K and V converted for each q block"),
    Variant("d", 7, "C plus the resident K and V"),
    Variant("e", 15, "D plus the decode spans (the preset value)"),
)}
# The column order of the rate table
TABLE_KEYS = "abcde"


@dataclass(frozen=True)
class Block:
    """One kind of run: the tool, its arguments, the variants, the rounds, the time limit (s), the cache
    type, a profile flag, the KiB of MemAvailable that the gate requires, and the text."""
    key: str
    tool: str
    args: str
    variants: str
    rounds: int
    limit: int
    kv: str
    profile: bool
    gate_kb: int
    text: str


# The estimates of the tool time (s). The llama-bench blocks: the test start times (test_time) of the stage
# bench-kv (4B, HTP0, 2026-09-24) give a run cycle of 19 to 24 s for pp512 d0,4096 -r 3, 37 to 38 s for tg32
# d0,1024,4096 -r 2 and 43 to 44 s for tg32 d16384 -r 2. A cycle includes about 10 s of gate and checks,
# thus the tool time of a run with -r 1 is about 11 s (p), 15 s (t) and 30 s (l). The estimates of q, f
# and prof add the prefill and decode times of those runs. The op tests, the KL runs and kvkl have no
# earlier measurement: their estimates are upper limits.
BLOCKS = {b.key: b for b in (
    Block("ops", "test-backend-ops test", "fa4b", "abtrse", 1, 100, "", False, OPS_KB,
          "test-backend-ops test -b HTP0, the FLASH_ATTN_EXT cases of the 4B shape and of its vision encoder "
          "(tests/fa4b.txt)"),
    Block("suite1", "test-backend-ops test", "-o FLASH_ATTN_EXT -p \"hsk=(40|64|72|80|96|128),\"", "e", 1, 100, "",
          False, OPS_KB, "test-backend-ops test -b HTP0, the FLASH_ATTN_EXT suite, head sizes 40 to 128"),
    Block("suite2", "test-backend-ops test", "-o FLASH_ATTN_EXT -p \"hsk=(192|256|320|512|576),\"", "e", 1, 100, "",
          False, OPS_KB, "test-backend-ops test -b HTP0, the FLASH_ATTN_EXT suite, head sizes 192 to 576"),
    Block("vit", "test-backend-ops perf", "vit", "abce", 1, 60, "", False, OPS_KB,
          "test-backend-ops perf -b HTP0, the FLASH_ATTN_EXT ops of the vision encoder (tests/vit.txt)"),
    Block("klp", "llama-perplexity", f"-c 512 -b 512 --chunks 4 --kl-divergence-base {KLD_BASE} --kl-divergence",
          "cd", 1, 90, "q8_0", False, MODEL_KB, "the prefill KL (-b 512, 4 chunks) against the naive oracle"),
    Block("kld", "llama-perplexity",
          f"-c 512 -b 1 -ub 1 --chunks 1 --kl-divergence-base {KLD_BASE} --kl-divergence",
          "e", 1, 100, "q8_0", False, MODEL_KB, "the decode KL (-b 1, 1 chunk) against the naive oracle"),
    Block("kl16", "kvkl", f"-f {KVKL_TEXT} -c 16384 --stride 32 --tail 64 --base {KVKL_16K}",
          "e", 1, 110, "q8_0", False, MODEL_KB,
          "kvkl at 16384 tokens against the f32-activation oracle base (prefill rows at each depth, decode tail)"),
    Block("p", "llama-bench", "-p 512 -n 0 -d 0,4096 -r 1", "abcd", 3, 60, "q8_0", False, MODEL_KB,
          "llama-bench pp512 at the depths 0 and 4096, 1 repetition"),
    Block("q", "llama-bench", "-p 1024 -n 0 -d 3072 -r 1", "abd", 3, 40, "q8_0", False, MODEL_KB,
          "llama-bench pp1024 at the depth 3072 (one ubatch), 1 repetition"),
    Block("t", "llama-bench", "-p 0 -n 32 -d 0,4096 -r 1", "abe", 3, 60, "q8_0", False, MODEL_KB,
          "llama-bench tg32 at the depths 0 and 4096, 1 repetition"),
    Block("l", "llama-bench", "-p 0 -n 32 -d 16384 -r 1", "ae", 3, 60, "q8_0", False, MODEL_KB,
          "llama-bench tg32 at the depth 16384, 1 repetition"),
    Block("f", "llama-bench", "-p 0 -n 32 -d 4096,16384 -r 1", "e", 3, 90, "f16", False, MODEL_KB,
          "llama-bench tg32 with an F16 cache at the depths 4096 and 16384, 1 repetition (the release rule)"),
    Block("prof", "llama-bench", "-p 1024 -n 8 -d 3072 -r 1 -v", "abce", 1, 60, "q8_0", True, MODEL_KB,
          "GGML_HEXAGON_PROFILE=1 (with -v): one 1024-token ubatch at the depth 3072, then 8 decode tokens"),
)}
# The blocks of one group run round by round: round 1 of each block, then round 2 of each block. The
# variants run in the order of the block in an odd round and in the reverse order in an even round,
# thus a slow drift of the clocks or the heat goes equally to each variant over two rounds.
GROUPS = (("ops", "suite1", "suite2", "vit"), ("klp", "kld", "kl16"), ("p", "q", "t", "l", "f"), ("prof",))
EST_S = {"ops": 40, "suite1": 50, "suite2": 50, "vit": 15, "klp": 40, "kld": 60, "kl16": 90, "p": 11, "q": 12,
         "t": 15, "l": 30, "f": 37, "prof": 20}


@dataclass(frozen=True)
class Run:
    """One phone run of one block, round and variant."""
    block: Block
    round: int
    variant: Variant

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"{self.block.key}-{self.round}-{self.variant.key}"


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


# ---- The FA cases of the 4B shape (make_test_cases_from_file of test-backend-ops) ----

FA_DK, FA_HEADS, FA_KV_HEADS = 256, 16, 4
TYPES = {"f32": (0, 1, 4), "f16": (1, 1, 2), "q8_0": (8, 32, 34)}
# (queries, KV rows): decode, the MTP verify, prefill ubatches at depths, and tails that are not a
# multiple of the tile sizes.
# The CPU reference of test-backend-ops dominates the time of a run, thus the largest prefill case is
# 1024 x 8192 and only for Q8_0.
FA_SHAPES = {
    "q8_0": ((1, 512), (1, 4096), (1, 16384), (1, 5000), (4, 4096), (512, 512), (1000, 1000), (1024, 4096),
             (1024, 8192), (700, 3700)),
    "f16": ((1, 4096), (1, 16384), (512, 512), (1024, 4096)),
}


def row_bytes(t: str, n: int) -> int:
    """The bytes of n elements of type t."""
    _, blck, size = TYPES[t]
    return n // blck * size


def src(t: str, ne: tuple[int, ...], nb: list[int] | None = None) -> str:
    """One source of a test-file line: the type id, ne0..3 and nb0..3."""
    if nb is None:
        _, blck, size = TYPES[t]
        nb = [size, size * (ne[0] // blck)]
        nb += [nb[1] * ne[1], nb[1] * ne[1] * ne[2]]
    return " ".join(str(x) for x in (TYPES[t][0], *ne, *nb))


def f32_bits(x: float) -> int:
    """The bits of a float as the int32 of an op param."""
    return struct.unpack("<i", struct.pack("<f", x))[0]


def fa_line(op_fa: int, t: str, n: int, kv: int) -> str:
    """The test-file line of one FA case with the layouts of the model: q is the permuted view of the
    [256, 16, n] output of the rope, k and v are the views of the cache [256 x 4 heads, kv rows], the mask
    has n rows. O(1)."""
    d = FA_DK
    q = src("f32", (d, n, FA_HEADS, 1), [4, 4 * d * FA_HEADS, 4 * d, 4 * d * FA_HEADS * n])
    r1, r2 = row_bytes(t, d * FA_KV_HEADS), row_bytes(t, d)
    kvs = src(t, (d, kv, FA_KV_HEADS, 1), [TYPES[t][2], r1, r2, r1 * kv])
    mask = src("f16", (kv, n, 1, 1))
    params = [f32_bits(1.0 / math.sqrt(d)), 0, 0, 10]
    name = f"fa_{t}_q{n}_kv{kv}"
    return " ".join(str(x) for x in (op_fa, 0, d, FA_HEADS, n, 1, len(params), *params, 4)) + \
        f" {q} {kvs} {kvs} {mask} {name}"


# The flash attention of the vision encoder: head size 64, 16 heads, and the patch counts of a small image
# and of the 672-token image of decode-timeline (2688 patches).
VIT_DK, VIT_HEADS = 64, 16
VIT_POS = (1024, 2688)


def vit_line(op_fa: int, n: int) -> str:
    """The test-file line of one FA op of the vision encoder (tools/mtmd/models/qwen3vl.cpp): q is the
    permuted view of the Q half of the roped [64, 32, n] QK tensor, k and v are the F16 casts [64, n, 16] of
    build_attn, and the op has no mask. O(1)."""
    d, h = VIT_DK, VIT_HEADS
    q = src("f32", (d, n, h, 1), [4, 4 * d * 2 * h, 4 * d, 4 * d * 2 * h * n])
    kv = src("f16", (d, n, h, 1))
    params = [f32_bits(1.0 / math.sqrt(d)), 0, 0, 10]
    return " ".join(str(x) for x in (op_fa, 0, d, h, n, 1, len(params), *params, 3)) + f" {q} {kv} {kv} fa_vit_n{n}"


def ggml_op_value(name: str) -> int:
    """The value of one ggml_op name in the stage tree.

    Raises:
        FileNotFoundError: If build/fak/tree is not there (run build.sh on the box first)
    """
    text = (HERE / "tree/ggml/include/ggml.h").read_text()
    body = text[text.index("enum ggml_op {"):]
    body = body[:body.index("};")]
    names = re.findall(r"^\s*(GGML_OP_\w+)", body, re.M)
    return names.index(name)


# ---- The command file ----

HEADER = """\
# Phone stage "fak": the flash attention of HTP0 (the chunk cost model, the tile softmax HTP_FA_KERNEL_HMX2, the
# resident K and V, the decode spans) against the flash attention of HEAD, on one library set. The switch
# GGML_HEXAGON_FA_OPT selects the parts: A 0 (HEAD), B 1 (chunk model alone), T 2 (tile softmax alone), R 6 (tile
# softmax with resident K and V), S 8 (decode spans alone), C 3 (B + T), D 7 (C + resident), E 15 (D + decode spans,
# the preset). The op tests and the KL runs come first, thus a failure on the chip shows early and names its part.
# The 4B Q8_0 model, the Q8_0 cache with the FWHT rotation, n_batch = n_ubatch = 1024, 4 threads, flash attention on
# HTP0 (the flags of bench-kv variant b). The op tests also have the flash attention of the vision encoder.
#
# The libraries (build/fak/build.sh): the patched llama.cpp tree of HEAD (tests/sanitizers/llama-copy.sh) plus the
# candidate patches of build/fak/patches, built with the preset and the flags of scripts/build-native.sh.
#
# The runs ({n_runs}):
{run_text}
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,
# MemAvailable, the caps), the tool under timeout -s KILL (110 s or less), the exit code and the conditions after
# the run, then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about {tool_min:.0f} minutes of tool time plus
# about 10 s of gate and checks for each run, about {total_min:.0f} minutes, plus the waits for thermal status 0.
# Then, on the box: build/fak/stage.py table
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files and the KL bases."""
    bins = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    libs = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l {KLD_BASE} {WIKI} {KVKL_16K} {KVKL_TEXT} "
        f"$(dirname {MODEL})/ | grep -c -E \"kvb|kld|raw|4B-Q8_0\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/tests {PHONE}/out'",
        f"{ADB} push {bins} {PHONE}/bin/",
        f"{ADB} push {libs} {PHONE}/lib/",
        f"{ADB} push {LAPTOP_STAGE}/phone/tests/fa4b.txt {LAPTOP_STAGE}/phone/tests/vit.txt {PHONE}/tests/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
        f"{ADB} shell 'echo gzip: $(command -v gzip) timeout: $(command -v timeout)'",
    ]


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    b, v = run.block, run.variant
    stem = f"{PHONE}/out/{run.name}"
    env = " ".join(x for x in (LIB_ENV, f"GGML_HEXAGON_FA_OPT={v.opt}", "GGML_HEXAGON_PROFILE=1" if b.profile else "")
                   if x)
    if b.tool.startswith("test-backend-ops"):
        args = f"--test-file {PHONE}/tests/{b.args}.txt" if b.args in TEST_FILES else b.args
        cmd = f"{PHONE}/bin/{b.tool} -b HTP0 {args}"
        prefix = f"{MARKER}; "
    elif b.tool == "kvkl":
        cmd = f"{PHONE}/bin/kvkl -m {MODEL} -ctk {b.kv} -ctv {b.kv} -dev HTP0 -t 4 {b.args}"
        prefix = ""
    elif b.tool == "llama-perplexity":
        cmd = (f"{PHONE}/bin/llama-perplexity -m {MODEL} -dev HTP0 -ngl 99 -t 4 -fa on -ctk {b.kv} -ctv {b.kv} "
               f"-f {WIKI} {b.args}")
        prefix = ""
    else:
        cmd = f"{PHONE}/bin/llama-bench -m {MODEL} {BENCH_ARGS} -ctk {b.kv} -ctv {b.kv} {b.args}"
        prefix = ""
    if b.profile:
        # One profile line for each op: gzip (when the phone has it) keeps the file small. The parser reads
        # a gzip file and a plain file alike.
        tool = (f"{{ Z=cat; command -v gzip > /dev/null && Z=\"gzip -1\"; set -o pipefail; "
                f"timeout -s KILL {b.limit} env {env} {cmd} 2>&1 > {stem}.out | $Z > {stem}.log.z; }}; ")
    else:
        tool = f"{{ timeout -s KILL {b.limit} env {env} {cmd} > {stem}.out 2> {stem}.log; }}; "
    shell = (f"{prefix}sh {PHONE}/bin/gate.sh {b.gate_kb} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
             f"{tool}echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt")
    title = "REAL-MODEL Qwen3.5-4B-Q8_0" if b.gate_kb == MODEL_KB else "OPS"
    return ["#", f"# {title}: {run.name}, {b.text}, {v.key.upper()}: GGML_HEXAGON_FA_OPT={v.opt} ({v.text})",
            THERMAL, f"{ADB} shell '{shell}'", PGREP]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone
    directory goes only when the pull has each of its files."""
    return [
        "#", "# ---- The outputs ----", "#", THERMAL,
        f"{ADB} shell 'pgrep -x llama-bench; pgrep -x kvkl; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]")
OWN_LIB_RE = re.compile(r"^lib(llama|ggml|mtmd)")


def missing_needed(phone: Path) -> list[str]:
    """The llama, ggml and mtmd libraries that a program or a library of phone/ needs (the NEEDED entries
    of readelf -d) and that phone/lib does not have. The system libraries of the phone are not checked.
    O(files)."""
    out = []
    for f in sorted([*(phone / "bin").iterdir(), *(phone / "lib").glob("*.so")]):
        if f.suffix == ".sh":
            continue
        dyn = subprocess.run(["readelf", "-d", str(f)], capture_output=True, text=True, check=True).stdout
        for lib in NEEDED_RE.findall(dyn):
            if OWN_LIB_RE.match(lib) and not (phone / "lib" / lib).exists():
                out.append(f"{f.relative_to(phone)} needs {lib}")
    return out


def write_files() -> int:
    """Write phone/tests/fa4b.txt, phone/SHA256SUMS and phone-commands.txt. The binaries and the
    libraries of phone/ come from build.sh. The function stops when a file of STAGE_FILES is missing, or
    when a program or a library needs a llama, ggml or mtmd library that phone/lib does not have."""
    phone = HERE / "phone"
    missing = [f for f in STAGE_FILES if not (phone / f).exists()]
    if missing:
        print(f"stage.py: {phone} has no {missing}: run tools/stages/fak/build.sh", file=sys.stderr)
        return 1
    needed = missing_needed(phone)
    if needed:
        print("stage.py: phone/lib has no library that a file needs:\n  " + "\n  ".join(needed), file=sys.stderr)
        return 1
    (phone / "tests").mkdir(parents=True, exist_ok=True)
    op_fa = ggml_op_value("GGML_OP_FLASH_ATTN_EXT")
    vit = [vit_line(op_fa, n) for n in VIT_POS]
    lines = [fa_line(op_fa, t, n, kv) for t, shapes in FA_SHAPES.items() for n, kv in shapes] + vit
    (phone / "tests/fa4b.txt").write_text("\n".join(lines) + "\n")
    (phone / "tests/vit.txt").write_text("\n".join(vit) + "\n")
    files = sorted(p for p in phone.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(phone)}" for p in files]
    (phone / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    runs = all_runs()
    tool = sum(EST_S[r.block.key] for r in runs)
    run_text = "\n".join(f"#   {b.key:7s} {b.text}; {', '.join(x.upper() for x in b.variants)}; {b.rounds} round(s)"
                         for b in BLOCKS.values())
    head = HEADER.format(n_runs=len(runs), run_text=run_text, tool_min=tool / 60,
                         total_min=(tool + 10 * len(runs)) / 60)
    out = head.rstrip("\n").split("\n") + setup_lines()
    for run in runs:
        out += run_lines(run)
    out += output_lines()
    (HERE / "phone-commands.txt").write_text("\n".join(out) + "\n")
    print(f"phone-commands.txt: {len(out)} lines, {len(runs)} runs, tool time about {tool / 60:.1f} min")
    return 0


# ---- The table ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
KL_RE = re.compile(r"^Mean\s+KLD:\s+([\d.eE+-]+)\s+±\s+([\d.eE+-]+)", re.M)
TOP_RE = re.compile(r"^Same top p:\s+([\d.]+)", re.M)
KLMAX_RE = re.compile(r"^Maximum KLD:\s+([\d.eE+-]+)", re.M)
TEST_RE = re.compile(r"^\s*(\d+)/(\d+) tests passed", re.M)
# A case line of test-backend-ops test, after the color codes go: "OP(params): OK" or, for a failed case,
# "[OP] ERR = 0.0017 > 0.0005   OP(params): FAIL".
CASE_RE = re.compile(r"^(?:\[\w+\] ERR = ([\d.eE+-]+) > [\d.eE+-]+\s+)?\s*FLASH_ATTN_EXT\((.*)\): (OK|FAIL)", re.M)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PROF_RE = re.compile(r"profile-op ([A-Z0-9_+]+)\|.*\|usec (\d+) cycles")


@dataclass
class Result:
    """The parsed files of one run. ok is False when the run did not run or failed. flags names each
    condition that makes the run not comparable (changed caps, heat)."""
    run: Run
    ok: bool
    flags: list[str]
    caps: str
    out: str
    log: str

    def bench(self) -> dict[tuple[int, int, int], float]:
        """The llama-bench rates: (n_prompt, n_gen, n_depth) -> the median of the samples in t/s."""
        rates = {}
        for line in self.out.splitlines():
            if line.startswith("{"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rates[(rec["n_prompt"], rec["n_gen"], rec["n_depth"])] = statistics.median(rec["samples_ts"])
        return rates


def read_text(path: Path) -> str:
    """A text file, or a gzip file of the phone (gzip -1), or an empty string."""
    for p in (path, path.with_name(path.name + ".z")):
        if p.exists():
            data = p.read_bytes()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data.decode(errors="replace")
    return ""


def read_result(root: Path, run: Run) -> Result:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    gate = read_text(root / f"{run.name}-gate.txt")
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
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        flags.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    if after and after.group(1) not in ("", "0"):
        flags.append(f"thermal {after.group(1)} after the run")
    return Result(run, ok, flags, caps, read_text(root / f"{run.name}.out"), read_text(root / f"{run.name}.log"))


def usable(res: Result | None, include_all: bool) -> bool:
    """True when the run goes into the medians."""
    return res is not None and res.ok and (include_all or not res.flags)


def op_table(results: dict[str, Result]) -> list[str]:
    """The test-backend-ops runs: the passed count, and each failed case with its error. A failed case of
    both A and E is a failure that HEAD has too."""
    out = ["== 1. test-backend-ops FLASH_ATTN_EXT on HTP0 against the CPU (the error is the NMSE of test-backend-ops) =="]
    for key in ("ops", "suite1", "suite2"):
        fails_of: dict[str, dict[str, str]] = {}
        for v in BLOCKS[key].variants:
            res = results.get(f"{key}-1-{v}")
            if res is None:
                out.append(f"  {key}-1-{v}: no output")
                continue
            text = ANSI_RE.sub("", res.out + res.log)
            passed = TEST_RE.findall(text)
            fails = {c: err or "?" for err, c, st in CASE_RE.findall(text) if st == "FAIL"}
            fails_of[v] = fails
            summary = ", ".join(f"{a}/{b}" for a, b in passed) or "no summary line"
            out.append(f"  {key:7s} {v.upper()}: passed {summary}; {len(fails)} FAIL"
                       + (f"; flags: {', '.join(res.flags)}" if res.flags else ""))
        for v in fails_of:
            if v == "a" or "a" not in fails_of:
                continue
            only_v = sorted(set(fails_of[v]) - set(fails_of["a"]))
            only_a = sorted(set(fails_of["a"]) - set(fails_of[v]))
            out.append(f"    FAIL in {v.upper()} and not in A: {len(only_v)}; in A and not in {v.upper()}: {len(only_a)}")
            for c in only_v[:16]:
                out.append(f"      {v.upper()} only, ERR {fails_of[v][c]}: {c}")
    return out


PERF_RE = re.compile(r"FLASH_ATTN_EXT\(name=(\w+),.*?\):\s+(\d+) runs -\s+([\d.]+) us/run")


def vit_table(results: dict[str, Result]) -> list[str]:
    """The perf runs of the vision encoder ops: the us of one op for each variant, and the ratio to A."""
    out = ["", "== 1b. The FLASH_ATTN_EXT ops of the vision encoder (test-backend-ops perf): us of one op, the ratio to A =="]
    us: dict[str, dict[str, float]] = {}
    for v in BLOCKS["vit"].variants:
        res = results.get(f"vit-1-{v}")
        if res is None or not res.ok:
            out.append(f"  vit-1-{v}: no usable run")
            continue
        for name, _, val in PERF_RE.findall(ANSI_RE.sub("", res.out)):
            us.setdefault(name, {})[v] = float(val)
    for name, per in us.items():
        base = per.get("a")
        cells = [f"{v.upper()} {per[v]:9.1f}" + (f" ({per[v] / base:.3f})" if base and v != "a" else "")
                 for v in BLOCKS["vit"].variants if v in per]
        out.append(f"  {name:16s} " + " | ".join(cells))
    return out


def kl_table(results: dict[str, Result]) -> list[str]:
    """The KL runs: the mean KL, its error, the maximum and the top-1 share, and the kvkl summary lines."""
    out = ["", "== 2. KL against the naive oracle (4B Q8_0, Q8_0 cache) =="]
    for key in ("klp", "kld"):
        for v in BLOCKS[key].variants:
            res = results.get(f"{key}-1-{v}")
            if res is None:
                out.append(f"  {key}-1-{v}: no output")
                continue
            text = res.out + res.log
            m, mx, top = KL_RE.search(text), KLMAX_RE.search(text), TOP_RE.search(text)
            cell = (f"mean {m.group(1)} ± {m.group(2)}, max {mx.group(1) if mx else '?'}, top-1 "
                    f"{top.group(1) if top else '?'} %") if m else "no KLD line"
            out.append(f"  {key:5s} {v.upper()}: {cell}" + (f"; flags: {', '.join(res.flags)}" if res.flags else ""))
    for v in BLOCKS["kl16"].variants:
        res = results.get(f"kl16-1-{v}")
        rows = [ln for ln in (res.out.splitlines() if res else []) if re.match(r"^(decode|all)\b", ln)]
        out.append(f"  kl16  {v.upper()}: " + (" | ".join(rows) if rows else "no summary line")
                   + (f"; flags: {', '.join(res.flags)}" if res and res.flags else ""))
    return out


# The rows of the rate table: the text, the block and the llama-bench key (n_prompt, n_gen, n_depth).
ROWS = (
    ("pp512 d0", "p", (512, 0, 0)),
    ("pp512 d4096", "p", (512, 0, 4096)),
    ("pp1024 d3072", "q", (1024, 0, 3072)),
    ("tg32 d0", "t", (0, 32, 0)),
    ("tg32 d4096", "t", (0, 32, 4096)),
    ("tg32 d16384", "l", (0, 32, 16384)),
    ("tg32 d4096 F16 cache", "f", (0, 32, 4096)),
    ("tg32 d16384 F16 cache", "f", (0, 32, 16384)),
)


def num(x: float) -> str:
    """A rate with 2 decimals below 100 and 1 decimal from 100."""
    return f"{x:.2f}" if x < 100 else f"{x:.1f}"


def rate_table(results: dict[str, Result], include_all: bool) -> list[str]:
    """The t/s table: per row and variant the median of the rounds, the paired difference to A, the lowest
    and the highest round and the count of rounds. O(runs)."""
    out = ["", "== 3. Rates (t/s): the median of the rounds, the difference to A (the median of the paired ratios), "
               "[the lowest and the highest round] and the round count ==",
           f"  {'measurement':24s}| " + " | ".join(f"{k.upper()} {VARIANTS[k].opt:<2d}{'':27s}" for k in TABLE_KEYS)]
    for text, key, bkey in ROWS:
        block = BLOCKS[key]
        per: dict[str, dict[int, float]] = {k: {} for k in block.variants}
        for rnd in range(1, block.rounds + 1):
            for k in block.variants:
                res = results.get(f"{key}-{rnd}-{k}")
                if usable(res, include_all):
                    val = res.bench().get(bkey)
                    if val is not None:
                        per[k][rnd] = val
        cells = []
        for k in TABLE_KEYS:
            vals = per.get(k, {})
            if not vals:
                cells.append(f"{'-':32s}")
                continue
            cell = num(statistics.median(vals.values()))
            if k != "a" and "a" in per:
                ratios = [vals[r] / per["a"][r] for r in vals if r in per["a"]]
                cell += f" {100 * (statistics.median(ratios) - 1):+.1f}%" if ratios else " ?"
            cell += f" [{num(min(vals.values()))}-{num(max(vals.values()))}] n{len(vals)}"
            cells.append(f"{cell:32s}")
        out.append(f"  {text:24s}| " + " | ".join(cells))
    # The release rule: the Q8_0 decode of E against the F16 decode of E, per round.
    for depth, qkey in ((4096, "t"), (16384, "l")):
        ratios = []
        for rnd in range(1, 4):
            q = results.get(f"{qkey}-{rnd}-e")
            f = results.get(f"f-{rnd}-e")
            if usable(q, include_all) and usable(f, include_all):
                qv, fv = q.bench().get((0, 32, depth)), f.bench().get((0, 32, depth))
                if qv and fv:
                    ratios.append(qv / fv)
        if ratios:
            out.append(f"  release rule, E, tg32 d{depth}: Q8_0 over F16 {statistics.median(ratios):.3f} "
                       f"(rounds {', '.join(f'{r:.3f}' for r in ratios)}); it must be 1.000 or more")
    return out


def fa_shape(line: str) -> tuple[int, int] | None:
    """The (queries, KV rows) of a FLASH_ATTN_EXT profile line: the dims field holds "ne0:ne1:ne2:ne3" of each
    source, q first and k second."""
    fields = line.split("|")
    if len(fields) < 3:
        return None
    dims = fields[2].split(" x ")
    try:
        return int(dims[0].split(":")[1]), int(dims[1].split(":")[1])
    except (IndexError, ValueError):
        return None


def prof_table(results: dict[str, Result]) -> list[str]:
    """The FA ops of the profile runs by shape: the op count, the median time of one op in us and the kernel
    form, and the share of FA in the DSP time of the run. llama-bench fills the depth for each test, thus
    the run holds the ubatches of the fill, the pp1024 ubatch at d3072 (1024 queries, 4096 KV rows) and the
    8 decode tokens (1 query, 3073 to 3080 KV rows at d3072)."""
    out = ["", "== 4. The FA ops of the profile runs (GGML_HEXAGON_PROFILE=1): queries x KV rows, ops, the median us of "
               "one op, the kernel =="]
    for k in BLOCKS["prof"].variants:
        res = results.get(f"prof-1-{k}")
        if res is None or not res.ok:
            out.append(f"  {k.upper()}: no usable run")
            continue
        by_shape: dict[tuple[int, int], list[int]] = {}
        kernel: dict[tuple[int, int], Counter] = {}
        total = fa_total = 0
        for line in res.log.splitlines():
            if "profile-op " not in line or "OPBATCH" in line:
                continue
            m = PROF_RE.search(line)
            if not m:
                continue
            us = int(m.group(2))
            total += us
            if "FLASH_ATTN_EXT" not in m.group(1).split("+"):
                continue
            fa_total += us
            shape = fa_shape(line)
            if shape is None:
                continue
            # A decode token at the depth d has d + t KV rows: the class keeps the depth to 256 rows.
            key = (shape[0], shape[1] if shape[0] > 32 else shape[1] // 256 * 256)
            by_shape.setdefault(key, []).append(us)
            fields = line.split("|")
            kernel.setdefault(key, Counter())[fields[5].split(" vtcm")[0] if len(fields) > 5 else "?"] += 1
        out.append(f"  {k.upper()} {VARIANTS[k].opt:2d}: FA {fa_total / 1e3:.1f} ms of {total / 1e3:.1f} ms of DSP op time")
        for key in sorted(by_shape, key=lambda x: (-x[0], x[1])):
            vals = by_shape[key]
            rows = f"{key[1]}+" if key[0] <= 32 else f"{key[1]}"
            out.append(f"      q {key[0]:4d} kv {rows:>6s}: {len(vals):3d} ops, {statistics.median(vals):9.1f} us, "
                       + ", ".join(f"{n} x{c}" for n, c in kernel[key].items()))
    return out


def checks(results: dict[str, Result]) -> list[str]:
    """The conditions of the runs."""
    runs = all_runs()
    got = [results[r.name] for r in runs if r.name in results]
    out = [f"{len(got)} of {len(runs)} runs have a gate file, {sum(r.ok for r in got)} ran with exit code 0, "
           f"{sum(r.ok and not r.flags for r in got)} have no flag"]
    caps = Counter(r.caps for r in got if r.ok)
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.most_common()))
    for r in got:
        if r.flags:
            out.append(f"  {r.run.name}: " + ", ".join(r.flags))
        if r.ok and r.run.variant.opt != 15 and f"options 0x{r.run.variant.opt:x}" not in r.log \
                and not r.run.block.tool.startswith("test-backend-ops") and not r.run.block.profile:
            out.append(f"  {r.run.name}: the log has no line of GGML_HEXAGON_FA_OPT={r.run.variant.opt}")
    return out


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    results = {r.name: read_result(root, r) for r in all_runs() if (root / f"{r.name}-gate.txt").exists()}
    for part in (checks(results), op_table(results), vit_table(results), kl_table(results),
                 rate_table(results, include_all), prof_table(results)):
        print("\n".join(part))
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("files", help="write the test files, SHA256SUMS and the phone command file")
    t = sub.add_parser("table", help="print the tables from the pulled logs")
    t.add_argument("--root", type=Path, default=HERE / "phone-out")
    t.add_argument("--all", action="store_true", help="also use the runs with changed caps or heat")
    a = ap.parse_args()
    if a.cmd == "files":
        return write_files()
    return table(a.root, a.all)


if __name__ == "__main__":
    sys.exit(main())
