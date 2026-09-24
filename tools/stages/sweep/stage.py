#!/usr/bin/env python3
"""The phone stage "sweep": the HMX peak of HTP0 and the rate of each prefill op shape of the 4B Q8_0.

Usage (tools/stages/sweep/build.sh makes the binaries first and then runs "files"):
    stage.py files                        write phone/ (tests, libraries, SHA256SUMS) and phone-commands.txt
    stage.py plan                         print the kernel that the host selects for each case (no phone)
    stage.py table [--root DIR] [--all]   print the tables from the pulled outputs (phone-out)

The questions:
    1. The f16 rate of the HMX engine on this part (i8read: deep MAC chains from VTCM, no feed).
    2. The rate of each MUL_MAT shape of the 4B for n = 1 to 1024 tokens, for Q8_0, F16 and Q4_0
       weights, and the kernel of each case (HVX GEMV, HVX multirow, HMX 2D).
    3. The row count where the HMX path is faster than the HVX path (GGML_HEXAGON_MM_SELECT=2).
    4. FLASH_ATTN_EXT with the 4B attention shape (head dim 256, 16 query heads, 4 KV heads), Q8_0 and
       F16 K and V, 1 and 1024 queries, 512 to 16384 KV rows.
    5. GATED_DELTA_NET (chunk kernel and sequential kernel) and SSM_CONV at the 4B prefill shapes.
    6. For each 4B shape at 1024 tokens: the time that the HMX works, and the phase that holds the
       HMX when it waits (GGML_HEXAGON_PROFILE=3 trace).

Each op runs alone: GGML_HEXAGON_OPFUSION=0. test-backend-ops perf puts one node many times into one
graph, and with the fusion on the backend merges these copies into MUL_MAT_NX of 4 (the same weight
4 times, one activation conversion). The run "nx" measures that form on purpose.

A run name is the stem of its three output files on the phone: <run>-gate.txt (the conditions and
the exit code), <run>.out (stdout) and <run>.log (stderr). The table uses a run when its gate passed,
its exit code is 0, the CPU caps did not change, the thermal status after it is 0, and the screen stayed
on without the keyguard. --all also uses the other runs. The table only reads files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import re
import shutil
import statistics
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# The script lives in tools/stages/sweep, and build/sweep/stage.py is a link to it. The stage files
# (phone/, phone-commands.txt, phone-out/) are in build/sweep of the repository in each case.
REPO = Path(__file__).resolve().parents[3]
HERE = REPO / "build/sweep"

# ---- The stage paths and the phone lines ----

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/sweep"
LAPTOP_STAGE = "build/sweep"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/sweep"
# The ls of this file makes the laptop runner treat a line as a model run: it waits for the unlocked
# phone, stops the Qwen app, wakes the screen, checks MemAvailable against MIN_2B_MB (6000 MB) and
# prints the caps and the screen state. No run of this stage loads the file.
MARKER = "ls /data/local/tmp/qwen/models/Qwen3.5-2B-Q8_0.gguf > /dev/null"
GATE_KB = 2097152
LIB_ENV = f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib"
THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x test-backend-op; pgrep -x run_main_on_he; echo pgrep-done'"
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
SCREEN = ('screen=$(dumpsys power | grep -o "mWakefulness=[A-Za-z]*" | head -n 1 | cut -d= -f2)'
          ' keyguard=$(dumpsys window | grep -m1 -o "isKeyguardShowing=[a-z]*" | cut -d= -f2)')
BEFORE = f'echo "before: nsp={NSP} {SCREEN}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP} {SCREEN}"')

# The libraries of build/bench-kv/phone/lib that test-backend-ops loads (the APK set of HEAD e8a3a07
# plus the switch GGML_HEXAGON_FWHT), and the binary of build/perf-tbo from the same tree and flags.
LIBS = ("libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-htp-v79.so", "libggml-opencl.so",
        "libggml.so", "libllama-common.so", "libllama.so")
HMX_FILES = ("i8read.so", "hmx_sustain.so", "run_main_on_hexagon", "librun_main_on_hexagon_skel.so", "i8read.farf",
             "hmx_sustain.farf", "run_main_on_hexagon.farf")

# ---- The facts of the part and of the model ----

VTCM = 8388608         # the session VTCM, bytes (the hwinfo line of the log gives the MB)
N_THREADS = 6          # the HVX threads of the session (the hwinfo line of the log gives them)
CLOCK_MHZ = 2112.0     # the DSP clock when no profile line gives the measured value
TILE = 2048            # bytes of one 32 x 32 f16 HMX tile
TILE_MAC_FLOP = 2 * 32 * 32 * 32

# The weight matrices of one 1024-token prefill ubatch of the 4B (GGUF of weights/gguf, measured
# 2026-09-24): (k, m, count per ubatch, the tensors). 24 GDN layers, 8 attention layers. The head is
# not in the list, because a prefill ubatch applies it to one token only.
MM_4B = (
    (2560, 9216, 64, "ffn_gate, ffn_up"),
    (9216, 2560, 32, "ffn_down"),
    (2560, 8192, 32, "attn_qkv (GDN), attn_q"),
    (2560, 4096, 24, "attn_gate (GDN)"),
    (4096, 2560, 32, "ssm_out (GDN), attn_output"),
    (2560, 1024, 16, "attn_k, attn_v"),
)
ALPHA_BETA = (2560, 32, 48, "ssm_alpha, ssm_beta (F32)")
FA_DK, FA_HEADS, FA_KV_HEADS, FA_LAYERS = 256, 16, 4, 8
GDN_D, GDN_K_HEADS, GDN_V_HEADS, GDN_LAYERS, CONV_DIM, CONV_K = 128, 16, 32, 24, 8192, 4
# bench-kv, 2026-09-24 (HTP0, the app configuration, Q8_0 K and V): one 1024-token ubatch.
BENCH_KV = {"weight MUL_MAT d0": 559.0, "weight MUL_MAT d3072": 709.0, "FA d0": 30.0, "FA d3072": 105.0}

# name -> (ggml type id, elements per block, bytes per block)
TYPES = {"f32": (0, 1, 4), "f16": (1, 1, 2), "q4_0": (2, 32, 18), "q8_0": (8, 32, 34)}
# The repacked tile of 32 x 32 weights and its VTCM slot (htp/matmul-ops.h)
TILE_BYTES = {"q4_0": 576, "q8_0": 1088}
TILE_ALIGNED = {"q4_0": 640, "q8_0": 1152}
QUANT = ("q4_0", "q8_0")


def ggml_ops() -> dict[str, int]:
    """Read the ggml_op enum of the stage tree and return the value of each name.

    Raises:
        FileNotFoundError: If the tree of build/bench-kv/src is not there (run the script on the box)
    """
    src = (REPO / "build/bench-kv/src/ggml/include/ggml.h").read_text()
    body = src[src.index("enum ggml_op {"):]
    body = body[:body.index("};")]
    return {n: i for i, n in enumerate(re.findall(r"^\s*(GGML_OP_\w+)", body, re.M))}


def up(v: int, a: int) -> int:
    """The value v aligned up to a multiple of a."""
    return (v + a - 1) // a * a


def down(v: int, a: int) -> int:
    """The value v aligned down to a multiple of a."""
    return v // a * a


def cdiv(a: int, b: int) -> int:
    """The quotient of a and b, rounded up."""
    return -(-a // b)


def row_bytes(t: str, n: int) -> int:
    """The bytes of n elements of type t."""
    _, blck, size = TYPES[t]
    return n // blck * size


# ---- The cases ----

@dataclass(frozen=True)
class Case:
    """One op of a test file.

    op is "mm", "fa", "gdn" or "conv". For "mm": k the reduction, m the weight rows, n the tokens, t the
    weight type. For "fa": n the queries, kv the KV rows, t the K and V type. For "gdn" and "conv": n the
    tokens. The name is unique in the stage and goes into the test file, thus the output names the case.
    """
    op: str
    t: str
    n: int
    k: int = 0
    m: int = 0
    kv: int = 0
    tag: str = ""

    @property
    def name(self) -> str:
        """The case id: letters, digits, "_" and "x" only."""
        base = {"mm": f"mm_{self.t}_{self.k}x{self.m}_n{self.n}",
                "fa": f"fa_{self.t}_q{self.n}_kv{self.kv}",
                "gdn": f"gdn_n{self.n}", "conv": f"conv_n{self.n}"}[self.op]
        return f"{base}_{self.tag}" if self.tag else base

    @property
    def flops(self) -> float:
        """The FLOP of the op. FA counts every KV block, because the kernel skips no masked block.
        GDN counts the products of the chunk form (64 tokens a chunk)."""
        if self.op == "mm":
            return 2.0 * self.k * self.m * self.n
        if self.op == "fa":
            return 2.0 * self.n * self.kv * (FA_DK + FA_DK) * FA_HEADS
        if self.op == "gdn":
            c, d = 64, GDN_D
            return 2.0 * (self.n / c) * GDN_V_HEADS * (5 * c * c * d + 3 * c * d * d)
        return 2.0 * self.n * CONV_DIM * CONV_K

    @property
    def stream_bytes(self) -> float:
        """The bytes that the op must read one time: the weights (mm), K and V (fa), all inputs (else)."""
        if self.op == "mm":
            return row_bytes(self.t, self.k) * self.m
        if self.op == "fa":
            return 2.0 * row_bytes(self.t, FA_DK) * FA_KV_HEADS * self.kv
        return float(self.op_size)

    @property
    def op_size(self) -> int:
        """The bytes of dst and of each source, as test-backend-ops counts them (ggml_nbytes)."""
        if self.op == "mm":
            return row_bytes(self.t, self.k) * self.m + 4 * self.k * self.n + 4 * self.m * self.n
        if self.op == "fa":
            q = 4 * FA_DK * FA_HEADS * self.n
            kv = 2 * row_bytes(self.t, FA_DK * FA_KV_HEADS) * self.kv
            return q + kv + 2 * self.kv * self.n + q
        if self.op == "gdn":
            qk = 2 * 4 * GDN_D * GDN_K_HEADS * self.n
            v = 4 * CONV_DIM * self.n
            gb = 2 * 4 * GDN_V_HEADS * self.n
            st = 4 * GDN_D * GDN_D * GDN_V_HEADS
            return qk + v + gb + st + 4 * GDN_D * GDN_V_HEADS * (self.n + GDN_D)
        return 4 * CONV_DIM * (self.n + CONV_K - 1) + 4 * CONV_DIM * CONV_K + 4 * CONV_DIM * self.n


def mm(t: str, k: int, m: int, n: int, tag: str = "") -> Case:
    """A MUL_MAT case."""
    return Case("mm", t, n, k=k, m=m, tag=tag)


def fa(t: str, n: int, kv: int, tag: str = "") -> Case:
    """A FLASH_ATTN_EXT case."""
    return Case("fa", t, n, kv=kv, tag=tag)


# ---- The kernel that the host selects (a port of ggml-hexagon.cpp and htp/matmul-ops.h) ----

@dataclass
class MmPlan:
    """The host choice for one MUL_MAT: the kernel, and for the HMX path the chunks and the tile counts."""
    kernel: str
    mc: int = 0
    nc: int = 0
    pipeline: bool = False
    act_threads: int = 0
    vtcm: int = 0
    mblocks: int = 0
    out_tiles: int = 0
    tile_macs: int = 0

    def text(self) -> str:
        """One short line."""
        if not self.kernel.startswith("HMX"):
            return self.kernel
        return (f"{self.kernel} mc {self.mc} nc {self.nc} {'pipe' if self.pipeline else 'serial'} "
                f"act-thr {self.act_threads} weight passes {self.mblocks} vtcm {self.vtcm}")


def hmx_2d_vtcm(t: str, k: int, mc: int, nc: int, pipeline: bool, act_threads: int) -> int:
    """The VTCM bytes of the HMX 2D layout (htp_mm_hmx_vtcm_layout_build, HTP_MM_KERNEL_HMX_2D)."""
    quant = t in QUANT
    stride = k * (2 if t == "f16" else 4)
    vec = 2 * k
    weight = up((nc // 32) * (k // 32) * TILE_ALIGNED[t], TILE) if quant else up(nc * stride, TILE)
    act = up(mc * vec, TILE)
    out = up(mc * nc * 2, TILE)
    scratch = up(nc * vec, TILE)
    group_a = TILE + act
    group_b = weight * (2 if pipeline else 1) + out * (2 if pipeline else 1) + scratch * (2 if pipeline else 1)
    min_f32 = up(act_threads * 4 * k * 4, 128)
    group_c = up(min(act_threads * 64 * k * 4, max(min_f32, group_b)), 128)
    return group_a + max(group_b, group_c)


def hmx_2d_chunks(t: str, k: int, m_pad: int, n: int, m_cost: int, pipeline: bool) -> tuple[int, int] | None:
    """The (mc, nc) of htp_mm_hmx_compute_chunks, or None. O(n / 32)."""
    quant = t in QUANT
    stride = k * (2 if t == "f16" else 4)
    vec = 2 * k
    qstride = (k // 32) * TILE_ALIGNED[t] // 32 if quant else 0
    per_n = (2 if pipeline else 1) * (qstride if quant else stride) + (2 * vec if pipeline else vec)
    per_m = vec
    per_mn = (2 if pipeline else 1) * 2
    overhead = (7 if pipeline else 5) * TILE + 256
    usable = VTCM - overhead
    best = None
    n_max = down(min(n, usable // per_n), 32)
    for nc in range(n_max, 31, -32):
        n_fixed = nc * per_n
        if n_fixed >= usable:
            continue
        mc = min(down((usable - n_fixed) // (per_m + nc * per_mn), 32), m_pad)
        if mc == 0:
            continue
        cost = cdiv(m_pad, mc) * n * 3 + cdiv(n, nc) * m_cost * 2
        key = (cost, -mc * nc)
        if best is None or key < best[0]:
            best = (key, mc, nc)
    return (best[1], best[2]) if best else None


def mm_plan(c: Case, mm_select: int = 3) -> MmPlan:
    """The kernel of one MUL_MAT case, as ggml_hexagon_precompute_matmul_params_impl gives it."""
    t, k, m, n = c.t, c.k, c.m, c.n
    m_pad = up(m, 32) if t in QUANT else m
    hmx_ok = mm_select >= 3 and m_pad % 32 == 0 and k % 32 == 0 and n > 4
    if hmx_ok:
        n_pad = up(n, 32)
        for pipeline in (n > 4, False):
            ch = hmx_2d_chunks(t, k, n_pad, m_pad, n, pipeline)
            if ch is None:
                continue
            mc, nc = ch
            for act_threads in (N_THREADS, N_THREADS // 2, 1):
                size = hmx_2d_vtcm(t, k, mc, nc, pipeline, act_threads)
                if size <= VTCM:
                    row_tiles = sum(cdiv(min(mc, n - r), 32) for r in range(0, n, mc))
                    return MmPlan("HMX_2D", mc, nc, pipeline, act_threads, size, cdiv(n, mc),
                                  row_tiles * m_pad // 32, row_tiles * m_pad // 32 * (k // 32))
            if not pipeline:
                break
    if t in QUANT:
        if n < N_THREADS:
            kern = "HVX_QUANT_MULTIROW" if 2 <= n <= 4 else "HVX_QUANT_BLOCK"
        else:
            kern = "HVX_QUANT_ROW"
        return MmPlan(kern)
    return MmPlan("HVX_F16_F16_VTCM" if t == "f16" else "HVX_F32_F32_VTCM")


@dataclass
class FaPlan:
    """The host choice for one FLASH_ATTN_EXT."""
    kernel: str
    br: int = 0
    bc: int = 0
    kv_blocks: int = 0
    threads: int = 0
    pipeline: bool = False
    vtcm: int = 0
    out_tiles: int = 0
    tile_macs: int = 0
    qk_macs: int = 0

    def text(self) -> str:
        """One short line."""
        return (f"{self.kernel} Br {self.br} Bc {self.bc} kv-blocks {self.kv_blocks} threads {self.threads} "
                f"{'pipe' if self.pipeline else 'seq'} vtcm {self.vtcm}")


def fa_vtcm(g: int, br: int, bc: int, pipeline: bool) -> int:
    """The VTCM bytes of the HMX FA layout (hmx_fa_vtcm_layout_build), Q in f32, DK = DV = 256."""
    dk = dv = FA_DK
    g_br = up(g * br, 32)
    q_t, o_t = up(g_br * dk * 2, TILE), up(g_br * dv * 2, TILE)
    k_t, v_t, s_t = up(bc * dk * 2, TILE), up(bc * dv * 2, TILE), up(g_br * bc * 2, TILE)
    d_t = (g_br // 32) * TILE
    two = 2 if pipeline else 1
    off = q_t + 2 * o_t + d_t * (1 + two)
    off = up(off, TILE)
    col, row = up(g_br * 4, 256), up(bc * 2, 256)
    group_b = two * (k_t + v_t + 2 * s_t) + 2 * col + row * 2 * N_THREADS
    off += max(group_b, up(g_br * dk * 4, 128))
    kd, vd = up(bc * up(dk * 2, 128), 128), up(bc * up(dv * 2, 128), 128)
    mbuf = up(br * up(bc * 2, 128), 256) * 4
    return off + 2 * kd + 2 * vd + 2 * col + 512 + mbuf + up(g_br * 2, 128)


def fa_plan(c: Case) -> FaPlan:
    """The kernel of one FA case (ggml_hexagon_precompute_flash_attn_params, hmx_fa_find_chunk_size).
    O(n / Br_unit x kv / 64) layouts."""
    g, qo, kv = FA_HEADS // FA_KV_HEADS, c.n, c.kv
    br_unit, bc_unit = cdiv(32, g), 64
    can_pipe = kv >= 3 * bc_unit and N_THREADS >= 2
    br_max = down(qo, br_unit) if qo >= br_unit else br_unit
    bc_limit = down(kv // 3, bc_unit) if can_pipe else (down(kv, bc_unit) if kv >= bc_unit else bc_unit)
    best = None
    for br in range(br_max, br_unit - 1, -br_unit):
        for bc in range(bc_limit, bc_unit - 1, -bc_unit):
            if fa_vtcm(g, br, bc, can_pipe) <= VTCM:
                kvb = cdiv(kv, bc)
                thr = N_THREADS if kvb >= 3 else 1
                vec_cnt = cdiv(br * g, 64)
                use = min(vec_cnt, thr)
                cost = cdiv(qo, br) * (800 + kvb * (200 + 600 * cdiv(vec_cnt, use)))
                key = (cost, -br * bc)
                if best is None or key < best[0]:
                    best = (key, br, bc)
                break
    if best is None:
        return FaPlan("HVX")
    br, bc = best[1], best[2]
    kvb = cdiv(kv, bc)
    thr = N_THREADS if kvb >= 3 else 1
    out = macs = qk = 0
    dt = FA_DK // 32
    for q0 in range(0, qo, br):
        rt = cdiv(min(br, qo - q0) * g, 32)
        for b in range(kvb):
            ct = cdiv(min(bc, kv - b * bc), 32)
            qk += rt * ct * dt
            out += rt * ct + rt * dt
            macs += rt * ct * dt + rt * dt * (ct + 1)
        out += rt * dt
        macs += rt * dt
    return FaPlan("HMX", br, bc, kvb, thr, kvb >= 3, fa_vtcm(g, br, bc, can_pipe),
                  out * FA_KV_HEADS, macs * FA_KV_HEADS, qk * FA_KV_HEADS)


# ---- The test-file lines (make_test_cases_from_file of test-backend-ops) ----

def strides(t: str, ne: tuple[int, ...]) -> list[int]:
    """The contiguous byte strides of a tensor."""
    _, blck, size = TYPES[t]
    nb = [size, size * (ne[0] // blck)]
    nb += [nb[1] * ne[1], nb[1] * ne[1] * ne[2]]
    return nb


def src(t: str, ne: tuple[int, ...], nb: list[int] | None = None) -> str:
    """One source of a test-file line: the type id, ne0..3 and nb0..3."""
    nb = nb or strides(t, ne)
    return " ".join(str(x) for x in (TYPES[t][0], *ne, *nb))


def f32_bits(x: float) -> int:
    """The bits of a float as the int32 of an op param."""
    return struct.unpack("<i", struct.pack("<f", x))[0]


def line(c: Case, ops: dict[str, int]) -> str:
    """The test-file line of one case, with the model layout of each source."""
    if c.op == "mm":
        return " ".join(str(x) for x in (ops["GGML_OP_MUL_MAT"], 0, c.m, c.n, 1, 1, 0, 2)) + \
            f" {src(c.t, (c.k, c.m, 1, 1))} {src('f32', (c.k, c.n, 1, 1))} {c.name}"
    if c.op == "fa":
        # q: the permuted view of the contiguous [256, 16, n] output of the rope. k and v: the permuted
        # views of the cache [256 x 4 heads, kv rows]. The mask has n rows (no pad in this tree).
        n, kv, d = c.n, c.kv, FA_DK
        q = src("f32", (d, n, FA_HEADS, 1), [4, 4 * d * FA_HEADS, 4 * d, 4 * d * FA_HEADS * n])
        r1, r2 = row_bytes(c.t, d * FA_KV_HEADS), row_bytes(c.t, d)
        kvs = src(c.t, (d, kv, FA_KV_HEADS, 1), [TYPES[c.t][2], r1, r2, r1 * kv])
        mask = src("f16", (kv, n, 1, 1))
        params = [f32_bits(1.0 / math.sqrt(d)), 0, 0, 10]
        return " ".join(str(x) for x in (ops["GGML_OP_FLASH_ATTN_EXT"], 0, d, FA_HEADS, n, 1, len(params), *params, 4)) + \
            f" {q} {kvs} {kvs} {mask} {c.name}"
    if c.op == "gdn":
        n, d = c.n, GDN_D
        qk = src("f32", (d, GDN_K_HEADS, n, 1))
        # v is the view of the conv output: rows of 8192 floats (q, k, v of one token)
        v = src("f32", (d, GDN_V_HEADS, n, 1), [4, 4 * d, 4 * CONV_DIM, 4 * CONV_DIM * n])
        gb = src("f32", (1, GDN_V_HEADS, n, 1))
        st = src("f32", (d, d, GDN_V_HEADS, 1))
        return " ".join(str(x) for x in (ops["GGML_OP_GATED_DELTA_NET"], 0, d * GDN_V_HEADS, n + d, 1, 1, 1, 1, 6)) + \
            f" {qk} {qk} {v} {gb} {gb} {st} {c.name}"
    n = c.n
    return " ".join(str(x) for x in (ops["GGML_OP_SSM_CONV"], 0, CONV_DIM, n, 1, 1, 0, 2)) + \
        f" {src('f32', (n + CONV_K - 1, CONV_DIM, 1, 1))} {src('f32', (CONV_K, CONV_DIM, 1, 1))} {c.name}"


# ---- The runs ----

@dataclass(frozen=True)
class Run:
    """One phone run: test-backend-ops perf over one test file with one environment, or one DSP
    program of phone/hmx (program is its file name). profile adds GGML_HEXAGON_PROFILE=1: one line with
    the DSP time of each op, thus the time series of the copies of an op (fresh against sustained)."""
    key: str
    text: str
    cases: tuple[Case, ...] = ()
    env: str = "GGML_HEXAGON_OPFUSION=0"
    program: str = ""
    verbose: bool = False
    trace: int = 0         # GGML_HEXAGON_OPTRACE when > 0 (GGML_HEXAGON_PROFILE=3)
    profile: bool = False
    mm_select: int = 3


NS = (1, 2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024)
FA_KV = (512, 1024, 2048, 4096, 8192, 16384)


def q8_block(shapes: tuple[tuple[int, int], ...]) -> tuple[Case, ...]:
    """The Q8_0 cases of some 4B shapes over NS."""
    return tuple(mm("q8_0", k, m, n) for k, m in shapes for n in NS)


def runs() -> list[Run]:
    """The runs of the stage, in the order of the command file."""
    s = [(k, m) for k, m, _, _ in MM_4B]
    main = (2560, 9216)
    down_ = (9216, 2560)
    # The shapes with one weight pass and a low DMA to HMX ratio in the plan give the best rate of the
    # kernel. F16 8192x8192 at 1024 tokens has 5 weight passes: it shows the cost of the chunk choice.
    peak = (
        mm("q8_0", 2560, 16384, 512), mm("f16", 2560, 16384, 512), mm("q8_0", 2048, 16384, 768),
        mm("f16", 2048, 16384, 768), mm("q8_0", 4096, 8192, 384), mm("q8_0", 1024, 16384, 2048),
        mm("q8_0", *main, 2048), mm("f16", *main, 2048), mm("f16", 8192, 8192, 1024),
    )
    trace_mm = tuple(mm("q8_0", k, m, 1024) for k, m in s) + (
        mm("q8_0", *main, 256), mm("f16", *main, 1024), mm("q8_0", 2048, 16384, 768))
    rep = tuple(mm("q8_0", k, m, 1024) for k, m in reversed(s)) + (
        mm("q8_0", 2048, 16384, 768), fa("q8_0", 1024, 4096))
    # k in {2560, 4096, 9216} x m in {2560, 9216}: the three shapes that the other runs do not have
    kxm = tuple(mm("q8_0", k, m, n) for k, m in ((2560, 2560), (4096, 9216), (9216, 9216)) for n in (256, 512, 1024))
    return [
        Run("i8a", "i8read: the f16 and the int8 HMX MAC chains from VTCM (the engine rate, no feed)",
            program="i8read.so"),
        Run("hs", "hmx_sustain: 800 ms of f16 MAC chains from VTCM, then the recovery after pauses of 1 to 160 ms",
            program="hmx_sustain.so"),
        Run("map", "the kernel map of Q8_0 MUL_MAT: GGML_HEXAGON_VERBOSE=1, one case of each kernel choice", (
            *(mm("q8_0", *main, n) for n in (1, 2, 4, 5, 8, 64, 1024)),
            *(mm("q8_0", *down_, n) for n in (1, 4, 5, 1024)),
            *(mm("q8_0", k, m, 1024) for k, m in s[2:]),
        ), verbose=True),
        Run("map3", "the kernel map of F16, Q4_0 and F32 MUL_MAT and of the peak shapes: GGML_HEXAGON_VERBOSE=1", (
            *(mm("f16", *main, n) for n in (1, 4, 5, 1024)),
            *(mm("q4_0", *main, n) for n in (1, 4, 5, 1024)),
            mm("f32", *ALPHA_BETA[:2], 1), mm("f32", *ALPHA_BETA[:2], 1024),
            mm("f16", 2560, 16384, 512), mm("q8_0", 2048, 16384, 768), mm("f16", 8192, 8192, 1024),
        ), verbose=True),
        Run("map2", "the kernel map of FA, GATED_DELTA_NET and SSM_CONV: GGML_HEXAGON_VERBOSE=1", (
            fa("q8_0", 1, 512), fa("f16", 1, 512), fa("q8_0", 1024, 1024), fa("f16", 1024, 1024),
            Case("gdn", "f32", 1024), Case("conv", "f32", 1024),
        ), verbose=True),
        Run("q8a", "Q8_0 MUL_MAT, ffn_gate/up 2560x9216, n 1 to 1024", q8_block((main,)) +
            tuple(mm("q8_0", *main, n) for n in (640, 768)), profile=True),
        Run("q8d", "Q8_0 MUL_MAT, ffn_down 9216x2560, n 1 to 1024", q8_block((down_,)) +
            tuple(mm("q8_0", *down_, n) for n in (640, 768)), profile=True),
        Run("q8b", "Q8_0 MUL_MAT, attn_qkv/q 2560x8192 and attn_gate 2560x4096, n 1 to 1024", q8_block(tuple(s[2:4])),
            profile=True),
        Run("q8c", "Q8_0 MUL_MAT, ssm_out/attn_output 4096x2560 and attn_k/v 2560x1024, n 1 to 1024",
            q8_block(tuple(s[4:6])), profile=True),
        Run("kxm", "Q8_0 MUL_MAT, k 2560/4096/9216 x m 2560/9216, the three shapes that no other run has, n 256 to 1024",
            kxm, profile=True),
        Run("hvx", "Q8_0 MUL_MAT on the HVX path (GGML_HEXAGON_MM_SELECT=2), n 5 to 128",
            tuple(mm("q8_0", k, m, n, "hvx") for k, m in (main, down_) for n in (5, 6, 8, 16, 32, 64, 128)),
            env="GGML_HEXAGON_OPFUSION=0 GGML_HEXAGON_MM_SELECT=2", mm_select=2),
        Run("f16", "F16 MUL_MAT, 2560x9216 and 9216x2560, n 1 to 1024",
            tuple(mm("f16", k, m, n) for k, m in (main, down_) for n in NS if n != 5), profile=True),
        Run("q4", "Q4_0 MUL_MAT at 6 row counts, and the F32 ssm_alpha/ssm_beta shape",
            (*(mm("q4_0", k, m, n) for k, m in (main, down_) for n in (1, 4, 5, 64, 256, 1024)),
             mm("f32", *ALPHA_BETA[:2], 1), mm("f32", *ALPHA_BETA[:2], 1024))),
        Run("peak", "the shapes with one weight pass, and 2048 tokens: the best HMX rate of the MUL_MAT kernel", peak,
            profile=True),
        Run("nx", "the fused form: fusion on, 4 copies of the node become one MUL_MAT_NX of 4",
            (*(mm("q8_0", k, m, n, "nx") for k, m in s for n in (256, 1024)), mm("f16", *main, 1024, "nx")),
            env="GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1", profile=True),
        Run("fa1", "FLASH_ATTN_EXT, 1 query (decode), Q8_0 and F16 K and V, 512 to 16384 KV rows",
            tuple(fa(t, 1, kv) for t in ("q8_0", "f16") for kv in FA_KV)),
        Run("faq8", "FLASH_ATTN_EXT, 1024 queries, Q8_0 K and V, 512 to 4096 KV rows",
            tuple(fa("q8_0", 1024, kv) for kv in FA_KV[:4]), profile=True),
        Run("faq8b", "FLASH_ATTN_EXT, 1024 queries, Q8_0 K and V, 8192 KV rows (the app context)",
            (fa("q8_0", 1024, 8192),), profile=True),
        Run("faf16", "FLASH_ATTN_EXT, 1024 queries, F16 K and V, 1024 and 4096 KV rows (the prefill at d0 and d3072)",
            (fa("f16", 1024, 1024), fa("f16", 1024, 4096)), profile=True),
        Run("gdn", "GATED_DELTA_NET (the chunk kernel from 2 tokens) and SSM_CONV at 1024 tokens",
            (*(Case("gdn", "f32", n) for n in (1, 64, 256, 512, 1024)), Case("conv", "f32", 1024)), profile=True),
        Run("gdnseq", "GATED_DELTA_NET, the sequential kernel (GGML_HEXAGON_GDN_CHUNK=0)",
            tuple(Case("gdn", "f32", n, tag="seq") for n in (64, 1024)),
            env="GGML_HEXAGON_OPFUSION=0 GGML_HEXAGON_GDN_CHUNK=0", profile=True),
        # The trace keeps the first OPTRACE events of each thread in each batch. An op of ffn_down at 1024
        # tokens gives about 1700 events on thread 0, an FA op at 4096 KV rows about 10000: the sizes keep
        # two full ops or more. The shm of the trace is 11 x OPTRACE x 8 bytes x 32 batches.
        Run("trmm", "the phase trace (GGML_HEXAGON_PROFILE=3) of each 4B Q8_0 shape at 1024 tokens", trace_mm,
            trace=8192),
        Run("trfa", "the phase trace of FA at 1024 queries, Q8_0 K/V, 1024 and 4096 KV rows (d0 and d3072)",
            (fa("q8_0", 1024, 1024), fa("q8_0", 1024, 4096)), trace=24576),
        Run("trfa2", "the phase trace of FA with F16 K/V at 4096 KV rows, and of GATED_DELTA_NET at 1024 tokens",
            (fa("f16", 1024, 4096), Case("gdn", "f32", 1024)), trace=24576),
        Run("rep", "the key cases again in the reverse order: the drift over the stage", rep, profile=True),
        Run("i8b", "i8read again: the drift of the engine rate over the stage", program="i8read.so"),
    ]


# ---- The time estimate (for the split into runs of less than 110 s) ----

# A long run of HMX ops slows down to about 1 / 1.9 of the fresh rate (the roofline note of 2026-09-24),
# and the loop of each case is such a run.
SLOW = 1.9


def est_op_us(c: Case, mm_select: int) -> float:
    """A rough time of one op in a long loop. Only for the run split, not a result."""
    if c.op == "mm":
        p = mm_plan(c, mm_select)
        if p.kernel.startswith("HMX"):
            return max(SLOW * c.flops / 11e12, c.stream_bytes * p.mblocks / 45e9) * 1e6 + 30
        if c.t in QUANT:
            return max(c.stream_bytes / 45e9, c.flops / 1.0e12) * 1e6 + 10
        return max(c.stream_bytes / 50e9, c.flops / 0.5e12) * 1e6 + 10
    if c.op == "fa":
        if c.n >= 32:
            return SLOW * c.flops / 3.5e12 * 1e6 + 50
        return c.stream_bytes / 30e9 * 1e6 + 30
    if c.op == "gdn":
        return c.n * (25.0 if c.tag == "seq" else 6.0 * SLOW) + 50
    return c.op_size / 10e9 * 1e6 + 30


def est_case_s(c: Case, mm_select: int) -> float:
    """The seconds of one case: the init, the warm run and the timed loop of at least 1 s.
    n_runs = min(8191, 32 GiB / op_size) + 1 copies of the node go into one graph."""
    op = est_op_us(c, mm_select) + 4
    n_runs = min(8191, (32 << 30) // c.op_size) + 1
    graph = n_runs * op / 1e6
    timed = graph * math.ceil(1.0 / graph) if graph < 1.0 else graph
    return 0.4 + c.op_size * 6e-9 + op / 1e6 + timed


def est_run_s(r: Run) -> float:
    """The seconds of one run without the gate."""
    if r.program:
        return 8.0 if r.program == "hmx_sustain.so" else 6.0
    return 2.0 + sum(est_case_s(c, r.mm_select) for c in r.cases)


def limit_s(r: Run) -> int:
    """The timeout of one run: the estimate with a margin, at most 110 s."""
    return min(110, int(1.5 * est_run_s(r) + 15))


# ---- The command file ----

HEADER = """\
# Phone stage "sweep": the HMX peak of HTP0 and the rate of the prefill op shapes of the 4B Q8_0.
#
# What it measures, with test-backend-ops perf --test-file (one op at a time, fusion off) and two DSP programs:
#   i8a, i8b  i8read: the f16 and int8 HMX MAC chains from VTCM, no feed (the engine rate), start and end
#   hs        hmx_sustain: 800 ms of f16 MAC chains from VTCM (does the HMX alone slow down in a long run?),
#             then the rate after pauses of 1 to 160 ms (the recovery)
#   map, map2, map3  GGML_HEXAGON_VERBOSE=1 over one case of each kernel choice: the kernel of each case
#   q8a, q8d, q8b, q8c  Q8_0 MUL_MAT of the six 4B weight shapes, n = 1, 2, 4, 5, 8, ..., 1024 tokens
#   kxm       Q8_0 MUL_MAT 2560x2560, 4096x9216 and 9216x9216 at 256, 512, 1024 tokens (k against m)
#   hvx       the FFN Q8_0 shapes on the HVX path (GGML_HEXAGON_MM_SELECT=2), n 5 to 128: the HMX crossover
#   f16, q4   F16 and Q4_0 weights at the two FFN shapes, and the F32 alpha/beta shape
#   peak      the shapes with one weight pass in the host plan, and 2048 tokens: the best rate of the kernel
#   nx        fusion on: 4 copies of the node fuse to MUL_MAT_NX of 4 (one activation conversion)
#   fa1       FLASH_ATTN_EXT, head dim 256, 16/4 heads, 1 query, Q8_0 and F16 K/V, 512 to 16384 KV rows
#   faq8, faq8b, faf16  FLASH_ATTN_EXT, 1024 queries: Q8_0 K/V at 512 to 8192 KV rows, F16 K/V at 1024 and 4096
#   gdn, gdnseq       GATED_DELTA_NET (chunk and sequential) and SSM_CONV at the 4B prefill shapes
#   trmm, trfa, trfa2 GGML_HEXAGON_PROFILE=3 traces: the HMX busy time and the phase that holds the HMX
#   rep       the key cases again in the reverse order (drift)
# The runs with many HMX ops set GGML_HEXAGON_PROFILE=1: one line with the DSP time of each op copy. The stderr of
# these runs and of the trace runs goes through gzip when the phone has it. A long HMX run slows down and a pause
# of about 40 ms gives the rate back, thus the us/run of test-backend-ops (a loop of 1 s or more) is the sustained
# rate, and the first copies of each case (after the host init of the case, the HMX idles) give the fresh rate.
# The libraries are the set of build/bench-kv (HEAD e8a3a07, the app build), test-backend-ops comes from the
# same tree and flags (build/perf-tbo). Each run: the thermal line, the gate (bin/gate.sh, 2 GB MemAvailable),
# the tool under timeout -s KILL (110 s or less), the exit code and the conditions after the run.
# Each run line names the 2B model file in an ls (no run loads it): thus the runner waits for the unlocked phone,
# wakes the screen and prints the caps for it, and it requires MIN_2B_MB (6000 MB) of MemAvailable.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about {tool_min:.0f} minutes of tool time
# ({n_runs} runs) plus about 15 s of gate and checks for each run, about {total_min:.0f} minutes, plus the waits
# for thermal status 0. Then, on the box: build/sweep/stage.py table
"""


def setup_lines() -> list[str]:
    """The lines that copy the stage to the phone and check its files."""
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/hmx {PHONE}/tests {PHONE}/out'",
        f"{ADB} push {LAPTOP_STAGE}/phone/bin/test-backend-ops {LAPTOP_STAGE}/phone/bin/gate.sh {PHONE}/bin/",
        f"{ADB} push " + " ".join(f"{LAPTOP_STAGE}/phone/lib/{x}" for x in LIBS) + f" {PHONE}/lib/",
        f"{ADB} push " + " ".join(f"{LAPTOP_STAGE}/phone/hmx/{x}" for x in HMX_FILES) + f" {PHONE}/hmx/",
        f"{ADB} push " + " ".join(f"{LAPTOP_STAGE}/phone/tests/{r.key}.txt" for r in runs() if r.cases) + f" {PHONE}/tests/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/* {PHONE}/hmx/*'",
        f"{ADB} shell 'echo gzip: $(command -v gzip) uniq: $(command -v uniq) timeout: $(command -v timeout)'",
    ]


def run_lines(r: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{PHONE}/out/{r.key}"
    gate = (f"{MARKER}; sh {PHONE}/bin/gate.sh {GATE_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && ")
    tail = f"echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt"
    # Each tool part is one group: a gate that fails skips all of it, and $? after it is the exit code of
    # the tool (or of the gate).
    if r.program:
        tool = (f"{{ cd {PHONE}/hmx && timeout -s KILL {limit_s(r)} env ADSP_LIBRARY_PATH={PHONE}/hmx "
                f"./run_main_on_hexagon 3 {r.program} --out {stem}.txt > {stem}.out 2> {stem}.log; }}; ")
        tail = (f"echo \"rc=$?\" >> {stem}-gate.txt; logcat -d -s adsprpc -t 4000 | grep -E \"i8read:|sustain:\" > "
                f"{stem}-logcat.txt; {AFTER} >> {stem}-gate.txt; cat {stem}-gate.txt")
    else:
        env = " ".join(x for x in (LIB_ENV, r.env,
                                   "GGML_HEXAGON_VERBOSE=1" if r.verbose else "",
                                   f"GGML_HEXAGON_PROFILE=3 GGML_HEXAGON_OPTRACE={r.trace}" if r.trace else "",
                                   "GGML_HEXAGON_PROFILE=1" if r.profile else "") if x)
        cmd = (f"timeout -s KILL {limit_s(r)} env {env} {PHONE}/bin/test-backend-ops perf -b HTP0 "
               f"--test-file {PHONE}/tests/{r.key}.txt")
        if r.verbose:
            # The verbose map prints one line for each packed copy of a node: uniq keeps one of each.
            tool = f"{{ set -o pipefail; {cmd} 2>&1 | uniq > {stem}.out; }}; "
        elif r.profile or r.trace:
            # One profile line for each op copy: gzip (when the phone has it) keeps the file small. The
            # parser reads a gzip file and a plain file alike.
            tool = (f"{{ Z=cat; command -v gzip > /dev/null && Z=\"gzip -1\"; set -o pipefail; "
                    f"{cmd} 2>&1 > {stem}.out | $Z > {stem}.log.z; }}; ")
        else:
            tool = f"{{ {cmd} > {stem}.out 2> {stem}.log; }}; "
    what = f"{len(r.cases)} cases, " if r.cases else ""
    return ["#", f"# SWEEP {r.key}: {r.text} ({what}estimate {est_run_s(r):.0f} s, limit {limit_s(r)} s)",
            THERMAL, f"{ADB} shell '{gate}{tool}{tail}'", PGREP]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory."""
    return [
        "#", "# ---- The outputs ----", "#", THERMAL,
        f"{ADB} shell 'pgrep -x test-backend-op; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_files() -> int:
    """Write phone/ and phone-commands.txt. A case name must be unique in its run."""
    ops = ggml_ops()
    phone = HERE / "phone"
    for sub in ("bin", "lib", "tests"):
        (phone / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "build/perf-tbo/phone/bin/test-backend-ops", phone / "bin")
    shutil.copy2(REPO / "build/bench-kv/phone/bin/gate.sh", phone / "bin")
    for lib in LIBS:
        shutil.copy2(REPO / "build/bench-kv/phone/lib" / lib, phone / "lib")
    hmx = phone / "hmx"
    for extra in ("i8hello.so", "i8probe.so"):
        (hmx / extra).unlink(missing_ok=True)
    for farf in ("i8read.farf", "hmx_sustain.farf", "run_main_on_hexagon.farf"):
        (hmx / farf).write_text("0x1f\n")
    for old in (phone / "tests").glob("*.txt"):
        old.unlink()
    all_runs = runs()
    for r in all_runs:
        names = [c.name for c in r.cases]
        if len(set(names)) != len(names):
            raise ValueError(f"the run {r.key} has a case name two times: the parser keys a result by run and name")
        if r.cases:
            (phone / "tests" / f"{r.key}.txt").write_text("".join(line(c, ops) + "\n" for c in r.cases))
    files = sorted(p for p in phone.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(phone)}" for p in files]
    (phone / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    missing = [x for x in HMX_FILES if not (hmx / x).exists()]
    if missing:
        raise FileNotFoundError(f"{hmx} has no {missing}: run tools/stages/sweep/build.sh")
    tool = sum(est_run_s(r) for r in all_runs)
    head = HEADER.format(tool_min=tool / 60, n_runs=len(all_runs), total_min=(tool + 15 * len(all_runs)) / 60)
    lines = head.rstrip("\n").split("\n") + setup_lines()
    for r in all_runs:
        lines += run_lines(r)
    lines += output_lines()
    (HERE / "phone-commands.txt").write_text("\n".join(lines) + "\n")
    for r in all_runs:
        print(f"  {r.key:8s} {len(r.cases):3d} cases  estimate {est_run_s(r):5.1f} s  limit {limit_s(r):3d} s")
    print(f"phone-commands.txt: {len(lines)} lines, {len(all_runs)} runs, tool time about {tool / 60:.1f} min")
    return 0


def print_plan() -> int:
    """Print the host kernel choice of each case of the stage."""
    for r in runs():
        if not r.cases:
            continue
        print(f"{r.key}: {r.text}")
        for c in r.cases:
            if c.op == "mm":
                p = mm_plan(c, r.mm_select)
                extra = f" out-tiles {p.out_tiles} tile-MACs {p.tile_macs}" if p.tile_macs else ""
                print(f"  {c.name:34s} {p.text()}{extra}")
            elif c.op == "fa":
                p = fa_plan(c)
                print(f"  {c.name:34s} {p.text()} tile-MACs {p.tile_macs} (QK {p.qk_macs})")
            else:
                print(f"  {c.name:34s} {'chunk HMX' if c.op == 'gdn' and c.n > 1 and c.tag != 'seq' else 'HVX'}")
    return 0


# ---- The outputs ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
BEFORE_RE = re.compile(r"before: nsp=(\d*) screen=(\S*) keyguard=(\S*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*) nsp=(\d*) "
                      r"screen=(\S*) keyguard=(\S*)")
# A result is "  OP(name=X,...): " and then "not supported" or "N runs - U us/run". In a run that merges
# stderr into stdout, the unbuffered stderr lines of the next case can come between the two parts, thus the
# parser pairs each name with the next timing in the stream.
NAME_RE = re.compile(r"^\s+[A-Z_]+\(name=([A-Za-z0-9_x]+),[^\n]*?\):\s*(not supported)?", re.M)
TIMING_RE = re.compile(r"(\d+) runs -\s+([\d.]+) us/run")
I8_RE = re.compile(r"i8read: (f16swap|f16|i8 \w+) k=(\d+) pass=(\d+)")
HWINFO_RE = re.compile(r"hwinfo: threads (\d+), hvx (\d+), hmx (\d+), vtcm (\d+) MB")
EXEC_RE = re.compile(r"execute-op ([A-Z_0-9+]+)\|[^|]*\|([^|]*)\|([^|]*)\|[^|]*\|[^|]*\|([^|]*)\|")
N_I8_COLS = 8


@dataclass
class RunOut:
    """The parsed files of one run."""
    run: Run
    ok: bool
    flags: list[str]
    caps: str
    us: dict[str, float] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    log: str = ""
    out: str = ""
    i8: dict[str, dict[int, int]] = field(default_factory=dict)
    series: dict[str, list[float]] = field(default_factory=dict)   # case -> DSP us of each copy, in order
    sustain: list[tuple[int, int, int, int]] = field(default_factory=list)      # (t_us, passes, us, pcycles)
    recovery: list[tuple[int, int, int, int, int, int, int]] = field(default_factory=list)


def read_run(root: Path, r: Run) -> RunOut:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    gate_p = root / f"{r.key}-gate.txt"
    gate = gate_p.read_text(errors="replace") if gate_p.exists() else ""
    before, bnsp, after = GATE_RE.search(gate), BEFORE_RE.search(gate), AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    flags: list[str] = []
    ok = "gate: OK" in gate and rc is not None and rc.group(1) == "0"
    if not gate:
        flags.append("no gate file")
    elif "gate: OK" not in gate:
        flags.append("the gate stopped the run")
    elif not ok:
        flags.append(f"exit code {rc.group(1) if rc else '?'}")
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        flags.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    if after and after.group(1) not in ("", "0"):
        flags.append(f"thermal {after.group(1)} after the run")
    for label, m, si, ki in (("before", bnsp, 2, 3), ("after", after, 7, 8)):
        if m and (m.group(si) != "Awake" or m.group(ki) != "false"):
            flags.append(f"screen {m.group(si)} keyguard {m.group(ki)} {label} the run")
    res = RunOut(r, ok, flags, caps)
    out_p = root / f"{r.key}.out"
    res.out = out_p.read_text(errors="replace") if out_p.exists() else ""
    res.log = read_log(root, r.key)
    if r.program:
        read_program(root, r, res)
        return res
    names = list(NAME_RE.finditer(res.out))
    for i, m in enumerate(names):
        if m.group(2):
            res.unsupported.append(m.group(1))
            continue
        end = names[i + 1].start() if i + 1 < len(names) else len(res.out)
        t = TIMING_RE.search(res.out, m.end(), end)
        if t:
            res.us[m.group(1)] = float(t.group(2))
    missing = [c.name for c in r.cases if c.name not in res.us and c.name not in res.unsupported]
    if ok and missing:
        flags.append(f"{len(missing)} cases without a result, the first {missing[0]}")
    if res.unsupported:
        flags.append(f"not supported: {', '.join(res.unsupported)}")
    if r.profile or r.trace:
        res.series = op_series(res.log, r.cases)
        if ok and not res.series:
            flags.append("no profile-op line of a case")
    return res


def read_log(root: Path, key: str) -> str:
    """The stderr of one run: <key>.log, or <key>.log.z (gzip, or plain when the phone had no gzip)."""
    plain, packed = root / f"{key}.log", root / f"{key}.log.z"
    if plain.exists():
        return plain.read_text(errors="replace")
    if packed.exists():
        data = packed.read_bytes()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return data.decode(errors="replace")
    return ""


SUSTAIN_RE = re.compile(r"sustain: series i=\d+ t_us=(\d+) passes=(\d+) us=(\d+) pcycles=(\d+)")
RECOVERY_RE = re.compile(r"sustain: recovery pause_ms=(\d+) window=(\d+) passes=(\d+) us=(\d+) pcycles=(\d+) "
                         r"heat_passes=(\d+) heat_us=(\d+)")


def read_program(root: Path, r: Run, res: RunOut) -> None:
    """Read the result file of a DSP program (i8read or hmx_sustain), else its logcat lines."""
    tag = "i8read:" if r.program == "i8read.so" else "sustain:"
    txt = root / f"{r.key}.txt"
    text = txt.read_text(errors="replace") if txt.exists() else ""
    if tag not in text:
        lc = root / f"{r.key}-logcat.txt"
        text = lc.read_text(errors="replace") if lc.exists() else ""
        first = "i8read: check p4 k=64" if tag == "i8read:" else "sustain: config"
        start = text.rfind(first)
        text = text[start:] if start >= 0 else text
        if text:
            res.flags.append(f"{r.program}: the result file is empty, the logcat lines are used")
    for m in I8_RE.finditer(text):
        res.i8.setdefault(m.group(1), {})[int(m.group(2))] = int(m.group(3))
    res.sustain = [tuple(int(x) for x in m.groups()) for m in SUSTAIN_RE.finditer(text)]
    res.recovery = [tuple(int(x) for x in m.groups()) for m in RECOVERY_RE.finditer(text)]
    if res.ok and not (res.i8 or res.sustain):
        res.flags.append(f"{r.program}: no result line")


PROF_RE = re.compile(r"profile-op ([A-Z_0-9+]+)\|[^|]*\|([^|]*)\|([^|]*)\|[^|]*\|[^|]*\|usec \d+ cycles (\d+) "
                     r"start \d+ mhz ([\d.]+)")


def op_series(log: str, cases: tuple[Case, ...]) -> dict[str, list[float]]:
    """The DSP time of each copy of each case, in log order, from the profile-op lines. A MUL_MAT_NX
    line holds several copies: its time goes to each copy in equal parts. O(lines x cases)."""
    out: dict[str, list[float]] = defaultdict(list)
    memo: dict[tuple, tuple[Case | None, int]] = {}
    for m in PROF_RE.finditer(log):
        key = (m.group(1), m.group(2), m.group(3))
        if key not in memo:
            c = match_case(list(cases), m.group(1), m.group(2), m.group(3))
            memo[key] = (c, max(1, m.group(2).count(" x ")) if m.group(1) == "MUL_MAT_NX" else 1)
        c, copies = memo[key]
        if c is None:
            continue
        mhz = float(m.group(5)) or CLOCK_MHZ
        us = int(m.group(4)) / mhz / copies
        out[c.name].extend([us] * copies)
    return dict(out)


def usable(o: RunOut | None, include_all: bool) -> bool:
    """True when the run goes into the tables: its gate passed and its exit code is 0. A flag (changed
    caps, heat, screen) does not drop the run: the tables mark it with "*" and section 0 names the flag.
    include_all is kept for the command line and has no effect."""
    return o is not None and o.ok


def mark(outs: dict[str, RunOut], key: str) -> str:
    """The run key, with "*" when the run has a flag."""
    o = outs.get(key)
    return f"{key}*" if o is not None and o.flags else key


@dataclass
class Engine:
    """The f16 HMX engine rate from i8read: pcycles of one output tile = a + b x k-tiles."""
    a: float
    b: float
    clock: float
    points: dict[int, float]

    def tflops(self, kt: int) -> float:
        """The f16 rate of one output tile with kt k-tiles, in TFLOPS."""
        return TILE_MAC_FLOP * kt * self.clock * 1e6 / (self.a + self.b * kt) / 1e12

    def floor_us(self, out_tiles: int, tile_macs: int) -> float:
        """The HMX time of a job set at the engine rate, in us."""
        return (out_tiles * self.a + tile_macs * self.b) / self.clock


# i8read times a pass of 8 output tiles with the core clock and one syncht at the end. Below about 48
# k-tiles a pass gives 19 to 120 pcycles for each output tile at every k (measured 2026-09-24), which is the
# issue time and not the HMX time: the HMX queue is not full, thus the core does not wait. From 48 k-tiles the
# issue waits for the HMX and the time grows with k.
I8_KMIN = 48


def fit(points: dict[int, float], kmin: int = I8_KMIN) -> tuple[float, float]:
    """The least squares line c = a + b k over the points with k >= kmin."""
    xs = [k for k in points if k >= kmin]
    ys = [points[k] for k in xs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def clock_mhz(outs: dict[str, RunOut]) -> tuple[float, str]:
    """The DSP clock: the median mhz of the profile lines of the trace runs, else CLOCK_MHZ (assumed)."""
    vals = [float(m.group(1)) for k in ("trmm", "trfa", "trfa2") if k in outs
            for m in re.finditer(r"profile-op [A-Z_0-9+]+\|.* mhz ([\d.]+)", outs[k].log)]
    vals = [v for v in vals if v > 100]
    if vals:
        return statistics.median(vals), f"measured, median of {len(vals)} ops"
    return CLOCK_MHZ, "assumed"


def i8_tables(outs: dict[str, RunOut], clock: float, include_all: bool) -> tuple[list[str], Engine | None]:
    """The engine table: per output tile pcycles against k for each i8read run, and the fit."""
    lines = ["== 1. The HMX engine (i8read, measured): pcycles of one output tile against the k-tiles of its chain =="]
    engine = None
    for key in ("i8a", "i8b"):
        o = outs.get(key)
        if not usable(o, include_all) or "f16" not in o.i8:
            lines.append(f"  {key}: no usable result" + (f" ({', '.join(o.flags)})" if o and o.flags else ""))
            continue
        f16 = {k: v / N_I8_COLS for k, v in o.i8["f16"].items()}
        a, b = fit(f16)
        lines.append(f"  {key}: f16 per output tile = {a:.1f} + {b:.3f} x k-tiles pcycles (fit over k >= {I8_KMIN}); "
                     f"{TILE_MAC_FLOP * clock * 1e6 / b / 1e12:.2f} TFLOPS at the marginal rate, clock {clock:.0f} MHz")
        cols = sorted(f16)
        lines.append("    k-tiles " + " ".join(f"{k:>7d}" for k in cols))
        for variant in ("f16", "f16swap", "i8 none", "i8 p4"):
            if variant in o.i8:
                row = {k: v / N_I8_COLS for k, v in o.i8[variant].items()}
                lines.append(f"    {variant:7s} " + " ".join(f"{row.get(k, float('nan')):7.1f}" for k in cols))
        lines.append("    TFLOPS  " + " ".join(f"{TILE_MAC_FLOP * k * clock * 1e6 / f16[k] / 1e12:7.2f}" if k >= I8_KMIN
                                         else f"{'-':>7s}" for k in cols))
        lines.append(f"    pcyc/MAC" + " ".join(f"{f16[k] / k:7.2f}" if k >= I8_KMIN else f"{'-':>7s}" for k in cols))
        if engine is None:
            engine = Engine(a, b, clock, f16)
    if engine is not None:
        lines.append(f"  the f16 rate of the fit at the 4B reductions (computed; the fit holds from 48 k-tiles): "
                     + ", ".join(f"K {k}: {engine.tflops(k // 32):.2f} TFLOPS" for k in (2560, 4096, 9216)))
        lines.append("  The short i8read pass reads low: 7.06 pcycles per tile-MAC at 80 k-tiles against 8.07 in the 800 ms "
                     "run of hmx_sustain and 8.05 to 8.08 in the kernel (section 6). 7.06 x 8 / 7 = 8.07: the syncht at "
                     "the end of a pass of 8 output tiles appears not to wait for the last tile (assumed).")
    return lines, engine


def case_us(outs: dict[str, RunOut], key: str, name: str, include_all: bool) -> float | None:
    """The us/run of one case of one run, or None."""
    o = outs.get(key)
    return o.us.get(name) if usable(o, include_all) else None


def find_us(outs: dict[str, RunOut], name: str, include_all: bool, keys: tuple[str, ...]) -> tuple[float | None, str]:
    """The us/run of a case from the first run of keys that has it."""
    for k in keys:
        us = case_us(outs, k, name, include_all)
        if us is not None:
            return us, k
    return None, ""


def kernel_map(outs: dict[str, RunOut]) -> dict[tuple, str]:
    """The kernel path of each (op, dims) from the verbose map run and the profile lines."""
    seen: dict[tuple, str] = {}
    for key in ("map", "map2", "map3", "trmm", "trfa", "trfa2"):
        o = outs.get(key)
        if o is None:
            continue
        for m in EXEC_RE.finditer(o.out + o.log):
            seen[(m.group(1), m.group(2), m.group(3))] = m.group(4).strip()
        for m in re.finditer(r"profile-op ([A-Z_0-9+]+)\|[^|]*\|([^|]*)\|([^|]*)\|[^|]*\|([^|]*)\|usec", o.log):
            seen.setdefault((m.group(1), m.group(2), m.group(3)), m.group(4).strip())
    return seen


def dims_prefix(c: Case) -> str:
    """The start of the dims field of the op lines of a case. A tensor with ne2 = ne3 = 1 prints as
    "ne0:ne1", other tensors as "ne0:ne1:ne2:ne3" (htp-opnode.h format_tensor_dims)."""
    if c.op == "mm":
        return f"{c.k}:{c.m} x {c.k}:{c.n} -> "
    if c.op == "fa":
        return f"{FA_DK}:{c.n}:{FA_HEADS}:1 x {FA_DK}:{c.kv}:{FA_KV_HEADS}:1 x "
    if c.op == "gdn":
        return f"{GDN_D}:{GDN_K_HEADS}:{c.n}:1 x "
    return f"{c.n + CONV_K - 1}:{CONV_DIM} x "


OP_NAMES = {"mm": "MUL_MAT", "fa": "FLASH_ATTN_EXT", "gdn": "GATED_DELTA_NET", "conv": "SSM_CONV"}


def mm_path(kmap: dict[tuple, str], c: Case) -> str:
    """The logged kernel path of a case, or "-"."""
    for (op, dims, types), path in kmap.items():
        if op != "MUL_MAT_NX" and match_case([c], op, dims, types):
            return path
    return "-"


@dataclass
class Series:
    """The DSP time series of the copies of one case. warm: the copy of the warm run (a graph of its own
    after the host init). fresh: the median of timed copies 1 to 3. sustained: the median of the last 30 %
    of the copies. onset_ms: the DSP time from the first timed copy to the first copy where the median of 5
    copies is 15 % above fresh, or None."""
    n: int
    warm: float
    fresh: float
    sustained: float
    onset_ms: float | None

    @property
    def slow(self) -> float:
        """sustained / fresh."""
        return self.sustained / self.fresh if self.fresh else float("nan")


def series_of(ops: list[float]) -> Series | None:
    """The statistics of one time series, or None below 8 copies. O(n)."""
    if len(ops) < 8:
        return None
    timed = ops[1:]
    fresh = statistics.median(timed[:3])
    tail = timed[int(0.7 * len(timed)):]
    onset, t = None, 0.0
    for i in range(len(timed)):
        t += timed[i]
        if i >= 4 and statistics.median(timed[i - 4:i + 1]) > 1.15 * fresh:
            onset = t / 1e3
            break
    return Series(len(ops), ops[0], fresh, statistics.median(tail), onset)


def find_series(outs: dict[str, RunOut], name: str, include_all: bool, keys: tuple[str, ...]) -> Series | None:
    """The series of a case from the first run of keys that has it."""
    for k in keys:
        o = outs.get(k)
        if usable(o, include_all) and name in o.series:
            st = series_of(o.series[name])
            if st:
                return st
    return None


PROFILE_KEYS = ("q8a", "q8d", "q8b", "q8c", "kxm", "f16", "peak", "rep", "trmm")


def cell(x: float | None, fmt: str, width: int) -> str:
    """A number, or "-" when it is None."""
    return f"{x:{width}{fmt}}" if x is not None else f"{'-':>{width}s}"


def mm_tables(outs: dict[str, RunOut], engine: Engine | None, kmap: dict[tuple, str], include_all: bool) -> list[str]:
    """The MUL_MAT tables: per shape and type, the rate against n (the host us/run of the loop, and the DSP
    time of the fresh and of the sustained copies), the kernel, the HVX path and the NX form."""
    lines = ["", "== 2. MUL_MAT (measured: host us/run, DSP fresh and sustained us; computed from them: TFLOPS, weight "
             "GB/s, HMX%; plan: the host choice, computed; path: the logged kernel, measured) =="]
    keys = ("q8a", "q8d", "q8b", "q8c", "kxm", "f16", "q4", "peak", "map", "map3", "rep")
    shapes = [("q8_0", k, m) for k, m, _, _ in MM_4B] + \
             [("q8_0", 2560, 2560), ("q8_0", 4096, 9216), ("q8_0", 9216, 9216),
              ("f16", 2560, 9216), ("f16", 9216, 2560), ("q4_0", 2560, 9216), ("q4_0", 9216, 2560),
              ("f32", 2560, 32), ("q8_0", 2560, 16384), ("f16", 2560, 16384), ("q8_0", 2048, 16384),
              ("f16", 2048, 16384), ("q8_0", 4096, 8192), ("q8_0", 1024, 16384), ("f16", 8192, 8192)]
    best: tuple[float, str] | None = None
    for t, k, m in shapes:
        rows = []
        for n in sorted({*NS, 384, 640, 768, 2048}):
            c = mm(t, k, m, n)
            us, src_key = find_us(outs, c.name, include_all, keys)
            if us is None:
                continue
            p = mm_plan(c)
            sr = find_series(outs, c.name, include_all, PROFILE_KEYS)
            hvx = case_us(outs, "hvx", mm(t, k, m, n, "hvx").name, include_all)
            nx = case_us(outs, "nx", mm(t, k, m, n, "nx").name, include_all)
            floor = engine.floor_us(p.out_tiles, p.tile_macs) if engine and p.tile_macs else None
            fresh = sr.fresh if sr else None
            ref = fresh or us
            tf_f = c.flops / fresh / 1e6 if fresh else None
            if tf_f and (best is None or tf_f > best[0]):
                best = (tf_f, c.name)
            rows.append(f"    {n:5d} {us:9.1f} {cell(fresh, '.1f', 9)} {cell(sr.sustained if sr else None, '.1f', 9)} "
                        f"{cell(tf_f, '.2f', 6)} {c.flops / us / 1e6:6.2f} {c.stream_bytes / ref / 1e3:6.1f} "
                        f"{cell(100 * floor / ref if floor else None, '.1f', 5)} "
                        f"{cell(sr.onset_ms if sr else None, '.0f', 5)} {cell(hvx, '.1f', 9)} {cell(nx, '.1f', 9)} "
                        f"{mark(outs, src_key):6s} {mm_path(kmap, c):10s} {p.text()}")
        if rows:
            lines.append(f"  {t} k={k} m={m}")
            lines.append(f"    {'n':>5s} {'host us':>9s} {'fresh us':>9s} {'sust us':>9s} {'TF fr':>6s} {'TF lp':>6s} "
                         f"{'wGB/s':>6s} {'HMX%':>5s} {'onset':>5s} {'HVX us':>9s} {'NX4 us':>9s} {'run':6s} "
                         f"{'path':10s} plan")
            lines += rows
    lines.append("  host us: us/run of the loop of test-backend-ops (1 s or more of copies: the sustained rate, with the "
                 "host share). fresh us, sust us: DSP time of the timed copies 1-3 and of the last 30 %. TF fr: TFLOPS of "
                 "fresh. TF lp: TFLOPS of host us. wGB/s and HMX% use fresh (host us without a series). HMX%: the HMX "
                 "time of the plan at the engine rate of section 1. onset: DSP ms from the first timed copy until 5 "
                 "copies are 15 % slower than fresh. HVX us: GGML_HEXAGON_MM_SELECT=2. NX4 us: us/run of the fused "
                 "form (4 copies, one activation conversion).")
    if best:
        lines.append(f"  the best fresh MUL_MAT rate of the sweep (measured): {best[0]:.2f} TFLOPS ({best[1]})")
    lines.append("  the crossover (measured, host us): the smallest n where the HMX path is faster than the HVX path")
    for k, m in ((2560, 9216), (9216, 2560)):
        found = None
        for n in (5, 6, 8, 16, 32, 64, 128):
            hvx = case_us(outs, "hvx", mm("q8_0", k, m, n, "hvx").name, include_all)
            hmx, _ = find_us(outs, mm("q8_0", k, m, n).name, include_all, keys)
            if hvx and hmx and hmx < hvx and found is None:
                found = n
        lines.append(f"    q8_0 {k}x{m}: {found if found else 'none in the sweep'}")
    return lines


def prefill_tables(outs: dict[str, RunOut], engine: Engine | None, include_all: bool) -> list[str]:
    """The 1024-token ubatch of the 4B from the isolated cases, against the bench-kv op split."""
    keys = ("q8a", "q8d", "q8b", "q8c", "rep", "map")
    lines = ["", "== 3. One 1024-token prefill ubatch of the 4B from the isolated MUL_MATs (computed from measured cases) =="]
    tot = {"host": 0.0, "fresh": 0.0, "sust": 0.0, "nx": 0.0, "floor": 0.0}
    flops = 0.0
    ok = True
    for k, m, cnt, what in (*MM_4B, ALPHA_BETA):
        t = "f32" if m == 32 else "q8_0"
        c = mm(t, k, m, 1024)
        us, _ = find_us(outs, c.name, include_all, ("q4",) + keys if t == "f32" else keys)
        sr = find_series(outs, c.name, include_all, PROFILE_KEYS + ("q4",))
        nx = case_us(outs, "nx", mm(t, k, m, 1024, "nx").name, include_all)
        p = mm_plan(c)
        fl = engine.floor_us(p.out_tiles, p.tile_macs) if engine and p.tile_macs else 0.0
        if us is None:
            ok = False
            lines.append(f"  {what:32s} no result")
            continue
        fresh = sr.fresh if sr else us
        sust = sr.sustained if sr else us
        for key, v in (("host", us), ("fresh", fresh), ("sust", sust), ("nx", nx or us), ("floor", fl)):
            tot[key] += cnt * v
        flops += cnt * c.flops
        lines.append(f"  {what:32s} {cnt:3d} x fresh {fresh:8.1f} us = {cnt * fresh / 1e3:6.1f} ms "
                     f"({c.flops / fresh / 1e6:5.2f} TF), sustained {cnt * sust / 1e3:6.1f} ms, loop {cnt * us / 1e3:6.1f} ms, "
                     f"NX4 {cnt * (nx or us) / 1e3:6.1f} ms, HMX floor {cnt * fl / 1e3:5.1f} ms")
    if ok:
        lines.append(f"  sum: fresh {tot['fresh'] / 1e3:.1f} ms ({flops / tot['fresh'] / 1e6:.2f} TFLOPS), sustained "
                     f"{tot['sust'] / 1e3:.1f} ms, loop {tot['host'] / 1e3:.1f} ms, NX4 {tot['nx'] / 1e3:.1f} ms, "
                     f"HMX floor {tot['floor'] / 1e3:.1f} ms; bench-kv (measured in the model): "
                     f"{BENCH_KV['weight MUL_MAT d0']:.0f} ms at d0, {BENCH_KV['weight MUL_MAT d3072']:.0f} ms at d3072")
    for label, kv in (("d0", 1024), ("d3072", 4096)):
        c = fa("q8_0", 1024, kv)
        us, _ = find_us(outs, c.name, include_all, ("faq8", "rep", "map2"))
        sr = find_series(outs, c.name, include_all, ("faq8", "rep", "trfa"))
        if us:
            fr = f"fresh {FA_LAYERS * sr.fresh / 1e3:.1f} ms, sustained {FA_LAYERS * sr.sustained / 1e3:.1f} ms, " if sr else ""
            lines.append(f"  FA {label} (n_kv {kv}): {FA_LAYERS} x: {fr}loop {FA_LAYERS * us / 1e3:.1f} ms; "
                         f"bench-kv FA {label}: {BENCH_KV['FA ' + label]:.0f} ms")
    return lines


def fa_tables(outs: dict[str, RunOut], engine: Engine | None, include_all: bool) -> list[str]:
    """The FA table: us, the kernel TFLOPS (every KV block), the K and V GB/s, the HMX share."""
    lines = ["", "== 4. FLASH_ATTN_EXT, head dim 256, 16 query heads, 4 KV heads (measured us; TFLOPS over every KV "
             "block, the kernel skips none; HMX% from the plan at the deep-chain engine rate, computed) =="]
    lines.append(f"  {'type':5s} {'n_q':>5s} {'n_kv':>6s} {'host us':>10s} {'fresh us':>10s} {'sust us':>10s} "
                 f"{'TF fr':>6s} {'KV GB/s':>8s} {'HMX%':>6s} plan")
    for t in ("q8_0", "f16"):
        for n in (1, 1024):
            for kv in FA_KV:
                c = fa(t, n, kv)
                us, _ = find_us(outs, c.name, include_all, ("fa1", "faq8", "faq8b", "faf16", "rep", "map2"))
                if us is None:
                    continue
                sr = find_series(outs, c.name, include_all, ("faq8", "faq8b", "faf16", "rep", "trfa", "trfa2"))
                ref = sr.fresh if sr else us
                p = fa_plan(c)
                fl = engine.floor_us(p.out_tiles, p.tile_macs) if engine else None
                lines.append(f"  {t:5s} {n:5d} {kv:6d} {us:10.1f} {cell(sr.fresh if sr else None, '.1f', 10)} "
                             f"{cell(sr.sustained if sr else None, '.1f', 10)} {c.flops / ref / 1e6:6.2f} "
                             f"{c.stream_bytes / ref / 1e3:8.1f} {cell(100 * fl / ref if fl else None, '.1f', 6)} {p.text()}")
    return lines


def sustain_tables(outs: dict[str, RunOut], include_all: bool) -> list[str]:
    """The pure HMX run of hmx_sustain: the rate against the time of the run, and after each pause."""
    lines = ["", "== 8. The HMX alone in a long run (hmx_sustain, measured): f16 deep chains of 80 k-tiles from VTCM =="]
    o = outs.get("hs")
    if not usable(o, include_all) or not o.sustain:
        return lines + ["  no usable hs run" + (f" ({', '.join(o.flags)})" if o and o.flags else "")]
    flop_pass = 8 * 80 * TILE_MAC_FLOP
    edges = (0, 5, 20, 50, 100, 200, 400, 800, 10 ** 9)
    lines.append(f"  {'from ms':>8s} {'to ms':>6s} {'windows':>8s} {'TFLOPS':>7s} {'pcyc/pass':>10s} {'core MHz':>9s}")
    for a, b in zip(edges, edges[1:]):
        ws = [w for w in o.sustain if a * 1000 <= w[0] < b * 1000]
        if not ws:
            continue
        passes, us, pc = (sum(w[i] for w in ws) for i in (1, 2, 3))
        lines.append(f"  {a:8d} {min(b, 800):6d} {len(ws):8d} {passes * flop_pass / us / 1e6:7.2f} "
                     f"{pc / passes:10.0f} {pc / us:9.1f}")
    lines.append(f"  {'pause ms':>8s} {'heat end TF':>11s} {'1st window TF':>13s} {'5 windows TF':>12s}")
    by_pause: dict[int, list[tuple]] = defaultdict(list)
    for rec in o.recovery:
        by_pause[rec[0]].append(rec)
    for pause, recs in sorted(by_pause.items()):
        first = min(recs, key=lambda r: r[1])
        heat = first[5] * flop_pass / first[6] / 1e6 if first[6] else float("nan")
        passes, us = sum(r[2] for r in recs), sum(r[3] for r in recs)
        lines.append(f"  {pause:8d} {heat:11.2f} {first[2] * flop_pass / first[3] / 1e6:13.2f} "
                     f"{passes * flop_pass / us / 1e6:12.2f}")
    return lines


def gdn_tables(outs: dict[str, RunOut], include_all: bool) -> list[str]:
    """The GDN and conv table."""
    lines = ["", "== 5. GATED_DELTA_NET and SSM_CONV (measured us; GFLOPS of the chunk form, computed) =="]
    for c in (*(Case("gdn", "f32", n) for n in (1, 64, 256, 512, 1024)),
              *(Case("gdn", "f32", n, tag="seq") for n in (64, 1024)), Case("conv", "f32", 1024)):
        us, key = find_us(outs, c.name, include_all, ("gdn", "gdnseq", "map2", "trfa2"))
        if us is None:
            continue
        sr = find_series(outs, c.name, include_all, ("gdn", "gdnseq", "trfa2"))
        ref = sr.fresh if sr else us
        per_layer = f"  x{GDN_LAYERS} = {GDN_LAYERS * ref / 1e3:.1f} ms per ubatch" if c.n == 1024 else ""
        lines.append(f"  {c.name:16s} loop {us:9.1f} us  fresh {cell(sr.fresh if sr else None, '.1f', 9)} us  "
                     f"sustained {cell(sr.sustained if sr else None, '.1f', 9)} us  {c.flops / ref / 1e3:8.1f} GFLOPS  "
                     f"{c.op_size / ref / 1e3:6.1f} GB/s of op bytes{per_layer}")
    lines.append("  The model runs the conv as the fused GDN_CONV_CHUNK (concat, conv, silu), which no single op of a "
                 "test file gives: SSM_CONV here is the unfused kernel.")
    return lines


# The priority of the phases of thread 0 when the HMX waits: the first phase in this list that
# covers a moment of the wait gets that moment.
WAIT_ORDER = ("HVX_A_PREP", "HVX_W_DEQUANT", "HVX_O_PROC", "HVX_K_PREP", "HVX_V_PREP", "HVX_SFM_FA",
              "HVX_Q_PREP", "HVX_COMP", "HVX_A_QUANT", "INIT", "L2FLUSH", "BUFF", "DMA")


def intervals_overlap(a0: int, a1: int, spans: list[tuple[int, int]]) -> int:
    """The cycles of [a0, a1) that the spans cover (the spans do not overlap). O(spans)."""
    return sum(max(0, min(a1, e) - max(a0, s)) for s, e in spans)


def merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The union of spans as sorted spans that do not overlap. O(n log n)."""
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def trace_tables(outs: dict[str, RunOut], engine: Engine | None, include_all: bool) -> list[str]:
    """The phase split of the traced ops: the HMX busy share, the HMX pcycles per tile-MAC while busy,
    and the phase of thread 0 at each moment that the HMX waits. Each op that the trace holds in full
    (except the first op of each batch) counts, and the table gives the median over these ops."""
    lines = ["", "== 6. The phase trace (GGML_HEXAGON_PROFILE=3, measured; tile counts from the plan, computed) =="]
    sys.path.insert(0, str(REPO / "tools/trace"))
    try:
        import htp_trace  # noqa: E402
    except ImportError as e:
        return lines + [f"  tools/trace/htp_trace.py does not import: {e}"]
    for key in ("trmm", "trfa", "trfa2"):
        o = outs.get(key)
        if not usable(o, include_all):
            lines.append(f"  {key}: no usable run")
            continue
        log = htp_trace.parse_lines(o.log.splitlines(), key)
        cases = list(o.run.cases)
        per_case: dict[str, list[dict]] = defaultdict(list)
        for session in log.sessions.values():
            for batch in session.batches:
                if not batch.ops or not batch.events:
                    continue
                phases, _, _ = htp_trace.pair_phases(batch)
                cap = int(o.run.trace)
                last: dict[int, int] = {}
                counts = batch.evt_cnt or ()
                for ev in batch.events:
                    last[ev.thread] = max(last.get(ev.thread, 0), ev.abs_cycles)
                # The DSP stops at OPTRACE events, thus a full thread shows a count equal to OPTRACE.
                capped_end = min((last[t] for t in last if t < len(counts) and counts[t] >= cap), default=None)
                for op in batch.ops[1:]:
                    end = op.abs_cycles + op.cycles
                    if capped_end is not None and end > capped_end:
                        break
                    c = match_case(cases, op.name, op.dims, op.types)
                    if c is None:
                        continue
                    per_case[c.name].append(op_split(op, phases))
        for c in cases:
            rows = per_case.get(c.name, [])
            if not rows:
                lines.append(f"  {c.name}: no op in full in the trace")
                continue
            keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float))}
            med = {k: statistics.median(r.get(k, 0) for r in rows) for k in keys}
            waits: dict[str, float] = defaultdict(float)
            for r in rows:
                for name, cyc in r["wait"].items():
                    waits[name] += cyc / r["cycles"] / len(rows)
            p = mm_plan(c) if c.op == "mm" else fa_plan(c) if c.op == "fa" else None
            macs = getattr(p, "tile_macs", 0) if p else 0
            per_tile = med["hmx"] / macs if macs else float("nan")
            eng = f", engine {engine.b:.2f}" if engine else ""
            lines.append(f"  {c.name}: {len(rows)} ops, op {med['cycles'] / med['mhz']:.0f} us, HMX busy "
                         f"{100 * med['hmx'] / med['cycles']:.1f}% ({med['jobs']:.0f} jobs), HMX pcycles per tile-MAC "
                         f"while busy {per_tile:.2f}{eng}")
            lines.append("    the HMX waits " + f"{100 * (1 - med['hmx'] / med['cycles']):.1f}% of the op; thread 0 in that "
                         "time: " + ", ".join(f"{n} {100 * v:.1f}%" for n, v in sorted(waits.items(), key=lambda x: -x[1])
                                               if v >= 0.002))
            hvx = {k[4:]: v for k, v in med.items() if k.startswith("hvx:")}
            lines.append("    HVX busy, mean over the threads: " + ", ".join(
                f"{n} {100 * v / med['cycles'] / N_THREADS:.1f}%" for n, v in sorted(hvx.items(), key=lambda x: -x[1])))
    return lines


def match_case(cases: list[Case], opname: str, dims: str, types: str) -> Case | None:
    """The case of an op line, from its op name, its dims and its types. A MUL_MAT_NX line has more than
    one weight before the activation. The types separate the cases of one shape with other types."""
    for c in cases:
        if not opname.startswith(OP_NAMES[c.op]):
            continue
        if c.op == "mm":
            if dims.startswith(f"{c.k}:{c.m} x ") and f" x {c.k}:{c.n} -> {c.m}:{c.n}" in dims and types.startswith(c.t):
                return c
        elif dims.startswith(dims_prefix(c)) and (c.op != "fa" or f" x {c.t} x " in types):
            return c
    return None


def op_split(op, phases) -> dict:
    """The phase sums of one op: HMX busy cycles, HVX busy cycles per phase, thread 0 in the HMX waits."""
    s, e = op.abs_cycles, op.abs_cycles + op.cycles
    inside = [p for p in phases if s <= p.start_cycles < e and p.end_cycles > p.start_cycles]
    hmx = merge([(p.start_cycles, min(p.end_cycles, e)) for p in inside if p.name == "HMX_COMP"])
    busy = sum(b - a for a, b in hmx)
    out: dict = {"cycles": op.cycles, "mhz": op.mhz if op.mhz > 0 else CLOCK_MHZ, "hmx": busy,
                 "jobs": sum(1 for p in inside if p.name == "HMX_COMP")}
    for p in inside:
        if p.name.startswith("HVX_") or p.name == "INIT":
            out[f"hvx:{p.name}"] = out.get(f"hvx:{p.name}", 0) + (min(p.end_cycles, e) - p.start_cycles)
    gaps, cur = [], s
    for a, b in hmx:
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < e:
        gaps.append((cur, e))
    t0 = defaultdict(list)
    for p in inside:
        if p.thread == 0:
            t0[p.name].append((p.start_cycles, min(p.end_cycles, e)))
    t0m = {n: merge(v) for n, v in t0.items()}
    wait: dict[str, float] = defaultdict(float)
    for a, b in gaps:
        left = [(a, b)]
        for name in WAIT_ORDER + tuple(n for n in t0m if n not in WAIT_ORDER):
            spans = t0m.get(name)
            if not spans:
                continue
            nxt = []
            for x, y in left:
                wait[name] += intervals_overlap(x, y, spans)
                cur2 = x
                for ps, pe in spans:
                    if pe <= x or ps >= y:
                        continue
                    if ps > cur2:
                        nxt.append((cur2, ps))
                    cur2 = max(cur2, pe)
                if cur2 < y:
                    nxt.append((cur2, y))
            left = nxt
        wait["no phase"] += sum(y - x for x, y in left)
    out["wait"] = dict(wait)
    return out


def drift_table(outs: dict[str, RunOut], include_all: bool) -> list[str]:
    """The key cases of the run rep against their first run."""
    lines = ["", "== 7. Drift: the run rep (near the end) against the first run of each case (measured) =="]
    o = outs.get("rep")
    if not usable(o, include_all):
        return lines + ["  no usable rep run"]
    for c in o.run.cases:
        first, key = find_us(outs, c.name, include_all, ("q8a", "q8d", "q8b", "q8c", "peak", "faq8"))
        last = o.us.get(c.name)
        f_first = find_series(outs, c.name, include_all, (key,)) if key else None
        f_last = series_of(o.series.get(c.name, []))
        fresh = (f" fresh {f_first.fresh:9.1f} us, rep fresh {f_last.fresh:9.1f} us, "
                 f"{100 * (f_last.fresh / f_first.fresh - 1):+.1f}%") if f_first and f_last else ""
        if first and last:
            lines.append(f"  {c.name:30s} {key:5s} loop {first:10.1f} us, rep loop {last:10.1f} us, "
                         f"{100 * (last / first - 1):+.1f}%;{fresh}")
    return lines


def conditions(outs: dict[str, RunOut]) -> list[str]:
    """The conditions and the flags of the runs."""
    lines = ["== 0. The conditions =="]
    for o in outs.values():
        hw = HWINFO_RE.search(o.log + o.out)
        info = f" hwinfo threads {hw.group(1)} hmx {hw.group(3)} vtcm {hw.group(4)} MB" if hw else ""
        lines.append(f"  {o.run.key:7s} {'ok' if o.ok else 'FAILED':6s} caps {o.caps}{info}"
                     + (f"  flags: {'; '.join(o.flags)}" if o.flags else ""))
    return lines


def table(root: Path, include_all: bool) -> int:
    """Print the tables."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    outs = {r.key: read_run(root, r) for r in runs() if (root / f"{r.key}-gate.txt").exists()}
    clock, how = clock_mhz(outs)
    parts = [conditions(outs), [f"  the DSP clock: {clock:.1f} MHz ({how})"]]
    i8, engine = i8_tables(outs, clock, include_all)
    hs = outs.get("hs")
    if usable(hs, include_all) and hs.sustain:
        passes, pcyc = sum(w[1] for w in hs.sustain), sum(w[3] for w in hs.sustain)
        engine = Engine(0.0, pcyc / passes / (8 * 80), clock, {})
        i8.append(f"  the engine reference of the HMX% columns and of the floors: hmx_sustain, {engine.b:.2f} pcycles per "
                  f"tile-MAC in chains of 80 k-tiles with the clear and the store of each output tile, "
                  f"{TILE_MAC_FLOP * clock * 1e6 / engine.b / 1e12:.2f} TFLOPS (measured)")
    parts.append(i8)
    kmap = kernel_map(outs)
    parts += [mm_tables(outs, engine, kmap, include_all), prefill_tables(outs, engine, include_all),
              fa_tables(outs, engine, include_all), gdn_tables(outs, include_all),
              trace_tables(outs, engine, include_all), drift_table(outs, include_all),
              sustain_tables(outs, include_all)]
    for p in parts:
        print("\n".join(p))
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("files", help="write phone/ and phone-commands.txt")
    sub.add_parser("plan", help="print the host kernel choice of each case")
    t = sub.add_parser("table", help="print the tables from the pulled outputs")
    t.add_argument("--root", type=Path, default=HERE / "phone-out")
    t.add_argument("--all", action="store_true", help="also use the runs with flags")
    a = ap.parse_args()
    if a.cmd == "files":
        return write_files()
    if a.cmd == "plan":
        return print_plan()
    return table(a.root, a.all)


if __name__ == "__main__":
    sys.exit(main())
