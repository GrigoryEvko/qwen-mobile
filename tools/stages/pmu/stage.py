#!/usr/bin/env python3
"""The phone stage pmu: why each op of the 4B Q8_0 takes its time on HTP0, from the PMU counters.

Usage:
    stage.py setup                              make build/pmu/phone from the files of build/bench-kv/phone
    stage.py commands [--out PATH]              write the phone command file build/pmu/phone-commands.txt
    stage.py events                             print each run with its event ids, names and sources
    stage.py table [--root DIR] [--strict] [--top N]
                                                print the tables from the pulled logs (build/pmu/phone-out)

The runs. Each run is one memprobe process with the context of the app: n_ctx 8192, n_batch = n_ubatch =
1024, 4 host threads, flash attention, Q8_0 K and V with the rotation as the DSP op FWHT,
GGML_HEXAGON_OPFUSION=1 and GGML_HEXAGON_OPFUSION_STATE=1 (the variant B of build/bench-kv). The tool
decodes a prompt of 4096 tokens in 4 ubatches, then 8 or 32 tokens at the depth 4096. Thus one run gives
the prefill ubatches at the depths 0, 1024, 2048 and 3072, and the decode tokens. A counter run sets
GGML_HEXAGON_PROFILE to 8 event ids (profile mode 2: one delta of each counter for each op). A reference
run sets GGML_HEXAGON_PROFILE=1 (the times only), as the runs fdec of build/bench-kv do. The run "default"
sets GGML_HEXAGON_PROFILE=2 (mode 2 with the 8 events that the backend has as its preset). The trace run
sets GGML_HEXAGON_PROFILE=3 (the phase events of each DSP thread) for one prefill ubatch of 1024 tokens.

The table reads three kinds of graph from each run:
    p0    the prefill ubatch of 1024 tokens at the depth 0 (graph 1)
    p3    the prefill ubatch of 1024 tokens at the depth 3072 (graph 4)
    dec   one decode token at the depth 4096: the sum of the decode graphs after the first, divided by
          their count. The first decode graph holds the costs of the first decode, thus the table does
          not use it.
It also prints the ops of the 4 prefill ubatches layer by layer (the HMX slowdown inside a ubatch), and the
decode ops that take much more time than the same op in the other tokens (the decode stall events).

How to read a counter. The counters are global: each one counts the events of all 6 hardware threads, of
the HMX and of the DMA engine. The cycles of an op are the elapsed cycles of DSP thread 0. Thus a rate
per cycle can be more than 1 when an event counts for each thread or for each cluster. The top-down
events (*_PVIEW_CYCLES) give one reason for each cycle of each cluster in which the cluster commits no
packet. The table adds the cycles with a commit (CYCLES_n_PACKET_COMMITTED) to the stall cycles, thus the
column "sum" gives the number of clusters when the top-down count is complete.

The event ids. The ids of tools/prof/pmu.py agree with the V79 column of the table that maps the itrace
event names to raw PMU ids in libitrace.so of the Hexagon SDK 6.6.0.0 (libs/itrace/prebuilt). The new ids
in NEW_EVENTS come from two sources: that same table (the names of libs/itrace/inc/itrace_dsp_events_pmu.h),
and, for the HMX block 0x200 thru 0x295, the enum _PMU_EVENTS_ENUM_ of libhexagonissv79.so (Hexagon Tools
19.0.07, the simulator). The public itrace table has no name for the ids 0x193 thru 0x23f on V79. The
Snapdragon Profiler 2026.9.0 (pluginNPUV2) registers 0x200 as PMU_HMX_ACTIVE for the metric "HMX
Utilization" on the CDSP of V68 and later, thus one id of the block has a second source. This stage
confirms the block: an HMX event counts on the ops with an HMX kernel and stays near zero on the other ops.

The bytes and the FLOPs of an op come from tools/prof/optable.py (the bytes that the tensors of the op hold,
the dense FLOP count), thus the two tools agree.

The table uses a run when its gate passed, its exit code is 0, its thermal status after the run is 0, its
log has no failure line, the log has the profile mode and the events of the run, and the run has the
expected graph count. A change of the CPU caps is a note: the caps change the host part of a decode step,
not the DSP cycles of an op. --strict also removes the runs with a note. The table only reads files. Time
O(size of the logs).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import math
import os
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# The file is tools/stages/pmu/stage.py, and build/pmu/stage.py is a link to it. resolve() follows the link.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "prof"))
sys.path.insert(0, str(REPO / "tools" / "trace"))
import optable  # noqa: E402  # type: ignore[import-not-found]
import pmu  # noqa: E402  # type: ignore[import-not-found]

ADB = "adb -s 192.168.14.130:5555"
PHONE = "/data/local/tmp/qwen/pmu"
MODEL_FILE = "Qwen3.5-4B-Q8_0.gguf"
MODEL = f"/data/local/tmp/qwen/models/{MODEL_FILE}"
GATE_KB = 8388608
LAPTOP_STAGE = "build/pmu"
BOX = "grigory@10.10.20.200:airi/qwen-mobile/build/pmu"
STAGE_DIR = REPO / LAPTOP_STAGE
SOURCE = REPO / "build" / "bench-kv" / "phone"
STAGE_FILES = ("bin/gate.sh", "bin/memprobe", "lib/libggml-base.so", "lib/libggml-cpu.so", "lib/libggml-hexagon.so",
               "lib/libggml-htp-v79.so", "lib/libggml-opencl.so", "lib/libggml.so", "lib/libllama-common.so",
               "lib/libllama.so", "lib/libmtmd.so")
# The environment of the app (init_impl in llama_jni.cpp) and the stage libraries: the variant B of bench-kv.
LIB_ENV = (f"LD_LIBRARY_PATH={PHONE}/lib ADSP_LIBRARY_PATH={PHONE}/lib "
           "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1")
# The context of the app (load_impl in llama_jni.cpp), as the runs fdec of bench-kv.
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"
UBATCH = 1024
# The decode tokens of a short run and of a long run. The long runs catch the decode stall events of
# about 3 ms, which occur 3 or 4 times in 8 tokens.
SHORT, LONG = 8, 32
# The trace events for each DSP thread and batch. 11 slots of 16384 events of 8 bytes add 1.4 MB to the
# message of a batch, and the log holds at most 3 x 11 x 16384 lines. The events of an op after the limit
# of a thread are lost, thus the trace covers the first ops of each batch.
OPTRACE = 16384
# The limit of one run in seconds. The gate and the lines after the tool take about 6 s, thus a phone
# command stays under 120 s. A long run takes about 17 s.
LIMIT = 100

THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
PGREP = f"{ADB} shell 'pgrep -x llama-bench; pgrep -x memprobe; echo pgrep-done'"
# The highest temperature of the NPU thermal zones (type nsp*) in millidegrees, or an empty value when no
# zone of that type is readable.
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')

# ---- The events ----

SDK = "SDK 6.6.0.0 libitrace.so, V79 column"
ISS = "libhexagonissv79.so enum (Hexagon Tools 19.0.07)"
SDP = "Snapdragon Profiler pluginNPUV2 PMU_HMX_ACTIVE"

# name -> (raw id, source, description). The events that tools/prof/pmu.py does not have.
NEW_EVENTS: dict[str, tuple[int, str, str]] = {
    "DU_CONFLICT_PVIEW_CYCLES": (0xEC, SDK, "cycles the cluster cannot commit because of a DU resource conflict"),
    "CYCLES_1_PACKET_COMMITTED": (0x300, SDK, "cycles with one packet committed"),
    "CYCLES_2_PACKET_COMMITTED": (0x301, SDK, "cycles with two packets committed"),
    "CYCLES_3_PACKET_COMMITTED": (0x302, SDK, "cycles with three packets committed"),
    "CYCLES_4_PACKET_COMMITTED": (0x303, SDK, "cycles with four packets committed"),
    "CYCLES_1_THREAD_RUNNING": (0x3B, SDK, "cycles with exactly 1 thread not in wait or stop"),
    "CYCLES_2_THREAD_RUNNING": (0x3C, SDK, "cycles with exactly 2 threads not in wait or stop"),
    "CYCLES_3_THREAD_RUNNING": (0x3D, SDK, "cycles with exactly 3 threads not in wait or stop"),
    "CYCLES_4_THREAD_RUNNING": (0x3E, SDK, "cycles with exactly 4 threads not in wait or stop"),
    "CYCLES_5_THREAD_RUNNING": (0x0A, SDK, "cycles with exactly 5 threads not in wait or stop"),
    "CYCLES_6_THREAD_RUNNING": (0x0B, SDK, "cycles with exactly 6 threads not in wait or stop"),
    "CYCLES_1_HVX_CONTEXTS_RUNNING": (0x115, SDK, "cycles with 1 HVX context in operation"),
    "CYCLES_2_HVX_CONTEXTS_RUNNING": (0x116, SDK, "cycles with 2 HVX contexts in operation"),
    "CYCLES_3_HVX_CONTEXTS_RUNNING": (0x117, SDK, "cycles with 3 HVX contexts in operation"),
    "CYCLES_4_HVX_CONTEXTS_RUNNING": (0x12C, SDK, "cycles with 4 HVX contexts in operation"),
    "CYCLES_5_HVX_CONTEXTS_RUNNING": (0x191, SDK, "cycles with 5 HVX contexts in operation"),
    "CYCLES_6_HVX_CONTEXTS_RUNNING": (0x192, SDK, "cycles with 6 HVX contexts in operation"),
    "HVX_ST_L2_OUTSTANDING": (0x104, SDK, "HVX stall cycles, a store is not yet in L2"),
    "HVX_SCATGATH_FULL": (0x106, SDK, "HVX stall cycles, the scatter/gather scoreboard is full"),
    "HVX_SCATGATH_IN_FULL": (0x107, SDK, "HVX stall cycles, the scatter/gather input buffer is full"),
    "HVX_VOLTAGE_UNDER": (0x10A, SDK, "HVX throttle cycles, the voltage model is below its limit"),
    "HVX_POWER_OVER": (0x10B, SDK, "HVX throttle cycles, the sustained power is over the budget"),
    "HVX_PKT_PARTIAL": (0x10C, SDK, "HVX stall cycles because of a multi-issue packet"),
    "HVX_CORE_VFIFO_FULL_STALL": (0x113, SDK, "cycles a thread stalls because the vector FIFO is full"),
    "HVX_VFIFO_EMPTY": (0x190, SDK, "cycles a thread has an empty vector FIFO"),
    "THREAD_LMH_THROTTLE": (0x28, SDK, "a thread is over the limits management threshold (throttle by priority)"),
    "LMH_THROTTLE": (0x29, SDK, "throttle: the peak current is over the LMH current limit"),
    "GLOBAL_POWERLIMITS_OVER": (0x2D, SDK, "the sustained global power is over the global limit"),
    "DPM_AVG_COMPRESSED": (0x340, SDK, "the digital power meter: overflows of the local DPM accumulator"),
    "UDMA_DMPOLL_CYCLES": (0x245, SDK, "cycles of the DMA command dmpoll"),
    "UDMA_DMWAIT_CYCLES": (0x246, SDK, "cycles of the DMA command dmwait (a thread waits for the DMA)"),
    "UDMA_NONCOHERENT_RD_CYCLES": (0x262, SDK, "cycles the DMA waits for a read that bypasses the caches"),
    "UDMA_VTCM_WR_CYCLES": (0x265, SDK, "cycles the DMA waits for a write to VTCM"),
    "UDMA_RD_BUFFER_LEVEL_FULL": (0x269, SDK, "cycles with the DMA read buffer fully allocated"),
    "L2_UDMA_VTCM_CONGESTION": (0x270, SDK, "cycles a DMA access to VTCM waits for a conflict or for credits"),
    "AXI_READ_REQUEST_EVEN": (0xF2, SDK, "read requests of the even interleaved AXI master"),
    "AXI_LINE32_READ_REQUEST_EVEN": (0xF3, SDK, "32-byte read requests of the even AXI master"),
    "AXI_LINE64_READ_REQUEST_EVEN": (0xF8, SDK, "64-byte read requests of the even AXI master"),
    "AXI_LINE128_READ_REQUEST_EVEN": (0xF1, SDK, "128-byte read requests of the even AXI master"),
    "AXI_LINE256_READ_REQUEST_EVEN": (0xFC, SDK, "256-byte read requests of the even AXI master"),
    "AXI3_READ_REQUEST": (0x2FB, SDK, "read requests of the AXI3 master"),
    "AXI3_LINE64_READ_REQUEST": (0x316, SDK, "64-byte read requests of the AXI3 master"),
    "HMX_ACTIVE": (0x200, f"{ISS}; {SDP}", "cycles the HMX is active"),
    "HMX_CVT_FULL": (0x201, ISS, "the HMX convert FIFO is full"),
    "HMX_MAC_FULL": (0x202, ISS, "the HMX MAC FIFO is full"),
    "HMX_DROP": (0x203, ISS, "HMX drop"),
    "HMX_CVT": (0x204, ISS, "HMX convert"),
    "HMX_MAC": (0x205, ISS, "HMX MAC"),
    "HMX_PKT_THREAD": (0x206, ISS, "HMX packets by thread"),
    "HMX_MXFIFO_FULL": (0x207, ISS, "the HMX instruction FIFO is full"),
    "HMXMAC_ACT_OUTSTANDING": (0x209, ISS, "MAC stall: the activation read is not complete"),
    "HMXMAC_WGT_OUTSTANDING": (0x20A, ISS, "MAC stall: the weight read is not complete"),
    "HMXMAC_MULT_DROP": (0x20B, ISS, "MAC multiply drop"),
    "HMXMAC_POWER_OVER": (0x20D, ISS, "MAC throttle for power"),
    "HMXMAC_FXP_PARTIAL": (0x20E, ISS, "MAC fixed-point partial cycles"),
    "HMXMAC_FLT_PARTIAL": (0x20F, ISS, "MAC float partial cycles"),
    "HMXMAC_DRAIN_PARTIAL": (0x210, ISS, "MAC drain partial cycles"),
    "HMXMAC_FXP": (0x211, ISS, "MAC fixed-point cycles"),
    "HMXMAC_FLT": (0x212, ISS, "MAC float cycles"),
    "HMXMAC_DRAIN": (0x213, ISS, "MAC drain cycles"),
    "HMX_CLK": (0x214, ISS, "HMX clock cycles"),
    "HMXMAC_ORDER": (0x228, ISS, "MAC order stall"),
    "HMXRDACT_PARTIAL": (0x230, ISS, "partial activation reads from VTCM"),
    "HMXRDWGT_PARTIAL": (0x231, ISS, "partial weight reads from VTCM"),
    "HMXRDACT_ACT": (0x232, ISS, "activation reads from VTCM"),
    "HMXRDWGT_WGT": (0x233, ISS, "weight reads from VTCM"),
    "HMXRDWGT_SCALE": (0x234, ISS, "scale reads from VTCM"),
    "HMXWR_OUTSTANDING": (0x237, ISS, "HMX write to VTCM not complete"),
    "HMXWR": (0x23B, ISS, "HMX writes to VTCM"),
    "HMX_MXFIFO_EMPTY": (0x291, ISS, "the HMX instruction FIFO is empty"),
    "HMX_LIMITS_THROTTLE_TLMH": (0x292, ISS, "HMX throttle by the thread limits management"),
    "HMX_LIMITS_THROTTLE_LMH": (0x293, ISS, "HMX throttle by the limits management"),
    "HMX_DPM_AVG_COMPRESSED": (0x294, ISS, "the digital power meter of the HMX"),
    "HMX_POWERLIMITS_OVER": (0x295, ISS, "HMX over the power limits"),
}

EVENTS: dict[str, int] = {name: eid for name, (eid, _) in pmu.EVENTS.items()}
EVENTS.update({name: eid for name, (eid, _, _) in NEW_EVENTS.items()})
NAME_OF: dict[int, str] = {eid: name for name, eid in EVENTS.items()}
if len(NAME_OF) != len(EVENTS):
    raise SystemExit("stage.py: two event names have the same raw id. Correct NEW_EVENTS.")

# The sets of this stage. Each one holds 8 events, because the hardware has 8 counters.
NEW_SETS: dict[str, tuple[str, list[str]]] = {
    "topdown-a": ("the top-down stall reasons, part 1 (with THREAD_IDLE, the thread wait)",
                  ["THREAD_IDLE_PVIEW_CYCLES", "ARCH_LOCK_PVIEW_CYCLES", "REDIRECT_PVIEW_CYCLES",
                   "IU_NO_PKT_PVIEW_CYCLES", "DU_CACHE_MISS_PVIEW_CYCLES", "DU_BUSY_OTHER_PVIEW_CYCLES",
                   "CU_BUSY_PVIEW_CYCLES", "DU_CONFLICT_PVIEW_CYCLES"]),
    "topdown-b": ("the top-down stall reasons, part 2, and the cycles with a commit",
                  ["COPROC_BUSY_PVIEW_CYCLES", "DU_UNCACHED_PVIEW_CYCLES", "SYSTEM_BUSY_PVIEW_CYCLES",
                   "CYCLES_1_PACKET_COMMITTED", "CYCLES_2_PACKET_COMMITTED", "CYCLES_3_PACKET_COMMITTED",
                   "CYCLES_4_PACKET_COMMITTED", "COMMITTED_PKT_ANY"]),
    "engines": ("the HMX, the HVX, the DMA and the threads in one run: which engine is busy",
                ["HMX_ACTIVE", "HMXMAC_FLT", "HMX_MXFIFO_EMPTY", "HVX_ACTIVE", "UDMA_ACTIVE",
                 "COPROC_BUSY_PVIEW_CYCLES", "SYSTEM_BUSY_PVIEW_CYCLES", "THREAD_IDLE_PVIEW_CYCLES"]),
    "hmx-mac-a": ("the HMX MAC cycles and the two read stalls of the MAC",
                  ["HMX_ACTIVE", "HMX_CLK", "HMXMAC_FLT", "HMXMAC_FLT_PARTIAL", "HMXMAC_ACT_OUTSTANDING",
                   "HMXMAC_WGT_OUTSTANDING", "HMXMAC_DRAIN", "HMXMAC_DRAIN_PARTIAL"]),
    "hmx-mac-b": ("the other MAC stalls and the instruction FIFO of the HMX",
                  ["HMX_ACTIVE", "HMXMAC_MULT_DROP", "HMXMAC_POWER_OVER", "HMXMAC_FXP", "HMXMAC_FXP_PARTIAL",
                   "HMXMAC_ORDER", "HMX_MXFIFO_FULL", "HMX_MXFIFO_EMPTY"]),
    "hmx-rd": ("the VTCM reads and writes of the HMX",
               ["HMX_ACTIVE", "HMXRDACT_ACT", "HMXRDWGT_WGT", "HMXRDWGT_SCALE", "HMXRDACT_PARTIAL",
                "HMXRDWGT_PARTIAL", "HMXWR", "HMXWR_OUTSTANDING"]),
    "hmx-cvt": ("the convert and MAC FIFOs of the HMX",
                ["HMX_ACTIVE", "HMX_CVT_FULL", "HMX_MAC_FULL", "HMX_DROP", "HMX_CVT", "HMX_MAC",
                 "HMX_PKT_THREAD", "HMX_POWERLIMITS_OVER"]),
    "power-a": ("the HMX clock, the HMX power limits and the HMX power meter (the HMX slowdown in a ubatch)",
                ["HMX_CLK", "HMX_ACTIVE", "HMXMAC_FLT", "HMXMAC_POWER_OVER", "HMX_POWERLIMITS_OVER",
                 "HMX_LIMITS_THROTTLE_TLMH", "HMX_LIMITS_THROTTLE_LMH", "HMX_DPM_AVG_COMPRESSED"]),
    "power-b": ("the throttles of the core and the HVX, and the power meter of the core",
                ["THREAD_LMH_THROTTLE", "LMH_THROTTLE", "GLOBAL_POWERLIMITS_OVER", "HVX_POWER_OVER",
                 "HVX_VOLTAGE_UNDER", "DPM_AVG_COMPRESSED", "HMX_CLK", "COMMITTED_PKT_ANY"]),
    "hvx2": ("the HVX stall reasons that the set hvx does not have (the full HVX utilization of the simulator)",
             ["HVX_ST_L2_OUTSTANDING", "HVX_SCATGATH_FULL", "HVX_SCATGATH_IN_FULL", "HVX_PKT_PARTIAL",
              "HVX_VOLTAGE_UNDER", "HVX_PKT", "HVX_CORE_VFIFO_FULL_STALL", "HVX_VFIFO_EMPTY"]),
    "bw-even": ("the reads of the even AXI master and of AXI3, against the set bandwidth",
                ["AXI_READ_REQUEST_EVEN", "AXI_LINE32_READ_REQUEST_EVEN", "AXI_LINE64_READ_REQUEST_EVEN",
                 "AXI_LINE128_READ_REQUEST_EVEN", "AXI_LINE256_READ_REQUEST_EVEN", "AXI_READ_REQUEST",
                 "AXI3_READ_REQUEST", "AXI3_LINE64_READ_REQUEST"]),
    "udma": ("where the DMA waits: the thread wait for the DMA, the DDR read, the VTCM write",
             ["UDMA_ACTIVE", "UDMA_DMWAIT_CYCLES", "UDMA_DMPOLL_CYCLES", "UDMA_NONCOHERENT_RD_CYCLES",
              "UDMA_VTCM_WR_CYCLES", "L2_UDMA_VTCM_CONGESTION", "UDMA_RD_BUFFER_LEVEL_FULL",
              "L2_UDMA_BYPASS_RD"]),
    "threads": ("the number of threads that run (not in wait or stop) and the thread wait",
                ["CYCLES_1_THREAD_RUNNING", "CYCLES_2_THREAD_RUNNING", "CYCLES_3_THREAD_RUNNING",
                 "CYCLES_4_THREAD_RUNNING", "CYCLES_5_THREAD_RUNNING", "CYCLES_6_THREAD_RUNNING",
                 "THREAD_IDLE_PVIEW_CYCLES", "COMMITTED_PKT_ANY"]),
    "hvxctx": ("the number of HVX contexts in operation",
               ["CYCLES_1_HVX_CONTEXTS_RUNNING", "CYCLES_2_HVX_CONTEXTS_RUNNING", "CYCLES_3_HVX_CONTEXTS_RUNNING",
                "CYCLES_4_HVX_CONTEXTS_RUNNING", "CYCLES_5_HVX_CONTEXTS_RUNNING", "CYCLES_6_HVX_CONTEXTS_RUNNING",
                "HVX_ACTIVE", "COMMITTED_PKT_ANY"]),
}


@dataclass(frozen=True)
class Run:
    """One phone run: its key, the profile mode (1, 2 or 3), the 8 events of mode 2, the prompt and decode
    token counts, more environment and a description. When preset is True, the run sets
    GGML_HEXAGON_PROFILE to the mode alone, thus the backend takes its preset events (which must be the
    events of the run)."""
    key: str
    mode: int
    events: tuple[str, ...]
    n_prompt: int
    n_gen: int
    env: str
    text: str
    preset: bool = False

    @property
    def name(self) -> str:
        """The run name, which is also the stem of its output files."""
        return f"4b-{self.key}"

    @property
    def args(self) -> str:
        """The prompt and decode arguments of memprobe."""
        return f"-p {self.n_prompt} -n {self.n_gen}"

    @property
    def n_prefill(self) -> int:
        """The prefill graphs of the run: one for each ubatch."""
        return -(-self.n_prompt // UBATCH)

    @property
    def n_graphs(self) -> int:
        """The graphs of the run: the prefill ubatches and the decode tokens."""
        return self.n_prefill + self.n_gen

    @property
    def profile(self) -> str:
        """The value of GGML_HEXAGON_PROFILE."""
        if self.mode == 2 and self.events and not self.preset:
            return ",".join(f"0x{EVENTS[n]:x}" for n in self.events)
        return str(self.mode)


def set_run(key: str, set_name: str, pass_no: int = 1, n_gen: int = SHORT) -> Run:
    """A counter run of one pass of a set of pmu.py or of NEW_SETS."""
    if set_name in NEW_SETS:
        text, names = NEW_SETS[set_name]
    else:
        text, passes = pmu.SETS[set_name]
        names = passes[pass_no - 1]
        text = f"pmu.py set {set_name} pass {pass_no}: {text}"
    if len(names) != 8:
        raise SystemExit(f"stage.py: the set {set_name} has {len(names)} events, not 8")
    return Run(key, 2, tuple(names), 4 * UBATCH, n_gen, "", f"{text.rstrip('.')}, {n_gen} decode tokens")


def ref_run(key: str, where: str) -> Run:
    """A reference run in mode 1 (no counters) with the long decode."""
    return Run(key, 1, (), 4 * UBATCH, LONG, "", f"the time reference, mode 1, {where}, {LONG} decode tokens")


# The order spreads the three reference runs over the stage, thus they show a drift of the clocks.
RUNS: tuple[Run, ...] = (
    ref_run("ref1", "at the start (the command of bench-kv 4b-fdec-1-b with more decode tokens)"),
    set_run("stalls", "stalls", n_gen=LONG),
    set_run("topdown-a", "topdown-a", n_gen=LONG),
    set_run("topdown-b", "topdown-b", n_gen=LONG),
    set_run("engines", "engines", n_gen=LONG),
    set_run("hmx-mac-a", "hmx-mac-a"),
    set_run("hmx-mac-b", "hmx-mac-b"),
    set_run("hmx-rd", "hmx-rd"),
    set_run("hmx-cvt", "hmx-cvt"),
    set_run("power-a", "power-a"),
    set_run("power-b", "power-b"),
    ref_run("ref2", "in the middle of the stage"),
    set_run("hvx", "hvx"),
    set_run("hvx2", "hvx2"),
    set_run("pipes", "pipes"),
    set_run("bw1", "bandwidth", 1),
    set_run("bw2", "bandwidth", 2),
    set_run("bw-even", "bw-even"),
    set_run("dma", "dma"),
    set_run("udma", "udma", n_gen=LONG),
    set_run("threads", "threads", n_gen=LONG),
    set_run("hvxctx", "hvxctx"),
    Run("default", 2, tuple(pmu.SETS["default"][1][0]), 4 * UBATCH, SHORT, "",
        f"mode 2 with the preset events of the backend (GGML_HEXAGON_PROFILE=2), {SHORT} decode tokens", preset=True),
    ref_run("ref3", "at the end of the stage"),
    Run("trace", 3, (), UBATCH, 0, f"GGML_HEXAGON_OPTRACE={OPTRACE}",
        f"mode 3, the phase events of each DSP thread, one prefill ubatch of 1024 tokens, {OPTRACE} events "
        "for each thread and batch"),
)
RUN_BY_KEY = {r.key: r for r in RUNS}
REF_KEYS = ("ref1", "ref2", "ref3")

# ---- The command file ----

N_LONG = sum(r.n_gen == LONG for r in RUNS)
HEADER = f"""\
# Phone stage "pmu": why each op of the 4B Q8_0 takes its time on HTP0, from the PMU counters, in the
# configuration of the app (Q8_0 K and V with the FWHT rotation, OPFUSION=1, OPFUSION_STATE=1: the variant B of
# bench-kv). One tool: memprobe with the context of the app, a prompt of 4096 tokens (4 ubatches of 1024), then
# {SHORT} or {LONG} decode tokens at the depth 4096, as the runs fdec of bench-kv. Thus each run gives the 4 prefill
# ubatches (depth 0 to 3072) and the decode tokens at depth 4096.
#
# The runs, {len(RUNS)}, one memprobe process each ({N_LONG} of them with {LONG} decode tokens, to catch the decode stall events):
#   ref1 ref2 ref3   GGML_HEXAGON_PROFILE=1, no counters: the time reference at the start, the middle and the end
#   stalls hvx pipes bw1 bw2 dma default    the sets of tools/prof/pmu.py (bw1 and bw2: the two passes of bandwidth)
#   topdown-a topdown-b    all 11 top-down stall reasons and the cycles with a commit (a full top-down count)
#   engines          HMX, HVX, DMA and the thread wait in one run
#   hmx-mac-a hmx-mac-b hmx-rd hmx-cvt    the HMX block 0x200-0x295 (names from libhexagonissv79.so): confirms the ids
#   power-a power-b  the HMX clock, the HMX and core throttles and power meters: the HMX slowdown inside a ubatch
#   hvx2             the HVX stalls that the set hvx does not have
#   bw-even          the even AXI master and AXI3, against bw1
#   udma             the DMA waits (dmwait, the DDR read, the VTCM write, the congestion)
#   threads hvxctx   the threads and the HVX contexts in operation, the thread wait between the jobs of an op
#   trace            GGML_HEXAGON_PROFILE=3, one prefill ubatch of 1024 tokens, the phases of each DSP thread
# The libraries: the files of build/bench-kv/phone (HEAD e8a3a07 plus the switch GGML_HEXAGON_FWHT, preset 1), hard
# links in build/pmu/phone. The phone directory is {PHONE}. The model is not pushed.
# Each run: the thermal line, then bin/gate.sh (the Qwen app stopped, the screen on, thermal 0, no charger,
# MemAvailable 8 GB), memprobe under timeout -s KILL {LIMIT}, the exit code and the conditions after the run (thermal,
# caps, battery, NSP zone), then the pgrep line.
#
# Put this file into /tmp/phone-timing-stages.txt: it is a timing stage (unlocked phone, no charger).
# Run from /home/grigory/airi/qwen-mobile on the laptop, in order. Time: about 13 minutes, plus the waits for thermal
# status 0 and a battery of 38 C or less. The tool time is about 6.5 minutes ({len(RUNS)} runs of 14 s, {N_LONG} of them
# 3 s longer). The push is 125 MB, the pull about 90 to 130 MB (the trace log and the long runs are the largest files).
# Then, on the box: ssh grigory@10.10.20.200 'cd ~/airi/qwen-mobile && python3 build/pmu/stage.py table'
"""


def run_lines(run: Run) -> list[str]:
    """The lines of one run: a title, the thermal line, the run and the pgrep line."""
    stem = f"{PHONE}/out/{run.name}"
    env = " ".join(x for x in (LIB_ENV, f"GGML_HEXAGON_PROFILE={run.profile}", run.env) if x)
    cmd = (f"sh {PHONE}/bin/gate.sh {GATE_KB} > {stem}-gate.txt && {BEFORE} >> {stem}-gate.txt && "
           f"timeout -s KILL {LIMIT} env {env} "
           f"{PHONE}/bin/memprobe -m {MODEL} {PROBE_ARGS} {run.args} "
           f"> {stem}.out 2> {stem}.log; echo \"rc=$?\" >> {stem}-gate.txt; {AFTER} >> {stem}-gate.txt; "
           f"cat {stem}-gate.txt")
    return ["#", f"# REAL-MODEL {MODEL_FILE.removesuffix('.gguf')}: {run.name}, {run.text}",
            THERMAL, f"{ADB} shell '{cmd}'", PGREP]


def setup_lines() -> list[str]:
    """The lines that copy the stage to the laptop and to the phone and check its files."""
    local_bin = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("bin/"))
    local_lib = " ".join(f"{LAPTOP_STAGE}/phone/{f}" for f in STAGE_FILES if f.startswith("lib/"))
    return [
        f"mkdir -p {LAPTOP_STAGE} && rsync -a --delete {BOX}/phone/ {LAPTOP_STAGE}/phone/",
        f"(cd {LAPTOP_STAGE}/phone && sha256sum -c SHA256SUMS)",
        # No "models/Qwen3.5" in this line: the runner gates each line with that text as a model run.
        f"{ADB} shell 'ls -l /data/local/tmp/qwen/models | grep -E \"{MODEL_FILE}\"'",
        f"{ADB} shell 'rm -rf {PHONE} && mkdir -p {PHONE}/bin {PHONE}/lib {PHONE}/out'",
        f"{ADB} push {local_bin} {PHONE}/bin/",
        f"{ADB} push {local_lib} {PHONE}/lib/",
        f"{ADB} push {LAPTOP_STAGE}/phone/SHA256SUMS {PHONE}/",
        f"{ADB} shell 'cd {PHONE} && sha256sum -c SHA256SUMS | grep -c OK && chmod 755 {PHONE}/bin/*'",
    ]


def output_lines() -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the phone directory. The phone
    directory goes only when the pull has each of its files."""
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        THERMAL,
        f"{ADB} shell 'pgrep -x llama-bench; pgrep -x memprobe; ls {PHONE}/out | wc -l; du -sh {PHONE}/out'",
        f"rm -rf {LAPTOP_STAGE}/phone-out",
        f"{ADB} pull {PHONE}/out {LAPTOP_STAGE}/phone-out",
        f"rsync -a --delete {LAPTOP_STAGE}/phone-out/ {BOX}/phone-out/",
        f"test \"$(ls {LAPTOP_STAGE}/phone-out | wc -l)\" -eq "
        f"\"$({ADB} shell 'ls {PHONE}/out | wc -l' | tr -d '\\r')\" "
        f"&& {ADB} shell 'rm -rf {PHONE}' && echo removed {PHONE}",
    ]


def write_commands(path: Path) -> int:
    """Write the command file and return its line count."""
    lines = HEADER.rstrip("\n").split("\n") + setup_lines()
    lines += ["#", f"# ==== {MODEL_FILE.removesuffix('.gguf')}: {len(RUNS)} runs ===="]
    for run in RUNS:
        lines += run_lines(run)
    lines += output_lines()
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def sha256(path: Path) -> str:
    """The SHA-256 of one file. O(file size)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup() -> int:
    """Make build/pmu/phone: hard links to the stage files of build/bench-kv/phone (a copy when a link is
    not possible), after a check of each file against the SHA256SUMS of bench-kv. Write SHA256SUMS."""
    want = {}
    for line in (SOURCE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        want[name.strip()] = digest
    out = STAGE_DIR / "phone"
    if out.exists():
        shutil.rmtree(out)
    (out / "bin").mkdir(parents=True)
    (out / "lib").mkdir()
    sums = []
    for rel in STAGE_FILES:
        src, dst = SOURCE / rel, out / rel
        digest = sha256(src)
        if want.get(rel) != digest:
            print(f"stage.py: {src} does not agree with {SOURCE}/SHA256SUMS. Build bench-kv again.", file=sys.stderr)
            return 1
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        sums.append(f"{digest}  {rel}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"{out}: {len(sums)} files, the bytes of {SOURCE}")
    return 0


def print_events() -> int:
    """Print each run with its events, their raw ids and their sources."""
    for run in RUNS:
        print(f"{run.name:14s} {run.args:14s} GGML_HEXAGON_PROFILE={run.profile}  {run.text}")
        for n in run.events:
            src = NEW_EVENTS[n][1] if n in NEW_EVENTS else "tools/prof/pmu.py"
            print(f"    0x{EVENTS[n]:03x} {n:32s} {src}")
    return 0


# ---- The parse ----

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
BEFORE_RE = re.compile(r"before: nsp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
MODE_RE = re.compile(r"Profiling mode (\d+) : pmu-evt \[ ([^\]]*)\]")
PREFILL_RE = re.compile(r"^TIME prefill ([\d.]+) tokens=(\d+)", re.M)
STEPS_RE = re.compile(r"^TIME decode-steps min=([\d.]+) median=([\d.]+) max=([\d.]+)", re.M)
OP_RE = re.compile(r"profile-op (?P<op>[A-Z0-9_+]+)\|(?P<names>[^|]*)\|(?P<dims>[^|]*)\|(?P<types>[^|]*)\|"
                   r"(?P<strides>[^|]*)\|(?P<kp>[^|]*)\|usec (?P<usec>\d+) cycles (?P<cycles>\d+) "
                   r"start (?P<start>\d+) mhz (?P<mhz>[\d.]+)(?: pmu \[(?P<pmu>[\d,]+)\])?")
# A line of a log that removes the run from the tables (the rule of tools/stages/fixed/stage.py).
FAILURE_RE = re.compile(r"follow-failed|AddressSanitizer|GGML_ASSERT|dspqueue_read failed")
# The first DSP op of each graph of the model: the norm of layer 0 on the input embedding.
GRAPH_START = "-> attn_norm-0"
# The bytes of one element of each type, for the weight bytes. A Q8_0 block holds 32 values in 34 bytes.
TYPE_BYTES = optable.TYPE_BYTES


def kind_graphs(run: Run) -> dict[str, tuple[int, ...]]:
    """The graphs of each kind of one run, as indices into its graphs. The first decode graph is not used."""
    first_dec = run.n_prefill + 1
    return {"p0": (0,), "p3": (3,), "dec": tuple(range(first_dec, run.n_graphs))}


KINDS = (
    ("p0", "the prefill ubatch of 1024 tokens at depth 0"),
    ("p3", "the prefill ubatch of 1024 tokens at depth 3072"),
    ("dec", "one decode token at depth 4096 (the decode graphs after the first, divided by their count)"),
)


def split_list(text: str) -> tuple[list[str], str]:
    """The source items and the output item of a field of a profile line, split at " -> " and " x "."""
    head, _, out = text.partition(" -> ")
    return head.split(" x "), out


def role(names: str) -> str:
    """The short name of the first source tensor that has a name: without the layer, the view marks and
    the numbers of the graph."""
    srcs, out = split_list(names)
    for s in srcs + [out]:
        s = re.sub(r"HTP\d#", "", s)
        s = re.sub(r"\((reshaped|view|permuted|cont|transposed|copy of [^)]*)\)", "", s)
        s = re.sub(r"blk\.\d+\.", "", s)
        s = re.sub(r"#\d+", "", s).strip()
        s = re.sub(r"(_l\d+|-\d+)$", "", s)
        if s:
            return s
    return "-"


def dims_of(item: str) -> list[int]:
    """The dimensions of one item of the dims field, for example "2560:9216" gives [2560, 9216]."""
    try:
        return [int(x) for x in item.strip().split(":") if x]
    except ValueError:
        return []


@dataclass
class OpLine:
    """One profile-op line of a DSP op."""
    op: str
    names: str
    dims: str
    types: str
    kernel: str
    usec: int
    cycles: int
    start: int
    pmu: tuple[int, ...] | None

    @functools.cached_property
    def group(self) -> str:
        """The group of the op in the tables: the op, the kernel and the role of the first source."""
        return f"{self.op} {self.kernel} {role(self.names)}"

    @property
    def is_hmx(self) -> bool:
        """True when the kernel of the op uses the HMX."""
        return self.kernel.startswith("hmx")

    @functools.cached_property
    def model(self) -> optable.Op:
        """The op in the form of tools/prof/optable.py, for its byte and FLOP model."""
        srcs, dst = optable._split(self.names)
        sd, dd = optable._split(self.dims)
        st, dt = optable._split(self.types)
        return optable.Op(self.op, srcs, dst, [optable._dims(x) for x in sd], optable._dims(dd), st, dt,
                          self.kernel, self.usec, self.cycles, self.start)

    @functools.cached_property
    def weight_bytes(self) -> float:
        """The bytes of the weight sources of the op (the names with ".weight"). O(sources)."""
        names, _ = split_list(self.names)
        dims, _ = split_list(self.dims)
        types, _ = split_list(self.types)
        total = 0.0
        # A field with fewer items than the names gives no bytes for the items after its end.
        for n, d, t in zip(names, dims, types, strict=False):
            if n.strip().endswith(".weight"):
                total += math.prod(dims_of(d)) * TYPE_BYTES.get(t.strip(), 4.0)
        return total


@dataclass
class BatchLine:
    """One OPBATCH line and its ops."""
    n_ops: int
    usec: int
    cycles: int
    start: int
    mhz: float
    ops: list[OpLine] = field(default_factory=list)

    @property
    def op_cycles(self) -> int:
        """The sum of the cycles of the ops of the batch."""
        return sum(o.cycles for o in self.ops)


@dataclass
class GraphData:
    """The DSP batches and the ops of one graph."""
    batches: list[BatchLine] = field(default_factory=list)

    @property
    def ops(self) -> list[OpLine]:
        """The ops of the graph in log order."""
        return [o for b in self.batches for o in b.ops]

    def keyed(self) -> list[tuple[tuple[str, int], OpLine]]:
        """The ops with a key that is the same in each graph of one kind: the group and the index of the op
        in its group (for a layer op, the layer order)."""
        seen: Counter = Counter()
        out = []
        for op in self.ops:
            out.append(((op.group, seen[op.group]), op))
            seen[op.group] += 1
        return out


def parse_graphs(log: str) -> list[GraphData]:
    """Divide the profile of one run into graphs. A graph starts at the op GRAPH_START. An OPBATCH line comes
    before the ops of its DSP batch, thus a batch goes to the graph of its first op. O(lines)."""
    graphs: list[GraphData] = []
    pending: BatchLine | None = None
    current: BatchLine | None = None
    for line in log.splitlines():
        if "profile-op " not in line:
            continue
        m = OP_RE.search(line)
        if m is None:
            continue
        if m["op"] == "OPBATCH":
            n_ops = int(m["dims"].split()[1]) if m["dims"].startswith("n-ops ") else 0
            pending = BatchLine(n_ops, int(m["usec"]), int(m["cycles"]), int(m["start"]), float(m["mhz"]))
            continue
        kp = m["kp"].strip()
        op = OpLine(m["op"], m["names"], m["dims"], m["types"], kp.split()[0] if kp and kp != "----" else "-",
                    int(m["usec"]), int(m["cycles"]), int(m["start"]),
                    tuple(int(x) for x in m["pmu"].split(",")) if m["pmu"] else None)
        if not graphs or (m["names"].rstrip().endswith(GRAPH_START) and op.op.startswith("RMS_NORM")):
            graphs.append(GraphData())
            # A graph that starts inside a batch gets a batch line of its own with no batch time.
            current = None
        if pending is not None:
            graphs[-1].batches.append(pending)
            current, pending = pending, None
        if current is None:
            current = BatchLine(0, 0, 0, 0, 0.0)
            graphs[-1].batches.append(current)
        current.ops.append(op)
    return graphs


@dataclass
class Agg:
    """The totals of one group of ops in one kind of graph of one run."""
    calls: int = 0
    usec: int = 0
    cycles: int = 0
    counts: list[int] = field(default_factory=lambda: [0] * 8)
    weight_bytes: float = 0.0
    nbytes: float = 0.0
    flops: float = 0.0

    def add(self, op: OpLine) -> None:
        """Add one op to the totals."""
        self.calls += 1
        self.usec += op.usec
        self.cycles += op.cycles
        self.weight_bytes += op.weight_bytes
        self.nbytes += op.model.nbytes
        self.flops += op.model.flops
        if op.pmu:
            for i, v in enumerate(op.pmu[:8]):
                self.counts[i] += v


ALL = "all ops"


@dataclass
class RunData:
    """The parsed files of one run. ok is False when the run did not run or failed. flags names each
    defect that removes the run from the tables (a failed gate or tool, a thermal status above 0 after the
    run, a failure line in the log, a wrong profile mode or event list, a wrong graph count). notes names a
    change of the CPU caps: the caps change the host part of a decode step, not the DSP cycles of an op,
    thus a note removes the run only with --strict."""
    run: Run
    ok: bool
    flags: list[str]
    notes: list[str]
    caps: str
    nsp: tuple[float | None, float | None]
    events: tuple[int, ...]
    graphs: list[GraphData]
    kinds: dict[str, dict[str, Agg]]
    n_kind: dict[str, int]
    prefill_ms: float | None
    steps_median: float | None
    log_path: Path

    def usable(self, strict: bool) -> bool:
        """True when the tables use the run."""
        return self.ok and not self.flags and not (strict and self.notes)

    def index(self, name: str) -> int | None:
        """The counter slot of one event in this run, or None."""
        eid = EVENTS.get(name)
        return self.events.index(eid) if eid is not None and eid in self.events else None


def read_run(root: Path, run: Run) -> RunData:
    """Read the gate file, the stdout and the stderr of one run. O(size of the files)."""
    gate_path, out_path, log_path = (root / f"{run.name}{s}" for s in ("-gate.txt", ".out", ".log"))
    gate = gate_path.read_text(errors="replace") if gate_path.exists() else ""
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    rc = re.search(r"^rc=(\d+)", gate, re.M)
    flags: list[str] = []
    notes: list[str] = []
    ok = "gate: OK" in gate and rc is not None and rc.group(1) == "0"
    if not gate:
        flags.append("no gate file")
    elif "gate: OK" not in gate:
        flags.append("gate stopped the run")
    elif not ok:
        flags.append(f"exit code {rc.group(1) if rc else '?'}")
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        notes.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    if after and after.group(1) not in ("", "0"):
        flags.append(f"thermal {after.group(1)} after the run")
    nb = BEFORE_RE.search(gate)
    nsp = (int(nb.group(1)) / 1000 if nb and nb.group(1) else None,
           int(after.group(6)) / 1000 if after and after.group(6) else None)
    out = out_path.read_text(errors="replace") if out_path.exists() else ""
    events: tuple[int, ...] = ()
    graphs: list[GraphData] = []
    kinds: dict[str, dict[str, Agg]] = {}
    n_kind: dict[str, int] = {}
    if ok and run.mode != 3:
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        mode = MODE_RE.search(log)
        want = [EVENTS[n] for n in run.events]
        got_mode = int(mode.group(1)) if mode else None
        got = [int(x, 0) for x in mode.group(2).replace(" ", "").split(",") if x] if mode else []
        if got_mode != run.mode:
            flags.append(f"profile mode {got_mode}, {run.mode} expected")
        if run.mode == 2:
            if want and got != want:
                flags.append(f"events {got}, {want} expected")
            events = tuple(got)
        if FAILURE_RE.search(log):
            flags.append("the log has a failure line")
        graphs = parse_graphs(log)
        if len(graphs) != run.n_graphs:
            flags.append(f"{len(graphs)} graphs, {run.n_graphs} expected")
        else:
            for kind, idx in kind_graphs(run).items():
                groups: dict[str, Agg] = defaultdict(Agg)
                for gi in idx:
                    for op in graphs[gi].ops:
                        groups[op.group].add(op)
                        groups[ALL].add(op)
                kinds[kind] = dict(groups)
                n_kind[kind] = len(idx)
    pm = PREFILL_RE.search(out)
    sm = STEPS_RE.search(out)
    return RunData(run, ok, flags, notes, caps, nsp, events, graphs, kinds, n_kind,
                   float(pm.group(1)) if pm else None, float(sm.group(2)) if sm else None, log_path)


# ---- The views of the counters ----

class Store:
    """The runs of the stage and the accessors that combine their counters. A value per cycle of one group
    comes from one run: the counts of the event and the cycles of the same ops in the same run. A metric of
    events of different runs combines their values per cycle."""

    def __init__(self, runs: dict[str, RunData], use: Callable[[RunData], bool]) -> None:
        """Keep the usable runs in the order of RUNS."""
        self.runs = [runs[r.key] for r in RUNS if r.key in runs and use(runs[r.key])]

    def source(self, name: str, prefer: str | None = None) -> RunData | None:
        """The run that gives an event: the preferred run when it has the event, else the first run."""
        cands = [r for r in self.runs if r.index(name) is not None]
        for r in cands:
            if r.run.key == prefer:
                return r
        return cands[0] if cands else None

    def count(self, kind: str, group: str, name: str, prefer: str | None = None) -> tuple[float, int, int] | None:
        """The count of one event for one group, and the cycles and microseconds of the group in that run."""
        r = self.source(name, prefer)
        if r is None:
            return None
        agg = r.kinds.get(kind, {}).get(group)
        if agg is None or agg.cycles == 0:
            return None
        return float(agg.counts[r.index(name)]), agg.cycles, agg.usec

    def per_call(self, kind: str, group: str, name: str, prefer: str | None = None) -> float | None:
        """The count of one event for each call of the group (for a work quantity such as bytes or MACs)."""
        r = self.source(name, prefer)
        agg = r.kinds.get(kind, {}).get(group) if r else None
        if agg is None or agg.calls == 0:
            return None
        return agg.counts[r.index(name)] / agg.calls  # type: ignore[union-attr,index]

    def rate(self, kind: str, group: str, name: str, prefer: str | None = None) -> float | None:
        """The count of one event for each cycle of the group."""
        c = self.count(kind, group, name, prefer)
        return None if c is None else c[0] / c[1]

    def rates(self, kind: str, group: str, names: list[str], prefer: str | None = None) -> float | None:
        """The sum of the values per cycle of some events, or None when one of them has no value."""
        vals = [self.rate(kind, group, n, prefer) for n in names]
        return None if any(v is None for v in vals) else sum(vals)  # type: ignore[arg-type]

    def ref_run(self, kind: str) -> RunData | None:
        """The first usable reference run that has the kind, else the first usable run that has it."""
        for r in sorted(self.runs, key=lambda r: r.run.key not in REF_KEYS):
            if kind in r.kinds:
                return r
        return None

    def ref(self, kind: str, group: str) -> Agg | None:
        """The totals of one group in the reference run of the kind."""
        r = self.ref_run(kind)
        return r.kinds[kind].get(group) if r else None

    def per_graph(self, kind: str) -> int:
        """The graph count of the kind in the reference run, which divides its totals."""
        r = self.ref_run(kind)
        return r.n_kind.get(kind, 1) if r else 1


def pct(x: float | None) -> str:
    """A share in percent with one decimal, or a dash."""
    return f"{100 * x:6.1f}" if x is not None else f"{'-':>6s}"


def num(x: float | None, fmt: str = "6.2f") -> str:
    """A number in one format, or a dash of the same width."""
    if x is None:
        width = int(fmt.split(".")[0]) if fmt[0].isdigit() else 6
        return f"{'-':>{width}s}"
    return format(x, fmt)


def ratio(a: float | None, b: float | None) -> float | None:
    """a / b, or None when one is None or b is 0."""
    return None if a is None or b in (None, 0) else a / b


def axi_bytes(store: Store, kind: str, group: str, even: bool, write: bool) -> float | None:
    """The AXI bytes of one group with the formula of itrace_dsp_events_derived_pmu.h: the line requests
    times their size plus 8 bytes for each other request. The counts come from another run than the
    reference, thus the bytes are the bytes per call of that run times the calls of the reference. A
    stall event adds cycles and no bytes, thus a value per cycle is not correct for bytes.

    The primary counters (AXI_READ_REQUEST and the line counters) count all reads of the DSP: on the
    phone, AXI_READ_REQUEST_EVEN is 50.0 % of AXI_READ_REQUEST for each op, thus the even counters count
    a subset (the even interleave), not a second master."""
    if write:
        names = ["AXI_WRITE_REQUEST", "AXI_LINE32_WRITE_REQUEST", "AXI_LINE64_WRITE_REQUEST",
                 "AXI_LINE128_WRITE_REQUEST", "AXI_LINE256_WRITE_REQUEST"]
        prefer = "bw2"
    else:
        names = ["AXI_READ_REQUEST", "AXI_LINE32_READ_REQUEST", "AXI_LINE64_READ_REQUEST",
                 "AXI_LINE128_READ_REQUEST", "AXI_LINE256_READ_REQUEST"]
        prefer = "bw1"
        if even:
            names = [n + "_EVEN" for n in names]
            prefer = "bw-even"
    vals = [store.per_call(kind, group, n, prefer) for n in names]
    agg = store.ref(kind, group)
    if any(v is None for v in vals) or agg is None:
        return None
    anyr, l32, l64, l128, l256 = vals  # type: ignore[misc]
    per_call = l32 * 32 + l64 * 64 + l128 * 128 + l256 * 256 + max(anyr - l32 - l64 - l128 - l256, 0.0) * 8
    return per_call * agg.calls


@dataclass(frozen=True)
class Col:
    """One column of a table: its label, its width and the function that gives the text of a cell."""
    label: str
    width: int
    cell: Callable[[Store, str, str], str]


def share(name: str, prefer: str | None = None) -> Callable[[Store, str, str], str]:
    """A cell: the count of one event per cycle of the group, in percent."""
    return lambda s, k, g: pct(s.rate(k, g, name, prefer))


def ref_cell(fn: Callable[[Agg, int], float | None], fmt: str) -> Callable[[Store, str, str], str]:
    """A cell from the totals of the group in the reference run and the graph count of the kind."""
    def cell(s: Store, k: str, g: str) -> str:
        agg = s.ref(k, g)
        return num(fn(agg, s.per_graph(k)) if agg else None, fmt)
    return cell


# The MAC cycles of the HMX. Most of them count as HMXMAC_FLT_PARTIAL on the phone, HMXMAC_FLT counts about
# 2 % of them, and the fixed-point pair counts 0 (the f16 kernels). One count is 8 x 32 x 32 MACs (16384
# FLOPs): the FLOPs of an op over its counts give exactly 16384 for each weight matmul.
MAC_BUSY = ["HMXMAC_FXP_PARTIAL", "HMXMAC_FLT_PARTIAL", "HMXMAC_FXP", "HMXMAC_FLT"]
# The stall terms of HMXMAC_UTILIZATION of the simulator. The formula has HMXMAC_DRAIN_PARTIAL two times.
MAC_STALL = ["HMXMAC_ACT_OUTSTANDING", "HMXMAC_WGT_OUTSTANDING", "HMXMAC_MULT_DROP", "HMXMAC_POWER_OVER",
             "HMXMAC_DRAIN_PARTIAL", "HMXMAC_DRAIN_PARTIAL", "HMXMAC_DRAIN"]
FLOP_PER_MAC = 16384


def mac_busy(s: Store, k: str, g: str) -> float | None:
    """The MAC cycles of the HMX per op cycle (the four MAC events of the runs hmx-mac-a and hmx-mac-b)."""
    return s.rates(k, g, MAC_BUSY, "hmx-mac-a")


def mac_util(s: Store, k: str, g: str) -> float | None:
    """The HMXMAC_UTILIZATION of the simulator: the MAC cycles over the MAC cycles plus the MAC stalls."""
    busy, stall = mac_busy(s, k, g), s.rates(k, g, MAC_STALL, "hmx-mac-a")
    return None if busy is None or stall is None else ratio(busy, busy + stall)


def mac_share(name: str) -> Callable[[Store, str, str], str]:
    """A cell: one MAC stall over the MAC cycles plus the MAC stalls of the simulator formula, in percent."""
    def cell(s: Store, k: str, g: str) -> str:
        busy, stall = mac_busy(s, k, g), s.rates(k, g, MAC_STALL, "hmx-mac-a")
        part = s.rate(k, g, name, "hmx-mac-a")
        return pct(None if busy is None or stall is None else ratio(part, busy + stall))
    return cell


def same_run(num_name: str, den_name: str, prefer: str) -> Callable[[Store, str, str], float | None]:
    """A value: one event over another event of the same run, for one group."""
    def value(s: Store, k: str, g: str) -> float | None:
        a, b = s.count(k, g, num_name, prefer), s.count(k, g, den_name, prefer)
        if a is None or b is None or s.source(num_name, prefer) is not s.source(den_name, prefer):
            return None
        return ratio(a[0], b[0])
    return value


def hmx_mhz(s: Store, k: str, g: str) -> str:
    """A cell: the HMX clock while the HMX is active, in MHz: the op clock times HMX_CLK over HMX_ACTIVE
    of the run hmx-mac-a. This reads HMX_CLK as the clock cycles of the HMX (assumed)."""
    r = same_run("HMX_CLK", "HMX_ACTIVE", "hmx-mac-a")(s, k, g)
    c = s.count(k, g, "HMX_CLK", "hmx-mac-a")
    return num(r * c[1] / c[2] if r is not None and c and c[2] else None, "6.0f")


def busy_per_clk(s: Store, k: str, g: str) -> str:
    """A cell: the MAC cycles over HMX_CLK, both of the run hmx-mac-a (the FXP pair counts 0), in percent."""
    num_ = s.rates(k, g, ["HMXMAC_FLT", "HMXMAC_FLT_PARTIAL"], "hmx-mac-a")
    return pct(ratio(num_, s.rate(k, g, "HMX_CLK", "hmx-mac-a")))


def flop_per_mac(s: Store, k: str, g: str) -> str:
    """A cell: the dense FLOPs of one call over the MAC counts of one call (16384 when the HMX does all
    the dense FLOPs and no other MACs)."""
    c = s.per_call(k, g, "HMXMAC_FLT", "hmx-mac-a")
    cp = s.per_call(k, g, "HMXMAC_FLT_PARTIAL", "hmx-mac-a")
    agg = s.ref(k, g)
    if c is None or cp is None or agg is None or c + cp == 0 or agg.flops == 0:
        return num(None, "7.0f")
    return num(agg.flops / agg.calls / (c + cp), "7.0f")


def hvx_denominator(s: Store, k: str, g: str) -> float | None:
    """The denominator of HVX_UTILIZATION of the simulator: the HVX packets plus the HVX stall reasons."""
    return s.rates(k, g, ["HVX_REG_ORDER", "HVX_ACC_ORDER", "HVX_LD_L2_OUTSTANDING", "HVX_ST_L2_OUTSTANDING",
                          "HVX_VTCM_OUTSTANDING", "HVX_SCATGATH_FULL", "HVX_SCATGATH_IN_FULL", "HVX_PKT_PARTIAL",
                          "HVX_VOLTAGE_UNDER", "HVX_PKT"], "hvx")


def hvx_share(name: str) -> Callable[[Store, str, str], str]:
    """A cell: one HVX event over the denominator of HVX_UTILIZATION, in percent."""
    return lambda s, k, g: pct(ratio(s.rate(k, g, name, "hvx"), hvx_denominator(s, k, g)))


def per_pkt(name: str) -> Callable[[Store, str, str], str]:
    """A cell: one pipe event per HVX_PKT count (the resource utilization of the simulator)."""
    return lambda s, k, g: num(ratio(s.rate(k, g, name), s.rate(k, g, "HVX_PKT", "hvx")), "5.2f")


def pkt_per_active(s: Store, k: str, g: str) -> str:
    """A cell: the HVX packets per HVX active cycle (HVX_PKT counts 2 for each packet in 128-byte mode)."""
    active = s.rate(k, g, "HVX_ACTIVE", "hvx")
    return num(ratio(s.rate(k, g, "HVX_PKT", "hvx"), 2 * active if active else None), "6.2f")


def running_mean(prefix: str) -> Callable[[Store, str, str], str]:
    """A cell: the mean number of HVX contexts over the cycles with at least one of them."""
    def cell(s: Store, k: str, g: str) -> str:
        vals = [s.rate(k, g, f"CYCLES_{n}_{prefix}") for n in range(1, 7)]
        if any(v is None for v in vals) or sum(vals) == 0:  # type: ignore[arg-type]
            return num(None, "5.2f")
        return num(sum(n * v for n, v in zip(range(1, 7), vals, strict=True)) / sum(vals), "5.2f")  # type: ignore[misc]
    return cell


def running_any(prefix: str) -> Callable[[Store, str, str], str]:
    """A cell: the share of the op cycles with at least one HVX context, in percent."""
    return lambda s, k, g: pct(s.rates(k, g, [f"CYCLES_{n}_{prefix}" for n in range(1, 7)]))


def per_kcycle(name: str, prefer: str) -> Callable[[Store, str, str], str]:
    """A cell: the count of one event per 1000 op cycles."""
    return lambda s, k, g: num((lambda v: v * 1000 if v is not None else None)(s.rate(k, g, name, prefer)), "6.2f")


def gbps(fn: Callable[[Store, str, str], float | None]) -> Callable[[Store, str, str], str]:
    """A cell: bytes of the group over the microseconds of the reference run, in GB/s."""
    def cell(s: Store, k: str, g: str) -> str:
        agg = s.ref(k, g)
        b = fn(s, k, g)
        return num(b / agg.usec / 1000 if b is not None and agg and agg.usec else None, "6.1f")
    return cell


def rw_over_model(s: Store, k: str, g: str) -> str:
    """A cell: the AXI read and write bytes over the bytes of the op model (optable.py), where the model
    gives 1 GB/s or more."""
    agg = s.ref(k, g)
    rd, wr = axi_bytes(s, k, g, False, False), axi_bytes(s, k, g, False, True)
    if agg is None or rd is None or agg.nbytes < 1000 * agg.usec:
        return num(None, "5.2f")
    return num((rd + (wr or 0.0)) / agg.nbytes, "5.2f")


def even_share(s: Store, k: str, g: str) -> str:
    """A cell: AXI_READ_REQUEST_EVEN over AXI_READ_REQUEST of the run bw-even, in percent."""
    return pct(same_run("AXI_READ_REQUEST_EVEN", "AXI_READ_REQUEST", "bw-even")(s, k, g))


def bytes_per_bypass(s: Store, k: str, g: str) -> str:
    """A cell: the AXI read bytes over L2_UDMA_BYPASS_RD, both of the run bw1."""
    names = ["AXI_READ_REQUEST", "AXI_LINE32_READ_REQUEST", "AXI_LINE64_READ_REQUEST", "AXI_LINE128_READ_REQUEST",
             "AXI_LINE256_READ_REQUEST", "L2_UDMA_BYPASS_RD"]
    c = [s.count(k, g, n, "bw1") for n in names]
    if any(x is None for x in c) or not c[5][0] or any(s.source(n, "bw1") is not s.source(names[0], "bw1")
                                                       for n in names):
        return num(None, "6.0f")
    anyr, l32, l64, l128, l256, byp = (x[0] for x in c)  # type: ignore[index]
    b = l32 * 32 + l64 * 64 + l128 * 128 + l256 * 256 + max(anyr - l32 - l64 - l128 - l256, 0.0) * 8
    return num(b / byp, "6.0f")


def hmx_vtcm(names: list[str]) -> Callable[[Store, str, str], str]:
    """A cell: the HMX VTCM bytes per op cycle, 128 bytes for each access."""
    return lambda s, k, g: num((lambda v: v * 128 if v is not None else None)(s.rates(k, g, names, "hmx-rd")), "6.0f")


TABLES: dict[str, tuple[str, list[Col]]] = {
    "time": ("the time of each group in the reference run, and the model rates", [
        Col("ms", 8, ref_cell(lambda a, n: a.usec / 1000 / n, "8.2f")),
        Col("calls", 6, ref_cell(lambda a, n: a.calls / n, "6.0f")),
        Col("B GB/s", 7, ref_cell(lambda a, n: ratio(a.nbytes, a.usec * 1000), "7.1f")),
        Col("W GB/s", 7, ref_cell(lambda a, n: ratio(a.weight_bytes, a.usec * 1000), "7.1f")),
        Col("TFLOPS", 7, ref_cell(lambda a, n: ratio(a.flops, a.usec * 1e6), "7.2f")),
        Col("MHz", 6, ref_cell(lambda a, n: ratio(a.cycles, a.usec), "6.0f")),
    ]),
    "topdown": ("the 11 top-down stall reasons and the commit cycles, in % of the op cycles (sum = clusters)", [
        Col("idle", 6, share("THREAD_IDLE_PVIEW_CYCLES", "topdown-a")),
        Col("lock", 6, share("ARCH_LOCK_PVIEW_CYCLES", "topdown-a")),
        Col("redir", 6, share("REDIRECT_PVIEW_CYCLES", "topdown-a")),
        Col("iq-0", 6, share("IU_NO_PKT_PVIEW_CYCLES", "topdown-a")),
        Col("dmiss", 6, share("DU_CACHE_MISS_PVIEW_CYCLES", "topdown-a")),
        Col("du-oth", 6, share("DU_BUSY_OTHER_PVIEW_CYCLES", "topdown-a")),
        Col("cu", 6, share("CU_BUSY_PVIEW_CYCLES", "topdown-a")),
        Col("du-cf", 6, share("DU_CONFLICT_PVIEW_CYCLES", "topdown-a")),
        Col("copro", 6, share("COPROC_BUSY_PVIEW_CYCLES", "topdown-b")),
        Col("uncac", 6, share("DU_UNCACHED_PVIEW_CYCLES", "topdown-b")),
        Col("sys", 6, share("SYSTEM_BUSY_PVIEW_CYCLES", "topdown-b")),
        Col("commit", 6, lambda s, k, g: pct(s.rates(k, g, [f"CYCLES_{n}_PACKET_COMMITTED" for n in range(1, 5)]))),
        Col("sum", 6, lambda s, k, g: pct(s.rates(k, g, [
            "THREAD_IDLE_PVIEW_CYCLES", "ARCH_LOCK_PVIEW_CYCLES", "REDIRECT_PVIEW_CYCLES", "IU_NO_PKT_PVIEW_CYCLES",
            "DU_CACHE_MISS_PVIEW_CYCLES", "DU_BUSY_OTHER_PVIEW_CYCLES", "CU_BUSY_PVIEW_CYCLES",
            "DU_CONFLICT_PVIEW_CYCLES", "COPROC_BUSY_PVIEW_CYCLES", "DU_UNCACHED_PVIEW_CYCLES",
            "SYSTEM_BUSY_PVIEW_CYCLES"] + [f"CYCLES_{n}_PACKET_COMMITTED" for n in range(1, 5)]))),
        Col("i/pkt", 5, lambda s, k, g: num(ratio(s.rate(k, g, "COMMITTED_INSTS", "stalls"),
                                                  s.rate(k, g, "COMMITTED_PKT_ANY", "stalls")), "5.2f")),
    ]),
    "engines": ("which engine is busy, in % of the op cycles (one run: engines)", [
        Col("HMXact", 6, share("HMX_ACTIVE", "engines")),
        Col("MXq-0", 6, share("HMX_MXFIFO_EMPTY", "engines")),
        Col("HVXact", 6, share("HVX_ACTIVE", "engines")),
        Col("DMAact", 6, share("UDMA_ACTIVE", "engines")),
        Col("copro", 6, share("COPROC_BUSY_PVIEW_CYCLES", "engines")),
        Col("sys", 6, share("SYSTEM_BUSY_PVIEW_CYCLES", "engines")),
        Col("idle", 6, share("THREAD_IDLE_PVIEW_CYCLES", "engines")),
    ]),
    "hmx": ("the HMX: clock, activity, MAC cycles and stalls, instruction FIFO, VTCM traffic, power", [
        Col("active", 6, share("HMX_ACTIVE", "hmx-mac-a")),
        Col("clk", 6, share("HMX_CLK", "hmx-mac-a")),
        Col("HMXMHz", 6, hmx_mhz),
        Col("MAC", 6, lambda s, k, g: pct(mac_busy(s, k, g))),
        Col("MAC/ck", 6, busy_per_clk),
        Col("MACutl", 6, lambda s, k, g: pct(mac_util(s, k, g))),
        Col("act-w", 6, mac_share("HMXMAC_ACT_OUTSTANDING")),
        Col("wgt-w", 6, mac_share("HMXMAC_WGT_OUTSTANDING")),
        Col("m-pwr", 6, mac_share("HMXMAC_POWER_OVER")),
        Col("MXq-F", 6, share("HMX_MXFIFO_FULL", "hmx-mac-b")),
        Col("MXq-0", 6, share("HMX_MXFIFO_EMPTY", "hmx-mac-b")),
        Col("macF", 6, share("HMX_MAC_FULL", "hmx-cvt")),
        Col("FLOP/c", 7, flop_per_mac),
        Col("rdB/c", 6, hmx_vtcm(["HMXRDACT_ACT", "HMXRDWGT_WGT", "HMXRDWGT_SCALE", "HMXRDACT_PARTIAL",
                                  "HMXRDWGT_PARTIAL"])),
        Col("plim", 6, share("HMX_POWERLIMITS_OVER", "power-a")),
        Col("tlmh", 6, share("HMX_LIMITS_THROTTLE_TLMH", "power-a")),
        Col("dpm/kc", 6, per_kcycle("HMX_DPM_AVG_COMPRESSED", "power-a")),
    ]),
    "hvx": ("the HVX: activity, utilization and stalls (simulator formula), the pipes per packet", [
        Col("active", 6, share("HVX_ACTIVE", "hvx")),
        Col("pkt/ac", 6, pkt_per_active),
        Col("util", 6, hvx_share("HVX_PKT")),
        Col("reg", 6, hvx_share("HVX_REG_ORDER")),
        Col("acc", 6, hvx_share("HVX_ACC_ORDER")),
        Col("ld-L2", 6, hvx_share("HVX_LD_L2_OUTSTANDING")),
        Col("st-L2", 6, hvx_share("HVX_ST_L2_OUTSTANDING")),
        Col("vtcm", 6, hvx_share("HVX_VTCM_OUTSTANDING")),
        Col("multi", 6, hvx_share("HVX_PKT_PARTIAL")),
        Col("vqF", 6, share("HVX_CORE_VFIFO_FULL_STALL", "hvx2")),
        Col("vq-0", 6, share("HVX_VFIFO_EMPTY", "hvx2")),
        Col("alu", 5, per_pkt("HVXPIPE_ALU")),
        Col("mpy", 5, per_pkt("HVXPIPE_MPY")),
        Col("shift", 5, per_pkt("HVXPIPE_SHIFT")),
        Col("perm", 5, per_pkt("HVXPIPE_PERM")),
        Col("volt", 6, share("HVX_VOLTAGE_UNDER", "power-b")),
    ]),
    "memory": ("DDR and DMA: AXI GB/s, the model GB/s, the DMA waits in % of the op cycles", [
        Col("rd", 6, gbps(lambda s, k, g: axi_bytes(s, k, g, False, False))),
        Col("ev%", 6, even_share),
        Col("wr", 6, gbps(lambda s, k, g: axi_bytes(s, k, g, False, True))),
        Col("B", 6, gbps(lambda s, k, g: s.ref(k, g).nbytes if s.ref(k, g) else None)),
        Col("rw/B", 5, rw_over_model),
        Col("B/byp", 6, bytes_per_bypass),
        Col("DMAact", 6, share("UDMA_ACTIVE", "udma")),
        Col("dmpoll", 6, share("UDMA_DMPOLL_CYCLES", "udma")),
        Col("ddr-rd", 6, share("UDMA_NONCOHERENT_RD_CYCLES", "udma")),
        Col("rbFull", 6, share("UDMA_RD_BUFFER_LEVEL_FULL", "udma")),
        Col("L2miss", 6, per_kcycle("L2_DU_READ_MISS", "dma")),
    ]),
    "threads": ("the threads and the HVX contexts, the thread wait", [
        Col("six-T", 6, share("CYCLES_6_THREAD_RUNNING", "threads")),
        Col("idle", 6, share("THREAD_IDLE_PVIEW_CYCLES", "threads")),
        Col("any-X", 6, running_any("HVX_CONTEXTS_RUNNING")),
        Col("mean-X", 6, running_mean("HVX_CONTEXTS_RUNNING")),
        Col("six-X", 6, share("CYCLES_6_HVX_CONTEXTS_RUNNING", "hvxctx")),
        Col("dpm/kc", 6, per_kcycle("DPM_AVG_COMPRESSED", "power-b")),
    ]),
}

LEGEND = {
    "time": "ms: the time of the group in the reference run (for dec: per token). B GB/s: the bytes of the op model "
            "of optable.py over the time. W GB/s: the bytes of the weight sources over the time. TFLOPS: the dense "
            "FLOPs over the time (FA without the causal mask).",
    "topdown": "idle THREAD_IDLE (a thread is off, waits or pauses), lock ARCH_LOCK, redir REDIRECT, iq-0 IU_NO_PKT, "
               "dmiss DU_CACHE_MISS, du-oth DU_BUSY_OTHER, cu CU_BUSY, du-cf DU_CONFLICT, copro COPROC_BUSY (the "
               "coprocessor queue is full), uncac DU_UNCACHED, sys SYSTEM_BUSY (DMA sync, AXI busy, cache ops), commit "
               "the cycles with 1 to 4 packets committed. i/pkt from the run stalls.",
    "engines": "HMXact HMX_ACTIVE, MXq-0 HMX_MXFIFO_EMPTY, HVXact HVX_ACTIVE, DMAact UDMA_ACTIVE.",
    "hmx": "active HMX_ACTIVE and clk HMX_CLK per op cycle. HMXMHz: the op clock times HMX_CLK over HMX_ACTIVE (the HMX "
           "clock while it is active, if HMX_CLK counts HMX clock cycles). MAC: the MAC cycles (FLT, FLT_PARTIAL, FXP, "
           "FXP_PARTIAL) per op cycle. MAC/ck: the MAC cycles over HMX_CLK. MACutl HMXMAC_UTILIZATION of the simulator. "
           "act-w, wgt-w, m-pwr: that MAC stall over the MAC cycles plus stalls. MXq-F/MXq-0 the HMX instruction FIFO "
           "full/empty, macF HMX_MAC_FULL, per op cycle. FLOP/c: dense FLOPs of a call over its MAC counts. rdB/c: "
           "HMX VTCM read bytes per op cycle. plim, tlmh: HMX power limit and throttle per op cycle. dpm/kc: HMX power "
           "meter per 1000 cycles.",
    "hvx": "active HVX_ACTIVE per op cycle, pkt/ac HVX packets per active cycle (HVX_PKT counts 2 per packet). util "
           "and the stalls: over HVX_PKT plus the 9 HVX stall events (HVX_UTILIZATION of the simulator). vqF "
           "HVX_CORE_VFIFO_FULL_STALL, vq-0 HVX_VFIFO_EMPTY, volt HVX_VOLTAGE_UNDER, per op cycle. alu mpy shift perm: "
           "HVXPIPE events per HVX_PKT count.",
    "memory": "rd, wr: AXI bytes (the formula of itrace_dsp_events_derived_pmu.h) of a call times the calls, over the "
              "time. The primary counters count all reads, ev% is the even interleave share. B: the bytes of the op "
              "model over the time. rw/B: the AXI bytes over the model bytes (only where B is 1 GB/s or more). B/byp: "
              "AXI read bytes per L2_UDMA_BYPASS_RD. dmpoll: a thread polls the DMA. ddr-rd: the DMA waits for a read "
              "that bypasses the caches. rbFull: the DMA read buffer is full. L2miss: L2_DU_READ_MISS per 1000 cycles.",
    "threads": "six-T: cycles with 6 threads not in wait or stop (CYCLES_1..5_THREAD_RUNNING count 0 on the phone). "
               "idle: THREAD_IDLE_PVIEW per op cycle. any-X, mean-X, six-X: the HVX contexts in operation. dpm/kc: "
               "the core power meter per 1000 cycles.",
}


def top_groups(store: Store, kind: str, top: int) -> list[str]:
    """The groups of one kind with the most time in the reference run, then the row of all ops."""
    r = store.ref_run(kind)
    if r is None:
        return []
    groups = r.kinds[kind]
    rows = sorted((g for g in groups if g != ALL), key=lambda g: -groups[g].usec)[:top]
    return rows + [ALL]


def print_table(store: Store, kind: str, name: str, groups: list[str]) -> list[str]:
    """One table of one kind: a row for each group and a column for each metric."""
    title, cols = TABLES[name]
    width = min(52, max((len(g) for g in groups), default=10))
    head = f"  {'group':{width}s}" + "".join(f" {c.label:>{c.width}s}" for c in cols)
    out = [f"{kind}: {title}", head]
    for g in groups:
        out.append(f"  {g[:width]:{width}s}" + "".join(f" {c.cell(store, kind, g):>{c.width}s}" for c in cols))
    out.append(f"  ({LEGEND[name]})")
    return out


# ---- The layer view of the prefill ubatches ----

LAYER_BIN = 4


def layer_ops(r: RunData, gi: int, group: str) -> list[OpLine]:
    """The ops of one group in one graph of one run, in log order (for a layer op: the layer order)."""
    if gi >= len(r.graphs):
        return []
    return [op for op in r.graphs[gi].ops if op.group == group]


def layer_sum(r: RunData | None, gi: int, group: str, lo: int, names: list[str]) -> tuple[list[int], int, int, int]:
    """The counts of some events of one run over the ops lo thru lo + LAYER_BIN - 1 of one group in one graph,
    with the cycles, the microseconds and the op count."""
    if r is None or any(r.index(n) is None for n in names):
        return [], 0, 0, 0
    ops = layer_ops(r, gi, group)[lo:lo + LAYER_BIN]
    counts = [sum(o.pmu[r.index(n)] for o in ops if o.pmu) for n in names]  # type: ignore[index]
    return counts, sum(o.cycles for o in ops), sum(o.usec for o in ops), len(ops)


def layer_table(store: Store, group: str) -> list[str]:
    """The ops of one group layer by layer over the 4 prefill ubatches, in bins of LAYER_BIN layers. The
    question: does an HMX op become slower inside a ubatch at a constant core clock, and which counter
    changes with it. The MAC counts of a call show the work, HMX_CLK over the time shows the HMX clock.
    O(ops)."""
    ref = store.ref_run("p0")
    if ref is None:
        return []
    mac_a = store.source("HMXMAC_FLT_PARTIAL", "hmx-mac-a")
    pwr_a = store.source("HMX_POWERLIMITS_OVER", "power-a")
    mac_b = store.source("HMX_MXFIFO_FULL", "hmx-mac-b")
    eng = store.source("HMX_ACTIVE", "engines")
    out = [f"The layer view of '{group}' over the 4 prefill ubatches (depth 0, 1024, 2048, 3072), {LAYER_BIN} layers "
           "for each row.",
           "  ms/op and MHz: the reference run. MAC M: the MAC counts of one call in millions (run hmx-mac-a). "
           "clkMHz: HMX_CLK over the op time, actMHz: HMX_CLK over the HMX_ACTIVE time (run power-a). MAC/ck: MAC "
           "counts over HMX_CLK (hmx-mac-a). pwr/ck: HMXMAC_POWER_OVER over HMX_CLK. plim, tlmh: per op cycle in %. "
           "dpm/kc: HMX power meter per 1000 cycles. MXq-F: HMX_MXFIFO_FULL per op cycle (hmx-mac-b). HVXact: "
           "HVX_ACTIVE per op cycle (engines).",
           f"  {'ubatch':6s} {'layers':7s} {'ms/op':>6s} {'MHz':>5s} {'MAC M':>6s} {'clkMHz':>6s} {'actMHz':>6s} "
           f"{'MAC/ck':>6s} {'pwr/ck':>6s} {'plim':>6s} {'tlmh':>6s} {'dpm/kc':>6s} {'MXq-F':>6s} {'HVXact':>6s}"]
    for gi in range(ref.run.n_prefill):
        n_ref = len(layer_ops(ref, gi, group))
        for lo in range(0, n_ref, LAYER_BIN):
            _, cyc, us, n = layer_sum(ref, gi, group, lo, [])
            a, a_cyc, _, a_n = layer_sum(mac_a, gi, group, lo, ["HMXMAC_FLT", "HMXMAC_FLT_PARTIAL", "HMX_CLK"])
            p, p_cyc, p_us, _ = layer_sum(pwr_a, gi, group, lo, ["HMX_CLK", "HMX_ACTIVE", "HMXMAC_POWER_OVER",
                                                                "HMX_POWERLIMITS_OVER", "HMX_LIMITS_THROTTLE_TLMH",
                                                                "HMX_DPM_AVG_COMPRESSED"])
            b, b_cyc, _, _ = layer_sum(mac_b, gi, group, lo, ["HMX_MXFIFO_FULL"])
            e, e_cyc, _, _ = layer_sum(eng, gi, group, lo, ["HVX_ACTIVE"])
            mac = (a[0] + a[1]) / a_n / 1e6 if a and a_n else None
            clk_mhz = p[0] / p_us if p and p_us else None
            act_mhz = p[0] / p[1] * p_cyc / p_us if p and p[1] and p_us else None
            out.append(
                f"  {gi + 1:6d} {lo:2d}-{lo + n - 1:<4d} {us / n / 1000:6.2f} {num(ratio(cyc, us), '5.0f')} "
                f"{num(mac, '6.3f')} {num(clk_mhz, '6.0f')} {num(act_mhz, '6.0f')} "
                f"{pct(ratio(a[0] + a[1], a[2]) if a else None)} {pct(ratio(p[2], p[0]) if p else None)} "
                f"{pct(ratio(p[3], p_cyc) if p else None)} {pct(ratio(p[4], p_cyc) if p else None)} "
                f"{num(ratio(p[5] * 1000, p_cyc) if p else None, '6.1f')} {pct(ratio(b[0], b_cyc) if b else None)} "
                f"{pct(ratio(e[0], e_cyc) if e else None)}")
    return out


# ---- The decode stall events ----

def decode_outliers(store: Store, limit: int) -> list[str]:
    """The decode ops that take much more time than the same op in the other decode tokens of the run: more
    than 2 times the median of the op and 0.5 ms more. For each such op, the extra count of each event of
    the run over the extra cycles of the op, thus a value near 1 for a cycle event names the cause of the
    extra time. The line "median" of a run gives the median of each value over its events. O(ops of the
    decode graphs)."""
    out = ["The decode stall events: ops of more than 2 times the median time of the same op in the other decode "
           "tokens, and 0.5 ms more. MHz: the cycles over the time of the op. extra/cyc: the extra count of each "
           "event over the extra cycles of the op."]
    for r in store.runs:
        idx = kind_graphs(r.run)["dec"]
        if len(idx) < 4 or len(r.graphs) != r.run.n_graphs:
            continue
        by_key: dict[tuple[str, int], list[tuple[int, OpLine]]] = defaultdict(list)
        for gi in idx:
            for key, op in r.graphs[gi].keyed():
                by_key[key].append((gi, op))
        events = []
        for key, lst in by_key.items():
            if len(lst) < 4:
                continue
            med = statistics.median(op.usec for _, op in lst)
            for gi, op in lst:
                if op.usec > 2 * med and op.usec - med > 500:
                    events.append((op.usec - med, gi, key, op, med, lst))
        extra = sum(e[0] for e in events) / 1000
        out.append(f"  {r.run.key}: {len(events)} events in {len(idx)} tokens, {extra:.1f} ms extra in total, "
                   f"{extra / len(idx):.2f} ms for each token")
        names = [NAME_OF.get(e, f"0x{e:x}") for e in r.events]
        per_event: dict[str, list[float]] = defaultdict(list)
        for _, gi, _, op, _, lst in events:
            base = [o for g, o in lst if g != gi]
            med_cyc = statistics.median(o.cycles for o in base)
            extra_cyc = op.cycles - med_cyc
            if op.pmu and extra_cyc > 0:
                for j, n in enumerate(names):
                    per_event[n].append((op.pmu[j] - statistics.median(o.pmu[j] for o in base if o.pmu)) / extra_cyc)
        if per_event:
            out.append("    median extra/cyc: " + " ".join(f"{n}={statistics.median(v):.2f}" for n, v in per_event.items()))
        for _, gi, key, op, med, _ in sorted(events, key=lambda e: -e[0])[:limit]:
            out.append(f"    token {gi - r.run.n_prefill + 1:2d} {key[0][:40]:40s} #{key[1]:<2d} {op.usec / 1000:6.2f} ms "
                       f"(median {med / 1000:5.2f}) {op.cycles / op.usec if op.usec else 0:5.0f} MHz")
    return out


# ---- The other tables ----

def uses_hmx(op: OpLine) -> bool:
    """True when the op runs work on the HMX: an HMX kernel, or the chunked delta rule (GATED_DELTA_NET of a
    prefill, gdn-chunk-ops.c, which pushes jobs to the HMX queue)."""
    return op.is_hmx or op.op == "GATED_DELTA_NET"


def hmx_confirm(store: Store) -> list[str]:
    """The confirmation of the HMX ids: the count per cycle on the ops that use the HMX (uses_hmx) against the
    count per cycle on the other ops. O(ops of the runs)."""
    out = ["The HMX ids: counts per 1000 cycles on the ops that use the HMX (HMX kernels, GATED_DELTA_NET) and on "
           "the other ops, for each run that has the id.",
           f"  {'id':6s} {'event':26s} {'run':10s} {'HMX ops':>10s} {'other ops':>10s} verdict"]
    controls = (EVENTS["COPROC_BUSY_PVIEW_CYCLES"], EVENTS["HVX_ACTIVE"], EVENTS["UDMA_ACTIVE"])
    ids = sorted({e for r in store.runs for e in r.events if NAME_OF.get(e, "").startswith("HMX") or e in controls})
    for eid in ids:
        name = NAME_OF.get(eid, f"0x{eid:x}")
        for r in store.runs:
            if eid not in r.events:
                continue
            slot = r.events.index(eid)
            tot = {True: [0, 0], False: [0, 0]}
            for g in r.graphs:
                for op in g.ops:
                    if op.pmu:
                        tot[uses_hmx(op)][0] += op.pmu[slot]
                        tot[uses_hmx(op)][1] += op.cycles
            hmx = tot[True][0] / tot[True][1] if tot[True][1] else 0.0
            oth = tot[False][0] / tot[False][1] if tot[False][1] else 0.0
            if hmx < 1e-5 and oth < 1e-5:
                verdict = "no count"
            elif oth <= 0.02 * hmx:
                verdict = "HMX: counts only on the HMX ops"
            elif hmx >= 2 * oth or oth >= 2 * hmx:
                verdict = "a different rate on the HMX ops"
            else:
                verdict = "the same rate on all ops"
            out.append(f"  0x{eid:03x}  {name:26s} {r.run.key:10s} {1000 * hmx:10.3f} {1000 * oth:10.3f} {verdict}")
    out.append("  (COPROC_BUSY_PVIEW, HVX_ACTIVE and UDMA_ACTIVE are the controls: they count on the other ops too.)")
    return out


def zero_events(store: Store) -> list[str]:
    """The events that count 0 in each op of each run that has them: not implemented on the phone, or no such
    work in the model. O(ops of the runs)."""
    total: dict[int, int] = defaultdict(int)
    for r in store.runs:
        for j, e in enumerate(r.events):
            total[e] += sum(op.pmu[j] for g in r.graphs for op in g.ops if op.pmu)
    zero = sorted(e for e, v in total.items() if v == 0)
    return ["The events with a count of 0 in all ops of all runs: " +
            ", ".join(f"0x{e:03x} {NAME_OF.get(e, '?')}" for e in zero)]


def consistency(store: Store) -> list[str]:
    """The events of more than one run: the value per cycle of all ops of each kind in each run, thus the
    spread between runs shows the noise of the counters and of the clocks."""
    out = ["The events of more than one run, per 1000 op cycles of all ops (the spread between the runs):"]
    by_event: dict[str, list[RunData]] = defaultdict(list)
    for r in store.runs:
        for e in r.events:
            by_event[NAME_OF.get(e, f"0x{e:x}")].append(r)
    for name, runs in sorted(by_event.items()):
        if len(runs) < 2:
            continue
        cells = []
        for kind, _ in KINDS:
            vals = [store.rate(kind, ALL, name, r.run.key) for r in runs]
            vals = [v for v in vals if v is not None]
            if vals:
                cells.append(f"{kind} {1000 * min(vals):.1f}..{1000 * max(vals):.1f}")
        out.append(f"  {name:28s} {len(runs)} runs: " + ", ".join(cells))
    return out


def timing(runs: dict[str, RunData]) -> list[str]:
    """The DSP time of each kind in each run against the median of the reference runs, thus a counter set
    that changes the time shows. O(runs)."""
    def kind_ms(r: RunData, kind: str) -> float | None:
        if len(r.graphs) != r.run.n_graphs:
            return None
        idx = kind_graphs(r.run)[kind]
        return sum(b.usec for gi in idx for b in r.graphs[gi].batches) / 1000 / len(idx) if idx else None

    refs = [runs[k] for k in REF_KEYS if k in runs and runs[k].usable(False)]
    base = {}
    for kind, _ in KINDS:
        vals = [v for v in (kind_ms(r, kind) for r in refs) if v is not None]
        base[kind] = statistics.median(vals) if vals else None
    out = ["DSP time per graph (the sum of the batch times, for dec per token), and the difference to the median of "
           "the usable reference runs:",
           f"  {'run':12s} {'p0 ms':>9s} {'p3 ms':>9s} {'dec ms':>8s} {'p0':>7s} {'p3':>7s} {'dec':>7s} "
           f"{'prefill4096':>11s} {'step med':>8s} {'NSP C':>11s} flags"]
    for run in RUNS:
        r = runs.get(run.key)
        if r is None:
            continue
        vals = {kind: kind_ms(r, kind) for kind, _ in KINDS}
        rel = {k: (f"{100 * (vals[k] / base[k] - 1):+6.1f}%" if vals[k] and base[k] else f"{'-':>7s}") for k in vals}
        nsp = "/".join(f"{t:.0f}" if t is not None else "?" for t in r.nsp)
        out.append(f"  {run.key:12s} {num(vals['p0'], '9.1f')} {num(vals['p3'], '9.1f')} {num(vals['dec'], '8.2f')} "
                   f"{rel['p0']} {rel['p3']} {rel['dec']} {num(r.prefill_ms, '11.1f')} {num(r.steps_median, '8.1f')} "
                   f"{nsp:>11s} {', '.join(r.flags + r.notes)}")
    return out


def decode_batches(store: Store) -> list[str]:
    """The DSP batches of one decode token (question: the transitions): for each batch position of a token
    the op count, the batch time, the op time, the time outside the ops, and the gap before the batch on the
    cycle counter (a lower bound: the counter stops while the DSP sleeps). Then the step time of memprobe
    against the batch time of a token. The medians over the decode tokens of the reference runs."""
    out = ["The DSP batches of one decode token (medians over the tokens of the reference runs):",
           f"  {'batch':5s} {'ops':>5s} {'batch ms':>9s} {'op ms':>8s} {'outside':>8s} {'gap before':>10s}"]
    rows: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    token_ms, steps = [], []
    for r in store.runs:
        if r.run.key not in REF_KEYS or len(r.graphs) != r.run.n_graphs:
            continue
        if r.steps_median is not None:
            steps.append(r.steps_median)
        for gi in kind_graphs(r.run)["dec"]:
            prev = r.graphs[gi - 1].batches[-1]
            token_ms.append(sum(b.usec for b in r.graphs[gi].batches) / 1000)
            for i, b in enumerate(r.graphs[gi].batches):
                mhz = b.mhz or 1.0
                gap = (b.start - (prev.start + prev.cycles)) / mhz / 1000
                rows[i].append((b.n_ops, b.usec / 1000, b.op_cycles / mhz / 1000, gap))
                prev = b
    for i in sorted(rows):
        vals = rows[i]
        med = [statistics.median(v[j] for v in vals) for j in range(4)]
        out.append(f"  {i + 1:5d} {med[0]:5.0f} {med[1]:9.2f} {med[2]:8.2f} {med[1] - med[2]:8.2f} {med[3]:10.3f}")
    if token_ms:
        tm = statistics.median(token_ms)
        line = f"  token: {tm:.2f} ms in the DSP batches"
        if steps:
            sm = statistics.median(steps)
            line += f", memprobe step median {sm:.2f} ms, thus {sm - tm:.2f} ms of the step are outside the DSP batches"
        out.append(line)
    return out


def trace_summary(root: Path, strict: bool, top: int) -> list[str]:
    """The phases of the DSP threads in the ops of the trace run: for each group of ops, the time of each
    phase as a share of the op time, for the HMX thread and for the mean of the 6 HVX threads. An op goes
    into the table only when each thread has all its events in the op (the limit OPTRACE cuts the events of
    a thread in a long batch). O(events)."""
    run = RUN_BY_KEY["trace"]
    rd = read_run(root, run)
    if not rd.usable(strict):
        return [f"trace: no usable run ({', '.join(rd.flags + rd.notes) or 'no files'})"]
    import htp_trace  # type: ignore[import-not-found]
    log = htp_trace.parse_file(rd.log_path)
    n_hvx = 6
    per: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    op_time: dict[str, float] = defaultdict(float)
    n_ops: dict[str, int] = defaultdict(int)
    covered, total = 0, 0
    for session in log.sessions.values():
        for batch in session.batches:
            phases, _, _ = htp_trace.pair_phases(batch)
            cut = math.inf
            last: dict[int, int] = {}
            for e in batch.events:
                last[e.thread] = max(last.get(e.thread, 0), e.abs_cycles)
            for t, cnt in enumerate(batch.evt_cnt or ()):
                if cnt > OPTRACE and t in last:
                    cut = min(cut, last[t])
            for op in batch.ops:
                total += 1
                end = op.abs_cycles + op.cycles
                if end > cut:
                    continue
                covered += 1
                kp = op.kparams.strip()
                g = f"{op.name} {kp.split()[0] if kp and kp != '----' else '-'} {role(op.names)}"
                op_time[g] += op.cycles
                n_ops[g] += 1
                for p in phases:
                    lo, hi = max(p.start_cycles, op.abs_cycles), min(p.end_cycles, end)
                    if hi > lo:
                        who = "hmx" if p.thread == htp_trace.HMX_THREAD else "hvx"
                        per[g][f"{who}:{p.name}"] += (hi - lo) / (1 if who == "hmx" else n_hvx)
    names = sorted({k for d in per.values() for k in d})
    rows = sorted(op_time, key=lambda g: -op_time[g])[:top]
    out = [f"trace: {covered} of {total} ops with all their events. The time of each phase in % of the op time "
           f"(hmx: the HMX thread; hvx: the mean of the {n_hvx} HVX threads). The phases can overlap on one thread.",
           "  " + f"{'group':44s} {'ops':>4s}" + "".join(f" {n[:14]:>14s}" for n in names)]
    for g in rows:
        out.append(f"  {g[:44]:44s} {n_ops[g]:4d}" + "".join(f" {100 * per[g][n] / op_time[g]:14.1f}" for n in names))
    return out


def checks(runs: dict[str, RunData]) -> list[str]:
    """The conditions of the runs: gates, exit codes, caps, heat, the profile modes and the events."""
    got = [runs[r.key] for r in RUNS if r.key in runs]
    out = [f"{MODEL_FILE}: {len(got)} of {len(RUNS) - 1} counter and reference runs have a gate file, "
           f"{sum(r.ok for r in got)} ran with exit code 0, {sum(r.usable(False) for r in got)} have no defect, "
           f"{sum(r.usable(True) for r in got)} also have no change of the caps"]
    caps: Counter = Counter()
    for r in got:
        if r.ok:
            caps[r.caps] += 1
    out.append("  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{c} x{n}" for c, n in caps.items()))
    for i, when in enumerate(("before", "after")):
        temps = [r.nsp[i] for r in got if r.ok and r.nsp[i] is not None]
        if temps:
            out.append(f"  NPU zone temperature {when} the runs: {min(temps):.1f} to {max(temps):.1f} C")
    for r in got:
        if r.flags or r.notes:
            out.append(f"  {r.run.name}: " + ", ".join(r.flags + r.notes))
    return out


LAYER_GROUPS = ("MUL_MAT_NX hmx-tiled ffn_gate.weight", "MUL_MAT+ADD hmx-tiled ffn_down.weight")


def table(root: Path, strict: bool, top: int) -> int:
    """Print the tables of the stage."""
    if not root.is_dir():
        print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
        return 1
    runs = {r.key: read_run(root, r) for r in RUNS if r.mode != 3 and (root / f"{r.name}-gate.txt").exists()}
    store = Store(runs, lambda r: r.usable(strict))
    parts = [checks(runs), timing(runs), decode_batches(store)]
    for kind, text in KINDS:
        groups = top_groups(store, kind, top)
        if not groups:
            continue
        parts.append([f"==== {kind}: {text} ===="])
        for name in TABLES:
            parts.append(print_table(store, kind, name, groups))
    parts += [layer_table(store, g) for g in LAYER_GROUPS]
    parts += [decode_outliers(store, 12), hmx_confirm(store), zero_events(store), consistency(store),
              trace_summary(root, strict, top)]
    for part in parts:
        print("\n".join(part))
        print()
    return 0


def main() -> int:
    """Run the subcommand of the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="make build/pmu/phone from the files of build/bench-kv/phone")
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=STAGE_DIR / "phone-commands.txt")
    sub.add_parser("events", help="print each run with its events and their sources")
    t = sub.add_parser("table", help="print the tables from the pulled logs")
    t.add_argument("--root", type=Path, default=STAGE_DIR / "phone-out")
    t.add_argument("--strict", action="store_true", help="do not use the runs with changed caps")
    t.add_argument("--top", type=int, default=20, help="the number of op groups in each table")
    a = ap.parse_args()
    if a.cmd == "setup":
        return setup()
    if a.cmd == "commands":
        n = write_commands(a.out)
        print(f"{a.out}: {n} lines, {len(RUNS)} runs")
        return 0
    if a.cmd == "events":
        return print_events()
    return table(a.root, a.strict, a.top)


if __name__ == "__main__":
    sys.exit(main())
