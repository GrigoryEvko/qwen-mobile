#!/usr/bin/env python3
"""Convert a profile log of the llama.cpp Hexagon backend into a Perfetto timeline.

The host side of the backend writes one ``profile-op`` line for each op and one
``OPBATCH`` line for each batch when ``GGML_HEXAGON_PROFILE`` is set. Level 3 adds
one ``trace-evt`` line for each phase event of each DSP thread. This module reads
such a log, puts the events on a microsecond timeline, and writes a Chrome trace
event JSON file that https://ui.perfetto.dev opens. The module also prints a
summary table, and it compares two logs class by class.

Procedure on the phone
======================

The package is in ``/data/local/tmp/qwen/htp`` (``bin`` and ``lib``).

1. Set the variables in the adb shell::

       export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
       export GGML_HEXAGON_OPFUSION=1
       export GGML_HEXAGON_PROFILE=3

   The profile levels are: 1 = times, 2 = times plus eight PMU counters, 3 = times
   plus the phase events of each DSP thread. There is no other variable for the
   PMU events. Eight comma-separated event ids in ``GGML_HEXAGON_PROFILE`` select
   level 2 with these events, for example
   ``GGML_HEXAGON_PROFILE=0x3,0x111,0x100,0x105,0x240,0x256,0x7d,0x8c`` (the preset
   list). ``GGML_HEXAGON_OPTRACE=N`` sets the number of trace events for each thread
   at level 3 (the preset value is 256 times ``GGML_HEXAGON_OPBATCH``).

2. Run the model with the debug log and the timestamps (the run must not be
   longer than two minutes, and the phone must be at thermal status 0)::

       timeout -s KILL 120 ./bin/llama-completion -m ../models/MODEL.gguf -dev HTP0 -ngl 99 \\
           -p "The capital of France is" -n 8 -v --log-timestamps --log-file ../logs/prof.log

3. Pull the log::

       adb pull /data/local/tmp/qwen/logs/prof.log

4. Convert the log::

       python tools/trace/htp_trace.py convert prof.log -o prof.json

5. Open https://ui.perfetto.dev, select **Open trace file**, and select ``prof.json``.

Other modes::

    python tools/trace/htp_trace.py summary prof.log --batches 1:
    python tools/trace/htp_trace.py diff before.log after.log --batches 1:

``--batches`` takes a Python slice of the batches of each session, thus ``1:``
removes the first batch (the prompt) and keeps the decode batches.

Tracks in the trace
===================

One process for each session (``HTP0``, ``HTP1`` with two devices). In a process:

- ``batches``: one complete event for each ``OPBATCH`` line
- ``ops``: one complete event for each op, with the names, the dims, the types,
  the strides, the kernel parameters and the PMU counters as arguments
- ``pmu NAME``: one counter track for each PMU counter at level 2, with the delta
  of the counter during each op as the value
- ``dsp t0 (main)`` thru ``dsp t9`` and ``hmx queue``: the phase events of each DSP
  thread at level 3, one slice for each start/stop pair, with the phase name as the
  event name (``DMA``, ``HVX_COMP``, ``HMX_COMP``, ``HVX_W_DEQUANT``, ``HVX_A_QUANT``,
  ``HVX_QK_FA``, ``L2FLUSH`` and the other names of ``htp_event_name``)
- ``dma t0`` thru ``dma t9``: the ``DMA`` phases of each thread on their own track,
  because more than one DMA descriptor can be active on one thread.

Thread 0 is the main thread of the DSP, and it also runs job 0 of each work-queue
task. Threads 1 thru 9 are the work-queue workers. Thread 10 is the HMX queue thread.

Conditions of the timeline
==========================

The DSP reads the 64-bit cycle counter ``c15:14`` (``hex_get_cycles``) at the start
and at the end of a batch, and the ``OPBATCH`` line shows the 64-bit ``start``. For
each op, ``htp_prof_desc`` keeps the low 32 bits only: the ``start`` of an op is the
low 32 bits of the same counter, read immediately before the op. A trace event
keeps the same low 32 bits. The converter obeys these rules:

- The offset of an op in its batch is ``((op.start - batch.start) mod 2^32) / mhz``
  microseconds. The converter adds ``2^32`` cycles each time the low 32 bits become
  smaller than the value of the previous op of the batch. Thus a batch can be longer
  than ``2^32`` cycles (2.0 s at 2112 MHz) if two adjacent ops are less than ``2^32``
  cycles apart. The same rule applies to the trace events of one thread.
- The ``mhz`` of the batch (``cycles / usec``) converts cycles to microseconds. The
  ``usec`` of an op is an integer, thus the ``mhz`` of a small op is not accurate.
  The duration of an op is ``op.cycles / batch.mhz``.
- The cycle counter does not run at the batch clock while the DSP is idle between
  batches. In the sample logs, the host log shows 11 ms between two decode batches
  where the counter shows 0.5 ms. Thus the converter puts each batch on the host
  clock: the timestamp of the ``OPBATCH`` line minus ``usec`` is the batch start, and
  the first batch of the log is at 0. The timestamp of the line is the time of the
  response on the host, thus the position of each batch is after its true position
  by the latency of the response queue.
- Without timestamps in the log (no ``--log-timestamps``), the converter uses the
  cycle counter and the batch ``mhz`` for the distance between batches. Then the
  idle time between batches is too short.

The summary uses the same times: the time of an op is ``cycles / batch.mhz``, the
time of a batch is its ``usec``, and the gap is the batch time minus the sum of the
op times. The gap holds the cache flush at the start and at the end of the batch,
the time to prepare the buffers, and the time between the ops.

PMU counters
============

Level 2 shows eight counter deltas for each op. The event ids come from the log
line ``Profiling mode 2 : pmu-evt [ ... ]``, or from ``DEFAULT_PMU_EVENTS`` when the
log has no such line. ``PMU_EVENTS_V79`` gives the name of each event id. The names
come from the ``_PMU_EVENTS_ENUM_`` text in ``libhexagonissv79.so`` of Hexagon Tools
19.0.07 (Hexagon SDK 6.6.0.0 in the container image
``ghcr.io/snapdragon-toolchain/arm64-android:v0.7``). The itrace headers of the SDK
(``libs/itrace/inc/itrace_dsp_events_pmu.h``) give the same names with descriptions,
but their values are itrace indices, not the PMU event ids. ``PMU_EVENT_OVERRIDES``
holds the entries that are different on v75 (Snapdragon 8 Gen 3) and on v81. The
converter reads the architecture from the ``libggml-htp-vNN.so`` line of the log,
or from ``--arch``. An id without a name keeps its hexadecimal form.

The preset events are:

- ``0x003 COMMITTED_PKT_ANY``: packets that all the threads commit
- ``0x111 HVX_PKT``: packets with HVX instructions (two for each packet in 128-byte mode)
- ``0x100 HVX_ACTIVE``: cycles in which the vector FIFO is not empty
- ``0x105 HVX_VTCM_OUTSTANDING``: stall cycles while a VTCM transaction is not completed
- ``0x240 UDMA_ACTIVE_CYCLES``: cycles with the user DMA not idle
- ``0x256 L2_UDMA_BYPASS_RD``: user DMA reads that bypass the cache hierarchy
- ``0x07d L2_DU_READ_MISS``: L2 read misses from a data unit
- ``0x08c L2_DU_STORE_MISS``: L2 store misses from a data unit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

JsonDict = dict[str, Any]

CYCLE_MASK = (1 << 32) - 1
CYCLE_WRAP = 1 << 32
HTP_MAX_NTHREADS = 10
HMX_THREAD = HTP_MAX_NTHREADS
DEFAULT_ARCH = 79
DEFAULT_PMU_EVENTS: tuple[int, ...] = (0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C)

TID_BATCHES = 1
TID_OPS = 2
TID_THREAD_BASE = 100
TID_DMA_BASE = 200
DMA_PHASE = "DMA"

_TIMESTAMP = r"(?:(?P<min>\d+)\.(?P<sec>\d{2})\.(?P<ms>\d{3})\.(?P<us>\d{3}) [A-Z] )?"
_LINE_RE = re.compile(_TIMESTAMP + r"ggml-hex: (?P<session>\S+) (?P<kind>profile-op|trace-evt) (?P<rest>.*)$")
_TIMING_RE = re.compile(
    r"usec (?P<usec>\d+) cycles (?P<cycles>\d+) start (?P<start>\d+) mhz (?P<mhz>[\d.]+)"
    r"(?: pmu \[(?P<pmu>[\d,]+)\])?\s*$"
)
_TRACE_RE = re.compile(r"(?P<name>\S+): thread (?P<thread>\d+) info (?P<info>\d+) (?P<state>start|stop) (?P<cycles>\d+)\s*$")
_MODE_RE = re.compile(r"ggml-hex: Profiling mode (?P<mode>\d+) : pmu-evt \[ (?P<events>[^\]]*)\]")
_ARCH_RE = re.compile(r"libggml-htp-v(?P<arch>\d+)\.so")

PMU_EVENTS_V79: dict[int, str] = {
    0x001: "COUNTER0_OVERFLOW",
    0x002: "COUNTER2_OVERFLOW",
    0x003: "COMMITTED_PKT_ANY",
    0x004: "COMMITTED_PKT_BSB",
    0x005: "COUNTER4_OVERFLOW",
    0x006: "COUNTER6_OVERFLOW",
    0x007: "COMMITTED_PKT_B2B",
    0x008: "COMMITTED_PKT_SMT",
    0x009: "IU_CREDIT_FAIL",
    0x00a: "CYCLES_5_THREAD_RUNNING",
    0x00b: "CYCLES_6_THREAD_RUNNING",
    0x00c: "COMMITTED_PKT_T0",
    0x00d: "COMMITTED_PKT_T1",
    0x00e: "COMMITTED_PKT_T2",
    0x00f: "COMMITTED_PKT_T3",
    0x010: "COMMITTED_PKT_T4",
    0x011: "COMMITTED_PKT_T5",
    0x012: "ICACHE_DEMAND_MISS",
    0x013: "DCACHE_DEMAND_MISS",
    0x014: "DCACHE_STORE_MISS",
    0x015: "COMMITTED_PKT_T6",
    0x016: "COMMITTED_PKT_T7",
    0x017: "CU_PKT_READY_NOT_DISPATCHED",
    0x018: "COMMITTED_PKT_5_THREAD_RUNNING",
    0x019: "COMMITTED_PKT_6_THREAD_RUNNING",
    0x01a: "COMMITTED_PKT_7_THREAD_RUNNING",
    0x01b: "COMMITTED_PKT_8_THREAD_RUNNING",
    0x01c: "IU_L1S_ACCESS",
    0x01d: "IU_L1S_PREFETCH",
    0x01e: "IU_L1S_AXIS_STALL",
    0x01f: "IU_L1S_NO_GRANT",
    0x020: "ANY_IU_REPLAY",
    0x021: "ANY_DU_REPLAY",
    0x022: "CYCLES_7_THREAD_RUNNING",
    0x023: "ISSUED_PACKETS",
    0x025: "COMMITTED_PKT_1_THREAD_RUNNING",
    0x026: "COMMITTED_PKT_2_THREAD_RUNNING",
    0x027: "COMMITTED_PKT_3_THREAD_RUNNING",
    0x028: "THREAD_LMH_THROTTLE",
    0x029: "LMH_THROTTLE",
    0x02a: "COMMITTED_INSTS",
    0x02b: "COMMITTED_TC1_INSTS",
    0x02c: "COMMITTED_PRIVATE_INSTS",
    0x02d: "GLOBAL_POWERLIMITS_OVER",
    0x02e: "CYCLES_8_THREAD_RUNNING",
    0x02f: "COMMITTED_PKT_4_THREAD_RUNNING",
    0x030: "COMMITTED_LOADS",
    0x031: "COMMITTED_STORES",
    0x032: "COMMITTED_MEMOPS",
    0x033: "COMMITTED_NOPS",
    0x034: "ISSUED_INSTS",
    0x035: "DISPATCHED_PACKETS",
    0x036: "DISPATCHED_INSTS",
    0x037: "COMMITTED_PROGRAM_FLOW_INSTS",
    0x038: "COMMITTED_PKT_CHANGED_FLOW",
    0x039: "COMMITTED_PKT_ENDLOOP",
    0x03a: "PST_USED_P0P1BUSY",
    0x03b: "CYCLES_1_THREAD_RUNNING",
    0x03c: "CYCLES_2_THREAD_RUNNING",
    0x03d: "CYCLES_3_THREAD_RUNNING",
    0x03e: "CYCLES_4_THREAD_RUNNING",
    0x03f: "AXI_LINE128_READ_REQUEST",
    0x040: "AXI_READ_REQUEST",
    0x041: "AXI_LINE32_READ_REQUEST",
    0x042: "AXI_WRITE_REQUEST",
    0x043: "AXI_LINE32_WRITE_REQUEST",
    0x044: "AHB_READ_REQUEST",
    0x045: "AHB_WRITE_REQUEST",
    0x046: "AXI_LINE128_WRITE_REQUEST",
    0x047: "AXI_SLAVE_MULTI_BEAT_ACCESS",
    0x048: "AXI_SLAVE_SINGLE_BEAT_ACCESS",
    0x049: "AXI2_READ_REQUEST",
    0x04a: "AXI2_LINE32_READ_REQUEST",
    0x04b: "AXI2_WRITE_REQUEST",
    0x04c: "AXI2_LINE32_WRITE_REQUEST",
    0x04d: "AXI2_CONGESTION",
    0x04e: "DMA_SLAVE_MULTI_BEAT_ACCESS",
    0x04f: "DMA_SLAVE_SINGLE_BEAT_ACCESS",
    0x050: "COMMITTED_FPS",
    0x051: "REDIRECT_BIMODAL_MISPREDICT",
    0x052: "REDIRECT_TARGET_MISPREDICT",
    0x053: "REDIRECT_LOOP_MISPREDICT",
    0x054: "REDIRECT_MISC",
    0x055: "AXI_LINE256_WRITE_REQUEST",
    0x056: "NUM_PACKET_CRACKED",
    0x057: "MND_throttle",
    0x058: "JTLB_MISS",
    0x059: "ACD_Droop",
    0x05a: "COMMITTED_PKT_RETURN",
    0x05b: "COMMITTED_PKT_INDIRECT_JUMP",
    0x05c: "COMMITTED_BIMODAL_BRANCH_INSTS",
    0x05d: "BRANCH_QUEUE_FULL",
    0x05e: "DU_REQUESTED_BUBBLE_INSERTED",
    0x05f: "VTCM_FIFO_FULL_CYCLES",
    0x060: "DU_STORE_BUFFER_COALESCED",
    0x061: "DU_L1S_LOAD_ACCESS",
    0x062: "ICACHE_ACCESS",
    0x063: "BTB_HIT",
    0x064: "BTB_MISS",
    0x065: "IU_DEMAND_SECONDARY_MISS",
    0x066: "IU_LINE_FROM_HWLOOP",
    0x067: "FAST_FETCH_KILLED",
    0x068: "IU_1_PKT_AVAILABLE_TO_ISSUE",
    0x069: "FETCHED_PACKETS_DROPPED",
    0x06a: "IU_REQUESTS_TO_L2_REPLAYED",
    0x06b: "IU_PREFETCHES_SENT_TO_L2",
    0x06c: "ITLB_MISS",
    0x06d: "IU_2_PKT_AVAILABLE_TO_ISSUE",
    0x06e: "IU_3_PKT_AVAILABLE_TO_ISSUE",
    0x06f: "IU_REQUEST_STALLED",
    0x070: "IU_BIMODAL_L2_ELIGIBLE",
    0x071: "IU_0_PKT_AVAILABLE_TO_ISSUE",
    0x072: "FETCH_2_CYCLE",
    0x073: "FETCH_3_CYCLE",
    0x074: "IU_PREFETCHES_DROPPED",
    0x075: "L2_IU_SECONDARY_MISS",
    0x076: "L2_IU_ACCESS",
    0x077: "L2_IU_MISS",
    0x078: "L2_IU_PREFETCH_ACCESS",
    0x079: "L2_IU_PREFETCH_MISS",
    0x07a: "L2_IU_BRANCH_CACHE_WRITE_REQUEST",
    0x07b: "L2_IU_BRANCH_CACHE_WRITE",
    0x07c: "L2_DU_READ_ACCESS",
    0x07d: "L2_DU_READ_MISS",
    0x07e: "L2FETCH_ACCESS",
    0x07f: "L2FETCH_MISS",
    0x080: "L2_AXI_INTERLEAVE_DROP",
    0x081: "L2_ACCESS",
    0x082: "L2_PIPE_CONFLICT_STALL",
    0x083: "L2_TAG_ARRAY_CONFLICT",
    0x084: "AXI_RD_CONGESTION",
    0x085: "AHB_CONGESTION",
    0x086: "SNOOP_BLOCK",
    0x087: "TCM_DU_ACCESS",
    0x088: "TCM_DU_READ_ACCESS",
    0x089: "TCM_IU_ACCESS",
    0x08a: "L2_CASTOUT",
    0x08b: "L2_DU_STORE_ACCESS",
    0x08c: "L2_DU_STORE_MISS",
    0x08d: "L2_DU_PREFETCH_ACCESS",
    0x08e: "L2_DU_PREFETCH_MISS",
    0x08f: "L2_DU_RETURN_NOT_ACKED",
    0x090: "L2_DU_LOAD_SECONDARY_MISS",
    0x091: "L2FETCH_COMMAND",
    0x092: "L2FETCH_COMMAND_KILLED",
    0x093: "L2FETCH_COMMAND_OVERWRITE",
    0x094: "L2FETCH_ACCESS_CREDIT_FAIL",
    0x095: "AXI_SLAVE_READ_BUSY",
    0x096: "AXI_SLAVE_WRITE_BUSY",
    0x097: "L2_ACCESS_EVEN",
    0x098: "CLADE_HIGH_PRIO_L2_ACCESS",
    0x099: "CLADE_LOW_PRIO_L2_ACCESS",
    0x09a: "CLADE_HIGH_PRIO_L2_MISS",
    0x09b: "CLADE_LOW_PRIO_L2_MISS",
    0x09c: "CLADE_HIGH_PRIO_EXCEPTION",
    0x09d: "CLADE_LOW_PRIO_EXCEPTION",
    0x09e: "AXI2_SLAVE_READ_BUSY",
    0x09f: "AXI2_SLAVE_WRITE_BUSY",
    0x0a0: "ANY_DU_STALL",
    0x0a1: "DU_BANK_CONFLICT_REPLAY",
    0x0a2: "DU_CREDIT_REPLAY",
    0x0a3: "L2_FIFO_FULL_REPLAY",
    0x0a4: "DU_STORE_BUFFER_FULL_REPLAY",
    0x0a5: "DU_STORE_BUFFER_FORCED_DRAIN",
    0x0a6: "DU_SNOOP_CONFLICT_REPLAY",
    0x0a7: "DU_SNOOP_REQUEST",
    0x0a8: "DU_FILL_REPLAY",
    0x0a9: "PST_3STORETYPE_SBCONF_REPLAY",
    0x0aa: "DU_SNOOP_REQUEST_CLEAN_HIT",
    0x0ab: "DU_EVICTIONS_SENT_TO_L2",
    0x0ac: "DU_READ_TO_L2",
    0x0ad: "DU_WRITE_TO_L2",
    0x0ae: "PST_3LDST_L2FIFOCONF_REPLAY",
    0x0af: "DCZERO_COMMITTED",
    0x0b0: "L2ITCM_IU_READ",
    0x0b1: "L2ITCM_DU_READ",
    0x0b2: "L2ITCM_DU_WRITE",
    0x0b3: "DTLB_MISS",
    0x0b4: "L2ITCM_BIMODAL_WRITES_SUCCESS",
    0x0b5: "DU_STORE_BUFFER_ACCESS",
    0x0b6: "STORE_BUFFER_HIT_REPLAY",
    0x0b7: "STORE_BUFFER_FORCE_REPLAY",
    0x0b8: "TAG_WRITE_CONFLICT_REPLAY",
    0x0b9: "SMT_BANK_CONFLICT",
    0x0ba: "PORT_CONFLICT_REPLAY",
    0x0bb: "L2ITCM_BIMODAL_WRITES_DROPPED",
    0x0bc: "L2ITCM_IU_PREFETCH_READ",
    0x0bd: "PAGE_CROSS_REPLAY",
    0x0be: "PST_STORE_SENTON_OTHPORT",
    0x0bf: "DU_DEMAND_SECONDARY_MISS",
    0x0c0: "DU_MISC_REPLAY",
    0x0c1: "GUARDBUF_SETMATCH_CRACKING_REPLAY",
    0x0c2: "DU_STATE_REPLAY",
    0x0c3: "DCFETCH_COMMITTED",
    0x0c4: "DCFETCH_HIT",
    0x0c5: "DCFETCH_MISS",
    0x0c6: "DCACHE_EVICTION_IN_PIPE_REPLAY",
    0x0c7: "STBUF_MATCH_PARTIAL_CRACK_REPLAY",
    0x0c8: "DU_LOAD_UNCACHEABLE",
    0x0c9: "DU_DUAL_LOAD_UNCACHEABLE",
    0x0ca: "DU_STORE_UNCACHEABLE",
    0x0cb: "DU_STORE_RELEASE_CREDIT_STALL",
    0x0cc: "MISS_TO_PREFETCH",
    0x0cd: "AXI_LINE256_READ_REQUEST",
    0x0ce: "AXI_LINE64_READ_REQUEST",
    0x0cf: "AXI_LINE64_WRITE_REQUEST",
    0x0d0: "AXI_WR_CONGESTION",
    0x0d1: "AHB_8_READ_REQUEST",
    0x0d2: "AXI_INCOMPLETE_WRITE_REQUEST",
    0x0d3: "L2FETCH_COMMAND_PAGE_TERMINATION",
    0x0d4: "REQUEST_STALL_WRITE_BUFFER_EXHAUSTION",
    0x0d5: "L2_DU_STORE_COALESCE",
    0x0d6: "L2_STORE_LINK",
    0x0d7: "L2_SCOREBOARD_70_PERCENT_FULL",
    0x0d8: "L2_SCOREBOARD_80_PERCENT_FULL",
    0x0d9: "L2_SCOREBOARD_90_PERCENT_FULL",
    0x0da: "L2_SCOREBOARD_FULL_REJECT",
    0x0db: "L2_DU_RETURN_REPLAYED",
    0x0dc: "L2_EVICTION_BUFFERS_FULL",
    0x0dd: "AHB_MULTI_BEAT_READ_REQUEST",
    0x0de: "L2_CLADE_REQ_LIMIT_STALL",
    0x0df: "L2_DU_LOAD_SECONDARY_MISS_ON_SW_PREFETCH",
    0x0e0: "L2FETCH_DROP",
    0x0e1: "REPLAY_MAXIMUM_FORCE",
    0x0e2: "SCHEDULER_WATCHDOG_FORCE",
    0x0e3: "LIVELOCK_REFETCH",
    0x0e4: "CYCLES_LIVELOCK_WARNING",
    0x0e5: "THREAD_IDLE_PVIEW_CYCLES",
    0x0e6: "ARCH_LOCK_PVIEW_CYCLES",
    0x0e7: "REDIRECT_PVIEW_CYCLES",
    0x0e8: "IU_NO_PKT_PVIEW_CYCLES",
    0x0e9: "DU_CACHE_MISS_PVIEW_CYCLES",
    0x0ea: "DU_BUSY_OTHER_PVIEW_CYCLES",
    0x0eb: "CU_BUSY_PVIEW_CYCLES",
    0x0ec: "DU_CONFLICT_PVIEW_CYCLES",
    0x0ed: "COPROC_BUSY_PVIEW_CYCLES",
    0x0ee: "DU_UNCACHED_PVIEW_CYCLES",
    0x0ef: "SYSTEM_BUSY_PVIEW_CYCLES",
    0x0f0: "APP_REPORTED",
    0x0f1: "AXI_LINE128_READ_REQUEST_EVEN",
    0x0f2: "AXI_READ_REQUEST_EVEN",
    0x0f3: "AXI_LINE32_READ_REQUEST_EVEN",
    0x0f4: "AXI_WRITE_REQUEST_EVEN",
    0x0f5: "AXI_LINE32_WRITE_REQUEST_EVEN",
    0x0f6: "AXI_LINE128_WRITE_REQUEST_EVEN",
    0x0f7: "AXI_RD_CONGESTION_EVEN",
    0x0f8: "AXI_LINE64_READ_REQUEST_EVEN",
    0x0f9: "AXI_LINE64_WRITE_REQUEST_EVEN",
    0x0fa: "AXI_WR_CONGESTION_EVEN",
    0x0fb: "AXI_INCOMPLETE_WRITE_REQUEST_EVEN",
    0x0fc: "AXI_LINE256_READ_REQUEST_EVEN",
    0x0fd: "AXI_LINE256_WRITE_REQUEST_EVEN",
    0x0fe: "CYCLES_3_COPROC_THREADS_ONE_CLUSTER",
    0x0ff: "VOLTAGE_CLOCK_GATING_CYCLES",
    0x100: "HVX_ACTIVE",
    0x101: "HVX_REG_ORDER",
    0x102: "HVX_ACC_ORDER",
    0x103: "HVX_LD_L2_OUTSTANDING",
    0x104: "HVX_ST_L2_OUTSTANDING",
    0x105: "HVX_VTCM_OUTSTANDING",
    0x106: "HVX_SCATGATH_FULL",
    0x107: "HVX_SCATGATH_IN_FULL",
    0x108: "HVX_ST_FULL",
    0x10a: "HVX_VOLTAGE_UNDER",
    0x10b: "HVX_POWER_OVER",
    0x10c: "HVX_PKT_PARTIAL",
    0x111: "HVX_PKT",
    0x112: "HVX_PKT_THREAD",
    0x113: "HVX_CORE_VFIFO_FULL_STALL",
    0x115: "CYCLES_1_HVX_CONTEXTS_RUNNING",
    0x116: "CYCLES_2_HVX_CONTEXTS_RUNNING",
    0x117: "CYCLES_3_HVX_CONTEXTS_RUNNING",
    0x118: "HVXLD_L2",
    0x119: "HVXLD_L2_TCM",
    0x11a: "HVXLD_L2_MISS",
    0x11b: "HVXLD_L2_SECONDARY_MISS",
    0x11c: "HVXST_L2_WR",
    0x11d: "HVXST_SLD_CONFLICT",
    0x11e: "HVXST_VTCM_GATH_CONFLICT",
    0x121: "HVXST_L2_FULL",
    0x122: "HVXST_VTCM_FULL",
    0x123: "HVXST_L2",
    0x124: "HVXST_L2_MISS",
    0x125: "HVXST_L2TCM",
    0x126: "HVXST_VTCM",
    0x127: "HVXST_L2_SECODARY_MISS",
    0x128: "HVXPIPE_ALU",
    0x129: "HVXPIPE_MPY",
    0x12a: "HVXPIPE_SHIFT",
    0x12b: "HVXPIPE_PERM",
    0x12c: "CYCLES_4_HVX_CONTEXTS_RUNNING",
    0x12d: "HVXVREGRD_EARLY_WR_1PKT",
    0x12e: "HVXVREGRD_LATE_WR_1PKT",
    0x12f: "HVXVREGRD_EARLY_WR_2PKT",
    0x130: "HVXVREGRD_LATE_WR_2PKT",
    0x131: "HVXVREGRD_EARLY_WR_3PKT",
    0x132: "HVXVREGRD_LATE_WR_3PKT",
    0x133: "HVXVREGRD_EARLY_WR",
    0x134: "HVXVREGRD_LATE_WR",
    0x135: "HVXVREGWR_EARLY_WR_1PKT",
    0x136: "HVXVREGWR_LATE_WR_1PKT",
    0x137: "HVXVREGWR_EARLY_WR_2PKT",
    0x138: "HVXVREGWR_LATE_WR_2PKT",
    0x139: "HVXVREGWR_EARLY_WR_3PKT",
    0x13a: "HVXVREGWR_LATE_WR_3PKT",
    0x13b: "HVXVREGWR_EARLY_WR",
    0x13c: "HVXVREGWR_LATE_WR",
    0x13d: "SCATGATH_ACTIVE",
    0x13e: "SCATGATHIN_OUTSTANDING",
    0x13f: "SCATGATHIN",
    0x140: "SCATGATH_OUTSTANDING",
    0x141: "SCATGATH_VST_CONFLICT",
    0x142: "SCATGATH_CONFLICT",
    0x143: "SCATGATH_FULL",
    0x144: "SCATGATH_SCAT",
    0x145: "SCATGATH_GATH",
    0x146: "SCATGATHBANK_CONFLICT",
    0x147: "SCATGATHBANK_RD",
    0x148: "SCATGATHBANK_WR",
    0x149: "SCATGATHBANK_MODIFY_RD",
    0x14a: "HVXWR_CONFLICT",
    0x14b: "HVXWR",
    0x14c: "VTCM_ACTIVE",
    0x14d: "VTCMCOMMITTED_FULL",
    0x14e: "VTCMCOMMITTED_DROP",
    0x14f: "VTCMCOMMITTED_TRACK",
    0x150: "VTCMOLDEST_OUTSTANDING",
    0x151: "VTCMOLDEST_READY",
    0x157: "VTCMEXEC_OUTSTANDING",
    0x159: "VTCMEXEC_VLD",
    0x15a: "VTCMEXEC_SLD",
    0x15b: "VTCMEXEC_REL",
    0x15c: "VTCMEXEC_SST",
    0x15d: "VTCMEXEC_MRD",
    0x15e: "VTCMEXEC_VST",
    0x15f: "VTCMEXEC_SCATGATH",
    0x160: "VTCMEXEC_MEMCPY",
    0x161: "VTCMEXEC_MWR",
    0x162: "SCATGATHBANK_FULL",
    0x163: "VTCM_BANDWIDTH_OVER",
    0x164: "VTCM_RD",
    0x165: "VTCM_WR",
    0x166: "VTCM_MODIFY_RD",
    0x167: "VTCMSLD_OUTSTANDING",
    0x168: "VTCMSLD_VST_CONFLICT",
    0x16a: "VTCMSLD_FULL",
    0x16b: "VTCMSLD",
    0x16e: "VTCMEXT_WR_OUTSTANDING",
    0x16f: "VTCMEXT_RD_FULL",
    0x170: "VTCMEXT_WR_FULL",
    0x171: "VTCMEXT_CONFLICT",
    0x172: "VTCMEXT_BANDWIDTH_OVER",
    0x173: "VTCMEXT_RD",
    0x174: "VTCMEXT_WR",
    0x176: "HVXLDSCFIFO_REL_ORDER",
    0x177: "VTCMEXTRET_FULL",
    0x178: "VTCMEXTRET_PARTIAL",
    0x179: "VTCMEXTRET",
    0x17a: "HVXWR_PARTIAL",
    0x17b: "HVXLDSCFIFO_BANK_CONFLICT",
    0x17c: "HVXLDSCFIFO_VLD",
    0x17d: "HVXLDSCFIFO_SLD",
    0x17e: "HVXLDSCFIFO_SST",
    0x17f: "HVXLDSCFIFO_REL_PERF",
    0x180: "COPROC0_PKT_XE",
    0x181: "COPROC0_FIFO_FULL_STALL",
    0x182: "COPROC0_VEXTRACT_STALL",
    0x183: "COPROC0_CYCLES_RUNNING",
    0x184: "COPROC1_PKT_XE",
    0x185: "COPROC1_FIFO_FULL_STALL",
    0x186: "COPROC1_VEXTRACT_STALL",
    0x187: "COPROC1_CYCLES_RUNNING",
    0x188: "COPROC2_PKT_XE",
    0x189: "COPROC2_FIFO_FULL_STALL",
    0x18a: "COPROC2_VEXTRACT_STALL",
    0x18b: "COPROC2_CYCLES_RUNNING",
    0x18c: "COPROC3_PKT_XE",
    0x18d: "COPROC3_FIFO_FULL_STALL",
    0x18e: "COPROC3_VEXTRACT_STALL",
    0x18f: "COPROC3_CYCLES_RUNNING",
    0x190: "HVX_VFIFO_EMPTY",
    0x191: "CYCLES_5_HVX_CONTEXTS_RUNNING",
    0x192: "CYCLES_6_HVX_CONTEXTS_RUNNING",
    0x200: "HMX_ACTIVE",
    0x201: "HMX_CVT_FULL",
    0x202: "HMX_MAC_FULL",
    0x203: "HMX_DROP",
    0x204: "HMX_CVT",
    0x205: "HMX_MAC",
    0x206: "HMX_PKT_THREAD",
    0x207: "HMX_MXFIFO_FULL",
    0x208: "HMXRDWGT_REUSE_PARTIAL",
    0x209: "HMXMAC_ACT_OUTSTANDING",
    0x20a: "HMXMAC_WGT_OUTSTANDING",
    0x20b: "HMXMAC_MULT_DROP",
    0x20c: "HMXRDACTR_REUSE1_PARTIAL",
    0x20d: "HMXMAC_POWER_OVER",
    0x20e: "HMXMAC_FXP_PARTIAL",
    0x20f: "HMXMAC_FLT_PARTIAL",
    0x210: "HMXMAC_DRAIN_PARTIAL",
    0x211: "HMXMAC_FXP",
    0x212: "HMXMAC_FLT",
    0x213: "HMXMAC_DRAIN",
    0x214: "HMX_CLK",
    0x215: "HMXCVT_MACORDER",
    0x216: "HMXCVT_BUSY",
    0x217: "HMXCVT_LDORDER",
    0x218: "HMXCVT_BUF_FULL",
    0x219: "HMXRDWGT_REUSE",
    0x21a: "HMXCVT_POWER_OVER",
    0x21b: "HMXCVT_FXP_PARTIAL",
    0x21c: "HMXCVT_FLT_PARTIAL",
    0x21e: "HMXCVT_FXP",
    0x21f: "HMXCVT_FLT",
    0x220: "HMXCVT_STORDER",
    0x221: "HMXCVT_CLR",
    0x222: "HMXARRAY_FXP_MPY",
    0x223: "HMXARRAY_FLT_MPY",
    0x224: "HMXARRAY_FXP_ACC",
    0x225: "HMXARRAY_FLT_ACC",
    0x226: "HMXARRAY_FXP_CVT",
    0x227: "HMXARRAY_FLT_CVT",
    0x228: "HMXMAC_ORDER",
    0x22b: "HMXRDACT_POWER_OVER",
    0x22c: "HMXRDACT_BUF_FULL",
    0x22d: "HMXRDWGT_WGT_BUF_FULL",
    0x22e: "HMXRDWGT_SCALE_BUF_FULL",
    0x22f: "HMXRDACT_CONFLICT",
    0x230: "HMXRDACT_PARTIAL",
    0x231: "HMXRDWGT_PARTIAL",
    0x232: "HMXRDACT_ACT",
    0x233: "HMXRDWGT_WGT",
    0x234: "HMXRDWGT_SCALE",
    0x235: "HMXRDACT_REUSE",
    0x236: "HMXRDACTR_REUSE2_PARTIAL",
    0x237: "HMXWR_OUTSTANDING",
    0x238: "HMXWR_CONFLICT",
    0x239: "HMXWR_PARTIAL",
    0x23a: "HMXWR_DROP",
    0x23b: "HMXWR",
    0x23c: "HMXWR_SCALE",
    0x23d: "HMXRDACT_DROP",
    0x23e: "HMXRDACTR_REUSE1",
    0x23f: "HMXRDACTR_REUSE2",
    0x240: "UDMA_ACTIVE_CYCLES",
    0x241: "UDMA_STALL_DESCRIPTOR_FETCH",
    0x243: "UDMA_STALL_TLB_MISS",
    0x244: "UDMA_STALL_MONITOR_GUEST_MODE",
    0x245: "UDMA_DMPOLL_CYCLES",
    0x246: "UDMA_DMWAIT_CYCLES",
    0x247: "UDMA_SYNCHT_CYCLES",
    0x248: "UDMA_TLBSYNCH_CYCLES",
    0x24a: "UDMA_DMPOLL",
    0x24b: "UDMA_DMWAIT",
    0x24c: "UDMA_TLB_MISS",
    0x24d: "UDMA_DESCRIPTOR_DONE",
    0x24e: "UDMA_DMSTART",
    0x24f: "UDMA_DMLINK",
    0x250: "UDMA_DMRESUME",
    0x251: "L2_UDMA_COHERENT_WR",
    0x252: "L2_UDMA_COHERENT_WR_MISS",
    0x253: "L2_UDMA_COHERENT_RD",
    0x254: "L2_UDMA_COHERENT_RD_MISS",
    0x255: "L2_UDMA_BYPASS_WR",
    0x256: "L2_UDMA_BYPASS_RD",
    0x257: "UDMA_VTCM_WR",
    0x258: "UDMA_VTCM_RD",
    0x259: "UDMA_DLBC_FETCH",
    0x25a: "UDMA_DLBC_FETCH_CYCLES",
    0x25b: "UDMA_UNALIGNED_DESCRIPTOR",
    0x25c: "UDMA_ORDERING_DESCRIPTOR",
    0x25d: "UDMA_PADDING_DESCRIPTOR",
    0x25e: "UDMA_UNALIGNED_RD",
    0x25f: "UDMA_UNALIGNED_WR",
    0x260: "UDMA_COHERENT_RD_CYCLES",
    0x261: "UDMA_COHERENT_WR_CYCLES",
    0x262: "UDMA_NONCOHERENT_RD_CYCLES",
    0x263: "UDMA_NONCOHERENT_WR_CYCLES",
    0x264: "UDMA_VTCM_RD_CYCLES",
    0x265: "UDMA_VTCM_WR_CYCLES",
    0x266: "UDMA_RD_BUFFER_LEVEL_LOW",
    0x267: "UDMA_RD_BUFFER_LEVEL_HALF",
    0x268: "UDMA_RD_BUFFER_LEVEL_HIGH",
    0x269: "UDMA_RD_BUFFER_LEVEL_FULL",
    0x26c: "AXI_SLAVE_VTCM_ACCESS",
    0x26d: "AXI2_SLAVE_VTCM_ACCESS",
    0x26e: "AXI_SLAVE_VTCM_RD",
    0x26f: "AXI2_SLAVE_VTCM_RD",
    0x270: "L2_UDMA_VTCM_CONGESTION",
    0x271: "L2_AXIS_VTCM_CONGESTION",
    0x272: "L2_AXI2_SLAVE_VTCM_CONGESTION",
    0x273: "L2_MEMCPY_VTCM_CONGESTION",
    0x274: "AXI_SLAVE_MULTI_BEAT_ACCESS_ILV0",
    0x275: "AXI_SLAVE_SINGLE_BEAT_ACCESS_ILV0",
    0x276: "AXI_SLAVE_MULTI_BEAT_ACCESS_ILV1",
    0x277: "AXI_SLAVE_SINGLE_BEAT_ACCESS_ILV1",
    0x278: "AXI_SLAVE_MULTI_BEAT_ACCESS_ILV2",
    0x279: "AXI_SLAVE_SINGLE_BEAT_ACCESS_ILV2",
    0x27a: "AXI_SLAVE_MULTI_BEAT_ACCESS_ILV3",
    0x27b: "AXI_SLAVE_SINGLE_BEAT_ACCESS_ILV3",
    0x280: "HMXCVTWR_OUTSTANDING",
    0x281: "HMXCVTWR_BUF_FULL",
    0x282: "HMXCVTWR_WR_SCALE",
    0x283: "HMXCVTWR_WR_PARTIAL",
    0x284: "HMXCVTWR_WR",
    0x285: "HMXCVT_WRORDER",
    0x286: "HMXWGTDCOMP_FLUSH",
    0x287: "HMXWGTDCOMP_OUTSTANDING",
    0x288: "HMXWGTDCOMP_BUF_FULL",
    0x289: "HMXWGTDCOMP_ISSUE",
    0x28a: "HMXWGTCOMP_OUTSTANDING",
    0x28b: "HMXWGTCOMP_BUF_FULL",
    0x28c: "HMXWGTCOMP_COMPDATA_RD",
    0x28d: "HMXWGTCOMP_METADATA_RD",
    0x28e: "HMXRDWGT_POWER_OVER",
    0x28f: "HMXRDWGT_CONFLICT",
    0x290: "HMXRDWGT_DROP",
    0x291: "HMX_MXFIFO_EMPTY",
    0x292: "HMX_LIMITS_THROTTLE_TLMH",
    0x293: "HMX_LIMITS_THROTTLE_LMH",
    0x294: "HMX_DPM_AVG_COMPRESSED",
    0x295: "HMX_POWERLIMITS_OVER",
    0x296: "HMXCVTBIAS_EMPTY",
    0x297: "HMXCVTBIAS_LD",
    0x298: "HMXCVTBIAS_LDOUTSTANDING",
    0x299: "HMXCVTBIAS_ST",
    0x29a: "HMXCVTBIAS_STORDER",
    0x29b: "HMXCVTBIAS_ORDER",
    0x29c: "HMXCVTMEM_EMPTY",
    0x29d: "HMXCVTMEM_WR",
    0x29e: "HMXCVTMEM_BST",
    0x29f: "HMXCVTMEM_WRFULL",
    0x2a0: "HMXCVTMEM_LDORDER",
    0x2a1: "HMXCVTMEM_STORDER",
    0x2a2: "HMXCVTMEM_ORDER",
    0x2a3: "HMXCVTINTBUF_EMPTY",
    0x2a4: "HMXCVTINTBUF_CVTORDER",
    0x2a5: "HMXCVTINTBUF_MXWR_FULL",
    0x2a6: "HMXCVTINTBUF_MXWR_DATA",
    0x2a7: "HMXCVTINTBUF_CVT",
    0x2a8: "HMXCVTINTBUF_MXWR_SCALE",
    0x2bc: "IU_CREDIT_FAIL_IU0",
    0x2bd: "ICACHE_DEMAND_MISS_IU0",
    0x2be: "ANY_IU_REPLAY_IU0",
    0x2bf: "ISSUED_PACKETS_IU0",
    0x2c0: "ISSUED_INSTS_IU0",
    0x2c1: "ICACHE_ACCESS_IU0",
    0x2c2: "BTB_HIT_IU0",
    0x2c3: "BTB_MISS_IU0",
    0x2c4: "IU_DEMAND_SECONDARY_MISS_IU0",
    0x2c5: "IU_LINE_FROM_HWLOOP_IU0",
    0x2c6: "FAST_FETCH_KILLED_IU0",
    0x2c7: "IU_1_PKT_AVAILABLE_TO_ISSUE_IU0",
    0x2c8: "FETCHED_PACKETS_DROPPED_IU0",
    0x2c9: "IU_REQUESTS_TO_L2_REPLAYED_IU0",
    0x2ca: "IU_PREFETCHES_SENT_TO_L2_IU0",
    0x2cb: "ITLB_MISS_IU0",
    0x2cc: "IU_2_PKT_AVAILABLE_TO_ISSUE_IU0",
    0x2cd: "IU_3_PKT_AVAILABLE_TO_ISSUE_IU0",
    0x2ce: "IU_REQUEST_STALLED_IU0",
    0x2cf: "IU_BIMODAL_L2_ELIGIBLE_IU0",
    0x2d0: "IU_0_PKT_AVAILABLE_TO_ISSUE_IU0",
    0x2d1: "FETCH_2_CYCLE_IU0",
    0x2d2: "FETCH_3_CYCLE_IU0",
    0x2d3: "IU_PREFETCHES_DROPPED_IU0",
    0x2d4: "L2ITCM_IU_READ_IU0",
    0x2d5: "L2ITCM_BIMODAL_WRITES_SUCCESS_IU0",
    0x2d6: "L2ITCM_BIMODAL_WRITES_DROPPED_IU0",
    0x2d7: "L2ITCM_IU_PREFETCH_READ_IU0",
    0x2d8: "IU_CREDIT_FAIL_IU1",
    0x2d9: "ICACHE_DEMAND_MISS_IU1",
    0x2da: "ANY_IU_REPLAY_IU1",
    0x2db: "ISSUED_PACKETS_IU1",
    0x2dc: "ISSUED_INSTS_IU1",
    0x2dd: "ICACHE_ACCESS_IU1",
    0x2de: "BTB_HIT_IU1",
    0x2df: "BTB_MISS_IU1",
    0x2e0: "IU_DEMAND_SECONDARY_MISS_IU1",
    0x2e1: "IU_LINE_FROM_HWLOOP_IU1",
    0x2e2: "FAST_FETCH_KILLED_IU1",
    0x2e3: "IU_1_PKT_AVAILABLE_TO_ISSUE_IU1",
    0x2e4: "FETCHED_PACKETS_DROPPED_IU1",
    0x2e5: "IU_REQUESTS_TO_L2_REPLAYED_IU1",
    0x2e6: "IU_PREFETCHES_SENT_TO_L2_IU1",
    0x2e7: "ITLB_MISS_IU1",
    0x2e8: "IU_2_PKT_AVAILABLE_TO_ISSUE_IU1",
    0x2e9: "IU_3_PKT_AVAILABLE_TO_ISSUE_IU1",
    0x2ea: "IU_REQUEST_STALLED_IU1",
    0x2eb: "IU_BIMODAL_L2_ELIGIBLE_IU1",
    0x2ec: "IU_0_PKT_AVAILABLE_TO_ISSUE_IU1",
    0x2ed: "FETCH_2_CYCLE_IU1",
    0x2ee: "FETCH_3_CYCLE_IU1",
    0x2ef: "IU_PREFETCHES_DROPPED_IU1",
    0x2f0: "L2ITCM_IU_READ_IU1",
    0x2f1: "L2ITCM_BIMODAL_WRITES_SUCCESS_IU1",
    0x2f2: "L2ITCM_BIMODAL_WRITES_DROPPED_IU1",
    0x2f3: "L2ITCM_IU_PREFETCH_READ_IU1",
    0x2f4: "TAGE_TABLE_ALLOC_IU0",
    0x2f5: "TAGE_TABLE_HIT_IU0",
    0x2f6: "TAGE_BRANCH_OVERRIDE_IU0",
    0x2f7: "TAGE_TABLE_ALLOC_IU1",
    0x2f8: "TAGE_TABLE_HIT_IU1",
    0x2f9: "TAGE_BRANCH_OVERRIDE_IU1",
    0x2fa: "L2_CLEAN_CASTOUT",
    0x2fb: "AXI3_READ_REQUEST",
    0x2fc: "AXI3_LINE32_READ_REQUEST",
    0x2fd: "AXI3_WRITE_REQUEST",
    0x2fe: "AXI3_LINE32_WRITE_REQUEST",
    0x2ff: "AXI3_RD_CONGESTION",
    0x300: "CYCLES_1_PACKET_COMMITTED",
    0x301: "CYCLES_2_PACKET_COMMITTED",
    0x302: "CYCLES_3_PACKET_COMMITTED",
    0x303: "CYCLES_4_PACKET_COMMITTED",
    0x304: "SMT_CLUSTER0",
    0x305: "SMT_CLUSTER1",
    0x306: "SMT_INTERCLUSTER",
    0x307: "SMT_CONFLICT_FOR_REG_READ_OR_CU_FWD",
    0x308: "COMMITTED_PKT_2_THREAD_RUNNING_2T_PLUS_0T",
    0x309: "COMMITTED_PKT_2_THREAD_RUNNING_1T_PLUS_1T",
    0x30a: "COMMITTED_PKT_3_THREAD_RUNNING_3T_PLUS_0T",
    0x30b: "COMMITTED_PKT_3_THREAD_RUNNING_2T_PLUS_1T",
    0x30c: "COMMITTED_PKT_4_THREAD_RUNNING_4T_PLUS_0T",
    0x30d: "COMMITTED_PKT_4_THREAD_RUNNING_3T_PLUS_1T",
    0x30e: "COMMITTED_PKT_4_THREAD_RUNNING_2T_PLUS_2T",
    0x30f: "COMMITTED_PKT_5_THREAD_RUNNING_4T_PLUS_1T",
    0x310: "COMMITTED_PKT_5_THREAD_RUNNING_3T_PLUS_2T",
    0x311: "COMMITTED_PKT_6_THREAD_RUNNING_4T_PLUS_2T",
    0x312: "COMMITTED_PKT_6_THREAD_RUNNING_3T_PLUS_3T",
    0x313: "ICACHE_DEMAND_MISS_PREFETCH_MISS",
    0x314: "SIMPLE_PACKET",
    0x315: "AXI3_LINE64_WRITE_REQUEST",
    0x316: "AXI3_LINE64_READ_REQUEST",
    0x317: "AXI3_WR_CONGESTION",
    0x318: "AXI3_INCOMPLETE_WRITE_REQUEST",
    0x319: "ICACHE_DATA_REPLAY",
    0x31a: "L2_PIPE_ACCESS_SPLITS",
    0x31b: "L2_BANK_CONFLICT_STALL",
    0x31c: "SMT_PKT_PICKED_BUT_NOT_DISP",
    0x322: "CLADE2_EB_FULL",
    0x323: "CLADE2_RD_REQ",
    0x324: "CLADE2_RDCACHE_MISS",
    0x325: "CLADE2_WR_REQ",
    0x326: "CLADE2_WRCACHE_MISS",
    0x327: "AXI_EWD_REQUEST",
    0x328: "AXI_EWD_REQUEST_EVEN",
    0x329: "AXI_CMO_REQUEST",
    0x32a: "AXI_CMO_REQUEST_EVEN",
    0x32b: "ICACHE_DEMAND_MISS_PREFETCH_MISS_IU0",
    0x32c: "ICACHE_DEMAND_MISS_PREFETCH_MISS_IU1",
    0x32d: "TAGE_TABLE_ALLOC",
    0x32e: "TAGE_TABLE_HIT",
    0x32f: "TAGE_BRANCH_OVERRIDE",
    0x330: "VMEM_ST_SMT_DU_PORT_CONFLICT_REPLAY",
    0x331: "L2_BUS_SCOREBOARD_RD_CONGESTION",
    0x332: "L2_BUS_SCOREBOARD_WR_CONGESTION",
    0x333: "DU_SPF_DTLBPGCROSS",
    0x334: "DU_SPF_DCACHE_HIT",
    0x335: "DU_SPF_DCACHE_MISS",
    0x336: "DU_SPF_L2FIFOFULL_RETRY",
    0x337: "DU_SPF_L2BUFFULL_RETRY",
    0x338: "DU_SPF_CONFLICT_RETRY",
    0x339: "MINICACHE_ACCESS",
    0x33a: "MINICACHE_HIT",
    0x33b: "MINICACHE_MISS",
    0x33c: "JTLB_READ",
    0x33d: "JTLB_WRITE",
    0x33e: "COMMITTED_PKT_INTMAC_B2B",
    0x33f: "COMMITTED_PKT_INTMAC_B2B_SLOTFLIP",
    0x340: "DPM_AVG_COMPRESSED",
    0x341: "CU_REFETCH",
    0x343: "CU_REG_WR_REPLAY",
    0x344: "COMMITTED_INSNS_TC2",
    0x345: "COMMITTED_INSNS_TC2LATEPRED",
    0x346: "COMMITTED_INSNS_TC3",
    0x347: "COMMITTED_INSNS_TC3X",
    0x348: "COMMITTED_INSNS_TCNEWVJUMP",
    0x349: "COMMITTED_INSNS_TC3STALL",
    0x34a: "COMMITTED_INSNS_TCLD",
    0x34b: "COMMITTED_INSNS_TCST",
    0x34c: "COMMITTED_INSNS_TC2EARLY",
    0x34d: "COMMITTED_INSNS_TC4X",
    0x34e: "COMMITTED_INSNS_TCLATEPRED_LDAIA",
    0x34f: "COMMITTED_INSNS_TCLATEPRED_STAIA",
    0x350: "DU_NUM_WAY_PREDICTIONS",
    0x351: "DU_WAY_PRED_REPLAYS",
    0x352: "DU_BANKCONFLICTREPLAY_INVALID",
    0x353: "CU_CYCLES_1_MEM_OP_DISPATCHED",
    0x354: "CU_CYCLES_2_MEM_OP_DISPATCHED",
    0x355: "CU_CYCLES_3_MEM_OP_DISPATCHED",
    0x356: "CU_L2FIFO_THROTTLE",
    0x357: "ET_C0_FIFO_OVERFLOW",
    0x358: "ET_C1_FIFO_OVERFLOW",
    0x359: "ET_C0_ETB_FULL",
    0x35a: "ET_C1_ETB_full",
    0x35b: "ET_C0_ETB_OVER_HALF",
    0x35c: "ET_C1_ETB_OVER_HALF",
    0x35d: "ET_C0_ETB_OVER_THREE_FOURTHS",
    0x35e: "ET_C1_ETB_OVER_THREE_FOURTHS",
    0x35f: "ET_REQUEST_CU_STALL",
    0x360: "TAGE_BRANCH_OVERRIDE_INCORRECT",
    0x361: "AHB_8_WRITE_REQUEST",
    0x362: "DU_MINOR_PIPE_DTLB_MISS",
    0x363: "DU_ANY_MINOR_PIPE_REPLAY",
    0x364: "DU_AIA_SPF_ISSUED",
    0x365: "DU_PC_SPF_ISSUED",
    0x366: "DU_R29_SPF_ISSUED",
    0x367: "DU_R29_LOAD_MISS",
    0x368: "DU_R29_STORE_SECMISS",
    0x369: "DU_HMX_LATE_READ_MISPREDICTION_REPLAY",
    0x36a: "ICSMT_THREAD_IDLE_PVIEW_CYCLES",
    0x36b: "ICSMT_ARCH_LOCK_PVIEW_CYCLES",
    0x36c: "ICSMT_REDIRECT_PVIEW_CYCLES",
    0x36d: "ICSMT_IU_NO_PKT_PVIEW_CYCLES",
    0x36e: "ICSMT_DU_CACHE_MISS_PVIEW_CYCLES",
    0x36f: "ICSMT_DU_BUSY_OTHER_PVIEW_CYCLES",
    0x370: "ICSMT_CU_BUSY_PVIEW_CYCLES",
    0x371: "ICSMT_DU_CONFLICT_PVIEW_CYCLES",
    0x372: "ICSMT_COPROC_BUSY_PVIEW_CYCLES",
    0x373: "ICSMT_DU_UNCACHED_PVIEW_CYCLES",
    0x374: "ICSMT_SYSTEM_BUSY_PVIEW_CYCLES",
    0x375: "ICSMT_RESRC_CONFLICT_PVIEW_CYCLES",
    0x376: "ICSMT_NO_TID_PVIEW_CYCLES",
    0x377: "THREAD_LMH_THROTTLE_T0",
    0x378: "THREAD_LMH_THROTTLE_T1",
    0x379: "THREAD_LMH_THROTTLE_T2",
    0x37a: "THREAD_LMH_THROTTLE_T3",
    0x37b: "THREAD_LMH_THROTTLE_T4",
    0x37c: "THREAD_LMH_THROTTLE_T5",
    0x37d: "THREAD_LMH_THROTTLE_T6",
    0x37e: "THREAD_LMH_THROTTLE_T7",
    0x37f: "QNS_OOO_FRAG",
    0x380: "L2_DMA_RETURN_BANK_CONFLICT",
    0x381: "L2_DU_UNCACHED_STORE_ACCESS",
    0x382: "L2_DU_UNCACHED_STORE_COALESCE",
    0x383: "DU_MINOR_SLVRCONFLICT_REPLAY",
    0x384: "L2_BUS_WRITE_DATA_BEATS",
    0x385: "L2_BUS_READ_DATA_BEATS",
    0x386: "S1ST_S0LD_NOFWD_REPLAY",
    0x387: "CU_BANK_CONFLICT_BLOCK",
    0x388: "CU_BANK_CONFLICT_FLUSH",
    0x389: "CU_BANK_CONFLICT_PREDICTION",
    0x38a: "DU_STORE_HIT_ALIASED_LINE",
    0x38b: "DU_R30_SPF_ISSUED",
    0x38c: "DU_R30_LOAD_MISS",
    0x38d: "DU_R30_STORE_SECMISS",
    0x38e: "CORE_DPM_AVG_COMPRESSED",
    0x38f: "XU_PMU_MULTIPLY",
    0x390: "XU_PMU_SHIFT",
    0x391: "IU_LRU_UPDATE_TO_L2",
    0x392: "COMMITTED_LOAD_LOCKS",
    0x393: "COMMITTED_STORE_CONDITIONALS",
    0x394: "LOAD_LOCK_STALLS",
    0x395: "STORE_CONDITIONAL_STALLS",
    0x3e8: "HVX_L2_STORE_ACCESS_UARCH",
    0x3e9: "DU_BACKPRESSURE_CYCLES",
    0x3ea: "SYNCHT_CYCLES",
    0x3eb: "IU_CYCLES",
    0x3ec: "PAUSE_CYCLES",
    0x3ed: "WAIT_CYCLES",
    0x3ee: "vecx_regdep",
    0x3ef: "CYCLES_0_THREAD_RUNNING",
    0x3f0: "FP_CYCLES",
    0x3f1: "PREDICATE_KILLS",
    0x3f2: "HVX_L2_LOAD_ACCESS_UARCH",
    0x3f3: "HVX_TCM_STORE_ACCESS_UARCH",
    0x3f4: "HVX_TCM_LOAD_ACCESS_UARCH",
    0x3f5: "SCATGATH_SCAT_UARCH",
    0x3f6: "SCATGATH_IN_UARCH",
    0x3f7: "VTCM_EXEC_SLD_UARCH",
    0x3f8: "VTCM_EXEC_SST_UARCH",
    0x3f9: "VTCM_EXEC_SCATGATH_UARCH",
    0x3fb: "VTCM_EXEC_VLD_UARCH",
    0x3fc: "DU_UNCACHED_ALREADY_IN_DCACHE",
    0x3fd: "IU_L1S_REQUEST_QOS",
    0x3fe: "DU_LOAD_MISS_CYCLES",
    0x3ff: "DU_STORE_MISS_CYCLES",
    0x400: "DCACHE_STORE_ALLOCATES_REUSED",
    0x401: "COPROC_CORE_DU_UNREADY_CYCLES",
    0x402: "HVX_L2FIFO_CYCLES",
    0x403: "HVX_DU_RESOURCE_CYCLES",
    0x404: "HVX_VFIFO_UNREADY_CYCLES",
    0x405: "HVX_VFIFO_FULL_CYCLES",
    0x406: "COPROC_CORE_EXT_BUSY_CYCLES",
    0x407: "DU_WAY_RESERVED_CYCLES",
    0x408: "CU_DU_UNCACHED_CYCLES",
    0x409: "DU_BANK_CONFLICT_CYCLES",
    0x40a: "DU_UTLB_MISS_CYCLES",
    0x40b: "DU_PAGE_CROSS_CYCLES",
    0x40c: "DU_STORE_BUFFER_FULL_CYCLES",
    0x40d: "DU_STORE_BUFFER_HIT_REPLAY_CYCLES",
    0x40e: "DU_STORE_BUFFER_FORCE_REPLAY_CYCLES",
    0x40f: "PAGE_COLOR_MISMATCH",
    0x410: "JU_TOTAL_WAIT_CYCLES",
    0x411: "L2TAG_OP_STALL_CYCLES",
    0x412: "CU_INTERLOCK_CYCLES",
    0x413: "COPROC_FIFO_FULL_CYCLES",
    0x415: "BE_DUMMY_STALL_CYCLES",
    0x416: "BE_BRANCH_DELAY_CYCLES",
    0x417: "DU_ALL_WAY_RESERVED",
    0x418: "CU_VFIFO_FULL_CYCLES",
    0x419: "CU_MXFIFO_FULL_CYCLES",
    0x41a: "INDIRECT_JUMP_CYCLES",
    0x41b: "DU_FILL_CNFLT_CYCLES",
    0x41c: "VTCM_EXEC_VST_UARCH",
    0x41d: "CU_VTCM_FIFO_FULL_CYCLES",
    0x41e: "CU_NO_DISPATCH_CYCLES",
    0x41f: "CU_THREAD_OFF_CYCLES",
    0x420: "CU_ARCH_LOCK_CYCLES",
    0x421: "CU_IQ_EMPTY_CYCLES",
    0x422: "CU_IU_BRANCH_MISS_CYCLES",
    0x423: "CU_DU_MISS_CYCLES",
    0x424: "CU_DU_BUSY_CYCLES",
    0x425: "CU_REG_INTERLOCK_CYCLES",
    0x426: "CU_BE_NOB2B_CYCLES",
    0x427: "CU_LATE_READ_CYCLES",
    0x428: "CU_EARLY_WRITE_CYCLES",
    0x429: "CU_NOB2B_EXTENSION_CYCLES",
    0x42a: "CU_RC_LDST_CYCLES",
    0x42d: "CU_STALL_LAST_CYCLES",
    0x42e: "CU_RC_SOLO_CYCLES",
    0x42f: "CU_RC_FILL_CYCLES",
    0x431: "CU_LOCKED_CYCLES",
    0x432: "MISPREDICT_TIME_CYCLES",
    0x433: "NOT_SELECTED_CYCLES",
    0x434: "TOFF_CYCLES",
    0x435: "K0LOCKED_CYCLES",
    0x436: "BE_DUMMY_EXCEPT_STALL_CYCLES",
    0x437: "FE_NEWVALUE_MISPREDICT_CYCLES",
    0x438: "CU_RXX_INTERLOCK_CYCLES",
    0x439: "POST_REPLAY_B2B_BLOCK",
    0x43a: "CU_DU_EVICT_CYCLES",
    0x43b: "DU_EVICT_PLACEHOLDER",
    0x43c: "IU_FETCH_CROSS_CYCLES",
    0x43d: "IU_WAY_RESERVED_CYCLES",
    0x43e: "IU_BACKPRESSURE_CYCLES",
    0x43f: "IU_UTLB_MISS_CYCLES",
    0x440: "CU_JUMP_MISSPEC_CYCLES",
    0x441: "DU_MISS_CYCLES",
    0x442: "DU_P0_STBUF_FULL",
    0x443: "DU_P1_STBUF_FULL",
    0x444: "DU_SECMISS_CYCLES",
    0x446: "DU_PORT_CONFLICT_CYCLES",
    0x447: "DU_FILL_CONFLICT_CYCLES",
    0x448: "COPROC_STORE_BUFFER_REPLAY",
    0x449: "DU_STATE_CONFLICT_CYCLES",
    0x44a: "DU_INDEX_CONFLICT_CYCLES",
    0x44b: "COPROC2_CORE_EXT_BUSY_CYCLES",
    0x44e: "IU_RAS_MISS_CYCLES",
    0x450: "IU_UNCACHED_CYCLES",
    0x452: "IU_PAGE_CROSS_CYCLES",
    0x453: "L2FETCH_BACKPRESSURE_CYCLES",
    0x458: "IU_HIT_CYCLES",
    0x459: "CU_BE_NOBSB_CYCLES",
    0x45a: "L2BUS_WAY_RESERVED_CYCLES",
    0x45e: "FE_MISPREDICT_TIME_CYCLES",
    0x45f: "IU_DEALLOC_RETURN_CYCLES",
    0x461: "COPROC_SILVER_EXTRACT_CYCLES",
    0x462: "VECX_REGREAD_CYCLES",
    0x463: "IU_FILL_REPLAY_CYCLES",
    0x464: "COPROC_SILVER_VIDX_CYCLES",
    0x465: "COPROC_SILVER_L1S_CYCLES",
    0x466: "COPROC_SILVER_ALIGN_CYCLES",
    0x467: "COPROC_SILVER_L1S_LOOKUP_CYCLES",
    0x468: "COPROC_SILVER_IDLE_CYCLES",
    0x469: "COPROC_SILVER_NOP_CYCLES",
    0x46a: "IU_TLB_MISS_CYCLES",
    0x46b: "DU_TLB_MISS_CYCLES",
    0x46c: "WALKER_DU_HIT_CYCLES",
    0x46d: "WALKER_DU_MISS_CYCLES",
    0x46e: "WALKER_DU_RETRY_CYCLES",
    0x46f: "L2F_UTLB_MISS_CYCLES",
    0x470: "L2F_TLB_MISS_CYCLES",
    0x471: "IU_TLB_WALK_DUPLICATE_CYCLES",
    0x472: "DU_TLB_WALK_DUPLICATE_CYCLES",
    0x473: "L2F_TLB_WALK_DUPLICATE_CYCLES",
    0x474: "IU_TLB_WALK_CONTEXTS_FULL_CYCLES",
    0x475: "DU_TLB_WALK_CONTEXTS_FULL_CYCLES",
    0x476: "L2F_TLB_WALK_CONTEXTS_FULL_CYCLES",
    0x477: "COPROC_SILVER_IF_VFIFO_FULL_CYCLES",
    0x478: "COPROC_HVX_ENGINE_POP_CYCLES",
    0x479: "COPROC_HVX_ENGINE_IDLE_CYCLES",
    0x47a: "L2CACHE_BACKPRESSURE_CYCLES",
    0x47b: "L2CACHE_BUSY_CYCLES",
    0x47c: "L2CACHE_IDLE_CYCLES",
    0x47d: "FE_STALL_CYCLES",
    0x47e: "CU_DU_XU_NO_FWD_CYCLES",
    0x47f: "CU_FP_RX_NO_NTWK_CYCLES",
    0x481: "CU_PREG_INTERLOCK_CYCLES",
    0x482: "DCACHE_DEMAND_MISS_CYCLES",
    0x483: "COPROC_VFIFO_ENQUEUE",
    0x484: "IU_POP",
    0x485: "IU_POP_CYCLES",
    0x486: "ICACHE_DEMAND_MISS_CYCLES",
    0x487: "CU_LOOP_MISPREDICT_CYCLES",
    0x488: "CU_MAX_DISPATCH_CYCLES",
    0x489: "CU_QOS_NODISPATCH_CYCLES",
    0x48a: "BPUPDATE_SENT",
    0x48b: "BPUPDATE_APPLIED",
    0x48c: "BPUPDATE_RETRIEVED",
    0x48d: "CU_RC_SBFD_CYCLES",
    0x48e: "PCTRACE_UNACCOUNTED_MISPREDICTS",
    0x48f: "PCTRACE_UNACCOUNTED_MISPREDICTS_INTERRUPTS",
    0x490: "COMMITTED_PKT_TRACE",
    0x491: "COMMITTED_PKT_TRAP",
    0x492: "IPREFETCH_DROP_L2_MISS",
    0x493: "CU_FP_GRANDPARENT_BLOCK_CYCLES",
    0x494: "DU_DEALLOC_SECURITY_REPLAY_CYCLES",
    0x495: "REQUEST_STALL_SB_ENTRY_EXHAUSTION",
    0x496: "L2SB_PIPELINE_LOADS",
    0x497: "PKT_HAS_DUAL_WRITES_TO_SAME_REG",
    0x498: "PKT_HAS_DUAL_WRITES_TO_SAME_REG_AND_DEP",
    0x499: "CU_DUAL_WRITE_INTERLOCK_CYCLES",
    0x49a: "BPUPDATE_DROPPED",
    0x49b: "COPROC_SILVER_ENGINE_POP_CYCLES",
    0x49c: "L2FIFO_STORE_COALESCED",
    0x49d: "COPROC_SILVER_ENGINE_REG_INTERLOCK_CYCLES",
    0x49e: "COPROC_SILVER_ENGINE_BUSY_CYCLES",
    0x49f: "COPROC_SILVER_IF_VFIFO_UNREADY_CYCLES",
    0x4a1: "SBUF_DUAL_DRAIN",
    0x4a2: "SBUF_FWD_ON_LOAD_HIT",
    0x4a3: "SBUF_STORE_MATCH",
    0x4a4: "MEMCPY_BUSY_CYCLES",
    0x4aa: "IU_PKTQ_FULL",
    0x4af: "IU_PREFETCH_ACCESS",
    0x4b0: "IU_PREFETCH_USED",
    0x4b1: "FPKR_IPREF_OVER_FNEXT",
    0x4b2: "DUNCACHED_DEMAND_MISS_CYCLES",
    0x4b3: "ENDLOOP_CYCLES",
    0x4b4: "EXCEPTION_CYCLES",
    0x4b5: "BTB_ACCEL_PIPELINE_SCHEDULED",
    0x4b6: "BTB_ACCEL_PIPELINE_PICKED",
    0x4c2: "FE_PICK_OTHER",
    0x4c3: "FE_PICK_IPREFETCH",
    0x4c4: "FE_PICK_L2FILL",
    0x4c5: "FE_TAG_CONFLICT",
    0x4c6: "DU_SLOT0_REPLAY",
    0x4c7: "CU_RC_DUREP_CYCLES",
    0x4c8: "BTB_HIT_WRONG_SRC",
    0x4c9: "BTB_HIT_RIGHT_DEST_ICACHE_MISS",
    0x4ca: "BTB_HIT_WRONG_DST_PC",
    0x4cb: "BTB_MISS_ICACHE_MISS",
    0x4cc: "DCACHE_DEMAND_LOAD_MISS_CYCLES",
    0x4cd: "DCACHE_DEMAND_STORE_MISS_CYCLES",
    0x4ce: "DCACHE_L2HIT_DEMAND_LOAD_MISS_CYCLES",
    0x4cf: "DCACHE_L2HIT_DEMAND_STORE_MISS_CYCLES",
    0x4d0: "DCACHE_TCM_DEMAND_LOAD_MISS_CYCLES",
    0x4d1: "DCACHE_TCM_DEMAND_STORE_MISS_CYCLES",
    0x4d2: "DCACHE_AXI_DEMAND_LOAD_MISS_CYCLES",
    0x4d3: "DCACHE_AXI_DEMAND_STORE_MISS_CYCLES",
    0x4d4: "DCACHE_AXI2_DEMAND_LOAD_MISS_CYCLES",
    0x4d5: "DCACHE_AXI2_DEMAND_STORE_MISS_CYCLES",
    0x4d6: "DCACHE_AHB_DEMAND_LOAD_MISS_CYCLES",
    0x4d7: "DCACHE_AHB_DEMAND_STORE_MISS_CYCLES",
    0x4d8: "DUNCACHED_TCM_DEMAND_MISS_CYCLES",
    0x4d9: "DUNCACHED_AXI_DEMAND_MISS_CYCLES",
    0x4da: "DUNCACHED_AXI2_DEMAND_MISS_CYCLES",
    0x4db: "DUNCACHED_AHB_DEMAND_MISS_CYCLES",
    0x4dd: "LOAD_ON_STORE_HIT_REPLAY",
    0x4de: "STORE_ON_STORE_HIT_REPLAY",
    0x4df: "DU_SECMISS_REPLAY_CYCLES",
    0x4e0: "COMMITTED_DWORD_LOADS",
    0x4e1: "COMMITTED_DWORD_STORES",
    0x4ef: "CANCELED_DOTNEWLOADS_EXCEPTION",
    0x4f0: "CANCELED_DOTNEWSTORES_EXCEPTION",
    0x4f1: "UTRACE_LD_FORCED_WIDTH_1",
    0x4f2: "UTRACE_ST_FORCED_WIDTH_1",
    0x4f3: "PKT_HAS_NONSOLO_RETURN",
    0x4f4: "DCACHE_ACCESS",
    0x4f5: "CONSECUTIVE_SAMELINE_DCACCESS",
    0x4f6: "REDIRECT_MISC_UNEXPLAINED",
    0x4f7: "REDIRECT_MISC_JUMPR",
    0x4f8: "REDIRECT_MISC_TRAP",
    0x4f9: "REDIRECT_MISC_EXCEPTION",
    0x4fa: "ASID_VAHIGH_MISMATCH_SBUF",
    0x4fb: "MEMCPY_CYCLES_ACTIVE",
    0x4fc: "MEMCPY_CYCLES_STALLED",
    0x4fd: "MEMCPY_CYCLES_IDLE",
    0x4fe: "L2_ECC_PARTIAL_STORES",
    0x508: "INTERTHREAD_SBUF_HIT_REPLAY",
    0x509: "ICACHE_L2HIT_DEMAND_MISS_CYCLES",
    0x50a: "ICACHE_L2MISS_DEMAND_MISS_CYCLES",
    0x50b: "FE_POPQ",
    0x50c: "REDIRECT_MISC_INITLOOP",
    0x50d: "L2SB_STORE_TO_LOAD_FWD",
    0x50e: "CU_CREG_INTERLOCK_CYCLES",
    0x50f: "FE_PKTQ_FULL",
    0x510: "FE_ISYNC",
    0x511: "MISC_STALL",
    0x512: "TFE_CU_LOOP_RACE",
    0x513: "IU_NOTHIT_STOP_STALL",
    0x514: "IU_HIT_STOP_PKTQ_FULL",
    0x515: "IU_HIT_STOP_BTB_HIT",
    0x516: "IU_HIT_STOP_BTB_MISS",
    0x517: "IU_HIT_STOP_JUMP_UNKNOWN",
    0x518: "IU_HIT_STOP_ENDLOOP_TAKEN",
    0x519: "IU_HIT_STOP_ENDLOOP_FALLTHROUGH",
    0x51a: "IU_HIT_STOP_ISSUE_MAX",
    0x51b: "IU_HIT_STOP_PARTIAL",
    0x51c: "IU_HIT_STOP_EOL",
    0x51d: "IU_HIT_STOP_RETURN",
    0x51e: "IU_HIT_STOP_PARTIAL_BOL",
    0x51f: "IU_HIT_STOP_ISSUE_MAX_NEXT_NONPARTIAL",
    0x520: "IU_RAS_ADDR_UNAVAILABLE",
    0x521: "BTB_HIT_SLOWDOWN_SAME_PAGE_CHECK",
    0x522: "CU_WRITE_PORT_BLOCK_CYCLES",
    0x523: "CU_WRITE_REG_BLOCK_CYCLES",
    0x524: "CU_BE_NO_IMT3_NCJ_CYCLES",
    0x525: "TAIA_FWD_SLOT0_TO_SLOT0",
    0x526: "TAIA_FWD_SLOT0_TO_SLOT1",
    0x527: "TAIA_FWD_SLOT1_TO_SLOT0",
    0x528: "TAIA_FWD_SLOT1_TO_SLOT1",
    0x529: "TPREDUSE_BSB",
    0x52a: "CU_VEXTRACT_CYCLES",
    0x52b: "CU_NON_LAST_SUBPKT_CYCLES",
    0x52c: "FE_PKT_DRAIN",
    0x52d: "CU_REVERSED_PREDICATE_CYCLES",
    0x52e: "FE_WAY_DELAY_FOR_BTB_MISS",
    0x52f: "FE_WAY_DELAY_FOR_ENDLOOP_TAKEN",
    0x530: "FE_WAY_DELAY_FOR_FIRST_CROSS",
    0x531: "FE_WAY_DELAY_FOR_CU_REDIRECT",
    0x532: "COMMITTED_ICLASS_EXTENDER",
    0x533: "COMMITTED_ICLASS_CJ",
    0x534: "COMMITTED_ICLASS_NCJ",
    0x535: "COMMITTED_ICLASS_V4LDST",
    0x536: "COMMITTED_ICLASS_V2LDST",
    0x537: "COMMITTED_ICLASS_J",
    0x538: "COMMITTED_ICLASS_CR",
    0x539: "COMMITTED_ICLASS_ALU32_2OP",
    0x53a: "COMMITTED_ICLASS_S_2OP",
    0x53b: "COMMITTED_ICLASS_LD",
    0x53c: "COMMITTED_ICLASS_ST",
    0x53d: "COMMITTED_ICLASS_ALU32_ADDI",
    0x53e: "COMMITTED_ICLASS_S_3OP",
    0x53f: "COMMITTED_ICLASS_ALU64",
    0x540: "COMMITTED_ICLASS_M",
    0x541: "COMMITTED_ICLASS_ALU32_3OP",
    0x542: "COMMITTED_ICLASS_COPROC_VX",
    0x543: "COMMITTED_ICLASS_COPROC_VMEM",
    0x544: "COMMITTED_ICLASS_SUBINSN",
    0x545: "PKTSTATS_MEM_LD_ST",
    0x546: "PKTSTATS_MEM_LD_LD",
    0x547: "PKTSTATS_MEM_ST_ST",
    0x548: "PKTSTATS_INSN1",
    0x549: "PKTSTATS_INSN2",
    0x54a: "PKTSTATS_INSN3",
    0x54b: "PKTSTATS_INSN4",
    0x54c: "PKTSTATS_VMEM_DUAL",
    0x54d: "PKTSTATS_VMEM_LD",
    0x54e: "PKTSTATS_EXT_INSN1",
    0x54f: "PKTSTATS_EXT_INSN2",
    0x550: "PKTSTATS_EXT_INSN3",
    0x551: "PKTSTATS_EXT_INSN4",
    0x553: "HMX_PKT",
    0x554: "PKTSTATS_VMEM_ST",
    0x555: "INTERESTING_CYCLES",
    0x556: "TOTAL_SIMULATION_CYCLES",
    0x557: "START_ROI",
    0x565: "PKTSTATS_NOEXT_INSN1",
    0x566: "PKTSTATS_NOEXT_INSN2",
    0x567: "PKTSTATS_NOEXT_INSN3",
    0x568: "PKTSTATS_NOEXT_INSN4",
    0x569: "PKTSTATS_MEM_ST",
    0x56a: "PKTSTATS_MEM_LD",
    0x56b: "PKTSTATS_HMX_ST",
    0x56c: "PKTSTATS_HMX_LD",
    0x56d: "PKTSTATS_HMX_LD_LD",
    0x578: "TCU_WRITE_PORT_BLOCK",
    0x579: "TCU_EARLY_WRITE_PORT_BLOCK",
    0x57a: "TCU_WRITE_PORT_USAGE_0",
    0x57b: "TCU_WRITE_PORT_USAGE_1",
    0x57c: "TCU_WRITE_PORT_USAGE_2",
    0x57d: "TCU_WRITE_PORT_USAGE_3",
    0x57e: "TCU_WRITE_PORT_USAGE_4",
    0x57f: "TCU_WRITE_PORT_USAGE_5",
    0x580: "TCU_WRITE_PORT_USAGE_6",
    0x581: "TCU_WRITE_PORT_USAGE_7",
    0x582: "TCU_WRITE_PORT_USAGE_8",
    0x583: "TCU_WRITE_PORT_USAGE_MORE_THAN_8",
    0x584: "TCU_READ_PORT_USAGE_0",
    0x585: "TCU_READ_PORT_USAGE_1",
    0x586: "TCU_READ_PORT_USAGE_2",
    0x587: "TCU_READ_PORT_USAGE_3",
    0x588: "TCU_READ_PORT_USAGE_4",
    0x589: "TCU_READ_PORT_USAGE_5",
    0x58a: "TCU_READ_PORT_USAGE_6",
    0x58b: "TCU_READ_PORT_USAGE_7",
    0x58c: "TCU_READ_PORT_USAGE_8",
    0x58d: "TCU_READ_PORT_USAGE_MORE_THAN_8",
    0x58e: "TCU_FWD_NTWK_USAGE_0",
    0x58f: "TCU_FWD_NTWK_USAGE_1",
    0x590: "TCU_FWD_NTWK_USAGE_2",
    0x591: "TCU_FWD_NTWK_USAGE_3",
    0x592: "TCU_FWD_NTWK_USAGE_4",
    0x593: "TCU_FWD_NTWK_USAGE_5",
    0x594: "TCU_FWD_NTWK_USAGE_6",
    0x595: "TCU_FWD_NTWK_USAGE_7",
    0x596: "TCU_FWD_NTWK_USAGE_8",
    0x597: "TCU_FWD_NTWK_USAGE_MORE_THAN_8",
    0x598: "DC_FLUSHOP_COUNT",
    0x599: "MODEM_TRACE_HAD_MISMATCH",
    0x5a6: "L2FETCH_EVICTED",
    0x5a7: "HMX_MXFIFO_POP",
    0x5a8: "HMX_MAC_POP",
    0x5a9: "HMX_CONVERT_POP",
    0x5aa: "HMX_BUSY_CYCLES",
    0x5ab: "VTCM_FIFO_MINMAX_STALL_CYCLES",
    0x5ac: "HMXARRAY_MPY_ACTIVE",
    0x5ad: "HMXARRAY_MPY_INACTIVE",
    0x5ae: "HMXARRAY_MPY_NONZERO",
    0x5af: "HMXARRAY_MPY_ZERO",
    0x7d0: "COMMITTED_BYTE_LOADS",
    0x7d1: "COMMITTED_BYTE_STORES",
    0x7d2: "COMMITTED_BYTE_MEMOPS",
    0x7d3: "COMMITTED_HWORD_LOADS",
    0x7d4: "COMMITTED_HWORD_STORES",
    0x7d5: "COMMITTED_HWORD_MEMOPS",
    0x7d6: "COMMITTED_WORD_LOADS",
    0x7d7: "COMMITTED_WORD_STORES",
    0x7d8: "COMMITTED_WORD_MEMOPS",
    0x7d9: "COMMITTED_DWORD_MEMOPS",
    0x7da: "DMA_INSN_CYCLES",
    0x7db: "L2FIFO_LOAD_BYPASS_STORES",
    0x7dc: "XACT_SENT_TO_L2FIFO",
    0x7dd: "L2FIFO_LOAD_BYPASS_LOADS_AND_STORES",
    0x7df: "SMT_DISPATCH_WITHIN_CLUSTER",
    0x7e1: "CYCLES_0_PACKET_DISPATCHED",
    0x7e2: "CYCLES_1_PACKET_DISPATCHED",
    0x7e3: "CYCLES_2_PACKET_DISPATCHED",
    0x7e4: "CYCLES_3_PACKET_DISPATCHED",
    0x7e5: "CYCLES_4_PACKET_DISPATCHED",
    0x7e6: "simple_pkt_num_insns0",
    0x7e7: "simple_pkt_num_insns1",
    0x7e8: "simple_pkt_num_insns2",
    0x7e9: "simple_pkt_num_insns3",
    0x7ea: "simple_pkt_num_insns4",
    0x7eb: "cluster_num_insns0",
    0x7ec: "cluster_num_insns1",
    0x7ed: "cluster_num_insns2",
    0x7ee: "cluster_num_insns3",
    0x7ef: "cluster_num_insns4",
    0x7f0: "cluster_num_insns5",
    0x7f1: "cluster_num_insns6",
    0x7f2: "cluster_num_insns7",
    0x7f3: "cluster_num_insns8",
    0x7f6: "COPROC0_PKT_EXEC",
    0x7f7: "COPROC0_PKT_1_VMEM",
    0x7f8: "COPROC0_PKT_2_VMEM",
    0x7f9: "COPROC0_REPLAY",
    0x7fa: "COPROC0_IDLE",
    0x7fb: "COPROC0_AXISLAVE_ACCESS",
    0x7fc: "COPROC0_PKT_1_VEXTRACT",
    0x7fd: "COPROC0_PKT_2_VEXTRACT",
    0x7fe: "COPROC0_FIFO_DISPATCH",
    0x7ff: "COPROC0_REG_INTERLOCK_REPLAY",
    0x800: "COPROC0_MNOC_AXI_REPLAY",
    0x801: "COPROC0_RFIFO_REPLAY",
    0x802: "COPROC0_IU_REPLAY",
    0x803: "COPROC0_IU_L1S_REQUEST",
    0x804: "COPROC1_PKT_EXEC",
    0x805: "COPROC1_PKT_1_VMEM",
    0x806: "COPROC1_PKT_2_VMEM",
    0x807: "COPROC1_REPLAY",
    0x808: "COPROC1_IDLE",
    0x809: "COPROC1_PKT_1_VEXTRACT",
    0x80a: "COPROC1_PKT_2_VEXTRACT",
    0x80b: "COPROC1_FIFO_DISPATCH",
    0x80c: "COPROC1_REG_INTERLOCK_REPLAY",
    0x80d: "COPROC1_MNOC_AXI_REPLAY",
    0x80e: "COPROC1_RFIFO_REPLAY",
    0x80f: "CU_AUTOAND_INTERLOCK_CYCLES",
    0x818: "ICACHE_L2TCMHIT_DEMAND_MISS_CYCLES",
    0x819: "ICACHE_L2HIT_DEMAND_MISS",
    0x81a: "ICACHE_L2MISS_DEMAND_MISS",
    0x81b: "ICACHE_L2TCMHIT_DEMAND_MISS",
    0x81c: "ICACHE_L2ITCMHIT_DEMAND_MISS_CYCLES",
    0x81d: "ICACHE_L2ITCMHIT_DEMAND_MISS",
    0x81e: "HMXCVT_ACTIVE",
    0x81f: "HMXUMAC_ACTIVE",
    0x820: "HMXUMAC_OUTSTANDING",
    0x821: "HMX_OUTSTANDING",
    0x822: "HMXMAC_ACTIVE_NOCVT",
    0x823: "HMXCVT_ACTIVE_NOMAC",
    0x824: "HMX_CYCLES_RUNNING",
    0x829: "UARCH_TRACE_LD_DROPPED",
    0x82a: "UARCH_TRACE_ST_DROPPED",
    0x82b: "CU_TRACE_BUSY_CYCLES",
    0x82c: "DCACHE_DEMAND_MISS_L2HIT",
    0x82d: "DCACHE_DEMAND_MISS_TCM",
    0x82e: "DCACHE_DEMAND_MISS_AXI",
    0x82f: "DCACHE_DEMAND_MISS_AHB",
    0x830: "DCACHE_DEMAND_MISS_AXI2",
    0x831: "DCACHE_DEMAND_MISS_L2HIT_T0",
    0x832: "DCACHE_DEMAND_MISS_TCM_T0",
    0x833: "DCACHE_DEMAND_MISS_AXI_T0",
    0xbb8: "DCACHE_DEMAND_MISS_AHB_T0",
    0xbb9: "DCACHE_DEMAND_MISS_AXI2_T0",
    0xbba: "DCACHE_L2HIT_DEMAND_LOAD_MISS_CYCLES_T0",
    0xbbb: "DCACHE_TCM_DEMAND_LOAD_MISS_CYCLES_T0",
    0xbbc: "DCACHE_AXI_DEMAND_LOAD_MISS_CYCLES_T0",
    0xbbd: "DCACHE_AHB_DEMAND_LOAD_MISS_CYCLES_T0",
    0xbbe: "DCACHE_AXI2_DEMAND_LOAD_MISS_CYCLES_T0",
    0xbbf: "DU_CACHE_MISS_L2HIT_PVIEW_CYCLES",
    0xbc0: "DU_CACHE_MISS_TCM_PVIEW_CYCLES",
    0xbc1: "DU_CACHE_MISS_AXI_PVIEW_CYCLES",
    0xbc2: "DU_CACHE_MISS_AHB_PVIEW_CYCLES",
    0xbc3: "DU_CACHE_MISS_AXI2_PVIEW_CYCLES",
    0xbc4: "ICACHE_DEMAND_MISS_L2HIT_PRI",
    0xbc5: "ICACHE_DEMAND_MISS_L2MISS_PRI",
    0xbc6: "ICACHE_DEMAND_MISS_L2TCMHIT_PRI",
    0xbc7: "ICACHE_DEMAND_MISS_L2ITCMHIT_PRI",
    0xbc8: "ICACHE_IPREFETCHES_SENT_L2HIT",
    0xbc9: "ICACHE_IPREFECHES_SENT_L2MISS",
    0xbca: "ICACHE_IPREFETCHES_SENT_L2TCM",
    0xbcb: "ICACHE_IPREFETCHES_SENT_L2ITCM",
    0xbcd: "CYCLES_HVX_RUNNING",
    0xbce: "DU_SPF_AIA_DISPATCH",
    0xbcf: "DU_SPF_PCBASED_DISPATCH",
    0xbd0: "DU_SPF_ACCESS",
    0xbd1: "DU_SPF_ALLOC_FAIL_TABLE_FULL",
    0xbd2: "DU_SPF_ALLOC_FAIL_SHORT_LOOP",
    0xbd3: "DU_SPF_ACCESS_HIT",
    0xbd4: "DU_SPF_LINE_USED",
    0xbd5: "DU_SPF_LINE_EVICT_BEFORE_USE",
    0xbd6: "DU_SPF_ALLOC",
    0xbd7: "DU_SPF_ACCESS_FULL_THREAD",
    0xbd8: "DU_SPF_REQ_DROP_ANY",
    0xbd9: "DU_SPF_EMPTY_CYCLES",
    0xbda: "DU_SPF_OCCASION_DU_EXAM",
    0xbdb: "PACKETS_IN_LOOPS",
    0xbdc: "PACKETS_DELIVERED_BY_MINICACHE",
    0xbdd: "CU_NO_DISPATCH_3Memops",
    0xbde: "CU_NO_DISPATCH_3Stores",
    0xbdf: "CU_NO_DISPATCH_3Loads",
    0xbe0: "CU_NO_DISPATCH_slot0_ld_slot1_st",
    0xbe1: "CU_NO_DISPATCH_l2fifo",
    0xbe2: "DU_PST_USED",
    0xbe3: "DU_TAGPORT_CONFLICT_REPLAY",
    0xbe4: "DU_BANKPORT_CONFLICT_REPLAY",
    0xbe5: "DU_L2FIFOPORT_CONFLICT_REPLAY",
    0xbe6: "DU_TAGPORT_REQ",
    0xbe7: "DU_TAGPORT_REQ_CYCLE",
    0xbe8: "DU_TAGPORT_REQ_CUPKT_LOAD",
    0xbe9: "DU_TAGPORT_REQ_CUPKT_STORE",
    0xbea: "DU_TAGPORT_REQ_CUPKT_DCFETCH",
    0xbeb: "DU_TAGPORT_REQ_CUPKT_OTHER",
    0xbec: "DU_TAGPORT_REQ_L2FILL_LOAD",
    0xbed: "DU_TAGPORT_REQ_L2FILL_STORE",
    0xbee: "DU_TAGPORT_REQ_L2FILL_DCFETCH",
    0xbef: "DU_TAGPORT_REQ_L2FILL_OTHER",
    0xbf0: "DU_TAGPORT_REQ_L2FILL_SPF",
    0xbf1: "DU_TAGPORT_REQ_SPF",
    0xbf2: "DU_TAGPORT_REQ_OTHER",
    0xbf3: "ft_endloop0_minicache_hit",
    0xbf4: "ft_endloop0_minicache_miss",
    0xbf5: "COMMITTED_PKT_FPMAC_IMT3",
    0xbf6: "COMMITTED_PKT_FPMAC_IMT3_SLOTFLIP",
    0xbf7: "CU_NO_DISPATCH_slot0_duplex_ld_slot1_duplex_alu",
    0xbf8: "DU_SPF_ACCESS_IGNORED",
    0xbf9: "L2_DEVICE_TYPE_ACCESS",
    0xbfa: "CU_VSTORE_VTCM_DDR_SWTICH",
    0xbfb: "CU_VLOAD_VTCM_DDR_SWTICH",
    0xbfc: "CU_PKT_SMT_PIPELINE",
    0xbfd: "NO_ICSMT_FOR_NO_INTLV",
    0xbfe: "NO_ICSMT_FOR_ALU_FWD",
    0xbff: "NO_ICSMT_FOR_XUDU_FWD",
    0xc00: "NO_ICSMT_FOR_RC_ALU",
    0xc01: "NO_ICSMT_FOR_RC_XU",
    0xc02: "NO_ICSMT_FOR_RC_VMEM_L2",
    0xc03: "NO_ICSMT_FOR_RC_SCALAR_LDST",
    0xc04: "NO_ICSMT_FOR_RC_OTHER",
    0xc05: "CU_SMT_NO_INTLV_CYCLES",
    0xc06: "CU_SMT_NO_ALU_FWD_CYCLES",
    0xc07: "CU_SMT_NO_XUDU_FWD_CYCLES",
    0xc08: "CU_SMT_RC_ALU_CYCLES",
    0xc09: "CU_SMT_RC_XU_CYCLES",
    0xc0a: "CU_SMT_RC_VMEM_L2_CYCLES",
    0xc0b: "CU_SMT_RC_SCALAR_LDST_CYCLES",
    0xc0c: "CU_SMT_RC_OTHER_CYCLES",
    0xc0d: "NUM_COMMITTED_PAUSE",
    0xc0e: "NUM_COMMITTED_UNPAUSE",
    0xc0f: "WAKEUP_UNPAUSE",
    0xc10: "WAKEUP_PAUSE_EXPIRE",
    0xc11: "CONDITIONAL_DOTOLD_LOAD",
    0xc12: "CONDITIONAL_DOTNEW_LOAD",
    0xc13: "PREDICATE_CANCELLED_DOTOLD_LOAD",
    0xc14: "PREDICATE_CANCELLED_DOTNEW_LOAD",
    0xc15: "CONDITIONAL_DOTOLD_STORE",
    0xc16: "CONDITIONAL_DOTNEW_STORE",
    0xc17: "PREDICATE_CANCELLED_DOTOLD_STORE",
    0xc18: "PREDICATE_CANCELLED_DOTNEW_STORE",
    0xc19: "CU_PVIEW_BE_NOB2B",
    0xc1a: "CU_PVIEW_BE_NOBSB",
    0xc1b: "CU_PVIEW_BE_NO_IMT3_NCJ",
    0xc1c: "CU_PVIEW_CREG_INTERLOCK",
    0xc1d: "CU_PVIEW_PREG_INTERLOCK",
    0xc1e: "CU_PVIEW_WRITE_REG_BLOCK",
    0xc1f: "CU_PVIEW_NON_LAST_SUBPKT",
    0xc20: "CU_PVIEW_BE",
    0xc21: "CU_PVIEW_RXX_INTERLOCK",
    0xc22: "CU_PVIEW_AUTOAND_INTERLOCK",
    0xc23: "CU_PVIEW_DUAL_WRITE_INTERLOCK",
    0xc24: "CU_PVIEW_FP_GRANDPARENT_BLOCK",
    0xc25: "CU_PVIEW_FP_RX_NO_NTWK",
    0xc26: "CU_PVIEW_DU_XU_NO_FWD",
    0xc27: "CU_PVIEW_REVERSED_PREDICATE",
    0xc28: "CU_PVIEW_LATE_READ",
    0xc29: "CU_PVIEW_EARLY_WRITE",
    0xc2a: "CU_PVIEW_RC_SOLO",
    0xc2b: "CU_PVIEW_NO_DISPATCH",
    0xc2c: "CU_PVIEW_WRITE_PORT_BLOCK",
    0xc2d: "L2_CONV2UNCACHED",
    0xc2e: "COMMITTED_STORE_PREDICATED",
    0xc2f: "COMMITTED_STORE_PREDICATED_CANCELLED",
    0xc30: "COMMITTED_L2FETCH",
    0xc31: "L2_DU_READ_ACCESS_NONAXI",
    0xc32: "L2_DU_READ_ACCESS_UNCACHED",
    0xc33: "L2_DU_READ_ETC",
    0xc34: "L2_DU_STORE_ACCESS_NONAXI",
    0xc35: "L2_DU_STORE_MISS_NONAXI",
    0xc36: "L2_DU_STORE_ACCESS_UNCACHED",
    0xc37: "L2_DU_STORE_ACCESS_ETC",
    0xc38: "CU_HMX_LATE_READ_PKT",
    0xc39: "lc0_pred_table_hit",
    0xc3a: "lc0_pred_table_miss",
    0xc3b: "lc0_pred_table_write",
    0xc3c: "lc0_pred_table_update",
    0xc3d: "lc0_pred_table_clear_entry",
    0xc3e: "fe_occurrence",
    0xc3f: "CU_DU_PAUSE_CYCLES",
    0xc40: "CLADE2_REQ",
    0xc41: "CLADE2_REQ_REG",
    0xc42: "CLADE2_INBUF_BACKPRESSURE",
    0xc5f: "CLADE2_DECOMPRESSORS_FULL_CYCLES",
    0xc60: "CLADE2_COMPRESSORS_FULL_CYCLES",
    0xc61: "CLADE2_READ_GUARD_CONFLICT",
    0xc62: "CLADE2_WRITE_GUARD_CONFLICT",
    0xc63: "CLADE2_AXIM3_48B_READ",
    0xc64: "CLADE2_AXIM3_48B_WRITE",
    0xc65: "CLADE2_AXIM3_16B_READ",
    0xc66: "CLADE2_AXIM3_16B_WRITE",
    0xc67: "CLADE2_AXIM3_8B_READ",
    0xc68: "CLADE2_AXIM3_8B_WRITE",
    0xc69: "CLADE2_METADATA_16B_READ",
    0xc6a: "CLADE2_METADATA_16B_WRITE",
    0xc6b: "CLADE2_METADATA_8B_READ",
    0xc6c: "CLADE2_METADATA_8B_WRITE",
    0xc6d: "CLADE2_MDC_BACKPRESSURE",
    0xc6e: "MDC_BUS_BACKPRESSURE",
    0xc6f: "BUS_MDC_BACKPRESSURE",
    0xc70: "CLADE2_MDC_ACCESS_WRITE",
    0xc71: "CLADE2_MDC_MISS",
    0xc72: "VTCM_REQ_NO_BUFF_NO_CRED",
    0xc73: "VTCM_REQ_NO_BUFF",
    0xc74: "VTCM_REQ_NO_CRED",
    0xc75: "LOOPCACHE_PACKETS",
    0xc76: "CU_NO_DISPATCH_l2fifo_throttle",
    0xc77: "CU_NO_DISPATCH_solo",
    0xc78: "CU_PVIEW_ICSMT_SMT_NO_INTLV",
    0xc79: "CU_PVIEW_ICSMT_SMT_NO_ALU_FWD",
    0xc7a: "CU_PVIEW_ICSMT_SMT_NO_XUDU_FWD",
    0xc7b: "CU_PVIEW_ICSMT_SMT_RC_ALU",
    0xc7c: "CU_PVIEW_ICSMT_SMT_RC_XU",
    0xc7d: "CU_PVIEW_ICSMT_SMT_RC_VMEM_L2",
    0xc7e: "CU_PVIEW_ICSMT_SMT_RC_SCALAR_LDST",
    0xc7f: "CU_PVIEW_ICSMT_SMT_RC_OTHER",
    0xc80: "COPROC_EXTSILVER_EXTRACT_CYCLES",
    0xc81: "SILVER_NOT_READY_CYCLES",
    0xc82: "COPROC_EXTSILVER_SCALAR_L1S_CYCLES",
    0xc84: "CU_L2FIFO_THROTTLE_CYCLES",
    0xf00: "CLADE2_DISABLE",
    0xf01: "CLADE2_COUNTER0_OVERFLOW",
    0xf02: "CLADE2_REQ_READ",
    0xf03: "CLADE2_REQ_WRITE",
    0xf04: "CLADE2_DECOMPRESSION",
    0xf05: "CLADE2_COMPRESSION",
    0xf06: "CLADE2_ZERO_DECOMPRESSION",
    0xf07: "CLADE2_ZERO_COMPRESSION",
    0xf0a: "CLADE2_FREE_LIST_16_DDR_READS",
    0xf0b: "CLADE2_FREE_LIST_16_DDR_STORES",
    0xf0c: "CLADE2_FREE_LIST_32_DDR_READS",
    0xf0d: "CLADE2_FREE_LIST_32_DDR_STORES",
    0xf0e: "CLADE2_FREE_LIST_48_DDR_READS",
    0xf0f: "CLADE2_FREE_LIST_48_DDR_STORES",
    0xf10: "CLADE2_FREE_LIST_64_DDR_READS",
    0xf11: "CLADE2_FREE_LIST_64_DDR_STORES",
    0xf12: "CLADE2_RMW",
    0xf13: "CLADE2_DEVICE_ACCESS",
    0xf14: "CLADE2_STALL_MASTER_IF",
    0xf15: "CLADE2_STALL_SLAVE_IF",
    0xf16: "CLADE2_64B_READ",
    0xf17: "CLADE2_64B_WRITE",
    0xf18: "CLADE2_48B_READ",
    0xf19: "CLADE2_48B_WRITE",
    0xf1a: "CLADE2_32B_READ",
    0xf1b: "CLADE2_32B_WRITE",
    0xf1c: "CLADE2_16B_READ",
    0xf1d: "CLADE2_16B_WRITE",
    0xf1e: "CLADE2_MDC_ACCESS",
    0xf1f: "CLADE2_MDC_ACCESS_READ",
    0xf20: "CLADE2_MDC_HIT",
    0xf21: "CLADE2_MDC_HIT_READ",
    0xf22: "CLADE2_MDC_HIT_SECONDARY",
    0xf23: "CLADE2_MDC_BLOCKING",
    0xf24: "CLADE2_MDC_DIRTY_EVICT",
    0xf25: "CLADE2_MDC_REPLAY",
    0xf26: "CLADE2_MDC_CYCLES",
}

PMU_EVENT_OVERRIDES: dict[int, dict[int, str | None]] = {
    75: {
        0x024: "LOOPCACHE_PACKETS",
        0x109: "HVX_VOLTAGE_VIRUS_OVER",
        0x114: "HVX_MAX_VOLT_UNDERSHOOT",
        0x11f: "HVXST_L2_CONFLICT",
        0x120: "HVXST_VTCM_CONFLICT",
        0x169: "VTCMSLD_CONFLICT",
        0x16c: "VTCMOLDEST_NETBOARD_FULL",
        0x191: None,
        0x192: None,
        0x208: None,
        0x215: "HMXCVT_ORDER",
        0x217: "HMXCVT_LD_OUTSTANDING",
        0x219: None,
        0x21d: "HMXCVT_LD",
        0x220: "HMXCVT_ST",
        0x285: "HMXCVT_WR",
        0x296: None,
        0x297: None,
        0x298: None,
        0x299: None,
        0x29a: None,
        0x29b: None,
        0x29c: None,
        0x29d: None,
        0x29e: None,
        0x29f: None,
        0x2a0: None,
        0x2a1: None,
        0x2a2: None,
        0x2a3: None,
        0x2a4: None,
        0x2a5: None,
        0x2a6: None,
        0x2a7: None,
        0x2a8: None,
        0x31c: "SMT_PKT_PICKED_BUT_NOT_COMMIT_PVIEW_CYCLES",
        0x31d: "SMT_PKT_IQ_NO_PKT_PVIEW_CYCLES",
        0x31e: "SMT_PKT_NOT_SIMPLE_PVIEW_CYCLES",
        0x31f: "SMT_PKT_NOT_READY_PVIEW_CYCLES",
        0x320: "SMT_PKT_SLOT_CONFLICT_PVIEW_CYCLES",
        0x321: "SMT_PKT_REG_FWD_BLOCK_PVIEW_CYCLES",
        0x383: None,
        0x384: None,
        0x385: None,
        0x386: None,
        0x387: None,
        0x388: None,
        0x389: None,
        0x38a: None,
        0x38b: None,
        0x38c: None,
        0x38d: None,
        0x38e: None,
        0x38f: None,
        0x390: None,
        0x391: None,
        0x392: None,
        0x393: None,
        0x394: None,
        0x395: None,
        0x557: None,
        0xc40: None,
        0xc41: None,
        0xc42: None,
        0xc5f: None,
        0xc60: None,
        0xc61: None,
        0xc62: None,
        0xc63: None,
        0xc64: None,
        0xc65: None,
        0xc66: None,
        0xc67: None,
        0xc68: None,
        0xc69: None,
        0xc6a: None,
        0xc6b: None,
        0xc6c: None,
        0xc6d: None,
        0xc6e: None,
        0xc6f: None,
        0xc70: None,
        0xc71: None,
        0xc72: None,
        0xc73: None,
        0xc74: None,
        0xc75: None,
        0xc76: None,
        0xc77: None,
        0xc78: None,
        0xc79: None,
        0xc7a: None,
        0xc7b: None,
        0xc7c: None,
        0xc7d: None,
        0xc7e: None,
        0xc7f: None,
        0xc80: None,
        0xc81: None,
        0xc82: None,
        0xc84: None,
        0xf00: None,
        0xf01: None,
        0xf02: None,
        0xf03: None,
        0xf04: None,
        0xf05: None,
        0xf06: None,
        0xf07: None,
        0xf0a: None,
        0xf0b: None,
        0xf0c: None,
        0xf0d: None,
        0xf0e: None,
        0xf0f: None,
        0xf10: None,
        0xf11: None,
        0xf12: None,
        0xf13: None,
        0xf14: None,
        0xf15: None,
        0xf16: None,
        0xf17: None,
        0xf18: None,
        0xf19: None,
        0xf1a: None,
        0xf1b: None,
        0xf1c: None,
        0xf1d: None,
        0xf1e: None,
        0xf1f: None,
        0xf20: None,
        0xf21: None,
        0xf22: None,
        0xf23: None,
        0xf24: None,
        0xf25: None,
        0xf26: None,
    },
    81: {
        0x059: "CU_GLBL_CR_WR_REPLAY",
        0x0f0: "OTHERTILE_BUSY_PVIEW_CYCLES",
        0x105: "HVX_LD_VTCM_OUTSTANDING",
        0x106: "HVX_OTHER_CTX_OUTSTANDING",
        0x10d: "HVX_WARMUP",
        0x10e: "HVX_ST_VTCM_OUTSTANDING",
        0x10f: "HVX_SCATGATH_OUTSTANDING",
        0x110: "HVX_EMPTY",
        0x142: None,
        0x193: "CYCLES_7_HVX_CONTEXTS_RUNNING",
        0x194: "CYCLES_8_HVX_CONTEXTS_RUNNING",
        0x195: "HVXLDSCFIFO_BEATBUF_FULL",
        0x196: "HVX_WAIT",
        0x1c5: "COMMITTED_PKT_T8",
        0x1c6: "COMMITTED_PKT_T9",
        0x1c7: "COMMITTED_PKT_T10",
        0x1c8: "COMMITTED_PKT_T11",
        0x1c9: "COMMITTED_PKT_T12",
        0x1ca: "COMMITTED_PKT_T13",
        0x1cb: "COMMITTED_PKT_T14",
        0x1cc: "COMMITTED_PKT_T15",
        0x1cd: "CYCLES_9_THREAD_RUNNING",
        0x1ce: "CYCLES_10_THREAD_RUNNING",
        0x1cf: "CYCLES_11_THREAD_RUNNING",
        0x1d0: "CYCLES_12_THREAD_RUNNING",
        0x1d1: "CYCLES_13_THREAD_RUNNING",
        0x1d2: "CYCLES_14_THREAD_RUNNING",
        0x1d3: "CYCLES_15_THREAD_RUNNING",
        0x1d4: "CYCLES_16_THREAD_RUNNING",
        0x1d5: "COMMITTED_PKT_9_THREAD_RUNNING",
        0x1d6: "COMMITTED_PKT_10_THREAD_RUNNING",
        0x1d7: "COMMITTED_PKT_11_THREAD_RUNNING",
        0x1d8: "COMMITTED_PKT_12_THREAD_RUNNING",
        0x1d9: "COMMITTED_PKT_13_THREAD_RUNNING",
        0x1da: "COMMITTED_PKT_14_THREAD_RUNNING",
        0x1db: "COMMITTED_PKT_15_THREAD_RUNNING",
        0x1dc: "COMMITTED_PKT_16_THREAD_RUNNING",
        0x1dd: "COMMITTED_PKT_UNBALANCED",
        0x1de: "CYCLES_UNBALANCED",
        0x20b: None,
        0x216: "HMXCVT_FB_ORDER",
        0x22d: "HMXRDWGT_BEATBUF_FULL",
        0x22e: "HMXRDWGT_BUF_FULL",
        0x296: None,
        0x29c: None,
        0x2a3: None,
        0x2a9: "HMXACDMND_THROTTLE",
        0x2aa: "HMXMAC_WARMUP",
        0x2ab: "HMXMAC_RAMP_UP",
        0x2ac: "HMXMAC_RAMP_DOWN",
        0x2ad: "HMX_LIMITS_THROTTLE_TLMH_SLEW",
        0x2ae: "HMXCVT_TP_CONFLICT",
        0x2af: "HMXPVM_NOTOVER_UNREADY",
        0x2b0: "HMXPVM_OVER_EMPTY",
        0x2b1: "HMXPVM_OVER_UNREADY",
        0x2b2: "HMXPVM_RAMP_UP",
        0x2b3: "HMXPVM_RAMP_DOWN",
        0x2b4: "HMXPVM_OVER_ISSUE",
        0x2b5: "HMXPVM_OVERPEAK_ISSUE",
        0x2b6: "HMXPVM_ISSUE",
        0x2b7: "HMXRDACT_BEATBUF_FULL",
        0x2b8: "HMXCVTWR_BEATBUF_FULL",
        0x2b9: "IU_DROP_PKT",
        0x2ba: "IU_ONEWAY_CACHE_RD",
        0x2bb: "IU_TWOWAY_CACHE_RD",
        0x2bc: "HMXMAC_EMPTY",
        0x2bd: "HMXMAC_WAIT",
        0x2be: None,
        0x2c1: None,
        0x2c2: None,
        0x2c3: None,
        0x2c4: None,
        0x2c5: None,
        0x2c6: None,
        0x2c7: None,
        0x2c8: None,
        0x2c9: None,
        0x2ca: None,
        0x2cb: None,
        0x2cc: None,
        0x2cd: None,
        0x2ce: None,
        0x2cf: None,
        0x2d0: None,
        0x2d1: None,
        0x2d2: None,
        0x2d3: None,
        0x2d4: None,
        0x2d5: None,
        0x2d6: None,
        0x2d7: None,
        0x2d8: None,
        0x2d9: None,
        0x2da: None,
        0x2dd: None,
        0x2de: None,
        0x2df: None,
        0x2e0: None,
        0x2e1: None,
        0x2e2: None,
        0x2e3: None,
        0x2e4: None,
        0x2e5: None,
        0x2e6: None,
        0x2e7: None,
        0x2e8: None,
        0x2e9: None,
        0x2ea: None,
        0x2eb: None,
        0x2ec: None,
        0x2ed: None,
        0x2ee: None,
        0x2ef: None,
        0x2f0: None,
        0x2f1: None,
        0x2f2: None,
        0x2f3: None,
        0x2f4: None,
        0x2f5: None,
        0x2f6: None,
        0x2f7: None,
        0x2f8: None,
        0x2f9: None,
        0x2fa: None,
        0x308: None,
        0x309: None,
        0x30a: None,
        0x30b: None,
        0x30c: None,
        0x30d: None,
        0x30e: None,
        0x30f: None,
        0x310: None,
        0x311: None,
        0x312: None,
        0x31d: "APP_REPORTED",
        0x32b: None,
        0x32c: None,
        0x396: "TOTAL_SNOOP_BLOCK",
        0x397: "DU_STORECOND_FAIL",
        0x398: "DU_TLSNP_FILTER_HIT_COUNT",
        0x399: "DU_TLSNP_DCACHE_HIT_COUNT",
        0x39a: "DU_L2FIFO_DRAINDELAY_ORDERING",
        0x39b: "DU_L2FIFO_DRAINDELAY_TLCREDIT",
        0x39c: "DU_L2FIFO_DRAINDELAY_SYNCHOLD",
        0x39d: "DU_DEMAND_FILL_DRAINDELAY",
        0x39e: "DU_LDLCK_WAITON_OTHTL_STCND",
        0x39f: "L2_TAG_CONF_NO_DATA_CONF",
        0x3a0: "ET_TILE1_C0_FIFO_OVERFLOW",
        0x3a1: "ET_TILE1_C1_FIFO_OVERFLOW",
        0x3a2: "ET_TILE1_C0_ETB_FULL",
        0x3a3: "ET_TILE1_C1_ETB_FULL",
        0x3a4: "ET_TILE1_C0_ETB_OVER_HALF",
        0x3a5: "ET_TILE1_C1_ETB_OVER_HALF",
        0x3a6: "ET_TILE1_C0_ETB_OVER_THREE_FOURTHS",
        0x3a7: "ET_TILE1_C1_ETB_OVER_THREE_FOURTHS",
        0x3a8: "L2_PMU_DUStMergeReq_ANY",
        0x3a9: "ICSMT_OTHERTILE_BUSY_PVIEW_CYCLES",
        0x3aa: "THREAD_LMH_THROTTLE_T8",
        0x3ab: "THREAD_LMH_THROTTLE_T9",
        0x3ac: "THREAD_LMH_THROTTLE_T10",
        0x3ad: "THREAD_LMH_THROTTLE_T11",
        0x3ae: "THREAD_LMH_SLEW_THROTTLE",
        0x3af: "JU_SHADOW_SEARCH",
        0x3b0: "DCACHE_DEMAND_PRIMARY_MISS_INALOOP",
        0x3b1: "DCACHE_DEMAND_SECONDARY_MISS_INALOOP",
        0x3b2: "DU_SAMEPKT_BANK_CONFLICT_INALOOP",
        0x3b3: "DU_SMT_BANK_CONFLICT_INALOOP",
        0xc83: "CU_L2FIFO_THROTTLE_CYCLES",
        0xc84: "HMXCVTBIAS_EMPTY",
        0xc85: "HMXCVTMEM_EMPTY",
        0xc86: "HMXCVTINTBUF_EMPTY",
        0xc87: "JU_TOTAL_REQ",
        0xc88: "JU_DROPPED_REQ",
        0xc89: "JU_CANCELLED_REQ",
        0xc8a: "HMXMAC_MULT_DROP",
        0xc8b: "DCACHE_TAGR_MISS",
        0xc8c: "DCACHE_SNOOP_DUALTILE",
        0xc8d: "DU_STAB_MATCH_CYCLES",
        0xc8e: "HVX_VTCM_OUTSTANDING",
    },
}


def pmu_event_names(arch: int | None) -> dict[int, str]:
    """Give the dict from a PMU event id to its name for one Hexagon architecture.

    Args:
        arch: The architecture number (75, 79 or 81). None and an unknown number
            give the v79 table

    Returns:
        A new dict, the v79 table with the overrides of the architecture applied
    """
    names = dict(PMU_EVENTS_V79)
    for event_id, name in PMU_EVENT_OVERRIDES.get(arch or DEFAULT_ARCH, {}).items():
        if name is None:
            names.pop(event_id, None)
        else:
            names[event_id] = name
    return names


def pmu_counter_labels(events: Sequence[int], arch: int | None) -> list[str]:
    """Give one label for each PMU event id, the name when it is known.

    Args:
        events: The event ids in counter order
        arch: The architecture number for the name table

    Returns:
        One label for each id, ``NAME`` or ``0x1ab`` when the id has no name
    """
    names = pmu_event_names(arch)
    return [names.get(event_id, f"0x{event_id:03x}") for event_id in events]


@dataclass
class Op:
    """One ``profile-op`` line of a batch."""

    name: str
    names: str
    dims: str
    types: str
    strides: str
    kparams: str
    usec: int
    cycles: int
    start: int
    mhz: float
    pmu: tuple[int, ...] | None
    index: int = 0
    abs_cycles: int = 0

    def duration_us(self, batch_mhz: float) -> float:
        """Give the time of the op in microseconds from its cycles and the batch clock."""
        return self.cycles / batch_mhz

    @property
    def path(self) -> str:
        """Give the kernel path word of the kernel parameters, or an empty string."""
        if self.kparams in ("", "----"):
            return ""
        return self.kparams.split()[0]


@dataclass
class TraceEvent:
    """One ``trace-evt`` line of a batch."""

    name: str
    thread: int
    info: int
    stop: bool
    cycles: int
    abs_cycles: int = 0


@dataclass
class Phase:
    """One start/stop pair of trace events on one thread."""

    name: str
    thread: int
    info: int
    start_cycles: int
    end_cycles: int
    note: str = ""


@dataclass
class Batch:
    """One ``OPBATCH`` line with its ops and its trace events."""

    session: str
    seq: int
    n_ops: int
    usec: int
    cycles: int
    start: int
    mhz: float
    evt_cnt: tuple[int, ...] | None
    host_us: float | None
    ops: list[Op] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    start_us: float = 0.0

    @property
    def clock_mhz(self) -> float:
        """Give the batch clock, or 1.0 when the batch has no time (this prevents a division by zero)."""
        return self.mhz if self.mhz > 0 else 1.0

    @property
    def op_time_us(self) -> float:
        """Give the sum of the op times in microseconds, from the cycles."""
        return sum(op.cycles for op in self.ops) / self.clock_mhz

    @property
    def op_usec_sum(self) -> int:
        """Give the sum of the integer ``usec`` fields of the ops."""
        return sum(op.usec for op in self.ops)

    @property
    def gap_us(self) -> float:
        """Give the batch time minus the sum of the op times."""
        return self.usec - self.op_time_us

    def offset_us(self, abs_cycles: int) -> float:
        """Give the microseconds between the batch start and a counter value of the batch."""
        return (abs_cycles - self.start) / self.clock_mhz

    def add_op(self, op: Op) -> None:
        """Add an op to the batch and give it the 64-bit counter value of its start."""
        previous = self.ops[-1].abs_cycles if self.ops else self.start
        op.index = len(self.ops)
        op.abs_cycles = unwrap_cycles(previous, op.start)
        self.ops.append(op)

    def add_event(self, event: TraceEvent, last_by_thread: dict[int, int]) -> None:
        """Add a trace event to the batch and give it the 64-bit counter value.

        Args:
            event: The event, with the low 32 bits of the counter in ``cycles``
            last_by_thread: The last counter value of each thread, updated here
        """
        previous = last_by_thread.get(event.thread, self.start)
        event.abs_cycles = unwrap_cycles(previous, event.cycles)
        last_by_thread[event.thread] = event.abs_cycles
        self.events.append(event)


@dataclass
class Session:
    """The batches of one DSP session, in log order."""

    name: str
    batches: list[Batch] = field(default_factory=list)


@dataclass
class ProfileLog:
    """The content of one profile log."""

    path: str
    sessions: dict[str, Session] = field(default_factory=dict)
    level: int = 0
    mode: int | None = None
    pmu_events: tuple[int, ...] = DEFAULT_PMU_EVENTS
    arch: int | None = None
    clock: str = "cycles"
    orphan_ops: int = 0
    orphan_events: int = 0
    bad_lines: int = 0

    def selected(self, batches: slice | None) -> list[tuple[Session, list[Batch]]]:
        """Give the batches of each session that a slice keeps.

        Args:
            batches: A slice of the batches of each session, or None for all

        Returns:
            One (session, batches) pair for each session in log order
        """
        return [(session, session.batches[batches] if batches else list(session.batches)) for session in self.sessions.values()]


def unwrap_cycles(previous: int, low32: int) -> int:
    """Give the 64-bit counter value that follows ``previous`` and has the low 32 bits ``low32``.

    The result is in the range ``previous`` thru ``previous + 2^32 - 1``.
    """
    return previous + ((low32 - previous) & CYCLE_MASK)


def _host_us(match: re.Match[str]) -> float | None:
    """Give the log timestamp of a line in microseconds, or None when the line has no timestamp."""
    if match.group("min") is None:
        return None
    minutes, seconds = int(match.group("min")), int(match.group("sec"))
    return ((minutes * 60 + seconds) * 1000 + int(match.group("ms"))) * 1000.0 + int(match.group("us"))


def _parse_timing(text: str) -> tuple[int, int, int, float, tuple[int, ...] | None] | None:
    """Parse the ``usec N cycles N start N mhz F [pmu [...]]`` end of a profile line."""
    match = _TIMING_RE.search(text)
    if not match:
        return None
    pmu_text = match.group("pmu")
    pmu = tuple(int(value) for value in pmu_text.split(",")) if pmu_text else None
    return int(match.group("usec")), int(match.group("cycles")), int(match.group("start")), float(match.group("mhz")), pmu


def _parse_evt_cnt(text: str) -> tuple[int, ...] | None:
    """Parse the ``evt-cnt a,b,...`` field of an ``OPBATCH`` line, or give None for ``----``."""
    if not text.startswith("evt-cnt "):
        return None
    return tuple(int(value) for value in text[len("evt-cnt "):].split(","))


def parse_lines(lines: Iterable[str], path: str = "-") -> ProfileLog:
    """Parse the profile lines of a log of level 1, 2 or 3.

    The parser keeps the lines in log order: an op line and a trace line are part of
    the last ``OPBATCH`` line of the same session. Complexity O(n) in the number of
    lines.

    Args:
        lines: The lines of the log
        path: The name of the log for the reports

    Returns:
        The parsed log, with the batches on the timeline (refer to ``place_batches``)
    """
    log = ProfileLog(path=path)
    current: dict[str, Batch] = {}
    last_event_cycles: dict[str, dict[int, int]] = defaultdict(dict)
    has_pmu = False
    has_trace = False

    for line in lines:
        match = _LINE_RE.search(line)
        if not match:
            mode = _MODE_RE.search(line)
            if mode:
                log.mode = int(mode.group("mode"))
                events = [int(value, 0) for value in mode.group("events").replace(" ", "").split(",") if value]
                if len(events) == len(DEFAULT_PMU_EVENTS):
                    log.pmu_events = tuple(events)
            arch = _ARCH_RE.search(line)
            if arch and log.arch is None:
                log.arch = int(arch.group("arch"))
            continue

        session_name = match.group("session")
        rest = match.group("rest")
        if match.group("kind") == "trace-evt":
            trace = _TRACE_RE.match(rest)
            batch = current.get(session_name)
            if not trace:
                log.bad_lines += 1
            elif batch is None:
                log.orphan_events += 1
            else:
                has_trace = True
                event = TraceEvent(
                    name=trace.group("name"),
                    thread=int(trace.group("thread")),
                    info=int(trace.group("info")),
                    stop=trace.group("state") == "stop",
                    cycles=int(trace.group("cycles")),
                )
                batch.add_event(event, last_event_cycles[session_name])
            continue

        parts = rest.split("|")
        timing = _parse_timing(parts[-1]) if len(parts) == 7 else None
        if timing is None:
            log.bad_lines += 1
            continue
        usec, cycles, start, mhz, pmu = timing
        op_name = parts[0]
        if op_name == "OPBATCH":
            session = log.sessions.setdefault(session_name, Session(session_name))
            n_ops = int(parts[2].split()[1]) if parts[2].startswith("n-ops ") else 0
            evt_cnt = _parse_evt_cnt(parts[3])
            batch = Batch(
                session=session_name,
                seq=len(session.batches),
                n_ops=n_ops,
                usec=usec,
                cycles=cycles,
                start=start,
                mhz=mhz,
                evt_cnt=evt_cnt,
                host_us=_host_us(match),
            )
            session.batches.append(batch)
            current[session_name] = batch
            last_event_cycles[session_name] = {}
            has_trace = has_trace or evt_cnt is not None
            continue

        batch = current.get(session_name)
        if batch is None:
            log.orphan_ops += 1
            continue
        has_pmu = has_pmu or pmu is not None
        batch.add_op(Op(op_name, parts[1], parts[2], parts[3], parts[4], parts[5], usec, cycles, start, mhz, pmu))

    log.level = 3 if has_trace else 2 if has_pmu else 1 if log.sessions else 0
    log.clock = place_batches(log)
    return log


def parse_file(path: str | Path) -> ProfileLog:
    """Parse a profile log from a file, or from stdin when the path is ``-``."""
    if str(path) == "-":
        return parse_lines(sys.stdin, "-")
    with open(path, encoding="utf-8", errors="replace") as handle:
        return parse_lines(handle, str(path))


def place_batches(log: ProfileLog) -> str:
    """Set the timeline start of each batch in microseconds.

    The host clock is used when each batch has a log timestamp: the batch start is
    the timestamp minus ``usec``, and the earliest batch is at 0. Without timestamps
    the cycle counter of each session is used, converted with the batch ``mhz``.

    Returns:
        ``"host"`` or ``"cycles"``, the clock that was used
    """
    batches = [batch for session in log.sessions.values() for batch in session.batches]
    if batches and all(batch.host_us is not None for batch in batches):
        origin = min(batch.host_us - batch.usec for batch in batches if batch.host_us is not None)
        for batch in batches:
            assert batch.host_us is not None
            batch.start_us = batch.host_us - batch.usec - origin
        return "host"
    for session in log.sessions.values():
        if not session.batches:
            continue
        origin = session.batches[0].start
        for batch in session.batches:
            batch.start_us = (batch.start - origin) / batch.clock_mhz
    return "cycles"


def pair_phases(batch: Batch) -> tuple[list[Phase], int, int]:
    """Make phases from the start and stop trace events of a batch.

    A stop connects to the newest open start of the same thread with the same name
    and the same info. When no such start is open, it connects to the newest open
    start of the same name, because some kernels give a different info to the stop.
    Complexity O(n * k), with k the number of open starts on a thread (small).

    Returns:
        The phases, the number of starts without a stop, and the number of stops
        without a start. An event without a pair becomes a phase of zero length with
        a note.
    """
    phases: list[Phase] = []
    open_starts: dict[int, list[TraceEvent]] = defaultdict(list)
    unpaired_stops = 0
    for event in batch.events:
        opens = open_starts[event.thread]
        if not event.stop:
            opens.append(event)
            continue
        found = _find_open(opens, event.name, event.info)
        if found is None:
            unpaired_stops += 1
            phases.append(Phase(event.name, event.thread, event.info, event.abs_cycles, event.abs_cycles, "stop without start"))
            continue
        start = opens.pop(found)
        phases.append(Phase(event.name, event.thread, start.info, start.abs_cycles, event.abs_cycles))
    unpaired_starts = 0
    for opens in open_starts.values():
        for start in opens:
            unpaired_starts += 1
            phases.append(Phase(start.name, start.thread, start.info, start.abs_cycles, start.abs_cycles, "start without stop"))
    return phases, unpaired_starts, unpaired_stops


def _find_open(opens: list[TraceEvent], name: str, info: int) -> int | None:
    """Give the index of the newest open start with the name and the info, else with the name only."""
    for index in range(len(opens) - 1, -1, -1):
        if opens[index].name == name and opens[index].info == info:
            return index
    for index in range(len(opens) - 1, -1, -1):
        if opens[index].name == name:
            return index
    return None


def thread_name(thread: int) -> str:
    """Give the track name of a DSP thread."""
    if thread == HMX_THREAD:
        return "hmx queue"
    if thread == 0:
        return "dsp t0 (main)"
    return f"dsp t{thread}"


def _round(value: float) -> float:
    """Give a time with three decimals, because Perfetto keeps nanoseconds."""
    return round(value, 3)


def _metadata(pid: int, tid: int | None, name: str, sort_index: int) -> list[JsonDict]:
    """Give the Chrome metadata events with the name and the order of a process or a thread."""
    if tid is None:
        return [
            {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}},
            {"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": sort_index}},
        ]
    return [
        {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}},
        {"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": sort_index}},
    ]


def chrome_events(log: ProfileLog, batches: slice | None = None, arch: int | None = None) -> list[JsonDict]:
    """Build the Chrome trace events of a log.

    Args:
        log: The parsed log
        batches: The slice of the batches of each session to keep, or None for all
        arch: The architecture for the PMU names, or None for the one of the log

    Returns:
        The events, in the order of the sessions, the batches and the ops
    """
    labels = pmu_counter_labels(log.pmu_events, arch or log.arch)
    events: list[JsonDict] = []
    for pid, (session, kept) in enumerate(log.selected(batches), start=1):
        events += _metadata(pid, None, session.name, pid)
        events += _metadata(pid, TID_BATCHES, "batches", 0)
        events += _metadata(pid, TID_OPS, "ops", 1)
        threads_seen: set[int] = set()
        dma_seen: set[int] = set()
        for batch in kept:
            events.append(_batch_event(pid, batch))
            for op in batch.ops:
                ts = batch.start_us + batch.offset_us(op.abs_cycles)
                events.append(_op_event(pid, batch, op, ts, labels))
                if op.pmu is not None:
                    values = dict(zip(labels, op.pmu, strict=False))
                    events.append({"name": "pmu", "ph": "C", "ts": _round(ts), "pid": pid, "args": values})
            phases, _, _ = pair_phases(batch)
            for phase in phases:
                is_dma = phase.name == DMA_PHASE
                (dma_seen if is_dma else threads_seen).add(phase.thread)
                events.append(_phase_event(pid, batch, phase, is_dma))
        for thread in sorted(threads_seen):
            events += _metadata(pid, TID_THREAD_BASE + thread, thread_name(thread), 10 + thread)
        for thread in sorted(dma_seen):
            events += _metadata(pid, TID_DMA_BASE + thread, f"dma t{thread}", 30 + thread)
    return events


def _batch_event(pid: int, batch: Batch) -> JsonDict:
    """Give the complete event of one batch."""
    args: JsonDict = {
        "seq": batch.seq,
        "n_ops": batch.n_ops,
        "usec": batch.usec,
        "cycles": batch.cycles,
        "start": batch.start,
        "mhz": batch.mhz,
        "op_time_us": _round(batch.op_time_us),
        "gap_us": _round(batch.gap_us),
    }
    if batch.evt_cnt is not None:
        args["evt_cnt"] = list(batch.evt_cnt)
    if batch.host_us is not None:
        args["host_us"] = batch.host_us
    return {
        "name": f"batch {batch.seq} ({batch.n_ops} ops)",
        "cat": "batch",
        "ph": "X",
        "ts": _round(batch.start_us),
        "dur": _round(float(batch.usec)),
        "pid": pid,
        "tid": TID_BATCHES,
        "args": args,
    }


def _op_event(pid: int, batch: Batch, op: Op, ts: float, labels: Sequence[str]) -> JsonDict:
    """Give the complete event of one op."""
    args: JsonDict = {
        "batch": batch.seq,
        "index": op.index,
        "names": op.names,
        "dims": op.dims,
        "types": op.types,
        "strides": op.strides,
        "kparams": op.kparams,
        "usec": op.usec,
        "cycles": op.cycles,
        "mhz": op.mhz,
    }
    if op.pmu is not None:
        args["pmu"] = dict(zip(labels, op.pmu, strict=False))
    return {
        "name": op.name,
        "cat": "op",
        "ph": "X",
        "ts": _round(ts),
        "dur": _round(op.duration_us(batch.clock_mhz)),
        "pid": pid,
        "tid": TID_OPS,
        "args": args,
    }


def _phase_event(pid: int, batch: Batch, phase: Phase, is_dma: bool) -> JsonDict:
    """Give the complete event of one phase of one DSP thread."""
    ts = batch.start_us + batch.offset_us(phase.start_cycles)
    args: JsonDict = {"batch": batch.seq, "thread": phase.thread, "info": phase.info}
    if phase.note:
        args["note"] = phase.note
    return {
        "name": phase.name,
        "cat": "dma" if is_dma else "phase",
        "ph": "X",
        "ts": _round(ts),
        "dur": _round((phase.end_cycles - phase.start_cycles) / batch.clock_mhz),
        "pid": pid,
        "tid": (TID_DMA_BASE if is_dma else TID_THREAD_BASE) + phase.thread,
        "args": args,
    }


def write_chrome_trace(log: ProfileLog, out: TextIO, batches: slice | None = None, arch: int | None = None) -> int:
    """Write the Chrome trace JSON of a log.

    Returns:
        The number of trace events written
    """
    events = chrome_events(log, batches, arch)
    document = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {"source": log.path, "profile_level": log.level, "clock": log.clock, "arch": log.arch},
    }
    json.dump(document, out, separators=(",", ":"))
    out.write("\n")
    return len(events)


@dataclass
class ClassStat:
    """The count and the time of one op class."""

    count: int = 0
    total_us: float = 0.0
    usec_sum: int = 0

    @property
    def per_op_us(self) -> float:
        """Give the mean time of one op of the class."""
        return self.total_us / self.count if self.count else 0.0


@dataclass
class Summary:
    """The class table of the selected batches of one session."""

    session: str
    n_batches: int = 0
    n_ops: int = 0
    batch_us: float = 0.0
    op_us: float = 0.0
    op_usec_sum: int = 0
    idle_us: float | None = None
    classes: dict[str, ClassStat] = field(default_factory=dict)
    phases: dict[str, ClassStat] = field(default_factory=dict)
    unpaired: int = 0

    @property
    def gap_us(self) -> float:
        """Give the batch time minus the op time."""
        return self.batch_us - self.op_us

    @property
    def idle_per_batch_us(self) -> float | None:
        """Give the mean time between the end of a batch and the start of the next one."""
        if self.idle_us is None or self.n_batches < 2:
            return None
        return self.idle_us / (self.n_batches - 1)

    def share(self, value_us: float) -> float:
        """Give a time as a percentage of the batch time."""
        return 100.0 * value_us / self.batch_us if self.batch_us else 0.0


KeyFunc = Callable[[Op], str]

KEYS: dict[str, KeyFunc] = {
    "op": lambda op: op.name,
    "op-path": lambda op: f"{op.name} {op.path}".rstrip(),
    "op-types": lambda op: f"{op.name} {op.types}",
}


def summarize(log: ProfileLog, batches: slice | None = None, key: str = "op") -> list[Summary]:
    """Build the class table of each session.

    Args:
        log: The parsed log
        batches: The slice of the batches of each session, or None for all
        key: The class key: ``op``, ``op-path`` (with the kernel path) or ``op-types``

    Returns:
        One summary for each session, in log order
    """
    key_func = KEYS[key]
    summaries: list[Summary] = []
    for session, kept in log.selected(batches):
        summary = Summary(session=session.name, n_batches=len(kept))
        if log.clock == "host" and len(kept) > 1:
            summary.idle_us = sum(after.start_us - (before.start_us + before.usec) for before, after in zip(kept, kept[1:], strict=False))
        for batch in kept:
            summary.batch_us += batch.usec
            summary.op_us += batch.op_time_us
            summary.op_usec_sum += batch.op_usec_sum
            summary.n_ops += len(batch.ops)
            for op in batch.ops:
                stat = summary.classes.setdefault(key_func(op), ClassStat())
                stat.count += 1
                stat.total_us += op.duration_us(batch.clock_mhz)
                stat.usec_sum += op.usec
            phases, unpaired_starts, unpaired_stops = pair_phases(batch)
            summary.unpaired += unpaired_starts + unpaired_stops
            for phase in phases:
                stat = summary.phases.setdefault(phase.name, ClassStat())
                stat.count += 1
                stat.total_us += (phase.end_cycles - phase.start_cycles) / batch.clock_mhz
        summaries.append(summary)
    return summaries


def format_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Format a table with the first column left-aligned and the other columns right-aligned."""
    widths = [len(cell) for cell in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    lines = []
    for row in (header, *rows):
        cells = [row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def format_summary(summary: Summary, top: int | None = None) -> str:
    """Format the class table of one session with the gap as the last row."""
    head = (
        f"session {summary.session}: {summary.n_batches} batches, {summary.n_ops} ops, "
        f"batch time {summary.batch_us:.1f} us, op time {summary.op_us:.1f} us "
        f"({summary.share(summary.op_us):.1f}%), gap {summary.gap_us:.1f} us ({summary.share(summary.gap_us):.1f}%), "
        f"op usec sum {summary.op_usec_sum}"
    )
    if summary.idle_us is not None and summary.idle_per_batch_us is not None:
        head += (
            f"\ntime between batches on the host clock: {summary.idle_us:.1f} us total, "
            f"{summary.idle_per_batch_us:.1f} us for each of the {summary.n_batches - 1} intervals"
        )
    ranked = sorted(summary.classes.items(), key=lambda item: item[1].total_us, reverse=True)
    if top is not None:
        ranked = ranked[:top]
    rows = [
        [name, str(stat.count), f"{stat.total_us:.1f}", f"{stat.per_op_us:.1f}", f"{summary.share(stat.total_us):.1f}%"]
        for name, stat in ranked
    ]
    gap_per_batch = summary.gap_us / summary.n_batches if summary.n_batches else 0.0
    rows.append(["(gap)", str(summary.n_batches), f"{summary.gap_us:.1f}", f"{gap_per_batch:.1f}", f"{summary.share(summary.gap_us):.1f}%"])
    text = head + "\n" + format_table(["class", "count", "total us", "us/op", "share"], rows)
    if summary.phases:
        phase_rows = [
            [name, str(stat.count), f"{stat.total_us:.1f}", f"{stat.per_op_us:.1f}"]
            for name, stat in sorted(summary.phases.items(), key=lambda item: item[1].total_us, reverse=True)
        ]
        text += "\n\nphases of the DSP threads (the sum of all the threads):\n"
        text += format_table(["phase", "count", "total us", "us/event"], phase_rows)
        if summary.unpaired:
            text += f"\ntrace events without a pair: {summary.unpaired}"
    return text


def format_batches(log: ProfileLog, batches: slice | None = None) -> str:
    """Format one line for each selected batch: the ops, the times and the gap."""
    rows = []
    for session, kept in log.selected(batches):
        for batch in kept:
            rows.append(
                [
                    f"{session.name} #{batch.seq}",
                    str(batch.n_ops),
                    f"{batch.start_us:.1f}",
                    str(batch.usec),
                    f"{batch.op_time_us:.1f}",
                    f"{batch.gap_us:.1f}",
                    f"{batch.mhz:.1f}",
                ]
            )
    return format_table(["batch", "ops", "start us", "usec", "op time us", "gap us", "mhz"], rows)


def format_diff(before: Summary, after: Summary, top: int | None = None) -> str:
    """Compare two summaries class by class and format the table, the largest change first."""
    names = set(before.classes) | set(after.classes)
    rows = []
    for name in names:
        stat_a = before.classes.get(name, ClassStat())
        stat_b = after.classes.get(name, ClassStat())
        rows.append((name, stat_a, stat_b, stat_b.total_us - stat_a.total_us))
    rows.sort(key=lambda row: abs(row[3]), reverse=True)
    if top is not None:
        rows = rows[:top]
    table = []
    for name, stat_a, stat_b, delta in rows:
        percent = f"{100.0 * delta / stat_a.total_us:+.1f}%" if stat_a.total_us else "new"
        table.append(
            [
                name,
                str(stat_a.count),
                str(stat_b.count),
                f"{stat_a.total_us:.1f}",
                f"{stat_b.total_us:.1f}",
                f"{delta:+.1f}",
                percent,
                f"{stat_a.per_op_us:.1f}",
                f"{stat_b.per_op_us:.1f}",
            ]
        )
    head = (
        f"A: {before.n_batches} batches, batch time {before.batch_us:.1f} us, op time {before.op_us:.1f} us, gap {before.gap_us:.1f} us\n"
        f"B: {after.n_batches} batches, batch time {after.batch_us:.1f} us, op time {after.op_us:.1f} us, gap {after.gap_us:.1f} us\n"
        f"delta: batch time {after.batch_us - before.batch_us:+.1f} us, op time {after.op_us - before.op_us:+.1f} us, "
        f"gap {after.gap_us - before.gap_us:+.1f} us"
    )
    header = ["class", "count A", "count B", "total A us", "total B us", "delta us", "delta", "us/op A", "us/op B"]
    return head + "\n" + format_table(header, table)


def parse_slice(text: str | None) -> slice | None:
    """Parse a Python slice text such as ``1:``, ``:3``, ``-2:`` or ``2``."""
    if text is None:
        return None
    parts = text.split(":")
    if len(parts) == 1:
        index = int(parts[0])
        return slice(index, None if index == -1 else index + 1)
    if len(parts) > 3:
        raise ValueError(f"The batch slice {text!r} has more than three parts")
    return slice(*[int(part) if part else None for part in parts])


def describe(log: ProfileLog) -> str:
    """Give one line about a parsed log: the level, the clock, the sessions and the counts."""
    sessions = ", ".join(f"{name} ({len(session.batches)} batches)" for name, session in log.sessions.items()) or "none"
    arch = f"v{log.arch}" if log.arch else "unknown"
    text = f"{log.path}: level {log.level}, clock {log.clock}, arch {arch}, sessions: {sessions}"
    if log.level >= 2:
        labels = pmu_counter_labels(log.pmu_events, log.arch)
        text += "\npmu events: " + ", ".join(f"0x{event_id:x} {label}" for event_id, label in zip(log.pmu_events, labels, strict=True))
    if log.orphan_ops or log.orphan_events or log.bad_lines:
        text += (
            f"\nops without a batch: {log.orphan_ops}, trace events without a batch: {log.orphan_events}, "
            f"unreadable lines: {log.bad_lines}"
        )
    return text


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser with the convert, summary and diff commands."""
    parser = argparse.ArgumentParser(description="Convert, summarize or compare profile logs of the llama.cpp Hexagon backend.")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert", help="write the Chrome trace JSON for ui.perfetto.dev")
    convert.add_argument("log", help="the profile log, or - for stdin")
    convert.add_argument("-o", "--output", help="the JSON file (the preset value is the log name with .json)")
    convert.add_argument("--batches", help="a slice of the batches of each session, for example 1:")
    convert.add_argument("--arch", type=int, help="the Hexagon architecture for the PMU names (75, 79 or 81)")

    summary = commands.add_parser("summary", help="print the per-class table and the gap")
    summary.add_argument("log", help="the profile log, or - for stdin")
    summary.add_argument("--batches", help="a slice of the batches of each session, for example 1:")
    summary.add_argument("--key", choices=sorted(KEYS), default="op", help="the class key")
    summary.add_argument("--top", type=int, help="the number of classes to print")
    summary.add_argument("--show-batches", action="store_true", help="also print one line for each batch")

    diff = commands.add_parser("diff", help="compare two logs class by class")
    diff.add_argument("log_a", help="the profile log A")
    diff.add_argument("log_b", help="the profile log B")
    diff.add_argument("--batches", help="a slice of the batches of each session, for the two logs")
    diff.add_argument("--key", choices=sorted(KEYS), default="op", help="the class key")
    diff.add_argument("--top", type=int, help="the number of classes to print")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns the exit code."""
    args = _build_parser().parse_args(argv)
    batches = parse_slice(args.batches)

    if args.command == "convert":
        log = parse_file(args.log)
        output = args.output or (str(Path(args.log).with_suffix(".json")) if args.log != "-" else "-")
        if output == "-":
            count = write_chrome_trace(log, sys.stdout, batches, args.arch)
        else:
            with open(output, "w", encoding="utf-8") as handle:
                count = write_chrome_trace(log, handle, batches, args.arch)
        print(describe(log), file=sys.stderr)
        print(f"wrote {count} events to {output}", file=sys.stderr)
        return 0

    if args.command == "summary":
        log = parse_file(args.log)
        print(describe(log))
        for summary in summarize(log, batches, args.key):
            print()
            print(format_summary(summary, args.top))
        if args.show_batches:
            print()
            print(format_batches(log, batches))
        return 0

    log_a = parse_file(args.log_a)
    log_b = parse_file(args.log_b)
    print("A: " + describe(log_a))
    print("B: " + describe(log_b))
    summaries_b = {summary.session: summary for summary in summarize(log_b, batches, args.key)}
    for summary_a in summarize(log_a, batches, args.key):
        summary_b = summaries_b.get(summary_a.session)
        if summary_b is None:
            print(f"\nsession {summary_a.session} is not in B")
            continue
        print(f"\nsession {summary_a.session}")
        print(format_diff(summary_a, summary_b, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
