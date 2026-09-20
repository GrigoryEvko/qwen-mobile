#!/usr/bin/env python3
"""Named Hexagon PMU event sets, in the shape ``GGML_HEXAGON_PROFILE`` accepts.

The backend takes exactly eight comma-separated event ids and then runs in mode
2, one counter delta per op (ggml-hexagon.cpp:8163). The hardware has eight
counters, thus a question that needs more than eight events needs more than one
run. This module holds the sets, checks the width and prints the command line.

The set the backend ships by default holds no stall-attribution event at all,
thus a run on the phone can say that an op is slow and never say why. The
``stalls`` set here is the one that answers it.

Usage:
    tools/prof/pmu.py list
    tools/prof/pmu.py show stalls
    tools/prof/pmu.py env stalls
    tools/prof/pmu.py env bandwidth --pass 2

Event ids are the raw hardware ids. Every id below is quoted from the V79
Programmer Reference Manual chapter 10 or from the SDK event headers. Ids that
we have not confirmed are in NEEDS_ID and deliberately carry no number.
"""

from __future__ import annotations

import argparse
import sys

# name -> (raw id, one-line description)
EVENTS: dict[str, tuple[int, str]] = {
    # Issue and commit. The pair gives instructions per packet, which is the
    # VLIW slot utilisation and has no GPU counterpart.
    "COMMITTED_PKT_ANY": (0x3, "packets committed by any thread"),
    "COMMITTED_INSTS": (0x2A, "instructions committed"),
    "CU_PKT_READY_NOT_DISPATCHED": (0x17, "ready at the scheduler but not picked"),

    # The top-down stall accounting. Mutually exclusive: cycles the cluster
    # could not commit because of X. This family is the SASS stall-reason
    # analogue and the one the default set omits.
    "THREAD_IDLE_PVIEW_CYCLES": (0xE5, "a thread is off, waiting or paused"),
    "ARCH_LOCK_PVIEW_CYCLES": (0xE6, "a kernel lock or a TLB lock"),
    "REDIRECT_PVIEW_CYCLES": (0xE7, "a redirect such as a branch mispredict"),
    "IU_NO_PKT_PVIEW_CYCLES": (0xE8, "the issue queue is empty"),
    "DU_CACHE_MISS_PVIEW_CYCLES": (0xE9, "a D-cache cacheable miss"),
    "DU_BUSY_OTHER_PVIEW_CYCLES": (0xEA, "a DU replay, a DU bubble or a DTLB miss"),
    "CU_BUSY_PVIEW_CYCLES": (0xEB, "a register interlock, a port conflict or a timing class"),
    "COPROC_BUSY_PVIEW_CYCLES": (0xED, "the coprocessor is busy"),
    "DU_UNCACHED_PVIEW_CYCLES": (0xEE, "a D-cache uncacheable access"),
    "SYSTEM_BUSY_PVIEW_CYCLES": (0xEF, "DMA synchronization, a full ETM or a busy AXI bus"),

    # HVX occupancy and its ordering interlocks.
    "HVX_ACTIVE": (0x100, "the vector FIFO is not empty"),
    "HVX_REG_ORDER": (0x101, "stall cycles due to register interlocks"),
    "HVX_ACC_ORDER": (0x102, "stall cycles due to an accumulator not yet produced"),
    "HVX_LD_L2_OUTSTANDING": (0x103, "stall cycles with a load pending"),
    "HVX_VTCM_OUTSTANDING": (0x105, "stall cycles with a VTCM transaction pending"),
    "HVX_ST_FULL": (0x108, "the store buffer is full"),
    "HVX_PKT": (0x111, "HVX packets, counted twice in 128-byte mode"),
    "HVX_PKT_THREAD": (0x112, "committed packets on a thread with the XE bit set"),

    # The four HVX pipes. The best per-pipe roofline we can get.
    "HVXPIPE_ALU": (0x128, "packets that used the HVX ALU pipe"),
    "HVXPIPE_MPY": (0x129, "packets that used the HVX multiply pipe"),
    "HVXPIPE_SHIFT": (0x12A, "packets that used the HVX shift pipe"),
    "HVXPIPE_PERM": (0x12B, "packets that used the HVX permute pipe"),

    # DDR traffic. The SDK gives the byte formula in
    # libs/itrace/inc/itrace_dsp_events_derived_pmu.h:
    #   bytes = L32*32 + L64*64 + L128*128 + L256*256 + (ANY - the four)*8
    "AXI_READ_REQUEST": (0x40, "any AXI read request"),
    "AXI_LINE32_READ_REQUEST": (0x41, "32-byte AXI read requests"),
    "AXI_LINE64_READ_REQUEST": (0xCE, "64-byte AXI read requests"),
    "AXI_LINE128_READ_REQUEST": (0x3F, "128-byte AXI read requests"),
    "AXI_LINE256_READ_REQUEST": (0xCD, "256-byte AXI read requests"),
    "AXI_WRITE_REQUEST": (0x42, "any AXI write request"),
    "AXI_LINE32_WRITE_REQUEST": (0x43, "32-byte AXI write requests"),
    "AXI_LINE64_WRITE_REQUEST": (0xCF, "64-byte AXI write requests"),
    "AXI_LINE128_WRITE_REQUEST": (0x46, "128-byte AXI write requests"),
    "AXI_LINE256_WRITE_REQUEST": (0x55, "256-byte AXI write requests"),

    # Cache, prefetch and the user DMA. The DMA path is the one that reaches
    # the full stream rate; a vector load reaches a fraction of it.
    "L2_DU_READ_MISS": (0x7D, "L2 read misses from the data unit"),
    "L2_DU_STORE_MISS": (0x8C, "L2 store misses from the data unit"),
    "L2FETCH_ACCESS": (0x7E, "accesses from the L2 prefetch engine"),
    "L2FETCH_MISS": (0x7F, "L2 prefetch accesses that missed"),
    "UDMA_ACTIVE": (0x240, "cycles with the user DMA not idle"),
    "L2_UDMA_BYPASS_RD": (0x256, "user DMA reads that bypass the cache hierarchy"),
    "ICACHE_DEMAND_MISS": (0x12, "I-cache demand misses"),
    "DU_BANK_CONFLICT_REPLAY": (0xA1, "replays from a dual access to one bank"),
}

# The coprocessor family that the Snapdragon Profiler NPU plugin collects. The
# names are quoted from the plugin binary. Their raw ids are NOT confirmed, thus
# they carry no number here and must be discovered before use. One member of the
# family, COPROC_BUSY_PVIEW_CYCLES, is confirmed at 0xED and is in EVENTS above.
# This family is the only HMX instrumentation on V79: the simulator runs HMX
# functionally and never retires it in timing mode.
NEEDS_ID: dict[str, str] = {
    "PMU_COPROC_IDLE": "cycles the coprocessor was idle, with an empty queue and pipeline",
    "PMU_COPROC_CYCLES_RUNNING": "cycles the coprocessor was running",
    "PMU_COPROC_PKT_EXEC": "packets the coprocessor executed",
    "PMU_COPROC_PKT_THREAD": "coprocessor packets by thread",
    "PMU_COPROC_FIFO_DISPATCH": "dispatches into the coprocessor FIFO",
    "PMU_COPROC_FIFO_FULL_REPLAY": "replays because the queue to the coprocessor was full",
    "PMU_COPROC_REPLAY": "coprocessor replays",
    "PMU_COPROC_VEXTRACT_STALL": "stalls on a vector extract",
    "PMU_COPROC_AXISLAVE_ACCESS": "coprocessor memory accessed by the AXI slave",
}

# Named sets, in the style of an Nsight Compute section. Each pass holds at most
# eight events, because the hardware has eight counters.
SETS: dict[str, tuple[str, list[list[str]]]] = {
    "default": (
        "what the backend ships today. It holds no stall event, thus it cannot "
        "say why an op is slow.",
        [["COMMITTED_PKT_ANY", "HVX_PKT", "HVX_ACTIVE", "HVX_VTCM_OUTSTANDING",
          "UDMA_ACTIVE", "L2_UDMA_BYPASS_RD", "L2_DU_READ_MISS", "L2_DU_STORE_MISS"]],
    ),
    "stalls": (
        "the top-down stall accounting: why the cluster could not commit. "
        "Start every investigation here.",
        [["COMMITTED_PKT_ANY", "COMMITTED_INSTS", "DU_CACHE_MISS_PVIEW_CYCLES",
          "CU_BUSY_PVIEW_CYCLES", "COPROC_BUSY_PVIEW_CYCLES", "IU_NO_PKT_PVIEW_CYCLES",
          "DU_BUSY_OTHER_PVIEW_CYCLES", "SYSTEM_BUSY_PVIEW_CYCLES"]],
    ),
    "hvx": (
        "HVX occupancy and the ordering interlocks that hold it back.",
        [["COMMITTED_PKT_ANY", "COMMITTED_INSTS", "HVX_ACTIVE", "HVX_PKT",
          "HVX_REG_ORDER", "HVX_ACC_ORDER", "HVX_LD_L2_OUTSTANDING", "HVX_VTCM_OUTSTANDING"]],
    ),
    "pipes": (
        "which of the four HVX pipes the work used, for a per-pipe roofline.",
        [["COMMITTED_PKT_ANY", "COMMITTED_INSTS", "HVXPIPE_ALU", "HVXPIPE_MPY",
          "HVXPIPE_SHIFT", "HVXPIPE_PERM", "HVX_ACTIVE", "HVX_ST_FULL"]],
    ),
    "bandwidth": (
        "DDR bytes moved, for a measured roofline instead of a modelled one. "
        "Ten events, thus two passes.",
        [["AXI_READ_REQUEST", "AXI_LINE32_READ_REQUEST", "AXI_LINE64_READ_REQUEST",
          "AXI_LINE128_READ_REQUEST", "AXI_LINE256_READ_REQUEST", "COMMITTED_PKT_ANY",
          "UDMA_ACTIVE", "L2_UDMA_BYPASS_RD"],
         ["AXI_WRITE_REQUEST", "AXI_LINE32_WRITE_REQUEST", "AXI_LINE64_WRITE_REQUEST",
          "AXI_LINE128_WRITE_REQUEST", "AXI_LINE256_WRITE_REQUEST", "COMMITTED_PKT_ANY",
          "L2_DU_READ_MISS", "L2FETCH_MISS"]],
    ),
    "dma": (
        "whether the weight stream goes through the DMA engine or through "
        "vector loads. The DMA path reaches the full rate and a vector load "
        "reaches a fraction of it, thus this set decides a large question.",
        [["COMMITTED_PKT_ANY", "UDMA_ACTIVE", "L2_UDMA_BYPASS_RD", "L2FETCH_ACCESS",
          "L2FETCH_MISS", "L2_DU_READ_MISS", "DU_CACHE_MISS_PVIEW_CYCLES",
          "HVX_LD_L2_OUTSTANDING"]],
    ),
}


def _fmt(names: list[str]) -> str:
    """The comma-separated id list that the backend parses.

    Args:
        names: The event names of one pass

    Returns:
        The value for GGML_HEXAGON_PROFILE

    Raises:
        KeyError: If a name is not in EVENTS
        ValueError: If the pass does not hold exactly eight events
    """
    if len(names) != 8:
        raise ValueError(f"a pass must hold exactly 8 events, this one holds {len(names)}")
    return ",".join(f"0x{EVENTS[n][0]:x}" for n in names)


def cmd_list(_: argparse.Namespace) -> int:
    """Print every set with its purpose and its pass count.

    Args:
        _: The parsed arguments, unused

    Returns:
        The exit status
    """
    width = max(len(k) for k in SETS)
    for name, (why, passes) in SETS.items():
        print(f"{name:<{width}}  {len(passes)} pass(es)  {why}")
    print()
    print(f"{len(NEEDS_ID)} coprocessor events are named but have no confirmed id yet.")
    print("They are the only HMX instrumentation on V79. Run 'pmu.py todo' for the list.")
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    """Print the events of one set, one line per event.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    why, passes = SETS[a.set]
    print(f"{a.set}: {why}")
    for i, names in enumerate(passes, 1):
        print(f"\n  pass {i}:")
        for n in names:
            eid, desc = EVENTS[n]
            print(f"    0x{eid:<4x} {n:<32} {desc}")
    return 0


def cmd_env(a: argparse.Namespace) -> int:
    """Print the environment assignment for one pass of one set.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    _, passes = SETS[a.set]
    if not 1 <= a.pass_no <= len(passes):
        print(f"{a.set} has {len(passes)} pass(es), not {a.pass_no}", file=sys.stderr)
        return 2
    print(f"GGML_HEXAGON_PROFILE={_fmt(passes[a.pass_no - 1])}")
    return 0


def cmd_todo(_: argparse.Namespace) -> int:
    """Print the coprocessor events whose raw id is still unknown.

    Args:
        _: The parsed arguments, unused

    Returns:
        The exit status
    """
    print("Named by the Snapdragon Profiler NPU plugin, raw id not yet confirmed.")
    print("COPROC_BUSY_PVIEW_CYCLES is confirmed at 0xed and is already usable.")
    print()
    width = max(len(k) for k in NEEDS_ID)
    for n, desc in NEEDS_ID.items():
        print(f"  {n:<{width}}  {desc}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="every set with its purpose").set_defaults(fn=cmd_list)

    s = sub.add_parser("show", help="the events of one set")
    s.add_argument("set", choices=sorted(SETS))
    s.set_defaults(fn=cmd_show)

    e = sub.add_parser("env", help="the environment assignment for one pass")
    e.add_argument("set", choices=sorted(SETS))
    e.add_argument("--pass", dest="pass_no", type=int, default=1)
    e.set_defaults(fn=cmd_env)

    sub.add_parser("todo", help="coprocessor events that need an id").set_defaults(fn=cmd_todo)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
