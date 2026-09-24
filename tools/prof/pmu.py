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

Event ids are the raw hardware ids. The ids of the core, HVX, AXI and DMA events
agree with the V79 column of the table in libitrace.so of the Hexagon SDK
6.6.0.0, which maps the names of libs/itrace/inc/itrace_dsp_events_pmu.h to raw
ids (the header values are not raw ids). The HMX ids come from the enum
_PMU_EVENTS_ENUM_ of libhexagonissv79.so (Hexagon Tools 19.0.07). The phone
stage pmu of 2026-09-24 (tools/stages/pmu) confirmed each HMX id in EVENTS: it
counts on the ops that use the HMX and it is 0 on all other ops. The HMX ids of
NEEDS_ID counted 0 on all ops, thus they are not confirmed.
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
    "DU_CONFLICT_PVIEW_CYCLES": (0xEC, "a DU resource conflict"),
    # The coprocessor of the core is the HVX (the SDK names the HVX contexts "coprocessor" contexts).
    "COPROC_BUSY_PVIEW_CYCLES": (0xED, "the coprocessor (HVX) is busy"),
    "DU_UNCACHED_PVIEW_CYCLES": (0xEE, "a D-cache uncacheable access"),
    "SYSTEM_BUSY_PVIEW_CYCLES": (0xEF, "DMA synchronization, a full ETM or a busy AXI bus"),
    # The cycles with a commit. The 11 reasons above plus these four give about 2 for each cycle of an op
    # on the phone (two clusters). The sum is less when two clusters commit in the same cycle.
    "CYCLES_1_PACKET_COMMITTED": (0x300, "cycles with one packet committed"),
    "CYCLES_2_PACKET_COMMITTED": (0x301, "cycles with two packets committed"),
    "CYCLES_3_PACKET_COMMITTED": (0x302, "cycles with three packets committed"),
    "CYCLES_4_PACKET_COMMITTED": (0x303, "cycles with four packets committed"),
    # CYCLES_1..5_THREAD_RUNNING (0x3B, 0x3C, 0x3D, 0x3E, 0x0A) count 0 on the phone. Only this one counts.
    "CYCLES_6_THREAD_RUNNING": (0x0B, "cycles with 6 threads not in wait or stop"),

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
    "UDMA_DMPOLL_CYCLES": (0x245, "cycles of the DMA command dmpoll (a thread polls the DMA)"),
    "UDMA_NONCOHERENT_RD_CYCLES": (0x262, "cycles the DMA waits for a read that bypasses the caches"),
    "UDMA_RD_BUFFER_LEVEL_FULL": (0x269, "cycles with the DMA read buffer fully allocated"),
    "L2_UDMA_BYPASS_RD": (0x256, "user DMA reads that bypass the cache hierarchy"),
    "ICACHE_DEMAND_MISS": (0x12, "I-cache demand misses"),
    "DU_BANK_CONFLICT_REPLAY": (0xA1, "replays from a dual access to one bank"),

    # The HMX. Each id counts on the ops that use the HMX (the hmx-tiled and hmx-pipe kernels and the
    # chunked delta rule) and is 0 on all other ops (stage pmu, 2026-09-24).
    # One MAC count (HMXMAC_FLT or HMXMAC_FLT_PARTIAL) is 8 x 32 x 32 MACs, 16384 FLOPs. HMXMAC_FLT_PARTIAL
    # holds about 98 % of the MAC counts of the f16 kernels. HMX_CLK over HMX_ACTIVE times the core clock
    # gives the HMX clock while the HMX is active: 1210 MHz at the start of a prefill, about 750 MHz after
    # 4 ubatches of 1024 tokens.
    "HMX_ACTIVE": (0x200, "cycles the HMX is active (core clock)"),
    "HMX_CVT_FULL": (0x201, "the HMX convert FIFO is full"),
    "HMX_MAC_FULL": (0x202, "the HMX MAC FIFO is full"),
    "HMX_CVT": (0x204, "HMX convert"),
    "HMX_MAC": (0x205, "HMX MAC"),
    "HMX_PKT_THREAD": (0x206, "HMX packets by thread"),
    "HMX_MXFIFO_FULL": (0x207, "the HMX instruction FIFO is full"),
    "HMXMAC_ACT_OUTSTANDING": (0x209, "MAC stall: the activation read is not complete"),
    "HMXMAC_WGT_OUTSTANDING": (0x20A, "MAC stall: the weight read is not complete"),
    "HMXMAC_POWER_OVER": (0x20D, "MAC throttle for power"),
    "HMXMAC_FLT_PARTIAL": (0x20F, "MAC float cycles (partial)"),
    "HMXMAC_FLT": (0x212, "MAC float cycles"),
    "HMX_CLK": (0x214, "HMX clock cycles"),
    "HMXMAC_ORDER": (0x228, "MAC order stall"),
    "HMXRDACT_PARTIAL": (0x230, "partial activation reads from VTCM"),
    "HMXRDWGT_PARTIAL": (0x231, "partial weight reads from VTCM"),
    "HMXRDACT_ACT": (0x232, "activation reads from VTCM"),
    "HMXRDWGT_WGT": (0x233, "weight reads from VTCM"),
    "HMXRDWGT_SCALE": (0x234, "scale reads from VTCM"),
    "HMXWR_OUTSTANDING": (0x237, "HMX write to VTCM not complete"),
    "HMXWR": (0x23B, "HMX writes to VTCM"),
    "HMX_MXFIFO_EMPTY": (0x291, "the HMX instruction FIFO is empty"),
    "HMX_LIMITS_THROTTLE_TLMH": (0x292, "HMX throttle by the thread limits management"),
    "HMX_DPM_AVG_COMPRESSED": (0x294, "the digital power meter of the HMX"),
    "HMX_POWERLIMITS_OVER": (0x295, "HMX over the power limits"),
}

# HMX events of the simulator enum that counted 0 on all ops of the stage pmu of 2026-09-24, thus their ids
# are candidates, not confirmed: name -> (candidate raw id, description). The f16 kernels do no fixed-point
# MACs, thus 0 is the expected count for the FXP pair.
#
# The 9 PMU_COPROC_* names of the Snapdragon Profiler NPU plugin that this table had before are NOT HMX
# events. They are the HVX coprocessor events of V60 to V62: the plugin programs their ids 0xF0 to 0xFC only
# below V65 or on a DSP that is not a compute DSP. On V79 the ids 0xF0 to 0xFD count the even AXI master,
# and the family is COPROC0..3_* (one for each HVX context) at 0x180 to 0x18F.
NEEDS_ID: dict[str, tuple[int, str]] = {
    "HMX_DROP": (0x203, "HMX drop"),
    "HMXMAC_MULT_DROP": (0x20B, "MAC multiply drop"),
    "HMXMAC_FXP_PARTIAL": (0x20E, "MAC fixed-point cycles (partial)"),
    "HMXMAC_DRAIN_PARTIAL": (0x210, "MAC drain cycles (partial)"),
    "HMXMAC_FXP": (0x211, "MAC fixed-point cycles"),
    "HMXMAC_DRAIN": (0x213, "MAC drain cycles"),
    "HMX_LIMITS_THROTTLE_LMH": (0x293, "HMX throttle by the limits management"),
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
    "topdown": (
        "all 11 top-down stall reasons and the cycles with a commit, thus the shares close "
        "(the set stalls has 6 of the 11 reasons). Two passes.",
        [["THREAD_IDLE_PVIEW_CYCLES", "ARCH_LOCK_PVIEW_CYCLES", "REDIRECT_PVIEW_CYCLES",
          "IU_NO_PKT_PVIEW_CYCLES", "DU_CACHE_MISS_PVIEW_CYCLES", "DU_BUSY_OTHER_PVIEW_CYCLES",
          "CU_BUSY_PVIEW_CYCLES", "DU_CONFLICT_PVIEW_CYCLES"],
         ["COPROC_BUSY_PVIEW_CYCLES", "DU_UNCACHED_PVIEW_CYCLES", "SYSTEM_BUSY_PVIEW_CYCLES",
          "CYCLES_1_PACKET_COMMITTED", "CYCLES_2_PACKET_COMMITTED", "CYCLES_3_PACKET_COMMITTED",
          "CYCLES_4_PACKET_COMMITTED", "COMMITTED_PKT_ANY"]],
    ),
    "hmx": (
        "the HMX: activity, clock, MAC cycles and the reasons the MAC does not compute "
        "(power, the instruction FIFO, the VTCM reads), and the power limits. Two passes.",
        [["HMX_ACTIVE", "HMX_CLK", "HMXMAC_FLT", "HMXMAC_FLT_PARTIAL", "HMXMAC_POWER_OVER",
          "HMX_MXFIFO_FULL", "HMX_MXFIFO_EMPTY", "HMX_POWERLIMITS_OVER"],
         ["HMX_ACTIVE", "HMXMAC_ACT_OUTSTANDING", "HMXMAC_WGT_OUTSTANDING", "HMX_LIMITS_THROTTLE_TLMH",
          "HMX_DPM_AVG_COMPRESSED", "HMX_MAC_FULL", "HMXRDACT_PARTIAL", "HMXRDWGT_PARTIAL"]],
    ),
    "dma-wait": (
        "where the DMA and the threads wait: dmpoll, the read that bypasses the caches, the "
        "full read buffer. A decode stall event of 3 ms shows as dmpoll and a DDR read wait.",
        [["UDMA_ACTIVE", "UDMA_DMPOLL_CYCLES", "UDMA_NONCOHERENT_RD_CYCLES", "UDMA_RD_BUFFER_LEVEL_FULL",
          "L2_UDMA_BYPASS_RD", "SYSTEM_BUSY_PVIEW_CYCLES", "DU_CACHE_MISS_PVIEW_CYCLES", "COMMITTED_PKT_ANY"]],
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
    print(f"{len(NEEDS_ID)} HMX events have a candidate id that counted 0 on the 4B, thus it is not confirmed.")
    print("Run 'pmu.py todo' for the list.")
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
    """Print the HMX events whose candidate id is not confirmed.

    Args:
        _: The parsed arguments, unused

    Returns:
        The exit status
    """
    print("HMX events of the simulator enum (libhexagonissv79.so) that counted 0 on all ops of the")
    print("4B Q8_0 (stage pmu, 2026-09-24). A workload with that work (for example an int8 HMX kernel")
    print("for the FXP pair) can confirm them. The confirmed HMX ids are in EVENTS and in the set hmx.")
    print()
    width = max(len(k) for k in NEEDS_ID)
    for n, (eid, desc) in NEEDS_ID.items():
        print(f"  0x{eid:03x}  {n:<{width}}  {desc}")
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
