#!/usr/bin/env python3
"""Aggregate a Hexagon profile log into a per-operation stall table.

This is the report half of the ``stalls`` event set. The backend prints one
``profile-op`` line per operation per batch when ``GGML_HEXAGON_PROFILE`` names
eight events and ``llama-bench`` runs with ``-v``. This tool groups those lines
and prints where the time goes and why.

The stall events are the ``*_PVIEW_CYCLES`` family, a top-down accounting of
cycles the cluster could not commit. They do not sum to the cycle count,
because a cycle that commits work is in none of them, thus each is shown as a
share of the cycles of that operation.

Instructions per packet is the VLIW slot utilisation. The maximum is 4. A low
value means the compiler could not fill the packet, which no GPU counter has an
analogue for and which is the largest lever on a hand-written kernel.

Usage:
    tools/prof/stalls.py tools/prof/store/<stamp>-stalls-p1.log
    tools/prof/stalls.py <log> --by-tensor --top 20
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# The order of the eight counters in the `stalls` set of pmu.py.
STALL_EVENTS = ["COMMITTED_PKT_ANY", "COMMITTED_INSTS", "DU_CACHE_MISS", "CU_BUSY",
                "COPROC_BUSY", "IU_NO_PKT", "DU_BUSY_OTHER", "SYSTEM_BUSY"]

# ggml-hex: HTP0 profile-op NAME|tensors|dims|types|strides|kernel|usec U cycles C start S mhz M pmu [a,b,...]
LINE = re.compile(
    r"profile-op\s+(?P<op>[A-Z_0-9]+)\|(?P<rest>.*?)\|"
    r"usec\s+(?P<usec>\d+)\s+cycles\s+(?P<cycles>\d+)\s+start\s+\d+\s+mhz\s+[\d.]+"
    r"\s+pmu\s+\[(?P<pmu>[\d,]+)\]")


@dataclass
class Agg:
    """The running totals of one group of operations.

    Attributes:
        calls: The number of profile lines in the group
        usec: The total microseconds
        cycles: The total cycles
        counters: The total of each of the eight counters
    """

    calls: int = 0
    usec: int = 0
    cycles: int = 0
    counters: list[int] = field(default_factory=lambda: [0] * 8)

    def add(self, usec: int, cycles: int, pmu: list[int]) -> None:
        """Add one profile line to the group.

        Args:
            usec: The microseconds of the operation
            cycles: The cycles of the operation
            pmu: The eight counter values
        """
        self.calls += 1
        self.usec += usec
        self.cycles += cycles
        for i, v in enumerate(pmu[:8]):
            self.counters[i] += v

    @property
    def ipp(self) -> float:
        """Instructions per packet, the VLIW slot utilisation, at most 4."""
        pkt = self.counters[0]
        return self.counters[1] / pkt if pkt else 0.0

    def share(self, idx: int) -> float:
        """The share of the cycles that one counter accounts for.

        Args:
            idx: The counter index, 2 through 7 for the stall events

        Returns:
            A fraction between 0 and 1, or more when events overlap
        """
        return self.counters[idx] / self.cycles if self.cycles else 0.0


def parse(path: str, by_tensor: bool) -> dict[str, Agg]:
    """Read a profile log and group its operations.

    Args:
        path: The log file
        by_tensor: True to group by the operation and its first tensor name,
            False to group by the operation kind alone

    Returns:
        The groups, keyed by the group name

    Raises:
        OSError: If the log cannot be read
    """
    groups: dict[str, Agg] = defaultdict(Agg)
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        key = m["op"]
        if by_tensor:
            first = m["rest"].split("|")[0].split(" x ")[0].strip()
            key = f"{m['op']} {first}"
        pmu = [int(x) for x in m["pmu"].split(",")]
        groups[key].add(int(m["usec"]), int(m["cycles"]), pmu)
    return groups


def report(groups: dict[str, Agg], top: int) -> None:
    """Print the stall table, the slowest group first.

    Args:
        groups: The aggregated groups
        top: The number of rows to print
    """
    total = sum(g.usec for g in groups.values())
    rows = sorted(groups.items(), key=lambda kv: -kv[1].usec)[:top]
    width = min(48, max(len(k) for k, _ in rows))

    print(f"{'operation':<{width}} {'ms':>9} {'%':>6} {'calls':>6} {'i/pkt':>6}"
          f" {'cache':>7} {'cu':>7} {'coproc':>7} {'sys':>7} {'ifetch':>7}")
    print("-" * (width + 68))
    for name, g in rows:
        print(f"{name[:width]:<{width}} {g.usec / 1000:9.1f} {100 * g.usec / total:6.1f}"
              f" {g.calls:6d} {g.ipp:6.2f}"
              f" {100 * g.share(2):6.1f}% {100 * g.share(3):6.1f}%"
              f" {100 * g.share(4):6.1f}% {100 * g.share(7):6.1f}%"
              f" {100 * g.share(5):6.1f}%")
    print("-" * (width + 68))
    print(f"{'total':<{width}} {total / 1000:9.1f}")
    print()
    print("i/pkt   instructions per packet, the VLIW slot fill, at most 4")
    print("cache   DU_CACHE_MISS_PVIEW_CYCLES,  a data-cache miss held the cluster")
    print("cu      CU_BUSY_PVIEW_CYCLES,        a register interlock or a port conflict")
    print("coproc  COPROC_BUSY_PVIEW_CYCLES,    the coprocessor was busy")
    print("sys     SYSTEM_BUSY_PVIEW_CYCLES,    DMA sync, a busy AXI bus, cache maintenance")
    print("ifetch  IU_NO_PKT_PVIEW_CYCLES,      the issue queue was empty")
    print()
    print("A share above 100 % is not an error. The counters are not filtered to one")
    print("hardware thread, thus a threaded operation sums the cycles of every thread")
    print("while the cycle column holds the elapsed time of one. Read those rows as")
    print("'this is the dominant stall', never as an absolute fraction.")


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and print the report.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--by-tensor", action="store_true",
                    help="group by the operation and its first tensor, not by the kind")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)

    groups = parse(a.log, a.by_tensor)
    if not groups:
        print(f"no profile-op lines in {a.log}. "
              "The run needs GGML_HEXAGON_PROFILE with eight events and llama-bench -v.",
              file=sys.stderr)
        return 1
    report(groups, a.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
