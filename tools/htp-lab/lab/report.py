#!/usr/bin/env python3
"""The report of one lab run: the result lines, the per-function profile, and the stall table.

The script reads the per-packet profile of hexagon-sim (--packet_analyze), the symbol table
(hexagon-nm -S -n) and the disassembly (hexagon-llvm-objdump -d) of the program, and the program
output with the "lab:" result lines. It prints:

1. The result lines of the program.
2. The functions of the kernel, sorted by cycles: commits, stall cycles, and the stall types.
3. The stall types of the kernel functions with their share.
4. The packets with the most stall cycles, with their disassembly.

A packet entry of the profile has "commits" (the number of times the packet committed) and
"stalls" (the cycles the packet waited, by type). The cycles of a function are the sum of its
commits and its stall cycles. The HVX profile has the same shape for the vector packets.
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# The functions of the kernels. The lab targets wrap each kernel in a function with this prefix,
# thus the kernel keeps its own symbol after inlining.
KERNEL_PATTERN = re.compile(r"^(kernel_|tiled_|accum_|unpack_|quantize_|gdn_|hvx_|hmx_|core_dot|op_)")

# The pseudo address of the idle thread in the profile
IDLE_PC = 0xDEADBEEF


@dataclass
class Symbol:
    """One function of the program."""

    addr: int
    size: int
    name: str


@dataclass
class FuncStats:
    """The profile of one function."""

    commits: int = 0
    stalls: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    @property
    def cycles(self) -> int:
        """The cycles the function spent: one per commit plus the stall cycles."""
        return self.commits + self.stalls


def read_symbols(path: Path) -> list[Symbol]:
    """Reads the output of hexagon-nm -S -n and keeps the code symbols with a size."""
    symbols: list[Symbol] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in ("T", "t", "W", "w"):
            symbols.append(Symbol(int(parts[0], 16), int(parts[1], 16), parts[3]))
    symbols.sort(key=lambda s: s.addr)
    return symbols


def find_symbol(symbols: list[Symbol], addrs: list[int], pc: int) -> Symbol | None:
    """Finds the function that contains the address. Complexity O(log n)."""
    i = bisect.bisect_right(addrs, pc) - 1
    if i < 0:
        return None
    s = symbols[i]
    if s.size and pc >= s.addr + s.size:
        return None
    return s


def read_disasm(path: Path) -> dict[int, str]:
    """Maps each instruction address of the disassembly to its text."""
    out: dict[int, str] = {}
    pat = re.compile(r"^\s*([0-9a-f]+):\s*(.*)$")
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = pat.match(line)
        if m:
            out[int(m.group(1), 16)] = m.group(2).strip()
    return out


def packet_text(disasm: dict[int, str], pc: int) -> str:
    """Joins the instructions of the packet that starts at pc (up to 4 words, until the close brace)."""
    parts = []
    for off in range(0, 16, 4):
        text = disasm.get(pc + off)
        if text is None:
            break
        parts.append(text)
        if "}" in text:
            break
    return " ".join(parts) if parts else "?"


def is_kernel(name: str, pattern: re.Pattern[str]) -> bool:
    """Tells whether the function is a kernel function (and not the runtime or the harness)."""
    return pattern.match(name) is not None


def main() -> None:
    """Prints the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pa", required=True, type=Path)
    ap.add_argument("--symbols", required=True, type=Path)
    ap.add_argument("--disasm", type=Path, default=Path("disasm.txt"))
    ap.add_argument("--stdout", type=Path, default=Path("stdout.txt"))
    ap.add_argument("--target", default="")
    ap.add_argument("--top", type=int, default=24, help="the number of packets in the stall table")
    ap.add_argument("--all-functions", action="store_true", help="include the runtime functions")
    ap.add_argument("--kernel", default=KERNEL_PATTERN.pattern, help="the regular expression of the kernel functions")
    args = ap.parse_args()
    pattern = re.compile(args.kernel)

    print(f"== htp-lab report: {args.target}")
    if args.stdout.exists():
        for line in args.stdout.read_text(errors="replace").splitlines():
            if line.startswith("lab:") or line.startswith("farf ERROR"):
                print(line)

    pa = json.loads(args.pa.read_text())
    symbols = read_symbols(args.symbols)
    addrs = [s.addr for s in symbols]
    disasm = read_disasm(args.disasm)

    funcs: dict[str, FuncStats] = {}
    packets: list[tuple[int, int, int, dict[str, int], str]] = []
    for section in ("core_packet_profile", "hvx_packet_profile"):
        for key, entry in pa.get(section, {}).items():
            pc = int(key, 16)
            if pc == IDLE_PC:
                continue
            commits = int(entry.get("commits", 0))
            stalls = entry.get("stalls", {})
            total = int(stalls.get("TOTAL_STALLS", 0))
            sym = find_symbol(symbols, addrs, pc)
            name = sym.name if sym else "?"
            fs = funcs.setdefault(name, FuncStats())
            if section == "core_packet_profile":
                fs.commits += commits
            fs.stalls += total
            for t, v in stalls.items():
                if t != "TOTAL_STALLS":
                    fs.by_type[t] = fs.by_type.get(t, 0) + int(v)
            packets.append((total, pc, commits, stalls, name))

    print("\n== functions (cycles = commits + stall cycles), kernel functions first")
    rows = sorted(funcs.items(), key=lambda kv: kv[1].cycles, reverse=True)
    kernel_cycles = 0
    kernel_types: dict[str, int] = {}
    print(f"{'function':48s} {'cycles':>12s} {'commits':>10s} {'stalls':>10s} {'stall%':>7s}")
    for name, fs in rows:
        if not args.all_functions and not is_kernel(name, pattern):
            continue
        share = 100.0 * fs.stalls / fs.cycles if fs.cycles else 0.0
        print(f"{name[:48]:48s} {fs.cycles:12d} {fs.commits:10d} {fs.stalls:10d} {share:6.1f}%")
        kernel_cycles += fs.cycles
        for t, v in fs.by_type.items():
            kernel_types[t] = kernel_types.get(t, 0) + v

    print("\n== stall types of the kernel functions")
    for t, v in sorted(kernel_types.items(), key=lambda kv: kv[1], reverse=True):
        share = 100.0 * v / kernel_cycles if kernel_cycles else 0.0
        print(f"{t:40s} {v:12d} {share:6.1f}% of the kernel cycles")
    help_core = pa.get("core_packet_profile_help", {})
    for t in list(kernel_types)[:8]:
        h = help_core.get(t)
        if h and h.get("Hint_to_programmer", "none") != "none":
            print(f"  {t}: {h.get('Description', '')} Hint: {h.get('Hint_to_programmer')}")

    print(f"\n== the {args.top} packets with the most stall cycles (kernel functions)")
    packets.sort(key=lambda p: p[0], reverse=True)
    shown = 0
    for total, pc, commits, stalls, name in packets:
        if total == 0 or (not args.all_functions and not is_kernel(name, pattern)):
            continue
        types = ", ".join(f"{t}={v}" for t, v in stalls.items() if t != "TOTAL_STALLS")
        print(f"0x{pc:08x} {name[:32]:32s} stalls {total:8d} commits {commits:7d} [{types}]")
        print(f"    {packet_text(disasm, pc)}")
        shown += 1
        if shown >= args.top:
            break

    stats = pa.get("stats", {})
    wanted = ("COMMITTED_PKT_ANY", "HVX_PKT_THREAD", "HVX_PKT_ANY", "DCACHE_DEMAND_MISS", "L2_DU_LOAD_MISS",
              "AXI_READ_REQUEST", "AXI_WRITE_REQUEST", "DU_UNALIGNED_LOAD", "DU_UNALIGNED_STORE")
    print("\n== counters of the full run")
    for entry in stats.values():
        if entry.get("Name") in wanted:
            print(f"{entry['Name']:32s} {int(entry.get('Value', 0)):12d}")


if __name__ == "__main__":
    main()
