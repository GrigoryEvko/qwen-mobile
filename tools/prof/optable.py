#!/usr/bin/env python3
"""Per-op tables of a Hexagon profile log: the time, the bytes, the FLOPs, the rates and the clock.

The backend prints one ``profile-op`` line for each op and one ``OPBATCH`` line for each DSP batch
when ``GGML_HEXAGON_PROFILE`` is set (mode 1 or mode 2). This tool reads those lines and prints
four tables. It does not need a phone:

- graphs:  the time of each forward pass, inside and outside the ops
- ops:     the time, the bytes, the FLOPs and the rates of each op type or weight shape
- top:     the slowest ops of one graph
- batches: the prologue, the epilogue and the gaps of each DSP batch

Usage:
    tools/prof/optable.py graphs LOG
    tools/prof/optable.py ops LOG --graphs 4-11 [--by type|shape|tensor] [--top N]
    tools/prof/optable.py top LOG --graph 0 [--top 20]
    tools/prof/optable.py batches LOG --graphs 4-11

A graph is one forward pass. It starts at the op that has the destination ``attn_norm-0`` (the
norm of layer 0), thus a prefill ubatch without logits also starts a graph. The ops before the
first such op are graph 0. Change the marker with ``--start``. Graph numbers start at 0.
``--graphs 4-11`` selects graphs 4 thru 11, and ``--graphs 0,3`` selects two graphs. A value in the
ops table is the median over the selected graphs of the sum in one graph, thus a column of us is
the time of one graph.

The bytes of an op are the bytes that its tensors hold, not the bytes of the full buffers:
- A tensor that has two names in one op (for example ``cache_s_l0 (reshaped)`` and
  ``cache_s_l0 (view)``) is one read. The backend finds a tensor by its data pointer and gives it
  one descriptor. Thus, such an op reads its state one time and writes it one time.
- The src1 of CPY is its destination. The op does not read src1.
- The destination of SET_ROWS is the full cache, but the op writes only one row for each source
  row. GET_ROWS reads only the rows that it gives. Thus, the tool counts the rows for these two
  ops, not the full tensor.
A rate that is much more than the roofline of 51 to 56 GB/s shows a wrong byte count, not a fast op.

The FLOPs are 2*k*m*n for each weight of a matrix multiplication, 2*n_q*n_kv*heads*(dk+dv) for
flash attention and 6*T*dk*dv*heads for the chunked delta rule. The flash attention count is the
dense count. The kernel can skip the masked blocks. The delta rule count is the count of the
recurrent form. Other ops have no FLOP count.

The clock of an op is its cycles over its microseconds. The cycles come from the DSP cycle counter
and the microseconds come from the timer. Thus, if the DSP clock decreases, this column shows it.
If the time of an op increases at the same clock, its cycle count increases, and the cause is in
the memory or in the work, not in the clock.

The batch times use the cycle counter. The OPBATCH line gives the full 64-bit start. An op line
gives the low 32 bits of its start. A gap between two batches is correct only if the counter
continues while the DSP waits. Compare the sum with the time of the host before you use it.

The run time is O(lines of the log). The tool only reads the log.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# The bytes of one element of each type. A block type gives the bytes of one block over its size.
TYPE_BYTES: dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "i64": 8.0, "i32": 4.0, "i16": 2.0, "i8": 1.0,
    "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "iq4_nl": 18 / 32, "mxfp4": 17 / 32,
    "q4_k": 144 / 256, "q6_k": 210 / 256, "q8_k": 292 / 256,
}

# The clock that converts cycles to time when a graph has no batch line. The DSP of the phone
# (HTP0, v79) runs at this clock in each profile of 2026-09.
DEFAULT_MHZ = 2112.0

# The op line and the batch line of the backend. The PMU list (profile mode 2) is optional.
OP_RE = re.compile(
    r"profile-op (?P<op>[A-Z_0-9+]+)\|(?P<names>.*?)\|(?P<dims>[0-9: x>-]+)\|(?P<types>[\w: x>-]+)\|"
    r"(?P<strides>[0-9: x>!-]+)\|(?P<kernel>.*?)\|usec (?P<usec>\d+) cycles (?P<cycles>\d+)"
    r"(?: start (?P<start>\d+))?")
BATCH_RE = re.compile(
    r"profile-op OPBATCH\|----\|n-ops (?P<n>\d+)\|.*?\|usec (?P<usec>\d+) cycles (?P<cycles>\d+)"
    r" start (?P<start>\d+)")
LAYER_RE = re.compile(r"blk\.\d+\.")
INDEX_RE = re.compile(r"-\d+$|_l\d+$|_\d+$")

WRAP = 1 << 32


@dataclass
class Op:
    """One profile-op line.

    Attributes:
        op: The op name, with a + for a fused op (for example MUL_MAT+ADD)
        srcs: The names of the sources
        dst: The name of the destination
        sdims: The dimensions of each source
        ddims: The dimensions of the destination
        stypes: The type of each source
        dtype: The type of the destination
        kernel: The kernel field of the line (for example hmx-tiled, or ---- for none)
        usec: The time of the op in microseconds
        cycles: The DSP cycles of the op
        start: The low 32 bits of the cycle counter at the start of the op, or None
        batch: The index of the batch of the op, or -1 when the log has no batch line before it
    """

    op: str
    srcs: list[str]
    dst: str
    sdims: list[list[int]]
    ddims: list[int]
    stypes: list[str]
    dtype: str
    kernel: str
    usec: int
    cycles: int
    start: int | None
    batch: int = -1
    _bytes: float | None = field(default=None, repr=False)

    @property
    def weights(self) -> list[int]:
        """The indices of the model weights at the start of the sources (two for MUL_MAT_NX)."""
        out = []
        for i, n in enumerate(self.srcs[: len(self.sdims)]):
            if not n.endswith(".weight"):
                break
            out.append(i)
        return out

    @property
    def is_weight_mm(self) -> bool:
        """True for a matrix multiplication that has a weight of the model as src0."""
        return any(p.startswith("MUL_MAT") for p in self.op.split("+")) and bool(self.weights) \
            and len(self.sdims) > len(self.weights)

    @property
    def is_rotation(self) -> bool:
        """True for a Hadamard rotation of the KV cache, as MUL_MAT or as the op FWHT."""
        return self.op == "FWHT" or any("_rot#" in n or n.endswith("_rot") for n in self.srcs)

    def op_class(self) -> str:
        """The class of the op. It is a class of build/bench-kv/stage.py or the op type."""
        if self.is_rotation:
            return "FWHT" if self.op == "FWHT" or self.kernel.startswith("----") else "ROT MUL_MAT"
        if self.is_weight_mm:
            return "W MUL_MAT"
        return self.op

    def shape_key(self) -> str:
        """The op type with the weight names, the shape and the row count, or the class of the op."""
        if self.is_weight_mm:
            names = "+".join(LAYER_RE.sub("", self.srcs[i]).removesuffix(".weight") for i in self.weights)
            k, m = self.sdims[0][0], self.sdims[0][1]
            n = _prod(self.sdims[len(self.weights)][1:])
            return f"{self.op} {names} {k}x{m} {self.stypes[0]} n{n} {self.kernel.split(' ')[0]}"
        return self.op_class()

    def tensor_key(self) -> str:
        """The class of the op with the name of its destination, without the layer index."""
        dst = self.dst.split(" (")[0]
        return f"{self.op_class()} {INDEX_RE.sub('', LAYER_RE.sub('', dst))}"

    @property
    def nbytes(self) -> float:
        """The bytes that the op reads and writes, by the rules of the module docstring."""
        if self._bytes is None:
            self._bytes = _op_bytes(self)
        return self._bytes

    @property
    def flops(self) -> float:
        """The FLOPs of the op, or 0 when the tool has no count for the op type."""
        return _op_flops(self)


@dataclass
class Batch:
    """One OPBATCH line and its ops.

    Attributes:
        n_ops: The op count that the line gives
        usec: The time of the batch in microseconds
        cycles: The DSP cycles of the batch
        start: The 64-bit cycle counter at the start of the batch
        ops: The ops of the batch
        graph: The graph of the first op of the batch
    """

    n_ops: int
    usec: int
    cycles: int
    start: int
    ops: list[Op] = field(default_factory=list)
    graph: int = -1

    @property
    def mhz(self) -> float:
        """The DSP clock over the batch."""
        return self.cycles / self.usec if self.usec else DEFAULT_MHZ

    def split(self) -> tuple[int, int, int, int]:
        """The cycles of the prologue, of the ops, of the gaps between the ops and of the epilogue.

        The prologue is the time from the start of the batch to the start of its first op: the
        cache flush, the buffer mappings and the wakeup of the workers. The epilogue is the time
        from the end of the last op to the end of the batch: the queue flush and the cache flush.

        Returns:
            Four cycle counts. Their sum is the cycles of the batch.
        """
        op_cyc = sum(o.cycles for o in self.ops)
        if not self.ops or self.ops[0].start is None or self.ops[-1].start is None:
            return 0, op_cyc, 0, self.cycles - op_cyc
        base = self.start % WRAP
        first = (self.ops[0].start - base) % WRAP
        last = self.ops[-1]
        end = ((last.start - base) % WRAP) + last.cycles
        return first, op_cyc, end - first - op_cyc, self.cycles - end


def _prod(xs: list[int]) -> int:
    """The product of a list of dims, 1 for an empty list."""
    out = 1
    for x in xs:
        out *= x
    return out


def _dims(group: str) -> list[int]:
    """The integers of one ``ne0:ne1:...`` group, or [0] when the group does not parse."""
    try:
        return [int(x) for x in group.split(":")]
    except ValueError:
        return [0]


def _split(text: str) -> tuple[list[str], str]:
    """Divide an ``a x b -> c`` field into its sources and its destination."""
    srcs, _, dst = text.partition(" -> ")
    return [p.strip() for p in srcs.split(" x ")], dst.strip()


def _tbytes(dims: list[int], typ: str) -> float:
    """The bytes of the elements of one tensor. An unknown type counts as 4 bytes."""
    return _prod(dims) * TYPE_BYTES.get(typ, 4.0)


def _base(name: str) -> str:
    """The name of a tensor without the view suffixes, thus two views of one tensor have one name."""
    return name.split(" (")[0]


def _op_bytes(o: Op) -> float:
    """The bytes that one op reads and writes. O(sources)."""
    parts = o.op.split("+")
    dst_base = _base(o.dst)
    read: dict[str, float] = {}
    for i, (n, d) in enumerate(zip(o.srcs, o.sdims)):
        typ = o.stypes[i] if i < len(o.stypes) else "f32"
        if "CPY" in parts and i == 1:
            continue
        if "SET_ROWS" in parts and _base(n) == dst_base:
            continue
        b = _tbytes(d, typ)
        if parts[0] == "GET_ROWS" and i == 0:
            b = _tbytes(o.ddims, typ)
        # A view without a name (" (reshaped) (view)") has an empty base name. Keep each such view
        # as a different tensor.
        read.setdefault(_base(n) or f"#{i}", b)
    write = _tbytes(o.ddims, o.dtype)
    if "SET_ROWS" in parts and o.sdims:
        write = _tbytes(o.sdims[0], o.dtype)
    return sum(read.values()) + write


def _op_flops(o: Op) -> float:
    """The FLOPs of one op by the rules of the module docstring. O(sources)."""
    parts = o.op.split("+")
    if o.is_weight_mm:
        w = o.weights
        n = _prod(o.sdims[len(w)][1:])
        return sum(2.0 * o.sdims[i][0] * o.sdims[i][1] * n for i in w)
    if parts[0] == "MUL_MAT" and len(o.sdims) >= 2 and not o.kernel.startswith("----"):
        a, b = o.sdims[0], o.sdims[1]
        return 2.0 * a[0] * _prod(a[1:2]) * _prod(b[1:])
    if "FLASH_ATTN_EXT" in parts and len(o.sdims) >= 3:
        q, k, v = o.sdims[0], o.sdims[1], o.sdims[2]
        nq, nh = (q[1] if len(q) > 1 else 1), (q[2] if len(q) > 2 else 1)
        nkv = k[1] if len(k) > 1 else 1
        return 2.0 * nq * nkv * nh * (q[0] + v[0])
    if parts[0] == "GATED_DELTA_NET" and len(o.sdims) >= 3:
        q, v = o.sdims[0], o.sdims[2]
        heads = v[1] if len(v) > 1 else 1
        tokens = v[2] if len(v) > 2 else 1
        return 6.0 * tokens * q[0] * v[0] * heads
    return 0.0


def parse(path: str, start_marker: str) -> tuple[list[Batch], list[list[Op]]]:
    """Read a profile log into its batches and its graphs.

    Args:
        path: The log file
        start_marker: The text of the op line that starts a graph

    Returns:
        The batches in log order, and the graphs, each a list of ops in log order

    Raises:
        OSError: If the tool cannot read the log
    """
    batches: list[Batch] = []
    graphs: list[list[Op]] = []
    with open(path, errors="replace") as f:
        for raw in f:
            if "profile-op " not in raw:
                continue
            m = BATCH_RE.search(raw)
            if m:
                batches.append(Batch(int(m["n"]), int(m["usec"]), int(m["cycles"]), int(m["start"])))
                continue
            m = OP_RE.search(raw)
            if not m:
                continue
            srcs, dst = _split(m["names"])
            sd, dd = _split(m["dims"])
            st, dt = _split(m["types"])
            o = Op(m["op"], srcs, dst, [_dims(x) for x in sd], _dims(dd), st, dt, m["kernel"],
                   int(m["usec"]), int(m["cycles"]), int(m["start"]) if m["start"] else None)
            if start_marker in raw or not graphs:
                graphs.append([])
            graphs[-1].append(o)
            if batches:
                o.batch = len(batches) - 1
                batches[-1].ops.append(o)
                if batches[-1].graph < 0:
                    batches[-1].graph = len(graphs) - 1
    return batches, graphs


def select(spec: str | None, count: int) -> list[int]:
    """The graph indices of a ``--graphs`` value such as ``4-11`` or ``0,3``, or all for None.

    Raises:
        ValueError: If an index is out of range
    """
    if not spec:
        return list(range(count))
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    bad = [i for i in out if not 0 <= i < count]
    if bad:
        raise ValueError(f"the log has {count} graphs (0 thru {count - 1}), not {bad}")
    return out


def rows_of(g: list[Op]) -> int:
    """The token count of a graph: the columns of the destination of its first op."""
    return _prod(g[0].ddims[1:]) if g else 0


def fmt_rate(num: float, us: float, scale: float) -> str:
    """A rate as a string, or a dash when it has no value."""
    return f"{num / us / scale:8.2f}" if num and us else f"{'-':>8}"


def cmd_graphs(batches: list[Batch], graphs: list[list[Op]], _: argparse.Namespace) -> int:
    """Print one row per graph: the batch time, the op time by class, the clock and the gaps."""
    by_graph: dict[int, list[Batch]] = defaultdict(list)
    for b in batches:
        by_graph[b.graph].append(b)
    print(f"{'g':>3} {'rows':>5} {'ops':>4} {'nb':>3} {'batch ms':>9} {'op ms':>8} {'W mm ms':>8}"
          f" {'W MHz':>6} {'other ms':>9} {'oth MHz':>7} {'prolog':>7} {'in-gap':>7} {'epilog':>7}"
          f" {'b-gap':>7} {'next':>8}")

    def clk(ops: list[Op]) -> str:
        """The clock of a group of ops, cycles over microseconds."""
        us = sum(o.usec for o in ops)
        return f"{sum(o.cycles for o in ops) / us:.0f}" if us else "-"

    for gi, g in enumerate(graphs):
        bs = by_graph.get(gi, [])
        wmm = [o for o in g if o.is_weight_mm]
        oth = [o for o in g if not o.is_weight_mm]
        pro = inn = epi = 0
        for b in bs:
            p, _, i, e = b.split()
            pro, inn, epi = pro + p, inn + i, epi + e
        mhz = statistics.median(b.mhz for b in bs) if bs else DEFAULT_MHZ
        gaps = [bs[i + 1].start - bs[i].start - bs[i].cycles for i in range(len(bs) - 1)]
        later = by_graph.get(gi + 1)
        nxt = f"{(later[0].start - bs[-1].start - bs[-1].cycles) / mhz / 1000:8.2f}" if bs and later else ""
        print(f"{gi:3d} {rows_of(g):5d} {len(g):4d} {len(bs):3d} {sum(b.usec for b in bs) / 1000:9.2f}"
              f" {sum(o.usec for o in g) / 1000:8.2f} {sum(o.usec for o in wmm) / 1000:8.2f} {clk(wmm):>6}"
              f" {sum(o.usec for o in oth) / 1000:9.2f} {clk(oth):>7} {pro / mhz / 1000:7.2f}"
              f" {inn / mhz / 1000:7.2f} {epi / mhz / 1000:7.2f} {sum(gaps) / mhz / 1000:7.2f} {nxt:>8}")
    print()
    print("batch ms  the sum of the OPBATCH times of the graph. op ms: the sum of the op times.")
    print("W mm      the matrix multiplications that have a weight as src0. MHz: cycles over microseconds.")
    print("prolog    batch start to first op, in-gap: between the ops, epilog: last op to batch end.")
    print("b-gap     between the batches of the graph, next: to the first batch of the next graph.")
    print("          The gaps use the cycle counter at the median batch clock, in ms.")
    return 0


def cmd_ops(_: list[Batch], graphs: list[list[Op]], a: argparse.Namespace) -> int:
    """Print the ops table: one row per group, the median over the selected graphs."""
    sel = select(a.graphs, len(graphs))
    key = {"type": Op.op_class, "shape": Op.shape_key, "tensor": Op.tensor_key}[a.by]
    per: dict[str, list[list[float]]] = defaultdict(lambda: [[0.0] * 5 for _ in sel])
    for j, gi in enumerate(sel):
        for o in graphs[gi]:
            r = per[key(o)][j]
            r[0] += 1
            r[1] += o.usec
            r[2] += o.nbytes
            r[3] += o.flops
            r[4] += o.cycles
    rows = []
    for k, vals in per.items():
        med = [statistics.median(v[i] for v in vals) for i in range(5)]
        rows.append((k, *med))
    rows.sort(key=lambda r: -r[2])
    total_us = sum(r[2] for r in rows)
    width = min(64, max(len(r[0]) for r in rows))
    print(f"graphs {sel[0]}..{sel[-1]} ({len(sel)}), rows {rows_of(graphs[sel[0]])}, the median of each"
          f" value over the graphs")
    print(f"{'group':<{width}} {'n':>5} {'us':>10} {'%':>5} {'MB':>9} {'GB/s':>8} {'GFLOP':>9}"
          f" {'TFLOPS':>8} {'MHz':>6}")
    for k, n, us, b, fl, cyc in rows[: a.top]:
        print(f"{k[:width]:<{width}} {n:5.0f} {us:10.0f} {100 * us / total_us:5.1f} {b / 1e6:9.2f}"
              f" {fmt_rate(b, us, 1e3)} {fl / 1e9:9.2f} {fmt_rate(fl, us, 1e6)}"
              f" {cyc / us if us else 0:6.0f}")
    tb = sum(r[3] for r in rows)
    tf = sum(r[4] for r in rows)
    print(f"{'total':<{width}} {sum(r[1] for r in rows):5.0f} {total_us:10.0f} {100:5.0f} {tb / 1e6:9.2f}"
          f" {fmt_rate(tb, total_us, 1e3)} {tf / 1e9:9.2f} {fmt_rate(tf, total_us, 1e6)}")
    return 0


def cmd_top(_: list[Batch], graphs: list[list[Op]], a: argparse.Namespace) -> int:
    """Print the slowest ops of one graph with their bytes and rates."""
    g = graphs[select(str(a.graph), len(graphs))[0]]
    total = sum(o.usec for o in g)
    print(f"graph {a.graph}: {len(g)} ops, {total / 1000:.2f} ms of op time, rows {rows_of(g)}")
    print(f"{'us':>7} {'%':>5} {'MB':>8} {'GB/s':>8} {'TFLOPS':>8}  op")
    for o in sorted(g, key=lambda x: -x.usec)[: a.top]:
        name = f"{o.op} {o.srcs[0]} -> {o.dst}"
        print(f"{o.usec:7d} {100 * o.usec / total:5.1f} {o.nbytes / 1e6:8.2f} {fmt_rate(o.nbytes, o.usec, 1e3)}"
              f" {fmt_rate(o.flops, o.usec, 1e6)}  {name[:90]}")
    return 0


def cmd_batches(batches: list[Batch], graphs: list[list[Op]], a: argparse.Namespace) -> int:
    """Print one row per batch of the selected graphs: its time and the parts of its time."""
    sel = set(select(a.graphs, len(graphs)))
    print(f"{'g':>3} {'b':>4} {'ops':>4} {'batch us':>9} {'MHz':>6} {'prolog':>7} {'ops us':>8}"
          f" {'in-gap':>7} {'epilog':>7} {'gap after':>9}  first op -> last op")
    for i, b in enumerate(batches):
        if b.graph not in sel:
            continue
        p, oc, g, e = b.split()
        mhz = b.mhz
        after = f"{(batches[i + 1].start - b.start - b.cycles) / mhz:9.0f}" if i + 1 < len(batches) else ""
        first = f"{b.ops[0].op} {b.ops[0].dst}" if b.ops else ""
        last = f"{b.ops[-1].op} {b.ops[-1].dst}" if b.ops else ""
        print(f"{b.graph:3d} {i:4d} {b.n_ops:4d} {b.usec:9d} {mhz:6.0f} {p / mhz:7.0f} {oc / mhz:8.0f}"
              f" {g / mhz:7.0f} {e / mhz:7.0f} {after:>9}  {first[:28]} -> {last[:28]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the selected subcommand."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        """Add the arguments of each subcommand."""
        p.add_argument("log")
        p.add_argument("--start", default="-> attn_norm-0|", help="the text of the op line that starts a graph")

    p = sub.add_parser("graphs", help="one row per graph")
    common(p)
    p.set_defaults(fn=cmd_graphs)
    p = sub.add_parser("ops", help="the table of op types or weight shapes")
    common(p)
    p.add_argument("--graphs", default=None, help="graph indices, for example 4-11 or 0,3")
    p.add_argument("--by", default="type", choices=("type", "shape", "tensor"))
    p.add_argument("--top", type=int, default=60)
    p.set_defaults(fn=cmd_ops)
    p = sub.add_parser("top", help="the slowest ops of one graph")
    common(p)
    p.add_argument("--graph", type=int, default=0)
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(fn=cmd_top)
    p = sub.add_parser("batches", help="one row per DSP batch")
    common(p)
    p.add_argument("--graphs", default=None)
    p.set_defaults(fn=cmd_batches)

    a = ap.parse_args(argv)
    try:
        batches, graphs = parse(a.log, a.start)
    except OSError as e:
        print(f"optable.py: cannot read {a.log}: {e.strerror}", file=sys.stderr)
        return 1
    if not graphs:
        print(f"optable.py: no profile-op lines in {a.log}. Run with GGML_HEXAGON_PROFILE=1 and -v.",
              file=sys.stderr)
        return 1
    try:
        return a.fn(batches, graphs, a)
    except ValueError as e:
        print(f"optable.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
