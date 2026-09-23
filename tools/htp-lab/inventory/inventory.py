#!/usr/bin/env python3
"""Make the float instruction inventory of the four DSP libraries.

The script reads the disassembly of libggml-htp-v73.so, -v75.so, -v79.so and
-v81.so. It puts each instruction in one class: qf arithmetic, qf conversion,
IEEE HVX float, int and float conversion, bf16, float min and max, float
compare, integer round and saturate, vlut, vgather, vscatter, HMX, scalar float
and calls to the float routines of the C library. Then it writes CSV files:

- function_classes.csv: the count of each class for each library and function
- function_forms.csv: the count of each float instruction form, the same key
- op_classes.csv: the classes that each HTP op code reaches, for each version
- op_functions.csv: the functions that each HTP op code reaches
- scalar_float.csv: each scalar float instruction and each libm call, with its site
- loops.csv: the loop bodies of each Qwen3.5 function, with packets and density
- hot_loops.csv: the three loop bodies of each Qwen3.5 op variant with the most
  float and HVX instructions
- arch_branches.csv: each __HVX_ARCH__ and __HEXAGON_ARCH__ branch of htp/
- version_diff.csv: the functions whose float classes differ between versions
- form_ops.csv: each float instruction form and the Qwen3.5 op variants that hold it
- priority.csv: the ops in the order of their share of the runtime

The disassembly comes from hexagon-llvm-objdump 19.0.07 in the Snapdragon
image. Give --dis-dir to use disassembly files that exist, or --lib-dir to make
them in a temporary directory with podman.

Assumptions:
- The libraries keep their symbol table, thus each function has a name.
- A function reference in position-independent code is "rX = add(pc,##imm)".
  The target is the address of the packet plus imm.
- The work of an op is in its worker functions. The curated table OP_MAP names
  them. The closure of their callees gives the instruction classes of the op.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

VERSIONS: tuple[str, ...] = ("v73", "v75", "v79", "v81")

SNAPDRAGON_IMAGE = (
    "ghcr.io/snapdragon-toolchain/arm64-android@sha256:"
    "c012b8174f4154088ee027077e9cb80e68cc9a494d46f63454a050fa4789897b"
)
OBJDUMP = "/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump"

# The classes of the inventory, in the column order of the CSV files.
FLOAT_CLASSES: tuple[str, ...] = (
    "qf32_mpy",
    "qf32_wmpy_hf",
    "qf32_addsub",
    "qf16_mpy",
    "qf16_addsub",
    "cvt_qf32_to_sf",
    "cvt_wqf32_to_hf",
    "cvt_qf16_to_hf",
    "cvt_sf_to_qf32",
    "cvt_hf_to_qf16",
    "ieee_hvx_float",
    "int_float_cvt",
    "bf16",
    "fminmax",
    "fcmp",
    "int_mpy_rnd_sat",
    "int_rnd_sat",
    "vlut",
    "vgather",
    "vscatter",
    "hmx",
    "scalar_float",
    "libm_call",
)
OTHER_CLASSES: tuple[str, ...] = ("int_mpy", "hvx_other", "scalar_other")
ALL_CLASSES: tuple[str, ...] = FLOAT_CLASSES + OTHER_CLASSES

# The C library routines that compute a float value. The DSP firmware supplies
# them at run time, thus their result can change with the device.
LIBM_NAMES: frozenset[str] = frozenset(
    {
        "expf", "exp2f", "logf", "powf", "sqrtf", "floorf", "ceilf", "cosf",
        "sinf", "floor", "_Log", "__truncsfhf2", "__extendhfsf2",
        "__hexagon_divdf3", "__hexagon_divsf3", "__hexagon_sqrtf",
        "__hexagon_adddf3", "__hexagon_muldf3", "__hexagon_subdf3",
        "tanhf", "erff", "roundf", "fmaxf", "fminf", "fabsf",
    }
)

# The columns of the loop tables.
LOOP_KEYS: tuple[str, ...] = ("packets", "insns", "nops", "hvx", "float", "vmem", "insn_per_packet", "hvx_per_packet")

# The class of each normalized form. analyze_library fills it.
FORM_CLASS: dict[str, str] = {}

RE_FUNC = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
RE_WORD = re.compile(r"^\s+([0-9a-f]+):\s(.*)$")
RE_CALL = re.compile(r"\bcall 0x([0-9a-f]+) <([^>+]+)(\+0x[0-9a-f]+)?>")
RE_JUMP = re.compile(r"\bjump(?::[nt]+)? 0x([0-9a-f]+) <([^>+]+)(\+0x[0-9a-f]+)?>")
RE_PCREF = re.compile(r"= add\(pc,##(-?0x[0-9a-f]+|-?\d+)\)")
RE_LOOP = re.compile(r"\b(?:p3 = )?(sp[123])?loop([01])\(0x([0-9a-f]+),")
RE_VREG = re.compile(r"\bv\d+\b|\bq[0-3]\b|\bvmem|\bvtmp\b|vgather|vscatter")


@dataclass
class Packet:
    """One instruction packet: its address and its instructions."""

    addr: int
    insns: list[str] = field(default_factory=list)
    endloop: str = ""


@dataclass
class Function:
    """One function of a library: its name, its range and its packets."""

    name: str
    start: int
    end: int = 0
    packets: list[Packet] = field(default_factory=list)


@dataclass
class Loop:
    """One loop of a function: the range of its packets and its kind."""

    func: str
    start: int
    end: int
    kind: str
    packets: list[Packet]


def base_name(name: str) -> str:
    """Remove the ".N" suffix that LTO adds to a local copy of a function."""
    return re.sub(r"\.\d+$", "", name)


def disassemble(lib_dir: Path, out_dir: Path) -> None:
    """Make the disassembly of the four libraries with podman.

    The disassembly of one library takes approximately 15 s.
    Raises RuntimeError if podman or objdump gives an error.
    """
    script = " ; ".join(
        f"{OBJDUMP} -d --no-show-raw-insn --mcpu=hexagon{v} "
        f"--mattr=+{v},+hvx{v},+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp,+hmx,+hmx{v},+audio "
        f"/htplib/libggml-htp-{v}.so > /htpout/{v}.s"
        for v in VERSIONS
    )
    # The mount points must not hide a directory of the image, for example /lib.
    cmd = [
        "podman", "run", "--rm", "--userns=keep-id", "--security-opt", "label=disable",
        "-v", f"{lib_dir.resolve()}:/htplib:ro", "-v", f"{out_dir.resolve()}:/htpout",
        SNAPDRAGON_IMAGE, "bash", "-c", script,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"objdump in the container failed with code {result.returncode}: {result.stderr.strip()}. "
            f"Make sure that podman can get the image {SNAPDRAGON_IMAGE}."
        )


def split_word(text: str) -> tuple[bool, bool, str, list[str]]:
    """Split the text of one instruction word.

    Returns (packet start, packet end, endloop mark, instructions). A duplex
    word holds two instructions, separated by a semicolon.
    """
    starts = "{" in text
    ends = "}" in text
    mark = ""
    m = re.search(r":(endloop01|endloop0|endloop1)", text)
    if m:
        mark = m.group(1)
    body = re.sub(r":endloop0?1?", "", text).replace("{", "").replace("}", "")
    insns = [p.strip() for p in body.split(";") if p.strip()]
    return starts, ends, mark, insns


def parse_disassembly(path: Path) -> list[Function]:
    """Read one objdump file and give its functions with their packets.

    Complexity is O(n) in the number of lines. The words that objdump cannot
    decode are padding after a return, thus the function ignores them.
    """
    funcs: list[Function] = []
    cur: Function | None = None
    pkt: Packet | None = None
    for line in path.read_text().splitlines():
        m = RE_FUNC.match(line)
        if m:
            cur = Function(name=m.group(2), start=int(m.group(1), 16))
            funcs.append(cur)
            pkt = None
            continue
        m = RE_WORD.match(line)
        if not m or cur is None:
            continue
        addr = int(m.group(1), 16)
        text = m.group(2)
        cur.end = addr + 4
        if "<unknown>" in text:
            continue
        starts, ends, mark, insns = split_word(text)
        if starts or pkt is None:
            pkt = Packet(addr=addr)
            cur.packets.append(pkt)
        pkt.insns.extend(insns)
        if ends:
            pkt.endloop = mark
            pkt = None
    return funcs


def is_hvx(insn: str) -> bool:
    """Tell if an instruction uses the HVX unit."""
    return bool(RE_VREG.search(insn))


def classify(insn: str) -> tuple[str, str]:
    """Give the class of one instruction and its normalized form.

    The form replaces the register numbers and the immediates, thus the same
    operation at different sites has the same form.
    """
    form = normalize(insn)
    if insn.startswith("immext") or insn == "nop":
        return "scalar_other", form
    m = RE_CALL.search(insn) or RE_JUMP.search(insn)
    if m and m.group(2).endswith("@plt") and m.group(2)[:-4] in LIBM_NAMES:
        return "libm_call", f"call {m.group(2)[:-4]}"
    # The symbol of a branch target can hold any word, thus remove it first.
    s = re.sub(r"<[^>]*>", "", insn)
    if re.search(r"mxmem|mxclracc|mxswapacc|\bactivation\.|\bweight\.|\bbias\s*=|\bcvt\.|mxshl", s):
        return "hmx", form
    if "vgather" in s:
        return "vgather", form
    if "vscatter" in s:
        return "vscatter", form
    if re.search(r"\bsf(add|sub|mpy|recipa|invsqrta|fixup[dnr]|cmp\.|max|min|make|class|ffma)", s) or re.search(
        r"\bdf(add|sub|mpy|cmp\.|max|min|make|class)", s
    ) or re.search(r"\bconvert_[a-z]+2[a-z]+\(", s):
        if not is_hvx(s):
            return "scalar_float", form
    if not is_hvx(s):
        return "scalar_other", form
    if ".bf" in s:
        return "bf16", form
    if re.search(r"\.qf32 = vmpy\(", s):
        if re.match(r"v\d+:\d+\.qf32", s):
            return "qf32_wmpy_hf", form
        return "qf32_mpy", form
    if re.search(r"\.qf32 = v(add|sub)\(", s):
        return "qf32_addsub", form
    if re.search(r"\.qf16 = vmpy\(", s):
        return "qf16_mpy", form
    if re.search(r"\.qf16 = v(add|sub)\(", s):
        return "qf16_addsub", form
    if re.fullmatch(r"v\d+\.sf = v\d+\.qf32", s):
        return "cvt_qf32_to_sf", form
    if re.fullmatch(r"v\d+\.hf = v\d+:\d+\.qf32", s):
        return "cvt_wqf32_to_hf", form
    if re.fullmatch(r"v\d+\.hf = v\d+\.qf16", s):
        return "cvt_qf16_to_hf", form
    if re.fullmatch(r"v\d+\.qf32 = v\d+\.sf", s):
        return "cvt_sf_to_qf32", form
    if re.fullmatch(r"v\d+\.qf16 = v\d+\.hf", s):
        return "cvt_hf_to_qf16", form
    if re.fullmatch(r"v\d+(:\d+)?\.(sf|hf) = v\d+(:\d+)?\.(w|uw|h|uh|b|ub)", s) or re.fullmatch(
        r"v\d+(:\d+)?\.(w|uw|h|uh|b|ub) = v\d+(:\d+)?\.(sf|hf)", s
    ) or re.search(r"\bvcvt|\bvconv", s):
        return "int_float_cvt", form
    if re.search(r"v(max|min)\(.*\.(sf|hf)", s):
        return "fminmax", form
    if re.search(r"vcmp\.\w+\(v\d+\.(sf|hf)", s):
        return "fcmp", form
    if re.search(r"\.(sf|hf) = v(add|sub|mpy|fmpy|fadd|fneg|abs|fmax|fmin)", s) or "vfneg" in s:
        return "ieee_hvx_float", form
    if re.search(r"vlut(4|16|32)", s):
        return "vlut", form
    is_mpy = re.search(r"\bv(r|d|t)?mp(y|a|s)", s) is not None
    if is_mpy and re.search(r":(rnd|sat)", s):
        return "int_mpy_rnd_sat", form
    if re.search(r":(rnd|sat)\b|\bvround|\bvsat", s):
        return "int_rnd_sat", form
    if is_mpy:
        return "int_mpy", form
    return "hvx_other", form


def normalize(insn: str) -> str:
    """Replace the registers and immediates of an instruction by placeholders."""
    s = re.sub(r"<[^>]*>", "", insn)
    s = re.sub(r"\bv\d+:\d+\b", "W", s)
    s = re.sub(r"\bv\d+\b", "V", s)
    s = re.sub(r"\bq\d\b", "Q", s)
    s = re.sub(r"\br\d+:\d+\b", "RR", s)
    s = re.sub(r"\br\d+\b", "R", s)
    s = re.sub(r"\bp\d\b", "P", s)
    s = re.sub(r"##?-?0x[0-9a-f]+|##?-?\d+", "#I", s)
    s = re.sub(r"\b0x[0-9a-f]+\b", "A", s)
    return re.sub(r"\s+", " ", s).strip()


def real_insns(pkt: Packet) -> list[str]:
    """Give the instructions of a packet, without the constant extenders."""
    return [i for i in pkt.insns if not i.startswith("immext")]


def function_edges(funcs: list[Function]) -> dict[str, set[str]]:
    """Give the callees of each function.

    The edges are direct calls, jumps to a different function (tail calls) and
    function addresses that the code makes with add(pc,##imm). Complexity is
    O(n log m), n instructions and m functions.
    """
    starts = {f.start: base_name(f.name) for f in funcs}
    edges: dict[str, set[str]] = defaultdict(set)
    for f in funcs:
        me = base_name(f.name)
        for pkt in f.packets:
            for insn in pkt.insns:
                for rx in (RE_CALL, RE_JUMP):
                    m = rx.search(insn)
                    if m:
                        target = int(m.group(1), 16)
                        if not (f.start <= target < f.end) and target in starts:
                            edges[me].add(starts[target])
                m = RE_PCREF.search(insn)
                if m:
                    target = pkt.addr + int(m.group(1), 0)
                    if target in starts and starts[target] != me:
                        edges[me].add(starts[target])
    return edges


def closure(roots: Iterable[str], edges: dict[str, set[str]], stop: frozenset[str]) -> set[str]:
    """Give all functions that the roots can reach, without the stop set.

    Complexity is O(V + E) of the call graph.
    """
    seen: set[str] = set()
    todo = [r for r in roots]
    while todo:
        f = todo.pop()
        if f in seen or f in stop:
            continue
        seen.add(f)
        todo.extend(edges.get(f, ()))
    return seen


def find_loops(f: Function) -> list[Loop]:
    """Give the loops of a function.

    A hardware loop goes from the target of loopN() to the packet with the
    endloopN mark. A software loop goes from the target of a backward jump to
    the packet of that jump. Complexity is O(p^2) in the worst case, p packets.
    """
    loops: list[Loop] = []
    by_addr = {p.addr: i for i, p in enumerate(f.packets)}
    for i, pkt in enumerate(f.packets):
        for insn in pkt.insns:
            m = RE_LOOP.search(insn)
            if m:
                level = m.group(2)
                target = int(m.group(3), 16)
                if target not in by_addr:
                    continue
                j0 = by_addr[target]
                for j in range(j0, len(f.packets)):
                    mark = f.packets[j].endloop
                    if mark == f"endloop{level}" or mark == "endloop01":
                        loops.append(Loop(f.name, target, f.packets[j].addr, f"hw{level}", f.packets[j0 : j + 1]))
                        break
            m = RE_JUMP.search(insn)
            if m and "jumpr" not in insn:
                target = int(m.group(1), 16)
                if f.start <= target <= pkt.addr and target in by_addr:
                    j0 = by_addr[target]
                    loops.append(Loop(f.name, target, pkt.addr, "sw", f.packets[j0 : i + 1]))
    return loops


RE_VTOK = re.compile(r"\bv(\d+)(?::(\d+))?\b")
RE_ASSIGN = re.compile(r"^(.*?)\s*(\+=|-=|\|=|&=|=)\s*(.*)$")
NON_ARITH = frozenset({"scalar_float", "libm_call", "hmx"})


def vregs(text: str) -> list[int]:
    """Give the HVX register numbers in a text. A pair v1:0 gives 1 and 0."""
    out: list[int] = []
    for hi, lo in RE_VTOK.findall(text):
        out.append(int(hi))
        if lo:
            out.append(int(lo))
    return out


def operand_groups(text: str) -> list[list[int]]:
    """Give the HVX operands of the right side, one register group for each operand."""
    return [[int(hi)] + ([int(lo)] if lo else []) for hi, lo in RE_VTOK.findall(text)]


def result_type(form: str) -> str:
    """Give the element type of the result of a normalized form, or "" if it has none."""
    m = re.match(r"^[VW](?:\.(\w+))?\s*(?:\+=|-=|=)", form)
    return (m.group(1) or "") if m else ""


def operand_types(form: str) -> list[str]:
    """Give the element type of each HVX operand of a normalized form.

    An accumulate form reads its destination first. An operand without a type
    suffix gives "".
    """
    m = re.match(r"^(.*?)\s*(\+=|-=|\|=|&=|=)\s*(.*)$", form)
    if not m:
        return []
    lhs, op, rhs = m.groups()
    types = [t or "" for t in re.findall(r"\b[VW](?:\.(\w+))?", rhs)]
    if op != "=":
        types = [result_type(form.replace(op, "=", 1))] + types
    return types


def insn_defs_uses(insn: str) -> tuple[list[int], list[list[int]], bool]:
    """Give the defined registers, the operand groups and the flag for a conditional def.

    A store or a vscatter defines no register. An accumulate form (+=) also
    reads its destination.
    """
    s = re.sub(r"<[^>]*>", "", insn).strip()
    cond = s.startswith("if ")
    body = re.sub(r"^if \([^)]*\)\s*", "", s)
    m = RE_ASSIGN.match(body)
    if not m or body.startswith("vmem") or "vscatter" in body:
        return [], operand_groups(body), cond
    lhs, op, rhs = m.groups()
    dst = vregs(lhs)
    uses = operand_groups(rhs)
    if op != "=" and dst:
        uses = [dst] + uses
    return dst, uses, cond


def build_blocks(f: Function, loops: list[Loop]) -> tuple[list[tuple[int, int]], dict[int, set[int]]]:
    """Split a function into basic blocks and give the successors of each block.

    Returns the list of (first packet, last packet) index pairs and a map from
    a block index to its successor block indices. A call does not end a block.
    A jump through a register (a switch or a return) has no known successor.
    """
    n = len(f.packets)
    by_addr = {p.addr: i for i, p in enumerate(f.packets)}
    loop_back = {by_addr[lp.end]: by_addr[lp.start] for lp in loops if lp.kind.startswith("hw")}
    leaders = {0}
    for i, pkt in enumerate(f.packets):
        ends_block = False
        for insn in pkt.insns:
            m = RE_JUMP.search(insn)
            if m and int(m.group(1), 16) in by_addr:
                leaders.add(by_addr[int(m.group(1), 16)])
                ends_block = True
            elif "jumpr" in insn or "dealloc_return" in insn or m:
                ends_block = True
            lm = RE_LOOP.search(insn)
            if lm and int(lm.group(3), 16) in by_addr:
                leaders.add(by_addr[int(lm.group(3), 16)])
        if pkt.endloop:
            ends_block = True
        if ends_block and i + 1 < n:
            leaders.add(i + 1)
    starts = sorted(leaders)
    blocks = [(s, (starts[k + 1] - 1) if k + 1 < len(starts) else n - 1) for k, s in enumerate(starts)]
    block_of = {s: k for k, (s, _) in enumerate(blocks)}
    succ: dict[int, set[int]] = defaultdict(set)
    for k, (_, last) in enumerate(blocks):
        pkt = f.packets[last]
        falls = True
        for insn in pkt.insns:
            m = RE_JUMP.search(insn)
            unconditional = not insn.lstrip().startswith("if") and "cmp." not in insn
            if m:
                t = int(m.group(1), 16)
                if t in by_addr:
                    succ[k].add(block_of[by_addr[t]])
                if unconditional:
                    falls = False
            elif ("jumpr" in insn or "dealloc_return" in insn) and not insn.lstrip().startswith("if"):
                falls = False
        if last in loop_back:
            succ[k].add(block_of[loop_back[last]])
        if falls and last + 1 < n:
            succ[k].add(block_of[last + 1])
    return blocks, succ


def chain_signatures(f: Function) -> Counter[str]:
    """Give the float dataflow signatures of a function.

    A signature is the form of one float instruction plus, for each operand,
    the forms of the float instructions whose results can reach it. A value
    from a load, a permute or a splat has the tag "x". A register copy passes
    the tags of its source. An iterative pass over the basic blocks finds the
    definitions that can arrive at each operand. Complexity is O(b * p) for b
    blocks and p packets, with a small constant for the 32 registers.
    """
    loops = find_loops(f)
    blocks, succ = build_blocks(f, loops)
    pred: dict[int, set[int]] = defaultdict(set)
    for k, ss in succ.items():
        for s in ss:
            pred[s].add(k)
    empty: tuple[frozenset[str], ...] = tuple(frozenset({"x"}) for _ in range(32))
    state_in: dict[int, tuple[frozenset[str], ...]] = {0: empty}
    state_out: dict[int, tuple[frozenset[str], ...]] = {}
    info = [[(classify(i), insn_defs_uses(i)) for i in p.insns] for p in f.packets]

    def run_block(k: int, st: list[frozenset[str]], record: Counter[str] | None) -> list[frozenset[str]]:
        first, last = blocks[k]
        for j in range(first, last + 1):
            writes: list[tuple[list[int], frozenset[str], bool]] = []
            for (cl, form), (dst, uses, cond) in info[j]:
                arith = cl in FLOAT_CLASSES and cl not in NON_ARITH
                if arith and record is not None:
                    parts = []
                    for grp, want in zip(uses, operand_types(form)):
                        tags = frozenset().union(*(st[r] for r in grp if r < 32))
                        ftags = sorted(t for t in tags if t != "x" and result_type(t) == want)
                        parts.append("/".join(ftags) if ftags else "x")
                    record[f"{form} <= {' | '.join(parts)}"] += 1
                if not dst:
                    continue
                if arith:
                    tag = frozenset({form})
                elif form == "V = V" and uses:
                    tag = frozenset().union(*(st[r] for r in uses[0] if r < 32))
                else:
                    tag = frozenset({"x"})
                writes.append((dst, tag, cond))
            for insn in f.packets[j].insns:
                if RE_CALL.search(insn):
                    st = [frozenset({"x"})] * 32
            for dst, tag, cond in writes:
                for r in dst:
                    if r < 32:
                        st[r] = st[r] | tag if cond else tag
        return st

    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed = False
        rounds += 1
        for k in range(len(blocks)):
            ins = [state_out[p] for p in pred.get(k, ()) if p in state_out]
            if k == 0:
                ins.append(empty)
            if not ins:
                continue
            merged = tuple(frozenset().union(*(s[r] for s in ins)) for r in range(32))
            if state_in.get(k) == merged and k in state_out:
                continue
            state_in[k] = merged
            out = tuple(run_block(k, list(merged), None))
            if state_out.get(k) != out:
                state_out[k] = out
                changed = True
    sigs: Counter[str] = Counter()
    for k in range(len(blocks)):
        if k in state_in:
            run_block(k, list(state_in[k]), sigs)
    return sigs


# ---------------------------------------------------------------------------
# The curated map of the ops. The source is htp/main.c (execute_op) and the
# dispatch code of each op file. A root that ends with "!" adds only that
# function. A root without "!" adds the function and all its callees.
# ---------------------------------------------------------------------------

# Functions of the infrastructure. They move data or wake threads. The
# closure does not go into them.
INFRA: frozenset[str] = frozenset(
    {
        "work_queue_run_async", "work_queue_run", "dma_queue_push", "dma_queue_pop",
        "dma_queue_flush", "dma_queue_push_single_1d", "HAP_debug_v2@plt", "_HAP_debug_v2",
        "htp_trace_event_stop", "htp_tensor_mdev_partition", "htp_tensor_mdev_rows_per_chunk",
        "htp_tensor_can_row_partition", "htp_mm_get_tiled_row_stride", "memset@plt", "memcpy@plt",
        "qurt_futex_wake@plt", "__hexagon_udivsi3@plt", "__hexagon_umodsi3@plt",
        "__hexagon_udivdi3@plt", "__hexagon_modsi3@plt", "__wrap_free@plt", "__wrap_memalign@plt",
    }
)

HMX_MM_Q8_0: tuple[str, ...] = (
    "hmx_mm_2d_f32!", "dequantize_tiled_worker_loop_q8_0", "dequantize_tiled_weight_chunk_to_fp16_tiles!",
    "transfer_activation_chunk_worker_fn", "transfer_activation_chunk_col_chunk_worker_fn",
    "hmx_matmul_worker_fn", "transfer_output_chunk_worker_fn", "transfer_output_chunk_col_chunk_worker_fn",
)
ATTN_F16_MM: tuple[str, ...] = (
    "quantize_f32_f16_flat", "hvx_mm_2d!", "hvx_mv_2d!", "hvx_mm_4d!", "vec_dot_f16_f16_aa_1x1",
    "vec_dot_f16_f16_aa_2x1", "vec_dot_f16_f16_aa_2x2", "vec_dot_f16_f32_uu_1x1", "vec_dot_f16_f16_uu_1x1",
    "hmx_mm_f16_f32_batched_simple!", "convert_f16_worker_loop", "hmx_matmul_worker_fn",
    "transfer_activation_chunk_worker_fn", "transfer_output_chunk_worker_fn",
)

# (op code, variant, use in Qwen3.5 2B and 4B, roots). "-" in the use column
# tells that the Qwen3.5 graph does not send the op.
OP_MAP: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("MUL_MAT", "q8_0-matvec", "decode: each weight matmul of 1 row",
     ("quantize_f32_q8_0_tiled_block", "hvx_mv_2d_repacked_q8_0")),
    ("MUL_MAT", "q8_0-multirow", "speculative verify: 2 to 4 rows",
     ("quantize_f32_q8_0_tiled_compact", "hvx_mm_2d_repacked_q8_0_multirow")),
    ("MUL_MAT", "q8_0-hvx-rows", "fallback: rows > 4 when HMX is off or VTCM is short",
     ("quantize_f32_q8_0_tiled", "hvx_mm_2d_repacked_q8_0", "tiled_vec_dot_q8_0_32x2")),
    ("MUL_MAT", "q8_0-hmx", "prefill: each weight matmul of more than 4 rows", HMX_MM_Q8_0),
    ("MUL_MAT", "f16-kv", "attention with flash attention off: KQ and KQV", ATTN_F16_MM),
    ("MUL_MAT_ADD", "q8_0-matvec", "decode: matmul plus residual (fusion)",
     ("quantize_f32_q8_0_tiled_block", "hvx_mv_2d_repacked_q8_0")),
    ("MUL_MAT_ADD", "q8_0-hmx", "prefill: matmul plus residual (fusion)", HMX_MM_Q8_0),
    ("MUL_MAT_NX", "q8_0-matvec", "decode: gate+up, qkv+z, beta+alpha (fusion)",
     ("quantize_f32_q8_0_tiled_block", "hvx_mm_nx_2d_repacked_q8_0")),
    ("MUL_MAT_NX", "q8_0-hmx", "prefill: gate+up, qkv+z, beta+alpha (fusion)", HMX_MM_Q8_0),
    ("GDN_STATE_STEP", "fused", "decode: delta-rule state of each GDN layer",
     ("op_gdn_state_step!", "gated_delta_net_f32_tg_thread")),
    ("GDN_CONV_STEP", "fused", "decode: causal conv of each GDN layer", ("op_gdn_conv_step!", "gdn_conv_step_thread")),
    ("GDN_CONV_CHUNK", "fused", "prefill: causal conv of each GDN layer", ("op_gdn_conv_chunk!", "gdn_conv_chunk_thread")),
    ("GATED_DELTA_NET", "chunked-hmx", "prefill: delta rule of each GDN layer",
     ("op_gated_delta_net!", "op_gated_delta_net_chunked")),
    ("GATED_DELTA_NET", "sequential", "fallback: state fusion or chunking off",
     ("op_gated_delta_net!", "gated_delta_net_f32_pp_thread", "gated_delta_net_f32_tg_thread")),
    ("SSM_CONV", "f32", "fallback: state fusion off", ("op_ssm_conv_f32!", "ssm_conv_thread_f32_f32_hvx", "ssm_conv_thread_f32_f32")),
    ("RMS_NORM_MUL", "f32", "decode+prefill: attn, post, output, q, k and ssm norms (fusion)",
     ("unary_task_f32_rms_norm_mul",)),
    ("RMS_NORM", "f32", "decode+prefill: the l2 norm of q and k in GDN layers", ("unary_task_f32_rms_norm",)),
    ("SCALE", "f32", "decode+prefill: 1/sqrt(n) after the l2 norm, state zero",
     ("unary_task_f32_scale", "unary_task_f32_tiled_scale")),
    ("UNARY_SIGMOID", "f32", "decode+prefill: beta, attention gate",
     ("unary_task_f32_unary_sigmoid", "unary_task_f32_tiled_unary_sigmoid")),
    ("UNARY_SOFTPLUS", "f32", "decode+prefill: alpha of the GDN gate",
     ("unary_task_f32_unary_softplus", "unary_task_f32_tiled_unary_softplus")),
    ("UNARY_SILU", "f32", "decode+prefill: silu(z) of the gated norm",
     ("unary_task_f32_unary_silu", "unary_task_f32_tiled_unary_silu")),
    ("GLU_SWIGLU", "f32", "decode+prefill: the FFN", ("glu_swiglu_f32_per_thread",)),
    ("ADD", "binary-jobs", "decode+prefill: residuals, alpha+dt (f32)",
     ("binary_job_vector_same_shape", "binary_job_vector_row_broadcast", "binary_job_element_repeat",
      "binary_job_vector_complex", "binary_job_scalar")),
    ("MUL", "binary-jobs", "decode+prefill: gates, norm*silu (f32)",
     ("binary_job_vector_same_shape", "binary_job_vector_row_broadcast", "binary_job_element_repeat",
      "binary_job_vector_complex", "binary_job_scalar")),
    ("ROPE", "imrope-f32", "decode+prefill: q and k of attention layers", ("op_rope!", "rope_job_f32")),
    ("FLASH_ATTN_EXT", "hmx", "decode+prefill: head dim 256 selects HMX at every row count",
     ("op_flash_attn_ext!", "hmx_flash_attn_ext")),
    ("FLASH_ATTN_EXT", "hvx", "fallback: HMX off or VTCM short", ("op_flash_attn_ext!", "flash_attn_ext_f16_thread")),
    ("SOFTMAX", "f32", "attention with flash attention off", ("op_softmax!", "softmax_job_f32")),
    ("SET_ROWS", "f16-kv", "decode+prefill: K and V into the f16 cache",
     ("set_rows_thread_dma_f16_int64_t", "set_rows_thread_dma_f16_int32_t")),
    ("SET_ROWS", "f32", "MTP drafter: the scatter of the reduced head",
     ("set_rows_thread_dma_f32_int32_t", "set_rows_thread_dma_f32_int64_t")),
    ("GET_ROWS", "all", "MTP embedding, output ids, state rows when state fusion is off",
     ("get_rows_thread_f32_int32_t", "get_rows_thread_f32_int64_t", "get_rows_thread_q8_0_int32_t",
      "get_rows_thread_q8_0_int64_t", "get_rows_thread_st_int32_t", "get_rows_thread_st_int64_t")),
    ("CPY", "f32-to-f16", "f32 to f16 copies (cpy_thread_f16_f32 writes f16)", ("op_cpy!", "cpy_thread_f16_f32_sameshape")),
    ("CPY", "f16-to-f32", "f16 to f32 copies (cpy_thread_f32_f16 writes f32)", ("op_cpy!", "cpy_thread_f32_f16_sameshape")),
    ("CPY", "same-type", "decode+prefill: the gate cont, state copies",
     ("op_cpy!", "cpy_thread_f32_sameshape", "cpy_dma_sametype_sameshape", "cpy_thread_f32_reshape",
      "cpy_thread_f32_transpose2d", "cpy_thread_f16_sameshape", "cpy_thread_f16_reshape")),
    ("CONCAT", "all", "MTP drafter: enorm and hnorm. conv state when fusion is off", ("op_concat",)),
    ("REPEAT", "all", "MTP drafter: the fill of the reduced head", ("op_repeat",)),
)

# The op codes without a Qwen3.5 variant. The closure of their entry point
# gives all types and all paths.
OTHER_OPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SUB", ("op_binary",)), ("DIV", ("op_binary",)), ("ADD_ID", ("op_binary",)),
    ("MUL_MAT_ID", ("op_matmul_id",)), ("MUL_MAT_ID_NX", ("op_matmul_id_nx",)),
    ("NORM", ("unary_task_f32_norm", "unary_task_f16_norm")),
    ("L2_NORM", ("unary_task_f32_l2_norm", "unary_task_f16_l2_norm")),
    ("CLAMP", ("unary_task_f32_clamp", "unary_task_f32_tiled_clamp", "unary_task_f16_clamp")),
    ("LEAKY_RELU", ("unary_task_f32_leaky_relu", "unary_task_f32_tiled_leaky_relu")),
    ("SQR", ("unary_task_f32_sqr", "unary_task_f32_tiled_sqr", "unary_task_f16_sqr")),
    ("SQRT", ("unary_task_f32_sqrt", "unary_task_f32_tiled_sqrt", "unary_task_f16_sqrt")),
    ("UNARY_NEG", ("unary_task_f32_unary_neg", "unary_task_f32_tiled_unary_neg")),
    ("UNARY_EXP", ("unary_task_f32_unary_exp", "unary_task_f32_tiled_unary_exp")),
    ("UNARY_GELU", ("unary_task_f32_unary_gelu", "unary_task_f32_tiled_unary_gelu")),
    ("UNARY_GELU_QUICK", ("unary_task_f32_unary_gelu_quick", "unary_task_f32_tiled_unary_gelu_quick")),
    ("UNARY_TANH", ("unary_task_f32_unary_tanh", "unary_task_f32_tiled_unary_tanh")),
    ("UNARY_ABS", ("unary_task_f32_unary_abs", "unary_task_f32_tiled_unary_abs", "unary_task_f16_unary_abs")),
    ("UNARY_LOG", ("unary_task_f32_unary_log", "unary_task_f32_tiled_unary_log", "unary_task_f16_unary_log")),
    ("UNARY_RELU", ("unary_task_f32_unary_relu", "unary_task_f32_tiled_unary_relu")),
    ("TRI", ("unary_task_f32_tri", "unary_task_f32_tiled_tri")),
    ("GLU_SWIGLU_OAI", ("glu_swiglu_oai_f32_per_thread",)), ("GLU_GEGLU", ("glu_geglu_f32_per_thread",)),
    ("GLU_SWIGLU_CLAMP", ("glu_swiglu_clamp_f32_per_thread",)),
    ("SUM_ROWS", ("op_sum_rows",)), ("ARGSORT", ("op_argsort",)), ("CUMSUM", ("op_cumsum",)),
    ("FILL", ("op_fill",)), ("DIAG", ("op_diag",)), ("SOLVE_TRI", ("op_solve_tri",)), ("PAD", ("op_pad",)),
    ("IM2COL", ("op_im2col",)), ("ALLREDUCE", ("op_allreduce",)), ("ALLREDUCE_ADD", ("op_allreduce",)),
    ("CPY_FENCE", ("op_cpy",)),
)

# The measured share of each op in the runtime of the 4B Q8_0 (hvx-packet-rules-v79,
# decode-timeline). Decode streams the full 4.40 GiB file for each token.
PRIORITY: tuple[tuple[int, str, str, str], ...] = (
    (1, "MUL_MAT, MUL_MAT_NX, MUL_MAT_ADD (q8_0-hmx)", "prefill", "181 ms of 536 ms, 34 %, at the HMX compute peak"),
    (2, "GDN_CONV_CHUNK", "prefill", "147 ms, 27 %, 1.27 instructions per packet in the PMU profile"),
    (3, "Elementwise: GLU_SWIGLU, MUL, UNARY_SIGMOID, SCALE, RMS_NORM(_MUL), UNARY_SILU, UNARY_SOFTPLUS", "prefill",
     "114 ms, 21 %, f32 and DDR bound"),
    (4, "GATED_DELTA_NET (chunked-hmx)", "prefill", "78 ms, 15 %"),
    (5, "MUL_MAT, MUL_MAT_NX, MUL_MAT_ADD (q8_0-matvec)", "decode",
     "about 90 % of the token: 4.40 GiB at 47 GB/s, the DDR roofline"),
    (6, "GDN_STATE_STEP", "decode", "one for each GDN layer (18 of 24 in 2B, 24 of 32 in 4B)"),
    (7, "GDN_CONV_STEP", "decode", "one for each GDN layer"),
    (8, "RMS_NORM_MUL, RMS_NORM, SCALE, ADD, MUL, SIGMOID, SOFTPLUS, SILU, GLU_SWIGLU", "decode",
     "about 440 small ops, 2 ms, dispatch bound"),
    (9, "ROPE", "decode+prefill", "attention layers only (6 of 24 in 2B, 8 of 32 in 4B)"),
    (10, "FLASH_ATTN_EXT (hmx) or SOFTMAX", "decode+prefill", "attention layers only"),
    (11, "SET_ROWS f16-kv, CPY f32<->f16, GET_ROWS", "decode+prefill", "small: KV writes and copies"),
)

# The numeric effect of each arch branch, keyed by (file, line). The text
# tells what the two sides compute and if they give different values by
# construction, with the same instruction semantics on each version.
# "same" : the two sides compile to the same instruction sequence.
# "differs" : the two sides compute differently by construction.
# "chain" : the same math. The compiler can remove a conversion on one side
#           only (refer to version_diff.csv).
# "none" : no float effect.
ARCH_NOTES: dict[tuple[str, int], tuple[str, str, str, str]] = {
    ("hvx-base.h", 105): ("hvx_vec_neg_f32", "same",
        "v79+: Q6_Vsf_vfneg_Vsf. v73/v75: xor of the sign bit. The v79/v81 libraries hold no vfneg. The compiler emits the xor.",
        "hvx_exp_f32 (UNARY_EXP, softmax exp, GDN exp)"),
    ("hvx-base.h", 125): ("hvx_vec_f32_to_f16_shuff (all f32->f16)", "differs",
        "v81: qf32 = Vqf32_equals_Vsf(sf), a new opcode. v73-v79: qf32 = vadd(sf, +0). The v73-v79 compiler also joins "
        "the sf = qf32 before it into the vadd, thus the hf comes from the qf32 without the sf rounding. v81 rounds to sf first "
        "(245 sites of sf->qf32 after qf32->sf).",
        "SET_ROWS f16, CPY f32->f16, HMX matmul activations, FA q/k/v, GDN_CONV_CHUNK, GATED_DELTA_NET chunked, int16 activations"),
    ("hvx-base.h", 140): ("hvx_vec_f16_to_f32(_shuff)", "same",
        "v79+: Wsf_vmpy_VhfVhf(x, 1.0). v73/v75: Wqf32_vmpy and two Vsf_equals_Vqf32. The v79 compiler lowers the IEEE form "
        "to the same qf pair.",
        "FA, GDN chunk, tiled dots, norms, sqrt, scale, log, quant"),
    ("hvx-base.h", 188): ("hvx_vec_mpyacc_f32_f16", "chain",
        "v79+: Wsf_vmpyacc_WsfVhfVhf. v73/v75: Wqf32 product, qf32 add of the sf accumulator, round to sf. The compiler "
        "lowers the two sides to the same pair.",
        "HVX flash attention dots and MADs, f16 vec dots (MUL_MAT f16-kv)"),
    ("hvx-base.h", 207): ("hvx_vec_add/sub/mul_f16_f16, hvx_vec_add/sub/mul_f32_f32", "differs",
        "f16 add and sub: v79+ Vhf_vadd gives qf16 = vadd(hf,hf) and hf = qf16 (a qf16 intermediate with an 11-bit "
        "mantissa). v73/v75 widen the two inputs to qf32 (x*1.0, y*-1.0), add in qf32 and round one time to hf. f16 mul: the "
        "same Wqf32 pair on the two sides. f32 add, sub and mul: the same qf32 pair on the two sides. Only the v79/v81 IEEE "
        "form lets the compiler remove the sf rounding inside a chain.",
        "f16: FA softmax (fa_softmax_impl), HVX FA, GATED_DELTA_NET chunked phase C, f16 binary ops. f32: GDN state step, GDN chunk"),
    ("hvx-reduce.h", 47): ("hvx_vec_reduce_sum_f32x4/x2/n_f32", "chain",
        "v79+: each tree step is an IEEE add, rounded to sf. v73/v75: f32x4 and f32x2 keep the sum in qf32 across the tree "
        "and round only the rotated copy. The v79 compiler removes the sf rounding of the partial sum. The binary chains of "
        "gdn_step_token_f32 are equal on the four versions.",
        "GDN_STATE_STEP (gdn_mul_dot*), HVX FA, flat f32 dots, softmax, norms"),
    ("hmx-queue.h", 20): ("HMX_QUEUE_POLL_COUNT", "none", "v81: 2000 polls. Older versions: 1. It changes the schedule only.",
        "HMX queue"),
    ("hvx-fa-kernels.h", 10): ("HVX_OP_ADD/SUB/MUL_F32 of the HVX FA kernels", "chain",
        "v73/v75: explicit qf32 op and Vsf_equals_Vqf32. v79+: IEEE intrinsic. The compiler lowers it to the same pair.",
        "FLASH_ATTN_EXT hvx"),
    ("main.c", 55): ("htp_mmap", "none", "v73: HAP_mmap with a 2 GB limit. v75+: HAP_mmap2.", "buffer map"),
    ("main.c", 75): ("htp_munmap", "none", "v73: HAP_munmap. v75+: HAP_munmap2.", "buffer map"),
    ("main.c", 472): ("DCVS protected bus corners", "none", "v79+: HAP_set_dcvs_v3_protected_bus_corners.", "power"),
    ("main.c", 489): ("HMX power request", "none", "v75+: HAP_power_set_HMX_v2 with the clock. v73: HAP_power_set_HMX.", "power"),
    ("hvx-mm-kernels-flat.h", 1223): ("HVX_OP_ADD/MUL_F32 of the flat f32 dots", "chain",
        "v73/v75: explicit qf32 op and convert. v79+: IEEE intrinsic. The compiler lowers it to the same pair.",
        "MUL_MAT f32-f32 only (not Qwen3.5)"),
    ("hvx-sqrt.h", 15): ("HVX_OP_MUL of sqrt (x * rsqrt(x))", "chain",
        "The compiler lowers the two sides to the same qf pair. The rsqrt chain above it is explicit qf32 on all versions.",
        "SQRT, L2 norm"),
    ("hvx-arith.h", 40): ("HVX_OP_ADD/SUB/MUL_F32 of the binary and fused chains", "chain",
        "The compiler lowers the two sides to the same qf pair. The f32 chains of the binary jobs and of SWIGLU "
        "(hvx_mul_mul_f32_aa) are equal on the four versions. The binary jobs differ only in their f16 and DIV paths.",
        "ADD, MUL, SUB f32, SWIGLU product"),
    ("hvx-mm-kernels-tiled.h", 270): ("hvx_vec_mul_f16_f16_to_f32_lower32 (scale product of the tiled dots)", "same",
        "v79+: Wsf_vmpy_VhfVhf. v73/v75: Wqf32 and two converts. The four libraries hold the same instructions.",
        "q8_0 matvec (tiled_vec_dot_q8_0_*)"),
    ("hvx-div.h", 17): ("HVX_OP_MUL_F32/F16 of the division", "chain", "The compiler lowers the two sides to the same qf pairs.",
        "DIV"),
    ("hvx-div.h", 27): ("hvx_div_mul_f16_const_using_f32", "same",
        "f16->f32 widen. The compiler lowers the two sides to the same pair.", "DIV f16 by scalar"),
    ("hvx-div.h", 40): ("f32->f16 narrow of the division", "chain",
        "v73/v75: hvx_vec_f32_to_f16. v79+: Vhf_vcvt_VsfVsf, which the compiler lowers to qf32 = vadd(qf32, +0), thus the "
        "sf rounding of the quotient is not there.",
        "DIV f16"),
    ("hvx-div.h", 69): ("f16 division by a scalar", "differs",
        "v73/v75: multiply by the f16 reciprocal 1/val rounded to f16 (qf32 product, one rounding to hf). v79+: multiply in "
        "f32 by the f32 reciprocal, then narrow.", "DIV f16 by scalar"),
    ("hvx-div.h", 78): ("f16 division by a scalar, tail", "differs", "The same as line 69, for the tail vector.",
        "DIV f16 by scalar"),
    ("hvx-div.h", 114): ("hvx_vec_div_f16_using_f32, widen", "same",
        "f16->f32 widen. The compiler lowers the two sides to the same pair.", "DIV f16"),
    ("hvx-div.h", 145): ("hvx_vec_div_f16_using_f32, narrow", "chain", "The same as line 40.", "DIV f16"),
    ("hvx-div.h", 156): ("hvx_vec_hybrid_div_f16", "differs",
        "v73/v75: f16 reciprocal from a 3rd-order integer polynomial (vlut4, vmpa/vmps :sat, vcl0, integer exponent) and "
        "an f16 multiply. v79+: f32 Newton reciprocal (3 steps, qf32), an f32 multiply and a narrow. 33 test cases fail on "
        "v73/v75 at 1.5e-7 to 3.2e-7.",
        "DIV f16"),
    ("dma-queue.h", 31): ("2D DMA descriptor layout", "none", "v73: 16-bit fields. v75+: 24-bit fields.", "all DMA"),
    ("dma-queue.h", 212): ("2D DMA descriptor fill", "none", "v73: type 0 and 16-bit counts. v75+: type 9 and split counts.",
        "all DMA"),
    ("dma-queue.h", 297): ("dma_queue_push", "none",
        "v73: a transfer with a field above 65535 becomes chained descriptors.", "all DMA"),
    ("cumsum-ops.c", 64): ("hvx_cumsum_vadd", "chain",
        "v73/v75: explicit qf32 add and convert. v76+: IEEE add. The compiler lowers it to the same pair.",
        "CUMSUM (not Qwen3.5)"),
    ("hex-profile.h", 19): ("hex_get_pmu", "none", "v79+: upmucnt registers. Older versions: qurt_pmu_get.", "profile"),
}

# The build flags that differ between the libraries (build.ninja of each ExternalProject).
BUILD_DIFFS: tuple[tuple[str, str, str], ...] = (
    ("-mllvm -enable-xqf-gen=true", "v79, v81 only",
     "The SDK file build/cmake/hexagon_arch.cmake sets it. The IEEE HVX intrinsics lower to qf ops and conversions. Then "
     "the QF optimization removes conversions inside a chain. v73/v75 have no such flag."),
    ("-mcpu=vNN -mhvx=vNN", "each library", "The schedule, the unroll and the set of opcodes (v81 adds qf32 = sf)."),
)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class Library:
    """The parsed data of one library."""

    version: str
    funcs: list[Function]
    classes: dict[str, Counter[str]]
    forms: dict[str, Counter[str]]
    sigs: dict[str, Counter[str]]
    edges: dict[str, set[str]]


def analyze_library(version: str, path: Path) -> Library:
    """Parse one disassembly file and count the classes, forms and chains of each function.

    Complexity is O(n) in the instructions, plus the chain pass of each function.
    """
    funcs = parse_disassembly(path)
    classes: dict[str, Counter[str]] = defaultdict(Counter)
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    sigs: dict[str, Counter[str]] = defaultdict(Counter)
    for f in funcs:
        name = base_name(f.name)
        for pkt in f.packets:
            for insn in pkt.insns:
                cl, form = classify(insn)
                classes[name][cl] += 1
                if cl in FLOAT_CLASSES:
                    forms[name][form] += 1
                    FORM_CLASS[form] = cl
        sigs[name] += chain_signatures(f)
    return Library(version, funcs, classes, forms, sigs, function_edges(funcs))


def resolve_roots(roots: Iterable[str], edges: dict[str, set[str]], known: set[str]) -> list[str]:
    """Give the functions that a root list names, in a stable order.

    A root with the suffix "!" gives only itself. A root that the library does
    not hold is skipped, because LTO can inline it.
    """
    out: set[str] = set()
    for r in roots:
        if r.endswith("!"):
            if r[:-1] in known:
                out.add(r[:-1])
        elif r in known:
            out |= closure([r], edges, INFRA)
    return sorted(out)


def loop_metrics(packets: list[Packet]) -> dict[str, float]:
    """Give the size and the density of a group of packets.

    The density is the number of instructions for each packet, without the
    constant extenders. The float count holds the float classes of FLOAT_CLASSES.
    """
    n_insn = n_hvx = n_float = n_nop = n_mem = 0
    for pkt in packets:
        for insn in real_insns(pkt):
            if insn == "nop":
                n_nop += 1
                continue
            n_insn += 1
            cl, _ = classify(insn)
            if is_hvx(insn):
                n_hvx += 1
            if cl in FLOAT_CLASSES:
                n_float += 1
            if "vmem" in insn:
                n_mem += 1
    n_pkt = max(len(packets), 1)
    return {
        "packets": len(packets), "insns": n_insn, "nops": n_nop, "hvx": n_hvx, "float": n_float,
        "vmem": n_mem, "insn_per_packet": round(n_insn / n_pkt, 2), "hvx_per_packet": round(n_hvx / n_pkt, 2),
    }


def hot_loops(f: Function, top: int = 3) -> list[tuple[str, int, int, dict[str, float]]]:
    """Give the loop bodies of a function with the most float and HVX instructions.

    The body of a loop is its own packets, without the packets of the loops
    inside it, thus an outer loop that holds the arithmetic and a small inner
    copy loop gives both bodies. Returns (kind, start, end, metrics) for each
    body. The kind is hw0, hw1 or sw, plus "/outer" for a body with an inner
    loop. A function without a loop gives one entry of kind "function".
    Complexity is O(l^2 + p) for l loops and p packets.
    """
    loops: dict[tuple[int, int], Loop] = {}
    for lp in find_loops(f):
        loops.setdefault((lp.start, lp.end), lp)
    rows: list[tuple[str, int, int, dict[str, float]]] = []
    for (start, end), lp in loops.items():
        inner = [(s2, e2) for (s2, e2) in loops if (s2, e2) != (start, end) and start <= s2 and e2 <= end]
        body = [pk for pk in lp.packets if not any(s2 <= pk.addr <= e2 for s2, e2 in inner)]
        if not body:
            continue
        kind = lp.kind + ("/outer" if inner else "")
        rows.append((kind, start, end, loop_metrics(body)))
    rows.sort(key=lambda r: (-(r[3]["float"] + r[3]["hvx"]), -r[3]["insns"]))
    if not rows and f.packets:
        rows = [("function", f.start, f.packets[-1].addr, loop_metrics(f.packets))]
    return rows[:top]


def scan_arch_branches(src_dir: Path) -> list[tuple[str, int, str]]:
    """Give each preprocessor or C branch on __HVX_ARCH__ or __HEXAGON_ARCH__ in the DSP sources.

    The lines #else and #endif are not branches, thus the scan skips them.
    """
    rows: list[tuple[str, int, str]] = []
    for path in sorted(src_dir.glob("*.[ch]")):
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
            if not re.search(r"__HVX_ARCH__|__HEXAGON_ARCH__", line):
                continue
            s = line.strip()
            if s.startswith("#endif") or s.startswith("#else") or s.startswith("//"):
                continue
            rows.append((path.name, i, s.rstrip("\\").strip()))
    return rows


def mix_verdict(counts: list[Counter[str]], chain_diff: bool) -> str:
    """Give the verdict for a function whose four libraries hold the same set of float forms."""
    if all(c == counts[0] for c in counts):
        return "same instructions" + (" (the chain difference comes from a merge in the analysis)" if chain_diff else "")
    if all(proportional(counts[0], c) for c in counts[1:]):
        return "same mix, other unroll" + (" (the chain difference comes from a merge in the analysis)" if chain_diff else "")
    if not chain_diff:
        return "same dataflow (equal chain sets), other unroll or inline count"
    return "mix and chains differ: conversion removal or unroll (check)"


def proportional(a: Counter[str], b: Counter[str]) -> bool:
    """Tell if two form counts differ only by a common factor, as an unroll does.

    Each count can miss the scaled value by one or by 15 %, because the tail
    code of a loop does not unroll.
    """
    if set(a) != set(b):
        return False
    sa, sb = sum(a.values()), sum(b.values())
    if sa == 0 or sb == 0:
        return sa == sb
    k = sb / sa
    return all(abs(b[f] - k * a[f]) <= max(1.0, 0.15 * b[f]) for f in a)


def diff_cause(present: dict[str, set[str]]) -> str:
    """Name the known cause of a set of forms that only some versions hold."""
    causes: list[str] = []
    only = {v: present[v] - set().union(*(present[w] for w in VERSIONS if w != v)) for v in VERSIONS}
    f16_addsub = re.compile(r"qf16 = v(add|sub)\(V\.hf,V\.hf\)")
    if any(f16_addsub.search(k) for k in present["v79"] - present["v75"]):
        causes.append("hvx-base.h:207 f16 add/sub (qf16 on v79+, qf32 widen on v73/v75)")
    if any(k.startswith("V.qf32 = V.sf") for k in only["v81"]):
        causes.append("hvx-base.h:125 v81 qf32 = sf convert")
    if any("vlut4" in k or "vmpa" in k for k in present["v75"]) and not any(
        "vlut4" in k or "vmpa" in k for k in present["v79"]
    ):
        causes.append("hvx-div.h:156 f16 reciprocal polynomial on v73/v75")
    if not causes:
        causes.append("compiler: conversion removal or unroll (check)")
    return " + ".join(causes)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> int:
    """Write one CSV file and give the number of data rows."""
    n = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
            n += 1
    return n


def op_functions(lib: Library, roots: Iterable[str]) -> list[str]:
    """Give the functions of one op variant in one library."""
    return resolve_roots(roots, lib.edges, set(lib.classes))


def write_function_tables(libs: dict[str, Library], names: list[str], out_dir: Path) -> None:
    """Write function_classes.csv and function_forms.csv."""
    write_csv(
        out_dir / "function_classes.csv", ["version", "function"] + list(ALL_CLASSES),
        ([v, n] + [libs[v].classes[n].get(c, 0) for c in ALL_CLASSES]
         for v in VERSIONS for n in names if n in libs[v].classes),
    )
    write_csv(
        out_dir / "function_forms.csv", ["function", "form", "class"] + list(VERSIONS),
        ([n, form, classify_form(form)] + [libs[v].forms.get(n, Counter()).get(form, 0) for v in VERSIONS]
         for n in names for form in sorted(set().union(*(set(libs[v].forms.get(n, {})) for v in VERSIONS)))),
    )


def write_op_tables(libs: dict[str, Library], out_dir: Path) -> set[str]:
    """Write op_classes.csv, op_functions.csv and form_ops.csv.

    Returns the set of functions that the Qwen3.5 variants reach.
    """
    op_rows: list[list[object]] = []
    opf_rows: list[list[object]] = []
    fo_rows: list[list[object]] = []
    q35_funcs: set[str] = set()
    entries = [(op, var, use, roots) for op, var, use, roots in OP_MAP]
    entries += [(op, "all", "-", roots) for op, roots in OTHER_OPS]
    for op, var, use, roots in entries:
        per_v: dict[str, Counter[str]] = {}
        for v in VERSIONS:
            lib = libs[v]
            funcs = op_functions(lib, roots)
            if use != "-":
                q35_funcs |= set(funcs)
            total: Counter[str] = Counter()
            forms: Counter[str] = Counter()
            for fn in funcs:
                total += lib.classes[fn]
                forms += lib.forms.get(fn, Counter())
            per_v[v] = forms
            op_rows.append([op, var, use, v, len(funcs)] + [total.get(c, 0) for c in ALL_CLASSES])
            if v == "v79":
                for fn in funcs:
                    fl = sum(lib.classes[fn].get(c, 0) for c in FLOAT_CLASSES)
                    opf_rows.append([op, var, fn, fl] + [lib.classes[fn].get(c, 0) for c in FLOAT_CLASSES])
        # The form is the join key with the per-instruction measurements of the simulator.
        if use != "-":
            for form in sorted(set().union(*(set(c) for c in per_v.values()))):
                fo_rows.append([form, classify_form(form), op, var] + [per_v[v].get(form, 0) for v in VERSIONS])
    write_csv(out_dir / "op_classes.csv", ["op", "variant", "qwen35_use", "version", "n_functions"] + list(ALL_CLASSES),
              op_rows)
    write_csv(out_dir / "op_functions.csv", ["op", "variant", "function", "float_total_v79"] + list(FLOAT_CLASSES),
              opf_rows)
    write_csv(out_dir / "form_ops.csv", ["form", "class", "op", "variant", *VERSIONS], fo_rows)
    return q35_funcs


def write_scalar_table(libs: dict[str, Library], names: list[str], q35_funcs: set[str], out_dir: Path) -> None:
    """Write scalar_float.csv: each scalar float instruction and each C library call."""
    rows: list[list[object]] = []
    for n in names:
        keys = sorted(set().union(*(
            {f for f in libs[v].forms.get(n, {}) if classify_form(f) in ("scalar_float", "libm_call")} for v in VERSIONS
        )))
        for k in keys:
            vals = [libs[v].forms.get(n, Counter()).get(k, 0) for v in VERSIONS]
            rows.append([n, k, classify_form(k), "yes" if n in q35_funcs else "no"] + vals
                        + ["yes" if len(set(vals)) == 1 else "no"])
    write_csv(out_dir / "scalar_float.csv", ["function", "form", "class", "qwen35", *VERSIONS, "same_all"], rows)


def write_version_diff(libs: dict[str, Library], names: list[str], q35_funcs: set[str], out_dir: Path) -> None:
    """Write version_diff.csv: the functions whose float forms or chains differ between versions."""
    rows: list[list[object]] = []
    for n in names:
        present = {v: set(libs[v].forms.get(n, {})) for v in VERSIONS}
        sigsets = {v: set(libs[v].sigs.get(n, {})) for v in VERSIONS}
        cls = {v: tuple(libs[v].classes.get(n, Counter()).get(c, 0) for c in FLOAT_CLASSES) for v in VERSIONS}
        form_diff = sorted(k for k in set().union(*present.values()) if len({k in present[v] for v in VERSIONS}) > 1)
        sig_diff = sorted(k for k in set().union(*sigsets.values()) if len({k in sigsets[v] for v in VERSIONS}) > 1)
        if not form_diff and not sig_diff and len(set(cls.values())) == 1:
            continue
        pairs = "".join("=" if sigsets[a] == sigsets[b] and present[a] == present[b] else "|"
                        for a, b in (("v73", "v75"), ("v75", "v79"), ("v79", "v81")))
        verdict = (diff_cause(present) if form_diff
                   else mix_verdict([libs[v].forms.get(n, Counter()) for v in VERSIONS], bool(sig_diff)))
        rows.append([
            n, "yes" if n in q35_funcs else "no", pairs, len(form_diff), len(sig_diff), verdict,
            " || ".join(f"{k} {[libs[v].forms.get(n, Counter()).get(k, 0) for v in VERSIONS]}" for k in form_diff),
            " || ".join(f"{k} {[libs[v].sigs.get(n, Counter()).get(k, 0) for v in VERSIONS]}" for k in sig_diff[:12]),
        ])
    write_csv(out_dir / "version_diff.csv",
              ["function", "qwen35", "same_73_75|75_79|79_81", "n_form_diff", "n_chain_diff", "cause",
               "forms_only_in_some_versions [v73,v75,v79,v81]", "chain_signatures_only_in_some_versions"], rows)


def write_loop_tables(libs: dict[str, Library], q35_funcs: set[str], out_dir: Path) -> None:
    """Write loops.csv (each Qwen3.5 function) and hot_loops.csv (each Qwen3.5 op variant)."""
    lp_rows: list[list[object]] = []
    for v in VERSIONS:
        for f in libs[v].funcs:
            n = base_name(f.name)
            if n in q35_funcs:
                for kind, start, end, m in hot_loops(f, top=6):
                    lp_rows.append([v, n, kind, hex(start), hex(end)] + [m[k] for k in LOOP_KEYS])
    write_csv(out_dir / "loops.csv", ["version", "function", "kind", "start", "end"] + list(LOOP_KEYS), lp_rows)

    hot_rows: list[list[object]] = []
    for op, var, _use, roots in OP_MAP:
        for v in VERSIONS:
            lib = libs[v]
            funcs = set(op_functions(lib, roots))
            cands = [(base_name(f.name), kind, start, end, m)
                     for f in lib.funcs if base_name(f.name) in funcs
                     for kind, start, end, m in hot_loops(f, top=2)]
            cands.sort(key=lambda c: (-c[4]["float"] - c[4]["hvx"], -c[4]["insns"]))
            for rank, (fn, kind, start, end, m) in enumerate(cands[:3], start=1):
                hot_rows.append([op, var, v, rank, fn, kind, hex(start), hex(end)] + [m[k] for k in LOOP_KEYS])
    write_csv(out_dir / "hot_loops.csv",
              ["op", "variant", "version", "rank", "function", "kind", "start", "end"] + list(LOOP_KEYS), hot_rows)


def write_arch_table(src_dir: Path, out_dir: Path) -> None:
    """Write arch_branches.csv: each arch branch of the DSP source and each build flag difference."""
    rows: list[list[object]] = []
    for fname, line, cond in scan_arch_branches(src_dir):
        what, effect, detail, users = ARCH_NOTES.get((fname, line), ("?", "?", "no note for this line", "?"))
        rows.append([fname, line, cond, what, effect, detail, users])
    for flag, where, detail in BUILD_DIFFS:
        rows.append(["build.ninja", 0, flag, where, "chain", detail, "every qf chain"])
    write_csv(out_dir / "arch_branches.csv", ["file", "line", "condition", "what", "effect", "detail", "users"], rows)


def run(dis_dir: Path, src_dir: Path, out_dir: Path) -> None:
    """Make all CSV files of the inventory in out_dir."""
    libs = {v: analyze_library(v, dis_dir / f"{v}.s") for v in VERSIONS}
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(set().union(*(set(lib.classes) for lib in libs.values())))
    write_function_tables(libs, names, out_dir)
    q35_funcs = write_op_tables(libs, out_dir)
    write_scalar_table(libs, names, q35_funcs, out_dir)
    write_version_diff(libs, names, q35_funcs, out_dir)
    write_loop_tables(libs, q35_funcs, out_dir)
    write_arch_table(src_dir, out_dir)
    write_csv(out_dir / "priority.csv", ["rank", "ops", "phase", "share"], ([*p] for p in PRIORITY))


def classify_form(form: str) -> str:
    """Give the class of a normalized form that analyze_library recorded."""
    return FORM_CLASS.get(form, "?")


def main(argv: list[str]) -> int:
    """Parse the arguments, make the disassembly if necessary, and write the CSV files."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dis-dir", type=Path, help="directory with v73.s, v75.s, v79.s, v81.s")
    ap.add_argument("--lib-dir", type=Path, help="directory with libggml-htp-vNN.so, used when --dis-dir is absent")
    ap.add_argument("--src-dir", type=Path, required=True, help="the htp/ source directory")
    ap.add_argument("--out", type=Path, required=True, help="output directory for the CSV files")
    args = ap.parse_args(argv)
    if args.dis_dir:
        missing = [v for v in VERSIONS if not (args.dis_dir / f"{v}.s").is_file()]
        if missing:
            print(f"error: {args.dis_dir} has no disassembly for {', '.join(missing)}. "
                  f"Give --lib-dir to make it.", file=sys.stderr)
            return 1
        run(args.dis_dir, args.src_dir, args.out)
        return 0
    if not args.lib_dir:
        print("error: give --dis-dir or --lib-dir", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="htp-inventory-") as tmp:
        disassemble(args.lib_dir, Path(tmp))
        run(Path(tmp), args.src_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
