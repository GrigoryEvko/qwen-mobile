#!/usr/bin/env python3
"""Generate the HVX instruction census: the op list, the compiler probe and the C kernels.

The census measures each HVX instruction that touches floating point, or that converts between
integer and float, plus the integer instructions with rounding or saturation. The generator does
three steps:

1. ``parse``: it reads ``hvx_hexagon_protos.h`` of the SDK (the header that hexagon-clang uses),
   selects the intrinsics, and gets the minimum architecture of each. The SDK header gates only
   the v69 and more recent entries. The LLVM copy of the header gives the gate of the others.
2. ``probe``: it compiles one small translation unit for each intrinsic, each version (v73, v75,
   v79, v81) and each mode, and disassembles it. The modes are: the plain intrinsic, the intrinsic
   in a function with the ``hvx-ieee-fp`` target attribute, and the instruction as inline asm with
   that attribute. The result goes to ``probe.json``. The probe needs the SDK container.
3. ``generate``: it writes ``isa_ops.csv``, ``isa_kernels.h`` and ``isa_kernels.c``. An intrinsic
   that compiles to one instruction on each version becomes one op. An intrinsic that does not
   compile on each version without the attribute (the IEEE-form instructions) becomes two ops:
   ``ieee.<name>`` (the instruction as inline asm) and ``cc.<name>`` (what the compiler emits for
   the intrinsic). The hand-written sequences (``seq.*``) come last. They include the helpers of
   ``hvx-base.h`` and ``hvx-exp.h``, copied from the llama.cpp tree with the prefix ``isa_``, the
   integer-only routines of ``tools/htp-lab/lab/hvx-exact.h``, the chains of two or more IEEE steps,
   and the sf to int32 conversions. Each op also gets a cost bench (``isa_bench``): an inlined copy
   on four independent iterations for each trip of the loop.

Usage (in the SDK container, from the repository root):
    python3 tools/htp-lab/isa/gen_census.py probe
    python3 tools/htp-lab/isa/gen_census.py generate
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ISA_DIR = Path(__file__).resolve().parent
REPO_DIR = ISA_DIR.parents[2]
TOOLS = Path("/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools")
PROTOS_SDK = TOOLS / "target/hexagon/include/hvx_hexagon_protos.h"
PROTOS_LLVM = TOOLS / "lib/clang/19/include/hvx_hexagon_protos.h"
HTP_DIR = REPO_DIR / "third_party/llama.cpp/ggml/src/ggml-hexagon/htp"

ARCHES = (73, 75, 79, 81)
FLOAT_TYPES = {"sf", "hf", "qf32", "qf16", "bf", "f8", "x"}
QF_TYPES = {"qf32", "qf16"}
IEEE_TYPES = {"sf", "hf", "bf", "f8"}
INT_ROUND_RE = re.compile(r"_rnd|_sat|vround|vrmpy|vdmpy|vmpa|vlut16|vlut32|vnormamt|vcl0|vavg")
IMM_VALUES = {1: (0, 1), 3: (0, 1, 7)}

# The same flags as the lab build (CMakeLists.txt), without -g
CFLAGS = ["-mhvx-length=128B", "-mhmx", "-O2", "-G0", "-fvectorize", "-fno-zero-initialized-in-bss"]

# The corpus streams. The order is the stream id in isa_kernels.h. Keep it in step with target_isa.c.
STREAMS = [
    "none",
    "sf_a", "sf_b", "sf_c", "hf_a", "hf_b", "hf_c",
    "qf32_a", "qf32_b", "qf32_c", "qf16_a", "qf16_b", "qf16_c",
    "bf_a", "bf_b", "bf_c", "f8_a", "f8_b", "f8_c",
    "int_a", "int_b", "int_c", "seq16", "wconv",
]
# The element types. The order is the type id in isa_kernels.h.
ETYPES = ["", "sf", "hf", "qf32", "qf16", "bf", "f8", "x", "b", "ub", "h", "uh", "w", "uw", "pred"]
STREAM_VECTORS = 4096
INT_ITERATIONS = 1024


@dataclasses.dataclass
class Operand:
    """One parameter of an intrinsic.

    ``ctype`` is V (vector), W (vector pair), Q (predicate), R (32-bit scalar), P (64-bit scalar)
    or I (immediate). ``etype`` is the element type from the assembly syntax, or "" if it has none.
    """

    cname: str
    ctype: str
    etype: str
    acc: bool = False
    imm_bits: int = 0


@dataclasses.dataclass
class Intrinsic:
    """One entry of hvx_hexagon_protos.h, with the parsed operands and the census class."""

    name: str
    asm: str
    proto: str
    itype: str
    slots: str
    min_arch: int
    ret: str
    ret_etype: str
    args: list[Operand]
    cls: str = ""


@dataclasses.dataclass
class Op:
    """One census op: one entry in the op table of isa_kernels.c.

    ``slots`` gives for each of the three inputs of isa_run the stream name and the bytes per
    iteration. ``arg_slot`` gives for each non-immediate operand the input slot and the byte offset
    in the chunk of that slot.
    """

    name: str
    source: str
    asm: str
    cls: str
    kind: str
    min_arch: int
    cond: str
    out_etype: str
    out_ctype: str
    args: list[Operand]
    imm: dict[str, int]
    slots: list[tuple[str, int, str]]
    arg_slot: list[tuple[int, int]]
    n_vectors: int
    core: str
    probe: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------------------------

BLOCK_RE = re.compile(
    r"(?:#if __HVX_ARCH__ >= (\d+)\s*)?/\*\s*=+\s*\n\s*Assembly Syntax:\s*(.*?)\n"
    r"\s*C Intrinsic Prototype:\s*(.*?)\n\s*Instruction Type:\s*(.*?)\n\s*Execution Slots:\s*(.*?)\n"
)
CTYPE = {"HVX_Vector": "V", "HVX_VectorPair": "W", "HVX_VectorPred": "Q", "Word32": "R", "Word64": "P"}


def parse_header(path: Path) -> dict[str, tuple[int | None, str, str, str, str]]:
    """Read one protos header and return {name: (gate, asm, prototype, type, slots)}.

    The gate is the N of "#if __HVX_ARCH__ >= N" before the entry, or None if the entry has no gate.
    O(size of the header).
    """
    text = path.read_text()
    out: dict[str, tuple[int | None, str, str, str, str]] = {}
    for m in BLOCK_RE.finditer(text):
        proto = m.group(3).strip()
        name = re.search(r"(Q6_\w+)\(", proto).group(1)
        gate = int(m.group(1)) if m.group(1) else None
        out[name] = (gate, m.group(2).strip(), proto, m.group(4).strip(), m.group(5).strip())
    return out


def reg_etype(asm: str, reg: str) -> str | None:
    """Return the element type of the register ``reg`` (for example "Vu") in the assembly syntax.

    Returns "" if the register has no type suffix, and None if the register is not in the text.
    """
    m = re.search(r"(?<![A-Za-z])" + re.escape(reg) + r"\d+(?:\.(\w+))?", asm)
    if not m:
        return None
    return m.group(1) or ""


def parse_intrinsic(name: str, gate: int, asm: str, proto: str, itype: str, slots: str) -> Intrinsic | None:
    """Parse the prototype and the assembly syntax of one intrinsic into an Intrinsic.

    Returns None for an intrinsic with a pointer or void type (the stores, gathers and scatters).
    """
    m = re.match(r"(\w+)\s+(Q6_\w+)\((.*)\)", proto)
    if not m:
        return None
    ret_c, _, arglist = m.groups()
    if ret_c not in CTYPE or any(x.strip().split()[0] not in CTYPE for x in arglist.split(",") if x.strip()):
        return None
    dm = re.match(r"\s*([A-Za-z]+?)(\d+)(?:\.(\w+))?\s*([+|&^-]?=)", asm)
    dest_reg = dm.group(1) if dm else ""
    dest_etype = (dm.group(3) or "") if dm else ""
    if not dest_etype:
        head = name.split("_")[1]
        dest_etype = head[1:] if head[0] in "VW" else ""
    args: list[Operand] = []
    for a in [x.strip() for x in arglist.split(",") if x.strip()]:
        ctype_c, cname = a.split()
        ctype = CTYPE[ctype_c]
        if ctype == "R" and cname.startswith("I"):
            bits = int(re.search(r"\d+", cname).group(0))
            args.append(Operand(cname, "I", "", False, bits))
            continue
        et = reg_etype(asm, cname)
        args.append(Operand(cname, ctype, et or "", cname == dest_reg))
    return Intrinsic(name, asm, proto, itype, slots, gate, CTYPE[ret_c], dest_etype, args)


def classify(it: Intrinsic) -> str | None:
    """Return the census class of an intrinsic, or None if the census does not include it."""
    ins = {a.etype for a in it.args if a.ctype != "I"} - {""}
    types = ins | {it.ret_etype}
    if not types & FLOAT_TYPES:
        return "int_round" if INT_ROUND_RE.search(it.name) else None
    if "bf" in types:
        return "bf16"
    if "f8" in types:
        return "f8"
    if "x" in types:
        return "qf_ext"
    if it.ret == "Q":
        return "fp_cmp"
    out = it.ret_etype
    fin = ins & FLOAT_TYPES
    if out in QF_TYPES:
        if "equals" in it.name and fin <= IEEE_TYPES:
            return "qf_conv"
        return "qf_arith"
    if fin & QF_TYPES:
        return "qf_conv" if out in IEEE_TYPES else "int_float_conv"
    if out not in FLOAT_TYPES or not fin:
        return "int_float_conv"
    return "ieee_form"


def load_intrinsics(sdk: Path, llvm: Path) -> list[Intrinsic]:
    """Parse the two headers and return the census intrinsics in the order of the SDK header."""
    sdk_map = parse_header(sdk)
    llvm_map = parse_header(llvm) if llvm.exists() else {}
    out: list[Intrinsic] = []
    for name, (gate, asm, proto, itype, slots) in sdk_map.items():
        arch = gate or (llvm_map.get(name, (None,))[0]) or 60
        it = parse_intrinsic(name, arch, asm, proto, itype, slots)
        cls = classify(it) if it else None
        if cls:
            it.cls = cls
            out.append(it)
    return out


# ---------------------------------------------------------------------------------------------
# The C text of one core function
# ---------------------------------------------------------------------------------------------

CT_C = {"V": "HVX_Vector", "W": "HVX_VectorPair", "Q": "HVX_VectorPred", "R": "int32_t", "P": "int64_t"}


def asm_text(it: Intrinsic, imm: dict[str, int]) -> tuple[str, list[Operand]] | None:
    """Translate the assembly syntax into an inline asm template.

    Returns (template, the input operands in the order of %1, %2, ...), or None if the syntax has a
    predicate operand, which inline asm cannot bind.
    """
    if it.ret == "Q" or any(a.ctype == "Q" for a in it.args):
        return None
    text = it.asm
    dm = re.match(r"\s*([A-Za-z]+?)(\d+)", text)
    dest = dm.group(1)
    text = re.sub(r"(?<![A-Za-z])" + re.escape(dest) + r"\d+", "%0", text)
    inputs = [a for a in it.args if a.ctype != "I" and not a.acc]
    for idx, a in enumerate(inputs, start=1):
        text = re.sub(r"(?<![A-Za-z])" + re.escape(a.cname) + r"\d+", "%" + str(idx), text)
    for a in it.args:
        if a.ctype == "I":
            text = text.replace("#u" + str(a.imm_bits), "#" + str(imm[a.cname]))
    return text, inputs


# The v79 libgcc.a of Hexagon Tools 19.0.07 has __qf_convert_hf_to_uh_rne, which the v79 compiler
# calls for Q6_Vuh_vcvt_Vhf. It writes r20 to r23 and does not restore them, but the ABI makes
# r16 to r27 callee-saved. A core function that makes such a call thus saves r16 to r27 itself:
# the empty asm with the clobber list makes the compiler save them in the prologue.
CALL_GUARD = ('    __asm__ volatile("" ::: "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23", '
              '"r24", "r25", "r26", "r27");\n')


def core_function(it: Intrinsic, mode: str, imm: dict[str, int], fname: str, guard_archs: set[int] | None = None) -> str | None:
    """Return the C text of a function that runs the intrinsic once on its parameters.

    ``mode`` is "plain", "attr" (the hvx-ieee-fp target attribute) or "asm" (inline asm with the
    attribute). ``guard_archs`` lists the versions where the compiler emits a library call for the
    intrinsic: the function then saves the callee-saved registers itself (refer to CALL_GUARD).
    Returns None when the mode is not possible for the intrinsic.
    """
    params = [a for a in it.args if a.ctype != "I"]
    plist = ", ".join(f"{CT_C[a.ctype]} {a.cname}" for a in params) or "void"
    ret_c = CT_C[it.ret]
    attr = '__attribute__((noinline, target("hvx-ieee-fp")))' if mode in ("attr", "asm") else "__attribute__((noinline))"
    if mode in ("plain", "attr"):
        call_args = ", ".join(a.cname if a.ctype != "I" else str(imm[a.cname]) for a in it.args)
        body = f"    return {it.name}({call_args});"
        if guard_archs:
            cond = " || ".join(f"__HVX_ARCH__ == {a}" for a in sorted(guard_archs))
            body = f"#if {cond}\n{CALL_GUARD}#endif\n{body}"
    else:
        tr = asm_text(it, imm)
        if tr is None:
            return None
        tmpl, inputs = tr
        acc = [a for a in it.args if a.acc]
        cons = {"V": "v", "W": "v", "R": "r", "P": "r"}
        ins = ", ".join(f'"{cons[a.ctype]}"({a.cname})' for a in inputs)
        if acc:
            body = f'    __asm__("{tmpl}" : "+v"({acc[0].cname}) : {ins});\n    return {acc[0].cname};'
        else:
            body = f"    {ret_c} r;\n    __asm__(\"{tmpl}\" : \"=v\"(r) : {ins});\n    return r;"
    return f"{attr}\n{ret_c} {fname}({plist}) {{\n{body}\n}}\n"


def imm_variants(it: Intrinsic) -> list[dict[str, int]]:
    """Return the immediate values of each op of the intrinsic ({} for none)."""
    imms = [a for a in it.args if a.ctype == "I"]
    if not imms:
        return [{}]
    out: list[dict[str, int]] = [{}]
    for a in imms:
        out = [dict(d, **{a.cname: v}) for d in out for v in IMM_VALUES.get(a.imm_bits, (0,))]
    return out


def imm_suffix(imm: dict[str, int]) -> str:
    """Return the op name suffix of the immediate values, for example "_I1"."""
    return "".join(f"_I{v}" for v in imm.values())


# ---------------------------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------------------------

PROBE_HEAD = "#include <stdint.h>\n#include <hexagon_types.h>\n#include <hvx_hexagon_protos.h>\n"


def parse_objdump(text: str, fname: str) -> list[list[str]]:
    """Return the packets of the function ``fname`` from objdump text, without jumpr r31 and nop.

    O(lines of the text).
    """
    packets: list[list[str]] = []
    cur: list[str] = []
    inside = False
    for line in text.splitlines():
        if re.match(r"^[0-9a-f]+ <" + re.escape(fname) + r">:", line):
            inside = True
            continue
        if not inside:
            continue
        if re.match(r"^[0-9a-f]+ <", line) or not line.strip():
            if cur:
                packets.append(cur)
            break
        m = re.match(r"^\s*[0-9a-f]+:\s*(.*)$", line)
        if not m:
            continue
        s = m.group(1)
        start = "{" in s
        end = "}" in s
        s = s.replace("{", "").replace("}", "").strip()
        if start and cur:
            packets.append(cur)
            cur = []
        if s and s not in ("jumpr r31", "nop"):
            cur.append(" ".join(s.split()))
        if end:
            packets.append(cur)
            cur = []
    return [p for p in packets if p]


def probe_one(job: tuple[str, int, str, str]) -> tuple[str, int, str, dict]:
    """Compile and disassemble one probe. The job is (key, arch, mode, C text).

    Returns (key, arch, mode, {"ok": bool, "packets": [[insn, ...], ...], "error": str}).
    """
    key, arch, mode, src = job
    with tempfile.TemporaryDirectory() as tmp:
        c = Path(tmp) / "p.c"
        o = Path(tmp) / "p.o"
        c.write_text(PROBE_HEAD + src)
        cmd = [str(TOOLS / "bin/hexagon-clang"), f"-mv{arch}", f"-mhvx=v{arch}", *CFLAGS, "-c", str(c), "-o", str(o)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not o.exists():
            err = [l for l in r.stderr.splitlines() if "error" in l]
            return key, arch, mode, {"ok": False, "packets": [], "error": (err[0] if err else r.stderr[:200]).strip()}
        d = subprocess.run(
            [str(TOOLS / "bin/hexagon-llvm-objdump"), "-d", "--no-show-raw-insn", f"--mcpu=hexagonv{arch}",
             f"--mattr=+hvxv{arch},+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp", str(o)],
            capture_output=True, text=True)
        return key, arch, mode, {"ok": True, "packets": parse_objdump(d.stdout, "isa_probe"), "error": ""}


def run_probe(intrinsics: list[Intrinsic], jobs: int) -> dict:
    """Probe each intrinsic on each version in each mode. Returns {key: {arch: {mode: result}}}.

    The key is the intrinsic name plus the immediate suffix. O(intrinsics x versions x modes) compiles.
    """
    work: list[tuple[str, int, str, str]] = []
    for it in intrinsics:
        modes = ["plain"] if it.cls == "int_round" else ["plain", "attr", "asm"]
        for imm in imm_variants(it):
            key = it.name + imm_suffix(imm)
            for arch in ARCHES:
                if arch < it.min_arch:
                    continue
                for mode in modes:
                    src = core_function(it, mode, imm, "isa_probe")
                    if src is not None:
                        work.append((key, arch, mode, src))
    result: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        for key, arch, mode, res in ex.map(probe_one, work):
            result.setdefault(key, {}).setdefault(str(arch), {})[mode] = res
    return result


# ---------------------------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------------------------


def stream_for(etype: str, slot: int, conv_int_input: bool) -> str:
    """Return the corpus stream of an operand of type ``etype`` in input slot ``slot``."""
    letter = "abc"[slot]
    if conv_int_input and etype in ("b", "ub", "h", "uh"):
        return "seq16"
    if conv_int_input and etype in ("w", "uw"):
        return "wconv"
    base = {"sf": "sf", "hf": "hf", "qf32": "qf32", "qf16": "qf16", "bf": "bf", "f8": "f8"}.get(etype, "int")
    return f"{base}_{letter}"


def assign_slots(args: list[Operand], out_etype: str, cls: str) -> tuple[list[tuple[str, int, str]], list[tuple[int, int]]]:
    """Assign each non-immediate operand to one of the three input slots of isa_run.

    Vector and predicate operands that are not accumulators take slots 0, 1, 2 in order. The
    accumulator takes slot 2 if it is free, else the next free slot. Scalars take the next free
    slot, else they read the first word of the chunk of slot 0.
    Returns (slots, arg_slot): slots[k] = (stream, bytes per iteration, element type),
    arg_slot[j] = (slot, byte offset) for the j-th non-immediate operand.
    """
    params = [a for a in args if a.ctype != "I"]
    conv = cls == "int_float_conv" and out_etype in FLOAT_TYPES
    slots: list[tuple[str, int, str] | None] = [None, None, None]
    where: dict[int, tuple[int, int]] = {}

    def take(pref: int | None) -> int | None:
        """Return the preferred slot if it is free, else the first free slot, else None."""
        if pref is not None and slots[pref] is None:
            return pref
        for k in range(3):
            if slots[k] is None:
                return k
        return None

    order = [j for j, a in enumerate(params) if a.ctype in "VWQ" and not a.acc]
    order += [j for j, a in enumerate(params) if a.acc]
    order += [j for j, a in enumerate(params) if a.ctype in "RP"]
    for j in order:
        a = params[j]
        k = take(2 if a.acc else None)
        if k is None:
            if a.ctype not in "RP":
                raise ValueError(f"no free input slot for {a.cname}")
            where[j] = (0, 0)
            continue
        nbytes = 256 if a.ctype == "W" else 128
        slots[k] = (stream_for(a.etype, k, conv), nbytes, a.etype if a.ctype != "Q" else "pred")
        where[j] = (k, 0)
    final = [s if s is not None else ("none", 0, "") for s in slots]
    return final, [where[j] for j in range(len(params))]


def n_vectors_for(slots: list[tuple[str, int, str]], cls: str) -> int:
    """Return the iterations of the census run: the full streams for float ops, less for integers."""
    most = max((b for _, b, _ in slots), default=128) or 128
    full = STREAM_VECTORS * 128 // most
    return min(full, INT_ITERATIONS) if cls == "int_round" else full


# The instructions that move a predicate to or from a vector register at a call boundary
GLUE_RE = re.compile(r"^(r\d+ = #-0x1|q\d = vand\(v\d+,r\d+\)|v\d+ = vand\(q\d,r\d+\))$")


def real_insns(packets: list[list[str]]) -> list[str]:
    """Return the instructions of the packets without the predicate glue of the call ABI."""
    return [i for p in packets for i in p if not GLUE_RE.match(i)]


def arch_cond(archs: set[int], min_arch: int) -> str:
    """Return the preprocessor condition that is true on the versions in ``archs``.

    When the set is every probed version from its minimum on, the condition is a ">=" test, thus
    a version after v81 also gets the op.
    """
    if not archs:
        return "0"
    lo = min(archs)
    if archs == {a for a in ARCHES if a >= lo}:
        return f"__HVX_ARCH__ >= {max(lo, min_arch)}"
    return " || ".join(f"__HVX_ARCH__ == {a}" for a in sorted(archs))


def build_ops(intrinsics: list[Intrinsic], probe: dict) -> list[Op]:
    """Make the census ops from the intrinsics and the probe results, then add the sequences."""
    ops: list[Op] = []
    for it in intrinsics:
        for imm in imm_variants(it):
            key = it.name + imm_suffix(imm)
            pr = probe.get(key, {})
            want = {a for a in ARCHES if a >= it.min_arch}
            plain_ok = {a for a in want if pr.get(str(a), {}).get("plain", {}).get("ok")}
            attr_ok = {a for a in want if pr.get(str(a), {}).get("attr", {}).get("ok")}
            asm_ok = {a for a in want if pr.get(str(a), {}).get("asm", {}).get("ok")}
            one_insn = all(len(real_insns(pr[str(a)]["plain"]["packets"])) == 1 for a in plain_ok)
            slots, arg_slot = assign_slots(it.args, it.ret_etype, it.cls)
            common = dict(asm=it.asm, cls=it.cls, min_arch=it.min_arch, out_etype=it.ret_etype,
                          out_ctype=it.ret, args=it.args, imm=imm, slots=slots, arg_slot=arg_slot,
                          n_vectors=n_vectors_for(slots, it.cls), probe=pr)
            if plain_ok == want and one_insn:
                ops.append(Op(name=key, source=it.name, kind="insn", cond=arch_cond(plain_ok, it.min_arch),
                              core=core_function(it, "plain", imm, "@CORE@"), **common))
                continue
            if asm_ok:
                ops.append(Op(name="ieee." + key, source=it.name, kind="ieee",
                              cond=arch_cond(asm_ok, it.min_arch),
                              core=core_function(it, "asm", imm, "@CORE@"), **common))
            cc_mode = "plain" if plain_ok == want else "attr"
            cc_ok = plain_ok if cc_mode == "plain" else attr_ok
            calls = {a for a in cc_ok if any(i.startswith("call") for i in real_insns(pr[str(a)][cc_mode]["packets"]))}
            same_as_asm = cc_ok == asm_ok and all(
                real_insns(pr[str(a)][cc_mode]["packets"]) == real_insns(pr[str(a)]["asm"]["packets"]) for a in cc_ok)
            if same_as_asm:
                # The compiler emits the IEEE-form instruction itself on each version: the ieee op covers it
                continue
            if plain_ok == want:
                # The intrinsic compiles everywhere, but to more than one instruction somewhere
                ops.append(Op(name="cc." + key, source=it.name, kind="cc", cond=arch_cond(plain_ok, it.min_arch),
                              core=core_function(it, "plain", imm, "@CORE@", calls), **common))
            elif attr_ok:
                ops.append(Op(name="cc." + key, source=it.name, kind="cc",
                              cond=arch_cond(attr_ok, it.min_arch),
                              core=core_function(it, "attr", imm, "@CORE@", calls), **common))
            if not asm_ok and not attr_ok and not plain_ok:
                print(f"note: {key} compiles on no version, the census skips it", file=sys.stderr)
    ops += seq_ops()
    return ops


# ---------------------------------------------------------------------------------------------
# The sequences (hand-written)
# ---------------------------------------------------------------------------------------------

V = "V"
W = "W"


def seq(name: str, source: str, out: tuple[str, str], ins: list[tuple[str, str, bool]], body: str,
        min_arch: int = 73, cond: str = "", ieee: bool = False) -> Op:
    """Make one sequence op.

    ``out`` is (element type, V or W). ``ins`` lists (element type, V or W, accumulator) for each
    parameter a0, a1, a2 of the core function. ``body`` is the C body of the core function. With
    ``ieee`` set, the core function gets the hvx-ieee-fp target attribute.
    """
    args = [Operand(f"a{j}", ct, et, acc) for j, (et, ct, acc) in enumerate(ins)]
    slots, arg_slot = assign_slots(args, out[0], "seq")
    plist = ", ".join(f"{CT_C[a.ctype]} {a.cname}" for a in args)
    attr = '__attribute__((noinline, target("hvx-ieee-fp")))' if ieee else "__attribute__((noinline))"
    core = f"{attr}\n{CT_C[out[1]]} @CORE@({plist}) {{\n{body}\n}}\n"
    return Op(name=name, source=source, asm="", cls="seq", kind="seq", min_arch=min_arch,
              cond=cond or f"__HVX_ARCH__ >= {min_arch}", out_etype=out[0], out_ctype=out[1], args=args,
              imm={}, slots=slots, arg_slot=arg_slot, n_vectors=n_vectors_for(slots, "seq"), core=core)


EXP2_STAGES = [
    ("x_clamp", "hf"), ("x_minus_half", "hf"), ("k", "h"), ("f", "hf"), ("x_qf16", "qf16"),
    ("e5x", "qf16"), ("add_e4", "qf16"), ("mul1", "qf16"), ("add_e3", "qf16"), ("mul2", "qf16"),
    ("add_e2", "qf16"), ("mul3", "qf16"), ("add_e1", "qf16"), ("mul4", "qf16"), ("add_e0", "qf16"),
    ("mul5", "qf16"), ("add_one", "qf16"), ("y_hf", "hf"), ("result", "hf"),
]


def seq_ops() -> list[Op]:
    """Return the sequence ops: the helpers of hvx-base.h, the qf pairs and the exp2 stages."""
    sf, hf = "sf", "hf"
    ops = [
        seq("seq.hvx_vec_add_f32_f32", "hvx_vec_add_f32_f32", (sf, V), [(sf, V, False), (sf, V, False)],
            "    return isa_hvx_vec_add_f32_f32(a0, a1);"),
        seq("seq.hvx_vec_sub_f32_f32", "hvx_vec_sub_f32_f32", (sf, V), [(sf, V, False), (sf, V, False)],
            "    return isa_hvx_vec_sub_f32_f32(a0, a1);"),
        seq("seq.hvx_vec_mul_f32_f32", "hvx_vec_mul_f32_f32", (sf, V), [(sf, V, False), (sf, V, False)],
            "    return isa_hvx_vec_mul_f32_f32(a0, a1);"),
        seq("seq.hvx_vec_add_f16_f16", "hvx_vec_add_f16_f16", (hf, V), [(hf, V, False), (hf, V, False)],
            "    return isa_hvx_vec_add_f16_f16(a0, a1);"),
        seq("seq.hvx_vec_sub_f16_f16", "hvx_vec_sub_f16_f16", (hf, V), [(hf, V, False), (hf, V, False)],
            "    return isa_hvx_vec_sub_f16_f16(a0, a1);"),
        seq("seq.hvx_vec_mul_f16_f16", "hvx_vec_mul_f16_f16", (hf, V), [(hf, V, False), (hf, V, False)],
            "    return isa_hvx_vec_mul_f16_f16(a0, a1);"),
        seq("seq.hvx_vec_f32_to_f16", "hvx_vec_f32_to_f16", (hf, V), [(sf, V, False), (sf, V, False)],
            "    return isa_hvx_vec_f32_to_f16(a0, a1);"),
        seq("seq.hvx_vec_f32_to_f16_shuff", "hvx_vec_f32_to_f16_shuff", (hf, V), [(sf, V, False), (sf, V, False)],
            "    return isa_hvx_vec_f32_to_f16_shuff(a0, a1);"),
        seq("seq.hvx_vec_f16_to_f32", "hvx_vec_f16_to_f32", (sf, W), [(hf, V, False)],
            "    return isa_hvx_vec_f16_to_f32(a0);"),
        seq("seq.hvx_vec_f16_to_f32_shuff", "hvx_vec_f16_to_f32_shuff", (sf, W), [(hf, V, False)],
            "    return isa_hvx_vec_f16_to_f32_shuff(a0);"),
        seq("seq.hvx_vec_mpyacc_f32_f16", "hvx_vec_mpyacc_f32_f16", (sf, W),
            [(hf, V, False), (hf, V, False), (sf, W, True)],
            "    return isa_hvx_vec_mpyacc_f32_f16(a2, a0, a1);"),
        seq("seq.hvx_vec_i16_from_hf_rnd_sat", "hvx_vec_i16_from_hf_rnd_sat", ("h", V), [(hf, V, False)],
            "    return isa_hvx_vec_i16_from_hf_rnd_sat(a0);"),
        seq("seq.hvx_vec_exp2_f16", "hvx_vec_exp2_f16", (hf, V), [(hf, V, False)],
            "    return isa_hvx_vec_exp2_f16(a0);", min_arch=73),
        # The qf pairs, the same text on every version
        seq("seq.qfpair.mul_f32", "Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf)", (sf, V),
            [(sf, V, False), (sf, V, False)], "    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a0, a1));"),
        seq("seq.qfpair.add_f32", "Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf)", (sf, V),
            [(sf, V, False), (sf, V, False)], "    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a0, a1));"),
        seq("seq.qfpair.sub_f32", "Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf)", (sf, V),
            [(sf, V, False), (sf, V, False)], "    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a0, a1));"),
        seq("seq.qfpair.sf_roundtrip", "Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, 0))", (sf, V),
            [(sf, V, False)], "    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a0, Q6_V_vzero()));"),
        seq("seq.qfpair.mul_f16_wqf32", "Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf)", (hf, V),
            [(hf, V, False), (hf, V, False)], "    return Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a0, a1));"),
        seq("seq.qfpair.mul_f16_qf16", "Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf)", (hf, V),
            [(hf, V, False), (hf, V, False)], "    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(a0, a1));"),
        seq("seq.qfpair.add_f16_qf16", "Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf)", (hf, V),
            [(hf, V, False), (hf, V, False)], "    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a0, a1));"),
        seq("seq.qfpair.sub_f16_qf16", "Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf)", (hf, V),
            [(hf, V, False), (hf, V, False)], "    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(a0, a1));"),
        seq("seq.qfpair.hf_roundtrip_qf16", "Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(x, 0))", (hf, V),
            [(hf, V, False)], "    return Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(a0, Q6_V_vzero()));"),
        seq("seq.qfpair.f16_to_f32", "Q6_Vsf_equals_Vqf32 of Q6_Wqf32_vmpy_VhfVhf(x, 1.0)", (sf, W),
            [(hf, V, False)],
            "    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(a0, Q6_Vh_vsplat_R(0x3C00));\n"
            "    return Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p)), Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p)));"),
        seq("seq.qfpair.f32_to_f16_vadd0", "Q6_Vhf_equals_Wqf32 of Q6_Vqf32_vadd_VsfVsf(x, 0)", (hf, V),
            [(sf, V, False), (sf, V, False)],
            "    const HVX_Vector z = Q6_V_vzero();\n"
            "    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(a1, z), Q6_Vqf32_vadd_VsfVsf(a0, z)));"),
        seq("seq.qfpair.f32_to_f16_equals", "Q6_Vhf_equals_Wqf32 of Q6_Vqf32_equals_Vsf", (hf, V),
            [(sf, V, False), (sf, V, False)],
            "    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_equals_Vsf(a1), Q6_Vqf32_equals_Vsf(a0)));",
            min_arch=81),
        seq("seq.qfpair.f32_to_f16_mpy1", "Q6_Vhf_equals_Wqf32 of Q6_Vqf32_vmpy_VsfVsf(x, 1.0)", (hf, V),
            [(sf, V, False), (sf, V, False)],
            "    const HVX_Vector one = Q6_V_vsplat_R(0x3F800000);\n"
            "    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vmpy_VsfVsf(a1, one), Q6_Vqf32_vmpy_VsfVsf(a0, one)));"),
    ]
    # The qf extension of v79 and v81: a qf register holds more than the 32 or 16 bits that a store
    # writes. These ops put the qf result through memory (a volatile local) before the conversion,
    # and read the extension of a fresh result with vgetqfext (Rt = 0).
    ops += [
        seq("seq.qfext.mul_f32_via_mem", "Q6_Vqf32_vmpy_VsfVsf, store, load, Q6_Vsf_equals_Vqf32", (sf, V),
            [(sf, V, False), (sf, V, False)],
            "    volatile HVX_Vector t = Q6_Vqf32_vmpy_VsfVsf(a0, a1);\n    return Q6_Vsf_equals_Vqf32(t);"),
        seq("seq.qfext.add_f32_via_mem", "Q6_Vqf32_vadd_VsfVsf, store, load, Q6_Vsf_equals_Vqf32", (sf, V),
            [(sf, V, False), (sf, V, False)],
            "    volatile HVX_Vector t = Q6_Vqf32_vadd_VsfVsf(a0, a1);\n    return Q6_Vsf_equals_Vqf32(t);"),
        seq("seq.qfext.mul_f16_via_mem", "Q6_Vqf16_vmpy_VhfVhf, store, load, Q6_Vhf_equals_Vqf16", (hf, V),
            [(hf, V, False), (hf, V, False)],
            "    volatile HVX_Vector t = Q6_Vqf16_vmpy_VhfVhf(a0, a1);\n    return Q6_Vhf_equals_Vqf16(t);"),
        seq("seq.qfext.sub_f16_via_mem", "Q6_Vqf16_vsub_VhfVhf, store, load, Q6_Vhf_equals_Vqf16", (hf, V),
            [(hf, V, False), (hf, V, False)],
            "    volatile HVX_Vector t = Q6_Vqf16_vsub_VhfVhf(a0, a1);\n    return Q6_Vhf_equals_Vqf16(t);"),
        seq("seq.qfext.getext_mul_f32", "Q6_V_vgetqfext_VR(Q6_Vqf32_vmpy_VsfVsf, 0)", ("", V),
            [(sf, V, False), (sf, V, False)],
            "    return Q6_V_vgetqfext_VR(Q6_Vqf32_vmpy_VsfVsf(a0, a1), 0);", min_arch=79),
        seq("seq.qfext.getext_mul_f16", "Q6_V_vgetqfext_VR(Q6_Vqf16_vmpy_VhfVhf, 0)", ("", V),
            [(hf, V, False), (hf, V, False)],
            "    return Q6_V_vgetqfext_VR(Q6_Vqf16_vmpy_VhfVhf(a0, a1), 0);", min_arch=79),
    ]
    # Chains of two operations. The QFP optimizer of the compiler removes the conversion between
    # two qf operations, also an explicit one. The v79 and v81 compilers lower a chain of IEEE
    # intrinsics to one qf chain with one final rounding. The barrier op keeps the conversion.
    s3 = [(sf, V, False), (sf, V, False), (sf, V, False)]
    h3 = [(hf, V, False), (hf, V, False), (hf, V, False)]
    ops += [
        seq("seq.chain.mul_add_f32", "Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, b), c)", (sf, V), s3,
            "    return Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a0, a1), a2);", ieee=True),
        seq("seq.chain.add_add_f32", "Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(a, b), c)", (sf, V), s3,
            "    return Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(a0, a1), a2);", ieee=True),
        seq("seq.chain.mul_mul_f32", "Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a, b), c)", (sf, V), s3,
            "    return Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a0, a1), a2);", ieee=True),
        seq("seq.chain.mul_add_f16", "Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(a, b), c)", (hf, V), h3,
            "    return Q6_Vhf_vadd_VhfVhf(Q6_Vhf_vmpy_VhfVhf(a0, a1), a2);", ieee=True),
        seq("seq.chain.qf_mul_cvt_add_f32", "Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b)), c)",
            (sf, V), s3,
            "    const HVX_Vector p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a0, a1));\n"
            "    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p, a2));"),
        seq("seq.chain.qf_mul_cvt_add_f32_barrier", "the same with an asm barrier after the first conversion",
            (sf, V), s3,
            "    HVX_Vector p = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a0, a1));\n"
            "    __asm__ volatile(\"\" : \"+v\"(p));\n"
            "    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p, a2));"),
    ]
    # The chains of the CPU oracle: a*b+c is above; sum += x*y (four steps) and x*y + z*w. The
    # lanes of the later steps come from rotations of the inputs (vror by one lane per step).
    ops += [
        seq("seq.chain.dot4_f32", "s = c; 4 x s = Q6_Vsf_vadd_VsfVsf(s, Q6_Vsf_vmpy_VsfVsf(vror(a, k), vror(b, k)))",
            (sf, V), s3,
            "    HVX_Vector s = a2;\n"
            "    for (int k = 0; k < 4; k++) {\n"
            "        s = Q6_Vsf_vadd_VsfVsf(s, Q6_Vsf_vmpy_VsfVsf(Q6_V_vror_VR(a0, 4 * k), Q6_V_vror_VR(a1, 4 * k)));\n"
            "    }\n"
            "    return s;", ieee=True),
        seq("seq.chain.dot4_qf_acc", "the same sum in qf32 (vmpy qf32, vadd qf32), one conversion at the end", (sf, V), s3,
            "    HVX_Vector s = Q6_Vqf32_vadd_VsfVsf(a2, Q6_V_vzero());\n"
            "    for (int k = 0; k < 4; k++) {\n"
            "        s = Q6_Vqf32_vadd_Vqf32Vqf32(s, Q6_Vqf32_vmpy_VsfVsf(Q6_V_vror_VR(a0, 4 * k), Q6_V_vror_VR(a1, 4 * k)));\n"
            "    }\n"
            "    return Q6_Vsf_equals_Vqf32(s);"),
        seq("seq.chain.mul_add_mul_f32", "Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x, y), Q6_Vsf_vmpy_VsfVsf(z, vror(y, 1 lane)))",
            (sf, V), s3,
            "    return Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a0, a1), Q6_Vsf_vmpy_VsfVsf(a2, Q6_V_vror_VR(a1, 4)));",
            ieee=True),
        seq("seq.chain.dot4_hf_sf", "s = c (sf pair); 4 x s = Q6_Wsf_vmpyacc_WsfVhfVhf(s, vror(a, k), vror(b, k))", (sf, W),
            [(hf, V, False), (hf, V, False), (sf, W, True)],
            "    HVX_VectorPair s = a2;\n"
            "    for (int k = 0; k < 4; k++) {\n"
            "        s = Q6_Wsf_vmpyacc_WsfVhfVhf(s, Q6_V_vror_VR(a0, 2 * k), Q6_V_vror_VR(a1, 2 * k));\n"
            "    }\n"
            "    return s;", ieee=True),
    ]
    # sf -> int32 with round to nearest even (the CPU uses vcvtnq_s32_f32), and the Q8_0
    # quantization vcvtnq_s32_f32(x * id). The magic number 1.5 * 2^23 works for |x| < 2^22.
    magic = "    const HVX_Vector magic = Q6_V_vsplat_R(0x4b400000);\n"
    s2 = [(sf, V, False), (sf, V, False)]
    ops += [
        seq("seq.cvt.sf_to_w_rne_int", "isa_sf_to_w_rne (integer only)", ("w", V), [(sf, V, False)],
            "    return isa_sf_to_w_rne(a0);"),
        seq("seq.cvt.sf_to_w_magic_qf", "Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, 1.5 * 2^23)) - bits", ("w", V),
            [(sf, V, False)],
            magic + "    return Q6_Vw_vsub_VwVw(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a0, magic)), magic);"),
        seq("seq.cvt.sf_to_w_magic_ieee", "Q6_Vsf_vadd_VsfVsf(x, 1.5 * 2^23) - bits", ("w", V), [(sf, V, False)],
            magic + "    return Q6_Vw_vsub_VwVw(Q6_Vsf_vadd_VsfVsf(a0, magic), magic);", ieee=True),
        seq("seq.q8.mul_rne_int", "isa_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(x, id))", ("w", V), s2,
            "    return isa_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(a0, a1));", ieee=True),
        seq("seq.q8.mul_magic_ieee", "Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(x, id), 1.5 * 2^23) - bits", ("w", V), s2,
            magic + "    return Q6_Vw_vsub_VwVw(Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(a0, a1), magic), magic);", ieee=True),
        seq("seq.q8.qf_mul_magic", "Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(x, id), 1.5 * 2^23)) - bits",
            ("w", V), s2,
            magic + "    return Q6_Vw_vsub_VwVw(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(a0, a1), magic)), magic);"),
        seq("seq.q8.exact_mul_rne_int", "isa_sf_to_w_rne(isa_hvx_exact_sf_mul(x, id))", ("w", V), s2,
            "    return isa_sf_to_w_rne(isa_hvx_exact_sf_mul(a0, a1));"),
    ]
    # The integer-only IEEE routines of tools/htp-lab/lab/hvx-exact.h
    ops += [
        seq("seq.exact.sf_mul", "hvx_exact_sf_mul", (sf, V), s2, "    return isa_hvx_exact_sf_mul(a0, a1);"),
        seq("seq.exact.sf_add", "hvx_exact_sf_add", (sf, V), s2, "    return isa_hvx_exact_sf_add(a0, a1);"),
        seq("seq.exact.hf_to_sf", "hvx_exact_hf_to_sf of the even and odd halfwords (vzxt)", (sf, W), [(hf, V, False)],
            "    const HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(a0);\n"
            "    return Q6_W_vcombine_VV(isa_hvx_exact_hf_to_sf(Q6_V_hi_W(w)), isa_hvx_exact_hf_to_sf(Q6_V_lo_W(w)));"),
        seq("seq.exact.sf_to_hf", "hvx_exact_sf_to_hf of two vectors, packed with vpacke", (hf, V), s2,
            "    return Q6_Vh_vpacke_VwVw(isa_hvx_exact_sf_to_hf(a1), isa_hvx_exact_sf_to_hf(a0));"),
        seq("seq.bench.copy", "a0 (the load and store floor of the bench)", (sf, V), [(sf, V, False)], "    return a0;"),
    ]
    for idx, (sname, et) in enumerate(EXP2_STAGES, start=1):
        ops.append(seq(f"seq.exp2_f16.s{idx:02d}_{sname}", f"hvx_vec_exp2_f16 stage {idx}", (et, V),
                       [(hf, V, False)], f"    return isa_exp2_f16_stage(a0, {idx});", min_arch=73))
    return ops


# ---------------------------------------------------------------------------------------------
# Helper extraction from the llama.cpp tree
# ---------------------------------------------------------------------------------------------


def extract(path: Path, start: str, end: str, include_end: bool) -> str:
    """Return the lines of ``path`` from the line with ``start`` to the line with ``end``.

    Raises ValueError with the marker text if a marker is not in the file.
    """
    lines = path.read_text().splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if start in l)
        j = next(k for k in range(i + 1, len(lines)) if end in lines[k])
    except StopIteration:
        raise ValueError(f"{path}: the marker {start!r} or {end!r} is not in the file. Update gen_census.py.")
    return "\n".join(lines[i:j + (1 if include_end else 0)]) + "\n"


def helper_text(htp: Path) -> tuple[str, str]:
    """Return (C text, SHA-256 of the source text) of the copied helpers.

    The helpers come from hvx-base.h and hvx-exp.h of llama.cpp and from hvx-exact.h of the lab
    (the integer-only IEEE routines). The copy renames hvx_vec_ to isa_hvx_vec_ and hvx_exact_ to
    isa_hvx_exact_, thus it cannot collide with those headers in the same translation unit.
    """
    base = htp / "hvx-base.h"
    exp = htp / "hvx-exp.h"
    exact = REPO_DIR / "tools/htp-lab/lab/hvx-exact.h"
    parts = [
        extract(base, "static inline HVX_Vector hvx_vec_splat_f32", "static inline HVX_Vector hvx_vec_repl4", False),
        extract(base, "static inline HVX_Vector hvx_vec_f32_to_f16_shuff", "#endif // __HVX_ARCH__ < 79", True),
        extract(exp, "static inline HVX_Vector hvx_vec_exp2_f16", "#endif /* HVX_EXP_H */", False),
        extract(exact, "#define HVX_EXACT_SPLAT", "#endif /* HVX_EXACT_H */", False),
    ]
    src = "".join(parts)
    digest = hashlib.sha256(src.encode()).hexdigest()[:12]
    src = re.sub(r"\bhvx_vec_", "isa_hvx_vec_", src)
    src = re.sub(r"\bhvx_exact_", "isa_hvx_exact_", src)
    return src.replace("HVX_EXACT_SPLAT", "ISA_HVX_EXACT_SPLAT"), digest


EXP2_STAGE_FN = r"""
/* The body of hvx_vec_exp2_f16 (hvx-exp.h), stopped after the stage "stage" (1 to 19). Each stage
 * returns the value of one line of the upstream function, thus the census can find the first stage
 * that differs between two versions. Stage 19 is equal to the full function. */
static inline HVX_Vector isa_exp2_f16_stage(HVX_Vector x_v, int stage);

/* The int32 value of an f32 value with round to nearest even, with the rules of the ARM
 * vcvtnq_s32_f32: a value of 2^31 or more saturates, NaN gives 0. Integer instructions only. */
static inline HVX_Vector isa_sf_to_w_rne(HVX_Vector x) {
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector one  = Q6_V_vsplat_R(1);
    const HVX_Vector a    = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fffffff));
    const HVX_Vector e    = Q6_Vuw_vlsr_VuwR(a, 23);
    const HVX_Vector m    = Q6_V_vor_VV(Q6_V_vand_VV(a, Q6_V_vsplat_R(0x007fffff)), Q6_V_vsplat_R(0x00800000));
    /* e < 150: a shift right by s = min(150 - e, 31). The dropped bits, moved to the top of the
     * word, are more than half when they are above 0x80000000, and the odd bit decides a tie. */
    const HVX_Vector     s    = Q6_Vw_vmin_VwVw(Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(150), e), Q6_V_vsplat_R(31));
    HVX_Vector           q    = Q6_Vw_vlsr_VwVw(m, s);
    const HVX_Vector     rest = Q6_Vw_vasl_VwVw(m, Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(32), s));
    const HVX_VectorPred up   = Q6_Q_vcmp_gt_VuwVuw(Q6_V_vor_VV(rest, Q6_V_vand_VV(q, one)), Q6_V_vsplat_R(0x80000000));
    q = Q6_Vw_condacc_QVwVw(up, q, one);
    /* e >= 150: a shift left by e - 150, at most 7 below the saturation */
    const HVX_Vector l = Q6_Vw_vasl_VwVw(m, Q6_Vw_vsub_VwVw(e, Q6_V_vsplat_R(150)));
    HVX_Vector       r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(150), e), q, l);
    const HVX_VectorPred neg = Q6_Q_vcmp_gt_VwVw(zero, x);
    r = Q6_V_vmux_QVV(neg, Q6_Vw_vsub_VwVw(zero, r), r);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e, Q6_V_vsplat_R(157)),
                      Q6_V_vmux_QVV(neg, Q6_V_vsplat_R(0x80000000), Q6_V_vsplat_R(0x7fffffff)), r);
    return Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(a, Q6_V_vsplat_R(0x7f800000)), zero, r);
}

static inline HVX_Vector isa_exp2_f16_stage(HVX_Vector x_v, int stage) {
    const HVX_Vector zero_v    = Q6_V_vzero();
    const HVX_Vector half_hf_v = Q6_Vh_vsplat_R(0x3800);
    const HVX_Vector v_clamp_min = isa_hvx_vec_splat_f16(-24.0f);
    x_v = Q6_Vhf_vmax_VhfVhf(v_clamp_min, x_v);
    if (stage == 1) return x_v;
    HVX_Vector x_minus_half = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x_v, half_hf_v));
    if (stage == 2) return x_minus_half;
    HVX_Vector k_v = Q6_Vh_equals_Vhf(x_minus_half);
    if (stage == 3) return k_v;
    HVX_Vector f_v = Q6_Vhf_equals_Vh(k_v);
    if (stage == 4) return f_v;
    HVX_Vector x_qf16 = Q6_Vqf16_vsub_VhfVhf(x_v, f_v);
    if (stage == 5) return x_qf16;
    HVX_Vector y = Q6_Vqf16_vmpy_Vqf16Vhf(x_qf16, Q6_Vh_vsplat_R(0x090c));
    if (stage == 6) return y;
    y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x157d));
    if (stage == 7) return y;
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    if (stage == 8) return y;
    y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x20ed));
    if (stage == 9) return y;
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    if (stage == 10) return y;
    y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x2b1b));
    if (stage == 11) return y;
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    if (stage == 12) return y;
    y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x33b0));
    if (stage == 13) return y;
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    if (stage == 14) return y;
    y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x398c));
    if (stage == 15) return y;
    y = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    if (stage == 16) return y;
    y = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x3c00));
    if (stage == 17) return y;
    y = Q6_Vhf_equals_Vqf16(y);
    if (stage == 18) return y;
    HVX_Vector y_exp = Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(y, 1), 11);
    y_exp = Q6_Vh_vadd_VhVh(k_v, y_exp);
    HVX_VectorPred q_underflow = Q6_Q_vcmp_gt_VhVh(zero_v, y_exp);
    y = Q6_Vh_vaslacc_VhVhR(y, k_v, 10);
    return Q6_V_vmux_QVV(q_underflow, zero_v, y);
}
"""


# ---------------------------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------------------------


def c_str(s: str) -> str:
    """Return a C string literal of ``s``."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def io_lines(op: Op, fn: str, iexpr: str, tag: str, indent: str) -> list[str]:
    """Return the C lines that load the operands of the iteration ``iexpr``, call ``fn`` and store."""
    params = [a for a in op.args if a.ctype != "I"]
    lines, names = [], []
    for j, a in enumerate(params):
        slot, off = op.arg_slot[j]
        nbytes = op.slots[slot][1]
        addr = f"p[{slot}] + (size_t) ({iexpr}) * {nbytes} + {off}"
        v = f"x{tag}{j}"
        names.append(v)
        if a.ctype == "V":
            lines.append(f"{indent}const HVX_Vector {v} = *(const HVX_Vector *) ({addr});")
        elif a.ctype == "W":
            lines.append(f"{indent}const HVX_VectorPair {v} = *(const HVX_VectorPair *) ({addr});")
        elif a.ctype == "Q":
            lines.append(f"{indent}const HVX_VectorPred {v} = Q6_Q_vand_VR(*(const HVX_Vector *) ({addr}), 0x01010101);")
        elif a.ctype == "R":
            lines.append(f"{indent}const int32_t {v} = *(const int32_t *) ({addr});")
        elif a.ctype == "P":
            lines.append(f"{indent}const int64_t {v} = *(const int64_t *) ({addr});")
    call = f"{fn}({', '.join(names)})"
    oaddr = f"po + (size_t) ({iexpr}) * {out_bytes(op)}"
    if op.out_ctype == "V":
        lines.append(f"{indent}*(HVX_Vector *) ({oaddr}) = {call};")
    elif op.out_ctype == "W":
        lines.append(f"{indent}*(HVX_VectorPair *) ({oaddr}) = {call};")
    elif op.out_ctype == "Q":
        lines.append(f"{indent}*(HVX_Vector *) ({oaddr}) = Q6_V_vand_QR({call}, 0x01010101);")
    return lines


def loop_text(op: Op, idx: int) -> str:
    """Return the C text of the loop function of one op: load, call the core, store, n times."""
    lines = [f"static void isa_loop_{idx}(const uint8_t * p0, const uint8_t * p1, const uint8_t * p2, uint8_t * po, int n) {{",
             "    const uint8_t * const p[3] = { p0, p1, p2 };",
             "    for (int i = 0; i < n; i++) {"]
    lines += io_lines(op, f"isa_core_{idx}", "i", "", "        ")
    lines += ["    }", "}"]
    return "\n".join(lines) + "\n"


BENCH_UNROLL = 4


def bench_text(op: Op, idx: int) -> str:
    """Return the C text of the bench of one op: an inlined copy of the core, BENCH_UNROLL
    independent iterations for each trip of the loop, as a kernel would run it."""
    inl = op.core.replace("__attribute__((noinline))", "static inline __attribute__((always_inline))")
    inl = inl.replace("__attribute__((noinline, ", "static inline __attribute__((always_inline, ")
    inl = inl.replace("@CORE@", f"isa_inl_{idx}")
    attr = '__attribute__((target("hvx-ieee-fp"))) ' if 'target("hvx-ieee-fp")' in op.core else ""
    lines = [f"static void {attr}isa_bench_{idx}(const uint8_t * p0, const uint8_t * p1, const uint8_t * p2, uint8_t * po, int n) {{",
             "    const uint8_t * const p[3] = { p0, p1, p2 };",
             "    for (int i = 0; i < n; i++) {"]
    for u in range(BENCH_UNROLL):
        lines.append("        {")
        lines += io_lines(op, f"isa_inl_{idx}", f"{BENCH_UNROLL} * i + {u}", str(u), "            ")
        lines.append("        }")
    lines += ["    }", "}"]
    return inl + "\n".join(lines) + "\n"


def out_bytes(op: Op) -> int:
    """Return the output bytes per iteration of an op."""
    return 256 if op.out_ctype == "W" else 128


HEADER_TEMPLATE = r"""/* The HVX instruction census: the op table and the corpus file format.
 *
 * GENERATED by tools/htp-lab/isa/gen_census.py. Do not edit. Change the generator and run it again.
 *
 * The census runs each HVX instruction that touches floating point (sf, hf, qf32, qf16, bf, f8),
 * each conversion between integer and float, and the integer instructions with rounding or
 * saturation, on a fixed input corpus. compare.py compares each output with the CPU oracle (strict
 * IEEE, round to nearest even at each step) and with the outputs of the other Hexagon versions (or
 * of a real chip). isa_bench gives the cost of each op.
 *
 * isa_kernels.c has no dependency other than the Hexagon SDK headers. The kernel lab includes it
 * (lab/target_isa.c), and a DSP library can compile it without change.
 *
 * Use:
 *   1. Load the corpus streams (the files corpus_<stream>.bin, format below).
 *   2. For each op k: isa_run(k, stream[in_stream[0]], stream[in_stream[1]], stream[in_stream[2]],
 *      out, isa_ops[k].n_vectors). An iteration reads in_bytes[j] bytes from input j and writes
 *      out_bytes bytes. All pointers must have an alignment of 128 bytes. The caller must hold an
 *      HVX context in the 128-byte mode.
 *   3. Write the output as out_<name>.bin (format below), or compare isa_hash() of the output.
 */
#ifndef ISA_KERNELS_H
#define ISA_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define ISA_VEC_BYTES      128
#define ISA_STREAM_VECTORS @STREAM_VECTORS@ /* vectors in each corpus stream */
#define ISA_STREAM_BYTES   (ISA_STREAM_VECTORS * ISA_VEC_BYTES)
#define ISA_N_OPS          @N_OPS@

/* The corpus streams. target_isa.c makes them with integer code only, thus each version gets the
 * same bytes. Refer to target_isa.c for the contents of each stream. */
enum isa_stream {
@STREAM_ENUM@
    ISA_S_COUNT
};

/* The element types of the operands and of the result */
enum isa_type {
@TYPE_ENUM@
    ISA_T_COUNT
};

/* The kinds of op */
enum isa_kind {
    ISA_KIND_INSN = 0, /* the intrinsic, which compiles to one instruction on each version */
    ISA_KIND_IEEE = 1, /* the IEEE-form instruction as inline asm (target attribute hvx-ieee-fp) */
    ISA_KIND_CC   = 2, /* the intrinsic of an IEEE-form instruction: what the compiler emits */
    ISA_KIND_SEQ  = 3  /* a sequence of instructions (a helper of the llama.cpp kernels) */
};

typedef struct isa_op_desc {
    const char * name;       /* the op name and the output file stem */
    const char * source;     /* the intrinsic or the helper */
    const char * asm_syntax; /* the assembly syntax from the SDK header, "" for a sequence */
    const char * op_class;   /* the census class */
    uint8_t      kind;       /* enum isa_kind */
    uint8_t      min_arch;   /* the minimum __HVX_ARCH__ of the SDK header */
    uint8_t      out_type;   /* enum isa_type */
    uint8_t      in_type[3];
    uint8_t      in_stream[3]; /* enum isa_stream, ISA_S_NONE for an input that the op does not use */
    uint16_t     in_bytes[3];  /* bytes per iteration from each input: 0, 128 or 256 */
    uint16_t     out_bytes;    /* bytes per iteration to the output: 128 or 256 */
    uint16_t     n_vectors;    /* the iterations of the census run */
} isa_op_desc;

extern const isa_op_desc  isa_ops[ISA_N_OPS];
extern const char * const isa_stream_names[ISA_S_COUNT];
extern const char * const isa_type_names[ISA_T_COUNT];

/* Runs op op_id for n_vectors iterations. Returns 0, or -1 if the op is not available at the
 * __HVX_ARCH__ of this build, or -2 if op_id is not valid. */
int isa_run(int op_id, const void * in0, const void * in1, const void * in2, void * out, int n_vectors);

/* The cost bench: runs an inlined copy of op op_id on ISA_BENCH_UNROLL independent iterations for
 * each of n_iter trips, thus on ISA_BENCH_UNROLL * n_iter iterations of the inputs. The same
 * return codes as isa_run. The caller reads the cycle counter around the call. */
#define ISA_BENCH_UNROLL 4
int isa_bench(int op_id, const void * in0, const void * in1, const void * in2, void * out, int n_iter);

/* Returns 1 if the op is available in this build, else 0 */
int isa_available(int op_id);

/* The file format of a corpus stream and of an op output: one header of 128 bytes, then the raw
 * vectors (n_vectors * bytes_per_vector bytes). All fields are little endian. */
#define ISA_FILE_MAGIC_CORPUS "HVXC"
#define ISA_FILE_MAGIC_OUTPUT "HVXO"
#define ISA_FILE_VERSION      1

typedef struct isa_file_header {
    char     magic[4];         /* "HVXC" for a corpus stream, "HVXO" for an op output */
    uint32_t version;          /* ISA_FILE_VERSION */
    uint32_t id;               /* the stream id or the op id */
    uint32_t n_vectors;        /* the number of vectors (corpus) or iterations (output) */
    uint32_t bytes_per_vector; /* 128, or 256 for an output of vector pairs */
    uint32_t arch;             /* __HVX_ARCH__ of the program that wrote the file */
    uint32_t hash;             /* isa_hash() of the data */
    uint32_t source;           /* 0: hexagon-sim, 1: a real chip */
    char     name[64];         /* the stream name or the op name, zero padded */
    uint8_t  reserved[32];
} isa_file_header;

/* The hash of the census outputs: FNV-1a over 32-bit words in 32 independent lanes (lane k takes
 * the words k, k + 32, k + 64, ...), then FNV-1a over the 32 lane hashes. n must be a multiple of
 * 128. O(n). compare.py has the same function. */
static inline uint32_t isa_hash(const void * data, size_t n) {
    uint32_t h[32];
    for (int k = 0; k < 32; k++) {
        h[k] = 2166136261u;
    }
    const uint32_t * w = (const uint32_t *) data;
    for (size_t i = 0; i + 32 <= n / 4; i += 32) {
        for (int k = 0; k < 32; k++) {
            h[k] = (h[k] ^ w[i + k]) * 16777619u;
        }
    }
    uint32_t r = 2166136261u;
    for (int k = 0; k < 32; k++) {
        r = (r ^ h[k]) * 16777619u;
    }
    return r;
}

#endif
"""


C_TEMPLATE = r"""/* The HVX instruction census kernels. Refer to isa_kernels.h.
 *
 * GENERATED by tools/htp-lab/isa/gen_census.py. Do not edit. Change the generator and run it again.
 *
 * Each op has these functions:
 *   isa_core_<k>  runs the instruction (or the sequence) one time on its parameters. It is not
 *                 inlined, thus the disassembly of this file shows the code of each op by name.
 *   isa_loop_<k>  loads the operands of each iteration, calls the core function and stores the result.
 *   isa_inl_<k>   the same code as isa_core_<k>, always inlined.
 *   isa_bench_<k> the cost bench: isa_inl_<k> on ISA_BENCH_UNROLL independent iterations per trip.
 * A predicate operand comes from a vector: bit j is bit 0 of byte j. A predicate result goes out as
 * one byte for each bit (0 or 1). A scalar operand is the first word (or doubleword) of its chunk.
 * An op that the compiler cannot build for the __HVX_ARCH__ of this build has no functions, and
 * isa_run returns -1 for it.
 *
 * The IEEE-form ops (ISA_KIND_IEEE, ISA_KIND_CC) use the target attribute "hvx-ieee-fp". The v73
 * and v75 compilers need it for the IEEE-form intrinsics. The v79 and v81 compilers accept the
 * intrinsics without it, and lower them to other instructions.
 * WARNING: the IEEE-form opcodes give inf on the real v79 chip, and the simulator is not ground
 * truth for them. Do not use their census results without a run on silicon.
 */
#include "isa_kernels.h"

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

typedef void (*isa_loop_fn)(const uint8_t * p0, const uint8_t * p1, const uint8_t * p2, uint8_t * po, int n);

/* ---- The helpers of the llama.cpp kernels, copied from hvx-base.h and hvx-exp.h ----
 * Source: @HELPER_SOURCE@ (SHA-256 of the copied text @HELPER_HASH@).
 * The copy renames hvx_vec_ to isa_hvx_vec_. */
#if __HVX_ARCH__ >= 73
@HELPERS@
@EXP2_STAGE@
#endif

@OPS@

static const isa_loop_fn isa_loops[ISA_N_OPS] = {
@LOOP_TABLE@
};

static const isa_loop_fn isa_benches[ISA_N_OPS] = {
@BENCH_TABLE@
};

const isa_op_desc isa_ops[ISA_N_OPS] = {
@DESC_TABLE@
};

const char * const isa_stream_names[ISA_S_COUNT] = {
@STREAM_NAMES@
};

const char * const isa_type_names[ISA_T_COUNT] = {
@TYPE_NAMES@
};

int isa_available(int op_id) {
    return op_id >= 0 && op_id < ISA_N_OPS && isa_loops[op_id] != 0;
}

int isa_bench(int op_id, const void * in0, const void * in1, const void * in2, void * out, int n_iter) {
    if (op_id < 0 || op_id >= ISA_N_OPS) {
        return -2;
    }
    const isa_loop_fn fn = isa_benches[op_id];
    if (fn == 0) {
        return -1;
    }
    if (n_iter > 0) {
        fn((const uint8_t *) in0, (const uint8_t *) in1, (const uint8_t *) in2, (uint8_t *) out, n_iter);
    }
    return 0;
}

int isa_run(int op_id, const void * in0, const void * in1, const void * in2, void * out, int n_vectors) {
    if (op_id < 0 || op_id >= ISA_N_OPS) {
        return -2;
    }
    const isa_loop_fn fn = isa_loops[op_id];
    if (fn == 0) {
        return -1;
    }
    if (n_vectors > 0) {
        fn((const uint8_t *) in0, (const uint8_t *) in1, (const uint8_t *) in2, (uint8_t *) out, n_vectors);
    }
    return 0;
}
"""

KIND_C = {"insn": "ISA_KIND_INSN", "ieee": "ISA_KIND_IEEE", "cc": "ISA_KIND_CC", "seq": "ISA_KIND_SEQ"}


def type_id(et: str) -> str:
    """Return the C enum name of an element type."""
    return "ISA_T_" + (et.upper() if et else "NONE")


def stream_id(s: str) -> str:
    """Return the C enum name of a stream."""
    return "ISA_S_" + s.upper()


def emit(ops: list[Op], out_dir: Path, htp: Path) -> None:
    """Write isa_kernels.h, isa_kernels.c and isa_ops.csv."""
    helpers, digest = helper_text(htp)
    stream_enum = "\n".join(f"    {stream_id(s)} = {k}," for k, s in enumerate(STREAMS))
    type_enum = "\n".join(f"    {type_id(t)} = {k}," for k, t in enumerate(ETYPES))
    hdr = (HEADER_TEMPLATE.replace("@STREAM_VECTORS@", str(STREAM_VECTORS)).replace("@N_OPS@", str(len(ops)))
           .replace("@STREAM_ENUM@", stream_enum).replace("@TYPE_ENUM@", type_enum))
    (out_dir / "isa_kernels.h").write_text(hdr)

    blocks, loops, benches, descs = [], [], [], []
    for k, op in enumerate(ops):
        body = op.core.replace("@CORE@", f"isa_core_{k}")
        blocks.append(f"/* {k}: {op.name} ({op.kind}, {op.cls}) {op.asm} */\n#if {op.cond}\n"
                      f"#define ISA_HAVE_{k} 1\n{body}{loop_text(op, k)}{bench_text(op, k)}#endif\n")
        loops.append(f"#if defined(ISA_HAVE_{k})\n    isa_loop_{k},\n#else\n    0,\n#endif")
        benches.append(f"#if defined(ISA_HAVE_{k})\n    isa_bench_{k},\n#else\n    0,\n#endif")
        ins = [s for s in op.slots]
        in_type = ", ".join(type_id(t) for _, _, t in ins)
        in_stream = ", ".join(stream_id(s) for s, _, _ in ins)
        in_bytes = ", ".join(str(b) for _, b, _ in ins)
        descs.append(
            f"    {{ {c_str(op.name)}, {c_str(op.source)}, {c_str(op.asm)}, {c_str(op.cls)}, {KIND_C[op.kind]}, "
            f"{op.min_arch}, {type_id(op.out_etype if op.out_ctype != 'Q' else 'pred')}, {{ {in_type} }}, "
            f"{{ {in_stream} }}, {{ {in_bytes} }}, {out_bytes(op)}, {op.n_vectors} }},")
    src = (C_TEMPLATE.replace("@HELPERS@", helpers).replace("@HELPER_HASH@", digest)
           .replace("@HELPER_SOURCE@", "third_party/llama.cpp/ggml/src/ggml-hexagon/htp/hvx-base.h, hvx-exp.h")
           .replace("@EXP2_STAGE@", EXP2_STAGE_FN).replace("@OPS@", "\n".join(blocks))
           .replace("@LOOP_TABLE@", "\n".join(loops)).replace("@BENCH_TABLE@", "\n".join(benches))
           .replace("@DESC_TABLE@", "\n".join(descs))
           .replace("@STREAM_NAMES@", "\n".join(f"    {c_str(s)}," for s in STREAMS))
           .replace("@TYPE_NAMES@", "\n".join(f"    {c_str(t)}," for t in ETYPES)))
    (out_dir / "isa_kernels.c").write_text(src)

    with open(out_dir / "isa_ops.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "name", "kind", "class", "source", "asm", "min_arch", "cond", "out_type", "out_ctype",
                     "in_types", "in_streams", "in_bytes", "out_bytes", "n_vectors",
                     "insns_v73", "insns_v75", "insns_v79", "insns_v81"])
        for k, op in enumerate(ops):
            mode = {"insn": "plain", "cc": "attr", "ieee": "asm"}.get(op.kind, "")
            per_arch = []
            for a in ARCHES:
                r = op.probe.get(str(a), {}).get(mode, {}) if mode else {}
                per_arch.append(" | ".join(" ; ".join(p) for p in r.get("packets", [])) if r.get("ok") else
                                ("" if not r else "FAIL: " + r.get("error", "")))
            wr.writerow([k, op.name, op.kind, op.cls, op.source, op.asm, op.min_arch, op.cond, op.out_etype,
                         op.out_ctype, "/".join(t for _, _, t in op.slots), "/".join(s for s, _, _ in op.slots),
                         "/".join(str(b) for _, b, _ in op.slots), out_bytes(op), op.n_vectors, *per_arch])


def main() -> int:
    """Parse the command line and run the probe or the generation."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["list", "probe", "generate"])
    ap.add_argument("--protos", type=Path, default=PROTOS_SDK, help="the SDK hvx_hexagon_protos.h")
    ap.add_argument("--protos-llvm", type=Path, default=PROTOS_LLVM, help="the LLVM hvx_hexagon_protos.h")
    ap.add_argument("--htp", type=Path, default=HTP_DIR, help="the llama.cpp htp directory")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    intrinsics = load_intrinsics(args.protos, args.protos_llvm)
    if args.step == "list":
        for it in intrinsics:
            print(f"{it.min_arch:3d} {it.cls:15s} {it.name:40s} {it.asm}")
        print(f"{len(intrinsics)} intrinsics", file=sys.stderr)
        return 0
    if args.step == "probe":
        probe = run_probe(intrinsics, args.jobs)
        (ISA_DIR / "probe.json").write_text(json.dumps(probe, indent=1, sort_keys=True))
        print(f"probe: {len(probe)} intrinsics written to {ISA_DIR / 'probe.json'}", file=sys.stderr)
        return 0
    probe_path = ISA_DIR / "probe.json"
    if not probe_path.exists():
        print(f"error: {probe_path} does not exist. Run the probe step in the SDK container first.", file=sys.stderr)
        return 1
    probe = json.loads(probe_path.read_text())
    ops = build_ops(intrinsics, probe)
    emit(ops, ISA_DIR, args.htp)
    kinds: dict[str, int] = {}
    for op in ops:
        kinds[op.kind] = kinds.get(op.kind, 0) + 1
    print(f"generate: {len(ops)} ops {kinds}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
