#!/usr/bin/env python3
"""Compare the census ops of the DSP library with the same ops compiled as the lab compiles them.

The lab compiles tools/htp-lab/isa/isa_kernels.c with LAB_C_FLAGS into a static program for the
simulator. The DSP library of the silicon probe compiles the same file with the same flags plus
-fpic, and links it into a shared object. The DSP build also compiles the file one more time
without -fpic (lab_ref.o, the code of the simulator run). This script compares the instructions of
each function isa_core_<id> in the two disassemblies. Thus a result of the chip is known to come
from the instructions of the simulator result.

The comparison ignores what the link changes: the address of a branch target (the symbol and the
offset stay), and the target of a call (an object file has no relocation in this output).

Usage:
    check_disasm.py <skel.disasm.txt> <lab_ref.disasm.txt>

Both files come from "hexagon-llvm-objdump -d --no-show-raw-insn" with the same --mcpu and --mattr.
Returns 0 when each isa_core function agrees, 1 when one differs or is missing from the library.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FUNC_RE = re.compile(r"^[0-9a-f]+ <(\w+)>:")
INSN_RE = re.compile(r"^\s*[0-9a-f]+:\s*(.*)$")
TARGET_RE = re.compile(r"0x[0-9a-f]+ <([^>]+)>")
CALL_RE = re.compile(r"^call <[^>]+>$")


def functions(text: str, prefix: str) -> dict[str, list[str]]:
    """Return {name: instructions} of the functions whose names start with prefix. O(lines).

    Each instruction is one line of the disassembly without the address, the packet braces and
    the extra spaces. An address operand "0x1f0 <f+0x10>" becomes "<f+0x10>", and a call becomes
    "call".
    """
    out: dict[str, list[str]] = {}
    cur: list[str] | None = None
    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            cur = out.setdefault(m.group(1), []) if m.group(1).startswith(prefix) else None
            continue
        if cur is None:
            continue
        im = INSN_RE.match(line)
        if not im:
            if not line.strip():
                cur = None
            continue
        s = " ".join(im.group(1).replace("{", " ").replace("}", " ").split())
        s = TARGET_RE.sub(r"<\1>", s)
        if CALL_RE.match(s):
            s = "call"
        if s:
            cur.append(s)
    return out


def check(skel: Path, ref: Path) -> int:
    """Compare each isa_core function of ref with the function of the same name in skel.

    Returns the exit code: 0 if all agree, 1 otherwise.
    """
    lib = functions(skel.read_text(), "isa_core_")
    lab = functions(ref.read_text(), "isa_core_")
    if not lab:
        print(f"check_disasm: error: {ref} has no isa_core_ function")
        return 1
    problems = []
    for name in sorted(lab, key=lambda n: int(n.rsplit("_", 1)[1])):
        got = lib.get(name)
        if got is None:
            problems.append(f"{name}: missing from the library")
        elif got != lab[name]:
            first = next(i for i, (a, b) in enumerate(zip(got + [""], lab[name] + [""])) if a != b)
            problems.append(f"{name}: instruction {first}: library [{(got + [''])[first]}] "
                            f"lab [{(lab[name] + [''])[first]}]")
    for p in problems:
        print(f"check_disasm: DIFFERENT {p}")
    print(f"check_disasm: {len(lab) - len(problems)} of {len(lab)} census functions agree with the lab compile")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    """Parse the command line and run the comparison. Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("skel", type=Path, help="the disassembly of libisaprobe_skel.so")
    parser.add_argument("ref", type=Path, help="the disassembly of lab_ref.o")
    args = parser.parse_args(argv)
    return check(args.skel, args.ref)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
