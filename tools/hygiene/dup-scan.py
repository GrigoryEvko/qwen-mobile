#!/usr/bin/env python3
"""Report the identical line blocks that two or more files share.

    tools/hygiene/dup-scan.py tools/prof/*.py
    tools/hygiene/dup-scan.py quant/*.py analysis/*.py

The scan normalizes each line (it removes the leading and trailing spaces and
each empty line), then it compares each pair of files with difflib. A block of
MIN_LINES or more equal lines goes into the report.

The tool finds a copied block, and it does not find a semantic duplicate that
two writers spelled differently. Use ast-grep for that class.

Complexity: O(n^2) in the number of files, and O(m^2) in the lines of a pair.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

MIN_LINES = 4


def load(path: Path) -> tuple[list[str], list[int]]:
    """Give the normalized lines of a file and their line numbers."""
    lines: list[str] = []
    numbers: list[int] = []
    for number, text in enumerate(path.read_text(errors="replace").splitlines(), 1):
        stripped = text.strip()
        if not stripped:
            continue
        lines.append(stripped)
        numbers.append(number)
    return lines, numbers


def main(argv: list[str]) -> int:
    """Print each shared block of the files that the arguments name."""
    paths = [Path(a) for a in argv[1:]]
    data = {p: load(p) for p in paths}
    total = 0
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            la, na = data[a]
            lb, nb = data[b]
            matcher = difflib.SequenceMatcher(None, la, lb, autojunk=False)
            blocks = [m for m in matcher.get_matching_blocks() if m.size >= MIN_LINES]
            if not blocks:
                continue
            shared = sum(m.size for m in blocks)
            total += shared
            print(f"== {a.name} ({a.parent.name}) and {b.name} ({b.parent.name}): "
                  f"{shared} equal lines in {len(blocks)} blocks")
            for m in sorted(blocks, key=lambda m: -m.size):
                print(f"   {m.size:4d} lines: {a.parent.name}/{a.name}:{na[m.a]}-{na[m.a + m.size - 1]}"
                      f"  <->  {b.parent.name}/{b.name}:{nb[m.b]}-{nb[m.b + m.size - 1]}")
                print(f"        first line: {la[m.a][:90]}")
    print(f"total of the pairs: {total} equal lines")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
