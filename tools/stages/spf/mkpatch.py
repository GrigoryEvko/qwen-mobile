#!/usr/bin/env python3
"""Make one llama.cpp patch file from two trees: the header text, then the git diff of each file.

Usage:
    mkpatch.py BASE SRC HEADER OUT FILE [FILE ...]

BASE and SRC are two llama.cpp trees (for example build/spf/base, the tree of HEAD, and build/spf/src, the
same tree with the edits). HEADER is a text file with the subject line and the body of the patch. Each FILE
is a path relative to the root of the trees. The script runs "git diff --no-index" on the two copies of each
FILE and writes the paths of the diff relative to the root (a/FILE and b/FILE), thus "git apply" and
"patch -p1" in a llama.cpp tree take the patch. A FILE that is only in SRC gives a new file.

The script stops with an error when a FILE has no change, or when git fails. O(size of the files).
"""

import subprocess
import sys
from pathlib import Path


def file_diff(base: Path, src: Path, rel: str) -> str:
    """The git diff of one file between the two trees, with the paths a/REL and b/REL."""
    old = base / rel
    new = src / rel
    if not new.exists():
        sys.exit(f"mkpatch.py: {new} does not exist")
    left = str(old) if old.exists() else "/dev/null"
    proc = subprocess.run(["git", "diff", "--no-index", "--no-color", "--full-index", "--", left, str(new)],
                          capture_output=True, text=True)
    # git diff --no-index gives 1 when the files differ and 0 when they are equal
    if proc.returncode == 0:
        sys.exit(f"mkpatch.py: {rel} has no change")
    if proc.returncode != 1:
        sys.exit(f"mkpatch.py: git diff failed for {rel}: {proc.stderr.strip()}")
    lines = []
    for line in proc.stdout.splitlines(keepends=True):
        if line.startswith("diff --git "):
            line = f"diff --git a/{rel} b/{rel}\n"
        elif line.startswith("--- ") and not line.startswith("--- /dev/null"):
            line = f"--- a/{rel}\n"
        elif line.startswith("+++ "):
            line = f"+++ b/{rel}\n"
        lines.append(line)
    return "".join(lines)


def main() -> int:
    """Write the patch file."""
    if len(sys.argv) < 6:
        print(__doc__, file=sys.stderr)
        return 2
    base, src, header, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    text = header.read_text().rstrip("\n") + "\n\n"
    text += "".join(file_diff(base, src, rel) for rel in sys.argv[5:])
    out.write_text(text)
    print(f"{out}: {len(text.splitlines())} lines, {len(sys.argv) - 5} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
