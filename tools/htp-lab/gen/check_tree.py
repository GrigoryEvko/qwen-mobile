#!/usr/bin/env python3
"""Run each generator of tools/htp-lab/gen and compare its output with its header in the kernel tree.

A generator writes a header of the HTP kernels. A landed patch can change that header by hand. If the
generator does not hold the change, a run of the generator over the header deletes the change. Thus
the output of each generator must be the bytes of its header in the tree, and this check fails when
it is not. It also fails for a script of gen/ that is not in GENERATORS, thus a new generator cannot
stay out of the check.

Usage:
    tools/htp-lab/gen/check_tree.py [--htp <the htp directory of a llama.cpp tree>]

The preset directory is the htp directory of the submodule third_party/llama.cpp. tools/htp-lab/run.sh
all runs this check. The exit code is 0 when each generator gives the bytes of its header.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GEN_DIR = Path(__file__).resolve().parent
REPO_DIR = GEN_DIR.parents[2]
DEFAULT_HTP = REPO_DIR / "third_party/llama.cpp/ggml/src/ggml-hexagon/htp"

# Each generator and the header of the htp directory that it writes
GENERATORS = {
    "act_i16.py": "hvx-act-i16.h",
    "silu_i16.py": "hvx-silu-i16.h",
    "softplus_i16.py": "hvx-softplus.h",
}

# The scripts of gen/ that write no header: the shared module of the generators and this check
NOT_GENERATORS = {"i16_fit.py", "check_tree.py"}


def first_difference(got: str, want: str) -> str:
    """The line number and the two texts of the first line that differs. Complexity O(lines)."""
    a, b = got.splitlines(), want.splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i + 1}: the generator gives {x!r}, the tree holds {y!r}"
    return f"the generator gives {len(a)} lines, the tree holds {len(b)} lines"


def check(htp: Path) -> int:
    """Run each generator and compare its output with its header.

    Args:
        htp: The htp directory of the kernel tree

    Returns:
        The number of generators that do not give the bytes of their header, plus the number of
        scripts of gen/ that are not in GENERATORS
    """
    bad = 0
    for script in sorted(p.name for p in GEN_DIR.glob("*.py")):
        if script in NOT_GENERATORS:
            continue
        if script not in GENERATORS:
            print(f"lab: generator {script} FAIL: it is not in GENERATORS of check_tree.py, thus no check covers it")
            bad += 1
            continue
        header = htp / GENERATORS[script]
        run = subprocess.run([sys.executable, str(GEN_DIR / script)], capture_output=True, text=True)
        if run.returncode != 0:
            print(f"lab: generator {script} FAIL: it ended with the exit code {run.returncode}: {run.stderr.strip()[-300:]}")
            bad += 1
            continue
        want = header.read_text() if header.exists() else None
        if want is None:
            print(f"lab: generator {script} FAIL: the tree holds no {header}")
            bad += 1
        elif run.stdout != want:
            print(f"lab: generator {script} FAIL: its output is not {GENERATORS[script]} of the tree, "
                  f"{first_difference(run.stdout, want)}")
            bad += 1
        else:
            print(f"lab: generator {script} PASS: its output is the bytes of {GENERATORS[script]}")
    return bad


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the check.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit code: 0 when each generator gives the bytes of its header
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--htp", type=Path, default=DEFAULT_HTP, help="the htp directory of the kernel tree")
    a = ap.parse_args(argv)
    return 0 if check(a.htp) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
