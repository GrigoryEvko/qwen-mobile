#!/usr/bin/env python3
"""The command line of a host check of lab run directories.

A host check reads the files that one lab target wrote into its run directory, compares them with an
exact reference, and, for more than one directory, compares the output files byte for byte. Thus it
shows whether two Hexagon versions give the same bits. tools/htp-lab/exact/check.py (target exact)
and check_q8.py (target q8oracle) use this module, and tests/suite/lab-checks.sh runs the two.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

# A check of one run directory: it prints its result and gives the count of lanes or blocks that differ
# from the reference, and the names of the output files that the cross-version compare reads.
CheckDir = Callable[[Path], tuple[int, list[str]]]


def cross_version(dirs: Sequence[Path], names: Sequence[str]) -> int:
    """Compare each output file byte for byte across the run directories.

    Complexity O(bytes of the files).

    Args:
        dirs: The run directories, one for each Hexagon version
        names: The names of the output files

    Returns:
        The number of files that are not the same in every directory
    """
    print("cross-version:")
    bad = 0
    for f in names:
        blobs = [(d / f).read_bytes() for d in dirs]
        same = all(b == blobs[0] for b in blobs[1:])
        print(f"  {f}: {'identical' if same else 'DIFFERENT'} in {len(dirs)} directories")
        bad += 0 if same else 1
    return bad


def main_for(doc: str, check_dir: CheckDir, argv: Sequence[str] | None = None) -> int:
    """Check each run directory of the command line, then compare the output files across them.

    The names of the output files come from the check of the last directory.

    Args:
        doc: The docstring of the check, which the script prints when the command line has no directory
        check_dir: The check of one run directory
        argv: The arguments, or None for sys.argv[1:]

    Returns:
        The exit code: 0 when every directory agrees with the reference and with the others, 1 when not,
        and 2 when the command line has no directory
    """
    dirs = [Path(p) for p in (sys.argv[1:] if argv is None else argv)]
    if not dirs:
        print(doc)
        return 2
    bad = 0
    names: list[str] = []
    for d in dirs:
        print(f"{d}:")
        n, names = check_dir(d)
        bad += n
    if len(dirs) > 1:
        bad += cross_version(dirs, names)
    return 0 if bad == 0 else 1
