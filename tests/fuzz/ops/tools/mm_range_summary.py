"""Summarize the TSV files of mm_range per weight class, context and role.

Each row of a TSV file holds the statistics of one (context, weight, op, role) key. This script
groups the weights by class (the name without the block number), and for each class it prints the
largest |value|, the block of that value, and the counts above 32768 and 65504.

Usage: mm_range_summary.py MTP_BLOCK TSV [TSV...]
MTP_BLOCK is the block number of the MTP block of the model, for example 24 for the 2B.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path


def weight_class(name: str, mtp_block: int) -> tuple[str, int]:
    """Give the class of a weight name and its block number (-1 for no block).

    Args:
        name: The tensor name, for example "blk.3.ffn_down.weight"
        mtp_block: The block number of the MTP block

    Returns:
        The class name ("mtp:" in front for the MTP block) and the block number
    """
    m = re.match(r"blk\.(\d+)\.(.+?)(\.weight)?$", name)
    if not m:
        return name.removesuffix(".weight"), -1
    blk = int(m.group(1))
    cls = m.group(2)
    return ("mtp:" + cls if blk == mtp_block else cls), blk


def main(paths: list[str], mtp_block: int) -> None:
    """Print one table for each TSV file.

    Args:
        paths: The TSV files
        mtp_block: The block number of the MTP block
    """
    for path in paths:
        rows = Path(path).read_text().splitlines()[1:]
        agg: dict[tuple[str, str, str], dict] = defaultdict(
            lambda: {"max": 0.0, "blk": -1, "n32": 0, "n65": 0, "nf": 0, "n": 0, "rows": 0}
        )
        for row in rows:
            ctx, w, op, role, mx, n, n32, n65, nf, calls, mrows = row.split("\t")
            cls, blk = weight_class(w, mtp_block)
            a = agg[(ctx, cls, role)]
            if float(mx) > a["max"]:
                a["max"], a["blk"] = float(mx), blk
            a["n32"] += int(n32)
            a["n65"] += int(n65)
            a["nf"] += int(nf)
            a["n"] += int(n)
            a["rows"] = max(a["rows"], int(mrows))
        print(f"== {path}")
        print(f"{'context':10} {'class':24} {'role':4} {'max|v|':>10} {'block':>5} {'>32768':>8} {'>65504':>8} {'nonfin':>6} {'values':>12} {'rows':>5}")
        for (ctx, cls, role), a in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][2], -kv[1]["max"])):
            print(f"{ctx:10} {cls:24} {role:4} {a['max']:10.1f} {a['blk']:5d} {a['n32']:8d} {a['n65']:8d} {a['nf']:6d} {a['n']:12d} {a['rows']:5d}")


if __name__ == "__main__":
    main(sys.argv[2:], int(sys.argv[1]))
