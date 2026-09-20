"""Build the per-decode-token DDR byte ledger from a Hexagon profile log.

Where bytes.py models the traffic from the hyperparameters, this tool measures it. Run it on
the log of a run that generated tokens, and it prints what one decode token really moved.

Usage:
    tools/prof/ledger.py tools/prof/store/<stamp>-stalls-p1.log

TWO TRAPS, both of which cost a wrong answer once:
  - A tensor that appears twice in one operation is ONE descriptor, not two. add_tensor
    (ggml-hexagon.cpp) looks a tensor up by its data pointer and reuses the entry when the
    shape matches, thus the recurrent state, which the fused state operation names as an
    input twice and as the output once, is read once and written once. Counting the three
    names gives 77.7 GB/s for that operation, which is above the roofline and thus impossible.
  - A declared destination is not always moved. SET_ROWS names the whole key cache as its
    destination and writes one row, thus its declared bytes are 500 times its real traffic.
    Read any per-operation rate against the 51 to 55 GB/s roofline: a rate far above it means
    the byte count is wrong, not that the operation is fast.

Each ``profile-op`` line carries the dimensions, the types and the byte strides of every
source and of the destination, thus the bytes one op touches are exact and not modelled.
A pass ends at the output head, which runs once per forward pass, so the passes are the
segments between head operations. A decode pass is one whose head destination has one row.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"profile-op ([A-Z_0-9+]+)\|(.*?)\|([0-9: x>-]+)\|([\w: x>-]+)\|([0-9: x>-]+)\|"
    r"(.*?)\|usec (\d+) cycles (\d+)")


def split_arrow(s: str) -> tuple[list[str], str]:
    """Split a ``a x b -> c`` field into its sources and its destination."""
    srcs, _, dst = s.partition(" -> ")
    return [p.strip() for p in srcs.split(" x ")], dst.strip()


def dims_of(field: str) -> list[int]:
    """The integers of a ``ne0:ne1`` group."""
    return [int(x) for x in field.split(":")]


def tensor_bytes(dim: str, stride: str) -> int:
    """Bytes of one tensor: the row count times the row stride.

    Args:
        dim: The ``ne0:ne1`` field
        stride: The ``nb0:nb1`` field

    Returns:
        The byte count, or 0 when the fields do not parse
    """
    try:
        d, s = dims_of(dim), dims_of(stride)
    except ValueError:
        return 0
    if len(d) < 2 or len(s) < 2:
        return 0
    return d[1] * s[1]


def main(path: str) -> int:
    """Segment the log into passes and print the decode ledger."""
    passes: list[list[tuple]] = []
    cur: list[tuple] = []
    for raw in open(path, errors="replace"):
        m = LINE.search(raw)
        if not m:
            continue
        op, names, dims, types, strides, kern, usec, cyc = m.groups()
        sname, dname = split_arrow(names)
        sdim, ddim = split_arrow(dims)
        sstr, dstr = split_arrow(strides)
        rec = (op, sname, dname, sdim, ddim, sstr, dstr, kern, int(usec))
        cur.append(rec)
        # the head closes a forward pass
        if op.startswith("MUL_MAT") and ddim.startswith("248320:"):
            passes.append(cur)
            cur = []
    if cur:
        passes.append(cur)

    dec = [p for p in passes if p and p[-1][4] == "248320:1"]
    pre = [p for p in passes if p and p[-1][4] != "248320:1"]
    print(f"passes {len(passes)}: decode {len(dec)}, other {len(pre)}")
    if not dec:
        return 1

    # take the median-length decode pass, so a warm-up or a split pass does not skew it
    dec.sort(key=len)
    p = dec[len(dec) // 2]
    print(f"decode pass: {len(p)} ops, {sum(r[8] for r in p) / 1000:.2f} ms")

    by_op: dict[str, list] = defaultdict(lambda: [0, 0.0, 0])   # calls, us, bytes
    weight_b = act_b = 0
    seen_w: dict[str, int] = {}
    rows = []
    for op, sname, dname, sdim, ddim, sstr, dstr, kern, usec in p:
        b = 0
        for n, d, s in zip(sname, sdim, sstr):
            tb = tensor_bytes(d, s)
            b += tb
            if ".weight" in n or "token_embd" in n:
                weight_b += tb
                seen_w[n] = seen_w.get(n, 0) + 1
            else:
                act_b += tb
        db = tensor_bytes(ddim, dstr)
        b += db
        act_b += db
        e = by_op[op]
        e[0] += 1
        e[1] += usec / 1000.0
        e[2] += b
        rows.append((op, sname, dname, b, usec))

    print(f"\n{'op':<18} {'calls':>6} {'ms':>8} {'MB':>10} {'GB/s':>7}")
    for op, (c, ms, b) in sorted(by_op.items(), key=lambda kv: -kv[1][2]):
        print(f"{op:<18} {c:6d} {ms:8.2f} {b/1e6:10.1f} {b/ms/1e6 if ms else 0:7.1f}")
    tot_b = sum(v[2] for v in by_op.values())
    tot_ms = sum(v[1] for v in by_op.values())
    print(f"{'TOTAL':<18} {sum(v[0] for v in by_op.values()):6d} {tot_ms:8.2f} {tot_b/1e6:10.1f} {tot_b/tot_ms/1e6:7.1f}")
    print(f"\nweights {weight_b/1e6:.1f} MB   activations+state {act_b/1e6:.1f} MB")

    print("\n=== weights read more than once in one decode pass ===")
    multi = {k: v for k, v in seen_w.items() if v > 1}
    print(f"{len(multi)} of {len(seen_w)} distinct weight tensors" if multi else "none")
    for k, v in sorted(multi.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v}x  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
