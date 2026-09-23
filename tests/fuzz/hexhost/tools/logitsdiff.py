#!/usr/bin/env python3
"""Compare the raw logits dumps of hexhost_logits_hash (HEXHOST_LOGITS_DUMP) with a reference dump.

Usage: logitsdiff.py N_VOCAB REF.bin RUN.bin [RUN.bin ...]

For each run and each row: the maximum absolute difference of the logits, the KL divergence
KL(ref || run) of the softmax, and whether the argmax is the same. The tool prints the first row
that differs, the maximum and the mean KL over the rows, and the count of rows with another argmax.
O(rows * n_vocab).
"""

import sys

import numpy as np


def rows(path: str, n_vocab: int) -> np.ndarray:
    """The logits of one dump as a float64 array [rows, n_vocab]."""
    a = np.fromfile(path, dtype=np.float32)
    if a.size % n_vocab:
        raise SystemExit(f"{path}: {a.size} floats is not a multiple of {n_vocab}")
    return a.reshape(-1, n_vocab).astype(np.float64)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """The log-softmax of each row."""
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> int:
    """Print one block for each run."""
    n_vocab = int(sys.argv[1])
    ref = rows(sys.argv[2], n_vocab)
    lp_ref = log_softmax(ref)
    p_ref = np.exp(lp_ref)
    for path in sys.argv[3:]:
        x = rows(path, n_vocab)
        if x.shape != ref.shape:
            print(f"{path}: {x.shape[0]} rows, the reference has {ref.shape[0]}")
            continue
        lp = log_softmax(x)
        kl = (p_ref * (lp_ref - lp)).sum(axis=1)
        mad = np.abs(x - ref).max(axis=1)
        same_top = (x.argmax(axis=1) == ref.argmax(axis=1))
        diff_rows = np.nonzero(mad > 0)[0]
        first = int(diff_rows[0]) if diff_rows.size else -1
        print(f"{path}: rows {x.shape[0]}, first row that differs {first}, max abs {mad.max():.3g}, "
              f"KL max {kl.max():.3g} mean {kl.mean():.3g}, rows with another argmax {int((~same_top).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
