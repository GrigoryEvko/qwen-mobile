"""The coverage-guided fuzzer (atheris, libFuzzer) of the block quantizers and the byte packers.

The input bytes give the type (Q8_0, Q4_0, IQ4_NL), the scale search
switch, the shape, and the float32 values, each value from 4 raw bytes,
thus NaN, the infinities, the subnormals and the extremes are all
possible. The oracle is the same as in test_qfz_grid.py:

- The indices and the scales are in range, and the scales are finite.
- gguf-py decodes the packed bytes to the values of the pipeline, bit for bit.
- A Q8_0 block keeps the format bound.
- A value that is not finite, or a block beyond the F16 scale range, gives
  ValueError, and no input in the range does (qfz_common.scale_domain).

    uv run python tests/fuzz/quant/atheris_qfz_pack.py -max_total_time=600 -timeout=20 \\
        -rss_limit_mb=4096 -artifact_prefix=build/fuzz/quant/atheris/ build/fuzz/quant/atheris/pack-corpus \\
        tests/fuzz/quant/seeds/pack

Give -artifact_prefix= in each run, else libFuzzer writes crash-* files into
the working directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import atheris  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

with atheris.instrument_imports(include=["quant.grid", "quant.grids", "gguf.quants"]):
    import gguf
    from quant.grid import dequantize, pack_nibbles, pack_q8_0, q8_0_dequantize, q8_0_quantize, quantize
    from quant.grids import IQ4NLGrid, Q4_0Grid

from qfz_common import block_view, q8_0_error_bound, scale_domain  # noqa: E402

KINDS = ("Q8_0", "Q4_0", "IQ4_NL")
torch.set_num_threads(1)


def test_one_input(data: bytes) -> None:
    """Quantize and pack one drawn matrix and check the oracle. Raise AssertionError on a defect."""
    fdp = atheris.FuzzedDataProvider(data)
    kind = KINDS[fdp.ConsumeIntInRange(0, 2)]
    search = fdp.ConsumeBool()
    rows, nblocks = fdp.ConsumeIntInRange(1, 4), fdp.ConsumeIntInRange(1, 4)
    raw = fdp.ConsumeBytes(rows * nblocks * 32 * 4)
    if len(raw) < rows * nblocks * 32 * 4:
        return
    w = np.frombuffer(raw, dtype=np.float32).reshape(rows, nblocks * 32).copy()
    domain = scale_domain(w, kind)
    wt = torch.from_numpy(w)
    grid = Q4_0Grid() if kind == "Q4_0" else IQ4NLGrid()
    try:
        q, d = q8_0_quantize(wt) if kind == "Q8_0" else quantize(grid, wt, search=search)
    except ValueError:
        assert domain != "in", f"the {kind} quantizer refuses a finite input in the F16 scale range"
        return
    assert domain != "out", f"the {kind} quantizer accepts a value that is not finite or a block beyond F16"
    if kind == "Q8_0":
        amax = np.abs(block_view(w.astype(np.float64))).max(-1)
        d32 = d.to(torch.float32).numpy()
        assert np.isfinite(d32).all(), "a Q8_0 scale is not finite"
        assert int(q.abs().max()) <= 127
        ours = q8_0_dequantize(q, d).numpy()
        decoded = gguf.quants.dequantize(pack_q8_0(q, d), gguf.GGMLQuantizationType.Q8_0)
        assert np.array_equal(decoded, ours), "gguf-py reads other Q8_0 values"
        err = np.abs(block_view(w.astype(np.float64)) - block_view(ours.astype(np.float64))).max(-1)
        assert (err <= q8_0_error_bound(amax, d32)).all(), "the Q8_0 error is over the bound"
        return
    idx = q
    assert int(idx.min()) >= 0 and int(idx.max()) <= 15, "an index is out of 0 .. 15"
    assert torch.isfinite(d).all(), "a 4-bit scale is not finite"
    decoded = gguf.quants.dequantize(pack_nibbles(idx, d), getattr(gguf.GGMLQuantizationType, kind))
    assert np.array_equal(decoded, dequantize(grid, idx, d).numpy()), f"gguf-py reads other {kind} values"


def main() -> None:
    """Start libFuzzer with the command-line options."""
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
