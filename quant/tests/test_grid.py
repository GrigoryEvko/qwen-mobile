"""The block grids against the ggml formats: the tables, the byte layout, the reference scales, the chunking.

    .venv/bin/python -m pytest quant/tests
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from quant import grid as grid_module
from quant.grid import BLOCK, block_error, dequantize, dequantize_pack, pack_nibbles, pack_q8_0, q8_0_quantize, quantize
from quant.grids import IQ4_NL_TABLE, Grid, IQ4NLGrid, Q4_0Grid

ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / "llama.cpp"


def gguf_module():
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    import gguf

    return gguf


def test_iq4_nl_table_is_the_ggml_table() -> None:
    """The levels of IQ4_NL are the kvalues_iq4nl table of ggml-common.h."""
    text = (LLAMA / "ggml" / "src" / "ggml-common.h").read_text()
    m = re.search(r"GGML_TABLE_BEGIN\(int8_t, kvalues_iq4nl, 16\)\s*([-\d,\s]+)GGML_TABLE_END", text)
    assert m is not None, "the table is not in ggml-common.h"
    table = tuple(int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip())
    assert table == IQ4_NL_TABLE


@pytest.mark.parametrize("kind", ["Q4_0", "IQ4_NL"])
def test_pack_round_trip_against_gguf_py(kind: str) -> None:
    """The packed bytes dequantize with the gguf-py reference to the values of the grid."""
    gguf = gguf_module()
    torch.manual_seed(0)
    w = torch.randn(6, 96) * 0.05
    grid = Q4_0Grid() if kind == "Q4_0" else IQ4NLGrid()
    idx, d = quantize(grid, w, search=True)
    packed = pack_nibbles(idx, d)
    assert packed.shape == (6, 3 * 18) and packed.dtype == np.uint8
    reference = gguf.quants.dequantize(packed, getattr(gguf.GGMLQuantizationType, kind))
    ours = dequantize(grid, idx, d).numpy()
    np.testing.assert_allclose(reference, ours, rtol=0, atol=1e-7)
    # The round trip is a reconstruction: the relative error is that of a 4-bit grid.
    rel = np.linalg.norm(ours - w.numpy()) / np.linalg.norm(w.numpy())
    assert rel < 0.2


def test_iq4_nl_bytes_by_hand() -> None:
    """The byte layout by hand: F16 scale, element j in the low nibble of byte j, element j + 16 in the high nibble."""
    torch.manual_seed(1)
    w = torch.randn(2, 32)
    grid = IQ4NLGrid()
    idx, d = quantize(grid, w, search=True)
    packed = pack_nibbles(idx, d)
    for r in range(2):
        scale = packed[r, :2].view(np.float16)[0]
        assert scale == d[r, 0].numpy()
        nibbles = packed[r, 2:]
        levels = [IQ4_NL_TABLE[int(b & 15)] for b in nibbles] + [IQ4_NL_TABLE[int(b >> 4)] for b in nibbles]
        expected = np.float32(scale) * np.array(levels, dtype=np.float32)
        np.testing.assert_allclose(expected, dequantize(grid, idx, d)[r].numpy(), atol=1e-7)


def test_q8_0_round_trip_against_gguf_py() -> None:
    """The Q8_0 packing matches the gguf-py reference, and the quantization is idempotent."""
    gguf = gguf_module()
    torch.manual_seed(2)
    w = torch.randn(4, 64)
    q, d = q8_0_quantize(w)
    packed = pack_q8_0(q, d)
    assert packed.shape == (4, 2 * 34)
    reference = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
    ours = (q.float().reshape(4, 2, 32) * d.float()[..., None]).reshape(4, 64).numpy()
    np.testing.assert_allclose(reference, ours, atol=1e-7)
    q2, d2 = q8_0_quantize(torch.from_numpy(ours))
    assert torch.equal(q, q2) and torch.equal(d, d2)


def test_reference_scale_maps_the_signed_maximum_to_the_largest_level() -> None:
    """A block whose maximum is positive gets a negative scale, thus the maximum lands on −8 (Q4_0) or −127 (IQ4_NL)."""
    blocks = torch.zeros(2, 1, BLOCK)
    blocks[0, 0, 5] = 3.0
    blocks[1, 0, 7] = -3.0
    for grid, top in ((Q4_0Grid(), -8.0), (IQ4NLGrid(), -127.0)):
        d = grid.scale_rtn(blocks)
        assert d[0, 0] < 0 < d[1, 0]
        idx = grid.quantize_blocks(blocks, d)
        assert grid.value(idx)[0, 0, 5] == top and grid.value(idx)[1, 0, 7] == top
    zero = torch.zeros(1, 1, BLOCK)
    assert Q4_0Grid().scale_rtn(zero)[0, 0] == 1.0


def test_scale_search_keeps_a_block_that_is_on_the_grid() -> None:
    """The reference scale is a candidate, thus a dequantized block quantizes to itself."""
    torch.manual_seed(3)
    grid = Q4_0Grid()
    w = torch.randn(8, 64)
    idx, d = quantize(grid, w, search=True)
    w_q = dequantize(grid, idx, d)
    idx2, d2 = quantize(grid, w_q, search=True)
    assert torch.equal(d, d2) and torch.equal(idx, idx2)


def test_block_error_is_chunk_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    """The row-chunked error equals the error on the whole matrix, with and without the column scale."""
    torch.manual_seed(4)
    grid = IQ4NLGrid()
    w = torch.randn(13, 64) * 0.1
    weights = torch.rand(64) + 0.1
    t = torch.exp(torch.randn(64) * 0.3)
    whole = block_error(grid, w, weights, col_scale=t)
    direct_idx, direct_d = quantize(grid, w * t, weights=weights, search=True)
    direct = ((dequantize(grid, direct_idx, direct_d) - w * t).pow(2) * weights[None, :]).sum()
    torch.testing.assert_close(whole, direct, rtol=1e-6, atol=1e-6)
    monkeypatch.setattr(grid_module, "ROW_CHUNK", 4)
    chunked = block_error(grid, w, weights, col_scale=t)
    torch.testing.assert_close(chunked, direct, rtol=1e-6, atol=1e-6)


def test_dequantize_pack_adds_the_low_rank_term(tmp_path: Path) -> None:
    """A pack with factors dequantizes to d · level[idx] + b · a."""
    torch.manual_seed(5)
    grid = Q4_0Grid()
    w = torch.randn(4, 32)
    idx, d = quantize(grid, w, search=True)
    a, b = torch.randn(2, 32), torch.randn(4, 2)
    np.savez(tmp_path / "p.npz", q=idx.numpy(), d=d.numpy().view(np.uint16), levels=grid.levels.numpy(),
             kind=np.array("Q4_0"), lora_a=a.numpy(), lora_b=b.numpy())
    got = dequantize_pack(np.load(tmp_path / "p.npz"), torch.device("cpu"))
    torch.testing.assert_close(got, dequantize(grid, idx, d) + b @ a)


def test_generic_grid_round_matches_q4_0_round() -> None:
    """The midpoint search of the generic grid and the direct rounding of Q4_0 agree off the ties."""
    torch.manual_seed(6)
    x = torch.randn(1000) * 4
    x = x[(x - x.round()).abs() > 1e-3]
    generic = Grid(torch.arange(-8, 8, dtype=torch.float32)).round(x)
    assert torch.equal(generic, Q4_0Grid().round(x))
