"""Fuzz the index maps of the export: the value-head order of the GDN tensors, the block permutation, the names.

The properties:

- Each row and column map of LinearAttentionLayout is a permutation.
- The packed path and the float path give the same GGUF order: the
  dequantized permuted pack equals the permuted dequantized matrix, and
  the permuted low-rank factors give the permuted product.
- block_permutation gives the block order of a column permutation that
  moves whole blocks, and it refuses a permutation that splits a block.
- dequantize_pack of a saved pack with a row and a block permutation
  equals the permuted dequantized matrix.
- Each GGUF name that names.to_gguf gives has a type in any plan.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from hypothesis import assume, example, given
from hypothesis import strategies as st

from qfz_hyp import counted, fuzz_settings
from quant.export import LinearAttentionLayout
from quant.grid import block_permutation, dequantize, dequantize_pack
from quant.grids import IQ4NLGrid, Q4_0Grid
from quant.names import _LAYER_SUFFIX_TO_GGUF, to_gguf
from quant.plan import Plan

GDN_TAILS = ("attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight",
             "ffn_up.weight", "ssm_norm.weight")
# The value-head structure of the 4B, two value heads per key head, with smaller head dimensions.
LAYOUT_4B = LinearAttentionLayout(2, 4, 32, 64)


@st.composite
def layouts(draw: st.DrawFn) -> LinearAttentionLayout:
    """Draw a GDN layout with value heads a multiple of the key heads and head dimensions a multiple of 32."""
    num_k = draw(st.integers(1, 4))
    per_k = draw(st.integers(1, 3))
    head_k = 32 * draw(st.integers(1, 2))
    head_v = 32 * draw(st.integers(1, 2))
    return LinearAttentionLayout(num_k, num_k * per_k, head_k, head_v)


def _shape(layout: LinearAttentionLayout, tail: str, hidden: int) -> tuple[int, int]:
    """Give the checkpoint shape [rows, cols] of a GDN tensor of the layout."""
    value = layout.num_v_heads * layout.head_v_dim
    return {
        "attn_qkv.weight": (2 * layout.num_k_heads * layout.head_k_dim + value, hidden),
        "attn_gate.weight": (value, hidden),
        "ssm_alpha.weight": (layout.num_v_heads, hidden),
        "ssm_beta.weight": (layout.num_v_heads, hidden),
        "ssm_out.weight": (hidden, value),
        "ffn_up.weight": (2 * hidden, hidden),
        "ssm_norm.weight": (1, layout.head_v_dim),
    }[tail]


@fuzz_settings()
@given(layout=layouts(), tail=st.sampled_from(GDN_TAILS), layer=st.integers(0, 40))
@example(layout=LAYOUT_4B, tail="attn_qkv.weight", layer=0)
@example(layout=LAYOUT_4B, tail="ssm_out.weight", layer=31)
@counted
def test_layout_maps_are_permutations(layout: LinearAttentionLayout, tail: str, layer: int) -> None:
    """The row map and the column map of each GDN tensor hold each index one time."""
    name = f"blk.{layer}.{tail}"
    rows, cols = _shape(layout, tail, 64)
    for perm, n in ((layout.rows(name), rows), (layout.cols(name), cols)):
        if perm is None:
            continue
        assert perm.numel() == n, f"{name}: the map has {perm.numel()} indices for {n} positions"
        assert torch.equal(torch.sort(perm).values, torch.arange(n)), f"{name}: the map is not a permutation"
    if layout.num_k_heads == layout.num_v_heads:
        assert layout.rows(name) is None and layout.cols(name) is None, "one value head per key head keeps the order"


@fuzz_settings()
@given(layout=layouts(), tail=st.sampled_from(("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
                                              "ssm_alpha.weight")),
       kind=st.sampled_from(("Q4_0", "IQ4_NL")), seed=st.integers(0, 2**32 - 1), rank=st.integers(0, 3))
@example(layout=LAYOUT_4B, tail="ssm_out.weight", kind="Q4_0", seed=0, rank=2)
@example(layout=LAYOUT_4B, tail="attn_qkv.weight", kind="IQ4_NL", seed=1, rank=0)
@counted
def test_packed_and_float_paths_give_the_same_order(layout: LinearAttentionLayout, tail: str, kind: str, seed: int,
                                                    rank: int) -> None:
    """dequantize(pack(idx, d)) equals array(dequantize(idx, d)), and the factors follow the same maps."""
    name = f"blk.1.{tail}"
    rows, cols = _shape(layout, tail, 64)
    assume(cols % 32 == 0)
    gen = torch.Generator().manual_seed(seed)
    grid = Q4_0Grid() if kind == "Q4_0" else IQ4NLGrid()
    idx = torch.randint(0, 16, (rows, cols), generator=gen, dtype=torch.int8)
    d = (torch.rand(rows, cols // 32, generator=gen) + 0.1).to(torch.float16)
    idx2, d2 = layout.pack(name, idx, d)
    expected = layout.array(name, dequantize(grid, idx, d).numpy())
    np.testing.assert_array_equal(dequantize(grid, idx2, d2).numpy(), expected)
    if rank:
        a = torch.randn(rank, cols, generator=gen).numpy()
        b = torch.randn(rows, rank, generator=gen).numpy()
        a2, b2 = layout.factors(name, a, b)
        np.testing.assert_allclose(b2 @ a2, layout.array(name, b @ a), rtol=1e-5, atol=1e-5)


@fuzz_settings()
@given(nblocks=st.integers(1, 12), seed=st.integers(0, 2**32 - 1))
@example(nblocks=4, seed=0)
@counted
def test_block_permutation_follows_whole_blocks(nblocks: int, seed: int) -> None:
    """A permutation of whole blocks gives its block order back."""
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(nblocks, generator=gen)
    cols = (order[:, None] * 32 + torch.arange(32)[None, :]).reshape(-1)
    assert torch.equal(block_permutation(cols), order)


@fuzz_settings()
@given(nblocks=st.integers(1, 8), seed=st.integers(0, 2**32 - 1), shift=st.integers(1, 31))
@example(nblocks=2, seed=0, shift=16)
@counted
def test_block_permutation_refuses_a_split(nblocks: int, seed: int, shift: int) -> None:
    """A column permutation that moves a part of a block raises ValueError."""
    gen = torch.Generator().manual_seed(seed)
    cols = torch.arange(nblocks * 32)
    if nblocks == 1:
        cols = torch.roll(cols, shift)
    else:
        cols = torch.roll(cols[torch.randperm(nblocks * 32, generator=gen)], shift)
    runs = cols.view(-1, 32)
    assume(not (torch.equal(runs, runs[:, :1] + torch.arange(32)) and int((runs[:, 0] % 32).max()) == 0))
    with pytest.raises(ValueError, match="splits a block"):
        block_permutation(cols)


@fuzz_settings()
@given(rows=st.integers(1, 6), nblocks=st.integers(1, 5), seed=st.integers(0, 2**32 - 1), levels=st.booleans(),
       rank=st.integers(0, 2))
@example(rows=3, nblocks=2, seed=0, levels=False, rank=1)
@counted
def test_dequantize_pack_applies_the_row_and_block_maps(rows: int, nblocks: int, seed: int, levels: bool,
                                                        rank: int) -> None:
    """A saved pack, read with a row map and a block map, equals the permuted dequantized matrix."""
    gen = torch.Generator().manual_seed(seed)
    cols = nblocks * 32
    grid = Q4_0Grid()
    idx = torch.randint(0, 16, (rows, cols), generator=gen, dtype=torch.int8)
    d = (torch.rand(rows, nblocks, generator=gen) - 0.5).to(torch.float16)
    fields = {"d": d.numpy().view(np.uint16)}
    if levels:
        fields.update(q=idx.numpy(), levels=grid.levels.numpy())
    else:
        fields.update(q=(idx.to(torch.int16) - 8).to(torch.int8).numpy())
    full = dequantize(grid, idx, d)
    if rank:
        a, b = torch.randn(rank, cols, generator=gen), torch.randn(rows, rank, generator=gen)
        fields.update(lora_a=a.numpy(), lora_b=b.numpy())
        full = full + b @ a
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    buffer.seek(0)
    z = np.load(buffer)
    row_map = torch.randperm(rows, generator=gen)
    block_map = torch.randperm(nblocks, generator=gen)
    col_map = (block_map[:, None] * 32 + torch.arange(32)[None, :]).reshape(-1)
    got = dequantize_pack(z, torch.device("cpu"), row_map, col_map)
    torch.testing.assert_close(got, full[row_map][:, col_map], rtol=1e-6, atol=1e-6)


@fuzz_settings()
@given(layer=st.integers(0, 64), suffix=st.sampled_from(sorted(_LAYER_SUFFIX_TO_GGUF)),
       prefix=st.sampled_from(("model.language_model.", "model.")), n_layers=st.integers(0, 64),
       bulk=st.sampled_from(("Q4_0", "IQ4_NL", "CB4", "Q8_0")), edges=st.lists(st.integers(0, 64), max_size=3))
@example(layer=24, suffix="mlp.down_proj.weight", prefix="model.", n_layers=24, bulk="Q4_0", edges=[0, 23])
@counted
def test_every_gguf_name_has_a_type(layer: int, suffix: str, prefix: str, n_layers: int, bulk: str,
                                    edges: list[int]) -> None:
    """to_gguf gives a converter name, and Plan.type_of gives it a type with no exception."""
    name = to_gguf(f"{prefix}layers.{layer}.{suffix}")
    assert name == f"blk.{layer}.{_LAYER_SUFFIX_TO_GGUF[suffix]}"
    plan = Plan(bulk=bulk, n_layers=n_layers, edge_layers=tuple(edges))
    kind = plan.type_of(name)
    assert kind in {"F32", "keep", plan.bulk, plan.kv_proj, plan.gdn_gate, plan.edge_type, plan.mtp}
