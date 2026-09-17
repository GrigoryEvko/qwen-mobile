"""The refold recovers the folded weights of a calibration from the source, the folds and the packs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from quant.grid import dequantize, q8_0_dequantize, q8_0_quantize, quantize
from quant.grids import Q4_0Grid
from quant.refold import Geometry, Refold, match_rows, row_ratio
from quant.scale import group_mean, head_channel_share, kv_group_rows, kv_group_share

HIDDEN, INTER = 256, 96
GEOMETRY = Geometry(heads=4, kv_heads=2, head_dim=8, v_dim=16)
V_HEADS = 3


def scales(n: int, gen: torch.Generator, share: torch.Tensor | None = None) -> torch.Tensor:
    """Column scales with the geometric mean one, as the search makes them."""
    log_t = 0.3 * torch.randn(n, generator=gen)
    if share is not None:
        log_t = group_mean(log_t, share)
    return (log_t - log_t.mean()).exp()


def q4_pack(w: torch.Tensor) -> torch.Tensor:
    """The dequantized Q4_0 pack of w, with the scale search."""
    grid = Q4_0Grid()
    idx, d = quantize(grid, w, search=True)
    return dequantize(grid, idx, d)


def q8_round_trip(w: torch.Tensor) -> torch.Tensor:
    q, d = q8_0_quantize(w)
    return q8_0_dequantize(q, d)


@pytest.fixture
def layer() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(source, folded, folds, packs) of one GDN layer 0 and one attention layer 1, as the flow makes them."""
    gen = torch.Generator().manual_seed(7)
    g = GEOMETRY
    src: dict[str, torch.Tensor] = {}
    folded: dict[str, torch.Tensor] = {}
    folds: dict[str, torch.Tensor] = {}
    packs: dict[str, torch.Tensor] = {}

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen) * 0.05

    # Layer 0: the GDN mixer. The input norm holds t_mix, the gated norm holds t_ch.
    t_mix = scales(HIDDEN, gen)
    src["blk.0.attn_norm.weight"] = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
    folds["blk.0.attn_norm.weight"] = src["blk.0.attn_norm.weight"] / t_mix
    for tail, rows in (("attn_qkv.weight", 2 * V_HEADS * g.v_dim + V_HEADS * g.v_dim), ("attn_gate.weight", V_HEADS * g.v_dim),
                       ("ssm_alpha.weight", V_HEADS), ("ssm_beta.weight", V_HEADS)):
        src[f"blk.0.{tail}"] = rand(rows, HIDDEN)
        folded[f"blk.0.{tail}"] = src[f"blk.0.{tail}"] * t_mix[None, :]
    folds["blk.0.ssm_alpha.weight"] = folded["blk.0.ssm_alpha.weight"]
    folds["blk.0.ssm_beta.weight"] = folded["blk.0.ssm_beta.weight"]
    t_out = scales(V_HEADS * g.v_dim, gen, head_channel_share(V_HEADS * g.v_dim, g.v_dim, torch.device("cpu")))
    src["blk.0.ssm_norm.weight"] = 1.0 + 0.1 * torch.randn(g.v_dim, generator=gen)
    folds["blk.0.ssm_norm.weight"] = src["blk.0.ssm_norm.weight"] / t_out[:g.v_dim]
    src["blk.0.ssm_out.weight"] = rand(HIDDEN, V_HEADS * g.v_dim)
    folded["blk.0.ssm_out.weight"] = src["blk.0.ssm_out.weight"] * t_out[None, :]

    # Layer 1: the attention mixer. The input norm holds t_mix, the v rows hold the o_proj scales.
    t_mix = scales(HIDDEN, gen)
    src["blk.1.attn_norm.weight"] = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
    folds["blk.1.attn_norm.weight"] = src["blk.1.attn_norm.weight"] / t_mix
    cols = g.heads * g.head_dim
    t_o = scales(cols, gen, kv_group_share(cols, g.heads, g.kv_heads, g.head_dim, torch.device("cpu")))
    for tail, rows in (("attn_q.weight", 2 * cols), ("attn_k.weight", g.kv_heads * g.head_dim), ("attn_v.weight", g.kv_heads * g.head_dim)):
        src[f"blk.1.{tail}"] = rand(rows, HIDDEN)
        folded[f"blk.1.{tail}"] = src[f"blk.1.{tail}"] * t_mix[None, :]
    folded["blk.1.attn_v.weight"] = folded["blk.1.attn_v.weight"] / kv_group_rows(t_o, g.heads, g.kv_heads, g.head_dim)[:, None]
    folds["blk.1.attn_v.weight"] = q8_round_trip(folded["blk.1.attn_v.weight"])
    folds["blk.1.attn_k.weight"] = q8_round_trip(folded["blk.1.attn_k.weight"])
    src["blk.1.attn_output.weight"] = rand(HIDDEN, cols)
    folded["blk.1.attn_output.weight"] = src["blk.1.attn_output.weight"] * t_o[None, :]

    # The MLP of both layers: the permutation, t_dn in the up rows, t_in in the post-attention norm.
    for li in (0, 1):
        perm = torch.randperm(INTER, generator=gen)
        t_dn, t_in = scales(INTER, gen), scales(HIDDEN, gen)
        src[f"blk.{li}.post_attention_norm.weight"] = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
        folds[f"blk.{li}.post_attention_norm.weight"] = src[f"blk.{li}.post_attention_norm.weight"] / t_in
        gate, up, down = rand(INTER, HIDDEN), rand(INTER, HIDDEN), rand(HIDDEN, INTER)
        src[f"blk.{li}.ffn_gate.weight"], src[f"blk.{li}.ffn_up.weight"], src[f"blk.{li}.ffn_down.weight"] = gate, up, down
        folded[f"blk.{li}.ffn_gate.weight"] = gate[perm] * t_in[None, :]
        folded[f"blk.{li}.ffn_up.weight"] = up[perm] * t_in[None, :] / t_dn[:, None]
        folded[f"blk.{li}.ffn_down.weight"] = down[:, perm] * t_dn[None, :]
        for tail in ("ffn_gate.weight", "ffn_up.weight"):
            packs[f"blk.{li}.{tail}"] = q4_pack(folded[f"blk.{li}.{tail}"])
    # The head: its scales in the output norm.
    t_h = scales(HIDDEN, gen)
    src["output_norm.weight"] = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
    folds["output_norm.weight"] = src["output_norm.weight"] / t_h
    src["output.weight"] = rand(40, HIDDEN)
    folded["output.weight"] = src["output.weight"] * t_h[None, :]
    return src, folded, folds, packs


def refold_of(src, folds, packs) -> Refold:
    return Refold(lambda n: src[n].numpy(), lambda n: folds[n].numpy(), lambda n: packs.get(n), GEOMETRY, torch.device("cpu"))


def test_norm_ratio_classes_are_exact(layer) -> None:
    """The mixer inputs, ssm_out and the head come back to float precision from the norm ratios."""
    src, folded, folds, packs = layer
    refold = refold_of(src, folds, packs)
    for name in ("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_out.weight",
                 "blk.1.attn_q.weight", "blk.1.attn_k.weight", "output.weight"):
        torch.testing.assert_close(refold.weight(name, src[name].numpy()), folded[name], atol=1e-6, rtol=1e-5)


def test_attention_output_scales_come_from_the_v_rows(layer) -> None:
    """The o_proj scales come back through the Q8 rows of attn_v to the precision of Q8_0."""
    src, folded, folds, packs = layer
    refold = refold_of(src, folds, packs)
    got = refold.weight("blk.1.attn_output.weight", src["blk.1.attn_output.weight"].numpy())
    want = folded["blk.1.attn_output.weight"]
    assert ((got - want).norm() / want.norm()).item() < 2e-3
    got_v = refold.weight("blk.1.attn_v.weight", src["blk.1.attn_v.weight"].numpy())
    assert ((got_v - folded["blk.1.attn_v.weight"]).norm() / want.norm()).item() < 2e-3


def test_mlp_permutation_and_down_scales_come_from_the_packs(layer) -> None:
    """The permutation is exact and the down scales come back to the noise of the Q4 rows."""
    src, folded, folds, packs = layer
    refold = refold_of(src, folds, packs)
    for li in (0, 1):
        gate = refold.weight(f"blk.{li}.ffn_gate.weight", src[f"blk.{li}.ffn_gate.weight"].numpy())
        torch.testing.assert_close(gate, folded[f"blk.{li}.ffn_gate.weight"], atol=1e-6, rtol=1e-5)
        for tail in ("ffn_up.weight", "ffn_down.weight"):
            got = refold.weight(f"blk.{li}.{tail}", src[f"blk.{li}.{tail}"].numpy())
            want = folded[f"blk.{li}.{tail}"]
            assert ((got - want).norm() / want.norm()).item() < 0.02, tail
        err = refold.pack_error(f"blk.{li}.ffn_up.weight", src[f"blk.{li}.ffn_up.weight"].numpy())
        assert err is not None and 0.05 < err < 0.2, "the pack differs from the folded source by the 4-bit rounding only"
    assert any("MLP: match cosine" in n for n in refold.notes)


def test_row_match_rejects_a_disagreement(layer) -> None:
    """Two packs with different permutations are an error, not a silent guess."""
    src, folded, folds, packs = layer
    bad = dict(packs)
    bad["blk.0.ffn_up.weight"] = packs["blk.0.ffn_up.weight"].flip(0)
    with pytest.raises(ValueError, match="different channel permutations"):
        refold_of(src, folds, bad).weight("blk.0.ffn_down.weight", src["blk.0.ffn_down.weight"].numpy())


def test_row_ratio_and_match_rows() -> None:
    """The ratio is exact for a scaled row, and the match finds a permutation under noise."""
    gen = torch.Generator().manual_seed(1)
    a = torch.randn(10, 64, generator=gen)
    t = torch.rand(10, generator=gen) + 0.5
    torch.testing.assert_close(row_ratio(a, a / t[:, None]), t, atol=1e-5, rtol=1e-5)
    perm = torch.randperm(10, generator=gen)
    noisy = a[perm] + 0.1 * torch.randn(10, 64, generator=gen)
    got, cos, margin = match_rows(noisy, a)
    assert torch.equal(got, perm) and cos > 0.9 and margin > 0.5


def test_geometry_from_gguf_fields() -> None:
    """The head counts come from the architecture fields of the reader."""
    class Field:
        def __init__(self, v: int) -> None:
            self.v = v

        def contents(self) -> int:
            return self.v

    class Reader:
        fields = {"qwen35.attention.head_count": Field(8), "qwen35.attention.head_count_kv": Field(2),
                  "qwen35.attention.key_length": Field(256)}

    assert Geometry.from_gguf(Reader(), "qwen35", 128) == Geometry(8, 2, 256, 128)
    assert np.isfinite(row_ratio(torch.ones(2, 4), torch.ones(2, 4))).all()
