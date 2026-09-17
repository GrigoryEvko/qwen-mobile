"""The function-preserving transform on a toy model with the tensor names of the checkpoint."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
import torch.nn.functional as F

from quant.checkpoint import LM, MTP, MTP_HEAD_ROT, MTP_HNORM_ROT, OUTPUT_ROT
from quant.transform import _rotate_output, hadamard, rotation_matrix, transform

EPS = 1e-6


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The zero-centered RMSNorm of Qwen3.5: norm(x) · (1 + w)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w)


def toy_layer_tensors(t: "OrderedDict[str, torch.Tensor]", p: str, kind: str, d: int, gen: torch.Generator) -> None:
    """Put the tensors of one decoder layer with the prefix ``p`` into ``t``."""
    inter, heads, dim, key_dim, value_dim, v_heads = 3 * d, 2, d // 2, d, 2 * d, 4

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32) / shape[-1] ** 0.5

    t[p + "input_layernorm.weight"] = 0.3 * torch.randn(d, generator=gen)
    t[p + "post_attention_layernorm.weight"] = 0.3 * torch.randn(d, generator=gen)
    if kind == "full_attention":
        t[p + "self_attn.q_proj.weight"] = rand(heads * dim * 2, d)
        t[p + "self_attn.k_proj.weight"] = rand(heads * dim, d)
        t[p + "self_attn.v_proj.weight"] = rand(heads * dim, d)
        t[p + "self_attn.o_proj.weight"] = rand(d, heads * dim)
        t[p + "self_attn.q_norm.weight"] = 0.1 * torch.randn(dim, generator=gen)
        t[p + "self_attn.k_norm.weight"] = 0.1 * torch.randn(dim, generator=gen)
    else:
        t[p + "linear_attn.in_proj_qkv.weight"] = rand(2 * key_dim + value_dim, d)
        t[p + "linear_attn.in_proj_z.weight"] = rand(value_dim, d)
        t[p + "linear_attn.in_proj_a.weight"] = rand(v_heads, d)
        t[p + "linear_attn.in_proj_b.weight"] = rand(v_heads, d)
        t[p + "linear_attn.out_proj.weight"] = rand(d, value_dim)
        t[p + "linear_attn.norm.weight"] = torch.ones(value_dim // v_heads)
    t[p + "mlp.gate_proj.weight"] = rand(inter, d)
    t[p + "mlp.up_proj.weight"] = rand(inter, d)
    t[p + "mlp.down_proj.weight"] = rand(d, inter)


def toy_tensors(d: int, vocab: int, layer_types: list[str], untied: bool, merger: bool,
                gen: torch.Generator, mtp: bool = False) -> "OrderedDict[str, torch.Tensor]":
    """A checkpoint dictionary of a toy model: attention and GDN layers, an MLP, the head, the merger.

    With ``mtp`` the dictionary holds the MTP block of the checkpoint: the
    two input norms, ``fc``, one full-attention layer and the head norm.
    """

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32) / shape[-1] ** 0.5

    t: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    t[LM + "embed_tokens.weight"] = rand(vocab, d)
    t[LM + "norm.weight"] = 0.3 * torch.randn(d, generator=gen)
    if untied:
        t["lm_head.weight"] = rand(vocab, d)
    for i, kind in enumerate(layer_types):
        toy_layer_tensors(t, f"{LM}layers.{i}.", kind, d, gen)
    if merger:
        t["model.visual.merger.linear_fc2.weight"] = rand(d, 2 * d)
        t["model.visual.merger.linear_fc2.bias"] = 0.1 * torch.randn(d, generator=gen)
    if mtp:
        t[MTP + "pre_fc_norm_embedding.weight"] = 0.3 * torch.randn(d, generator=gen)
        t[MTP + "pre_fc_norm_hidden.weight"] = 0.3 * torch.randn(d, generator=gen)
        t[MTP + "fc.weight"] = rand(d, 2 * d)
        toy_layer_tensors(t, f"{MTP}layers.0.", "full_attention", d, gen)
        t[MTP + "norm.weight"] = 0.3 * torch.randn(d, generator=gen)
    return t


def toy_layer(f: dict[str, torch.Tensor], p: str, kind: str, h: torch.Tensor) -> torch.Tensor:
    """One decoder layer on the residual ``h``. The mixers are nonlinear functions of their projections."""
    n = rmsnorm(h, f[p + "input_layernorm.weight"])
    if kind == "full_attention":
        q = n @ f[p + "self_attn.q_proj.weight"].T
        k = n @ f[p + "self_attn.k_proj.weight"].T
        v = n @ f[p + "self_attn.v_proj.weight"].T
        m = torch.tanh(q[:, : k.shape[1]]) * k + torch.sigmoid(q[:, k.shape[1]:]) * v
        h = h + m @ f[p + "self_attn.o_proj.weight"].T
    else:
        qkv = n @ f[p + "linear_attn.in_proj_qkv.weight"].T
        z = n @ f[p + "linear_attn.in_proj_z.weight"].T
        a = n @ f[p + "linear_attn.in_proj_a.weight"].T
        b = n @ f[p + "linear_attn.in_proj_b.weight"].T
        value = qkv[:, -z.shape[1]:]
        m = torch.tanh(value) * F.silu(z) * torch.sigmoid(a.sum(-1, keepdim=True)) * F.softplus(b.sum(-1, keepdim=True))
        h = h + m @ f[p + "linear_attn.out_proj.weight"].T
    n2 = rmsnorm(h, f[p + "post_attention_layernorm.weight"])
    mlp = F.silu(n2 @ f[p + "mlp.gate_proj.weight"].T) * (n2 @ f[p + "mlp.up_proj.weight"].T)
    return h + mlp @ f[p + "mlp.down_proj.weight"].T


def toy_head(f: dict[str, torch.Tensor], n: torch.Tensor) -> torch.Tensor:
    """The head path of the graph on a normed state: the dense map M of a tied head, then the head."""
    if OUTPUT_ROT in f:
        n = n @ f[OUTPUT_ROT].T
    head = f["lm_head.weight"] if "lm_head.weight" in f else f[LM + "embed_tokens.weight"]
    return n @ head.T


def toy_trunk(t: dict[str, torch.Tensor], layer_types: list[str], ids: torch.Tensor,
              image: torch.Tensor | None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The float64 tensors and the final-norm output of the toy model (``h_nextn`` of the graph)."""
    f = {k: v.to(torch.float64) for k, v in t.items()}
    h = f[LM + "embed_tokens.weight"][ids]
    if image is not None:
        w, b = f["model.visual.merger.linear_fc2.weight"], f["model.visual.merger.linear_fc2.bias"]
        h = torch.cat([image.to(torch.float64) @ w.T + b, h], 0)
    for i, kind in enumerate(layer_types):
        h = toy_layer(f, f"{LM}layers.{i}.", kind, h)
    return f, rmsnorm(h, f[LM + "norm.weight"])


def toy_forward(t: dict[str, torch.Tensor], layer_types: list[str], ids: torch.Tensor,
                image: torch.Tensor | None) -> torch.Tensor:
    """The logits of the toy model: the trunk, the final norm, the head path."""
    f, n = toy_trunk(t, layer_types, ids, image)
    return toy_head(f, n)


def toy_mtp_forward(t: dict[str, torch.Tensor], layer_types: list[str], ids: torch.Tensor,
                    draft_ids: list[torch.Tensor]) -> list[torch.Tensor]:
    """The draft logits of the MTP block, one tensor per chained step, as the llama.cpp graph computes them.

    Step 0 reads the final-norm output of the trunk. Each subsequent step
    reads the ``h_nextn`` of the MTP block of the step before it, which
    ``common_speculative_impl_draft_mtp::draft`` sends as the next input.
    The token of each step is given, thus the test does not use the argmax.
    """
    f, h = toy_trunk(t, layer_types, ids, None)
    emb = f[LM + "embed_tokens.weight"]
    logits = []
    for step_ids in draft_ids:
        if MTP_HNORM_ROT in f:
            h = h @ f[MTP_HNORM_ROT].T
        h_norm = rmsnorm(h, f[MTP + "pre_fc_norm_hidden.weight"])
        e_norm = rmsnorm(emb[step_ids], f[MTP + "pre_fc_norm_embedding.weight"])
        x = torch.cat([e_norm, h_norm], dim=-1) @ f[MTP + "fc.weight"].T
        x = toy_layer(f, f"{MTP}layers.0.", "full_attention", x)
        h = rmsnorm(x, f[MTP + "norm.weight"])
        if MTP_HEAD_ROT in f:
            h = h @ f[MTP_HEAD_ROT].T
        logits.append(toy_head(f, h))
    return logits


@pytest.mark.parametrize("n", [1, 2, 32, 40, 2560])
def test_hadamard_is_orthogonal(n: int) -> None:
    """The Sylvester matrix, and the Kronecker product with the seeded odd factor, are orthogonal."""
    h = hadamard(n, torch.device("cpu"))
    assert h.shape == (n, n) and h.dtype == torch.float64
    torch.testing.assert_close(h @ h.T, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=0)
    if n > 1:
        assert (h.abs() > 1e-9).all(), "every entry mixes every channel"


def test_rotation_matrix_is_orthogonal_and_seeded() -> None:
    q = rotation_matrix(64, 16, seed=3, device=torch.device("cpu"))
    torch.testing.assert_close(q @ q.T, torch.eye(64, dtype=torch.float64), atol=1e-12, rtol=0)
    assert torch.equal(q, rotation_matrix(64, 16, seed=3, device=torch.device("cpu")))
    assert not torch.equal(q, rotation_matrix(64, 16, seed=4, device=torch.device("cpu")))
    with pytest.raises(ValueError):
        rotation_matrix(64, 24, seed=0, device=torch.device("cpu"))


@pytest.mark.parametrize("d,block", [(32, None), (32, 8), (40, None), (40, 8)])
@pytest.mark.parametrize("permute", [False, True])
@pytest.mark.parametrize("untied", [False, True])
def test_transform_keeps_the_function(d: int, block: int | None, permute: bool, untied: bool) -> None:
    """The fold, the rotation (Sylvester or Kronecker), the permutation and the merger keep the logits."""
    gen = torch.Generator().manual_seed(d + 7 * permute + 13 * untied)
    layer_types = ["linear_attention", "full_attention"]
    tensors = toy_tensors(d, vocab=50, layer_types=layer_types, untied=untied, merger=True, gen=gen)
    ids = torch.randint(0, 50, (9,), generator=gen)
    image = torch.randn(3, 2 * d, generator=gen)
    before = toy_forward(tensors, layer_types, ids, image)
    out = transform(tensors, 2, layer_types, rotate=True, block=block, seed=1, permute_mlp=permute,
                    device=torch.device("cpu"))
    after = toy_forward(out, layer_types, ids, image)
    torch.testing.assert_close(after, before, atol=2e-4 * before.abs().max().item(), rtol=0)
    for name in (LM + "norm.weight", f"{LM}layers.0.input_layernorm.weight", f"{LM}layers.1.post_attention_layernorm.weight"):
        assert torch.equal(out[name], torch.zeros(d)), "a folded norm is the identity (zero-centered)"
    assert out[f"{LM}layers.0.linear_attn.norm.weight"].dtype == torch.float32
    q = rotation_matrix(d, block, 1, torch.device("cpu"))
    emb = tensors[LM + "embed_tokens.weight"].to(torch.float64)
    torch.testing.assert_close(out[LM + "embed_tokens.weight"], (emb @ q).to(torch.float32))
    if untied:
        assert not torch.allclose(out["lm_head.weight"], out[LM + "embed_tokens.weight"]), "the untied head stays its own"


@pytest.mark.parametrize("d,block", [(32, None), (40, 8)])
def test_tied_transform_keeps_the_function(d: int, block: int | None) -> None:
    """With tie_head the head stays the embedding, and the dense map after the final norm gives the same logits."""
    gen = torch.Generator().manual_seed(d)
    layer_types = ["linear_attention", "full_attention"]
    tensors = toy_tensors(d, vocab=50, layer_types=layer_types, untied=False, merger=False, gen=gen)
    ids = torch.randint(0, 50, (9,), generator=gen)
    before = toy_forward(tensors, layer_types, ids, None)
    kw = dict(rotate=True, block=block, seed=1, permute_mlp=True, device=torch.device("cpu"))
    tied = transform(tensors, 2, layer_types, tie_head=True, **kw)
    untied = transform(tensors, 2, layer_types, tie_head=False, **kw)
    assert "lm_head.weight" not in tied and OUTPUT_ROT not in untied
    m = tied[OUTPUT_ROT]
    assert m.shape == (d, d) and m.dtype == torch.float32
    torch.testing.assert_close(m, m.T, atol=0, rtol=0)
    after = toy_forward(tied, layer_types, ids, None)
    torch.testing.assert_close(after, before, atol=2e-4 * before.abs().max().item(), rtol=0)
    assert torch.equal(tied[LM + "norm.weight"], torch.zeros(d))
    assert torch.equal(tied[LM + "embed_tokens.weight"], untied[LM + "embed_tokens.weight"])
    # E' · M is the untied head: the two variants differ only in where diag(γ_f) · Q sits.
    e_m = tied[LM + "embed_tokens.weight"].to(torch.float64) @ m.to(torch.float64)
    torch.testing.assert_close(e_m, untied["lm_head.weight"].to(torch.float64), atol=1e-5, rtol=1e-4)


def test_tie_head_refuses_a_checkpoint_with_its_own_head() -> None:
    gen = torch.Generator().manual_seed(2)
    tensors = toy_tensors(32, vocab=20, layer_types=["full_attention"], untied=True, merger=False, gen=gen)
    with pytest.raises(ValueError):
        transform(tensors, 1, ["full_attention"], rotate=True, block=None, seed=0, permute_mlp=False,
                  device=torch.device("cpu"), tie_head=True)
    tensors["lm_head.weight"] = tensors[LM + "embed_tokens.weight"].clone()
    out = transform(tensors, 1, ["full_attention"], rotate=True, block=None, seed=0, permute_mlp=False,
                    device=torch.device("cpu"), tie_head=True)
    assert "lm_head.weight" not in out, "a head equal to the embedding is a tied head"


def test_vision_tensors_keep_their_dtype() -> None:
    """The rotated merger goes back to the dtype of the vision tower, the text tensors become float32."""
    gen = torch.Generator().manual_seed(5)
    layer_types = ["full_attention"]
    tensors = toy_tensors(32, vocab=20, layer_types=layer_types, untied=False, merger=True, gen=gen)
    for name in ("model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.bias"):
        tensors[name] = tensors[name].to(torch.bfloat16)
    tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(8, 8, generator=gen).to(torch.bfloat16)
    tensors[LM + "layers.0.self_attn.q_norm.weight"] = torch.zeros(16, dtype=torch.bfloat16)
    out = transform(tensors, 1, layer_types, rotate=True, block=None, seed=0, permute_mlp=False,
                    device=torch.device("cpu"))
    q = rotation_matrix(32, None, 0, torch.device("cpu"))
    w = tensors["model.visual.merger.linear_fc2.weight"].to(torch.float64)
    assert out["model.visual.merger.linear_fc2.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(out["model.visual.merger.linear_fc2.weight"].to(torch.float64), q.T @ w, atol=0, rtol=2**-7)
    assert torch.equal(out["model.visual.blocks.0.attn.qkv.weight"], tensors["model.visual.blocks.0.attn.qkv.weight"])
    assert out[LM + "layers.0.self_attn.q_norm.weight"].dtype == torch.float32


def test_no_rotation_with_permutation_keeps_the_function() -> None:
    gen = torch.Generator().manual_seed(11)
    layer_types = ["full_attention"]
    tensors = toy_tensors(32, vocab=20, layer_types=layer_types, untied=False, merger=False, gen=gen)
    ids = torch.randint(0, 20, (5,), generator=gen)
    before = toy_forward(tensors, layer_types, ids, None)
    out = transform(tensors, 1, layer_types, rotate=False, block=None, seed=0, permute_mlp=True,
                    device=torch.device("cpu"))
    torch.testing.assert_close(toy_forward(out, layer_types, ids, None), before, atol=1e-5, rtol=0)
    assert torch.equal(out[LM + "embed_tokens.weight"], tensors[LM + "embed_tokens.weight"])


def test_mmproj_rotation_matches_the_checkpoint_transform() -> None:
    """rotate_mmproj applies Qᵀ·W and Qᵀ·b, the same as _rotate_output in the transform."""
    q = rotation_matrix(32, None, 0, torch.device("cpu"))
    w = torch.randn(32, 48, dtype=torch.float64)
    b = torch.randn(32, dtype=torch.float64)
    torch.testing.assert_close(q.T @ w, _rotate_output(w, q))
    torch.testing.assert_close(q.T @ b, (q.T @ b[:, None]).squeeze(1))


@pytest.mark.parametrize("d,block", [(32, None), (40, 8)])
@pytest.mark.parametrize("tie", [False, True])
def test_mtp_transform_keeps_the_draft_logits(d: int, block: int | None, tie: bool) -> None:
    """Three chained draft steps give the logits of the original block, tied or untied, with the MLP permutation."""
    gen = torch.Generator().manual_seed(d + 3 * tie)
    layer_types = ["linear_attention", "full_attention"]
    tensors = toy_tensors(d, vocab=50, layer_types=layer_types, untied=False, merger=False, gen=gen, mtp=True)
    ids = torch.randint(0, 50, (7,), generator=gen)
    drafts = [torch.randint(0, 50, (7,), generator=gen) for _ in range(3)]
    before = toy_mtp_forward(tensors, layer_types, ids, drafts)
    out = transform(tensors, 2, layer_types, rotate=True, block=block, seed=1, permute_mlp=True,
                    device=torch.device("cpu"), tie_head=tie)
    after = toy_mtp_forward(out, layer_types, ids, drafts)
    for step, (a, b) in enumerate(zip(after, before)):
        torch.testing.assert_close(a, b, atol=2e-4 * b.abs().max().item(), rtol=0, msg=f"draft step {step}")
    q = rotation_matrix(d, block, 1, torch.device("cpu"))
    gamma_f = 1.0 + tensors[LM + "norm.weight"].to(torch.float64)
    gamma_s = 1.0 + tensors[MTP + "norm.weight"].to(torch.float64)
    torch.testing.assert_close(out[MTP_HNORM_ROT].to(torch.float64), gamma_f[:, None] * q, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out[MTP_HEAD_ROT].to(torch.float64), q.T @ ((gamma_s / gamma_f)[:, None] * q),
                               atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out[MTP_HEAD_ROT], out[MTP_HEAD_ROT].T, atol=0, rtol=0)
    for name in (MTP + "pre_fc_norm_embedding.weight", MTP + "norm.weight", f"{MTP}layers.0.input_layernorm.weight",
                 f"{MTP}layers.0.post_attention_layernorm.weight"):
        assert torch.equal(out[name], torch.zeros(d)), f"{name} is the identity after the fold"
    assert torch.equal(out[MTP + "pre_fc_norm_hidden.weight"], tensors[MTP + "pre_fc_norm_hidden.weight"])
    assert out[MTP + "fc.weight"].shape == (d, 2 * d) and out[MTP + "fc.weight"].dtype == torch.float32
    assert torch.equal(out[f"{MTP}layers.0.self_attn.q_norm.weight"], tensors[f"{MTP}layers.0.self_attn.q_norm.weight"])


def test_mtp_maps_are_the_same_tied_and_untied() -> None:
    """The head map of the MTP block is the same with and without the tie: the tied file applies M after it."""
    gen = torch.Generator().manual_seed(21)
    layer_types = ["full_attention"]
    tensors = toy_tensors(32, vocab=20, layer_types=layer_types, untied=False, merger=False, gen=gen, mtp=True)
    kw = dict(rotate=True, block=None, seed=2, permute_mlp=False, device=torch.device("cpu"))
    tied = transform(tensors, 1, layer_types, tie_head=True, **kw)
    untied = transform(tensors, 1, layer_types, tie_head=False, **kw)
    for name in (MTP_HNORM_ROT, MTP_HEAD_ROT, MTP + "fc.weight"):
        assert torch.equal(tied[name], untied[name]), name


def test_mtp_transform_is_optional() -> None:
    """Without ``mtp`` the block passes through as it is, and ``mtp`` is not possible without the block."""
    gen = torch.Generator().manual_seed(4)
    layer_types = ["full_attention"]
    kw = dict(rotate=True, block=None, seed=0, permute_mlp=False, device=torch.device("cpu"))
    tensors = toy_tensors(32, vocab=20, layer_types=layer_types, untied=False, merger=False, gen=gen, mtp=True)
    out = transform(tensors, 1, layer_types, mtp=False, **kw)
    assert MTP_HNORM_ROT not in out and MTP_HEAD_ROT not in out
    for name in tensors:
        if name.startswith(MTP):
            assert torch.equal(out[name], tensors[name]), name
    assert MTP_HNORM_ROT in transform(tensors, 1, layer_types, **kw), "the default is on when the block exists"
    plain = toy_tensors(32, vocab=20, layer_types=layer_types, untied=False, merger=False, gen=gen)
    assert MTP_HNORM_ROT not in transform(plain, 1, layer_types, **kw)
    with pytest.raises(ValueError):
        transform(plain, 1, layer_types, mtp=True, **kw)


def test_mtp_transform_refuses_a_final_norm_near_zero() -> None:
    gen = torch.Generator().manual_seed(6)
    layer_types = ["full_attention"]
    tensors = toy_tensors(32, vocab=20, layer_types=layer_types, untied=False, merger=False, gen=gen, mtp=True)
    tensors[LM + "norm.weight"][3] = -1.0
    with pytest.raises(ValueError):
        transform(tensors, 1, layer_types, rotate=True, block=None, seed=0, permute_mlp=False,
                  device=torch.device("cpu"))
