"""The export against the converter: the value-head order of the GDN tensors, the LoRA adapter, the guard, the tie."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from quant.blockopt import weighted_low_rank
from quant.export import LinearAttentionLayout, _f32_of, block_permutation, export, write_adapter
from quant.grid import dequantize, pack_nibbles, q8_0_dequantize, q8_0_quantize, quantize
from quant.grids import Q4_0Grid
from quant.plan import Plan

ROOT = Path(__file__).resolve().parents[2]
LLAMA = ROOT / "llama.cpp"


def converter_reorder():
    """The _reorder_v_heads of the llama.cpp converter, the reference of the GGUF order."""
    for p in (LLAMA, LLAMA / "gguf-py"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from conversion.qwen import _LinearAttentionVReorderBase

    return _LinearAttentionVReorderBase._reorder_v_heads


def gguf_module():
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    import gguf

    return gguf


LAYOUT = LinearAttentionLayout(num_k_heads=2, num_v_heads=6, head_k_dim=32, head_v_dim=32)


def test_layout_rows_match_the_converter() -> None:
    """attn_qkv (V rows), attn_gate, ssm_alpha and ssm_beta take the tiled order of the converter."""
    reorder = converter_reorder()
    per_k = LAYOUT.num_v_heads // LAYOUT.num_k_heads
    qk = 2 * LAYOUT.num_k_heads * LAYOUT.head_k_dim
    hidden = 48
    qkv = torch.randn(qk + LAYOUT.num_v_heads * LAYOUT.head_v_dim, hidden)
    expected = torch.cat([qkv[:qk], reorder(qkv[qk:], 0, LAYOUT.num_k_heads, per_k, LAYOUT.head_v_dim)], 0)
    got = torch.from_numpy(LAYOUT.array("blk.3.attn_qkv.weight", qkv.numpy()))
    assert torch.equal(got, expected)
    z = torch.randn(LAYOUT.num_v_heads * LAYOUT.head_v_dim, hidden)
    assert torch.equal(torch.from_numpy(LAYOUT.array("blk.3.attn_gate.weight", z.numpy())),
                       reorder(z, 0, LAYOUT.num_k_heads, per_k, LAYOUT.head_v_dim))
    a = torch.randn(LAYOUT.num_v_heads, hidden)
    for name in ("blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight"):
        assert torch.equal(torch.from_numpy(LAYOUT.array(name, a.numpy())), reorder(a, 0, LAYOUT.num_k_heads, per_k, 1))
    norm = np.random.rand(LAYOUT.head_v_dim).astype(np.float32)
    assert np.array_equal(LAYOUT.array("blk.0.ssm_norm.weight", norm), norm)
    assert np.array_equal(LAYOUT.array("blk.0.ffn_gate.weight", a.numpy()), a.numpy())


def test_layout_cols_match_the_converter_through_the_pack() -> None:
    """The solved out_proj pack, permuted by blocks, dequantizes to the converter's column order."""
    reorder = converter_reorder()
    per_k = LAYOUT.num_v_heads // LAYOUT.num_k_heads
    torch.manual_seed(0)
    w = torch.randn(48, LAYOUT.num_v_heads * LAYOUT.head_v_dim) * 0.05
    grid = Q4_0Grid()
    idx, d = quantize(grid, w, search=True)
    idx2, d2 = LAYOUT.pack("blk.1.ssm_out.weight", idx, d)
    expected = reorder(dequantize(grid, idx, d), 1, LAYOUT.num_k_heads, per_k, LAYOUT.head_v_dim)
    assert torch.equal(dequantize(grid, idx2, d2), expected)
    a, b = np.random.rand(4, w.shape[1]).astype(np.float32), np.random.rand(48, 4).astype(np.float32)
    a2, b2 = LAYOUT.factors("blk.1.ssm_out.weight", a, b)
    assert np.array_equal(b2, b)
    assert torch.equal(torch.from_numpy(a2), reorder(torch.from_numpy(a), 1, LAYOUT.num_k_heads, per_k, LAYOUT.head_v_dim))
    b_gate = np.random.rand(LAYOUT.num_v_heads * LAYOUT.head_v_dim, 4).astype(np.float32)
    a3, b3 = LAYOUT.factors("blk.1.attn_gate.weight", a, b_gate)
    assert np.array_equal(a3, a), "the input side of attn_gate is the hidden size, not permuted"
    assert torch.equal(torch.from_numpy(b3), reorder(torch.from_numpy(b_gate), 0, LAYOUT.num_k_heads, per_k, LAYOUT.head_v_dim))


def test_layout_is_the_identity_with_one_value_head_per_key_head() -> None:
    """The 2B (16 key heads, 16 value heads) keeps the checkpoint order."""
    same = LinearAttentionLayout(16, 16, 128, 128)
    for name in ("blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_out.weight"):
        assert same.rows(name) is None and same.cols(name) is None


def test_block_permutation_refuses_a_split_block() -> None:
    cols = torch.arange(64)
    assert block_permutation(cols).tolist() == [0, 1]
    assert block_permutation(torch.cat([cols[32:], cols[:32]])).tolist() == [1, 0]
    with pytest.raises(ValueError):
        block_permutation(torch.cat([cols[16:], cols[:16]]))


def test_lora_adapter_shapes_pass_the_llama_cpp_checks(tmp_path: Path) -> None:
    """lora_a is [rank, in] and lora_b is [out, rank], alpha = rank, as llama-adapter.cpp validates them."""
    gguf = gguf_module()
    torch.manual_seed(1)
    rows, cols, rank = 20, 64, 3
    w_ref = torch.randn(rows, cols)
    grid = Q4_0Grid()
    idx, d = quantize(grid, w_ref, search=True)
    x = torch.randn(200, cols)
    hessian = x.T @ x / 200
    a, b = weighted_low_rank(w_ref - dequantize(grid, idx, d), hessian, rank)
    assert a.shape == (rank, cols) and b.shape == (rows, rank)
    # The rank-r term is the best Hessian-weighted correction: it lowers the weighted error.
    low = torch.linalg.cholesky(hessian + 0.01 * hessian.diag().mean() * torch.eye(cols))
    err_before = ((w_ref - dequantize(grid, idx, d)) @ low).norm()
    err_after = ((w_ref - dequantize(grid, idx, d) - b @ a) @ low).norm()
    assert err_after < err_before
    path = tmp_path / "adapter.gguf"
    write_adapter(gguf, "qwen35", path, {"blk.0.ffn_gate.weight": (a.numpy(), b.numpy())})
    reader = gguf.GGUFReader(str(path))
    assert float(reader.fields["adapter.lora.alpha"].contents()) == float(rank)
    assert bytes(reader.fields["adapter.type"].parts[-1]).decode() == "lora"
    tensors = {t.name: [int(x) for x in t.shape] for t in reader.tensors}
    model_ne = [cols, rows]
    a_ne, b_ne = tensors["blk.0.ffn_gate.weight.lora_a"], tensors["blk.0.ffn_gate.weight.lora_b"]
    assert model_ne[0] == a_ne[0] and model_ne[1] == b_ne[1], "the base tensor shape check of llama-adapter.cpp"
    assert a_ne[1] == b_ne[0] == rank, "the transposed lora_a check of llama-adapter.cpp"


def test_gguf_keeps_the_numpy_row_order_of_a_2d_tensor(tmp_path: Path) -> None:
    """A numpy [out, in] array comes back as [out, in], and the GGUF shape (ne) is the reverse.

    llama.cpp writes the head as numpy [vocab, hidden] and computes
    ``ggml_mul_mat(output, h) = W·h``, thus a numpy [out, in] array is the
    convention for ``ggml_mul_mat(W, v) = W·v``. The tied export writes M
    the same way.
    """
    gguf = gguf_module()
    a = np.arange(15, dtype=np.float32).reshape(3, 5)
    path = tmp_path / "shape.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen35")
    writer.add_tensor("output_rot.weight", a.astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    t = gguf.GGUFReader(str(path)).tensors[0]
    assert [int(x) for x in t.shape] == [5, 3], "the GGUF shape is ne, the reverse of the numpy shape"
    assert np.array_equal(_f32_of(t), a)


def write_toy_f16(gguf, path: Path, vocab: int, d: int, gen: np.random.Generator, tied: bool = False,
                  layer: bool = False, mtp: bool = False) -> dict[str, np.ndarray]:
    """A small F16 GGUF with the head tensors of the converter: token_embd, output_norm, output.

    ``tied`` gives no ``output.weight``, as the converter does for a tied
    checkpoint. ``layer`` adds one decoder block: two F16 matrices, the F16
    GDN control ``ssm_alpha`` and an F32 norm. ``mtp`` adds the MTP block as
    block 1 of a rotated checkpoint: three F16 matrices, two F32 norms and
    the two F16 dense maps.
    """
    tensors = {
        "token_embd.weight": gen.standard_normal((vocab, d)).astype(np.float16),
        "output_norm.weight": (0.5 + gen.random(d)).astype(np.float32),
    }
    if not tied:
        tensors["output.weight"] = gen.standard_normal((vocab, d)).astype(np.float16)
    if layer:
        tensors["blk.0.ffn_gate.weight"] = gen.standard_normal((2 * d, d)).astype(np.float16)
        tensors["blk.0.attn_gate.weight"] = gen.standard_normal((d, d)).astype(np.float16)
        tensors["blk.0.ssm_alpha.weight"] = gen.standard_normal((2, d)).astype(np.float16)
        tensors["blk.0.attn_norm.weight"] = (0.5 + gen.random(d)).astype(np.float32)
    if mtp:
        tensors["blk.1.nextn.eh_proj.weight"] = gen.standard_normal((d, 2 * d)).astype(np.float16)
        tensors["blk.1.attn_q.weight"] = gen.standard_normal((2 * d, d)).astype(np.float16)
        tensors["blk.1.ffn_down.weight"] = gen.standard_normal((d, 2 * d)).astype(np.float16)
        tensors["blk.1.attn_norm.weight"] = (0.5 + gen.random(d)).astype(np.float32)
        tensors["blk.1.nextn.hnorm.weight"] = (0.5 + gen.random(d)).astype(np.float32)
        tensors["blk.1.nextn.hnorm_rot.weight"] = gen.standard_normal((d, d)).astype(np.float16)
        tensors["blk.1.nextn.shared_head_rot.weight"] = gen.standard_normal((d, d)).astype(np.float16)
    writer = gguf.GGUFWriter(str(path), "qwen35")
    writer.add_uint32("qwen35.embedding_length", d)
    # The head geometry that the refold of a promoted class reads.
    writer.add_uint32("qwen35.attention.head_count", 4)
    writer.add_uint32("qwen35.attention.head_count_kv", 2)
    writer.add_uint32("qwen35.attention.key_length", d // 4)
    for name, a in tensors.items():
        writer.add_tensor(name, a)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return tensors


def read_all(gguf, path: Path) -> dict[str, tuple[str, object]]:
    """(type name, reader tensor) by name."""
    return {t.name: (t.tensor_type.name, t) for t in gguf.GGUFReader(str(path)).tensors}


def test_tied_export_drops_the_head_and_writes_the_map(tmp_path: Path) -> None:
    """With tie_head: no output.weight, output_rot in F16 as [out, in], the identity output norm without folds."""
    gguf = gguf_module()
    gen = np.random.default_rng(0)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    write_toy_f16(gguf, src, vocab, d, gen)
    rot = gen.standard_normal((d, d)).astype(np.float32)
    assert not np.allclose(rot, rot.T), "the test map is not symmetric, thus the orientation matters"
    rot_path = tmp_path / "rot.npy"
    np.save(rot_path, rot)
    packs = tmp_path / "packs"
    packs.mkdir()
    out = tmp_path / "tied-f16.gguf"
    export(src, out, packs, Plan(embedding="Q4_0", n_layers=0), LLAMA, torch.device("cpu"), only="^$",
           tie_head=True, rot=rot_path)
    got = read_all(gguf, out)
    assert set(got) == {"token_embd.weight", "output_norm.weight", "output_rot.weight"}
    assert got["token_embd.weight"][0] == "F16" and got["output_rot.weight"][0] == "F16"
    assert np.array_equal(_f32_of(got["output_norm.weight"][1]), np.ones(d, dtype=np.float32))
    np.testing.assert_allclose(_f32_of(got["output_rot.weight"][1]), rot.astype(np.float16).astype(np.float32))
    with pytest.raises(ValueError):
        export(out, tmp_path / "untied.gguf", packs, Plan(n_layers=0), LLAMA, torch.device("cpu"), only="^$")
    with pytest.raises(ValueError):
        export(src, tmp_path / "no-map.gguf", packs, Plan(n_layers=0), LLAMA, torch.device("cpu"), tie_head=True)


def test_q8_0_export_of_a_plain_source_matches_gguf_py(tmp_path: Path) -> None:
    """A plain tied F16 source with the bulk Q8_0: each 2-D weight of the plan is Q8_0 as gguf-py reads it.

    The norms and the GDN control stay F32 with their values, the file type
    is MOSTLY_Q8_0, and no pack directory is necessary.
    """
    gguf = gguf_module()
    gen = np.random.default_rng(2)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    tensors = write_toy_f16(gguf, src, vocab, d, gen, tied=True, layer=True)
    out = tmp_path / "q8.gguf"
    export(src, out, tmp_path / "no-packs", Plan(bulk="Q8_0", gdn_gate="Q8_0", n_layers=1), LLAMA, torch.device("cpu"))
    reader = gguf.GGUFReader(str(out))
    assert int(reader.fields["general.file_type"].contents()) == int(gguf.LlamaFileType.MOSTLY_Q8_0)
    got = {t.name: t for t in reader.tensors}
    assert set(got) == set(tensors)
    for name in ("token_embd.weight", "blk.0.ffn_gate.weight", "blk.0.attn_gate.weight"):
        assert got[name].tensor_type.name == "Q8_0", name
        w = torch.from_numpy(tensors[name].astype(np.float32))
        expected = q8_0_dequantize(*q8_0_quantize(w)).numpy()
        packed = np.asarray(got[name].data).reshape(w.shape[0], -1)
        reference = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
        np.testing.assert_allclose(reference, expected, rtol=0, atol=1e-7)
        rel = np.linalg.norm(expected - w.numpy()) / np.linalg.norm(w.numpy())
        assert rel < 0.01, f"{name}: the 8-bit round trip has the relative error {rel:.4f}"
    for name in ("output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ssm_alpha.weight"):
        assert got[name].tensor_type.name == "F32", name
        np.testing.assert_array_equal(_f32_of(got[name]), tensors[name].astype(np.float32))


def mtp_maps_for(tensors: dict[str, np.ndarray], out_norm: np.ndarray) -> dict[str, np.ndarray]:
    """The two maps of the toy MTP block for the exported output norm, after the F16 round trip."""
    h = tensors["blk.1.nextn.hnorm_rot.weight"].astype(np.float32) / out_norm[None, :]
    s = tensors["blk.1.nextn.shared_head_rot.weight"].astype(np.float32) * out_norm[:, None]
    return {"blk.1.nextn.hnorm_rot.weight": h.astype(np.float16).astype(np.float32),
            "blk.1.nextn.shared_head_rot.weight": s.astype(np.float16).astype(np.float32)}


def test_export_passes_the_mtp_block_and_scales_its_maps(tmp_path: Path) -> None:
    """The MTP block keeps F16 or takes the type ``mtp``, its norms stay F32, and its maps take the output norm.

    An untied source without folds exports its own output norm, thus the
    maps take that norm: diag(s)⁻¹ on the input of ``hnorm_rot`` and
    diag(s) on the output of ``shared_head_rot``.
    """
    gguf = gguf_module()
    gen = np.random.default_rng(3)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    tensors = write_toy_f16(gguf, src, vocab, d, gen, layer=True, mtp=True)
    expected_maps = mtp_maps_for(tensors, tensors["output_norm.weight"])
    for mtp_type, out_name in (("F16", "mtp-f16.gguf"), ("Q8_0", "mtp-q8.gguf")):
        out = tmp_path / out_name
        export(src, out, tmp_path / "no-packs", Plan(bulk="Q8_0", gdn_gate="Q8_0", n_layers=1, mtp=mtp_type), LLAMA,
               torch.device("cpu"))
        got = read_all(gguf, out)
        assert set(got) == set(tensors)
        for name in ("blk.1.nextn.eh_proj.weight", "blk.1.attn_q.weight", "blk.1.ffn_down.weight"):
            assert got[name][0] == mtp_type, (mtp_type, name)
            w = torch.from_numpy(tensors[name].astype(np.float32))
            if mtp_type == "F16":
                np.testing.assert_array_equal(_f32_of(got[name][1]), w.numpy())
            else:
                packed = np.asarray(got[name][1].data).reshape(w.shape[0], -1)
                reference = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
                np.testing.assert_allclose(reference, q8_0_dequantize(*q8_0_quantize(w)).numpy(), rtol=0, atol=1e-7)
        for name in ("blk.1.attn_norm.weight", "blk.1.nextn.hnorm.weight"):
            assert got[name][0] == "F32", name
            np.testing.assert_array_equal(_f32_of(got[name][1]), tensors[name])
        for name, expected in expected_maps.items():
            assert got[name][0] == "F16", name
            np.testing.assert_array_equal(_f32_of(got[name][1]), expected)
        assert not np.array_equal(_f32_of(got["blk.1.nextn.hnorm_rot.weight"][1]),
                                  tensors["blk.1.nextn.hnorm_rot.weight"].astype(np.float32)), "the norm is not one"


def test_tied_export_scales_the_mtp_maps_by_the_folded_norm(tmp_path: Path) -> None:
    """A tied export takes the identity output norm without folds, and the folded norm with them, in its maps too."""
    gguf = gguf_module()
    gen = np.random.default_rng(4)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    tensors = write_toy_f16(gguf, src, vocab, d, gen, mtp=True)
    rot = gen.standard_normal((d, d)).astype(np.float32)
    rot_path = tmp_path / "rot.npy"
    np.save(rot_path, rot)
    packs = tmp_path / "packs"
    packs.mkdir()
    out = tmp_path / "tied-plain.gguf"
    export(src, out, packs, Plan(n_layers=1), LLAMA, torch.device("cpu"), only="^$", tie_head=True, rot=rot_path)
    got = read_all(gguf, out)
    for name, expected in mtp_maps_for(tensors, np.ones(d, dtype=np.float32)).items():
        np.testing.assert_array_equal(_f32_of(got[name][1]), expected)
    norm = (1.0 + 0.1 * gen.standard_normal(d)).astype(np.float32)
    np.savez(packs / "folds.npz", **{"output_norm.weight": norm, "output_rot.weight": rot})
    out = tmp_path / "tied-folded.gguf"
    export(src, out, packs, Plan(n_layers=1), LLAMA, torch.device("cpu"), source_folded=True, tie_head=True)
    got = read_all(gguf, out)
    np.testing.assert_array_equal(_f32_of(got["output_norm.weight"][1]), norm)
    for name, expected in mtp_maps_for(tensors, norm).items():
        np.testing.assert_array_equal(_f32_of(got[name][1]), expected)


def test_tied_export_takes_the_head_pack_and_the_folds(tmp_path: Path) -> None:
    """The solved tied head becomes token_embd, and folds.npz supplies the final norm and the map."""
    gguf = gguf_module()
    gen = np.random.default_rng(1)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    write_toy_f16(gguf, src, vocab, d, gen)
    torch.manual_seed(1)
    grid = Q4_0Grid()
    w = torch.randn(vocab, d) * 0.05
    idx, sc = quantize(grid, w, search=True)
    packs = tmp_path / "packs"
    packs.mkdir()
    np.savez(packs / "token_embd.weight.npz", q=idx.numpy(), d=sc.numpy().view(np.uint16), levels=grid.levels.numpy(),
             kind=np.array("Q4_0"))
    rot = gen.standard_normal((d, d)).astype(np.float32)
    norm = (1.0 + 0.1 * gen.standard_normal(d)).astype(np.float32)
    np.savez(packs / "folds.npz", **{"output_norm.weight": norm, "output_rot.weight": rot})
    out = tmp_path / "tied-q4.gguf"
    export(src, out, packs, Plan(embedding="Q4_0", n_layers=0), LLAMA, torch.device("cpu"), source_folded=True,
           tie_head=True)
    got = read_all(gguf, out)
    assert "output.weight" not in got
    assert got["token_embd.weight"][0] == "Q4_0"
    assert np.array_equal(np.asarray(got["token_embd.weight"][1].data).reshape(vocab, -1), pack_nibbles(idx, sc))
    assert np.array_equal(_f32_of(got["output_norm.weight"][1]), norm)
    np.testing.assert_allclose(_f32_of(got["output_rot.weight"][1]), rot.astype(np.float16).astype(np.float32))


def solved_pack(path: Path, w: np.ndarray) -> None:
    """A Q4_0 pack of ``w`` [rows, cols], as the solver writes one."""
    grid = Q4_0Grid()
    idx, d = quantize(grid, torch.from_numpy(w.astype(np.float32)), search=True)
    np.savez(path, q=idx.numpy(), d=d.numpy().view(np.uint16), levels=grid.levels.numpy(), kind=np.array("Q4_0"))


def test_export_refuses_a_plan_change_on_an_unfolded_source(tmp_path: Path) -> None:
    """A solved class that the plan moves to Q8_0 needs the folded reference, or --promote.

    An old folds.npz holds no plan, thus the plan guard does not see the
    change. Without this check the export would take the unfolded weight
    of the F16 GGUF and the file would be wrong.
    """
    gguf = gguf_module()
    gen = np.random.default_rng(11)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    tensors = write_toy_f16(gguf, src, vocab, d, gen, layer=True)
    packs = tmp_path / "packs"
    packs.mkdir()
    solved_pack(packs / "blk.0.attn_gate.weight.npz", tensors["blk.0.attn_gate.weight"])
    np.savez(packs / "folds.npz", **{"blk.0.attn_norm.weight": tensors["blk.0.attn_norm.weight"]})
    with pytest.raises(ValueError, match="the source is not folded"):
        export(src, tmp_path / "bad.gguf", packs, Plan(gdn_gate="Q8_0", n_layers=1), LLAMA, torch.device("cpu"))
    # The same plan on the folded reference is correct, and so is the pack on this source.
    export(src, tmp_path / "folded.gguf", packs, Plan(gdn_gate="Q8_0", n_layers=1), LLAMA, torch.device("cpu"),
           source_folded=True)
    export(src, tmp_path / "packed.gguf", packs, Plan(n_layers=1), LLAMA, torch.device("cpu"))
    assert read_all(gguf, tmp_path / "folded.gguf")["blk.0.attn_gate.weight"][0] == "Q8_0"
    assert read_all(gguf, tmp_path / "packed.gguf")["blk.0.attn_gate.weight"][0] == "Q4_0"


def test_promote_writes_the_folded_weight_of_an_unfolded_source(tmp_path: Path) -> None:
    """--promote takes the column scales out of the norm and writes Q8_0 of the folded weight.

    The calibration scaled the columns of the mixer inputs by t and put t
    into the input norm. Thus the export must write Q8_0 of source · t,
    and not Q8_0 of the source.
    """
    gguf = gguf_module()
    gen = np.random.default_rng(12)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    tensors = write_toy_f16(gguf, src, vocab, d, gen, layer=True)
    t = np.exp(0.3 * gen.standard_normal(d)).astype(np.float32)
    t = t / np.exp(np.log(t).mean())
    folded_gate = tensors["blk.0.attn_gate.weight"].astype(np.float32) * t[None, :]
    packs = tmp_path / "packs"
    packs.mkdir()
    solved_pack(packs / "blk.0.attn_gate.weight.npz", folded_gate)
    np.savez(packs / "folds.npz",
             **{"blk.0.attn_norm.weight": (tensors["blk.0.attn_norm.weight"] / t).astype(np.float32)})
    out = tmp_path / "promoted.gguf"
    export(src, out, packs, Plan(n_layers=1), LLAMA, torch.device("cpu"), promote=r"attn_gate\.weight")
    got = read_all(gguf, out)
    assert got["blk.0.attn_gate.weight"][0] == "Q8_0"
    packed = np.asarray(got["blk.0.attn_gate.weight"][1].data).reshape(d, -1)
    written = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
    expected = q8_0_dequantize(*q8_0_quantize(torch.from_numpy(folded_gate))).numpy()
    np.testing.assert_allclose(written, expected, rtol=0, atol=1e-6)
    unfolded = q8_0_dequantize(*q8_0_quantize(torch.from_numpy(tensors["blk.0.attn_gate.weight"].astype(np.float32))))
    assert not np.allclose(written, unfolded.numpy(), atol=1e-4), "the fold moved the columns"
    # The class of the plan that stays 4-bit keeps its pack, thus only the promoted class changes.
    assert got["blk.0.ffn_gate.weight"][0] == "Q4_0"


def test_promote_to_f16_keeps_the_folded_weight(tmp_path: Path) -> None:
    """--promote-type F16 writes the folded weight with no quantization."""
    gguf = gguf_module()
    gen = np.random.default_rng(13)
    vocab, d = 48, 64
    src = tmp_path / "src.gguf"
    tensors = write_toy_f16(gguf, src, vocab, d, gen, layer=True)
    packs = tmp_path / "packs"
    packs.mkdir()
    np.savez(packs / "folds.npz", **{"blk.0.attn_norm.weight": tensors["blk.0.attn_norm.weight"]})
    out = tmp_path / "f16.gguf"
    export(src, out, packs, Plan(n_layers=1), LLAMA, torch.device("cpu"), promote=r"attn_gate\.weight",
           promote_type="F16")
    got = read_all(gguf, out)
    assert got["blk.0.attn_gate.weight"][0] == "F16"
    np.testing.assert_array_equal(_f32_of(got["blk.0.attn_gate.weight"][1]),
                                  tensors["blk.0.attn_gate.weight"].astype(np.float32))
    with pytest.raises(ValueError, match="Q8_0 or F16"):
        export(src, tmp_path / "bad.gguf", packs, Plan(n_layers=1), LLAMA, torch.device("cpu"),
               promote=r"attn_gate\.weight", promote_type="Q4_0")
