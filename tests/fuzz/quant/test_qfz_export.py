"""Fuzz the export of quant/export.py end to end: toy sources, plans, packs, and the written GGUF.

A drawn toy source (a geometry, a value profile, a seed, a tie, an MTP
block, extra metadata) goes through export() with a drawn plan, an
optional tensor filter, and optional solved packs with low-rank factors.

The properties of the written file:

- gguf-py reads it, and the tensor set and each tensor type follow the plan.
- Each value is what the plan promises: the F32 and the kept tensors are
  the source, the round-to-nearest tensors are quantize() of the source,
  the packs are the packed blocks in the value-head order of the
  converter, the MTP maps are scaled by the exported output norm.
- Each metadata field of the source comes through with its type and value.
- The ggml loader, in the one sanitizer build of the run, loads the file and
  decodes each tensor to the same float32 values as gguf-py, bit for bit.
- A small sample of the files runs in llama-perplexity of the build without
  a sanitizer (TOY_BIN), and the logits are finite.

The expected type of a filtered IQ4_NL tensor follows the known defect
only-filter-skips-iq4-nl while it is in qfz_common.KNOWN_DEFECTS. The
regression tests hold the export of a source with an empty array and of a
plan with an unknown type name, which need a source from raw bytes or a
type name out of the supported set.
"""

from __future__ import annotations

import dataclasses
import io
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

import gguf
from qfz_checks import check_available, ggml_loader_check, kl_statistics, read_tensors, run_perplexity
from qfz_common import LLAMA_DIR, known_open, san_dir, scratch
from qfz_hyp import counted, fuzz_settings
from qfz_toy import PROFILES, SMALL, Geometry, make_text, output_rot_for, write_source
from quant.export import FILE_TYPES, GGUF_4BIT, LinearAttentionLayout, export, mtp_map
from quant.grid import dequantize, dequantize_pack, q8_0_dequantize, q8_0_quantize, quantize
from quant.grids import make_grid
from quant.plan import MTP_MAPS, Plan

CPU = torch.device("cpu")
TYPES_4 = ("Q4_0", "IQ4_NL")
TYPES = TYPES_4 + ("Q8_0",)
FILTERS = (None, r"ffn_", r"attn_(qkv|gate)", r"^token_embd", r"output")
SKIPPED_KEYS = {"general.architecture", "general.file_type", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
# The build without a sanitizer of the profile of this run (run.sh build none). run.sh builds it from the
# submodule with the patch series, thus it has the fixes of patches/.
TOY_BIN = san_dir("none") / "llama" / "bin"


@st.composite
def geometries(draw: st.DrawFn) -> Geometry:
    """Draw a small valid geometry: the widths, the layers, the GDN heads, the vocabulary, the MTP block."""
    k_heads, per_k = draw(st.sampled_from([(1, 1), (1, 2), (2, 2)]))
    return dataclasses.replace(
        SMALL,
        n_embd=32 * draw(st.integers(1, 3)),
        n_ff=32 * draw(st.integers(2, 4)),
        n_layer=draw(st.integers(1, 3)),
        interval=draw(st.integers(1, 3)),
        ssm_k_heads=k_heads,
        ssm_v_heads=k_heads * per_k,
        n_vocab=draw(st.integers(265, 300)),
        mtp=draw(st.booleans()),
    )


@st.composite
def plans(draw: st.DrawFn, n_layers: int) -> Plan:
    """Draw a plan with the supported type names of each class."""
    bulk = draw(st.sampled_from(TYPES))
    return Plan(
        bulk=bulk,
        head=draw(st.sampled_from(TYPES + ("F16",))),
        embedding=draw(st.sampled_from(TYPES + ("F16",))),
        kv_proj=draw(st.sampled_from(TYPES)),
        gdn_gate=draw(st.sampled_from(TYPES)),
        ssm_out=draw(st.sampled_from((None,) + TYPES)),
        ffn_down=draw(st.sampled_from((None,) + TYPES)),
        edge_layers=tuple(draw(st.lists(st.integers(0, max(0, n_layers - 1)), max_size=2, unique=True))),
        n_layers=n_layers,
        mtp=draw(st.sampled_from(("F16", "Q8_0", "Q4_0", "IQ4_NL"))),
    )


SCALARS = {
    gguf.GGUFValueType.UINT8: st.integers(0, 255), gguf.GGUFValueType.INT8: st.integers(-128, 127),
    gguf.GGUFValueType.UINT16: st.integers(0, 65535), gguf.GGUFValueType.INT16: st.integers(-32768, 32767),
    gguf.GGUFValueType.UINT32: st.integers(0, 2**32 - 1), gguf.GGUFValueType.INT32: st.integers(-2**31, 2**31 - 1),
    gguf.GGUFValueType.UINT64: st.integers(0, 2**64 - 1), gguf.GGUFValueType.INT64: st.integers(-2**63, 2**63 - 1),
    gguf.GGUFValueType.FLOAT32: st.floats(width=32, allow_nan=False),
    gguf.GGUFValueType.FLOAT64: st.floats(allow_nan=False),
    gguf.GGUFValueType.BOOL: st.booleans(), gguf.GGUFValueType.STRING: st.text(max_size=12),
}


@st.composite
def metadata(draw: st.DrawFn) -> dict[str, tuple[object, gguf.GGUFValueType, gguf.GGUFValueType | None]]:
    """Draw extra metadata fields of every scalar type, strings, and arrays of them."""
    out: dict[str, tuple[object, gguf.GGUFValueType, gguf.GGUFValueType | None]] = {}
    for i in range(draw(st.integers(0, 5))):
        vtype = draw(st.sampled_from(sorted(SCALARS, key=int)))
        key = f"qfz.extra.{i}"
        if draw(st.booleans()):
            # gguf-py cannot write an empty array, thus a regression test builds that source from raw bytes.
            values = draw(st.lists(SCALARS[vtype], min_size=1, max_size=4))
            out[key] = (values, gguf.GGUFValueType.ARRAY, vtype)
        else:
            out[key] = (draw(SCALARS[vtype]), vtype, None)
    if draw(st.booleans()):
        out["general.alignment"] = (draw(st.sampled_from([64, 128, 256])), gguf.GGUFValueType.UINT32, None)
    return out


def _solved_pack(path: Path, rows: int, cols: int, kind: str, gen: np.random.Generator, rank: int) -> dict:
    """Write a random solved pack of a 4-bit type, with the grid levels and the kind, as the solver writes one."""
    grid = make_grid(kind)
    fields = {"q": gen.integers(0, 16, (rows, cols)).astype(np.int8),
              "d": (gen.uniform(-0.05, 0.05, (rows, cols // 32))).astype(np.float16).view(np.uint16),
              "levels": grid.levels.numpy(), "kind": np.array(kind)}
    if rank:
        fields["lora_a"] = gen.standard_normal((rank, cols)).astype(np.float32)
        fields["lora_b"] = gen.standard_normal((rows, rank)).astype(np.float32)
    np.savez(path, **fields)
    return fields


def expected_kind(plan: Plan, name: str, only: str | None, invert: bool) -> str:
    """Give the type that the export must write for a tensor, before the source type of a kept tensor.

    The docstring of export() says that the filter keeps every tensor out
    of the match in F16. The code applies the filter to Q4_0 and Q8_0 only,
    thus an IQ4_NL tensor out of the match stays IQ4_NL: the known defect
    only-filter-skips-iq4-nl. The expectation follows the code while that
    defect is in KNOWN_DEFECTS.
    """
    kind = plan.type_of(name)
    filtered = ("Q4_0", "Q8_0") if known_open("only-filter-skips-iq4-nl") else ("Q4_0", "IQ4_NL", "Q8_0")
    if only is not None and kind in filtered and (re.search(only, name) is None) != invert:
        kind = "keep"
    return kind


def _pack_values(fields: dict, rows: torch.Tensor | None, cols: torch.Tensor | None) -> np.ndarray:
    """Give the float32 values of a pack without its low-rank term, in the GGUF order."""
    buffer = io.BytesIO()
    np.savez(buffer, **{k: v for k, v in fields.items() if not k.startswith("lora")})
    buffer.seek(0)
    return dequantize_pack(np.load(buffer), CPU, rows, cols).numpy()


@dataclasses.dataclass(frozen=True)
class ExportCase:
    """One drawn export: the source, the plan, the filter, the metadata and the packs.

    Attributes:
        geo: The geometry of the toy source
        plan: The export plan
        profile: The value profile of the source
        seed: The seed of the values and of the pack choice
        tie: True for a tied head with output_rot
        filt: The --only expression, or None
        invert: The --invert switch
        extra: More metadata of the source
        pack_rate: The fraction of the 4-bit tensors that get a solved pack
        rank: The largest rank of the low-rank factors of a pack
    """

    geo: Geometry
    plan: Plan
    profile: str
    seed: int
    tie: bool
    filt: str | None
    invert: bool
    extra: dict
    pack_rate: float
    rank: int


@st.composite
def export_cases(draw: st.DrawFn) -> ExportCase:
    """Draw an export case: a geometry, a plan for its layers, and the other choices."""
    geo = draw(geometries())
    return ExportCase(geo, draw(plans(geo.n_layer)), draw(st.sampled_from(PROFILES)), draw(st.integers(0, 2**31 - 1)),
                      draw(st.booleans()), draw(st.sampled_from(FILTERS)), draw(st.booleans()), draw(metadata()),
                      draw(st.sampled_from([0.0, 0.5, 1.0])), draw(st.integers(0, 2)))


CASE_PACKED = ExportCase(dataclasses.replace(SMALL, mtp=True), Plan(n_layers=2, bulk="Q4_0", mtp="Q8_0"), "normal", 0,
                         False, None, False, {}, 1.0, 1)
CASE_FILTERED = ExportCase(SMALL, Plan(n_layers=2, bulk="IQ4_NL", head="Q8_0", embedding="F16", ffn_down="Q8_0"),
                           "large", 1, True, r"ffn_", True,
                           {"qfz.extra.0": ([1, 2], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)}, 0.5, 2)
CASE_ALIGNMENT = ExportCase(SMALL, Plan(n_layers=2, bulk="Q8_0"), "mixed", 2, False, None, False,
                      {"general.alignment": (64, gguf.GGUFValueType.UINT32, None)}, 0.0, 0)
CASE_ONLY_IQ4_NL = ExportCase(SMALL, Plan(n_layers=2, bulk="IQ4_NL"), "tiny", 3, False, r"^token_embd", False, {}, 0.0, 0)


@fuzz_settings(0.5)
@given(case=export_cases())
@example(case=CASE_PACKED)
@example(case=CASE_FILTERED)
@example(case=CASE_ALIGNMENT)
@example(case=CASE_ONLY_IQ4_NL)
@counted
def test_export_writes_what_the_plan_promises(case: ExportCase) -> None:
    """The export of a drawn source and plan gives the promised types, values, metadata, and a loadable file."""
    with scratch() as tmp_path:
        _check_export(tmp_path, case)


def _check_export(tmp_path: Path, case: ExportCase) -> None:
    """Write a source, export it with the plan and the packs of the case, and check the file (module docstring)."""
    geo, plan, seed, tie, filt, invert = case.geo, case.plan, case.seed, case.tie, case.filt, case.invert
    src = write_source(tmp_path / "src.gguf", geo, case.profile, seed, tied=tie, extra_kv=case.extra)
    gen = np.random.default_rng(seed)
    packs = tmp_path / "packs"
    packs.mkdir()
    layout = LinearAttentionLayout(geo.ssm_k_heads, geo.ssm_v_heads, geo.ssm_state, geo.ssm_state)
    packed: dict[str, dict] = {}
    for name, a in src.tensors.items():
        kind = plan.type_of(name)
        if kind in GGUF_4BIT and a.ndim == 2 and gen.random() < case.pack_rate:
            rank = int(gen.integers(0, case.rank + 1))
            packed[name] = _solved_pack(packs / f"{name}.npz", a.shape[0], a.shape[1], kind, gen, rank)
    rot_path = None
    if tie:
        rot_path = tmp_path / "rot.npy"
        np.save(rot_path, output_rot_for(geo, seed))
    out = tmp_path / "out.gguf"
    export(src.path, out, packs, plan, LLAMA_DIR, CPU, only=filt, invert=invert, tie_head=tie, rot=rot_path)

    got = read_tensors(out)
    expected_names = set(src.tensors) | ({"output_rot.weight"} if tie else set())
    assert set(got) == expected_names, f"the tensor set differs: {sorted(set(got) ^ expected_names)}"
    out_norm = np.ones(geo.n_embd, np.float32) if tie else src.tensors["output_norm.weight"]
    for name, a in src.tensors.items():
        kind = expected_kind(plan, name, filt, invert)
        wrote, values = got[name]
        source = a.astype(np.float32)
        if name in packed and kind in GGUF_4BIT:
            assert wrote == kind, f"{name}: the pack of {kind} came out as {wrote}"
            ref = _pack_values(packed[name], layout.rows(name), layout.cols(name))
            np.testing.assert_array_equal(values, ref, err_msg=f"{name}: the pack is not in the converter order")
        elif kind in GGUF_4BIT:
            assert wrote == kind, f"{name}: {kind} came out as {wrote}"
            grid = make_grid(kind)
            np.testing.assert_array_equal(values, dequantize(grid, *quantize(grid, torch.from_numpy(source))).numpy())
        elif kind == "Q8_0":
            assert wrote == "Q8_0", f"{name}: Q8_0 came out as {wrote}"
            np.testing.assert_array_equal(values, q8_0_dequantize(*q8_0_quantize(torch.from_numpy(source))).numpy())
        elif kind == "F32":
            assert wrote == "F32", f"{name}: F32 came out as {wrote}"
            np.testing.assert_array_equal(values, out_norm if name == "output_norm.weight" else source)
        elif name.endswith(MTP_MAPS):
            assert wrote == "F16", f"{name}: the MTP map came out as {wrote}"
            want = mtp_map(name, source, out_norm).astype(np.float16).astype(np.float32)
            np.testing.assert_array_equal(values, want)
        else:
            assert wrote == ("F16" if a.dtype == np.float16 else "F32"), f"{name}: a kept tensor came out as {wrote}"
            np.testing.assert_array_equal(values, source)
    if tie:
        want_rot = np.load(rot_path).astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(got["output_rot.weight"][1], want_rot)

    reader_in, reader_out = gguf.GGUFReader(str(src.path)), gguf.GGUFReader(str(out))
    for key, field in reader_in.fields.items():
        if key in SKIPPED_KEYS:
            continue
        assert key in reader_out.fields, f"the export dropped the metadata field {key}"
        assert reader_out.fields[key].types == field.types, f"{key}: the type changed"
        assert reader_out.fields[key].contents() == field.contents(), f"{key}: the value changed"
    file_type = reader_out.fields["general.file_type"].contents()
    assert file_type == int(getattr(gguf.LlamaFileType, FILE_TYPES.get(plan.bulk, "MOSTLY_Q4_0")))
    adapter = out.with_name(out.stem + "-lora.gguf")
    used = [z for name, z in packed.items() if expected_kind(plan, name, filt, invert) in GGUF_4BIT]
    assert adapter.exists() == any("lora_a" in z for z in used), "the adapter file does not follow the packs"

    if check_available():
        loaded = ggml_loader_check(out)
        for name, (_, values) in got.items():
            np.testing.assert_array_equal(loaded[name], values.reshape(-1),
                                          err_msg=f"{name}: the ggml loader decodes other values than gguf-py")


@fuzz_settings(0.05)
@given(profile=st.sampled_from(PROFILES), seed=st.integers(0, 2**31 - 1), kind=st.sampled_from(TYPES),
       tie=st.booleans())
@example(profile="normal", seed=0, kind="Q4_0", tie=False)
@counted
def test_exported_toy_model_runs_in_llama_cpp(profile: str, seed: int, kind: str, tie: bool) -> None:
    """The build of llama.cpp without a sanitizer loads an exported toy model and computes finite logits on a text."""
    if not (TOY_BIN / "llama-perplexity").exists():
        pytest.skip(f"{TOY_BIN}/llama-perplexity is missing: run tests/fuzz/quant/run.sh build none")
    with scratch() as tmp_path:
        _run_toy(tmp_path, profile, seed, kind, tie)


def _run_toy(tmp_path: Path, profile: str, seed: int, kind: str, tie: bool) -> None:
    """Export a SMALL toy model, run it two times in TOY_BIN, and check the KL of the two runs."""
    src = write_source(tmp_path / "src.gguf", SMALL, profile, seed, tied=tie)
    plan = Plan(bulk=kind, head=kind, embedding=kind, kv_proj=kind, gdn_gate=kind, n_layers=SMALL.n_layer)
    rot = None
    if tie:
        rot = tmp_path / "rot.npy"
        np.save(rot, output_rot_for(SMALL, seed))
    out = tmp_path / "model.gguf"
    export(src.path, out, tmp_path / "no-packs", plan, LLAMA_DIR, CPU, tie_head=tie, rot=rot)
    text = tmp_path / "text.txt"
    text.write_text(make_text(900, seed))
    status, log = run_perplexity(TOY_BIN, out, text, tmp_path / "base.kld", write_base=True, threads=2)
    assert status == 0, f"llama-perplexity failed with {status}:\n{log[-3000:]}"
    status, log = run_perplexity(TOY_BIN, out, text, tmp_path / "base.kld", write_base=False, threads=2)
    assert status == 0, f"llama-perplexity with the KL base failed with {status}:\n{log[-3000:]}"
    kld = kl_statistics(log)
    assert np.isfinite(kld["mean"]), f"the KL of a run against itself is not finite: {kld}\n{log[-4000:]}"
    assert kld["mean"] <= 1e-3, f"one build gives a different result on a second run: {kld}"
