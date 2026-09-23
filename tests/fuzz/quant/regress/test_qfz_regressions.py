"""The minimal example of each finding of the quant fuzz campaign, as a regression test.

Each test states the correct behavior. While a finding is open, its test
is a strict xfail: the suite passes, and a test that starts to pass fails
the suite (XPASS), thus the marker must go when the fix lands. A run with
QFZ_FIXED=all (or a list of identifiers) treats the findings as fixed, for
a check of the proposed patches in build/fuzz/quant/fixes.

The identifiers, the files and the lines are in qfz_common.KNOWN_FINDINGS
and in the final report of the campaign.
"""

from __future__ import annotations

import argparse
import struct
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import gguf
import quant.run
from qfz_checks import CHECK_BIN, ggml_loader_status, read_tensors
from qfz_common import LLAMA_DIR, fixed
from qfz_rawgguf import array_value, build_file, header, string_value, tensor_info
from qfz_strategies import MatrixSpec
from qfz_toy import SMALL, write_source
from quant.export import export
from quant.grid import block_error, dequantize, pack_nibbles, q8_0_quantize, quantize
from quant.grids import IQ4NLGrid, Q4_0Grid
from quant.plan import Plan

CPU = torch.device("cpu")


def xfail_open(finding: str, reason: str) -> pytest.MarkDecorator:
    """Give the strict xfail marker of an open finding, or no marker when the run treats it as fixed."""
    return pytest.mark.xfail(condition=not fixed(finding), strict=True, reason=f"{finding}: {reason}")


def _plain_source(path: Path, align: int | None = None) -> Path:
    """Write a small tied-free F16 source: token_embd, output_norm, output, with an optional custom alignment."""
    gen = np.random.default_rng(0)
    w = gguf.GGUFWriter(str(path), "qwen35")
    if align is not None:
        w.add_custom_alignment(align)
    w.add_uint32("qwen35.embedding_length", 64)
    w.add_tensor("token_embd.weight", gen.standard_normal((48, 64)).astype(np.float16))
    w.add_tensor("output_norm.weight", (0.5 + gen.random(64)).astype(np.float32))
    w.add_tensor("output.weight", gen.standard_normal((48, 64)).astype(np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    return path


# --- QF1 (task #167): non-finite F16 scales ---------------------------------------------------------

def test_qf1_q8_0_refuses_a_block_beyond_the_scale_range() -> None:
    """q8_0_quantize refuses the block maximum 8.4e6, whose scale is inf in F16 (the block decoded to NaN)."""
    w = torch.zeros(1, 32)
    w[0, 0] = 8.4e6
    with pytest.raises(ValueError):
        q8_0_quantize(w)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_qf1_q8_0_refuses_a_non_finite_input(bad: float) -> None:
    """q8_0_quantize refuses a NaN or an infinite weight, which gave a scale that is not finite."""
    w = torch.zeros(1, 32)
    w[0, 3] = bad
    with pytest.raises(ValueError):
        q8_0_quantize(w)


def test_qf1_q4_0_refuses_a_block_beyond_the_scale_range() -> None:
    """quantize refuses the Q4_0 block maximum 6e5, whose scale was -inf in F16."""
    w = torch.zeros(1, 32)
    w[0, 0] = 6e5
    with pytest.raises(ValueError):
        quantize(Q4_0Grid(), w, search=True)


# --- QF2 (task #167): a zero F16 scale ------------------------------------------------------------

def test_qf2_block_error_of_a_tiny_block_with_a_zero() -> None:
    """block_error of a block whose scale rounds to zero in F16 is the energy of the block.

    0/0 gave NaN, NaN to int32 gave -2^31, and levels[-2^31] raised
    IndexError. The zero scale decodes the block to zero.
    """
    w = torch.zeros(1, 32)
    w[0, 0], w[0, 1] = 1e-7, -5e-8
    got = float(block_error(Q4_0Grid(), w, torch.ones(32)))
    assert got == pytest.approx(float(w.pow(2).sum()), rel=1e-6)


# --- QF3: pack validation ------------------------------------------------------------------------

@xfail_open("QF3", "pack_nibbles packs an index of 17 into the neighbor nibble")
def test_qf3_pack_nibbles_refuses_an_index_out_of_range() -> None:
    """quant/grid.py:165-167: index 17 gives the byte 0x11, thus two wrong nibbles and no error."""
    idx = torch.full((1, 32), 17, dtype=torch.int16)
    with pytest.raises(ValueError):
        pack_nibbles(idx, torch.ones(1, 1, dtype=torch.float16))


@xfail_open("QF3", "the export writes a pack whose shape is not the shape of the source tensor")
def test_qf3_export_refuses_a_pack_of_another_shape(tmp_path: Path) -> None:
    """quant/export.py:339-349: a pack [96, 32] of output.weight [48, 64] becomes a tensor of another shape.

    The element count is the same, thus pack_nibbles does not see it, and
    the file holds output.weight with the shape of the pack.
    """
    src = _plain_source(tmp_path / "src.gguf")
    packs = tmp_path / "packs"
    packs.mkdir()
    grid = Q4_0Grid()
    idx, d = quantize(grid, torch.randn(96, 32, generator=torch.Generator().manual_seed(0)))
    np.savez(packs / "output.weight.npz", q=idx.numpy(), d=d.numpy().view(np.uint16), levels=grid.levels.numpy(),
             kind=np.array("Q4_0"))
    with pytest.raises(ValueError):
        export(src, tmp_path / "out.gguf", packs, Plan(n_layers=0), LLAMA_DIR, CPU)


# --- QF4: general.alignment -----------------------------------------------------------------------

@xfail_open("QF4", "the export copies general.alignment 64, but the writer aligns to 32")
def test_qf4_export_of_a_source_with_a_custom_alignment_loads(tmp_path: Path) -> None:
    """quant/export.py:294-300: the output declares alignment 64 with data at 32; gguf-py and ggml refuse it."""
    src = _plain_source(tmp_path / "src.gguf", align=64)
    out = tmp_path / "out.gguf"
    export(src, out, tmp_path / "no-packs", Plan(n_layers=0, bulk="Q8_0", head="Q8_0"), LLAMA_DIR, CPU)
    got = read_tensors(out)
    assert set(got) == {"token_embd.weight", "output_norm.weight", "output.weight"}
    if CHECK_BIN.exists():
        assert ggml_loader_status(out)[0] == 0, "the ggml loader refuses the export"


# --- QF5: an empty array in the source ------------------------------------------------------------

@xfail_open("QF5", "an empty array field makes the writer raise after the header is on the disk")
def test_qf5_export_of_a_source_with_an_empty_array(tmp_path: Path) -> None:
    """quant/export.py:295-300, 381-382: the export must copy the empty array, or refuse before it writes."""
    gen = np.random.default_rng(1)
    raw = build_file(
        [("general.architecture", string_value("qwen35")), ("qfz.empty", array_value(5, b"", 0))],
        [("token_embd.weight", gen.standard_normal((8, 32)).astype(np.float16), 1),
         ("output_norm.weight", np.ones(32, np.float32), 0),
         ("output.weight", gen.standard_normal((8, 32)).astype(np.float16), 1)])
    src = tmp_path / "src.gguf"
    src.write_bytes(raw)
    assert gguf.GGUFReader(str(src)).fields["qfz.empty"].contents() == []
    out = tmp_path / "out.gguf"
    try:
        export(src, out, tmp_path / "no-packs", Plan(n_layers=0, bulk="Q8_0", head="Q8_0"), LLAMA_DIR, CPU)
    except ValueError:
        assert not out.exists(), f"the failed export left a partial file of {out.stat().st_size} bytes"
        return
    assert gguf.GGUFReader(str(out)).fields["qfz.empty"].contents() == []


# --- QF6: unknown plan types ----------------------------------------------------------------------

@xfail_open("QF6", "Plan(head='Q6_K') writes the head as the F16 source with no error")
def test_qf6_export_refuses_an_unknown_plan_type(tmp_path: Path) -> None:
    """quant/export.py:374-378 (and run.py:355-358 without choices): an unknown type must raise."""
    src = _plain_source(tmp_path / "src.gguf")
    with pytest.raises(ValueError):
        export(src, tmp_path / "out.gguf", tmp_path / "no-packs", Plan(n_layers=0, head="Q6_K"), LLAMA_DIR, CPU)


# --- QF7 (task #169): the llama.cpp path ------------------------------------------------------------

def test_qf7_the_converter_that_run_py_starts_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmd_convert of quant/run.py starts a converter that exists (quant/paths.py gives the llama.cpp tree).

    The test catches the command of cmd_convert and does not start it.
    """
    started: list[list[str]] = []
    monkeypatch.setattr(quant.run.subprocess, "run", lambda cmd, **kw: started.append(cmd))
    quant.run.cmd_convert(argparse.Namespace(model="Qwen3.5-2B", source="t"))
    assert Path(started[0][1]).exists(), f"the converter {started[0][1]} does not exist"


# --- QF8: F16 overflow of the dense maps ------------------------------------------------------------

@xfail_open("QF8", "hnorm_rot / out_norm with an entry 1e-7 overflows F16 to inf with no error")
def test_qf8_export_refuses_a_map_that_overflows_f16(tmp_path: Path) -> None:
    """quant/export.py:372 (and 369 for output_rot): the F16 cast has no finiteness check."""
    import dataclasses

    geo = dataclasses.replace(SMALL, n_layer=1, interval=1, mtp=True)
    src = write_source(tmp_path / "src.gguf", geo, "normal", 0)
    raw = bytearray(src.path.read_bytes())
    reader = gguf.GGUFReader(str(src.path))
    norm = next(t for t in reader.tensors if t.name == "output_norm.weight")
    struct.pack_into("<f", raw, int(norm.data_offset), 1e-7)
    src.path.write_bytes(bytes(raw))
    out = tmp_path / "out.gguf"
    try:
        export(src.path, out, tmp_path / "no-packs", Plan(n_layers=1, bulk="Q8_0", head="Q8_0"), LLAMA_DIR, CPU)
    except ValueError:
        return
    values = read_tensors(out)["blk.1.nextn.hnorm_rot.weight"][1]
    assert np.isfinite(values).all(), f"the export wrote {int((~np.isfinite(values)).sum())} inf values in F16"


# --- QF9: the F16 rounding of the searched scale ------------------------------------------------------

@xfail_open("QF9", "in the F16 subnormal range the searched scale loses to the plain reference scale")
def test_qf9_scale_search_is_not_worse_after_the_f16_rounding() -> None:
    """quant/grid.py:47-56, 78: the search compares float32 candidates, the store rounds the winner to F16."""
    spec = MatrixSpec(3, 6, (("gauss",) * 6, ("spread",) + ("gauss",) * 5, ("gauss",) * 6),
                      ((0,) * 6, (-12, 0, 0, 0, 0, 0), (0,) * 6), seed=264, f16=True)
    w = torch.from_numpy(spec.build())[1:2, :32]
    grid = IQ4NLGrid()
    sse = {s: float((dequantize(grid, *quantize(grid, w, search=s)) - w).double().pow(2).sum()) for s in (True, False)}
    assert sse[True] <= sse[False], f"the search gives {sse[True]:.4e}, the plain scale {sse[False]:.4e}"


# --- QF10: --only and IQ4_NL ----------------------------------------------------------------------------

@xfail_open("QF10", "the filter keeps IQ4_NL tensors out of the match in IQ4_NL")
def test_qf10_only_keeps_every_other_tensor_in_f16(tmp_path: Path) -> None:
    """quant/export.py:318: the docstring promises F16 for each tensor out of the match; IQ4_NL stays IQ4_NL."""
    src = _plain_source(tmp_path / "src.gguf")
    out = tmp_path / "out.gguf"
    export(src, out, tmp_path / "no-packs", Plan(n_layers=0, head="IQ4_NL", embedding="IQ4_NL"), LLAMA_DIR, CPU,
           only="^$")
    got = read_tensors(out)
    assert got["output.weight"][0] == "F16" and got["token_embd.weight"][0] == "F16"


# --- QF11: sys.path growth -----------------------------------------------------------------------------

@xfail_open("QF11", "each call of _load_gguf_module adds one more copy of the same sys.path entry")
def test_qf11_export_does_not_grow_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """quant/export.py:46-50: 100 exports in one process give 100 copies of llama_dir/gguf-py in sys.path."""
    import sys

    from quant.export import _load_gguf_module

    monkeypatch.setattr(sys, "path", list(sys.path))
    _load_gguf_module(LLAMA_DIR)
    before = len(sys.path)
    for _ in range(10):
        _load_gguf_module(LLAMA_DIR)
    assert len(sys.path) == before, f"sys.path grew by {len(sys.path) - before} entries in 10 calls"


# --- QR1, QR2, QR3 (task #171): the gguf-py reader ---------------------------------------------------

def test_qr1_reader_refuses_a_scalar_array_longer_than_the_file(tmp_path: Path) -> None:
    """The reader refuses a scalar array count that the rest of the file cannot hold, in less than 0.5 s.

    Without the check the reader looped over the count: 4 000 000 items took
    18 s and 4.7 GB, and GGUF_MAX_ARRAY_ELEMENTS (2^30) permits about 1.2 TB.
    """
    path = tmp_path / "qr1.gguf"
    path.write_bytes(header(0, [("a", struct.pack("<IIQ", 9, 0, 300_000))]))
    t0 = time.monotonic()
    with pytest.raises(ValueError):
        gguf.GGUFReader(str(path))
    assert time.monotonic() - t0 < 0.5


def test_qr2_reader_refuses_an_offset_that_wraps(tmp_path: Path) -> None:
    """The reader refuses the tensor offset 2^64 - 64, which wrapped in uint64 to the start of the file."""
    raw = header(1, []) + tensor_info("t", [4], 0, 2**64 - 64)
    raw += b"\0" * ((-len(raw)) % 32) + np.arange(4, dtype=np.float32).tobytes()
    path = tmp_path / "qr2.gguf"
    path.write_bytes(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            gguf.GGUFReader(str(path))


def test_qr2_reader_refuses_a_tensor_past_the_end(tmp_path: Path) -> None:
    """The reader refuses a tensor of 0 bytes at the offset 64 of the data section of an 80-byte file."""
    raw = header(1, []) + tensor_info("t", [0], 0, 64)
    raw += b"\0" * ((-len(raw)) % 32) + bytes(16)
    path = tmp_path / "qr2-past-end.gguf"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        gguf.GGUFReader(str(path))


def test_qr3_reader_gives_a_clear_error_for_a_truncated_file(tmp_path: Path) -> None:
    """The reader gives ValueError for a file that stops before the type of its first KV field."""
    path = tmp_path / "qr3.gguf"
    path.write_bytes(header(0, [("a", b"")]))
    with pytest.raises(ValueError):
        gguf.GGUFReader(str(path))


def test_qr4_reader_gives_a_clear_error_for_a_block_tensor_with_no_dimension(tmp_path: Path) -> None:
    """The reader gives ValueError for a Q4_0 tensor with no dimension (one element, not a full block).

    Without the check, quants.quant_shape_to_byte_shape read the last
    dimension of an empty shape and raised IndexError. The atheris reader
    target found it. regress/reader/qr4-zero-dim-block-type.seed holds this file.
    """
    raw = header(1, []) + tensor_info("t", [], 2, 0)
    raw += b"\0" * ((-len(raw)) % 32) + bytes(32)
    path = tmp_path / "qr4.gguf"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        gguf.GGUFReader(str(path))
