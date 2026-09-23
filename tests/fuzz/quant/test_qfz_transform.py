"""Fuzz the rotations and the function-preserving transform of quant/transform.py.

The properties:

- hadamard(n) is orthogonal for n = 2^m · r with r odd.
- rotation_matrix is orthogonal for each block that divides n, and it
  refuses a block that does not divide n.
- The transform keeps the logits of the toy model of quant/tests, for
  drawn widths, layer sequences, blocks, seeds, norm magnitudes, the tied
  head, the MLP permutation, and the MTP block with its chained drafts.
- The MTP transform refuses a final norm with an entry near zero.

The toy model comes from quant/tests/test_transform.py, which this module
reads and does not change.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import assume, example, given
from hypothesis import strategies as st

from qfz_hyp import counted, fuzz_settings
from quant.checkpoint import LM, MTP
from quant.tests.test_transform import toy_forward, toy_mtp_forward, toy_tensors
from quant.transform import GAMMA_MIN, hadamard, rotation_matrix, transform

CPU = torch.device("cpu")
LAYER_TYPES = st.lists(st.sampled_from(["linear_attention", "full_attention"]), min_size=1, max_size=3)


def _divisors(n: int) -> list[int]:
    """Give the divisors of n. Complexity is O(n)."""
    return [b for b in range(1, n + 1) if n % b == 0]


@st.composite
def widths(draw: st.DrawFn, max_pow: int = 5, max_odd: int = 7, min_pow: int = 1) -> int:
    """Draw an even width 2^m · r with m >= min_pow and r odd."""
    m = draw(st.integers(min_pow, max_pow))
    r = draw(st.sampled_from([x for x in range(1, max_odd + 1, 2)]))
    return (1 << m) * r


@st.composite
def width_blocks(draw: st.DrawFn, max_pow: int = 5, max_odd: int = 7, extra: bool = False,
                 min_block: int = 1, min_pow: int = 1) -> tuple[int, int | None]:
    """Draw a width and a rotation block: None, a divisor of the width, or (with extra) the width plus one."""
    n = draw(widths(max_pow, max_odd, min_pow))
    choices: list[int | None] = [None, *[b for b in _divisors(n) if b >= min_block]]
    if extra:
        choices.append(n + 1)
    return n, draw(st.sampled_from(choices))


@st.composite
def non_divisors(draw: st.DrawFn) -> tuple[int, int]:
    """Draw a width and a block less than the width that does not divide it."""
    n = draw(widths(max_pow=6))
    assume(n > 3)
    block = draw(st.integers(2, n - 1).filter(lambda b: n % b != 0))
    return n, block


@fuzz_settings()
@given(m=st.integers(0, 8), r=st.sampled_from([1, 3, 5, 7, 9, 15]))
@example(m=9, r=5)
@counted
def test_hadamard_is_orthogonal(m: int, r: int) -> None:
    """The Sylvester matrix and its Kronecker product with the seeded odd factor are orthogonal."""
    n = (1 << m) * r
    assume(n <= 2560)
    h = hadamard(n, CPU)
    torch.testing.assert_close(h @ h.T, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=0)


@fuzz_settings()
@given(nb=width_blocks(max_pow=7, extra=True), seed=st.integers(0, 2**31 - 1))
@example(nb=(40, 8), seed=0)
@example(nb=(96, None), seed=3)
@counted
def test_rotation_matrix_is_orthogonal_for_each_block(nb: tuple[int, int | None], seed: int) -> None:
    """Q = H_block · D is orthogonal for each block that divides n, and a larger block is the full Hadamard."""
    n, block = nb
    q = rotation_matrix(n, block, seed, CPU)
    torch.testing.assert_close(q @ q.T, torch.eye(n, dtype=torch.float64), atol=1e-12, rtol=0)


@fuzz_settings()
@given(nb=non_divisors())
@example(nb=(64, 24))
@counted
def test_rotation_matrix_refuses_a_block_that_does_not_divide(nb: tuple[int, int]) -> None:
    """A block less than n that does not divide n raises ValueError."""
    n, block = nb
    with pytest.raises(ValueError, match="does not divide"):
        rotation_matrix(n, block, 0, CPU)


def _scale_norms(tensors: dict[str, torch.Tensor], factor: float, near_zero: bool, gen: torch.Generator) -> None:
    """Scale the zero-centered norm weights, and put some effective gammas near zero when asked."""
    for name, t in tensors.items():
        gated = name.endswith("linear_attn.norm.weight")
        if name.endswith("layernorm.weight") or (name.endswith("norm.weight") and not gated):
            t.mul_(factor)
            if near_zero and name.startswith(f"{LM}layers."):
                where = torch.rand(t.shape, generator=gen) < 0.1
                t[where] = -1.0 + 1e-6


@fuzz_settings(0.4)
@given(db=width_blocks(max_pow=4, max_odd=5, min_block=2), types=LAYER_TYPES, tie=st.booleans(),
       permute=st.booleans(), rotate=st.booleans(), seed=st.integers(0, 2**31 - 1),
       factor=st.sampled_from([0.0, 0.3, 1.0, 3.0]), near_zero=st.booleans())
@example(db=(40, 8), types=["linear_attention", "full_attention"], tie=False, permute=True, rotate=True, seed=0,
         factor=0.3, near_zero=False)
@example(db=(24, None), types=["full_attention"], tie=True, permute=False, rotate=True, seed=1, factor=3.0,
         near_zero=True)
@counted
def test_transform_keeps_the_logits(db: tuple[int, int | None], types: list[str], tie: bool, permute: bool,
                                   rotate: bool, seed: int, factor: float, near_zero: bool) -> None:
    """The fold, the rotation, the permutation and the tied head keep the logits of the toy model."""
    d, block = db
    gen = torch.Generator().manual_seed(seed)
    tensors = toy_tensors(d, vocab=40, layer_types=types, untied=not tie, merger=not tie, gen=gen)
    _scale_norms(tensors, factor, near_zero, gen)
    ids = torch.randint(0, 40, (7,), generator=gen)
    image = None if tie else torch.randn(2, 2 * d, generator=gen)
    before = toy_forward(tensors, types, ids, image)
    out = transform(tensors, len(types), types, rotate=rotate, block=block, seed=seed % 1000, permute_mlp=permute,
                    device=CPU, tie_head=tie)
    after = toy_forward(out, types, ids, image)
    scale = before.abs().max().item()
    torch.testing.assert_close(after, before, atol=2e-4 * scale + 1e-12, rtol=0)


@fuzz_settings(0.3)
@given(db=width_blocks(max_pow=4, max_odd=3, min_block=2, min_pow=3), tie=st.booleans(),
       seed=st.integers(0, 2**31 - 1), factor=st.sampled_from([0.1, 0.3, 1.0]))
@example(db=(32, None), tie=True, seed=0, factor=0.3)
@example(db=(48, 16), tie=False, seed=2, factor=1.0)
@counted
def test_mtp_transform_keeps_the_draft_logits(db: tuple[int, int | None], tie: bool, seed: int, factor: float) -> None:
    """Three chained draft steps of the MTP block keep their logits after the transform.

    The width is 8 or more. At the width 2 the chained RMSNorm of three
    steps turns a float32 rounding of 1e-7 into 1e-3 in the logits: the
    same 1e-7 noise on the original weights gives 5e-4 to 1.1e-3 (measured
    on 2026-09-23), thus the transform is not the cause there.
    """
    d, block = db
    gen = torch.Generator().manual_seed(seed)
    types = ["linear_attention", "full_attention"]
    tensors = toy_tensors(d, vocab=40, layer_types=types, untied=False, merger=False, gen=gen, mtp=True)
    _scale_norms(tensors, factor, False, gen)
    gamma_f = 1.0 + tensors[LM + "norm.weight"]
    assume(float(gamma_f.abs().min()) >= GAMMA_MIN)
    ids = torch.randint(0, 40, (5,), generator=gen)
    drafts = [torch.randint(0, 40, (5,), generator=gen) for _ in range(3)]
    before = toy_mtp_forward(tensors, types, ids, drafts)
    out = transform(tensors, 2, types, rotate=True, block=block, seed=seed % 1000, permute_mlp=True, device=CPU,
                    tie_head=tie)
    after = toy_mtp_forward(out, types, ids, drafts)
    # The head map divides by the final gamma, thus its conditioning sets the tolerance.
    cond = float((1.0 + tensors[MTP + "norm.weight"]).abs().max() / gamma_f.abs().min())
    for step, (a, b) in enumerate(zip(after, before, strict=True)):
        torch.testing.assert_close(a, b, atol=2e-4 * max(1.0, cond) * b.abs().max().item() + 1e-12, rtol=0,
                                   msg=f"draft step {step}")


@fuzz_settings()
@given(d=widths(max_pow=3, max_odd=3), index=st.integers(0, 1000),
       gamma=st.floats(-GAMMA_MIN * 0.999, GAMMA_MIN * 0.999))
@example(d=32, index=3, gamma=0.0)
@counted
def test_mtp_transform_refuses_a_final_gamma_near_zero(d: int, index: int, gamma: float) -> None:
    """An effective final gamma with a magnitude less than GAMMA_MIN raises ValueError."""
    gen = torch.Generator().manual_seed(index)
    types = ["full_attention"]
    tensors = toy_tensors(d, vocab=16, layer_types=types, untied=False, merger=False, gen=gen, mtp=True)
    tensors[LM + "norm.weight"][index % d] = gamma - 1.0
    with pytest.raises(ValueError, match="final norm weight"):
        transform(tensors, 1, types, rotate=True, block=None, seed=0, permute_mlp=False, device=CPU)
