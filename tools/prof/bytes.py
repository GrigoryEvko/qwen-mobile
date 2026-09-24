#!/usr/bin/env python3
"""The per-token byte budget of a decode step, and the economics of a speculative step.

Decode on this phone is memory-bandwidth bound. At a fixed weight format the only
levers are the number of bytes one step reads and the number of tokens one step
yields. This module computes both, thus a change can be ranked before anyone
writes a kernel.

The model is analytic and takes the hyperparameters of a model, because the GGUF
files live on the phone and not in this repository. Pass ``--gguf`` to refine the
weight bytes from a real file when one is available.

Usage:
    tools/prof/bytes.py budget 4b
    tools/prof/bytes.py budget 2b --mtp Q8_0
    tools/prof/bytes.py spec 4b --draft 3 --accept 1.0
    tools/prof/bytes.py spec 4b --draft 3 --accept 0.65 --draft-head 3072 --mtp Q8_0 --hmx-verify
    tools/prof/bytes.py plan 4b

Every rate this module prints is a ceiling from bytes alone. It ignores host
overhead, the fixed cost per call and thermal throttling, thus a measured rate is
always lower. Use it to rank changes, never to predict a number.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace

# Bits per weight of each GGUF type we ship. Q8_0 packs 32 int8 quants and one
# f16 scale into 34 bytes, thus 8.5 bits and not 8.0.
BPW: dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q8_64": 8.25,   # a 64-weight block with one f16 scale, not yet a real type
    "Q8_128": 8.125,  # a 128-weight block with one f16 scale, not yet a real type
    "Q8_ROW": 8.0,   # one f32 scale per row, not yet a real type
    "Q4_0": 4.5,
}

# The measured weight-stream rate of the NPU at thermal 0, in bytes per second.
# Source: our own runs, 51 to 56 GB/s. The low end is the honest planning figure.
STREAM_BPS = 51.0e9


@dataclass(frozen=True)
class Model:
    """The hyperparameters that decide the weight bytes of one decode step.

    Attributes:
        name: The name used in the output
        n_embd: The hidden size
        n_ff: The intermediate size of one MLP
        n_layer: The number of decoder layers, without the MTP block
        n_gdn: The number of gated-delta-net layers
        vocab: The number of rows of the output head
        n_head_kv: The number of key and value heads of a full-attention layer
        head_dim: The dimension of one attention head
        tied_head: True when the head shares its weights with the embedding
    """

    name: str
    n_embd: int
    n_ff: int
    n_layer: int
    n_gdn: int
    vocab: int
    n_head_kv: int
    head_dim: int
    n_head: int
    d_state: int
    n_v_heads: int
    n_k_heads: int
    tied_head: bool = True

    @property
    def n_gqa(self) -> int:
        """The number of full-attention layers."""
        return self.n_layer - self.n_gdn

    @property
    def key_dim(self) -> int:
        """The width of the q and the k projection of a gated-delta-net layer."""
        return self.d_state * self.n_k_heads

    @property
    def value_dim(self) -> int:
        """The width of the v projection and of the gate of a gated-delta-net layer."""
        return self.d_state * self.n_v_heads

    @property
    def conv_dim(self) -> int:
        """The width of the fused qkv projection of a gated-delta-net layer."""
        return 2 * self.key_dim + self.value_dim

    @property
    def state_bytes(self) -> int:
        """The bytes of the recurrent state of one gated-delta-net layer, at f32."""
        return self.d_state * self.d_state * self.n_v_heads * 4


# The shapes come from the metadata and the tensors of the GGUF files (weights/gguf/Qwen3.5-4B-Q8_0.gguf
# and Qwen3.5-2B-Q8_0.gguf): qwen35.attention.head_count, head_count_kv, key_length,
# ssm.state_size, ssm.group_count and ssm.time_step_rank. The tensors of the 4B:
#   ffn_gate/up/down  2560 x 9216     thus n_ff is 9216 and not 9728
#   attn_qkv          2560 x 8192     thus conv_dim 8192 = 2*key_dim + value_dim
#   attn_gate         2560 x 4096     thus value_dim 4096 = d_state 128 x 32 heads
#   ssm_out           4096 x 2560     thus key_dim 2048 = d_state 128 x 16 heads
#   attn_q            2560 x 8192     16 heads of 256, doubled: the attention is gated
#   attn_k, attn_v    2560 x 1024     4 KV heads of 256
#   token_embd        2560 x 248320   the head, tied
# The byte counts of a 4B decode token agree with a device profile
# (tools/prof/store/20260919T213512Z-stalls-p1.log): the sum is 4468 MB against 4482 MB
# measured, and with the F16 multi-token-prediction block of 241 MB it gives 4710 MB against
# the 4.40 GiB file. The 2B has 8 heads and 2 KV heads of 256, and 16 value heads.
MODELS: dict[str, Model] = {
    "4b": Model("Qwen3.5-4B", n_embd=2560, n_ff=9216, n_layer=32, n_gdn=24,
                vocab=248320, n_head_kv=4, head_dim=256, n_head=16,
                d_state=128, n_v_heads=32, n_k_heads=16),
    "2b": Model("Qwen3.5-2B", n_embd=2048, n_ff=6144, n_layer=24, n_gdn=18,
                vocab=248320, n_head_kv=2, head_dim=256, n_head=8,
                d_state=128, n_v_heads=16, n_k_heads=16),
}


@dataclass
class Plan:
    """The weight type of each tensor class.

    Attributes:
        head: The output head and the tied embedding
        mlp: The gate, up and down matrices of every decoder layer
        gdn: The projections of a gated-delta-net layer
        attn: The projections of a full-attention layer
        mtp: The matrices of the multi-token-prediction block
    """

    head: str = "Q8_0"
    mlp: str = "Q8_0"
    gdn: str = "Q8_0"
    attn: str = "Q8_0"
    mtp: str = "F16"


def _bytes_of(params: int, dtype: str) -> float:
    """The bytes that ``params`` weights of type ``dtype`` occupy.

    Args:
        params: The number of weights
        dtype: A key of BPW

    Returns:
        The number of bytes

    Raises:
        KeyError: If dtype is not a known type
    """
    return params * BPW[dtype] / 8.0


@dataclass
class Budget:
    """The bytes one full target pass reads, by tensor class.

    Attributes:
        parts: The bytes of each class, keyed by the class name
    """

    parts: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """The sum over every class, in bytes."""
        return sum(self.parts.values())

    def table(self) -> str:
        """A table of each class with its share of the total."""
        rows = sorted(self.parts.items(), key=lambda kv: -kv[1])
        width = max(len(k) for k in self.parts)
        out = []
        for name, b in rows:
            out.append(f"  {name:<{width}}  {b / 1e6:9.1f} MB  {100 * b / self.total:5.1f} %")
        out.append(f"  {'total':<{width}}  {self.total / 1e6:9.1f} MB")
        return "\n".join(out)


def target_budget(m: Model, p: Plan) -> Budget:
    """The weight bytes one full forward pass of the target model reads.

    The count covers the matrices only. Norms are f32 and under 0.1 % of the
    total, and activations are excluded because they do not scale with the
    weight format.

    Args:
        m: The model
        p: The weight type of each class

    Returns:
        The budget, one entry per tensor class
    """
    b = Budget()
    b.parts["output head"] = _bytes_of(m.vocab * m.n_embd, p.head)
    b.parts["MLP"] = _bytes_of(m.n_layer * 3 * m.n_embd * m.n_ff, p.mlp)
    # A gated-delta-net layer reads the fused qkv projection, the gate, the output
    # projection, the two f32 per-head maps and the depthwise conv taps.
    gdn_q8 = m.n_gdn * (m.conv_dim * m.n_embd + m.value_dim * m.n_embd + m.value_dim * m.n_embd)
    b.parts["GDN projections"] = _bytes_of(gdn_q8, p.gdn)
    # alpha and beta stay f32 in the file, thus they are counted at their real width
    b.parts["GDN per-head maps"] = m.n_gdn * 2 * m.n_embd * m.n_v_heads * 4.0
    # The attention is gated: the q projection is twice the head width.
    aq = 2 * m.head_dim * m.n_head * m.n_embd
    akv = 2 * m.n_head_kv * m.head_dim * m.n_embd
    ao = m.head_dim * m.n_head * m.n_embd
    b.parts["attn projections"] = _bytes_of(m.n_gqa * (aq + akv + ao), p.attn)
    # Not a weight, and read and written on every token: the recurrent state of each
    # gated-delta-net layer. The delta rule touches the whole matrix, thus this is
    # irreducible in count and only the width of the state can change it.
    b.parts["GDN state r+w"] = m.n_gdn * 2.0 * m.state_bytes
    return b


def mtp_layer_bytes(m: Model, p: Plan) -> float:
    """The weight bytes of the multi-token-prediction block, without its head.

    The block holds one full-attention decoder layer plus the ``eh_proj`` matrix, which maps
    the concatenation of the hidden state and the embedding back to the hidden size. Its
    attention is gated as in the trunk: the q projection is twice the head width.

    Args:
        m: The model
        p: The weight type of each class

    Returns:
        The number of bytes
    """
    eh = 2 * m.n_embd * m.n_embd
    mlp = 3 * m.n_embd * m.n_ff
    kv = 2 * m.n_head_kv * m.head_dim * m.n_embd
    q = 2 * m.head_dim * m.n_head * m.n_embd
    o = m.head_dim * m.n_head * m.n_embd
    return _bytes_of(eh + mlp + kv + q + o, p.mtp)


def draft_step_bytes(m: Model, p: Plan, draft_head_rows: int | None = None) -> tuple[float, float]:
    """The weight bytes one MTP draft step reads.

    A draft step runs the MTP block and then projects onto the output head to
    pick a token. With a tied head that projection reads the whole vocabulary,
    which dominates the step.

    Args:
        m: The model
        p: The weight type of each class
        draft_head_rows: The number of candidate rows the drafter projects onto.
            None means the full vocabulary

    Returns:
        A pair of the layer bytes and the head bytes
    """
    rows = m.vocab if draft_head_rows is None else min(draft_head_rows, m.vocab)
    return mtp_layer_bytes(m, p), _bytes_of(rows * m.n_embd, p.head)


def expected_accepted(draft: int, per_token_accept: float) -> float:
    """The expected number of accepted draft tokens of a chain.

    A chain accepts token i only when every token before it is accepted, thus the
    expectation is the sum of the powers of the acceptance rate.

    Args:
        draft: The number of drafted tokens
        per_token_accept: The probability that one drafted token is accepted

    Returns:
        The expected count of accepted tokens, between 0 and draft
    """
    total = 0.0
    p = 1.0
    for _ in range(draft):
        p *= per_token_accept
        total += p
    return total


@dataclass
class SpecResult:
    """The outcome of one speculative step, in units of a plain decode step.

    Attributes:
        draft_cost: The cost of every draft pass
        verify_cost: The cost of the verify pass
        tokens: The expected number of tokens the step yields
        speedup: The tokens per unit cost, against a plain step
    """

    draft_cost: float
    verify_cost: float
    tokens: float
    speedup: float


def spec_step(m: Model, p: Plan, draft: int, accept: float,
              draft_head_rows: int | None = None,
              verify_row_cost: float = 1.9) -> SpecResult:
    """The economics of one speculative step, in units of a plain decode step.

    The unit is one full target pass. A draft pass costs its own bytes over the
    bytes of a target pass. The verify pass reads the target weights once, thus
    it cannot cost less than one unit, and it costs more when the kernel is not
    at the byte floor for several rows.

    Args:
        m: The model
        p: The weight type of each class
        draft: The number of drafted tokens per step
        accept: The probability that one drafted token is accepted
        draft_head_rows: The candidate rows of the draft head, None for the full
            vocabulary
        verify_row_cost: The cost of the verify pass in units of a plain pass.
            Our HVX matvec measures 1.9 at four rows. The HMX tile is 32 by 32,
            thus a kernel on the matrix engine returns to the byte floor of 1.0

    Returns:
        The cost, the yield and the speedup
    """
    target = target_budget(m, p).total
    layer, head = draft_step_bytes(m, p, draft_head_rows)
    draft_cost = draft * (layer + head) / target
    verify_cost = verify_row_cost
    tokens = 1.0 + expected_accepted(draft, accept)
    cost = draft_cost + verify_cost
    return SpecResult(draft_cost, verify_cost, tokens, tokens / cost)


def plain_rate(m: Model, p: Plan, overhead: float = 1.0) -> float:
    """The ceiling on the plain decode rate from the weight bytes alone.

    Args:
        m: The model
        p: The weight type of each class
        overhead: A multiple on the bytes that covers activations and the state.
            Our measurement puts a token about 14 % above the weight bytes

    Returns:
        Tokens per second
    """
    return STREAM_BPS / (target_budget(m, p).total * overhead)


def _model_from_args(a: argparse.Namespace) -> Model:
    """Build the model from the named preset and any override on the command line.

    Args:
        a: The parsed arguments

    Returns:
        The model
    """
    m = MODELS[a.model]
    if a.n_ff is not None:
        m = replace(m, n_ff=a.n_ff)
    return m


def _plan_from_args(a: argparse.Namespace) -> Plan:
    """Build the weight plan from the command line.

    Args:
        a: The parsed arguments

    Returns:
        The plan
    """
    return Plan(head=a.head, mlp=a.body, gdn=a.body, attn=a.body, mtp=a.mtp)


def cmd_budget(a: argparse.Namespace) -> int:
    """Print the byte budget of one target pass.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    m, p = _model_from_args(a), _plan_from_args(a)
    b = target_budget(m, p)
    layer, head = draft_step_bytes(m, p)
    print(f"{m.name}  head {p.head}  body {p.mlp}  MTP {p.mtp}  n_ff {m.n_ff}")
    print(b.table())
    print()
    print(f"  plain decode ceiling from bytes    {plain_rate(m, p):6.1f} t/s")
    print(f"  MTP draft step                     {(layer + head) / 1e6:9.1f} MB"
          f"  = {100 * (layer + head) / b.total:.1f} % of a target pass")
    print(f"    of which the head                {head / 1e6:9.1f} MB"
          f"  = {100 * head / (layer + head):.1f} % of the draft step")
    return 0


def cmd_spec(a: argparse.Namespace) -> int:
    """Print the economics of one speculative step.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    m, p = _model_from_args(a), _plan_from_args(a)
    verify = 1.0 if a.hmx_verify else a.verify_cost
    r = spec_step(m, p, a.draft, a.accept, a.draft_head, verify)
    base = plain_rate(m, p)
    print(f"{m.name}  draft {a.draft}  acceptance {a.accept:.2f}"
          f"  draft head {a.draft_head or m.vocab} rows  verify {verify:.2f}x")
    print(f"  draft cost    {r.draft_cost:6.3f} target passes")
    print(f"  verify cost   {r.verify_cost:6.3f} target passes")
    print(f"  tokens        {r.tokens:6.3f}")
    print(f"  speedup       {r.speedup:6.2f}x  -> {base * r.speedup:5.1f} t/s"
          f"  (plain ceiling {base:.1f} t/s)")
    if r.speedup < 1.0:
        print("  NOTE: below 1.0, thus speculation loses at this acceptance rate.")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    """Print a ladder of Q8-preserving changes with the speedup of each.

    Every step keeps the body at 8-bit fidelity. No step lowers the weight
    precision of the model the user sees.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    m = _model_from_args(a)
    base_plan = Plan()
    steps: list[tuple[str, Plan, dict]] = [
        ("today: F16 MTP, full draft head, HVX verify",
         base_plan, dict(draft_head=None, verify_row_cost=1.9)),
        ("+ MTP block at Q8_0 (one line in quant/plan.py)",
         replace(base_plan, mtp="Q8_0"), dict(draft_head=None, verify_row_cost=1.9)),
        ("+ draft head cut to 3072 rows",
         replace(base_plan, mtp="Q8_0"), dict(draft_head=3072, verify_row_cost=1.9)),
        ("+ verify on the HMX",
         replace(base_plan, mtp="Q8_0"), dict(draft_head=3072, verify_row_cost=1.0)),
    ]
    print(f"{m.name}, body stays Q8_0 throughout. Acceptance {a.accept:.2f}, draft {a.draft}.")
    print()
    for label, p, kw in steps:
        r = spec_step(m, p, a.draft, a.accept,
                      kw["draft_head"], kw["verify_row_cost"])
        rate = plain_rate(m, p) * r.speedup
        print(f"  {label:<48}  {r.speedup:5.2f}x  {rate:5.1f} t/s")
    print()
    print("  Then the deeper draft that the HMX verify makes affordable:")
    p = replace(base_plan, mtp="Q8_0")
    for k in (3, 5, 7):
        r = spec_step(m, p, k, a.accept, 3072, 1.0)
        print(f"    draft {k:<2}  {r.speedup:5.2f}x  {plain_rate(m, p) * r.speedup:5.1f} t/s"
              f"   (tokens per step {r.tokens:.2f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        """Add the arguments that every subcommand takes."""
        q.add_argument("model", choices=sorted(MODELS))
        q.add_argument("--head", default="Q8_0", choices=sorted(BPW))
        q.add_argument("--body", default="Q8_0", choices=sorted(BPW))
        q.add_argument("--mtp", default="F16", choices=sorted(BPW))
        q.add_argument("--n-ff", type=int, default=None,
                       help="override the intermediate size, which is in dispute for the 4B")

    b = sub.add_parser("budget", help="the byte budget of one target pass")
    common(b)
    b.set_defaults(fn=cmd_budget)

    s = sub.add_parser("spec", help="the economics of one speculative step")
    common(s)
    s.add_argument("--draft", type=int, default=3)
    s.add_argument("--accept", type=float, default=1.0)
    s.add_argument("--draft-head", type=int, default=None,
                   help="candidate rows of the draft head, default the full vocabulary")
    s.add_argument("--verify-cost", type=float, default=1.9,
                   help="cost of the verify pass in plain passes, measured 1.9 on HVX at 4 rows")
    s.add_argument("--hmx-verify", action="store_true",
                   help="assume the verify runs at the byte floor, as the HMX tile allows")
    s.set_defaults(fn=cmd_spec)

    p = sub.add_parser("plan", help="the ladder of Q8-preserving changes")
    common(p)
    p.add_argument("--draft", type=int, default=3)
    p.add_argument("--accept", type=float, default=1.0)
    p.set_defaults(fn=cmd_plan)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
