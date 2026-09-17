"""Function-preserving transforms of the Qwen3.5 text model.

Three transforms, all offline, all exact up to floating point:

1. Fold the RMSNorm weights into the linears that consume them, thus every
   residual norm becomes a plain RMSNorm, which commutes with a rotation.
2. Rotate the residual stream with one orthogonal matrix Q: the embedding
   rows, every input projection (on the input side), every output projection
   (on the output side), and the head. The head is untied from the embedding,
   because the final norm weight folds into the head only.
3. Permute the MLP intermediate dimension, which is free because SiLU and the
   product are element-wise, thus the blocks of 32 in ``down_proj`` hold
   channels of similar magnitude.

The rotation is Q = H · D with H a Hadamard matrix (full or block diagonal)
and D a random sign diagonal. RMSNorm(Qᵀh) = Qᵀ RMSNorm(h) for orthogonal Q,
thus the network function does not change.

With ``tie_head`` the head stays the embedding E' = E·Q. The final norm
weight and the rotation go into one dense matrix M = Qᵀ·diag(γ_f)·Q that the
graph applies after the final norm: logits = E'·M·norm(h'). Because
E'·M = E·diag(γ_f)·Q, the tied head gives the logits of the untied head, and
the file holds one tensor for the lookup and the head.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

from .checkpoint import LM, OUTPUT_ROT

GDN_INPUTS = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a")
ATTN_INPUTS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
MLP_INPUTS = ("mlp.gate_proj", "mlp.up_proj")


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    """An orthogonal mixing matrix of order n, float64.

    For a power of two it is the normalized Sylvester Hadamard matrix. For
    n = 2^m · r with r odd (the 4B hidden size 2560 = 512 · 5) it is the
    Kronecker product of the Hadamard matrix of order 2^m and a seeded
    orthogonal matrix of order r, thus every entry still mixes all channels.
    """
    m = n & -n
    r = n // m
    h = torch.ones(1, 1, dtype=torch.float64, device=device)
    while h.shape[0] < m:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    h = h / math.sqrt(m)
    if r == 1:
        return h
    gen = torch.Generator(device="cpu").manual_seed(r)
    q, _ = torch.linalg.qr(torch.randn(r, r, generator=gen, dtype=torch.float64))
    q = q.to(device)
    return (h[:, None, :, None] * q[None, :, None, :]).reshape(n, n)


def rotation_matrix(n: int, block: int | None, seed: int, device: torch.device) -> torch.Tensor:
    """Q = H_block · D in float64. ``block`` None gives the full Hadamard."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=gen) * 2 - 1).to(torch.float64).to(device)
    if block is None or block >= n:
        h = hadamard(n, device)
    else:
        if n % block:
            raise ValueError(f"block {block} does not divide {n}")
        h = torch.block_diag(*([hadamard(block, device)] * (n // block)))
    return h * signs[None, :]


def _fold_input(w: torch.Tensor, gamma: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """W' = W · diag(γ) · Q for a linear that reads the normed residual."""
    return (w.to(torch.float64) * gamma.to(torch.float64)[None, :]) @ q


def _rotate_output(w: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """W' = Qᵀ · W for a linear that writes into the residual."""
    return q.T @ w.to(torch.float64)


def transform(tensors: "OrderedDict[str, torch.Tensor]", n_layers: int, layer_types: list[str],
              rotate: bool, block: int | None, seed: int, permute_mlp: bool,
              device: torch.device, tie_head: bool = False) -> "OrderedDict[str, torch.Tensor]":
    """Apply the transforms and return a new dictionary in float32.

    With ``tie_head`` the output has no ``lm_head.weight`` and holds the
    dense map ``OUTPUT_ROT`` (float32, d × d) in its place. The vision
    tensors pass through unchanged. Complexity is O(params · d) for the
    rotation, dominated by the embedding (V × d × d).

    Raises ValueError when ``tie_head`` is set and the checkpoint has a head
    that is not its embedding.
    """
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    d = tensors[LM + "embed_tokens.weight"].shape[1]
    q = rotation_matrix(d, block, seed, device) if rotate else torch.eye(d, dtype=torch.float64, device=device)

    def get(name: str) -> torch.Tensor:
        return tensors[name].to(device)

    # Qwen3.5 RMSNorm is zero-centered: y = norm(x) · (1 + w). The effective γ is 1 + w,
    # and the identity weight after the fold is 0.
    def gamma(name: str) -> torch.Tensor:
        return 1.0 + get(name).to(torch.float64)

    zero = torch.zeros(d, dtype=torch.float32)

    # The embedding and the head. logits = H · diag(γ_f) · Q · norm(h'), with H the head of the
    # checkpoint, or the embedding when the checkpoint ties them. The untied output folds
    # diag(γ_f) · Q into the head. The tied output keeps the head as the embedding and puts
    # M = Qᵀ · diag(γ_f) · Q after the final norm: E · Q · M = E · diag(γ_f) · Q.
    emb = get(LM + "embed_tokens.weight").to(torch.float64)
    gamma_f = gamma(LM + "norm.weight")
    out[LM + "embed_tokens.weight"] = (emb @ q).to(torch.float32).cpu()
    skip: set[str] = set()
    if tie_head:
        own_head = "lm_head.weight" in tensors and not torch.equal(tensors["lm_head.weight"], tensors[LM + "embed_tokens.weight"])
        if own_head:
            raise ValueError("tie_head needs a checkpoint whose head is its embedding, this one has its own lm_head.weight")
        out[OUTPUT_ROT] = (q.T @ (gamma_f[:, None] * q)).to(torch.float32).cpu()
        skip.add("lm_head.weight")
    else:
        head = get("lm_head.weight").to(torch.float64) if "lm_head.weight" in tensors else emb
        out["lm_head.weight"] = ((head * gamma_f[None, :]) @ q).to(torch.float32).cpu()
    out[LM + "norm.weight"] = zero.clone()

    # The vision merger writes image features into the residual stream.
    merger = "model.visual.merger.linear_fc2."
    if merger + "weight" in tensors:
        out[merger + "weight"] = _rotate_output(get(merger + "weight"), q).to(torch.float32).cpu()
        out[merger + "bias"] = (q.T @ get(merger + "bias").to(torch.float64)).to(torch.float32).cpu()

    for i in range(n_layers):
        p = f"{LM}layers.{i}."
        gamma_in = gamma(p + "input_layernorm.weight")
        inputs = GDN_INPUTS if layer_types[i] == "linear_attention" else ATTN_INPUTS
        for name in inputs:
            out[p + name + ".weight"] = _fold_input(get(p + name + ".weight"), gamma_in, q).to(torch.float32).cpu()
        out[p + "input_layernorm.weight"] = zero.clone()
        out_name = "linear_attn.out_proj" if layer_types[i] == "linear_attention" else "self_attn.o_proj"
        out[p + out_name + ".weight"] = _rotate_output(get(p + out_name + ".weight"), q).to(torch.float32).cpu()

        gamma_post = gamma(p + "post_attention_layernorm.weight")
        gate = _fold_input(get(p + "mlp.gate_proj.weight"), gamma_post, q)
        up = _fold_input(get(p + "mlp.up_proj.weight"), gamma_post, q)
        down = _rotate_output(get(p + "mlp.down_proj.weight"), q)
        if permute_mlp:
            # Order the intermediate channels by the column RMS of down_proj, thus each
            # block of 32 input columns of down_proj holds channels of similar magnitude.
            order = torch.argsort(down.pow(2).mean(dim=0))
            gate, up, down = gate[order, :], up[order, :], down[:, order]
        out[p + "mlp.gate_proj.weight"] = gate.to(torch.float32).cpu()
        out[p + "mlp.up_proj.weight"] = up.to(torch.float32).cpu()
        out[p + "mlp.down_proj.weight"] = down.to(torch.float32).cpu()
        out[p + "post_attention_layernorm.weight"] = zero.clone()

    # Everything not produced above passes through: the small GDN and attention
    # tensors, the vision tower, and the MTP block. The MTP block is not exact
    # after the transform: it reads the final-norm output, and the fold of the
    # final-norm weight into the head changes that vector. llama.cpp loads the
    # MTP block only for the draft-mtp speculative mode.
    for name, tensor in tensors.items():
        if name not in out and name not in skip:
            out[name] = tensor.to(torch.float32) if not name.startswith("model.visual") else tensor
    # The vision tower keeps its dtype, the rotated merger tensors included.
    for suffix in ("weight", "bias"):
        if merger + suffix in tensors:
            out[merger + suffix] = out[merger + suffix].to(tensors[merger + suffix].dtype)
    return out
