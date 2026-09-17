"""The map between the checkpoint tensor names and the GGUF tensor names.

The checkpoint names carry ``model.language_model.`` (the conditional
generation class) or ``model.`` (the causal LM class). Both map to the same
GGUF names that the llama.cpp converter writes for the ``qwen35`` architecture.
"""

from __future__ import annotations

import re

_LAYER_SUFFIX_TO_GGUF = {
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
}

_TOP_TO_GGUF = {
    "embed_tokens.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
}

_LAYER_RE = re.compile(r"^(?:model\.language_model\.|model\.)layers\.(\d+)\.(.+)$")
_TOP_RE = re.compile(r"^(?:model\.language_model\.|model\.)(embed_tokens\.weight|norm\.weight)$")


def to_gguf(name: str) -> str | None:
    """The GGUF name of a checkpoint tensor, or None when llama.cpp does not use it."""
    if name == "lm_head.weight":
        return "output.weight"
    m = _LAYER_RE.match(name)
    if m:
        suffix = _LAYER_SUFFIX_TO_GGUF.get(m.group(2))
        return f"blk.{m.group(1)}.{suffix}" if suffix else None
    m = _TOP_RE.match(name)
    if m:
        return _TOP_TO_GGUF[m.group(1)]
    return None
