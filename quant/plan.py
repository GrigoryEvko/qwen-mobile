"""The precision plan: which GGUF tensor gets which type.

The plan is targeted, not uniform. The recurrence controls and the norms
stay F32, the tensors with the heaviest tails stay Q8_0, and the bulk goes
to Q4_0. Every class is a switch, thus the KL harness can move a class.

The blocks after ``n_layers`` are the multi-token prediction (MTP) block,
which llama.cpp loads only for the draft-mtp speculative mode. Its matrices
take the type ``mtp``: F16 as the converter wrote them, or a round-to-nearest
type, because the calibration does not solve the block. Its norms stay F32
and its two dense maps stay F16.
"""

from __future__ import annotations

from dataclasses import dataclass, field

F32_TAILS = (
    "ssm_a", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight",
    "ssm_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
    "attn_norm.weight", "post_attention_norm.weight",
)
# The 2-D weights of the MTP block, in GGUF space.
MTP_MATRICES = (
    "nextn.eh_proj.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
)
# The two dense maps of a rotated MTP block: the export scales them by the exported output norm.
MTP_MAPS = ("nextn.hnorm_rot.weight", "nextn.shared_head_rot.weight")


@dataclass
class Plan:
    """The precision of each class. Values are GGUF type names."""

    bulk: str = "Q4_0"
    head: str = "Q4_0"
    embedding: str = "Q8_0"
    kv_proj: str = "Q8_0"
    gdn_gate: str = "Q4_0"
    edge_layers: tuple[int, ...] = field(default_factory=tuple)
    edge_type: str = "Q8_0"
    n_layers: int = 24
    mtp: str = "F16"

    def type_of(self, gguf_name: str) -> str:
        """The type for one tensor. ``keep`` means: copy from the F16 GGUF as it is."""
        if gguf_name == "output_norm.weight":
            return "F32"
        if gguf_name == "output.weight":
            return self.head
        if gguf_name == "token_embd.weight":
            return self.embedding
        if not gguf_name.startswith("blk."):
            return "keep"
        _, layer, tail = gguf_name.split(".", 2)
        layer_idx = int(layer)
        if layer_idx >= self.n_layers:
            return self.mtp if tail in MTP_MATRICES and self.mtp != "F16" else "keep"
        if tail in F32_TAILS:
            return "F32"
        if tail in ("attn_k.weight", "attn_v.weight"):
            return self.kv_proj
        if tail == "attn_gate.weight":
            return self.gdn_gate if layer_idx not in self.edge_layers else self.edge_type
        if tail in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "attn_qkv.weight",
                    "ssm_out.weight", "attn_q.weight", "attn_output.weight"):
            return self.edge_type if layer_idx in self.edge_layers else self.bulk
        return "keep"

    def solved_types(self) -> set[str]:
        """The 4-bit grid types that the calibrated solver produces. Others use round-to-nearest."""
        return {"Q4_0", "IQ4_NL", "CB4"}

    def calibrated(self) -> dict:
        """The fields that the calibration sets. A subsequent export gives the same values, or it uses the folded reference."""
        from dataclasses import asdict

        d = asdict(self)
        for uncalibrated in ("embedding", "mtp"):
            d.pop(uncalibrated, None)
        return d
