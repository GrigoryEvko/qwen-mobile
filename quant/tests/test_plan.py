"""The plan gives each GGUF tensor its type, and the two guarded classes follow their own switch."""

from __future__ import annotations

import json

from quant.plan import Plan

GDN = ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight")
ATTENTION = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
MLP = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")


def test_the_default_plan_of_the_targeted_rows() -> None:
    """The bulk takes Q4_0, the k and v projections Q8_0, the controls and the norms F32."""
    plan = Plan(n_layers=24)
    for tail in ("attn_qkv.weight", "ssm_out.weight", "attn_gate.weight", "attn_q.weight",
                 "attn_output.weight", *MLP):
        assert plan.type_of(f"blk.3.{tail}") == "Q4_0", tail
    for tail in ("attn_k.weight", "attn_v.weight"):
        assert plan.type_of(f"blk.3.{tail}") == "Q8_0", tail
    for tail in ("ssm_a", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_alpha.weight", "ssm_beta.weight",
                 "ssm_norm.weight", "attn_norm.weight", "post_attention_norm.weight"):
        assert plan.type_of(f"blk.3.{tail}") == "F32", tail
    assert plan.type_of("output_norm.weight") == "F32"
    assert plan.type_of("output.weight") == "Q4_0"
    assert plan.type_of("token_embd.weight") == "Q8_0"


def test_the_guarded_classes_take_their_own_type() -> None:
    """`ssm_out` and `ffn_down` move alone, and the other classes of the bulk stay."""
    plan = Plan(n_layers=24, ssm_out="Q8_0", ffn_down="Q8_0")
    assert plan.type_of("blk.3.ssm_out.weight") == "Q8_0"
    assert plan.type_of("blk.3.ffn_down.weight") == "Q8_0"
    for tail in ("attn_qkv.weight", "attn_q.weight", "attn_output.weight", "ffn_gate.weight", "ffn_up.weight"):
        assert plan.type_of(f"blk.3.{tail}") == "Q4_0", tail


def test_a_guarded_class_follows_the_bulk_by_default() -> None:
    """None means that the class takes the type of the bulk, thus an old plan keeps its meaning."""
    plan = Plan(n_layers=24, bulk="IQ4_NL")
    assert plan.type_of("blk.3.ssm_out.weight") == "IQ4_NL"
    assert plan.type_of("blk.3.ffn_down.weight") == "IQ4_NL"


def test_the_edge_layers_win_over_a_guarded_class() -> None:
    """An edge layer takes the edge type for every class of the bulk."""
    plan = Plan(n_layers=24, ssm_out="Q4_0", edge_layers=(0, 23), edge_type="Q8_0")
    assert plan.type_of("blk.0.ssm_out.weight") == "Q8_0"
    assert plan.type_of("blk.0.ffn_down.weight") == "Q8_0"
    assert plan.type_of("blk.1.ssm_out.weight") == "Q4_0"


def test_the_mtp_block_and_the_tensors_outside_the_layers() -> None:
    """The MTP matrices take the type `mtp`, and its norms and maps pass through."""
    plan = Plan(n_layers=24, mtp="Q8_0")
    assert plan.type_of("blk.24.ffn_down.weight") == "Q8_0"
    assert plan.type_of("blk.24.nextn.eh_proj.weight") == "Q8_0"
    assert plan.type_of("blk.24.nextn.hnorm_rot.weight") == "keep"
    assert Plan(n_layers=24).type_of("blk.24.ffn_down.weight") == "keep"
    assert plan.type_of("rope_freqs.weight") == "keep"


def test_an_old_saved_plan_loads_and_keeps_its_meaning() -> None:
    """A plan of folds.npz without the two new keys passes the guard of the export against the default.

    The export compares ``Plan(**saved).calibrated()`` with the JSON round
    trip of the calibrated fields of the new plan. Thus the test uses the
    same two sides.
    """
    saved = {"bulk": "Q4_0", "head": "Q4_0", "embedding": "Q4_0", "kv_proj": "Q8_0", "gdn_gate": "Q4_0",
             "edge_layers": [], "edge_type": "Q8_0", "n_layers": 24}
    old = Plan(**json.loads(json.dumps(saved)))
    wanted = json.loads(json.dumps(Plan(embedding="Q4_0", n_layers=24).calibrated()))
    assert old.calibrated() == wanted
    assert "embedding" not in old.calibrated() and "mtp" not in old.calibrated()
    assert old.calibrated()["ssm_out"] is None
    # A plan that moves a guarded class must not pass the guard of an old calibration.
    moved = json.loads(json.dumps(Plan(embedding="Q4_0", n_layers=24, ssm_out="Q8_0").calibrated()))
    assert old.calibrated() != moved
