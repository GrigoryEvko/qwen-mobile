"""The folded coordinates of a 2-D tensor of an unfolded source, from the folds and the packs.

The calibration folds column scales into the norms and into the Q8
matrices, and it permutes the intermediate channels of the MLP (refer to
``flow``). The folded reference (``--source tf``) holds every tensor in
those coordinates. When only the unfolded F16 GGUF is at hand, this module
makes the folded weight of one tensor from three sources:

- The norm ratio. A class that reads a norm has its column scales t in
  that norm: the folded norm is the source norm divided by t. Thus t is
  the ratio of the two norms. This gives t for the mixer input group
  (``attn_norm``), for the GDN output projection (``ssm_norm``, one value
  per channel of a value head), for the MLP gate and up projections
  (``post_attention_norm``) and for the head (``output_norm``).
- The row ratio of ``attn_v``. The column scales of ``attn_output`` are in
  the rows of ``attn_v``, which folds.npz holds. The least-squares ratio of
  the scaled source row to the folded row gives the scale of that KV row.
- The packs of ``ffn_gate`` and ``ffn_up``. The permutation is the row of
  the source that matches each row of the pack by the cosine, which no
  scale changes. The two packs must give the same permutation. The column
  scales of ``ffn_down`` are in the rows of ``ffn_up``, thus the
  least-squares ratio of the source row to the pack row gives them. Their
  geometric mean is one by construction, and the ratio takes that mean.

The block optimization moves the norms and the Q8 matrices after the fold,
thus a norm ratio carries that move as well. A promoted tensor with its
folded norm then computes the function of the source, not the optimized
one. ``notes`` records the size of the move: the F32 projections
``ssm_alpha`` and ``ssm_beta`` of the folds against the source scaled by
t. The quantized rows of the packs give the permutation and the down
scales with the noise of the 4-bit rounding, which the ratio averages over
the columns of a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

Source = Callable[[str], np.ndarray]
Pack = Callable[[str], "torch.Tensor | None"]

MIXER_INPUTS = ("attn_qkv.weight", "attn_gate.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "ssm_alpha.weight", "ssm_beta.weight")


@dataclass(frozen=True)
class Geometry:
    """The head structure that the folds of the output projections follow."""

    heads: int
    kv_heads: int
    head_dim: int
    v_dim: int

    @classmethod
    def from_gguf(cls, reader, arch: str, v_dim: int) -> "Geometry":
        """The attention heads from the GGUF fields, the value head dimension of the GDN as given."""
        def field(key: str) -> int:
            return int(reader.fields[f"{arch}.{key}"].contents())

        return cls(field("attention.head_count"), field("attention.head_count_kv"), field("attention.key_length"), v_dim)


def row_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The least-squares scale t per row with b ≈ a / t: ⟨a, b⟩ / ⟨b, b⟩.

    A rounding with a fitted scale leaves its error orthogonal to the
    rounded row, thus this ratio is exact for it.
    """
    return (a * b).sum(1) / (b * b).sum(1).clamp_min(1e-30)


def match_rows(pack: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """The source row of each pack row by the largest cosine. Returns (perm, min cosine, min margin).

    The margin is the best cosine minus the second best of a row. A small
    margin means that two source rows are almost parallel. Complexity is
    O(rows² · cols), on the device of the pack.
    """
    pn = pack / pack.norm(dim=1, keepdim=True).clamp_min(1e-30)
    sn = source / source.norm(dim=1, keepdim=True).clamp_min(1e-30)
    top, where = (pn @ sn.T).topk(2, dim=1)
    return where[:, 0], top[:, 0].min().item(), (top[:, 0] - top[:, 1]).min().item()


class Refold:
    """The folded weights of one calibration on an unfolded source.

    ``source`` and ``fold`` give a tensor of the F16 GGUF and of folds.npz
    by GGUF name, as float32 arrays in GGUF order. ``pack`` gives the
    dequantized pack of a name in GGUF order, or None. The scales of a
    layer are computed one time and kept.
    """

    def __init__(self, source: Source, fold: Source, pack: Pack, geometry: Geometry, device: torch.device) -> None:
        self.source, self.fold, self.pack = source, fold, pack
        self.geometry = geometry
        self.device = device
        self._mixer: dict[int, torch.Tensor] = {}
        self._gdn_out: dict[int, torch.Tensor] = {}
        self._attn_out: dict[int, torch.Tensor] = {}
        self._mlp: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.notes: list[str] = []

    def _tensor(self, get: Source, name: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(get(name))).to(self.device, torch.float32)

    def norm_ratio(self, name: str) -> torch.Tensor:
        """The column scales t in a norm: the source norm over the folded norm."""
        s, f = self._tensor(self.source, name), self._tensor(self.fold, name)
        if not torch.all(f != 0):
            raise ValueError(f"{name}: the folded norm has a zero, its scale is not defined")
        t = s / f
        self.notes.append(f"{name}: geometric mean {t.log().mean().exp().item():.4f}, "
                          f"rms log {t.log().pow(2).mean().sqrt().item():.3f}")
        return t

    def mixer_scales(self, li: int) -> torch.Tensor:
        """The column scales of the projections that read the input norm of layer li, [hidden]."""
        if li not in self._mixer:
            t = self.norm_ratio(f"blk.{li}.attn_norm.weight")
            for tail in ("ssm_alpha.weight", "ssm_beta.weight"):
                name = f"blk.{li}.{tail}"
                try:
                    f = self._tensor(self.fold, name)
                except KeyError:
                    continue
                s = self._tensor(self.source, name) * t[None, :]
                self.notes.append(f"{name}: the fold against the scaled source, relative difference "
                                  f"{((f - s).norm() / s.norm()).item():.2e}")
            self._mixer[li] = t
        return self._mixer[li]

    def gdn_out_scales(self, li: int, cols: int) -> torch.Tensor:
        """The column scales of ssm_out of layer li: one value per channel of a value head, [cols]."""
        if li not in self._gdn_out:
            t_ch = self.norm_ratio(f"blk.{li}.ssm_norm.weight")
            if t_ch.shape[0] != self.geometry.v_dim:
                raise ValueError(f"blk.{li}.ssm_norm.weight has {t_ch.shape[0]} values, the value head has "
                                 f"{self.geometry.v_dim}")
            self._gdn_out[li] = t_ch[torch.arange(cols, device=self.device) % self.geometry.v_dim]
        return self._gdn_out[li]

    def attention_out_scales(self, li: int) -> torch.Tensor:
        """The column scales of attn_output of layer li, from the rows of the folded attn_v, [heads · head_dim]."""
        if li not in self._attn_out:
            g = self.geometry
            v = self._tensor(self.source, f"blk.{li}.attn_v.weight") * self.mixer_scales(li)[None, :]
            t_rows = row_ratio(v, self._tensor(self.fold, f"blk.{li}.attn_v.weight"))
            if t_rows.shape[0] != g.kv_heads * g.head_dim:
                raise ValueError(f"blk.{li}.attn_v.weight has {t_rows.shape[0]} rows, the KV heads need "
                                 f"{g.kv_heads * g.head_dim}")
            self.notes.append(f"blk.{li}.attn_v.weight: row scales with the geometric mean "
                              f"{t_rows.log().mean().exp().item():.4f}, rms log {t_rows.log().pow(2).mean().sqrt().item():.3f}")
            j = torch.arange(g.heads * g.head_dim, device=self.device)
            self._attn_out[li] = t_rows[(j // g.head_dim // (g.heads // g.kv_heads)) * g.head_dim + j % g.head_dim]
        return self._attn_out[li]

    def mlp_folds(self, li: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(perm, t_in, t_dn) of the MLP of layer li: the channel permutation, the gate/up and the down column scales."""
        if li not in self._mlp:
            t_in = self.norm_ratio(f"blk.{li}.post_attention_norm.weight")
            gate_pack, up_pack = self.pack(f"blk.{li}.ffn_gate.weight"), self.pack(f"blk.{li}.ffn_up.weight")
            if gate_pack is None or up_pack is None:
                raise ValueError(f"layer {li}: the MLP folds need the packs of ffn_gate and ffn_up")
            gate = self._tensor(self.source, f"blk.{li}.ffn_gate.weight") * t_in[None, :]
            up = self._tensor(self.source, f"blk.{li}.ffn_up.weight") * t_in[None, :]
            perm, cos_g, margin_g = match_rows(gate_pack.to(self.device), gate)
            perm_u, cos_u, margin_u = match_rows(up_pack.to(self.device), up)
            if not torch.equal(perm, perm_u):
                raise ValueError(f"layer {li}: the gate and the up packs give different channel permutations "
                                 f"({int((perm != perm_u).sum())} rows differ)")
            if perm.unique().numel() != perm.numel():
                raise ValueError(f"layer {li}: the row match is not a permutation")
            t_dn = row_ratio(up[perm], up_pack.to(self.device))
            mean = t_dn.log().mean().exp().item()
            t_dn = t_dn / mean
            self.notes.append(f"blk.{li} MLP: match cosine {min(cos_g, cos_u):.4f}, margin {min(margin_g, margin_u):.3f}, "
                              f"down scales rms log {t_dn.log().pow(2).mean().sqrt().item():.3f}, "
                              f"geometric mean before the correction {mean:.4f}")
            self._mlp[li] = (perm, t_in, t_dn)
        return self._mlp[li]

    def weight(self, name: str, w: np.ndarray) -> torch.Tensor:
        """The folded weight of the source tensor ``w`` [rows, cols] of ``name``, float32 on the device."""
        x = torch.from_numpy(np.ascontiguousarray(w)).to(self.device, torch.float32)
        if name == "output.weight":
            return x * self.norm_ratio("output_norm.weight")[None, :]
        if not name.startswith("blk."):
            raise ValueError(f"{name}: no fold rule")
        _, layer, tail = name.split(".", 2)
        li = int(layer)
        if tail in MIXER_INPUTS:
            x = x * self.mixer_scales(li)[None, :]
            if tail == "attn_v.weight":
                g = self.geometry
                x = x / self.attention_out_scales(li).view(g.heads, g.head_dim)[::g.heads // g.kv_heads].reshape(-1)[:, None]
            return x
        if tail == "ssm_out.weight":
            return x * self.gdn_out_scales(li, x.shape[1])[None, :]
        if tail == "attn_output.weight":
            return x * self.attention_out_scales(li)[None, :]
        if tail in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
            perm, t_in, t_dn = self.mlp_folds(li)
            if tail == "ffn_down.weight":
                return x[:, perm] * t_dn[None, :]
            x = x[perm] * t_in[None, :]
            return x / t_dn[:, None] if tail == "ffn_up.weight" else x
        raise ValueError(f"{name}: no fold rule")

    def pack_error(self, name: str, w: np.ndarray) -> float | None:
        """The relative error of the pack of ``name`` against the folded source, or None without a pack."""
        pack = self.pack(name)
        if pack is None:
            return None
        folded = self.weight(name, w)
        return ((pack.to(self.device) - folded).norm() / folded.norm()).item()
