"""A tiny Qwen3.5 model in the layout of the llama.cpp converter, with random weights.

The model has the architecture ``qwen35`` of llama.cpp: gated delta net
(GDN) layers, full attention layers, the SwiGLU MLP, an optional tied
head with the dense map ``output_rot``, and an optional multi-token
prediction (MTP) block. The tokenizer is a byte-level BPE with 256 byte
tokens and a small number of merges, thus each byte of a text is at most
one token, and a short text gives the token count of a perplexity run.

``write_source`` writes the F16 GGUF that the converter would write. The
export of quant/export.py then makes the Q8_0 and Q4_0 files from it.

The value profiles of the weights:

- normal: the fan-in scale of a trained model
- tiny: 30 % of the blocks down to the F16 subnormal range, 10 % zero blocks
- large: 25 % of the blocks of the matrices after a norm near the F16
  maximum, and norm weights of 1e-4 .. 1e-3 before them, thus the
  activations stay moderate while the block scales are extreme
- mixed: a random magnitude 2^-20 .. 2^3 for each block of each matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import gguf

PROFILES = ("normal", "tiny", "large", "mixed")
BLOCK = 32
TEXT_WORDS = ("the", "a", "of", "tensor", "phone", "block", "scale", "grid", "token", "delta", "net", "rope",
              "head", "norm", "layer", "quant", "value", "row", "column", "cache", "kernel", "vector", "and",
              "to", "in", "is", "on", "with", "from", "that", "this", "model")


def bytes_to_unicode() -> dict[int, str]:
    """Give the byte-to-symbol map of the GPT-2 byte-level BPE.

    Returns:
        The symbol of each byte value 0 .. 255
    """
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    symbols = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            symbols.append(256 + n)
            n += 1
    return dict(zip(printable, (chr(c) for c in symbols), strict=True))


@dataclass(frozen=True)
class Vocab:
    """The token strings, the token types, the merges and the special ids of the toy tokenizer."""

    tokens: list[str]
    types: list[int]
    merges: list[str]
    eos: int
    pad: int


def make_vocab(size: int = 288) -> Vocab:
    """Give a byte-level BPE vocabulary of ``size`` tokens.

    Args:
        size: The number of tokens, at least 265

    Returns:
        The vocabulary

    Raises:
        ValueError: If size is less than 265
    """
    table = bytes_to_unicode()
    tokens = [table[b] for b in range(256)]
    types = [1] * 256
    merges = ["Ġ t", "Ġ a", "h e", "i n", "Ġ o"]
    tokens += [m.replace(" ", "") for m in merges]
    types += [1] * len(merges)
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
    if size < len(tokens) + len(specials):
        raise ValueError(f"the vocabulary needs at least {len(tokens) + len(specials)} tokens, not {size}")
    first_special = len(tokens)
    tokens += specials
    types += [3] * len(specials)
    k = 0
    while len(tokens) < size:
        tokens.append(f"<|reserved_{k}|>")
        types.append(3)
        k += 1
    return Vocab(tokens, types, merges, eos=first_special + 2, pad=first_special + 3)


@dataclass(frozen=True)
class Geometry:
    """The hyperparameters of a toy model.

    Attributes:
        n_embd: The width of the residual stream, a multiple of 32
        n_ff: The width of the MLP, a multiple of 32
        n_layer: The number of decoder layers without the MTP block
        interval: Each interval-th layer is full attention, the others are GDN
        n_head: The attention heads
        n_head_kv: The key and value heads of the attention
        head_dim: The key and value length of one attention head
        n_rot: The rotary dimensions of one head
        sections: The four rope sections, with a sum of n_rot / 2
        ssm_state: The key and value dimension of one GDN head
        ssm_k_heads: The key heads of the GDN
        ssm_v_heads: The value heads of the GDN, a multiple of the key heads
        conv: The kernel of the causal convolution of the GDN
        n_vocab: The number of tokens
        mtp: True to add one MTP block after the decoder layers
    """

    n_embd: int = 256
    n_ff: int = 768
    n_layer: int = 4
    interval: int = 4
    n_head: int = 2
    n_head_kv: int = 1
    head_dim: int = 256
    n_rot: int = 64
    sections: tuple[int, int, int, int] = (11, 11, 10, 0)
    ssm_state: int = 128
    ssm_k_heads: int = 1
    ssm_v_heads: int = 2
    conv: int = 4
    n_vocab: int = 288
    mtp: bool = False

    @property
    def value_dim(self) -> int:
        """The width of the value heads of the GDN."""
        return self.ssm_state * self.ssm_v_heads

    @property
    def key_dim(self) -> int:
        """The width of the key heads of the GDN."""
        return self.ssm_state * self.ssm_k_heads

    def recurrent(self) -> list[bool]:
        """Give the recurrent flag of each block, the MTP block included."""
        flags = [(i + 1) % self.interval != 0 for i in range(self.n_layer)]
        return flags + ([False] if self.mtp else [])


# A small geometry for the fuzz loop on the host: one GDN layer and one attention layer.
SMALL = Geometry(n_embd=64, n_ff=128, n_layer=2, interval=2, n_head=2, n_head_kv=1, head_dim=32, n_rot=8,
                 sections=(2, 1, 1, 0), ssm_state=32, ssm_k_heads=1, ssm_v_heads=2, n_vocab=272)
# The geometry of the phone set: the head dimensions of the 2B and 4B, the tiled value heads of the 4B.
PHONE = Geometry()


@dataclass
class Source:
    """The arrays of a written F16 source, by GGUF name, in numpy order [rows, cols]."""

    path: Path
    geometry: Geometry
    tensors: dict[str, np.ndarray] = field(default_factory=dict)
    tied: bool = False


def _matrix(gen: np.random.Generator, rows: int, cols: int, profile: str, big: bool, std: float) -> np.ndarray:
    """Give one float32 matrix of a value profile, before the F16 cast. Complexity is O(rows * cols).

    Args:
        gen: The random generator
        rows: The number of rows
        cols: The number of columns, a multiple of 32
        profile: The value profile
        big: True for a matrix that reads a norm output, which the profile "large" scales up
        std: The standard deviation of the profile "normal"
    """
    w = gen.standard_normal((rows, cols)) * std
    blocks = w.reshape(rows, cols // BLOCK, BLOCK)
    if profile == "tiny":
        pick = gen.random(blocks.shape[:2])
        blocks *= np.where(pick < 0.3, 2.0 ** -gen.uniform(14.0, 26.0, blocks.shape[:2]), 1.0)[..., None]
        blocks[pick > 0.9] = 0.0
    elif profile == "large" and big:
        pick = gen.random(blocks.shape[:2]) < 0.25
        peak = gen.uniform(2.0e4, 6.0e4, blocks.shape[:2])
        amax = np.abs(blocks).max(-1).clip(1e-30)
        blocks[pick] *= (peak / amax)[pick][..., None]
    elif profile == "mixed":
        blocks *= (2.0 ** gen.uniform(-20.0, 3.0, blocks.shape[:2]) / std)[..., None]
    return np.clip(w, -65504.0, 65504.0).astype(np.float32)


def build_tensors(geo: Geometry, profile: str, seed: int, tied: bool) -> dict[str, np.ndarray]:
    """Give the tensors of a toy model in numpy order, F16 for the matrices and F32 for the rest.

    Args:
        geo: The geometry
        profile: One of PROFILES
        seed: The seed of the values
        tied: True to leave out output.weight

    Returns:
        The arrays by GGUF name

    Raises:
        ValueError: If the profile is not known
    """
    if profile not in PROFILES:
        raise ValueError(f"the profile {profile} is not one of {PROFILES}")
    gen = np.random.default_rng(seed)
    d, ff = geo.n_embd, geo.n_ff
    norm_scale = (1e-4, 1e-3) if profile == "large" else (0.8, 1.2)
    t: dict[str, np.ndarray] = {}

    def mat(name: str, rows: int, cols: int, big: bool, gain: float = 1.0) -> None:
        t[name] = _matrix(gen, rows, cols, profile, big, gain / np.sqrt(cols)).astype(np.float16)

    def norm(name: str, n: int, after: bool) -> None:
        lo, hi = norm_scale if after else (0.8, 1.2)
        t[name] = gen.uniform(lo, hi, n).astype(np.float32)

    # The embedding keeps the normal scale in the profile "large", thus the residual stream stays moderate.
    t["token_embd.weight"] = _matrix(gen, geo.n_vocab, d, profile, profile != "large", 1.0).astype(np.float16)
    norm("output_norm.weight", d, True)
    if not tied:
        # The gain 4 gives logits with a spread of a few units, thus the distributions are not flat.
        mat("output.weight", geo.n_vocab, d, True, gain=4.0)
    for i in range(geo.n_layer + (1 if geo.mtp else 0)):
        p = f"blk.{i}."
        norm(p + "attn_norm.weight", d, True)
        norm(p + "post_attention_norm.weight", d, True)
        if i < geo.n_layer and geo.recurrent()[i]:
            mat(p + "attn_qkv.weight", 2 * geo.key_dim + geo.value_dim, d, True)
            mat(p + "attn_gate.weight", geo.value_dim, d, True)
            t[p + "ssm_conv1d.weight"] = (gen.standard_normal((2 * geo.key_dim + geo.value_dim, geo.conv)) * 0.3
                                          ).astype(np.float32)
            t[p + "ssm_dt.bias"] = (gen.standard_normal(geo.ssm_v_heads) * 0.5).astype(np.float32)
            t[p + "ssm_a"] = (-np.exp(gen.standard_normal(geo.ssm_v_heads) * 0.5)).astype(np.float32)
            t[p + "ssm_alpha.weight"] = (gen.standard_normal((geo.ssm_v_heads, d)) / np.sqrt(d)).astype(np.float16)
            t[p + "ssm_beta.weight"] = (gen.standard_normal((geo.ssm_v_heads, d)) / np.sqrt(d)).astype(np.float16)
            t[p + "ssm_norm.weight"] = gen.uniform(0.8, 1.2, geo.ssm_state).astype(np.float32)
            mat(p + "ssm_out.weight", d, geo.value_dim, False)
        else:
            mat(p + "attn_q.weight", 2 * geo.n_head * geo.head_dim, d, True)
            mat(p + "attn_k.weight", geo.n_head_kv * geo.head_dim, d, True)
            mat(p + "attn_v.weight", geo.n_head_kv * geo.head_dim, d, True)
            mat(p + "attn_output.weight", d, geo.n_head * geo.head_dim, False)
            t[p + "attn_q_norm.weight"] = gen.uniform(0.8, 1.2, geo.head_dim).astype(np.float32)
            t[p + "attn_k_norm.weight"] = gen.uniform(0.8, 1.2, geo.head_dim).astype(np.float32)
        mat(p + "ffn_gate.weight", ff, d, True)
        mat(p + "ffn_up.weight", ff, d, True)
        mat(p + "ffn_down.weight", d, ff, False)
        if i == geo.n_layer:
            mat(p + "nextn.eh_proj.weight", d, 2 * d, False)
            norm(p + "nextn.enorm.weight", d, False)
            norm(p + "nextn.hnorm.weight", d, False)
            norm(p + "nextn.shared_head_norm.weight", d, False)
            q = np.linalg.qr(gen.standard_normal((d, d)))[0]
            t[p + "nextn.hnorm_rot.weight"] = q.astype(np.float16)
            t[p + "nextn.shared_head_rot.weight"] = q.T.astype(np.float16)
    return t


def write_source(path: Path, geo: Geometry, profile: str, seed: int, tied: bool = False,
                 extra_kv: dict[str, tuple[object, gguf.GGUFValueType, gguf.GGUFValueType | None]] | None = None
                 ) -> Source:
    """Write the F16 source GGUF of a toy model, with the metadata of the converter.

    Args:
        path: The GGUF file to write
        geo: The geometry
        profile: One of PROFILES
        seed: The seed of the values
        tied: True to leave out output.weight, as the converter does for a tied checkpoint
        extra_kv: More metadata: key, then (value, type, array subtype)

    Returns:
        The written arrays
    """
    tensors = build_tensors(geo, profile, seed, tied)
    vocab = make_vocab(geo.n_vocab)
    arch = "qwen35"
    w = gguf.GGUFWriter(str(path), arch)
    w.add_string("general.name", f"qfz toy {profile} {seed}")
    w.add_uint32(f"{arch}.block_count", geo.n_layer + (1 if geo.mtp else 0))
    w.add_uint32(f"{arch}.context_length", 4096)
    w.add_uint32(f"{arch}.embedding_length", geo.n_embd)
    w.add_uint32(f"{arch}.feed_forward_length", geo.n_ff)
    w.add_uint32(f"{arch}.attention.head_count", geo.n_head)
    w.add_uint32(f"{arch}.attention.head_count_kv", geo.n_head_kv)
    w.add_key_value(f"{arch}.rope.dimension_sections", list(geo.sections), gguf.GGUFValueType.ARRAY,
                    sub_type=gguf.GGUFValueType.INT32)
    w.add_float32(f"{arch}.rope.freq_base", 10_000_000.0)
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32(f"{arch}.attention.key_length", geo.head_dim)
    w.add_uint32(f"{arch}.attention.value_length", geo.head_dim)
    if geo.mtp:
        w.add_uint32(f"{arch}.nextn_predict_layers", 1)
    w.add_uint32(f"{arch}.ssm.conv_kernel", geo.conv)
    w.add_uint32(f"{arch}.ssm.state_size", geo.ssm_state)
    w.add_uint32(f"{arch}.ssm.group_count", geo.ssm_k_heads)
    w.add_uint32(f"{arch}.ssm.time_step_rank", geo.ssm_v_heads)
    w.add_uint32(f"{arch}.ssm.inner_size", geo.value_dim)
    w.add_key_value(f"{arch}.attention.recurrent_layers", geo.recurrent(), gguf.GGUFValueType.ARRAY,
                    sub_type=gguf.GGUFValueType.BOOL)
    w.add_uint32(f"{arch}.full_attention_interval", geo.interval)
    w.add_uint32(f"{arch}.rope.dimension_count", geo.n_rot)
    w.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_F16))
    w.add_uint32("general.quantization_version", 2)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen35")
    w.add_key_value("tokenizer.ggml.tokens", vocab.tokens, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
    w.add_key_value("tokenizer.ggml.token_type", vocab.types, gguf.GGUFValueType.ARRAY,
                    sub_type=gguf.GGUFValueType.INT32)
    w.add_key_value("tokenizer.ggml.merges", vocab.merges, gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
    w.add_uint32("tokenizer.ggml.eos_token_id", vocab.eos)
    w.add_uint32("tokenizer.ggml.padding_token_id", vocab.pad)
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    for key, (value, vtype, sub) in (extra_kv or {}).items():
        if key == "general.alignment":
            # A plain copy of the key would not move the data: the writer must align to the value.
            w.add_custom_alignment(int(value))
        else:
            w.add_key_value(key, value, vtype, sub_type=sub)
    for name, a in tensors.items():
        w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    return Source(path, geo, tensors, tied)


def output_rot_for(geo: Geometry, seed: int) -> np.ndarray:
    """Give a dense map M = Qᵀ·diag(γ)·Q for a tied toy head, float32 [n_embd, n_embd].

    Args:
        geo: The geometry
        seed: The seed of Q and γ

    Returns:
        The map
    """
    gen = np.random.default_rng(seed + 7)
    q = np.linalg.qr(gen.standard_normal((geo.n_embd, geo.n_embd)))[0]
    gamma = gen.uniform(0.8, 1.2, geo.n_embd)
    return (q.T @ (gamma[:, None] * q)).astype(np.float32)


def make_text(n_words: int, seed: int) -> str:
    """Give a deterministic English-like text of ``n_words`` words, for a perplexity run.

    Args:
        n_words: The number of words
        seed: The seed of the word sequence

    Returns:
        The text, with a line break after each 12 words
    """
    gen = np.random.default_rng(seed)
    words = [TEXT_WORDS[i] for i in gen.integers(0, len(TEXT_WORDS), n_words)]
    lines = [" ".join(words[i:i + 12]) + "." for i in range(0, n_words, 12)]
    return "\n".join(lines) + "\n"
