#!/usr/bin/env python3
"""Write a tiny Qwen3.5 model and a tiny vision projector with random weights.

The fuzzers of the app layer load these files for each iteration. Thus an
iteration takes milliseconds and not seconds. The files have the same
architecture as the product models:

- The text model is ``qwen35``: three gated delta net layers, one full
  attention layer, and one MTP layer. The app can draft with it.
- The vocabulary is a byte-level BPE with the ``qwen35`` pre-tokenizer,
  the 256 byte tokens, a small set of merges, and the special tokens of the
  Qwen3.5 chat template. The chat template is the real Qwen3.5 template.
- The projector is ``qwen3vl_merger``, the type of the Qwen3.5 projector.

The weights are random, but they are not white noise. Each token embedding
holds one shared direction, and a small set of "hot" rows of the output head
points along that direction. Thus the model and its MTP block prefer the same
few tokens, the draft is often accepted, and the end token and the thinking
tags occur often. The speculative paths of the engine then see accepted,
partly accepted and rejected drafts, and answers that end inside a draft.

Usage::

    uv run --no-project --with numpy --with pyyaml python make_tiny_model.py \\
        --gguf-py <llama.cpp>/gguf-py --template <llama.cpp>/models/templates/Qwen3.5-4B.jinja \\
        --out <dir>

The script writes ``tiny-qwen35-f32.gguf``, ``tiny-qwen35-q8_0.gguf`` and
``tiny-qwen35-mmproj.gguf`` into the output directory. The seed is fixed,
thus two runs write the same bytes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The dimensions of the text model. Each row length is a multiple of 32, thus
# the Q8_0 variant quantizes every matrix.
N_EMBD = 64
N_FF = 128
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = 32
N_ROT = 8
ROPE_SECTIONS = [2, 1, 1, 0]
N_LAYER = 4
FULL_ATTN_INTERVAL = 4
N_NEXTN = 1
SSM_CONV = 4
SSM_STATE = 32
SSM_GROUPS = 2
SSM_DT_RANK = 2
N_CTX_TRAIN = 8192

# The dimensions of the vision encoder.
V_EMBD = 32
V_FF = 64
V_HEAD = 2
V_LAYERS = 2
V_PATCH = 16
V_IMAGE = 128
V_MERGE = 2

# The special tokens of the Qwen3.5 template and of mtmd, in this order after the byte tokens and the merges.
SPECIAL_CONTROL = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
    "<|image_pad|>", "<|video_pad|>",
]
SPECIAL_USER = [
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<tools>", "</tools>", "<function", "</function>", "<parameter", "</parameter>",
]

# The text that the merges come from: English, Russian and some source code.
MERGE_CORPUS = (
    "The quick brown fox jumps over the lazy dog. What is in the picture? A cat on the table. "
    "the the the and and and of of to to in in is is it it that that this this with with for for "
    "Привет, как дела? Что на картинке? Кошка на столе. и и в в не не на на что что "
    "def main(): return 0\nint main() { return 0; }\nfor i in range(10): print(i)\n"
    "Hello world. Thinking step by step. The answer is 42.\n\n"
)
N_MERGES = 160


def bytes_to_unicode() -> dict[int, str]:
    """Return the byte-to-character table of the GPT-2 byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def learn_merges(text: str, table: dict[int, str], count: int) -> list[tuple[str, str]]:
    """Learn the BPE merges of the corpus, the most frequent pair first. O(count x corpus)."""
    words = [[table[b] for b in (" " + w).encode("utf-8")] for w in text.split(" ") if w]
    merges: list[tuple[str, str]] = []
    for _ in range(count):
        pairs: dict[tuple[str, str], int] = {}
        for w in words:
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
        if not pairs:
            break
        best = max(sorted(pairs), key=lambda p: pairs[p])
        merges.append(best)
        joined = best[0] + best[1]
        for i, w in enumerate(words):
            out: list[str] = []
            j = 0
            while j < len(w):
                if j + 1 < len(w) and (w[j], w[j + 1]) == best:
                    out.append(joined)
                    j += 2
                else:
                    out.append(w[j])
                    j += 1
            words[i] = out
    return merges


def build_vocab() -> tuple[list[str], list[int], list[str]]:
    """Return the tokens, their types and the merges of the tiny vocabulary."""
    table = bytes_to_unicode()
    tokens = [table[b] for b in range(256)]
    merges = learn_merges(MERGE_CORPUS, table, N_MERGES)
    seen = set(tokens)
    for a, b in merges:
        if a + b not in seen:
            tokens.append(a + b)
            seen.add(a + b)
    types = [1] * len(tokens)
    for t in SPECIAL_CONTROL:
        tokens.append(t)
        types.append(3)
    for t in SPECIAL_USER:
        tokens.append(t)
        types.append(4)
    # The vocabulary size is a multiple of 32, thus the head quantizes in blocks.
    while len(tokens) % 32 != 0:
        tokens.append(f"[PAD{len(tokens)}]")
        types.append(5)
    return tokens, types, [f"{a} {b}" for a, b in merges]


def text_tensors(rng: np.random.Generator, tokens: list[str]) -> dict[str, np.ndarray]:
    """Return the tensors of the text model by GGUF name. The numpy shape is the reverse of the ggml shape."""
    n_vocab = len(tokens)
    u = rng.standard_normal(N_EMBD).astype(np.float32)
    u /= np.linalg.norm(u)
    t: dict[str, np.ndarray] = {}

    def mat(rows: int, cols: int, scale: float = 0.02) -> np.ndarray:
        """A random matrix of ggml shape {cols, rows}."""
        return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)

    def ones(n: int) -> np.ndarray:
        return (1.0 + 0.01 * rng.standard_normal(n)).astype(np.float32)

    # The shared direction u dominates each embedding, thus the residual stream holds it.
    embd = (np.sqrt(N_EMBD) * u[None, :] + 0.3 * rng.standard_normal((n_vocab, N_EMBD))).astype(np.float32)
    t["token_embd.weight"] = embd
    t["output_norm.weight"] = ones(N_EMBD)

    # The hot rows of the head point along u, the other rows are small noise.
    head = (0.05 * rng.standard_normal((n_vocab, N_EMBD))).astype(np.float32)
    index = {tok: i for i, tok in enumerate(tokens)}
    table = bytes_to_unicode()
    hot = [
        (index[table[ord("a")]], 1.00),
        (index[table[ord(" ")]], 0.97),
        (index[table[ord("\n")]], 0.93),
        (index["<|im_end|>"], 0.80),
        (index["</think>"], 0.86),
        (index["<think>"], 0.82),
        (index[table[0xD0]], 0.85),
        (index[table[0xF0]], 0.84),
    ]
    for row, a in hot:
        head[row] = a * u + 0.05 * rng.standard_normal(N_EMBD)
    t["output.weight"] = head.astype(np.float32)

    key_dim = SSM_STATE * SSM_GROUPS
    value_dim = SSM_STATE * SSM_DT_RANK
    conv_dim = 2 * key_dim + value_dim
    q_width = HEAD_DIM * N_HEAD
    kv_width = HEAD_DIM * N_HEAD_KV

    def attention(il: int) -> None:
        t[f"blk.{il}.attn_q.weight"] = mat(2 * q_width, N_EMBD)
        t[f"blk.{il}.attn_k.weight"] = mat(kv_width, N_EMBD)
        t[f"blk.{il}.attn_v.weight"] = mat(kv_width, N_EMBD)
        t[f"blk.{il}.attn_output.weight"] = mat(N_EMBD, q_width)
        t[f"blk.{il}.attn_q_norm.weight"] = ones(HEAD_DIM)
        t[f"blk.{il}.attn_k_norm.weight"] = ones(HEAD_DIM)

    def ffn(il: int) -> None:
        t[f"blk.{il}.ffn_gate.weight"] = mat(N_FF, N_EMBD)
        t[f"blk.{il}.ffn_up.weight"] = mat(N_FF, N_EMBD)
        t[f"blk.{il}.ffn_down.weight"] = mat(N_EMBD, N_FF)

    for il in range(N_LAYER):
        t[f"blk.{il}.attn_norm.weight"] = ones(N_EMBD)
        t[f"blk.{il}.post_attention_norm.weight"] = ones(N_EMBD)
        if (il + 1) % FULL_ATTN_INTERVAL == 0:
            attention(il)
        else:
            t[f"blk.{il}.attn_qkv.weight"] = mat(conv_dim, N_EMBD)
            t[f"blk.{il}.attn_gate.weight"] = mat(value_dim, N_EMBD)
            t[f"blk.{il}.ssm_conv1d.weight"] = mat(conv_dim, SSM_CONV, 0.2)
            t[f"blk.{il}.ssm_dt.bias"] = (0.1 * rng.standard_normal(SSM_DT_RANK)).astype(np.float32)
            t[f"blk.{il}.ssm_a"] = (-np.exp(0.1 * rng.standard_normal(SSM_DT_RANK))).astype(np.float32)
            t[f"blk.{il}.ssm_beta.weight"] = mat(SSM_DT_RANK, N_EMBD)
            t[f"blk.{il}.ssm_alpha.weight"] = mat(SSM_DT_RANK, N_EMBD)
            t[f"blk.{il}.ssm_norm.weight"] = ones(SSM_STATE)
            t[f"blk.{il}.ssm_out.weight"] = mat(N_EMBD, value_dim)
        ffn(il)

    # The MTP layer. eh_proj passes the two normalized halves through, thus its
    # hidden state holds u and its head prefers the same hot tokens.
    il = N_LAYER
    t[f"blk.{il}.attn_norm.weight"] = ones(N_EMBD)
    t[f"blk.{il}.post_attention_norm.weight"] = ones(N_EMBD)
    attention(il)
    ffn(il)
    eh = np.concatenate([0.5 * np.eye(N_EMBD), 0.5 * np.eye(N_EMBD)], axis=1)
    t[f"blk.{il}.nextn.eh_proj.weight"] = (eh + 0.01 * rng.standard_normal(eh.shape)).astype(np.float32)
    t[f"blk.{il}.nextn.enorm.weight"] = ones(N_EMBD)
    t[f"blk.{il}.nextn.hnorm.weight"] = ones(N_EMBD)
    t[f"blk.{il}.nextn.shared_head_norm.weight"] = ones(N_EMBD)
    return t


def write_text_model(gguf, path: Path, tensors: dict[str, np.ndarray], tokens: list[str], types: list[int],
                     merges: list[str], template: str, quant: str) -> None:
    """Write the text model. quant is "f32" or "q8_0"; vectors and the conv kernel stay F32."""
    w = gguf.GGUFWriter(str(path), "qwen35")
    w.add_name("tiny-qwen35-fuzz")
    w.add_block_count(N_LAYER + N_NEXTN)
    w.add_context_length(N_CTX_TRAIN)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1e7)
    w.add_rope_dimension_count(N_ROT)
    w.add_rope_dimension_sections(ROPE_SECTIONS)
    w.add_ssm_conv_kernel(SSM_CONV)
    w.add_ssm_state_size(SSM_STATE)
    w.add_ssm_group_count(SSM_GROUPS)
    w.add_ssm_time_step_rank(SSM_DT_RANK)
    w.add_ssm_inner_size(SSM_STATE * SSM_DT_RANK)
    w.add_full_attention_interval(FULL_ATTN_INTERVAL)
    w.add_nextn_predict_layers(N_NEXTN)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0 if quant == "q8_0" else gguf.LlamaFileType.ALL_F32)

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen35")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(tokens.index("<|im_end|>"))
    w.add_bos_token_id(tokens.index("<|endoftext|>"))
    w.add_pad_token_id(tokens.index("<|endoftext|>"))
    w.add_add_bos_token(False)
    w.add_chat_template(template)

    for name, arr in tensors.items():
        if quant == "q8_0" and arr.ndim == 2 and arr.shape[1] % 32 == 0 and "ssm_conv1d" not in name:
            q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_mmproj(gguf, path: Path, rng: np.random.Generator) -> None:
    """Write the vision projector: a two-layer ViT and the merger into the width of the text model."""
    w = gguf.GGUFWriter(str(path), "clip")
    w.add_name("tiny-qwen35-mmproj-fuzz")
    w.add_clip_has_vision_encoder(True)
    w.add_clip_projector_type(gguf.VisionProjectorType.QWEN3VL)
    w.add_vision_use_gelu(True)
    w.add_vision_image_size(V_IMAGE)
    w.add_vision_patch_size(V_PATCH)
    w.add_vision_embedding_length(V_EMBD)
    w.add_vision_feed_forward_length(V_FF)
    w.add_vision_projection_dim(N_EMBD)
    w.add_vision_block_count(V_LAYERS)
    w.add_vision_head_count(V_HEAD)
    w.add_vision_attention_layernorm_eps(1e-6)
    w.add_vision_image_mean([0.5, 0.5, 0.5])
    w.add_vision_image_std([0.5, 0.5, 0.5])
    w.add_vision_spatial_merge_size(V_MERGE)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    def r(*shape: int, scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    n_pos = (V_IMAGE // V_PATCH) ** 2
    merged = V_EMBD * V_MERGE * V_MERGE
    w.add_tensor("v.patch_embd.weight", r(V_EMBD, 3, V_PATCH, V_PATCH))
    w.add_tensor("v.patch_embd.weight.1", r(V_EMBD, 3, V_PATCH, V_PATCH))
    w.add_tensor("v.patch_embd.bias", r(V_EMBD))
    w.add_tensor("v.position_embd.weight", r(n_pos, V_EMBD))
    for il in range(V_LAYERS):
        w.add_tensor(f"v.blk.{il}.attn_qkv.weight", r(3 * V_EMBD, V_EMBD))
        w.add_tensor(f"v.blk.{il}.attn_qkv.bias", r(3 * V_EMBD))
        w.add_tensor(f"v.blk.{il}.attn_out.weight", r(V_EMBD, V_EMBD))
        w.add_tensor(f"v.blk.{il}.attn_out.bias", r(V_EMBD))
        w.add_tensor(f"v.blk.{il}.ln1.weight", (1.0 + r(V_EMBD)).astype(np.float32))
        w.add_tensor(f"v.blk.{il}.ln1.bias", r(V_EMBD))
        w.add_tensor(f"v.blk.{il}.ln2.weight", (1.0 + r(V_EMBD)).astype(np.float32))
        w.add_tensor(f"v.blk.{il}.ln2.bias", r(V_EMBD))
        w.add_tensor(f"v.blk.{il}.ffn_up.weight", r(V_FF, V_EMBD))
        w.add_tensor(f"v.blk.{il}.ffn_up.bias", r(V_FF))
        w.add_tensor(f"v.blk.{il}.ffn_down.weight", r(V_EMBD, V_FF))
        w.add_tensor(f"v.blk.{il}.ffn_down.bias", r(V_EMBD))
    w.add_tensor("v.post_ln.weight", (1.0 + r(V_EMBD)).astype(np.float32))
    w.add_tensor("v.post_ln.bias", r(V_EMBD))
    w.add_tensor("mm.0.weight", r(merged, merged))
    w.add_tensor("mm.0.bias", r(merged))
    w.add_tensor("mm.2.weight", r(N_EMBD, merged))
    w.add_tensor("mm.2.bias", r(N_EMBD))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> int:
    """Parse the arguments and write the three files. Returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf-py", required=True, type=Path, help="The gguf-py directory of the llama.cpp tree")
    ap.add_argument("--template", required=True, type=Path, help="The Qwen3.5 chat template (jinja)")
    ap.add_argument("--out", required=True, type=Path, help="The output directory")
    args = ap.parse_args()
    if not (args.gguf_py / "gguf" / "__init__.py").is_file():
        print(f"error: {args.gguf_py} is not a gguf-py directory", file=sys.stderr)
        return 2
    sys.path.insert(0, str(args.gguf_py))
    import gguf  # noqa: PLC0415 (the path comes from the command line)

    template = args.template.read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    tokens, types, merges = build_vocab()
    tensors = text_tensors(np.random.default_rng(20260923), tokens)
    for quant in ("f32", "q8_0"):
        write_text_model(gguf, args.out / f"tiny-qwen35-{quant}.gguf", tensors, tokens, types, merges, template, quant)
    write_mmproj(gguf, args.out / "tiny-qwen35-mmproj.gguf", np.random.default_rng(20260924))
    print(f"wrote {len(tokens)} tokens, {len(merges)} merges, {len(tensors)} tensors into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
