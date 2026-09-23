#!/usr/bin/env python3
"""Write a tiny llama model with a real tokenizer, for the llama.cpp model tests.

Some llama.cpp tests (label "model", and test-thread-safety) need a model with
a tokenizer. Upstream ctest downloads tinyllamas/stories15M-q4_0.gguf for them.
The suite has no network, and the generated models of test-llama-archs have no
tokenizer ("no_vocab"). This script copies the tokenizer of a vocab-only file
of third_party/llama.cpp/models (the preset is ggml-vocab-llama-spm.gguf, the
same SPM tokenizer of 32000 tokens that stories15M uses) into a llama model
with 2 layers, n_embd 64 and random F32 weights from a fixed seed. The output
is deterministic for one seed and one gguf-py version, and has approximately
17 MB.

Usage:
    PYTHONPATH=third_party/llama.cpp/gguf-py \
        python3 tests/suite/make-vocab-model.py --out <file.gguf> [--vocab <vocab.gguf>] [--seed N]

Requirements: python3 with numpy, and gguf-py of the llama.cpp submodule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, GGUFValueType, GGUFWriter
except ImportError as exc:  # pragma: no cover - the message is the whole point
    sys.exit(
        f"make-vocab-model: cannot import gguf ({exc}). "
        "Set PYTHONPATH=third_party/llama.cpp/gguf-py and install numpy."
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB = REPO_ROOT / "third_party/llama.cpp/models/ggml-vocab-llama-spm.gguf"

# The shape of the model. Small, thus MSan and TSan runs stay fast.
N_LAYER = 2
N_EMBD = 64
N_HEAD = 4
N_HEAD_KV = 2
N_FF = 128
N_CTX = 512


def copy_tokenizer(reader: GGUFReader, writer: GGUFWriter) -> int:
    """Copy each tokenizer.* key of the vocab file to the writer.

    Args:
        reader: The vocab-only GGUF file
        writer: The output file

    Returns:
        The number of tokens of the vocabulary.

    Raises:
        SystemExit: If the vocab file has no tokenizer.ggml.tokens array.

    Complexity: O(number of tokens).
    """
    n_vocab = 0
    for name, field in reader.fields.items():
        if not name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            sub = field.types[1]
            if sub == GGUFValueType.STRING:
                values = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
            else:
                values = [field.parts[i].tolist()[0] for i in field.data]
            writer.add_array(name, values)
            if name == "tokenizer.ggml.tokens":
                n_vocab = len(values)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(name, bytes(field.parts[field.data[0]]).decode("utf-8"))
        else:
            value = field.parts[field.data[0]].tolist()[0]
            writer.add_key_value(name, value, vtype)
    if n_vocab == 0:
        sys.exit("make-vocab-model: the vocab file has no tokenizer.ggml.tokens array.")
    return n_vocab


def add_weights(writer: GGUFWriter, n_vocab: int, seed: int) -> None:
    """Add the tensors of a llama model with random weights.

    Each weight is N(0, 0.02), except the output matrix, which is N(0, 0.5).
    The final RMS norm gives a hidden vector of length sqrt(N_EMBD) = 8, thus
    the logits have a standard deviation of approximately 4. That gives a
    peaked distribution, as a trained model has. The min-p tests of
    test-backend-sampler need it: with flat logits, min-p removes no token.

    A numpy shape (rows, cols) gives the ggml shape {cols, rows}.

    Args:
        writer: The output file
        n_vocab: The number of tokens
        seed: The seed of the random generator
    """
    rng = np.random.default_rng(seed)
    head_dim = N_EMBD // N_HEAD
    n_embd_kv = head_dim * N_HEAD_KV

    def rand(*shape: int, std: float = 0.02) -> np.ndarray:
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    writer.add_tensor("token_embd.weight", rand(n_vocab, N_EMBD))
    writer.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    writer.add_tensor("output.weight", rand(n_vocab, N_EMBD, std=0.5))
    for i in range(N_LAYER):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_q.weight", rand(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{i}.attn_k.weight", rand(n_embd_kv, N_EMBD))
        writer.add_tensor(f"blk.{i}.attn_v.weight", rand(n_embd_kv, N_EMBD))
        writer.add_tensor(f"blk.{i}.attn_output.weight", rand(N_EMBD, N_EMBD))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", rand(N_FF, N_EMBD))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", rand(N_FF, N_EMBD))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", rand(N_EMBD, N_FF))


def main() -> int:
    """Parse the arguments and write the model."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="the output GGUF file")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB, help="the vocab-only GGUF file")
    parser.add_argument("--seed", type=int, default=1234, help="the seed of the weights")
    args = parser.parse_args()

    if not args.vocab.is_file():
        sys.exit(f"make-vocab-model: the vocab file {args.vocab} does not exist.")
    reader = GGUFReader(args.vocab)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    writer = GGUFWriter(tmp, "llama")
    try:
        writer.add_name("suite-tiny-llama-spm")
        writer.add_block_count(N_LAYER)
        writer.add_context_length(N_CTX)
        writer.add_embedding_length(N_EMBD)
        writer.add_feed_forward_length(N_FF)
        writer.add_head_count(N_HEAD)
        writer.add_head_count_kv(N_HEAD_KV)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_dimension_count(N_EMBD // N_HEAD)
        writer.add_file_type(0)
        n_vocab = copy_tokenizer(reader, writer)
        writer.add_vocab_size(n_vocab)
        add_weights(writer, n_vocab, args.seed)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    tmp.replace(args.out)
    print(f"make-vocab-model: wrote {args.out} ({n_vocab} tokens, {N_LAYER} layers, n_embd {N_EMBD})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
