"""Write the inputs of the core fuzz harnesses.

The script writes two groups of files:

- The data directory (the default is build/fuzz/core/data, which is not in git):
  qwen35-vocab.gguf, the metadata of the Qwen3.5 2B model without tensors.
  llama.cpp loads it with vocab_only = true. The tokenizer and the chat
  template harnesses read it.
- The seed directory (tests/fuzz/core/seeds, which is in git): small seed
  inputs for each harness. Each seed is less than 64 KiB.

Run it with the uv environment of the repository, from the repository root:

    uv run python tests/fuzz/core/make_data.py --model weights/gguf/Qwen3.5-2B-Q8_0.gguf

The script reads the model file but does not change it. GGUFReader maps the
file into memory and reads only the header pages.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

import gguf
from gguf import GGUFReader, GGUFValueType, GGUFWriter

REPO = Path(__file__).resolve().parents[3]
VOCAB_DIR = REPO / "third_party" / "llama.cpp" / "models"


def copy_fields(reader: GGUFReader, writer: GGUFWriter, keep_tokens: int | None = None) -> None:
    """Copy each metadata key of reader into writer.

    The architecture key goes through the constructor of writer, thus this
    function skips it and the virtual GGUF.* keys. With keep_tokens, the token
    arrays keep their first keep_tokens entries, the merges keep the pairs of
    kept tokens, and each special token id that is not kept is removed.
    """
    kept: set[str] | None = None
    if keep_tokens is not None:
        tokens_field = reader.fields.get("tokenizer.ggml.tokens")
        if tokens_field is not None:
            kept = set(tokens_field.contents()[:keep_tokens])

    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        value = field.contents()
        if keep_tokens is not None:
            if name in ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
                value = value[:keep_tokens]
            elif name == "tokenizer.ggml.merges" and kept is not None:
                value = [m for m in value if all(p in kept for p in m.split(" ", 1))][: keep_tokens]
            elif name.endswith("_token_id") and isinstance(value, int) and value >= keep_tokens:
                continue
            elif name == "tokenizer.ggml.precompiled_charsmap":
                value = value[:256]
        if vtype == GGUFValueType.ARRAY and len(value) == 0:
            continue
        writer.add_key_value(name, value, vtype, sub_type)


def write_vocab_only(model: Path, out: Path) -> None:
    """Write the metadata of model, with no tensors, to out."""
    reader = GGUFReader(model)
    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(out, arch)
    copy_fields(reader, writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"data: {out} ({out.stat().st_size} bytes, arch {arch})")


def write_small_vocab(src: Path, out: Path, keep: int) -> None:
    """Write a vocab-only GGUF with the first keep tokens of src. The result is a seed of fuzz_model_load."""
    reader = GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    writer = GGUFWriter(out, arch)
    copy_fields(reader, writer, keep_tokens=keep)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_gguf_seeds(seed_dir: Path, model: Path) -> None:
    """Write the seeds of fuzz_gguf: files with each value type, tensors with data, and a truncated model head."""
    seed_dir.mkdir(parents=True, exist_ok=True)

    path = seed_dir / "all-types"
    w = GGUFWriter(path, "llama")
    w.add_uint8("t.u8", 7)
    w.add_int8("t.i8", -7)
    w.add_uint16("t.u16", 700)
    w.add_int16("t.i16", -700)
    w.add_uint32("t.u32", 70000)
    w.add_int32("t.i32", -70000)
    w.add_float32("t.f32", 1.5)
    w.add_uint64("t.u64", 1 << 40)
    w.add_int64("t.i64", -(1 << 40))
    w.add_float64("t.f64", 2.5)
    w.add_bool("t.bool", True)
    w.add_string("t.str", "text")
    w.add_array("t.arr.u32", [1, 2, 3])
    w.add_array("t.arr.str", ["a", "bc", ""])
    w.add_array("t.arr.f32", [0.5, -0.5])
    w.add_tensor("a", np.arange(8, dtype=np.float32).reshape(2, 4))
    w.add_tensor("b", np.arange(3, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    path = seed_dir / "aligned-q8"
    w = GGUFWriter(path, "llama")
    w.add_custom_alignment(64)
    w.add_uint32("general.quantization_version", 2)
    raw = np.zeros((2, 34), dtype=np.uint8)  # two Q8_0 blocks: a half scale and 32 int8 values each
    w.add_tensor("q", raw, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor("empty", np.zeros((0,), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    with open(model, "rb") as f:
        head = f.read(16384)
    (seed_dir / "qwen35-2b-head").write_bytes(head)

    # the smallest valid file: magic, version 3, no tensors, no keys
    (seed_dir / "empty-v3").write_bytes(b"GGUF" + struct.pack("<Iqq", 3, 0, 0))


def write_model_load_seeds(seed_dir: Path) -> None:
    """Write the seeds of fuzz_model_load: small vocab-only files of each tokenizer type."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    for name, keep in (
        ("qwen35", 400),
        ("qwen2", 300),
        ("llama-spm", 300),
        ("llama-bpe", 300),
        ("gpt-2", 300),
        ("bert-bge", 300),
        ("phi-3", 300),
        ("command-r", 300),
    ):
        src = VOCAB_DIR / f"ggml-vocab-{name}.gguf"
        if not src.exists():
            continue
        out = seed_dir / f"vocab-{name}"
        write_small_vocab(src, out, keep)
        if out.stat().st_size > 65536:
            out.unlink()
            print(f"seed: skip {name}, the small vocab is more than 64 KiB")


def write_text_seeds(seed_dir: Path) -> None:
    """Write the seeds of fuzz_tokenizer. The last byte of each seed holds the flags."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    texts = [
        b"Hello world",
        "Привет, мир! 你好，世界 🙂👍🏽".encode(),
        b"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n4<|im_end|>",
        b"  \t\n\n   leading and trailing   ",
        b"def f(x):\n    return x**2  # comment\n",
        b"1234567890 3.14159 1e-9 0x1F",
        b"\xff\xfe\xc3\x28\xe2\x82\x28\xf0\x9f\x98",
        b"<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
        b"<|vision_start|><|image_pad|><|vision_end|>",
        "é é ﬁ ​  ".encode(),
    ]
    for i, t in enumerate(texts):
        for flags in (0x00, 0x0F):
            (seed_dir / f"text-{i:02d}-{flags:02x}").write_bytes(t + bytes([flags]))


def write_chat_seeds(seed_dir: Path, template: str) -> None:
    """Write the seeds of fuzz_chat: the Qwen3.5 template as a jinja-mode seed, and some message-mode seeds."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    # The harness reads the mode from the last byte: an odd byte selects the jinja mode.
    (seed_dir / "jinja-qwen35").write_bytes(template.encode() + b"\x01")
    (seed_dir / "jinja-loop").write_bytes(b"{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\x01")
    for i in range(4):
        (seed_dir / f"msgs-{i}").write_bytes(bytes(range(i * 16, i * 16 + 200)) + b"\x00")


def write_binary_seeds(seed_dir: Path, sizes: tuple[int, ...]) -> None:
    """Write deterministic pseudo-random seeds of the given sizes, for the harnesses that decode a byte program."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260923)
    for i, n in enumerate(sizes):
        (seed_dir / f"rand-{i:02d}").write_bytes(rng.integers(0, 256, n, dtype=np.uint8).tobytes())


def main() -> None:
    """Parse the arguments and write the data files and the seeds."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=REPO / "weights" / "gguf" / "Qwen3.5-2B-Q8_0.gguf")
    parser.add_argument("--data-dir", type=Path, default=REPO / "build" / "fuzz" / "core" / "data")
    parser.add_argument("--seed-dir", type=Path, default=Path(__file__).resolve().parent / "seeds")
    parser.add_argument("--no-seeds", action="store_true", help="write the data directory only")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"the model {args.model} does not exist. Give the path of a Qwen3.5 GGUF with --model.")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    vocab = args.data_dir / "qwen35-vocab.gguf"
    write_vocab_only(args.model, vocab)

    template = GGUFReader(vocab).fields["tokenizer.chat_template"].contents()
    (args.data_dir / "qwen35-chat-template.jinja").write_text(template)

    if args.no_seeds:
        return
    write_gguf_seeds(args.seed_dir / "fuzz_gguf", args.model)
    write_model_load_seeds(args.seed_dir / "fuzz_model_load")
    write_text_seeds(args.seed_dir / "fuzz_tokenizer")
    write_chat_seeds(args.seed_dir / "fuzz_chat", template)
    write_binary_seeds(args.seed_dir / "fuzz_sampler", (64, 512, 4096))
    write_binary_seeds(args.seed_dir / "fuzz_recurrent", (64, 256, 1024))
    write_binary_seeds(args.seed_dir / "fuzz_npu_decode", (64, 256))
    write_binary_seeds(args.seed_dir / "tsan_threadpool", (32, 128))
    write_binary_seeds(args.seed_dir / "tsan_decode", (32, 128))
    write_binary_seeds(args.seed_dir / "tsan_topset", (32, 128))
    print(f"seeds: {args.seed_dir}")


if __name__ == "__main__":
    main()
