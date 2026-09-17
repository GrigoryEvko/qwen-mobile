"""Command line of the pipeline.

    python -m quant.run transform  --model Qwen3.5-2B [--no-rotate] [--block 32] [--permute-mlp]
    python -m quant.run verify     --model Qwen3.5-2B
    python -m quant.run convert    --model Qwen3.5-2B
    python -m quant.run quantize   --model Qwen3.5-2B [--n-seq 128] [--seq-len 2048]
    python -m quant.run export     --model Qwen3.5-2B [--head Q8_0] [--gdn-gate Q8_0] [--edge-layers 0,23]
    python -m quant.run eval       --model Qwen3.5-2B --gguf <file>

Paths are relative to the project root: weights/<model> is the official
checkpoint, weights/<model>-t is the transformed one, weights/gguf holds the
GGUF files, quant-out/<model> holds the solved blocks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from .checkpoint import layer_types, load_checkpoint, num_layers, save_checkpoint
from .plan import Plan

ROOT = Path(__file__).resolve().parent.parent


def cmd_transform(args: argparse.Namespace) -> None:
    src = ROOT / "weights" / args.model
    dst = ROOT / "weights" / f"{args.model}-t"
    from .transform import transform

    tensors = load_checkpoint(src)
    out = transform(tensors, num_layers(src), layer_types(src), rotate=not args.no_rotate,
                    block=args.block, seed=args.seed, permute_mlp=args.permute_mlp,
                    device=torch.device(args.device))
    save_checkpoint(out, src, dst, tie_word_embeddings=False)
    print(f"wrote {dst}: rotate={not args.no_rotate} block={args.block} permute_mlp={args.permute_mlp}")


@torch.no_grad()
def cmd_verify(args: argparse.Namespace) -> None:
    """Compare the logits of the original and the transformed model on a prompt."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    src = ROOT / "weights" / args.model
    dst = ROOT / "weights" / f"{args.model}-t"
    tok = AutoTokenizer.from_pretrained(src)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = Qwen3_5ForCausalLM.from_pretrained(src, dtype=torch.float32, device_map=args.device)
    a = ref(ids).logits.float()
    del ref
    torch.cuda.empty_cache()
    new = Qwen3_5ForCausalLM.from_pretrained(dst, dtype=torch.float32, device_map=args.device)
    b = new(ids).logits.float()
    diff = (a - b).abs()
    print(f"logits: max abs diff {diff.max():.3e}, mean {diff.mean():.3e}, "
          f"ref abs mean {a.abs().mean():.3f}, top-1 agree {(a.argmax(-1) == b.argmax(-1)).float().mean():.3f}")


def cmd_convert(args: argparse.Namespace) -> None:
    src = ROOT / "weights" / f"{args.model}-t"
    out = ROOT / "weights" / "gguf" / f"{args.model}-t-F16.gguf"
    cmd = [sys.executable, str(ROOT / "llama.cpp" / "convert_hf_to_gguf.py"), str(src),
           "--outtype", "f16", "--outfile", str(out)]
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")


def cmd_quantize(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    from .calib import build_calibration, quantize_layers

    src = ROOT / "weights" / f"{args.model}-t"
    out = ROOT / "quant-out" / args.model
    tok = AutoTokenizer.from_pretrained(src)
    ids = build_calibration(tok, args.n_seq, args.seq_len, args.seed, ROOT / "data" / f"calib-{args.n_seq}x{args.seq_len}.pt")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3_5ForCausalLM.from_pretrained(src, dtype=torch.float32, device_map=args.device)
    model.eval()
    plan = _plan(args, num_layers(ROOT / "weights" / args.model))
    quantize_layers(model, ids, plan, out, batch=args.batch, damp=args.damp)
    print(f"solved blocks in {out}")


def cmd_export(args: argparse.Namespace) -> None:
    from .export import export

    f16 = ROOT / "weights" / "gguf" / f"{args.model}-t-F16.gguf"
    out = ROOT / "weights" / "gguf" / (args.out or f"{args.model}-{args.tag}.gguf")
    plan = _plan(args, num_layers(ROOT / "weights" / args.model))
    export(f16, out, ROOT / "quant-out" / args.model, plan, ROOT / "llama.cpp", torch.device(args.device))


def cmd_eval(args: argparse.Namespace) -> None:
    """KL divergence of a GGUF against the F16 logits base on CUDA."""
    base = ROOT / "eval" / f"{args.model}-F16.wiki.c512x16.kld"
    cmd = [str(ROOT / "llama.cpp" / "build-cuda" / "bin" / "llama-perplexity"), "-m", str(args.gguf), "-ngl", "99",
           "-f", str(ROOT / "data" / "wiki.test.raw"), "-c", "512", "--chunks", "16",
           "--kl-divergence-base", str(base), "--kl-divergence"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in (res.stdout + res.stderr).splitlines():
        if any(k in line for k in ("Mean    KLD", "Maximum KLD", "99.9%   KLD", "99.0%   KLD", "Median  KLD",
                                   "RMS Δp", "Same top", "Mean PPL(Q)  ", "error", "failed")):
            print(line.split(" I ", 1)[-1])


def _plan(args: argparse.Namespace, n_layers: int) -> Plan:
    edges = tuple(int(x) for x in args.edge_layers.split(",")) if args.edge_layers else ()
    return Plan(bulk="Q4_0", head=args.head, embedding=args.embedding, kv_proj=args.kv_proj,
                gdn_gate=args.gdn_gate, edge_layers=edges, edge_type="Q8_0", n_layers=n_layers)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="Qwen3.5-2B")
    common.add_argument("--device", default="cuda")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transform", parents=[common])
    t.add_argument("--no-rotate", action="store_true")
    t.add_argument("--block", type=int, default=None, help="Hadamard block size, default full")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--permute-mlp", action="store_true")

    v = sub.add_parser("verify", parents=[common])
    v.add_argument("--prompt", default="The three laws of thermodynamics are")

    sub.add_parser("convert", parents=[common])

    q = sub.add_parser("quantize", parents=[common])
    q.add_argument("--n-seq", type=int, default=128)
    q.add_argument("--seq-len", type=int, default=2048)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--batch", type=int, default=8)
    q.add_argument("--damp", type=float, default=0.01)
    _plan_args(q)

    e = sub.add_parser("export", parents=[common])
    e.add_argument("--tag", default="Q4_0")
    e.add_argument("--out", default=None)
    _plan_args(e)

    ev = sub.add_parser("eval", parents=[common])
    ev.add_argument("--gguf", required=True)

    args = p.parse_args()
    {"transform": cmd_transform, "verify": cmd_verify, "convert": cmd_convert,
     "quantize": cmd_quantize, "export": cmd_export, "eval": cmd_eval}[args.cmd](args)


def _plan_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--head", default="Q4_0")
    sp.add_argument("--embedding", default="Q8_0")
    sp.add_argument("--kv-proj", default="Q8_0")
    sp.add_argument("--gdn-gate", default="Q4_0")
    sp.add_argument("--edge-layers", default="")


if __name__ == "__main__":
    main()
