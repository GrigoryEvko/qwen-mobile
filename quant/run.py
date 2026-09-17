"""Command line of the pipeline.

    python -m quant.run transform  --model Qwen3.5-2B [--no-rotate] [--block 32] [--permute-mlp]
    python -m quant.run verify     --model Qwen3.5-2B
    python -m quant.run convert    --model Qwen3.5-2B
    python -m quant.run quantize   --model Qwen3.5-2B [--solver qronos|gptq] [--no-scale] [--mismatch model|layer] [--drift]
    python -m quant.run convert    --model Qwen3.5-2B --source tf
    python -m quant.run export     --model Qwen3.5-2B --source tf [--head Q8_0] [--only <regex>] [--invert]
    python -m quant.run eval       --model Qwen3.5-2B --gguf <file>
    python -m quant.run drift      --model Qwen3.5-2B --source t --out analysis/<file>.drift.md

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
    src = ROOT / "weights" / f"{args.model}-{args.source}"
    out = ROOT / "weights" / "gguf" / f"{args.model}-{args.source}-F16.gguf"
    cmd = [sys.executable, str(ROOT / "llama.cpp" / "convert_hf_to_gguf.py"), str(src),
           "--outtype", "f16", "--outfile", str(out)]
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")


def _load_model(directory: Path, device: str):
    from transformers import Qwen3_5ForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    return Qwen3_5ForCausalLM.from_pretrained(directory, dtype=torch.float32, device_map=device).eval()


def _wiki_ids(tokenizer, n_seq: int, seq_len: int) -> torch.Tensor:
    from .drift import load_text_ids

    return load_text_ids(tokenizer, ROOT / "data" / "wiki.test.raw", n_seq, seq_len)


def cmd_quantize(args: argparse.Namespace) -> None:
    """Solve the plan in the quantized flow, save the folded reference, report the drift."""
    from transformers import AutoTokenizer

    from .blockopt import OptOptions
    from .calib import build_calibration
    from .drift import drift_report
    from .flow import Lockstep, Options, Quantizer, state_as_checkpoint

    src = ROOT / "weights" / f"{args.model}-t"
    out = ROOT / "quant-out" / args.model
    tok = AutoTokenizer.from_pretrained(src)
    ids = build_calibration(tok, args.n_seq, args.seq_len, args.seed, ROOT / "data" / f"calib-{args.n_seq}x{args.seq_len}.pt")
    ref = _load_model(src, args.device)
    work = _load_model(src, args.device)
    plan = _plan(args, num_layers(ROOT / "weights" / args.model))
    opts = Options(method=args.method, init=args.init, scale=not args.no_scale, permute_mlp=not args.no_permute,
                   mismatch=args.mismatch, damp=args.damp, refit_damp=args.refit_damp, batch=args.batch,
                   opt=OptOptions(epochs=args.epochs, batch=args.opt_batch, lr_weight=args.lr_weight,
                                  lr_scale=args.lr_scale, lr_other=args.lr_other, rank=args.rank,
                                  head_rank=args.head_rank, head_steps=args.head_steps))
    print(f"quantize {args.model}: {opts}", flush=True)
    Quantizer(Lockstep(ref, work, ids, args.batch), plan, out, opts).run()
    print(f"solved blocks in {out}", flush=True)

    dst = ROOT / "weights" / f"{args.model}-tf"
    save_checkpoint(state_as_checkpoint(ref, load_checkpoint(src)), src, dst, tie_word_embeddings=False)
    print(f"wrote the folded reference {dst}", flush=True)

    if args.drift:
        step = Lockstep(ref, work, _wiki_ids(tok, 16, 1024), 4)
        report = drift_report(step, f"{args.model} {args.tag}")
        path = ROOT / "analysis" / f"{args.model}-{args.tag}.drift.md"
        path.write_text(report)
        print(f"wrote {path}")


def cmd_drift(args: argparse.Namespace) -> None:
    """The drift report of the solved blocks on a source checkpoint."""
    from transformers import AutoTokenizer

    from .drift import apply_packs, drift_report
    from .flow import Lockstep

    src = ROOT / "weights" / f"{args.model}-{args.source}"
    tok = AutoTokenizer.from_pretrained(src)
    ref = _load_model(src, args.device)
    work = _load_model(src, args.device)
    stats = apply_packs(work, _packs(args))
    print(f"applied {len(stats)} solved blocks", flush=True)
    step = Lockstep(ref, work, _wiki_ids(tok, 16, 1024), 4)
    args.out.write_text(drift_report(step, f"{args.model} {args.tag}", stats))
    print(f"wrote {args.out}")


def cmd_export(args: argparse.Namespace) -> None:
    from .export import export

    f16 = ROOT / "weights" / "gguf" / f"{args.model}-{args.source}-F16.gguf"
    out = ROOT / "weights" / "gguf" / (args.out or f"{args.model}-{args.tag}.gguf")
    plan = _plan(args, num_layers(ROOT / "weights" / args.model))
    export(f16, out, _packs(args), plan, ROOT / "llama.cpp", torch.device(args.device),
           only=args.only, invert=args.invert)


def _packs(args: argparse.Namespace) -> Path:
    """The directory of the solved blocks: --packs, or quant-out/<model>."""
    return Path(args.packs) if args.packs else ROOT / "quant-out" / args.model


def cmd_eval(args: argparse.Namespace) -> None:
    """KL divergence of a GGUF against the F16 logits base on CUDA."""
    base = ROOT / "eval" / f"{args.model}-F16.wiki.c512x16.kld"
    cmd = [str(ROOT / "llama.cpp" / "build-cuda" / "bin" / "llama-perplexity"), "-m", str(args.gguf), "-ngl", "99",
           "-f", str(ROOT / "data" / "wiki.test.raw"), "-c", "512", "--chunks", "16",
           "--kl-divergence-base", str(base), "--kl-divergence"]
    lora = Path(args.gguf).with_name(Path(args.gguf).stem + "-lora.gguf")
    if lora.exists():
        cmd += ["--lora", str(lora)]
        print(f"with the adapter {lora.name}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in (res.stdout + res.stderr).splitlines():
        if any(k in line for k in ("Mean    KLD", "Maximum KLD", "99.9%   KLD", "99.0%   KLD", "Median  KLD",
                                   "RMS Δp", "Same top", "Mean PPL(Q)  ", "error", "failed")):
            print(line.split(" I ", 1)[-1])


def _plan(args: argparse.Namespace, n_layers: int) -> Plan:
    edges = tuple(int(x) for x in args.edge_layers.split(",")) if args.edge_layers else ()
    return Plan(bulk=args.bulk, head=args.head, embedding=args.embedding, kv_proj=args.kv_proj,
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

    c = sub.add_parser("convert", parents=[common])
    c.add_argument("--source", default="t", help="the checkpoint suffix: t (transformed) or tf (transformed, folded)")

    q = sub.add_parser("quantize", parents=[common])
    q.add_argument("--n-seq", type=int, default=128)
    q.add_argument("--seq-len", type=int, default=2048)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--batch", type=int, default=8)
    q.add_argument("--damp", type=float, default=0.01, help="rounding damping, relative to the mean diagonal")
    q.add_argument("--refit-damp", type=float, default=1e-6, help="refit damping, relative to the largest eigenvalue")
    q.add_argument("--method", choices=("blockopt", "solve"), default="blockopt",
                   help="block reconstruction by gradient with the quantizer in the loop, or the rounding only")
    q.add_argument("--init", choices=("rtn", "qronos", "gptq"), default="rtn", help="the rounding before the optimization")
    q.add_argument("--epochs", type=int, default=8, help="block optimization epochs over the calibration set")
    q.add_argument("--opt-batch", type=int, default=4, help="sequences per optimization step")
    q.add_argument("--lr-weight", type=float, default=1e-5)
    q.add_argument("--lr-scale", type=float, default=1e-4)
    q.add_argument("--lr-other", type=float, default=1e-4)
    q.add_argument("--head-steps", type=int, default=300)
    q.add_argument("--rank", type=int, default=0, help="rank of the low-rank correction per decoder matrix, 0 for none")
    q.add_argument("--head-rank", type=int, default=0, help="rank of the low-rank correction of the head")
    q.add_argument("--no-scale", action="store_true", help="no folded column scales")
    q.add_argument("--no-permute", action="store_true", help="no permutation of the MLP intermediate channels")
    q.add_argument("--mismatch", choices=("model", "layer"), default="model",
                   help="where the FP-flow reference restarts: never (model) or at each layer input")
    q.add_argument("--drift", action="store_true", help="write the drift report after the solve")
    q.add_argument("--tag", default="recipe", help="the label of the drift report")
    _plan_args(q)

    dr = sub.add_parser("drift", parents=[common])
    dr.add_argument("--source", default="t")
    dr.add_argument("--packs", default=None)
    dr.add_argument("--tag", default="gptq")
    dr.add_argument("--out", type=Path, required=True)

    e = sub.add_parser("export", parents=[common])
    e.add_argument("--tag", default="Q4_0")
    e.add_argument("--out", default=None)
    e.add_argument("--source", default="t", help="the F16 GGUF that supplies the unsolved tensors: t or tf")
    e.add_argument("--packs", default=None, help="the directory of the solved blocks, default quant-out/<model>")
    e.add_argument("--only", default=None, help="regex: quantize the matching tensors only, the rest stays F16")
    e.add_argument("--invert", action="store_true", help="with --only: quantize everything except the matches")
    _plan_args(e)

    ev = sub.add_parser("eval", parents=[common])
    ev.add_argument("--gguf", required=True)

    args = p.parse_args()
    {"transform": cmd_transform, "verify": cmd_verify, "convert": cmd_convert, "quantize": cmd_quantize,
     "export": cmd_export, "eval": cmd_eval, "drift": cmd_drift}[args.cmd](args)


def _plan_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--bulk", default="Q4_0", choices=("Q4_0", "IQ4_NL", "CB4"), help="the 4-bit grid of the bulk")
    sp.add_argument("--head", default="Q4_0")
    sp.add_argument("--embedding", default="Q8_0")
    sp.add_argument("--kv-proj", default="Q8_0")
    sp.add_argument("--gdn-gate", default="Q4_0")
    sp.add_argument("--edge-layers", default="")


if __name__ == "__main__":
    main()
