"""Command line of the pipeline.

    python -m quant.run transform  --model Qwen3.5-2B [--no-rotate] [--block 32] [--permute-mlp] [--tie-head] [--suffix t]
    python -m quant.run verify     --model Qwen3.5-2B
    python -m quant.run convert    --model Qwen3.5-2B
    python -m quant.run quantize   --model Qwen3.5-2B [--init qronos|gptq] [--no-scale] [--mismatch model|layer] [--drift]
    python -m quant.run quantize   --model Qwen3.5-2B --head-only --packs quant-out/<previous run> [--stream]
    python -m quant.run convert    --model Qwen3.5-2B --source tf
    python -m quant.run export     --model Qwen3.5-2B --source tf [--head Q8_0] [--only <regex>] [--invert] [--tie-head]
    python -m quant.run export     --model Qwen3.5-2B --f16 weights/gguf/Qwen3.5-2B-F16.gguf --bulk Q8_0 --gdn-gate Q8_0 --tag Q8_0
    python -m quant.run eval       --model Qwen3.5-2B --gguf <file> [--ngl 99]
    python -m quant.run drift      --model Qwen3.5-2B --source t --out analysis/<file>.drift.md

Paths are relative to the project root: weights/<model> is the official
checkpoint, weights/<model>-t is the transformed one, weights/gguf holds the
GGUF files, quant-out/<model> holds the solved blocks. A transformed
checkpoint with the tensor ``output_rot`` (``transform --tie-head``) is a
tied one: quantize and drift then wrap the heads with its dense map, and the
head pack is ``token_embd.weight``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from .checkpoint import OUTPUT_ROT, layer_types, load_checkpoint, load_tensor, num_layers, save_checkpoint
from .plan import Plan

ROOT = Path(__file__).resolve().parent.parent


def cmd_transform(args: argparse.Namespace) -> None:
    """Write the transformed checkpoint weights/<model>-<suffix>, tied or untied."""
    src = ROOT / "weights" / args.model
    dst = ROOT / "weights" / f"{args.model}-{args.suffix}"
    from .transform import transform

    tensors = load_checkpoint(src)
    out = transform(tensors, num_layers(src), layer_types(src), rotate=not args.no_rotate,
                    block=args.block, seed=args.seed, permute_mlp=args.permute_mlp,
                    device=torch.device(args.device), tie_head=args.tie_head)
    save_checkpoint(out, src, dst, tie_word_embeddings=args.tie_head)
    print(f"wrote {dst}: rotate={not args.no_rotate} block={args.block} permute_mlp={args.permute_mlp} "
          f"tie_head={args.tie_head}")


@torch.no_grad()
def cmd_verify(args: argparse.Namespace) -> None:
    """Compare the logits of the original and the transformed model on a prompt."""
    from transformers import AutoTokenizer

    src = ROOT / "weights" / args.model
    dst = ROOT / "weights" / f"{args.model}-{args.suffix}"
    tok = AutoTokenizer.from_pretrained(src)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    ref = _load_model(src, args.device)
    a = ref(ids).logits.float()
    del ref
    torch.cuda.empty_cache()
    new = _load_model(dst, args.device)
    _tie(dst, new)
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
    """The float32 model on ``device``. A CPU load takes no device map, thus it needs no accelerate."""
    from transformers import Qwen3_5ForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    placement = {} if device == "cpu" else {"device_map": device}
    return Qwen3_5ForCausalLM.from_pretrained(directory, dtype=torch.float32, **placement).eval()


def _tie(directory: Path, *models) -> bool:
    """Wrap the heads of the models with the dense map of a tied checkpoint. Returns True when it is tied."""
    from .flow import tie_head

    rot = load_tensor(directory, OUTPUT_ROT)
    if rot is None:
        return False
    for model in models:
        tie_head(model, rot)
    return True


def _wiki_ids(tokenizer, n_seq: int, seq_len: int) -> torch.Tensor:
    from .drift import load_text_ids

    return load_text_ids(tokenizer, ROOT / "data" / "wiki.test.raw", n_seq, seq_len)


def _link_packs(packs: Path, out: Path) -> int:
    """Put the layer packs of ``packs`` into ``out`` as hard links (copies on another file system).

    The stale packs of ``out`` go first, thus the export finds one
    consistent set. Returns the number of linked packs.
    """
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.npz"):
        stale.unlink()
    n = 0
    for path in sorted(packs.glob("blk.*.npz")):
        try:
            os.link(path, out / path.name)
        except OSError:
            shutil.copy2(path, out / path.name)
        n += 1
    return n


def cmd_quantize(args: argparse.Namespace) -> None:
    """Solve the plan in the quantized flow, save the folded reference, report the drift.

    With ``--head-only`` the layers of the working copy take the packs and
    the folds of a previous run (``--packs``), and only the head, and the
    tied embedding, is solved and optimized. With ``--stream`` the copies
    stay on the CPU and each layer moves to the device for its passes.
    """
    from transformers import AutoTokenizer

    from .blockopt import OptOptions
    from .calib import build_calibration
    from .drift import apply_folds, apply_packs, drift_report
    from .flow import Lockstep, Options, Quantizer, state_as_checkpoint

    src = ROOT / "weights" / f"{args.model}-t"
    out = ROOT / "quant-out" / args.model
    tok = AutoTokenizer.from_pretrained(src)
    ids = build_calibration(tok, args.n_seq, args.seq_len, args.seed, ROOT / "data" / f"calib-{args.n_seq}x{args.seq_len}.pt")
    store = "cpu" if args.stream else args.device
    ref = _load_model(src, store)
    work = _load_model(src, store)
    tied = _tie(src, ref, work)
    plan = _plan(args, num_layers(ROOT / "weights" / args.model), tied)
    opts = Options(method=args.method, init=args.init, scale=not args.no_scale, permute_mlp=not args.no_permute,
                   mismatch=args.mismatch, damp=args.damp, refit_damp=args.refit_damp, batch=args.batch,
                   opt=OptOptions(epochs=args.epochs, batch=args.opt_batch, freeze_weights=args.freeze_weights,
                                  lr_weight=args.lr_weight,
                                  lr_scale=args.lr_scale, lr_other=args.lr_other, rank=args.rank,
                                  head_rank=args.head_rank, head_steps=args.head_steps, head_chunk=args.head_chunk))
    if args.head_only:
        packs = _packs(args)
        if packs.resolve() == out.resolve():
            raise SystemExit(f"--head-only writes to {out}: give the packs of the previous run with --packs "
                             "from another directory")
        n_packs, n_folds = len(apply_packs(work, packs)), apply_folds(work, packs)
        if tied:
            # The folds of an untied run hold the column scales of its head in the final norm. A tied
            # head has no column scales, thus its final norm starts as the identity.
            work.model.norm.weight.data.zero_()
        print(f"applied {n_packs} packs and {n_folds} folded tensors from {packs}, "
              f"linked {_link_packs(packs, out)} layer packs into {out}", flush=True)
    print(f"quantize {args.model}: tied={tied} stream={args.stream} {opts}", flush=True)
    step = Lockstep(ref, work, ids, args.batch, device=torch.device(args.device))
    quantizer = Quantizer(step, plan, out, opts)
    if args.head_only:
        quantizer.run_head_only()
    else:
        quantizer.run()
    print(f"solved blocks in {out}", flush=True)

    if not args.head_only:
        dst = ROOT / "weights" / f"{args.model}-tf"
        save_checkpoint(state_as_checkpoint(ref, load_checkpoint(src)), src, dst, tie_word_embeddings=tied)
        print(f"wrote the folded reference {dst}", flush=True)

    if args.drift:
        step = Lockstep(ref, work, _wiki_ids(tok, 16, 1024), 4, device=torch.device(args.device))
        report = drift_report(step, f"{args.model} {args.tag}")
        path = ROOT / "analysis" / f"{args.model}-{args.tag}.drift.md"
        path.write_text(report)
        print(f"wrote {path}")


def cmd_drift(args: argparse.Namespace) -> None:
    """The drift report of the solved blocks on a source checkpoint."""
    from transformers import AutoTokenizer

    from .drift import apply_folds, apply_packs, drift_report
    from .flow import Lockstep

    src = ROOT / "weights" / f"{args.model}-{args.source}"
    tok = AutoTokenizer.from_pretrained(src)
    ref = _load_model(src, args.device)
    work = _load_model(src, args.device)
    _tie(src, ref, work)
    stats = apply_packs(work, _packs(args))
    print(f"applied {len(stats)} solved blocks and {apply_folds(work, _packs(args))} folded tensors", flush=True)
    step = Lockstep(ref, work, _wiki_ids(tok, 16, 1024), 4)
    args.out.write_text(drift_report(step, f"{args.model} {args.tag}", stats))
    print(f"wrote {args.out}")


def cmd_export(args: argparse.Namespace) -> None:
    """Write the GGUF of the plan from the F16 GGUF of ``--source``, or from the file ``--f16``."""
    from .export import export

    f16 = Path(args.f16) if args.f16 else ROOT / "weights" / "gguf" / f"{args.model}-{args.source}-F16.gguf"
    out = ROOT / "weights" / "gguf" / (args.out or f"{args.model}-{args.tag}.gguf")
    plan = _plan(args, num_layers(ROOT / "weights" / args.model), args.tie_head)
    export(f16, out, _packs(args), plan, ROOT / "llama.cpp", torch.device(args.device),
           only=args.only, invert=args.invert, source_folded=args.source == "tf",
           tie_head=args.tie_head, rot=Path(args.rot) if args.rot else None)


def _packs(args: argparse.Namespace) -> Path:
    """The directory of the solved blocks: --packs, or quant-out/<model>."""
    return Path(args.packs) if args.packs else ROOT / "quant-out" / args.model


def cmd_eval(args: argparse.Namespace) -> None:
    """KL divergence of a GGUF against the F16 logits base, with ``--ngl`` layers on CUDA."""
    base = ROOT / "eval" / f"{args.model}-F16.wiki.c512x16.kld"
    cmd = [str(ROOT / "llama.cpp" / "build-cuda" / "bin" / "llama-perplexity"), "-m", str(args.gguf), "-ngl", str(args.ngl),
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


def _plan(args: argparse.Namespace, n_layers: int, tied: bool = False) -> Plan:
    """The plan of the flags. A tied head takes the type of the embedding, because it is the embedding."""
    edges = tuple(int(x) for x in args.edge_layers.split(",")) if args.edge_layers else ()
    return Plan(bulk=args.bulk, head=args.embedding if tied else args.head, embedding=args.embedding, kv_proj=args.kv_proj,
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
    t.add_argument("--tie-head", action="store_true",
                   help="keep the head tied to the embedding and write the dense map output_rot after the final norm")
    t.add_argument("--suffix", default="t", help="the suffix of the output checkpoint, weights/<model>-<suffix>")

    v = sub.add_parser("verify", parents=[common])
    v.add_argument("--prompt", default="The three laws of thermodynamics are")
    v.add_argument("--suffix", default="t", help="the suffix of the transformed checkpoint")

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
    q.add_argument("--freeze-weights", action="store_true",
                   help="the latent 4-bit weights stay from the init: the scales, levels, factors, norms and the "
                        "F32 and Q8_0 matrices of the layer still move")
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
    q.add_argument("--head-only", action="store_true",
                   help="reuse the layer packs and folds of --packs, solve and optimize only the head")
    q.add_argument("--packs", default=None, help="with --head-only: the directory of the packs of the previous run")
    q.add_argument("--stream", action="store_true",
                   help="the copies stay on the CPU, each layer moves to --device for its passes")
    q.add_argument("--head-chunk", type=int, default=16384, help="rows of the head per chunk of the logits")
    _plan_args(q)

    dr = sub.add_parser("drift", parents=[common])
    dr.add_argument("--source", default="t")
    dr.add_argument("--packs", default=None)
    dr.add_argument("--tag", default="gptq")
    dr.add_argument("--out", type=Path, required=True)

    e = sub.add_parser("export", parents=[common])
    e.add_argument("--tag", default="Q4_0")
    e.add_argument("--out", default=None)
    e.add_argument("--source", default="t", help="the F16 GGUF that supplies the unsolved tensors: t or tf "
                                                 "(tf is necessary with --only or a plan that differs from the calibration)")
    e.add_argument("--f16", default=None, help="the source GGUF as a path, for example the F16 GGUF of the original "
                                              "checkpoint. The default is weights/gguf/<model>-<source>-F16.gguf")
    e.add_argument("--packs", default=None, help="the directory of the solved blocks, default quant-out/<model>")
    e.add_argument("--only", default=None, help="regex: quantize the matching tensors only, the rest stays F16")
    e.add_argument("--invert", action="store_true", help="with --only: quantize everything except the matches")
    e.add_argument("--tie-head", action="store_true",
                   help="no output.weight: the head is token_embd through the dense map output_rot (F16)")
    e.add_argument("--rot", default=None, help="with --tie-head: a .npy file with the dense map M, [out, in] float32")
    _plan_args(e)

    ev = sub.add_parser("eval", parents=[common])
    ev.add_argument("--gguf", required=True)
    ev.add_argument("--ngl", type=int, default=99,
                    help="the number of layers on the GPU. A model that does not fit in VRAM takes the split of its base")

    args = p.parse_args()
    {"transform": cmd_transform, "verify": cmd_verify, "convert": cmd_convert, "quantize": cmd_quantize,
     "export": cmd_export, "eval": cmd_eval, "drift": cmd_drift}[args.cmd](args)


def _plan_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--bulk", default="Q4_0", choices=("Q4_0", "IQ4_NL", "CB4", "Q8_0"),
                    help="the 4-bit grid of the bulk, or Q8_0 for a round-to-nearest 8-bit file")
    sp.add_argument("--head", default="Q4_0")
    sp.add_argument("--embedding", default="Q8_0")
    sp.add_argument("--kv-proj", default="Q8_0")
    sp.add_argument("--gdn-gate", default="Q4_0")
    sp.add_argument("--edge-layers", default="")


if __name__ == "__main__":
    main()
