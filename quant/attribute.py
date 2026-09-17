"""Attribute the KL of a quantized file to its tensor classes.

    python -m quant.attribute --model Qwen3.5-2B --packs-tag gptq

For each class, two files come from the same solved blocks: one with only
that class quantized (the rest F16) and one with everything except that
class. The two KL numbers bracket the share of the class. The table goes to
analysis/quant-attribution.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .sweep import ROOT, parse_eval, run

CLASSES: dict[str, str] = {
    "head": r"^output\.weight$",
    "embedding": r"^token_embd\.weight$",
    "attention k/v (Q8)": r"attn_k\.weight|attn_v\.weight",
    "attention q": r"attn_q\.weight",
    "attention o": r"attn_output\.weight",
    "GDN qkv": r"attn_qkv\.weight",
    "GDN gate z": r"attn_gate\.weight",
    "GDN out": r"ssm_out\.weight",
    "MLP gate/up": r"ffn_gate\.weight|ffn_up\.weight",
    "MLP down": r"ffn_down\.weight",
    "layers 0-2": r"^blk\.[0-2]\.",
    "layers 21-23": r"^blk\.2[1-3]\.",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--packs", required=True, help="the directory of the solved blocks")
    p.add_argument("--packs-tag", default="gptq", help="the label of the solved blocks in the table")
    p.add_argument("--source", default="tf", help="the folded reference: the kept F16 tensors share the fold coordinates")
    p.add_argument("--out", type=Path, default=ROOT / "analysis" / "quant-attribution.md")
    args = p.parse_args()

    lines = [f"# KL attribution, {args.model}, solved blocks: {args.packs_tag}", "",
             "Mean KL against the F16 base (WikiText-2, 16 x 512). 'only' quantizes the class and keeps the",
             "rest F16. 'except' quantizes everything but the class. The full file is the last row.", "",
             "| Class | only: mean KL | only: top-1 | except: mean KL | except: top-1 |", "|---|---|---|---|---|"]
    gguf = ROOT / "weights" / "gguf" / f"{args.model}-attr.gguf"
    for name, regex in CLASSES.items():
        cells = []
        for invert in (False, True):
            t0 = time.time()
            extra = ["--invert"] if invert else []
            run(["export", "--model", args.model, "--source", args.source, "--packs", args.packs,
                 "--out", gguf.name, "--only", regex, *extra])
            ev = parse_eval(run(["eval", "--model", args.model, "--gguf", str(gguf)]))
            cells += [ev["mean"], f"{ev['top1']} %"]
            print(f"{name:20s} {'except' if invert else 'only':6s} KL {ev['mean']} top-1 {ev['top1']} % "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
        args.out.write_text("\n".join(lines) + "\n")
    run(["export", "--model", args.model, "--source", args.source, "--packs", args.packs, "--out", gguf.name])
    ev = parse_eval(run(["eval", "--model", args.model, "--gguf", str(gguf)]))
    lines.append(f"| all | {ev['mean']} | {ev['top1']} % | | |")
    args.out.write_text("\n".join(lines) + "\n")
    gguf.unlink(missing_ok=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
