"""Run named configurations end to end and append their rows to the results file.

    python -m quant.sweep --model Qwen3.5-2B --configs gptq head-q8 gate-q8 edges-q8

A configuration names its transform flags, its plan flags, and if it needs a
new calibration. Configurations that only change the export reuse the solved
blocks of the last calibration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "analysis" / "quant-results.md"

CONFIGS: dict[str, dict] = {
    "gptq": {"desc": "Targeted plan, GPTQ Q4_0 (128 x 2048 C4 tokens, damp 0.01)", "quantize": True},
    "head-q8": {"desc": "As gptq, head Q8_0", "export": ["--head", "Q8_0"]},
    "gate-q8": {"desc": "As gptq, GDN gate (in_proj_z) Q8_0", "export": ["--gdn-gate", "Q8_0"]},
    "edges-q8": {"desc": "As gptq, layers 0 and 23 Q8_0", "export": ["--edge-layers", "0,23"]},
    "all-guards": {"desc": "As gptq, head + gate + edges Q8_0",
                   "export": ["--head", "Q8_0", "--gdn-gate", "Q8_0", "--edge-layers", "0,23"]},
    "permute-mlp": {"desc": "Full R1 + MLP intermediate permutation, GPTQ Q4_0",
                    "transform": ["--permute-mlp"], "quantize": True},
    "block32": {"desc": "Block-32 Hadamard R1, GPTQ Q4_0", "transform": ["--block", "32"], "quantize": True},
    "no-rotate": {"desc": "No rotation (control), GPTQ Q4_0", "transform": ["--no-rotate"], "quantize": True},
    "calib-256x4096": {"desc": "As gptq, calibration 256 x 4096", "quantize": True,
                       "quantize_args": ["--n-seq", "256", "--seq-len", "4096", "--batch", "4"]},
    "damp-0.1": {"desc": "As gptq, damp 0.1", "quantize": True, "quantize_args": ["--damp", "0.1"]},
}


def run(args: list[str]) -> str:
    """Run a pipeline command and return its output."""
    cmd = [sys.executable, "-m", "quant.run", *args]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{res.stderr[-3000:]}")
    return res.stdout + res.stderr


def parse_eval(text: str) -> dict[str, str]:
    """The KL numbers from the eval output."""
    out = {}
    for key, pat in {"mean": r"Mean\s+KLD:\s+([\d.]+)", "p999": r"99\.9%\s+KLD:\s+([\d.]+)",
                     "max": r"Maximum KLD:\s+([\d.]+)", "top1": r"Same top p:\s+([\d.]+)",
                     "ppl": r"Mean PPL\(Q\)\s+:\s+([\d.]+)"}.items():
        m = re.search(pat, text)
        out[key] = m.group(1) if m else "?"
    return out


def append_row(name: str, desc: str, size_gib: float, ev: dict[str, str]) -> None:
    """Append one row to the results table."""
    rows = RESULTS.read_text().rstrip("\n").splitlines()
    numbered = [r for r in rows if re.match(r"^\| \d+ \|", r)]
    n = len(numbered) + 1
    rows.append(f"| {n} | {desc} ({name}) | {size_gib:.2f} GiB file | {ev['mean']} | {ev['p999']} | {ev['max']} | {ev['top1']} % | {ev['ppl']} |")
    RESULTS.write_text("\n".join(rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen3.5-2B")
    p.add_argument("--configs", nargs="+", required=True)
    args = p.parse_args()
    log = ROOT / "logs" / f"sweep-{args.model}.log"
    log.parent.mkdir(exist_ok=True)
    for name in args.configs:
        cfg = CONFIGS[name]
        t0 = time.time()
        if "transform" in cfg:
            run(["transform", "--model", args.model, *cfg["transform"]])
            run(["convert", "--model", args.model])
        if cfg.get("quantize"):
            packs = ROOT / "quant-out" / args.model
            if packs.exists():
                for f in packs.glob("*.npz"):
                    f.unlink()
            run(["quantize", "--model", args.model, *cfg.get("quantize_args", [])])
        export_args = cfg.get("export", [])
        tag = f"{name}-Q4_0"
        run(["export", "--model", args.model, "--tag", tag, *export_args])
        gguf = ROOT / "weights" / "gguf" / f"{args.model}-{tag}.gguf"
        ev = parse_eval(run(["eval", "--model", args.model, "--gguf", str(gguf)]))
        append_row(name, cfg["desc"], gguf.stat().st_size / 2**30, ev)
        line = f"{name}: {json.dumps(ev)} in {(time.time() - t0) / 60:.1f} min"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
