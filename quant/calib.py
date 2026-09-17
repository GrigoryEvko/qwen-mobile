"""The calibration set."""

from __future__ import annotations

from pathlib import Path

import torch


def build_calibration(tokenizer, n_seq: int, seq_len: int, seed: int, out: Path) -> torch.Tensor:
    """Token ids [n_seq, seq_len] from C4 English, saved to ``out``."""
    if out.exists():
        return torch.load(out)
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    rows: list[torch.Tensor] = []
    buffer: list[int] = []
    for sample in ds:
        buffer.extend(tokenizer(sample["text"]).input_ids)
        buffer.append(tokenizer.eos_token_id or 0)
        while len(buffer) >= seq_len and len(rows) < n_seq:
            rows.append(torch.tensor(buffer[:seq_len]))
            buffer = buffer[seq_len:]
        if len(rows) >= n_seq:
            break
    ids = torch.stack(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, out)
    return ids
