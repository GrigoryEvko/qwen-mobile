"""The token sets that the pipeline measures on: the calibration set, and a text file."""

from __future__ import annotations

from pathlib import Path

import torch


def load_text_ids(tokenizer, path: Path, n_seq: int, seq_len: int) -> torch.Tensor:
    """Consecutive windows of a text file as token ids [n_seq, seq_len].

    The windows follow each other with no overlap, thus each token of the file is in one window
    at most. The file must hold a multiple of seq_len tokens in its first n_seq x seq_len tokens,
    because the last window is a partial one otherwise and the rows do not stack.

    Args:
        tokenizer: A tokenizer with a call that gives input_ids
        path: The text file
        n_seq: The number of windows
        seq_len: The number of tokens in one window

    Returns:
        The token ids [n_seq, seq_len]

    Raises:
        OSError: If the file cannot be read
        RuntimeError: If the file gives a partial last window

    Complexity: O(tokens of the file).
    """
    ids = tokenizer(path.read_text()).input_ids
    rows = [torch.tensor(ids[i:i + seq_len]) for i in range(0, min(len(ids), n_seq * seq_len), seq_len)]
    return torch.stack(rows[:n_seq])


def build_calibration(tokenizer, n_seq: int, seq_len: int, seed: int, out: Path) -> torch.Tensor:
    """Token ids [n_seq, seq_len] from C4 English, saved to ``out``.

    A cached file gives its ids and reads no corpus. Without the cache the function streams the
    corpus, and that path needs the package datasets of the group corpus of pyproject.toml, which
    the default install does not hold.

    Args:
        tokenizer: A tokenizer with a call that gives input_ids
        n_seq: The number of sequences
        seq_len: The number of tokens in one sequence
        seed: The seed of the shuffle of the stream
        out: The cache file

    Returns:
        The token ids [n_seq, seq_len]

    Raises:
        SystemExit: If the cache is absent and the package datasets is absent

    Complexity: O(n_seq x seq_len) tokens of the stream.
    """
    if out.exists():
        return torch.load(out)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            f"{out} does not exist, thus this run must read the corpus, and that path needs the "
            "package datasets of the group corpus, which the default install does not hold. Run "
            "the same command in a temporary environment that holds it: uv run --frozen --isolated "
            "--no-dev --group corpus python -m quant.run quantize ARGUMENTS. The header of "
            "pyproject.toml gives the reason for the separate environment.") from exc

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
