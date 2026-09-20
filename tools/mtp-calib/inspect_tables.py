"""Print the state of the calibration tables of one directory.

For each table: the totals, the proposals per draft depth, the acceptance
per depth when the table has it, the most frequent proposals decoded to
text, and the counts of the structural tokens of the tool and think formats.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path


def codec(tok_path: Path) -> tuple[Callable[[int], str], Callable[[str], list[int]]]:
    """Give a decoder and an encoder for the tokenizer, or id-only fallbacks.

    Args:
        tok_path: The tokenizer.json of the model

    Returns:
        A function from id to text, and a function from text to ids
    """
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tok_path))
    except Exception as e:  # noqa: BLE001 - the decode is a convenience only
        print(f"no tokenizer ({e}), ids only")
        return (lambda i: f"#{i}"), (lambda _: [])
    return (lambda i: tok.decode([i], skip_special_tokens=False),
            lambda s: tok.encode(s, add_special_tokens=False).ids)


def main() -> int:
    """Print the tables.

    Returns:
        The exit status
    """
    if len(sys.argv) != 3:
        print("usage: inspect_tables.py <table dir> <tokenizer.json>")
        return 2
    table_dir, tok_path = Path(sys.argv[1]), Path(sys.argv[2])
    dec, enc = codec(tok_path)

    for path in sorted(table_dir.glob("counts-calib-*.json")):
        t = json.loads(path.read_text())
        acc = t["draft_accepted"] / t["draft_n"] if t.get("draft_n") else 0.0
        print(f"\n=== {t['profile']}: prompts {t['prompts']} failed {t['failed']} "
              f"gen_tokens {t['generated_tokens']} proposals {t['positions']} distinct {t['distinct']} "
              f"accept {acc:.1%} tool_calls {t['tool_call_responses']} reasoning {t['reasoning_responses']}")
        print("   proposals by depth:", t["positions_by_pos"])
        if t.get("accept_rate_per_pos"):
            print("   accept by depth:   ", t["accept_rate_per_pos"], "over", t["verify_steps"], "steps")
        top = sorted(t["counts"].items(), key=lambda kv: -kv[1])[:16]
        print("   top:", "  ".join(f"{dec(int(k))!r}:{v}" for k, v in top))
        for s in ("<tool_call>", "</tool_call>", "<function=", "<parameter=", "</think>", "<think>"):
            ids = enc(s)
            print(f"   {s!r:14} ids {ids} counts {[t['counts'].get(str(i), 0) for i in ids]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
