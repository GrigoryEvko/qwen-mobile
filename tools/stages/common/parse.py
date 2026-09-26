"""The lines of the phone tools that more than one stage reads.

llama-bench with `-o jsonl` writes one JSON object for each test. llama-perplexity with --kl-divergence
writes the KL summary at its end. The Hexagon backend with GGML_HEXAGON_PROFILE=1 writes one profile line
for each op of each DSP batch, and one OPBATCH line for each batch:

    profile-op <OP>|<names>|<dims>|<types>|<strides>|<kp>|usec <N> cycles <N> start <N> mhz <N>

A stage that needs other groups of such a line writes its own pattern. The patterns here are the ones
that more than one stage reads in the same form.
"""

import json
import re
import statistics

# llama-perplexity --kl-divergence
KLD_RE = re.compile(r"Mean\s+KLD:\s+([\d.]+) ±\s+([\d.]+)")
MAXKL_RE = re.compile(r"Maximum KLD:\s+([\d.]+)")
TOP_RE = re.compile(r"Same top p:\s+([\d.]+) ±\s+([\d.]+)")

# The profile lines of the Hexagon backend
OP_RE = re.compile(r"profile-op ([A-Z0-9_+]+)\|")
USEC_RE = re.compile(r"\|usec (\d+) cycles")
OPBATCH_RE = re.compile(r"profile-op OPBATCH\|.*\|usec (\d+) cycles")
# The first op of a graph of the model. A profile log holds the graphs one after the other.
GRAPH_START = "-> attn_norm-0|"


def bench_values(out: str) -> dict[tuple[int, int, int], float]:
    """The t/s of each llama-bench test of one run, by (n_prompt, n_gen, n_depth).

    The value is the median of the samples of the test, because the phone gives a slow repetition now and
    then. A line that is not complete JSON belongs to a run that the timeout killed, and it goes out.
    O(lines).
    """
    vals: dict[tuple[int, int, int], float] = {}
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        vals[(rec["n_prompt"], rec["n_gen"], rec["n_depth"])] = statistics.median(rec["samples_ts"])
    return vals
