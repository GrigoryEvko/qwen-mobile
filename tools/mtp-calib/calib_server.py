"""Harvest the MTP draft proposals of the target model across several GPUs.

Each server holds the model, thus the model loads one time per server and the
host never reloads it per prompt. Several servers share one GPU, because one
server leaves the card idle during its host launch gap: measured on three RTX
PRO 6000 Blackwell, one server gives 1261 tok/s and four servers with CUDA MPS
give 5793 tok/s for the whole box.

The harvest follows the application exactly. The application sends a chat
turn through the template of the GGUF, with thinking on or off, with or
without tools, and with the sampler chain of ``rebuild_sampler`` in
``llama_jni.cpp``. Each of those four modes is one profile here. Each server
is bound to one profile, thus every proposal it prints belongs to that mode
and each mode gets its own table. The tables blend at selection time with a
weight per mode, in the same way as the corpus buckets.

The sampling values of each profile come from the model card of Qwen3.5-4B,
``README.md`` of the checkpoint, section Best Practices, item 1: the Instruct
row for thinking off and the Thinking row for thinking on. The application
ships the Instruct row as its defaults (``Settings.kt``).

Every server runs at debug verbosity and prints the draft candidate of every
position. This script reads that output through a pipe and keeps only the
histograms, thus a long run writes no log file. The greedy drafter proposes
the argmax of its own logits, thus the rank-0 candidate at each position is
the token the drafter would draft. The depth of a proposal inside its draft
is kept as well: the phone drafts the deeper positions only while its
acceptance window is high, thus the selector can weight them.

Each generated sequence is also saved with its prompt (gens-<profile>.jsonl),
because the same trajectories are the self-distillation set that a tuned
draft head trains on.

The run stops at a wall clock deadline, writes a checkpoint at an interval,
and stops early when the host memory falls under a floor. Host memory is the
binding limit on this box, not VRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TextIO

# pos is the depth inside one draft: 0 is the first drafted token, which every
# drafting step produces whatever its length, thus its counts do not depend on
# the draft policy of the phone. The deeper positions condition on drafted
# tokens, and the phone drafts them only while its acceptance window is high.
DRAFT_LINE = re.compile(
    r"draft candidate\s+(?P<rank>\d+), pos\s+(?P<pos>\d+):\s+(?P<id>\d+)\s+\(\s*(?P<prob>[\d.]+)\)"
)

# One line of the Prometheus output of /metrics. The per-position counter
# carries a position label, the totals carry none.
METRIC_LINE = re.compile(
    r"^llamacpp:(?P<name>\w+)(?:\{position=\"(?P<pos>\d+)\"\})?\s+(?P<value>[-+\d.eE]+)$", re.M
)

# The two sampler rows of the model card, plus the penalty fields that the
# application fixes: kPenaltyLastN = 256, the repeat and frequency penalties off.
INSTRUCT_SAMPLING = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repeat_penalty": 1.0, "frequency_penalty": 0.0,
}
THINKING_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repeat_penalty": 1.0, "frequency_penalty": 0.0,
}

# The four modes of the application. gen is longer with thinking, because the
# think block comes before the answer: at 1024 only one trace in six closed
# in the rehearsal. slot_ctx holds the tool block (about 700 tokens), the
# prompt and gen, and the context of a server is np times the slot_ctx of
# its profile.
PROFILES: dict[str, dict] = {
    "instruct":       {"thinking": False, "tools": False, "sampling": INSTRUCT_SAMPLING, "gen": 512,  "slot_ctx": 2048},
    "thinking":       {"thinking": True,  "tools": False, "sampling": THINKING_SAMPLING, "gen": 2048, "slot_ctx": 4096},
    "instruct-tools": {"thinking": False, "tools": True,  "sampling": INSTRUCT_SAMPLING, "gen": 512,  "slot_ctx": 2048},
    "thinking-tools": {"thinking": True,  "tools": True,  "sampling": THINKING_SAMPLING, "gen": 2048, "slot_ctx": 4096},
}

# A representative tool set. The application has no tool schema yet, thus
# these stand in for the structure a tool call carries: the tool_call tags,
# the JSON object, the name and the arguments. Run the harvest again with the
# real schemas when the application has them.
def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Give one tool in the OpenAI function shape that the template renders.

    Args:
        name: The function name
        description: What the function does
        properties: The JSON schema of each parameter
        required: The parameters that a call must give

    Returns:
        The tool entry
    """
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS: list[dict] = [
    tool("web_search", "Search the web and return the top results.",
         {"query": {"type": "string", "description": "The search query"}}, ["query"]),
    tool("get_weather", "Get the current weather of a location.",
         {"location": {"type": "string", "description": "City and country"},
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["location"]),
    tool("calculator", "Evaluate an arithmetic expression.",
         {"expression": {"type": "string"}}, ["expression"]),
    tool("run_python", "Run a Python snippet and return its output.",
         {"code": {"type": "string"}}, ["code"]),
    tool("read_file", "Read a text file from the device.",
         {"path": {"type": "string"}}, ["path"]),
    tool("write_file", "Write text to a file on the device.",
         {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    tool("get_current_time", "Get the current time in a time zone.",
         {"timezone": {"type": "string", "description": "IANA zone, for example Europe/Berlin"}}, ["timezone"]),
    tool("send_message", "Send a text message to a contact.",
         {"recipient": {"type": "string"}, "text": {"type": "string"}}, ["recipient", "text"]),
]


class Sink:
    """A per-server proposal histogram that a reader thread fills.

    Each server owns one sink, thus the reader threads never contend except
    for the short moment when the checkpoint copies a sink.
    """

    def __init__(self) -> None:
        """Make an empty sink."""
        self.lock = threading.Lock()
        self.counts: Counter[int] = Counter()
        self.by_pos: dict[int, Counter[int]] = {}
        self.positions = 0

    def add(self, token: int, pos: int) -> None:
        """Count one rank-0 proposal.

        Args:
            token: The vocabulary index that the drafter proposed
            pos: The depth of the proposal inside its draft, 0 for the first
        """
        with self.lock:
            self.counts[token] += 1
            self.by_pos.setdefault(pos, Counter())[token] += 1
            self.positions += 1

    def snapshot(self) -> tuple[Counter[int], int, dict[int, Counter[int]]]:
        """Give a copy of the histogram.

        Returns:
            The counts, the number of positions, and the counts per depth
        """
        with self.lock:
            return Counter(self.counts), self.positions, {p: Counter(c) for p, c in self.by_pos.items()}


def read_proposals(proc: subprocess.Popen, sink: Sink) -> None:
    """Read one server's output and keep only the rank-0 draft proposals.

    The cheap substring check runs first, because most lines of the debug
    output are not draft candidates and the regular expression is far more
    expensive. Complexity is O(L) in the lines that the server prints.

    Args:
        proc: The server process, with its output on a pipe
        sink: The histogram to fill
    """
    stream = proc.stdout
    if stream is None:
        return
    for raw in stream:
        if b"draft candidate" not in raw:
            continue
        m = DRAFT_LINE.search(raw.decode("utf-8", "ignore"))
        if m is not None and m["rank"] == "0":
            sink.add(int(m["id"]), int(m["pos"]))


def wait_health(port: int, timeout: float) -> bool:
    """Wait until a server answers /health with ok.

    Args:
        port: The port of the server
        timeout: The maximum wait in seconds

    Returns:
        True if the server became ready inside the timeout
    """
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if b"ok" in r.read():
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    return False


def launch(model: str, binary: str, gpu: int, port: int, np: int, ctx: int, lv: int,
           spec: str, ubatch: int, extra: list[str]) -> subprocess.Popen:
    """Launch one llama-server pinned to one GPU, with its output on a pipe.

    The micro batch holds a whole decode step: the slots plus the longest
    prompt. With a prompt split over two micro batches the copy of the MTP
    hidden state can race the next graph and the drafter proposes garbage
    (llama.cpp issue 27572), which would pollute the histogram.

    Args:
        model: The GGUF with the full MTP head
        binary: The llama-server of a draft-mtp build
        gpu: The CUDA device index
        port: The port to listen on
        np: The number of parallel slots
        ctx: The total context, split over the slots
        lv: The log verbosity, 5 to emit the draft candidate of every position
        spec: The speculation type, draft-mtp to harvest proposals
        ubatch: The micro batch, also the batch
        extra: More server flags, for example the thread counts

    Returns:
        The server process
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    # The host prompt cache of the server keeps the state of every finished
    # slot in RAM, 8192 MiB per server by default, and the harvest never
    # repeats a prompt: with it on, nine servers grow by about 1 GiB per
    # minute each until the memory guard fires.
    cmd = [binary, "-m", model, "--spec-type", spec, "-ngl", "99", "-fa", "on",
           "-np", str(np), "-c", str(ctx), "-b", str(ubatch), "-ub", str(ubatch),
           "--cache-ram", "0", "--host", "127.0.0.1", "--port", str(port),
           "-lv", str(lv), "--metrics", *extra]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, bufsize=1024 * 1024)


def stop_servers(procs: list[subprocess.Popen]) -> None:
    """Terminate the servers and wait for them, with a kill after a grace period.

    A second call is harmless, because a signal to a finished process is a
    no-op. The RAM guard calls this first, thus the memory is released before
    the in-flight requests drain.

    Args:
        procs: The server processes
    """
    for p in procs:
        p.terminate()
    # One deadline for all of them, thus a slow shutdown of nine servers
    # takes 30 seconds and not nine times that.
    deadline = time.time() + 30.0
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()


def load_prompts(path: Path) -> list:
    """Read one prompt per line, each a JSON string or a JSON list of messages.

    Args:
        path: The prompt file

    Returns:
        The prompts, in file order

    Raises:
        ValueError: If the file holds no prompt
    """
    prompts = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not prompts:
        raise ValueError(f"no prompt in {path}")
    return prompts


def prompt_seed(base: int, label: str, index: int) -> int:
    """Give the sampler seed of one prompt.

    The seed differs between prompts, thus the random draws of the set are
    not correlated, and it is a pure function of the prompt, thus a second
    decode of the same prompt, with or without a drafter, takes the same
    draws and the two are a matched pair.

    Args:
        base: The seed of the run
        label: The name of the profile or of the test
        index: The index of the prompt in its list

    Returns:
        A seed in the range of a positive 32 bit integer
    """
    digest = hashlib.blake2b(f"{base}:{label}:{index}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def post(port: int, route: str, body: dict, timeout: float) -> dict:
    """Post one JSON request to a server and give the parsed response.

    Args:
        port: The port of the server
        route: The path of the endpoint
        body: The request
        timeout: The maximum wait in seconds

    Returns:
        The parsed response, or an empty dict on failure
    """
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError):
        return {}


def complete(port: int, prompt: str | list, profile: dict, penalty_last_n: int, seed: int) -> dict:
    """Render one turn through the template of the model, then complete it raw.

    The application renders the chat through the template of the GGUF and
    decodes the result with no grammar. The harvest does the same in two
    calls: /apply-template gives the exact prompt the template makes, with
    the tool block and the think block, and /completion generates from it.
    The chat endpoint would add a tool grammar that the application does not
    have, thus it is not used.

    The sampler fields bind the target only. The drafter proposes the argmax
    of its own full head, thus top_k does not narrow the harvest.

    Args:
        port: The port of the server
        prompt: The user turn, or a list of messages that ends with a tool result
        profile: The mode: thinking, tools, sampling, and gen
        penalty_last_n: The window of the presence penalty
        seed: The seed of the sampler, a function of the prompt

    Returns:
        The parsed response of /completion with the rendered prompt under
        the key _prompt and the seed under _seed, or an empty dict on failure
    """
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    render: dict = {
        "messages": messages,
        # With thinking off the template emits an empty think block, which is
        # what the application sends by default (inputs.enable_thinking = false).
        "chat_template_kwargs": {"enable_thinking": profile["thinking"]},
    }
    if profile["tools"]:
        render["tools"] = TOOLS
    text = post(port, "/apply-template", render, 60.0).get("prompt")
    if not text:
        return {}
    resp = post(port, "/completion", {
        "prompt": text, "n_predict": profile["gen"], "cache_prompt": False,
        "repeat_last_n": penalty_last_n, "return_tokens": True, "seed": seed, **profile["sampling"],
    }, 1200.0)
    if resp:
        resp["_prompt"] = text
        resp["_seed"] = seed
    return resp


def account(resp: dict, state: dict, dump: TextIO | None) -> None:
    """Add one response to the running totals of its profile, and save its sequence.

    The token count comes from the timings block first and from the usage
    block as the fallback. The tool call and the reasoning counters tell
    whether a mode produced what it is meant to produce.

    Args:
        resp: The parsed response of one request
        state: The totals of the profile to update
        dump: The file that takes the prompt and the generated ids, or None
    """
    t = resp.get("timings") or {}
    content = resp.get("content") or ""
    state["done"] += 1
    state["failed"] += 0 if resp else 1
    state["tokens"] += int(t.get("predicted_n", resp.get("tokens_predicted", 0)) or 0)
    state["acc"] += int(t.get("draft_n_accepted", 0) or 0)
    state["draft"] += int(t.get("draft_n", 0) or 0)
    # The template of Qwen3.5 makes the model emit an XML call inside
    # tool_call tags, and a think block that closes with the end tag. With
    # thinking off the closed empty block is in the prompt, not in the output.
    if "<tool_call>" in content:
        state["tool_calls"] += 1
    if "</think>" in content:
        state["reasoning"] += 1
    if dump is not None and resp.get("tokens"):
        dump.write(json.dumps({"prompt": resp.get("_prompt", ""), "seed": resp.get("_seed", 0),
                               "draft_n": int(t.get("draft_n", 0) or 0),
                               "draft_accepted": int(t.get("draft_n_accepted", 0) or 0),
                               "tokens": resp["tokens"]}) + "\n")


def empty_metrics() -> dict:
    """Give the zero speculative counters.

    Returns:
        The verification steps, the draft tokens, the accepted tokens, and
        the accepted tokens per draft position
    """
    return {"verify_steps": 0, "draft_tokens": 0, "draft_accepted": 0, "accepted_per_pos": []}


def scrape_metrics(port: int) -> dict:
    """Read the speculative counters of one server from its /metrics endpoint.

    The server counts, per draft position, how many verification steps
    accepted the token at that position. The counters are cumulative since
    the server started and merge at the end of each request.

    Args:
        port: The port of the server

    Returns:
        The counters, all zero when the endpoint does not answer
    """
    out = empty_metrics()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            text = r.read().decode("utf-8", "ignore")
    except (urllib.error.URLError, ConnectionError, OSError):
        return out
    per_pos: dict[int, int] = {}
    for m in METRIC_LINE.finditer(text):
        name, pos, value = m["name"], m["pos"], int(float(m["value"]))
        if name == "spec_decode_num_drafts_total":
            out["verify_steps"] = value
        elif name == "spec_decode_num_draft_tokens_total":
            out["draft_tokens"] = value
        elif name == "spec_decode_num_accepted_tokens_total":
            out["draft_accepted"] = value
        elif name == "spec_decode_num_accepted_tokens_per_pos_total" and pos is not None:
            per_pos[int(pos)] = value
    out["accepted_per_pos"] = [per_pos.get(i, 0) for i in range(len(per_pos))]
    return out


def sum_metrics(ports: list[int]) -> dict:
    """Add the speculative counters of several servers.

    Args:
        ports: The ports of the servers

    Returns:
        The summed counters
    """
    total = empty_metrics()
    for port in ports:
        m = scrape_metrics(port)
        for key in ("verify_steps", "draft_tokens", "draft_accepted"):
            total[key] += m[key]
        per_pos = m["accepted_per_pos"]
        if len(total["accepted_per_pos"]) < len(per_pos):
            total["accepted_per_pos"] += [0] * (len(per_pos) - len(total["accepted_per_pos"]))
        for i, v in enumerate(per_pos):
            total["accepted_per_pos"][i] += v
    return total


def rates_per_pos(metrics: dict) -> list[float]:
    """Give the fraction of verification steps that accepted each draft position.

    Args:
        metrics: The summed counters

    Returns:
        One rate per draft position, empty when no step ran
    """
    steps = metrics["verify_steps"]
    if not steps:
        return []
    return [round(v / steps, 4) for v in metrics["accepted_per_pos"]]


def available_gib() -> float:
    """Give the available host memory in GiB.

    Returns:
        The available memory, or a large number when it cannot be read
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 1e9


def write_table(path: Path, sinks: list[Sink], name: str, profile: dict, model: str,
                elapsed: float, state: dict, metrics: dict) -> tuple[int, int]:
    """Merge the sinks of one profile and write its proposal table.

    The file is written to a temporary name and then renamed, thus a reader
    never sees a half written table.

    Args:
        path: The table to write
        sinks: The per-server histograms of the profile
        name: The name of the profile
        profile: The mode of the profile, for the record
        model: The model path, for the record
        elapsed: The seconds the run has taken
        state: The totals of the profile
        metrics: The summed speculative counters of the servers of the profile

    Returns:
        The number of proposals and the number of distinct ids
    """
    total: Counter[int] = Counter()
    by_pos: dict[int, Counter[int]] = {}
    positions = 0
    for s in sinks:
        c, p, bp = s.snapshot()
        total.update(c)
        positions += p
        for pos, pc in bp.items():
            by_pos.setdefault(pos, Counter()).update(pc)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "model": model, "profile": name,
        "thinking": profile["thinking"], "tools": profile["tools"], "sampling": profile["sampling"],
        "prompts": state["done"], "failed": state["failed"], "generated_tokens": state["tokens"],
        "draft_n": state["draft"], "draft_accepted": state["acc"],
        "verify_steps": metrics["verify_steps"], "accepted_per_pos": metrics["accepted_per_pos"],
        "accept_rate_per_pos": rates_per_pos(metrics),
        "tool_call_responses": state["tool_calls"], "reasoning_responses": state["reasoning"],
        "elapsed": round(elapsed, 1),
        "positions": positions, "tokens": positions, "distinct": len(total),
        "positions_by_pos": {str(p): sum(c.values()) for p, c in sorted(by_pos.items())},
        "counts": {str(k): v for k, v in total.items()},
        "counts_by_pos": {str(p): {str(k): v for k, v in c.items()} for p, c in sorted(by_pos.items())},
    }))
    shutil.move(str(tmp), str(path))
    return positions, len(total)


def new_state() -> dict:
    """Give the empty totals of one profile.

    Returns:
        The counters
    """
    return {"done": 0, "failed": 0, "tokens": 0, "acc": 0, "draft": 0, "tool_calls": 0, "reasoning": 0}


def main() -> int:
    """Run the calibration harvest and write one proposal table per profile.

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True, help="llama-server of a draft-mtp build")
    ap.add_argument("--prompts", type=Path, required=True, help="user turns for the chat profiles")
    ap.add_argument("--tool-prompts", type=Path, required=True, help="user turns for the tool profiles")
    ap.add_argument("--profiles", default=",".join(PROFILES),
                    help="the modes to harvest, servers are bound to them in turn")
    ap.add_argument("--gpus", default="0,0,0,0,1,1,1,1,2,2,2,2",
                    help="one entry per server, the value is its CUDA device")
    ap.add_argument("--base-port", type=int, default=8100)
    ap.add_argument("--np", type=int, default=32, help="parallel slots per server")
    ap.add_argument("--ctx", type=int, default=0,
                    help="total context per server, 0 = np times the slot context of its profile")
    ap.add_argument("--ubatch", type=int, default=1024,
                    help="the batch and the micro batch of a server, thus one decode step never "
                         "spans two micro batches; 2048 costs about 3 GiB more host memory per server")
    ap.add_argument("--lv", type=int, default=5, help="5 emits the draft candidates")
    ap.add_argument("--spec", default="draft-mtp")
    ap.add_argument("--extra", default="-t 4 -tb 8", help="more server flags, space separated")
    ap.add_argument("--penalty-last-n", type=int, default=256,
                    help="the window of the presence penalty, kPenaltyLastN of the application")
    ap.add_argument("--draft-n-max", type=int, default=3,
                    help="the draft depth, kDraftMax of the application")
    ap.add_argument("--concurrency", type=int, default=0, help="in-flight requests per server, 0 = np")
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--checkpoint-seconds", type=float, default=300.0)
    ap.add_argument("--min-free-gib", type=float, default=25.0,
                    help="stop when the available host memory falls under this")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    ap.add_argument("--no-dump", dest="dump", action="store_false",
                    help="do not save the generated sequences to gens-<profile>.jsonl")
    ap.add_argument("--seed", type=int, default=1,
                    help="the base of the per-prompt sampler seeds, which the saved sequences record")
    a = ap.parse_args()

    names = [n for n in a.profiles.split(",") if n]
    unknown = [n for n in names if n not in PROFILES]
    if unknown:
        raise ValueError(f"unknown profile {unknown}, the profiles are {list(PROFILES)}")
    gpus = [int(g) for g in a.gpus.split(",")]
    if len(gpus) < len(names):
        raise ValueError(f"{len(gpus)} servers cannot carry {len(names)} profiles")
    ports = [a.base_port + i for i in range(len(gpus))]
    chat_prompts = load_prompts(a.prompts)
    tool_prompts = load_prompts(a.tool_prompts)
    per_server = a.concurrency or a.np

    # Server i serves profile names[i % len(names)], thus with 12 servers and
    # 4 profiles every GPU carries one server of each mode.
    bound = [names[i % len(names)] for i in range(len(gpus))]
    ports_of = {n: [p for p, b in zip(ports, bound) if b == n] for n in names}
    sinks = [Sink() for _ in gpus]
    sinks_of = {n: [s for s, b in zip(sinks, bound) if b == n] for n in names}
    state_of = {n: new_state() for n in names}
    # Each profile walks its prompt list in its own shuffled order, thus the
    # language buckets of the file interleave, and two profiles that share a
    # list do not walk the same prompts in the same sequence.
    prompts_of = {n: tool_prompts if PROFILES[n]["tools"] else chat_prompts for n in names}
    order_of = {n: list(range(len(prompts_of[n]))) for n in names}
    for i, n in enumerate(names):
        random.Random(1000 + i).shuffle(order_of[n])
    cursor_of = {n: 0 for n in names}
    tables = {n: a.out_dir / f"counts-calib-{n}.json" for n in names}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    dumps: dict[str, TextIO | None] = {
        n: (a.out_dir / f"gens-{n}.jsonl").open("a", encoding="utf-8") if a.dump else None for n in names}

    extra = [*a.extra.split(), "--spec-draft-n-max", str(a.draft_n_max)]
    ctx_of = {n: a.ctx or a.np * PROFILES[n]["slot_ctx"] for n in names}
    procs = [launch(a.model, a.bin, g, p, a.np, ctx_of[n], a.lv, a.spec, a.ubatch, extra)
             for g, p, n in zip(gpus, ports, bound)]
    readers = [threading.Thread(target=read_proposals, args=(pr, sk), daemon=True)
               for pr, sk in zip(procs, sinks)]
    for t in readers:
        t.start()

    stop = threading.Event()
    table_lock = threading.Lock()
    last_metrics: dict[str, dict] = {n: empty_metrics() for n in names}
    t0 = time.time()

    def checkpoint() -> None:
        """Write every profile table and print one line per profile.

        The lock keeps the monitor thread and the final call apart, because
        the two rename the same temporary file. The final call runs after
        the servers stopped, thus it keeps the last counters they gave.
        """
        with table_lock:
            el = time.time() - t0
            for n in names:
                dump = dumps[n]
                if dump is not None:
                    dump.flush()
                metrics = sum_metrics(ports_of[n])
                if metrics["verify_steps"] > 0:
                    last_metrics[n] = metrics
                else:
                    metrics = last_metrics[n]
                pos, dis = write_table(tables[n], sinks_of[n], n, PROFILES[n], a.model, el,
                                       state_of[n], metrics)
                st = state_of[n]
                rate = st["acc"] / st["draft"] if st["draft"] else 0.0
                print(f"calib: [{el / 60:.1f} min] {n:15s} {st['done']:6d} prompts, {st['tokens']:9d} tokens, "
                      f"{pos:9d} proposals, {dis:6d} distinct, accept {rate:.1%} "
                      f"per pos {rates_per_pos(metrics)}, "
                      f"tool calls {st['tool_calls']}, reasoning {st['reasoning']}, failed {st['failed']}",
                      flush=True)

    def monitor() -> None:
        """Write a checkpoint at an interval and watch the host memory."""
        nxt = time.time() + a.checkpoint_seconds
        while not stop.wait(10.0):
            free = available_gib()
            if free < a.min_free_gib:
                print(f"calib: RAM GUARD, only {free:.1f} GiB free, stopping the servers", flush=True)
                stop.set()
                stop_servers(procs)
                return
            if time.time() >= nxt:
                nxt = time.time() + a.checkpoint_seconds
                checkpoint()
                print(f"calib: {free:.0f} GiB free", flush=True)

    try:
        for p in ports:
            if not wait_health(p, 600):
                raise RuntimeError(f"the server on port {p} did not become ready")
        print(f"calib: {len(gpus)} servers ready, profiles {names}, np {a.np}, "
              f"{per_server * len(ports)} in flight, deadline {a.max_seconds / 60:.0f} min", flush=True)
        for n in names:
            print(f"calib: {n:15s} servers {ports_of[n]}  {PROFILES[n]['sampling']}  "
                  f"thinking={PROFILES[n]['thinking']} tools={PROFILES[n]['tools']} "
                  f"gen={PROFILES[n]['gen']} ctx={ctx_of[n]}", flush=True)

        threading.Thread(target=monitor, daemon=True).start()
        deadline = t0 + a.max_seconds
        inflight: dict = {}
        busy = {p: 0 for p in ports}
        with ThreadPoolExecutor(max_workers=per_server * len(ports)) as pool:
            while not stop.is_set() and time.time() < deadline:
                # Every server keeps its own slots full. One global limit would
                # let the slow profiles queue while the fast profiles idle.
                for port, n in zip(ports, bound):
                    while busy[port] < per_server and not stop.is_set() and time.time() < deadline:
                        order = order_of[n]
                        index = order[cursor_of[n] % len(order)]
                        prompt = prompts_of[n][index]
                        # A prompt that comes around again takes a new seed.
                        seed = prompt_seed(a.seed + cursor_of[n] // len(order), n, index)
                        cursor_of[n] += 1
                        fut = pool.submit(complete, port, prompt, PROFILES[n], a.penalty_last_n, seed)
                        inflight[fut] = (n, port)
                        busy[port] += 1
                done, _ = wait(list(inflight), timeout=5.0, return_when=FIRST_COMPLETED)
                for f in done:
                    n, port = inflight.pop(f)
                    busy[port] -= 1
                    account(f.result(), state_of[n], dumps[n])
            print("calib: deadline reached, draining in-flight requests", flush=True)
            for f in wait(list(inflight), timeout=600.0).done:
                n = inflight.pop(f)[0]
                account(f.result(), state_of[n], dumps[n])
            # A request that hangs past the drain fails at once when its
            # server stops, thus the pool exits without the full HTTP timeout.
            stop_servers(procs)
    finally:
        stop.set()
        stop_servers(procs)
        for t in readers:
            t.join(timeout=30)

    checkpoint()
    for d in dumps.values():
        if d is not None:
            d.close()
    elapsed = time.time() - t0
    total_tokens = sum(s["tokens"] for s in state_of.values())
    print(f"calib: DONE in {elapsed / 60:.1f} min, {total_tokens} tokens "
          f"({total_tokens / elapsed:.0f} tok/s), tables {[str(t) for t in tables.values()]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
