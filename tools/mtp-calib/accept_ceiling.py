#!/usr/bin/env python3
"""Measure the ceiling of the naive speculative acceptance rule.

The drafter takes the top-1 token of the multi-token prediction head and never
stops early, because p_min is 0. The verifier samples the target and keeps the
draft token only when the two are equal (`common_sampler_sample_and_accept_n`,
common/sampling.cpp:733). Thus the acceptance of position 1 cannot be higher
than the probability that the target draws its own top token.

This program measures that probability. It asks a running llama-server for the
per-position candidate probabilities, which the server reports after the whole
sampler chain, and it averages the top one over every generated position.

A second pass with the presence penalty off separates the cost of that penalty
from the cost of the temperature.

Run this against an IDLE server. A server that carries the harvest on 32 slots
returns about one position for each request instead of n_predict, thus the
average is taken over a handful of first tokens and it is not the ceiling. The
first tokens of an answer are the most predictable ones, thus a busy server
biases the number upward. Start one server with -np 1 after the harvest ends.

Complexity: O(n_prompts * n_predict) requests of one token each on the server.
"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request

# The two sampler rows of the application, from tools/mtp-calib/calib_server.py.
INSTRUCT = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repeat_penalty": 1.0, "frequency_penalty": 0.0,
}
THINKING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repeat_penalty": 1.0, "frequency_penalty": 0.0,
}

PROMPTS = [
    "Explain why the sky is blue.",
    "Write a short function that reverses a linked list in C.",
    "What is the difference between a mutex and a semaphore?",
    "Summarise the causes of the 1929 stock market crash.",
    "Give three ways to reduce the memory of a Python program.",
    "Describe how a gated delta network differs from attention.",
    "What should I cook with chickpeas, lemon and spinach?",
    "Translate into Russian: the train leaves at nine in the morning.",
]


def post(port: int, path: str, body: dict) -> dict:
    """Send one JSON request to the server and give the parsed answer.

    Args:
        port: The port of the llama-server
        path: The path of the endpoint, for example /completion
        body: The request object

    Returns:
        The parsed response object
    """
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def render(port: int, text: str, thinking: bool) -> str:
    """Render one user turn through the chat template of the checkpoint.

    Args:
        port: The port of the llama-server
        text: The user message
        thinking: True to leave the think block open

    Returns:
        The rendered prompt
    """
    body = {
        "messages": [{"role": "user", "content": text}],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    return post(port, "/apply-template", body)["prompt"]


def top_probs(port: int, prompt: str, sampling: dict, n_predict: int, seed: int) -> list[float]:
    """Give the probability of the top candidate at each generated position.

    Args:
        port: The port of the llama-server
        prompt: The rendered prompt
        sampling: The sampler fields of the request
        n_predict: The tokens to generate
        seed: The seed of the sampler

    Returns:
        One probability for each generated position
    """
    body = dict(sampling)
    body.update({"prompt": prompt, "n_predict": n_predict, "n_probs": 1,
                 "seed": seed, "cache_prompt": False, "stream": False})
    out = post(port, "/completion", body)
    probs: list[float] = []
    for entry in out.get("completion_probabilities", []):
        # The server gives either top_logprobs (current) or probs (earlier).
        cand = entry.get("top_logprobs") or entry.get("probs") or []
        if not cand:
            continue
        first = cand[0]
        if "prob" in first:
            probs.append(float(first["prob"]))
        elif "logprob" in first:
            probs.append(float(2.718281828459045 ** first["logprob"]))
    return probs


def run(port: int, label: str, sampling: dict, thinking: bool, n_predict: int) -> None:
    """Measure and print the ceiling for one sampler configuration.

    Args:
        port: The port of the llama-server
        label: The name of the configuration
        sampling: The sampler fields of the request
        thinking: True to leave the think block open
        n_predict: The tokens to generate for each prompt
    """
    every: list[float] = []
    for i, text in enumerate(PROMPTS):
        prompt = render(port, text, thinking)
        every.extend(top_probs(port, prompt, sampling, n_predict, 1000 + i))
    if not every:
        print(f"{label:28s} no candidate probabilities came back")
        return
    every.sort()
    mean = statistics.fmean(every)
    print(f"{label:28s} positions {len(every):5d}  mean top-1 p {mean:.4f}  "
          f"median {statistics.median(every):.4f}  "
          f"p10 {every[len(every) // 10]:.4f}  p90 {every[9 * len(every) // 10]:.4f}")


def main() -> None:
    """Parse the arguments and run each configuration."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--n-predict", type=int, default=64)
    args = ap.parse_args()

    print("The ceiling of the naive rule is the mean top-1 probability below.")
    print("Position 1 of the draft cannot be accepted more often than that.")
    print()
    no_penalty_instruct = dict(INSTRUCT, presence_penalty=0.0)
    no_penalty_thinking = dict(THINKING, presence_penalty=0.0)
    greedy = dict(INSTRUCT, temperature=0.0)
    run(args.port, "instruct, as shipped", INSTRUCT, False, args.n_predict)
    run(args.port, "instruct, no presence penalty", no_penalty_instruct, False, args.n_predict)
    run(args.port, "instruct, temperature 0", greedy, False, args.n_predict)
    run(args.port, "thinking, as shipped", THINKING, True, args.n_predict)
    run(args.port, "thinking, no presence penalty", no_penalty_thinking, True, args.n_predict)


if __name__ == "__main__":
    main()
