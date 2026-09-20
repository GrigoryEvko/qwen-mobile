"""Verify that the decode and the MTP drafter of one llama-server build are self-consistent.

Every test runs greedy, thus a difference between two runs comes from the
build or from its numerics, and not from sampling. The tests are:

T1 batching. A prompt decoded alone and decoded while 31 other slots are
   busy gives the same tokens. A difference means that the batched kernels
   change the argmax, or that one slot reads the state of another slot.
T2 slot reuse. A prompt decoded in a slot after a different prompt used the
   slot gives the same tokens as a decode in a fresh slot. A difference means
   that the recurrent state of the previous request leaks into the next one.
T3 speculation. A decode with the MTP drafter gives the same tokens as a
   plain decode. A difference means that the verification batch or the state
   rollback changes what the target produces.
T4 drafter. The fraction of verification steps that accept each draft depth,
   on free text and on formulaic text, from the counters of the server. A
   broken head gives near zero at every depth. A healthy head gives about
   0.5 at depth 0 on free text and about 0.9 on formulaic text.
T5 reference. The CPU build of the same binary decodes a few prompts, and
   the CUDA decode must give the same tokens. The CPU path is the reference
   of the CUDA kernels.
T6 paired seeds. The tests T1, T3 and T5 again, at the sampler of the
   application, with one seed per prompt that is the same in the two arms
   of each pair, over several rounds. The sampler draws one random number
   per emitted token, and the verify loop of the speculative decode samples
   position by position and stops at the first token that differs from the
   draft, thus emitted token k comes from draw k in both arms. Identical
   distributions give identical strings, and a divergence needs the draw to
   land inside the probability shift between the arms. A seed shared by all
   prompts would replay one sequence of draws for the whole set, thus each
   prompt takes its own seed.
T7 CPU pair (--cpu-pair). A plain decode against a speculative decode, both
   on the CPU path of the same binary, greedy and with paired seeds. The CPU
   batched and single-token paths accumulate in fp32 and agree to about
   1e-4, thus a divergence here comes from the speculative path itself
   (the state rollback, the penalty window, the draw stream) and not from
   the kernels. This test tells a defect from the CUDA rounding of T6.

A greedy decode on a GPU is deterministic only when the batch composition is
the same, because the CUDA kernels select their path by batch size and the
sums differ in the last bits. Thus a divergence in T1, T3 and T5 is expected
at a low rate, and the report tells a rounding flip from a defect by the
probabilities at the first different token: a flip happens where the two top
candidates are near a tie, and the probability of each token is almost the
same in the two runs. A defect gives a large margin and a large drift.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from calib_server import load_prompts, post, prompt_seed, scrape_metrics, stop_servers, wait_health

# n_probs gives the top candidates of every generated token before sampling,
# thus the report can read the margin at a divergence.
GREEDY = {
    "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": 1,
    "n_probs": 4,
}

# The sampler chain of the application with thinking off: rebuild_sampler in
# llama_jni.cpp with the defaults of Settings.kt. The seed comes per request.
APP_SAMPLER = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repeat_last_n": 256, "repeat_penalty": 1.0, "frequency_penalty": 0.0,
    "n_probs": 4,
}

FORMULAIC = [
    "List the numbers from 1 to 400, separated by commas, on one line.",
    "Write the English alphabet in lowercase, then in uppercase, ten times each, one per line.",
    "Print a Python list of the first 120 multiples of 7.",
    "Напиши по порядку названия месяцев года, каждый по десять раз, через запятую.",
]


def launch(binary: str, model: str, gpu: int, port: int, spec: str, np: int, ctx: int,
           ngl: int, threads: int, draft: int = 3, lv: int = 1,
           log: Path | None = None) -> subprocess.Popen:
    """Launch one llama-server, with its output discarded or written to a file.

    Args:
        binary: The llama-server of the build under test
        model: The GGUF with the full MTP head
        gpu: The CUDA device index
        port: The port to listen on
        spec: The speculation type, none or draft-mtp
        np: The number of parallel slots
        ctx: The total context
        ngl: The number of layers on the GPU, 0 for the CPU reference
        threads: The CPU threads, which matter for the CPU reference only
        draft: The draft depth
        lv: The log verbosity
        log: The file that takes the output, or None to discard it

    Returns:
        The server process
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [binary, "-m", model, "--spec-type", spec, "-ngl", str(ngl), "-fa", "on",
           "-np", str(np), "-c", str(ctx), "--host", "127.0.0.1", "--port", str(port),
           "-lv", str(lv), "--metrics", "-t", str(threads), "-tb", str(threads),
           "--spec-draft-n-max", str(draft)]
    out = log.open("wb") if log is not None else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)


def render(port: int, user: str) -> str:
    """Render one user turn through the template of the model, thinking off.

    Args:
        port: The port of a server
        user: The user turn

    Returns:
        The prompt text
    """
    body = {"messages": [{"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}}
    text = post(port, "/apply-template", body, 60.0).get("prompt")
    if not text:
        raise RuntimeError(f"the server on port {port} did not render the template")
    return text


def greedy(port: int, text: str, n: int) -> dict:
    """Decode one prompt greedily and give the tokens, the text and the timings.

    Args:
        port: The port of the server
        text: The rendered prompt
        n: The number of tokens to generate

    Returns:
        The parsed response, with a tokens list, or an empty dict on failure
    """
    return post(port, "/completion", {"prompt": text, "n_predict": n, "cache_prompt": False,
                                      "return_tokens": True, **GREEDY}, 1800.0)


def sampled(port: int, text: str, n: int, seed: int) -> dict:
    """Decode one prompt with the sampler of the application and one seed.

    Args:
        port: The port of the server
        text: The rendered prompt
        n: The number of tokens to generate
        seed: The seed of the sampler

    Returns:
        The parsed response, with a tokens list, or an empty dict on failure
    """
    return post(port, "/completion", {"prompt": text, "n_predict": n, "cache_prompt": False,
                                      "return_tokens": True, "seed": seed, **APP_SAMPLER}, 1800.0)


def first_diff(a: list[int], b: list[int]) -> int:
    """Give the index of the first different token, or -1 when the lists agree.

    Args:
        a: One token list
        b: The other token list

    Returns:
        The index, or -1
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def probs_at(resp: dict, i: int) -> dict[int, float]:
    """Give the top candidates of generated position i as id to probability.

    Args:
        resp: The response of a decode with n_probs
        i: The generated position

    Returns:
        The probabilities of the top candidates, empty when the response has none
    """
    cp = resp.get("completion_probabilities") or []
    if i >= len(cp):
        return {}
    out: dict[int, float] = {}
    for x in cp[i].get("top_logprobs") or cp[i].get("top_probs") or []:
        p = math.exp(float(x["logprob"])) if "logprob" in x else float(x.get("prob", 0.0))
        out[int(x["id"])] = p
    return out


def drifts(left: dict, right: dict, until: int) -> list[float]:
    """Give the probability differences of the chosen token over the agreeing positions.

    Args:
        left: One decode
        right: The other decode of the same prompt
        until: The first different position, or the length when the two agree

    Returns:
        One difference per agreeing position where both runs report the
        probability of the token they chose
    """
    out: list[float] = []
    for i in range(until):
        tok = left.get("tokens", [])[i]
        pl, pr = probs_at(left, i), probs_at(right, i)
        if tok in pl and tok in pr:
            out.append(abs(pl[tok] - pr[tok]))
    return out


def quantiles(values: list[float]) -> str:
    """Give the median, the 95th percentile and the maximum as text.

    Args:
        values: The samples

    Returns:
        The three numbers, or a note when there is no sample
    """
    if not values:
        return "no sample"
    q = sorted(values)
    p95 = q[min(len(q) - 1, int(0.95 * len(q)))]
    return f"median {q[len(q) // 2]:.4f}, p95 {p95:.4f}, max {q[-1]:.4f} over {len(q)}"


def compare(title: str, left: list[dict], right: list[dict], left_name: str, right_name: str,
            examples: int = 3) -> tuple[int, int, int]:
    """Print how two lists of decodes of the same prompts agree.

    At each first divergence the report gives the probability that each run
    assigned to the two tokens, thus a near tie is visible as such.

    Args:
        title: The name of the test
        left: The responses of one run
        right: The responses of the other run, in the same prompt order
        left_name: The label of the first run
        right_name: The label of the second run
        examples: How many divergences to print in full

    Returns:
        The identical count, the pair count, and the tokens that agreed
        before a divergence or to the end
    """
    same = 0
    agreed = 0
    firsts = []
    margins = []
    all_drift: list[float] = []
    for i, (one, other) in enumerate(zip(left, right)):
        lt, rt = one.get("tokens", []), other.get("tokens", [])
        d = first_diff(lt, rt)
        all_drift.extend(drifts(one, other, len(lt) if d < 0 else d))
        if d < 0:
            same += 1
            agreed += len(lt)
            continue
        agreed += d
        firsts.append(d)
        pl, pr = probs_at(one, d), probs_at(other, d)
        a, b = lt[d], rt[d]
        margin = abs(pl.get(a, 0.0) - pl.get(b, 0.0))
        margins.append(round(margin, 3))
        if len(firsts) <= examples:
            print(f"   prompt {i}: first different token at {d} of {len(lt)}: "
                  f"{left_name} took {a} with p {pl.get(a, 0.0):.3f} (p {pr.get(a, 0.0):.3f} in {right_name}), "
                  f"{right_name} took {b} with p {pr.get(b, 0.0):.3f} (p {pl.get(b, 0.0):.3f} in {left_name})")
    line = (f"{title}: {same}/{len(left)} identical ({left_name} vs {right_name}), "
            f"drift of the chosen token's probability before a divergence: {quantiles(all_drift)}")
    if firsts:
        line += f", first differences at {sorted(firsts)}, margins at them {sorted(margins)}"
    print(line, flush=True)
    return same, len(left), agreed


def paired(title: str, left_port: int, right_port: int, texts: list[str], n: int, rounds: list[int],
           left_name: str, right_name: str, loaders: list[str] | None = None) -> None:
    """Decode the prompts in two arms with one seed per prompt, over several rounds.

    The seed of a prompt comes from the round and the prompt index, thus it
    differs between prompts and is the same in the two arms of a pair. The
    right arm runs under load when loaders is given: those prompts hold the
    other slots of its server while the test prompts decode.

    Args:
        title: The name of the test
        left_port: The server of the first arm
        right_port: The server of the second arm
        texts: The rendered prompts
        n: The tokens to generate
        rounds: The base seeds, one pair of runs per round
        left_name: The label of the first arm
        right_name: The label of the second arm
        loaders: The rendered filler prompts for the right arm, or None
    """
    same = pairs = agreed = 0
    for base in rounds:
        seeds = [prompt_seed(base, title, i) for i in range(len(texts))]
        left = [sampled(left_port, t, n, s) for t, s in zip(texts, seeds)]
        if loaders:
            with ThreadPoolExecutor(max_workers=len(loaders) + len(texts)) as pool:
                fill = [pool.submit(sampled, right_port, t, 256, prompt_seed(base, "load", i))
                        for i, t in enumerate(loaders)]
                time.sleep(2.0)
                right = list(pool.map(lambda ts: sampled(right_port, ts[0], n, ts[1]), zip(texts, seeds)))
                for f in fill:
                    f.result()
        else:
            right = [sampled(right_port, t, n, s) for t, s in zip(texts, seeds)]
        s, p, a = compare(f"{title} round {base}", left, right, left_name, right_name, examples=1)
        same, pairs, agreed = same + s, pairs + p, agreed + a
    total = pairs * n
    print(f"{title}: {same}/{pairs} identical over {len(rounds)} rounds, "
          f"{agreed}/{total} tokens agreed before any divergence, "
          f"mean run before the first difference {agreed / pairs if pairs else 0:.1f} tokens", flush=True)


def acceptance(port: int, texts: list[str], n: int) -> tuple[list[float], float, list[dict]]:
    """Decode prompts with the drafter and give the acceptance per draft depth.

    The counters of the server are cumulative, thus the function reads them
    before and after and takes the difference.

    Args:
        port: The port of the speculative server
        texts: The rendered prompts
        n: The tokens to generate per prompt

    Returns:
        The rate per depth, the overall rate from the timings, and the responses
    """
    before = scrape_metrics(port)
    resps = [greedy(port, t, n) for t in texts]
    after = scrape_metrics(port)
    steps = after["verify_steps"] - before["verify_steps"]
    earlier = before["accepted_per_pos"] or [0] * len(after["accepted_per_pos"])
    per_pos = [a - b for a, b in zip(after["accepted_per_pos"], earlier)]
    rates = [round(v / steps, 3) for v in per_pos] if steps else []
    drafted = sum(int((r.get("timings") or {}).get("draft_n", 0)) for r in resps)
    accepted = sum(int((r.get("timings") or {}).get("draft_n_accepted", 0)) for r in resps)
    return rates, (accepted / drafted if drafted else 0.0), resps


def cpu_pair(a: argparse.Namespace, prompts: list[str], seeds: list[int]) -> int:
    """Run T7: a plain and a speculative decode on the CPU path, greedy and paired.

    Args:
        a: The parsed arguments
        prompts: The user turns to decode
        seeds: The base seeds of the paired rounds

    Returns:
        The exit status
    """
    p_plain, p_spec = a.base_port, a.base_port + 1
    procs = [
        launch(a.bin, a.model, a.gpu, p_plain, "none", 1, 4096, 0, a.cpu_threads),
        launch(a.bin, a.model, a.gpu, p_spec, "draft-mtp", 1, 4096, 0, a.cpu_threads),
    ]
    try:
        for port in (p_plain, p_spec):
            if not wait_health(port, 900):
                raise RuntimeError(f"the server on port {port} did not become ready")
        print("verify: 2 CPU servers ready (plain np1, mtp np1)", flush=True)
        texts = [render(p_plain, t) for t in prompts]
        t0 = time.time()
        plain = [greedy(p_plain, t, a.cpu_gen) for t in texts]
        spec = [greedy(p_spec, t, a.cpu_gen) for t in texts]
        print(f"verify: greedy pair in {time.time() - t0:.0f} s", flush=True)
        compare("T7 cpu greedy   ", plain, spec, "plain", "mtp")
        t0 = time.time()
        paired("T7 cpu paired   ", p_plain, p_spec, texts, a.cpu_gen, seeds, "plain", "mtp")
        print(f"verify: paired rounds in {time.time() - t0:.0f} s", flush=True)
        rates, overall, _ = acceptance(p_spec, texts, a.cpu_gen)
        print(f"T7 cpu drafter: per depth {rates}, overall {overall:.3f}", flush=True)
    finally:
        stop_servers(procs)
    return 0


def cpu_probe(a: argparse.Namespace, prompts: list[str], fillers: list[str]) -> int:
    """Run T8: locate the source of the CPU divergence of the speculative path.

    Four CPU servers: plain np1, plain np4 for the busy control, speculative
    at depth 3 with its log kept, and speculative at depth 1. Greedy only.

    Args:
        a: The parsed arguments
        prompts: The user turns to decode
        fillers: The user turns that hold the other slots of the busy server

    Returns:
        The exit status
    """
    p_plain, p_busy, p_d3, p_d1 = (a.base_port + i for i in range(4))
    log = Path(a.log_dir) / "cpu-mtp-depth3.log"
    procs = [
        launch(a.bin, a.model, a.gpu, p_plain, "none", 1, 4096, 0, a.cpu_threads),
        launch(a.bin, a.model, a.gpu, p_busy, "none", 4, 16384, 0, a.cpu_threads),
        launch(a.bin, a.model, a.gpu, p_d3, "draft-mtp", 1, 4096, 0, a.cpu_threads, draft=3, lv=5, log=log),
        launch(a.bin, a.model, a.gpu, p_d1, "draft-mtp", 1, 4096, 0, a.cpu_threads, draft=1),
    ]
    try:
        for port in (p_plain, p_busy, p_d3, p_d1):
            if not wait_health(port, 900):
                raise RuntimeError(f"the server on port {port} did not become ready")
        print("verify: 4 CPU servers ready (plain np1, plain np4, mtp depth 3, mtp depth 1)", flush=True)
        texts = [render(p_plain, t) for t in prompts]
        plain = [greedy(p_plain, t, a.cpu_gen) for t in texts]

        # (a) the control: the same prompts while three fillers hold the other slots.
        with ThreadPoolExecutor(max_workers=8) as pool:
            fill = [pool.submit(greedy, p_busy, render(p_plain, f), 256) for f in fillers[:3]]
            time.sleep(2.0)
            busy = [greedy(p_busy, t, a.cpu_gen) for t in texts]
            for f in fill:
                f.result()
        compare("T8 cpu alone vs busy   ", plain, busy, "alone", "busy")

        # (b) a rollback of at most one token.
        d1 = [greedy(p_d1, t, a.cpu_gen) for t in texts]
        compare("T8 cpu plain vs depth 1", plain, d1, "plain", "mtp1")

        # (c) depth 3 on formulaic prompts, where a rejection is rare.
        form_texts = [render(p_plain, u) for u in FORMULAIC]
        form_plain = [greedy(p_plain, t, a.cpu_gen) for t in form_texts]
        form_d3 = [greedy(p_d3, t, a.cpu_gen) for t in form_texts]
        compare("T8 cpu formulaic depth 3", form_plain, form_d3, "plain", "mtp3")

        # (d) depth 3 on the free prompts again, with the log of the server kept.
        d3 = [greedy(p_d3, t, a.cpu_gen) for t in texts]
        compare("T8 cpu plain vs depth 3", plain, d3, "plain", "mtp3")
    finally:
        stop_servers(procs)
    text = log.read_text(errors="ignore") if log.exists() else ""
    restored = text.count("restoring speculative checkpoint")
    accepted = sum(1 for line in text.splitlines() if "accepted" in line and "draft tokens" in line)
    print(f"T8 depth-3 server log: {restored} checkpoint restores, {accepted} accept lines, "
          f"{len(text.splitlines())} lines in {log}", flush=True)
    for line in text.splitlines():
        if "seq_rm" in line or "rollback" in line.lower() or "n_rs_seq" in line:
            print("   ", line.strip()[:160])
            break
    return 0


def tokenize(port: int, text: str) -> list[int]:
    """Give the token ids of a rendered prompt, from the server's own tokenizer.

    Args:
        port: The port of a server
        text: The rendered prompt

    Returns:
        The ids, with the special tokens parsed
    """
    body = {"content": text, "add_special": False, "with_pieces": False}
    return post(port, "/tokenize", body, 60.0).get("tokens", [])


def cpu_split(a: argparse.Namespace, prompts: list[str]) -> int:
    """Run T9: the multi-token path against the single-token path with no speculation.

    The prompt and the first k tokens of a plain greedy decode go through
    one batched prefill, then the server predicts the next tokens one at a
    time. The plain decode produced those k tokens one at a time. A
    different next token at the same margins as T7 shows that the batched
    kernels, not the speculative loop, move the logits.

    Args:
        a: The parsed arguments
        prompts: The user turns to decode

    Returns:
        The exit status
    """
    port = a.base_port
    procs = [launch(a.bin, a.model, a.gpu, port, "none", 1, 4096, 0, a.cpu_threads)]
    try:
        if not wait_health(port, 900):
            raise RuntimeError(f"the server on port {port} did not become ready")
        texts = [render(port, t) for t in prompts]
        plain = [greedy(port, t, a.cpu_gen) for t in texts]
        splits = [8, 16, 24, 32, 48, 64, 96]
        same = total = 0
        margins: list[float] = []
        drift_all: list[float] = []
        for i, (text, resp) in enumerate(zip(texts, plain)):
            ids = tokenize(port, text)
            toks = resp.get("tokens", [])
            for k in splits:
                if k + 4 > len(toks):
                    continue
                again = post(port, "/completion", {"prompt": ids + toks[:k], "n_predict": 4,
                                                   "cache_prompt": False, "return_tokens": True, **GREEDY}, 600.0)
                got = again.get("tokens", [])
                total += 1
                pl, pr = probs_at(resp, k), probs_at(again, 0)
                for tok in (toks[k],):
                    if tok in pl and tok in pr:
                        drift_all.append(abs(pl[tok] - pr[tok]))
                if got[:1] == toks[k:k + 1]:
                    same += 1
                    continue
                b = got[0] if got else -1
                margins.append(round(abs(pl.get(toks[k], 0.0) - pl.get(b, 0.0)), 3))
                print(f"   prompt {i} split {k}: single path took {toks[k]} with p {pl.get(toks[k], 0.0):.3f}, "
                      f"batched prefill took {b} with p {pr.get(b, 0.0):.3f} "
                      f"(p {pl.get(b, 0.0):.3f} in the single path)", flush=True)
        print(f"T9 cpu prefill vs single: {same}/{total} next tokens identical, margins at the differences "
              f"{sorted(margins)}, drift of the chosen token's probability {quantiles(drift_all)}", flush=True)
    finally:
        stop_servers(procs)
    return 0


def main() -> int:
    """Run the tests and print the report.

    Returns:
        The exit status
    """
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--base-port", type=int, default=8600)
    ap.add_argument("--n-test", type=int, default=12, help="prompts per comparison")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--cpu-threads", type=int, default=32)
    ap.add_argument("--cpu-prompts", type=int, default=4)
    ap.add_argument("--cpu-gen", type=int, default=64)
    ap.add_argument("--seeds", default="1,2,3",
                    help="the base seeds of the paired sampled test, each prompt derives its own from them")
    ap.add_argument("--greedy", action="store_true", help="run the greedy tests T1, T3 and T5 as well")
    ap.add_argument("--cpu-pair", action="store_true",
                    help="run only T7: plain against speculative on the CPU path, greedy and paired")
    ap.add_argument("--cpu-probe", action="store_true",
                    help="run only T8: locate the CPU divergence (busy control, depth 1, formulaic)")
    ap.add_argument("--cpu-split", action="store_true",
                    help="run only T9: a batched prefill of the plain decode against its single-token path")
    ap.add_argument("--log-dir", default=".", help="where T8 keeps the server log")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]

    rng = random.Random(11)
    pool_prompts = [p for p in load_prompts(a.prompts) if isinstance(p, str)]
    rng.shuffle(pool_prompts)
    tests = pool_prompts[:a.n_test]
    fillers = pool_prompts[a.n_test:a.n_test + 24]
    long_prompt = pool_prompts[a.n_test + 24]

    if a.cpu_pair:
        return cpu_pair(a, tests[:a.cpu_prompts], seeds)
    if a.cpu_probe:
        return cpu_probe(a, tests[:a.cpu_prompts], fillers)
    if a.cpu_split:
        return cpu_split(a, tests[:a.cpu_prompts])

    p_plain, p_batch, p_spec, p_cpu = (a.base_port + i for i in range(4))
    procs = [
        launch(a.bin, a.model, a.gpu, p_plain, "none", 1, 8192, 99, 4),
        launch(a.bin, a.model, a.gpu, p_batch, "none", 32, 65536, 99, 4),
        launch(a.bin, a.model, a.gpu, p_spec, "draft-mtp", 1, 8192, 99, 4),
        launch(a.bin, a.model, a.gpu, p_cpu, "none", 1, 4096, 0, a.cpu_threads),
    ]
    try:
        for port in (p_plain, p_batch, p_spec, p_cpu):
            if not wait_health(port, 900):
                raise RuntimeError(f"the server on port {port} did not become ready")
        print("verify: 4 servers ready (plain np1, plain np32, mtp np1, cpu np1)", flush=True)
        texts = [render(p_plain, t) for t in tests]
        filler_texts = [render(p_plain, t) for t in fillers]
        long_text = render(p_plain, long_prompt)

        t0 = time.time()
        plain = [greedy(p_plain, t, a.gen) for t in texts]
        batch_alone = [greedy(p_batch, t, a.gen) for t in texts]
        print(f"verify: baselines in {time.time() - t0:.0f} s, "
              f"{sum(len(r.get('tokens', [])) for r in plain)} tokens", flush=True)
        print("   sample:", plain[0].get("content", "")[:160].replace("\n", " "), flush=True)

        if a.greedy:
            # T1: the same prompts while 24 fillers of 256 tokens hold the other slots.
            with ThreadPoolExecutor(max_workers=40) as pool:
                filler_futs = [pool.submit(greedy, p_batch, t, 256) for t in filler_texts]
                time.sleep(2.0)
                loaded = list(pool.map(lambda t: greedy(p_batch, t, a.gen), texts))
                for f in filler_futs:
                    f.result()
            compare("T1 batching   ", batch_alone, loaded, "alone", "loaded")
        compare("T1 np1 vs np32", plain, batch_alone, "np1", "np32")

        # T2: on the single-slot server every request reuses slot 0. A long
        # different prompt between two decodes of the same prompt must not
        # change the second decode. The np32 server gave each of its first
        # requests a fresh slot, thus batch_alone is the fresh reference.
        reused = []
        for t in texts[:6]:
            greedy(p_plain, long_text, 400)
            reused.append(greedy(p_plain, t, a.gen))
        compare("T2 slot reuse ", batch_alone[:6], reused, "fresh", "reused")
        compare("T2 same slot  ", plain[:6], reused, "first", "reused")

        if a.greedy:
            # T3: the drafter must not change the greedy output.
            spec = [greedy(p_spec, t, a.gen) for t in texts]
            compare("T3 speculation", plain, spec, "plain", "mtp")

        # T4: acceptance per depth on free text, then on formulaic text.
        free_rates, free_all, _ = acceptance(p_spec, texts[:8], 256)
        form_texts = [render(p_plain, u) for u in FORMULAIC]
        form_rates, form_all, form_resps = acceptance(p_spec, form_texts, 256)
        print(f"T4 drafter: free text per depth {free_rates}, overall {free_all:.3f}; "
              f"formulaic per depth {form_rates}, overall {form_all:.3f}", flush=True)
        print("   formulaic sample:", form_resps[0].get("content", "")[:120].replace("\n", " "), flush=True)

        if a.greedy:
            # T5: the CPU path of the same binary is the reference of the CUDA kernels.
            t0 = time.time()
            cpu = [greedy(p_cpu, t, a.cpu_gen) for t in texts[:a.cpu_prompts]]
            cuda_short = [{"tokens": r.get("tokens", [])[:a.cpu_gen], "content": r.get("content", "")}
                          for r in plain[:a.cpu_prompts]]
            print(f"verify: cpu reference in {time.time() - t0:.0f} s", flush=True)
            compare("T5 cpu vs cuda", cpu, cuda_short, "cpu", "cuda")

        # T6: the same comparisons at the sampler of the application, with the
        # seed shared by the two arms of each pair.
        t0 = time.time()
        paired("T6 plain vs mtp ", p_plain, p_spec, texts, a.gen, seeds, "plain", "mtp")
        paired("T6 alone vs busy", p_batch, p_batch, texts, a.gen, seeds, "alone", "busy", loaders=filler_texts)
        paired("T6 cpu vs cuda  ", p_cpu, p_plain, texts[:a.cpu_prompts], a.cpu_gen, seeds[:1], "cpu", "cuda")
        print(f"verify: paired seeds in {time.time() - t0:.0f} s", flush=True)
    finally:
        stop_servers(procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
