"""The driver of run.sh: run the targets of one mode, one sanitizer and one profile, and write results.jsonl.

    uv run python tests/fuzz/quant/qfz_run.py <test|fuzz> <none|asan|ubsan|tsan|msan> \\
        --profile <debug|release> [--budget-seconds N] [--jobs N]

run.sh builds the native code first and sets FUZZ_SANITIZER and
FUZZ_PROFILE. Each target writes one JSON line to
build/fuzz/quant-<profile>-<config>/results.jsonl, and a copy of the line
to build/fuzz/quant-<config>/results.jsonl:

    {area, target, sanitizer, profile, mode, seconds, executions, findings, crash_files}

The exit status is 1 when a target has a finding, 3 when a target could
not run (a missing build), and 0 otherwise.

The targets. The Python targets run only in the configuration "none",
because quant/ holds no native code of ours, and only in the release pass,
because a profile changes only native code. The regression tests run in
the two passes (rule R13):

- grid, layout, transform, solver, export, reader: Hypothesis fuzzers.
  test replays the explicit examples and the database; fuzz generates in
  rounds with new seeds until the budget ends, and stops at a failure.
- regressions: the minimal example of each finding. Each open finding (a
  strict xfail) counts as a finding with its evidence file (rule R8).
- atheris-reader, atheris-pack: coverage-guided (libFuzzer) Python fuzzers.

The native targets run in each configuration, with the build of that one
sanitizer:

- loader-check: qfz-gguf-check on the seed files and the phone set (test),
  and on the files that the reader and export fuzzers corrupt and write
  (fuzz). Each distinct sanitizer report of the ggml loader is a finding.
- llama-toy: llama-perplexity on toy models that export() writes, against
  the KL base of the x86 oracle build.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from qfz_common import (  # noqa: E402
    FUZZ_OUT,
    KNOWN_FINDINGS,
    ORACLE_BIN,
    PROFILES,
    REGRESS_DIR,
    ROOT,
    SANITIZERS,
    SEED_DIR,
    san_dir,
)

LOG = logging.getLogger("qfz.run")
PHONE = FUZZ_OUT / "phone"
_WRITE_LOCK = threading.Lock()


@dataclasses.dataclass
class Target:
    """One target: its name, its kind, its file, and the examples of one fuzz round."""

    name: str
    kind: str
    file: str = ""
    examples: int = 0
    seeds: str = ""


PYTHON_TARGETS = [
    Target("grid", "pytest", "test_qfz_grid.py", 1000),
    Target("layout", "pytest", "test_qfz_layout.py", 3000),
    Target("transform", "pytest", "test_qfz_transform.py", 400),
    Target("solver", "pytest", "test_qfz_solver.py", 800),
    Target("export", "pytest", "test_qfz_export.py", 300),
    Target("reader", "pytest", "test_qfz_reader.py", 2000),
    Target("regressions", "regress", "regress/test_qfz_regressions.py"),
    Target("atheris-reader", "atheris", "atheris_qfz_reader.py", seeds="reader"),
    Target("atheris-pack", "atheris", "atheris_qfz_pack.py", seeds="pack"),
]
NATIVE_TARGETS = [Target("loader-check", "loader"), Target("llama-toy", "llama")]


@dataclasses.dataclass
class Result:
    """The result line of one target."""

    target: str
    sanitizer: str
    profile: str
    mode: str
    seconds: float = 0.0
    executions: int = 0
    findings: int = 0
    crash_files: list[str] = dataclasses.field(default_factory=list)
    note: str = ""

    def line(self) -> str:
        """Give the JSON line of results.jsonl."""
        row = {"area": "quant", "target": self.target, "sanitizer": self.sanitizer, "profile": self.profile,
               "mode": self.mode, "seconds": round(self.seconds, 1), "executions": self.executions,
               "findings": self.findings, "crash_files": self.crash_files}
        if self.note:
            row["note"] = self.note
        return json.dumps(row)


class Runner:
    """Run the targets of one mode, one sanitizer and one profile."""

    def __init__(self, mode: str, sanitizer: str, profile: str, budget: float) -> None:
        self.mode, self.san, self.profile, self.budget = mode, sanitizer, profile, budget
        self.out = san_dir(sanitizer, profile)
        self.copy = ROOT / "build" / "fuzz" / f"quant-{sanitizer}"
        self.crashes = self.out / "crashes"
        self.crashes.mkdir(parents=True, exist_ok=True)
        self.copy.mkdir(parents=True, exist_ok=True)
        self.env = {**os.environ, "FUZZ_SANITIZER": sanitizer, "FUZZ_PROFILE": profile, "CUDA_VISIBLE_DEVICES": ""}

    def result(self, target: str) -> Result:
        """Give an empty result of a target of this run."""
        return Result(target, self.san, self.profile, self.mode)

    # --- helpers ---

    def _keep_log(self, target: str, text: str) -> str:
        """Save the log of a failed run in crashes/ and give its path."""
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{random.randrange(1 << 16):04x}"
        path = self.crashes / f"{target}-{self.mode}-{stamp}.log"
        path.write_text(text)
        return str(path.relative_to(ROOT))

    def _pytest(self, file: str, extra_env: dict[str, str], args: list[str], timeout: float) -> tuple[int, str, int]:
        """Run pytest on one file and give (exit status, output, example count)."""
        counts = self.out / f"counts-{os.getpid()}-{threading.get_ident()}.json"
        counts.unlink(missing_ok=True)
        env = {**self.env, **extra_env, "QFZ_COUNTS": str(counts)}
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rxX", str(HERE / file), *args]
        try:
            res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, errors="replace",
                                 timeout=timeout, check=False)
            status, text = res.returncode, res.stdout + res.stderr
        except subprocess.TimeoutExpired as exc:
            status, text = -9, f"pytest passed the limit of {timeout} s: {exc}"
        executions = sum(json.loads(counts.read_text()).values()) if counts.exists() else 0
        counts.unlink(missing_ok=True)
        return status, text, executions

    @staticmethod
    def _count(text: str, word: str) -> int:
        """Give the number before a word of the pytest summary, for example '3 failed'."""
        m = re.search(rf"(\d+) {word}\b", text)
        return int(m.group(1)) if m else 0

    # --- the kinds of target ---

    def run_pytest(self, t: Target) -> Result:
        """A Hypothesis fuzzer: one replay round (test), or generation rounds for the budget (fuzz)."""
        r = self.result(t.name)
        start = time.monotonic()
        if self.mode == "test":
            status, text, r.executions = self._pytest(t.file, {"QFZ_MODE": "test"}, [], 3600)
            if status != 0:
                r.findings = max(1, self._count(text, "failed") + self._count(text, "error"))
                r.crash_files.append(self._keep_log(t.name, text))
        else:
            while time.monotonic() - start < self.budget:
                seed = random.randrange(1 << 30)
                status, text, n = self._pytest(t.file, {"QFZ_EXAMPLES": str(t.examples)},
                                               [f"--hypothesis-seed={seed}"], max(600.0, 4 * self.budget))
                r.executions += n
                if status != 0:
                    r.findings = max(1, self._count(text, "failed") + self._count(text, "error"))
                    r.crash_files.append(self._keep_log(t.name, text))
                    break
        r.seconds = time.monotonic() - start
        return r

    def run_regressions(self, t: Target) -> Result:
        """The minimal examples of the findings: each open finding (strict xfail) and each failure is a finding."""
        r = self.result(t.name)
        start = time.monotonic()
        status, text, _ = self._pytest(t.file, {}, [], 1800)
        r.executions = sum(self._count(text, w) for w in ("passed", "failed", "xfailed", "xpassed"))
        findings_dir = self.out / "findings"
        findings_dir.mkdir(exist_ok=True)
        for line in text.splitlines():
            m = re.match(r"XFAIL (\S+) - (Q[FR]\d+): (.*)", line)
            if m:
                test_id, finding, reason = m.groups()
                path = findings_dir / f"{finding}-{test_id.split('::')[-1]}.txt"
                path.write_text(f"finding {finding}: {KNOWN_FINDINGS.get(finding, '')}\ntest {test_id}\n"
                                f"evidence: {reason}\nfix: build/fuzz/quant/fixes\n")
                r.crash_files.append(str(path.relative_to(ROOT)))
                r.findings += 1
        if status != 0:
            r.findings += max(1, self._count(text, "failed"))
            r.crash_files.append(self._keep_log(t.name, text))
        r.seconds = time.monotonic() - start
        return r

    def run_atheris(self, t: Target) -> Result:
        """A libFuzzer Python fuzzer: the seed files once (test), or libFuzzer for the budget (fuzz)."""
        r = self.result(t.name)
        start = time.monotonic()
        # The failing inputs of regress/ are seeds too, thus the mode test replays them.
        seeds = [str(d) for d in (SEED_DIR / t.seeds, REGRESS_DIR / t.seeds) if d.is_dir()]
        corpus = FUZZ_OUT / "atheris" / f"{t.name}-corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        prefix = self.crashes / f"{t.name}-"
        before = set(self.crashes.glob(f"{t.name}-*"))
        args = [f"-artifact_prefix={prefix}", "-timeout=20", "-rss_limit_mb=4096", "-print_final_stats=1"]
        if self.mode == "test":
            args += ["-runs=0", *seeds]
        else:
            args += [f"-max_total_time={int(self.budget)}", str(corpus), *seeds]
        cmd = [sys.executable, str(HERE / t.file), *args]
        try:
            res = subprocess.run(cmd, cwd=ROOT, env=self.env, capture_output=True, text=True, errors="replace",
                                 timeout=self.budget + 600, check=False)
            status, text = res.returncode, res.stdout + res.stderr
        except subprocess.TimeoutExpired as exc:
            status, text = -9, str(exc)
        m = re.search(r"stat::number_of_executed_units:\s+(\d+)", text) or re.search(r"Done (\d+) runs", text)
        r.executions = int(m.group(1)) if m else 0
        new = sorted(set(self.crashes.glob(f"{t.name}-*")) - before)
        r.crash_files = [str(p.relative_to(ROOT)) for p in new]
        if status != 0 or new:
            r.findings = max(1, len([p for p in new if not p.name.endswith(".log")]))
            r.crash_files.append(self._keep_log(t.name, text))
        r.seconds = time.monotonic() - start
        return r

    def _check_bin(self) -> Path:
        return self.out / "bin" / "qfz-gguf-check"

    def _llama_bin(self) -> Path:
        return self.out / "llama" / "bin"

    def run_loader(self, t: Target) -> Result:
        """The ggml loader check of this sanitizer: seed files (test), or the files of the fuzzers (fuzz)."""
        from qfz_checks import sanitizer_env

        r = self.result(t.name)
        start = time.monotonic()
        if not self._check_bin().exists():
            r.note = f"skipped: {self._check_bin()} is missing (run.sh build {self.san})"
            return r
        reports = self.out / "ggml-loader-reports"
        reports.mkdir(exist_ok=True)
        before = set(reports.glob("*.gguf.bin"))
        if self.mode == "test":
            env = {**self.env, **sanitizer_env(self.san)}
            meta = [p for d in (SEED_DIR / "reader", REGRESS_DIR / "reader", REGRESS_DIR / "ggml")
                    for p in sorted(d.glob("*.seed"))]
            files = [(p, True) for p in meta]
            files += [(p, False) for p in sorted((SEED_DIR / "reader").glob("toy-*.seed")) +
                      sorted((REGRESS_DIR / "ggml-full").glob("*.seed")) + sorted(PHONE.glob("*.gguf"))]
            for path, meta in files:
                cmd = [str(self._check_bin()), "--meta", str(path)] if meta else [str(self._check_bin()), str(path)]
                res = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace", timeout=300,
                                     check=False)
                r.executions += 1
                if res.returncode not in (0, 3):
                    r.findings += 1
                    r.crash_files.append(str(path.relative_to(ROOT)))
                    log = f"{' '.join(cmd)}\nexit {res.returncode}\n{res.stderr}"
                    r.crash_files.append(self._keep_log(t.name, log))
        else:
            per_round = {"test_qfz_reader.py": 1000, "test_qfz_export.py": 150}
            while time.monotonic() - start < self.budget:
                for file, n in per_round.items():
                    status, text, count = self._pytest(file, {"QFZ_EXAMPLES": str(n)},
                                                       [f"--hypothesis-seed={random.randrange(1 << 30)}"],
                                                       max(600.0, 4 * self.budget))
                    r.executions += count
                    if status != 0:
                        r.findings += 1
                        r.crash_files.append(self._keep_log(t.name, text))
                if r.findings:
                    break
            new = sorted(set(reports.glob("*.gguf.bin")) - before)
            r.findings += len(new)
            r.crash_files += [str(p.relative_to(ROOT)) for p in new]
        r.seconds = time.monotonic() - start
        return r

    def run_llama(self, t: Target) -> Result:
        """llama-perplexity of this build on toy models against the oracle base: fixed (test), new (fuzz)."""
        import numpy as np
        import torch

        from qfz_checks import NoKLStatistics, kl_statistics, parse_kld, run_perplexity, sanitizer_env
        from qfz_common import scratch
        from qfz_toy import PROFILES, SMALL, make_text, output_rot_for, write_source
        from quant.export import export
        from quant.plan import Plan

        r = self.result(t.name)
        start = time.monotonic()
        if not (self._llama_bin() / "llama-perplexity").exists():
            r.note = f"skipped: {self._llama_bin()}/llama-perplexity is missing (run.sh build {self.san})"
            return r
        if not (ORACLE_BIN / "llama-perplexity").exists():
            r.note = f"skipped: the oracle build {ORACLE_BIN} is missing"
            return r
        env = sanitizer_env(self.san)
        threads = 4
        no_block: list[str] = []

        def check(model: Path, text: Path, base: Path, label: str) -> None:
            """Run one model in this build against the base of the oracle, and count a finding.

            A run with the exit status 0 and no final KL statistics block is a
            finding too (the check of task #172).
            """
            status, log = run_perplexity(self._llama_bin(), model, text, base, write_base=False, threads=threads,
                                         extra_env=env, timeout=900)
            r.executions += 1
            kld = parse_kld(log)
            if status == 0:
                try:
                    kld = kl_statistics(log)
                except NoKLStatistics:
                    no_block.append(label)
            with _WRITE_LOCK, (self.out / f"llama-toy-{self.mode}.kl.jsonl").open("a") as f:
                f.write(json.dumps({"model": label, "status": status, **kld}) + "\n")
            if status != 0 or not np.isfinite(kld.get("mean", float("nan"))):
                r.findings += 1
                kept = self.crashes / f"{t.name}-{label}.gguf"
                shutil.copy(model, kept)
                r.crash_files += [str(kept.relative_to(ROOT)), self._keep_log(t.name, f"exit {status}\n{log}")]

        phone = sorted(PHONE.glob("*.gguf"))
        if self.mode == "test" and phone:
            text = PHONE / "text.txt"
            for model in phone:
                check(model, text, model.with_suffix(".kld"), model.stem)
            # The reproducer of QT1 (rule R13): 20 more runs of one comparison. Each run needs its statistics.
            for i in range(20):
                check(phone[0], text, phone[0].with_suffix(".kld"), f"{phone[0].stem}-repeat{i}")
        else:
            rng = random.Random(0 if self.mode == "test" else None)
            n = 0
            while (self.mode == "test" and n < 8) or (self.mode == "fuzz" and time.monotonic() - start < self.budget):
                profile = PROFILES[n % 4] if self.mode == "test" else rng.choice(PROFILES)
                kinds = ("Q8_0", "Q4_0", "IQ4_NL")
                kind = kinds[n % 2] if self.mode == "test" else rng.choice(kinds)
                tie, seed = (n % 3 == 0), rng.randrange(1 << 30)
                geo = dataclasses.replace(SMALL, ssm_v_heads=rng.choice((1, 2)), mtp=rng.random() < 0.3)
                with scratch() as tmp:
                    src = write_source(tmp / "src.gguf", geo, profile, seed, tied=tie)
                    rot = None
                    if tie:
                        rot = tmp / "rot.npy"
                        np.save(rot, output_rot_for(geo, seed))
                    plan = Plan(bulk=kind, head=kind, embedding=kind, kv_proj=kind, gdn_gate=kind,
                                n_layers=geo.n_layer, mtp=rng.choice(("F16", "Q8_0")))
                    model = tmp / "model.gguf"
                    export(src.path, model, tmp / "no-packs", plan, ROOT / "third_party" / "llama.cpp",
                           torch.device("cpu"), tie_head=tie, rot=rot)
                    text = tmp / "text.txt"
                    text.write_text(make_text(900, seed))
                    base = tmp / "base.kld"
                    status, log = run_perplexity(ORACLE_BIN, model, text, base, write_base=True, threads=threads)
                    if status != 0:
                        r.note = f"the oracle failed on a toy model: {log[-300:]}"
                        break
                    check(model, text, base, f"{profile}-{kind}-{seed}")
                    if self.mode == "test" and n == 0:
                        # The reproducer of QT1 (rule R13): 20 more runs of one comparison.
                        for i in range(20):
                            check(model, text, base, f"{profile}-{kind}-{seed}-repeat{i}")
                n += 1
        if no_block:
            r.note = (f"{len(no_block)} runs with the exit status 0 had no final KL statistics block "
                      f"(the defect of QT1, task #172): {', '.join(no_block[:4])}")
        r.seconds = time.monotonic() - start
        return r

    def run(self, t: Target) -> Result:
        """Run one target and append its line to results.jsonl."""
        LOG.info("start %s (%s, %s, %s)", t.name, self.mode, self.san, self.profile)
        kind = {"pytest": self.run_pytest, "regress": self.run_regressions, "atheris": self.run_atheris,
                "loader": self.run_loader, "llama": self.run_llama}[t.kind]
        try:
            result = kind(t)
        except Exception as exc:  # noqa: BLE001 - a broken target is a finding of the harness, with its trace
            import traceback

            result = self.result(t.name)
            result.findings = 1
            result.crash_files = [self._keep_log(t.name, traceback.format_exc())]
            result.note = f"the target raised {type(exc).__name__}"
        with _WRITE_LOCK:
            for path in (self.out / "results.jsonl", self.copy / "results.jsonl"):
                with path.open("a") as f:
                    f.write(result.line() + "\n")
        LOG.info("done %s: %s", t.name, result.line())
        return result


def main() -> int:
    """Run the command line. Give 1 for a finding, 3 for a target that could not run, 0 otherwise."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=("test", "fuzz"))
    p.add_argument("sanitizer", choices=SANITIZERS)
    p.add_argument("--profile", choices=PROFILES, default=os.environ.get("FUZZ_PROFILE", "release"))
    p.add_argument("--budget-seconds", type=float, default=600.0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--targets", default="", help="a comma list of target names, empty for all that apply")
    a = p.parse_args()
    python = []
    if a.sanitizer == "none":
        # A profile changes only native code: the pure Python targets run in the release pass, the
        # regression tests in the two passes (rule R13).
        python = PYTHON_TARGETS if a.profile == "release" else [t for t in PYTHON_TARGETS if t.kind == "regress"]
    targets = python + NATIVE_TARGETS
    if a.targets:
        wanted = set(a.targets.split(","))
        targets = [t for t in targets if t.name in wanted]
    runner = Runner(a.mode, a.sanitizer, a.profile, a.budget_seconds)
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        results = list(pool.map(runner.run, targets))
    findings = sum(r.findings for r in results)
    skipped = [r.target for r in results if r.note.startswith("skipped")]
    LOG.info("%d targets, %d findings, skipped: %s. Results: %s", len(results), findings, skipped or "none",
             runner.out / "results.jsonl")
    if findings:
        return 1
    return 3 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
