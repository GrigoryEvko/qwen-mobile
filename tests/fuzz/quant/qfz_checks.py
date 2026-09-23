"""The file checks of the export fuzzers: gguf-py, the ggml loader under the sanitizers, and llama.cpp.

The checks read a written GGUF file and compare it with what the plan
and the source promise. They hold no Hypothesis code, thus the fuzzers
and phone.py share them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import gguf
from qfz_common import CHECK_BIN, ROOT, SANITIZER

LOG = logging.getLogger("qfz")
# The shared suppression files of all the fuzz areas (rules R5 and R6): tests/sanitizers/<config>.supp.
SHARED_SUPPRESSIONS = ROOT / "tests" / "sanitizers"
# The exit status of a sanitizer report, one value for each sanitizer.
SANITIZER_EXIT = {"asan": 86, "ubsan": 87, "tsan": 88, "msan": 89}
DECODED_TYPES = {"F32", "F16", "Q4_0", "Q8_0", "IQ4_NL"}


def sanitizer_env(sanitizer: str | None = None) -> dict[str, str]:
    """Give the runtime options of one sanitizer: a report stops the run with the exit status of SANITIZER_EXIT.

    The shared suppression file tests/sanitizers/<config>.supp applies when
    it exists. This area keeps no suppression of its own.

    Args:
        sanitizer: One of SANITIZERS, or None for the sanitizer of this run

    Returns:
        The environment variables, empty for "none"
    """
    san = sanitizer or SANITIZER
    supp = SHARED_SUPPRESSIONS / f"{san}.supp"
    extra = f":suppressions={supp}" if supp.exists() else ""
    # An abort of ggml attaches gdb for a backtrace, which takes seconds and hides the report: turn it off.
    base = {"GGML_NO_BACKTRACE": "1"}
    env = {
        "none": {},
        "asan": {"ASAN_OPTIONS": f"detect_leaks=1:abort_on_error=0:exitcode=86{extra}",
                 "LSAN_OPTIONS": f"exitcode=86{extra}"},
        "ubsan": {"UBSAN_OPTIONS": f"print_stacktrace=1:halt_on_error=1:exitcode=87{extra}"},
        "tsan": {"TSAN_OPTIONS": f"halt_on_error=1:exitcode=88:second_deadlock_stack=1{extra}"},
        "msan": {"MSAN_OPTIONS": "halt_on_error=1:exitcode=89"},
    }
    return {**base, **env[san]}


def decode(t: object) -> np.ndarray:
    """Give the float32 values of a gguf-py reader tensor in numpy order [rows, cols] or [n].

    Args:
        t: A ReaderTensor

    Returns:
        The values

    Raises:
        ValueError: If gguf-py has no decoder for the type
    """
    kind = t.tensor_type.name
    if kind not in DECODED_TYPES:
        raise ValueError(f"{t.name}: gguf-py decodes no {kind} in this check")
    data = np.asarray(t.data)
    shape = [int(x) for x in reversed(t.shape)]
    if kind in ("F32", "F16"):
        return data.astype(np.float32).reshape(shape)
    return gguf.quants.dequantize(data, t.tensor_type).reshape(shape).astype(np.float32)


def read_tensors(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Give (type name, float32 values) of each tensor of a GGUF file, by name.

    Args:
        path: The GGUF file

    Returns:
        The tensors
    """
    reader = gguf.GGUFReader(str(path))
    return {t.name: (t.tensor_type.name, decode(t)) for t in reader.tensors}


def check_available() -> bool:
    """Tell if the loader check binary of the sanitizer of this run exists. run.sh builds it."""
    return CHECK_BIN.exists()


def ggml_loader_check(path: Path, timeout: float = 120.0) -> dict[str, np.ndarray]:
    """Load a GGUF file with the ggml loader of the sanitizer of this run, and give the decoded values.

    Args:
        path: The GGUF file
        timeout: The limit of the run in seconds

    Returns:
        The float32 values of each tensor, flat, by name

    Raises:
        AssertionError: If the loader refuses the file, a sanitizer reports, or a type has no decoder
        FileNotFoundError: If the binary is missing (run ``tests/fuzz/quant/run.sh build <sanitizer>``)
    """
    if not CHECK_BIN.exists():
        raise FileNotFoundError(f"{CHECK_BIN} is missing. Build it with: tests/fuzz/quant/run.sh build {SANITIZER}")
    with tempfile.TemporaryDirectory(prefix="qfz-dump-") as dump:
        env = {**os.environ, **sanitizer_env()}
        res = subprocess.run([str(CHECK_BIN), str(path), dump], capture_output=True, text=True, errors="replace",
                             timeout=timeout,
                             env=env, check=False)
        if res.returncode != 0:
            raise AssertionError(f"the ggml loader check of {path} gave the exit status {res.returncode}:\n"
                                 f"{res.stdout[-2000:]}\n{res.stderr[-4000:]}")
        out: dict[str, np.ndarray] = {}
        for line in res.stdout.splitlines():
            row = json.loads(line)
            out[row["name"]] = np.fromfile(Path(dump) / f"{row['index']}.f32", dtype=np.float32)
        return out


def ggml_loader_status(path: Path, timeout: float = 60.0) -> tuple[int, str]:
    """Give the exit status and the sanitizer summary of the metadata check of the ggml loader.

    The check reads the metadata and the tensor infos only, as the model
    loader of llama.cpp does, and it refuses a tensor out of the file.

    Args:
        path: The GGUF file, which can be malformed
        timeout: The limit of the run in seconds

    Returns:
        (status, summary): 0 for a load, 3 for a refusal, a value of
        SANITIZER_EXIT for a sanitizer report, -9 for a timeout. The summary
        is the SUMMARY line of a sanitizer report, or an empty string.
    """
    env = {**os.environ, **sanitizer_env()}
    try:
        res = subprocess.run([str(CHECK_BIN), "--meta", str(path)], capture_output=True, text=True, errors="replace",
                             timeout=timeout,
                             env=env, check=False)
    except subprocess.TimeoutExpired:
        return -9, "timeout"
    summary = next((line for line in res.stderr.splitlines() if line.startswith("SUMMARY:")), "")
    if res.returncode not in (0, 3, 4):
        LOG.info("the ggml loader check of %s gave %d: %s", path, res.returncode, summary)
    return res.returncode, summary


def record_ggml_report(data: bytes, summary: str, stderr_dir: Path) -> Path:
    """Keep one input for each distinct sanitizer summary of the ggml loader, for the core fuzz area.

    Args:
        data: The input bytes
        summary: The SUMMARY line of the report
        stderr_dir: The directory of the kept inputs

    Returns:
        The path of the kept input
    """
    stderr_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(summary.encode()).hexdigest()[:16]
    path = stderr_dir / f"{key}.gguf.bin"
    if not path.exists():
        path.write_bytes(data)
        (stderr_dir / f"{key}.txt").write_text(summary + "\n")
    return path


def run_perplexity(binary_dir: Path, model: Path, text: Path, base: Path, write_base: bool, threads: int = 4,
                   timeout: float = 600.0, extra_env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run llama-perplexity of a build on a model, and write or compare the KL base.

    Args:
        binary_dir: The bin directory of the build
        model: The GGUF file
        text: The text file
        base: The KL divergence base file
        write_base: True to write the base, False to compare against it
        threads: The CPU threads
        timeout: The limit of the run in seconds
        extra_env: More environment variables, for example the sanitizer options

    Returns:
        The exit status and the output
    """
    cmd = [str(binary_dir / "llama-perplexity"), "-m", str(model), "-f", str(text), "-c", "128", "-b", "128",
           "--chunks", "4", "-t", str(threads), "--kl-divergence-base", str(base)]
    if not write_base:
        cmd.append("--kl-divergence")
    env = {**os.environ, "LD_LIBRARY_PATH": str(binary_dir), **(extra_env or {})}
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, env=env, cwd=ROOT,
                             check=False)
    except subprocess.TimeoutExpired as exc:
        return -9, f"timeout after {timeout} s: {exc}"
    return res.returncode, res.stdout + res.stderr


def run_kld(binary_dir: Path, model: Path, text: Path, base: Path, threads: int = 4, timeout: float = 600.0,
            extra_env: dict[str, str] | None = None, retries: int = 2) -> tuple[int, str, list[str]]:
    """Compare a model against a KL base, and run it again when the statistics are missing (finding QT1).

    llama-perplexity does not flush its log before the exit (finding QT1),
    thus a run with the exit status 0 can lose the last chunk and the KL
    statistics. This function runs again, at most ``retries`` times, and it
    gives the logs of the lost runs, which the caller reports (rule R8).

    Args:
        binary_dir: The bin directory of the build
        model: The GGUF file
        text: The text file
        base: The KL base of the oracle
        threads: The CPU threads
        timeout: The limit of one run in seconds
        extra_env: More environment variables, for example the sanitizer options
        retries: The largest number of runs again

    Returns:
        The exit status and the output of the last run, and the outputs of the runs that lost the statistics
    """
    lost: list[str] = []
    while True:
        status, log = run_perplexity(binary_dir, model, text, base, write_base=False, threads=threads,
                                     timeout=timeout, extra_env=extra_env)
        if status != 0 or "Mean    KLD:" in log or len(lost) >= retries:
            return status, log, lost
        lost.append(log)


def parse_kld(text: str) -> dict[str, float]:
    """Give the KL numbers of a llama-perplexity output: mean, max, p99, top1 (percent), rms_dp (percent).

    Args:
        text: The output

    Returns:
        The numbers that the output holds
    """
    keys = {"Mean    KLD:": "mean", "Maximum KLD:": "max", "99.0%   KLD:": "p99", "Same top p:": "top1",
            "RMS Δp    :": "rms_dp"}
    out: dict[str, float] = {}
    for line in text.splitlines():
        for marker, key in keys.items():
            if marker in line:
                token = line.split(marker, 1)[1].split()[0]
                try:
                    out[key] = float(token)
                except ValueError:
                    out[key] = float("nan")
    return out
