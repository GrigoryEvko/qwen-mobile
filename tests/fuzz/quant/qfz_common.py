"""The shared paths, switches and numeric bounds of the fuzzers of the quantization pipeline.

Every fuzzer of tests/fuzz/quant imports this module. It holds no test.

The known findings are the open defects that the campaign recorded. Each
finding has an identifier (for example "QF1"). A fuzzer does not make the
inputs of an open finding, unless the environment variable ``QFZ_KNOWN``
holds "all" or the identifier.

The directory ``regress`` holds each failing input of the campaign:
``regress/test_qfz_regressions.py`` holds the minimal example of each
finding as a test, and ``regress/<target>`` holds the input files that the
test mode of run.sh replays. The directory ``seeds`` holds the other seed
files, which do not fail.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEED_DIR = HERE / "seeds"
REGRESS_DIR = HERE / "regress"
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"
FUZZ_OUT = ROOT / "build" / "fuzz" / "quant"
ORACLE_BIN = ROOT / "build" / "oracle-x86" / "bin"
HOST_BIN = ROOT / "build" / "host-llama" / "bin"
SANITIZERS = ("none", "asan", "ubsan", "tsan", "msan")
PROFILES = ("debug", "release")
# The sanitizer and the profile of this run: run.sh sets FUZZ_SANITIZER and FUZZ_PROFILE.
# One sanitizer per build and per run, never two.
SANITIZER = os.environ.get("FUZZ_SANITIZER", "none")
PROFILE = os.environ.get("FUZZ_PROFILE", "release")


def san_dir(sanitizer: str | None = None, profile: str | None = None) -> Path:
    """Give the build and result directory of a sanitizer and a profile: build/fuzz/quant-<profile>-<sanitizer>.

    Args:
        sanitizer: One of SANITIZERS, or None for the sanitizer of this run
        profile: One of PROFILES, or None for the profile of this run

    Returns:
        The directory

    Raises:
        ValueError: If the sanitizer or the profile is not known
    """
    san, prof = sanitizer or SANITIZER, profile or PROFILE
    if san not in SANITIZERS:
        raise ValueError(f"the sanitizer {san} is not one of {SANITIZERS}")
    if prof not in PROFILES:
        raise ValueError(f"the profile {prof} is not one of {PROFILES}")
    return ROOT / "build" / "fuzz" / f"quant-{prof}-{san}"


CHECK_BIN = san_dir() / "bin" / "qfz-gguf-check"
LLAMA_BIN = san_dir() / "llama" / "bin"

# The open findings of the campaign. The regression tests give the file, the line and the example.
KNOWN_FINDINGS = {
    "QF1": "a non-finite input or a block maximum out of the F16 scale range gives a non-finite F16 scale",
    "QF2": "Q4_0Grid.round casts NaN to int32, and a zero or NaN scale gives an IndexError",
    "QF3": "pack_nibbles and the export accept a pack with indices out of 0..15 or with a wrong shape",
    "QF4": "the export copies general.alignment, but the writer keeps the alignment 32",
    "QF5": "the export of a source with an empty array field fails after the header is on the disk",
    "QF6": "the export writes an unknown plan type as the source tensor, with no error",
    "QF8": "the export casts the dense maps to F16 with no check, thus an overflow gives inf",
    "QF9": "the scale search compares float32 scales but stores their F16 rounding, which can lose to the plain scale",
    "QF10": "the export filter --only applies to Q4_0 and Q8_0, thus an IQ4_NL tensor out of the match stays IQ4_NL",
    "QF11": "each export() call puts llama_dir/gguf-py at the front of sys.path again, thus sys.path only grows",
    "QR1": "the gguf-py reader loops over a scalar array that is longer than the file",
    "QR2": "the gguf-py reader does not check the data range of a tensor: a large offset wraps, a 0-byte tensor passes",
    "QR3": "the gguf-py reader gives an IndexError with no context for a truncated file",
    "QT1": "llama-perplexity does not flush its log at the exit, thus a run with the status 0 can lose its statistics",
}

# The largest finite F16 value, and the smallest positive subnormal F16 value.
F16_MAX = 65504.0
F16_TINY = 2.0 ** -24


def fixed(finding: str) -> bool:
    """Tell if a run must treat a finding as fixed: QFZ_FIXED holds "all" or the identifier.

    The regression tests expect a failure of each open finding (strict
    xfail). A run against a tree with the proposed patches sets QFZ_FIXED,
    and the tests then expect the correct behavior.

    Args:
        finding: The identifier of the finding

    Returns:
        True when the run treats the finding as fixed

    Raises:
        KeyError: If the identifier is not in KNOWN_FINDINGS
    """
    if finding not in KNOWN_FINDINGS:
        raise KeyError(f"{finding} is not a known finding. The known findings are {sorted(KNOWN_FINDINGS)}")
    wanted = {part.strip() for part in os.environ.get("QFZ_FIXED", "").split(",") if part.strip()}
    return "all" in wanted or finding in wanted


def known_open(finding: str) -> bool:
    """Tell if a fuzzer must skip the inputs of an open finding.

    Args:
        finding: The identifier of the finding, for example "QF1"

    Returns:
        True when the fuzzer must skip those inputs. QFZ_KNOWN="all" or a
        list that holds the identifier makes the fuzzer keep them. A finding
        that QFZ_FIXED holds is not open, thus the fuzzer also keeps them.

    Raises:
        KeyError: If the identifier is not in KNOWN_FINDINGS
    """
    wanted = {part.strip() for part in os.environ.get("QFZ_KNOWN", "").split(",") if part.strip()}
    return not (fixed(finding) or "all" in wanted or finding in wanted)


@contextmanager
def scratch() -> Iterator[Path]:
    """Give a temporary directory for one example, and remove it after the example.

    Hypothesis runs many examples in one test call, thus a test cannot use
    the pytest fixture tmp_path, which lives for the full call.

    Yields:
        The directory
    """
    with tempfile.TemporaryDirectory(prefix="qfz-") as path:
        yield Path(path)


def f16_round(x: np.ndarray) -> np.ndarray:
    """Give the values of ``x`` after a round trip through F16, as float32.

    Args:
        x: A float array

    Returns:
        The float32 array of the F16 values
    """
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def q8_0_error_bound(block_amax: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Give the largest permitted error of each Q8_0 block, for the stored F16 scale.

    An element with |w / d| <= 127 rounds to the nearest level, thus its
    error is at most |d| / 2. An element beyond 127 levels clamps to 127,
    and its error is |w| - 127 |d|. A small slack covers the float32
    rounding of the division and of the product. Complexity is O(blocks).

    Args:
        block_amax: The largest magnitude of each block, float32 [rows, nblocks]
        d: The stored scale of each block, as float32 [rows, nblocks]

    Returns:
        The bound for each block, float32 [rows, nblocks]
    """
    ad = np.abs(d.astype(np.float64))
    amax = block_amax.astype(np.float64)
    inside = 0.5 * ad
    clamp = amax - 127.0 * ad
    slack = 4.0 * np.spacing(np.maximum(amax, ad * 127.0).astype(np.float32)).astype(np.float64)
    return (np.maximum(inside, clamp) + slack).astype(np.float64)


def block_view(w: np.ndarray, block: int = 32) -> np.ndarray:
    """Give the view [rows, cols // block, block] of a matrix [rows, cols].

    Args:
        w: The matrix
        block: The block length

    Returns:
        The blocked view

    Raises:
        ValueError: If the columns are not a multiple of the block length
    """
    rows, cols = w.shape
    if cols % block:
        raise ValueError(f"the matrix has {cols} columns, which is not a multiple of {block}")
    return w.reshape(rows, cols // block, block)
