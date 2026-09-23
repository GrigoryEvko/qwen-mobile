"""The coverage-guided fuzzer (atheris, libFuzzer) of the gguf-py reader.

atheris instruments the Python bytecode of gguf-py, thus libFuzzer keeps
each input that reaches a new branch of the reader. The target writes the
input to a file in /dev/shm and reads it with GGUFReader.

The oracle:

- The reader gives a result or an exception of the types in ALLOWED.
- When it gives a result, the data of each tensor is inside the data
  section of the file.
- libFuzzer itself stops a run that is longer than -timeout or larger than
  -rss_limit_mb, and it saves the input.

    uv run python tests/fuzz/quant/atheris_qfz_reader.py -max_total_time=600 -timeout=10 \\
        -rss_limit_mb=3072 -artifact_prefix=build/fuzz/quant/atheris/ build/fuzz/quant/atheris/reader-corpus \\
        tests/fuzz/quant/seeds/reader tests/fuzz/quant/regress/reader

Give -artifact_prefix= in each run, else libFuzzer writes crash-* files into
the working directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import atheris

sys.path.insert(0, str(Path(__file__).resolve().parent))

with atheris.instrument_imports(include=["gguf"]):
    import gguf

ALLOWED = (ValueError, KeyError, UnicodeDecodeError, OverflowError)
SHM = Path("/dev/shm")
PATH = SHM / f"qfz-atheris-reader-{os.getpid()}.gguf"


def remove_stale_inputs() -> None:
    """Remove the input files of the ended processes of this target. Complexity is O(files).

    libFuzzer ends the process with exit() at -max_total_time or at a
    finding, thus the finally block of main() does not run, and its file
    stays in /dev/shm. The next run removes it.
    """
    for path in SHM.glob("qfz-atheris-reader-*.gguf"):
        pid = path.stem.rsplit("-", 1)[-1]
        if pid.isdigit() and not Path(f"/proc/{pid}").exists():
            path.unlink(missing_ok=True)


def test_one_input(data: bytes) -> None:
    """Read one input with the reader and check the oracle. Raise AssertionError on a defect."""
    if not data:
        return
    PATH.write_bytes(data)
    try:
        reader = gguf.GGUFReader(str(PATH))
    except ALLOWED:
        return
    for t in reader.tensors:
        start = int(t.data_offset)
        assert reader.data_offset <= start and start + int(t.n_bytes) <= len(data), (
            f"the reader accepts tensor {t.name} at {start} + {t.n_bytes}, out of the file of {len(data)} bytes")


def main() -> None:
    """Start libFuzzer with the command-line options."""
    remove_stale_inputs()
    try:
        atheris.Setup(sys.argv, test_one_input)
        atheris.Fuzz()
    finally:
        PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
