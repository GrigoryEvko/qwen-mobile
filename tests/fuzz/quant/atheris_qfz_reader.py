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

An input of an open finding (QR1, QR2, QR3) is skipped, unless QFZ_KNOWN
holds it. QR1 is recognized before the read: a scalar array whose count
does not fit the rest of the file.

    uv run python tests/fuzz/quant/atheris_qfz_reader.py -max_total_time=600 -timeout=10 \\
        -rss_limit_mb=3072 -artifact_prefix=build/fuzz/quant/atheris/ build/fuzz/quant/atheris/reader-corpus \\
        tests/fuzz/quant/seeds/reader tests/fuzz/quant/regress/reader

Give -artifact_prefix= in each run, else libFuzzer writes crash-* files into
the working directory.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import atheris

sys.path.insert(0, str(Path(__file__).resolve().parent))

with atheris.instrument_imports(include=["gguf"]):
    import gguf

from qfz_common import known_open  # noqa: E402
from qfz_rawgguf import SCALAR_WIDTH, walk  # noqa: E402

ALLOWED = (ValueError, KeyError, UnicodeDecodeError, OverflowError)
PATH = Path("/dev/shm") / f"qfz-atheris-reader-{os.getpid()}.gguf"


def long_scalar_array(data: bytes) -> bool:
    """Tell if the bytes hold a scalar array whose count does not fit the rest of the file (QR1).

    Args:
        data: The input

    Returns:
        True for an input of QR1
    """
    for field in walk(data):
        if field.role != "array_count":
            continue
        sub = struct.unpack_from("<I", data, field.pos - 4)[0]
        count = struct.unpack_from("<Q", data, field.pos)[0]
        width = SCALAR_WIDTH.get(sub, 0)
        if width and count * width > len(data) - field.pos - 8:
            return True
    return False


def test_one_input(data: bytes) -> None:
    """Read one input with the reader and check the oracle. Raise AssertionError on a defect."""
    if not data or (known_open("QR1") and long_scalar_array(data)):
        return
    PATH.write_bytes(data)
    try:
        reader = gguf.GGUFReader(str(PATH))
    except ALLOWED:
        return
    except IndexError:
        if known_open("QR3"):
            return
        raise
    for t in reader.tensors:
        start = int(t.data_offset)
        raw = int(t.field.parts[5][0])
        if known_open("QR2") and (int(reader.data_offset) + raw >= 2**64 or int(t.n_bytes) == 0):
            continue
        assert reader.data_offset <= start and start + int(t.n_bytes) <= len(data), (
            f"the reader accepts tensor {t.name} at {start} + {t.n_bytes}, out of the file of {len(data)} bytes")


def main() -> None:
    """Start libFuzzer with the command-line options."""
    try:
        atheris.Setup(sys.argv, test_one_input)
        atheris.Fuzz()
    finally:
        PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
