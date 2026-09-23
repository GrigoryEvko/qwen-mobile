"""The pytest configuration of the fuzzers of the quantization pipeline.

The fuzzers run on the CPU only. This file hides the CUDA devices before a
test module imports torch, thus no test can use the laptop GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Write the example counts of the quant fuzzers to QFZ_COUNTS, when run.sh asks for them."""
    target = os.environ.get("QFZ_COUNTS")
    if target:
        from qfz_hyp import write_counts

        write_counts(Path(target))
