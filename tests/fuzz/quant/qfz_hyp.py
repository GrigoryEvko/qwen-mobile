"""The Hypothesis settings of the fuzzers of the quantization pipeline.

The settings apply to each test through a decorator, and not through a
global profile. Thus a pytest session that also runs the fuzzers of other
areas keeps their settings.

Environment variables:
    QFZ_MODE: "test" runs only the explicit examples and the examples of
        the database, with no new generation. Any other value generates.
    QFZ_EXAMPLES: The number of generated examples of one test in one run.
        The preset value is 25, a smoke run. run.sh sets a larger value.
    QFZ_DB: The directory of the example database. The preset value is
        build/fuzz/quant/hypothesis-db, which keeps each failure for replay.
"""

from __future__ import annotations

import functools
import json
import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, Phase, settings
from hypothesis.database import DirectoryBasedExampleDatabase

from qfz_common import FUZZ_OUT

_DB = DirectoryBasedExampleDatabase(str(Path(os.environ.get("QFZ_DB", FUZZ_OUT / "hypothesis-db"))))
_REPLAY = (Phase.explicit, Phase.reuse)
_GENERATE = (Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink, Phase.explain)
_COUNTS: Counter[str] = Counter()


def counted(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Count each call of a test body, thus each example that Hypothesis runs. Put it below @given and @example.

    Args:
        fn: The test body

    Returns:
        The wrapped body, with the signature of ``fn``
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _COUNTS[fn.__name__] += 1
        return fn(*args, **kwargs)

    return wrapper


def write_counts(path: Path) -> None:
    """Add the example counts of this process to a JSON file, which run.sh reads for results.jsonl.

    Args:
        path: The JSON file, {test name: count}
    """
    old = json.loads(path.read_text()) if path.exists() else {}
    for name, n in _COUNTS.items():
        old[name] = old.get(name, 0) + n
    path.write_text(json.dumps(old))


def fuzz_settings(scale: float = 1.0) -> settings:
    """Give the Hypothesis settings of one fuzz test.

    Args:
        scale: The fraction of QFZ_EXAMPLES that this test runs. A slow
            test uses a fraction less than one.

    Returns:
        The settings object, which is also a test decorator
    """
    base = int(os.environ.get("QFZ_EXAMPLES", "25"))
    replay = os.environ.get("QFZ_MODE") == "test"
    return settings(
        max_examples=max(1, int(base * scale)),
        deadline=None,
        database=_DB,
        print_blob=True,
        phases=_REPLAY if replay else _GENERATE,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much,
                               HealthCheck.large_base_example],
    )
