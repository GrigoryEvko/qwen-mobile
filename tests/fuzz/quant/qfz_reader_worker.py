"""The isolated worker of the gguf-py reader fuzzer, and the pool that drives it.

A malformed file can make the reader loop for minutes or eat all the
memory (finding QR1). Thus the reader runs in a separate process that
imports numpy and gguf only, with an address-space limit and an alarm per
file. The pool starts a new worker after a timeout or a memory error.

The worker reads one JSON request per line on stdin and writes one JSON
result per line on stdout. A request holds the path and the reader
("plain" for gguf-py as it is, "fixed" for gguf-py with the short-read
check of the proposed fix of QR1). A result holds the status ("ok",
"error", "timeout", "memory"), the exception type and message, and for
"ok" the data section start, the file size, and the offset, the byte count
and the raw offset of each tensor.

    python tests/fuzz/quant/qfz_reader_worker.py     # the worker loop, for the pool only
"""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

MEMORY_LIMIT = 2 << 30
TIME_LIMIT = 5


class _Timeout(Exception):
    """The alarm of one request."""


def _on_alarm(signum: int, frame: object) -> None:
    """Raise _Timeout in the reader, from the alarm signal."""
    raise _Timeout()


def _fixed_reader(gguf: Any) -> type:
    """Give a GGUFReader subclass whose _get raises ValueError on a short read (the proposed fix of QR1)."""
    import numpy as np

    class FixedReader(gguf.GGUFReader):
        """GGUFReader with a check that each read is complete."""

        def _get(self, offset: int, dtype: Any, count: int = 1, override_order: Any = None) -> Any:
            count = int(count)
            itemsize = int(np.empty([], dtype=dtype).itemsize)
            if offset < 0 or offset + itemsize * count > self.data.nbytes:
                raise ValueError(f"a read of {count} x {itemsize} bytes at offset {offset} is beyond the end of "
                                 f"the file ({self.data.nbytes} bytes)")
            return super()._get(offset, dtype, count, override_order)

    return FixedReader


def _read(gguf: Any, readers: dict[str, type], request: dict[str, str]) -> dict[str, Any]:
    """Run one request and give its result."""
    reader_type = readers[request.get("reader", "plain")]
    signal.alarm(TIME_LIMIT)
    try:
        r = reader_type(request["path"])
        tensors = [{"name": t.name, "offset": int(t.data_offset), "nbytes": int(t.n_bytes),
                    "raw_offset": int(t.field.parts[5][0]), "type": t.tensor_type.name} for t in r.tensors]
        return {"status": "ok", "data_offset": int(r.data_offset), "size": int(r.data.nbytes), "tensors": tensors}
    except _Timeout:
        return {"status": "timeout"}
    except MemoryError:
        import gc

        gc.collect()
        return {"status": "memory"}
    except Exception as exc:  # noqa: BLE001 - the fuzzer classifies every exception type
        return {"status": "error", "type": type(exc).__name__, "message": str(exc)[:300]}
    finally:
        signal.alarm(0)


def serve() -> None:
    """Serve the requests on stdin until the end of the input."""
    import warnings

    warnings.simplefilter("ignore")
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    signal.signal(signal.SIGALRM, _on_alarm)
    import gguf

    readers = {"plain": gguf.GGUFReader, "fixed": _fixed_reader(gguf)}
    for line in sys.stdin:
        result = _read(gguf, readers, json.loads(line))
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


class ReaderPool:
    """One worker process of the reader, started again after a timeout, a memory error or a crash."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None

    def _start(self) -> subprocess.Popen[str]:
        """Start a new worker process that imports the same gguf package as this process.

        The directory of that package goes first: export() puts the gguf-py of
        its llama_dir at the front of sys.path, thus the order of sys.path is
        not a safe source.
        """
        import gguf

        package_dir = str(Path(gguf.__file__).resolve().parents[1])
        rest = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([package_dir, *rest])}
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve())], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, env=env)

    def read(self, path: Path, reader: str = "plain") -> dict[str, Any]:
        """Read one file in the worker and give the result.

        Args:
            path: The GGUF file
            reader: "plain" or "fixed"

        Returns:
            The result of the worker, or {"status": "crash"} when the worker died
        """
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._start()
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write(json.dumps({"path": str(path), "reader": reader}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except BrokenPipeError:
            line = ""
        if not line:
            self.close()
            return {"status": "crash"}
        result = json.loads(line)
        if result["status"] in ("timeout", "memory"):
            self.close()
        return result

    def close(self) -> None:
        """Stop the worker."""
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None


if __name__ == "__main__":
    serve()
