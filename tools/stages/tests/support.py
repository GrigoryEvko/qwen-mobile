"""Shared parts of the tests of the stage tools.

Each stage tool runs as a script, thus a test loads it from its path. A test that runs a line of a command
file uses a fake adb on PATH, which records its arguments and plays the part of the phone.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

STAGES = Path(__file__).resolve().parents[1]


def load(stage: str, tool: str = "stage") -> ModuleType:
    """The module of the tool tools/stages/<stage>/<tool>.py, loaded in this process. The command line of the
    module is empty during the load, thus a module that reads sys.argv at its load gets no argument."""
    path = STAGES / stage / f"{tool}.py"
    name = f"stage_{stage.replace('-', '_')}_{tool}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    saved = sys.argv
    sys.argv = [str(path)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


FAKE_ADB = """#!/usr/bin/env bash
# A fake adb: it records each call, lists the phone files for a find, and records a remove.
echo "$*" >> "$FAKE_ADB_LOG"
case "$*" in
    *find*) cat "$FAKE_PHONE_FILES" ;;
esac
exit 0
"""


class FakePhone:
    """A temporary laptop directory and a fake adb. `run(line)` runs one line of a command file with bash in
    the laptop directory, with the fake adb first on PATH, and gives its stdout. `calls()` gives the
    arguments of each adb call."""

    def __init__(self, phone_files: list[str]) -> None:
        """Make the directory, the fake adb, and the list of the files that the phone holds."""
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        bin_dir = self.root / "fakebin"
        bin_dir.mkdir()
        adb = bin_dir / "adb"
        adb.write_text(FAKE_ADB)
        adb.chmod(0o755)
        (self.root / "phone-files.txt").write_text("".join(f"./{f}\n" for f in phone_files))
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                        FAKE_ADB_LOG=str(self.root / "adb.log"), FAKE_PHONE_FILES=str(self.root / "phone-files.txt"))

    def put(self, directory: str, files: list[str], gate_ok: bool = True) -> None:
        """Make the files in a directory of the laptop. A gate file gets the text of a gate that passed, or of a
        gate that stopped the run."""
        base = self.root / directory
        for f in files:
            path = base / f
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("gate: OK\nrc=0\n" if f.endswith("-gate.txt") and gate_ok else
                            "gate: STOP, the thermal status is 2.\nrc=1\n" if f.endswith("-gate.txt") else "x\n")

    def run(self, line: str) -> str:
        """Run one line of a command file and give its stdout and stderr."""
        p = subprocess.run(["bash", "-c", line], cwd=self.root, env=self.env, capture_output=True, text=True)
        return p.stdout + p.stderr

    def calls(self) -> list[str]:
        """The arguments of each call of the fake adb."""
        log = self.root / "adb.log"
        return log.read_text().splitlines() if log.exists() else []

    def close(self) -> None:
        """Remove the temporary directory."""
        self._tmp.cleanup()
