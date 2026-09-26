"""The command file of unary-rows does not depend on the state of the build.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from support import STAGES


def command_file(libs: list[str] | None) -> str:
    """The command file of unary-rows in a copy of the stage tree whose build holds the libraries `libs` in
    lib-new, or no build directory for None."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(STAGES, root / "tools" / "stages", ignore=shutil.ignore_patterns("__pycache__"))
        if libs is not None:
            lib_new = root / "build" / "unary-rows" / "phone" / "lib-new"
            lib_new.mkdir(parents=True)
            for name in libs:
                (lib_new / name).write_bytes(b"")
        out = root / "out.txt"
        p = subprocess.run([sys.executable, str(root / "tools/stages/unary-rows/stage.py"), "commands", "--out",
                            str(out)], cwd=root, capture_output=True, text=True)
        if p.returncode != 0:
            raise AssertionError(p.stderr)
        return out.read_text()


class UnaryRowsLoad(unittest.TestCase):
    """The push line of lib-new takes the libraries of the build when the runner runs it."""

    def test_the_command_file_is_the_same_with_and_without_a_build(self) -> None:
        """Two builds with different libraries in lib-new, and no build, give the same command file."""
        none = command_file(None)
        self.assertEqual(none, command_file(["libggml-hexagon.so"]))
        self.assertEqual(none, command_file(["libggml-base.so", "libggml-hexagon.so"]))

    def test_the_push_takes_each_library_of_lib_new(self) -> None:
        """The push line of lib-new names the directory with a glob, not a list of files."""
        text = command_file(None)
        push = next(ln for ln in text.splitlines() if " push " in ln and "/lib-new/" in ln)
        self.assertIn("build/unary-rows/phone/lib-new/*.so", push)


if __name__ == "__main__":
    unittest.main()
