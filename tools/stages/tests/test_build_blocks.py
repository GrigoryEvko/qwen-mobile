"""Each locked container block of a build recipe stops at the first container call that fails.

A block "( flock 9 ... ) 9> build/.container.lock ... || die ..." runs on the left side of "||", and there bash
ignores set -e, also in the subshell. Thus a call that fails does not stop the block, and the exit code of the
block is the exit code of its last command. A failed build of the first tree then hides behind the build of the
second tree, and the recipe copies the files of an earlier build. A call that has a call after it in its block, or
that is in a loop, must end with "|| exit".

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import re
import unittest

from support import STAGES

CALL_RE = re.compile(r"^\s*(stage_build|snapdragon_run|compile_memprobe|container_run)\b")
LOOP_RE = re.compile(r"^\s*(for|while)\b.*;\s*do\s*$")
DONE_RE = re.compile(r"^\s*done\b")


def command_end(lines: list[str], start: int) -> int:
    """The index of the last line of the command that starts at the line `start`: the first line after which
    the single quotes are closed and the line has no continuation. O(lines of the command)."""
    quotes = 0
    for i in range(start, len(lines)):
        quotes += lines[i].count("'")
        if quotes % 2 == 0 and not lines[i].rstrip().endswith("\\"):
            return i
    raise ValueError(f"the command at the line {start + 1} has no end")


def unguarded_calls(text: str) -> list[int]:
    """The line numbers of the container calls in the locked blocks of `text` that do not stop their block
    when they fail. O(lines)."""
    lines = text.splitlines()
    bad: list[int] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "(" or i + 1 >= len(lines) or lines[i + 1].strip() != "flock 9":
            i += 1
            continue
        calls: list[tuple[int, int, bool]] = []
        depth, j = 0, i + 2
        while not lines[j].lstrip().startswith(") 9>"):
            if LOOP_RE.match(lines[j]):
                depth += 1
            elif DONE_RE.match(lines[j]):
                depth -= 1
            elif CALL_RE.match(lines[j]):
                end = command_end(lines, j)
                calls.append((j, end, depth > 0))
                j = end
            j += 1
        if not lines[j].lstrip().startswith(") 9> build/.container.lock"):
            calls = []
        for n, (first, end, in_loop) in enumerate(calls):
            last = n == len(calls) - 1 and not in_loop
            if not last and not lines[end].rstrip().endswith("|| exit"):
                bad.append(first + 1)
        i = j + 1
    return bad


class LockedBlocks(unittest.TestCase):
    """The build recipes of tools/stages."""

    def test_each_call_before_the_last_stops_the_block(self) -> None:
        """No recipe has a container call that can fail without a stop of its block."""
        found = {str(p.relative_to(STAGES)): unguarded_calls(p.read_text())
                 for p in sorted(STAGES.glob("*/*.sh"))}
        self.assertEqual({k: v for k, v in found.items() if v}, {})

    def test_the_check_finds_a_call_without_a_stop(self) -> None:
        """A block with two calls and no "|| exit" gives the line of the first call."""
        text = ("(\n    flock 9\n    stage_build a b 'c\nd'\n    stage_build e f g\n"
                ") 9> build/.container.lock > log 2>&1 || die x\n")
        self.assertEqual(unguarded_calls(text), [3])

    def test_a_call_in_a_loop_needs_a_stop(self) -> None:
        """A call in a loop has the calls of the next iterations after it."""
        text = ("(\n    flock 9\n    for t in a b; do\n        stage_build $t x y\n    done\n"
                ") 9> build/.container.lock > log 2>&1 || die x\n")
        self.assertEqual(unguarded_calls(text), [4])


if __name__ == "__main__":
    unittest.main()
