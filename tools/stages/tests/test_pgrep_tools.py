"""The pgrep line after each run of quick and unary-rows names the tool of the run.

The pgrep line prints the process id of each tool of the stage that still runs after a run. A tool that the line
does not name can stay alive after its timeout, and no line of the runner log shows it.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import unittest

from support import load

quick = load("quick")
rows = load("unary-rows")


class PgrepLines(unittest.TestCase):
    """The last line of each run is the pgrep line."""

    def check(self, stage: object) -> None:
        """Each run of `stage` (a quick.Stage) has a pgrep line that names its tool, cut to 15 characters."""
        for b, rnd, v in stage.runs():
            pgrep = stage.run_lines(b, rnd, v)[-1]
            self.assertIn(f"pgrep -x {b.tool[:15]};", pgrep, f"{b.key}-{rnd}-{v.key}")

    def test_quick(self) -> None:
        """The runs of quick."""
        self.check(quick.QUICK)

    def test_unary_rows(self) -> None:
        """The runs of unary-rows, which also runs unarycheck."""
        self.check(rows.STAGE)


if __name__ == "__main__":
    unittest.main()
