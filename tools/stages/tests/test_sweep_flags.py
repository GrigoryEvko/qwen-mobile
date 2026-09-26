"""The command line of sweep offers no flag that has no effect, and its docstring tells what the tables use.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import subprocess
import sys
import unittest

from support import STAGES, load

sweep = load("sweep")


class SweepFlags(unittest.TestCase):
    """The flag --all of the table of sweep never changed a table, thus the tool has no such flag."""

    def test_the_table_help_offers_no_flag_with_no_effect(self) -> None:
        """The help text of the subcommand table does not offer --all."""
        p = subprocess.run([sys.executable, str(STAGES / "sweep" / "stage.py"), "table", "--help"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("--all", p.stdout)

    def test_the_docstring_does_not_offer_the_flag(self) -> None:
        """The usage text of the module does not offer --all."""
        self.assertNotIn("--all", sweep.__doc__)

    def test_a_flagged_run_stays_in_the_tables(self) -> None:
        """A run whose gate passed and whose exit code is 0 goes into the tables also with a flag, as the
        docstring tells, and a run that failed does not."""
        run = next(r for r in sweep.runs() if r.key == "i8a")
        flagged = sweep.RunOut(run, True, ["caps 3532800/4320000 -> 3532800/4204800"], "3532800/4320000")
        failed = sweep.RunOut(run, False, ["exit code 137"], "3532800/4320000")
        self.assertTrue(sweep.usable(flagged))
        self.assertFalse(sweep.usable(failed))
        self.assertFalse(sweep.usable(None))


if __name__ == "__main__":
    unittest.main()
