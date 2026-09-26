"""The caps and the thermal status after a run are fields of gate.Conditions, apart from the text of the marks.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import sys
import unittest

from support import STAGES

sys.path.insert(0, str(STAGES))
from common import gate  # noqa: E402

BEFORE = "gate: screen=Awake thermal=0 cap0=3532800 cap7=4320000 battery=71% temp=331 charger=ac0/usb0/wl0\ngate: OK\n"


def gate_text(thermal: str, cap7: str) -> str:
    """A gate file of a run with exit code 0, and the thermal status and the cap of cpu7 after the run."""
    return BEFORE + f"before: nsp=61300\nrc=0\nafter: thermal={thermal} cap0=3532800 cap7={cap7} battery=70 temp=335\n"


class Fields(unittest.TestCase):
    """The fields caps_after and thermal_after, and the marks that come from them."""

    def test_a_run_with_no_change_has_no_mark(self) -> None:
        """The same caps and the thermal status 0 give no mark, and the fields hold the values."""
        c = gate.read(gate_text("0", "4320000"))
        self.assertEqual((c.caps_after, c.thermal_after), ("3532800/4320000", 0))
        self.assertEqual((c.caps_mark, c.thermal_mark, c.marks), ("", "", []))

    def test_the_two_marks_and_their_order(self) -> None:
        """A changed cap and a thermal status above 0 give the two marks, the caps first."""
        c = gate.read(gate_text("2", "4204800"))
        self.assertEqual(c.caps_mark, "caps 3532800/4320000 -> 3532800/4204800")
        self.assertEqual(c.thermal_mark, "thermal 2 after the run")
        self.assertEqual(c.marks, [c.caps_mark, c.thermal_mark])

    def test_the_cap_rule_moves_the_thermal_mark_to_removed(self) -> None:
        """With cap_min, the thermal mark goes into removed, and the caps mark stays in marks."""
        c = gate.read(gate_text("2", "4204800"), cap_min=gate.CAP_MIN_KHZ)
        self.assertEqual((c.marks, c.removed), ([c.caps_mark], [c.thermal_mark]))

    def test_no_after_line_gives_no_values(self) -> None:
        """A gate file without the line "after:" gives no value and no mark."""
        c = gate.read(BEFORE + "rc=0\n")
        self.assertEqual((c.caps_after, c.thermal_after, c.caps_mark, c.thermal_mark), ("?", None, "", ""))

    def test_an_empty_thermal_value_gives_none(self) -> None:
        """An after line whose thermal value is empty gives None, not 0."""
        self.assertIsNone(gate.read(gate_text("", "4320000")).thermal_after)


if __name__ == "__main__":
    unittest.main()
