"""The NSP read table of bw prints tmin, the median rate of the slowest thread, which its header names.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import tempfile
import unittest
from pathlib import Path

from support import load

bw = load("bw")

GATE = ("gate: screen=Awake thermal=0 cap0=3532800 cap7=4320000 battery=27% temp=328 charger=ac0/usb0/wl0\n"
        "gate: OK\nbefore: nsp=41700\nrc=0\nafter: thermal=0 cap0=3532800 cap7=4320000 battery=26 temp=324 nsp=50200\n")


def nsp_line(rep: int, gbs: float, tmin: float) -> str:
    """One ddrbw nsp line of the configuration hvx-t2 with the vote set max."""
    return (f"ddrbw: nsp votes=max cfg=hvx-t2 rep={rep} threads=2 dma=0 chunk=65536 row=0 depth=0 pf=2 flags=0 "
            f"region_kib=0 bytes=7799177216 ms=300.00 gbs={gbs:.2f} tmin={tmin:.2f} tmax=13.00 ghz=2.112 "
            f"t_go=1 t_end=2 status=0 check=ok\n")


class NspTable(unittest.TestCase):
    """One nsp run with three repetitions of one configuration."""

    def test_the_cell_prints_the_median_of_tmin(self) -> None:
        """The cell of hvx-t2 under the vote set max holds the median of tmin of its three repetitions."""
        run = next(r for r in bw.runs() if r.name == "nsp-1-max")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nsp-1-max-gate.txt").write_text(GATE)
            (root / "nsp-1-max.out").write_text(nsp_line(1, 26.0, 12.10) + nsp_line(2, 25.0, 11.70)
                                                + nsp_line(3, 24.0, 12.90))
            result = bw.read_result(root, run)
        lines, _ = bw.nsp_tables([result], True)
        row = next(ln for ln in lines if ln.strip().startswith("hvx-t2"))
        self.assertIn("25.00 [24.0-26.0] n3", row)
        self.assertIn("tmin 12.10", row)


if __name__ == "__main__":
    unittest.main()
