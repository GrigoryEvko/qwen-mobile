"""The rate table of unary-kl: a small step of a cap during a run keeps the run in, a collapsed cap keeps it out,
and the table names each run that it leaves out.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from support import load

ukl = load("unary-kl")


def gate_text(cap7_after: int) -> str:
    """The gate file of a run that passed, whose cpu7 cap goes from 4320000 kHz to cap7_after kHz."""
    return ("gate: screen=Awake thermal=0 cap0=3532800 cap7=4320000 battery=80% temp=300 charger=ac0/usb0/wl0\n"
            "gate: OK\nbefore: nsp=50000\nrc=0\n"
            f"after: thermal=0 cap0=3532800 cap7={cap7_after} battery=79 temp=305 nsp=60000\n")


def bench_out(ts: float) -> str:
    """The llama-bench output of the tests of the blocks p and g, each at the rate ts."""
    tests = ((512, 0), (1024, 0), (0, 32))
    return "".join(json.dumps({"n_prompt": p, "n_gen": g, "n_depth": 0, "samples_ts": [ts, ts]}) + "\n" for p, g in tests)


class RateTable(unittest.TestCase):
    """The llama-bench runs of the blocks p and g."""

    def table(self, cap7: dict[str, int], include_all: bool = False) -> str:
        """The rate part of the table for pulled runs whose cpu7 cap after the run is cap7[name], 4089600 kHz
        for a run that cap7 does not name."""
        names = [f"{b.key}-{r}-{v.key}" for b, r, v in ukl.runs() if b.tool == "llama-bench"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in names:
                (root / f"{n}-gate.txt").write_text(gate_text(cap7.get(n, 4089600)))
                (root / f"{n}.out").write_text(bench_out(100.0))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ukl.table(root, include_all)
        text = buf.getvalue()
        return text[text.index("llama-bench t/s"):]

    def test_a_small_step_of_a_cap_keeps_the_run(self) -> None:
        """The cpu7 cap of this phone steps down a little at the end of a run. Such runs go into the rows."""
        text = self.table({})
        self.assertIn("pp512 tg0 d0:", text)
        self.assertIn("A 100.00", text)

    def test_a_collapsed_cap_leaves_the_run_out_by_name(self) -> None:
        """A run whose cap falls below the limit goes out, and the table names it with the reason."""
        text = self.table({"p-1-f": 2438400})
        self.assertIn("pp512 tg0 d0:", text)
        self.assertIn("p-1-f", text)
        self.assertIn("a cap of 2438400 kHz", text)

    def test_an_empty_table_tells_why(self) -> None:
        """When no run qualifies, the table names each run and why it is out."""
        names = [f"{b.key}-{r}-{v.key}" for b, r, v in ukl.runs() if b.tool == "llama-bench"]
        text = self.table({n: 2438400 for n in names})
        self.assertNotIn("pp512 tg0 d0:", text)
        for n in names:
            self.assertIn(n, text)


if __name__ == "__main__":
    unittest.main()
