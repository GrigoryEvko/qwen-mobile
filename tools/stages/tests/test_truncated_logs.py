"""A truncated output of a run: the tables of sweep and vit name it and skip the run, with no crash and no silent
partial row.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import tempfile
import unittest
from pathlib import Path

from support import load

sweep = load("sweep")
vit = load("vit")

GATE_OK = "gate: OK\nrc=0\n"


class SweepEngineTable(unittest.TestCase):
    """The engine table of sweep reads the result file of the DSP program i8read, which ends with "i8read: done"."""

    def table(self, text: str) -> str:
        """The engine table of the run i8a whose result file holds `text`."""
        run = next(r for r in sweep.runs() if r.key == "i8a")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "i8a-gate.txt").write_text(GATE_OK)
            (root / "i8a.txt").write_text(text)
            out = sweep.read_run(root, run)
        lines, _ = sweep.i8_tables({"i8a": out}, 1000.0)
        return "\n".join(lines)

    def test_one_point_of_the_fit_does_not_crash(self) -> None:
        """A result that the timeout cut after one point of the fit range gives a skip that names the file and
        the last complete record."""
        text = self.table("i8read: f16 k=1 pass=10 per_tile_x100=5\ni8read: f16 k=48 pass=100 per_tile_x100=9\n")
        self.assertIn("i8a.txt", text)
        self.assertIn("i8read: f16 k=48 pass=100 per_tile_x100=9", text)
        self.assertNotIn("f16 per output tile", text)

    def test_a_cut_record_is_not_a_point(self) -> None:
        """A result whose last line the timeout cut does not give a silent fit: the cut value is not a point."""
        text = self.table("i8read: f16 k=48 pass=100 per_tile_x100=9\ni8read: f16 k=64 pass=130 per_tile_x100=12\n"
                          "i8read: f16 k=80 pass=1")
        self.assertIn("i8a.txt", text)
        self.assertIn("i8read: f16 k=64 pass=130 per_tile_x100=12", text)
        self.assertNotIn("f16 per output tile", text)

    def test_a_complete_result_gives_the_fit(self) -> None:
        """A result with its last line gives the fit line."""
        text = self.table("i8read: f16 k=48 pass=100 per_tile_x100=9\ni8read: f16 k=64 pass=130 per_tile_x100=12\n"
                          "i8read: done\n")
        self.assertIn("i8a: f16 per output tile", text)


def encode_log(n_qkv: int, n_up: int) -> str:
    """A log with one complete encode window of a level-1 profile run: n_qkv QKV matmuls and n_up up matmuls."""
    op = "ggml-hex: HTP0 profile-op MUL_MAT+ADD|v.blk.{i}.{k}.weight x x -> y|a|b|c|kernel|usec {us} cycles 1 mhz 2000"
    lines = ["vitprobe: STAMP encode-begin rep=0"]
    lines += [op.format(i=i, k="attn_qkv", us=100 + i) for i in range(n_qkv)]
    lines += [op.format(i=i, k="ffn_up", us=200 + i) for i in range(n_up)]
    lines.append("vitprobe: STAMP encode-end rep=0")
    return "\n".join(lines) + "\n"


class VitLayerSummary(unittest.TestCase):
    """The layer table of vit reads the last complete encode window of each level-1 profile run."""

    def setUp(self) -> None:
        """A stage with level-1 profile runs, and the name of its first such run."""
        self.stage = vit.STAGES["vit6"]
        self.name = next(vit.run_name(s, vk, r) for s, vk, r in self.stage.runs if vit.PROFILE_ENV in vit.V[vk].env)

    def row(self, res: object) -> list[str]:
        """The rows of the layer table that name the run."""
        text = vit.layer_summary(self.stage, {self.name: res})
        return [ln for ln in text if ln.strip().startswith(self.name)]

    def test_one_matmul_of_a_kind_does_not_crash(self) -> None:
        """A window with one QKV matmul gives a skip that names the log file and the last complete record."""
        rows = self.row(vit.Result(self.name, True, [], "3532800/4320000", "", encode_log(1, 1)))
        self.assertEqual(len(rows), 1, rows)
        self.assertIn(f"{self.name}.log", rows[0])
        self.assertIn("vitprobe: STAMP encode-end rep=0", rows[0])

    def test_a_run_that_failed_has_a_row_with_the_reason(self) -> None:
        """A run with no usable output gets a row that tells why, not no row."""
        rows = self.row(vit.Result(self.name, False, ["the gate stopped the run"], "?", "", ""))
        self.assertEqual(len(rows), 1, rows)
        self.assertIn("the gate stopped the run", rows[0])

    def test_a_complete_window_gives_the_medians(self) -> None:
        """A window with the 24 layers gives the medians of the two groups of layers."""
        rows = self.row(vit.Result(self.name, True, [], "3532800/4320000", "", encode_log(24, 24)))
        self.assertEqual(len(rows), 1, rows)
        self.assertIn("qkv", rows[0])
        self.assertIn("[100, 123]", rows[0])


if __name__ == "__main__":
    unittest.main()
