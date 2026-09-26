"""The check after the pull of the stage v81: the phone directory goes only when the laptop has each file.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import unittest

from support import FakePhone, load

v81 = load("v81")


def check_line(target: object, kit: bool) -> str:
    """The line of the output block that removes the phone directory."""
    return next(ln for ln in v81.output_lines(target, kit, v81.runs(target, kit)) if f"rm -rf {target.phone}" in ln)


def run_files(names: list[str]) -> list[str]:
    """The three files of each run whose gate passed."""
    return [f"{n}{s}" for n in names for s in ("-gate.txt", ".out", ".log")]


class PullCheck(unittest.TestCase):
    """The check of the pull of the kit set, whose pull also holds the two directories of the ISA probe."""

    def setUp(self) -> None:
        """The run names of the kit set, and the files of a full pull."""
        self.target = v81.KIT
        self.names = [r.name for r in v81.runs(self.target, True)]
        self.probe = ["probe-info/stdout.txt", "probe-census/stdout.txt", "probe-census/r0001.bin"]
        self.full = run_files(self.names) + self.probe
        self.local = f"{v81.LAPTOP_STAGE}/{self.target.out_dir}"

    def run_check(self, phone: list[str], laptop: list[str], stopped: tuple[str, ...] = ()) -> tuple[str, list[str]]:
        """Run the check line with a phone that holds `phone` and a laptop that holds `laptop`. The gate files
        of the runs in `stopped` have the text of a gate that stopped the run."""
        fake = FakePhone(phone)
        try:
            fake.put(self.local, [f for f in laptop if not any(f == f"{s}-gate.txt" for s in stopped)])
            fake.put(self.local, [f"{s}-gate.txt" for s in stopped if f"{s}-gate.txt" in laptop], gate_ok=False)
            text = fake.run(check_line(self.target, True))
            return text, fake.calls()
        finally:
            fake.close()

    def test_full_pull_removes_the_phone_directory(self) -> None:
        """A pull with each file removes the phone directory."""
        text, calls = self.run_check(self.full, self.full)
        self.assertTrue(any("rm -rf" in c for c in calls), text)

    def test_a_lost_file_keeps_the_phone_directory_and_is_named(self) -> None:
        """A pull that loses two files of the runs keeps the phone directory and names the two files. The two
        directories of the probe hold more files than the lost ones, thus a count of the pull passes here."""
        lost = [f"{self.names[0]}.out", f"{self.names[-1]}.log"]
        text, calls = self.run_check(self.full, [f for f in self.full if f not in lost])
        self.assertFalse(any("rm -rf" in c for c in calls), text)
        for f in lost:
            self.assertIn(f, text)

    def test_a_lost_file_of_the_probe_is_named(self) -> None:
        """A file that the phone holds and the laptop does not hold is named, also outside the runs."""
        text, calls = self.run_check(self.full, [f for f in self.full if f != "probe-census/r0001.bin"])
        self.assertFalse(any("rm -rf" in c for c in calls), text)
        self.assertIn("probe-census/r0001.bin", text)

    def test_a_run_that_the_gate_stopped_is_not_a_loss(self) -> None:
        """A run that the gate stopped writes only its gate file, thus its missing .out and .log are not a
        loss, and the phone directory goes."""
        stopped = self.names[1]
        files = [f for f in self.full if f not in (f"{stopped}.out", f"{stopped}.log")]
        text, calls = self.run_check(files, files, stopped=(stopped,))
        self.assertTrue(any("rm -rf" in c for c in calls), text)

    def test_a_lost_expected_file_is_named_one_time(self) -> None:
        """The probe output is an expected file and a file of the phone, and the fault names it one time."""
        text, calls = self.run_check(self.full, [f for f in self.full if f != "probe-info/stdout.txt"])
        self.assertFalse(any("rm -rf" in c for c in calls), text)
        self.assertEqual(text.count("probe-info/stdout.txt"), 1, text)

    def test_the_phone_set_with_its_adb_serial(self) -> None:
        """The phone set calls adb with -s: a lost file keeps the phone directory, a full pull removes it."""
        target = v81.PHONE
        names = [r.name for r in v81.runs(target, False)]
        full = run_files(names)
        local = f"{v81.LAPTOP_STAGE}/{target.out_dir}"
        for laptop, removed in ((full, True), ([f for f in full if f != f"{names[3]}.log"], False)):
            fake = FakePhone(full)
            try:
                fake.put(local, laptop)
                text = fake.run(check_line(target, False))
                self.assertEqual(any("rm -rf" in c for c in fake.calls()), removed, text)
                self.assertTrue(all(c.startswith("-s ") for c in fake.calls()), fake.calls())
                if not removed:
                    self.assertIn(f"{names[3]}.log", text)
            finally:
                fake.close()

    def test_a_run_with_no_gate_file_is_named(self) -> None:
        """A run with no gate file on the laptop is named, also when the phone does not hold it."""
        gone = self.names[2]
        files = [f for f in self.full if not f.startswith(f"{gone}-") and not f.startswith(f"{gone}.")]
        text, calls = self.run_check(files, files)
        self.assertFalse(any("rm -rf" in c for c in calls), text)
        self.assertIn(f"{gone}-gate.txt", text)


if __name__ == "__main__":
    unittest.main()
