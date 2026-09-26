"""device.lib_env gives the host and the DSP library paths of a run from the directories below the stage.

Run from the repository root: python3 -m unittest discover -s tools/stages/tests
"""

import sys
import unittest

from support import STAGES

sys.path.insert(0, str(STAGES))
from common import device  # noqa: E402

S = "/data/local/tmp/qwen/stage"


class LibEnv(unittest.TestCase):
    """The forms of the paths that the stages use."""

    def test_one_directory_for_the_host_and_the_dsp(self) -> None:
        """The preset gives lib to the host and to the DSP."""
        self.assertEqual(device.lib_env(S), f"LD_LIBRARY_PATH={S}/lib ADSP_LIBRARY_PATH={S}/lib")

    def test_two_host_directories_and_the_dsp_of_the_first(self) -> None:
        """Without dsp, the DSP uses the first host directory."""
        self.assertEqual(device.lib_env(S, "lib-new:lib-base"),
                         f"LD_LIBRARY_PATH={S}/lib-new:{S}/lib-base ADSP_LIBRARY_PATH={S}/lib-new")

    def test_a_dsp_directory_apart_from_the_host(self) -> None:
        """A dsp directory that is not the first host directory."""
        self.assertEqual(device.lib_env(S, "lib-new:lib-base", "lib-base"),
                         f"LD_LIBRARY_PATH={S}/lib-new:{S}/lib-base ADSP_LIBRARY_PATH={S}/lib-base")

    def test_the_dsp_path_only(self) -> None:
        """An empty host gives only the DSP path."""
        self.assertEqual(device.lib_env(S, "", "hmx"), f"ADSP_LIBRARY_PATH={S}/hmx")

    def test_no_directory_is_an_error(self) -> None:
        """An empty host and no dsp give no DSP directory."""
        with self.assertRaises(ValueError):
            device.lib_env(S, "")


if __name__ == "__main__":
    unittest.main()
