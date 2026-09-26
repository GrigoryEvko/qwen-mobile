"""The phone and the box layout: the device, its directories, and the conditions of a run.

The laptop runner reads a command file line by line and evaluates each line in its own subshell. Thus
every line holds its full paths, and a line that assigns a variable does not reach the next line. A line
that holds the text "models/Qwen3.5" is a model run for the runner: it waits for the unlocked phone,
wakes the screen, gates the memory and prints the caps.

One run writes three files into the out directory of the stage on the phone: <name>-gate.txt with the
conditions, <name>.out with the stdout of the tool, and <name>.log with its stderr.
"""

import os
from pathlib import Path
from typing import NamedTuple

# The phone over Wi-Fi. The USB transport (adb -s b704c0b4) is the fallback when Wi-Fi drops.
ADB = "adb -s 192.168.14.130:5555"
PHONE_ROOT = "/data/local/tmp/qwen"
MODEL_DIR = f"{PHONE_ROOT}/models"
EVAL_DIR = f"{PHONE_ROOT}/eval"
MODEL_4B = "Qwen3.5-4B-Q8_0.gguf"
MODEL_2B = "Qwen3.5-2B-Q8_0.gguf"
MMPROJ_4B = "Qwen3.5-4B-Q8_0.mmproj.gguf"
# The photo of the image stages: the copy on the laptop and its sha1. Every image stage uses one photo,
# thus the image token count and the embeddings of two stages are comparable.
PHOTO_LOCAL = "build/imgturn/photo.jpg"
PHOTO_SHA1 = "a2200d1a726ec0a8576b4a18dc2ef1aa4e4c797d"
BOX_BUILD = "grigory@10.10.20.200:airi/qwen-mobile/build"
LOCAL_BUILD = "build"

# The MemAvailable (KiB) that the gate requires. A run of the 4B needs 8 GiB, a run of the 2B 6 GiB, and
# a run that loads no model 2 GiB.
GATE_4B_KB = 8388608
GATE_2B_KB = 6291456
GATE_TOOL_KB = 2097152

# The environment of the app: the two fusion switches are on.
APP_ENV = "GGML_HEXAGON_OPFUSION=1 GGML_HEXAGON_OPFUSION_STATE=1"
# The memprobe arguments of the app context: the Q8_0 KV cache and the lazy embedding.
PROBE_ARGS = "-dev HTP0 -c 8192 -t 4 --outputs-max 5 --lazy on -ctk q8_0 -ctv q8_0"

THERMAL = f"{ADB} shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
# The highest NPU zone temperature in milli-degrees C, or an empty text when the phone gives no such zone.
NSP = ('$(for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type 2>/dev/null)" in (nsp*) '
       'cat $z/temp 2>/dev/null;; esac; done | sort -n | tail -n 1)')
BEFORE = f'echo "before: nsp={NSP}"'
AFTER = ('echo "after: thermal=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc 0-9)'
         ' cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)'
         ' cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)'
         ' battery=$(dumpsys battery | grep "^  level:" | tr -dc 0-9)'
         f' temp=$(dumpsys battery | grep "temperature:" | tr -dc 0-9) nsp={NSP}"')


class StagePaths(NamedTuple):
    """The four directories of one stage: on the phone, on the laptop, on the box, and from the current
    directory. `dir` is a relative path, thus the tool prints the same path from the repository root and
    through the link build/<name>/stage.py."""
    phone: str
    local: str
    box: str
    dir: Path


def stage_paths(name: str, file: str) -> StagePaths:
    """The directories of the stage `name`. `file` is the __file__ of the stage module: the repository
    root is its third parent, thus tools/stages/<name>/stage.py and the link build/<name>/stage.py find
    the same stage directory."""
    root = Path(file).resolve().parents[3]
    local = f"{LOCAL_BUILD}/{name}"
    return StagePaths(f"{PHONE_ROOT}/{name}", local, f"{BOX_BUILD}/{name}",
                      Path(os.path.relpath(root / local)))


def pgrep(*tools: str) -> str:
    """The line that prints the process id of each tool of the stage that still runs. The kernel keeps 15
    characters of a process name, thus a longer name goes in cut to 15 characters."""
    return f"{ADB} shell '" + "; ".join(f"pgrep -x {t}" for t in tools) + "; echo pgrep-done'"


def lib_env(stage: str, host: str = "lib", dsp: str | None = None) -> str:
    """The library search paths of one run: LD_LIBRARY_PATH for the host and ADSP_LIBRARY_PATH for the DSP.

    Args:
        stage: The phone directory of the stage
        host: The directories of the host libraries below the stage, in the order of the search and with ":"
            between them. An empty text gives no LD_LIBRARY_PATH, for a program that loads no library of the
            stage on the host
        dsp: The directory of the DSP libraries below the stage. None gives the first directory of host

    Raises:
        ValueError: If host is empty and dsp is None
    """
    dirs = host.split(":") if host else []
    if dsp is None:
        if not dirs:
            raise ValueError("lib_env: give the DSP directory, because host gives no directory")
        dsp = dirs[0]
    ld = [f"LD_LIBRARY_PATH={':'.join(f'{stage}/{d}' for d in dirs)}"] if dirs else []
    return " ".join(ld + [f"ADSP_LIBRARY_PATH={stage}/{dsp}"])

