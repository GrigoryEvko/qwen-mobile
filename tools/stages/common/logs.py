"""The text of the output files of one run, and the marks of the runner log."""

import gzip
import re
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# The title line of a run in a command file and in the log of the laptop runner.
TITLE_RE = re.compile(r"^# (?:REAL-MODEL|OPS|OP-TEST|KERNEL|NO-MODEL|SWEEP|MMSOLVE)[^:]*: (\S+?),", re.M)
CAPS_CHANGED_RE = re.compile(r"^CAPS .*CAPS-CHANGED", re.M)


def read_text(path: Path, *, strip_ansi: bool = False) -> str:
    """The text of a file, of its gzip form <name>.z, or an empty text when neither exists.

    A profile run gzips its log on the phone, because one profile line for each op makes a log of 100 MB.
    A byte that is not UTF-8 becomes the replacement character, thus a truncated log still reads.
    """
    for p in (path, path.with_name(path.name + ".z")):
        if p.exists():
            data = p.read_bytes()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            text = data.decode(errors="replace")
            return ANSI_RE.sub("", text) if strip_ansi else text
    return ""


def read_run(root: Path, name: str, suffixes: tuple[str, ...] = ("-gate.txt", ".out", ".log"),
             *, strip_ansi: bool = False) -> tuple[str, ...]:
    """The text of each output file of one run, in the order of `suffixes`."""
    return tuple(read_text(root / f"{name}{s}", strip_ansi=strip_ansi) for s in suffixes)


def runner_marks(path: Path | None, *, title_re: re.Pattern = TITLE_RE) -> dict[str, list[str]]:
    """The marks of the laptop runner for each run name, from its log.

    The runner prints a CAPS line for each model run, and it marks the line CAPS-CHANGED when the caps of
    the phone changed during the run. The gate file of the run does not hold that mark, thus a table that
    drops such a run needs this log. An empty path gives no mark. O(size of the log).
    """
    marks: dict[str, list[str]] = {}
    if path is None or not path.exists():
        return marks
    name = None
    for line in path.read_text(errors="replace").splitlines():
        m = title_re.match(line)
        if m:
            name = m.group(1)
        elif name is not None and CAPS_CHANGED_RE.match(line):
            marks.setdefault(name, []).append("the runner marks CAPS-CHANGED")
    return marks
