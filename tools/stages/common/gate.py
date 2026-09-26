"""The conditions of one phone run, from its gate file.

bin/gate.sh writes the first line before the run, and the command of the run adds the exit code and the
conditions after the run. Thus the gate file holds the state of the phone before and after the run:

    gate: screen=Awake thermal=0 cap0=3532800 cap7=4320000 battery=71% temp=331 charger=... MemAvailable=...
    before: nsp=61300
    rc=0
    after: thermal=0 cap0=3532800 cap7=4320000 battery=70 temp=335 nsp=78200

Thermal status 0 does not mean full clocks: the big-core cap can collapse while the status stays 0. Thus
a run is comparable only when the caps are the same before and after it, and a pair of runs is valid only
when the two runs have the same caps.
"""

import re
import statistics
from dataclasses import dataclass, field

GATE_RE = re.compile(r"gate: screen=(\S+) thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*)% temp=(\d*)")
AFTER_RE = re.compile(r"after: thermal=(\d*) cap0=(\d*) cap7=(\d*) battery=(\d*) temp=(\d*)(?: nsp=(\d*))?")
BEFORE_RE = re.compile(r"before: nsp=(\d*)")
RC_RE = re.compile(r"^rc=(\d+)", re.M)

# A run whose lowest cap is below this value ran at a collapsed clock. Its times are not comparable with
# the times of a run at the full caps, thus the tables leave it out.
CAP_MIN_KHZ = 3000000

# The lines of a log that tell of a failure of the DSP session or of a check.
FAIL_RE = re.compile(r"follow-failed|GGML_ASSERT|dspqueue_read failed|AddressSanitizer")


@dataclass(frozen=True)
class Conditions:
    """The conditions of one run.

    ok is True when the gate passed and the tool ran to its end with a permitted exit code. faults names
    each problem of the run itself, marks names each condition that makes the run not comparable, and
    removed names each condition that keeps the run out of the tables. caps is the pair of clock caps of
    cpu0 and cpu7 in kHz before the run, and nsp is the highest NPU zone temperature in C before and
    after the run.
    """
    ok: bool
    rc: int | None
    caps: str
    battery: str
    nsp: tuple[float | None, float | None]
    faults: list[str] = field(default_factory=list)
    marks: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        """The faults and the marks in one list, in the order of the gate file."""
        return self.faults + self.marks


def read(gate: str, *, ok_codes: tuple[int, ...] = (0,), cap_min: int | None = None) -> Conditions:
    """The conditions of one run from the text of its gate file.

    Args:
        gate: The text of the gate file, or an empty text when the run has no gate file
        ok_codes: The exit codes that count as a run to the end. An exit code that is not 0 is a fault
            of the run also when it is permitted
        cap_min: The lowest cap (kHz) of a comparable run. With a value, a lower cap and a thermal status
            after the run go into removed. Without one, the thermal status goes into marks
    """
    before, after = GATE_RE.search(gate), AFTER_RE.search(gate)
    m = RC_RE.search(gate)
    rc = int(m.group(1)) if m else None
    ok = "gate: OK" in gate and rc in ok_codes
    faults, marks, removed = [], [], []
    if not gate:
        faults.append("no gate file")
    elif "gate: OK" not in gate:
        faults.append("the gate stopped the run")
    elif rc != 0:
        faults.append(f"exit code {rc if rc is not None else '?'}")
    caps = f"{before.group(3)}/{before.group(4)}" if before else "?"
    if before and after and (before.group(3), before.group(4)) != (after.group(2), after.group(3)):
        marks.append(f"caps {caps} -> {after.group(2)}/{after.group(3)}")
    thermal = f"thermal {after.group(1)} after the run" if after and after.group(1) not in ("", "0") else ""
    if cap_min is not None:
        values = [int(v) for v in ((before.group(3), before.group(4)) if before else ())
                  + ((after.group(2), after.group(3)) if after else ()) if v]
        if values and min(values) < cap_min:
            removed.append(f"a cap of {min(values)} kHz")
        if thermal:
            removed.append(thermal)
    elif thermal:
        marks.append(thermal)
    battery = f"{before.group(5)}% {int(before.group(6)) / 10:.1f} C" if before and before.group(6) else "?"
    nsp_before = BEFORE_RE.search(gate)
    nsp = (int(nsp_before.group(1)) / 1000 if nsp_before and nsp_before.group(1) else None,
           int(after.group(6)) / 1000 if after and after.group(6) else None)
    return Conditions(ok, rc, caps, battery, nsp, faults, marks, removed)


def one_line(gate: str) -> str:
    """The exit code, the caps before the run, and each mark of one run, in one line."""
    c = read(gate)
    return f"rc={c.rc if c.rc is not None else '?'} caps {c.caps}" + (" " + ", ".join(c.marks) if c.marks else "")


def caps_counts(items: list[Conditions]) -> str:
    """One line with the count of the runs of each pair of caps. Only a run that ran counts."""
    counts: dict[str, int] = {}
    for c in items:
        if c.ok:
            counts[c.caps] = counts.get(c.caps, 0) + 1
    return "  caps cpu0/cpu7 kHz before the runs: " + ", ".join(f"{k} x{n}" for k, n in counts.items())


def nsp_range(items: list[Conditions], when: int) -> str:
    """One line with the lowest, the highest and the median NPU zone temperature of the runs that ran, or
    an empty text when the phone gave no temperature. `when` is 0 for before the runs and 1 for after."""
    temps = [c.nsp[when] for c in items if c.ok and c.nsp[when] is not None]
    if not temps:
        return ""
    word = ("before", "after")[when]
    return (f"  NPU zone temperature {word} the runs: {min(temps):.1f} to {max(temps):.1f} C, "
            f"median {statistics.median(temps):.1f} C")
