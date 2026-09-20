#!/usr/bin/env python3
"""Run one measurement on the phone under the thermal protocol, and store the result.

The protocol this driver enforces, because a run that breaks it produces a
number nobody can use:

- The phone must not be on a charger. A charging phone clocks differently and
  heats up.
- The thermal status must be 0 before the run and 0 after it. A run that starts
  cool and ends throttled is not comparable with one that does not.
- No single invocation may exceed the time limit, 120 seconds by default.
- Exactly one transport. Two attached transports make a bare adb command fail
  with "more than one device/emulator", thus the serial is always explicit.

Every run writes one JSON record that names the device, the libraries, the
event set and the whole environment, thus a number can always be attributed to
a build.

Usage:
    tools/prof/run.py check
    tools/prof/run.py bench --model /sdcard/qwen/models/Qwen3.5-4B-Q8_0.gguf --set stalls
    tools/prof/run.py bench --model ... --set bandwidth   # runs both passes

This driver never installs anything and never writes outside /data/local/tmp on
the phone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pmu  # type: ignore[import-not-found]  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STORE = REPO / "tools" / "prof" / "store"
DEFAULT_TIMEOUT = 120


class ProtocolError(RuntimeError):
    """The phone is not in a state where a measurement would be meaningful."""


@dataclass
class Device:
    """The phone and the state that decides whether a run may start.

    Attributes:
        serial: The adb serial of the one transport this run uses
        model: The product model string
        charging: True when any power source is attached
        thermal: The thermal status, where 0 is none
        battery: The battery level in percent
    """

    serial: str
    model: str
    charging: bool
    thermal: int
    battery: int


def _adb(serial: str, args: list[str], timeout: int = 30) -> str:
    """Run one adb command against one transport and return its output.

    Args:
        serial: The adb serial
        args: The argument vector after the serial
        timeout: The wall-clock limit in seconds

    Returns:
        The combined output, stripped

    Raises:
        subprocess.TimeoutExpired: If the command outlives the limit
        subprocess.CalledProcessError: If adb returns non-zero
    """
    cmd = ["adb", "-s", serial, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    return (r.stdout + r.stderr).strip()


def transports() -> list[str]:
    """Every attached adb transport.

    Returns:
        The serials, in the order adb lists them
    """
    r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=20, check=True)
    out = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            out.append(parts[0])
    return out


def pick_serial(explicit: str | None) -> str:
    """Choose the one transport to use, and refuse an ambiguous choice.

    Two transports are often attached at once, one over USB and one over TCP.
    A bare adb command then fails, thus this driver always names the serial.

    Args:
        explicit: A serial from the command line or the environment, or None

    Returns:
        The serial to use

    Raises:
        ProtocolError: If no transport is attached, or the choice is ambiguous
    """
    have = transports()
    if not have:
        raise ProtocolError("no adb transport is attached")
    if explicit:
        if explicit not in have:
            raise ProtocolError(f"{explicit} is not attached. Attached: {', '.join(have)}")
        return explicit
    env = os.environ.get("ANDROID_SERIAL")
    if env:
        return pick_serial(env)
    if len(have) > 1:
        raise ProtocolError(
            "more than one transport is attached and none was named: "
            + ", ".join(have)
            + ". Pass --serial, or set ANDROID_SERIAL. "
            "Prefer the TCP transport, because the USB cable charges the phone.")
    return have[0]


def read_device(serial: str) -> Device:
    """Read the state of the phone that the protocol gates on.

    Args:
        serial: The adb serial

    Returns:
        The device state
    """
    model = _adb(serial, ["shell", "getprop", "ro.product.model"])
    batt = _adb(serial, ["shell", "dumpsys", "battery"])
    powered = any(
        re.search(rf"{k} powered:\s*true", batt, re.I)
        for k in ("AC", "USB", "Wireless"))
    level = int(m.group(1)) if (m := re.search(r"^\s*level:\s*(\d+)", batt, re.M)) else -1
    th = _adb(serial, ["shell", "dumpsys", "thermalservice"])
    status = int(m.group(1)) if (m := re.search(r"Thermal Status:\s*(\d+)", th)) else -1
    return Device(serial, model, powered, status, level)


def gate(d: Device, allow_charging: bool = False) -> None:
    """Refuse to measure when the phone is not in a comparable state.

    Args:
        d: The device state
        allow_charging: True to permit a charging phone, for a smoke test only

    Raises:
        ProtocolError: If the state would make the number meaningless
    """
    if d.charging and not allow_charging:
        raise ProtocolError(
            "the phone is on a charger. Pull the cable and use the TCP transport. "
            "Pass --allow-charging only for a smoke test, never for a number.")
    if d.thermal != 0:
        raise ProtocolError(f"the thermal status is {d.thermal}, not 0. Let the phone cool.")
    if 0 <= d.battery < 20:
        raise ProtocolError(f"the battery is at {d.battery} %. Charge it, then measure off the cable.")


def cmd_check(a: argparse.Namespace) -> int:
    """Report whether the phone is ready for a measurement.

    Args:
        a: The parsed arguments

    Returns:
        0 when the phone is ready, 1 when it is not
    """
    serial = pick_serial(a.serial)
    d = read_device(serial)
    print(f"serial    {d.serial}")
    print(f"model     {d.model}")
    print(f"charging  {d.charging}")
    print(f"thermal   {d.thermal}")
    print(f"battery   {d.battery} %")
    try:
        gate(d, a.allow_charging)
    except ProtocolError as e:
        print(f"\nNOT READY: {e}")
        return 1
    print("\nready")
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    """Run llama-bench on the phone once per pass of the chosen event set.

    Profiling mode 2 prints one line per operation per batch, which is roughly
    900 lines per decoded token. Streaming that over adb is slow and fragile,
    thus the run writes to a file on the phone and this function pulls it.

    Args:
        a: The parsed arguments

    Returns:
        The exit status
    """
    serial = pick_serial(a.serial)
    d = read_device(serial)
    gate(d, a.allow_charging)

    libdir = a.libdir or str(Path(a.bindir).parent / "lib")
    STORE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    _, passes = pmu.SETS[a.set]
    records = []
    for i, names in enumerate(passes, 1):
        evt = ",".join(f"0x{pmu.EVENTS[n][0]:x}" for n in names)
        remote_log = f"/data/local/tmp/qwen/logs/prof-{stamp}-{a.set}-p{i}.log"
        inner = (
            f"cd {shlex.quote(a.bindir)} && "
            f"LD_LIBRARY_PATH={shlex.quote(libdir)} "
            f"ADSP_LIBRARY_PATH={shlex.quote(libdir)} "
            f"GGML_HEXAGON_PROFILE={evt} "
            f"./llama-bench -m {shlex.quote(a.model)} "
            f"-dev {a.dev} -ngl 99 -p {a.pp} -n {a.tg} -r {a.reps} -v "
            f"> {remote_log} 2>&1; echo rc=$?; tail -n 12 {remote_log}")
        print(f"--- pass {i} of {len(passes)}: {', '.join(names)}", file=sys.stderr)
        t0 = time.time()
        try:
            out = _adb(serial, ["shell", inner], timeout=a.timeout)
        except subprocess.TimeoutExpired:
            print(f"pass {i} exceeded {a.timeout} s and was killed. "
                  f"The partial log is at {remote_log} on the phone.", file=sys.stderr)
            return 1
        secs = round(time.time() - t0, 1)
        local_log = STORE / f"{stamp}-{a.set}-p{i}.log"
        try:
            _adb(serial, ["pull", remote_log, str(local_log)], timeout=120)
            size = local_log.stat().st_size
        except subprocess.CalledProcessError:
            size = -1
        records.append({"pass": i, "events": names, "events_hex": evt,
                        "seconds": secs, "log": local_log.name, "log_bytes": size,
                        "tail": out})
        print(out)
        print(f"[pass {i}: {secs} s, log {size} bytes -> {local_log.name}]", file=sys.stderr)

    after = read_device(serial)
    rec = {
        "when": stamp, "set": a.set, "model": a.model, "dev": a.dev,
        "bindir": a.bindir, "libdir": libdir,
        "pp": a.pp, "tg": a.tg, "reps": a.reps,
        "device_before": asdict(d), "device_after": asdict(after),
        "passes": records,
    }
    path = STORE / f"{stamp}-{a.set}.json"
    path.write_text(json.dumps(rec, indent=2))
    print(f"\nstored {path.relative_to(REPO)}", file=sys.stderr)

    if after.thermal != 0:
        print(f"WARNING: the thermal status ended at {after.thermal}, not 0. "
              "The later passes are not comparable with the first.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the chosen subcommand.

    Args:
        argv: The argument vector, or None for sys.argv

    Returns:
        The exit status
    """
    # The shared options sit on a parent parser, thus they are accepted both
    # before and after the subcommand name. argparse otherwise refuses them
    # after it, which is where a reader naturally puts them.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--serial", default=None, help="the adb transport to use")
    shared.add_argument("--allow-charging", action="store_true",
                        help="permit a charging phone. For a smoke test only, never for a number")

    ap = argparse.ArgumentParser(description=__doc__, parents=[shared],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", parents=[shared], help="is the phone ready for a measurement")
    c.set_defaults(fn=cmd_check)

    b = sub.add_parser("bench", parents=[shared],
                       help="run llama-bench once per pass of an event set")
    b.add_argument("--model", required=True, help="the GGUF path on the phone")
    b.add_argument("--set", default="stalls", choices=sorted(pmu.SETS))
    b.add_argument("--bindir", default="/data/local/tmp/qwen/htpapk/bin")
    b.add_argument("--libdir", default=None,
                   help="the shared libraries, default <bindir>/../lib")
    b.add_argument("--dev", default="HTP0")
    b.add_argument("--pp", type=int, default=512)
    b.add_argument("--tg", type=int, default=128)
    b.add_argument("--reps", type=int, default=5)
    b.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    b.set_defaults(fn=cmd_bench)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ProtocolError as e:
        print(f"protocol: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
