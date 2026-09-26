"""The command line of a stage tool.

A stage tool has the same two subcommands: `commands` writes the phone command file, and `table` reads the
pulled outputs and prints the tables. A stage whose command line needs more than that builds its own
parser.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable


def run(stage_dir: Path, doc: str, *, write: Callable[[Path], int], count: Callable[[], int],
        table: Callable[..., int], all_flag: bool = True,
        all_help: str = "also use the runs with changed caps or heat",
        table_help: str = "print the tables from the pulled logs",
        runner_log: str = "", runner_log_default: bool = False) -> int:
    """Parse the command line of a stage tool and run the subcommand.

    Args:
        stage_dir: The stage directory, which gives the preset command file and the preset output directory
        doc: The docstring of the stage module. Its first line is the description of the tool
        write: The function that writes the command file and returns its line count
        count: The function that gives the number of runs of the stage
        table: The function that prints the tables. It takes the output directory, then the value of the
            --all flag when the stage has one, then the runner log when the stage takes one
        all_flag: True when the stage has the flag --all
        all_help: The help text of --all
        table_help: The help text of the subcommand table
        runner_log: The help text of --runner-log, or an empty text when the stage takes no runner log
        runner_log_default: True when --runner-log has the runner log of the stage directory as its preset
    """
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commands", help="write the phone command file")
    c.add_argument("--out", type=Path, default=stage_dir / "phone-commands.txt")
    t = sub.add_parser("table", help=table_help)
    t.add_argument("--root", type=Path, default=stage_dir / "phone-out")
    if all_flag:
        t.add_argument("--all", action="store_true", help=all_help)
    if runner_log:
        t.add_argument("--runner-log", type=Path, help=runner_log,
                       default=stage_dir / "runner.log" if runner_log_default else None)
    a = ap.parse_args()
    if a.cmd == "commands":
        n = write(a.out)
        print(f"{a.out}: {n} lines, {count()} runs")
        return 0
    args: list = [a.root]
    if all_flag:
        args.append(a.all)
    if runner_log:
        args.append(a.runner_log)
    return table(*args)


def missing_root(root: Path) -> int:
    """Print the message for an output directory that does not exist, and give the exit code 1."""
    print(f"stage.py: {root} is not a directory. Pull the outputs of the stage first.", file=sys.stderr)
    return 1
