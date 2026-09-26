"""The lines of a phone command file.

A command file starts with a header of comment lines, then the lines that copy the stage to the phone,
then five lines for each run, then the lines that pull the outputs. The runner evaluates each line in its
own subshell, thus each line holds its full paths.

The command of one run has the same shape in each stage: the gate of the run, the conditions before it,
the tool under `timeout -s KILL`, the exit code, the conditions after it, and the gate file on stdout. A
phone run must stay under two minutes.
"""

from pathlib import Path
from typing import Iterable

from common.device import ADB, AFTER, BEFORE, MODEL_DIR, PHOTO_LOCAL, PHOTO_SHA1, THERMAL, StagePaths


def gate_head(stem: str, gate_kb: int, stage: str, *, prefix: str = "", model: str = "",
              before: str = BEFORE) -> str:
    """The head of the command of one run: the gate of the run, then the conditions before it.

    Args:
        stem: The path of the run in the out directory of the phone, without a suffix
        gate_kb: The MemAvailable (KiB) that the gate requires
        stage: The stage directory on the phone
        prefix: A command that goes before the gate, for example a marker line
        model: A model file that the line names, thus the runner treats the line as a model run
        before: The line that records the conditions before the run
    """
    test = f"test -r {MODEL_DIR}/{model} && " if model else ""
    return (f"{prefix}{test}sh {stage}/bin/gate.sh {gate_kb} > {stem}-gate.txt && "
            f"{before} >> {stem}-gate.txt && ")


def gate_tail(stem: str, *, clean: str = "", after: str = AFTER) -> str:
    """The tail of the command of one run: the exit code of the tool, the conditions after the run, and
    the gate file on stdout. `clean` is a command that removes the work files of the run."""
    return f"echo \"rc=$?\" >> {stem}-gate.txt; {clean}{after} >> {stem}-gate.txt; cat {stem}-gate.txt"


def timeout_cmd(limit: int, env: str, tool: str) -> str:
    """The tool under the kill timeout, with its environment. A phone run must stay under two minutes,
    and the kill signal stops a tool that an adb session does not reach."""
    return f"timeout -s KILL {limit} env {env} {tool}" if env else f"timeout -s KILL {limit} {tool}"


def redirect(stem: str, cmd: str) -> str:
    """The command with its stdout in <name>.out and its stderr in <name>.log."""
    return f"{cmd} > {stem}.out 2> {stem}.log"


def gzip_group(stem: str, cmd: str) -> str:
    """The command in a group that gzips its stderr, when the phone has gzip. One profile line for each
    op makes a log of 100 MB, and logs.read_text reads a gzip file and a plain file alike. The exit code
    after the group is the exit code of the tool, because of `set -o pipefail`."""
    return (f"{{ Z=cat; command -v gzip > /dev/null && Z=\"gzip -1\"; set -o pipefail; "
            f"{cmd} 2>&1 > {stem}.out | $Z > {stem}.log.z; }}; ")


def gated_run(stem: str, gate_kb: int, limit: int, env: str, tool: str, *, stage: str, prefix: str = "",
              model: str = "", pre: str = "", clean: str = "", before: str = BEFORE,
              after: str = AFTER) -> str:
    """The full command of one run: the gate, the tool under the kill timeout, and the conditions.

    Args:
        stem: The path of the run in the out directory of the phone, without a suffix
        gate_kb: The MemAvailable (KiB) that the gate requires
        limit: The kill timeout in seconds
        env: The environment of the tool
        tool: The tool with its arguments
        stage: The stage directory on the phone
        prefix: A command that goes before the gate
        model: A model file that the line names, thus the runner treats the line as a model run
        pre: A command that goes after the gate and before the tool
        clean: A command that goes after the tool and before the conditions
        before: The line that records the conditions before the run
        after: The line that records the conditions after the run
    """
    return (gate_head(stem, gate_kb, stage, prefix=prefix, model=model, before=before) + pre
            + redirect(stem, timeout_cmd(limit, env, tool)) + "; "
            + gate_tail(stem, clean=clean, after=after))


def run_lines(title: str, cmd: str, pgrep_line: str, *, adb: str = ADB,
              thermal: str = THERMAL) -> list[str]:
    """The five lines of one run: an empty comment, the title, the thermal line, the run, and the pgrep
    line. The title holds its own "# "."""
    return ["#", title, thermal, f"{adb} shell '{cmd}'", pgrep_line]


def setup_lines(paths: StagePaths, pushes: dict[str, Iterable[str]], *, model_check: str = "",
                sub: str = "phone", adb: str = ADB, extra: Iterable[str] = (), sums: str = "SHA256SUMS",
                count: int = 0, chmod: Iterable[str] = ("bin",)) -> list[str]:
    """The lines that copy the stage from the box to the laptop, check its files, and push them.

    The checksum file of the stage goes to the phone last, and `sha256sum -c` on the phone makes sure
    that each file arrived. A program whose shared library the stage does not push exits with "CANNOT
    LINK EXECUTABLE", thus the checksum file must name each file of the stage.

    Args:
        paths: The directories of the stage
        pushes: The directory of the phone stage for each group of files of the laptop stage. An empty
            group gets its directory on the phone and no push line
        model_check: One line that tests the model files, or an empty text for a stage with no model
        sub: The directory of the laptop stage that holds the files
        adb: The adb command
        extra: More lines after the checksum check
        sums: The name of the checksum file, for a stage whose targets have one file each
        count: The file count of the stage. With a value, the last line prints it, thus the reader sees
            that the push moved each file that the build wrote
        chmod: The directories of the phone stage whose files become executable
    """
    stage, local = paths.phone, paths.local
    dirs = " ".join(f"{stage}/{d}" for d in list(pushes) + ["out"])
    lines = [
        f"mkdir -p {local} && rsync -a --delete {paths.box}/{sub}/ {local}/{sub}/",
        f"(cd {local}/{sub} && sha256sum -c {sums})",
    ]
    if model_check:
        lines.append(model_check)
    lines.append(f"{adb} shell 'rm -rf {stage} && mkdir -p {dirs}'")
    for d, files in pushes.items():
        group = " ".join(files)
        if group:
            lines.append(f"{adb} push {group} {stage}/{d}/")
    tail = f"; echo {count} files; " if count else " && "
    lines += [
        f"{adb} push {local}/{sub}/{sums} {stage}/",
        f"{adb} shell 'cd {stage} && sha256sum -c {sums} | grep -c OK{tail}"
        + "chmod 755 " + " ".join(f"{stage}/{d}/*" for d in chmod) + "'",
    ]
    return lines + list(extra)


def output_lines(paths: StagePaths, *, tools: Iterable[str] = (), out_dir: str = "phone-out",
                 box: str = "", adb: str = ADB, thermal: str = THERMAL,
                 extra: Iterable[str] = ()) -> list[str]:
    """The lines that pull the outputs, copy them to the box and remove the stage from the phone.

    The stage goes from the phone only when the pull has each of its files, thus a lost pull does not
    lose the run. `box` is the box directory of the outputs, when it is not the box directory of the
    stage.
    """
    stage, local = paths.phone, paths.local
    pg = "".join(f"pgrep -x {t}; " for t in tools)
    return [
        "#",
        "# ---- The outputs ----",
        "#",
        thermal,
        f"{adb} shell '{pg}ls {stage}/out | wc -l; du -sh {stage}/out'",
        f"rm -rf {local}/{out_dir}",
        f"{adb} pull {stage}/out {local}/{out_dir}",
        f"rsync -a --delete {local}/{out_dir}/ {box or paths.box}/{out_dir}/",
        f"test \"$(ls {local}/{out_dir} | wc -l)\" -eq "
        f"\"$({adb} shell 'ls {stage}/out | wc -l' | tr -d '\\r')\" "
        f"&& {adb} shell 'rm -rf {stage}' && echo removed {stage}",
    ] + list(extra)


def photo_lines(box: str, photo: str, *, adb: str = ADB) -> list[str]:
    """The lines that copy the photo of the image stages from the box, check its sha1, and push it to the
    phone. `box` is the box build directory and `photo` is the path of the photo on the phone."""
    return [
        f"mkdir -p {Path(PHOTO_LOCAL).parent} && rsync -a {box}/imgturn/photo.jpg {PHOTO_LOCAL}",
        f"echo '{PHOTO_SHA1}  {PHOTO_LOCAL}' | sha1sum -c",
        f"{adb} push {PHOTO_LOCAL} {photo}",
        f"{adb} shell 'ls -l {photo} && sha1sum {photo}'",
    ]


def write_commands(path: Path, lines: Iterable[str]) -> int:
    """Write the command file and return its line count."""
    text = list(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n")
    return len(text)


def header_lines(header: str) -> list[str]:
    """The lines of the header text of a command file, without its last empty line."""
    return header.rstrip("\n").split("\n")
