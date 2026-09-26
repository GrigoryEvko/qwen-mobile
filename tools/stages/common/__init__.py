"""The parts that each phone stage tool of tools/stages shares.

A stage tool has two jobs. `stage.py commands` writes the command file that the laptop runner sends to
the phone, and `stage.py table` reads the pulled outputs and prints the tables. The modules here hold
the parts that are the same for each stage:

    device    the phone and the box layout
    commands  the lines of a phone command file
    gate      the conditions of one run, from its gate file
    logs      the text of an output file, and the marks of the runner log
    parse     the lines of the phone tools that more than one stage reads
    tables    the numbers of a table cell
    cli       the command line of a stage tool

A stage module holds only what makes that stage different: its runs, its tool arguments, its header and
its own tables. A stage tool finds this package with two lines, because the tool runs as a script:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import commands, device, gate  # noqa: E402

The name of one thing is the same in each stage: `caps` is the pair of clock caps, `nsp` is the NPU zone
temperature, `flags` names each condition that makes a run not comparable, and `removed` names each
condition that keeps a run out of the tables.
"""
