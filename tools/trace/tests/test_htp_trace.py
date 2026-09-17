"""Tests of the profile log converter: the parser, the timeline, the phases, the tables and the PMU names."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import htp_trace  # noqa: E402
from htp_trace import (  # noqa: E402
    DEFAULT_PMU_EVENTS,
    TID_BATCHES,
    TID_DMA_BASE,
    TID_OPS,
    TID_THREAD_BASE,
    ProfileLog,
    chrome_events,
    format_diff,
    format_summary,
    pair_phases,
    parse_lines,
    parse_slice,
    pmu_counter_labels,
    pmu_event_names,
    summarize,
    unwrap_cycles,
)

MHZ = 2000.0
WRAP = 1 << 32


def batch_line(start: int, usec: int, n_ops: int, stamp: str | None = "0.04.309.617", evt_cnt: str = "----") -> str:
    """Make one ``OPBATCH`` line with the 64-bit start and the cycles that agree with the clock."""
    prefix = f"{stamp} D " if stamp else ""
    cycles = int(usec * MHZ)
    return (
        f"{prefix}ggml-hex: HTP0 profile-op OPBATCH|----|n-ops {n_ops}|{evt_cnt}|----|----"
        f"|usec {usec} cycles {cycles} start {start} mhz {MHZ:.1f}"
    )


def op_line(
    name: str, start: int, cycles: int, pmu: str | None = None, kparams: str = "----", stamp: str | None = "0.04.309.700"
) -> str:
    """Make one ``profile-op`` line with the low 32 bits of the counter as the start."""
    prefix = f"{stamp} D " if stamp else ""
    usec = int(cycles // MHZ)
    pmu_text = f" pmu [{pmu}]" if pmu else ""
    return (
        f"{prefix}ggml-hex: HTP0 profile-op {name}|blk.0.w x act-0 -> out-0|2048:2 x 2048:2 -> 2048:2|f32 x f32 -> f32"
        f"|4:8192 x 4:8192 -> 4:8192|{kparams}|usec {usec} cycles {cycles} start {start & (WRAP - 1)} mhz {MHZ:.1f}{pmu_text}"
    )


def trace_line(name: str, thread: int, info: int, state: str, cycles: int) -> str:
    """Make one ``trace-evt`` line of level 3."""
    return f"0.04.309.800 D ggml-hex: HTP0 trace-evt {name}: thread {thread} info {info} {state} {cycles & (WRAP - 1)}"


def level1_lines() -> list[str]:
    """Give a level 1 log: two batches with timestamps, one op without a batch, and a token before a line."""
    first = 10 * WRAP + 1000
    second = first + 400_000
    return [
        "0.00.624.041 I ggml-hex: HTP0 new session : session-id 0 domain-id 3 uri file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke",
        op_line("MUL", 5, 2000),
        batch_line(first, usec=100, n_ops=3, stamp="0.04.309.617"),
        op_line("RMS_NORM", first + 2000, 4000),
        op_line("MUL_MAT", first + 6000, 100_000, kparams="hvx-tiled vtcm 4128768"),
        op_line("MUL_MAT", first + 106_000, 80_000, kparams="hmx-tiled vtcm 8314880"),
        "The capital of France is" + batch_line(second, usec=50, n_ops=1, stamp="0.04.310.000"),
        op_line("MUL_MAT", second + 4000, 60_000, kparams="hvx-tiled vtcm 4128768"),
        "0.04.311.000 D not a profile line",
    ]


def level2_lines() -> list[str]:
    """Give a level 2 log with the mode line and eight custom PMU events."""
    first = 3 * WRAP + 500
    return [
        "0.00.001.000 I ggml-hex: Profiling mode 2 : pmu-evt [ 0x3,0x111,0x100,0x105,0x240,0x256,0x7d,0x999 ]",
        batch_line(first, usec=10, n_ops=2),
        op_line("ADD", first + 100, 4000, pmu="1,2,3,4,5,6,7,8"),
        op_line("MUL", first + 5000, 6000, pmu="10,20,30,40,50,60,70,80"),
    ]


def level3_lines() -> list[str]:
    """Give a level 3 log: one batch, one op, and the trace events of two threads plus the HMX queue."""
    first = 7 * WRAP + 100
    return [
        batch_line(first, usec=20, n_ops=1, evt_cnt="evt-cnt 6,2,0,0,0,0,0,0,0,0,2"),
        op_line("MUL_MAT", first + 1000, 30_000),
        trace_line("L2FLUSH", 0, 0, "start", first + 10),
        trace_line("L2FLUSH", 0, 0, "stop", first + 500),
        trace_line("DMA", 0, 4, "start", first + 1000),
        trace_line("DMA", 0, 5, "start", first + 1200),
        trace_line("DMA", 0, 4, "stop", first + 3000),
        trace_line("DMA", 0, 5, "stop", first + 3400),
        trace_line("HVX_COMP", 1, 0, "start", first + 2000),
        trace_line("HVX_COMP", 1, 64, "stop", first + 9000),
        trace_line("HMX_COMP", 10, 3, "start", first + 2500),
        trace_line("HMX_COMP", 10, 9, "stop", first + 2500),
        trace_line("HMX_COMP", 10, 1, "start", first + 12_000),
    ]


@pytest.fixture
def level1() -> ProfileLog:
    """Parse the level 1 log."""
    return parse_lines(level1_lines(), "level1.log")


def test_unwrap_cycles_adds_the_wrap_when_the_low_bits_are_smaller() -> None:
    """The counter value that follows a previous value keeps the low 32 bits of the new sample."""
    previous = 5 * WRAP + 0xFFFFFF00
    assert unwrap_cycles(previous, 0xFFFFFF80) == previous + 0x80
    assert unwrap_cycles(previous, 0x100) == 6 * WRAP + 0x100
    assert unwrap_cycles(previous, 0xFFFFFF00) == previous


def test_level1_counts_and_orphans(level1: ProfileLog) -> None:
    """The parser finds the session, the batches, the ops, the arch, and counts the op without a batch."""
    assert level1.level == 1
    assert level1.arch == 79
    assert level1.mode is None
    assert level1.orphan_ops == 1
    assert level1.bad_lines == 0
    session = level1.sessions["HTP0"]
    assert [len(batch.ops) for batch in session.batches] == [3, 1]
    assert [batch.n_ops for batch in session.batches] == [3, 1]
    assert session.batches[1].seq == 1
    assert session.batches[0].ops[1].path == "hvx-tiled"
    assert session.batches[0].ops[0].path == ""


def test_op_offset_comes_from_the_low_32_bits_of_the_batch_counter(level1: ProfileLog) -> None:
    """The op start is the low 32 bits of the batch counter, thus the offset is the cycle distance divided by the clock."""
    batch = level1.sessions["HTP0"].batches[0]
    assert batch.offset_us(batch.ops[0].abs_cycles) == pytest.approx(2000 / MHZ)
    assert batch.offset_us(batch.ops[2].abs_cycles) == pytest.approx(106_000 / MHZ)
    assert batch.ops[2].duration_us(batch.clock_mhz) == pytest.approx(40.0)


def test_op_offset_survives_a_wrap_of_the_low_32_bits() -> None:
    """An op with low 32 bits smaller than those of the batch start is after the batch start."""
    start = 4 * WRAP + 0xFFFFFF00
    log = parse_lines([batch_line(start, usec=10, n_ops=2), op_line("ADD", start + 0x80, 100), op_line("MUL", start + 0x300, 100)])
    batch = log.sessions["HTP0"].batches[0]
    assert batch.ops[0].abs_cycles == start + 0x80
    assert batch.ops[1].abs_cycles == start + 0x300
    assert batch.offset_us(batch.ops[1].abs_cycles) == pytest.approx(0x300 / MHZ)


def test_host_clock_puts_a_batch_at_its_timestamp_minus_its_time(level1: ProfileLog) -> None:
    """With timestamps, the batch start is the log time minus the batch time, and the first batch is at 0."""
    assert level1.clock == "host"
    first, second = level1.sessions["HTP0"].batches
    assert first.start_us == pytest.approx(0.0)
    assert first.host_us == pytest.approx(4_309_617)
    assert second.start_us == pytest.approx((4_310_000 - 50) - (4_309_617 - 100))


def test_cycle_clock_is_used_without_timestamps() -> None:
    """Without timestamps, the distance between batches comes from the 64-bit counter and the batch clock."""
    first = 10 * WRAP + 1000
    log = parse_lines(
        [
            batch_line(first, usec=100, n_ops=1, stamp=None),
            op_line("ADD", first + 10, 100, stamp=None),
            batch_line(first + 500_000, usec=50, n_ops=1, stamp=None),
            op_line("ADD", first + 500_010, 100, stamp=None),
        ]
    )
    assert log.clock == "cycles"
    batches = log.sessions["HTP0"].batches
    assert batches[0].start_us == 0.0
    assert batches[1].start_us == pytest.approx(500_000 / MHZ)
    assert batches[0].host_us is None


def test_gap_is_the_batch_time_minus_the_op_time(level1: ProfileLog) -> None:
    """The gap of a batch is its usec minus the cycle time of its ops."""
    batch = level1.sessions["HTP0"].batches[0]
    assert batch.op_time_us == pytest.approx((4000 + 100_000 + 80_000) / MHZ)
    assert batch.gap_us == pytest.approx(100 - 92.0)
    assert batch.op_usec_sum == 2 + 50 + 40


def test_level2_pmu_events_and_labels() -> None:
    """The mode line gives the event ids, and the labels are the names, or the hex id without a name."""
    log = parse_lines(level2_lines())
    assert log.level == 2
    assert log.mode == 2
    assert log.pmu_events == (0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x999)
    labels = pmu_counter_labels(log.pmu_events, 79)
    assert labels[0] == "COMMITTED_PKT_ANY"
    assert labels[1] == "HVX_PKT"
    assert labels[3] == "HVX_VTCM_OUTSTANDING"
    assert labels[-1] == "0x999"
    assert log.sessions["HTP0"].batches[0].ops[1].pmu == (10, 20, 30, 40, 50, 60, 70, 80)


def test_default_pmu_events_have_names_on_every_arch() -> None:
    """The eight preset events get names on v75, v79 and v81, and 0x105 has a different name on v81."""
    for arch in (75, 79, 81):
        labels = pmu_counter_labels(DEFAULT_PMU_EVENTS, arch)
        assert all(not label.startswith("0x") for label in labels), (arch, labels)
    assert pmu_event_names(79)[0x105] == "HVX_VTCM_OUTSTANDING"
    assert pmu_event_names(81)[0x105] == "HVX_LD_VTCM_OUTSTANDING"
    assert pmu_event_names(None)[0x3] == "COMMITTED_PKT_ANY"
    assert 0x24 in pmu_event_names(75) and 0x24 not in pmu_event_names(79)


def test_level3_phases_pair_by_info_then_by_name() -> None:
    """A stop connects to the open start of the same info, else to the newest start of the same name."""
    log = parse_lines(level3_lines())
    assert log.level == 3
    batch = log.sessions["HTP0"].batches[0]
    assert batch.evt_cnt == (6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2)
    phases, unpaired_starts, unpaired_stops = pair_phases(batch)
    by_key = {(phase.name, phase.thread, phase.info): phase for phase in phases}
    dma4 = by_key[("DMA", 0, 4)]
    dma5 = by_key[("DMA", 0, 5)]
    assert dma4.end_cycles - dma4.start_cycles == 2000
    assert dma5.end_cycles - dma5.start_cycles == 2200
    hvx = by_key[("HVX_COMP", 1, 0)]
    assert hvx.end_cycles - hvx.start_cycles == 7000
    assert hvx.note == ""
    assert unpaired_starts == 1
    assert unpaired_stops == 0
    notes = sorted(phase.note for phase in phases if phase.note)
    assert notes == ["start without stop"]


def test_stop_without_start_is_counted() -> None:
    """A stop with no open start of its name becomes a zero-length phase with a note."""
    start = 2 * WRAP
    log = parse_lines([batch_line(start, usec=10, n_ops=0), trace_line("HVX_COMP", 2, 0, "stop", start + 100)])
    phases, unpaired_starts, unpaired_stops = pair_phases(log.sessions["HTP0"].batches[0])
    assert (unpaired_starts, unpaired_stops) == (0, 1)
    assert phases[0].note == "stop without start"
    assert phases[0].start_cycles == phases[0].end_cycles == start + 100


def test_trace_events_before_a_batch_are_orphans() -> None:
    """A trace line before the first batch of its session is counted and not kept."""
    log = parse_lines([trace_line("DMA", 0, 0, "start", 5), batch_line(WRAP, usec=1, n_ops=0)])
    assert log.orphan_events == 1
    assert log.level == 1


def test_chrome_events_have_the_tracks(level1: ProfileLog) -> None:
    """The trace has the process, the batches track, the ops track, and one complete event for each op and batch."""
    events = chrome_events(level1)
    names = {(event["name"], event.get("tid")) for event in events if event["ph"] == "M"}
    assert ("process_name", None) in names
    assert ("thread_name", TID_BATCHES) in names
    assert ("thread_name", TID_OPS) in names
    process = next(event for event in events if event["ph"] == "M" and event["name"] == "process_name")
    assert process["args"]["name"] == "HTP0"
    batches = [event for event in events if event["ph"] == "X" and event["tid"] == TID_BATCHES]
    ops = [event for event in events if event["ph"] == "X" and event["tid"] == TID_OPS]
    assert len(batches) == 2 and len(ops) == 4
    assert batches[0]["dur"] == 100.0
    assert batches[0]["args"]["gap_us"] == pytest.approx(8.0)
    for op in ops:
        batch = batches[op["args"]["batch"]]
        assert batch["ts"] <= op["ts"]
        assert op["ts"] + op["dur"] <= batch["ts"] + batch["dur"] + 1e-6
    assert ops[1]["name"] == "MUL_MAT"
    assert ops[1]["args"]["kparams"] == "hvx-tiled vtcm 4128768"
    assert ops[1]["args"]["types"] == "f32 x f32 -> f32"


def test_chrome_counters_carry_the_pmu_names() -> None:
    """Level 2 gives one counter event for each op with the eight named values."""
    events = chrome_events(parse_lines(level2_lines()), arch=79)
    counters = [event for event in events if event["ph"] == "C"]
    assert len(counters) == 2
    assert counters[0]["name"] == "pmu"
    assert counters[0]["args"]["COMMITTED_PKT_ANY"] == 1
    assert counters[1]["args"]["0x999"] == 80
    op = next(event for event in events if event["ph"] == "X" and event["tid"] == TID_OPS)
    assert op["args"]["pmu"]["HVX_PKT"] == 2


def test_chrome_phases_go_to_the_thread_tracks_and_dma_tracks() -> None:
    """Level 3 phases are on the track of their thread, and DMA phases are on the DMA track of the thread."""
    events = chrome_events(parse_lines(level3_lines()))
    phases = [event for event in events if event["ph"] == "X" and event["cat"] in ("phase", "dma")]
    tids = {event["tid"] for event in phases}
    assert tids == {TID_THREAD_BASE + 0, TID_THREAD_BASE + 1, TID_THREAD_BASE + 10, TID_DMA_BASE + 0}
    thread_names = {event["tid"]: event["args"]["name"] for event in events if event["ph"] == "M" and event["name"] == "thread_name"}
    assert thread_names[TID_THREAD_BASE + 0] == "dsp t0 (main)"
    assert thread_names[TID_THREAD_BASE + 10] == "hmx queue"
    assert thread_names[TID_DMA_BASE + 0] == "dma t0"
    dma = [event for event in phases if event["cat"] == "dma"]
    assert [event["dur"] for event in dma] == [pytest.approx(1.0), pytest.approx(1.1)]
    unpaired = [event for event in phases if "note" in event["args"]]
    assert len(unpaired) == 1 and unpaired[0]["dur"] == 0.0


def test_summary_table_and_gap(level1: ProfileLog) -> None:
    """The summary gives the count, the time, the time for each op and the share of each class, plus the gap row."""
    summary = summarize(level1)[0]
    assert summary.n_batches == 2 and summary.n_ops == 4
    assert summary.batch_us == 150.0
    assert summary.op_us == pytest.approx((4000 + 100_000 + 80_000 + 60_000) / MHZ)
    assert summary.gap_us == pytest.approx(150 - 122)
    mul_mat = summary.classes["MUL_MAT"]
    assert mul_mat.count == 3
    assert mul_mat.total_us == pytest.approx(120.0)
    assert mul_mat.per_op_us == pytest.approx(40.0)
    assert summary.share(mul_mat.total_us) == pytest.approx(80.0)
    assert summary.idle_us == pytest.approx((4_310_000 - 50) - (4_309_617 - 100) - 100)
    text = format_summary(summary)
    assert "MUL_MAT" in text and "(gap)" in text
    assert "time between batches" in text
    lines = text.splitlines()
    assert lines[-1].startswith("(gap)")
    assert lines[-1].split()[1] == "2"


def test_summary_keys_and_slice(level1: ProfileLog) -> None:
    """The batch slice keeps a part of the batches, and the op-path key divides a class by the kernel path."""
    summary = summarize(level1, parse_slice("1:"))[0]
    assert summary.n_batches == 1 and summary.n_ops == 1
    assert summary.batch_us == 50.0
    assert summary.idle_us is None
    by_path = summarize(level1, key="op-path")[0]
    assert set(by_path.classes) == {"RMS_NORM", "MUL_MAT hvx-tiled", "MUL_MAT hmx-tiled"}
    assert by_path.classes["MUL_MAT hvx-tiled"].count == 2


def test_diff_lists_the_largest_change_first(level1: ProfileLog) -> None:
    """The diff puts the class with the largest change first, and a class that is only in B is new."""
    before = summarize(level1)[0]
    after = summarize(parse_lines(level2_lines()))[0]
    text = format_diff(before, after)
    lines = text.splitlines()
    assert lines[3].startswith("class")
    assert lines[4].startswith("MUL_MAT")
    assert "new" in text
    assert "-100.0%" in text


def test_parse_slice_forms() -> None:
    """The slice text has the Python forms, and one index selects one batch."""
    assert parse_slice(None) is None
    assert parse_slice("1:") == slice(1, None)
    assert parse_slice(":3") == slice(None, 3)
    assert parse_slice("-2:") == slice(-2, None)
    assert parse_slice("2") == slice(2, 3)
    assert parse_slice("-1") == slice(-1, None)
    assert parse_slice("0:4:2") == slice(0, 4, 2)
    with pytest.raises(ValueError):
        parse_slice("1:2:3:4")


def test_main_convert_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command line writes a JSON file that holds the events, and the summary prints the table."""
    log_path = tmp_path / "prof.log"
    log_path.write_text("\n".join(level1_lines()) + "\n")
    assert htp_trace.main(["convert", str(log_path)]) == 0
    document = json.loads((tmp_path / "prof.json").read_text())
    assert document["metadata"]["clock"] == "host"
    assert document["metadata"]["profile_level"] == 1
    assert len([event for event in document["traceEvents"] if event["ph"] == "X"]) == 6
    assert htp_trace.main(["summary", str(log_path), "--show-batches", "--top", "1"]) == 0
    out = capsys.readouterr().out
    assert "level 1, clock host, arch v79" in out
    assert "MUL_MAT" in out and "RMS_NORM" not in out.split("(gap)")[0].split("class")[1]
    assert "HTP0 #1" in out
    assert htp_trace.main(["diff", str(log_path), str(log_path), "--batches", "1:"]) == 0
    assert "delta: batch time +0.0 us" in capsys.readouterr().out
