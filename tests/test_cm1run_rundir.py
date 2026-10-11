"""The run directory contract (``gm1.cm1-run/v1``), the segment deck and the event reader."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest

from cm1import_support import SUPERCELL, edited, namelist_tool
from cm1run_support import make_run

from cm1core.cm1run.events import EventTail, last_record, read_records
from cm1core.cm1run.launch import LaunchRefusal, StartRequest, parse_seconds, plan_segment
from cm1core.cm1run.rundir import (
    RunDirectory,
    RunDirectoryError,
    create_run_directory,
    derive_deck,
    sha256_file,
    write_json_atomic,
)
from cm1core.cm1run.status import run_status


def test_a_namelist_becomes_a_run_directory(tmp_path: Path) -> None:
    from cm1core.cm1import import import_namelist

    tool = namelist_tool()
    run = create_run_directory(SUPERCELL, tmp_path / "supercell", tool=tool,
                               gm1_only={"runtime.memory_budget_mib": 5000})
    record = run.load()
    assert record["schema"] == "gm1.cm1-run/v1" and record["name"] == "supercell"
    assert (run.path / "namelist.input.original").read_bytes() == SUPERCELL.read_bytes()
    assert record["namelist"]["original_sha256"] == sha256_file(SUPERCELL)
    assert record["import"]["result"] == "runnable"
    assert record["import"]["component"] == "morrison-mp10"
    assert record["cm1"]["rstfrq"] == -3600.0 and record["cm1"]["timax"] == 7200.0
    assert record["gm1"]["install"] in ("checkout", "wheel")
    expected = import_namelist(SUPERCELL, tool=tool, gm1_only={"runtime.memory_budget_mib": 5000}).toml()
    assert run.deck.read_text(encoding="utf-8") == expected
    assert (run.path / "control").is_dir() and record["segments"] == []
    assert run_status(run).state == "ready"
    # The GUI rule for a namelist that writes no restarts: one at the end.
    plan = plan_segment(run, record, StartRequest())
    assert plan.restart_interval_s == 7200.0 and plan.frames_expected == 9 and plan.restarts_expected == 1
    with pytest.raises(RunDirectoryError, match="not empty"):
        create_run_directory(SUPERCELL, run.path, tool=tool)


def test_a_refused_namelist_still_gets_a_directory_that_says_why(tmp_path: Path) -> None:
    tool = namelist_tool()
    nssl = edited(SUPERCELL, tmp_path, {"ptype": "26"})
    run = create_run_directory(nssl, tmp_path / "nssl", tool=tool)
    record = run.load()
    assert record["import"]["result"] == "refused"
    assert any("NSSL" in reason for reason in record["import"]["refused"])
    assert not run.deck.exists()
    answer = run_status(run)
    assert answer.state == "imported" and "NSSL" in answer.detail


def test_a_namelist_that_reads_a_sounding_needs_one(tmp_path: Path) -> None:
    tool = namelist_tool()
    reads_file = edited(SUPERCELL, tmp_path, {"isnd": "7"})
    with pytest.raises(RunDirectoryError, match="input_sounding"):
        create_run_directory(reads_file, tmp_path / "isnd7", tool=tool)
    assert not (tmp_path / "isnd7").exists()
    sounding = tmp_path / "input_sounding"
    sounding.write_text(" 1000.0 300.0 14.0\n 250.0 300.5 13.0 5.0 0.0\n", encoding="ascii")
    run = create_run_directory(reads_file, tmp_path / "isnd7", sounding=sounding, tool=tool)
    assert (run.path / "input_sounding").read_bytes() == sounding.read_bytes()
    assert run.load()["sounding"]["read"] is True


def test_the_segment_deck_changes_only_the_named_keys() -> None:
    text = (
        "# header\n[run]\ndt_s = 6.0\nduration_s = 7200.0\nrestart_interval_s = 3600.0\n\n"
        "[grid]\nnx = 120\n"
    )
    derived = derive_deck(text, {"run": {"duration_s": 9000.0, "irst": 1, "rstnum": 2}},
                          header=("# segment 002",))
    lines = derived.splitlines()
    assert lines[0] == "# segment 002" and lines[1] == "# header"
    parsed = tomllib.loads(derived)
    assert parsed["run"] == {"dt_s": 6.0, "duration_s": 9000.0, "restart_interval_s": 3600.0,
                             "irst": 1, "rstnum": 2}
    assert parsed["grid"] == {"nx": 120}
    # Keys the table lacked go at its end, before the blank line.
    assert derived.index("rstnum = 2") < derived.index("[grid]")
    added = derive_deck(text, {"initialization": {"irst": 1}})
    assert tomllib.loads(added)["initialization"] == {"irst": 1}


def test_frames_follow_the_latest_lineage(tmp_path: Path) -> None:
    run = RunDirectory(make_run(tmp_path))

    def segment(number: int, frames: dict[int, float], restarts: dict[int, tuple[int, float, int]],
                resumed_from: dict | None) -> None:
        path = run.segments_root / f"{number:03d}"
        for index, when in frames.items():
            frame = path / "frames" / f"frame-{index:06d}"
            frame.mkdir(parents=True)
            (frame / "history-publication.json").write_text(json.dumps({"step": int(when / 6), "model_time_s": when}))
        for number_, (step, when, next_index) in restarts.items():
            restart = path / "restarts" / f"restart-{number_:06d}"
            restart.mkdir(parents=True)
            (restart / "restart-publication.json").write_text(json.dumps({
                "restart_number": number_, "step": step, "model_time_s": when, "next_history_index": next_index}))
        record = run.load()
        record["segments"].append({"number": number, "resumed_from": resumed_from})
        write_json_atomic(run.run_json, record)

    segment(1, {0: 0.0, 1: 60.0, 2: 120.0, 3: 180.0, 4: 240.0}, {1: (20, 120.0, 3), 2: (40, 240.0, 5)}, None)
    # Segment 2 branched from restart 1 (120 s): segment 1's frames after 120 s are not the run's.
    segment(2, {3: 180.0, 4: 240.0, 5: 300.0}, {2: (40, 240.0, 5)}, {"segment": 1, "restart": 1, "model_time_s": 120.0})
    frames = run.frames()
    assert [(frame.index, frame.segment) for frame in frames] == [(0, 1), (1, 1), (2, 1), (3, 2), (4, 2), (5, 2)]
    # Restart 2 exists on both branches; the newest segment's is the run's.
    assert run.find_restart(2).segment == 2
    assert run.find_restart(2, segment=1).segment == 1
    assert run.find_restart().number == 2 and run.find_restart().segment == 2
    assert run.find_restart(9) is None


def test_the_event_reader_leaves_a_torn_line_for_later(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    tail = EventTail(path)
    assert tail.poll() == []
    path.write_text('{"kind": "admitted", "seq": 0}\n{"kind": "prog', encoding="utf-8")
    assert [record["kind"] for record in tail.poll()] == ["admitted"]
    assert read_records(path) == [{"kind": "admitted", "seq": 0}]
    with path.open("a", encoding="utf-8") as stream:
        stream.write('ress", "seq": 1}\n')
    assert [record["kind"] for record in tail.poll()] == ["progress"]
    assert tail.poll() == []
    assert last_record(path)["seq"] == 1
    # A new file under the same name is read from its start.
    path.unlink()
    path.write_text('{"kind": "end", "seq": 0}\n', encoding="utf-8")
    assert [record["kind"] for record in tail.poll()] == ["end"]


@pytest.mark.parametrize(("text", "seconds"), [
    ("9000", 9000.0), ("9000s", 9000.0), ("150min", 9000.0), ("2.5h", 9000.0), ("90m", 5400.0), (600, 600.0),
])
def test_times_are_read_with_their_units(text, seconds) -> None:
    assert parse_seconds(text) == seconds


def test_a_restart_cadence_that_misses_the_end_is_refused_with_choices(tmp_path: Path) -> None:
    run = RunDirectory(make_run(tmp_path, duration=360.0, restart=120.0))
    with pytest.raises(LaunchRefusal, match="rstfrq = 120 s") as caught:
        plan_segment(run, run.load(), StartRequest(until_s=300.0))
    assert "240, 360" in caught.value.remedy
    plan = plan_segment(run, run.load(), StartRequest(until_s=240.0))
    assert plan.until_s == 240.0 and plan.restart_interval_s == 120.0 and plan.frames_expected == 5


def _fake_segment(run: RunDirectory, number: int, frames: dict[int, float],
                  restarts: dict[int, tuple[int, float, int]], resumed_from: dict | None) -> None:
    path = run.segments_root / f"{number:03d}"
    path.mkdir(parents=True, exist_ok=True)
    for index, when in frames.items():
        frame = path / "frames" / f"frame-{index:06d}"
        frame.mkdir(parents=True)
        (frame / "history-publication.json").write_text(json.dumps({"step": int(when / 6), "model_time_s": when}))
    for number_, (step, when, next_index) in restarts.items():
        restart = path / "restarts" / f"restart-{number_:06d}"
        restart.mkdir(parents=True)
        (restart / "restart-publication.json").write_text(json.dumps({
            "restart_number": number_, "step": step, "model_time_s": when, "next_history_index": next_index}))
    record = run.load()
    record["segments"].append({"number": number, "resumed_from": resumed_from,
                               "until_s": 360.0, "kind": "fresh" if resumed_from is None else "resume"})
    write_json_atomic(run.run_json, record)


def test_continue_follows_the_run_as_its_frames_show_it_not_a_branch_it_left(tmp_path: Path) -> None:
    run = RunDirectory(make_run(tmp_path, duration=360.0, restart=120.0))
    # Segment 1 ran to the end: restarts 1-3 at 120, 240 and 360 s.
    _fake_segment(run, 1, {0: 0.0, 1: 60.0, 2: 120.0, 3: 180.0, 4: 240.0, 5: 300.0, 6: 360.0},
                  {1: (20, 120.0, 3), 2: (40, 240.0, 5), 3: (60, 360.0, 7)}, None)
    # Segment 2 went back to restart 1 (120 s) and was stopped at 180 s, before restart 2.
    _fake_segment(run, 2, {3: 180.0}, {}, {"segment": 1, "restart": 1, "model_time_s": 120.0})
    assert [(row.segment, row.number) for row in run.lineage_restarts()] == [(1, 1)]
    # "Continue" goes on from 120 s, the trajectory frames 0-3 show; not from
    # segment 1's 360 s restart, which belongs to the branch the run left.
    assert (run.find_restart().segment, run.find_restart().number) == (1, 1)
    plan = plan_segment(run, run.load(), StartRequest(command="resume"))
    assert plan.resumed_from.number == 1 and plan.start_model_time_s == 120.0 and plan.until_s == 360.0
    plan = plan_segment(run, run.load(), StartRequest(command="run", continue_run=True))
    assert plan.kind == "resume" and plan.resumed_from.number == 1
    # The other branch stays reachable by name.
    assert run.find_restart(3).segment == 1
    assert run.find_restart(3, segment=1).model_time_s == 360.0
    # A start over that has written nothing yet has no restart to continue:
    # --continue starts it again rather than resuming the old run.
    _fake_segment(run, 3, {}, {}, None)
    write_json_atomic(run.segments_root / "003" / "segment.json", {
        "schema": "gm1.cm1-run-segment/v1", "number": 3, "state": "ended", "reason": "no-final-record",
        "rc": 1, "host": "this-host-or-not", "supervisor": None, "child": None,
    })
    assert run.lineage_restarts() == [] and run.find_restart() is None
    plan = plan_segment(run, run.load(), StartRequest(command="run", continue_run=True))
    assert plan.kind == "fresh" and plan.resumed_from is None
    with pytest.raises(LaunchRefusal, match="latest segment") as caught:
        plan_segment(run, run.load(), StartRequest(command="resume"))
    assert "--from-segment" in caught.value.remedy
    answer = run_status(run)
    assert answer.state == "lost"
    assert [row["id"] for row in answer.actions] == ["start-over", "job-script"]


def test_a_segment_created_a_moment_ago_reads_starting_not_lost(tmp_path: Path) -> None:
    run = RunDirectory(make_run(tmp_path))
    (run.segments_root / "001").mkdir(parents=True)
    answer = run_status(run)
    assert answer.state == "running" and answer.label == "Starting"
    old = __import__("time").time() - 600
    __import__("os").utime(run.segments_root / "001", (old, old))
    assert run_status(run).state == "lost"
