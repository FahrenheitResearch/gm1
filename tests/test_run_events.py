"""The ``gm1.run-events/v1`` log: its schema, its writer and the door's end records."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import pytest

from cm1core.run_events import (
    END_REASONS,
    EVENT_KINDS,
    HEARTBEAT_LIMIT_S,
    RUN_EVENTS_SCHEMA,
    RunEventLog,
    load_schema,
    main,
    read_events,
    validate_file,
    validate_lines,
    validate_record,
)

ROOT = Path(__file__).resolve().parents[1]


def _common(kind: str, seq: int = 0) -> dict:
    return {
        "schema": RUN_EVENTS_SCHEMA, "seq": seq, "kind": kind,
        "time_utc": "2026-09-23T00:00:00.000+00:00", "elapsed_s": 0.5, "pid": 1234,
    }


GOOD = {
    "admitted": {
        "config": "deck.toml", "config_sha256": "a" * 64,
        "grid": {"nx": 120, "ny": 120, "nz": 40, "dx_m": 1000.0, "dy_m": 1000.0,
                 "dz_m": 500.0, "stretch_z": 0, "ztop_m": 20000.0},
        "microphysics": "wsm6-mp6", "steps": 1200, "start_step": 0, "dt_s": 6.0,
        "nsound": 4, "adaptive_timestep": False, "duration_s": 7200.0,
        "history_interval_s": 900.0, "frames_expected": 9,
        "restart_interval_s": 3600.0, "restarts_expected": 2,
        "history_directory": "/r/frames", "restart_directory": "/r/restarts",
        "resumed_from": None, "receipt": "/r/receipt.json",
        "control_directory": "/r/control", "history_envelope": "off",
        "history_initial": True, "host": "desk", "gpu_uuid_requested": None,
    },
    "startup": {"stages": {"runtime.construction": 9.5}, "nvrtc_compile_s": 7.0,
                "nvrtc_programs": 65, "seconds": 9.5,
                "gpu": {"name": "NVIDIA GeForce RTX 3080", "uuid": "GPU-1"}},
    "progress": {"phase": "stepping", "heartbeat": False, "step": 10, "total_steps": 1200,
                 "model_time_s": 60.0, "duration_s": 7200.0, "s_per_step_recent": 0.04,
                 "s_per_step_mean": 0.041, "model_s_per_wall_s": 140.0, "eta_s": 51.0,
                 "last_io": {"kind": "history", "label": "frame-000001", "seconds": 1.2,
                             "bytes": 53126444, "background": True,
                             "blocking_seconds": 0.9},
                 "gpu": {"memory_used_bytes": 5, "memory_total_bytes": 10}},
    "frame": {"index": 0, "step": 0, "model_time_s": 0.0, "directory": "/r/frames/frame-000000",
              "bytes": 1, "initial_condition": True, "restart_written": None},
    "restart": {"number": 1, "step": 600, "model_time_s": 3600.0,
                "directory": "/r/restarts/restart-000001", "bytes": 1,
                "next_history_index": 5},
    "warning": {"code": "X", "message": "y"},
    "refused": {"code": "CM1-X", "gate": "g", "reason": "r", "remedy": "m", "option": None},
    "end": {"reason": "completed", "rc": 0, "receipt": "/r/receipt.json", "step": 1200,
            "model_time_s": 7200.0, "frames_published": 9, "restarts_published": 2,
            "signal": None, "request": None, "message": None},
    "stats": {"step": 10, "model_time_s": 60.0, "values": {"wmax": 1.0}},
}

BAD = [
    ("admitted", {"steps": -1}),
    ("admitted", {"config_sha256": "not-a-hash"}),
    ("admitted", {"history_envelope": "maybe"}),
    ("progress", {"phase": "thinking"}),
    ("progress", {"eta_s": -5.0}),
    ("progress", {"step": 1.5}),
    ("frame", {"bytes": "many"}),
    ("end", {"reason": "finished"}),
    ("end", {"rc": -1}),
    ("end", {"request": "resume"}),
    ("restart", {"number": 0}),
]


def test_every_kind_has_a_good_example_and_the_schema_names_every_kind() -> None:
    schema = load_schema()
    assert schema["properties"]["kind"]["enum"] == list(EVENT_KINDS)
    assert schema["$defs"]["end"]["properties"]["reason"]["enum"] == list(END_REASONS)
    assert set(GOOD) == set(EVENT_KINDS)
    for kind, fields in GOOD.items():
        assert validate_record({**_common(kind), **fields}) == [], kind


@pytest.mark.parametrize("kind,change", BAD)
def test_a_bad_record_is_named(kind: str, change: dict) -> None:
    record = {**_common(kind), **GOOD[kind], **change}
    assert validate_record(record) != []


@pytest.mark.parametrize("kind", EVENT_KINDS)
def test_a_missing_required_field_is_named(kind: str) -> None:
    schema = load_schema()
    for name in schema["$defs"][kind]["required"]:
        record = {**_common(kind), **GOOD[kind]}
        del record[name]
        assert any(repr(name) in error for error in validate_record(record)), (kind, name)


def test_the_stdlib_validator_agrees_with_the_jsonschema_package() -> None:
    """The evaluator here is held to the reference implementation's verdict."""

    jsonschema = pytest.importorskip("jsonschema")
    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    records = [{**_common(kind), **fields} for kind, fields in GOOD.items()]
    records += [{**_common(kind), **GOOD[kind], **change} for kind, change in BAD]
    for kind in EVENT_KINDS:
        for name in schema["$defs"][kind]["required"]:
            record = {**_common(kind), **GOOD[kind]}
            del record[name]
            records.append(record)
    records.append({**_common("end"), **GOOD["end"], "schema": "gm1.run-events/v0"})
    records.append({key: value for key, value in _common("warning").items() if key != "pid"})
    for record in records:
        assert (not validate_record(record, schema)) == validator.is_valid(record), record


def test_a_log_is_a_new_file_of_flushed_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    log = RunEventLog(path)
    log.emit("warning", code="A", message="first")
    # Flushed per line: another reader sees it before the log is closed.
    assert [e["kind"] for e in read_events(path)] == ["warning"]
    log.emit("end", **GOOD["end"])
    # Nothing is written after the end record.
    assert log.emit("warning", code="B", message="late") is None
    log.close()
    events = read_events(path)
    assert [e["seq"] for e in events] == [0, 1]
    assert validate_file(path) == []
    with pytest.raises(FileExistsError):
        RunEventLog(path)
    with pytest.raises(ValueError):
        RunEventLog(tmp_path / "other.jsonl").emit("finished")


def test_a_non_finite_number_is_written_as_null(tmp_path) -> None:
    log = RunEventLog(tmp_path / "events.jsonl")
    fields = dict(GOOD["progress"], eta_s=float("inf"), model_s_per_wall_s=float("nan"))
    record = log.progress({k: v for k, v in fields.items() if k not in ("phase", "heartbeat")})
    log.close()
    assert record["eta_s"] is None and record["model_s_per_wall_s"] is None
    assert validate_file(tmp_path / "events.jsonl") == []


def test_a_failed_write_turns_the_log_off_and_never_raises(tmp_path, capsys) -> None:
    log = RunEventLog(tmp_path / "events.jsonl")
    log._handle.close()
    assert log.emit("warning", code="A", message="b") is None
    assert log.failed is not None
    assert "turned off" in capsys.readouterr().err
    assert log.emit("warning", code="A", message="b") is None


def test_the_heartbeat_writes_when_the_log_goes_quiet(tmp_path) -> None:
    now = [0.0]
    log = RunEventLog(tmp_path / "events.jsonl", clock=lambda: now[0])
    log.set_phase("starting")
    log._beat()
    assert read_events(tmp_path / "events.jsonl") == []  # not quiet yet
    now[0] = 9.0
    log._beat()
    beats = read_events(tmp_path / "events.jsonl")
    assert len(beats) == 1 and beats[0]["heartbeat"] is True
    assert beats[0]["phase"] == "starting" and beats[0]["step"] is None
    # It repeats the last known progress, marked as a heartbeat.
    log.set_phase("stepping")
    log.progress({key: value for key, value in GOOD["progress"].items()
                  if key not in ("phase", "heartbeat")})
    now[0] = 30.0
    log._beat()
    last = read_events(tmp_path / "events.jsonl")[-1]
    assert last["heartbeat"] is True and last["step"] == 10 and last["phase"] == "stepping"
    log.close()
    assert validate_file(tmp_path / "events.jsonl") == []


def test_the_heartbeat_thread_itself_writes_while_the_caller_is_blocked(tmp_path) -> None:
    """The thread, not only _beat(): a record appears while the caller waits."""

    import time

    now = [0.0]
    log = RunEventLog(tmp_path / "events.jsonl", clock=lambda: now[0])
    log.set_phase("waiting-for-gpu")
    log.start_heartbeat(tick_s=0.01)
    try:
        now[0] = 9.0  # quiet for longer than HEARTBEAT_AFTER_S
        deadline = time.monotonic() + 5.0
        while not read_events(tmp_path / "events.jsonl") and time.monotonic() < deadline:
            time.sleep(0.01)  # the "stepping thread", blocked elsewhere
        beats = read_events(tmp_path / "events.jsonl")
        assert beats and beats[0]["heartbeat"] is True
        assert beats[0]["phase"] == "waiting-for-gpu"
        # One record per quiet spell: the clock has not moved since it wrote.
        time.sleep(0.1)
        assert len(read_events(tmp_path / "events.jsonl")) == 1
    finally:
        log.close()
    assert validate_file(tmp_path / "events.jsonl") == []


def test_the_heartbeat_thread_starts_stops_and_touches_nothing_else(tmp_path) -> None:
    log = RunEventLog(tmp_path / "events.jsonl").start_heartbeat(tick_s=0.01)
    assert any(t.name == "gm1-run-events-heartbeat" for t in threading.enumerate())
    log.close()
    assert not any(t.name == "gm1-run-events-heartbeat" for t in threading.enumerate())
    assert HEARTBEAT_LIMIT_S <= 10.0


def test_the_stream_rules_are_checked_beyond_each_record() -> None:
    lines = [json.dumps({**_common("end", 0), **GOOD["end"]}),
             json.dumps({**_common("warning", 1), **GOOD["warning"]})]
    assert "the end record is not the last record" in validate_lines(lines)
    lines = [json.dumps({**_common("warning", 1), **GOOD["warning"]})]
    assert any("seq" in error for error in validate_lines(lines))
    assert any("not JSON" in error for error in validate_lines(["{"]))


def test_the_command_line_validator(tmp_path, capsys) -> None:
    path = tmp_path / "events.jsonl"
    log = RunEventLog(path)
    log.emit("end", **GOOD["end"])
    log.close()
    assert main(["validate", os.fspath(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True and report["kinds"] == {"end": 1}
    path.write_text('{"kind": "end"}\n')
    assert main(["validate", os.fspath(path)]) == 2


# ------------------------------------------------------------ the door's records


def test_the_door_maps_each_ending_to_its_end_record(tmp_path) -> None:
    from cm1core.background_publication import TerminationRequested
    from cm1core.cartesian_door import _RunReporting
    from cm1core.errors import CM1Refusal, refuse

    def ended(error, *, admitted=True, reason=None, rc=None, message=None):
        path = tmp_path / f"events-{len(list(tmp_path.iterdir()))}.jsonl"
        log = RunEventLog(path)
        reporting = _RunReporting(log)
        reporting.admitted = admitted
        reporting.reason = reason
        reporting.message = message
        reporting.end(error, rc)
        log.close()
        assert validate_file(path) == []
        return read_events(path)

    def refusal(code):
        try:
            refuse(code=code, gate="g", reason="why", remedy="fix", option="o", value=1)
        except CM1Refusal as error:
            return error

    end = ended(None, rc=0)[-1]
    assert (end["reason"], end["rc"], end["signal"]) == ("completed", 0, None)
    end = ended(None, reason="paused", rc=0)[-1]
    assert (end["reason"], end["rc"]) == ("paused", 0)
    # A [nest.tornado_criterion] stop rule: stopped short of the duration, by
    # no control request, and the message names the rule.
    end = ended(None, reason="stopped", rc=0,
                message="stopped by nest.tornado_criterion.first_tornado at 600.0 model seconds")[-1]
    assert (end["reason"], end["rc"], end["request"], end["signal"]) == ("stopped", 0, None, None)
    assert "nest.tornado_criterion" in end["message"]
    end = ended(TerminationRequested())[-1]
    assert (end["reason"], end["rc"], end["signal"]) == ("stopped", 143, "SIGTERM")
    end = ended(KeyboardInterrupt())[-1]
    assert (end["reason"], end["rc"], end["signal"]) == ("stopped", 130, "SIGINT")
    events = ended(refusal("CM1-CARTESIAN-GRID"), admitted=False)
    assert [e["kind"] for e in events] == ["refused", "end"]
    assert events[0]["code"] == "CM1-CARTESIAN-GRID" and events[0]["remedy"] == "fix"
    assert (events[-1]["reason"], events[-1]["rc"]) == ("refused", 2)
    events = ended(refusal("CM1-CARTESIAN-RUN-FAILED"))
    assert (events[-1]["reason"], events[-1]["rc"]) == ("error", 2)
    events = ended(refusal("CM1-CARTESIAN-COURANT-STOP"))
    assert [e["kind"] for e in events] == ["end"]
    assert (events[-1]["reason"], events[-1]["rc"]) == ("courant-stop", 2)
    end = ended(RuntimeError("device lost"))[-1]
    assert (end["reason"], end["rc"]) == ("error", 1) and "device lost" in end["message"]


def _stock_config():
    from cm1core.config import load_config

    return load_config(ROOT / "examples" / "cartesian-uniform-grid.toml")


@pytest.mark.parametrize("flag", ["events", "control_dir", "history_initial"])
def test_a_run_flag_without_integrate_is_refused_before_anything_is_created(
    tmp_path, flag
) -> None:
    from cm1core.cartesian_door import run_cartesian
    from cm1core.errors import CM1Refusal

    value = True if flag == "history_initial" else tmp_path / flag
    with pytest.raises(CM1Refusal) as refused:
        run_cartesian(
            _stock_config(), config_label="deck", output=tmp_path / "receipt.json",
            **{flag: value},
        )
    assert refused.value.detail.code == "CM1-CARTESIAN-RUN-FLAG"
    assert "--integrate" in refused.value.detail.remedy
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(CM1Refusal) as refused:
        run_cartesian(
            _stock_config(), config_label="deck", output=tmp_path / "receipt.json",
            plan=True, integrate=True, **{flag: value},
        )
    assert "--plan" in refused.value.detail.remedy


def test_a_bad_history_envelope_value_and_an_existing_log_are_refused(tmp_path) -> None:
    from cm1core.cartesian_door import run_cartesian
    from cm1core.errors import CM1Refusal

    with pytest.raises(CM1Refusal) as refused:
        run_cartesian(_stock_config(), config_label="deck", output=tmp_path / "r.json",
                      history_envelope="none")
    assert refused.value.detail.code == "CM1-CARTESIAN-HISTORY-ENVELOPE"
    existing = tmp_path / "events.jsonl"
    existing.write_text("")
    with pytest.raises(CM1Refusal) as refused:
        run_cartesian(_stock_config(), config_label="deck", output=tmp_path / "r.json",
                      integrate=True, events=existing)
    assert refused.value.detail.code == "CM1-CARTESIAN-EVENTS"
    assert existing.read_text() == ""


def test_an_admission_refusal_reaches_the_log_as_refused_then_end(tmp_path) -> None:
    """A run refused on the host still leaves a log a watcher can read."""

    from cm1core.cartesian_door import run_cartesian
    from cm1core.errors import CM1Refusal

    events = tmp_path / "events.jsonl"
    with pytest.raises(CM1Refusal) as refused:
        # --history-envelope off with no --history-dir is refused by the door
        # after the log is open.
        run_cartesian(_stock_config(), config_label="deck", output=tmp_path / "r.json",
                      integrate=True, events=events, history_envelope="off")
    records = read_events(events)
    assert validate_file(events) == []
    assert [record["kind"] for record in records] == ["refused", "end"]
    assert records[0]["code"] == refused.value.detail.code
    assert records[-1]["reason"] == "refused" and records[-1]["rc"] == 2
    assert records[-1]["receipt"] is None and records[-1]["frames_published"] == 0


def test_the_cli_carries_the_four_flags_with_the_defaults_unchanged() -> None:
    from cm1core.cli import build_parser

    parser = build_parser()
    plain = parser.parse_args(["cartesian", "deck.toml", "--output", "r.json"])
    assert plain.events is None and plain.control_dir is None
    assert plain.history_envelope == "on" and plain.history_initial is False
    flagged = parser.parse_args([
        "cartesian", "deck.toml", "--output", "r.json", "--integrate",
        "--events", "e.jsonl", "--control-dir", "control",
        "--history-envelope", "off", "--history-initial",
    ])
    assert flagged.events == Path("e.jsonl") and flagged.control_dir == Path("control")
    assert flagged.history_envelope == "off" and flagged.history_initial is True
