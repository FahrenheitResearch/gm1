"""Run control (stop, pause) and the t=0 frame, on the shipped integrate().

The integration is ``CudaCartesianRuntime.integrate`` itself, bound to the
host double of ``test_cartesian_restart``: the schedule, the staging, the
atomic renames and the restart cadence are the shipped code.  dt is 0.5 s, so
``duration 4.0 / history 1.0 / restart 2.0`` publishes frames at steps 2, 4,
6, 8 and restarts at steps 4 and 8.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cm1core.cuda_cartesian_runtime import (
    FRAME_STAGING_DIRECTORY,
    INTEGRATION_CONTROL_PAUSE_STATUS,
    INTEGRATION_CONTROL_STOP_STATUS,
    INTEGRATION_STATUS,
)
from cm1core.run_control import PAUSE, STOP, RunControl, require_control_directory
from cm1core.run_events import RunEventLog, read_events, validate_file

from test_cartesian_restart import (  # noqa: E402 - the shared host double
    CpuCartesianRuntime,
    _config,
    forecast_publisher,  # noqa: F401 - the session fixture
)

FRAME_FILES_WITHOUT_ENVELOPE = [
    "history-carriers.f32", "history-field-table.tsv", "history-frame-location.json",
    "history-publication.json", "history-scheduler-scalars.tsv",
]


def _staging_left(root: Path) -> list[str]:
    return sorted(
        os.fspath(path) for path in root.rglob("*") if FRAME_STAGING_DIRECTORY in path.parts
    )


def _request_after(runtime: CpuCartesianRuntime, control_dir: Path, name: str, step: int) -> None:
    """Create ``control_dir/name`` right after large step ``step`` completes."""

    stepping = runtime.step

    def step_and_request() -> None:
        stepping()
        if runtime.clock.completed_steps == step:
            (control_dir / name).touch()

    runtime.step = step_and_request


def _integrate(runtime, root: Path, *, publisher=None, restart_publisher=None,
               restarts: bool = False, **keywords):
    return runtime.integrate(
        duration_s=4.0, history_interval_s=1.0,
        history_root=root / "frames", history_publisher=publisher,
        restart_root=(root / "restarts") if restarts else None,
        restart_interval_s=2.0 if restarts else None,
        restart_publisher=restart_publisher,
        **keywords,
    )


# ------------------------------------------------------------------ stop


def test_a_stop_file_ends_the_run_at_the_next_step_boundary(tmp_path) -> None:
    control_dir = require_control_directory(tmp_path / "control")
    # A directory made a minute ago, as a GUI's would be by the time it runs.
    old = os.stat(control_dir).st_mtime - 60.0
    os.utime(control_dir, (old, old))
    runtime = CpuCartesianRuntime(_config(), seed=3)
    # Step 5 is between the frames at steps 4 and 6.
    _request_after(runtime, control_dir, STOP, 5)
    # A frozen clock: no forced listing (LIST_AT_LEAST_EVERY_S) however slow
    # the machine running this test is.
    control = RunControl(control_dir, clock=lambda: 0.0)
    receipt = _integrate(runtime, tmp_path, control=control)

    assert receipt["status"] == INTEGRATION_CONTROL_STOP_STATUS
    assert receipt["completed_steps"] == 5 and receipt["executed_steps"] == 5
    stop = receipt["control_stop"]
    assert stop["request"] == STOP
    assert stop["seen_at_step"] == stop["ended_at_step"] == 5
    assert stop["frames_published"] == 2 and stop["frames_requested"] == 4
    # Nothing was due at step 5, so nothing was taken; both frames before it
    # are complete, and no staging is left behind.
    frames = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert frames == ["frame-000001", "frame-000002"]
    for name in frames:
        assert sorted(p.name for p in (tmp_path / "frames" / name).iterdir()) == (
            FRAME_FILES_WITHOUT_ENVELOPE
        )
    assert not _staging_left(tmp_path)
    assert receipt["claim_limit"].startswith("ended at a run-control request")
    # One stat() per step: every completed step polled once, plus the poll
    # before the first step; the directory was listed only when it changed
    # (once at the start, once when the stop file appeared).
    assert receipt["run_control"]["polls"] == 5 + 1
    assert receipt["run_control"]["listings"] == 2


def test_a_stop_seen_on_an_event_step_publishes_that_event_first(tmp_path) -> None:
    control_dir = require_control_directory(tmp_path / "control")
    runtime = CpuCartesianRuntime(_config(), seed=3)
    _request_after(runtime, control_dir, STOP, 4)
    receipt = _integrate(runtime, tmp_path, control=RunControl(control_dir))
    assert receipt["status"] == INTEGRATION_CONTROL_STOP_STATUS
    assert receipt["completed_steps"] == 4
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == [
        "frame-000001", "frame-000002",
    ]
    assert receipt["frames"][-1]["step"] == 4


def test_a_stop_present_before_the_first_step_takes_no_step(tmp_path) -> None:
    control_dir = require_control_directory(tmp_path / "control")
    runtime = CpuCartesianRuntime(_config(), seed=3)
    control = RunControl(control_dir)
    stepping = runtime.step
    steps = []

    def counted() -> None:
        steps.append(1)
        stepping()

    runtime.step = counted
    # Created after admission (a stale file is refused), before the first step.
    original_begin = control.begin

    def begin(*args, **kwargs):
        (control_dir / STOP).touch()
        original_begin(*args, **kwargs)

    control.begin = begin
    receipt = _integrate(runtime, tmp_path, control=control)
    assert receipt["status"] == INTEGRATION_CONTROL_STOP_STATUS
    assert steps == [] and receipt["completed_steps"] == 0
    assert receipt["control_stop"]["frames_published"] == 0


def test_a_completed_run_with_a_control_directory_keeps_its_status(tmp_path) -> None:
    control_dir = require_control_directory(tmp_path / "control")
    runtime = CpuCartesianRuntime(_config(), seed=3)
    receipt = _integrate(runtime, tmp_path, control=RunControl(control_dir))
    assert receipt["status"] == INTEGRATION_STATUS
    assert receipt["control_stop"] is None
    assert receipt["run_control"]["polls"] == 8 + 1
    # A stop that arrives with the final event does not turn a completed run
    # into a stopped one.
    runtime = CpuCartesianRuntime(_config(), seed=3)
    control_dir = require_control_directory(tmp_path / "control-2")
    _request_after(runtime, control_dir, STOP, 8)
    receipt = _integrate(runtime, tmp_path / "second", control=RunControl(control_dir))
    assert receipt["status"] == INTEGRATION_STATUS
    assert len(receipt["frames"]) == 4


def test_a_run_without_control_keeps_exactly_its_receipt_keys(tmp_path) -> None:
    receipt = _integrate(CpuCartesianRuntime(_config(), seed=3), tmp_path)
    assert "control_stop" not in receipt and "run_control" not in receipt
    assert "history_initial" not in receipt


# ------------------------------------------------------------------ pause


def test_a_pause_ends_the_run_right_after_the_next_restart_is_durable(
    forecast_publisher, tmp_path
) -> None:
    control_dir = require_control_directory(tmp_path / "control")
    runtime = CpuCartesianRuntime(_config(), seed=3)
    # Seen at step 1; the next restart is at step 4.
    _request_after(runtime, control_dir, PAUSE, 1)
    receipt = _integrate(
        runtime, tmp_path, publisher=forecast_publisher,
        restart_publisher=forecast_publisher, restarts=True,
        control=RunControl(control_dir),
    )
    assert receipt["status"] == INTEGRATION_CONTROL_PAUSE_STATUS
    pause = receipt["control_stop"]
    assert pause["request"] == PAUSE and pause["seen_at_step"] == 1
    assert pause["ended_at_step"] == 4 and pause["restart_number"] == 1
    assert sorted(p.name for p in (tmp_path / "restarts").iterdir()) == ["restart-000001"]
    assert receipt["restarts"][-1]["step"] == 4
    assert receipt["claim_limit"].endswith("and the last restart continues it")
    assert not _staging_left(tmp_path)


def test_a_paused_run_continued_from_its_restart_ends_where_an_uninterrupted_one_does(
    forecast_publisher, tmp_path
) -> None:
    """Pause, then --restart-from: the words of the double's whole carried state.

    The double carries no swaths (restart profile v2 puts those in the restart), so this is
    the claim for every carrier a restart holds today.
    """

    from cm1core.cartesian_restart import read_cartesian_restart

    control_dir = require_control_directory(tmp_path / "control")
    paused = CpuCartesianRuntime(_config(), seed=3)
    _request_after(paused, control_dir, PAUSE, 1)
    receipt = _integrate(
        paused, tmp_path / "first", publisher=forecast_publisher,
        restart_publisher=forecast_publisher, restarts=True,
        control=RunControl(control_dir),
    )
    restart = receipt["control_stop"]["restart_number"]
    resumed = CpuCartesianRuntime(_config(), seed=3)
    resumed.restore_restart(read_cartesian_restart(
        tmp_path / "first" / "restarts" / f"restart-{restart:06d}",
        publisher=forecast_publisher, into=tmp_path / "restored",
    ))
    second = resumed.integrate(
        duration_s=4.0, history_interval_s=1.0, history_root=tmp_path / "second" / "frames",
        history_publisher=forecast_publisher, restart_root=tmp_path / "second" / "restarts",
        restart_interval_s=2.0, restart_publisher=forecast_publisher,
        first_history_index=receipt["next_history_index"],
        first_restart_number=restart + 1,
    )
    assert second["steps_before_integration"] == 4 and second["completed_steps"] == 8
    uninterrupted = CpuCartesianRuntime(_config(), seed=3)
    _integrate(uninterrupted, tmp_path / "whole", publisher=forecast_publisher,
               restart_publisher=forecast_publisher, restarts=True)
    assert resumed.raw_state() == uninterrupted.raw_state()
    # The continued run publishes the frames the paused one did not.
    assert sorted(p.name for p in (tmp_path / "second" / "frames").iterdir()) == [
        "frame-000003", "frame-000004",
    ]


def test_a_pause_with_no_restart_to_write_says_so_and_keeps_running(tmp_path) -> None:
    control_dir = require_control_directory(tmp_path / "control")
    log = RunEventLog(tmp_path / "events.jsonl")
    runtime = CpuCartesianRuntime(_config(), seed=3)
    _request_after(runtime, control_dir, PAUSE, 1)
    control = RunControl(control_dir, log)
    receipt = _integrate(runtime, tmp_path, control=control)
    log.close()
    assert receipt["status"] == INTEGRATION_STATUS
    assert receipt["run_control"]["pause_refused"] is True
    warnings = [e for e in read_events(tmp_path / "events.jsonl") if e["kind"] == "warning"]
    assert len(warnings) == 1
    assert warnings[0]["code"] == "GM1-RUN-CONTROL-PAUSE-UNAVAILABLE"
    assert "no restarts" in warnings[0]["message"]


def test_a_stop_after_a_pause_still_stops(tmp_path) -> None:
    control_dir = require_control_directory(tmp_path / "control")
    runtime = CpuCartesianRuntime(_config(), seed=3)
    stepping = runtime.step

    def step() -> None:
        stepping()
        if runtime.clock.completed_steps == 1:
            (control_dir / PAUSE).touch()
        if runtime.clock.completed_steps == 3:
            (control_dir / STOP).touch()

    runtime.step = step
    receipt = _integrate(runtime, tmp_path, control=RunControl(control_dir))
    assert receipt["status"] == INTEGRATION_CONTROL_STOP_STATUS
    assert receipt["control_stop"]["seen_at_step"] == 3 == receipt["completed_steps"]


# ------------------------------------------------------------ the directory


def test_a_stale_request_is_refused_before_the_run(tmp_path) -> None:
    control_dir = tmp_path / "control"
    assert require_control_directory(control_dir) == control_dir and control_dir.is_dir()
    for name in (STOP, PAUSE):
        (control_dir / name).touch()
        with pytest.raises(ValueError, match=name):
            require_control_directory(control_dir)
        (control_dir / name).unlink()
    (tmp_path / "file").write_text("")
    with pytest.raises(ValueError, match="not a directory"):
        require_control_directory(tmp_path / "file")


def test_polling_stats_the_directory_once_and_lists_it_only_when_it_changes(
    tmp_path, monkeypatch
) -> None:
    import cm1core.run_control as run_control

    control_dir = require_control_directory(tmp_path / "control")
    old = os.stat(control_dir).st_mtime - 60.0
    os.utime(control_dir, (old, old))
    clock = [100.0]
    control = RunControl(control_dir, clock=lambda: clock[0])
    listed = []
    real_listdir = os.listdir
    monkeypatch.setattr(run_control.os, "listdir", lambda path: listed.append(path) or real_listdir(path))
    assert control.poll() is None and len(listed) == 1
    for _ in range(10):
        assert control.poll() is None
    assert len(listed) == 1 and control.polls == 11
    (control_dir / PAUSE).touch()
    assert control.poll() == PAUSE
    assert control.request_seen["request"] == PAUSE


def test_a_stop_is_seen_even_when_the_directory_time_does_not_move(tmp_path) -> None:
    """NFS can serve a directory's cached attributes for up to a minute.

    The directory is listed at least every LIST_AT_LEAST_EVERY_S whatever its
    modification time says, so a stop written from another host is seen.
    """

    from cm1core.run_control import LIST_AT_LEAST_EVERY_S

    control_dir = require_control_directory(tmp_path / "control")
    old = os.stat(control_dir).st_mtime - 60.0
    os.utime(control_dir, (old, old))
    clock = [100.0]
    control = RunControl(control_dir, clock=lambda: clock[0])
    assert control.poll() is None and control.listings == 1
    # The file lands, but the stat() still reports the old time, as a cached
    # attribute would.
    (control_dir / STOP).touch()
    os.utime(control_dir, (old, old))
    clock[0] += LIST_AT_LEAST_EVERY_S / 2
    assert control.poll() is None and control.listings == 1
    clock[0] += LIST_AT_LEAST_EVERY_S / 2
    assert control.poll() == STOP and control.listings == 2
    assert control.stop_requested


# ------------------------------------------------------------ frame 0


def test_history_initial_publishes_the_initialized_state_as_frame_zero(tmp_path) -> None:
    runtime = CpuCartesianRuntime(_config(), seed=3)
    initial = runtime.raw_state()
    progress = []
    receipt = _integrate(runtime, tmp_path, history_initial=True, progress=progress.append)
    frames = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert frames == [f"frame-{index:06d}" for index in range(5)]
    zero = tmp_path / "frames" / "frame-000000"
    publication = json.loads((zero / "history-publication.json").read_text())
    assert publication["step"] == 0 and publication["model_time_s"] == 0.0
    assert publication["initial_condition"] is True
    assert publication["quiescent_boundary"].startswith("initialized source state")
    location = json.loads((zero / "history-frame-location.json").read_text())
    assert location["index"] == 0 and location["published_directory"] == os.fspath(zero)
    # Reported first, in model order, like every other frame.
    assert [row["index"] for row in progress] == [0, 1, 2, 3, 4]
    assert receipt["frames"][0]["index"] == 0 and receipt["frames"][0]["step"] == 0
    assert receipt["history_initial"]["frame_index"] == 0
    # Publishing frame 0 touched no word of the state the run integrated.
    reference = CpuCartesianRuntime(_config(), seed=3)
    assert reference.raw_state() == initial
    for _ in range(8):
        reference.step()
    assert reference.raw_state() == runtime.raw_state()
    # The published words are the initialized carriers.
    arena = (zero / "history-carriers.f32").read_bytes()
    offset = publication["arena"]["carrier_offset_bytes"]["th3d"]
    th3d = CpuCartesianRuntime(_config(), seed=3).dynamics.fields["th3d"]
    assert arena[offset:offset + th3d.nbytes] == th3d.astype("<f4").tobytes()


def test_a_stopped_run_counts_frame_zero_as_its_event_log_does(tmp_path) -> None:
    """control_stop's frame counts include frame 0, like the end record's."""

    control_dir = require_control_directory(tmp_path / "control")
    runtime = CpuCartesianRuntime(_config(), seed=3)
    _request_after(runtime, control_dir, STOP, 5)
    events = RunEventLog(tmp_path / "events.jsonl")
    progress = []
    receipt = _integrate(
        runtime, tmp_path, control=RunControl(control_dir, events),
        history_initial=True, progress=progress.append,
    )
    events.close()
    stop = receipt["control_stop"]
    frames = sorted(p.name for p in (tmp_path / "frames").iterdir())
    assert frames == ["frame-000000", "frame-000001", "frame-000002"]
    assert stop["frames_published"] == len(frames) == len(progress) == 3
    assert stop["frames_requested"] == 4 + 1


def test_history_initial_is_refused_on_a_resumed_runtime(tmp_path) -> None:
    runtime = CpuCartesianRuntime(_config(), seed=3)
    runtime.step()
    with pytest.raises(ValueError, match="frame 0 belongs to the run that started it"):
        _integrate(runtime, tmp_path, history_initial=True)


def test_history_initial_leaves_dbz_out_of_frame_zero_only(tmp_path) -> None:
    from cm1core.cartesian_history import (
        INITIAL_FRAME_REFLECTIVITY_OMITTED,
        cartesian_history_profile,
        initial_frame_history_profile,
    )

    config = _config(output_dbz=True)
    full = cartesian_history_profile(config)
    initial = initial_frame_history_profile(config)
    full_names = [carrier.name for carrier in full.carriers]
    assert {"dbz", "cref"} <= set(full_names)
    assert [carrier.name for carrier in initial.carriers] == [
        name for name in full_names if name not in ("dbz", "cref", "dbz_level")
    ]
    assert "seed" in INITIAL_FRAME_REFLECTIVITY_OMITTED
    # Without output_dbz the two profiles are the same frame.
    plain = _config()
    assert (
        [c.name for c in initial_frame_history_profile(plain).carriers]
        == [c.name for c in cartesian_history_profile(plain).carriers]
    )


# ------------------------------------------------------------ envelope off


def test_envelope_off_publishes_no_history_envelope_and_still_publishes_restarts(
    forecast_publisher, tmp_path
) -> None:
    runtime = CpuCartesianRuntime(_config(), seed=3)
    receipt = _integrate(
        runtime, tmp_path, publisher=None, restart_publisher=forecast_publisher,
        restarts=True,
    )
    assert receipt["status"] == INTEGRATION_STATUS
    assert not list((tmp_path / "frames").rglob("history.cm1fhs"))
    assert not list((tmp_path / "frames").rglob(".history-field-table.publisher.tsv"))
    assert sorted(p.name for p in (tmp_path / "restarts").iterdir()) == [
        "restart-000001", "restart-000002",
    ]
    for restart in (tmp_path / "restarts").iterdir():
        assert (restart / "restart.cm1fcp").is_file()
    for frame in (tmp_path / "frames").iterdir():
        publication = json.loads((frame / "history-publication.json").read_text())
        assert publication["forecast_envelope"]["published"] is False
        assert publication["finite_checked_by_rust"] is False


def test_the_published_table_names_the_arena_and_the_publisher_table_is_gone(
    forecast_publisher, tmp_path
) -> None:
    runtime = CpuCartesianRuntime(_config(), seed=3)
    _integrate(runtime, tmp_path, publisher=forecast_publisher)
    for frame in sorted((tmp_path / "frames").iterdir()):
        assert (frame / "history.cm1fhs").is_file()
        assert not (frame / ".history-field-table.publisher.tsv").exists()
        rows = (frame / "history-field-table.tsv").read_text().splitlines()
        assert {row.split("\t")[6] for row in rows} == {"history-carriers.f32"}


# ------------------------------------------------------------ events


def test_the_integration_writes_valid_progress_and_frame_records(tmp_path) -> None:
    from cm1core.cartesian_door import _RunReporting

    log = RunEventLog(tmp_path / "events.jsonl")
    control = RunControl(None, log, progress_interval_s=0.0)
    reporting = _RunReporting(log, control)
    reporting.admit(
        config="double", config_sha256="0" * 64,
        grid={"nx": 9, "ny": 11, "nz": 8, "dx_m": 500.0, "dy_m": 500.0,
              "dz_m": 500.0, "stretch_z": 0, "ztop_m": 4000.0},
        microphysics="wsm6-mp6", steps=8, start_step=0, dt_s=0.5, nsound=6,
        duration_s=4.0, history_interval_s=1.0, frames_expected=5,
        restart_interval_s=None, restarts_expected=0,
        history_directory=os.fspath(tmp_path / "frames"), restart_directory=None,
        receipt=os.fspath(tmp_path / "receipt.json"), control_directory=None,
        history_envelope="off", history_initial=True, host="test",
    )
    runtime = CpuCartesianRuntime(_config(), seed=3)
    _integrate(runtime, tmp_path, history_initial=True, control=control,
               progress=reporting.frame_progress)
    reporting.end(None, 0)
    log.close()
    assert validate_file(tmp_path / "events.jsonl") == []
    events = read_events(tmp_path / "events.jsonl")
    kinds = [event["kind"] for event in events]
    assert kinds[0] == "admitted" and kinds[1] == "startup" and kinds[-1] == "end"
    frames = [event for event in events if event["kind"] == "frame"]
    assert [event["index"] for event in frames] == [0, 1, 2, 3, 4]
    assert frames[0]["initial_condition"] is True and frames[0]["model_time_s"] == 0.0
    assert all(event["bytes"] > 0 for event in frames)
    progress = [event for event in events if event["kind"] == "progress"]
    stepping = [
        event for event in progress
        if event["phase"] == "stepping" and not event["heartbeat"]
    ]
    assert [event["step"] for event in stepping] == list(range(0, 9))
    assert stepping[-1]["total_steps"] == 8 and stepping[-1]["duration_s"] == 4.0
    assert progress[-1]["phase"] == "finishing"
    end = events[-1]
    assert end["reason"] == "completed" and end["rc"] == 0
    assert end["step"] == 8 and end["frames_published"] == 5
