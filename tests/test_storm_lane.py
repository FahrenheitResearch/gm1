"""CPU admission and arithmetic for the sheared-lane integration entry point.

Nothing here opens a CUDA device or executes a CM1 operator.  What is checked
is the part that must be right before a device is opened at all: the CM1
base.F base-state selectors as strict configuration, the large-step frame
schedule, the boundedness of everything the runtime accumulates per step, and
the atomic layout of the published frames.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cm1core import cuda_cartesian_runtime as runtime_module
from cm1core.cartesian_history import COMPLETE_CARRIER_PROFILE
import cm1core.cartesian_door as cartesian_door
from cm1core.cartesian_door import (
    moving_frame_record,
    require_history_cadence,
    require_initializer_selection,
    require_moving_frame,
    require_new_history_root,
    run_cartesian,
)
from cm1core.config import Lane, load_config, parse_mapping
from cm1core.cuda_cartesian_runtime import (
    FRAME_DIRECTORY_FORMAT,
    FRAME_LOCATION_FILENAME,
    FRAME_STAGING_DIRECTORY,
    INTEGRATION_COURANT_STOP_STATUS,
    INTEGRATION_SCHEMA,
    INTEGRATION_STATUS,
    CartesianCourantStop,
    CudaCartesianRuntime,
)
from cm1core.errors import CM1Refusal
from cm1core.gm1_idealized_residency import (
    FIXED_FRAME,
    SINGLE_SOUNDING_SELECTION,
    IdealizedSource,
    InitializerSelection,
    TranslatingFrame,
)
from cm1core.gm1_runtime import (
    CM1_COURANT_STOP,
    CM1_COURANT_STOP_AUTHORITY,
    CM1_COURANT_TARGET,
    RuntimeClock,
    RuntimeConfig,
    advective_courant_defect,
    history_cadence,
)
from cm1core.grid_geometry import CartesianGridGeometry
from cm1core.preflight import blockers


ROOT = Path(__file__).resolve().parents[1]
ANALYTIC = ROOT / "examples" / "cartesian-wk-analytic-sounding-shear.toml"
TRANSLATING = ROOT / "examples" / "cartesian-supercell-translating-frame.toml"
FIXED_DOMAIN = ROOT / "examples" / "cartesian-supercell-attempt.toml"
SINGLE_SOUNDING = ROOT / "examples" / "cartesian-uniform-grid.toml"


# --------------------------------------------------------------- cadence


def test_the_cm1_idealized_deck_is_fourteen_thousand_steps_and_a_hundred_frames() -> None:
    """CM1's own two-hour deck, at the half second this lane defaults to.

    The step is a free parameter now (``solve2.F:145``); 0.5 s is what
    ``RuntimeConfig`` and the shipped examples declare, and this pins the
    cadence that default produces.
    """

    cadence = history_cadence(dt_s=0.5, duration_s=7200.0, interval_s=60.0)
    assert cadence.total_steps == 14400
    assert cadence.steps_per_frame == 120
    assert cadence.frame_count == 120
    assert cadence.frame_steps()[0] == 120
    assert cadence.frame_steps()[-1] == 14400
    assert cadence.frame_model_seconds()[0] == 60.0
    assert cadence.frame_model_seconds()[-1] == 7200.0
    assert len(cadence.frame_steps()) == cadence.frame_count
    assert cadence.record()["last_frame_model_seconds"] == 7200.0


@pytest.mark.parametrize(
    ("duration_s", "interval_s", "frames", "steps_per_frame"),
    (
        # The interval equals the duration: exactly one end-of-run frame.
        (5.0, 5.0, 1, 10),
        # The interval equals the timestep: every completed step is a frame.
        (5.0, 0.5, 10, 1),
        (7200.0, 3600.0, 2, 7200),
        (7200.0, 0.5, 14400, 1),
        (30.0, 5.0, 6, 10),
    ),
)
def test_frame_count_is_exact_at_both_ends_of_the_interval(
    duration_s: float, interval_s: float, frames: int, steps_per_frame: int
) -> None:
    cadence = history_cadence(dt_s=0.5, duration_s=duration_s, interval_s=interval_s)
    assert (cadence.frame_count, cadence.steps_per_frame) == (frames, steps_per_frame)
    # Every frame lands on a completed step, and the last one is the run's end.
    assert cadence.frame_steps()[-1] == cadence.total_steps
    assert cadence.frame_model_seconds()[-1] == duration_s
    assert all(step >= 1 for step in cadence.frame_steps())


@pytest.mark.parametrize(
    ("duration_s", "interval_s", "match"),
    (
        (5.0, 10.0, "end before its first frame"),
        (30.0, 4.0, "stop short of the requested duration"),
        (5.25, 0.5, "duration_s=5.25 is not a whole number"),
        (5.0, 0.3, "history_interval_s=0.3 is not a whole number"),
        (5.0, 0.0, "history_interval_s must be finite and positive"),
        (5.0, -60.0, "history_interval_s must be finite and positive"),
        (0.25, 0.25, "duration_s=0.25 is not a whole number"),
        (float("inf"), 60.0, "duration_s must be finite and positive"),
        (float("nan"), 60.0, "duration_s must be finite and positive"),
    ),
)
def test_inadmissible_durations_and_intervals_refuse_by_value(
    duration_s: float, interval_s: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        history_cadence(dt_s=0.5, duration_s=duration_s, interval_s=interval_s)


@pytest.mark.parametrize("bad", (True, "60", None))
def test_cadence_numbers_do_not_coerce(bad: object) -> None:
    with pytest.raises(TypeError):
        history_cadence(dt_s=0.5, duration_s=7200.0, interval_s=bad)


# ------------------------------------------------ base.F selector contract


def test_selection_names_the_cm1_branches_and_refuses_the_discarded_pair() -> None:
    record = SINGLE_SOUNDING_SELECTION.record()
    assert (record["isnd"], record["iwnd"], record["iinit"]) == (7, 0, 1)
    assert SINGLE_SOUNDING_SELECTION.reads_input_sounding
    analytic = InitializerSelection(isnd=5, iwnd=2)
    assert (analytic.record()["isnd"], analytic.record()["iwnd"]) == (5, 2)
    assert not analytic.reads_input_sounding
    # base.F:2263 zeroes the wind for isnd != 7, so isnd=5 with iwnd=0 is a
    # defined, windless base state rather than a contradiction.
    assert InitializerSelection(isnd=5, iwnd=0).iwnd == 0
    # base.F:495-607's isnd=17 reads the same file for thermodynamics only and
    # takes its winds from the iwnd branch (param.F:849-851).
    assert InitializerSelection(isnd=17, iwnd=1).reads_input_sounding
    with pytest.raises(ValueError, match="param.F:836-852"):
        InitializerSelection(isnd=7, iwnd=2)
    # Every other base.F arm is transcribed; 6 (removed from CM1), 15 (needs
    # testcase 4/5's constants) and values with no arm remain refused.
    for isnd, iwnd in ((6, 0), (15, 0), (16, 6)):
        with pytest.raises(ValueError, match="transcribed"):
            InitializerSelection(isnd=isnd, iwnd=iwnd)
    # base.F:2286-2640 has no iwnd=7 arm.
    with pytest.raises(ValueError, match="no arm for iwnd=7"):
        InitializerSelection(isnd=5, iwnd=7)
    for bad in (True, 5.0, "5"):
        with pytest.raises(TypeError):
            InitializerSelection(isnd=bad, iwnd=0)


def test_an_initializer_without_the_selectors_refuses_the_lane_by_name() -> None:
    """Wiring a selector a library cannot take would build another atmosphere."""

    geometry = CartesianGridGeometry(9, 9, 8, 500.0, 500.0, 500.0)
    seen: list[dict[str, object]] = []

    class WithoutSelectors:
        def __init__(self, library, sounding_bytes, *, iinit, geometry):
            seen.append({"iinit": iinit})
            self.groups = {}

        def close(self) -> None:
            pass

    class WithSelectors(WithoutSelectors):
        def __init__(self, library, sounding_bytes, *, iinit, geometry, isnd, iwnd):
            super().__init__(library, sounding_bytes, iinit=iinit, geometry=geometry)
            seen[-1].update(isnd=isnd, iwnd=iwnd)

    # The single-sounding lane never depends on the new arguments.
    IdealizedSource(__file__, b"sounding", geometry, owner_factory=WithoutSelectors)
    assert seen[-1] == {"iinit": 1}

    with pytest.raises(ValueError, match="initializer lane is unavailable"):
        IdealizedSource(
            __file__,
            b"",
            geometry,
            selection=InitializerSelection(isnd=5, iwnd=2),
            owner_factory=WithoutSelectors,
        )

    source = IdealizedSource(
        __file__,
        b"",
        geometry,
        selection=InitializerSelection(isnd=5, iwnd=2),
        owner_factory=WithSelectors,
    )
    assert seen[-1] == {"iinit": 1, "isnd": 5, "iwnd": 2}
    assert source.selection.record()["isnd"] == 5
    assert source.selection.record()["iwnd"] == 2


# --------------------------------------------- base.F:2661-2674 frame


def test_the_translating_frame_is_the_cm1_change_of_variable() -> None:
    assert FIXED_FRAME.record() == {"u_move_m_s": 0.0, "v_move_m_s": 0.0}
    assert not FIXED_FRAME.translates
    assert FIXED_FRAME.imove == 0
    # run/namelist.input:101-102, CM1's own supercell deck.
    deck = TranslatingFrame(u_move_m_s=12.5, v_move_m_s=3.0)
    assert deck.translates
    assert deck.imove == 1
    assert deck.record() == {"u_move_m_s": 12.5, "v_move_m_s": 3.0}
    # An integer speed is a speed; a bool is not.
    assert TranslatingFrame(u_move_m_s=12, v_move_m_s=0).u_move_m_s == 12.0
    for bad in (True, "12.5", None):
        with pytest.raises(TypeError, match="must be a float in m/s"):
            TranslatingFrame(u_move_m_s=bad)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="not finite"):
            TranslatingFrame(v_move_m_s=bad)
    for bad in (151.0, -1.0e6):
        with pytest.raises(ValueError, match="exceeds"):
            TranslatingFrame(u_move_m_s=bad)
    assert TranslatingFrame(u_move_m_s=150.0, v_move_m_s=-150.0).translates


def test_an_initializer_without_the_frame_refuses_a_translating_run_by_name() -> None:
    """A dropped frame builds a ground-relative state that nothing labels."""

    geometry = CartesianGridGeometry(9, 9, 8, 500.0, 500.0, 500.0)
    seen: list[dict[str, object]] = []

    class WithoutFrame:
        def __init__(self, library, sounding_bytes, *, iinit, geometry):
            seen.append({"iinit": iinit})
            self.groups = {}

        def close(self) -> None:
            pass

    class WithFrame(WithoutFrame):
        def __init__(
            self, library, sounding_bytes, *, iinit, geometry, u_move_m_s, v_move_m_s
        ):
            super().__init__(library, sounding_bytes, iinit=iinit, geometry=geometry)
            seen[-1].update(u_move_m_s=u_move_m_s, v_move_m_s=v_move_m_s)

    # A fixed frame is the historical call and passes nothing new, so an
    # older owner keeps working unchanged.
    source = IdealizedSource(__file__, b"sounding", geometry, owner_factory=WithoutFrame)
    assert seen[-1] == {"iinit": 1}
    assert source.moving_frame == FIXED_FRAME

    with pytest.raises(ValueError, match="wrong by u_move_m_s times the model time"):
        IdealizedSource(
            __file__,
            b"sounding",
            geometry,
            moving_frame=TranslatingFrame(u_move_m_s=12.5, v_move_m_s=3.0),
            owner_factory=WithoutFrame,
        )

    source = IdealizedSource(
        __file__,
        b"sounding",
        geometry,
        moving_frame=TranslatingFrame(u_move_m_s=12.5, v_move_m_s=3.0),
        owner_factory=WithFrame,
    )
    assert seen[-1] == {"iinit": 1, "u_move_m_s": 12.5, "v_move_m_s": 3.0}
    assert source.moving_frame.record() == {"u_move_m_s": 12.5, "v_move_m_s": 3.0}

    with pytest.raises(TypeError, match="moving_frame must be TranslatingFrame"):
        IdealizedSource(
            __file__, b"sounding", geometry, moving_frame=(12.5, 3.0),
            owner_factory=WithFrame,
        )


def test_the_analytic_lane_carries_the_declared_frame_into_the_receipt() -> None:
    """Round trip: what the configuration asked for is what is recorded."""

    config = load_config(ANALYTIC)
    assert require_moving_frame(config) == FIXED_FRAME

    payload = config.canonical_dict()
    raw = {
        section: copy.deepcopy(payload[section])
        for section in (
            "run", "grid", "dynamics", "boundaries", "closures", "physics",
            "runtime", "initialization",
        )
    }
    raw["closures"].update(moving_frame=True, u_move_m_s=12.5, v_move_m_s=3.0)
    moving = parse_mapping(raw)
    frame = require_moving_frame(moving)
    assert frame.record() == {"u_move_m_s": 12.5, "v_move_m_s": 3.0}
    assert moving_frame_record(frame)["u_move_m_s"] == moving.closures.u_move_m_s
    assert moving_frame_record(frame)["v_move_m_s"] == moving.closures.v_move_m_s


def test_the_shipped_translating_frame_profile_binds_cm1s_deck_velocity() -> None:
    """The example is the documented surface; it must not rot."""

    config = load_config(TRANSLATING)
    assert config.closures.moving_frame
    # run/namelist.input:101-102.
    assert (config.closures.u_move_m_s, config.closures.v_move_m_s) == (12.5, 3.0)
    frame = require_moving_frame(config)
    assert frame.record() == {"u_move_m_s": 12.5, "v_move_m_s": 3.0}
    # The point of the frame: 60 km is enough where the fixed-domain profile
    # beside it needs 220 km for the same two hours of storm track.
    fixed = load_config(FIXED_DOMAIN)
    assert not fixed.closures.moving_frame
    assert require_moving_frame(fixed) == FIXED_FRAME
    assert config.grid.nx < fixed.grid.nx
    assert config.run.duration_s == fixed.run.duration_s
    assert config.initialization.isnd == fixed.initialization.isnd == 5
    assert config.initialization.iwnd == fixed.initialization.iwnd == 2


# ------------------------------------------------------ bounded accounting


class _MemoryPool:
    def used_bytes(self) -> int:
        return 7

    def total_bytes(self) -> int:
        return 9


def _sampling_shell(budget: int | None):
    shell = object.__new__(CudaCartesianRuntime)
    state = {"free": 100}

    def mem_get_info():
        # A monotonically shrinking free pool: the peak is the last sample,
        # which a truncated log would otherwise never see.
        state["free"] -= 1
        return state["free"], 1000

    shell.cp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=mem_get_info)),
        get_default_memory_pool=lambda: _MemoryPool(),
    )
    shell.registry = SimpleNamespace(registered_allocation_bytes=3)
    shell.memory_samples = []
    shell.memory_samples_dropped = 0
    shell._memory_peaks = dict.fromkeys(runtime_module.MEMORY_SAMPLE_KEYS, 0)
    shell.memory_sample_budget = budget
    return shell


def test_memory_sampling_is_bounded_and_still_reports_the_true_peak() -> None:
    """Twenty phases a step over 14,400 steps cannot be a retained list."""

    shell = _sampling_shell(64)
    for step in range(1, 2001):
        shell._sample("acoustic", step=step)
    assert len(shell.memory_samples) == 64
    assert shell.memory_samples_dropped == 2000 - 64
    # The last sample has the most device memory in use and was dropped.
    assert shell.peak_bytes("device_used_bytes") == 1000 - (100 - 2000)
    assert shell.peak_bytes("device_used_bytes") > max(
        row["device_used_bytes"] for row in shell.memory_samples
    )


def test_unbounded_sampling_is_still_the_default_for_the_timing_harness() -> None:
    shell = _sampling_shell(None)
    for step in range(1, 51):
        shell._sample("acoustic", step=step)
    assert len(shell.memory_samples) == 50
    assert shell.memory_samples_dropped == 0
    assert shell.peak_bytes("device_used_bytes") == max(
        row["device_used_bytes"] for row in shell.memory_samples
    )


# -------------------------------------------------------- the integration


class _NoOpBackend:
    """A CartesianBackend that executes nothing, so only the clock moves."""

    def execute(self, operation, *, step):
        pass

    def synchronize(self):
        pass


def _integration_shell(config: RuntimeConfig):
    """A CudaCartesianRuntime shell: only the integration inputs are emulated."""

    shell = object.__new__(CudaCartesianRuntime)
    shell.config = config
    shell.cp = None
    shell.stream = SimpleNamespace(synchronize=lambda: None)
    shell.registry = SimpleNamespace(registered_allocation_bytes=0, buffers={})
    # The real clock, so the integration receipt's domain_location is
    # cm1.F:3512-3513's own running sum and not a fixture constant.
    shell.clock = RuntimeClock(config)
    shell.finish = SimpleNamespace(
        completed_physics_step=0, completed_solve3_step=0, step_receipts={}
    )
    shell.dynamics = SimpleNamespace(last_receipts={}, _expected_phase=("held", 1))
    shell.closure = SimpleNamespace(last_receipts={})
    shell.source_identity = {"kind": "test-shell", "initialization": {"isnd": 5, "iwnd": 2}}
    shell.runtime_source_identity = {"aggregate_sha256": "22" * 32}
    shell.warm_component = {"levels": config.grid.nz}
    shell.component_scratch_reserve = {"levels": config.grid.nz}
    shell.phase_wall_seconds = {}
    shell.memory_samples = []
    shell.memory_samples_dropped = 0
    shell._memory_peaks = dict.fromkeys(runtime_module.MEMORY_SAMPLE_KEYS, 0)
    shell.checks = 0
    # misclibs.F:2714.  The integration measures CM1's advective Courant
    # number before its first step; the shell records that it was asked and
    # answers with a value the run is allowed to continue from.
    shell.courant_number_value = 0.25
    shell.courant_checks = []
    # CM1 &param9.  This shell emulates the scheduler, not the kernels: the
    # run receipt must still name the selection this run resolved, and the
    # frame publisher must still be handed a products mapping.
    shell.output_fields = {}
    shell.output = SimpleNamespace(
        reflectivity_wanted=False,
        per_step=lambda step: None,
        materialize=lambda: shell.output_fields,
        record=lambda: {
            "selection": config.output.record(),
            "unvalidated": [dict(row) for row in config.output.unvalidated_markers()],
        },
    )
    shell._reflectivity_due = False

    backend = _NoOpBackend()

    def step() -> None:
        shell.clock.step(backend)
        shell.finish.completed_physics_step = shell.clock.completed_steps
        shell.finish.completed_solve3_step = shell.clock.completed_steps
        # One memory sample per scheduled phase, exactly as execute() takes.
        for phase in range(20):
            shell.memory_samples_dropped += 0
            if (
                shell.memory_sample_budget is None
                or len(shell.memory_samples) < shell.memory_sample_budget
            ):
                shell.memory_samples.append({"phase": phase, "device_used_bytes": 1})
            else:
                shell.memory_samples_dropped += 1

    shell.step = step
    shell.synchronize = lambda: None

    # A measurement schedule, when a test needs the number to change: the first
    # entry answers the pre-first-step check and one entry answers each frame.
    shell.courant_sequence = None

    def advective_courant_record() -> dict[str, object]:
        if shell.courant_sequence:
            shell.courant_number_value = shell.courant_sequence.pop(0)
        defect = advective_courant_defect(
            courant=shell.courant_number_value, dt_s=config.dt_s
        )
        return {
            "number": shell.courant_number_value,
            "stop": CM1_COURANT_STOP,
            "target": CM1_COURANT_TARGET,
            "above_adaptive_target": shell.courant_number_value > CM1_COURANT_TARGET,
            "stop_reached": defect is not None,
            "stop_reason": defect,
            "authority": CM1_COURANT_STOP_AUTHORITY,
        }

    shell.advective_courant_record = advective_courant_record

    def field_checks() -> dict[str, object]:
        shell.checks += 1
        return {
            "th3d": {"finite": True, "minimum": 280.0, "maximum": 320.0},
            "advective_courant": shell.advective_courant_record(),
        }

    shell.field_checks = field_checks

    def require_advective_courant(*, when: str) -> float:
        shell.courant_checks.append(when)
        record = shell.advective_courant_record()
        if record["stop_reason"] is not None:
            raise RuntimeError(f"{when}: {record['stop_reason']}")
        return float(record["number"])

    shell.require_advective_courant = require_advective_courant
    return shell


def _install_publication(monkeypatch, write, *, seen: list[Path] | None = None):
    """Replace the frame snapshot and commit the way cartesian_history splits them.

    The snapshot runs on the stepping thread and reads the runtime; the commit
    may run on the background worker, after the runtime has stepped on, so it
    reads only what the snapshot captured -- exactly the rule the shipped pair
    keeps.  ``write(directory, captured)`` writes the frame and returns the
    receipt.
    """

    def snapshot(runtime, directory, *, publisher=None, profile=None,
                 carrier_profile=COMPLETE_CARRIER_PROFILE,
                 initial_condition=False, host=None):
        directory = Path(directory)
        if seen is not None:
            seen.append(directory)
        return SimpleNamespace(
            directory=directory,
            step=int(runtime.clock.completed_steps),
            model_time_s=float(runtime.clock.model_seconds),
            moving_frame=runtime.clock.domain_location_record(),
        )

    def commit(captured, *, io=None, cancel=None):
        return write(captured.directory, captured)

    monkeypatch.setattr(runtime_module, "snapshot_cartesian_history", snapshot)
    monkeypatch.setattr(runtime_module, "commit_cartesian_history", commit)


def _fake_publisher(monkeypatch, *, seen: list[Path] | None = None):
    """Write a frame the way cartesian_history does: into the given directory."""

    def write(directory, captured):
        directory.mkdir(parents=True)
        arena = directory / "history-carriers.f32"
        arena.write_bytes(b"\0" * 16)
        table = directory / "history-field-table.tsv"
        table.write_text("th3d\t...\n", encoding="utf-8")
        scalars = directory / "history-scheduler-scalars.tsv"
        scalars.write_text("domainlocx\tf64-bits\t0000000000000000\n", encoding="utf-8")
        return {
            "step": captured.step,
            "model_time_s": captured.model_time_s,
            "arena": {"path": os.fspath(arena), "sha256": "ab" * 32, "nbytes": 16},
            "field_table_path": os.fspath(table),
            "scalar_table_path": os.fspath(scalars),
            "moving_frame": captured.moving_frame,
            "forecast_envelope": {"published": False, "reason": "no publisher"},
            "finite_checked_by_rust": False,
        }

    _install_publication(monkeypatch, write, seen=seen)
    return write


def _config(nx: int = 9, ny: int = 9, nz: int = 8) -> RuntimeConfig:
    return RuntimeConfig(CartesianGridGeometry(nx, ny, nz, 500.0, 500.0, 500.0))


def test_integration_runs_to_the_duration_and_publishes_every_interval(
    monkeypatch, tmp_path: Path
) -> None:
    shell = _integration_shell(_config())
    _fake_publisher(monkeypatch)
    root = tmp_path / "frames"
    receipt = shell.integrate(
        duration_s=30.0, history_interval_s=5.0, history_root=root
    )
    assert receipt["schema"] == INTEGRATION_SCHEMA
    assert receipt["cadence"]["frame_count"] == 6
    assert receipt["executed_steps"] == 60
    assert shell.clock.completed_steps == 60
    assert receipt["model_seconds_at_end"] == 30.0
    assert shell.checks == 6
    published = sorted(path.name for path in root.glob("frame-*"))
    assert published == [FRAME_DIRECTORY_FORMAT.format(index=i) for i in range(1, 7)]
    # The staging subdirectory is removed once nothing is left in it.
    assert not (root / FRAME_STAGING_DIRECTORY).exists()
    for index, name in enumerate(published, start=1):
        location = json.loads((root / name / FRAME_LOCATION_FILENAME).read_text())
        assert location["index"] == index
        assert location["step"] == index * 10
        assert location["model_time_s"] == index * 5.0
        assert location["published_directory"] == os.fspath(root / name)
        # The recorded artifact paths are the published ones, not the staged
        # ones cartesian_history wrote into.
        assert Path(location["arena_path"]).is_file()
        assert Path(location["field_table_path"]).is_file()
        assert FRAME_STAGING_DIRECTORY in location["staged_directory"]
    assert [row["model_time_s"] for row in receipt["frames"]] == [
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0
    ]


def test_a_frame_directory_only_appears_once_it_is_complete(
    monkeypatch, tmp_path: Path
) -> None:
    """A watcher globbing frame-* must never open a half-written frame."""

    shell = _integration_shell(_config())
    observed: list[list[str]] = []
    root = tmp_path / "frames"

    def write(directory, captured):
        directory.mkdir(parents=True)
        # Mid-publication: nothing under the watched name exists yet.
        observed.append(sorted(path.name for path in root.glob("frame-*")))
        arena = directory / "history-carriers.f32"
        arena.write_bytes(b"\0" * 16)
        table = directory / "history-field-table.tsv"
        table.write_text("th3d\t...\n", encoding="utf-8")
        scalars = directory / "history-scheduler-scalars.tsv"
        scalars.write_text("domainlocx\tf64-bits\t0000000000000000\n", encoding="utf-8")
        return {
            "step": captured.step,
            "model_time_s": captured.model_time_s,
            "arena": {"path": os.fspath(arena), "sha256": "cd" * 32, "nbytes": 16},
            "field_table_path": os.fspath(table),
            "scalar_table_path": os.fspath(scalars),
            "moving_frame": captured.moving_frame,
            "forecast_envelope": {"published": False, "reason": "no publisher"},
            "finite_checked_by_rust": False,
        }

    _install_publication(monkeypatch, write)
    shell.integrate(duration_s=6.0, history_interval_s=2.0, history_root=root)
    assert observed == [[], ["frame-000001"], ["frame-000001", "frame-000002"]]


def test_nothing_the_integration_retains_grows_with_the_step_count(
    monkeypatch, tmp_path: Path
) -> None:
    """Ten times the steps must not mean ten times the receipt."""

    _fake_publisher(monkeypatch)
    sizes = {}
    for factor, duration in ((1, 50.0), (10, 500.0)):
        shell = _integration_shell(_config())
        receipt = shell.integrate(
            duration_s=duration,
            history_interval_s=10.0,
            history_root=tmp_path / f"frames-{factor}",
            memory_sample_budget=32,
            frame_row_budget=4,
        )
        assert receipt["executed_steps"] == int(duration / 0.5)
        assert len(shell.memory_samples) == 32
        assert shell.memory_samples_dropped == receipt["executed_steps"] * 20 - 32
        assert len(receipt["frames"]) == 4
        assert receipt["frame_rows_dropped"] == receipt["cadence"]["frame_count"] - 4
        # The per-step wall clock is folded, never listed.
        assert receipt["step_wall_seconds"]["count"] == receipt["executed_steps"]
        assert "individual_step_wall_seconds" not in receipt
        sizes[factor] = len(
            json.dumps(receipt, allow_nan=False, sort_keys=True, default=str)
        )
    # Only the small integer fields differ in width between the two runs.
    assert abs(sizes[10] - sizes[1]) < 256


def test_the_integration_refuses_before_taking_a_step(monkeypatch, tmp_path: Path) -> None:
    seen: list[Path] = []
    _fake_publisher(monkeypatch, seen=seen)
    shell = _integration_shell(_config())
    with pytest.raises(ValueError, match="stop short of the requested duration"):
        shell.integrate(
            duration_s=30.0, history_interval_s=4.0, history_root=tmp_path / "a"
        )
    assert shell.clock.completed_steps == 0 and not seen

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        shell.integrate(duration_s=5.0, history_interval_s=5.0, history_root=existing)
    assert shell.clock.completed_steps == 0 and not seen

    with pytest.raises((ValueError, OSError)):
        shell.integrate(
            duration_s=5.0,
            history_interval_s=5.0,
            history_root=tmp_path / "b",
            history_publisher=tmp_path / "absent",
        )
    assert shell.clock.completed_steps == 0 and not seen

    # An unpublishable grid is refused by the frame profile, before stepping.
    huge = _integration_shell(_config(1400, 1400, 140))
    with pytest.raises(ValueError, match="bytes"):
        huge.integrate(
            duration_s=5.0, history_interval_s=5.0, history_root=tmp_path / "c"
        )
    assert huge.clock.completed_steps == 0 and not seen

    for budget in (0, -1, 1.0):
        with pytest.raises(ValueError, match="positive integer"):
            shell.integrate(
                duration_s=5.0,
                history_interval_s=5.0,
                history_root=tmp_path / "d",
                memory_sample_budget=budget,
            )
    assert shell.clock.completed_steps == 0 and not seen


# ------------------------------------------------------------- the door


pytestmark_estate = pytest.mark.skipif(
    bool(blockers(load_config(ANALYTIC))),
    reason="the audited ArWen estate is not installed, so admission stops earlier",
)


def test_the_door_binds_the_declared_selectors_and_the_default_lane() -> None:
    analytic, declared = require_initializer_selection(load_config(ANALYTIC))
    assert declared and (analytic.record()["isnd"], analytic.record()["iwnd"]) == (5, 2)
    assert not analytic.reads_input_sounding
    default, declared = require_initializer_selection(load_config(SINGLE_SOUNDING))
    assert not declared and default is SINGLE_SOUNDING_SELECTION
    assert default.reads_input_sounding


def test_the_door_admits_the_cm1_deck_cadence_and_refuses_an_unplaceable_one() -> None:
    config = load_config(ANALYTIC)
    assert require_history_cadence(config).frame_count == 120
    # The door does not trust a configuration that reached it another way.
    unplaceable = replace(config, run=replace(config.run, history_interval_s=7.0))
    with pytest.raises(CM1Refusal) as caught:
        require_history_cadence(unplaceable)
    assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-CADENCE"
    assert caught.value.detail.option == "run.history_interval_s"


def test_history_root_must_be_new_under_an_existing_parent(tmp_path: Path) -> None:
    assert require_new_history_root(None, option="history-dir") is None
    assert require_new_history_root(tmp_path / "frames", option="history-dir") == (
        tmp_path.resolve() / "frames"
    )
    existing = tmp_path / "used"
    existing.mkdir()
    for candidate in (existing, tmp_path / "absent" / "frames"):
        with pytest.raises(CM1Refusal) as caught:
            require_new_history_root(candidate, option="history-dir")
        assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-DIRECTORY"


#: A stand-in for base.F:3143-3189's report, shaped like the record the Rust
#: transcription returns.  Only the door's handling of it is under test here;
#: the record's own arithmetic is proved in the Rust and oracle tests.
_STUB_SOUNDING_REPORT = {
    "schema": "gm1.sounding-parameters/v1",
    "storm_motion": "bunkers-right",
    "lowest_mass_level_m_agl": 250.0,
    "bunkers_right_u_m_per_s": 12.0,
    "bunkers_right_v_m_per_s": 6.0,
    "shear_0_1km_m_per_s": 11.0,
    "shear_0_6km_m_per_s": 28.0,
    "srh_0_1km_m2_per_s2": 150.0,
    "srh_0_3km_m2_per_s2": 210.0,
    **{
        parcel: {
            "cape_j_per_kg": 2500.0,
            "cin_j_per_kg": 12.0,
            "lcl_m_agl": 900.0,
            "lfc_m_agl": 1200.0,
            "el_m_agl": 12000.0,
            "source_pressure_pa": 96000.0,
            "source_temperature_k": 300.0,
            "source_qv_kg_per_kg": 0.015,
            "converged": True,
        }
        for parcel in ("surface_based", "most_unstable", "mixed_layer")
    },
}


def test_the_door_writes_the_stopped_receipt_and_then_refuses(
    monkeypatch, tmp_path: Path
) -> None:
    """CM1 keeps the tripping state's output; this door keeps its record too.

    ``run_cartesian`` writes no receipt for a run that stopped, because a
    partial trajectory must never read as a completed one.  CM1's advective
    Courant stop is the one exception it has to make: cm1.F:3517-3520 forces
    ``dostat``/``dowriteout`` so the tripping state's output survives, and the
    frames are already published by the time the stop is raised.  Discarding
    the receipt would throw away the only record naming which frame, which step
    and which model time -- so it is written, under a status of its own, and
    followed by a refusal so the exit status can never read as success.

    The device, the estate and the initializer are stood in for: what is under
    test is the door's own bookkeeping, which is pure host logic.
    """

    from cm1core.cuda_cartesian_runtime import CartesianCourantStop

    library = tmp_path / "libcm1_state.so"
    library.write_bytes(b"")
    frames = tmp_path / "frames"
    receipt_path = tmp_path / "receipt.json"
    stop_receipt = {
        "schema": INTEGRATION_SCHEMA,
        "status": INTEGRATION_COURANT_STOP_STATUS,
        "courant_stop": {
            "reason": "the advective Courant number 1.550000 has reached CM1's "
                      "own stop value 1.5 (misclibs.F:2714)",
            "frame_index": 2,
            "step": 240,
            "model_time_s": 120.0,
            "frame_directory": str(frames / "frame-000002"),
            "advective_courant": 1.55,
            "frames_requested": 120,
            "frames_published": 2,
            "authority": CM1_COURANT_STOP_AUTHORITY,
        },
        "executed_steps": 240,
    }

    monkeypatch.setattr(cartesian_door, "require_runnable", lambda config, **kwargs: None)
    monkeypatch.setattr(
        cartesian_door, "resolve_initializer_library", lambda argument: library
    )
    monkeypatch.setattr(
        cartesian_door,
        "resolve_history_publisher",
        lambda history_dir, argument=None: None,
    )
    monkeypatch.setattr(cartesian_door, "require_cuda_runtime", lambda: None)
    # base.F:3143-3189's t=0 report is stood in for the same way the device,
    # the estate and the initializer are: it reads the real Rust library, and
    # the library here is a stand-in.  What is under test is the door's
    # bookkeeping around the stop, and the assertions below check that the
    # report still reaches the stopped receipt.
    monkeypatch.setattr(
        cartesian_door, "require_sounding_report", lambda *a, **k: _STUB_SOUNDING_REPORT
    )

    def _stop(**kwargs):
        raise CartesianCourantStop("stopped", receipt=dict(stop_receipt))

    monkeypatch.setattr(cartesian_door, "_execute", _stop)

    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            load_config(ANALYTIC),
            config_label=str(ANALYTIC),
            output=receipt_path,
            history_dir=frames,
            integrate=True,
        )
    assert caught.value.detail.code == "CM1-CARTESIAN-COURANT-STOP"
    assert "misclibs.F:2714" in caught.value.detail.reason
    assert "frame-000002" in caught.value.detail.remedy
    assert str(receipt_path) in caught.value.detail.remedy

    written = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written["status"] == "stopped-at-cm1-advective-courant-stop"
    assert written["courant_stop"]["step"] == 240
    assert written["courant_stop"]["model_time_s"] == 120.0
    assert written["runtime_benchmark"]["status"] == INTEGRATION_COURANT_STOP_STATUS
    assert "nothing in this record is a completed run" in written["claim_limit"]
    # base.F:3143-3189.  A run abandoned at the Courant stop still has to say
    # what atmosphere it was integrating; without it the only surviving record
    # of the failure names a frame and a step and nothing about the air.
    assert written["initial_sounding_parameters"] == _STUB_SOUNDING_REPORT


@pytestmark_estate
def test_an_inadmissible_run_refuses_before_the_cuda_import(tmp_path: Path) -> None:
    """The refusal must be the cadence or the directory, never the device.

    Reaching require_cuda_runtime would mean the admission order had drifted:
    on a machine with a GPU that call opens a CUDA context, and on one without
    it the refusal code would be CM1-CARTESIAN-CUPY-MISSING, which is the
    marker this asserts against.
    """

    config = load_config(ANALYTIC)
    unplaceable = replace(config, run=replace(config.run, history_interval_s=7.0))
    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            unplaceable,
            config_label=str(ANALYTIC),
            output=tmp_path / "receipt.json",
            history_dir=tmp_path / "frames",
            integrate=True,
        )
    assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-CADENCE"

    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            config,
            config_label=str(ANALYTIC),
            output=tmp_path / "receipt.json",
            integrate=True,
        )
    assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-DIRECTORY"
    assert caught.value.detail.option == "history-dir"

    used = tmp_path / "used"
    used.mkdir()
    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            config,
            config_label=str(ANALYTIC),
            output=tmp_path / "receipt.json",
            history_dir=used,
            integrate=True,
        )
    assert caught.value.detail.code == "CM1-CARTESIAN-HISTORY-DIRECTORY"
    assert not (tmp_path / "receipt.json").exists()


@pytestmark_estate
def test_an_inadmissible_integration_never_touches_the_cuda_import(tmp_path: Path) -> None:
    """Proved in a child process whose cupy raises the moment it is imported."""

    fake = tmp_path / "fake-site" / "cupy"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(
        "raise AssertionError('cupy imported before admission')", encoding="utf-8"
    )
    used = tmp_path / "used"
    used.mkdir()
    command = (
        "import sys, json; from cm1core.cli import main; "
        "status = main(['cartesian', sys.argv[1], '--output', sys.argv[2], "
        "'--history-dir', sys.argv[3], '--integrate', '--json']); "
        "print(json.dumps({'status': status, 'cupy': any("
        "name == 'cupy' or name.startswith('cupy.') for name in sys.modules)}))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (
            str(tmp_path / "fake-site"),
            str(ROOT / "src"),
            os.environ.get("PYTHONPATH", ""),
        ) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", command, str(ANALYTIC),
         str(tmp_path / "receipt.json"), str(used)],
        cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "status": 2, "cupy": False,
    }
    assert json.loads(completed.stderr)["code"] == "CM1-CARTESIAN-HISTORY-DIRECTORY"
    assert not (tmp_path / "receipt.json").exists()


@pytestmark_estate
def test_the_analytic_lane_refuses_an_input_sounding_it_would_never_read(
    tmp_path: Path, monkeypatch
) -> None:
    library = tmp_path / "libcm1_state.so"
    library.write_bytes(b"not executed by this test")
    profile = tmp_path / "input_sounding"
    profile.write_text("  1000.0  300.5  5.0\n  100.0  300.5  5.0  0.0  0.0\n", encoding="utf-8")
    with pytest.raises(CM1Refusal) as caught:
        run_cartesian(
            load_config(ANALYTIC),
            config_label=str(ANALYTIC),
            output=tmp_path / "receipt.json",
            library=library,
            sounding=profile,
            history_dir=tmp_path / "frames",
            integrate=True,
        )
    assert caught.value.detail.code == "CM1-CARTESIAN-SOUNDING-UNREAD"
    assert caught.value.detail.option == "sounding"
    assert not (tmp_path / "receipt.json").exists()


@pytestmark_estate
def test_plan_prices_the_analytic_lane_without_opening_a_device(
    tmp_path: Path, capsys
) -> None:
    from cm1core.cli import main

    receipt = tmp_path / "receipt.json"
    status = main([
        "cartesian", str(ANALYTIC), "--output", str(receipt),
        "--history-dir", str(tmp_path / "frames"), "--integrate", "--plan",
    ])
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "priced-no-device-opened"
    # d0f0429 widened InitializerSelection.record() to carry iinit and every
    # option-struct word; the deck declares only isnd/iwnd, so the rest are
    # the defaults of that selection.
    assert payload["initialization"] == json.loads(
        json.dumps(InitializerSelection(isnd=5, iwnd=2).record())
    )
    assert (payload["initialization"]["isnd"], payload["initialization"]["iwnd"]) == (5, 2)
    assert payload["initialization_declared"] is True
    assert payload["integration_cadence"]["frame_count"] == 120
    assert payload["integration_cadence"]["total_steps"] == 14400
    assert payload["measured_steps"] == 14400
    assert not receipt.exists()
    assert not (tmp_path / "frames").exists()


def test_the_installed_command_carries_the_duration_interval_and_lane() -> None:
    from cm1core.cli import build_parser

    arguments = build_parser().parse_args([
        "cartesian", str(ANALYTIC), "--output", "receipt.json",
        "--history-dir", "frames", "--integrate",
    ])
    assert arguments.handler is cartesian_door.cartesian_door
    assert arguments.integrate is True
    assert arguments.history_dir == Path("frames")
    config = load_config(arguments.config if hasattr(arguments, "config") else ANALYTIC)
    assert config.run.lane is Lane.CARTESIAN
    assert (config.run.duration_s, config.run.history_interval_s) == (7200.0, 60.0)
    assert (config.initialization.isnd, config.initialization.iwnd) == (5, 2)


def test_the_integration_measures_cm1s_courant_number_before_its_first_step(
    monkeypatch, tmp_path: Path
) -> None:
    """``misclibs.F:2714``: measure, then stop rather than integrate.

    With ``run.dt_s`` a free parameter this is the check that catches a step
    too large for the declared grid, at the one moment it costs nothing -- and
    before the history root is created, so the corrected retry can use the same
    ``--history-root``.
    """

    shell = _integration_shell(_config())
    _fake_publisher(monkeypatch)
    receipt = shell.integrate(
        duration_s=10.0, history_interval_s=5.0, history_root=tmp_path / "frames"
    )
    assert shell.courant_checks[0] == (
        "before the first large step of this integration"
    )
    assert receipt["initial_advective_courant"] == 0.25

    blown = _integration_shell(_config())
    blown.courant_number_value = 1.5
    with pytest.raises(RuntimeError) as caught:
        blown.integrate(
            duration_s=10.0,
            history_interval_s=5.0,
            history_root=tmp_path / "refused",
        )
    message = str(caught.value)
    assert "before the first large step" in message
    assert "misclibs.F:2714" in message
    # Nothing was stepped: the refusal costs no GPU time at all.
    assert blown.clock.completed_steps == 0


TIMESTEP_KEYWORDS = frozenset({"dt", "dt_s", "dt_seconds", "dts", "dtl"})


def _cartesian_path_modules() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "src/cm1core"
    return sorted(root.glob("cuda_cartesian_*.py")) + [root / "cartesian_door.py"]


def test_a_courant_refusal_leaves_no_history_root_for_the_retry_to_trip_on(
    monkeypatch, tmp_path: Path
) -> None:
    """The refusal must not create the directory the corrected re-run needs.

    ``require_new_history_directory`` (cartesian_history.py:274-282) admits
    only a root that does not exist.  A Courant refusal raised after
    ``root.mkdir`` therefore left an empty directory behind and the operator's
    next attempt -- with the corrected ``run.dt_s`` and the same
    ``--history-root`` -- was refused for a reason that had nothing to do with
    the model.  Nothing is created, and the same path then works.
    """

    _fake_publisher(monkeypatch)
    root = tmp_path / "frames"

    blown = _integration_shell(_config())
    blown.courant_number_value = 1.6
    with pytest.raises(RuntimeError):
        blown.integrate(
            duration_s=10.0, history_interval_s=5.0, history_root=root
        )
    assert not root.exists(), "the refusal left a history root behind"

    # The same path, with a state the run may continue from.
    good = _integration_shell(_config())
    receipt = good.integrate(
        duration_s=10.0, history_interval_s=5.0, history_root=root
    )
    assert receipt["status"] == INTEGRATION_STATUS
    assert sorted(p.name for p in root.iterdir()) == ["frame-000001", "frame-000002"]


def test_a_refusal_after_the_root_exists_still_leaves_nothing_behind(
    monkeypatch, tmp_path: Path
) -> None:
    """The general case the Courant defect was one instance of.

    Anything that raises between ``root.mkdir`` and the first published frame
    -- a non-finite field scan, a failed step, a cancelled run -- would leave
    the empty root that ``require_new_history_directory`` then refuses.
    ``cartesian_history.py:428-437`` already removes an empty *frame*
    directory on refusal for exactly this reason; the root follows it.
    """

    _fake_publisher(monkeypatch)
    root = tmp_path / "frames"
    shell = _integration_shell(_config())

    def blown_checks() -> dict[str, object]:
        raise RuntimeError("completed model state contains nonfinite physical words")

    shell.field_checks = blown_checks
    with pytest.raises(RuntimeError, match="nonfinite"):
        shell.integrate(
            duration_s=10.0, history_interval_s=5.0, history_root=root
        )
    # The steps ran, so this is past the mkdir: the cleanup is what removed it.
    assert shell.clock.completed_steps > 0
    assert not root.exists()

    good = _integration_shell(_config())
    good.integrate(duration_s=10.0, history_interval_s=5.0, history_root=root)
    assert sorted(p.name for p in root.iterdir()) == ["frame-000001", "frame-000002"]


def test_the_frame_that_reaches_the_stop_is_published_before_the_run_ends(
    monkeypatch, tmp_path: Path
) -> None:
    """``cm1.F:3517-3520`` then ``cm1.F:4385-4392``: write, then stop.

    CM1 sets ``dostat`` and ``dowriteout`` when ``stopit`` is set precisely so
    the statistics and the history output for the tripping state survive, and
    only then abandons the run.  Discarding that frame would throw away the one
    a user needs to see what the wind was doing, and the receipt naming the
    step and the model time with it.
    """

    _fake_publisher(monkeypatch)
    root = tmp_path / "frames"
    shell = _integration_shell(_config())
    # pre-step, frame 1, frame 2 -- the second frame reaches CM1's stop.
    shell.courant_sequence = [0.25, 0.4, 1.55]
    with pytest.raises(CartesianCourantStop) as caught:
        shell.integrate(
            duration_s=30.0, history_interval_s=10.0, history_root=root
        )

    published = sorted(p.name for p in root.iterdir())
    assert published == ["frame-000001", "frame-000002"], published
    receipt = caught.value.receipt
    assert receipt["status"] == INTEGRATION_COURANT_STOP_STATUS
    stop = receipt["courant_stop"]
    assert stop["frame_index"] == 2
    assert stop["frames_published"] == 2 and stop["frames_requested"] == 3
    assert stop["advective_courant"] == 1.55
    assert stop["frame_directory"].endswith("frame-000002")
    assert "misclibs.F:2714" in stop["reason"]
    # The record still names the step count and the model time it stopped at.
    assert receipt["executed_steps"] == stop["step"]
    assert receipt["model_seconds_at_end"] == stop["model_time_s"] == 20.0
    assert receipt["frames"][-1]["courant_stop_reason"] == stop["reason"]
    # And it never reads as a completed run.
    assert receipt["status"] != INTEGRATION_STATUS
    assert "no part of this record is a completed run" in receipt["claim_limit"]


def test_a_run_that_stays_below_the_stop_records_the_number_at_every_frame(
    monkeypatch, tmp_path: Path
) -> None:
    _fake_publisher(monkeypatch)
    shell = _integration_shell(_config())
    shell.courant_sequence = [0.25, 0.4, 1.2, 1.4]
    receipt = shell.integrate(
        duration_s=30.0, history_interval_s=10.0, history_root=tmp_path / "frames"
    )
    assert receipt["status"] == INTEGRATION_STATUS
    assert receipt["courant_stop"] is None
    assert [row["advective_courant"] for row in receipt["frames"]] == [0.4, 1.2, 1.4]
    assert all(row["courant_stop_reason"] is None for row in receipt["frames"])
    # misclibs.F:617 is a target for an adaptive step, not a stop; a run above
    # it is reported and continues, exactly as CM1's does.
    assert receipt["field_checks"]["advective_courant"]["above_adaptive_target"]


def test_no_cartesian_stage_supplies_a_timestep_of_its_own() -> None:
    """The named defect this branch exists to prevent, guarded structurally.

    ``solve2.F:145`` carries one ``dt`` through the whole large step.  If any
    stage of the configurable Cartesian path passed anything but the declared
    step, that stage would keep advancing by the old half second while every
    other stage used the declared one -- a wrong answer that looks like a right
    one, because nothing would refuse and nothing would be non-finite.

    A literal is only the spelling a careless author would use.  A named
    constant (``dt_seconds=CM1_DEFAULT_DT_S``), a cast literal
    (``dt=cp.float32(0.5)``) and any other expression that is not a read of
    this run's configuration are the same defect, so the rule here is positive:
    every timestep argument on this path must *contain* a
    ``.dt_s`` attribute read and nothing else may stand in for it.
    """

    import ast

    # The one exemption, and it is the opposite of the defect: this is the
    # sole writer of ``config.dt_s``.  misclibs.F:785 changes the large step
    # between steps under adapt_dt, and this function is where the new value
    # is bound into the configuration every other stage then reads.  Its
    # ``dt_s=`` arguments carry that new step into ``dtsmall`` and into
    # ``dataclasses.replace``; a stage that read ``.dt_s`` here would read the
    # step being replaced.
    writers = {"rebind_large_step"}

    offenders: list[str] = []
    for path in _cartesian_path_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        exempt: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in writers:
                exempt.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if node.lineno in exempt:
                continue
            for keyword in node.keywords:
                if keyword.arg not in TIMESTEP_KEYWORDS:
                    continue
                value = keyword.value
                reads = {
                    inner.attr
                    for inner in ast.walk(value)
                    if isinstance(inner, ast.Attribute)
                }
                if "dt_s" not in reads:
                    offenders.append(
                        f"{path.name}:{value.lineno} "
                        f"{keyword.arg}={ast.unparse(value)}"
                    )
    assert offenders == [], (
        "a Cartesian stage supplies its own timestep instead of the "
        f"configured one: {offenders}"
    )


def test_no_timestep_argument_on_this_path_can_be_omitted() -> None:
    """The four spellings an AST keyword scan cannot see, closed by signature.

    A guard that only rejects a literal keyword misses an omitted keyword, a
    named constant, a positional argument and a cast inside a launch tuple.
    The omission is the dangerous one, because a defaulted parameter turns it
    into a silent half second rather than an error -- and that pattern was live
    in this tree: ``cuda_gm1_tilestream_rk3_interposed_prefix`` called the
    eqtset2 removal with no ``dt_seconds`` at all.

    So no function in ``src/cm1core`` gives a timestep parameter a default: an
    omission is a ``TypeError`` at the call, which the language enforces rather
    than a test.  Every stage launcher the Cartesian large step drives is then
    checked to take its step keyword-only as well, so a shifted argument list
    cannot slide something else into that slot; the one-argument helpers that
    take ``dt`` positionally on purpose are unambiguous and are left alone.
    """

    import ast

    root = Path(__file__).resolve().parents[1] / "src/cm1core"
    defaulted: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            plain = args.posonlyargs + args.args
            pad = [None] * (len(plain) - len(args.defaults))
            for argument, default in zip(plain, pad + list(args.defaults)):
                if argument.arg not in TIMESTEP_KEYWORDS:
                    continue
                if default is not None:
                    defaulted.append(
                        f"{path.name}:{node.lineno} {node.name}"
                        f"({argument.arg}={ast.unparse(default)})"
                    )
            for argument, default in zip(args.kwonlyargs, args.kw_defaults):
                if argument.arg in TIMESTEP_KEYWORDS and default is not None:
                    defaulted.append(
                        f"{path.name}:{node.lineno} {node.name}"
                        f"({argument.arg}={ast.unparse(default)})"
                    )
    assert defaulted == [], (
        "a timestep parameter has a default, so omitting it applies that step "
        f"silently instead of raising: {defaulted}"
    )

    # The other spelling the keyword scan cannot see is a positional argument.
    # A one-argument helper such as ``rk_stage_dt_f32(dt, nrk)`` takes it
    # positionally on purpose and is unambiguous; what must never be positional
    # is a stage launcher with a long argument list, where one shifted argument
    # would silently move the step into another slot.  Each one the Cartesian
    # large step drives is named here.
    import importlib
    import inspect

    launchers = (
        ("cuda_gm1_tilestream_rk3_eqtset2_remove_eos",
         "launch_cuda_gm1_rk3_eqtset2_remove_eos_local", "dt_seconds"),
        ("cuda_gm1_tilestream_rk3_eqtset2_remove_eos",
         "cpu_gm1_rk3_eqtset2_remove_eos_inplace", "dt_seconds"),
        ("cuda_gm1_tilestream_rk3_interposed_prefix",
         "launch_cuda_gm1_rk3_interposed_prefix", "dt_seconds"),
        ("cuda_psolver3_terrain_open",
         "launch_cuda_psolver3_terrain_open_stage", "dt"),
        ("cuda_scalar_moist_rk", "launch_cuda_scalar_moist_rk_stage", "dt"),
        ("cuda_gm1_child_stretched_sgs_tke",
         "launch_cuda_gm1_child_stretched_sgs_stress", "dt"),
    )
    for module_name, function_name, parameter_name in launchers:
        module = importlib.import_module(f"cm1core.{module_name}")
        parameter = inspect.signature(
            getattr(module, function_name)
        ).parameters[parameter_name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function_name
        assert parameter.default is inspect.Parameter.empty, function_name


def test_the_eqtset2_removal_and_its_oracle_require_the_step() -> None:
    """Exercised, not only parsed: omitting the step must raise.

    ``dataclasses`` and ``inspect`` prove the signature; calling the CPU oracle
    without a step proves the signature is the one Python enforces.
    """

    import inspect

    from cm1core.cuda_gm1_tilestream_rk3_eqtset2_remove_eos import (
        cpu_gm1_rk3_eqtset2_remove_eos_inplace,
        launch_cuda_gm1_rk3_eqtset2_remove_eos_local,
    )
    from cm1core.cuda_gm1_tilestream_rk3_interposed_prefix import (
        launch_cuda_gm1_rk3_interposed_prefix,
    )

    for function in (
        cpu_gm1_rk3_eqtset2_remove_eos_inplace,
        launch_cuda_gm1_rk3_eqtset2_remove_eos_local,
        launch_cuda_gm1_rk3_interposed_prefix,
    ):
        parameter = inspect.signature(function).parameters["dt_seconds"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function
        assert parameter.default is inspect.Parameter.empty, function

    with pytest.raises(TypeError, match="dt_seconds"):
        cpu_gm1_rk3_eqtset2_remove_eos_inplace({})


def test_the_final_rk_removal_receives_the_configured_step() -> None:
    """``cuda_cartesian_finish`` must hand the eqtset2 removal ``config.dt_s``.

    The kernel at ``solve2.F:1569-1576`` is the one stage that used to weld
    the half second into its PTX; the call site is checked by name so a future
    edit cannot quietly reintroduce a constant there.
    """

    import ast

    path = Path(__file__).resolve().parents[1] / "src/cm1core/cuda_cartesian_finish.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "dt_seconds":
                found.append(ast.dump(keyword.value))
    assert found, "the final-RK removal call no longer passes dt_seconds"
    for dumped in found:
        assert "attr='dt_s'" in dumped, dumped
