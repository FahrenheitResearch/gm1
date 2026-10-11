"""The swaths cross a restart: restart_write.F:696-758 and restart_read.F:1064-1126.

CM1 writes ``sws``..``shs`` into every restart file, and under ``nrain = 2``
(a moving domain) the translated ``sws2``..``shs2``; ``restart_read.F`` reads
them straight back, so a paused-and-resumed run publishes the same swaths as
one that never stopped.  GM1's parent restart did not carry them (before
restart profile v2): after a resume every swath counted from the resume, and the
ground canvas was sized for the resumed run plus the steps it had already
taken.

What is asserted here is raw words.  The pure half runs the shipped
``SurfaceSwaths`` fold with NumPy in CuPy's place, through the shipped
restart words and read-back; the envelope half drives the shipped
integration loop, the real ``cm1-forecast-publish`` checkpoint and restore,
and the door, against the host double of ``test_cartesian_restart``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cm1core.cartesian_door import build_runtime_config, require_run_schedule, resume_selection
from cm1core.cartesian_restart import (
    CARTESIAN_RESTART_PROFILE_ID,
    CARTESIAN_RESTART_PROFILE_ID_V1,
    RestoredCartesianRestart,
    _swath_restart_words,
    cartesian_restart_profile,
    read_cartesian_restart,
    read_restored_swaths,
    require_restart_receipt,
    state_profile_sha256,
    write_cartesian_restart,
)
from cm1core.cartesian_tornado_diagnostics import (
    CartesianTornadoDiagnostics,
    swath_restart_layout,
)
from cm1core.config import load_config
from cm1core.cuda_cartesian_runtime import FRAME_DIRECTORY_FORMAT
from cm1core.tornado_diagnostics import (
    SWATH_INITIAL_VALUES,
    SWATH_NAMES,
    SWATH_RESTART_GROUP,
    GroundAnchor,
    LowerBoundaryState,
    SurfaceSwaths,
    SwathRestartLayout,
    TornadoDiagnosticSelection,
    binary32_chunks_as_binary64,
    binary64_as_binary32_chunks,
    tornado_report,
)

from test_cartesian_restart import (  # noqa: E402 - the shared host double
    _TOML,
    CpuCartesianRuntime,
    CpuStorage,
    door,  # noqa: F401 - the door fixture
    forecast_publisher,  # noqa: F401 - the session fixture
)
from test_tornado_diagnostics import _Grid, _column, _host_runtime  # noqa: E402


# ---------------------------------------------------------------- the pure half

NX, NY, NZ, D = 12, 10, 24, 250.0
# 0.3 cells east and 0.1 cells south per step: the ground canvas grows by
# whole cells at different steps in x and y, the placement residual is never
# zero, and the canvas origin moves south as the run gets longer.
UMOVE, VMOVE, DT = 37.5, -12.5, 2.0


def _fields(step: int) -> dict:
    rng = np.random.default_rng(1000 + step)
    return {
        "u": (rng.standard_normal((NZ, NY, NX + 1)) * 30.0).astype(np.float32),
        "v": (rng.standard_normal((NZ, NY + 1, NX)) * 30.0).astype(np.float32),
        "w": (rng.standard_normal((NZ + 1, NY, NX)) * 12.0).astype(np.float32),
        "prs": (100000.0 + rng.standard_normal((NZ, NY, NX)) * 300.0).astype(np.float32),
        "prs0_column": np.full((NY, NX), np.float32(100000.0), dtype=np.float32),
        "q3d": np.abs(rng.standard_normal((6, NZ, NY, NX)) * 1e-3).astype(np.float32),
        "updraft_helicity_plane": (rng.standard_normal((NY, NX)) * 120.0).astype(np.float32),
    }


def _open(total: int, *, opened: int = 0, names=SWATH_NAMES) -> SurfaceSwaths:
    anchor = GroundAnchor(
        dx_m=D, dy_m=D, u_move_m_s=UMOVE, v_move_m_s=VMOVE, dt_s=DT, total_steps=total,
    )
    selection = TornadoDiagnosticSelection(**{f"output_{name}": 1 for name in names})
    return SurfaceSwaths(
        np, grid=_Grid(NX, NY, NZ, D), selection=selection, anchor=anchor,
        column=_column(NZ, D), opened_step=opened,
    )


def _fold(swaths: SurfaceSwaths, first: int, last: int) -> SurfaceSwaths:
    for step in range(first, last + 1):
        swaths.update(
            step=step, **_fields(step),
            domainlocx_m=step * DT * UMOVE, domainlocy_m=step * DT * VMOVE,
        )
    return swaths


class _Shell:
    """A runtime whose only open diagnostic is one SurfaceSwaths."""

    def __init__(self, swaths: SurfaceSwaths) -> None:
        self.tornado = SimpleNamespace(
            restart_layout=swaths.restart_layout, restart_words=swaths.restart_words,
        )


def _through_restart(swaths: SurfaceSwaths, tmp_path: Path, name: str = "arena"):
    """The shipped restart words, as float32 words at offsets, read back the shipped way."""

    layout = swaths.restart_layout()
    arrays, scalars, _ = _swath_restart_words(_Shell(swaths), layout, step=swaths.last_step)
    path = tmp_path / f"{name}.f32"
    offsets, shapes, blob = {}, {}, bytearray()
    for carrier, *_, shape, _key in layout.carriers():
        words = np.ascontiguousarray(arrays[carrier], dtype="<f4")
        assert words.shape == tuple(shape)
        assert np.all(np.isfinite(words)), carrier  # the envelope admits finite words only
        offsets[carrier], shapes[carrier] = len(blob), tuple(shape)
        blob += words.tobytes()
    path.write_bytes(bytes(blob))
    restored = RestoredCartesianRestart(
        receipt={"profile": {"profile_id": CARTESIAN_RESTART_PROFILE_ID, "swaths": layout.record()},
                 "restart_number": 1},
        restore_receipt={}, directory=tmp_path, arena_path=path,
        offsets=MappingProxyType(offsets), shapes=MappingProxyType(shapes),
        scalars=MappingProxyType(scalars), step=swaths.last_step,
        model_time_s=swaths.last_step * DT,
    )
    return read_restored_swaths(restored)


def _report(swaths: SurfaceSwaths) -> dict:
    row = tornado_report(
        swaths=swaths, grid=_Grid(NX, NY, NZ, D), column=_column(NZ, D),
        lower=LowerBoundaryState(bbc=1),
    )
    row.pop("swaths")
    return row


def _assert_same_swaths(resumed: SurfaceSwaths, whole: SurfaceSwaths) -> None:
    assert resumed.names == whole.names
    for name in whole.names:
        assert resumed.frame[name].tobytes() == whole.frame[name].tobytes(), name
    assert set(resumed.ground) == set(whole.ground)
    for name, canvas in whole.ground.items():
        assert resumed.ground[name].shape == canvas.shape, name
        assert resumed.ground[name].tobytes() == canvas.tobytes(), f"{name}2"
    for name, series in whole.series.items():
        assert resumed.series[name].tobytes() == series.tobytes(), name
    assert resumed.vortex_index.tobytes() == whole.vortex_index.tobytes()
    assert resumed.domain_location == whole.domain_location
    assert resumed.max_residual_m == whole.max_residual_m
    assert resumed.host_series() == pytest.approx(whole.host_series(), nan_ok=True)
    left, right = resumed.record(), whole.record()
    for row in (left, right):
        row.pop("updates_this_segment")
    assert left == right
    assert _report(resumed) == _report(whole)


def test_binary64_travels_as_exact_binary32_chunks() -> None:
    values = np.array([
        0.0, -0.0, 5e-324, 2.2250738585072014e-308, 1.5e300, -7.25, math.pi,
        12345.678901234567, -4.0e7, 18750.0,
    ])
    chunks = binary64_as_binary32_chunks(values)
    assert chunks.dtype == np.float32 and chunks.shape == (values.size, 3)
    assert np.all(np.isfinite(chunks))
    assert binary32_chunks_as_binary64(chunks).tobytes() == values.tobytes()
    bad = chunks.copy()
    bad[3, 1] = np.float32(0.5)
    with pytest.raises(ValueError, match="22-bit range"):
        binary32_chunks_as_binary64(bad)


def test_a_resume_publishes_the_uninterrupted_swaths_word_for_word(tmp_path) -> None:
    """Pause at step 12 of 30, resume, and every swath word is the whole run's."""

    whole = _fold(_open(30), 1, 30)
    assert whole.ground and whole.max_residual_m != (0.0, 0.0)
    stopped = _fold(_open(30), 1, 12)
    state = _through_restart(stopped, tmp_path)
    resumed = _open(30, opened=12)
    record = resumed.continue_from(state)
    assert record["continued"] == list(SWATH_NAMES)
    assert record["restart_here"] == {} and record["not_continued"] == []
    _fold(resumed, 13, 30)
    _assert_same_swaths(resumed, whole)
    assert resumed.record()["whole_run"] is True
    assert resumed.record()["updates"] == 30 and resumed.updates == 18


@pytest.mark.parametrize("planned, resumed_to", [(30, 40), (40, 20)])
def test_the_ground_canvas_is_replaced_by_its_origin_when_the_run_length_changes(
    tmp_path, planned: int, resumed_to: int,
) -> None:
    """Continue and extend, or continue and cut short: the canvas is placed, not copied.

    The canvas is sized from the run's length, so the restart of a run
    planned for ``planned`` steps holds a canvas of another extent and origin
    than the one a run of ``resumed_to`` steps opens.  The resumed swath must
    still be the one an uninterrupted run of ``resumed_to`` steps publishes.
    """

    whole = _fold(_open(resumed_to), 1, resumed_to)
    stopped = _fold(_open(planned), 1, 12)
    state = _through_restart(stopped, tmp_path)
    assert state.layout.canvas_shape != whole.anchor.canvas_shape(NY, NX)
    assert state.layout.canvas_origin_cells_xy != whole.anchor.origin_cells()
    resumed = _open(resumed_to, opened=12)
    record = resumed.continue_from(state)
    assert record["ground_canvas"]["restart_origin_cells_xy"] == list(state.layout.canvas_origin_cells_xy)
    _fold(resumed, 13, resumed_to)
    _assert_same_swaths(resumed, whole)


@pytest.mark.parametrize("umove, vmove", [(-37.5, 12.5), (-50.0, -40.0), (0.0, 30.0), (0.0, 0.0)])
@pytest.mark.parametrize("planned, resumed_to", [(30, 45), (45, 20)])
def test_every_frame_direction_resumes_word_for_word(
    tmp_path, monkeypatch, umove: float, vmove: float, planned: int, resumed_to: int,
) -> None:
    """West, north, both negative, one axis only, and a stationary domain.

    A westward or southward frame puts the canvas origin below zero and moves
    it as the run gets longer; a stationary one keeps no canvas at all.
    """

    import sys

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "UMOVE", umove)
    monkeypatch.setattr(module, "VMOVE", vmove)
    whole = _fold(_open(resumed_to), 1, resumed_to)
    stopped = _fold(_open(planned), 1, 12)
    state = _through_restart(stopped, tmp_path)
    assert (state.layout.canvas_shape is None) == (umove == 0.0 and vmove == 0.0)
    resumed = _open(resumed_to, opened=12)
    resumed.continue_from(state)
    _fold(resumed, 13, resumed_to)
    _assert_same_swaths(resumed, whole)


def test_a_swath_the_restart_does_not_carry_starts_at_the_resume_and_says_so(tmp_path) -> None:
    stopped = _fold(_open(30, names=("sws", "svs")), 1, 12)
    state = _through_restart(stopped, tmp_path)
    resumed = _open(30, opened=12, names=("sws", "sps"))
    record = resumed.continue_from(state)
    assert record["continued"] == ["sws"]
    assert record["restart_here"] == {"sps": 13}
    assert record["not_continued"] == ["svs"]
    _fold(resumed, 13, 30)
    whole = _fold(_open(30, names=("sws", "sps")), 1, 30)
    fresh = _fold(_open(30, opened=12, names=("sws", "sps")), 13, 30)
    assert resumed.frame["sws"].tobytes() == whole.frame["sws"].tobytes()
    assert resumed.frame["sps"].tobytes() == fresh.frame["sps"].tobytes()
    assert resumed.frame["sps"].tobytes() != whole.frame["sps"].tobytes()
    row = resumed.record()
    assert row["first_step"] == {"sws": 1, "sps": 13}
    assert row["whole_run"] is False
    # The series of the swath that restarted is NaN before it did, not 0.
    series = resumed.host_series()["sps_min"]
    assert all(math.isnan(value) for value in series[:12])
    assert series[12:] == whole.host_series()["sps_min"][12:]


def test_a_restart_is_refused_where_continuing_it_would_misplace_a_word(tmp_path) -> None:
    stopped = _fold(_open(40), 1, 12)
    state = _through_restart(stopped, tmp_path)
    with pytest.raises(ValueError, match="stand at step 12 and these open at step 11"):
        _open(40, opened=11).continue_from(state)
    # A folded word on ground a shorter run's canvas does not cover cannot
    # have come from the first 12 steps; it is refused, not dropped.
    state.canvases["sws"][0, -1] = np.float32(55.0)
    with pytest.raises(ValueError, match="does not cover"):
        _open(20, opened=12).continue_from(state)
    with pytest.raises(ValueError, match="folds every large step"):
        _fold(_open(40), 1, 3).update(
            step=5, **_fields(5), domainlocx_m=0.0, domainlocy_m=0.0,
        )


# ------------------------------------------------------ the profile and layout


def _supercell_config(tmp_path: Path, *, total_steps: int = 40):
    from test_tornado_diagnostics import _BASE_DOCUMENT, _DIAGNOSTICS_BLOCK, _document

    config = load_config(_document(tmp_path, "on.toml", _BASE_DOCUMENT + _DIAGNOSTICS_BLOCK))
    return build_runtime_config(config)


def test_the_v2_profile_carries_the_swaths_and_its_state_is_the_v1_profile(tmp_path) -> None:
    runtime_config = _supercell_config(tmp_path)
    layout = swath_restart_layout(runtime_config, total_steps=40)
    assert layout is not None and layout.names == SWATH_NAMES
    assert layout.canvas_shape is not None  # the document's frame moves
    v1 = cartesian_restart_profile(runtime_config, profile_id=CARTESIAN_RESTART_PROFILE_ID_V1)
    bare = cartesian_restart_profile(runtime_config)
    carried = cartesian_restart_profile(runtime_config, swaths=layout)
    assert carried.profile_id == CARTESIAN_RESTART_PROFILE_ID == "gm1.cartesian-restart/v2"
    assert "swaths" not in v1.record()
    assert carried.record()["swaths"] == layout.record()
    assert bare.record()["swaths"] is None
    # The state part of both v2 profiles is the v1 profile, byte for byte:
    # that is what lets a v1 restart resume, and a restart of a run planned
    # to another length resume at all.
    assert carried.state_sha256 == bare.state_sha256 == v1.sha256 == v1.state_sha256
    assert state_profile_sha256(carried.record()) == v1.sha256
    longer = cartesian_restart_profile(
        runtime_config, swaths=swath_restart_layout(runtime_config, total_steps=80),
    )
    assert longer.sha256 != carried.sha256 and longer.state_sha256 == carried.state_sha256
    swath_rows = [c for c in carried.carriers if c.group == SWATH_RESTART_GROUP]
    names = [c.name for c in swath_rows]
    for name in SWATH_NAMES:
        assert f"swath.{name}" in names and f"swath.{name}2" in names
    assert carried.state_only().carriers == bare.carriers
    assert carried.payload_nbytes == bare.payload_nbytes + sum(c.nbytes for c in swath_rows)


def test_the_priced_layout_is_the_live_one(tmp_path) -> None:
    runtime_config = _supercell_config(tmp_path)
    host = _host_runtime(runtime_config)
    tornado = CartesianTornadoDiagnostics(host, total_steps=40)
    assert tornado.restart_layout() == swath_restart_layout(runtime_config, total_steps=40)


# ------------------------------------------------ through the real envelope

_DIAGNOSTICS = """
[diagnostics]
output_uh = 0
output_vort = 0
output_pv = 0
output_sws = 1
output_svs = 1
output_sps = 0
output_srs = 0
output_sgs = 0
output_sus = 0
output_shs = 0
tornado_report = false
"""


def _swath_document(tmp_path: Path, *, duration: float, name: str, selectors: str = "",
                    swaths: bool = True) -> Path:
    text = _TOML.format(
        duration=duration, history=5.0, restart=10.0,
        selectors=selectors + ("\n" if selectors else ""),
    )
    # A faster frame than the base document's, east and south, so the ground
    # canvas grows by whole cells and its origin moves with the run's length.
    text = text.replace("u_move_m_s = 12.5", "u_move_m_s = 50.0")
    text = text.replace("v_move_m_s = 3.0", "v_move_m_s = -25.0")
    assert "u_move_m_s = 50.0" in text and "v_move_m_s = -25.0" in text
    path = tmp_path / name
    path.write_text(text + (_DIAGNOSTICS if swaths else ""), encoding="utf-8")
    return path


class _NumpyStorage(CpuStorage):
    """The host double's array module, with the rest of NumPy the swath fold uses."""

    def __getattr__(self, name):
        return getattr(np, name)


class SwathCpuRuntime(CpuCartesianRuntime):
    """The shared host double, folding the swaths the way the device runtime does."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.cp = _NumpyStorage(self.stream)

    def step(self) -> None:
        super().step()
        # CudaCartesianRuntime.step: solve3.F:855-1136 at the end of every step.
        if self.tornado is not None:
            self.tornado.update(int(self.clock.completed_steps))


def _integrate(runtime, config, tmp_path: Path, tag: str, publisher, **kwargs) -> dict:
    return runtime.integrate(
        duration_s=config.run.duration_s,
        history_interval_s=config.run.history_interval_s,
        history_root=tmp_path / f"frames-{tag}",
        history_publisher=publisher,
        restart_root=tmp_path / f"restarts-{tag}",
        restart_interval_s=config.run.restart_interval_s,
        restart_publisher=publisher,
        restart_reset_frqtim=config.run.reset_frqtim,
        **kwargs,
    )


def _frame_words(root: Path, index: int) -> dict[str, bytes]:
    frame = root / FRAME_DIRECTORY_FORMAT.format(index=index)
    arena = (frame / "history-carriers.f32").read_bytes()
    words = {}
    for line in (frame / "history-field-table.tsv").read_text().splitlines():
        columns = line.split("\t")
        count = int(np.prod([int(v) for v in columns[4].split(",")]))
        offset = int(columns[7])
        words[columns[0]] = arena[offset:offset + 4 * count]
    return words


def _resume(tmp_path, publisher, *, restarts: Path, number: int, duration: float, tag: str):
    document = _swath_document(
        tmp_path, duration=duration, name=f"{tag}.toml",
        selectors=f"irst = 1\nrstnum = {number}",
    )
    config = load_config(document)
    selected, receipt = resume_selection(config, restarts)
    schedule = require_run_schedule(config, restart=receipt, write_restarts=True)
    runtime = SwathCpuRuntime(build_runtime_config(config), seed=31)
    restored = read_cartesian_restart(selected, publisher=publisher, into=tmp_path / f"restored-{tag}")
    resume = runtime.restore_restart(restored)
    integrated = _integrate(
        runtime, config, tmp_path, tag, publisher,
        first_history_index=schedule.first_history_index,
        first_restart_number=schedule.first_restart_number,
    )
    return runtime, resume, integrated


@pytest.mark.parametrize("planned, resumed_to", [(30.0, 30.0), (20.0, 30.0), (30.0, 20.0)])
def test_a_resumed_run_publishes_the_uninterrupted_frames_and_restarts(
    forecast_publisher, tmp_path, planned: float, resumed_to: float,
) -> None:
    """The plan's A/B pair through the real envelope: every carrier, every word.

    ``planned`` is the length of the run that writes restart 1 (at 10 s);
    ``resumed_to`` the length of the resumed run and of the uninterrupted run
    it is compared with.  Unequal lengths re-place the ground canvas.
    """

    whole_config = load_config(_swath_document(tmp_path, duration=resumed_to, name="whole.toml"))
    whole = SwathCpuRuntime(build_runtime_config(whole_config), seed=31)
    whole_receipt = _integrate(whole, whole_config, tmp_path, "whole", forecast_publisher)

    first_config = load_config(_swath_document(tmp_path, duration=planned, name="first.toml"))
    first = SwathCpuRuntime(build_runtime_config(first_config), seed=31)
    _integrate(first, first_config, tmp_path, "first", forecast_publisher)
    carried = require_restart_receipt(tmp_path / "restarts-first" / "restart-000001")[0]
    assert carried["profile"]["profile_id"] == CARTESIAN_RESTART_PROFILE_ID
    assert carried["swaths"]["carried"] == ["sws", "svs"]
    assert carried["swaths"]["whole_run"] is True
    assert carried["profile"]["swaths"]["ground_canvas_shape"] is not None

    resumed, resume, resumed_receipt = _resume(
        tmp_path, forecast_publisher, restarts=tmp_path / "restarts-first", number=1,
        duration=resumed_to, tag="resumed",
    )
    assert resume["swaths"]["continued"] == ["sws", "svs"]
    assert resume["swaths"]["warning"] is None
    assert resumed.raw_state() == whole.raw_state()

    whole_frames = {row["index"]: row for row in whole_receipt["frames"]}
    compared = 0
    for row in resumed_receipt["frames"]:
        left = _frame_words(tmp_path / "frames-resumed", row["index"])
        right = _frame_words(tmp_path / "frames-whole", row["index"])
        assert set(left) == set(right)
        assert {"sws", "svs", "sws2", "svs2"} <= set(left)
        differing = sorted(name for name in right if left[name] != right[name])
        assert differing == [], (row["index"], differing)
        assert row["arena_sha256"] == whole_frames[row["index"]]["arena_sha256"]
        compared += 1
    assert compared == int((resumed_to - 10.0) / 5.0)
    # The restart the resumed run writes is the uninterrupted run's restart.
    whole_restarts = {row["number"]: row for row in whole_receipt["restarts"]}
    for row in resumed_receipt["restarts"]:
        left = require_restart_receipt(Path(row["directory"]))[0]
        right = require_restart_receipt(Path(whole_restarts[row["number"]]["directory"]))[0]
        assert left["arena"]["carrier_sha256"] == right["arena"]["carrier_sha256"]
        assert left["profile_sha256"] == right["profile_sha256"]
        assert left["scheduler_scalars"] == right["scheduler_scalars"]
        assert left["swaths"] == right["swaths"]
    tornado = resumed_receipt["tornado_diagnostics"]["swaths"]
    assert tornado["whole_run"] is True and tornado["restored_from_restart"] is True


def test_a_v1_restart_still_resumes_and_its_swaths_restart_there(
    forecast_publisher, tmp_path,
) -> None:
    """An old restart is read, with a "swaths restart here" warning (plan)."""

    config = load_config(_swath_document(tmp_path, duration=30.0, name="whole.toml"))
    settings = build_runtime_config(config)
    whole = SwathCpuRuntime(settings, seed=31)
    _integrate(whole, config, tmp_path, "whole", forecast_publisher)

    # A v1 restart at step 20: the state carriers only, under the v1 profile.
    stopped = SwathCpuRuntime(settings, seed=31)
    stopped.open_tornado_diagnostics(total_steps=60)
    for _ in range(20):
        stopped.step()
    old = tmp_path / "old" / "restart-000001"
    old.parent.mkdir()
    write_cartesian_restart(
        stopped, old, publisher=forecast_publisher,
        profile=cartesian_restart_profile(settings, profile_id=CARTESIAN_RESTART_PROFILE_ID_V1),
        next_history_index=3,
    )
    receipt = require_restart_receipt(old)[0]
    assert receipt["profile"]["profile_id"] == CARTESIAN_RESTART_PROFILE_ID_V1
    assert "swaths" not in receipt and "swaths" not in receipt["profile"]

    resumed, resume, _ = _resume(
        tmp_path, forecast_publisher, restarts=tmp_path / "old", number=1,
        duration=30.0, tag="resumed",
    )
    assert resumed.raw_state() == whole.raw_state()
    swaths = resume["swaths"]
    assert swaths["restart_profile"] == CARTESIAN_RESTART_PROFILE_ID_V1
    assert swaths["restart_here"] == ["sws", "svs"] and swaths["continued"] == []
    assert swaths["warning"].startswith("swaths restart here: sws, svs")
    assert "before GM1 carried the swaths" in swaths["warning"]
    row = resumed.tornado.record()["swaths"]
    assert row["first_step"] == {"sws": 21, "svs": 21} and row["whole_run"] is False
    # And it is true: the resumed run's per-step series start at step 21,
    # where the whole run's start at step 1 (this double's winds only grow,
    # so the two maxima themselves happen to agree).
    resumed_series = resumed.tornado.swaths.host_series()["sws_max"]
    whole_series = whole.tornado.swaths.host_series()["sws_max"]
    assert all(math.isnan(value) for value in resumed_series[:20])
    assert not any(math.isnan(value) for value in whole_series[:20])
    assert resumed_series[20:] == whole_series[20:]


def test_the_door_warns_when_a_resume_starts_the_swaths(door, monkeypatch, tmp_path) -> None:  # noqa: F811
    """The warning reaches stderr and the run's event log (gm1.run-events/v1)."""

    from cm1core import cuda_cartesian_runtime as runtime_module

    def _factory(source, settings, *, cupy, stream, capacity_profile=None):
        runtime = SwathCpuRuntime(settings, seed=77)
        door.built.append(runtime)
        return runtime

    monkeypatch.setattr(runtime_module, "CudaCartesianRuntime", _factory)
    from test_cartesian_restart import _run

    plain = _swath_document(tmp_path, duration=10.0, name="plain.toml", swaths=False)
    _run(plain, door=door, output=tmp_path / "plain.json", integrate=True,
         history_dir=tmp_path / "frames-1", restart_dir=tmp_path / "restarts-1")
    resume_document = _swath_document(
        tmp_path, duration=20.0, name="resume.toml", selectors="irst = 1\nrstnum = 1",
    )
    events = tmp_path / "events.jsonl"
    resumed = _run(
        resume_document, door=door, output=tmp_path / "resumed.json", integrate=True,
        history_dir=tmp_path / "frames-2", restart_dir=tmp_path / "restarts-2",
        restart_from=tmp_path / "restarts-1", events=events,
    )
    swaths = resumed["runtime_benchmark"]["resume"]["swaths"]
    assert swaths["restart_here"] == ["sws", "svs"]
    assert "did not keep them" in swaths["warning"]
    records = [json.loads(line) for line in events.read_text().splitlines()]
    warnings = [r for r in records if r["kind"] == "warning"]
    assert [r["code"] for r in warnings] == ["GM1-SWATHS-RESTART-HERE"]
    assert warnings[0]["message"] == swaths["warning"]
    # A resume that carries its swaths warns about nothing.
    again = _swath_document(
        tmp_path, duration=30.0, name="again.toml", selectors="irst = 1\nrstnum = 2",
    )
    events_2 = tmp_path / "events-2.jsonl"
    continued = _run(
        again, door=door, output=tmp_path / "again.json", integrate=True,
        history_dir=tmp_path / "frames-3", restart_dir=tmp_path / "restarts-3",
        restart_from=tmp_path / "restarts-2", events=events_2,
    )
    assert continued["runtime_benchmark"]["resume"]["swaths"]["continued"] == ["sws", "svs"]
    kinds = [json.loads(line)["kind"] for line in events_2.read_text().splitlines()]
    assert "warning" not in kinds


def test_initial_swath_values_are_what_an_untouched_canvas_holds() -> None:
    swaths = _open(30)
    for name in SWATH_NAMES:
        assert np.all(swaths.ground[name] == np.float32(SWATH_INITIAL_VALUES[name]))
    layout = swaths.restart_layout()
    assert isinstance(layout, SwathRestartLayout)
    assert SwathRestartLayout.from_record(json.loads(json.dumps(layout.record()))) == layout


def test_the_storm_driver_opens_the_swaths_after_its_restore() -> None:
    """``tools/run_cartesian_storm.py`` drives the runtime itself.

    Its swaths must open after the restore -- at the restart step, sized for
    the run's final step, continuing what the restart carries -- and its
    restarts must carry them.  Opened before the restore they would start at
    step 0 and refuse the first resumed fold.
    """

    source = (Path(__file__).resolve().parents[1] / "tools" / "run_cartesian_storm.py").read_text(
        encoding="utf-8"
    )
    assert source.index("restore_cartesian_restart(") < source.index("open_tornado_diagnostics(")
    assert "swaths=None if tornado is None else tornado.restart_layout()" in source
