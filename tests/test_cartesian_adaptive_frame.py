"""CM1's ``do_adapt_move`` must be selectable from a document, and refused by value.

The controller itself is proved against CM1 in ``test_cuda_adaptive_move.py``
and ``test_cuda_adaptive_move_open.py``.  What is proved here is the seam that
did not exist: that two keys in a TOML reach it, that every combination it was
never validated on is refused with a named wrong result, that a fixed-frame
document is untouched by any of it, and that the controller is driven once per
completed large step with CM1's own diagnosis cadence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cm1core.cartesian_adaptive_frame import (
    ADAPTIVE_MOVE_RESTART_PREFIX,
    AdaptiveFrameBinding,
    CartesianAdaptiveFrame,
    cartesian_frame_mode_record,
    cm1_reference_axis,
    launch_cartesian_adaptive_move_after_large_step,
    require_adaptive_frame,
    validate_cartesian_adaptive_move_completion,
)
from cm1core.cartesian_door import require_moving_frame
from cm1core.config import Lane, load_config, parse_mapping
from cm1core.cuda_adaptive_move import (
    ADAPTIVE_MOVE_RESTART_SOURCE_ORDER,
    I32_STATE_FIELDS,
    AdaptiveMoveState,
    cpu_advance_adaptive_move_after_large_step,
    download_adaptive_move_state,
    initialize_adaptive_move_state,
    upload_adaptive_move_state,
)
from cm1core.cuda_stretch_z1 import build_stretch_z1_metric_proof
from cm1core.errors import CM1Refusal
from cm1core.gm1_runtime import RuntimeClock, RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_EXAMPLE = ROOT / "examples" / "cartesian-adaptive-moving-domain.toml"
FIXED_EXAMPLE = ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml"
SECTIONS = (
    "run", "initialization", "grid", "dynamics", "boundaries", "closures",
    "physics", "runtime",
)


def adaptive_mapping() -> dict[str, object]:
    payload = load_config(ADAPTIVE_EXAMPLE).canonical_dict()
    return {section: copy.deepcopy(payload[section]) for section in SECTIONS}


def configured(**overrides: dict[str, object]):
    raw = adaptive_mapping()
    for section, values in overrides.items():
        raw[section].update(values)  # type: ignore[union-attr]
    return parse_mapping(raw)


def refusal(**overrides: dict[str, object]) -> object:
    with pytest.raises(CM1Refusal) as caught:
        config = configured(**overrides)
        require_adaptive_frame(config)
    return caught.value.detail


def rebound(config, **sections):
    """Reach the Cartesian runtime's door gate directly for a combination the configuration grammar refuses first.

    ``config.py`` seals this lane's boundary family, its terrain word and its
    lower boundary before the door is ever called, so the only way to exercise
    the door's own adaptive-move admission on those is to hand it the bound
    configuration it would have seen.  These are separate gates and both are
    load-bearing: a programmatic caller that builds a CM1Config without the
    parser still reaches the door.
    """

    import dataclasses

    changed = {
        name: dataclasses.replace(getattr(config, name), **values)
        for name, values in sections.items()
    }
    return dataclasses.replace(config, **changed)


def rebound_binding(binding, **fields):
    """A binding with one admitted field changed, for a value-by-value test."""

    import dataclasses

    return dataclasses.replace(binding, **fields)


def door_refusal(config) -> object:
    with pytest.raises(CM1Refusal) as caught:
        require_adaptive_frame(config)
    return caught.value.detail


# --------------------------------------------------------------- acceptance


def test_the_two_adapt_move_keys_are_accepted_and_reach_the_controller() -> None:
    """README.namelist:1789-1797's own words, bound to the sealed operator."""

    config = load_config(ADAPTIVE_EXAMPLE)
    assert config.closures.do_adapt_move is True
    assert config.closures.adapt_move_frq == 300.0
    # param.F:704-717's coupling, stated by value in the admitted document.
    assert config.closures.moving_frame is True
    assert config.closures.u_move_m_s == 0.0
    assert config.closures.v_move_m_s == 0.0

    binding = require_adaptive_frame(config)
    assert binding is not None
    assert binding.adapt_move_frq_s == 300.0
    # The four code-2 lateral boundaries select the open-halo bc2d branch.
    assert binding.branch == "radiative-open"
    assert binding.module.__name__.endswith("cuda_adaptive_move_open")
    assert (binding.nx, binding.ny, binding.nz) == (960, 960, 75)
    # param.F:7406-7408's centre, in the iorigin=1 frame this lane uses.
    assert binding.centerx_m == 960 * 50.0 / 2.0
    assert binding.centery_m == 960 * 50.0 / 2.0
    # The runtime's own column, and the height kval = 1 has on it.
    from cm1core.cartesian_door import require_grid_extent

    assert binding.stretch_z == 1
    assert binding.vertical_grid_sha256 == require_grid_extent(config).levels.sha256
    assert binding.lowest_scalar_level_m == 10.0
    assert (binding.dx_m, binding.dy_m) == (50.0, 50.0)


def test_a_four_periodic_domain_selects_the_periodic_operator() -> None:
    """bc.F:12-172's wrapping bc2d is a different halo from the open one."""

    from cm1core.cartesian_adaptive_frame import ADAPTIVE_MOVE_BRANCH_BY_BOUNDARY

    assert ADAPTIVE_MOVE_BRANCH_BY_BOUNDARY[1] == "periodic"
    assert ADAPTIVE_MOVE_BRANCH_BY_BOUNDARY[2] == "radiative-open"
    periodic = rebound(
        load_config(ADAPTIVE_EXAMPLE),
        boundaries={"west": 1, "east": 1, "south": 1, "north": 1},
    )
    binding = require_adaptive_frame(periodic)
    assert binding is not None
    assert binding.branch == "periodic"
    assert binding.module.__name__.endswith("cuda_adaptive_move")
    assert not binding.module.__name__.endswith("cuda_adaptive_move_open")


def test_a_lane_with_no_completed_step_driver_is_refused() -> None:
    from cm1core.config import Lane as _Lane

    config = rebound(load_config(ADAPTIVE_EXAMPLE), run={"lane": _Lane.DRY_DYCORE})
    detail = door_refusal(config)
    assert detail.code == "CM1-CARTESIAN-ADAPT-MOVE-LANE"
    assert "never called" in detail.reason


def test_the_receipt_names_the_frame_mode_the_way_it_names_the_vertical_family() -> None:
    """A reader must be able to tell a fixed run from a following one."""

    config = load_config(ADAPTIVE_EXAMPLE)
    record = cartesian_frame_mode_record(
        require_moving_frame(config), require_adaptive_frame(config)
    )
    assert record["family"] == "adaptive-moving-domain"
    assert record["do_adapt_move"] is True
    # param.F:711 raises imove even though param.F:706-707 starts at zero.
    assert record["imove"] == 1
    assert record["adapt_move_frq_s"] == 300.0
    assert record["branch"] == "radiative-open"
    assert record["operator"] == "cm1core.cuda_adaptive_move_open"
    assert record["lateral_boundary_code"] == 2
    assert record["initial_u_move_m_s"] == 0.0
    assert "azimavg.F:4376-4528" in record["authority"]
    assert record["shifted_carriers"]["u"] == ["u0", "ua", "u3d"]


# ------------------------------------------------------------------ refusals


def test_a_half_declared_pair_is_refused_by_key() -> None:
    raw = adaptive_mapping()
    del raw["closures"]["adapt_move_frq"]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-ADAPT-MOVE-PARTIAL"
    assert "implicit 3600 s" in detail.reason


def test_a_nonpositive_cadence_is_refused_by_value() -> None:
    for value in (0.0, -300.0):
        with pytest.raises(CM1Refusal) as caught:
            configured(closures={"adapt_move_frq": value})
        detail = caught.value.detail
        assert detail.code == "CM1-CFG-ADAPT-MOVE-FRQ"
        assert detail.option == "closures.adapt_move_frq"
        # The concrete wrong result: a division by a nonpositive interval and
        # a controller that fires on every large step.
        assert "divides the centre departure by it" in detail.reason


def test_a_cadence_that_is_not_a_cm1_real_is_refused() -> None:
    with pytest.raises(CM1Refusal) as caught:
        configured(closures={"adapt_move_frq": 300.0000000001})
    assert caught.value.detail.code == "CM1-CFG-ADAPT-MOVE-FRQ-PRECISION"


def test_a_cadence_declared_with_the_controller_off_is_refused() -> None:
    with pytest.raises(CM1Refusal) as caught:
        configured(
            closures={
                "do_adapt_move": False,
                "adapt_move_frq": 300.0,
                "moving_frame": False,
            }
        )
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-ADAPT-MOVE-INACTIVE"
    assert "cm1.F:3669" in detail.reason


def test_a_declared_frame_speed_beside_the_controller_is_refused() -> None:
    with pytest.raises(CM1Refusal) as caught:
        configured(closures={"u_move_m_s": 12.5})
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-ADAPT-MOVE-FRAME-SPEED"
    assert "param.F:706-707" in detail.reason


def test_the_controller_without_imove_is_refused() -> None:
    with pytest.raises(CM1Refusal) as caught:
        configured(closures={"moving_frame": False})
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-ADAPT-MOVE-IMOVE"
    assert "param.F:711" in detail.reason


def test_a_cadence_that_never_lands_on_a_frame_is_refused() -> None:
    """cm1.F:3642-3651: no output step, no centre, no correction, ever."""

    with pytest.raises(CM1Refusal) as caught:
        configured(closures={"adapt_move_frq": 90.0})
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-ADAPT-MOVE-CADENCE-UNWRITTEN"
    assert "getpsmth and getpcent are skipped" in detail.reason
    assert "umove and vmove never leave 0.0" in detail.reason


def test_a_mixed_lateral_boundary_set_is_refused() -> None:
    """A halo the domain does not have would find a centre that is not there."""

    config = rebound(load_config(ADAPTIVE_EXAMPLE), boundaries={"west": 1})
    detail = door_refusal(config)
    assert detail.code == "CM1-CARTESIAN-ADAPT-MOVE-BOUNDARIES"
    assert "bc2d" in detail.reason
    assert "param.F:1906-1950" in detail.reason
    # A rigid wall is refused by the same rule, and CM1 stops for it outright.
    walled = rebound(
        load_config(ADAPTIVE_EXAMPLE),
        boundaries={"west": 3, "east": 3, "south": 3, "north": 3},
    )
    assert door_refusal(walled).code == "CM1-CARTESIAN-ADAPT-MOVE-BOUNDARIES"


def test_terrain_under_an_adaptive_frame_is_refused() -> None:
    config = rebound(load_config(ADAPTIVE_EXAMPLE), grid={"terrain": True})
    detail = door_refusal(config)
    assert detail.code == "CM1-CARTESIAN-ADAPT-MOVE-TERRAIN"
    assert "param.F:1890-1903" in detail.reason


@pytest.mark.parametrize("spacing", (100.0, 250.0))
def test_a_storm_parent_at_any_exact_spacing_is_admitted(spacing: float) -> None:
    """getpsmth/getpcent read xh/yh/xfref/yfref and never dx itself."""

    config = configured(grid={"dx_m": spacing, "dy_m": spacing})
    binding = require_adaptive_frame(config)
    assert binding is not None
    assert (binding.dx_m, binding.dy_m) == (spacing, spacing)
    assert binding.centerx_m == 960 * spacing / 2.0
    assert binding.centery_m == 960 * spacing / 2.0
    xfref, xh, yfref, yh = binding.reference_coordinates()
    assert xfref.dtype == np.float32 and xfref.shape == (961,)
    assert float(xfref[-1]) == 960 * spacing
    assert float(xh[0]) == 0.5 * spacing and float(yh[-1]) == 959.5 * spacing
    record = cartesian_frame_mode_record(require_moving_frame(config), binding)
    assert record["horizontal_spacing_m"] == [spacing, spacing]


def test_the_centred_origin_forms_the_same_exact_faces() -> None:
    """param.F:6649-6663's iorigin = 2 subtracts 0.5d0*dble(dx)*nx."""

    faces, mass = cm1_reference_axis(-0.5 * 960 * 250.0, 250.0, 960)
    assert float(faces[0]) == -120000.0 and float(faces[-1]) == 120000.0
    assert float(mass[480]) == 125.0
    with pytest.raises(ValueError, match="neither iorigin"):
        cm1_reference_axis(-1.0, 250.0, 960)


def test_a_spacing_whose_faces_binary32_cannot_hold_is_refused_by_name() -> None:
    third = float(np.float32(100.0 / 3.0))
    detail = door_refusal(
        rebound(load_config(ADAPTIVE_EXAMPLE), grid={"dx_m": third, "dy_m": third})
    )
    assert detail.code == "CM1-CARTESIAN-ADAPT-MOVE-GRID"
    assert "once narrowed to real" in detail.reason
    assert "param.F:6687,7393" in detail.reason

    decimal = door_refusal(
        rebound(load_config(ADAPTIVE_EXAMPLE), grid={"dx_m": 33.3, "dy_m": 33.3})
    )
    assert decimal.code == "CM1-CARTESIAN-ADAPT-MOVE-GRID"
    assert "is not a CM1 real" in decimal.reason


def test_a_stretched_column_with_a_different_lowest_layer_is_admitted() -> None:
    """kval = 1 on every column; the receipt names that level's own height."""

    config = configured(
        grid={
            "nz": 66, "dz_m": 250.0, "ztop_m": 19500.0, "str_bot_m": 500.0,
            "str_top_m": 11500.0, "dz_bot_m": 50.0, "dz_top_m": 500.0,
        }
    )
    binding = require_adaptive_frame(config)
    assert binding is not None and binding.nz == 66
    assert binding.stretch_z == 1
    assert binding.lowest_scalar_level_m == 25.0
    record = cartesian_frame_mode_record(require_moving_frame(config), binding)
    assert record["lowest_scalar_level_m"] == 25.0
    assert "sigma(1) = 25.0 m" in record["diagnosis_level"]


def test_a_column_whose_lowest_level_cannot_be_named_is_refused(monkeypatch) -> None:
    import cm1core.cuda_adaptive_move as periodic

    def unnamed(vertical):  # noqa: ANN001
        raise ValueError("sigma(1) is not the midpoint of the first layer")

    monkeypatch.setattr(periodic, "validate_adaptive_move_vertical_grid", unnamed)
    detail = door_refusal(load_config(ADAPTIVE_EXAMPLE))
    assert detail.code == "CM1-CARTESIAN-ADAPT-MOVE-VERTICAL"
    assert "azimavg.F:4046-4075" in detail.reason
    assert "sigma(1) is not the midpoint" in detail.reason


def test_surface_drag_under_an_adaptive_frame_is_refused() -> None:
    from cm1core.config import LowerBoundaryConfig

    config = rebound(
        load_config(ADAPTIVE_EXAMPLE),
        closures={
            "lower_boundary": LowerBoundaryConfig(
                bbc=2, sfcmodel=0, isfcflx=0, cecd=1, cnstcd=0.001
            )
        },
    )
    detail = door_refusal(config)
    assert detail.code == "CM1-CARTESIAN-ADAPT-MOVE-SURFACE"
    assert "turb.F:296-307" in detail.reason
    assert "frozen RuntimeConfig" in detail.reason


# ------------------------------------------------- the default is unchanged


def test_a_fixed_frame_document_is_untouched_by_any_of_this() -> None:
    """The keys must cost an existing document nothing: not one byte."""

    config = load_config(FIXED_EXAMPLE)
    closures = config.canonical_dict()["closures"]
    assert "do_adapt_move" not in closures
    assert "adapt_move_frq" not in closures
    # The properties still read CM1's own distributed defaults.
    assert config.closures.do_adapt_move is False
    assert config.closures.adapt_move_frq == 3600.0
    assert require_adaptive_frame(config) is None
    record = cartesian_frame_mode_record(require_moving_frame(config), None)
    assert record["family"] == "fixed-frame"
    assert record["u_move_m_s"] == 12.5
    assert record["v_move_m_s"] == 3.0


def test_a_fixed_frame_clock_still_integrates_the_constant_it_always_did() -> None:
    """RuntimeClock's cm1.F:3512-3513 branch must be bit-for-bit unchanged."""

    grid = CartesianGridGeometry(nx=8, ny=8, nz=8, dx_m=1000.0, dy_m=1000.0, dz_m=500.0)
    config = RuntimeConfig(grid, dt_s=0.5, u_move_m_s=12.5, v_move_m_s=3.0)
    clock = RuntimeClock(config)
    assert clock.adaptive_frame is None

    class Backend:
        def execute(self, operation, *, step):  # noqa: ANN001, D401
            return None

        def synchronize(self) -> None:
            return None

    backend = Backend()
    for _ in range(4):
        clock.step(backend)
    assert clock.completed_steps == 4
    assert clock.domainlocx_m == 4 * 0.5 * 12.5
    assert clock.domainlocy_m == 4 * 0.5 * 3.0
    assert clock.domain_location_record()["umove"] == 12.5
    assert clock.domain_location_record()["imove"] == 1


# ------------------------------------------------------------------ cadence


class _RecordingFrame:
    """A stand-in for the controller that records cm1.F:3669's call cadence."""

    def __init__(self, clock: RuntimeClock, *, dt_s: float, frq_s: float) -> None:
        self.clock = clock
        self.dt_s = dt_s
        self.frq_s = frq_s
        self.calls: list[int] = []
        self.corrections: list[float] = []
        self.adaptmovetim = 0.0
        self.u_move_m_s = 0.0
        self.v_move_m_s = 0.0
        self.domainlocx_m = 0.0
        self.domainlocy_m = 0.0

    def after_large_step(self) -> None:
        step = int(self.clock.completed_steps)
        self.calls.append(step)
        mtime = step * self.dt_s
        # cm1.F:3512-3513: the integral advances with the pre-update velocity.
        self.domainlocx_m += self.dt_s * self.u_move_m_s
        self.domainlocy_m += self.dt_s * self.v_move_m_s
        # cm1.F:3671-3679: one update, one advance, no catch-up loop.
        if mtime >= self.adaptmovetim:
            self.corrections.append(mtime)
            self.u_move_m_s += 1.0
            self.adaptmovetim += self.frq_s


def test_the_controller_runs_on_every_completed_large_step() -> None:
    """cm1.F:3512-3513 and cm1.F:3669-3679 are inside the completed step."""

    grid = CartesianGridGeometry(nx=8, ny=8, nz=8, dx_m=1000.0, dy_m=1000.0, dz_m=500.0)
    config = RuntimeConfig(grid, dt_s=1.0, u_move_m_s=0.0, v_move_m_s=0.0)
    clock = RuntimeClock(config)
    frame = _RecordingFrame(clock, dt_s=1.0, frq_s=4.0)
    clock.adaptive_frame = frame

    class Backend:
        def execute(self, operation, *, step):  # noqa: ANN001
            return None

        def synchronize(self) -> None:
            return None

    backend = Backend()
    for _ in range(12):
        clock.step(backend)
    assert frame.calls == list(range(1, 13))
    # CM1's cadence: adaptmovetim starts at 0 (cm1.F:388), so the first
    # completed step is already due, and every fourth second after it.
    assert frame.corrections == [1.0, 4.0, 8.0, 12.0]
    # The clock publishes the controller's integral, not n*dt*umove.
    assert clock.domainlocx_m == frame.domainlocx_m
    assert clock.domainlocx_m != clock.completed_steps * clock.domainlocx_increment_m
    record = clock.domain_location_record()
    assert record["imove"] == 1
    assert record["umove"] == frame.u_move_m_s


def test_the_declared_cadence_is_a_whole_number_of_output_steps() -> None:
    """The refusal above is what makes this arithmetic safe to rely on."""

    config = load_config(ADAPTIVE_EXAMPLE)
    interval = config.run.history_interval_s
    frequency = config.closures.adapt_move_frq
    assert (frequency / interval).is_integer()
    # Every cadence instant is therefore a completed step that publishes a
    # frame, which is exactly cm1.F:3642-3651's dowriteout gate.
    steps_per_frame = interval / config.run.dt_s
    assert steps_per_frame.is_integer()
    cadence_step = frequency / config.run.dt_s
    assert (cadence_step / steps_per_frame).is_integer()


# ----------------------------------------------------------------- restart


def test_the_controller_state_round_trips_through_the_restart_scalars() -> None:
    """restart_write.F:377-401 / restart_read.F:397-421, in CM1's own order."""

    state = AdaptiveMoveState(
        umove=3.5, umovem1=2.5, umovem2=1.5,
        vmove=-0.5, vmovem1=-1.5, vmovem2=-2.5,
        xcenter=19200.0, ycenter=19000.0,
        domainlocx=1234.5, domainlocy=-98.25, adaptmovetim=600.0,
        icenter=385, jcenter=381, nwritemv=3, mvrec=2,
    )
    frame = CartesianAdaptiveFrame.__new__(CartesianAdaptiveFrame)
    frame.host_state = state
    frame._snapshot_step = 0

    class Clock:
        completed_steps = 0

    frame.runtime = type("R", (), {"clock": Clock()})()
    rows = CartesianAdaptiveFrame.restart_scalars(frame)
    assert list(rows) == [
        f"{ADAPTIVE_MOVE_RESTART_PREFIX}{name}"
        for name in ADAPTIVE_MOVE_RESTART_SOURCE_ORDER
    ]

    class Restored:
        step = 7

        def __init__(self, rows):
            self.rows = rows

        def f64(self, name):
            import struct

            kind, value = self.rows[name]
            assert kind == "f64-bits"
            return struct.unpack(">d", bytes.fromhex(value))[0]

    frame.stream = None
    frame.state = None

    def _upload(value, *, stream=None):
        frame.uploaded = value
        return None

    import cm1core.cuda_adaptive_move as periodic

    original = periodic.upload_adaptive_move_state
    periodic.upload_adaptive_move_state = _upload
    try:
        CartesianAdaptiveFrame.restore(frame, Restored(rows))
    finally:
        periodic.upload_adaptive_move_state = original
    assert frame.uploaded == state
    # cm1.F:2663 raises ``restarted`` when the restart is read and cm1.F:4404
    # clears it at the end of the one iteration that follows, so the step the
    # centre diagnosis is skipped on is the restored step plus one.
    assert frame._restarted_step == 8
    for name in I32_STATE_FIELDS:
        assert isinstance(getattr(frame.uploaded, name), int)


# -------------------------------------------------------------------- CUDA


def _cuda_or_skip():
    pytest.importorskip("cupy")
    import cupy

    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"no CUDA device: {error}")
    return cupy


def test_the_lane_carriers_are_shifted_by_the_same_delta_the_oracle_computes() -> None:
    """The sealed kernels, swept over this lane's carriers and nobody else's."""

    cupy = _cuda_or_skip()
    from tools.oracle.adaptive_move_oracle import make_fixture
    from tools.run_cuda_anelp_poisson_periodic import DEFAULT_LOCK, shared_gpu_lock
    from cm1core.cuda_adaptive_move import compile_cuda_adaptive_move

    pressure, xh, yh, xfref, yfref, winds = make_fixture()
    from cm1core.cuda_stretch_z1 import StretchZ1Parameters

    proof = build_stretch_z1_metric_proof(
        StretchZ1Parameters(75, 100.0, 17650.0, 200.0, 9650.0, 20.0, 400.0)
    )
    ni, nj = 16, 14
    centerx, centery = 400.0, 350.0

    oracle_state = initialize_adaptive_move_state(
        ni=ni, nj=nj, centerx=centerx, centery=centery
    )
    oracle_winds = {
        name: np.ascontiguousarray(value.copy()) for name, value in winds.items()
    }
    transition = cpu_advance_adaptive_move_after_large_step(
        oracle_state,
        oracle_winds,
        nstep=1,
        mtime_after=1.0,
        dbldt=1.0,
        adapt_move_frq=60.0,
        centerx=centerx,
        centery=centery,
        pressure_lowest=pressure,
        xh=xh,
        yh=yh,
        xfref=xfref,
        yfref=yfref,
        dowriteout=True,
        vertical_grid_proof=proof,
    )
    assert transition.due and transition.diagnosed

    binding = AdaptiveFrameBinding(
        branch="periodic",
        adapt_move_frq_s=60.0,
        centerx_m=centerx,
        centery_m=centery,
        origin_x_m=0.0,
        origin_y_m=0.0,
        nx=ni,
        ny=nj,
        nz=int(winds["ug"].size),
        dx_m=50.0,
        dy_m=50.0,
        stretch_z=1,
        vertical_grid_sha256=proof.proof_sha256,
        lowest_scalar_level_m=10.0,
    )
    with shared_gpu_lock(DEFAULT_LOCK):
        compiled = compile_cuda_adaptive_move()
        device_state = upload_adaptive_move_state(
            initialize_adaptive_move_state(
                ni=ni, nj=nj, centerx=centerx, centery=centery
            )
        )
        # This lane's inventory: the two padded momentum owners per axis and
        # the base wind compressed to one word per level.
        carriers = {
            "u0": cupy.asarray(np.ascontiguousarray(winds["ug"].copy())),
            "ua": cupy.asarray(np.ascontiguousarray(winds["ua"].copy())),
            "u3d": cupy.asarray(np.ascontiguousarray(winds["u3d"].copy())),
            "v0": cupy.asarray(np.ascontiguousarray(winds["vg"].copy())),
            "va": cupy.asarray(np.ascontiguousarray(winds["va"].copy())),
            "v3d": cupy.asarray(np.ascontiguousarray(winds["v3d"].copy())),
        }
        before = {name: cupy.asnumpy(value) for name, value in carriers.items()}
        launch = launch_cartesian_adaptive_move_after_large_step(
            binding,
            compiled,
            device_state,
            carriers,
            nstep=1,
            mtime_after=1.0,
            dbldt=1.0,
            centerx=centerx,
            centery=centery,
            pressure_lowest=cupy.asarray(pressure),
            xh=cupy.asarray(xh),
            yh=cupy.asarray(yh),
            xfref=cupy.asarray(xfref),
            yfref=cupy.asarray(yfref),
            dowriteout=True,
            vertical_grid_proof=proof,
        )
        validate_cartesian_adaptive_move_completion(launch)
        gpu_state = download_adaptive_move_state(device_state)
        after = {name: cupy.asnumpy(value) for name, value in carriers.items()}

    # The controller state is the oracle's, word for word.
    for name in ADAPTIVE_MOVE_RESTART_SOURCE_ORDER:
        assert getattr(gpu_state, name) == getattr(oracle_state, name), name
    # And every carrier this lane owns moved by exactly azimavg.F:4351-4366's
    # (oldumove - umove), in raw bits.
    delta_u = np.float32(transition.delta_u)
    delta_v = np.float32(transition.delta_v)
    assert float(delta_u) != 0.0
    for name, delta in (
        ("u0", delta_u), ("ua", delta_u), ("u3d", delta_u),
        ("v0", delta_v), ("va", delta_v), ("v3d", delta_v),
    ):
        expected = np.add(before[name], delta, dtype=np.float32)
        assert np.array_equal(
            after[name].view(np.uint32), expected.view(np.uint32)
        ), name


def test_a_carrier_inventory_that_is_not_this_lane_s_is_refused() -> None:
    """A partial rectangle would leave the halo in the old frame."""

    cupy = _cuda_or_skip()
    from cm1core.cartesian_adaptive_frame import _require_cartesian_carriers
    from tools.run_cuda_anelp_poisson_periodic import DEFAULT_LOCK, shared_gpu_lock

    with shared_gpu_lock(DEFAULT_LOCK):
        good = {
            "u0": cupy.zeros((3,), dtype=cupy.float32),
            "ua": cupy.zeros((5, 4, 5), dtype=cupy.float32),
            "u3d": cupy.zeros((5, 4, 5), dtype=cupy.float32),
            "v0": cupy.zeros((3,), dtype=cupy.float32),
            "va": cupy.zeros((5, 5, 4), dtype=cupy.float32),
            "v3d": cupy.zeros((5, 5, 4), dtype=cupy.float32),
        }
        assert set(_require_cartesian_carriers(cupy, good, nz=3)) == set(good)

        missing = dict(good)
        del missing["u3d"]
        with pytest.raises(ValueError, match="missing adaptive-frame carriers"):
            _require_cartesian_carriers(cupy, missing, nz=3)

        wrong_stagger = dict(good)
        wrong_stagger["va"] = cupy.zeros((5, 4, 5), dtype=cupy.float32)
        wrong_stagger["v3d"] = cupy.zeros((5, 4, 5), dtype=cupy.float32)
        with pytest.raises(ValueError, match="CM1 u/v stagger"):
            _require_cartesian_carriers(cupy, wrong_stagger, nz=3)

        strided = dict(good)
        strided["ua"] = good["ua"][:, :, 1:]
        with pytest.raises(ValueError, match="C-order"):
            _require_cartesian_carriers(cupy, strided, nz=3)


# --------------------------------------------------- config -> plan receipt


def test_the_plan_receipt_names_the_controller_the_document_selected(
    tmp_path, monkeypatch, capsys
) -> None:
    """Two keys in a TOML, and the priced plan says which controller ran."""

    import cm1core.cartesian_door as cartesian_door
    from cm1core.cartesian_door import run_cartesian

    monkeypatch.setattr(
        cartesian_door, "require_runnable", lambda config, **kwargs: None
    )
    assert run_cartesian(
        load_config(ADAPTIVE_EXAMPLE),
        config_label=str(ADAPTIVE_EXAMPLE),
        output=tmp_path / "receipt.json",
        plan=True,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    mode = payload["frame_mode"]
    assert mode["family"] == "adaptive-moving-domain"
    assert mode["branch"] == "radiative-open"
    assert mode["adapt_move_frq_s"] == 300.0
    # param.F:706-707's zero is still the declared frame, and the receipt says
    # both things: the constant is zero and the controller owns it.
    assert payload["moving_frame"]["u_move_m_s"] == 0.0
    # The spin-up window, not the whole run, is what a zero frame is judged on.
    # The analytic lane is evaluated on base.F's own winds and reported.
    assert payload["run_geometry"]["status"] in {
        "adaptive-frame-spin-up-clear", "tracked-member-leaves-reported",
        "not-evaluated-no-readable-profile",
    }


def test_a_fixed_frame_plan_receipt_is_unchanged_except_for_one_added_block(
    tmp_path, monkeypatch, capsys
) -> None:
    import cm1core.cartesian_door as cartesian_door
    from cm1core.cartesian_door import run_cartesian

    monkeypatch.setattr(
        cartesian_door, "require_runnable", lambda config, **kwargs: None
    )
    assert run_cartesian(
        load_config(FIXED_EXAMPLE),
        config_label=str(FIXED_EXAMPLE),
        output=tmp_path / "receipt.json",
        plan=True,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frame_mode"]["family"] == "fixed-frame"
    assert payload["frame_mode"]["u_move_m_s"] == 12.5
    assert payload["moving_frame"]["u_move_m_s"] == 12.5
    # The analytic lane is evaluated on base.F's own winds and reported.
    assert payload["run_geometry"]["status"] in {
        "tracked-member-stays-inside", "tracked-member-leaves-reported",
        "not-evaluated-no-readable-profile",
    }


# ------------------------------------------------------------------ spin-up


#: A synthetic CM1 input_sounding with a strong unidirectional shear, so the
#: bubble's own steering wind is large and computable.  Same shape as
#: ``tests/test_run_geometry.py``'s: header row, then height, theta, qv, u, v.
SHEARED_SOUNDING = "\n".join(
    ["   1000.00    300.00    10.00"]
    + [
        f"{z:10.1f} {300.0:10.3f} {10.0:10.4f} {-4.0 + 0.006 * z:10.4f} {0.0:10.4f}"
        for z in range(500, 20_001, 500)
    ]
) + "\n"


def test_a_storm_that_leaves_before_the_first_correction_is_refused() -> None:
    """param.F:706-707 starts stationary; the grid must survive that interval."""

    from cm1core.cartesian_door import (
        require_bubble_center,
        require_grid_extent,
        require_run_geometry,
    )

    config = load_config(ADAPTIVE_EXAMPLE)
    binding = require_adaptive_frame(config)
    grid = require_grid_extent(config)
    # base.F isnd=7/iwnd=0, the one lane whose hodograph this door reads.
    from cm1core.gm1_idealized_residency import SINGLE_SOUNDING_SELECTION

    with pytest.raises(CM1Refusal) as caught:
        require_run_geometry(
            config,
            grid,
            selection=SINGLE_SOUNDING_SELECTION,
            moving_frame=require_moving_frame(config),
            center_xy_m=require_bubble_center(config, grid),
            sounding_label="sheared",
            sounding_bytes=SHEARED_SOUNDING.encode("utf-8"),
            adaptive_frame=rebound_binding(binding, adapt_move_frq_s=36000.0),
        )
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-ADAPT-MOVE-SPINUP"
    assert "param.F:706-707" in detail.reason
    assert "before the adaptive controller's first correction" in detail.reason

    # The same grid with the cadence this document actually declares clears the
    # spin-up, and the record says which window was evaluated.
    record = require_run_geometry(
        config,
        grid,
        selection=SINGLE_SOUNDING_SELECTION,
        moving_frame=require_moving_frame(config),
        center_xy_m=require_bubble_center(config, grid),
        sounding_label="sheared",
        sounding_bytes=SHEARED_SOUNDING.encode("utf-8"),
        adaptive_frame=binding,
    )
    assert record["status"] == "adaptive-frame-spin-up-clear"
    assert record["evaluated_window_s"] == binding.adapt_move_frq_s
    assert "no host-side constant-frame track describes it" in str(
        record["adaptive_frame_note"]
    )
