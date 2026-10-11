"""CM1's adaptive large step, checked line by line against its own source.

``adapt_dt = 1`` is set by 15 of CM1 r21.1's 23 shipped configuration decks.
Everything the controller does is arithmetic on two measured numbers, so the
whole of ``misclibs.F:593-810`` can be checked here, on the host, without a
device and without a storm.
"""

from __future__ import annotations

import inspect
import math
import re

import pytest

from cm1core.cm1_adaptive_timestep import (
    CM1_CFL_LIMIT,
    CM1_KS_LIMIT_2D,
    CM1_KS_LIMIT_3D,
    CM1_MAX_CHANGE,
    CM1_MAX_NSOUND,
    RampDownTarget,
    cm1_diffusion_number,
    cm1_dtsmall,
    cm1_getnewdt,
    cm1_ks_limit,
)
from cm1core.gm1_runtime import cm1_diagnostic_acoustic_substeps


# ---------------------------------------------------------------------------
# dtsmall (misclibs.F:818-915)


def test_dtsmall_agrees_with_the_psolver3_helper_the_door_already_used() -> None:
    """Two transcriptions of one source routine must not disagree."""

    for dt, dx, dy in ((0.5, 50.0, 50.0), (1.0, 1000.0, 1000.0), (6.0, 1000.0, 1000.0), (2.0, 250.0, 250.0)):
        _, nsound = cm1_dtsmall(dt_s=dt, nx=32, ny=32, dx_m=dx, dy_m=dy, dz_m=dx / 2)
        assert nsound == cm1_diagnostic_acoustic_substeps(dt_s=dt, dx_m=dx, dy_m=dy)


def test_dtsmall_never_returns_an_odd_count_or_fewer_than_four() -> None:
    """misclibs.F:883-884."""

    for dt in (0.05, 0.5, 1.0, 3.0, 6.0, 20.0):
        _, nsound = cm1_dtsmall(dt_s=dt, nx=64, ny=64, dx_m=1000.0, dy_m=1000.0, dz_m=500.0)
        assert nsound >= 4 and nsound % 2 == 0


def test_dtsmall_small_step_never_exceeds_the_stability_bound() -> None:
    """misclibs.F:885 is a second pass, not a rounding: it must hold after."""

    for dt in (0.5, 1.0, 2.0, 5.0, 11.0, 37.0):
        limit = 0.60 * 1000.0 / 360.0
        _, nsound = cm1_dtsmall(dt_s=dt, nx=64, ny=64, dx_m=1000.0, dy_m=1000.0, dz_m=500.0)
        assert dt / nsound <= limit


@pytest.mark.parametrize("psolver", [1, 4, 5])
def test_the_fully_explicit_and_anelastic_solvers_take_one_small_step(psolver: int) -> None:
    """misclibs.F:837-840."""

    dt, nsound = cm1_dtsmall(
        dt_s=3.0, nx=32, ny=32, dx_m=1000.0, dy_m=1000.0, dz_m=500.0, psolver=psolver
    )
    assert (dt, nsound) == (3.0, 1)


def test_psolver2_bounds_the_small_step_by_dz_as_well() -> None:
    """misclibs.F:844-855: the fully explicit solver is not implicit in z."""

    _, three = cm1_dtsmall(dt_s=3.0, nx=32, ny=32, dx_m=1000.0, dy_m=1000.0, dz_m=100.0, psolver=2)
    _, implicit = cm1_dtsmall(dt_s=3.0, nx=32, ny=32, dx_m=1000.0, dy_m=1000.0, dz_m=100.0, psolver=3)
    assert three > implicit


def test_the_two_dimensional_branches_use_the_surviving_horizontal_spacing() -> None:
    """misclibs.F:859-866."""

    _, xz = cm1_dtsmall(dt_s=1.0, nx=64, ny=1, dx_m=100.0, dy_m=9.9e9, dz_m=100.0)
    _, yz = cm1_dtsmall(dt_s=1.0, nx=1, ny=64, dx_m=9.9e9, dy_m=100.0, dz_m=100.0)
    assert xz == yz
    _, both = cm1_dtsmall(dt_s=1.0, nx=64, ny=64, dx_m=100.0, dy_m=100.0, dz_m=100.0)
    assert both == xz


def test_a_step_needing_more_than_max_nsound_stops_or_shrinks() -> None:
    """misclibs.F:886-910: the two branches differ only in adapt_dt."""

    with pytest.raises(ValueError, match="needs to be smaller"):
        cm1_dtsmall(dt_s=600.0, nx=64, ny=64, dx_m=50.0, dy_m=50.0, dz_m=50.0)
    shrunk, nsound = cm1_dtsmall(
        dt_s=600.0, nx=64, ny=64, dx_m=50.0, dy_m=50.0, dz_m=50.0, adapt_dt=True
    )
    assert nsound == CM1_MAX_NSOUND
    assert shrunk < 600.0
    assert shrunk == pytest.approx(CM1_MAX_NSOUND * 0.60 * 50.0 / 360.0)


# ---------------------------------------------------------------------------
# ks_limit and the diffusion number


def test_the_diffusion_ceiling_is_halved_in_three_dimensions() -> None:
    """misclibs.F:619-628."""

    assert cm1_ks_limit(nx=64, ny=64) == CM1_KS_LIMIT_3D
    assert cm1_ks_limit(nx=64, ny=1) == CM1_KS_LIMIT_2D
    assert cm1_ks_limit(nx=1, ny=64) == CM1_KS_LIMIT_2D
    assert CM1_KS_LIMIT_3D * 2.0 == CM1_KS_LIMIT_2D


def test_the_diffusion_number_is_the_calcksquick_norm_on_a_uniform_grid() -> None:
    """misclibs.F:2841-2866 with uh = vh = 1.

    ``tem1 = sqrt((km*dtdx)**2 + (km*dtdy)**2)`` with ``dtdx = dt*rdx*rdx``.
    """

    dt, dx, dy, km = 2.0, 500.0, 250.0, 37.5
    dtdx, dtdy = dt / (dx * dx), dt / (dy * dy)
    assert cm1_diffusion_number(
        dt_s=dt, dx_m=dx, dy_m=dy, max_eddy_coefficient_m2_s=km
    ) == pytest.approx(math.sqrt((km * dtdx) ** 2 + (km * dtdy) ** 2))


def test_the_diffusion_number_is_linear_in_the_timestep() -> None:
    """getnewdt relies on this at misclibs.F:648: ksmax*dbldt/dt."""

    kwargs = {"dx_m": 100.0, "dy_m": 100.0, "max_eddy_coefficient_m2_s": 12.0}
    assert cm1_diffusion_number(dt_s=2.0, **kwargs) == pytest.approx(
        2.0 * cm1_diffusion_number(dt_s=1.0, **kwargs)
    )


def test_a_quiescent_closure_gives_a_zero_diffusion_number() -> None:
    assert cm1_diffusion_number(
        dt_s=1.0, dx_m=100.0, dy_m=100.0, max_eddy_coefficient_m2_s=0.0
    ) == 0.0


# ---------------------------------------------------------------------------
# getnewdt (misclibs.F:635-802)


def _decide(**kwargs):
    base = {"dt_s": 1.0, "cflmax": 0.5, "ksmax": 0.0, "dtl_s": 1.0, "ks_limit": CM1_KS_LIMIT_3D}
    base.update(kwargs)
    return cm1_getnewdt(**base)


def test_a_courant_number_above_the_limit_shrinks_the_step_in_proportion() -> None:
    """misclibs.F:639-641."""

    decision = _decide(dt_s=2.0, cflmax=4.0, dtl_s=2.0)
    assert decision.dt_s == pytest.approx(2.0 * (CM1_CFL_LIMIT / 4.0))
    # misclibs.F:800-801 rescales the measured number to the new step, and
    # that is exactly the target.
    assert decision.cflmax_scaled == pytest.approx(CM1_CFL_LIMIT)
    assert decision.limited_by == "cfl"


def test_growth_is_capped_at_ten_percent_per_step() -> None:
    """misclibs.F:643: min(1+max_change, cfl_limit/cflmax)."""

    decision = _decide(cflmax=1.0e-6, dtl_s=100.0)
    assert decision.dt_s == pytest.approx(1.0 + CM1_MAX_CHANGE)
    assert decision.limited_by == "cfl-growth-cap"


def test_growth_below_the_cap_follows_the_courant_ratio_exactly() -> None:
    decision = _decide(cflmax=0.95, dtl_s=100.0)
    assert decision.dt_s == pytest.approx(1.0 / 0.95)


def test_a_domain_at_rest_does_not_divide_by_zero() -> None:
    """misclibs.F:637 floors cflmax before the division."""

    assert _decide(cflmax=0.0, dtl_s=100.0).dt_s == pytest.approx(1.0 + CM1_MAX_CHANGE)


def test_the_diffusion_number_can_take_the_step_below_the_courant_answer() -> None:
    """misclibs.F:647-650."""

    loose = _decide(cflmax=0.5, ksmax=0.0, dtl_s=100.0)
    tight = _decide(cflmax=0.5, ksmax=0.5, dtl_s=100.0)
    assert tight.dt_s < loose.dt_s
    assert tight.limited_by == "diffusion"
    # misclibs.F:649: dt*ks_limit/ksmax.
    assert tight.dt_s == pytest.approx(1.0 * CM1_KS_LIMIT_3D / 0.5)
    assert tight.ksmax_scaled == pytest.approx(CM1_KS_LIMIT_3D)


def test_the_diffusion_check_never_raises_the_courant_answer() -> None:
    """misclibs.F:649 takes a min, so a quiet closure cannot grow the step."""

    quiet = _decide(cflmax=2.0, ksmax=1.0e-9, dtl_s=100.0)
    assert quiet.dt_s == pytest.approx(0.5)


def test_the_step_never_exceeds_twice_the_declared_dtl() -> None:
    """misclibs.F:654."""

    decision = _decide(dt_s=1.9, cflmax=1.0e-9, dtl_s=1.0)
    assert decision.dt_s == pytest.approx(2.0)
    assert decision.limited_by == "twice-dtl"


def test_the_step_lands_exactly_on_an_upcoming_output_time() -> None:
    """misclibs.F:668-680, the three-way ramp-down."""

    target = RampDownTarget("taptim", 100.0, 20.0, False)
    # remaining 2.5*dt -> dt/3 of it
    third = cm1_getnewdt(
        dt_s=1.0, cflmax=0.5, ksmax=0.0, dtl_s=100.0, ks_limit=CM1_KS_LIMIT_3D,
        model_time_s=100.0 - 2.5, ramp_down_targets=(target,),
    )
    assert third.dt_s == pytest.approx(2.5 / 3.0)
    assert third.limited_by == "event:taptim"
    # remaining 1.5*dt -> half of it
    half = cm1_getnewdt(
        dt_s=1.0, cflmax=0.5, ksmax=0.0, dtl_s=100.0, ks_limit=CM1_KS_LIMIT_3D,
        model_time_s=100.0 - 1.5, ramp_down_targets=(target,),
    )
    assert half.dt_s == pytest.approx(0.75)
    # remaining <= dt -> exactly the remainder, so the frame time is hit
    last = cm1_getnewdt(
        dt_s=1.0, cflmax=0.5, ksmax=0.0, dtl_s=100.0, ks_limit=CM1_KS_LIMIT_3D,
        model_time_s=100.0 - 0.4, ramp_down_targets=(target,),
    )
    assert last.dt_s == pytest.approx(0.4)


def test_the_first_call_skips_every_scheduled_event() -> None:
    """misclibs.F:658: cm1.F:3039 calls getnewdt with doinit true."""

    target = RampDownTarget("taptim", 100.0, 20.0, False)
    decision = cm1_getnewdt(
        dt_s=1.0, cflmax=0.5, ksmax=0.0, dtl_s=100.0, ks_limit=CM1_KS_LIMIT_3D,
        model_time_s=100.0 - 0.4, ramp_down_targets=(target,), apply_ramp_down=False,
    )
    assert decision.dt_s == pytest.approx(1.0 + CM1_MAX_CHANGE)


def test_an_event_that_is_due_now_uses_the_following_one() -> None:
    """misclibs.F:661-664 adds the frequency when the write happens here."""

    due = RampDownTarget("taptim", 100.0, 20.0, True)
    decision = cm1_getnewdt(
        dt_s=1.0, cflmax=0.5, ksmax=0.0, dtl_s=100.0, ks_limit=CM1_KS_LIMIT_3D,
        model_time_s=100.0, ramp_down_targets=(due,),
    )
    # 20 s away, far more than 3*dt, so nothing ramps.
    assert decision.dt_s == pytest.approx(1.0 + CM1_MAX_CHANGE)


def test_a_controller_that_produced_a_nonpositive_step_raises() -> None:
    """A target the clock has already passed would ramp dt to a negative."""

    with pytest.raises(ValueError, match="non-positive large step"):
        cm1_getnewdt(
            dt_s=1.0, cflmax=0.5, ksmax=0.0, dtl_s=1.0, ks_limit=CM1_KS_LIMIT_3D,
            model_time_s=10.0,
            ramp_down_targets=(RampDownTarget("taptim", 5.0, 20.0, False),),
        )


# ---------------------------------------------------------------------------
# The path from a document to a changed step


def test_the_configuration_door_admits_adapt_dt_and_carries_dtl() -> None:
    from pathlib import Path

    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import load_config, parse_mapping

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "cartesian-uniform-grid.toml")
    raw = {
        name: value
        for name, value in config.canonical_dict().items()
        if name in {"run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime", "initialization"}
    }
    raw["dynamics"]["adaptive_timestep"] = True
    runtime = build_runtime_config(parse_mapping(raw))
    assert runtime.adaptive_timestep is True
    assert runtime.dtl_s == raw["run"]["dt_s"]
    assert runtime.record()["adapt_dt"] is True


def test_adapt_dt_lets_an_event_time_that_is_not_a_whole_step_through() -> None:
    """The whole point: the events set the steps, not the other way round."""

    from pathlib import Path

    from cm1core.config import load_config, parse_mapping
    from cm1core.errors import CM1Refusal

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples" / "cartesian-uniform-grid.toml")
    raw = {
        name: value
        for name, value in config.canonical_dict().items()
        if name in {"run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime", "initialization"}
    }
    # 0.7 s does not divide 60 s in binary64, so a fixed-step clock cannot
    # place the last frame on a step boundary; an adaptive one lands on it by
    # shortening the step that would have overshot (misclibs.F:668-680).
    raw["run"]["dt_s"] = 0.7
    raw["run"]["duration_s"] = 60.0
    raw["run"]["history_interval_s"] = 30.0
    raw["run"]["restart_interval_s"] = 60.0
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-CLOCK-ALIGNMENT"
    raw["dynamics"]["adaptive_timestep"] = True
    assert parse_mapping(raw).run.dt_s == 0.7


def test_the_runtime_rebinds_nsound_with_every_change_of_dt() -> None:
    """misclibs.F:785 -- the two are one decision, never one without the other."""

    from cm1core import cuda_cartesian_runtime as runtime

    source = inspect.getsource(runtime.CudaCartesianRuntime.rebind_large_step)
    assert "cm1_dtsmall(" in source
    for holder in ("self.config", "self.dynamics.config", "self.closure.config", "self.clock.config"):
        assert f"{holder} = updated" in source
    loop = inspect.getsource(runtime.CudaCartesianRuntime.integrate)
    # The controller runs before the step it sizes, and the loop is driven by
    # the model clock rather than by a step count.
    assert "self.adaptive_large_step(" in loop
    # a3b8c75 (merge of int/dynamics onto the restart schedule) wraps the
    # condition across lines inside the per-event loop; the claim is the same.
    assert re.search(r"while\s*\(?\s*self\.clock\.model_seconds\s*<", loop)


def test_the_clock_accumulates_model_time_when_the_step_varies() -> None:
    from cm1core.gm1_runtime import RuntimeClock, RuntimeConfig
    from cm1core.grid_geometry import CartesianGridGeometry

    grid = CartesianGridGeometry(nx=32, ny=32, nz=40, dx_m=1000.0, dy_m=1000.0, dz_m=500.0)
    config = RuntimeConfig(grid, dt_s=1.0, acoustic_substeps=4, adaptive_timestep=True)
    clock = RuntimeClock(config)
    assert clock.model_seconds == 0.0
    clock.completed_steps, clock.accumulated_model_s = 3, 2.75
    assert clock.model_seconds == 2.75
    fixed = RuntimeConfig(grid, dt_s=1.0, acoustic_substeps=4)
    assert RuntimeClock(fixed, completed_steps=3, accumulated_model_s=3.0).model_seconds == 3.0
