from __future__ import annotations

import math

import pytest

from cm1core.gm1_runtime import (
    CM1_COURANT_STOP,
    CM1_COURANT_STOP_AUTHORITY,
    CM1_COURANT_TARGET,
    CM1_MAX_ACOUSTIC_SUBSTEPS,
    CM1_MIN_ACOUSTIC_SUBSTEPS,
    CM1_MIN_NORMAL_DT_S,
    CM1_PSOLVER3_ACOUSTIC_COURANT,
    CM1_PSOLVER3_ACOUSTIC_SPEED_M_S,
    DOMAIN_LOCATION_EXACT_STEP_LIMIT,
    DOMAIN_LOCATION_INITIAL_M,
    RuntimeClock,
    RuntimeConfig,
    StepPhase,
    acoustic_small_step_defect,
    advective_courant_defect,
    cm1_acoustic_small_step_limit_s,
    cm1_advective_courant,
    cm1_courant_limited_speed_m_s,
    cm1_diagnostic_acoustic_substeps,
    cm1_largest_clock_aligned_large_step_s,
    cm1_max_large_step_s,
    cm1_real,
    history_cadence,
    large_step_schedule,
    positive_normal_f32_bits,
)
from cm1core.grid_geometry import CartesianGridGeometry


def config(**kwargs):
    return RuntimeConfig(CartesianGridGeometry(400, 400, 237, 75, 75, 75), **kwargs)


def test_requested_grid_is_a_runtime_configuration_without_a_fixture_identity():
    value = config()
    assert value.grid.mass_shape == (237, 400, 400)
    assert value.grid.ztop_m == 17775
    assert value.record()["grid"]["dx_m"] == 75
    # sound.F's rounded RK1 count also supports even counts not divisible
    # by three.  Configuration must not turn the default six into a limit.
    assert config(acoustic_substeps=8).acoustic_substeps == 8


def test_half_second_remains_the_default_large_step():
    assert config().dt_s == 0.5


@pytest.mark.parametrize("dt", [0.25, 0.1, 0.500000001, 0.05, 0.0125])
def test_a_finer_large_step_is_admitted_now_that_no_stage_welds_one(dt):
    """A 50 m or finer tornado attempt needs a step below the CM1 default.

    Nothing in the schedule carries a compile-time timestep any more, so the
    only bound left is CM1's own: ``misclibs.F:818-915``.
    """

    value = config(dt_s=dt)
    assert value.dt_s == dt
    assert value.record()["acoustic_small_step_s"] == dt / value.acoustic_substeps


def test_dtsmall_is_cm1s_own_psolver3_three_dimensional_branch():
    """``misclibs.F:867``: ``dtsm`` is 0.60 times the smaller horizontal spacing over 360."""

    assert CM1_PSOLVER3_ACOUSTIC_COURANT == 0.60
    assert CM1_PSOLVER3_ACOUSTIC_SPEED_M_S == 360.0
    assert CM1_MIN_ACOUSTIC_SUBSTEPS == 4
    assert CM1_MAX_ACOUSTIC_SUBSTEPS == 192
    for dx, dy in ((50.0, 50.0), (1000.0, 250.0), (25.0, 100.0)):
        assert cm1_acoustic_small_step_limit_s(dx_m=dx, dy_m=dy) == (
            0.60 * min(dx, dy) / 360.0
        )


@pytest.mark.parametrize(
    "dt_s,dx_m,expected",
    [
        # misclibs.F:885-888 with the 50 m child GM1 already runs: nint(6)=6,
        # even, and 0.5/6 does not exceed dtsm, so CM1 would choose exactly the
        # six acoustic substeps this port declares.
        (0.5, 50.0, 6),
        # A kilometre grid needs no substructure at half a second, so CM1's
        # floor of four decides.
        (0.5, 1000.0, 4),
        # CM1's own supercell deck: dtl=7.5 s at dx=1000 m.
        (7.5, 1000.0, 6),
        # nint rounds half away from zero, and an odd count is raised to even.
        (0.25, 25.0, 6),
    ],
)
def test_diagnostic_substep_count_matches_cm1s_dtsmall(dt_s, dx_m, expected):
    assert cm1_diagnostic_acoustic_substeps(dt_s=dt_s, dx_m=dx_m, dy_m=dx_m) == expected
    limit = cm1_acoustic_small_step_limit_s(dx_m=dx_m, dy_m=dx_m)
    assert dt_s / expected <= limit


def test_a_large_step_whose_acoustic_substep_is_unstable_refuses_by_name():
    """dx=dy=75 m admits 0.125 s of small step; six substeps of 1.0 s is 0.167."""

    with pytest.raises(ValueError) as caught:
        config(dt_s=1.0)
    message = str(caught.value)
    assert "acoustic small step" in message
    assert "misclibs.F:867" in message
    # 0.60 is CM1's margin against a 360 m/s reference sound speed, not the
    # instability.  The explicit horizontal terms reach that at an acoustic
    # Courant number of 1, and the message must say which is which.
    assert "CM1's margin, not the instability" in message
    assert "reaches 1" in message
    assert "grow without bound" not in message
    assert "nsound=8" in message
    # The same step is admissible once the substep count CM1 would derive is
    # declared, so the refusal names a remedy that actually works.
    assert config(dt_s=1.0, acoustic_substeps=8).dt_s == 1.0


def test_max_large_step_and_defect_agree_on_the_boundary():
    limit = cm1_max_large_step_s(dx_m=75.0, dy_m=75.0, acoustic_substeps=6)
    assert limit == 6 * (0.60 * 75.0 / 360.0)
    assert acoustic_small_step_defect(
        dt_s=limit, dx_m=75.0, dy_m=75.0, acoustic_substeps=6
    ) is None
    assert acoustic_small_step_defect(
        dt_s=limit * 1.0000001, dx_m=75.0, dy_m=75.0, acoustic_substeps=6
    ) is not None


def test_the_printed_maximum_is_admitted_for_every_spacing_and_substep_count():
    """A remedy that is a fixed point the same door rejects is not a remedy.

    ``nsound*dtsm`` is the exact answer but the door's test is ``dt/nsound >
    dtsm``, and float64 rounds ``(nsound*dtsm)/nsound`` strictly above ``dtsm``
    for many pairs.  ``dx=125 m`` with 22 substeps was one: the refusal printed
    4.583333333333334 s and refused that value with the same code and the same
    number.  One pinned pair cannot see this, so the whole surface is swept.
    """

    refused: list[tuple[float, int, float]] = []
    for dx_m in (
        20.0, 25.0, 31.25, 50.0, 62.5, 75.0, 100.0, 125.0,
        150.0, 200.0, 250.0, 333.0, 500.0, 1000.0,
    ):
        for nsound in range(
            CM1_MIN_ACOUSTIC_SUBSTEPS, CM1_MAX_ACOUSTIC_SUBSTEPS + 1, 2
        ):
            maximum = cm1_max_large_step_s(
                dx_m=dx_m, dy_m=dx_m, acoustic_substeps=nsound
            )
            if acoustic_small_step_defect(
                dt_s=maximum, dx_m=dx_m, dy_m=dx_m, acoustic_substeps=nsound
            ) is not None:
                refused.append((dx_m, nsound, maximum))
            # And the walk-down never overshoots: the value stays within two
            # ulps of the exact-arithmetic product it started from, so the
            # refusal gives up no admissible step to buy its own consistency.
            product = nsound * cm1_acoustic_small_step_limit_s(
                dx_m=dx_m, dy_m=dx_m
            )
            floor = math.nextafter(math.nextafter(product, 0.0), 0.0)
            assert floor <= maximum <= product, (dx_m, nsound)
    assert refused == [], (
        "the printed maximum large step is refused by the same criterion "
        f"that printed it for {len(refused)} pairs: {refused[:5]}"
    )


def test_no_remedy_names_a_substep_count_above_cm1s_own_ceiling():
    """``misclibs.F:887-911``: above ``max_nsound`` CM1 has no substep route.

    A 20 m grid at a 20 s step needs 600 small steps, which
    ``_acoustic_substeps`` refuses outright.  Naming it would send the operator
    to a second refusal with a different code.
    """

    message = acoustic_small_step_defect(
        dt_s=20.0, dx_m=20.0, dy_m=20.0, acoustic_substeps=6
    )
    assert message is not None
    assert "nsound=600" in message
    assert f"above max_nsound={CM1_MAX_ACOUSTIC_SUBSTEPS}" in message
    assert "Time step (dtl) needs to be smaller" in message
    assert "use at least that many acoustic substeps" not in message

    # Below the ceiling the substep route is offered and is the count CM1 uses.
    workable = acoustic_small_step_defect(
        dt_s=1.0, dx_m=75.0, dy_m=75.0, acoustic_substeps=6
    )
    assert workable is not None and "use at least that many" in workable


def test_the_clock_aligned_step_divides_every_declared_event():
    for events in ((600.0, 10.0, 600.0), (7200.0, 60.0, 7200.0), (5.0, 5.0, 5.0)):
        dt = cm1_largest_clock_aligned_large_step_s(
            dx_m=125.0, dy_m=125.0, acoustic_substeps=22, event_seconds=events
        )
        assert dt is not None
        assert acoustic_small_step_defect(
            dt_s=dt, dx_m=125.0, dy_m=125.0, acoustic_substeps=22
        ) is None
        for event in events:
            assert (event / dt).is_integer(), (event, dt)


def test_cm1s_own_substep_ceiling_is_refused_by_name():
    assert acoustic_small_step_defect(
        dt_s=0.5, dx_m=75.0, dy_m=75.0, acoustic_substeps=CM1_MAX_ACOUSTIC_SUBSTEPS
    ) is None
    defect = acoustic_small_step_defect(
        dt_s=0.5, dx_m=75.0, dy_m=75.0, acoustic_substeps=CM1_MAX_ACOUSTIC_SUBSTEPS + 2
    )
    assert defect is not None and "max_nsound=192" in defect


@pytest.mark.parametrize("dt", [0.0, -0.5, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_large_step_still_refuses(dt):
    with pytest.raises(ValueError, match="finite and positive"):
        config(dt_s=dt)


def test_final_mass_correction_preserves_interposed_source_work():
    order = list(large_step_schedule())
    at = {operation.phase: index for index, operation in enumerate(order) if operation.nrk == 3}
    assert at[StepPhase.SCALAR_MOIST] < at[StepPhase.REMOVE_HELD_EOS]
    assert at[StepPhase.REMOVE_HELD_EOS] < at[StepPhase.MASS_BEGIN]
    assert at[StepPhase.MASS_BEGIN] < at[StepPhase.MOMENTUM_BOUNDARIES]
    assert at[StepPhase.MOMENTUM_BOUNDARIES] < at[StepPhase.TKE] < at[StepPhase.MASS_FINISH]
    assert StepPhase.SCALAR_BOUNDARIES not in at  # solve2.F:1682 nrk < nrkmax
    for phase in (StepPhase.PDEFQ, StepPhase.MICROPHYSICS, StepPhase.POST_PHYSICS, StepPhase.SOLVE3):
        assert next(i for i, item in enumerate(order) if item.phase == phase) > at[StepPhase.MASS_FINISH]


class RecordingBackend:
    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def execute(self, operation, *, step):
        self.calls.append((step, operation))
        if operation.phase == self.fail:
            raise ValueError("operator failure")

    def synchronize(self):
        if self.fail == "synchronize":
            raise RuntimeError("deferred CUDA failure")


@pytest.mark.parametrize("failure", [StepPhase.ACOUSTIC, StepPhase.SOLVE3, "synchronize"])
def test_partial_or_asynchronously_failed_step_never_advances_clock(failure):
    clock = RuntimeClock(config())
    clock.step(RecordingBackend())
    assert clock.completed_steps == 1 and clock.model_seconds == 0.5
    with pytest.raises((RuntimeError, ValueError)):
        clock.step(RecordingBackend(failure))
    assert clock.completed_steps == 1 and clock.model_seconds == 0.5
    with pytest.raises(RuntimeError, match="restoring a completed state"):
        clock.step(RecordingBackend())


def test_each_completed_step_runs_all_stages_once():
    clock, backend = RuntimeClock(config()), RecordingBackend()
    clock.step(backend)
    clock.step(backend)
    assert clock.model_seconds == 1.0
    for step in (1, 2):
        stages = [item.nrk for number, item in backend.calls
                  if number == step and item.phase == StepPhase.ACOUSTIC]
        assert stages == [1, 2, 3]


def test_history_cadence_is_whole_large_steps_end_to_end():
    """CM1's idealized deck: timax=7200 at dt=0.5, written every 60 s."""

    cadence = history_cadence(dt_s=0.5, duration_s=7200.0, interval_s=60.0)
    assert (cadence.total_steps, cadence.steps_per_frame) == (14400, 120)
    assert cadence.frame_count == 120
    assert cadence.frame_steps() == tuple(range(120, 14401, 120))
    assert cadence.frame_model_seconds()[-1] == cadence.duration_s
    assert cadence.record()["steps_per_frame"] == 120


@pytest.mark.parametrize(
    ("duration_s", "interval_s"),
    [(7200.0, 7.0), (5.0, 6.0), (5.1, 0.5), (5.0, 0.75), (5.0, 0.0)],
)
def test_history_cadence_refuses_what_the_fixed_clock_cannot_place(duration_s, interval_s):
    with pytest.raises(ValueError):
        history_cadence(dt_s=0.5, duration_s=duration_s, interval_s=interval_s)


def test_the_frame_speed_bound_is_the_same_number_in_rust_and_python() -> None:
    """One bound, two languages: the refusals must name the same metre.

    The configuration refuses an absurd frame velocity before a device is
    priced and the Rust initializer refuses it again at the ABI.  If the two
    numbers drifted apart there would be a band of speeds Python admits and
    Rust rejects, and the run would die at initialization with a different
    message than the one the configuration promised.
    """

    import re
    from pathlib import Path

    from cm1core.gm1_runtime import MAX_FRAME_SPEED_M_S

    source = (
        Path(__file__).resolve().parents[1]
        / "rust/cm1-state/src/idealized_initialization.rs"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"pub const MAX_FRAME_SPEED_M_PER_S: f32 = ([0-9.]+)_f32;", source
    )
    assert match is not None, "the Rust frame-speed bound was renamed or removed"
    assert float(match.group(1)) == MAX_FRAME_SPEED_M_S


# ------------------------------------------- cm1.F:3512-3513 domain location


def test_a_fixed_frame_never_leaves_cm1s_initial_domain_location():
    """cm1.F:348-349 start both at zero and cm1.F:3512-3513 add dt*0.

    A fixed-frame run's grid is ground-fixed, so every frame it publishes is
    at the origin CM1 started it at, for the whole run.
    """

    clock, backend = RuntimeClock(config()), RecordingBackend()
    assert clock.domainlocx_m == DOMAIN_LOCATION_INITIAL_M
    assert clock.domainlocy_m == DOMAIN_LOCATION_INITIAL_M
    for expected in range(1, 1_001):
        clock.step(backend)
        assert clock.completed_steps == expected
        assert clock.domainlocx_m == 0.0 and clock.domainlocy_m == 0.0
    assert clock.domain_location_record()["imove"] == 0


def test_a_two_hour_run_accumulates_umove_times_t_with_no_drift():
    """cm1.F:3512-3513 over CM1's own idealized deck: 14,400 steps of 0.5 s.

    Exact equality, not closeness.  ``umove`` is CM1 ``real``, so the
    increment ``0.5*umove`` carries at most 24 significant bits; ``n`` times
    that needs at most ``24 + ceil(log2(n))`` bits, and binary64 holds 53, so
    every partial sum is representable and no addition rounds.
    """

    clock = RuntimeClock(config(u_move_m_s=12.5, v_move_m_s=3.0))
    backend = RecordingBackend()
    for step in range(1, 14_401):
        clock.step(backend)
        seconds = step * 0.5
        assert clock.domainlocx_m == seconds * 12.5
        assert clock.domainlocy_m == seconds * 3.0
    # Two hours at 12.5 m/s is 90 km, to the metre.
    assert clock.domainlocx_m == 90_000.0
    assert clock.domainlocy_m == 21_600.0
    assert clock.model_seconds == 7_200.0


@pytest.mark.parametrize(
    "speed", [0.25, -0.125, 0.0625, 12.5, -7.5, 2.25, 150.0, 1.0 / 1024.0]
)
def test_exactness_holds_for_every_admitted_frame_speed(speed):
    """The proof is about binary32 significands, not about round numbers."""

    clock = RuntimeClock(config(u_move_m_s=speed, v_move_m_s=-speed))
    backend = RecordingBackend()
    for _ in range(14_400):
        clock.step(backend)
    assert clock.domainlocx_m == 7_200.0 * speed
    assert clock.domainlocy_m == 7_200.0 * -speed
    assert DOMAIN_LOCATION_EXACT_STEP_LIMIT > 14_400


def test_a_failed_step_never_advances_the_domain_location():
    """cm1.F:3510-3513 advance together, after the step has completed."""

    clock = RuntimeClock(config(u_move_m_s=12.5, v_move_m_s=3.0))
    clock.step(RecordingBackend())
    assert (clock.completed_steps, clock.domainlocx_m) == (1, 6.25)
    with pytest.raises((RuntimeError, ValueError)):
        clock.step(RecordingBackend(StepPhase.SOLVE3))
    assert (clock.completed_steps, clock.domainlocx_m) == (1, 6.25)


def test_the_clock_resumes_the_integral_rather_than_restarting_it():
    """restart_read.F:419-420 reads the pair back into an inout pair.

    An accumulator that restarted at zero would mislocate every frame after
    the resume by the whole distance the domain had already travelled.
    """

    settings = config(u_move_m_s=12.5, v_move_m_s=3.0)
    backend = RecordingBackend()

    whole = RuntimeClock(settings)
    for _ in range(14_400):
        whole.step(backend)

    first = RuntimeClock(settings)
    for _ in range(7_200):
        first.step(backend)
    resumed = RuntimeClock.resume(
        settings,
        completed_steps=first.completed_steps,
        domainlocx_m=first.domainlocx_m,
        domainlocy_m=first.domainlocy_m,
    )
    assert resumed.completed_steps == 7_200
    for _ in range(7_200):
        resumed.step(backend)
    assert resumed.domainlocx_m == whole.domainlocx_m
    assert resumed.domainlocy_m == whole.domainlocy_m
    # The defect this closes: a reset accumulator is half the distance out.
    assert resumed.domainlocx_m != whole.domainlocx_m / 2.0


def test_a_resumed_value_that_is_not_the_integral_is_refused_by_name():
    """A persisted pair from another frame or another step count is not it."""

    settings = config(u_move_m_s=12.5, v_move_m_s=3.0)
    with pytest.raises(ValueError, match="restored domainlocx"):
        RuntimeClock.resume(
            settings, completed_steps=7_200, domainlocx_m=1.0, domainlocy_m=10_800.0
        )
    with pytest.raises(ValueError, match="wrong patch of ground"):
        RuntimeClock.resume(
            settings, completed_steps=7_200, domainlocx_m=45_000.0, domainlocy_m=0.0
        )
    with pytest.raises(ValueError, match="must be finite"):
        RuntimeClock.resume(
            settings, completed_steps=0, domainlocx_m=float("nan"), domainlocy_m=0.0
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        RuntimeClock.resume(
            settings, completed_steps=-1, domainlocx_m=0.0, domainlocy_m=0.0
        )


def test_a_frame_speed_that_is_not_a_cm1_real_is_refused_by_key():
    """base.F:2667 subtracts a binary32 word; cm1.F:3512 must integrate it.

    A speed that does not survive binary32 would be subtracted from the base
    winds as one number and integrated as another, and the published domain
    location would walk away from the carriers it describes.
    """

    for name in ("u_move_m_s", "v_move_m_s"):
        with pytest.raises(ValueError, match=f"{name}=12.3 m/s is not a CM1 `real`"):
            config(**{name: 12.3})
    assert cm1_real(12.5) == 12.5
    assert cm1_real(12.3) != 12.3
    # The admitted speeds are exactly what CM1 would carry.
    assert config(u_move_m_s=12.5).record()["u_move_m_s"] == 12.5


def test_the_domain_location_record_uses_cm1s_names_units_and_descriptions():
    """writeout.F:9634-9648, domaindiag.F:464-483, writeout_nc.F:283-289."""

    clock = RuntimeClock(config(u_move_m_s=12.5, v_move_m_s=3.0))
    backend = RecordingBackend()
    for _ in range(120):
        clock.step(backend)
    record = clock.domain_location_record()
    assert record["domainlocx"] == 750.0 and record["domainlocy"] == 180.0
    assert record["umove"] == 12.5 and record["vmove"] == 3.0
    assert record["imove"] == 1
    assert record["completed_steps"] == 120 and record["model_time_s"] == 60.0
    assert record["units"] == {
        "umove": "m/s", "vmove": "m/s", "domainlocx": "m", "domainlocy": "m",
    }
    assert record["descriptions"]["domainlocx"] == "x location of (center of) domain"
    assert record["descriptions"]["domainlocy"] == "y location of (center of) domain"
    assert "cm1.F:348-349,3512-3513" in record["authority"]
    assert record["cadence"] == "post-large-step-after-model-time"


def test_the_exactness_guard_is_the_rust_proof_and_fires_on_drift():
    """A drifted accumulator stops the run instead of publishing a location.

    The value cannot drift under this lane's own dt, so the guard is provoked
    directly: it is the check that keeps a future widened timestep or a
    longer run from publishing ground positions that walk away from umove*t.
    """

    clock = RuntimeClock(config(u_move_m_s=12.5, v_move_m_s=3.0))
    clock.step(RecordingBackend())
    clock.domainlocx_m = 6.25 + 1e-9
    with pytest.raises(ValueError, match="wrong patch of ground"):
        clock.step(RecordingBackend())

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "rust/cm1-state/src/domain_location.rs"
    ).read_text(encoding="utf-8")
    # One step limit, two languages, so the two guards fire together.
    assert "pub const DOMAIN_LOCATION_EXACT_STEP_LIMIT: u64 = 1 << 29;" in source
    assert DOMAIN_LOCATION_EXACT_STEP_LIMIT == 1 << 29
    # The Rust and Python names for the pair are one set.
    for name in ("domainlocx", "domainlocy", "umove", "vmove"):
        assert f'"{name}"' in source


# ---------------------------------------------------------------------------
# CM1's advective Courant number: misclibs.F:2726-2814 and its 2714 stop.


def test_advective_courant_is_cm1s_own_three_component_norm():
    """``misclibs.F:2763-2765``: sqrt of the summed squared cell Courants.

    CM1 multiplies ``0.5*dt*rdx`` by ``ua(i)+ua(i+1)``, so the factor of two
    cancels and each cell contributes ``u*dt/dx``.
    """

    value = cm1_advective_courant(
        dt_s=0.5, dx_m=50.0, dy_m=50.0, dz_m=25.0,
        u_m_s=30.0, v_m_s=40.0, w_m_s=0.0,
    )
    assert value == pytest.approx(0.5 * 50.0 / 50.0)
    vertical = cm1_advective_courant(
        dt_s=0.5, dx_m=50.0, dy_m=50.0, dz_m=25.0,
        u_m_s=0.0, v_m_s=0.0, w_m_s=50.0,
    )
    assert vertical == pytest.approx(0.5 * 50.0 / 25.0)
    # A tornado-scale run is exactly the case the pinned half second could not
    # serve: 80 m/s through a 20 m cell is past CM1's stop at dt=0.5 s and
    # comfortably inside it at 0.05 s.
    assert cm1_advective_courant(
        dt_s=0.5, dx_m=20.0, dy_m=20.0, dz_m=20.0,
        u_m_s=80.0, v_m_s=0.0, w_m_s=0.0,
    ) == pytest.approx(2.0)
    assert cm1_advective_courant(
        dt_s=0.05, dx_m=20.0, dy_m=20.0, dz_m=20.0,
        u_m_s=80.0, v_m_s=0.0, w_m_s=0.0,
    ) == pytest.approx(0.2)


def test_cm1_stop_and_adaptive_target_are_cm1s_own_numbers():
    assert CM1_COURANT_STOP == 1.5
    assert CM1_COURANT_TARGET == 1.0


def test_courant_limited_speed_is_the_worst_direction():
    """The published capacity must bound the Courant number in every direction."""

    import math

    dt, dx, dy, dz = 0.5, 100.0, 100.0, 50.0
    speed = cm1_courant_limited_speed_m_s(dt_s=dt, dx_m=dx, dy_m=dy, dz_m=dz)
    # The smallest spacing decides: a purely vertical wind of exactly this
    # magnitude reaches the stop value, and no other direction exceeds it.
    assert speed == pytest.approx(CM1_COURANT_STOP * dz / dt)
    for u, v, w in (
        (speed, 0.0, 0.0),
        (0.0, speed, 0.0),
        (speed / math.sqrt(2.0), speed / math.sqrt(2.0), 0.0),
        (speed / math.sqrt(3.0),) * 3,
    ):
        assert cm1_advective_courant(
            dt_s=dt, dx_m=dx, dy_m=dy, dz_m=dz, u_m_s=u, v_m_s=v, w_m_s=w
        ) <= CM1_COURANT_STOP + 1e-12
    assert cm1_advective_courant(
        dt_s=dt, dx_m=dx, dy_m=dy, dz_m=dz, u_m_s=0.0, v_m_s=0.0, w_m_s=speed
    ) == pytest.approx(CM1_COURANT_STOP)
    assert cm1_courant_limited_speed_m_s(
        dt_s=dt, dx_m=dx, dy_m=dy, dz_m=dz, courant=CM1_COURANT_TARGET
    ) < speed
    # Halving the step doubles the wind the same grid can carry.
    assert cm1_courant_limited_speed_m_s(
        dt_s=dt / 2, dx_m=dx, dy_m=dy, dz_m=dz
    ) == pytest.approx(2.0 * speed)


def test_advective_courant_defect_names_cm1s_stop():
    assert advective_courant_defect(courant=1.0, dt_s=0.5) is None
    assert advective_courant_defect(courant=1.4999, dt_s=0.5) is None
    message = advective_courant_defect(courant=1.5, dt_s=0.5)
    assert message is not None
    # cm1.F:3020-3023 is not authority here: it is adapt_dt=1-only,
    # #ifdef MPI-only, and cm1.F:3035 clears its stopit before any reader.
    assert "misclibs.F:2714" in message and "statpack.F:469" in message
    assert "cm1.F:4385-4392" in message
    assert "cm1.F:3023" not in message and "cm1.F:3020" not in message
    assert "diverge instead of forecasting" in message
    blown = advective_courant_defect(courant=float("nan"), dt_s=0.5)
    assert blown is not None and "non-finite" in blown


def test_runtime_record_publishes_the_courant_capacity_of_the_declared_step():
    record = config().record()
    assert record["advective_courant_stop"] == CM1_COURANT_STOP
    assert record["advective_courant_target"] == CM1_COURANT_TARGET
    assert record["advective_courant_stop_speed_m_s"] > (
        record["advective_courant_target_speed_m_s"]
    )
    assert "misclibs.F:2726-2814" in record["advective_courant_authority"]
    # The published capacity must agree with the norm it inverts, in the
    # direction that reaches the stop value first.
    speed = record["advective_courant_stop_speed_m_s"]
    g = config().grid
    assert cm1_advective_courant(
        dt_s=0.5, dx_m=g.dx_m, dy_m=g.dy_m, dz_m=g.dz_m,
        u_m_s=speed, v_m_s=0.0, w_m_s=0.0,
    ) == pytest.approx(CM1_COURANT_STOP)


# ---------------------------------------------------------------------------
# The default-REAL word the large step is applied as.


def test_only_an_unusable_binary32_step_is_rejected():
    assert positive_normal_f32_bits(0.5) == 0x3F000000
    assert positive_normal_f32_bits(CM1_MIN_NORMAL_DT_S) == 0x00800000
    for unusable in (0.0, -0.5, float("nan"), float("inf"), 1e-40, True, "0.5", None):
        assert positive_normal_f32_bits(unusable) is None
    assert CM1_MIN_NORMAL_DT_S == pytest.approx(1.1754943508222875e-38)


def test_no_file_cites_cm1s_dead_adaptive_courant_block_as_authority():
    """``cm1.F:3020-3023`` stops nothing, so it may never stand as authority.

    Verified line by line against the read-only release tree: the block sits
    inside ``get_adapt_dt: if( adapt_dt.eq.1 )`` (cm1.F:2947) and
    ``if( .not. restarted )`` (cm1.F:3013), which this port never enters
    because ``dynamics.adaptive_timestep=true`` is refused at the door; its
    stop at ``cflmax`` 1.50 is inside ``#ifdef MPI``
    (cm1.F:3021/3025), so a serial build never compiles it; and cm1.F:3035
    clears ``stopit`` twelve lines later, before the first reader at
    cm1.F:3517.  It was cited across source, refusal text, records, docs and a
    shipped example.  Every surviving mention must say what it is not.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for directory in ("src", "docs", "examples", "tests", "tools"):
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.suffix not in {
                ".py", ".md", ".toml", ".csv", ".rs", ".txt"
            }:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if "cm1.F:3020" not in line and "cm1.F:3023" not in line:
                    continue
                if "not" in line.lower():
                    continue
                offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
    assert offenders == [], (
        "a dead CM1 branch is cited as the advective Courant authority: "
        f"{offenders}"
    )


def test_the_courant_authority_names_the_reachable_stop():
    assert "misclibs.F:2714" in CM1_COURANT_STOP_AUTHORITY
    assert "statpack.F:469" in CM1_COURANT_STOP_AUTHORITY
    assert "cm1.F:3517-3520" in CM1_COURANT_STOP_AUTHORITY
    assert "cm1.F:4385-4392" in CM1_COURANT_STOP_AUTHORITY
    assert "3020" not in CM1_COURANT_STOP_AUTHORITY
    assert "3023" not in CM1_COURANT_STOP_AUTHORITY
    assert config().record()["advective_courant_authority"] == (
        CM1_COURANT_STOP_AUTHORITY
    )
