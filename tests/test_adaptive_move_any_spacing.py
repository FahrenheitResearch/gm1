"""CM1's adaptive moving domain at spacings other than the one it was sealed on.

Three things are proved here without a device:

* the frame clock stays CM1's own binary64 recurrence after the controller
  changes ``umove`` -- on a 100 m grid at ``dt = 0.5 s`` -- and the fixed-frame
  ``n*dt*umove`` proof is deliberately not applied to it;
* the shipped CUDA kernel text, compiled by a host C++ compiler, reproduces the
  Python source-order oracle raw word for raw word at 50, 100 and 250 m, on both
  lateral-halo branches, on both origins, across several cadences;
* the independent Fortran transcription agrees with that oracle at 100 and
  250 m too.
"""

from __future__ import annotations

import shutil
from fractions import Fraction

import numpy as np
import pytest

from cm1core.cartesian_adaptive_frame import cm1_reference_axis
from cm1core.cuda_adaptive_move import initialize_adaptive_move_state
from cm1core.cuda_adaptive_move_open import (
    OPEN_HISTORY_FIELDS,
    cpu_advance_adaptive_move_open_after_large_step,
)
from cm1core.gm1_runtime import RuntimeClock, RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


class _HostController:
    """CM1's controller on the host, called where the device frame is called."""

    def __init__(self, clock: RuntimeClock, *, spacing: float, dt_s: float,
                 frq_s: float, steps_per_frame: int) -> None:
        grid = clock.config.grid
        self.clock = clock
        self.dt_s = dt_s
        self.frq_s = frq_s
        self.steps_per_frame = steps_per_frame
        self.levels = grid.levels
        ni, nj, nk = grid.nx, grid.ny, 4
        self.xfref, self.xh = cm1_reference_axis(0.0, spacing, ni)
        self.yfref, self.yh = cm1_reference_axis(0.0, spacing, nj)
        half = np.float32(0.5)
        self.centerx = float(self.xfref[0] + half * (self.xfref[-1] - self.xfref[0]))
        self.centery = float(self.yfref[0] + half * (self.yfref[-1] - self.yfref[0]))
        self.state = initialize_adaptive_move_state(
            ni=ni, nj=nj, centerx=self.centerx, centery=self.centery
        )
        self.winds = {
            "u0": np.zeros((nk, nj, ni + 1), np.float32),
            "ua": np.zeros((nk, nj, ni + 1), np.float32),
            "u3d": np.zeros((nk, nj, ni + 1), np.float32),
            "ug": np.zeros(nk, np.float32),
            "v0": np.zeros((nk, nj + 1, ni), np.float32),
            "va": np.zeros((nk, nj + 1, ni), np.float32),
            "v3d": np.zeros((nk, nj + 1, ni), np.float32),
            "vg": np.zeros(nk, np.float32),
        }
        self.histories = {
            "radbcw": np.zeros((nk + 2, nj + 6), np.float32),
            "radbce": np.zeros((nk + 2, nj + 6), np.float32),
            "radbcs": np.zeros((nk + 2, ni + 6), np.float32),
            "radbcn": np.zeros((nk + 2, ni + 6), np.float32),
        }
        assert set(self.histories) == set(OPEN_HISTORY_FIELDS)
        # A low east-north-east of the domain centre, so the controller has a
        # departure to correct and changes umove/vmove at every cadence.
        j, i = np.indices((nj, ni), dtype=np.float64)
        r2 = (i - 0.72 * ni) ** 2 + (j - 0.61 * nj) ** 2
        self.pressure = np.ascontiguousarray(
            -1.0e-3 * np.maximum(0.0, 1.0 - r2 / 64.0) ** 2, dtype=np.float32
        )
        self.speeds_used: list[tuple[float, float]] = []

    def after_large_step(self) -> None:
        nstep = int(self.clock.completed_steps)
        self.speeds_used.append((self.state.umove, self.state.vmove))
        cpu_advance_adaptive_move_open_after_large_step(
            self.state, self.winds, self.histories,
            nstep=nstep, mtime_after=float(self.clock.model_seconds),
            dbldt=self.dt_s, adapt_move_frq=self.frq_s,
            centerx=self.centerx, centery=self.centery,
            pressure_lowest=self.pressure, xh=self.xh, yh=self.yh,
            xfref=self.xfref, yfref=self.yfref,
            dowriteout=nstep % self.steps_per_frame == 0,
            vertical_grid_proof=self.levels,
        )

    @property
    def u_move_m_s(self) -> float:
        return self.state.umove

    @property
    def v_move_m_s(self) -> float:
        return self.state.vmove

    @property
    def domainlocx_m(self) -> float:
        return self.state.domainlocx

    @property
    def domainlocy_m(self) -> float:
        return self.state.domainlocy


class _Backend:
    def execute(self, operation, *, step):  # noqa: ANN001
        return None

    def synchronize(self) -> None:
        return None


def test_the_frame_clock_after_adaptive_updates_on_a_100_m_grid_is_cm1_s_own_sum() -> None:
    """cm1.F:3512-3513 after azimavg.F:4431-4435 has changed umove, at dt = 0.5."""

    spacing, dt_s, frq_s = 100.0, 0.5, 60.0
    grid = CartesianGridGeometry(nx=48, ny=40, nz=8, dx_m=spacing, dy_m=spacing, dz_m=250.0)
    config = RuntimeConfig(grid, dt_s=dt_s, u_move_m_s=0.0, v_move_m_s=0.0)
    clock = RuntimeClock(config)
    frame = _HostController(clock, spacing=spacing, dt_s=dt_s, frq_s=frq_s,
                            steps_per_frame=120)
    clock.adaptive_frame = frame
    backend = _Backend()
    steps = 6 * 120 + 37
    for _ in range(steps):
        clock.step(backend)

    speeds = {u for u, _ in frame.speeds_used}
    # Six corrections after the stationary start: the frame velocity really
    # did change, several times, to words that are not simple fractions.
    assert len(speeds) >= 6
    assert frame.state.umove > 0.0 and frame.state.vmove > 0.0

    # CM1's own recurrence, recomputed independently in binary64.
    x = y = 0.0
    exact_x = exact_y = Fraction(0)
    for u, v in frame.speeds_used:
        increment_x = dt_s * float(np.float32(u))
        increment_y = dt_s * float(np.float32(v))
        # dbldt*dble(umove) is exact at dt = 0.5: a 24-bit significand halved.
        assert Fraction(increment_x) == Fraction(dt_s) * Fraction(u)
        assert Fraction(increment_y) == Fraction(dt_s) * Fraction(v)
        x += increment_x
        y += increment_y
        exact_x += Fraction(increment_x)
        exact_y += Fraction(increment_y)
    assert clock.domainlocx_m == x and clock.domainlocy_m == y
    # On this run the running sum also happens to round nothing -- but that is
    # a fact about these words, not a proof, and nothing relies on it.
    assert Fraction(x) == exact_x and Fraction(y) == exact_y

    # The fixed-frame proof would have refused this run: n*dt*umove is not the
    # integral of a velocity that changed.  The adaptive branch does not apply
    # it, and the publication reads the controller's own umove.
    assert clock.domainlocx_m != clock.completed_steps * clock.domainlocx_increment_m
    record = clock.domain_location_record()
    assert record["umove"] == frame.state.umove
    assert record["domainlocx"] == frame.state.domainlocx

    # A resume carries the controller, so the restored integral is continued
    # rather than checked against a constant frame it never had ...
    resumed = RuntimeClock.resume(
        config, completed_steps=steps, domainlocx_m=x, domainlocy_m=y,
        adaptive_frame=frame,
    )
    assert resumed.domainlocx_m == x
    # ... and without it the same pair is refused by the fixed-frame proof,
    # which is why cartesian_restart.py hands the controller to the resume.
    with pytest.raises(ValueError, match="exact integral"):
        RuntimeClock.resume(
            config, completed_steps=steps, domainlocx_m=x, domainlocy_m=y
        )


@pytest.mark.skipif(shutil.which("g++") is None, reason="no host C++ compiler")
def test_the_shipped_kernel_text_matches_the_oracle_at_every_spacing(tmp_path) -> None:
    from tools.oracle.adaptive_move_host_oracle import SPACINGS_M, verify_host_kernels

    receipt = verify_host_kernels(tmp_path)
    assert receipt["status"] == "passed-host-raw-word-exact"
    cases = receipt["cases"]
    assert {case["spacing_m"] for case in cases} == set(SPACINGS_M)
    assert {case["branch"] for case in cases} == {"periodic", "radiative-open"}
    assert {case["origin"] for case in cases} == {"corner", "centred"}
    for case in cases:
        assert case["raw_word_exact"] is True
        assert case["vertical_decks_identical"] is True
        assert case["centre_within_one_face"] is True
    # The smoothed pressure and the centre's face index do not depend on the
    # spacing: the same pressure words give the same psmth at 50, 100 and 250 m.
    by_key: dict[tuple[str, str], set[str]] = {}
    for case in cases:
        by_key.setdefault((case["branch"], case["origin"]), set()).add(
            case["first_psmth_sha256"]
        )
    assert all(len(values) == 1 for values in by_key.values())
    assert receipt["lowest_scalar_levels_m"] == {
        "stretch_z1_20m_bottom": 10.0,
        "stretch_z1_50m_bottom": 25.0,
    }


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="no gfortran")
def test_the_independent_fortran_transcription_agrees_at_100_and_250_m(tmp_path) -> None:
    from tools.oracle.adaptive_move_host_oracle import verify_fortran_spacings

    receipt = verify_fortran_spacings(tmp_path)
    assert receipt["status"] == "passed-native-raw-word-exact"
    assert {(row["branch"], row["spacing_m"]) for row in receipt["comparisons"]} == {
        (branch, spacing)
        for branch in ("periodic", "radiative-open")
        for spacing in (50.0, 100.0, 250.0)
    }
    assert all(row["passed"] for row in receipt["comparisons"])
