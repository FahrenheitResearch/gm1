"""CM1's ``param2``/``param3`` dynamical words, from TOML text to behaviour.

Every test here follows one CM1 namelist word from a document, through the
configuration door and the runtime contract, to the arithmetic it changes.
The point of the file is that the path exists: before it, ``icor``, ``fcor``,
``betaplane``, ``alph`` and ``kdiv`` had no key at all, and the CUDA operator
that implements CM1's Coriolis term could be reached by no lane on any grid.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from cm1core.cartesian_door import build_runtime_config
from cm1core.config import load_config, parse_mapping
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
CARTESIAN = EXAMPLES / "cartesian-uniform-grid.toml"
TRANSLATING = EXAMPLES / "cartesian-supercell-translating-frame.toml"


def _mapping(path: Path = CARTESIAN) -> dict[str, object]:
    payload = load_config(path).canonical_dict()
    return {
        section: payload[section]
        for section in payload
        if section in {"run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime", "initialization"}
    }


# ---------------------------------------------------------------------------
# icor / fcor / betaplane


def test_absent_dynamics_words_bind_cm1s_own_distributed_values() -> None:
    """run/namelist.input:29-30,42-44,59-60,92-96, by value."""

    d = load_config(CARTESIAN).dynamics
    assert (d.coriolis, d.beta_plane, d.fcor_s_inv) == (0, 0, 5.0e-5)
    assert (d.alph, d.kdiv) == (0.60, 0.10)
    assert (d.idiff, d.mdiff, d.difforder) == (0, 0, 6)
    assert (d.kdiff2, d.kdiff6) == (75.0, 0.040)


def test_coriolis_reaches_the_runtime_contract_by_cm1s_own_names() -> None:
    raw = _mapping()
    raw["dynamics"]["coriolis"] = 1  # type: ignore[index]
    raw["dynamics"]["fcor_s_inv"] = 1.0e-4  # type: ignore[index]
    runtime = build_runtime_config(parse_mapping(raw))
    assert (runtime.icor, runtime.betaplane, runtime.fcor_s_inv) == (1, 0, 1.0e-4)


def test_beta_plane_without_coriolis_is_refused_by_key_and_value() -> None:
    raw = _mapping()
    raw["dynamics"]["beta_plane"] = 1  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-BETAPLANE-WITHOUT-CORIOLIS"
    assert caught.value.detail.option == "dynamics.beta_plane"
    assert "solve2.F:632" in caught.value.detail.reason


def test_beta_plane_outside_asin_domain_is_refused_before_it_makes_nan() -> None:
    raw = _mapping()
    raw["dynamics"]["coriolis"] = 1  # type: ignore[index]
    raw["dynamics"]["beta_plane"] = 1  # type: ignore[index]
    raw["dynamics"]["fcor_s_inv"] = 2.0e-4  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-BETAPLANE-LATITUDE"
    assert "asin" in caught.value.detail.reason


def test_the_pole_is_admitted_and_one_ulp_past_it_is_not() -> None:
    """2*omega is the Coriolis parameter at the pole; asin(1) is defined."""

    from cm1core.gm1_runtime import CM1_MAX_ABS_FCOR_S_INV

    raw = _mapping()
    raw["dynamics"]["coriolis"] = 1  # type: ignore[index]
    raw["dynamics"]["beta_plane"] = 1  # type: ignore[index]
    raw["dynamics"]["fcor_s_inv"] = CM1_MAX_ABS_FCOR_S_INV  # type: ignore[index]
    assert parse_mapping(raw).dynamics.beta_plane == 1
    raw["dynamics"]["fcor_s_inv"] = math.nextafter(  # type: ignore[index]
        CM1_MAX_ABS_FCOR_S_INV, math.inf
    )
    with pytest.raises(CM1Refusal):
        parse_mapping(raw)


def test_beta_plane_in_a_translating_frame_is_refused_with_the_error_it_makes() -> None:
    """solve2.F:667-672 omits the add-back solve2.F:645,647 makes.

    This is the one combination the f-plane branch gets right and the beta
    branch does not, and the refusal has to say how wrong it would be rather
    than that it is unsupported.
    """

    raw = _mapping(TRANSLATING)
    assert raw["closures"]["moving_frame"] is True  # type: ignore[index]
    raw["dynamics"]["coriolis"] = 1  # type: ignore[index]
    raw["dynamics"]["beta_plane"] = 1  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-BETAPLANE-TRANSLATING-FRAME"
    # 5e-5 * 3.0 m/s and 5e-5 * 12.5 m/s, the deck's own frame.
    assert "1.500e-04" in detail.reason and "6.250e-04" in detail.reason
    assert "beta_plane = 0" in detail.remedy


def test_the_f_plane_in_a_translating_frame_is_admitted_and_carries_the_frame() -> None:
    """icor=1 with imove=1 is CM1-correct, so it must not be refused."""

    raw = _mapping(TRANSLATING)
    raw["dynamics"]["coriolis"] = 1  # type: ignore[index]
    runtime = build_runtime_config(parse_mapping(raw))
    assert runtime.icor == 1
    assert (runtime.u_move_m_s, runtime.v_move_m_s) == (12.5, 3.0)


# ---------------------------------------------------------------------------
# The rotation operator itself


def test_rotation_selection_built_from_a_document_carries_the_frame_add_back() -> None:
    """The f-plane kernel scalar is fcor, umove and vmove from the document.

    ``cpu_rotation_forcing_inplace`` is the independently evidenced CM1
    operator; driving it with the selection the door produced is the shortest
    honest statement that the document reaches CM1's arithmetic.
    """

    from cm1core.rotation_forcing import (
        RotationForcingSelection,
        cpu_rotation_forcing_inplace,
    )

    raw = _mapping(TRANSLATING)
    raw["dynamics"]["coriolis"] = 1  # type: ignore[index]
    raw["dynamics"]["fcor_s_inv"] = 1.0e-4  # type: ignore[index]
    runtime = build_runtime_config(parse_mapping(raw))
    selection = RotationForcingSelection(
        icor=runtime.icor,
        beta_plane=runtime.betaplane,
        axisymmetric=0,
        fcor_s_inv=runtime.fcor_s_inv,
        u_move_m_s=runtime.u_move_m_s,
        v_move_m_s=runtime.v_move_m_s,
    )
    ni = nj = nk = 4
    arrays = {
        "u3d_window": np.zeros((nk, nj + 2, ni + 2), dtype=np.float32),
        "v3d_window": np.zeros((nk, nj + 2, ni + 2), dtype=np.float32),
        "uten": np.zeros((nk, nj + 1, ni + 1), dtype=np.float32),
        "vten": np.zeros((nk, nj + 1, ni + 1), dtype=np.float32),
        "yh_window": np.zeros((nj + 2,), dtype=np.float32),
        "f2d": np.empty((nj + 2, ni + 2), dtype=np.float32),
    }
    cpu_rotation_forcing_inplace(arrays, ni=ni, nj=nj, nk=nk, selection=selection)
    # A frame-relative wind of exactly zero is a ground-relative wind of
    # (umove, vmove), so solve2.F:645,647 leave fcor*vmove and -fcor*umove.
    assert np.allclose(arrays["uten"], np.float32(1.0e-4 * 3.0))
    assert np.allclose(arrays["vten"], -np.float32(1.0e-4 * 12.5))


def test_icor0_leaves_the_translating_frame_exactly_galilean() -> None:
    """The whole correctness argument for imove=1 rests on this.

    With no Coriolis term the momentum equations contain no absolute
    velocity, so subtracting a constant from every wind is an exact change of
    variable.  The operator is refused for icor=0 rather than run with a zero
    coefficient, so nothing can be added to a tendency at all.
    """

    from cm1core.rotation_forcing import RotationForcingSelection, cpu_rotation_forcing_inplace

    with pytest.raises(CM1Refusal) as caught:
        cpu_rotation_forcing_inplace(
            {}, ni=4, nj=4, nk=4,
            selection=RotationForcingSelection(
                icor=0, beta_plane=0, axisymmetric=0, fcor_s_inv=1.0e-4
            ),
        )
    assert caught.value.detail.option == "icor"


def test_the_rotation_hook_is_enqueued_between_divx_and_advu() -> None:
    """solve2.F puts the Coriolis term after divx and before advu.

    The flat Cartesian lane's momentum operator had no seam there at all; the
    stretched child's did.  Both are checked, so neither can drift.
    """

    from cm1core import cuda_gm1_child_stretched_momentum as stretched
    from cm1core import cuda_momentum_advection5 as flat

    for source in (
        inspect.getsource(flat.launch_cuda_terrain_open_momentum_advection5),
        inspect.getsource(stretched.launch_cuda_gm1_child_stretched_momentum5),
    ):
        divergence = source.index("divergence(")
        hook = source.index("post_divergence_hook(", divergence)
        advu = source.index("advu(", hook)
        assert divergence < hook < advu


def test_the_cartesian_lane_passes_the_rotation_hook_only_under_icor1() -> None:
    from cm1core import cuda_cartesian_dynamics as dynamics

    pre = inspect.getsource(dynamics.CartesianDynamics.pre_acoustic)
    assert "post_divergence_hook=self._rotation_hook(nrk)" in pre
    bind = inspect.getsource(dynamics.CartesianDynamics._bind_rotation)
    assert 'if getattr(config, "icor", 0) != 1:' in bind
    hook = inspect.getsource(dynamics.CartesianDynamics._rotation_hook)
    assert "if binding is None:" in hook and "return None" in hook


# ---------------------------------------------------------------------------
# alph / kdiv


def test_alph_default_is_cm1s_060_and_reaches_the_acoustic_launch() -> None:
    from cm1core import cuda_cartesian_dynamics as dynamics
    from cm1core.cuda_psolver3_terrain_open import CM1_ALPH_F32, CM1_KDIV_F32

    # README.namelist:756-759, and the value all 23 shipped decks set.
    assert CM1_ALPH_F32 == 0.60
    assert CM1_KDIV_F32 == 0.10
    runtime = build_runtime_config(load_config(CARTESIAN))
    assert (runtime.alph, runtime.kdiv) == (0.60, 0.10)
    acoustic = inspect.getsource(dynamics.CartesianDynamics.acoustic)
    assert "alph=self.config.alph" in acoustic
    assert "kdiv=self.config.kdiv" in acoustic


def test_alph_is_carried_to_a_declared_value_and_recorded() -> None:
    raw = _mapping()
    raw["dynamics"]["alph"] = 0.75  # type: ignore[index]
    config = parse_mapping(raw)
    assert build_runtime_config(config).alph == 0.75
    assert config.canonical_dict()["dynamics"]["alph"] == 0.75


@pytest.mark.parametrize("name", ["alph", "kdiv"])
@pytest.mark.parametrize("bad", [-0.01, 1.5, math.nan])
def test_acoustic_weights_outside_the_unit_interval_are_refused(name: str, bad: float) -> None:
    raw = _mapping()
    raw["dynamics"][name] = bad  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-NUMERIC-RANGE"
    assert caught.value.detail.option == f"dynamics.{name}"


# ---------------------------------------------------------------------------
# idiff


@pytest.mark.parametrize("idiff", [1, 2])
def test_explicit_diffusion_is_refused_by_naming_the_missing_branch(idiff: int) -> None:
    raw = _mapping()
    raw["dynamics"]["idiff"] = idiff  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-CARTESIAN-EXPLICIT-DIFFUSION"
    assert detail.option == "dynamics.idiff"
    assert "no difforder branch" in detail.reason
    assert "idiff = 0" in detail.remedy


def test_the_diffusion_words_are_admitted_and_stay_out_of_the_identity() -> None:
    """They are configurable, and inert at idiff=0, so they change nothing."""

    raw = _mapping()
    before = parse_mapping(raw).sha256
    raw["dynamics"]["difforder"] = 2  # type: ignore[index]
    raw["dynamics"]["kdiff2"] = 120.0  # type: ignore[index]
    raw["dynamics"]["mdiff"] = 1  # type: ignore[index]
    after = parse_mapping(raw)
    assert (after.dynamics.difforder, after.dynamics.kdiff2, after.dynamics.mdiff) == (2, 120.0, 1)
    assert after.sha256 == before


# ---------------------------------------------------------------------------
# The analytic inertial oscillation
#
# A storm run is the wrong test for Coriolis: 1/f is about three hours, so a
# five-minute integration barely turns the wind.  The right test is the one
# case with a closed-form answer.  On an f plane with no pressure gradient and
# a horizontally uniform wind, advection and the pressure term vanish
# identically and solve2.F:639-653 is the whole momentum equation:
#
#     du/dt =  f v        u(t) = u0 cos(ft) + v0 sin(ft)
#     dv/dt = -f u        v(t) = v0 cos(ft) - u0 sin(ft)
#
# an anticyclonic circle of period 2*pi/f.  Integrating CM1's own operator
# with CM1's own RK3 stage weights and comparing against that closed form
# checks the sign, the magnitude and the staggering in one number, at a cost
# of no GPU time at all.


def _rk3_inertial_step(u0: float, v0: float, *, fcor: float, dt: float) -> tuple[float, float]:
    """One CM1 RK3 large step of the pure rotation problem.

    ``cm1.F``'s Wicker-Skamarock RK3 takes dt/3, then dt/2, then dt, each
    from the state at the start of the large step with the tendency evaluated
    at the previous substage -- solve2.F:161-171 for the stage timestep.
    """

    from cm1core.rotation_forcing import (
        RotationForcingSelection,
        cpu_rotation_forcing_inplace,
    )

    selection = RotationForcingSelection(
        icor=1, beta_plane=0, axisymmetric=0, fcor_s_inv=fcor
    )
    ni = nj = nk = 1
    u, v = u0, v0
    for weight in (3.0, 2.0, 1.0):
        arrays = {
            "u3d_window": np.full((nk, nj + 2, ni + 2), u, dtype=np.float32),
            "v3d_window": np.full((nk, nj + 2, ni + 2), v, dtype=np.float32),
            "uten": np.zeros((nk, nj + 1, ni + 1), dtype=np.float32),
            "vten": np.zeros((nk, nj + 1, ni + 1), dtype=np.float32),
            "yh_window": np.zeros((nj + 2,), dtype=np.float32),
            "f2d": np.empty((nj + 2, ni + 2), dtype=np.float32),
        }
        cpu_rotation_forcing_inplace(arrays, ni=ni, nj=nj, nk=nk, selection=selection)
        stage_dt = dt / weight
        u = u0 + stage_dt * float(arrays["uten"][0, 0, 0])
        v = v0 + stage_dt * float(arrays["vten"][0, 0, 0])
    return u, v


def test_the_operator_integrates_the_analytic_inertial_oscillation() -> None:
    fcor, dt = 1.0e-4, 10.0
    quarter = math.pi / (2.0 * fcor)
    steps = round(quarter / dt)
    u, v = 20.0, 0.0
    for _ in range(steps):
        u, v = _rk3_inertial_step(u, v, fcor=fcor, dt=dt)
    angle = fcor * steps * dt
    expect_u = 20.0 * math.cos(angle)
    expect_v = -20.0 * math.sin(angle)
    # A quarter period: the wind has turned from due east to due south, which
    # is the anticyclonic sense f > 0 must produce.
    assert abs(v) > 19.9 and v < 0.0
    speed = math.hypot(u, v)
    # RK3 is third-order, so over 1571 steps at f*dt = 1e-3 the amplitude and
    # phase error are far below the binary32 the tendency was formed in.
    assert abs(speed - 20.0) < 2.0e-3, speed
    assert abs(u - expect_u) < 2.0e-3
    assert abs(v - expect_v) < 2.0e-3


def test_the_rotation_conserves_speed_over_a_full_inertial_period() -> None:
    """The exact solution is a rotation, so |(u,v)| is invariant."""

    fcor, dt = 1.0e-4, 10.0
    steps = round(2.0 * math.pi / (fcor * dt))
    u, v = 12.0, -5.0
    speed0 = math.hypot(u, v)
    for _ in range(steps):
        u, v = _rk3_inertial_step(u, v, fcor=fcor, dt=dt)
    assert abs(math.hypot(u, v) - speed0) / speed0 < 1.0e-5
    # ... and returns to where it started.
    assert abs(u - 12.0) < 5.0e-3 and abs(v + 5.0) < 5.0e-3


def test_a_southern_hemisphere_f_turns_the_wind_the_other_way() -> None:
    """fcor < 0 is admitted, and must reverse the sense of the circle."""

    fcor, dt = -1.0e-4, 10.0
    steps = round(math.pi / (2.0 * abs(fcor)) / dt)
    u, v = 20.0, 0.0
    for _ in range(steps):
        u, v = _rk3_inertial_step(u, v, fcor=fcor, dt=dt)
    assert v > 19.9
