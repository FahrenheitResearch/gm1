"""The child window on a rotating child: the header, the Coriolis terms, lspgrad=1, the closure.

CM1 adds the f-plane Coriolis force to the *stage* tendency (solve2.F:633-653),
so no word the window streams carries it; the budget re-forms it from the
record's own winds and the header's ``fcor`` and frame velocity.  Checked here:

* the force on every face is solve2.F:639-653's own float32 expression of the
  window's padded words (a literal loop over CM1's indices);
* a closed-form flow whose vorticity equation *includes* f closes at every
  point of every record to 1e-5 of its largest term -- with coriolis=1 and
  with coriolis=0 -- and does not without the Coriolis term;
* lspgrad=1's words (solve1.F:590-612), which the held tendency does carry,
  are removed from friction and published once, as ``lspg``;
* a non-rotating stream and its budget are the v2 words they always were.

The closed form is the stretching vortex of :mod:`test_nest_window_budget`
with the planetary vorticity added: ``u = -a x/2 - Omega y``, ``v = -a y/2 +
Omega x``, ``w = a z``, so ``D zeta/Dt = (zeta + f) dw/dz`` gives
``2 Omega + f = (2 Omega_0 + f) e^{a t}``.  Its Coriolis term is exactly
``-f (du/dx + dv/dy) = f a`` and its stretching ``zeta a``.
"""

from __future__ import annotations

import json
import math

import pytest

np = pytest.importorskip("numpy")

from cm1core import nest_child_window as ncw  # noqa: E402
from cm1core import nest_parcel_backward as nb  # noqa: E402
from cm1core import nest_window_budget as nwb  # noqa: E402

from test_nest_child_window import (  # noqa: E402
    NG,
    OMEGA_0,
    STRETCH_A,
    _admission,
    centre_of,
    child_arrays,
    child_config,
    open_stream,
    stream_root,
)

#: Far larger than the Earth's so the planetary terms are a visible share of
#: the budget (f a / (zeta a) = 5 %); the physical value is checked separately.
F_TEST = 5.0e-3
U_MOVE, V_MOVE = 16.5, -7.25


def omega_rotating(t: float, f: float) -> float:
    return 0.5 * ((2.0 * OMEGA_0 + f) * math.exp(STRETCH_A * t) - f)


def equivalent_time(t: float, f: float) -> float:
    """The time at which the non-rotating fixture's Omega equals the rotating one's."""

    return math.log(omega_rotating(t, f) / OMEGA_0) / STRETCH_A


def rotating_config(*, f=F_TEST, lspgrad=0, **overrides):
    values = dict(icor=1, betaplane=0, fcor_s_inv=f, lspgrad=lspgrad,
                  u_move_m_s=U_MOVE, v_move_m_s=V_MOVE, imove=1)
    values.update(overrides)
    return child_config(**values)


def u0_v0(deck):
    zh = np.asarray(deck.zh[1:deck.nz + 1], np.float64)
    return (np.asarray(2.0 + 0.01 * zh, np.float32), np.asarray(-1.0 + 0.02 * zh, np.float32))


STRESS_C = 3.0e-5     # s-2 per m: a held u stress linear in height, curl_y = STRESS_C


def arrays_at(deck, t, f, *, xc, yc, lspgrad=0):
    arrays = child_arrays(deck, equivalent_time(t, f) if f else t, xc=xc, yc=yc)
    if lspgrad:
        u0, v0 = u0_v0(deck)
        arrays["u0_profile"], arrays["v0_profile"] = u0, v0
        lu, lv = ncw.lspgrad1_profiles(fcor_s_inv=f, u_move_m_s=U_MOVE, v_move_m_s=V_MOVE,
                                       u0=u0, v0=v0)
        zh = np.asarray(deck.zh[1:deck.nz + 1], np.float32)
        # The held tendency as the child holds it: (Rayleigh - ul) + stress,
        # Rayleigh zero below the layer.
        arrays["uten"] = np.float32(lu[:, None, None] + np.float32(STRESS_C) * zh[:, None, None]) \
            * np.ones(arrays["uten"].shape, np.float32)
        arrays["vten"] = lv[:, None, None] * np.ones(arrays["vten"].shape, np.float32)
    return arrays


def write_rotating(stream, deck, *, steps, f, xc, yc, lspgrad=0):
    dt = stream.child_dt_s
    for n in range(1, steps + 1):
        t = n * dt
        stream.capture(arrays_at(deck, t - 0.5 * dt, f, xc=xc, yc=yc, lspgrad=lspgrad),
                       child_step=n, model_time_s=t)
    return stream.close()


def build(tmp_path, *, f=F_TEST, lspgrad=0, steps=24, name="rot"):
    config = rotating_config(f=f, lspgrad=lspgrad) if f else None
    g, deck, stream = open_stream(tmp_path, config=config, name=name)
    xc, yc = centre_of(g)
    receipt = write_rotating(stream, deck, steps=steps, f=f, xc=xc, yc=yc, lspgrad=lspgrad)
    return g, deck, ncw.read_window_stream(stream_root(tmp_path, name)), (xc, yc), receipt


def record_residual(opened, index, f):
    """``max |D omega/Dt - sum of terms| / max |term|`` over every point of one record."""

    record = opened.records[index]
    deck = opened.deck_for(record)
    out = nwb.record_budget(opened.fields(index), deck, rotation=opened.rotation)
    t_mid = record.model_time_s - 0.5 * opened.dt_s
    omega = omega_rotating(t_mid, f) if f else OMEGA_0 * math.exp(STRETCH_A * t_mid)
    rate = {"x": 0.0, "y": 0.0, "z": STRETCH_A * (2.0 * omega + f)}
    terms = [t for t in (*nwb.CLOSURE_TERMS, "coriolis", "lspg") if f"{t}_z" in out]
    worst, scale = 0.0, 0.0
    for axis in ("x", "y", "z"):
        total = sum(np.asarray(out[f"{t}_{axis}"], np.float64) for t in terms)
        worst = max(worst, float(np.abs(total - rate[axis]).max()))
        scale = max(scale, *(float(np.abs(out[f"{t}_{axis}"]).max()) for t in terms))
    return worst / scale, out, omega


# ---------------------------------------------------------------------------
# The door.
# ---------------------------------------------------------------------------
def test_f_plane_coriolis_is_admitted_and_only_the_beta_plane_and_other_lspgrads_refused():
    assert _admission(coriolis=1) is None
    assert _admission(coriolis=1, lspgrad=1) is None
    defect = _admission(coriolis=1, beta_plane=1)
    assert defect is not None and defect[0] == "CORIOLIS-BETA-PLANE"
    assert defect[1] == "dynamics.beta_plane" and "one fcor" in defect[2]
    defect = _admission(coriolis=1, lspgrad=2)
    assert defect is not None and defect[0] == "LSPGRAD" and "friction" in defect[2]


def test_a_beta_plane_child_is_refused_at_build_too(tmp_path):
    with pytest.raises(ncw.NestChildWindowError) as caught:
        open_stream(tmp_path, config=rotating_config(betaplane=1), name="beta")
    assert "one fcor" in str(caught.value)


def test_the_config_lspgrad_reader_takes_the_bound_value_and_absence_as_zero():
    from types import SimpleNamespace

    assert ncw._config_lspgrad(SimpleNamespace(coriolis=1)) == 0
    assert ncw._config_lspgrad(SimpleNamespace(lspgrad=None)) == 0
    assert ncw._config_lspgrad(SimpleNamespace(lspgrad=None, lspgrad_value=1)) == 1


# ---------------------------------------------------------------------------
# The header.
# ---------------------------------------------------------------------------
def test_a_rotating_stream_is_v3_and_its_header_names_fcor_and_the_frame(tmp_path):
    _g, _deck, opened, _c, receipt = build(tmp_path, steps=2, name="hdr")
    assert receipt["schema"] == ncw.NEST_CHILD_WINDOW_ROTATING_SCHEMA
    rotation = opened.rotation
    assert rotation["icor"] == 1 and rotation["betaplane"] == 0 and rotation["lspgrad"] == 0
    assert rotation["fcor_s_inv"] == float(np.float32(F_TEST))
    assert rotation["fcor_f32_bits"] == int(np.float32(F_TEST).view(np.uint32))
    assert (rotation["u_move_m_s"], rotation["v_move_m_s"]) == (U_MOVE, V_MOVE)
    assert "solve2.F:633-653" in rotation["cm1_authority"]["coriolis_force"]
    assert "lspg_u_m_s2" not in rotation
    assert "Coriolis force itself is not in it" in receipt["held_tendency"]


def test_a_non_rotating_stream_is_still_v2_with_no_rotation_header(tmp_path):
    _g, _deck, opened, _c, receipt = build(tmp_path, f=0.0, steps=2, name="v2")
    assert receipt["schema"] == ncw.NEST_CHILD_WINDOW_SCHEMA
    assert "rotation" not in receipt and opened.rotation is None


def test_a_rotating_morrison_stream_is_one_v3_carrying_species_and_rotation(tmp_path):
    """A Morrison child with icor=1 writes one v3 stream.

    The window's Coriolis and species changes both named their
    addition v3; merged, v3 carries either or both, each declared in the
    receipt, so the budget reads the rotation header and the species rebuild
    qhyd from the same stream.
    """

    from test_nest_child_window import MORRISON, _morrison_carriers

    config = rotating_config(f=F_TEST, lspgrad=0, microphysics=MORRISON)
    g, deck, stream = open_stream(tmp_path, config=config, name="both")
    xc, yc = centre_of(g)
    for n in (1, 2):
        stream.capture(_morrison_carriers(deck, 0.125 * n, xc=xc, yc=yc),
                       child_step=n, model_time_s=0.25 * n)
    receipt = stream.close()
    assert receipt["schema"] == ncw.NEST_CHILD_WINDOW_SCHEMA_V3
    assert receipt["schema_version"] == 3 and "species" in receipt
    assert receipt["rotation"]["icor"] == 1 and receipt["rotation"]["lspgrad"] == 0
    opened = ncw.read_window_stream(stream_root(tmp_path, "both"))
    assert opened.has_species and opened.rotation is not None
    words = opened.fields(0)
    assert {"qc", "qr", "qis", "qg", "qhyd"} <= set(words)
    path = stream_root(tmp_path, "both") / ncw.WINDOW_RECEIPT_FILENAME
    path.write_text(json.dumps({k: v for k, v in receipt.items() if k != "species"}))
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.read_window_stream(stream_root(tmp_path, "both"))
    assert "lists fields" in str(caught.value)


def test_a_non_rotating_morrison_stream_is_v3_with_species_and_no_rotation(tmp_path):
    from test_nest_child_window import MORRISON, _morrison_carriers, child_config as cc

    g, deck, stream = open_stream(tmp_path, config=cc(microphysics=MORRISON), name="sp")
    xc, yc = centre_of(g)
    stream.capture(_morrison_carriers(deck, 0.125, xc=xc, yc=yc), child_step=1,
                   model_time_s=0.25)
    receipt = stream.close()
    assert receipt["schema"] == ncw.NEST_CHILD_WINDOW_SCHEMA_V3 and "rotation" not in receipt
    opened = ncw.read_window_stream(stream_root(tmp_path, "sp"))
    assert opened.has_species and opened.rotation is None


def test_the_reader_refuses_a_schema_and_header_that_disagree(tmp_path):
    _g, _deck, _o, _c, _r = build(tmp_path, steps=2, name="bad")
    path = stream_root(tmp_path, "bad") / ncw.WINDOW_RECEIPT_FILENAME
    receipt = json.loads(path.read_text())
    stripped = {k: v for k, v in receipt.items() if k != "rotation"}
    path.write_text(json.dumps(stripped))
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.read_window_stream(stream_root(tmp_path, "bad"))
    assert "must carry a rotation header" in str(caught.value)
    path.write_text(json.dumps({**receipt, "schema": ncw.NEST_CHILD_WINDOW_SCHEMA}))
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.read_window_stream(stream_root(tmp_path, "bad"))
    assert "non-rotating by definition" in str(caught.value)
    path.write_text(json.dumps({**receipt, "rotation": {**receipt["rotation"], "lspgrad": 1}}))
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.read_window_stream(stream_root(tmp_path, "bad"))
    assert "lspg_u_m_s2" in str(caught.value)


# ---------------------------------------------------------------------------
# The force: solve2.F:639-653's words, on the window's own faces.
# ---------------------------------------------------------------------------
def test_the_coriolis_force_is_solve2s_own_float32_words_on_every_face():
    rng = np.random.default_rng(7)
    nz, ny, nx = 3, 4, 5
    box = ncw.WindowBox(0, 0, nx, ny, nz)
    shapes = box.shapes(surface=False)
    u = (rng.standard_normal(shapes["uadv"]) * 20.0).astype(np.float32)
    v = (rng.standard_normal(shapes["vadv"]) * 20.0).astype(np.float32)
    fcor, um, vm = np.float32(8.47e-5), np.float32(U_MOVE), np.float32(V_MOVE)
    fu, fv = nwb.coriolis_face_forces(u, v, fcor_s_inv=float(fcor), u_move_m_s=float(um),
                                      v_move_m_s=float(vm))
    assert fu.shape == shapes["uten"] and fv.shape == shapes["vten"]
    f = np.float32
    for k in range(nz + 1):
        for j in range(ny + 2):
            for i in range(nx + 3):
                # ring face (k, j, i) is padded (k+1, j-1+NG, i-1+NG)
                K, J, I = k + 1, j - 1 + NG, i - 1 + NG
                s = f(f(v[K, J, I] + v[K, J + 1, I]) + f(v[K, J, I - 1] + v[K, J + 1, I - 1]))
                want = f(fcor * f(f(f(0.25) * s) + vm))
                assert fu[k, j, i] == want, (k, j, i)
        for j in range(ny + 3):
            for i in range(nx + 2):
                K, J, I = k + 1, j - 1 + NG, i - 1 + NG
                s = f(f(u[K, J, I] + u[K, J, I + 1]) + f(u[K, J - 1, I] + u[K, J - 1, I + 1]))
                want = f(-f(fcor * f(f(f(0.25) * s) + um)))
                assert fv[k, j, i] == want, (k, j, i)


def test_mismatched_wind_blocks_are_refused():
    box = ncw.WindowBox(0, 0, 5, 4, 3)
    shapes = box.shapes(surface=False)
    with pytest.raises(nwb.TrajectoryBudgetError):
        nwb.coriolis_face_forces(np.zeros(shapes["uadv"], np.float32),
                                 np.zeros(shapes["uadv"], np.float32),
                                 fcor_s_inv=1e-4, u_move_m_s=0.0, v_move_m_s=0.0)


# ---------------------------------------------------------------------------
# The closure, point by point, against the closed form.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f", [F_TEST, 8.47e-5, 0.0])
def test_every_record_closes_to_1e_5_of_its_largest_term(tmp_path, f):
    _g, _deck, opened, (xc, yc), _r = build(tmp_path, f=f, steps=6, name=f"close{f}")
    for index in range(len(opened.records)):
        ratio, out, omega = record_residual(opened, index, f)
        assert ratio <= 1.0e-5, (index, ratio)
    if f:
        assert out["coriolis_z"] == pytest.approx(f * STRETCH_A, rel=1e-5)
        assert np.abs(out["coriolis_x"]).max() <= 1e-12
        assert np.abs(out["coriolis_y"]).max() <= 1e-12
        assert out["stretching_z"] == pytest.approx(2.0 * omega * STRETCH_A, rel=1e-5)
    else:
        assert "coriolis_z" not in out


def test_without_the_coriolis_term_the_rotating_record_does_not_close(tmp_path):
    _g, _deck, opened, _c, _r = build(tmp_path, steps=3, name="omit")
    record = opened.records[1]
    out = nwb.record_budget(opened.fields(1), opened.deck_for(record))   # no rotation
    omega = omega_rotating(record.model_time_s - 0.5 * opened.dt_s, F_TEST)
    rate = STRETCH_A * (2.0 * omega + F_TEST)
    total = sum(np.asarray(out[f"{t}_z"], np.float64) for t in nwb.CLOSURE_TERMS)
    assert float(np.abs(total - rate).max()) / float(np.abs(out["stretching_z"]).max()) > 0.04


def test_the_momentum_rows_are_minus_f_k_cross_the_ground_relative_wind(tmp_path):
    g, _deck, opened, (xc, yc), _r = build(tmp_path, steps=3, name="accel")
    record = opened.records[1]
    deck = opened.deck_for(record)
    out = nwb.record_budget(opened.fields(1), deck, rotation=opened.rotation)
    omega = omega_rotating(record.model_time_s - 0.5 * opened.dt_s, F_TEST)
    x = deck.minx_m + (np.arange(deck.nx) + 0.5) * deck.dx_m
    y = deck.miny_m + (np.arange(deck.ny) + 0.5) * deck.dy_m
    X, Y = np.meshgrid(x - xc, y - yc)
    u = -0.5 * STRETCH_A * X - omega * Y
    v = -0.5 * STRETCH_A * Y + omega * X
    scale = F_TEST * (np.abs(v).max() + abs(V_MOVE))
    assert np.abs(out["coriolis_accel_x"] - F_TEST * (v + V_MOVE)[None]).max() <= 2e-5 * scale
    assert np.abs(out["coriolis_accel_y"] + F_TEST * (u + U_MOVE)[None]).max() <= 2e-5 * scale


# ---------------------------------------------------------------------------
# lspgrad = 1: counted once, as itself.
# ---------------------------------------------------------------------------
def test_lspgrad1_is_taken_out_of_friction_and_published_once(tmp_path):
    _g, deck, opened, _c, receipt = build(tmp_path, lspgrad=1, steps=4, name="lspg")
    rotation = opened.rotation
    assert rotation["lspgrad"] == 1
    nzw = opened.records[0].nz
    u0, v0 = u0_v0(deck)
    lu, lv = ncw.lspgrad1_profiles(fcor_s_inv=F_TEST, u_move_m_s=U_MOVE, v_move_m_s=V_MOVE,
                                   u0=u0[:nzw + 1], v0=v0[:nzw + 1])
    assert rotation["lspg_u_m_s2"] == [float(w) for w in lu]
    assert rotation["lspg_v_m_s2"] == [float(w) for w in lv]
    f = np.float32
    assert lu[3] == f(-f(f(F_TEST) * f(v0[3] + f(V_MOVE))))

    words = opened.fields(2)
    wd = opened.deck_for(opened.records[2])
    out = nwb.record_budget(words, wd, rotation=rotation)
    # friction is the held stress alone: curl_y of STRESS_C * z
    assert out["friction_y"] == pytest.approx(STRESS_C, rel=2e-4)
    assert np.abs(out["friction_x"]).max() <= 1e-10
    assert np.abs(out["friction_z"]).max() <= 1e-10
    # lspg is -f dU0/dz, -f dV0/dz, 0
    assert out["lspg_x"] == pytest.approx(-F_TEST * 0.01, rel=2e-4)
    assert out["lspg_y"] == pytest.approx(-F_TEST * 0.02, rel=2e-4)
    assert np.abs(out["lspg_z"]).max() <= 1e-12
    # Not double counted: the split sums back to the curl of the whole held tendency.
    whole = nwb.centred_curl(words["uten"], words["vten"], words["wten"], dx=wd.dx_m,
                             dy=wd.dy_m, zh=wd.zh)
    for axis in ("x", "y", "z"):
        split = np.asarray(out[f"friction_{axis}"], np.float64) + out[f"lspg_{axis}"]
        largest = max(float(np.abs(whole[f"friction_{axis}"]).max()), 1e-30)
        scale = max(largest, float(np.abs(out["lspg_y"]).max()))
        assert float(np.abs(split - whole[f"friction_{axis}"]).max()) <= 1e-5 * scale
    assert out["lspg_accel_x"] == pytest.approx(np.broadcast_to(lu[:-1, None, None],
                                                                out["lspg_accel_x"].shape))


def test_lspgrad1_without_the_base_winds_is_refused(tmp_path):
    g, deck, stream = open_stream(tmp_path, config=rotating_config(lspgrad=1), name="nobase")
    xc, yc = centre_of(g)
    arrays = child_arrays(deck, 0.125, xc=xc, yc=yc)
    with pytest.raises(ncw.NestChildWindowError) as caught:
        stream.capture(arrays, child_step=1, model_time_s=0.25)
    assert "large-scale pressure gradient as friction" in str(caught.value)
    stream.abandon()


# ---------------------------------------------------------------------------
# Along the trajectories, and the published budget.
# ---------------------------------------------------------------------------
def _trajectories(opened, xc, yc):
    frames = opened.frames()
    seeds = nwb.circuit_positions(centre_x_m=xc, centre_y_m=yc, radius_m=250.0,
                                  height_m=175.0, count=8)
    request = nb.BackwardRequest(None, frames[-1].model_time_s, frames[0].model_time_s,
                                 opened.dt_s, wind_source="model-step", positions=seeds)
    return nb.integrate_backward(frames, request)


def test_the_rotating_budget_closes_along_trajectories_and_publishes_v3(tmp_path):
    _g, _deck, opened, (xc, yc), _r = build(tmp_path, lspgrad=1, name="traj")
    result = _trajectories(opened, xc, yc)
    budget = nwb.budget_along(opened, result, verify=True)
    names = [name for name, _ in budget.row_spec]
    assert budget.rows.shape[1] == len(names)
    for row in ("coriolis_z", "coriolis_accel_x", "lspg_y", "lspg_accel_y"):
        assert row in names
    cor = budget.rows[:, names.index("coriolis_z")][budget.sampled]
    assert cor == pytest.approx(F_TEST * STRETCH_A, rel=1e-4)
    # The shared-base fixture's lspg_x is a real (unbalanced) forcing of xi, so
    # the closure that has to hold is the vertical one.
    assert budget.closure["zvort"]["residual_over_scale"]["median"] < 2e-3
    receipt = nwb.write_trajectory_budget(tmp_path / "out", budget, stream=opened)
    assert receipt["schema"] == nwb.TRAJECTORY_BUDGET_ROTATING_SCHEMA
    assert "+ coriolis + lspg" in receipt["closure_definition"]
    assert "solve2.F:639-653" in receipt["coriolis_definition"]
    assert "lspg" in receipt["friction_definition"]
    assert receipt["rotation"]["fcor_s_inv"] == float(np.float32(F_TEST))
    table = (tmp_path / "out" / nwb.BUDGET_TABLE_FILENAME).read_text().splitlines()
    assert len(table) == len(names) + 1 and "coriolis_x" in table[1 + names.index("coriolis_x")]


def test_omitting_the_coriolis_term_would_leave_it_in_the_residual(tmp_path):
    _g, _deck, opened, (xc, yc), _r = build(tmp_path, name="trajomit")
    result = _trajectories(opened, xc, yc)
    with_rotation = nwb.budget_along(opened, result).closure["zvort"]
    opened.receipt.pop("rotation")
    without = nwb.budget_along(opened, result).closure["zvort"]
    assert with_rotation["residual_over_scale"]["median"] < 2e-3
    assert without["residual_over_scale"]["median"] > 10 * with_rotation["residual_over_scale"]["median"]


def test_a_non_rotating_budget_keeps_the_v2_rows_and_schema(tmp_path):
    _g, _deck, opened, (xc, yc), _r = build(tmp_path, f=0.0, name="v2budget")
    result = _trajectories(opened, xc, yc)
    budget = nwb.budget_along(opened, result)
    assert budget.row_spec == nwb.BUDGET_ROWS and budget.rotation is None
    receipt = nwb.write_trajectory_budget(tmp_path / "out", budget, stream=opened)
    assert receipt["schema"] == nwb.TRAJECTORY_BUDGET_SCHEMA
    assert "rotation" not in receipt and "coriolis_definition" not in receipt
    assert nwb.budget_rows(None) is nwb.BUDGET_ROWS
