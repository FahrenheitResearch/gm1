"""The vorticity budget along a trajectory, against a closed form that closes exactly.

The field is the stretching vortex of :mod:`cm1core.nest_window_budget`'s
docstring:

    u = -a x/2 - Omega(t) y,  v = -a y/2 + Omega(t) x,  w = a z,
    Omega(t) = Omega_0 exp(a t)

It is non-divergent, its vorticity is ``zeta = 2 Omega(t)`` everywhere, its
tilting and divergence terms vanish identically, and its stretching term
``zeta dw/dz = 2 a Omega`` is exactly ``D zeta / Dt``.  So the budget must
close to rounding, term by term, and the two terms that are not stretching
must be zero -- which is a far stronger statement than a residual being small.
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
    CHILD_DZ_M,
    OMEGA_0,
    STRETCH_A,
    centre_of,
    child_arrays,
    open_stream,
    settings,
    stream_root,
    write_records,
)

NAMES = [name for name, _ in nwb.BUDGET_ROWS]


def build(tmp_path, *, steps=24, name="budget"):
    g, deck, stream = open_stream(tmp_path, name=name)
    xc, yc = centre_of(g)
    write_records(stream, deck, g, steps=steps, xc=xc, yc=yc)
    opened = ncw.read_window_stream(stream_root(tmp_path, name))
    return g, deck, opened, (xc, yc)


def trajectories(opened, xc, yc, *, count=8, radius=250.0, height=175.0):
    frames = opened.frames()
    start = frames[-1].model_time_s
    end = frames[0].model_time_s
    seeds = nwb.circuit_positions(centre_x_m=xc, centre_y_m=yc, radius_m=radius,
                                  height_m=height, count=count)
    request = nb.BackwardRequest(None, start, end, opened.dt_s, wind_source="model-step",
                                 positions=seeds)
    return frames, request, nb.integrate_backward(frames, request)


# ---------------------------------------------------------------------------
# One record's budget, against the closed form.
# ---------------------------------------------------------------------------
def test_one_records_budget_is_the_closed_forms_term_by_term(tmp_path):
    _g, _deck, opened, (xc, yc) = build(tmp_path, steps=4, name="one")
    record = opened.records[2]
    deck = opened.deck_for(record)
    metric = nwb.vertical_metric_record(deck)
    assert metric["uniform"] is True
    assert metric["spacing_m"]["min"] == pytest.approx(CHILD_DZ_M)
    fields = nwb.record_budget(opened.fields(2), deck)
    # The record holds the step's average wind, so its Omega is the midpoint's.
    t_mid = record.model_time_s - 0.5 * opened.dt_s
    omega = OMEGA_0 * math.exp(STRETCH_A * t_mid)
    zeta = 2.0 * omega
    assert fields["zvort"] == pytest.approx(zeta, abs=2e-7)
    assert fields["xvort"] == pytest.approx(0.0, abs=2e-7)
    assert fields["yvort"] == pytest.approx(0.0, abs=2e-7)
    assert fields["stretching_z"] == pytest.approx(zeta * STRETCH_A, rel=1e-4)
    assert fields["tilting_z"] == pytest.approx(0.0, abs=1e-9)
    assert fields["divergence_z"] == pytest.approx(0.0, abs=1e-9)
    # Barotropic: constant rho and prs, so the solenoid is identically zero.
    assert fields["solenoid_z"] == pytest.approx(0.0, abs=1e-12)
    assert fields["rho"] == pytest.approx(1.1)


def test_the_solenoid_is_centred_everywhere_in_the_window_including_its_edge():
    """A ring of the child's own neighbours makes every point centred, edge included."""

    nz, ny, nx = 4, 5, 6
    z = np.arange(nz + 2, dtype=np.float64)[:, None, None]
    y = np.arange(ny + 2, dtype=np.float64)[None, :, None]
    x = np.arange(nx + 2, dtype=np.float64)[None, None, :]
    dx, dy, dz = 20.0, 25.0, 50.0
    b, d = 3.0e-4, 7.0
    rho = np.asarray(1.2 + b * (x - 1) * dx + 0.0 * y + 0.0 * z, np.float32) * np.ones(
        (nz + 2, ny + 2, nx + 2), np.float32)
    prs = np.asarray(95000.0 + d * (y - 1) * dy + 0.0 * x + 0.0 * z, np.float32) * np.ones(
        (nz + 2, ny + 2, nx + 2), np.float32)
    zh = np.arange(nz + 2, dtype=np.float32) * np.float32(dz)
    out = nwb.centred_solenoid(rho, prs, dx=dx, dy=dy, zh=zh)
    core = rho[1:-1, 1:-1, 1:-1].astype(np.float64)
    want = b * d / (core * core)
    assert out["solenoid_z"].shape == (nz, ny, nx)
    assert out["solenoid_z"] == pytest.approx(want, rel=2e-4)
    assert out["solenoid_x"] == pytest.approx(0.0, abs=1e-12)
    assert out["solenoid_y"] == pytest.approx(0.0, abs=1e-12)


def test_the_vertical_derivative_is_taken_over_the_decks_own_levels(tmp_path):
    """A stretched child is the only kind a tornado run has; one nominal dz would mis-scale it."""

    from dataclasses import replace

    _g, _deck, opened, _c = build(tmp_path, steps=3, name="stretchcheck")
    deck = opened.deck_for(opened.records[0])
    words = opened.fields(0)
    # A deck whose levels are the same everywhere but whose nominal dz_m is
    # not: the budget must follow the levels, not the nominal word.
    misnamed = replace(deck, dz_reference_m=deck.dz_reference_m * 7.0)
    assert nwb.record_budget(words, misnamed)["stretching_z"] == pytest.approx(
        nwb.record_budget(words, deck)["stretching_z"], rel=1e-6)
    # And a genuinely stretched deck reports itself as one.
    zh = np.array(deck.zh, dtype=np.float32)
    zf = np.array(deck.zf, dtype=np.float32)
    for k in range(len(zh)):
        zh[k] = np.float32(float(zh[k]) * (1.0 + 0.05 * k))
    metric = nwb.vertical_metric_record(replace(deck, zh=zh, zf=zf))
    assert metric["uniform"] is False
    assert metric["spacing_m"]["max"] > metric["spacing_m"]["min"]
    assert "misclibs.F:1628" in metric["cm1_equivalent"]


def test_a_stretched_deck_and_the_uniform_solenoid_are_refused_together():
    import numpy as _np

    from cm1core.vorticity_budget import cpu_vorticity_budget

    nz, ny, nx = 4, 4, 4
    ng = 3
    u = _np.zeros((nz + 2, ny + 2 * ng, nx + 1 + 2 * ng), _np.float32)
    v = _np.zeros((nz + 2, ny + 1 + 2 * ng, nx + 2 * ng), _np.float32)
    w = _np.zeros((nz + 3, ny + 2 * ng, nx + 2 * ng), _np.float32)
    block = _np.ones((nz, ny, nx), _np.float32)
    with pytest.raises(ValueError) as caught:
        cpu_vorticity_budget(u=u, v=v, w=w, rho=block, prs=block, nx=nx, ny=ny, nz=nz,
                             dx=20.0, dy=20.0, dz=50.0,
                             zh=_np.arange(nz + 2, dtype=_np.float32))
    assert "two different vertical metrics" in str(caught.value) or \
        "cannot be combined with a stretched deck" in str(caught.value)


def test_a_stretched_child_deck_gets_the_right_stretching_term(tmp_path):
    """Every tornado-grade child is stretched; a nominal dz would be wrong by 5x here."""

    from cm1core.cm1_vertical_grid import build_vertical_levels
    from cm1core.nest_parcels import deck_from_levels, fill_velocity_bottom_ghost

    from test_nest_child_window import child_arrays, geometry

    levels = build_vertical_levels(stretch_z=1, nk=27, dz_m=100.0, ztop_m=1500.0,
                                   str_bot_m=90.0, str_top_m=500.0, dz_bot_m=5.0,
                                   dz_top_m=200.0)
    g = geometry()
    deck = deck_from_levels(name="child", nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m,
                            minx_m=g.origin_x_m, miny_m=g.origin_y_m, levels=levels,
                            lateral=(2, 2, 2, 2), periodic=False)
    box_nz = 22
    spacings = np.diff(np.asarray(deck.zh[:box_nz + 2], dtype=np.float64))
    assert spacings.max() > 4.0 * spacings.min()      # genuinely stretched
    xc, yc = centre_of(g)
    arrays = child_arrays(deck, 1.0, xc=xc, yc=yc)
    box = ncw.WindowBox(9, 9, 7, 7, box_nz)
    words = {k: np.ascontiguousarray(v) for k, v in ncw.cut_window(np, arrays, box).items()}
    wd = ncw.window_deck(deck, box)
    fill_velocity_bottom_ghost(np, words["uadv"], words["vadv"], wd, bbc=1,
                               u_move_m_s=0.0, v_move_m_s=0.0)
    ncw.fill_window_w_ghost(words["wadv"])
    fields = nwb.record_budget(words, wd)
    # w = a z exactly, so dw/dz = a and stretching_z = zeta * a at every level.
    omega = OMEGA_0 * math.exp(STRETCH_A * 1.0)
    assert fields["stretching_z"] == pytest.approx(2.0 * omega * STRETCH_A, rel=2e-3)
    metric = nwb.vertical_metric_record(wd)
    assert metric["uniform"] is False
    assert metric["spacing_m"]["min"] == pytest.approx(float(spacings.min()), rel=1e-6)
    assert metric["spacing_m"]["max"] == pytest.approx(float(spacings.max()), rel=1e-6)


# ---------------------------------------------------------------------------
# Along the trajectory.
# ---------------------------------------------------------------------------
def test_the_budget_closes_along_every_trajectory_to_rounding(tmp_path):
    _g, _deck, opened, (xc, yc) = build(tmp_path, name="closes")
    _frames, _request, result = trajectories(opened, xc, yc)
    budget = nwb.budget_along(opened, result, verify=True)
    assert budget.rows.shape == (result.rows.shape[0], len(nwb.BUDGET_ROWS), result.rows.shape[2])
    assert budget.sampled.all()
    zeta = budget.rows[:, NAMES.index("zvort")]
    stretch = budget.rows[:, NAMES.index("stretching_z")]
    tilt = budget.rows[:, NAMES.index("tilting_z")]
    div = budget.rows[:, NAMES.index("divergence_z")]
    sol = budget.rows[:, NAMES.index("solenoid_z")]
    # Every term but stretching is identically zero in this flow.
    assert np.abs(tilt).max() < 1e-8
    assert np.abs(div).max() < 1e-8
    assert np.abs(sol).max() == 0.0
    # And stretching is zeta * a, at every position of every trajectory.
    assert stretch == pytest.approx(zeta * STRETCH_A, rel=2e-3)
    report = budget.closure
    assert report["samples"] == budget.rows.shape[0] * budget.rows.shape[2]
    assert report["abs_measured_tendency_s2"]["median"] > 0.0
    assert report["residual_over_scale"]["max"] < 0.02
    assert report["residual_over_scale"]["median"] < 2e-3


def test_the_measured_tendency_is_the_closed_forms_and_the_residual_is_the_gap(tmp_path):
    _g, _deck, opened, (xc, yc) = build(tmp_path, name="tendency")
    _frames, _request, result = trajectories(opened, xc, yc)
    budget = nwb.budget_along(opened, result)
    measured = budget.rows[:, NAMES.index("zvort_tendency_measured")]
    summed = budget.rows[:, NAMES.index("zvort_tendency_terms")]
    residual = budget.rows[:, NAMES.index("zvort_residual")]
    usable = measured > np.float32(-1.0e29)
    assert usable.any()
    # D(zeta)/Dt = 2 a Omega(t) in closed form; the sampled tendency is that.
    for step, t in enumerate(budget.times_s):
        if not bool(usable[step].any()):
            continue
        want = 2.0 * STRETCH_A * OMEGA_0 * math.exp(STRETCH_A * (t - 0.5 * opened.dt_s))
        assert float(np.median(measured[step][usable[step]])) == pytest.approx(want, rel=5e-3)
    # Interior rows: the centred difference minus the terms at that row.  (The
    # two end rows compare a one-sided difference with the mean of the terms
    # at the two rows it spans.)
    inner = usable.copy()
    inner[0] = inner[-1] = False
    assert residual[inner] == pytest.approx((measured - summed)[inner], abs=1e-9)


def test_the_receipt_publishes_friction_and_says_what_the_residual_holds(tmp_path):
    _g, _deck, opened, (xc, yc) = build(tmp_path, name="receipt")
    _frames, _request, result = trajectories(opened, xc, yc)
    budget = nwb.budget_along(opened, result)
    nwb.write_trajectory_budget(tmp_path / "out", budget, stream=opened)
    on_disk = json.loads((tmp_path / "out" / nwb.BUDGET_RECEIPT_FILENAME).read_text())
    assert on_disk["schema"] == nwb.TRAJECTORY_BUDGET_SCHEMA
    assert on_disk["kind"] == "offline-diagnostic"
    assert on_disk["refused_terms"] == {}
    assert "held momentum tendency" in on_disk["friction_definition"]
    assert "implicit dissipation of the odd-order advection" in on_disk["closure"]["residual_holds"]
    for component in ("xvort", "yvort", "zvort"):
        assert component in on_disk["closure"]
    assert on_disk["unvalidated"].startswith("UNVALIDATED against CM1")
    assert any("misclibs.F" in c for c in on_disk["citations"])
    # The fixture's held tendency is zero, so the friction rows are zero.
    for axis in ("x", "y", "z"):
        row = budget.rows[:, NAMES.index(f"friction_{axis}")]
        assert (row[budget.sampled] == 0.0).all()
    assert (tmp_path / "out" / nwb.BUDGET_ARENA_FILENAME).stat().st_size == budget.rows.nbytes
    table = (tmp_path / "out" / nwb.BUDGET_TABLE_FILENAME).read_text().splitlines()
    assert table[0] == "index\tname\tunits"
    assert len(table) == len(nwb.BUDGET_ROWS) + 1


def test_friction_is_the_centred_curl_of_the_held_tendency_on_its_own_faces():
    """A linear held tendency: every component of its curl is known exactly."""

    nz, ny, nx = 4, 5, 6
    dx, dy = 20.0, 25.0
    zh = np.asarray([-5.0, 5.0, 15.0, 27.0, 41.0, 57.0, 75.0], np.float32)  # stretched
    zf = np.asarray([-10.0, 0.0, 10.0, 20.0, 34.0, 48.0, 66.0, 84.0], np.float32)
    a, b, c, d = 2.0e-3, -3.0e-3, 4.0e-4, 5.0e-4

    def grid(nk, nj, ni, *, xf=False, yf=False, wf=False):
        k = np.arange(nk)[:, None, None]
        j = np.arange(nj)[None, :, None]
        i = np.arange(ni)[None, None, :]
        # ring index 0 is the cell one west/south of the window's first cell
        x = (i - (1.0 if xf else 0.5)) * dx
        y = (j - (1.0 if yf else 0.5)) * dy
        z = (zf[1 + k] if wf else zh[1 + k])
        return np.broadcast_arrays(x, y, np.asarray(z, np.float64))

    x, y, z = grid(nz + 1, ny + 2, nx + 3, xf=True)
    uten = (a * y + c * z).astype(np.float32)
    x, y, z = grid(nz + 1, ny + 3, nx + 2, yf=True)
    vten = (b * x + d * z).astype(np.float32)
    x, y, z = grid(nz + 2, ny + 2, nx + 2, wf=True)
    wten = (c * x + d * y).astype(np.float32)
    out = nwb.centred_curl(uten, vten, wten, dx=dx, dy=dy, zh=zh)
    assert out["friction_z"].shape == (nz, ny, nx)
    assert out["friction_x"] == pytest.approx(d - d, abs=1e-8)       # dwdy - dvdz
    assert out["friction_y"] == pytest.approx(c - c, abs=1e-8)       # dudz - dwdx
    assert out["friction_z"] == pytest.approx(b - a, rel=1e-4)       # dvdx - dudy
    # And a tendency that does have a horizontal curl.
    wten2 = (3.0 * c * x).astype(np.float32)
    out2 = nwb.centred_curl(uten, vten, wten2, dx=dx, dy=dy, zh=zh)
    assert out2["friction_y"] == pytest.approx(c - 3.0 * c, rel=1e-4)


def test_the_solenoid_uses_the_total_density_with_its_loading(tmp_path):
    """Condensate loading's horizontal gradient is baroclinic; dry air alone would drop it."""

    _g, _deck, opened, _c = build(tmp_path, steps=3, name="loading")
    record = opened.records[0]
    deck = opened.deck_for(record)
    words = {k: np.array(v) for k, v in opened.fields(0).items()}
    nzr, nyr, nxr = words["qhyd"].shape
    # A condensate gradient in x under a pressure gradient in z: solenoid_y.
    words["qhyd"] = (np.arange(nxr, dtype=np.float32)[None, None, :] * np.float32(1e-3)
                     * np.ones((nzr, nyr, nxr), np.float32))
    words["prs"] = (np.float32(95000.0) - np.asarray(deck.zh[1:nzr + 1], np.float32)[:, None, None]
                    * np.float32(11.0)) * np.ones((nzr, nyr, nxr), np.float32)
    out = nwb.record_budget(words, deck)
    assert np.abs(out["solenoid_y"]).max() > 0.0
    dry = dict(words)
    dry["qhyd"] = np.zeros_like(words["qhyd"])
    assert np.abs(nwb.record_budget(dry, deck)["solenoid_y"]).max() == 0.0


def test_the_budget_is_sampled_only_where_a_record_exists_never_interpolated(tmp_path):
    _g, _deck, opened, (xc, yc) = build(tmp_path, name="missing")
    _frames, _request, result = trajectories(opened, xc, yc)
    steps = len(result.times_s)
    result.times_s[3] = result.times_s[3] + 0.013
    budget = nwb.budget_along(opened, result)
    assert len(budget.times_s) == steps - 1
    assert result.times_s[3] not in budget.times_s
    result.times_s[:] = [t + 0.013 for t in result.times_s]
    with pytest.raises(nwb.TrajectoryBudgetError) as caught:
        nwb.budget_along(opened, result)
    assert "the error this stream exists to remove" in str(caught.value)


# ---------------------------------------------------------------------------
# Seeding into the core, and the material circuit (phase two off the same stream).
# ---------------------------------------------------------------------------
def test_a_core_disk_puts_every_parcel_inside_the_core_unlike_a_lattice():
    disk = nwb.core_disk_positions(centre_x_m=100.0, centre_y_m=200.0, radius_m=50.0,
                                   heights_m=(10.0, 30.0), rings=4, per_ring=8)
    assert disk.shape == (3, 2 * (1 + 3 * 8))
    r = np.hypot(disk[0] - 100.0, disk[1] - 200.0)
    assert r.max() <= 50.0 + 1e-4
    assert sorted(set(np.round(disk[2], 6))) == [10.0, 30.0]
    with pytest.raises(nwb.TrajectoryBudgetError):
        nwb.core_disk_positions(centre_x_m=0.0, centre_y_m=0.0, radius_m=0.0,
                                heights_m=(10.0,), rings=2, per_ring=4)


def test_a_material_circuits_circulation_is_conserved_in_the_barotropic_flow(tmp_path):
    """Kelvin: the circuit is material and the flow barotropic, so C must not change."""

    _g, _deck, opened, (xc, yc) = build(tmp_path, steps=24, name="circuit")
    _frames, _request, result = trajectories(opened, xc, yc, count=24, radius=250.0)
    report = nwb.circulation_along(result, first_parcel=0, count=24, name="core-ring")
    values = [s["circulation_m2_s"] for s in report["samples"]]
    # Every row but the last, which begins no step and so publishes no wind.
    assert len(values) == result.rows.shape[0] - 1
    # The closed form at the seed: a 24-gon of radius 250 m in solid rotation
    # Omega, plus the radial convergence, which contributes nothing to a
    # circular circuit's line integral.
    from cm1core.circulation_circuits import solid_body_circulation

    start = result.times_s[0]
    omega = OMEGA_0 * math.exp(STRETCH_A * (start - 0.5 * opened.dt_s))
    assert values[0] == pytest.approx(
        solid_body_circulation(omega_s_inv=omega, radius_m=250.0, count=24), rel=5e-3)
    # And it is conserved along the circuit's own history, to a fraction of a
    # per cent over the whole integration.
    drift = max(abs(v - values[0]) for v in values) / abs(values[0])
    assert drift < 0.01, values
    assert "Markowski" in report["citation"]


def test_a_circuit_needs_three_vertices_to_enclose_anything():
    with pytest.raises(nwb.TrajectoryBudgetError) as caught:
        nwb.circuit_positions(centre_x_m=0.0, centre_y_m=0.0, radius_m=10.0,
                              height_m=5.0, count=2)
    assert "identically zero for every flow" in str(caught.value)


def test_the_whole_pipeline_runs_on_a_stretched_child_and_still_closes(tmp_path):
    """The deck every tornado child has: 5 m at the ground, stretched above it."""

    from test_nest_child_window import settings as window_settings
    from test_nest_child_window import stretched_child_levels

    levels = stretched_child_levels()
    # top_m = 165 m is 20 child layers here, and the deck is genuinely stretched
    # inside them (5 m at the ground, 102.5 m at the top of the window).
    s = window_settings(top_m=165.0)
    g, deck, stream = open_stream(tmp_path, s=s, name="stretched", levels=levels)
    xc, yc = centre_of(g)
    write_records(stream, deck, g, steps=24, xc=xc, yc=yc)
    opened = ncw.read_window_stream(stream_root(tmp_path, "stretched"))
    metric = nwb.vertical_metric_record(opened.deck_for(opened.records[0]))
    assert metric["uniform"] is False
    assert metric["spacing_m"]["max"] > 4.0 * metric["spacing_m"]["min"]
    assert metric["lowest_mass_level_m"] == pytest.approx(2.5)

    frames = opened.frames()
    request = nb.BackwardRequest(
        None, frames[-1].model_time_s, frames[0].model_time_s, opened.dt_s,
        wind_source="model-step",
        positions=nwb.core_disk_positions(centre_x_m=xc, centre_y_m=yc, radius_m=250.0,
                                          heights_m=(20.0, 80.0), rings=3, per_ring=6))
    result = nb.integrate_backward(frames, request)
    assert (result.status == nb.STATUS_ACTIVE).all()
    # The closed form, on a deck the integrator searches face by face.
    from test_nest_child_window import analytic_position

    start, end = result.times_s[0], result.times_s[-1]
    worst = 0.0
    for q in range(result.rows.shape[2]):
        x1, y1, z1 = (float(result.rows[0, r, q]) for r in range(3))
        want = analytic_position(x1, y1, z1, start, end, xc=xc, yc=yc)
        got = tuple(float(result.rows[-1, r, q]) for r in range(3))
        worst = max(worst, math.dist(want, got))
    assert worst < 0.05, worst

    budget = nwb.budget_along(opened, result)
    zeta = budget.rows[:, NAMES.index("zvort")]
    stretch = budget.rows[:, NAMES.index("stretching_z")]
    assert stretch == pytest.approx(zeta * STRETCH_A, rel=3e-3)
    assert budget.closure["residual_over_scale"]["median"] < 5e-3
    assert budget.vertical_metric["uniform"] is False


def test_a_morrison_v3_stream_gives_the_v2_budget_to_float_rounding(tmp_path):
    """The budget reads schema v3's rebuilt qhyd exactly where it read v2's word."""

    from test_nest_child_window import child_config, geometry, physical_carriers

    opened = {}
    for component, numq in (("wsm6-mp6", 6), ("morrison-mp10", 10)):
        g, deck, stream = open_stream(tmp_path, name=component,
                                      config=child_config(microphysics=component))
        xc, yc = centre_of(g)
        dt = stream.child_dt_s
        rng = np.random.default_rng(29)
        condensate = np.abs(rng.standard_normal((5, deck.nz, deck.ny, deck.nx))).astype(
            np.float32) * np.float32(1.0e-3)
        for n in range(1, 13):
            arrays = child_arrays(deck, (n - 0.5) * dt, xc=xc, yc=yc)
            qa = physical_carriers(deck, numq=numq)["qa"]
            qa[0] = np.float32(0.01)
            qa[1:6] = condensate
            if numq > 6:
                qa[6:] = np.float32(1.0e6)
            arrays["qa"] = qa
            stream.capture(arrays, child_step=n, model_time_s=n * dt)
        stream.close()
        opened[component] = ncw.read_window_stream(stream_root(tmp_path, component))
    v2, v3 = opened["wsm6-mp6"], opened["morrison-mp10"]
    assert v2.schema_version == 2 and v3.schema_version == 3
    deck = v2.deck_for(v2.records[3])
    base = v2.receipt["base_columns"]
    a = nwb.record_budget(v2.fields(3), deck, base_columns=base)
    b = nwb.record_budget(v3.fields(3), v3.deck_for(v3.records[3]), base_columns=base)
    for name in ("qhyd", "buoyancy", "solenoid_x", "solenoid_y", "solenoid_z", "stretching_z"):
        scale = float(np.max(np.abs(a[name]))) or 1.0
        assert float(np.max(np.abs(a[name] - b[name]))) <= 1e-5 * scale, name
    xc, yc = centre_of(geometry())
    _frames, _request, result = trajectories(v3, xc, yc)
    budget = nwb.budget_along(v3, result)
    receipt = nwb.write_trajectory_budget(tmp_path / "out-v3", budget, stream=v3)
    assert receipt["stream"]["schema"] == ncw.NEST_CHILD_WINDOW_SCHEMA_V3
    assert receipt["stream"]["qhyd_source"].startswith("rebuilt")
