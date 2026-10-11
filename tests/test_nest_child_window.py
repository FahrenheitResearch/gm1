"""The vortex-following child window: the cut, the refusals, the stream, the round trip.

No CUDA context is opened.  The stream is driven on NumPy through the same
code the device path takes -- the cut is a strided copy either way, and there
is no kernel in this module -- and every scientific claim is checked against a
closed form or against the child's own words.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from cm1core import nest_child_window as ncw  # noqa: E402
from cm1core import nest_parcel_backward as nb  # noqa: E402
from cm1core import nest_parcels as npc  # noqa: E402
from cm1core.cm1_vertical_grid import build_vertical_levels  # noqa: E402
from cm1core.lagrangian_config import (  # noqa: E402
    NEST_CHILD_WINDOW_KEYS,
    LagrangianConfigError,
    parse_lagrangian_table,
)
from cm1core.nest_geometry import NestGeometry  # noqa: E402

NG = ncw.NG
CHILD_DZ_M = 50.0
CHILD_NK = 20


def child_levels():
    """A uniform child deck: the budget's one dz is then the deck's own."""

    return build_vertical_levels(stretch_z=0, nk=CHILD_NK, dz_m=CHILD_DZ_M,
                                 ztop_m=CHILD_DZ_M * CHILD_NK)


def geometry(**overrides) -> NestGeometry:
    values = dict(parent_nx=40, parent_ny=40, parent_dx_m=600.0, parent_dy_m=600.0,
                  refinement_ratio=3, time_refinement_ratio=3, parent_i0=6, parent_j0=6,
                  parent_span_x=12, parent_span_y=12, spec_zone=1, relax_zone=4, feedback=1)
    values.update(overrides)
    return NestGeometry(**values)


def stretched_child_levels():
    """A 5 m lowest layer under a 1500 m top: the shape every tornado child has."""

    return build_vertical_levels(stretch_z=1, nk=27, dz_m=100.0, ztop_m=1500.0, str_bot_m=90.0,
                                 str_top_m=500.0, dz_bot_m=5.0, dz_top_m=200.0)


def child_deck(g: NestGeometry, levels=None) -> npc.ParcelDeck:
    return npc.deck_from_levels(name="child", nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m,
                                minx_m=g.origin_x_m, miny_m=g.origin_y_m,
                                levels=child_levels() if levels is None else levels,
                                lateral=(2, 2, 2, 2), periodic=False)


#: The default geometry's child spans 3600..10800 m each way, so its middle --
#: where the fixed-centre windows below sit -- is 7200 m.
CHILD_MIDDLE_M = 7200.0


def settings(**overrides):
    table = {"frame": "parent-grid-origin", "follow": "fixed",
             "center_x_m": CHILD_MIDDLE_M, "center_y_m": CHILD_MIDDLE_M,
             "half_width_x_m": 600.0, "half_width_y_m": 600.0, "top_m": 500.0,
             "stride_child_steps": 1, "start_s": 0.0, "end_s": 1000.0,
             "budget_bytes_per_model_minute": 1 << 40}
    table.update(overrides)
    return parse_lagrangian_table({"nest_child_window": table}).nest_child_window


def child_config(**overrides):
    values = dict(dt_s=0.25, bbc=1, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0, imove=0,
                  microphysics="wsm6-mp6")
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# The closed-form stretching vortex (see cm1core.nest_window_budget's docstring).
# ---------------------------------------------------------------------------
STRETCH_A = 0.01     # 1/s: -a/2 horizontal convergence, +a vertical stretching
OMEGA_0 = 0.05       # 1/s: half the initial vertical vorticity


def analytic_wind(x, y, z, t, *, xc, yc):
    omega = OMEGA_0 * math.exp(STRETCH_A * t)
    dx, dy = x - xc, y - yc
    return (-0.5 * STRETCH_A * dx - omega * dy,
            -0.5 * STRETCH_A * dy + omega * dx,
            STRETCH_A * z)


def analytic_position(x0, y0, z0, t0, t1, *, xc, yc):
    """Where the parcel at ``(x0, y0, z0)`` at ``t0`` is at ``t1``, in closed form.

    ``d(x + iy)/dt = (-a/2 + i Omega(t)) (x + iy)`` integrates to a rotation
    through ``int Omega dt = (Omega0/a)(e^{a t1} - e^{a t0})`` and a contraction
    by ``e^{-a (t1 - t0)/2}``; ``dz/dt = a z`` to ``z0 e^{a (t1 - t0)}``.  It
    holds for ``t1 < t0`` too, which is how the backward integration is checked.
    """

    span = t1 - t0
    turn = (OMEGA_0 / STRETCH_A) * (math.exp(STRETCH_A * t1) - math.exp(STRETCH_A * t0))
    scale = math.exp(-0.5 * STRETCH_A * span)
    c = complex(x0 - xc, y0 - yc) * scale * complex(math.cos(turn), math.sin(turn))
    return xc + c.real, yc + c.imag, z0 * math.exp(STRETCH_A * span)


def child_arrays(deck: npc.ParcelDeck, t: float, *, xc: float, yc: float):
    """Whole-child padded carriers holding the analytic field at model time ``t``."""

    nx, ny, nz = deck.nx, deck.ny, deck.nz
    zh = deck.zh.astype(np.float64)
    zf = deck.zf.astype(np.float64)

    def axis(n, origin, spacing, face):
        cm1 = np.arange(n) + 1 - NG
        return origin + ((cm1 - 1) if face else (cm1 - 0.5)) * spacing

    def build(levels, jn, jface, in_, iface, component):
        z = levels[:, None, None]
        y = axis(jn, deck.miny_m, deck.dy_m, jface)[None, :, None]
        x = axis(in_, deck.minx_m, deck.dx_m, iface)[None, None, :]
        x, y, z = np.broadcast_arrays(x, y, z)
        return np.asarray(analytic_wind(x, y, z, t, xc=xc, yc=yc)[component],
                          dtype=np.float64).astype(np.float32) * np.ones(x.shape, np.float32)

    u = build(zh, ny + 2 * NG, False, nx + 1 + 2 * NG, True, 0)
    v = build(zh, ny + 1 + 2 * NG, True, nx + 2 * NG, False, 1)
    w = build(zf, ny + 2 * NG, False, nx + 2 * NG, False, 2)
    return {"rru": u, "rrv": v, "rrw": w, **physical_carriers(deck)}


def physical_carriers(deck: npc.ParcelDeck, rng=None, numq=6):
    """The child's physical scalar, moisture and held-tendency carriers.

    Uniform (and the held tendency zero) unless ``rng`` is given, in which case
    every word is distinct so an index identity can tell any two apart.
    """

    nx, ny, nz = deck.nx, deck.ny, deck.nz

    def field(shape, value):
        if rng is None:
            return np.full(shape, np.float32(value), dtype=np.float32)
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "rho": field((nz, ny, nx), 1.1), "prs": field((nz, ny, nx), 95000.0),
        "tha": field((nz, ny, nx), 0.0), "qa": field((numq, nz, ny, nx), 0.0),
        "uten": field((nz, ny, nx + 1), 0.0), "vten": field((nz, ny + 1, nx), 0.0),
        "wten": field((nz + 1, ny, nx), 0.0),
        "th0_profile": np.full(nz, np.float32(300.0), np.float32),
        "qv0_profile": np.full(nz, np.float32(0.01), np.float32),
    }


def stream_root(tmp_path, name="stream"):
    return tmp_path / name / ncw.NEST_CHILD_WINDOW_DIRECTORY


def open_stream(tmp_path, *, g=None, s=None, config=None, name="stream", levels=None):
    g = geometry() if g is None else g
    s = settings() if s is None else s
    config = child_config() if config is None else config
    deck = child_deck(g, levels)
    stream = ncw.NestChildWindowStream.build(
        settings=s, geometry=g, child_deck=deck, child_config=config, parent_dt_s=0.75,
    )
    stream.open(tmp_path / name)
    return g, deck, stream


def centre_of(g: NestGeometry):
    return (g.origin_x_m + 0.5 * g.nx * g.dx_m, g.origin_y_m + 0.5 * g.ny * g.dy_m)


def write_records(stream, deck, g, *, steps: int, xc: float, yc: float):
    """One record per child step, each holding that step's own *average* wind.

    A record is labelled with the model time at the **end** of the step it
    completes and holds the acoustic small-step average over that step
    (``solve2.F:1290-1301``), which is a centred-in-time wind.  The stand-in
    for that average here is the analytic field at the step's midpoint, which
    is what makes the record second-order in time -- and is why reversing one
    record per step is second-order too.
    """

    dt = stream.child_dt_s
    for n in range(1, steps + 1):
        t = n * dt
        stream.capture(child_arrays(deck, t - 0.5 * dt, xc=xc, yc=yc),
                       child_step=n, model_time_s=t)
    return stream.close()


# ---------------------------------------------------------------------------
# Configuration refusals, each naming the wrong result it prevents.
# ---------------------------------------------------------------------------
def test_every_window_selector_is_required_so_none_is_inherited():
    for missing in sorted(NEST_CHILD_WINDOW_KEYS):
        table = {k: v for k, v in _full_table().items() if k != missing}
        with pytest.raises(LagrangianConfigError) as caught:
            parse_lagrangian_table({"nest_child_window": table})
        assert missing in str(caught.value)


def _full_table():
    return {"frame": "parent-grid-origin", "follow": "fixed", "center_x_m": 1.0,
            "center_y_m": 2.0, "half_width_x_m": 600.0, "half_width_y_m": 600.0,
            "top_m": 500.0, "stride_child_steps": 1, "start_s": 0.0, "end_s": 10.0,
            "budget_bytes_per_model_minute": 1000}


@pytest.mark.parametrize(("key", "value", "phrase"), [
    ("frame", "domain-centre", "parent grid's origin"),
    ("follow", "storm", "the air beside the tornado"),
    ("half_width_x_m", 0.0, "holds no cell"),
    ("top_m", -1.0, "holds no cell"),
    ("stride_child_steps", 0, ">= 1"),
    ("end_s", 0.0, "without a single record"),
    ("budget_bytes_per_model_minute", 0, ">= 1"),
])
def test_a_window_value_that_names_no_stream_is_refused_by_value(key, value, phrase):
    table = {**_full_table(), key: value}
    with pytest.raises(LagrangianConfigError) as caught:
        parse_lagrangian_table({"nest_child_window": table})
    assert phrase in str(caught.value)


@pytest.mark.parametrize(("follow", "flag"), [
    ("tornado-criterion", "follows_criterion"), ("nest-centre", "follows_nest"),
])
def test_a_declared_centre_the_run_would_ignore_is_refused(follow, flag):
    for centre in ((1.0, 2.0), (1.0, 0.0), (0.0, 2.0)):
        table = {**_full_table(), "follow": follow,
                 "center_x_m": centre[0], "center_y_m": centre[1]}
        with pytest.raises(LagrangianConfigError) as caught:
            parse_lagrangian_table({"nest_child_window": table})
        assert "the run then ignores" in str(caught.value) and follow in str(caught.value)
    table["center_x_m"] = table["center_y_m"] = 0.0
    assert getattr(parse_lagrangian_table({"nest_child_window": table}).nest_child_window, flag)


def test_the_sealed_projection_names_every_window_key_and_only_when_declared():
    without = parse_lagrangian_table({})
    assert "nest_child_window" not in without.projection()
    with_table = parse_lagrangian_table({"nest_child_window": _full_table()})
    projected = with_table.projection()["nest_child_window"]
    assert set(projected) == set(NEST_CHILD_WINDOW_KEYS)


# ---------------------------------------------------------------------------
# Admission: the extent, the top, the criterion and the disk.
# ---------------------------------------------------------------------------
def _admission(**overrides):
    kwargs = dict(settings=settings(), geometry=geometry(),
                  child_grid=SimpleNamespace(nz=CHILD_NK, dz_m=CHILD_DZ_M,
                                             vertical=child_levels()),
                  child_dt_s=0.25, duration_s=1000.0, bbc=1, terrain=False,
                  criterion_declared=True, component="wsm6-mp6", nest_moves=True)
    kwargs.update(overrides)
    return ncw.nest_child_window_admission_defect(**kwargs)


def test_an_admissible_window_is_admitted():
    assert _admission() is None


def test_a_window_that_would_reach_the_parent_forced_rows_is_refused():
    defect = _admission(settings=settings(half_width_x_m=10_000.0))
    assert defect is not None and defect[0] == "EXTENT"
    assert "parent-interpolated boundary row" in defect[2]


def test_a_window_reaching_the_childs_lid_is_refused_by_name():
    defect = _admission(settings=settings(top_m=CHILD_DZ_M * CHILD_NK))
    assert defect is not None and defect[0] == "TOP"
    assert "free-slip extrapolation" in defect[2] and "a lid the child's interior does not have" in defect[2]


def test_a_criterion_following_window_without_the_criterion_is_refused():
    following = settings(follow="tornado-criterion", center_x_m=0.0, center_y_m=0.0)
    defect = _admission(settings=following, criterion_declared=False)
    assert defect is not None and defect[0] == "NO-CRITERION"
    assert "the stream would be empty" in defect[2]


def test_a_nest_centre_window_is_admitted_without_the_criterion():
    riding = settings(follow="nest-centre", center_x_m=0.0, center_y_m=0.0)
    assert _admission(settings=riding, criterion_declared=False) is None


def test_a_nest_centre_window_on_a_static_nest_is_refused():
    riding = settings(follow="nest-centre", center_x_m=0.0, center_y_m=0.0)
    defect = _admission(settings=riding, criterion_declared=False, nest_moves=False)
    assert defect is not None and defect[0] == "NEST-STATIC"
    assert "nest.follow_storm = false" in defect[2] and 'follow = "fixed"' in defect[2]
    # the other modes do not depend on the nest moving
    assert _admission(nest_moves=False) is None
    following = settings(follow="tornado-criterion", center_x_m=0.0, center_y_m=0.0)
    assert _admission(settings=following, nest_moves=False) is None


def test_a_window_over_the_declared_disk_budget_is_refused_with_the_number():
    defect = _admission(settings=settings(budget_bytes_per_model_minute=1000))
    assert defect is not None and defect[0] == "DISK-BUDGET"
    assert "bytes per model minute" in defect[2] and "filling the disk" in defect[2]


def test_a_window_past_the_runs_own_duration_is_refused():
    defect = _admission(settings=settings(end_s=2000.0), duration_s=1000.0)
    assert defect is not None and defect[0] == "INTERVAL"


def test_a_window_reaching_the_rayleigh_layer_is_refused_because_friction_would_hold_damping():
    defect = _admission(damping_bottom_m=400.0)       # the window's top face is 500 m
    assert defect is not None and defect[0] == "DAMPING"
    assert "report the damping as friction" in defect[2]
    assert _admission(damping_bottom_m=5000.0) is None


def test_hrdamp_and_beta_plane_coriolis_are_refused_by_name():
    defect = _admission(hrdamp=1)
    assert defect is not None and defect[0] == "DAMPING" and defect[1] == "closures.hrdamp"
    # f-plane Coriolis is admitted (tests/test_nest_window_coriolis.py); the
    # beta plane, whose f the header cannot carry, is not.
    assert _admission(coriolis=1) is None
    defect = _admission(coriolis=1, beta_plane=1)
    assert defect is not None and defect[0] == "CORIOLIS-BETA-PLANE"


def test_a_carrier_of_the_wrong_family_is_refused_rather_than_cut_displaced(tmp_path):
    g, deck, stream = open_stream(tmp_path, name="family")
    xc, yc = centre_of(g)
    arrays = child_arrays(deck, 0.125, xc=xc, yc=yc)
    # A padded rho where the live child holds a physical one: every word cut
    # from it would be one level and three columns away from the one named.
    arrays["rho"] = np.ones((deck.nz + 2, deck.ny + 2 * NG, deck.nx + 2 * NG), np.float32)
    stream.after_parent_step(SimpleNamespace(geometry=g, parent=SimpleNamespace(
        clock=SimpleNamespace(completed_steps=0)), criterion=None))
    with pytest.raises(ncw.NestChildWindowError) as caught:
        stream.capture(arrays, child_step=1, model_time_s=0.25)
    assert "every word cut from it would be displaced" in str(caught.value)
    stream.abandon()


def test_the_ring_ghost_is_the_linear_extrapolation_in_the_decks_own_heights():
    zh = np.asarray([-5.0, 5.0, 15.0, 30.0], np.float32)
    block = np.stack([np.full((2, 2), 2.0 * z, np.float32) for z in zh[1:]])
    filled = ncw.fill_ring_bottom_ghost(block, zh)
    assert filled.shape == (4, 2, 2)
    assert np.allclose(filled[0], -10.0)
    assert np.array_equal(filled[1:], block)


def test_the_stated_disk_cost_is_the_records_own_size_times_their_cadence():
    box = ncw.WindowBox(0, 0, 7, 7, 10)
    per_record = box.record_bytes(surface=False)
    assert per_record == 4 * box.record_words(surface=False)
    cost = ncw.stream_bytes_per_model_minute(box, child_dt_s=0.1, stride=1, surface=False)
    assert cost == pytest.approx(per_record * 600.0)
    assert ncw.stream_bytes_per_model_minute(box, child_dt_s=0.1, stride=2, surface=False) \
        == pytest.approx(cost / 2.0)


# ---------------------------------------------------------------------------
# The cut is an index identity against the child's own words.
# ---------------------------------------------------------------------------
def test_every_published_word_is_the_childs_own_word_at_the_same_cm1_index():
    g = geometry()
    deck = child_deck(g)
    arrays = child_arrays(deck, 3.0, xc=centre_of(g)[0], yc=centre_of(g)[1])
    box = ncw.WindowBox(9, 11, 7, 5, 10)
    cut = ncw.cut_window(np, arrays, box, surface=None)
    wd = ncw.window_deck(deck, box)
    rng = np.random.default_rng(11)
    # Window CM1 index (i, j, k) must be the child's (i0 + i, j0 + j, k), for
    # every field over its own staggered extent, ghost columns included.
    extents = {"uadv": ("rru", box.nx + 1, box.ny, box.nz + 2),
               "vadv": ("rrv", box.nx, box.ny + 1, box.nz + 2),
               "wadv": ("rrw", box.nx, box.ny, box.nz + 3)}
    for name, (source, ni, nj, nk) in extents.items():
        for _ in range(40):
            i = int(rng.integers(1 - NG, ni + 1 + NG))
            j = int(rng.integers(1 - NG, nj + 1 + NG))
            k = int(rng.integers(0, nk))
            assert cut[name][k, j - 1 + NG, i - 1 + NG] == \
                arrays[source][k, box.j0 + j - 1 + NG, box.i0 + i - 1 + NG]
    # The scalars and the held tendency keep CM1 levels 1..nz+1 and one ring,
    # the child's own neighbours of the window, cut from the PHYSICAL carriers
    # the live child holds them in: record (k, j, i) is the child's physical
    # (k, j0 - 1 + j, i0 - 1 + i).
    arrays.update(physical_carriers(deck, rng=np.random.default_rng(5)))
    cut = ncw.cut_window(np, arrays, box, surface=None)
    shapes = box.shapes(surface=False)
    for name, source in (("rho", "rho"), ("prs", "prs"), ("th", "tha"),
                         ("uten", "uten"), ("vten", "vten"), ("wten", "wten")):
        assert cut[name].shape == shapes[name], name
        for _ in range(20):
            k, j, i = (int(rng.integers(0, n)) for n in shapes[name])
            assert cut[name][k, j, i] == arrays[source][k, box.j0 - 1 + j, box.i0 - 1 + i], name
    assert cut["qv"].shape == shapes["qv"]
    qa = arrays["qa"]
    for _ in range(20):
        k, j, i = (int(rng.integers(0, n)) for n in shapes["qhyd"])
        at = (k, box.j0 - 1 + j, box.i0 - 1 + i)
        assert cut["qv"][k, j, i] == qa[(0, *at)]
        want = qa[(1, *at)]
        for row in (2, 3, 4, 5):
            want = np.float32(want + qa[(row, *at)])
        assert cut["qhyd"][k, j, i] == want
    # And the window's own deck puts its west and south faces where the cut is.
    assert wd.minx_m == pytest.approx(deck.minx_m + box.i0 * deck.dx_m)
    assert wd.miny_m == pytest.approx(deck.miny_m + box.j0 * deck.dy_m)
    assert wd.nz == box.nz
    assert float(wd.zf[wd.nz + 1]) == float(deck.zf[box.nz + 1])


def test_the_window_is_placed_on_its_centre_and_clamped_at_the_free_interior():
    g = geometry()
    cx, cy = centre_of(g)
    i0, j0, clamped = ncw.place_window(centre_x_m=cx, centre_y_m=cy, geometry=g, nx=7, ny=7)
    assert not clamped
    # The centre cell of the window is the cell the centre lies in.
    assert i0 + 3 == int(math.floor((cx - g.origin_x_m) / g.dx_m))
    far = ncw.place_window(centre_x_m=g.origin_x_m, centre_y_m=cy, geometry=g, nx=7, ny=7)
    assert far[2] is True
    assert far[0] == g.spec_zone + g.relax_zone + NG


def test_a_window_that_cannot_fit_the_free_interior_refuses_rather_than_clamping():
    g = geometry()
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.place_window(centre_x_m=0.0, centre_y_m=0.0, geometry=g, nx=g.nx, ny=g.ny)
    assert "parent-interpolated boundary row" in str(caught.value)


def test_the_bottom_ghost_is_parcel_Fs_own_words_and_the_w_ghost_is_the_reflection(tmp_path):
    g, deck, stream = open_stream(tmp_path)
    xc, yc = centre_of(g)
    stream.after_parent_step(SimpleNamespace(geometry=g, parent=SimpleNamespace(
        clock=SimpleNamespace(completed_steps=0)), criterion=None))
    arrays = child_arrays(deck, 0.25, xc=xc, yc=yc)
    assert stream.capture(arrays, child_step=1, model_time_s=0.25)
    stream.close()
    opened = ncw.read_window_stream(stream_root(tmp_path))
    words = opened.fields(0)
    # parcel.F:63-72's free-slip extrapolation, on the window's own block.
    c1, c2, c3 = (np.float32(c) for c in deck.cgs)
    expected = (c1 * words["uadv"][1] + c2 * words["uadv"][2]) + c3 * words["uadv"][3]
    assert np.array_equal(words["uadv"][0], expected)
    # u is independent of height, so the extrapolation reproduces the interior.
    assert np.allclose(words["uadv"][0], words["uadv"][1], atol=1e-5)
    # w's below-ground ghost is the reflection about the ground.
    assert np.array_equal(words["wadv"][0], np.float32(0.0) - words["wadv"][2])


# ---------------------------------------------------------------------------
# The stream: cadence, interval, held centres, receipt, integrity.
# ---------------------------------------------------------------------------
def test_the_stream_writes_one_record_per_child_step_with_a_readable_index(tmp_path):
    g, deck, stream = open_stream(tmp_path)
    xc, yc = centre_of(g)
    receipt = write_records(stream, deck, g, steps=8, xc=xc, yc=yc)
    assert receipt["records_written"] == 8
    assert receipt["arena_bytes"] == 8 * stream.record_bytes
    assert receipt["schema"] == ncw.NEST_CHILD_WINDOW_SCHEMA
    assert receipt["unvalidated"].startswith("UNVALIDATED against CM1")
    assert receipt["counts"]["records"] == 8
    opened = ncw.read_window_stream(stream_root(tmp_path))
    assert [r.child_step for r in opened.records] == list(range(1, 9))
    assert opened.times_s == pytest.approx([0.25 * n for n in range(1, 9)])
    assert opened.dt_s == pytest.approx(0.25)
    # Every record verifies against the SHA-256 its index row carries.
    for q in range(len(opened.records)):
        opened.fields(q, verify=True)


def test_a_rewritten_arena_is_caught_by_the_records_own_sha256(tmp_path):
    g, deck, stream = open_stream(tmp_path)
    xc, yc = centre_of(g)
    write_records(stream, deck, g, steps=4, xc=xc, yc=yc)
    arena = stream_root(tmp_path) / ncw.WINDOW_ARENA_FILENAME
    raw = bytearray(arena.read_bytes())
    raw[0:4] = np.float32(1234.0).tobytes()
    arena.write_bytes(bytes(raw))
    opened = ncw.read_window_stream(stream_root(tmp_path))
    with pytest.raises(ncw.NestChildWindowError) as caught:
        opened.fields(0, verify=True)
    assert "the trajectories through it would be through other words" in str(caught.value)


def test_a_stride_writes_every_nth_child_step_and_says_what_that_costs(tmp_path):
    g, deck, stream = open_stream(tmp_path, s=settings(stride_child_steps=2), name="strided")
    xc, yc = centre_of(g)
    receipt = write_records(stream, deck, g, steps=8, xc=xc, yc=yc)
    assert receipt["records_written"] == 4
    assert [r.child_step for r in ncw.read_window_stream(stream_root(tmp_path, "strided")).records] == [2, 4, 6, 8]
    assert receipt["bytes_per_model_minute"] == pytest.approx(
        stream.record_bytes * 60.0 / (0.25 * 2))


def test_records_outside_the_declared_interval_are_skipped_and_counted(tmp_path):
    g, deck, stream = open_stream(tmp_path, s=settings(start_s=0.5, end_s=1.0), name="interval")
    xc, yc = centre_of(g)
    receipt = write_records(stream, deck, g, steps=8, xc=xc, yc=yc)
    assert receipt["records_written"] == 3           # t = 0.50, 0.75, 1.00
    assert receipt["counts"]["skipped_outside_the_declared_interval"] == 5


def test_a_criterion_following_window_writes_nothing_until_a_centre_is_found(tmp_path):
    g = geometry()
    deck = child_deck(g)
    following = settings(follow="tornado-criterion", center_x_m=0.0, center_y_m=0.0)
    stream = ncw.NestChildWindowStream.build(settings=following, geometry=g, child_deck=deck,
                                             child_config=child_config(), parent_dt_s=0.75)
    stream.open(tmp_path / "following")
    xc, yc = centre_of(g)
    arrays = child_arrays(deck, 0.25, xc=xc, yc=yc)
    assert stream.capture(arrays, child_step=1, model_time_s=0.25) is False
    # A parent step whose sample found nothing still leaves the window unplaced.
    blank = SimpleNamespace(found=False, centre_i=-1, centre_j=-1)
    driver = SimpleNamespace(geometry=g, parent=SimpleNamespace(
        clock=SimpleNamespace(completed_steps=1, domainlocx_m=0.0, domainlocy_m=0.0)),
        criterion=SimpleNamespace(last_sample=blank))
    stream.after_parent_step(driver)
    assert stream.capture(arrays, child_step=2, model_time_s=0.5) is False
    # Then a centre is found, and the window sits on it.  A sample carries its
    # own GROUND position, which already holds the placement it was taken on.
    found = SimpleNamespace(found=True, centre_i=int((xc - g.origin_x_m) / g.dx_m),
                            centre_j=int((yc - g.origin_y_m) / g.dy_m),
                            ground_x_m=xc, ground_y_m=yc)
    driver.criterion.last_sample = found
    stream.after_parent_step(driver)
    assert stream.capture(arrays, child_step=3, model_time_s=0.75) is True
    # A later sample that finds nothing holds the last centre, and says so.
    driver.criterion.last_sample = blank
    stream.after_parent_step(driver)
    assert stream.capture(arrays, child_step=4, model_time_s=1.0) is True
    receipt = stream.close()
    assert receipt["counts"]["skipped_before_a_centre_was_found"] == 2
    assert receipt["counts"]["centre_held"] == 1
    records = ncw.read_window_stream(stream_root(tmp_path, "following")).records
    assert [r.centre_held for r in records] == [False, True]


def test_the_receipt_names_the_cm1_words_the_stream_carries_and_what_is_gm1s(tmp_path):
    g, deck, stream = open_stream(tmp_path)
    xc, yc = centre_of(g)
    receipt = write_records(stream, deck, g, steps=3, xc=xc, yc=yc)
    on_disk = json.loads((stream_root(tmp_path) / ncw.WINDOW_RECEIPT_FILENAME).read_text())
    assert on_disk["schema"] == receipt["schema"]
    assert "solve2.F:1290-1301" in on_disk["cm1_authority"]["advecting_velocity"]
    assert "parcel.F:59-96" in on_disk["cm1_authority"]["bottom_ghost"]
    assert on_disk["cm1_authority"]["nesting"].startswith("GM1 EXTENSION")
    assert "strided copy of the child's own padded carrier" in on_disk["gm1_choices"]["cut"]
    assert [f["name"] for f in on_disk["fields"]] == [
        "uadv", "vadv", "wadv", "rho", "prs", "th", "qv", "qhyd", "uten", "vten", "wten"]
    assert on_disk["base_columns"]["th0"][0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Backward through the stream: the model's own words, and the round trip.
# ---------------------------------------------------------------------------
def _seed(xc, yc, count=6, radius=300.0, height=175.0):
    angles = [2.0 * math.pi * q / count for q in range(count)]
    return np.stack([
        np.asarray([xc + radius * math.cos(a) for a in angles], np.float32),
        np.asarray([yc + radius * math.sin(a) for a in angles], np.float32),
        np.full(count, np.float32(height), np.float32),
    ])


def _stream_and_result(tmp_path, *, steps=40, dt_factor=1, name="traj"):
    g, deck, stream = open_stream(tmp_path, name=name)
    xc, yc = centre_of(g)
    write_records(stream, deck, g, steps=steps, xc=xc, yc=yc)
    opened = ncw.read_window_stream(stream_root(tmp_path, name))
    dt = opened.dt_s * dt_factor
    frames = opened.frames()[dt_factor - 1::dt_factor]
    start = frames[-1].model_time_s
    end = frames[0].model_time_s - dt
    request = nb.BackwardRequest(None, start, end, dt, wind_source="model-step",
                                 positions=_seed(xc, yc))
    return opened, frames, request, nb.integrate_backward(frames, request), (xc, yc), start


def test_backward_through_the_stream_reproduces_the_closed_form_trajectory(tmp_path):
    opened, frames, request, result, (xc, yc), start = _stream_and_result(tmp_path)
    assert (result.status == nb.STATUS_ACTIVE).all()
    end = result.times_s[-1]
    errors = []
    for q in range(result.rows.shape[2]):
        x1, y1, z1 = (float(result.rows[0, r, q]) for r in range(3))
        # Where the closed form says the parcel that ends at (x1, y1, z1) at
        # `start` was at `end`: integrate the closed form backward in time.
        back = analytic_position(x1, y1, z1, start, end, xc=xc, yc=yc)
        got = tuple(float(result.rows[-1, r, q]) for r in range(3))
        errors.append(math.dist(back, got))
    assert max(errors) < 0.02, errors      # 2 cm over 10 s of a 15 m/s vortex


def test_the_time_interpolation_error_of_the_model_step_source_is_zero_not_estimated(tmp_path):
    _opened, _frames, _request, result, _c, _s = _stream_and_result(tmp_path, name="zero")
    row = result.rows[:-1, 8]
    placed = row > np.float32(-1.0e29)
    assert placed.any()
    assert (row[placed] == 0.0).all()


def test_the_round_trip_residual_is_measured_and_closes_at_the_seed(tmp_path):
    _opened, _frames, _request, result, _c, _s = _stream_and_result(tmp_path, name="round")
    residual = result.rows[0, 10]
    assert (residual > np.float32(-1.0e29)).all()
    assert residual.max() < 0.02          # the RK2 scheme's own irreversibility
    # The forward pass starts where the backward pass ended, so the residual
    # there is exactly zero and grows back toward the seed.
    assert (result.rows[-1, 10] == 0.0).all()
    # The sensitivity amplification is 1 at the target and grows backward.
    amp = result.rows[:, 11]
    assert (amp[0] == 1.0).all()
    assert (np.diff(amp, axis=0) >= -1e-6).all()
    assert amp[-1].min() > 1.0
    # The published estimate is the round trip plus half a window cell carried
    # back through that amplification -- not the round trip alone, which in a
    # locally linear flow is nearly blind (RK2 forward and backward are
    # inverses to O(dt^4) there).
    half_cell = 0.5 * _opened.receipt["child"]["dx_m"]
    assert result.rows[0, 9] == pytest.approx(
        result.rows[0, 10] + half_cell * result.rows[0, 11], rel=1e-5)
    assert result.rows[-1, 9] == pytest.approx(
        result.rows[-1, 10] + half_cell * result.rows[-1, 11], rel=1e-5)


def test_a_stream_written_every_fourth_step_is_refused_by_the_model_step_source(tmp_path):
    """A record is the average over ONE child step; four of them apart it is not the step's."""

    g, deck, stream = open_stream(tmp_path, s=settings(stride_child_steps=4), name="strided4")
    xc, yc = centre_of(g)
    write_records(stream, deck, g, steps=32, xc=xc, yc=yc)
    opened = ncw.read_window_stream(stream_root(tmp_path, "strided4"))
    frames = opened.frames()
    request = nb.BackwardRequest(None, frames[-1].model_time_s, frames[0].model_time_s,
                                 opened.dt_s, wind_source="model-step",
                                 positions=_seed(xc, yc))
    with pytest.raises(nb.BackwardTrajectoryError) as caught:
        nb.integrate_backward(frames, request)
    assert "written every 4 child steps" in str(caught.value)
    assert "would say the time-interpolation error is zero" in str(caught.value)
    assert "history-blend" in str(caught.value)


def test_what_the_stride_refusal_is_protecting_against(tmp_path):
    """The measurement that says why the stream is written every child step.

    The same closed-form flow and the same window words, read with the same
    integrator at one, four and eight times the record interval.  The four- and
    eight-times cases are the counterfactual the refusal above forbids -- their
    records are built here with the ``child_step`` key removed, so the stride
    check cannot see them -- and the point is exactly how much they cost: two
    orders of magnitude of trajectory error for four times the interval.  That
    is the same T-dependence that made a 30 s history interval worth 800-1200 m
    on the real child.
    """

    from dataclasses import replace

    g, deck, stream = open_stream(tmp_path, name="compare")
    xc, yc = centre_of(g)
    write_records(stream, deck, g, steps=40, xc=xc, yc=yc)
    opened = ncw.read_window_stream(stream_root(tmp_path, "compare"))
    frames = opened.frames()
    seeds = _seed(xc, yc)
    start_t = frames[-1].model_time_s

    def error_at(factor):
        dt = opened.dt_s * factor
        chosen = frames[factor - 1::factor]
        if factor != 1:
            chosen = [replace(f, placement={k: v for k, v in f.placement.items()
                                            if k != "child_step"}) for f in chosen]
        end_t = chosen[0].model_time_s
        result = nb.integrate_backward(
            chosen, nb.BackwardRequest(None, start_t, end_t, dt, wind_source="model-step",
                                       positions=seeds.copy()))
        worst = 0.0
        for q in range(result.rows.shape[2]):
            x1, y1, z1 = (float(result.rows[0, r, q]) for r in range(3))
            want = analytic_position(x1, y1, z1, start_t, result.times_s[-1], xc=xc, yc=yc)
            got = tuple(float(result.rows[-1, r, q]) for r in range(3))
            worst = max(worst, math.dist(want, got))
        return worst

    fine, four, eight = error_at(1), error_at(4), error_at(8)
    assert four > 50.0 * fine, (fine, four)
    assert eight > four
    # And a stream actually written at that stride is refused rather than read
    # this way; only this test's stripped placements get past the check.
    assert all("child_step" in f.placement for f in frames)


def test_a_seed_outside_the_window_is_refused_rather_than_extrapolated(tmp_path):
    opened, frames, _request, _result, (xc, yc), start = _stream_and_result(
        tmp_path, name="outside")
    far = _seed(xc + 5000.0, yc)
    request = nb.BackwardRequest(None, start, start - 1.0, opened.dt_s,
                                 wind_source="model-step", positions=far)
    with pytest.raises(nb.BackwardTrajectoryError) as caught:
        nb.integrate_backward(frames, request)
    assert "a trajectory cannot start where the stream holds no wind" in str(caught.value)


def test_a_dt_that_is_not_the_streams_own_record_spacing_is_refused(tmp_path):
    opened, frames, _request, _result, _c, start = _stream_and_result(tmp_path, name="dt")
    request = nb.BackwardRequest(None, start, start - 1.0, opened.dt_s * 2.0,
                                 wind_source="model-step", positions=_seed(*_c_or(opened)))
    with pytest.raises(nb.BackwardTrajectoryError) as caught:
        nb.integrate_backward(frames, request)
    assert "skip a record's wind or reuse one" in str(caught.value)


def _c_or(opened):
    r = opened.records[0]
    return (r.centre_x_m, r.centre_y_m)


def test_an_unadmitted_wind_source_is_refused_by_name():
    with pytest.raises(nb.BackwardTrajectoryError) as caught:
        nb.BackwardRequest(None, 10.0, 0.0, 1.0, wind_source="guess",
                           positions=np.zeros((3, 2), np.float32))
    assert "the published error estimate would be the wrong one" in str(caught.value)


def test_a_gap_in_the_stream_is_refused_rather_than_reversed_as_one_step(tmp_path):
    opened, frames, _request, _result, _c, _s = _stream_and_result(tmp_path, name="gap")
    holed = frames[:5] + frames[6:]
    request = nb.BackwardRequest(None, holed[-1].model_time_s, holed[0].model_time_s - opened.dt_s,
                                 opened.dt_s, wind_source="model-step",
                                 positions=_seed(*_c_or(opened)))
    with pytest.raises(nb.BackwardTrajectoryError) as caught:
        nb.integrate_backward(holed, request)
    assert "a missing record would silently reverse two" in str(caught.value)


# ---------------------------------------------------------------------------
# The door, on the shipped deck.
# ---------------------------------------------------------------------------
def _example():
    from pathlib import Path

    from cm1core.config import load_config

    root = Path(__file__).resolve().parents[1]
    return load_config(root / "examples" / "cartesian-nested-tornado-window.toml")


def test_the_shipped_deck_is_admitted_and_its_stated_disk_cost_is_the_real_one():
    from cm1core.cartesian_door import require_nest
    from cm1core.nest_vertical import levels_of_grid

    config = _example()
    binding = require_nest(config)             # runs the window's own admission
    s = config.lagrangian.nest_child_window
    assert s.follows_criterion and config.nest.tornado_criterion is not None
    levels = levels_of_grid(binding.child.grid)
    nx, ny, nz = ncw.window_cells(s, dx_m=binding.geometry.dx_m,
                                  dy_m=binding.geometry.dy_m, zf=levels.zf)
    assert (nx, ny, nz) == (61, 61, 22)
    assert float(levels.zf[nz + 1]) < float(levels.zf[binding.child.grid.nz + 1])
    box = ncw.WindowBox(0, 0, nx, ny, nz)
    cost = ncw.stream_bytes_per_model_minute(box, child_dt_s=binding.child.dt_s, stride=1,
                                             surface=int(config.closures.bbc) == 3)
    assert cost <= s.budget_bytes_per_model_minute
    # The deck's own comment states these; they must be the numbers, not a hope.
    assert round(box.record_bytes(surface=True) / 1e6, 3) == 4.290
    assert round(cost / 1e9, 3) == 2.059


def test_a_window_without_a_nest_is_refused_at_the_door():
    import copy
    import tomllib
    from pathlib import Path

    from cm1core.config import parse_mapping
    from cm1core.errors import CM1Refusal

    root = Path(__file__).resolve().parents[1]
    raw = tomllib.loads(
        (root / "examples" / "cartesian-nested-tornado-window.toml").read_text())
    raw = copy.deepcopy(raw)
    del raw["nest"]
    raw["lagrangian"]["nest_child_window"]["follow"] = "fixed"
    raw["lagrangian"]["nest_child_window"]["center_x_m"] = 40000.0
    raw["lagrangian"]["nest_child_window"]["center_y_m"] = 40000.0
    from cm1core.cartesian_door import require_nest

    with pytest.raises(CM1Refusal) as caught:
        require_nest(parse_mapping(raw))
    assert caught.value.detail.code == "CM1-CARTESIAN-NEST-WINDOW-NO-NEST"
    assert "100 m words while its receipt named the child's" in caught.value.detail.reason


def test_a_window_over_its_declared_budget_is_refused_at_the_door_with_the_number():
    import copy
    import tomllib
    from pathlib import Path

    from cm1core.cartesian_door import require_nest
    from cm1core.config import parse_mapping
    from cm1core.errors import CM1Refusal

    root = Path(__file__).resolve().parents[1]
    raw = copy.deepcopy(tomllib.loads(
        (root / "examples" / "cartesian-nested-tornado-window.toml").read_text()))
    raw["lagrangian"]["nest_child_window"]["budget_bytes_per_model_minute"] = 10_000_000
    with pytest.raises(CM1Refusal) as caught:
        require_nest(parse_mapping(raw))
    assert caught.value.detail.code == "CM1-CARTESIAN-NEST-WINDOW-DISK-BUDGET"
    assert "61x61x22 child cells" in caught.value.detail.reason
    assert caught.value.detail.option.endswith("budget_bytes_per_model_minute")


# ---------------------------------------------------------------------------
# The runtime call sites: the same moment in the same step as the parcels.
# ---------------------------------------------------------------------------
def test_the_child_runtime_captures_the_window_at_the_step_it_completes(tmp_path):
    """The hook must label the record with the END of the step whose wind it holds."""

    from cm1core.nest_live import make_child_runtime_class

    class Base:
        def execute(self, operation, *, step):
            self.executed = (operation, step)

        def _capture_parcel_velocity(self):
            self.captured = True

        def _advance_lagrangian(self):
            self.advanced = True

    g, deck, stream = open_stream(tmp_path, name="callsite")
    xc, yc = centre_of(g)
    runtime = make_child_runtime_class(Base)()
    runtime.nest_parcels = None
    runtime.nest_child_window = stream
    runtime.closure = None                       # bbc = 1: no zntmp to read
    runtime.config = child_config()
    carriers = child_arrays(deck, 0.125, xc=xc, yc=yc)
    runtime.dynamics = SimpleNamespace(
        arrays={k: carriers[k] for k in ("rru", "rrv", "rrw")},
        fields={k: carriers[k] for k in ("rho", "prs", "tha", "qa")},
        held={k: carriers[k] for k in ("uten", "vten", "wten")},
        base_profiles={"th0": carriers["th0_profile"], "qv0": carriers["qv0_profile"]},
    )
    # gm1_runtime.RuntimeClock.step increments after the phases, so at capture
    # time the clock still reads the step before the one being completed.
    runtime.clock = SimpleNamespace(completed_steps=7, model_seconds=7 * 0.25)
    runtime._capture_parcel_velocity()
    assert runtime.captured is True              # CM1's own capture still ran
    receipt = stream.close()
    assert receipt["records_written"] == 1
    record = ncw.read_window_stream(stream_root(tmp_path, "callsite")).records[0]
    assert record.child_step == 8
    assert record.model_time_s == pytest.approx(8 * 0.25)


def test_the_driver_places_the_window_after_the_criterion_feedback_and_any_move():
    """The index origin the next child steps are cut at must be the one they run on."""

    import inspect

    from cm1core.nest_live import NestDriver

    source = inspect.getsource(NestDriver.after_parent_step)
    order = [source.index(call) for call in (
        "self.criterion.after_parent_step(self)",
        "self.feed_back()",
        "self.mover.after_parent_step(self)",
        "self.parcels.after_parent_step(self)",
        "self.window.after_parent_step(self)",
    )]
    assert order == sorted(order)
    build = inspect.getsource(NestDriver.__init__)
    assert "lagrangian.nest_child_window" in build
    assert "child.nest_child_window = self.window" in build


def test_the_integration_opens_closes_and_abandons_the_stream_where_the_track_is():
    import inspect

    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime

    integrate = inspect.getsource(CudaCartesianRuntime.integrate)
    # Opened beside the child's parcel track, closed into the receipt beside
    # it, and stopped without a receipt on the failure path.
    assert integrate.index("self._nest_parcel_track(root)") < integrate.index(
        "self._nest_child_window(root)")
    assert "**self._nest_child_window()," in integrate
    assert "window.abandon()" in integrate


def test_an_abandoned_stream_leaves_nothing_that_can_be_read_as_a_whole_one(tmp_path):
    """A run that died wrote no receipt; a directory without one is refused by name."""

    g, deck, stream = open_stream(tmp_path, name="abandoned")
    xc, yc = centre_of(g)
    for n in (1, 2, 3):
        stream.capture(child_arrays(deck, n * 0.25 - 0.125, xc=xc, yc=yc),
                       child_step=n, model_time_s=n * 0.25)
    stream.abandon()
    root = stream_root(tmp_path, "abandoned")
    assert (root / ncw.WINDOW_ARENA_FILENAME).exists()
    assert not (root / ncw.WINDOW_RECEIPT_FILENAME).exists()
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.read_window_stream(root)
    assert "cannot be read as anything" in str(caught.value)
    assert stream.close() is None            # closing an abandoned stream is a no-op


def test_a_moved_nest_moves_the_streams_ground_origin_with_it(tmp_path):
    """A following nest re-places the child; the record's ground origin must follow."""

    first = geometry(parent_i0=6, parent_j0=6)
    moved = geometry(parent_i0=8, parent_j0=6)          # two parent cells east
    deck = child_deck(first)
    s = settings(center_x_m=CHILD_MIDDLE_M, center_y_m=CHILD_MIDDLE_M)
    stream = ncw.NestChildWindowStream.build(settings=s, geometry=first, child_deck=deck,
                                             child_config=child_config(), parent_dt_s=0.75)
    stream.open(tmp_path / "moved")
    xc, yc = centre_of(first)
    arrays = child_arrays(deck, 0.125, xc=xc, yc=yc)
    clock = SimpleNamespace(completed_steps=1)
    stream.after_parent_step(SimpleNamespace(geometry=first, criterion=None,
                                             parent=SimpleNamespace(clock=clock)))
    assert stream.capture(arrays, child_step=1, model_time_s=0.25)
    stream.after_parent_step(SimpleNamespace(geometry=moved, criterion=None,
                                             parent=SimpleNamespace(clock=clock)))
    assert stream.capture(arrays, child_step=2, model_time_s=0.5)
    stream.close()
    records = ncw.read_window_stream(stream_root(tmp_path, "moved")).records
    # The centre is fixed in the parent's frame, so the window's ground origin
    # is unchanged while its index into the moved child lattice shifts by the
    # cells the child moved.
    shift_cells = int((moved.origin_x_m - first.origin_x_m) / first.dx_m)
    assert shift_cells > 0
    assert records[1].i0 == records[0].i0 - shift_cells
    assert records[1].origin_x_m == pytest.approx(records[0].origin_x_m)
    assert records[1].origin_y_m == pytest.approx(records[0].origin_y_m)


def test_a_centre_sampled_before_a_move_is_not_re_derived_on_the_moved_lattice(tmp_path):
    """The criterion samples before the move; its index belongs to the old placement.

    Re-deriving the centre from the geometry the window is cut on would shift it
    by exactly the cells the nest just moved -- silently, because the sample's
    corner index is a perfectly good index on both lattices.
    """

    before = geometry(parent_i0=6, parent_j0=6)
    after = geometry(parent_i0=7, parent_j0=6)          # one parent cell east
    deck = child_deck(before)
    s = settings(follow="tornado-criterion", center_x_m=0.0, center_y_m=0.0)
    stream = ncw.NestChildWindowStream.build(settings=s, geometry=before, child_deck=deck,
                                             child_config=child_config(), parent_dt_s=0.75)
    stream.open(tmp_path / "presample")
    xc, yc = centre_of(before)
    # The sample was taken on `before`: its corner index is relative to that
    # placement, and its ground position already carries it.
    sample = SimpleNamespace(
        found=True,
        centre_i=int((xc - before.origin_x_m) / before.dx_m),
        centre_j=int((yc - before.origin_y_m) / before.dy_m),
        ground_x_m=xc + 1234.0, ground_y_m=yc + 1234.0)   # domainloc = 1234 m
    clock = SimpleNamespace(completed_steps=3, domainlocx_m=1234.0, domainlocy_m=1234.0)
    stream.after_parent_step(SimpleNamespace(
        geometry=after, parent=SimpleNamespace(clock=clock),
        criterion=SimpleNamespace(last_sample=sample)))
    assert stream.centre == pytest.approx((xc, yc))
    assert stream.capture(child_arrays(deck, 0.125, xc=xc, yc=yc),
                          child_step=1, model_time_s=0.25)
    stream.close()
    record = ncw.read_window_stream(stream_root(tmp_path, "presample")).records[0]
    assert record.centre_x_m == pytest.approx(xc)
    # The index is relative to the MOVED lattice, and the ground origin the
    # record publishes puts the window back where the centre actually is.
    assert record.clamped is False
    assert record.origin_x_m == pytest.approx(
        after.origin_x_m + record.i0 * after.dx_m)
    # xc lies inside the window's own centre cell, on the moved lattice.
    west = record.origin_x_m + (record.nx // 2) * after.dx_m
    assert 0.0 <= xc - west < after.dx_m


# ---------------------------------------------------------------------------
# A strided stream, read with the blend at the step midpoints; and forward.
# ---------------------------------------------------------------------------
def _strided(tmp_path, name, stride=4, steps=64):
    g, deck, stream = open_stream(tmp_path, s=settings(stride_child_steps=stride), name=name)
    xc, yc = centre_of(g)
    write_records(stream, deck, g, steps=steps, xc=xc, yc=yc)
    return ncw.read_window_stream(stream_root(tmp_path, name)), (xc, yc)


def _worst(result, xc, yc):
    worst = 0.0
    for q in range(result.rows.shape[2]):
        x1, y1, z1 = (float(result.rows[0, r, q]) for r in range(3))
        want = analytic_position(x1, y1, z1, result.times_s[0], result.times_s[-1], xc=xc, yc=yc)
        got = tuple(float(result.rows[-1, r, q]) for r in range(3))
        worst = max(worst, math.dist(want, got))
    return worst


def test_a_strided_stream_blended_at_step_midpoints_reproduces_the_closed_form(tmp_path):
    opened, (xc, yc) = _strided(tmp_path, "blend")
    shift = opened.midpoint_shift_s
    frames = opened.frames(time_shift_s=shift)
    child_dt = opened.receipt["child"]["dt_s"]
    start, end = frames[-2].model_time_s, frames[1].model_time_s
    request = nb.BackwardRequest(None, start, end, child_dt, positions=_seed(xc, yc))
    result = nb.integrate_backward(frames, request)
    assert (result.status == nb.STATUS_ACTIVE).all()
    good = _worst(result, xc, yc)
    # Labelled at the step END instead, every blended wind is half a step late.
    late = nb.integrate_backward(opened.frames(), nb.BackwardRequest(
        None, start, end, child_dt, positions=_seed(xc, yc)))
    assert good < 0.05, good
    assert _worst(late, xc, yc) > 5.0 * good


def test_forward_parcels_through_the_same_records_reproduce_the_closed_form(tmp_path):
    opened, (xc, yc) = _strided(tmp_path, "forward")
    frames = opened.frames(time_shift_s=opened.midpoint_shift_s)
    child_dt = opened.receipt["child"]["dt_s"]
    start, end = frames[1].model_time_s, frames[-2].model_time_s
    request = nb.BackwardRequest(None, start, end, child_dt, positions=_seed(xc, yc),
                                 direction="forward")
    result = nb.integrate_backward(frames, request)
    assert result.times_s[-1] == pytest.approx(end)
    assert (result.status == nb.STATUS_ACTIVE).all()
    assert _worst(result, xc, yc) < 0.05
    with pytest.raises(nb.BackwardTrajectoryError) as caught:
        nb.BackwardRequest(None, start, end, child_dt, positions=_seed(xc, yc),
                           direction="forward", wind_source="model-step")
    assert "reversal of the model's own step" in str(caught.value)


def test_a_window_that_jumps_away_for_a_few_records_is_bridged_per_parcel(tmp_path):
    """The criterion's centre can jump to another vortex for a few records.

    Those records then hold none of the parcels.  The nearest-frame blend
    stops every parcel there; the containing bracket blends across the gap with
    the frames that do hold each parcel, and still lands on the closed form.
    """

    from dataclasses import replace

    opened, (xc, yc) = _strided(tmp_path, "jump")
    frames = opened.frames(time_shift_s=opened.midpoint_shift_s)
    away = []
    for q, f in enumerate(frames):
        if 6 <= q <= 8:                                   # 1.5 s with the window elsewhere
            deck = replace(f.deck, minx_m=f.deck.minx_m + 50_000.0)
            placement = dict(f.placement, origin_x_m=f.placement["origin_x_m"] + 50_000.0)
            f = replace(f, deck=deck, placement=placement)
        away.append(f)
    child_dt = opened.receipt["child"]["dt_s"]
    start, end = away[-2].model_time_s, away[1].model_time_s
    nearest = nb.integrate_backward(away, nb.BackwardRequest(
        None, start, end, child_dt, positions=_seed(xc, yc)))
    assert (nearest.status != nb.STATUS_ACTIVE).all()
    bridged = nb.integrate_backward(away, nb.BackwardRequest(
        None, start, end, child_dt, positions=_seed(xc, yc), bracket="containing"))
    assert (bridged.status == nb.STATUS_ACTIVE).all()
    assert _worst(bridged, xc, yc) < 1.0
    # With nothing missing, the containing bracket IS the nearest-frame blend.
    same = nb.integrate_backward(frames, nb.BackwardRequest(
        None, start, end, child_dt, positions=_seed(xc, yc), bracket="containing"))
    plain = nb.integrate_backward(frames, nb.BackwardRequest(
        None, start, end, child_dt, positions=_seed(xc, yc)))
    assert np.array_equal(same.rows[:, 0:3], plain.rows[:, 0:3])
    # and the wider span shows in the time-interpolation bound
    assert np.nanmax(bridged.rows[:-1, 8]) > np.nanmax(plain.rows[:-1, 8])


# ---------------------------------------------------------------------------
# Schema v3: Morrison's species in qhyd's slot.
# ---------------------------------------------------------------------------
MORRISON = "morrison-mp10"


def _morrison_carriers(deck, t, *, xc, yc, seed=17):
    """Analytic winds plus distinct, positive words in every one of Morrison's ten rows."""

    arrays = child_arrays(deck, t, xc=xc, yc=yc)
    arrays.update(physical_carriers(deck, rng=np.random.default_rng(seed), numq=10))
    arrays["qa"] = np.abs(arrays["qa"]) * np.float32(1.0e-3)
    arrays["qa"][6:] = np.float32(1.0e6)      # the number rows: kg-1 words, never condensate
    return arrays


def test_the_layout_is_v2_for_wsm6_and_v3_with_rows_found_by_name_for_morrison():
    wsm6 = ncw.window_layout("wsm6-mp6")
    assert wsm6.schema == ncw.NEST_CHILD_WINDOW_SCHEMA and not wsm6.species
    assert wsm6.fields == ncw.WINDOW_FIELDS
    morrison = ncw.window_layout(MORRISON)
    assert morrison.schema == ncw.NEST_CHILD_WINDOW_SCHEMA_V3 and morrison.version == 3
    # param.F:3414-3473: qv qc qr qi qs qg nci ncs ncr ncg.
    assert dict(morrison.species_rows) == {"qc": (1,), "qr": (2,), "qis": (3, 4), "qg": (5,)}
    assert morrison.numq == 10
    names = [name for name, *_ in morrison.fields]
    assert names == ["uadv", "vadv", "wadv", "rho", "prs", "th", "qv",
                     "qc", "qr", "qis", "qg", "uten", "vten", "wten"]
    assert "qhyd" not in names


def test_a_species_the_component_does_not_carry_is_refused_by_name(monkeypatch):
    from cm1core import moisture_species as ms

    real = ms.moisture_species_profile(MORRISON)
    fake = SimpleNamespace(names=tuple(n for n in real.names if n != "qs"),
                           mass_names=tuple(n for n in real.mass_names if n != "qs"),
                           numq=real.numq - 1, axis_authority="test")
    monkeypatch.setattr(ms, "moisture_species_profile", lambda cid: fake)
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.window_layout(MORRISON)
    assert "'qis'" in str(caught.value) and "qs" in str(caught.value)
    with pytest.raises(ncw.NestChildWindowError):
        ncw.window_layout(None)


def test_a_dry_or_unadmitted_component_is_refused_at_the_door_under_the_window_code():
    # 'off' (a dry child) and an untranscribed component carry no species
    # table; before the door checked the component they reached the first
    # capture and failed there, mid-run.
    for component in ("off", "kessler"):
        with pytest.raises(ncw.NestChildWindowError):
            ncw.window_layout(component)
        defect = _admission(component=component)
        assert defect is not None and defect[0] == "SPECIES", (component, defect)


def test_a_v2_component_not_leading_with_cm1s_mass_rows_is_refused(monkeypatch):
    from cm1core import moisture_species as ms

    fake = SimpleNamespace(names=("qv", "qc", "qr", "nci", "qi", "qs", "qg"),
                           mass_names=("qv", "qc", "qr", "qi", "qs", "qg"), numq=7,
                           axis_authority="test")
    monkeypatch.setattr(ms, "moisture_species_profile", lambda cid: fake)
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.window_layout("thompson-mp8")
    assert "condensate loading" in str(caught.value)


def test_the_v3_cut_is_the_childs_own_species_rows_and_qis_is_qi_plus_qs():
    g = geometry()
    deck = child_deck(g)
    xc, yc = centre_of(g)
    arrays = _morrison_carriers(deck, 3.0, xc=xc, yc=yc)
    box = ncw.WindowBox(9, 11, 7, 5, 10)
    layout = ncw.window_layout(MORRISON)
    cut = ncw.cut_window(np, arrays, box, surface=None, species_rows=layout.species_rows)
    assert "qhyd" not in cut
    shapes = box.shapes(surface=False, species=True)
    qa = arrays["qa"]
    rng = np.random.default_rng(3)
    for name in ("qc", "qr", "qis", "qg"):
        assert cut[name].shape == shapes[name]
    for _ in range(40):
        k, j, i = (int(rng.integers(0, n)) for n in shapes["qc"])
        at = (k, box.j0 - 1 + j, box.i0 - 1 + i)
        assert cut["qc"][k, j, i] == qa[(1, *at)]
        assert cut["qr"][k, j, i] == qa[(2, *at)]
        assert cut["qis"][k, j, i] == np.float32(qa[(3, *at)] + qa[(4, *at)])
        assert cut["qg"][k, j, i] == qa[(5, *at)]
    # And the rebuilt loading is parcel.F:745-775's own sum to float rounding.
    v2 = ncw.cut_window(np, arrays, box, surface=None)
    rebuilt = ncw.rebuild_qhyd(cut)
    assert np.allclose(rebuilt, v2["qhyd"], rtol=4 * np.finfo(np.float32).eps, atol=0.0)


def test_a_v3_record_costs_three_more_ring_fields_and_the_door_prices_them():
    box = ncw.WindowBox(0, 0, 7, 7, 10)
    ring = 11 * 9 * 9
    assert box.record_words(surface=False, species=True) == \
        box.record_words(surface=False) + 3 * ring
    assert box.record_words(surface=True, species=True) == \
        box.record_words(surface=True) + 3 * ring
    v2 = ncw.stream_bytes_per_model_minute(box, child_dt_s=0.1, stride=1, surface=True)
    v3 = ncw.stream_bytes_per_model_minute(box, child_dt_s=0.1, stride=1, surface=True,
                                           species=True)
    assert v3 - v2 == pytest.approx(4 * 3 * ring * 600.0)
    # The budget that just admits the v2 stream refuses the same window under Morrison.
    admitted = _admission()
    assert admitted is None
    s = settings()
    nx, ny, nz = ncw.window_cells(s, dx_m=geometry().dx_m, dy_m=geometry().dy_m,
                                  zf=child_levels().zf)
    full = ncw.WindowBox(0, 0, nx, ny, nz)
    v2_cost = ncw.stream_bytes_per_model_minute(full, child_dt_s=0.25, stride=1, surface=False)
    tight = settings(budget_bytes_per_model_minute=int(math.ceil(v2_cost)))
    assert _admission(settings=tight) is None
    defect = _admission(settings=tight, component=MORRISON)
    assert defect is not None and defect[0] == "DISK-BUDGET"
    assert "schema-v3" in defect[2]
    assert _admission(component=None)[0] == "SPECIES"


def test_a_wsm6_stream_is_byte_identical_to_schema_v2(tmp_path):
    g, deck, stream = open_stream(tmp_path)
    assert stream.schema == ncw.NEST_CHILD_WINDOW_SCHEMA and not stream.species
    xc, yc = centre_of(g)
    receipt = write_records(stream, deck, g, steps=2, xc=xc, yc=yc)
    assert "schema_version" not in receipt and "species" not in receipt
    assert [f["name"] for f in receipt["fields"]] == [n for n, *_ in ncw.WINDOW_FIELDS]
    # The record is exactly v2's words in v2's order: rebuild it by hand.
    arrays = child_arrays(deck, 2 * 0.25 - 0.125, xc=xc, yc=yc)
    opened = ncw.read_window_stream(stream_root(tmp_path))
    words = opened.fields(1)
    assert opened.schema_version == 2 and not opened.has_species
    box = ncw.WindowBox(opened.records[1].i0, opened.records[1].j0, stream.nx, stream.ny, stream.nz)
    cut = ncw.cut_window(np, arrays, box)
    for name in ("rho", "prs", "th", "qv", "qhyd", "uten", "vten", "wten"):
        assert np.array_equal(words[name], cut[name]), name
    assert receipt["record_bytes"] == box.record_bytes(surface=False)


def test_a_morrison_stream_round_trips_its_species_and_the_reader_rebuilds_qhyd(tmp_path):
    config = child_config(microphysics=MORRISON)
    g, deck, stream = open_stream(tmp_path, config=config)
    assert stream.schema == ncw.NEST_CHILD_WINDOW_SCHEMA_V3
    xc, yc = centre_of(g)
    carriers = []
    for n in (1, 2, 3):
        arrays = _morrison_carriers(deck, (n - 0.5) * 0.25, xc=xc, yc=yc, seed=100 + n)
        carriers.append(arrays)
        assert stream.capture(arrays, child_step=n, model_time_s=n * 0.25)
    receipt = stream.close()
    assert receipt["schema"] == ncw.NEST_CHILD_WINDOW_SCHEMA_V3
    assert receipt["schema_version"] == 3
    assert receipt["field_names"] == [n for n, *_ in ncw.WINDOW_FIELDS_V3]
    assert receipt["species"]["component_id"] == MORRISON
    assert receipt["species"]["qa_rows_zero_based"]["qis"] == [3, 4]
    assert receipt["arena_bytes"] == 3 * stream.record_bytes
    box0 = ncw.WindowBox(0, 0, stream.nx, stream.ny, stream.nz)
    assert stream.record_bytes == box0.record_bytes(surface=False, species=True)
    opened = ncw.read_window_stream(stream_root(tmp_path))
    assert opened.schema_version == 3 and opened.has_species
    for q, arrays in enumerate(carriers):
        record = opened.records[q]
        words = opened.fields(q, verify=True)
        box = ncw.WindowBox(record.i0, record.j0, record.nx, record.ny, record.nz)
        want = ncw.cut_window(np, arrays, box,
                              species_rows=ncw.window_layout(MORRISON).species_rows)
        for name in ("qc", "qr", "qis", "qg", "qv", "rho", "prs", "th"):
            assert np.array_equal(words[name], want[name]), name
        v2 = ncw.cut_window(np, arrays, box)
        # The number rows never reach the loading, and qhyd is parcel.F's sum to rounding.
        assert float(np.max(words["qhyd"])) < 1.0
        assert np.allclose(words["qhyd"], v2["qhyd"], rtol=4 * np.finfo(np.float32).eps, atol=0)
    # Backward trajectories read a v3 stream exactly as a v2 one.
    frame = opened.frame(0)
    assert frame.u.shape == words["uadv"].shape


def test_a_morrison_child_with_another_qa_layout_is_refused(tmp_path):
    config = child_config(microphysics=MORRISON)
    g, deck, stream = open_stream(tmp_path, config=config)
    xc, yc = centre_of(g)
    arrays = child_arrays(deck, 0.125, xc=xc, yc=yc)     # six rows: WSM6's layout
    with pytest.raises(ncw.NestChildWindowError) as caught:
        stream.capture(arrays, child_step=1, model_time_s=0.25)
    assert "not the 10" in str(caught.value)


def test_a_receipt_whose_fields_disagree_with_its_schema_is_refused(tmp_path):
    config = child_config(microphysics=MORRISON)
    g, deck, stream = open_stream(tmp_path, config=config)
    xc, yc = centre_of(g)
    stream.capture(_morrison_carriers(deck, 0.125, xc=xc, yc=yc), child_step=1,
                   model_time_s=0.25)
    stream.close()
    path = stream_root(tmp_path) / ncw.WINDOW_RECEIPT_FILENAME
    receipt = json.loads(path.read_text())
    receipt["schema"] = ncw.NEST_CHILD_WINDOW_SCHEMA
    path.write_text(json.dumps(receipt))
    with pytest.raises(ncw.NestChildWindowError) as caught:
        ncw.read_window_stream(stream_root(tmp_path))
    assert "lists fields" in str(caught.value)


def test_the_rayleigh_layer_of_an_unstretched_grid_starts_below_nz_dz_not_below_zero():
    closures = SimpleNamespace(upper_damping=1, upper_damping_depth_m=5000.0)
    flat = SimpleNamespace(closures=closures,
                           grid=SimpleNamespace(stretch_z=False, nz=40, dz_m=500.0, ztop_m=0.0))
    assert ncw._damping_bottom_m(flat) == pytest.approx(15000.0)
    stretched = SimpleNamespace(closures=closures,
                                grid=SimpleNamespace(stretch_z=1, nz=80, dz_m=100.0,
                                                     ztop_m=17600.0))
    assert ncw._damping_bottom_m(stretched) == pytest.approx(12600.0)


def test_a_nest_centre_window_rides_the_moving_nest(tmp_path):
    """``follow = "nest-centre"``: the window stays on the child's centre across moves.

    The mode a 1 km, 2 m child needs: it cannot hold the tornado criterion's
    1000 m edge margin, so the storm-following nest's tracker places the vortex
    and the window follows the nest.
    """

    first = geometry(parent_i0=6, parent_j0=6)
    moved = geometry(parent_i0=8, parent_j0=7)
    deck = child_deck(first)
    s = settings(follow="nest-centre", center_x_m=0.0, center_y_m=0.0)
    assert s.follows_nest and not s.follows_criterion
    assert "center_x_m/center_y_m are not read" in s.record()["centre_rule"]
    stream = ncw.NestChildWindowStream.build(settings=s, geometry=first, child_deck=deck,
                                             child_config=child_config(), parent_dt_s=0.75)
    assert stream.centre == pytest.approx((first.centre_x_m, first.centre_y_m))
    stream.open(tmp_path / "riding")
    clock = SimpleNamespace(completed_steps=1)
    xc, yc = centre_of(first)
    arrays = child_arrays(deck, 0.125, xc=xc, yc=yc)
    stream.after_parent_step(SimpleNamespace(geometry=first, criterion=None,
                                             parent=SimpleNamespace(clock=clock)))
    assert stream.capture(arrays, child_step=1, model_time_s=0.25)
    stream.after_parent_step(SimpleNamespace(geometry=moved, criterion=None,
                                             parent=SimpleNamespace(clock=clock)))
    assert stream.centre == pytest.approx((moved.centre_x_m, moved.centre_y_m))
    assert stream.capture(arrays, child_step=2, model_time_s=0.5)
    stream.close()
    records = ncw.read_window_stream(stream_root(tmp_path, "riding")).records
    # Same index box in the child lattice; the ground origin moved with the nest.
    assert (records[1].i0, records[1].j0) == (records[0].i0, records[0].j0)
    assert records[1].origin_x_m - records[0].origin_x_m == pytest.approx(moved.origin_x_m - first.origin_x_m)
    assert records[1].origin_y_m - records[0].origin_y_m == pytest.approx(moved.origin_y_m - first.origin_y_m)
    assert not records[0].clamped and not records[1].clamped
