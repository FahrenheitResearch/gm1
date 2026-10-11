"""The moving live nest, held to its CPU references and its WRF rules on the host.

No test here opens a CUDA context.  The packaged kernel text is compiled with
``g++`` (``tools/oracle/nest_move_oracle.py``); the launchers and
``MovingNest.execute`` run unchanged on NumPy arrays through a RawModule
stand-in that executes that text.  Device validation is a separate, gated step.
"""

from __future__ import annotations

import math
import shutil
from types import SimpleNamespace

import pytest

numpy = pytest.importorskip("numpy")

from cm1core import nest_move as nm  # noqa: E402
from cm1core.errors import CM1Refusal, refuse  # noqa: E402
from cm1core.nest_geometry import NestGeometry  # noqa: E402
from cm1core.nest_live_kernels import (  # noqa: E402
    LATTICE_OF_STAGGER,
    lattice_shape,
    reference_capture,
    reference_restrict,
)
from cm1core.nest_move_kernels import (  # noqa: E402
    CENTROID_WORDS,
    launch_shift,
    launch_vortex_centroid,
    launch_zeta_box,
    reference_shift,
    reference_vortex_centroid,
    reference_zeta_box,
)

requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")

RATIO = 3
STAGGERS = ("u", "v", "w", "mass")
MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


@pytest.fixture(scope="module")
def move_harness():
    from tools.oracle.nest_move_oracle import open_harness

    return open_harness()


@pytest.fixture(scope="module")
def live_harness():
    from tools.oracle.nest_live_oracle import open_harness

    return open_harness()


def _geometry(**overrides) -> NestGeometry:
    values = dict(
        parent_nx=18, parent_ny=16, parent_dx_m=600.0, parent_dy_m=600.0,
        refinement_ratio=RATIO, time_refinement_ratio=3, parent_i0=6, parent_j0=5,
        parent_span_x=6, parent_span_y=5, spec_zone=1, relax_zone=4, feedback=1,
    )
    values.update(overrides)
    return NestGeometry(**values)


def _moved(geometry: NestGeometry, move_i: int, move_j: int) -> NestGeometry:
    from dataclasses import replace

    return replace(geometry, parent_i0=geometry.parent_i0 + move_i,
                   parent_j0=geometry.parent_j0 + move_j)


def _capture(parent_volume, geometry: NestGeometry, stagger: str):
    nz = parent_volume.shape[0] - LATTICE_OF_STAGGER[stagger][2]
    cl = lattice_shape(stagger, nz, geometry.ny, geometry.nx)
    value, _ = reference_capture(
        parent_volume, child_lattice=cl, child_volume=None, side="full", width=5,
        ratio=geometry.refinement_ratio, parent_i0=geometry.parent_i0,
        parent_j0=geometry.parent_j0, stagger=stagger, parent_dt_s=1.0,
    )
    return value


def _shift_fill(child_volume, parent_volume, geometry: NestGeometry, move_i: int,
                move_j: int, stagger: str):
    """The move on one lattice volume: capture at the new placement, shift the rest."""

    new = _moved(geometry, move_i, move_j)
    target = _capture(parent_volume, new, stagger)
    return reference_shift(child_volume, target, origin=(0, 0, 0), lattice=target.shape,
                           shift_j=RATIO * move_j, shift_i=RATIO * move_i), new


# ---------------------------------------------------------------------------
# 1. Kernel text vs CPU reference, raw words.
# ---------------------------------------------------------------------------
@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
@pytest.mark.parametrize("move", MOVES + ((2, 0), (0, -2)))
def test_the_shift_kernel_text_is_the_reference(move_harness, stagger, move) -> None:
    rng = numpy.random.default_rng(STAGGERS.index(stagger) * 31 + MOVES.index(move) if move in MOVES else 99)
    lattice = lattice_shape(stagger, 4, 15, 18)
    origin = (1, 3, 3)
    snap = rng.uniform(-5, 5, size=(lattice[0] + 2, lattice[1] + 7, lattice[2] + 8)).astype(numpy.float32)
    out = rng.uniform(-5, 5, size=snap.shape).astype(numpy.float32)
    sj, si = RATIO * move[1], RATIO * move[0]
    expected = reference_shift(snap, out, origin=origin, lattice=lattice, shift_j=sj, shift_i=si)
    for order in ("linear", "shuffled"):
        got = move_harness.shift(snap=snap, out=out, origin=origin, lattice=lattice,
                                 shift_j=sj, shift_i=si, order=order)
        assert got.tobytes() == expected.tobytes(), (stagger, move, order)
    # Nothing outside the lattice rectangle is written.
    k0, j0, i0 = origin
    mask = numpy.ones(out.shape, dtype=bool)
    mask[k0:k0 + lattice[0], j0:j0 + lattice[1], i0:i0 + lattice[2]] = False
    assert expected[mask].tobytes() == out[mask].tobytes()


@requires_gxx
@pytest.mark.parametrize("h", (1, 2, 4))
def test_the_box_vorticity_kernel_text_is_the_reference(move_harness, h) -> None:
    rng = numpy.random.default_rng(h)
    zeta = rng.normal(0.0, 0.02, size=(31, 37)).astype(numpy.float32)
    bounds = dict(lo_j=6, hi_j=24, lo_i=5, hi_i=31)
    expected = reference_zeta_box(zeta, h=h, **bounds)
    for order in ("linear", "shuffled"):
        got = move_harness.zeta_box(zeta=zeta, h=h, order=order, **bounds)
        assert got.tobytes() == expected.tobytes(), order
    # dx*dy times the box sum is the circulation around the box outline.
    assert numpy.count_nonzero(expected) == (24 - 6 - 2 * h + 1) * (31 - 5 - 2 * h + 1)


@requires_gxx
def test_the_vortex_centroid_kernel_text_is_the_reference(move_harness) -> None:
    rng = numpy.random.default_rng(5)
    zbar = rng.normal(0.0, 1.0, size=(30, 34))
    cases = [
        ((4, 29, 4, 25), (4, 29, 4, 25), 3.0),
        ((10, 18, 6, 12), (4, 29, 4, 25), 2.5),
        ((27, 29, 23, 25), (4, 29, 4, 25), 5.75),
    ]
    for window, centres, d in cases:
        expected = numpy.array(reference_vortex_centroid(zbar, window=window, centres=centres, d=d))
        for order in ("linear", "shuffled"):
            got = move_harness.centroid(zbar=zbar, window=window, centres=centres, d=d, order=order)
            assert got.tobytes() == expected.tobytes(), (window, order)
    # A flat field: the first maximum in DO j / DO i order and no centroid
    # (every weight is zero), as mediation_nest_move.F:590 leaves it.
    flat = numpy.full((30, 34), 0.25)
    got = move_harness.centroid(zbar=flat, window=(4, 29, 4, 25), centres=(4, 29, 4, 25), d=2.0)
    assert list(got[:4]) == [1.0, 4.0, 4.0, 0.25]
    assert got[8] == 0.0 and got[9] == 0.0 and got[10] == 0.0
    assert got.tobytes() == numpy.array(reference_vortex_centroid(
        flat, window=(4, 29, 4, 25), centres=(4, 29, 4, 25), d=2.0)).tobytes()
    # No finite value: not found, which is WRF's fatal "method failure".
    nan = numpy.full((30, 34), numpy.nan)
    got = move_harness.centroid(zbar=nan, window=(4, 29, 4, 25), centres=(4, 29, 4, 25), d=2.0)
    assert got[0] == 0.0


@requires_gxx
def test_the_launchers_drive_the_kernel_text(move_harness) -> None:
    """The launchers' argument order is the kernels' signature."""

    from tools.oracle.nest_move_oracle import GxxModule

    module = GxxModule(move=move_harness)
    rng = numpy.random.default_rng(11)
    lattice = lattice_shape("u", 3, 15, 18)
    owner = rng.uniform(-1, 1, size=(5, 21, 25)).astype(numpy.float32)
    out = rng.uniform(-1, 1, size=owner.shape).astype(numpy.float32)
    expected = reference_shift(owner, out, origin=(1, 3, 3), lattice=lattice, shift_j=-3, shift_i=3)
    launch_shift(numpy, module, snap=owner, out=out, origin=(1, 3, 3), lattice=lattice,
                 shift_j=-3, shift_i=3)
    assert out.tobytes() == expected.tobytes()

    zeta = rng.normal(0, 1e-2, size=(22, 25)).astype(numpy.float32)
    zbar = numpy.zeros(zeta.shape, dtype=numpy.float64)
    launch_zeta_box(numpy, module, zeta=zeta, out=zbar, h=2, lo_j=4, hi_j=17, lo_i=4, hi_i=20)
    assert zbar.tobytes() == reference_zeta_box(zeta, h=2, lo_j=4, hi_j=17, lo_i=4, hi_i=20).tobytes()
    words = numpy.zeros(CENTROID_WORDS, dtype=numpy.float64)
    launch_vortex_centroid(numpy, module, zbar=zbar, out=words, window=(6, 18, 6, 15),
                           centres=(6, 18, 6, 15), d=2.0)
    assert words.tobytes() == numpy.array(reference_vortex_centroid(
        zbar, window=(6, 18, 6, 15), centres=(6, 18, 6, 15), d=2.0)).tobytes()
    with pytest.raises(ValueError, match="in-place"):
        launch_shift(numpy, module, snap=out, out=out, origin=(1, 3, 3), lattice=lattice,
                     shift_j=0, shift_i=3)


# ---------------------------------------------------------------------------
# 2. The shift + fill, on the CPU reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stagger", STAGGERS)
@pytest.mark.parametrize("move", MOVES)
def test_the_move_copies_the_child_and_captures_only_the_exposed_strip(stagger, move) -> None:
    rng = numpy.random.default_rng(3 + STAGGERS.index(stagger))
    g = _geometry()
    nz = 3
    parent = rng.uniform(250, 350, size=lattice_shape(stagger, nz, g.parent_ny, g.parent_nx)).astype(numpy.float32)
    child = rng.uniform(-9, 9, size=lattice_shape(stagger, nz, g.ny, g.nx)).astype(numpy.float32)
    moved, new = _shift_fill(child, parent, g, *move, stagger)
    captured = _capture(parent, new, stagger)
    nzl, nyl, nxl = child.shape
    si, sj = RATIO * move[0], RATIO * move[1]
    exposed = 0
    for j in range(nyl):
        for i in range(nxl):
            inside = 0 <= j + sj < nyl and 0 <= i + si < nxl
            if inside:
                assert moved[:, j, i].tobytes() == child[:, j + sj, i + si].tobytes()
            else:
                exposed += 1
                assert moved[:, j, i].tobytes() == captured[:, j, i].tobytes()
    # A move of one parent cell exposes r lattice columns (and/or rows).
    ex_i = abs(si) * nyl
    ex_j = abs(sj) * nxl
    assert exposed == ex_i + ex_j - abs(si) * abs(sj)


# ---------------------------------------------------------------------------
# 3. A moving nest over a translating field stays the interpolated parent.
# ---------------------------------------------------------------------------
def _translating_parent(g: NestGeometry, stagger: str, t: float, *, uniform: bool):
    nz = 2
    shape = lattice_shape(stagger, nz, g.parent_ny, g.parent_nx)
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    x = (numpy.arange(shape[2]) + (0.0 if x_face else 0.5)) * g.parent_dx_m
    y = (numpy.arange(shape[1]) + (0.0 if y_face else 0.5)) * g.parent_dy_m
    X, Y = numpy.meshgrid(x, y)
    base = {"u": 14.5, "v": 3.25, "w": 0.0, "mass": 301.5}[stagger]
    if uniform:
        plane = numpy.full(X.shape, base)
    else:
        cx, cy = 3000.0 + 15.0 * t, 4200.0 + 4.0 * t
        plane = base + 2.0 * numpy.exp(-(((X - cx) / 900.0) ** 2 + ((Y - cy) / 700.0) ** 2))
    return numpy.broadcast_to(plane, shape).astype(numpy.float32).copy()


@pytest.mark.parametrize("stagger", STAGGERS)
def test_a_moving_nest_over_uniform_translating_flow_is_the_interpolated_parent(stagger) -> None:
    """Uniform flow translating under the frame: every move leaves the child bit-equal.

    The child is not integrated here: records/CM1-NEST-SEAM.md section 5.5
    measured that an integrated child departs from the captured parent by its
    own spacing's closure (TKE floor and mixing length), at the size of the
    single-domain 600 m-to-200 m column difference, so equality to rounding is
    not a property of an integrated nest in this lane.  What the move must not
    add is any error of its own, and for that the case is exact.
    """

    g = _geometry(parent_nx=26, parent_ny=22, parent_i0=8, parent_j0=8)
    parent = _translating_parent(g, stagger, 0.0, uniform=True)
    child = _capture(parent, g, stagger)
    path = [(1, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (1, 0), (1, -1)]
    for step, move in enumerate(path, start=1):
        parent = _translating_parent(g, stagger, 60.0 * step, uniform=True)
        child, g = _shift_fill(child, parent, g, *move, stagger)
        assert g.placement_defect() is None
        assert child.tobytes() == _capture(parent, g, stagger).tobytes()
        assert numpy.unique(child).size == 1


@pytest.mark.parametrize("stagger", STAGGERS)
def test_a_moving_nest_over_a_translating_feature_is_the_interpolated_parent(stagger) -> None:
    """The move commutes with interpolation for any parent field, bit for bit.

    At each re-centring the child holds the parent interpolated at its current
    placement (the state a child that added no error of its own would hold);
    shifting it and capturing the exposed strip gives exactly the parent
    interpolated at the new placement, because a whole-parent-cell shift maps
    every child point onto the same donor and the same sub-cell offset.
    """

    g = _geometry(parent_nx=26, parent_ny=22, parent_i0=8, parent_j0=8)
    for step, move in enumerate([(1, 0), (0, 1), (1, 0), (1, -1), (-1, 1)], start=1):
        parent = _translating_parent(g, stagger, 45.0 * step, uniform=False)
        child = _capture(parent, g, stagger)
        moved, g = _shift_fill(child, parent, g, *move, stagger)
        assert moved.tobytes() == _capture(parent, g, stagger).tobytes(), (stagger, step)


# ---------------------------------------------------------------------------
# 4. Feedback across a move conserves what it claims to.
# ---------------------------------------------------------------------------
def _fed_back_mask(g: NestGeometry, parent_shape):
    probe = numpy.zeros((1, g.ny, g.nx), dtype=numpy.float32) + 1.0
    marked = reference_restrict(probe, numpy.zeros((1, *parent_shape[1:]), dtype=numpy.float32),
                                child_lattice=probe.shape, span_x=g.parent_span_x,
                                span_y=g.parent_span_y, ratio=RATIO, relax_zone=g.relax_zone,
                                parent_i0=g.parent_i0, parent_j0=g.parent_j0, stagger="mass")
    return marked[0] == 1.0


def test_feedback_across_a_move_keeps_the_parent_mass() -> None:
    rng = numpy.random.default_rng(21)
    # Four fed-back blocks per axis (child blocks starting at 6, 9, 12, 15),
    # so a diagonal move leaves, keeps and enters covered parent cells.
    g = _geometry(parent_nx=20, parent_ny=20, parent_i0=6, parent_j0=6,
                  parent_span_x=8, parent_span_y=8)
    nz = 2
    parent0 = rng.uniform(0.9, 1.1, size=(nz, g.parent_ny, g.parent_nx)).astype(numpy.float32)
    mask_a = _fed_back_mask(g, parent0.shape)
    # The child's own departure from the parent, only inside blocks fed back
    # from the unmoved footprint.
    child = _capture(parent0, g, "mass")
    for jp in range(g.parent_span_y):
        for ip in range(g.parent_span_x):
            if mask_a[g.parent_j0 + jp, g.parent_i0 + ip]:
                block = child[:, jp * RATIO:(jp + 1) * RATIO, ip * RATIO:(ip + 1) * RATIO]
                block += rng.uniform(-1e-3, 1e-3, size=block.shape).astype(numpy.float32)
    restrict = dict(span_x=g.parent_span_x, span_y=g.parent_span_y, ratio=RATIO,
                    relax_zone=g.relax_zone, stagger="mass")
    # med_nest_feedback, then med_nest_move (frame/module_integrate.F:443,448).
    p1 = reference_restrict(child, parent0, child_lattice=child.shape,
                            parent_i0=g.parent_i0, parent_j0=g.parent_j0, **restrict)
    moved, new = _shift_fill(child, p1, g, 1, -1, "mass")
    mask_b = _fed_back_mask(new, parent0.shape)
    p2 = reference_restrict(moved, p1, child_lattice=moved.shape,
                            parent_i0=new.parent_i0, parent_j0=new.parent_j0, **restrict)
    both, entering, leaving = mask_a & mask_b, mask_b & ~mask_a, mask_a & ~mask_b
    assert both.any() and entering.any() and leaving.any()
    # Blocks covered before and after carry the same child words: bit-identical.
    assert p2[:, both].tobytes() == p1[:, both].tobytes()
    # Cells the nest left keep the child's last fed-back block means.
    assert p2[:, leaving].tobytes() == p1[:, leaving].tobytes()
    assert not numpy.array_equal(p1[:, leaving], parent0[:, leaving])
    # Cells the nest entered get the block mean of the parent's own
    # interpolation: the float64 block-mean identity, narrowed once per child
    # value and once for the mean.
    eps = float(numpy.finfo(numpy.float32).eps)
    assert numpy.all(numpy.abs(p2[:, entering].astype(numpy.float64) - p1[:, entering])
                     <= 2.0 * eps * numpy.abs(p1[:, entering].astype(numpy.float64)))
    total1 = math.fsum(float(v) for v in p1.reshape(-1))
    total2 = math.fsum(float(v) for v in p2.reshape(-1))
    bound = 2.0 * eps * math.fsum(abs(float(v)) for v in p1[:, entering].reshape(-1))
    assert abs(total2 - total1) <= bound
    # Everywhere else the parent is untouched.
    untouched = ~(mask_a | mask_b)
    assert p2[:, untouched].tobytes() == parent0[:, untouched].tobytes()


# ---------------------------------------------------------------------------
# 5. The surface-vortex tracker.
# ---------------------------------------------------------------------------
def _winds(g: NestGeometry, vortices, *, nz: int = 1):
    """C-grid u, v of Lamb-Oseen vortices, in the padded layout.

    ``vortices`` are ``(x_m, y_m, core_radius_m, circulation_m2_s)``.  The
    vorticity ``circulation/(pi rc^2) exp(-r^2/rc^2)`` is positive and radially
    decreasing, so a vortex has the net circulation a real one has (a Gaussian
    streamfunction would not) and its box mean peaks on the vortex.
    """

    ng = 3

    def velocity(x, y):
        uu = numpy.zeros(numpy.broadcast(x, y).shape)
        vv = numpy.zeros_like(uu)
        for cx, cy, rc, gamma in vortices:
            ddx, ddy = x - cx, y - cy
            r2 = ddx * ddx + ddy * ddy
            with numpy.errstate(divide="ignore", invalid="ignore"):
                factor = numpy.where(
                    r2 > 0.0,
                    gamma / (2.0 * math.pi * r2) * -numpy.expm1(-r2 / (rc * rc)),
                    gamma / (2.0 * math.pi * rc * rc),
                )
            uu = uu - factor * ddy
            vv = vv + factor * ddx
        return uu, vv

    u = numpy.zeros((nz + 2, g.ny + 2 * ng, g.nx + 1 + 2 * ng), dtype=numpy.float32)
    v = numpy.zeros((nz + 2, g.ny + 1 + 2 * ng, g.nx + 2 * ng), dtype=numpy.float32)
    xu = g.origin_x_m + (numpy.arange(u.shape[2]) - ng) * g.dx_m
    yu = g.origin_y_m + (numpy.arange(u.shape[1]) - ng + 0.5) * g.dy_m
    u[1] = velocity(*numpy.meshgrid(xu, yu))[0].astype(numpy.float32)
    xv = g.origin_x_m + (numpy.arange(v.shape[2]) - ng + 0.5) * g.dx_m
    yv = g.origin_y_m + (numpy.arange(v.shape[1]) - ng) * g.dy_m
    v[1] = velocity(*numpy.meshgrid(xv, yv))[1].astype(numpy.float32)
    return u, v


def _tornado_geometry(**overrides) -> NestGeometry:
    values = dict(
        parent_nx=120, parent_ny=90, parent_dx_m=60.0, parent_dy_m=60.0,
        refinement_ratio=RATIO, time_refinement_ratio=3, parent_i0=30, parent_j0=25,
        parent_span_x=40, parent_span_y=40, spec_zone=1, relax_zone=4, feedback=0,
    )
    values.update(overrides)
    return NestGeometry(**values)


def _settings(**overrides) -> nm.FollowSettings:
    values = dict(tracker=nm.SURFACE_TRACKER, interval_steps=10, interval_s=30.0,
                  max_speed_m_s=10.0, smoothing_radius_m=200.0, corral_cells_x=3,
                  corral_cells_y=3)
    values.update(overrides)
    return nm.FollowSettings(**values)


def test_the_tracker_follows_the_circulation_not_the_sub_vortex_peak() -> None:
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, _settings())
    assert geo.h == 10
    main = (g.origin_x_m + 70 * g.dx_m, g.origin_y_m + 55 * g.dy_m, 150.0, 1.0e5)
    # A gust-front sub-vortex: more than twice the point vorticity, a thirtieth
    # of the circulation.
    sub = (g.origin_x_m + 30 * g.dx_m, g.origin_y_m + 30 * g.dy_m, 15.0, 3.0e3)
    u, v = _winds(g, [main, sub])
    zeta = nm.lowest_level_corner_vorticity(u, v, nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m, xp=numpy)
    peak_j, peak_i = numpy.unravel_index(int(numpy.argmax(zeta)), zeta.shape)
    assert (abs(peak_i - 30) <= 1) and (abs(peak_j - 30) <= 1)  # the instantaneous maximum
    state = nm.TrackerState()
    words, window = nm.surface_tracker_reference(u, v, geo, state)
    assert window == geo.centres
    assert nm.apply_centroid(state, words)
    assert abs(state.vc_i - 70.0) < 0.5 and abs(state.vc_j - 55.0) < 0.5


def test_the_search_window_bounds_the_tracker_by_the_declared_speed() -> None:
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, _settings(max_speed_m_s=10.0, interval_s=30.0))
    assert geo.search_cells == 15.0
    main = (g.origin_x_m + 60 * g.dx_m, g.origin_y_m + 60 * g.dy_m, 150.0, 1.0e5)
    decoy = (g.origin_x_m + 25 * g.dx_m, g.origin_y_m + 25 * g.dy_m, 150.0, 2.0e5)
    u, v = _winds(g, [main, decoy])
    fresh = nm.TrackerState()
    words, _ = nm.surface_tracker_reference(u, v, geo, fresh)
    nm.apply_centroid(fresh, words)
    assert abs(fresh.vc_i - 25.0) < 0.5  # the first search sees the whole child
    held = nm.TrackerState(vc_i=58.0, vc_j=61.0)
    words, window = nm.surface_tracker_reference(u, v, geo, held)
    assert window == (43, 73, 46, 76)  # int(vc -/+ 15), mediation_nest_move.F:482-485
    nm.apply_centroid(held, words)
    assert abs(held.vc_i - 60.0) < 0.5 and abs(held.vc_j - 60.0) < 0.5


def test_a_vortex_faster_than_the_declared_speed_stops_the_run_by_name() -> None:
    """The search window clips the maximum; the tracker names it instead of chasing the edge.

    Measured on this case: with 10 m/s declared against a 12 m/s vortex the
    unchecked tracker's centre sat 80 m (four child cells) from the vortex and
    the nest lagged 420 m; with 12 m/s or more declared the error was 3.5 m.
    """

    g = _tornado_geometry()
    settings = _settings(max_speed_m_s=10.0, interval_s=30.0)
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    x0, y0 = g.origin_x_m + 60 * g.dx_m, g.origin_y_m + 60 * g.dy_m
    state = nm.TrackerState()
    u, v = _winds(g, [(x0, y0, 150.0, 1.0e5)])
    words, window = nm.surface_tracker_reference(u, v, geo, state)
    assert nm.window_edge_defect(words, window, geo.centres) is None
    nm.apply_centroid(state, words)
    # One interval later the vortex is 360 m east: 18 child cells, beyond the 15 in reach.
    u, v = _winds(g, [(x0 + 360.0, y0, 150.0, 1.0e5)])
    words, window = nm.surface_tracker_reference(u, v, geo, state)
    assert int(words[1]) == window[1]
    mover = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse, surface=geo)
    driver = SimpleNamespace(geometry=g)
    with pytest.raises(CM1Refusal) as caught:
        mover._accept_surface(driver, 20, words, window)
    assert caught.value.detail.code == "CM1-CARTESIAN-NEST-FOLLOW-SPEED"
    assert "follow_max_speed_m_s" in caught.value.detail.reason
    assert mover.tracks == []


def test_the_move_decision_is_wrfs_one_parent_cell_rule() -> None:
    assert nm.wrf_move_cells(60.0, 120, 3) == 0
    assert nm.wrf_move_cells(62.9, 120, 3) == 0  # int(2.9/3) == 0
    assert nm.wrf_move_cells(63.0, 120, 3) == 1
    assert nm.wrf_move_cells(95.0, 120, 3) == 1  # clamped to one parent cell
    assert nm.wrf_move_cells(57.1, 120, 3) == 0
    assert nm.wrf_move_cells(57.0, 120, 3) == -1
    assert nm.wrf_move_cells(0.0, 120, 3) == -1
    # A ratio-9 nest moves: WRF's irad=6 clamp would make int(6/9) == 0 forever.
    assert nm.wrf_move_cells(40.0 + 9.0, 80, 9) == 1


def test_a_translating_vortex_is_followed_and_the_corral_stops_the_run_by_name() -> None:
    g = _tornado_geometry(parent_nx=90, parent_i0=30)
    # The declared speed bounds the search window and must be at least the
    # storm's (see the refusal test below for what happens when it is not).
    settings = _settings(max_speed_m_s=20.0)
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    speed_x, speed_y, dt = 12.0, 3.0, 3.0  # parent step 3 s, tracker every 10 steps
    x0, y0 = g.centre_x_m, g.centre_y_m
    state = nm.TrackerState()
    moves = []
    step = 0
    worst = 0.0
    while True:
        step += 1
        t = step * dt
        vx, vy = x0 + speed_x * t, y0 + speed_y * t
        if not state.ready or step % settings.interval_steps == 0:
            u, v = _winds(g, [(vx, vy, 150.0, 1.0e5)])
            words, window = nm.surface_tracker_reference(u, v, geo, state)
            assert nm.window_edge_defect(words, window, geo.centres) is None
            assert nm.apply_centroid(state, words)
            # Measured 3.54 m worst over this run (child dx 20 m).
            error = max(abs(g.origin_x_m + state.vc_i * g.dx_m - vx),
                        abs(g.origin_y_m + state.vc_j * g.dy_m - vy))
            worst = max(worst, error)
            assert error <= g.dx_m
        mi = nm.wrf_move_cells(state.vc_i, g.nx, RATIO)
        mj = nm.wrf_move_cells(state.vc_j, g.ny, RATIO)
        if mi == 0 and mj == 0:
            continue
        defect = nm.corral_defect(g, move_i=mi, move_j=mj, corral_x=3, corral_y=3)
        if defect is not None:
            break
        state.vc_i -= mi * RATIO
        state.vc_j -= mj * RATIO
        g = _moved(g, mi, mj)
        moves.append((step, mi, mj))
        # The target is refreshed once per interval, so the nest centre lags the
        # vortex by at most one interval's travel (measured 300 m of 360 m).
        assert abs(g.centre_x_m - vx) <= speed_x * settings.interval_s
        assert abs(g.centre_y_m - vy) <= speed_x * settings.interval_s
        assert g.placement_defect() is None
    assert len(moves) >= 15
    assert worst <= g.dx_m
    assert all(abs(mi) <= 1 and abs(mj) <= 1 for _, mi, mj in moves)
    assert g.parent_nx - (g.parent_i0 + g.parent_span_x) == 3
    assert "corral" in defect


# ---------------------------------------------------------------------------
# 6. The carrier inventory is closed over what a real child allocates.
# ---------------------------------------------------------------------------
def _following_config(**nest):
    import tomllib

    from test_nest_config_to_call_site import BASE_TOML, _nest_text

    from cm1core.config import parse_mapping

    text = _nest_text(**nest)
    return parse_mapping(tomllib.loads(BASE_TOML + text))


def _child_registry_shapes(child_config) -> dict[str, tuple[int, ...]]:
    from cm1core.arwen_wsm6_component import WSM6_COMPONENT_PERSISTENCE, WSM6_SURFACE_SCRATCH
    from cm1core.cuda_cartesian_dynamics import cartesian_dynamics_state_shapes

    g = child_config.grid
    prefixes = {"physical_fields": "field.", "immutable_metrics": "metric.", "held": "held.",
                "tke": "tke.", "persistent_padded": "padded.", "compact_base_profiles": "base."}
    shapes = {}
    for group, table in cartesian_dynamics_state_shapes(child_config).items():
        for key, shape in table.items():
            shapes[prefixes[group] + key] = tuple(shape)
    for name in WSM6_COMPONENT_PERSISTENCE:
        shapes[f"physics.{name}"] = g.surface_shape if name in WSM6_SURFACE_SCRATCH else g.mass_shape
    shapes["physics.asq"] = (6,)
    # cuda_immutable_metrics registers stretched zh/zf as compact profiles under
    # their own prefix; a real child carries them, so the model must too.
    shapes["metric-profile.zh"] = (g.nz,)
    shapes["metric-profile.zf"] = (g.nz + 1,)
    for name in ("znt", "xland", "mavail", "tsk"):
        shapes[f"closure.surface.{name}"] = g.surface_shape
    for name in ("sws", "svs", "sps", "srs", "sgs", "sus", "shs"):
        shapes[f"output.{name}"] = g.surface_shape
    for name in ("thv", "ppterm", "piadv", "uten", "vten", "wten", "ppd", "tri_aa"):
        shapes[f"padded.{name}"] = (g.nz + 2, g.ny + 2, g.nx + 2)
    return shapes


def test_every_carried_child_buffer_has_a_move_rule() -> None:
    from cm1core.cartesian_door import require_nest
    from cm1core.cartesian_restart import cartesian_restart_profile

    config = _following_config()
    binding = require_nest(config)
    child = binding.child
    shapes = _child_registry_shapes(child)
    shifted, kept, unknown = nm.classify_carriers(shapes, grid=child.grid)
    assert unknown == []
    shifted_names = {rule.name for rule in shifted}
    group_prefix = {"padded": "padded.", "fields": "field.", "held": "held.", "tke": "tke.",
                     "component": "physics."}
    for carrier in cartesian_restart_profile(child).carriers:
        name = group_prefix[carrier.group] + carrier.key
        # Everything a restart must carry is shifted, except the child's own
        # edge-indexed radiative histories and the getset products.
        assert name in shifted_names or name in nm.EDGE_HISTORY or name in nm.REBUILT, name
    for name in ("padded.u3d", "padded.ua", "field.q3d", "tke.tke3d", "physics.mp_rainnc",
                 "closure.surface.znt", "output.sws", "field.th3d", "padded.pp3d"):
        assert name in shifted_names
    _, _, unknown = nm.classify_carriers({**shapes, "closure.mystery": (4, 4)}, grid=child.grid)
    assert unknown == ["closure.mystery"]


# ---------------------------------------------------------------------------
# 7. MovingNest.execute on the host, through the kernel text.
# ---------------------------------------------------------------------------
class _Stream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synchronize(self):
        pass


def _small_driver(rng, g: NestGeometry):
    from cm1core.grid_geometry import CartesianGridGeometry

    pg = CartesianGridGeometry(g.parent_nx, g.parent_ny, 3, g.parent_dx_m, g.parent_dy_m, 500.0)
    cg = CartesianGridGeometry(g.nx, g.ny, 3, g.dx_m, g.dy_m, 500.0)

    def registry(grid):
        from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes

        p = terrain_open_sound_shapes(grid.nx, grid.ny, grid.nz)
        shapes = {f"padded.{n}": p[n] for n in ("ua", "u3d", "va", "v3d", "wa", "w3d", "rru", "rrv", "rrw",
                                                  "ppi", "pp3d", "ppx", "ppten", "radbcw", "radbcs")}
        shapes.update({f"field.{n}": grid.mass_shape for n in ("ppi", "pp3d", "tha", "th3d", "prs", "pi0")})
        shapes.update({"field.q3d": (6, *grid.mass_shape), "field.qa": (6, *grid.mass_shape),
                       "field.rf": grid.w_shape, "held.uten": grid.u_shape, "held.qten": (6, *grid.mass_shape),
                       "tke.tke3d": grid.w_shape, "physics.mp_rainnc": grid.surface_shape,
                       "physics.effc": grid.mass_shape, "closure.surface.znt": grid.surface_shape,
                       "output.sws": grid.surface_shape, "metric.rho0_3d": grid.mass_shape,
                       "base.u0": (grid.nz,)})
        buffers = {}
        for name, shape in shapes.items():
            if name.startswith(("metric.", "base.")):
                buffers[name] = numpy.full(shape, 1.25, dtype=numpy.float32)
            else:
                buffers[name] = rng.uniform(-2, 2, size=shape).astype(numpy.float32)
        return buffers

    child_buffers, parent_buffers = registry(cg), registry(pg)
    refills = []
    driver = SimpleNamespace(
        geometry=g,
        forcing=SimpleNamespace(width=g.spec_zone + g.relax_zone, geometry=g),
        child=SimpleNamespace(registry=SimpleNamespace(buffers=child_buffers),
                              config=SimpleNamespace(grid=cg),
                              dynamics=SimpleNamespace(numq=6, arrays=child_buffers)),
        parent=SimpleNamespace(registry=SimpleNamespace(buffers=parent_buffers),
                               config=SimpleNamespace(grid=pg, dt_s=0.5),
                               clock=SimpleNamespace(completed_steps=7, model_seconds=3.5)),
    )
    return driver, refills


@requires_gxx
def test_moving_nest_execute_moves_every_carrier_through_the_kernel_text(
    move_harness, live_harness, monkeypatch
) -> None:
    from tools.oracle.nest_move_oracle import GxxModule

    import cm1core.nest_live as nest_live

    rng = numpy.random.default_rng(8)
    g = _geometry()
    driver, refills = _small_driver(rng, g)
    driver.module = GxxModule(live=live_harness)
    monkeypatch.setattr(nest_live, "_refill_halos_and_thermodynamics",
                        lambda dynamics, **kw: refills.append(dynamics))
    before_child = {k: v.copy() for k, v in driver.child.registry.buffers.items()}
    parent = driver.parent.registry.buffers
    mover = nm.MovingNest(settings=_settings(tracker="cm1-pressure-centre", smoothing_radius_m=None),
                          cupy=numpy, stream=_Stream(), refuse=refuse,
                          module=GxxModule(move=move_harness))
    new = mover.execute(driver, move_i=1, move_j=-1)
    assert (new.parent_i0, new.parent_j0) == (g.parent_i0 + 1, g.parent_j0 - 1)
    assert driver.geometry is new and driver.forcing.geometry is new
    assert len(refills) == 1
    cg, pg = driver.child.config.grid, driver.parent.config.grid
    rules = {rule.name: rule for rule in nm.classify_carriers(
        {k: v.shape for k, v in before_child.items()}, grid=cg)[0]}
    assert set(rules) == set(mover.shifted_carriers)
    for name, value in driver.child.registry.buffers.items():
        if name not in rules:
            assert value.tobytes() == before_child[name].tobytes(), name
            continue
        rule = rules[name]
        cl = nm.rule_lattice(rule, nz=cg.nz, ny=cg.ny, nx=cg.nx)
        pl = nm.rule_lattice(rule, nz=pg.nz, ny=pg.ny, nx=pg.nx)
        for c_old, c_new, p_owner in zip(nm.rule_owners(rule, before_child[name], numq=6),
                                         nm.rule_owners(rule, value, numq=6),
                                         nm.rule_owners(rule, parent[name], numq=6)):
            k0, j0, i0 = rule.origin
            pvol = p_owner[k0:k0 + pl[0], j0:j0 + pl[1], i0:i0 + pl[2]]
            target, _ = reference_capture(
                pvol, child_lattice=cl, child_volume=None, side="full", width=5, ratio=RATIO,
                parent_i0=new.parent_i0, parent_j0=new.parent_j0, stagger=rule.stagger,
                parent_dt_s=0.5)
            expected = c_old.copy()
            expected[k0:k0 + cl[0], j0:j0 + cl[1], i0:i0 + cl[2]] = target
            expected = reference_shift(c_old, expected, origin=rule.origin, lattice=cl,
                                       shift_j=-RATIO, shift_i=RATIO)
            assert c_new.tobytes() == expected.tobytes(), name


@requires_gxx
def test_a_move_on_a_refined_vertical_deck_fills_the_strip_through_the_vertical_window(
    move_harness, live_harness, monkeypatch
) -> None:
    """A child on its own deck: the exposed strip goes vertical-then-horizontal.

    The real ``NestVerticalCoupling.relocate`` and ``source`` run, with
    ``gm1_nest_vinterp`` and ``gm1_nest_capture`` executed from their packaged
    text, and the moved child is held to
    ``nest_vertical.reference_capture_refined`` at the new placement.
    """

    from tools.oracle.nest_move_oracle import GxxModule
    from tools.oracle.nest_vertical_oracle import open_harness as open_vertical

    import cm1core.nest_live as nest_live
    from cm1core.grid_geometry import CartesianGridGeometry
    from cm1core.nest_vertical import (
        VerticalWindow,
        plan_vertical_refinement,
        reference_capture_refined,
        upload_vertical_map,
    )
    from test_nest_vertical_refinement import small_child, small_parent

    parent_levels, child_levels = small_parent(), small_child()
    plan = plan_vertical_refinement(parent_levels, child_levels)
    g = _geometry()
    pg = CartesianGridGeometry(g.parent_nx, g.parent_ny, parent_levels.nk, g.parent_dx_m, g.parent_dy_m, 100.0)
    cg = CartesianGridGeometry(g.nx, g.ny, child_levels.nk, g.dx_m, g.dy_m, 100.0)
    rng = numpy.random.default_rng(77)

    def buffers(grid):
        from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes

        p = terrain_open_sound_shapes(grid.nx, grid.ny, grid.nz)
        shapes = {"padded.u3d": p["u3d"], "padded.w3d": p["w3d"], "padded.pp3d": p["pp3d"],
                  "field.th3d": grid.mass_shape, "field.q3d": (6, *grid.mass_shape),
                  "tke.tke3d": grid.w_shape, "closure.surface.znt": grid.surface_shape}
        return {n: rng.uniform(-2, 2, size=shape).astype(numpy.float32) for n, shape in shapes.items()}

    child_buffers, parent_buffers = buffers(cg), buffers(pg)
    coupling = object.__new__(nest_live.NestVerticalCoupling)
    coupling.cp, coupling.stream, coupling.refinement = numpy, _Stream(), plan
    coupling.module = GxxModule(vertical=open_vertical())
    coupling.maps = {"mass": upload_vertical_map(numpy, plan.mass), "w": upload_vertical_map(numpy, plan.face)}
    coupling.windows = {s: VerticalWindow.for_stagger(g, s) for s in STAGGERS}
    coupling.scratch = {
        s: numpy.empty((child_levels.nk + (1 if s == "w" else 0), coupling.windows[s].nj,
                        coupling.windows[s].ni), dtype=numpy.float32)
        for s in STAGGERS
    }
    refills = []
    monkeypatch.setattr(nest_live, "_refill_halos_and_thermodynamics",
                        lambda dynamics, **kw: refills.append(dynamics))
    driver = SimpleNamespace(
        geometry=g, module=GxxModule(live=live_harness),
        forcing=SimpleNamespace(width=g.spec_zone + g.relax_zone, geometry=g, vertical=coupling),
        child=SimpleNamespace(registry=SimpleNamespace(buffers=child_buffers),
                              config=SimpleNamespace(grid=cg),
                              dynamics=SimpleNamespace(numq=6, arrays=child_buffers)),
        parent=SimpleNamespace(registry=SimpleNamespace(buffers=parent_buffers),
                               config=SimpleNamespace(grid=pg, dt_s=0.5),
                               clock=SimpleNamespace(completed_steps=4, model_seconds=2.0)),
    )
    before = {k: v.copy() for k, v in child_buffers.items()}
    mover = nm.MovingNest(settings=_settings(tracker="cm1-pressure-centre", smoothing_radius_m=None),
                          cupy=numpy, stream=_Stream(), refuse=refuse,
                          module=GxxModule(move=move_harness))
    new = mover.execute(driver, move_i=-1, move_j=1)
    assert mover.vertical_capture and mover.record()["exposed_strip_vertical_window"]
    for s in STAGGERS:
        assert coupling.windows[s] == VerticalWindow.for_stagger(new, s)
    rules = {r.name: r for r in nm.classify_carriers({k: v.shape for k, v in before.items()}, grid=cg)[0]}
    assert set(rules) == set(child_buffers)
    for name, rule in rules.items():
        cl = nm.rule_lattice(rule, nz=cg.nz, ny=cg.ny, nx=cg.nx)
        pl = nm.rule_lattice(rule, nz=pg.nz, ny=pg.ny, nx=pg.nx)
        for c_old, c_new, p_owner in zip(nm.rule_owners(rule, before[name], numq=6),
                                         nm.rule_owners(rule, child_buffers[name], numq=6),
                                         nm.rule_owners(rule, parent_buffers[name], numq=6)):
            k0, j0, i0 = rule.origin
            pvol = p_owner[k0:k0 + pl[0], j0:j0 + pl[1], i0:i0 + pl[2]]
            if rule.plane:
                target, _ = reference_capture(
                    pvol, child_lattice=cl, child_volume=None, side="full", width=5, ratio=RATIO,
                    parent_i0=new.parent_i0, parent_j0=new.parent_j0, stagger=rule.stagger,
                    parent_dt_s=0.5)
            else:
                target, _ = reference_capture_refined(
                    pvol, vmap=plan.map_for(rule.stagger), window=VerticalWindow.for_stagger(new, rule.stagger),
                    child_lattice=cl, child_volume=None, side="full", width=5, ratio=RATIO,
                    stagger=rule.stagger, parent_dt_s=0.5)
            expected = c_old.copy()
            expected[k0:k0 + cl[0], j0:j0 + cl[1], i0:i0 + cl[2]] = target
            expected = reference_shift(c_old, expected, origin=rule.origin, lattice=cl,
                                       shift_j=RATIO, shift_i=-RATIO)
            assert c_new.tobytes() == expected.tobytes(), name
    assert coupling.module.calls["gm1_nest_vinterp"] > 0
    assert len(refills) == 1


def test_a_move_into_the_corral_stops_the_run_and_nothing_moves() -> None:
    g = _geometry(parent_i0=9, parent_span_x=6)  # 3 cells clear on the east
    driver = SimpleNamespace(
        geometry=g,
        parent=SimpleNamespace(clock=SimpleNamespace(completed_steps=5, model_seconds=2.5)),
    )
    mover = nm.MovingNest(settings=_settings(interval_steps=1000), cupy=numpy, stream=_Stream(),
                          refuse=refuse)
    mover.state.vc_i, mover.state.vc_j = g.nx / 2 + RATIO, g.ny / 2.0
    with pytest.raises(CM1Refusal) as caught:
        mover.after_parent_step(driver)
    assert caught.value.detail.code == "CM1-CARTESIAN-NEST-FOLLOW-CORRAL"
    assert "storm-following" in caught.value.detail.reason
    assert driver.geometry is g and mover.moves == []


def test_the_move_receipt_names_step_cells_and_tracker() -> None:
    g = _geometry()

    class _Recorded(nm.MovingNest):
        def execute(self, driver, *, move_i, move_j):
            return _moved(g, move_i, move_j)

    driver = SimpleNamespace(
        geometry=g,
        parent=SimpleNamespace(clock=SimpleNamespace(completed_steps=40, model_seconds=20.0)),
    )
    mover = _Recorded(settings=_settings(interval_steps=1000), cupy=numpy, stream=_Stream(),
                      refuse=refuse)
    mover.state.vc_i, mover.state.vc_j = g.nx / 2 - RATIO, g.ny / 2 + 2 * RATIO
    mover.tracks.append({"parent_step": 30, "value": 0.042})
    row = mover.after_parent_step(driver)
    assert row["parent_step"] == 40
    assert row["from_parent_cell"] == [g.parent_i0, g.parent_j0]
    assert row["to_parent_cell"] == [g.parent_i0 - 1, g.parent_j0 + 1]
    assert row["shift_parent_cells"] == [-1, 1]
    assert row["tracker_value"] == 0.042 and row["tracked_at_parent_step"] == 30
    assert row["tracker_centre_child_faces"] == [g.nx / 2 - RATIO, g.ny / 2 + 2 * RATIO]
    # vc is carried with the move (mediation_nest_move.F:703-704).
    assert mover.state.vc_i == g.nx / 2 and mover.state.vc_j == g.ny / 2 + RATIO
    assert mover.record()["moves"] == [row]
    assert mover.record()["move_count"] == 1 and "carried_from_restart" not in mover.record()


def test_a_resumed_mover_reports_the_carried_track_and_the_whole_runs_counts() -> None:
    """A move decided from a centre carried across a restart names the track it came
    from, and the counts continue the interrupted run's (cm1core.nest_restart)."""

    g = _geometry()

    class _Recorded(nm.MovingNest):
        def execute(self, driver, *, move_i, move_j):
            return _moved(g, move_i, move_j)

    driver = SimpleNamespace(
        geometry=g,
        parent=SimpleNamespace(clock=SimpleNamespace(completed_steps=41, model_seconds=20.5)),
    )
    mover = _Recorded(settings=_settings(interval_steps=1000), cupy=numpy, stream=_Stream(),
                      refuse=refuse, moves_before_restart=6, tracker_events_before_restart=4,
                      carried_last_track={"parent_step": 30, "value": 0.042})
    mover.state.vc_i, mover.state.vc_j = g.nx / 2 - RATIO, g.ny / 2.0
    row = mover.after_parent_step(driver)
    assert row["tracker_value"] == 0.042 and row["tracked_at_parent_step"] == 30
    assert (mover.move_count, mover.tracker_event_count) == (7, 4)
    record = mover.record()
    assert record["move_count"] == 7 and record["moves"] == [row]
    assert record["carried_from_restart"] == {
        "moves": 6, "tracker_events": 4, "last_track": {"parent_step": 30, "value": 0.042},
    }
    # This run's own track replaces the carried one.
    mover.tracks.append({"parent_step": 42, "value": 0.05})
    assert mover.last_track() == {"parent_step": 42, "value": 0.05}
    assert mover.tracker_event_count == 5


# ---------------------------------------------------------------------------
# 8. Configuration.
# ---------------------------------------------------------------------------
def test_the_surface_tracker_is_admitted_with_its_smoothing_radius() -> None:
    from cm1core.cartesian_door import require_nest

    import tomllib

    from test_nest_config_to_call_site import BASE_TOML, NEST_TABLE

    from cm1core.config import parse_mapping

    text = NEST_TABLE + 'follow_tracker = "wrf-surface-circulation"\nfollow_smoothing_radius_m = 1000.0\nfollow_corral_m = 5000.0\n'
    config = parse_mapping(tomllib.loads(BASE_TOML + text))
    assert config.nest.follow_smoothing_radius_m == 1000.0
    projected = config.canonical_dict()["nest"]
    assert projected["follow_smoothing_radius_m"] == 1000.0 and projected["follow_corral_m"] == 5000.0
    binding = require_nest(config)
    settings = binding.follow_settings()
    assert settings.tracker == nm.SURFACE_TRACKER
    assert settings.corral_cells_x == 5 and settings.corral_cells_y == 5
    assert binding.record(parent_dt_s=0.5)["storm_following"]["follow_smoothing_radius_m"] == 1000.0


def _refusal(text: str):
    import tomllib

    from test_nest_config_to_call_site import BASE_TOML, NEST_TABLE

    from cm1core.config import parse_mapping

    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(BASE_TOML + NEST_TABLE + text))
    return caught.value.detail


@pytest.mark.parametrize(
    "text,code,fragment",
    [
        ('follow_tracker = "wrf-surface-circulation"\n', "CM1-CFG-NEST-FOLLOW-SMOOTHING", "no scientific default"),
        ("follow_smoothing_radius_m = 500.0\n", "CM1-CFG-NEST-FOLLOW-SMOOTHING", "never evaluated"),
        ('follow_tracker = "wrf-surface-circulation"\nfollow_smoothing_radius_m = 100.0\n',
         "CM1-CFG-NEST-FOLLOW-SMOOTHING", "single corner"),
        ('follow_tracker = "wrf-surface-circulation"\nfollow_smoothing_radius_m = 9000.0\n',
         "CM1-CFG-NEST-FOLLOW-SMOOTHING", "no admissible centre"),
        ("follow_corral_m = 2000.0\n", "CM1-CFG-NEST-FOLLOW-CORRAL", "halo words"),
        ("follow_corral_m = 16000.0\n", "CM1-CFG-NEST-FOLLOW-CORRAL", "inside its own"),
    ],
)
def test_the_following_keys_are_refused_by_key_and_value(text, code, fragment) -> None:
    detail = _refusal(text)
    assert detail.code == code
    assert fragment in detail.reason


# ---------------------------------------------------------------------------
# 9. nest.follow_on_edge = "hold" (cm1core.nest_move, THE HOLD POLICY).
#
# WRF bounds its search window by max_vortex_speed and does not check what it
# found (share/mediation_nest_move.F:471-487), stops only when nothing is
# found (:544-548), and cancels a move into the corral per axis and keeps
# integrating (:849-859).  "hold" keeps the previous centre through an edge,
# no-circulation or too-fast sample for follow_hold_limit samples, then
# relocates within follow_relocation_radius_m; "stop" (or no key) is the code
# before the key.
# ---------------------------------------------------------------------------
LOST_WORDS = [0.0, -1.0, -1.0, 0.0] + [0.0] * (CENTROID_WORDS - 4)
SURFACE_KEYS = 'follow_tracker = "wrf-surface-circulation"\nfollow_smoothing_radius_m = 1000.0\n'
#: The base deck's reach is 55 m/s * 60 s = 3300 m on a 200 m child.
#: A relocation spans follow_hold_limit + 1 intervals: (4 + 1) * 3300 m + 200 m = 16700 m.
HOLD_KEYS = 'follow_on_edge = "hold"\nfollow_hold_limit = 4\nfollow_relocation_radius_m = 16700.0\n'

#: Measured by running the ``_stop_path`` scenario below on b200ad4, before
#: nest.follow_on_edge existed: sha256 of the sorted-key JSON of the refusal
#: details, tracker rows and move rows, and of ``MovingNest.record()``.
#: The pin covers the edge and lost refusals' codes and gates as GM1 spells them now; every other
#: word of the behaviour is the one the earlier pin sealed.
STOP_BEHAVIOUR_SHA256 = "c08c8e3246c7fc82d864226cc305ce29640d2ce7d94e1a12dd18ba4f70b5404d"
STOP_RECORD_SHA256 = "6c822a6f616b1d2d99b65b8e3eb78fba2a25d7ecbd05b1f839260222670dcac9"

#: The keys ``hold``/declared ``stop`` add to ``MovingNest.record()``.
HOLD_RECORD_KEYS = {
    "on_edge", "hold_limit", "relocation_radius_m", "corral_hold_limit_parent_steps", "holds",
    "holds_not_listed", "hold_counts", "max_consecutive_holds", "max_consecutive_corral_parent_steps",
}
#: ``hold_counts`` of a run that held nothing: every listed event kind, in receipt order.
NO_HOLDS = dict.fromkeys(("edge-hold", "lost-hold", "too-fast-hold", "corral-hold", "relocation"), 0)


def _hold_settings(**overrides) -> nm.FollowSettings:
    """A 15-cell (300 m) reach on the 20 m child; the relocating sample of a
    limit of 2 spans 45 cells, so the radius is the admissible 46 (920 m)."""

    values = dict(max_speed_m_s=10.0, interval_s=30.0, on_edge="hold", hold_limit=2,
                  relocation_radius_m=920.0)
    values.update(overrides)
    return _settings(**values)


def _plane_search(zbar, geo: nm.SurfaceTrackerGeometry):
    """The relocation search ``_track_surface`` launches, on a host box-mean plane."""

    def search(window):
        return reference_vortex_centroid(zbar, window=window, centres=geo.centres, d=geo.d)

    return search


def _search_argument(mover: nm.MovingNest, search) -> dict:
    """The relocation search, passed only when an edge policy is declared.

    Without the key the host harness calls ``_accept_surface`` exactly as the
    code before the key did, so the ``stop`` measurements below run unchanged
    on a tree that predates ``nest.follow_on_edge``.
    """

    return {} if getattr(mover.settings, "on_edge", None) is None else {"search": search}


def _search_window(mover: nm.MovingNest):
    """The window ``_track_surface`` launches for ``mover``'s next sample.

    Under an edge policy it spans the follow intervals since the last accepted
    centre (``MovingNest.search_intervals``); without the key it is the window
    the code before the key launched.
    """

    if getattr(mover.settings, "on_edge", None) is None:
        return mover.surface.window(mover.state)
    return mover.surface.window(mover.state, intervals=mover.search_intervals)


def _reference_plane(u, v, geo: nm.SurfaceTrackerGeometry):
    zeta = nm.lowest_level_corner_vorticity(u, v, nx=geo.nx, ny=geo.ny, dx_m=geo.dx_m,
                                            dy_m=geo.dy_m, xp=numpy)
    lo_j, hi_j, lo_i, hi_i = geo.box
    return reference_zeta_box(zeta, h=geo.h, lo_j=lo_j, hi_j=hi_j, lo_i=lo_i, hi_i=hi_i)


class _HostTracked(nm.MovingNest):
    """The surface tracker's CPU reference in place of the device launch.

    ``track`` computes the box-mean plane and the centroid words with the
    references section 1 holds the kernel text to, and hands them, with a
    relocation search over the same plane, to the same ``_accept_surface`` the
    device path calls; ``execute`` moves only the geometry.
    """

    def track(self, driver, step):
        u, v = _winds(driver.geometry, driver.vortices(float(driver.parent.clock.model_seconds)))
        search = _plane_search(_reference_plane(u, v, self.surface), self.surface)
        window = _search_window(self)
        self._accept_surface(driver, step, search(window), window, **_search_argument(self, search))

    def execute(self, driver, *, move_i, move_j):
        driver.geometry = _moved(driver.geometry, move_i, move_j)
        return driver.geometry


class _Untracked(nm.MovingNest):
    """The move decision alone: the centre is set by the test."""

    def track(self, driver, step):
        pass

    def execute(self, driver, *, move_i, move_j):
        driver.geometry = _moved(driver.geometry, move_i, move_j)
        return driver.geometry


def _clocked(g: NestGeometry, step: int, *, dt: float = 3.0, vortices=None):
    clock = SimpleNamespace(completed_steps=step, model_seconds=step * dt)
    return SimpleNamespace(geometry=g, parent=SimpleNamespace(clock=clock), vortices=vortices)


def _translating_run(settings: nm.FollowSettings, *, steps: int = 400, dt: float = 3.0,
                     speed: tuple[float, float] = (12.0, 3.0)):
    """Section 5's translating vortex, run toward the parent's east boundary."""

    g = _tornado_geometry(parent_nx=90, parent_i0=30)
    x0, y0 = g.centre_x_m, g.centre_y_m
    driver = _clocked(g, 0, dt=dt,
                      vortices=lambda t: [(x0 + speed[0] * t, y0 + speed[1] * t, 150.0, 1.0e5)])
    mover = _HostTracked(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                         surface=nm.SurfaceTrackerGeometry.build(g, settings))
    for step in range(1, steps + 1):
        driver.parent.clock.completed_steps = step
        driver.parent.clock.model_seconds = step * dt
        try:
            mover.after_parent_step(driver)
        except CM1Refusal as caught:
            return mover, driver, caught.detail
    return mover, driver, None


def _edge_case(settings: nm.FollowSettings, *, second_x_m: float = 360.0):
    """A 12 m/s vortex under a 10 m/s window: the second sample is edge-clipped."""

    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    x0, y0 = g.origin_x_m + 60 * g.dx_m, g.origin_y_m + 60 * g.dy_m
    state = nm.TrackerState()
    u, v = _winds(g, [(x0, y0, 150.0, 1.0e5)])
    words, window = nm.surface_tracker_reference(u, v, geo, state)
    assert nm.apply_centroid(state, words)
    u, v = _winds(g, [(x0 + second_x_m, y0, 150.0, 1.0e5)])
    zbar = _reference_plane(u, v, geo)
    window = geo.window(state)
    words = reference_vortex_centroid(zbar, window=window, centres=geo.centres, d=geo.d)
    return g, geo, state, words, window, _plane_search(zbar, geo)


def _stop_path(settings: nm.FollowSettings):
    import hashlib
    import json
    from dataclasses import replace

    mover, _, corral = _translating_run(settings)
    slow = replace(settings, max_speed_m_s=10.0)
    g, geo, state, words, window, _ = _edge_case(slow)
    edge_mover = nm.MovingNest(settings=slow, cupy=numpy, stream=_Stream(), refuse=refuse,
                               surface=geo, state=state)
    with pytest.raises(CM1Refusal) as edge:
        edge_mover._accept_surface(_clocked(g, 20), 20, words, window)
    lost_mover = nm.MovingNest(settings=slow, cupy=numpy, stream=_Stream(), refuse=refuse,
                               surface=geo, state=nm.TrackerState(vc_i=60.0, vc_j=60.0))
    with pytest.raises(CM1Refusal) as lost:
        lost_mover._accept_surface(_clocked(g, 20), 20, list(LOST_WORDS), geo.window(lost_mover.state))
    behaviour = {
        "corral": corral.as_dict(), "edge": edge.value.detail.as_dict(),
        "lost": lost.value.detail.as_dict(), "tracks": mover.tracks, "moves": mover.moves,
        "edge_tracks": edge_mover.tracks, "lost_tracks": lost_mover.tracks,
        "lost_centre": [lost_mover.state.vc_i, lost_mover.state.vc_j],
    }
    digest = hashlib.sha256(json.dumps(behaviour, sort_keys=True).encode()).hexdigest()
    return digest, mover.record()


def test_stop_absent_or_declared_is_byte_identical_to_the_code_before_the_key() -> None:
    import hashlib
    import json

    behaviour, record = _stop_path(_settings(max_speed_m_s=20.0))
    assert behaviour == STOP_BEHAVIOUR_SHA256
    assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == STOP_RECORD_SHA256
    assert not HOLD_RECORD_KEYS & set(record)
    assert set(record["settings"]) == {"tracker", "interval_parent_steps", "interval_s", "max_speed_m_s",
                                       "smoothing_radius_m", "corral_cells_x", "corral_cells_y"}
    assert "relocation_radius_child_cells" not in record["surface_tracker"]

    declared_behaviour, declared = _stop_path(_settings(max_speed_m_s=20.0, on_edge="stop"))
    assert declared_behaviour == STOP_BEHAVIOUR_SHA256
    assert declared["on_edge"] == "stop" and declared["hold_limit"] is None
    assert declared["relocation_radius_m"] is None
    assert declared["holds"] == [] and declared["hold_counts"] == NO_HOLDS
    assert declared["departures_from_wrf"] == nm.DEPARTURES_FROM_WRF
    settings = declared.pop("settings")
    assert settings.pop("on_edge") == "stop" and settings.pop("hold_limit") is None
    assert settings.pop("relocation_radius_m") is None
    for key in HOLD_RECORD_KEYS:
        declared.pop(key)
    assert {**declared, "settings": settings} == record


def test_an_edge_hit_is_held_and_the_tracker_resumes_on_a_clean_centroid() -> None:
    settings = _hold_settings()
    g, geo, state, words, window, search = _edge_case(settings)
    assert nm.window_edge_defect(words, window, geo.centres) is not None
    centre = [state.vc_i, state.vc_j]
    mover = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                          surface=geo, state=state)
    mover._accept_surface(_clocked(g, 20), 20, words, window, search=search)
    # vc is not moved: no centroid of the window edge is taken.
    assert [state.vc_i, state.vc_j] == centre
    assert mover.consecutive_holds == 1
    (hold,) = mover.holds
    assert hold == {
        "parent_step": 20, "model_time_s": 60.0, "event": "edge-hold",
        "reason": nm.window_edge_defect(words, window, geo.centres), "window": list(window),
        "search_reach_child_cells": 15.0,
        "maximum_child_face": [int(words[1]), int(words[2])], "value": words[3],
        "centroid_child_faces": [words[9], words[10]], "centre_child_faces": centre,
        "centre_m": [g.origin_x_m + centre[0] * g.dx_m, g.origin_y_m + centre[1] * g.dy_m],
        "consecutive": 1,
    }
    # One child cell of slack past the 15-cell reach (HOLD_WINDOW_SLACK_CELLS).
    assert "search window" in hold["reason"] and hold["window"] == [44, 76, 44, 76]
    assert mover.tracks[-1]["held"] is True and mover.tracks[-1]["relocated"] is False
    assert mover.tracks[-1]["centre_child_faces"] == centre
    # One interval later the vortex is 480 m (24 child cells) from the held
    # centre: beyond one interval's 15-cell reach, inside the 30 cells two
    # intervals reach since the last accepted centre (WRF's max_vortex_speed
    # over the elapsed time, mediation_nest_move.F:482-485).  The window has
    # widened with the held sample, so the centroid is clean and the count resets.
    assert mover.search_intervals == 2
    x0, y0 = g.origin_x_m + 60 * g.dx_m, g.origin_y_m + 60 * g.dy_m
    u, v = _winds(g, [(x0 + 480.0, y0, 150.0, 1.0e5)])
    window = _search_window(mover)
    assert window == geo.window(state, intervals=2) == (29, 91, 29, 91)
    plane = _reference_plane(u, v, geo)
    words = reference_vortex_centroid(plane, window=window, centres=geo.centres, d=geo.d)
    assert nm.window_edge_defect(words, window, geo.centres) is None
    # The window a held sample left pinned would clip it again.
    pinned = geo.window(state)
    assert nm.window_edge_defect(reference_vortex_centroid(plane, window=pinned, centres=geo.centres,
                                                           d=geo.d), pinned, geo.centres) is not None
    mover._accept_surface(_clocked(g, 30), 30, words, window)
    assert abs(state.vc_i - 84.0) < 0.5 and abs(state.vc_j - 60.0) < 0.5
    assert mover.search_intervals == 1
    assert mover.consecutive_holds == 0
    assert mover.tracks[-1]["held"] is False and mover.tracks[-1]["relocated"] is False
    record = mover.record()
    assert record["hold_counts"] == {**NO_HOLDS, "edge-hold": 1}
    assert record["max_consecutive_holds"] == 1 and record["on_edge"] == "hold"
    assert record["settings"]["on_edge"] == "hold" and record["settings"]["hold_limit"] == 2
    assert record["relocation_radius_m"] == 920.0
    assert record["settings"]["relocation_radius_m"] == 920.0
    assert record["surface_tracker"]["relocation_radius_child_cells"] == 46.0
    assert record["departures_from_wrf"] == {**nm.DEPARTURES_FROM_WRF, **nm.HOLD_DEPARTURES_FROM_WRF}


def test_a_held_sample_moves_the_nest_only_toward_the_accepted_centre() -> None:
    """The child holds its position: a held sample adds no move of its own."""

    settings = _hold_settings(interval_steps=10)
    g, geo, state, words, window, search = _edge_case(settings)
    mover = _Untracked(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                       surface=geo, state=state)
    driver = _clocked(g, 20)
    mover._accept_surface(driver, 20, words, window, search=search)
    for step in range(20, 30):
        driver.parent.clock.completed_steps = step
        assert mover.after_parent_step(driver) is None
    assert driver.geometry is g and mover.moves == []


def test_a_sample_with_no_circulation_or_beyond_reach_is_held() -> None:
    settings = _hold_settings(hold_limit=4, relocation_radius_m=1520.0)
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    mover = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                          surface=geo, state=nm.TrackerState(vc_i=60.0, vc_j=61.0))
    first = _search_window(mover)
    assert first == (44, 76, 45, 77)  # the 15-cell reach plus one cell of slack
    # The maximum is inside the window; its centroid is 15.5 cells east of vc.
    far = [1.0, 74.0, 61.0, 0.02, 64.0, 84.0, 51.0, 71.0, 2.0, 75.5, 61.0, 0.001]
    mover._accept_surface(_clocked(g, 10), 10, far, first)
    # Each held sample widens the next window by one interval's reach.
    assert _search_window(mover) == (29, 91, 30, 92)
    words = [1.0, 60.0, 61.0, -0.004, 50.0, 70.0, 51.0, 71.0, 1.5, 60.2, 61.1, -0.01]
    mover._accept_surface(_clocked(g, 20), 20, list(words), _search_window(mover))
    assert _search_window(mover) == (16, 104, 16, 104)  # 45 cells plus slack, clamped
    flat = [1.0, 60.0, 61.0, 0.003, 50.0, 70.0, 51.0, 71.0, 0.0, 0.0, 0.0, 0.003]
    mover._accept_surface(_clocked(g, 30), 30, flat, _search_window(mover))
    assert (mover.state.vc_i, mover.state.vc_j) == (60.0, 61.0)
    assert [h["event"] for h in mover.holds] == ["too-fast-hold", "lost-hold", "lost-hold"]
    assert "15.5" in mover.holds[0]["reason"] and "15.0 cells" in mover.holds[0]["reason"]
    assert "in one interval" in mover.holds[0]["reason"]
    assert mover.holds[0]["centroid_child_faces"] == [75.5, 61.0]
    assert "no cyclonic circulation" in mover.holds[1]["reason"]
    assert "flat" in mover.holds[2]["reason"] and mover.holds[2]["centroid_child_faces"] is None
    assert [h["search_reach_child_cells"] for h in mover.holds] == [15.0, 30.0, 45.0]
    assert [h["window"] for h in mover.holds] == [[44, 76, 45, 77], [29, 91, 30, 92], [16, 104, 16, 104]]
    assert [t["held"] for t in mover.tracks] == [True, True, True]
    # 35 cells east: beyond two intervals' 30 cells, inside the 60 cells four
    # intervals reach since the last accepted centre.  Accepted.
    near = [1.0, 94.0, 61.0, 0.02, 84.0, 104.0, 51.0, 71.0, 2.0, 95.0, 61.0, 0.001]
    mover._accept_surface(_clocked(g, 40), 40, near, _search_window(mover))
    assert (mover.state.vc_i, mover.state.vc_j) == (95.0, 61.0) and mover.consecutive_holds == 0
    assert mover.record()["hold_counts"] == {**NO_HOLDS, "lost-hold": 2, "too-fast-hold": 1}
    assert mover.max_consecutive_holds == 3
    # The reach boundary, two intervals after the last accepted centre.
    state = nm.TrackerState(vc_i=60.0, vc_j=61.0)
    wide = geo.window(state, intervals=2)
    beyond = [1.0, 89.0, 61.0, 0.02, 79.0, 99.0, 51.0, 71.0, 2.0, 90.5, 61.0, 0.001]
    event, reason = nm.hold_classification(beyond, wide, geo.centres, state, geo.search_cells, intervals=2)
    assert event == "too-fast-hold" and "30.5" in reason and "30.0 cells" in reason
    assert "in the 2 intervals since the last accepted centre" in reason
    inside = beyond[:9] + [90.0, 61.0, 0.001]
    assert nm.hold_classification(inside, wide, geo.centres, state, geo.search_cells, intervals=2) is None
    # A window that does not span the intervals since the last accepted centre is not classified.
    held = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                         surface=geo, state=nm.TrackerState(vc_i=60.0, vc_j=61.0), consecutive_holds=1)
    with pytest.raises(ValueError, match="not the tracker's search window"):
        held._accept_surface(_clocked(g, 20), 20, inside, first)
    # Under stop the same samples are accepted, as before the key.
    stop = nm.MovingNest(settings=_settings(max_speed_m_s=10.0, interval_s=30.0), cupy=numpy,
                         stream=_Stream(), refuse=refuse, surface=geo,
                         state=nm.TrackerState(vc_i=60.0, vc_j=61.0))
    stop._accept_surface(_clocked(g, 10), 10, list(words), first)
    assert (stop.state.vc_i, stop.state.vc_j) == (60.2, 61.1)
    stop._accept_surface(_clocked(g, 30), 30, far, first)
    assert (stop.state.vc_i, stop.state.vc_j) == (75.5, 61.0)


@pytest.mark.parametrize("gamma", [0.0, -1.0e5])
def test_a_window_with_no_circulation_is_lost_not_an_edge(gamma) -> None:
    """No positive box mean: the first strict maximum is a window corner, not a vortex.

    With no winds every box mean is zero, and under a broad anticyclone every
    one is negative and least negative far from its centre; in both the
    kernel's first strict maximum lies on an unclamped window edge.  The
    sample is ``lost-hold`` (no cyclonic circulation within reach), never
    ``edge-hold``, whose reason would say a vortex moved too fast.
    """

    settings = _hold_settings()
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    mover = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                          surface=geo, state=nm.TrackerState(vc_i=60.0, vc_j=61.0))
    x0, y0 = g.origin_x_m + 60 * g.dx_m, g.origin_y_m + 61 * g.dy_m
    u, v = _winds(g, [(x0, y0, 400.0, gamma)])
    window = _search_window(mover)
    words = reference_vortex_centroid(_reference_plane(u, v, geo), window=window,
                                      centres=geo.centres, d=geo.d)
    assert words[0] == 1.0 and not words[3] > 0.0
    if gamma == 0.0:
        assert words[1:4] == [44.0, 45.0, 0.0] and words[8] == 0.0  # the (is, js) corner
    assert nm.window_edge_defect(words, window, geo.centres) is not None
    mover._accept_surface(_clocked(g, 10), 10, words, window)
    (hold,) = mover.holds
    assert hold["event"] == "lost-hold" and "no cyclonic circulation" in hold["reason"]
    assert "follow_max_speed_m_s" not in hold["reason"]
    assert (mover.state.vc_i, mover.state.vc_j) == (60.0, 61.0)


def test_a_window_with_no_finite_box_mean_is_never_held() -> None:
    settings = _hold_settings(hold_limit=4)
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    mover = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                          surface=geo, state=nm.TrackerState(vc_i=60.0, vc_j=61.0))
    window = geo.window(mover.state)
    with pytest.raises(CM1Refusal) as caught:
        mover._accept_surface(_clocked(g, 10), 10, list(LOST_WORDS), window)
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-NEST-FOLLOW-LOST"
    assert "WRF stops on the same failure" in detail.reason and "non-finite" in detail.reason
    assert detail.value == {"parent_step": 10} and mover.holds == [] and mover.tracks == []
    # The first sample of a run has no centre to keep and is taken as stop
    # takes it: all-zero weights leave vc unset and the row says so.
    first = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse, surface=geo)
    flat = [1.0, 60.0, 61.0, 0.0, 50.0, 70.0, 51.0, 71.0, 0.0, 0.0, 0.0, 0.0]
    first._accept_surface(_clocked(g, 1), 1, flat, geo.window(first.state))
    assert not first.state.ready and first.tracks[-1]["centre_m"] is None
    assert first.holds == [] and first.tracks[-1]["held"] is False


def test_a_non_finite_centroid_stops_by_name_under_hold() -> None:
    """A NaN box mean one box from a finite maximum: ``vmin``, the weight sum and
    the centroid are NaN (``mediation_nest_move.F:556-606``).

    ``stop`` takes that centroid as WRF does, so ``vc`` is NaN and the next
    move decision raises an unnamed ``ValueError`` (``fortran_int``): the
    path before the key, kept.  ``hold`` cannot classify it (NaN compares
    false everywhere) and refuses ``CM1-CARTESIAN-NEST-FOLLOW-LOST`` before
    any centre is taken, on a sample with or without a kept centre and on a
    relocation.
    """

    settings = _hold_settings(hold_limit=1, relocation_radius_m=640.0)
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    x0, y0 = g.origin_x_m + 60 * g.dx_m, g.origin_y_m + 61 * g.dy_m
    u, v = _winds(g, [(x0, y0, 150.0, 1.0e5)])
    zbar = _reference_plane(u, v, geo)
    state = nm.TrackerState(vc_i=60.0, vc_j=61.0)
    window = geo.window(state)
    clean = reference_vortex_centroid(zbar, window=window, centres=geo.centres, d=geo.d)
    assert clean[:3] == [1.0, 60.0, 61.0] and nm.centroid_is_finite(clean)
    box_is, box_js = int(clean[4]), int(clean[6])
    zbar[box_js, box_is] = numpy.nan  # the (js, is) corner of the centroid box, not the maximum
    words = reference_vortex_centroid(zbar, window=window, centres=geo.centres, d=geo.d)
    assert words[:3] == clean[:3] and words[3] == clean[3]
    assert all(math.isnan(words[k]) for k in (8, 9, 10, 11)) and not nm.centroid_is_finite(words)
    assert nm.hold_classification(words, window, geo.centres, state, geo.search_cells) is None
    # stop: the path before the key.  The NaN centre ends the run one step later, unnamed.
    stop = nm.MovingNest(settings=_settings(max_speed_m_s=10.0, interval_s=30.0), cupy=numpy,
                         stream=_Stream(), refuse=refuse, surface=geo,
                         state=nm.TrackerState(vc_i=60.0, vc_j=61.0))
    stop._accept_surface(_clocked(g, 10), 10, list(words), window)
    assert math.isnan(stop.state.vc_i) and math.isnan(stop.state.vc_j)
    with pytest.raises(ValueError, match="non-finite tracker coordinate"):
        nm.wrf_move_cells(stop.state.vc_i, g.nx, g.refinement_ratio)
    # hold with a kept centre.
    mover = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                          surface=geo, state=nm.TrackerState(vc_i=60.0, vc_j=61.0))
    with pytest.raises(CM1Refusal) as caught:
        mover._accept_surface(_clocked(g, 10), 10, list(words), window)
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-NEST-FOLLOW-LOST" and "not finite" in detail.reason
    assert "does not hold this sample" in detail.reason and "NaN centre" in detail.reason
    assert detail.value["centroid_box"] == [int(words[4]), int(words[5]), int(words[6]), int(words[7])]
    assert detail.value["maximum_child_face"] == [60, 61] and math.isnan(detail.value["centroid_weight_sum"])
    assert (mover.state.vc_i, mover.state.vc_j) == (60.0, 61.0) and mover.holds == [] and mover.tracks == []
    # hold before the first centre.
    first = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse, surface=geo)
    whole = geo.window(first.state)
    with pytest.raises(CM1Refusal) as caught:
        first._accept_surface(_clocked(g, 1), 1, reference_vortex_centroid(
            zbar, window=whole, centres=geo.centres, d=geo.d), whole)
    assert caught.value.detail.code == "CM1-CARTESIAN-NEST-FOLLOW-LOST" and not first.state.ready
    # hold on a relocation: the limit is spent, the relocation's centroid is NaN.
    g2, geo2, held, edge_words, edge_window, _ = _held_twice(settings)
    with pytest.raises(CM1Refusal) as caught:
        held._accept_surface(_clocked(g2, 30), 30, edge_words, edge_window, search=lambda w: list(words))
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-NEST-FOLLOW-LOST" and "not finite" in detail.reason
    assert detail.value["window"] == list(geo2.relocation_window(held.state))
    assert [h["event"] for h in held.holds] == ["edge-hold"] and held.consecutive_holds == 1


def _held_twice(settings: nm.FollowSettings, *, second_x_m: float = 720.0):
    """Section 5's vortex under a 10 m/s window, 12 m/s east and then further.

    The first follow sample is edge-held (360 m, 18 cells, against a 15-cell
    reach); the second finds the vortex ``second_x_m`` from the kept centre,
    and its window spans the two intervals since that centre (30 cells).
    """

    g, geo, state, words, window, search = _edge_case(settings)
    mover = nm.MovingNest(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                          surface=geo, state=state)
    mover._accept_surface(_clocked(g, 20), 20, words, window, search=search)
    assert mover.holds[-1]["event"] == "edge-hold" and mover.search_intervals == 2
    x0, y0 = g.origin_x_m + 60 * g.dx_m, g.origin_y_m + 60 * g.dy_m
    u, v = _winds(g, [(x0 + second_x_m, y0, 150.0, 1.0e5)])
    zbar = _reference_plane(u, v, geo)
    second = _search_window(mover)
    assert second == (29, 91, 29, 91)
    words = reference_vortex_centroid(zbar, window=second, centres=geo.centres, d=geo.d)
    return g, geo, mover, words, second, _plane_search(zbar, geo)


def test_holding_past_the_limit_relocates_to_the_strongest_circulation_in_the_radius() -> None:
    # A limit of 1: the relocating sample spans two intervals (30 cells); 40 cells searched.
    settings = _hold_settings(hold_limit=1, relocation_radius_m=800.0)
    g, geo, mover, words, window, search = _held_twice(settings)
    state = mover.state
    centre = [state.vc_i, state.vc_j]
    # 720 m (36 cells) east: on the edge of the widened window too.
    assert nm.window_edge_defect(words, window, geo.centres, intervals=2) is not None
    with pytest.raises(ValueError, match="box-mean plane"):
        mover._accept_surface(_clocked(g, 30), 30, words, window)
    assert mover.consecutive_holds == 1 and len(mover.holds) == 1
    mover._accept_surface(_clocked(g, 30), 30, words, window, search=search)
    relocation = mover.holds[-1]
    assert relocation["event"] == "relocation"
    # 40 cells plus the one cell of slack every hold window carries.
    assert relocation["window"] == [19, 101, 19, 101] == list(geo.relocation_window(nm.TrackerState(*centre)))
    assert relocation["trigger"]["event"] == "edge-hold" and relocation["trigger"]["window"] == list(window)
    assert "2 follow intervals since the last accepted centre" in relocation["trigger"]["reason"]
    assert relocation["from_centre_child_faces"] == centre
    assert abs(state.vc_i - 96.0) < 1.0 and abs(state.vc_j - 60.0) < 0.5
    assert relocation["to_centre_child_faces"] == [state.vc_i, state.vc_j]
    assert abs(relocation["displacement_m"][0] - 720.0) < 20.0 and abs(relocation["displacement_m"][1]) < 10.0
    assert relocation["relocation_radius_m"] == 800.0 and relocation["held_samples"] == 1
    assert "follow_hold_limit = 1" in relocation["reason"] and "800.0 m" in relocation["reason"]
    assert mover.consecutive_holds == 0 and mover.max_consecutive_holds == 1
    row = mover.tracks[-1]
    assert row["relocated"] is True and row["held"] is False
    assert row["search_window_child_faces"] == [19, 101, 19, 101]
    assert mover.record()["hold_counts"] == {**NO_HOLDS, "edge-hold": 1, "relocation": 1}
    # The nest then moves toward the relocated centre one parent cell per parent
    # step, and each move names the relocation as the sample it follows.
    driver = _clocked(g, 30)
    tracked = _Untracked(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                         surface=geo, state=state, tracks=mover.tracks)
    shifts = []
    for step in range(31, 40):
        driver.parent.clock.completed_steps = step
        row = tracked.after_parent_step(driver)
        if row is not None:
            shifts.append(row["shift_parent_cells"])
            assert row["tracked_at_parent_step"] == 30 and row["tracker_held_samples"] == 0
    assert shifts == [[1, 0]] * 9
    assert driver.geometry.parent_i0 == g.parent_i0 + 9


@pytest.mark.parametrize(
    "case,fragment",
    [
        ("edge", "at or beyond the declared radius"),
        ("negative", "no cyclonic circulation lies within follow_relocation_radius_m"),
        ("flat", "no centre is defined"),
    ],
)
def test_a_relocation_that_finds_no_circulation_stops_by_name(case, fragment) -> None:
    # 32 cells: one past the 31 the relocating sample's two intervals need
    # (both windows then carry the same one cell of slack).
    settings = _hold_settings(hold_limit=1, relocation_radius_m=640.0)
    g, geo, mover, words, window, search = _held_twice(settings)
    if case == "negative":
        search = lambda w: [1.0, 60.0, 60.0, -0.001] + [0.0] * (CENTROID_WORDS - 4)  # noqa: E731
    elif case == "flat":
        search = lambda w: [1.0, 60.0, 60.0, 0.001] + [0.0] * (CENTROID_WORDS - 4)  # noqa: E731
    state = mover.state
    centre = [state.vc_i, state.vc_j]
    with pytest.raises(CM1Refusal) as caught:
        mover._accept_surface(_clocked(g, 30), 30, words, window, search=search)
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-NEST-FOLLOW-RELOCATION"
    assert detail.option == "nest.follow_relocation_radius_m"
    assert fragment in detail.reason and "search window" in detail.reason
    assert detail.value["relocation_window"] == [27, 93, 27, 93]
    assert detail.value["trigger_event"] == "edge-hold" and detail.value["held_samples"] == 1
    assert detail.value["trigger_window"] == [29, 91, 29, 91]
    assert [state.vc_i, state.vc_j] == centre and [h["event"] for h in mover.holds] == ["edge-hold"]


def test_a_corral_move_is_cancelled_per_axis_then_refused_after_the_limit() -> None:
    g = _geometry(parent_i0=9, parent_span_x=6)  # 3 cells clear on the east
    settings = _hold_settings(interval_steps=3)
    mover = _Untracked(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse)
    driver = _clocked(g, 1, dt=0.5)
    mover.state.vc_i, mover.state.vc_j = g.nx / 2 + RATIO, g.ny / 2 + RATIO
    # East is blocked, north is clear: WRF cancels X and moves Y (:849-859).
    row = mover.after_parent_step(driver)
    assert row["shift_parent_cells"] == [0, 1] and row["corral_cancelled_parent_cells"] == [1, 0]
    assert driver.geometry.parent_i0 == 9 and driver.geometry.parent_j0 == g.parent_j0 + 1
    assert mover.state.vc_i == g.nx / 2 + RATIO and mover.state.vc_j == g.ny / 2
    assert mover.holds[0]["event"] == "corral-hold"
    assert mover.holds[0]["cancelled_shift_parent_cells"] == [1, 0]
    assert "corral" in mover.holds[0]["reason"] and "849-859" in mover.holds[0]["reason"]
    # follow_hold_limit * interval_steps = 6 consecutive cancelled parent steps are held.
    for step in range(2, 7):
        driver.parent.clock.completed_steps = step
        assert mover.after_parent_step(driver) is None
    assert mover.consecutive_corral == 6 and len(mover.holds) == 6
    moved = driver.geometry
    driver.parent.clock.completed_steps = 7
    with pytest.raises(CM1Refusal) as caught:
        mover.after_parent_step(driver)
    detail = caught.value.detail
    assert detail.code == "CM1-CARTESIAN-NEST-FOLLOW-CORRAL"
    assert detail.value == {"parent_step": 7, "move": [1, 0], "parent_i0": 9,
                            "parent_j0": moved.parent_j0,
                            "consecutive_corral_parent_steps": 7, "hold_limit_parent_steps": 6}
    assert "follow_hold_limit*interval_steps = 6" in detail.reason
    assert driver.geometry is moved and len(mover.moves) == 1
    record = mover.record()
    assert record["hold_counts"]["corral-hold"] == 6
    assert record["max_consecutive_corral_parent_steps"] == 7
    assert record["corral_hold_limit_parent_steps"] == 6
    assert record["departures_from_wrf"]["corral"] == nm.HOLD_DEPARTURES_FROM_WRF["corral"]


def test_a_step_the_corral_does_not_cancel_resets_the_corral_count() -> None:
    g = _geometry(parent_i0=9, parent_span_x=6)
    mover = _Untracked(settings=_hold_settings(interval_steps=3, hold_limit=1),
                       cupy=numpy, stream=_Stream(), refuse=refuse)
    driver = _clocked(g, 1, dt=0.5)
    mover.state.vc_i, mover.state.vc_j = g.nx / 2 + RATIO, g.ny / 2
    for step in (1, 2, 3):
        driver.parent.clock.completed_steps = step
        assert mover.after_parent_step(driver) is None
    assert mover.consecutive_corral == 3
    mover.state.vc_i = g.nx / 2  # the vortex came back: no move is asked for
    driver.parent.clock.completed_steps = 4
    assert mover.after_parent_step(driver) is None and mover.consecutive_corral == 0
    mover.state.vc_i = g.nx / 2 - RATIO  # a move away from the corral goes ahead
    driver.parent.clock.completed_steps = 5
    row = mover.after_parent_step(driver)
    assert row["shift_parent_cells"] == [-1, 0] and row["corral_cancelled_parent_cells"] == [0, 0]
    assert mover.consecutive_corral == 0 and mover.max_consecutive_corral == 3


def test_a_translating_vortex_is_still_followed_past_the_corral_under_hold() -> None:
    """Where ``stop`` ends the run at the first corral hit, ``hold`` keeps following on y.

    Measured on this case: ``stop`` refuses at parent step 35 after 17 moves;
    with ``hold`` and a limit of 4 (40 parent steps) the nest made 5 more
    one-cell moves north and the run stopped by name at step 75.  Before the
    corral, ``hold`` holds no sample and its tracker rows and moves are
    ``stop``'s.
    """

    stop_mover, _, stopped = _translating_run(_settings(max_speed_m_s=20.0))
    assert stopped.code == "CM1-CARTESIAN-NEST-FOLLOW-CORRAL" and stopped.value["parent_step"] == 35
    mover, driver, refusal = _translating_run(_hold_settings(max_speed_m_s=20.0, interval_s=30.0,
                                                             hold_limit=4))
    assert refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-CORRAL"
    assert refusal.value["parent_step"] == 75
    assert refusal.value["consecutive_corral_parent_steps"] == 41
    after = [m for m in mover.moves if m["parent_step"] > 35]
    assert len(after) == 5
    assert all(m["shift_parent_cells"] == [0, 1] and m["corral_cancelled_parent_cells"] == [1, 0]
               for m in after)
    assert mover.hold_counts == {**NO_HOLDS, "corral-hold": 40}
    assert all(t["held"] is False and t["relocated"] is False for t in mover.tracks)
    # hold's windows reach one cell past stop's on each unclamped side
    # (HOLD_WINDOW_SLACK_CELLS); the same maxima and centroids are found in them.
    strip = {"held", "relocated", "search_window_child_faces"}
    assert [{k: v for k, v in t.items() if k not in strip} for t in mover.tracks[:len(stop_mover.tracks)]] \
        == [{k: v for k, v in t.items() if k != "search_window_child_faces"} for t in stop_mover.tracks]
    lo_i, hi_i, lo_j, hi_j = mover.surface.centres
    for held_row, stop_row in zip(mover.tracks[1:], stop_mover.tracks[1:]):
        is0, ie0, js0, je0 = stop_row["search_window_child_faces"]
        assert held_row["search_window_child_faces"] == [
            max(is0 - 1, lo_i), min(ie0 + 1, hi_i), max(js0 - 1, lo_j), min(je0 + 1, hi_j)]
    before = [{k: v for k, v in m.items() if k not in {"corral_cancelled_parent_cells", "tracker_held_samples"}}
              for m in mover.moves[:len(stop_mover.moves)]]
    assert before == stop_mover.moves
    g = driver.geometry
    assert g.parent_nx - (g.parent_i0 + g.parent_span_x) == 3 and g.placement_defect() is None


def test_held_events_past_the_cap_are_counted_not_listed(monkeypatch) -> None:
    monkeypatch.setattr(nm, "HOLD_EVENT_CAP", 2)
    g = _geometry(parent_i0=9, parent_span_x=6)
    mover = _Untracked(settings=_hold_settings(interval_steps=10, hold_limit=1),
                       cupy=numpy, stream=_Stream(), refuse=refuse)
    driver = _clocked(g, 1, dt=0.5)
    mover.state.vc_i, mover.state.vc_j = g.nx / 2 + RATIO, g.ny / 2
    for step in range(1, 6):
        driver.parent.clock.completed_steps = step
        mover.after_parent_step(driver)
    record = mover.record()
    assert len(record["holds"]) == 2 and record["holds_not_listed"] == 3
    assert record["hold_counts"]["corral-hold"] == 5
    # Corral holds (one per parent step) and held samples (one per follow
    # interval) are capped separately, and every relocation is listed: a long
    # run of corral hits never unlists the jump of a tracked centre.
    mover = nm.MovingNest(settings=_hold_settings(), cupy=numpy, stream=_Stream(), refuse=refuse)
    order = ["corral-hold"] * 5 + ["edge-hold", "relocation", "lost-hold", "too-fast-hold", "relocation",
                                   "corral-hold", "edge-hold", "relocation"]
    for number, event in enumerate(order):
        mover._record_hold({"event": event, "parent_step": number})
    assert [(h["event"], h["parent_step"]) for h in mover.holds] == [
        ("corral-hold", 0), ("corral-hold", 1), ("edge-hold", 5), ("relocation", 6), ("lost-hold", 7),
        ("relocation", 9), ("relocation", 12),
    ]
    assert mover.holds_not_listed == 6
    assert mover.hold_counts == {"edge-hold": 2, "lost-hold": 1, "too-fast-hold": 1, "corral-hold": 6,
                                 "relocation": 3}


# ---------------------------------------------------------------------------
# 10. nest.follow_on_edge / follow_hold_limit / follow_relocation_radius_m.
# ---------------------------------------------------------------------------
#: CM1Config.sha256 and the sha256 of the sorted-key JSON of
#: ``NestBinding.record(parent_dt_s=run.dt_s)``, measured on b200ad4 before
#: the keys existed. The nested example's record carries the Morrison
#: component-validation words, so it was re-measured when
#: those words (and nothing else in the record) were corrected, and again
#: after 8ed0215c added the stock-deck differential to them (the record's
#: only change: the validation and not_checked lines of morrison-mp10).
SEALED_SURFACE_TRACKER_SHA256 = "36c4fc6202d827a130330a53d2ed5e092d739d3c410f60add3f32546f6d0b0f4"
SEALED_SURFACE_TRACKER_RECORD_SHA256 = "f79e81cfba0db848f5ca8396a7aa584e972fe051a0dc509e421fba2625101017"
SEALED_NESTED_EXAMPLE_SHA256 = "be470b625b3487a8b759818968161ef9652d44a28539c5d60a61ae75b9004ef5"
SEALED_NESTED_EXAMPLE_RECORD_SHA256 = "a2628892be12530fd6100657d9bea1a027571369ed5e893ae13b7506652e0fba"
EDGE_KEYS = ("follow_on_edge", "follow_hold_limit", "follow_relocation_radius_m")


def _binding_record_sha256(config) -> str:
    import hashlib
    import json

    from cm1core.cartesian_door import require_nest

    record = require_nest(config).record(parent_dt_s=config.run.dt_s)
    return hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


def test_undeclared_edge_keys_keep_every_sealed_nest_identity() -> None:
    import tomllib
    from pathlib import Path

    from test_nest_config_to_call_site import BASE_TOML, NEST_TABLE

    from cm1core.config import load_config, parse_mapping

    text = NEST_TABLE + SURFACE_KEYS + "follow_corral_m = 5000.0\n"
    config = parse_mapping(tomllib.loads(BASE_TOML + text))
    assert all(getattr(config.nest, key) is None for key in EDGE_KEYS)
    projected = config.canonical_dict()["nest"]
    assert not set(EDGE_KEYS) & set(projected)
    assert config.sha256 == SEALED_SURFACE_TRACKER_SHA256
    assert _binding_record_sha256(config) == SEALED_SURFACE_TRACKER_RECORD_SHA256
    example = load_config(Path(__file__).resolve().parents[1] / "examples"
                          / "cartesian-nested-storm-following-mesh.toml")
    assert example.sha256 == SEALED_NESTED_EXAMPLE_SHA256
    assert _binding_record_sha256(example) == SEALED_NESTED_EXAMPLE_RECORD_SHA256


@pytest.mark.parametrize("policy", ["hold", "stop"])
def test_the_edge_policy_is_admitted_projected_and_bound(policy) -> None:
    import tomllib

    from test_nest_config_to_call_site import BASE_TOML, NEST_TABLE

    from cm1core.cartesian_door import require_nest
    from cm1core.config import NEST_FOLLOW_ON_EDGE_POLICIES, parse_mapping

    # One policy set: the gate admits exactly what the executor implements.
    assert NEST_FOLLOW_ON_EDGE_POLICIES == frozenset(nm.EDGE_POLICIES) == {"stop", "hold"}
    text = NEST_TABLE + SURFACE_KEYS + (HOLD_KEYS if policy == "hold" else 'follow_on_edge = "stop"\n')
    config = parse_mapping(tomllib.loads(BASE_TOML + text))
    limit, radius = (4, 16700.0) if policy == "hold" else (None, None)
    assert config.nest.follow_on_edge == policy and config.nest.follow_hold_limit == limit
    assert config.nest.follow_relocation_radius_m == radius
    projected = config.canonical_dict()["nest"]
    assert projected["follow_on_edge"] == policy
    assert ("follow_hold_limit" in projected) == ("follow_relocation_radius_m" in projected) == (policy == "hold")
    assert config.sha256 != SEALED_SURFACE_TRACKER_SHA256
    binding = require_nest(config)
    settings = binding.follow_settings()
    assert settings.on_edge == policy and settings.hold_limit == limit
    assert settings.relocation_radius_m == radius
    assert settings.holds == (policy == "hold")
    following = binding.record(parent_dt_s=0.5)["storm_following"]
    assert following["follow_on_edge"] == policy and following["follow_hold_limit"] == limit
    assert following["follow_relocation_radius_m"] == radius
    assert ("relocates the centre" in following["live_motion"]) == (policy == "hold")


def _edge_refusal(text: str):
    import tomllib

    from test_nest_config_to_call_site import BASE_TOML

    from cm1core.config import parse_mapping

    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(tomllib.loads(BASE_TOML + text))
    return caught.value.detail


def _fixed_nest_table() -> str:
    from test_nest_config_to_call_site import _nest_text

    return _nest_text(follow_storm=False, follow_interval_s=0.0, follow_max_speed_m_s=0.0)


@pytest.mark.parametrize(
    "text,code,option,fragment",
    [
        ("S" + 'follow_on_edge = "skip"\nfollow_hold_limit = 4\n', "CM1-CFG-NEST-FOLLOW-ON-EDGE",
         "nest.follow_on_edge", "names no implemented policy"),
        ("S" + "follow_on_edge = 1\n", "CM1-CFG-STRING", "nest.follow_on_edge", "exact string"),
        ("P" + HOLD_KEYS, "CM1-CFG-NEST-FOLLOW-ON-EDGE", "nest.follow_on_edge", "never evaluated"),
        ("P" + 'follow_on_edge = "stop"\n', "CM1-CFG-NEST-FOLLOW-ON-EDGE",
         "nest.follow_on_edge", "never evaluated"),
        ("F" + HOLD_KEYS, "CM1-CFG-NEST-FOLLOW-ON-EDGE", "nest.follow_on_edge", "never evaluated"),
        ("S" + 'follow_on_edge = "hold"\nfollow_relocation_radius_m = 6000.0\n',
         "CM1-CFG-NEST-FOLLOW-HOLD-LIMIT", "nest.follow_hold_limit", "no scientific default"),
        ("S" + "follow_hold_limit = 4\n", "CM1-CFG-NEST-FOLLOW-HOLD-LIMIT",
         "nest.follow_hold_limit", "no sample was ever counted"),
        ("S" + 'follow_on_edge = "stop"\nfollow_hold_limit = 4\n', "CM1-CFG-NEST-FOLLOW-HOLD-LIMIT",
         "nest.follow_hold_limit", "no sample was ever counted"),
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 0\nfollow_relocation_radius_m = 6000.0\n',
         "CM1-CFG-INTEGER", "nest.follow_hold_limit", "integer"),
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 2.5\nfollow_relocation_radius_m = 6000.0\n',
         "CM1-CFG-INTEGER", "nest.follow_hold_limit", "integer"),
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = true\nfollow_relocation_radius_m = 6000.0\n',
         "CM1-CFG-INTEGER", "nest.follow_hold_limit", "integer"),
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 4\n', "CM1-CFG-NEST-FOLLOW-RELOCATION",
         "nest.follow_relocation_radius_m", "no scientific default for how far"),
        ("S" + "follow_relocation_radius_m = 6000.0\n", "CM1-CFG-NEST-FOLLOW-RELOCATION",
         "nest.follow_relocation_radius_m", "never evaluated"),
        ("S" + 'follow_on_edge = "stop"\nfollow_relocation_radius_m = 6000.0\n',
         "CM1-CFG-NEST-FOLLOW-RELOCATION", "nest.follow_relocation_radius_m", "never evaluated"),
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 4\nfollow_relocation_radius_m = 0.0\n',
         "CM1-CFG-NUMERIC-RANGE", "nest.follow_relocation_radius_m", "non-positive"),
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 4\nfollow_relocation_radius_m = true\n',
         "CM1-CFG-NUMERIC-TYPE", "nest.follow_relocation_radius_m", "non-numeric"),
        # (4 + 1) * 3300 m reach + one 200 m child cell = 16700 m: the sample that
        # relocates spans five follow intervals.
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 4\nfollow_relocation_radius_m = 16699.0\n',
         "CM1-CFG-NEST-FOLLOW-RELOCATION", "nest.follow_relocation_radius_m", "no further than the window"),
        # One interval's reach plus a cell is no longer enough once samples are held.
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 4\nfollow_relocation_radius_m = 3500.0\n',
         "CM1-CFG-NEST-FOLLOW-RELOCATION", "nest.follow_relocation_radius_m", "spans 5 follow intervals"),
        # (1 + 1) * 3300 m + 200 m = 6800 m.
        ("S" + 'follow_on_edge = "hold"\nfollow_hold_limit = 1\nfollow_relocation_radius_m = 6799.0\n',
         "CM1-CFG-NEST-FOLLOW-RELOCATION", "nest.follow_relocation_radius_m", "spans 2 follow intervals"),
    ],
)
def test_the_edge_policy_keys_are_refused_by_key_and_value(text, code, option, fragment) -> None:
    from test_nest_config_to_call_site import NEST_TABLE

    table = {"S": NEST_TABLE + SURFACE_KEYS, "P": NEST_TABLE, "F": _fixed_nest_table()}[text[0]]
    detail = _edge_refusal(table + text[1:])
    assert detail.code == code, detail.reason
    assert detail.option == option
    assert fragment in detail.reason
    assert detail.remedy


@pytest.mark.parametrize("limit,radius_m", [(4, 16700.0), (1, 6800.0)])
def test_a_relocation_radius_one_child_cell_past_the_relocating_reach_is_admitted(limit, radius_m) -> None:
    import tomllib

    from test_nest_config_to_call_site import BASE_TOML, NEST_TABLE

    from cm1core.config import parse_mapping

    text = NEST_TABLE + SURFACE_KEYS + (
        f'follow_on_edge = "hold"\nfollow_hold_limit = {limit}\nfollow_relocation_radius_m = {radius_m!r}\n'
    )
    assert parse_mapping(tomllib.loads(BASE_TOML + text)).nest.follow_relocation_radius_m == radius_m


# ---------------------------------------------------------------------------
# 11. The hold policy through the tracker's launches: on the host through the
# g++-compiled kernel text, and on a device.  The device test is opt-in
# exactly like tests/test_nest_live_operators.py; on a GPU node:
#   GM1_NEST_DEVICE_TESTS=1 python3 -m pytest -q -p no:cacheprovider \
#       tests/test_nest_move.py -k "tracker_launch"
# ---------------------------------------------------------------------------
import os as _os  # noqa: E402

requires_opted_in_device = pytest.mark.skipif(
    _os.environ.get("GM1_NEST_DEVICE_TESTS") != "1",
    reason="GM1_NEST_DEVICE_TESTS=1 is not set; no CUDA context is opened unless asked for by name",
)


class _Downloadable(numpy.ndarray):
    def get(self, stream=None):
        return numpy.asarray(self)


class _HostCupy:
    """NumPy under the CuPy names ``_track_surface`` reads, with ``.get()`` on its buffers."""

    def __getattr__(self, name):
        return getattr(numpy, name)

    @staticmethod
    def zeros(shape, dtype=float):
        return numpy.zeros(shape, dtype=dtype).view(_Downloadable)


def _hold_through_the_tracker_launch(cp, module, stream) -> None:
    """``MovingNest._track_surface`` end to end: launched kernels, downloaded words, policy.

    The words must match the CPU reference (the found flag and the maximum's
    face exactly), and the policy must follow, hold the edge-clipped sample,
    relocate on the next one through a second launch over the same plane, and
    resume on a clean one, exactly as it does on the reference words.
    """

    settings = _hold_settings(hold_limit=1, relocation_radius_m=800.0)
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, settings)
    seen = []

    class _Seen(nm.MovingNest):
        def _accept_surface(self, driver, step, words, window, *, search=None):
            def watched(other):
                found = search(other)
                seen.append(("relocation", list(found), other))
                return found

            seen.append(("sample", list(words), window))
            nm.MovingNest._accept_surface(self, driver, step, words, window, search=watched)

    mover = _Seen(settings=settings, cupy=cp, stream=stream, refuse=refuse, surface=geo, module=module)
    # A 12 m/s vortex under a 10 m/s window: edge-held at 20 (18 cells against
    # 15), still on the edge of the two-interval window at 30 (36 against 30),
    # relocated there, then followed at 40 (3 cells in one interval).
    x0, y0 = g.origin_x_m + 40 * g.dx_m, g.origin_y_m + 60 * g.dy_m
    for step, dx_m, event in ((10, 0.0, None), (20, 360.0, "edge-hold"), (30, 720.0, "relocation"),
                              (40, 780.0, None)):
        u, v = _winds(g, [(x0 + dx_m, y0, 150.0, 1.0e5)])
        before = nm.TrackerState(vc_i=mover.state.vc_i, vc_j=mover.state.vc_j)
        zbar = _reference_plane(u, v, geo)
        want_window = geo.window(before, intervals=mover.search_intervals)
        want = reference_vortex_centroid(zbar, window=want_window, centres=geo.centres, d=geo.d)
        with stream:
            arrays = {"u3d": cp.asarray(u), "v3d": cp.asarray(v)}
        stream.synchronize()
        driver = _clocked(g, step)
        driver.child = SimpleNamespace(dynamics=SimpleNamespace(arrays=arrays))
        count = len(mover.holds)
        mover._track_surface(driver, step)
        launched = [s for s in seen if s[0] == "sample"][-1]
        assert launched[2] == want_window
        if step == 30:
            assert launched[2] == (16, 71, 29, 91)  # widened by the held sample and the slack, clamped west
        assert launched[1][:3] == want[:3], step
        numpy.testing.assert_allclose(launched[1], want, rtol=1.0e-6, atol=1.0e-9)
        if event == "relocation":
            _, words, window = seen[-1]
            assert window == geo.relocation_window(before)
            expected = reference_vortex_centroid(zbar, window=window, centres=geo.centres, d=geo.d)
            assert words[:3] == expected[:3]
            numpy.testing.assert_allclose(words, expected, rtol=1.0e-6, atol=1.0e-9)
        assert [h["event"] for h in mover.holds[count:]] == ([] if event is None else [event]), step
        assert mover.tracks[-1]["held"] is (event == "edge-hold"), step
        assert mover.tracks[-1]["relocated"] is (event == "relocation"), step
    assert mover.hold_counts == {**NO_HOLDS, "edge-hold": 1, "relocation": 1}
    assert mover.consecutive_holds == 0
    assert abs(mover.state.vc_i - 79.0) < 0.5 and abs(mover.state.vc_j - 60.0) < 0.5


@requires_gxx
def test_the_hold_policy_reads_the_tracker_launch_words_on_the_host(move_harness) -> None:
    from tools.oracle.nest_move_oracle import GxxModule

    _hold_through_the_tracker_launch(_HostCupy(), GxxModule(move=move_harness), _Stream())


@requires_opted_in_device
def test_the_hold_policy_reads_the_tracker_launch_words_on_a_device() -> None:
    cupy = pytest.importorskip("cupy")
    from cm1core.nest_move_kernels import compile_nest_move_module

    _hold_through_the_tracker_launch(cupy, compile_nest_move_module(cupy),
                                     cupy.cuda.Stream(non_blocking=True))


# ---------------------------------------------------------------------------
# 12. A sub-vortex at the search-window edge while the parent-scale circulation
# is still inside: the failure that ended a 2-hour moving 20 m nest after 90 s
# (CM1-CARTESIAN-NEST-FOLLOW-SPEED with follow_max_speed_m_s = 20,
# follow_interval_s = 15 and a 200 m smoothing radius), reproduced on
# synthetic lowest-level winds, and the smoothing radius measured against it.
# ---------------------------------------------------------------------------
ORBIT_DT_S = 0.375  # parent step; 40 steps per 15 s follow interval


def _orbit_geometry() -> NestGeometry:
    """A 100 m parent with a 5 km, 20 m child (ratio 5)."""

    return NestGeometry(
        parent_nx=400, parent_ny=400, parent_dx_m=100.0, parent_dy_m=100.0, refinement_ratio=5,
        time_refinement_ratio=3, parent_i0=175, parent_j0=175, parent_span_x=50, parent_span_y=50,
        spec_zone=1, relax_zone=4, feedback=1,
    )


def _orbit_settings(radius_m: float, **overrides) -> nm.FollowSettings:
    values = dict(tracker=nm.SURFACE_TRACKER, interval_steps=40, interval_s=15.0, max_speed_m_s=20.0,
                  smoothing_radius_m=radius_m, corral_cells_x=30, corral_cells_y=30)
    values.update(overrides)
    return nm.FollowSettings(**values)


def _box_mean_plane(zeta, geo: nm.SurfaceTrackerGeometry, *, noise: float = 0.0):
    """``reference_zeta_box`` vectorised over centres, in its own ``a, b`` order.

    Every centre accumulates the same float64 words in the same order as the
    reference loop, so the plane is bit-identical (proved below) and the
    tracker's first strict maximum is the one the kernel finds.
    """

    zeta = numpy.asarray(zeta, dtype=numpy.float32)
    if noise:
        j, i = numpy.meshgrid(numpy.arange(zeta.shape[0]), numpy.arange(zeta.shape[1]), indexing="ij")
        hashed = numpy.sin(i * 12.9898 + j * 78.233) * 43758.5453
        zeta = zeta + (noise * math.sqrt(3.0) * (2.0 * (hashed - numpy.floor(hashed)) - 1.0)).astype(numpy.float32)
    h = geo.h
    lo_j, hi_j, lo_i, hi_i = geo.box
    ny1, nx1 = zeta.shape
    j0, j1 = max(lo_j + h, 0), min(hi_j - h, ny1 - 1)
    i0, i1 = max(lo_i + h, 0), min(hi_i - h, nx1 - 1)
    acc = numpy.zeros((j1 - j0 + 1, i1 - i0 + 1), dtype=numpy.float64)
    for a in range(-h, h + 1):
        for b in range(-h, h + 1):
            acc = acc + zeta[j0 + a:j1 + a + 1, i0 + b:i1 + b + 1].astype(numpy.float64)
    out = numpy.zeros((ny1, nx1), dtype=numpy.float64)
    out[j0:j1 + 1, i0:i1 + 1] = acc / float((2 * h + 1) ** 2)
    return out


def _orbiting(g: NestGeometry, *, main=(1000.0, 4.0e5), sub=(30.0, 2.0e4), orbit_m=350.0,
              storm=(5.0, 3.0)):
    """A broad Lamb-Oseen circulation translating with ``storm`` m/s, and a compact
    sub-vortex orbiting it at ``orbit_m`` with the circulation's tangential wind
    there.  The defaults are a 1 km-core parent circulation (40 m/s at 1 km)
    and a 30 m-core sub-vortex orbiting 350 m out at 21.0 m/s, just past the
    declared 20 m/s."""

    x0, y0 = g.centre_x_m, g.centre_y_m
    rc, gamma = main
    speed = gamma / (2.0 * math.pi * orbit_m) * -math.expm1(-orbit_m * orbit_m / (rc * rc))

    def vortices(t):
        mx, my = x0 + storm[0] * t, y0 + storm[1] * t
        out = [(mx, my, rc, gamma)]
        if sub is not None:
            theta = speed / orbit_m * t
            out.append((mx + orbit_m * math.cos(theta), my + orbit_m * math.sin(theta), *sub))
        return out

    return vortices, speed


class _OrbitTracked(nm.MovingNest):
    """``_HostTracked`` on the 5 km child, with the vectorised box mean."""

    noise = 0.0

    def track(self, driver, step):
        t = float(driver.parent.clock.model_seconds)
        g = driver.geometry
        u, v = _winds(g, driver.vortices(t))
        zeta = nm.lowest_level_corner_vorticity(u, v, nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m, xp=numpy)
        search = _plane_search(_box_mean_plane(zeta, self.surface, noise=self.noise), self.surface)
        window = _search_window(self)
        self._accept_surface(driver, step, search(window), window, **_search_argument(self, search))
        if self.state.ready:
            mx, my = driver.vortices(t)[0][:2]
            driver.samples.append({
                "t": t, "held": self.tracks[-1].get("held"),
                "tracker_error_m": math.hypot(g.origin_x_m + self.state.vc_i * g.dx_m - mx,
                                              g.origin_y_m + self.state.vc_j * g.dy_m - my),
                "nest_offset_m": math.hypot(g.centre_x_m - mx, g.centre_y_m - my),
                "main_child_face": ((mx - g.origin_x_m) / g.dx_m, (my - g.origin_y_m) / g.dy_m),
            })

    def execute(self, driver, *, move_i, move_j):
        driver.geometry = _moved(driver.geometry, move_i, move_j)
        return driver.geometry


def _orbit_run(settings: nm.FollowSettings, vortices, *, seconds: float, noise: float = 0.0):
    g = _orbit_geometry()
    driver = _clocked(g, 0, dt=ORBIT_DT_S, vortices=vortices)
    driver.samples = []
    driver.geometries = [g]
    mover = _OrbitTracked(settings=settings, cupy=numpy, stream=_Stream(), refuse=refuse,
                          surface=nm.SurfaceTrackerGeometry.build(g, settings))
    mover.noise = noise
    for step in range(1, int(round(seconds / ORBIT_DT_S)) + 1):
        driver.parent.clock.completed_steps = step
        driver.parent.clock.model_seconds = step * ORBIT_DT_S
        try:
            mover.after_parent_step(driver)
        except CM1Refusal as caught:
            return mover, driver, caught.detail
        driver.geometries.append(driver.geometry)
    return mover, driver, None


def test_the_vectorised_box_mean_is_the_reference_word_for_word() -> None:
    g = _tornado_geometry()
    geo = nm.SurfaceTrackerGeometry.build(g, _settings())
    main = (g.origin_x_m + 70 * g.dx_m, g.origin_y_m + 55 * g.dy_m, 150.0, 1.0e5)
    sub = (g.origin_x_m + 40 * g.dx_m, g.origin_y_m + 50 * g.dy_m, 15.0, 3.0e3)
    u, v = _winds(g, [main, sub])
    assert _box_mean_plane(nm.lowest_level_corner_vorticity(
        u, v, nx=g.nx, ny=g.ny, dx_m=g.dx_m, dy_m=g.dy_m, xp=numpy), geo).tobytes() \
        == _reference_plane(u, v, geo).tobytes()


def test_a_sub_vortex_orbiting_out_of_reach_stops_the_run_under_stop() -> None:
    """The reproduction: ``stop`` ends the run on a window edge within 90 s.

    Measured: the first sample's maximum is the 200 m box holding the
    sub-vortex and the shoulder of the parent circulation, 266 m from the
    parent core; the sub-vortex then orbits at 21 m/s and the maximum reaches
    the edge of the 300 m search window at 75 s, while the parent-scale
    circulation is still inside that window.
    """

    vortices, speed = _orbiting(_orbit_geometry())
    assert 20.0 < speed < 21.5
    mover, driver, refusal = _orbit_run(_orbit_settings(200.0), vortices, seconds=600.0)
    assert refusal is not None and refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-SPEED"
    assert refusal.value["parent_step"] * ORBIT_DT_S == 75.0
    assert driver.samples[0]["tracker_error_m"] > 200.0  # captured on the first sample
    window = refusal.value["window"]
    t_stop = refusal.value["parent_step"] * ORBIT_DT_S
    mx, my = vortices(t_stop)[0][:2]
    g = driver.geometry
    face = ((mx - g.origin_x_m) / g.dx_m, (my - g.origin_y_m) / g.dy_m)
    assert window[0] < face[0] < window[1] and window[2] < face[1] < window[3]
    # The refused maximum is INSIDE the declared reach: 14.65 cells (293 m in
    # 15 s, 19.5 m/s) east of the kept centre 124.35, on the INT-truncated
    # window edge int(124.35 + 15) = 139 (module docstring, THE HOLD POLICY).
    # stop keeps WRF's edges bit for bit; hold widens its own by one cell.
    kept_i = mover.state.vc_i  # the refused sample took no centroid
    assert 14.0 < refusal.value["maximum_child_face"][0] - kept_i < 15.0
    assert window[1] == int(kept_i + 15.0) == refusal.value["maximum_child_face"][0]


def _orbit_hold(radius_m: float = 200.0) -> nm.FollowSettings:
    """``hold`` on the reproduction's tracker: a limit of 4 and the smallest
    admissible relocation radius, (4 + 1) * 300 m + one 20 m cell."""

    return _orbit_settings(radius_m, on_edge="hold", hold_limit=4, relocation_radius_m=1520.0)


def test_hold_survives_the_sub_vortex_while_the_tracker_stays_on_it(monkeypatch) -> None:
    """``hold`` keeps the reproduction running; it does not make the tracker follow
    the parent circulation.  The smoothing radius does.

    Measured with ``R`` = 200 m, a limit of 4 and a 1520 m radius: the run
    survives ten minutes holding no sample at all -- the maximum ``stop``
    refused at 75 s lay inside the reach on the truncated window edge, and
    ``hold``'s windows reach one cell past it.  With WRF's edges (the slack
    set to zero) the same run holds 6 samples, none consecutive, and relocates
    nothing.  Either way, on every one of the 41 samples the tracked centre is
    the box holding the sub-vortex and the circulation's shoulder, 219-272 m
    (219-319 m with WRF's edges) from the parent circulation, and the nest
    centre is within 360 (440) m of it.  The same run with ``R`` = 600 m holds
    nothing and tracks the parent circulation within 51 m (as ``stop`` does,
    below).
    """

    settings = _orbit_hold()
    vortices, _ = _orbiting(_orbit_geometry())
    mover, driver, refusal = _orbit_run(settings, vortices, seconds=600.0)
    assert refusal is None and len(driver.samples) == 41
    assert mover.record()["hold_counts"] == NO_HOLDS and mover.max_consecutive_holds == 0
    assert mover.record()["surface_tracker"]["search_window_slack_child_cells"] == 1.0
    errors = [s["tracker_error_m"] for s in driver.samples]
    # On the sub-vortex box on every sample, never on the parent circulation.
    assert 200.0 <= min(errors) and max(errors) <= 330.0
    assert max(s["nest_offset_m"] for s in driver.samples) <= 450.0
    assert all(row["held"] is False for row in mover.tracks)
    for old, new in zip(driver.geometries, driver.geometries[1:]):
        assert abs(new.parent_i0 - old.parent_i0) <= 1 and abs(new.parent_j0 - old.parent_j0) <= 1
    assert len({m["parent_step"] for m in mover.moves}) == len(mover.moves)
    assert all(m["tracked_at_parent_step"] is not None for m in mover.moves)
    lo_i, hi_i, lo_j, hi_j = mover.surface.centres
    assert all(lo_i < s["main_child_face"][0] < hi_i and lo_j < s["main_child_face"][1] < hi_j
               for s in driver.samples)
    wide, wide_driver, wide_refusal = _orbit_run(_orbit_hold(600.0), vortices, seconds=600.0)
    assert wide_refusal is None and wide.hold_counts == NO_HOLDS and len(wide_driver.samples) == 41
    assert max(s["tracker_error_m"] for s in wide_driver.samples) <= 60.0
    # WRF's edges: the samples hold holds are the ones stop refuses, and every
    # held sample keeps the centre where it was.
    monkeypatch.setattr(nm, "HOLD_WINDOW_SLACK_CELLS", 0.0)
    mover, driver, refusal = _orbit_run(settings, vortices, seconds=600.0)
    assert refusal is None and len(driver.samples) == 41
    counts = mover.record()["hold_counts"]
    assert counts == {**NO_HOLDS, "edge-hold": 6} and mover.max_consecutive_holds == 1
    assert mover.record()["surface_tracker"]["search_window_slack_child_cells"] == 0.0
    errors = [s["tracker_error_m"] for s in driver.samples]
    assert 200.0 <= min(errors) and max(errors) <= 330.0
    assert max(s["nest_offset_m"] for s in driver.samples) <= 450.0
    events = mover.holds
    assert len(events) == sum(counts.values())
    for event in events:
        assert event["parent_step"] % settings.interval_steps == 0
        assert event["reason"] and len(event["window"]) == 4
        assert event["search_reach_child_cells"] == 15.0
    held_rows = 0
    for before, row in zip(mover.tracks, mover.tracks[1:]):
        if row["held"]:
            held_rows += 1
            # The tracked centre does not move on a held sample (vc is carried with the
            # moves, so its ground position is the invariant).
            assert row["centre_m"] == pytest.approx(before["centre_m"], abs=1.0e-6)
    assert held_rows == 6


def _translating(speed_m_s: float, *, core_m: float = 1000.0, gamma: float = 4.0e5):
    """A ``core_m``-core vortex translating east at ``speed_m_s``."""

    g = _orbit_geometry()
    x0, y0 = g.centre_x_m, g.centre_y_m
    return lambda t: [(x0 + speed_m_s * t, y0, core_m, gamma)]


@pytest.mark.parametrize("speed_m_s,wrf_edges", [(19.0, (10, 660.0)), (19.5, (25, 970.0)), (20.0, None)])
def test_a_vortex_within_the_declared_speed_is_not_held_on_the_truncated_edge(
        speed_m_s, wrf_edges, monkeypatch) -> None:
    """``HOLD_WINDOW_SLACK_CELLS``, measured against WRF's ``INT``-truncated edges.

    A 1 km-core vortex translating at 19, 19.5 and 20 m/s under the
    reproduction's 20 m/s, 15 s reach (15 cells) with ``R`` = 600 m.  With
    WRF's edges (``mediation_nest_move.F:482-485``: with the kept centre at
    128.28 the east edge is int(143.28) = 143, and the vortex at 142.75, 14.47
    cells away, has its corner maximum there) ``hold`` edge-holds 10 and 25 of
    41 samples at 19 and 19.5 m/s with the nest parked up to 660 and 970 m off,
    and at 20 m/s spends the limit and stops with -RELOCATION at 150 s; ``stop``
    refuses the 19 m/s vortex at 45 s.  With the windows one cell past the
    reach no sample is held and the centre stays within 4 m, and the accepted
    centres still never move faster than the declared speed (the too-fast test
    keeps the reach).
    """

    vortices = _translating(speed_m_s)
    mover, driver, refusal = _orbit_run(_orbit_hold(600.0), vortices, seconds=600.0)
    assert refusal is None and len(driver.samples) == 41 and mover.hold_counts == NO_HOLDS
    assert max(s["tracker_error_m"] for s in driver.samples) <= 5.0
    assert max(s["nest_offset_m"] for s in driver.samples) <= 400.0
    from tools.check_nest_following import summarize

    steps = summarize(mover.record())["centre_steps"]
    assert steps and all(step["axis_distance_m"] <= step["allowed_m"] for step in steps)
    monkeypatch.setattr(nm, "HOLD_WINDOW_SLACK_CELLS", 0.0)
    mover, driver, refusal = _orbit_run(_orbit_hold(600.0), vortices, seconds=600.0)
    if wrf_edges is None:
        assert refusal is not None and refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-RELOCATION"
        assert refusal.value["parent_step"] * ORBIT_DT_S == 150.0
        assert "raise nest.follow_relocation_radius_m" in refusal.remedy
        assert "nest.follow_max_speed_m_s" in refusal.remedy
        return
    held, parked_m = wrf_edges
    assert refusal is None and mover.hold_counts == {**NO_HOLDS, "edge-hold": held}
    assert parked_m - 20.0 <= max(s["nest_offset_m"] for s in driver.samples) <= parked_m
    for event in mover.holds:
        # Each held maximum lies on the truncated east edge, inside the reach.
        assert event["maximum_child_face"][0] == event["window"][1]
        assert 0.0 < event["window"][1] - event["centre_child_faces"][0] < 15.0 * event["consecutive"]
    if speed_m_s == 19.0:
        _, _, stopped = _orbit_run(_orbit_settings(600.0), vortices, seconds=600.0)
        assert stopped is not None and stopped.code == "CM1-CARTESIAN-NEST-FOLLOW-SPEED"
        assert stopped.value["parent_step"] * ORBIT_DT_S == 45.0


def test_a_lagging_accepted_centre_is_recovered_only_with_the_slack(monkeypatch) -> None:
    """What the widened reach guarantees, and what it does not.

    A 300 m-core vortex at 19.5 m/s under ``R`` = 840 m.  With WRF's edges the
    first sample is edge-held (14.6 cells against a 15-cell edge at 14.7),
    and the accepted centres that follow lag the vortex: a centroid box
    clipped by the admissible boundary puts the centre 4 cells behind at
    105 s and 21 cells behind at 180 s, each cycle's widened windows grow at
    exactly the vortex's speed from a centre further behind, and the run
    stops with -RELOCATION at 480 s with the tracked centre 2.1 km off after
    24 edge-holds.  With the slack: 4 edge-holds, none consecutive, the centre
    within 293 m, ten minutes.
    """

    vortices = _translating(19.5, core_m=300.0, gamma=1.0e5)
    mover, driver, refusal = _orbit_run(_orbit_hold(840.0), vortices, seconds=600.0)
    assert refusal is None and len(driver.samples) == 41
    assert mover.hold_counts == {**NO_HOLDS, "edge-hold": 4} and mover.max_consecutive_holds == 1
    assert max(s["tracker_error_m"] for s in driver.samples) <= 300.0
    monkeypatch.setattr(nm, "HOLD_WINDOW_SLACK_CELLS", 0.0)
    mover, driver, refusal = _orbit_run(_orbit_hold(840.0), vortices, seconds=600.0)
    assert refusal is not None and refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-RELOCATION"
    assert refusal.value["parent_step"] * ORBIT_DT_S == 480.0
    assert mover.hold_counts == {**NO_HOLDS, "edge-hold": 24} and mover.max_consecutive_holds == 4
    assert max(s["tracker_error_m"] for s in driver.samples) >= 2000.0


def test_over_the_declared_speed_only_the_relocation_radius_keeps_the_run_alive() -> None:
    """The smallest admitted relocation radius has no margin.

    A 1 km-core vortex at 20.5 m/s, 2.5 per cent over the declared 20 m/s,
    under ``R`` = 600 m: every sample is edge-held, the widened window never
    outpaces the vortex, and after the limit the relocation with the smallest
    admitted radius (1520 m) finds the maximum on its own edge and stops the
    run at 90 s.  Twice that (3040 m) survives ten minutes with 7 relocations
    and the nest up to 2 km off; a limit of 8 with the same radius parks the
    nest 3.5 km off with no relocation at all.  Only the radius helps.
    """

    vortices = _translating(20.5)
    _, _, refusal = _orbit_run(_orbit_hold(600.0), vortices, seconds=600.0)
    assert refusal is not None and refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-RELOCATION"
    assert refusal.value["parent_step"] * ORBIT_DT_S == 90.0
    assert refusal.value["trigger_event"] == "edge-hold" and "at or beyond the declared radius" in refusal.reason
    assert refusal.remedy.startswith("raise nest.follow_relocation_radius_m")
    wide = _orbit_settings(600.0, on_edge="hold", hold_limit=4, relocation_radius_m=3040.0)
    mover, driver, refusal = _orbit_run(wide, vortices, seconds=600.0)
    assert refusal is None and len(driver.samples) == 41
    assert mover.hold_counts["relocation"] == 7 and mover.max_consecutive_holds == 4
    assert 1500.0 <= max(s["nest_offset_m"] for s in driver.samples) <= 2100.0
    longer = _orbit_settings(600.0, on_edge="hold", hold_limit=8, relocation_radius_m=3040.0)
    mover, driver, refusal = _orbit_run(longer, vortices, seconds=600.0)
    assert refusal is None and mover.hold_counts["relocation"] == 0
    assert max(s["nest_offset_m"] for s in driver.samples) >= 3000.0


def _vanishing(speed_m_s: float, gone: tuple[float, float]):
    """A 300 m-core vortex translating east at ``speed_m_s`` whose circulation is
    zero over ``gone`` (inclusive): the samples it misses see no winds at all."""

    g = _orbit_geometry()
    x0, y0 = g.centre_x_m, g.centre_y_m
    return lambda t: [(x0 + speed_m_s * t, y0, 300.0, 0.0 if gone[0] <= t <= gone[1] else 1.0e5)]


@pytest.mark.parametrize("speed_m_s", [15.0, 18.0])
@pytest.mark.parametrize("missed", [1, 4])
def test_a_vortex_missed_for_held_samples_is_found_again_within_the_elapsed_reach(speed_m_s, missed) -> None:
    """The search reach grows with the samples held since the last accepted centre.

    WRF bounds its search by ``max_vortex_speed`` over the time since the
    previous centre (``mediation_nest_move.F:482-485``).  A 20 m child with 20
    m/s over 15 s reaches 15 cells per interval; a vortex at 15 or 18 m/s
    missed for ``missed`` samples is ``missed + 1`` intervals of travel from
    the kept centre, beyond one interval's reach and inside the widened one.
    Measured: every missed sample is one ``lost-hold`` (no winds, so no
    circulation, not an edge), the next sample is accepted, nothing relocates,
    and the tracked centre is within 4.1 m of the vortex after it.
    """

    gone = (59.0, 61.0 + 15.0 * (missed - 1))
    mover, driver, refusal = _orbit_run(_orbit_hold(), _vanishing(speed_m_s, gone), seconds=300.0)
    assert refusal is None and len(driver.samples) == 21
    assert mover.hold_counts == {**NO_HOLDS, "lost-hold": missed}
    assert mover.max_consecutive_holds == missed
    assert [h["search_reach_child_cells"] for h in mover.holds] == [15.0 * (n + 1) for n in range(missed)]
    assert all("no cyclonic circulation" in h["reason"] for h in mover.holds)
    after = [s["tracker_error_m"] for s in driver.samples if s["t"] > gone[1]]
    assert len(after) == 21 - 4 - missed and max(after) <= 5.0


def test_a_pinned_search_window_loses_a_vortex_one_missed_sample_later(monkeypatch) -> None:
    """The negative control: the code before the reach grew, on the same run.

    With the window left at one interval's reach around the kept centre, the
    15 m/s vortex missed once is 30 cells away at the next sample and sits on
    the window edge until the limit is spent: 3 edge-held samples and a
    relocation, the tracked centre 898 m off.  With a 1 km radius (admitted
    before this check) that relocation found nothing and stopped the run at
    120 s.
    """

    monkeypatch.setattr(nm.MovingNest, "search_intervals", property(lambda self: 1))
    mover, driver, refusal = _orbit_run(_orbit_hold(), _vanishing(15.0, (59.0, 61.0)), seconds=300.0)
    assert refusal is None
    assert mover.hold_counts == {**NO_HOLDS, "lost-hold": 1, "edge-hold": 3, "relocation": 1}
    assert max(s["tracker_error_m"] for s in driver.samples) >= 800.0
    short = _orbit_settings(200.0, on_edge="hold", hold_limit=4, relocation_radius_m=1000.0)
    _, _, refusal = _orbit_run(short, _vanishing(15.0, (59.0, 61.0)), seconds=300.0)
    assert refusal is not None and refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-RELOCATION"
    assert refusal.value["parent_step"] * ORBIT_DT_S == 120.0


def test_a_vortex_missed_past_the_limit_stops_by_name_at_the_relocation() -> None:
    """Five missed samples with a limit of 4: the relocation finds no circulation."""

    mover, driver, refusal = _orbit_run(_orbit_hold(), _vanishing(15.0, (59.0, 121.0)), seconds=300.0)
    assert refusal is not None and refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-RELOCATION"
    assert refusal.value["parent_step"] * ORBIT_DT_S == 120.0
    assert "no cyclonic circulation lies within follow_relocation_radius_m" in refusal.reason
    assert mover.hold_counts == {**NO_HOLDS, "lost-hold": 4}


def _receipt(tmp_path, name: str, mover: nm.MovingNest):
    import json

    path = tmp_path / name
    path.write_text(json.dumps({"benchmark": {"nest_execution": {"storm_following": mover.record()}}},
                               allow_nan=False), encoding="utf-8")
    return path


def test_the_following_check_reads_the_receipt_the_tracker_wrote(tmp_path, capsys) -> None:
    """``tools/check_nest_following.py``, the device check's pass condition, on host receipts.

    It passes the reproduction under ``hold``, reports the sub-vortex capture
    as the separation from the ``R`` = 600 m run, and fails a receipt whose
    accepted centre jumped past the reach or that relocated more than allowed.
    """

    import json

    from tools.check_nest_following import main

    vortices, _ = _orbiting(_orbit_geometry())
    narrow, _, refusal = _orbit_run(_orbit_hold(), vortices, seconds=600.0)
    wide, _, wide_refusal = _orbit_run(_orbit_hold(600.0), vortices, seconds=600.0)
    assert refusal is None and wide_refusal is None
    narrow_path, wide_path = _receipt(tmp_path, "narrow.json", narrow), _receipt(tmp_path, "wide.json", wide)
    assert main([str(narrow_path), "--max-relocations", "0", "--compare", str(wide_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["failures"] == [] and report["tracker_samples"] == 41 and report["held_samples"] == 0
    # Nothing held: every accepted centre within one interval's 300 m (measured 264 m).
    assert report["relocations"] == 0 and 200.0 < report["max_axis_step_m"] <= 300.0
    assert report["compare"]["common_samples"] == 41
    # Measured: 179-295 m (median 190 m) on every sample; R = 600 and 840 m,
    # both on the parent circulation, agree within 15-25 m.
    assert 150.0 <= report["compare"]["min_m"] and report["compare"]["max_m"] <= 330.0
    assert main([str(wide_path), "--compare",
                 str(_receipt(tmp_path, "wider.json", _orbit_run(_orbit_hold(840.0), vortices,
                                                                  seconds=600.0)[0]))]) == 0
    assert json.loads(capsys.readouterr().out)["compare"]["max_m"] <= 30.0
    # An accepted centre two reaches away from the one before it.
    record = narrow.record()
    row = record["tracker_events"][2]
    assert row["held"] is False and record["tracker_events"][1]["held"] is False
    row["centre_m"] = [row["centre_m"][0] + 600.0, row["centre_m"][1]]
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps({"storm_following": record}), encoding="utf-8")
    assert main([str(tampered)]) == 1
    assert "beyond the 300.0 m 1 follow interval(s) reach" in json.loads(capsys.readouterr().out)["failures"][0]
    # A relocation is counted against --max-relocations and bounded by radius + R.
    settings = _hold_settings(hold_limit=1, relocation_radius_m=800.0)
    g, geo, mover, words, window, search = _held_twice(settings)
    mover._accept_surface(_clocked(g, 30), 30, words, window, search=search)
    relocated = _receipt(tmp_path, "relocated.json", mover)
    assert main([str(relocated), "--max-relocations", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["relocations"] == 1
    assert main([str(relocated), "--max-relocations", "0"]) == 1
    assert "more than --max-relocations 0" in json.loads(capsys.readouterr().out)["failures"][0]
    with pytest.raises(SystemExit, match="0 storm_following records"):
        (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
        main([str(tmp_path / "empty.json")])


@pytest.mark.parametrize("radius_m", [300.0, 600.0, 840.0])
def test_a_smoothing_radius_near_the_orbit_follows_the_parent_circulation(radius_m) -> None:
    """Measured under ``stop`` on the reproduction: the tracked centre stays
    within 191 m of the parent circulation for ten minutes with 300 m, 51 m
    with 600 m and 33 m with 840 m (WRF's 25/9 of the reach)."""

    vortices, _ = _orbiting(_orbit_geometry())
    mover, driver, refusal = _orbit_run(_orbit_settings(radius_m), vortices, seconds=600.0)
    assert refusal is None and len(driver.samples) == 41
    assert max(s["tracker_error_m"] for s in driver.samples) <= {300.0: 220.0, 600.0: 60.0, 840.0: 40.0}[radius_m]


@pytest.mark.parametrize("radius_m,noise", [(200.0, 0.0), (200.0, 0.02), (300.0, 0.0), (600.0, 0.0),
                                            (840.0, 0.0)])
def test_a_smoothing_radius_many_times_the_core_cannot_centre_a_compact_vortex(radius_m, noise) -> None:
    """Why WRF's 25/9 of the reach (840 m here) is not a default: the box mean saturates.

    A 60 m-core vortex translating at 10 m/s.  Measured under ``stop``: ``R``
    = 200 m keeps the tracked centre within 3 m (53 m with 0.02 s-1 corner
    noise) and 300 m within 87 m; at 600 m and 840 m every box near the vortex
    holds its whole circulation, the first strict maximum falls on the edge of
    that plateau, and the run stops on the window edge at the first follow
    sample.
    """

    vortices, _ = _orbiting(_orbit_geometry(), main=(60.0, 4.0e4), sub=None, storm=(8.0, 6.0))
    mover, driver, refusal = _orbit_run(_orbit_settings(radius_m), vortices, seconds=300.0, noise=noise)
    errors = [s["tracker_error_m"] for s in driver.samples]
    if radius_m >= 600.0:
        # Either the plateau's edge stops the run or the centre sits far off the vortex.
        assert (refusal is not None and refusal.code == "CM1-CARTESIAN-NEST-FOLLOW-SPEED") \
            or min(errors[1:]) >= 100.0
        assert max(errors) >= 250.0
        # hold keeps the run alive on the plateau (measured: 1 and 10 held
        # samples), but it does not centre the vortex: that is R's job.
        held, held_driver, held_refusal = _orbit_run(_orbit_hold(radius_m), vortices, seconds=300.0,
                                                     noise=noise)
        assert held_refusal is None and len(held_driver.samples) == 21
        assert held.hold_counts["edge-hold"] >= 1 and held.hold_counts["relocation"] == 0
        assert max(s["tracker_error_m"] for s in held_driver.samples) >= 250.0
        return
    assert refusal is None and len(errors) == 21
    assert max(errors) <= {(200.0, 0.0): 5.0, (200.0, 0.02): 60.0, (300.0, 0.0): 120.0}[(radius_m, noise)]
    if radius_m == 300.0:
        assert min(errors[1:]) >= 50.0
