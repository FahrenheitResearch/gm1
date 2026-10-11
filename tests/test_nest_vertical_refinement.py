"""Vertical refinement: CM1's rule, WRF's order, and the feedback conservation.

No test here opens a CUDA context.  The kernel text is compiled with ``g++``
(``tools/oracle/nest_vertical_oracle.py``) and compared raw against the NumPy
references in :mod:`cm1core.nest_vertical`; the composition with the existing
capture and restriction kernels is checked through
``tools/oracle/nest_live_oracle.py``.
"""

from __future__ import annotations

import math
import shutil

import pytest

numpy = pytest.importorskip("numpy")

from cm1core.cm1_vertical_grid import build_vertical_levels  # noqa: E402
from cm1core.nest_live_kernels import (  # noqa: E402
    LATTICE_OF_STAGGER,
    lattice_shape,
    reference_capture,
    reference_restrict,
)
from cm1core.nest_vertical import (  # noqa: E402
    VerticalWindow,
    build_cm1_vertical_map,
    cm1_vertical_interpolate,
    feedback_weights,
    plan_vertical_refinement,
    reference_capture_refined,
    reference_restrict_vertical,
    reference_vinterp,
    vertical_refinement_defect,
)

requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")


def parent_deck():
    return build_vertical_levels(
        stretch_z=1, nk=75, dz_m=100.0, ztop_m=15000.0, str_bot_m=200.0,
        str_top_m=7000.0, dz_bot_m=20.0, dz_top_m=320.0,
    )


def child_deck():
    """The shipped 5 m child: nk1=34, nk2=44, nk3=24 under param.F:7828-7877."""

    return build_vertical_levels(
        stretch_z=1, nk=102, dz_m=100.0, ztop_m=15000.0, str_bot_m=170.0,
        str_top_m=7320.0, dz_bot_m=5.0, dz_top_m=320.0,
    )


def small_parent():
    """nk1=3, nk2=4, nk3=5: 20 m at the ground, 200 m aloft, top 1500 m."""

    return build_vertical_levels(
        stretch_z=1, nk=12, dz_m=100.0, ztop_m=1500.0, str_bot_m=60.0,
        str_top_m=500.0, dz_bot_m=20.0, dz_top_m=200.0,
    )


def small_child():
    """nk1=18, nk2=4, nk3=5: 5 m at the ground under the same top."""

    return build_vertical_levels(
        stretch_z=1, nk=27, dz_m=100.0, ztop_m=1500.0, str_bot_m=90.0,
        str_top_m=500.0, dz_bot_m=5.0, dz_top_m=200.0,
    )


# ---------------------------------------------------------------------------
# CM1's search and plane state machine, against a second transliteration.
# ---------------------------------------------------------------------------
def _cm1_literal(old, new, nzold):
    """restart_read.F:564-575 and prep_for_interp, written as CM1 writes them.

    Independent of the module under test: 1-based CM1 indices, explicit
    plane copies, and ``do_interp``'s statement evaluated per column.
    """

    f32 = numpy.float32
    iz, rz = [], []
    kold = 0
    for k in range(1, len(new) + 1):
        while old[kold + 1] < new[k - 1]:
            kold += 1
        iz.append(kold)
        rz.append(f32(f32(new[k - 1] - old[kold]) / f32(old[kold + 1] - old[kold])))
    planes = []
    kold1, kold2, alldone = -1, 0, False
    d1 = d2 = None
    for k in range(1, len(new) + 1):
        if iz[k - 1] + 1 > kold2:
            if iz[k - 1] + 1 <= nzold:
                for kk in range(kold2 + 1, iz[k - 1] + 2):
                    kold1 += 1
                    kold2 += 1
                    d1 = d2
                    d2 = kold2
                if kold1 == 0:
                    d1 = d2
            else:
                if not alldone:
                    d1 = d2
                alldone = True
        planes.append((d1, d2))
    return iz, rz, planes


@pytest.mark.parametrize("pair", ["shipped", "small", "identical"])
@pytest.mark.parametrize("staggering", ["sigma", "sigmaf"])
def test_the_map_is_cm1s_search_and_plane_state_machine(pair, staggering) -> None:
    parent, child = {
        "shipped": (parent_deck(), child_deck()),
        "small": (small_parent(), small_child()),
        "identical": (parent_deck(), parent_deck()),
    }[pair]
    if staggering == "sigma":
        old, new, nzold = parent.sigma, child.sigma[1:child.nk + 1], parent.nk
    else:
        old, new, nzold = parent.sigmaf, child.sigmaf[1:child.nk + 2], parent.nk + 1
    vmap = build_cm1_vertical_map(old, new, nzold=nzold)
    iz, rz, planes = _cm1_literal(old, new, nzold)
    assert list(vmap.iz) == iz
    assert numpy.asarray(rz, dtype=numpy.float32).tobytes() == vmap.fraction.tobytes()
    assert [(int(a) + 1, int(b) + 1) for a, b in zip(vmap.lower, vmap.upper)] == planes


def test_below_the_first_parent_level_cm1_uses_the_ghost_fraction_on_duplicated_planes() -> None:
    """The 2.5 m and 7.5 m child levels sit under the parent's 10 m level.

    CM1's ``rz`` there comes from ``sigma0(0) = -10`` (param.F:8172), not zero,
    and both planes hold parent level 1 (``prep_for_interp`` ``kold1 == 0``).
    """

    vmap = plan_vertical_refinement(parent_deck(), child_deck()).mass
    assert float(child_deck().sigma[1]) == 2.5 and float(child_deck().sigma[2]) == 7.5
    assert (int(vmap.lower[0]), int(vmap.upper[0])) == (0, 0)
    assert (int(vmap.lower[1]), int(vmap.upper[1])) == (0, 0)
    assert float(vmap.fraction[0]) == 0.625
    assert float(vmap.fraction[1]) == 0.875
    assert bool(vmap.below_support[0]) and bool(vmap.below_support[1])
    # A constant parent column stays that constant through the duplicated planes.
    column = numpy.full((75, 1, 1), numpy.float32(301.25), dtype=numpy.float32)
    out = cm1_vertical_interpolate(column, vmap)
    assert (out == numpy.float32(301.25)).all()


def test_identical_decks_interpolate_to_the_same_words() -> None:
    """Every new level matches an old one exactly, so rz = 1 on datk2 (or rz from the ghost on duplicated planes)."""

    deck = parent_deck()
    vmap = build_cm1_vertical_map(deck.sigma, deck.sigma[1:deck.nk + 1], nzold=deck.nk)
    rng = numpy.random.default_rng(3)
    volume = rng.uniform(-50.0, 50.0, size=(deck.nk, 4, 5)).astype(numpy.float32)
    assert cm1_vertical_interpolate(volume, vmap).tobytes() == volume.tobytes()
    face = build_cm1_vertical_map(deck.sigmaf, deck.sigmaf[1:deck.nk + 2], nzold=deck.nk + 1)
    wvol = rng.uniform(-50.0, 50.0, size=(deck.nk + 1, 4, 5)).astype(numpy.float32)
    assert cm1_vertical_interpolate(wvol, face).tobytes() == wvol.tobytes()


# ---------------------------------------------------------------------------
# Admission.
# ---------------------------------------------------------------------------
def test_the_shipped_five_metre_child_is_admitted() -> None:
    parent, child = parent_deck(), child_deck()
    assert vertical_refinement_defect(parent, child) is None
    assert float(child.zf[2]) - float(child.zf[1]) == 5.0
    plan = plan_vertical_refinement(parent, child)
    assert not plan.identical
    assert plan.record()["child_lowest_layer_m"] == 5.0


def test_a_different_model_top_is_refused_by_name() -> None:
    other = build_vertical_levels(
        stretch_z=1, nk=103, dz_m=100.0, ztop_m=15320.0, str_bot_m=170.0,
        str_top_m=7320.0, dz_bot_m=5.0, dz_top_m=320.0,
    )
    defect = vertical_refinement_defect(parent_deck(), other)
    assert defect is not None and "model top" in defect


def test_a_child_coarser_than_its_parent_anywhere_is_refused_by_name() -> None:
    # 25 levels in the stretch region reach 25.7 m layers below 200 m, where
    # the parent is 20 m.
    coarse = build_vertical_levels(
        stretch_z=1, nk=102, dz_m=100.0, ztop_m=15000.0, str_bot_m=125.0,
        str_top_m=10200.0, dz_bot_m=5.0, dz_top_m=320.0,
    )
    defect = vertical_refinement_defect(parent_deck(), coarse)
    assert defect is not None and "coarser than its parent" in defect


def test_refusal_goes_through_the_error_contract() -> None:
    from cm1core.errors import CM1Refusal
    from cm1core.nest_vertical import require_vertical_refinement

    coarse = build_vertical_levels(
        stretch_z=1, nk=102, dz_m=100.0, ztop_m=15000.0, str_bot_m=125.0,
        str_top_m=10200.0, dz_bot_m=5.0, dz_top_m=320.0,
    )
    with pytest.raises(CM1Refusal) as excinfo:
        require_vertical_refinement(
            parent_deck(), coarse, code="CM1-NEST-VERTICAL-REFINEMENT",
            gate="cartesian-runtime", option="nest.child_str_bot_m",
        )
    assert excinfo.value.detail.code == "CM1-NEST-VERTICAL-REFINEMENT"


# ---------------------------------------------------------------------------
# A smooth analytic column to a known accuracy.
# ---------------------------------------------------------------------------
def test_a_smooth_column_is_interpolated_within_the_linear_bound() -> None:
    """theta(z) = 300 + 4 sin(2 pi z / 3000) + 0.003 z, on the shipped pair.

    Linear interpolation between levels h apart errs by at most
    h^2/8 * max|f''|; with CM1's constant extension below the first parent
    level the error there is bounded by |f'| times the distance.  Both bounds
    are evaluated per child level and the float32 words must honour them.
    """

    parent, child = parent_deck(), child_deck()
    plan = plan_vertical_refinement(parent, child)
    amplitude, wavelength = 4.0, 3000.0
    k2 = (2.0 * math.pi / wavelength) ** 2

    def f(z):
        return 300.0 + amplitude * numpy.sin(2.0 * math.pi * z / wavelength) + 0.003 * z

    pz = numpy.asarray(parent.zh[1:parent.nk + 1], dtype=numpy.float64)
    cz = numpy.asarray(child.zh[1:child.nk + 1], dtype=numpy.float64)
    column = f(pz).astype(numpy.float32).reshape(parent.nk, 1, 1)
    out = cm1_vertical_interpolate(column, plan.mass)[:, 0, 0].astype(numpy.float64)
    exact = f(cz)
    worst = 0.0
    for k in range(child.nk):
        if cz[k] <= pz[0]:
            bound = (amplitude * 2.0 * math.pi / wavelength + 0.003) * (pz[0] - cz[k])
        elif cz[k] >= pz[-1]:
            bound = (amplitude * 2.0 * math.pi / wavelength + 0.003) * (cz[k] - pz[-1])
        else:
            hi = int(numpy.searchsorted(pz, cz[k]))
            h = pz[hi] - pz[hi - 1]
            bound = h * h / 8.0 * amplitude * k2
        error = abs(out[k] - exact[k])
        # One float32 ulp of 300 K is 3.05e-5 K.
        assert error <= bound + 6.2e-5, (k, cz[k], error, bound)
        worst = max(worst, error)
    assert worst < 0.12  # the 320 m upper layers: 320^2/8 * 4 * k2 = 0.22 K bound


# ---------------------------------------------------------------------------
# Kernel text against CPU references, and the identical-deck composition.
# ---------------------------------------------------------------------------
RATIO = 3
PARENT_NX, PARENT_NY = 14, 12
PARENT_I0, PARENT_J0 = 3, 4
SPAN_X, SPAN_Y = 5, 4
CHILD_NX, CHILD_NY = SPAN_X * RATIO, SPAN_Y * RATIO
WIDTH = 5
PAD = (1, 3, 3)
STAGGERS = ("u", "v", "w", "mass")


class _Geometry:
    refinement_ratio = RATIO
    parent_i0 = PARENT_I0
    parent_j0 = PARENT_J0
    parent_span_x = SPAN_X
    parent_span_y = SPAN_Y
    nx = CHILD_NX
    ny = CHILD_NY

    @staticmethod
    def sub_cell_offsets():
        return tuple((2 * m + 1 - RATIO) / (2 * RATIO) for m in range(RATIO))


@pytest.fixture(scope="module")
def vharness():
    from tools.oracle.nest_vertical_oracle import open_harness

    return open_harness()


@pytest.fixture(scope="module")
def lharness():
    from tools.oracle.nest_live_oracle import open_harness

    return open_harness()


def _padded(rng, lattice):
    nzl, nyl, nxl = lattice
    owner = rng.uniform(-3.0, 3.0, size=(nzl + 2, nyl + 7, nxl + 8)).astype(numpy.float32)
    return owner, owner[1:1 + nzl, 3:3 + nyl, 3:3 + nxl]


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
@pytest.mark.parametrize("pair", ["small", "identical"])
def test_the_vinterp_kernel_text_is_the_reference(vharness, stagger, pair) -> None:
    parent, child = (small_parent(), small_child()) if pair == "small" else (small_parent(), small_parent())
    if pair == "small":
        vmap = plan_vertical_refinement(parent, child).map_for(stagger)
    elif stagger == "w":
        vmap = build_cm1_vertical_map(parent.sigmaf, parent.sigmaf[1:parent.nk + 2], nzold=parent.nk + 1)
    else:
        vmap = build_cm1_vertical_map(parent.sigma, parent.sigma[1:parent.nk + 1], nzold=parent.nk)
    rng = numpy.random.default_rng(STAGGERS.index(stagger) + 17)
    pl = lattice_shape(stagger, parent.nk, PARENT_NY, PARENT_NX)
    owner, physical = _padded(rng, pl)
    window = VerticalWindow.for_stagger(_Geometry, stagger)
    expected = reference_vinterp(physical, vmap, window)
    out_shape = (vmap.levels, window.nj, window.ni)
    params = [owner.shape[1], owner.shape[2], *PAD, window.j0, window.i0, *out_shape]
    for order in ("linear", "shuffled"):
        got = vharness.vinterp(
            parent=owner, lower=vmap.lower, upper=vmap.upper, rz=vmap.fraction,
            out_shape=out_shape, params=params, order=order,
        )
        assert got.tobytes() == expected.tobytes(), (stagger, pair, order)


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
def test_identical_decks_through_the_refined_capture_are_the_copy_path_bit_for_bit(
    vharness, lharness, stagger
) -> None:
    """vinterp (kernel) then capture (kernel) on a window == capture on the parent."""

    deck = small_parent()
    if stagger == "w":
        vmap = build_cm1_vertical_map(deck.sigmaf, deck.sigmaf[1:deck.nk + 2], nzold=deck.nk + 1)
    else:
        vmap = build_cm1_vertical_map(deck.sigma, deck.sigma[1:deck.nk + 1], nzold=deck.nk)
    rng = numpy.random.default_rng(40 + STAGGERS.index(stagger))
    pl = lattice_shape(stagger, deck.nk, PARENT_NY, PARENT_NX)
    cl = lattice_shape(stagger, deck.nk, CHILD_NY, CHILD_NX)
    owner, physical = _padded(rng, pl)
    child_owner, child = _padded(rng, cl)
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    window = VerticalWindow.for_stagger(_Geometry, stagger)
    out_shape = (vmap.levels, window.nj, window.ni)
    column = vharness.vinterp(
        parent=owner, lower=vmap.lower, upper=vmap.upper, rz=vmap.fraction,
        out_shape=out_shape, params=[owner.shape[1], owner.shape[2], *PAD, window.j0, window.i0, *out_shape],
    )
    assert column.tobytes() == numpy.ascontiguousarray(
        physical[:, window.j0:window.j0 + window.nj, window.i0:window.i0 + window.ni]
    ).tobytes()
    for side in ("full", "west", "north"):
        from cm1core.nest_live_kernels import side_shape

        shape = side_shape(side, cl, WIDTH)
        code = {"full": 0, "west": 1, "north": 4}[side]
        direct = [
            owner.shape[1], owner.shape[2], *PAD, pl[1], pl[2],
            child_owner.shape[1], child_owner.shape[2], *PAD, *cl, code, WIDTH,
            RATIO, PARENT_I0, PARENT_J0, x_face, y_face, 1,
        ]
        windowed = [
            column.shape[1], column.shape[2], 0, 0, 0, window.nj, window.ni,
            child_owner.shape[1], child_owner.shape[2], *PAD, *cl, code, WIDTH,
            RATIO, 1, 1, x_face, y_face, 1,
        ]
        a = lharness.capture(parent=owner, child=child_owner, value_shape=shape, params=direct, rdt=2.0)
        b = lharness.capture(parent=column, child=child_owner, value_shape=shape, params=windowed, rdt=2.0)
        assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes(), (stagger, side)
        # And the host references agree with each other the same way.
        ra = reference_capture(
            physical, child_lattice=cl, child_volume=child, side=side, width=WIDTH, ratio=RATIO,
            parent_i0=PARENT_I0, parent_j0=PARENT_J0, stagger=stagger, parent_dt_s=0.5,
        )
        rb = reference_capture_refined(
            physical, vmap=vmap, window=window, child_lattice=cl, child_volume=child,
            side=side, width=WIDTH, ratio=RATIO, stagger=stagger, parent_dt_s=0.5,
        )
        assert ra[0].tobytes() == rb[0].tobytes() and ra[1].tobytes() == rb[1].tobytes()
        assert a[1].tobytes() == ra[1].tobytes()


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
def test_the_refined_capture_kernels_are_the_reference(vharness, lharness, stagger) -> None:
    parent, child_levels = small_parent(), small_child()
    plan = plan_vertical_refinement(parent, child_levels)
    vmap = plan.map_for(stagger)
    rng = numpy.random.default_rng(60 + STAGGERS.index(stagger))
    pl = lattice_shape(stagger, parent.nk, PARENT_NY, PARENT_NX)
    cl = lattice_shape(stagger, child_levels.nk, CHILD_NY, CHILD_NX)
    owner, physical = _padded(rng, pl)
    child_owner, child = _padded(rng, cl)
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    window = VerticalWindow.for_stagger(_Geometry, stagger)
    out_shape = (vmap.levels, window.nj, window.ni)
    column = vharness.vinterp(
        parent=owner, lower=vmap.lower, upper=vmap.upper, rz=vmap.fraction, out_shape=out_shape,
        params=[owner.shape[1], owner.shape[2], *PAD, window.j0, window.i0, *out_shape],
    )
    from cm1core.nest_live_kernels import side_shape

    for side, code in (("full", 0), ("east", 2), ("south", 3)):
        shape = side_shape(side, cl, WIDTH)
        params = [
            column.shape[1], column.shape[2], 0, 0, 0, window.nj, window.ni,
            child_owner.shape[1], child_owner.shape[2], *PAD, *cl, code, WIDTH,
            RATIO, 1, 1, x_face, y_face, 1,
        ]
        got = lharness.capture(parent=column, child=child_owner, value_shape=shape, params=params, rdt=2.0)
        want = reference_capture_refined(
            physical, vmap=vmap, window=window, child_lattice=cl, child_volume=child,
            side=side, width=WIDTH, ratio=RATIO, stagger=stagger, parent_dt_s=0.5,
        )
        assert got[0].tobytes() == want[0].tobytes() and got[1].tobytes() == want[1].tobytes()


def _density(levels, stagger):
    z = numpy.asarray(
        (levels.zf[1:levels.nk + 2] if stagger == "w" else levels.zh[1:levels.nk + 1]),
        dtype=numpy.float64,
    )
    return 1.16 * numpy.exp(-z / 8500.0)


def _restrict_params(child_owner, parent_owner, cl, nz_parent, stagger, count):
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    return [
        child_owner.shape[1], child_owner.shape[2], *PAD,
        parent_owner.shape[1], parent_owner.shape[2], *PAD,
        nz_parent, cl[1], cl[2], SPAN_Y + y_face, SPAN_X + x_face,
        RATIO, 2, PARENT_I0, PARENT_J0, x_face, y_face, count,
    ]


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
def test_the_vertical_restriction_kernel_text_is_the_reference(vharness, stagger) -> None:
    parent, child_levels = small_parent(), small_child()
    plan = plan_vertical_refinement(parent, child_levels)
    weights = feedback_weights(plan, stagger=stagger, child_density=_density(child_levels, stagger))
    rng = numpy.random.default_rng(80 + STAGGERS.index(stagger))
    pl = lattice_shape(stagger, parent.nk, PARENT_NY, PARENT_NX)
    cl = lattice_shape(stagger, child_levels.nk, CHILD_NY, CHILD_NX)
    child_owner, child = _padded(rng, cl)
    parent_owner, physical = _padded(rng, pl)
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    count = pl[0] * (SPAN_Y + y_face) * (SPAN_X + x_face)
    expected = reference_restrict_vertical(
        child, physical, weights=weights, child_lattice=cl, span_x=SPAN_X, span_y=SPAN_Y,
        ratio=RATIO, relax_zone=2, parent_i0=PARENT_I0, parent_j0=PARENT_J0, stagger=stagger,
    )
    for order in ("linear", "shuffled"):
        got = vharness.restrict_vertical(
            child=child_owner, parent=parent_owner, offsets=weights.offsets, index=weights.index,
            weight=weights.weight, params=_restrict_params(child_owner, parent_owner, cl, pl[0], stagger, count),
            order=order,
        )
        assert got[1:1 + pl[0], 3:3 + pl[1], 3:3 + pl[2]].tobytes() == expected.tobytes()
    assert not numpy.array_equal(expected, physical)


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
def test_identical_decks_through_the_vertical_restriction_are_the_existing_feedback(
    vharness, lharness, stagger
) -> None:
    deck = small_parent()
    plan = plan_vertical_refinement(deck, deck)
    # An identical plan has no maps, but its weights are still defined.
    weights = feedback_weights(
        type(plan)(deck, deck, False, None, None), stagger=stagger,
        child_density=_density(deck, stagger),
    )
    assert (weights.weight == 1.0).all() and (numpy.diff(weights.offsets) == 1).all()
    rng = numpy.random.default_rng(90 + STAGGERS.index(stagger))
    pl = lattice_shape(stagger, deck.nk, PARENT_NY, PARENT_NX)
    cl = lattice_shape(stagger, deck.nk, CHILD_NY, CHILD_NX)
    child_owner, child = _padded(rng, cl)
    parent_owner, _ = _padded(rng, pl)
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    count = pl[0] * (SPAN_Y + y_face) * (SPAN_X + x_face)
    params = _restrict_params(child_owner, parent_owner, cl, pl[0], stagger, count)
    a = lharness.restrict(child=child_owner, parent=parent_owner, params=params)
    b = vharness.restrict_vertical(
        child=child_owner, parent=parent_owner, offsets=weights.offsets, index=weights.index,
        weight=weights.weight, params=params,
    )
    assert a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# Conservation of column mass and heat through feedback.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pair", ["shipped", "small"])
def test_feedback_conserves_the_childs_column_mass_and_heat(pair) -> None:
    parent, child = (parent_deck(), child_deck()) if pair == "shipped" else (small_parent(), small_child())
    plan = plan_vertical_refinement(parent, child)
    rho = _density(child, "mass")
    weights = feedback_weights(plan, stagger="mass", child_density=rho)
    cf = numpy.asarray(child.zf[1:child.nk + 2], dtype=numpy.float32).astype(numpy.float64)
    dz = numpy.diff(cf)
    # The parent measures tile the child's: sum_k M_k = sum_m rho_m dz_m.
    assert weights.mass.sum() == pytest.approx(float((rho * dz).sum()), rel=1e-14)
    rng = numpy.random.default_rng(5)
    theta = 300.0 + rng.uniform(-8.0, 8.0, size=child.nk)
    qv = numpy.abs(rng.normal(0.01, 0.004, size=child.nk))
    for carrier in (theta, qv):
        restricted = numpy.zeros(parent.nk)
        for k in range(parent.nk):
            total = 0.0
            for e in range(int(weights.offsets[k]), int(weights.offsets[k + 1])):
                total = total + float(weights.weight[e]) * float(carrier[int(weights.index[e])])
            restricted[k] = total
        child_integral = float((rho * dz * carrier).sum())
        parent_integral = float((weights.mass * restricted).sum())
        assert parent_integral == pytest.approx(child_integral, rel=1e-13)
    # A column that is constant stays constant, and a limited column stays within its bounds.
    for k in range(parent.nk):
        row = weights.weight[int(weights.offsets[k]):int(weights.offsets[k + 1])]
        assert (row > 0.0).all() and float(row.sum()) == pytest.approx(1.0, abs=4e-16)


def test_feedback_through_the_restriction_reference_conserves_a_block_column(pair="small") -> None:
    """The full 3-D reference, float32 in and out, over one covered block."""

    parent, child_levels = small_parent(), small_child()
    plan = plan_vertical_refinement(parent, child_levels)
    rho = _density(child_levels, "mass")
    weights = feedback_weights(plan, stagger="mass", child_density=rho)
    rng = numpy.random.default_rng(12)
    cl = lattice_shape("mass", child_levels.nk, CHILD_NY, CHILD_NX)
    pl = lattice_shape("mass", parent.nk, PARENT_NY, PARENT_NX)
    child = (300.0 + rng.uniform(-5.0, 5.0, size=cl)).astype(numpy.float32)
    out = reference_restrict_vertical(
        child, numpy.zeros(pl, dtype=numpy.float32), weights=weights, child_lattice=cl,
        span_x=SPAN_X, span_y=SPAN_Y, ratio=RATIO, relax_zone=0, parent_i0=PARENT_I0,
        parent_j0=PARENT_J0, stagger="mass",
    )
    cf = numpy.asarray(child_levels.zf[1:child_levels.nk + 2], dtype=numpy.float32).astype(numpy.float64)
    dz = numpy.diff(cf)
    child_heat = float((child.astype(numpy.float64) * (rho * dz)[:, None, None]).sum()) / RATIO**2
    parent_heat = float(
        (out[:, PARENT_J0:PARENT_J0 + SPAN_Y, PARENT_I0:PARENT_I0 + SPAN_X].astype(numpy.float64)
         * weights.mass[:, None, None]).sum()
    )
    # float32 narrowing of each parent word: one ulp of 300 K is 3e-5 K.
    assert parent_heat == pytest.approx(child_heat, rel=2e-7)
    # And the one-deck restriction it generalizes does the same.
    same = reference_restrict(
        child[: parent.nk], numpy.zeros(pl, dtype=numpy.float32),
        child_lattice=(parent.nk, cl[1], cl[2]), span_x=SPAN_X, span_y=SPAN_Y, ratio=RATIO,
        relax_zone=0, parent_i0=PARENT_I0, parent_j0=PARENT_J0, stagger="mass",
    )
    assert same.shape == out.shape


def test_w_feedback_copies_the_boundary_faces_and_weights_dual_cells() -> None:
    parent, child = parent_deck(), child_deck()
    plan = plan_vertical_refinement(parent, child)
    weights = feedback_weights(plan, stagger="w", child_density=_density(child, "w"))
    assert weights.parent_levels == parent.nk + 1
    assert list(weights.index[weights.offsets[0]:weights.offsets[1]]) == [0]
    assert list(weights.index[weights.offsets[-2]:weights.offsets[-1]]) == [child.nk]
    for k in range(1, parent.nk):
        members = weights.index[weights.offsets[k]:weights.offsets[k + 1]]
        assert 0 not in members and child.nk not in members


# ---------------------------------------------------------------------------
# The host downscale path.
# ---------------------------------------------------------------------------
def test_the_host_downscale_through_a_finer_deck_matches_the_window_references() -> None:
    from cm1core.nest_geometry import NestGeometry
    from cm1core.nest_vertical import downscale_volume_refined

    geometry = NestGeometry(
        parent_nx=20, parent_ny=18, parent_dx_m=250.0, parent_dy_m=250.0,
        refinement_ratio=5, time_refinement_ratio=5, parent_i0=6, parent_j0=5,
        parent_span_x=8, parent_span_y=7, spec_zone=1, relax_zone=4, feedback=0,
    )
    parent, child_levels = small_parent(), small_child()
    plan = plan_vertical_refinement(parent, child_levels)
    rng = numpy.random.default_rng(21)
    for vertical, xl, yl, stagger in (
        ("mass", "mass", "mass", "mass"), ("mass", "face", "mass", "u"),
        ("mass", "mass", "face", "v"), ("w", "mass", "mass", "w"),
    ):
        x_face, y_face, z_extra = LATTICE_OF_STAGGER[stagger]
        volume = rng.uniform(-10.0, 10.0, size=(parent.nk + z_extra, 18 + y_face, 20 + x_face)).astype(numpy.float32)
        got = downscale_volume_refined(volume, geometry, plan, x_lattice=xl, y_lattice=yl, vertical=vertical)
        cl = lattice_shape(stagger, child_levels.nk, geometry.ny, geometry.nx)
        assert got.shape == cl
        window = VerticalWindow.for_stagger(geometry, stagger)
        want, _ = reference_capture_refined(
            volume, vmap=plan.map_for(vertical), window=window, child_lattice=cl,
            child_volume=None, side="full", width=5, ratio=5, stagger=stagger, parent_dt_s=1.0,
        )
        assert got.astype(numpy.float32).tobytes() == want.tobytes()


# ---------------------------------------------------------------------------
# Configuration: the child's own deck through [nest], refused by name.
# ---------------------------------------------------------------------------
REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
LIVE_DECK = REPO / "tests" / "data" / "decks" / "nest" / "nest-100m-parent-20m-child-5m-lowest.toml"
DOWNSCALE_CHILD = REPO / "tests" / "data" / "decks" / "nest" / "child-20m-6km-5m-lowest.toml"


def _deck_with(**replacements):
    import tomllib

    from cm1core.config import parse_mapping

    text = LIVE_DECK.read_text()
    for key, value in replacements.items():
        lines = [line for line in text.splitlines() if not line.startswith(f"{key} =")]
        if value is not None:
            index = lines.index("[runtime]")
            lines.insert(index - 1, f"{key} = {value}")
        text = "\n".join(lines) + "\n"
    return parse_mapping(tomllib.loads(text))


def test_the_shipped_live_deck_binds_a_102_level_child_on_a_five_metre_layer() -> None:
    import tomllib

    from cm1core.cartesian_door import require_nest
    from cm1core.config import nest_child_vertical_coordinate, parse_mapping

    config = parse_mapping(tomllib.loads(LIVE_DECK.read_text()))
    levels = nest_child_vertical_coordinate(config)
    assert levels.nk == 102 and float(levels.zf[2]) == 5.0
    assert levels.sha256 == child_deck().sha256
    assert config.canonical_dict()["nest"]["child_vertical"]["nz"] == 102
    binding = require_nest(config)
    assert binding.child.grid.nz == 102
    assert binding.child.grid.vertical.sha256 == child_deck().sha256


def test_a_nest_without_child_keys_projects_no_child_vertical_key() -> None:
    config = _deck_with(child_nz=None, child_stretch_z=None, child_str_bot_m=None,
                        child_str_top_m=None, child_dz_bot_m=None, child_dz_top_m=None)
    assert config.nest.child_vertical is None
    assert "child_vertical" not in config.canonical_dict()["nest"]
    from cm1core.cartesian_door import require_nest

    binding = require_nest(config)
    assert binding.child.grid.nz == 75
    assert binding.child.grid.vertical.sha256 == parent_deck().sha256


def test_a_half_declared_child_deck_is_refused_by_name() -> None:
    from cm1core.errors import CM1Refusal

    with pytest.raises(CM1Refusal) as excinfo:
        _deck_with(child_stretch_z=None)
    assert excinfo.value.detail.code == "CM1-CFG-NEST-CHILD-VERTICAL"
    assert excinfo.value.detail.option == "nest.child_stretch_z"


def test_a_child_coarser_than_the_parent_is_refused_by_name() -> None:
    from cm1core.errors import CM1Refusal

    with pytest.raises(CM1Refusal) as excinfo:
        _deck_with(child_str_bot_m="125.0", child_str_top_m="10200.0")
    assert excinfo.value.detail.code == "CM1-CFG-NEST-VERTICAL-REFINEMENT"
    assert "coarser than its parent" in excinfo.value.detail.reason


def test_an_infeasible_child_deck_is_refused_on_the_child_key() -> None:
    from cm1core.errors import CM1Refusal

    with pytest.raises(CM1Refusal) as excinfo:
        _deck_with(child_str_bot_m="160.0")
    assert excinfo.value.detail.code == "CM1-CFG-NEST-CHILD-VERTICAL-DECK"
    assert (excinfo.value.detail.option or "").startswith("nest.child_")


def test_a_uniform_child_deck_is_refused_by_value() -> None:
    from cm1core.errors import CM1Refusal

    with pytest.raises(CM1Refusal):
        _deck_with(child_stretch_z="0")


def test_the_downscale_child_deck_is_admitted_against_its_window() -> None:
    import tomllib

    from cm1core.config import parse_mapping, require_vertical_coordinate

    child = require_vertical_coordinate(parse_mapping(tomllib.loads(DOWNSCALE_CHILD.read_text())))
    window = parse_mapping(tomllib.loads((REPO / "tests" / "data" / "decks" / "nest" / "downscale-window-100m-to-20m.toml").read_text()))
    parent = require_vertical_coordinate(window)
    plan = plan_vertical_refinement(parent, child)
    assert not plan.identical and child.sha256 == child_deck().sha256


# ---------------------------------------------------------------------------
# The one-way downscale tool's transfer, on synthetic arenas (no Rust needed).
# ---------------------------------------------------------------------------
def _write_arena(path, entries):
    offset = 0
    table = {}
    with open(path, "wb") as handle:
        for name, (stagger, array) in entries.items():
            array = numpy.ascontiguousarray(array, dtype="<f4")
            handle.write(array.tobytes())
            table[name] = {"stagger": stagger, "shape": tuple(array.shape), "offset": offset}
            offset += array.nbytes
    return table


@pytest.mark.parametrize("refined", [False, True])
def test_the_downscale_tool_transfers_every_carrier_through_the_right_path(tmp_path, refined) -> None:
    import sys

    sys.path.insert(0, str(REPO))
    from cm1core.nest_downscale import downscale_level, downscale_level_staggered
    from cm1core.nest_geometry import NestGeometry
    from cm1core.nest_vertical import downscale_volume_refined
    from tools.nest_downscale_init import transfer_carriers

    geometry = NestGeometry(
        parent_nx=20, parent_ny=18, parent_dx_m=250.0, parent_dy_m=250.0,
        refinement_ratio=5, time_refinement_ratio=5, parent_i0=6, parent_j0=5,
        parent_span_x=8, parent_span_y=7, spec_zone=1, relax_zone=4, feedback=0,
    )
    parent = small_parent()
    child_levels = small_child() if refined else parent
    plan = plan_vertical_refinement(parent, child_levels)
    pz, cz = parent.nk, child_levels.nk
    nx, ny = geometry.nx, geometry.ny
    rng = numpy.random.default_rng(8)

    def field(*shape):
        return rng.uniform(-5.0, 5.0, size=shape).astype(numpy.float32)

    parent_fields = {
        "u3d": ("u", field(pz, 18, 21)),
        "w3d": ("w", field(pz + 1, 18, 20)),
        "th3d": ("mass", field(pz, 18, 20)),
        "q3d": ("mass", numpy.abs(field(2, pz, 18, 20))),
        "closure.tkea": ("w", numpy.abs(field(pz + 1, 18, 20))),
    }
    parent_table = _write_arena(tmp_path / "parent.f32", parent_fields)
    tke_sentinel = numpy.full((cz + 1, ny, nx), 0.125, dtype=numpy.float32)
    child_table = _write_arena(tmp_path / "child.f32", {
        "padded.ua": ("padded-ua", numpy.zeros((cz + 2, ny + 6, nx + 7), dtype=numpy.float32)),
        "padded.wa": ("padded-wa", numpy.zeros((cz + 3, ny + 6, nx + 6), dtype=numpy.float32)),
        "fields.tha": ("mass", numpy.zeros((cz, ny, nx), dtype=numpy.float32)),
        "fields.qa": ("mass", numpy.zeros((2, cz, ny, nx), dtype=numpy.float32)),
        "tke.tkea": ("w", tke_sentinel),
    })
    written, skipped = transfer_carriers(
        parent_table=parent_table, parent_arena=tmp_path / "parent.f32",
        child_table=child_table, child_arena=tmp_path / "child.f32",
        geometry=geometry, parent_nz=pz, refinement=plan,
    )
    arena = numpy.fromfile(tmp_path / "child.f32", dtype="<f4")

    def child(name):
        entry = child_table[name]
        count = int(numpy.prod(entry["shape"]))
        return arena[entry["offset"] // 4: entry["offset"] // 4 + count].reshape(entry["shape"])

    def expect(name, xl, yl, vertical):
        volume = parent_fields[name][1]
        if refined:
            return downscale_volume_refined(volume, geometry, plan, x_lattice=xl, y_lattice=yl,
                                            vertical=vertical).astype(numpy.float32)
        levels = [downscale_level(volume[k], geometry) if (xl, yl) == ("mass", "mass")
                  else downscale_level_staggered(volume[k], geometry, x_lattice=xl, y_lattice=yl)
                  for k in range(volume.shape[0])]
        return numpy.stack(levels).astype(numpy.float32)

    assert child("padded.ua")[1:1 + cz, 3:3 + ny, 3:3 + nx + 1].tobytes() == expect("u3d", "face", "mass", "mass").tobytes()
    assert child("padded.wa")[1:2 + cz, 3:3 + ny, 3:3 + nx].tobytes() == expect("w3d", "mass", "mass", "w").tobytes()
    assert child("fields.tha").tobytes() == expect("th3d", "mass", "mass", "mass").tobytes()
    q = parent_fields["q3d"][1]
    for s in range(2):
        volume = q[s]
        if refined:
            want = downscale_volume_refined(volume, geometry, plan, x_lattice="mass", y_lattice="mass",
                                            vertical="mass").astype(numpy.float32)
        else:
            want = numpy.stack([downscale_level(volume[k], geometry) for k in range(pz)]).astype(numpy.float32)
        assert child("fields.qa")[s].tobytes() == want.tobytes()
        assert float(child("fields.qa")[s].min()) >= 0.0
    if refined:
        assert child("tke.tkea").tobytes() == tke_sentinel.tobytes()
        assert [row["parent_carrier"] for row in skipped] == ["closure.tkea"]
        assert {row["vertical"] for row in written} == {"sigma", "sigmaf"}
    else:
        assert child("tke.tkea").tobytes() == expect("closure.tkea", "mass", "mass", "w").tobytes()
        assert skipped == [] and {row["vertical"] for row in written} == {"copied"}
