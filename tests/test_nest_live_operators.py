"""The live nest's device operators, held to their CPU references word for word.

No test here opens a CUDA context.  The packaged kernel text is compiled with
``g++`` (``tools/oracle/nest_live_oracle.py``) and its raw float32 output is
compared against the NumPy references in :mod:`cm1core.nest_live_kernels`,
over every lattice the nest forces (``u`` face/centre, ``v`` centre/face,
``w`` and the scalars centre/centre), every side, padded owners with non-zero
origins, and a shuffled thread order.

The second half is the evidence ``records/CM1-NEST-SEAM.md`` section 5.1 rests
on for reusing the staggered reconstruction as a re-imposed boundary tendency:
the flux through every parent face is carried by the child's coincident
sub-faces exactly (in the float64 reconstruction), so the block-integrated
divergence of the child equals the parent's, and a horizontally linear field
is reproduced exactly.
"""

from __future__ import annotations

import shutil

import pytest

numpy = pytest.importorskip("numpy")

from cm1core.nest_live_kernels import (  # noqa: E402
    LATTICE_OF_STAGGER,
    lattice_shape,
    owned_band_mask,
    reference_capture,
    reference_restrict,
    reference_spec_apply,
    side_shape,
)

requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")

RATIO = 3
PARENT_NX, PARENT_NY, NZ = 14, 12, 3
PARENT_I0, PARENT_J0 = 3, 4
SPAN_X, SPAN_Y = 5, 4
CHILD_NX, CHILD_NY = SPAN_X * RATIO, SPAN_Y * RATIO
WIDTH, SPEC, RELAX = 5, 1, 4
PAD = (1, 3, 3)
STAGGERS = ("u", "v", "w", "mass")
SIDES = ("full", "west", "east", "south", "north")


@pytest.fixture(scope="module")
def harness():
    from tools.oracle.nest_live_oracle import open_harness

    return open_harness()


def _rng(seed: int):
    return numpy.random.default_rng(seed)


def _padded(rng, lattice):
    nzl, nyl, nxl = lattice
    owner = rng.uniform(-3.0, 3.0, size=(nzl + 2, nyl + 7, nxl + 8)).astype(numpy.float32)
    return owner, owner[1:1 + nzl, 3:3 + nyl, 3:3 + nxl]


def _parent_lattice(stagger):
    return lattice_shape(stagger, NZ, PARENT_NY, PARENT_NX)


def _child_lattice(stagger):
    return lattice_shape(stagger, NZ, CHILD_NY, CHILD_NX)


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
@pytest.mark.parametrize("side", SIDES)
def test_the_capture_kernel_text_is_the_reference_on_every_lattice_and_side(
    harness, stagger, side
) -> None:
    rng = _rng(STAGGERS.index(stagger) * 16 + SIDES.index(side))
    pl, cl = _parent_lattice(stagger), _child_lattice(stagger)
    parent_owner, parent = _padded(rng, pl)
    child_owner, child = _padded(rng, cl)
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    shape = side_shape(side, cl, WIDTH)
    want = side != "full"
    params = [
        parent_owner.shape[1], parent_owner.shape[2], *PAD, pl[1], pl[2],
        child_owner.shape[1], child_owner.shape[2], *PAD,
        cl[0], cl[1], cl[2], {"full": 0, "west": 1, "east": 2, "south": 3, "north": 4}[side],
        WIDTH, RATIO, PARENT_I0, PARENT_J0, x_face, y_face, 1 if want else 0,
    ]
    dt = 2.25
    for order in ("linear", "shuffled"):
        value, tend = harness.capture(
            parent=parent_owner, child=child_owner, value_shape=shape,
            params=params, rdt=1.0 / dt, order=order,
        )
        expected_value, expected_tend = reference_capture(
            parent, child_lattice=cl, child_volume=child if want else None,
            side=side, width=WIDTH, ratio=RATIO, parent_i0=PARENT_I0,
            parent_j0=PARENT_J0, stagger=stagger, parent_dt_s=dt,
        )
        assert value.tobytes() == expected_value.tobytes(), (stagger, side, order)
        if want:
            assert tend.tobytes() == expected_tend.tobytes(), (stagger, side, order)


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
@pytest.mark.parametrize("side", ("west", "east", "south", "north"))
@pytest.mark.parametrize("spec_zone", (1, 2))
def test_the_specified_update_is_spec_bdyupdate_on_the_owned_band(
    harness, stagger, side, spec_zone
) -> None:
    rng = _rng(7 + spec_zone)
    cl = _child_lattice(stagger)
    owner, field = _padded(rng, cl)
    tend = rng.uniform(-1.0, 1.0, size=side_shape(side, cl, WIDTH)).astype(numpy.float32)
    band = side_shape(side, cl, spec_zone)
    code = {"west": 1, "east": 2, "south": 3, "north": 4}[side]
    gather = [owner.shape[1], owner.shape[2], *PAD, *cl, code, spec_zone]
    snap = harness.gather(field=owner, snap_shape=band, params=gather)
    mask = owned_band_mask(side, cl, spec_zone)
    count = band[0] * band[1] * band[2]
    dts = 0.0833333358
    # Advance the field inside the band by some other operator first, which is
    # what the acoustic step does between the gather and the apply.
    advanced_owner = owner.copy()
    advanced_owner[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]] += numpy.float32(0.5)
    params = [owner.shape[1], owner.shape[2], *PAD, *cl, code, spec_zone, WIDTH, 0, 0, count]
    got = harness.apply(
        field=advanced_owner, snap=snap, base=numpy.zeros(1, numpy.float32), tend=tend,
        params=params, dt=dts,
    )
    expected = reference_spec_apply(
        advanced_owner[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]], lattice=cl, tend=tend,
        side=side, spec_zone=spec_zone, width=WIDTH, dt=dts,
        snap=numpy.where(mask, snap, numpy.float32(0.0)),
    )
    physical = got[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]]
    assert physical.tobytes() == expected.tobytes()
    halo = got.copy()
    halo[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]] = 0.0
    reference_halo = advanced_owner.copy()
    reference_halo[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]] = 0.0
    assert halo.tobytes() == reference_halo.tobytes(), "a word outside the lattice was written"
    # mode 1: from the stage's time-level-n field.
    base_owner = rng.uniform(-2.0, 2.0, size=owner.shape).astype(numpy.float32)
    params[11] = 1
    got = harness.apply(
        field=advanced_owner, snap=numpy.zeros(1, numpy.float32), base=base_owner,
        tend=tend, params=params, dt=0.25,
    )
    expected = reference_spec_apply(
        advanced_owner[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]], lattice=cl, tend=tend,
        side=side, spec_zone=spec_zone, width=WIDTH, dt=0.25,
        base=base_owner[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]],
    )
    assert got[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]].tobytes() == expected.tobytes()


def test_the_four_owned_bands_cover_the_perimeter_exactly_once() -> None:
    for stagger in STAGGERS:
        cl = _child_lattice(stagger)
        for spec_zone in (1, 2, 3):
            hits = numpy.zeros(cl[1:], dtype=int)
            nzl, nyl, nxl = cl
            for side in ("west", "east", "south", "north"):
                mask = owned_band_mask(side, cl, spec_zone)[0]
                for a in range(mask.shape[0]):
                    for b in range(mask.shape[1]):
                        if not mask[a, b]:
                            continue
                        if side == "south":
                            hits[a, b] += 1
                        elif side == "north":
                            hits[nyl - 1 - a, b] += 1
                        elif side == "west":
                            hits[a, b] += 1
                        else:
                            hits[a, nxl - 1 - b] += 1
            jj, ii = numpy.meshgrid(numpy.arange(nyl), numpy.arange(nxl), indexing="ij")
            distance = numpy.minimum.reduce([jj, nyl - 1 - jj, ii, nxl - 1 - ii])
            assert numpy.array_equal(hits, (distance < spec_zone).astype(int)), (stagger, spec_zone)


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
def test_the_feedback_kernel_text_is_the_reference(harness, stagger) -> None:
    rng = _rng(99)
    cl, pl = _child_lattice(stagger), _parent_lattice(stagger)
    child_owner, child = _padded(rng, cl)
    parent_owner, parent = _padded(rng, pl)
    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    count = cl[0] * (SPAN_Y + y_face) * (SPAN_X + x_face)
    params = [
        child_owner.shape[1], child_owner.shape[2], *PAD,
        parent_owner.shape[1], parent_owner.shape[2], *PAD,
        cl[0], cl[1], cl[2], SPAN_Y + y_face, SPAN_X + x_face,
        RATIO, 2, PARENT_I0, PARENT_J0, x_face, y_face, count,
    ]
    for order in ("linear", "shuffled"):
        got = harness.restrict(child=child_owner, parent=parent_owner, params=params, order=order)
        expected = reference_restrict(
            child, parent, child_lattice=cl, span_x=SPAN_X, span_y=SPAN_Y,
            ratio=RATIO, relax_zone=2, parent_i0=PARENT_I0, parent_j0=PARENT_J0,
            stagger=stagger,
        )
        assert got[1:1 + pl[0], 3:3 + pl[1], 3:3 + pl[2]].tobytes() == expected.tobytes()
        assert not numpy.array_equal(expected, parent), "the feedback wrote nothing"


# ---------------------------------------------------------------------------
# Evidence for the staggered reconstruction as a boundary tendency.
# ---------------------------------------------------------------------------
class _Window:
    def __init__(self, ratio, i0, j0, nx, ny):
        self.refinement_ratio, self.parent_i0, self.parent_j0 = ratio, i0, j0
        self.nx, self.ny = nx, ny

    def sub_cell_offsets(self):
        r = self.refinement_ratio
        return tuple((2 * m + 1 - r) / (2 * r) for m in range(r))


@pytest.mark.parametrize("ratio", (3, 5, 9))
def test_every_parent_face_flux_is_carried_by_its_child_sub_faces(ratio) -> None:
    """The conservation statement a re-imposed normal velocity needs.

    The ``r`` child ``u`` faces coincident with one parent ``u`` face have
    tangential offsets that are exact negations in pairs, and the limiter is one
    scalar, so in the float64 reconstruction their pair-ordered sum is exactly
    ``r`` times the parent face value.  That is true of every coincident face,
    not only the child's outermost one, so the flux through every parent cell
    face inside the window is the parent's.
    """

    from cm1core.nest_downscale import downscale_level_staggered

    rng = _rng(ratio)
    pnx, pny, span = 20, 18, 6
    parent_u = rng.normal(size=(pny, pnx + 1))
    window = _Window(ratio, 5, 6, span * ratio, span * ratio)
    child = downscale_level_staggered(parent_u, window, x_lattice="face", y_lattice="mass")
    for fp in range(span + 1):
        column = fp * ratio
        for jp in range(span):
            rows = [jp * ratio + m for m in range(ratio)]
            total = 0.0
            for m in range(ratio // 2):
                total += child[rows[m], column] + child[rows[ratio - 1 - m], column]
            total += child[rows[ratio // 2], column]
            assert total == ratio * parent_u[6 + jp, 5 + fp] or math_isclose(
                total, ratio * parent_u[6 + jp, 5 + fp]
            )


def math_isclose(a, b):
    return abs(a - b) <= 4 * numpy.finfo(numpy.float64).eps * max(abs(a), abs(b), 1.0)


@pytest.mark.parametrize("ratio", (3, 5))
def test_the_child_block_divergence_integrates_to_the_parent_divergence(ratio) -> None:
    from cm1core.nest_downscale import downscale_level_staggered

    rng = _rng(3 * ratio)
    pnx = pny = 16
    span = 6
    i0 = j0 = 5
    dx = 100.0
    u = rng.normal(size=(pny, pnx + 1))
    v = rng.normal(size=(pny + 1, pnx))
    window = _Window(ratio, i0, j0, span * ratio, span * ratio)
    cu = downscale_level_staggered(u, window, x_lattice="face", y_lattice="mass")
    cv = downscale_level_staggered(v, window, x_lattice="mass", y_lattice="face")
    dxc = dx / ratio
    worst = 0.0
    for jp in range(span):
        for ip in range(span):
            parent_div = ((u[j0 + jp, i0 + ip + 1] - u[j0 + jp, i0 + ip]) * dx
                          + (v[j0 + jp + 1, i0 + ip] - v[j0 + jp, i0 + ip]) * dx)
            child_div = 0.0
            for a in range(ratio):
                for b in range(ratio):
                    j, i = jp * ratio + a, ip * ratio + b
                    child_div += ((cu[j, i + 1] - cu[j, i]) * dxc
                                  + (cv[j + 1, i] - cv[j, i]) * dxc)
            worst = max(worst, abs(child_div - parent_div))
    assert worst < 1.0e-11 * dx


@pytest.mark.parametrize("stagger", STAGGERS)
def test_a_horizontally_linear_field_is_reproduced_exactly(stagger) -> None:
    """Why an unforced sheared environment stays equal to the parent."""

    x_face, y_face, _ = LATTICE_OF_STAGGER[stagger]
    pl = lattice_shape(stagger, 2, PARENT_NY, PARENT_NX)
    cl = lattice_shape(stagger, 2, CHILD_NY, CHILD_NX)
    # Coordinates in parent cells of each lattice point.
    def coords(n_points, face):
        return numpy.arange(n_points, dtype=numpy.float64) + (0.0 if face else 0.5)

    px, py = coords(pl[2], x_face), coords(pl[1], y_face)
    parent = numpy.empty(pl, dtype=numpy.float64)
    for k in range(pl[0]):
        parent[k] = 10.0 + 0.25 * k + 0.5 * px[None, :] - 0.125 * py[:, None]
    target, _ = reference_capture(
        parent.astype(numpy.float32), child_lattice=cl, child_volume=None, side="full",
        width=WIDTH, ratio=RATIO, parent_i0=PARENT_I0, parent_j0=PARENT_J0,
        stagger=stagger, parent_dt_s=1.0,
    )
    cx = PARENT_I0 + (numpy.arange(cl[2]) + (0.0 if x_face else 0.5)) / RATIO
    cy = PARENT_J0 + (numpy.arange(cl[1]) + (0.0 if y_face else 0.5)) / RATIO
    for k in range(cl[0]):
        exact = (10.0 + 0.25 * k + 0.5 * cx[None, :] - 0.125 * cy[:, None]).astype(numpy.float32)
        assert numpy.max(numpy.abs(target[k] - exact)) <= 4.0e-6


# ---------------------------------------------------------------------------
# The same kernels on a device: NVRTC compiles the packaged string and the
# launch geometry covers every element once.  GPU-gated exactly like
# test_nest_relaxation_kernel.py's device tests.
# ---------------------------------------------------------------------------
import os as _os

requires_opted_in_device = pytest.mark.skipif(
    _os.environ.get("GM1_NEST_DEVICE_TESTS") != "1",
    reason="GM1_NEST_DEVICE_TESTS=1 is not set; no CUDA context is opened unless asked for by name",
)


@requires_opted_in_device
def test_the_live_nest_kernels_launch_on_a_device_and_match_the_cpu_references() -> None:
    cupy = pytest.importorskip("cupy")
    from cm1core.nest_live_kernels import (
        compile_nest_live_module,
        launch_capture,
        launch_restrict,
        launch_spec_apply,
        launch_spec_gather,
    )

    module = compile_nest_live_module(cupy)
    stream = cupy.cuda.Stream(non_blocking=True)
    for stagger in STAGGERS:
        rng = _rng(1000 + STAGGERS.index(stagger))
        pl, cl = _parent_lattice(stagger), _child_lattice(stagger)
        parent_owner, parent = _padded(rng, pl)
        child_owner, child = _padded(rng, cl)
        # Every allocation and upload is made on the launch stream, as the live
        # nest makes them; an upload queued on another stream is not ordered
        # against the kernel.
        with stream:
            dev_parent, dev_child = cupy.asarray(parent_owner), cupy.asarray(child_owner)
        for side in SIDES:
            shape = side_shape(side, cl, WIDTH)
            with stream:
                value = cupy.zeros(shape, dtype=cupy.float32)
                tend = None if side == "full" else cupy.zeros(shape, dtype=cupy.float32)
            with stream:
                launch_capture(
                    cupy, module, parent=dev_parent, parent_origin=PAD, parent_lattice=pl,
                    child=None if side == "full" else dev_child, child_origin=PAD,
                    child_lattice=cl, out_value=value, out_tend=tend, side=side,
                    width=WIDTH, ratio=RATIO, parent_i0=PARENT_I0, parent_j0=PARENT_J0,
                    stagger=stagger, parent_dt_s=2.25, stream=stream,
                )
            stream.synchronize()
            want_value, want_tend = reference_capture(
                parent, child_lattice=cl, child_volume=None if side == "full" else child,
                side=side, width=WIDTH, ratio=RATIO, parent_i0=PARENT_I0,
                parent_j0=PARENT_J0, stagger=stagger, parent_dt_s=2.25,
            )
            assert cupy.asnumpy(value).tobytes() == want_value.tobytes(), (stagger, side)
            if tend is not None:
                assert cupy.asnumpy(tend).tobytes() == want_tend.tobytes(), (stagger, side)
            if side == "full":
                continue
            with stream:
                snap = cupy.zeros(side_shape(side, cl, SPEC), dtype=cupy.float32)
                field = dev_child.copy()
            with stream:
                launch_spec_gather(cupy, module, field=field, origin=PAD, lattice=cl,
                                   snap=snap, side=side, spec_zone=SPEC, stream=stream)
                launch_spec_apply(cupy, module, field=field, origin=PAD, lattice=cl,
                                  tend=tend, side=side, spec_zone=SPEC, width=WIDTH,
                                  dt=0.0833333358, snap=snap, stream=stream)
            stream.synchronize()
            want = reference_spec_apply(
                child, lattice=cl, tend=want_tend, side=side, spec_zone=SPEC,
                width=WIDTH, dt=0.0833333358,
                snap=numpy.where(owned_band_mask(side, cl, SPEC),
                                 cupy.asnumpy(snap), numpy.float32(0.0)),
            )
            got = cupy.asnumpy(field)[1:1 + cl[0], 3:3 + cl[1], 3:3 + cl[2]]
            assert got.tobytes() == want.tobytes(), (stagger, side, "spec")
        with stream:
            target = dev_parent.copy()
            launch_restrict(cupy, module, child=dev_child, child_origin=PAD,
                            child_lattice=cl, parent=target, parent_origin=PAD,
                            parent_lattice=pl, span_x=SPAN_X, span_y=SPAN_Y, ratio=RATIO,
                            relax_zone=2, parent_i0=PARENT_I0, parent_j0=PARENT_J0,
                            stagger=stagger, stream=stream)
        stream.synchronize()
        want = reference_restrict(child, parent, child_lattice=cl, span_x=SPAN_X,
                                  span_y=SPAN_Y, ratio=RATIO, relax_zone=2,
                                  parent_i0=PARENT_I0, parent_j0=PARENT_J0, stagger=stagger)
        got = cupy.asnumpy(target)[1:1 + pl[0], 3:3 + pl[1], 3:3 + pl[2]]
        assert got.tobytes() == want.tobytes(), (stagger, "restrict")


def test_the_driver_puts_a_restored_child_on_the_parents_clock() -> None:
    """``NestDriver.__init__`` calls ``_align_child_clock``; the method must exist.

    A merge once folded the method's body into ``__init__`` after the parcel
    block and dropped its ``def`` line, so every live nest raised
    ``AttributeError`` at construction.  This drives the method on a parent
    restored at step 7 with a time ratio of 3 (``restart_read.F:419-420``).
    """

    from pathlib import Path
    from types import SimpleNamespace

    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import load_config
    from cm1core.nest_live import NestDriver

    repo = Path(__file__).resolve().parents[1]
    config = build_runtime_config(
        load_config(repo / "examples" / "cartesian-supercell-translating-frame.toml")
    )
    parent = SimpleNamespace(clock=SimpleNamespace(
        completed_steps=7, domainlocx_m=87.5, domainlocy_m=21.0, model_seconds=7 * 1.5,
    ))
    child = SimpleNamespace(config=config, finish=SimpleNamespace(),
                            closure=SimpleNamespace(_held_step=0, _pending_step=1, _tke_stage=0))
    driver = object.__new__(NestDriver)
    driver.parent, driver.child, driver.q = parent, child, 3
    assert callable(getattr(NestDriver, "_align_child_clock", None))
    driver._align_child_clock()
    assert child.clock.completed_steps == 21
    assert (child.clock.domainlocx_m, child.clock.domainlocy_m) == (87.5, 21.0)
    assert child.finish.completed_physics_step == 21
    assert child.finish.completed_solve3_step == 21
    assert (child.closure._held_step, child.closure._pending_step, child.closure._tke_stage) == (21, None, 3)
    assert child.clock.model_seconds == parent.clock.model_seconds
