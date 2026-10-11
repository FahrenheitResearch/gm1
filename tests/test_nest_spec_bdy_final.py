"""WRF's end-of-step ``spec_bdy_final`` in the live nest.

WRF ends ``solve_em`` by setting the specified rows to ``bdy + dtbc*bdy_tend``
(WRF v4.6.1 ``dyn_em/solve_em.F:4531-4716``, ``share/module_bc.F:2066-2216``);
a nested domain does it for u, v, w, t, ph, mu and every moist, tracer and
scalar field.  The ArWen chain added it first; the live nest's child
kept whatever microphysics did to its specified rows until the next parent
step.  These tests hold:

* the kernel text of modes 2 and 3 (``gm1_nest_spec_apply``) to its CPU
  reference word for word, compiled with ``g++`` (no CUDA context);
* the live nest's :meth:`NestLateralForcing.finalize_specified` to WRF's
  nested lane: every carrier, the y sides owning the corners, species held at
  zero or above, nothing inside the specified rows touched;
* the child runtime to WRF's placement: once per step, after microphysics,
  before ``solve3``.

``GM1_NEST_DEVICE_TESTS=1`` adds the kernel on a device.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import types

import pytest

numpy = pytest.importorskip("numpy")

from cm1core.nest_live_kernels import (  # noqa: E402
    lattice_shape,
    owned_band_mask,
    reference_spec_apply,
    side_shape,
)

requires_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")

NZ, NY, NX = 3, 12, 15
WIDTH = 5
PAD = (1, 3, 3)
STAGGERS = ("u", "v", "w", "mass")
SIDES = ("west", "east", "south", "north")
CODES = {"west": 1, "east": 2, "south": 3, "north": 4}


@pytest.fixture(scope="module")
def harness():
    from tools.oracle.nest_live_oracle import open_harness

    return open_harness()


def _padded(rng, lattice):
    owner = rng.uniform(-3.0, 3.0, size=(lattice[0] + 2, lattice[1] + 7, lattice[2] + 7)).astype(numpy.float32)
    return owner


@requires_gxx
@pytest.mark.parametrize("stagger", STAGGERS)
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("spec_zone", (1, 2))
@pytest.mark.parametrize("mode", (2, 3))
@pytest.mark.parametrize("order", ("linear", "shuffled"))
def test_spec_bdy_final_kernel_text_is_its_reference(harness, stagger, side, spec_zone, mode, order) -> None:
    rng = numpy.random.default_rng(100 * spec_zone + mode + len(side))
    cl = lattice_shape(stagger, NZ, NY, NX)
    owner = _padded(rng, cl)
    shape = side_shape(side, cl, WIDTH)
    value = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
    tend = rng.uniform(-1.0e-3, 1.0e-3, size=shape).astype(numpy.float32)
    band = side_shape(side, cl, spec_zone)
    count = band[0] * band[1] * band[2]
    dtbc = 37.5
    params = [owner.shape[1], owner.shape[2], *PAD, *cl, CODES[side], spec_zone, WIDTH, mode, 0, count]
    got = harness.apply(field=owner, snap=value, base=numpy.zeros(1, numpy.float32), tend=tend,
                        params=params, dt=dtbc, order=order)
    physical = (slice(1, 1 + cl[0]), slice(3, 3 + cl[1]), slice(3, 3 + cl[2]))
    want = reference_spec_apply(owner[physical], lattice=cl, tend=tend, side=side, spec_zone=spec_zone,
                                width=WIDTH, dt=dtbc, value=value, nonnegative=(mode == 3))
    assert got[physical].tobytes() == want.tobytes()
    # The owned band is value + dtbc*tend (held at zero in mode 3); nothing else moved.
    mask = owned_band_mask(side, cl, spec_zone)
    changed = got[physical] != owner[physical]
    assert int(changed.sum()) <= int(mask.sum())
    halo = got.copy()
    halo[physical] = 0.0
    ref = owner.copy()
    ref[physical] = 0.0
    assert halo.tobytes() == ref.tobytes()
    if mode == 3:
        assert float(got[physical].min()) >= 0.0 or float(owner[physical].min()) < 0.0


def test_the_reference_is_value_plus_dtbc_tend_on_the_owned_rows() -> None:
    cl = lattice_shape("mass", NZ, NY, NX)
    field = numpy.full(cl, 9.0, numpy.float32)
    for side in SIDES:
        shape = side_shape(side, cl, WIDTH)
        value = numpy.full(shape, -1.0, numpy.float32)
        tend = numpy.full(shape, 0.01, numpy.float32)
        out = reference_spec_apply(field, lattice=cl, tend=tend, side=side, spec_zone=1, width=WIDTH,
                                   dt=50.0, value=value)
        clamped = reference_spec_apply(field, lattice=cl, tend=tend, side=side, spec_zone=1, width=WIDTH,
                                       dt=50.0, value=value, nonnegative=True)
        rows = {"south": out[:, 0, :], "north": out[:, NY - 1, :],
                "west": out[:, 1:NY - 1, 0], "east": out[:, 1:NY - 1, NX - 1]}[side]
        assert numpy.all(rows == numpy.float32(-0.5))
        crows = {"south": clamped[:, 0, :], "north": clamped[:, NY - 1, :],
                 "west": clamped[:, 1:NY - 1, 0], "east": clamped[:, 1:NY - 1, NX - 1]}[side]
        assert numpy.all(crows == 0.0)


# ---------------------------------------------------------------------------
# The live nest's forcing, on a NumPy stand-in (the kernel replaced by its reference).
# ---------------------------------------------------------------------------
def _fake_cupy():
    return types.SimpleNamespace(
        float32=numpy.float32, asarray=numpy.asarray, zeros=numpy.zeros, copyto=numpy.copyto,
        asnumpy=numpy.asarray, ascontiguousarray=numpy.ascontiguousarray,
    )


class _Stream(contextlib.AbstractContextManager):
    def __exit__(self, *exc):
        return False

    def synchronize(self):
        return None


def host_spec_apply(cupy, module, *, field, origin, lattice, tend, side, spec_zone, width, dt,
                    snap=None, base=None, value=None, nonnegative=False, stream=None):
    k0, j0, i0 = origin
    sl = (slice(k0, k0 + lattice[0]), slice(j0, j0 + lattice[1]), slice(i0, i0 + lattice[2]))
    view = field[sl]
    view[...] = reference_spec_apply(
        view, lattice=lattice, tend=tend, side=side, spec_zone=spec_zone, width=width, dt=dt,
        snap=snap, base=None if base is None else base[sl], value=value, nonnegative=nonnegative,
    )


def _child(numq=3):
    arrays = {name: numpy.zeros((NZ + 2, NY + 7, NX + 7), numpy.float32)
              for name in ("u3d", "ua", "v3d", "va", "w3d", "wa")}
    fields = {name: numpy.zeros((NZ, NY, NX), numpy.float32) for name in ("pp3d", "ppi", "th3d", "tha")}
    fields["q3d"] = numpy.zeros((numq, NZ, NY, NX), numpy.float32)
    fields["qa"] = numpy.zeros((numq, NZ, NY, NX), numpy.float32)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(dt_s=2.0), numq=numq,
        geometry=types.SimpleNamespace(nz=NZ, ny=NY, nx=NX), arrays=arrays, fields=fields,
        nest_forcing=None,
    )


def _forcing(monkeypatch, child):
    from cm1core import nest_live

    monkeypatch.setattr(nest_live, "compile_nest_live_module", lambda cupy: None)
    monkeypatch.setattr(nest_live, "compile_nest_relaxation_module", lambda cupy: None)
    monkeypatch.setattr(nest_live, "launch_spec_apply", host_spec_apply)
    geometry = types.SimpleNamespace(spec_zone=1, relax_zone=4)
    return nest_live.NestLateralForcing(cupy=_fake_cupy(), stream=_Stream(), geometry=geometry,
                                        child=child, parent_dt_s=6.0)


def test_the_live_nest_resets_every_carrier_at_the_end_of_the_step(monkeypatch) -> None:
    from cm1core import nest_live

    child = _child()
    forcing = _forcing(monkeypatch, child)
    rng = numpy.random.default_rng(5)
    for sides in forcing.tables.values():
        for value, tend in sides.values():
            value[...] = rng.uniform(-1.0, 1.0, size=value.shape)
            tend[...] = rng.uniform(-0.5, 0.5, size=tend.shape)
    with pytest.raises(RuntimeError, match="before the boundary tables"):
        forcing.finalize_specified(child)
    forcing.tables_ready = True
    for _ in range(2):
        forcing.advance_clock()
    dtbc = numpy.float32(forcing.dtbc)
    assert dtbc == numpy.float32(4.0)
    for name in child.arrays:
        child.arrays[name][...] = rng.normal(size=child.arrays[name].shape)
    for name in ("pp3d", "th3d", "q3d"):
        child.fields[name][...] = rng.normal(size=child.fields[name].shape)
    before_arrays = {k: v.copy() for k, v in child.arrays.items()}
    before_fields = {k: v.copy() for k, v in child.fields.items()}
    forcing.finalize_specified(child)
    assert forcing.final_specifications == 1
    for carrier in forcing.all_carriers:
        owner, origin = nest_live._current(child, carrier)
        lattice = nest_live._grid_lattice(child, carrier.stagger)
        k0, j0, i0 = origin
        f = owner[k0:k0 + lattice[0], j0:j0 + lattice[1], i0:i0 + lattice[2]]
        nzl, nyl, nxl = lattice
        clamp = (lambda a: numpy.maximum(a, numpy.float32(0.0))) if carrier.species is not None else (lambda a: a)
        target = {side: clamp((forcing.tables[carrier.name][side][0]
                               + dtbc * forcing.tables[carrier.name][side][1]).astype(numpy.float32))
                  for side in SIDES}
        assert numpy.array_equal(f[:, 0, :], target["south"][:, 0, :]), carrier.name
        assert numpy.array_equal(f[:, nyl - 1, :], target["north"][:, 0, :]), carrier.name
        assert numpy.array_equal(f[:, 1:nyl - 1, 0], target["west"][:, 1:nyl - 1, 0]), carrier.name
        assert numpy.array_equal(f[:, 1:nyl - 1, nxl - 1], target["east"][:, 1:nyl - 1, 0]), carrier.name
    # Inside the specified rows nothing moved, and the time-level-n arrays are untouched.
    assert numpy.array_equal(child.fields["th3d"][:, 1:-1, 1:-1], before_fields["th3d"][:, 1:-1, 1:-1])
    assert numpy.array_equal(child.fields["q3d"][:, :, 1:-1, 1:-1], before_fields["q3d"][:, :, 1:-1, 1:-1])
    for name in ("ua", "va", "wa"):
        assert numpy.array_equal(child.arrays[name], before_arrays[name])
    for name in ("ppi", "tha", "qa"):
        assert numpy.array_equal(child.fields[name], before_fields[name])
    record = forcing.record()["spec_bdy_final"]
    assert record["lane"] == "nested" and record["steps"] == 1
    assert record["carriers"] == ["u", "v", "w", "pp", "th", "q0", "q1", "q2"]
    assert "solve_em.F:4531-4716" in record["authority"]


def test_the_child_runtime_resets_after_microphysics_and_before_solve3() -> None:
    from cm1core.gm1_runtime import StepPhase
    from cm1core.nest_live import make_child_runtime_class

    calls: list[str] = []

    class Base:
        def execute(self, operation, *, step):
            calls.append(operation.phase.value)

    forcing = types.SimpleNamespace(finalize_specified=lambda d: calls.append("spec_bdy_final"))
    runtime = make_child_runtime_class(Base)()
    runtime.dynamics = types.SimpleNamespace(nest_forcing=forcing)
    for phase in StepPhase:
        runtime.execute(types.SimpleNamespace(phase=phase), step=1)
    kept = [p.value for p in StepPhase if p not in (StepPhase.MASS_BEGIN, StepPhase.MASS_FINISH)]
    assert calls.count("spec_bdy_final") == 1
    at = calls.index("spec_bdy_final")
    assert calls[at + 1] == StepPhase.SOLVE3.value
    assert calls[:at] + calls[at + 1:] == kept
    # A runtime with no forcing (a parent, or a single domain) is untouched.
    calls.clear()
    runtime.dynamics = types.SimpleNamespace(nest_forcing=None)
    runtime.execute(types.SimpleNamespace(phase=StepPhase.SOLVE3), step=1)
    assert calls == [StepPhase.SOLVE3.value]


@pytest.mark.skipif(os.environ.get("GM1_NEST_DEVICE_TESTS") != "1",
                    reason="GM1_NEST_DEVICE_TESTS=1 is not set; no CUDA context is opened unless asked for by name")
def test_spec_bdy_final_on_a_device_is_its_reference() -> None:
    cupy = pytest.importorskip("cupy")
    from cm1core.nest_live_kernels import compile_nest_live_module, launch_spec_apply

    module = compile_nest_live_module(cupy)
    stream = cupy.cuda.Stream(non_blocking=True)
    for stagger in STAGGERS:
        cl = lattice_shape(stagger, NZ, NY, NX)
        for side in SIDES:
            for nonnegative in (False, True):
                rng = numpy.random.default_rng(len(stagger) * 10 + len(side) + int(nonnegative))
                owner = _padded(rng, cl)
                shape = side_shape(side, cl, WIDTH)
                value = rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
                tend = rng.uniform(-1.0e-3, 1.0e-3, size=shape).astype(numpy.float32)
                with stream:
                    dev = cupy.asarray(owner)
                    dvalue, dtend = cupy.asarray(value), cupy.asarray(tend)
                    launch_spec_apply(cupy, module, field=dev, origin=PAD, lattice=cl, tend=dtend,
                                      side=side, spec_zone=1, width=WIDTH, dt=37.5, value=dvalue,
                                      nonnegative=nonnegative, stream=stream)
                stream.synchronize()
                physical = (slice(1, 1 + cl[0]), slice(3, 3 + cl[1]), slice(3, 3 + cl[2]))
                want = reference_spec_apply(owner[physical], lattice=cl, tend=tend, side=side, spec_zone=1,
                                            width=WIDTH, dt=37.5, value=value, nonnegative=nonnegative)
                assert cupy.asnumpy(dev)[physical].tobytes() == want.tobytes(), (stagger, side, nonnegative)
