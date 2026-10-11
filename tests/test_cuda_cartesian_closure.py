"""CPU storage/cadence tests for the new production orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cm1core import cuda_cartesian_closure as closure
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


class _Array(np.ndarray):
    def get(self, *, stream=None):
        return np.asarray(self)

    def set(self, value, stream=None):
        np.copyto(np.asarray(self), value)


class _Stream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class _Registry:
    def __init__(self):
        self.buffers = {}
        self.peak = 0

    def allocate(self, name, shape, *, zero=False, dtype=None):
        assert name not in self.buffers
        value = (np.zeros if zero else np.empty)(shape, dtype=np.float32 if dtype is None else dtype).view(_Array)
        self.buffers[name] = value
        self.peak = max(self.peak, sum(value.nbytes for value in self.buffers.values()))
        return value

    def release(self, names):
        for name in names:
            self.buffers.pop(name)


def _cp():
    # `maximum` is the closure's ksmax fold (misclibs.F:2846-2866); numpy's has
    # cupy's signature, including the `out=` the fold accumulates through.
    return SimpleNamespace(copyto=np.copyto, broadcast_to=np.broadcast_to, maximum=np.maximum, float32=np.float32, float64=np.float64, uint32=np.uint32, uint8=np.uint8)


def _shell(grid, **surface):
    instance = closure.CartesianClosure.__new__(closure.CartesianClosure)
    instance.grid = grid
    instance.config = RuntimeConfig(grid, tile_nx=11, tile_ny=8, **surface)
    instance.plan = closure._tile_plan(grid, 11, 8)
    instance.allocation_plan = closure.cartesian_closure_allocation_plan(instance.config)
    instance.cp, instance.stream, instance.registry = _cp(), _Stream(), _Registry()
    instance.vertical = SimpleNamespace(rds=np.ones(grid.nz + 2, np.float32), rdsf=np.ones(grid.nz + 3, np.float32))
    instance.tauh_profile = instance.taus_profile = np.zeros(grid.nz, np.float32)
    instance.tauf_profile = np.zeros(grid.nz + 1, np.float32)
    instance.sgs_compiled = SimpleNamespace(sgs_source_sha256="a" * 64, tke_source_sha256="b" * 64, max_nk=grid.nz)
    # The lower boundary compiles nothing for the free-slip default.
    instance.surface_compiled = None
    instance._pending_step, instance._held_step, instance._tke_stage = None, 0, 3
    instance.last_receipts = {}
    # src/init_surface.F:40-322, once, exactly as the constructor runs it.
    instance._surface_fields()
    return instance


@pytest.mark.parametrize("stagger", ("mass", "u", "v", "w"))
def test_strided_canonical_wind_packs_and_each_true_face_scatters_once(stagger):
    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    plan = closure._tile_plan(g, 11, 8)
    padded = np.full(g.padded_shape(stagger, halo_xy=3, halo_z=1), -999, np.float32)
    source = padded[g.physical_slices(stagger, halo_xy=3, halo_z=1)]
    expected = np.arange(source.size, dtype=np.float32).reshape(source.shape)
    source[:] = expected
    assert not source.flags.c_contiguous
    result = np.full(source.shape, np.nan, np.float32)
    coverage = np.zeros(source.shape[-2:], np.uint8)
    for spec in plan.specs:
        shape = (source.shape[0], spec.cny + (stagger == "v"), spec.cnx + (stagger == "u"))
        tile = np.empty(shape, np.float32)
        closure._copy_tile(np, tile, source, spec)
        closure._scatter(np, result, tile, spec, stagger)
        y0, y1, x0, x1 = spec.owned_rect(stagger)
        coverage[y0:y1, x0:x1] += 1
    np.testing.assert_array_equal(result, expected)
    assert np.all(coverage == 1)


def test_allocation_ledger_prices_237_levels_and_no_full_closure_replica():
    config = RuntimeConfig(CartesianGridGeometry(400, 400, 237, 75, 75, 75))
    plan = closure.cartesian_closure_allocation_plan(config)
    assert plan["compute_shape_zyx"] == [237, 70, 70]
    # 511976464 before the ksmax fold of misclibs.F:2846-2866 added its own
    # four-byte device scalar beside the invalid-operand counter.
    assert plan["held_phase_payload_bytes"] == 511976468
    assert plan["tke_phase_payload_bytes"] == 206372384
    assert plan["persistent_profile_payload_bytes"] == 12428
    assert all(shape[-2:] != (400, 400) for shapes in plan["held_groups"].values() for shape in shapes.values() if len(shape) >= 2)


def test_fused_tile_chain_has_exact_source_order_and_releases_all_phase_storage():
    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    instance = _shell(g)
    instance.dynamics = SimpleNamespace(
        tke={name: np.zeros(g.w_shape, np.float32) for name in ("tkea", "tke3d", "tketen")},
        held={name: np.zeros(shape, np.float32) for name, shape in (("uten", g.u_shape), ("vten", g.v_shape), ("wten", g.w_shape), ("thten1", g.mass_shape), ("qten", (6, *g.mass_shape)))},
    )
    calls = []
    instance._gather = lambda *_args: calls.append("gather")
    instance._refresh_nm_def = lambda *_args: calls.append("nm/def")

    def coefficients(work, _retained, _invalid, _spec, *, first):
        # src/init3d.F:2688-2760: a fresh start runs tkekm on its zero TKE;
        # only an interpolated start seeds it with turbsmag.
        assert first is False
        calls.append("coefficients")
        for index, name in enumerate(("tkea", "tke3d", "tketen"), 1):
            work[name].fill(index)

    def held(tile, *_args):
        calls.append("held")
        for index, name in enumerate(("uten", "vten", "wten", "thten1", "qten"), 1):
            tile[name].fill(index + 10)

    instance._tke_coefficients = coefficients
    instance._tke_held = lambda *_args: calls.append("TKE source/diffusion")
    instance._held_tile = held
    instance.closure(1)
    assert calls == []
    instance.held_tendencies(1)
    assert calls == ["gather", "nm/def", "coefficients", "TKE source/diffusion", "held"] * instance.plan.ntiles
    for index, name in enumerate(("tkea", "tke3d", "tketen"), 1):
        assert np.all(instance.dynamics.tke[name] == index)
    for index, name in enumerate(("uten", "vten", "wten", "thten1", "qten"), 1):
        assert np.all(instance.dynamics.held[name] == index + 10)
    assert instance.registry.buffers == {}
    assert instance.registry.peak == instance.allocation_plan["held_phase_payload_bytes"]
    assert instance._held_step == 1 and instance._tke_stage == 0
    with pytest.raises(RuntimeError, match="matching closure"):
        instance.held_tendencies(1)
    with pytest.raises(RuntimeError, match="large-step boundary"):
        instance.closure(2)


def test_semi_slip_inserts_cm1s_surface_phases_into_the_same_chain():
    """CM1 r21.1 ``sfc_and_turb``: ``getsfc`` first, then ``tkebc``.

    ``src/turb.F:522-616`` builds the lowest-level wind and the exchange
    coefficients before anything else in the turbulence driver, and
    ``src/turb.F:1378`` applies ``tkebc`` after the coefficient operators
    have written the free-slip lower face.  Nothing runs at all for the
    free-slip bottom, so that chain is exactly what it was.
    """

    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    instance = _shell(g, bbc=3, sfcmodel=1, cnstcd=0.0035)
    instance.surface_compiled = None
    instance.dynamics = SimpleNamespace(
        tke={name: np.zeros(g.w_shape, np.float32) for name in ("tkea", "tke3d", "tketen")},
        held={name: np.zeros(shape, np.float32) for name, shape in (("uten", g.u_shape), ("vten", g.v_shape), ("wten", g.w_shape), ("thten1", g.mass_shape), ("qten", (6, *g.mass_shape)))},
    )
    calls = []
    surfaces = []
    instance._gather = lambda *_args: calls.append("gather")
    instance._refresh_nm_def = lambda *_args: calls.append("nm/def")
    instance._tke_coefficients = lambda *_a, **_k: calls.append("coefficients")
    instance._tke_held = lambda *_args: calls.append("TKE source/diffusion")
    instance._held_tile = lambda *_args: calls.append("held")

    def surface_layer(_tile, _work, surface, *_args):
        calls.append("getsfc")
        surfaces.append(dict(surface))

    instance._surface_layer = surface_layer
    instance._surface_enthalpy_flux = lambda *_args: None
    instance._surface_tke_face = lambda *_args: calls.append("tkebc")
    instance.closure(1)
    instance.held_tendencies(1)
    assert calls == [
        "gather",
        "getsfc",
        "nm/def",
        "coefficients",
        "tkebc",
        "TKE source/diffusion",
        "held",
    ] * instance.plan.ntiles
    # The per-tile surface carriers exist, are zeroed, and are released with
    # the phase.  The four whole-domain fields init_surface filled before the
    # first step are model state, so they outlive the phase and are named in
    # the plan separately.
    from cm1core.cuda_surface_drag import (
        SURFACE_PERSISTENT_NAMES,
        SURFACE_STATE_NAMES,
    )

    assert set(surfaces[0]) == set(SURFACE_STATE_NAMES) | set(SURFACE_PERSISTENT_NAMES)
    assert all(np.all(value == 0.0) for value in surfaces[0].values())
    assert set(instance.registry.buffers) == {
        f"closure.surface.{name}" for name in SURFACE_PERSISTENT_NAMES
    }
    assert (
        instance.registry.peak
        == instance.allocation_plan["held_phase_payload_bytes"]
        + instance.allocation_plan["surface_state_payload_bytes"]
    )


def test_a_free_slip_closure_runs_no_surface_phase_and_binds_no_operator():
    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    instance = _shell(g)
    assert instance.surface_selection.bbc == 1
    assert not instance.surface_selection.active
    assert instance.stress_selection.bbc == 1
    assert instance._surface_binding({}) is None
    # No allocation, so nothing to release and nothing to launch.
    assert "surface" not in instance.allocation_plan["held_groups"]


def test_tke_stage_gathers_one_immutable_snapshot_and_publishes_only_after_all_tiles():
    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    instance = _shell(g)
    initial = np.arange(np.prod(g.w_shape), dtype=np.float32).reshape(g.w_shape)
    current = initial.copy()
    instance.dynamics = SimpleNamespace(
        tke={"tkea": initial.copy(), "tke3d": current, "tketen": np.zeros(g.w_shape, np.float32)},
        metrics={**{name: np.ones(g.w_shape, np.float32) for name in ("c1", "c2", "rrf0")}, "gz": np.ones(g.surface_shape, np.float32)},
        moist_bridge=SimpleNamespace(divx=np.zeros(g.mass_shape, np.float32)),
        arrays={
            "rru": np.zeros((g.nz + 2, g.ny + 6, g.nx + 7), np.float32),
            "rrv": np.zeros((g.nz + 2, g.ny + 7, g.nx + 6), np.float32),
            "rrw": np.zeros((g.nz + 3, g.ny + 6, g.nx + 6), np.float32),
        },
    )
    calls = []

    def enqueue(_grid, _block, args, *, stream):
        np.testing.assert_array_equal(current, initial)
        source, output = args[0], args[11]
        assert output.shape == (g.nz + 1, int(args[13]), int(args[12]))
        assert int(args[14]) == g.nz and int(args[15]) == 3
        np.add(source, 1, out=output)  # enqueue-contract surrogate, not CM1 physics
        calls.append(True)

    instance.sgs_compiled.tke_kernels = (None, enqueue)
    instance._held_step, instance._tke_stage = 1, 0
    instance.tke_rk(1)
    assert len(calls) == instance.plan.ntiles > 1
    np.testing.assert_array_equal(current, initial + 1)
    np.testing.assert_array_equal(instance.dynamics.tke["tkea"], initial)
    assert instance.registry.buffers == {}
    assert instance.registry.peak == instance.allocation_plan["tke_phase_payload_bytes"]
    with pytest.raises(RuntimeError, match="next stage"):
        instance.tke_rk(3)


# ------------------------------------------- the bbc=3 lowest-level heating


def _surface_heating_fixture(ni: int, nj: int, nk: int, dz: float):
    """Carriers for one tile's ``src/solve1.F:917-988`` block."""

    rng = np.random.default_rng(5)
    f32 = np.float32
    tile = {
        "thten1": np.zeros((nk, nj, ni), f32),
        "pi0": rng.uniform(0.85, 1.0, (nk, nj, ni)).astype(f32),
        "ppi": rng.uniform(-0.01, 0.01, (nk, nj, ni)).astype(f32),
        "qa": np.zeros((6, nk, nj, ni), f32),
    }
    tile["qa"][0] = rng.uniform(0.004, 0.016, (nk, nj, ni)).astype(f32)
    tile["qa"][1] = rng.uniform(0.0, 0.001, (nk, nj, ni)).astype(f32)
    tile["qa"][2] = rng.uniform(0.0, 0.0005, (nk, nj, ni)).astype(f32)
    work = {
        "dissten": rng.uniform(1e-4, 2e-3, (nk + 1, nj, ni)).astype(f32),
        "zf": (np.arange(nk + 1, dtype=np.float32)[:, None, None] * f32(dz))
        * np.ones((1, nj, ni), f32),
    }
    from cm1core.cuda_surface_drag import SURFACE_STATE_NAMES

    surface = {name: np.zeros((nj, ni), f32) for name in SURFACE_STATE_NAMES}
    surface["znt"][:] = f32(0.0321)
    surface["ust"][:] = rng.uniform(0.4, 1.1, (nj, ni)).astype(f32)
    return tile, work, surface


def test_semi_slip_writes_the_surface_rate_before_the_lowest_level_reads_it(monkeypatch):
    """``src/solve1.F:932-935,950-962``: k=1 is heated, and by a live rate.

    The rate is a carrier in this port rather than a Fortran scalar, so the
    launch that fills it has to be enqueued before the heating reads it.
    This drives the closure's own ``solve1`` block with a recording surface
    image and checks three things the previous arrangement got wrong: the
    rate is written first, the lowest model level really is heated, and the
    interior consumer is handed the ``k=2..nk`` slice so nothing heats
    ``k=1`` twice.
    """

    from cm1core import cuda_surface_drag as drag
    from cm1core.eqtset2_surface_dissipation import (
        SurfaceDissipationCarriers,
        cpu_surface_dissipative_heating_inplace,
    )

    import sys

    # enqueue_surface_dissipation imports cupy for its uint32 launch words;
    # the kernel below is a host stand-in, so a host cupy stand-in suffices
    # (the same one the surface-phase ordering test below installs).
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        type(
            "cupy",
            (),
            {"uint32": np.uint32, "float32": np.float32, "ndarray": np.ndarray},
        ),
    )

    f32 = np.float32
    ni, nj, nk, dz = 6, 5, 7, 250.0
    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    instance = _shell(g, bbc=3, sfcmodel=1, cnstcd=0.0035)
    tile, work, surface = _surface_heating_fixture(ni, nj, nk, dz)
    order: list[str] = []

    # src/solve1.F:956-957, transcribed here by hand, is what the CUDA
    # cm1_sfc_dissipation_v1 image computes; the recording stand-in writes
    # exactly those words so the closure's ordering is what is under test.
    karman = f32(drag.CM1_KARMAN_F32)
    expected_rate = np.empty((nj, ni), f32)
    for j in range(nj):
        for i in range(ni):
            dz1 = f32(work["zf"][1, j, i] - work["zf"][0, j, i])
            znt = surface["znt"][j, i]
            ust = surface["ust"][j, i]
            cubed = f32(f32(ust * ust) * ust)
            expected_rate[j, i] = f32(
                f32(cubed * f32(np.log(f32(f32(dz1 + znt) / znt))))
                / f32(karman * dz1)
            )
    assert np.all(expected_rate > 0.0)

    def rate_kernel(_grid, _block, args, *, stream=None):
        order.append("cm1_sfc_dissipation_v1")
        args[3][:] = expected_rate

    kernels = [None] * len(drag.SURFACE_DRAG_KERNEL_NAMES)
    kernels[6] = rate_kernel
    instance.surface_compiled = SimpleNamespace(
        kernels=tuple(kernels), source_sha256="c" * 64, target_arch="86"
    )

    class _Idiss:
        def __init__(self):
            self.slice_shape = None

        def apply(self, carriers):
            order.append("interior-idiss")
            self.slice_shape = tuple(carriers.thten1.shape)
            carriers.thten1[...] += f32(1.0)

    idiss = _Idiss()
    seen: dict[str, np.ndarray] = {}

    def record_heating(carriers, *, nx, ny, nz, equation_set=2, stream=None):
        order.append("surface-heating")
        seen["epsd"] = np.array(carriers.epsd_surface, copy=True)
        return cpu_surface_dissipative_heating_inplace(
            carriers, nx=nx, ny=ny, nz=nz, equation_set=equation_set
        )

    instance.cp = _cp()
    original = closure.apply_surface_dissipative_heating
    closure.apply_surface_dissipative_heating = record_heating
    try:
        instance._dissipative_heating(tile, work, surface, idiss, ni, nj, nk)
    finally:
        closure.apply_surface_dissipative_heating = original

    # The rate is written before anything reads it.
    assert order.index("cm1_sfc_dissipation_v1") < order.index("surface-heating")
    assert np.array_equal(seen["epsd"], expected_rate)
    assert np.all(seen["epsd"] > 0.0)
    # src/solve1.F:932-935: the interior average starts at k=2.
    assert idiss.slice_shape == (nk - 1, nj, ni)
    # The lowest model level is heated, and by the surface rate alone.
    assert np.all(tile["thten1"][0] > 0.0)
    expected = SurfaceDissipationCarriers(
        thten1=np.zeros((nk, nj, ni), f32),
        epsd_surface=expected_rate,
        pi0=tile["pi0"],
        ppi=tile["ppi"],
        **{
            name: tile["qa"][index]
            for index, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg"))
        },
    )
    cpu_surface_dissipative_heating_inplace(expected, nx=ni, ny=nj, nz=nk)
    assert np.array_equal(
        tile["thten1"][0].view(np.uint32), expected.thten1[0].view(np.uint32)
    )
    # and the interior consumer never touched it
    assert np.all(tile["thten1"][1:] == f32(1.0))


def test_a_free_slip_closure_heats_every_level_from_the_interior_average():
    """``src/solve1.F:932-935``: ``k1`` is 1 unless ``bbc=3``."""

    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    instance = _shell(g)
    ni, nj, nk = 6, 5, 7
    tile, work, surface = _surface_heating_fixture(ni, nj, nk, 250.0)
    order: list[str] = []

    class _Idiss:
        def apply(self, carriers):
            order.append("interior-idiss")
            assert tuple(carriers.thten1.shape) == (nk, nj, ni)

    instance._dissipative_heating(tile, work, surface, _Idiss(), ni, nj, nk)
    assert order == ["interior-idiss"]


def test_tkebc_is_launched_once_per_tile_and_after_the_coefficients(monkeypatch):
    """``src/turb.F:1351`` then ``:1378``: one ``tkebc``, after ``tkekm``.

    ``tkebc``'s ``bbc=2/3`` branch rescales the scalar lower face from the
    ``k=2`` eddy coefficients, so it cannot be enqueued before the operators
    that write them -- there it would read uninitialized workspace -- and it
    must not be enqueued twice.  This drives the real surface phases against
    a recording image, so it also proves the allocation plan supplies every
    carrier those launches index.
    """

    import sys

    from cm1core import cuda_surface_drag as drag

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        type(
            "cupy",
            (),
            {"uint32": np.uint32, "float32": np.float32, "ndarray": np.ndarray},
        ),
    )
    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    instance = _shell(g, bbc=3, sfcmodel=1, cnstcd=0.0035)
    instance.dynamics = SimpleNamespace(
        tke={name: np.zeros(g.w_shape, np.float32) for name in ("tkea", "tke3d", "tketen")},
        held={name: np.zeros(shape, np.float32) for name, shape in (("uten", g.u_shape), ("vten", g.v_shape), ("wten", g.w_shape), ("thten1", g.mass_shape), ("qten", (6, *g.mass_shape)))},
    )
    launches: list[str] = []

    def _recorder(index):
        def call(_grid, _block, _args, *, stream=None):
            launches.append(drag.SURFACE_DRAG_KERNEL_NAMES[index])

        return call

    instance.surface_compiled = SimpleNamespace(
        kernels=tuple(
            _recorder(index) for index in range(len(drag.SURFACE_DRAG_KERNEL_NAMES))
        ),
        source_sha256="d" * 64,
        target_arch="86",
    )
    instance._gather = lambda *_args: None
    instance._refresh_nm_def = lambda *_args: None
    instance._tke_coefficients = lambda *_a, **_k: launches.append("tkekm")
    instance._tke_held = lambda *_args: None
    instance._held_tile = lambda *_args: None
    instance.closure(1)
    instance.held_tendencies(1)

    per_tile = [
        "cm1_sfc_ground_relative_wind_v1",
        "cm1_sfc_getcecd_cnstcd_v1",
        "tkekm",
        "cm1_sfc_tkebc_lower_v1",
    ]
    assert launches == per_tile * instance.plan.ntiles
    assert launches.count("cm1_sfc_tkebc_lower_v1") == instance.plan.ntiles


def test_the_idiss_receipt_names_the_boundary_the_run_integrates(monkeypatch):
    """No workaround: the consumer is told the run's real ``bbc``.

    The sealed interior consumer used to refuse anything but the free-slip
    rigid bottom, and the only way to use it for ``bbc=3`` was to construct
    it with the default.  Its receipt then named a lower boundary the run had
    not integrated.
    """

    from cm1core.eqtset2_energy_terms import Eqtset2IdissConsumer

    captured: list[dict] = []

    class _Consumer(Eqtset2IdissConsumer):
        def __init__(self, **kwargs):
            captured.append(dict(kwargs))
            super().__init__(**kwargs)

    monkeypatch.setattr(closure, "Eqtset2IdissConsumer", _Consumer)
    g = CartesianGridGeometry(31, 23, 8, 75, 75, 75)
    for bbc, sfcmodel, expected_nz in ((1, 0, g.nz), (2, 0, g.nz), (3, 1, g.nz - 1)):
        captured.clear()
        instance = _shell(g)
        instance.config = RuntimeConfig(
            g, tile_nx=11, tile_ny=8, bbc=bbc, sfcmodel=sfcmodel, cnstcd=0.0035
        )
        instance.allocation_plan = closure.cartesian_closure_allocation_plan(
            instance.config
        )
        instance.surface_compiled = None
        instance.dynamics = SimpleNamespace(
            tke={name: np.zeros(g.w_shape, np.float32) for name in ("tkea", "tke3d", "tketen")},
            held={name: np.zeros(shape, np.float32) for name, shape in (("uten", g.u_shape), ("vten", g.v_shape), ("wten", g.w_shape), ("thten1", g.mass_shape), ("qten", (6, *g.mass_shape)))},
        )
        instance._gather = lambda *_args: None
        instance._refresh_nm_def = lambda *_args: None
        instance._tke_coefficients = lambda *_a, **_k: None
        instance._tke_held = lambda *_args: None
        instance._held_tile = lambda *_args: None
        instance._surface_layer = lambda *_args: None
        instance._surface_tke_face = lambda *_args: None
        instance.closure(1)
        instance.held_tendencies(1)
        assert captured
        for kwargs in captured:
            assert kwargs["bottom_boundary"] == bbc
            assert kwargs["nz"] == expected_nz
        consumer = _Consumer(nx=3, ny=2, nz=expected_nz, bottom_boundary=bbc)
        assert consumer.bottom_boundary == bbc


def test_isfcflx_launches_cm1s_surface_flux_inside_the_same_getsfc_block(monkeypatch):
    """``src/turb.F:645-649``: sfcflux runs right after getcecd, before tkekm.

    This drives the closure's real surface phases with recording kernels and
    a whole-domain surface state, so the test walks the same path a run does:
    the four whole-domain fields are gathered into the tile, getcecd and its
    enthalpy coefficients run, the surface pressure and the flux follow, and
    the roughness length is written back to the domain.
    """

    from cm1core import cuda_surface_drag as drag

    monkeypatch.setitem(
        __import__("sys").modules,
        "cupy",
        type(
            "cupy",
            (),
            {"uint32": np.uint32, "float32": np.float32, "ndarray": np.ndarray},
        ),
    )
    g = CartesianGridGeometry(31, 23, 12, 500, 500, 250)
    instance = _shell(
        g, bbc=3, sfcmodel=1, isfcflx=1, cecd=1, cnstcd=0.0035, cnstce=0.0012
    )
    assert set(instance.surface_state) == set(drag.SURFACE_PERSISTENT_NAMES)
    # src/init_surface.F:40-65 ran for the whole domain before step one.
    assert np.all(instance.surface_state["tsk"] == np.float32(299.28))
    assert np.all(instance.surface_state["xland"] == np.float32(2.0))
    assert np.all(instance.surface_state["mavail"] == np.float32(1.0))
    assert float(instance.surface_state["znt"].flat[0]) > 0.0
    instance.surface_state["znt"][:] = np.float32(0.0)  # getcecd must refill it

    launches: list[str] = []
    carriers: dict[str, tuple] = {}

    def _recorder(index):
        def call(_grid, _block, args, *, stream=None):
            name = drag.SURFACE_DRAG_KERNEL_NAMES[index]
            launches.append(name)
            carriers[name] = tuple(
                a.shape for a in args if isinstance(a, np.ndarray)
            )
            if name == "cm1_sfc_getcecd_cnstcd_v1":
                args[4][:] = np.float32(0.0321)  # znt, as getcecd writes it

        return call

    instance.surface_compiled = SimpleNamespace(
        kernels=tuple(
            _recorder(index) for index in range(len(drag.SURFACE_DRAG_KERNEL_NAMES))
        ),
        source_sha256="e" * 64,
        target_arch="86",
    )
    spec = instance.plan.specs[0]
    ni, nj, nk = spec.cnx, spec.cny, g.nz
    shapes = drag.surface_drag_shapes(ni, nj, nk)
    surface = {
        name: np.zeros(shapes[name], np.float32)
        for name in (
            *drag.SURFACE_STATE_NAMES,
            *drag.SURFACE_PERSISTENT_NAMES,
            *drag.SURFACE_FLUX_STATE_NAMES,
        )
    }
    tile = {
        name: np.zeros(shapes[name], np.float32)
        for name in ("u3d", "v3d", "terrain_msl", "prs", "th0", "tha")
    }
    tile["qa"] = np.zeros((6, nk, nj, ni), np.float32)
    work = {"zh": np.zeros((nk, nj, ni), np.float32)}
    flag = np.zeros((1,), np.uint32)

    instance._surface_layer(tile, work, surface, ni, nj, nk, spec, flag)
    instance._surface_enthalpy_flux(tile, surface, ni, nj, nk)

    assert launches == [
        "cm1_sfc_ground_relative_wind_v1",
        "cm1_sfc_getcecd_cnstcd_v1",
        "cm1_sfc_getcecd_chcq_v1",
        "cm1_sfc_psfc_v1",
        "cm1_sfc_flux_v1",
    ]
    # The whole-domain fields reached the tile, and the roughness getcecd
    # wrote went back to the domain (src/sfcphys.F:155 reads it next step).
    assert np.all(surface["tsk"] == np.float32(299.28))
    y0, y1, x0, x1 = spec.owned_rect("mass")
    assert np.all(
        instance.surface_state["znt"][y0:y1, x0:x1] == np.float32(0.0321)
    )
    # The flux kernel was handed the lowest water-vapour level, not the array.
    assert (nk, nj, ni) in carriers["cm1_sfc_flux_v1"]


def test_a_drag_only_run_launches_no_surface_flux(monkeypatch):
    """``isfcflx=0`` leaves the surface energy budget alone entirely."""

    from cm1core import cuda_surface_drag as drag

    monkeypatch.setitem(
        __import__("sys").modules,
        "cupy",
        type(
            "cupy",
            (),
            {"uint32": np.uint32, "float32": np.float32, "ndarray": np.ndarray},
        ),
    )
    g = CartesianGridGeometry(31, 23, 12, 500, 500, 250)
    instance = _shell(g, bbc=3, sfcmodel=1, cnstcd=0.0035)
    launches: list[str] = []
    instance.surface_compiled = SimpleNamespace(
        kernels=tuple(
            (lambda index: lambda _g, _b, _a, *, stream=None: launches.append(
                drag.SURFACE_DRAG_KERNEL_NAMES[index]
            ))(index)
            for index in range(len(drag.SURFACE_DRAG_KERNEL_NAMES))
        ),
        source_sha256="e" * 64,
        target_arch="86",
    )
    spec = instance.plan.specs[0]
    ni, nj, nk = spec.cnx, spec.cny, g.nz
    shapes = drag.surface_drag_shapes(ni, nj, nk)
    surface = {
        name: np.zeros(shapes[name], np.float32)
        for name in (*drag.SURFACE_STATE_NAMES, *drag.SURFACE_PERSISTENT_NAMES)
    }
    tile = {
        name: np.zeros(shapes[name], np.float32)
        for name in ("u3d", "v3d", "terrain_msl")
    }
    work = {"zh": np.zeros((nk, nj, ni), np.float32)}
    instance._surface_layer(tile, work, surface, ni, nj, nk, spec, None)
    instance._surface_enthalpy_flux(tile, surface, ni, nj, nk)
    assert launches == [
        "cm1_sfc_ground_relative_wind_v1",
        "cm1_sfc_getcecd_cnstcd_v1",
    ]
