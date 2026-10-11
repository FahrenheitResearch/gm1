"""The Cartesian storm lane runs any admitted moisture component, on host doubles.

No CUDA context is opened and no kernel is compiled.  A numpy-backed stand-in
for CuPy (``_DeviceArray`` exposes ``.data.ptr`` the way a CuPy array does) is
installed as ``cupy`` so the real launch validators -- the startup density
publication, the eqtset=2 stage-term bridge, the post-physics EOS/qa finish and
the Cartesian finish's block loop -- run their own shape, pointer and receipt
checks on a Morrison axis.  Kernels are recording callables.

What is proven here, per scheme:

* the runtime warms the component the document selects, with that scheme's
  own carrier set, instead of refusing every non-WSM6 run at startup;
* the startup density, eqtset=2 stage terms, pdefq and the post-physics EOS
  read CM1's six mass rows and never a number concentration
  (src/init3d.F:2393-2396, src/solve2.F:317-412, src/mp_driver.F:505-520,
  :603-617);
* a Morrison physics block is staged with all ten rows, handed to the column
  component whole, all ten rows are drained back to ``q3d``, ``qa`` keeps the
  step's start until solve3, and solve3 publishes all ten rows to ``qa``
  (src/cm1.F:3416,3432; src/solve3.F:624-643);
* the restart carries the component's own persistence;
* nest injection maps species by CM1 name and refuses what it cannot fill;
* WSM6 records, carriers and restart profiles are unchanged.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.errors import CM1Refusal
from cm1core.grid_geometry import CartesianGridGeometry
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.moisture_species import (
    CM1_MASS_PREFIX_NAMES,
    CM1_SPECIES_CLASSIFICATION_AUTHORITY,
    MORRISON_PROFILE,
    MOISTURE_COMPONENT_IDS,
    THOMPSON_AEROSOL_PROFILE,
    THOMPSON_PROFILE,
    WSM6_PROFILE,
    MoistureSpeciesProfile,
    _mass,
    _number,
    moisture_species_profile,
    require_cm1_mass_prefix,
)


# ------------------------------------------------------------ host doubles


class _DeviceArray(np.ndarray):
    """A numpy array that answers ``.data.ptr`` and ``.get()`` like CuPy's."""

    @property
    def data(self):  # type: ignore[override]
        return SimpleNamespace(ptr=int(self.__array_interface__["data"][0]))

    def get(self, stream=None):
        return np.asarray(self)


def _dev(array: np.ndarray) -> _DeviceArray:
    return np.ascontiguousarray(array).view(_DeviceArray)


class _Event:
    def __init__(self, **_):
        pass

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


class _Stream:
    ptr = 4242

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def synchronize(self):
        pass


def _fake_cupy() -> types.ModuleType:
    module = types.ModuleType("cupy")
    module.ndarray = _DeviceArray
    module.float32 = np.float32
    module.float64 = np.float64
    module.uint32 = np.uint32
    module.uint64 = np.uint64
    module.dtype = np.dtype
    module.zeros = lambda shape, dtype=np.float32: _dev(np.zeros(shape, dtype=dtype))
    module.empty = lambda shape, dtype=np.float32: _dev(np.zeros(shape, dtype=dtype))
    module.ones = lambda shape, dtype=np.float32: _dev(np.ones(shape, dtype=dtype))
    module.full = lambda shape, value, dtype=np.float32: _dev(np.full(shape, value, dtype=dtype))
    module.arange = lambda n, dtype=np.float32: _dev(np.arange(n, dtype=dtype))
    module.copyto = np.copyto
    module.multiply = np.multiply
    module.add = np.add
    module.subtract = np.subtract
    module.divide = np.divide
    module.asnumpy = np.asarray
    pool = SimpleNamespace(free_all_blocks=lambda: None)
    module.get_default_memory_pool = lambda: pool
    module.cuda = SimpleNamespace(
        Event=_Event,
        Device=lambda *_: SimpleNamespace(id=0),
        get_current_stream=lambda: _Stream(),
        runtime=SimpleNamespace(memGetInfo=lambda: (6 * 1024**3, 8 * 1024**3)),
    )
    return module


@pytest.fixture
def cupy(monkeypatch):
    module = _fake_cupy()
    monkeypatch.setitem(sys.modules, "cupy", module)
    return module


class _Recorder:
    """A kernel: records every launch's argument tuple."""

    def __init__(self, action=None):
        self.calls: list[tuple] = []
        self.action = action

    def __call__(self, grid, block, args, stream=None):
        self.calls.append(args)
        if self.action is not None:
            self.action(args)


def _grid(nz: int = 8) -> CartesianGridGeometry:
    # RuntimeConfig requires the fifth-order stencil: nx, ny >= 7 and nz >= 8.
    return CartesianGridGeometry(9, 8, nz, 100.0, 100.0, 100.0)


def _config(component: str, **changes) -> RuntimeConfig:
    return RuntimeConfig(grid=_grid(), microphysics=component, **changes)  # type: ignore[arg-type]


# ------------------------------------------------ species classification


@pytest.mark.parametrize("component", MOISTURE_COMPONENT_IDS)
def test_every_admitted_axis_leads_with_cm1s_six_mass_rows(component: str) -> None:
    profile = moisture_species_profile(component)
    assert require_cm1_mass_prefix(profile) == 6
    assert profile.names[:6] == CM1_MASS_PREFIX_NAMES
    # Every row past the prefix is a number concentration, and only those.
    assert profile.mass_indices == tuple(range(6))
    assert profile.number_indices == tuple(range(6, profile.numq))
    assert all(profile.scalars[i].unit == "#/kg" for i in profile.number_indices)
    # pdefq repairs exactly the mass rows (src/mp_driver.F:505-520).
    assert profile.repaired_scalars == 6


def test_morrison_classification_is_cm1s_ptype5_table() -> None:
    # src/param.F:3420-3473: nqv=1, nql1..nql2=2..3, nqs1..nqs2=4..6, nnc1..nnc2=7..10.
    p = MORRISON_PROFILE
    assert (p.nqv, p.nql1, p.nql2, p.nqs1, p.nqs2, p.nnc1, p.nnc2) == (1, 2, 3, 4, 6, 7, 10)
    assert p.mass_names == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert p.number_names == ("nci", "ncs", "ncr", "ncg")
    assert p.condensate_indices == (1, 2, 3, 4, 5)
    for key in ("eos_buoyancy_loading", "post_physics_eos", "positive_definite_repair",
                "calcnm_total_water", "startup_density", "dry_mass_ledger"):
        assert CM1_SPECIES_CLASSIFICATION_AUTHORITY[key].startswith("src/")


def test_an_axis_with_a_count_inside_the_mass_window_is_refused_by_name() -> None:
    bad = MoistureSpeciesProfile(
        component_id="count-in-mass-window", cm1_ptype=None,
        scalars=(_mass("qv"), _number("nci"), _mass("qc"), _mass("qr"),
                 _mass("qi"), _mass("qs"), _mass("qg")),
        nqv=1, nql1=3, nql2=4, nqs1=5, nqs2=7, nnc1=2, nnc2=2,
        iice=1, idm=1, cm1_eqtset=2,
    )
    with pytest.raises(CM1Refusal) as refused:
        require_cm1_mass_prefix(bad)
    assert "GM1-MOISTURE-MASS-PREFIX" in str(refused.value)
    assert "water loading" in str(refused.value)


# ------------------------------------------------------ warm component


def _warm(monkeypatch, cupy, component: str):
    from cm1core import cuda_cartesian_runtime as runtime

    seen: dict[str, object] = {}

    class _Component:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def apply(self, carriers, *, refl_10cm_due):
            seen["carriers"] = carriers
            return SimpleNamespace(ready_event=_Event(), receipt={"component": component})

    def wsm6_factory(**kwargs):
        seen["wsm6_factory"] = kwargs
        return _Component(**kwargs)

    monkeypatch.setattr(runtime, "column_component_class", lambda cid: (seen.setdefault("class", cid), _Component)[1])
    monkeypatch.setattr(runtime, "create_arwen_wsm6_component", wsm6_factory)
    record = runtime._warm_public_component(cupy, _Stream(), _config(component), None)
    return record, seen


def test_the_warm_up_builds_the_selected_morrison_component_not_wsm6(monkeypatch, cupy) -> None:
    from cm1core.arwen_morrison_component import ArwenMorrisonCarriers

    record, seen = _warm(monkeypatch, cupy, "morrison-mp10")
    assert seen["class"] == "morrison-mp10"
    assert "wsm6_factory" not in seen
    assert seen["kwargs"]["component_id"] == "morrison-mp10"
    carriers = seen["carriers"]
    assert isinstance(carriers, ArwenMorrisonCarriers)
    for name in ("nci", "ncs", "ncr", "ncg"):
        assert getattr(carriers, name).shape == (8, 1, 1)
    assert record["model_initialization"] is False


def test_the_warm_up_hands_thompson_its_interface_carrier(monkeypatch, cupy) -> None:
    _, seen = _warm(monkeypatch, cupy, "thompson-mp8")
    assert seen["carriers"].w.shape == (9, 1, 1)
    assert seen["carriers"].ncr.shape == (8, 1, 1)


def test_the_wsm6_warm_up_is_the_construction_it_always_was(monkeypatch, cupy) -> None:
    from cm1core.arwen_wsm6_component import ArwenWsm6Carriers

    _, seen = _warm(monkeypatch, cupy, "wsm6-mp6")
    assert "class" not in seen
    assert set(seen["wsm6_factory"]) == {
        "component_id", "nx", "ny", "nz", "dx_m", "dy_m", "dz_m", "dt_s", "capacity_profile",
    }
    assert isinstance(seen["carriers"], ArwenWsm6Carriers)


# ------------------------------------------------ startup density (qv only)


def _startup(cupy, numq: int, q_rows: int, **kw):
    from cm1core.cuda_cartesian_startup import launch_cuda_cartesian_startup

    nk, nj, ni = 6, 4, 5
    mass, face = (nk, nj, ni), (nk + 1, nj, ni)
    fields = {name: _dev(np.full(mass, 1.0, np.float32))
              for name in ("pi0", "pp3d", "th0", "th3d", "prs", "rho", "rr")}
    fields["rf"] = _dev(np.zeros(face, np.float32))
    fields["q3d"] = _dev(np.arange(q_rows * np.prod(mass), dtype=np.float32).reshape(q_rows, *mass))
    metrics = {"c1": _dev(np.zeros(face, np.float32)), "c2": _dev(np.zeros(face, np.float32))}
    kernel = _Recorder()
    compiled = SimpleNamespace(kernels=(kernel,), target_arch="86", runtime_identity=(),
                               source_sha256="0" * 64)
    try:
        launch_cuda_cartesian_startup(fields, metrics, ni=ni, nj=nj, nk=nk, cupy=cupy,
                                      stream=_Stream(), compiled=compiled,
                                      density_kernel=_Recorder(), numq=numq, **kw)
    except (AttributeError, KeyError, TypeError):
        # Receipt assembly past the launch reads compiled-image attributes a
        # recording double does not model; the launch itself already ran.
        if not kernel.calls:
            raise
    return fields, kernel


def test_startup_density_reads_only_the_vapour_row_of_a_morrison_axis(cupy) -> None:
    fields, kernel = _startup(cupy, numq=10, q_rows=10)
    (args,) = kernel.calls
    vapour = args[4]
    assert vapour.shape == fields["q3d"].shape[1:]
    assert vapour.data.ptr == fields["q3d"].data.ptr
    assert np.array_equal(vapour, fields["q3d"][0])


def test_startup_refuses_an_axis_that_is_not_the_runs_own(cupy) -> None:
    with pytest.raises(ValueError, match=r"q3d shape must be \(10,"):
        _startup(cupy, numq=10, q_rows=6)
    # The default is still the six-row WSM6 contract.
    with pytest.raises(ValueError, match=r"q3d shape must be \(6,"):
        _startup(cupy, numq=6, q_rows=10)


# ---------------------------------------------- eqtset=2 stage terms (mass)


def test_the_stage_terms_get_the_mass_window_and_the_bridge_stays_six_rows(monkeypatch, cupy) -> None:
    from cm1core.cuda_cartesian_dynamics import CartesianDynamics
    from cm1core.cuda_moist_stage_bridge import (
        CudaMoistStageBridgeState,
        launch_cuda_eqtset2_stage_terms,
    )
    from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY as ng
    from cm1core.cuda_terrain_open_boundary import terrain_static_shapes

    ni, nj, nk = 5, 4, 6
    q3d = _dev(np.random.default_rng(1).random((10, nk, nj, ni), dtype=np.float32))
    dynamics = object.__new__(CartesianDynamics)
    dynamics.numq = 10
    dynamics.fields = {"q3d": q3d}
    dynamics.config = SimpleNamespace(profile=MORRISON_PROFILE)
    window = dynamics.mass_prefix_q3d()
    assert window.shape == (6, nk, nj, ni) and window.flags.c_contiguous
    assert window.data.ptr == q3d.data.ptr
    assert np.shares_memory(window, q3d)

    # A WSM6 axis hands over its own array object, exactly as before.
    wsm6 = object.__new__(CartesianDynamics)
    wsm6.numq = 6
    six = _dev(np.zeros((6, nk, nj, ni), np.float32))
    wsm6.fields = {"q3d": six}
    wsm6.config = SimpleNamespace(profile=WSM6_PROFILE)
    assert wsm6.mass_prefix_q3d() is six

    shapes = {
        "thv": (nk + 2, nj + 2, ni + 2), "ppterm": (nk + 2, nj + 2, ni + 2),
        "piadv": (nk + 2, nj + 2, ni + 2), "pp3d": (nk + 2, nj + 2, ni + 2),
        "uten": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "vten": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "wten": (nk + 3, nj + 2 * ng, ni + 2 * ng),
        "rru": (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng),
        "rrv": (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng),
        "rrw": (nk + 3, nj + 2 * ng, ni + 2 * ng),
    }
    arrays = {name: _dev(np.zeros(shape, np.float32)) for name, shape in shapes.items()}
    metrics = {name: _dev(np.zeros(shape, np.float32))
               for name, shape in terrain_static_shapes(ni=ni, nj=nj, nk=nk).items()}
    terms = _Recorder()
    stream = _Stream()

    def state():
        return CudaMoistStageBridgeState(
            ni, nj, nk, True, _dev(np.zeros((nk + 2, nj + 2 * ng, ni + 2 * ng), np.float32)),
            *(_dev(np.zeros((nk, nj, ni), np.float32)) for _ in range(4)),
            stream, 0, SimpleNamespace(kernels=(terms, _Recorder())),
            SimpleNamespace(kernels=tuple(_Recorder() for _ in range(7))),
        )

    base = {name: _dev(np.ones((nk, nj, ni), np.float32)) for name in ("th0", "qv0", "pi0", "th3d")}
    launch_cuda_eqtset2_stage_terms(state(), arrays, metrics, q3d=window, stream=stream, **base)
    (args,) = terms.calls
    # (th0, qv0, qc0, qi0, pi0, th3d, q3d, ...): base.F's qc0/qi0 come after qv0.
    assert args[6].shape[0] == 6 and args[6].data.ptr == q3d.data.ptr
    # The bridge's own contract is unchanged: a ten-row array is refused, so a
    # caller that forgot the window cannot put number rows into the loading.
    with pytest.raises(ValueError, match="q3d has shape"):
        launch_cuda_eqtset2_stage_terms(state(), arrays, metrics, q3d=q3d, stream=stream, **base)


# ------------------------------------------ Cartesian finish, one Morrison step


class _Registry:
    def __init__(self):
        self.live: dict[str, _DeviceArray] = {}

    def allocate(self, name, shape, zero=False, dtype=np.float32):
        value = _dev(np.zeros(shape, dtype=dtype))
        self.live[name] = value
        return value

    def release(self, names):
        for name in names:
            self.live.pop(name, None)


def _finish(monkeypatch, cupy, component: str):
    from cm1core import cuda_cartesian_finish as finish_module
    from cm1core.column_components import column_component_spec
    from cm1core.cuda_gm1_tilestream_post_wsm6 import CompiledGm1TiledPostWsm6

    config = _config(component)
    g = config.grid
    spec = column_component_spec(component)
    profile = config.profile
    rng = np.random.default_rng(7)
    fields = {name: _dev(rng.random(g.mass_shape, dtype=np.float32) + 1.0)
              for name in ("pi0", "th0", "pp3d", "th3d", "rho", "prs")}
    fields["q3d"] = _dev(rng.random((profile.numq, *g.mass_shape), dtype=np.float32))
    fields["qa"] = _dev(np.zeros((profile.numq, *g.mass_shape), np.float32))
    fields["w3d"] = _dev(np.zeros((g.nz + 1, g.ny, g.nx), np.float32))
    # What solve3 equates (solve3.F:600-643) and the rho kernel reads.
    for name in ("ppx", "ppi", "tha", "rr"):
        fields[name] = _dev(np.zeros(g.mass_shape, np.float32))
    for name, shape in (("u3d", g.u_shape), ("ua", g.u_shape), ("v3d", g.v_shape),
                        ("va", g.v_shape), ("wa", g.w_shape), ("rf", g.w_shape)):
        fields[name] = _dev(np.zeros(shape, np.float32))
    synced: list[int] = []
    dynamics = SimpleNamespace(
        fields=fields,
        arrays={"ppx": _dev(np.zeros((g.nz + 2, g.ny + 2, g.nx + 2), np.float32))},
        tke={name: _dev(np.zeros(g.w_shape, np.float32)) for name in ("tkea", "tke3d")},
        metrics={name: _dev(np.zeros((g.nz + 1, g.ny, g.nx), np.float32)) for name in ("zf", "c1", "c2")},
        release_phase_workspace=lambda phase: None,
        sync_padded_from_physical=lambda: synced.append(1),
    )
    registry = _Registry()
    finish = object.__new__(finish_module.CartesianFinish)
    finish.dynamics, finish.config, finish.grid = dynamics, config, g
    finish.cp, finish.stream, finish.registry = cupy, _Stream(), registry
    finish.source_identity_sha256 = "ab" * 32
    finish.capacity_profile = None
    finish.spec, finish.profile, finish.numq = spec, profile, profile.numq
    finish.repaired_scalars = profile.repaired_scalars
    finish.reflectivity_sink = None
    finish.output_dbz = False
    finish.step_receipts = {"pdefq": {"schema": "host-double"}}
    finish.completed_physics_step = 0
    finish.completed_solve3_step = 0
    finish.forecast = SimpleNamespace(kernels=(None, _Recorder(), None, None))
    finish.persistence = {}
    for name in spec.component_persistence:
        shape = g.surface_shape if name in spec.surface_persistence else g.mass_shape
        finish.persistence[name] = _dev(np.zeros(shape, np.float32))

    eos_q3d_rows: list[int] = []

    def eos(args):
        # The real finish kernel's qa loop: s < 6 (post_wsm6 source line 314).
        n, q3d, qa = int(args[0]), args[7], args[9]
        eos_q3d_rows.append(q3d.shape[0])
        qa.reshape(6, n)[:] = q3d.reshape(q3d.shape[0], n)[:6]

    finish.post_compiled = CompiledGm1TiledPostWsm6(
        None, _Recorder(), _Recorder(eos), b"", b"", "86", (),
    )
    # cuda_cartesian_moist_heating: CM1's entry-state cvm before the component
    # and the constant-volume conversion after it, before the sealed finish.
    finish.moist_heating_compiled = SimpleNamespace(
        snapshot_kernel=_Recorder(), heating_kernel=_Recorder(), probe_kernel=None,
        cubin_sha256="0" * 64, options=(),
    )

    from cm1core.cuda_gm1_tilestream_column import (
        TILESTREAM_RECORD_IDENTITY,
        CudaGm1TiledColumnLaunch,
    )
    from cm1core.cuda_gm1_tilestream_post_wsm6 import _canonical

    handed: list[dict] = []

    def column_launch(spec_, values, *, nx, ny, nk, block_index, active_nx, active_ny,
                      source_identity_sha256, stream, **_):
        handed.append({name: value.shape for name, value in values.items()})
        # The scheme advances every transported row; mark each one so the
        # publication can be checked row by row.
        for row in range(values["q3d"].shape[0]):
            values["q3d"][row] = np.float32(1000 + row)
        identity = TILESTREAM_RECORD_IDENTITY[spec_.component_id]
        receipt = {
            **identity, "fixed_shape_zyx": [nk, ny, nx], "active_shape_yx": [active_ny, active_nx],
            "block_index": block_index, "source_identity_sha256": source_identity_sha256,
            "stream_pointer": int(stream.ptr), "same_stream_no_host_sync": True,
            "pdefq_receipt_sha256": "cd" * 32,
        }
        receipt["operation_token_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        return CudaGm1TiledColumnLaunch(_Event(), None, None, dict(values), receipt)

    monkeypatch.setattr(finish_module, "launch_cuda_gm1_tiled_column_block", column_launch)
    return finish, fields, handed, eos_q3d_rows


def test_one_morrison_physics_step_stages_runs_and_publishes_all_ten_rows(monkeypatch, cupy) -> None:
    finish, fields, handed, eos_rows = _finish(monkeypatch, cupy, "morrison-mp10")
    finish.microphysics(step=1)
    receipt = finish.step_receipts["physics"]
    assert receipt["component_id"] == "morrison-mp10"
    assert receipt["numq"] == 10 and receipt["cm1_ptype"] == 5
    assert handed and all(shapes["q3d"][0] == 10 for shapes in handed)
    assert all("nc" in shapes and "effr" in shapes for shapes in handed)
    # The EOS finish kernel never saw a number row.
    assert eos_rows and set(eos_rows) == {6}
    # Every one of numq rows is drained back to q3d, mass and number alike,
    # and qa still holds the step's start: CM1 runs mp_driver before solve3
    # (cm1.F:3416,3432), and the qv budget's actual tendency reads (q3d-qa)/dt
    # here.  The drain used to publish qa too, which made that tendency zero.
    for row in range(10):
        assert np.all(fields["q3d"][row] == np.float32(1000 + row))
        assert not np.any(fields["qa"][row])
    # solve3.F:624-643: every one of numq rows reaches qa, mass and number alike.
    finish.solve3(step=1)
    for row in range(10):
        assert np.all(fields["qa"][row] == np.float32(1000 + row))


@pytest.mark.parametrize(
    "component", ["wsm6-mp6", "morrison-mp10", "thompson-mp8", "thompson-aerosol-mp28"],
)
def test_every_component_is_heated_at_constant_volume_before_the_rebuild(monkeypatch, cupy, component) -> None:
    """src/mp_driver.F:603-617's fixed-rho rebuild gets CM1's eqtset=2 heating.

    The component heats at constant pressure; the conversion kernel runs after
    it and before the sealed finish, on the same block, for every component.
    """

    from cm1core import cuda_cartesian_moist_heating as mh

    finish, fields, handed, eos_rows = _finish(monkeypatch, cupy, component)
    order: list[str] = []
    heating = finish.moist_heating_compiled
    heating.snapshot_kernel.action = lambda args: order.append("rcvm")
    heating.heating_kernel.action = lambda args: order.append(f"heat:{int(args[1])}:{int(args[2])}")
    finish.post_compiled.finish_kernel.action = (
        lambda args, previous=finish.post_compiled.finish_kernel.action: (order.append("rebuild"), previous(args))
    )
    finish.microphysics(step=1)
    selection = mh.moist_heating_selection(component)
    assert order == ["rcvm", f"heat:{selection.mode}:{selection.cpm_kind}", "rebuild"]
    receipt = finish.step_receipts["physics"]
    assert receipt["moist_heating"]["cm1_configuration"] is (component != "wsm6-mp6")
    assert all("moist_heating" in block for block in receipt["blocks"])


@pytest.mark.parametrize("component", ["thompson-mp8", "thompson-aerosol-mp28"])
def test_the_other_double_moment_axes_take_the_same_path(monkeypatch, cupy, component) -> None:
    finish, fields, handed, eos_rows = _finish(monkeypatch, cupy, component)
    finish.microphysics(step=1)
    numq = moisture_species_profile(component).numq
    assert set(eos_rows) == {6}
    assert all(shapes["q3d"][0] == numq for shapes in handed)
    assert not np.any(fields["qa"])
    finish.solve3(step=1)
    assert np.all(fields["qa"][numq - 1] == np.float32(1000 + numq - 1))


def test_the_post_physics_receipt_names_a_non_wsm6_component(monkeypatch, cupy) -> None:
    from cm1core import cuda_cartesian_finish as finish_module

    records = []

    def spying(name):
        real = getattr(finish_module, name)

        def spy(*args, **kwargs):
            result = real(*args, **kwargs)
            records.append(result.receipt)
            return result

        monkeypatch.setattr(finish_module, name, spy)

    spying("launch_cuda_gm1_tiled_post_wsm6_eos_qa")
    spying("launch_cuda_gm1_tiled_post_column_eos_qa")
    finish, *_ = _finish(monkeypatch, cupy, "morrison-mp10")
    finish.microphysics(step=1)
    assert records[0]["column_component_id"] == "morrison-mp10"
    assert "morrison-mp10" in records[0]["source_order"][1]

    records.clear()
    finish, *_ = _finish(monkeypatch, cupy, "wsm6-mp6")
    finish.microphysics(step=1)
    assert records[0]["status"] == "ENQUEUED_POST_WSM6_EOS_AND_PHYSICAL_QA"
    assert "column_component_id" not in records[0]
    assert "mass_prefix_only" not in records[0]
    assert records[0]["source_order"][1] == "accepted tile-local public WSM6 on the same stream"


def test_a_relabelled_launch_receipt_is_still_refused(monkeypatch, cupy) -> None:
    from cm1core import cuda_cartesian_finish as finish_module

    finish, *_ = _finish(monkeypatch, cupy, "morrison-mp10")
    launch = finish_module.launch_cuda_gm1_tiled_column_block

    def relabelled(*args, **kwargs):
        result = launch(*args, **kwargs)
        receipt = dict(result.receipt)
        receipt["status"] = "ENQUEUED_TILE_LOCAL_PUBLIC_WSM6"  # Morrison schema, WSM6 status
        return type(result)(result.ready_event, None, None, result.arrays, receipt)

    monkeypatch.setattr(finish_module, "launch_cuda_gm1_tiled_column_block", relabelled)
    with pytest.raises(ValueError, match="column launch identity/order differs"):
        finish.microphysics(step=1)


def test_pdefq_and_the_dry_mass_ledger_read_only_mass_rows(monkeypatch, cupy) -> None:
    from cm1core import cuda_cartesian_finish as finish_module

    finish, fields, *_ = _finish(monkeypatch, cupy, "morrison-mp10")
    g = finish.grid
    finish.dynamics.arrays = {"uh": _dev(np.ones(g.nx + 6, np.float32)),
                              "vh": _dev(np.ones(g.ny + 6, np.float32))}
    finish.dynamics.metrics["rmh"] = _dev(np.ones(g.mass_shape, np.float32))
    finish.eos_invalid = _dev(np.zeros((1,), np.uint32))
    finish.asq = _dev(np.zeros((6,), np.float64))
    finish.pdefq_compiled = object()
    finish.pdefq_workspace = object()
    seen = {}

    def pdefq(arrays, **kwargs):
        seen.update(arrays)
        return SimpleNamespace(receipt={"schema": "host-double"}, ready_event=_Event())

    monkeypatch.setattr(finish_module, "launch_cuda_wsm6_pdefq_pdscheme1", pdefq)
    monkeypatch.setattr(finish_module, "create_cuda_wsm6_pdefq_workspace", lambda **_: object())
    finish.pdefq()
    assert seen["q3d"].shape[0] == 6
    assert seen["q3d"].data.ptr == fields["q3d"].data.ptr
    mass = finish._mass_arrays()
    assert mass["qv"].data.ptr == fields["q3d"].data.ptr and mass["qv"].shape == g.mass_shape
    assert not any(getattr(v, "ndim", 0) == 4 for v in mass.values())


# ---------------------------------------------------------------- restart


def test_a_morrison_restart_carries_morrisons_own_persistence() -> None:
    from cm1core.arwen_morrison_component import MORRISON_SPEC
    from cm1core.cartesian_restart import cartesian_restart_profile

    profile = cartesian_restart_profile(_config("morrison-mp10"))
    carriers = {c.name: c for c in profile.carriers}
    component = [c.key for c in profile.carriers if c.group == "component"]
    assert component == list(MORRISON_SPEC.component_persistence)
    assert "effr" in component and "nc" in component
    for name in ("fields.q3d", "fields.qa", "held.qten"):
        assert carriers[name].shape[0] == 10
        assert carriers[name].stagger == "mass-species"
        assert "nci,ncs,ncr,ncg" in carriers[name].meaning
    assert carriers["fields.q3d"].units == "kg kg-1; kg-1"
    assert carriers["component.nc"].units == "kg-1"


def test_the_wsm6_restart_profile_is_unchanged() -> None:
    from cm1core.arwen_wsm6_component import WSM6_RESTART_REQUIRED
    from cm1core.cartesian_restart import cartesian_restart_profile

    profile = cartesian_restart_profile(_config("wsm6-mp6"))
    carriers = {c.name: c for c in profile.carriers}
    assert [c.key for c in profile.carriers if c.group == "component"] == list(WSM6_RESTART_REQUIRED)
    assert carriers["fields.q3d"].meaning == "current qv,qc,qr,qi,qs,qg mixing ratios"
    assert carriers["fields.qa"].meaning == "previous-step qv,qc,qr,qi,qs,qg mixing ratios"
    assert carriers["fields.q3d"].units == "kg kg-1"
    assert carriers["held.qten"].meaning == "held moisture tendencies"


# ------------------------------------------------------------ nest species


def test_a_wsm6_parent_into_a_morrison_child_is_refused_by_species_name() -> None:
    from cm1core.nest_downscale import plan_species_transfer

    with pytest.raises(CM1Refusal) as refused:
        plan_species_transfer(WSM6_PROFILE, MORRISON_PROFILE)
    text = str(refused.value)
    assert "CM1-NEST-DOWNSCALE-SPECIES" in text
    assert "nci, ncs, ncr, ncg" in text
    assert "scheme-lambda-bound" in text  # the remedy names the scheme's own relation


def test_the_scheme_lambda_bound_is_explicit_cited_and_only_for_morrison() -> None:
    from cm1core.nest_downscale import SCHEME_LAMBDA_BOUND, plan_species_transfer

    plan = plan_species_transfer(WSM6_PROFILE, MORRISON_PROFILE, missing_numbers=SCHEME_LAMBDA_BOUND)
    sources = {row["name"]: row["source"] for row in plan["rows"]}
    assert all(sources[n] == "parent" for n in CM1_MASS_PREFIX_NAMES)
    assert all(sources[n] == "zero-then-scheme-lambda-bound" for n in ("nci", "ncs", "ncr", "ncg"))
    authority = plan["missing_number_concentrations"]
    assert "morrison.F:1876-1885" in authority["cm1"]
    assert "morr_bound_one" in authority["arwen"]
    # Classic Thompson has no such relation admitted here.
    with pytest.raises(CM1Refusal, match="no slope-bound relation"):
        plan_species_transfer(WSM6_PROFILE, THOMPSON_PROFILE, missing_numbers=SCHEME_LAMBDA_BOUND)
    with pytest.raises(CM1Refusal):
        plan_species_transfer(WSM6_PROFILE, MORRISON_PROFILE, missing_numbers="diagnostic")


def test_names_not_positions_map_between_axes() -> None:
    from cm1core.nest_downscale import plan_species_transfer

    # Aerosol Thompson carries nci,ncr,ncc,... ; classic Thompson nci,ncr.
    plan = plan_species_transfer(THOMPSON_AEROSOL_PROFILE, THOMPSON_PROFILE)
    rows = {row["name"]: row["parent_row"] for row in plan["rows"]}
    assert rows == {name: THOMPSON_AEROSOL_PROFILE.index(name) for name in THOMPSON_PROFILE.names}
    assert plan["parent_species_not_transferred"] == ["ncc", "nwfa", "nifa"]
    # Morrison's ncr is row 8; aerosol Thompson's is row 7 -- matched by name.
    plan = plan_species_transfer(MORRISON_PROFILE, MORRISON_PROFILE)
    assert [row["parent_row"] for row in plan["rows"]] == list(range(10))


def _nest_case(tmp_path: Path, parent_rows: dict[str, np.ndarray], child_rows: int):
    from cm1core.nest_geometry import NestGeometry
    from cm1core.nest_vertical import plan_vertical_refinement
    from test_nest_vertical_refinement import _write_arena, small_parent

    geometry = NestGeometry(
        parent_nx=20, parent_ny=18, parent_dx_m=250.0, parent_dy_m=250.0,
        refinement_ratio=5, time_refinement_ratio=5, parent_i0=6, parent_j0=5,
        parent_span_x=8, parent_span_y=7, spec_zone=1, relax_zone=4, feedback=0,
    )
    levels = small_parent()
    plan = plan_vertical_refinement(levels, levels)
    parent_table = _write_arena(tmp_path / "parent.f32", {
        name: ("mass", rows) for name, rows in parent_rows.items()
    })
    child_table = _write_arena(tmp_path / "child.f32", {
        "fields.q3d": ("mass", np.zeros((child_rows, levels.nk, geometry.ny, geometry.nx), np.float32)),
    })
    return geometry, levels, plan, parent_table, child_table


def _child_rows(tmp_path: Path, table) -> np.ndarray:
    entry = table["fields.q3d"]
    arena = np.fromfile(tmp_path / "child.f32", dtype="<f4")
    count = int(np.prod(entry["shape"]))
    return arena[entry["offset"] // 4: entry["offset"] // 4 + count].reshape(entry["shape"])


def test_a_morrison_frame_injects_its_number_concentrations_by_name(tmp_path: Path) -> None:
    from cm1core.nest_downscale import plan_species_transfer
    from tools.nest_downscale_init import transfer_carriers
    from test_nest_vertical_refinement import small_parent

    pz = small_parent().nk
    rng = np.random.default_rng(3)
    mass = rng.uniform(0.0, 1.0e-3, (6, pz, 18, 20)).astype(np.float32)
    numbers = (np.arange(4, dtype=np.float32)[:, None, None, None] + 1.0) * np.ones((4, pz, 18, 20), np.float32) * 1.0e4
    geometry, levels, plan, parent_table, child_table = _nest_case(
        tmp_path, {"q3d": mass, "nc3d": numbers}, child_rows=10,
    )
    species = plan_species_transfer(MORRISON_PROFILE, MORRISON_PROFILE)
    written, _ = transfer_carriers(
        parent_table=parent_table, parent_arena=tmp_path / "parent.f32",
        child_table=child_table, child_arena=tmp_path / "child.f32",
        geometry=geometry, parent_nz=pz, refinement=plan, species_plan=species,
    )
    child = _child_rows(tmp_path, child_table)
    for row, name in enumerate(("nci", "ncs", "ncr", "ncg")):
        # A constant parent field downscales to that constant.
        assert np.all(child[6 + row] == np.float32((row + 1) * 1.0e4)), name
    assert np.all(child[:6] >= 0.0) and float(child[:6].max()) > 0.0
    assert written[0]["species"] == list(MORRISON_PROFILE.names)


def test_the_silent_zero_fill_of_a_wider_child_axis_is_refused(tmp_path: Path) -> None:
    from tools.nest_downscale_init import transfer_carriers
    from test_nest_vertical_refinement import small_parent

    pz = small_parent().nk
    mass = np.ones((6, pz, 18, 20), np.float32) * 1.0e-3
    geometry, levels, plan, parent_table, child_table = _nest_case(tmp_path, {"q3d": mass}, child_rows=10)
    # The shipped defect: a 6-row WSM6 frame into a 10-row Morrison restart
    # with no species map left nci..ncg at zero and said nothing.
    with pytest.raises(SystemExit, match="carries 10 q3d rows and the parent frame supplies 6"):
        transfer_carriers(
            parent_table=parent_table, parent_arena=tmp_path / "parent.f32",
            child_table=child_table, child_arena=tmp_path / "child.f32",
            geometry=geometry, parent_nz=pz, refinement=plan,
        )


def test_a_frame_with_number_rows_needs_a_species_map(tmp_path: Path) -> None:
    from tools.nest_downscale_init import transfer_carriers
    from test_nest_vertical_refinement import small_parent

    pz = small_parent().nk
    geometry, levels, plan, parent_table, child_table = _nest_case(
        tmp_path,
        {"q3d": np.ones((6, pz, 18, 20), np.float32), "nc3d": np.ones((4, pz, 18, 20), np.float32)},
        child_rows=6,
    )
    with pytest.raises(SystemExit, match="nc3d"):
        transfer_carriers(
            parent_table=parent_table, parent_arena=tmp_path / "parent.f32",
            child_table=child_table, child_arena=tmp_path / "child.f32",
            geometry=geometry, parent_nz=pz, refinement=plan,
        )


def test_the_lambda_bound_option_writes_exact_zero_numbers(tmp_path: Path) -> None:
    from cm1core.nest_downscale import SCHEME_LAMBDA_BOUND, plan_species_transfer
    from tools.nest_downscale_init import transfer_carriers
    from test_nest_vertical_refinement import small_parent

    pz = small_parent().nk
    mass = np.ones((6, pz, 18, 20), np.float32) * 1.0e-3
    geometry, levels, plan, parent_table, child_table = _nest_case(tmp_path, {"q3d": mass}, child_rows=10)
    species = plan_species_transfer(WSM6_PROFILE, MORRISON_PROFILE, missing_numbers=SCHEME_LAMBDA_BOUND)
    transfer_carriers(
        parent_table=parent_table, parent_arena=tmp_path / "parent.f32",
        child_table=child_table, child_arena=tmp_path / "child.f32",
        geometry=geometry, parent_nz=pz, refinement=plan, species_plan=species,
    )
    child = _child_rows(tmp_path, child_table)
    assert np.all(child[6:] == 0.0)
    assert np.allclose(child[:6], 1.0e-3)


def test_a_live_nest_refuses_two_schemes() -> None:
    from cm1core.nest_live import NestDriver

    def domain(dt, component, numq):
        return SimpleNamespace(config=SimpleNamespace(dt_s=dt, microphysics=component),
                               dynamics=SimpleNamespace(numq=numq))

    with pytest.raises(ValueError, match="same physics.component"):
        NestDriver(parent=domain(5.0, "wsm6-mp6", 6), child=domain(1.0, "morrison-mp10", 10),
                   geometry=SimpleNamespace(time_refinement_ratio=5), cupy=None, stream=None)
