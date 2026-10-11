"""CPU storage/cadence checks; these do not claim numerical CUDA validation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cuda_cartesian_dynamics import CartesianDynamics, cartesian_dynamics_allocation_plan
from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes
from cm1core.gm1_idealized_residency import IdealizedSource
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]


class Stream:
    def __init__(self):
        self.depth = 0
        self.ptr = 1

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *_):
        self.depth -= 1

    def synchronize(self):
        pass


class Event:
    def __init__(self, **_):
        pass

    def record(self, stream):
        self.stream = stream


class CpuStorage:
    """Only array storage is emulated, never kernels or scientific outputs."""

    float32 = np.float32
    float64 = np.float64
    cuda = SimpleNamespace(Device=lambda: SimpleNamespace(id=0), Event=Event)

    def __init__(self, stream):
        self.stream = stream
        self.allocations = 0

    def empty(self, shape, *, dtype):
        assert self.stream.depth > 0
        self.allocations += 1
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype):
        assert self.stream.depth > 0
        self.allocations += 1
        return np.zeros(shape, dtype=dtype)

    def asarray(self, value, *, dtype, order):
        assert self.stream.depth > 0
        self.allocations += 1
        return np.array(value, dtype=dtype, order=order, copy=True)

    def get_default_memory_pool(self):
        return SimpleNamespace(free_all_blocks=lambda: None)


class MetadataOwner:
    def __init__(self, _library, _sounding, *, iinit, geometry, **_selectors):
        nz, ny, nx = geometry.shape_zyx
        g = CartesianGridGeometry(nx, ny, nz, *geometry.spacing_m)
        shapes = terrain_open_sound_shapes(nx, ny, nz)
        self.groups = {"terrain_open": {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}}
        terrain = self.groups["terrain_open"]
        for name in (
            "pi0", "th0", "qv0", "rho0", "prs0", "rr0", "tha", "th3d", "rf0",
            # init3d.F:2386-2466's initialized pressure and density, which the
            # runtime reads instead of the base-state prs0/rho0/rr0/rf0.
            "prs", "rho", "rr", "rf",
        ):
            terrain[name] = np.zeros(g.padded_shape("mass", halo_xy=3, halo_z=1), dtype=np.float32)
        self.groups["moist"] = {name: np.zeros((6, *g.padded_shape("mass", halo_xy=3, halo_z=1)), dtype=np.float32) for name in ("qa", "q3d")}
        self.groups["terrain_static"] = {name: np.zeros(shape, dtype=np.float32) for name, shape in terrain_static_shapes(ni=nx, nj=ny, nk=nz).items()}
        self.groups["base_wind"] = {"u0": np.zeros(g.u_shape, dtype=np.float32), "v0": np.zeros(g.v_shape, dtype=np.float32)}

    def close(self):
        pass


@pytest.mark.parametrize("nz", (8, 237))
def test_phase_storage_retains_canonical_fields_and_distinct_time_levels(monkeypatch, nz):
    config = RuntimeConfig(CartesianGridGeometry(9, 11, nz, 75, 75, 75), memory_budget_bytes=200_000_000)
    stream = Stream()
    storage = CpuStorage(stream)
    monkeypatch.setattr(CartesianDynamics, "_compile_operators", lambda self: (setattr(self, "moist_compiled", object()), setattr(self, "open_compiled", object())))
    monkeypatch.setattr(CartesianDynamics, "_initial_dry_mass", lambda self: 1.0)
    # The startup publication added in f23a533 launches real kernels, so it is
    # stubbed here exactly as the dry-mass ledger already is: this case proves
    # phase storage, aliasing and time-level separation, never thermodynamics.
    # The stub is marked so a caller cannot mistake it for a real publication,
    # and test_startup_publication_is_not_optional below replaces what the stub
    # gives up.
    monkeypatch.setattr(
        CartesianDynamics,
        "_publish_startup_thermodynamics",
        lambda self: {"schema": "stub", "performed": False},
    )
    with IdealizedSource("unused", b"metadata", config.grid, owner_factory=MetadataOwner) as source:
        state = CartesianDynamics(source, config, cupy=storage, stream=stream)
    for baseline, current, stagger in (("ua", "u3d", "u"), ("va", "v3d", "v"), ("wa", "w3d", "w")):
        assert state.fields[current].shape == config.grid.physical_shape(stagger)
        assert np.shares_memory(state.fields[current], state.arrays[current])
        assert np.shares_memory(state.fields[baseline], state.arrays[baseline])
        assert not np.shares_memory(state.fields[baseline], state.fields[current])
        assert not state.fields[current].flags.c_contiguous
    assert state.scalar_rk.theta_current is state.fields["th3d"]
    assert state.scalar_rk.moisture_a is state.fields["qa"]
    assert state.scalar_rk.moisture_held is state.held["qten"]
    assert state.fields["rho0"] is state.metrics["rho0_3d"]
    for name in ("uh", "vh", "ruf", "rvf"):
        assert np.all(state.arrays[name].view(np.uint32) == np.float32(1.0).view(np.uint32))
    persistent = state.registry.registered_allocation_bytes
    state._ensure_arrays(state._padded_shapes)
    assert state.registry.registered_allocation_bytes > persistent
    before = state.fields["q3d"]
    state.release_phase_workspace("acoustic")
    assert state.registry.registered_allocation_bytes == persistent
    state._ensure_scalar_workspace()
    assert state.scalar_rk.advection.shape == config.grid.mass_shape
    assert state.registry.registered_allocation_bytes > persistent
    state.release_phase_workspace("scalar")
    assert state.registry.registered_allocation_bytes == persistent
    assert state.fields["q3d"] is before
    assert state.scalar_rk.moisture_current is before
    assert state.scalar_rk.advection is None


def test_requested_grid_plan_counts_owned_winds_once_and_fits_without_allocation():
    config = RuntimeConfig(CartesianGridGeometry(400, 400, 237, 75, 75, 75))
    plan = cartesian_dynamics_allocation_plan(config)
    assert plan["persistent_payload_bytes"] == 9_330_274_376
    assert plan["acoustic_phase_payload_bytes"] == 12_592_163_992
    assert plan["scalar_phase_payload_bytes"] == 11_181_386_072
    # The four volume-persistence fields belonging to the physics owner are
    # budgeted separately, and must still fit beside the acoustic phase.
    assert plan["acoustic_phase_payload_bytes"] + 4 * 237 * 400 * 400 * 4 < 13_429_506_048


def test_cadence_refuses_before_touching_a_numerical_operator():
    state = object.__new__(CartesianDynamics)
    state.stream = Stream()
    state._expected_phase = ("held", 1)
    with pytest.raises(ValueError, match="expects"):
        state.pre_acoustic(2)
    with pytest.raises(ValueError, match="expects"):
        state.scalar_moist(1)


def test_startup_publication_is_not_optional() -> None:
    """CM1 getset runs before the ledger; no flag may skip it.

    A guard that can be turned off is a guard that will be off.  Assert the
    call site exists, is unconditional, and precedes the dry-mass reduction,
    so a future edit cannot quietly reorder or gate it.
    """

    source = (ROOT / "src/cm1core/cuda_cartesian_dynamics.py").read_text(encoding="utf-8")
    publish = source.index("self.startup_receipt = self._publish_startup_thermodynamics()")
    ledger = source.index("self.mass1_f64 = self._initial_dry_mass()")
    assert publish < ledger, "startup thermodynamics must precede the dry-mass ledger"
    statement = source[source.rindex("\n", 0, publish) + 1 : publish]
    assert statement.strip() == "", "the startup publication must be unconditional"
    assert source.count("_publish_startup_thermodynamics(") == 2
