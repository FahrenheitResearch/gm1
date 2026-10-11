"""``grid.stretch_z`` from TOML text to the operator that divides by it.

The Cartesian lane used to refuse every ``param.F:7814-8139`` branch with
``CM1-CARTESIAN-UNIFORM-DZ-OPERATORS`` because four of its operators rebuilt
the column from a scalar ``dz``.  Those four now select by
``grid.stretch_z``: the uniform family for ``param.F:7789-7811`` -- where
``param.F:8170`` makes every ``mh`` word exactly one and ``adv.F:572``'s
``rdz*mh`` collapses to ``1/dz`` -- and the profile-bound family otherwise,
which multiplies by ``rdsf(k)`` (``adv.F:551``, ``adv.F:710``,
``init_terrain.F:292-298``) and carries no ``dz`` argument at all.

These checks run entirely on the host.  The kernel arithmetic is the business
of the sealed stretched-operator oracles; what is proved here is that a
configuration file reaches it.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.cartesian_door import (
    CARTESIAN_OPERATOR_FAMILIES,
    build_runtime_config,
    cartesian_operator_family,
)
from cm1core.cm1_vertical_grid import VerticalLevels, build_vertical_levels
from cm1core.config import load_config
from cm1core.cuda_cartesian_dynamics import CartesianDynamics, cartesian_dynamics_allocation_plan
from cm1core.cuda_psolver3_terrain_open import (
    COLUMN_DIAGNOSTIC_NAMES,
    terrain_open_acoustic_workspace_shapes,
    terrain_open_sound_shapes,
)
from cm1core.cuda_terrain_open_boundary import terrain_static_shapes
from cm1core.gm1_idealized_residency import IdealizedSource
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]
TORNADO_TOML = ROOT / "examples/cartesian-stretched-tornado-grade-grid.toml"
UNIFORM_TOML = ROOT / "examples/cartesian-uniform-grid.toml"

# param.F:7819-7953 on a domain small enough to allocate on the host.  Two
# 20 m layers, four quadratic layers of nominal 50 m, four 80 m layers:
# 2*20 + 4*50 + 4*80 = 560 m, which is param.F:7867-7877's own consistency
# test.  dz stays a free metric reference (param.F:8170).
SMALL_STRETCH = {
    "nk": 10, "dz_m": 50.0, "ztop_m": 560.0,
    "str_bot_m": 40.0, "str_top_m": 240.0, "dz_bot_m": 20.0, "dz_top_m": 80.0,
}


def _small_levels() -> VerticalLevels:
    return build_vertical_levels(stretch_z=1, **SMALL_STRETCH)


# ---------------------------------------------------------------------------
# 1.  The configuration file
# ---------------------------------------------------------------------------

def test_tornado_grade_toml_is_admitted_and_carries_its_own_column() -> None:
    config = load_config(TORNADO_TOML)
    assert config.grid.stretch_z == 1
    runtime = build_runtime_config(config)
    grid = runtime.grid
    assert grid.stretch_z == 1
    assert grid.ztop_m == 15000.0
    levels = grid.levels
    zf = np.asarray(levels.zf, dtype=np.float32)
    spacings = [float(zf[k + 1] - zf[k]) for k in range(1, levels.nk + 1)]
    # param.F:7925-7927: nint(str_bot/dz_bot) = 10 layers of exactly dz_bot,
    # which is eleven faces at multiples of 20 m.
    assert spacings[:10] == [20.0] * 10
    assert float(levels.lowest_mass_level_m) == 10.0
    assert levels.levels_below_m(1000.0) == 22
    assert max(spacings) == pytest.approx(320.0, abs=1.0e-3)


def test_the_admitted_column_is_the_sealed_tornado_grade_child_word_for_word() -> None:
    """The TOML and ``stretch_z1_child_15km`` are the same eight profiles.

    The sealed grid was generated once and its per-profile SHA-256 words were
    recorded.  A configuration file that claims to be that grid has to produce
    those exact binary32 words, or it is a different column wearing the same
    name.
    """

    import hashlib

    from cm1core.stretch_z1_child_15km import (
        STRETCH_Z1_CHILD_15KM_PARAMETERS,
        STRETCH_Z1_CHILD_15KM_PROFILE_SHA256,
    )

    levels = build_runtime_config(load_config(TORNADO_TOML)).grid.levels
    sealed = STRETCH_Z1_CHILD_15KM_PARAMETERS
    assert (levels.nk, levels.dz_reference_m, levels.ztop_m) == (
        sealed.nk, sealed.dz, sealed.ztop,
    )
    assert (levels.str_bot_m, levels.str_top_m) == (sealed.str_bot, sealed.str_top)
    assert (levels.dz_bot_m, levels.dz_top_m) == (sealed.dz_bot, sealed.dz_top)
    for name, expected in STRETCH_Z1_CHILD_15KM_PROFILE_SHA256.items():
        words = np.ascontiguousarray(getattr(levels, name), dtype=np.float32)
        assert hashlib.sha256(words.astype("<f4", copy=False).tobytes()).hexdigest() == expected, name


def test_the_tornado_deck_generates_vorticity_at_the_layer_it_resolves() -> None:
    """A 20 m lowest layer and a free-slip bottom would not be a tornado deck.

    ``turb.F:4596-4605`` writes the surface stress at ``k=1`` and
    ``turbtend.F:1454`` feeds it to the lowest row of the implicit vertical
    momentum solve, so the drag acts across the lowest model layer.  The
    coordinate only matters because that layer is 20 m deep.
    """

    runtime = build_runtime_config(load_config(TORNADO_TOML))
    assert runtime.bbc == 3
    assert runtime.sfcmodel == 1
    assert runtime.cecd == 1
    assert runtime.cnstcd == 0.001
    assert runtime.isfcflx == 0
    zf = np.asarray(runtime.grid.levels.zf, dtype=np.float32)
    assert float(zf[2] - zf[1]) == 20.0


def test_uniform_default_is_untouched_by_the_stretched_lane() -> None:
    config = load_config(UNIFORM_TOML)
    assert config.grid.stretch_z == 0
    grid = build_runtime_config(config).grid
    # A stretch_z=0 geometry carries no coordinate object at all, so its
    # record, hash and every derived word are exactly what they were before
    # the vertical-coordinate keys existed.
    assert grid.vertical is None
    assert grid.stretch_z == 0
    assert "vertical" not in grid.record()
    assert grid.ztop_m == grid.nz * grid.dz_m
    assert grid.levels.stretch_z == 0
    assert grid.levels.ztop_m == grid.nz * grid.dz_m


def test_operator_family_is_selected_by_stretch_z_alone() -> None:
    uniform = cartesian_operator_family(0)
    stretched = cartesian_operator_family(1)
    assert uniform == CARTESIAN_OPERATOR_FAMILIES["uniform"]
    for branch in (1, 3, 4):
        assert cartesian_operator_family(branch) == stretched
    assert "scalar dz" in uniform["vertical_metric"]
    assert "rdz*mh(1,1,k)" in stretched["vertical_metric"]
    for phase in ("pre_acoustic", "acoustic", "post_acoustic", "scalar_moist"):
        assert uniform[phase] != stretched[phase]


# ---------------------------------------------------------------------------
# 2.  The generated coordinate is what every vertical difference divides by
# ---------------------------------------------------------------------------

def test_rdsf_is_the_layer_reciprocal_and_rdz_mh_is_its_uniform_spelling() -> None:
    """``adv.F:551`` vs ``adv.F:572`` on the same column.

    CM1's terrain branch multiplies by ``rdsf(k)`` and its Cartesian branch by
    ``rdz*mh(k)``.  ``init_terrain.F:297`` makes the first ``1/(sigmaf(k+1)
    -sigmaf(k))`` and ``param.F:8170`` makes the second ``(1/dz)*(dz/(zf(k+1)
    -zf(k)))``, so they are the same number and GM1's flat terrain-coordinate
    lane may use either.  A bare ``1/dz`` is neither.
    """

    config = load_config(TORNADO_TOML)
    levels = build_runtime_config(config).grid.levels
    rdsf = np.asarray(levels.rdsf, dtype=np.float32)
    mh = np.asarray(levels.mh, dtype=np.float32)
    dz = np.float32(levels.dz_reference_m)
    rdz = np.float32(np.float32(1.0) / dz)
    bare = float(rdz)
    for k in range(1, levels.nk + 1):
        cartesian = np.float32(rdz * mh[k])
        assert abs(float(rdsf[k]) - float(cartesian)) <= 2.0 * float(
            np.spacing(cartesian)
        ), k
    # The lowest layer is 20 m under a 100 m metric reference, so a kernel that
    # divided by dz alone would advect it five times too slowly.
    assert abs(float(rdsf[1]) / bare - 5.0) < 1.0e-5
    assert float(rdsf[levels.nk]) / bare == pytest.approx(100.0 / 320.0, rel=1e-5)


def test_allocation_plan_prices_the_stretched_metrics_it_will_upload() -> None:
    """``param.F:8170,8176`` and ``param.F:8606,8622`` stop being constants."""

    grid = CartesianGridGeometry(40, 40, SMALL_STRETCH["nk"], 100.0, 100.0,
                                 SMALL_STRETCH["dz_m"], _small_levels())
    uniform = CartesianGridGeometry(40, 40, SMALL_STRETCH["nk"], 100.0, 100.0,
                                    SMALL_STRETCH["dz_m"])
    stretched_plan = cartesian_dynamics_allocation_plan(
        RuntimeConfig(grid, memory_budget_bytes=4_000_000_000)
    )
    uniform_plan = cartesian_dynamics_allocation_plan(
        RuntimeConfig(uniform, memory_budget_bytes=4_000_000_000)
    )
    assert (
        stretched_plan["groups_payload_bytes"]["immutable_metrics"]
        > uniform_plan["groups_payload_bytes"]["immutable_metrics"]
    )


def test_both_branches_own_the_acoustic_column_report_at_one_word(monkeypatch) -> None:
    """The extent this lane allocates is the extent its image must require.

    ``sound.F:659-730``'s six ``tri_*`` arrays are a per-column report that no
    later phase reads -- ``post_acoustic`` releases every non-persistent padded
    carrier before the bridge runs -- and on the tornado-grade domain they are
    762 MB.  The lane owns one word each on *both* branches, which is exactly
    why each branch must compile the acoustic image that never stores into
    them: a stretched run that owned one word and compiled the storing image
    stops on ``tri_aa`` at its first acoustic stage, which is what the first
    real stretched run did.
    """

    for levels in (None, _small_levels()):
        dynamics = _build_dynamics(monkeypatch, levels)
        full = terrain_open_sound_shapes(
            dynamics.geometry.nx, dynamics.geometry.ny, dynamics.geometry.nz
        )
        for name in COLUMN_DIAGNOSTIC_NAMES:
            assert dynamics._padded_shapes[name] == (1,), name
            assert full[name] != (1,), name


# ---------------------------------------------------------------------------
# 3.  The levels reach the Rust initializer, or are refused by name
# ---------------------------------------------------------------------------

class _NoLevelsOwner:
    """An initializer build that predates ``param.F:7814-8139``."""

    def __init__(self, _library, _sounding, *, iinit, geometry):
        self.groups = {}

    def close(self):
        pass


class _LevelsOwner(_NoLevelsOwner):
    seen: object = None

    def __init__(self, _library, _sounding, *, iinit, geometry, levels="absent"):
        super().__init__(_library, _sounding, iinit=iinit, geometry=geometry)
        type(self).seen = levels


def test_a_build_that_cannot_take_levels_is_refused_by_what_it_would_do() -> None:
    grid = CartesianGridGeometry(9, 9, SMALL_STRETCH["nk"], 100.0, 100.0,
                                 SMALL_STRETCH["dz_m"], _small_levels())
    with pytest.raises(ValueError) as error:
        IdealizedSource("unused", b"metadata", grid, owner_factory=_NoLevelsOwner)
    message = str(error.value)
    assert "param.F:7814-8139" in message
    # The refusal names the concrete wrong result: 25 m instead of 10 m.
    assert "25.0 m instead of 10.0 m" in message


def test_a_uniform_run_never_passes_levels_at_all() -> None:
    grid = CartesianGridGeometry(9, 9, 10, 100.0, 100.0, 50.0)
    _LevelsOwner.seen = "unset"
    IdealizedSource("unused", b"metadata", grid, owner_factory=_LevelsOwner)
    assert _LevelsOwner.seen == "absent"
    # An old build still initializes the uniform column.
    IdealizedSource("unused", b"metadata", grid, owner_factory=_NoLevelsOwner)


def test_a_stretched_run_hands_rust_its_own_generated_levels() -> None:
    levels = _small_levels()
    grid = CartesianGridGeometry(9, 9, levels.nk, 100.0, 100.0,
                                 SMALL_STRETCH["dz_m"], levels)
    _LevelsOwner.seen = "unset"
    IdealizedSource("unused", b"metadata", grid, owner_factory=_LevelsOwner)
    assert _LevelsOwner.seen == levels
    assert _LevelsOwner.seen.sha256 == levels.sha256


# ---------------------------------------------------------------------------
# 4.  Every large-step phase, from the geometry to the bound launch
# ---------------------------------------------------------------------------

class _Stream:
    def __init__(self) -> None:
        self.depth = 0
        self.ptr = 1

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *_):
        self.depth -= 1

    def synchronize(self):
        pass


class _Event:
    def __init__(self, **_):
        pass

    def record(self, stream):
        self.stream = stream

    def synchronize(self):
        pass


class _HostStorage:
    """Array storage only.  No kernel and no scientific value is emulated."""

    float32 = np.float32
    float64 = np.float64
    uint32 = np.uint32
    cuda = SimpleNamespace(Device=lambda: SimpleNamespace(id=0), Event=_Event)
    copyto = staticmethod(np.copyto)
    broadcast_to = staticmethod(np.broadcast_to)

    def __init__(self, stream):
        self.stream = stream

    def empty(self, shape, *, dtype):
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape, *, dtype):
        return np.zeros(shape, dtype=dtype)

    def asarray(self, value, *, dtype, order):
        return np.array(value, dtype=dtype, order=order, copy=True)

    def get_default_memory_pool(self):
        return SimpleNamespace(free_all_blocks=lambda: None)


class _Owner:
    """Rust carrier metadata for one geometry, with a declared coordinate."""

    seen_levels: object = None

    def __init__(self, _library, _sounding, *, iinit, geometry, levels="absent"):
        type(self).seen_levels = levels
        nz, ny, nx = geometry.shape_zyx
        g = CartesianGridGeometry(nx, ny, nz, *geometry.spacing_m)
        mass = g.padded_shape("mass", halo_xy=3, halo_z=1)
        shapes = terrain_open_sound_shapes(nx, ny, nz)
        terrain = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
        for name in ("pi0", "th0", "qv0", "rho0", "prs0", "rr0", "tha", "th3d", "rf0"):
            terrain[name] = np.zeros(mass, dtype=np.float32)
        # d0f0429: the prognostic upload reads the INITIALIZED prs/rho/rr/rf
        # (initial_prognostic_requests), not only the base state; rf keeps the
        # Rust 0:nk+1 extent the source layout rule expects.
        for name in ("prs", "rho", "rr"):
            terrain[name] = np.zeros(mass, dtype=np.float32)
        terrain["rf"] = np.zeros((nz + 2, ny + 6, nx + 6), dtype=np.float32)
        static = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in terrain_static_shapes(ni=nx, nj=ny, nk=nz).items()
        }
        # A stretched column's mh/rmh/mf/rmf/c1/c2 vary with k, so give them
        # distinct words: a constant carrier would be shared by the metric map
        # and would not exercise the per-array upload the stretched operators
        # actually receive.
        for name in ("mh", "rmh", "mf", "rmf", "c1", "c2"):
            value = static[name]
            for k in range(value.shape[0]):
                value[k] = np.float32(1.0 + 0.25 * k)
        self.groups = {
            "terrain_open": terrain,
            "moist": {name: np.zeros((6, *mass), dtype=np.float32) for name in ("qa", "q3d")},
            "terrain_static": static,
            "base_wind": {
                "u0": np.zeros(g.u_shape, dtype=np.float32),
                "v0": np.zeros(g.v_shape, dtype=np.float32),
            },
        }

    def close(self):
        pass


def _noop(*_args, **_kwargs):
    return None


def _build_dynamics(monkeypatch, levels, **declared) -> CartesianDynamics:
    grid = CartesianGridGeometry(9, 9, SMALL_STRETCH["nk"], 100.0, 100.0,
                                 SMALL_STRETCH["dz_m"], levels)
    config = RuntimeConfig(grid, dt_s=0.5, acoustic_substeps=6,
                           memory_budget_bytes=2_000_000_000, **declared)
    stream = _Stream()
    storage = _HostStorage(stream)
    def _stub_operators(self):
        for name in (
            "open_compiled", "terrain_compiled", "lateral_compiled",
            "momentum_advection_compiled", "psolver_compiled", "moist_compiled",
            "stretched_moist_compiled", "scalar_advection_compiled",
            "scalar_compiled", "boundary_compiled", "forecast_compiled",
            "child_core_compiled", "apmasscon_compiled", "startup_compiled",
        ):
            setattr(self, name, SimpleNamespace(kernels=(_noop, _noop, _noop), target_arch="0"))
        # momentum_rk publishes exactly two kernels; pre_acoustic unpacks them.
        self.momentum_compiled = SimpleNamespace(kernels=(_noop, _noop), target_arch="0")
        # c350390 binds the lateral family inside _compile_operators; the
        # stub binds the sealed all-open default, which allocates nothing.
        from cm1core.cuda_boundary_families import DEFAULT_LATERAL_BOUNDARY_SELECTION

        self.lateral_selection = DEFAULT_LATERAL_BOUNDARY_SELECTION
        self.family_compiled = SimpleNamespace(kernels=(), target_arch="0")
        self.lateral_workspace = {}
        # e4024cc keeps the profile-bound kernels in their own table and the
        # scalar stage's pack/legacy pair beside scalar_compiled.
        stub = SimpleNamespace(kernels=(_noop, _noop, _noop), target_arch="0")
        self.scalar_legacy_compiled = stub
        self.scalar_pack_compiled = stub
        self.stretched_compiled = (
            {name: stub for name in ("momentum", "psolver", "moist", "advection", "pdef")}
            if self.geometry.stretch_z != 0 else {}
        )

    monkeypatch.setattr(CartesianDynamics, "_compile_operators", _stub_operators)
    monkeypatch.setattr(CartesianDynamics, "_initial_dry_mass", lambda self: 1.0)
    monkeypatch.setattr(
        CartesianDynamics, "_publish_startup_thermodynamics",
        lambda self: {"schema": "stub", "performed": False},
    )
    source = IdealizedSource("unused", b"metadata", grid, owner_factory=_Owner)
    return CartesianDynamics(source, config, cupy=storage, stream=stream)


def _record(calls, name):
    def capture(*args, **kwargs):
        # The acoustic workspace is a live dict the phase clears after the
        # launch, so its keys are snapshotted here rather than read later.
        snapshot = {
            key: sorted(value) if type(value) is dict else value
            for key, value in kwargs.items()
        }
        calls.append((name, args, snapshot))
        return SimpleNamespace(
            receipt={"schema": name}, ready_event=_Event(), divx=None,
        )
    return capture


def test_every_large_step_phase_reaches_the_profile_bound_family(monkeypatch) -> None:
    """The four operators the retired refusal named, driven end to end.

    Each phase is called on a real ``CartesianDynamics`` built from a real
    ``stretch_z=1`` geometry.  The uniform entry points are replaced with
    functions that fail the test if they are reached at all, and the stretched
    ones record their keyword arguments, so this proves both which family runs
    and that no scalar ``dz`` is handed to it.
    """

    from cm1core import (
        cuda_cartesian_dynamics as dynamics_module,
        cuda_gm1_child_psolver3_open as psolver_module,
        cuda_gm1_child_stretched_moist as moist_module,
        cuda_gm1_child_stretched_momentum as momentum_module,
        cuda_moist_stage_bridge as bridge_module,
        cuda_momentum_advection5 as legacy_momentum,
        cuda_psolver3_terrain_open as legacy_psolver,
        cuda_scalar_moist_rk as scalar_module,
    )

    levels = _small_levels()
    state = _build_dynamics(monkeypatch, levels)
    assert _Owner.seen_levels == levels
    # e4024cc: the profile-bound family accepts only a published
    # Gm1ChildStretchedMetrics binding, not an arbitrary namespace.
    from cm1core.cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics

    # The Cartesian lane is terrain_flag=.false.: its binding carries the
    # column's own mh/mf, which is what selects CM1's flat statements.
    vertical = Gm1ChildStretchedMetrics(
        sigma=None, sigmaf=None, rds=None, rdsf=None,
        ztop_f32=560.0, dz_reference_f32=50.0,
        target_arch="0", identity={"schema": "test"},
        mh=object(), mf=object(),
    )
    assert vertical.flat_column
    state.vertical = vertical

    calls: list = []
    for module, name in (
        (legacy_momentum, "launch_cuda_terrain_open_momentum_advection5"),
        (legacy_psolver, "launch_cuda_psolver3_terrain_open_stage"),
        (bridge_module, "launch_cuda_post_acoustic_flux_bridge"),
    ):
        monkeypatch.setattr(
            module, name,
            lambda *a, _n=name, **k: pytest.fail(f"uniform {_n} reached on a stretched grid"),
        )
    monkeypatch.setattr(
        momentum_module, "launch_cuda_gm1_child_stretched_momentum5",
        _record(calls, "momentum"),
    )
    monkeypatch.setattr(
        psolver_module, "launch_cuda_gm1_child_psolver3_open_stage",
        _record(calls, "acoustic"),
    )
    monkeypatch.setattr(
        moist_module, "launch_cuda_gm1_child_stretched_post_acoustic",
        _record(calls, "post_acoustic"),
    )
    # solve2.F:897-906 only: the uten/vten buoyancy term of :971-1008 is a
    # terrain_flag statement, so the stretched column never reaches the
    # uniform (terrain-form) buoyancy kernel.
    monkeypatch.setattr(
        moist_module, "launch_cuda_gm1_flat_stretched_buoyancy",
        _record(calls, "buoyancy"),
    )
    monkeypatch.setattr(
        bridge_module, "launch_cuda_moist_buoyancy",
        lambda *a, **k: pytest.fail("terrain-form buoyancy reached on a flat stretched grid"),
    )
    # e4024cc routes the stretched scalar stage through the child step
    # composer's profile-bound launcher; the uniform launcher must not run.
    from cm1core import cuda_gm1_child_step_composer as composer_module

    monkeypatch.setattr(
        scalar_module, "launch_cuda_scalar_moist_rk_stage",
        lambda *a, **k: pytest.fail("uniform scalar stage reached on a stretched grid"),
    )
    monkeypatch.setattr(
        composer_module, "_launch_cuda_gm1_child_stretched_scalar_stage",
        _record(calls, "scalar"),
    )
    for name in ("launch_cuda_eqtset2_stage_terms", "begin_cuda_moist_stage"):
        monkeypatch.setattr(bridge_module, name, _noop)
    monkeypatch.setattr(dynamics_module, "pack_cartesian_held_momentum", _noop)
    # solve2.F:484-487 (bcwsfc, bc2d(w3d)) are terrain_flag statements: a
    # flat stretched column must not reach the terrain boundary kernels.
    state.terrain_compiled = SimpleNamespace(
        kernels=tuple(
            (lambda *a, _i=i, **k: pytest.fail(f"terrain boundary kernel {_i} on a flat column"))
            for i in range(3)
        ),
        target_arch="0",
    )

    state._expected_phase = ("pre", 1)
    state.pre_acoustic(1)
    state.acoustic(1)
    state.post_acoustic(1)
    state.scalar_moist(1)

    assert [name for name, _, _ in calls] == [
        "momentum", "buoyancy", "acoustic", "post_acoustic", "scalar",
    ]
    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    # solve2.F:575's bcw(rrw,0) is a terrain_flag statement.
    assert by_name["momentum"][1]["rrw_boundary_hook"] is None
    assert by_name["buoyancy"][1]["compiled"] is state.stretched_compiled["moist"]
    for phase in ("momentum", "acoustic", "post_acoustic", "scalar"):
        args, kwargs = by_name[phase]
        assert "dz" not in kwargs, phase
        assert any(value is vertical for value in (*args, *kwargs.values())), phase
    # The stretched family averages base density across the open lateral
    # faces, so the padded carrier has to have reached it.
    assert "rho0_padded" in by_name["momentum"][0][1]
    assert "rho0_padded" in by_name["acoustic"][0][1]
    assert by_name["acoustic"][1]["selection"].stretch_z == 1
    # sound.F:81-82,756,782 read param3's alph and kdiv on every coordinate;
    # the stretched stage is handed the document's words, as the uniform is.
    assert by_name["acoustic"][1]["alph"] == state.config.alph
    assert by_name["acoustic"][1]["kdiv"] == state.config.kdiv
    # sound.F:274-302 and param.F:9182-9219 read the lateral family; the
    # stage is handed the bound fills' family, all four open on this grid.
    assert tuple(by_name["acoustic"][1]["lateral_sides"]) == (2, 2, 2, 2)
    # nothing was allocated for the uniform acoustic radiation-packing scratch
    assert state._acoustic_workspace == {}
    # A second stage enters the nrk>=2 branch that the terrain column uses
    # for bcwsfc; the guard above fails the test if it is reached.
    state._expected_phase = ("pre", 2)
    state.pre_acoustic(2)
    assert [name for name, _, _ in calls][-2:] == ["momentum", "buoyancy"]


def test_a_declared_alph_and_kdiv_reach_the_stretched_acoustic_stage(monkeypatch) -> None:
    """The stretched stage used to bind CM1_ALPH_F32/CM1_KDIV_F32 whatever was declared.

    config.py admits any alph and kdiv in [0, 1] on every grid, and CM1's
    sound.F has one code path for both coordinates, so a stretch_z=1 run that
    declared kdiv=0.35 integrated 0.10 while its receipt recorded 0.35.
    """

    from cm1core import (
        cuda_cartesian_dynamics as dynamics_module,
        cuda_gm1_child_psolver3_open as psolver_module,
    )
    from cm1core.cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics

    state = _build_dynamics(monkeypatch, _small_levels(), alph=0.55, kdiv=0.35)
    state.vertical = Gm1ChildStretchedMetrics(
        sigma=None, sigmaf=None, rds=None, rdsf=None,
        ztop_f32=560.0, dz_reference_f32=50.0,
        target_arch="0", identity={"schema": "test"},
    )
    calls: list = []
    monkeypatch.setattr(
        psolver_module, "launch_cuda_gm1_child_psolver3_open_stage",
        _record(calls, "acoustic"),
    )
    monkeypatch.setattr(dynamics_module, "pack_cartesian_held_momentum", _noop)
    state._expected_phase = ("acoustic", 2)
    state.acoustic(2)
    (name, _, kwargs), = calls
    assert (kwargs["alph"], kwargs["kdiv"]) == (0.55, 0.35)


def test_the_same_phases_on_a_uniform_grid_reach_the_uniform_family(monkeypatch) -> None:
    """``stretch_z=0`` must keep selecting exactly what it selected before."""

    from cm1core import (
        cuda_cartesian_dynamics as dynamics_module,
        cuda_gm1_child_psolver3_open as psolver_module,
        cuda_gm1_child_stretched_moist as moist_module,
        cuda_gm1_child_stretched_momentum as momentum_module,
        cuda_moist_stage_bridge as bridge_module,
        cuda_momentum_advection5 as legacy_momentum,
        cuda_psolver3_terrain_open as legacy_psolver,
        cuda_scalar_moist_rk as scalar_module,
    )

    state = _build_dynamics(monkeypatch, None)
    assert state.geometry.stretch_z == 0
    assert state.stretched_metrics is None
    assert _Owner.seen_levels == "absent"

    calls: list = []
    for module, name in (
        (momentum_module, "launch_cuda_gm1_child_stretched_momentum5"),
        (psolver_module, "launch_cuda_gm1_child_psolver3_open_stage"),
        (moist_module, "launch_cuda_gm1_child_stretched_post_acoustic"),
    ):
        monkeypatch.setattr(
            module, name,
            lambda *a, _n=name, **k: pytest.fail(f"stretched {_n} reached on a uniform grid"),
        )
    monkeypatch.setattr(
        legacy_momentum, "launch_cuda_terrain_open_momentum_advection5",
        _record(calls, "momentum"),
    )
    monkeypatch.setattr(
        legacy_psolver, "launch_cuda_psolver3_terrain_open_stage",
        _record(calls, "acoustic"),
    )
    monkeypatch.setattr(
        bridge_module, "launch_cuda_post_acoustic_flux_bridge",
        _record(calls, "post_acoustic"),
    )
    monkeypatch.setattr(
        scalar_module, "launch_cuda_scalar_moist_rk_stage", _record(calls, "scalar"),
    )
    for name in ("launch_cuda_eqtset2_stage_terms", "launch_cuda_moist_buoyancy",
                 "begin_cuda_moist_stage"):
        monkeypatch.setattr(bridge_module, name, _noop)
    monkeypatch.setattr(dynamics_module, "pack_cartesian_held_momentum", _noop)

    state._expected_phase = ("pre", 1)
    state.pre_acoustic(1)
    state.acoustic(1)
    state.post_acoustic(1)
    state.scalar_moist(1)

    assert [name for name, _, _ in calls] == [
        "momentum", "acoustic", "post_acoustic", "scalar",
    ]
    # Lock the whole uniform call, not just the absence of a coordinate: an
    # argument that appeared or vanished here would change what stretch_z=0
    # integrates, which is the one thing this work is not allowed to do.
    expected = {
        "momentum": {
            "ni", "nj", "nk", "nrk", "dx", "dy", "dz", "divx", "stream",
            "compiled", "rrw_boundary_hook",
            # eeea32b: CM1's solve2.F icor=1 seam; None on this icor=0 grid,
            # asserted below, so the uniform integration is unchanged.
            "post_divergence_hook",
            # adv.F advu i1/i2, advv j1/j2 and doitw..doitn: the bound family,
            # all four open on this grid (asserted below).
            "lateral_sides",
        },
        "acoustic": {
            "ni", "nj", "nk", "nrk", "dt", "nsound", "dx", "dy", "dz",
            "ablations", "stream", "compiled", "terrain_compiled",
            "open_compiled", "lateral_compiled", "workspace",
            # c350390 (lateral family step), eeea32b (CM1 alph/kdiv words) and
            # d951954 (live nest substep hook); each is pinned to the
            # single-domain all-open default below.
            "lateral", "alph", "kdiv", "nest_substep",
        },
        "post_acoustic": {"dx", "dy", "dz", "stream"},
        "scalar": {
            "nrk", "dt", "dx", "dy", "dz", "stream", "compiled",
            "advection_compiled", "open_compiled",
        },
    }
    for name, _, kwargs in calls:
        assert set(kwargs) == expected[name], name
        assert kwargs["dz"] == state.geometry.dz_m, name
        assert "vertical" not in kwargs, name
    assert calls[0][2]["post_divergence_hook"] is None
    assert tuple(calls[0][2]["lateral_sides"]) == (2, 2, 2, 2)
    assert calls[1][2]["lateral"] is None
    assert calls[1][2]["nest_substep"] is None
    assert calls[1][2]["alph"] == state.config.alph
    assert calls[1][2]["kdiv"] == state.config.kdiv
    # The uniform acoustic stage still owns registry-priced radiation packing.
    assert set(calls[1][2]["workspace"]) == set(
        terrain_open_acoustic_workspace_shapes(
            state.geometry.nx, state.geometry.ny, state.geometry.nz
        )
    )


def test_the_stage_refuses_both_columns_and_neither(monkeypatch) -> None:
    """``dz`` on a stretched column is a metric reference, not a depth."""

    from cm1core.cuda_scalar_moist_rk import launch_cuda_scalar_moist_rk_stage

    for kwargs in ({}, {"dz": 100.0, "vertical": object()}):
        with pytest.raises(ValueError, match="exactly one vertical coordinate"):
            launch_cuda_scalar_moist_rk_stage(
                SimpleNamespace(next_nrk=1, completed=False), {}, {}, None, None,
                nrk=1, dt=0.5, dx=50.0, dy=50.0, **kwargs,
            )
