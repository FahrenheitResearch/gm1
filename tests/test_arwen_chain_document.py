"""The [chain] table, the door's host binding and the chain driver's clock.

Host only.  The driver test runs :class:`ChainDriver` against a NumPy stand-in
for CuPy with the live nest's kernels stubbed out: what it checks is the
chain's own logic -- which interval's tables are loaded before which step,
that ``dtbc`` restarts at every boundary time and advances by ``dt`` before
every step, and that the initial state lands in both time levels of every
carrier as GM1 perturbations.  The kernels themselves are the live nest's,
raw-word checked in ``tests/test_nest_live_operators.py``.
"""

from __future__ import annotations

import contextlib
import tomllib
import types

import pytest

np = pytest.importorskip("numpy")

from cm1core import arwen_chain as chain  # noqa: E402
from cm1core.config import parse_mapping  # noqa: E402
from cm1core.errors import CM1Refusal  # noqa: E402

import test_arwen_chain as synth  # noqa: E402

DOCUMENT = """
[run]
lane = "cartesian"
duration_s = 600.0
dt_s = 5.0
history_interval_s = 300.0
restart_interval_s = 600.0

[initialization]
isnd = 7
iwnd = 0

[grid]
nx = 15
ny = 15
nz = 20
dx_m = 1000.0
dy_m = 1000.0
dz_m = 400.0
terrain = false
stretch_x = false
stretch_y = false
stretch_z = false

[dynamics]
psolver = 3
adaptive_timestep = false
scalar_horizontal_order = 5
scalar_vertical_order = 5
velocity_horizontal_order = 5
velocity_vertical_order = 5
scalar_weno = 2
velocity_weno = 0
weno_order = 5
dry_mass_correction = true
coriolis = 1
beta_plane = 0
fcor_s_inv = {fcor!r}
alph = 0.60
kdiv = 0.10

[boundaries]
west = 2
east = 2
south = 2
north = 2
bottom = 1
top = 1
radiative_order = 4

[closures]
sgs_model = 1
vertical_implicit_diffusion = true
upper_damping = 1
upper_damping_depth_m = 2000.0
upper_damping_rate_s_inv = 0.0033333333333
moving_frame = false
u_move_m_s = 0.0
v_move_m_s = 0.0

[runtime]
acoustic_substeps = 4
tile_nx = 64
tile_ny = 64
warmup_steps = 0
memory_budget_mib = 8192
capacity_profile = "default"

[physics]
component = "wsm6-mp6"
equation_set = 2
"""

CHAIN = """
[chain]
history_directory = "{history}"
domain_id = 2
start_time = "{start}"
end_time = "{end}"
parent_i0 = 5
parent_j0 = 5
refinement_ratio = 3
spec_zone = 1
relax_zone = 4
terrain_relief_limit_m = 50.0
map_factor_departure_limit = 0.01
prepared_directory = "{prepared}"
"""


def document(tmp_path, *, chain_table=True, replace=(), **chain_overrides):
    text = DOCUMENT.format(fcor=synth.FCOR)
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    raw = tomllib.loads(text)
    if chain_table:
        table = tomllib.loads(CHAIN.format(
            history=tmp_path, start=synth.stamp(0), end=synth.stamp(600),
            prepared=tmp_path / "prepared",
        ))["chain"]
        table.update(chain_overrides)
        raw["chain"] = table
    return raw


def refused(code):
    return pytest.raises(CM1Refusal, match=code)


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------
def test_a_document_without_chain_projects_no_chain_key(tmp_path):
    config = parse_mapping(document(tmp_path, chain_table=False))
    assert config.chain is None
    assert "chain" not in config.canonical_dict()


def test_the_chain_table_parses_and_is_projected(tmp_path):
    config = parse_mapping(document(tmp_path))
    assert config.chain.refinement_ratio == 3
    assert config.canonical_dict()["chain"]["start_time"] == synth.stamp(0)
    assert config.sha256 != parse_mapping(document(tmp_path, chain_table=False)).sha256


def test_the_chain_table_is_declared_whole(tmp_path):
    raw = document(tmp_path)
    del raw["chain"]["terrain_relief_limit_m"]
    with pytest.raises(CM1Refusal):
        parse_mapping(raw)
    raw = document(tmp_path)
    raw["chain"]["spec_exp"] = 0.3
    with refused("CM1-CFG-UNKNOWN"):
        parse_mapping(raw)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"refinement_ratio": 4}, "CM1-CFG-CHAIN-RATIO"),
        ({"relax_zone": 1}, "CM1-CFG-CHAIN-ZONES"),
        ({"start_time": "2026-05-01 18:00"}, "CM1-CFG-CHAIN-TIME"),
        ({"end_time": synth.stamp(-300)}, "CM1-CFG-CHAIN-TIME"),
        ({"end_time": synth.stamp(300)}, "CM1-CFG-CHAIN-DURATION"),
    ],
)
def test_table_refusals(tmp_path, overrides, code):
    with refused(code):
        parse_mapping(document(tmp_path, **overrides))


@pytest.mark.parametrize(
    ("replace", "code"),
    [
        ((("nx = 15", "nx = 16"),), "CM1-CFG-CHAIN-GRID"),
        ((("west = 2", "west = 1"),), "CM1-CFG-CHAIN-LATERAL"),
        ((("moving_frame = false\nu_move_m_s = 0.0", "moving_frame = true\nu_move_m_s = 10.0"),), "CM1-CFG-CHAIN-MOVING-FRAME"),
        ((("isnd = 7\niwnd = 0", "isnd = 5\niwnd = 2"),), "CM1-CFG-CHAIN-INITIALIZATION"),
    ],
)
def test_document_refusals(tmp_path, replace, code):
    with refused(code):
        parse_mapping(document(tmp_path, replace=replace))


# ---------------------------------------------------------------------------
# The door's host binding.
# ---------------------------------------------------------------------------
def _prepare(tmp_path, config):
    frames = [synth.make_frame(tmp_path / "history", t) for t in (0, 300, 600)]
    return chain.prepare_chain(frames, chain.chain_request_from_config(config), config.chain.prepared_directory)


def test_the_door_binds_a_prepared_chain_with_its_own_sounding(tmp_path):
    config = parse_mapping(document(tmp_path))
    _prepare(tmp_path, config)
    sounding = (tmp_path / "prepared" / "input_sounding").read_bytes()
    binding = chain.require_chain(config, sounding_bytes=sounding)
    assert binding.prepared.steps_per_interval == 60 and binding.relax_zone == 4
    with refused("GM1-CHAIN-SOUNDING"):
        chain.require_chain(config, sounding_bytes=sounding + b"1 300 0 0 0\n")


def test_the_door_refuses_a_chain_prepared_for_another_clock(tmp_path):
    config = parse_mapping(document(tmp_path))
    _prepare(tmp_path, config)
    other = parse_mapping(document(tmp_path, replace=(("dt_s = 5.0", "dt_s = 2.5"),)))
    sounding = (tmp_path / "prepared" / "input_sounding").read_bytes()
    with refused("GM1-CHAIN-PREPARED"):
        chain.require_chain(other, sounding_bytes=sounding)


def test_the_door_refuses_restart_and_iconly(tmp_path):
    config = parse_mapping(document(tmp_path))
    with refused("GM1-CHAIN-RESTART"):
        chain.require_chain_execution(config, restart_dir=tmp_path / "r", initial_condition_only=False)
    with refused("GM1-CHAIN-INITIAL-CONDITION"):
        chain.require_chain_execution(config, restart_dir=None, initial_condition_only=True)


# ---------------------------------------------------------------------------
# The driver's clock and table loading, on a NumPy stand-in.
# ---------------------------------------------------------------------------
def _fake_cupy():
    fake = types.SimpleNamespace(
        float32=np.float32, asarray=np.asarray, zeros=np.zeros, copyto=np.copyto,
        asnumpy=np.asarray, ascontiguousarray=np.ascontiguousarray, maximum=np.maximum,
        arange=np.arange, clip=np.clip, where=np.where,
    )
    return fake


def host_spec_apply(cupy, module, *, field, origin, lattice, tend, side, spec_zone, width, dt,
                    snap=None, base=None, value=None, nonnegative=False, stream=None):
    """``launch_spec_apply`` on the host: the kernel's CPU reference on the owner's lattice view.

    The kernel text is held to this reference word for word by tests/test_nest_live_operators.py.
    """

    from cm1core.nest_live_kernels import reference_spec_apply

    k0, j0, i0 = origin
    sl = (slice(k0, k0 + lattice[0]), slice(j0, j0 + lattice[1]), slice(i0, i0 + lattice[2]))
    view = field[sl]
    view[...] = reference_spec_apply(
        view, lattice=lattice, tend=tend, side=side, spec_zone=spec_zone, width=width, dt=dt,
        snap=snap, base=None if base is None else base[sl], value=value, nonnegative=nonnegative,
    )


class _Stream(contextlib.AbstractContextManager):
    def __exit__(self, *exc):
        return False

    def synchronize(self):
        return None


def _fake_runtime(config, prepared_initial_pi):
    from cm1core.moisture_species import moisture_species_profile

    nz, ny, nx = config.grid.nz, config.grid.ny, config.grid.nx
    numq = moisture_species_profile(config.physics.component).numq
    th0 = np.broadcast_to(np.float32(300.0), (nz, ny, nx)).copy()
    pi0 = np.broadcast_to(prepared_initial_pi[:, :1, :1], (nz, ny, nx)).astype(np.float32).copy()
    dyn = types.SimpleNamespace(
        config=types.SimpleNamespace(dt_s=config.run.dt_s),
        numq=numq,
        geometry=types.SimpleNamespace(nz=nz, ny=ny, nx=nx),
        arrays={name: np.zeros((nz + 2, ny + 7, nx + 7), np.float32)
                for name in ("u3d", "ua", "v3d", "va", "w3d", "wa", "rru", "rrv")}
        | {name: np.zeros((nz + 2, ny + 2, nx + 2), np.float32) for name in ("pp3d", "ppi")},
        fields={name: np.zeros((nz, ny, nx), np.float32) for name in ("pp3d", "ppi", "th3d", "tha")}
        | {"q3d": np.zeros((numq, nz, ny, nx), np.float32), "qa": np.zeros((numq, nz, ny, nx), np.float32),
           "th0": th0, "pi0": pi0},
        nest_forcing=None,
    )
    runtime = types.SimpleNamespace(
        dynamics=dyn,
        config=types.SimpleNamespace(microphysics=config.physics.component, dt_s=config.run.dt_s),
        clock=types.SimpleNamespace(completed_steps=0),
        finish=types.SimpleNamespace(_initial_mass=lambda: 1.0),
        closure=types.SimpleNamespace(interpolated_tke_start=False),
    )
    return runtime


def test_the_driver_loads_wrf_tables_at_boundary_times_and_advances_dtbc(tmp_path, monkeypatch):
    from cm1core import arwen_chain_runtime as rt
    from cm1core import nest_live

    monkeypatch.setattr(nest_live, "compile_nest_live_module", lambda cupy: None)
    monkeypatch.setattr(nest_live, "compile_nest_relaxation_module", lambda cupy: None)
    monkeypatch.setattr(rt, "_refill_halos_and_thermodynamics", lambda d: None)
    config = parse_mapping(document(tmp_path))
    fields = {"u": lambda x, y, z, t: 6.0 + 1e-4 * x + 0.002 * t + 0.0 * z}
    frames = [synth.make_frame(tmp_path / "history", t, fields=fields) for t in (0, 300, 600)]
    chain.prepare_chain(frames, chain.chain_request_from_config(config), config.chain.prepared_directory)
    prepared = chain.PreparedChain.open(config.chain.prepared_directory)
    runtime = _fake_runtime(config, prepared.initial("pi"))
    driver = rt.ChainDriver(runtime=runtime, prepared=prepared, spec_zone=1, relax_zone=4,
                            cupy=_fake_cupy(), stream=_Stream())
    assert runtime.dynamics.nest_forcing is driver.forcing
    with pytest.raises(RuntimeError, match="before its initial state"):
        driver.before_step()
    record = driver.initialize()
    # An interpolated start: CM1 diagnoses its first TKE with turbsmag
    # (src/init3d.F:2688-2760), which a fresh iinit start does not.
    assert runtime.closure.interpolated_tke_start is True
    d = runtime.dynamics
    u0 = prepared.initial("u")
    assert np.array_equal(d.arrays["u3d"][1:1 + 20, 3:3 + 15, 3:3 + 16], u0)
    assert np.array_equal(d.arrays["ua"][1:1 + 20, 3:3 + 15, 3:3 + 16], u0)
    theta = prepared.initial("theta")
    assert np.array_equal(d.fields["th3d"], chain.perturbation(theta, np.full(20, 300.0, np.float32)))
    assert np.array_equal(d.arrays["ppi"][1:21, 1:16, 1:16], d.fields["ppi"])
    assert record["tke"].startswith("GM1 initializer")

    dt = config.run.dt_s
    for step in range(0, 121):
        runtime.clock.completed_steps = step
        if step == 120:
            with refused("GM1-CHAIN-DURATION"):
                driver.before_step()
            break
        driver.before_step()
        # dtbc restarts at each boundary time and counts this step's end.
        assert driver.forcing.dtbc == pytest.approx(dt * (step % 60 + 1))
        if step in (0, 60):
            value, tend = driver.forcing.tables["u"]["west"]
            start = prepared.strip(step // 60, "u", "west")
            end = prepared.strip(step // 60 + 1, "u", "west")
            want_value, want_tend = chain.interval_tables(start, end, 300.0)
            assert np.array_equal(value, want_value) and np.array_equal(tend, want_tend)
            assert np.max(np.abs(tend - np.float32(0.002))) < 1e-5
    assert driver.forcing.loaded_intervals == [0, 1]
    assert driver.record()["intervals_loaded"] == [0, 1]


def test_the_specified_rows_end_every_step_at_wrfs_target(tmp_path, monkeypatch):
    """WRF spec_bdy_final (verifier): after microphysics the scalar carriers' specified rows are reset.

    On the RTX 3080 the chain's outer row had drifted from the driving run by up to 6.4 K in theta and
    14 g/kg in rain water after 30 min, because microphysics acted on those rows and nothing restored them.
    """

    from cm1core import arwen_chain_runtime as rt
    from cm1core import nest_live

    monkeypatch.setattr(nest_live, "compile_nest_live_module", lambda cupy: None)
    monkeypatch.setattr(nest_live, "compile_nest_relaxation_module", lambda cupy: None)
    monkeypatch.setattr(rt, "_refill_halos_and_thermodynamics", lambda d: None)
    monkeypatch.setattr(nest_live, "launch_spec_apply", host_spec_apply)
    config = parse_mapping(document(tmp_path))
    fields = {"theta": lambda x, y, z, t: 300.0 + 1e-4 * x + 0.004 * z + 0.001 * t,
              "u": lambda x, y, z, t: 6.0 + 1e-4 * x + 0.002 * t + 0.0 * z}
    frames = [synth.make_frame(tmp_path / "history", t, fields=fields) for t in (0, 300, 600)]
    chain.prepare_chain(frames, chain.chain_request_from_config(config), config.chain.prepared_directory)
    prepared = chain.PreparedChain.open(config.chain.prepared_directory)
    runtime = _fake_runtime(config, prepared.initial("pi"))
    driver = rt.ChainDriver(runtime=runtime, prepared=prepared, spec_zone=1, relax_zone=4,
                            cupy=_fake_cupy(), stream=_Stream())
    driver.initialize()
    for step in range(7):
        runtime.clock.completed_steps = step
        driver.before_step()
    d = runtime.dynamics
    rng = np.random.default_rng(3)
    for name in ("th3d", "pp3d"):
        d.fields[name][...] = rng.normal(size=d.fields[name].shape).astype(np.float32)
    d.fields["q3d"][...] = rng.random(d.fields["q3d"].shape).astype(np.float32)
    for name in ("u3d", "v3d", "w3d"):
        d.arrays[name][...] = rng.normal(size=d.arrays[name].shape).astype(np.float32)
    before = {name: d.fields[name].copy() for name in ("th3d", "pp3d", "q3d")}
    w_before = d.arrays["w3d"].copy()
    driver.forcing.finalize_specified(d)
    dtbc = np.float32(driver.forcing.dtbc)
    nz, ny, nx = d.fields["th3d"].shape
    for carrier, field in (("th", d.fields["th3d"]), ("pp", d.fields["pp3d"]), ("q0", d.fields["q3d"][0])):
        tables = driver.forcing.tables[carrier]
        want = lambda side: tables[side][0] + dtbc * tables[side][1]  # noqa: E731
        assert np.array_equal(field[:, 0, :], want("south")[:, 0, :])          # the y sides own the corners
        assert np.array_equal(field[:, ny - 1, :], want("north")[:, 0, :])
        assert np.array_equal(field[:, 1:ny - 1, 0], want("west")[:, 1:ny - 1, 0])
        assert np.array_equal(field[:, 1:ny - 1, nx - 1], want("east")[:, 1:ny - 1, 0])
    # Nothing inside the specified rows is touched.
    assert np.array_equal(d.fields["th3d"][:, 1:-1, 1:-1], before["th3d"][:, 1:-1, 1:-1])
    assert np.array_equal(d.fields["q3d"][:, :, 1:-1, 1:-1], before["q3d"][:, :, 1:-1, 1:-1])
    # theta's target is the driving state as a perturbation from th0, 7 steps into the interval.
    value, tend = driver.forcing.tables["th"]["south"]
    assert driver.forcing.dtbc == pytest.approx(7 * float(config.run.dt_s))
    assert np.max(np.abs(tend - np.float32(0.001))) < 1e-5 and np.array_equal(d.fields["th3d"][:, 0, :], (value + dtbc * tend)[:, 0, :])
    # WRF's specified lane resets u and v too (solve_em.F:4540-4565), and w only when nested (:4566).
    u = d.arrays["u3d"][1:21, 3:3 + ny, 3:3 + nx + 1]
    value, tend = driver.forcing.tables["u"]["west"]
    assert np.array_equal(u[:, 1:ny - 1, 0], (value + dtbc * tend)[:, 1:ny - 1, 0])
    value, tend = driver.forcing.tables["u"]["south"]
    assert np.array_equal(u[:, 0, :], (value + dtbc * tend)[:, 0, :])
    assert np.array_equal(d.arrays["w3d"], w_before)
    assert driver.forcing.final_specifications == 1
    record = driver.record()["lateral_forcing"]["spec_bdy_final"]
    assert record["steps"] == 1 and record["lane"] == "specified"
    assert record["carriers"][:4] == ["u", "v", "pp", "th"] and "w" not in record["carriers"]


# ---------------------------------------------------------------------------
# The moisture lanes: WRF's have_bcs_moist.
# ---------------------------------------------------------------------------
def _p3_frames(root, times=(0, 300, 600)):
    """A history that ran P3 (MP_PHYSICS 50): one free ice category, a rime mass and volume."""

    shape = (synth.NZ, synth.NY, synth.NX)
    extra = {"QCLOUD": np.full(shape, 2.0e-4), "QRAIN": np.full(shape, 1.0e-4),
             "QICE": np.full(shape, 3.0e-4), "QNICE": np.full(shape, 1.0e5), "QNRAIN": np.full(shape, 2.0e3),
             "QIR": np.full(shape, 1.0e-4), "QIB": np.full(shape, 2.0e-7)}
    for name in ("QSNOW", "QGRAUP"):
        extra[name] = None
    frames = []
    for t in times:
        fields = {k: v for k, v in extra.items() if v is not None}
        frames.append(synth.make_frame(root, t, extra=fields, drop=("QSNOW", "QGRAUP"),
                                       attributes={"MP_PHYSICS": 50}))
    return frames


def test_a_p3_history_is_refused_in_the_all_species_lane_and_the_remedy_names_vapour_only(tmp_path):
    frames = _p3_frames(tmp_path / "h")
    config = parse_mapping(document(tmp_path))
    with refused("GM1-CHAIN-MICROPHYSICS") as caught:
        chain.admit_frames(frames, chain.chain_request_from_config(config))
    assert "P3 (50)" in str(caught.value) and "vapour-only" in str(caught.value)


def test_the_vapour_only_lane_admits_any_scheme_and_records_what_it_leaves_out(tmp_path):
    frames = _p3_frames(tmp_path / "h")
    config = parse_mapping(document(tmp_path, moisture="vapour-only"))
    assert config.chain.moisture == "vapour-only"
    request = chain.chain_request_from_config(config)
    assert request.moisture == "vapour-only" and request.contract()["moisture"] == "vapour-only"
    record = chain.admit_frames(frames, request)
    micro = record["microphysics"]
    assert micro["scheme"] == "P3 (50)" and micro["transferred"] == ["qv"]
    assert micro["not_transferred"] == ["qc", "qr", "qi", "qs", "qg"]
    left = micro["left_out_at_t0"]
    assert set(left["window_mean_path_kg_m2"]) == {"QCLOUD", "QRAIN", "QICE"}
    assert left["hydrometeor_path_kg_m2"] > 0 and 0 < left["fraction_of_total_water"] < 0.1
    # 0.6 g/kg of hydrometeor against 10 g/kg of vapour: the paths keep that ratio.
    assert left["hydrometeor_path_kg_m2"] / left["vapour_path_kg_m2"] == pytest.approx(0.06, rel=1e-6)
    receipt = chain.prepare_chain(frames, request, config.chain.prepared_directory)
    assert [c["name"] for c in receipt["carriers"]] == ["u", "v", "w", "pi", "theta", "qv"]
    assert receipt["moisture"]["lane"] == "vapour-only"
    prepared = chain.PreparedChain.open(config.chain.prepared_directory, contract=request.contract())
    assert prepared.moisture == "vapour-only"
    with pytest.raises(KeyError):
        prepared.initial("qc")


def test_the_moisture_key_is_optional_and_refused_by_value(tmp_path):
    absent = parse_mapping(document(tmp_path))
    assert absent.chain.moisture is None and "moisture" not in absent.canonical_dict()["chain"]
    assert chain.chain_request_from_config(absent).contract().get("moisture") is None
    declared = parse_mapping(document(tmp_path, moisture="all-species"))
    assert declared.canonical_dict()["chain"]["moisture"] == "all-species"
    with refused("CM1-CFG-CHAIN-MOISTURE"):
        parse_mapping(document(tmp_path, moisture="liquid"))


def test_the_vapour_only_driver_forces_qv_alone_and_hydrometeors_take_flow_dep_bdy(tmp_path, monkeypatch):
    """WRF have_bcs_moist = .false.: qv relaxed, specified and reset; the rest start at zero and take
    flow_dep_bdy (share/module_bc.F:2335-2456) after each RK scalar stage."""

    from cm1core import arwen_chain_runtime as rt
    from cm1core import nest_live

    monkeypatch.setattr(nest_live, "compile_nest_live_module", lambda cupy: None)
    monkeypatch.setattr(nest_live, "compile_nest_relaxation_module", lambda cupy: None)
    monkeypatch.setattr(rt, "_refill_halos_and_thermodynamics", lambda d: None)
    monkeypatch.setattr(nest_live, "launch_spec_apply", host_spec_apply)
    frames = _p3_frames(tmp_path / "h")
    config = parse_mapping(document(tmp_path, moisture="vapour-only"))
    chain.prepare_chain(frames, chain.chain_request_from_config(config), config.chain.prepared_directory)
    prepared = chain.PreparedChain.open(config.chain.prepared_directory)
    runtime = _fake_runtime(config, prepared.initial("pi"))
    driver = rt.ChainDriver(runtime=runtime, prepared=prepared, spec_zone=1, relax_zone=4,
                            cupy=_fake_cupy(), stream=_Stream())
    forcing = driver.forcing
    assert [c.name for c in forcing.carriers] == ["u", "v", "w", "pp", "th", "q0"]
    assert [c.name for c in forcing.flow_dependent] == ["q1", "q2", "q3", "q4", "q5"]
    assert set(forcing.tables) == {"u", "v", "w", "pp", "th", "q0"}
    d = runtime.dynamics
    d.fields["q3d"][...] = 7.0
    record = driver.initialize()
    assert record["moisture"]["started_at_zero"] == ["qc", "qr", "qi", "qs", "qg"]
    assert np.all(d.fields["q3d"][1:] == 0.0) and np.all(d.fields["qa"][1:] == 0.0)
    assert np.array_equal(d.fields["q3d"][0], prepared.initial("qv"))
    runtime.clock.completed_steps = 0
    driver.before_step()

    nz, ny, nx = d.fields["th3d"].shape
    rng = np.random.default_rng(11)
    q = d.fields["q3d"]
    q[...] = rng.random(q.shape).astype(np.float32) + np.float32(0.5)
    before = q.copy()
    # Mass fluxes: inflow on the west and south faces, outflow on the east and north ones, except
    # one level of each side reversed.
    rru, rrv = d.arrays["rru"], d.arrays["rrv"]
    rru[...] = 1.0
    rrv[...] = -1.0
    rru[1 + 2] = -1.0   # level 2: westward everywhere
    rrv[1 + 2] = 1.0    # level 2: northward everywhere
    forcing.after_scalar_stage(d, 1)
    for s in range(1, 6):
        f, b = q[s], before[s]
        # Interior untouched.
        assert np.array_equal(f[:, 1:-1, 1:-1], b[:, 1:-1, 1:-1])
        for k in range(nz):
            westward, southward = (k == 2), (k != 2)
            # South row (owns the corners): v < 0 at its outer face is outflow -> first interior row.
            want = b[k, 1, np.clip(np.arange(nx), 1, nx - 2)] if southward else np.zeros(nx, np.float32)
            assert np.array_equal(f[k, 0, :], want), (s, k, "south")
            # North row: v > 0 at its outer face is outflow.
            want = b[k, ny - 2, np.clip(np.arange(nx), 1, nx - 2)] if not southward else np.zeros(nx, np.float32)
            assert np.array_equal(f[k, ny - 1, :], want), (s, k, "north")
            # West column (rows 1..ny-2): u < 0 is outflow.
            want = b[k, 1:ny - 1, 1] if westward else np.zeros(ny - 2, np.float32)
            assert np.array_equal(f[k, 1:ny - 1, 0], want), (s, k, "west")
            want = b[k, 1:ny - 1, nx - 2] if not westward else np.zeros(ny - 2, np.float32)
            assert np.array_equal(f[k, 1:ny - 1, nx - 1], want), (s, k, "east")
    # qv is not flow-dependent: its rows are untouched by this rule.
    assert np.array_equal(q[0], before[0])
    # spec_bdy_final: u, v, pi', theta and qv only.
    record = driver.record()["lateral_forcing"]
    assert record["spec_bdy_final"]["carriers"] == ["u", "v", "pp", "th", "q0"]
    assert record["moisture"]["flow_dependent_species"] == ["qc", "qr", "qi", "qs", "qg"]
    assert record["moisture"]["flow_dependent_stages"] == 1
    assert "flow_dep_bdy" in record["moisture"]["flow_dep_bdy_authority"]
