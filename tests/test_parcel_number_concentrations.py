"""A double-moment run's parcels publish its number concentrations, as CM1's do.

``src/param.F:6435-6439`` adds one parcel attribute per ``n = nnc1..nnc2`` when
``prcl_nc = 1`` and ``idm = 1`` (``:6269-6271`` switches it off for a
single-moment axis), named ``qname(n)`` in ``1/kg`` (``parcel.F:2013-2024``),
immediately after ``prcl_q``'s mass rows.  ``parcel_interp`` samples each one as
``q3d(n)`` on the scalar-point stencil with the same eight weights it uses for
``qv`` and the hydrometeor mass rows (``parcel.F:1424-1432``).

The attribute table was written for WSM6 (``idm = 0``), so ``prcl_nc`` added
nothing whatever the component: a ``morrison-mp10`` run -- ``prcl_nc`` defaults
to 1 -- published a parcel track without nci, ncs, ncr or ncg.  These tests
run no CUDA: the layout, the sampler, the stage and the runtime's stage builder
are the real code on NumPy.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cm1core.lagrangian_parcels import (
    PARCEL_SWITCH_ORDER,
    cm1_parcel_switch_context,
    parcel_attribute_layout,
    parcel_attribute_units,
    parcel_number_concentrations,
)
from cm1core.moisture_species import MORRISON_PROFILE, WSM6_PROFILE, moisture_species_profile

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SEED = 20260916


def _bits(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)


def test_the_number_rows_are_param_F_nnc1_to_nnc2_by_qname():
    # param.F:3427-3428 (nnc1=7, nnc2=10) and :3441-3450 (qname 7..10), zero-based.
    assert parcel_number_concentrations(MORRISON_PROFILE) == (
        ("nci", 6), ("ncs", 7), ("ncr", 8), ("ncg", 9),
    )
    # param.F:6268-6270: idm=0 publishes none.
    assert parcel_number_concentrations(WSM6_PROFILE) == ()
    thompson = moisture_species_profile("thompson-mp8")
    assert parcel_number_concentrations(thompson) == (("nci", 6), ("ncr", 7))


def _context(profile, *, cm1setup=1, ipbl=0, sfcmodel=0, bbc=1):
    """``profile``'s axis on CM1's LES deck, or on another closure/surface."""

    return cm1_parcel_switch_context(imoist=1, profile=profile, cm1setup=cm1setup, sgsmodel=1,
                                     ipbl=ipbl, sfcmodel=sfcmodel, oceanmodel=1, bbc=bbc, iptra=0,
                                     terrain=False)


def test_prcl_nc_adds_the_number_rows_after_the_mass_rows_in_1_per_kg():
    switches = {name: 1 for name in PARCEL_SWITCH_ORDER}
    rows = parcel_number_concentrations(MORRISON_PROFILE)
    # cm1setup=4 with ipbl=1 is the one deck param.F:6273-6276 keeps prcl_km on.
    context = _context(MORRISON_PROFILE, cm1setup=4, ipbl=1, sfcmodel=1, bbc=3)
    layout = parcel_attribute_layout(switches, context)
    # param.F:6422-6448: prcl_q's qv..qg, then prcl_nc's nnc1..nnc2, then prcl_km.
    assert layout.names[10:16] == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert layout.names[16:20] == ("nci", "ncs", "ncr", "ncg")
    assert layout.names[20:24] == ("kmh", "kmv", "khh", "khv")
    assert layout.number_rows == rows
    record = layout.record()
    assert record["schema"] == "gpuwm-cm1.lagrangian-parcels/v3"
    assert record["units"][16:20] == ["1/kg"] * 4
    assert record["number_concentration_rows"] == {"nci": 6, "ncs": 7, "ncr": 8, "ncg": 9}
    assert all(parcel_attribute_units(name) == "1/kg" for name, _ in rows)
    # The switch off removes exactly those four, and binds no rows to sample.
    off = parcel_attribute_layout({**switches, "prcl_nc": 0}, context)
    assert set(layout.names) - set(off.names) == {"nci", "ncs", "ncr", "ncg"}
    assert off.number_rows == ()
    assert "number_concentration_rows" not in off.record()
    # Requested off is not an override: the record names only what param.F changed.
    assert "prcl_nc" not in off.record()["switch_overrides"]


def test_a_single_moment_axis_resolves_prcl_nc_off_and_says_why():
    """param.F:6269-6271: ``idm = 0`` turns ``prcl_nc`` off, and CM1 prints the 0."""

    for switches in ({name: 1 for name in PARCEL_SWITCH_ORDER},
                     {"prcl_th": 1, "prcl_q": 1, "prcl_nc": 1, "prcl_vort": 1}):
        wsm6 = parcel_attribute_layout(switches, _context(WSM6_PROFILE))
        requested_off = parcel_attribute_layout({**switches, "prcl_nc": 0}, _context(WSM6_PROFILE))
        assert wsm6.names == requested_off.names
        assert wsm6.switches == requested_off.switches
        record = wsm6.record()
        assert record["switches"]["prcl_nc"] == 0
        assert record["requested_switches"]["prcl_nc"] == 1
        assert record["switch_overrides"]["prcl_nc"] == "src/param.F:6269-6271: idm=0"
        assert "number_concentration_rows" not in record


def test_a_number_row_that_collides_with_an_attribute_is_refused():
    import dataclasses

    context = _context(MORRISON_PROFILE)
    for rows in ((("qv", 6),), (("nci", 6), ("nci", 7)), (("nci", -1),)):
        with pytest.raises(ValueError, match="distinct new rows"):
            dataclasses.replace(context, number_rows=rows)


def _deck():
    from test_nest_parcels import child_deck, geometry

    # A stretched child deck (5 m at the ground under a 1500 m top).
    return child_deck(geometry())


def _runtime(deck, q3d):
    nz, ny, nx = deck.nz, deck.ny, deck.nx
    mass, face = (nz, ny, nx), (nz + 1, ny, nx)
    full = lambda value, shape: np.full(shape, np.float32(value), dtype=np.float32)  # noqa: E731
    return SimpleNamespace(
        config=SimpleNamespace(bbc=1, tbc=1, u_move_m_s=0.0, v_move_m_s=0.0),
        dynamics=SimpleNamespace(
            arrays={},
            fields={"th3d": full(0.5, mass), "th0": full(300.0, mass), "pi0": full(0.9, mass),
                    "pp3d": full(0.0, mass), "qv0": full(0.0, mass), "prs": full(9.0e4, mass),
                    "rho": full(1.1, mass), "q3d": q3d},
            tke={"tke3d": full(0.25, face)},
            metrics={"c1": full(0.5, face), "c2": full(0.5, face)},
        ),
    )


def _positions(deck, rng, n=64):
    x = rng.uniform(deck.minx_m + 0.2 * deck.dx_m, deck.minx_m + (deck.nx - 0.2) * deck.dx_m, n)
    y = rng.uniform(deck.miny_m + 0.2 * deck.dy_m, deck.miny_m + (deck.ny - 0.2) * deck.dy_m, n)
    z = rng.uniform(0.0, float(deck.zh[deck.nz]) + 30.0, n)
    # Exactly on half levels and faces, where the stencil choice matters.
    z[:6] = deck.zh[1:7]
    x[6:12] = [deck.face_x(i) for i in range(2, 8)]
    return np.array([x, y, z], dtype=np.float32)


def test_a_number_row_is_sampled_exactly_as_parcel_interp_samples_a_mass_row():
    """``parcel.F:1424-1432``: ``q3d(n)`` for nnc1..nnc2 on the stencil ``qc`` uses.

    Runtime B holds runtime A's number-row words in its qc..qs rows; B's qc..qs
    samples are then the samples of those words through the path every WSM6
    track has published, and A's nci..ncg must be exactly them.
    """

    from cm1core import nest_parcels as npc

    rng = np.random.default_rng(SEED)
    deck = _deck()
    mass = (deck.nz, deck.ny, deck.nx)
    q3d = rng.uniform(0.0, 1.0e-3, (10, *mass)).astype(np.float32)
    q3d[6:] = rng.uniform(0.0, 2.0e5, (4, *mass)).astype(np.float32)
    swapped = q3d[:6].copy()
    swapped[1:5] = q3d[6:10]
    positions = _positions(deck, rng)
    rows = dict(parcel_number_concentrations(MORRISON_PROFILE))
    wide = npc.sample_parcel_attributes(
        np, npc.RuntimeCarriers(_runtime(deck, q3d), deck, np, 3), positions,
        ["qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg"], rows,
    )
    reference = npc.sample_parcel_attributes(
        np, npc.RuntimeCarriers(_runtime(deck, swapped), deck, np, 3), positions,
        ["qc", "qr", "qi", "qs"],
    )
    for number, mass_name in zip(("nci", "ncs", "ncr", "ncg"), ("qc", "qr", "qi", "qs")):
        assert np.array_equal(_bits(wide[number]), _bits(reference[mass_name])), number
        assert np.all(wide[number] > 1.0), number  # sampled number rows, not mass rows
    # And the mass rows themselves are untouched by carrying the number rows.
    plain = npc.sample_parcel_attributes(
        np, npc.RuntimeCarriers(_runtime(deck, np.ascontiguousarray(q3d[:6])), deck, np, 3),
        positions, ["qv", "qc", "qr", "qi", "qs", "qg"],
    )
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        assert np.array_equal(_bits(wide[name]), _bits(plain[name])), name


def _morrison_runtime_config():
    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import parse_mapping

    document = tomllib.loads((EXAMPLES / "cartesian-uniform-grid.toml").read_text(encoding="utf-8"))
    document["physics"]["component"] = "morrison-mp10"
    document["lagrangian"] = {
        "iprcl": 1, "nparcels": 4, "seeding": "box", "seed_nx": 2, "seed_ny": 2, "seed_nz": 1,
        "seed_x0_m": 20000.0, "seed_y0_m": 20000.0, "seed_z0_m": 1000.0,
    }
    return build_runtime_config(parse_mapping(document))


def test_the_runtime_stage_of_a_morrison_run_publishes_its_number_concentrations(tmp_path):
    """From the configuration text through the stage builder to the track's columns."""

    from cm1core import nest_parcels as npc
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
    from cm1core.lagrangian_stage import PARCEL_TABLE_FILENAME, write_parcel_track

    config = _morrison_runtime_config()
    assert config.profile.component_id == "morrison-mp10"
    stage = CudaCartesianRuntime._build_lagrangian_stage(SimpleNamespace(config=config, cp=np))
    assert stage.layout.number_rows == (("nci", 6), ("ncs", 7), ("ncr", 8), ("ncg", 9))
    names = stage.attribute_names()
    assert names[names.index("qg") + 1:names.index("qg") + 5] == ("nci", "ncs", "ncr", "ncg")
    assert stage.attribute_units()[names.index("nci")] == "1/kg"
    configuration = stage.record()["configuration"]["parcel_attributes"]
    assert configuration["names"] == list(stage.layout.names)
    assert configuration["number_concentration_rows"] == {"nci": 6, "ncs": 7, "ncr": 8, "ncg": 9}

    deck = stage.deck
    mass = (deck.nz, deck.ny, deck.nx)
    q3d = np.zeros((10, *mass), np.float32)
    for n, value in zip(range(6, 10), (5.0e4, 2.0e5, 2.0e4, 1.0e3)):
        q3d[n] = np.float32(value)
    runtime = _runtime(deck, q3d)
    runtime.dynamics.arrays = {name: np.zeros(shape, np.float32) for name, shape in (
        ("u3d", (deck.nz + 2, deck.ny + 6, deck.nx + 7)),
        ("v3d", (deck.nz + 2, deck.ny + 7, deck.nx + 6)),
        ("w3d", (deck.nz + 3, deck.ny + 6, deck.nx + 6)),
    )}
    frame = stage.parcel_frame(step=1, model_time_s=1.0, carriers=npc.RuntimeCarriers(runtime, deck, np, 3))
    columns = frame["attributes"]
    placed = columns[stage.layout.index("x")] > -1.0e29
    assert placed.any()
    for name, value in zip(("nci", "ncs", "ncr", "ncg"), (5.0e4, 2.0e5, 2.0e4, 1.0e3)):
        assert columns[stage.layout.index(name), placed].tolist() == pytest.approx([value] * int(placed.sum()))
    write_parcel_track(tmp_path / "track", stage, [frame])
    table = (tmp_path / "track" / PARCEL_TABLE_FILENAME).read_text().splitlines()
    assert any(line.split("\t")[1:3] == ["nci", "1/kg"] for line in table[1:])
    assert any(line.split("\t")[1:3] == ["ncg", "1/kg"] for line in table[1:])


def test_a_wsm6_runtime_stage_is_the_stage_its_namelist_words_build():
    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import parse_mapping
    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
    from cm1core.lagrangian_config import parse_lagrangian_table
    from cm1core.lagrangian_stage import LagrangianStage

    document = tomllib.loads((EXAMPLES / "cartesian-uniform-grid.toml").read_text(encoding="utf-8"))
    table = {"iprcl": 1, "nparcels": 4, "seeding": "box", "seed_nx": 2, "seed_ny": 2, "seed_nz": 1,
             "seed_x0_m": 20000.0, "seed_y0_m": 20000.0, "seed_z0_m": 1000.0}
    document["lagrangian"] = table
    config = build_runtime_config(parse_mapping(document))
    assert config.profile.numq == 6
    built = CudaCartesianRuntime._build_lagrangian_stage(SimpleNamespace(config=config, cp=np))
    g = config.grid
    legacy = LagrangianStage.build(
        parse_lagrangian_table(table), nx=g.nx, ny=g.ny, nz=g.nz, dx_m=g.dx_m, dy_m=g.dy_m,
        dz_m=g.dz_m, dt_s=config.dt_s, xp=np, iorigin=int(config.iorigin),
        switch_context=_context(WSM6_PROFILE),
    )
    assert built.layout == legacy.layout
    assert built.attribute_units() == legacy.attribute_units()
    assert built.record()["configuration"] == legacy.record()["configuration"]


def test_the_nest_child_tracker_samples_number_rows_on_the_grid_that_owns_the_parcel(tmp_path):
    """The nest tracker hands the same rows to both grids' samplers (one component, both domains)."""

    from test_nest_parcels import _Clock, _settings, child_deck, geometry, padded_winds, parent_deck

    from cm1core import nest_parcels as npc

    g = geometry()
    rows = parcel_number_concentrations(MORRISON_PROFILE)
    layout = parcel_attribute_layout({"prcl_th": 1, "prcl_q": 1, "prcl_nc": 1, "prcl_rho": 1},
                                     _context(MORRISON_PROFILE))
    assert layout.number_rows == rows
    settings = _settings(nparcels=3, seed_nx=3, seed_dx_m=725.0, seed_x0_m=4700.0)
    tracker = npc.NestParcelTracker(settings=settings, layout=layout, child_deck=child_deck(g),
                                    parent_deck=parent_deck(), child_dt_s=1.0, parent_dt_s=3.0,
                                    xp=np, ng=3)
    tracker.seed(g, model_time_s=0.0, parent_step=0)
    still = lambda x, y, z: (0 * x, 0 * y, 0 * z)  # noqa: E731

    def grid_runtime(deck, base):
        mass = (deck.nz, deck.ny, deck.nx)
        q3d = np.zeros((10, *mass), np.float32)
        for n in range(6, 10):
            q3d[n] = np.float32(base * (n - 5))
        runtime = _runtime(deck, q3d)
        u, v, w = padded_winds(deck, still)
        runtime.dynamics.arrays = {"u3d": u, "v3d": v, "w3d": w, "rru": u, "rrv": v, "rrw": w}
        return runtime

    child = grid_runtime(child_deck(g), 1000.0)
    parent = grid_runtime(parent_deck(), 7.0)
    driver = SimpleNamespace(geometry=g, parent=SimpleNamespace(clock=_Clock(), **vars(parent)), child=child)
    tracker.open_track(tmp_path / "track")
    positions = np.asarray(tracker.positions)
    # The third parcel as the parent's (a hand-off sets this; the sampler reads only the code).
    owner = np.asarray(tracker.owner).copy()
    assert owner.tolist() == [npc.OWNER_CHILD] * 3
    owner[2] = npc.OWNER_PARENT
    tracker.record_frame(driver, step=1, model_time_s=3.0, positions=positions, owner=owner)
    receipt = tracker.publish_track()
    names = receipt["attribute_order"]
    # The child track's receipt carries the resolved layout record, not only the names.
    assert receipt["tracker"]["schema"] == "gm1.cartesian-nest-parcels/v2"
    assert receipt["tracker"]["parcel_attributes"] == layout.record()
    assert receipt["tracker"]["parcel_attributes"]["switches"]["prcl_nc"] == 1
    assert names[names.index("qg") + 1:names.index("qg") + 5] == ["nci", "ncs", "ncr", "ncg"]
    arena = np.frombuffer((tmp_path / "track" / npc.NEST_PARCEL_ARENA_FILENAME).read_bytes(),
                          "<f4").reshape(1, len(names), -1)
    for p, code in enumerate(owner.tolist()):
        base = 1000.0 if code == npc.OWNER_CHILD else 7.0
        for m, name in enumerate(("nci", "ncs", "ncr", "ncg")):
            assert float(arena[0, names.index(name), p]) == pytest.approx(base * (m + 1)), (p, name)
