"""From configuration text to changed behaviour, and back to the receipt.

This is the door test for the ``[lagrangian]`` table: every capability it can
select is followed from the TOML bytes through the parser, the sealed
document, the runtime contract and the stage, to a *measurable* difference in
what the stage does -- parcels that move, circuits that close, a window that
is written, forcings that appear.  It runs no CUDA.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from cm1core import doctor
from cm1core.cartesian_door import build_runtime_config
from cm1core.config import load_config, parse_mapping
from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.errors import CM1Refusal
from cm1core.high_frequency_output import (
    HIFRQ_ARENA_FILENAME,
    HIFRQ_TABLE_FILENAME,
    HighFrequencyWindow,
    full_domain_window,
    high_frequency_cadence,
    write_high_frequency_window,
)
from cm1core.lagrangian_config import LAGRANGIAN_KEYS, parse_lagrangian_table
from cm1core.lagrangian_parcels import ParcelVelocityField, parcel_switch_context
from cm1core.lagrangian_stage import (
    PARCEL_ARENA_FILENAME,
    PARCEL_RECEIPT_FILENAME,
    PARCEL_TABLE_FILENAME,
    UNVALIDATED_MARKERS,
    LagrangianStage,
    write_parcel_track,
)
from cm1core.cuda_pdcomp_open import pdcomp_selection_defect
from cm1core.pressure_decomposition import (
    PDCOMP_SCALAR_NAMES,
    PDCOMP_UNVALIDATED,
    PDCOMP_W_NAMES,
    pdcomp_selection_for,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SHIPPED = EXAMPLES / "cartesian-tornado-lagrangian-analysis.toml"
BASE = EXAMPLES / "cartesian-uniform-grid.toml"


def _base_document() -> dict:
    return tomllib.loads(BASE.read_text(encoding="utf-8"))


# --------------------------------------------------------------- the grammar


def test_a_document_without_the_table_keeps_its_exact_sealed_identity():
    """Every recorded receipt's plan_sha256 names a document hashed this way.

    The identities are the ones the rest of the suite pins
    (test_default_configuration_is_unchanged, test_cuda_surface_drag): three of
    these documents were legitimately re-sealed by later merged work (the
    variable large step, the surface and output tables), and what this test
    guards is only that the absent [lagrangian] table adds nothing to them.
    """

    expected = {
        "cartesian-uniform-grid":
            "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48",
        "cartesian-supercell-attempt":
            "f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b",
        "official-supercell-product":
            "b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038",
        "cartesian-stretched-tornado-grade-grid":
            "d4f0b4a9fa9bf371a6e726067fa4ce4998925cbdd876831fbad6365212a9d811",
    }
    for name, sha in expected.items():
        config = load_config(EXAMPLES / f"{name}.toml")
        assert config.lagrangian is None
        assert "lagrangian" not in config.canonical_dict()
        assert config.sha256 == sha, name


def test_declaring_the_table_projects_every_key_it_resolves():
    document = {**_base_document(), "lagrangian": {"iprcl": 1, "nparcels": 8}}
    config = parse_mapping(document)
    projected = config.canonical_dict()["lagrangian"]
    # [lagrangian.nest_child] and [lagrangian.nest_child_window] are sub-tables
    # projected only when declared, so a document that omits them keeps its
    # bytes (cm1core.nest_parcels, cm1core.nest_child_window).  The two
    # vorticity-budget selectors are projected only when true, for the same
    # reason (lagrangian_config._PROJECTED_WHEN_TRUE).
    from cm1core.lagrangian_config import _PROJECTED_WHEN_TRUE

    sub_tables = {"nest_child", "nest_child_window"}
    assert set(projected) == set(LAGRANGIAN_KEYS) - sub_tables - _PROJECTED_WHEN_TRUE
    assert not _PROJECTED_WHEN_TRUE & set(projected)
    # The two the document wrote...
    assert projected["iprcl"] == 1 and projected["nparcels"] == 8
    # ...and everything else stated by value rather than inherited.
    assert projected["prclfrq_s"] == 60.0
    # CM1 has one iorigin and this port spells it once, in [initialization].
    assert "iorigin" not in projected
    assert projected["output_pdcomp"] == 0
    assert projected["dohifrq"] is False


def test_an_unknown_lagrangian_key_refuses_by_name():
    document = {**_base_document(), "lagrangian": {"iprcl": 1, "iprcl_typo": 1}}
    with pytest.raises(CM1Refusal) as excinfo:
        parse_mapping(document)
    assert excinfo.value.detail.code == "CM1-CFG-UNKNOWN"
    assert "lagrangian.iprcl_typo" in str(excinfo.value)


@pytest.mark.parametrize(
    "table, fragment",
    [
        ({"iprcl": 2}, "must be 0 or 1"),
        ({"iprcl": True}, "not a TOML boolean"),
        ({"direction": "sideways"}, "forward"),
        ({"seeding": "everywhere"}, "seeding must be one of"),
        ({"seed_dx_m": 0.0}, "stacks every parcel"),
        ({"prclfrq_s": -60.0}, "never records them"),
        ({"iprcl": 1, "circuits": 1, "circuit_parcels": 2,
          "circuit_radius_m": 100.0}, "at least 3"),
        ({"iprcl": 1, "circuits": 1, "circuit_parcels": 8,
          "circuit_radius_m": 0.0}, "encloses no area"),
        ({"circuits": 1, "circuit_parcels": 8, "circuit_radius_m": 100.0},
         "requires lagrangian.iprcl = 1"),
        ({"iprcl": 1, "nparcels": 8, "circuits": 2, "circuit_parcels": 8,
          "circuit_radius_m": 100.0}, "past the parcel array"),
        ({"vorticity_budget": True, "vorticity_budget_frq_s": 0.0},
         "never publish it"),
        ({"dohifrq": True, "hifrqfrq_s": 0.0}, "terabytes"),
    ],
)
def test_every_inadmissible_lagrangian_value_refuses_and_names_the_defect(table, fragment):
    document = {**_base_document(), "lagrangian": table}
    with pytest.raises(CM1Refusal) as excinfo:
        parse_mapping(document)
    assert excinfo.value.detail.code == "CM1-CFG-LAGRANGIAN-VALUE"
    assert fragment in str(excinfo.value)


def test_the_retired_lagrangian_iorigin_spelling_is_refused_by_name():
    """A second, separately declarable iorigin could disagree with [initialization]'s."""

    document = {**_base_document(), "lagrangian": {"iprcl": 1, "iorigin": 2}}
    with pytest.raises(CM1Refusal) as excinfo:
        parse_mapping(document)
    assert excinfo.value.detail.code == "CM1-CFG-UNKNOWN"
    assert "lagrangian.iorigin" in str(excinfo.value)


def test_the_door_and_the_runtime_share_the_initialization_frame():
    """2e: the circuit extent check and the runtime's parcel frame are one iorigin."""

    for iorigin, centre in ((1, 24000.0), (2, 0.0)):
        document = {**_base_document(), "lagrangian": {
            "iprcl": 1, "nparcels": 64, "circuits": 1, "circuit_parcels": 32,
            "circuit_radius_m": 2000.0, "circuit_center_x_m": centre,
            "circuit_center_y_m": centre, "circuit_center_z_m": 1000.0,
            "seeding": "box", "seed_nx": 1, "seed_ny": 1, "seed_nz": 1,
            "seed_x0_m": centre, "seed_y0_m": centre, "seed_z0_m": 1000.0,
        }}
        document["initialization"] = {"isnd": 5, "iwnd": 2, "iorigin": iorigin}
        runtime = build_runtime_config(parse_mapping(document))
        assert runtime.iorigin == iorigin
        from types import SimpleNamespace

        from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime

        fake = SimpleNamespace(config=runtime, cp=np)
        stage = CudaCartesianRuntime._build_lagrangian_stage(fake)
        assert stage.grid.iorigin == iorigin
        record = stage.seed()
        assert record["placed_parcels"] == 32
        ring = stage.positions[:, :32]
        assert float(ring[0].min()) >= stage.grid.minx_m
        assert float(ring[0].max()) <= stage.grid.maxx_m
        other = 2 if iorigin == 1 else 1
        document["initialization"]["iorigin"] = other
        with pytest.raises(CM1Refusal) as excinfo:
            build_runtime_config(parse_mapping(document))
        assert excinfo.value.detail.code == "CM1-CARTESIAN-CIRCUIT-EXTENT"


def test_a_valid_table_is_not_refused():
    """A refusal that fires on valid input is a defect, so check the converse."""

    config = load_config(SHIPPED)
    assert config.lagrangian is not None
    runtime = build_runtime_config(config)
    assert runtime.lagrangian is config.lagrangian


def test_the_door_refuses_a_window_outside_the_grid_rather_than_clipping_it():
    document = {**_base_document(), "lagrangian": {
        "dohifrq": True, "hifrqfrq_s": 30.0,
        "hifrq_i0": 0, "hifrq_i1": 9999, "hifrq_j0": 0, "hifrq_j1": 4,
        "hifrq_k0": 0, "hifrq_k1": 4,
    }}
    with pytest.raises(CM1Refusal) as excinfo:
        build_runtime_config(parse_mapping(document))
    assert excinfo.value.detail.code == "CM1-CARTESIAN-HIFRQ-WINDOW"


def test_the_door_refuses_a_circuit_the_periodic_exit_would_wrap():
    document = {**_base_document(), "lagrangian": {
        "iprcl": 1, "nparcels": 64, "circuits": 1, "circuit_parcels": 32,
        "circuit_radius_m": 900000.0, "circuit_center_z_m": 1000.0,
    }}
    with pytest.raises(CM1Refusal) as excinfo:
        build_runtime_config(parse_mapping(document))
    assert excinfo.value.detail.code == "CM1-CARTESIAN-CIRCUIT-EXTENT"
    assert "parcel.F:577-592" in str(excinfo.value)


def test_a_circuit_inside_the_grid_is_admitted():
    # No [initialization]: iorigin = 1, so the 48 km grid runs 0..48000 m.
    document = {**_base_document(), "lagrangian": {
        "iprcl": 1, "nparcels": 64, "circuits": 1, "circuit_parcels": 32,
        "circuit_radius_m": 2000.0, "circuit_center_x_m": 24000.0,
        "circuit_center_y_m": 24000.0, "circuit_center_z_m": 1000.0,
    }}
    runtime = build_runtime_config(parse_mapping(document))
    assert runtime.lagrangian.circuits_enabled


def test_the_table_is_refused_outside_the_cartesian_lane():
    document = tomllib.loads(
        (EXAMPLES / "official-supercell-product.toml").read_text(encoding="utf-8")
    )
    document["lagrangian"] = {"iprcl": 1}
    with pytest.raises(CM1Refusal) as excinfo:
        parse_mapping(document)
    assert excinfo.value.detail.code == "CM1-CFG-UNKNOWN"


# ----------------------------------------------- configuration -> behaviour


def _uniform_field(nx, ny, nz, *, u=0.0, v=0.0, w=0.0):
    shapes = terrain_open_sound_shapes(nx, ny, nz)
    return ParcelVelocityField(
        np.full(shapes["u3d"], np.float32(u)),
        np.full(shapes["v3d"], np.float32(v)),
        np.full(shapes["w3d"], np.float32(w)),
    )


def _stage(table, *, nx=24, ny=24, nz=20, dt_s=1.0):
    config = parse_lagrangian_table(table)
    # These stages seed about (0, 0): the domain-centred frame.
    from test_lagrangian_parcels import wsm6_switch_context

    return LagrangianStage.build(
        config, nx=nx, ny=ny, nz=nz, dx_m=1000.0, dy_m=1000.0, dz_m=500.0,
        dt_s=dt_s, xp=np, iorigin=2, switch_context=wsm6_switch_context(),
    )


def _carriers(stage, *, u=0.0, v=0.0, w=0.0, th0=300.0, th3d=0.0):
    """A runtime's live carriers as parcel_interp reads them (cm1.F:4010-4020)."""

    from types import SimpleNamespace

    from cm1core.nest_parcels import RuntimeCarriers

    deck = stage.deck
    nz, ny, nx = deck.nz, deck.ny, deck.nx
    state = _uniform_field(nx, ny, nz, u=u, v=v, w=w)
    mass, face = (nz, ny, nx), (nz + 1, ny, nx)
    full = lambda value, shape: np.full(shape, np.float32(value), dtype=np.float32)  # noqa: E731
    runtime = SimpleNamespace(
        config=SimpleNamespace(bbc=stage.bbc, tbc=stage.tbc, u_move_m_s=0.0, v_move_m_s=0.0),
        dynamics=SimpleNamespace(
            arrays={"u3d": state.u, "v3d": state.v, "w3d": state.w},
            fields={"th3d": full(th3d, mass), "th0": full(th0, mass), "pi0": full(0.9, mass),
                    "pp3d": full(0.0, mass), "qv0": full(0.0, mass), "prs": full(9.0e4, mass),
                    "rho": full(1.1, mass), "q3d": np.zeros((6, *mass), np.float32)},
            tke={"tke3d": full(0.25, face)},
            metrics={"c1": full(0.5, face), "c2": full(0.5, face)},
        ),
    )
    return RuntimeCarriers(runtime, deck, np, 3)


def test_iprcl_0_integrates_nothing_and_iprcl_1_moves_every_parcel():
    table = {"nparcels": 16, "seeding": "box", "seed_nx": 4, "seed_ny": 2,
             "seed_nz": 2, "seed_x0_m": -2000.0, "seed_y0_m": -2000.0,
             "seed_z0_m": 1000.0, "seed_dx_m": 500.0, "seed_dy_m": 500.0,
             "seed_dz_m": 500.0}
    field = _uniform_field(24, 24, 20, u=12.0)
    off = _stage({**table, "iprcl": 0})
    off.seed()
    before = off.positions.copy()
    off.advance(field)
    assert off.steps_taken == 0
    assert np.array_equal(off.positions, before)

    on = _stage({**table, "iprcl": 1})
    on.seed()
    before = on.positions.copy()
    for _ in range(5):
        on.advance(field)
    assert on.steps_taken == 5
    assert np.allclose(on.positions[0] - before[0], 60.0, atol=1e-3)


def test_the_seeding_mode_selected_in_text_is_the_seeding_that_happens():
    grid_stage = _stage({"iprcl": 1, "nparcels": 36000, "seeding": "grid"})
    grid_record = grid_stage.seed()
    assert grid_record["seeding"]["mode"] == "grid"
    assert grid_record["seeding"]["beyond_cm1"] is False

    box_stage = _stage({"iprcl": 1, "nparcels": 24, "seeding": "box",
                        "seed_nx": 3, "seed_ny": 2, "seed_nz": 2,
                        "seed_x0_m": 0.0, "seed_y0_m": 0.0, "seed_z0_m": 500.0})
    box_record = box_stage.seed()
    assert box_record["seeding"]["beyond_cm1"] is True
    assert box_record["placed_parcels"] == 12
    assert float(box_stage.positions[0, 0]) == 0.0


def test_the_prcl_switches_selected_in_text_decide_the_published_columns():
    lean = _stage({"iprcl": 1, "nparcels": 4,
                   **{name: 0 for name in
                      ("prcl_th", "prcl_t", "prcl_prs", "prcl_ptra", "prcl_q",
                       "prcl_nc", "prcl_km", "prcl_kh", "prcl_tke", "prcl_dbz",
                       "prcl_b", "prcl_vpg", "prcl_vort", "prcl_rho",
                       "prcl_qsat", "prcl_sfc")},
                   "seeding": "box", "seed_nx": 2, "seed_ny": 2, "seed_nz": 1})
    assert lean.layout.npvals == 7
    rich = _stage({"iprcl": 1, "nparcels": 4, "seeding": "box",
                   "seed_nx": 2, "seed_ny": 2, "seed_nz": 1})
    assert rich.layout.npvals > lean.layout.npvals
    assert "zvort" in rich.layout.names and "zvort" not in lean.layout.names


def test_circuits_declared_in_text_occupy_the_front_of_the_parcel_array():
    stage = _stage({"iprcl": 1, "nparcels": 64, "circuits": 2,
                    "circuit_parcels": 16, "circuit_radius_m": 1500.0,
                    "circuit_center_z_m": 1000.0, "circuit_spacing_z_m": 500.0,
                    "seeding": "box", "seed_nx": 4, "seed_ny": 4, "seed_nz": 2,
                    "seed_z0_m": 1000.0})
    record = stage.seed()
    assert [c["first_parcel"] for c in record["circuits"]] == [0, 16]
    ring0 = stage.positions[:, 0:16]
    ring1 = stage.positions[:, 16:32]
    assert float(np.hypot(ring0[0], ring0[1]).mean()) == pytest.approx(1500.0, rel=1e-4)
    assert float(ring0[2].mean()) == pytest.approx(1000.0)
    assert float(ring1[2].mean()) == pytest.approx(1500.0)
    field = _uniform_field(24, 24, 20, u=5.0)
    stage.advance(field)
    report = stage.circulation_frame(model_time_s=1.0)
    assert [c["name"] for c in report["circuits"]] == ["circuit00", "circuit01"]
    assert report["unvalidated"] == UNVALIDATED_MARKERS["circulation"]


def test_a_recorded_frame_carries_the_attributes_and_names_the_unfilled_ones():
    stage = _stage({"iprcl": 1, "nparcels": 4, "seeding": "box",
                    "seed_nx": 2, "seed_ny": 2, "seed_nz": 1,
                    "seed_z0_m": 1000.0})
    stage.seed()
    # The advecting average and the completed state differ on purpose: CM1
    # steps with the first (solve3.F:768) and records the second
    # (cm1.F:4010-4020, parcel.F:1218-1343).
    field = _uniform_field(24, 24, 20, u=3.0, w=-0.25)
    stage.advance(field)
    carriers = _carriers(stage, u=7.0, w=0.5, th0=300.0, th3d=3.25)
    frame = stage.parcel_frame(step=1, model_time_s=1.0, carriers=carriers)
    rows = frame["attributes"]
    assert rows.shape == (stage.layout.npvals, 4)
    assert float(rows[stage.layout.index("u"), 0]) == pytest.approx(7.0)
    assert float(rows[stage.layout.index("w"), 0]) == pytest.approx(0.5)
    assert float(rows[stage.layout.index("mtime"), 0]) == pytest.approx(1.0)
    assert float(rows[stage.layout.index("th"), 0]) == pytest.approx(303.25)
    assert stage.velocities[:, 0].tolist() == pytest.approx([7.0, 0.0, 0.5], abs=1e-5)
    # dbz/qsl/qsi have no carrier in this lane and stay at the sentinel (kmh..khv
    # and znt/ust are not laid out at all: param.F:6273-6276 and :6280-6282).
    assert set(frame["unfilled"]) == set(stage.layout.unfilled)
    for name in stage.layout.unfilled:
        assert float(rows[stage.layout.index(name), 0]) < -1.0e29


def test_the_track_writer_publishes_an_arena_a_table_and_a_receipt(tmp_path):
    stage = _stage({"iprcl": 1, "nparcels": 6, "seeding": "box",
                    "seed_nx": 3, "seed_ny": 2, "seed_nz": 1,
                    "seed_z0_m": 1000.0})
    stage.seed()
    field = _uniform_field(24, 24, 20, u=3.0)
    frames = []
    for step in range(1, 4):
        stage.advance(field)
        frames.append(stage.parcel_frame(step=step, model_time_s=float(step)))
    receipt = write_parcel_track(tmp_path / "track", stage, frames)
    root = tmp_path / "track"
    assert receipt["arena_shape_frames_npvals_nparcels"] == [3, stage.layout.npvals, 6]
    raw = (root / PARCEL_ARENA_FILENAME).read_bytes()
    assert len(raw) == receipt["arena_bytes"] == 3 * stage.layout.npvals * 6 * 4
    back = np.frombuffer(raw, dtype="<f4").reshape(3, stage.layout.npvals, 6)
    assert np.allclose(np.diff(back[:, stage.layout.index("x"), 0]), 3.0)
    table = (root / PARCEL_TABLE_FILENAME).read_text().splitlines()
    assert table[0] == "index\tname\tunits\tfilled"
    assert len(table) == stage.layout.npvals + 1
    assert any(row.endswith("\tno") for row in table[1:])
    published = json.loads((root / PARCEL_RECEIPT_FILENAME).read_text())
    assert published["stage"]["changes_the_simulated_storm"] is False
    assert published["model_times_s"] == [1.0, 2.0, 3.0]


# --------------------------------------------------- hifrq and pdcomp


def test_the_window_written_is_the_window_the_document_named(tmp_path):
    window = HighFrequencyWindow(2, 6, 3, 5, 0, 4)
    fields = {"w3d": np.arange(8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)}
    receipt = write_high_frequency_window(
        tmp_path / "hf", fields, window, order=["w3d"],
        nx=12, ny=10, nz=8, model_time_s=30.0, step=3,
    )
    assert receipt["window"]["shape_zyx"] == [4, 2, 4]
    raw = (tmp_path / "hf" / HIFRQ_ARENA_FILENAME).read_bytes()
    assert len(raw) == 4 * 2 * 4 * 4
    back = np.frombuffer(raw, dtype="<f4").reshape(4, 2, 4)
    assert np.array_equal(back, fields["w3d"][0:4, 3:5, 2:6])
    assert (tmp_path / "hf" / HIFRQ_TABLE_FILENAME).exists()


def test_the_default_window_is_hifrq_F_s_own_whole_domain():
    assert full_domain_window(nx=5, ny=6, nz=7).shape == (7, 6, 5)


def test_a_high_frequency_cadence_that_is_not_a_whole_step_count_is_refused():
    assert high_frequency_cadence(dt_s=2.0, duration_s=600.0, interval_s=10.0) == 5
    with pytest.raises(ValueError):
        high_frequency_cadence(dt_s=2.0, duration_s=600.0, interval_s=3.0)
    with pytest.raises(ValueError, match="no high-frequency frame"):
        high_frequency_cadence(dt_s=2.0, duration_s=10.0, interval_s=60.0)


def _pdcomp_document(**boundaries) -> dict:
    document = _base_document()
    document["lagrangian"] = {"output_pdcomp": 1}
    document.setdefault("boundaries", {}).update(boundaries)
    return document


def test_pdcomp_publishes_cm1_names_and_says_what_is_unvalidated():
    """writeout.F:6252-6340 and :8478-8580 name the eight fields."""

    assert PDCOMP_SCALAR_NAMES == ("pipb", "pipdl", "pipdn", "pipc")
    assert PDCOMP_W_NAMES == ("pgradb", "pgraddl", "pgraddn", "pgradc")
    assert "UNVALIDATED" in PDCOMP_UNVALIDATED
    assert UNVALIDATED_MARKERS["pdcomp"] == PDCOMP_UNVALIDATED
    runtime = build_runtime_config(parse_mapping(_pdcomp_document()))
    selection = pdcomp_selection_for(runtime)
    assert pdcomp_selection_defect(selection) is None
    # param.F:5188-5199: open laterals double both transform axes.
    assert (selection.imirror, selection.jmirror) == (1, 1)
    assert (selection.ipe, selection.jpe) == (2 * runtime.grid.nx, 2 * runtime.grid.ny)
    assert selection.names[:3] == PDCOMP_SCALAR_NAMES[:3]


def test_pdcomp_on_laterals_its_forcing_cannot_serve_is_refused_by_name():
    # A periodic side gives poiss.F its un-mirrored transform (param.F:5188-5198),
    # which this port has not compared with CM1; that is what is refused.
    with pytest.raises(CM1Refusal) as caught:
        build_runtime_config(parse_mapping(_pdcomp_document(west=1, east=1)))
    assert "CM1-CARTESIAN-PDCOMP" in str(caught.value)
    assert "un-mirrored" in str(caught.value)
    document = _pdcomp_document(west=1, east=1)
    document["lagrangian"]["output_pdcomp"] = 0
    build_runtime_config(parse_mapping(document))
    # A free-slip wall keeps the mirrored inversion, and the nonlinear forcing's
    # advu/advv/advw take the run's own family, so it is admitted.
    runtime = build_runtime_config(parse_mapping(_pdcomp_document(west=3, east=3)))
    assert pdcomp_selection_defect(pdcomp_selection_for(runtime)) is None


# -------------------------------------------------------------- the markers


def test_every_capability_without_a_cm1_comparison_is_marked_in_the_stage():
    stage = _stage({
        "iprcl": 1, "nparcels": 64, "direction": "backward",
        "seeding": "vorticity-threshold", "vorticity_budget": True,
        "circuits": 1, "circuit_parcels": 16, "circuit_radius_m": 1000.0,
        "circuit_center_z_m": 1000.0,
        "output_pdcomp": 1, "dohifrq": True, "hifrqfrq_s": 10.0,
    })
    marked = stage.unvalidated()
    assert set(marked) == set(UNVALIDATED_MARKERS)
    for value in marked.values():
        assert "UNVALIDATED" in value
    assert stage.record()["unvalidated"] == marked
    assert stage.record()["changes_the_simulated_storm"] is False


def test_a_plain_forward_parcel_run_carries_no_unvalidated_marker():
    stage = _stage({"iprcl": 1, "nparcels": 8, "seeding": "box",
                    "seed_nx": 2, "seed_ny": 2, "seed_nz": 2})
    assert stage.unvalidated() == {}


def test_doctor_lists_the_table_and_every_unvalidated_capability():
    findings = doctor.collect()
    subjects = {finding.subject for finding in findings}
    assert "GM1 inline Lagrangian and budget analysis" in subjects
    assert "GM1 parcel trajectories vs CM1" in subjects
    # doctor lists other lanes' unvalidated capabilities too; these are this table's.
    unvalidated = [f for f in doctor.unvalidated(findings)
                   if f.subject.startswith("GM1 lagrangian: ")]
    assert {f.subject.split(": ", 1)[1] for f in unvalidated} == set(UNVALIDATED_MARKERS)
    rendered = doctor.render(findings, explain=True)
    assert "UNVALID" in rendered
    assert "unvalidated exposed capabilities: " in rendered
    # None of them blocks: they are diagnostics, not a red scientific gate.
    assert not any(f.required for f in unvalidated)


def test_the_shipped_example_exercises_every_capability_the_table_can_select():
    config = load_config(SHIPPED)
    selectors = config.lagrangian
    assert selectors.parcels_enabled
    assert selectors.circuits_enabled
    assert selectors.pdcomp_enabled
    assert selectors.hifrq_enabled
    assert selectors.budget_enabled
    assert selectors.values["seeding"] == "box"
    runtime = build_runtime_config(config)
    stage = LagrangianStage.build(
        runtime.lagrangian, nx=runtime.grid.nx, ny=runtime.grid.ny,
        nz=runtime.grid.nz, dx_m=runtime.grid.dx_m, dy_m=runtime.grid.dy_m,
        dz_m=runtime.grid.dz_m, dt_s=runtime.dt_s, xp=np, iorigin=runtime.iorigin,
        bbc=runtime.bbc, switch_context=parcel_switch_context(runtime),
    )
    assert len(stage.circuits) == 4
    assert stage.window.shape == (12, 32, 32)
    record = stage.seed()
    assert record["placed_parcels"] == 15 * 15 * 8
    # Every ring and the box lie inside the grid in the frame the runtime uses.
    placed = stage.positions[:, stage.positions[0] > -1.0e29]
    assert float(placed[0].min()) >= stage.grid.minx_m and float(placed[0].max()) <= stage.grid.maxx_m
    assert float(placed[1].min()) >= stage.grid.miny_m and float(placed[1].max()) <= stage.grid.maxy_m
    # The circuits overwrite the first 256 box parcels: one integrator carries
    # both families, and circulation_circuits.CircuitDefinition names which is
    # which, so the count is the box seeding and not the sum.
    assert stage.positions.shape[1] == 4096


# --------------------------------- the advecting velocity (solve2.F:1290)


def test_the_stage_snapshots_the_velocity_and_fills_cm1s_open_lateral_ring():
    """``solve2.F:1290-1301`` and ``bc.F:550-562``, on the real carriers.

    The snapshot exists because ``rru``/``rrv``/``rrw`` do not survive the
    scalar stage: ``cuda_advection5.py:330`` refills them with the mass flux
    ``rho0*u`` for CM1's ``src/adv.F`` contract.  This checks that what the
    parcels are handed is a *copy*, taken before that, and that the one
    lateral ring the trilinear stencil reaches is filled zero-gradient.
    """

    stage = _stage({"iprcl": 1, "nparcels": 4, "seeding": "box",
                    "seed_nx": 2, "seed_ny": 2, "seed_nz": 1})
    nx = ny = 24
    nz = 20
    shapes = terrain_open_sound_shapes(nx, ny, nz)
    ng = 3
    arrays = {
        "rru": np.zeros(shapes["u3d"], dtype=np.float32),
        "rrv": np.zeros(shapes["v3d"], dtype=np.float32),
        "rrw": np.zeros(shapes["w3d"], dtype=np.float32),
    }
    arrays["rru"][:, ng:ng + ny, :] = np.float32(5.0)
    arrays["rrv"][:, :, ng:ng + nx] = np.float32(-2.0)
    arrays["rrw"][:, ng:ng + ny, ng:ng + nx] = np.float32(0.5)
    field = stage.capture_advecting_velocity(arrays)
    assert field.u is not arrays["rru"]
    assert float(field.u[5, ng - 1, ng + 2]) == pytest.approx(5.0)
    assert float(field.u[5, ng + ny, ng + 2]) == pytest.approx(5.0)
    assert float(field.v[5, ng + 2, ng - 1]) == pytest.approx(-2.0)
    assert float(field.v[5, ng + 2, ng + nx]) == pytest.approx(-2.0)
    assert float(field.w[5, ng - 1, ng - 1]) == pytest.approx(0.5)
    assert float(field.w[5, ng + ny, ng + nx]) == pytest.approx(0.5)
    # A second capture reuses the same buffers rather than reallocating them.
    assert stage.capture_advecting_velocity(arrays) is field
    assert stage.advecting_velocity is field
    # ...and it tracks the carriers, so a later refill of rru does not reach
    # the parcels through this copy.
    arrays["rru"][...] = np.float32(999.0)
    assert float(field.u[5, ng + 2, ng + 2]) == pytest.approx(5.0)


def test_a_window_receipt_carries_its_marker_on_disk_not_only_in_memory(tmp_path):
    """A marker that exists only in the returned dict is a marker nobody sees."""

    window = HighFrequencyWindow(0, 2, 0, 2, 0, 2)
    fields = {"w3d": np.zeros((2, 2, 2), dtype=np.float32)}
    returned = write_high_frequency_window(
        tmp_path / "hf", fields, window, order=["w3d"], nx=2, ny=2, nz=2,
        model_time_s=1.0, step=1,
        extra={"kind": "vorticity-budget", "unvalidated": "UNVALIDATED: pinned"},
    )
    written = json.loads((tmp_path / "hf" / "hifrq-window.json").read_text())
    assert written["kind"] == "vorticity-budget"
    assert written["unvalidated"] == "UNVALIDATED: pinned"
    assert written == returned


def test_the_budget_frame_writer_names_every_term_and_the_unfilled_one(tmp_path):
    from cm1core.lagrangian_stage import BUDGET_UNFILLED_TERMS
    from cm1core.vorticity_budget import (
        VORTICITY_BUDGET_FIELD_NAMES, VORTICITY_COMPONENT_NAMES,
    )

    stage = _stage({"vorticity_budget": True}, nx=8, ny=8, nz=8)
    fields = {
        name: np.zeros((8, 8, 8), dtype=np.float32)
        for name in (*VORTICITY_COMPONENT_NAMES, *VORTICITY_BUDGET_FIELD_NAMES)
    }
    receipt = stage.write_budget_frame(tmp_path / "b", fields, step=4, model_time_s=8.0)
    assert receipt["kind"] == "vorticity-budget"
    assert receipt["fields"] == [*VORTICITY_COMPONENT_NAMES, *VORTICITY_BUDGET_FIELD_NAMES]
    assert receipt["unfilled_terms"] == dict(BUDGET_UNFILLED_TERMS)
    assert "friction" in receipt["unfilled_terms"]
    written = json.loads((tmp_path / "b" / "hifrq-window.json").read_text())
    assert written["unvalidated"] == UNVALIDATED_MARKERS["vorticity_budget"]


def receipt_grid_iorigin(root: Path) -> int:
    receipt = json.loads((root / "run-receipt.lagrangian.json").read_text())
    return int(receipt["lagrangian"]["grid"]["iorigin"])


def test_the_shipped_full_selector_evidence_matches_this_tree():
    """The recorded RTX 5070 Ti run's document is still admitted here."""

    root = Path(__file__).resolve().parents[1] / "evidence/lagrangian/full-selector-run-32"
    # The recorded document spells iorigin in [lagrangian], the table it lived
    # in until d0f0429 moved CM1's one iorigin to [initialization].  It is kept
    # byte for byte; the migrated document differs by exactly that one move.
    recorded = tomllib.loads((root / "gpuwm-cm1.toml").read_text(encoding="utf-8"))
    migrated = tomllib.loads((root / "gpuwm-cm1.migrated.toml").read_text(encoding="utf-8"))
    moved = dict(recorded)
    moved["lagrangian"] = {k: v for k, v in recorded["lagrangian"].items() if k != "iorigin"}
    moved["initialization"] = {**recorded["initialization"], "iorigin": recorded["lagrangian"]["iorigin"]}
    assert migrated == moved
    with pytest.raises(CM1Refusal):
        load_config(root / "gpuwm-cm1.toml")
    config = load_config(root / "gpuwm-cm1.migrated.toml")
    assert config.initialization.iorigin == receipt_grid_iorigin(root)
    selectors = config.lagrangian
    assert selectors.parcels_enabled and selectors.budget_enabled
    assert selectors.circuits_enabled and selectors.pdcomp_enabled
    assert selectors.hifrq_enabled
    receipt = json.loads((root / "run-receipt.lagrangian.json").read_text())
    stage = receipt["lagrangian"]
    assert stage["changes_the_simulated_storm"] is False
    assert sorted(stage["unvalidated"]) == [
        "circulation", "high_frequency_window", "pdcomp", "vorticity_budget",
    ]
    assert stage["eulerian_frame_kinds"] == [
        "high-frequency-window", "pdcomp-sources", "vorticity-budget",
    ]
    assert stage["parcel_frames_recorded"] == 6
    assert stage["circulation_rows_recorded"] == 6
    rows = json.loads((root / "circulation-circuits.json").read_text())
    assert [c["name"] for c in rows[0]["circuits"]] == ["circuit00", "circuit01"]
    # The circuits are barely deformed over 60 s, so their radius is still the
    # seeded 3000 m; the circulation they enclose is not zero.
    for row in rows:
        for circuit in row["circuits"]:
            assert abs(circuit["mean_radius_m"] - 3000.0) < 10.0
    assert abs(rows[-1]["circuits"][0]["circulation_m2_s"]) > 0.1
