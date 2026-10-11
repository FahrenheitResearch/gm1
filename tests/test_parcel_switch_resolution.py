"""The parcel switches CM1 r21.1 resolves, and the columns it lays out from them.

``src/param.F:6239-6282`` overrides eight ``&param13`` switches from the rest of
the namelist before ``:6385-6500`` lays out a single parcel column: ``iptra=0``
turns ``prcl_ptra`` off, ``imoist=0`` the four moisture switches, ``ptype=1,4``
``prcl_dbz``, ``idm=0`` ``prcl_nc``, ``sgsmodel<=0 .or. ipbl<=0`` ``prcl_km`` and
``prcl_kh``, ``.not. iusetke`` ``prcl_tke``, and ``bbc/=3`` ``prcl_sfc``.  GM1
recorded and laid out the switches as requested, so a CM1 LES deck published
``kmh``..``khv``, ``znt`` and ``ust`` columns CM1 never writes, and a WSM6 record
said ``prcl_nc = 1``.

The authority here is CM1's own output:
``evidence/oracle/parcel-switch-resolution/cm1-parcel-switch-resolution.json``
is what the built CM1 r21.1 binary printed (the resolved ``prcl_*``, ``npvals``,
the ``pr*`` offsets) and wrote (``cm1out_parcel.ctl``) for seventeen namelists,
one per branch, each requesting all sixteen switches
(``tools/oracle/parcel_switch_resolution_oracle.py``, reproducible byte for
byte).  No CUDA runs here.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Only names the unresolved tree already had are imported here, so the runtime tests
# below collect -- and fail on CM1's columns -- against the unresolved tree too.
from cm1core.lagrangian_parcels import PARCEL_SWITCH_ORDER, parcel_attribute_layout, parcel_attribute_units
from cm1core.moisture_species import MORRISON_PROFILE, THOMPSON_PROFILE, WSM6_PROFILE

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "evidence/oracle/parcel-switch-resolution/cm1-parcel-switch-resolution.json"
EXAMPLES = ROOT / "examples"

#: param.F:3169-3183: Kessler's axis (iice and idm keep :3100-3101's 0).
KESSLER_AXIS = SimpleNamespace(
    cm1_ptype=1, names=("qv", "qc", "qr"), nqv=1, nql1=2, nql2=3, nqs1=-2, nqs2=-2,
    nnc1=-2, nnc2=-2, iice=0, idm=0, component_id="cm1-ptype-1",
)
#: param.F:3222-3247: the LFO ice axis (ptype=2,4), single moment.
LFO_AXIS = SimpleNamespace(
    cm1_ptype=4, names=("qv", "qc", "qr", "qi", "qs", "qg"), nqv=1, nql1=2, nql2=3, nqs1=4,
    nqs2=6, nnc1=-2, nnc2=-2, iice=1, idm=0, component_id="cm1-ptype-4",
)
AXES = {5: MORRISON_PROFILE, 7: WSM6_PROFILE, 3: THOMPSON_PROFILE, 1: KESSLER_AXIS, 4: LFO_AXIS}

#: CM1's pr* offset -> the column it names (param.F:6385-6500).
OFFSET_COLUMNS = {
    "prth": "th", "prt": "t", "prprs": "prs", "prqv": "qv", "prkm": "kmh", "prkh": "khh",
    "prtke": "tke", "prdbz": "dbz", "prb": "b", "prvpg": "vpg", "przv": "zvort", "prrho": "rho",
    "prqsl": "qsl", "prqsi": "qsi", "prznt": "znt", "prust": "ust",
}

CARRIED = (
    "morrison-les-free-slip", "wsm6-single-moment", "thompson-two-number-rows",
    "kessler-warm-rain", "lfo-ice", "dry", "semi-slip-surface-layer", "smagorinsky",
    "two-part-s94", "no-subgrid-closure", "mesoscale-ysu", "mesoscale-mynn",
    "les-inside-mesoscale", "mynn-surface-without-closure", "bbc-forced-by-surface-layer",
)


def _record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _context(case: dict):
    """GM1's context from the namelist words alone; CM1's printed words are the check."""

    from cm1core.lagrangian_parcels import cm1_parcel_switch_context

    words = case["namelist_words"]
    imoist = int(words["imoist"])
    return cm1_parcel_switch_context(
        imoist=imoist, profile=AXES[int(words["ptype"])] if imoist == 1 else None,
        cm1setup=words["cm1setup"], sgsmodel=words["sgsmodel"], ipbl=words["ipbl"],
        sfcmodel=words["sfcmodel"], oceanmodel=words["oceanmodel"], bbc=words["bbc"],
        iptra=words["iptra"], terrain=bool(words["terrain_flag"]),
    )


#: The recorded file's digest.  Regenerate it with the tool (it reproduces byte
#: for byte) and re-pin; a hand edit would change the authority these tests use.
RECORD_SHA256 = "91fe535e04afa3a8f6d625573c5e86aada57923226228bba041a87b50cd9bf6f"


def test_the_oracle_record_covers_every_override_rule_and_layout_gate():
    import hashlib

    assert hashlib.sha256(RECORD.read_bytes()).hexdigest() == RECORD_SHA256
    record = _record()
    assert record["schema"] == "gpuwm-cm1.cm1-parcel-switch-resolution/v1"
    assert record["cm1_release"] == "r21.1"
    cases = record["cases"]
    assert set(cases) == set(CARRIED) | {"passive-tracer", "terrain"}
    turned_off = {
        name for case in cases.values() for name, value in case["resolved_switches"].items()
        if value == 0
    }
    # Every switch param.F:6257-6282 can override was overridden in at least one run...
    assert turned_off == {"prcl_ptra", "prcl_q", "prcl_nc", "prcl_km", "prcl_kh", "prcl_tke",
                          "prcl_dbz", "prcl_qsat", "prcl_sfc"}
    # ...and kept in another, so no rule is recorded only one way.
    for name in turned_off:
        assert any(case["resolved_switches"][name] == 1 for case in cases.values()), name
    for case in cases.values():
        assert case["requested_switches"] == {name: 1 for name in PARCEL_SWITCH_ORDER}
        assert len(case["columns"]) == case["offsets"]["npvals"]


@pytest.mark.parametrize("name", CARRIED)
def test_gm1_resolves_and_lays_out_every_branch_exactly_as_cm1_wrote_it(name):
    case = _record()["cases"][name]
    context = _context(case)
    printed = case["cm1_printed_words"]
    # The derived words are the ones CM1 held at param.F:6257: bbc after :873-889,
    # sgsmodel after :1074-1085, iusetke after :4989-4997/:5107-5108, and the
    # axis's iice/idm.
    assert context.sgsmodel == printed["sgsmodel"]
    assert context.iusetke is printed["iusetke"]
    assert (context.iice, context.idm) == (printed["iice"], printed["idm"])
    assert (context.cm1setup, context.ipbl, context.bbc) == (
        printed["cm1setup"], printed["ipbl"], printed["bbc"])

    layout = parcel_attribute_layout(case["requested_switches"], context)
    assert layout.switches == case["resolved_switches"]
    assert list(layout.names) == [column["name"] for column in case["columns"]]
    assert [parcel_attribute_units(n) for n in layout.names] == [
        column["units"] for column in case["columns"]]
    assert layout.npvals == case["offsets"]["npvals"]
    for offset, column in OFFSET_COLUMNS.items():
        at = case["offsets"][offset]
        if at == 0:
            assert column not in layout.names, (offset, column)
        else:
            assert layout.index(column) + 1 == at, (offset, column)
    if case["offsets"]["prnc1"]:
        rows = [n for n, _ in layout.number_rows]
        assert layout.index(rows[0]) + 1 == case["offsets"]["prnc1"]
        assert layout.index(rows[-1]) + 1 == case["offsets"]["prnc2"]
    else:
        assert layout.number_rows == ()
    # The record says what was asked, what CM1 holds, and which rule changed it.
    record = layout.record()
    assert record["schema"] == "gpuwm-cm1.lagrangian-parcels/v3"
    assert record["switches"] == case["resolved_switches"]
    assert record["requested_switches"] == case["requested_switches"]
    assert set(record["switch_overrides"]) == {
        n for n, v in case["resolved_switches"].items() if v == 0}
    assert all(rule.startswith("src/param.F:62") for rule in record["switch_overrides"].values())


def test_the_bbc_the_resolution_reads_is_the_one_param_f_873_889_leaves():
    """sfcmodel>=1, oceanmodel=2 or ipbl>=1 force bbc=3 before :6280-6282 reads it.

    CM1's own run covers the sfcmodel trigger (``bbc-forced-by-surface-layer``:
    namelist bbc=1, printed bbc=3, prcl_sfc kept).  The other two cannot be run
    alone -- with sfcmodel=0 the forced bbc=3 stops CM1 at :2073-2088 -- so they
    are held to the source's condition here.
    """

    from cm1core.lagrangian_parcels import cm1_parcel_switch_context

    case = _record()["cases"]["bbc-forced-by-surface-layer"]
    assert case["namelist_words"]["bbc"] == 1 and case["cm1_printed_words"]["bbc"] == 3
    assert case["resolved_switches"]["prcl_sfc"] == 1
    assert _context(case).bbc == 3

    def bbc(**words):
        base = dict(imoist=1, profile=WSM6_PROFILE, cm1setup=2, sgsmodel=1, ipbl=0,
                    sfcmodel=0, oceanmodel=1, bbc=1, iptra=0, terrain=False)
        return cm1_parcel_switch_context(**{**base, **words}).bbc

    assert bbc() == 1
    assert bbc(sfcmodel=1) == 3
    assert bbc(oceanmodel=2) == 3
    assert bbc(ipbl=1) == 3
    assert bbc(oceanmodel=0) == 1 and bbc(bbc=2) == 2


def test_a_switch_cm1_keeps_on_can_still_lay_out_no_column():
    """param.F:6443: sfcmodel=6 sets iusetke without an LES or PBL closure."""

    case = _record()["cases"]["mynn-surface-without-closure"]
    layout = parcel_attribute_layout(case["requested_switches"], _context(case))
    assert layout.switches["prcl_tke"] == 1 == case["resolved_switches"]["prcl_tke"]
    assert "tke" not in layout.names
    assert set(layout.record()["switches_without_columns"]) == {"prcl_tke"}


@pytest.mark.parametrize("name,columns", [("passive-tracer", {"pt1"}), ("terrain", {"zs", "sigma"})])
def test_the_two_cm1_branches_this_port_does_not_carry_are_refused(name, columns):
    case = _record()["cases"][name]
    assert columns <= {column["name"] for column in case["columns"]}
    with pytest.raises(ValueError, match="iptra=1" if name == "passive-tracer" else "terrain_flag"):
        _context(case)


def test_a_layout_without_the_run_context_is_refused():
    with pytest.raises(TypeError, match="ParcelSwitchContext"):
        parcel_attribute_layout({name: 1 for name in PARCEL_SWITCH_ORDER}, ())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The runtime paths: the stage a document builds, and the nest child's tracker.
# --------------------------------------------------------------------------

_BOX = {"iprcl": 1, "nparcels": 4, "seeding": "box", "seed_nx": 2, "seed_ny": 2, "seed_nz": 1,
        "seed_x0_m": 20000.0, "seed_y0_m": 20000.0, "seed_z0_m": 1000.0}
_CHILD = {"seed_frame": "parent-grid-origin", "nparcels": 1, "prclfrq_s": 10.0,
          "seed_nx": 1, "seed_ny": 1, "seed_nz": 1, "seed_dx_m": 100.0, "seed_dy_m": 100.0,
          "seed_dz_m": 100.0, "seed_x0_m": 1000.0, "seed_y0_m": 1000.0, "seed_z0_m": 100.0}

#: A GM1 document for each CM1 case the Cartesian lane admits: the component
#: CM1's ptype names, and CM1's bbc/sfcmodel.  Every prcl_* is left to its
#: default, which is CM1's shipped 1.
RUNTIME_CASES = {
    "morrison-les-free-slip": ("morrison-mp10", None),
    "wsm6-single-moment": ("wsm6-mp6", None),
    "thompson-two-number-rows": ("thompson-mp8", None),
    "semi-slip-surface-layer": ("morrison-mp10", {"bbc": 3, "sfcmodel": 1}),
}


def _runtime_config(component: str, surface: dict | None):
    from cm1core.cartesian_door import build_runtime_config
    from cm1core.config import parse_mapping

    document = tomllib.loads((EXAMPLES / "cartesian-uniform-grid.toml").read_text(encoding="utf-8"))
    document["physics"]["component"] = component
    if surface is not None:
        document["boundaries"]["bottom"] = surface["bbc"]
        document["closures"].update({**surface, "cecd": 1, "cnstcd": 0.001, "isfcflx": 0})
    document["lagrangian"] = {**_BOX, "nest_child": dict(_CHILD)}
    return build_runtime_config(parse_mapping(document))


@pytest.mark.parametrize("name", sorted(RUNTIME_CASES))
def test_a_runtime_stage_publishes_what_cm1_writes_for_the_same_namelist(name, tmp_path):
    """From the document through the runtime's stage builder to the track's table and receipt."""

    from cm1core.cuda_cartesian_runtime import CudaCartesianRuntime
    from cm1core.lagrangian_stage import PARCEL_TABLE_FILENAME, write_parcel_track

    case = _record()["cases"][name]
    config = _runtime_config(*RUNTIME_CASES[name])
    assert (config.cm1setup, config.sgsmodel, config.ipbl) == (
        case["namelist_words"]["cm1setup"], case["namelist_words"]["sgsmodel"],
        case["namelist_words"]["ipbl"])
    assert (config.bbc, config.sfcmodel) == (
        case["namelist_words"]["bbc"], case["namelist_words"]["sfcmodel"])
    stage = CudaCartesianRuntime._build_lagrangian_stage(SimpleNamespace(config=config, cp=np))
    columns = [column["name"] for column in case["columns"]]
    assert list(stage.layout.names) == columns
    configuration = stage.record()["configuration"]
    # The sealed selectors stay the document's request...
    assert all(configuration["selectors"][n] == 1 for n in PARCEL_SWITCH_ORDER)
    # ...and both published layouts are CM1's, main grid and nest child alike.
    for key in ("parcel_attributes", "nest_child_attributes"):
        assert configuration[key]["switches"] == case["resolved_switches"], key
        assert configuration[key]["names"] == columns, key
    write_parcel_track(tmp_path / "track", stage, [])
    table = (tmp_path / "track" / PARCEL_TABLE_FILENAME).read_text().splitlines()[1:]
    assert [row.split("\t")[1] for row in table] == columns
    assert [row.split("\t")[2] for row in table] == [c["units"] for c in case["columns"]]


@pytest.mark.parametrize("name", ["wsm6-single-moment", "semi-slip-surface-layer"])
def test_the_nest_child_tracker_carries_cm1s_layout_on_both_grids(name):
    from cm1core.lagrangian_parcels import parcel_switch_context
    from cm1core.nest_live import child_runtime_config, nest_child_parcel_layout

    case = _record()["cases"][name]
    parent = _runtime_config(*RUNTIME_CASES[name])
    child = child_runtime_config(parent, parent.grid, child_dt_s=parent.dt_s,
                                 acoustic_substeps=parent.acoustic_substeps)
    layout = nest_child_parcel_layout(parent, child)
    assert list(layout.names) == [column["name"] for column in case["columns"]]
    assert layout.switches == case["resolved_switches"]
    assert layout.context == parcel_switch_context(parent)
    # A pair whose child resolves differently cannot share one track.
    other = dataclasses.replace(child, bbc=2 if parent.bbc == 1 else 1, sfcmodel=0)
    with pytest.raises(ValueError, match="resolve the prcl_\\* switches differently"):
        nest_child_parcel_layout(parent, other)
    # No [lagrangian.nest_child] table, no child layout.
    assert nest_child_parcel_layout(dataclasses.replace(parent, lagrangian=None), child) is None


def test_the_nest_child_tracker_refuses_a_layout_resolved_for_another_grid():
    """The tracker holds its layout to both grids' words, as the parent stage does."""

    from cm1core.lagrangian_parcels import parcel_switch_context
    from cm1core.nest_live import child_runtime_config, nest_child_parcel_layout
    from cm1core.nest_parcels import NestParcelTracker

    parent = _runtime_config(*RUNTIME_CASES["wsm6-single-moment"])
    child = child_runtime_config(parent, parent.grid, child_dt_s=parent.dt_s,
                                 acoustic_substeps=parent.acoustic_substeps)
    layout = nest_child_parcel_layout(parent, child)
    other = dataclasses.replace(child, bbc=2)
    assert parcel_switch_context(other).bbc == 2
    build = dict(settings=parent.lagrangian.nest_child, geometry=None, xp=np)
    with pytest.raises(ValueError, match="nest child integrates bbc=2.*prcl_sfc"):
        NestParcelTracker.build(layout=layout, parent_config=parent, child_config=other, **build)
    with pytest.raises(ValueError, match="nest parent integrates bbc=1.*prcl_sfc"):
        NestParcelTracker.build(layout=parent.lagrangian.attribute_layout(
            parcel_switch_context(other)), parent_config=parent, child_config=other, **build)


def _driver(monkeypatch, parent_config, child_config):
    """A NestDriver up to its parcel block, on CPU: forcing and clock stubbed out."""

    from cm1core import nest_live

    monkeypatch.setattr(nest_live, "NestLateralForcing",
                        lambda **_: SimpleNamespace(module=None))
    monkeypatch.setattr(nest_live.NestDriver, "_align_child_clock", lambda self: None)
    numq = parent_config.profile.numq

    def runtime(config):
        return SimpleNamespace(config=config, dynamics=SimpleNamespace(numq=numq),
                               clock=SimpleNamespace(model_seconds=0.0, completed_steps=0))

    return lambda: nest_live.NestDriver(
        parent=runtime(parent_config), child=runtime(child_config),
        geometry=SimpleNamespace(time_refinement_ratio=1), cupy=None, stream=None)


def test_the_live_nest_refuses_a_switch_resolution_mismatch_as_itself(monkeypatch):
    """Not as CM1-CARTESIAN-NEST-PARCELS-SEED-EXTENT, whose remedy is to move the box."""

    from cm1core.nest_live import child_runtime_config

    parent = _runtime_config(*RUNTIME_CASES["wsm6-single-moment"])
    child = child_runtime_config(parent, parent.grid, child_dt_s=parent.dt_s,
                                 acoustic_substeps=parent.acoustic_substeps)
    with pytest.raises(ValueError, match="resolve the prcl_\\* switches differently"):
        _driver(monkeypatch, parent, dataclasses.replace(child, bbc=2))()


@pytest.mark.parametrize("seed_extent", [True, False])
def test_only_a_seed_extent_defect_is_refused_as_one(monkeypatch, seed_extent):
    from cm1core.errors import CM1Refusal
    from cm1core.nest_live import child_runtime_config
    from cm1core.nest_parcels import NestParcelSeedExtentError, NestParcelTracker

    parent = _runtime_config(*RUNTIME_CASES["wsm6-single-moment"])
    child = child_runtime_config(parent, parent.grid, child_dt_s=parent.dt_s,
                                 acoustic_substeps=parent.acoustic_substeps)
    error = NestParcelSeedExtentError("box outside") if seed_extent else ValueError("other defect")

    def build(**_):
        raise error

    monkeypatch.setattr(NestParcelTracker, "build", build)
    construct = _driver(monkeypatch, parent, child)
    if seed_extent:
        with pytest.raises(CM1Refusal) as excinfo:
            construct()
        assert excinfo.value.detail.code == "CM1-CARTESIAN-NEST-PARCELS-SEED-EXTENT"
    else:
        with pytest.raises(ValueError, match="other defect"):
            construct()


def test_the_stage_refuses_a_context_resolved_for_another_bbc():
    from cm1core.lagrangian_config import parse_lagrangian_table
    from cm1core.lagrangian_parcels import cm1_parcel_switch_context
    from cm1core.lagrangian_stage import LagrangianStage

    context = cm1_parcel_switch_context(imoist=1, profile=WSM6_PROFILE, cm1setup=1, sgsmodel=1,
                                        ipbl=0, sfcmodel=0, oceanmodel=1, bbc=1, iptra=0,
                                        terrain=False)
    with pytest.raises(ValueError, match="prcl_sfc"):
        LagrangianStage.build(parse_lagrangian_table(dict(_BOX)), nx=24, ny=24, nz=20,
                              dx_m=1000.0, dy_m=1000.0, dz_m=500.0, dt_s=1.0, xp=np, iorigin=1,
                              bbc=3, switch_context=context)
