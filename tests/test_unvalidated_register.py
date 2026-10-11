"""The register of exposed-but-unmeasured capabilities, and its two readers."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from cm1core import doctor
from cm1core.config import parse_mapping
from cm1core.unvalidated import (
    UNVALIDATED,
    UNVALIDATED_CAPABILITIES,
    UnvalidatedCapability,
    receipt,
    selected,
    selected_by_document,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_row_names_a_key_a_document_can_actually_write() -> None:
    """A row nobody can select is noise; a key nobody marks is silence."""

    from cm1core.config import _SCHEMA  # the grammar a document is held to

    assert UNVALIDATED_CAPABILITIES, "the register must not be silently empty"
    for row in UNVALIDATED_CAPABILITIES:
        table, _, key = row.option.partition(".")
        assert table in _SCHEMA, row.option
        assert key in _SCHEMA[table], row.option
        # A marker that does not say what was not checked is not a marker.
        assert len(row.not_checked) > 40 and len(row.would_show) > 20
        assert ":" in row.authority and row.authority.startswith("src/")


def test_a_row_covers_only_the_values_it_names() -> None:
    row = UnvalidatedCapability(
        option="grid.stretch_z", values=(3, 4),
        not_checked="x" * 41, would_show="y" * 21, authority="src/param.F:1",
    )
    assert row.covers(3) and row.covers(4)
    assert not row.covers(0) and not row.covers(1)
    # True == 1 in Python; a boolean must not be read as an integer value.
    assert not row.covers(True)
    every = UnvalidatedCapability(
        option="grid.stretch_z", values=(),
        not_checked="x" * 41, would_show="y" * 21, authority="src/param.F:1",
    )
    assert every.covers(0) and every.covers(4)


def test_only_the_rows_a_document_selects_reach_its_receipt() -> None:
    assert selected({}) == ()
    assert selected({"grid.stretch_z": 0}) == ()
    for branch in (1, 3, 4):
        hit = selected({"grid.stretch_z": branch})
        assert len(hit) == 1 and hit[0].option == "grid.stretch_z", branch

    # An unstretched deck selects nothing.
    uniform = tomllib.loads((ROOT / "examples" / "cartesian-uniform-grid.toml").read_text())
    assert selected_by_document(parse_mapping(uniform)) == ()

    # Every stretched one does, whichever branch built its column.
    document = tomllib.loads(
        (ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml").read_text()
    )
    rows = selected_by_document(parse_mapping(document))
    assert [row.option for row in rows] == ["grid.stretch_z"]

    # The same column re-declared as param.F:8072-8132's explicit w level
    # list is still stretched, and still selects the row.
    from cm1core.cartesian_door import require_grid_extent

    faces = list(require_grid_extent(parse_mapping(document)).face_heights_m)
    grid = document["grid"]
    grid["stretch_z"] = 4
    grid["levels_m"] = faces
    for name in ("str_bot_m", "str_top_m", "dz_bot_m", "dz_top_m", "ztop_m"):
        grid[name] = 0.0
    rows = selected_by_document(parse_mapping(document))
    assert [row.option for row in rows] == ["grid.stretch_z"]


def test_the_receipt_block_is_present_even_when_it_is_empty() -> None:
    empty = receipt(())
    assert empty["count"] == 0 and empty["capabilities"] == []
    assert empty["marker"] == UNVALIDATED
    # It must survive the receipt's own serialization rules.
    json.dumps(receipt(), allow_nan=False, sort_keys=True)
    for row in receipt()["capabilities"]:
        assert row["marker"] == UNVALIDATED
        assert set(row) == {
            "marker", "option", "values", "not_checked", "would_show", "authority",
        }


def test_doctor_reports_every_row_and_none_of_them_block() -> None:
    findings = doctor.collect()
    # doctor has other UNVALIDATED producers since eeea32b (dynamics), 4c7e294
    # (surface layer), 59b359b (&param9 output) and 022193c (Lagrangian); this
    # register's rows are the findings doctor._unvalidated_findings names.
    rows = [
        f
        for f in findings
        if f.status == doctor.UNVALIDATED and f.subject.startswith("unvalidated capability ")
    ]
    assert len(rows) == len(UNVALIDATED_CAPABILITIES)
    assert [f.subject for f in rows] == [
        f"unvalidated capability {row.option}" for row in UNVALIDATED_CAPABILITIES
    ]
    for finding in rows:
        assert UNVALIDATED in finding.detail
        assert finding not in doctor.blocking(findings)
        assert finding.remedy and "differentially against CM1" in finding.remedy
    rendered = doctor.render(findings, explain=True)
    assert "UNVALID" in rendered


def test_a_row_that_names_no_real_key_is_caught() -> None:
    bogus = UnvalidatedCapability(
        option="grid.no_such_key", values=(),
        not_checked="x" * 41, would_show="y" * 21, authority="src/param.F:1",
    )
    with pytest.raises(TypeError):
        selected(["grid.no_such_key"])  # type: ignore[arg-type]
    assert bogus.covers(7)


# ------------------------------------------------ the run-wide register


def _document(name: str, **tables):
    document = tomllib.loads((ROOT / "examples" / name).read_text())
    for table, values in tables.items():
        document.setdefault(table, {}).update(values)
    return parse_mapping(document)


def test_a_morrison_run_names_morrison_in_the_top_level_register() -> None:
    from cm1core.unvalidated import selected_for_run

    rows = selected_for_run(_document("cartesian-supercell-morrison-hail.toml"))
    block = receipt(rows)
    assert block["count"] >= 1
    component = [row for row in block["capabilities"] if row["option"] == "physics.component"]
    assert len(component) == 1 and component[0]["values"] == ["morrison-mp10"]
    assert component[0]["marker"] == UNVALIDATED
    assert "GM1-UNVALIDATED-CAPABILITY morrison-mp10" in component[0]["not_checked"]
    assert "a differential against native CM1 ptype=5" in component[0]["not_checked"]
    assert component[0]["authority"].startswith("src/morrison.F")
    json.dumps(block, allow_nan=False, sort_keys=True)


def test_a_measured_component_and_cm1s_own_kdiv_add_no_row() -> None:
    from cm1core.unvalidated import selected_for_run

    uniform = _document("cartesian-uniform-grid.toml")
    assert uniform.physics.component == "wsm6-mp6" and uniform.dynamics.kdiv == 0.10
    assert selected_for_run(uniform) == ()
    # Any other kdiv is the capability no run has measured, and says so by value.
    moved = selected_for_run(_document("cartesian-uniform-grid.toml", dynamics={"kdiv": 0.35}))
    assert [(row.option, row.values) for row in moved] == [("dynamics.kdiv", (0.35,))]
    assert "0.10" in moved[0].not_checked


def test_lspgrad_and_flagged_initialization_selectors_join_the_register() -> None:
    from cm1core.initialization_validation import unvalidated_initialization
    from cm1core.cartesian_door import require_initializer_selection
    from cm1core.unvalidated import selected_for_run

    config = _document(
        "cartesian-uniform-grid.toml",
        dynamics={"lspgrad": 1, "coriolis": 1, "beta_plane": 0},
    )
    options = [row.option for row in selected_for_run(config)]
    assert options == ["dynamics.lspgrad"]
    selection, _ = require_initializer_selection(config)
    flagged = {"schema": "x", "items": [{
        "selector": "irandp", "value": 1, "marker": "UNVALIDATED",
        "not_checked": "the random perturbation stream is GM1's own", "method": None,
        "evidence": "records/CM1-INITIALIZATION-VALIDATION.json",
    }]}
    rows = selected_for_run(config, initialization=flagged)
    assert [row.option for row in rows] == ["dynamics.lspgrad", "initialization.irandp"]
    # The run's own block for this document flags nothing.
    assert unvalidated_initialization(selection, config)["items"] == []
