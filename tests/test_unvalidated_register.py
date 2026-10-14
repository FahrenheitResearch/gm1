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

    # The register may be empty; today it holds the row for stretched levels
    # other than the column compared with CM1 in a storm
    # (test_unequal_end_layers_carry_a_row_that_does_not_stop_the_run and the
    # tests after it).
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


def test_only_the_rows_a_document_selects_reach_its_receipt(monkeypatch) -> None:
    from cm1core import unvalidated

    row = UnvalidatedCapability(
        option="grid.stretch_z", values=(1, 3, 4),
        not_checked="x" * 41, would_show="y" * 21, authority="src/param.F:1",
    )
    monkeypatch.setattr(unvalidated, "UNVALIDATED_CAPABILITIES", (row,))
    assert selected({}) == ()
    assert selected({"grid.stretch_z": 0}) == ()
    for branch in (1, 3, 4):
        hit = selected({"grid.stretch_z": branch})
        assert len(hit) == 1 and hit[0].option == "grid.stretch_z", branch

    # An unstretched deck selects nothing.
    uniform = tomllib.loads((ROOT / "examples" / "cartesian-uniform-grid.toml").read_text())
    assert selected_by_document(parse_mapping(uniform)) == ()

    # A stretched one selects a row that names its key, whichever branch built its column.
    document = tomllib.loads(
        (ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml").read_text()
    )
    rows = selected_by_document(parse_mapping(document))
    assert [row.option for row in rows] == ["grid.stretch_z"]

    from cm1core.cartesian_door import require_grid_extent

    faces = list(require_grid_extent(parse_mapping(document)).face_heights_m)
    grid = document["grid"]
    grid["stretch_z"] = 4
    grid["levels_m"] = faces
    for name in ("str_bot_m", "str_top_m", "dz_bot_m", "dz_top_m", "ztop_m"):
        grid[name] = 0.0
    rows = selected_by_document(parse_mapping(document))
    assert [row.option for row in rows] == ["grid.stretch_z"]


def test_the_stretched_levels_row_is_retired_by_its_measurement() -> None:
    """The compared column with Morrison carries no row.

    The row said no stretched run had reached condensation.  The comparison
    that retired it, for this one column with Morrison, is CM1's stock
    supercell on 53 stretched levels for two hours in both models; where the
    development evidence is present, its verdict and its rain record are read
    here.
    """

    from cm1core.unvalidated import COMPARED_STRETCHED_COLUMN

    config = parse_mapping(_stretched(**_grid_words(COMPARED_STRETCHED_COLUMN), component="morrison-mp10"))
    from cm1core.cartesian_door import require_grid_extent

    levels = require_grid_extent(config).levels
    assert levels.lowest_layers_equally_spaced and levels.highest_layers_equally_spaced
    assert selected_by_document(config) == ()
    record = ROOT / "evidence" / "oracle" / "stretched-supercell" / "comparison.json"
    if not record.is_file():
        pytest.skip("the development evidence is not distributed")
    comparison = json.loads(record.read_text())
    verdict = comparison["verdicts"]["gm1"]
    assert verdict["all_inside_band"] is True
    assert all(row["inside_band"] for row in verdict["rows"].values())
    rain = comparison["rain_at_the_ground"]
    assert (rain["gm1"]["first_frame_with_rain_at_the_ground_s"]
            == rain["cm1"]["first_frame_with_rain_at_the_ground_s"])
    final = rain["gm1"]["rain_total_ratio_to_reference"]["7200.0"]
    assert abs(final - 1.0) <= 0.15


def _stretched(component: str | None = None, **grid):
    """The stretched tornado-grade example (WSM6) with other ``grid`` words, and another scheme if given."""

    document = tomllib.loads(
        (ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml").read_text()
    )
    document["grid"].update(grid)
    if component is not None:
        document["physics"]["component"] = component
    return document


def _grid_words(record):
    """``grid.*`` words of a flat record as the [grid] table spells them."""

    return {key.split(".", 1)[1]: value for key, value in record.items() if key.startswith("grid.")}


def _flat(config) -> dict:
    return {f"{table}.{name}": value for table, fields in config.canonical_dict().items()
            if isinstance(fields, dict) for name, value in fields.items()}


#: CM1's hurricane column: 59 levels that stretch from the ground (50, 73.7, 97.4 m ...).
HURRICANE_COLUMN = dict(nz=59, ztop_m=25000.0, str_bot_m=0.0, str_top_m=5500.0,
                        dz_bot_m=50.0, dz_top_m=500.0)


def test_unequal_end_layers_carry_a_row_that_does_not_stop_the_run() -> None:
    """Levels that stretch from the ground or to the top run, with an honest row.

    The supercell on CM1's hurricane column (str_bot = 0) is inside the release
    bands with ground friction, and outside one band (max vertical vorticity,
    1.156) with a free-slip ground.  Those columns are admitted, so the row
    marks them without refusing; columns with three equal layers at each end
    keep no row.
    """

    from cm1core.cartesian_door import require_grid_extent
    from cm1core.unvalidated import (
        end_layers_unequal,
        selected_for_run,
        stretched_levels_not_compared,
        stretched_levels_reasons,
    )

    rows = [row for row in UNVALIDATED_CAPABILITIES if row.option == "grid.stretch_z"]
    assert len(rows) == 1
    row = rows[0]
    assert row.values == (1, 3, 4) and row.applies is stretched_levels_not_compared
    for fragment in ("str_bot = 0", "ground friction", "all ten", "free-slip ground",
                     "max vertical vorticity came out 15.6 % high", "11.9 % high",
                     "WSM6 and Thompson have not been run on stretched levels"):
        assert fragment in row.not_checked, fragment

    # From the ground: admitted, and selected, with Morrison and with the example's WSM6.
    ground = parse_mapping(_stretched(**HURRICANE_COLUMN, component="morrison-mp10"))
    levels = require_grid_extent(ground).levels
    assert not levels.lowest_layers_equally_spaced and levels.highest_layers_equally_spaced
    assert stretched_levels_reasons(_flat(ground)) == ("ends",)
    assert [r.option for r in selected_by_document(ground)] == ["grid.stretch_z"]
    assert [r.option for r in selected_for_run(ground)][:1] == ["grid.stretch_z"]
    block = receipt(selected_by_document(ground))
    assert block["count"] == 1 and block["capabilities"][0]["values"] == [1, 3, 4]
    assert set(block["capabilities"][0]) == {
        "marker", "option", "values", "not_checked", "would_show", "authority",
    }
    # The receipt says this run's reason, not the whole register's.
    assert block["capabilities"][0]["not_checked"].startswith("levels whose lowest or highest three layers")
    assert "15.6 % high" in block["capabilities"][0]["not_checked"]
    json.dumps(block, allow_nan=False, sort_keys=True)
    assert stretched_levels_reasons(_flat(parse_mapping(_stretched(**HURRICANE_COLUMN)))) == ("ends", "scheme")

    # To the top: the highest three layers are unequal, and the row is selected too.
    top = parse_mapping(_stretched(**{**HURRICANE_COLUMN, "str_bot_m": 150.0, "nz": 52, "ztop_m": 24650.0,
                                      "str_top_m": 24650.0, "dz_top_m": 950.0}))
    levels = require_grid_extent(top).levels
    assert levels.lowest_layers_equally_spaced and not levels.highest_layers_equally_spaced
    assert [r.option for r in selected_by_document(top)] == ["grid.stretch_z"]

    # Three equal layers at the ground on the same column: not the unequal-ends reason, but a column deeper than
    # the compared one, so still a row (test_a_deeper_column_with_equal_end_layers_carries_the_row).
    equal = parse_mapping(_stretched(**{**HURRICANE_COLUMN, "nz": 62, "str_bot_m": 150.0,
                                        "str_top_m": 5650.0, "ztop_m": 25150.0}, component="morrison-mp10"))
    assert require_grid_extent(equal).levels.fornberg_matches_pinned_kernels()
    assert not end_layers_unequal(_flat(equal))
    assert stretched_levels_reasons(_flat(equal)) == ("deeper",)

    # The same column from the ground, written as explicit w levels, is still selected.
    faces = list(require_grid_extent(ground).levels.face_heights_m)
    explicit = _stretched(**HURRICANE_COLUMN)
    explicit["grid"].update(stretch_z=4, levels_m=faces, str_bot_m=0.0, str_top_m=0.0,
                            dz_bot_m=0.0, dz_top_m=0.0, ztop_m=0.0)
    assert [r.option for r in selected_by_document(parse_mapping(explicit))] == ["grid.stretch_z"]

    # A record that does not carry the column's words selects nothing.
    assert selected({"grid.stretch_z": 1}) == ()
    assert not end_layers_unequal({"grid.stretch_z": 0, "grid.nz": 40, "grid.dz_m": 500.0})


def test_the_unequal_end_layers_row_says_what_the_storm_comparison_found() -> None:
    """The row's numbers are the comparison's, where the development evidence is present."""

    folder = ROOT / "evidence" / "oracle" / "stretched-ground-supercell"
    if not (folder / "free-slip.json").is_file():
        pytest.skip("the development evidence is not distributed")
    from cm1core.cm1import.register import STRETCHED_LEVELS_PLAIN, STRETCHED_LEVELS_TEXT
    from cm1core.unvalidated import _NOT_CHECKED_BY_REASON

    (row,) = [row for row in UNVALIDATED_CAPABILITIES if row.option == "grid.stretch_z"]
    drag = json.loads((folder / "drag.json").read_text())["verdicts"]["gm1"]
    assert drag["all_inside_band"] is True
    assert f"{drag['ratio_min']:.3f} to {drag['ratio_max']:.3f}" in row.not_checked
    assert f"{drag['ratio_min']:.3f} to {drag['ratio_max']:.3f}" in _NOT_CHECKED_BY_REASON["ends"]
    free = json.loads((folder / "free-slip.json").read_text())["verdicts"]["gm1"]
    assert free["all_inside_band"] is False
    outside = {name: item["ratio"] for name, item in free["rows"].items() if not item["inside_band"]}
    assert list(outside) == ["zvort_max"]
    assert f"{100 * (outside['zvort_max'] - 1):.1f} % high" in row.not_checked
    assert f"{100 * (outside['zvort_max'] - 1):.1f} % high" in _NOT_CHECKED_BY_REASON["ends"]
    assert f"{round(100 * (outside['zvort_max'] - 1))} % high" in STRETCHED_LEVELS_TEXT["ends"]
    assert f"{round(100 * (outside['zvort_max'] - 1))} % high" in STRETCHED_LEVELS_PLAIN["ends"]
    # The deeper column with three equal 50 m layers at the ground, under free slip.
    equal = json.loads((folder / "three-equal-layers.json").read_text())["verdicts"]["gm1"]
    assert equal["all_inside_band"] is False
    outside = {name: item["ratio"] for name, item in equal["rows"].items() if not item["inside_band"]}
    assert list(outside) == ["zvort_max"]
    assert f"{100 * (outside['zvort_max'] - 1):.1f} % high" in row.not_checked
    assert f"{100 * (outside['zvort_max'] - 1):.1f} % high" in _NOT_CHECKED_BY_REASON["deeper"]
    assert f"{round(100 * (outside['zvort_max'] - 1))} % high" in STRETCHED_LEVELS_TEXT["deeper"]
    assert f"{round(100 * (outside['zvort_max'] - 1))} % high" in STRETCHED_LEVELS_PLAIN["deeper"]


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


# ------------------------------------------------ stretched levels unlike the compared column


def _record(words, component="morrison-mp10", **extra):
    """A flat record: a column's ``grid.*`` words, a scheme, and any other keys."""

    record = {"grid.dz_m": 500.0, **dict(words), "physics.component": component}
    record.update(extra)
    return record


#: A 25.15 km column with three equal 50 m layers at the ground: the free-slip supercell on it had max vertical
#: vorticity 1.119 of CM1's (evidence/oracle/stretched-ground-supercell/three-equal-layers.txt).
DEEP_EQUAL = {"grid.stretch_z": 1, "grid.nz": 62, "grid.ztop_m": 25150.0, "grid.str_bot_m": 150.0,
              "grid.str_top_m": 5650.0, "grid.dz_bot_m": 50.0, "grid.dz_top_m": 500.0}
#: The tornado-grade column of the stretched examples: 20 m layers at the ground to 320 m, 75 levels, 15 km.
TORNADO_GRADE = {"grid.stretch_z": 1, "grid.nz": 75, "grid.dz_m": 100.0, "grid.ztop_m": 15000.0,
                 "grid.str_bot_m": 200.0, "grid.str_top_m": 7000.0, "grid.dz_bot_m": 20.0, "grid.dz_top_m": 320.0}
#: CM1's hurricane vortex over the presets' constant ocean drag.
HURRICANE_CASE = {"initialization.iinit": 7, "closures.bbc": 3, "closures.sfcmodel": 1}


def _reasons(record):
    from cm1core.unvalidated import stretched_levels_reasons

    return stretched_levels_reasons(record)


def _words(record):
    from cm1core.cm1import.register import stretched_levels_words

    return stretched_levels_words(record)


def _binary32_scalar_heights(words) -> list[float]:
    """The scalar heights of a column as CM1's input_grid_z holds them: binary32, printed to 6 decimals."""

    import numpy as np

    from cm1core.unvalidated import _record_levels

    return [float(f"{float(np.float32(value)):.6f}") for value in _record_levels(_record(words)).mass_heights_m]


def test_the_compared_column_with_morrison_carries_no_row_however_it_is_written() -> None:
    """The 53 levels with Morrison: no row from the stretch words, from w levels, or from binary32 heights."""

    from cm1core.unvalidated import COMPARED_STRETCHED_COLUMN, _record_levels, end_layers_unequal

    assert _reasons(_record(COMPARED_STRETCHED_COLUMN)) == ()
    assert selected(_record(COMPARED_STRETCHED_COLUMN)) == ()
    faces = list(_record_levels(COMPARED_STRETCHED_COLUMN).face_heights_m)
    as_w = {"grid.stretch_z": 4, "grid.nz": 53, "grid.levels_m": faces}
    assert _reasons(_record(as_w)) == ()
    # CM1's input_grid_z of the same column (stretch_z = 3): binary32 heights leave the three top layers a few
    # mm apart, which the kernels' exact test sees and this register reads as the equal layers they are.
    as_file = {"grid.stretch_z": 3, "grid.nz": 53, "grid.levels_m": _binary32_scalar_heights(COMPARED_STRETCHED_COLUMN)}
    levels = _record_levels(_record(as_file))
    top = [levels.spacings_m[-1 - index] for index in range(3)]
    assert max(top) - min(top) > 1e-3 and not levels.fornberg_matches_pinned_kernels()
    assert not end_layers_unequal(_record(as_file))
    assert _reasons(_record(as_file)) == ()
    assert selected(_record(as_file)) == ()


def test_a_deeper_column_with_equal_end_layers_carries_the_row() -> None:
    """The 25.15 km column with three equal 50 m layers at the ground: free slip came out 11.9 % high."""

    assert _reasons(_record(DEEP_EQUAL)) == ("deeper",)
    (row,) = selected(_record(DEEP_EQUAL))
    assert row.option == "grid.stretch_z"
    assert row.not_checked.startswith("these levels reach higher than the one stretched column compared with CM1")
    assert "11.9 % high" in row.not_checked and "13 to 14 km" in row.would_show
    text, plain = _words(_record(DEEP_EQUAL))
    assert text.startswith("Runs. These levels go higher than the one stretched column compared with CM1")
    assert "12 % high" in text and "12 % high" in plain
    # Its own binary32 heights are the same column, and the same reason.
    as_file = {"grid.stretch_z": 3, "grid.nz": 62, "grid.levels_m": _binary32_scalar_heights(DEEP_EQUAL)}
    assert _reasons(_record(as_file)) == ("deeper",)


def test_other_stretched_levels_carry_the_row() -> None:
    """Equal end layers, no higher than the compared column, but not it: no storm on them was compared."""

    from cm1core.unvalidated import COMPARED_STRETCHED_COLUMN

    assert _reasons(_record(TORNADO_GRADE)) == ("levels",)
    (row,) = selected(_record(TORNADO_GRADE))
    assert row.not_checked.startswith("these are not the stretched levels compared with CM1")
    text, plain = _words(_record(TORNADO_GRADE))
    assert text.startswith("Runs. These are not the stretched levels compared with CM1 in a storm")
    assert "and not this one" in plain
    # One more level on the compared column's words is another column.
    other = {**COMPARED_STRETCHED_COLUMN, "grid.nz": 54, "grid.ztop_m": 19800.0}
    assert _reasons(_record(other)) == ("deeper",)
    lower = {**COMPARED_STRETCHED_COLUMN, "grid.nz": 52, "grid.ztop_m": 18800.0}
    assert _reasons(_record(lower)) == ("levels",)


def test_wsm6_and_thompson_on_stretched_levels_carry_the_row() -> None:
    """No storm on stretched levels has run WSM6 or Thompson, on the compared column or any other."""

    from cm1core.unvalidated import COMPARED_STRETCHED_COLUMN

    for component, name, ptype in (("wsm6-mp6", "WSM6", "ptype = 7"), ("thompson-mp8", "Thompson", "ptype = 3"),
                                   ("thompson-aerosol-mp28", "Thompson with aerosols", "no CM1 ptype")):
        record = _record(COMPARED_STRETCHED_COLUMN, component)
        assert _reasons(record) == ("scheme",), component
        (row,) = selected(record)
        assert row.not_checked.startswith(f"{name} ({ptype}) has not been run on stretched levels"), component
        text, plain = _words(record)
        assert text == (f"Runs. {name} ({ptype}) has not been run on stretched levels; the storms compared with "
                        "CM1 on stretched levels used Morrison (ptype = 5)")
        assert plain.startswith(f"These levels run, but {name} has not been checked against CM1 on stretched levels")
        # With other levels too: both reasons in the text, the scheme first in the plain words.
        assert _reasons(_record(DEEP_EQUAL, component)) == ("deeper", "scheme")
        text, plain = _words(_record(DEEP_EQUAL, component))
        assert "go higher" in text and f"{name} ({ptype})" in text and name in plain


def test_the_hurricane_presets_levels_in_their_own_case_carry_no_row() -> None:
    """CM1 ran the 8 km hurricane on these levels with its vortex, drag and Morrison; a supercell on them is
    another storm, and a deeper column than the compared one."""

    from cm1core.unvalidated import HURRICANE_PRESET_COLUMN

    assert _reasons(_record(HURRICANE_PRESET_COLUMN, **HURRICANE_CASE)) == ()
    assert selected(_record(HURRICANE_PRESET_COLUMN, **HURRICANE_CASE)) == ()
    # Not the hurricane: a free-slip storm, a storm without the vortex, or the vortex without drag.
    assert _reasons(_record(HURRICANE_PRESET_COLUMN)) == ("deeper",)
    assert _reasons(_record(HURRICANE_PRESET_COLUMN, **{**HURRICANE_CASE, "initialization.iinit": 1})) == ("deeper",)
    assert _reasons(_record(HURRICANE_PRESET_COLUMN, **{**HURRICANE_CASE, "closures.bbc": 1})) == ("deeper",)
    # The hurricane with another scheme.
    assert _reasons(_record(HURRICANE_PRESET_COLUMN, "wsm6-mp6", **HURRICANE_CASE)) == ("scheme",)
    # CM1's own hurricane column (str_bot = 0) in the same case is not the presets' levels.
    assert _reasons(_record({**dict(HURRICANE_PRESET_COLUMN), "grid.str_bot_m": 0.0, "grid.str_top_m": 5500.0},
                            **HURRICANE_CASE)) == ("ends",)


def test_a_dry_run_on_equal_end_layers_carries_no_row() -> None:
    """Without cloud, stretched levels are compared with CM1 word for word; unequal ends keep their row."""

    assert _reasons(_record(TORNADO_GRADE, "off")) == ()
    assert _reasons(_record(DEEP_EQUAL, "off")) == ()
    ground = {**DEEP_EQUAL, "grid.nz": 59, "grid.ztop_m": 25000.0, "grid.str_bot_m": 0.0, "grid.str_top_m": 5500.0}
    assert _reasons(_record(ground, "off")) == ("ends",)
    # A record that does not carry the scheme is read as dry: it reports what a run chose.
    no_scheme = {key: value for key, value in _record(DEEP_EQUAL).items() if key != "physics.component"}
    assert _reasons(no_scheme) == ()


def test_every_plain_sentence_of_the_stretched_levels_row_keeps_the_copy_rules() -> None:
    """A full sentence of at most 140 characters, whatever the reasons and the scheme."""

    from cm1core.cm1import.register import STRETCHED_LEVELS_PLAIN
    from cm1core.unvalidated import STRETCHED_SCHEME_WORDS

    for reason, plain in STRETCHED_LEVELS_PLAIN.items():
        for name, _ in [("Morrison", None), *STRETCHED_SCHEME_WORDS.values()]:
            sentence = plain.format(scheme=name)
            assert sentence.endswith(".") and len(sentence) <= 140, (reason, len(sentence), sentence)


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
