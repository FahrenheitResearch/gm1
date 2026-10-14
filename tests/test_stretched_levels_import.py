"""Stretched levels through the importer: the level count CM1 needs, the level file, and the refusals.

CM1 r21.1 does not choose ``nz`` for ``stretch_z = 1``: ``param.F:7823-7877`` counts the layers its five stretch
words imply and stops unless ``nz`` is their sum and each part is a whole number of layers.  The importer and
``gpuwm-cm1 check`` say which ``nz`` fits.  ``stretch_z = 3`` and ``4`` read their heights from ``input_grid_z``
(``param.F:8013``, ``:8083``), which the importer reads from beside the namelist.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from cm1core.cm1_vertical_grid import (
    LEVEL_COUNT_HINT,
    VerticalGridError,
    build_vertical_levels,
    stretch_z1_fit,
    stretch_z1_fit_hint,
)
from cm1core.cm1import import levels as level_file
from cm1import_support import SUPERCELL, edited, namelist_tool

#: The worked example of the guide: 100 m layers at the ground growing to 500 m.
STRETCHED = {"stretch_z": "1", "nz": "53", "ztop": "19300.0", "str_bot": "300.0", "str_top": "9300.0",
             "dz_bot": "100.0", "dz_top": "500.0"}
WORDS = dict(ztop_m=19300.0, str_bot_m=300.0, str_top_m=9300.0, dz_bot_m=100.0, dz_top_m=500.0)


@pytest.fixture(autouse=True)
def _tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", str(namelist_tool()))


def _cli(capsys, *argv: str) -> tuple[int, str]:
    from cm1core.cli import main

    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _rows(rundir: Path, category: str = "refused") -> list[dict]:
    report = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    return [row for section in report["sections"] if section["id"] == category for row in section["rows"]]


# ---------------------------------------------------------------- CM1's level count


def test_the_three_layer_counts_are_cm1s() -> None:
    fit = stretch_z1_fit(**WORDS)
    assert (fit.nk1, fit.nk2, fit.nk3, fit.nz) == (3, 30, 20, 53)
    assert fit.mean_depth_m == 300.0
    # CM1's stock hurricane column stretches from the ground: no layer of dz_bot below str_bot.
    hurricane = stretch_z1_fit(ztop_m=25000.0, str_bot_m=0.0, str_top_m=5500.0, dz_bot_m=50.0, dz_top_m=500.0)
    assert (hurricane.nk1, hurricane.nk2, hurricane.nk3, hurricane.nz) == (0, 20, 39, 59)
    # The count is the one the generator accepts, for both.
    for count, words in ((53, WORDS), (59, dict(ztop_m=25000.0, str_bot_m=0.0, str_top_m=5500.0,
                                                 dz_bot_m=50.0, dz_top_m=500.0))):
        assert build_vertical_levels(stretch_z=1, nk=count, dz_m=500.0, **words).nk == count
    # Words CM1 cannot use at all have no fit.
    assert stretch_z1_fit(ztop_m=1000.0, str_bot_m=2000.0, str_top_m=500.0, dz_bot_m=100.0, dz_top_m=500.0) is None


def test_a_part_that_is_not_a_whole_number_of_layers_has_no_level_count() -> None:
    # NINT(2.5) is 3 in Fortran; 3 layers of 100 m are not 250 m, and CM1 stops.
    fit = stretch_z1_fit(**{**WORDS, "str_bot_m": 250.0})
    assert fit.nk1 == 3 and not fit.bottom_whole and fit.nz is None
    hint = stretch_z1_fit_hint(nk=53, **{**WORDS, "str_bot_m": 250.0})
    assert "str_bot = 250 m is not a whole number of dz_bot = 100 m layers" in hint
    assert "the nearest are 200 and 300 m" in hint
    assert LEVEL_COUNT_HINT not in hint
    top = stretch_z1_fit_hint(nk=53, **{**WORDS, "str_top_m": 9500.0})
    assert "ztop - str_top = 9800 m is not a whole number of dz_top = 500 m layers" in top
    assert "str_top - str_bot = 9200 m is not a whole number of layers of the mean depth" in top


def test_the_hint_names_the_level_count_that_fits() -> None:
    assert stretch_z1_fit_hint(nk=53, **WORDS) == ""
    hint = stretch_z1_fit_hint(nk=40, **WORDS)
    assert f"{LEVEL_COUNT_HINT} 53" in hint
    assert "3 of dz_bot below str_bot, 30 stretched between str_bot and str_top, 20 of dz_top above str_top" in hint
    with pytest.raises(VerticalGridError) as caught:
        build_vertical_levels(stretch_z=1, nk=40, dz_m=500.0, **WORDS)
    assert f"{LEVEL_COUNT_HINT} 53" in str(caught.value)


def test_check_says_the_level_count_that_fits(tmp_path: Path, capsys) -> None:
    from cm1core.config import parse_mapping
    from cm1core.errors import CM1Refusal

    deck = edited(SUPERCELL, tmp_path, STRETCHED)
    code, _ = _cli(capsys, "import", str(deck), "--out", str(tmp_path / "run"))
    assert code == 0
    raw = tomllib.loads((tmp_path / "run" / "deck.toml").read_text(encoding="utf-8"))
    raw["grid"]["nz"] = 40
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-VERTICAL-COORDINATE"
    assert f"{LEVEL_COUNT_HINT} 53" in caught.value.detail.reason


def test_the_import_refuses_a_wrong_level_count_once_and_says_the_right_one(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {**STRETCHED, "nz": "40"})
    rundir = tmp_path / "run"
    code, out = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 2
    refused = _rows(rundir)
    assert [row["keys"] for row in refused] == [["nz"]]
    assert "nz = 40 does not fit these stretched levels: CM1 needs nz = 53 here" in refused[0]["text"]
    assert refused[0]["fix"] == "Set nz = 53."
    assert "need 53" in refused[0]["plain"]
    assert "Fix: Set nz = 53." in out
    # The edit a user then makes runs.
    code, out = _cli(capsys, "edit", str(rundir), "--set", "nz=53")
    assert code == 0 and "Result: Runs" in out


def test_a_levels_refusal_does_not_hide_the_other_refusals(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {**STRETCHED, "str_bot": "250.0", "ptype": "26"})
    rundir = tmp_path / "run"
    code, _ = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 2
    refused = {tuple(row["keys"]): row for row in _rows(rundir)}
    assert set(refused) == {("ptype",), ("str_bot",)}
    assert "not a whole number of dz_bot = 100 m layers" in refused[("str_bot",)]["text"]
    assert "whole number of layers" in refused[("str_bot",)]["fix"]


def test_a_column_that_stretches_from_the_ground_imports(tmp_path: Path, capsys) -> None:
    """CM1's hurricane column on the supercell: str_bot = 0 is CM1's common setting."""

    from cm1core.cartesian_door import require_grid_extent
    from cm1core.config import load_config

    deck = edited(SUPERCELL, tmp_path, {"stretch_z": "1", "nz": "59", "ztop": "25000.0", "str_bot": "0.0",
                                        "str_top": "5500.0", "dz_bot": "50.0", "dz_top": "500.0", "zd": "20000.0"})
    rundir = tmp_path / "run"
    code, out = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 0, out
    assert _rows(rundir) == []
    # It runs, and the report says what the storm comparison on this column found.
    (row,) = _rows(rundir, "not_checked")
    assert row["keys"] == ["stretch_z"]
    assert row["text"].startswith("Runs.")
    assert "with ground friction" in row["text"] and "all ten statistics are within their bands" in row["text"]
    assert "with a free-slip ground one statistic, max vertical vorticity, came out 16 % high" in row["text"]
    assert "16 % high" in row["plain"]
    levels = require_grid_extent(load_config(rundir / "deck.toml")).levels
    assert levels.spacings_m[:3] == pytest.approx((50.0, 73.684, 97.368), abs=1e-3)
    assert not levels.lowest_layers_equally_spaced and levels.highest_layers_equally_spaced


def test_the_compared_column_adds_no_row(tmp_path: Path, capsys) -> None:
    """The measured case: the stretched supercell's 53 levels with Morrison."""

    deck = edited(SUPERCELL, tmp_path, STRETCHED)
    rundir = tmp_path / "run"
    code, out = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 0, out
    assert _rows(rundir) == [] and _rows(rundir, "not_checked") == []


def _stretched_row(tmp_path: Path, capsys, words: dict, *options: str) -> dict | None:
    deck = edited(SUPERCELL, tmp_path, words)
    rundir = tmp_path / "run"
    code, out = _cli(capsys, "import", str(deck), "--out", str(rundir), *options)
    assert code == 0, out
    assert _rows(rundir) == []
    rows = [row for row in _rows(rundir, "not_checked") if row["keys"] == ["stretch_z"]]
    assert len(rows) <= 1
    return rows[0] if rows else None


@pytest.mark.parametrize("case, words, options, text, plain", [
    pytest.param(
        "deeper",
        {"stretch_z": "1", "nz": "62", "ztop": "25150.0", "str_bot": "150.0", "str_top": "5650.0",
         "dz_bot": "50.0", "dz_top": "500.0", "zd": "20000.0"}, (),
        "Runs. These levels go higher than the one stretched column compared with CM1 in a storm",
        "These levels run, but go higher than the one stretched set checked against CM1",
        id="a 25 km column with three equal 50 m layers at the ground"),
    pytest.param(
        "levels",
        {"stretch_z": "1", "nz": "75", "dz": "100.0", "ztop": "15000.0", "str_bot": "200.0", "str_top": "7000.0",
         "dz_bot": "20.0", "dz_top": "320.0", "zd": "10000.0"}, (),
        "Runs. These are not the stretched levels compared with CM1 in a storm",
        "These levels run, but only one stretched set has been checked against CM1 in a storm, and not this one.",
        id="the tornado-grade column"),
    pytest.param(
        "wsm6", {**STRETCHED, "ptype": "7"}, (),
        "Runs. WSM6 (ptype = 7) has not been run on stretched levels",
        "These levels run, but WSM6 has not been checked against CM1 on stretched levels",
        id="WSM6 on the compared column"),
    pytest.param(
        "thompson", {**STRETCHED, "ptype": "3", "nt_c": "100.0"}, ("--acknowledge-thompson",),
        "Runs. Thompson (ptype = 3) has not been run on stretched levels",
        "These levels run, but Thompson has not been checked against CM1 on stretched levels",
        id="Thompson on the compared column"),
])
def test_stretched_levels_unlike_the_compared_column_have_a_row_with_the_reason(
        tmp_path: Path, capsys, case: str, words: dict, options: tuple, text: str, plain: str) -> None:
    row = _stretched_row(tmp_path, capsys, words, *options)
    assert row is not None, case
    assert row["text"].startswith(text), row["text"]
    assert row["plain"].startswith(plain) and row["plain"].endswith(".") and len(row["plain"]) <= 140
    assert row["evidence"] == "src/cm1core/unvalidated.py"


def test_the_compared_column_from_binary32_heights_adds_no_row(tmp_path: Path, capsys) -> None:
    """CM1's input_grid_z of the compared levels: binary32 heights, the same column to a few mm, no row."""

    import numpy as np

    source = tmp_path / "case"
    source.mkdir()
    deck = edited(SUPERCELL, source, {"stretch_z": "3", "nz": "53"})
    (source / "input_grid_z").write_text(
        "".join(f"{float(np.float32(value)):14.6f}\n" for value in _heights()), encoding="ascii")
    rundir = tmp_path / "run"
    code, out = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 0, out
    assert _rows(rundir) == [] and _rows(rundir, "not_checked") == []


# ---------------------------------------------------------------- input_grid_z


def _heights(count: int = 53) -> list[float]:
    return list(build_vertical_levels(stretch_z=1, nk=count, dz_m=500.0, **WORDS).mass_heights_m)


def test_the_level_file_is_read_as_cm1_reads_it() -> None:
    text = "  2.5\n\n 7.6, the second level\n1.29D1  ignored\n2.0e1\n".encode()
    assert level_file.parse(text, stretch_z=3, nz=4) == (2.5, 7.6, 12.9, 20.0)
    assert level_file.parse(b"0.0\n100.\n250\n", stretch_z=4, nz=2) == (0.0, 100.0, 250.0)
    assert level_file.expected_count(3, 40) == 40 and level_file.expected_count(4, 40) == 41
    for data, stretch, nz, fragment in (
        (b"50\n150\n", 3, 3, "holds 2 heights, and stretch_z = 3 with nz = 3 reads 3 scalar levels"),
        (b"0\n100\n", 4, 2, "reads 3 w levels"),
        (b"50\nabc\n", 3, 2, "line 2 does not start with a height in metres"),
        (b"50\n40\n", 3, 2, "heights must rise"),
        (b"10\n100\n200\n", 4, 2, "the first is the ground, 0 m"),
        (b"0\n100\n", 3, 2, "the first is above the ground"),
    ):
        with pytest.raises(level_file.LevelsFileError) as caught:
            level_file.parse(data, stretch_z=stretch, nz=nz)
        assert fragment in str(caught.value)


def test_the_import_reads_input_grid_z_from_beside_the_namelist(tmp_path: Path, capsys) -> None:
    from cm1core.cartesian_door import require_grid_extent
    from cm1core.config import load_config

    source = tmp_path / "case"
    source.mkdir()
    deck = edited(SUPERCELL, source, {"stretch_z": "3", "nz": "53"})
    (source / "input_grid_z").write_text("".join(f"{value:14.6f}\n" for value in _heights()), encoding="ascii")
    rundir = tmp_path / "run"
    code, out = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 0, out
    assert _rows(rundir) == []
    assert (rundir / "input_grid_z").read_bytes() == (source / "input_grid_z").read_bytes()
    report = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    assert report["files"]["levels"] == "input_grid_z"
    grid = tomllib.loads((rundir / "deck.toml").read_text(encoding="utf-8"))["grid"]
    assert grid["stretch_z"] == 3 and len(grid["levels_m"]) == 53
    # CM1 reads none of the analytic stretch words on this branch, and overwrites ztop.
    assert [grid[name] for name in ("ztop_m", "str_bot_m", "str_top_m", "dz_bot_m", "dz_top_m")] == [0.0] * 5
    # The column is the one stretch_z = 1 builds from the same heights, to the file's printed precision.
    levels = require_grid_extent(load_config(rundir / "deck.toml")).levels
    analytic = build_vertical_levels(stretch_z=1, nk=53, dz_m=500.0, **WORDS)
    assert max(abs(a - b) for a, b in zip(levels.face_heights_m, analytic.face_heights_m)) < 0.01
    # An edit imports again from the run folder and keeps the file it was given.
    (source / "input_grid_z").unlink()
    code, out = _cli(capsys, "edit", str(rundir), "--set", "tapfrq=300")
    assert code == 0, out
    assert len(tomllib.loads((rundir / "deck.toml").read_text(encoding="utf-8"))["grid"]["levels_m"]) == 53


def test_without_the_level_file_the_import_says_where_to_put_it(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {"stretch_z": "3", "nz": "53"})
    rundir = tmp_path / "run"
    code, out = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 2
    refused = _rows(rundir)
    assert [row["keys"] for row in refused] == [["stretch_z"]]
    assert "there is no input_grid_z beside the namelist" in refused[0]["text"]
    assert refused[0]["fix"].startswith("Put CM1's input_grid_z beside the namelist and import again")
    assert json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))["files"]["levels"] is None


def test_a_level_file_cm1_would_stop_on_stops_the_import(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {"stretch_z": "3", "nz": "53"})
    (tmp_path / "input_grid_z").write_text("50.0\n150.0\n", encoding="ascii")
    code, out = _cli(capsys, "import", str(deck), "--out", str(tmp_path / "run"))
    assert code == 2
    assert "input_grid_z holds 2 heights, and stretch_z = 3 with nz = 53 reads 53 scalar levels" in out


def test_stretch_z_2_says_why_and_what_runs_the_same_levels(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {"stretch_z": "2"})
    rundir = tmp_path / "run"
    code, _ = _cli(capsys, "import", str(deck), "--out", str(rundir))
    assert code == 2
    refused = _rows(rundir)
    assert [row["keys"] for row in refused] == [["stretch_z"]]
    assert "does not make them end at ztop" in refused[0]["text"]
    assert "a GM1 grid has one model top" in refused[0]["text"]
    assert "input_grid_z file and use stretch_z = 3" in refused[0]["fix"]
