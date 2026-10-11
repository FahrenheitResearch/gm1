"""``gpuwm-cm1 import NAMELIST --out RUNDIR``: the run directory and the exit codes.

0 runs, 2 refused, 3 runs once acknowledged.  The directory holds
the user's file byte for byte, the run's namelist, the report as JSON and text,
settings.toml and, when the namelist runs, deck.toml -- which the
configuration door admits.  Importing an edited namelist into the same
directory re-imports it and keeps the diff; a directory that holds a run is
refused.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from cm1import_support import R22_SUPERCELL, SUPERCELL, edited, flat_toml, namelist_tool


@pytest.fixture(autouse=True)
def _tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", str(namelist_tool()))


def _import(capsys, *argv: str) -> tuple[int, str]:
    from cm1core.cli import main

    code = main(["import", *argv])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def test_the_stock_deck_imports_and_its_deck_is_admitted(tmp_path: Path, capsys) -> None:
    from cm1core.cm1import import import_namelist
    from cm1core.config import load_config

    rundir = tmp_path / "run1"
    code, out = _import(capsys, str(SUPERCELL), "--out", str(rundir))
    assert code == 0
    assert out.startswith("CM1 namelist: ") and "Result: Runs, with the differences below" in out
    names = {path.name for path in rundir.iterdir()}
    assert names == {"namelist.input.original", "namelist.input", "import-report.json", "import-report.txt",
                     "settings.toml", "deck.toml", "run.json"}
    assert (rundir / "namelist.input.original").read_bytes() == SUPERCELL.read_bytes()
    report = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    assert report["exit"] == 0 and report["verdict"] == "runs"
    assert report["command"].startswith("gpuwm-cm1 import ") and report["files"]["deck"] == "deck.toml"
    assert (rundir / "import-report.txt").read_text(encoding="utf-8") == out
    config = load_config(rundir / "deck.toml")
    assert config.run.lane.value == "cartesian"
    # For a namelist that needs no set-aside, the deck is the importer's, except
    # that rstfrq < 0 (CM1 writes no restart) keeps one restart at the end (plan
    # 4.4, cm1run.launch) rather than the translator's |rstfrq| cadence.
    deck = flat_toml((rundir / "deck.toml").read_text())
    shared = flat_toml(import_namelist(SUPERCELL, tool=namelist_tool()).toml())
    assert shared["run.restart_interval_s"] == 3600.0
    assert deck["run.restart_interval_s"] == deck["run.duration_s"] == 7200.0
    assert deck == {**shared, "run.restart_interval_s": 7200.0}
    settings = tomllib.loads((rundir / "settings.toml").read_text(encoding="utf-8"))
    assert settings["schema"] == "gm1.cm1-run-settings/v1" and settings["output"]["format"] == 1


def test_a_refused_namelist_exits_2_without_a_deck(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {"stretch_x": "1", "ptype": "26", "isnd": "15"})
    rundir = tmp_path / "run2"
    code, out = _import(capsys, str(deck), "--out", str(rundir), "--json")
    assert code == 2
    payload = json.loads(out)
    assert payload["verdict"] == "refused" and payload["exit"] == 2
    refused = {key for row in payload["sections"][0]["rows"] for key in row["keys"]}
    assert {"stretch_x", "ptype", "isnd"} <= refused
    assert not (rundir / "deck.toml").exists()


def test_acknowledgment_turns_3_into_0(tmp_path: Path, capsys) -> None:
    deck = edited(R22_SUPERCELL, tmp_path, {"brkup_db": "300.0e-6"})
    code, out = _import(capsys, str(deck), "--out", str(tmp_path / "a"))
    assert code == 3 and "--acknowledge-output-keys" in out
    assert (tmp_path / "a" / "deck.toml").is_file()
    code, _ = _import(capsys, str(deck), "--out", str(tmp_path / "b"), "--acknowledge-output-keys")
    assert code == 0
    report = json.loads((tmp_path / "b" / "import-report.json").read_text(encoding="utf-8"))
    assert report["acknowledgments"] == {"needed": ["output-keys"], "given": ["output-keys"], "missing": []}


def test_an_edited_namelist_is_reimported_and_its_diff_kept(tmp_path: Path, capsys) -> None:
    rundir = tmp_path / "run"
    assert _import(capsys, str(SUPERCELL), "--out", str(rundir))[0] == 0
    namelist = rundir / "namelist.input"
    namelist.write_text(namelist.read_text(encoding="ascii").replace("kdiv    = 0.10", "kdiv    = 0.20"),
                        encoding="ascii")
    code, out = _import(capsys, str(namelist), "--out", str(rundir))
    assert code == 0
    assert (rundir / "namelist.input.original").read_bytes() == SUPERCELL.read_bytes()
    diff = (rundir / "namelist.diff").read_text(encoding="utf-8")
    assert "-" in diff and "kdiv    = 0.20" in diff
    report = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    assert report["reimport"] is True and report["namelist"]["edited"] is True
    assert any(row["keys"] == ["kdiv"] for row in report["sections"][3]["rows"])


def test_a_directory_that_is_not_an_import_or_holds_a_run_is_refused(tmp_path: Path, capsys) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("mine", encoding="ascii")
    code, out = _import(capsys, str(SUPERCELL), "--out", str(other))
    assert code == 2 and "not an imported run directory" in out
    rundir = tmp_path / "run"
    assert _import(capsys, str(SUPERCELL), "--out", str(rundir))[0] == 0
    (rundir / "segments").mkdir()
    code, out = _import(capsys, str(SUPERCELL), "--out", str(rundir))
    assert code == 2 and "holds a run" in out


def test_an_unreadable_namelist_writes_a_refused_report(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "namelist.input"
    broken.write_text(" &param0\n nx = = 3\n /\n", encoding="ascii")
    rundir = tmp_path / "run"
    code, out = _import(capsys, str(broken), "--out", str(rundir))
    assert code == 2
    report = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "refused" and report["error"]
    assert report["sections"][0]["rows"][0]["code"] == "CM1-IMPORT-NAMELIST-SYNTAX"


def test_a_file_sounding_is_copied_and_kept_on_reimport(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {"isnd": "7", "iwnd": "0"})
    sounding = tmp_path / "my_sounding"
    lines = ["1000.0 300.0 14.0"] + [f"{z:.1f} {300 + z / 250:.2f} {max(14 - z / 1000, 0.0):.3f} 12.5 3.0"
                                     for z in range(250, 25001, 250)]
    sounding.write_text("\n".join(lines) + "\n", encoding="ascii")
    rundir = tmp_path / "run"
    code, out = _import(capsys, str(deck), "--out", str(rundir), "--sounding", str(sounding))
    assert code == 0, out
    assert (rundir / "input_sounding").read_bytes() == sounding.read_bytes()
    code, out = _import(capsys, str(rundir / "namelist.input"), "--out", str(rundir))
    assert code == 0, out
    report = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    assert report["sounding"]["path"].endswith("input_sounding") and report["sounding"]["problems"] == []


def test_irandp_gets_a_seed_in_settings_and_deck(tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {"irandp": "1"})
    rundir = tmp_path / "run"
    code, _ = _import(capsys, str(deck), "--out", str(rundir), "--seed", "7")
    assert code == 0
    assert tomllib.loads((rundir / "settings.toml").read_text())["initialization"]["random_seed"] == 7
    assert tomllib.loads((rundir / "deck.toml").read_text())["initialization"]["random_seed"] == 7
    # Re-importing the edited namelist keeps the seed the run directory has.
    namelist = rundir / "namelist.input"
    namelist.write_text(namelist.read_text().replace("kdiv    = 0.10", "kdiv    = 0.20"))
    code, _ = _import(capsys, str(namelist), "--out", str(rundir))
    assert code == 0 and (rundir / "namelist.diff").is_file()
    assert tomllib.loads((rundir / "deck.toml").read_text())["initialization"]["random_seed"] == 7
    code, _ = _import(capsys, str(namelist), "--out", str(rundir), "--seed", "9")
    assert tomllib.loads((rundir / "deck.toml").read_text())["initialization"]["random_seed"] == 9


@pytest.mark.parametrize("seed", ["0", "-3", str(2**64), "1.5"])
def test_a_seed_config_would_refuse_is_stopped_at_the_door(seed: str, tmp_path: Path, capsys) -> None:
    deck = edited(SUPERCELL, tmp_path, {"irandp": "1"})
    rundir = tmp_path / "run"
    with pytest.raises(SystemExit) as stopped:
        _import(capsys, str(deck), "--out", str(rundir), "--seed", seed)
    assert stopped.value.code == 2
    assert "--seed" in capsys.readouterr().err
    assert not rundir.exists()
    from cm1core.cm1import.door import ImportDoorError, run_import

    if seed.lstrip("-").isdigit():
        with pytest.raises(ImportDoorError):
            run_import(deck, rundir, seed=int(seed))
        assert not rundir.exists()


def test_a_large_grid_gets_gm1s_own_tiles_in_its_deck(tmp_path: Path, capsys) -> None:
    from cm1core.config import load_config

    deck = edited(SUPERCELL, tmp_path, {"nx": "600", "ny": "600"})
    rundir = tmp_path / "run"
    code, out = _import(capsys, str(deck), "--out", str(rundir))
    assert code == 0, out
    runtime = tomllib.loads((rundir / "deck.toml").read_text())["runtime"]
    assert runtime["tile_nx"] == runtime["tile_ny"] == 128
    assert tomllib.loads((rundir / "settings.toml").read_text())["runtime"]["tile_nx"] == 128
    load_config(rundir / "deck.toml")
    report = json.loads((rundir / "import-report.json").read_text(encoding="utf-8"))
    assert report["check"]["status"] == "passed" and report["check"]["set_aside"] == {}


def test_an_interrupted_or_annotated_import_can_be_imported_again(tmp_path: Path, capsys) -> None:
    rundir = tmp_path / "run"
    rundir.mkdir()
    (rundir / "namelist.input.original").write_bytes(SUPERCELL.read_bytes())
    (rundir / "namelist.input").write_bytes(SUPERCELL.read_bytes())
    code, out = _import(capsys, str(rundir / "namelist.input"), "--out", str(rundir))
    assert code == 0, out
    (rundir / "commands.log").write_text("gpuwm-cm1 import ...\n", encoding="ascii")  # the GUI's log
    code, out = _import(capsys, str(rundir / "namelist.input"), "--out", str(rundir))
    assert code == 0, out


def test_the_show_command_line_repeats_the_import(tmp_path: Path) -> None:
    """A GUI import (run_import, no command given) records the full line, flags included."""

    import shlex

    from cm1core.cm1import.door import run_import

    deck = edited(SUPERCELL, tmp_path, {"irandp": "1", "ptype": "3", "nt_c": "100.0"})
    rundir = tmp_path / "run"
    code, payload, _ = run_import(deck, rundir, acknowledge=("thompson",), seed=11)
    assert code == 0
    argv = shlex.split(payload["command"])
    assert argv[:5] == ["gpuwm-cm1", "import", str(deck), "--out", str(rundir)]
    assert "--acknowledge-thompson" in argv and argv[argv.index("--seed") + 1] == "11"
    # The line, run from a terminal, gives the same verdict and deck.
    from cm1core.cli import main

    other = tmp_path / "again"
    assert main([argv[1], argv[2], "--out", str(other), *argv[5:]]) == 0
    assert (other / "deck.toml").read_text() == (rundir / "deck.toml").read_text()


def test_a_parse_error_names_the_users_file_and_the_line(tmp_path: Path, capsys) -> None:
    text = SUPERCELL.read_text(encoding="ascii").replace(" kdiv    = 0.10,", " kdiv    = 0.10,  ! 5° damping")
    pasted = tmp_path / "pasted.input"
    pasted.write_text(text, encoding="utf-8")
    number = next(index for index, line in enumerate(text.splitlines(), start=1) if "°" in line)
    code, out = _import(capsys, str(pasted), "--out", str(tmp_path / "run"))
    assert code == 2
    assert f"{pasted}: CM1 NAMELIST must use ASCII source bytes: line {number}, column" in out
    assert str(tmp_path / "run" / "namelist.input") not in out
    # A byte offset in the parser's message is given as a line and column too.
    broken = tmp_path / "broken.input"
    broken.write_text(text.replace("\u00b0", "").replace("kdiv    = 0.10,", "kdiv    = 0.1 0.2 junk,"),
                      encoding="ascii")
    code, out = _import(capsys, str(broken), "--out", str(tmp_path / "run2"))
    assert code == 2 and f"(line {number}, column 16)" in out


# ------------------------------------------------ the run directory (cm1run)


def _status(rundir: Path):
    from cm1core.cm1run.status import run_status

    return run_status(rundir)


def test_an_import_is_a_run_directory_that_run_and_status_read(tmp_path: Path, capsys) -> None:
    """run.json is written last, from the report: ``run`` and ``status`` read the import's verdict."""

    from cm1core.cm1run.rundir import RUN_SCHEMA, RunDirectory, sha256_file

    rundir = tmp_path / "run"
    assert _import(capsys, str(SUPERCELL), "--out", str(rundir))[0] == 0
    record = RunDirectory(rundir).load()
    assert record["schema"] == RUN_SCHEMA and record["name"] == "run" and record["segments"] == []
    assert record["import"]["result"] == "runnable" and record["import"]["exit"] == 0
    assert record["import"]["component"] == "morrison-mp10" and record["import"]["refused"] == []
    assert record["deck"] == {"file": "deck.toml", "sha256": sha256_file(rundir / "deck.toml")}
    assert record["namelist"]["original_sha256"] == sha256_file(SUPERCELL)
    assert record["cm1"]["rstfrq"] == -3600.0 and record["cm1"]["timax"] == 7200.0
    status = _status(rundir)
    assert status.state == "ready" and status.label == "Ready"
    created = record["created_utc"]
    # A re-import keeps the run's name and creation time and replaces the verdict.
    namelist = rundir / "namelist.input"
    namelist.write_text(namelist.read_text(encoding="ascii").replace(" ptype     =  5,", " ptype     = 26,"),
                        encoding="ascii")
    assert _import(capsys, str(namelist), "--out", str(rundir))[0] == 2
    record = RunDirectory(rundir).load()
    assert namelist.read_text(encoding="ascii").count(" ptype     = 26,") == 1
    assert record["created_utc"] == created and record["deck"] is None
    assert record["import"]["result"] == "refused" and record["import"]["refused"][0].startswith("ptype = 26")
    status = _status(rundir)
    assert status.state == "imported" and status.detail.startswith("Cannot run as written: ptype = 26")


def test_a_run_needing_an_acknowledgment_waits_for_it(tmp_path: Path, capsys) -> None:
    from cm1core.cm1run.launch import LaunchRefusal, StartRequest, prepare
    from cm1core.cm1run.rundir import RunDirectory

    deck = edited(R22_SUPERCELL, tmp_path, {"brkup_db": "300.0e-6"})
    rundir = tmp_path / "run"
    assert _import(capsys, str(deck), "--out", str(rundir))[0] == 3
    record = RunDirectory(rundir).load()
    assert record["import"]["result"] == "acknowledge" and record["acknowledgments"] == []
    assert [item["id"] for item in record["import"]["acknowledgments_needed"]] == ["output-keys"]
    assert _status(rundir).detail.startswith("Needs your acknowledgment of 1 setting")
    with pytest.raises(LaunchRefusal) as refused:
        prepare(RunDirectory(rundir), StartRequest())
    assert refused.value.code == "GM1-RUN-ACKNOWLEDGE" and "GM1 does not read" in refused.value.reason
    assert _import(capsys, str(rundir / "namelist.input"), "--out", str(rundir),
                   "--acknowledge-output-keys")[0] == 0
    record = RunDirectory(rundir).load()
    assert record["acknowledgments"] == ["output-keys"] and record["import"]["result"] == "runnable"
    assert _status(rundir).state == "ready"


def test_an_unreadable_namelist_still_makes_a_run_directory_that_says_why(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "namelist.input"
    broken.write_text(" &param0\n nx = = 3\n /\n", encoding="ascii")
    rundir = tmp_path / "run"
    assert _import(capsys, str(broken), "--out", str(rundir))[0] == 2
    status = _status(rundir)
    assert status.state == "imported" and "cannot" in status.detail.lower()

