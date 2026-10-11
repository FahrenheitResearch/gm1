"""``gpuwm-cm1 edit`` and the run's own settings: what the setup screens' every change runs.

- ``--set`` changes CM1 namelist words in place and imports the run again; the diff against the user's original
  file is kept, and a word that is not a CM1 r21.1 setting is refused before anything is written.
- ``--setting`` changes a value CM1 keeps in Fortran (the bubble, cold pool, line thermal, parcel seed box) in
  settings.toml; it reaches deck.toml and survives every later import of the run; ``--reset-setting`` puts it back.
- The acknowledgments the last import was given are kept.  A run that has run is not edited.
- The import report's ``setup`` block, and the estimator's recommended frame-and-bubble pair.
CPU only; the real native namelist parser.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib

import pytest

from cm1import_support import R22_SUPERCELL, SQUALL_LINE, SUPERCELL, edited, namelist_tool

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def parser(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


def _main(*argv: str) -> int:
    from cm1core.cli import main

    return main([str(arg) for arg in argv])


def _imported(tmp_path: Path, source: Path = SUPERCELL, name: str = "run") -> Path:
    rundir = tmp_path / name
    assert _main("import", source, "--out", rundir) == 0
    return rundir


def _deck(rundir: Path) -> dict:
    return tomllib.loads((rundir / "deck.toml").read_text())


def test_set_changes_the_word_in_place_and_imports_again(tmp_path, parser, capsys):
    rundir = _imported(tmp_path)
    original = (rundir / "namelist.input").read_text()
    capsys.readouterr()
    assert _main("edit", rundir, "--set", "umove=15.0", "--set", "tapfrq=600", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    text = (rundir / "namelist.input").read_text()
    assert " umove   = 15.0," in text and " tapfrq =   600," in text
    # every other line is the user's
    assert [a for a, b in zip(original.splitlines(), text.splitlines()) if a != b] == [
        " tapfrq =   900.0,", " umove   = 12.5,"]
    assert (rundir / "namelist.input.original").read_text() == original
    assert (rundir / "namelist.diff").read_text().startswith("--- namelist.input.original")
    assert payload["reimport"] is True and payload["exit"] == 0
    assert payload["edit"]["namelist_changed"] is True and "+ umove   = 15.0," in payload["edit"]["diff"]
    assert payload["command"].startswith("gpuwm-cm1 edit ")
    deck = _deck(rundir)
    assert deck["run"]["history_interval_s"] == 600.0


def test_a_word_cm1_does_not_have_is_refused_before_anything_is_written(tmp_path, parser, capsys):
    rundir = _imported(tmp_path)
    before = {p.name: p.read_bytes() for p in rundir.iterdir() if p.is_file()}
    assert _main("edit", rundir, "--set", "bubble_radius=3") == 2
    assert "not a CM1 r21.1 namelist setting" in capsys.readouterr().err
    assert _main("edit", rundir, "--setting", "initialization.zc=1") == 2
    assert _main("edit", rundir, "--setting", "initialization.bubble_center_x_m=0") == 2
    assert "set together" in capsys.readouterr().err
    assert _main("edit", rundir, "--setting", "initialization.bhrad_m=-3") == 2
    after = {p.name: p.read_bytes() for p in rundir.iterdir() if p.is_file()}
    assert after == before


def test_a_value_cm1_cannot_read_for_its_word_is_refused_before_anything_is_written(tmp_path, parser, capsys):
    # Every r21.1 namelist word is an integer, a real or a logical; a value of another kind would be written into
    # namelist.input and leave the run folder with a namelist CM1's reader stops on.
    rundir = _imported(tmp_path)
    before = {p.name: p.read_bytes() for p in rundir.iterdir() if p.is_file()}
    capsys.readouterr()
    for assignment, words in (("nx='abc'", "a whole number"), ("nx=120.5", "a whole number"),
                              ("procfiles=1", ".true. or .false."), ("dx=nan", "a number"),
                              ("umove=.true.", "a number"), ("dx='1000'", "a number")):
        assert _main("edit", rundir, "--set", assignment) == 2, assignment
        assert words in capsys.readouterr().err, assignment
    assert {p.name: p.read_bytes() for p in rundir.iterdir() if p.is_file()} == before
    # A whole number written as a real is still one; a real word takes a whole number, as CM1's reader does.
    assert _main("edit", rundir, "--set", "nx=120.0", "--set", "dx=1000", "--set", "procfiles=F") == 0
    text = (rundir / "namelist.input").read_text()
    assert " nx           =     120," in text and " dx     =  1000," in text and "procfiles    = .false.," in text


def test_a_setting_reaches_the_deck_and_survives_every_later_import(tmp_path, parser):
    rundir = _imported(tmp_path)
    assert _main("edit", rundir, "--setting", "initialization.zc_m=1500",
                 "--setting", "initialization.bubble_center_x_m=5000",
                 "--setting", "initialization.bubble_center_y_m=-5000") == 0
    settings = tomllib.loads((rundir / "settings.toml").read_text())
    assert settings["initialization"] == {"zc_m": 1500.0, "bubble_center_x_m": 5000.0,
                                          "bubble_center_y_m": -5000.0}
    init = _deck(rundir)["initialization"]
    assert (init["zc_m"], init["bubble_center_x_m"], init["bubble_center_y_m"]) == (1500.0, 5000.0, -5000.0)
    # a plain re-import of the run keeps them
    assert _main("import", rundir / "namelist.input", "--out", rundir) == 0
    assert _deck(rundir)["initialization"]["zc_m"] == 1500.0
    report = json.loads((rundir / "import-report.json").read_text())
    bubble = next(g for g in report["setup"]["fortran"] if g["id"] == "bubble")
    zc = next(f for f in bubble["fields"] if f["key"] == "initialization.zc_m")
    assert (zc["value"], zc["cm1_value"], zc["changed"], zc["unit"]) == (1.5, 1.4, True, "km")
    # the pair resets together
    assert _main("edit", rundir, "--reset-setting", "initialization.bubble_center_x_m") == 0
    init = _deck(rundir)["initialization"]
    assert "bubble_center_x_m" not in init and "bubble_center_y_m" not in init and init["zc_m"] == 1500.0


def test_a_bubble_outside_the_domain_is_refused_by_the_run_checks(tmp_path, parser, capsys):
    rundir = _imported(tmp_path)
    capsys.readouterr()
    assert _main("edit", rundir, "--setting", "initialization.bubble_center_x_m=80000",
                 "--setting", "initialization.bubble_center_y_m=0", "--json") == 2
    payload = json.loads(capsys.readouterr().out)
    refused = payload["sections"][0]["rows"]
    assert [row["code"] for row in refused] == ["CM1-CFG-BUBBLE-CENTER-OUTSIDE"]
    assert not (rundir / "deck.toml").exists()


def test_the_acknowledgments_are_kept_unless_dropped(tmp_path, parser):
    # An r22.0 output word GM1 does not read (the "Not read" rows): runs once acknowledged.
    source = edited(R22_SUPERCELL, tmp_path, {"brkup_db": "300.0e-6"})
    rundir = tmp_path / "run"
    assert _main("import", source, "--out", rundir) == 3
    assert _main("edit", rundir, "--acknowledge-output-keys") == 0
    assert _main("edit", rundir, "--set", "tapfrq=600") == 0
    report = json.loads((rundir / "import-report.json").read_text())
    assert report["acknowledgments"]["given"] == ["output-keys"]
    assert _main("edit", rundir, "--drop-acknowledgments") == 3


def test_a_run_that_has_run_is_not_edited(tmp_path, parser, capsys):
    rundir = _imported(tmp_path)
    (rundir / "segments").mkdir()
    before = (rundir / "namelist.input").read_bytes()
    assert _main("edit", rundir, "--set", "tapfrq=600") == 2
    assert "fixed once it has run" in capsys.readouterr().err
    assert (rundir / "namelist.input").read_bytes() == before


def test_from_replaces_the_namelist_and_keeps_the_original(tmp_path, parser):
    rundir = _imported(tmp_path)
    edited = tmp_path / "edited.input"
    edited.write_text((rundir / "namelist.input").read_text().replace(" timax  =  7200.0,", " timax  =  3600.0,"))
    assert _main("edit", rundir, "--from", edited) == 0
    assert " timax  =  3600.0," in (rundir / "namelist.input").read_text()
    assert (rundir / "namelist.input.original").read_bytes() == SUPERCELL.read_bytes()
    assert _deck(rundir)["run"]["duration_s"] == 3600.0


def test_the_setup_block_follows_the_namelists_iinit(tmp_path, parser):
    supercell = json.loads((_imported(tmp_path) / "import-report.json").read_text())["setup"]
    squall = json.loads((_imported(tmp_path, SQUALL_LINE, "squall") / "import-report.json").read_text())["setup"]
    active = lambda setup: [g["id"] for g in setup["fortran"] if g["active"]]  # noqa: E731
    assert active(supercell) == ["bubble"] and active(squall) == ["line"]
    cold = next(g for g in supercell["fortran"] if g["id"] == "coldpool")
    amplitude = next(f for f in cold["fields"] if f["key"] == "initialization.bptpert_k")
    assert amplitude["value"] == -6.0  # init3d.F's own cold-pool arm, not the bubble's +1 K
    assert supercell["values"]["iorigin"] == 2 and squall["values"]["iorigin"] == 1
    centre = next(f for f in next(g for g in squall["fortran"] if g["id"] == "bubble")["fields"]
                  if f["key"] == "initialization.bubble_center_x_m")
    assert centre["cm1_value"] == 150.0  # iorigin = 1: the domain centre is nx*dx/2
    words = {w["word"]: w for w in supercell["output_words"]}
    assert len(words) == 62 and "output_format" not in words
    assert words["output_km"]["state"] == "refused" and words["output_rain"]["state"] == "written"
    # Morrison keeps CM1's rain and hail swaths: qr and qg at CM1's nqr = 3 and nqg = 6
    assert words["output_srs"]["state"] == "written" and words["output_thpert"]["state"] == "always"


def test_the_storm_track_recommends_a_frame_and_bubble_that_keep_themselves(tmp_path, parser, monkeypatch, capsys):
    gpuinfo = ROOT / "tests/data/gpuinfo"
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", os.fspath(gpuinfo / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", os.fspath(gpuinfo / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_CACHE_DIR", os.fspath(tmp_path / "cache"))
    rundir = _imported(tmp_path)
    capsys.readouterr()
    assert _main("estimate", rundir, "--json") == 0
    track = json.loads(capsys.readouterr().out)["storm_track"]
    pair = track["recommended_pair"]
    assert pair["frame_m_s"] == track["recommended_frame_m_s"] and pair["uses_it"] is False
    assert all(value * 8 == int(value * 8) for value in pair["frame_m_s"])  # binary32 words, 1/8 m/s
    assert track["hodograph_m_s"][0] == [0.0, 0.0, 0.0] and len(track["hodograph_m_s"]) >= 20
    (u, v), (x, y) = pair["frame_m_s"], pair["bubble_km"]
    assert _main("edit", rundir, "--set", f"umove={u}", "--set", f"vmove={v}",
                 "--setting", f"initialization.bubble_center_x_m={x * 1000.0}",
                 "--setting", f"initialization.bubble_center_y_m={y * 1000.0}") == 0
    capsys.readouterr()
    assert _main("estimate", rundir, "--json") == 0
    again = json.loads(capsys.readouterr().out)["storm_track"]
    assert again["recommended_pair"]["uses_it"] is True
    assert again["recommended_pair"]["frame_m_s"] == [u, v] and again["recommended_pair"]["bubble_km"] == [x, y]
    assert again["minimum_edge_clearance_km"] == pytest.approx(pair["minimum_edge_clearance_km"], abs=0.05)
    assert again["minimum_edge_clearance_km"] > track["minimum_edge_clearance_km"]


def test_usersettings_type_and_range_every_value():
    from cm1core.cm1import import usersettings as U

    assert U.check({"initialization.zc_m": "1500", "initialization.maintain_rh": "on",
                    "lagrangian.seed_nx": 30.0}) == {
        "initialization.zc_m": 1500.0, "initialization.maintain_rh": True, "lagrangian.seed_nx": 30}
    for bad in ({"initialization.zc_m": "x"}, {"lagrangian.seed_nx": 2.5}, {"initialization.bhrad_m": 0},
                {"initialization.zc_m": float("nan")}, {"output.also_netcdf": "maybe"}, {"nope.key": 1},
                {"initialization.bubble_center_y_m": 0.0}):
        with pytest.raises(U.SettingError):
            U.check(bad)
    merged = U.merge({"initialization.bubble_center_x_m": 1.0, "initialization.bubble_center_y_m": 2.0,
                      "initialization.zc_m": 1500.0}, {}, ["initialization.bubble_center_y_m"])
    assert merged == {"initialization.zc_m": 1500.0}
    assert set(U.deck_overrides({"initialization.zc_m": 1.0, "output.also_netcdf": True})) == {"initialization.zc_m"}


def test_usersettings_read_only_the_listed_keys(tmp_path):
    from cm1core.cm1import import usersettings as U

    path = tmp_path / "settings.toml"
    path.write_text('schema = "x"\n[initialization]\nrandom_seed = 7\nzc_m = 1500.0\nbubble_center_x_m = 1.0\n'
                    '[runtime]\ntile_nx = 128\n[output]\nalso_netcdf = false\nformat = 1\n')
    # the seed is the importer's own, the runtime table is chosen afresh, a half pair is dropped
    assert U.read_file(path) == {"initialization.zc_m": 1500.0}
    path.write_text("not toml [")
    assert U.read_file(path) == {}


def test_also_write_netcdf_is_the_cm1out_sidecars_in_the_job_script(tmp_path, parser, capsys):
    import subprocess

    rundir = _imported(tmp_path)
    capsys.readouterr()
    assert _main("jobscript", rundir, "--scheduler", "slurm") == 0
    assert "--format nc" not in capsys.readouterr().out
    assert _main("edit", rundir, "--setting", "output.also_netcdf=true") == 0
    capsys.readouterr()
    assert _main("jobscript", rundir, "--scheduler", "pbs") == 0
    script = capsys.readouterr().out
    # the one cm1out sidecar writes both formats (settings.toml); no second writer races it
    assert "--format nc" not in script and "NETCDF_PID" not in script
    assert "writes netCDF beside the GrADS files" in script and script.count('cm1out "$RUNDIR" --follow') == 1
    path = tmp_path / "job.sh"
    path.write_text(script)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
