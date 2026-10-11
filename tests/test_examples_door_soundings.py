"""`gpuwm-cm1 examples` says what each sounding deck needs, and ships what it can."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from cm1core import doors
from cm1core.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SOUNDINGS = ROOT / "src/cm1core/data/soundings"


def test_every_shipped_sounding_matches_its_recorded_identity() -> None:
    index = json.loads((SOUNDINGS / "index.json").read_text(encoding="utf-8"))
    assert index["soundings"]
    for row in index["soundings"]:
        payload = (SOUNDINGS / row["filename"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert len(payload) == row["bytes"]
        assert "17 U.S.C. 105" in row["rights"]
    provenance = (SOUNDINGS / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "/home" not in provenance


def test_each_sounding_deck_is_either_shipped_or_says_how_to_get_it() -> None:
    shipped = {row["sha256"] for row in doors._shipped_soundings().values()}
    for deck in sorted(EXAMPLES.glob("*.toml")):
        row = doors._example_summary(deck)
        if not row.get("needs_sounding"):
            continue
        sounding = row["sounding"]
        if sounding["shipped"]:
            assert sounding["sha256"] in shipped
            assert row["quickstart_ready"] is True
        else:
            assert row["quickstart_ready"] is False
            assert sounding["how"].startswith("needs sounding:")
            assert "--sounding" in sounding["how"]


def test_copy_writes_the_matching_sounding_beside_its_deck(tmp_path, monkeypatch, capsys) -> None:
    installed = tmp_path / "installed"
    installed.mkdir()
    for name in (
        "cartesian-supercell-moore-20130520-18z-16gib.toml",
        "cartesian-uniform-grid.toml",
    ):
        shutil.copy(EXAMPLES / name, installed / name)

    real_files = doors.resources.files

    def files(package):
        if package == "cm1core":
            return type("Root", (), {"__truediv__": lambda self, other: installed})()
        return real_files(package)

    monkeypatch.setattr(doors.resources, "files", files)
    out = tmp_path / "decks"
    code = main([
        "examples", "cartesian-supercell-moore-20130520-18z-16gib",
        "cartesian-uniform-grid", "--copy", str(out),
    ])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    copied = out / "cartesian-supercell-moore-20130520-18z-16gib.input_sounding"
    assert report["soundings"] == {
        "cartesian-supercell-moore-20130520-18z-16gib.toml": copied.name
    }
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == (
        "aa7d5afa3701ebf14f46221e532c144b01a94c4bc36baad0275db0c3da79e83d"
    )
    assert not (out / "cartesian-uniform-grid.input_sounding").exists()
    # Never overwrite: a second copy into the same directory refuses.
    with pytest.raises(FileExistsError):
        doors.examples_door(type("A", (), {
            "names": ["cartesian-supercell-moore-20130520-18z-16gib"], "copy": str(out),
        })())
