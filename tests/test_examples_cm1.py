"""`gpuwm-cm1 examples --cm1`: CM1's own shipped namelists, listed and copied unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPERCELL_SHA256 = "3854f731efe6a6a9d56d6aa3db198434ccaba85bc5584b5e8ce1c5edfd3b56a4"


def _gm1(*words: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "cm1core", *words], capture_output=True, text=True, cwd=cwd,
                          env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}, timeout=120)


def test_lists_the_two_cm1_namelists(tmp_path: Path) -> None:
    listed = _gm1("examples", "--cm1", cwd=tmp_path)
    assert listed.returncode == 0, listed.stderr
    rows = {row["case"]: row for row in json.loads(listed.stdout)["cm1_namelists"]}
    assert set(rows) == {"supercell", "squall_line"}
    assert rows["supercell"]["sha256"] == SUPERCELL_SHA256
    assert rows["supercell"]["source"] == "cm1r21.1 run/config_files/supercell/namelist.input, unchanged"


def test_copies_one_unchanged_and_never_overwrites(tmp_path: Path) -> None:
    copied = _gm1("examples", "--cm1", "supercell", "--copy", "cm1", cwd=tmp_path)
    assert copied.returncode == 0, copied.stderr
    target = tmp_path / "cm1" / "supercell" / "namelist.input"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == SUPERCELL_SHA256
    assert not (tmp_path / "cm1" / "squall_line").exists()
    again = _gm1("examples", "--cm1", "supercell", "--copy", "cm1", cwd=tmp_path)
    assert again.returncode != 0
    assert "never overwrite" in again.stderr


@pytest.mark.parametrize("name", ["supercel", "cartesian-uniform-grid"])
def test_an_unknown_case_is_named(tmp_path: Path, name: str) -> None:
    refused = _gm1("examples", "--cm1", name, cwd=tmp_path)
    assert refused.returncode != 0
    assert name in refused.stderr
