"""The Make-your-own-environment generator (cm1import/analytic_sounding.py).

- the file is CM1's isnd = 7 layout, deterministic, and CM1's own reader takes it (cm1import.sounding);
- WK82's temperature and moisture, as CM1 base.F isnd = 5 builds them: at CM1's cap (14 g/kg) the file's getcape
  CAPE equals CM1's own isnd = 5 base state's on the same grid;
- across a 5 x 5 grid of settings, getcape CAPE and 0-6 km shear on the 1 km preset grid are within 10 % of what
  the settings state (needs the installed initializer library, as the Sounding screen does);
- each regime's sentence (Weisman and Klemp 1982, 1984);
- the written file imports cleanly through the your-environment preset (`preset fit your-environment
  --instability I --shear S`), and settings outside 0-1 are refused in words.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cm1core.cm1import import analytic_sounding as A
from cm1core.cm1import.sounding import check_sounding
from cm1core.cm1run import presets as P

ROOT = Path(__file__).resolve().parents[1]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _library():
    from cm1core.cartesian_door import resolve_initializer_library

    try:
        return resolve_initializer_library(None)
    except Exception:  # noqa: BLE001 - no wheel natives on this host
        return None


needs_library = pytest.mark.skipif(_library() is None, reason="the initializer library (getcape) is not installed")


def test_the_file_is_cm1s_isnd7_layout_and_deterministic(tmp_path):
    made = A.generate(0.5, 0.5)
    again = A.generate(0.5, 0.5)
    assert made.text == again.text and made.sha256 == hashlib.sha256(made.text.encode("ascii")).hexdigest()
    lines = made.text.splitlines()
    surface = [float(x) for x in lines[0].split()]
    assert len(surface) == 3 and surface[0] == 1000.0 and surface[1] == 300.0
    levels = np.array([[float(x) for x in line.split()] for line in lines[1:]])
    assert levels.shape[1] == 5 and np.all(np.diff(levels[:, 0]) > 0)
    assert levels[0, 0] == A.LEVEL_STEP_M and levels[-1, 0] == A.TOP_M
    path = A.write(made, tmp_path / "input_sounding")
    checked = check_sounding(path, {"isnd": 7, "nz": 40, "dz": 500.0})
    assert checked.problems == (), [problem.as_dict() for problem in checked.problems]


def test_temperature_and_moisture_are_weisman_and_klemp_as_cm1_writes_them():
    z = np.array([0.0, 6000.0, 12000.0, 15000.0])
    theta = A.wk_theta(z)
    assert theta[0] == 300.0 and abs(theta[2] - 343.0) < 1e-9
    assert abs(theta[3] - 343.0 * np.exp(9.81 / (213.0 * 1005.7) * 3000.0)) < 1e-9
    assert A.wk_rh(np.array([0.0, 12000.0, 14000.0])).tolist() == [1.0, 0.25, 0.25]
    # The cap, and nothing else, moves with the instability setting.
    low, high = A.generate(0.0, 0.5), A.generate(1.0, 0.5)
    assert A.cap_for_cape(A.CAPE_RANGE_J_KG[0]) < A.cap_for_cape(A.CAPE_RANGE_J_KG[1])
    theta_low = [float(line.split()[1]) for line in low.text.splitlines()[1:]]
    theta_high = [float(line.split()[1]) for line in high.text.splitlines()[1:]]
    assert theta_low == theta_high


@pytest.mark.parametrize("hodograph", A.HODOGRAPHS)
def test_the_wind_is_calm_at_the_ground_and_has_the_chosen_shear(hodograph):
    z = np.arange(0.0, 8001.0, 100.0)
    u, v = A.wind(z, 25.0, hodograph)
    assert u[0] == 0.0 and v[0] == 0.0
    i6 = int(np.where(z == 6000.0)[0][0])
    assert abs(np.hypot(u[i6], v[i6]) - 25.0) < 1e-9
    assert np.allclose(u[z >= 6000.0], u[i6]) and np.allclose(v[z >= 6000.0], v[i6])
    if hodograph == "straight":
        assert np.all(v == 0.0)
    else:
        # CM1's iwnd = 2 shape: a quarter circle over the lowest 2 km, then straight.
        assert v[int(np.where(z == 2000.0)[0][0])] > 0.0


def test_settings_outside_0_to_1_and_unknown_hodographs_are_refused_in_words():
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(A.EnvironmentError, match="between 0 and 1"):
            A.generate(bad, 0.5)
        with pytest.raises(A.EnvironmentError, match="between 0 and 1"):
            A.generate(0.5, bad)
    with pytest.raises(A.EnvironmentError, match="curved or straight"):
        A.generate(0.5, 0.5, "wiggly")


@pytest.mark.parametrize("cape, shear, kind", [
    (300.0, 30.0, "stable"), (499.0, 5.0, "stable"),
    (1500.0, 5.0, "single"), (3000.0, 9.9, "single"),
    (1500.0, 15.0, "multicell"), (800.0, 30.0, "multicell"),
    (1000.0, 20.0, "supercell"), (3000.0, 40.0, "supercell"),
])
def test_each_regime_has_its_sentence(cape, shear, kind):
    assert A.regime(cape, shear) == kind
    made = A.generate(0.5, 0.5)
    words = A.describe(made, {"cape_j_kg": cape, "shear_0_6km_m_s": shear})
    assert words["regime"] == kind and words["numbers_basis"] == "measured"
    assert words["sentence"].startswith(A.SENTENCES[kind])
    assert (A.HODOGRAPH_SENTENCE in words["sentence"]) is (kind == "supercell")
    from cm1core.gui import copy_lint

    assert copy_lint.lint_text(words["sentence"]) == []


def test_without_numbers_the_sentence_says_it_is_from_the_targets():
    words = A.describe(A.generate(0.0, 0.0))
    assert words["numbers_basis"] == "target" and words["regime"] == "stable"


@needs_library
@pytest.mark.parametrize("instability", GRID)
@pytest.mark.parametrize("shear", GRID)
def test_getcape_numbers_land_within_10_percent_of_the_settings(instability, shear):
    made = A.generate(instability, shear)
    numbers = A.report(made.text)
    cape_target, shear_target = A.cape_target(instability), A.shear_target(shear)
    assert abs(numbers["cape_j_kg"] - cape_target) <= 0.10 * cape_target, (numbers["cape_j_kg"], cape_target)
    assert abs(numbers["shear_0_6km_m_s"] - shear_target) <= 0.10 * shear_target, (numbers["shear_0_6km_m_s"],
                                                                                 shear_target)


@needs_library
def test_at_cm1s_own_cap_the_file_is_cm1s_isnd5_environment():
    """The file at 14 g/kg (CM1's qv_pbl) has CM1's own isnd = 5 base state's getcape CAPE on the 1 km grid."""

    from cm1core.sounding_report import initial_sounding_report

    cm1 = initial_sounding_report(_library(), b"", isnd=5, iwnd=2, nz=40, dz_m=500.0)
    cap = A.QV_PBL_CM1 * 1000.0
    cape_at_cap = float(np.interp(cap, [row[0] for row in A.CAPE_TABLE], [row[1] for row in A.CAPE_TABLE]))
    assert abs(cape_at_cap - cm1["surface_based"]["cape_j_per_kg"]) < 5.0


@needs_library
def test_the_cape_table_is_what_the_calibration_measures():
    completed = subprocess.run([sys.executable, str(ROOT / "tools/presets/calibrate_environment.py"), "--check"],
                               capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@needs_library
def test_the_command_writes_the_file_and_the_preset_runs_it(tmp_path, capsys):
    from cm1core.cli import main

    out = tmp_path / "input_sounding"
    assert main(["preset", "environment", "--instability", "0.25", "--shear", "0.75", "--out", str(out),
                 "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["written"] == str(out.resolve()) and out.read_text() == A.generate(0.25, 0.75).text
    assert record["command"].startswith("gpuwm-cm1 preset environment --instability 0.25 --shear 0.75")
    assert record["numbers_basis"] == "measured" and record["sentence"]
    # never over a file
    with pytest.raises(P.PresetError, match="already exists"):
        P.environment_record(0.25, 0.75, out=out)
    preset = P.with_environment(P.load("your-environment"), 0.25, 0.75)
    examined = P.examine(preset)
    assert examined["offered"], examined["why_not"]
    assert preset.sounding_given["environment"]["sentence"] == record["sentence"]
    assert P.given_args(preset) == ["--instability", "0.25", "--shear", "0.75", "--hodograph", "curved"]
