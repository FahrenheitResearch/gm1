"""CM1's analytic soundings, the moist benchmark (iinit = 4) and ibalance = 2, config to initial state.

Correctness against CM1 is measured by tools/oracle/initialization_differential.py (its analytic-sounding rows) against the real
binary, and word for word per arm by ``analytic_soundings::every_arm_is_cm1s_column_word_for_word`` in the crate.
These tests keep the doors open: each value reaches the Rust initializer, the Settings screen and the importer.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from cm1core.errors import CM1Refusal
from cm1core.gm1_idealized_residency import ADMITTED_ISND, IdealizedSource, InitializerSelection
from cm1core.grid_geometry import CartesianGridGeometry

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust/cm1-state"


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("gm1-analytic-cdylib")
    subprocess.run(["cargo", "build", "--release", "--lib", "--target-dir", str(target)], cwd=CRATE, check=True)
    names = ("cm1_state.dll", "libcm1_state.so", "libcm1_state.dylib")
    found = [target / "release" / n for n in names if (target / "release" / n).is_file()]
    assert len(found) == 1
    return found[0]


def geometry(nz: int = 20, dz: float = 500.0) -> CartesianGridGeometry:
    return CartesianGridGeometry(nx=16, ny=16, nz=nz, dx_m=1000.0, dy_m=1000.0, dz_m=dz)


def build(library: Path, selection: InitializerSelection, names: tuple[str, ...], *, dz: float = 500.0,
          nz: int = 20) -> dict[str, np.ndarray]:
    with IdealizedSource(library, b"", geometry(nz, dz), selection=selection) as source:
        return {name: np.array(source.field(name, layout="physical"), dtype="<f4", copy=True) for name in names}


def test_every_cm1_sounding_but_the_removed_and_the_constants_one_is_admitted() -> None:
    assert ADMITTED_ISND == frozenset({1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23})
    for isnd in (6, 15, 16):
        with pytest.raises(ValueError):
            InitializerSelection(isnd=isnd, iwnd=2)


def test_the_moist_benchmark_carries_its_base_cloud_and_builds_its_bubble(library: Path) -> None:
    selection = InitializerSelection(isnd=4, iwnd=0, iinit=4)
    fields = build(library, selection, ("base_condensate.qc0", "base_condensate.qi0", "terrain_open.qv0",
                                        "terrain_open.th3d", "moist.q3d"))
    qc0, qv0 = fields["base_condensate.qc0"], fields["terrain_open.qv0"]
    # The dynamics uploads qc0/qi0 as physical volumes and as profiles (cuda_cartesian_dynamics).
    with IdealizedSource(library, b"", geometry(), selection=selection) as source:
        profile = np.asarray(source.field("base_condensate.qc0", layout="profile"))
        assert np.array_equal(profile, qc0[:, 0, 0])
    # base.F:347: qc0 = qt_mb - qv0, CM1's qt_mb = 0.020.
    assert np.array_equal(qc0, np.float32(0.020) - qv0)
    assert (qc0 > 0).all() and not fields["base_condensate.qi0"].any()
    # init3d.F:579-647: a bubble in tha, and saturated vapour and cloud everywhere.
    assert fields["terrain_open.th3d"].max() > 0.5
    q3d = fields["moist.q3d"]
    assert (q3d[1] > 0).all()


def test_a_sounding_without_base_cloud_publishes_exact_zeros(library: Path) -> None:
    fields = build(library, InitializerSelection(isnd=5, iwnd=2), ("base_condensate.qc0", "base_condensate.qi0"))
    assert not fields["base_condensate.qc0"].any() and not fields["base_condensate.qi0"].any()


def test_isnd_10_partitions_its_cloud_to_ice_as_cm1_with_these_schemes(library: Path) -> None:
    fields = build(library, InitializerSelection(isnd=10, iwnd=2), ("base_condensate.qc0", "base_condensate.qi0"))
    qi0 = fields["base_condensate.qi0"]
    assert np.allclose(qi0[1:], 0.005) and not fields["base_condensate.qc0"][1:].any()


def test_the_anelastic_balance_gives_the_bubble_a_pressure(library: Path) -> None:
    zero = build(library, InitializerSelection(isnd=5, iwnd=2, iinit=1, ibalance=0), ("terrain_open.pp3d",))
    anelastic = build(library, InitializerSelection(isnd=5, iwnd=2, iinit=1, ibalance=2), ("terrain_open.pp3d",))
    assert not zero["terrain_open.pp3d"].any()
    pp = anelastic["terrain_open.pp3d"]
    assert np.isfinite(pp).all() and np.abs(pp).max() > 1e-6


def test_the_settings_screen_shows_the_benchmark_words() -> None:
    from cm1core.cm1import.setup import fortran_groups

    effective = {"iinit": 4, "isnd": 4, "nx": 100, "ny": 16, "dx": 200.0, "dy": 200.0, "iorigin": 2}
    groups = {g["id"]: g for g in fortran_groups(effective, {}, {})}
    assert groups["benchmark"]["active"] and groups["benchmark_sounding"]["active"]
    assert not groups["bubble"]["active"]
    bubble = {f["key"]: f["value"] for f in groups["benchmark"]["fields"]}
    # init3d.F:585-589, in km and K as the screen shows them.
    assert bubble == {"initialization.bubble_center_x_m": 0.0, "initialization.zc_m": 2.0,
                      "initialization.bhrad_m": 2.0, "initialization.bvrad_m": 2.0,
                      "initialization.bptpert_k": 2.0}
    sounding = {f["key"]: f["value"] for f in groups["benchmark_sounding"]["fields"]}
    assert sounding == {"initialization.thec_mb_k": 320.0, "initialization.qt_mb": 0.02}


def test_refusal_texts_name_what_is_still_refused() -> None:
    from cm1core.gui.refusal_text import describe

    dry = describe(code="CM1-CFG-ISND-DRY", keys=("isnd",), value=2, context={}, category="refused")
    assert "imoist = 0" in dry.cm1 and "param.F:1501-1515" in dry.cm1
    removed = describe(code="CM1-CFG-UNSUPPORTED-VALUE", keys=("isnd",), value=6, context={}, category="refused")
    assert "code has been removed" in removed.cm1
    constants = describe(code="CM1-CFG-UNSUPPORTED-VALUE", keys=("testcase",), value=5, context={},
                         category="refused")
    assert "constants.F:78-98" in constants.cm1


#: The new initial states and whether a run from them is admitted: isnd = 9's and ibalance = 2's integration matched
#: CM1's over 15 minutes (like-for-like, records/CM1-INITIALIZATION-VALIDATION.json); the base-state cloud soundings',
#: isnd = 11's and the moist benchmark's did not, and the other analytic soundings' has not been compared, so they
#: import for run.iconly = 1 only.
NEW_STATES = (
    ({"isnd": "4", "iinit": "4", "iwnd": "0"}, "CM1-CFG-BASE-CLOUD-INTEGRATION"),
    ({"isnd": "10"}, "CM1-CFG-BASE-CLOUD-INTEGRATION"),
    ({"isnd": "11", "tsk0": "300.0"}, "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION"),
    ({"isnd": "1"}, "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION"),
    ({"isnd": "12"}, "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION"),
    ({"isnd": "19"}, "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION"),
    ({"isnd": "22"}, "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION"),
    ({"iinit": "4"}, "CM1-CFG-MOIST-BENCHMARK-INTEGRATION"),
    ({"isnd": "9"}, None),
    ({"ibalance": "2"}, None),
)


@pytest.mark.parametrize(("words", "refusal"), NEW_STATES)
def test_the_importer_admits_the_new_initial_states(tmp_path: Path, words: dict[str, str],
                                                    refusal: str | None) -> None:
    from cm1import_support import SUPERCELL, edited, namelist_tool

    from cm1core.cm1import.coverage import build_report, import_for_report

    deck = edited(SUPERCELL, tmp_path, words)
    report = build_report(import_for_report(deck, tool=namelist_tool()))
    rows = [row for row in report.category("refused") if set(row.keys) & {"isnd", "iinit", "ibalance", "tsk0"}]
    if refusal is None:
        assert not rows, [row.text for row in rows]
        assert report.exit_code != 2, [row.text for row in report.category("refused")]
        return
    # A run is refused with the reason in plain words; the same namelist with iconly = 1 imports.
    named = [row for row in rows if row.code == refusal]
    said = "not been checked" if refusal == "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION" and words.get("isnd") != "11"         else "did not match"
    assert named and all(said in (row.plain or "") for row in named), [row.text for row in rows]
    # The run check sets the refused isnd aside to go on checking the rest, but the Sounding block and screen
    # describe the namelist's own profile, never the stand-in (which would show Weisman-Klemp's numbers).
    from cm1core.gui.setup import sounding_view

    isnd = int(words.get("isnd", "5"))
    payload = report.as_dict()
    assert payload["sounding"]["isnd"] == isnd
    source = sounding_view(payload, None)["source"]
    assert f"(isnd = {isnd})" in source, source
    if isnd != 5:
        assert "Weisman" not in source and "shown only (iconly = 1)" in source.lower(), source
    (tmp_path / "iconly").mkdir()
    deck = edited(SUPERCELL, tmp_path / "iconly", {**words, "iconly": "1"})
    report = build_report(import_for_report(deck, tool=namelist_tool()))
    rows = [row for row in report.category("refused") if set(row.keys) & {"isnd", "iinit", "ibalance", "tsk0"}]
    assert not rows, [row.text for row in rows]


def test_refusal_texts_say_why_a_run_from_base_cloud_or_the_benchmark_is_refused() -> None:
    from cm1core.gui.refusal_text import describe

    cloud = describe(code="CM1-CFG-BASE-CLOUD-INTEGRATION", keys=("isnd",), value=10, context={},
                     category="refused")
    assert "iconly = 1" in cloud.cm1 and "did not match" in cloud.plain
    bench = describe(code="CM1-CFG-MOIST-BENCHMARK-INTEGRATION", keys=("iinit",), value=4, context={},
                     category="refused")
    assert "iconly = 1" in bench.cm1 and "did not match" in bench.plain
    failed = describe(code="CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION", keys=("isnd",), value=11, context={},
                      category="refused")
    assert "iconly = 1" in failed.cm1 and "did not match" in failed.plain
    for isnd in (1, 8, 12, 13, 14, 18, 19, 20, 21, 22, 23):
        uncompared = describe(code="CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION", keys=("isnd",), value=isnd, context={},
                              category="refused")
        assert "iconly = 1" in uncompared.cm1 and "not been checked" in uncompared.plain
        assert "compared with CM1's yet" in uncompared.cm1
