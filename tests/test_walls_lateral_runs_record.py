"""The whole-run record of the rigid-wall and lateral-setting fixes, held to what the labels say.

``evidence/oracle/walls-and-lateral-runs`` compares whole runs of CM1 r21.1 with GM1 from release 0.1.7 and from
the fixes. The labels a user sees (the boundary register, the import report) quote its numbers; these tests keep
the two from drifting apart.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evidence" / "oracle" / "walls-and-lateral-runs"


def _comparison(name: str) -> dict:
    return json.loads((RUNS / "comparisons" / f"{name}.json").read_text())


def _worst(record: dict, run: str) -> dict[str, float]:
    worst = record["worst_over_time"][run]
    return {field: max(value for key, value in row.items() if key in ("wall_band", "interior"))
            for field, row in worst.items()}


def test_the_walled_bubbles_follow_cm1_in_a_stable_layer() -> None:
    for case in ("walls3", "walls4"):
        record = _comparison(f"bubble-{case}")
        assert max(_worst(record, "gm1").values()) < 4e-5, case
        # Release 0.1.7 carried the sounding's wind through the walls in the first step.
        assert max(_worst(record, "gm1_before").values()) > 0.5, case
        last = record["table"][str(record["times_s"][-1])]
        assert set(last["gm1"]["wall_faces"].values()) == {0.0}


def test_the_walled_bubbles_stay_near_cm1s_own_spread_in_a_neutral_layer() -> None:
    for case in ("walls3-neutral", "walls4-neutral", "vstretch-walls34-neutral"):
        worst = _comparison(f"bubble-{case}")["worst_over_time"]
        for field, row in worst["gm1"].items():
            twin = worst["cm1_twin"][field]
            # Region by region (the cells beside the walls, the rest), and in root-mean-square.
            for region in ("wall_band", "interior"):
                assert row[region] <= 3.4 * twin[region], (case, field, region)
            assert row["rms_over_cm1_rms"] <= 2.25 * twin["rms_over_cm1_rms"], (case, field)


def test_the_open_controls_are_word_for_word_release_0_1_7() -> None:
    runs = json.loads((RUNS / "runs.json").read_text())["gm1"]["runs"]
    for case in ("bubble-open", "bubble-open-neutral"):
        assert (runs[case]["branch"]["frame_arenas_sha256_digest"]
                == runs[case]["release_0_1_7"]["frame_arenas_sha256_digest"])
    for case, row in runs.items():
        assert row["branch"]["deck_sha256"] == row["release_0_1_7"]["deck_sha256"], case


def test_release_0_1_7_gave_the_stretched_storm_the_same_words_with_either_setting() -> None:
    dropped = json.loads((RUNS / "runs.json").read_text())["release_0_1_7_dropped_the_settings"]
    assert dropped["stretched-irbc2"] == dropped["stretched-roflux1"]


def test_the_stretched_storms_are_inside_the_release_bands() -> None:
    for case, low, high in (("stretched-irbc2", 0.963, 1.010), ("stretched-roflux1", 0.955, 1.011)):
        verdict = _comparison(case)["verdicts"]["gm1"]
        assert verdict["all_inside_band"], case
        assert round(verdict["ratio_min"], 3) == low and round(verdict["ratio_max"], 3) == high, case


def test_the_walled_supercell_is_not_and_the_labels_say_so() -> None:
    from cm1core.boundary_validation_register import register

    verdicts = _comparison("walls-supercell")["verdicts"]
    assert verdicts["cm1_twin"]["all_inside_band"]
    gm1 = verdicts["gm1"]
    assert not gm1["all_inside_band"]
    assert round(gm1["rows"]["uh_max"]["ratio"], 2) == 1.26
    walls = register()["wbc/ebc/sbc/nbc=3,4 (rigid walls)"]
    assert walls["differential"] is None
    assert "1.26 times CM1's" in walls["not_checked"]
    for name in ("irbc=2 (Klemp-Wilhelmson on small steps) with cstar", "roflux=1 (restrict outward mass flux)"):
        assert "evidence/oracle/walls-and-lateral-runs" in register()[name]["differential"], name
