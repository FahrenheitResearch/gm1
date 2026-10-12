"""The two small host tools docs/NESTING.md leans on: tools/nest_place.py and tools/plan_summary.py.

Neither opens a device. The placement rule is held on made-up frame rows (the frames themselves are read with the
quick-looks' own reader, which has its tests), the box mean against a direct sum, and the plan summary on a plan of
the shape `gpuwm-cm1 cartesian --plan` prints.
"""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLACE = runpy.run_path(str(ROOT / "tools" / "nest_place.py"))
SUMMARY = runpy.run_path(str(ROOT / "tools" / "plan_summary.py"))


def _row(time_s: float, i: int, j: int, *, tight: bool = True, zeta: float = 0.14) -> dict:
    return {"frame": f"frame-{int(time_s):06d}", "time_s": time_s, "tight": tight,
            "box300": {"x_m": round((i + 0.5) * 100.0), "y_m": round((j + 0.5) * 100.0), "zeta_s_inv": zeta,
                       "i": i, "j": j},
            "grid": {"nx": 640, "ny": 768, "dx_m": 100.0, "dy_m": 100.0}}


def test_the_box_mean_is_the_mean_over_the_box() -> None:
    field = np.random.default_rng(7).normal(size=(12, 15)).astype(np.float32)
    mean = PLACE["box_mean"](field, 3)
    assert np.isnan(mean[0]).all() and np.isnan(mean[:, -1]).all()
    for j, i in ((1, 1), (5, 7), (10, 13)):
        assert mean[j, i] == pytest.approx(float(field[j - 1:j + 2, i - 1:i + 2].astype(np.float64).mean()), abs=1e-12)
    assert (PLACE["box_mean"](field, 1) == field).all()
    assert [PLACE["box_cells"](width, 100.0) for width in (100.0, 300.0, 500.0)] == [1, 3, 5]
    assert [PLACE["box_cells"](width, 20.0) for width in (100.0, 300.0, 500.0)] == [5, 15, 25]


def test_a_start_needs_three_tight_frames_and_a_smoothly_moving_maximum() -> None:
    rows = [_row(2160.0, 305, 314), _row(2220.0, 303, 310), _row(2280.0, 300, 306), _row(2340.0, 296, 301)]
    restarts = [(8, 2160.0), (9, 2340.0)]
    chosen = PLACE["propose"](rows, restarts, span=20, speed_max_m_s=11.7)
    assert (chosen["rstnum"], chosen["start_s"], chosen["parent_i0"], chosen["parent_j0"]) == (8, 2160.0, 295, 304)
    assert "warning" not in chosen
    # the middle frame is not tight: no restart has three tight frames around it
    loose = [rows[0], {**rows[1], "tight": False}, rows[2], rows[3]]
    assert PLACE["propose"](loose, restarts, span=20, speed_max_m_s=11.7) is None
    # a maximum that jumps 3 km in a minute is another vortex, not this one moving
    jump = [rows[0], _row(2220.0, 330, 330), rows[2], rows[3]]
    assert PLACE["propose"](jump, [(8, 2160.0)], span=20, speed_max_m_s=11.7) is None
    # without restarts the first frame time that qualifies is proposed, with no number
    free = PLACE["propose"](rows, None, span=60, speed_max_m_s=11.7)
    assert (free["rstnum"], free["start_s"], free["parent_i0"], free["parent_j0"]) == (None, 2160.0, 275, 284)
    # a child that would start at the side of the parent is flagged
    edge = [_row(t, 12, 300) for t in (0.0, 60.0, 120.0)]
    assert "warning" in PLACE["propose"](edge, None, span=20, speed_max_m_s=11.7)


def test_restart_times_are_read_from_their_publication_records(tmp_path: Path) -> None:
    for number, time_s in ((7, 2100.0), (8, 2160.0)):
        folder = tmp_path / f"restart-{number:06d}"
        folder.mkdir()
        (folder / "restart-publication.json").write_text(json.dumps({"model_time_s": time_s}), encoding="utf-8")
    (tmp_path / "restart-000009").mkdir()  # unfinished: no record
    assert PLACE["read_restart_times"](tmp_path) == [(7, 2100.0), (8, 2160.0)]
    assert PLACE["main"]([str(tmp_path / "nothing")]) == 2


def _plan(nest: bool) -> dict:
    gib = 2 ** 30
    plan = {
        "schema": "gpuwm-cm1.cartesian-plan/v1", "status": "priced-no-device-opened",
        "integration_cadence": {"total_steps": 8800, "duration_s": 3300.0},
        "run_schedule": {"start_step": 5760, "start_model_time_s": 2160.0, "history_count": 19, "restart_count": 4},
        "dynamics": {"maximum_listed_phase_payload_bytes": 14 * gib, "geometry": {"nx": 640, "ny": 768, "nz": 80}},
        "run_geometry": {"status": "tracked-member-stays-inside"}, "nest": None,
    }
    if nest:
        plan["nest"] = {"nx": 500, "ny": 500, "dx_m": 4.0, "child_dt_s": 0.03125, "child_steps_per_parent_step": 12,
                        "origin_x_m": 29500.0, "origin_y_m": 30400.0,
                        "storm_following": {"follow_tracker": "wrf-surface-circulation"}}
        plan["nest_allocation"] = {"dynamics": {"maximum_listed_phase_payload_bytes": 9 * gib},
                                   "closure": {"geometry": {"nz": 103}},
                                   "config": {"advective_courant_target_speed_m_s": 128.0}}
    return plan


def test_the_plan_summary_adds_the_child_to_the_parent(tmp_path: Path, capsys) -> None:
    alone = SUMMARY["summary"](_plan(False))
    assert (alone["parent_steps"], alone["frames"], alone["restarts"], alone["listed_gib"]) == (3040, 19, 4, 14.0)
    assert alone["nest"] is None
    nested = SUMMARY["summary"](_plan(True))
    assert nested["listed_gib"] == 23.0 and nested["nest"]["cells"] == [500, 500, 103]
    text = SUMMARY["render"](nested)
    assert "a wind of 128 m/s crosses one child cell per step" in text
    assert "23.00 GiB listed; allow about 4 GiB more on the card (about 27 GiB)" in text
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan(True)), encoding="utf-8")
    assert SUMMARY["main"]([str(path)]) == 0
    assert "child         500 x 500 x 103 cells at 4 m, 9.00 GiB listed" in capsys.readouterr().out
    # a refused plan leaves an empty file: say so, do not trace back
    path.write_text("", encoding="utf-8")
    assert SUMMARY["main"]([str(path)]) == 2
    assert "A refused plan" in capsys.readouterr().err
