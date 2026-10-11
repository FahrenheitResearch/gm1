"""A forcing initiation centred outside the domain runs as CM1 runs it, and says so plainly.

CM1 r21.1 applies iinit = 12's nudging wherever ``beta < 1`` (misclibs.F:575), iinit = 9's convergence wherever
``|umo| > 0.01`` below ``zdeep`` (misclibs.F:486-487, 527-528) and iinit = 10's u forcing inside its box
(solve1.F:1256), whatever the centre.  A centre outside the domain is therefore not refused (the run stays CM1's),
but the coverage report, the Check screen, the receipt (``bubble_center.forcing_reach``) and the run's own events
say "the push is centred outside the domain, so no storm will be started" when the push touches no grid point.
Host arithmetic only; nothing here opens a device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cm1core.cartesian_door import bubble_center_m, bubble_center_record, initiation_forcing_reach
from cm1core.config import load_config
from cm1core.initiation_forcing import FORCING_REACH_WARNING_CODE, resolve_initiation_forcing

from test_initialization_selectors import toml_document

PHRASE = "The push is centred outside the domain, so no storm will be started"
ROOT = Path(__file__).resolve().parents[1]


def config_for(tmp_path: Path, words: str):
    path = tmp_path / "gpuwm-cm1.toml"
    path.write_text(toml_document(f"[initialization]\nisnd = 5\niwnd = 2\n{words}"), encoding="utf-8")
    return load_config(path)


# The test grid is 16 x 16 km at 1 km, iorigin = 1: the domain is x, y in [0, 16] km.
@pytest.mark.parametrize(("words", "points_word"), [
    ("iinit = 12\nxc_wnudge = 40000.0\nyc_wnudge = 8000.0", "w points"),
    ("iinit = 12\nxc_wnudge = 8000.0\nyc_wnudge = -30000.0", "w points"),
    ("iinit = 9\nxcent = 100000.0\nycent = 8000.0", "u and v points"),
    ("iinit = 10\nxc_uforce = 60000.0", "u points"),
])
def test_a_push_centred_well_outside_warns_that_no_storm_will_start(tmp_path: Path, words: str,
                                                                    points_word: str) -> None:
    config = config_for(tmp_path, words)
    reach = initiation_forcing_reach(config)
    assert reach is not None
    assert reach["centre_inside_domain"] is False and reach["forced_points"] == 0
    assert reach["points_are"] == points_word
    assert reach["code"] == FORCING_REACH_WARNING_CODE
    assert reach["warning"].startswith(PHRASE + ": CM1's ")
    assert "CM1 runs this the same way" in reach["warning"]
    assert "tornado" not in reach["warning"].lower()


def test_a_push_whose_edge_reaches_in_says_so(tmp_path: Path) -> None:
    # 4 km beyond the east edge with CM1's 10 km radius: the ellipsoid's west part is inside.
    reach = initiation_forcing_reach(config_for(tmp_path, "iinit = 12\nxc_wnudge = 20000.0\nyc_wnudge = 8000.0"))
    assert reach["centre_inside_domain"] is False and reach["forced_points"] > 0
    assert reach["warning"].startswith(
        f"The push is centred outside the domain, so only its edge reaches in ({reach['forced_points']} w points) "
        "and a storm may not start")


def test_a_push_inside_the_domain_carries_no_warning(tmp_path: Path) -> None:
    for words in ("iinit = 12", "iinit = 9\nxcent = 8000.0\nycent = 8000.0", "iinit = 10"):
        reach = initiation_forcing_reach(config_for(tmp_path, words))
        assert reach["centre_inside_domain"] is True and reach["forced_points"] > 0, words
        assert reach["warning"] is None and reach["code"] is None, words


def test_a_push_inside_that_touches_nothing_warns_too(tmp_path: Path) -> None:
    # Centred inside, but 30 km up in a 6 km deep domain: no w face is inside the ellipsoid.
    reach = initiation_forcing_reach(config_for(tmp_path, "iinit = 12\nzc_wnudge = 30000.0"))
    assert reach["centre_inside_domain"] is True and reach["forced_points"] == 0
    assert reach["warning"].startswith("The push reaches no grid point, so no storm will be started")


def test_no_forcing_arm_no_reach(tmp_path: Path) -> None:
    config = config_for(tmp_path, "iinit = 1")
    assert initiation_forcing_reach(config) is None
    assert "forcing_reach" not in bubble_center_record(config, bubble_center_m(config))


def test_the_run_is_cm1s_the_warning_is_only_words(tmp_path: Path) -> None:
    """Admitted, with CM1's words, and the configuration record (its sha256 seals every frame) carries no warning."""

    outside = config_for(tmp_path, "iinit = 12\nxc_wnudge = 40000.0\nyc_wnudge = 8000.0")
    forcing = resolve_initiation_forcing(outside.initialization, nx=16, ny=16, dx_m=1000.0, dy_m=1000.0)
    assert forcing["xc_wnudge"] == 40000.0 and forcing["yc_wnudge"] == 8000.0
    assert "warning" not in json.dumps(forcing.record())
    record = bubble_center_record(outside, bubble_center_m(outside))
    assert record["bubble_center_x_m"] == 40000.0
    assert record["forcing_reach"]["warning"].startswith(PHRASE)


def test_the_coverage_report_and_its_text_warn(tmp_path: Path) -> None:
    from cm1import_support import SUPERCELL, edited, namelist_tool
    from cm1core.cm1import.coverage import build_report, import_for_report

    deck = edited(SUPERCELL, tmp_path, {"iinit": "12"})  # CM1's stock deck: 120 km, iorigin = 2
    imported = import_for_report(deck, tool=namelist_tool())
    inside = build_report(imported)
    assert inside.warnings == [] and "warnings" not in inside.as_dict()
    outside = build_report(import_for_report(deck, tool=namelist_tool()),
                           overrides={"initialization.xc_wnudge": 200000.0, "initialization.yc_wnudge": 0.0})
    assert outside.exit_code == 0  # runs, as CM1 does
    payload = json.loads(json.dumps(outside.as_dict(), default=str))
    (warning,) = payload["warnings"]
    assert warning["code"] == FORCING_REACH_WARNING_CODE and warning["forced_points"] == 0
    assert warning["text"].startswith(PHRASE)
    assert "x = 200.0 km" in warning["text"] and "x -60.0 km to 60.0 km" in warning["text"]
    assert "Warning: " + PHRASE in outside.text().replace("\n  ", " ")


def test_the_check_screen_and_the_run_view_show_it() -> None:
    from cm1core.gui.runstate import warning_words

    check = (ROOT / "src/cm1core/gui/static/js/setup/check.js").read_text(encoding="utf-8")
    assert "report.warnings || []" in check and 'class: "banner warn"' in check
    # The run's own event: emitted at admission by the Cartesian door, drawn by the run view as its message.
    door = (ROOT / "src/cm1core/cartesian_door.py").read_text(encoding="utf-8")
    assert 'reach_events.emit("warning", code=reach["code"], message=reach["warning"])' in door
    text = PHRASE + ": ..."
    assert warning_words({"code": FORCING_REACH_WARNING_CODE, "message": text}) == text
