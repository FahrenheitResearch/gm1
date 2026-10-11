"""The GUI's Downscale a WRF or ArWen run (3D), advanced mode, through the server with no browser
and no GPU.

* The three POSTs (open, check, new) each return their one ``gpuwm-cm1 chain`` command, take ``dry_run`` (same
  line, no file changed), and ``chain new`` replays from a shell to the same run folder (the replay walk).
* A box the chain refuses comes back in plain words, each check with its code; nothing is written.
* The run folder ``chain new`` makes is an ordinary run: the fake runner starts it, and its Results are served.
* The verification label is pinned, served with the words, and says what has and has not been checked.
* Guided keeps its four screens: the screen registers in advanced mode only; the words pass the copy lint; the page
  files are linked where the plan says (index.html, app.js) and the stylesheet uses only the base's tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

import pytest

pytest.importorskip("numpy")
pytest.importorskip("netCDF4")

from cm1core import chain_door  # noqa: E402
from cm1core.gui import commands as C  # noqa: E402
from cm1core.gui import copy_lint  # noqa: E402
from gui_support import ROOT, stop_server, wait_for  # noqa: E402
from test_arwen_chain_wrf import START, history as chain_history  # noqa: E402
from test_gui_advanced import Walk, _environment, _mixed, _shim  # noqa: E402,F401
import wrf_support as ws  # noqa: E402

STATIC = ROOT / "src/cm1core/gui/static"
CHAIN_JS = sorted((STATIC / "js/chain").glob("*.js"))
BOX = [5, 5, 5, 5]


@pytest.fixture
def wrf_history(tmp_path):
    """A WRF-like netCDF-4 history of three records 300 s apart (WSM6, d02), under the runs folder."""

    return chain_history(tmp_path / "runs" / "wrf", mp=6, use_theta_m=1, thm="dry")


def test_the_chain_actions_walk_the_api_table_and_replay(tmp_path, monkeypatch, wrf_history):
    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    scratch = tmp_path / "walk"
    scratch.mkdir()
    walk = Walk(server, client, root, _shim(tmp_path, "mixed"), scratch)
    try:
        listed = client.get("/api/chain/files").json
        assert [row["path"] for row in listed["files"]] == [os.fspath(wrf_history)]
        opened = walk.post("chain-open", "/api/chain/open", {"path": os.fspath(wrf_history)}, run=None)
        assert opened["producer"] == "WRF" and opened["count"] == 3 and opened["intervals_s"] == [300.0]
        assert opened["microphysics"]["component"] == "wsm6-mp6" and opened["verified"] == chain_door.VERIFIED
        assert opened["map"]["url"].startswith("/api/chain/pictures/maps/")
        picture = client.get(opened["map"]["url"])
        assert picture.status == 200 and picture.headers["content-type"] == "image/png"
        at = walk.post("chain-open", "/api/chain/open",
                       {"path": os.fspath(wrf_history), "map_time": ws.stamp(START, 300)}, run=None)
        assert at["map"]["time"] == ws.stamp(START, 300) and at["map"]["url"] != opened["map"]["url"]
        request = {"path": os.fspath(wrf_history), "window": BOX, "ratio": 3, "start": START,
                   "end": ws.stamp(START, 600)}
        checked = walk.post("chain-check", "/api/chain/check", request, run=None)
        assert checked["admitted"] is True and checked["rc"] == 0, [r for r in checked["checks"] if r["status"] != "pass"]
        assert checked["message"] == "Every check passes. GM1 can run this box."
        assert {row["id"] for row in checked["checks"]} == set(chain_door.chain.ADMISSION_CHECKS)
        assert all(row["status"] == "pass" and row["text"] for row in checked["checks"])
        assert checked["estimate"]["cells"] == 15 * 15 * checked["choice"]["nz"]
        assert checked["prepare"]["disk_bytes"] > 0
        made = walk.post("chain-new", "/api/chain/new", {**request, "name": "d1"}, run="d1")
        rundir = root / "d1"
        assert made["url"] == "d1" and made["rc"] == 0 and made["verdict"] == "runnable"
        assert made["command"] == C.display(C.chain_new_args([wrf_history], BOX, 3, rundir, start=START,
                                                             end=ws.stamp(START, 600)))
        assert made["start"] == f"gpuwm-cm1 run {rundir}"
        assert (rundir / "chain" / "chain-receipt.json").is_file() and (rundir / "input_sounding").is_file()
        assert "[chain]" in (rundir / "deck.toml").read_text()
        assert made["command"] in (rundir / "commands.log").read_text()
        # The folder is an ordinary run: its Check screen reads it.
        detail = client.get("/api/runs/d1").json
        assert detail["files"]["deck.toml"] and detail["run"]["chain"]["table"]["refinement_ratio"] == 3
    finally:
        stop_server(server)


def test_a_refused_box_comes_back_in_plain_words_and_writes_nothing(tmp_path, monkeypatch):
    """A WSM3 history over 300 m hills on a Lambert map: every refusal named, the rest still checked."""

    path = tmp_path / "runs" / "wrf3" / "wrfout_d01_2026-05-01_18:00:00"
    path.parent.mkdir(parents=True)
    ws.write_wrfout(path, fmt="netCDF-4", producer="WRF", nx=16, ny=16, nz=30, records=3, interval_s=300.0,
                    start=START, mp_physics=3, terrain_m=300.0)
    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    try:
        before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if ".gm1-gui" not in p.parts)
        reply = client.post("/api/chain/check", {"path": os.fspath(path), "window": BOX, "ratio": 3}).json
        assert reply["ok"] is True and reply["admitted"] is False and reply["rc"] == 3
        rows = {row["id"]: row for row in reply["checks"]}
        assert rows["microphysics"]["status"] == "refused" and rows["microphysics"]["code"] == "GM1-CHAIN-MICROPHYSICS"
        assert rows["microphysics"]["text"] == \
            ("Your run used WSM3 (3). GM1 downscales every cloud and rain field only from WSM6 (6), Thompson (8) and "
                 "Morrison (10) runs; its water vapour alone can be taken (Moisture: water vapour only, or "
                 "--moisture vapour-only).")
        assert rows["species"]["status"] == "not-checked"
        assert rows["species"]["text"] == "Not checked until the microphysics check passes."
        assert re.match(r"\d+ check\(s\) do not pass\. Each line says why\.", reply["message"])
        assert "estimate" not in reply
        made = client.post("/api/chain/new", {"path": os.fspath(path), "window": BOX, "ratio": 3, "name": "no"}).json
        assert made["ok"] is False and made["command"].startswith("gpuwm-cm1 chain new ")
        assert "WSM3" in made["message"]
        after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if ".gm1-gui" not in p.parts)
        assert after == before
        # Plain refusals of the request itself, each with the table's command.
        for body, words in (({"window": BOX, "ratio": 4}, "The ratio must be one of 3, 5, 7, 9."),
                            ({"window": [1, 2, 3], "ratio": 3}, "The box is not four whole numbers"),
                            ({"window": BOX, "ratio": 3, "start": "noon"}, "The start must be one of the saved times")):
            got = client.post("/api/chain/check", {"path": os.fspath(path), **body})
            assert got.status == 400 and got.json["message"].startswith(words), got.json
            assert got.json["command_template"].startswith("gpuwm-cm1 chain check ")
        missing = client.post("/api/chain/open", {"path": os.fspath(tmp_path / "nothing")})
        assert missing.status == 400 and "There is nothing at" in missing.json["message"]
    finally:
        stop_server(server)


def test_a_made_run_starts_and_its_results_are_served(tmp_path, monkeypatch, wrf_history):
    """chain new, then Start (the fake GPU), then Results: the run folder is an ordinary run."""

    server, client = _mixed(tmp_path, monkeypatch)
    try:
        made = client.post("/api/chain/new", {"path": os.fspath(wrf_history), "window": BOX, "ratio": 3,
                                              "name": "d2", "end": ws.stamp(START, 300)}).json
        assert made["ok"] is True, made
        started = client.post("/api/runs/d2/start", {}).json
        assert started["ok"] is True and started["command"].startswith("gpuwm-cm1 run ")
        wait_for(lambda: client.get("/api/runs/d2/status").json.get("state") in ("finished", "failed"), timeout=120)
        assert client.get("/api/runs/d2/status").json["state"] == "finished"
        results = client.get("/api/runs/d2/results?mode=advanced").json
        assert results["id"] == "d2" and results["status"]["state"] == "finished"
    finally:
        stop_server(server)


def test_pictures_are_served_from_the_servers_own_folder_only(tmp_path, monkeypatch):
    server, client = _mixed(tmp_path, monkeypatch)
    try:
        for url in ("/api/chain/pictures/maps/x/map.png", "/api/chain/pictures/..%2F..%2Fetc/passwd",
                    "/api/chain/pictures/maps/x/notes.txt"):
            assert client.get(url).status in (400, 404)
    finally:
        stop_server(server)


def test_the_verification_label_is_pinned_and_served_with_the_words(tmp_path, monkeypatch):
    assert chain_door.VERIFIED == (
        "Checked on one GPU (an RTX 3080) with an idealized ArWen storm run, with GM1 driving itself, and with one "
        "real-weather ArWen run (an atmospheric river over the ocean off California, water vapour only). "
        "Not compared with observations."
    )
    server, client = _mixed(tmp_path, monkeypatch)
    try:
        words = client.get("/api/chain/copy").json
        assert words["verified"] == chain_door.VERIFIED and words["ratios"] == [3, 5, 7, 9]
        assert words["screen"]["title"] == "Downscale a WRF or ArWen run (3D)"
    finally:
        stop_server(server)


def test_the_words_pass_the_copy_lint_and_every_check_has_its_words():
    assert copy_lint.lint_file(ROOT / "src/cm1core/gui/copy/chain.json") == []
    for path in CHAIN_JS:
        assert copy_lint.lint_js(path) == [], path.name
    words = json.loads((ROOT / "src/cm1core/gui/copy/chain.json").read_text())
    assert set(words["checks"]) - {"not_checked"} == set(chain_door.chain.ADMISSION_CHECKS)
    assert set(words["labels"]) == set(chain_door.chain.ADMISSION_CHECKS)
    assert chain_door.VERIFIED not in json.dumps(words)       # one source for the label


def test_the_screen_shows_the_worked_real_weather_example_the_record_ran():
    """The example on the screen is the box of GM1-ARWEN-CHAIN.md section 9.2 and the guide's worked example."""

    words = json.loads((ROOT / "src/cm1core/gui/copy/chain.json").read_text())["screen"]
    example = words["example"]
    for fact in ("36 x 66", "cell (6, 40)", "finer by 3", "water vapour only", "01:00 to 03:00 UTC",
                 "108 x 198 GM1 cells", "not a reconstruction"):
        assert fact in example, fact
    # the hint names the record a pip install carries, not a repository path
    assert words["example_hint"].startswith("GM1-ARWEN-CHAIN.md section 9.2") and "docs/" not in words["example_hint"]
    assert "limits_hint" not in words and "docs/" not in words["verified_hint"]
    guide = (ROOT / "docs/WRF-DOWNSCALING.md").read_text()
    assert "## A worked example" in guide and "--window 6,40,36,66" in guide
    record = (ROOT / "src/cm1core/records/GM1-ARWEN-CHAIN.md").read_text()
    assert "108 × 198 × 40 GM1 cells of 1014.79 m" in record
    screen = (STATIC / "js/chain/screen.js").read_text()
    assert "S.example_title" in screen and "S.example_hint" in screen


def test_guided_keeps_its_screens_and_the_files_are_linked_where_the_plan_says():
    index = (STATIC / "index.html").read_text()
    assert index.count('<link rel="stylesheet" href="/static/css/chain.css">') == 1
    assert index.index("guided/guided.css") < index.index("css/chain.css")
    app = (STATIC / "js/app.js").read_text()
    assert app.count('import "./chain/index.js";') == 1
    screen = (STATIC / "js/chain/screen.js").read_text()
    registrations = re.findall(r"register\(\"([\w-]+)\", render, \{([^}]*)\}", screen)
    assert registrations == [("downscale-3d", registrations[0][1])] and 'mode: "advanced"' in registrations[0][1]
    for path in CHAIN_JS:
        assert "guided" not in re.findall(r'register\("([\w-]+)"', path.read_text())
    css = (STATIC / "css/chain.css").read_text()
    assert set(re.findall(r"var\((--[\w-]+)", css)) <= {v for v in re.findall(r"var\((--[\w-]+)", css)
                                                         if v.startswith(("--gm1-", "--fs-"))}
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", css), "colours come from the base's tokens"
