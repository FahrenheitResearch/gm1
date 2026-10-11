"""The GUI's environment from a WRF or ArWen run, through the server with no browser, no GPU.

* The three POSTs (open, column, use) each return their one ``gpuwm-cm1`` command, take ``dry_run`` (same line,
  no file changed), and the import replays from a shell to the same run folder (the replay walk).
* Each Show command equals its CLI line, and the CLI lines exit 0.
* Refusals come back in plain words with their command; pictures are served from the server's own folder only.
* The card flow on the fake runner, its server side: open, column, use for guided and for import.
* Guided keeps four screens: nothing here registers a guided screen; the words pass the copy lint; the page files
  are linked from index.html and app.js, and the stylesheet uses only the base's tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("netCDF4")

from cm1core.gui import commands as C  # noqa: E402
from cm1core.gui import copy_lint  # noqa: E402
from cm1import_support import namelist_tool  # noqa: E402
from gui_support import ROOT, start_server, stop_server  # noqa: E402
from test_gui_advanced import Walk, _environment, _mixed, _shim  # noqa: E402,F401
import wrf_support as ws  # noqa: E402

STATIC = ROOT / "src/cm1core/gui/static"
WRF_JS = sorted((STATIC / "js/wrf").glob("*.js"))


@pytest.fixture
def history(tmp_path):
    folder = tmp_path / "runs" / "wrf"
    folder.mkdir(parents=True)
    path = folder / "wrfout_d01_2026-05-01_18:00:00"
    facts = ws.write_wrfout(path, nx=12, ny=10, nz=40, records=3, alpha_deg=12.0, producer="WRF",
                            use_theta_m=1, map_proj=1)
    return path, facts


def test_the_wrf_actions_walk_the_api_table_and_replay(tmp_path, monkeypatch, history):
    path, facts = history
    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    scratch = tmp_path / "walk"
    scratch.mkdir()
    walk = Walk(server, client, root, _shim(tmp_path, "mixed"), scratch)
    try:
        opened = walk.post("wrf-open", "/api/wrf/open", {"path": os.fspath(path), "record": 1}, run=None)
        assert opened["producer"] == "WRF" and opened["records"] == 3 and opened["nx"] == 12
        assert opened["map"]["url"].startswith("/api/wrf/pictures/maps/")
        picture = client.get(opened["map"]["url"])
        assert picture.status == 200 and picture.headers["content-type"] == "image/png"
        column = walk.post("wrf-column", "/api/wrf/column",
                           {"path": os.fspath(path), "time": facts["times"][2], "ij": [5, 4]}, run=None)
        assert column["point"]["i"] == 5 and column["point"]["j"] == 4 and column["source"]["record"] == 2
        assert column["preview"]["url"] and client.get(column["preview"]["url"]).status == 200
        assert column["coverage_note"].startswith("This uses one column of your run as the environment")
        assert column["hydrostatic"]["max_abs_hpa"] < 0.5
        lat, lon = float(facts["lat"][4, 7]), float(facts["lon"][4, 7])
        guided = walk.post("wrf-use", "/api/wrf/use",
                           {"path": os.fspath(path), "time": facts["times"][0], "at": [lat, lon],
                            "target": "guided", "budget": "1h"}, run=None)
        assert guided["route"].startswith("guided/fit/your-environment/1h?sounding=")
        assert Path(guided["sounding"]).is_file() and guided["fit_command"].startswith(
            "gpuwm-cm1 preset fit your-environment --budget 1h --sounding ")
        imported = walk.post("wrf-use", "/api/wrf/use",
                             {"path": os.fspath(path), "time": facts["times"][0], "at": [lat, lon],
                              "target": "import", "name": "w1", "frame": "follow"}, run="w1")
        rundir = root / "w1"
        assert imported["url"] == "w1" and imported["rc"] == 0 and "--wrf-frame follow" in imported["command"]
        frame = json.loads((rundir / "wrf-column.json").read_text())["frame"]
        assert frame["asked"] == "follow" and frame["action"] in ("keep", "follow")
        if frame["action"] == "follow":
            assert f"umove = {frame['umove']}" in (rundir / "namelist.diff").read_text().replace("  ", " ")
        assert (rundir / "wrf-column.json").is_file() and (rundir / "input_sounding").is_file()
        assert " isnd      =  7," in (rundir / "namelist.input").read_text()
        record = json.loads((rundir / "wrf-column.json").read_text())
        assert record["point"]["i"] == 8 and record["point"]["j"] == 5
        run = json.loads((rundir / "run.json").read_text())
        assert run["sounding"]["source_path"] == os.fspath(path) and run["sounding"]["source_column"] == "wrf-column.json"
        assert imported["command"] in (rundir / "commands.log").read_text()
    finally:
        stop_server(server)


def test_show_commands_equal_the_cli_lines_and_the_lines_run(tmp_path, monkeypatch, history):
    path, facts = history
    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    try:
        base = root / ".gm1-gui" / "wrf"
        shown = client.post("/api/wrf/column", {"path": os.fspath(path), "record": 1, "at": [35.0, -97.0],
                                                "dry_run": True}).json
        key_dir = Path(shown["command"].split("--out ")[1].split()[0]).parent
        assert key_dir.parent == base / "columns"
        want = C.display(C.wrf_column_args(path, when={"record": 1}, point={"at": [35.0, -97.0]},
                                           out=key_dir / "input_sounding", map_field="shear",
                                           map_png=key_dir / "map-shear.png", preview_png=key_dir / "preview.png"))
        assert shown["command"] == want
        # The same line from a shell (the program on PATH is this checkout's CLI) exits 0 and writes the files.
        out = tmp_path / "cli"
        line = C.display(C.wrf_column_args(path, when={"record": 1}, point={"at": [35.0, -97.0]},
                                           out=out / "input_sounding", preview_png=out / "preview.png"))
        env = {**os.environ, "PATH": f"{_shim(tmp_path, 'mixed')}{os.pathsep}{os.environ['PATH']}",
               "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
        done = subprocess.run(["bash", "-c", line], env=env, capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr
        assert (out / "input_sounding").is_file() and (out / "wrf-column.json").is_file() and (out / "preview.png").is_file()
        times = subprocess.run(["bash", "-c", C.display(["sounding", "times", str(path)])], env=env,
                               capture_output=True, text=True, timeout=300)
        assert times.returncode == 0 and "record 2: 2026-05-01_20:00:00" in times.stdout
        imported = client.post("/api/wrf/use", {"path": os.fspath(path), "ij": [3, 3], "target": "import",
                                                "name": "w2", "dry_run": True}).json
        from cm1core.data import packaged_input_path

        assert imported["command"] == C.display(C.wrf_import_args(
            packaged_input_path("cm1r21.1-supercell-namelist"), path, when={}, point={"ij": [3, 3]}, out=root / "w2"))
        assert not (root / "w2").exists()
    finally:
        stop_server(server)


def test_refusals_come_back_in_plain_words_with_their_command(tmp_path, monkeypatch, history):
    path, facts = history
    server, client = _mixed(tmp_path, monkeypatch)
    try:
        outside = client.post("/api/wrf/column", {"path": os.fspath(path), "at": [60.0, 10.0]}).json
        assert outside["ok"] is False and outside["code"] == "GM1-WRF-POINT"
        assert "outside this domain" in outside["message"] and outside["fix"] == "pick a point inside the domain"
        assert outside["command"].startswith("gpuwm-cm1 sounding from-wrf ")
        text = tmp_path / "runs" / "notes.txt"
        text.write_text("not a history\n")
        bad = client.post("/api/wrf/open", {"path": os.fspath(text)}).json
        assert bad["ok"] is False and "could not be opened as netCDF" in bad["message"]
        missing = client.post("/api/wrf/open", {"path": "/no/such/wrfout"})
        assert missing.status == 400 and missing.json["command_template"].startswith("gpuwm-cm1 sounding times")
        no_point = client.post("/api/wrf/column", {"path": os.fspath(path)})
        assert no_point.status == 400 and "Pick one point" in no_point.json["message"]
        bad_time = client.post("/api/wrf/column", {"path": os.fspath(path), "time": "tomorrow", "ij": [1, 1]})
        assert bad_time.status == 400
        wrong_time = client.post("/api/wrf/column", {"path": os.fspath(path), "time": "2026-05-02_00:00:00",
                                                     "ij": [1, 1]}).json
        assert wrong_time["ok"] is False and wrong_time["code"] == "GM1-WRF-TIME"
        target = client.post("/api/wrf/use", {"path": os.fspath(path), "ij": [1, 1], "target": "elsewhere"})
        assert target.status == 400
        traversal = client.get("/api/wrf/pictures/..%2F..%2Fconfig/x.png")
        assert traversal.status in (400, 404)
        listed = client.get("/api/wrf/files").json
        assert [row["name"] for row in listed["files"]] == [path.name]
    finally:
        stop_server(server)


def test_the_card_flow_on_the_fake_runner(tmp_path, history):
    """The fake runner runs the real sounding command (it opens no GPU) and its own stand-in import."""

    namelist_tool()  # the sounding command reads the preset's namelist with the native parser
    path, facts = history
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        assert "sounding" in client.get("/api/session").json["commands_available"]
        opened = client.post("/api/wrf/open", {"path": os.fspath(path)}).json
        assert opened["ok"] and opened["times"] == facts["times"]
        column = client.post("/api/wrf/column", {"path": os.fspath(path), "record": 0, "ij": [6, 5]}).json
        assert column["ok"] and column["levels"] == 40
        use = client.post("/api/wrf/use", {"path": os.fspath(path), "record": 0, "ij": [6, 5], "target": "guided"}).json
        assert use["ok"] and use["preset"] == "your-environment" and use["budget"] == "1h"
        assert use["summary"].startswith("Your run's column at ")
        # The budget screen fits the preset on the kept column: its line is the one the card's reply names.
        shown = client.post("/api/fit", {"preset": use["preset"], "budget": "1h", "sounding": use["sounding"],
                                         "dry_run": True}).json
        assert shown["command"] == use["fit_command"], (shown["command"], use["fit_command"])
        fitted = client.post("/api/fit", {"preset": use["preset"], "budget": "1h", "sounding": use["sounding"]}).json
        import hashlib

        assert fitted["fit"]["sounding"]["sha256"] == hashlib.sha256(Path(use["sounding"]).read_bytes()).hexdigest()
        # The fit screen names the column in words, not the kept file's path.
        assert fitted["fit"]["sounding"]["source"].startswith("one column of your WRF run wrfout_d01_"), fitted["fit"]["sounding"]
        assert "grid column 6, 5" in fitted["fit"]["sounding"]["source"]
        made = client.post("/api/wrf/use", {"path": os.fspath(path), "record": 0, "ij": [6, 5], "target": "import",
                                            "name": "f1"}).json
        rundir = tmp_path / "runs" / "f1"
        assert made["url"] == "f1" and (rundir / "input_sounding").is_file() and (rundir / "wrf-column.json").is_file()
        assert (rundir / "input_sounding").read_bytes() == Path(use["sounding"]).read_bytes()
    finally:
        stop_server(server)


def test_guided_keeps_four_screens_and_the_words_pass_the_lint():
    guided_index = (STATIC / "guided/index.js").read_text()
    assert re.findall(r'TITLES = \{([^}]*)\}', guided_index)[0].count(":") == 4
    for path in WRF_JS:
        text = path.read_text()
        assert not re.search(r"register\([^)]*mode:\s*\"guided\"", text), path
        assert "register(" not in text or path.name == "index.js", path
    problems = copy_lint.lint_copy(ROOT / "src/cm1core/gui/copy/wrf.json")
    for path in WRF_JS:
        problems += copy_lint.lint_js(path)
    assert not problems, problems
    # Every words key the pages use is in the copy file.
    strings = json.loads((ROOT / "src/cm1core/gui/copy/wrf.json").read_text())["strings"]
    used = set()
    for path in WRF_JS:
        used |= set(re.findall(r'\bt\("([a-z0-9_.]+)"', path.read_text()))
        used |= set(re.findall(r'\bhint\("([a-z0-9_.]+)"', path.read_text()))
        used |= set(re.findall(r'"(expect\.[a-z]+)"', path.read_text()))
    assert used <= set(strings), sorted(used - set(strings))


def test_the_page_files_are_linked_where_the_plan_says():
    index = (STATIC / "index.html").read_text()
    links = re.findall(r'href="/static/([^"]+\.css)"', index)
    # After the guided sheets (guided.css, and groups.css), so the card's rules win where they meet; the
    # later sheets follow in their order (looks.css), so only chain.css may follow it.
    after = links[links.index("css/wrf.css") + 1:]
    assert set(after) <= {"css/chain.css"} and links.index("css/wrf.css") > links.index("guided/groups.css")
    app = (STATIC / "js/app.js").read_text()
    assert app.index('import "./wrf/index.js";') > app.index('import "./advanced/index.js";')
    css = (STATIC / "css/wrf.css").read_text()
    assert set(re.findall(r"var\((--[a-z0-9-]+)\)", css)) <= {v for v in re.findall(
        r"(--[a-z0-9-]+):", (STATIC / "css/gm1.css").read_text())}
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", css), "wrf.css uses only the base's tokens"
    newrun = (STATIC / "js/setup/newrun.js").read_text()
    assert "wrfPanel" in newrun


def test_the_wrf_actions_are_in_the_api_table():
    posts = {e.action: e for e in C.API if e.method == "POST"}
    for action in ("wrf-open", "wrf-column", "wrf-use"):
        assert posts[action].command.startswith("gpuwm-cm1 "), action
    assert "sounding" in C.HOST_COMMANDS


def test_the_cli_refuses_in_plain_words(tmp_path, history):
    path, facts = history
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run([sys.executable, "-m", "cm1core", "sounding", "from-wrf", str(path), "--at=0,0",
                           "--out", str(tmp_path / "x"), "--json"], env=env, capture_output=True, text=True)
    assert done.returncode == 2
    body = json.loads(done.stdout)
    assert body["ok"] is False and body["code"] == "GM1-WRF-POINT"
    both = subprocess.run([sys.executable, "-m", "cm1core", "import", str(ROOT / "src/cm1core/data/"
                           "cm1r21.1-supercell-namelist.input"), "--wrf", str(path), "--out", str(tmp_path / "r")],
                          env=env, capture_output=True, text=True)
    assert both.returncode == 2 and "--wrf-at" in both.stderr and not (tmp_path / "r").exists()


def test_a_python_without_matplotlib_still_gets_the_facts_and_the_column(tmp_path, history, monkeypatch):
    """The pictures need matplotlib (a gpuwm dependency); without it the file's facts, the column, its numbers and
    the sounding still come back, and each picture says why it is missing instead of a traceback."""

    import argparse

    from cm1core.cm1import import sounding_door, wrfpictures

    def no_matplotlib(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'matplotlib'")

    monkeypatch.setattr(wrfpictures, "map_png", no_matplotlib)
    monkeypatch.setattr(wrfpictures, "preview_png", no_matplotlib)
    path, _ = history
    opened = sounding_door.times(argparse.Namespace(file=str(path), record=None, time=None, map="shear",
                                                    map_png=str(tmp_path / "map.png"), library=None))
    assert opened["ok"] and opened["records"] == 3
    assert opened["map"]["drawn"] is False and "matplotlib" in opened["map"]["why"]
    out = tmp_path / "input_sounding"
    column = sounding_door.from_wrf(argparse.Namespace(
        file=str(path), record=1, time=None, at=None, ij="3,4", out=str(out), map="shear",
        map_png=str(tmp_path / "m.png"), preview_png=str(tmp_path / "p.png"), top_m=None, library=None),
        command="gpuwm-cm1 sounding from-wrf ...")
    assert column["ok"] and out.is_file() and column["levels"] == 40
    assert column["preview"]["drawn"] is False and column["map"]["drawn"] is False
    assert not (tmp_path / "map.png").exists() and not (tmp_path / "p.png").exists()


def test_the_guided_card_refuses_a_column_below_its_storms_top_in_plain_words(tmp_path):
    """Next checks the column against the guided storm's own namelist (``--top-of``), so a short column is refused
    on the gallery in plain words, not one screen later by the fit; the advanced import checks its own namelist."""

    namelist_tool()  # --top-of reads the guided storm's namelist with the native parser
    folder = tmp_path / "runs" / "wrf"
    folder.mkdir(parents=True)
    path = folder / "wrfout_d01_2026-05-01_18:00:00"
    facts = ws.write_wrfout(path, nx=12, ny=10, nz=40, records=1, terrain_m=8000.0)
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        low = client.post("/api/wrf/use", {"path": os.fspath(path), "ij": [12, 5], "target": "guided"}).json
        assert low["ok"] is False and low["code"] == "GM1-WRF-TOP", low
        assert low["message"].startswith("The air in your run's column stops ") and "km up" in low["message"]
        assert "--top-of " in low["command"] and low["command"].split("--top-of ")[1].split()[0].endswith(
            "your-environment/namelist.input")
        high = client.post("/api/wrf/use", {"path": os.fspath(path), "ij": [1, 5], "target": "guided"}).json
        assert high["ok"] is True, high
    finally:
        stop_server(server)


def test_the_import_report_says_what_follow_did():
    from cm1core.cm1import.door import frame_sentence

    assert frame_sentence({"asked": "namelist"}) == ""
    moved = frame_sentence({"asked": "follow", "action": "follow", "umove": 2.5, "vmove": 16.5, "imove": 1,
                            "bubble_center_x_m": 8170.0, "bubble_center_y_m": -2239.1, "stays_inside": True,
                            "was": {"imove": 1, "umove": 12.5, "vmove": 3.0}})
    assert moved.startswith("Frame: moved to follow") and "was 12.5, 3 m/s" in moved and "keeps it inside" in moved
    wide = frame_sentence({"asked": "follow", "action": "domain", "required_domain_km": [57.2, 44.9]})
    assert "at least 57 x 45 km" in wide and "widen nx and ny" in wide
