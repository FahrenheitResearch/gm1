"""The guided/advanced mode plumbing: one engine, one run-directory contract; the mode is a
view stored per OS user in gpuwm-cm1/gui.json, never in a run directory.  CPU only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cm1core.gui.prefs import IMPORT_SWITCH_NOTICE, Prefs, config_dir
from gui_support import Client, import_supercell, start_server, stop_server


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file() and ".gm1-gui" not in path.parts}


def test_first_launch_with_no_runs_opens_guided(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        mode = client.get("/api/mode").json
        assert (mode["mode"], mode["source"]) == ("guided", "first-launch")
        assert client.get("/api/session").json["mode"] == "guided"
    finally:
        stop_server(server)


def test_runs_already_there_open_advanced_until_a_mode_is_stored(tmp_path):
    runs = tmp_path / "runs" / "from-a-terminal"
    runs.mkdir(parents=True)
    (runs / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "segments": []}))
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        assert client.get("/api/mode").json["source"] == "runs-present"
        assert client.get("/api/mode").json["mode"] == "advanced"
        assert client.post("/api/mode", {"mode": "guided"}).json["mode"] == "guided"
        assert client.get("/api/mode").json == {"mode": "guided", "source": "stored",
                                                 "modes": ["guided", "advanced"]}
    finally:
        stop_server(server)


def test_the_mode_survives_a_restart_and_a_second_browser_and_touches_no_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        import_supercell(client, "a")
        before = _tree_digest(tmp_path / "runs")
        reply = client.post("/api/mode", {"mode": "guided"}).json
        assert reply["ok"] and reply["command"] == "gpuwm-cm1 gui --mode guided"
        other_browser = Client(server.port, server.token)         # no cookie of the first one's
        assert other_browser.get("/api/mode").json["mode"] == "guided"
        assert client.post("/api/mode", {"mode": "advanced"}).json["command"] == "gpuwm-cm1 gui --mode advanced"
        assert _tree_digest(tmp_path / "runs") == before           # toggling changed no file in any run
        assert "a/commands.log" in before
        assert "gui --mode" not in (tmp_path / "runs" / "a" / "commands.log").read_text()
    finally:
        stop_server(server)
    stored = json.loads((tmp_path / "config" / "gui.json").read_text())
    assert stored["mode"] == "advanced"
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        assert client.get("/api/mode").json == {"mode": "advanced", "source": "stored",
                                                 "modes": ["guided", "advanced"]}
    finally:
        stop_server(server)


def test_importing_a_namelist_switches_to_advanced_once_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        assert client.get("/api/mode").json["mode"] == "guided"
        first = import_supercell(client, "one")
        assert first["mode"] == "advanced" and first["mode_notice"] == IMPORT_SWITCH_NOTICE
        client.post("/api/mode", {"mode": "guided"})               # the user goes back on purpose
        second = import_supercell(client, "two")
        assert second["mode"] == "guided" and second["mode_notice"] is None
    finally:
        stop_server(server)


def test_an_unknown_mode_is_refused_with_a_fix(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        answer = client.post("/api/mode", {"mode": "expert"})
        assert answer.status == 400 and answer.json["fix"] == "Use guided or advanced."
        assert not (tmp_path / "config" / "gui.json").exists()
    finally:
        stop_server(server)


def test_the_gui_command_stores_mode(tmp_path, monkeypatch):
    prefs = Prefs(tmp_path / "config")
    prefs.set_mode("advanced")
    assert prefs.mode(False) == ("advanced", "stored")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_dir() == tmp_path / "xdg" / "gpuwm-cm1"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert config_dir() == tmp_path / "home" / ".config" / "gpuwm-cm1"


def test_the_pages_draw_by_mode_from_one_attribute():
    static = Path(__file__).resolve().parents[1] / "src/cm1core/gui/static"
    css = (static / "css/app.css").read_text()
    assert 'body[data-mode="guided"] .advanced-only { display: none !important; }' in css
    assert 'body[data-mode="advanced"] .guided-only { display: none !important; }' in css
    api = (static / "js/api.js").read_text()
    assert "document.body.dataset.mode = mode" in api and 'post("/api/mode"' in api
