"""The setup screens (New run, Check, Sounding, Domain and time, Settings, Estimate).

1. A headless DOM test: the served page and every module it loads are parsed (``html.parser`` for the page, the
   module graph for the scripts), each setup screen is registered, every endpoint the screens call answers, and the
   JSON contract each screen draws from is held to a snapshot (``tests/data/gui/setup-contracts.json``).  No
   browser window.  (A browser walk in the development repository checks the pages in a real browser.)
2. "Use the recommended frame and bubble" writes a namelist diff that re-imports with exit 0, and the pair it
   writes is then the recommendation (a fixed point).
3. Every action on these screens returns its ``gpuwm-cm1`` command, which commands.log records and which replays
   from a shell to the same files; the page draws each through the one Show command component.

The real host-only commands (import, edit, check, estimate, jobscript) run where the native namelist parser is
available (a checkout builds it, as the importer tests do); the fake runner stands in for the GPU.  CPU only.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest

from cm1import_support import namelist_tool
from gui_support import ROOT, SUPERCELL, import_supercell, start_server, stop_server, wait_for

STATIC = ROOT / "src/cm1core/gui/static"
SNAPSHOT = ROOT / "tests/data/gui/setup-contracts.json"
GPUINFO = ROOT / "tests/data/gpuinfo"
SCREENS = ("new-run", "check", "sounding", "domain", "settings", "estimate")


def _gpu_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", os.fspath(GPUINFO / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", os.fspath(GPUINFO / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_CACHE_DIR", os.fspath(tmp_path / "cache"))
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    monkeypatch.setenv("GM1_FAKE_MEASURE_S", "0.1")


@pytest.fixture()
def mixed(tmp_path, monkeypatch):
    """A server running the real host-only commands, with the fake runner for the GPU."""

    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))
    _gpu_env(monkeypatch, tmp_path)
    from cm1core.gui.jobs import mixed_runner

    server, client = start_server(tmp_path / "runs", tmp_path / "config", runner=mixed_runner())
    yield server, client, tmp_path / "runs"
    stop_server(server)


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    _gpu_env(monkeypatch, tmp_path)
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    yield server, client, tmp_path / "runs"
    stop_server(server)


def _shape(value, depth=0):
    """A JSON value's keys, recursively (the contract), without its numbers."""

    if isinstance(value, dict):
        return {key: _shape(item, depth + 1) for key, item in sorted(value.items())} if depth < 2 else sorted(value)
    if isinstance(value, list):
        return [_shape(value[0], depth + 1)] if value and isinstance(value[0], dict) else "list"
    return type(value).__name__ if value is not None else "null"


def _keys(value) -> list[str]:
    return sorted(value) if isinstance(value, dict) else []


# ---------------------------------------------------------------- 1. the page and its modules


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[dict] = []
        self.links: list[dict] = []
        self.urls: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.scripts.append(attrs)
        if tag == "link":
            self.links.append(attrs)
        for name in ("src", "href"):
            if attrs.get(name):
                self.urls.append(attrs[name])


_IMPORT = re.compile(r'(?:import|from)\s*(?:\{[^}]*\}\s*from\s*)?"(\.{1,2}/[^"]+)"')


def _module_graph(client) -> dict[str, str]:
    """Every module the page loads, fetched from the server as a browser would (path -> text)."""

    index = client.get("/")
    assert index.status == 200 and index.headers["content-type"].startswith("text/html")
    page = _Page()
    page.feed(index.raw.decode("utf-8"))
    assert all(not re.match(r"^[a-z]+://", url) for url in page.urls), page.urls
    entry = [s["src"] for s in page.scripts if s.get("type") == "module"]
    assert entry == ["/static/js/app.js"]
    for link in page.links:
        answer = client.get(link["href"])
        # The tab icon is the one link that is not a stylesheet: served, so a page load logs no 404.
        kind = "image/png" if link.get("rel") == "icon" else "text/css"
        assert answer.status == 200 and answer.headers["content-type"].startswith(kind), link
    assert any(link.get("rel") == "icon" for link in page.links)
    seen: dict[str, str] = {}
    todo = list(entry)
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        answer = client.get(path)
        assert answer.status == 200, path
        assert answer.headers["content-type"].startswith("text/javascript"), path
        text = answer.raw.decode("utf-8")
        seen[path] = text
        base = path.rsplit("/", 1)[0]
        for ref in _IMPORT.findall(text):
            parts = (base + "/" + ref).split("/")
            out: list[str] = []
            for part in parts:
                if part == "..":
                    out.pop()
                elif part not in (".", ""):
                    out.append(part)
            todo.append("/" + "/".join(out))
    return seen


def test_the_page_loads_every_setup_screen_and_nothing_from_outside(fake):
    server, client, root = fake
    modules = _module_graph(client)
    registered = set()
    for text in modules.values():
        registered |= set(re.findall(r'register\("([a-z-]+)"', text))
    assert set(SCREENS) <= registered, registered
    for path, text in modules.items():
        assert not re.search(r"https?://", text), path
    # Every named import resolves to an export of the module it names.
    for path, text in modules.items():
        base = path.rsplit("/", 1)[0]
        for names, ref in re.findall(r'import\s*\{([^}]*)\}\s*from\s*"([^"]+)"', text):
            target = os.path.normpath(base + "/" + ref)
            exported = set(re.findall(r"export\s+(?:async\s+)?(?:function|const|let|class)\s+([\w$]+)",
                                      modules[target]))
            for group in re.findall(r"export\s*\{([^}]*)\}", modules[target]):
                exported |= {item.strip().split(" as ")[-1] for item in group.split(",") if item.strip()}
            for name in names.split(","):
                name = name.strip().split(" as ")[0].strip()
                assert not name or name in exported, (path, ref, name)


def test_the_setup_modules_parse_as_javascript_modules(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the browser walk parses them instead")
    for path in sorted((STATIC / "js").rglob("*.js")):
        copy = tmp_path / (path.stem + ".mjs")
        copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        done = subprocess.run([node, "--check", os.fspath(copy)], capture_output=True, text=True)
        assert done.returncode == 0, (path, done.stderr)


def test_each_setup_screen_endpoint_answers_its_contract(mixed):
    server, client, root = mixed
    contracts: dict[str, object] = {}

    examples = client.get("/api/examples?wait=1", timeout=300)
    assert examples.status == 200, examples.raw
    ex = examples.json
    ids = [row["id"] for row in ex["examples"]]
    assert ids[:2] == ["cm1-supercell", "cm1-squall-line"]
    supercell, squall = ex["examples"][0], ex["examples"][1]
    assert supercell["exit"] == 0 and supercell["guided"]["offered"] is True
    assert supercell["grid"] == {"nx": 120, "ny": 120, "nz": 40, "dx": 1000.0, "dy": 1000.0, "dz": 500.0}
    assert supercell["estimate"]["fits"] is True and supercell["estimate"]["smallest_card_gb"] == 8
    # The squall line's repeating sides no longer hold it back; guided mode still greys it, as it greys the
    # preset made from this namelist, because nobody has run it on a GPU yet.
    assert squall["guided"]["offered"] is False and "not run yet" in squall["guided"]["why_not"]
    assert supercell["commands"]["import"].startswith("gpuwm-cm1 import ")
    contracts["GET /api/examples"] = _shape({**ex, "examples": ex["examples"][:1]})

    imported = import_supercell(client)
    assert imported["ok"] and imported["rc"] == 0
    setup = client.get("/api/runs/sc1/setup").json
    assert setup["estimate"] is None and setup["estimate_stale"] is True
    assert [row["id"] for row in setup["steps"]] == ["check", "sounding", "domain", "settings", "estimate"]
    assert setup["values"]["nx"] == 120 and setup["values"]["iinit"] == 1
    assert [g["id"] for g in setup["setup"]["fortran"]] == [
        "bubble", "coldpool", "line", "nudging", "convergence", "uforce", "benchmark", "benchmark_sounding", "parcels"]
    assert len(setup["setup"]["output_words"]) == 62
    estimated = client.post("/api/runs/sc1/estimate").json
    assert estimated["ok"], estimated
    setup = client.get("/api/runs/sc1/setup").json
    assert setup["estimate_stale"] is False and setup["geometry_stale"] is False
    assert [row["id"] for row in setup["estimate"]["rows"]] == ["gpu", "host", "disk"]
    assert setup["estimate"]["gpu"]["name"] == "NVIDIA GeForce RTX 3080"
    assert setup["geometry"]["status"] == "evaluated" and setup["geometry"]["hodograph_m_s"]
    assert "About" in setup["plain"]["text"] and "RTX 3080" in setup["plain"]["text"]
    # run.json is the supervisor's record, passed through; its inner keys depend on the install (a git checkout names its
    # commit), so the contract holds only that it is there, with its top-level keys.
    contracts["GET /api/runs/RUN/setup"] = {**_shape({k: v for k, v in setup.items() if k != "run"}),
                                            "run": sorted(setup["run"])}
    contracts["setup.estimate"] = _shape(setup["estimate"])
    contracts["setup.sounding"] = _shape(setup["sounding"])
    contracts["setup.geometry.recommended_pair"] = _keys(setup["geometry"]["recommended_pair"])
    contracts["setup.setup.fortran[]"] = _keys(setup["setup"]["fortran"][0])
    contracts["setup.setup.fortran[].fields[]"] = _keys(setup["setup"]["fortran"][0]["fields"][0])
    contracts["setup.setup.output_words[]"] = _keys(setup["setup"]["output_words"][0])

    report = client.get("/api/runs/sc1/report").json
    contracts["GET /api/runs/RUN/report: sections[]"] = _keys(report["report"]["sections"][0])
    nml = client.get("/api/runs/sc1/namelist").json
    assert nml["edited"] is False and nml["editable"] is True and "&param1" in nml["text"]
    contracts["GET /api/runs/RUN/namelist"] = _keys(nml)
    contracts["GET /api/runs/RUN/measure"] = _keys(client.get("/api/runs/sc1/measure").json)

    posts = {
        "edit": ("/api/runs/sc1/edit", {"set": {"tapfrq": 600.0}}),
        "jobscript": ("/api/runs/sc1/jobscript", {"scheduler": "slurm", "time": "01:00:00"}),
        "acknowledge": ("/api/runs/sc1/acknowledge", {"output_keys": True}),
    }
    for action, (path, body) in posts.items():
        reply = client.post(path, body).json
        assert reply["command"].startswith("gpuwm-cm1 "), (action, reply)
        contracts[f"POST {action}"] = _keys(reply)

    current = json.loads(json.dumps(contracts, sort_keys=True))
    if os.environ.get("GM1_UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    recorded = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert current == recorded, "the setup screens' JSON contract changed: review, then GM1_UPDATE_SNAPSHOTS=1"


# ---------------------------------------------------------------- 2. the recommended frame and bubble


def test_the_recommended_frame_and_bubble_writes_a_diff_that_reimports(mixed, tmp_path):
    server, client, root = mixed
    import_supercell(client)
    rundir = root / "sc1"
    refused = client.post("/api/runs/sc1/recommended")
    assert refused.status == 409 and refused.json["command"].startswith("gpuwm-cm1 estimate ")
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    pair = client.get("/api/runs/sc1/setup").json["geometry"]["recommended_pair"]
    assert pair["uses_it"] is False
    before = tmp_path / "before"
    shutil.copytree(rundir, before)

    reply = client.post("/api/runs/sc1/recommended").json
    assert reply["ok"] and reply["rc"] == 0 and reply["verdict"] == "runnable", reply
    u, v = pair["frame_m_s"]
    assert f"+ umove   = {u!r}," in reply["diff"] and f"+ vmove   =  {v!r}," in reply["diff"]
    keys = {row["key"]: row["to"] for row in reply["settings_changed"]}
    assert keys == {"initialization.bubble_center_x_m": pair["bubble_km"][0] * 1000.0,
                    "initialization.bubble_center_y_m": pair["bubble_km"][1] * 1000.0}
    # The namelist diff is kept in the run folder, against the user's original file.
    diff = (rundir / "namelist.diff").read_text()
    assert diff.startswith("--- namelist.input.original") and f"+ umove   = {u!r}," in diff
    report = json.loads((rundir / "import-report.json").read_text())
    assert report["exit"] == 0 and report["reimport"] is True
    deck = (rundir / "deck.toml").read_text()
    assert f"bubble_center_x_m = {pair['bubble_km'][0] * 1000.0!r}" in deck

    # The edited namelist imports with exit 0 on its own, into a new folder, as gpuwm-cm1 import.
    fresh = tmp_path / "fresh"
    done = subprocess.run([sys.executable, "-m", "cm1core", "import", os.fspath(rundir / "namelist.input"),
                           "--out", os.fspath(fresh), "--json"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": os.fspath(ROOT / "src")})
    assert done.returncode == 0, done.stderr

    # The pair is a fixed point: estimated again, the run already uses the recommendation.
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    again = client.get("/api/runs/sc1/setup").json["geometry"]
    assert again["recommended_pair"]["uses_it"] is True
    assert again["minimum_edge_clearance_km"] >= pair["minimum_edge_clearance_km"] - 0.5
    unchanged = client.post("/api/runs/sc1/recommended").json
    assert unchanged["verdict"] == "unchanged" and unchanged["command"].startswith("gpuwm-cm1 edit ")

    # The command the server logged replays from a shell onto the copy taken before, to the same files.
    logged = [line for line in (rundir / "commands.log").read_text().splitlines()
              if line.startswith("gpuwm-cm1 edit ")][0]
    argv = shlex.split(logged.replace(os.fspath(rundir), os.fspath(before)))
    done = subprocess.run([sys.executable, "-m", "cm1core", *argv[1:]], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": os.fspath(ROOT / "src")})
    assert done.returncode == 0, done.stderr
    for name in ("namelist.input", "namelist.diff", "settings.toml", "deck.toml"):
        assert (before / name).read_bytes() == (rundir / name).read_bytes(), name


# ---------------------------------------------------------------- 3. every action returns its command


SETUP_ACTIONS = {"edit", "recommended", "namelist", "acknowledge", "sounding", "jobscript", "measure"}


def test_a_recommendation_older_than_the_namelist_is_not_written(mixed):
    server, client, root = mixed
    import_supercell(client)
    rundir = root / "sc1"
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    # The grid changes after the storm-track check ran: the pair it worked out is for the old grid.
    assert client.post("/api/runs/sc1/edit", {"set": {"nx": 80, "ny": 80}}).json["ok"]
    assert client.get("/api/runs/sc1/setup").json["geometry_stale"] is True
    before = (rundir / "namelist.input").read_bytes()
    stale = client.post("/api/runs/sc1/recommended")
    assert stale.status == 409 and "older than the run's namelist" in stale.json["message"]
    assert stale.json["command"].startswith("gpuwm-cm1 estimate ")
    assert (rundir / "namelist.input").read_bytes() == before
    # Estimated again, the pair is the new grid's, and it is written.
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    fresh = client.post("/api/runs/sc1/recommended").json
    assert fresh["ok"] and fresh["rc"] == 0, fresh


def test_a_start_the_track_model_does_not_describe_gets_its_reason_not_a_recommendation(mixed):
    server, client, root = mixed
    squall = ROOT / "src/cm1core/data/cm1r21.1-squall_line-namelist.input"
    assert client.post("/api/import", {"name": "sq", "namelist": {"path": os.fspath(squall)}}).json["rc"] == 0
    assert client.post("/api/runs/sq/estimate").json["ok"]
    assert client.get("/api/runs/sq/setup").json["geometry"]["status"] != "evaluated"
    before = (root / "sq" / "namelist.input").read_bytes()
    answer = client.post("/api/runs/sq/recommended")
    assert answer.status == 409 and "no recommended frame and bubble" in answer.json["message"]
    assert "iinit = 8" in answer.json["fix"] and (root / "sq" / "namelist.input").read_bytes() == before


def test_every_setup_action_is_in_the_api_table_with_its_command():
    from cm1core.gui.commands import API

    posts = {row.action: row for row in API if row.method == "POST"}
    assert SETUP_ACTIONS <= set(posts)
    for action in SETUP_ACTIONS | {"check", "estimate", "start", "import"}:
        assert posts[action].command.startswith("gpuwm-cm1 "), action


def test_every_setup_action_returns_its_command_and_commands_log_holds_it(mixed, tmp_path):
    server, client, root = mixed
    import_supercell(client)
    rundir = root / "sc1"
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    sounding = tmp_path / "input_sounding"
    sounding.write_text("1000.0 300.0 14.0\n")
    text = (rundir / "namelist.input").read_text()
    replies = {
        "edit": client.post("/api/runs/sc1/edit", {"set": {"tapfrq": 600.0},
                                                     "settings": {"initialization.zc_m": 1500.0}}).json,
    }
    # As the Domain and time page does: the edit re-imported the run, so it is estimated again before the
    # recommendation (worked out for the namelist as it now is) can be written.
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    replies |= {
        "recommended": client.post("/api/runs/sc1/recommended").json,
        "namelist": client.post("/api/runs/sc1/namelist", {"text": text.replace("600.0", "300.0", 1)}).json,
        "acknowledge": client.post("/api/runs/sc1/acknowledge", {"output_keys": True}).json,
        "jobscript": client.post("/api/runs/sc1/jobscript", {"scheduler": "pbs"}).json,
        "check": client.post("/api/runs/sc1/check").json,
    }
    # isnd = 5 reads no file: a sounding is taken and set aside (the report says it is not read).
    replies["sounding"] = client.post("/api/runs/sc1/sounding", {"sounding": {"path": os.fspath(sounding)}}).json
    replies["measure"] = client.post("/api/runs/sc1/measure").json
    wait_for(lambda: client.get("/api/runs/sc1/measure").json["state"] != "running", timeout=60)
    for action, reply in replies.items():
        assert isinstance(reply.get("command"), str) and reply["command"].startswith("gpuwm-cm1 "), (action, reply)
    assert replies["edit"]["ok"] and "--setting initialization.zc_m=1500" in replies["edit"]["command"]
    assert replies["namelist"]["ok"] and "--from" in replies["namelist"]["command"]
    assert replies["jobscript"]["ok"] and replies["jobscript"]["script"].startswith("#!/bin/bash")
    assert replies["measure"]["ok"] and replies["measure"]["command"].endswith("--measure --json")
    logged = [line for line in (rundir / "commands.log").read_text().splitlines() if line.startswith("gpuwm-cm1 ")]
    for action, reply in replies.items():
        if action == "check" or reply.get("verdict") == "unchanged":
            continue
        assert reply["command"] in logged, action
    # The settings.toml choice survives every later re-import (the editor's Save, the acknowledgment).
    assert "zc_m = 1500.0" in (rundir / "settings.toml").read_text()
    report = json.loads((rundir / "import-report.json").read_text())
    assert "output-keys" in report["acknowledgments"]["given"] or not report["acknowledgments"]["needed"]


def test_the_page_draws_every_command_through_the_show_command_component():
    api_js = (STATIC / "js/api.js").read_text()
    # Every POST reply that carries a command reaches the listeners...
    assert "for (const fn of commandListeners) fn(data)" in api_js
    app = (STATIC / "js/app.js").read_text()
    # ...and the frame draws them ("Commands this page ran", advanced).
    assert "api.onCommand(" in app and "paintRan" in app
    common = (STATIC / "js/setup/common.js").read_text()
    assert "showCommand(reply.command" in common
    for path in sorted((STATIC / "js/setup").glob("*.js")):
        text = path.read_text()
        # The screens show commands only through command.js's component (showCommand, or the advanced pages' actionButton
        # and liveCommand, which draw the same component beside a button).
        assert "copyline(" not in text or path.name == "estimate.js", path.name
        assert any(name in text for name in ("showCommand", "actionButton", "liveCommand")) or \
            path.name in ("index.js", "planmap.js"), path.name


# ---------------------------------------------------------------- guided and advanced, one engine


def test_a_guided_gallery_pick_keeps_guided_and_imports_the_example(fake):
    server, client, root = fake
    assert client.get("/api/mode").json == {"mode": "guided", "source": "first-launch",
                                            "modes": ["guided", "advanced"]}
    examples = client.get("/api/examples?wait=1", timeout=300).json["examples"]
    supercell = next(row for row in examples if row["id"] == "cm1-supercell")
    reply = client.post("/api/import", {"name": "supercell-1km", "namelist": {"path": supercell["path"]},
                                        "example": True}).json
    assert reply["ok"] and reply["mode"] == "guided" and reply["mode_notice"] is None
    assert client.get("/api/mode").json["source"] == "stored"
    setup = client.get("/api/runs/supercell-1km/setup").json
    assert setup["mode"] == "guided" and setup["commands"]["start"].startswith("gpuwm-cm1 run ")
    # A user's own namelist still switches to advanced once, with the notice.
    own = import_supercell(client, name="mine")
    assert own["mode"] == "advanced" and own["mode_notice"]


def test_the_fake_runner_speaks_the_setup_contract(fake):
    server, client, root = fake
    import_supercell(client)
    setup = client.get("/api/runs/sc1/setup").json
    assert setup["report"]["present"] and [g["id"] for g in setup["setup"]["fortran"]][0] == "bubble"
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    setup = client.get("/api/runs/sc1/setup").json
    assert [row["id"] for row in setup["estimate"]["rows"]] == ["gpu", "host", "disk"]
    edit = client.post("/api/runs/sc1/edit", {"settings": {"initialization.zc_m": 1500}}).json
    assert edit["ok"] and edit["command"].startswith("gpuwm-cm1 edit ")
    assert client.post("/api/runs/sc1/jobscript", {"scheduler": "slurm"}).json["script"].startswith("#!/bin/bash")
    measure = client.post("/api/runs/sc1/measure").json
    assert measure["ok"]
    wait_for(lambda: client.get("/api/runs/sc1/measure").json["state"] == "done", timeout=30)
    # an edit is refused once the run has run
    started = client.post("/api/runs/sc1/start").json
    assert started["ok"]
    wait_for(lambda: client.get("/api/runs/sc1/status").json["state"] == "finished", timeout=60)
    late = client.post("/api/runs/sc1/edit", {"set": {"tapfrq": 300}})
    assert late.status == 409 and late.json["command"].startswith("gpuwm-cm1 edit ")


def test_bad_edits_are_refused_before_anything_runs(fake):
    server, client, root = fake
    import_supercell(client)
    for body in ({}, {"set": {"Bad Key": 1}}, {"settings": {"zc": 1}}, {"set": {"tapfrq": [1]}},
                 {"seed": 0}, {"reset": "x"}):
        answer = client.post("/api/runs/sc1/edit", body)
        assert answer.status == 400, (body, answer.json)
    answer = client.post("/api/runs/sc1/jobscript", {"scheduler": "lsf"})
    assert answer.status == 400
    answer = client.post("/api/runs/sc1/jobscript", {"scheduler": "slurm", "partition": "a b; rm"})
    assert answer.status == 400
    answer = client.post("/api/runs/sc1/acknowledge", {})
    assert answer.status == 400


def test_also_write_netcdf_starts_no_second_writer_beside_the_cm1out_sidecar(fake):
    """cm1out writes netCDF beside GrADS in the same pass when settings.toml asks; a second
    `cm1out --format nc` writer would race it for the same files, so Start launches the standard two."""

    server, client, root = fake
    import_supercell(client)
    rundir = root / "sc1"
    settings = rundir / "settings.toml"
    settings.write_text(settings.read_text().replace("also_netcdf = false", "also_netcdf = true"))
    report = json.loads((rundir / "import-report.json").read_text())
    report["setup"]["values"]["output_format"] = 1
    (rundir / "import-report.json").write_text(json.dumps(report))
    started = client.post("/api/runs/sc1/start").json
    assert not any("--format nc" in line for line in started["sidecars"]), started["sidecars"]
    assert sum(" cm1out " in f" {line} " for line in started["sidecars"]) == 1, started["sidecars"]


# ---------------------------------------------------------------- the examples ship with the package


def test_cm1s_two_examples_are_packaged_with_their_digests():
    from cm1core.data import packaged_input_path

    tsv = (ROOT / "evidence/authority/cm1r21.1-content-sha256.tsv").read_text()
    for name, source in (("cm1r21.1-supercell-namelist", "run/config_files/supercell/namelist.input"),
                         ("cm1r21.1-squall-line-namelist", "run/config_files/squall_line/namelist.input")):
        path = packaged_input_path(name)
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert f"{digest}\t{path.stat().st_size}\t{source}" in tsv
    authority = json.loads((ROOT / "src/cm1core/records/CM1-AUTHORITY.json").read_text())
    bundled = {row["source_path"] for row in authority["redistribution"]["bundled"]}
    # The rows whose source is not one file: the defaults table and CM1's own output (two).
    rows_by_path = {row["path"]: row["source_path"] for row in authority["redistribution"]["bundled"]}
    assert bundled == {"run/config_files/supercell/namelist.input", "run/config_files/squall_line/namelist.input",
                       "run/config_files/hurricane_3d_cpm/namelist.input",
                       "run/config_files/hurricane_3d_cpm/input_sounding",
                       "run/LANDUSE.TBL", "run/namelist.input", rows_by_path["src/cm1core/records/CM1-R21-NAMELIST-DEFAULTS.json"],
                       rows_by_path["tests/data/cm1-ref-24x24/"],
                       rows_by_path["rust/cm1-state/evidence/cm1-base-columns.txt"],
                       rows_by_path["src/cm1core/records/CM1-R22-NEW-KEYS.json"],
                       rows_by_path["src/cm1core/records/CM1-KEY-ALIASES.json"],
                       rows_by_path["src/cm1core/records/CM1-STOCK-SUPERCELL-SERIES.json"],
                       rows_by_path["src/cm1core/records/GM1-HURRICANE-RUNS.json"]}
    in_the_package = {row["path"] for row in authority["redistribution"]["bundled"]
                      if row["path"].startswith("src/cm1core/data/")}
    assert in_the_package == {"src/cm1core/data/cm1r21.1-supercell-namelist.input",
                              "src/cm1core/data/cm1r21.1-squall_line-namelist.input"}


# ---------------------------------------------------------------- a source checkout, and the GPU memory budget


def test_native_dir_puts_the_checkouts_tools_on_every_command_that_opens_the_gpu(tmp_path, monkeypatch):
    from cm1core.gui.server import native_tools

    _gpu_env(monkeypatch, tmp_path)
    native = tmp_path / "native"
    native.mkdir()
    with pytest.raises(ValueError, match="libcm1_state.so"):
        native_tools(native)
    (native / "libcm1_state.so").write_bytes(b"")
    (native / "cm1_forecast_publish").write_bytes(b"")
    tools = native_tools(native)
    server, client = start_server(tmp_path / "runs", tmp_path / "config", native=tools)
    try:
        import_supercell(client)
        setup = client.get("/api/runs/sc1/setup").json
        both = f"--library {native / 'libcm1_state.so'} --history-publisher {native / 'cm1_forecast_publish'}"
        assert setup["commands"]["start"].endswith(both)
        assert f"--measure --library {native / 'libcm1_state.so'} --json" in setup["commands"]["measure"]
        assert both in client.post("/api/runs/sc1/jobscript", {"scheduler": "slurm"}).json["command"]
        assert client.post("/api/runs/sc1/estimate").json["ok"]
        started = client.post("/api/runs/sc1/start").json
        assert started["ok"] and started["command"].endswith(both)
        wait_for(lambda: client.get("/api/runs/sc1/status").json["state"] == "finished", timeout=60)
        results = client.get("/api/runs/sc1/results").json
        assert results["commands"]["resume"].endswith(both)
    finally:
        stop_server(server)
    from cm1core.cli import main

    assert main(["gui", "--root", os.fspath(tmp_path / "r2"), "--native-dir", os.fspath(tmp_path / "missing"),
                 "--config-dir", os.fspath(tmp_path / "c2")]) == 2


def test_the_gpu_memory_budget_is_a_setting_the_deck_carries(mixed):
    server, client, root = mixed
    import_supercell(client)
    rundir = root / "sc1"
    setup = client.get("/api/runs/sc1/setup").json
    assert setup["setup"]["gpu"]["memory_budget_mib"] == 8192 and setup["setup"]["gpu"]["chosen"] is None
    assert client.post("/api/runs/sc1/estimate").json["ok"]
    recommended = client.get("/api/runs/sc1/setup").json["estimate"]["recommendation"]["memory_budget_mib"]
    assert 256 <= recommended <= 8192
    reply = client.post("/api/runs/sc1/edit", {"settings": {"gpu.memory_budget_mib": recommended}}).json
    assert reply["ok"] and f"--setting gpu.memory_budget_mib={recommended}" in reply["command"]
    assert f"memory_budget_mib = {recommended}" in (rundir / "deck.toml").read_text()
    setup = client.get("/api/runs/sc1/setup").json
    assert setup["setup"]["gpu"] == {**setup["setup"]["gpu"], "memory_budget_mib": recommended, "chosen": recommended}
    # the importer's own [runtime] table is not read back as the user's choice; resetting restores its default
    assert client.post("/api/runs/sc1/edit", {"reset": ["gpu.memory_budget_mib"]}).json["ok"]
    assert "memory_budget_mib = 8192" in (rundir / "deck.toml").read_text()
    # below config.py's floor the edit is refused before anything is written, and says why
    low = client.post("/api/runs/sc1/edit", {"settings": {"gpu.memory_budget_mib": 100}}).json
    assert low["ok"] is False and "at least 256" in low["fix"] and low["command"].startswith("gpuwm-cm1 edit ")
    assert "memory_budget_mib = 8192" in (rundir / "deck.toml").read_text()


def test_sounding_shows_the_skew_t_card_once_the_quick_look_worker_has_drawn_it(fake):
    server, client, root = fake
    import_supercell(client)
    assert client.get("/api/runs/sc1/setup").json["sounding"]["card"] is None
    quicklook = root / "sc1" / "quicklook"
    quicklook.mkdir()
    (quicklook / "sounding-t0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (quicklook / "index.json").write_text(json.dumps({"products": [
        {"id": "reflectivity", "frames": [{"file": "r.png"}]},
        {"id": "sounding", "title": "Sounding at T+0", "frames": [{"file": "sounding-t0.png"}]}]}))
    card = client.get("/api/runs/sc1/setup").json["sounding"]["card"]
    assert card == {"title": "Sounding at T+0", "url": "/api/runs/sc1/files/quicklook/sounding-t0.png"}
    assert client.get(card["url"]).status == 200


def test_sounding_shows_the_column_skew_t_the_quick_looks_drew(fake):
    """The quick-look draws the skew-T on request (quicklook/slices/column-*.png, the starting sounding beside the
    column); the earliest one stands for the run's sounding."""

    server, client, root = fake
    import_supercell(client)
    slices = root / "sc1" / "quicklook" / "slices"
    slices.mkdir(parents=True)
    for name in ("column-max-t003600.png", "column-max-t000000.png", "column-10_m5-t0001800p500.png"):
        (slices / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    card = client.get("/api/runs/sc1/setup").json["sounding"]["card"]
    assert card == {"title": "Column skew-T at T+0:00:00",
                    "url": "/api/runs/sc1/files/quicklook/slices/column-max-t000000.png"}
    assert client.get(card["url"]).status == 200
