"""Guided mode's "Use my own sounding" card through the GUI server, on the fake runner. CPU only.

- the card's words, the shipped soundings it lists, and its place in the page (static files);
- POST /api/sounding/check: a pasted listing, a shipped sounding and a typed path are each one
  `gpuwm-cm1 sounding convert` command whose record the reply carries; "Show command" (dry_run) writes nothing;
  a refusal says why with the line, and offers what would fix it (made-up levels, the wind speed's unit);
- the fit and Start carry the kept file and the chosen start; the run is named from the file's name, the Runs list
  says "Your sounding: NAME", the run folder keeps the sounding (and the user's own file when it was converted),
  and Results says whose air it is;
- the Customize choices of the storm started by the push (a push in m/s, no bubble size).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.gui import commands as C
from cm1core.gui import copy_lint as L
from cm1core.gui import sounding as GS
from gui_support import start_server, stop_server, wait_for
from test_gui_advanced import Walk, _environment, _mixed, _shim  # noqa: F401
from cm1import_support import namelist_tool
import own_sounding_support as S

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"
STATIC = ROOT / "src" / "cm1core" / "gui" / "static"


@pytest.fixture(autouse=True)
def _fake_gpu(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(FIXTURES / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(FIXTURES / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", str(FIXTURES / "host-64gb-1tb.json"))
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.004")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.2")
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


@pytest.fixture()
def live(tmp_path):
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        yield server, client, Path(server.root.path)
    finally:
        stop_server(server)


# ------------------------------------------------------------------ the page's own files


def test_the_cards_words_pass_the_copy_lint_and_every_key_the_page_uses_exists():
    copy = GS.load_copy()
    assert copy["strings"]["card.title"]["text"] == "Use my own sounding"
    assert L.lint_file(GS.COPY_FILE) == []
    used = set()
    for name in ("card.js",):
        text = (STATIC / "js" / "sounding" / name).read_text(encoding="utf-8")
        used |= set(re.findall(r"""\bt\(\s*["']([a-z_.0-9]+)["']""", text))
        used |= {f"found.speed_{unit}" for unit in ("kt", "ms")}
    missing = sorted(key for key in used if key not in copy["strings"])
    assert not missing, missing
    assert L.lint_file(STATIC / "js" / "sounding" / "card.js") == []


def test_the_card_is_loaded_by_the_page_and_joins_the_gallery():
    assert 'import "./sounding/index.js";' in (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert '/static/css/sounding.css' in (STATIC / "index.html").read_text(encoding="utf-8")
    index = (STATIC / "js" / "sounding" / "index.js").read_text(encoding="utf-8")
    assert 'group: "explore"' in index and 'id: "own-sounding"' in index
    budget = (STATIC / "guided" / "budget.js").read_text(encoding="utf-8")
    assert 'given.start = params.get("start")' in budget and "startOptions(p.id, ownStart)" in budget
    assert "/api/sounding/options" in (STATIC / "js" / "sounding" / "options.js").read_text(encoding="utf-8")


def test_a_file_name_from_the_browser_is_made_plain():
    assert GS.safe_name("Norman balloon (no unit).csv") == "Norman-balloon-no-unit.csv"
    assert GS.safe_name("D:\\soundings\\oun.txt") == "oun.txt"
    assert GS.safe_name("../../etc/passwd") == "passwd"
    assert GS.safe_name("") == "sounding.txt" and GS.safe_name("///") == "sounding.txt"


def test_the_commands_the_card_runs():
    args = C.sounding_convert_args(Path("/in/x.txt"), Path("/out/input_sounding"), speed_unit="kt", extend=True,
                                   top_of=Path("/p/namelist.input"), name="x.txt", preview_png=Path("/out/p.png"))
    assert C.display(args) == ("gpuwm-cm1 sounding convert /in/x.txt /out/input_sounding --speed-unit kt "
                               "--extend-top --top-of /p/namelist.input --name x.txt --preview-png /out/p.png --json")
    assert C.preset_options_args("your-environment", "push") == ["preset", "options", "your-environment",
                                                                 "--start", "push", "--json"]
    actions = {endpoint.action for endpoint in C.API}
    assert {"sounding-check", "sounding-options"} <= actions


# ------------------------------------------------------------------ the server


def test_the_card_lists_the_shipped_balloon_soundings(live):
    server, client, root = live
    body = client.get("/api/sounding/shipped").json
    index = json.loads((ROOT / "src/cm1core/data/soundings/index.json").read_text(encoding="utf-8"))
    assert {row["id"] for row in body["soundings"]} == {row["filename"] for row in index["soundings"]}
    assert body["preset"] == "your-environment" and body["wsl"] is True and body["starts"] == list(GS.STARTS)
    first = body["soundings"][0]
    assert first["title"] == "Norman, Oklahoma, 2019-05-21 0000 UTC" and "path" not in first
    assert client.get("/api/sounding/copy").json["strings"]["next.button"]["text"] == "Next: how long can it run?"


def test_a_pasted_listing_is_checked_converted_and_kept(live):
    server, client, root = live
    text = S.wyoming(S.measured(), knots=True)
    request = {"text": text, "name": "My balloon (test).txt"}
    dry = client.post("/api/sounding/check", {**request, "dry_run": True}).json
    assert dry["dry_run"] is True and dry["command"].startswith("gpuwm-cm1 sounding convert ")
    assert not (root / ".gm1-gui" / "soundings").exists()          # asking for the command leaves nothing behind
    reply = client.post("/api/sounding/check", request).json
    assert reply["ok"] is True and reply["checked"] is True and reply["command"] == dry["command"]
    assert reply["format"] == "wyoming" and reply["converted"] is True and reply["label"] == "My-balloon-test"
    assert reply["name"] == "My-balloon-test.txt" and any("knots" in line for line in reply["conversions"])
    kept = Path(reply["sounding"])
    assert kept.is_file() and kept.parent.parent == root / ".gm1-gui" / "soundings"
    assert kept.read_text(encoding="ascii") == reply["text"]
    assert reply["start"]["recommended"] == "bubble" and len(reply["start"]["options"]) == 3
    assert reply["route"] == GS.SoundingApi.route(kept, "bubble")
    assert reply["route"].startswith("guided/fit/your-environment/10m?sounding=") and reply["route"].endswith("&start=bubble")
    assert "--start bubble" in reply["fit_command"]
    if reply.get("numbers"):                                        # with the initializer library installed
        assert reply["numbers"]["levels"] == {"nz": 40, "dz_m": 500.0} and reply["verdict"]["sentence"]
    if (reply.get("preview") or {}).get("url"):
        picture = client.get(reply["preview"]["url"])
        assert picture.status == 200 and picture.raw[:8] == b"\x89PNG\r\n\x1a\n"
    download = client.get(reply["download"])
    assert download.status == 200 and download.raw == kept.read_bytes()
    assert client.get("/api/sounding/files/../../x").status in (400, 404)
    assert client.get(f"/api/sounding/files/{reply['key']}/sounding.json").status in (400, 403, 404)


def test_a_refusal_says_why_and_offers_what_would_fix_it(live):
    server, client, root = live
    short = client.post("/api/sounding/check", {"text": S.table(S.measured(top_m=12000.0)), "name": "short.csv"}).json
    assert short["ok"] is False and short["checked"] is True and "sounding" not in short
    assert short["problems"][0]["code"] == "top" and short["problems"][0]["offer"] == {"extend_to_m": 20500.0}
    assert short["message"].startswith("The sounding stops at 12.0 km") and short["text"]
    longer = client.post("/api/sounding/check", {"text": S.table(S.measured(top_m=12000.0)), "name": "short.csv",
                                                 "extend": True}).json
    assert longer["ok"] is True and longer["extension"]["extended"] is True and "--extend-top" in longer["command"]
    assert longer["key"] != short["key"]
    table = S.table(S.measured(), header="pressure,height,temperature,dewpoint,direction,speed")
    asked = client.post("/api/sounding/check", {"text": table, "name": "t.csv"}).json
    assert asked["ok"] is False and asked["problems"][0]["offer"] == {"speed_unit": ["kt", "ms"]}
    assert client.post("/api/sounding/check", {"text": table, "name": "t.csv", "speed_unit": "kt"}).json["ok"] is True
    lines = S.made().splitlines()
    celsius = client.post("/api/sounding/check", {
        "text": lines[0].replace("300.000000", " 27.000000") + "\n" + "\n".join(lines[1:]) + "\n"}).json
    assert celsius["ok"] is False and celsius["problems"][0]["line"] == 1
    assert "looks like a temperature in C" in celsius["message"] and celsius["fix"]
    for body, status in (({}, 400), ({"text": "x", "shipped": "y"}, 400), ({"shipped": "nope"}, 400),
                         ({"path": "/no/such/file.txt"}, 400), ({"text": "x", "speed_unit": "mph"}, 400),
                         ({"text": 5}, 400)):
        answer = client.post("/api/sounding/check", body)
        assert answer.status == status, (body, answer.raw)
        assert answer.json["message"] and answer.json["fix"]


def test_a_shipped_sounding_and_a_typed_path(live, tmp_path):
    server, client, root = live
    shipped = client.post("/api/sounding/check", {"shipped": "koun-2013-05-20-1800z.input_sounding"}).json
    assert shipped["ok"] is True and shipped["format"] == "cm1" and shipped["converted"] is False
    assert shipped["shipped"].startswith("NWS rawinsonde, Norman, Oklahoma")
    assert shipped["start"]["recommended"] == "push" and shipped["route"].endswith("&start=push")
    kept = Path(shipped["sounding"])
    assert kept.read_bytes() == (ROOT / "src/cm1core/data/soundings/koun-2013-05-20-1800z.input_sounding").read_bytes()
    assert json.loads((kept.parent / "sounding.json").read_text(encoding="utf-8"))["shipped"] == shipped["shipped"]
    mine = tmp_path / "typed.snd"
    mine.write_text(S.sharppy(S.measured()), encoding="ascii")
    typed = client.post("/api/sounding/check", {"path": str(mine)}).json
    assert typed["ok"] is True and typed["format"] == "sharppy" and typed["label"] == "typed"
    assert f"sounding convert {mine} " in typed["command"]


def test_the_storm_is_fitted_started_named_and_kept(live):
    server, client, root = live
    text = S.wyoming(S.measured(), knots=True)
    checked = client.post("/api/sounding/check", {"text": text, "name": "My balloon.txt"}).json
    sounding = checked["sounding"]
    request = {"preset": "your-environment", "budget": "10m", "sounding": sounding, "start": "push"}
    fit = client.post("/api/fit", request).json
    assert fit["offered"] is True, fit
    assert "--sounding" in fit["command"] and "--start push" in fit["command"]
    record = fit["fit"]
    assert record["sounding"]["label"] == "My-balloon" and record["sounding"]["own"]["format"] == "wyoming"
    assert record["sounding"]["source"] == ("your sounding My-balloon.txt, converted from University of Wyoming "
                                            "text listing")
    assert any(change["knob"] == "start" for change in record["customized"])
    # the storm-track check's own answer, for a warm bubble there: said so, since the push's storm can move another way
    assert "push of rising air starts" in record["sentences"][0]
    assert "worked out for a warm bubble" in record["sentences"][0]
    assert fit["start"]["name"].startswith("My-balloon-")
    assert client.post("/api/fit", {**request, "start": "sideways"}).status == 400
    assert client.post("/api/fit", {"preset": "your-environment", "budget": "10m", "start": "push",
                                    "environment": {"instability": 0.5, "shear": 0.5}}).status == 400
    started = client.post("/api/guided/start", {**request, "name": fit["start"]["name"]})
    assert started.status == 200, started.raw
    name = started.json["run"]
    assert name == fit["start"]["name"]
    rundir = root / name
    run = json.loads((rundir / "run.json").read_text(encoding="utf-8"))
    assert run["case"] == "Your sounding: My-balloon" and run["guided"]["sounding"] == "My-balloon"
    assert (rundir / "input_sounding").read_text(encoding="ascii") == checked["text"]
    assert (rundir / "input_sounding.original").read_text(encoding="ascii") == text
    assert "iinit     =  12" in (rundir / "namelist.input").read_text(encoding="ascii")
    listed = wait_for(lambda: [row for row in client.get("/api/runs").json["runs"] if row["name"] == name])
    assert listed[0]["case"] == "Your sounding: My-balloon"
    wait_for(lambda: client.get(f"/api/runs/{name}").json["status"]["state"] in ("finished", "stopped", "failed"),
             timeout=120)
    results = client.get(f"/api/runs/{name}/results?mode=guided").json["guided"]
    assert results["sounding"]["label"] == "My-balloon"
    assert "The air is your sounding My-balloon.txt, converted from University of Wyoming text listing." in results["this_run"]
    assert "input_sounding.original" in results["this_run"]


def test_the_customize_choices_follow_the_start(live):
    server, client, root = live
    own = client.post("/api/sounding/options", {"start": "push"}).json
    assert own["command"] == "gpuwm-cm1 preset options your-environment --start push --json"
    knobs = {knob["id"]: knob for knob in own["customize"]["knobs"]}
    assert knobs["push"]["unit"] == "m/s" and knobs["push"]["kind"] == "updraft" and "bubble" not in knobs
    bubble = {knob["id"]: knob for knob in client.post("/api/sounding/options", {"start": "bubble"}).json["customize"]["knobs"]}
    assert bubble["push"]["unit"] == "K" and "bubble" in bubble
    assert client.post("/api/sounding/options", {"start": "auto"}).status == 400


def test_the_sounding_actions_walk_the_api_table(tmp_path, monkeypatch):
    """Each of the card's two actions as the page makes it: "Show command" names the line the press then runs, and
    asking for it changes no file (the Walk of tests/test_gui_advanced.py)."""

    server, client = _mixed(tmp_path, monkeypatch)
    root = Path(server.root.path)
    scratch = tmp_path / "walk"
    scratch.mkdir()
    walk = Walk(server, client, root, _shim(tmp_path, "mixed"), scratch)
    try:
        checked = walk.post("sounding-check", "/api/sounding/check",
                            {"text": S.wyoming(S.measured()), "name": "walked.txt"}, run=None)
        assert checked["ok"] is True and checked["format"] == "wyoming" and Path(checked["sounding"]).is_file()
        options = walk.post("sounding-options", "/api/sounding/options", {"start": "push"}, run=None)
        assert options["customize"]["preset"] == "your-environment"
    finally:
        stop_server(server)


# ------------------------------------------------------------------ what testing found (each fixed, each held here)


def test_the_page_does_what_testing_asked_of_it():
    """The browser side, read from its files (a headless browser walked each of these by hand):
    - Customize does not offer the made-air sliders for a sounding file (they changed nothing);
    - a refused or empty file clears what an earlier check showed, so Next cannot run the earlier sounding;
    - every line left out is listed; a missing EL is a dash with a sentence; the picture opens full size;
    - Next keeps the check for Back, and Start or Close forgets it."""

    budget = (STATIC / "guided" / "budget.js").read_text(encoding="utf-8")
    assert 'if (given.sounding && p.customize && p.customize.air)' in budget
    assert 'air: { available: false, why: t("budget.file_air") }' in budget
    assert 'sessionStorage.removeItem("gm1.sounding.back")' in budget
    card = (STATIC / "js" / "sounding" / "card.js").read_text(encoding="utf-8")
    for refusal in ('refuse(t("file.too_large"), t("file.too_large_fix"))', 'refuse(t("file.empty"), t("file.empty_fix"))',
                    'refuse(t("file.not_read"), "")', 'refuse(t("file.none"), "")'):
        assert refusal in card, refusal
    assert "function reset()" in card and "state.record = null;" in card
    assert "chosenName" not in card and "found.dropped_lines" in card and "n.el_note" in card
    assert 'h("a", { href: r.preview.url, target: "_blank"' in card
    assert "remember(state.request, state.start);" in card and "forget(); fold(false)" in card
    assert 'panel.querySelector("input[type=file]")' in card
    guided = json.loads((ROOT / "src/cm1core/gui/copy/guided.json").read_text(encoding="utf-8"))["strings"]
    assert "Make your own environment" not in guided["budget.file_air"]["text"]


def test_an_empty_file_a_folder_and_a_windows_path_off_wsl_say_so(live, tmp_path, monkeypatch):
    server, client, root = live
    empty = client.post("/api/sounding/check", {"text": "", "name": "balloon.txt"})
    assert empty.status == 400 and empty.json["message"] == "The file balloon.txt is empty."
    folder = client.post("/api/sounding/check", {"path": str(tmp_path)})
    assert folder.status == 400 and folder.json["message"].endswith("is a folder.")
    assert "file inside it" in folder.json["fix"]
    monkeypatch.setenv("GPUWM_CM1_WSL2", "0")
    windows = client.post("/api/sounding/check", {"path": "Q:\\soundings\\none.txt"})
    assert windows.status == 400 and "Windows path" in windows.json["message"]
    assert "Choose a file" in windows.json["fix"]
    mine = tmp_path / "typed.snd"
    mine.write_text(S.sharppy(S.measured()), encoding="ascii")
    odd = client.post("/api/sounding/check", {"path": str(mine), "name": "--top-m"}).json
    assert odd["ok"] is True and "--name top-m" in odd["command"]          # a name is a plain name


def test_the_download_is_named_from_the_file_and_a_pasted_cm1_file_with_a_blank_line_runs(live):
    server, client, root = live
    reply = client.post("/api/sounding/check", {"text": S.made() + "\n\n", "name": "made air.txt"}).json
    assert reply["ok"] is True and reply["tidied"] is True and reply["text"] == S.made()
    assert any("blank line(s)" in note for note in reply["notes"])
    download = client.get(reply["download"])
    assert download.status == 200
    assert download.headers["content-disposition"] == 'attachment; filename="made-air.input_sounding"'


def test_no_storm_from_a_bubble_on_the_users_own_air_points_to_the_push(tmp_path, monkeypatch):
    from cm1core.gui import storm

    rundir = tmp_path / "run"
    rundir.mkdir()
    (rundir / "run.json").write_text("{}", encoding="utf-8")
    fit = {"sounding": {"label": "mine", "own": {"verdict": {"id": "supercell"}}}}
    (rundir / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    (rundir / "namelist.input").write_text(" &param2\n iinit     =  1,\n /\n", encoding="ascii")
    from cm1core.gui.guided import load_copy

    ended = load_copy()["guided"]["strings"]["storm.none_ended"]["text"]
    monkeypatch.setattr(storm, "notes", lambda *args, **kwargs: {600.0: ended})
    said = storm.run_notes(rundir, {"products": [1]})[600.0]
    assert said.startswith(ended) and "Try the push of rising air" in said
    (rundir / "namelist.input").write_text(" &param2\n iinit     =  12,\n /\n", encoding="ascii")
    assert storm.run_notes(rundir, {"products": [1]})[600.0] == ended          # the push: the air's own answer
    (rundir / "namelist.input").write_text(" &param2\n iinit     =  1,\n /\n", encoding="ascii")
    fit["sounding"]["own"]["verdict"]["id"] = "stable"
    (rundir / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    assert storm.run_notes(rundir, {"products": [1]})[600.0] == ended          # stable air: no storm is expected
