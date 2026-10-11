"""The gallery's presets, their soundings, and the gallery's groups.

- the groups and their order are pinned, with the 1 km quick look as the first card; every offered
  preset imports with exit 0, nothing refused, nothing to acknowledge and nothing not yet checked;
- every shipped observed sounding is re-derived byte for byte from its stored archive response, with
  its provenance, rights, a top at or above 20 km and CM1's hydrostatic reconstruction within 1 hPa (one named
  exception); each famous preset carries its file byte for byte;
- every famous card carries the idealized-storm sentence and names the station and the date only;
- another sounding: ``preset fit|new --sounding FILE`` (fit.json records the file's SHA-256 and source, and the frame and
  bubble that follow its storm), ``--instability/--shear``, and the GUI's {"sounding"} / {"environment"}, each
  with its command and dry_run;
- every offered new preset has its own picture from a real run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from cm1core.cm1import.door import run_import
from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.gui import commands as C
from cm1core.gui import copy_lint as L

from cm1import_support import namelist_tool

ROOT = Path(__file__).resolve().parents[1]
SOUNDINGS = ROOT / "src/cm1core/data/soundings"
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"
IDEALIZED = "This day's sounding in an idealized storm, not a reconstruction of that day's storms."

#: The gallery, group by group, in order.
GALLERY = {
    "start": ["supercell-1km", "supercell-250m", "supercell-100m"],
    "famous": ["oun-2013-05-20", "oun-1999-05-03", "oun-2011-05-24", "bmx-2011-04-27", "oun-2019-05-20",
               "dnr-2015-06-01", "iln-2012-06-29", "tlh-2004-09-15", "jan-2014-06-30"],
    "explore": ["your-environment", "split-straight", "multicell-wk", "pulse-calm"],
    "other": ["line-open", "gust-front", "bubble-cloud-250m", "squall-line-1km", "squall-line-250m"],
}
NEW = [pid for group in ("famous", "explore", "other") for pid in GALLERY[group] if not pid.startswith("squall")]


def _library():
    from cm1core.cartesian_door import resolve_initializer_library

    try:
        return resolve_initializer_library(None)
    except Exception:  # noqa: BLE001 - no wheel natives on this host
        return None


needs_library = pytest.mark.skipif(_library() is None, reason="the initializer library is not installed")


@pytest.fixture(autouse=True)
def _parser(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def _rows(payload: dict, section: str) -> list[dict]:
    for block in payload.get("sections", []):
        if block["id"] == section:
            return block.get("rows", [])
    return []


# ------------------------------------------------------------------ the groups


def test_the_groups_their_order_and_the_first_card_are_pinned():
    presets = P.load_all()
    assert P.GROUPS == ("start", "explore", "other", "famous")
    by_group: dict[str, list[str]] = {}
    for preset in presets:
        by_group.setdefault(preset.group, []).append(preset.id)
    assert by_group == GALLERY
    assert [preset.id for preset in presets] == [pid for group in P.GROUPS for pid in GALLERY[group]]
    assert presets[0].id == "supercell-1km" and presets[0].name == "Supercell, quick look, 1 km"
    listed = P.list_record(check=False)
    assert listed["groups"] == list(P.GROUPS)
    assert [row["group"] for row in listed["presets"]] == [preset.group for preset in presets]


def test_the_gallery_opens_start_here_only_and_folds_the_rest():
    """The page's own rules (gallery.js): Start here never folds; the others show one row and Show all N."""

    source = (ROOT / "src/cm1core/gui/static/guided/gallery.js").read_text()
    assert 'const GROUPS = ["start", "explore", "other", "famous"];' in source
    assert 'const always = group === "start";' in source
    assert "export function registerCard(" in source
    copy = json.loads((ROOT / "src/cm1core/gui/copy/guided.json").read_text())["strings"]
    assert copy["gallery.show_all"]["text"] == "Show all {n}"
    assert [copy[f"gallery.group.{g}"]["text"] for g in P.GROUPS] == [
        "Start here", "Explore", "Other storm types", "Famous environments"]


def test_guided_keeps_its_four_screens():
    index = (ROOT / "src/cm1core/gui/static/guided/index.js").read_text()
    assert re.search(r'TITLES = \{ "": "gallery.title", fit: "budget.title", watch: "watch.title", '
                     r'results: "results.title" \}', index)
    registered = []
    for path in sorted((ROOT / "src/cm1core/gui/static").rglob("*.js")):
        registered += [path.name for _ in re.finditer(r'^register\([^;]*mode:\s*"guided"', path.read_text(), re.M)]
    assert registered == ["index.js"]


@pytest.mark.parametrize("preset_id", NEW)
def test_every_new_preset_imports_as_a_run_gm1_runs_as_cm1_does(preset_id, tmp_path):
    preset = P.load(preset_id)
    code, payload, _ = run_import(preset.namelist_path, tmp_path / preset_id, sounding=preset.sounding_path,
                                  overrides=preset.record.get("gm1_settings"))
    assert code == 0, payload.get("verdict_text")
    assert _rows(payload, "refused") == [] and _rows(payload, "not_read") == []
    assert payload["acknowledgments"]["needed"] == []
    assert _rows(payload, "not_checked") == []
    examined = P.examine(preset)
    withheld = preset.record.get("offered") is False
    assert examined["offered"] is not withheld, examined["why_not"]
    if withheld:
        assert examined["why_not"] == preset.record["not_offered_reason"]
    else:
        assert payload["guided"]["offered"] is True


def test_the_generator_matches_the_preset_files_and_every_change_has_its_reason():
    completed = subprocess.run([sys.executable, str(ROOT / "tools/presets/make_presets.py"), "--check"],
                               capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for preset in P.load_all():
        assert all(change["why"] for change in preset.record["source"]["changes"])
        if preset.record.get("gm1_settings"):
            assert preset.record.get("gm1_settings_why")


# ------------------------------------------------------------------ the soundings


def test_every_shipped_sounding_is_rederived_from_its_stored_response():
    completed = subprocess.run([sys.executable, str(ROOT / "tools/release/uwyo_to_input_sounding.py"), "ship",
                                "--check"], capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_shipped_sounding_has_its_provenance_and_passes_the_checks():
    index = json.loads((SOUNDINGS / "index.json").read_text())
    provenance = (SOUNDINGS / "PROVENANCE.md").read_text()
    names = [row["filename"] for row in index["soundings"]]
    assert len(names) == len(set(names)) == 9
    for row in index["soundings"]:
        data = (SOUNDINGS / row["filename"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"] and len(data) == row["bytes"]
        stored = ROOT / row["response_stored"]
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == row["response_sha256"]
        assert row["url"].startswith("https://weather.uwyo.edu/wsgi/sounding?") and "type=TEXT:LIST" in row["url"]
        assert "17 U.S.C. 105" in row["rights"] and row["retrieved_utc"]
        assert row["top_m_agl"] >= 20000.0
        assert row["hydrostatic_max_abs_hPa"] <= row["hydrostatic_limit_hPa"]
        if row["hydrostatic_limit_hPa"] > 1.0:
            assert row["filename"] == "koun-1999-05-03-1800z.input_sounding" and row["hydrostatic_exception"]
        assert row["filename"] in provenance and row["sha256"] in provenance and row["url"] in provenance
        assert row["selection"]
    assert "/home" not in provenance


def test_each_famous_preset_carries_its_balloon_byte_for_byte():
    index = {row["filename"]: row for row in json.loads((SOUNDINGS / "index.json").read_text())["soundings"]}
    for preset_id in GALLERY["famous"]:
        preset = P.load(preset_id)
        record = preset.record["sounding"]
        shipped = index[record["shipped_as"].split("/")[-1]]
        assert preset.sounding_sha256 == record["sha256"] == shipped["sha256"]
        assert record["source"] == shipped["url"] and record["rights"] == shipped["rights"]
        assert "isnd" in preset.namelist_text and re.search(r"^\s*isnd\s*=\s*7,", preset.namelist_text, re.M)


def test_the_made_soundings_are_the_generators_byte_for_byte():
    from cm1core.cm1import import analytic_sounding as A

    for preset in P.load_all():
        made = (preset.record.get("sounding") or {}).get("environment")
        if not made:
            continue
        again = A.generate(made["instability"], made["shear"], made["hodograph"])
        assert preset.sounding_text == again.text and preset.record["sounding"]["sha256"] == again.sha256
        assert preset.record["sounding"]["source"] == "gpuwm-cm1 " + " ".join(again.command())


# ------------------------------------------------------------------ the words


def test_every_famous_card_is_honest_about_what_it_is():
    for preset_id in GALLERY["famous"]:
        preset = P.load(preset_id)
        assert IDEALIZED in preset.record["note"], preset_id
        # the station, the local date and the launch: nothing about what happened that day
        assert re.fullmatch(r"[A-Z][a-z]+, [A-Z][a-z]+, \d{1,2} [A-Z][a-z]+ \d{4} \((midday|afternoon|evening) "
                            r"balloon\)", preset.name), preset.name
        if preset.record.get("offered") is False:
            # A greyed card says what was measured, never what set off that day's storms.
            reason = preset.record.get("not_offered_reason") or ""
            assert "measured" in reason and "that day's storms" not in reason.lower(), (preset_id, reason)
        else:
            # An offered card says plainly how its storm is started (CM1's updraft nudging) and what the
            # storm did in the run behind its picture, and runs iinit = 12.
            note = preset.record["note"]
            assert "a gentle push of rising air, as CM1's updraft nudging does" in note, preset_id
            assert "(measured)" in note or "(both measured)" in note, preset_id
            assert "that day's storms" not in note.replace(IDEALIZED, ""), preset_id
            assert ("iinit", 1 if preset_id not in ("iln-2012-06-29",) else 8, 12) in [
                (e["key"], e["from"], e["to"]) for e in preset.record["source"]["changes"]], preset_id
        assert "feeding" not in preset.record["line"] and "tropical storm" not in preset.record["note"], preset_id
    copy = json.loads((ROOT / "src/cm1core/gui/copy/guided.json").read_text())["strings"]
    assert copy["gallery.famous_note"]["text"] == IDEALIZED


def test_every_famous_cards_numbers_are_its_balloons():
    # The card's CAPE and shear are the shipped balloon's own numbers (index.json "parameters": surface-based
    # parcel on the measured profile), CAPE to the nearest 100 J/kg and shear to the nearest m/s (the
    # 20 May 2013 card once said 27 m/s for 26.49).
    index = {row["filename"]: row for row in json.loads((SOUNDINGS / "index.json").read_text())["soundings"]}
    for preset_id in GALLERY["famous"]:
        preset = P.load(preset_id)
        numbers = index[preset.record["sounding"]["shipped_as"].split("/")[-1]]["parameters"]
        line = preset.record["line"]
        cape = re.search(r"CAPE about (\d+) J/kg", line)
        if cape:
            assert int(cape.group(1)) == round(numbers["sbcape_j_kg"], -2), (preset_id, numbers["sbcape_j_kg"])
        shear6 = re.search(r"(\d+) m/s (?:of shear )?over 6 km", line)
        if shear6:
            assert int(shear6.group(1)) == round(numbers["shear_0_6km_m_s"]), (preset_id, numbers["shear_0_6km_m_s"])
        shear1 = re.search(r"(\d+) m/s in the lowest 1 km", line)
        if shear1:
            assert int(shear1.group(1)) == round(numbers["shear_0_1km_m_s"]), (preset_id, numbers["shear_0_1km_m_s"])
        assert cape or shear1, preset_id


def test_every_card_and_every_copy_file_passes_the_lint_with_the_damage_words():
    for word in ("damage", "destroyed", "killed", "fatal", "victims", "deadly"):
        assert L.lint_text(f"a {word} word") != [], word
    problems = L.lint_all()
    for preset in P.load_all():
        for field in ("name", "line", "note", "gm1_settings_why", "not_offered_reason"):
            if preset.record.get(field):
                problems += L.lint_text(preset.record[field], where=f"{preset.id}.{field}")
    assert problems == []


# ------------------------------------------------------------------ another sounding for a preset


def test_a_preset_that_makes_its_own_sounding_takes_no_file(tmp_path):
    other = tmp_path / "input_sounding"
    other.write_bytes((SOUNDINGS / "koun-2013-05-20-1800z.input_sounding").read_bytes())
    with pytest.raises(P.PresetError, match="makes its own sounding"):
        P.with_sounding(P.load("supercell-1km"), other)
    with pytest.raises(P.PresetError, match="no such file"):
        P.with_sounding(P.load("your-environment"), tmp_path / "missing")


@pytest.mark.parametrize("content, words", [
    (b"hello world\n", "is not three numbers"),
    (b"", "is empty"),
    (b"\xee\xff\x00binary", "is not text"),
    (None, "below the model's highest scalar level"),
])
def test_a_given_sounding_cm1_would_stop_on_is_refused_with_its_plain_reason(tmp_path, content, words):
    """A bad file gets CM1's own reason (cm1import.sounding), not the storm-track check's vaguer
    one, nor a decode error from deep in the fit."""

    bad = tmp_path / "input_sounding"
    if content is None:  # a real balloon cut at about 2 km, below the preset's 20 km model top
        lines = (SOUNDINGS / "koun-2013-05-20-1800z.input_sounding").read_text().splitlines()[:20]
        bad.write_text("\n".join(lines) + "\n")
    else:
        bad.write_bytes(content)
    with pytest.raises(P.PresetError, match=re.escape(words)):
        P.with_sounding(P.load("your-environment"), bad)


def test_the_made_environment_leaves_no_temporary_folder_behind(tmp_path):
    """Each slider move is a `preset fit --instability I --shear S`; its temporary files go when it ends."""

    script = (
        "import tempfile, sys\n"
        f"tempfile.tempdir = {str(tmp_path)!r}\n"
        "from cm1core.cm1run import presets as P\n"
        "made = P.with_environment(P.load('your-environment'), 0.4, 0.6)\n"
        "assert made.sounding_path.is_file()\n"
    )
    env_path = str(ROOT / "src")
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": env_path})
    assert done.returncode == 0, done.stderr[-2000:]
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("gm1-")] == []


@needs_library
def test_a_given_sounding_is_run_with_its_record_and_the_frame_that_follows_its_storm(tmp_path, capsys):
    from cm1core.cli import main

    other = tmp_path / "column" / "input_sounding"
    other.parent.mkdir()
    other.write_bytes((SOUNDINGS / "koun-2019-05-21-0000z.input_sounding").read_bytes())
    preset = P.with_sounding(P.load("your-environment"), other)
    record = P.sounding_record(preset)
    assert record["sha256"] == hashlib.sha256(other.read_bytes()).hexdigest() and record["given"] is True
    follow = preset.sounding_given["follow"]
    assert follow["sentence"].startswith("The view moves with the storm this sounding makes")
    assert set(follow["keys"]) == {"umove", "vmove", "initialization.bubble_center_x_m",
                                   "initialization.bubble_center_y_m"}
    assert P.given_args(preset) == ["--sounding", str(other.resolve())]
    runs = tmp_path / "runs"
    assert main(["preset", "new", "your-environment", "--budget", "10m", "--root", str(runs), "--name", "col",
                 "--sounding", str(other), "--json"]) in (0, 1)
    made = json.loads(capsys.readouterr().out)
    if made.get("ok"):
        fit = json.loads((runs / "col" / "fit.json").read_text())
        assert fit["sounding"]["sha256"] == record["sha256"] and fit["sounding"]["source"] == str(other.resolve())
        assert fit["sentences"][0] == follow["sentence"]
        assert (runs / "col" / "input_sounding").read_bytes() == other.read_bytes()
        assert "--sounding" in fit["command"]


def test_the_gui_commands_carry_the_sounding_flags():
    root = Path("/runs")
    assert C.preset_fit_args("your-environment", "10m", root, None, C.preset_given_args(sounding="/x/input_sounding")) \
        == ["preset", "fit", "your-environment", "--budget", "10m", "--sounding", "/x/input_sounding",
            "--root", "/runs", "--json"]
    given = C.preset_given_args(environment={"instability": 0.25, "shear": 1.0})
    assert given == ["--instability", "0.25", "--shear", "1", "--hodograph", "curved"]
    assert C.preset_new_args("your-environment", "1h", root, "n", None, given)[-7:-1] == given
    assert C.preset_given_args() == []


# ------------------------------------------------------------------ pictures from real runs


def test_every_offered_new_preset_has_its_own_picture():
    for preset_id in NEW:
        preset = P.load(preset_id)
        picture = preset.folder / "picture.png"
        assert picture.is_file(), preset_id
        record = json.loads((preset.folder / "picture.json").read_text())
        assert record["run"]["preset"] == preset_id and record["run"]["gpu"]
        assert "Simulated." in record["caption"]
        assert record["run"]["grid"]["dx_m"] and record["run"]["quicklook_sha256"]
        # the caption names the grid it was drawn at
        spacing = record["run"]["grid"]["dx_m"]
        words = f"{spacing / 1000:g} km" if spacing >= 1000 else f"{spacing:g} m"
        assert f"at {words}" in record["caption"]
    variants = P.variants(P.load("your-environment"))
    assert {row["id"] for row in variants} >= {"i0.5-s0.5"}
    for row in variants:
        assert 0.0 <= row["instability"] <= 1.0 and 0.0 <= row["shear"] <= 1.0


# ------------------------------------------------------------------ the GUI's endpoints


@pytest.fixture()
def live(tmp_path, monkeypatch):
    from gui_support import start_server, stop_server

    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(FIXTURES / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(FIXTURES / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    monkeypatch.setenv("GPUWM_CM1_HOST_FIXTURE", str(FIXTURES / "host-64gb-1tb.json"))
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        yield server, client, Path(server.root.path)
    finally:
        stop_server(server)


def test_the_gallery_lists_groups_and_serves_each_measured_setting(live):
    _, client, _ = live
    body = client.get("/api/presets").json
    assert body["groups"] == list(P.GROUPS)
    rows = {row["id"]: row for row in body["presets"]}
    assert rows["your-environment"]["card"] == "environment"
    for row in rows["your-environment"]["pictures"]:
        got = client.get(f"/api/presets/{row['picture']}/picture")
        assert got.status == 200 and got.raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get("/api/presets/your-environment.no-such/picture").status == 404


def test_fit_and_start_take_an_environment_or_a_sounding_with_their_commands(live, tmp_path):
    _, client, root = live
    environment = {"instability": 0.75, "shear": 0.25}
    dry = client.post("/api/fit", {"preset": "your-environment", "budget": "10m", "environment": environment,
                                   "dry_run": True})
    assert dry.status == 200 and dry.json["command"] == (
        f"gpuwm-cm1 preset fit your-environment --budget 10m --instability 0.75 --shear 0.25 --hodograph curved "
        f"--root {root} --json")
    reply = client.post("/api/fit", {"preset": "your-environment", "budget": "10m", "environment": environment})
    assert reply.status == 200 and reply.json["offered"], reply.raw
    made = reply.json["fit"]["sounding"]["environment"]
    assert made["instability"] == 0.75 and made["shear"] == 0.25 and made["sentence"]
    assert "--instability 0.75 --shear 0.25" in reply.json["start"]["commands"][0]
    column = tmp_path / "column" / "input_sounding"
    column.parent.mkdir()
    column.write_bytes((SOUNDINGS / "koun-2013-05-20-1800z.input_sounding").read_bytes())
    dry = client.post("/api/guided/start", {"preset": "your-environment", "budget": "10m", "sounding": str(column),
                                            "dry_run": True})
    assert dry.status == 200 and f"--sounding {column}" in dry.json["command"]
    for bad in ({"environment": {"instability": 2, "shear": 0.5}}, {"environment": {"instability": 0.5}},
                {"environment": {"instability": 0.5, "shear": 0.5, "hodograph": "zigzag"}},
                {"sounding": "relative/input_sounding"}, {"sounding": str(tmp_path / "missing")},
                {"sounding": str(column), "environment": environment}):
        answer = client.post("/api/fit", {"preset": "your-environment", "budget": "10m", **bad})
        assert answer.status == 400, (bad, answer.raw)
        assert answer.json["message"] and answer.json["fix"]
