"""Customize this storm: a guided preset's settings changed without leaving guided mode.

Every choice goes through the ordinary guided path (``presets.examine``, the fit, the import), so these tests ask
that path, not a copy of it: each offered value is offered by the guided check and fits; each greyed choice's words
are the check's own; a changed setting is pinned and the fit never changes it; and the three gaps found while
designing the panel stay closed (a changed namelist kept when a sounding is given, random bumps priced, the area
floor checked).  CPU only, with fake GPUs; the real native namelist parser and import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.cm1run import customize as C
from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.cm1run.gpuinfo import GpuInfo, HostInfo

from cm1import_support import namelist_tool

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"
MIB = 1024 * 1024
HOST = HostInfo(64 * 10**9, 60 * 10**9, "/runs", 2000 * 10**9, 1000 * 10**9, 16)


def card(name: str, total_mib: int, used_bytes: float, cc: str = "8.9") -> GpuInfo:
    used = int(round(used_bytes / MIB)) * MIB
    return GpuInfo(0, "GPU-fake", name, total_mib * MIB, used, total_mib * MIB - used, 0, cc, "610", None, (), False)


HEADLESS_16GB = card("NVIDIA GeForce RTX 4080", 16376, 0.004e9)
CARD_32GB = card("NVIDIA GeForce RTX 5090", 32607, 0.002e9, cc="12.0")
WEISMAN_KLEMP = ("supercell-1km", "supercell-250m", "multicell-wk")
GENERATED = ("your-environment", "split-straight", "pulse-calm")
DRIFT = "The storm would drift too close to the side of the area before the run ends."


@pytest.fixture(autouse=True)
def _parser_and_cache(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def offered_presets() -> list[str]:
    return [preset.id for preset in P.load_all() if P.examine(preset)["offered"]]


def knobs(preset: P.Preset) -> dict[str, dict]:
    return {knob["id"]: knob for knob in C.options(preset)["knobs"]}


def cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})}
    return subprocess.run([sys.executable, "-m", "cm1core", *args], capture_output=True, text=True,
                          env=environment, timeout=300, cwd=ROOT)


def fixture_env(name: str) -> dict:
    return {"GPUWM_CM1_NVIDIA_SMI": str(FIXTURES / "fake_nvidia_smi.py"),
            "FAKE_NVIDIA_SMI_FIXTURE": str(FIXTURES / f"{name}.json"), "GPUWM_CM1_WSL2": "0",
            "GPUWM_CM1_HOST_FIXTURE": str(FIXTURES / "host-64gb-1tb.json")}


# ------------------------------------------------------------------ the choices


@pytest.mark.parametrize("preset_id", ["supercell-1km", "supercell-250m", "oun-2013-05-20", "bubble-cloud-250m",
                                       "multicell-wk", "line-open"])
def test_every_default_is_the_presets_own_value(preset_id):
    preset = P.load(preset_id)
    facts = P.examine(preset)["facts"]
    by_id = knobs(preset)
    assert by_id["grid"]["default"] == facts["dx_m"]
    assert by_id["levels"]["default"] == facts["dz_m"]
    assert math.isclose(by_id["area"]["default"], max(facts["domain_km"]))
    assert math.isclose(by_id["time"]["default"] * 60.0, facts["duration_s"])
    assert math.isclose(by_id["pictures"]["default"] * 60.0, facts["output_interval_s"])
    assert by_id["scheme"]["default"] == "morrison" and by_id["friction"]["default"] == "off"
    assert by_id["rotation"]["default"] == "off" and by_id["bumps"]["default"] == "off"
    # Nothing changed, nothing applied: the preset itself comes back.
    assert C.apply(preset, {}) is preset


def test_the_list_carries_each_presets_choices():
    record = P.list_record()
    rows = {row["id"]: row for row in record["presets"]}
    assert rows["supercell-1km"]["customize"]["schema"] == C.OPTIONS_SCHEMA
    assert [knob["id"] for knob in rows["supercell-1km"]["customize"]["knobs"]] == [
        knob for knob in C.KNOB_IDS if knob in {k["id"] for k in rows["supercell-1km"]["customize"]["knobs"]}]


def test_the_air_is_offered_only_where_the_two_sliders_can_make_it():
    for preset in P.load_all():
        air = C.options(preset)["air"]
        if preset.id in WEISMAN_KLEMP or preset.id in GENERATED:
            assert air["available"], preset.id
        if preset.group == "famous" or preset.id in ("gust-front", "line-open"):
            assert not air["available"] and air["why"], preset.id
    assert C.options(P.load("oun-2013-05-20"))["air"]["why"] == C.AIR_FAMOUS
    assert C.options(P.load("gust-front"))["air"]["why"] == C.AIR_WIND


@pytest.mark.parametrize("knob, value", sorted(C.GREYED))
def test_a_greyed_choice_says_the_guided_checks_own_reason(knob, value):
    preset = P.load("supercell-1km")
    reason = C.GREYED[(knob, value)]
    examined = P.examine(C.apply(preset, {knob: value}, greyed_too=True))
    assert not examined["offered"] and reason in examined["reasons"]
    choice = next(c for c in knobs(preset)[knob]["choices"] if c["value"] == value)
    assert choice["available"] is False and choice["why"] == reason
    with pytest.raises(C.CustomError, match="You can still choose it in Advanced"):
        C.apply(preset, {knob: value})


def test_the_time_step_follows_the_grid_and_lands_on_every_picture_interval():
    for dx in C.GRIDS_M:
        for dz in C.LEVELS_M:
            dt = C.time_step(dx, dz)
            assert dt <= 0.006 * dx + 1e-9 and dt <= 0.012 * dz + 1e-9
            assert math.isclose(60.0 / dt, round(60.0 / dt)), (dx, dz, dt)
            for minutes in C.PICTURES_MIN:
                assert math.isclose(minutes * 60.0 / dt, round(minutes * 60.0 / dt)), (dx, dz, minutes)
    assert C.time_step(1000.0, 500.0) == 6.0 and C.time_step(250.0, 250.0) == 1.5
    assert C.time_step(1000.0, 250.0) == 3.0 and C.time_step(2000.0, 500.0) == 6.0


def _singles(preset: P.Preset) -> list[dict]:
    """Each offered value of each knob alone, and a few together (the storm times: the shortest, the longest)."""

    by_id = knobs(preset)
    out: list[dict] = []
    for knob_id, knob in by_id.items():
        if knob_id == "area":
            out += [{"area": knob["min_km"]}, {"area": knob["max_km"]}]
            continue
        values = [c["value"] for c in knob["choices"] if c.get("available", True) and c["value"] != knob["default"]]
        if knob_id == "time":
            values = [min(values), max(values)] if values else []
        if knob_id == "bumps":
            values = ["on", 4242]
        out += [{knob_id: value} for value in values]
    out.append({"grid": 500.0, "rotation": "on", "parcels": "on", "bumps": "on"})
    return out


@pytest.mark.parametrize("preset_id", ["supercell-1km", "supercell-250m", "multicell-wk", "your-environment",
                                       "oun-2013-05-20", "jan-2014-06-30", "bubble-cloud-250m", "pulse-calm"])
def test_every_offered_value_is_offered_by_the_guided_check_and_fits_overnight_on_a_32gb_card(preset_id):
    preset = P.load(preset_id)
    if not P.examine(preset)["offered"]:
        pytest.skip(f"{preset_id} is not offered on this build")
    problems = []
    fitted = 0
    for custom in _singles(preset):
        try:
            changed = C.apply(preset, custom)
        except C.CustomError as error:
            # Only the parcels' 120 km floor may turn a single choice down here, with its plain reason.
            assert "at least 120 km" in str(error), (custom, error)
            continue
        examined = P.examine(changed)
        if not examined["offered"]:
            # The one refusal a size or time choice may meet is the storm-track check's own: on a small area or
            # over a long storm time the storm can leave the area even with the view following it (the panel
            # shows this reason and "Let GM1 choose the ..." for the choice).
            if set(examined["reasons"]) != {DRIFT} or not (custom.keys() & {"grid", "time", "area"}):
                problems.append((custom, examined["reasons"]))
            continue
        result = P.fit(changed, "overnight", gpu=CARD_32GB, host=HOST)
        if not result.offered:
            problems.append((custom, result.reason))
            continue
        fitted += 1
        # A pinned setting is what the user chose.
        values = P.examine(changed)["facts"]
        assert result.namelist_text is not None
        for knob, key in (("grid", "dx_m"), ("time", "duration_s"), ("pictures", "output_interval_s")):
            if knob in custom:
                fitted_value = result.estimate.grid["dx_m"] if key == "dx_m" else (
                    result.estimate.duration_s if key == "duration_s" else result.estimate.history_interval_s)
                assert math.isclose(fitted_value, values[key]), (custom, knob)
    assert not problems, problems
    assert fitted >= 0.75 * len(_singles(preset)), fitted     # nearly every choice fits as it is


# ------------------------------------------------------------------ pins


def test_a_finer_grid_in_ten_minutes_keeps_its_spacing_and_shrinks_the_area():
    preset = C.apply(P.load("supercell-1km"), {"grid": 250.0})
    assert preset.record["pinned"] == ["grid"]
    result = P.fit(preset, "10m", gpu=HEADLESS_16GB, host=HOST)
    assert result.offered, result.reason
    assert result.estimate.grid["dx_m"] == 250.0
    assert result.estimate.grid["nx"] < 480
    assert any(sentence.startswith("Made the area") for sentence in result.sentences)


def test_with_the_area_and_time_pinned_too_it_is_not_offered_and_names_a_longer_time():
    preset = C.apply(P.load("supercell-1km"), {"grid": 250.0, "area": 120.0, "time": 120.0})
    result = P.fit(preset, "10m", gpu=HEADLESS_16GB, host=HOST)
    assert not result.offered
    assert "pick a longer time than about 10 minutes" in result.reason
    assert result.reason.endswith("or let GM1 choose the grid spacing, area and storm time.")
    assert result.needs["pinned"] == ["grid", "area", "time"]
    longer = P.fit(preset, "overnight", gpu=HEADLESS_16GB, host=HOST)
    assert longer.offered and longer.estimate.grid["nx"] == 480 and longer.estimate.duration_s == 7200.0


def test_a_pinned_storm_time_is_never_shortened():
    for budget in ("10m", "1h"):
        preset = C.apply(P.load("supercell-1km"), {"grid": 250.0, "area": 120.0, "time": 120.0})
        result = P.fit(preset, budget, gpu=CARD_32GB, host=HOST)
        if result.offered:
            assert result.estimate.duration_s == 7200.0
    unpinned = C.apply(P.load("supercell-1km"), {"grid": 250.0, "area": 120.0})
    result = P.fit(unpinned, "10m", gpu=HEADLESS_16GB, host=HOST)
    if result.offered:
        assert result.estimate.duration_s <= 7200.0


def test_a_pinned_picture_interval_on_a_small_disk_gives_the_disk_reason():
    small = HostInfo(64 * 10**9, 60 * 10**9, "/runs", 100 * 10**9, int(1.0e9), 16)
    preset = C.apply(P.load("supercell-1km"), {"pictures": 1.0})
    result = P.fit(preset, "overnight", gpu=CARD_32GB, host=small)
    assert not result.offered
    assert "of free disk even at its smallest size" in result.reason
    assert result.reason.endswith("Or let GM1 choose the picture interval.")
    free = P.fit(P.load("supercell-1km"), "overnight", gpu=CARD_32GB, host=small)
    assert free.offered and free.estimate.history_interval_s > 300.0   # the fit widened it when it could


# ------------------------------------------------------------------ the three gaps


def test_parcels_raise_the_area_floor_to_cm1s_parcel_grid_and_a_smaller_start_is_refused_in_words():
    preset = P.load("supercell-1km")
    with pytest.raises(C.CustomError, match="at least 120 km across because followed air parcels"):
        C.apply(preset, {"area": 96.0, "parcels": "on"})
    on = C.apply(preset, {"parcels": "on"})
    assert on.record["min_width_km"] == {"default": 120.0}
    result = P.fit(C.apply(preset, {"parcels": "on", "grid": 250.0}), "10m", gpu=HEADLESS_16GB, host=HOST)
    if result.offered:
        assert result.estimate.grid["nx"] * 250.0 >= 120000.0 - 1e-6
    assert C.apply(preset, {"area": 96.0}).record["pinned"] == ["area"]     # without parcels, 64 km is the floor


def test_random_bumps_are_priced_as_the_import_makes_them():
    preset = C.apply(P.load("supercell-1km"), {"bumps": "on"})
    assert "irandp" in preset.namelist_text and P.examine(preset)["offered"]
    result = P.fit(preset, "10m", gpu=HEADLESS_16GB, host=HOST)
    assert result.offered, result.reason
    assert "initialization.random_generator" not in result.gm1_settings      # the import's own choice, not ours
    seeded = C.apply(P.load("supercell-1km"), {"bumps": 4242})
    assert seeded.record["random_seed"] == 4242 and P.examine(seeded)["offered"]
    assert P.fit(seeded, "10m", gpu=HEADLESS_16GB, host=HOST).offered


def test_air_on_a_weisman_klemp_preset_follows_the_storm_this_air_makes():
    preset = P.with_environment(P.load("supercell-1km"), 0.2, 0.2, "curved")
    assert preset.namelist_file is not None and "isnd" in preset.namelist_text
    effective = C._effective(preset)
    assert int(effective["isnd"]) == 7 and int(effective["iwnd"]) == 0
    track = P.storm_track_for(preset)
    pair = (track or {}).get("recommended_pair") or {}
    assert pair.get("uses_it"), pair            # the frame and bubble applied are the check's own for this air
    follow = preset.sounding_given["follow"]
    assert follow["keys"]["umove"][1] == float(effective["umove"])
    assert P.examine(preset)["offered"]
    assert P.fit(preset, "10m", gpu=HEADLESS_16GB, host=HOST).offered
    assert preset.record["customized"][-1]["knob"] == "air"


def test_with_sounding_keeps_a_changed_namelist(tmp_path):
    preset = C.apply(P.load("your-environment"), {"area": 160.0})
    sounding = tmp_path / "input_sounding"
    sounding.write_bytes((P.load("your-environment").folder / "input_sounding").read_bytes())
    given = P.with_sounding(preset, sounding)
    assert given.namelist_original_text == P.load("your-environment").namelist_text
    assert "160" in given.namelist_text and given.record["custom"] == {"area": 160.0}


# ------------------------------------------------------------------ the command line and the run folder


def test_an_unknown_or_unoffered_setting_exits_2_with_the_reason():
    for args, words in ((["--custom", "grid=300"], "is not one of the choices"),
                        (["--custom", "colour=red"], "There is no setting called colour"),
                        (["--custom", "time"], "--custom takes KNOB=VALUE")):
        done = cli("preset", "fit", "supercell-1km", "--budget", "10m", *args, env=fixture_env("rtx5090-32gb"))
        assert done.returncode == 2 and words in done.stderr, (args, done.stderr)


def test_preset_options_lists_the_choices():
    done = cli("preset", "options", "supercell-1km", "--json")
    assert done.returncode == 0, done.stderr
    record = json.loads(done.stdout)
    assert record["schema"] == C.OPTIONS_SCHEMA and record["air"]["available"]
    text = cli("preset", "options", "oun-2013-05-20")
    assert text.returncode == 0 and "push" in text.stdout and C.AIR_FAMOUS in text.stdout


def test_preset_new_with_choices_labels_the_run_and_replays_byte_for_byte(tmp_path):
    root = tmp_path / "runs"
    args = ["--custom", "grid=500", "--custom", "rotation=on", "--custom", "bumps=seed:77", "--custom", "push=2"]
    made = cli("preset", "new", "supercell-1km", "--budget", "10m", "--root", str(root), *args, "--json",
               env=fixture_env("rtx5090-32gb"))
    assert made.returncode == 0, made.stderr
    record = json.loads(made.stdout)
    rundir = Path(record["run"])
    assert "-custom-" in rundir.name
    assert record["command"].endswith(" ".join(["--custom", "grid=500", "--custom", "push=2", "--custom",
                                                "bumps=seed:77", "--custom", "rotation=on"]))
    run = json.loads((rundir / "run.json").read_text())
    assert run["case"] == "Supercell, quick look, 1 km (customized)"
    assert run["guided"]["customized"] == ["grid", "push", "bumps", "rotation"]
    fit_json = json.loads((rundir / "fit.json").read_text())
    assert [entry["knob"] for entry in fit_json["customized"]] == ["grid", "push", "bumps", "rotation"]
    assert fit_json["customized"][0]["keys"]["dx"] == [1000.0, 500.0]
    assert fit_json["pinned"] == ["grid"]
    diff = (rundir / "namelist.diff").read_text()
    assert "dx" in diff and "500.0" in diff and "icor" in diff and "irandp" in diff
    assert (rundir / "namelist.input.original").read_text() == P.load("supercell-1km").namelist_text
    report = (rundir / "import-report.json").read_text()
    assert "GM1's own random numbers" in report and json.loads(report)["guided"]["offered"]
    assert "bptpert_k = 2.0" in (rundir / "settings.toml").read_text()
    # The recorded command makes the same namelist again.
    words = record["command"].split()[1:]
    words[words.index("--name") + 1] = "again"
    again = cli(*words, "--json", env=fixture_env("rtx5090-32gb"))
    assert again.returncode == 0, again.stderr
    first = hashlib.sha256((rundir / "namelist.input").read_bytes()).hexdigest()
    second = hashlib.sha256((root / "again" / "namelist.input").read_bytes()).hexdigest()
    assert first == second


# ------------------------------------------------------------------ what the record and the text say


def test_a_customized_fit_never_says_nothing_was_changed():
    preset = C.apply(P.load("supercell-1km"), {"grid": 250.0, "pictures": 1.0, "rotation": "on"})
    result = P.fit(preset, "overnight", gpu=CARD_32GB, host=HOST)
    assert result.offered, result.reason
    text = P._fit_text(P.fit_record(preset, result, command="gpuwm-cm1 preset fit"))
    assert "nothing was changed" not in text
    for change in preset.record["customized"]:
        assert f"You changed: {change['sentence']}" in text
    plain = P.load("supercell-1km")
    fitted = P.fit(plain, "overnight", gpu=CARD_32GB, host=HOST)
    record = P.fit_record(plain, fitted, command="x")
    if not record["sentences"]:
        assert "Runs as the preset is; nothing was changed." in P._fit_text(record)
    done = cli("preset", "fit", "supercell-1km", "--budget", "overnight", "--custom", "grid=250", "--custom",
               "pictures=1", "--custom", "rotation=on", env=fixture_env("rtx5090-32gb"))
    assert done.returncode in (0, 1), done.stderr
    assert "nothing was changed" not in done.stdout and "You changed: Grid spacing 250 m" in done.stdout


def test_a_frame_worked_out_for_the_presets_air_is_replaced_by_the_one_the_made_air_runs():
    preset = P.with_environment(C.apply(P.load("supercell-1km"), {"area": 64.0}), 0.5, 0.5, "curved")
    effective = C._effective(preset)
    follows = [entry for entry in preset.record["customized"] if entry["knob"] == "follow"]
    for entry in follows:
        assert entry["keys"]["umove"][1] == float(effective["umove"])
        assert entry["keys"]["vmove"][1] == float(effective["vmove"])
    follow = (preset.sounding_given or {}).get("follow")
    if follow:
        assert not follows      # one frame, the one that runs, said once
        assert follow["keys"]["umove"][1] == float(effective["umove"])
        assert follow["keys"]["umove"][0] == float(C._effective(P.load("supercell-1km"))["umove"])
    record = P.fit_record(preset, P.fit(preset, "overnight", gpu=CARD_32GB, host=HOST), command="x")
    for entry in [*record["customized"], *record["changes"]]:
        for key in ("umove", "vmove"):
            if key in (entry.get("keys") or {}):
                assert entry["keys"][key][1] == float(effective[key]), entry


def test_set_refuses_the_words_the_knobs_and_the_fit_own():
    for key in ("dtl", "dx", "dy", "dz", "nx", "ny", "nz", "timax", "tapfrq"):
        with pytest.raises(C.CustomError, match=f"--set {key} is not taken"):
            C.apply(P.load("supercell-1km"), {}, {key: 60})
    done = cli("preset", "fit", "supercell-1km", "--budget", "overnight", "--set", "dtl=60",
               env=fixture_env("rtx5090-32gb"))
    assert done.returncode == 2 and "--custom grid=M" in done.stderr, done.stderr


def test_the_area_is_refused_in_the_right_words_and_kept_on_its_steps():
    preset = P.load("supercell-1km")
    with pytest.raises(C.CustomError, match="The area must be at least 64 km across; 0.5 km is too small."):
        C.apply(preset, {"area": 0.5})
    with pytest.raises(C.CustomError, match="is larger than this storm offers"):
        C.apply(preset, {"area": 500.0})
    stepped = C._effective(C.apply(preset, {"area": 97.3}))
    nx0 = int(C._effective(preset)["nx"])
    assert (int(stepped["nx"]) - nx0) % C.AREA_STEP_CELLS == 0 and int(stepped["nx"]) == 96
    assert "seed:N" in P._options_text(C.options(preset))


def test_the_grid_sentence_names_the_time_step_only_when_it_changes():
    preset = P.load("supercell-1km")
    coarse = C.apply(preset, {"grid": 2000.0})
    sentence = next(entry["sentence"] for entry in coarse.record["customized"] if entry["knob"] == "grid")
    assert sentence == "Grid spacing 2 km instead of 1 km.", sentence
    fine = C.apply(preset, {"grid": 250.0})
    sentence = next(entry["sentence"] for entry in fine.record["customized"] if entry["knob"] == "grid")
    assert "time step" in sentence


def test_bumps_already_on_are_not_a_change():
    once = C.apply(P.load("supercell-1km"), {"bumps": "on"})
    assert [entry["knob"] for entry in once.record["customized"]] == ["bumps"]
    twice = C.apply(once, {"bumps": "on"})
    assert [entry["knob"] for entry in twice.record["customized"]] == ["bumps"]
    drawn = C.apply(once, {"bumps": 7})
    assert [entry["knob"] for entry in drawn.record["customized"]] == ["bumps", "bumps"]


def test_parcels_are_greyed_where_the_area_cannot_reach_cm1s_parcel_grid():
    preset = P.load("bubble-cloud-250m")
    parcels = knobs(preset)["parcels"]
    on = next(choice for choice in parcels["choices"] if choice["value"] == "on")
    assert not on["available"] and "at least 120 km" in on["why"]
    with pytest.raises(C.CustomError, match="Follow air parcels needs an area at least 120 km across"):
        C.apply(preset, {"parcels": "on"})
    assert next(c for c in knobs(P.load("supercell-1km"))["parcels"]["choices"] if c["value"] == "on")["available"]


def test_a_refusal_says_what_kind_it_is():
    withheld = P.load("supercell-100m")
    result = P.fit(withheld, "overnight", gpu=CARD_32GB, host=HOST)
    if not result.offered:
        assert P.fit_record(withheld, result, command="x")["reason_kind"] == "withheld"
    preset = C.apply(P.load("supercell-1km"), {"grid": 250.0, "area": 120.0, "time": 120.0})
    record = P.fit_record(preset, P.fit(preset, "10m", gpu=HEADLESS_16GB, host=HOST), command="x")
    assert record["reason_kind"] == "time"
    plain = P.load("supercell-1km")
    assert P.fit_record(plain, P.fit(plain, "overnight", gpu=CARD_32GB, host=HOST), command="x")["reason_kind"] is None


def test_sizes_are_said_as_people_say_them():
    from cm1core.cm1run.fit import _gb_round

    assert _gb_round(4704.4e9) == "4,700 GB" and _gb_round(1000.0e9) == "1,000 GB"
    assert _gb_round(470.3e9) == "470 GB" and _gb_round(47.2e9) == "47 GB" and _gb_round(4.72e9) == "4.7 GB"


def test_a_run_edited_after_its_fit_is_labelled_edited(tmp_path):
    root = tmp_path / "runs"
    made = cli("preset", "new", "supercell-1km", "--budget", "10m", "--root", str(root), "--json",
               env=fixture_env("rtx5090-32gb"))
    assert made.returncode == 0, made.stderr
    rundir = Path(json.loads(made.stdout)["run"])
    assert json.loads((rundir / "run.json").read_text())["case"] == "Supercell, quick look, 1 km"
    edited = cli("edit", str(rundir), "--set", "ptype=7")
    assert edited.returncode in (0, 3), edited.stderr
    run = json.loads((rundir / "run.json").read_text())
    assert run["case"] == "Supercell, quick look, 1 km (edited)" and run["guided"]["edited"] is True
