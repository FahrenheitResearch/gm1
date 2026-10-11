"""The guided presets and ``gpuwm-cm1 preset``.

Every preset's namelist imports with exit 0, no refusal, no
acknowledgment and no "Not yet checked" row.  The squall-line presets are the
exception while GM1's turbulence scheme treats repeating sides differently
from CM1: they are not offered, and the reason is the import report's
own.  When that row goes, they are offered, and this test asks for exit 0
like the others.  CPU only; the real native namelist parser.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from cm1core.cm1import import coverage
from cm1core.cm1import.door import run_import
from cm1core.cm1run import fit as fit_module
from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.cm1run.gpuinfo import GpuInfo, HostInfo

from cm1import_support import namelist_tool

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
HOST = HostInfo(
    ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, disk_root="/runs",
    disk_total_bytes=2000 * 10**9, disk_free_bytes=1000 * 10**9, cpu_count=16,
)
REPEATING_REASON = "Repeating sides have not been fully checked against CM1 yet."
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"


def card(name: str, total_mib: int, used_bytes: float, *, wsl2: bool = False) -> GpuInfo:
    used = int(round(used_bytes / MIB)) * MIB
    return GpuInfo(0, "GPU-fake", name, total_mib * MIB, used, total_mib * MIB - used, 0, "8.6", "610", None, (), wsl2)


WSL2_3080 = card("NVIDIA GeForce RTX 3080", 10240, 4.8e9, wsl2=True)
HEADLESS_16GB = card("NVIDIA GeForce RTX 4080", 16376, 0.004e9)


@pytest.fixture(autouse=True)
def _parser_and_cache(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})}
    return subprocess.run([sys.executable, "-m", "cm1core", *args], capture_output=True, text=True,
                          env=environment, timeout=300, cwd=ROOT)


def fixture_env(name: str, wsl2: bool = False) -> dict:
    """A fake GPU and a fixed host (64 GB RAM, 1 TB free disk), so a fit does not depend on this machine."""

    return {"GPUWM_CM1_NVIDIA_SMI": str(FIXTURES / "fake_nvidia_smi.py"),
            "FAKE_NVIDIA_SMI_FIXTURE": str(FIXTURES / f"{name}.json"),
            "GPUWM_CM1_WSL2": "1" if wsl2 else "0",
            "GPUWM_CM1_HOST_FIXTURE": str(FIXTURES / "host-64gb-1tb.json")}


def _offered_copy(tmp_path: Path, preset_id: str) -> Path:
    """The shipped presets with ``preset_id`` offered again, for the fit tests of a preset 0.1.0 withholds."""

    import shutil

    root = tmp_path / "offered-presets"
    shutil.copytree(P.presets_root(), root, ignore=shutil.ignore_patterns("__pycache__"))
    path = root / preset_id / "preset.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("offered", None)
    record.pop("not_offered_reason", None)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------ the files


def test_the_preset_files_are_what_the_generator_writes_from_cm1s_own_namelists():
    completed = subprocess.run([sys.executable, str(ROOT / "tools/presets/make_presets.py"), "--check"],
                               capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_every_preset_namelist_is_in_the_cm1_redistribution_inventory():
    """NOTICE lists the presets' namelists as modified copies of CM1's two example namelists, and says each
    bundled CM1 file is in CM1-AUTHORITY.json's inventory: one row per preset, with its current digest and the
    source file and digest its preset.json names."""

    import hashlib

    authority = json.loads((ROOT / "src/cm1core/records/CM1-AUTHORITY.json").read_text(encoding="utf-8"))
    rows = {row["path"]: row for row in authority["redistribution"]["bundled"]}
    presets_dir = ROOT / "src/cm1core/presets"
    namelists = sorted(presets_dir.glob("*/namelist.input"))
    assert namelists
    for namelist in namelists:
        path = namelist.relative_to(ROOT).as_posix()
        assert path in rows, path
        row = rows[path]
        assert row["sha256"] == hashlib.sha256(namelist.read_bytes()).hexdigest(), path
        source = json.loads((namelist.parent / "preset.json").read_text(encoding="utf-8"))["source"]
        assert source["file"] == "CM1 r21.1 " + row["source_path"], path
        assert source["sha256"] == row["source_sha256"], path
    listed = {path for path in rows if path.startswith("src/cm1core/presets/")}
    assert listed == {namelist.relative_to(ROOT).as_posix() for namelist in namelists}


def test_the_five_presets_of_7_3_are_installed_in_order():
    presets = P.load_all()
    ids = [preset.id for preset in presets]
    # The gallery is grouped: the supercells open "Start here", the squall lines close "Other
    # storm types" (Famous environments follow it); test_presets_gallery.py pins every group and its order.
    assert ids[:3] == ["supercell-1km", "supercell-250m", "supercell-100m"]
    other = [preset.id for preset in presets if preset.group == "other"]
    assert other[-2:] == ["squall-line-1km", "squall-line-250m"]
    names = {preset.id: preset.name for preset in presets}
    assert names["supercell-1km"] == "Supercell, quick look, 1 km"
    assert names["supercell-100m"] == "Supercell, 100 m, overnight"
    for preset in presets:
        assert preset.line and preset.budget in P.BUDGET_ORDER
        record = preset.record
        # Only the listed words differ from CM1's file, each with its reason.
        assert all(change["why"] for change in record["source"]["changes"])


def test_the_budgets_are_the_fits_budgets():
    assert P.BUDGETS == dict(fit_module.BUDGETS)
    assert set(P.BUDGET_BUTTONS) == set(fit_module.BUDGETS)


# ------------------------------------------------------------------ every preset imports cleanly


def _rows(payload: dict, section: str) -> list[dict]:
    for block in payload.get("sections", []):
        if block["id"] == section:
            return block.get("rows", [])
    return []


@pytest.mark.parametrize("preset_id", ["supercell-1km", "supercell-250m", "supercell-100m",
                                       "squall-line-1km", "squall-line-250m"])
def test_every_preset_imports_as_a_run_gm1_runs_as_cm1_does(preset_id, tmp_path):
    preset = P.load(preset_id)
    code, payload, _ = run_import(preset.namelist_path, tmp_path / preset_id,
                                  overrides=preset.record.get("gm1_settings"))
    examined = P.examine(preset)
    assert code == 0, payload.get("verdict_text")
    assert _rows(payload, "refused") == [] and _rows(payload, "not_read") == []
    assert payload["acknowledgments"]["needed"] == []
    not_checked = _rows(payload, "not_checked")
    if examined["offered"]:
        assert not_checked == [], not_checked
        assert payload["guided"]["offered"] is True
    elif preset.record.get("offered") is False:
        # Withheld by its own preset.json (0.1.0 withholds the 100 m supercell), though its namelist imports as a
        # run GM1 runs as CM1 does.
        assert not_checked == [], not_checked
        assert payload["guided"]["offered"] is True
        assert examined["why_not"] == preset.record["not_offered_reason"]
    else:
        # Until the periodic-sides fix: the squall lines' repeating north and south sides, and nothing else.
        assert preset_id.startswith("squall-line")
        assert [tuple(row["keys"]) for row in not_checked] == [("sbc", "nbc")]
        assert examined["reasons"] == [REPEATING_REASON]
        assert examined["why_not"] == "Not offered: repeating sides have not been fully checked against CM1 yet."


def test_the_supercell_presets_are_offered_and_the_squall_lines_wait_for_the_repeating_sides():
    offered = {preset.id: P.examine(preset)["offered"] for preset in P.load_all()}
    assert offered["supercell-1km"] and offered["supercell-250m"]
    # 0.1.0 withholds the 100 m supercell with its own reason (its preset.json).
    assert not offered["supercell-100m"]
    assert P.examine(P.load("supercell-100m"))["why_not"].startswith("Not offered in 0.1.0: ")
    from cm1core.boundary_validation_register import register as boundary_register

    caveat = boundary_register()["wbc/ebc/sbc/nbc=1 (periodic)"].get("caveat") or ""
    squall_expected = "SGS closure" not in caveat
    assert offered["squall-line-1km"] is squall_expected
    assert offered["squall-line-250m"] is squall_expected


def test_a_preset_json_can_withhold_a_preset_with_its_own_reason(tmp_path, monkeypatch):
    folder = tmp_path / "presets" / "supercell-1km"
    folder.mkdir(parents=True)
    source = P.load("supercell-1km")
    (folder / "namelist.input").write_text(source.namelist_text, encoding="ascii")
    record = {**source.record, "offered": False, "not_offered_reason": "Not offered yet: kept back for a test."}
    (folder / "preset.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv(P.PRESETS_ENV, str(tmp_path / "presets"))
    preset = P.load("supercell-1km")
    examined = P.examine(preset)
    assert not examined["offered"] and examined["why_not"] == "Not offered yet: kept back for a test."
    result = P.fit(preset, "1h", gpu=HEADLESS_16GB, host=HOST)
    assert not result.offered and result.reason == examined["why_not"]


# ------------------------------------------------------------------ the command


def test_preset_list_and_show(tmp_path):
    listed = cli("preset", "list", "--json")
    assert listed.returncode == 0, listed.stderr
    record = json.loads(listed.stdout)
    assert record["schema"] == P.LIST_SCHEMA and record["command"] == "gpuwm-cm1 preset list --json"
    rows = {row["id"]: row for row in record["presets"]}
    assert rows["supercell-1km"]["facts"]["words"] == "1 km grid · 120 × 120 km · 2 h of storm time"
    assert rows["supercell-1km"]["facts"]["scheme"] == "Morrison"
    assert [b["id"] for b in record["budgets"]] == ["10m", "1h", "overnight"]
    text = cli("preset", "list")
    assert text.returncode == 0 and "Supercell, 250 m" in text.stdout
    shown = cli("preset", "show", "supercell-100m", "--json")
    assert shown.returncode == 0
    body = json.loads(shown.stdout)
    assert body["namelist"] == P.load("supercell-100m").namelist_text
    assert "11 km east" in body["gm1_settings_why"]
    unknown = cli("preset", "show", "no-such-storm")
    assert unknown.returncode == 2 and "there is no preset 'no-such-storm'" in unknown.stderr


def test_preset_fit_on_a_fixture_gpu_says_what_it_changed(tmp_path):
    fitted = cli("preset", "fit", "supercell-250m", "--budget", "10m", "--json",
                 env=fixture_env("rtx3080-wsl2-desktop-4p8", wsl2=True))
    assert fitted.returncode == 0, fitted.stderr
    record = json.loads(fitted.stdout)
    assert record["status"] == "fits" and record["facts"]["dx_m"] == 500.0
    assert record["command"] == "gpuwm-cm1 preset fit supercell-250m --budget 10m"
    assert any("500 m grid spacing instead of 250 m" in sentence for sentence in record["sentences"])
    assert record["preset_settings"] == {} and "runtime.memory_budget_mib" in record["run_settings"]
    refused = cli("preset", "fit", "supercell-100m", "--budget", "overnight",
                  env={**fixture_env("rtx3080-wsl2-desktop-4p8", wsl2=True),
                       P.PRESETS_ENV: str(_offered_copy(tmp_path, "supercell-100m"))})
    assert refused.returncode == 1
    assert "of free GPU memory even at its smallest size" in refused.stdout
    withheld = cli("preset", "fit", "supercell-100m", "--budget", "overnight", env=fixture_env("rtx5090-32gb"))
    assert withheld.returncode == 1 and "Not offered in 0.1.0: " in withheld.stdout
    squall = cli("preset", "fit", "squall-line-1km", "--budget", "10m", "--json",
                 env=fixture_env("rtx5090-32gb"))
    body = json.loads(squall.stdout)
    offered = P.examine(P.load("squall-line-1km"))["offered"]
    assert squall.returncode == (0 if offered else 1)
    if not offered:
        assert body["reason"].startswith("Not offered: repeating sides")


def test_preset_new_writes_the_fitted_run_directory(tmp_path):
    root = tmp_path / "runs"
    made = cli("preset", "new", "supercell-250m", "--budget", "10m", "--root", str(root), "--name", "g250",
               "--json", env=fixture_env("rtx3080-wsl2-desktop-4p8", wsl2=True))
    assert made.returncode == 0, made.stderr
    record = json.loads(made.stdout)
    rundir = root / "g250"
    assert record["ok"] and record["run"] == str(rundir.resolve())
    assert record["command"] == (f"gpuwm-cm1 preset new supercell-250m --budget 10m --root {root.resolve()} "
                                 "--name g250")
    assert record["next"] == f"gpuwm-cm1 run {rundir.resolve()}"
    preset = P.load("supercell-250m")
    # The preset's namelist is the original; the fitted one is the run's, and the diff says what changed.
    assert (rundir / "namelist.input.original").read_text() == preset.namelist_text
    fitted = (rundir / "namelist.input").read_text()
    assert "dx     =  500.0" in fitted and "nx           =     240" in fitted
    diff = (rundir / "namelist.diff").read_text()
    assert "-" in diff and "500.0" in diff
    fit_json = json.loads((rundir / "fit.json").read_text())
    assert fit_json["schema"] == "gpuwm-cm1.fit/v1" and fit_json["preset"] == "supercell-250m"
    assert fit_json["namelist_sha256"] and fit_json["sentences"]
    run = json.loads((rundir / "run.json").read_text())
    assert run["preset"] == "supercell-250m" and run["case"] == "Supercell, 250 m"
    assert run["guided"] == {"preset": "supercell-250m", "budget": "10m", "fit": "fit.json"}
    report = json.loads((rundir / "import-report.json").read_text())
    assert report["exit"] == 0 and report["guided"]["offered"] and report["reimport"] is False
    assert report["command"] == record["command"]
    settings = (rundir / "settings.toml").read_text()
    assert "[runtime]" in settings and "tile_nx = 128" in settings and "memory_budget_mib" in settings
    deck = (rundir / "deck.toml").read_text()
    assert "tile_nx = 128" in deck
    budget = int(fit_json["run_settings"]["runtime.memory_budget_mib"])
    assert f"memory_budget_mib = {budget}" in deck and budget <= 0.95 * fit_json["gpu"]["free_bytes"] / MIB
    again = cli("preset", "new", "supercell-250m", "--budget", "10m", "--root", str(root), "--name", "g250",
                env=fixture_env("rtx3080-wsl2-desktop-4p8", wsl2=True))
    assert again.returncode == 2 and "already exists" in again.stderr


def test_preset_new_keeps_the_presets_bubble(tmp_path):
    root = tmp_path / "runs"
    made = cli("preset", "new", "supercell-100m", "--budget", "overnight", "--root", str(root), "--name", "g100",
               env={**fixture_env("rtx4080-16gb-headless"),
                    P.PRESETS_ENV: str(_offered_copy(tmp_path, "supercell-100m"))})
    assert made.returncode == 0, made.stderr
    rundir = root / "g100"
    assert "bubble_center_x_m = 11000.0" in (rundir / "settings.toml").read_text()
    assert "bubble_center_x_m = 11000.0" in (rundir / "deck.toml").read_text()
    assert not (rundir / "namelist.diff").exists()          # it runs as the preset is on a 16 GB card
    fit_json = json.loads((rundir / "fit.json").read_text())
    assert fit_json["preset_settings"] == {"initialization.bubble_center_x_m": 11000.0,
                                           "initialization.bubble_center_y_m": 0.0}


def test_preset_new_writes_nothing_for_a_preset_that_does_not_fit(tmp_path):
    root = tmp_path / "runs"
    made = cli("preset", "new", "supercell-100m", "--budget", "10m", "--root", str(root), "--name", "no",
               env=fixture_env("rtx3080-wsl2-desktop-4p8", wsl2=True))
    assert made.returncode == 1
    assert "Nothing was written" in made.stdout
    assert not (root / "no").exists()


def test_preset_new_says_not_offered_once_for_a_preset_gm1_does_not_offer_yet(tmp_path):
    root = tmp_path / "runs"
    made = cli("preset", "new", "squall-line-1km", "--budget", "10m", "--root", str(root), "--name", "sq",
               env=fixture_env("rtx3080-wsl2-desktop-4p8", wsl2=True))
    examined = P.examine(P.load("squall-line-1km"))
    if examined["offered"]:
        pytest.skip("the squall lines are offered (the periodic-sides fix has landed)")
    assert made.returncode == 1
    first = made.stdout.splitlines()[0]
    assert first == examined["why_not"] and first.lower().count("not offered") == 1, made.stdout
    assert not (root / "sq").exists()


def test_with_sounding_reads_the_changed_namelist_not_the_shipped_one(tmp_path):
    """A preset changed before its sounding is given (Customize) keeps its own namelist through with_sounding, so
    the storm-track check that places the frame and bubble reads the namelist that runs."""

    import dataclasses

    shipped = P.load("your-environment")
    text = fit_module.set_namelist_values(shipped.namelist_text, {"nx": 160, "ny": 160})
    changed_file = tmp_path / "namelist.input"
    changed_file.write_text(text, encoding="ascii")
    changed = dataclasses.replace(shipped, namelist_text=text, namelist_file=changed_file,
                                  namelist_original=shipped.namelist_text)
    other = tmp_path / "input_sounding"
    other.write_bytes((shipped.folder / "input_sounding").read_bytes())
    given = P.with_sounding(changed, other)
    assert "160" in given.namelist_text.split("nx", 1)[1].split(",", 1)[0]
    assert given.namelist_original_text == shipped.namelist_text
    track = P.storm_track_for(given)
    assert track and track["domain_km"] == [160.0, 160.0]


def test_a_fresh_import_with_an_original_keeps_it_and_writes_the_diff(tmp_path):
    preset = P.load("supercell-1km")
    rundir = tmp_path / "run"
    rundir.mkdir()
    staged = rundir / "namelist.input"
    staged.write_text(preset.namelist_text.replace("timax  =  7200.0", "timax  =  3600.0"), encoding="ascii")
    code, payload, _ = run_import(staged, rundir, original=preset.namelist_text.encode("ascii"),
                                  overrides={"runtime.memory_budget_mib": 2048})
    assert code == 0 and payload["reimport"] is False
    assert (rundir / "namelist.input.original").read_text() == preset.namelist_text
    assert "timax" in (rundir / "namelist.diff").read_text()
    assert "memory_budget_mib = 2048" in (rundir / "deck.toml").read_text()


def test_a_gm1_setting_must_name_its_table():
    imported = coverage.import_for_report(P.load("supercell-1km").namelist_path)
    with pytest.raises(ValueError):
        coverage.build_report(imported, overrides={"memory_budget_mib": 1})


def test_a_card_shows_its_own_picture_or_names_whose_it_is():
    presets = {preset.id: preset for preset in P.load_all()}
    for preset in presets.values():
        shown = P.picture(preset, presets)
        own = (preset.folder / "picture.png").is_file()
        if own:
            assert shown["own"] and shown["preset"] == preset.id
            record = json.loads((preset.folder / "picture.json").read_text())
            assert record["schema"] == P.PICTURE_SCHEMA and record["run"]["preset"] == preset.id
            assert record["caption"].endswith("Simulated.") or "Simulated." in record["caption"]
        elif preset.record.get("picture_from") and shown["available"]:
            other = presets[preset.record["picture_from"]]
            assert shown["preset"] == other.id and not shown["own"]
            assert shown["caption"].startswith(f"From {other.name} (this storm has no run of its own yet): ")
        else:
            # never another kind of storm's picture
            assert not shown["available"] and shown["caption"].startswith("No picture yet")
    assert not P.picture(presets["squall-line-1km"], presets)["available"]
    assert presets["supercell-100m"].record["picture_from"] == "supercell-250m"


def test_the_host_fixture_stands_in_for_this_machine(monkeypatch, tmp_path):
    from cm1core.cm1run.gpuinfo import HOST_ENV, query_host

    monkeypatch.setenv(HOST_ENV, str(FIXTURES / "host-64gb-1tb.json"))
    host = query_host(tmp_path)
    assert host.disk_free_bytes == 10**12 and host.ram_total_bytes == 64 * 10**9 and host.disk_root == str(tmp_path)


# ------------------------------------------------------------------ a small computer (7.4 steps 5 and 6)


def test_the_output_intervals_the_fit_tries_are_all_ones_the_runtime_admits():
    # Each splits the storm time into whole frames of whole time steps (the history cadence rule); doubling
    # 5 min over 2 h would reach 80 min, which the estimator refuses.
    assert fit_module._output_intervals(300.0, 7200.0, 6.0) == [300.0, 600.0, 900.0, 1200.0, 1800.0, 2400.0,
                                                                 3600.0, 7200.0]
    assert fit_module._output_intervals(900.0, 6300.0, 1.5) == [900.0, 6300.0]     # 1 h 45 min: 7 frames or 1
    assert fit_module._output_intervals(900.0, 7200.0, 1.5) == [900.0, 1800.0, 3600.0, 7200.0]


@pytest.mark.parametrize("disk_gb, expect", [(2.0, "fits"), (0.5, "fits"), (0.05, "not-offered")])
def test_a_nearly_full_disk_is_fitted_or_refused_in_words_never_a_crash(disk_gb, expect):
    host = HostInfo(ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, disk_root="/runs",
                    disk_total_bytes=2000 * 10**9, disk_free_bytes=int(disk_gb * 10**9), cpu_count=16)
    result = P.fit(P.load("supercell-1km"), "10m", gpu=WSL2_3080, host=host)
    assert result.status == expect
    if result.offered:
        assert result.estimate.disk.value <= 0.5 * host.disk_free_bytes
        assert any(sentence.startswith("Saves a picture every") for sentence in result.sentences)
    else:
        assert result.reason.startswith("Needs ") and "of free disk even at its smallest size" in result.reason


def test_a_preset_too_big_for_the_ram_says_so_rather_than_blaming_the_edge(tmp_path, monkeypatch):
    monkeypatch.setenv(P.PRESETS_ENV, str(_offered_copy(tmp_path, "supercell-100m")))
    host = HostInfo(ram_total_bytes=4 * 10**9, ram_available_bytes=3 * 10**9, disk_root="/runs",
                    disk_total_bytes=2000 * 10**9, disk_free_bytes=1000 * 10**9, cpu_count=16)
    result = P.fit(P.load("supercell-100m"), "overnight", gpu=HEADLESS_16GB, host=host)
    assert not result.offered
    assert result.reason.startswith("Needs ") and "of RAM even at its smallest size" in result.reason
    assert "edge" not in result.reason
