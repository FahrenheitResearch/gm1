"""The fit table over the shipped presets.

Four fake GPUs: an 8 GB card driving a desktop (4.3 GB in use), the WSL2 RTX 3080 (4.8 GB in use, no process
list), a 16 GB card headless, and a 32 GB card.  For every preset x budget x card, the fitted namelist:

- is written as a run directory by ``preset new`` and re-imports with exit 0 (the real importer and parser);
- needs <= 0.95 x free GPU memory, <= 50 % of RAM locked and <= 50 % of free disk;
- has a central wall estimate <= the budget, and <= 39.3 M cells;
- keeps the physics, boundary, sounding and initiation keys byte-identical to the preset's;
- has a fit.json that names every change in a sentence.

A preset that cannot fit comes back as not offered, with a reason naming the memory or the time it needs (or,
for a preset GM1 does not offer yet, the import report's reason).  7.4's worked rows are checked last.
"""

from __future__ import annotations

import json
import re

import pytest

from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.cm1run.fit import BUDGETS, FROZEN_KEYS, HOST_FRACTION, DISK_FRACTION
from cm1core.cm1run.gpuinfo import GpuInfo, HostInfo

from pathlib import Path

from cm1import_support import namelist_tool

MIB = 1024 * 1024
HOST = HostInfo(
    ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, disk_root="/runs",
    disk_total_bytes=2000 * 10**9, disk_free_bytes=1000 * 10**9, cpu_count=16,
)
LARGEST_CELLS = 39_321_600


def card(name: str, total_mib: int, used_bytes: float, *, wsl2: bool = False, cc: str = "8.6") -> GpuInfo:
    used = int(round(used_bytes / MIB)) * MIB
    return GpuInfo(0, "GPU-fake", name, total_mib * MIB, used, total_mib * MIB - used, 0, cc, "610", None, (), wsl2)


CARDS = {
    "8GB-desktop": card("NVIDIA GeForce RTX 4060", 8188, 4.3e9, cc="8.9"),
    "3080-WSL2": card("NVIDIA GeForce RTX 3080", 10240, 4.8e9, wsl2=True),
    "16GB-headless": card("NVIDIA GeForce RTX 4080", 16376, 0.004e9, cc="8.9"),
    "32GB": card("NVIDIA GeForce RTX 5090", 32607, 0.002e9, cc="12.0"),
}
PRESET_IDS = [preset.id for preset in P.load_all()]


@pytest.fixture(autouse=True)
def _parser_and_cache(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def assignments(text: str) -> dict[str, str]:
    """``key -> value as written`` for every assignment outside comments."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        body = line.split("!", 1)[0]
        for key, value in re.findall(r"([A-Za-z_]\w*)\s*=\s*([^,\n]*)", body):
            values.setdefault(key.lower(), value.strip())
    return values


def _offered_copy(tmp_path: Path, preset_id: str) -> Path:
    """The shipped presets with ``preset_id`` offered again, for the fit tests of a preset this version withholds."""

    import shutil

    root = tmp_path / "offered-presets"
    shutil.copytree(P.presets_root(), root, ignore=shutil.ignore_patterns("__pycache__"))
    path = root / preset_id / "preset.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("offered", None)
    record.pop("not_offered_reason", None)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return root


def test_the_largest_domain_is_the_records():
    assert int(rates.load_record()["largest_domain_cells"]["value"]) == LARGEST_CELLS


@pytest.mark.parametrize("card_name", list(CARDS))
@pytest.mark.parametrize("budget", list(BUDGETS))
@pytest.mark.parametrize("preset_id", PRESET_IDS)
def test_every_fit_is_a_run_gm1_runs_within_the_limits(preset_id, budget, card_name, tmp_path):
    preset = P.load(preset_id)
    gpu = CARDS[card_name]
    result = P.fit(preset, budget, gpu=gpu, host=HOST)
    if not result.offered:
        examined = P.examine(preset)
        if not examined["offered"]:
            assert result.reason == examined["why_not"]
        else:
            assert re.search(r"GB of free GPU memory|Takes about .* on your GPU|too close to the edge",
                             result.reason), result.reason
        assert result.namelist_text is None
        return
    estimate = result.estimate
    # the limits of 7.4 step 1, the size guard and the budget
    assert estimate.gpu_need.value <= 0.95 * gpu.free_bytes
    assert estimate.host_locked.value <= HOST_FRACTION * HOST.ram_total_bytes
    assert estimate.disk.value <= DISK_FRACTION * HOST.disk_free_bytes
    assert estimate.wall.value <= BUDGETS[budget]
    assert estimate.cells <= LARGEST_CELLS
    # physics, boundaries, sounding and initiation: byte-identical words
    before, after = assignments(preset.namelist_text), assignments(result.namelist_text)
    for key in FROZEN_KEYS:
        assert after.get(key) == before.get(key), key
    # the run directory: preset new's own, re-imported by the real importer with exit 0
    made = P.new_run(preset, budget, tmp_path / "runs", "fitted", gpu=gpu, host=HOST, command="test")
    rundir = tmp_path / "runs" / "fitted"
    assert made["ok"] and made["import"]["exit"] == 0
    report = json.loads((rundir / "import-report.json").read_text())
    assert report["exit"] == 0 and report["guided"]["offered"] is True
    assert (rundir / "namelist.input").read_text() == result.namelist_text
    # every change named in a sentence
    fit_json = json.loads((rundir / "fit.json").read_text())
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    named = {key for change in fit_json["changes"] for key in change["keys"]}
    assert changed <= named, (changed, named)
    assert all(change["sentence"].endswith(".") for change in fit_json["changes"])
    assert len(fit_json["sentences"]) == len(fit_json["changes"])


# ------------------------------------------------------------------ 7.4's worked table


def fitted(preset_id: str, budget: str, card_name: str):
    return P.fit(P.load(preset_id), budget, gpu=CARDS[card_name], host=HOST)


def test_supercell_1km_in_10_minutes_runs_as_is_on_every_card():
    for card_name in CARDS:
        result = fitted("supercell-1km", "10m", card_name)
        assert result.offered and result.changes == (), card_name
        assert result.estimate.cells == 120 * 120 * 40 and result.estimate.wall_s < 180


def test_supercell_250m_in_an_hour():
    desktop = fitted("supercell-250m", "1h", "3080-WSL2")
    assert desktop.offered and desktop.estimate.grid["dx_m"] == 250.0
    assert 256 <= desktop.estimate.grid["nx"] < 480          # 7.4: 312 (78 km) with 5.2 GB free; 5.9 GB here
    assert desktop.sentences[0].startswith("Made the area") and "free memory" in desktop.sentences[0]
    headless = fitted("supercell-250m", "1h", "16GB-headless")
    assert 448 <= headless.estimate.grid["nx"] <= 472          # 7.4: 464 (116 km)
    assert headless.estimate.wall_s <= 3600
    big = fitted("supercell-250m", "1h", "32GB")
    assert big.changes == () and big.estimate.wall_s == pytest.approx(20 * 60, rel=0.25)


def test_supercell_250m_in_10_minutes_takes_the_500m_rung_on_the_desktop():
    result = fitted("supercell-250m", "10m", "3080-WSL2")
    grid = result.estimate.grid
    assert (grid["nx"], grid["ny"], grid["dx_m"]) == (240, 240, 500.0)      # 7.4: 240 x 240 at 500 m
    assert result.estimate.gpu_need.value == pytest.approx(3.3e9, rel=0.1)
    assert 5 * 60 <= result.estimate.wall_s <= 10 * 60


def test_supercell_100m_overnight(tmp_path, monkeypatch):
    assert not fitted("supercell-100m", "overnight", "32GB").offered      # this version withholds it
    monkeypatch.setenv(P.PRESETS_ENV, str(_offered_copy(tmp_path, "supercell-100m")))
    desktop = fitted("supercell-100m", "overnight", "3080-WSL2")
    assert not desktop.offered
    assert desktop.reason.startswith("Needs ") and "yours has" in desktop.reason
    for card_name, hours in (("16GB-headless", 4.2), ("32GB", 1.3)):
        result = fitted("supercell-100m", "overnight", card_name)
        assert result.offered and result.changes == ()
        assert result.estimate.cells == 600 * 600 * 80
        assert result.estimate.wall_s / 3600 == pytest.approx(hours, rel=0.25)
