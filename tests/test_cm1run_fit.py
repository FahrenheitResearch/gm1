"""The guided mode's auto-fit, on fake GPUs.

The presets here are built from CM1's own supercell deck as the guided gallery lists
them; the package ships the preset folders.  Every fitted namelist is re-read by the
native parser inside :func:`fit_preset` (``verify=True``).
"""

from __future__ import annotations

import json
import re

import pytest

from cm1core.cm1run import rates
from cm1core.cm1run.fit import (
    FROZEN_KEYS,
    PresetSpec,
    article,
    fit_preset,
    set_namelist_values,
)
from cm1core.cm1run.gpuinfo import GpuInfo, HostInfo

from cm1import_support import SUPERCELL, SQUALL_LINE, namelist_tool

MIB = 1024 * 1024
GB = 1e9
HOST = HostInfo(
    ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, disk_root="/runs",
    disk_total_bytes=2000 * 10**9, disk_free_bytes=1000 * 10**9, cpu_count=16,
)


def card(name: str, total_mib: int, used_bytes: float, *, wsl2: bool = False, cc: str = "8.6") -> GpuInfo:
    used = int(round(used_bytes / MIB)) * MIB
    return GpuInfo(0, "GPU-fake", name, total_mib * MIB, used, total_mib * MIB - used, 0, cc, "610", None, (), wsl2)


#: The fit table's four fixtures.
DESKTOP_8GB = card("NVIDIA GeForce RTX 4060", 8188, 4.3e9, cc="8.9")
WSL2_3080 = card("NVIDIA GeForce RTX 3080", 10240, 4.8e9, wsl2=True)
HEADLESS_16GB = card("NVIDIA GeForce RTX 4080", 16376, 0.004e9, cc="8.9")
CARD_32GB = card("NVIDIA GeForce RTX 5090", 32607, 0.002e9, cc="12.0")
CARDS = {"8GB desktop": DESKTOP_8GB, "3080 WSL2": WSL2_3080, "16GB headless": HEADLESS_16GB, "32GB": CARD_32GB}


@pytest.fixture(autouse=True)
def _parser_and_cache(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))


def stock_text() -> str:
    return SUPERCELL.read_text(encoding="ascii")


def presets() -> dict[str, PresetSpec]:
    stock = stock_text()
    return {
        "supercell-1km": PresetSpec(
            "supercell-1km", "Supercell, quick look, 1 km", set_namelist_values(stock, {"tapfrq": 300.0}),
        ),
        "supercell-250m": PresetSpec(
            "supercell-250m", "Supercell, 250 m",
            set_namelist_values(stock, dict(nx=480, ny=480, nz=80, dx=250.0, dy=250.0, dz=250.0, dtl=1.5)),
            ladder_m=(500.0, 1000.0), min_width_km={100.0: 40.0, 0.0: 64.0},
        ),
        "supercell-100m": PresetSpec(
            "supercell-100m", "Supercell, 100 m, overnight",
            # 0.5 s, the preset's own step: with this moving domain 0.6 s stops the run at its sixth step.
            set_namelist_values(stock, dict(nx=600, ny=600, nz=80, dx=100.0, dy=100.0, dz=250.0, dtl=0.5)),
            min_width_km={100.0: 40.0, 0.0: 64.0},
            # The storm-track check's own recommendation for this box: the
            # bubble 11 km east of centre (iorigin = 2), stock frame kept.
            gm1_settings={"initialization.bubble_center_x_m": 11000.0, "initialization.bubble_center_y_m": 0.0},
        ),
        "squall-line-1km": PresetSpec(
            "squall-line-1km", "Squall line, 1 km", SQUALL_LINE.read_text(encoding="ascii"),
            min_duration_s=7200.0, shrink="repeating-first", min_along_repeating_km=30.0,
            offered=False,
            not_offered_reason=(
                "Not offered yet: GM1's turbulence scheme treats the repeating north and south sides "
                "differently from CM1 (being fixed)."
            ),
        ),
    }


def namelist_values(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        body = line.split("!", 1)[0]
        for key, value in re.findall(r"(\w+)\s*=\s*([^,]+)", body):
            values[key.lower()] = value.strip()
    return values


# ------------------------------------------------------------- namelist edits


def test_set_namelist_values_changes_only_the_named_words() -> None:
    text = stock_text()
    edited = set_namelist_values(text, {"nx": 312, "DX": 250.0, "timax": 5400.0})
    before, after = namelist_values(text), namelist_values(edited)
    assert after["nx"] == "312" and after["dx"] == "250.0" and after["timax"] == "5400.0"
    assert {k for k in before if before[k] != after[k]} == {"nx", "dx", "timax"}
    assert len(edited.splitlines()) == len(text.splitlines())


def test_a_key_the_text_lacks_goes_into_its_group() -> None:
    text = set_namelist_values(stock_text(), {"nx": 64})
    stripped = re.sub(r"(?m)^\s*tapfrq\s*=.*\n", "", text)
    edited = set_namelist_values(stripped, {"tapfrq": 600.0})
    group = re.search(r"(?s)&param1\b(.*?)\n\s*/", edited).group(1)
    assert "tapfrq = 600.0" in group


# ---------------------------------------------------------- the worked table


def test_the_1km_quick_look_runs_as_is_on_every_card() -> None:
    preset = presets()["supercell-1km"]
    for name, gpu in CARDS.items():
        result = fit_preset(preset, gpu, "10m", host=HOST)
        assert result.offered, name
        assert result.changes == ()
        assert result.estimate.cells == 120 * 120 * 40
        assert result.estimate.wall_s < 180.0


def test_supercell_250m_in_ten_minutes_on_the_desktop_3080_takes_the_500m_rung() -> None:
    """7.4: '500 m rung, 240 x 240 (120 km), dt 3 s: 4.6 M cells, 3.3 GB, about 8 min'."""

    result = fit_preset(presets()["supercell-250m"], WSL2_3080, "10m", host=HOST)
    assert result.offered
    grid = result.estimate.grid
    assert (grid["nx"], grid["ny"], grid["dx_m"]) == (240, 240, 500.0)
    assert result.estimate.dt_s == 3.0
    assert result.estimate.gpu_need.value == pytest.approx(3.3e9, rel=0.1)
    assert 5 * 60 <= result.estimate.wall_s <= 10 * 60
    assert any("500 m grid spacing instead of 250 m" in sentence for sentence in result.sentences)


def test_supercell_250m_in_an_hour_on_a_headless_16gb_card_shrinks_to_about_116km() -> None:
    """7.4: '464 x 464 (116 km), 2 h: 17.2 M cells, 9.1 GB, about 60 min'."""

    result = fit_preset(presets()["supercell-250m"], HEADLESS_16GB, "1h", host=HOST)
    assert result.offered
    nx = result.estimate.grid["nx"]
    assert 448 <= nx <= 472 and result.estimate.grid["dx_m"] == 250.0
    assert result.estimate.duration_s == 7200.0
    assert result.estimate.wall_s <= 3600.0
    assert result.sentences == [
        f"Made the area {nx * 250 / 1000:.0f} km wide instead of 120 km so it finishes in about 1 hour."
    ]


def test_a_pinned_grid_is_never_coarsened_and_the_reason_names_the_pin() -> None:
    from dataclasses import replace

    spec = replace(presets()["supercell-250m"], pinned=frozenset({"grid"}))
    result = fit_preset(spec, WSL2_3080, "10m", host=HOST)
    if result.offered:
        assert result.estimate.grid["dx_m"] == 250.0      # the unpinned preset takes the 500 m rung here
    else:
        assert result.reason.endswith("let GM1 choose the grid spacing.") and result.needs["pinned"] == ["grid"]
    unpinned = fit_preset(presets()["supercell-250m"], WSL2_3080, "10m", host=HOST)
    assert unpinned.offered and unpinned.estimate.grid["dx_m"] == 500.0


def test_pinned_area_time_and_pictures_stay_as_chosen() -> None:
    from dataclasses import replace

    spec = replace(presets()["supercell-250m"], pinned=frozenset({"area", "time", "pictures"}))
    small_disk = HostInfo(64 * 10**9, 60 * 10**9, "/runs", 2000 * 10**9, 60 * 10**9, 16)
    result = fit_preset(spec, HEADLESS_16GB, "overnight", host=small_disk)
    if result.offered:
        grid = result.estimate.grid
        assert grid["nx"] * grid["dx_m"] == 120000.0 and result.estimate.duration_s == 7200.0
        assert result.estimate.history_interval_s == 900.0
    else:
        assert "let GM1 choose the area, storm time and picture interval." in result.reason


def test_random_bumps_are_priced_with_cms_own_random_numbers() -> None:
    """irandp = 1 with no seed: the fit prices the run as the import makes it (random_generator = "cm1"), so the
    candidate is not refused for having no recorded seed; a seed given is priced as given."""

    from dataclasses import replace

    spec = replace(presets()["supercell-1km"], namelist_text=set_namelist_values(stock_text(),
                                                                                  {"tapfrq": 300.0, "irandp": 1}))
    result = fit_preset(spec, HEADLESS_16GB, "10m", host=HOST)
    assert result.offered, result.reason
    assert "initialization.random_generator" not in result.gm1_settings
    seeded = fit_preset(replace(spec, random_seed=99), HEADLESS_16GB, "10m", host=HOST)
    assert seeded.offered and seeded.gm1_settings["initialization.random_seed"] == 99


def test_supercell_250m_runs_as_is_on_a_32gb_card() -> None:
    result = fit_preset(presets()["supercell-250m"], CARD_32GB, "1h", host=HOST)
    assert result.offered and result.changes == ()
    assert result.estimate.grid["nx"] == 480
    assert result.estimate.wall_s == pytest.approx(20 * 60, rel=0.25)


def test_supercell_250m_in_an_hour_on_a_wsl2_3080_shrinks_for_memory() -> None:
    """The fit shrinks for the free memory the Windows desktop leaves (5.2 GB then, 5.9 GB here)."""

    result = fit_preset(presets()["supercell-250m"], WSL2_3080, "1h", host=HOST)
    assert result.offered
    estimate = result.estimate
    assert estimate.gpu_need.value <= 0.95 * WSL2_3080.free_bytes
    assert 64_000 <= estimate.grid["nx"] * 250.0 < 120_000
    assert "fits in your GPU's 5.9 GB of free memory" in result.sentences[0]


def test_supercell_100m_overnight_is_not_offered_on_a_wsl2_3080_and_says_why() -> None:
    result = fit_preset(presets()["supercell-100m"], WSL2_3080, "overnight", host=HOST)
    assert not result.offered
    assert result.reason.startswith("Needs ")
    assert "of free GPU memory even at its smallest size" in result.reason
    assert "yours has 5.9 GB" in result.reason


@pytest.mark.parametrize(("gpu", "hours"), [(HEADLESS_16GB, 4.2), (CARD_32GB, 1.3)])
def test_supercell_100m_overnight_runs_as_is_on_16_and_32gb_cards(gpu, hours) -> None:
    result = fit_preset(presets()["supercell-100m"], gpu, "overnight", host=HOST)
    assert result.offered and result.changes == ()
    assert result.estimate.cells == 600 * 600 * 80
    assert result.estimate.wall_s / 3600.0 == pytest.approx(hours, rel=0.25)


def test_the_squall_line_is_not_offered_until_the_periodic_fix_with_its_reason() -> None:
    result = fit_preset(presets()["squall-line-1km"], CARD_32GB, "1h", host=HOST)
    assert not result.offered
    assert "being fixed" in result.reason.lower() and "repeating north and south sides" in result.reason


# ------------------------------------------------------------- the guards


@pytest.mark.parametrize("preset_id", ["supercell-1km", "supercell-250m", "supercell-100m"])
@pytest.mark.parametrize("budget", ["10m", "1h", "overnight"])
def test_every_fit_honours_the_rule_and_the_guards(preset_id, budget) -> None:
    preset = presets()[preset_id]
    base = namelist_values(preset.namelist_text)
    for name, gpu in CARDS.items():
        result = fit_preset(preset, gpu, budget, host=HOST)
        record = json.loads(json.dumps(result.record()))
        assert record["schema"] == "gpuwm-cm1.fit/v1"
        if not result.offered:
            assert result.reason
            assert re.search(r"GB|min|h\b|edge", result.reason), result.reason
            continue
        estimate = result.estimate
        assert estimate.gpu_need.value <= 0.95 * gpu.free_bytes, name
        assert estimate.host_locked.value <= 0.5 * HOST.ram_total_bytes
        assert estimate.disk.value <= 0.5 * HOST.disk_free_bytes
        assert estimate.wall_s <= result.budget_s, name
        assert estimate.cells <= 39_321_600
        assert estimate.storm.get("meets_edge_rule") in (True, None)
        fitted = namelist_values(result.namelist_text)
        for key in FROZEN_KEYS:
            assert fitted.get(key) == base.get(key), (name, key)
        changed = {key for key in base if base[key] != fitted.get(key)}
        assert changed <= {"nx", "ny", "dx", "dy", "dtl", "timax", "tapfrq"}
        if changed:
            assert result.sentences and all(sentence.endswith(".") for sentence in result.sentences)
        # "A 120 × 120 km area", "An 86 × 86 km area": the article the area is said with.
        said, rest = result.summary.split(" ", 1)
        assert said == article(rest), result.summary
        assert "GPU memory" in result.summary
        # The tiling is the estimate's own call: tile 128 only where the whole
        # domain does not fit, or where tile 128 costs no step time.
        if "runtime.tile_nx" in result.gm1_settings:
            assert result.gm1_settings["runtime.tile_nx"] == 128 and estimate.tile.startswith("128 x 128")
        else:
            assert estimate.tile.startswith("whole domain")
            assert not estimate.recommendation.get("tile_nx"), name
        assert result.gm1_settings["runtime.memory_budget_mib"] * MIB >= estimate.priced_peak_bytes


def test_a_small_free_disk_raises_the_output_interval() -> None:
    small_disk = HostInfo(64 * 10**9, 60 * 10**9, "/runs", 100 * 10**9, 14 * 10**9, 16)
    result = fit_preset(presets()["supercell-250m"], CARD_32GB, "1h", host=small_disk)
    assert result.offered
    assert result.estimate.history_interval_s > 900.0
    assert result.estimate.disk.value <= 0.5 * small_disk.disk_free_bytes
    assert any("Saves a picture every" in sentence for sentence in result.sentences)


def test_a_preset_folder_reads_back(tmp_path) -> None:
    folder = tmp_path / "supercell-250m"
    folder.mkdir()
    preset = presets()["supercell-250m"]
    (folder / "namelist.input").write_text(preset.namelist_text)
    (folder / "preset.json").write_text(json.dumps({
        "id": "supercell-250m", "name": "Supercell, 250 m", "ladder_m": [500, 1000],
        "min_width_km": {"100": 40, "default": 64}, "min_duration_s": 3600,
    }))
    read = PresetSpec.from_directory(folder)
    assert read.ladder_m == (500.0, 1000.0)
    assert read.min_width_m(250.0) == 64_000.0 and read.min_width_m(100.0) == 40_000.0


def test_a_host_with_no_disk_or_ram_figures_still_fits(tmp_path) -> None:
    """A login node or container may report neither; the fit then judges the
    GPU and the time alone."""

    unknown = HostInfo(None, None, "/runs", None, None)
    result = fit_preset(presets()["supercell-250m"], DESKTOP_8GB, "10m", host=unknown)
    assert result.offered
    assert result.estimate.gpu_need.value <= 0.95 * DESKTOP_8GB.free_bytes
    assert result.estimate.wall_s <= 600.0


def test_the_fit_takes_the_estimates_tiling_for_each_candidate() -> None:
    """Tile 128 costs launch time on grids that are not multiples of it, so a
    candidate the whole domain fits keeps one tile unless tile 128 costs no
    time; where the whole domain does not fit, it takes tile 128."""

    roomy = fit_preset(presets()["supercell-250m"], CARD_32GB, "10m", host=HOST)
    assert roomy.offered and "runtime.tile_nx" not in roomy.gm1_settings
    assert roomy.estimate.tile.startswith("whole domain")
    tight = fit_preset(presets()["supercell-250m"], HEADLESS_16GB, "1h", host=HOST)
    assert tight.offered and tight.gm1_settings["runtime.tile_nx"] == 128
    assert tight.estimate.step["closure_tiles"] > 1
