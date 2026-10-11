"""``gpuwm-cm1 estimate``: memory, time, the storm-track check, learning.

The numbers are checked against the receipts the rates record cites, by the
record's receipt ids (the development repository keeps the files), so this
file runs on any CPU machine: no GPU, no CuPy.  ``nvidia-smi`` is the fake in
``tests/data/gpuinfo/`` answering from a fixture.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from cm1core.cm1run import rates
from cm1core.cm1run.estimate import (
    FITS_FRACTION_OF_FREE,
    EstimateError,
    deck_from_mapping,
    load_deck,
    price,
    storm_track,
)
from cm1core.cm1run.gpuinfo import GpuInfo, HostInfo, query_gpus
from cm1core.config import parse_mapping

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "gpuinfo"
FAKE_SMI = FIXTURES / "fake_nvidia_smi.py"
#: Receipt ids of GM1-GPU-RATES.json this file reads by name.
STOCK_RUN = "rtx3080-120x120x40-morrison"            # the stock deck, 0.58 M cells
TILE128_RUN = "rtx3080-256x256x80-morrison-tile128"  # 5.24 M cells, tile 128
RTX5090_RUN = "rtx5090-640x768x80-morrison-tile128"  # 39.3 M cells, tile 128
STOCK_TOML = ROOT / "tests" / "data" / "decks" / "stock-imported.gpuwm-cm1.toml"
CM1_STORM = ROOT / "evidence" / "oracle" / "supercell-fidelity" / "storm-morrison.json"
MIB = 1024 * 1024
GB = 1e9

HOST = HostInfo(
    ram_total_bytes=64 * 10**9, ram_available_bytes=60 * 10**9, disk_root="/runs",
    disk_total_bytes=2000 * 10**9, disk_free_bytes=1000 * 10**9, cpu_count=16,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """No test reads or writes the user's own measured rates."""

    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))


def fixture_gpus(name: str, monkeypatch, *, wsl2: bool = False):
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(FAKE_SMI))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(FIXTURES / f"{name}.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1" if wsl2 else "0")
    return query_gpus()


def rates_files(run: str) -> dict[str, Path]:
    """Where the development repository keeps receipt ``run`` of the rates record, its deck and its events."""

    from dev_records import RATES_RECEIPT_FILES  # absent from the public snapshot

    return RATES_RECEIPT_FILES[run]


def rates_deck(run: str) -> dict:
    return tomllib.loads(rates_files(run)["deck"].read_text(encoding="utf-8"))


def rates_receipt(run: str) -> dict:
    return json.loads(gzip.decompress(rates_files(run)["path"].read_bytes()))


def process_peak(row_run: str) -> tuple[int, int]:
    for row in rates.load_record()["memory_model"]["rows"]:
        if row["run"] == row_run:
            return int(row["process_peak_bytes"]), int(row["cells"])
    raise KeyError(row_run)


# ------------------------------------------------------------------ gpuinfo


def test_the_wsl2_desktop_fixture_reads_as_memory_actually_free(monkeypatch) -> None:
    query = fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True)
    assert query.status == "ok"
    gpu = query.select()
    assert gpu.name == "NVIDIA GeForce RTX 3080"
    assert gpu.total_bytes == 10240 * MIB
    # free = total - used - reserved, as nvidia-smi's memory.free says
    assert gpu.free_bytes == (10240 - round(4.3e9 / MIB) - 187) * MIB
    assert gpu.in_use_bytes > 4.3e9
    assert gpu.compute_apps == ()
    assert gpu.processes_visible is False
    assert gpu.sm == "sm_86"
    assert "Windows desktop" in gpu.memory_line()


def test_an_old_driver_is_asked_the_short_query(monkeypatch) -> None:
    gpu = fixture_gpus("a100-80gb-old-driver", monkeypatch).select()
    assert gpu.name == "NVIDIA A100-SXM4-80GB"
    assert gpu.reserved_bytes is None and gpu.compute_capability is None
    assert gpu.free_bytes == 81408 * MIB


def test_two_gpus_are_listed_and_selected_by_index_with_their_processes(monkeypatch) -> None:
    query = fixture_gpus("two-gpus-one-busy", monkeypatch)
    assert [gpu.index for gpu in query.gpus] == [0, 1]
    busy = query.select(0)
    assert busy.compute_apps and busy.compute_apps[0].pid == 962779
    assert query.select(1).name == "NVIDIA GeForce RTX 4090"
    with pytest.raises(ValueError, match="no GPU 7"):
        query.select(7)


def test_a_broken_driver_and_a_missing_nvidia_smi_are_answers_not_crashes(monkeypatch) -> None:
    query = fixture_gpus("broken-driver", monkeypatch)
    assert query.status == "error" and "couldn't communicate" in query.message
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", "none")
    query = query_gpus()
    assert query.status == "no-nvidia-smi" and query.select() is None


# -------------------------------------------------- memory


@pytest.mark.parametrize(
    "run",
    [
        STOCK_RUN,
        TILE128_RUN,
        "rtx3080-256x256x80-morrison-whole-tile",  # 5.24 M, whole tile
        "rtx3080-120x120x40-wsm6",
        "rtx3080-256x256x80-wsm6-tile128",
    ],
)
def test_memory_is_within_ten_percent_of_the_measured_process_peak(run) -> None:
    measured, cells = process_peak(run)
    estimate = price(deck_from_mapping(rates_deck(run), name=run), None, host=HOST)
    assert estimate.cells == cells
    assert abs(estimate.gpu_need.value / measured - 1.0) <= 0.10, (estimate.gpu_need.value, measured)


def test_memory_is_within_ten_percent_at_39_million_cells() -> None:
    """The RTX 5090 receipt: 640 x 768 x 80, tile 128, 19.38 GB measured."""

    estimate = price(deck_from_mapping(rates_deck(RTX5090_RUN), name=RTX5090_RUN), None, host=HOST)
    measured, cells = process_peak(RTX5090_RUN)
    assert estimate.cells == cells == 39_321_600
    assert abs(estimate.gpu_need.value / measured - 1.0) <= 0.10


@pytest.mark.parametrize("row", [row["run"] for row in rates.load_record()["memory_model"]["rows"]])
def test_every_recorded_memory_row_is_reproduced_from_its_own_deck(row) -> None:
    """Each row's deck, priced as a user's would be, lands within 10 % of the
    process peak its run measured -- the build rows and the out-of-sample
    verification runs (128 x 128 x 40, 128 x 128 x 80, 192 x 192 x 80) alike."""

    from dev_records import RATES_RECEIPT_FILES  # absent from the public snapshot

    record = rates.load_record()
    measured = next(item for item in record["memory_model"]["rows"] if item["run"] == row)
    deck_path = RATES_RECEIPT_FILES[row]["deck"]
    raw = tomllib.loads(deck_path.read_text(encoding="utf-8"))
    estimate = price(deck_from_mapping(raw, name=row), None, host=HOST)
    assert estimate.cells == measured["cells"]
    assert estimate.priced_peak_bytes == measured["priced_peak_bytes"]
    assert abs(estimate.gpu_need.value / measured["process_peak_bytes"] - 1.0) <= 0.10


def test_the_build_fit_was_low_out_of_sample() -> None:
    """Why the residual moved again: 0.48 GB + 78 B/cell was 11 % low at
    128 x 128 x 80, the direction that says fits and then runs out of memory."""

    row = next(
        item for item in rates.load_record()["memory_model"]["rows"]
        if item["run"] == "rtx3080-128x128x80-morrison"
    )
    old = row["priced_peak_bytes"] + 0.48e9 + 78 * row["cells"]
    assert old / row["process_peak_bytes"] < 0.90


def test_the_old_fit_would_have_failed_at_the_stock_deck() -> None:
    """Why the residual was refitted: 1.2 GB + 461 B/cell is 44 % high at 0.58 M."""

    measured, cells = process_peak("rtx3080-120x120x40-morrison")
    assert (1.2e9 + 461 * cells) / measured - 1.0 > 0.4


def test_the_host_locked_memory_is_the_two_arenas_the_receipt_reports() -> None:
    estimate = price(deck_from_mapping(rates_deck(TILE128_RUN), name=TILE128_RUN), None, host=HOST)
    receipt = rates_receipt(TILE128_RUN)
    arenas = receipt["runtime_benchmark"]["publication"]["host_arenas"]
    # This receipt was measured before restart profile v2 put the run's swaths in its
    # restarts (restart_write.F:696-758); a run of the same deck now adds the
    # swath group, whose bytes are the layout's own words.
    from cm1core.cartesian_door import build_runtime_config
    from cm1core.cartesian_tornado_diagnostics import swath_restart_layout

    config = parse_mapping(rates_deck(TILE128_RUN))
    layout = swath_restart_layout(
        build_runtime_config(config),
        total_steps=int(round(config.run.duration_s / config.run.dt_s)),
    )
    swath_bytes = sum(4 * math.prod(shape) for *_, shape, _key in layout.carriers())
    assert 0 < swath_bytes < 0.01 * arenas["restart"]["capacity_bytes"]
    measured = arenas["history"]["capacity_bytes"] + arenas["restart"]["capacity_bytes"] + swath_bytes
    # The restart arena is the plan's payload exactly; the history arena adds
    # its field table and alignment, 1.4 MB at 5.24 M cells.
    assert arenas["restart"]["capacity_bytes"] + swath_bytes == estimate.restart_bytes
    assert estimate.host_locked.value == pytest.approx(measured, rel=1e-3)


# ------------------------------------------------ wall time


def _events_wall(path: Path) -> float:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(row["elapsed_s"] for row in rows if row["kind"] == "end")


def test_the_stock_deck_wall_estimate_is_within_25_percent_on_the_3080(monkeypatch) -> None:
    gpu = fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True).select()
    estimate = price(deck_from_mapping(tomllib.loads(STOCK_TOML.read_text()), name="stock"), gpu, host=HOST)
    from dev_paths import dev_path

    imported_wall = _events_wall(dev_path("stock-imported-events"))
    native_wall = json.loads(dev_path("stock-native-wall").read_text())["launch_to_exit_s"]
    assert imported_wall == pytest.approx(73.2, abs=0.5)  # 72 s launch to exit, as recorded
    for measured in (72.0, imported_wall, native_wall):
        assert abs(estimate.wall_s / measured - 1.0) <= 0.25, (estimate.wall_s, measured)
    # A wall time for a run that has not happened is an estimate; what it
    # rests on is the card's measured rate.
    assert estimate.wall.label == "estimated" and estimate.wall.basis_label == "measured"
    assert estimate.gpu_need.label == "estimated" and estimate.gpu_need.basis_label == "measured"
    assert "estimated from a measured rate" in estimate.text()


def test_the_256x256x80_step_estimate_is_within_20_percent_of_0_2283(monkeypatch) -> None:
    gpu = fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True).select()
    estimate = price(deck_from_mapping(rates_deck(TILE128_RUN), name=TILE128_RUN), gpu, host=HOST)
    assert abs(estimate.step["seconds_per_step"] / 0.2283 - 1.0) <= 0.20
    assert estimate.step["regime"] == "per-cell"


def test_the_5090_row_reproduces_its_own_receipt() -> None:
    gpu = GpuInfo(0, "GPU-x", "NVIDIA GeForce RTX 5090", 32607 * MIB, 2 * MIB, 32155 * MIB, 450 * MIB, "12.0")
    estimate = price(deck_from_mapping(rates_deck(RTX5090_RUN), name=RTX5090_RUN), gpu, host=HOST)
    receipt = rates_receipt(RTX5090_RUN)
    assert abs(estimate.wall_s / receipt["process_wall_seconds"] - 1.0) <= 0.25


# ------------------------------------- never "fits" when it does not


FIXTURE_NAMES = (
    "rtx3080-wsl2-desktop", "rtx3080-wsl2-desktop-4p8", "rtx4060-8gb-desktop",
    "rtx4080-16gb-headless", "rtx5090-32gb", "two-gpus-one-busy", "a100-80gb-old-driver", "h100-sm90",
)


def _scaled_stock(nx: int, dx: float, nz: int = 80) -> dict:
    raw = tomllib.loads(STOCK_TOML.read_text())
    raw["grid"].update(nx=nx, ny=nx, nz=nz, dx_m=dx, dy_m=dx, dz_m=250.0)
    raw["run"].update(dt_s=1.5 * dx / 250.0)
    raw["runtime"].update(acoustic_substeps=4, tile_nx=128, tile_ny=128, memory_budget_mib=60000)
    return raw


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_it_never_says_fits_when_the_need_exceeds_free_memory(fixture, monkeypatch) -> None:
    query = fixture_gpus(fixture, monkeypatch, wsl2=fixture.startswith("rtx3080-wsl2"))
    for gpu in query.gpus:
        for nx in (128, 256, 384, 512, 640):
            estimate = price(deck_from_mapping(_scaled_stock(nx, 250.0), name=f"{nx}"), gpu, host=HOST)
            says = estimate.gpu_fits
            assert says == (estimate.gpu_need.value <= FITS_FRACTION_OF_FREE * gpu.free_bytes)
            if estimate.gpu_need.value > gpu.free_bytes:
                assert says is False and estimate.fits is False
                assert any(check.id == "gpu-memory" and check.level == "stop" for check in estimate.checks)


def test_the_windows_desktop_share_is_what_decides(monkeypatch) -> None:
    """A deck that fits the RTX 3080's 10 GB but not what the desktop leaves."""

    gpu = fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True).select()
    estimate = price(deck_from_mapping(_scaled_stock(384, 250.0), name="384"), gpu, host=HOST)
    assert estimate.gpu_need.value < 0.95 * gpu.total_bytes
    assert estimate.gpu_need.value > gpu.free_bytes
    assert estimate.gpu_fits is False
    shared = [check for check in estimate.checks if check.id == "gpu-shared"]
    assert shared and "Windows desktop" in shared[0].text


def test_a_never_run_architecture_is_named(monkeypatch) -> None:
    gpu = fixture_gpus("h100-sm90", monkeypatch).select()
    estimate = price(deck_from_mapping(tomllib.loads(STOCK_TOML.read_text()), name="stock"), gpu, host=HOST)
    assert any(check.id == "architecture" and "sm_90" in check.text for check in estimate.checks)
    assert estimate.wall.basis_label == "rough"
    assert "rough: no GM1 run on this card" in estimate.text()


def test_a_busy_gpu_names_the_program_holding_it(monkeypatch) -> None:
    gpu = fixture_gpus("two-gpus-one-busy", monkeypatch).select(0)
    estimate = price(deck_from_mapping(tomllib.loads(STOCK_TOML.read_text()), name="stock"), gpu, host=HOST)
    busy = [check for check in estimate.checks if check.id == "gpu-busy"]
    assert busy and "962779" in busy[0].text


def test_the_whole_domain_tile_is_priced_as_written_and_tile_128_recommended() -> None:
    raw = _scaled_stock(480, 250.0)
    del raw["runtime"]["tile_nx"], raw["runtime"]["tile_ny"]
    raw["runtime"]["memory_budget_mib"] = 8192
    estimate = price(deck_from_mapping(raw, name="480 whole"), None, host=HOST)
    assert estimate.tile.startswith("whole domain")
    assert estimate.recommendation["tile_nx"] == 128
    assert estimate.recommendation["gpu_memory_bytes"] < 0.6 * estimate.gpu_need.value
    budget = [check for check in estimate.checks if check.id == "memory-budget"]
    assert budget and budget[0].level == "stop" and "tile_nx = tile_ny = 128" in budget[0].text


# ---------------------------------------------- no GPU, fast


def test_the_command_finishes_in_under_two_seconds_with_no_gpu(tmp_path) -> None:
    env = {
        **os.environ,
        "GPUWM_CM1_NVIDIA_SMI": "none",
        "GPUWM_CM1_CACHE_DIR": str(tmp_path / "cache"),
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "cm1core", "estimate", str(STOCK_TOML), "--json", "--root", str(tmp_path)],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )
    elapsed = time.perf_counter() - started
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["gpu"] is None and payload["fits"]["gpu"] is None
    assert payload["gpu_query"]["status"] == "no-nvidia-smi"
    assert elapsed < 2.0, elapsed


def test_pricing_never_imports_cupy(tmp_path) -> None:
    code = (
        "import sys, tomllib\n"
        "from cm1core.cm1run.estimate import price, deck_from_mapping\n"
        "from cm1core.cm1run.gpuinfo import HostInfo\n"
        f"raw = tomllib.loads(open({str(STOCK_TOML)!r}).read())\n"
        "price(deck_from_mapping(raw), None, host=HostInfo(1, 1, '/', 1, 1))\n"
        "assert 'cupy' not in sys.modules, 'cupy imported'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GPUWM_CM1_CACHE_DIR": str(tmp_path)}
    completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr


# ------------------------------------ learning from a run


def _receipt_with_step(seconds: float) -> dict:
    receipt = rates_receipt(TILE128_RUN)
    receipt["runtime_benchmark"]["step_wall_seconds"]["mean"] = seconds
    return receipt


def test_after_a_run_the_next_estimate_uses_the_measured_rate(tmp_path, monkeypatch) -> None:
    gpu = fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True).select()
    deck = deck_from_mapping(rates_deck(TILE128_RUN), name=TILE128_RUN)
    before = price(deck, gpu, host=HOST)
    assert before.step["label_used"] == "measured"
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(_receipt_with_step(0.300)))
    point, added = rates.learn_receipt(receipt)
    assert added and point.cells == 5_242_880 and point.gpu_name == "NVIDIA GeForce RTX 3080"
    after = price(deck, gpu, host=HOST)
    assert after.step["label_used"] == "measured-here"
    assert after.step["seconds_per_step"] == pytest.approx(0.300, rel=1e-6)
    assert str(receipt.resolve()) in after.step["receipts"]
    # Learning the same receipt twice changes nothing.
    assert rates.learn_receipt(receipt)[1] is False


def test_a_small_run_teaches_the_floor_not_the_per_cell_rate(tmp_path, monkeypatch) -> None:
    gpu = fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True).select()
    receipt = rates_receipt(STOCK_RUN)
    receipt["runtime_benchmark"]["step_wall_seconds"]["mean"] = 0.061
    rates.learn_receipt(receipt)
    stock = price(deck_from_mapping(tomllib.loads(STOCK_TOML.read_text()), name="stock"), gpu, host=HOST)
    assert stock.step["floor_s"] == pytest.approx(0.061)
    big = price(deck_from_mapping(rates_deck(TILE128_RUN), name=TILE128_RUN), gpu, host=HOST)
    assert big.step["ns_per_cell_step"] == pytest.approx(41.8)


def test_estimating_a_run_folder_learns_its_finished_segments_and_writes_its_files(tmp_path, monkeypatch) -> None:
    fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True)
    rundir = tmp_path / "run1"
    (rundir / "segments" / "001").mkdir(parents=True)
    (rundir / "deck.toml").write_text(rates_files(TILE128_RUN)["deck"].read_text())
    (rundir / "segments" / "001" / "receipt.json").write_text(json.dumps(_receipt_with_step(0.25)))
    from cm1core.cli import main

    assert main(["estimate", str(rundir), "--json"]) == 0
    written = json.loads((rundir / "estimate.json").read_text())
    assert written["time"]["step"]["seconds_per_step"] == pytest.approx(0.25)
    assert written["learned"][0]["seconds_per_step"] == pytest.approx(0.25)
    geometry = json.loads((rundir / "geometry.json").read_text())
    assert geometry["schema"] == "gpuwm-cm1.storm-track/v1"
    assert written["command"].startswith("gpuwm-cm1 estimate ")


# -------------------------------------------- storm track


def test_the_stock_deck_track_is_evaluated_and_consistent_with_cm1s_max_w_columns() -> None:
    """CM1's max-w column at 5 km over 900-7200 s: rows 49-65, columns 52-58 of 120."""

    config = parse_mapping(tomllib.loads(STOCK_TOML.read_text()))
    track = storm_track(config, sample_interval_s=900.0)
    assert track["status"] == "evaluated"
    assert track["stays_inside"] is True and track["meets_edge_rule"] is True
    assert "iwnd=2" in track["wind_source"]
    cm1 = json.loads(CM1_STORM.read_text())["table"]["cm1"]
    columns = {int(float(t)): row["wmax_5km_ij"] for t, row in cm1.items()}
    assert sorted(columns) == [900, 1800, 2700, 3600, 4500, 5400, 6300, 7200]
    rows = [ij[0] for ij in columns.values()]
    cols = [ij[1] for ij in columns.values()]
    assert (min(rows), max(rows), min(cols), max(cols)) == (49, 65, 52, 58)
    # Both say the storm stays well inside: CM1's column never came nearer an
    # edge than the admission clearance, and neither does the predicted track.
    cm1_clearance = min(min(i + 0.5, 119.5 - i, j + 0.5, 119.5 - j) for j, i in columns.values())
    assert cm1_clearance >= track["admission_clearance_km"]
    assert track["minimum_edge_clearance_km"] >= track["admission_clearance_km"]
    # And the predicted right mover is where CM1's strongest updraft is, to
    # within the check's own stated uncertainty: the bubble radius plus the
    # steering allowance (4 m/s over the 30-min unorganised phase).
    origin_x, origin_y = track["origin_km"]
    allowed = 10.0 + 4.0 * 1800.0 / 1000.0
    worst = 0.0
    for t, x, y in track["track_km"]["right"]:
        if int(t) not in columns:
            continue
        j, i = columns[int(t)]
        cm1_x, cm1_y = origin_x + i + 0.5, origin_y + j + 0.5
        worst = max(worst, math.hypot(x - cm1_x, y - cm1_y))
    assert worst <= allowed, worst


def test_the_analytic_hodograph_is_base_f_s_weisman_klemp_profile() -> None:
    from cm1core.run_geometry import analytic_base_wind

    # base.F:2321-2357 at the quarter-circle's ends and on the linear leg.
    assert analytic_base_wind(2, 0.0) == pytest.approx((0.0, 0.0))
    assert analytic_base_wind(2, 2000.0) == pytest.approx((7.0, 7.0))
    assert analytic_base_wind(2, 4000.0) == pytest.approx((19.0, 7.0))
    assert analytic_base_wind(2, 9000.0) == pytest.approx((31.0, 7.0))
    assert analytic_base_wind(1, 1250.0) == pytest.approx((5.0, 0.0))
    assert analytic_base_wind(4, 3000.0)[0] == pytest.approx(35.0 * math.tanh(1.0))
    assert analytic_base_wind(6, 100.0, testcase=4) == (7.0, -5.5)
    with pytest.raises(ValueError, match="iwnd=7"):
        analytic_base_wind(7, 100.0)


def test_a_repeating_side_is_not_an_edge_and_a_doubly_periodic_one_is_not_checked() -> None:
    raw = tomllib.loads(STOCK_TOML.read_text())
    raw["boundaries"].update(south=1, north=1)
    periodic_y = storm_track(parse_mapping(raw))
    assert periodic_y["status"] == "evaluated"
    assert periodic_y["open_sides"] == {"west": True, "east": True, "south": False, "north": False}
    raw["boundaries"].update(west=1, east=1)
    assert storm_track(parse_mapping(raw))["status"] == "not-evaluated-doubly-periodic"


def test_the_recommended_frame_is_a_binary32_word_the_config_admits() -> None:
    raw = tomllib.loads(STOCK_TOML.read_text())
    raw["grid"].update(nx=60, ny=60)
    track = storm_track(parse_mapping(raw))
    assert track["stays_inside"] is False
    u, v = track["recommended_frame_m_s"]
    raw["closures"].update(u_move_m_s=u, v_move_m_s=v)
    parse_mapping(raw)  # CM1-CFG-MOVING-FRAME-PRECISION would refuse any other


# ----------------------------------------------------------- refusals


def test_a_refused_deck_is_reported_with_its_remedy() -> None:
    raw = tomllib.loads(STOCK_TOML.read_text())
    raw["grid"]["stretch_x"] = True
    with pytest.raises(EstimateError) as caught:
        price(deck_from_mapping(raw), None, host=HOST)
    assert caught.value.refusal["status"] == "refused"
    assert "What to change" in str(caught.value)


def test_load_deck_reads_a_toml_and_refuses_a_missing_path(tmp_path) -> None:
    deck = load_deck(STOCK_TOML)
    assert deck.kind == "toml" and deck.raw["grid"]["nx"] == 120
    with pytest.raises(EstimateError, match="does not exist"):
        load_deck(tmp_path / "nothing.toml")


def test_the_door_learns_every_run_it_writes_a_receipt_for(tmp_path, monkeypatch) -> None:
    """After every run the measured rate is kept and used next time."""

    from cm1core.cartesian_door import LEARN_RATES_ENV, learn_step_rate

    receipt = _receipt_with_step(0.2500)
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps(receipt))
    assert learn_step_rate(receipt, output) is True
    points = rates.read_cache()
    assert len(points) == 1 and points[0].receipt == str(output.resolve())
    assert points[0].seconds_per_step == pytest.approx(0.25)
    # The same receipt again adds nothing; one with no step time is not
    # learned and never raises; and the switch turns it off.
    assert learn_step_rate(receipt, output) is False
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"runtime_benchmark": {}}))
    assert learn_step_rate({"runtime_benchmark": {}}, empty) is False
    monkeypatch.setenv(LEARN_RATES_ENV, "0")
    other = tmp_path / "other.json"
    other.write_text(json.dumps(_receipt_with_step(0.26)))
    assert learn_step_rate(None, other) is False
    assert len(rates.read_cache()) == 1


def test_a_grid_with_no_writable_restart_is_stopped_by_name() -> None:
    """1200 x 1200 x 75 carries 24 GB of restart state, past the 16 GiB CM1FCP1 cap."""

    deck = load_deck(ROOT / "examples" / "cartesian-stretched-tornado-grade-grid.toml")
    estimate = price(deck, None, host=HOST)
    stops = {check.id: check.text for check in estimate.checks if check.level == "stop"}
    assert "restart-size" in stops and "CM1FCP1" in stops["restart-size"]
    assert estimate.fits is False


# ------------------------------------------- verification additions


def _receipt_for_grid(nx: int, ny: int, nz: int, seconds: float, *, count: int = 200) -> dict:
    receipt = _receipt_with_step(seconds)
    receipt["runtime_benchmark"]["config"]["grid"].update(nx=nx, ny=ny, nz=nz)
    receipt["runtime_benchmark"]["step_wall_seconds"]["count"] = count
    return receipt


def test_a_run_in_the_crossover_band_prices_its_own_grid_next_time(monkeypatch) -> None:
    """128 x 128 x 80 (1.31 M cells) sits between the floor and the per-cell
    regime on the RTX 3080, where a point measures neither term cleanly; the
    next estimate of that grid must still be its own measured step."""

    gpu = fixture_gpus("rtx3080-wsl2-desktop", monkeypatch, wsl2=True).select()
    deck = deck_from_mapping(_scaled_stock(128, 250.0), name="128")
    before = price(deck, gpu, host=HOST)
    assert before.step["label_used"] == "measured" and before.step["regime"] in ("floor", "per-cell")
    point, added = rates.learn_receipt(_receipt_for_grid(128, 128, 80, 0.080))
    assert added and point.cells == 1_310_720
    after = price(deck, gpu, host=HOST)
    assert after.step["regime"] == "this-grid"
    assert after.step["label_used"] == "measured-here"
    assert after.step["seconds_per_step"] == pytest.approx(0.080)
    assert after.wall.basis_label == "measured-here"
    assert "rate measured on this GPU" in after.text()
    # A grid of another size is still priced by the table's two terms.
    other = price(deck_from_mapping(rates_deck(TILE128_RUN), name=TILE128_RUN), gpu, host=HOST)
    assert other.step["regime"] == "per-cell" and other.step["ns_per_cell_step"] == pytest.approx(41.8)


def test_a_nested_run_or_a_very_short_one_teaches_nothing(tmp_path) -> None:
    from cm1core.cartesian_door import learn_step_rate

    nested = _receipt_with_step(0.9)
    nested["nest"] = {"child": {"nx": 256}}
    with pytest.raises(rates.ReceiptError, match="nest"):
        rates.learn_receipt(nested)
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(nested))
    assert learn_step_rate(nested, path) is False
    with pytest.raises(rates.ReceiptError, match="at least 20"):
        rates.learn_receipt(_receipt_for_grid(256, 256, 80, 0.3, count=8))
    assert rates.read_cache() == []


def test_the_disk_rule_matches_what_start_refuses() -> None:
    """Start refuses above free space minus 10 %, so the estimate
    never says fits there."""

    stock = deck_from_mapping(tomllib.loads(STOCK_TOML.read_text()), name="stock")
    need = price(stock, None, host=HOST).disk.value
    for share, fits, level in ((0.95, False, "stop"), (0.85, True, "warn"), (0.5, True, "ok")):
        host = HostInfo(64 * 10**9, 60 * 10**9, "/runs", 2000 * 10**9, int(need / share), 16)
        estimate = price(stock, None, host=host)
        assert estimate.disk_fits is fits, share
        assert [check.level for check in estimate.checks if check.id == "disk"] == [level]


def _fake_harness(tmp_path: Path, gpu_name: str, seconds: float) -> Path:
    """A stand-in for ``python -m cm1core cartesian``: writes a receipt for the deck it is given."""

    script = tmp_path / "fake_python"
    script.write_text(
        f"#!{sys.executable}\n"
        "import gzip, json, sys, tomllib\n"
        "args = sys.argv[1:]\n"
        "deck = tomllib.loads(open(args[3]).read())\n"
        "out = args[args.index('--output') + 1]\n"
        f"receipt = json.loads(gzip.decompress(open({str(rates_files(TILE128_RUN)['path'])!r}, 'rb').read()))\n"
        "bench = receipt['runtime_benchmark']\n"
        "bench['config']['grid'].update(nx=deck['grid']['nx'], ny=deck['grid']['ny'], nz=deck['grid']['nz'])\n"
        "runtime = deck.get('runtime', {})\n"
        "bench['config']['tile_shape_yx'] = [runtime.get('tile_ny', deck['grid']['ny']), runtime.get('tile_nx', deck['grid']['nx'])]\n"
        f"bench['step_wall_seconds'].update(mean={seconds!r}, count=57)\n"
        f"bench['gpu_name'] = {gpu_name!r}\n"
        "open(out, 'w').write(json.dumps(receipt))\n"
    )
    script.chmod(0o755)
    return script


def test_measure_speed_gives_the_harness_the_budget_its_own_peak_needs(tmp_path, monkeypatch) -> None:
    """A 600 x 600 x 80 deck prices well over 8 GB of registry; the importer's
    8192 MiB would have refused the measurement on a 32 GB card."""

    from cm1core.cm1run.estimate import measure_speed

    gpu = fixture_gpus("rtx5090-32gb", monkeypatch).select()
    raw = _scaled_stock(600, 100.0)
    del raw["runtime"]["tile_nx"], raw["runtime"]["tile_ny"]
    raw["runtime"]["memory_budget_mib"] = 8192
    deck = deck_from_mapping(raw, name="600")
    estimate = price(deck, gpu, host=HOST)
    fake = _fake_harness(tmp_path, gpu.name, 0.41)
    point, receipt = measure_speed(deck, gpu, steps=60, workdir=tmp_path / "m", python=str(fake))
    measured = tomllib.loads((tmp_path / "m" / "measure.toml").read_text())
    # It measures the tiling the estimate would have the run take.
    if estimate.recommendation.get("tile_nx"):
        assert measured["runtime"]["tile_nx"] == estimate.recommendation["tile_nx"]
        peak = price(deck.with_runtime(tile_nx=128, tile_ny=128), gpu, host=HOST).priced_peak_bytes
    else:
        assert "tile_nx" not in measured["runtime"]
        peak = estimate.priced_peak_bytes
    assert peak > 8192 * MIB
    assert peak <= measured["runtime"]["memory_budget_mib"] * MIB <= 0.95 * gpu.free_bytes
    assert point.cells == 600 * 600 * 80 and receipt.is_file()
    after = price(deck, gpu, host=HOST)
    assert after.step["regime"] == "this-grid" and after.step["seconds_per_step"] == pytest.approx(0.41)


def test_measure_speed_stops_before_running_a_deck_the_gpu_cannot_hold(tmp_path, monkeypatch) -> None:
    from cm1core.cm1run.estimate import measure_speed

    gpu = fixture_gpus("rtx4060-8gb-desktop", monkeypatch).select()
    deck = deck_from_mapping(_scaled_stock(640, 100.0), name="640")
    with pytest.raises(EstimateError, match="to measure"):
        measure_speed(deck, gpu, steps=60, workdir=tmp_path / "m", python="/nonexistent/python")
    with pytest.raises(ValueError, match="at least 25"):
        measure_speed(deck, gpu, steps=10, workdir=tmp_path / "m", python="/nonexistent/python")


def test_a_bad_receipt_to_learn_is_an_answer_not_a_traceback(tmp_path, capsys) -> None:
    from cm1core.cli import main

    bad = tmp_path / "receipt.json"
    bad.write_text(json.dumps({"runtime_benchmark": {}}))
    assert main(["estimate", "--learn", str(bad)]) == 2
    assert "was not learned" in capsys.readouterr().err
    assert main(["estimate", "--learn", str(tmp_path / "missing.json")]) == 2
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_receipt_with_step(0.22)))
    assert main(["estimate", str(STOCK_TOML), "--learn", str(good), "--json", "--root", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert f"--learn {good}" in payload["command"]


def test_every_kept_receipt_is_the_bytes_the_record_names() -> None:
    """The shipped record names each receipt by id and sha256 only; the development
    repository keeps the files, and each one still hashes to the record's digest."""

    from dev_records import RATES_RECEIPT_FILES  # absent from the public snapshot
    import hashlib

    receipts = rates.load_record()["receipts"]
    for rid, entry in receipts.items():
        if entry["receipt"] != "kept":
            assert "sha256_uncompressed" not in entry
            continue
        raw = gzip.decompress(RATES_RECEIPT_FILES[rid]["path"].read_bytes())
        assert hashlib.sha256(raw).hexdigest() == entry["sha256_uncompressed"], rid
    assert set(RATES_RECEIPT_FILES) <= set(receipts)
