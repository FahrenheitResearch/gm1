"""Measure speed on a real GPU: the timing harness runs, the rate is learned.

Opt-in (``CM1_RUN_GPU_TESTS=1``), with the Rust initializer in
``GM1_CM1_STATE_LIBRARY`` when this is a source checkout.  About 30 s on an
RTX 3080: 40 steps of the stock 1 km deck plus start-up.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from cm1core.cm1run import rates
from cm1core.cm1run.estimate import deck_from_mapping, measure_speed, price
from cm1core.cm1run.gpuinfo import HostInfo, query_gpus

ROOT = Path(__file__).resolve().parents[1]
STOCK_TOML = ROOT / "tests" / "data" / "decks" / "stock-imported.gpuwm-cm1.toml"

pytestmark = pytest.mark.skipif(
    os.environ.get("CM1_RUN_GPU_TESTS") != "1",
    reason="GPU test: set CM1_RUN_GPU_TESTS=1 (and GM1_CM1_STATE_LIBRARY in a checkout)",
)


def test_measure_speed_runs_the_deck_and_the_next_estimate_uses_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    gpu = query_gpus().select()
    assert gpu is not None, "no GPU"
    raw = tomllib.loads(STOCK_TOML.read_text())
    raw["runtime"]["memory_budget_mib"] = 5000
    deck = deck_from_mapping(raw, name="stock")
    point, receipt = measure_speed(
        deck, gpu, steps=40, library=os.environ.get("GM1_CM1_STATE_LIBRARY"), workdir=tmp_path / "measure",
    )
    assert receipt.is_file()
    assert point.cells == 120 * 120 * 40 and point.steps >= 30
    assert 0.01 < point.seconds_per_step < 1.0
    # The door learned the run as it wrote the receipt; "Measure speed" found
    # it already known, so the cache holds it once.
    cached = rates.read_cache()
    assert len(cached) == 1 and cached[0].receipt == str(receipt.resolve())
    host = HostInfo(64 * 10**9, 60 * 10**9, str(tmp_path), 10**12, 10**11)
    estimate = price(deck, gpu, host=host)
    assert estimate.step["floor_label"] == "measured-here"
    assert estimate.step["floor_s"] == pytest.approx(point.seconds_per_step)
