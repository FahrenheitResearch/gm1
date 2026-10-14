"""Any grid spacing: ``--custom grid=67`` and the Customize panel's slider and number box.

GM1 does not turn a spacing down because it is unusual.  These tests hold the pieces that make an arbitrary spacing
a run: the time step made to land on the output times, the cells and the levels the model takes, the area the fit
goes down to (with a warning, not a refusal, when the storm nears the edge), the time with no limit, and the few
things that are still refused because they cannot run, each with what would fit.  CPU only, with fake GPUs; the
real native namelist parser and import.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from cm1core.cm1run import customize as C
from cm1core.cm1run import estimate as E
from cm1core.cm1run import fit as F
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


CARD_24GB = card("NVIDIA GeForce RTX 4090", 24564, 0.3e9)
CARD_8GB = card("NVIDIA GeForce RTX 3070", 8192, 0.3e9, cc="8.6")
#: The spacings the release was tried at, the menu's and the odd ones between and beyond.
SPACINGS_M = (33.0, 50.0, 67.0, 100.0, 125.0, 150.0, 200.0, 250.0, 333.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0,
              3000.0)
LEVELS_M = (100.0, 125.0, 250.0, 500.0)


@pytest.fixture(autouse=True)
def _parser_and_cache(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})}
    return subprocess.run([sys.executable, "-m", "cm1core", *args], capture_output=True, text=True,
                          env=environment, timeout=300, cwd=ROOT)


def fixture_env(name: str) -> dict:
    return {"GPUWM_CM1_NVIDIA_SMI": str(FIXTURES / "fake_nvidia_smi.py"),
            "FAKE_NVIDIA_SMI_FIXTURE": str(FIXTURES / f"{name}.json"), "GPUWM_CM1_WSL2": "0",
            "GPUWM_CM1_HOST_FIXTURE": str(FIXTURES / "host-64gb-1tb.json")}


def fitted(preset_id: str, budget: str, gpu: GpuInfo = CARD_24GB, **custom) -> tuple[P.Preset, dict]:
    preset = C.apply(P.load(preset_id), custom)
    result = P.fit(preset, budget, gpu=gpu, host=HOST)
    return preset, P.fit_record(preset, result, command="gpuwm-cm1 preset fit")


# ------------------------------------------------------------------ the time step, the cells, the levels


def test_every_spacing_from_20_m_to_4_km_gets_whole_steps_and_a_grid_the_model_takes():
    """Property-style: the sweep's every spacing, at four level heights and several output times."""

    sweep = sorted(set(range(20, 4001, 7)) | {int(value) for value in SPACINGS_M} | {21, 49, 99, 101, 999, 3999})
    outputs = ((7200.0, 900.0), (7200.0, 300.0), (3600.0, 60.0), (5400.0, 600.0), (1200.0, 120.0))
    for dx in sweep:
        for dz in LEVELS_M:
            raw = C.time_step(float(dx), dz)
            for index, (timax, tapfrq) in enumerate(outputs):
                if index and (dx % 5 or dz != 250.0):
                    continue  # every spacing at CM1's own output times, a fifth of them at the others
                dt = C.snapped_time_step(float(dx), dz, [timax, tapfrq])
                # No longer than CM1's rule gives, and not much shorter.
                # (A coarse grid's step is a whole fraction of a minute with few steps in it, 60/13 or 60/14 s, and
                # binary64 has no 60/13 that divides a minute exactly, so there the next count is taken.)
                assert 0.85 * raw < dt <= raw * (1.0 + 1e-12), (dx, dz, dt, raw)
                assert dt > 0.96 * raw or 60.0 / dt < 100.0, (dx, dz, dt, raw)   # within 4 % under 250 m or so
                # The runtime's own rule (config.py's clock alignment): each quotient is a whole binary64 number.
                for interval in (timax, tapfrq):
                    steps = interval / dt
                    assert steps >= 1.0 and float(steps).is_integer(), (dx, dz, interval, dt)
                # Whole steps a picture, and a whole minute: so other picture intervals stay possible.
                assert C.lands(60.0, dt) or C.lands(tapfrq, dt), (dx, dz, dt)
                # A domain that moves takes a short binary fraction: at most a fifth under CM1's rule, exact on
                # every output time, and a sum of the domain's position that never rounds (the run checks it).
                moving = C.snapped_time_step(float(dx), dz, [timax, tapfrq], moving=True)
                assert 0.8 * raw <= moving <= raw * (1.0 + 1e-12), (dx, dz, moving, raw)
                steps = timax / moving
                assert float(steps).is_integer() and float(tapfrq / moving).is_integer(), (dx, dz, moving)
                numerator, denominator = moving.as_integer_ratio()
                assert numerator < 256 and denominator & (denominator - 1) == 0, (dx, dz, moving)
                for speed in (12.5, 3.0, 11.375, 1.125, 17.25):
                    assert E.first_inexact_step(moving, speed, int(steps)) is None, (dx, dz, moving, speed)
        for side_km in (120.0, 60.0, 59.0, 300.0):
            cells = C.cells_across(side_km * 1000.0, float(dx))
            assert cells >= C.MIN_CELLS
            exact = side_km * 1000.0 / dx
            assert cells == max(C.MIN_CELLS, round(exact) if abs(exact - round(exact)) < 1e-6 else math.ceil(exact))
            # The area simulated is never smaller than the area asked for, and larger by less than one cell.
            assert -1e-6 <= cells * dx - side_km * 1000.0 < dx or cells == C.MIN_CELLS


def test_the_menu_spacings_keep_the_steps_they_always_had():
    for dx, dz, dt in ((1000.0, 500.0, 6.0), (500.0, 500.0, 3.0), (250.0, 250.0, 1.5), (250.0, 500.0, 1.5),
                       (2000.0, 500.0, 6.0), (1000.0, 250.0, 3.0), (100.0, 250.0, 0.6), (125.0, 250.0, 0.75)):
        assert C.snapped_time_step(dx, dz, [7200.0, 900.0]) == dt, (dx, dz)
    # 0.402 s (67 m) does not divide a minute: it is shortened to 0.4 s, 150 steps a minute, not refused.
    assert C.time_step(67.0, 500.0) == pytest.approx(0.402)
    assert C.snapped_time_step(67.0, 500.0, [7200.0, 900.0]) == 0.4
    # With a domain that moves the step is a short binary fraction: the menu's stay, the odd ones are the nearest.
    for dx, dz, dt in ((1000.0, 500.0, 6.0), (500.0, 500.0, 3.0), (250.0, 250.0, 1.5), (125.0, 250.0, 0.75),
                       (67.0, 500.0, 0.390625), (100.0, 250.0, 0.5859375), (150.0, 500.0, 0.78125),
                       (200.0, 500.0, 1.171875), (33.0, 500.0, 0.1953125)):
        assert C.snapped_time_step(dx, dz, [7200.0, 300.0], moving=True) == dt, (dx, dz)
    assert C.binary_steps(300.0) == [1.0, 1.171875, 1.25, 1.5, 1.5625, 1.875]
    assert C.cells_across(59_000.0, 67.0) == 881 and C.cells_across(120_000.0, 250.0) == 480
    assert C.cells_across(120_000.0, 2000.0) == 60        # a whole count is kept as it is
    assert C.levels_for(20_000.0, 125.0) == 160 and C.levels_for(20_000.0, 300.0) == 67


@pytest.mark.parametrize("dx", SPACINGS_M)
def test_the_guided_check_takes_every_spacing_tried(dx):
    """The released import and check, not a copy of their rules: each spacing imports as a run GM1 runs as CM1
    does, with the step, the cells and the area this file's other tests promise."""

    preset = P.load("supercell-1km")
    changed = C.apply(preset, {"grid": dx})
    examined = P.examine(changed)
    assert examined["offered"], (dx, examined["reasons"])
    effective = C._effective(changed)
    assert float(effective["dx"]) == dx == float(effective["dy"])
    dt = float(effective["dtl"])
    assert dt <= C.time_step(dx, 500.0) + 1e-12 and (7200.0 / dt).is_integer() and (900.0 / dt).is_integer()
    assert int(effective["nx"]) == int(effective["ny"]) >= 8
    assert int(effective["nx"]) * int(effective["ny"]) * int(effective["nz"]) <= C.START_CELLS_MAX
    if dx != 1000.0:
        sentence = changed.record["customized"][0]["sentence"]
        assert sentence.startswith(f"Grid spacing {C._words_m(dx)} instead of 1 km"), sentence
        assert changed.record["pinned"] == ["grid"]


@pytest.mark.parametrize("dz, levels", [(100.0, 200), (125.0, 160), (79.0, 253), (300.0, 67)])
def test_any_level_height_keeps_the_top_to_the_nearest_whole_level(dz, levels):
    preset = P.load("supercell-1km")
    changed = C.apply(preset, {"levels": dz})
    effective = C._effective(changed)
    assert (float(effective["dz"]), int(effective["nz"])) == (dz, levels)
    assert abs(levels * dz - 20_000.0) < dz / 2.0
    assert P.examine(changed)["offered"], P.examine(changed)["reasons"]
    dt = float(effective["dtl"])
    assert dt <= 0.012 * dz + 1e-12 and (7200.0 / dt).is_integer() and (900.0 / dt).is_integer()
    assert f"({levels} levels up to " in changed.record["customized"][0]["sentence"]


def test_levels_the_model_is_not_built_for_are_refused_with_what_fits():
    preset = P.load("supercell-1km")
    with pytest.raises(C.CustomError) as fine:
        C.apply(preset, {"levels": 50.0})
    assert str(fine.value) == ("Levels every 50 m up to 20 km would be 400 levels, and the model is built for at "
                               "most 256; the finest levels here are every 78 m.")
    assert fine.value.suggest == {"levels": 78.0}
    # What it says fits, fits.
    assert P.examine(C.apply(preset, {"levels": 78.0}))["offered"]
    assert C._effective(C.apply(preset, {"levels": 78.0}))["nz"] == 256
    # Seven levels are one too few (the run's own check needs 8): said here, with the thickest that are taken,
    # where a 7-level choice was once let through and then greyed with no reason.
    for thick in (2700.0, 2857.0, 3000.0, 4000.0):
        with pytest.raises(C.CustomError, match="needs at least 8") as few:
            C.apply(preset, {"levels": thick})
        assert str(few.value).endswith("the thickest levels here are every 2.666 km.")
        assert few.value.suggest == {"levels": 2666.0}
    thickest = C.apply(preset, {"levels": 2666.0})
    assert C._effective(thickest)["nz"] == 8 and P.examine(thickest)["offered"], P.examine(thickest)["reasons"]
    assert C.LEVELS_RANGE == (8, 256) and C.levels_bounds(20_000.0) == (78.0, 2666.0)


def test_a_spacing_that_cannot_be_a_grid_is_refused_with_the_coarsest_that_can():
    preset = P.load("supercell-1km")
    with pytest.raises(C.CustomError) as coarse:
        C.apply(preset, {"grid": 20000.0})
    assert str(coarse.value) == ("A 20 km grid leaves fewer than 8 cells across this storm's area, too few to be a "
                                 "grid; the coarsest grid here is 15 km.")
    assert P.examine(C.apply(preset, {"grid": 15000.0}))["offered"]
    for bad in (0.0, -67.0):
        with pytest.raises(C.CustomError, match="above zero"):
            C.apply(preset, {"grid": bad})


# ------------------------------------------------------------------ the fit


def test_67_m_is_fitted_and_says_the_area_and_the_cells_exactly():
    preset, record = fitted("supercell-1km", "any", grid=67.0)
    assert record["status"] == "fits", record["reason"]
    facts = record["facts"]
    # CM1's supercell moves its domain with the storm, so the step is a short binary fraction: 25/64 s.
    assert facts["dx_m"] == 67.0 and facts["dt_s"] == 0.390625 and facts["steps"] == 18432
    nx = facts["nx"]
    assert nx == facts["ny"]
    assert facts["exact"] == f"67 m grid, {nx * 0.067:.1f} × {nx * 0.067:.1f} km, {nx} × {nx} cells"
    assert f"({nx} × {nx} cells) at 67 m" in record["summary"]
    # The grid is the user's: the fit sized the area to the card and changed nothing else of theirs.
    assert record["pinned"] == ["grid"] and record["customized"][0]["knob"] == "grid"
    assert record["customized"][0]["sentence"].startswith(
        "Grid spacing 67 m instead of 1 km (and a 0.390625 s time step instead of 6 s).")
    assert record["estimate"]["gpu_memory_bytes"] <= 0.95 * CARD_24GB.free_bytes
    # Unusual, not refused: the notes say what is untried.
    assert any("finer than any storm GM1 ships (100 m)" in line for line in record["warnings"])
    kinds = [entry["kind"] for entry in record["warning_records"]]
    assert "spacing" in kinds
    if facts["cells"] > rates.load_record()["largest_domain_cells"]["value"]:
        assert "size" in kinds and any("untried" in line for line in record["warnings"])
    # fit.json is strict JSON, the time with no limit included.
    text = json.dumps(record, allow_nan=False, default=str)
    assert json.loads(text)["budget_s"] is None and record["budget_words"] == "as long as it takes"


def test_a_domain_larger_than_any_gm1_has_run_is_a_warning_not_a_refusal():
    largest = rates.load_record()["largest_domain_cells"]["value"]
    _, record = fitted("supercell-1km", "any", grid=100.0)
    assert record["status"] == "fits" and record["facts"]["cells"] > largest
    assert [entry["kind"] for entry in record["warning_records"]].count("size") == 1


def test_the_no_change_case_is_still_the_preset_byte_for_byte():
    preset = P.load("supercell-1km")
    assert C.apply(preset, {}) is preset
    plain = P.fit(preset, "1h", gpu=CARD_24GB, host=HOST)
    # The preset's own spacing, chosen by hand: pinned, and not one byte of the namelist differs.
    same = C.apply(preset, {"grid": 1000.0})
    assert same.namelist_text == preset.namelist_text and same.record["customized"] == []
    held = P.fit(same, "1h", gpu=CARD_24GB, host=HOST)
    assert held.offered and held.namelist_text == preset.namelist_text == plain.namelist_text
    assert held.warnings == () and plain.warnings == ()
    record = P.fit_record(preset, plain, command="x")
    assert record["warnings"] == [] and record["facts"]["exact"] == "1 km grid, 120.0 × 120.0 km, 120 × 120 cells"


def test_a_small_area_runs_with_a_warning_that_says_when_the_storm_nears_the_edge():
    """Below the area the preset was set up for the run is offered, and the storm-track check's answer is said."""

    preset, record = fitted("supercell-1km", "any", grid=100.0, area=24.0)
    assert record["status"] == "fits", record["reason"]
    assert record["facts"]["domain_km"] == [24.0, 24.0]
    edge = [entry for entry in record["warning_records"] if entry["kind"] == "edge"]
    assert len(edge) == 1, record["warning_records"]
    assert edge[0]["sentence"].startswith(
        "This 24 km area is smaller than the 64 km this storm was set up for: the storm may leave an area this "
        "small; pictures near the edges then show edge effects."), edge[0]["sentence"]
    assert " The storm-track check puts it closer than " in edge[0]["sentence"]
    assert edge[0]["sentence"].endswith(" from the start.") and "stays inside" not in edge[0]["sentence"]
    # The area is the user's, so Customize already moved the view with the storm (the track check's own frame).
    assert any(entry["knob"] == "follow" for entry in record["customized"])
    # The whole warm bubble still lies inside the area: the run's own start takes it.
    assert P.examine(preset)["offered"], P.examine(preset)["reasons"]


def test_an_area_the_fit_shrinks_offers_to_follow_the_storm():
    _, record = fitted("supercell-1km", "any", grid=33.0)
    assert record["status"] == "fits", record["reason"]
    edge = next(entry for entry in record["warning_records"] if entry["kind"] == "edge")
    area = edge["follow_area_km"]
    assert area == pytest.approx(max(record["facts"]["domain_km"]), abs=1e-3)
    assert f"--custom area={round(area, 2):g}" in edge["sentence"]
    assert "moves the view as the storm-track check predicts the storm moves" in edge["sentence"]
    assert edge["sentence"].endswith("; the prediction is not exact.") and "inside" not in edge["sentence"]
    assert " area is smaller than the " in edge["lead"] and "the storm may leave an area this small" in edge["lead"]
    assert edge["lead"].endswith("min.") or edge["lead"].endswith("from the start."), edge["lead"]
    # Taking the offer keeps the area and adds the frame.
    followed, again = fitted("supercell-1km", "any", grid=33.0, area=area)
    assert again["status"] == "fits" and again["facts"]["nx"] == record["facts"]["nx"]
    assert any(entry["knob"] == "follow" for entry in again["customized"])


def test_what_does_not_fit_even_the_smallest_area_says_so_and_what_would():
    _, record = fitted("supercell-1km", "any", grid=12.0)
    assert record["status"] == "not-offered" and record["reason_kind"] == "gpu"
    reason = record["reason"]
    assert "of free GPU memory even at its smallest size (20.1 km)" in reason
    # No "widest area that fits": below the smallest area a storm starts in there is no area to pick.
    assert "widest area" not in reason
    finest = record["needs"]["finest_grid_m"]
    # Priced, not only summed: the spacing named is one the fit takes (no "about").
    assert f"The finest grid that fits its smallest area (20 km) here is {finest:.0f} m." in reason
    assert record["needs"]["suggest"] == {"grid": finest}
    # It is true: that spacing fits, with the storm's smallest area, on the same card.
    _, fits = fitted("supercell-1km", "any", grid=float(finest))
    assert fits["status"] == "fits", fits["reason"]
    # On a smaller card the same spacing is the card's limit, not the menu's.
    _, small = fitted("supercell-1km", "any", CARD_8GB, grid=25.0)
    assert small["status"] == "not-offered" and small["reason_kind"] == "gpu"
    _, coarser = fitted("supercell-1km", "any", CARD_8GB, grid=67.0)
    assert coarser["status"] == "fits", coarser["reason"]


def test_a_time_too_short_for_the_smallest_area_is_said_as_time_not_as_memory():
    # 67 m on a 16 GB card: the whole area needs more memory than the card has, but a small area fits it, and
    # what stops that one in ten minutes is the time.  The reason says so, and the next time takes the choice.
    small = card("NVIDIA GeForce RTX 4080", 16376, 0.3e9)
    _, held = fitted("supercell-1km", "10m", small, grid=67.0)
    assert held["status"] == "not-offered" and held["reason_kind"] == "time", held["reason"]
    assert "pick a longer time than about 10 minutes" in held["reason"]
    assert "GPU memory" not in held["reason"]
    _, longer = fitted("supercell-1km", "1h", small, grid=67.0)
    assert longer["status"] == "fits", longer["reason"]
    assert longer["estimate"]["wall_s"] <= 3600.0


def test_overnight_is_ten_hours_and_the_time_with_no_limit_takes_what_it_cannot():
    assert F.BUDGETS["overnight"] == 36000.0 and math.isinf(F.BUDGETS["any"])
    assert P.BUDGET_ORDER == ("10m", "1h", "overnight", "any") and P.BUDGET_BUTTONS["any"] == "As long as it takes"
    # 50 m, the whole storm time and a 40 km area: more than ten hours on this card, and offered with no limit.
    choices = {"grid": 50.0, "area": 40.0, "time": 120.0}
    _, held = fitted("supercell-1km", "overnight", **choices)
    _, free = fitted("supercell-1km", "any", **choices)
    assert free["status"] == "fits", free["reason"]
    if held["status"] != "fits":
        assert held["reason_kind"] == "time" and free["estimate"]["wall_s"] > 36000.0


def test_the_100_m_preset_is_offered_again_and_takes_a_spacing_of_its_own():
    preset = P.load("supercell-100m")
    assert P.examine(preset)["offered"] and preset.record.get("offered") is not False
    plain = P.fit(preset, "overnight", gpu=CARD_24GB, host=HOST)
    assert plain.offered and plain.estimate.dt_s == 0.5
    _, record = fitted("supercell-100m", "any", grid=67.0)
    assert record["status"] == "fits" and record["facts"]["dt_s"] == 0.390625
    assert record["customized"][0]["sentence"].startswith(
        "Grid spacing 67 m instead of 100 m (and a 0.390625 s time step")


@pytest.mark.parametrize("dx", [33.0, 50.0, 67.0, 100.0, 125.0, 150.0, 200.0, 333.0, 750.0])
def test_the_runs_own_check_of_the_domains_position_never_trips_at_a_step_gm1_picks(dx):
    """cm1.F:3512-3513 through the run's own clock, unchanged: CM1's supercell frame (umove 12.5, vmove 3.0) and a
    frame the storm-track check recommends, at the step each spacing takes, for a whole two-hour run's steps."""

    from cm1core.gm1_runtime import RuntimeClock, RuntimeConfig
    from cm1core.grid_geometry import CartesianGridGeometry

    class Backend:
        def execute(self, operation, *, step):
            pass

        def synchronize(self):
            pass

    dt = C.snapped_time_step(dx, 500.0, [7200.0, 300.0], moving=True)
    steps = int(7200.0 / dt)
    for u, v in ((12.5, 3.0), (11.375, 1.125)):
        clock = RuntimeClock(RuntimeConfig(CartesianGridGeometry(64, 64, 40, dx, dx, 500), dt_s=dt,
                                           u_move_m_s=u, v_move_m_s=v))
        backend = Backend()
        for _ in range(min(steps, 20_000)):
            clock.step(backend)          # raises the moment the sum is not n * dt * umove
        assert clock.domainlocx_m == clock.completed_steps * dt * u
        assert E.first_inexact_step(dt, u, steps) is None and E.first_inexact_step(dt, v, steps) is None


def test_the_step_that_stopped_the_100_m_preset_is_named_before_a_run():
    """0.6 s with vmove = 3.0: six additions of 1.7999999999999998 m give 10.8 m where the product is
    10.799999999999999 m, and the run's exact check stops there.  The estimate now says so, with a step that works."""

    import tempfile

    total = 0.0
    for _ in range(6):
        total += 0.6 * 3.0
    assert (total, 6 * (0.6 * 3.0)) == (10.8, 10.799999999999999)
    assert E.first_inexact_step(0.6, 3.0, 12_000) == 6
    assert E.first_inexact_step(0.6, 0.0, 12_000) is None and E.first_inexact_step(0.5, 3.0, 12_000) is None
    # The same float pattern at the steps other spacings would take by CM1's rule alone.
    for dx, raw in ((67.0, 0.402), (100.0, 0.6), (150.0, 0.9), (200.0, 1.2), (33.0, 0.198)):
        assert C.time_step(dx, 500.0) == pytest.approx(raw)
        assert E.first_inexact_step(raw, 3.0, 20_000) or E.first_inexact_step(raw, 12.5, 20_000), dx
    preset = P.load("supercell-100m")
    scratch = Path(tempfile.mkdtemp(prefix="gm1-test-"))
    (scratch / "namelist.input").write_text(F.set_namelist_values(preset.namelist_text, {"dtl": 0.6}), encoding="ascii")
    deck = E.load_deck(scratch / "namelist.input", gm1_only=dict(preset.record.get("gm1_settings") or {}))
    checks = {check.id: check for check in E.price(deck, CARD_24GB, host=HOST).checks}
    assert checks["moving-step"].level == "stop"
    assert "a 0.6 s time step stops the run after 6 steps" in checks["moving-step"].text
    assert checks["moving-step"].text.endswith("0.5859375 s works here.")
    # The preset as it ships has no such check to make.
    shipped = E.load_deck(preset.namelist_path, gm1_only=dict(preset.record.get("gm1_settings") or {}))
    assert "moving-step" not in {check.id for check in E.price(shipped, CARD_24GB, host=HOST).checks}


# ------------------------------------------------------------------ the command line


def test_preset_fit_custom_grid_67_from_the_command_line():
    done = cli("preset", "fit", "supercell-1km", "--budget", "1h", "--custom", "grid=67", "--json",
               env=fixture_env("rtx5090-32gb"))
    assert done.returncode == 0, done.stderr + done.stdout
    record = json.loads(done.stdout)
    assert record["command"] == "gpuwm-cm1 preset fit supercell-1km --budget 1h --custom grid=67"
    assert record["facts"]["dx_m"] == 67.0 and record["facts"]["dt_s"] == 0.390625
    assert record["customized"][0]["sentence"].startswith("Grid spacing 67 m instead of 1 km")
    text = cli("preset", "fit", "supercell-1km", "--budget", "any", "--custom", "grid=67", "--custom", "levels=125",
               env=fixture_env("rtx5090-32gb"))
    assert text.returncode == 0, text.stderr + text.stdout
    assert "as long as it takes" in text.stdout and "- You changed: Grid spacing 67 m instead of 1 km" in text.stdout
    assert "- You changed: Height levels every 125 m instead of every 500 m (160 levels up to 20 km)." in text.stdout
    assert "67 m grid, " in text.stdout and "cells; time step 0.390625 s" in text.stdout
    assert "- Note: A 67 m grid is finer than any storm GM1 ships" in text.stdout
    options = cli("preset", "options", "supercell-1km")
    assert options.returncode == 0 and "any spacing up to 15000 m (grid=67)" in options.stdout
    assert "any height from 78 to 2666 m (256 to 8 levels" in options.stdout
    assert "rounded up to whole cells" in options.stdout


def test_preset_new_writes_a_67_m_run_folder_that_imports(tmp_path):
    root = tmp_path / "runs"
    made = cli("preset", "new", "supercell-1km", "--budget", "10m", "--root", str(root), "--name", "fine",
               "--custom", "grid=67", "--json", env=fixture_env("rtx5090-32gb"))
    assert made.returncode == 0, made.stderr + made.stdout
    record = json.loads(made.stdout)
    assert record["ok"] and record["import"]["exit"] == 0
    namelist = (root / "fine" / "namelist.input").read_text()
    assert C._effective(P.Preset("fine", root / "fine", {}, namelist))["dx"] == 67.0
    fit = json.loads((root / "fine" / "fit.json").read_text())
    assert fit["facts"]["dx_m"] == 67.0 and fit["facts"]["dt_s"] == 0.390625 and fit["pinned"] == ["grid"]
    assert fit["customized"][0]["sentence"].startswith("Grid spacing 67 m instead of 1 km")
    # Advanced mode prices the same folder (gpuwm-cm1 estimate): the same grid, step, memory and disk as the fit said.
    priced = cli("estimate", str(root / "fine"), "--json", env=fixture_env("rtx5090-32gb"))
    estimate = json.loads(priced.stdout)
    assert estimate["run"]["grid"] == fit["estimate"]["grid"]
    assert (estimate["run"]["dt_s"], estimate["run"]["steps"], estimate["run"]["cells"]) == (
        0.390625, fit["estimate"]["steps"], fit["estimate"]["cells"])
    assert int(estimate["needs"]["gpu_memory_bytes"]["value"]) == fit["estimate"]["gpu_memory_bytes"]
    assert int(estimate["needs"]["disk_bytes"]["value"]) == fit["estimate"]["disk_bytes"]
    assert "stop" not in [check["level"] for check in estimate["checks"]]


def test_a_grid_no_restart_file_can_hold_is_refused_with_the_finest_that_can():
    preset = P.load("supercell-1km")
    with pytest.raises(C.CustomError) as error:
        C.apply(preset, {"grid": 10.0})
    assert str(error.value) == (
        "A 10 m grid over the smallest area a storm starts in (20 km) is 160 million cells with these levels, and one "
        "restart file holds about 60 million; the finest grid here is about 17 m.")
    assert error.value.suggest == {"grid": 17.0}
    # 17 m is a grid the guided check takes (whether a card holds it is the fit's to say).
    assert P.examine(C.apply(preset, {"grid": 17.0}))["offered"]


def test_the_options_say_the_grid_and_the_levels_take_any_value():
    record = C.options(P.load("supercell-1km"))
    by_id = {knob["id"]: knob for knob in record["knobs"]}
    grid = by_id["grid"]
    assert grid["free"] and [choice["value"] for choice in grid["choices"]] == [100.0, 250.0, 500.0, 1000.0, 2000.0]
    assert grid["slider_m"] == [20.0, 4000.0] and grid["max_m"] == 15000.0
    levels = by_id["levels"]
    assert levels["free"] and levels["levels_range"] == [8, 256]
    assert (levels["min_m"], levels["max_m"]) == (78.0, 2666.0)
    area = by_id["area"]
    # The slider goes down to the smallest area a storm starts in; the preset's own floor is kept beside it.
    assert area["floor_km"] == 20.0 and area["comfortable_km"] == 64.0 and area["min_km"] == 20.0
    assert json.loads(json.dumps(P.list_record(), allow_nan=False, default=str))["budgets"][-1]["id"] == "any"


# ------------------------------------------------------------------ typed, odd and edge-of-range choices


CARD_16GB = card("NVIDIA GeForce RTX 4080", 16376, 0.3e9)


def test_the_presets_own_step_is_kept_as_it_is_at_its_own_spacing():
    """6.0 * 1000.0001 / 1000.0001 is 5.999999999999999 in binary64: a step worked out that way at the spacing in
    use failed the run's clock check for about one typed decimal spacing in six."""

    assert 6.0 * 1000.0001 / 1000.0001 != 6.0
    base = F._Base(imported=None, effective={}, nx=120, ny=120, dx_m=1000.0001, dt_s=6.0, duration_s=7200.0,
                   output_interval_s=900.0, periodic_x=False, periodic_y=False, dz_min_m=500.0)
    assert F.rung_dt_s(base, 1000.0001) == 6.0
    for dx in (82.9, 62.7, 1742.4, 250.000025, 67.0):
        dt = C.snapped_time_step(dx, 500.0, [7200.0, 300.0], moving=True)
        assert F.rung_dt_s(F._Base(None, {}, 120, 120, dx, dt, 7200.0, 300.0, False, False, 500.0), dx) == dt


@pytest.mark.parametrize("dx", [82.9, 62.7, 1742.4, 1000.0001])
def test_a_typed_decimal_spacing_is_fitted_like_any_other(dx):
    _, record = fitted("supercell-1km", "any", CARD_16GB, grid=dx)
    assert record["status"] == "fits", record["reason"]
    assert record["facts"]["dx_m"] == C.model_length_m(dx) and math.isclose(record["facts"]["dx_m"], dx, rel_tol=1e-7)
    if dx == 1000.0001:
        # Not "1 km instead of 1 km": the spacing is said with the digits that tell it from the preset's.
        assert record["customized"][0]["sentence"] == "Grid spacing 1.0000001 km instead of 1 km."


def _held_exactly(value: float) -> bool:
    """Whether a 4-byte REAL holds ``value`` exactly: the test the run's own start makes of dx, dy and dz."""

    return struct.unpack("<f", struct.pack("<f", value))[0] == value


@pytest.mark.parametrize("custom", [
    {"grid": 82.9}, {"grid": 62.7, "levels": 130.3}, {"grid": 66.67}, {"grid": 1742.4, "levels": 333.3},
    {"levels": 78.2}, {"grid": 250.000025},
])
def test_a_typed_decimal_length_is_one_the_model_holds(custom):
    """Measured on an RTX 4090: an 82.9 m run the fit offered stopped at its start, "dx must be positive finite and
    exactly representable in default REAL".  CM1 reads dx, dy and dz into a 4-byte REAL, so the spacing a run is
    given is the one that word holds for the number typed."""

    changed = C.apply(P.load("supercell-1km"), custom)
    effective = C._effective(changed)
    for key in ("dx", "dy", "dz"):
        assert _held_exactly(float(effective[key])), (key, effective[key])
    if "grid" in custom:
        assert float(effective["dx"]) == float(effective["dy"]) == C.model_length_m(custom["grid"])
        assert math.isclose(float(effective["dx"]), custom["grid"], rel_tol=1e-7)
    if "levels" in custom:
        assert math.isclose(float(effective["dz"]), custom["levels"], rel_tol=1e-7)
    # The step still lands on the run's times, and the words say the number that was typed.
    assert C.lands(float(effective["timax"]), float(effective["dtl"]))
    assert C.lands(float(effective["tapfrq"]), float(effective["dtl"]))
    said = " ".join(entry["sentence"] for entry in changed.record["customized"])
    for knob, value in custom.items():
        assert f"{value:g} m" in said or f"{value / 1000.0:g} km" in said, said


def test_lengths_a_real_holds_are_left_alone_and_others_go_to_the_nearest():
    for value in (67.0, 125.0, 62.5, 78.125, 250.0, 1000.0, 333.0):
        assert C.model_length_m(value) == value
    assert C.model_length_m(82.9) == 82.9000015258789
    assert C.model_length_m(130.3) == 130.3000030517578
    # What cannot be a spacing is left for the refusal that names it.
    for value in (0.5, 1e-300, 0.0, -5.0, float("inf")):
        assert C.model_length_m(value) == value
    assert math.isnan(C.model_length_m(float("nan")))


def test_a_spacing_under_a_metre_is_a_sentence_not_an_overflow():
    preset = P.load("supercell-1km")
    for dx in (1e-300, 1e-320, 0.5):
        with pytest.raises(C.CustomError) as error:
            C.apply(preset, {"grid": dx})
        assert "far more cells than one restart file holds" in str(error.value)
        assert str(error.value).endswith("the finest grid here is about 17 m.")
        assert error.value.suggest == {"grid": 17.0}


def test_a_coarse_spacing_says_the_area_whole_cells_come_to():
    preset = P.load("supercell-1km")
    # 120 km at 3350 m is 35.8 cells: 36 of them, 120.6 km (within 1 %, so nothing more is said).
    near = C.apply(preset, {"grid": 3350.0})
    assert C._effective(near)["nx"] == 36 and "Whole cells" not in near.record["customized"][0]["sentence"]
    # At 7 km it is 18 cells, 126 km: the sentence says so, where the area once moved to 112 km without a word.
    far = C.apply(preset, {"grid": 7000.0})
    assert C._effective(far)["nx"] == 18
    assert "Whole cells make the area 126 km wide instead of 120 km." in far.record["customized"][0]["sentence"]
    # Never smaller than the preset's own area: a storm set up to just stay inside it still does.
    for dx in (590.0, 670.0, 680.0, 770.0, 900.0, 970.0):
        cloud = C.apply(P.load("bubble-cloud-250m"), {"grid": dx})
        effective = C._effective(cloud)
        assert int(effective["nx"]) * dx >= 40_000.0 - 1e-6
        examined = P.examine(cloud)
        assert examined["offered"], (dx, examined["reasons"])


def test_a_whole_cell_area_of_the_users_own_is_kept_cell_for_cell():
    preset = P.load("supercell-1km")
    kept = C.apply(preset, {"grid": 33.0, "area": 36.696})          # 1112 cells of 33 m, to the metre
    assert C._effective(kept)["nx"] == 1112
    stepped = C.apply(preset, {"grid": 250.0, "area": 97.3})        # not whole cells: the slider's steps of 8
    assert (C._effective(stepped)["nx"] - 480) % 8 == 0


def test_an_area_no_saved_picture_holds_is_refused_with_one_that_is_saved():
    preset = P.load("supercell-1km")
    with pytest.raises(C.CustomError) as error:
        C.apply(preset, {"grid": 67.0, "levels": 100.0, "area": 240.0})
    text = str(error.value)
    assert "more than one saved picture of the storm holds (about 150 million)" in text
    area = error.value.suggest["area"]
    assert text.endswith(f"An area up to {area:.4g} km wide can be saved.")
    saved = C.apply(preset, {"grid": 67.0, "levels": 100.0, "area": area})
    effective = C._effective(saved)
    assert int(effective["nx"]) * int(effective["ny"]) * int(effective["nz"]) <= C.START_CELLS_MAX
    assert P.examine(saved)["offered"], P.examine(saved)["reasons"]


# ------------------------------------------------------------------ what the fit offers, the run's start takes


def test_a_famous_day_is_offered_only_at_an_area_its_own_start_takes(tmp_path):
    """The famous days start from a push of rising air (iinit = 12).  The estimate's storm-track check follows a
    warm bubble only, so the fit once offered them small areas that the import then refused at Start."""

    preset = P.load("oun-2013-05-20")
    if not P.examine(preset)["offered"]:
        pytest.skip("oun-2013-05-20 is not offered on this build")
    # Its own spacing held, ten minutes: offered, and the folder is made (the import takes what the fit chose).
    held = C.apply(preset, {"grid": 250.0})
    made = P.new_run(held, "10m", tmp_path / "runs", "held", gpu=CARD_16GB, host=HOST)
    assert made["ok"] and made["import"]["exit"] == 0, made["message"]
    # 67 m on a 16 GB card: the largest area that fits is one the storm leaves.  Said as that, with a spacing
    # that keeps it inside, and that spacing is taken from the fit to the import.
    _, record = fitted("oun-2013-05-20", "any", CARD_16GB, grid=67.0)
    assert record["status"] == "not-offered" and record["reason_kind"] == "limits", record["reason"]
    reason = record["reason"]
    assert reason.startswith("At 67 m the largest area that fits your GPU's free memory is ")
    assert "and the storm leaves it before the end, which the run's own start refuses" in reason
    assert "GPU memory even at its smallest" not in reason
    coarser = record["needs"]["suggest"]["grid"]
    assert 67.0 < coarser < 120.0 and f"A {coarser:g} m grid over that area fits." in reason
    again = C.apply(preset, {"grid": coarser})
    taken = P.new_run(again, "any", tmp_path / "runs", "coarser", gpu=CARD_16GB, host=HOST)
    assert taken["ok"] and taken["import"]["exit"] == 0, taken["message"]
    # The area the storm needs is the storm's, to within a few cells, not where the fit's search happened to stop:
    # the same from another spacing, and so is the grid that covers it.
    needs = []
    for dx in (67.0, 56.0):
        _, other = fitted("oun-2013-05-20", "any", CARD_16GB, grid=dx)
        needs.append(float(other["reason"].split("it needs about ")[1].split(" km")[0]))
        assert other["needs"]["suggest"]["grid"] == coarser, (dx, other["reason"])
    assert abs(needs[0] - needs[1]) < 1.0, needs
    # Ten minutes: the areas the storm leaves are too slow as well, so the search never tried the storm's own
    # smallest area.  The reason is what stops that area (the same 47 km), not the far larger one it started at.
    _, quick = fitted("oun-2013-05-20", "10m", CARD_16GB, grid=92.0)
    assert quick["status"] == "not-offered" and quick["reason_kind"] == "time", quick["reason"]
    said = float(quick["reason"].split("the smallest area the storm stays inside (")[1].split(" km")[0])
    assert abs(said - needs[0]) < 1.0, quick["reason"]
    _, tight = fitted("oun-2013-05-20", "10m", CARD_16GB, grid=56.0)
    assert tight["reason_kind"] == "gpu" and tight["needs"]["suggest"]["grid"] == coarser, tight["reason"]
    assert "even at the smallest area the storm stays inside (" in tight["reason"]
    assert f"The finest grid that fits that area here is {coarser:g} m." in tight["reason"]


def test_a_storm_that_leaves_every_area_that_fits_is_not_said_as_too_little_memory():
    """A warm bubble on a sounding file (the run's start refuses a storm that leaves): every area the card holds is
    too small for the storm.  The reason is the storm's room, whatever the time picked, never "too little GPU
    memory" for an area that needs a third of the card."""

    preset = P.load("bubble-cloud-250m")
    if not P.examine(preset)["offered"]:
        pytest.skip("bubble-cloud-250m is not offered on this build")
    _, record = fitted("bubble-cloud-250m", "any", grid=67.0, levels=78.0)
    assert record["status"] == "not-offered" and record["reason_kind"] == "limits", record["reason"]
    reason = record["reason"]
    assert reason.startswith("At 67 m the largest area that fits your GPU's free memory is ")
    assert "the storm leaves it before the end" in reason and "it needs about " in reason
    assert "too little" not in reason and "pick a longer time" not in reason
    coarser = record["needs"]["suggest"]["grid"]
    _, fits = fitted("bubble-cloud-250m", "any", grid=coarser, levels=78.0)
    assert fits["status"] == "fits", fits["reason"]
    # Ten minutes: no "pick a longer time", which no time would mend; the area named is one the storm stays in.
    _, short = fitted("bubble-cloud-250m", "10m", grid=67.0, levels=78.0)
    assert short["status"] == "not-offered" and "pick a longer time" not in short["reason"]
    assert "the storm" in short["reason"]


def test_the_finest_grid_a_refusal_names_is_one_the_fit_takes():
    """supercell-1km with 100 m levels on a 16 GB card: the sum alone named 51 or 52 m, which were refused; the
    spacing is now priced first, so the same number comes back wherever the search starts and it fits."""

    named = set()
    for dx in (24.0, 30.0, 34.0, 50.0):
        _, record = fitted("supercell-1km", "any", CARD_16GB, grid=dx, levels=100.0)
        assert record["status"] == "not-offered" and record["reason_kind"] == "gpu", (dx, record["reason"])
        assert "widest area" not in record["reason"]
        named.add(record["needs"]["suggest"]["grid"])
    assert len(named) == 1, named
    finest = named.pop()
    assert f"here is {finest:g} m." in record["reason"]
    _, fits = fitted("supercell-1km", "any", CARD_16GB, grid=finest, levels=100.0)
    assert fits["status"] == "fits", fits["reason"]
    # 200 levels: the time comes from runs with at most 80, and the note says what the one tall run did.
    tall = [entry for entry in fits["warning_records"] if entry["kind"] == "levels"]
    assert len(tall) == 1 and "With 200 levels expect it to take longer" in tall[0]["sentence"]
    _, plain = fitted("supercell-1km", "any", CARD_16GB, grid=100.0)
    assert not [entry for entry in plain["warning_records"] if entry["kind"] == "levels"]


def test_the_finest_grid_is_priced_where_an_off_centre_bubble_still_fits():
    """The 100 m supercell starts its bubble 11 km off centre, so its smallest area is 42 km, wider than the 40 km
    the preset names: priced at 40 km the bubble does not fit, and the spacing came back unchecked ("about")."""

    spec = C.apply(P.load("supercell-100m"), {"grid": 62.7}).spec()
    assert spec.hard_min_width_km == 42.0 and spec.smallest_width_m(62.7) == 42_000.0
    _, record = fitted("supercell-100m", "any", CARD_16GB, grid=62.7)
    assert record["status"] == "not-offered" and record["reason_kind"] == "gpu", record["reason"]
    finest = record["needs"]["suggest"]["grid"]
    assert f"The finest grid that fits its smallest area (42 km) here is {finest:g} m." in record["reason"]
    _, fits = fitted("supercell-100m", "any", CARD_16GB, grid=finest)
    assert fits["status"] == "fits", fits["reason"]


def test_the_finest_grid_for_an_area_of_the_users_own_is_priced_over_that_area():
    _, record = fitted("supercell-1km", "any", CARD_16GB, grid=50.0, area=50.0)
    assert record["status"] == "not-offered" and record["reason_kind"] == "gpu", record["reason"]
    assert "at the area you chose (50 km)" in record["reason"] and "its smallest area" not in record["reason"]
    finest = record["needs"]["suggest"]["grid"]
    assert f"The finest grid that fits that area here is {finest:g} m." in record["reason"]
    _, fits = fitted("supercell-1km", "any", CARD_16GB, grid=finest, area=50.0)
    assert fits["status"] == "fits", fits["reason"]
    # A picture every minute fills the disk as well: the grid named is still the one the memory takes, priced.
    _, busy = fitted("supercell-100m", "any", CARD_16GB, grid=50.0, pictures=1.0)
    if busy["status"] == "not-offered" and busy["reason_kind"] == "gpu":
        assert "here is about" not in busy["reason"], busy["reason"]


def test_a_need_and_what_is_free_never_read_the_same():
    assert F._gb_versus(8.31e9, 8.27e9) == ("8.31 GB", "8.27 GB")
    assert F._gb_versus(30.9e9, 25.4e9) == ("30.9 GB", "25.4 GB")


def test_the_changes_name_the_limit_that_binds_and_the_area_the_fit_started_from():
    """67 m in an hour on a 16 GB card uses a third of the card: the time is what made the area small, and the
    82 km it is compared with is the most one restart file holds, which the sentence now says."""

    _, record = fitted("supercell-1km", "1h", CARD_16GB, grid=67.0)
    assert record["status"] == "fits", record["reason"]
    area = next(line for line in record["sentences"] if line.startswith("Made the area"))
    assert "(the most one restart file holds at 67 m)" in area
    assert area.endswith("so it finishes in about 1 hour.") and "free memory" not in area
    assert record["estimate"]["gpu_memory_bytes"] < 0.5 * CARD_16GB.free_bytes
    _, night = fitted("supercell-1km", "overnight", CARD_16GB, grid=40.0)
    if night["status"] == "fits":
        assert not any("finishes in overnight" in line for line in night["sentences"])
    assert F._finish_words("overnight", 36000.0) == "finishes overnight (within 10 hours)"
    assert F._finish_words("1h", 3600.0) == "finishes in about 1 hour"
    # A time too short says the area its number is for, so a longer time's larger area does not contradict it.
    _, held = fitted("supercell-1km", "10m", CARD_16GB, grid=67.0)
    assert held["reason_kind"] == "time" and "even at its smallest area (" in held["reason"]


def test_with_no_gpu_found_the_summary_says_memory_was_not_checked():
    _, record = fitted("supercell-1km", "any", None, grid=250.0)
    assert record["status"] == "fits", record["reason"]
    assert record["summary"].endswith("No GPU was found, so whether that memory is free was not checked.")
    _, seen = fitted("supercell-1km", "any", CARD_16GB, grid=250.0)
    assert "No GPU was found" not in seen["summary"]


def test_an_area_said_with_a_vowel_takes_an_in_the_fit_in_advanced_and_in_results():
    """The gust front's own area is 80 km: "An 80 × 80 km area at 500 m", not "A 80".  The fit's first line, Advanced
    mode's summary and Results' "This run" line take the article as the area is said aloud, the rule the spacing
    already followed ("An 82.9 m grid")."""

    from cm1core.gui.guided import panels
    from cm1core.gui.setup import plain_summary

    _, gust = fitted("gust-front", "any")
    assert gust["status"] == "fits", gust["reason"]
    assert gust["summary"].startswith("An 80 × 80 km area at 500 m, "), gust["summary"]
    for area, said in ((88.0, "An 88 × 88 km area at 1 km, "), (64.0, "A 64 × 64 km area at 1 km, ")):
        _, mine = fitted("supercell-1km", "any", area=area)
        assert mine["summary"].startswith(said), mine["summary"]
    _, own = fitted("supercell-1km", "any")
    assert own["summary"].startswith("A 120 × 120 km area at 1 km, "), own["summary"]
    for words, said in (("80 × 80", "An"), ("88 × 80", "An"), ("11 × 11", "An"), ("18 × 18", "An"), ("8 × 8", "An"),
                        ("120 × 120", "A"), ("100 × 100", "A"), ("110 × 110", "A"), ("180 × 180", "A")):
        assert F.article(words) == said, words
    summary = plain_summary({"nx": 160, "ny": 160, "dx": 500.0, "dy": 500.0, "timax": 7200.0}, None)["text"]
    assert summary.startswith("An 80 × 80 km area at 500 m, "), summary
    summary = plain_summary({"nx": 120, "ny": 110, "dx": 1000.0, "dy": 1000.0, "timax": 7200.0}, None)["text"]
    assert summary.startswith("A 120 × 110 km area at 1 km, "), summary
    copy = json.loads((ROOT / "src" / "cm1core" / "gui" / "copy" / "looks.json").read_text(encoding="utf-8"))
    facts = {"dx_m": 500.0, "nx": 160, "ny": 160, "duration_s": 7200.0, "interval_s": 300.0}
    assert panels(facts, copy)["this_run"].startswith("This run: 500 m grid spacing, an 80 × 80 km area, ")
    assert panels({**facts, "nx": 240, "ny": 240}, copy)["this_run"].startswith(
        "This run: 500 m grid spacing, a 120 × 120 km area, ")


# ------------------------------------------------------------------ following the storm, names and labels


@pytest.mark.parametrize("minutes", [30.0, 45.0, 60.0, 90.0, 120.0])
def test_an_area_of_the_users_own_follows_the_storm_at_every_storm_time(minutes):
    """The preset writes no restart of its own (rstfrq < 0) and the run keeps one at its end.  Read as an interval,
    45 or 90 minutes of a 60 minute one was refused before the storm-track check ran, so no track came back and
    the view stayed on the preset's frame: the storm drifted to the edge of a 67 m run."""

    preset = C.apply(P.load("supercell-1km"), {"grid": 67.0, "area": 33.77, "time": minutes})
    assert P.storm_track_for(preset) is not None
    follow = [entry for entry in preset.record["customized"] if entry["knob"] == "follow"]
    assert len(follow) == 1, [entry["knob"] for entry in preset.record["customized"]]
    assert follow[0]["sentence"].startswith("The view moves with the storm as the storm-track check predicts it (")
    assert "inside" not in follow[0]["sentence"], follow[0]["sentence"]
    effective = C._effective(preset)
    assert (float(effective["umove"]), float(effective["vmove"])) != (12.5, 3.0)
    # The same at the preset's own spacing.
    coarse = C.apply(P.load("supercell-1km"), {"area": 60.0, "time": minutes})
    assert P.storm_track_for(coarse) is not None


def test_the_estimate_says_a_storm_short_of_room_from_the_start():
    preset = C.apply(P.load("supercell-1km"), {"grid": 100.0, "area": 24.0})
    deck = E.load_deck(preset.namelist_path, gm1_only=dict(preset.record.get("gm1_settings") or {}))
    checks = {check.id: check for check in E.price(deck, CARD_24GB, host=HOST).checks}
    text = checks["storm-track"].text
    assert text.startswith("The storm is too close to the edge of the area from the start (it comes within ")
    assert "at about 0 s" not in text and checks["storm-track"].level == "warn"
    # The fit's own note: the area is small, whatever the view does.
    _, record = fitted("supercell-1km", "any", grid=100.0, area=24.0)
    edge = next(entry for entry in record["warning_records"] if entry["kind"] == "edge")
    assert edge["sentence"].startswith("This 24 km area is smaller than the 64 km this storm was set up for: ")


@pytest.mark.parametrize("minutes", [45.0, 120.0])
def test_a_small_area_keeps_its_edge_note_whatever_the_track_check_predicts(minutes):
    """On an RTX 4090 a 67 m storm in a 33.8 km area left it within 45 minutes, although Customize had moved the
    view as the storm-track check advised and said the storm stays inside.  Below the area the preset was set up
    for the note is always there, nothing says the storm stays inside, and the one-tap area is the preset's own."""

    preset, record = fitted("supercell-1km", "any", grid=67.0, area=33.77, time=minutes)
    assert record["status"] == "fits", record["reason"]
    edge = [entry for entry in record["warning_records"] if entry["kind"] == "edge"]
    assert len(edge) == 1, record["warning_records"]
    assert "the storm may leave an area this small; pictures near the edges then show edge effects." in edge[0]["lead"]
    assert edge[0]["set_up_for_km"] == 64.0
    said = " ".join([entry["sentence"] for entry in record["warning_records"]]
                    + [entry["sentence"] for entry in record["customized"]] + list(record.get("sentences") or []))
    assert "stays inside" not in said and "keeps it inside" not in said, said
    assert any(entry["knob"] == "follow" for entry in record["customized"])


def test_a_small_area_the_fit_chose_says_so_too():
    """An area the fit shrank to fit the card, below the area the preset was set up for, gets the same note."""

    _, record = fitted("supercell-1km", "1h", CARD_16GB, grid=67.0)
    assert record["status"] == "fits", record["reason"]
    side = min(record["facts"]["domain_km"])
    edge = [entry for entry in record["warning_records"] if entry["kind"] == "edge"]
    assert (side < 64.0) == bool(edge), (side, record["warning_records"])


def test_a_run_is_named_and_labelled_with_the_spacing_it_has(tmp_path):
    grid = [{"knob": "grid", "keys": {"dx": [1000.0, 150.0]}}]
    assert P.run_id_for("supercell-1km", grid) == "supercell-150m"
    assert P.run_id_for("bubble-cloud-250m", [{"knob": "grid", "keys": {"dx": [250.0, 1234.0]}}]) == "bubble-cloud-1.234km"
    assert P.run_id_for("oun-2013-05-20", [{"knob": "grid", "keys": {"dx": [250.0, 67.0]}}]) == "oun-2013-05-20-67m"
    assert P.run_id_for("supercell-1km", [{"knob": "pictures", "keys": {}}]) == "supercell-1km"
    assert P.run_id_for("supercell-1km", None) == "supercell-1km"
    name = "Supercell, quick look, 1 km"
    assert P.case_words(name, grid) == "Supercell, 150 m grid (customized from Supercell, quick look, 1 km)"
    levels = [*grid, {"knob": "levels", "keys": {"dz": [500.0, 125.0], "nz": [40, 160]}}]
    assert P.case_words(name, levels) == (
        "Supercell, 150 m grid, levels every 125 m (customized from Supercell, quick look, 1 km)")
    assert P.case_words(name, [{"knob": "pictures", "keys": {}}]) == "Supercell, quick look, 1 km (customized)"
    assert P.case_words(name, []) == name
    preset = C.apply(P.load("supercell-1km"), {"grid": 150.0})
    root = tmp_path / "runs"
    assert P.default_name(preset, root).startswith("supercell-150m-custom-")
    made = P.new_run(preset, "10m", root, P.default_name(preset, root), gpu=CARD_24GB, host=HOST)
    assert made["ok"], made["message"]
    run = json.loads((Path(made["run"]) / "run.json").read_text())
    assert run["preset"] == "supercell-1km"
    assert run["case"] == "Supercell, 150 m grid (customized from Supercell, quick look, 1 km)"


def test_a_fine_grid_whose_largest_saved_area_the_storm_leaves_says_that():
    preset = P.load("bubble-cloud-250m")
    if not P.examine(preset)["offered"]:
        pytest.skip("bubble-cloud-250m is not offered on this build")
    _, record = fitted("bubble-cloud-250m", "any", grid=25.0)
    if record["status"] == "fits" or record["reason_kind"] != "drift":
        pytest.skip("the storm stays inside the largest area saved at 25 m on this build")
    assert record["reason"].startswith("At a 25 m grid the largest area one restart file holds (about ")
    assert "Make the area larger" not in record["reason"] and "Use a coarser grid" in record["reason"]


def test_the_command_line_error_carries_the_value_that_is_taken():
    refused = cli("preset", "fit", "supercell-1km", "--budget", "any", "--custom", "grid=0.5", "--json",
                  env=fixture_env("rtx5090-32gb"))
    assert refused.returncode == 2
    detail = json.loads(refused.stderr)
    assert detail["suggest"] == {"grid": 17.0} and "about 17 m" in detail["reason"]
    unknown = cli("preset", "fit", "no-such-storm", "--budget", "any", "--json", env=fixture_env("rtx5090-32gb"))
    assert unknown.returncode == 2 and "suggest" not in json.loads(unknown.stderr)


# ------------------------------------------------------------------ the panel's script


def test_the_panel_does_not_redraw_under_a_control_in_use():
    """customize.js: a fit's reply once drew the panel again under a half-typed number or a held slider, which
    entered the half number, dropped the thumb and threw.  The drawing now waits for the control, the handlers of a
    control taken off the page do nothing, and one drawing never starts inside another.  (A browser shows the
    behaviour itself; this holds the guards in the script so they are not taken out by accident.)"""

    script = (ROOT / "src" / "cm1core" / "gui" / "static" / "guided" / "customize.js").read_text(encoding="utf-8")
    assert "if (painting) { paintAgain = true; return; }" in script
    assert script.count("if (painting || !") >= 5            # both boxes, both sliders, the lists
    assert "if (busy()) {" in script and "pendingPaint = true;" in script
    assert 'body.addEventListener("pointerdown"' in script and 'window.addEventListener("pointerup", letGo)' in script
    assert "el.value !== el.dataset.committed" in script
    # Whole cells as customize.cells_across counts them, and the model's own fewest levels.
    assert "Math.ceil(exact)" in script and "[8, 256]" in script and "[7, 256]" not in script
    budget = (ROOT / "src" / "cm1core" / "gui" / "static" / "guided" / "budget.js").read_text(encoding="utf-8")
    assert "function suggestButtons(suggest)" in budget and "err.body.suggest" in budget
    # The one-tap larger area: the first step as wide as the area the storm was set up for is tried with the spread.
    assert "clearOfEdge(mine, li, w.set_up_for_km)" in budget and "larger.find((a) => a >= Number(setUpForKm)" in budget
    # A typed decimal spacing is shown with the figures typed, not the 4-byte word's sixteen.
    assert "real(f.dx_m)" in budget and "real(f.dz_m)" in budget
