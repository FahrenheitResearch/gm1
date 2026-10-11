"""``gpuwm-cm1 cm1out``: CM1 r21.1's netCDF history and statistics files from GM1 frames.

The reference files in ``tests/data/cm1-ref-24x24`` are CM1 r21.1's own
output of its distributed supercell namelist at 24 x 24 x 40
(``README.md`` there says how they were made).  A frame carrying CM1's own
words must convert to CM1's own file, and the statistics of CM1's own state
must be CM1's own statistics, word for word.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from cm1core.cm1out import convert  # noqa: E402
from cm1core.cm1out import frames as F  # noqa: E402
from cm1core.cm1out import names as N  # noqa: E402
from cm1core.cm1out import settings as S  # noqa: E402
from cm1core.cm1out import stats as T  # noqa: E402
from cm1core.cm1out.convert import Cm1outError  # noqa: E402
from cm1core.cm1out.fields import ground_window  # noqa: E402

from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import (  # noqa: E402
    REF, REF_GRID, end_event, frame_from_cm1, publish, species_record, write_frame,
)

ROOT = Path(__file__).resolve().parents[1]
NOT_PRODUCED_STOCK = {"srs", "sgs", "rain2", "srs2", "sgs2", "kmh", "kmv", "khh", "khv"}


@pytest.fixture()
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


def _words(array) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.float32)).view(np.uint32)


def _rundir(tmp_path: Path, namelist: Path, frames: list[tuple[dict, dict]], segment: str = "001") -> Path:
    run = tmp_path / "run"
    root = run / "segments" / segment / "frames"
    root.mkdir(parents=True)
    shutil.copy(namelist, run / "namelist.input")
    for index, (carriers, frame) in enumerate(frames):
        publish(root, index, carriers, **frame)
    return run


def _header(name: str) -> dict:
    return json.loads((REF / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- history files

def test_a_frame_of_cm1_words_converts_to_cm1s_own_file(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    summary = convert(run, stats=False)
    assert summary.files == ["cm1out_000001.nc"]
    ours = netCDF4.Dataset(run / "cm1out" / "cm1out_000001.nc")
    cm1 = netCDF4.Dataset(REF / "cm1out_000001.nc")
    try:
        assert ours.data_model == cm1.data_model == "NETCDF4_CLASSIC"
        assert [(k, len(d), d.isunlimited()) for k, d in ours.dimensions.items()] == \
               [(k, len(d), d.isunlimited()) for k, d in cm1.dimensions.items()]
        # CM1's 71 global attributes, in CM1's order, with CM1's values and types.
        names = cm1.ncattrs()
        assert len(names) == 71
        assert ours.ncattrs()[:71] == names
        for name in names:
            mine, theirs = ours.getncattr(name), cm1.getncattr(name)
            if name == "CM1 version":
                assert mine.startswith(theirs)
                continue
            assert np.asarray(mine).dtype == np.asarray(theirs).dtype, name
            assert np.array_equal(np.asarray(mine), np.asarray(theirs)), name
        assert all(n.startswith("gm1_") for n in ours.ncattrs()[71:])
        produced = [v for v in cm1.variables if v in ours.variables]
        missing = [v for v in cm1.variables if v not in ours.variables]
        assert set(missing) == NOT_PRODUCED_STOCK
        assert set(ours.getncattr("gm1_not_produced").split()) == NOT_PRODUCED_STOCK
        reasons = json.loads(ours.getncattr("gm1_not_produced_reasons"))
        assert set(reasons) == NOT_PRODUCED_STOCK and all(reasons.values())
        assert [v for v in ours.variables] == produced, "CM1's variable order"
        for name in produced:
            a, b = cm1[name], ours[name]
            assert b.dimensions == a.dimensions, name
            assert {att: b.getncattr(att) for att in b.ncattrs()} == {att: a.getncattr(att) for att in a.ncattrs()}, name
            assert np.array_equal(_words(b[:]), _words(a[:])), f"{name}: the words differ"
    finally:
        ours.close()
        cm1.close()


def test_the_file_matches_cm1out_000002s_recorded_layout(tmp_path, parser_tool):
    """The plan's anchor file: dims, coordinates in km (xh -11.5..11.5), variable attributes."""

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    convert(run, stats=False)
    header = _header("cm1out_000002.header.json")
    ours = netCDF4.Dataset(run / "cm1out" / "cm1out_000001.nc")
    try:
        assert [[k, len(d), d.isunlimited()] for k, d in ours.dimensions.items()] == header["dimensions"]
        assert ours.ncattrs()[:71] == [name for name, _value in header["global_attributes"]]
        for row in header["variables"]:
            if row["name"] not in ours.variables:
                assert row["name"] in NOT_PRODUCED_STOCK
                continue
            var = ours[row["name"]]
            assert list(var.dimensions) == row["dimensions"]
            assert {att: var.getncattr(att) for att in var.ncattrs()} == row["attributes"]
            if "float32_words" in row and row["name"] not in ("time",):
                assert [f"{int(w):08x}" for w in _words(var[:])] == row["float32_words"], row["name"]
        assert float(ours["xh"][0]) == pytest.approx(-11.5, abs=1e-5)
        assert float(ours["xh"][-1]) == pytest.approx(11.5, abs=1e-5)
    finally:
        ours.close()


def test_morrison_number_concentrations_convert(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    carriers["nc3d"] = (np.arange(4 * 40 * 24 * 24, dtype=np.float32).reshape(4, 40, 24, 24) + 1.0)
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    convert(run, stats=False)
    ds = netCDF4.Dataset(run / "cm1out" / "cm1out_000001.nc")
    try:
        for index, name in enumerate(("nci", "ncs", "ncr", "ncg")):
            assert ds[name].units == "#/kg" and ds[name].long_name == name
            assert np.array_equal(ds[name][0], carriers["nc3d"][index])
    finally:
        ds.close()


def test_translated_swaths_are_the_canvas_window_under_the_domain(tmp_path, parser_tool):
    """At 120 s the stock domain has moved 1.5 cells east: CM1's grid sees canvas columns 2..25."""

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    frame = dict(frame, time_s=120.0, step=20, domainloc=(1500.0, 360.0))
    canvas = np.arange(25 * 27, dtype=np.float32).reshape(25, 27)
    for name in ("sws2", "svs2", "sps2", "sus2", "shs2"):
        carriers[name] = canvas
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    convert(run, stats=False)
    ds = netCDF4.Dataset(run / "cm1out" / "cm1out_000003.nc")
    try:
        assert np.array_equal(ds["sws2"][0], canvas[0:24, 2:26])
        assert "gm1_translated_swaths" in ds.ncattrs()
        assert json.loads(ds.getncattr("gm1_formed"))["sws2"] == "sampled"
    finally:
        ds.close()


def test_the_ground_window_follows_the_anchor_rule_both_ways():
    class _Frame:
        def __init__(self, loc, u, v):
            self.moving_frame = {"domainlocx": loc[0], "domainlocy": loc[1]}
            self.publication = {"tornado_diagnostics": {"swaths": {"moving_frame": {
                "dt_s": 6.0, "total_steps": 20, "u_move_m_s": u, "v_move_m_s": v}}}}

    geometry = S.Geometry(24, 24, 40, 1000.0, 1000.0, 500.0, None, None, None, None, 6.0)
    # Eastward: the canvas origin is the start; westward: the canvas starts final-offset cells west.
    assert ground_window(_Frame((1500.0, 360.0), 12.5, 3.0), geometry) == (2, 0)
    assert ground_window(_Frame((-1500.0, 0.0), -12.5, 0.0), geometry) == (0, 0)
    assert ground_window(_Frame((0.0, 0.0), -12.5, 0.0), geometry) == (2, 0)


def test_filetype_1_holds_every_time_in_one_file(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    frames = []
    for n, t in enumerate((0.0, 60.0, 120.0)):
        c = dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(n)))
        frames.append((c, dict(frame, time_s=t, step=10 * n, domainloc=(12.5 * t, 3.0 * t))))
    run = _rundir(tmp_path, REF / "namelist.input", frames)
    summary = convert(run, filetype=1, stats=False)
    assert summary.files == ["cm1out.nc"]
    ds = netCDF4.Dataset(run / "cm1out" / "cm1out.nc")
    try:
        assert list(ds["time"][:]) == [0.0, 60.0, 120.0]
        assert [float(ds["w"][i].max()) for i in range(3)] == [0.0, 1.0, 2.0]
        assert len(ds.getncattr("gm1_frames").split()) == 3
    finally:
        ds.close()


def test_filetype_1_gives_a_variable_a_later_frame_brings_cm1s_missing_value_before(tmp_path, parser_tool):
    """One cm1out.nc is fill-off: a variable first defined at a later time must not leave unwritten words."""

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    first = {name: array for name, array in carriers.items() if name != "uh"}
    frames = [(first, dict(frame, time_s=0.0, step=0)), (carriers, dict(frame, time_s=60.0, step=10))]
    run = _rundir(tmp_path, REF / "namelist.input", frames)
    convert(run, filetype=1, stats=False)
    with netCDF4.Dataset(run / "cm1out" / "cm1out.nc") as ds:
        ds.set_auto_mask(False)
        assert np.all(np.asarray(ds["uh"][0]) == np.float32(-999999.9))
        assert np.asarray(ds["uh"][1], np.float32).tobytes() == carriers["uh"].tobytes()


def test_a_resumed_segment_replaces_the_times_it_repeats(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")

    def at(t: float, value: float) -> tuple[dict, dict]:
        return (dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(value))),
                dict(frame, time_s=t, step=int(t / 6), domainloc=(12.5 * t, 3.0 * t)))

    run = _rundir(tmp_path, REF / "namelist.input", [at(0.0, 1.0), at(60.0, 1.0), at(120.0, 1.0)])
    convert(run)
    root2 = run / "segments" / "002" / "frames"
    root2.mkdir(parents=True)
    for index, (c, f) in enumerate([at(60.0, 2.0), at(120.0, 2.0), at(180.0, 2.0)]):
        publish(root2, index, c, **f)
    summary = convert(run)
    assert summary.frames_converted == 3
    for name, expected in (("cm1out_000001.nc", 1.0), ("cm1out_000002.nc", 2.0),
                           ("cm1out_000003.nc", 2.0), ("cm1out_000004.nc", 2.0)):
        with netCDF4.Dataset(run / "cm1out" / name) as ds:
            assert float(ds["w"][0].max()) == expected, name
    with netCDF4.Dataset(run / "cm1out" / "cm1out_stats.nc") as ds:
        assert list(ds["time"][:]) == [0.0, 60.0, 120.0, 180.0]
        assert list(ds["wmax"][:]) == [1.0, 2.0, 2.0, 2.0]
    assert convert(run).frames_converted == 0


def test_the_namelists_grads_choice_is_written_as_grads(tmp_path, parser_tool):
    """output_format = 1 writes CM1's GrADS files (tests/test_cm1out_grads.py holds their layout)."""

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    text = (REF / "namelist.input").read_text().replace("output_format    = 2", "output_format    = 1")
    namelist = tmp_path / "grads.namelist.input"
    namelist.write_text(text)
    run = _rundir(tmp_path, namelist, [(carriers, frame)])
    summary = convert(run, stats=False)
    assert summary.format == "grads"
    assert {"cm1out_000001_s.dat", "cm1out_s.ctl", "cm1out_metadata.dat"} <= set(summary.files)
    # --format nc still writes netCDF from the same frames, into the same folder.
    assert convert(run, fmt="nc", stats=False).files == ["cm1out_000001.nc"]
    assert (run / "cm1out" / "cm1out_000001_s.dat").is_file()


# ---------------------------------------------------------------- frames

def test_staging_directories_are_never_listed_or_opened(tmp_path, parser_tool, monkeypatch):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    root = run / "segments" / "001" / "frames"
    # A frame still being written, and a published frame whose table names
    # the staging path its arena was written at (an older publisher).
    write_frame(root / ".incomplete" / "frame-000002", {"w3d": np.zeros(3, np.float32)},
                time_s=120.0, step=20, grid=REF_GRID, dt_s=6.0)
    write_frame(root / "frame-000001", carriers, staged_path_in_table=True,
                **dict(frame, time_s=60.0, step=10, domainloc=(750.0, 180.0)))
    opened: list[Path] = []
    original = F.open_frame
    monkeypatch.setattr(F, "open_frame", lambda d: (opened.append(Path(d)), original(d))[1])
    summary = convert(run, stats=False)
    assert summary.files == ["cm1out_000001.nc", "cm1out_000002.nc"]
    assert [p.name for p in F.published_frames(root)] == ["frame-000000", "frame-000001"]
    assert opened and not any(".incomplete" in p.parts for p in opened)
    with pytest.raises(F.FrameError, match="staging"):
        F.open_frame(root / ".incomplete" / "frame-000002")


def test_follow_converts_each_frame_as_it_is_published_and_exits_with_the_run(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = tmp_path / "run"
    seg = run / "segments" / "001"
    root = seg / "frames"
    root.mkdir(parents=True)
    shutil.copy(REF / "namelist.input", run / "namelist.input")
    (seg / "events.jsonl").write_text(json.dumps({"schema": "gm1.run-events/v1", "kind": "admitted"}) + "\n")

    def model() -> None:
        for index, t in enumerate((0.0, 60.0, 120.0)):
            time.sleep(0.3)
            publish(root, index, carriers, **dict(frame, time_s=t, step=10 * index, domainloc=(12.5 * t, 3.0 * t)))
        time.sleep(0.2)
        end_event(seg / "events.jsonl", frames=3)

    thread = threading.Thread(target=model)
    started = time.monotonic()
    thread.start()
    summary = convert(run, follow=True, poll_s=0.05)
    thread.join()
    assert time.monotonic() - started < 30
    assert summary.files == ["cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc"]
    assert summary.ended and summary.end_reason == "completed"
    assert not list((run / "cm1out").glob("*.partial"))


def test_a_second_follower_waits_for_the_first_then_carries_on(tmp_path, parser_tool):
    """After a Stop and an immediate Continue the previous segment's cm1out can still be writing: the next
    follower waits on the run folder's lock (it reads no state and writes nothing), then carries on."""

    from cm1core.cm1out.follow_lock import follow_lock

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = tmp_path / "run"
    seg = run / "segments" / "001"
    root = seg / "frames"
    root.mkdir(parents=True)
    shutil.copy(REF / "namelist.input", run / "namelist.input")
    (seg / "events.jsonl").write_text(json.dumps({"schema": "gm1.run-events/v1", "kind": "admitted"}) + "\n")
    for index, t in enumerate((0.0, 60.0)):
        publish(root, index, carriers, **dict(frame, time_s=t, step=10 * index, domainloc=(12.5 * t, 3.0 * t)))
    end_event(seg / "events.jsonl", frames=2)
    said: list[str] = []
    result: dict = {}
    with follow_lock(run, "cm1out") as held:
        assert held
        # A second follower that is stopped while it waits leaves without touching anything.
        with pytest.raises(Cm1outError, match="still writing"):
            convert(run, follow=True, poll_s=0.05, should_stop=lambda: True)
        thread = threading.Thread(target=lambda: result.update(summary=convert(run, follow=True, poll_s=0.05,
                                                                               log=said.append)))
        thread.start()
        time.sleep(1.0)
        assert thread.is_alive()
        assert not (run / "cm1out" / ".cm1out-state.json").exists()
        assert any("waiting for it to finish" in line for line in said), said
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert result["summary"].files == ["cm1out_000001.nc", "cm1out_000002.nc"]
    assert (run / "cm1out" / ".cm1out-state.json").is_file()


# ---------------------------------------------------------------- statistics

def _stats_state():
    data = np.load(REF / "stats-run-t1500.npz")
    record = json.loads((REF / "stats-run-t1500.cm1-stats.json").read_text())
    return data, record


def _species():
    return S.species(type("Frame", (), {"publication": {"config": {"moisture_species": species_record()}},
                                        "directory": REF})())


def _geometry():
    zf = (np.arange(41, dtype=np.float32) * np.float32(500.0)).astype(np.float32)
    zh = (np.float32(0.5) * (zf[1:] + zf[:-1])).astype(np.float32)
    mh = (np.float32(500.0) / (zf[1:] - zf[:-1])).astype(np.float32)
    return S.Geometry(24, 24, 40, 1000.0, 1000.0, 500.0, zf, zh, mh, (np.float32(1.0) / mh).astype(np.float32), 6.0)


def _inputs(data, *, base_state: bool) -> T.StatInputs:
    species = _species()
    return T.StatInputs(
        time=float(data["time"]), u=data["u"], v=data["v"], w=data["w"], tha=data["thpert"], ppi=data["pipert"],
        prs=data["prs"], rho=data["rho"], q=[data[n] for n in species.names], tke=data["tke"], prate=data["prate"],
        th0=data["th0"] if base_state else None, pi0=data["pi0"] if base_state else None,
        prs0=data["prs0"] if base_state else None, qv0=data["qv0"] if base_state else None,
        th=data["th"], umove=float(data["umove"]), vmove=float(data["vmove"]),
    )


STATS_NOT_PRODUCED = {
    "kmhmax", "kmhmin", "kmvmax", "kmvmin", "khhmax", "khhmin", "khvmax", "khvmin", "kshmax", "ksvmax",
    "tcond", "tevac", "tauto", "taccr", "tevar", "train", "erain", "qsfc", "esfc", "erad",
    *(f"{p}{n}" for p in ("as", "bs") for n in ("qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg")),
}


def test_statistics_of_cm1s_state_are_cm1s_statistics_word_for_word(parser_tool):
    data, record = _stats_state()
    settings = S.from_namelist(REF / "stats-run.namelist.input")
    result = T.compute(_inputs(data, base_state=True), settings, _geometry(), _species())
    cm1 = {name: np.uint32(int(word, 16)).view(np.float32) for name, word in record["cm1_stats_float32_words"].items()}
    assert set(result.not_produced) == STATS_NOT_PRODUCED
    assert set(result.values) | set(result.not_produced) == set(cm1)
    assert len(result.values) == 102
    differ = {n: (v, float(cm1[n])) for n, v in result.values.items() if np.float32(v) != cm1[n]}
    assert differ == {}


def test_without_the_base_state_the_statistics_agree_to_a_part_in_a_million(parser_tool):
    data, record = _stats_state()
    settings = S.from_namelist(REF / "stats-run.namelist.input")
    result = T.compute(_inputs(data, base_state=False), settings, _geometry(), _species())
    cm1 = {name: float(np.uint32(int(word, 16)).view(np.float32)) for name, word in record["cm1_stats_float32_words"].items()}
    assert set(result.not_produced) == STATS_NOT_PRODUCED | {"divmax", "divmin"}
    for name, value in result.values.items():
        assert value == pytest.approx(cm1[name], rel=2e-6, abs=1e-12), name
    assert {"ppmax", "rhmax", "themax", "ei"} <= set(result.derived)


def test_cm1out_stats_carries_cm1s_names_units_and_order(tmp_path, parser_tool):
    """Through the writer: a frame of CM1's 1500 s state carries CM1's statistics names, units and order."""

    data, record = _stats_state()
    species = _species()
    carriers = {
        "closure.tkea": data["tke"], "pp3d": data["pipert"], "prs": data["prs"],
        "q3d": np.stack([data[n] for n in species.names[:6]]), "nc3d": np.stack([data[n] for n in species.names[6:]]),
        "rho": data["rho"], "th3d": data["thpert"], "u3d": data["u"], "v3d": data["v"], "w3d": data["w"],
        "zf": np.broadcast_to(_geometry().zf.reshape(-1, 1, 1), (41, 24, 24)).copy(),
        "prate": data["prate"], "th": data["th"],
        "pi0": data["pi0"], "th0": data["th0"], "prs0": data["prs0"], "qv0": data["qv0"],
    }
    t = float(data["time"])
    frame = dict(time_s=t, step=int(t / 6), grid=REF_GRID, dt_s=6.0, umove=float(data["umove"]),
                 vmove=float(data["vmove"]), domainloc=(float(data["umove"]) * t, float(data["vmove"]) * t))
    run = _rundir(tmp_path, REF / "stats-run.namelist.input", [(carriers, frame)])
    convert(run)
    header = _header("cm1out_stats.header.json")
    cm1_vars = [row for row in header["variables"] if row["name"] not in ("xh", "yh", "zh", "time")]
    ds = netCDF4.Dataset(run / "cm1out" / "cm1out_stats.nc")
    try:
        # CM1's file holds its own number of records; the unlimited time is ours.
        assert [[k, None if d.isunlimited() else len(d), d.isunlimited()] for k, d in ds.dimensions.items()] == \
               [[k, None if unlimited else n, unlimited] for k, n, unlimited in header["dimensions"]]
        for row in header["variables"]:
            if row["name"] in ("xh", "yh", "zh", "time"):
                var = ds[row["name"]]
                assert list(var.dimensions) == row["dimensions"]
                assert {a: var.getncattr(a) for a in var.ncattrs()} == row["attributes"]
        ours = [v for v in ds.variables if v not in ("xh", "yh", "zh", "time")]
        assert ours == [row["name"] for row in cm1_vars if row["name"] in ds.variables]
        for row in cm1_vars:
            if row["name"] in ds.variables:
                assert {a: ds[row["name"]].getncattr(a) for a in ds[row["name"]].ncattrs()} == row["attributes"]
            else:
                assert row["name"] in ds.getncattr("gm1_not_produced").split()
        words = record["cm1_stats_float32_words"]
        differ = [n for n in ours if f"{int(_words(ds[n][:]).reshape(-1)[0]):08x}" != words[n]]
        assert differ == []
        assert [n for n in ds.ncattrs()][:3] == ["CM1 version", "Conventions", "missing_value"]
    finally:
        ds.close()


# ---------------------------------------------------------------- names and CLI

def test_the_stock_namelist_selects_cm1s_variables_in_cm1s_order(parser_tool):
    settings = S.from_namelist(REF / "namelist.input")
    species = _species()
    chosen = [row.name for row in N.selected(settings.values, N.species_variables(species.names, species.units))]
    cm1 = [row["name"] for row in _header("cm1out_000002.header.json")["variables"]
           if len(row["dimensions"]) >= 3]
    assert chosen == cm1


def test_the_cli_writes_the_files_and_a_json_summary(tmp_path, parser_tool, capsys):
    from cm1core.cli import main

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    assert main(["cm1out", os.fspath(run), "--no-stats", "--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "gm1.cm1out-summary/v1"
    assert record["files"] == ["cm1out_000001.nc"] and record["format"] == "nc"
    assert set(record["not_produced"]) == NOT_PRODUCED_STOCK
    assert main(["cm1out", os.fspath(tmp_path / "nowhere")]) == 2


def test_the_converter_never_imports_cupy(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    code = (
        "import sys; from cm1core.cli import main; "
        f"rc = main(['cm1out', {os.fspath(run)!r}, '--no-stats']); "
        "assert rc == 0; assert 'cupy' not in sys.modules, 'cupy imported'"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.fspath(ROOT / "src"), env_path()]))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True)


def env_path() -> str:
    return os.environ.get("PYTHONPATH", "")


def test_the_user_script_runs_unchanged_on_cm1s_file_and_on_ours(tmp_path, parser_tool):
    """A CM1 user's xarray script reads both files.  The script needs xarray and pandas: ``GM1_XARRAY_PYTHON``
    names an interpreter that has them when the test's own does not."""

    python = os.environ.get("GM1_XARRAY_PYTHON") or sys.executable
    probe = subprocess.run([python, "-c", "import xarray, pandas"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip(f"{python} has no xarray/pandas; set GM1_XARRAY_PYTHON")
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input", [(carriers, frame)])
    convert(run, stats=False)
    script = ROOT / "tools" / "release" / "cm1_user_script.py"
    outputs = []
    for path in (REF / "cm1out_000001.nc", run / "cm1out" / "cm1out_000001.nc"):
        done = subprocess.run([python, "-W", "ignore", os.fspath(script), os.fspath(path)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        outputs.append(done.stdout.split("\n", 1)[1])
    assert outputs[0] == outputs[1]


def test_without_a_namelist_the_deck_it_ran_supplies_the_settings(tmp_path):
    """No native parser needed: the imported stock deck, and GM1's one-file-per-time layout."""

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    # Every GM1 frame carries these (STATE words); CM1's default file does not.
    zeros = np.zeros((40, 24, 24), dtype=np.float32)
    carriers.update({"th3d": zeros, "pp3d": zeros, "rho": zeros + np.float32(1.0)})
    run = tmp_path / "out"
    root = run / "frames"
    root.mkdir(parents=True)
    publish(root, 0, carriers, **frame)
    shutil.copy(ROOT / "tests" / "data" / "decks" / "stock-imported.gpuwm-cm1.toml", run / "deck.toml")
    summary = convert(run, stats=False)
    assert summary.settings_source.startswith("deck ")
    assert summary.files == ["cm1out_000001.nc"]
    header = _header("cm1out_000002.header.json")
    with netCDF4.Dataset(run / "cm1out" / "cm1out_000001.nc") as ds:
        assert ds.ncattrs()[:71] == [name for name, _value in header["global_attributes"]]
        for name, value in header["global_attributes"]:
            if name not in ("CM1 version",):
                assert np.asarray(ds.getncattr(name)).tolist() == value["value"], name
        assert float(ds["xh"][0]) == pytest.approx(-11.5, abs=1e-5)
        # The deck's own selection: GM1 wrote neither km/kh nor reflectivity for
        # Morrison, and always writes thpert, pipert and rho.
        assert set(ds.getncattr("gm1_not_produced").split()) == {"rain2"}
        assert {"thpert", "pipert", "rho"} <= set(ds.variables)


def test_a_resumed_segment_keeps_cm1out_nc_in_time_order(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")

    def at(t: float, value: float) -> tuple[dict, dict]:
        return (dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(value))),
                dict(frame, time_s=t, step=int(t / 6), domainloc=(12.5 * t, 3.0 * t)))

    run = _rundir(tmp_path, REF / "namelist.input", [at(0.0, 1.0), at(60.0, 1.0), at(120.0, 1.0)])
    convert(run, filetype=1, stats=False)
    root2 = run / "segments" / "002" / "frames"
    root2.mkdir(parents=True)
    for index, (c, f) in enumerate([at(60.0, 2.0), at(120.0, 2.0), at(180.0, 2.0)]):
        publish(root2, index, c, **f)
    convert(run, filetype=1, stats=False)
    with netCDF4.Dataset(run / "cm1out" / "cm1out.nc") as ds:
        assert list(ds["time"][:]) == [0.0, 60.0, 120.0, 180.0]
        assert [float(ds["w"][i].max()) for i in range(4)] == [1.0, 2.0, 2.0, 2.0]


# ---------------------------------------------------------------- CM1 variant runs

def _variants() -> dict:
    return json.loads((REF / "variants.json").read_text(encoding="utf-8"))


def _variant_namelist(tmp_path: Path, name: str) -> Path:
    """The conformance namelist with one variant's edits, as the variant run used it."""

    import re

    text = (REF / "namelist.input").read_text(encoding="ascii")
    for key, value in _variants()["edits"][name].items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^,\n]*)(,?)", re.M)
        text, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
        if count == 0 and key == "outunits":
            text, count = text.replace(" &param0\n", f" &param0\n outunits = {value},\n", 1), 1
        assert count == 1, key
    path = tmp_path / f"{name}.namelist.input"
    path.write_text(text, encoding="ascii")
    return path


def _wsm6_species():
    spec = {"names": ["qv", "qc", "qr", "qi", "qs", "qg"],
            "cm1_index_ranges": {"nqv": 1, "nql1": 2, "nql2": 3, "nqs1": 4, "nqs2": 6},
            "iice": 1, "idm": 0, "numq": 6, "component_id": "wsm6-mp6", "cm1_ptype": 7}
    return S.species(type("Frame", (), {"publication": {"config": {"moisture_species": species_record(spec)}},
                                        "directory": REF})())


_GRID_DIMS = {"2": ["time", "yh", "xh"], "s": ["time", "zh", "yh", "xh"], "u": ["time", "zh", "yh", "xf"],
              "v": ["time", "zh", "yf", "xh"], "w": ["time", "zf", "yh", "xh"]}


@pytest.mark.parametrize("variant", sorted(json.loads((REF / "variants.json").read_text())["variants"]))
def test_cm1s_variant_runs_select_cm1s_variables(tmp_path, parser_tool, variant):
    """Every variable CM1 wrote for each variant, and no other: names, order, dims, long_name, units.

    The variants turn the switches and settings that change what CM1 writes:
    the budgets (param.F's td_/qd_/ud_/vd_/wd_ slots and the Morrison process
    rates), sgsmodel = 2 (no tke), the base state and derived fields, the
    moving domain, Coriolis, units and the three file layouts.
    """

    settings = S.from_namelist(_variant_namelist(tmp_path, variant))
    species = _species() if settings.int("ptype") == 5 else _wsm6_species()
    chosen = [[row.name, _GRID_DIMS[row.grid], row.long_name, row.units]
              for row in N.selected(settings.values, N.species_variables(species.names, species.units))]
    assert chosen == _variants()["variants"][variant]["variables"]


def test_coordinates_widen_cm1s_real_dx(tmp_path, parser_tool):
    """dx = 333.33 is a REAL in CM1: xfref is dble(dx)*(i-1), not the decimal's binary64."""

    settings = S.from_namelist(_variant_namelist(tmp_path, "dx"))
    geometry = S.Geometry(24, 24, 40, 333.33, 333.33, 500.0, _geometry().zf, _geometry().zh,
                          _geometry().mh, _geometry().rmh, 6.0)
    from cm1core.cm1out.netcdf import coordinates

    coords = coordinates(settings, geometry)
    words = _variants()["variants"]["dx"]["float32_words"]
    for name in ("xh", "xf", "yh", "yf"):
        assert [f"{int(w):08x}" for w in _words(coords[name])] == words[name], name


@pytest.mark.parametrize("variant", ["stretch", "stretchg"])
def test_stretched_coordinates_and_fornberg_attributes_are_cm1s(tmp_path, parser_tool, variant):
    from cm1core.cm1_vertical_grid import build_vertical_levels
    from cm1core.cm1out.netcdf import coordinates

    settings = S.from_namelist(_variant_namelist(tmp_path, variant))
    levels = build_vertical_levels(
        stretch_z=1, nk=40, dz_m=settings.float("dz"), ztop_m=settings.float("ztop"),
        str_bot_m=settings.float("str_bot"), str_top_m=settings.float("str_top"),
        dz_bot_m=settings.float("dz_bot"), dz_top_m=settings.float("dz_top"))
    zf = np.asarray(levels.zf, dtype=np.float32).reshape(-1)
    zf = zf if zf.size == 41 else zf[1:42]
    zh = (np.float32(0.5) * (zf[1:] + zf[:-1])).astype(np.float32)
    mh = (np.float32(500.0) / (zf[1:] - zf[:-1])).astype(np.float32)
    geometry = S.Geometry(24, 24, 40, 1000.0, 1000.0, 500.0, zf, zh, mh, (np.float32(1.0) / mh).astype(np.float32), 6.0)
    record = _variants()["variants"][variant]
    coords = coordinates(settings, geometry)
    for name in ("zh", "zf", "ztop"):
        assert [f"{int(w):08x}" for w in _words(coords[name]).reshape(-1)] == record["float32_words"][name], name
    ends = T.fornberg_ends(geometry)
    for prefix in ("cgs", "dgs", "cgt", "dgt"):
        for index, value in enumerate(ends[prefix], 1):
            assert [f"{int(w):08x}" for w in _words([value])] == record["global_attributes"][f"{prefix}{index}"]


def test_pi_is_pi0_plus_pipert_and_a_dropped_prs_is_rebuilt(tmp_path, parser_tool):
    """writeout.F:3336-3343 writes pi0+ppi (psolver 3); the reduced carrier profile drops prs."""

    block = _variants()["variants"]["derived"]["block"]
    unwords = lambda rows: np.array([int(w, 16) for w in rows], dtype=np.uint32).view(np.float32)  # noqa: E731
    pipert = unwords(block["pipert"]).reshape(40, 4, 4)
    pi0 = unwords(block["pi0_column"])
    zeros = np.zeros((40, 4, 4), np.float32)
    zf = (np.arange(41, dtype=np.float32) * np.float32(500.0)).astype(np.float32)
    carriers = {
        "u3d": np.zeros((40, 4, 5), np.float32), "v3d": np.zeros((40, 5, 4), np.float32),
        "w3d": np.zeros((41, 4, 4), np.float32), "closure.tkea": np.zeros((41, 4, 4), np.float32),
        "q3d": np.zeros((6, 40, 4, 4), np.float32), "nc3d": np.zeros((4, 40, 4, 4), np.float32),
        "th3d": zeros, "pp3d": pipert, "rho": zeros + np.float32(1.0), "base.pi0": pi0,
        "metrics.zf_profile": zf,
    }
    grid = {"nx": 4, "ny": 4, "nz": 40, "dx_m": 1000.0, "dy_m": 1000.0, "dz_m": 500.0, "ztop_m": 20000.0}
    run = _rundir(tmp_path, _variant_namelist(tmp_path, "derived"),
                  [(carriers, dict(time_s=60.0, step=10, grid=grid, dt_s=6.0))])
    convert(run)
    with netCDF4.Dataset(run / "cm1out" / "cm1out_000002.nc") as ds:
        assert [f"{int(w):08x}" for w in _words(ds["pi"][0]).reshape(-1)] == block["pi"]
        formed = json.loads(ds.getncattr("gm1_formed"))
        assert formed["pi"] == "derived" and formed["prs"] == "derived"
        cm1_prs = unwords(block["prs"]).view(np.int32).astype(np.int64)
        ours = _words(ds["prs"][0]).reshape(-1).view(np.int32).astype(np.int64)
        assert np.abs(ours - cm1_prs).max() <= 1, "prs rebuilt within one unit in the last place"
    with netCDF4.Dataset(run / "cm1out" / "cm1out_stats.nc") as ds:
        assert "ppmax" in ds.variables and "sprsmin" in ds.variables


def test_settings_whose_fields_gm1_never_writes_are_named(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    import re

    text = (REF / "namelist.input").read_text()
    for key in ("output_sfcflx", "output_fallvel", "output_zs", "output_radten"):
        text = re.sub(rf"^(\s*{key}\s*=\s*)0", r"\g<1>1", text, count=1, flags=re.M)
    namelist = tmp_path / "switches.namelist.input"
    namelist.write_text(text)
    run = _rundir(tmp_path, namelist, [(carriers, frame)])
    summary = convert(run, stats=False)
    # output_zs needs terrain and output_radten a radiation scheme (param.F:4648, :4730-4735):
    # CM1 turns both off itself, so neither is named.
    assert set(summary.switches_not_produced) == {"output_sfcflx", "output_fallvel"}
    with netCDF4.Dataset(run / "cm1out" / "cm1out_000001.nc") as ds:
        assert ds.getncattr("gm1_switches_not_produced").split() == ["output_fallvel", "output_sfcflx"]
        assert "zs" not in ds.variables


# ---------------------------------------------------------------- the conversion's own bookkeeping

def _at(carriers: dict, frame: dict, t: float, value: float) -> tuple[dict, dict]:
    return (dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(value))),
            dict(frame, time_s=t, step=int(t / 6), domainloc=(12.5 * t, 3.0 * t)))


def test_frames_off_the_tapfrq_grid_never_share_a_file(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input",
                  [_at(carriers, frame, t, n + 1.0) for n, t in enumerate((0.0, 30.0, 60.0, 120.0))])
    summary = convert(run, stats=False)
    assert summary.files == ["cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc", "cm1out_000004.nc"]
    for n, name in enumerate(summary.files):
        with netCDF4.Dataset(run / "cm1out" / name) as ds:
            assert float(ds["w"][0].max()) == n + 1.0, name


def test_a_new_layout_or_compression_starts_the_output_again(tmp_path, parser_tool):
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = _rundir(tmp_path, REF / "namelist.input", [_at(carriers, frame, t, 1.0) for t in (0.0, 60.0)])
    out = run / "cm1out"
    convert(run, stats=False)
    convert(run, stats=False, filetype=1)
    assert sorted(p.name for p in out.glob("*.nc")) == ["cm1out.nc"]
    convert(run, stats=False, filetype=1, deflate=1)
    with netCDF4.Dataset(out / "cm1out.nc") as ds:
        assert list(ds["time"][:]) == [0.0, 60.0]
        assert ds["w"].filters()["zlib"] is True
    convert(run, stats=False)
    assert sorted(p.name for p in out.glob("*.nc")) == ["cm1out_000001.nc", "cm1out_000002.nc"]


def test_follow_waits_for_a_run_that_has_not_started(tmp_path, parser_tool):
    """A sidecar started before its run: no segments/ yet, then a segment, frames and the end record."""

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(REF / "namelist.input", run / "namelist.input")

    def model() -> None:
        time.sleep(0.5)
        seg = run / "segments" / "001"
        (seg / "frames").mkdir(parents=True)
        (seg / "events.jsonl").write_text(json.dumps({"schema": "gm1.run-events/v1", "kind": "admitted"}) + "\n")
        for index, t in enumerate((0.0, 60.0)):
            time.sleep(0.2)
            fields, where = _at(carriers, frame, t, 1.0)
            publish(seg / "frames", index, fields, **where)
        end_event(seg / "events.jsonl", frames=2)

    thread = threading.Thread(target=model)
    started = time.monotonic()
    thread.start()
    summary = convert(run, follow=True, poll_s=0.05, stats=False,
                      should_stop=lambda: time.monotonic() - started > 30)
    thread.join()
    assert time.monotonic() - started < 30
    assert summary.files == ["cm1out_000001.nc", "cm1out_000002.nc"] and summary.ended
    # Without --follow a directory with no history is still refused by name.
    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copy(REF / "namelist.input", empty / "namelist.input")
    with pytest.raises(F.FrameError, match="holds no GM1 history"):
        convert(empty, stats=False)
