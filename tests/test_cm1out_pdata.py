"""``cm1out_pdata``: CM1's parcel file from a GM1 run's parcel track.

The track is written by the runtime's own writer
(``lagrangian_stage.write_parcel_track``).  ``cm1out_pdata.nc`` must read
back to ``parcel-attributes.f32`` word for word and be laid
out as CM1 r21.1's own (``tests/data/cm1-ref-24x24/grads/pg/
cm1out_pdata.header.json``, a CM1 run of the conformance namelist at 120 x
120 with ``iprcl = 1``); ``cm1out_pdata.dat`` is the arena's own records and
``cm1out_parcel.ctl`` is CM1's descriptor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from cm1core.cm1out import convert  # noqa: E402
from cm1core.cm1out import grads as G  # noqa: E402
from cm1core.cm1out import pdata as P  # noqa: E402

from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import REF, end_event, frame_from_cm1, publish, write_track  # noqa: E402

GREF = REF / "grads"
HEADER = json.loads((GREF / "pg" / "cm1out_pdata.header.json").read_text(encoding="utf-8"))
#: The conformance namelist's &param13 columns, as CM1 wrote them (pg/cm1out_parcel.ctl).
CM1_NAMES = [row["name"] for row in HEADER["variables"] if row["name"] not in ("xh", "yh", "zh", "time")]
UNITS = {row["name"]: row["attributes"]["units"] for row in HEADER["variables"]}


@pytest.fixture()
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


def _track_frames(nparcels: int, times: list[float], seed: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = []
    for t in times:
        rows = rng.normal(0.0, 1.0, (len(CM1_NAMES), nparcels)).astype(np.float32)
        rows[CM1_NAMES.index("mtime")] = np.float32(t)
        rows[CM1_NAMES.index("qsl")] = np.float32(-1.0e30)  # an attribute GM1 does not fill
        rows[:, -1] = np.float32(-1.0e30)                    # a parcel never placed
        frames.append(rows)
    return frames


def _run(tmp_path: Path, *, output_format: int, times: list[float], nparcels: int = 50,
         segment: str = "001") -> tuple[Path, list[np.ndarray]]:
    run = tmp_path / "run"
    root = run / "segments" / segment / "frames"
    root.mkdir(parents=True, exist_ok=True)
    text = (REF / "namelist.input").read_text().replace("output_format    = 2", f"output_format    = {output_format}")
    (run / "namelist.input").write_text(text)
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    publish(root, 0, carriers, **frame)
    frames = _track_frames(nparcels, times)
    write_track(root / P.TRACK_DIRECTORY, CM1_NAMES, [UNITS[n] for n in CM1_NAMES], frames, times,
                unfilled=("qsl",))
    return run, frames


def test_cm1out_pdata_nc_round_trips_the_arena_word_for_word(tmp_path, parser_tool):
    """cm1out_pdata.nc reads back to the parcel arena word for word."""

    run, _frames = _run(tmp_path, output_format=2, times=[60.0, 120.0, 180.0])
    summary = convert(run, stats=False)
    assert summary.pdata_files == ["cm1out_pdata.nc"]
    arena = (run / "segments" / "001" / "frames" / P.TRACK_DIRECTORY / P.ARENA).read_bytes()
    data, names, times = P.read_back(run / "cm1out" / "cm1out_pdata.nc")
    assert names == CM1_NAMES and times == [60.0, 120.0, 180.0]
    assert data.tobytes() == arena


def test_cm1out_pdata_nc_is_laid_out_as_cm1s(tmp_path, parser_tool):
    run, _frames = _run(tmp_path, output_format=2, times=[60.0, 120.0])
    convert(run, stats=False)
    with netCDF4.Dataset(run / "cm1out" / "cm1out_pdata.nc") as ds:
        assert ds.data_model == HEADER["data_model"] == "NETCDF4_CLASSIC"
        assert [(n, d.isunlimited()) for n, d in ds.dimensions.items()] == \
               [(n, unlimited) for n, _size, unlimited in HEADER["dimensions"]]
        assert len(ds.dimensions["xh"]) == 50 and len(ds.dimensions["time"]) == 2
        assert list(ds.variables) == [row["name"] for row in HEADER["variables"]]
        for row in HEADER["variables"]:
            var = ds[row["name"]]
            assert list(var.dimensions) == row["dimensions"] and str(var.dtype) == row["dtype"], row["name"]
            assert {a: var.getncattr(a) for a in var.ncattrs()} == row["attributes"], row["name"]
        cm1_globals = [name for name, _value in HEADER["global_attributes"]]
        assert ds.ncattrs()[:len(cm1_globals)] == cm1_globals
        assert ds.getncattr("CM1 version").startswith("cm1r21.1")
        assert ds.getncattr("missing_value") == np.float32(-999999.9)
        assert all(n.startswith("gm1_") for n in ds.ncattrs()[len(cm1_globals):])
        assert ds.getncattr("gm1_unfilled") == "qsl"
        assert list(ds["xh"][:4]) == [1.0, 2.0, 3.0, 4.0]
        notes = json.loads(ds.getncattr("gm1_notes"))
        assert "first_time" in notes and "unfilled" in notes


def test_grads_pdata_is_the_arena_and_cm1s_descriptor(tmp_path, parser_tool):
    run, _frames = _run(tmp_path, output_format=1, times=[60.0, 120.0], nparcels=36000)
    summary = convert(run, stats=False)
    out = run / "cm1out"
    assert summary.pdata_files == ["cm1out_parcel.ctl", "cm1out_pdata.dat"]
    arena = (run / "segments" / "001" / "frames" / P.TRACK_DIRECTORY / P.ARENA).read_bytes()
    assert (out / "cm1out_pdata.dat").read_bytes() == arena
    mine = (out / "cm1out_parcel.ctl").read_text(encoding="ascii").split("\n")
    cm1 = (GREF / "pg" / "cm1out_parcel.ctl").read_text(encoding="ascii").split("\n")
    assert mine[1] == cm1[1].replace("version cm1r21.1", f"version {G.GRADS_VERSION}")
    assert [line for i, line in enumerate(mine) if i != 1] == [line for i, line in enumerate(cm1) if i != 1]
    data, names, times = P.read_back(out / "cm1out_pdata.dat")
    assert names == CM1_NAMES and times == [60.0, 120.0] and data.tobytes() == arena
    assert "parcels" in summary.open_with() and summary.open_with()["parcels"].endswith("cm1out_parcel.ctl")
    # The oracle's reader of CM1 history leaves the parcel descriptor alone (its dbz, th, u ... are
    # parcel columns, not history fields).
    from cm1core.oracle.grads import open_history

    assert "parcel" not in open_history(out).datasets and "s" in open_history(out).datasets


def test_a_run_without_parcels_writes_no_pdata(tmp_path, parser_tool):
    run = tmp_path / "run"
    root = run / "segments" / "001" / "frames"
    root.mkdir(parents=True)
    shutil.copy(REF / "namelist.input", run / "namelist.input")
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    publish(root, 0, carriers, **frame)
    summary = convert(run, stats=False)
    assert summary.pdata_files == [] and not (run / "cm1out" / "cm1out_pdata.nc").exists()


def test_the_lineage_decides_which_segments_parcels_count(tmp_path, parser_tool):
    """A resume from 120 s: the first segment counts to 120 s, the second (reseeded) after it."""

    from cm1core.cm1run.rundir import RUN_SCHEMA, RunDirectory, write_json_atomic

    run, first = _run(tmp_path, output_format=2, times=[60.0, 120.0, 180.0])
    write_json_atomic(run / "run.json", {"schema": RUN_SCHEMA, "name": "run", "segments": [
        {"number": 1, "directory": "segments/001", "resumed_from": None}]})
    root2 = run / "segments" / "002" / "frames"
    root2.mkdir(parents=True)
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    publish(root2, 0, carriers, **dict(frame, time_s=180.0, step=30))
    second = _track_frames(50, [180.0, 240.0], seed=9)
    write_track(root2 / P.TRACK_DIRECTORY, CM1_NAMES, [UNITS[n] for n in CM1_NAMES], second, [180.0, 240.0],
                unfilled=("qsl",))
    RunDirectory(run).update(lambda record: record["segments"].append(
        {"number": 2, "directory": "segments/002", "resumed_from": {"segment": 1, "model_time_s": 120.0,
                                                                      "restart": 1}}))
    summary = convert(run, stats=False)
    data, _names, times = P.read_back(run / "cm1out" / "cm1out_pdata.nc")
    assert times == [60.0, 120.0, 180.0, 240.0]
    assert data[:2].tobytes() == np.stack(first[:2]).tobytes()
    assert data[2:].tobytes() == np.stack(second).tobytes()
    assert "reseeded" in summary.pdata["notes"] and "180" in summary.pdata["notes"]["reseeded"]


def test_follow_writes_the_track_the_run_publishes_at_its_end(tmp_path, parser_tool):
    run = tmp_path / "run"
    seg = run / "segments" / "001"
    root = seg / "frames"
    root.mkdir(parents=True)
    shutil.copy(REF / "namelist.input", run / "namelist.input")
    (seg / "events.jsonl").write_text(json.dumps({"schema": "gm1.run-events/v1", "kind": "admitted"}) + "\n")
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    frames = _track_frames(20, [60.0])

    def model() -> None:
        time.sleep(0.2)
        publish(root, 0, carriers, **frame)
        time.sleep(0.2)
        # the runtime writes the track as it finishes, before the end record
        write_track(root / P.TRACK_DIRECTORY, CM1_NAMES, [UNITS[n] for n in CM1_NAMES], frames, [60.0])
        end_event(seg / "events.jsonl", frames=1)

    thread = threading.Thread(target=model)
    thread.start()
    summary = convert(run, follow=True, poll_s=0.05, stats=False)
    thread.join()
    assert summary.ended and summary.pdata_files == ["cm1out_pdata.nc"]
    data, _names, _times = P.read_back(run / "cm1out" / "cm1out_pdata.nc")
    assert data.tobytes() == np.stack(frames).tobytes()


def test_a_changed_track_is_written_again_and_an_unchanged_one_is_not(tmp_path, parser_tool):
    run, _frames = _run(tmp_path, output_format=2, times=[60.0])
    convert(run, stats=False)
    path = run / "cm1out" / "cm1out_pdata.nc"
    stamp = path.stat().st_mtime_ns
    convert(run, stats=False)
    assert path.stat().st_mtime_ns == stamp
    track = run / "segments" / "001" / "frames" / P.TRACK_DIRECTORY
    shutil.rmtree(track)
    frames = _track_frames(50, [60.0, 120.0], seed=11)
    write_track(track, CM1_NAMES, [UNITS[n] for n in CM1_NAMES], frames, [60.0, 120.0])
    convert(run, stats=False)
    data, _names, times = P.read_back(path)
    assert times == [60.0, 120.0] and data.tobytes() == np.stack(frames).tobytes()
    assert convert(run, stats=False, pdata=False).pdata_files == ["cm1out_pdata.nc"]


def test_descriptions_are_cm1s_forty_characters():
    assert P.description("mtime") == "model time (seconds since beginning of s"
    assert P.description("ncr") == "ncr"
    assert P.description("xvortb_tilt").startswith("GM1 vorticity budget")
