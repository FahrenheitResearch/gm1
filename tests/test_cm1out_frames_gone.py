"""``gpuwm-cm1 cm1out`` keeps the files of converted times whose saved frames are gone.

A run's saved frames can be deleted by hand.  The files converted from them are then all there is of those times:
a pass, Convert again (``--force``) and a change of layout must not delete them.  Only the run's own records (run.json:
which segment resumed from which restart) say that a converted time belongs to a branch the run has left, and only
those times' files are removed, as before (``test_gui_release_seams``).

CPU only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

pytest.importorskip("netCDF4")

from cm1core.cm1out import convert  # noqa: E402
from cm1core.cm1out import pdata as P  # noqa: E402
from cm1core.cm1out.convert import Cm1outError, converted_lost  # noqa: E402

from cm1import_support import namelist_tool  # noqa: E402
from cm1out_support import REF, frame_from_cm1, publish, write_track  # noqa: E402


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))


def run_directory(tmp_path: Path) -> Path:
    from cm1core.cm1run.rundir import RUN_SCHEMA, write_json_atomic

    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(REF / "namelist.input", run / "namelist.input")
    write_json_atomic(run / "run.json", {"schema": RUN_SCHEMA, "name": "run", "segments": []})
    return run


def segment(run: Path, number: int, frames: list[tuple[int, float]], *,
            resumed_from: tuple[int, float] | None = None) -> None:
    from cm1core.cm1run.rundir import RunDirectory

    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    root = run / "segments" / f"{number:03d}" / "frames"
    root.mkdir(parents=True)
    for index, t in frames:
        publish(root, index, dict(carriers, w3d=np.full_like(carriers["w3d"], np.float32(t / 60.0 + 1.0))),
                **dict(frame, time_s=t, step=int(t / 6), domainloc=(12.5 * t, 3.0 * t)))

    def add(record: dict) -> None:
        record["segments"].append({"number": number, "directory": f"segments/{number:03d}",
                                   "resumed_from": None if resumed_from is None else {
                                       "segment": resumed_from[0], "model_time_s": resumed_from[1], "restart": 1}})

    RunDirectory(run).update(add)


def files(run: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run / "cm1out").iterdir())
            if p.is_file()}


def test_converted_files_whose_frames_were_deleted_are_kept_and_convert_again_is_refused(tmp_path, capsys):
    from cm1core.cli import main

    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0), (2, 120.0), (3, 180.0)])
    convert(run)
    before = files(run)
    assert [name for name in before if name.startswith("cm1out_0")] == [
        "cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc", "cm1out_000004.nc"]
    for index in (1, 3):
        shutil.rmtree(run / "segments" / "001" / "frames" / f"frame-{index:06d}")
    # a pass keeps every file: those two times were never left by the run, their frames were only deleted
    summary = convert(run)
    assert summary.frames_converted == 0 and files(run) == before
    # Convert again would delete the two files it cannot write again: refused, nothing deleted
    with pytest.raises(Cm1outError) as refused:
        convert(run, force=True)
    assert str(refused.value) == converted_lost(2, 4, "converting every frame again")
    assert files(run) == before
    assert main(["cm1out", os.fspath(run), "--force"]) == 2
    assert "2 of this run's 4 converted times have lost their saved frames" in capsys.readouterr().err
    # so would another layout (one file for every time)
    with pytest.raises(Cm1outError, match="writing them in another layout or compression would delete them"):
        convert(run, filetype=1)
    assert files(run) == before
    # every frame deleted: a pass still keeps every file
    shutil.rmtree(run / "segments" / "001" / "frames")
    (run / "segments" / "001" / "frames").mkdir()
    assert convert(run).frames_converted == 0 and files(run) == before
    with pytest.raises(Cm1outError):
        convert(run, force=True)
    assert files(run) == before


def test_a_segment_whose_folder_was_deleted_is_not_a_branch_the_run_left(tmp_path):
    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0)])
    segment(run, 2, [(2, 120.0)], resumed_from=(1, 60.0))
    convert(run)
    before = files(run)
    assert [name for name in before if name.startswith("cm1out_0")] == [
        "cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc"]
    shutil.rmtree(run / "segments" / "001")
    assert convert(run).frames_converted == 0 and files(run) == before


def test_a_branch_the_run_left_still_loses_its_files_beside_frames_that_were_deleted(tmp_path):
    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0), (2, 120.0), (3, 180.0)])
    convert(run)
    shutil.rmtree(run / "segments" / "001" / "frames" / "frame-000000")       # deleted by hand: kept
    segment(run, 2, [(2, 120.0)], resumed_from=(1, 60.0))                      # 180 s: the branch the run left
    summary = convert(run)
    assert summary.frames_converted == 1
    assert sorted(name for name in files(run) if name.startswith("cm1out_0")) == [
        "cm1out_000001.nc", "cm1out_000002.nc", "cm1out_000003.nc"]


def test_a_parcel_file_whose_track_was_deleted_is_kept(tmp_path):
    """The parcel track is kept beside the saved frames: deleted with them, CM1's parcel file is all that is left of
    it."""

    import json

    header = json.loads((REF / "grads" / "pg" / "cm1out_pdata.header.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in header["variables"] if row["name"] not in ("xh", "yh", "zh", "time")]
    units = {row["name"]: row["attributes"]["units"] for row in header["variables"]}
    run = tmp_path / "run"
    root = run / "segments" / "001" / "frames"
    root.mkdir(parents=True)
    shutil.copy(REF / "namelist.input", run / "namelist.input")
    carriers, frame = frame_from_cm1(REF / "cm1out_000001.nc")
    publish(root, 0, carriers, **frame)
    times = [60.0, 120.0, 180.0]
    rng = np.random.default_rng(5)
    rows = [rng.normal(0.0, 1.0, (len(names), 20)).astype(np.float32) for _t in times]
    for row, t in zip(rows, times):
        row[names.index("mtime")] = np.float32(t)
    write_track(root / P.TRACK_DIRECTORY, names, [units[n] for n in names], rows, times)
    assert convert(run, stats=False).pdata_files == [P.NETCDF_FILENAME]
    before = files(run)
    shutil.rmtree(root / P.TRACK_DIRECTORY)
    convert(run, stats=False)
    assert files(run) == before


# ---------------------------------------------------------------- a resumed run whose newest segment was deleted

def test_a_resumed_run_whose_newest_segment_folder_was_deleted_keeps_its_files(tmp_path):
    """Resumed from 60 s: segment 2 converted 120 s again.  With segment 2's whole folder deleted, segment 1's own
    120 s frame is of the branch the run left: the run's 120 s file is kept, not written again from that branch."""

    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0), (2, 120.0)])
    segment(run, 2, [(2, 120.0)], resumed_from=(1, 60.0))
    convert(run)
    before = files(run)
    shutil.rmtree(run / "segments" / "002")
    assert convert(run).frames_converted == 0 and files(run) == before
    with pytest.raises(Cm1outError, match="1 of this run's 3 converted times have lost their saved frames"):
        convert(run, force=True)
    assert files(run) == before


# ---------------------------------------------------------------- the record of what was converted lost

def converted(run: Path) -> dict[str, str]:
    return {name: digest for name, digest in files(run).items() if name.startswith("cm1out")}


@pytest.mark.parametrize("how", ["gone", "torn"])
def test_a_lost_record_does_not_let_a_pass_write_over_files_of_lost_frames(tmp_path, capsys, how):
    """Without its record a pass took every converted time for not converted: it wrote the statistics again from
    the frames left, fewer times than they held.  What the files hold is read from them instead."""

    from cm1core.cli import main

    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0), (2, 120.0), (3, 180.0)])
    convert(run)
    before = converted(run)
    assert "cm1out_stats.nc" in before
    for index in (1, 3):
        shutil.rmtree(run / "segments" / "001" / "frames" / f"frame-{index:06d}")
    state = run / "cm1out" / ".cm1out-state.json"
    if how == "gone":
        state.unlink()
    else:
        state.write_text(state.read_text()[:300])
    for force in (False, True):
        with pytest.raises(Cm1outError) as refused:
            convert(run, force=force)
        assert "4 converted times, 2 of which have lost their saved frames" in str(refused.value), force
        assert "Nothing was changed; the files are kept" in str(refused.value)
    assert main(["cm1out", os.fspath(run)]) == 2
    assert "is missing or cannot be read" in capsys.readouterr().err
    assert converted(run) == before


def test_one_file_for_every_time_is_not_written_over_without_its_record(tmp_path):
    """output_filetype = 1: a pass without the record appended the frames left into cm1out.nc's first places,
    over the times whose frames are gone."""

    import netCDF4

    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0), (2, 120.0), (3, 180.0)])
    convert(run, filetype=1)
    before = converted(run)
    shutil.rmtree(run / "segments" / "001" / "frames" / "frame-000001")
    (run / "cm1out" / ".cm1out-state.json").unlink()
    with pytest.raises(Cm1outError, match="1 of which has lost its saved frame and could not be written again"):
        convert(run, filetype=1)
    assert converted(run) == before
    with netCDF4.Dataset(run / "cm1out" / "cm1out.nc") as ds:
        assert [float(t) for t in ds.variables["time"][:]] == [0.0, 60.0, 120.0, 180.0]


def test_a_lost_record_with_every_frame_there_converts_as_before(tmp_path):
    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0)])
    convert(run)
    names = sorted(converted(run))
    (run / "cm1out" / ".cm1out-state.json").unlink()
    assert convert(run).frames_converted == 2 and sorted(converted(run)) == names


def test_the_gui_refuses_convert_when_the_record_is_lost_and_frames_are_gone(tmp_path):
    from cm1core.gui.files import converted_unrecorded
    from cm1core.gui.runview import convert_refused

    run = run_directory(tmp_path)
    segment(run, 1, [(0, 0.0), (1, 60.0), (2, 120.0)])
    convert(run)
    (run / "cm1out" / ".cm1out-state.json").unlink()
    assert not converted_unrecorded(run) and convert_refused(run) is None
    shutil.rmtree(run / "segments" / "001" / "frames" / "frame-000001")
    assert converted_unrecorded(run)
    assert "The record of what this run converted is missing" in convert_refused(run)

