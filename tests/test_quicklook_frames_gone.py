"""A run whose saved frames are gone, all of them or some, keeps every picture it has.

A run's saved frames can be deleted by hand, or by the GUI's Keep pictures only (which keeps the pictures).  Its
pictures are then all there is of those times: a quick-look pass, ``--force``, the radar, a look drawn once per run,
a film made from the frames and a comparison must not take them away or draw over them from less.  Only the run's
own records (run.json: which segment resumed from which restart) say that a drawn time belongs to a branch the run
has left, and only those times' pictures are removed, as before.

CPU only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import warnings

import pytest

from cm1import_support import namelist_tool
from cm1out_support import end_event
from quicklook_support import NAMELIST_24, publish_storm

from cm1core.quicklook import render as RD

NEEDS_MATPLOTLIB = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                                      reason="matplotlib is not installed (it comes with gpuwm)")


@pytest.fixture(autouse=True)
def parser_tool(monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))
    # the quicklook command, run in this process, makes it a sidecar (quicklook.worker): its settings are undone after
    from cm1core.quicklook.worker import THREAD_VARIABLES

    for name in ("CUDA_VISIBLE_DEVICES", "MPLBACKEND", *THREAD_VARIABLES):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:                                   # recorded as unset, so it is unset again afterwards
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)


def add_segment(run: Path, number: int, frames: list[tuple[int, float]], *,
                resumed_from: tuple[int, float] | None = None) -> None:
    """``segments/NNN`` with published frames (index, time), its end record and its row in run.json."""

    from cm1core.cm1run.rundir import RunDirectory

    root = run / "segments" / f"{number:03d}" / "frames"
    root.mkdir(parents=True)
    for index, t in frames:
        publish_storm(root, index, t, scale=0.6 + 0.4 * t / 1500.0)
    end_event(root.parent / "events.jsonl", frames=len(frames))

    def add(record: dict) -> None:
        record["segments"].append({"number": number, "directory": f"segments/{number:03d}",
                                   "resumed_from": None if resumed_from is None else {
                                       "segment": resumed_from[0], "model_time_s": resumed_from[1], "restart": 1}})

    RunDirectory(run).update(add)


def gui_run(root: Path, name: str = "storm", times: tuple[float, ...] = (900.0, 1200.0, 1500.0)) -> Path:
    """A ``gpuwm-cm1 run`` directory (run.json, segments/001) holding CM1's storm at ``times``."""

    from cm1core.cm1run.rundir import RUN_SCHEMA, write_json_atomic

    run = root / name
    run.mkdir(parents=True)
    shutil.copy(NAMELIST_24, run / "namelist.input")
    write_json_atomic(run / "run.json", {"schema": RUN_SCHEMA, "name": name, "segments": []})
    add_segment(run, 1, list(enumerate(times)))
    return run


def draw(run: Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return RD.render(run, **kwargs)


def pictures(run: Path) -> dict[str, str]:
    """Every picture under the run's quicklook folder, and its bytes' digest."""

    folder = run / "quicklook"
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*.png"))}


def index_times(run: Path) -> list[float]:
    return [f["time_s"] for f in json.loads((run / "quicklook" / "index.json").read_text())["frames"]]


def delete_frames(run: Path) -> None:
    for frames in run.glob("segments/*/frames"):
        shutil.rmtree(frames)


@pytest.fixture()
def drawn(tmp_path):
    run = gui_run(tmp_path / "runs")
    draw(run)
    assert index_times(run) == [900.0, 1200.0, 1500.0]
    return run


@NEEDS_MATPLOTLIB
def test_a_pass_over_a_run_whose_frames_were_deleted_keeps_every_picture(drawn):
    run = drawn
    before, index = pictures(run), (run / "quicklook" / "index.json").read_bytes()
    assert len(before) > 100
    delete_frames(run)
    assert not RD.has_saved_frames(RD.Renderer(run).layout)
    seen: list[str] = []
    summary = draw(run, log=seen.append)
    # nothing removed, nothing drawn, the index as it was; the pass says why in plain words
    assert pictures(run) == before and (run / "quicklook" / "index.json").read_bytes() == index
    assert summary.frames_drawn == 0 and summary.notes == [RD.FRAMES_GONE]
    assert not [line for line in seen if "removed its pictures" in line] and RD.FRAMES_GONE in seen


@NEEDS_MATPLOTLIB
def test_force_refuses_a_run_whose_frames_were_deleted_and_deletes_nothing(drawn, capsys):
    from cm1core.cli import main

    run = drawn
    before = pictures(run)
    delete_frames(run)
    with pytest.raises(RD.QuicklookError) as refused:
        draw(run, force=True)
    assert str(refused.value) == RD.FRAMES_GONE_FORCE and pictures(run) == before
    # the commands, with and without --force (the GUI's Draw every picture again and Draw the new looks run them)
    assert main(["quicklook", os.fspath(run), "--force"]) == 2
    assert RD.FRAMES_GONE_FORCE in capsys.readouterr().err and pictures(run) == before
    assert main(["quicklook", os.fspath(run)]) == 0
    assert RD.FRAMES_GONE in capsys.readouterr().err and pictures(run) == before
    # the radar of every saved time cannot be drawn again either: refused, its settings unchanged
    assert main(["quicklook", os.fspath(run), "--radar-site", "10", "5"]) == 2
    assert "the radar cannot scan it again" in capsys.readouterr().err
    assert pictures(run) == before and not (run / "quicklook" / "radar.json").exists()


@NEEDS_MATPLOTLIB
def test_a_run_trimmed_to_its_pictures_keeps_them_through_a_pass_and_force(drawn, capsys):
    """What Keep pictures only leaves: no saved frames, restarts or cm1out, and trimmed.json."""

    from cm1core.cli import main

    run = drawn
    before = pictures(run)
    delete_frames(run)
    (run / "trimmed.json").write_text(json.dumps({"schema": "gm1.gui-trimmed/v1", "freed_bytes": 1,
                                                  "removed": ["segments/001/frames"]}))
    for args in ([], ["--force"]):
        main(["quicklook", os.fspath(run), *args])
        capsys.readouterr()
        assert pictures(run) == before, args


@NEEDS_MATPLOTLIB
def test_a_picture_whose_frame_alone_was_deleted_is_kept(drawn):
    run = drawn
    before = pictures(run)
    shutil.rmtree(run / "segments" / "001" / "frames" / "frame-000001")      # the 1200 s frame
    draw(run)
    kept = pictures(run)
    assert {name for name in before if "t001200" in name} <= set(kept)
    assert all(kept[name] == digest for name, digest in before.items() if "t001200" in name)
    assert index_times(run) == [900.0, 1200.0, 1500.0]


@NEEDS_MATPLOTLIB
def test_a_branch_the_run_left_still_loses_its_pictures(drawn):
    """Resumed from the 900 s restart: 1200 s is drawn again from the new segment, and 1500 s is the old branch's."""

    run = drawn
    add_segment(run, 2, [(1, 1200.0)], resumed_from=(1, 900.0))
    seen: list[str] = []
    draw(run, log=seen.append)
    assert index_times(run) == [900.0, 1200.0]
    assert not (run / "quicklook" / "reflectivity" / "t001500.png").exists()
    assert (run / "quicklook" / "reflectivity" / "t001200.png").is_file()
    assert "t = 1500 s: removed its pictures, from a branch the run has left" in seen
    index = json.loads((run / "quicklook" / "index.json").read_text())
    assert [f["frame"] for f in index["frames"]] == ["segments/001/frames/frame-000000",
                                                     "segments/002/frames/frame-000001"]
    # and once the new branch's frames are deleted too, its pictures stay
    before = pictures(run)
    delete_frames(run)
    draw(run)
    assert pictures(run) == before


def test_left_behind_reads_the_runs_records_not_the_frames_on_disk(tmp_path):
    run = gui_run(tmp_path / "runs", times=(900.0, 1200.0))
    add_segment(run, 2, [(1, 1200.0), (2, 1500.0)], resumed_from=(1, 900.0))
    layout = RD.Renderer(run).layout
    from cm1core.quicklook.frame import frames_of

    lineage = frames_of(layout)
    rows = {"900.0": {"frame": "segments/001/frames/frame-000000", "segment": 1, "time": 900.0},
            "1200.0": {"frame": "segments/001/frames/frame-000001", "segment": 1, "time": 1200.0},
            "1500.0": {"frame": "segments/002/frames/frame-000002", "segment": 2, "time": 1500.0},
            # an old record without its segment: the key names it
            "300.0": {"frame": "segments/003/frames/frame-000009", "time": 300.0}}
    assert RD.left_behind(layout, lineage, rows) == ["1200.0", "300.0"]
    # the newest segment's own frame deleted: still on the run's line, so kept
    shutil.rmtree(run / "segments" / "002" / "frames" / "frame-000002")
    assert RD.left_behind(layout, frames_of(layout), rows) == ["1200.0", "300.0"]
    # no saved frame left at all: nothing is ever taken away
    delete_frames(run)
    assert RD.left_behind(layout, frames_of(layout), rows) == []
    # a folder without run.json (a plain output folder) has no branches
    os.remove(run / "run.json")
    assert RD.left_behind(RD.Renderer(run).layout, [], rows) == []


# ---------------------------------------------------------------- the GUI: Draw the new looks, Draw every picture again

STATIC = Path(__file__).resolve().parents[1] / "src" / "cm1core" / "gui" / "static"
FAVOURITES = ("reflectivity", "spin-near-ground", "updraft-5km", "swath")


@NEEDS_MATPLOTLIB
@pytest.mark.parametrize("how", ["deleted", "trimmed"])
def test_the_gui_offers_no_pass_over_a_run_whose_frames_are_gone_and_refuses_one(tmp_path, capsys, how):
    from cm1core.cli import main
    from cm1core.gui import looks as LK
    from gui_support import start_server, stop_server

    run = gui_run(tmp_path / "runs")
    draw(run, only=FAVOURITES)                      # as a run drawn by an earlier version
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        results = client.get("/api/runs/storm/results?mode=advanced").json
        offered = {row["id"]: row for row in results["more_looks"]}
        assert results["frames_gone"] is False and offered["clouds-from-above"]["action"] == "all"
        if how == "trimmed":
            # the GUI's own Keep pictures only
            assert client.post("/api/runs/storm/delete", {"keep_pictures": True}).json["ok"]
            assert (run / "trimmed.json").is_file()
        else:
            delete_frames(run)
        before, index = pictures(run), (run / "quicklook" / "index.json").read_bytes()
        results = client.get("/api/runs/storm/results?mode=advanced").json
        rows = {row["id"]: row for row in results["more_looks"]}
        assert results["frames_gone"] is True
        # no button draws from the frames: neither Draw the new looks nor Draw it, nor a film made from the data;
        # a run whose frames were deleted by hand can still have its run card, drawn from its receipt
        offered = [row["id"] for row in results["more_looks"] if row["action"]]
        assert offered == ([] if how == "trimmed" else ["run-card"])
        words = LK.load_copy()["why_not"]["frames_gone"]["text"]
        for lid in ("clouds-from-above", "infrared", "side-view", "environment", "swirl", "storm3d"):
            assert rows[lid]["why_not"] == words and rows[lid]["action"] is None, lid
        assert "the new looks are drawn from the saved frames" in words and "kept" in words
        assert rows["storm3d"]["film_too"] is False
        # what the buttons send is refused in plain words, and nothing is started
        for body in ({"look": "all"}, {"look": "all", "again": True}, {"look": "environment"}, {"look": "storm3d"}):
            refused = client.post("/api/runs/storm/look", body)
            assert refused.status == 409 and "deleted" in refused.json["message"], body
            assert "kept" in refused.json["fix"], body
        refused = client.post("/api/runs/storm/radar", {"site": "storm"})
        assert refused.status == 409 and "radar cannot scan it again" in refused.json["message"]
        refused = client.post("/api/runs/storm/film", {"look": "swirl"})
        assert refused.status == 409 and "deleted" in refused.json["message"]
        # nor is Convert: it has nothing left to convert, and says so (before, it ran and wrote nothing)
        assert results["converted_missing"] == 0
        assert results["convert_why_not"] == ("This run's saved frames were deleted, so it has nothing left to "
                                              "convert: CM1's files are written from the saved frames.")
        refused = client.post("/api/runs/storm/convert", {})
        assert refused.status == 409 and refused.json["message"] == results["convert_why_not"]
        assert not (run / "cm1out").exists()
        assert not server.jobs.launched("storm")
        # the commands those buttons run, with and without --force, take nothing away either
        for args in ([], ["--force"]):
            main(["quicklook", os.fspath(run), *args])
        assert "deleted" in capsys.readouterr().err
        assert pictures(run) == before and (run / "quicklook" / "index.json").read_bytes() == index
    finally:
        stop_server(server)


def test_a_run_with_its_frames_or_without_pictures_is_not_one_whose_frames_are_gone(tmp_path):
    from cm1core.gui.files import frames_gone
    from cm1core.gui.runview import convert_refused

    run = gui_run(tmp_path / "runs", times=(900.0,))
    drawn = {"frames": [{"time_s": 900.0}]}
    assert not frames_gone(run, drawn) and not frames_gone(run, None) and not frames_gone(run, {"frames": []})
    assert convert_refused(run, drawn) is None and convert_refused(run) is None
    delete_frames(run)
    assert frames_gone(run, drawn) and not frames_gone(run, {"frames": []})
    # Convert is refused for a run whose pictures outlived its frames, not for one that never drew any
    assert "nothing left to convert" in convert_refused(run, drawn) and convert_refused(run) is None


# ---------------------------------------------------------------- some saved frames deleted, not all

def lose(run: Path, *indices: int) -> None:
    """Delete some of segment 1's saved frames by hand."""

    for index in indices:
        shutil.rmtree(run / "segments" / "001" / "frames" / f"frame-{index:06d}")


def once_rows(run: Path) -> dict:
    return json.loads((run / "quicklook" / ".state" / "state.json").read_text())["once"]


@NEEDS_MATPLOTLIB
def test_force_and_the_radar_refuse_a_run_with_some_frames_deleted_and_delete_nothing(drawn, capsys):
    from cm1core.cli import main

    run = drawn
    before, index = pictures(run), (run / "quicklook" / "index.json").read_bytes()
    lose(run, 1)                                    # the 1200 s frame
    with pytest.raises(RD.QuicklookError) as refused:
        draw(run, force=True)
    assert str(refused.value) == RD.frames_lost_force(1, 3, 2)
    assert "1 of this run's 3 drawn saved times have lost their saved frames" in str(refused.value)
    assert pictures(run) == before and (run / "quicklook" / "index.json").read_bytes() == index
    # Draw every picture again runs this command: refused, nothing deleted
    assert main(["quicklook", os.fspath(run), "--force"]) == 2
    assert "--force would delete them. Nothing was deleted" in capsys.readouterr().err
    assert pictures(run) == before and (run / "quicklook" / "index.json").read_bytes() == index
    # the radar of every saved time would leave the old radar's pictures beside the new: refused, nothing changed
    assert main(["quicklook", os.fspath(run), "--radar-site", "10", "5"]) == 2
    assert RD.radar_lost(1, 3) in capsys.readouterr().err
    assert pictures(run) == before and not (run / "quicklook" / "radar.json").exists()
    # a pass without --force keeps every picture and every time, and says what it keeps
    seen: list[str] = []
    summary = draw(run, log=seen.append)
    assert RD.frames_lost_note(1, 3) in seen and summary.notes == [RD.frames_lost_note(1, 3)]
    assert pictures(run) == before and index_times(run) == [900.0, 1200.0, 1500.0]


@NEEDS_MATPLOTLIB
def test_the_looks_drawn_once_are_kept_when_some_frames_are_gone(drawn, tmp_path, capsys):
    """A copy of a drawn run (the run card is drawn again when its records move) with its first frame deleted (the
    environment's): neither is drawn again from what is left, nor is the storm over time."""

    from cm1core.cli import main

    run = tmp_path / "copy" / "storm"
    shutil.copytree(drawn, run)
    drawn_once = {pid: row for pid, row in once_rows(run).items() if "file" in row}
    assert {"environment", "run-card"} <= set(drawn_once)
    before = pictures(run)
    lose(run, 0)
    draw(run)
    assert pictures(run) == before
    assert {pid: row for pid, row in once_rows(run).items() if "file" in row} == drawn_once
    # asked for by name, each is refused in plain words and its picture kept
    words = {"environment": "is drawn from the run's first saved frame, which was deleted",
             "run-card": "have lost their saved frames: drawn again, it would show fewer of them"}
    for pid in sorted(drawn_once):
        assert main(["quicklook", os.fspath(run), "--look", pid]) == 2, pid
        said = capsys.readouterr().err
        assert words.get(pid, "drawn again, it would show fewer of them") in said and "is kept" in said, pid
    assert pictures(run) == before
    assert {pid: row for pid, row in once_rows(run).items() if "file" in row} == drawn_once


@NEEDS_MATPLOTLIB
def test_a_look_drawn_once_that_a_run_lacks_is_drawn_from_the_frames_that_are_left(drawn):
    run = drawn
    rows = json.loads((run / "quicklook" / ".state" / "state.json").read_text())
    rows["once"].pop("environment")                 # as a run drawn before the environment look existed
    (run / "quicklook" / ".state" / "state.json").write_text(json.dumps(rows))
    shutil.rmtree(run / "quicklook" / "environment")
    before = pictures(run)
    lose(run, 0)
    draw(run)
    after = pictures(run)
    assert {name: digest for name, digest in after.items() if name in before} == before
    assert sorted(set(after) - set(before)) == ["environment/t001200.png", "environment/thumb/t001200.png"]


def test_a_segment_whose_folder_was_deleted_is_still_on_the_runs_line(tmp_path):
    """Only run.json says that a run left a branch: deleting a segment's whole folder does not."""

    from cm1core.quicklook.frame import frames_of

    run = gui_run(tmp_path / "runs", times=(900.0, 1200.0))
    add_segment(run, 2, [(2, 1500.0)], resumed_from=(1, 1200.0))
    rows = {"900.0": {"frame": "segments/001/frames/frame-000000", "segment": 1, "time": 900.0},
            "1200.0": {"frame": "segments/001/frames/frame-000001", "segment": 1, "time": 1200.0},
            "1500.0": {"frame": "segments/002/frames/frame-000002", "segment": 2, "time": 1500.0}}
    layout = RD.Renderer(run).layout
    assert RD.left_behind(layout, frames_of(layout), rows) == [] == RD.lost_times(layout, frames_of(layout), rows)
    # the first segment's folder deleted: its times are lost, not left behind
    moved = tmp_path / "segment-001"
    shutil.move(os.fspath(run / "segments" / "001"), moved)
    layout = RD.Renderer(run).layout
    assert RD.left_behind(layout, frames_of(layout), rows) == []
    assert sorted(RD.lost_times(layout, frames_of(layout), rows)) == ["1200.0", "900.0"]
    # the newest segment's folder deleted instead: the same
    shutil.move(os.fspath(moved), run / "segments" / "001")
    shutil.rmtree(run / "segments" / "002")
    layout = RD.Renderer(run).layout
    assert RD.left_behind(layout, frames_of(layout), rows) == []
    assert RD.lost_times(layout, frames_of(layout), rows) == ["1500.0"]
    from cm1core.cm1run.rundir import RunDirectory

    assert RunDirectory(run).recorded_line() == {2: None, 1: 1200.0}


# ---------------------------------------------------------------- films made from the saved frames

def fake_film(run: Path, look: str, facts: dict | None = None) -> dict[str, bytes]:
    """A film of ``look`` as quicklook.film leaves it: its pictures, its animated PNG and its row in the index."""

    from cm1core.quicklook import film as F

    folder = run / "quicklook" / "films" / look
    (folder / "frames").mkdir(parents=True)
    (folder / "frames" / "00000.png").write_bytes(b"picture-0")
    (folder / "film.png").write_bytes(b"film")
    index = run / "quicklook" / "films" / "index.json"
    films = json.loads(index.read_text())["films"] if index.is_file() else {}
    films[look] = {"look": look, "pictures": 1, "files": {"apng": f"films/{look}/film.png"},
                   "frames": [f"films/{look}/frames/00000.png"], "facts": facts or {}}
    index.write_text(json.dumps({"schema": F.INDEX_SCHEMA, "films": films}))
    return {p.relative_to(run).as_posix(): p.read_bytes() for p in sorted((run / "quicklook" / "films").rglob("*"))
            if p.is_file()}


def films_now(run: Path) -> dict[str, bytes]:
    return {p.relative_to(run).as_posix(): p.read_bytes() for p in sorted((run / "quicklook" / "films").rglob("*"))
            if p.is_file()}


@NEEDS_MATPLOTLIB
def test_a_film_made_from_the_saved_frames_is_not_made_again_over_one_made_from_more(drawn, capsys):
    from cm1core.cli import main
    from cm1core.quicklook import film as F

    run = drawn
    fake_film(run, "swirl")
    fake_film(run, "parcels")
    kept = fake_film(run, "storm3d", {"frame": "segments/001/frames/frame-000001"})
    lose(run, 1)
    for look in ("swirl", "parcels", "storm3d"):
        job = F.FilmJob(run, look, (), 12.0, 8, run / "quicklook")
        with pytest.raises(F.FilmKept) as refused:
            F.make(job)
        assert "Nothing was changed; the film it has is kept." in str(refused.value), look
        assert main(["quicklook", os.fspath(run), "--film", look]) == 2, look
        assert "the film it has is kept" in capsys.readouterr().err, look
    # every film as it was, its progress untouched (no "failed" over a film that is there)
    assert films_now(run) == kept
    assert "lost their saved frames" in F.remake_refused(F.FilmJob(run, "swirl", (), 12.0, 8, run / "quicklook"))
    # the 3-D film's own saved time still there: it can be made again (at any time)
    index = json.loads((run / "quicklook" / "films" / "index.json").read_text())
    index["films"]["storm3d"]["facts"]["frame"] = "segments/001/frames/frame-000002"
    (run / "quicklook" / "films" / "index.json").write_text(json.dumps(index))
    assert F.remake_refused(F.FilmJob(run, "storm3d", (), 12.0, 8, run / "quicklook")) is None


@NEEDS_MATPLOTLIB
def test_a_film_not_made_again_keeps_the_pictures_it_had(drawn):
    from cm1core.quicklook import film as F

    run = drawn
    kept = fake_film(run, "swirl")
    job = F.FilmJob(run, "swirl", (), 18.0, 8, run / "quicklook", should_stop=lambda: True)
    with pytest.raises(F.FilmError, match="stopped before its first picture"):
        F.make(job)
    assert {name: data for name, data in films_now(run).items() if "progress" not in name} == kept
    assert not (run / "quicklook" / "films" / "swirl" / ".frames-before").exists()


# ---------------------------------------------------------------- comparisons

@NEEDS_MATPLOTLIB
def test_a_comparison_is_not_drawn_again_from_fewer_saved_frames(tmp_path):
    from cm1core.quicklook import compare as CO
    from quicklook_support import run_directory

    a = run_directory(tmp_path, name="a")
    b = run_directory(tmp_path, name="b")
    draw(b)
    folder = tmp_path / "cmp"
    CO.make([a, b], out=folder, only=("series",))
    before = {p.name: p.read_bytes() for p in folder.glob("*.png")}
    shutil.rmtree(b / "frames" / "frame-000001")
    with pytest.raises(CO.CompareError) as refused:
        CO.make([a, b], out=folder, only=("series",))
    assert "B (b) has lost 1 of its saved frames since" in str(refused.value)
    assert "nothing was drawn and the pictures it has are kept" in str(refused.value)
    assert {p.name: p.read_bytes() for p in folder.glob("*.png")} == before
    # into a new folder it is drawn from what is left, and drawn again there it is the same
    record = CO.make([a, b], out=tmp_path / "cmp2", only=("series",))
    assert record["runs"][1]["saved_times"] == 2
    CO.make([a, b], out=tmp_path / "cmp2", only=("series",))


# ---------------------------------------------------------------- the GUI

@NEEDS_MATPLOTLIB
def test_the_gui_offers_no_redraw_that_would_delete_pictures_of_a_run_with_some_frames_gone(tmp_path):
    from cm1core.cm1out import convert
    from cm1core.gui import looks as LK
    from gui_support import start_server, stop_server

    run = gui_run(tmp_path / "runs")
    draw(run)
    convert(run)
    fake_film(run, "swirl")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        results = client.get("/api/runs/storm/results?mode=advanced").json
        assert results["frames_missing"] == 0 and results["converted_missing"] == 0
        assert results["again_why_not"] is None and results["convert_why_not"] is None
        assert results["radar"]["why_not"] is None
        assert {row["id"]: row for row in results["more_looks"]}["swirl"]["film_why_not"] is None
        lose(run, 1)
        before = pictures(run)
        converted = sorted(p.name for p in (run / "cm1out").iterdir())
        results = client.get("/api/runs/storm/results?mode=advanced").json
        assert results["frames_gone"] is False
        assert results["frames_missing"] == 1 and results["converted_missing"] == 1
        assert results["again_why_not"] == LK.again_refused(run, json.loads((run / "quicklook" / "index.json")
                                                                            .read_text()))
        assert "1 of this run's 3 drawn saved times have lost their saved frames" in results["again_why_not"]
        assert "would delete them" in results["convert_why_not"]
        assert "radar cannot scan those times again" in results["radar"]["why_not"]
        rows = {row["id"]: row for row in results["more_looks"]}
        assert "The film it has is kept." in rows["swirl"]["film_why_not"]
        # what the buttons send is refused in plain words, and nothing is started
        for path, body, words in (("look", {"look": "all", "again": True}, "would delete them"),
                                  ("radar", {"site": "storm"}, "radar cannot scan those times again"),
                                  ("film", {"look": "swirl"}, "The film it has is kept."),
                                  ("convert", {}, "would delete them")):
            refused = client.post(f"/api/runs/storm/{path}", body)
            assert refused.status == 409 and words in refused.json["message"], (path, refused.json)
        assert not server.jobs.launched("storm")
        assert pictures(run) == before and sorted(p.name for p in (run / "cm1out").iterdir()) == converted
    finally:
        stop_server(server)


def test_a_run_whose_pictures_name_their_frames_by_folder_alone_has_lost_none(tmp_path):
    """Pictures drawn before the first release named each saved frame by its folder alone (frame-000001): such a
    run with every frame there is offered everything, and one whose frame was deleted says so."""

    from cm1core.gui.files import converted_missing, frames_missing

    run = gui_run(tmp_path / "runs")
    index = {"frames": [{"time_s": t, "segment": 1, "frame": f"frame-{n:06d}"}
                        for n, t in enumerate((900.0, 1200.0, 1500.0))]}
    assert frames_missing(run, index) == 0
    state = {"frames": {repr(t): {"frame": f"segments/001/frames/frame-{n:06d}", "time": t}
                        for n, t in enumerate((900.0, 1200.0, 1500.0))}}
    (run / "cm1out").mkdir()
    (run / "cm1out" / ".cm1out-state.json").write_text(json.dumps(state))
    assert converted_missing(run) == 0
    lose(run, 1)
    assert frames_missing(run, index) == 1 and converted_missing(run) == 1


def test_results_disable_draw_again_convert_and_the_radar_with_the_servers_reason():
    results = (STATIC / "js" / "run" / "results.js").read_text(encoding="utf-8")
    assert "const gone = Boolean(data.trimmed) || Boolean(data.frames_gone) || Boolean(data.again_why_not);" in results
    assert "again.button.disabled = running || Boolean(none) || gone || Boolean(data.drawing_looks);" in results
    assert "gone ? data.again_why_not ||" in results
    assert "Boolean(data.convert_why_not);" in results and 'none ? "no saved frame to convert" : data.convert_why_not' \
        in results
    assert "data.frames_gone, (data.looks_about_s || {}).again, data.again_why_not," in results
    parts = (STATIC / "js" / "run" / "parts.js").read_text(encoding="utf-8")
    assert "(!more || more.available) && !data.frames_gone) {" in parts
    assert "disabled: live || Boolean(data.drawing_looks) || Boolean(r.why_not) || null, title: r.why_not || null" \
        in parts
    assert "bar.append(h(\"p\", { class: \"note\" }, more.film_why_not));" in parts


# ---------------------------------------------------------------- a resumed run whose newest segment was deleted

@NEEDS_MATPLOTLIB
def test_a_resumed_run_whose_newest_segment_folder_was_deleted_keeps_its_pictures(tmp_path, capsys):
    """Resumed from the 1200 s restart: segment 2 has 1500 s again.  With segment 2's whole folder deleted, the
    1500 s frame segment 1 still holds is of the branch the run left (run.json says so): the run's 1500 s is lost,
    not drawn again from that branch."""

    from cm1core.cli import main
    from cm1core.cm1run.rundir import RunDirectory
    from cm1core.quicklook.frame import frames_of

    run = gui_run(tmp_path / "runs")
    add_segment(run, 2, [(2, 1500.0)], resumed_from=(1, 1200.0))
    draw(run)
    state_path = run / "quicklook" / ".state" / "state.json"
    rows = json.loads(state_path.read_text())["frames"]
    assert rows["1500.0"]["frame"] == "segments/002/frames/frame-000002"
    before = pictures(run)
    shutil.rmtree(run / "segments" / "002")
    assert [info.model_time_s for info in RunDirectory(run).frames()] == [900.0, 1200.0]
    layout = RD.Renderer(run).layout
    assert [t for t, *_ in frames_of(layout)] == [900.0, 1200.0]
    assert RD.lost_times(layout, frames_of(layout), rows) == ["1500.0"]
    draw(run)
    assert pictures(run) == before
    assert json.loads(state_path.read_text())["frames"]["1500.0"]["frame"] == "segments/002/frames/frame-000002"
    assert main(["quicklook", os.fspath(run), "--force"]) == 2
    assert "1 of this run's 3 drawn saved times have lost their saved frames" in capsys.readouterr().err
    assert pictures(run) == before



# ---------------------------------------------------------------- a 3-D still drawn at a time the run left

@NEEDS_MATPLOTLIB
def test_a_3d_still_drawn_again_at_a_time_the_run_left_is_kept(tmp_path, monkeypatch):
    """Resumed from the 900 s restart, segment 2 holds 1200 s again.  The still of 1200 s drawn from segment 1 is of
    the branch the run left; drawn again from segment 2 it is written to the same file, which the clearing of the
    branch left then deleted, leaving the record naming a file that is not there."""

    from cm1core.quicklook import storm3d as D

    run = gui_run(tmp_path / "runs")
    add_segment(run, 2, [(1, 1200.0)], resumed_from=(1, 900.0))
    out = run / "quicklook"
    for name in ("storm3d/t001200.png", "storm3d/thumb/t001200.png"):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(b"the branch left")
    D._write_drawn(out, {"1200.0": {"frame": "segments/001/frames/frame-000001", "segment": 1, "time": 1200.0,
                                    "file": "storm3d/t001200.png", "thumb": "storm3d/thumb/t001200.png",
                                    "facts": {}}})

    def still(_looks, file, **_options):
        for path in (file, file.parent / "thumb" / file.name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"the run's own")
        return {}

    monkeypatch.setattr(D, "still", still)
    record = D.draw_still(run, time_s=1200.0)
    assert record["frame"] == "segments/002/frames/frame-000001"
    assert (out / "storm3d" / "t001200.png").read_bytes() == b"the run's own"
    assert (out / "storm3d" / "thumb" / "t001200.png").read_bytes() == b"the run's own"
    assert D.read_drawn(out)["1200.0"]["segment"] == 2


# ---------------------------------------------------------------- the pass's own records lost

def tear(path: Path, how: str) -> None:
    """The pass's record deleted, emptied or cut short (a disk that filled, a copy stopped part way)."""

    if how == "gone":
        path.unlink()
    elif how == "empty":
        path.write_text("")
    else:
        path.write_text(path.read_text()[:400])


@NEEDS_MATPLOTLIB
@pytest.mark.parametrize("how", ["gone", "empty", "torn", "index-too"])
def test_a_lost_record_keeps_every_picture_of_a_run_with_some_frames_deleted(drawn, capsys, how):
    """Without its record, a pass took the times whose frames are gone for never drawn: the looks drawn once per
    run and the time series were drawn again from the frames left, wrote an index without those times, and then --force
    deleted their pictures.  The record is rebuilt from the index and the pictures instead."""

    from cm1core.cli import main

    run = drawn
    folder = run / "quicklook"
    before = pictures(run)
    lose(run, 1)
    if how == "index-too":
        shutil.rmtree(folder / ".state")
        (folder / "index.json").unlink()
    else:
        tear(folder / ".state" / "state.json", how)
    summary = draw(run)
    assert RD.REBUILT in summary.notes and RD.frames_lost_note(1, 3) in summary.notes
    # nothing drawn again over what was there, the run-once looks and the time series included; every time listed
    assert pictures(run) == before and index_times(run) == [900.0, 1200.0, 1500.0]
    index = json.loads((folder / "index.json").read_text())
    assert [len(p["frames"]) for p in index["products"] if p["id"] == "reflectivity"] == [3]
    assert {p["id"] for p in index["products"] if p.get("once") and p["frames"]} >= {"environment", "run-card"}
    # and so Draw every picture again (--force) and the radar are refused, and delete nothing
    assert main(["quicklook", os.fspath(run), "--force"]) == 2
    assert "1 of this run's 3 drawn saved times have lost their saved frames" in capsys.readouterr().err
    assert main(["quicklook", os.fspath(run), "--radar-site", "10", "5"]) == 2
    assert RD.radar_lost(1, 3) in capsys.readouterr().err
    assert pictures(run) == before and index_times(run) == [900.0, 1200.0, 1500.0]


@NEEDS_MATPLOTLIB
def test_force_counts_pictures_whose_record_was_lost_before_any_pass(drawn, tmp_path):
    run = drawn
    whole = tmp_path / "whole" / "storm"
    shutil.copytree(run, whole)
    before = pictures(run)
    lose(run, 1)
    shutil.rmtree(run / "quicklook" / ".state")
    (run / "quicklook" / "index.json").unlink()
    with pytest.raises(RD.QuicklookError, match="1 of this run's 3 drawn saved times have lost their saved frames"):
        draw(run, force=True)
    assert pictures(run) == before
    # a run with every frame there draws every picture again, as before
    shutil.rmtree(whole / "quicklook" / ".state")
    (whole / "quicklook" / "index.json").unlink()
    draw(whole, force=True)
    assert set(pictures(whole)) == set(before) and index_times(whole) == [900.0, 1200.0, 1500.0]


@NEEDS_MATPLOTLIB
def test_the_gui_counts_the_pictures_of_lost_frames_when_the_records_are_lost(tmp_path):
    from gui_support import start_server, stop_server

    run = gui_run(tmp_path / "runs")
    draw(run)
    lose(run, 1)
    shutil.rmtree(run / "quicklook" / ".state")
    (run / "quicklook" / "index.json").unlink()
    before = pictures(run)
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        results = client.get("/api/runs/storm/results?mode=advanced").json
        assert results["frames_missing"] == 1 and results["frames_gone"] is False
        assert "1 of this run's 3 drawn saved times have lost their saved frames" in results["again_why_not"]
        assert "radar cannot scan those times again" in results["radar"]["why_not"]
        refused = client.post("/api/runs/storm/look", {"look": "all", "again": True})
        assert refused.status == 409 and "would delete them" in refused.json["message"]
        assert not server.jobs.launched("storm") and pictures(run) == before
    finally:
        stop_server(server)



# ---------------------------------------------------------------- however the run's folder is named

@NEEDS_MATPLOTLIB
def test_a_run_drawn_by_one_path_and_passed_by_another_is_not_drawn_again(drawn, monkeypatch):
    """A frame's key was its folder under the run when the run was named by an absolute path (as the GUI names it),
    but the folder alone when named by a relative one: a pass named the other way had every picture drawn again, and the
    radar looks of a run with some frames deleted came out worse."""

    from cm1core.quicklook.frame import frame_key, frames_of

    run = drawn
    monkeypatch.chdir(run.parent)
    keys = [key for _t, key, _p, _s in frames_of(RD.Renderer(Path(run.name)).layout)]
    assert keys == [key for _t, key, _p, _s in frames_of(RD.Renderer(run).layout)]
    assert keys == [f"segments/001/frames/frame-{n:06d}" for n in range(3)]
    assert frame_key(Path(run.name) / "x", run.name) == "x"
    assert frame_key("/elsewhere/frame-000001", run) == "frame-000001"
    before = pictures(run)
    lose(run, 1)
    assert draw(Path(run.name)).frames_drawn == 0 and pictures(run) == before
    # a record of an older pass named by a relative path (each frame by its folder alone) is drawn from the same
    state_path = run / "quicklook" / ".state" / "state.json"
    state = json.loads(state_path.read_text())
    for row in state["frames"].values():
        row["frame"] = Path(row["frame"]).name
    state_path.write_text(json.dumps(state))
    assert draw(run).frames_drawn == 0 and pictures(run) == before
    with pytest.raises(RD.QuicklookError, match="1 of this run's 3 drawn saved times"):
        draw(run, force=True)



# ---------------------------------------------------------------- the radar moved, stopped part way

@NEEDS_MATPLOTLIB
def test_a_radar_move_stopped_part_way_keeps_its_pictures_listed(drawn):
    """The radar's pass marks every radar picture to be drawn again; stopped before drawing them, and with a frame
    deleted since, the times whose frames are gone keep their radar pictures in the index."""

    def listed(run: Path) -> dict:
        index = json.loads((run / "quicklook" / "index.json").read_text())
        return {p["id"]: {f["time_s"]: f for f in p["frames"]} for p in index["products"] if p["id"] in RD.RADAR_IDS}

    run = drawn
    listed_before = listed(run)
    renderer = RD.Renderer(run, only=RD.RADAR_IDS)
    assert renderer.forget(RD.RADAR_IDS) == 6           # what the radar's pass does first; then it was stopped
    before = pictures(run)
    lose(run, 1)
    draw(run)
    listed_after = listed(run)
    assert {pid: sorted(rows) for pid, rows in listed_after.items()} == {pid: [900.0, 1200.0, 1500.0]
                                                                        for pid in RD.RADAR_IDS}
    # listed as they were, with the numbers each showed
    assert all(listed_after[pid][1200.0] == listed_before[pid][1200.0] for pid in RD.RADAR_IDS)
    after = pictures(run)
    assert all(after[name] == digest for name, digest in before.items() if "t001200" in name)
    assert set(after) == set(before)



# ---------------------------------------------------------------- a comparison whose run lost its index

@NEEDS_MATPLOTLIB
@pytest.mark.parametrize("record", ["new", "old"])
def test_a_comparison_is_not_drawn_again_from_fewer_frames_when_the_index_is_lost(tmp_path, record):
    from cm1core.quicklook import compare as CO
    from quicklook_support import run_directory

    a = run_directory(tmp_path, name="a")
    b = run_directory(tmp_path, name="b")
    draw(a)
    folder = tmp_path / "cmp"
    made = CO.make([a, b], out=folder, only=("series",))
    assert [f["time"] for f in made["runs"][0]["frames"]] == [900.0, 1200.0, 1500.0]
    if record == "old":
        # a comparison drawn before its record named the frames
        doc = json.loads((folder / "compare.json").read_text())
        for row in doc["runs"]:
            row.pop("frames")
        (folder / "compare.json").write_text(json.dumps(doc))
    before = {p.name: p.read_bytes() for p in folder.glob("*.png")}
    shutil.rmtree(a / "frames" / "frame-000001")
    (a / "quicklook" / "index.json").unlink()
    shutil.rmtree(a / "quicklook" / ".state")
    with pytest.raises(CO.CompareError, match=r"A \(a\) has lost 1 of its saved frames since"):
        CO.make([a, b], out=folder, only=("series",))
    assert {p.name: p.read_bytes() for p in folder.glob("*.png")} == before



# ---------------------------------------------------------------- a film remake stopped part way

@NEEDS_MATPLOTLIB
def test_a_film_remake_stopped_part_way_keeps_the_films_pictures(drawn):
    """A swirl film made again was stopped (killed) after it set its pictures aside; with a frame deleted since,
    the next remake is refused, and the film's own pictures are put back, not deleted."""

    from cm1core.quicklook import film as F

    run = drawn
    kept = fake_film(run, "swirl")
    folder = run / "quicklook" / "films" / "swirl"
    os.replace(folder / "frames", folder / ".frames-before")
    (folder / "frames").mkdir()
    (folder / "frames" / "00000.png").write_bytes(b"half a remake")
    lose(run, 1)
    with pytest.raises(F.FilmKept):
        F.make(F.FilmJob(run, "swirl", (), 18.0, 8, run / "quicklook"))
    assert films_now(run) == kept and not (folder / ".frames-before").exists()

