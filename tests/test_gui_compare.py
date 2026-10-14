"""Results: Compare with other runs.

``POST /api/runs/RUN/compare {with: [...], time_s?, box_km?}`` launches ``gpuwm-cm1 quicklook RUNDIR --compare OTHER
...`` on the CPU (nice 19, no GPU), with the runs picked from the Runs list; ``GET /api/runs/RUN/compare`` lists
the comparisons it has made, newest first, with each picture's address.  The command is in the API table and on the commands page,
"Show command" gets it with ``dry_run``, a bad request is refused in words, and with the real runner the pictures
are the real ones (:mod:`cm1core.quicklook.compare`).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re

import pytest

from cm1core.gui import commands as C
from cm1core.gui.jobs import real_runner

from cm1import_support import namelist_tool  # noqa: E402
from gui_support import start_server, stop_server, wait_for  # noqa: E402
from quicklook_support import run_directory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "cm1core" / "gui" / "static"


@pytest.fixture()
def storm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))
    root = tmp_path / "runs"
    root.mkdir()
    run_directory(root, times=(900.0, 1500.0), name="storm")
    run_directory(root, times=(900.0, 1200.0, 1500.0), name="other")
    run_directory(root, times=(1500.0,), name="third")
    return root


def test_the_compare_command_is_a_quicklook_line_in_the_api_table():
    run = Path("/runs/storm")
    others = [Path("/runs/a"), Path("/runs/b")]
    assert C.display(C.compare_args(run, others)) == \
        "gpuwm-cm1 quicklook /runs/storm --compare /runs/a /runs/b --json"
    assert C.display(C.compare_args(run, others[:1], time_s=4800.0, box_km=30)) == \
        "gpuwm-cm1 quicklook /runs/storm --compare /runs/a --time 4800 --box 30 --json"
    endpoint = C.endpoint_for(["runs", "x", "compare"])
    assert endpoint is not None and endpoint.kind == "launch" and endpoint.action == "compare"
    assert endpoint.command.startswith("gpuwm-cm1 quicklook RUNDIR --compare OTHER")
    from cm1core.cli import build_parser

    parsed = build_parser().parse_args(C.compare_args(run, others, time_s=4800.0, box_km=30))
    assert parsed.compare == others and parsed.time == 4800.0 and parsed.box == 30.0
    page = (ROOT / "docs" / "COMMANDS.md").read_text(encoding="utf-8")
    assert "| Results: Compare with other runs | `gpuwm-cm1 quicklook RUNDIR --compare OTHER" in page


def test_show_command_and_the_refusals(storm_root, tmp_path):
    server, client = start_server(storm_root, tmp_path / "config")
    try:
        dry = client.post("/api/runs/storm/compare", {"with": ["other", "third"], "time_s": 1200, "dry_run": True})
        assert dry.status == 200 and dry.json["dry_run"] is True, dry.json
        assert dry.json["command"] == C.display(C.compare_args(storm_root / "storm",
                                                               [storm_root / "other", storm_root / "third"],
                                                               time_s=1200.0))
        assert not (storm_root / "storm" / "quicklook" / "compare").exists()       # nothing ran
        for body, status, words in (({}, 400, "with must list"), ({"with": []}, 400, "with must list"),
                                    ({"with": "other"}, 400, "with must list"),
                                    ({"with": ["storm"]}, 400, "this run is always the first"),
                                    ({"with": ["other", "other"]}, 400, "picked once"),
                                    ({"with": ["a", "b", "c", "d", "e", "f"]}, 400, "At most 6 runs"),
                                    ({"with": ["nosuch"]}, 404, "No run called"),
                                    ({"with": ["other"], "time_s": "late"}, 400, "time_s"),
                                    ({"with": ["other"], "box_km": 0.1}, 400, "box_km")):
            answer = client.post("/api/runs/storm/compare", body)
            assert answer.status == status and words in answer.json["message"], (body, answer.json)
        listed = client.get("/api/runs/storm/compare")
        assert listed.status == 200 and listed.json == {"comparisons": [], "drawing": False, "max_runs": 6}
        # the command runs in the background; when it stops, the listing says why (here the fake runner finds no
        # GUI run folder's frames in these plain output folders)
        launched = client.post("/api/runs/storm/compare", {"with": ["other"]})
        assert launched.status == 200 and launched.json["ok"] and launched.json["pid"] > 0, launched.json

        def failed():
            rows = client.get("/api/runs/storm/compare").json["comparisons"]
            return [r for r in rows if r["state"] == "failed"]

        rows = wait_for(failed, timeout=60.0, every=0.2)
        assert "nothing of its storm to compare" in rows[0]["progress"]["message"]
        assert "pictures" not in rows[0]
    finally:
        stop_server(server)


def test_the_runs_that_can_only_be_refused_are_named_before_draw(tmp_path, monkeypatch):
    from quicklook_support import NAMELIST_24

    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))
    root = tmp_path / "runs"
    root.mkdir()
    run_directory(root, times=(1500.0,), name="storm")
    run_directory(root, times=(1500.0,), name="other")
    swirl, count = re.subn(r"(?im)^(\s*iinit\s*=\s*)[^,\n]+", r"\g<1>7", NAMELIST_24.read_text(encoding="utf-8"))
    assert count == 1
    (tmp_path / "swirl.namelist.input").write_text(swirl, encoding="utf-8")
    run_directory(root, times=(1500.0,), name="swirl", namelist=tmp_path / "swirl.namelist.input")
    bare = root / "bare"
    bare.mkdir()
    (bare / "run.json").write_text("{}\n", encoding="utf-8")       # a run with neither frames nor pictures
    server, client = start_server(root, tmp_path / "config")
    try:
        assert "cannot" not in client.get("/api/runs/storm/compare").json        # asked for, once, by the list
        cannot = client.get("/api/runs/storm/compare?choices=1").json["cannot"]
        assert set(cannot) == {"swirl", "bare"}, cannot
        assert "hurricane" in cannot["swirl"] and "nothing of its storm to compare" in cannot["bare"]
        back = client.get("/api/runs/swirl/compare?choices=1").json["cannot"]
        assert set(back) == {"storm", "other", "bare"} and "thunderstorm" in back["storm"], back
    finally:
        stop_server(server)
    panel = (STATIC / "js" / "run" / "compare.js").read_text(encoding="utf-8")
    assert "/compare?choices=1`" in panel and "blocked.has(o.value)" in panel


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="matplotlib is not installed (it comes with gpuwm)")
def test_the_real_runner_draws_the_comparison_and_lists_its_pictures(storm_root, tmp_path):
    server, client = start_server(storm_root, tmp_path / "config", runner=real_runner())
    try:
        answer = client.post("/api/runs/storm/compare", {"with": ["other"]})
        assert answer.status == 200 and answer.json["ok"], answer.json
        assert answer.json["command"] == C.display(C.compare_args(storm_root / "storm", [storm_root / "other"]))
        assert answer.json["about_s"] >= 1 and "processor" in answer.json["message"]

        def finished():
            rows = client.get("/api/runs/storm/compare").json["comparisons"]
            return rows if rows and rows[0]["state"] in ("done", "failed") else None

        rows = wait_for(finished, timeout=240.0, every=0.5)
        row = rows[0]
        assert row["state"] == "done", row
        assert [r["letter"] for r in row["runs"]] == ["A", "B"] and row["time_s"] == 1500.0
        keys = [p["key"] for p in row["pictures"]]
        assert keys == ["storm-reflectivity", "storm-surface", "series", "soundings"]
        for picture in row["pictures"]:
            got = client.get(picture["url"])
            assert got.status == 200 and got.headers["content-type"] == "image/png", picture
            assert client.get(picture["thumb_url"]).status == 200
        log = (storm_root / "storm" / "commands.log").read_text(encoding="utf-8")
        assert "nice -n 19 env CUDA_VISIBLE_DEVICES= gpuwm-cm1 quicklook" in log and "--compare" in log
        # a second comparison with another set gets its own folder; the newest is listed first
        wait_for(lambda: not client.get("/api/runs/storm/compare").json["drawing"], timeout=60.0, every=0.2)
        again = client.post("/api/runs/storm/compare", {"with": ["other", "third"], "time_s": 900})
        assert again.status == 200 and again.json["ok"], again.json
        wait_for(lambda: len([r for r in client.get("/api/runs/storm/compare").json["comparisons"]
                              if r["state"] == "done"]) == 2, timeout=240.0, every=0.5)
        rows = client.get("/api/runs/storm/compare").json["comparisons"]
        assert [len(r["runs"]) for r in rows] == [3, 2] and rows[0]["time_s"] == 900.0
    finally:
        stop_server(server)


def test_the_results_screen_offers_the_comparison_in_both_views():
    results = (STATIC / "js" / "run" / "results.js").read_text(encoding="utf-8")
    panel = (STATIC / "js" / "run" / "compare.js").read_text(encoding="utf-8")
    assert results.count("comparePanel(id)") == 2
    assert 'api.get("/api/runs")' in panel and "`${api.runPath(id)}/compare`" in panel
    assert 'actionButton(id, "compare", "Draw the comparison"' in panel
    assert '<link rel="stylesheet" href="/static/css/compare.css">' in (STATIC / "index.html").read_text()
    from cm1core.gui.copy_lint import lint_text

    words = [w for w in re.findall(r'"([^"]{12,})"', panel) if " " in w]
    assert words and [(w, p) for w in words for p in lint_text(w)] == []
