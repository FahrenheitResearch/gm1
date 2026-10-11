"""The GUI's picture endpoints: slices, sections, the column skew-T and the field list are ``gpuwm-cm1 quicklook``.

Each POST runs one ``gpuwm-cm1 quicklook`` command as a host-only call (no GPU, nice 19), returns that command and
the picture's URL, and appends the command to the run's ``commands.log``.  With the fake runner the pictures are
placeholders; with the real runner they are the real quick-look renderer on frames of CM1's own storm.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from cm1core.gui import commands as C
from cm1core.gui.jobs import Jobs, fake_runner, real_runner

from cm1import_support import namelist_tool  # noqa: E402
from gui_support import start_server, stop_server  # noqa: E402
from quicklook_support import run_directory  # noqa: E402


@pytest.fixture()
def storm_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GM1_CM1_R21_NAMELIST", os.fspath(namelist_tool()))
    root = tmp_path / "runs"
    root.mkdir()
    run_directory(root, times=(900.0, 1500.0), name="storm")
    return root


def _png_size(data: bytes) -> tuple[int, int]:
    import struct

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_the_picture_commands_are_quicklook_lines():
    run = Path("/runs/storm")
    assert C.display(C.slice_args(run, "w", height_m=5000.0, line_km=(-8.0, -2.5, 8, 3), time_s=1500.0)) == \
        "gpuwm-cm1 quicklook /runs/storm --slice w --height 5000 --line -8 -2.5 8 3 --time 1500 --json"
    assert C.display(C.slice_args(run, "qr", level=3)) == "gpuwm-cm1 quicklook /runs/storm --slice qr --level 3 --json"
    assert C.display(C.section_args(run, "w", (-10, 0), (10.5, 2), top_m=12000)) == \
        "gpuwm-cm1 quicklook /runs/storm --section w --from -10 0 --to 10.5 2 --top 12000 --json"
    assert C.display(C.column_args(run)) == "gpuwm-cm1 quicklook /runs/storm --column --json"
    assert C.display(C.column_args(run, (1.5, -2))) == "gpuwm-cm1 quicklook /runs/storm --column --at 1.5 -2 --json"
    assert C.display(C.fields_args(run, 900)) == "gpuwm-cm1 quicklook /runs/storm --fields --time 900 --json"
    actions = {row.action: row for row in C.API}
    assert {"fields", "slice", "section", "column"} <= set(actions)
    assert all(actions[a].kind == "host" and actions[a].command.startswith("gpuwm-cm1 quicklook RUNDIR")
               for a in ("fields", "slice", "section", "column"))


def test_picture_commands_never_write_a_number_argparse_takes_for_an_option():
    """A click just west of x = 0 is -1e-05 km: in exponent form argparse reads it as an option and refuses."""

    from cm1core.cli import build_parser

    run = Path("/runs/storm")
    parser = build_parser()
    for args in (C.section_args(run, "w", (-1e-05, -2.5), (0.1 + 0.2, 3.0)),
                 C.column_args(run, (-4e-07, -12.3456789)),
                 C.slice_args(run, "u", height_m=12.5, line_km=(-1e-05, 1e-05, -3.0, 4.25), time_s=0.00001)):
        assert "e-" not in " ".join(args) and "e+" not in " ".join(args), args
        parsed = parser.parse_args(args)
        assert parsed.target == run
    parsed = parser.parse_args(C.section_args(run, "w", (-1e-05, -2.5), (0.3, 3.0)))
    assert parsed.point_a == [-1e-05, -2.5] and parsed.point_b == [0.3, 3.0]


def test_a_follow_is_never_a_host_call(tmp_path):
    jobs = Jobs(fake_runner(), tmp_path)
    try:
        with pytest.raises(ValueError, match="sidecar"):
            jobs.host(["quicklook", str(tmp_path), "--follow"])
    finally:
        jobs.close()


def test_the_fake_runner_answers_every_picture_endpoint(storm_root, tmp_path):
    server, client = start_server(storm_root, tmp_path / "config")
    try:
        for action, body in (("slice", {"field": "w", "height_m": 5000}),
                             ("section", {"field": "w", "from_km": [-8, 0], "to_km": [8, 0]}),
                             ("column", {}), ("fields", {"time_s": 900})):
            answer = client.post(f"/api/runs/storm/{action}", body)
            assert answer.status == 200 and answer.json["ok"], (action, answer.json)
            assert answer.json["command"].startswith("gpuwm-cm1 quicklook ")
            if action != "fields":
                picture = client.get(answer.json["picture_url"])
                assert picture.status == 200 and picture.headers["content-type"] == "image/png"
        log = (storm_root / "storm" / "commands.log").read_text()
        assert log.count("gpuwm-cm1 quicklook") == 4
        for bad in ({"field": "w; rm -rf /"}, {"field": "w", "height_m": "high"}, {"field": "w", "level": 0},
                    {"field": "w", "line_km": [1, 2]}):
            answer = client.post("/api/runs/storm/slice", bad)
            assert answer.status == 400 and answer.json["message"], bad
        answer = client.post("/api/runs/storm/section", {"field": "w", "from_km": [0, 0]})
        assert answer.status == 400 and "to_km" in answer.json["message"]
    finally:
        stop_server(server)


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="matplotlib is not installed (it comes with gpuwm)")
def test_the_real_runner_draws_the_slice_and_results_link_every_picture(storm_root, tmp_path):
    from cm1core.quicklook.render import render

    render(storm_root / "storm")
    server, client = start_server(storm_root, tmp_path / "config", runner=real_runner())
    try:
        answer = client.post("/api/runs/storm/slice", {"field": "w", "height_m": 5000, "line_km": [-8, -2, 8, 3]})
        assert answer.status == 200 and answer.json["ok"], answer.json
        assert answer.json["command"] == C.display(C.slice_args(storm_root / "storm", "w", height_m=5000.0,
                                                                line_km=(-8, -2, 8, 3)))
        record = answer.json["result"]
        assert record["level"] == 11 and record["time_s"] == 1500.0 and record["worker"]["nice"] == 19
        picture = client.get(answer.json["picture_url"])
        assert picture.status == 200 and _png_size(picture.raw) == (1920, 1080)
        column = client.post("/api/runs/storm/column", {"at_km": [0, 0], "time_s": 900})
        assert column.json["ok"] and column.json["result"]["time_s"] == 900.0
        missing = client.post("/api/runs/storm/slice", {"field": "nosuch"})
        assert missing.status == 200 and not missing.json["ok"] and "not a cm1out field" in missing.json["fix"]
        results = client.get("/api/runs/storm/results").json
        looks = {look["id"]: look for look in results["looks"]}
        assert list(looks)[:4] == ["reflectivity", "spin-near-ground", "updraft-5km", "swath"]
        frame = looks["swath"]["frames"][-1]
        assert client.get(frame["url"]).status == 200 and client.get(frame["thumb_url"]).status == 200
        assert _png_size(client.get(frame["thumb_url"]).raw) == (480, 270)
        assert results["series"]["file"] == "series.json"
        assert client.get(results["series"]["url"]).json["schema"] == "gm1.quicklook-series/v1"
        assert client.get(results["series"]["picture_url"]).status == 200
        assert results["quicklook_run"]["microphysics"] == "Morrison with hail"
    finally:
        stop_server(server)
