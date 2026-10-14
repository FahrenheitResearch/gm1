"""The Runs screen's Open CM1 output: CM1's own netCDF output opened for pictures, with no run controls.

A folder of CM1 history files under the runs folder, or one the Runs screen made for files kept elsewhere
(``cm1-output.json``), is listed view-only.  Its pictures are ``gpuwm-cm1 quicklook`` of the files where they are,
drawn into the folder's own ``quicklook/``; nothing is run and the files are never changed or deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from cm1core.gui import commands as C
from cm1core.gui.files import CM1_OUTPUT_FILE, CM1_OUTPUT_SCHEMA, cm1_output, is_run_dir, is_view_only
from cm1core.gui.runstate import read_status

from gui_support import start_server, stop_server, wait_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CM1_FILE = ROOT / "tests" / "data" / "cm1-ref-24x24" / "cm1out_000001.nc"


def _cm1_run(where: Path) -> Path:
    where.mkdir(parents=True)
    for n in (1, 2):
        shutil.copy(CM1_FILE, where / f"cm1out_{n:06d}.nc")
    (where / "cm1out_stats.nc").write_bytes(b"")
    return where


def test_a_folder_of_cm1_files_is_cm1_output_and_opens_view_only(tmp_path):
    folder = _cm1_run(tmp_path / "cm1run")
    assert cm1_output(folder) == {"source": folder, "opened": False}
    assert is_view_only(folder) and is_run_dir(folder)
    status = read_status(folder)
    assert status.state == "finished" and status.label == "CM1 output"
    assert status.detail["actions"] == [] and status.detail["cm1_output"] is True
    # its pictures are drawn from the folder itself, into its quicklook/
    assert C.display(C.quicklook_pass_args(folder)) == f"gpuwm-cm1 quicklook {folder}"
    # a GM1 run keeps its own cm1out/ inside and is not CM1 output
    run = tmp_path / "gm1run"
    (run / "cm1out").mkdir(parents=True)
    (run / "run.json").write_text("{}")
    shutil.copy(CM1_FILE, run / "cm1out" / "cm1out_000001.nc")
    assert cm1_output(run) is None and not is_view_only(run)
    assert cm1_output(tmp_path / "empty") is None


def test_cm1_output_kept_elsewhere_is_drawn_where_it_is():
    import tempfile

    source = Path("/data/cm1run")

    with tempfile.TemporaryDirectory() as tmp:
        rundir = Path(tmp) / "mine"
        rundir.mkdir()
        (rundir / CM1_OUTPUT_FILE).write_text(json.dumps({"schema": CM1_OUTPUT_SCHEMA, "source": str(source)}))
        assert cm1_output(rundir) == {"source": source, "opened": True}
        assert C.quicklook_target(rundir) == [str(source), "--out", str(rundir / "quicklook"), "--label", "mine"]
        assert C.display(C.slice_args(rundir, "w", height_m=5000.0)) == \
            f"gpuwm-cm1 quicklook {source} --out {rundir / 'quicklook'} --label mine --slice w --height 5000 --json"
        assert C.film_args(rundir, "swath")[:6] == ["quicklook", str(source), "--out", str(rundir / "quicklook"),
                                                     "--label", "mine"]
    actions = {row.action: row for row in C.API}
    assert actions["cm1-output"].kind == "launch" and actions["cm1-output"].path == "/api/cm1-output"


def test_open_cm1_output_through_the_gui(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    source = _cm1_run(tmp_path / "data" / "cm1run")
    server, client = start_server(root, tmp_path / "config")
    try:
        dry = client.post("/api/cm1-output", {"path": str(source), "name": "mycm1", "dry_run": True})
        assert dry.status == 200 and dry.json["dry_run"], dry.json
        assert dry.json["command"] == f"gpuwm-cm1 quicklook {source} --out {root / 'mycm1' / 'quicklook'} --label mycm1"
        assert not (root / "mycm1").exists()
        opened = client.post("/api/cm1-output", {"path": str(source), "name": "mycm1"})
        assert opened.status == 200 and opened.json["ok"], opened.json
        assert opened.json["url"] == "mycm1" and opened.json["fields"] > 0
        rundir = root / "mycm1"
        record = json.loads((rundir / CM1_OUTPUT_FILE).read_text())
        assert record["schema"] == CM1_OUTPUT_SCHEMA and record["source"] == str(source)
        # what the files hold, kept from the check: the list and the header show the files' last saved time
        # from the start, not the time of the last picture drawn so far
        assert record["grid"] == {"nx": 24, "ny": 24, "nz": 40, "dx_m": 1000.0}
        assert record["saved_times_s"] == [0.0, 300.0] and record["saved_count"] == 2
        status = read_status(rundir)
        assert status.detail["model_time_s"] == 300.0 and status.detail["grid"]["nx"] == 24
        wait_for(lambda: (rundir / "quicklook" / "index.json").is_file())
        row = next(r for r in client.get("/api/runs").json["runs"] if r["id"] == "mycm1")
        assert row["cm1_output"] == {"source": str(source), "opened": True} and row["case"] == "CM1 output"
        assert row["view_only"] and row["status"]["actions"] == []
        results = client.get("/api/runs/mycm1/results").json
        assert results["cm1_output"]["source"] == str(source) and results["looks"]
        assert results["cm1_output"]["draw"].startswith(f"gpuwm-cm1 quicklook {source} --out ")
        picture = client.post("/api/runs/mycm1/slice", {"field": "w", "height_m": 5000})
        assert picture.status == 200 and picture.json["ok"], picture.json
        assert picture.json["command"].startswith(f"gpuwm-cm1 quicklook {source} --out {rundir / 'quicklook'} ")
        assert client.get(picture.json["picture_url"]).status == 200
        # no run controls: the folder opens view-only
        refused = client.post("/api/runs/mycm1/start", {})
        assert refused.status == 409
        # what cannot be opened says why
        for body, words in (({"path": str(tmp_path / "nothing"), "name": "x"}, "nothing at"),
                            ({"path": "", "name": "x"}, "empty"),
                            ({"path": str(source), "name": "mycm1"}, "already exists")):
            answer = client.post("/api/cm1-output", body)
            assert answer.status in (400, 409) and words in answer.json["message"], answer.json
        # removing it from the list deletes the pictures, never CM1's files
        gone = client.post("/api/runs/mycm1/delete", {})
        assert gone.status == 200 and gone.json["ok"], gone.json
        assert not rundir.exists() and (source / "cm1out_000001.nc").is_file()
    finally:
        stop_server(server)


def test_a_gm1_runs_cm1out_folder_in_the_runs_folder_opens_and_a_listed_folder_is_refused(tmp_path):
    root = tmp_path / "runs"
    run = root / "gm1run"
    (run / "cm1out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "name": "gm1run"}))
    for n in (1, 2):
        shutil.copy(CM1_FILE, run / "cm1out" / f"cm1out_{n:06d}.nc")
    listed = _cm1_run(root / "cm1run")                     # CM1 output kept in the runs folder: listed itself
    server, client = start_server(root, tmp_path / "config")
    try:
        ids = {r["id"] for r in client.get("/api/runs").json["runs"]}
        assert ids == {"gm1run", "cm1run"}                    # the run's cm1out/ is inside the run, not listed
        opened = client.post("/api/cm1-output", {"path": str(run / "cm1out"), "name": "gm1run-netcdf"})
        assert opened.status == 200 and opened.json["ok"], opened.json
        assert json.loads((root / "gm1run-netcdf" / CM1_OUTPUT_FILE).read_text())["source"] == str(run / "cm1out")
        one = client.post("/api/cm1-output", {"path": str(run / "cm1out" / "cm1out_000001.nc"), "name": "one-file"})
        assert one.status == 200 and one.json["ok"], one.json
        refused = client.post("/api/cm1-output", {"path": str(listed), "name": "again"})
        assert refused.status == 409 and refused.json["message"] == f"{listed} is already in the runs list, as cm1run."
        assert not (root / "again").exists()
    finally:
        stop_server(server)
