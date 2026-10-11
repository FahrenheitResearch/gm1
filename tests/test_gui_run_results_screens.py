"""The Run, Results, Runs and System screens, attach, and the data behind them.  CPU only, no browser.

What is checked:

1. A run started from the CLI (the real ``gpuwm-cm1 run`` supervisor, from a shell, with the fake door for the GPU
   integration) shows the right status in the GUI, and Stop works on it through the control file.
2. A foreign-host run shows Stale, and "Stopped without a final record" once its batch job is no longer listed.
3. The Results screen lists the cm1out files and the copy lines (netCDF and GrADS).
5. The Run and Results screens take a ``mode``: guided draws guided mode's header and the favourite-look order from the
   same components (the guided screens are those two screens with mode = "guided").

A scripted browser walkthrough of the screens on an RTX 3080 is kept with the development repository's records.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tarfile

import pytest

from cm1core.gui.runstate import read_status, warning_words
from cm1run_support import environment, finish, make_run, progress_at_least, start
from gui_support import ROOT, import_supercell, start_server, stop_server, wait_for

STATIC = ROOT / "src" / "cm1core" / "gui" / "static"
GPUINFO = ROOT / "tests" / "data" / "gpuinfo"


def _utc(seconds_ago: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat(timespec="milliseconds")


@pytest.fixture()
def fake_gpu(monkeypatch):
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(GPUINFO / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(GPUINFO / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")


@pytest.fixture()
def live(tmp_path, monkeypatch, fake_gpu):
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.001")
    monkeypatch.setenv("GM1_FAKE_STARTUP_S", "0.05")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    yield server, client, tmp_path / "runs"
    stop_server(server)


def _finished_run(client, name="sc1"):
    assert import_supercell(client, name)["ok"]
    assert client.post(f"/api/runs/{name}/estimate").json["ok"]
    assert client.post(f"/api/runs/{name}/start").json["ok"]
    wait_for(lambda: client.get(f"/api/runs/{name}/status").json["state"] == "finished", timeout=120)
    # the sidecars end with the run's end record; wait for their last files
    wait_for(lambda: "cm1out_stats.nc" in {row["name"] for row in client.get(f"/api/runs/{name}/results").json["cm1out"]},
             timeout=60)
    wait_for(lambda: len(client.get(f"/api/runs/{name}/results").json["frames"]) == 9, timeout=60)


def _logged(rundir: Path) -> list[str]:
    return [line for line in (rundir / "commands.log").read_text().splitlines() if line and not line.startswith("#")]


# ---------------------------------------------------------------- 1. a run started from the command line


def test_a_run_started_from_a_shell_shows_its_status_and_stops_from_the_gui(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    env = environment(tmp_path, FAKE_STEP_S="0.05")
    for key in ("GPUWM_CM1_NVIDIA_SMI", "GPUWM_GPU_LOCK_ROOT"):
        monkeypatch.setenv(key, env[key])
    first = make_run(root, name="shell1", duration=6000.0, history=600.0, restart=1200.0)
    second = make_run(root, name="shell2", duration=600.0, history=300.0, restart=600.0)
    server, client = start_server(root, tmp_path / "config")
    process = start("run", str(first), env=env, log=tmp_path / "shell1.log")
    waiting = None
    try:
        wait_for(lambda: progress_at_least(first, 5), timeout=60)
        runs = {row["id"]: row for row in client.get("/api/runs").json["runs"]}
        assert runs["shell1"]["status"]["state"] == "running", runs["shell1"]["status"]
        assert runs["shell1"]["status"]["label"].startswith("Running")
        assert runs["shell1"]["host"] and runs["shell1"]["wall_s"] is not None and runs["shell1"]["disk_bytes"] > 0

        # the second run waits in the supervisor's queue for the same card: segment.json says so before any record exists
        waiting = start("run", str(second), env=env, log=tmp_path / "shell2.log")
        state = wait_for(lambda: (client.get("/api/runs/shell2/status").json["state"] == "waiting") or None, timeout=30)
        assert state
        assert client.get("/api/runs/shell2/status").json["label"] == "Waiting for GPU"

        live = client.get("/api/runs/shell1/live").json
        assert live["attached"]["started_by"] == "a terminal or another GUI" and not live["attached"]["launched_here"]
        assert live["admitted"]["steps"] == 1000 and live["progress"]["step"] >= 5
        assert live["commands"]["stop"] == f"gpuwm-cm1 stop {first}"
        assert live["segments"][0]["segment"] == "001" and live["segments"][0]["kind"] == "start"

        stopped = client.post("/api/runs/shell1/stop").json
        assert stopped["ok"] and stopped["command"] == f"gpuwm-cm1 stop {first}"
        assert (first / "control" / "stop").exists() or client.get("/api/runs/shell1/status").json["state"] == "stopped"
        assert finish(process, timeout=60) == 0
        status = client.get("/api/runs/shell1/status").json
        assert (status["state"], status["label"], status["end"]["reason"]) == ("stopped", "Stopped", "stopped")
        assert "resume" in status["actions"] or status["last_restart"] is None
        assert _logged(first) == [f"gpuwm-cm1 stop {first}"]
        # the second run now has the card
        wait_for(lambda: client.get("/api/runs/shell2/status").json["state"] in ("running", "finished"), timeout=60)
        assert finish(waiting, timeout=120) == 0
        waiting = None
        assert client.get("/api/runs/shell2/status").json["state"] == "finished"
    finally:
        for proc in (process, waiting):
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, 9)
        stop_server(server)


# ---------------------------------------------------------------- 2. a run on another host


def _foreign_run(root: Path, name: str, age_s: float, job: str | None) -> Path:
    rundir = root / name
    segment = rundir / "segments" / "001"
    segment.mkdir(parents=True)
    scheduler = None if job is None else {"kind": "slurm", "job_id": job}
    row = {"number": 1, "directory": "segments/001", "host": "gpu17.cluster.example", "pid": 424242,
           "pid_start_ticks": 1, "scheduler": scheduler, "started_utc": _utc(age_s + 600)}
    (rundir / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "name": name, "segments": [row],
                                                 "host": "gpu17.cluster.example", "scheduler": scheduler}))
    records = [
        {"kind": "admitted", "host": "gpu17.cluster.example", "pid": 424242, "time_utc": _utc(age_s + 500),
         "duration_s": 7200.0, "steps": 1200, "start_step": 0, "grid": {"nx": 120, "ny": 120, "nz": 40, "dx_m": 1000.0}},
        {"kind": "frame", "pid": 424242, "time_utc": _utc(age_s + 60), "index": 4, "model_time_s": 3600.0},
        {"kind": "progress", "pid": 424242, "time_utc": _utc(age_s), "phase": "stepping", "step": 640,
         "total_steps": 1200, "model_time_s": 3840.0, "duration_s": 7200.0, "eta_s": 300.0},
    ]
    with (segment / "events.jsonl").open("w") as stream:
        for seq, record in enumerate(records):
            stream.write(json.dumps({"schema": "gm1.run-events/v1", "seq": seq, **record}) + "\n")
    return rundir


def test_a_run_on_another_host_reads_stale_then_stopped_once_its_job_is_gone(tmp_path, monkeypatch, fake_gpu):
    root = tmp_path / "runs"
    fresh = _foreign_run(root, "fresh", 20.0, "9001")
    old = _foreign_run(root, "old", 90.0, "9002")
    gone = _foreign_run(root, "gone", 90.0, "9003")
    # squeue on the login node: job 9002 still running, 9003 no longer listed
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    squeue = bin_dir / "squeue"
    squeue.write_text("#!/bin/sh\n"
                      "for a in \"$@\"; do case \"$a\" in 9002) echo RUNNING;; esac; done\nexit 0\n")
    squeue.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    server, client = start_server(root, tmp_path / "config")
    try:
        rows = {row["id"]: row for row in client.get("/api/runs").json["runs"]}
        assert rows["fresh"]["status"]["state"] == "running"
        stale = rows["old"]["status"]
        assert (stale["state"], stale["label"]) == ("stale", "Stale (no word for 2 min)")
        assert stale["scheduler_state"] == "RUNNING" and stale["local"] is False
        assert "stop" in stale["actions"] and "force-stop" not in stale["actions"]
        assert rows["old"]["scheduler"] == {"kind": "slurm", "job_id": "9002"} and rows["old"]["host"] == "gpu17.cluster.example"
        # Wall counts up to the run's last word, not up to now, once it has gone quiet
        assert rows["old"]["wall_s"] == pytest.approx(600.0, abs=3.0)
        lost = rows["gone"]["status"]
        assert (lost["state"], lost["label"]) == ("stopped", "Stopped without a final record (last frame 4)")
        assert lost["final_record"] is False and lost["scheduler_state"] == "gone"
        # /proc is never read for another host's pid: 424242 means nothing here
        assert read_status(old, host="login1", scheduler=lambda job: "unknown").state == "stale"
        live = client.get("/api/runs/old/live").json
        assert live["attached"]["started_by"] == "slurm job 9002" and live["attached"]["local"] is False
        assert live["last_write"]["kind"] == "frame" and live["last_write"]["index"] == 4
        # Stop on a stale remote run writes the control file (the node reads it), and never signals
        reply = client.post("/api/runs/old/stop").json
        assert reply["ok"] and (old / "control" / "stop").exists() and reply["escalates_to"] is None
    finally:
        stop_server(server)
    assert gone.is_dir() and fresh.is_dir()


# ---------------------------------------------------------------- 3. Results: files and copy lines


def test_results_lists_the_cm1out_files_copy_lines_receipt_and_disk(live):
    server, client, root = live
    _finished_run(client)
    results = client.get("/api/runs/sc1/results?mode=advanced").json
    names = [row["name"] for row in results["cm1out"]]
    assert "cm1out_000001.nc" in names and "cm1out_stats.nc" in names
    assert all(row["url"].startswith("/api/runs/sc1/files/cm1out/") and row["bytes"] > 0 for row in results["cm1out"])
    copy = {row["label"]: row["text"] for row in results["copy"]}
    folder = root / "sc1" / "cm1out"
    assert copy["ncview"] == f"ncview {folder}/cm1out_0*.nc"
    assert copy["Python (xarray)"] == f"import xarray as xr; ds = xr.open_mfdataset('{folder}/cm1out_0*.nc', combine='by_coords')"
    assert results["disk"]["total_bytes"] >= results["disk"]["cm1out_bytes"] > 0
    assert [row["label"] for row in results["receipt_summary"]][:1] == ["Status"]
    assert results["receipt_url"] == "/api/runs/sc1/files/segments/001/receipt.json"
    assert len(results["frames"]) == 9 and results["frames"][0]["time_s"] == 0
    assert results["segments"][0]["reason"] == "completed"
    for key in ("resume", "convert", "reproduce", "status", "stop", "pause", "force_stop"):
        assert results["commands"][key].startswith("gpuwm-cm1 "), key
    assert client.get(results["cm1out"][0]["url"]).status == 200


def test_grads_descriptors_get_open_lines():
    from cm1core.gui.api import results_copy_lines

    lines = results_copy_lines(Path("/r/cm1out"), ["cm1out_s.ctl", "cm1out_s.dat", "cm1out_stats.ctl",
                                                   "cm1out_metadata.ctl", "cm1out.nc"])
    assert {row["text"] for row in lines} == {
        "ncview /r/cm1out/cm1out.nc", "import xarray as xr; ds = xr.open_dataset('/r/cm1out/cm1out.nc')",
        "open /r/cm1out/cm1out_s.ctl", "open /r/cm1out/cm1out_stats.ctl"}


# ---------------------------------------------------------------- the Run screen's data


def test_the_run_screen_reads_progress_startup_last_write_looks_series_and_log(live):
    server, client, root = live
    _finished_run(client)
    wait_for(lambda: client.get("/api/runs/sc1/live").json["frames_drawn"] == 9, timeout=60)
    live_view = client.get("/api/runs/sc1/live").json
    assert live_view["status"]["state"] == "finished" and live_view["attached"]["started_by"] == "this GUI"
    assert live_view["admitted"]["grid"]["nx"] == 120 and live_view["admitted"]["microphysics"] == "morrison-mp10"
    assert [name for name, _ in live_view["startup"]["stages"]] == ["compile", "initializer"]
    assert live_view["progress"]["step"] == 1200 and live_view["progress"]["s_per_step_mean"] > 0
    assert live_view["last_write"]["kind"] in ("frame", "restart") and live_view["last_write"]["age_s"] >= 0
    assert live_view["frames"] == 9 and live_view["restarts"] >= 1
    # the Run screen's order: reflectivity, updraft, surface, rain, swath, then the rest
    assert [look["id"] for look in live_view["looks"]][:5] == ["reflectivity", "updraft-5km", "surface", "rain", "swath"]
    assert all(look["url"].startswith("/api/runs/sc1/files/quicklook/") for look in live_view["looks"])
    assert live_view["series"]["schema"] == "gm1.quicklook-series/v1" and live_view["series"]["charts"]
    assert set(live_view["commands"]) >= {"stop", "pause", "force_stop", "resume", "status", "convert", "reproduce"}
    assert live_view["commands_log"] == "/api/runs/sc1/files/commands.log"


def test_a_swath_warning_reads_as_one_plain_line():
    record = {"code": "GM1-SWATHS-RESTART-HERE",
              "message": "swaths restart here: sws, svs hold the extremes since t=900 s (step 150), not since the "
                         "start of the run, because the run that wrote restart 000001 did not keep them"}
    assert warning_words(record) == ("The swaths start again at 15 min: the save point this run continued from did "
                                     "not keep them.")
    assert warning_words({"code": "GM1-OTHER", "message": "as written"}) == "as written"


# ---------------------------------------------------------------- Results: convert again, reproduce bundle


def test_convert_again_is_refused_while_the_run_goes_and_runs_after(live, monkeypatch):
    server, client, root = live
    monkeypatch.setenv("GM1_FAKE_STEP_S", "0.02")
    assert import_supercell(client, "slow")["ok"]
    assert client.post("/api/runs/slow/start").json["ok"]
    wait_for(lambda: client.get("/api/runs/slow/status").json["state"] == "running", timeout=30)
    refused = client.post("/api/runs/slow/convert")
    assert refused.status == 409 and refused.json["command"] == f"gpuwm-cm1 cm1out {root / 'slow'} --force"
    client.post("/api/runs/slow/stop")
    wait_for(lambda: client.get("/api/runs/slow/status").json["state"] in ("stopped", "finished"), timeout=60)
    wait_for(lambda: not server.jobs.launched("slow"), timeout=60)
    reply = client.post("/api/runs/slow/convert", {"format": "nc"}).json
    assert reply["ok"] and reply["command"] == f"gpuwm-cm1 cm1out {root / 'slow'} --force --format nc"
    assert client.post("/api/runs/slow/convert", {"format": "hdf5"}).status == 400
    wait_for(lambda: not server.jobs.launched("slow"), timeout=60)
    logged = _logged(root / "slow")
    assert f"nice -n 19 env CUDA_VISIBLE_DEVICES= gpuwm-cm1 cm1out {root / 'slow'} --force --format nc" in logged


def test_the_reproduce_bundle_holds_the_recipe_and_the_record(live):
    server, client, root = live
    _finished_run(client)
    rundir = root / "sc1"
    assert client.post("/api/runs/sc1/edit", {"settings": {"initialization.zc_m": 1500.0}}).status in (200, 409)
    reply = client.post("/api/runs/sc1/reproduce").json
    assert reply["ok"] and reply["command"] == f"gpuwm-cm1 reproduce {rundir} --json"
    bundle = rundir / "reproduce"
    names = {row["name"] for row in reply["bundle"]["files"]}
    assert {"README.txt", "SHA256SUMS", "version.json", "namelist.input", "namelist.input.original", "deck.toml",
            "import-report.json", "run.json", "commands.log", "segments/001/receipt.json"} <= names
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest, name
    version = json.loads((bundle / "version.json").read_text())
    assert version["schema"] == "gpuwm-cm1.reproduce-bundle/v1" and version["gpuwm_cm1"]["version"]
    readme = (bundle / "README.txt").read_text()
    assert "gpuwm-cm1 import namelist.input --out sc1-again" in readme and "gpuwm-cm1 run sc1-again" in readme
    with tarfile.open(bundle / "sc1-reproduce.tar.gz") as archive:
        members = archive.getnames()
    assert "sc1-reproduce/README.txt" in members and "sc1-reproduce/segments/001/receipt.json" in members
    assert not any(name.endswith(".tar.gz") for name in members)
    assert reply["tarball_url"] == "/api/runs/sc1/files/reproduce/sc1-reproduce.tar.gz"
    download = client.get(reply["tarball_url"])
    assert download.status == 200 and download.headers["content-disposition"].startswith("attachment")
    assert f"gpuwm-cm1 reproduce {rundir} --json" in _logged(rundir)
    results = client.get("/api/runs/sc1/results").json
    assert results["reproduce"]["tarball_url"] == reply["tarball_url"]
    # a second bundle replaces the first
    assert client.post("/api/runs/sc1/reproduce").json["ok"]
    assert not list(rundir.glob(".reproduce*"))


def test_reproduce_lists_the_settings_edits_and_acknowledgments(tmp_path):
    from cm1core.cm1run.reproduce import build_bundle, rerun_commands
    from cm1core.cm1run.rundir import RunDirectory

    rundir = make_run(tmp_path, name="r1")
    (rundir / "input_sounding").write_text("1000.0 300.0 14.0\n")
    (rundir / "settings.toml").write_text("[initialization]\nzc_m = 1500.0\n[output]\nalso_netcdf = true\n")
    record = json.loads((rundir / "run.json").read_text())
    record["acknowledgments"] = ["output-keys", "thompson"]
    (rundir / "run.json").write_text(json.dumps(record))
    lines = rerun_commands(RunDirectory(rundir), record)
    assert lines[0] == "gpuwm-cm1 import namelist.input --sounding input_sounding --out r1-again --acknowledge-output-keys"
    assert lines[1] == ("gpuwm-cm1 edit r1-again --setting initialization.zc_m=1500 --setting output.also_netcdf=true "
                        "--acknowledge-thompson")
    assert lines[2] == "gpuwm-cm1 run r1-again"
    summary = build_bundle(rundir)
    assert "input_sounding" in {row["name"] for row in summary["files"]}
    with pytest.raises(ValueError):
        build_bundle(tmp_path / "not-a-run")


# ---------------------------------------------------------------- Runs: the GPU strip; System


def test_the_gpu_strip_says_what_nvidia_smi_says(live):
    server, client, root = live
    gpus = client.get("/api/gpus").json
    assert gpus["status"] == "ok" and gpus["wsl2"] is True
    (card,) = gpus["gpus"]
    assert card["name"] == "NVIDIA GeForce RTX 3080" and card["driver"] == "610.74" and card["sm"] == "sm_86"
    assert card["total_bytes"] == 10240 * 1024 ** 2 and card["processes_visible"] is False
    assert "cannot name Windows programs" in card["note"] and card["gm1_runs"] == []


def test_the_gpu_strip_with_no_nvidia_smi(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", "none")
    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        gpus = client.get("/api/gpus").json
        assert gpus["status"] == "no-nvidia-smi" and gpus["gpus"] == [] and "nvidia-smi" in gpus["message"]
    finally:
        stop_server(server)


def test_system_is_doctor_in_plain_words(live):
    server, client, root = live
    system = client.get("/api/system").json
    assert system["command"] == f"gpuwm-cm1 doctor --plain --root {root} --json"
    report = system["report"]
    areas = [row["area"] for row in report["rows"]]
    assert areas[:2] == ["Driver", "CUDA"] and "GPU 0" in areas
    # The presets row, when the GPU is usable, comes last.
    assert areas[-4:] == ["Python", "Disk", "RAM", "Presets"] or areas[-3:] == ["Python", "Disk", "RAM"]
    assert {"ArWen", "Native tools", "GPU 0 memory"} <= set(areas)
    for row in report["rows"]:
        assert row["state"] in ("ok", "note", "fail")
        if row["state"] == "fail":
            assert row["fix"], row
    assert report["exit"] == (1 if any(row["state"] == "fail" for row in report["rows"]) else 0)
    assert client.get("/api/system").json["cached_s"] >= 0


def test_doctor_plain_names_each_failure_with_its_fix(tmp_path, monkeypatch, capsys):
    from cm1core.cli import main
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.system import driver_row, gpu_rows, system_report

    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", "none")
    monkeypatch.setenv("GPUWM_CM1_WSL2", "0")
    assert main(["doctor", "--plain", "--root", str(tmp_path)]) == 1
    text = capsys.readouterr().out
    assert "FIX   Driver" in text and "fix: Install the NVIDIA driver R580" in text
    report = system_report(tmp_path, query=gpuinfo.GpuQuery("no-nvidia-smi", message="none"))
    assert report["rows"][0]["state"] == "fail" and report["command"] == f"gpuwm-cm1 doctor --plain --root {tmp_path}"

    monkeypatch.setenv("GPUWM_CM1_NVIDIA_SMI", str(GPUINFO / "fake_nvidia_smi.py"))
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(GPUINFO / "a100-80gb-old-driver.json"))
    old = driver_row(gpuinfo.query_gpus())
    assert old.state == "fail" and "R580" in old.text and old.fix
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(GPUINFO / "rtx3080-wsl2-desktop-4p8.json"))
    monkeypatch.setenv("GPUWM_CM1_WSL2", "1")
    rows = {row.id: row for row in gpu_rows(gpuinfo.query_gpus())}
    assert rows["gpu-0"].state == "ok" and "GM1 has run on this architecture" in rows["gpu-0"].text
    assert rows["gpu-0-memory"].state == "note" and "Windows desktop" in rows["gpu-0-memory"].text
    monkeypatch.setenv("FAKE_NVIDIA_SMI_FIXTURE", str(GPUINFO / "h100-sm90.json"))
    rows = {row.id: row for row in gpu_rows(gpuinfo.query_gpus())}
    assert rows["gpu-0"].state == "note" and "not run on this architecture" in rows["gpu-0"].text
    assert main(["doctor", "--plain", "--root", str(tmp_path), "--json"]) in (0, 1)
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "gpuwm-cm1.system/v1" and payload["gpus"][0]["sm"] == "sm_90"


# ---------------------------------------------------------------- attach: a plain cartesian output folder


def test_a_plain_cartesian_folder_opens_view_only(live):
    server, client, root = live
    folder = root / "plain"
    (folder / "frames").mkdir(parents=True)
    with (folder / "events.jsonl").open("w") as stream:
        for seq, record in enumerate([
            {"kind": "admitted", "host": "here", "pid": 1, "time_utc": _utc(100), "duration_s": 600.0, "steps": 100,
             "grid": {"nx": 8, "ny": 8, "nz": 8, "dx_m": 1000.0}},
            {"kind": "frame", "pid": 1, "time_utc": _utc(60), "index": 1, "model_time_s": 300.0},
            {"kind": "end", "pid": 1, "time_utc": _utc(50), "reason": "completed", "model_time_s": 600.0},
        ]):
            stream.write(json.dumps({"schema": "gm1.run-events/v1", "seq": seq, "elapsed_s": 50.0, **record}) + "\n")
    rows = {row["id"]: row for row in client.get("/api/runs").json["runs"]}
    assert rows["plain"]["view_only"] is True and rows["plain"]["status"]["state"] == "finished"
    assert rows["plain"]["status"]["actions"] == [] and rows["plain"]["wall_s"] == 50.0
    live_view = client.get("/api/runs/plain/live").json
    assert live_view["view_only"] and live_view["attached"]["started_by"].startswith("gpuwm-cm1 cartesian")
    assert live_view["commands"] == {"convert": f"gpuwm-cm1 cm1out {folder}", "quicklook": f"gpuwm-cm1 quicklook {folder}"}
    results = client.get("/api/runs/plain/results").json
    assert results["view_only"] and results["restarts"] == [] and results["segments"] == []
    for action in ("start", "resume", "convert", "reproduce", "stop"):
        answer = client.post(f"/api/runs/plain/{action}")
        assert answer.status == 409 and "view-only" in answer.json["message"], action


# ---------------------------------------------------------------- 5. one screen, two modes


def test_guided_results_order_the_favourite_looks_and_add_the_panels(live):
    server, client, root = live
    _finished_run(client)
    wait_for(lambda: len((client.get("/api/runs/sc1/results").json["looks"] or [{}])[0].get("frames", [])) == 9,
             timeout=60)
    guided = client.get("/api/runs/sc1/results?mode=guided").json
    advanced = client.get("/api/runs/sc1/results?mode=advanced").json
    assert [look["id"] for look in guided["looks"]][:4] == ["reflectivity", "spin-near-ground", "updraft-5km", "swath"]
    assert all(look["favourite"] for look in guided["looks"][:4]) and "guided" in guided and "guided" not in advanced
    assert set(guided["guided"]["looks"]) >= {"reflectivity", "spin-near-ground", "updraft-5km", "swath"}
    # the same data otherwise: one engine, one contract
    assert guided["cm1out"] == advanced["cm1out"] and guided["copy"] == advanced["copy"]
    assert guided["frames"] == advanced["frames"]


def test_the_guided_screens_are_the_run_and_results_screens_in_guided_mode():
    watch = (STATIC / "guided" / "watch.js").read_text()
    results = (STATIC / "guided" / "results.js").read_text()
    assert 'import { renderRun } from "../js/run/run.js"' in watch and 'mode: "guided"' in watch
    assert 'import { renderResults } from "../js/run/results.js"' in results and 'mode: "guided"' in results
    run = (STATIC / "js" / "run" / "run.js").read_text()
    res = (STATIC / "js" / "run" / "results.js").read_text()
    for source in (run, res):
        assert re.search(r'mode === "guided" \? guided\w+\(body, id, frame\) : advanced\w+\(body, id, frame\)', source)
        assert "headerClock(" in source and "viewer(" in source and 'from "./parts.js"' in source
    # both modes' pictures come from one viewer; guided keeps the four favourites in guided mode's order
    parts = (STATIC / "js" / "run" / "parts.js").read_text()
    assert 'export const FAVOURITES = ["reflectivity", "spin-near-ground", "updraft-5km", "swath"];' in parts
    app = (STATIC / "js" / "app.js").read_text()
    for module in ("./runs.js", "./run/run.js", "./run/results.js", "./system.js"):
        assert f'import "{module}";' in app


def test_every_new_page_string_passes_the_word_rules():
    """No marketing words and no tornado wording on the new screens (section 6), as the guided copy lint holds for guided."""

    from cm1core.gui.copy_lint import BANNED, js_strings, viz20_text_lint

    for path in [STATIC / "js" / "runs.js", STATIC / "js" / "system.js", *sorted((STATIC / "js" / "run").glob("*.js"))]:
        for text in js_strings(path.read_text()):
            low = text.lower()
            assert not [p for p in BANNED if re.search(p, low)], (path.name, text)
            assert not viz20_text_lint(text), (path.name, text)


def test_the_api_table_names_convert_and_reproduce():
    from cm1core.gui.commands import API

    table = {endpoint.action: endpoint for endpoint in API}
    assert table["convert"].command == "gpuwm-cm1 cm1out RUNDIR --force [--format nc|grads]"
    assert table["reproduce"].command == "gpuwm-cm1 reproduce RUNDIR --json"
    assert table["convert"].path == "/api/runs/RUN/convert" and table["reproduce"].method == "POST"


# ---------------------------------------------------------------- verification: edges


def _ended_run(root: Path, name: str, *, reason: str, model_time_s: float, until_s: float) -> Path:
    """A run on this computer whose one segment ended with ``reason`` at ``model_time_s`` (plan until ``until_s``)."""

    from cm1core.gui.runstate import this_host

    rundir = root / name
    segment = rundir / "segments" / "001"
    segment.mkdir(parents=True)
    row = {"number": 1, "directory": "segments/001", "host": this_host(), "kind": "fresh", "until_s": until_s,
           "started_utc": _utc(120), "ended_utc": _utc(10), "reason": reason}
    (rundir / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "name": name, "segments": [row],
                                                 "host": this_host()}))
    (segment / "segment.json").write_text(json.dumps({
        "state": "ended", "host": this_host(), "reason": reason, "ended_utc": _utc(10),
        "plan": {"until_s": until_s, "restart_interval_s": 600.0}}))
    records = [
        {"kind": "admitted", "time_utc": _utc(110), "duration_s": until_s, "steps": 100, "start_step": 0},
        {"kind": "restart", "time_utc": _utc(12), "number": 1, "model_time_s": model_time_s, "step": 100},
        {"kind": "end", "time_utc": _utc(10), "reason": reason, "model_time_s": model_time_s, "rc": 0},
    ]
    with (segment / "events.jsonl").open("w") as stream:
        for seq, record in enumerate(records):
            stream.write(json.dumps({"schema": "gm1.run-events/v1", "seq": seq, **record}) + "\n")
    return rundir


def test_a_pause_on_the_last_restart_reads_finished_as_the_cli_says(tmp_path):
    from cm1core.cm1run.status import run_status

    at_end = _ended_run(tmp_path, "at-end", reason="paused", model_time_s=600.0, until_s=600.0)
    before = _ended_run(tmp_path, "before", reason="paused", model_time_s=300.0, until_s=600.0)
    gui, cli = read_status(at_end), run_status(at_end)
    assert (gui.state, gui.label, cli.state) == ("finished", "Finished", "finished")
    assert "resume" in gui.detail["actions"]
    gui, cli = read_status(before), run_status(before)
    assert (gui.state, gui.label, cli.state) == ("paused", "Paused", "paused")


def test_a_malformed_run_folder_does_not_break_the_runs_list(tmp_path, fake_gpu):
    root = tmp_path / "runs"
    listed = root / "listed"
    (listed / "segments" / "001").mkdir(parents=True)
    (listed / "run.json").write_text("[1, 2]")                      # valid JSON, not an object
    torn = root / "torn"
    (torn / "segments" / "001").mkdir(parents=True)
    (torn / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "name": "torn",
                                               "segments": [{"number": 1, "host": "x"}]}))
    (torn / "segments" / "001" / "segment.json").write_text("[1, 2]")
    odd = root / "odd"
    (odd / "segments" / "001").mkdir(parents=True)
    (odd / "run.json").write_text(json.dumps({"schema": "gm1.cm1-run/v1", "name": "odd", "gpu": "RTX",
                                              "scheduler": "slurm", "segments": ["001", {"number": 1}]}))
    with (odd / "segments" / "001" / "events.jsonl").open("w") as stream:
        stream.write(json.dumps({"seq": 0, "kind": "admitted", "time_utc": _utc(30),
                                 "resumed_from": "/elsewhere/restarts/000001"}) + "\n")
        stream.write(json.dumps({"seq": 1, "kind": "startup", "time_utc": _utc(29), "stages": [1, 2]}) + "\n")
        stream.write("not json\n")
    server, client = start_server(root, tmp_path / "config")
    try:
        answer = client.get("/api/runs")
        assert answer.status == 200, answer.raw
        assert {row["id"] for row in answer.json["runs"]} == {"listed", "torn", "odd"}
        for name in ("listed", "torn", "odd"):
            for page in ("live", "results", "status"):
                assert client.get(f"/api/runs/{name}/{page}").status == 200, (name, page)
        odd_segment = client.get("/api/runs/odd/live").json["segments"][0]
        assert odd_segment["resumed_from"] == {"directory": "/elsewhere/restarts/000001"}
        assert odd_segment["kind"] == "continue"
        assert client.get("/api/gpus").status == 200
    finally:
        stop_server(server)


def test_the_bundle_name_never_leaves_the_bundle(tmp_path):
    from cm1core.cm1run.reproduce import build_bundle

    rundir = make_run(tmp_path, name="r2")
    record = json.loads((rundir / "run.json").read_text())
    record["name"] = "../../escape me"
    (rundir / "run.json").write_text(json.dumps(record))
    summary = build_bundle(rundir)
    assert summary["tarball_name"] == "escape-me-reproduce.tar.gz"
    assert Path(summary["tarball"]).parent == rundir / "reproduce"
    assert not list(tmp_path.glob("*.tar.gz")) and not list(rundir.glob("*.tar.gz"))
    with tarfile.open(summary["tarball"]) as archive:
        assert all(name.startswith("escape-me-reproduce/") for name in archive.getnames())


def test_the_ram_row_states_the_locked_memory_the_estimator_uses():
    from cm1core.cm1run import gpuinfo
    from cm1core.cm1run.system import ram_row

    host = gpuinfo.HostInfo(disk_root="/", disk_free_bytes=None, disk_total_bytes=None,
                            ram_total_bytes=8 * 10**9, ram_available_bytes=6 * 10**9, wsl2=False)
    row = ram_row(host)
    assert row.state == "note" and "365 bytes of RAM per grid cell" in row.text and "third of a byte" not in row.text


def test_the_viewer_follows_the_newest_frame_and_the_run_screen_replaces_its_status():
    parts = (STATIC / "js" / "run" / "parts.js").read_text()
    update = parts[parts.index("box.update = "):parts.index("box.select = ")]
    # whether the shown frame was the newest is read before the new list replaces the old one
    assert update.index("const atEnd") < update.index("looks = [")
    run = (STATIC / "js" / "run" / "run.js").read_text()
    assert "status = { ...status, ...(live.status" not in run and run.count("if (live.status) status = live.status;") == 2


def test_a_mistyped_run_link_says_there_is_no_such_run(tmp_path, fake_gpu):
    """Run/Watching and Results ask runExists before drawing. A name the API refuses as a path answers 400
    ("Use a plain name.") and an unknown one 404; both are "There is no run called ..." with Back to Runs,
    never an empty page waiting for a first picture."""

    import re

    server, client = start_server(tmp_path / "runs", tmp_path / "config")
    try:
        typo = client.get("/api/runs/Supercell%20typo/status")
        missing = client.get("/api/runs/nosuchrun/status")
    finally:
        stop_server(server)
    parts = (STATIC / "js" / "run" / "parts.js").read_text()
    statuses = {int(n) for n in re.search(r"NO_SUCH_RUN_STATUSES = \[([\d, ]+)\]", parts).group(1).split(",")}
    assert {typo.status, missing.status} <= statuses, (typo.status, missing.status)
    assert "NO_SUCH_RUN_STATUSES.includes(err.status)" in parts
    for screen in ("run.js", "results.js"):
        source = (STATIC / "js" / "run" / screen).read_text()
        assert "await runExists(id)" in source and "runNotFound(" in source, screen


def test_a_continue_time_that_cannot_be_read_is_said_not_dropped():
    """The advanced Run and Results pages' Continue fields: "90" (the old minutes) or "2h" is not T+H:MM:SS.
    The button says so and sends nothing, as the Continue page does, instead of a {} that goes on to the run's
    default end; a blank field still means that end."""

    parts = (STATIC / "js" / "run" / "parts.js").read_text()
    assert 'export function untilProblem(text)' in parts
    assert 'String(text ?? "").trim() === "" || parseClock(text) ? null : UNTIL_HELP' in parts
    click = parts[parts.index('b.addEventListener("click"'):parts.index("wrap.button = b;")]
    assert click.index("opts.check") < click.index("api.post(")
    assert 'if (problem) { notice(problem, "stop"); return; }' in click
    for screen in ("run.js", "results.js"):
        source = (STATIC / "js" / "run" / screen).read_text()
        resume = source[source.index('"resume", "Continue"'):]
        resume = resume[:resume.index("});") + 3]
        assert "check: () => untilProblem(until.value)" in resume, screen
    # The Continue page says the same words, from the same constant.
    continue_page = (STATIC / "js" / "advanced" / "continue.js").read_text()
    assert re.search(r'export const UNTIL_HELP = "Give the storm time to end at as T\+H:MM:SS[^"]*"', parts)
    assert 'import { UNTIL_HELP } from "../run/parts.js";' in continue_page
    assert 'if (select.value === "custom" && !until) return { error: UNTIL_HELP };' in continue_page
