"""Shared fixtures for the run-directory tests (cm1core.cm1run).

Runs here go through the real ``gpuwm-cm1 run`` supervisor in a subprocess,
with ``tests/cm1run_fake_door.py`` in place of the GPU integration, a fake
``nvidia-smi`` and a private lock directory, so the queue, control files,
status and crash detection are the shipped code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"

DECK = """\
# test deck
[run]
dt_s = 6.0
duration_s = {duration}
history_interval_s = {history}
lane = "cartesian"
restart_interval_s = {restart}

[grid]
dx_m = 1000.0
nx = 8
ny = 8
nz = 8

[physics]
component = "morrison-mp10"
"""


def make_run(root: Path, *, name: str = "run", duration: float = 360.0, history: float = 60.0,
             restart: float = 120.0, rstfrq: float | None = None, import_block: dict | None = None) -> Path:
    """A run directory as ``gpuwm-cm1 import`` leaves it (no native parser needed)."""

    from cm1core.cm1run.rundir import RUN_SCHEMA, write_json_atomic

    path = root / name
    path.mkdir(parents=True)
    (path / "control").mkdir()
    (path / "deck.toml").write_text(
        DECK.format(duration=float(duration), history=float(history), restart=float(restart)),
        encoding="utf-8",
    )
    (path / "namelist.input").write_text("&param1\n/\n", encoding="ascii")
    (path / "namelist.input.original").write_text("&param1\n/\n", encoding="ascii")
    write_json_atomic(path / "run.json", {
        "schema": RUN_SCHEMA,
        "name": name,
        "created_utc": "2026-09-23T00:00:00+00:00",
        "cm1": {"timax": duration, "tapfrq": history, "rstfrq": restart if rstfrq is None else rstfrq,
                "dtl": 6.0, "isnd": 5, "ptype": 5, "imoist": 1, "nx": 8, "ny": 8, "nz": 8,
                "dx": 1000.0, "dy": 1000.0, "dz": 500.0},
        "import": import_block or {"result": "runnable", "component": "morrison-mp10",
                                   "refused": [], "acknowledgments_needed": []},
        "acknowledgments": [],
        "deck": {"file": "deck.toml"},
        "host": None, "gpu": None, "scheduler": None,
        "segments": [],
    })
    return path


FAKE_SMI = """#!/bin/sh
# fake nvidia-smi: one card; memory in use from $FAKE_SMI_USED_MIB
echo "0, GPU-fake-0000-0000, Fake GPU 10 GB, 10240, ${FAKE_SMI_USED_MIB:-4300}, 999.99"
"""


def environment(tmp_path: Path, **extra: str) -> dict[str, str]:
    smi = tmp_path / "bin" / "nvidia-smi"
    if not smi.exists():
        smi.parent.mkdir(parents=True, exist_ok=True)
        smi.write_text(FAKE_SMI, encoding="ascii")
        smi.chmod(0o755)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(SRC), str(TESTS)]),
        "GPUWM_CM1_RUN_CHILD_MODULE": "cm1run_fake_door",
        "GPUWM_CM1_NVIDIA_SMI": str(smi),
        "GPUWM_GPU_LOCK_ROOT": str(tmp_path / "locks"),
        "FAKE_STEP_S": "0.02",
        "PYTHONUNBUFFERED": "1",
    })
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("SLURM_JOB_ID", None)
    env.pop("PBS_JOBID", None)
    env.update(extra)
    return env


def gm1(*words: str, env: dict[str, str], timeout: float = 120.0, check: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "cm1core", *words], env=env, capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(f"gpuwm-cm1 {' '.join(words)} -> {completed.returncode}\n{completed.stderr}")
    return completed


def start(*words: str, env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
    """``gpuwm-cm1 WORDS`` from a shell: its own session, no parent but this test."""

    handle = open(log, "wb")
    return subprocess.Popen(
        [sys.executable, "-m", "cm1core", *words], env=env, stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )


def status(path: Path, env: dict[str, str]) -> dict[str, Any]:
    completed = gm1("status", str(path), "--json", env=env, check=True)
    return json.loads(completed.stdout)


def wait_for(predicate: Callable[[], Any], *, timeout: float = 60.0, every: float = 0.05, what: str = "") -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(every)
    raise AssertionError(f"timed out after {timeout} s waiting for {what or predicate}")


def events_of(path: Path, segment: int = 1) -> list[dict[str, Any]]:
    from cm1core.cm1run.events import read_records

    return read_records(path / "segments" / f"{segment:03d}" / "events.jsonl")


def progress_at_least(path: Path, step: int, segment: int = 1) -> bool:
    return any(
        record.get("kind") == "progress" and (record.get("step") or 0) >= step
        for record in events_of(path, segment)
    )


def finish(process: subprocess.Popen[bytes], timeout: float = 60.0) -> int:
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        raise
