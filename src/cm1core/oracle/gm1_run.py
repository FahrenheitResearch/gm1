"""Run gpuwm-cm1 on a differential case, locally or on a remote CUDA host.

The CUDA runtime needs a device; CM1 does not.  So the two halves of a
differential are usually run on different machines, and this module keeps that
explicit: ``run_case`` emits the derived TOML, drives
``gpuwm-cm1 cartesian --integrate --history-dir`` where the device is, and
brings the history frames back.

Remote mode copies only the emitted TOML out and the frame directories back.
It never installs anything and never touches the remote estate's own runs.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .gm1_frames import GM1History, open_history
from .spec import DifferentialCase


class GM1RunError(RuntimeError):
    pass


#: Where a remote CUDA host keeps the pieces.  There are no built-in values:
#: the layout of a CUDA host is the operator's, so each one is named by an
#: environment variable (or, for ``root`` and ``python``, an explicit
#: argument).  A remote node without an installed wheel and Rust toolchain is
#: driven from source: :func:`deploy` puts this worktree's ``cm1core`` and two
#: locally built native artifacts beside each other under ``root`` and
#: ``run_case`` runs them with the node's own cupy.
REMOTE_SETTINGS = {
    "root": ("GM1_REMOTE_ROOT", "directory on the CUDA host that receives cm1core and native/"),
    "python": ("GM1_REMOTE_PYTHON", "interpreter on the CUDA host with cupy installed"),
    "arwen": ("GM1_REMOTE_ARWEN", "admitted ArWen tree on the CUDA host, put on PYTHONPATH"),
}


def remote_setting(name: str) -> str:
    variable, meaning = REMOTE_SETTINGS[name]
    value = os.environ.get(variable)
    if not value:
        raise GM1RunError(
            f"remote runs need the {meaning}; set {variable} "
            "(see docs/ORACLE-DIFFERENTIAL-HARNESS.md)"
        )
    return value


def deploy(remote: str, repo: Path, *, root: str | None = None, log: Path | None = None) -> str:
    """Put this worktree's gpuwm-cm1 and its native artifacts on ``remote``.

    Build the two artifacts first; cargo names the binary after its source
    file, so the target is ``cm1_forecast_publish``::

        cd rust/cm1-state && cargo build --release --bin cm1_forecast_publish --lib
    """

    root = root or remote_setting("root")
    publisher = repo / "rust/cm1-state/target/release/cm1_forecast_publish"
    library = repo / "rust/cm1-state/target/release/libcm1_state.so"
    for path in (publisher, library):
        if not path.is_file():
            raise GM1RunError(
                f"{path} is absent; build it first with "
                "`cd rust/cm1-state && cargo build --release --bin cm1_forecast_publish --lib`"
            )
    sink = log or Path(os.devnull)
    quoted = shlex.quote(root)
    steps = (
        ["ssh", remote, f"mkdir -p {quoted}/native"],
        ["rsync", "-a", "--exclude", "__pycache__", str(repo / "src/cm1core"), f"{remote}:{root}/"],
        ["scp", str(publisher), f"{remote}:{root}/native/cm1-forecast-publish"],
        ["scp", str(library), f"{remote}:{root}/native/"],
        ["ssh", remote, f"chmod +x {quoted}/native/cm1-forecast-publish"],
    )
    for argv in steps:
        code = _run(argv, sink, 1800.0)
        if code != 0:
            raise GM1RunError(f"deploy step failed ({code}): {' '.join(argv)}")
    return root


@dataclass(frozen=True)
class GM1Run:
    config: Path
    history_root: Path
    receipt: Path | None
    log: Path
    returncode: int
    wall_s: float
    history: GM1History


def _run(argv: Sequence[str], log: Path, timeout_s: float) -> int:
    with log.open("ab") as handle:
        handle.write(("\n$ " + " ".join(shlex.quote(a) for a in argv) + "\n").encode())
        handle.flush()
        completed = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                                   timeout=timeout_s, check=False)
    return completed.returncode


def emit_config(case: DifferentialCase, base: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case.name}.toml"
    path.write_text(case.gm1_toml(base), encoding="utf-8", newline="\n")
    translation = case.translate(base)
    (directory / f"{case.name}.translation.json").write_text(
        json.dumps(translation.receipt(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return path


def run_case(
    case: DifferentialCase,
    base: Path,
    directory: Path,
    *,
    remote: str | None = None,
    remote_root: str | None = None,
    remote_python: str | None = None,
    executable: str = "gpuwm-cm1",
    timeout_s: float = 3600.0,
    extra_args: Sequence[str] = (),
) -> GM1Run:
    directory.mkdir(parents=True, exist_ok=True)
    config = emit_config(case, base, directory)
    log = directory / "gm1.log"
    log.write_bytes(b"")
    history_root = directory / "frames"
    receipt = directory / "receipt.json"
    started = time.monotonic()

    if remote is None:
        argv = [
            executable, "cartesian", str(config), "--output", str(receipt),
            "--integrate", "--history-dir", str(history_root), *extra_args,
        ]
        code = _run(argv, log, timeout_s)
    else:
        root = remote_root or remote_setting("root")
        python = remote_python or remote_setting("python")
        arwen = remote_setting("arwen")
        stem = f"{root}/differential-{case.name}-{int(time.time())}"
        quoted = shlex.quote(stem)
        code = _run(["ssh", remote, f"mkdir -p {quoted}"], log, 120.0)
        if code == 0:
            code = _run(["scp", str(config), f"{remote}:{stem}/config.toml"], log, 300.0)
        if code == 0:
            # The node has no installed manifest for a source checkout, so the
            # initializer library and the history publisher are named
            # explicitly -- cartesian_door.py:1116 documents exactly this.
            command = (
                f"cd {quoted} && PYTHONPATH={shlex.quote(root)}:{shlex.quote(arwen)} "
                f"{shlex.quote(python)} -m cm1core cartesian config.toml "
                f"--output receipt.json --integrate --history-dir frames "
                f"--library {shlex.quote(root)}/native/libcm1_state.so "
                f"--history-publisher {shlex.quote(root)}/native/cm1-forecast-publish "
                + " ".join(shlex.quote(a) for a in extra_args)
            )
            code = _run(["ssh", remote, command], log, timeout_s)
        if code == 0:
            # The sealed envelope duplicates the raw arena the comparator
            # reads, so leave it on the node rather than copy it twice.
            code = _run(
                ["rsync", "-a", "--exclude", "history.cm1fhs",
                 f"{remote}:{stem}/frames", str(history_root.parent) + "/"],
                log, 3600.0,
            )
            _run(["scp", f"{remote}:{stem}/receipt.json", str(receipt)], log, 300.0)
    wall = time.monotonic() - started
    if code != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise GM1RunError(f"gpuwm-cm1 exited {code}:\n{tail}")
    return GM1Run(
        config=config,
        history_root=history_root,
        receipt=receipt if receipt.exists() else None,
        log=log,
        returncode=code,
        wall_s=wall,
        history=open_history(history_root),
    )
