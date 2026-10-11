"""Slurm and PBS job scripts for a run directory.

On a cluster the run executes on a GPU node and the GUI, on the login node,
only reads the run directory and writes control files.  The script:

* runs ``gpuwm-cm1 run RUNDIR --continue`` on the node's GPU, so the same
  script starts the run the first time and continues it from its last
  restart when it is submitted again (after a time limit, a pause or a
  node failure);
* starts two sidecars on the node's CPUs, never its GPU
  (``CUDA_VISIBLE_DEVICES=""``, ``nice -n 19``): ``gpuwm-cm1 cm1out --follow``
  converts each frame to CM1's netCDF as it lands and ``gpuwm-cm1 quicklook
  --follow`` draws the quick-looks, so no sustained work runs on the login
  node.  They start once this job's segment exists (or the run has ended
  without one), so a sidecar never mistakes the previous job's ended segment
  for this job's.  A sidecar whose command this installation does not have
  yet is skipped with a line in the log;
* asks the run to stop cleanly before the time limit (Slurm: ``--signal``;
  PBS: the scheduler's SIGTERM), so the frames and restarts written before it
  are complete, and a resubmitted script continues from the last restart.  A
  signal to the job only writes the stop request: the scheduler signals every
  process of the job itself, and a second SIGTERM would cut the run's own
  cleanup short.

Importing this module imports neither CuPy nor numpy.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any

from .rundir import RunDirectory, installation, utc_now

SCHEDULERS = ("slurm", "pbs")
#: Seconds before the time limit at which Slurm is asked to signal the job.
STOP_BEFORE_LIMIT_S = 300
#: Locked host memory, 365 B per cell (measured), plus the sidecars' working set.
LOCKED_HOST_BYTES_PER_CELL = 365.0
SIDECAR_HOST_BYTES = 4e9


class JobScriptError(ValueError):
    pass


def _also_netcdf(rundir: Path) -> bool:
    """settings.toml's "Also write netCDF", for a namelist that asks GrADS (output_format = 1)."""

    import tomllib

    try:
        output = tomllib.loads((Path(rundir) / "settings.toml").read_text(encoding="utf-8")).get("output") or {}
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    return output.get("also_netcdf") is True and output.get("format") == 1


def _walltime(text: str) -> str:
    """``HH:MM:SS`` (or ``D-HH:MM:SS`` for Slurm), checked; minutes alone are accepted."""

    value = str(text).strip()
    if re.fullmatch(r"\d+", value):
        minutes = int(value)
        return f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    if re.fullmatch(r"(\d+-)?\d{1,3}:\d{2}(:\d{2})?", value):
        return value if value.count(":") == 2 else value + ":00"
    raise JobScriptError(f"{text!r} is not a time limit; write HH:MM:SS, D-HH:MM:SS or minutes")


def _memory_gb(run: RunDirectory, requested: str | None) -> int:
    if requested:
        match = re.fullmatch(r"(\d+)\s*([GgMm]?)[Bb]?", str(requested).strip())
        if match is None:
            raise JobScriptError(f"{requested!r} is not a memory size; write e.g. 32G")
        amount = int(match.group(1))
        return max(1, math.ceil(amount / 1024)) if match.group(2).lower() == "m" else amount
    cm1 = run.load().get("cm1") or {}
    try:
        cells = int(cm1.get("nx")) * int(cm1.get("ny")) * int(cm1.get("nz"))
    except (TypeError, ValueError):
        return 32
    return max(16, math.ceil((LOCKED_HOST_BYTES_PER_CELL * cells * 1.5 + SIDECAR_HOST_BYTES) / 1e9))


def _python_prefix() -> tuple[list[str], str | None]:
    """How the script calls gpuwm-cm1: this interpreter, and PYTHONPATH for a checkout."""

    package_parent = Path(__file__).resolve().parents[2]
    pythonpath = None
    if not {"site-packages", "dist-packages"} & set(package_parent.parts):
        pythonpath = os.fspath(package_parent)
    return [sys.executable, "-m", "cm1core"], pythonpath


def render(
    path: Path | str,
    scheduler: str,
    *,
    time_limit: str = "04:00:00",
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    cpus: int = 4,
    memory: str | None = None,
    until_s: float | None = None,
    gpu_type: str | None = None,
    setup: list[str] | None = None,
    python: str | None = None,
    library: str | None = None,
    history_publisher: str | None = None,
) -> str:
    """The job script text for ``RUNDIR`` under ``scheduler`` (``slurm`` or ``pbs``).

    A run whose settings.toml asks "Also write netCDF" beside GrADS output
    gets it from the one cm1out sidecar, which reads settings.toml and writes
    both formats in the same pass, as a run started from the GUI does.
    """

    if scheduler not in SCHEDULERS:
        raise JobScriptError(f"--scheduler must be one of {', '.join(SCHEDULERS)}, not {scheduler!r}")
    run = RunDirectory(path)
    record = run.load()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(record.get("name") or run.path.name))[:48]
    walltime = _walltime(time_limit)
    memory_gb = _memory_gb(run, memory)
    if cpus < 3:
        raise JobScriptError("ask for at least 3 CPUs: one for the run and one for each sidecar")
    prefix, pythonpath = _python_prefix()
    if python:
        prefix = [python, "-m", "cm1core"]
    logs = run.logs
    rundir = os.fspath(run.path)
    if re.search(r"[\s'\"\\$`%]", rundir):
        # #SBATCH and #PBS lines are not shell: a quoted path is taken with
        # its quotes, and Slurm reads % as a pattern (%j, %u ...).
        raise JobScriptError(
            f"the run directory's path {rundir!r} has a space or one of ' \" \\ $ ` %, which a "
            "scheduler's #SBATCH/#PBS output line cannot hold; move or link the run directory "
            "to a plain path"
        )
    q = shlex.quote

    lines = ["#!/bin/bash"]
    if scheduler == "slurm":
        gres = f"gpu:{gpu_type}:1" if gpu_type else "gpu:1"
        lines += [
            f"#SBATCH --job-name=gm1-{name}",
            f"#SBATCH --gres={gres}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={int(cpus)}",
            f"#SBATCH --mem={memory_gb}G",
            f"#SBATCH --time={walltime}",
            f"#SBATCH --output={os.fspath(logs)}/slurm-%j.out",
            f"#SBATCH --signal=B:USR1@{STOP_BEFORE_LIMIT_S}",
        ]
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if account:
            lines.append(f"#SBATCH --account={account}")
        if qos:
            lines.append(f"#SBATCH --qos={qos}")
        job_id = "${SLURM_JOB_ID:-local}"
        submit = f"sbatch {q(os.fspath(run.path / 'job-slurm.sh'))}"
    else:
        if walltime.count("-"):
            raise JobScriptError("PBS time limits are HH:MM:SS; convert days to hours")
        select = f"select=1:ncpus={int(cpus)}:ngpus=1:mem={memory_gb}gb"
        if gpu_type:
            select += f":gpu_type={gpu_type}"
        lines += [
            f"#PBS -N gm1-{name[:11]}",  # 15 characters: Torque's limit
            f"#PBS -l {select}",
            f"#PBS -l walltime={walltime}",
            "#PBS -j oe",
            f"#PBS -o {os.fspath(logs)}/pbs.out",
        ]
        if partition:
            lines.append(f"#PBS -q {partition}")
        if account:
            lines.append(f"#PBS -A {account}")
        job_id = "${PBS_JOBID:-local}"
        submit = f"qsub {q(os.fspath(run.path / 'job-pbs.sh'))}"
    install = installation()
    lines += [
        "#",
        f"# gpuwm-cm1 job script for the run directory {rundir}",
        f"# Written {utc_now()} by gpuwm-cm1 {install.get('version')} "
        f"({install.get('install')}{', ' + install['git_commit'][:12] if install.get('git_commit') else ''}).",
        f"# Submit it with:  {submit}",
        "# The same script continues the run from its last restart when submitted again.",
        "# The run owns the GPU. The two sidecars convert frames to CM1 netCDF and draw",
        "# quick-looks on this node's CPUs, never its GPU.",
        "",
        "set -u",
        f"RUNDIR={q(rundir)}",
        f"JOB_ID=\"{job_id}\"",
        "GM1=(" + " ".join(q(word) for word in prefix) + ")",
    ]
    if pythonpath:
        lines.append(f"export PYTHONPATH={q(pythonpath)}\"${{PYTHONPATH:+:$PYTHONPATH}}\"")
    for line in setup or []:
        lines.append(line)
    lines += [
        'mkdir -p "$RUNDIR/logs" "$RUNDIR/control"',
        'cd "$RUNDIR" || exit 1',
        "",
        "have() { \"${GM1[@]}\" \"$1\" --help >/dev/null 2>&1; }",
        "",
        "# A stop request for the run, only while it is running: at the end of its",
        "# current step (after any write in progress), or before it starts.",
        'request_stop() { [ -n "${RUN_PID:-}" ] && kill -0 "$RUN_PID" 2>/dev/null && : >"$RUNDIR/control/stop"; }',
        "# Near the time limit (Slurm --signal), so the next submission continues",
        "# from a complete restart.",
        'on_limit() { echo "time limit near: stop requested" >&2; request_stop; }',
        "trap on_limit USR1",
        "# Cancelled or preempted: the scheduler signals every process of the job",
        "# itself. The script asks for a clean stop and never signals the run a",
        "# second time, which would cut its own cleanup short.",
        'on_term() { echo "job signalled: stop requested" >&2; request_stop; }',
        "trap on_term TERM INT",
        "",
    ]
    run_words = ["run", "$RUNDIR", "--continue"]
    if until_s is not None:
        run_words += ["--until", f"{float(until_s):g}"]
    if library:
        run_words += ["--library", library]
    if history_publisher:
        run_words += ["--history-publisher", history_publisher]
    rendered = " ".join(
        '"$RUNDIR"' if word == "$RUNDIR" else q(word) for word in run_words
    )
    lines += [
        'SEGMENTS_BEFORE=$(ls "$RUNDIR/segments" 2>/dev/null | wc -l)',
        f'"${{GM1[@]}}" {rendered} &',
        "RUN_PID=$!",
        "",
        "# Sidecars: CPU only, lowest priority. They follow this job's segment, so",
        "# they start once the run has created it (or has ended without one).",
        "for _ in $(seq 600); do",
        '  kill -0 "$RUN_PID" 2>/dev/null || break',
        '  [ "$(ls "$RUNDIR/segments" 2>/dev/null | wc -l)" -gt "$SEGMENTS_BEFORE" ] && break',
        "  sleep 1",
        "done",
        "CM1OUT_PID=",
        "QUICKLOOK_PID=",
        "if have cm1out; then",
        '  CUDA_VISIBLE_DEVICES="" nice -n 19 "${GM1[@]}" cm1out "$RUNDIR" --follow '
        '>"$RUNDIR/logs/cm1out-$JOB_ID.log" 2>&1 &',
        "  CM1OUT_PID=$!",
        "else",
        '  echo "gpuwm-cm1 cm1out is not in this installation; no netCDF sidecar" >&2',
        "fi",
        *([
            "# Also write netCDF (settings.toml): the cm1out sidecar above writes netCDF beside the GrADS files.",
        ] if _also_netcdf(run.path) else []),
        "if have quicklook; then",
        '  CUDA_VISIBLE_DEVICES="" nice -n 19 "${GM1[@]}" quicklook "$RUNDIR" --follow '
        '>"$RUNDIR/logs/quicklook-$JOB_ID.log" 2>&1 &',
        "  QUICKLOOK_PID=$!",
        "else",
        '  echo "gpuwm-cm1 quicklook is not in this installation; no quick-look sidecar" >&2',
        "fi",
        "",
        "# wait returns early when a trapped signal arrives; wait for the run itself.",
        'while kill -0 "$RUN_PID" 2>/dev/null; do wait "$RUN_PID"; done',
        'wait "$RUN_PID"; RC=$?',
        "",
        "# The sidecars finish the last frame, then exit; give them up to 10 minutes.",
        "for pid in $CM1OUT_PID $QUICKLOOK_PID; do",
        "  for _ in $(seq 600); do kill -0 \"$pid\" 2>/dev/null || break; sleep 1; done",
        '  kill "$pid" 2>/dev/null',
        "done",
        'exit "$RC"',
        "",
    ]
    return "\n".join(lines)


def write(path: Path | str, scheduler: str, **options: Any) -> Path:
    """Write ``RUNDIR/job-<scheduler>.sh`` (mode 0755) and return its path."""

    run = RunDirectory(path)
    text = render(path, scheduler, **options)
    run.logs.mkdir(exist_ok=True)
    target = run.path / f"job-{scheduler}.sh"
    target.write_text(text, encoding="utf-8")
    target.chmod(0o755)
    return target


__all__ = ["JobScriptError", "SCHEDULERS", "render", "write"]
