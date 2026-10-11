"""``gpuwm-cm1 jobscript``: Slurm and PBS scripts with compute-node sidecars (4.6).

The scripts are checked with ``bash -n`` and then executed with plain bash
(no scheduler) against the stand-in integration: the run starts, a second
submission continues it, the time-limit signal stops it cleanly, and the
sidecars are started CPU-only at nice 19.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest

from cm1run_support import environment, gm1, make_run, progress_at_least, status, wait_for

from cm1core.cm1run.jobscript import JobScriptError, render

SIDECAR_LINES = {
    "cm1out": 'CUDA_VISIBLE_DEVICES="" nice -n 19 "${GM1[@]}" cm1out "$RUNDIR" --follow',
    "quicklook": 'CUDA_VISIBLE_DEVICES="" nice -n 19 "${GM1[@]}" quicklook "$RUNDIR" --follow',
}


@pytest.mark.parametrize("scheduler", ["slurm", "pbs"])
def test_the_script_is_valid_bash_with_cpu_only_sidecars(tmp_path: Path, scheduler: str) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path)
    completed = gm1("jobscript", str(run), "--scheduler", scheduler, "--partition", "gpu",
                    "--account", "abc123", "--time", "02:00:00", env=env)
    assert completed.returncode == 0, completed.stderr
    script = completed.stdout
    path = tmp_path / f"job-{scheduler}.sh"
    path.write_text(script)
    checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    for line in SIDECAR_LINES.values():
        assert line in script
    assert '"${GM1[@]}" run "$RUNDIR" --continue &' in script
    if scheduler == "slurm":
        assert "#SBATCH --gres=gpu:1" in script and "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --signal=B:USR1@300" in script and "#SBATCH --partition=gpu" in script
        assert "#SBATCH --account=abc123" in script
    else:
        assert "#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb" in script and "#PBS -l walltime=02:00:00" in script
        assert "#PBS -q gpu" in script and "#PBS -A abc123" in script
    assert (run / "logs").is_dir()


def test_bad_options_are_refused(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    env = environment(tmp_path)
    assert gm1("jobscript", str(run), "--scheduler", "lsf", env=env).returncode == 2
    assert gm1("jobscript", str(run), "--scheduler", "slurm", "--time", "two hours", env=env).returncode == 2
    with pytest.raises(JobScriptError):
        render(run, "slurm", cpus=2)
    with pytest.raises(JobScriptError):
        render(run, "pbs", time_limit="1-00:00:00")
    # #SBATCH/#PBS lines are not shell: a path they cannot hold is refused, not mangled.
    spaced = make_run(tmp_path, name="my run")
    with pytest.raises(JobScriptError, match="space"):
        render(spaced, "slurm")
    percent = make_run(tmp_path, name="run%1")
    with pytest.raises(JobScriptError):
        render(percent, "slurm")


def _wrapped_python(tmp_path: Path) -> Path:
    """A 'python' for the script that answers the two sidecar commands itself.

    It records the environment each sidecar was started with, so the test sees
    the sidecars run CPU-only at nice 19, and hands every other command to the
    real interpreter.
    """

    record = tmp_path / "sidecars.txt"
    wrapper = tmp_path / "bin" / "python-on-node"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/bin/bash\n"
        'case "$3" in\n'
        "  cm1out|quicklook)\n"
        '    [ "$4" = --help ] && exit 0\n'
        f'    echo "$3 cvd=[${{CUDA_VISIBLE_DEVICES-unset}}] nice=$(nice) '
        f'segments=$(ls "$4/segments" 2>/dev/null | wc -l) args=$4,$5" >> {record}\n'
        "    exit 0;;\n"
        "esac\n"
        f'exec {sys.executable} "$@"\n',
        encoding="ascii",
    )
    wrapper.chmod(0o755)
    return wrapper


def test_the_script_runs_continues_and_stops_before_the_limit(tmp_path: Path) -> None:
    if shutil.which("bash") is None:
        pytest.skip("no bash")
    run = make_run(tmp_path, duration=1800.0, restart=600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03", CUDA_VISIBLE_DEVICES="0")
    python = _wrapped_python(tmp_path)
    written = gm1("jobscript", str(run), "--scheduler", "slurm", "--python", str(python), "--write", env=env)
    assert written.returncode == 0, written.stderr
    script = run / "job-slurm.sh"
    assert os.access(script, os.X_OK)
    # The time limit approaches: Slurm sends the batch shell USR1.
    job = subprocess.Popen(["bash", str(script)], env={**env, "SLURM_JOB_ID": "777"},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    wait_for(lambda: progress_at_least(run, 110), timeout=60, what="stepping past the first restart")
    os.kill(job.pid, signal.SIGUSR1)
    output = job.communicate(timeout=60)[0].decode()
    assert job.returncode == 0, output
    assert "stop requested" in output
    answer = status(run, env)
    assert answer["state"] == "stopped" and answer["scheduler"]["job_id"] == "777"
    sidecars = (tmp_path / "sidecars.txt").read_text().splitlines()
    assert sorted(line.split()[0] for line in sidecars) == ["cm1out", "quicklook"]
    for line in sidecars:
        assert "cvd=[]" in line and "nice=19" in line and line.endswith(f"args={run},--follow")
    # Submitted again, the same script continues from the last restart.
    again = subprocess.run(["bash", str(script)], env={**env, "SLURM_JOB_ID": "778"},
                           capture_output=True, text=True, timeout=120)
    assert again.returncode == 0, again.stdout + again.stderr
    answer = status(run, env)
    assert answer["state"] == "finished"
    # The second job's sidecars started once its own segment existed, so they
    # follow segment 2 and do not take segment 1's end record for theirs.
    later = (tmp_path / "sidecars.txt").read_text().splitlines()[len(sidecars):]
    assert sorted(line.split()[0] for line in later) == ["cm1out", "quicklook"]
    assert all("segments=2 " in line for line in later), later
    record = json.loads((run / "run.json").read_text())
    assert [row["kind"] for row in record["segments"]] == ["fresh", "resume"]
    assert record["segments"][1]["scheduler"]["job_id"] == "778"
    # A third submission finds nothing left and exits 0.
    third = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert third.returncode == 0 and "nothing left" in third.stdout + third.stderr
