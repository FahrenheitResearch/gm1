"""Under WSL a run keeps the distribution up until it ends: ``cm1run.wsl_keepalive``.

Windows shuts a WSL distribution down about 15 s after its last window closes, and that ends a run started in it.
Each run (and sidecar) started under WSL therefore starts a hidden Windows ``wsl.exe`` that waits for it with
``tail --pid``.  Here Windows is a stand-in: a ``powershell.exe`` on ``PATH`` that records its arguments and runs
the ``tail`` the real one would start in Windows, so the holder's whole life is checked on Linux.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from cm1run_support import environment, gm1, make_run, status, wait_for

from cm1core.cm1run import wsl_keepalive as K
from cm1core.gui import jobs as J

# Records its arguments; with FAKE_PS_RC=0 runs what -ArgumentList asks wsl.exe to run, detached (as Windows would).
FAKE_POWERSHELL = r"""#!/bin/sh
printf '%s\n' "$@" > "$FAKE_PS_DIR/args-$$"
rc=${FAKE_PS_RC:-0}
if [ "$rc" = 0 ]; then
  inside=$(printf '%s' "$5" | sed -n "s/.*-ArgumentList '\(.*\)'.*/\1/p")
  set -- $inside
  shift 4   # -d NAME --cd /
  [ "$1" = --exec ] && shift
  setsid "$@" < /dev/null > /dev/null 2>&1 &
  echo $! > "$FAKE_PS_DIR/holder-$$"
fi
exit "$rc"
"""


def _powershell(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    folder = tmp_path / "windows"
    folder.mkdir(exist_ok=True)
    script = folder / "powershell.exe"
    script.write_text(FAKE_POWERSHELL, encoding="ascii")
    script.chmod(0o755)
    return folder, {"WSL_DISTRO_NAME": "Ubuntu-24.04", "FAKE_PS_DIR": str(folder),
                    "PATH": f"{folder}{os.pathsep}{os.environ.get('PATH', '')}"}


def _calls(folder: Path) -> list[list[str]]:
    return [path.read_text().splitlines() for path in sorted(folder.glob("args-*"))]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(K, "_held", set())


class Recorder:
    def __init__(self, rc: int = 0, error: Exception | None = None) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.rc, self.error = rc, error

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self.error is not None:
            raise self.error
        return subprocess.Popen(["sh", "-c", f"echo nope >&2; exit {self.rc}"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)


def test_the_holder_command_is_hidden_powershell_starting_wsl_tail_on_the_pid() -> None:
    argv = K.holder_command("Ubuntu-24.04", 4321, "/mnt/c/ps.exe")
    assert argv == ["/mnt/c/ps.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                    "Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList "
                    "'-d Ubuntu-24.04 --cd / --exec tail --pid=4321 -f /dev/null'"]
    # PowerShell itself is never told to hide: it shares the GM1 window's console and would hide that window.
    assert argv.index("-Command") == 4 and "-WindowStyle" not in argv[:5]
    for name in ("x'; Remove-Item C:\\", "two words", ""):
        with pytest.raises(ValueError):
            K.holder_command(name, 1, "/mnt/c/ps.exe")


def test_under_wsl_a_holder_is_started_for_the_pid(tmp_path) -> None:
    folder, env = _powershell(tmp_path)
    popen = Recorder()
    holder = K.hold(os.getpid(), env=env, popen=popen)
    assert holder is not None and holder.warning() is None
    [(argv, kwargs)] = popen.calls
    assert argv == K.holder_command("Ubuntu-24.04", os.getpid(), str(folder / "powershell.exe"))
    assert kwargs["start_new_session"] is True and kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["cwd"] == str(folder)


def test_outside_wsl_nothing_is_started(tmp_path) -> None:
    _, env = _powershell(tmp_path)
    env.pop("WSL_DISTRO_NAME")
    popen = Recorder()
    assert K.hold(os.getpid(), env=env, popen=popen) is None and popen.calls == []
    assert K.distro({}) is None and K.distro({"WSL_DISTRO_NAME": ""}) is None


def test_a_process_is_held_once(tmp_path) -> None:
    _, env = _powershell(tmp_path)
    popen = Recorder()
    first = K.hold(os.getpid(), env=env, popen=popen)
    assert first is not None
    for _ in range(3):
        assert K.hold(os.getpid(), env=env, popen=popen) is None
    assert len(popen.calls) == 1
    # Another process (another pid) gets its own.
    other = subprocess.Popen(["sleep", "5"])
    try:
        assert K.hold(other.pid, env=env, popen=popen) is not None and len(popen.calls) == 2
    finally:
        other.kill()
        other.wait()


@pytest.mark.parametrize("case", ["no-powershell", "popen-raises", "powershell-fails", "bad-name"])
def test_a_holder_that_cannot_start_is_a_warning_never_an_error(tmp_path, monkeypatch, case) -> None:
    _, env = _powershell(tmp_path)
    popen = Recorder(rc=1 if case == "powershell-fails" else 0,
                     error=OSError(8, "Exec format error") if case == "popen-raises" else None)
    if case == "no-powershell":
        env["PATH"] = str(tmp_path / "nothing")
        monkeypatch.setattr(K, "POWERSHELL", str(tmp_path / "nothing" / "powershell.exe"))
    if case == "bad-name":
        env["WSL_DISTRO_NAME"] = "a b"
    reported: list[str] = []
    holder = K.hold(os.getpid(), env=env, popen=popen, report=reported.append)
    assert holder is not None
    warning = holder.warning()
    assert warning and warning.startswith("GM1 could not keep WSL running for this run (")
    assert "keep the GM1 window open" in warning
    wait_for(lambda: reported, timeout=10, what="the warning")
    assert reported == [warning]
    if case == "powershell-fails":
        assert "exited with code 1: nope" in warning
    # A report that itself fails is swallowed too.
    monkeypatch.setattr(K, "_held", set())
    assert K.hold(os.getpid(), env=env, popen=popen, report=lambda text: 1 / 0) is not None


def test_the_gui_holds_runs_and_sidecars_but_not_its_short_jobs(tmp_path, monkeypatch) -> None:
    from cm1core.cm1run import wsl_keepalive

    held: list[int] = []
    monkeypatch.setattr(wsl_keepalive, "hold", lambda pid, **kwargs: held.append(pid))
    runner = J.Runner("fake", (sys.executable, "-c", "import time; time.sleep(0.2)"))
    jobs = J.Jobs(runner, tmp_path / "server")
    rundir = tmp_path / "run"
    rundir.mkdir()
    try:
        rows = {role: jobs.launch("run", rundir, args, role=role) for role, args in (
            ("run", ["run", str(rundir)]), ("resume", ["resume", str(rundir)]),
            ("cm1out", ["cm1out", str(rundir), "--follow"]), ("quicklook", ["quicklook", str(rundir), "--follow"]),
            ("measure", ["estimate", str(rundir), "--measure"]))}
        assert held == [rows[role].process.pid for role in ("run", "resume", "cm1out", "quicklook")]
    finally:
        jobs.close()


def test_a_gui_run_whose_holder_fails_still_runs_and_the_log_says_why(tmp_path, monkeypatch) -> None:
    folder, env = _powershell(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FAKE_PS_RC", "3")
    runner = J.Runner("fake", (sys.executable, "-c", "import time; time.sleep(1.5)"))
    jobs = J.Jobs(runner, tmp_path / "server")
    rundir = tmp_path / "run"
    rundir.mkdir()
    try:
        row = jobs.launch("run", rundir, ["run", str(rundir)], role="run")
        assert row.alive
        wait_for(lambda: "could not keep WSL running" in row.log.read_text(), timeout=20, what="the warning")
        assert "PowerShell exited with code 3" in row.log.read_text()
        assert row.process.wait(timeout=20) == 0
    finally:
        jobs.close()


def test_a_detached_run_under_wsl_is_held_until_it_ends(tmp_path) -> None:
    folder, extra = _powershell(tmp_path)
    run = make_run(tmp_path, duration=600.0, restart=600.0)
    env = environment(tmp_path, FAKE_STEP_S="0.03", **extra)
    completed = gm1("run", str(run), "--detach", "--json", env=env)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert "warning" not in result
    [argv] = _calls(folder)
    assert argv[-1].endswith(f"'-d Ubuntu-24.04 --cd / --exec tail --pid={result['pid']} -f /dev/null'")
    [holder_file] = folder.glob("holder-*")
    holder = int(holder_file.read_text())
    assert _alive(holder)
    wait_for(lambda: status(run, env)["state"] == "finished", timeout=60, what="the detached run")
    # The holder ends with the run: WSL may then shut the distribution down as before.
    wait_for(lambda: not _alive(holder), timeout=15, what="the holder to end")


def test_a_detached_run_whose_holder_fails_starts_and_says_so(tmp_path) -> None:
    _, extra = _powershell(tmp_path)
    run = make_run(tmp_path, duration=360.0)
    env = environment(tmp_path, FAKE_PS_RC="1", **extra)
    completed = gm1("run", str(run), "--detach", "--json", env=env)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["started"] is True and "could not keep WSL running" in result["warning"]
    assert result["warning"] in result["message"]
    wait_for(lambda: status(run, env)["state"] == "finished", timeout=60, what="the detached run")
    assert shlex.split(result["command"])[:2] == ["gpuwm-cm1", "run"]
