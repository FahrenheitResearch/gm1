"""Keep WSL running while a run started in it runs.

Under WSL2, Windows shuts a distribution down about 15 seconds after its last
``wsl.exe`` window closes, and that ends every process in it, a run started in
its own session included.  So a run (or one of its sidecars) started under WSL
also starts a holder: a hidden Windows ``wsl.exe`` that runs
``tail --pid=PID -f /dev/null`` in the same distribution.  While it waits, the
distribution stays up; when the process ends, the holder ends too and WSL
shuts the distribution down as it would have.

PowerShell's ``Start-Process -WindowStyle Hidden`` starts the holder, so it is
a Windows process of its own (it outlives the window GM1 was started from) and
shows no window.  Each process is held once (by pid and start time), whoever
asks again.

Best effort: outside WSL nothing happens.  Under WSL without Windows interop,
or when PowerShell fails, the run is not touched and the caller gets a warning
to log.  Shutting down or restarting Windows, or ``wsl --shutdown``, still ends
a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Any, Callable, Mapping

from .rundir import process_start_ticks

#: PowerShell, where WSL mounts the Windows drive by default.
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
#: How long PowerShell may take to hand the holder to Windows.
START_TIMEOUT_S = 30.0
#: The names WSL gives distributions; anything else is not put on a command line.
DISTRO_NAME = re.compile(r"[A-Za-z0-9._-]+")
WARNING = ("GM1 could not keep WSL running for this run ({reason}). On Windows, keep the GM1 window open "
           "until it has finished: closing the window would stop it.")

_lock = threading.Lock()
_held: set[tuple[int, int | None]] = set()


def distro(env: Mapping[str, str] | None = None) -> str | None:
    """The WSL distribution this process runs in (``WSL_DISTRO_NAME``), or None outside WSL."""

    return (os.environ if env is None else env).get("WSL_DISTRO_NAME") or None


def find_powershell(env: Mapping[str, str] | None = None) -> str | None:
    """Windows PowerShell, through WSL interop: on ``PATH``, or where Windows keeps it."""

    found = shutil.which("powershell.exe", path=(os.environ if env is None else env).get("PATH"))
    if found:
        return found
    return POWERSHELL if os.access(POWERSHELL, os.X_OK) else None


def holder_command(name: str, pid: int, powershell: str) -> list[str]:
    """PowerShell starting the hidden ``wsl.exe`` that waits in distribution ``name`` for ``pid`` to end."""

    if not DISTRO_NAME.fullmatch(name):
        raise ValueError(f"unexpected WSL distribution name {name!r}")
    inside = f"-d {name} --cd / --exec tail --pid={int(pid)} -f /dev/null"
    script = f"Start-Process -WindowStyle Hidden -FilePath wsl.exe -ArgumentList '{inside}'"
    return [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script]


@dataclass
class Holder:
    """The start of one holder: the PowerShell that starts it, or why it could not be started."""

    pid: int
    command: list[str]
    process: Any = None
    failure: str = ""
    _answer: list[str | None] = field(default_factory=list, repr=False)
    _waiting: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def warning(self, timeout: float = START_TIMEOUT_S) -> str | None:
        """None once PowerShell has started the holder; otherwise the warning to log (the same on every call)."""

        with self._waiting:
            if not self._answer:
                self._answer.append(self._wait(timeout))
            return self._answer[0]

    def _wait(self, timeout: float) -> str | None:
        if self.process is None:
            return WARNING.format(reason=self.failure)
        try:
            _, err = self.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return WARNING.format(reason=f"PowerShell did not answer in {timeout:.0f} s")
        except (OSError, ValueError) as error:
            return WARNING.format(reason=str(error))
        if self.process.returncode != 0:
            text = (err or b"").decode("utf-8", "replace").strip().splitlines()
            return WARNING.format(reason=f"PowerShell exited with code {self.process.returncode}"
                                         + (f": {text[-1]}" if text else ""))
        return None


def hold(pid: int, *, report: Callable[[str], None] | None = None, env: Mapping[str, str] | None = None,
         popen: Callable[..., Any] = subprocess.Popen) -> Holder | None:
    """Under WSL, keep the distribution running until process ``pid`` ends.

    None outside WSL, or when this process is already held.  Never raises and
    never waits: with ``report``, a thread waits for PowerShell and passes the
    warning, if any, to ``report``.
    """

    name = distro(env)
    if name is None:
        return None
    key = (int(pid), process_start_ticks(pid))
    with _lock:
        if key in _held:
            return None
        _held.add(key)
    holder = Holder(int(pid), [])
    try:
        powershell = find_powershell(env)
        if powershell is None:
            raise OSError("Windows PowerShell is not reachable from WSL")
        holder.command = holder_command(name, pid, powershell)
        holder.process = popen(
            holder.command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            cwd=os.fspath(Path(powershell).parent), start_new_session=True, close_fds=True)
    except Exception as error:  # noqa: BLE001 - a run never fails over its holder
        holder.failure = str(error) or type(error).__name__
    if report is not None:
        def wait() -> None:
            warning = holder.warning()
            if warning:
                try:
                    report(warning)
                except Exception:  # noqa: BLE001 - nor over its warning
                    pass

        threading.Thread(target=wait, name=f"gm1-wsl-hold-{pid}", daemon=True).start()
    return holder


__all__ = ["DISTRO_NAME", "Holder", "POWERSHELL", "WARNING", "distro", "find_powershell", "hold",
           "holder_command"]
