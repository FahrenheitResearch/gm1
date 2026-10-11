"""Open the GUI's link in the user's own browser, when there is one to open it in.

``gpuwm-cm1 gui`` always prints its link; this opens it too, so a person who
double-clicked the GM1 shortcut lands on the page.  Where it opens:

- WSL2: the Windows default browser (WSL2 forwards Windows' ``127.0.0.1`` to
  the server inside it), through the first of these that exits with success:

  1. ``rundll32.exe url.dll,FileProtocolHandler LINK``: hands the link to
     Windows' own handler for it.  It takes the link as written (``?``, ``=``
     and ``&`` included), needs no console and no Windows working folder, and
     exits in a fraction of a second;
  2. ``cmd.exe /c start "" LINK``, started on the Windows drive (cmd.exe
     complains about a Linux working folder).  Not used for a link with a
     character cmd.exe reads as its own (``&`` would cut the link short);
  3. ``powershell.exe -NoProfile -Command Start-Process 'LINK'``;
  4. ``wslview LINK`` when it is installed.

  ``explorer.exe LINK`` is never used for a link: given a link with a query
  (``?token=...``) it opens a File Explorer window on Documents instead of
  the browser, and it exits 1 whether it worked or not.
- a Linux desktop (``DISPLAY`` or ``WAYLAND_DISPLAY`` set): Python's
  :mod:`webbrowser`, which uses ``xdg-open`` and the desktop's default browser.

A helper is waited for, a few seconds at most, and only its exit with success
counts as opened: the caller says "Opened" for nothing less.  A helper that
has not ended by then is killed with its whole session, so none stays behind
to keep WSL up, and none holds this process's output (its three streams are
the null device).  After a helper that did not answer in time no other is
tried: a slow computer may have opened the page all the same, and a second
helper would open a second tab.

:func:`open_folder` shows a folder in the file manager: File Explorer under
WSL2 (here ``explorer.exe`` is the right program, given the folder's Windows
name), ``xdg-open`` on a Linux desktop.

Never on a session that has no screen of its own: an ssh login
(``SSH_CONNECTION``, ``SSH_CLIENT`` or ``SSH_TTY`` set), a Linux session with
no display, or output that is not a terminal (a script, a test, a job), and
never with ``--no-browser`` or ``GPUWM_CM1_NO_BROWSER=1``.  Only the standard
library; a browser that cannot be started leaves the printed link, nothing
else.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Callable, Mapping, Sequence

NO_BROWSER_ENV = "GPUWM_CM1_NO_BROWSER"
SSH_ENV = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
#: Where WSL finds Windows' programs when its PATH does not include them.
WINDOWS_SYSTEM = Path("/mnt/c/Windows/System32")
WINDOWS_PROGRAMS = {
    "rundll32.exe": WINDOWS_SYSTEM / "rundll32.exe",
    "cmd.exe": WINDOWS_SYSTEM / "cmd.exe",
    "powershell.exe": WINDOWS_SYSTEM / "WindowsPowerShell" / "v1.0" / "powershell.exe",
    "explorer.exe": Path("/mnt/c/Windows/explorer.exe"),
}
#: Characters cmd.exe reads as its own; a link holding one is not handed to ``cmd.exe /c start``.
CMD_SPECIAL = frozenset('&|<>^%"\'()! \t\r\n')
#: How long a helper gets to exit.  Measured on Windows 11: rundll32 0.1 to 0.2 s, cmd 0.1 s, PowerShell 0.3 s.
WAIT_S = 6.0


def why_not(env: Mapping[str, str] | None = None, *, wsl: bool | None = None, terminal: bool | None = None,
            platform: str | None = None) -> str | None:
    """Why the link is not opened here, or None when it is."""

    env = os.environ if env is None else env
    if env.get(NO_BROWSER_ENV, "").strip() not in ("", "0"):
        return f"{NO_BROWSER_ENV} is set"
    if terminal is None:
        try:
            terminal = sys.stdout.isatty()
        except (AttributeError, ValueError):
            terminal = False
    if not terminal:
        return "the output is not a terminal"
    if any(env.get(name) for name in SSH_ENV):
        return "this is an ssh session"
    if wsl is None:
        from ..cm1run.gpuinfo import is_wsl2

        wsl = is_wsl2()
    if wsl:
        return None
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux") and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return "there is no display"
    return None


def _end(process: object) -> None:
    """Kill a helper that did not exit, with its whole session, and reap it."""

    pid = getattr(process, "pid", None)
    try:
        if isinstance(pid, int) and pid > 0:
            os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()  # type: ignore[attr-defined]
    except (OSError, AttributeError):
        try:
            process.kill()  # type: ignore[attr-defined]
        except (OSError, AttributeError):
            pass
    try:
        process.wait(timeout=2.0)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - nothing more can be done for a helper that will not be reaped
        pass


def helper_outcome(words: Sequence[str], *, popen: Callable[..., object] = subprocess.Popen,
                   cwd: str | None = None, wait_s: float = WAIT_S) -> tuple[str, int | None]:
    """Run a helper detached from this process and wait for it: (what happened, its exit code).

    What happened is ``"exited"`` (with the code), ``"timeout"`` (it had not exited after ``wait_s``; it is
    then killed, so it cannot stay behind) or ``"failed"`` (it could not be started or waited for).  Its
    input, output and errors are the null device, it has its own session, and it inherits no other
    descriptor, so it can neither hold this process's output nor be stopped with it.
    """

    try:
        process = popen(list(words), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True, close_fds=True, cwd=cwd)
    except (OSError, ValueError):
        return "failed", None
    try:
        code = process.wait(timeout=wait_s)  # type: ignore[attr-defined]
    except subprocess.TimeoutExpired:
        _end(process)
        return "timeout", None
    except Exception:  # noqa: BLE001 - a helper that cannot be waited on did not report success
        _end(process)
        return "failed", None
    return ("exited", int(code)) if isinstance(code, int) else ("failed", None)


def run_helper(words: Sequence[str], *, popen: Callable[..., object] = subprocess.Popen, cwd: str | None = None,
               wait_s: float = WAIT_S) -> int | None:
    """:func:`helper_outcome` as one value: the helper's exit code, or None when it did not exit by itself."""

    return helper_outcome(words, popen=popen, cwd=cwd, wait_s=wait_s)[1]


def _windows_program(name: str, which: Callable[[str], str | None], exists: Callable[[Path], bool]) -> str | None:
    found = which(name)
    if found:
        return found
    usual = WINDOWS_PROGRAMS[name]
    return str(usual) if exists(usual) else None


def wsl_openers(url: str, which: Callable[[str], str | None] = shutil.which,
                exists: Callable[[Path], bool] = Path.exists) -> list[tuple[str, list[str], str | None]]:
    """The ways to open ``url`` in the Windows default browser from WSL2, most reliable first.

    Each is (name, the whole command, working folder).  The order is the one measured on Windows 11 with the
    link GM1 prints (see the module's words): rundll32, cmd's start, PowerShell's Start-Process, wslview.
    """

    windows_folder = "/mnt/c" if exists(Path("/mnt/c")) else None
    openers: list[tuple[str, list[str], str | None]] = []
    rundll32 = _windows_program("rundll32.exe", which, exists)
    if rundll32:
        openers.append(("rundll32.exe", [rundll32, "url.dll,FileProtocolHandler", url], windows_folder))
    cmd = _windows_program("cmd.exe", which, exists)
    if cmd and not (set(url) & CMD_SPECIAL):
        # The empty argument is start's window title, so the link is not taken for one.
        openers.append(("cmd.exe", [cmd, "/c", "start", "", url], windows_folder))
    powershell = _windows_program("powershell.exe", which, exists)
    if powershell and "'" not in url:
        openers.append(("powershell.exe", [powershell, "-NoProfile", "-NonInteractive", "-Command",
                                           f"Start-Process '{url}'"], windows_folder))
    wslview = which("wslview")
    if wslview:
        openers.append(("wslview", [wslview, url], None))
    return openers


def open_link(url: str, *, env: Mapping[str, str] | None = None, wsl: bool | None = None,
              terminal: bool | None = None, platform: str | None = None,
              which: Callable[[str], str | None] = shutil.which, exists: Callable[[Path], bool] = Path.exists,
              popen: Callable[..., object] = subprocess.Popen,
              webbrowser_open: Callable[[str], bool] | None = None, wait_s: float = WAIT_S) -> tuple[bool, str]:
    """Open ``url``; (opened, how or why not).  Never raises.

    ``opened`` is True only when the program asked to open it exited with success.
    """

    reason = why_not(env, wsl=wsl, terminal=terminal, platform=platform)
    if reason is not None:
        return False, reason
    if wsl is None:
        from ..cm1run.gpuinfo import is_wsl2

        wsl = is_wsl2()
    if wsl:
        openers = wsl_openers(url, which, exists)
        if not openers:
            return False, "no Windows program to open it with was found"
        failed = []
        for name, words, cwd in openers:
            what, code = helper_outcome(words, popen=popen, cwd=cwd, wait_s=wait_s)
            if what == "exited" and code == 0:
                return True, name
            if what == "timeout":
                # It may have opened the page before it was ended: another helper would open a second tab.
                return False, f"{name} did not answer in {wait_s:.0f} s; the page may have opened all the same"
            failed.append(name)
        return False, f"Windows did not open it ({', '.join(failed)} tried)"
    if webbrowser_open is None:
        import webbrowser

        webbrowser_open = webbrowser.open
    try:
        opened = bool(webbrowser_open(url))
    except Exception:  # noqa: BLE001 - a broken browser setup leaves the printed link
        opened = False
    return (True, "webbrowser") if opened else (False, "no browser answered")


def folder_opener(*, env: Mapping[str, str] | None = None, wsl: bool | None = None, platform: str | None = None,
                  which: Callable[[str], str | None] = shutil.which,
                  exists: Callable[[Path], bool] = Path.exists) -> tuple[str, str] | None:
    """(name, program) of what shows a folder on this computer's screen, or None when nothing can.

    File Explorer under WSL2, ``xdg-open`` on a Linux desktop; never over ssh (the window would open on a
    screen the person is not at).
    """

    env = os.environ if env is None else env
    if any(env.get(name) for name in SSH_ENV):
        return None
    if wsl is None:
        from ..cm1run.gpuinfo import is_wsl2

        wsl = is_wsl2()
    if wsl:
        explorer = _windows_program("explorer.exe", which, exists)
        return ("explorer.exe", explorer) if explorer else None
    platform = sys.platform if platform is None else platform
    if platform.startswith("linux") and (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        opener = which("xdg-open")
        return ("xdg-open", opener) if opener else None
    return None


def open_folder(path: str | os.PathLike[str], *, windows: str | None = None, env: Mapping[str, str] | None = None,
                wsl: bool | None = None, platform: str | None = None,
                which: Callable[[str], str | None] = shutil.which, exists: Callable[[Path], bool] = Path.exists,
                popen: Callable[..., object] = subprocess.Popen, wait_s: float = WAIT_S) -> tuple[bool, str]:
    """Show the folder ``path`` in the file manager; (started, the program or why not).  Never raises.

    ``windows``: the folder's Windows name (:func:`cm1core.cm1run.wslpaths.windows_path`), which File
    Explorer is given under WSL2.  explorer.exe exits 1 even when its window opened, so here a helper that
    started and exited counts as started.
    """

    found = folder_opener(env=env, wsl=wsl, platform=platform, which=which, exists=exists)
    if found is None:
        return False, "this computer has no screen of its own to show a folder on"
    name, program = found
    if name == "explorer.exe":
        if not windows:
            return False, "the folder's Windows name could not be worked out"
        words, cwd = [program, windows], ("/mnt/c" if exists(Path("/mnt/c")) else None)
    else:
        words, cwd = [program, os.fspath(path)], None
    code = run_helper(words, popen=popen, cwd=cwd, wait_s=wait_s)
    if code is None:
        return False, f"{name} did not start"
    if name != "explorer.exe" and code != 0:
        return False, f"{name} exited {code}"
    return True, name


__all__ = ["NO_BROWSER_ENV", "WAIT_S", "folder_opener", "helper_outcome", "open_folder", "open_link", "run_helper",
           "why_not", "wsl_openers"]
