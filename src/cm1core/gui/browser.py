"""Open the GUI's link in the user's own browser, when there is one to open it in.

``gpuwm-cm1 gui`` always prints its link; this opens it too, so a person who
double-clicked the GM1 shortcut lands on the page.  Where it opens:

- WSL2: the Windows browser, through ``wslview`` when it is installed, else
  ``explorer.exe``, else ``cmd.exe /c start`` (WSL2 forwards Windows'
  ``127.0.0.1`` to the server inside it);
- a Linux desktop (``DISPLAY`` or ``WAYLAND_DISPLAY`` set): Python's
  :mod:`webbrowser`, which uses ``xdg-open`` and the desktop's default browser.

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
import subprocess
import sys
from typing import Callable, Mapping, Sequence

NO_BROWSER_ENV = "GPUWM_CM1_NO_BROWSER"
SSH_ENV = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
#: Where WSL finds Windows' programs when its PATH does not include them.
WINDOWS_EXPLORER = Path("/mnt/c/Windows/explorer.exe")
WINDOWS_CMD = Path("/mnt/c/Windows/System32/cmd.exe")


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


#: How long a helper that can fail after it starts (wslview, when WSL interop is off) gets to say so.
CHECK_S = 2.0


def _start(words: Sequence[str], *, popen: Callable[..., object], cwd: str | None = None,
           check_s: float = 0.0) -> bool:
    try:
        process = popen(list(words), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        start_new_session=True, close_fds=True, cwd=cwd)
    except (OSError, ValueError):
        return False
    wait = getattr(process, "wait", None)
    if check_s > 0 and callable(wait):
        try:
            code = wait(timeout=check_s)
        except subprocess.TimeoutExpired:
            return True  # still running: it is opening the browser
        except Exception:  # noqa: BLE001 - a helper that cannot be waited on is taken as started
            return True
        return code == 0
    return True


def wsl_openers(which: Callable[[str], str | None] = shutil.which,
                exists: Callable[[Path], bool] = Path.exists) -> list[tuple[list[str], str | None]]:
    """The ways to reach the Windows browser from WSL2, best first: (words before the link, working folder)."""

    openers: list[tuple[list[str], str | None]] = []
    wslview = which("wslview")
    if wslview:
        openers.append(([wslview], None))
    explorer = which("explorer.exe") or (str(WINDOWS_EXPLORER) if exists(WINDOWS_EXPLORER) else None)
    if explorer:
        openers.append(([explorer], None))
    cmd = which("cmd.exe") or (str(WINDOWS_CMD) if exists(WINDOWS_CMD) else None)
    if cmd:
        # cmd.exe complains about a Linux working folder; start it on the Windows drive.
        # The empty argument is start's window title, so the link is not taken for one.
        openers.append(([cmd, "/c", "start", ""], "/mnt/c" if exists(Path("/mnt/c")) else None))
    return openers


def open_link(url: str, *, env: Mapping[str, str] | None = None, wsl: bool | None = None,
              terminal: bool | None = None, platform: str | None = None,
              which: Callable[[str], str | None] = shutil.which, exists: Callable[[Path], bool] = Path.exists,
              popen: Callable[..., object] = subprocess.Popen,
              webbrowser_open: Callable[[str], bool] | None = None) -> tuple[bool, str]:
    """Open ``url``; (opened, how or why not).  Never raises and never waits for the browser."""

    reason = why_not(env, wsl=wsl, terminal=terminal, platform=platform)
    if reason is not None:
        return False, reason
    if wsl is None:
        from ..cm1run.gpuinfo import is_wsl2

        wsl = is_wsl2()
    if wsl:
        for words, cwd in wsl_openers(which, exists):
            name = Path(words[0]).name
            # wslview exits non-zero at once when it cannot reach Windows: then the next one is tried.
            if _start([*words, url], popen=popen, cwd=cwd, check_s=CHECK_S if name == "wslview" else 0.0):
                return True, name
        return False, "no Windows program to open it with was found"
    if webbrowser_open is None:
        import webbrowser

        webbrowser_open = webbrowser.open
    try:
        opened = bool(webbrowser_open(url))
    except Exception:  # noqa: BLE001 - a broken browser setup leaves the printed link
        opened = False
    return (True, "webbrowser") if opened else (False, "no browser answered")


__all__ = ["NO_BROWSER_ENV", "open_link", "why_not", "wsl_openers"]
