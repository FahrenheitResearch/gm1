"""Under WSL2: the Windows name of a Linux path, and the Windows drive that holds the Linux disk.

A person on Windows finds their storms in File Explorer, not at ``~/gm1-runs``.  Windows reaches
that folder as ``\\\\wsl.localhost\\DISTRO\\home\\NAME\\gm1-runs``; :func:`windows_path` gives that name
(``wslpath -w`` when it answers, else built from ``WSL_DISTRO_NAME``).

The Linux disk of a WSL2 distro is one file on a Windows drive, and it can claim far more room than that
drive has left.  :func:`windows_drive` finds the drive (the distro's ``BasePath`` in the Windows registry,
read with ``reg.exe``) and its free space (the drive as WSL mounts it under ``/mnt``).

Only the standard library; every helper answers None instead of raising, and none runs outside WSL2.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping

DISTRO_ENV = "WSL_DISTRO_NAME"
#: Where WSL finds Windows' programs when its PATH does not include them.
WINDOWS_SYSTEM = Path("/mnt/c/Windows/System32")
LXSS_KEY = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss"
HELPER_TIMEOUT_S = 4.0
_MOUNTED_DRIVE = re.compile(r"^/mnt/([a-zA-Z])(/.*)?$")
_DISTRO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _is_wsl(wsl: bool | None) -> bool:
    if wsl is not None:
        return bool(wsl)
    from .gpuinfo import is_wsl2

    return is_wsl2()


def _run_text(words: list[str], run: Callable[..., Any], cwd: str | None = None) -> str | None:
    try:
        done = run(words, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                   timeout=HELPER_TIMEOUT_S, cwd=cwd, start_new_session=True)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if getattr(done, "returncode", 1) != 0:
        return None
    out = done.stdout
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    return str(out or "")


def built_windows_path(path: str, distro: str | None) -> str | None:
    """The Windows name of an absolute Linux path, from the path alone (no helper program)."""

    if not path.startswith("/"):
        return None
    drive = _MOUNTED_DRIVE.match(path)
    if drive:
        return f"{drive.group(1).upper()}:" + (drive.group(2) or "/").replace("/", "\\")
    if not distro or not _DISTRO_NAME.match(distro):
        return None
    return f"\\\\wsl.localhost\\{distro}" + path.rstrip("/").replace("/", "\\")


def windows_path(path: str | os.PathLike[str], *, env: Mapping[str, str] | None = None, wsl: bool | None = None,
                 run: Callable[..., Any] = subprocess.run,
                 which: Callable[[str], str | None] = shutil.which) -> str | None:
    """What Windows calls ``path`` (``\\\\wsl.localhost\\DISTRO\\...`` or ``C:\\...``), or None outside WSL2."""

    if not _is_wsl(wsl):
        return None
    env = os.environ if env is None else env
    text = os.fspath(path)
    if not text.startswith("/"):
        return None
    wslpath = which("wslpath")
    if wslpath:
        answer = (_run_text([wslpath, "-w", text], run) or "").strip()
        if answer and "\n" not in answer and (answer.startswith("\\\\") or re.match(r"^[A-Za-z]:\\", answer)):
            return answer
    return built_windows_path(text, env.get(DISTRO_ENV))


def _program(name: str, which: Callable[[str], str | None], exists: Callable[[Path], bool]) -> str | None:
    found = which(name)
    if found:
        return found
    usual = WINDOWS_SYSTEM / name
    return str(usual) if exists(usual) else None


def base_path(listing: str, distro: str) -> str | None:
    """The ``BasePath`` of ``distro`` in the output of ``reg.exe query ...\\Lxss /s``."""

    name = path = None
    for line in listing.splitlines() + ["HKEY_END"]:
        words = line.split(None, 2)
        if line.startswith("HKEY_"):
            if name == distro and path:
                return path
            name = path = None
        elif len(words) == 3 and words[1] == "REG_SZ":
            if words[0] == "DistributionName":
                name = words[2].strip()
            elif words[0] == "BasePath":
                path = words[2].strip()
    return None


def windows_drive(*, env: Mapping[str, str] | None = None, wsl: bool | None = None,
                  run: Callable[..., Any] = subprocess.run, which: Callable[[str], str | None] = shutil.which,
                  exists: Callable[[Path], bool] = Path.exists,
                  usage: Callable[[str], Any] = shutil.disk_usage,
                  ismount: Callable[[str], bool] = os.path.ismount) -> dict[str, Any] | None:
    """The Windows drive that holds this distro's Linux disk: ``{"drive": "C:", "free_bytes", "total_bytes"}``.

    None when it cannot be found cheaply (not WSL2, no distro name, ``reg.exe`` does not answer, or the drive
    is not mounted under ``/mnt``).
    """

    if not _is_wsl(wsl):
        return None
    env = os.environ if env is None else env
    distro = env.get(DISTRO_ENV)
    if not distro:
        return None
    reg = _program("reg.exe", which, exists)
    if not reg:
        return None
    # A Windows program complains about a Linux working folder; start it on the Windows drive.
    cwd = "/mnt/c" if exists(Path("/mnt/c")) else None
    listing = _run_text([reg, "query", LXSS_KEY, "/s"], run, cwd)
    if not listing:
        return None
    base = base_path(listing, distro)
    if not base:
        return None
    match = re.match(r"^(?:\\\\\?\\)?([A-Za-z]):\\", base)
    if not match:
        return None
    letter = match.group(1)
    mount = f"/mnt/{letter.lower()}"
    try:
        if not ismount(mount):
            return None  # the drive is not mounted here: its folder would read as the Linux disk
        found = usage(mount)
        free, total = int(found.free), int(found.total)
    except (OSError, AttributeError, TypeError, ValueError):
        return None
    return {"drive": f"{letter.upper()}:", "free_bytes": free, "total_bytes": total}


__all__ = ["DISTRO_ENV", "base_path", "built_windows_path", "windows_drive", "windows_path"]
