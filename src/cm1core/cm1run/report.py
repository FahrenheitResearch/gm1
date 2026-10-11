"""``gpuwm-cm1 doctor --report``: everything a bug report needs, as one Markdown block to paste.

The versions (GM1, ArWen, CuPy, NVRTC, numpy, Python), the computer (OS, WSL2,
glibc, the NVIDIA driver, each GPU's name, architecture and memory), how GM1
was installed, every ``doctor --plain`` row with its fix, and the newest run in
the runs folder: its state, its error and the end of its log when it failed.

Nothing in it names the person or the computer: the home folder reads ``~``,
any other user folder ``<user>``, and the host name ``<host>``.  Like
``doctor --plain`` it never imports CuPy or numpy and never touches the GPU.
"""

from __future__ import annotations

from collections import deque
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import sys
from typing import Any, Mapping, Sequence

from . import gpuinfo
from .system import system_report

REPORT_TITLE = "GM1 doctor report"
#: The end of a failed run's log that the report carries.
LOG_TAIL_LINES = 25
LOG_LINE_CHARS = 300
#: Where Linux keeps the users' home folders (spelled in parts, so the source names no home path).
HOMES = "/".join(("", "home", ""))


# ---------------------------------------------------------------- privacy

def _host_names() -> list[str]:
    names = set()
    try:
        name = socket.gethostname()
    except OSError:
        name = ""
    for candidate in (name, name.split(".", 1)[0]):
        if len(candidate) >= 3:
            names.add(candidate)
    return sorted(names, key=len, reverse=True)


#: Where a folder name ends: a slash, a space, the end, punctuation, or a full stop that ends a sentence.
_NAME_END = r"(?=$|[/\\\s|`'\"(),;:]|\.(?:$|\s))"
#: A user name in a path, up to where it ends.
_NAME = r"[^/\\\s|`'\"(),;:]+?" + _NAME_END


def scrub(text: str, *, home: str | None = None, hosts: Sequence[str] | None = None) -> str:
    """``text`` with the home folder as ``~``, other user folders as ``<user>`` and host names as ``<host>``."""

    home = str(Path.home()) if home is None else home
    hosts = _host_names() if hosts is None else hosts
    if home and home not in ("/", ""):
        text = re.sub(re.escape(home.rstrip("/")) + _NAME_END, "~", text)
    # Another user's folder (a run folder under someone else's home, a Windows profile seen from WSL).
    text = re.sub(r"(?<![\w.])(" + re.escape(HOMES) + r")" + _NAME, r"\1<user>", text)
    text = re.sub(r"(?i)(/mnt/[a-z]/users/)" + _NAME, r"\1<user>", text)
    text = re.sub(r"(?i)(\b[a-z]:\\+users\\+)" + _NAME, r"\1<user>", text)
    # The runs folder as Windows names it under WSL: \\wsl.localhost\DISTRO\home\NAME\...
    text = re.sub(r"(?i)(\\wsl(?:\.localhost|\$)\\+[^\\\s]+\\+home\\+)" + _NAME, r"\1<user>", text)
    text = re.sub(r"(?<![\w.])(/Users/)" + _NAME, r"\1<user>", text)
    for host in hosts:
        text = re.sub(r"(?<![\w-])" + re.escape(host) + r"(?![\w-])", "<host>", text)
    return text


# ---------------------------------------------------------------- facts

def _version(*names: str) -> str:
    for name in names:
        try:
            return f"{name} {metadata.version(name)}"
        except metadata.PackageNotFoundError:
            continue
    return "not installed"


def _cupy() -> str:
    from .system import _cupy_distribution

    found = _cupy_distribution()
    return "not installed" if found is None else f"{found[0]} {found[1]}"


def _gm1_version() -> str:
    try:
        from .._version import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


def os_words() -> str:
    pretty = ""
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                pretty = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pretty = ""
    kernel = f"{platform.system()} {platform.release()} {platform.machine()}".strip()
    return f"{pretty}, {kernel}" if pretty else kernel


def glibc_words() -> str:
    library, version = platform.libc_ver()
    return f"{library} {version}".strip() if version else "unknown"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: Sequence[tuple[str, Any]]) -> list[str]:
    lines = ["| | |", "|---|---|"]
    lines += [f"| {_cell(key)} | {_cell(value)} |" for key, value in rows]
    return lines


# ---------------------------------------------------------------- the newest run

def _run_time(rundir: Path) -> float:
    newest = 0.0
    for path in [rundir / "run.json", rundir / "events.jsonl", *sorted((rundir / "segments").glob("*/events.jsonl"))]:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def last_run(root: Path) -> dict[str, Any] | None:
    """The newest run under ``root``: its name, status and, when it failed or stopped, the end of its log."""

    from ..gui.files import Root
    from ..gui.runstate import read_status, segment_dirs

    runs_root = Root(root)
    try:
        runs = list(runs_root.iter_runs())
    except OSError:
        return None
    if not runs:
        return None
    rundir = max(runs, key=_run_time)
    try:
        status = read_status(rundir, scheduler=lambda _job: "unknown")
    except Exception as error:  # noqa: BLE001 - a damaged run folder is itself worth reporting
        return {"name": runs_root.run_id(rundir), "state": "unreadable", "label": f"Could not be read: {error}"}
    found: dict[str, Any] = {"name": runs_root.run_id(rundir), "state": status.state, "label": status.label}
    if status.detail.get("message"):
        found["message"] = str(status.detail["message"])
    if status.percent is not None:
        found["percent"] = status.percent
    if status.state in ("failed", "stopped", "stale"):
        segments = segment_dirs(rundir)
        log = segments[-1] / "run.log" if segments else None
        if log is not None and log.is_file():
            try:
                with open(log, encoding="utf-8", errors="replace") as handle:
                    tail = deque(handle, maxlen=LOG_TAIL_LINES)
                found["log_tail"] = [line.rstrip("\n")[:LOG_LINE_CHARS] for line in tail]
            except OSError:
                pass
    return found


def _browser(root: Path) -> str | None:
    """How the GUI last opened its link (``gpuwm-cm1 gui`` records it in the runs folder), or None."""

    from ..gui.files import SERVER_DIR

    try:
        record = json.loads((Path(root) / SERVER_DIR / "browser.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not record.get("how"):
        return None
    return f"{'opened with' if record.get('opened') else 'not opened:'} {record['how']}"


# ---------------------------------------------------------------- the block

def report_lines(system: Mapping[str, Any], *, root: Path, run: Mapping[str, Any] | None,
                 query: gpuinfo.GpuQuery, run_folder: bool = True) -> list[str]:
    gpus = query.gpus
    facts: list[tuple[str, Any]] = [
        ("GM1", f"gpuwm-cm1 {_gm1_version()}"),
        ("ArWen", f"{_version('gpuwm')}, {_version('gpuwm-data')}"),
        ("CuPy", _cupy()),
        ("NVRTC", _version("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu13", "nvidia-cuda-nvrtc-cu12")),
        ("nvJitLink", _version("nvidia-nvjitlink", "nvidia-nvjitlink-cu13", "nvidia-nvjitlink-cu12")),
        ("numpy", _version("numpy").removeprefix("numpy ")),
        ("Python", f"{platform.python_version()} ({platform.python_implementation()})"),
        ("OS", os_words()),
        ("WSL2", "yes" if system.get("wsl2") else "no"),
        ("glibc", glibc_words()),
        ("NVIDIA driver", (gpus[0].driver_version if gpus else None) or f"none found ({query.status})"),
    ]
    for gpu in gpus:
        facts.append((f"GPU {gpu.index}", f"{gpu.name}, {gpu.sm or 'unknown architecture'}, "
                      f"{gpu.total_bytes / 1024**3:.1f} GB, {gpu.free_bytes / 1024**3:.1f} GB free"))
    facts.append(("Installed with", system.get("install", {}).get("words", "unknown")))
    facts.append(("Runs folder", str(root) if run_folder else f"none yet (the disk check reads {root})"))
    browser = _browser(root) if run_folder else None
    if browser:
        facts.append(("Browser", browser))

    lines = [f"### {REPORT_TITLE}", "", *_table(facts), "",
             f"**Doctor:** {system['summary']}", "", "| | Area | What it found | Fix |", "|---|---|---|---|"]
    for row in system["rows"]:
        lines.append(f"| {row['state']} | {_cell(row['area'])} | {_cell(row['text'])} | {_cell(row.get('fix', ''))} |")
    lines.append("")
    if run is None:
        lines.append("**Last run:** none in the runs folder." if run_folder else "**Last run:** none (no runs folder yet).")
    else:
        words = f"`{run['name']}`: {run['label']}"
        if run.get("message"):
            words += f". {run['message']}"
        lines.append(f"**Last run:** {words}")
        if run.get("log_tail"):
            lines += ["", "<details><summary>The end of its log</summary>", "", "```text", *run["log_tail"], "```",
                      "", "</details>"]
    return lines


def doctor_report(root: str | os.PathLike[str], *, native_dir: str | os.PathLike[str] | None = None,
                  query: gpuinfo.GpuQuery | None = None, host: gpuinfo.HostInfo | None = None,
                  look_for_runs: bool = True) -> str:
    """The pasteable block, with no user, home or host name in it.

    ``look_for_runs`` False: ``root`` is only where the disk is measured (the home folder before the GUI made
    ``~/gm1-runs``), so no run is looked for in it.
    """

    root = Path(root).expanduser()
    query = gpuinfo.query_gpus() if query is None else query
    system = system_report(root, native_dir=native_dir, query=query, host=host)
    try:
        run = last_run(root) if look_for_runs else None
    except Exception as error:  # noqa: BLE001 - the report still prints without its last run
        run = {"name": "?", "state": "unreadable", "label": f"The runs folder could not be read: {error}"}
    lines = report_lines(system, root=root, run=run, query=query, run_folder=look_for_runs)
    text = "\n".join(lines)
    return scrub(text)


__all__ = ["REPORT_TITLE", "doctor_report", "glibc_words", "last_run", "os_words", "report_lines", "scrub"]
