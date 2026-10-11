"""Paths the GUI server may touch, and nothing else.

Every run directory is ``--root``/NAME or ``--root``/FOLDER/NAME, each part a
plain name.  A path is resolved (symlinks followed) and must stay under the
resolved root, so neither ``..`` nor a link can reach outside it.  The same
rule confines files served from a run directory and from ``static/``.

A run directory is any directory holding ``run.json`` (``gm1.cm1-run/v1``).
A plain ``gpuwm-cm1 cartesian`` output directory (an ``events.jsonl`` or a
``receipt.json`` beside frames) is listed too, view-only.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
from typing import Any, Iterator

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
RUN_FILE = "run.json"
#: Directories the server keeps its own state in, under the root.
SERVER_DIR = ".gm1-gui"
#: How deep under the root run directories are looked for (FOLDER/NAME).
SCAN_DEPTH = 2
#: A scan stops after this many directories, so a huge --root stays quick.
SCAN_LIMIT = 4000
JSON_LIMIT = 16 * 1024 * 1024

STATIC_DIR = Path(__file__).resolve().parent / "static"

_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "text/plain; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".toml": "text/plain; charset=utf-8",
    ".ctl": "text/plain; charset=utf-8",
    ".input": "text/plain; charset=utf-8",
    ".original": "text/plain; charset=utf-8",
    ".tsv": "text/plain; charset=utf-8",
    ".csv": "text/plain; charset=utf-8",
    ".diff": "text/plain; charset=utf-8",
    ".sh": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
    ".ico": "image/x-icon",
}
#: Types a browser may show inline; everything else is sent as a download.
INLINE_PREFIXES = ("text/", "image/png", "application/json", "font/", "image/x-icon")
#: Files from a run directory are data, never page code: only these are shown
#: inline, and HTML, scripts, style sheets and SVG are sent as plain downloads,
#: so nothing written into a run folder runs with the GUI's origin.
RUN_FILE_INLINE = ("image/png", "application/json", "text/plain")


class PathRefused(ValueError):
    """A name or path the server will not touch; the message says why."""


def require_name(part: str, what: str = "name") -> str:
    if not isinstance(part, str) or not NAME_RE.match(part):
        raise PathRefused(
            f"The {what} {part!r} is not allowed. Use letters, digits, '.', '_' or '-' "
            "(up to 64, starting with a letter or digit).")
    return part


def split_run_id(run_id: str) -> list[str]:
    parts = str(run_id).split("/")
    if not 1 <= len(parts) <= SCAN_DEPTH or not all(parts):
        raise PathRefused(f"{run_id!r} is not a run name (NAME or FOLDER/NAME).")
    return [require_name(part, "run name" if i == len(parts) - 1 else "folder name")
            for i, part in enumerate(parts)]


def within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    return True


class Root:
    """The ``--root`` directory and the run directories under it."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()

    @property
    def server_dir(self) -> Path:
        return self.path / SERVER_DIR

    def run_path(self, run_id: str) -> Path:
        """The directory for ``run_id``, which may not exist yet."""

        candidate = self.path.joinpath(*split_run_id(run_id))
        if not within(self.path, candidate):
            raise PathRefused(f"{run_id!r} leads outside the runs folder.")
        return candidate

    def existing_run(self, run_id: str) -> Path:
        path = self.run_path(run_id)
        if not path.is_dir() or not is_run_dir(path):
            raise FileNotFoundError(f"No run called {run_id!r} in {self.path}.")
        return path

    def run_id(self, path: Path) -> str:
        return path.resolve().relative_to(self.path).as_posix()

    def iter_runs(self) -> Iterator[Path]:
        """Run directories under the root, depth first, names sorted."""

        seen = 0

        def walk(directory: Path, depth: int) -> Iterator[Path]:
            nonlocal seen
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError:
                return
            for entry in entries:
                if seen >= SCAN_LIMIT:
                    return
                if entry.name.startswith(".") or not NAME_RE.match(entry.name):
                    continue
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                seen += 1
                path = Path(entry.path)
                if not within(self.path, path):
                    continue
                if is_run_dir(path):
                    yield path
                elif depth < SCAN_DEPTH:
                    yield from walk(path, depth + 1)

        if self.path.is_dir():
            yield from walk(self.path, 1)

    def has_runs(self) -> bool:
        return next(self.iter_runs(), None) is not None


def is_run_dir(path: Path) -> bool:
    return (path / RUN_FILE).is_file() or is_view_only(path)


def is_view_only(path: Path) -> bool:
    """A plain ``cartesian`` output directory: events or a receipt beside frames."""

    if (path / RUN_FILE).exists():
        return False
    if (path / "events.jsonl").is_file():
        return True
    return (path / "receipt.json").is_file() and any(
        (path / name).is_dir() for name in ("frames", "history"))


def safe_file(base: Path, relative: str) -> Path:
    """A regular file ``relative`` under ``base``, or PathRefused."""

    parts = [part for part in relative.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." or "\x00" in part for part in parts):
        raise PathRefused(f"{relative!r} is not a file path this server serves.")
    candidate = base.joinpath(*parts)
    if not within(base, candidate):
        raise PathRefused(f"{relative!r} leads outside its folder.")
    if not candidate.is_file():
        raise FileNotFoundError(relative)
    return candidate


def static_file(relative: str) -> Path:
    return safe_file(STATIC_DIR, relative)


def content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TYPES:
        return _TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def is_inline(kind: str) -> bool:
    return kind.startswith(INLINE_PREFIXES)


def run_file_type(path: Path) -> tuple[str, bool]:
    """(Content-Type, inline) for a file served from a run directory."""

    kind = content_type(path)
    if kind.startswith(RUN_FILE_INLINE):
        return kind, True
    return "application/octet-stream", False


def download_name(name: str) -> str:
    """A file name safe inside a Content-Disposition header."""

    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:128] or "download"


def read_json(path: Path, default: Any = None, limit: int = JSON_LIMIT) -> Any:
    try:
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
    except OSError:
        return default
    if len(data) > limit:
        return default
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


UPLOADS_DIR = "uploads"


def stage_upload(server_dir: Path, filename: str, data: bytes, *, write: bool = True) -> Path:
    """A file the browser sent, kept where a command can read it: ``SERVER_DIR/uploads/SHA/FILENAME``.

    The folder is named by the content's SHA-256, so the same text always lands
    at the same path: the command "Show command" prints before a press is the
    line the press then runs, and a replayed ``commands.log`` finds the file.
    A staged file is written once and never changed.  ``write=False`` (a dry
    run) names the path without writing it, so asking for the command leaves
    nothing behind, however often the page asks as the text is typed.
    """

    if not NAME_RE.match(filename):
        raise PathRefused(f"{filename!r} is not a plain file name.")
    digest = hashlib.sha256(data).hexdigest()[:24]
    directory = Path(server_dir) / UPLOADS_DIR / digest
    path = directory / filename
    if not write:
        return path
    try:
        if path.read_bytes() == data:
            return path
    except OSError:
        pass
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{filename}.{os.getpid()}.tmp"
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return path


def tree_bytes(path: Path) -> int:
    """Bytes of every regular file under ``path`` (links not followed)."""

    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    stack = [path]
    while stack:
        directory = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def listing(directory: Path, pattern: str = "*") -> list[dict[str, Any]]:
    """Regular files in one directory: name and bytes, names sorted."""

    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and not path.name.startswith("."):
            rows.append({"name": path.name, "bytes": path.stat().st_size})
    return rows


__all__ = [
    "PathRefused",
    "Root",
    "RUN_FILE",
    "SERVER_DIR",
    "STATIC_DIR",
    "content_type",
    "is_inline",
    "is_run_dir",
    "is_view_only",
    "listing",
    "read_json",
    "require_name",
    "run_file_type",
    "download_name",
    "safe_file",
    "split_run_id",
    "stage_upload",
    "static_file",
    "tree_bytes",
    "within",
    "write_json_atomic",
]
