"""Paths the GUI server may touch, and nothing else.

Every run directory is ``--root``/NAME or ``--root``/FOLDER/NAME, each part a
plain name.  A path is resolved (symlinks followed) and must stay under the
resolved root, so neither ``..`` nor a link can reach outside it.  The same
rule confines files served from a run directory and from ``static/``.

A run directory is any directory holding ``run.json`` (``gm1.cm1-run/v1``).
A plain ``gpuwm-cm1 cartesian`` output directory (an ``events.jsonl`` or a
``receipt.json`` beside frames) is listed too, view-only, and so is a folder of
CM1 netCDF output (:func:`cm1_output`): one holding CM1's history files, or the
``cm1-output.json`` the Runs screen writes when it opens CM1 output kept
elsewhere.  Those open for pictures only.
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
#: The record a folder opened for pictures of CM1 output kept elsewhere holds (POST /api/cm1-output).
CM1_OUTPUT_FILE = "cm1-output.json"
CM1_OUTPUT_SCHEMA = "gm1.cm1-output/v1"
#: CM1's netCDF history files by name: cm1out.nc (every time), cm1out_NNNNNN.nc (one time) and
#: cm1out_PPPPPP_NNNNNN.nc (one process's piece).
CM1_HISTORY_NAME = re.compile(r"^cm1out(_\d{6}){0,2}\.nc$")

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
    """A plain ``cartesian`` output directory (events or a receipt beside frames), or CM1 netCDF output opened for
    pictures (:func:`cm1_output`)."""

    if (path / RUN_FILE).exists():
        return False
    if (path / "events.jsonl").is_file():
        return True
    if (path / "receipt.json").is_file() and any((path / name).is_dir() for name in ("frames", "history")):
        return True
    return cm1_output(path) is not None


def cm1_output(path: Path) -> dict[str, Any] | None:
    """A folder of CM1 netCDF output opened for pictures: ``{"source": Path, "opened": bool}``, or None.

    ``source`` is what ``gpuwm-cm1 quicklook`` draws: the folder itself when it holds CM1's history files
    (``cm1out.nc``, ``cm1out_NNNNNN.nc``), or the folder or file its ``cm1-output.json`` names (``opened``: made
    by the Runs screen's Open CM1 output, the files kept where they are).  A GM1 run, which keeps its own
    ``cm1out/`` inside, is not one."""

    if (path / RUN_FILE).exists() or (path / "segments").is_dir() or (path / "frames").is_dir():
        return None
    record = path / CM1_OUTPUT_FILE
    if record.is_file():
        data = read_json(record, default=None)
        if isinstance(data, dict) and data.get("schema") == CM1_OUTPUT_SCHEMA and isinstance(data.get("source"), str):
            return {"source": Path(data["source"]), "opened": True}
        return None
    if (path / "events.jsonl").is_file():
        return None
    try:
        with os.scandir(path) as entries:
            if any(CM1_HISTORY_NAME.match(entry.name) and entry.is_file() for entry in entries):
                return {"source": path, "opened": False}
    except OSError:
        return None
    return None


#: A published frame's folder (cm1out.frames.FRAME_NAME, kept here so the server does not load that package).
FRAME_DIR_NAME = re.compile(r"^frame-\d{6}$")


def _holds_frames(folder: Path) -> bool:
    try:
        with os.scandir(folder) as entries:
            return any(FRAME_DIR_NAME.match(entry.name) and entry.is_dir(follow_symlinks=False) for entry in entries)
    except OSError:
        return False


#: A per-time picture's file name (quicklook.render._tag), and the folders under quicklook/ that hold no per-time
#: looks (the looks drawn once per run, the 3-D stills drawn on request, films, comparisons, slices, the record).
PICTURE_NAME = re.compile(r"^t(\d+)(?:p(\d+))?\.png$")
NOT_PER_TIME = frozenset({".state", "films", "compare", "slices", "storm3d", "environment", "parcels", "run-card",
                          "updraft-time-height", "spin-time-height", "thumb"})
#: The quick-look pass's own record (quicklook.render STATE_DIR/STATE_FILE).
QUICKLOOK_STATE = Path("quicklook") / ".state" / "state.json"
#: A published frame's clock (cm1out.frames.LOCATION; before it, PUBLICATION).
FRAME_LOCATION = "history-frame-location.json"
FRAME_PUBLICATION = "history-publication.json"


def _picture_times(quicklook: Path) -> set[float]:
    """The saved times the per-time pictures in ``quicklook`` were drawn at, by their file names."""

    found: set[float] = set()
    try:
        folders = [entry.path for entry in os.scandir(quicklook)
                   if entry.is_dir(follow_symlinks=False) and entry.name not in NOT_PER_TIME]
    except OSError:
        return found
    for folder in folders:
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    match = PICTURE_NAME.match(entry.name)
                    if match is not None:
                        whole, part = match.groups()
                        found.add(round(float(f"{int(whole)}.{part}") if part else float(int(whole)), 6))
        except OSError:
            continue
    return found


def drawn_times(path: Path, index: Any) -> list[dict[str, Any]]:
    """The saved times a run's pictures were drawn at, each its frame's key, segment and time: those of its
    quick-look ``index``, else of the pass's own record (an index that was lost), and every time a picture there
    shows that neither holds (no frame named: its record was lost)."""

    rows: dict[float, dict[str, Any]] = {}
    frames = index.get("frames") if isinstance(index, dict) else None
    for row in frames if isinstance(frames, list) else []:
        time_s = row.get("time_s") if isinstance(row, dict) else None
        if isinstance(time_s, (int, float)) and not isinstance(time_s, bool):
            rows[round(float(time_s), 6)] = {"frame": row.get("frame"), "segment": row.get("segment"),
                                             "time": float(time_s), "recovered": row.get("recovered")}
    if not rows:
        state = read_json(path / QUICKLOOK_STATE, default=None)
        held = state.get("frames") if isinstance(state, dict) else None
        for key, row in (held.items() if isinstance(held, dict) else ()):
            try:
                t = float(row.get("time") if isinstance(row, dict) and row.get("time") is not None else key)
            except (TypeError, ValueError):
                continue
            if isinstance(row, dict):
                rows[round(t, 6)] = {"frame": row.get("frame"), "segment": row.get("segment"), "time": t,
                                     "recovered": row.get("recovered")}
    for t in _picture_times(path / "quicklook"):
        rows.setdefault(t, {"frame": None, "segment": None, "time": t, "recovered": "pictures"})
    return [rows[t] for t in sorted(rows)]


def _frame_times(path: Path) -> set[float]:
    """The model times of every saved frame left in the run (any segment's, a cartesian folder's frames/)."""

    found: set[float] = set()
    for folder in _frame_folders(path):
        for name in (FRAME_LOCATION, FRAME_PUBLICATION):
            record = read_json(folder / name, default=None)
            value = record.get("model_time_s") if isinstance(record, dict) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.add(round(float(value), 6))
                break
    return found


def frames_gone(path: Path, index: Any) -> bool:
    """Whether a run has pictures (its quick-look ``index``, else the pictures themselves) but none of its saved
    frames are left: they were deleted by hand, or by Keep pictures only.  Its pictures are all there is of it, and
    no look can be drawn for it again, so the screens offer no pass over it.  CM1 output opened for pictures is drawn
    from its own files."""

    if cm1_output(path) is not None:
        return False
    frames = index.get("frames") if isinstance(index, dict) else None
    if not frames and not drawn_times(path, index):
        return False
    return not _any_frames(path)


def _any_frames(path: Path) -> bool:
    """Whether any saved frame's folder is left in the run (any segment's, a cartesian folder's frames/, or its own)."""

    segments = path / "segments"
    try:
        with os.scandir(segments) as entries:
            roots = [Path(entry.path) / "frames" for entry in entries if entry.is_dir() and entry.name.isdigit()]
    except OSError:
        roots = []
    return any(_holds_frames(root) for root in (*roots, path / "frames", path))


def _frame_there(path: Path, row: dict[str, Any]) -> bool:
    """Whether the saved frame a drawn or converted time names is still in the run.  Its key is the frame's folder
    under the run (``segments/NNN/frames/frame-XXXXXX``); pictures drawn before the first release named it by the
    folder alone (``frame-XXXXXX``), which is looked for in the time's segment and in ``frames/``."""

    key = row["frame"]
    if (path / key).is_dir():
        return True
    if "/" in key:
        return False
    segment = row.get("segment")
    places = [path / "frames" / key]
    if isinstance(segment, int) and not isinstance(segment, bool):
        places.insert(0, path / "segments" / f"{segment:03d}" / "frames" / key)
    return any(place.is_dir() for place in places)


def _lost(path: Path, rows: list[Any]) -> int:
    """How many of ``rows`` (drawn or converted times: each its frame's key, segment and time) have lost their saved
    frame's folder, though the run's records (run.json) do not put them on a branch it has left.  In a run with no
    saved frame left at all, every one whose folder is gone."""

    keys = [row for row in rows if isinstance(row, dict) and isinstance(row.get("frame"), str) and row["frame"]
            and ".." not in Path(row["frame"]).parts and not Path(row["frame"]).is_absolute()]
    gone = [row for row in keys if not _frame_there(path, row)]
    # a time known only from its pictures (its record was lost) is lost when no saved frame is at its time
    unnamed = [row for row in rows if isinstance(row, dict) and not row.get("frame") and row.get("recovered")
               and isinstance(row.get("time"), (int, float))]
    unnamed_lost = 0
    if unnamed:
        times = _frame_times(path)
        unnamed_lost = len([row for row in unnamed if round(float(row["time"]), 6) not in times])
    if not gone:
        return unnamed_lost
    line: dict[int, Any] = {}
    if (path / RUN_FILE).is_file() and _any_frames(path):
        try:
            from ..cm1run.rundir import RunDirectory

            line = RunDirectory(path).recorded_line()
        except Exception:  # noqa: BLE001 - a torn run.json: nothing is taken to be left behind
            line = {}
    from ..cm1run.rundir import off_the_line

    return unnamed_lost + len([row for row in gone if not off_the_line(line, row)])


def frames_missing(path: Path, index: Any) -> int:
    """How many of the saved times a run's pictures were drawn at (:func:`drawn_times`: its quick-look ``index``,
    else the pass's record, and the pictures themselves) have lost their saved frames (deleted by hand, or by Keep
    pictures only), though the run never left them.  Their pictures cannot be drawn again, so nothing that would
    draw over them or delete them is offered.  0 for CM1 output opened for pictures, which is drawn from its own
    files."""

    if cm1_output(path) is not None:
        return 0
    rows = drawn_times(path, index)
    return _lost(path, rows) if rows else 0


#: cm1out's records of what it converted, one per format (cm1out.convert.STATE_FILENAMES, written out here so the
#: server does not load that package), and the files each format writes.
CM1OUT_STATES = (".cm1out-state.json", ".cm1out-state-grads.json")
CM1OUT_FILES = (re.compile(r"^cm1out.*\.nc$"), re.compile(r"^cm1out.*\.(dat|ctl)$"))


def converted_missing(path: Path) -> int:
    """How many of a run's converted times (``cm1out/``, either format) have lost their saved frames though the run
    never left them: their files cannot be written again, so Convert again, which would delete them, is not
    offered."""

    most = 0
    for name in CM1OUT_STATES:
        state = read_json(path / "cm1out" / name, default=None)
        rows = state.get("frames") if isinstance(state, dict) and isinstance(state.get("frames"), dict) else {}
        most = max(most, _lost(path, [dict(row, time=row.get("time", key)) for key, row in rows.items()
                                      if isinstance(row, dict)]))
    return most


def converted_unrecorded(path: Path, index: Any = None) -> bool:
    """Whether a run's ``cm1out/`` holds converted files of a format whose record of what was converted is missing
    or cannot be read, while some of the run's saved frames are gone (its pictures' times have lost them, or there
    are more converted files than saved frames): converting again could write over files of times it cannot write
    again.  ``gpuwm-cm1 cm1out`` reads the files themselves and refuses then; the screen says so before it runs."""

    folder = path / "cm1out"
    try:
        names = [entry.name for entry in os.scandir(folder) if entry.is_file()]
    except OSError:
        return False
    for state_name, pattern in zip(CM1OUT_STATES, CM1OUT_FILES):
        files = [name for name in names if pattern.match(name)]
        if not files:
            continue
        state = read_json(folder / state_name, default=None)
        if isinstance(state, dict) and isinstance(state.get("frames"), dict):
            continue
        # one per converted time: a netCDF file, or a GrADS time's scalar file
        per_time = [name for name in files if re.match(r"^cm1out_(\d{6}_)?\d{6}(_s\.dat|\.nc)$", name)]
        frames = sum(1 for _ in _frame_folders(path))
        if frames_gone(path, index) or frames_missing(path, index) or len(per_time) > frames or not frames:
            return True
    return False


def _frame_folders(path: Path) -> Iterator[Path]:
    segments = path / "segments"
    try:
        with os.scandir(segments) as entries:
            roots = [Path(entry.path) / "frames" for entry in entries if entry.is_dir() and entry.name.isdigit()]
    except OSError:
        roots = []
    for root in (*roots, path / "frames"):
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if FRAME_DIR_NAME.match(entry.name) and entry.is_dir(follow_symlinks=False):
                        yield Path(entry.path)
        except OSError:
            continue


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
    "cm1_output",
    "converted_missing",
    "converted_unrecorded",
    "drawn_times",
    "frames_gone",
    "frames_missing",
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
