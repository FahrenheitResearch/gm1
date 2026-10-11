"""Published GM1 history frames: find them, and read their carriers.

A Cartesian run publishes each history time as ``frame-NNNNNN/`` under its
history directory.  The frame is written inside ``.incomplete/`` and renamed
into place only once every file in it is on disk
(``cuda_cartesian_runtime.py`` ``FRAME_STAGING_DIRECTORY``), so a directory
named ``frame-NNNNNN`` directly under the history root is complete.

This module lists only those directories.  It never enters ``.incomplete/``
or any other dot-directory, and it never opens a path a frame's own records
name when that path is absolute: older publishers wrote the staging path of
the arena into column 7 of the field table, which names a file under
``.incomplete/`` that no longer exists.  The arena is always the file of
that name inside the published frame directory.

Run layouts it understands:

``RUNDIR/segments/NNN/frames/frame-*``
    the GUI run directory (``gm1.cm1-run/v1``), one segment per
    start or resume; ``segments/NNN/events.jsonl`` is that segment's event
    log.  With its ``run.json`` the run's frames are the newest segment's
    lineage (``cm1run.rundir.RunDirectory.frames``).
``DIR/frames/frame-*``
    a ``gpuwm-cm1 cartesian`` output directory whose history went to
    ``DIR/frames`` (``DIR/events.jsonl`` when ``--events`` pointed there).
``DIR/frame-*``
    a history directory on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping

FRAME_NAME = re.compile(r"^frame-(\d{6})$")
FIELD_TABLE = "history-field-table.tsv"
ARENA = "history-carriers.f32"
LOCATION = "history-frame-location.json"
PUBLICATION = "history-publication.json"
EVENTS = "events.jsonl"
RUN_JSON = "run.json"


class FrameError(ValueError):
    """A published frame is not what its own records say it is."""


@dataclass(frozen=True, slots=True)
class Carrier:
    name: str
    meaning: str
    units: str
    stagger: str
    shape: tuple[int, ...]
    offset_bytes: int

    @property
    def count(self) -> int:
        total = 1
        for extent in self.shape:
            total *= extent
        return total


@dataclass(eq=False)
class Frame:
    """One published history frame, read lazily."""

    directory: Path
    carriers: Mapping[str, Carrier]
    location: Mapping[str, Any]
    _arena: Any = field(default=None, repr=False)
    _publication: Any = field(default=None, repr=False)

    @property
    def index(self) -> int:
        match = FRAME_NAME.match(self.directory.name)
        return int(match.group(1)) if match else int(self.location.get("index", -1))

    @property
    def model_time_s(self) -> float:
        return float(self.location["model_time_s"])

    @property
    def step(self) -> int:
        return int(self.location.get("step", 0))

    @property
    def arena_sha256(self) -> str | None:
        return self.location.get("arena_sha256")

    @property
    def moving_frame(self) -> Mapping[str, Any]:
        return self.location.get("moving_frame") or {}

    @property
    def publication(self) -> Mapping[str, Any]:
        if self._publication is None:
            path = self.directory / PUBLICATION
            self._publication = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return self._publication

    def has(self, name: str) -> bool:
        return name in self.carriers

    def array(self, name: str) -> Any:
        """The carrier as a read-only float32 view of the arena, in its declared shape."""

        import numpy as np

        carrier = self.carriers[name]
        if self._arena is None:
            self._arena = np.memmap(self.directory / ARENA, dtype="<f4", mode="r")
        start = carrier.offset_bytes // 4
        words = self._arena[start:start + carrier.count]
        if words.size != carrier.count:
            raise FrameError(
                f"{self.directory}: carrier {name} runs past the end of {ARENA}"
            )
        return words.reshape(carrier.shape)

    def close(self) -> None:
        arena, self._arena = self._arena, None
        if arena is not None:
            mm = getattr(arena, "_mmap", None)
            if mm is not None:
                mm.close()


def open_frame(directory: Path | str) -> Frame:
    """Read one published frame's field table and location record."""

    directory = Path(directory)
    if any(part.startswith(".") for part in directory.parts[-2:]):
        raise FrameError(f"{directory} is a staging directory, not a published frame")
    table = directory / FIELD_TABLE
    arena = directory / ARENA
    if not table.is_file() or not arena.is_file():
        raise FrameError(
            f"{directory} is not a published history frame: it needs {FIELD_TABLE} and {ARENA}"
        )
    size = arena.stat().st_size
    carriers: dict[str, Carrier] = {}
    for number, line in enumerate(table.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) != 8:
            raise FrameError(f"{table}:{number}: a field-table row has 8 columns, not {len(columns)}")
        name, meaning, units, stagger, shape_csv, _bounds, arena_name, offset = columns
        # Column 7 is the arena relative to the frame (current publishers) or
        # the absolute path it was staged at (older ones).  Either way the
        # bytes are the arena inside this directory; a staging path is never
        # opened.
        if Path(arena_name).name != ARENA:
            raise FrameError(f"{table}:{number}: carrier {name} names arena {arena_name!r}, not {ARENA}")
        carrier = Carrier(
            name=name, meaning=meaning, units=units, stagger=stagger,
            shape=tuple(int(value) for value in shape_csv.split(",")),
            offset_bytes=int(offset),
        )
        if carrier.offset_bytes % 4 or carrier.offset_bytes + 4 * carrier.count > size:
            raise FrameError(
                f"{table}:{number}: carrier {name} declares bytes "
                f"[{carrier.offset_bytes}, {carrier.offset_bytes + 4 * carrier.count}) "
                f"but {ARENA} is {size} bytes"
            )
        carriers[name] = carrier
    location_path = directory / LOCATION
    if location_path.is_file():
        location = json.loads(location_path.read_text(encoding="utf-8"))
    else:
        # A frame from before the location record: the clock is in the
        # publication receipt.
        publication = json.loads((directory / PUBLICATION).read_text(encoding="utf-8"))
        location = {
            "model_time_s": publication["model_time_s"],
            "step": publication.get("step", 0),
            "moving_frame": publication.get("moving_frame"),
        }
    if "model_time_s" not in location:
        raise FrameError(f"{directory}: the frame records no model time")
    return Frame(directory=directory, carriers=carriers, location=location)


def published_frames(root: Path | str) -> list[Path]:
    """The published ``frame-NNNNNN`` directories directly under ``root``, in index order."""

    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for entry in root.iterdir():
        # A dot-directory (the staging ``.incomplete/``) is never listed.
        if entry.name.startswith(".") or not FRAME_NAME.match(entry.name):
            continue
        if entry.is_dir() and not entry.is_symlink():
            found.append(entry)
    return sorted(found, key=lambda path: path.name)


@dataclass(frozen=True, slots=True)
class Segment:
    """One run segment: its history root and its event log, when it has one."""

    number: int
    frames_root: Path
    events: Path | None


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Where a run's frames are, and where its CM1 settings and output belong."""

    kind: str  # "rundir", "output", or "frames"
    root: Path
    segments: tuple[Segment, ...]

    @property
    def default_out(self) -> Path:
        if self.kind == "frames":
            return self.root.parent / "cm1out"
        return self.root / "cm1out"

    def candidate_namelists(self) -> tuple[Path, ...]:
        base = self.root if self.kind != "frames" else self.root.parent
        return (base / "namelist.input",)

    def candidate_decks(self) -> tuple[Path, ...]:
        base = self.root if self.kind != "frames" else self.root.parent
        found = [base / "deck.toml"]
        # The event log's admitted record names the configuration it ran.
        for segment in self.segments:
            if segment.events is not None and segment.events.is_file():
                for record in _read_events(segment.events):
                    if record.get("kind") == "admitted" and record.get("config"):
                        found.append(Path(record["config"]))
                        break
        return tuple(found)

    def run_directory(self) -> Any:
        """The ``gpuwm-cm1 run`` directory (``cm1run.rundir``, with its run.json) this is, or None."""

        if self.kind != "rundir" or not (self.root / RUN_JSON).is_file():
            return None
        from ..cm1run.rundir import RunDirectory

        return RunDirectory(self.root)

    def lineage_settled(self) -> bool:
        """Whether the newest segment's place in the run is recorded yet.

        ``gpuwm-cm1 run`` creates ``segments/NNN`` a moment before run.json
        says which restart it continues from; until then the run's lineage is
        not known and nothing converted is taken to be left behind.
        """

        run = self.run_directory()
        if run is None:
            return True
        latest = run.latest_segment()
        return latest is None or run.segment_record(latest.number) is not None

    def frames(self) -> list[tuple[Segment, Path]]:
        """The published frames that form the run, oldest first.

        In a ``gpuwm-cm1 run`` directory these are the frames of the newest
        segment's lineage (``RunDirectory.frames``): after a resume from an
        earlier restart, or a start over, the frames of the branch the run
        has left are not the run's.  Elsewhere, every published frame of
        every segment, oldest segment first (a later segment's frame at the
        same time replaces an earlier one).
        """

        run = self.run_directory()
        if run is not None:
            by_number = {segment.number: segment for segment in self.segments}
            return [
                (by_number[info.segment], info.path)
                for info in run.frames() if info.segment in by_number
            ]
        rows = []
        for segment in self.segments:
            rows.extend((segment, path) for path in published_frames(segment.frames_root))
        return rows

    def ended(self) -> tuple[bool, Mapping[str, Any] | None]:
        """Whether the newest segment's event log holds its ``end`` record."""

        if not self.segments:
            return False, None
        newest = self.segments[-1]
        if newest.events is None or not newest.events.is_file():
            return False, None
        for record in reversed(_read_events(newest.events)):
            if record.get("kind") == "end":
                return True, record
        return False, None

    def has_event_log(self) -> bool:
        return any(s.events is not None and s.events.is_file() for s in self.segments)


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        # The writer flushes whole lines; a line still being written is
        # skipped until it is complete.
        if not line.endswith("}"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def discover(target: Path | str) -> RunLayout:
    """Recognise a run directory, a cartesian output directory or a history directory."""

    target = Path(target)
    if not target.is_dir():
        raise FrameError(f"{target} is not a directory")
    segments_dir = target / "segments"
    if segments_dir.is_dir():
        segments = []
        for entry in sorted(segments_dir.iterdir(), key=lambda path: path.name):
            if entry.name.startswith(".") or not entry.is_dir() or not entry.name.isdigit():
                continue
            events = entry / EVENTS
            segments.append(Segment(int(entry.name), entry / "frames", events))
        return RunLayout("rundir", target, tuple(segments))
    if (target / "frames").is_dir():
        return RunLayout("output", target, (Segment(1, target / "frames", target / EVENTS),))
    if published_frames(target) or (target / ".incomplete").is_dir():
        events = target.parent / EVENTS
        return RunLayout("frames", target, (Segment(1, target, events),))
    raise FrameError(
        f"{target} holds no GM1 history: expected segments/NNN/frames/, frames/ or frame-NNNNNN/ directories"
    )


__all__ = [
    "ARENA",
    "Carrier",
    "FIELD_TABLE",
    "Frame",
    "FrameError",
    "RunLayout",
    "Segment",
    "discover",
    "open_frame",
    "published_frames",
]
