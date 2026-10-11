"""Reading a segment's ``gm1.run-events/v1`` log while it grows.

The runtime writes one JSON object per line and flushes each line
(:mod:`cm1core.run_events`).  A reader can open the file at any moment: the
last line may be half written, so it is left for the next read.  Nothing here
writes the log.

Importing this module imports neither CuPy nor numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .rundir import parse_utc

#: How far from the end :func:`last_record` reads before it gives up looking
#: for a line start (a progress record is well under 2 KB).
_TAIL_BYTES = 64 * 1024


#: CUDA's own words for "no usable NVIDIA driver or card" (cudaErrorInsufficientDriver, cudaErrorNoDevice and their
#: messages), which the runtime's device check passes on as they are.
_NO_DEVICE = ("cudaErrorInsufficientDriver", "cudaErrorNoDevice", "CUDA driver version is insufficient",
              "no CUDA-capable device", "CUDA_ERROR_NO_DEVICE", "CUDA_ERROR_INSUFFICIENT_DRIVER")
NO_DEVICE_REASON = ("No NVIDIA graphics card can be used: the NVIDIA driver is missing or older than R580.")
NO_DEVICE_REMEDY = "Run gpuwm-cm1 doctor --plain for the fix."


def plain_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """A ``refused`` or ``end`` record whose reason or message is CUDA's "no usable driver or card", in plain
    words (the original stays under ``technical``); any other record as it is."""

    text = " ".join(str(record.get(key) or "") for key in ("reason", "message"))
    if not any(words in text for words in _NO_DEVICE):
        return record
    plain = dict(record)
    plain["technical"] = text.strip()
    if record.get("reason") and record.get("kind") == "refused":
        plain["reason"] = NO_DEVICE_REASON
        plain["remedy"] = NO_DEVICE_REMEDY
    if record.get("message"):
        plain["message"] = f"{NO_DEVICE_REASON} {NO_DEVICE_REMEDY}"
    return plain


def read_records(path: Path | str) -> list[dict[str, Any]]:
    """Every complete record of a log; a torn last line is left out."""

    try:
        with open(path, "rb") as stream:
            data = stream.read()
    except OSError:
        return []
    return _parse(data, complete_only=True)[0]


def _parse(data: bytes, *, complete_only: bool) -> tuple[list[dict[str, Any]], int]:
    """Records in ``data`` and the byte count consumed (up to the last newline)."""

    records: list[dict[str, Any]] = []
    end = data.rfind(b"\n") + 1 if complete_only else len(data)
    for line in data[:end].splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records, end


def last_record(path: Path | str) -> dict[str, Any] | None:
    """The last complete record, read from the end of the file."""

    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _TAIL_BYTES))
            data = stream.read()
    except OSError:
        return None
    end = data.rfind(b"\n")
    if end < 0:
        return None
    lines = data[:end].splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            return record
    return None


class EventTail:
    """Follow one log: each :meth:`poll` returns the records written since the last.

    The file may not exist yet (the run is still waiting for its GPU); it
    is picked up when it appears.  A half-written last line is kept back
    until its newline arrives.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.offset = 0
        self._identity: tuple[int, int] | None = None

    def poll(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, "rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self._identity or stat.st_size < self.offset:
                    # A new file (or a truncated one): start from its beginning.
                    self._identity = identity
                    self.offset = 0
                stream.seek(self.offset)
                data = stream.read()
        except OSError:
            return []
        records, consumed = _parse(data, complete_only=True)
        self.offset += consumed
        return records


@dataclass
class SegmentEvents:
    """What a segment's log says so far."""

    records: int = 0
    admitted: Mapping[str, Any] | None = None
    startup: Mapping[str, Any] | None = None
    last_progress: Mapping[str, Any] | None = None
    frames: list[Mapping[str, Any]] = field(default_factory=list)
    restarts: list[Mapping[str, Any]] = field(default_factory=list)
    warnings: list[Mapping[str, Any]] = field(default_factory=list)
    refused: Mapping[str, Any] | None = None
    end: Mapping[str, Any] | None = None
    last: Mapping[str, Any] | None = None
    #: The last record's time (seconds since the epoch), the heartbeat.
    last_time: float | None = None

    def add(self, records: Iterable[Mapping[str, Any]]) -> "SegmentEvents":
        for record in records:
            self.records += 1
            kind = record.get("kind")
            if kind == "admitted":
                self.admitted = record
            elif kind == "startup":
                self.startup = record
            elif kind == "progress":
                self.last_progress = record
            elif kind == "frame":
                self.frames.append(record)
            elif kind == "restart":
                self.restarts.append(record)
            elif kind == "warning":
                self.warnings.append(record)
            elif kind == "refused":
                self.refused = plain_record(record)
            elif kind == "end":
                self.end = plain_record(record)
            self.last = record
            moment = parse_utc(record.get("time_utc"))
            if moment is not None:
                self.last_time = moment
        return self

    @property
    def last_frame(self) -> Mapping[str, Any] | None:
        return self.frames[-1] if self.frames else None

    @property
    def last_restart(self) -> Mapping[str, Any] | None:
        return self.restarts[-1] if self.restarts else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "admitted": self.admitted,
            "last_progress": self.last_progress,
            "frames": len(self.frames),
            "last_frame": self.last_frame,
            "restarts": len(self.restarts),
            "last_restart": self.last_restart,
            "warnings": list(self.warnings),
            "refused": self.refused,
            "end": self.end,
            "last_time": self.last_time,
        }


def summarize(path: Path | str) -> SegmentEvents:
    return SegmentEvents().add(read_records(path))


__all__ = ["EventTail", "SegmentEvents", "last_record", "read_records", "summarize"]
