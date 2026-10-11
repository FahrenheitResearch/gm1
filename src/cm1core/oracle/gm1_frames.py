"""Read a gpuwm-cm1 history frame directory.

``gpuwm-cm1 cartesian --integrate --history-dir DIR`` publishes one
``frame-NNNNNN`` directory per ``run.history_interval_s``.  Beside the sealed
``history.cm1fhs`` envelope the publisher leaves the exact raw inputs it
sealed: ``history-carriers.f32`` (concatenated little-endian IEEE binary32
carriers, no header, no padding), ``history-field-table.tsv`` (name, meaning,
units, stagger, slow-to-fast shape, bounds, carrier path, byte offset) and
``history-publication.json`` (the same offsets plus per-carrier sha256 and the
frame's clock).

This reader takes the raw arena and the field table.  It verifies the declared
offsets and shapes against the arena's own size, so a frame that was truncated
or renamed refuses instead of comparing the wrong bytes.
"""

from __future__ import annotations

import array
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class GM1Frame:
    directory: Path
    arena: Path
    carriers: Mapping[str, Carrier]
    step: int
    model_time_s: float
    dt_s: float

    def read(self, name: str) -> tuple[array.array, tuple[int, ...]]:
        carrier = self.carriers[name]
        words = array.array("f")
        with self.arena.open("rb") as handle:
            handle.seek(carrier.offset_bytes)
            raw = handle.read(carrier.count * 4)
        if len(raw) != carrier.count * 4:
            raise ValueError(
                f"{self.arena} is short for carrier {name}: wanted {carrier.count * 4} bytes "
                f"at {carrier.offset_bytes}, got {len(raw)}"
            )
        words.frombytes(raw)
        return words, carrier.shape


def _clock(directory: Path) -> tuple[int, float, float]:
    publication = directory / "history-publication.json"
    if publication.exists():
        payload = json.loads(publication.read_text(encoding="utf-8"))
        # The Rust publisher's receipt states the frame clock at the top level;
        # the sealed envelope header states the same clock as f64 bit patterns.
        if "model_time_s" in payload:
            return (
                int(payload.get("step", 0)),
                float(payload["model_time_s"]),
                float(payload.get("dt_s", 0.0)),
            )
        clock = payload.get("clock") or payload.get("frame", {}).get("clock") or {}
        step = int(clock.get("step", 0))
        model = clock.get("model_time_s")
        dt = clock.get("dt_s")
        if model is None and "model_time_f64_bits" in clock:
            model = _f64(clock["model_time_f64_bits"])
        if dt is None and "dt_f64_bits" in clock:
            dt = _f64(clock["dt_f64_bits"])
        if model is not None:
            return step, float(model), float(dt or 0.0)
    return 0, float("nan"), float("nan")


def _f64(bits: str) -> float:
    import struct

    return struct.unpack(">d", bytes.fromhex(bits))[0]


def open_frame(directory: Path) -> GM1Frame:
    table = directory / "history-field-table.tsv"
    if not table.is_file():
        raise FileNotFoundError(
            f"{directory} carries no history-field-table.tsv; a gpuwm-cm1 history frame "
            "directory is the one its Rust publisher wrote, not the envelope alone"
        )
    arena = directory / "history-carriers.f32"
    if not arena.is_file():
        raise FileNotFoundError(f"{directory} carries no history-carriers.f32 raw arena")
    size = arena.stat().st_size
    carriers: dict[str, Carrier] = {}
    for line in table.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 8:
            raise ValueError(f"{table}: a field-table row has eight columns; got {len(parts)}")
        name, meaning, units, stagger, shape_csv, _bounds, _path, offset = parts
        carrier = Carrier(
            name=name,
            meaning=meaning,
            units=units,
            stagger=stagger,
            shape=tuple(int(v) for v in shape_csv.split(",")),
            offset_bytes=int(offset),
        )
        end = carrier.offset_bytes + carrier.count * 4
        if end > size:
            raise ValueError(
                f"{table}: carrier {name} declares bytes [{carrier.offset_bytes}, {end}) but "
                f"{arena.name} is {size} bytes"
            )
        carriers[name] = carrier
    step, model_time, dt = _clock(directory)
    return GM1Frame(directory, arena, carriers, step, model_time, dt)


@dataclass(frozen=True)
class GM1History:
    root: Path
    frames: tuple[GM1Frame, ...]

    @property
    def model_time_s(self) -> tuple[float, ...]:
        return tuple(f.model_time_s for f in self.frames)

    def field_names(self) -> Sequence[str]:
        return sorted(self.frames[0].carriers) if self.frames else []


def open_history(root: Path) -> GM1History:
    directories = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("frame-"))
    if not directories:
        raise FileNotFoundError(f"{root} holds no frame-NNNNNN directories")
    return GM1History(root, tuple(open_frame(p) for p in directories))
