"""``gm1.run-events/v1``: the JSON-lines event log of one Cartesian run.

``gpuwm-cm1 cartesian --events PATH`` writes one JSON object per line and
flushes each line, so a watcher (the GUI, a job-script sidecar, ``tail -f``)
reads the run's state without a socket and without parsing the stderr text.
Every record the stderr lines carry is built by the same code
(:func:`cm1core.runtime_profiler.progress_fields`); this module only writes
it as JSON beside them.

The kinds, in the order a run emits them:

``admitted``
    Every host-side gate passed; the run will open a device.  Config hash,
    grid, steps, dt, nsound, the frames and restarts it will publish, and
    where they go.
``startup``
    Written once, right before the first large step: seconds spent in each
    start-up stage (NVRTC compiles, the initializer, the runtime construction,
    the host arenas).
``progress``
    Step, model time, s/step (recent and mean), model-s per wall-s, ETA, the
    last I/O event and GPU telemetry.  One from the stepping thread at least
    every :data:`PROGRESS_INTERVAL_S`; when the stepping thread is busy
    elsewhere (compiling, waiting for the GPU mutex, joining the last
    publication) a heartbeat thread writes one with the last known values and
    ``heartbeat: true``, so no two records are more than
    :data:`HEARTBEAT_LIMIT_S` apart while the process lives.
``frame`` / ``restart``
    One per published directory, written only once it has been renamed into
    place.
``warning``
    Something the run did differently from what was asked, and why.
``refused``
    The refusal document ``{code, gate, reason, remedy}``.
``end``
    Always the last record: ``reason`` (completed, stopped, paused,
    courant-stop, refused, error), the process exit status, and the receipt.

A process killed without a chance to run Python (SIGKILL, the OOM killer, a
lost node) leaves a log with no ``end`` record; a reader treats a log whose
last record is older than :data:`STALE_AFTER_S` and has no ``end`` as a run
that stopped without a final record.

Importing this module imports neither CuPy nor numpy.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any

RUN_EVENTS_SCHEMA = "gm1.run-events/v1"
SCHEMA_FILENAME = "GM1-RUN-EVENTS-V1.schema.json"

EVENT_KINDS = (
    "admitted", "startup", "progress", "stats", "frame", "restart", "warning",
    "refused", "end",
)
END_REASONS = ("completed", "stopped", "paused", "courant-stop", "refused", "error")
PHASES = ("waiting-for-gpu", "starting", "stepping", "finishing")

#: Wall seconds between progress records from the stepping thread.
PROGRESS_INTERVAL_S = 5.0
#: The heartbeat thread writes a record when nothing has been written for this
#: long; with its one-second tick no two records are further apart than
#: :data:`HEARTBEAT_LIMIT_S` while the interpreter runs.
HEARTBEAT_AFTER_S = 8.0
HEARTBEAT_LIMIT_S = 10.0
#: The age at which a log with no end record reads as a stopped run.
STALE_AFTER_S = 60.0


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "records" / SCHEMA_FILENAME


def load_schema() -> dict[str, Any]:
    """The ``gm1.run-events/v1`` JSON Schema this module validates against."""

    return json.loads(_schema_path().read_text(encoding="utf-8"))


def _finite(value: Any) -> Any:
    """JSON has no NaN or infinity; a non-finite number is written as null."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunEventLog:
    """One run's event log: a new file, one flushed JSON line per record.

    Thread-safe: the stepping thread, the publication worker's callbacks and
    the heartbeat thread all write through :meth:`emit`.  A write that fails
    (a full disk, a removed directory) turns the log off with one line on
    stderr and never stops the run, as the profiler's instruments do.
    """

    def __init__(self, path: Path | str, *, clock: Callable[[], float] = time.monotonic,
                 stream: Any = None) -> None:
        self.path = Path(path)
        # "x": a log is never appended to or replaced; two runs' records in
        # one file would be read as one run.
        self._handle = self.path.open("x", encoding="utf-8")
        self._clock = clock
        self._opened = clock()
        self._lock = threading.RLock()
        self._seq = 0
        self._last_write = self._opened
        self._stderr = stream
        self.failed: str | None = None
        self.closed = False
        self.ended = False
        self.phase = "starting"
        self.last_progress: dict[str, Any] | None = None
        self._heartbeat: threading.Thread | None = None
        self._stop = threading.Event()
        self.counts: dict[str, int] = {}

    # ---- writing

    def emit(self, kind: str, **fields: Any) -> dict[str, Any] | None:
        """Write one record; returns it, or ``None`` when the log is off."""

        if kind not in EVENT_KINDS:
            raise ValueError(f"{kind!r} is not a {RUN_EVENTS_SCHEMA} event kind")
        with self._lock:
            if self.closed or self.failed is not None or self.ended:
                return None
            now = self._clock()
            record = {
                "schema": RUN_EVENTS_SCHEMA,
                "seq": self._seq,
                "kind": kind,
                "time_utc": _utc_now(),
                "elapsed_s": max(0.0, now - self._opened),
                "pid": os.getpid(),
                **_finite(fields),
            }
            try:
                line = json.dumps(record, sort_keys=True, allow_nan=False)
                self._handle.write(line + "\n")
                self._handle.flush()
            except (OSError, ValueError, TypeError) as error:
                self._fail(error)
                return None
            self._seq += 1
            self._last_write = now
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if kind == "progress":
                self.last_progress = {
                    key: value for key, value in record.items()
                    if key not in ("schema", "seq", "kind", "time_utc", "elapsed_s", "pid",
                                   "phase", "heartbeat")
                }
            if kind == "end":
                self.ended = True
            return record

    def _fail(self, error: BaseException) -> None:
        self.failed = f"{type(error).__name__}: {error}"
        try:
            print(
                f"gm1 events: {self.path} turned off after {self.failed}; the run continues",
                file=self._stderr if self._stderr is not None else sys.stderr, flush=True,
            )
        except Exception:  # noqa: BLE001
            pass

    def set_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"{phase!r} is not one of {PHASES}")
        self.phase = phase

    def progress(self, fields: Mapping[str, Any], *, heartbeat: bool = False) -> dict[str, Any] | None:
        return self.emit(
            "progress", phase=self.phase, heartbeat=bool(heartbeat), **dict(fields),
        )

    # ---- heartbeat

    def seconds_since_last_record(self) -> float:
        with self._lock:
            return self._clock() - self._last_write

    def _beat(self) -> None:
        with self._lock:
            if self._clock() - self._last_write < HEARTBEAT_AFTER_S:
                return
            fields = dict(self.last_progress) if self.last_progress is not None else {
                "step": None, "total_steps": None, "model_time_s": None,
                "duration_s": None, "s_per_step_recent": None,
                "s_per_step_mean": None, "model_s_per_wall_s": None, "eta_s": None,
                "last_io": None, "gpu": None,
            }
            self.progress(fields, heartbeat=True)

    def start_heartbeat(self, tick_s: float = 1.0) -> "RunEventLog":
        """A daemon thread that keeps the log from going quiet.

        It reads nothing but this object: no CUDA call, no runtime state.  It
        writes a progress record with the last known values only when nothing
        has been written for :data:`HEARTBEAT_AFTER_S`.
        """

        if self._heartbeat is not None:
            return self

        def run() -> None:
            while not self._stop.wait(tick_s):
                try:
                    self._beat()
                except Exception:  # noqa: BLE001 - a heartbeat never stops a run
                    return

        self._heartbeat = threading.Thread(target=run, name="gm1-run-events-heartbeat", daemon=True)
        self._heartbeat.start()
        return self

    def stop_heartbeat(self) -> None:
        self._stop.set()
        thread, self._heartbeat = self._heartbeat, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def close(self) -> None:
        self.stop_heartbeat()
        with self._lock:
            if self.closed:
                return
            self.closed = True
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except (OSError, ValueError):
                pass
            try:
                self._handle.close()
            except OSError:
                pass


def host_name() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def directory_bytes(path: Path | str) -> int:
    """Bytes of the regular files directly inside one published directory."""

    total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


# ------------------------------------------------------------- validation
#
# A small JSON Schema (2020-12) evaluator for exactly the keywords the v1
# schema uses: type, const, enum, required, properties, additionalProperties,
# minimum, exclusiveMinimum, pattern, allOf, if/then and local $ref.  It is
# here so a GUI or a job script can check a log with the standard library
# alone; tests/test_run_events.py holds it to the jsonschema package's verdict.

_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    ),
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}
_KNOWN_KEYWORDS = frozenset({
    "$schema", "$id", "title", "description", "$defs", "type", "const", "enum",
    "required", "properties", "additionalProperties", "minimum",
    "exclusiveMinimum", "pattern", "allOf", "if", "then", "$ref",
})


def _resolve(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"only local references are admitted, not {reference!r}")
    node: Any = root
    for part in reference[2:].split("/"):
        node = node[part]
    return node


def _errors(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], where: str) -> list[str]:
    import re

    unknown = set(schema) - _KNOWN_KEYWORDS
    if unknown:
        raise ValueError(f"schema keyword(s) {sorted(unknown)} at {where} are not evaluated here")
    found: list[str] = []
    if "$ref" in schema:
        found += _errors(value, _resolve(root, schema["$ref"]), root, where)
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_TYPES[kind](value) for kind in kinds):
            return found + [f"{where}: {value!r} is not of type {kinds}"]
    if "const" in schema and value != schema["const"]:
        found.append(f"{where}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{where}: {value!r} is not one of {schema['enum']}")
    number = _TYPES["number"](value)
    if number and "minimum" in schema and value < schema["minimum"]:
        found.append(f"{where}: {value!r} is less than {schema['minimum']}")
    if number and "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        found.append(f"{where}: {value!r} is not greater than {schema['exclusiveMinimum']}")
    if isinstance(value, str) and "pattern" in schema and re.search(schema["pattern"], value) is None:
        found.append(f"{where}: {value!r} does not match {schema['pattern']!r}")
    if isinstance(value, dict):
        for name in schema.get("required", ()):
            if name not in value:
                found.append(f"{where}: {name!r} is a required property")
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                found += _errors(item, properties[name], root, f"{where}.{name}")
            elif "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    found.append(f"{where}: additional property {name!r} is not allowed")
                elif isinstance(extra, Mapping):
                    found += _errors(item, extra, root, f"{where}.{name}")
    for index, item in enumerate(schema.get("allOf", ())):
        found += _errors(value, item, root, where)
    if "if" in schema:
        if not _errors(value, schema["if"], root, where):
            if "then" in schema:
                found += _errors(value, schema["then"], root, where)
    return found


def validate_record(record: Any, schema: Mapping[str, Any] | None = None) -> list[str]:
    """Every way one record departs from ``gm1.run-events/v1``; empty if none."""

    schema = load_schema() if schema is None else schema
    return _errors(record, schema, schema, "$")


def validate_lines(lines: Iterable[str], schema: Mapping[str, Any] | None = None) -> list[str]:
    """Check a whole log: every record, then the order the kinds come in."""

    schema = load_schema() if schema is None else schema
    found: list[str] = []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            found.append(f"line {number}: blank line")
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            found.append(f"line {number}: not JSON ({error})")
            continue
        found += [f"line {number}: {text}" for text in validate_record(record, schema)]
        if isinstance(record, dict):
            records.append(record)
    for position, record in enumerate(records):
        if record.get("seq") != position:
            found.append(f"record {position}: seq is {record.get('seq')!r}, not {position}")
            break
    ends = [position for position, record in enumerate(records) if record.get("kind") == "end"]
    if len(ends) > 1:
        found.append(f"{len(ends)} end records; a run ends once")
    if ends and ends[-1] != len(records) - 1:
        found.append("the end record is not the last record")
    return found


def validate_file(path: Path | str, schema: Mapping[str, Any] | None = None) -> list[str]:
    return validate_lines(Path(path).read_text(encoding="utf-8").splitlines(), schema)


def read_events(path: Path | str) -> list[dict[str, Any]]:
    """Every complete record of a log; a torn last line is left out."""

    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def main(argv: list[str] | None = None) -> int:
    """``python -m cm1core.run_events validate EVENTS.jsonl``."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m cm1core.run_events")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("validate", help=f"check a log against {RUN_EVENTS_SCHEMA}")
    command.add_argument("events", type=Path)
    arguments = parser.parse_args(argv)
    errors = validate_file(arguments.events)
    records = read_events(arguments.events)
    kinds: dict[str, int] = {}
    for record in records:
        kinds[record.get("kind", "?")] = kinds.get(record.get("kind", "?"), 0) + 1
    print(json.dumps({
        "schema": RUN_EVENTS_SCHEMA,
        "events": os.fspath(arguments.events),
        "records": len(records),
        "kinds": kinds,
        "valid": not errors,
        "errors": errors[:50],
    }, indent=2, sort_keys=True))
    return 0 if not errors else 2


__all__ = [
    "END_REASONS",
    "EVENT_KINDS",
    "HEARTBEAT_AFTER_S",
    "HEARTBEAT_LIMIT_S",
    "PHASES",
    "PROGRESS_INTERVAL_S",
    "RUN_EVENTS_SCHEMA",
    "RunEventLog",
    "STALE_AFTER_S",
    "directory_bytes",
    "host_name",
    "load_schema",
    "read_events",
    "validate_file",
    "validate_lines",
    "validate_record",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
