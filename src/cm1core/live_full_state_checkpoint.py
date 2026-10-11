"""Opaque live host/device transport for the Rust full-state checkpoint.

Rust owns the 76-carrier inventory, shapes, offsets, semantic validation,
envelope, and restore metadata.  This module only quiesces caller-supplied
views and moves their existing binary32 bytes to or from that boundary.  It
performs no atmospheric transform and imports NumPy/CuPy lazily.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import stat
import struct
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .gm1_runtime import CM1_MIN_NORMAL_DT_S, positive_normal_f32_bits


LIVE_FULL_STATE_CLI_ABI = "gpuwm-cm1.live-full-state-cli/v1"
LIVE_FULL_STATE_CARRIER_ABI = "gpuwm-cm1.live-full-state-carriers/v1"
LIVE_FULL_STATE_LAYOUT_SCHEMA = "gpuwm-cm1.live-full-state-layout/v1"
LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA = (
    "gpuwm-cm1.live-full-state-scheduler-extension/v1"
)
LIVE_FULL_STATE_CHECKPOINT_RECEIPT_SCHEMA = (
    "gpuwm-cm1.live-full-state-checkpoint-receipt/v1"
)
LIVE_FULL_STATE_RESTORE_MANIFEST_SCHEMA = (
    "gpuwm-cm1.live-full-state-restore-directory/v1"
)
LIVE_FULL_STATE_RESTORE_RECEIPT_SCHEMA = "gpuwm-cm1.live-full-state-restore-receipt/v1"
LIVE_FULL_STATE_CAPTURE_PHASE = "completed-large-step-boundary"
LIVE_FULL_STATE_RESTORE_PHASE = "before-arming-next-large-step-or-final-rk-transition"
LIVE_FULL_STATE_ARENA_FILE = "carrier-arena.f32le"
LIVE_FULL_STATE_MANIFEST_FILE = "manifest.json"
LIVE_FULL_STATE_RECEIPT_FILE = "receipt.json"
LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES = 2_144_851_136
LIVE_FULL_STATE_REAL_SHAPE_ZYX = (75, 300, 300)
# The large step this lane's own product run was captured at.  It is a recorded
# fact about that run, not a pin and not a default: CM1 carries dt as a
# default-REAL variable (``solve2.F:145``) and the artifact records the step it
# was written with in ``dt_f64_bits``, so a checkpoint is self-describing.
#
# Neither the capture nor the restore gives its step a default.  A defaulted
# ``resume_dt_s`` made every caller that simply omitted it compare the
# checkpoint against the half second, which silently refused every checkpoint
# captured at any other step while reading as if the run's own step had been
# checked.  Both are keyword-required, so a caller has to say which step it is
# capturing at or resuming with.
LIVE_FULL_STATE_REAL_DT_S = 0.5

TERRAIN_OPEN_CARRIER_NAMES = (
    "pi0",
    "prs0",
    "th0",
    "thv0",
    "rth0",
    "qv0",
    "rho0",
    "rr0",
    "rf0",
    "rrf0",
    "rho0s",
    "pi0s",
    "prs0s",
    "rth0s",
    "terrain_msl_halo",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "ppi",
    "pp3d",
    "ppx",
    "piadv",
    "ppterm",
    "ppten",
    "thv",
    "tha",
    "th3d",
    "thten",
    "thten1",
    "ua",
    "u3d",
    "uten",
    "uten1",
    "va",
    "v3d",
    "vten",
    "vten1",
    "wa",
    "w3d",
    "wten",
    "wten1",
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
)
WSM6_TRANSITION_ARRAY_NAMES = (
    "state.pp3d",
    "state.th3d",
    "state.prs",
    "state.rho",
    "state.rr",
    "state.q3d",
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
)
SCALAR_RESTART_ARRAY_NAMES = (
    "scalar.qa",
    "scalar.q3d",
    "scalar.qten",
    "scalar.qten1",
)
CLOSURE_RESTART_ARRAY_NAMES = ("closure.tkea",)
LIVE_FULL_STATE_ARRAY_NAMES = tuple(
    sorted(
        TERRAIN_OPEN_CARRIER_NAMES
        + WSM6_TRANSITION_ARRAY_NAMES
        + SCALAR_RESTART_ARRAY_NAMES
        + CLOSURE_RESTART_ARRAY_NAMES
    )
)
CORE_SCHEDULER_NAMES = frozenset(
    {
        "acoustic_substep",
        "completed_large_steps",
        "completed_wsm6_transitions",
        "large_step_boundary",
        "nrk",
        "radiation_history_initialized",
        "scalar_stage",
        "wsm6_transition_ready",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_EXTENSION_KEY = re.compile(r"[a-z0-9._-]{1,128}\Z")


@dataclass(frozen=True, slots=True)
class LiveFullStateCarrier:
    name: str
    shape: tuple[int, ...]
    offset_nbytes: int
    nbytes: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LiveFullStateLayout:
    carriers: tuple[LiveFullStateCarrier, ...]
    payload_nbytes: int
    contract_sha256: str

    def carrier(self, name: str) -> LiveFullStateCarrier:
        for carrier in self.carriers:
            if carrier.name == name:
                return carrier
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class LiveFullStateCheckpoint:
    path: Path
    contract_sha256: str
    artifact_sha256: str
    artifact_nbytes: int
    payload_sha256: str
    raw_arena_sha256: str
    raw_arena_nbytes: int
    step: int
    completed_wsm6_transitions: int
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RestoredLiveFullState:
    directory: Path
    arena: Any
    arrays: Mapping[str, Any]
    carriers: tuple[LiveFullStateCarrier, ...]
    payload_sha256: str
    payload_nbytes: int
    checkpoint_artifact_sha256: str
    checkpoint_contract_sha256: str
    scheduler_scalars: Mapping[str, Any]
    scheduler_extension: Mapping[str, Any]
    step: int
    model_time_f64_bits: str
    dt_f64_bits: str
    device_resident: bool
    ready_event: Any | None

    def array(self, name: str) -> Any:
        return self.arrays[name]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys are {sorted(actual)}; expected {sorted(expected)}"
        )
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lower-case hexadecimal characters")
    return value


def _finite_f64_bits(value: object, label: str, *, nonnegative: bool) -> float:
    if not isinstance(value, str) or _HEX16.fullmatch(value) is None:
        raise ValueError(f"{label} must be 16 lower-case hexadecimal f64 bits")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must encode an admitted finite value")
    return result


def _regular_file(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected a regular non-symlink file: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_canonical_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    _regular_file(path)
    raw = path.read_bytes()
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if _canonical_json_bytes(parsed) != raw:
        raise ValueError(f"JSON is not canonical in {path}")
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON root must be an object in {path}")
    return raw, parsed


def _run_rust(binary: Path, arguments: list[str]) -> str:
    _regular_file(binary)
    result = subprocess.run(
        [str(binary), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"cm1-live-full-state refused: {detail}")
    return result.stdout.strip()


def _require_rust_abi(binary: Path) -> None:
    lines = _run_rust(binary, ["--abi"]).splitlines()
    if lines != [LIVE_FULL_STATE_CLI_ABI, f"carrier_abi={LIVE_FULL_STATE_CARRIER_ABI}"]:
        raise ValueError("cm1-live-full-state CLI or carrier ABI changed")


def load_live_full_state_layout(rust_binary: Path | str) -> LiveFullStateLayout:
    """Ask Rust for the only admitted production arena layout."""

    binary = Path(rust_binary).resolve(strict=True)
    _require_rust_abi(binary)
    raw_text = _run_rust(binary, ["layout"])
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("Rust live full-state layout is not JSON") from error
    if _canonical_json_bytes(value) != raw_text.encode("ascii"):
        raise ValueError("Rust live full-state layout is not canonical JSON")
    value = _exact_keys(
        value,
        {
            "admission",
            "carrier_abi",
            "carrier_count",
            "carriers",
            "claim_limit",
            "contract_sha256",
            "dtype",
            "itemsize",
            "order",
            "payload_nbytes",
            "profile_id",
            "schema",
            "shape_zyx",
        },
        "live full-state layout",
    )
    if (
        value["schema"] != LIVE_FULL_STATE_LAYOUT_SCHEMA
        or value["carrier_abi"] != LIVE_FULL_STATE_CARRIER_ABI
        or value["admission"] != "real-product-300x300x75"
        or value["dtype"] != "<f4"
        or value["itemsize"] != 4
        or value["order"] != "C"
        or value["shape_zyx"] != list(LIVE_FULL_STATE_REAL_SHAPE_ZYX)
        or value["carrier_count"] != len(LIVE_FULL_STATE_ARRAY_NAMES)
        or value["payload_nbytes"] != LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES
        or value["profile_id"] != "cm1-21.1-real-live-full-state/v1"
    ):
        raise ValueError("Rust live full-state production layout contract changed")
    carriers = _parse_carriers(value["carriers"], hashes=False)
    _validate_layout_carriers(
        carriers, payload_nbytes=LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES
    )
    return LiveFullStateLayout(
        carriers,
        LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES,
        _sha256(value["contract_sha256"], "live full-state contract SHA-256"),
    )


def _parse_carriers(value: object, *, hashes: bool) -> tuple[LiveFullStateCarrier, ...]:
    if not isinstance(value, list):
        raise ValueError("live full-state carriers must be an array")
    carriers: list[LiveFullStateCarrier] = []
    for index, raw in enumerate(value):
        expected = {"name", "nbytes", "offset_nbytes", "shape"}
        if hashes:
            expected |= {"dtype", "itemsize", "order", "sha256"}
        raw = _exact_keys(raw, expected, f"live full-state carrier {index}")
        name = raw["name"]
        shape = raw["shape"]
        if not isinstance(name, str) or not isinstance(shape, list) or not shape:
            raise ValueError(f"live full-state carrier {index} name/shape is invalid")
        parsed_shape = tuple(
            _positive_int(extent, f"carrier {name} extent") for extent in shape
        )
        offset = _nonnegative_int(raw["offset_nbytes"], f"carrier {name} offset")
        nbytes = _positive_int(raw["nbytes"], f"carrier {name} nbytes")
        if nbytes != math.prod(parsed_shape) * 4 or offset % 4:
            raise ValueError(f"live full-state carrier {name} byte geometry changed")
        digest = None
        if hashes:
            if raw["dtype"] != "<f4" or raw["itemsize"] != 4 or raw["order"] != "C":
                raise ValueError(f"live full-state carrier {name} layout changed")
            digest = _sha256(raw["sha256"], f"carrier {name} SHA-256")
        carriers.append(
            LiveFullStateCarrier(name, parsed_shape, offset, nbytes, digest)
        )
    return tuple(carriers)


def _validate_layout_carriers(
    carriers: tuple[LiveFullStateCarrier, ...], *, payload_nbytes: int
) -> None:
    if tuple(carrier.name for carrier in carriers) != LIVE_FULL_STATE_ARRAY_NAMES:
        raise ValueError("live full-state carrier names or Rust order changed")
    offset = 0
    for carrier in carriers:
        if carrier.offset_nbytes != offset:
            raise ValueError(
                f"live full-state carrier {carrier.name} is not contiguous at {offset}"
            )
        offset += carrier.nbytes
    if offset != payload_nbytes:
        raise ValueError("live full-state carrier bytes do not cover the exact arena")


def _scheduler_extension_bytes(entries: Mapping[str, object] | None) -> bytes:
    if entries is None:
        entries = {}
    if not isinstance(entries, Mapping):
        raise TypeError("scheduler_entries must be a mapping")
    copied: dict[str, object] = {}
    for name, value in entries.items():
        if not isinstance(name, str) or _EXTENSION_KEY.fullmatch(name) is None:
            raise ValueError(
                "scheduler extension keys must use 1..128 lower-case [a-z0-9._-] characters"
            )
        if name in CORE_SCHEDULER_NAMES:
            raise ValueError(
                f"scheduler extension key {name!r} collides with core state"
            )
        copied[name] = value
    return _canonical_json_bytes(
        {"entries": copied, "schema": LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA}
    )


def _is_cuda_array(value: object) -> bool:
    return hasattr(value, "__cuda_array_interface__")


def _validate_views(
    views: Mapping[str, Any], layout: LiveFullStateLayout
) -> tuple[dict[str, Any], bool]:
    if not isinstance(views, Mapping):
        raise TypeError("live full-state views must be a mapping")
    if set(views) != set(LIVE_FULL_STATE_ARRAY_NAMES):
        missing = sorted(set(LIVE_FULL_STATE_ARRAY_NAMES) - set(views))
        extra = sorted(set(views) - set(LIVE_FULL_STATE_ARRAY_NAMES))
        raise ValueError(
            f"live full-state view inventory differs; missing={missing}, extra={extra}"
        )
    np = importlib.import_module("numpy")
    result: dict[str, Any] = {}
    any_cuda = False
    for carrier in layout.carriers:
        value = views[carrier.name]
        cuda = _is_cuda_array(value)
        any_cuda |= cuda
        if cuda:
            cupy = importlib.import_module("cupy")
            if not isinstance(value, cupy.ndarray):
                raise TypeError(
                    f"live full-state view {carrier.name} exposes CUDA but is not cupy.ndarray"
                )
            dtype_ok = value.dtype == cupy.dtype("float32")
            contiguous = bool(value.flags.c_contiguous)
        else:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"live full-state view {carrier.name} must be numpy.ndarray or cupy.ndarray"
                )
            dtype_ok = value.dtype == np.dtype("float32") and value.dtype.isnative
            contiguous = bool(value.flags.c_contiguous)
        if not dtype_ok or tuple(value.shape) != carrier.shape or not contiguous:
            raise ValueError(
                f"live full-state view {carrier.name} must be native C-order float32 shape {carrier.shape}"
            )
        result[carrier.name] = value
    return result, any_cuda


def _quiesce_views(
    views: Mapping[str, Any], *, any_cuda: bool, ready_event: Any | None
) -> None:
    if ready_event is not None:
        synchronize = getattr(ready_event, "synchronize", None)
        if not callable(synchronize):
            raise TypeError("ready_event must expose synchronize()")
        synchronize()
    elif any_cuda:
        cupy = importlib.import_module("cupy")
        cupy.cuda.get_current_stream().synchronize()


def _host_bytes(value: Any) -> memoryview:
    np = importlib.import_module("numpy")
    if _is_cuda_array(value):
        cupy = importlib.import_module("cupy")
        value = cupy.asnumpy(value)
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise RuntimeError("validated live full-state view changed before capture")
    return memoryview(value).cast("B")


def _write_arena(
    path: Path, views: Mapping[str, Any], layout: LiveFullStateLayout
) -> str:
    hasher = hashlib.sha256()
    with path.open("xb", buffering=0) as stream:
        for carrier in layout.carriers:
            raw = _host_bytes(views[carrier.name])
            if len(raw) != carrier.nbytes:
                raise RuntimeError(
                    f"live full-state view {carrier.name} byte count changed during capture"
                )
            stream.write(raw)
            hasher.update(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.stat().st_size != layout.payload_nbytes:
        raise RuntimeError("live full-state raw arena size changed after capture")
    return hasher.hexdigest()


def _temporary_path(parent: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=parent, prefix=".cm1-live-full-state-", suffix=suffix
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def capture_live_full_state_checkpoint_new(
    views: Mapping[str, Any],
    output: Path | str,
    *,
    rust_binary: Path | str,
    initialized_product_restart_sha256: str,
    initialized_product_manifest_sha256: str,
    initialized_product_payload_sha256: str,
    terrain_open_restart_sha256: str,
    wsm6_transition_restart_sha256: str,
    step: int,
    model_time_s: float,
    completed_wsm6_transitions: int,
    scheduler_entries: Mapping[str, object] | None = None,
    ready_event: Any | None = None,
    dt_s: float,
) -> LiveFullStateCheckpoint:
    """Capture one quiescent completed-step mapping into a new Rust artifact."""

    binary = Path(rust_binary).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"live full-state checkpoint already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent does not exist: {output.parent}")
    hashes = [
        _sha256(initialized_product_restart_sha256, "product restart SHA-256"),
        _sha256(initialized_product_manifest_sha256, "product manifest SHA-256"),
        _sha256(initialized_product_payload_sha256, "product payload SHA-256"),
        _sha256(terrain_open_restart_sha256, "terrain/open restart SHA-256"),
        _sha256(wsm6_transition_restart_sha256, "WSM6 restart SHA-256"),
    ]
    step = _nonnegative_int(step, "step")
    completed_wsm6_transitions = _nonnegative_int(
        completed_wsm6_transitions, "completed_wsm6_transitions"
    )
    if step != completed_wsm6_transitions:
        raise ValueError(
            "completed-step capture requires step == completed_wsm6_transitions"
        )
    if (
        isinstance(model_time_s, bool)
        or not isinstance(model_time_s, (int, float))
        or not math.isfinite(float(model_time_s))
        or float(model_time_s) < 0.0
    ):
        raise ValueError("model_time_s must be finite and nonnegative")
    if positive_normal_f32_bits(dt_s) is None:
        raise ValueError(
            "live full-state dt_s must be a finite step of at least "
            f"{CM1_MIN_NORMAL_DT_S!r} s; the artifact records it as the step "
            "every carrier in this arena was advanced by, and a step that is "
            "not a positive normal default-REAL word could not have produced "
            "them"
        )
    layout = load_live_full_state_layout(binary)
    checked, any_cuda = _validate_views(views, layout)
    extension = _scheduler_extension_bytes(scheduler_entries)
    _quiesce_views(checked, any_cuda=any_cuda, ready_event=ready_event)
    arena_path = _temporary_path(output.parent, ".f32le")
    extension_path = _temporary_path(output.parent, ".json")
    try:
        raw_sha256 = _write_arena(arena_path, checked, layout)
        with extension_path.open("xb", buffering=0) as stream:
            stream.write(extension)
            stream.flush()
            os.fsync(stream.fileno())
        text = _run_rust(
            binary,
            [
                "build",
                str(arena_path),
                str(output),
                raw_sha256,
                *hashes,
                str(step),
                repr(float(model_time_s)),
                repr(float(dt_s)),
                str(completed_wsm6_transitions),
                str(extension_path),
            ],
        )
    finally:
        arena_path.unlink(missing_ok=True)
        extension_path.unlink(missing_ok=True)
    if not output.exists():
        raise RuntimeError("Rust reported success without publishing the checkpoint")
    _regular_file(output)
    try:
        receipt = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Rust live full-state checkpoint receipt is not JSON"
        ) from error
    receipt = _exact_keys(
        receipt,
        {
            "artifact_nbytes",
            "artifact_sha256",
            "carrier_abi",
            "carrier_count",
            "capture_phase",
            "claim_limit",
            "completed_wsm6_transitions",
            "contract_sha256",
            "payload_sha256",
            "publication_contract",
            "raw_arena_nbytes",
            "raw_arena_sha256",
            "schema",
            "source",
            "step",
        },
        "live full-state checkpoint receipt",
    )
    if (
        receipt["schema"] != LIVE_FULL_STATE_CHECKPOINT_RECEIPT_SCHEMA
        or receipt["carrier_abi"] != LIVE_FULL_STATE_CARRIER_ABI
        or receipt["carrier_count"] != len(LIVE_FULL_STATE_ARRAY_NAMES)
        or receipt["capture_phase"] != LIVE_FULL_STATE_CAPTURE_PHASE
        or receipt["raw_arena_sha256"] != raw_sha256
        or receipt["raw_arena_nbytes"] != layout.payload_nbytes
        or receipt["step"] != step
        or receipt["completed_wsm6_transitions"] != completed_wsm6_transitions
        or receipt["contract_sha256"] != layout.contract_sha256
    ):
        raise ValueError("Rust live full-state checkpoint receipt changed")
    artifact_sha256 = _sha256(receipt["artifact_sha256"], "artifact SHA-256")
    artifact_nbytes = _positive_int(receipt["artifact_nbytes"], "artifact_nbytes")
    if (
        output.stat().st_size != artifact_nbytes
        or _file_sha256(output) != artifact_sha256
    ):
        raise ValueError("published checkpoint differs from the Rust receipt")
    return LiveFullStateCheckpoint(
        path=output,
        contract_sha256=_sha256(receipt["contract_sha256"], "contract SHA-256"),
        artifact_sha256=artifact_sha256,
        artifact_nbytes=artifact_nbytes,
        payload_sha256=_sha256(receipt["payload_sha256"], "payload SHA-256"),
        raw_arena_sha256=raw_sha256,
        raw_arena_nbytes=layout.payload_nbytes,
        step=step,
        completed_wsm6_transitions=completed_wsm6_transitions,
        receipt=MappingProxyType(receipt),
    )


def compose_live_full_state_views(
    terrain_open: Mapping[str, Any] | object,
    scalar: Mapping[str, Any],
    wsm6: Mapping[str, Any] | object,
    closure: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Merge scheduler-owned exact namespaces without copying array data.

    ``wsm6`` may be a restart mapping or an initialized transition exposing
    ``capture_restart()``.  ``terrain_open`` may be an exact mapping or an
    object whose ``arrays`` and ``metrics`` mappings together expose the
    Rust-named terrain/open inventory.  ``closure`` supplies the completed-step
    prognostic ``closure.tkea`` view; derived SGS arrays remain recomputable.
    """

    if hasattr(wsm6, "capture_restart"):
        wsm6 = wsm6.capture_restart().arrays
    if not isinstance(wsm6, Mapping):
        arrays = getattr(wsm6, "arrays", None)
        if isinstance(arrays, Mapping):
            wsm6 = arrays
    terrain_parts: list[Mapping[str, Any]] = []
    if isinstance(terrain_open, Mapping):
        terrain_parts.append(terrain_open)
    else:
        for attribute in ("arrays", "metrics", "restart_arrays"):
            value = getattr(terrain_open, attribute, None)
            if isinstance(value, Mapping):
                terrain_parts.append(value)
    if (
        not terrain_parts
        or not isinstance(scalar, Mapping)
        or not isinstance(wsm6, Mapping)
        or not isinstance(closure, Mapping)
    ):
        raise TypeError(
            "terrain_open, scalar, wsm6, and closure must expose array mappings"
        )
    result: dict[str, Any] = {}
    for source in (*terrain_parts, scalar, wsm6, closure):
        for name, value in source.items():
            if name not in LIVE_FULL_STATE_ARRAY_NAMES:
                continue
            if name in result and result[name] is not value:
                raise ValueError(f"live full-state carrier {name} has two providers")
            result[name] = value
    if set(result) != set(LIVE_FULL_STATE_ARRAY_NAMES):
        missing = sorted(set(LIVE_FULL_STATE_ARRAY_NAMES) - set(result))
        raise ValueError(f"composed live full-state views are missing {missing}")
    return MappingProxyType(result)


def restore_live_full_state_checkpoint_new(
    checkpoint: LiveFullStateCheckpoint | Path | str,
    output_directory: Path | str,
    *,
    rust_binary: Path | str,
    expected_contract_sha256: str | None = None,
    expected_artifact_sha256: str | None = None,
    expected_initialized_product_restart_sha256: str,
    resume_dt_s: float,
    device: bool = False,
) -> RestoredLiveFullState:
    """Ask Rust to revalidate/export a checkpoint, then expose exact views."""

    if isinstance(checkpoint, LiveFullStateCheckpoint):
        path = checkpoint.path
        contract_sha = checkpoint.contract_sha256
        artifact_sha = checkpoint.artifact_sha256
    else:
        path = Path(checkpoint).resolve(strict=True)
        if expected_contract_sha256 is None or expected_artifact_sha256 is None:
            raise ValueError(
                "path restore requires independent expected contract and artifact hashes"
            )
        contract_sha = expected_contract_sha256
        artifact_sha = expected_artifact_sha256
    contract_sha = _sha256(contract_sha, "expected contract SHA-256")
    artifact_sha = _sha256(artifact_sha, "expected artifact SHA-256")
    product_restart_sha = _sha256(
        expected_initialized_product_restart_sha256,
        "expected initialized-product restart SHA-256",
    )
    _regular_file(path)
    binary = Path(rust_binary).resolve(strict=True)
    _require_rust_abi(binary)
    output = Path(output_directory).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"live full-state restore destination exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"restore parent does not exist: {output.parent}")
    text = _run_rust(
        binary,
        [
            "restore",
            str(path),
            str(output),
            contract_sha,
            artifact_sha,
            product_restart_sha,
        ],
    )
    try:
        command_receipt = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Rust live full-state restore receipt is not JSON") from error
    return load_restored_live_full_state(
        output,
        expected_manifest_sha256=_sha256(
            command_receipt.get("manifest_sha256"), "restore manifest SHA-256"
        ),
        expected_payload_sha256=_sha256(
            command_receipt.get("payload_sha256"), "restore payload SHA-256"
        ),
        expected_checkpoint_artifact_sha256=artifact_sha,
        expected_checkpoint_contract_sha256=contract_sha,
        resume_dt_s=resume_dt_s,
        device=device,
    )


def load_restored_live_full_state(
    directory: Path | str,
    *,
    expected_manifest_sha256: str,
    expected_payload_sha256: str,
    expected_checkpoint_artifact_sha256: str,
    expected_checkpoint_contract_sha256: str,
    resume_dt_s: float,
    device: bool = False,
) -> RestoredLiveFullState:
    """Strictly load a Rust restore directory as host or one device arena.

    ``resume_dt_s`` is the large step the caller intends to continue with, and
    is required: it is compared against the step the checkpoint was written at
    rather than against a constant, because a restore that silently changes
    ``dt`` would place every subsequent history frame at a model time the state
    never passed through, and would hand ``solve2.F:1569-1576`` a held-rate
    snapshot taken over a different step.  With a default it would have been
    compared against the half second for every caller that omitted it, which is
    the same pin under another name.
    """

    if type(device) is not bool:
        raise TypeError("device must be a strict boolean")
    if positive_normal_f32_bits(resume_dt_s) is None:
        raise ValueError(
            "resume_dt_s must be a finite step of at least "
            f"{CM1_MIN_NORMAL_DT_S!r} s"
        )
    expected_manifest_sha256 = _sha256(
        expected_manifest_sha256, "expected restore manifest SHA-256"
    )
    expected_payload_sha256 = _sha256(
        expected_payload_sha256, "expected restore payload SHA-256"
    )
    expected_checkpoint_artifact_sha256 = _sha256(
        expected_checkpoint_artifact_sha256,
        "expected checkpoint artifact SHA-256",
    )
    expected_checkpoint_contract_sha256 = _sha256(
        expected_checkpoint_contract_sha256,
        "expected checkpoint contract SHA-256",
    )
    root = Path(directory).resolve(strict=True)
    if Path(directory).is_symlink() or not root.is_dir():
        raise ValueError("restore path must be a real directory")
    if {entry.name for entry in root.iterdir()} != {
        LIVE_FULL_STATE_ARENA_FILE,
        LIVE_FULL_STATE_MANIFEST_FILE,
        LIVE_FULL_STATE_RECEIPT_FILE,
    }:
        raise ValueError("live full-state restore directory inventory differs")
    arena_path = root / LIVE_FULL_STATE_ARENA_FILE
    manifest_path = root / LIVE_FULL_STATE_MANIFEST_FILE
    receipt_path = root / LIVE_FULL_STATE_RECEIPT_FILE
    _regular_file(arena_path)
    manifest_raw, manifest = _read_canonical_json(manifest_path)
    _, receipt = _read_canonical_json(receipt_path)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha != expected_manifest_sha256:
        raise ValueError("restore manifest differs from independent SHA-256")
    receipt = _exact_keys(
        receipt,
        {
            "carrier_abi",
            "carrier_count",
            "checkpoint_artifact_sha256",
            "checkpoint_contract_sha256",
            "claim_limit",
            "manifest_nbytes",
            "manifest_sha256",
            "payload_nbytes",
            "payload_sha256",
            "publication_contract",
            "schema",
            "step",
        },
        "live full-state restore receipt",
    )
    if (
        receipt["schema"] != LIVE_FULL_STATE_RESTORE_RECEIPT_SCHEMA
        or receipt["carrier_abi"] != LIVE_FULL_STATE_CARRIER_ABI
        or receipt["carrier_count"] != len(LIVE_FULL_STATE_ARRAY_NAMES)
        or receipt["manifest_nbytes"] != len(manifest_raw)
        or receipt["manifest_sha256"] != manifest_sha
        or receipt["checkpoint_artifact_sha256"] != expected_checkpoint_artifact_sha256
        or receipt["checkpoint_contract_sha256"] != expected_checkpoint_contract_sha256
    ):
        raise ValueError("live full-state restore receipt contract changed")
    manifest = _exact_keys(
        manifest,
        {
            "capture_phase",
            "carrier_abi",
            "carriers",
            "claim_limit",
            "dtype",
            "itemsize",
            "order",
            "payload",
            "restore_phase",
            "scheduler_extension",
            "scheduler_scalars",
            "schema",
            "source",
            "source_checkpoint",
        },
        "live full-state restore manifest",
    )
    if (
        manifest["schema"] != LIVE_FULL_STATE_RESTORE_MANIFEST_SCHEMA
        or manifest["carrier_abi"] != LIVE_FULL_STATE_CARRIER_ABI
        or manifest["capture_phase"] != LIVE_FULL_STATE_CAPTURE_PHASE
        or manifest["restore_phase"] != LIVE_FULL_STATE_RESTORE_PHASE
        or manifest["dtype"] != "<f4"
        or manifest["itemsize"] != 4
        or manifest["order"] != "C"
    ):
        raise ValueError("live full-state restore manifest contract changed")
    payload = _exact_keys(
        manifest["payload"], {"file", "nbytes", "sha256"}, "restore payload"
    )
    payload_nbytes = _positive_int(payload["nbytes"], "restore payload nbytes")
    payload_sha = _sha256(payload["sha256"], "restore payload SHA-256")
    if (
        payload["file"] != LIVE_FULL_STATE_ARENA_FILE
        or payload_sha != expected_payload_sha256
        or payload_nbytes != arena_path.stat().st_size
        or receipt["payload_sha256"] != payload_sha
        or receipt["payload_nbytes"] != payload_nbytes
    ):
        raise ValueError("live full-state restore payload identity changed")
    if _file_sha256(arena_path) != payload_sha:
        raise ValueError("live full-state restore arena differs from SHA-256")
    carriers = _parse_carriers(manifest["carriers"], hashes=True)
    _validate_layout_carriers(carriers, payload_nbytes=payload_nbytes)
    source_checkpoint = _exact_keys(
        manifest["source_checkpoint"],
        {
            "artifact_sha256",
            "contract_sha256",
            "dt_f64_bits",
            "model_time_f64_bits",
            "step",
        },
        "restore source checkpoint",
    )
    if (
        source_checkpoint["artifact_sha256"] != expected_checkpoint_artifact_sha256
        or source_checkpoint["contract_sha256"] != expected_checkpoint_contract_sha256
    ):
        raise ValueError("restore manifest belongs to another checkpoint")
    step = _nonnegative_int(source_checkpoint["step"], "restore step")
    checkpoint_dt_s = _finite_f64_bits(
        source_checkpoint["dt_f64_bits"], "restore dt", nonnegative=False
    )
    if checkpoint_dt_s != float(resume_dt_s):
        raise ValueError(
            f"this checkpoint was written at dt={checkpoint_dt_s!r} s but the "
            f"run would resume at dt={float(resume_dt_s)!r} s; the model clock "
            "would advance by a step the carried state was never produced with, "
            "so every history frame after the restore would be stamped at a "
            "model time the solution never passed through"
        )
    _finite_f64_bits(
        source_checkpoint["model_time_f64_bits"],
        "restore model time",
        nonnegative=True,
    )
    source = _exact_keys(
        manifest["source"],
        {
            "initialized_product_manifest_sha256",
            "initialized_product_payload_sha256",
            "initialized_product_restart_sha256",
            "terrain_open_restart_sha256",
            "wsm6_transition_restart_sha256",
        },
        "restore source",
    )
    for name, digest in source.items():
        _sha256(digest, f"restore source {name}")
    scheduler_scalars = manifest["scheduler_scalars"]
    if (
        not isinstance(scheduler_scalars, dict)
        or set(scheduler_scalars) != CORE_SCHEDULER_NAMES
    ):
        raise ValueError("restore scheduler scalar inventory changed")
    if scheduler_scalars != {
        "acoustic_substep": 0,
        "completed_large_steps": step,
        "completed_wsm6_transitions": step,
        "large_step_boundary": True,
        "nrk": 0,
        "radiation_history_initialized": True,
        "scalar_stage": 0,
        "wsm6_transition_ready": True,
    }:
        raise ValueError("restore scheduler is not at a completed large-step boundary")
    extension = _exact_keys(
        manifest["scheduler_extension"],
        {"entries", "schema"},
        "restore scheduler extension",
    )
    if extension[
        "schema"
    ] != LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA or not isinstance(
        extension["entries"], dict
    ):
        raise ValueError("restore scheduler extension changed")
    for name in extension["entries"]:
        if _EXTENSION_KEY.fullmatch(name) is None or name in CORE_SCHEDULER_NAMES:
            raise ValueError("restore scheduler extension key is not admitted")
    np = importlib.import_module("numpy")
    host_arena = np.memmap(arena_path, dtype=np.uint8, mode="r")
    if device:
        cupy = importlib.import_module("cupy")
        arena = cupy.asarray(host_arena)
        arrays = {
            carrier.name: arena[
                carrier.offset_nbytes : carrier.offset_nbytes + carrier.nbytes
            ]
            .view(cupy.float32)
            .reshape(carrier.shape)
            for carrier in carriers
        }
        event = cupy.cuda.Event(disable_timing=True)
        event.record(cupy.cuda.get_current_stream())
    else:
        arena = host_arena
        arrays = {
            carrier.name: np.ndarray(
                carrier.shape,
                dtype=np.dtype("<f4"),
                buffer=host_arena,
                offset=carrier.offset_nbytes,
                order="C",
            )
            for carrier in carriers
        }
        event = None
    return RestoredLiveFullState(
        directory=root,
        arena=arena,
        arrays=MappingProxyType(arrays),
        carriers=carriers,
        payload_sha256=payload_sha,
        payload_nbytes=payload_nbytes,
        checkpoint_artifact_sha256=expected_checkpoint_artifact_sha256,
        checkpoint_contract_sha256=expected_checkpoint_contract_sha256,
        scheduler_scalars=MappingProxyType(dict(scheduler_scalars)),
        scheduler_extension=MappingProxyType(dict(extension["entries"])),
        step=step,
        model_time_f64_bits=source_checkpoint["model_time_f64_bits"],
        dt_f64_bits=source_checkpoint["dt_f64_bits"],
        device_resident=device,
        ready_event=event,
    )


def copy_restored_live_full_state_into(
    restored: RestoredLiveFullState,
    destinations: Mapping[str, Any],
    *,
    stream: Any | None = None,
) -> Any | None:
    """Copy every restored word into pre-existing scheduler-owned live views."""

    if not isinstance(restored, RestoredLiveFullState):
        raise TypeError("restored must be a RestoredLiveFullState")
    layout = LiveFullStateLayout(
        restored.carriers,
        restored.payload_nbytes,
        restored.checkpoint_contract_sha256,
    )
    checked, any_cuda = _validate_views(destinations, layout)
    np = importlib.import_module("numpy")
    cupy = importlib.import_module("cupy") if any_cuda else None
    if restored.ready_event is not None:
        restored.ready_event.synchronize()
    for carrier in restored.carriers:
        source = restored.arrays[carrier.name]
        destination = checked[carrier.name]
        if _is_cuda_array(destination):
            assert cupy is not None
            target_stream = stream or cupy.cuda.get_current_stream()
            with target_stream:
                cupy.copyto(destination, cupy.asarray(source))
        else:
            if _is_cuda_array(source):
                assert cupy is not None
                source = cupy.asnumpy(source)
            np.copyto(destination, source, casting="no")
    if any_cuda:
        assert cupy is not None
        target_stream = stream or cupy.cuda.get_current_stream()
        event = cupy.cuda.Event(disable_timing=True)
        event.record(target_stream)
        return event
    return None


__all__ = [
    "CLOSURE_RESTART_ARRAY_NAMES",
    "LIVE_FULL_STATE_ARRAY_NAMES",
    "LIVE_FULL_STATE_CARRIER_ABI",
    "LIVE_FULL_STATE_CLI_ABI",
    "LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES",
    "LIVE_FULL_STATE_REAL_SHAPE_ZYX",
    "LiveFullStateCarrier",
    "LiveFullStateCheckpoint",
    "LiveFullStateLayout",
    "RestoredLiveFullState",
    "capture_live_full_state_checkpoint_new",
    "compose_live_full_state_views",
    "copy_restored_live_full_state_into",
    "load_live_full_state_layout",
    "load_restored_live_full_state",
    "restore_live_full_state_checkpoint_new",
]
