"""Restart-continuous lifecycle for CM1 r21.1 adaptive moving domains.

The lifecycle joins the already sealed flat serial periodic and four-code-2
radiative/open adaptive-move operators to an exact checkpoint envelope.  The
envelope persists every authoritative controller word, every wind carrier
shifted by the controller, the exact fixed-grid coordinate carriers, and (for
the open branch) the four separately evolved ``radbc*`` histories.

The checkpoint is a bounded transport representation, not CM1's complete
Fortran restart file.  Its field inventory is source-bound to CM1's binary and
NetCDF restart payloads.  Terrain plus moving-domain, MPI, rigid boundaries,
and periodic/open branch substitution remain refused.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from . import cuda_adaptive_move as periodic
from . import cuda_adaptive_move_open as radiative_open
from .cuda_stretch_z1 import StretchZ1MetricProof
from .errors import refuse


ADAPTIVE_MOVE_RESTART_LIFECYCLE_SCHEMA = "gpuwm-cm1.adaptive-move-restart-lifecycle/v1"
ADAPTIVE_MOVE_RESTART_LIFECYCLE_ABI = (
    "gpuwm-cm1.adaptive-move-restart-lifecycle-canonical/v1"
)
ADAPTIVE_MOVE_RESTART_LIFECYCLE_CLAIM_LIMIT = (
    "CM1 r21.1 flat nproc=1 stretch_z=1 adaptive moving-domain checkpoint "
    "continuity for exactly either four-periodic or four-code-2 radiative/open "
    "boundaries. Persists typed controller state, fixed-grid coordinates, all "
    "source-shifted u/v/ug/vg carriers, and open radbc histories when selected. "
    "No terrain+imove, MPI, rigid boundary, branch substitution, nesting, "
    "external future-LBC interpolation, full-model restart, or forecast claim"
)

PERIODIC_OPERATOR_COMMIT = "8914a3267049c7a61f9a6e0f533c15e816fd56dc"
PERIODIC_OPERATOR_SOURCE_SHA256 = (
    "319d847b1b72d300ec03144c67ae32ad5e58857655e59405c5cd5770e2d87344"
)
OPEN_OPERATOR_COMMIT = "2eab8259a06a6edc8d95a7dd167be7ce8c6dee35"
OPEN_OPERATOR_SOURCE_SHA256 = (
    "0a1493aade436659963df024c7c01e70a7739798d880e6cba1f07b0b62154ca3"
)

AdaptiveMoveBranch = Literal["periodic", "radiative-open"]
BRANCHES: tuple[AdaptiveMoveBranch, ...] = ("periodic", "radiative-open")
COORDINATE_FIELDS = ("xh", "yh", "xfref", "yfref")


@dataclass(frozen=True, slots=True)
class AdaptiveMoveLifecycleCheckpoint:
    manifest_bytes: bytes
    payload: bytes
    manifest_sha256: str
    payload_sha256: str
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class RestoredAdaptiveMoveLifecycle:
    branch: AdaptiveMoveBranch
    completed_steps: int
    model_time: float
    state: periodic.AdaptiveMoveState
    winds: Mapping[str, Any]
    coordinates: Mapping[str, Any]
    open_histories: Mapping[str, Any] | None
    checkpoint_identity: Mapping[str, object]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _f64_bits(value: float) -> str:
    return f"{struct.unpack('>Q', struct.pack('>d', value))[0]:016x}"


def _f64_from_bits(bits: object) -> float:
    if not isinstance(bits, str) or len(bits) != 16:
        raise ValueError("model_time_f64_bits must be 16 lowercase hex characters")
    try:
        raw = bytes.fromhex(bits)
    except ValueError as error:
        raise ValueError("model_time_f64_bits is not hexadecimal") from error
    if bits != bits.lower():
        raise ValueError("model_time_f64_bits must use lowercase hex")
    value = struct.unpack(">d", raw)[0]
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("restored model time must be finite and nonnegative")
    return value


def _require_branch(branch: object) -> AdaptiveMoveBranch:
    if branch not in BRANCHES:
        refuse(
            code="ARWEN-ADAPTIVE-MOVE-RESTART-BRANCH",
            gate="adaptive-moving-domain-restart-lifecycle",
            option="branch",
            value=branch,
            reason="only exact periodic or four-code-2 radiative-open branches are sealed",
            remedy="select branch='periodic' or branch='radiative-open'",
        )
    return branch  # type: ignore[return-value]


def _branch_identity(branch: AdaptiveMoveBranch) -> dict[str, object]:
    if branch == "periodic":
        periodic.require_adaptive_move_selection()
        selection = {
            name: getattr(periodic.DEFAULT_ADAPTIVE_MOVE_SELECTION, name)
            for name in periodic.DEFAULT_ADAPTIVE_MOVE_SELECTION.__dataclass_fields__
        }
        return {
            "schema": periodic.CUDA_ADAPTIVE_MOVE_SCHEMA,
            "abi": periodic.CUDA_ADAPTIVE_MOVE_ABI,
            "selection": selection,
            "module_commit": PERIODIC_OPERATOR_COMMIT,
            "module_source_sha256": PERIODIC_OPERATOR_SOURCE_SHA256,
            "kernel_source_sha256": periodic.CUDA_ADAPTIVE_MOVE_SOURCE_SHA256,
        }
    radiative_open.require_adaptive_move_open_selection()
    contract = radiative_open.adaptive_move_open_contract()
    return {
        "schema": radiative_open.CUDA_ADAPTIVE_MOVE_OPEN_SCHEMA,
        "abi": radiative_open.CUDA_ADAPTIVE_MOVE_OPEN_ABI,
        "selection": dict(contract["selection"]),
        "contract_sha256": contract["contract_sha256"],
        "module_commit": OPEN_OPERATOR_COMMIT,
        "module_source_sha256": OPEN_OPERATOR_SOURCE_SHA256,
        "kernel_source_sha256": radiative_open.CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_SHA256,
    }


def adaptive_move_restart_lifecycle_contract() -> Mapping[str, object]:
    body = {
        "schema": ADAPTIVE_MOVE_RESTART_LIFECYCLE_SCHEMA,
        "abi": ADAPTIVE_MOVE_RESTART_LIFECYCLE_ABI,
        "branches": {branch: _branch_identity(branch) for branch in BRANCHES},
        "controller_restart": periodic.adaptive_move_restart_contract(),
        "checkpoint_inventory": {
            "state": ["state.f32", "state.f64", "state.i32"],
            "coordinates": list(COORDINATE_FIELDS),
            "winds": list(periodic.WIND_FIELDS),
            "open_only": list(radiative_open.OPEN_HISTORY_FIELDS),
        },
        "source_order": {
            "completed_step_domain_location": periodic.SOURCE_ANCHORS[
                "completed_step_domain_location"
            ],
            "diagnosis_cadence": periodic.SOURCE_ANCHORS["diagnosis_cadence"],
            "controller_and_carrier_shift": periodic.SOURCE_ANCHORS[
                "adaptive_controller"
            ],
            "restart_binary": periodic.SOURCE_ANCHORS["restart_payload_binary"],
            "restart_netcdf": periodic.SOURCE_ANCHORS["restart_payload_netcdf"],
            "open_history_restart_write": radiative_open.SOURCE_ANCHORS[
                "open_history_restart_write"
            ],
            "open_history_restart_read": radiative_open.SOURCE_ANCHORS[
                "open_history_restart_read"
            ],
        },
        "claim_limit": ADAPTIVE_MOVE_RESTART_LIFECYCLE_CLAIM_LIMIT,
    }
    encoded = _canonical_json(body)
    return MappingProxyType(
        {**body, "contract_sha256": hashlib.sha256(encoded).hexdigest()}
    )


def _require_completed_step_clock(
    completed_steps: object, model_time: object
) -> tuple[int, float]:
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 0
    ):
        raise ValueError("completed_steps must be a nonnegative integer")
    if isinstance(model_time, bool) or not isinstance(model_time, (int, float)):
        raise TypeError("model_time must be a real scalar")
    model_time64 = float(model_time)
    if not math.isfinite(model_time64) or model_time64 < 0.0:
        raise ValueError("model_time must be finite and nonnegative")
    return completed_steps, model_time64


def _require_checkpoint_inputs(
    branch: AdaptiveMoveBranch,
    state: periodic.AdaptiveMoveState,
    winds: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    open_histories: Mapping[str, Any] | None,
) -> tuple[tuple[Any, Any, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    np = importlib.import_module("numpy")
    state_arrays = periodic.pack_adaptive_move_state(state)
    if not isinstance(winds, Mapping) or set(winds) != set(periodic.WIND_FIELDS):
        raise ValueError(
            "checkpoint wind inventory must be exactly the eight CM1 carriers"
        )
    checked_winds = periodic._require_host_winds(winds)
    u_shape, v_shape = checked_winds["u0"].shape, checked_winds["v0"].shape
    nk, nj, ni = u_shape[0], u_shape[1], v_shape[2]
    if not isinstance(coordinates, Mapping) or set(coordinates) != set(
        COORDINATE_FIELDS
    ):
        raise ValueError("coordinate inventory must be exactly xh/yh/xfref/yfref")
    zero_pressure = np.zeros((nj, ni), dtype=np.float32)
    _, xh, yh, xfref, yfref, _, _ = periodic._require_center_arrays(
        zero_pressure,
        coordinates["xh"],
        coordinates["yh"],
        coordinates["xfref"],
        coordinates["yfref"],
    )
    checked_coordinates = {"xh": xh, "yh": yh, "xfref": xfref, "yfref": yfref}
    all_arrays = (*state_arrays, *checked_winds.values(), *checked_coordinates.values())
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(all_arrays)
        for right in all_arrays[index + 1 :]
    ):
        raise ValueError("checkpoint state/wind/coordinate carriers must not alias")
    if branch == "periodic":
        if open_histories is not None:
            refuse(
                code="ARWEN-ADAPTIVE-MOVE-RESTART-BRANCH-SUBSTITUTION",
                gate="adaptive-moving-domain-restart-lifecycle",
                option="open_histories",
                value="supplied-for-periodic",
                reason="periodic restart inventory cannot contain radiative histories",
                remedy="remove open histories or select the radiative-open branch",
            )
        return state_arrays, checked_winds, checked_coordinates, None
    if open_histories is None:
        raise ValueError("radiative-open checkpoint requires all four radbc histories")
    checked_histories = radiative_open._require_host_open_histories(
        open_histories,
        ni=ni,
        nj=nj,
        nk=nk,
        ngxy=radiative_open.DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION.ngxy,
    )
    if any(
        np.shares_memory(left, right)
        for left in checked_histories.values()
        for right in (*checked_winds.values(), *checked_coordinates.values())
    ):
        raise ValueError(
            "open histories must not alias restart wind/coordinate carriers"
        )
    return state_arrays, checked_winds, checked_coordinates, checked_histories


def _entry_dtype(value: Any) -> str:
    np = importlib.import_module("numpy")
    if value.dtype == np.dtype("float32"):
        return "little-endian-f32"
    if value.dtype == np.dtype("float64"):
        return "little-endian-f64"
    if value.dtype == np.dtype("int32"):
        return "little-endian-i32"
    raise ValueError(f"unsupported checkpoint dtype {value.dtype}")


def capture_adaptive_move_lifecycle_checkpoint(
    branch: AdaptiveMoveBranch,
    state: periodic.AdaptiveMoveState,
    winds: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    *,
    completed_steps: int,
    model_time: float,
    vertical_grid_proof: StretchZ1MetricProof,
    open_histories: Mapping[str, Any] | None = None,
) -> AdaptiveMoveLifecycleCheckpoint:
    """Capture one exact completed-step controller/carrier checkpoint."""

    branch = _require_branch(branch)
    periodic.validate_adaptive_move_stretch_proof(vertical_grid_proof)
    completed_steps, model_time = _require_completed_step_clock(
        completed_steps, model_time
    )
    state_arrays, checked_winds, checked_coordinates, checked_histories = (
        _require_checkpoint_inputs(branch, state, winds, coordinates, open_histories)
    )
    inventory: list[tuple[str, Any]] = [
        ("state.f32", state_arrays[0]),
        ("state.f64", state_arrays[1]),
        ("state.i32", state_arrays[2]),
        *(
            (f"coordinate.{name}", checked_coordinates[name])
            for name in COORDINATE_FIELDS
        ),
        *((f"wind.{name}", checked_winds[name]) for name in periodic.WIND_FIELDS),
    ]
    if checked_histories is not None:
        inventory.extend(
            (f"open_history.{name}", checked_histories[name])
            for name in radiative_open.OPEN_HISTORY_FIELDS
        )
    payload_parts: list[bytes] = []
    entries: list[dict[str, object]] = []
    offset = 0
    for name, value in inventory:
        raw = value.tobytes(order="C")
        entries.append(
            {
                "name": name,
                "dtype": _entry_dtype(value),
                "shape": list(value.shape),
                "offset": offset,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        payload_parts.append(raw)
        offset += len(raw)
    payload = b"".join(payload_parts)
    manifest = {
        "schema": ADAPTIVE_MOVE_RESTART_LIFECYCLE_SCHEMA,
        "abi": ADAPTIVE_MOVE_RESTART_LIFECYCLE_ABI,
        "status": "COMPLETED_STEP_ADAPTIVE_MOVE_CHECKPOINT",
        "branch": branch,
        "completed_steps": completed_steps,
        "model_time_f64_bits": _f64_bits(model_time),
        "vertical_grid_proof_sha256": vertical_grid_proof.proof_sha256,
        "branch_identity": _branch_identity(branch),
        "contract_sha256": adaptive_move_restart_lifecycle_contract()[
            "contract_sha256"
        ],
        "entries": entries,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "source_order": (
            "completed-step domainloc using pre-update speed; due diagnosis; "
            "controller state update; source-ordered wind rebase; cadence increment"
        ),
        "claim_limit": ADAPTIVE_MOVE_RESTART_LIFECYCLE_CLAIM_LIMIT,
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    payload_sha256 = manifest["payload_sha256"]
    artifact_sha256 = hashlib.sha256(
        len(manifest_bytes).to_bytes(8, "little") + manifest_bytes + payload
    ).hexdigest()
    return AdaptiveMoveLifecycleCheckpoint(
        manifest_bytes,
        payload,
        manifest_sha256,
        str(payload_sha256),
        artifact_sha256,
    )


def _expected_entry_specs(
    branch: AdaptiveMoveBranch, ni: int, nj: int, nk: int
) -> list[tuple[str, str, tuple[int, ...]]]:
    specs = [
        ("state.f32", "little-endian-f32", (8,)),
        ("state.f64", "little-endian-f64", (3,)),
        ("state.i32", "little-endian-i32", (4,)),
        ("coordinate.xh", "little-endian-f32", (ni,)),
        ("coordinate.yh", "little-endian-f32", (nj,)),
        ("coordinate.xfref", "little-endian-f32", (ni + 1,)),
        ("coordinate.yfref", "little-endian-f32", (nj + 1,)),
        ("wind.u0", "little-endian-f32", (nk, nj, ni + 1)),
        ("wind.ua", "little-endian-f32", (nk, nj, ni + 1)),
        ("wind.u3d", "little-endian-f32", (nk, nj, ni + 1)),
        ("wind.v0", "little-endian-f32", (nk, nj + 1, ni)),
        ("wind.va", "little-endian-f32", (nk, nj + 1, ni)),
        ("wind.v3d", "little-endian-f32", (nk, nj + 1, ni)),
        ("wind.ug", "little-endian-f32", (nk,)),
        ("wind.vg", "little-endian-f32", (nk,)),
    ]
    if branch == "radiative-open":
        ngxy = radiative_open.DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION.ngxy
        specs.extend(
            (
                f"open_history.{name}",
                "little-endian-f32",
                (nk + 2, nj + 2 * ngxy)
                if name in ("radbcw", "radbce")
                else (nk + 2, ni + 2 * ngxy),
            )
            for name in radiative_open.OPEN_HISTORY_FIELDS
        )
    return specs


def restore_adaptive_move_lifecycle_checkpoint(
    checkpoint: AdaptiveMoveLifecycleCheckpoint,
    *,
    expected_branch: AdaptiveMoveBranch,
    vertical_grid_proof: StretchZ1MetricProof,
) -> RestoredAdaptiveMoveLifecycle:
    """Fail closed and restore an exact adaptive controller/carrier envelope."""

    np = importlib.import_module("numpy")
    expected_branch = _require_branch(expected_branch)
    periodic.validate_adaptive_move_stretch_proof(vertical_grid_proof)
    if not isinstance(checkpoint, AdaptiveMoveLifecycleCheckpoint):
        raise TypeError("checkpoint must be an AdaptiveMoveLifecycleCheckpoint")
    if (
        hashlib.sha256(checkpoint.manifest_bytes).hexdigest()
        != checkpoint.manifest_sha256
    ):
        raise ValueError("adaptive lifecycle manifest hash mismatch")
    if hashlib.sha256(checkpoint.payload).hexdigest() != checkpoint.payload_sha256:
        raise ValueError("adaptive lifecycle payload hash mismatch")
    artifact_sha256 = hashlib.sha256(
        len(checkpoint.manifest_bytes).to_bytes(8, "little")
        + checkpoint.manifest_bytes
        + checkpoint.payload
    ).hexdigest()
    if artifact_sha256 != checkpoint.artifact_sha256:
        raise ValueError("adaptive lifecycle artifact hash mismatch")
    try:
        manifest = json.loads(checkpoint.manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("adaptive lifecycle manifest is not canonical JSON") from error
    if _canonical_json(manifest) != checkpoint.manifest_bytes:
        raise ValueError("adaptive lifecycle manifest is not compact canonical JSON")
    if (
        manifest.get("schema") != ADAPTIVE_MOVE_RESTART_LIFECYCLE_SCHEMA
        or manifest.get("abi") != ADAPTIVE_MOVE_RESTART_LIFECYCLE_ABI
        or manifest.get("status") != "COMPLETED_STEP_ADAPTIVE_MOVE_CHECKPOINT"
    ):
        raise ValueError("adaptive lifecycle checkpoint identity mismatch")
    actual_branch = _require_branch(manifest.get("branch"))
    if actual_branch != expected_branch:
        refuse(
            code="ARWEN-ADAPTIVE-MOVE-RESTART-BRANCH-SUBSTITUTION",
            gate="adaptive-moving-domain-restart-lifecycle",
            option="branch",
            value=actual_branch,
            reason=f"checkpoint branch differs from required {expected_branch}",
            remedy="restore with the exact branch that captured the checkpoint",
        )
    if manifest.get("branch_identity") != _branch_identity(actual_branch):
        raise ValueError("adaptive lifecycle branch/operator identity mismatch")
    if (
        manifest.get("contract_sha256")
        != adaptive_move_restart_lifecycle_contract()["contract_sha256"]
    ):
        raise ValueError("adaptive lifecycle contract hash mismatch")
    if manifest.get("vertical_grid_proof_sha256") != vertical_grid_proof.proof_sha256:
        raise ValueError("adaptive lifecycle vertical-grid proof mismatch")
    completed_steps, model_time = _require_completed_step_clock(
        manifest.get("completed_steps"),
        _f64_from_bits(manifest.get("model_time_f64_bits")),
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) < 15:
        raise ValueError("adaptive lifecycle entry inventory is malformed")
    by_name = {}
    offset = 0
    arrays: dict[str, Any] = {}
    dtype_map = {
        "little-endian-f32": np.dtype("<f4"),
        "little-endian-f64": np.dtype("<f8"),
        "little-endian-i32": np.dtype("<i4"),
    }
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "dtype",
            "shape",
            "offset",
            "bytes",
            "sha256",
        }:
            raise ValueError("adaptive lifecycle entry schema mismatch")
        name, dtype_name, shape = entry["name"], entry["dtype"], entry["shape"]
        if not isinstance(name, str) or name in by_name or dtype_name not in dtype_map:
            raise ValueError(
                "adaptive lifecycle entry name/dtype is invalid or duplicate"
            )
        if entry["offset"] != offset or not isinstance(shape, list) or not shape:
            raise ValueError(
                "adaptive lifecycle entries are not exact contiguous inventory"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in shape
        ):
            raise ValueError("adaptive lifecycle entry shape is invalid")
        expected_bytes = int(np.prod(shape)) * dtype_map[dtype_name].itemsize
        if entry["bytes"] != expected_bytes or offset + expected_bytes > len(
            checkpoint.payload
        ):
            raise ValueError("adaptive lifecycle entry byte extent mismatch")
        raw = checkpoint.payload[offset : offset + expected_bytes]
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"adaptive lifecycle entry hash mismatch: {name}")
        arrays[name] = (
            np.frombuffer(raw, dtype=dtype_map[dtype_name]).copy().reshape(shape)
        )
        by_name[name] = entry
        offset += expected_bytes
    if offset != len(checkpoint.payload):
        raise ValueError("adaptive lifecycle entries do not consume payload exactly")
    if (
        manifest.get("payload_bytes") != offset
        or manifest.get("payload_sha256") != checkpoint.payload_sha256
    ):
        raise ValueError("adaptive lifecycle payload seal mismatch")
    try:
        nk, nj, ni1 = arrays["wind.u0"].shape
        _, nj1, ni = arrays["wind.v0"].shape
    except (KeyError, ValueError) as error:
        raise ValueError(
            "adaptive lifecycle wind inventory lacks valid stagger"
        ) from error
    if ni1 != ni + 1 or nj1 != nj + 1:
        raise ValueError("adaptive lifecycle wind stagger is inconsistent")
    expected_specs = _expected_entry_specs(actual_branch, ni, nj, nk)
    observed_specs = [
        (entry["name"], entry["dtype"], tuple(entry["shape"])) for entry in entries
    ]
    if observed_specs != expected_specs:
        raise ValueError("adaptive lifecycle entry inventory/order differs from ABI")
    state = periodic.unpack_adaptive_move_state(
        arrays["state.f32"], arrays["state.f64"], arrays["state.i32"]
    )
    winds = {name: arrays[f"wind.{name}"] for name in periodic.WIND_FIELDS}
    coordinates = {name: arrays[f"coordinate.{name}"] for name in COORDINATE_FIELDS}
    histories = (
        {
            name: arrays[f"open_history.{name}"]
            for name in radiative_open.OPEN_HISTORY_FIELDS
        }
        if actual_branch == "radiative-open"
        else None
    )
    _require_checkpoint_inputs(actual_branch, state, winds, coordinates, histories)
    return RestoredAdaptiveMoveLifecycle(
        actual_branch,
        completed_steps,
        model_time,
        state,
        MappingProxyType(winds),
        MappingProxyType(coordinates),
        MappingProxyType(histories) if histories is not None else None,
        MappingProxyType(
            {
                "manifest_sha256": checkpoint.manifest_sha256,
                "payload_sha256": checkpoint.payload_sha256,
                "artifact_sha256": checkpoint.artifact_sha256,
            }
        ),
    )


def cpu_advance_adaptive_move_lifecycle_step(
    branch: AdaptiveMoveBranch,
    state: periodic.AdaptiveMoveState,
    winds: Mapping[str, Any],
    *,
    nstep: int,
    mtime_after: float,
    dbldt: float,
    adapt_move_frq: float,
    centerx: float,
    centery: float,
    vertical_grid_proof: StretchZ1MetricProof,
    open_histories: Mapping[str, Any] | None = None,
    **cadence_and_diagnosis: Any,
) -> periodic.AdaptiveMoveTransition:
    """Dispatch one completed large step without permitting branch substitution."""

    branch = _require_branch(branch)
    if branch == "periodic":
        if open_histories is not None:
            refuse(
                code="ARWEN-ADAPTIVE-MOVE-RESTART-BRANCH-SUBSTITUTION",
                gate="adaptive-moving-domain-restart-lifecycle",
                option="open_histories",
                value="supplied-for-periodic",
                reason="periodic lifecycle step cannot consume radiative histories",
                remedy="remove histories or select radiative-open",
            )
        return periodic.cpu_advance_adaptive_move_after_large_step(
            state,
            winds,
            nstep=nstep,
            mtime_after=mtime_after,
            dbldt=dbldt,
            adapt_move_frq=adapt_move_frq,
            centerx=centerx,
            centery=centery,
            vertical_grid_proof=vertical_grid_proof,
            **cadence_and_diagnosis,
        )
    if open_histories is None:
        raise ValueError("radiative-open lifecycle step requires radbc histories")
    return radiative_open.cpu_advance_adaptive_move_open_after_large_step(
        state,
        winds,
        open_histories,
        nstep=nstep,
        mtime_after=mtime_after,
        dbldt=dbldt,
        adapt_move_frq=adapt_move_frq,
        centerx=centerx,
        centery=centery,
        vertical_grid_proof=vertical_grid_proof,
        **cadence_and_diagnosis,
    )


def launch_cuda_adaptive_move_lifecycle_step(
    branch: AdaptiveMoveBranch,
    state: periodic.DeviceAdaptiveMoveState,
    winds: Mapping[str, Any],
    *,
    nstep: int,
    mtime_after: float,
    dbldt: float,
    adapt_move_frq: float,
    centerx: float,
    centery: float,
    vertical_grid_proof: StretchZ1MetricProof,
    open_histories: Mapping[str, Any] | None = None,
    **launch_options: Any,
) -> Any:
    """Enqueue one exact branch lifecycle step on the caller's CUDA stream."""

    branch = _require_branch(branch)
    if branch == "periodic":
        if open_histories is not None:
            refuse(
                code="ARWEN-ADAPTIVE-MOVE-RESTART-BRANCH-SUBSTITUTION",
                gate="adaptive-moving-domain-restart-lifecycle",
                option="open_histories",
                value="supplied-for-periodic",
                reason="periodic CUDA lifecycle cannot consume radiative histories",
                remedy="remove histories or select radiative-open",
            )
        return periodic.launch_cuda_adaptive_move_after_large_step(
            state,
            winds,
            nstep=nstep,
            mtime_after=mtime_after,
            dbldt=dbldt,
            adapt_move_frq=adapt_move_frq,
            centerx=centerx,
            centery=centery,
            vertical_grid_proof=vertical_grid_proof,
            **launch_options,
        )
    if open_histories is None:
        raise ValueError("radiative-open CUDA lifecycle requires radbc histories")
    return radiative_open.launch_cuda_adaptive_move_open_after_large_step(
        state,
        winds,
        open_histories,
        nstep=nstep,
        mtime_after=mtime_after,
        dbldt=dbldt,
        adapt_move_frq=adapt_move_frq,
        centerx=centerx,
        centery=centery,
        vertical_grid_proof=vertical_grid_proof,
        **launch_options,
    )


def validate_cuda_adaptive_move_lifecycle_completion(
    branch: AdaptiveMoveBranch, launch: Any
) -> None:
    branch = _require_branch(branch)
    if branch == "periodic":
        periodic.validate_cuda_adaptive_move_completion(launch)
    else:
        radiative_open.validate_cuda_adaptive_move_open_completion(launch)


__all__ = (
    "ADAPTIVE_MOVE_RESTART_LIFECYCLE_ABI",
    "ADAPTIVE_MOVE_RESTART_LIFECYCLE_CLAIM_LIMIT",
    "ADAPTIVE_MOVE_RESTART_LIFECYCLE_SCHEMA",
    "AdaptiveMoveBranch",
    "AdaptiveMoveLifecycleCheckpoint",
    "BRANCHES",
    "COORDINATE_FIELDS",
    "OPEN_OPERATOR_COMMIT",
    "OPEN_OPERATOR_SOURCE_SHA256",
    "PERIODIC_OPERATOR_COMMIT",
    "PERIODIC_OPERATOR_SOURCE_SHA256",
    "RestoredAdaptiveMoveLifecycle",
    "adaptive_move_restart_lifecycle_contract",
    "capture_adaptive_move_lifecycle_checkpoint",
    "cpu_advance_adaptive_move_lifecycle_step",
    "launch_cuda_adaptive_move_lifecycle_step",
    "restore_adaptive_move_lifecycle_checkpoint",
    "validate_cuda_adaptive_move_lifecycle_completion",
)
