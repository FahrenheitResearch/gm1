"""Deterministic structural zero-step envelope.

The wire format is intentionally small and language-neutral so the future Rust
CM1 writer can consume the same plan without inheriting Python objects:

``CM1SZS1\0`` | ``uint64_le(header_bytes)`` | canonical UTF-8 JSON | payloads

Payloads are little-endian float32 words in the plan's fixed field order.  The
JSON header binds every payload offset, length, SHA-256, staggering, native
bounds, and confirmed-interior slice.  Its claim boundary remains structural:
the bytes are not a scientific initial condition or a history/restart product.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from math import isfinite
from typing import Any

from . import CM1_AUTHORITY_SHA256
from .config import CM1Config, Lane
from .layout import (
    GRID_COORDINATES,
    STRUCTURAL_DYCORE_LAYOUTS,
    TARGET_HALO,
    GridShape,
    Halo,
    Stagger,
)
from .state import (
    DEFAULT_STRUCTURAL_ALLOCATION_LIMIT,
    DEFERRED_STATE_FAMILIES,
    STRUCTURAL_CLAIM_LIMIT,
    STRUCTURAL_STATE_PLAN_SCHEMA,
    STRUCTURAL_STATE_SCHEMA,
    PlannedField,
    StructuralState,
    build_structural_state_plan,
    initialize_structural_state,
    structural_profile_geometry,
    validate_structural_field_payload,
)


STRUCTURAL_ZERO_STEP_SCHEMA = "gpuwm-cm1.structural-zero-step/v1"
STRUCTURAL_ZERO_STEP_MAGIC = b"CM1SZS1\0"
_HEADER_LENGTH = struct.Struct("<Q")
MAX_STRUCTURAL_ZERO_STEP_HEADER_BYTES = 64 * 1024
MAX_STRUCTURAL_ZERO_STEP_ENVELOPE_BYTES = (
    len(STRUCTURAL_ZERO_STEP_MAGIC)
    + _HEADER_LENGTH.size
    + MAX_STRUCTURAL_ZERO_STEP_HEADER_BYTES
    + DEFAULT_STRUCTURAL_ALLOCATION_LIMIT
)
_REQUIRED_HEADER_KEYS = {
    "claim_limit",
    "payload_byte_order",
    "payload_order",
    "payloads",
    "plan",
    "plan_sha256",
    "schema",
    "state_schema",
    "total_payload_bytes",
}
_REQUIRED_PLAN_KEYS = {
    "authority_sha256",
    "claim_limit",
    "config_sha256",
    "deferred_state_families",
    "fields",
    "grid",
    "lane",
    "schema",
    "status",
    "total_nbytes",
}
_REQUIRED_GRID_KEYS = {
    "axis_order",
    "coordinates",
    "dimensions",
    "fortran_bounds",
    "halo",
    "spacing_m",
    "stagger_geometry",
}


@dataclass(frozen=True, slots=True)
class ParsedFieldPayload:
    name: str
    data: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class ParsedStructuralZeroStep:
    header: dict[str, Any]
    payloads: tuple[ParsedFieldPayload, ...]
    artifact_sha256: str

    def payload(self, name: str) -> ParsedFieldPayload:
        for row in self.payloads:
            if row.name == name:
                return row
        raise KeyError(name)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _header(state: StructuralState) -> dict[str, object]:
    offset = 0
    payloads: list[dict[str, object]] = []
    for buffer in state.buffers:
        payloads.append(
            {
                "name": buffer.field.layout.native_name,
                "nbytes": len(buffer.data),
                "offset": offset,
                "sha256": buffer.sha256,
            }
        )
        offset += len(buffer.data)
    return {
        "claim_limit": STRUCTURAL_CLAIM_LIMIT,
        "payload_byte_order": "little",
        "payload_order": "plan-field-order",
        "payloads": payloads,
        "plan": state.plan.as_dict(),
        "plan_sha256": state.plan.sha256,
        "schema": STRUCTURAL_ZERO_STEP_SCHEMA,
        "state_schema": state.schema,
        "total_payload_bytes": offset,
    }


def serialize_structural_zero_step(state: StructuralState) -> bytes:
    """Serialize and hash-bind a structural state without filesystem I/O."""

    header = _canonical_json(_header(state))
    payload = b"".join(buffer.data for buffer in state.buffers)
    if len(payload) != state.total_nbytes:
        raise AssertionError("structural payload length changed during serialization")
    if len(header) > MAX_STRUCTURAL_ZERO_STEP_HEADER_BYTES:
        raise ValueError("structural zero-step JSON header exceeds the v1 size limit")
    envelope = (
        STRUCTURAL_ZERO_STEP_MAGIC
        + _HEADER_LENGTH.pack(len(header))
        + header
        + payload
    )
    if len(envelope) > MAX_STRUCTURAL_ZERO_STEP_ENVELOPE_BYTES:
        raise ValueError("structural zero-step envelope exceeds the v1 size limit")
    return envelope


def build_structural_zero_step(
    config: CM1Config,
    *,
    max_bytes: int = DEFAULT_STRUCTURAL_ALLOCATION_LIMIT,
) -> bytes:
    """Build the plan, materialize sentinels, and return the v1 envelope."""

    plan = build_structural_state_plan(config)
    state = initialize_structural_state(plan, max_bytes=max_bytes)
    return serialize_structural_zero_step(state)


def structural_zero_step_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> "NoReturn":
    from typing import NoReturn

    raise ValueError(f"non-finite JSON token {value!r} is forbidden")


def _required_integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _require_exact_keys(
    mapping: dict[str, Any], expected: set[str], label: str
) -> None:
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"{label} keys differ: missing={missing}, extra={extra}")


def _required_sha256(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{key} must be a lower-case SHA-256 hex digest")
    return value


def _validate_plan(
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], tuple[PlannedField, ...]]:
    """Validate every structural v1 semantic, not merely the plan digest."""

    _require_exact_keys(plan, _REQUIRED_PLAN_KEYS, "structural zero-step plan")
    if plan["schema"] != STRUCTURAL_STATE_PLAN_SCHEMA:
        raise ValueError("structural zero-step state-plan schema mismatch")
    if plan["claim_limit"] != STRUCTURAL_CLAIM_LIMIT:
        raise ValueError("structural zero-step plan claim boundary mismatch")
    if plan["status"] != "structural-allocation-contract-only":
        raise ValueError("structural zero-step plan status mismatch")
    if _required_sha256(plan, "authority_sha256") != CM1_AUTHORITY_SHA256:
        raise ValueError("structural zero-step CM1 authority mismatch")
    _required_sha256(plan, "config_sha256")
    if plan["lane"] not in {member.value for member in Lane}:
        raise ValueError("structural zero-step lane is not sealed")
    expected_shape, expected_spacing = structural_profile_geometry(plan["lane"])

    grid_row = plan["grid"]
    if not isinstance(grid_row, dict):
        raise ValueError("structural zero-step grid must be a JSON object")
    _require_exact_keys(grid_row, _REQUIRED_GRID_KEYS, "structural zero-step grid")
    dimensions = grid_row["dimensions"]
    if not isinstance(dimensions, dict) or set(dimensions) != {"nx", "ny", "nz"}:
        raise ValueError("structural zero-step dimensions must contain nx, ny, and nz")
    dimension_values = {
        key: _required_integer(dimensions, key) for key in ("nx", "ny", "nz")
    }
    if min(dimension_values.values()) < 1:
        raise ValueError("structural zero-step dimensions must be positive")
    grid = GridShape(**dimension_values)
    if (grid.nx, grid.ny, grid.nz) != expected_shape:
        raise ValueError("structural zero-step dimensions differ from the lane profile")

    halo_row = grid_row["halo"]
    if not isinstance(halo_row, dict) or set(halo_row) != {"x", "y", "z"}:
        raise ValueError("structural zero-step halo must contain x, y, and z")
    halo = Halo(**{key: _required_integer(halo_row, key) for key in ("x", "y", "z")})
    if halo != TARGET_HALO:
        raise ValueError("structural zero-step v1 target halo mismatch")
    spacing = grid_row["spacing_m"]
    if (
        not isinstance(spacing, list)
        or len(spacing) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0.0
            for value in spacing
        )
    ):
        raise ValueError("structural zero-step spacing_m must contain three positive numbers")
    if tuple(spacing) != expected_spacing:
        raise ValueError("structural zero-step spacing differs from the lane profile")
    if grid_row["axis_order"] != "zyx" or grid_row["fortran_bounds"] != "inclusive":
        raise ValueError("structural zero-step axis/bounds convention mismatch")

    expected_coordinates = []
    for coordinate in GRID_COORDINATES:
        bounds = coordinate.native_bounds(grid, halo)
        expected_coordinates.append(
            {
                "axis": coordinate.axis.value,
                "count": bounds.count,
                "location": coordinate.location.value,
                "name": coordinate.native_name,
                "native_bounds": bounds.as_dict(),
                "status": "allocation-only-values-deferred-to-O-GRID",
                "units": coordinate.units,
            }
        )
    if grid_row["coordinates"] != expected_coordinates:
        raise ValueError("structural zero-step coordinate allocation metadata mismatch")
    expected_staggers = {
        stagger.value: grid.geometry(stagger, halo).as_dict() for stagger in Stagger
    }
    if grid_row["stagger_geometry"] != expected_staggers:
        raise ValueError("structural zero-step stagger geometry mismatch")

    plan_fields = plan["fields"]
    if not isinstance(plan_fields, list) or any(
        not isinstance(row, dict) for row in plan_fields
    ):
        raise ValueError("structural zero-step plan fields must be objects")
    planned_fields = tuple(
        PlannedField(layout=row, geometry=row.geometry(grid, halo))
        for row in STRUCTURAL_DYCORE_LAYOUTS
    )
    expected_fields = [field.as_dict() for field in planned_fields]
    if plan_fields != expected_fields:
        raise ValueError("structural zero-step field metadata/geometry mismatch")
    expected_deferred = [row.as_dict() for row in DEFERRED_STATE_FAMILIES]
    if plan["deferred_state_families"] != expected_deferred:
        raise ValueError("structural zero-step deferred-state gate inventory mismatch")
    total_nbytes = sum(int(row["nbytes"]) for row in expected_fields)
    if _required_integer(plan, "total_nbytes") != total_nbytes:
        raise ValueError("structural zero-step plan total byte count mismatch")
    return plan_fields, planned_fields


def parse_structural_zero_step(blob: bytes) -> ParsedStructuralZeroStep:
    """Strictly validate a v1 envelope and all payload hashes."""

    if not isinstance(blob, bytes):
        raise TypeError("structural zero-step input must be immutable bytes")
    if len(blob) > MAX_STRUCTURAL_ZERO_STEP_ENVELOPE_BYTES:
        raise ValueError("structural zero-step envelope exceeds the v1 size limit")
    prefix_bytes = len(STRUCTURAL_ZERO_STEP_MAGIC) + _HEADER_LENGTH.size
    if len(blob) < prefix_bytes:
        raise ValueError("structural zero-step envelope is truncated")
    if blob[: len(STRUCTURAL_ZERO_STEP_MAGIC)] != STRUCTURAL_ZERO_STEP_MAGIC:
        raise ValueError("structural zero-step magic/version mismatch")
    header_size = _HEADER_LENGTH.unpack_from(blob, len(STRUCTURAL_ZERO_STEP_MAGIC))[0]
    if header_size > MAX_STRUCTURAL_ZERO_STEP_HEADER_BYTES:
        raise ValueError("structural zero-step JSON header exceeds the v1 size limit")
    header_start = prefix_bytes
    header_end = header_start + header_size
    if header_end > len(blob):
        raise ValueError("structural zero-step JSON header is truncated")
    raw_header = blob[header_start:header_end]
    try:
        decoded = json.loads(
            raw_header.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"invalid structural zero-step JSON header: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError("structural zero-step header must be a JSON object")
    if raw_header != _canonical_json(decoded):
        raise ValueError("structural zero-step JSON header is not canonical")
    header: dict[str, Any] = decoded
    _require_exact_keys(header, _REQUIRED_HEADER_KEYS, "structural zero-step header")
    if header["schema"] != STRUCTURAL_ZERO_STEP_SCHEMA:
        raise ValueError("structural zero-step schema mismatch")
    if header["claim_limit"] != STRUCTURAL_CLAIM_LIMIT:
        raise ValueError("structural zero-step claim boundary mismatch")
    if header["payload_byte_order"] != "little":
        raise ValueError("structural zero-step byte order mismatch")
    if header["payload_order"] != "plan-field-order":
        raise ValueError("structural zero-step payload order mismatch")
    if header["state_schema"] != STRUCTURAL_STATE_SCHEMA:
        raise ValueError("structural zero-step state schema mismatch")
    plan = header["plan"]
    if not isinstance(plan, dict):
        raise ValueError("structural zero-step plan must be a JSON object")
    plan_hash = hashlib.sha256(_canonical_json(plan)).hexdigest()
    if _required_sha256(header, "plan_sha256") != plan_hash:
        raise ValueError("structural zero-step plan hash mismatch")
    plan_fields, planned_fields = _validate_plan(plan)
    planned_names = [row.get("name") for row in plan_fields]
    if any(not isinstance(name, str) or not name for name in planned_names):
        raise ValueError("structural zero-step plan field names are invalid")
    if len(planned_names) != len(set(planned_names)):
        raise ValueError("structural zero-step plan field names are duplicated")
    manifest = header["payloads"]
    if not isinstance(manifest, list) or any(not isinstance(row, dict) for row in manifest):
        raise ValueError("structural zero-step payload manifest must contain objects")
    manifest_names = [row.get("name") for row in manifest]
    if manifest_names != planned_names:
        raise ValueError("payload manifest order does not match the plan")
    payload_bytes = blob[header_end:]
    if _required_integer(header, "total_payload_bytes") != len(payload_bytes):
        raise ValueError("structural zero-step total payload size mismatch")
    parsed: list[ParsedFieldPayload] = []
    expected_offset = 0
    for plan_row, planned_field, manifest_row in zip(
        plan_fields, planned_fields, manifest
    ):
        if set(manifest_row) != {"name", "nbytes", "offset", "sha256"}:
            raise ValueError("structural zero-step payload manifest keys differ")
        offset = _required_integer(manifest_row, "offset")
        nbytes = _required_integer(manifest_row, "nbytes")
        if offset != expected_offset:
            raise ValueError("structural zero-step payload offsets are not contiguous")
        if plan_row.get("nbytes") != nbytes:
            raise ValueError("structural zero-step plan/payload byte counts disagree")
        end = offset + nbytes
        if end > len(payload_bytes):
            raise ValueError("structural zero-step field payload is truncated")
        data = payload_bytes[offset:end]
        digest = hashlib.sha256(data).hexdigest()
        if _required_sha256(manifest_row, "sha256") != digest:
            raise ValueError(f"structural zero-step payload hash mismatch for {manifest_row['name']}")
        validate_structural_field_payload(planned_field, data)
        parsed.append(
            ParsedFieldPayload(
                name=str(manifest_row["name"]),
                data=data,
                sha256=digest,
            )
        )
        expected_offset = end
    if expected_offset != len(payload_bytes):
        raise ValueError("structural zero-step envelope has trailing payload bytes")
    return ParsedStructuralZeroStep(
        header=header,
        payloads=tuple(parsed),
        artifact_sha256=structural_zero_step_sha256(blob),
    )


__all__ = [
    "MAX_STRUCTURAL_ZERO_STEP_ENVELOPE_BYTES",
    "MAX_STRUCTURAL_ZERO_STEP_HEADER_BYTES",
    "ParsedFieldPayload",
    "ParsedStructuralZeroStep",
    "STRUCTURAL_ZERO_STEP_MAGIC",
    "STRUCTURAL_ZERO_STEP_SCHEMA",
    "build_structural_zero_step",
    "parse_structural_zero_step",
    "serialize_structural_zero_step",
    "structural_zero_step_sha256",
]
