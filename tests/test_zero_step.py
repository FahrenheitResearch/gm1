from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

import cm1core.zero_step as zero_step
from cm1core.config import load_config
from cm1core.state import STRUCTURAL_STATE_PLAN_SCHEMA, STRUCTURAL_STATE_SCHEMA
from cm1core.zero_step import (
    STRUCTURAL_ZERO_STEP_MAGIC,
    STRUCTURAL_ZERO_STEP_SCHEMA,
    build_structural_zero_step,
    parse_structural_zero_step,
    structural_zero_step_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DRY = ROOT / "examples" / "dry-psolver2-bootstrap.toml"


def _split(payload: bytes) -> tuple[dict[str, object], bytes]:
    header_start = len(STRUCTURAL_ZERO_STEP_MAGIC) + 8
    header_size = struct.unpack_from("<Q", payload, len(STRUCTURAL_ZERO_STEP_MAGIC))[0]
    header_end = header_start + header_size
    return json.loads(payload[header_start:header_end]), payload[header_end:]


def _reframe(header: dict[str, object], field_payloads: bytes) -> bytes:
    encoded = json.dumps(
        header, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return STRUCTURAL_ZERO_STEP_MAGIC + struct.pack("<Q", len(encoded)) + encoded + field_payloads


def test_structural_zero_step_is_byte_deterministic_and_hash_pinned() -> None:
    config = load_config(DRY)
    first = build_structural_zero_step(config)
    second = build_structural_zero_step(config)

    assert first == second
    assert first.startswith(STRUCTURAL_ZERO_STEP_MAGIC)
    # The pin covers the header's plan as it names its deferred oracle gates (O-GRID, O-ARWEN-WSM6,
    # O-CLOSURES, O-INSTRUMENT) and the dry bootstrap's configuration identity with its profile's
    # description; every field payload is the bytes the earlier pin sealed.
    assert structural_zero_step_sha256(first) == (
        "1482fada63702183932283b2f7bd3fbb85df3b9e5c6d7e85588fcf954bf6ede4"
    )


def test_structural_zero_step_round_trip_binds_plan_and_payloads() -> None:
    parsed = parse_structural_zero_step(build_structural_zero_step(load_config(DRY)))

    assert parsed.header["schema"] == STRUCTURAL_ZERO_STEP_SCHEMA
    assert parsed.header["state_schema"] == STRUCTURAL_STATE_SCHEMA
    assert parsed.header["plan"]["schema"] == STRUCTURAL_STATE_PLAN_SCHEMA
    assert parsed.header["plan"]["status"] == "structural-allocation-contract-only"
    assert "not CM1 base-state" in parsed.header["claim_limit"]
    assert [row.name for row in parsed.payloads] == [
        "ua",
        "u3d",
        "va",
        "v3d",
        "wa",
        "w3d",
        "ppi",
        "pp3d",
        "tha",
        "th3d",
    ]
    assert len(parsed.payload("ua").data) == 36_432
    assert parsed.header["total_payload_bytes"] == 358_688


def test_structural_zero_step_detects_payload_corruption() -> None:
    payload = bytearray(build_structural_zero_step(load_config(DRY)))
    payload[-1] ^= 1
    with pytest.raises(ValueError, match="payload hash mismatch"):
        parse_structural_zero_step(bytes(payload))


def test_structural_zero_step_rejects_wrong_magic() -> None:
    payload = bytearray(build_structural_zero_step(load_config(DRY)))
    payload[0] ^= 1
    with pytest.raises(ValueError, match="magic/version mismatch"):
        parse_structural_zero_step(bytes(payload))


def test_structural_zero_step_rejects_wrong_state_schema() -> None:
    header, field_payloads = _split(build_structural_zero_step(load_config(DRY)))
    header["state_schema"] = "gpuwm-cm1.structural-state/v999"

    with pytest.raises(ValueError, match="state schema mismatch"):
        parse_structural_zero_step(_reframe(header, field_payloads))


def test_structural_zero_step_rejects_rehashed_semantic_plan_drift() -> None:
    header, field_payloads = _split(build_structural_zero_step(load_config(DRY)))
    plan = header["plan"]
    assert isinstance(plan, dict)
    fields = plan["fields"]
    assert isinstance(fields, list) and isinstance(fields[0], dict)
    fields[0]["live_region_policy"] = "scalar-current-3d"
    canonical_plan = json.dumps(
        plan, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    header["plan_sha256"] = hashlib.sha256(canonical_plan).hexdigest()

    with pytest.raises(ValueError, match="field metadata/geometry mismatch"):
        parse_structural_zero_step(_reframe(header, field_payloads))


def test_structural_zero_step_rejects_noncanonical_json() -> None:
    payload = build_structural_zero_step(load_config(DRY))
    header_start = len(STRUCTURAL_ZERO_STEP_MAGIC) + 8
    header_size = struct.unpack_from("<Q", payload, len(STRUCTURAL_ZERO_STEP_MAGIC))[0]
    header_end = header_start + header_size
    noncanonical = b" " + payload[header_start:header_end]
    reframed = (
        STRUCTURAL_ZERO_STEP_MAGIC
        + struct.pack("<Q", len(noncanonical))
        + noncanonical
        + payload[header_end:]
    )

    with pytest.raises(ValueError, match="JSON header is not canonical"):
        parse_structural_zero_step(reframed)


def test_structural_zero_step_rejects_oversized_header_before_decode() -> None:
    oversized = zero_step.MAX_STRUCTURAL_ZERO_STEP_HEADER_BYTES + 1
    payload = STRUCTURAL_ZERO_STEP_MAGIC + struct.pack("<Q", oversized)

    with pytest.raises(ValueError, match="JSON header exceeds the v1 size limit"):
        parse_structural_zero_step(payload)


def test_structural_zero_step_rejects_oversized_envelope_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(zero_step, "MAX_STRUCTURAL_ZERO_STEP_ENVELOPE_BYTES", 15)
    payload = STRUCTURAL_ZERO_STEP_MAGIC + struct.pack("<Q", 0)

    with pytest.raises(ValueError, match="envelope exceeds the v1 size limit"):
        parse_structural_zero_step(payload)


def test_structural_zero_step_rejects_rehashed_non_sentinel_payload() -> None:
    header, field_payloads = _split(build_structural_zero_step(load_config(DRY)))
    tampered = bytearray(field_payloads)
    tampered[0] ^= 1
    manifest = header["payloads"]
    assert isinstance(manifest, list) and isinstance(manifest[0], dict)
    manifest[0]["sha256"] = hashlib.sha256(tampered[:36_432]).hexdigest()

    with pytest.raises(ValueError, match="violates the structural sentinel policy"):
        parse_structural_zero_step(_reframe(header, bytes(tampered)))
