from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/gm1-real-5m-child-auxiliary-stream-v2"
VALIDATION_SHA256 = (
    "6c7852db9d55bb96e844b577312b269f2e84cdbb206669051349afb43f7442d7"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt() -> dict[str, object]:
    path = EVIDENCE / "validation-receipt.json"
    assert _sha256(path) == VALIDATION_SHA256
    return json.loads(path.read_bytes())


def test_real_5m_child_auxiliary_external_artifact_is_exact_and_source_bound() -> None:
    receipt = _receipt()
    assert receipt["status"] == (
        "PASS_REAL_SOURCE_COORDINATE_3000X3000_5M_CHILD_AUXILIARY_STREAM_ROUNDTRIP"
    )
    artifact = receipt["artifact"]
    assert artifact["directory"] == (
        "F:\\gm1-artifacts\\work\\gm1-real-5m-child-auxiliary-v2"
    )
    assert artifact["external_payload_committed_to_git"] is False
    assert artifact["inventory_count"] == 6
    assert artifact["total_payload_nbytes"] == 111_389_817_536
    assert artifact["files"] == {
        "base-wind.f32le": {
            "nbytes": 5_401_800_000,
            "sha256": (
                "224c82d242f202a186ae65c0726a368b3cb231d8ab8b1d6891b779b4d1db04e6"
            ),
        },
        "live-density.f32le": {
            "nbytes": 10_836_000_000,
            "sha256": (
                "e3f0e32677828420bbf401d3653eeffe7a77ad313f258ebf7cb1038cf86305a5"
            ),
        },
        "manifest.json": {
            "nbytes": 14_928,
            "sha256": (
                "282b1ddbc09b3fe70b45c38ba60ce9ec96dc30c5cee2420a645247515a723002"
            ),
        },
        "receipt.json": {
            "nbytes": 754,
            "sha256": (
                "f0762b758a997f4443ad311c66ade1e5b43f3d51353b1a45d9fa4553e24009e9"
            ),
        },
        "terrain-open.f32le": {
            "nbytes": 92_416_017_536,
            "sha256": (
                "66f8d27901e2468e3a998f247c9bb5608f61f3d002435b713440b7773177701e"
            ),
        },
        "total-geopotential-zf.f32le": {
            "nbytes": 2_736_000_000,
            "sha256": (
                "89b214becc8166008c87c1dea1c0a99b4f89ef4853423ab427e107c6fc0d0074"
            ),
        },
    }


def test_real_5m_child_auxiliary_streaming_and_capacity_are_frozen() -> None:
    receipt = _receipt()
    assert receipt["capacity"] == {
        "build_time_free_bytes": 905_474_998_272,
        "planned_payload_copies": 1,
        "post_publish_free_bytes": 794_085_180_736,
        "reserve_bytes": 8_589_934_592,
        "status": "PASS_SINGLE_COPY_STREAMED_CAPACITY",
    }
    invariants = receipt["invariants"]
    assert all(invariants.values())
    assert receipt["tests"]["build_action"] == "built"
    assert receipt["tests"]["independent_action"] == "inspected"
    assert receipt["tests"]["independent_every_byte_inspect"] is True
    assert receipt["tests"]["stream_chunk_bytes"] == 1 << 20


def test_real_5m_child_auxiliary_producer_and_50m_parity_are_frozen() -> None:
    receipt = _receipt()
    producer = receipt["producer"]
    assert producer["source_sha256"] == _sha256(
        ROOT / "rust/cm1-state/src/child_auxiliary_stream.rs"
    )
    assert producer["cli_source_sha256"] == _sha256(
        ROOT / "rust/cm1-state/src/bin/gm1_child_auxiliary.rs"
    )
    assert producer["executable_sha256"] == (
        "868eb1f5662dabf90258a20e6ac0dd3360b0df33bf450e654e3a0aea256a9fca"
    )
    regression = receipt["historical_300x300_50m_regression"]
    assert regression["raw_bit_equal_to_accepted_historical_payloads"] is True
    assert regression["independent_every_byte_inspect"] is True
    assert regression["manifest_sha256"] == (
        "622e7df0ceda6c3645bce3b15f782e9163a371c66e2bbd4cb11c659883bcb4b2"
    )
    assert regression["receipt_sha256"] == (
        "2c6483332e330fc6e1d6e3b328343d8fe836e9ba77f3066d4a84dee706ba1706"
    )
