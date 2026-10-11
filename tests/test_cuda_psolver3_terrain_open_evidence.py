from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cm1core.cuda_initialized_product_terrain_open import (
    CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256,
    PREPARATION_COMPARE_NAMES,
)
from cm1core.cuda_psolver3_terrain_open import (
    COMPARE_OUTPUT_NAMES,
    CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256,
)
from measured_source import measured_identity
from cm1core.cuda_terrain_open_boundary import (
    CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/cuda-psolver3-terrain-open-v1"
SYNTHETIC = EVIDENCE / "synthetic-sm86.json"
REAL_CROP = EVIDENCE / "corrected-real-crop-sm86.json"
GUARDED_RUN = EVIDENCE / "guarded-run-receipt.json"
INVENTORY = EVIDENCE / "artifact-inventory.json"

EXPECTED_RECEIPT_HASHES = {
    "corrected-real-crop-sm86.json": (
        11_108,
        "e47594844f67851e339699caab71b3b40defafdb5c1b7d392719c8dd0f3bcf06",
    ),
    "synthetic-sm86.json": (
        39_844,
        "056eb66e0afffffe0f7490fbf2b1b71bf9a35d5c79758b96e8bbeb42d7a14ba9",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    assert b"\r" not in payload
    return json.loads(payload)


def _assert_raw_exact(comparison: dict[str, object], names: set[str]) -> None:
    assert comparison["passed"] is True
    assert comparison["mode"] == "raw-bit"
    assert comparison["max_ulp"] == 0
    assert comparison["null_or_missing_is_pass"] is False
    fields = comparison["fields"]
    assert set(fields) == names
    for field in fields.values():
        assert field["exact"] is True
        assert field["expected_sha256"] == field["candidate_sha256"]


def test_guarded_receipt_bytes_and_raw_exact_contract_are_frozen() -> None:
    for name, (nbytes, digest) in EXPECTED_RECEIPT_HASHES.items():
        path = EVIDENCE / name
        assert path.stat().st_size == nbytes
        assert _sha256(path) == digest

    synthetic = _load(SYNTHETIC)
    assert synthetic["schema"] == (
        "gpuwm-cm1.cuda-psolver3-terrain-open-synthetic/v1"
    )
    assert synthetic["status"] == "passed-10-of-10-raw-bit-exact"
    assert set(synthetic["cases"]) == {
        "advection",
        "all",
        "core",
        "divergence",
        "family_fills",
        "implicit",
        "open",
        "pgrad",
        "pressure_fill",
        "radiation",
    }
    for comparison in synthetic["cases"].values():
        _assert_raw_exact(comparison, set(COMPARE_OUTPUT_NAMES))
    assert synthetic["kernel"] == {
        "source_sha256": CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256,
        "target_arch": "86",
    }


def test_corrected_real_crop_receipt_has_exact_preparation_and_acoustics() -> None:
    receipt = _load(REAL_CROP)
    assert receipt["schema"] == (
        "gpuwm-cm1.cuda-psolver3-terrain-open-real-crop/v1"
    )
    assert receipt["status"] == "passed-preparation-and-acoustic-raw-bit-exact"
    assert receipt["domain"] == {"ngxy": 3, "ni": 16, "nj": 16, "nk": 75}
    _assert_raw_exact(receipt["preparation_comparison"], set(PREPARATION_COMPARE_NAMES))
    _assert_raw_exact(receipt["acoustic_comparison"], set(COMPARE_OUTPUT_NAMES))

    cadence = receipt["cadence"]
    assert [stage["nloop"] for stage in cadence] == [2, 3, 6]
    assert [stage["open_consumptions"] for stage in cadence] == [2, 3, 6]
    assert [stage["pressure_fills"] for stage in cadence] == [1, 2, 5]
    assert [stage["radiation_productions"] for stage in cadence] == [1, 1, 1]
    assert [stage["dts_f32"] for stage in cadence] == [
        0.10000000149011612,
        0.10000000149011612,
        0.10000000149011612,
    ]

    invariants = receipt["invariants"]
    assert invariants["all_outputs_finite"] is True
    assert invariants["initial_radiation_histories_exact_zero"] is True
    assert invariants["final_radiation_nonzero_words"] == {
        "radbce": 1200,
        "radbcn": 1200,
        "radbcs": 1200,
        "radbcw": 1200,
    }
    assert invariants["maximum_cpu_tridiagonal_residual"] <= 1.0e-7
    assert receipt["kernel"]["adapter_source_sha256"] == (
        CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256
    )
    assert receipt["kernel"]["acoustic_source_sha256"] == (
        CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256
    )
    assert receipt["kernel"]["target_arch"] == "86"
    assert receipt["source"] == {
        "artifact_sha256": (
            "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f"
        ),
        "contract_sha256": (
            "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12"
        ),
        "manifest_sha256": (
            "2c09879f73afff9f8004134896eedb4b28e96c91194e871967c18f9359fb57ad"
        ),
        "payload_sha256": (
            "2237bc92d839cfd96b6e4cc82c9c39dc55c8f308353fc9b0b8e7db1db1bf6f82"
        ),
    }


def test_guarded_session_and_inventory_are_live_and_bounded() -> None:
    guarded = _load(GUARDED_RUN)
    assert guarded["status"] == "passed"
    assert guarded["pytest"] == {"passed": 3, "seconds": 43.45}
    assert guarded["gpu_mutex"]["acquired_by_this_run"] is True
    assert guarded["gpu_mutex"]["released_by_this_run"] is True
    assert guarded["gpu_mutex"]["lock_absent_after_release"] is True
    assert guarded["kernel_source_sha256"] == {
        "acoustic_composition": CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256,
        "initialized_product_adapter": CUDA_INITIALIZED_TERRAIN_OPEN_SOURCE_SHA256,
        "terrain_open_boundary": CUDA_TERRAIN_OPEN_BOUNDARY_SOURCE_SHA256,
    }
    assert guarded["evidence"] == {
        name: {"bytes": nbytes, "sha256": digest}
        for name, (nbytes, digest) in EXPECTED_RECEIPT_HASHES.items()
    }
    assert guarded["full_domain_acoustic_proof_executed"] is False

    inventory = _load(INVENTORY)
    assert inventory["fixture_id"] == "VALIDATION-TERRAIN-OPEN-PSOLVER3-V1"
    assert inventory["payload_exclusions"]["full_rust_payload_redistributed"] is False
    for artifact in inventory["artifacts"]:
        label = artifact["path"]
        assert label.startswith("source-distribution:")
        path = ROOT / label.removeprefix("source-distribution:")
        assert path.is_file()
        assert measured_identity(path) == (artifact["bytes"], artifact["sha256"])
        assert artifact["directly_recomputable_from_source_distribution"] is True
