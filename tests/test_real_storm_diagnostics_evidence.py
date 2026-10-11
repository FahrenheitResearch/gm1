from __future__ import annotations

import hashlib
import json
from pathlib import Path

from host_portability import recorded_name


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/validation/real-storm-diagnostics-v1"
RECEIPT = EVIDENCE / "receipt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_initialized_diagnostic_receipt_is_pinned_and_claim_limited() -> None:
    assert RECEIPT.stat().st_size == 7_085
    assert _sha256(RECEIPT) == (
        "a52600b1555035ad3e1c9245f1f6502259df600483bc2bc4117d45e488f5349d"
    )
    receipt = json.loads(RECEIPT.read_bytes())
    assert receipt["schema"] == "gpuwm-cm1.real-storm-diagnostics-rust/v1"
    assert receipt["status"] == "passed-real-initialized-diagnostic-not-forecast"
    assert receipt["configuration"] == {
        "domain": {"nx": 300, "ny": 300, "nz": 75},
        "forecast_evolution": False,
        "model_time_s": 0.0,
        "profile_id": "hrrr-cm1-real-state-v1",
        "schema": "gpuwm-cm1.real-storm-diagnostics-rust/v1",
        "spacing_m": {"dx": 100.0, "dy": 100.0, "dz": 200.0},
        "step": 0,
        "wsm6_transitions": 1,
    }
    assert "no dynamical forecast evolution" in receipt["claim_limit"]
    assert receipt["mutex_released"] is True
    assert receipt["runtime"]["device"] == "NVIDIA GeForce RTX 3080"
    assert receipt["runtime"]["compute_capability"] == "86"
    assert receipt["history"]["contract_sha256"] == (
        "0735df1a3d1d68b25798c0c9b710c2ad6d62dd81e1abb1086dcf8dbdd4727dc4"
    )


def test_real_initialized_diagnostic_fields_are_full_domain_and_finite() -> None:
    receipt = json.loads(RECEIPT.read_bytes())
    expected = {
        "reflectivity_dbz": (
            "a90667a8c9b43d6e4a739a41894773be8222c1725999b6fc90ba43f380696bc0",
            -35.0,
            10.287071228027344,
        ),
        "updraft_ms": (
            "3bae52e0dca3d1a2a9a9bc14dbb97a6e4165e6f7e5685121a0a202ce787f8a06",
            -0.0783490389585495,
            0.0566815584897995,
        ),
        "vertical_vorticity_s": (
            "4d5a50ebc8b7de5f714866310575654cac40d81e7ff14b23f97d924eefcceba1",
            -0.0003493450931273401,
            0.0005049280589446425,
        ),
    }
    assert set(receipt["fields"]) == set(expected)
    for name, (sha256, minimum, maximum) in expected.items():
        field = receipt["fields"][name]
        assert field["shape"] == [75, 300, 300]
        assert field["dtype"] == "<f4"
        assert field["nbytes"] == 27_000_000
        assert field["finite"] is True
        assert field["raw_sha256"] == sha256
        assert field["minimum"] == minimum
        assert field["maximum"] == maximum


def test_real_initialized_diagnostic_pngs_are_hash_bound() -> None:
    expected = {
        "real-t0-reflectivity.png": (
            3_367_946,
            "96d17509a0b3ba817272b8dde14b1b5db18ac16125616cda0264eec532932610",
        ),
        "real-t0-storm-diagnostics.png": (
            3_403_679,
            "382ebf070aa60d06a50a547f3efe9ea53fe0ac303dce50eb5a9f118baf0c713d",
        ),
        "real-t0-updraft.png": (
            3_367_950,
            "285daa9dbbb5ba6c1cc1ec62753358937335d94d2d05ac50a116d1264bdcfd36",
        ),
        "real-t0-vorticity.png": (
            3_367_966,
            "927df39e6dd0c5e7673d59af3dd24c651533d4b6f34e641150605cc444b5a540",
        ),
    }
    receipt = json.loads(RECEIPT.read_bytes())
    recorded = {
        recorded_name(value["path"]): value
        for value in receipt["renders"].values()
    }
    assert set(recorded) == set(expected)
    for name, (nbytes, sha256) in expected.items():
        path = EVIDENCE / name
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(payload) == nbytes == recorded[name]["nbytes"]
        assert hashlib.sha256(payload).hexdigest() == sha256 == recorded[name]["sha256"]
