from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from host_portability import recorded_relative_to


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "evidence" / "validation" / "real-moist-stage-bridge-v1"


def test_real_moist_stage_bridge_receipt_and_png_are_hash_bound() -> None:
    receipt_path = DIRECTORY / "full-domain-sm86.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "gpuwm-cm1.real-moist-stage-bridge-evidence/v1"
    assert receipt["result"] == "passed-real-product-coupled-rk1-stage-diagnostic"
    assert receipt["classification"] == "stage-1-operator-diagnostic-not-forecast-not-tornado"
    assert receipt["selection"]["domain"] == [300, 300, 75]
    assert receipt["selection"]["spacing_m"] == [100.0, 100.0, 200.0]
    assert receipt["selection"]["rk_stage"] == 1
    assert receipt["selection"]["source_data"] == (
        "real HRRR initialized product; no synthetic atmospheric fields"
    )
    assert receipt["source_export"]["manifest_sha256"] == (
        "7fd218a005b1c9f01d37b57452b281c2df6c797973268e8fe86b996eb6b61e2e"
    )
    assert receipt["source_export"]["payload_sha256"] == (
        "8f589cee64b044afcfef34dca6b638c3829946a999537885b1db3576d4ec7981"
    )
    for item in receipt["statistics"].values():
        assert item["nonfinite_words"] == 0
        assert item["nonzero_words"] > 0
    assert receipt["gpu_mutex"]["existing_owner_evicted"] is False
    assert receipt["gpu_mutex"]["clean_release"] is True
    figure = receipt["figure"]
    recorded = recorded_relative_to(figure["path"], "evidence")
    assert recorded == PurePosixPath(
        "evidence/validation/real-moist-stage-bridge-v1"
        "/real-coupled-rk1-stage-diagnostic.png"
    )
    path = ROOT / recorded
    payload = path.read_bytes()
    assert len(payload) == figure["nbytes"]
    assert hashlib.sha256(payload).hexdigest() == figure["sha256"]
    assert figure["label"] == "coupled-rk1-stage-diagnostic-not-forecast"
