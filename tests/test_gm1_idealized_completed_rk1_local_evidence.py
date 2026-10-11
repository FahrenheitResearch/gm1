from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / (
    "evidence/validation/"
    "gm1-idealized-iinit1-completed-rk1-local-v1-attempt-0002/receipt.json"
)
RECEIPT_SHA256 = (
    "4739087f65d8377b729f980d6676eca5f48d5a8c1fa285c321d3112adddd907b"
)


def test_immutable_local_idealized_completed_rk1_pass_is_sealed() -> None:
    raw = RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    receipt = json.loads(raw)
    assert receipt["schema"] == "gm1.idealized-iinit1-zero-disk-completed-rk1/v1"
    assert receipt["status"] == "PASS_IDEALIZED_IINIT1_ZERO_DISK_COMPLETED_RK1"
    assert receipt["producer_git_commit"] == (
        "e4bb231bb1d8e4cd0145ef4e5b977c686c0483d0"
    )
    assert receipt["completed_rk1"] is True
    assert receipt["rk2_rk3"] is False
    assert receipt["solve3"] is False
    assert receipt["product"] is False
    assert receipt["tornado_claim"] is False
    assert receipt["tokens"]["completed_rk1"] == (
        "8078c36bab4d46716430befcddf8e319c6680b92372ca18b4c397347db779d41"
    )
    assert receipt["tokens"]["source_ordered_tke"] == (
        "4ed3fc3c65b49b0d94aaafafee1b4f082567a94510d56f263e4d7a353de49cc7"
    )
    assert receipt["runtime"]["wall_seconds"] > 0.0
