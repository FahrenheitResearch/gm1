from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PASS = ROOT / (
    "evidence/validation/"
    "gm1-idealized-iinit1-completed-rk3-local-v1-attempt-0002/receipt.json"
)
RK1 = ROOT / (
    "evidence/validation/"
    "gm1-idealized-iinit1-completed-rk3-local-v1-attempt-0002-rk1/receipt.json"
)
REFUSAL = ROOT / (
    "evidence/validation/"
    "gm1-idealized-iinit1-completed-rk3-local-v1-attempt-0001-refusal/receipt.json"
)
RK1_REFUSAL = ROOT / (
    "evidence/validation/"
    "gm1-idealized-iinit1-completed-rk3-local-v1-attempt-0001-rk1-refusal/receipt.json"
)
PASS_SHA256 = "ac384ef130c06d1df08bd628960c5535ca809daa711dfd35f2ba2964d57627c0"
RK1_SHA256 = "475c681adeed6b24947a940691f629b08ebbaeb0cc79bf5b38fe8f2e2f183b81"
REFUSAL_SHA256 = "52dcf6da2b1693cac92bb293ce26793fffb530c7a3386fc41ece8af115c532ed"
RK1_REFUSAL_SHA256 = (
    "c69f02dba8fce85b23a35a257dc41bfa9fb3e245b16315bea96bf3a59e6291c7"
)
PRODUCER = "047334f570e6c991bf965564b0d73aa1f7e1686f"
RUNNER_SHA256 = "4bc36da008862ee90ee1257585444105d9854c1a719709f7830ff8daebbf88a0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(path: Path, expected_sha256: str) -> dict[str, object]:
    assert _sha(path) == expected_sha256
    return json.loads(path.read_text(encoding="ascii"))


def _sha_at_commit(commit: str, relative: str) -> str:
    raw = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(raw).hexdigest()


def test_real_idealized_same_residency_completed_rk3_is_sealed() -> None:
    receipt = _receipt(PASS, PASS_SHA256)
    assert receipt["schema"] == "gm1.idealized-iinit1-zero-disk-completed-rk3/v1"
    assert receipt["status"] == "PASS_IDEALIZED_IINIT1_ZERO_DISK_COMPLETED_RK3"
    assert receipt["producer"] == {
        "git_commit": PRODUCER,
        "runner_sha256": RUNNER_SHA256,
    }
    assert _sha_at_commit(
        PRODUCER, "tools/run_real_gm1_idealized_tilestream_rk2_rk3.py"
    ) == RUNNER_SHA256
    assert receipt["same_live_cuda_residency"] is True
    assert receipt["uniform_idealized_validation"] is True
    assert receipt["domain"] == {
        "shape_zyx": [75, 300, 300],
        "spacing_m": [50.0, 50.0, 200.0],
        "dt_seconds": 0.5,
    }
    assert receipt["tokens"] == {
        "completed_rk1": "24f5bcca4405d9f9916decdab1372292d91e7c45a0382bd09a34ae1d54f1a718",
        "completed_rk2": "0bf4cccb2ff93f50da0d88afed96d11030dfd89dcd371a9b1714026d7d7e7047",
        "completed_rk3": "ab82adf26dd8aabb3d7527eed2b605e9e06a0ebb514a2a0613a0a813468201fc",
        "rk3_mass_finish": "799a02f8b79d8b40c0dde9dfe81cb8ff27e0e9fed0936ac2225681b8bb7aeaa4",
    }
    assert receipt["post_rk3_physics"] is False
    assert receipt["solve3"] is False
    assert receipt["completed_time_step"] is False
    assert receipt["product"] is False
    assert receipt["tornado_claim"] is False


def test_initial_mass_and_rk1_lineage_are_sealed() -> None:
    receipt = _receipt(PASS, PASS_SHA256)
    mass = receipt["source"]["initial_dry_mass"]
    assert mass["mass1_f64"] == 3998626.449108124
    assert mass["mass1_f64_le_sha256"] == (
        "1e2cbf8b97061a4a4f84df86274e79d8d11985e4b1d200f42f91e138d71e370c"
    )
    assert mass["source_sha256"] == {
        "src/init3d.F": "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        "src/param.F": "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
    }
    assert receipt["source"]["sounding_sha256"] == (
        "56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0"
    )
    assert receipt["source"]["rust_library_sha256"] == (
        "10ba6473422e9cffac8955a6ef68b6d47e0b58571d7550da7250a19dabcd754d"
    )
    rk1 = _receipt(RK1, RK1_SHA256)
    assert rk1["producer_git_commit"] == PRODUCER
    assert rk1["tokens"]["completed_rk1"] == receipt["tokens"]["completed_rk1"]
    assert rk1["zero_disk_rust_to_cuda"] is True
    assert rk1["completed_rk1"] is True


def test_failed_binding_lineage_attempt_is_preserved() -> None:
    refusal = _receipt(REFUSAL, REFUSAL_SHA256)
    nested = _receipt(RK1_REFUSAL, RK1_REFUSAL_SHA256)
    for receipt in (refusal, nested):
        assert receipt["producer_git_commit"] == (
            "17ecd908b55363d91d58bcd8246c8fddd6c3fa90"
        )
        assert receipt["error"] == "pending RK23 pre-acoustic identity changed"
        assert receipt["error_type"] == "ValueError"
        assert receipt["mutex_clean_release"] is True
        assert receipt["product"] is False
        assert receipt["tornado_claim"] is False

