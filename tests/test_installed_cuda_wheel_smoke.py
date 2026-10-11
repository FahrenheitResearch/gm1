from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from types import ModuleType
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SMOKE = runpy.run_path(str(ROOT / "tools" / "smoke_installed_cuda_wheel.py"))
EVIDENCE = ROOT / "evidence" / "distribution"


def test_wheel_cuda_inventory_hashes_only_top_level_cuda_modules(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture.whl"
    source = b"CUDA_TEST_SOURCE = 'kernel'\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cm1core/cuda_test.py", source)
        archive.writestr("cm1core/not_cuda.py", b"ignored = True\n")
        archive.writestr("cm1core/nested/cuda_nested.py", b"ignored = True\n")

    assert SMOKE["wheel_cuda_inventory"](wheel) == [
        {
            "bytes": len(source),
            "member": "cm1core/cuda_test.py",
            "module": "cm1core.cuda_test",
            "sha256": hashlib.sha256(source).hexdigest(),
        }
    ]


def test_source_pin_rows_require_matching_text_bytes_and_sha256() -> None:
    module = ModuleType("fixture")
    module.CUDA_TEST_SOURCE = "extern \"C\" __global__ void test() {}\n"
    module.CUDA_TEST_SOURCE_BYTES = module.CUDA_TEST_SOURCE.encode("utf-8")
    module.CUDA_TEST_SOURCE_SHA256 = hashlib.sha256(
        module.CUDA_TEST_SOURCE_BYTES
    ).hexdigest()

    rows = SMOKE["source_pin_rows"](module)
    assert len(rows) == 1
    assert rows[0]["source"] == "CUDA_TEST_SOURCE"
    assert rows[0]["pin"] == "CUDA_TEST_SOURCE_SHA256"
    assert rows[0]["passed"] is True

    module.CUDA_TEST_SOURCE_BYTES = b"changed"
    assert SMOKE["source_pin_rows"](module)[0]["passed"] is False


def test_source_pin_rows_refuse_unpinned_cuda_source() -> None:
    module = ModuleType("fixture")
    module.CUDA_TEST_SOURCE = "kernel"

    rows = SMOKE["source_pin_rows"](module)
    assert len(rows) == 1
    assert rows[0]["expected_sha256"] is None
    assert rows[0]["passed"] is False


def test_installed_cuda_wheel_evidence_is_cross_version_and_byte_pinned() -> None:
    build_receipt = json.loads(
        (EVIDENCE / "cuda-wheel-build-receipt-v1.json").read_text(
            encoding="utf-8"
        )
    )
    inspection = json.loads(
        (EVIDENCE / "cuda-wheel-artifact-inspection-v1.json").read_text(
            encoding="utf-8"
        )
    )
    reports = [
        json.loads(
            (
                EVIDENCE
                / f"installed-cuda-wheel-smoke-py{version}-v1.json"
            ).read_text(encoding="utf-8")
        )
        for version in ("311", "313")
    ]

    wheel_sha256 = "3a57026f66ecc7c43eb8c56bc5ffd2c4b806691abb28a87a1ad40d1011b7f733"
    assert build_receipt["passed"] is True
    assert build_receipt["source_head"] == (
        "c3015ef2b9319636e50cd22d345869539ca7da8d"
    )
    assert build_receipt["artifacts"]["wheel"]["sha256"] == wheel_sha256
    assert inspection["passed"] is True
    assert inspection["artifacts"]["wheel"]["sha256"] == wheel_sha256
    assert {report["environment"]["python"] for report in reports} == {
        "3.11.9",
        "3.13.7",
    }
    for report in reports:
        assert report["passed"] is True
        assert report["wheel"]["sha256"] == wheel_sha256
        assert report["checks"]["installed-cuda-module-inventory"] == {
            "installed_count": 25,
            "missing": [],
            "passed": True,
            "unexpected": [],
            "wheel_count": 25,
        }
        assert report["checks"]["embedded-cuda-source-pins"] == {
            "failures": [],
            "passed": True,
            "pin_count": 22,
        }
        assert all(check["passed"] is True for check in report["checks"].values())

    installed = {
        row["module"]: row for row in reports[-1]["installed_cuda_modules"]
    }
    assert installed["cm1core.cuda_moist_stage_bridge"]["file_sha256"] == (
        "4d64ec89dc79d118c9876106b38c8b3fbda5fcee8a6311ea806af7dda5771200"
    )
    assert installed["cm1core.cuda_momentum_rk"]["file_sha256"] == (
        "b0bfd729782945049a6e1c58b809244fbd922057a4b787a8ea7cdaf5211feb92"
    )
    for name in ("cm1core.cuda_moist_stage_bridge", "cm1core.cuda_momentum_rk"):
        assert installed[name]["installed_under_package"] is True
        assert len(installed[name]["source_pins"]) == 1
        assert installed[name]["source_pins"][0]["passed"] is True
