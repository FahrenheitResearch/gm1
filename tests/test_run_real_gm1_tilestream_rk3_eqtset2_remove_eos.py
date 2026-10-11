from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_real_gm1_tilestream_rk3_eqtset2_remove_eos.py"


def _namespace() -> dict[str, object]:
    return runpy.run_path(str(RUNNER), run_name="gm1_rk3_eqtset2_runner_test")


def test_runner_declares_only_real_detached_local_evidence() -> None:
    namespace = _namespace()
    assert namespace["STATUS"] == (
        "PASS_REAL_CHILD_300X300X75_RK3_EQTSET2_LOCAL_TILED_RAW_BIT_EXACT"
    )
    claim = str(namespace["CLAIM_LIMIT"])
    assert "real admitted f02 source-coordinate" in claim
    assert "exact CM1 restart positive-zero held-tendency disposition" in claim
    assert "not a fresh live RK3 scalar state" in claim
    assert "synthetic" not in claim.lower()
    assert tuple(namespace["FIELD_ORDER"]) == (
        "pi0",
        "th0",
        "pp3d",
        "th3d",
        "qv",
        "qc",
        "qpten",
        "qtten",
        "qvten",
        "qcten",
        "prs",
        "rho",
    )


def test_raw_word_comparator_detects_one_changed_bit() -> None:
    namespace = _namespace()
    compare = namespace["_word_mismatches"]
    left = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    right = left.copy()
    assert compare(left, right) == 0
    right.view(np.uint32)[1] ^= np.uint32(1)
    assert compare(left, right) == 1


def test_failed_attempt_is_preserved_without_a_pass_claim(tmp_path: Path) -> None:
    namespace = _namespace()
    preserve = namespace["_preserve_refusal"]
    path = preserve(tmp_path, RuntimeError("expected gate failure"))
    payload = json.loads(path.read_bytes())
    assert payload["status"] == "REFUSED_NO_SCIENTIFIC_PASS"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "expected gate failure"
    assert "numerical or product claim" in payload["claim_limit"]


def test_runner_source_keeps_fixed_buffer_and_lineage_refusals_explicit() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "single_fixed_device_buffer_set_reused" in source
    assert "fresh_rk3_scalar_moist_lineage_bound" in source
    assert '"completed_rk3_emitted": False' in source
    assert '"synthetic_translation": False' in source
    assert "tile_device[name].set(tile_host[name], stream=stream)" in source
    assert "resident and fixed-tiled eqtset2 raw words differ" in source

