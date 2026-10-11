from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cm1core.cuda_gm1_child_post_final_rk as post_final
from cm1core.gm1_child_operator_manifest import (
    GM1_BASELINE_OPERATOR_SOURCE_PATHS,
    GM1_CHILD_OPERATOR_SOURCE_PATHS,
    build_gm1_child_operator_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_child_operator_manifest_is_exact_no_newline_and_cross_sealed() -> None:
    manifest = build_gm1_child_operator_manifest(ROOT)
    assert not manifest.canonical_bytes.endswith(b"\n")
    assert hashlib.sha256(manifest.canonical_bytes).hexdigest() == manifest.sha256
    parsed = json.loads(manifest.canonical_bytes)
    assert parsed == {
        "model": "GM1",
        "operators": dict(manifest.operators),
        "schema": "gm1.operator-source-manifest/v1",
    }
    assert post_final.require_child_post_final_rk_operator_sources(
        manifest.operators
    ) == manifest.sha256


def test_baseline_is_preserved_and_child_sources_are_namespaced() -> None:
    manifest = build_gm1_child_operator_manifest(ROOT)
    assert tuple(GM1_BASELINE_OPERATOR_SOURCE_PATHS) == tuple(
        post_final.GM1_BASELINE_OPERATOR_KEYS
    )
    assert not set(GM1_BASELINE_OPERATOR_SOURCE_PATHS) & set(
        GM1_CHILD_OPERATOR_SOURCE_PATHS
    )
    assert manifest.operators["closure.tke-evolution"] == hashlib.sha256(
        (ROOT / GM1_BASELINE_OPERATOR_SOURCE_PATHS["closure.tke-evolution"])
        .read_bytes()
    ).hexdigest()
    assert manifest.operators[post_final.CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY] == (
        hashlib.sha256(
            (
                ROOT
                / GM1_CHILD_OPERATOR_SOURCE_PATHS[
                    post_final.CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY
                ]
            ).read_bytes()
        ).hexdigest()
    )
    assert (
        manifest.operators["closure.tke-evolution"]
        != manifest.operators[post_final.CUDA_GM1_CHILD_STRETCHED_SGS_TKE_OPERATOR_KEY]
    )


def test_manifest_refuses_source_drift_before_publication(tmp_path: Path) -> None:
    # A fake root cannot accidentally inherit or silently synthesize sources.
    try:
        build_gm1_child_operator_manifest(tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing exact operator files were accepted")
