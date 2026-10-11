"""Admission tests for the corrected stretched-z child closure consumer.

The child's fresh subgrid closure used to be admitted by one frozen SHA-256 per
file, and the artifact those hashes named is the one this project's own
evidence marks ``REFUSED_STRETCHED_Z_ORACLE``: its ``defv``/``defh`` carriers
are byte-identical to the sealed uniform-z negative control, so the fresh
km/kh and TKE the child starts from were diagnosed for a vertical grid the
child does not have.

These tests cover the replacement: admit any closure that is internally
consistent with its own receipt, and refuse the discredited result by identity
and by fingerprint so a rerun that reproduces the defect under a fresh digest
is caught too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cm1core import gm1_child_stretched_closure_v3 as closure


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def test_the_superseded_uniform_z_closure_is_refused_by_identity() -> None:
    with pytest.raises(ValueError) as error:
        closure.refuse_uniform_z_closure(
            manifest_sha256=closure.REFUSED_UNIFORM_Z_MANIFEST_SHA256,
            payload_sha256="00" * 32,
            receipt_sha256="11" * 32,
            carrier_sha256={},
        )
    assert "superseded uniform-z child closure is refused" in str(error.value)
    assert "vertical grid the child does not have" in str(error.value)


def test_a_rerun_reproducing_the_uniform_z_deformation_is_refused_by_fingerprint() -> None:
    """A fresh digest does not launder the defect.

    A hash pin would admit this: every file digest is new.  What is wrong is
    the deformation itself.
    """

    for name, digest in (
        ("defv", closure.REFUSED_UNIFORM_Z_DEFV_SHA256),
        ("defh", closure.REFUSED_UNIFORM_Z_DEFH_SHA256),
    ):
        with pytest.raises(ValueError) as error:
            closure.refuse_uniform_z_closure(
                manifest_sha256="aa" * 32,
                payload_sha256="bb" * 32,
                receipt_sha256="cc" * 32,
                carrier_sha256={name: digest},
            )
        assert f"closure carrier {name} is byte-identical" in str(error.value)
        assert "stretched-grid deformation" in str(error.value)


def test_the_corrected_stretched_closure_is_not_refused() -> None:
    closure.refuse_uniform_z_closure(
        manifest_sha256="aa" * 32,
        payload_sha256="bb" * 32,
        receipt_sha256="cc" * 32,
        carrier_sha256={
            "defv": "35c0ce245ef7b647331cf6796b492d81bd1d6da6d63bfe19f21c3e1bce5c4167",
            "defh": "930f8c270bdc59e8ba2d6d37301bbfaab79de9d81f3cbe490fc74b14c22f0184",
        },
    )


def _write(root: Path, manifest: dict, receipt: dict, payload: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "closure.f32le").write_bytes(payload)
    (root / "manifest.json").write_bytes(canonical(manifest))
    (root / "receipt.json").write_bytes(canonical(receipt))


def _documents(payload: bytes) -> tuple[dict, dict]:
    grid = closure.GRID_50M
    manifest = {
        "abi": closure.ABI,
        "artifact_schema": closure.SCHEMA,
        "grid": grid.record() | {"vertical_grid_sha256": closure.VERTICAL_GRID_SHA256},
        "payload": {
            "file": "closure.f32le",
            "nbytes": closure.closure_payload_nbytes(grid),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "schema": closure.MANIFEST_SCHEMA,
        "status": closure.STATUS,
    }
    receipt = {
        "abi": closure.ABI,
        "artifact_schema": closure.SCHEMA,
        "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "payload_nbytes": closure.closure_payload_nbytes(grid),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "schema": closure.RECEIPT_SCHEMA,
        "status": closure.STATUS,
    }
    return manifest, receipt


def test_a_receipt_that_does_not_anchor_its_own_manifest_is_refused(
    tmp_path: Path,
) -> None:
    payload = b"\x00" * 16
    manifest, receipt = _documents(payload)
    receipt["manifest_sha256"] = "ff" * 32
    _write(tmp_path / "closure", manifest, receipt, payload)
    with pytest.raises(ValueError) as error:
        closure.load_gm1_child_stretched_closure(tmp_path / "closure")
    assert "not the manifest its own receipt anchors" in str(error.value)


def test_a_foreign_closure_family_is_refused_by_schema(tmp_path: Path) -> None:
    payload = b"\x00" * 16
    manifest, receipt = _documents(payload)
    manifest["schema"] = "gm1.real-50m-child-fresh-closure-manifest/v2"
    receipt["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    _write(tmp_path / "closure", manifest, receipt, payload)
    with pytest.raises(ValueError) as error:
        closure.load_gm1_child_stretched_closure(tmp_path / "closure")
    assert "corrected stretched-z family" in str(error.value)


def test_a_closure_for_another_grid_is_refused(tmp_path: Path) -> None:
    payload = b"\x00" * 16
    manifest, receipt = _documents(payload)
    manifest["grid"] = closure.GRID_5M.record() | {
        "vertical_grid_sha256": closure.VERTICAL_GRID_SHA256
    }
    receipt["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    _write(tmp_path / "closure", manifest, receipt, payload)
    with pytest.raises(ValueError) as error:
        closure.load_gm1_child_stretched_closure(tmp_path / "closure")
    assert "offset belonging to a different column" in str(error.value)


def test_a_payload_that_is_not_the_recorded_arena_is_refused(tmp_path: Path) -> None:
    payload = b"\x00" * 16
    manifest, receipt = _documents(payload)
    _write(tmp_path / "closure", manifest, receipt, payload)
    with pytest.raises(ValueError) as error:
        closure.load_gm1_child_stretched_closure(tmp_path / "closure")
    assert "no published layout" in str(error.value)


def test_a_non_canonical_document_is_refused(tmp_path: Path) -> None:
    payload = b"\x00" * 16
    manifest, receipt = _documents(payload)
    root = tmp_path / "closure"
    _write(root, manifest, receipt, payload)
    (root / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2).encode("ascii")
    )
    with pytest.raises(ValueError) as error:
        closure.load_gm1_child_stretched_closure(root)
    assert "canonical sorted-key compact JSON" in str(error.value)


def test_the_production_child_composer_consumes_the_corrected_closure() -> None:
    """The child large-step composer must not load the discredited artifact."""

    from cm1core import cuda_gm1_child_step_composer as composer

    assert composer.GM1_CHILD_CLOSURE_NAME == "gm1-t300-child-stretched-closure-v3"
    source = Path(composer.__file__).read_text(encoding="utf-8")
    assert "load_gm1_child_closure_v2" not in source
    assert "load_gm1_child_stretched_closure" in source
