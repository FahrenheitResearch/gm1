from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from cm1core.cuda_gm1_tilestream import CHECKPOINT_NAMES, checkpoint_carrier_shape
from cm1core.gm1_native_5m_complete_envelope import (
    CHECKPOINT_NBYTES,
    FIRST_STEP_SCHEMA,
    MANIFEST_SCHEMA,
    MISSING_LIFECYCLE_GATES,
    STATUS,
    audit_native_5m_complete_envelope,
    build_envelope_documents,
    build_persistent_sources,
    default_paths,
    inspect_envelope,
    publish_envelope,
)


ROOT = Path(__file__).resolve().parents[1]


def _carrier(name: str, shape: tuple[int, ...], *, offset: int = 0) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(shape),
        "offset_nbytes": offset,
        "nbytes": math.prod(shape) * 4,
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
    }


def _minimal_manifests() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    atmosphere_names = [
        name
        for name in CHECKPOINT_NAMES
        if name.startswith("component.")
        or name.startswith("held.")
        or name in {"qa", "q3d"}
    ]
    atmosphere_rows = []
    offset = 0
    for name in atmosphere_names:
        row = _carrier(name, checkpoint_carrier_shape(name, nx=3000, ny=3000, nz=75))
        row["offset_bytes"] = offset
        row.pop("offset_nbytes")
        offset += int(row["nbytes"])
        atmosphere_rows.append(row)
    atmosphere = {
        "arena": {"file": "arena.f32le"},
        "carriers": atmosphere_rows,
    }

    auxiliary_shapes = {
        "ppi": (77, 3002, 3002),
        "pp3d": (77, 3002, 3002),
        "ppx": (77, 3002, 3002),
        "tha": (77, 3006, 3006),
        "th3d": (77, 3006, 3006),
        "ua": (77, 3006, 3007),
        "u3d": (77, 3006, 3007),
        "va": (77, 3007, 3006),
        "v3d": (77, 3007, 3006),
        "wa": (78, 3006, 3006),
        "w3d": (78, 3006, 3006),
        "radbcw": (77, 3006),
        "radbce": (77, 3006),
        "radbcs": (77, 3006),
        "radbcn": (77, 3006),
    }
    terrain_rows = [
        _carrier(name, shape, offset=index * 1000)
        for index, (name, shape) in enumerate(auxiliary_shapes.items())
    ]
    density_rows = [
        _carrier(
            name,
            checkpoint_carrier_shape(name, nx=3000, ny=3000, nz=75),
            offset=index * 1000,
        )
        for index, name in enumerate(("prs", "rho", "rr", "rf"))
    ]
    auxiliary = {
        "outputs": [
            {"file": "terrain-open.f32le", "carriers": terrain_rows},
            {"file": "live-density.f32le", "carriers": density_rows},
        ]
    }
    closure = {
        "payload": {"file": "closure.f32le"},
        "carriers": [
            _carrier("tkea", checkpoint_carrier_shape("closure.tkea", nx=3000, ny=3000, nz=75))
        ],
    }
    return atmosphere, auxiliary, closure


def _fake_audit() -> dict[str, object]:
    return {
        "schema": "fixture",
        "status": STATUS,
        "runnable_5m_product": False,
        "canonical_checkpoint_projection": {"carrier_count": 37},
    }


def test_canonical_37_carrier_projection_is_exact_and_refused() -> None:
    rows = build_persistent_sources(*_minimal_manifests())
    assert tuple(row["name"] for row in rows) == CHECKPOINT_NAMES
    assert len(rows) == 37
    assert sum(int(row["nbytes"]) for row in rows) == CHECKPOINT_NBYTES
    assert CHECKPOINT_NBYTES == 97_603_303_392
    assert next(row for row in rows if row["name"] == "u3d")["source"][
        "source_slices"
    ] == [[1, 76, 1], [3, 3003, 1], [3, 3004, 1]]
    assert next(row for row in rows if row["name"] == "closure.tkea")["shape"] == [
        76,
        3000,
        3000,
    ]


def test_carrier_projection_refuses_wrong_physical_view() -> None:
    atmosphere, auxiliary, closure = _minimal_manifests()
    row = next(
        item
        for item in auxiliary["outputs"][0]["carriers"]
        if item["name"] == "u3d"
    )
    row["shape"] = [77, 3006, 3003]
    with pytest.raises(ValueError, match="u3d source slices exceed"):
        build_persistent_sources(atmosphere, auxiliary, closure)


def test_documents_keep_first_step_checkpoint_and_product_refused() -> None:
    documents = build_envelope_documents(
        _fake_audit(),
        producer={"module_sha256": "1" * 64, "source_sha256": "2" * 64},
    )
    assert tuple(documents) == (
        "checkpoint-admission.json",
        "first-step-refusal.json",
        "manifest.json",
        "receipt.json",
    )
    manifest = json.loads(documents["manifest.json"])
    first = json.loads(documents["first-step-refusal.json"])
    checkpoint = json.loads(documents["checkpoint-admission.json"])
    receipt = json.loads(documents["receipt.json"])
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert first["schema"] == FIRST_STEP_SCHEMA
    assert first["first_large_step_ready"] is False
    assert first["next_required_gate"] == MISSING_LIFECYCLE_GATES[0]
    assert checkpoint["initialization_boundary_admitted_as_checkpoint"] is False
    assert checkpoint["status"].startswith("REFUSED")
    assert receipt["runnable_5m_product"] is False


def test_atomic_reference_envelope_roundtrip_and_overwrite_refusal(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "envelope"
    result = publish_envelope(
        destination,
        _fake_audit(),
        producer={"module_sha256": "1" * 64, "source_sha256": "2" * 64},
    )
    assert result == inspect_envelope(destination)
    assert result["runnable_5m_product"] is False
    assert sorted(path.name for path in destination.iterdir()) == [
        "checkpoint-admission.json",
        "first-step-refusal.json",
        "manifest.json",
        "receipt.json",
    ]
    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_envelope(
            destination,
            _fake_audit(),
            producer={"module_sha256": "1" * 64, "source_sha256": "2" * 64},
        )


def test_real_native_component_audit_when_artifacts_are_attached() -> None:
    paths = default_paths(ROOT)
    if not paths.auxiliary.is_dir() or not paths.closure.is_dir():
        pytest.skip("external native 5 m F: artifacts are not attached")
    result = audit_native_5m_complete_envelope(paths)
    projection = result["canonical_checkpoint_projection"]
    assert projection["carrier_count"] == 37
    assert projection["nbytes"] == CHECKPOINT_NBYTES
    assert result["corrected_closure"]["historical_impact_record_count"] == 59
    assert result["historical_300x300_complete_child"]["status"] == (
        "REFUSED_REVALIDATION_REQUIRED"
    )
    assert result["initialization_component_set_admitted"] is True
    assert result["first_large_step_ready"] is False
    assert result["checkpoint_available"] is False
    assert result["runnable_5m_product"] is False
