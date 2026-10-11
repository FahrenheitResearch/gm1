from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def _module():
    return importlib.import_module("tools.run_real_gm1_f02_roi_parent")


def test_import_is_cuda_cold() -> None:
    before = set(sys.modules)
    module = _module()
    assert module.F02_ARTIFACT.startswith("356fe302")
    assert "cupy" not in set(sys.modules) - before


def test_snapshot_admission_requires_both_magnitude_and_interior_distance() -> None:
    module = _module()
    structure = {
        "fields": {
            "low_level_reflectivity_dbz": {
                "maximum": 45.0,
                "maximum_edge_distance_points": 50,
            }
        }
    }
    assert (
        module._snapshot_admission(
            structure,
            minimum_reflectivity_dbz=20.0,
            minimum_edge_distance_points=30,
        )["admitted"]
        is True
    )
    assert (
        module._snapshot_admission(
            structure,
            minimum_reflectivity_dbz=50.0,
            minimum_edge_distance_points=30,
        )["admitted"]
        is False
    )
    assert (
        module._snapshot_admission(
            structure,
            minimum_reflectivity_dbz=20.0,
            minimum_edge_distance_points=60,
        )["admitted"]
        is False
    )


def test_fixed_terrain_domain_and_child_candidate_are_fail_closed() -> None:
    module = _module()
    binding = module._load_domain_grid_binding(
        SimpleNamespace(
            domain_grid_roi_receipt=module.DOMAIN_GRID_ROI,
            domain_grid_roi_sha256=module.DOMAIN_GRID_ROI_SHA256,
            domain_grid_fixed_component=module.DOMAIN_GRID_FIXED,
            domain_grid_fixed_sha256=module.DOMAIN_GRID_FIXED_SHA256,
            f02_artifact_sha256=module.F02_ARTIFACT,
        )
    )
    assert binding["terrain"] is True
    assert binding["imove"] == 0
    assert binding["native_moving_continuation"] is False
    structure = {
        "fields": {
            "low_level_reflectivity_dbz": {
                "maximum": 63.18,
                "maximum_i": 243,
                "maximum_j": 217,
            },
            "column_updraft_ms": {
                "maximum": 37.31,
                "maximum_i": 264,
                "maximum_j": 199,
            },
            "low_level_abs_vertical_vorticity_s": {
                "maximum": 0.0463,
                "maximum_i": 247,
                "maximum_j": 258,
            },
        }
    }
    candidate = module._child_50m_candidate(structure)
    assert candidate["parent_crop_origin_ij"] == [149, 149]
    assert (
        min(
            value["candidate_child_edge_distance_points"]
            for value in candidate["anchors"].values()
        )
        >= 69
    )
    assert candidate["native_moving_domain"] is False


def test_source_identity_writer_pins_the_complete_f02_tuple(
    monkeypatch, tmp_path: Path
) -> None:
    module = _module()
    expected_source = {
        "geopotential_manifest_sha256": module.GEOPOTENTIAL_MANIFEST,
        "geopotential_payload_sha256": module.GEOPOTENTIAL_PAYLOAD,
        "geopotential_receipt_sha256": module.GEOPOTENTIAL_RECEIPT,
        "geopotential_source_restart_sha256": module.PRODUCT_RESTART,
        "product_manifest_sha256": module.PRODUCT_MANIFEST,
        "product_payload_sha256": module.PRODUCT_PAYLOAD,
        "product_source_artifact_sha256": module.F02_ARTIFACT,
        "product_source_contract_sha256": module.CONTRACT,
    }
    stdout = json.dumps(
        {
            "schema": "gm1.forecast-source-identities/v1",
            "source": expected_source,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, label: subprocess.CompletedProcess(
            command, 0, stdout=stdout + "\n", stderr=""
        ),
    )
    args = SimpleNamespace(
        checkpoint_bin=Path("checkpoint.exe"),
        geopotential_manifest_sha256=module.GEOPOTENTIAL_MANIFEST,
        geopotential_payload_sha256=module.GEOPOTENTIAL_PAYLOAD,
        geopotential_receipt_sha256=module.GEOPOTENTIAL_RECEIPT,
        product_restart_sha256=module.PRODUCT_RESTART,
        product_manifest_sha256=module.PRODUCT_MANIFEST,
        product_payload_sha256=module.PRODUCT_PAYLOAD,
        f02_artifact_sha256=module.F02_ARTIFACT,
        contract_sha256=module.CONTRACT,
    )
    path, digest, observed = module._write_f02_source_identities(args, tmp_path)
    assert observed == expected_source
    assert json.loads(path.read_bytes()) == {
        "schema": "gm1.forecast-source-identities/v1",
        "source": expected_source,
    }
    assert len(digest) == 64
