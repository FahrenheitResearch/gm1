from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def _module():
    return importlib.import_module("cm1core.gm1_resume")


def _fixture() -> tuple[dict[str, object], dict[str, object], object, object, object]:
    module = _module()
    source_profile = module.GM1_RESUME_SOURCE_PROFILES["legacy-initial-product-v3"]
    checkpoint = importlib.import_module("cm1core.gm1_checkpoint")
    carriers = []
    offset = 0
    for name in checkpoint.GM1_CHECKPOINT_FIELDS:
        row = {
            "dtype": "<f4",
            "itemsize": 4,
            "name": name,
            "nbytes": 4,
            "offset_nbytes": offset,
            "order": "C",
            "sha256": "1" * 64,
            "shape": [1, 1, 1],
        }
        carriers.append(row)
        offset += 4
    layout = {
        "carriers": [
            {
                "name": row["name"],
                "nbytes": row["nbytes"],
                "offset_nbytes": row["offset_nbytes"],
                "shape": row["shape"],
            }
            for row in carriers
        ],
        "payload_nbytes": offset,
    }
    selection = SimpleNamespace(
        acoustic_substeps=6,
        coriolis=0,
        dry_mass_correction=True,
        dt_s=0.5,
        equation_set=2,
        nz=75,
        ny=300,
        nx=300,
        dx_m=100.0,
        dy_m=100.0,
        dz_m=200.0,
        lateral_boundaries=(2, 2, 2, 2),
        microphysics="wsm6-mp6",
        moving_frame=False,
        psolver=3,
        terrain=True,
    )
    product = SimpleNamespace(
        export=SimpleNamespace(
            manifest_sha256=source_profile["product_manifest_sha256"],
            payload_sha256=source_profile["product_payload_sha256"],
            source_artifact_sha256=source_profile["product_source_artifact_sha256"],
            source_contract_sha256=source_profile["product_source_contract_sha256"],
        )
    )
    geopotential = SimpleNamespace(
        manifest_sha256=source_profile["geopotential_manifest_sha256"],
        payload_sha256=source_profile["geopotential_payload_sha256"],
        receipt_sha256=source_profile["geopotential_receipt_sha256"],
    )
    operator = {
        "model": "GM1",
        "operators": {"forecast.coordinator": "9" * 64},
        "schema": "gm1.operator-source-manifest/v1",
    }
    manifest = {
        "aliases": [
            {
                "alias": alias,
                "canonical": canonical,
                "relation": "same live allocation and exact byte span",
            }
            for alias, canonical in checkpoint.GM1_WSM6_ALIASES.items()
        ],
        "capture_phase": "post-large-step",
        "carrier_abi": checkpoint.GM1_CHECKPOINT_CARRIER_ABI,
        "carriers": carriers,
        "claim_limit": "bounded by Rust",
        "configuration": {
            "admission": "production-full-domain",
            "capture_domain": {
                "crop_origin_yx": [0, 0],
                "crop_shape_zyx": [75, 300, 300],
                "source_shape_zyx": [75, 300, 300],
            },
            "selection": module._expected_selection(selection),
        },
        "dtype": "<f4",
        "itemsize": 4,
        "model": "GM1",
        "operator_manifest": operator,
        "order": "C",
        "payload": {
            "file": "gm1-carriers.f32le",
            "nbytes": offset,
            "sha256": "a" * 64,
        },
        "restore_phase": "before-next-large-step",
        "scheduler_extension": {
            "entries": {
                "closure.completed_stage": 0,
                "closure.refresh_index": 4,
                "closure.tke3d_reconstructible_from_tkea": True,
                "reflectivity.schedule_counter": 2,
            },
            "schema": module.GM1_SCHEDULER_EXTENSION_SCHEMA,
        },
        "scheduler_scalars": {
            "completed_large_steps": 3,
            "completed_wsm6_bridge_calls": 3,
            "completed_wsm6_component_calls": 3,
            "completed_wsm6_transitions": 3,
            "large_step_boundary": True,
            "mass1_f64": "40c81cd6c8b43958",
            "moist_buoyancy_applied": False,
            "moist_post_acoustic_ready": False,
            "moist_terms_ready": False,
            "next_stage": 1,
            "post_stage_hook_active": False,
            "quiescent_capture": True,
            "radiation_history_initialized": True,
            "scalar_completed": False,
            "scalar_next_nrk": 1,
            "solve1_ready": False,
        },
        "schema": module.GM1_RESTORE_DIRECTORY_SCHEMA,
        "source": {
            "geopotential_manifest_sha256": geopotential.manifest_sha256,
            "geopotential_payload_sha256": geopotential.payload_sha256,
            "geopotential_receipt_sha256": geopotential.receipt_sha256,
            "geopotential_source_restart_sha256": source_profile[
                "geopotential_source_restart_sha256"
            ],
            "product_manifest_sha256": product.export.manifest_sha256,
            "product_payload_sha256": product.export.payload_sha256,
            "product_source_artifact_sha256": product.export.source_artifact_sha256,
            "product_source_contract_sha256": product.export.source_contract_sha256,
        },
        "source_checkpoint": {
            "artifact_sha256": "b" * 64,
            "contract_sha256": "c" * 64,
            "dt_f64_bits": "3fe0000000000000",
            "model_time_f64_bits": "3ff8000000000000",
            "step": 3,
        },
        "species_order": ["qv", "qc", "qr", "qi", "qs", "qg"],
    }
    return manifest, layout, product, geopotential, selection


def test_import_is_cuda_cold() -> None:
    before = set(sys.modules)
    module = _module()
    assert module.GM1_LIVE_RESUME_SCHEMA.endswith("/v1")
    assert "cupy" not in set(sys.modules) - before


def test_restore_manifest_binds_every_safe_boundary_counter() -> None:
    module = _module()
    manifest, layout, product, geopotential, selection = _fixture()
    metadata = module._validate_restore_manifest(
        manifest,
        layout=layout,
        product=product,
        geopotential=geopotential,
        operator_manifest=manifest["operator_manifest"],
        contract_sha256="c" * 64,
        artifact_sha256="b" * 64,
        selection=selection,
    )
    assert metadata == {
        "completed_transitions": 3,
        "mass1_f64": pytest.approx(12345.677999999998),
        "mass1_f64_bits": "40c81cd6c8b43958",
        "model_time_f64_bits": "3ff8000000000000",
        "model_time_s": 1.5,
        "payload_sha256": "a" * 64,
        "reflectivity_schedule_counter": 2,
        "refresh_index": 4,
        "step": 3,
    }


def test_restore_manifest_admits_only_exact_f02_roi_source_and_scheduler() -> None:
    module = _module()
    manifest, layout, product, geopotential, selection = _fixture()
    profile = module.GM1_RESUME_SOURCE_PROFILES["real-hrrr-f02-roi"]
    product.export.manifest_sha256 = profile["product_manifest_sha256"]
    product.export.payload_sha256 = profile["product_payload_sha256"]
    product.export.source_artifact_sha256 = profile["product_source_artifact_sha256"]
    product.export.source_contract_sha256 = profile["product_source_contract_sha256"]
    geopotential.manifest_sha256 = profile["geopotential_manifest_sha256"]
    geopotential.payload_sha256 = profile["geopotential_payload_sha256"]
    geopotential.receipt_sha256 = profile["geopotential_receipt_sha256"]
    manifest["source"] = dict(profile)
    entries = manifest["scheduler_extension"]["entries"]
    entries.update(
        {
            "domain-grid.fixed-component-sha256": (
                module.GM1_F02_ROI_FIXED_DOMAIN_COMPONENT_SHA256
            ),
            "domain-grid.imove": 0,
            "domain-grid.roi-receipt-sha256": (
                module.GM1_F02_ROI_DOMAIN_GRID_RECEIPT_SHA256
            ),
            "source.initialization-absolute-offset-s": 7200.0,
            "source.initialization-artifact-sha256": profile[
                "product_source_artifact_sha256"
            ],
            "source.initialization-contract-sha256": profile[
                "product_source_contract_sha256"
            ],
            "source.initialization-kind": "boundary-analysis-history-f02-roi",
            "source.relative-model-time-s": 1.5,
            "source.roi-center-latitude": module._f64_from_bits(
                module.GM1_F02_ROI_CENTER_LAT_F64_BITS, "latitude"
            ),
            "source.roi-center-longitude": module._f64_from_bits(
                module.GM1_F02_ROI_CENTER_LON_F64_BITS, "longitude"
            ),
            "source.series-receipt-sha256": (module.GM1_F02_ROI_SERIES_RECEIPT_SHA256),
            "source.synthetic-translation": False,
        }
    )
    metadata = module._validate_restore_manifest(
        manifest,
        layout=layout,
        product=product,
        geopotential=geopotential,
        operator_manifest=manifest["operator_manifest"],
        contract_sha256="c" * 64,
        artifact_sha256="b" * 64,
        selection=selection,
    )
    assert metadata["step"] == 3

    entries["source.synthetic-translation"] = True
    with pytest.raises(ValueError, match="f02 ROI source scheduler identity"):
        module._validate_restore_manifest(
            manifest,
            layout=layout,
            product=product,
            geopotential=geopotential,
            operator_manifest=manifest["operator_manifest"],
            contract_sha256="c" * 64,
            artifact_sha256="b" * 64,
            selection=selection,
        )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("scheduler_scalars", "next_stage"), 2, "scheduler cadence"),
        (
            ("scheduler_extension", "entries", "closure.refresh_index"),
            3,
            "closure cadence",
        ),
        (("scheduler_extension", "entries", "unknown"), 0, "keys changed"),
        (
            ("configuration", "admission"),
            "bounded-corrected-real-proof",
            "full production",
        ),
        (("source_checkpoint", "model_time_f64_bits"), "3fe0000000000000", "clock"),
    ],
)
def test_restore_manifest_fails_closed_on_drift(path, value, match) -> None:
    module = _module()
    manifest, layout, product, geopotential, selection = _fixture()
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        module._validate_restore_manifest(
            manifest,
            layout=layout,
            product=product,
            geopotential=geopotential,
            operator_manifest=manifest["operator_manifest"],
            contract_sha256="c" * 64,
            artifact_sha256="b" * 64,
            selection=selection,
        )
