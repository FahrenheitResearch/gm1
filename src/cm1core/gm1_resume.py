"""Strict live-CUDA resume for the Rust-owned GM1 checkpoint envelope.

The durable bytes, carrier layout, source identities, and cadence metadata are
owned and validated by Rust.  This module performs only the live binding that
cannot happen inside the Rust process: copy the validated words to their CUDA
owners, rebuild explicitly recomputable padded/scratch state, restore every
safe-boundary counter, and prove the resulting live carrier hashes before
returning a state that may enter the next complete large step.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cm1core.gm1_checkpoint import (
    GM1_CHECKPOINT_CARRIER_ABI,
    GM1_CHECKPOINT_FIELDS,
    GM1_PHYSICAL_CHECKPOINT_FIELDS,
    GM1_RADIATION_HISTORY_FIELDS,
    GM1_WSM6_ALIASES,
    GM1_WSM6_UNIQUE_FIELDS,
    _executable_path,
    _host_bytes,
    _layout,
    _regular_file,
    _validate_views,
    compose_live_gm1_checkpoint_views,
    restore_gm1_checkpoint_to_views,
)


GM1_LIVE_RESUME_SCHEMA = "gm1.cuda-live-resume/v1"
GM1_RESTORE_DIRECTORY_SCHEMA = "gm1.forecast-restore-directory/v1"
GM1_RESTORE_RECEIPT_SCHEMA = "gm1.forecast-restore-receipt/v1"
GM1_SCHEDULER_EXTENSION_SCHEMA = "gm1.scheduler-extension/v1"
GM1_GEOPOTENTIAL_SOURCE_RESTART_SHA256 = (
    "dfb80b614daf24f547d124d5590d17069f8001d3470e0474a998649e21a8db4a"
)
GM1_F02_ROI_SERIES_RECEIPT_SHA256 = (
    "7c2c5d04f6d3a25a7cb6e9508ba95cba18718fb998cece0c49a1d068de16956e"
)
GM1_F02_ROI_CENTER_LAT_F64_BITS = "4044b28393b636d0"
GM1_F02_ROI_CENTER_LON_F64_BITS = "c05792177a2c3f9c"
GM1_F02_ROI_DOMAIN_GRID_RECEIPT_SHA256 = (
    "96766878a53311f2b00935173cb4a77bb6b95d3d52bd604e69a5e1f5be5f6b7b"
)
GM1_F02_ROI_FIXED_DOMAIN_COMPONENT_SHA256 = (
    "1cf3eca58abcdcbb958bd5bfdbeaa4a684efd900b82e1f27014e018243f2bb88"
)
GM1_RESUME_SOURCE_PROFILES = MappingProxyType(
    {
        "legacy-initial-product-v3": MappingProxyType(
            {
                "geopotential_manifest_sha256": "01c028338ddbfaa001048c2fb230d90585542f4fea798f3a5c4ae1a4eb5bba4b",
                "geopotential_payload_sha256": "deec02bec84da5136eaa23f2fc7780dba04e1140068eede39cd540a04ed83a4b",
                "geopotential_receipt_sha256": "80f3a80632c834d7889ae29a5fcb6e4609bfea9cb329a8702471a042621a04dd",
                "geopotential_source_restart_sha256": GM1_GEOPOTENTIAL_SOURCE_RESTART_SHA256,
                "product_manifest_sha256": "7fd218a005b1c9f01d37b57452b281c2df6c797973268e8fe86b996eb6b61e2e",
                "product_payload_sha256": "8f589cee64b044afcfef34dca6b638c3829946a999537885b1db3576d4ec7981",
                "product_source_artifact_sha256": "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f",
                "product_source_contract_sha256": "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12",
            }
        ),
        "real-hrrr-f02-roi": MappingProxyType(
            {
                "geopotential_manifest_sha256": "da37e1a22f7ef93a4abdd98fa8373fc3d50f26091aeac4812a9c216970433d1a",
                "geopotential_payload_sha256": "74625673f6a3aba254b90eff65dedda0688dd64d62f94eeec8709b365748ae3e",
                "geopotential_receipt_sha256": "09eb31934987b30ff953fb9846a2ab10bf73c8f13452e2612b26a0b2306b64b4",
                "geopotential_source_restart_sha256": "9efe09771763a3f727af715f7dd41a2c6256cd5613aa2f31637cc08c9721ed6e",
                "product_manifest_sha256": "b25de4008d17ddad46b8057f022b50a76e6bff61b537fa9952f68e638adf2567",
                "product_payload_sha256": "ddafeaa8ca523e81925cb4b53e0577963a3e57e06114b6b7df8c6d436e0d68b1",
                "product_source_artifact_sha256": "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d",
                "product_source_contract_sha256": "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896",
            }
        ),
    }
)
GM1_LIVE_RESUME_CLAIM_LIMIT = (
    "exact live-state reconstruction at the production GM1 completed-large-step "
    "boundary and admission to the next large step; no mid-stage restart, "
    "external lateral-boundary driver state, multi-device/MPI state, arbitrary "
    "configuration, forecast skill, tornado prediction, or model validation"
)

_MANIFEST_KEYS = {
    "aliases",
    "capture_phase",
    "carrier_abi",
    "carriers",
    "claim_limit",
    "configuration",
    "dtype",
    "itemsize",
    "model",
    "operator_manifest",
    "order",
    "payload",
    "restore_phase",
    "scheduler_extension",
    "scheduler_scalars",
    "schema",
    "source",
    "source_checkpoint",
    "species_order",
}
_SCHEDULER_SCALAR_KEYS = {
    "completed_large_steps",
    "completed_wsm6_bridge_calls",
    "completed_wsm6_component_calls",
    "completed_wsm6_transitions",
    "large_step_boundary",
    "mass1_f64",
    "moist_buoyancy_applied",
    "moist_post_acoustic_ready",
    "moist_terms_ready",
    "next_stage",
    "post_stage_hook_active",
    "quiescent_capture",
    "radiation_history_initialized",
    "scalar_completed",
    "scalar_next_nrk",
    "solve1_ready",
}
_LEGACY_SCHEDULER_EXTENSION_KEYS = {
    "closure.completed_stage",
    "closure.refresh_index",
    "closure.tke3d_reconstructible_from_tkea",
    "reflectivity.schedule_counter",
}
_F02_ROI_SOURCE_EXTENSION_KEYS = {
    "domain-grid.fixed-component-sha256",
    "domain-grid.imove",
    "domain-grid.roi-receipt-sha256",
    "source.initialization-absolute-offset-s",
    "source.initialization-artifact-sha256",
    "source.initialization-contract-sha256",
    "source.initialization-kind",
    "source.relative-model-time-s",
    "source.roi-center-latitude",
    "source.roi-center-longitude",
    "source.series-receipt-sha256",
    "source.synthetic-translation",
}
_SOURCE_KEYS = {
    "geopotential_manifest_sha256",
    "geopotential_payload_sha256",
    "geopotential_receipt_sha256",
    "geopotential_source_restart_sha256",
    "product_manifest_sha256",
    "product_payload_sha256",
    "product_source_artifact_sha256",
    "product_source_contract_sha256",
}
_SOURCE_CHECKPOINT_KEYS = {
    "artifact_sha256",
    "contract_sha256",
    "dt_f64_bits",
    "model_time_f64_bits",
    "step",
}
_PAYLOAD_KEYS = {"file", "nbytes", "sha256"}
_CONFIGURATION_KEYS = {"admission", "capture_domain", "selection"}
_SELECTION_KEYS = {
    "acoustic_substeps",
    "coriolis",
    "dry_mass_correction",
    "dt_f64_bits",
    "equation_set",
    "grid_shape_zyx",
    "grid_spacing_m_f64_bits",
    "lateral_boundaries",
    "microphysics",
    "moving_frame",
    "psolver",
    "terrain",
}


@dataclass(frozen=True, slots=True)
class RestoredCudaGm1Forecast:
    """One live forecast state plus the runner-owned restored schedule counter."""

    state: object
    reflectivity_schedule_counter: int
    rust_restore: Mapping[str, Any]
    live_carrier_sha256: Mapping[str, str]
    live_arena_sha256: str
    restore_manifest_sha256: str
    source_checkpoint_step: int
    source_checkpoint_time_f64_bits: str
    mass1_f64_bits: str
    claim_limit: str = GM1_LIVE_RESUME_CLAIM_LIMIT

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_limit": self.claim_limit,
            "live_arena_sha256": self.live_arena_sha256,
            "live_carrier_sha256": dict(self.live_carrier_sha256),
            "mass1_f64_bits": self.mass1_f64_bits,
            "model": "GM1",
            "reflectivity_schedule_counter": self.reflectivity_schedule_counter,
            "restore_manifest_sha256": self.restore_manifest_sha256,
            "rust_restore": dict(self.rust_restore),
            "schema": GM1_LIVE_RESUME_SCHEMA,
            "source_checkpoint_step": self.source_checkpoint_step,
            "source_checkpoint_time_f64_bits": self.source_checkpoint_time_f64_bits,
            "status": "ready-before-next-large-step",
        }


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} keys changed: {observed}")
    return value


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a strict nonnegative integer")
    return value


def _strict_positive_int(value: object, label: str) -> int:
    result = _strict_nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be a strict positive integer")
    return result


def _f64_from_bits(value: object, label: str) -> float:
    if (
        not isinstance(value, str)
        or len(value) != 16
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 16 lowercase hexadecimal characters")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(result):
        raise ValueError(f"{label} must encode a finite binary64 value")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _expected_selection(selection: object) -> dict[str, Any]:
    return {
        "acoustic_substeps": selection.acoustic_substeps,
        "coriolis": selection.coriolis,
        "dry_mass_correction": selection.dry_mass_correction,
        "dt_f64_bits": struct.pack(">d", float(selection.dt_s)).hex(),
        "equation_set": selection.equation_set,
        "grid_shape_zyx": [selection.nz, selection.ny, selection.nx],
        "grid_spacing_m_f64_bits": [
            struct.pack(">d", float(value)).hex()
            for value in (selection.dx_m, selection.dy_m, selection.dz_m)
        ],
        "lateral_boundaries": list(selection.lateral_boundaries),
        "microphysics": selection.microphysics,
        "moving_frame": selection.moving_frame,
        "psolver": selection.psolver,
        "terrain": selection.terrain,
    }


def _resume_source_profile(
    product: object, geopotential: object
) -> tuple[str, Mapping[str, str]]:
    observed = {
        "geopotential_manifest_sha256": geopotential.manifest_sha256,
        "geopotential_payload_sha256": geopotential.payload_sha256,
        "geopotential_receipt_sha256": geopotential.receipt_sha256,
        "product_manifest_sha256": product.export.manifest_sha256,
        "product_payload_sha256": product.export.payload_sha256,
        "product_source_artifact_sha256": product.export.source_artifact_sha256,
        "product_source_contract_sha256": product.export.source_contract_sha256,
    }
    matches = [
        (name, profile)
        for name, profile in GM1_RESUME_SOURCE_PROFILES.items()
        if all(observed[key] == profile[key] for key in observed)
    ]
    if len(matches) != 1:
        raise ValueError(
            "GM1 live resume source arenas are not one exact sealed source profile"
        )
    return matches[0]


def _validate_restore_manifest(
    manifest: object,
    *,
    layout: Mapping[str, Any],
    product: object,
    geopotential: object,
    operator_manifest: Mapping[str, Any],
    contract_sha256: str,
    artifact_sha256: str,
    selection: object,
) -> dict[str, Any]:
    value = _exact_keys(manifest, _MANIFEST_KEYS, "GM1 restore manifest")
    if (
        value["schema"] != GM1_RESTORE_DIRECTORY_SCHEMA
        or value["model"] != "GM1"
        or value["capture_phase"] != "post-large-step"
        or value["restore_phase"] != "before-next-large-step"
        or value["carrier_abi"] != GM1_CHECKPOINT_CARRIER_ABI
        or value["dtype"] != "<f4"
        or value["itemsize"] != 4
        or value["order"] != "C"
        or value["species_order"] != ["qv", "qc", "qr", "qi", "qs", "qg"]
    ):
        raise ValueError("GM1 restore manifest semantic identity changed")

    expected_aliases = [
        {
            "alias": alias,
            "canonical": canonical,
            "relation": "same live allocation and exact byte span",
        }
        for alias, canonical in GM1_WSM6_ALIASES.items()
    ]
    if value["aliases"] != expected_aliases:
        raise ValueError("GM1 restore alias contract changed")

    carriers = value["carriers"]
    if not isinstance(carriers, list) or len(carriers) != len(GM1_CHECKPOINT_FIELDS):
        raise ValueError("GM1 restore carrier inventory changed")
    layout_rows = layout["carriers"]
    for restored, expected in zip(carriers, layout_rows, strict=True):
        restored = _exact_keys(
            restored,
            {
                "dtype",
                "itemsize",
                "name",
                "nbytes",
                "offset_nbytes",
                "order",
                "sha256",
                "shape",
            },
            "GM1 restored carrier",
        )
        if any(
            restored[key] != expected[key]
            for key in ("name", "nbytes", "offset_nbytes", "shape")
        ):
            raise ValueError(
                f"GM1 restored carrier layout changed for {restored['name']}"
            )
        if (
            restored["dtype"] != "<f4"
            or restored["itemsize"] != 4
            or restored["order"] != "C"
        ):
            raise ValueError(
                f"GM1 restored carrier encoding changed for {restored['name']}"
            )
        _sha256(restored["sha256"], f"{restored['name']} SHA-256")

    payload = _exact_keys(value["payload"], _PAYLOAD_KEYS, "GM1 restored payload")
    if (
        payload["file"] != "gm1-carriers.f32le"
        or payload["nbytes"] != layout["payload_nbytes"]
    ):
        raise ValueError("GM1 restored payload layout changed")
    _sha256(payload["sha256"], "GM1 restored payload SHA-256")

    configuration = _exact_keys(
        value["configuration"], _CONFIGURATION_KEYS, "GM1 restore configuration"
    )
    if configuration["admission"] != "production-full-domain":
        raise ValueError("live GM1 resume admits only the full production domain")
    expected_domain = {
        "crop_origin_yx": [0, 0],
        "crop_shape_zyx": [selection.nz, selection.ny, selection.nx],
        "source_shape_zyx": [selection.nz, selection.ny, selection.nx],
    }
    if configuration["capture_domain"] != expected_domain:
        raise ValueError("GM1 restore domain differs from the selected live state")
    restored_selection = _exact_keys(
        configuration["selection"], _SELECTION_KEYS, "GM1 restore selection"
    )
    if restored_selection != _expected_selection(selection):
        raise ValueError(
            "GM1 restore selection differs from the live forecast selection"
        )

    source_profile_name, expected_source = _resume_source_profile(product, geopotential)
    source = _exact_keys(value["source"], _SOURCE_KEYS, "GM1 restore source")
    if source != dict(expected_source):
        raise ValueError("GM1 restore source identity differs from the live arenas")
    if value["operator_manifest"] != operator_manifest:
        raise ValueError(
            "GM1 restore operator manifest differs from the independent input"
        )

    source_checkpoint = _exact_keys(
        value["source_checkpoint"],
        _SOURCE_CHECKPOINT_KEYS,
        "GM1 source checkpoint",
    )
    if (
        source_checkpoint["contract_sha256"] != contract_sha256
        or source_checkpoint["artifact_sha256"] != artifact_sha256
    ):
        raise ValueError("GM1 restore source checkpoint identity changed")
    step = _strict_positive_int(source_checkpoint["step"], "checkpoint step")
    model_time = _f64_from_bits(
        source_checkpoint["model_time_f64_bits"], "checkpoint model time"
    )
    dt = _f64_from_bits(source_checkpoint["dt_f64_bits"], "checkpoint dt")
    if dt != float(selection.dt_s) or model_time != float(step * selection.dt_s):
        raise ValueError("GM1 restore clock differs from the selected step cadence")

    scheduler = _exact_keys(
        value["scheduler_scalars"], _SCHEDULER_SCALAR_KEYS, "GM1 scheduler scalars"
    )
    completed = _strict_positive_int(
        scheduler["completed_wsm6_transitions"], "completed WSM6 transitions"
    )
    expected_scalars = {
        "completed_large_steps": step,
        "completed_wsm6_bridge_calls": completed,
        "completed_wsm6_component_calls": completed,
        "completed_wsm6_transitions": completed,
        "large_step_boundary": True,
        "mass1_f64": scheduler["mass1_f64"],
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
    }
    if scheduler != expected_scalars or completed != step:
        raise ValueError("GM1 restored safe-boundary scheduler cadence changed")
    mass1 = _f64_from_bits(scheduler["mass1_f64"], "GM1 mass1")
    if mass1 <= 0.0:
        raise ValueError("GM1 restored mass ledger must be positive")

    extension = _exact_keys(
        value["scheduler_extension"],
        {"entries", "schema"},
        "GM1 scheduler extension",
    )
    if extension["schema"] != GM1_SCHEDULER_EXTENSION_SCHEMA:
        raise ValueError("GM1 scheduler extension schema changed")
    extension_keys = set(_LEGACY_SCHEDULER_EXTENSION_KEYS)
    if source_profile_name == "real-hrrr-f02-roi":
        extension_keys.update(_F02_ROI_SOURCE_EXTENSION_KEYS)
    entries = _exact_keys(
        extension["entries"], extension_keys, "GM1 scheduler extension entries"
    )
    refresh_index = _strict_positive_int(
        entries["closure.refresh_index"], "closure refresh index"
    )
    reflectivity_counter = _strict_nonnegative_int(
        entries["reflectivity.schedule_counter"], "reflectivity schedule counter"
    )
    if (
        entries["closure.completed_stage"] != 0
        or entries["closure.tke3d_reconstructible_from_tkea"] is not True
        or refresh_index != step + 1
    ):
        raise ValueError("GM1 restored closure cadence changed")
    if source_profile_name == "real-hrrr-f02-roi":
        expected_f02_entries = {
            "domain-grid.fixed-component-sha256": (
                GM1_F02_ROI_FIXED_DOMAIN_COMPONENT_SHA256
            ),
            "domain-grid.imove": 0,
            "domain-grid.roi-receipt-sha256": (GM1_F02_ROI_DOMAIN_GRID_RECEIPT_SHA256),
            "source.initialization-absolute-offset-s": 7200.0,
            "source.initialization-artifact-sha256": expected_source[
                "product_source_artifact_sha256"
            ],
            "source.initialization-contract-sha256": expected_source[
                "product_source_contract_sha256"
            ],
            "source.initialization-kind": "boundary-analysis-history-f02-roi",
            "source.relative-model-time-s": model_time,
            "source.roi-center-latitude": _f64_from_bits(
                GM1_F02_ROI_CENTER_LAT_F64_BITS, "f02 ROI center latitude"
            ),
            "source.roi-center-longitude": _f64_from_bits(
                GM1_F02_ROI_CENTER_LON_F64_BITS, "f02 ROI center longitude"
            ),
            "source.series-receipt-sha256": GM1_F02_ROI_SERIES_RECEIPT_SHA256,
            "source.synthetic-translation": False,
        }
        if any(
            entries[key] != expected for key, expected in expected_f02_entries.items()
        ):
            raise ValueError("GM1 restored f02 ROI source scheduler identity changed")

    return {
        "completed_transitions": completed,
        "mass1_f64": mass1,
        "mass1_f64_bits": scheduler["mass1_f64"],
        "model_time_f64_bits": source_checkpoint["model_time_f64_bits"],
        "model_time_s": model_time,
        "payload_sha256": payload["sha256"],
        "reflectivity_schedule_counter": reflectivity_counter,
        "refresh_index": refresh_index,
        "step": step,
    }


def _allocate_restore_views(
    product: object, layout: Mapping[str, Any]
) -> tuple[dict[str, object], dict[str, object]]:
    cupy = importlib.import_module("cupy")
    carriers: dict[str, object] = {}
    for spec in layout["carriers"]:
        name = spec["name"]
        if name in GM1_PHYSICAL_CHECKPOINT_FIELDS:
            value = product.field(name)
        else:
            value = cupy.empty(tuple(spec["shape"]), dtype=cupy.float32, order="C")
        carriers[name] = value
    aliases = {
        alias: carriers[canonical] for alias, canonical in GM1_WSM6_ALIASES.items()
    }
    _validate_views(carriers, aliases, layout)
    return carriers, aliases


def _manifest_operator(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError("GM1 operator manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("GM1 operator manifest must be a JSON object")
    return value


def _live_hashes(
    carriers: Mapping[str, object], manifest: Mapping[str, Any]
) -> tuple[dict[str, str], str]:
    digest = hashlib.sha256()
    hashes: dict[str, str] = {}
    for spec in manifest["carriers"]:
        name = spec["name"]
        payload = _host_bytes(carriers[name], name)
        observed = hashlib.sha256(payload).hexdigest()
        if observed != spec["sha256"]:
            raise ValueError(f"live GM1 restore changed raw words for {name}")
        hashes[name] = observed
        digest.update(payload)
    aggregate = digest.hexdigest()
    if aggregate != manifest["payload"]["sha256"]:
        raise ValueError("live GM1 restore changed the aggregate raw arena")
    return hashes, aggregate


def restore_cuda_gm1_forecast_state(
    checkpoint: Path | str,
    product: object,
    geopotential: object,
    operator_manifest: Path | str,
    output_directory: Path | str,
    *,
    contract_sha256: str,
    artifact_sha256: str,
    executable: Path | str | None = None,
    selection: object | None = None,
    source_identities: Path | str | None = None,
) -> RestoredCudaGm1Forecast:
    """Restore and bind a production checkpoint before the next large step.

    ``product`` and ``geopotential`` must be freshly initialized from the exact
    hash-bound source arenas named by the checkpoint.  The function returns
    only after every persistent carrier has matched its Rust-published SHA-256
    in live device memory and all recomputable state has been rebuilt.
    """

    contract_sha256 = _sha256(contract_sha256, "checkpoint contract SHA-256")
    artifact_sha256 = _sha256(artifact_sha256, "checkpoint artifact SHA-256")
    executable_path = _executable_path(executable)
    operator_path = _regular_file(operator_manifest, "GM1 operator manifest")
    operator_value = _manifest_operator(operator_path)
    layout = _layout(executable_path, operator_path)

    forecast = importlib.import_module("cm1core.cuda_gm1_forecast")
    transition_module = importlib.import_module(
        "cm1core.initialized_product_wsm6_transition"
    )
    closure_module = importlib.import_module("cm1core.cuda_gm1_closure")
    initialized_module = importlib.import_module("cm1core.cuda_initialized_product")
    if not isinstance(product, initialized_module.CudaInitializedProduct):
        raise TypeError("product must be a fresh CudaInitializedProduct")
    if not isinstance(geopotential, transition_module.CudaWsm6Geopotential):
        raise TypeError("geopotential must be a fresh CudaWsm6Geopotential")
    if (
        geopotential.source_product_manifest_sha256 != product.export.manifest_sha256
        or geopotential.source_product_payload_sha256 != product.export.payload_sha256
    ):
        raise ValueError("geopotential does not belong to the fresh product arena")
    if (
        tuple(product.export.crop_origin_yx) != (0, 0)
        or not product.export.is_full_domain
    ):
        raise ValueError("live GM1 resume admits only the full production product")
    selection = forecast.DEFAULT_GM1_SELECTION if selection is None else selection
    if not isinstance(selection, forecast.Gm1ForecastSelection):
        raise TypeError("selection must be a Gm1ForecastSelection")

    cupy = importlib.import_module("cupy")
    current_device = int(cupy.cuda.runtime.getDevice())
    current_stream = cupy.cuda.get_current_stream()
    if (
        current_device != product.device_id
        or current_device != geopotential.device_id
        or int(current_stream.ptr) != product.stream_pointer
        or int(current_stream.ptr) != geopotential.stream_pointer
    ):
        raise ValueError("GM1 resume requires the fresh source-owning CUDA stream")

    with product.stream:
        carriers, aliases = _allocate_restore_views(product, layout)
        rust_restore = restore_gm1_checkpoint_to_views(
            checkpoint,
            carriers,
            aliases,
            operator_path,
            output_directory,
            contract_sha256=contract_sha256,
            artifact_sha256=artifact_sha256,
            executable=executable_path,
            source_identities=source_identities,
        )
    output = Path(output_directory)
    manifest_path = _regular_file(output / "manifest.json", "GM1 restore manifest")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ValueError("Rust GM1 restore manifest is not valid JSON") from error
    metadata = _validate_restore_manifest(
        manifest,
        layout=layout,
        product=product,
        geopotential=geopotential,
        operator_manifest=operator_value,
        contract_sha256=contract_sha256,
        artifact_sha256=artifact_sha256,
        selection=selection,
    )
    if (
        not isinstance(rust_restore, dict)
        or rust_restore.get("schema") != GM1_RESTORE_RECEIPT_SCHEMA
        or rust_restore.get("model") != "GM1"
        or rust_restore.get("checkpoint_contract_sha256") != contract_sha256
        or rust_restore.get("checkpoint_artifact_sha256") != artifact_sha256
        or rust_restore.get("payload_sha256") != metadata["payload_sha256"]
        or rust_restore.get("step") != metadata["step"]
    ):
        raise ValueError("Rust GM1 restore receipt differs from its restored manifest")

    state = forecast.create_cuda_gm1_forecast_state(
        product, geopotential, selection=selection
    )
    wsm_arrays: dict[str, object] = {}
    for name in transition_module.PRODUCT_WSM6_RESTART_FIELDS:
        if name in GM1_WSM6_ALIASES:
            wsm_arrays[name] = carriers[GM1_WSM6_ALIASES[name]]
        elif name in GM1_WSM6_UNIQUE_FIELDS:
            wsm_arrays[name] = carriers[name]
        else:
            raise RuntimeError(f"unbound WSM6 checkpoint carrier {name}")
    restart = transition_module.ProductWsm6Restart(
        schema=transition_module.INITIALIZED_PRODUCT_WSM6_RESTART_SCHEMA,
        transition_abi=transition_module.INITIALIZED_PRODUCT_WSM6_TRANSITION_ABI,
        source_product_manifest_sha256=product.export.manifest_sha256,
        source_product_payload_sha256=product.export.payload_sha256,
        completed_transitions=metadata["completed_transitions"],
        arrays=MappingProxyType(wsm_arrays),
        durable_checkpoint_artifact_sha256=artifact_sha256,
        durable_restore_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        durable_payload_sha256=metadata["payload_sha256"],
    )
    with state.stream:
        state.transition = transition_module.initialize_product_wsm6_transition(
            product, geopotential, restart=restart
        )
        for name in GM1_RADIATION_HISTORY_FIELDS:
            cupy.copyto(state.prepared.arrays[name], carriers[name])
        state.closure.ready_event.synchronize()
        state.closure.retained.clear()
        cupy.copyto(state.closure.sgs["tkea"], carriers["closure.tkea"])
        state.closure.completed_stage = 3
        state.closure.refresh_index = metadata["refresh_index"] - 1
        closure_module.refresh_cuda_gm1_closure_for_next_step(state.closure)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)

    state.mass1_f64 = metadata["mass1_f64"]
    state.step_index = metadata["step"]
    state.time_seconds = metadata["model_time_s"]
    state.next_stage = 1
    state.solve1_ready = False
    state.ready_event = event
    state.ready_event.synchronize()
    rebound, rebound_aliases, _ = compose_live_gm1_checkpoint_views(state)
    _validate_views(rebound, rebound_aliases, layout)
    live_hashes, live_arena_sha256 = _live_hashes(rebound, manifest)
    restore_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    result = RestoredCudaGm1Forecast(
        state=state,
        reflectivity_schedule_counter=metadata["reflectivity_schedule_counter"],
        rust_restore=MappingProxyType(dict(rust_restore)),
        live_carrier_sha256=MappingProxyType(live_hashes),
        live_arena_sha256=live_arena_sha256,
        restore_manifest_sha256=restore_manifest_sha256,
        source_checkpoint_step=metadata["step"],
        source_checkpoint_time_f64_bits=metadata["model_time_f64_bits"],
        mass1_f64_bits=metadata["mass1_f64_bits"],
    )
    json.dumps(result.as_dict(), allow_nan=False, sort_keys=True)
    return result


__all__ = [
    "GM1_LIVE_RESUME_CLAIM_LIMIT",
    "GM1_LIVE_RESUME_SCHEMA",
    "RestoredCudaGm1Forecast",
    "restore_cuda_gm1_forecast_state",
]
