"""Fail-closed admission for a real source-coordinate GM1 50 m child.

Rust owns checkpoint decoding and all future atmospheric remapping.  This
module validates the sealed parent-to-child plan and the independent official
15 km ``stretch_z=1`` grid proof while CUDA is cold.  It deliberately refuses
numerical child construction until a later producer receipt binds terrain
composition, hydrostatic/base-state reconstruction, and open-boundary state.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn

from cm1core.errors import refuse


GM1_CHILD_PLAN_SCHEMA = "gm1.source-coordinate-50m-child-remap-plan/v1"
GM1_CHILD_PLAN_SHA256 = (
    "14609494f7c87937fcb5ec0db3f214809ed514380560c43beb18183e6eebce17"
)
GM1_CHILD_PARENT_CHECKPOINT_SHA256 = (
    "4f3a00ae66e0b68f0f05f4eff54a3625e846c0efee31b3ecbdee01c3c87ac1b4"
)
GM1_CHILD_PARENT_DIAGNOSTICS_SHA256 = (
    "920af2c520e6b043a140c27583a2c98d257cf146218a6f934236830b1464e818"
)
GM1_CHILD_PARENT_PDEFQ_SHA256 = (
    "dddeef7c35747a4c2a8fef3791821bdb937bbfc603e17c36cd6f43386245eff2"
)
GM1_CHILD_FLAT_GRID_RECEIPT_SHA256 = (
    "ce75a4b2258b0b05b56cbc7834f22489c26defab285c29e08281994c6f453c0b"
)
GM1_CHILD_FLAT_GRID_SHA256 = (
    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
)
GM1_CHILD_FLAT_GRID_PROOF_SHA256 = (
    "277f2b8831f316b2e99a81a903f5ac20ee5d781e2c2b058963539f648f9ec02e"
)
GM1_CHILD_PRODUCER_REQUIREMENTS_SCHEMA = (
    "gm1.source-coordinate-50m-child-producer-requirements/v1"
)
GM1_CHILD_PRODUCER_REQUIREMENTS_SHA256 = (
    "439b55b69f0fbe85901a85afdc1439d1317499140f8e5660e60a27f4a23f94a1"
)
GM1_CHILD_TERRAIN_OPERATOR_GATE_SHA256 = (
    "9c99ba42f47363329323cd5ac088f988a7a8380b60712ad9f00d176caa390515"
)
GM1_CHILD_TERRAIN_OPERATOR_SOURCE_SHA256 = (
    "905bec56020e2bf1a73c0c854a9d5f462ccc372334ab3ceee2ad50f041be2f18"
)
GM1_CHILD_CLAIM_LIMIT = (
    "strict CUDA-cold admission of one real pdefq-clean T+300 parent binding, "
    "source-coordinate 300x300x75 50 m child geometry, carrier dispositions, "
    "and the independent official flat 75-level/15-km stretch_z=1 proof only; "
    "no numerical remap, terrain/stretch composition, hydrostatic rebalance, "
    "boundary initialization, child state, trajectory, forecast, or tornado claim"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REMAP_3D = (
    "closure.tkea",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.h_diabatic",
    "held.qcten",
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "pp3d",
    "ppi",
    "ppx",
    "q3d",
    "qa",
    "th3d",
    "tha",
    "u3d",
    "ua",
    "v3d",
    "va",
    "w3d",
    "wa",
)
_REMAP_SURFACE = (
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_sr",
)
_RECOMPUTE = ("prs", "rf", "rho", "rr")
_BOUNDARY_INITIALIZE = ("radbce", "radbcn", "radbcs", "radbcw")
_PROFILE_NAMES = ("mf", "mh", "rmf", "rmh", "sigma", "sigmaf", "zf", "zh")
_TERRAIN_STRETCH_CARRIERS = (
    "terrain_msl",
    "zh",
    "zf",
    "mh",
    "rmh",
    "mf",
    "rmf",
    "c1",
    "c2",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "gx",
    "gxu",
    "gy",
    "gyv",
    "rds",
    "rdsf",
)
_HYDROSTATIC_OPEN_CARRIERS = (
    "pi0",
    "prs0",
    "th0",
    "thv0",
    "rth0",
    "qv0",
    "rho0",
    "rr0",
    "rf0",
    "rrf0",
    "rho0s",
    "pi0s",
    "prs0s",
    "rth0s",
    "terrain_msl_halo",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "ppi",
    "pp3d",
    "ppx",
    "piadv",
    "ppterm",
    "ppten",
    "thv",
    "tha",
    "th3d",
    "thten",
    "thten1",
    "ua",
    "u3d",
    "uten",
    "uten1",
    "va",
    "v3d",
    "vten",
    "vten1",
    "wa",
    "w3d",
    "wten",
    "wten1",
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
)


@dataclass(frozen=True, slots=True)
class Gm1ChildInitializationPlan:
    plan_path: Path
    plan_sha256: str
    flat_grid_receipt_path: Path
    flat_grid_receipt_sha256: str
    receipt: Mapping[str, Any]
    remap_3d_carriers: tuple[str, ...]
    remap_surface_carriers: tuple[str, ...]
    recompute_carriers: tuple[str, ...]
    boundary_initialize_carriers: tuple[str, ...]
    numerical_child_ready: bool = False


@dataclass(frozen=True, slots=True)
class Gm1ChildProducerRequirements:
    path: Path
    sha256: str
    parent_plan_sha256: str
    receipt: Mapping[str, Any]
    terrain_stretch_carriers: tuple[str, ...]
    hydrostatic_open_boundary_carriers: tuple[str, ...]
    numerical_child_output_allowed: bool = False


@dataclass(frozen=True, slots=True)
class Gm1ChildTerrainOperatorGate:
    path: Path
    sha256: str
    requirements_sha256: str
    receipt: Mapping[str, Any]
    carrier_names: tuple[str, ...]
    real_child_terrain_ready: bool = False


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    result = _object(value, label)
    actual = set(result)
    if actual != expected:
        raise ValueError(
            f"{label} keys changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return result


def _read_sealed_json(
    path: str | Path,
    *,
    expected_sha256: str,
    label: str,
    require_canonical: bool = True,
) -> tuple[Path, Mapping[str, Any]]:
    expected_sha256 = _sha256(expected_sha256, f"expected {label} SHA-256")
    source = Path(path)
    resolved = source.resolve(strict=True)
    if source.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = resolved.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} differs from independent SHA-256: {actual}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    canonical = (
        json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
        + b"\n"
    )
    if require_canonical and raw != canonical:
        raise ValueError(f"{label} must use canonical JSON plus one newline")
    return resolved, _object(value, label)


def _validate_carrier_disposition(value: object) -> None:
    disposition = _exact_keys(
        value,
        {
            "child_boundary_initialize",
            "horizontal_and_physical_height_remap_after_vertical_admission",
            "horizontal_surface_remap",
            "recompute_after_remap",
            "source_carrier_count",
        },
        "child carrier disposition",
    )
    expected = {
        "horizontal_and_physical_height_remap_after_vertical_admission": _REMAP_3D,
        "horizontal_surface_remap": _REMAP_SURFACE,
        "recompute_after_remap": _RECOMPUTE,
        "child_boundary_initialize": _BOUNDARY_INITIALIZE,
    }
    inventory: list[str] = []
    for name, wanted in expected.items():
        observed = disposition[name]
        if not isinstance(observed, list) or tuple(observed) != wanted:
            raise ValueError(f"child carrier disposition {name} changed")
        inventory.extend(observed)
    if disposition["source_carrier_count"] != 37 or len(set(inventory)) != 37:
        raise ValueError(
            "child carrier disposition no longer covers 37 unique carriers"
        )


def _validate_parent(value: object) -> None:
    parent = _exact_keys(
        value,
        {"checkpoint", "clock", "diagnostics", "model", "pdefq", "schema", "status"},
        "child parent binding",
    )
    checkpoint = _exact_keys(
        parent["checkpoint"],
        {
            "artifact_sha256",
            "contract_sha256",
            "operator_manifest_sha256",
            "product_manifest_sha256",
            "raw_arena_sha256",
        },
        "child parent checkpoint",
    )
    diagnostics = _object(parent["diagnostics"], "child parent diagnostics")
    pdefq = _exact_keys(
        parent["pdefq"],
        {"qa_q3d_nonnegative", "qa_q3d_raw_bit_equal", "receipt_sha256", "timing"},
        "child parent pdefq",
    )
    if (
        parent["model"] != "GM1"
        or parent["schema"] != "gm1.pdefq-clean-t300-parent-binding/v1"
        or parent["status"] != "PASS_PDEFQ_CLEAN_T300"
        or checkpoint["artifact_sha256"] != GM1_CHILD_PARENT_CHECKPOINT_SHA256
        or diagnostics.get("receipt_sha256") != GM1_CHILD_PARENT_DIAGNOSTICS_SHA256
        or pdefq["receipt_sha256"] != GM1_CHILD_PARENT_PDEFQ_SHA256
        or pdefq["qa_q3d_nonnegative"] is not True
        or pdefq["qa_q3d_raw_bit_equal"] is not True
        or pdefq["timing"] != "q3d repaired before WSM6; solve3 copies q3d to qa"
    ):
        raise ValueError("child parent checkpoint/pdefq identity changed")
    clock = _exact_keys(
        parent["clock"], {"dt_f64_bits", "model_time_f64_bits", "step"}, "child clock"
    )
    if clock != {
        "dt_f64_bits": "3fe0000000000000",
        "model_time_f64_bits": "4072c00000000000",
        "step": 600,
    }:
        raise ValueError("child parent must be the completed T+300 large-step boundary")


def _validate_geometry(value: object) -> None:
    geometry = _exact_keys(value, {"child", "parent", "selection"}, "child geometry")
    child = _object(geometry["child"], "child grid")
    parent = _object(geometry["parent"], "parent grid")
    selection = _object(geometry["selection"], "child grid selection")
    if (
        (child.get("nx"), child.get("ny"), child.get("nz")) != (300, 300, 75)
        or child.get("dx_m_f64_bits") != "4049000000000000"
        or child.get("dy_m_f64_bits") != "4049000000000000"
        or (parent.get("nx"), parent.get("ny"), parent.get("nz")) != (300, 300, 75)
        or parent.get("dx_m_f64_bits") != "4059000000000000"
        or parent.get("dy_m_f64_bits") != "4059000000000000"
        or selection.get("child_center_parent_mass_i") != 224
        or selection.get("child_center_parent_mass_j") != 224
        or selection.get("nominal_parent_origin_i") != 149
        or selection.get("nominal_parent_origin_j") != 149
        or selection.get("nominal_parent_count_x") != 150
        or selection.get("nominal_parent_count_y") != 150
        or selection.get("synthetic_translation") is not False
    ):
        raise ValueError("source-coordinate 50 m child geometry changed")


def _validate_flat_grid(plan_vertical: object, gate: Mapping[str, Any]) -> None:
    vertical = _exact_keys(
        plan_vertical,
        {
            "abi",
            "derived",
            "extrapolation_allowed",
            "hashes",
            "parameters",
            "proof",
            "schema",
            "status",
        },
        "child vertical gate",
    )
    pins = _object(gate.get("pins"), "flat child grid pins")
    record = _object(gate.get("record"), "flat child grid record")
    if (
        gate.get("schema") != "gpuwm-cm1.stretch-z1-child-15km-cuda-gate/v1"
        or gate.get("status") != "PASS_NATIVE_CPU_CUDA_RAW_BIT_EXACT"
        or gate.get("cpu_cuda_raw_bit", {}).get("passed") is not True
        or pins.get("grid_sha256") != GM1_CHILD_FLAT_GRID_SHA256
        or pins.get("proof_sha256") != GM1_CHILD_FLAT_GRID_PROOF_SHA256
        or record.get("schema") != "gpuwm-cm1.stretch-z1-child-15km/v1"
        or record.get("abi") != "gpuwm-cm1.stretch-z1-child-15km-profiles-f32/v1"
        or vertical["schema"] != "gpuwm-cm1.stretch-z1-child-15km/v1"
        or vertical["status"] != "PINNED_FLAT_CPU_CUDA_BLOCKED_ON_TERRAIN_COMPOSITION"
        or vertical["extrapolation_allowed"] is not False
        or _object(vertical["hashes"], "child vertical hashes").get(
            "combined_grid_sha256"
        )
        != GM1_CHILD_FLAT_GRID_SHA256
        or _object(vertical["proof"], "child vertical proof").get("sha256")
        != GM1_CHILD_FLAT_GRID_PROOF_SHA256
        or _object(vertical["proof"], "child vertical proof").get("cuda_receipt_sha256")
        != GM1_CHILD_FLAT_GRID_RECEIPT_SHA256
    ):
        raise ValueError("official 75-level/15-km child grid proof changed")
    parameters = _object(vertical["parameters"], "child vertical parameters")
    if parameters != {
        "dz_bot_f32_bits": "41a00000",
        "dz_f32_bits": "42c80000",
        "dz_top_f32_bits": "43a00000",
        "nk": 75,
        "str_bot_f32_bits": "43480000",
        "str_top_f32_bits": "45dac000",
        "ztop_f32_bits": "466a6000",
    }:
        raise ValueError("official child stretch_z=1 parameters changed")
    plan_hashes = _object(vertical["hashes"], "child vertical hashes")
    proof_profiles = _object(pins.get("profile_sha256"), "flat grid profile pins")
    for name in _PROFILE_NAMES:
        if plan_hashes.get(f"{name}_sha256") != proof_profiles.get(name):
            raise ValueError(
                f"child {name} profile hash differs across Rust/CM1 proofs"
            )


def load_gm1_child_initialization_plan(
    plan_path: str | Path,
    flat_grid_receipt_path: str | Path,
    *,
    expected_plan_sha256: str = GM1_CHILD_PLAN_SHA256,
    expected_flat_grid_receipt_sha256: str = GM1_CHILD_FLAT_GRID_RECEIPT_SHA256,
) -> Gm1ChildInitializationPlan:
    """Validate the exact real parent/child plan without importing CUDA."""

    resolved_plan, plan = _read_sealed_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        label="GM1 child remap plan",
    )
    resolved_grid, flat_grid = _read_sealed_json(
        flat_grid_receipt_path,
        expected_sha256=expected_flat_grid_receipt_sha256,
        label="GM1 flat child grid receipt",
    )
    plan = _exact_keys(
        plan,
        {
            "authority",
            "builder_abi",
            "carrier_disposition",
            "claim_limit",
            "geometry",
            "horizontal_mapping",
            "model",
            "parent",
            "schema",
            "status",
            "terrain_domain",
            "vertical_gate",
        },
        "GM1 child remap plan",
    )
    authority = _object(plan["authority"], "child authority")
    terrain = _object(plan["terrain_domain"], "child terrain domain")
    if (
        plan["schema"] != GM1_CHILD_PLAN_SCHEMA
        or plan["status"] != "PASS_PARENT_CHECKPOINT_AWAITING_CHILD_PRODUCERS"
        or plan["model"] != "GM1"
        or plan["builder_abi"] != "gm1.t300-child-remap-builder/v1"
        or authority.get("model") != "GM1"
        or authority.get("model_source") != "CM1 r21.1"
        or "no synthetic data or field translation"
        not in str(authority.get("source_selection"))
        or terrain.get("terrain") is not True
        or terrain.get("imove") != 0
        or terrain.get("mode") != "fixed"
    ):
        raise ValueError("GM1 child source/terrain admission changed")
    _validate_carrier_disposition(plan["carrier_disposition"])
    _validate_parent(plan["parent"])
    _validate_geometry(plan["geometry"])
    _validate_flat_grid(plan["vertical_gate"], flat_grid)
    return Gm1ChildInitializationPlan(
        plan_path=resolved_plan,
        plan_sha256=expected_plan_sha256,
        flat_grid_receipt_path=resolved_grid,
        flat_grid_receipt_sha256=expected_flat_grid_receipt_sha256,
        receipt=MappingProxyType(dict(plan)),
        remap_3d_carriers=_REMAP_3D,
        remap_surface_carriers=_REMAP_SURFACE,
        recompute_carriers=_RECOMPUTE,
        boundary_initialize_carriers=_BOUNDARY_INITIALIZE,
    )


def _required_pending(value: Mapping[str, Any], label: str) -> None:
    if value.get("status") != "REQUIRED_NOT_YET_BOUND":
        raise ValueError(f"{label} must remain explicitly unbound in requirements")


def load_gm1_child_producer_requirements(
    plan: Gm1ChildInitializationPlan,
    requirements_path: str | Path,
    *,
    expected_sha256: str = GM1_CHILD_PRODUCER_REQUIREMENTS_SHA256,
) -> Gm1ChildProducerRequirements:
    """Validate Rust's non-numerical producer contract for the admitted plan."""

    if not isinstance(plan, Gm1ChildInitializationPlan):
        raise TypeError("plan must be a Gm1ChildInitializationPlan")
    resolved, receipt = _read_sealed_json(
        requirements_path,
        expected_sha256=expected_sha256,
        label="GM1 child producer requirements",
    )
    receipt = _exact_keys(
        receipt,
        {
            "authority",
            "child_identity",
            "claim_limit",
            "gate_abi",
            "model",
            "numerical_child_output_allowed",
            "parent_plan",
            "producer_requirements",
            "schema",
            "status",
        },
        "GM1 child producer requirements",
    )
    child = _exact_keys(
        receipt["child_identity"],
        {
            "dx_m_f64_bits",
            "dy_m_f64_bits",
            "imove",
            "nx",
            "ny",
            "nz",
            "synthetic_translation",
            "terrain",
            "ztop_m_f64_bits",
        },
        "producer child identity",
    )
    parent = _exact_keys(
        receipt["parent_plan"],
        {
            "checkpoint_artifact_sha256",
            "diagnostics_receipt_sha256",
            "pdefq_receipt_sha256",
            "schema",
            "sha256",
        },
        "producer parent plan",
    )
    if (
        receipt["schema"] != GM1_CHILD_PRODUCER_REQUIREMENTS_SCHEMA
        or receipt["status"]
        != "PASS_PARENT_AND_REQUIREMENTS_BLOCKED_ON_NUMERICAL_PRODUCERS"
        or receipt["gate_abi"] != "gm1.t300-child-producer-gate/v1"
        or receipt["model"] != "GM1"
        or receipt["numerical_child_output_allowed"] is not False
        or child
        != {
            "dx_m_f64_bits": "4049000000000000",
            "dy_m_f64_bits": "4049000000000000",
            "imove": 0,
            "nx": 300,
            "ny": 300,
            "nz": 75,
            "synthetic_translation": False,
            "terrain": True,
            "ztop_m_f64_bits": "40cd4c0000000000",
        }
        or parent
        != {
            "checkpoint_artifact_sha256": GM1_CHILD_PARENT_CHECKPOINT_SHA256,
            "diagnostics_receipt_sha256": GM1_CHILD_PARENT_DIAGNOSTICS_SHA256,
            "pdefq_receipt_sha256": GM1_CHILD_PARENT_PDEFQ_SHA256,
            "schema": GM1_CHILD_PLAN_SCHEMA,
            "sha256": plan.plan_sha256,
        }
    ):
        raise ValueError("GM1 child producer source/parent identity changed")
    producers = _exact_keys(
        receipt["producer_requirements"],
        {
            "geopotential",
            "hydrostatic_and_open_boundary",
            "terrain_stretch_composition",
        },
        "child producer inventory",
    )
    geopotential = _exact_keys(
        producers["geopotential"],
        {
            "carrier_abi",
            "carrier_name",
            "must_bind",
            "receipt_schema",
            "requirements",
            "schema",
            "status",
        },
        "child geopotential requirement",
    )
    hydrostatic = _exact_keys(
        producers["hydrostatic_and_open_boundary"],
        {
            "carrier_abi",
            "identity_schema",
            "must_bind",
            "open_boundary_invariants",
            "receipt_schema",
            "required_carriers",
            "required_diagnostics",
            "status",
        },
        "child hydrostatic/open-boundary requirement",
    )
    terrain = _exact_keys(
        producers["terrain_stretch_composition"],
        {
            "abi",
            "configuration",
            "flat_grid_sha256",
            "must_bind",
            "required_carriers",
            "schema",
            "status",
        },
        "child terrain/stretch requirement",
    )
    for label, value in (
        ("child geopotential", geopotential),
        ("child hydrostatic/open-boundary", hydrostatic),
        ("child terrain/stretch", terrain),
    ):
        _required_pending(value, label)
    if (
        geopotential["schema"] != "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary/v1"
        or geopotential["receipt_schema"]
        != "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary-receipt/v1"
        or hydrostatic["identity_schema"]
        != "gpuwm-cm1.terrain-open-auxiliary-identity/v1"
        or hydrostatic["receipt_schema"]
        != "gpuwm-cm1.terrain-open-auxiliary-restart-receipt/v1"
        or terrain["schema"] != "gpuwm-cm1.cuda-stretch-z1-terrain/v1"
        or terrain["abi"]
        != "gpuwm-cm1.cuda-stretch-z1-terrain-physical-carriers-f32/v1"
        or terrain["flat_grid_sha256"] != GM1_CHILD_FLAT_GRID_SHA256
        or tuple(terrain["required_carriers"]) != _TERRAIN_STRETCH_CARRIERS
        or tuple(hydrostatic["required_carriers"]) != _HYDROSTATIC_OPEN_CARRIERS
    ):
        raise ValueError("GM1 child numerical producer ABI/carrier contract changed")
    boundary = _object(
        hydrostatic["open_boundary_invariants"], "child open-boundary invariants"
    )
    if boundary != {
        "all_four_boundaries_open": True,
        "cold_start_history_nonzero_words": 0,
        "parent_radiation_history_reuse_allowed": False,
        "source_valid_child_halos_required": True,
    }:
        raise ValueError("GM1 child open-boundary initialization contract changed")
    return Gm1ChildProducerRequirements(
        path=resolved,
        sha256=expected_sha256,
        parent_plan_sha256=plan.plan_sha256,
        receipt=MappingProxyType(dict(receipt)),
        terrain_stretch_carriers=_TERRAIN_STRETCH_CARRIERS,
        hydrostatic_open_boundary_carriers=_HYDROSTATIC_OPEN_CARRIERS,
    )


def load_gm1_child_terrain_operator_gate(
    requirements: Gm1ChildProducerRequirements,
    gate_path: str | Path,
    *,
    expected_sha256: str = GM1_CHILD_TERRAIN_OPERATOR_GATE_SHA256,
) -> Gm1ChildTerrainOperatorGate:
    """Admit the terrain/stretch operator proof, never its analytic weather input."""

    if not isinstance(requirements, Gm1ChildProducerRequirements):
        raise TypeError("requirements must be a Gm1ChildProducerRequirements")
    resolved, gate = _read_sealed_json(
        gate_path,
        expected_sha256=expected_sha256,
        label="GM1 child terrain/stretch operator gate",
        require_canonical=False,
    )
    gate = _exact_keys(
        gate,
        {
            "claim_boundary",
            "claim_limit",
            "compact_raw_bit_gate",
            "fixture_class",
            "full_analytic_gpu_gate",
            "gpu_mutex",
            "native_fortran_oracle",
            "operator",
            "pngs",
            "real_child_product_gate",
            "schema",
            "status",
            "time_utc",
            "vertical",
        },
        "GM1 child terrain/stretch operator gate",
    )
    operator = _exact_keys(
        gate["operator"], {"abi", "authority", "kernel", "schema"}, "terrain operator"
    )
    kernel = _object(operator["kernel"], "terrain operator kernel")
    vertical = _object(gate["vertical"], "terrain operator vertical grid")
    full = _object(gate["full_analytic_gpu_gate"], "terrain operator full gate")
    invariants = _object(full.get("invariants"), "terrain operator invariants")
    real = _exact_keys(
        gate["real_child_product_gate"],
        {
            "accepted_parent_checkpoint_sha256",
            "refusal_semantics",
            "required_next_input",
            "status",
            "validated_parent_plan",
        },
        "real child terrain gate",
    )
    plan = _object(real["validated_parent_plan"], "terrain gate parent plan")
    if (
        gate["schema"] != "gpuwm-cm1.cuda-stretch-z1-terrain-validation/v1"
        or gate["status"] != "PASS_NATIVE_CPU_CUDA_COMPACT_AND_FULL_ANALYTIC_HARDWARE"
        or "analytic" not in str(gate["fixture_class"])
        or operator["schema"] != "gpuwm-cm1.cuda-stretch-z1-terrain/v1"
        or operator["abi"]
        != "gpuwm-cm1.cuda-stretch-z1-terrain-physical-carriers-f32/v1"
        or kernel.get("source_sha256") != GM1_CHILD_TERRAIN_OPERATOR_SOURCE_SHA256
        or vertical
        != {
            "grid_sha256": GM1_CHILD_FLAT_GRID_SHA256,
            "proof_sha256": GM1_CHILD_FLAT_GRID_PROOF_SHA256,
        }
        or full.get("shape_zyx") != [75, 300, 300]
        or full.get("dx_m") != 50.0
        or full.get("dy_m") != 50.0
        or full.get("ztop_m") != 15000.0
        or real["status"] != "BLOCKED_PENDING_SOURCE_COORDINATE_TERRAIN_PRODUCER"
        or real["accepted_parent_checkpoint_sha256"]
        != GM1_CHILD_PARENT_CHECKPOINT_SHA256
        or plan.get("sha256") != requirements.parent_plan_sha256
    ):
        raise ValueError("GM1 child terrain/stretch operator gate identity changed")
    for name in (
        "all_fields_finite",
        "positive_vertical_and_terrain_metrics",
        "strictly_increasing_physical_faces",
        "surface_raw_bit_equal",
        "top_within_one_binary32_ulp",
    ):
        if invariants.get(name) is not True:
            raise ValueError(f"terrain/stretch operator invariant {name} did not pass")
    fields = _object(full.get("fields"), "terrain operator full fields")
    if set(fields) != set(_TERRAIN_STRETCH_CARRIERS):
        raise ValueError("terrain/stretch operator 23-carrier inventory changed")
    if any(
        _object(value, f"terrain operator field {name}").get("finite") is not True
        for name, value in fields.items()
    ):
        raise ValueError("terrain/stretch operator full carrier is nonfinite")
    return Gm1ChildTerrainOperatorGate(
        path=resolved,
        sha256=expected_sha256,
        requirements_sha256=requirements.sha256,
        receipt=MappingProxyType(dict(gate)),
        carrier_names=_TERRAIN_STRETCH_CARRIERS,
    )


def require_gm1_child_producer_readiness(
    plan: Gm1ChildInitializationPlan,
    readiness_receipt: Mapping[str, Any] | Gm1ChildProducerRequirements | None = None,
) -> NoReturn:
    """Refuse numerics until a later exact all-producer schema is registered."""

    if not isinstance(plan, Gm1ChildInitializationPlan):
        raise TypeError("plan must be a Gm1ChildInitializationPlan")
    if readiness_receipt is None:
        refuse(
            code="GM1-50M-CHILD-PRODUCERS-PENDING",
            gate="child-initialization",
            option="readiness_receipt",
            value=None,
            reason=(
                "the source-coordinate plan admits only geometry and the flat "
                "vertical grid; terrain/stretch composition, hydrostatic/base-state "
                "reconstruction, and open-boundary initialization are not all sealed"
            ),
            remedy="provide the later exact Rust all-producer readiness receipt",
        )
    if isinstance(readiness_receipt, Gm1ChildProducerRequirements):
        if readiness_receipt.parent_plan_sha256 != plan.plan_sha256:
            raise ValueError("producer requirements bind another child plan")
        producers = _object(
            readiness_receipt.receipt["producer_requirements"],
            "child producer inventory",
        )
        refuse(
            code="GM1-50M-CHILD-NUMERICAL-PRODUCERS-UNBOUND",
            gate="child-initialization",
            option="producer_status",
            value={name: value.get("status") for name, value in producers.items()},
            reason=(
                "the exact readiness-only contract still marks terrain/stretch, "
                "hydrostatic/open-boundary, and geopotential producers as unbound"
            ),
            remedy="seal all three numerical producer artifacts against this exact plan",
        )
    refuse(
        code="GM1-50M-CHILD-PRODUCER-SCHEMA-UNREGISTERED",
        gate="child-initialization",
        option="schema",
        value=readiness_receipt.get("schema"),
        reason="no numerical child producer receipt schema is admitted by these bytes",
        remedy="register and test the final exact Rust readiness schema before constructing arrays",
    )


__all__ = [
    "GM1_CHILD_CLAIM_LIMIT",
    "GM1_CHILD_FLAT_GRID_RECEIPT_SHA256",
    "GM1_CHILD_FLAT_GRID_SHA256",
    "GM1_CHILD_PARENT_CHECKPOINT_SHA256",
    "GM1_CHILD_PLAN_SCHEMA",
    "GM1_CHILD_PLAN_SHA256",
    "GM1_CHILD_PRODUCER_REQUIREMENTS_SCHEMA",
    "GM1_CHILD_PRODUCER_REQUIREMENTS_SHA256",
    "GM1_CHILD_TERRAIN_OPERATOR_GATE_SHA256",
    "GM1_CHILD_TERRAIN_OPERATOR_SOURCE_SHA256",
    "Gm1ChildInitializationPlan",
    "Gm1ChildProducerRequirements",
    "Gm1ChildTerrainOperatorGate",
    "load_gm1_child_initialization_plan",
    "load_gm1_child_producer_requirements",
    "load_gm1_child_terrain_operator_gate",
    "require_gm1_child_producer_readiness",
]
