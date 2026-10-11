"""Source-pinned interfaces for the fixed 50 m GM1 child producers.

This module contains no interpolation or atmospheric arithmetic.  It turns the
sealed Rust requirements into explicit callable/output contracts and verifies
their CM1 r21.1 authority bytes before a numerical producer may be connected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from cm1core.gm1_child_initialization import (
    Gm1ChildInitializationPlan,
    Gm1ChildProducerRequirements,
)


GM1_CHILD_PRODUCER_INTERFACES_SCHEMA = "gm1.fixed-50m-child-producer-interfaces/v1"
GM1_CHILD_TERRAIN_ARTIFACT_SCHEMA = "gm1.source-coordinate-50m-child-terrain/v1"
GM1_CHILD_TERRAIN_MANIFEST_SCHEMA = (
    "gm1.source-coordinate-50m-child-terrain-manifest/v1"
)
GM1_CHILD_TERRAIN_ARTIFACT_ABI = "gm1.source-coordinate-50m-child-terrain-f32/v1"
GM1_CHILD_TERRAIN_METRIC_API = (
    "cm1core.cuda_gm1_child_terrain.launch_cuda_gm1_child_terrain_metrics"
)
GM1_CHILD_TERRAIN_INPUT_SHA256 = MappingProxyType(
    {
        "manifest_sha256": "bc6006491174c4b7594f4c12582749f3d3494015bba2eb692106005d3aeb391a",
        "payload_sha256": "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91",
        "receipt_sha256": "0c50ea83658b4ea47a711580200fe7280face00baba7b089d3e5572280004bbf",
        "producer_source_sha256": "51cf54b29ed25efa0bab281930923563443ec3311a287bbb8465ac8079888acd",
        "producer_executable_sha256": "6d7b7d525e0fa1c3be2b9892c6b2ee7fcf60671159a4f97c35de18695edcabbf",
    }
)
GM1_CHILD_HYDROSTATIC_RECEIPT_SCHEMA = (
    "gpuwm-cm1.terrain-open-auxiliary-restart-receipt/v1"
)
GM1_CHILD_HYDROSTATIC_IDENTITY_SCHEMA = "gpuwm-cm1.terrain-open-auxiliary-identity/v1"
GM1_CHILD_HYDROSTATIC_CARRIER_ABI = "gpuwm-cm1.terrain-open-auxiliary-carriers/v1"
GM1_CHILD_GEOPOTENTIAL_SCHEMA = "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary/v1"
GM1_CHILD_GEOPOTENTIAL_RECEIPT_SCHEMA = (
    "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary-receipt/v1"
)
GM1_CHILD_GEOPOTENTIAL_CARRIER_ABI = "gpuwm-cm1.arwen-wsm6-total-geopotential/v1"
GM1_CHILD_PRODUCER_INTERFACE_CLAIM_LIMIT = (
    "exact source/configuration/input/output contracts for the fixed 300x300x75 "
    "50 m child terrain metrics, terrain-varying hydrostatic/base state, fresh "
    "four-open boundary state, and child WSM6 geopotential only; no atmospheric "
    "remap, numerical producer output, child state, evolution, forecast, or tornado claim"
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = MappingProxyType(
    {
        "src/init_terrain.F": "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
        "src/param.F": "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
        "src/base.F": "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
        "src/init3d.F": "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        "src/restart_read.F": "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a",
        "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
        "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
        "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    }
)
SOURCE_ANCHORS = MappingProxyType(
    {
        "terrain_coordinates": "src/init_terrain.F:175-297",
        "vertical_and_boundary_metrics": "src/param.F:7789-7812,8162-8207,8464-8492,8988-9062",
        "terrain_hydrostatic_base": "src/base.F:625-749,2060-2243,3209",
        "restart_grid_state": "src/restart_read.F:1206-1211,2657-2664,3302-3465",
        "state_pressure_density_publish": "src/init3d.F:2280-2467",
        "open_boundary_families": "src/bc.F:12-172,180-1206,1948-2172",
        "fresh_radiation_history": "src/cm1.F:1559-1566",
    }
)


@dataclass(frozen=True, slots=True)
class Gm1ChildProducerInterfaces:
    schema: str
    parent_plan_sha256: str
    requirements_sha256: str
    authority_root: Path
    interfaces: Mapping[str, Mapping[str, Any]]
    numerical_child_output_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model": "GM1",
            "parent_plan_sha256": self.parent_plan_sha256,
            "requirements_sha256": self.requirements_sha256,
            "authority": {
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
                "anchors": dict(SOURCE_ANCHORS),
            },
            "interfaces": {
                name: dict(value) for name, value in self.interfaces.items()
            },
            "numerical_child_output_allowed": False,
            "claim_limit": GM1_CHILD_PRODUCER_INTERFACE_CLAIM_LIMIT,
        }


def validate_gm1_child_producer_authority(
    authority_root: str | Path,
) -> Path:
    root = Path(authority_root).resolve(strict=True)
    if not root.is_dir() or Path(authority_root).is_symlink():
        raise ValueError("GM1 child producer authority must be a real directory")
    for relative, expected in AUTHORITY_SOURCE_SHA256.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular authority source {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"authority source {relative} differs: {actual}")
    return root


def build_gm1_child_producer_interfaces(
    plan: Gm1ChildInitializationPlan,
    requirements: Gm1ChildProducerRequirements,
    *,
    authority_root: str | Path,
) -> Gm1ChildProducerInterfaces:
    if not isinstance(plan, Gm1ChildInitializationPlan):
        raise TypeError("plan must be a Gm1ChildInitializationPlan")
    if not isinstance(requirements, Gm1ChildProducerRequirements):
        raise TypeError("requirements must be a Gm1ChildProducerRequirements")
    if requirements.parent_plan_sha256 != plan.plan_sha256:
        raise ValueError("producer requirements bind another child plan")
    root = validate_gm1_child_producer_authority(authority_root)
    raw = requirements.receipt["producer_requirements"]
    terrain = raw["terrain_stretch_composition"]
    hydrostatic = raw["hydrostatic_and_open_boundary"]
    geopotential = raw["geopotential"]
    interfaces: dict[str, Mapping[str, Any]] = {
        "terrain_stretch_composition": MappingProxyType(
            {
                "status": "READY_FOR_STRICT_REAL_TERRAIN_LOADER_BINDING",
                "input_directory_inventory": [
                    "manifest.json",
                    "receipt.json",
                    "terrain-msl.f32le",
                ],
                "input_schema": GM1_CHILD_TERRAIN_ARTIFACT_SCHEMA,
                "input_manifest_schema": GM1_CHILD_TERRAIN_MANIFEST_SCHEMA,
                "input_abi": GM1_CHILD_TERRAIN_ARTIFACT_ABI,
                "input_dtype": "<f4",
                "input_shape": [300, 300],
                "input_nbytes": 360000,
                "exact_input_identity": dict(GM1_CHILD_TERRAIN_INPUT_SHA256),
                "consumer_api": GM1_CHILD_TERRAIN_METRIC_API,
                "output_schema": terrain["schema"],
                "output_abi": terrain["abi"],
                "required_carriers": list(terrain["required_carriers"]),
                "must_bind": list(terrain["must_bind"]),
            }
        ),
        "hydrostatic_and_open_boundary": MappingProxyType(
            {
                "status": "BLOCKED_PENDING_CHILD_ATMOSPHERIC_REMAP_AND_TERRAIN_METRICS",
                "output_identity_schema": GM1_CHILD_HYDROSTATIC_IDENTITY_SCHEMA,
                "output_receipt_schema": GM1_CHILD_HYDROSTATIC_RECEIPT_SCHEMA,
                "output_carrier_abi": GM1_CHILD_HYDROSTATIC_CARRIER_ABI,
                "required_carriers": list(hydrostatic["required_carriers"]),
                "required_diagnostics": list(hydrostatic["required_diagnostics"]),
                "open_boundary_invariants": dict(
                    hydrostatic["open_boundary_invariants"]
                ),
                "must_bind": list(hydrostatic["must_bind"]),
                "source_order": [
                    "terrain-varying base profiles and hydrostatic pi0/prs0/rho0",
                    "preserve restart-grid-interpolated ppi/tha/qa; the init3d "
                    "ibalance=1 pressure pass occurred before restart_read and is scratch",
                    "source-family halo fills then exact A-to-current copies",
                    "publish prs/rho then rr/rf using child c1/c2",
                    "psolver=3 overwrites independently remapped ppx from ppi",
                    "source-family open halos and fresh zero radbcw/e/s/n",
                ],
            }
        ),
        "geopotential": MappingProxyType(
            {
                "status": "BLOCKED_PENDING_CHILD_INITIALIZED_PRODUCT_AND_TERRAIN_ZF",
                "output_schema": GM1_CHILD_GEOPOTENTIAL_SCHEMA,
                "output_receipt_schema": GM1_CHILD_GEOPOTENTIAL_RECEIPT_SCHEMA,
                "output_carrier_abi": GM1_CHILD_GEOPOTENTIAL_CARRIER_ABI,
                "carrier_name": geopotential["carrier_name"],
                "must_bind": list(geopotential["must_bind"]),
                "requirements": list(geopotential["requirements"]),
            }
        ),
    }
    return Gm1ChildProducerInterfaces(
        schema=GM1_CHILD_PRODUCER_INTERFACES_SCHEMA,
        parent_plan_sha256=plan.plan_sha256,
        requirements_sha256=requirements.sha256,
        authority_root=root,
        interfaces=MappingProxyType(interfaces),
    )


__all__ = [
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "GM1_CHILD_GEOPOTENTIAL_CARRIER_ABI",
    "GM1_CHILD_GEOPOTENTIAL_RECEIPT_SCHEMA",
    "GM1_CHILD_GEOPOTENTIAL_SCHEMA",
    "GM1_CHILD_HYDROSTATIC_CARRIER_ABI",
    "GM1_CHILD_HYDROSTATIC_IDENTITY_SCHEMA",
    "GM1_CHILD_HYDROSTATIC_RECEIPT_SCHEMA",
    "GM1_CHILD_PRODUCER_INTERFACE_CLAIM_LIMIT",
    "GM1_CHILD_PRODUCER_INTERFACES_SCHEMA",
    "GM1_CHILD_TERRAIN_ARTIFACT_ABI",
    "GM1_CHILD_TERRAIN_ARTIFACT_SCHEMA",
    "GM1_CHILD_TERRAIN_MANIFEST_SCHEMA",
    "GM1_CHILD_TERRAIN_METRIC_API",
    "GM1_CHILD_TERRAIN_INPUT_SHA256",
    "Gm1ChildProducerInterfaces",
    "SOURCE_ANCHORS",
    "build_gm1_child_producer_interfaces",
    "validate_gm1_child_producer_authority",
]
