"""Sealed CM1 r21.1 75-level, 15 km-top stretched child grid."""

from __future__ import annotations

import hashlib
import importlib
from typing import Any

from .cuda_stretch_z1 import (
    CUDA_STRETCH_Z1_SOURCE_SHA256,
    StretchZ1Grid,
    StretchZ1MetricProof,
    StretchZ1Parameters,
    build_stretch_z1_grid,
    build_stretch_z1_metric_proof,
)
from .errors import refuse


STRETCH_Z1_CHILD_15KM_SCHEMA = "gpuwm-cm1.stretch-z1-child-15km/v1"
STRETCH_Z1_CHILD_15KM_ABI = "gpuwm-cm1.stretch-z1-child-15km-profiles-f32/v1"
STRETCH_Z1_CHILD_15KM_PARAMETERS = StretchZ1Parameters(
    nk=75,
    dz=100.0,
    ztop=15000.0,
    str_bot=200.0,
    str_top=7000.0,
    dz_bot=20.0,
    dz_top=320.0,
)
STRETCH_Z1_CHILD_15KM_DERIVED = {
    "nk1": 10,
    "nk2": 40,
    "nk3": 25,
    "nominal_dz_f32": 170.0,
    "c1_f32": 0.09502262622117996,
    "c2_f32": 0.00013308490451890975,
}
STRETCH_Z1_CHILD_15KM_PROFILE_SHA256 = {
    "zf": "ceb1624e4dc10eb31562c7e8003f939d6d157e1ecd73d1dee479de5459f2a90e",
    "zh": "e0c800cd95ac68f526efef46de91db5463784e91860f52ea90ad76c9e18037c3",
    "sigmaf": "ceb1624e4dc10eb31562c7e8003f939d6d157e1ecd73d1dee479de5459f2a90e",
    "sigma": "e0c800cd95ac68f526efef46de91db5463784e91860f52ea90ad76c9e18037c3",
    "mh": "bc520e91ecab72cd55662dbb301c29f78cc04864ec371b3abba5dca8d8c6cbcc",
    "mf": "e686a95701fb4694da25878934e7da4713b512831a06adf549ef61e1704e245f",
    "rmh": "54833ee529867d7c005060a52c3d9e883bfe34731fd5fb74cdba15afbd039b35",
    "rmf": "ab5f395ab5c7726528eb7f7f4dd1946911d03537eb94ea9b38e1ed1dc8551515",
}
STRETCH_Z1_CHILD_15KM_GRID_SHA256 = "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
STRETCH_Z1_CHILD_15KM_PROOF_SHA256 = "277f2b8831f316b2e99a81a903f5ac20ee5d781e2c2b058963539f648f9ec02e"
AUTHORITY_TARBALL_SHA256 = "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
AUTHORITY_PARAM_F_SHA256 = "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
CM1_STRETCH_Z1_CHILD_15KM_CLAIM_LIMIT = (
    "CM1 r21.1 flat Cartesian stretch_z=1 grid only: 75 physical layers, "
    "20 m bottom spacing, 15,000 m top, dz=100 m metric reference, exact "
    "10/40/25 segment counts and sealed f32 profiles; no terrain composition, "
    "moving domain, vertical remap, pressure solve, coupled child trajectory, "
    "forecast, storm, tornado, or performance claim"
)


def require_stretch_z1_child_15km_parameters(parameters: StretchZ1Parameters) -> None:
    if not isinstance(parameters, StretchZ1Parameters):
        raise TypeError("parameters must be StretchZ1Parameters")
    target = STRETCH_Z1_CHILD_15KM_PARAMETERS
    for name in target.__dataclass_fields__:
        value, expected = getattr(parameters, name), getattr(target, name)
        if type(value) is not type(expected) or value != expected:  # noqa: E721
            refuse(
                code="ARWEN-STRETCH-Z1-CHILD-15KM-PIN",
                gate="child-vertical-grid",
                option=name,
                value=value,
                reason="the parent-matched child deck is sealed to one exact official stretch_z=1 grid",
                remedy=f"select {name}={expected!r} or add an independently evidenced child deck",
            )


def _profile_sha256(value: Any) -> str:
    np = importlib.import_module("numpy")
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32).astype("<f4", copy=False).tobytes()
    ).hexdigest()


def build_stretch_z1_child_15km_grid(
    parameters: StretchZ1Parameters = STRETCH_Z1_CHILD_15KM_PARAMETERS,
) -> StretchZ1Grid:
    require_stretch_z1_child_15km_parameters(parameters)
    grid = build_stretch_z1_grid(parameters)
    actual_derived = {
        name: getattr(grid.derived, name) for name in grid.derived.__dataclass_fields__
    }
    if actual_derived != STRETCH_Z1_CHILD_15KM_DERIVED:
        raise RuntimeError("15 km child derived segment constants differ from their pins")
    if grid.sha256 != STRETCH_Z1_CHILD_15KM_GRID_SHA256:
        raise RuntimeError("15 km child combined grid words differ from their pin")
    for name, expected in STRETCH_Z1_CHILD_15KM_PROFILE_SHA256.items():
        if _profile_sha256(getattr(grid, name)) != expected:
            raise RuntimeError(f"15 km child {name} words differ from their pin")
    return grid


def build_stretch_z1_child_15km_proof() -> StretchZ1MetricProof:
    build_stretch_z1_child_15km_grid()
    proof = build_stretch_z1_metric_proof(STRETCH_Z1_CHILD_15KM_PARAMETERS)
    if proof.proof_sha256 != STRETCH_Z1_CHILD_15KM_PROOF_SHA256:
        raise RuntimeError("15 km child metric proof differs from its pin")
    return proof


def stretch_z1_child_15km_record() -> dict[str, object]:
    grid = build_stretch_z1_child_15km_grid()
    proof = build_stretch_z1_child_15km_proof()
    return {
        "schema": STRETCH_Z1_CHILD_15KM_SCHEMA,
        "abi": STRETCH_Z1_CHILD_15KM_ABI,
        "parameters": {
            name: getattr(STRETCH_Z1_CHILD_15KM_PARAMETERS, name)
            for name in STRETCH_Z1_CHILD_15KM_PARAMETERS.__dataclass_fields__
        },
        "derived": dict(STRETCH_Z1_CHILD_15KM_DERIVED),
        "profiles": {
            name: {
                "shape": list(getattr(grid, name).shape),
                "dtype": "<f4",
                "sha256": STRETCH_Z1_CHILD_15KM_PROFILE_SHA256[name],
            }
            for name in STRETCH_Z1_CHILD_15KM_PROFILE_SHA256
        },
        "grid_sha256": grid.sha256,
        "proof_sha256": proof.proof_sha256,
        "cuda_source_sha256": CUDA_STRETCH_Z1_SOURCE_SHA256,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "src/param.F_sha256": AUTHORITY_PARAM_F_SHA256,
            "anchors": ["src/param.F:7814-7953", "src/param.F:8162-8225"],
        },
        "physical": {
            "lowest_scalar_level_m": float(grid.zh[1]),
            "physical_top_face_m": float(grid.zf[STRETCH_Z1_CHILD_15KM_PARAMETERS.nk + 1]),
            "horizontal_spacing_m": 50.0,
            "horizontal_extent_m": [15000.0, 15000.0],
        },
        "claim_limit": CM1_STRETCH_Z1_CHILD_15KM_CLAIM_LIMIT,
    }


__all__ = (
    "AUTHORITY_PARAM_F_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_STRETCH_Z1_CHILD_15KM_CLAIM_LIMIT",
    "STRETCH_Z1_CHILD_15KM_ABI",
    "STRETCH_Z1_CHILD_15KM_DERIVED",
    "STRETCH_Z1_CHILD_15KM_GRID_SHA256",
    "STRETCH_Z1_CHILD_15KM_PARAMETERS",
    "STRETCH_Z1_CHILD_15KM_PROFILE_SHA256",
    "STRETCH_Z1_CHILD_15KM_PROOF_SHA256",
    "STRETCH_Z1_CHILD_15KM_SCHEMA",
    "build_stretch_z1_child_15km_grid",
    "build_stretch_z1_child_15km_proof",
    "require_stretch_z1_child_15km_parameters",
    "stretch_z1_child_15km_record",
)
