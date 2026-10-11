"""Source-exact bounded CM1 r21.1 pre-landuse surface initialization.

This CPU operator owns ``init_surface.F:40-227`` on supplied compact
carriers: the four documented ``initsfc`` choices, default surface fields,
multi-layer soil initialization, horizontal surface length scales, and the
source-order ``dosfcflx`` compatibility decision.  It stops before the
external ``LANDUSE.TBL`` lookup and deliberately does not claim a coupled
surface model, ocean model, turbulence step, or production configuration.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.errors import refuse


CM1_SURFACE_INITIALIZATION_SCHEMA = "gpuwm-cm1.surface-initialization/v1"
CM1_SURFACE_INITIALIZATION_STATUS = (
    "ADMITTED_CM1_R21_1_PRE_LANDUSE_SURFACE_INITIALIZATION_OPERATOR_ONLY"
)
CM1_SURFACE_INITIALIZATION_CLAIM_LIMIT = (
    "CM1 r21.1 init_surface.F:40-227 compact source-order operator for "
    "initsfc=1/2/3/4, default surface fields, soil slabs, dsxy, and dosfcflx "
    "only; no LANDUSE.TBL/getlanduse, post-landuse roughness, ocean model, "
    "surface-flux tendency, MPI, coupled step, production configuration, "
    "trajectory, forecast, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_SURFACE_INITIALIZATION_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/init_surface.F": (
            "b4437486b7582464c287794b1904bdadec5abe369a50f444c7facbc0f5ccb66b"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
    }
)
CM1_SURFACE_INITIALIZATION_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:916-989",
        "default_fields": "src/init_surface.F:40-65",
        "initsfc_dispatch": "src/init_surface.F:67-145",
        "soil_slabs": "src/init_surface.F:152-172",
        "surface_length_scale": "src/init_surface.F:174-203",
        "surface_flux_compatibility": "src/init_surface.F:207-227",
        "excluded_landuse_boundary": "src/init_surface.F:236-243",
    }
)


@dataclass(frozen=True, slots=True)
class SurfaceInitializationSelection:
    """Exact discrete selectors and default-REAL inputs for one source block."""

    initsfc: int
    sfcmodel: int
    bbc: int
    axisymm: int
    cm1setup: int
    ipbl: int
    isfcflx: int
    tsk0_k: float
    tmn0_k: float
    xland0: float
    lu0: int
    minx_m: float
    maxx_m: float
    centerx_m: float
    dx_m: float
    dy_m: float


@dataclass(frozen=True, slots=True)
class SurfaceInitializationResult:
    """Completed in-place pre-landuse state and immutable receipt."""

    arrays: Mapping[str, Any]
    dosfcflx: bool
    receipt: Mapping[str, object]


_FLOAT_2D = ("tsk", "tmn", "xland", "znt", "dsxy")
_FLOAT_1D = ("xh", "ruh", "rvh", "slab_zs", "slab_dzs")
_FLOAT_3D = ("tslb",)
_INTEGER_2D = ("lu_index",)
_OUTPUT_NAMES = (
    "tsk",
    "tmn",
    "xland",
    "lu_index",
    "znt",
    "slab_zs",
    "slab_dzs",
    "tslb",
    "dsxy",
)


def surface_initialization_shapes(
    ni: int, nj: int, num_soil_layers: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (
        ("ni", ni),
        ("nj", nj),
        ("num_soil_layers", num_soil_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    result: dict[str, tuple[int, ...]] = {
        "xh": (ni,),
        "ruh": (ni,),
        "rvh": (nj,),
        "slab_zs": (num_soil_layers,),
        "slab_dzs": (num_soil_layers,),
        "tslb": (num_soil_layers, nj, ni),
        "lu_index": (nj, ni),
    }
    result.update({name: (nj, ni) for name in _FLOAT_2D})
    return result


def _f32(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite default REAL")
    return result


def _validate_selection(
    np: Any, selection: SurfaceInitializationSelection
) -> dict[str, Any]:
    if not isinstance(selection, SurfaceInitializationSelection):
        raise TypeError("selection must be a SurfaceInitializationSelection")
    for name in (
        "initsfc",
        "sfcmodel",
        "bbc",
        "axisymm",
        "cm1setup",
        "ipbl",
        "isfcflx",
        "lu0",
    ):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    admitted = {
        "initsfc": range(1, 5),
        "sfcmodel": range(0, 8),
        "bbc": range(1, 4),
        "axisymm": range(0, 2),
        "cm1setup": range(0, 5),
        "ipbl": range(0, 7),
        "isfcflx": range(0, 2),
    }
    for name, values in admitted.items():
        value = getattr(selection, name)
        if value not in values:
            refuse(
                code="CM1-SURFACE-INITIALIZATION-SELECTION",
                gate="GM1-PHYSICS-SUPPORT",
                option=name,
                value=value,
                reason=(
                    "the bounded pre-landuse source operator does not admit "
                    f"this {name} value"
                ),
                remedy=f"select one of {list(values)}",
            )
    if not 1 <= selection.lu0 <= 33:
        raise ValueError("lu0 must be a LANDUSE.TBL category in [1, 33]")
    scalar = {
        name: _f32(np, name, getattr(selection, name))
        for name in (
            "tsk0_k",
            "tmn0_k",
            "xland0",
            "minx_m",
            "maxx_m",
            "centerx_m",
            "dx_m",
            "dy_m",
        )
    }
    if scalar["maxx_m"] <= scalar["minx_m"]:
        raise ValueError("maxx_m must exceed minx_m")
    if scalar["dx_m"] <= np.float32(0.0) or scalar["dy_m"] <= np.float32(0.0):
        raise ValueError("dx_m and dy_m must be strictly positive")
    return scalar


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    num_soil_layers: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = surface_initialization_shapes(ni, nj, num_soil_layers)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError("missing surface-initialization arrays: " + ", ".join(missing))
    all_values: list[tuple[str, Any]] = []
    for name in (*_FLOAT_2D, *_FLOAT_1D, *_FLOAT_3D):
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(f"{name} must have native float32 dtype")
        if tuple(value.shape) != shapes[name]:
            raise ValueError(f"{name} shape {value.shape} != {shapes[name]}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
        all_values.append((name, value))
    for name in _INTEGER_2D:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype(np.int32):
            raise TypeError(f"{name} must have native int32 dtype")
        if tuple(value.shape) != shapes[name]:
            raise ValueError(f"{name} shape {value.shape} != {shapes[name]}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        all_values.append((name, value))
    for name in _OUTPUT_NAMES:
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    if np.any(arrays["ruh"] <= np.float32(0.0)) or np.any(
        arrays["rvh"] <= np.float32(0.0)
    ):
        raise ValueError("ruh and rvh must be strictly positive")
    for index, (left_name, left) in enumerate(all_values):
        for right_name, right in all_values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "surface-initialization arrays must not alias: "
                    f"{left_name}, {right_name}"
                )


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_surface_initialization_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    num_soil_layers: int,
    selection: SurfaceInitializationSelection,
) -> SurfaceInitializationResult:
    """Execute the bounded ``init_surface.F:40-227`` block in-place."""

    np = importlib.import_module("numpy")
    scalar = _validate_selection(np, selection)
    _validate_arrays(
        np,
        arrays,
        ni=ni,
        nj=nj,
        num_soil_layers=num_soil_layers,
    )

    outer_active = selection.sfcmodel >= 1 or selection.bbc == 3
    soil_active = selection.sfcmodel in (2, 3, 4, 6, 7)
    dsxy_branch = "inactive"
    if outer_active:
        arrays["tsk"].fill(scalar["tsk0_k"])
        arrays["tmn"].fill(scalar["tmn0_k"])
        arrays["xland"].fill(scalar["xland0"])
        arrays["lu_index"].fill(np.int32(selection.lu0))

        if selection.initsfc == 2:
            xspan = _sub(np, scalar["maxx_m"], scalar["minx_m"])
            x1 = _add(
                np,
                scalar["minx_m"],
                _mul(np, _div(np, np.float32(3.0), np.float32(8.0)), xspan),
            )
            x2 = _add(
                np,
                scalar["minx_m"],
                _mul(np, _div(np, np.float32(5.0), np.float32(8.0)), xspan),
            )
            for j in range(nj):
                for i in range(ni):
                    if arrays["xh"][i] >= x1 and arrays["xh"][i] <= x2:
                        arrays["xland"][j, i] = np.float32(1.0)
                        arrays["lu_index"][j, i] = np.int32(18)
                        arrays["tsk"][j, i] = np.float32(280.0)
                    else:
                        arrays["xland"][j, i] = np.float32(2.0)
                        arrays["lu_index"][j, i] = np.int32(16)
                        arrays["tsk"][j, i] = np.float32(287.0)
                    arrays["tmn"][j, i] = _sub(
                        np, arrays["tsk"][j, i], np.float32(0.5)
                    )
        elif selection.initsfc == 3:
            for j in range(nj):
                for i in range(ni):
                    arrays["znt"][j, i] = (
                        np.float32(0.010)
                        if arrays["xh"][i] <= scalar["centerx_m"]
                        else np.float32(0.001)
                    )
        elif selection.initsfc == 4:
            xcoast = _add(
                np,
                scalar["minx_m"],
                _mul(
                    np,
                    np.float32(0.5),
                    _sub(np, scalar["maxx_m"], scalar["minx_m"]),
                ),
            )
            for j in range(nj):
                for i in range(ni):
                    if arrays["xh"][i] < xcoast:
                        arrays["xland"][j, i] = np.float32(1.0)
                        arrays["lu_index"][j, i] = np.int32(18)
                    else:
                        arrays["xland"][j, i] = np.float32(2.0)
                        arrays["lu_index"][j, i] = np.int32(16)

        if soil_active and num_soil_layers != 1:
            arrays["slab_dzs"][0] = np.float32(0.01)
            arrays["slab_zs"][0] = _mul(
                np, np.float32(0.5), arrays["slab_dzs"][0]
            )
            for level in range(1, num_soil_layers):
                arrays["slab_dzs"][level] = _mul(
                    np, np.float32(2.0), arrays["slab_dzs"][level - 1]
                )
                arrays["slab_zs"][level] = _add(
                    np,
                    _add(
                        np,
                        arrays["slab_zs"][level - 1],
                        _mul(
                            np,
                            np.float32(0.5),
                            arrays["slab_dzs"][level - 1],
                        ),
                    ),
                    _mul(np, np.float32(0.5), arrays["slab_dzs"][level]),
                )
            denominator = _sub(
                np, arrays["slab_zs"][-1], arrays["slab_zs"][0]
            )
            for level in range(num_soil_layers):
                tsk_weight = _sub(
                    np, arrays["slab_zs"][-1], arrays["slab_zs"][level]
                )
                tmn_weight = _sub(
                    np, arrays["slab_zs"][level], arrays["slab_zs"][0]
                )
                for j in range(nj):
                    for i in range(ni):
                        numerator = _add(
                            np,
                            _mul(np, arrays["tsk"][j, i], tsk_weight),
                            _mul(np, arrays["tmn"][j, i], tmn_weight),
                        )
                        arrays["tslb"][level, j, i] = _div(
                            np, numerator, denominator
                        )

        if selection.sfcmodel >= 1:
            if selection.axisymm == 1 or nj == 1:
                dsxy_branch = "dx_ruh"
                for j in range(nj):
                    for i in range(ni):
                        arrays["dsxy"][j, i] = _mul(
                            np, scalar["dx_m"], arrays["ruh"][i]
                        )
            elif ni == 1:
                dsxy_branch = "dy_rvh"
                for j in range(nj):
                    for i in range(ni):
                        arrays["dsxy"][j, i] = _mul(
                            np, scalar["dy_m"], arrays["rvh"][j]
                        )
            else:
                dsxy_branch = "max_dx_ruh_dy_rvh"
                for j in range(nj):
                    for i in range(ni):
                        arrays["dsxy"][j, i] = max(
                            _mul(np, scalar["dx_m"], arrays["ruh"][i]),
                            _mul(np, scalar["dy_m"], arrays["rvh"][j]),
                        )

    dosfcflx = False
    if selection.cm1setup >= 1 or selection.ipbl == 2:
        if selection.isfcflx == 1:
            dosfcflx = True
        if soil_active:
            dosfcflx = True
    if selection.ipbl in (1, 3, 4, 5):
        dosfcflx = False
    if selection.cm1setup == 4 and selection.isfcflx == 1:
        dosfcflx = True

    for name in _OUTPUT_NAMES:
        if arrays[name].dtype == np.dtype(np.float32) and not np.all(
            np.isfinite(arrays[name])
        ):
            refuse(
                code="CM1-SURFACE-INITIALIZATION-NONFINITE",
                gate="GM1-PHYSICS-SUPPORT",
                option="surface_initialization_output",
                value=name,
                reason="source-order initialization produced a non-finite carrier",
                remedy="inspect supplied surface defaults and metric carriers",
            )

    receipt: dict[str, object] = {
        "schema": CM1_SURFACE_INITIALIZATION_SCHEMA,
        "status": CM1_SURFACE_INITIALIZATION_STATUS,
        "claim_limit": CM1_SURFACE_INITIALIZATION_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_SURFACE_INITIALIZATION_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_SURFACE_INITIALIZATION_SOURCE_ANCHORS),
        },
        "selection": {
            "initsfc": selection.initsfc,
            "sfcmodel": selection.sfcmodel,
            "bbc": selection.bbc,
            "axisymm": selection.axisymm,
            "cm1setup": selection.cm1setup,
            "ipbl": selection.ipbl,
            "isfcflx": selection.isfcflx,
            "lu0": selection.lu0,
            **{
                f"{name}_raw_bits": int(np.asarray(value).view(np.uint32))
                for name, value in scalar.items()
            },
        },
        "domain": {
            "ni": ni,
            "nj": nj,
            "num_soil_layers": num_soil_layers,
        },
        "branch": {
            "outer_active": outer_active,
            "soil_active": soil_active,
            "dsxy": dsxy_branch,
            "dosfcflx": dosfcflx,
        },
        "output_sha256": {name: _sha256_array(arrays[name]) for name in _OUTPUT_NAMES},
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return SurfaceInitializationResult(
        arrays=MappingProxyType({name: arrays[name] for name in _OUTPUT_NAMES}),
        dosfcflx=dosfcflx,
        receipt=MappingProxyType(receipt),
    )


__all__ = [
    "CM1_SURFACE_INITIALIZATION_AUTHORITY_SHA256",
    "CM1_SURFACE_INITIALIZATION_CLAIM_LIMIT",
    "CM1_SURFACE_INITIALIZATION_SCHEMA",
    "CM1_SURFACE_INITIALIZATION_SOURCE_ANCHORS",
    "CM1_SURFACE_INITIALIZATION_STATUS",
    "SurfaceInitializationResult",
    "SurfaceInitializationSelection",
    "cpu_surface_initialization_inplace",
    "surface_initialization_shapes",
]
