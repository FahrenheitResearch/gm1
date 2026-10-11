"""Complete active CM1 r21.1 Rayleigh-damping CPU operator.

This bounded operator covers vertical ``irdamp=1,2`` coefficient construction,
all-four-side ``hrdamp=1`` coefficient construction, the serial area-weighted
horizontal averages required by ``irdamp=2``, and both tendency branches in
``solve1.F:390-454``.  It deliberately excludes terrain coupling, MPI
reductions, diagnostics, and coupled model integration.
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


CM1_RAYLEIGH_DAMPING_SCHEMA = "gpuwm-cm1.rayleigh-damping/v1"
CM1_RAYLEIGH_DAMPING_STATUS = "ADMITTED_CM1_R21_1_RAYLEIGH_OPERATOR_ONLY"
CM1_RAYLEIGH_DAMPING_CLAIM_LIMIT = (
    "CM1 r21.1 cm1.F coefficient-zero initialization, param.F vertical and "
    "all-four-side horizontal coefficient construction, serial get_avg_uvt, "
    "and solve1.F base-state/horizontal-average Rayleigh tendency assignment "
    "only; no terrain average damping, MPI reduction, diagnostics, coupled "
    "trajectory, production configuration, forecast, or validation claim"
)
CM1_R21_1_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
CM1_RAYLEIGH_AUTHORITY_SHA256 = MappingProxyType(
    {
        "README.namelist": (
            "7b95be56db51f5c9396c59dca252cf96b918a312cc70107451f91149a34ab3b5"
        ),
        "src/cm1.F": (
            "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
        "src/testcase_simple_phys.F": (
            "ce0a74ba558634cf712669aae3cbee2eafeb4e23a116e0fe3d819886938ffc72"
        ),
    }
)
CM1_RAYLEIGH_SOURCE_ANCHORS = MappingProxyType(
    {
        "documentation": "README.namelist:391-402",
        "selection_normalization": "src/param.F:724-725",
        "terrain_average_refusal": "src/param.F:1730-1737",
        "coefficient_initialization": "src/cm1.F:850-864",
        "vertical_coefficients": "src/param.F:8631-8673",
        "lateral_coefficients": "src/param.F:8684-8778",
        "serial_horizontal_average": "src/testcase_simple_phys.F:245-299",
        "tendency_assignment": "src/solve1.F:390-454",
    }
)


@dataclass(frozen=True, slots=True)
class RayleighDampingSelection:
    """Explicit active Rayleigh selector and continuous parameters."""

    upper_damping: int
    horizontal_damping: int
    rdalpha_s_inv: float
    zd_m: float
    maxz_m: float
    xhd_m: float
    minx_m: float
    maxx_m: float
    miny_m: float
    maxy_m: float
    dx_m: float
    dy_m: float
    terrain_flag: bool = False
    axisymmetric: int = 0


@dataclass(frozen=True, slots=True)
class RayleighDampingResult:
    """Completed coefficient/average/tendency operator and receipt."""

    receipt: Mapping[str, object]


def rayleigh_damping_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    halo = (nk, nj + 2, ni + 2)
    state = (nk, nj + 1, ni + 1)
    return {
        "zh": halo,
        "zf": (nk + 1, nj + 2, ni + 2),
        "xh": (ni + 2,),
        "yh": (nj + 2,),
        "tauh": halo,
        "taus": halo,
        "tauf": (nk + 1, nj + 2, ni + 2),
        "ua": state,
        "va": state,
        "wa": state,
        "tha": state,
        "u0": state,
        "v0": state,
        "th0": state,
        "uten1": state,
        "vten1": state,
        "wten1": state,
        "thten1": state,
        "ruh": (ni,),
        "ruf": (ni,),
        "rvh": (nj,),
        "rvf": (nj,),
        "uavg": (nk,),
        "vavg": (nk,),
        "thavg": (nk,),
        "cavg": (nk, 3),
    }


def _f32(np: Any, name: str, value: object) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite default REAL")
    return result


def _validate_selection(np: Any, selection: RayleighDampingSelection) -> int:
    if not isinstance(selection, RayleighDampingSelection):
        raise TypeError("selection must be a RayleighDampingSelection")
    for name in ("upper_damping", "horizontal_damping", "axisymmetric"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if selection.upper_damping not in (0, 1, 2):
        refuse(
            code="CM1-RAYLEIGH-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="irdamp",
            value=selection.upper_damping,
            reason="only CM1 r21.1 irdamp=0,1,2 are represented",
            remedy="use irdamp=0,1,2",
        )
    if selection.horizontal_damping not in (0, 1):
        refuse(
            code="CM1-RAYLEIGH-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="hrdamp",
            value=selection.horizontal_damping,
            reason="the namelist exposes hrdamp=0,1 only",
            remedy="use hrdamp=0 or 1",
        )
    if selection.upper_damping == 0 and selection.horizontal_damping == 0:
        refuse(
            code="CM1-RAYLEIGH-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="irdamp/hrdamp",
            value=[0, 0],
            reason="the active operator does not own the disabled identity path",
            remedy="use the separate no-damping identity path",
        )
    if not isinstance(selection.terrain_flag, bool):
        raise TypeError("terrain_flag must be bool")
    effective_horizontal = (
        selection.upper_damping
        if selection.upper_damping >= 1 and selection.horizontal_damping >= 1
        else selection.horizontal_damping
    )
    if selection.terrain_flag:
        if selection.upper_damping == 2 or effective_horizontal == 2:
            refuse(
                code="CM1-RAYLEIGH-TERRAIN-AVERAGE",
                gate="DYNAMICS-BRANCH-MATRIX",
                option="terrain_flag",
                value=True,
                reason=(
                    "param.F refuses irdamp=2 or normalized hrdamp=2 with terrain"
                ),
                remedy="use the flat bounded operator",
            )
        refuse(
            code="CM1-RAYLEIGH-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="terrain_flag",
            value=True,
            reason="the source-exact bounded operator admits only flat Cartesian grids",
            remedy="use terrain_flag=false",
        )
    if selection.axisymmetric != 0:
        refuse(
            code="CM1-RAYLEIGH-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="axisymmetric",
            value=selection.axisymmetric,
            reason="the all-four-side horizontal coefficient operator is Cartesian",
            remedy="use axisymmetric=0",
        )
    values = {
        name: _f32(np, name, getattr(selection, name))
        for name in (
            "rdalpha_s_inv",
            "zd_m",
            "maxz_m",
            "xhd_m",
            "minx_m",
            "maxx_m",
            "miny_m",
            "maxy_m",
            "dx_m",
            "dy_m",
        )
    }
    if values["rdalpha_s_inv"] < 0 or values["dx_m"] <= 0 or values["dy_m"] <= 0:
        raise ValueError("rdalpha must be nonnegative and dx/dy positive")
    if values["maxx_m"] <= values["minx_m"] or values["maxy_m"] <= values["miny_m"]:
        raise ValueError("domain maximums must exceed minimums")
    if selection.upper_damping >= 1:
        if values["zd_m"] < np.float32(0.5) * values["maxz_m"]:
            raise ValueError("zd_m must be at or above half maxz_m")
        if values["zd_m"] >= values["maxz_m"]:
            raise ValueError("zd_m must be below maxz_m for an active upper damper")
    if selection.horizontal_damping >= 1:
        half_x = np.float32(0.5) * np.float32(values["maxx_m"] - values["minx_m"])
        half_y = np.float32(0.5) * np.float32(values["maxy_m"] - values["miny_m"])
        if values["xhd_m"] <= 0 or values["xhd_m"] > min(half_x, half_y):
            raise ValueError("xhd_m must be positive and no larger than either half-domain")
    return effective_horizontal


def _validate_arrays(
    np: Any,
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    if not isinstance(arrays, MutableMapping):
        raise TypeError("arrays must be a mutable mapping")
    shapes = rayleigh_damping_shapes(ni, nj, nk)
    missing = [name for name in shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing Rayleigh arrays: {', '.join(missing)}")
    values = []
    for name, shape in shapes.items():
        value = arrays[name]
        expected_dtype = np.float64 if name == "cavg" else np.float32
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype(expected_dtype):
            raise TypeError(f"{name} must have native {np.dtype(expected_dtype)} dtype")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        values.append((name, value))
    for name in (
        "tauh",
        "taus",
        "tauf",
        "uten1",
        "vten1",
        "wten1",
        "thten1",
        "uavg",
        "vavg",
        "thavg",
        "cavg",
    ):
        if not arrays[name].flags.writeable:
            raise ValueError(f"{name} must be writeable")
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    f"Rayleigh arrays must not alias: {left_name}, {right_name}"
                )


def _coefficient(np: Any, distance: Any, depth: Any) -> Any:
    pi = np.float32("3.1415926535897932384626433")
    angle = np.float32(pi * distance)
    angle = np.float32(angle / depth)
    cosine = np.float32(np.cos(angle))
    value = np.float32(np.float32(1.0) - cosine)
    return np.float32(np.float32(0.5) * value)


def _build_coefficients(
    np: Any,
    arrays: MutableMapping[str, Any],
    selection: RayleighDampingSelection,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    tauh, taus, tauf = arrays["tauh"], arrays["taus"], arrays["tauf"]
    tauh.fill(np.float32(0.0))
    taus.fill(np.float32(0.0))
    tauf.fill(np.float32(0.0))
    zd = _f32(np, "zd_m", selection.zd_m)
    if selection.upper_damping >= 1:
        for j in range(nj + 2):
            for i in range(ni + 2):
                top_depth = np.float32(arrays["zf"][nk, j, i] - zd)
                for k in range(nk):
                    height = arrays["zh"][k, j, i]
                    if height > zd:
                        distance = np.float32(height - zd)
                        tauh[k, j, i] = _coefficient(np, distance, top_depth)
                        taus[k, j, i] = tauh[k, j, i]
                for k in range(nk + 1):
                    height = arrays["zf"][k, j, i]
                    if height > zd:
                        distance = np.float32(height - zd)
                        tauf[k, j, i] = _coefficient(np, distance, top_depth)
    if selection.horizontal_damping >= 1:
        xhd = _f32(np, "xhd_m", selection.xhd_m)
        minx = _f32(np, "minx_m", selection.minx_m)
        maxx = _f32(np, "maxx_m", selection.maxx_m)
        miny = _f32(np, "miny_m", selection.miny_m)
        maxy = _f32(np, "maxy_m", selection.maxy_m)
        for k in range(nk):
            for j in range(nj + 2):
                for i in range(ni + 2):
                    west = np.float32(np.float32(xhd + minx) - arrays["xh"][i])
                    if west > 0:
                        tauf[k, j, i] = max(
                            tauf[k, j, i], _coefficient(np, west, xhd)
                        )
                    east = np.float32(arrays["xh"][i] - np.float32(maxx - xhd))
                    if east > 0:
                        tauf[k, j, i] = max(
                            tauf[k, j, i], _coefficient(np, east, xhd)
                        )
                    south = np.float32(np.float32(xhd + miny) - arrays["yh"][j])
                    if south > 0:
                        tauf[k, j, i] = max(
                            tauf[k, j, i], _coefficient(np, south, xhd)
                        )
                    north = np.float32(arrays["yh"][j] - np.float32(maxy - xhd))
                    if north > 0:
                        tauf[k, j, i] = max(
                            tauf[k, j, i], _coefficient(np, north, xhd)
                        )


def _horizontal_averages(
    np: Any,
    arrays: MutableMapping[str, Any],
    selection: RayleighDampingSelection,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    cavg = arrays["cavg"]
    cavg.fill(np.float64(0.0))
    tem = np.float32(
        _f32(np, "dx_m", selection.dx_m) * _f32(np, "dy_m", selection.dy_m)
    )
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                term = np.float32(arrays["ua"][k, j, i] * tem)
                term = np.float32(term * arrays["ruf"][i])
                term = np.float32(term * arrays["rvh"][j])
                cavg[k, 0] = np.float64(cavg[k, 0] + np.float64(term))
                term = np.float32(arrays["va"][k, j, i] * tem)
                term = np.float32(term * arrays["ruh"][i])
                term = np.float32(term * arrays["rvf"][j])
                cavg[k, 1] = np.float64(cavg[k, 1] + np.float64(term))
                theta = np.float32(arrays["th0"][k, j, i] + arrays["tha"][k, j, i])
                term = np.float32(theta * tem)
                term = np.float32(term * arrays["ruh"][i])
                term = np.float32(term * arrays["rvh"][j])
                cavg[k, 2] = np.float64(cavg[k, 2] + np.float64(term))
    x_extent = np.float32(
        _f32(np, "maxx_m", selection.maxx_m)
        - _f32(np, "minx_m", selection.minx_m)
    )
    y_extent = np.float32(
        _f32(np, "maxy_m", selection.maxy_m)
        - _f32(np, "miny_m", selection.miny_m)
    )
    temd = np.float64(1.0) / (np.float64(x_extent) * np.float64(y_extent))
    arrays["uavg"][...] = (cavg[:, 0] * temd).astype(np.float32)
    arrays["vavg"][...] = (cavg[:, 1] * temd).astype(np.float32)
    arrays["thavg"][...] = (cavg[:, 2] * temd).astype(np.float32)


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def cpu_rayleigh_damping_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: RayleighDampingSelection,
) -> RayleighDampingResult:
    """Build coefficients/averages and assign the selected Rayleigh tendencies."""

    np = importlib.import_module("numpy")
    effective_horizontal = _validate_selection(np, selection)
    if selection.horizontal_damping >= 1 and (ni <= 1 or nj <= 1):
        refuse(
            code="CM1-RAYLEIGH-SELECTION",
            gate="DYNAMICS-BRANCH-MATRIX",
            option="ni/nj",
            value=[ni, nj],
            reason="the bounded lateral operator owns the all-four-side 3D branch",
            remedy="use ni>1 and nj>1 or a separately audited reduced-dimensional path",
        )
    _validate_arrays(np, arrays, ni=ni, nj=nj, nk=nk)
    _build_coefficients(np, arrays, selection, ni=ni, nj=nj, nk=nk)
    average_branch = selection.upper_damping == 2 or effective_horizontal == 2
    scalar_difference = np.empty_like(arrays["thten1"])
    if average_branch:
        _horizontal_averages(np, arrays, selection, ni=ni, nj=nj, nk=nk)

    rdalpha = _f32(np, "rdalpha_s_inv", selection.rdalpha_s_inv)
    neg_half = np.float32(np.float32(-rdalpha) * np.float32(0.5))
    neg = np.float32(-rdalpha)
    u_coefficient = np.empty_like(arrays["uten1"])
    v_coefficient = np.empty_like(arrays["vten1"])
    np.add(arrays["tauh"][:, 1:, :-1], arrays["tauh"][:, 1:, 1:], out=u_coefficient)
    np.multiply(u_coefficient, neg_half, out=u_coefficient)
    np.add(arrays["tauh"][:, :-1, 1:], arrays["tauh"][:, 1:, 1:], out=v_coefficient)
    np.multiply(v_coefficient, neg_half, out=v_coefficient)

    if average_branch:
        np.subtract(arrays["ua"], arrays["uavg"][:, None, None], out=arrays["uten1"])
        np.multiply(arrays["uten1"], u_coefficient, out=arrays["uten1"])
        np.subtract(arrays["va"], arrays["vavg"][:, None, None], out=arrays["vten1"])
        np.multiply(arrays["vten1"], v_coefficient, out=arrays["vten1"])
        np.subtract(
            arrays["th0"],
            arrays["thavg"][:, None, None],
            out=scalar_difference,
        )
        np.add(scalar_difference, arrays["tha"], out=scalar_difference)
    else:
        np.subtract(arrays["ua"], arrays["u0"], out=arrays["uten1"])
        np.multiply(arrays["uten1"], u_coefficient, out=arrays["uten1"])
        np.subtract(arrays["va"], arrays["v0"], out=arrays["vten1"])
        np.multiply(arrays["vten1"], v_coefficient, out=arrays["vten1"])
        scalar_difference[...] = arrays["tha"]
    np.multiply(neg, arrays["tauf"][:nk, 1:, 1:], out=arrays["wten1"])
    np.multiply(arrays["wten1"], arrays["wa"], out=arrays["wten1"])
    np.multiply(neg, arrays["taus"][:, 1:, 1:], out=arrays["thten1"])
    np.multiply(arrays["thten1"], scalar_difference, out=arrays["thten1"])

    receipt: dict[str, object] = {
        "schema": CM1_RAYLEIGH_DAMPING_SCHEMA,
        "status": CM1_RAYLEIGH_DAMPING_STATUS,
        "claim_limit": CM1_RAYLEIGH_DAMPING_CLAIM_LIMIT,
        "authority": {
            "release": "CM1 r21.1",
            "tarball_sha256": CM1_R21_1_TARBALL_SHA256,
            "files": dict(CM1_RAYLEIGH_AUTHORITY_SHA256),
            "source_anchors": dict(CM1_RAYLEIGH_SOURCE_ANCHORS),
        },
        "selection": {
            "irdamp": selection.upper_damping,
            "hrdamp_input": selection.horizontal_damping,
            "hrdamp_effective": effective_horizontal,
            "tendency_reference": "horizontal-average" if average_branch else "base-state",
            "terrain_flag": selection.terrain_flag,
            "axisymmetric": selection.axisymmetric,
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "output_sha256": {
            name: _sha256_array(arrays[name])
            for name in (
                "tauh",
                "taus",
                "tauf",
                "uavg",
                "vavg",
                "thavg",
                "cavg",
                "uten1",
                "vten1",
                "wten1",
                "thten1",
            )
        },
        "libm_claim": (
            "native GNU Fortran raw-word identity is bounded to the committed "
            "all-active-configuration oracle deck"
        ),
    }
    receipt["operation_token_sha256"] = _canonical_sha256(receipt)
    return RayleighDampingResult(receipt=MappingProxyType(receipt))


__all__ = [
    "CM1_RAYLEIGH_AUTHORITY_SHA256",
    "CM1_RAYLEIGH_DAMPING_CLAIM_LIMIT",
    "CM1_RAYLEIGH_DAMPING_SCHEMA",
    "CM1_RAYLEIGH_DAMPING_STATUS",
    "CM1_RAYLEIGH_SOURCE_ANCHORS",
    "RayleighDampingResult",
    "RayleighDampingSelection",
    "cpu_rayleigh_damping_inplace",
    "rayleigh_damping_shapes",
]
