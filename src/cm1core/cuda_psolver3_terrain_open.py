"""CM1 r21.1 terrain + all-open ``psolver=3`` acoustic composition.

The CPU path in this module is an independent source-ordered implementation
of the terrain branch of ``sound`` composed with the sealed all-open boundary
lifecycle.  It covers the native 2/3/6 acoustic cadence, transformed terrain
pressure gradient/divergence/advection, the vertically implicit tridiagonal
solve, pressure update, velocity averages, radiation-history production, and
stage family fills.

This remains a bounded acoustic/RK composition gate.  Large-step momentum,
theta, and moisture tendencies are supplied by the caller and may be held for
operator validation.  The Rust initialized-product arena has no source halos
and retains a horizontally broadcast base-density policy; product admission
is therefore refused.  This module makes no forecast or tornado claim.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32
from cm1core.cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    cpu_bcp_open_inplace,
    cpu_bcs_open_inplace,
    cpu_bcu_open_inplace,
    cpu_bcv_open_inplace,
    cpu_bcw_open_inplace,
    cpu_radiation4_composed_inplace,
)
from cm1core.cuda_terrain_open_boundary import (
    DEFAULT_TERRAIN_OPEN_SELECTION,
    TERRAIN_STATIC_REQUIRED,
    TerrainOpenSelection,
    cpu_bcwsfc_terrain_physical_inplace,
    cpu_ssopen_terrain_all_inplace,
    terrain_coordinate_profiles,
    terrain_static_shapes,
    validate_terrain_open_selection,
)


CUDA_PSOLVER3_TERRAIN_OPEN_SCHEMA = "gpuwm-cm1.cuda-psolver3-terrain-open/v1"
CUDA_PSOLVER3_TERRAIN_OPEN_STAGE_SCHEMA = (
    "gpuwm-cm1.cuda-psolver3-terrain-open-stage/v1"
)
CM1_CP_F32 = 1005.7
# README.namelist:756-759 -- "A value of 0.5 is centered-in-time.  Slight
# forward-in-time bias is recommended.  Default value is 0.60."  sound.F:78-82
# forms beta=1-alph and asq=alph*alph from it, and all 23 of CM1's shipped
# decks set 0.60.  This module carried 0.5 -- exactly centred, so the
# vertically implicit acoustic step damped nothing -- until the runtime made
# it configurable; the default is now CM1's own, which changes every
# psolver=3 result this port produces.
CM1_ALPH_F32 = 0.60
CM1_KDIV_F32 = 0.1
CM1_SMEPS_F32 = 1.0e-20

SOURCE_ANCHORS = {
    "sound_schedule": "src/sound.F:81-113",
    "terrain_coefficients": "src/sound.F:145-188",
    "acoustic_stage_preparation": "src/sound.F:191-248",
    "open_consumption": "src/sound.F:253-302;src/bc.F:2392-2684",
    "terrain_pressure_gradient": "src/sound.F:364-410",
    "terrain_divergence": "src/sound.F:446-534",
    "terrain_pressure_advection": "src/sound.F:596-654;src/bc.F:1151-1178",
    "implicit_vertical_solve": "src/sound.F:659-730",
    "terrain_pressure_update": "src/sound.F:763-789",
    "pressure_fill_and_average": "src/sound.F:791-833",
    "held_final_ppx": "src/sound.F:842-853",
    "radiation_history": "src/solve2.F:1258-1271;src/bc.F:2054-2172",
    "stage_boundaries": "src/solve2.F:1665-1706",
}

CUDA_PSOLVER3_TERRAIN_OPEN_CLAIM_LIMIT = (
    "CM1 r21.1 serial terrain psolver=3 acoustic composition with all four "
    "open boundaries, native 2+3+6 small-step cadence, transformed pressure "
    "gradient/divergence/advection, implicit vertical solve, pressure update, "
    "held final-stage ppx, velocity averages, irbc=4 history lifecycle, and "
    "source-ordered family fills; large-step advection/physics tendencies, "
    "authority-valid terrain halos, terrain-consistent hydrostatic base "
    "admission, full trajectory, forecast, product, tornado, and performance "
    "claims are excluded"
)

CUDA_PSOLVER3_TERRAIN_OPEN_STAGE_CLAIM_LIMIT = (
    "one explicitly selected CM1 r21.1 terrain/all-open psolver=3 RK acoustic "
    "stage using the same native small-step, transformed-metric, implicit "
    "column, radiation-history, and family-boundary operators as the sealed "
    "three-stage composition; the caller must preserve A/current carriers, "
    "large-step tendencies, histories, and source cadence between stages; "
    "this API alone makes no full-step, physics, forecast, product, tornado, "
    "or performance claim"
)


PRESSURE_NAMES = (
    "ppi",
    "pp3d",
    "ppx",
    "ppd",
    "piadv",
    "ppterm",
    "ppten",
    "thv",
    "qk",
    "pk1",
    "pk2",
    "mm",
    "fpk",
    "divh",
    "hadvp",
    "dum3",
    "elim_c",
    "elim_d",
    "tri_aa",
    "tri_bb",
    "tri_cc",
    "tri_dd",
    "tri_pivot",
    "tri_residual",
)
U_NAMES = ("ua", "u3d", "uten", "rru")
V_NAMES = ("va", "v3d", "vten", "rrv")
W_NAMES = ("wa", "w3d", "wten", "rrw")
BOUNDARY_NAMES = (
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
    "uh",
    "vh",
    "ruf",
    "rvf",
)
ALL_ARRAY_NAMES = PRESSURE_NAMES + U_NAMES + V_NAMES + W_NAMES + BOUNDARY_NAMES + ("scalar",)
COMPARE_OUTPUT_NAMES = (
    "u3d",
    "v3d",
    "w3d",
    "pp3d",
    "ppd",
    "ppx",
    "rru",
    "rrv",
    "rrw",
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
    "tri_pivot",
    "tri_residual",
)


def terrain_open_sound_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    ng = CM1_OPEN_NGXY
    pressure = (nk + 2, nj + 2, ni + 2)
    u = (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng)
    v = (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng)
    w = (nk + 3, nj + 2 * ng, ni + 2 * ng)
    scalar = (nk + 2, nj + 2 * ng, ni + 2 * ng)
    shapes = {name: pressure for name in PRESSURE_NAMES}
    shapes.update({name: u for name in U_NAMES})
    shapes.update({name: v for name in V_NAMES})
    shapes.update({name: w for name in W_NAMES})
    shapes.update(
        {
            "radbcw": (nk + 2, nj + 2 * ng),
            "radbce": (nk + 2, nj + 2 * ng),
            "radbcs": (nk + 2, ni + 2 * ng),
            "radbcn": (nk + 2, ni + 2 * ng),
            "uh": (ni + 2 * ng,),
            "vh": (nj + 2 * ng,),
            "ruf": (ni + 1 + 2 * ng,),
            "rvf": (nj + 1 + 2 * ng,),
            "scalar": scalar,
        }
    )
    return shapes


def _require_arrays(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = terrain_open_sound_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing terrain/open sound arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must be a float32 numpy.ndarray")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must have C-order shape {shapes[name]}")
        if not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("terrain/open sound arrays must not alias")
    one = np.float32(1.0).view(np.uint32)
    for name in ("uh", "vh", "ruf", "rvf"):
        if not bool(np.all(result[name].view(np.uint32) == one)):
            raise ValueError(f"{name} must be exactly one for the uniform horizontal grid")
    return result


def _require_metrics(
    metrics: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    missing = [name for name in TERRAIN_STATIC_REQUIRED if name not in metrics]
    if missing:
        raise ValueError(f"missing terrain/open metrics: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in TERRAIN_STATIC_REQUIRED:
        value = metrics[name]
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"metric {name} must be a float32 numpy.ndarray")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"metric {name} must have C-order shape {shapes[name]}")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"metric {name} contains non-finite values")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("terrain/open metrics must not alias")
    return result


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _x(index: int) -> int:
    return index + CM1_OPEN_NGXY - 1


def _y(index: int) -> int:
    return index + CM1_OPEN_NGXY - 1


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


@dataclass(frozen=True, slots=True)
class TerrainOpenSoundAblations:
    open_consumption: bool = True
    terrain_pressure_gradient: bool = True
    transformed_divergence: bool = True
    pressure_advection: bool = True
    implicit_vertical_solve: bool = True
    between_step_pressure_fill: bool = True
    radiation_history_production: bool = True
    stage_family_fills: bool = True


@dataclass(frozen=True, slots=True)
class TerrainOpenSoundStageDiagnostics:
    nrk: int
    nloop: int
    dts_f32: float
    dttmp_f32: float
    open_consumptions: int
    pressure_fills: int
    radiation_productions: int
    max_tridiagonal_residual: float
    ppx_written: bool


def _prepare_coefficients(
    np: Any,
    a: MutableMapping[str, Any],
    m: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dts: Any,
    dz: float,
) -> None:
    profiles = terrain_coordinate_profiles(nk, dz)
    cp = np.float32(CM1_CP_F32)
    for name in (
        "qk",
        "pk1",
        "pk2",
        "mm",
        "fpk",
        "divh",
        "hadvp",
        "dum3",
        "elim_c",
        "elim_d",
        "tri_aa",
        "tri_bb",
        "tri_cc",
        "tri_dd",
        "tri_pivot",
        "tri_residual",
    ):
        a[name].fill(np.float32(0.0))
    for k in range(1, nk + 1):
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                tem = _mul(np, dts, profiles["rdsf"][k])
                a["qk"][k, j, i] = _mul(
                    np,
                    _mul(np, tem, a["ppterm"][k, j, i]),
                    m["gz"][j0, i0],
                )
                tem2 = _mul(
                    np,
                    _mul(np, _mul(np, dts, m["gz"][j0, i0]), profiles["rdsf"][k]),
                    m["rr0"][k - 1, j0, i0],
                )
                pk2bracket = _add(
                    np,
                    _mul(np, -m["c2"][k, j0, i0], a["piadv"][k + 1, j, i]),
                    _mul(
                        np,
                        _sub(np, np.float32(1.0), m["c1"][k, j0, i0]),
                        a["piadv"][k, j, i],
                    ),
                )
                a["pk2"][k, j, i] = _mul(
                    np, _mul(np, tem2, m["rf0"][k, j0, i0]), pk2bracket
                )
                pk1bracket = _add(
                    np,
                    _mul(np, m["c1"][k - 1, j0, i0], a["piadv"][k - 1, j, i]),
                    _mul(
                        np,
                        _sub(np, m["c2"][k - 1, j0, i0], np.float32(1.0)),
                        a["piadv"][k, j, i],
                    ),
                )
                a["pk1"][k, j, i] = _mul(
                    np,
                    _mul(np, tem2, m["rf0"][k - 1, j0, i0]),
                    pk1bracket,
                )
                if k >= 2:
                    tem1 = _mul(np, _mul(np, dts, profiles["rds"][k]), cp)
                    theta = _add(
                        np,
                        _mul(np, m["c2"][k - 1, j0, i0], a["thv"][k, j, i]),
                        _mul(np, m["c1"][k - 1, j0, i0], a["thv"][k - 1, j, i]),
                    )
                    a["mm"][k, j, i] = _mul(
                        np, _mul(np, tem1, theta), m["gz"][j0, i0]
                    )
                else:
                    a["dum3"][1, j, i] = np.float32(0.0)
                    a["dum3"][nk + 1, j, i] = np.float32(0.0)
                if k == 1:
                    a["pk1"][k, j, i] = np.float32(0.0)
                if k == nk:
                    a["pk2"][k, j, i] = np.float32(0.0)


def _terrain_pressure_gradient(
    np: Any,
    a: MutableMapping[str, Any],
    m: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dts: Any,
    rdx: Any,
    rdy: Any,
    dz: float,
) -> None:
    profiles = terrain_coordinate_profiles(nk, dz)
    vertical = a["hadvp"]
    vertical.fill(np.float32(0.0))
    for k in range(2, nk + 1):
        for j in range(0, nj + 2):
            for i in range(0, ni + 2):
                vertical[k, j, i] = _mul(
                    np,
                    _sub(np, a["ppd"][k, j, i], a["ppd"][k - 1, j, i]),
                    profiles["rds"][k],
                )
    halfcp = _mul(np, np.float32(CM1_CP_F32), np.float32(0.5))
    eighth = np.float32(0.125)
    for k in range(1, nk + 1):
        for j0 in range(nj):
            j = j0 + 1
            y = _y(j)
            for face in range(2, ni + 1):
                i0 = face - 1
                thsum = _add(np, a["thv"][k, j, face], a["thv"][k, j, face - 1])
                horizontal = _mul(
                    np,
                    _sub(np, a["ppd"][k, j, face], a["ppd"][k, j, face - 1]),
                    rdx,
                )
                vsum = _add(
                    np,
                    _add(np, vertical[k + 1, j, face], vertical[k + 1, j, face - 1]),
                    _add(np, vertical[k, j, face], vertical[k, j, face - 1]),
                )
                transformed = _mul(
                    np,
                    _mul(np, eighth, vsum),
                    _add(np, m["gxu"][k - 1, j0, i0], m["gxu"][k, j0, i0]),
                )
                pg = _mul(np, _mul(np, halfcp, thsum), _add(np, horizontal, transformed))
                update = _mul(
                    np,
                    dts,
                    _sub(np, a["uten"][k, y, _x(face)], pg),
                )
                a["u3d"][k, y, _x(face)] = _add(
                    np, a["u3d"][k, y, _x(face)], update
                )
        for face in range(2, nj + 1):
            j0 = face - 1
            y = _y(face)
            for i0 in range(ni):
                i = i0 + 1
                thsum = _add(np, a["thv"][k, face, i], a["thv"][k, face - 1, i])
                horizontal = _mul(
                    np,
                    _sub(np, a["ppd"][k, face, i], a["ppd"][k, face - 1, i]),
                    rdy,
                )
                vsum = _add(
                    np,
                    _add(np, vertical[k + 1, face, i], vertical[k + 1, face - 1, i]),
                    _add(np, vertical[k, face, i], vertical[k, face - 1, i]),
                )
                transformed = _mul(
                    np,
                    _mul(np, eighth, vsum),
                    _add(np, m["gyv"][k - 1, j0, i0], m["gyv"][k, j0, i0]),
                )
                pg = _mul(np, _mul(np, halfcp, thsum), _add(np, horizontal, transformed))
                update = _mul(
                    np,
                    dts,
                    _sub(np, a["vten"][k, y, _x(i)], pg),
                )
                a["v3d"][k, y, _x(i)] = _add(
                    np, a["v3d"][k, y, _x(i)], update
                )


def _transformed_velocity_terms(
    np: Any,
    a: MutableMapping[str, Any],
    m: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
) -> tuple[Any, Any]:
    profiles = terrain_coordinate_profiles(nk, dz)
    xu = np.zeros((nk + 2, nj, ni + 1), dtype=np.float32)
    yv = np.zeros((nk + 2, nj + 1, ni), dtype=np.float32)
    for k in range(1, nk + 1):
        for j0 in range(nj):
            y = _y(j0 + 1)
            for face in range(ni + 1):
                xu[k, j0, face] = _mul(
                    np, a["u3d"][k, y, _x(face + 1)], m["rgzu"][j0, face]
                )
        for face in range(nj + 1):
            y = _y(face + 1)
            for i0 in range(ni):
                yv[k, face, i0] = _mul(
                    np, a["v3d"][k, y, _x(i0 + 1)], m["rgzv"][face, i0]
                )
    a["dum3"].fill(np.float32(0.0))
    zt = np.float32(nk * dz)
    rzt = _div(np, np.float32(1.0), zt)
    half = np.float32(0.5)
    for kface in range(2, nk + 1):
        r2 = _mul(
            np,
            _sub(np, profiles["sigmaf"][kface], profiles["sigma"][kface - 1]),
            profiles["rds"][kface],
        )
        r1 = _sub(np, np.float32(1.0), r2)
        height = _mul(np, _sub(np, profiles["sigmaf"][kface], zt), rzt)
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                usum_hi = _add(np, xu[kface, j0, i0], xu[kface, j0, i0 + 1])
                usum_lo = _add(np, xu[kface - 1, j0, i0], xu[kface - 1, j0, i0 + 1])
                vsum_hi = _add(np, yv[kface, j0, i0], yv[kface, j0 + 1, i0])
                vsum_lo = _add(np, yv[kface - 1, j0, i0], yv[kface - 1, j0 + 1, i0])
                xterm = _mul(
                    np,
                    _add(np, _mul(np, r2, usum_hi), _mul(np, r1, usum_lo)),
                    m["dzdx"][j0, i0],
                )
                yterm = _mul(
                    np,
                    _add(np, _mul(np, r2, vsum_hi), _mul(np, r1, vsum_lo)),
                    m["dzdy"][j0, i0],
                )
                value = _mul(np, half, _add(np, xterm, yterm))
                value = _mul(np, _mul(np, value, height), m["gz"][j0, i0])
                a["dum3"][kface, j, i] = value
    return xu, yv


def _initial_fpk(
    np: Any,
    a: MutableMapping[str, Any],
    m: Mapping[str, Any],
    xu: Any,
    yv: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dts: Any,
    rdx: Any,
    rdy: Any,
    dz: float,
    beta: Any,
    transformed_divergence: bool,
) -> None:
    profiles = terrain_coordinate_profiles(nk, dz)
    smeps = np.float32(CM1_SMEPS_F32)
    for k in range(1, nk + 1):
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                div = np.float32(0.0)
                if transformed_divergence:
                    horizontal = _add(
                        np,
                        _mul(np, _sub(np, xu[k, j0, i0 + 1], xu[k, j0, i0]), rdx),
                        _mul(np, _sub(np, yv[k, j0 + 1, i0], yv[k, j0, i0]), rdy),
                    )
                    vertical = _mul(
                        np,
                        _sub(np, a["dum3"][k + 1, j, i], a["dum3"][k, j, i]),
                        profiles["rdsf"][k],
                    )
                    div = _mul(np, m["gz"][j0, i0], _add(np, horizontal, vertical))
                    if bool(np.abs(div) < smeps):
                        div = np.float32(0.0)
                a["divh"][k, j, i] = div
                w1 = np.float32(0.0) if k == 1 else a["w3d"][k, _y(j), _x(i)]
                w2 = np.float32(0.0) if k == nk else a["w3d"][k + 1, _y(j), _x(i)]
                source = _mul(
                    np,
                    dts,
                    _sub(np, a["ppten"][k, j, i], _mul(np, a["ppterm"][k, j, i], div)),
                )
                qpart = _mul(np, -a["qk"][k, j, i], _sub(np, w2, w1))
                a["fpk"][k, j, i] = _add(np, source, _mul(np, beta, qpart))


def _pressure_advection(
    np: Any,
    a: MutableMapping[str, Any],
    m: Mapping[str, Any],
    xu: Any,
    yv: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    dts: Any,
    rdx: Any,
    rdy: Any,
    dz: float,
    beta: Any,
) -> None:
    profiles = terrain_coordinate_profiles(nk, dz)
    # The interim base is an exact horizontal broadcast, so its source halo
    # extension is the same profile word.  A varying-base product must provide
    # explicit halo carriers instead of using this validation path.
    rho_profile = m["rho0_3d"][:, 0, 0]
    zt = np.float32(nk * dz)
    rzt = _div(np, np.float32(1.0), zt)
    half = np.float32(0.5)
    for k in range(1, nk + 1):
        density = rho_profile[k - 1]
        for j0 in range(nj):
            for face in range(ni + 1):
                xu[k, j0, face] = _mul(np, density, xu[k, j0, face])
        for face in range(nj + 1):
            for i0 in range(ni):
                yv[k, face, i0] = _mul(np, density, yv[k, face, i0])
    a["hadvp"].fill(np.float32(0.0))
    for kface in range(2, nk + 1):
        r2 = _mul(
            np,
            _sub(np, profiles["sigmaf"][kface], profiles["sigma"][kface - 1]),
            profiles["rds"][kface],
        )
        r1 = _sub(np, np.float32(1.0), r2)
        height = _mul(np, _sub(np, profiles["sigmaf"][kface], zt), rzt)
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                usum_hi = _add(np, xu[kface, j0, i0], xu[kface, j0, i0 + 1])
                usum_lo = _add(np, xu[kface - 1, j0, i0], xu[kface - 1, j0, i0 + 1])
                vsum_hi = _add(np, yv[kface, j0, i0], yv[kface, j0 + 1, i0])
                vsum_lo = _add(np, yv[kface - 1, j0, i0], yv[kface - 1, j0 + 1, i0])
                xterm = _mul(np, _add(np, _mul(np, r2, usum_hi), _mul(np, r1, usum_lo)), m["dzdx"][j0, i0])
                yterm = _mul(np, _add(np, _mul(np, r2, vsum_hi), _mul(np, r1, vsum_lo)), m["dzdy"][j0, i0])
                value = _mul(np, half, _add(np, xterm, yterm))
                value = _mul(np, _mul(np, _mul(np, value, height), m["gz"][j0, i0]), m["rrf0"][kface - 1, j0, i0])
                a["hadvp"][kface, j, i] = value
    temx = _mul(np, _mul(np, dts, np.float32(0.5)), rdx)
    temy = _mul(np, _mul(np, dts, np.float32(0.5)), rdy)
    for k in range(1, nk + 1):
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                u2 = _mul(np, temx, xu[k, j0, i0 + 1])
                u1 = _mul(np, temx, xu[k, j0, i0])
                v2 = _mul(np, temy, yv[k, j0 + 1, i0])
                v1 = _mul(np, temy, yv[k, j0, i0])
                w2 = _add(np, a["hadvp"][k + 1, j, i], _mul(np, beta, a["w3d"][k + 1, _y(j), _x(i)]))
                w1 = _add(np, a["hadvp"][k, j, i], _mul(np, beta, a["w3d"][k, _y(j), _x(i)]))
                xadv = _add(
                    np,
                    _mul(np, u2, _sub(np, a["piadv"][k, j, i + 1], a["piadv"][k, j, i])),
                    _mul(np, u1, _sub(np, a["piadv"][k, j, i], a["piadv"][k, j, i - 1])),
                )
                yadv = _add(
                    np,
                    _mul(np, v2, _sub(np, a["piadv"][k, j + 1, i], a["piadv"][k, j, i])),
                    _mul(np, v1, _sub(np, a["piadv"][k, j, i], a["piadv"][k, j - 1, i])),
                )
                horizontal = _mul(np, -_add(np, xadv, yadv), m["rr0"][k - 1, j0, i0])
                horizontal = _mul(np, horizontal, m["gz"][j0, i0])
                vertical = _add(np, _mul(np, w1, a["pk1"][k, j, i]), _mul(np, w2, a["pk2"][k, j, i]))
                a["fpk"][k, j, i] = _add(np, a["fpk"][k, j, i], _add(np, horizontal, vertical))


def _implicit_and_pressure_update(
    np: Any,
    a: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dts: Any,
    nrk: int,
    n: int,
    nloop: int,
    alph: Any,
    beta: Any,
    kdiv: Any,
    implicit: bool,
) -> None:
    asq = _mul(np, alph, alph)
    smeps = np.float32(CM1_SMEPS_F32)
    for j0 in range(nj):
        j = j0 + 1
        y = _y(j)
        for i0 in range(ni):
            i = i0 + 1
            x = _x(i)
            if implicit:
                for k in range(2, nk + 1):
                    aa = np.float32(0.0)
                    cc = np.float32(0.0)
                    if k >= 3:
                        aa = _mul(np, -asq, _mul(np, a["mm"][k, j, i], _add(np, a["pk1"][k - 1, j, i], a["qk"][k - 1, j, i])))
                    if k < nk:
                        cc = _mul(np, -asq, _mul(np, -a["mm"][k, j, i], _sub(np, a["pk2"][k, j, i], a["qk"][k, j, i])))
                    diagonal = _add(np, -_add(np, a["pk1"][k, j, i], a["qk"][k, j, i]), _sub(np, a["pk2"][k - 1, j, i], a["qk"][k - 1, j, i]))
                    bb = _sub(np, np.float32(1.0), _mul(np, asq, _mul(np, a["mm"][k, j, i], diagonal)))
                    pressure_delta = _sub(np, a["pp3d"][k, j, i], a["pp3d"][k - 1, j, i])
                    fwk = _add(np, _mul(np, dts, a["wten"][k, y, x]), _mul(np, beta, _mul(np, -a["mm"][k, j, i], pressure_delta)))
                    updated_delta = _sub(np, _add(np, a["pp3d"][k, j, i], a["fpk"][k, j, i]), _add(np, a["pp3d"][k - 1, j, i], a["fpk"][k - 1, j, i]))
                    dd = _add(np, _add(np, a["w3d"][k, y, x], fwk), _mul(np, alph, _mul(np, -a["mm"][k, j, i], updated_delta)))
                    pivot = bb if k == 2 else _add(np, _mul(np, aa, a["elim_c"][k - 1, j, i]), bb)
                    inverse = _div(np, np.float32(1.0), pivot)
                    for name, value in (("tri_aa", aa), ("tri_bb", bb), ("tri_cc", cc), ("tri_dd", dd), ("tri_pivot", pivot)):
                        a[name][k, j, i] = value
                    if k < nk:
                        a["elim_c"][k, j, i] = _mul(np, -cc, inverse)
                        numerator = dd if k == 2 else _sub(np, dd, _mul(np, aa, a["elim_d"][k - 1, j, i]))
                        a["elim_d"][k, j, i] = _mul(np, numerator, inverse)
                    else:
                        a["w3d"][k, y, x] = _mul(np, _sub(np, dd, _mul(np, aa, a["elim_d"][k - 1, j, i])), inverse)
                for k in range(nk - 1, 1, -1):
                    a["w3d"][k, y, x] = _add(np, _mul(np, a["elim_c"][k, j, i], a["w3d"][k + 1, y, x]), a["elim_d"][k, j, i])
                for k in range(2, nk + 1):
                    lhs = _add(np, _mul(np, a["tri_aa"][k, j, i], a["w3d"][k - 1, y, x]), _mul(np, a["tri_bb"][k, j, i], a["w3d"][k, y, x]))
                    lhs = _add(np, lhs, _mul(np, a["tri_cc"][k, j, i], a["w3d"][k + 1, y, x]))
                    a["tri_residual"][k, j, i] = _sub(np, lhs, a["tri_dd"][k, j, i])
            for k in range(1, nk + 1):
                w1 = np.float32(0.0) if k == 1 else a["w3d"][k, y, x]
                w2 = np.float32(0.0) if k == nk else a["w3d"][k + 1, y, x]
                old = a["pp3d"][k, j, i]
                vertical = _add(np, _mul(np, w1, _add(np, a["pk1"][k, j, i], a["qk"][k, j, i])), _mul(np, w2, _sub(np, a["pk2"][k, j, i], a["qk"][k, j, i])))
                updated = _add(np, _add(np, old, a["fpk"][k, j, i]), _mul(np, alph, vertical))
                if bool(np.abs(updated) < smeps):
                    updated = np.float32(0.0)
                a["pp3d"][k, j, i] = updated
                tendency = _mul(np, kdiv, _sub(np, updated, old))
                a["elim_c"][k, j, i] = tendency
                a["ppd"][k, j, i] = _add(np, updated, tendency)
                if nrk == 3 and n == nloop:
                    a["ppx"][k, j, i] = tendency


def _average_velocities(
    np: Any,
    a: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    n: int,
    nloop: int,
    alph: Any,
) -> None:
    tavg = _div(np, np.float32(1.0), np.float32(nloop))
    for k in range(1, nk + 1):
        for j in range(1, nj + 2):
            y = _y(j)
            for i in range(1, ni + 2):
                x = _x(i)
                value = _add(np, a["rru"][k, y, x], a["u3d"][k, y, x])
                a["rru"][k, y, x] = value if n < nloop else _mul(np, value, tavg)
                value = _add(np, a["rrv"][k, y, x], a["v3d"][k, y, x])
                a["rrv"][k, y, x] = value if n < nloop else _mul(np, value, tavg)
                addw = a["w3d"][k, y, x] if n < nloop else _mul(np, a["w3d"][k, y, x], alph)
                value = _add(np, a["rrw"][k, y, x], addw)
                a["rrw"][k, y, x] = value if n < nloop else _mul(np, value, tavg)


def cpu_psolver3_terrain_open_stage_inplace(
    arrays: MutableMapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nsound: int,
    nrk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: TerrainOpenSelection = DEFAULT_TERRAIN_OPEN_SELECTION,
    ablations: TerrainOpenSoundAblations | None = None,
    alph: float | None = None,
    kdiv: float | None = None,
) -> TerrainOpenSoundStageDiagnostics:
    """Execute one source-ordered terrain/open acoustic RK stage on CPU.

    ``alph`` and ``kdiv`` are CM1's ``param3`` reals (README.namelist:751-759).
    ``None`` binds CM1's own defaults, 0.60 and 0.10.
    """

    validate_terrain_open_selection(selection)
    dt, dx, dy, dz = (
        _positive("dt", dt),
        _positive("dx", dx),
        _positive("dy", dy),
        _positive("dz", dz),
    )
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    if ablations is None:
        ablations = TerrainOpenSoundAblations()
    if not isinstance(ablations, TerrainOpenSoundAblations):
        raise TypeError("ablations must be TerrainOpenSoundAblations")
    np = importlib.import_module("numpy")
    a = _require_arrays(arrays, ni=ni, nj=nj, nk=nk)
    m = _require_metrics(metrics, ni=ni, nj=nj, nk=nk)
    nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
    dts = np.float32(dts_value)
    alph = np.float32(CM1_ALPH_F32 if alph is None else alph)
    kdiv = np.float32(CM1_KDIV_F32 if kdiv is None else kdiv)
    beta = _sub(np, np.float32(1.0), alph)
    rdx = _div(np, np.float32(1.0), np.float32(dx))
    rdy = _div(np, np.float32(1.0), np.float32(dy))
    _prepare_coefficients(np, a, m, ni=ni, nj=nj, nk=nk, dts=dts, dz=dz)
    a["ppd"][1 : nk + 1] = a["ppx"][1 : nk + 1]
    if nrk != 1:
        a["pp3d"][1 : nk + 1] = a["ppi"][1 : nk + 1]
    for k in range(1, nk + 1):
        for j in range(0, nj + 2):
            for i in range(0, ni + 2):
                y, x = _y(j), _x(i)
                a["rru"][k, y, x] = np.float32(0.0)
                a["rrv"][k, y, x] = np.float32(0.0)
                a["rrw"][k, _y(j), _x(i)] = _mul(
                    np, a["wa"][k, _y(j), _x(i)], beta
                )
                if nrk != 1:
                    a["u3d"][k, y, x] = a["ua"][k, y, x]
                    a["v3d"][k, y, x] = a["va"][k, y, x]
                    a["w3d"][k, y, x] = a["wa"][k, y, x]
    for n in range(1, nloop + 1):
        if ablations.open_consumption:
            cpu_ssopen_terrain_all_inplace(
                a,
                m,
                ni=ni,
                nj=nj,
                nk=nk,
                dts=float(dts),
                dx=dx,
                dy=dy,
                dz=dz,
                apply_bcwsfc=False,
                selection=selection,
            )
        if ablations.terrain_pressure_gradient:
            _terrain_pressure_gradient(np, a, m, ni=ni, nj=nj, nk=nk, dts=dts, rdx=rdx, rdy=rdy, dz=dz)
        xu, yv = _transformed_velocity_terms(np, a, m, ni=ni, nj=nj, nk=nk, dz=dz)
        _initial_fpk(np, a, m, xu, yv, ni=ni, nj=nj, nk=nk, dts=dts, rdx=rdx, rdy=rdy, dz=dz, beta=beta, transformed_divergence=ablations.transformed_divergence)
        cpu_bcwsfc_terrain_physical_inplace(a, m, ni=ni, nj=nj, nk=nk)
        if ablations.pressure_advection:
            _pressure_advection(np, a, m, xu, yv, ni=ni, nj=nj, nk=nk, dts=dts, rdx=rdx, rdy=rdy, dz=dz, beta=beta)
        _implicit_and_pressure_update(np, a, ni=ni, nj=nj, nk=nk, dts=dts, nrk=nrk, n=n, nloop=nloop, alph=alph, beta=beta, kdiv=kdiv, implicit=ablations.implicit_vertical_solve)
        if n < nloop and ablations.between_step_pressure_fill:
            cpu_bcp_open_inplace(a["ppd"], ni=ni, nj=nj, nk=nk)
        _average_velocities(np, a, ni=ni, nj=nj, nk=nk, n=n, nloop=nloop, alph=alph)
    if ablations.radiation_history_production:
        cpu_radiation4_composed_inplace(
            a,
            ni=ni,
            nj=nj,
            nk=nk,
            dx=dx,
            dy=dy,
            dttmp=rk_stage_dt_f32(dt, nrk),
        )
    if ablations.stage_family_fills:
        cpu_bcu_open_inplace(a["u3d"], ni=ni, nj=nj, nk=nk)
        cpu_bcv_open_inplace(a["v3d"], ni=ni, nj=nj, nk=nk)
        cpu_bcw_open_inplace(a["w3d"], ni=ni, nj=nj, nk=nk, flag=1)
        cpu_bcwsfc_terrain_physical_inplace(a, m, ni=ni, nj=nj, nk=nk)
        if nrk < 3:
            cpu_bcp_open_inplace(a["pp3d"], ni=ni, nj=nj, nk=nk)
            cpu_bcs_open_inplace(a["scalar"], ni=ni, nj=nj, nk=nk)
    residual = a["tri_residual"][2 : nk + 1, 1 : nj + 1, 1 : ni + 1]
    return TerrainOpenSoundStageDiagnostics(
        nrk,
        nloop,
        float(dts),
        rk_stage_dt_f32(dt, nrk),
        nloop if ablations.open_consumption else 0,
        nloop - 1 if ablations.between_step_pressure_fill else 0,
        1 if ablations.radiation_history_production else 0,
        float(np.max(np.abs(residual), initial=np.float32(0.0))),
        nrk == 3,
    )


@dataclass(frozen=True, slots=True)
class CpuPsolver3TerrainOpenResult:
    cadence: tuple[TerrainOpenSoundStageDiagnostics, ...]


def cpu_psolver3_terrain_open_three_stage_inplace(
    arrays: MutableMapping[str, Any], metrics: Mapping[str, Any], **kwargs: Any
) -> CpuPsolver3TerrainOpenResult:
    """Execute CM1's three source-ordered acoustic RK stages."""

    return CpuPsolver3TerrainOpenResult(
        tuple(
            cpu_psolver3_terrain_open_stage_inplace(
                arrays, metrics, nrk=nrk, **kwargs
            )
            for nrk in (1, 2, 3)
        )
    )


def compare_psolver3_terrain_open_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every declared output with raw-bit/no-null semantics."""

    np = importlib.import_module("numpy")
    fields: dict[str, Any] = {}
    for name in COMPARE_OUTPUT_NAMES:
        if name not in expected or name not in candidate:
            raise ValueError(f"missing terrain/open comparison output {name}")
        left, right = expected[name], candidate[name]
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError(f"{name} comparison values must be numpy.ndarray")
        if left.dtype != np.dtype("float32") or right.dtype != np.dtype("float32") or left.shape != right.shape:
            raise ValueError(f"{name} comparison dtype/shape mismatch")
        exact = bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))
        fields[name] = {
            "exact": exact,
            "expected_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest(),
            "candidate_sha256": hashlib.sha256(right.tobytes(order="C")).hexdigest(),
        }
    return {
        "mode": "raw-bit",
        "max_ulp": 0,
        "null_or_missing_is_pass": False,
        "passed": all(item["exact"] for item in fields.values()),
        "fields": fields,
    }


CUDA_PSOLVER3_TERRAIN_OPEN_KERNEL_NAMES = (
    "cm1_tos_prepare_v1",
    "cm1_tos_pgrad_v1",
    "cm1_tos_column_v1",
    "cm1_tos_average_v1",
)


CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE = r'''// gpuwm-cm1.cuda-psolver3-terrain-open-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long pidx(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long idx3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ float sig(unsigned int k,float dz){return m32(s32((float)k,0.5f),dz);}
__device__ __forceinline__ float sigf(unsigned int k,float dz){return m32((float)(k-1u),dz);}
__device__ __forceinline__ float rds(float dz){return d32(1.0f,dz);}
__device__ __forceinline__ float xu(const float* u,const float* rgzu,unsigned int k,unsigned int j,unsigned int f,unsigned int ni,unsigned int nj,unsigned int ng){return m32(u[idx3(k,j+ng,f+ng, nj+2u*ng,ni+1u+2u*ng)],rgzu[(unsigned long long)j*(ni+1u)+f]);}
__device__ __forceinline__ float yv(const float* v,const float* rgzv,unsigned int k,unsigned int f,unsigned int i,unsigned int ni,unsigned int nj,unsigned int ng){return m32(v[idx3(k,f+ng,i+ng,nj+1u+2u*ng,ni+2u*ng)],rgzv[(unsigned long long)f*ni+i]);}
__device__ __forceinline__ float d3div(const float* u,const float* v,const float* rgzu,const float* rgzv,const float* dzdx,const float* dzdy,const float* gz,unsigned int kf,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){
 if(kf==1u||kf==nk+1u)return 0.0f;float r2=m32(s32(sigf(kf,dz),sig(kf-1u,dz)),rds(dz)),r1=s32(1.0f,r2);
 float uh=a32(xu(u,rgzu,kf,j,i,ni,nj,ng),xu(u,rgzu,kf,j,i+1u,ni,nj,ng)),ul=a32(xu(u,rgzu,kf-1u,j,i,ni,nj,ng),xu(u,rgzu,kf-1u,j,i+1u,ni,nj,ng));
 float vh=a32(yv(v,rgzv,kf,j,i,ni,nj,ng),yv(v,rgzv,kf,j+1u,i,ni,nj,ng)),vl=a32(yv(v,rgzv,kf-1u,j,i,ni,nj,ng),yv(v,rgzv,kf-1u,j+1u,i,ni,nj,ng));
 float xt=m32(a32(m32(r2,uh),m32(r1,ul)),dzdx[(unsigned long long)j*ni+i]),yt=m32(a32(m32(r2,vh),m32(r1,vl)),dzdy[(unsigned long long)j*ni+i]);
 float zt=m32((float)nk,dz),height=m32(s32(sigf(kf,dz),zt),d32(1.0f,zt));return m32(m32(m32(0.5f,a32(xt,yt)),height),gz[(unsigned long long)j*ni+i]);
}
__device__ __forceinline__ float xurho(const float* u,const float* rgzu,const float* rho,unsigned int k,unsigned int j,unsigned int f,unsigned int ni,unsigned int nj,unsigned int ng){return m32(rho[(unsigned long long)(k-1u)*nj*ni],xu(u,rgzu,k,j,f,ni,nj,ng));}
__device__ __forceinline__ float yvrho(const float* v,const float* rgzv,const float* rho,unsigned int k,unsigned int f,unsigned int i,unsigned int ni,unsigned int nj,unsigned int ng){return m32(rho[(unsigned long long)(k-1u)*nj*ni],yv(v,rgzv,k,f,i,ni,nj,ng));}
__device__ __forceinline__ float d3adv(const float* u,const float* v,const float* rgzu,const float* rgzv,const float* rho,const float* rrf0,const float* dzdx,const float* dzdy,const float* gz,unsigned int kf,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dz){
 if(kf==1u||kf==nk+1u)return 0.0f;float r2=m32(s32(sigf(kf,dz),sig(kf-1u,dz)),rds(dz)),r1=s32(1.0f,r2);
 float uh=a32(xurho(u,rgzu,rho,kf,j,i,ni,nj,ng),xurho(u,rgzu,rho,kf,j,i+1u,ni,nj,ng)),ul=a32(xurho(u,rgzu,rho,kf-1u,j,i,ni,nj,ng),xurho(u,rgzu,rho,kf-1u,j,i+1u,ni,nj,ng));
 float vh=a32(yvrho(v,rgzv,rho,kf,j,i,ni,nj,ng),yvrho(v,rgzv,rho,kf,j+1u,i,ni,nj,ng)),vl=a32(yvrho(v,rgzv,rho,kf-1u,j,i,ni,nj,ng),yvrho(v,rgzv,rho,kf-1u,j+1u,i,ni,nj,ng));
 float xt=m32(a32(m32(r2,uh),m32(r1,ul)),dzdx[(unsigned long long)j*ni+i]),yt=m32(a32(m32(r2,vh),m32(r1,vl)),dzdy[(unsigned long long)j*ni+i]);float zt=m32((float)nk,dz),height=m32(s32(sigf(kf,dz),zt),d32(1.0f,zt));
 return m32(m32(m32(m32(0.5f,a32(xt,yt)),height),gz[(unsigned long long)j*ni+i]),rrf0[((unsigned long long)(kf-1u)*nj+j)*ni+i]);
}
extern "C" __global__ void cm1_tos_prepare_v1(
 const float* ua,const float* va,const float* wa,const float* ppi,const float* ppx,
 const float* rr0,const float* rf0,const float* c1,const float* c2,const float* gz,
 const float* piadv,const float* ppterm,const float* thv,float* u,float* v,float* w,float* pp3d,float* ppd,float* rru,float* rrv,float* rrw,
 float* qk,float* pk1,float* pk2,float* mm,float* fpk,float* divh,float* hadvp,float* dum3,float* ec,float* ed,float* aa,float* bb,float* cc,float* dd,float* pivot,float* residual,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk,float dts,float dz,float cp,float beta){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/(ni*nj))+1u,pj=j+1u,pi=i+1u;unsigned long long p=pidx(k,pj,pi,ni,nj),mi=((unsigned long long)(k-1u)*nj+j)*ni+i;
  float tem=m32(dts,rds(dz));qk[p]=m32(m32(tem,ppterm[p]),gz[(unsigned long long)j*ni+i]);float tem2=m32(m32(m32(dts,gz[(unsigned long long)j*ni+i]),rds(dz)),rr0[mi]);
  float b2=a32(m32(-c2[((unsigned long long)k*nj+j)*ni+i],piadv[pidx(k+1u,pj,pi,ni,nj)]),m32(s32(1.0f,c1[((unsigned long long)k*nj+j)*ni+i]),piadv[p]));pk2[p]=m32(m32(tem2,rf0[((unsigned long long)k*nj+j)*ni+i]),b2);
  float b1=a32(m32(c1[((unsigned long long)(k-1u)*nj+j)*ni+i],piadv[pidx(k-1u,pj,pi,ni,nj)]),m32(s32(c2[((unsigned long long)(k-1u)*nj+j)*ni+i],1.0f),piadv[p]));pk1[p]=m32(m32(tem2,rf0[((unsigned long long)(k-1u)*nj+j)*ni+i]),b1);
  float tem1=m32(m32(dts,rds(dz)),cp);mm[p]=(k>=2u)?m32(m32(tem1,a32(m32(c2[((unsigned long long)(k-1u)*nj+j)*ni+i],thv[p]),m32(c1[((unsigned long long)(k-1u)*nj+j)*ni+i],thv[pidx(k-1u,pj,pi,ni,nj)]))),gz[(unsigned long long)j*ni+i]):0.0f;if(k==1u)pk1[p]=0.0f;if(k==nk)pk2[p]=0.0f;
  fpk[p]=divh[p]=hadvp[p]=dum3[p]=ec[p]=ed[p]=aa[p]=bb[p]=cc[p]=dd[p]=pivot[p]=residual[p]=0.0f;
 }
 const unsigned long long np=(unsigned long long)nk*(nj+2u)*(ni+2u),ps=(unsigned long long)(nj+2u)*(ni+2u);
 for(unsigned long long z=first;z<np;z+=step){unsigned long long p=ps+z;ppd[p]=ppx[p];if(nrk!=1u)pp3d[p]=ppi[p];}
 unsigned long long nu=(unsigned long long)(nk+2u)*nyu*nxu,nv=(unsigned long long)(nk+2u)*nyv*nxv,nw=(unsigned long long)(nk+3u)*nyw*nxw;
 for(unsigned long long z=first;z<nu;z+=step){unsigned int ix=(unsigned int)(z%nxu);unsigned long long qq=z/nxu;unsigned int iy=(unsigned int)(qq%nyu),kk=(unsigned int)(qq/nyu);bool live=(kk>=1u&&kk<=nk&&iy+1u>=ng&&iy<ng+nj+1u&&ix+1u>=ng&&ix<ng+ni+1u);if(live){rru[z]=0.0f;if(nrk!=1u)u[z]=ua[z];}}for(unsigned long long z=first;z<nv;z+=step){unsigned int ix=(unsigned int)(z%nxv);unsigned long long qq=z/nxv;unsigned int iy=(unsigned int)(qq%nyv),kk=(unsigned int)(qq/nyv);bool live=(kk>=1u&&kk<=nk&&iy+1u>=ng&&iy<ng+nj+1u&&ix+1u>=ng&&ix<ng+ni+1u);if(live){rrv[z]=0.0f;if(nrk!=1u)v[z]=va[z];}}for(unsigned long long z=first;z<nw;z+=step){unsigned int ix=(unsigned int)(z%nxw);unsigned long long qq=z/nxw;unsigned int iy=(unsigned int)(qq%nyw),kk=(unsigned int)(qq/nyw);bool live=(kk>=1u&&kk<=nk&&iy+1u>=ng&&iy<ng+nj+1u&&ix+1u>=ng&&ix<ng+ni+1u);if(live){rrw[z]=m32(wa[z],beta);if(nrk!=1u)w[z]=wa[z];}}
}
extern "C" __global__ void cm1_tos_pgrad_v1(const float* ppd,const float* thv,const float* uten,const float* vten,const float* gxu,const float* gyv,float* u,float* v,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,float dts,float rdx,float rdy,float dz,float cp){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;unsigned long long nu=(unsigned long long)nk*nj*(ni-1u),nv=(unsigned long long)nk*(nj-1u)*ni,total=nu+nv;float halfcp=m32(cp,0.5f),eighth=0.125f;
 for(unsigned long long z=first;z<total;z+=step){if(z<nu){unsigned int ff=(unsigned int)(z%(ni-1u))+2u;unsigned long long q=z/(ni-1u);unsigned int j=(unsigned int)(q%nj)+1u,k=(unsigned int)(q/nj)+1u,j0=j-1u,i0=ff-1u;float h=m32(s32(ppd[pidx(k,j,ff,ni,nj)],ppd[pidx(k,j,ff-1u,ni,nj)]),rdx);float dph=(k<nk)?m32(s32(ppd[pidx(k+1u,j,ff,ni,nj)],ppd[pidx(k,j,ff,ni,nj)]),rds(dz)):0.0f,dphl=(k<nk)?m32(s32(ppd[pidx(k+1u,j,ff-1u,ni,nj)],ppd[pidx(k,j,ff-1u,ni,nj)]),rds(dz)):0.0f,dpl=(k>1u)?m32(s32(ppd[pidx(k,j,ff,ni,nj)],ppd[pidx(k-1u,j,ff,ni,nj)]),rds(dz)):0.0f,dpll=(k>1u)?m32(s32(ppd[pidx(k,j,ff-1u,ni,nj)],ppd[pidx(k-1u,j,ff-1u,ni,nj)]),rds(dz)):0.0f;float vs=a32(a32(dph,dphl),a32(dpl,dpll));float tr=m32(m32(eighth,vs),a32(gxu[((unsigned long long)(k-1u)*nj+j0)*(ni+1u)+i0],gxu[((unsigned long long)k*nj+j0)*(ni+1u)+i0]));float pg=m32(m32(halfcp,a32(thv[pidx(k,j,ff,ni,nj)],thv[pidx(k,j,ff-1u,ni,nj)])),a32(h,tr));unsigned long long x=idx3(k,j0+ng,i0+ng,nyu,nxu);u[x]=a32(u[x],m32(dts,s32(uten[x],pg)));}
  else{unsigned long long zz=z-nu;unsigned int i=(unsigned int)(zz%ni)+1u;unsigned long long q=zz/ni;unsigned int ff=(unsigned int)(q%(nj-1u))+2u,k=(unsigned int)(q/(nj-1u))+1u,j0=ff-1u,i0=i-1u;float h=m32(s32(ppd[pidx(k,ff,i,ni,nj)],ppd[pidx(k,ff-1u,i,ni,nj)]),rdy);float dph=(k<nk)?m32(s32(ppd[pidx(k+1u,ff,i,ni,nj)],ppd[pidx(k,ff,i,ni,nj)]),rds(dz)):0.0f,dphl=(k<nk)?m32(s32(ppd[pidx(k+1u,ff-1u,i,ni,nj)],ppd[pidx(k,ff-1u,i,ni,nj)]),rds(dz)):0.0f,dpl=(k>1u)?m32(s32(ppd[pidx(k,ff,i,ni,nj)],ppd[pidx(k-1u,ff,i,ni,nj)]),rds(dz)):0.0f,dpll=(k>1u)?m32(s32(ppd[pidx(k,ff-1u,i,ni,nj)],ppd[pidx(k-1u,ff-1u,i,ni,nj)]),rds(dz)):0.0f;float vs=a32(a32(dph,dphl),a32(dpl,dpll));float tr=m32(m32(eighth,vs),a32(gyv[((unsigned long long)(k-1u)*(nj+1u)+j0)*ni+i0],gyv[((unsigned long long)k*(nj+1u)+j0)*ni+i0]));float pg=m32(m32(halfcp,a32(thv[pidx(k,ff,i,ni,nj)],thv[pidx(k,ff-1u,i,ni,nj)])),a32(h,tr));unsigned long long x=idx3(k,j0+ng,i0+ng,nyv,nxv);v[x]=a32(v[x],m32(dts,s32(vten[x],pg)));}}
}
extern "C" __global__ void cm1_tos_column_v1(
 const float* piadv,const float* ppterm,const float* ppten,const float* wten,const float* rho,const float* rr0,const float* rrf0,const float* rf0,const float* c1,const float* c2,const float* gz,const float* rgzu,const float* rgzv,const float* dzdx,const float* dzdy,
 const float* u,const float* v,float* w,float* pp3d,float* ppd,float* ppx,float* qk,float* pk1,float* pk2,float* mm,float* fpk,float* divh,float* hadvp,float* ec,float* ed,float* aa0,float* bb0,float* cc0,float* dd0,float* piv,float* res,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk,unsigned int nstep,unsigned int nloop,float dts,float rdx,float rdy,float dz,float alph,float beta,float kdiv,float smeps,unsigned int do_initial,unsigned int do_div,unsigned int do_adv,unsigned int do_implicit,unsigned int do_update){
 unsigned int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=ni*nj)return;unsigned int j=col/ni,i=col%ni,pj=j+1u,pi=i+1u,y=j+ng,x=i+ng;unsigned int nxw=ni+2u*ng,nyw=nj+2u*ng;float asq=m32(alph,alph),temx=m32(m32(dts,0.5f),rdx),temy=m32(m32(dts,0.5f),rdy);
 for(unsigned int k=1u;k<=nk;++k){unsigned long long p=pidx(k,pj,pi,ni,nj);if(do_initial){float div=0.0f;if(do_div){float h=a32(m32(s32(xu(u,rgzu,k,j,i+1u,ni,nj,ng),xu(u,rgzu,k,j,i,ni,nj,ng)),rdx),m32(s32(yv(v,rgzv,k,j+1u,i,ni,nj,ng),yv(v,rgzv,k,j,i,ni,nj,ng)),rdy));float vv=m32(s32(d3div(u,v,rgzu,rgzv,dzdx,dzdy,gz,k+1u,j,i,ni,nj,nk,ng,dz),d3div(u,v,rgzu,rgzv,dzdx,dzdy,gz,k,j,i,ni,nj,nk,ng,dz)),rds(dz));div=m32(gz[(unsigned long long)j*ni+i],a32(h,vv));if(fabsf(div)<smeps)div=0.0f;}divh[p]=div;float w1=(k==1u)?0.0f:w[idx3(k,y,x,nyw,nxw)],w2=(k==nk)?0.0f:w[idx3(k+1u,y,x,nyw,nxw)];fpk[p]=a32(m32(dts,s32(ppten[p],m32(ppterm[p],div))),m32(beta,m32(-qk[p],s32(w2,w1))));}
  if(do_adv){float u2=m32(temx,xurho(u,rgzu,rho,k,j,i+1u,ni,nj,ng)),u1=m32(temx,xurho(u,rgzu,rho,k,j,i,ni,nj,ng)),v2=m32(temy,yvrho(v,rgzv,rho,k,j+1u,i,ni,nj,ng)),v1=m32(temy,yvrho(v,rgzv,rho,k,j,i,ni,nj,ng));float ww2=a32(d3adv(u,v,rgzu,rgzv,rho,rrf0,dzdx,dzdy,gz,k+1u,j,i,ni,nj,nk,ng,dz),m32(beta,w[idx3(k+1u,y,x,nyw,nxw)])),ww1=a32(d3adv(u,v,rgzu,rgzv,rho,rrf0,dzdx,dzdy,gz,k,j,i,ni,nj,nk,ng,dz),m32(beta,w[idx3(k,y,x,nyw,nxw)]));float xa=a32(m32(u2,s32(piadv[pidx(k,pj,pi+1u,ni,nj)],piadv[p])),m32(u1,s32(piadv[p],piadv[pidx(k,pj,pi-1u,ni,nj)]))),ya=a32(m32(v2,s32(piadv[pidx(k,pj+1u,pi,ni,nj)],piadv[p])),m32(v1,s32(piadv[p],piadv[pidx(k,pj-1u,pi,ni,nj)])));float ha=m32(m32(-a32(xa,ya),rr0[((unsigned long long)(k-1u)*nj+j)*ni+i]),gz[(unsigned long long)j*ni+i]);fpk[p]=a32(fpk[p],a32(ha,a32(m32(ww1,pk1[p]),m32(ww2,pk2[p]))));}}
 if(do_implicit){for(unsigned int k=2u;k<=nk;++k){unsigned long long p=pidx(k,pj,pi,ni,nj),pb=pidx(k-1u,pj,pi,ni,nj);float aa=(k>=3u)?m32(-asq,m32(mm[p],a32(pk1[pb],qk[pb]))):0.0f,cc=(k<nk)?m32(-asq,m32(-mm[p],s32(pk2[p],qk[p]))):0.0f,diag=a32(-a32(pk1[p],qk[p]),s32(pk2[pb],qk[pb])),bb=s32(1.0f,m32(asq,m32(mm[p],diag))),pd=s32(pp3d[p],pp3d[pb]),fw=a32(m32(dts,wten[idx3(k,y,x,nyw,nxw)]),m32(beta,m32(-mm[p],pd))),upd=s32(a32(pp3d[p],fpk[p]),a32(pp3d[pb],fpk[pb])),dd=a32(a32(w[idx3(k,y,x,nyw,nxw)],fw),m32(alph,m32(-mm[p],upd))),pivot=(k==2u)?bb:a32(m32(aa,ec[pb]),bb),inv=d32(1.0f,pivot);aa0[p]=aa;bb0[p]=bb;cc0[p]=cc;dd0[p]=dd;piv[p]=pivot;if(k<nk){ec[p]=m32(-cc,inv);ed[p]=m32((k==2u)?dd:s32(dd,m32(aa,ed[pb])),inv);}else w[idx3(k,y,x,nyw,nxw)]=m32(s32(dd,m32(aa,ed[pb])),inv);}for(int k=(int)nk-1;k>=2;--k){unsigned long long p=pidx((unsigned int)k,pj,pi,ni,nj);w[idx3((unsigned int)k,y,x,nyw,nxw)]=a32(m32(ec[p],w[idx3((unsigned int)k+1u,y,x,nyw,nxw)]),ed[p]);}for(unsigned int k=2u;k<=nk;++k){unsigned long long p=pidx(k,pj,pi,ni,nj);float lhs=a32(m32(aa0[p],w[idx3(k-1u,y,x,nyw,nxw)]),m32(bb0[p],w[idx3(k,y,x,nyw,nxw)]));lhs=a32(lhs,m32(cc0[p],w[idx3(k+1u,y,x,nyw,nxw)]));res[p]=s32(lhs,dd0[p]);}}
 if(do_update){for(unsigned int k=1u;k<=nk;++k){unsigned long long p=pidx(k,pj,pi,ni,nj);float w1=(k==1u)?0.0f:w[idx3(k,y,x,nyw,nxw)],w2=(k==nk)?0.0f:w[idx3(k+1u,y,x,nyw,nxw)],old=pp3d[p],vert=a32(m32(w1,a32(pk1[p],qk[p])),m32(w2,s32(pk2[p],qk[p]))),up=a32(a32(old,fpk[p]),m32(alph,vert));if(fabsf(up)<smeps)up=0.0f;pp3d[p]=up;float tend=m32(kdiv,s32(up,old));ec[p]=tend;ppd[p]=a32(up,tend);if(nrk==3u&&nstep==nloop)ppx[p]=tend;}}
}
extern "C" __global__ void cm1_tos_average_v1(const float* u,const float* v,const float* w,float* rru,float* rrv,float* rrw,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nstep,unsigned int nloop,float alph){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,total=(unsigned long long)nk*(nj+1u)*(ni+1u);unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;float tavg=d32(1.0f,(float)nloop);
 for(unsigned long long z=first;z<total;z+=step){unsigned int i=(unsigned int)(z%(ni+1u));unsigned long long q=z/(ni+1u);unsigned int j=(unsigned int)(q%(nj+1u)),k=(unsigned int)(q/(nj+1u))+1u;unsigned long long xu0=idx3(k,j+ng,i+ng,nyu,nxu),xv0=idx3(k,j+ng,i+ng,nyv,nxv),xw0=idx3(k,j+ng,i+ng,nyw,nxw);float vu=a32(rru[xu0],u[xu0]),vv=a32(rrv[xv0],v[xv0]),add=(nstep<nloop)?w[xw0]:m32(w[xw0],alph),vw=a32(rrw[xw0],add);rru[xu0]=(nstep<nloop)?vu:m32(vu,tavg);rrv[xv0]=(nstep<nloop)?vv:m32(vv,tavg);rrw[xw0]=(nstep<nloop)?vw:m32(vw,tavg);}
}
'''

CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_BYTES = CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE.encode("utf-8")
CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256 = (
    "611be85d02d9f714bea3f126199bab0dc71c52bd0c1968e18f76c9f3b5581e77"
)

COLUMN_DIAGNOSTIC_NAMES = ("tri_aa", "tri_bb", "tri_cc", "tri_dd", "tri_pivot", "tri_residual")


def psolver3_terrain_open_source(*, retain_column_diagnostics: bool = True) -> str:
    """Select diagnostic retention without changing the live solve equations.

    The diagnostic arrays are written after coefficients/pivots are computed
    and read solely for the residual report after back-substitution.  ec/ed
    remain the live elimination storage.  The original variant stays exact.

    The compact variant, which is the one the runtime compiles, also drops
    the stores nothing reads, and only those:

    * ``cm1_tos_prepare_v1`` zeroed ``fpk``, ``divh``, ``hadvp``, ``dum3``,
      ``ec`` and ``ed`` on every stage.  Column pass 1 (``do_initial`` is 1 on
      every launch) assigns ``fpk`` at every level before pass 2 reads or adds
      to it; no kernel reads ``divh``, ``hadvp`` or ``dum3`` at all; and the
      elimination assigns ``ec``/``ed`` at level ``k`` before reading level
      ``k-1`` (``k=2`` reads neither) and back-substitution reads only levels
      that same sweep assigned.
    * The column kernel's ``divh[p]=div`` has no reader.
    * Its update's ``ec[p]=tend`` has none either: the next substep's
      elimination assigns every ``ec`` it reads first, and ``tend`` reaches
      ``ppd`` and ``ppx`` through its own local.
    """
    if type(retain_column_diagnostics) is not bool:
        raise TypeError("retain_column_diagnostics must be bool")
    source = CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE
    if retain_column_diagnostics:
        return source
    replacements = (
        ("\n  fpk[p]=divh[p]=hadvp[p]=dum3[p]=ec[p]=ed[p]=aa[p]=bb[p]=cc[p]=dd[p]=pivot[p]=residual[p]=0.0f;", ""),
        ("aa0[p]=aa;bb0[p]=bb;cc0[p]=cc;dd0[p]=dd;piv[p]=pivot;", ""),
        ("for(unsigned int k=2u;k<=nk;++k){unsigned long long p=pidx(k,pj,pi,ni,nj);float lhs=a32(m32(aa0[p],w[idx3(k-1u,y,x,nyw,nxw)]),m32(bb0[p],w[idx3(k,y,x,nyw,nxw)]));lhs=a32(lhs,m32(cc0[p],w[idx3(k+1u,y,x,nyw,nxw)]));res[p]=s32(lhs,dd0[p]);}", ""),
        ("if(fabsf(div)<smeps)div=0.0f;}divh[p]=div;float w1=", "if(fabsf(div)<smeps)div=0.0f;}float w1="),
        ("float tend=m32(kdiv,s32(up,old));ec[p]=tend;ppd[p]=a32(up,tend);", "float tend=m32(kdiv,s32(up,old));ppd[p]=a32(up,tend);"),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("psolver diagnostic-only source slice changed; review its dataflow before updating")
        source = source.replace(before, after)
    return source


def terrain_open_acoustic_workspace_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    """Explicit radiation packing and terrain-edge snapshot storage."""
    terrain_open_sound_shapes(ni, nj, nk)
    return {
        "pua": (nk, nj + 2, ni + 3), "pu": (nk, nj + 2, ni + 3),
        "pva": (nk, nj + 3, ni + 2), "pv": (nk, nj + 3, ni + 2),
        "pruf": (ni + 3,), "prvf": (nj + 3,),
        "prw": (nk, nj), "pre": (nk, nj), "prs": (nk, ni), "prn": (nk, ni),
        "tous": ((nk + 1) * nj, 4), "tovs": ((nk + 1) * ni, 4),
    }


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledPsolver3TerrainOpenKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]
    retain_column_diagnostics: bool = True
    source_sha256: str = CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256


def compile_cuda_psolver3_terrain_open(*, retain_column_diagnostics: bool = True) -> CompiledPsolver3TerrainOpenKernels:
    digest = hashlib.sha256(CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_BYTES).hexdigest()
    if digest != CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256:
        raise RuntimeError("packaged terrain/open psolver3 CUDA source differs from its pin")
    source = psolver3_terrain_open_source(retain_column_diagnostics=retain_column_diagnostics)
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from the active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        source,
        "cm1_psolver3_terrain_open_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        source,
        "cm1_psolver3_terrain_open_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", text) for name in CUDA_PSOLVER3_TERRAIN_OPEN_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks terrain/open psolver3 entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled terrain/open psolver3 image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_PSOLVER3_TERRAIN_OPEN_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledPsolver3TerrainOpenKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity, retain_column_diagnostics, hashlib.sha256(source.encode()).hexdigest())


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, retain_column_diagnostics: bool = True) -> dict[str, Any]:
    shapes = terrain_open_sound_shapes(ni, nj, nk)
    if not retain_column_diagnostics:
        shapes.update({name: (1,) for name in COLUMN_DIAGNOSTIC_NAMES})
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing CUDA terrain/open sound arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"CUDA terrain/open sound {name} has invalid dtype, shape, or layout")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("CUDA terrain/open sound arrays must not alias")
    return result


def _require_device_metrics(cupy: Any, metrics: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> dict[str, Any]:
    from .cuda_immutable_metrics import allows_compact_coordinate
    shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    missing = [name for name in TERRAIN_STATIC_REQUIRED if name not in metrics]
    if missing:
        raise ValueError(f"missing CUDA terrain/open metrics: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in TERRAIN_STATIC_REQUIRED:
        value = metrics[name]
        if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or (not value.flags.c_contiguous and not allows_compact_coordinate(metrics, name, value)):
            raise ValueError(f"CUDA terrain/open metric {name} has invalid dtype, shape, or layout")
        result[name] = value
    return result


@dataclass(frozen=True, slots=True)
class CudaPsolver3TerrainOpenLaunch:
    ready_event: Any
    compiled: CompiledPsolver3TerrainOpenKernels
    terrain_compiled: Any
    open_compiled: Any
    lateral_compiled: Any
    retained_arrays: tuple[Any, ...]
    cadence: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def launch_cuda_psolver3_terrain_open_three_stage(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    dz: float,
    selection: TerrainOpenSelection = DEFAULT_TERRAIN_OPEN_SELECTION,
    ablations: TerrainOpenSoundAblations | None = None,
    stream: Any | None = None,
    compiled: CompiledPsolver3TerrainOpenKernels | None = None,
    terrain_compiled: Any | None = None,
    open_compiled: Any | None = None,
    lateral_compiled: Any | None = None,
    lateral: Any | None = None,
    workspace: Mapping[str, Any] | None = None,
    alph: float | None = None,
    kdiv: float | None = None,
    _nrk_sequence: tuple[int, ...] = (1, 2, 3),
    nest_substep: Any | None = None,
) -> CudaPsolver3TerrainOpenLaunch:
    """Enqueue the source-ordered terrain/open three-stage acoustic gate.

    ``alph`` is README.namelist:756-759's off-centering weight for the
    vertically implicit acoustic step and ``kdiv`` README.namelist:751-754's
    divergence-damper coefficient; ``None`` binds CM1's own 0.60 and 0.10.
    """

    validate_terrain_open_selection(selection)
    for name, value in (("alph", alph), ("kdiv", kdiv)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric or None")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name}={value} is outside CM1's admitted [0, 1]")
    if _nrk_sequence not in ((1, 2, 3), (1,), (2,), (3,)):
        raise ValueError(
            "internal terrain/open acoustic sequence must be (1,2,3) or one "
            "explicit RK stage"
        )
    dt, dx, dy, dz = (_positive("dt", dt), _positive("dx", dx), _positive("dy", dy), _positive("dz", dz))
    if ablations is None:
        ablations = TerrainOpenSoundAblations()
    cupy = importlib.import_module("cupy")
    from cm1core.cuda_lateral_closure import compile_cuda_lateral_closure
    from cm1core.cuda_sound3_open_boundary import compile_cuda_sound3_open_boundary
    from cm1core.cuda_terrain_open_boundary import compile_cuda_terrain_open_boundary

    retain_diagnostics = getattr(compiled, "retain_column_diagnostics", True)
    a = _require_device_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk, retain_column_diagnostics=retain_diagnostics)
    m = _require_device_metrics(cupy, metrics, ni=ni, nj=nj, nk=nk)
    compiled = compiled or compile_cuda_psolver3_terrain_open()
    terrain_compiled = terrain_compiled or compile_cuda_terrain_open_boundary()
    from . import cuda_boundary_families as _bc

    open_compiled = open_compiled or compile_cuda_sound3_open_boundary()
    lateral_compiled = lateral_compiled or compile_cuda_lateral_closure()
    stream = stream or cupy.cuda.get_current_stream()
    f32, u32 = cupy.float32, cupy.uint32
    alph = f32(CM1_ALPH_F32 if alph is None else alph)
    kdiv_f32 = f32(CM1_KDIV_F32 if kdiv is None else kdiv)
    prepare, pgrad, column, average = compiled.kernels
    tosnap, toss, tobcw = terrain_compiled.kernels
    _, _, _, kpwe, kpsn, kfwe, kfsn, kwrigid, kpack, kunpack = open_compiled.kernels
    threads = (128, 1, 1)
    grid = ((max(max(value.size for value in a.values()), max(value.size for value in m.values())) + 127) // 128, 1, 1)
    edge_grid = ((max(ni * nk, nj * nk) + 127) // 128, 1, 1)
    col_grid = ((ni * nj + 127) // 128, 1, 1)
    face_grid = ((max(nk * nj * max(1, ni - 1), nk * max(1, nj - 1) * ni) + 127) // 128, 1, 1)
    workspace_shapes = terrain_open_acoustic_workspace_shapes(ni, nj, nk)
    if workspace is None:
        scratch = {name: cupy.empty(shape, dtype=cupy.float32) for name, shape in workspace_shapes.items()}
    else:
        if set(workspace) != set(workspace_shapes):
            raise ValueError("supplied acoustic workspace has the wrong named buffer inventory")
        scratch = dict(workspace)
        for name, shape in workspace_shapes.items():
            value = scratch[name]
            if not isinstance(value, cupy.ndarray) or value.shape != shape or value.dtype != cupy.dtype("float32") or not value.flags.c_contiguous:
                raise ValueError(f"supplied acoustic workspace {name} has the wrong dtype, shape or layout")
            if int(value.device.id) != int(cupy.cuda.Device().id):
                raise ValueError("supplied acoustic workspace belongs to another CUDA device")
        pointers = [int(value.data.ptr) for value in (*a.values(), *scratch.values())]
        if len(pointers) != len(set(pointers)):
            raise ValueError("supplied acoustic workspace aliases live state or another workspace")
        if set(pointers).intersection(int(value.data.ptr) for value in m.values()):
            raise ValueError("supplied acoustic workspace or state aliases an immutable terrain metric")
    pua, pu, pva, pv, pruf, prvf, prw, pre, prs, prn, tous, tovs = (scratch[name] for name in workspace_shapes)
    retained = (pua, pu, pva, pv, pruf, prvf, prw, pre, prs, prn, tous, tovs)
    cadence: list[dict[str, Any]] = []
    with stream:
        for nrk in _nrk_sequence:
            nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
            dts = f32(dts_value)
            beta = f32(f32(1.0) - alph)
            prepare(
                grid,
                threads,
                (
                    a["ua"], a["va"], a["wa"], a["ppi"], a["ppx"],
                    m["rr0"], m["rf0"], m["c1"], m["c2"], m["gz"],
                    a["piadv"], a["ppterm"], a["thv"], a["u3d"], a["v3d"], a["w3d"], a["pp3d"], a["ppd"], a["rru"], a["rrv"], a["rrw"],
                    a["qk"], a["pk1"], a["pk2"], a["mm"], a["fpk"], a["divh"], a["hadvp"], a["dum3"], a["elim_c"], a["elim_d"], a["tri_aa"], a["tri_bb"], a["tri_cc"], a["tri_dd"], a["tri_pivot"], a["tri_residual"],
                    u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(nrk), dts, f32(dz), f32(CM1_CP_F32), beta,
                ),
                stream=stream,
            )
            for n in range(1, nloop + 1):
                # A live nest's child re-imposes its specified zone inside every
                # small step, as WRF does (dyn_em/solve_em.F:1262-1604): the
                # words are gathered before the step's own advance.
                if nest_substep is not None:
                    nest_substep.begin(nrk=nrk, n=n, dts=float(dts))
                # src/sound.F:260-268: irbc=2 rebuilds the Klemp-Wilhelmson
                # phase speeds from u3d/v3d at the top of every small step.
                if lateral is not None and lateral.selection.irbc == 2:
                    _bc.enqueue_radbc(a, ni=ni, nj=nj, nk=nk, ng=CM1_OPEN_NGXY, selection=lateral.selection, compiled=lateral.compiled, stream=stream, momentum="3d")
                if ablations.open_consumption:
                    # src/sound.F:274-302 calls ssopenbc* only on a side whose
                    # value is 2; the sealed operator does all four, so a
                    # periodic or rigid face is held across it.
                    if lateral is not None:
                        _bc.enqueue_save_non_open_faces(a, lateral.workspace, ni=ni, nj=nj, nk=nk, ng=CM1_OPEN_NGXY, selection=lateral.selection, compiled=lateral.compiled, stream=stream)
                    tosnap(edge_grid, threads, (a["u3d"], a["v3d"], tous, tovs, u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), f32(dz)), stream=stream)
                    toss(edge_grid, threads, (a["uten"], a["vten"], a["uh"], a["vh"], a["radbcw"], a["radbce"], a["radbcs"], a["radbcn"], m["gz"], m["rgzu"], m["rgzv"], m["gx"], m["gy"], tous, tovs, a["u3d"], a["v3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), dts, f32(1.0 / dx), f32(1.0 / dy), f32(dz)), stream=stream)
                    if lateral is not None:
                        _bc.enqueue_restore_non_open_faces(a, lateral.workspace, ni=ni, nj=nj, nk=nk, ng=CM1_OPEN_NGXY, selection=lateral.selection, compiled=lateral.compiled, stream=stream)
                        # src/sound.F:283-302, in CM1's own position: after the
                        # open normal-velocity step and before the pressure
                        # gradient, so the rescaled face is what the divergence
                        # of this small step sees.
                        if lateral.selection.roflux == 1:
                            _bc.enqueue_restrict_outward_flux(a, m, lateral.workspace, ni=ni, nj=nj, nk=nk, ng=CM1_OPEN_NGXY, selection=lateral.selection, compiled=lateral.compiled, stream=stream)
                if ablations.terrain_pressure_gradient:
                    pgrad(face_grid, threads, (a["ppd"], a["thv"], a["uten"], a["vten"], m["gxu"], m["gyv"], a["u3d"], a["v3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), dts, f32(1.0 / dx), f32(1.0 / dy), f32(dz), f32(CM1_CP_F32)), stream=stream)
                # src/sound.F:386-406 runs i=1+ibw..ni+1-ibe (and j likewise);
                # param.F:6678 makes ibw zero exactly for a periodic side, so
                # those faces are the pressure gradient's and the sealed kernel
                # omits them.  They belong to the same loop: the divergence of
                # sound.F:446-530, formed in the column pass below, reads faces
                # 1 and ni+1 of this small step, not the previous one's.
                if lateral is not None:
                    _bc.enqueue_periodic_face_pressure_gradient(a, m, ni=ni, nj=nj, nk=nk, ng=CM1_OPEN_NGXY, dts=float(dts), dx=dx, dy=dy, dz=dz, cp=CM1_CP_F32, selection=lateral.selection, compiled=lateral.compiled, stream=stream)
                # dyn_em/solve_em.F:1275 advance_uv, then :1341,1351
                # spec_bdyupdate on u_2/v_2 before advance_mu_t reads them.
                if nest_substep is not None:
                    nest_substep.after_momentum(nrk=nrk, n=n, dts=float(dts))
                column(col_grid, threads, (a["piadv"], a["ppterm"], a["ppten"], a["wten"], m["rho0_3d"], m["rr0"], m["rrf0"], m["rf0"], m["c1"], m["c2"], m["gz"], m["rgzu"], m["rgzv"], m["dzdx"], m["dzdy"], a["u3d"], a["v3d"], a["w3d"], a["pp3d"], a["ppd"], a["ppx"], a["qk"], a["pk1"], a["pk2"], a["mm"], a["fpk"], a["divh"], a["hadvp"], a["elim_c"], a["elim_d"], a["tri_aa"], a["tri_bb"], a["tri_cc"], a["tri_dd"], a["tri_pivot"], a["tri_residual"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(nrk), u32(n), u32(nloop), dts, f32(1.0 / dx), f32(1.0 / dy), f32(dz), alph, beta, kdiv_f32, f32(CM1_SMEPS_F32), u32(1), u32(ablations.transformed_divergence), u32(0), u32(0), u32(0)), stream=stream)
                tobcw(grid, threads, (a["u3d"], a["v3d"], m["gz"], m["dzdx"], m["dzdy"], a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)), stream=stream)
                column(col_grid, threads, (a["piadv"], a["ppterm"], a["ppten"], a["wten"], m["rho0_3d"], m["rr0"], m["rrf0"], m["rf0"], m["c1"], m["c2"], m["gz"], m["rgzu"], m["rgzv"], m["dzdx"], m["dzdy"], a["u3d"], a["v3d"], a["w3d"], a["pp3d"], a["ppd"], a["ppx"], a["qk"], a["pk1"], a["pk2"], a["mm"], a["fpk"], a["divh"], a["hadvp"], a["elim_c"], a["elim_d"], a["tri_aa"], a["tri_bb"], a["tri_cc"], a["tri_dd"], a["tri_pivot"], a["tri_residual"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(nrk), u32(n), u32(nloop), dts, f32(1.0 / dx), f32(1.0 / dy), f32(dz), alph, beta, kdiv_f32, f32(CM1_SMEPS_F32), u32(0), u32(0), u32(ablations.pressure_advection), u32(ablations.implicit_vertical_solve), u32(1)), stream=stream)
                # dyn_em/solve_em.F:1457,1467 (t_2, mu_2) and :1582-1604
                # (ph_2, w_2): the thermodynamic and vertical specified update
                # after the implicit column step that advanced them.
                if nest_substep is not None:
                    nest_substep.after_pressure(nrk=nrk, n=n, dts=float(dts))
                if n < nloop and ablations.between_step_pressure_fill:
                    kpwe(grid, threads, (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream)
                    kpsn(grid, threads, (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream)
                average(grid, threads, (a["u3d"], a["v3d"], a["w3d"], a["rru"], a["rrv"], a["rrw"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(n), u32(nloop), alph), stream=stream)
            # src/solve2.F:1258-1271 produces the Durran-Klemp histories after
            # each RK pressure stage.  irbc=1 and irbc=2 build radbc* from the
            # Klemp-Wilhelmson formula instead (src/bc.F:1948-2053), so the
            # Durran-Klemp production must not also run and overwrite them.
            if ablations.radiation_history_production and (
                lateral is None or lateral.selection.irbc == 4
            ):
                kpack(grid, threads, (a["ua"], a["u3d"], a["va"], a["v3d"], a["ruf"], a["rvf"], pua, pu, pva, pv, pruf, prvf, u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)), stream=stream)
                lateral_compiled.kernels[3](edge_grid, (256, 1, 1), (pruf, prvf, pua, pu, pva, pv, prw, pre, prs, prn, u32(ni), u32(nj), u32(nk), f32(dx), f32(dy), f32(rk_stage_dt_f32(dt, nrk)), f32(350.0)), stream=stream)
                kunpack(edge_grid, threads, (prw, pre, prs, prn, a["radbcw"], a["radbce"], a["radbcs"], a["radbcn"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)), stream=stream)
            if ablations.stage_family_fills:
                kfwe(grid, threads, (a["u3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(1)), stream=stream)
                kfsn(grid, threads, (a["u3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(1)), stream=stream)
                kfsn(grid, threads, (a["v3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(2)), stream=stream)
                kfwe(grid, threads, (a["v3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(2)), stream=stream)
                kfwe(grid, threads, (a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(3)), stream=stream)
                kfsn(grid, threads, (a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(3)), stream=stream)
                kwrigid(grid, threads, (a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)), stream=stream)
                tobcw(grid, threads, (a["u3d"], a["v3d"], m["gz"], m["dzdx"], m["dzdy"], a["w3d"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY)), stream=stream)
                if nrk < 3:
                    kpwe(grid, threads, (a["pp3d"], u32(ni), u32(nj), u32(nk)), stream=stream)
                    kpsn(grid, threads, (a["pp3d"], u32(ni), u32(nj), u32(nk)), stream=stream)
                    kfwe(grid, threads, (a["scalar"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(0)), stream=stream)
                    kfsn(grid, threads, (a["scalar"], u32(ni), u32(nj), u32(nk), u32(CM1_OPEN_NGXY), u32(0)), stream=stream)
            cadence.append({"nrk": nrk, "nloop": nloop, "dts_f32": float(dts), "open_consumptions": nloop if ablations.open_consumption else 0, "pressure_fills": nloop - 1 if ablations.between_step_pressure_fill else 0, "radiation_productions": 1 if ablations.radiation_history_production else 0})
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_PSOLVER3_TERRAIN_OPEN_SCHEMA,
        "claim_limit": CUDA_PSOLVER3_TERRAIN_OPEN_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk, "ngxy": CM1_OPEN_NGXY},
        "cadence": cadence,
        # README.namelist:751-759.  Recorded by value so a reader never has to
        # guess which off-centering weight produced these words.
        "acoustic_reals": {"alph": float(alph), "kdiv": float(kdiv_f32)},
        "comparison_contract": {"mode": "raw-bit", "max_ulp": 0, "null_or_missing_is_pass": False},
        "kernel": {"source_sha256": getattr(compiled, "source_sha256", CUDA_PSOLVER3_TERRAIN_OPEN_SOURCE_SHA256), "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(), "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(), "target_arch": compiled.target_arch},
        "runtime": dict(compiled.runtime_identity),
        "lateral_boundaries": (
            lateral.record()
            if lateral is not None
            else {"selection": {"wbc": 2, "ebc": 2, "sbc": 2, "nbc": 2, "irbc": 4, "roflux": 0, "nudgeobc": 0}}
        ),
        "base_policy": "horizontal-broadcast validation only; product admission refused",
        "source_halo_policy": "physical-domain metrics only; no source halo claim",
    }
    if nest_substep is not None:
        receipt["nest_specified_zone"] = nest_substep.record()
    if not retain_diagnostics:
        receipt["column_diagnostics"] = {"retained": False, "omitted_arrays": list(COLUMN_DIAGNOSTIC_NAMES), "live_elimination_arrays": ["elim_c", "elim_d"]}
    return CudaPsolver3TerrainOpenLaunch(event, compiled, terrain_compiled, open_compiled, lateral_compiled, retained, tuple(cadence), receipt)


def launch_cuda_psolver3_terrain_open_stage(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    dz: float,
    selection: TerrainOpenSelection = DEFAULT_TERRAIN_OPEN_SELECTION,
    ablations: TerrainOpenSoundAblations | None = None,
    stream: Any | None = None,
    compiled: Any | None = None,
    terrain_compiled: Any | None = None,
    open_compiled: Any | None = None,
    lateral_compiled: Any | None = None,
    lateral: Any | None = None,
    workspace: Mapping[str, Any] | None = None,
    alph: float | None = None,
    kdiv: float | None = None,
    nest_substep: Any | None = None,
) -> CudaPsolver3TerrainOpenLaunch:
    """Enqueue exactly one acoustic RK stage for source-ordered coupling."""

    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must select explicit CM1 RK stage 1, 2, or 3")
    launch = launch_cuda_psolver3_terrain_open_three_stage(
        arrays,
        metrics,
        ni=ni,
        nj=nj,
        nk=nk,
        dt=dt,
        nsound=nsound,
        dx=dx,
        dy=dy,
        dz=dz,
        selection=selection,
        ablations=ablations,
        stream=stream,
        compiled=compiled,
        terrain_compiled=terrain_compiled,
        open_compiled=open_compiled,
        lateral_compiled=lateral_compiled,
        lateral=lateral,
        workspace=workspace,
        alph=alph,
        kdiv=kdiv,
        _nrk_sequence=(nrk,),
        nest_substep=nest_substep,
    )
    receipt = {
        **launch.receipt,
        "claim_limit": CUDA_PSOLVER3_TERRAIN_OPEN_STAGE_CLAIM_LIMIT,
        "nrk": nrk,
        "schema": CUDA_PSOLVER3_TERRAIN_OPEN_STAGE_SCHEMA,
        "three_stage_composition": False,
    }
    return CudaPsolver3TerrainOpenLaunch(
        launch.ready_event,
        launch.compiled,
        launch.terrain_compiled,
        launch.open_compiled,
        launch.lateral_compiled,
        launch.retained_arrays,
        launch.cadence,
        receipt,
    )
