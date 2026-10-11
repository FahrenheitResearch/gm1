"""CM1 r21.1 serial all-open boundary composition for ``psolver=3``.

The module is deliberately narrow.  It composes the committed
``cuda_lateral_closure`` radiation/mass kernels with the source-ordered
open-normal velocity calls that occur on every acoustic small step, the
between-step pressure fill, and the family-specific RK boundary fills.
Arrays retain the native ``ngxy=3`` family extents and full persistent
``radbc*`` ghost planes.  Terrain and MPI are refused.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_lateral_closure import (
    CUDA_LATERAL_CLOSURE_SOURCE_SHA256,
    DEFAULT_LATERAL_CLOSURE_SELECTION,
    compile_cuda_lateral_closure,
    cpu_mass_ledger_inplace,
    cpu_radiation4_inplace,
    rk_stage_dt_f32,
)
from cm1core.cuda_sound3_column import sound3_stage_schedule_f32
from cm1core.errors import refuse


CUDA_SOUND3_OPEN_SCHEMA = "gpuwm-cm1.cuda-sound3-open-boundary/v1"
CUDA_SOUND3_OPEN_ABI = "gpuwm-cm1.cuda-sound3-open-boundary-kernels/v1"
CUDA_SOUND3_OPEN_THREADS = 128
CM1_OPEN_NGXY = 3
CM1_OPEN_CP_F32 = 1005.7
CM1_OPEN_SPEED_LIMIT_F32 = 350.0
CUDA_SOUND3_OPEN_CLAIM_LIMIT = (
    "CM1 r21.1 serial flat Cartesian all-open boundary side of psolver=3: "
    "irbc=4 histories consumed on every native acoustic small step and "
    "produced after each RK pressure stage, open bcp between small steps, "
    "source-ordered ngxy=3 bcu/bcv/bcw/bcs/bcp fills and corners, rigid w "
    "faces, full radbc restart-plane bytes, and deterministic operator-level "
    "dry/total-water accounting; no terrain, MPI, nonlinear apmasscon, SGS, "
    "complete sound column, complete solve2/solve3, or forecast admission"
)

SOURCE_ANCHORS = {
    "sound_schedule": "src/sound.F:84-103",
    "sound_open_calls": "src/sound.F:253-302",
    "sound_horizontal_uv": "src/sound.F:317-360",
    "sound_pressure_fill": "src/sound.F:739-803",
    "open_normal_velocity": "src/bc.F:2392-2684",
    "radiation4": "src/solve2.F:1258-1271;src/bc.F:2054-2172",
    "stage_boundary_calls": "src/solve2.F:1665-1706",
    "family_boundaries": "src/bc.F:180-1143",
    "open_face_weights": "src/param.F:9180-9220",
    "restart_write": "src/restart_write.F:1839-1970",
    "restart_read": "src/restart_read.F:2329-2474",
}


@dataclass(frozen=True, slots=True)
class OpenBoundarySelection:
    psolver: int = 3
    west_boundary: int = 2
    east_boundary: int = 2
    south_boundary: int = 2
    north_boundary: int = 2
    radiative_order: int = 4
    terrain: bool = False
    mpi: bool = False
    axisymmetric: int = 0
    restricted_outflow: int = 0
    ngxy: int = CM1_OPEN_NGXY


DEFAULT_OPEN_BOUNDARY_SELECTION = OpenBoundarySelection()
_EXPECTED_SELECTION = {
    "dynamics.psolver": ("psolver", 3),
    "boundaries.west": ("west_boundary", 2),
    "boundaries.east": ("east_boundary", 2),
    "boundaries.south": ("south_boundary", 2),
    "boundaries.north": ("north_boundary", 2),
    "boundaries.radiative_order": ("radiative_order", 4),
    "grid.terrain": ("terrain", False),
    "runtime.mpi": ("mpi", False),
    "grid.axisymmetric": ("axisymmetric", 0),
    "boundaries.restricted_outflow": ("restricted_outflow", 0),
    "grid.ngxy": ("ngxy", CM1_OPEN_NGXY),
}


def _validate_selection(selection: OpenBoundarySelection) -> None:
    if not isinstance(selection, OpenBoundarySelection):
        raise TypeError("selection must be an OpenBoundarySelection")
    for option, (attribute, expected) in _EXPECTED_SELECTION.items():
        value = getattr(selection, attribute)
        if type(value) is not type(expected) or value != expected:
            refuse(
                code="CM1-OPEN-BOUNDARY-UNSUPPORTED",
                gate="boundaries-closures",
                option=option,
                value=value,
                reason=(
                    f"the composed operator is measured only for {option}={expected!r}; "
                    "another value changes a family extent, corner winner, acoustic "
                    "boundary equation, or communication path and would produce the "
                    "wrong live/restart state"
                ),
                remedy="use the sealed serial flat all-open profile or add a separate source oracle",
            )


def _validate_dimensions(ni: int, nj: int, nk: int) -> None:
    for name, value, lower in (("ni", ni, 4), ("nj", nj, 3), ("nk", nk, 3)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def open_boundary_array_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_dimensions(ni, nj, nk)
    ng = CM1_OPEN_NGXY
    u = (nk + 2, nj + 2 * ng, ni + 1 + 2 * ng)
    v = (nk + 2, nj + 1 + 2 * ng, ni + 2 * ng)
    w = (nk + 3, nj + 2 * ng, ni + 2 * ng)
    scalar = (nk + 2, nj + 2 * ng, ni + 2 * ng)
    pressure = (nk + 2, nj + 2, ni + 2)
    physical = (nk, nj, ni)
    shapes: dict[str, tuple[int, ...]] = {
        "ua": u,
        "u3d": u,
        "uten": u,
        "va": v,
        "v3d": v,
        "vten": v,
        "w3d": w,
        "scalar": scalar,
        "pp3d": pressure,
        "ppd": pressure,
        "ppx": pressure,
        "thv": pressure,
        "dtu0": (nj + 1, ni + 1),
        "dtv0": (nj + 1, ni + 1),
        "uh": (ni + 2 * ng,),
        "vh": (nj + 2 * ng,),
        "ruf": (ni + 1 + 2 * ng,),
        "rvf": (nj + 1 + 2 * ng,),
        "radbcw": (nk + 2, nj + 2 * ng),
        "radbce": (nk + 2, nj + 2 * ng),
        "radbcs": (nk + 2, ni + 2 * ng),
        "radbcn": (nk + 2, ni + 2 * ng),
        "ruh": (ni,),
        "rvh": (nj,),
        "rmh": physical,
        "rho": physical,
        "qv": physical,
        "qc": physical,
        "qr": physical,
        "qi": physical,
        "qs": physical,
        "qg": physical,
        "dry_layer_before": (nk,),
        "water_layer_before": (nk,),
        "mass_totals_before": (2,),
        "dry_layer_after": (nk,),
        "water_layer_after": (nk,),
        "mass_totals_after": (2,),
    }
    return shapes


FLOAT64_NAMES = (
    "dry_layer_before",
    "water_layer_before",
    "mass_totals_before",
    "dry_layer_after",
    "water_layer_after",
    "mass_totals_after",
)
ALL_ARRAY_NAMES = tuple(open_boundary_array_shapes(4, 3, 3))
WRITABLE_NAMES = (
    "u3d",
    "v3d",
    "w3d",
    "scalar",
    "pp3d",
    "ppd",
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
    *FLOAT64_NAMES,
)


def _require_host_arrays(arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = open_boundary_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing open-boundary arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        dtype = np.dtype("float64" if name in FLOAT64_NAMES else "float32")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != dtype or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid dtype, shape, or C layout")
        if name in WRITABLE_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("open-boundary arrays must not alias")
    return result


def _add(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) + np.float32(y))


def _sub(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) - np.float32(y))


def _mul(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) * np.float32(y))


def _x(i: int) -> int:
    return i + CM1_OPEN_NGXY - 1


def _y(j: int) -> int:
    return j + CM1_OPEN_NGXY - 1


def cpu_bcp_open_inplace(value: Any, *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32") or value.shape != (nk + 2, nj + 2, ni + 2):
        raise ValueError("bcp array must be C-contiguous float32 (nk+2,nj+2,ni+2)")
    for k in range(1, nk + 1):
        for j in range(0, nj + 2):
            value[k, j, 0] = value[k, j, 1]
        for j in range(0, nj + 2):
            value[k, j, ni + 1] = value[k, j, ni]
        for i in range(0, ni + 2):
            value[k, 0, i] = value[k, 1, i]
        for i in range(0, ni + 2):
            value[k, nj + 1, i] = value[k, nj, i]


def cpu_bcs_open_inplace(value: Any, *, ni: int, nj: int, nk: int) -> None:
    ng = CM1_OPEN_NGXY
    for k in range(1, nk + 1):
        for j in range(0, nj + 2):
            for i in range(1 - ng, 1):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(1)]
        for j in range(0, nj + 2):
            for i in range(ni + 1, ni + ng + 1):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(ni)]
        for j in range(1 - ng, 1):
            for i in range(0, ni + 2):
                value[k, _y(j), _x(i)] = value[k, _y(1), _x(i)]
        for j in range(nj + 1, nj + ng + 1):
            for i in range(0, ni + 2):
                value[k, _y(j), _x(i)] = value[k, _y(nj), _x(i)]


def cpu_bcu_open_inplace(value: Any, *, ni: int, nj: int, nk: int) -> None:
    ng = CM1_OPEN_NGXY
    for k in range(1, nk + 1):
        for j in range(0, nj + 2):
            for i in range(1 - ng, 1):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(1)]
        for j in range(0, nj + 2):
            for i in range(ni + 2, ni + ng + 2):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(ni + 1)]
        for j in range(1 - ng, 1):
            for i in range(0, ni + 3):
                value[k, _y(j), _x(i)] = value[k, _y(1), _x(i)]
        for j in range(nj + 1, nj + ng + 1):
            for i in range(0, ni + 3):
                value[k, _y(j), _x(i)] = value[k, _y(nj), _x(i)]


def cpu_bcv_open_inplace(value: Any, *, ni: int, nj: int, nk: int) -> None:
    ng = CM1_OPEN_NGXY
    for k in range(1, nk + 1):
        for j in range(1 - ng, 1):
            for i in range(0, ni + 2):
                value[k, _y(j), _x(i)] = value[k, _y(1), _x(i)]
        for j in range(nj + 2, nj + ng + 2):
            for i in range(0, ni + 2):
                value[k, _y(j), _x(i)] = value[k, _y(nj + 1), _x(i)]
        for j in range(0, nj + 3):
            for i in range(1 - ng, 1):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(1)]
        for j in range(0, nj + 3):
            for i in range(ni + 1, ni + ng + 1):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(ni)]


def cpu_bcw_open_inplace(value: Any, *, ni: int, nj: int, nk: int, flag: int = 1) -> None:
    ng = CM1_OPEN_NGXY
    for k in range(2, nk + 1):
        for j in range(0, nj + 2):
            for i in range(1 - ng, 1):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(1)]
        for j in range(0, nj + 2):
            for i in range(ni + 1, ni + ng + 1):
                value[k, _y(j), _x(i)] = value[k, _y(j), _x(ni)]
        for j in range(1 - ng, 1):
            for i in range(0, ni + 2):
                value[k, _y(j), _x(i)] = value[k, _y(1), _x(i)]
        for j in range(nj + 1, nj + ng + 1):
            for i in range(0, ni + 2):
                value[k, _y(j), _x(i)] = value[k, _y(nj), _x(i)]
    if flag == 1:
        zero = importlib.import_module("numpy").float32(0.0)
        for j in range(0, nj + 2):
            for i in range(0, ni + 2):
                value[1, _y(j), _x(i)] = zero
                value[nk + 1, _y(j), _x(i)] = zero


def cpu_ssopen_all_inplace(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dts: float, rdx: float, rdy: float
) -> None:
    np = importlib.import_module("numpy")
    dts32, rdx32, rdy32 = np.float32(dts), np.float32(rdx), np.float32(rdy)
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            yi = _y(j)
            history = arrays["radbcw"][k, yi]
            grad = _sub(np, arrays["u3d"][k, yi, _x(2)], arrays["u3d"][k, yi, _x(1)])
            term = _mul(np, -history, grad)
            term = _mul(np, term, rdx32)
            term = _mul(np, term, arrays["uh"][_x(1)])
            term = _add(np, term, arrays["uten"][k, yi, _x(1)])
            arrays["u3d"][k, yi, _x(1)] = _add(np, arrays["u3d"][k, yi, _x(1)], _mul(np, dts32, term))
            history = arrays["radbce"][k, yi]
            grad = _sub(np, arrays["u3d"][k, yi, _x(ni + 1)], arrays["u3d"][k, yi, _x(ni)])
            term = _mul(np, -history, grad)
            term = _mul(np, term, rdx32)
            term = _mul(np, term, arrays["uh"][_x(ni)])
            term = _add(np, term, arrays["uten"][k, yi, _x(ni + 1)])
            arrays["u3d"][k, yi, _x(ni + 1)] = _add(np, arrays["u3d"][k, yi, _x(ni + 1)], _mul(np, dts32, term))
        for i in range(1, ni + 1):
            xi = _x(i)
            history = arrays["radbcs"][k, xi]
            grad = _sub(np, arrays["v3d"][k, _y(2), xi], arrays["v3d"][k, _y(1), xi])
            term = _mul(np, -history, grad)
            term = _mul(np, term, rdy32)
            term = _mul(np, term, arrays["vh"][_y(1)])
            term = _add(np, term, arrays["vten"][k, _y(1), xi])
            arrays["v3d"][k, _y(1), xi] = _add(np, arrays["v3d"][k, _y(1), xi], _mul(np, dts32, term))
            history = arrays["radbcn"][k, xi]
            grad = _sub(np, arrays["v3d"][k, _y(nj + 1), xi], arrays["v3d"][k, _y(nj), xi])
            term = _mul(np, -history, grad)
            term = _mul(np, term, rdy32)
            term = _mul(np, term, arrays["vh"][_y(nj)])
            term = _add(np, term, arrays["vten"][k, _y(nj + 1), xi])
            arrays["v3d"][k, _y(nj + 1), xi] = _add(np, arrays["v3d"][k, _y(nj + 1), xi], _mul(np, dts32, term))


def cpu_horizontal_uv_inplace(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dts: float, rdx: float, rdy: float, cp: float = CM1_OPEN_CP_F32
) -> None:
    np = importlib.import_module("numpy")
    dts32 = np.float32(dts)
    tem1 = _mul(np, _mul(np, np.float32(rdx), np.float32(cp)), np.float32(0.5))
    tem2 = _mul(np, _mul(np, np.float32(rdy), np.float32(cp)), np.float32(0.5))
    for k in range(1, nk + 1):
        for j in range(1, nj + 2):
            for i in range(1, ni + 2):
                pg = _mul(np, tem1, _sub(np, arrays["ppd"][k, j, i], arrays["ppd"][k, j, i - 1]))
                pg = _mul(np, pg, _add(np, arrays["thv"][k, j, i], arrays["thv"][k, j, i - 1]))
                inc = _mul(np, dts32, arrays["dtu0"][j - 1, i - 1])
                inc = _mul(np, inc, _sub(np, arrays["uten"][k, _y(j), _x(i)], pg))
                arrays["u3d"][k, _y(j), _x(i)] = _add(np, arrays["u3d"][k, _y(j), _x(i)], inc)
                pg = _mul(np, tem2, _sub(np, arrays["ppd"][k, j, i], arrays["ppd"][k, j - 1, i]))
                pg = _mul(np, pg, _add(np, arrays["thv"][k, j, i], arrays["thv"][k, j - 1, i]))
                inc = _mul(np, dts32, arrays["dtv0"][j - 1, i - 1])
                inc = _mul(np, inc, _sub(np, arrays["vten"][k, _y(j), _x(i)], pg))
                arrays["v3d"][k, _y(j), _x(i)] = _add(np, arrays["v3d"][k, _y(j), _x(i)], inc)


def _pack_w5_host(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> tuple[Any, ...]:
    np = importlib.import_module("numpy")
    u_shape = (nk, nj + 2, ni + 3)
    v_shape = (nk, nj + 3, ni + 2)
    packed = [np.zeros(u_shape, dtype=np.float32) for _ in range(2)] + [np.zeros(v_shape, dtype=np.float32) for _ in range(2)]
    pua, pu, pva, pv = packed
    for k in range(1, nk + 1):
        for j in range(0, nj + 2):
            for i in range(0, ni + 3):
                pua[k - 1, j, i] = arrays["ua"][k, _y(j), _x(i)]
                pu[k - 1, j, i] = arrays["u3d"][k, _y(j), _x(i)]
        for j in range(0, nj + 3):
            for i in range(0, ni + 2):
                pva[k - 1, j, i] = arrays["va"][k, _y(j), _x(i)]
                pv[k - 1, j, i] = arrays["v3d"][k, _y(j), _x(i)]
    ruf = np.empty((ni + 3,), dtype=np.float32)
    rvf = np.empty((nj + 3,), dtype=np.float32)
    for i in range(0, ni + 3):
        ruf[i] = arrays["ruf"][_x(i)]
    for j in range(0, nj + 3):
        rvf[j] = arrays["rvf"][_y(j)]
    return ruf, rvf, pua, pu, pva, pv


def cpu_radiation4_composed_inplace(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dx: float, dy: float, dttmp: float
) -> None:
    np = importlib.import_module("numpy")
    ruf, rvf, ua, u3d, va, v3d = _pack_w5_host(arrays, ni=ni, nj=nj, nk=nk)
    rw = np.empty((nk, nj), dtype=np.float32)
    re = np.empty((nk, nj), dtype=np.float32)
    rs = np.empty((nk, ni), dtype=np.float32)
    rn = np.empty((nk, ni), dtype=np.float32)
    cpu_radiation4_inplace(
        ruf, rvf, ua, u3d, va, v3d, rw, re, rs, rn,
        ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dttmp=dttmp,
        selection=DEFAULT_LATERAL_CLOSURE_SELECTION,
    )
    arrays["radbcw"][1 : nk + 1, CM1_OPEN_NGXY : CM1_OPEN_NGXY + nj] = rw
    arrays["radbce"][1 : nk + 1, CM1_OPEN_NGXY : CM1_OPEN_NGXY + nj] = re
    arrays["radbcs"][1 : nk + 1, CM1_OPEN_NGXY : CM1_OPEN_NGXY + ni] = rs
    arrays["radbcn"][1 : nk + 1, CM1_OPEN_NGXY : CM1_OPEN_NGXY + ni] = rn


@dataclass(frozen=True, slots=True)
class CpuOpenBoundaryStage:
    nrk: int
    nloop: int
    dts_f32: float
    dttmp_f32: float
    radiation_consumptions: int
    ppd_boundary_fills: int
    velocity_family_fills: int
    scalar_pressure_family_fills: int


def cpu_open_boundary_stage_inplace(
    arrays: MutableMapping[str, Any], *, ni: int, nj: int, nk: int, dt: float, nsound: int,
    nrk: int, dx: float, dy: float, cp: float = CM1_OPEN_CP_F32,
    selection: OpenBoundarySelection = DEFAULT_OPEN_BOUNDARY_SELECTION,
) -> CpuOpenBoundaryStage:
    _validate_selection(selection)
    _validate_dimensions(ni, nj, nk)
    dt, dx, dy = _positive("dt", dt), _positive("dx", dx), _positive("dy", dy)
    a = _require_host_arrays(arrays, ni, nj, nk)
    np = importlib.import_module("numpy")
    nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
    dts = np.float32(dts_value)
    if nrk != 1:
        for k in range(1, nk + 1):
            for j in range(0, nj + 2):
                for i in range(0, ni + 2):
                    a["u3d"][k, _y(j), _x(i)] = a["ua"][k, _y(j), _x(i)]
                    a["v3d"][k, _y(j), _x(i)] = a["va"][k, _y(j), _x(i)]
    a["ppd"][1 : nk + 1] = a["ppx"][1 : nk + 1]
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for i in range(1, ni + 1):
                a["scalar"][k, _y(j), _x(i)] = _add(np, a["scalar"][k, _y(j), _x(i)], _mul(np, np.float32(0.25), np.float32(nrk)))
                a["pp3d"][k, j, i] = _add(np, a["pp3d"][k, j, i], _mul(np, np.float32(0.0004), np.float32(nrk)))
                a["w3d"][k, _y(j), _x(i)] = _add(np, a["w3d"][k, _y(j), _x(i)], _mul(np, np.float32(0.01), np.float32(nrk)))
    for n in range(1, nloop + 1):
        cpu_ssopen_all_inplace(a, ni=ni, nj=nj, nk=nk, dts=float(dts), rdx=1.0 / dx, rdy=1.0 / dy)
        cpu_horizontal_uv_inplace(a, ni=ni, nj=nj, nk=nk, dts=float(dts), rdx=1.0 / dx, rdy=1.0 / dy, cp=cp)
        if n < nloop:
            cpu_bcp_open_inplace(a["ppd"], ni=ni, nj=nj, nk=nk)
    dttmp = rk_stage_dt_f32(dt, nrk)
    cpu_radiation4_composed_inplace(a, ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dttmp=dttmp)
    cpu_bcu_open_inplace(a["u3d"], ni=ni, nj=nj, nk=nk)
    cpu_bcv_open_inplace(a["v3d"], ni=ni, nj=nj, nk=nk)
    cpu_bcw_open_inplace(a["w3d"], ni=ni, nj=nj, nk=nk, flag=1)
    if nrk < 3:
        cpu_bcp_open_inplace(a["pp3d"], ni=ni, nj=nj, nk=nk)
        cpu_bcs_open_inplace(a["scalar"], ni=ni, nj=nj, nk=nk)
    return CpuOpenBoundaryStage(nrk, nloop, float(dts), dttmp, nloop, nloop - 1, 3, 2 if nrk < 3 else 0)


def radiation_restart_inventory(ni: int, nj: int, nk: int) -> dict[str, Any]:
    _validate_dimensions(ni, nj, nk)
    ng = CM1_OPEN_NGXY
    return {
        "schema": "gpuwm-cm1.radbc-restart-inventory/v1",
        "capture_cadence": "complete large-step boundary state; preserve across checkpoint/resume before the next solve1/solve2",
        "dtype": "float32",
        "arrays": {
            "radbcw": {"shape_zyx": [nk + 2, nj + 2 * ng], "fortran_bounds": ["jb:je", "kb:ke"], "physical_write": ["j=1:nj", "k=1:nk"]},
            "radbce": {"shape_zyx": [nk + 2, nj + 2 * ng], "fortran_bounds": ["jb:je", "kb:ke"], "physical_write": ["j=1:nj", "k=1:nk"]},
            "radbcs": {"shape_zyx": [nk + 2, ni + 2 * ng], "fortran_bounds": ["ib:ie", "kb:ke"], "physical_write": ["i=1:ni", "k=1:nk"]},
            "radbcn": {"shape_zyx": [nk + 2, ni + 2 * ng], "fortran_bounds": ["ib:ie", "kb:ke"], "physical_write": ["i=1:ni", "k=1:nk"]},
        },
        "restore_rule": "restore every stored word, including transverse and vertical ghosts, before any acoustic history consumption",
    }


def capture_radiation_restart_state(arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = open_boundary_array_shapes(ni, nj, nk)
    result: dict[str, Any] = {}
    for name in ("radbcw", "radbce", "radbcs", "radbcn"):
        value = arrays.get(name)
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} cannot be captured: dtype/shape/layout mismatch")
        result[name] = value.copy(order="C")
    return result


def restore_radiation_restart_state(arrays: Mapping[str, Any], state: Mapping[str, Any], *, ni: int, nj: int, nk: int) -> None:
    np = importlib.import_module("numpy")
    shapes = open_boundary_array_shapes(ni, nj, nk)
    for name in ("radbcw", "radbce", "radbcs", "radbcn"):
        target, source = arrays.get(name), state.get(name)
        if not isinstance(target, np.ndarray) or not isinstance(source, np.ndarray) or target.dtype != np.dtype("float32") or source.dtype != np.dtype("float32") or target.shape != shapes[name] or source.shape != shapes[name] or not target.flags.c_contiguous or not source.flags.c_contiguous:
            raise ValueError(f"{name} cannot be restored: dtype/shape/layout mismatch")
        np.copyto(target, source)


CUDA_SOUND3_OPEN_KERNEL_NAMES = (
    "cm1_open_prepare_stage_f32_v1",
    "cm1_open_ssopen_f32_v1",
    "cm1_open_horizontal_uv_f32_v1",
    "cm1_open_bcp_we_f32_v1",
    "cm1_open_bcp_sn_f32_v1",
    "cm1_open_family_we_f32_v1",
    "cm1_open_family_sn_f32_v1",
    "cm1_open_bcw_rigid_f32_v1",
    "cm1_open_pack_w5_f32_v1",
    "cm1_open_unpack_radbc_f32_v1",
)

CUDA_SOUND3_OPEN_SOURCE = r"""// gpuwm-cm1.cuda-sound3-open-boundary-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default REAL requires float32");
static_assert(sizeof(unsigned long long) == 8, "CM1 offsets require uint64");

__device__ __forceinline__ float a32(const float a,const float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(const float a,const float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(const float a,const float b){return __fmul_rn(a,b);}
__device__ __forceinline__ unsigned long long i3(
    const unsigned int k,const unsigned int j,const unsigned int i,
    const unsigned int ny,const unsigned int nx)
{ return ((unsigned long long)k*ny+j)*nx+i; }

extern "C" __global__
void cm1_open_prepare_stage_f32_v1(
    const float* ua,const float* va,const float* ppx,
    float* u3d,float* v3d,float* ppd,float* pp3d,float* scalar,float* w3d,
    unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int nrk)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng, nyu=nj+2u*ng;
    const unsigned int nxv=ni+2u*ng, nyv=nj+1u+2u*ng;
    const unsigned int nxs=ni+2u*ng, nys=nj+2u*ng;
    const unsigned long long live=(unsigned long long)nk*(nj+2u)*(ni+2u);
    for(unsigned long long q=first;q<live;q+=step){
        const unsigned int i=(unsigned int)(q%(ni+2u));
        const unsigned long long t=q/(ni+2u);
        const unsigned int j=(unsigned int)(t%(nj+2u));
        const unsigned int k=(unsigned int)(t/(nj+2u))+1u;
        const unsigned int xf=i+ng-1u, yf=j+ng-1u;
        ppd[i3(k,j,i,nj+2u,ni+2u)]=ppx[i3(k,j,i,nj+2u,ni+2u)];
        if(nrk!=1u){
            u3d[i3(k,yf,xf,nyu,nxu)]=ua[i3(k,yf,xf,nyu,nxu)];
            v3d[i3(k,yf,xf,nyv,nxv)]=va[i3(k,yf,xf,nyv,nxv)];
        }
        if(i>=1u&&i<=ni&&j>=1u&&j<=nj){
            const unsigned long long si=i3(k,yf,xf,nys,nxs);
            scalar[si]=a32(scalar[si],m32(0.25f,(float)nrk));
            const unsigned long long pi=i3(k,j,i,nj+2u,ni+2u);
            pp3d[pi]=a32(pp3d[pi],m32(0.0004f,(float)nrk));
            w3d[si]=a32(w3d[si],m32(0.01f,(float)nrk));
        }
    }
}

extern "C" __global__
void cm1_open_ssopen_f32_v1(
    const float* uten,const float* vten,const float* uh,const float* vh,
    const float* radbcw,const float* radbce,const float* radbcs,const float* radbcn,
    float* u3d,float* v3d,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
    float dts,float rdx,float rdy)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng;
    const unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const unsigned long long we=(unsigned long long)nk*nj;
    for(unsigned long long q=first;q<we;q+=step){
        const unsigned int j=(unsigned int)(q%nj)+1u;
        const unsigned int k=(unsigned int)(q/nj)+1u;
        const unsigned int y=j+ng-1u;
        const unsigned int xw=ng,xi=ng+1u,xe=ni+ng,xin=ni+ng-1u;
        unsigned long long b=i3(k,y,xw,nyu,nxu);
        float term=m32(-radbcw[(unsigned long long)k*(nj+2u*ng)+y],s32(u3d[i3(k,y,xi,nyu,nxu)],u3d[b]));
        term=m32(term,rdx); term=m32(term,uh[ng]); term=a32(term,uten[b]);
        u3d[b]=a32(u3d[b],m32(dts,term));
        b=i3(k,y,xe,nyu,nxu);
        term=m32(-radbce[(unsigned long long)k*(nj+2u*ng)+y],s32(u3d[b],u3d[i3(k,y,xin,nyu,nxu)]));
        term=m32(term,rdx); term=m32(term,uh[ni+ng-1u]); term=a32(term,uten[b]);
        u3d[b]=a32(u3d[b],m32(dts,term));
    }
    const unsigned long long sn=(unsigned long long)nk*ni;
    for(unsigned long long q=first;q<sn;q+=step){
        const unsigned int i=(unsigned int)(q%ni)+1u;
        const unsigned int k=(unsigned int)(q/ni)+1u;
        const unsigned int x=i+ng-1u;
        const unsigned int ys=ng,yi=ng+1u,yn=nj+ng,yin=nj+ng-1u;
        unsigned long long b=i3(k,ys,x,nyv,nxv);
        float term=m32(-radbcs[(unsigned long long)k*(ni+2u*ng)+x],s32(v3d[i3(k,yi,x,nyv,nxv)],v3d[b]));
        term=m32(term,rdy); term=m32(term,vh[ng]); term=a32(term,vten[b]);
        v3d[b]=a32(v3d[b],m32(dts,term));
        b=i3(k,yn,x,nyv,nxv);
        term=m32(-radbcn[(unsigned long long)k*(ni+2u*ng)+x],s32(v3d[b],v3d[i3(k,yin,x,nyv,nxv)]));
        term=m32(term,rdy); term=m32(term,vh[nj+ng-1u]); term=a32(term,vten[b]);
        v3d[b]=a32(v3d[b],m32(dts,term));
    }
}

extern "C" __global__
void cm1_open_horizontal_uv_f32_v1(
    const float* ppd,const float* thv,const float* uten,const float* vten,
    const float* dtu0,const float* dtv0,float* u3d,float* v3d,
    unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,
    float dts,float rdx,float rdy,float cp)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng;
    const unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const float tem1=m32(m32(rdx,cp),0.5f),tem2=m32(m32(rdy,cp),0.5f);
    const unsigned long long count=(unsigned long long)nk*(nj+1u)*(ni+1u);
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i=(unsigned int)(q%(ni+1u))+1u;
        const unsigned long long t=q/(ni+1u);
        const unsigned int j=(unsigned int)(t%(nj+1u))+1u;
        const unsigned int k=(unsigned int)(t/(nj+1u))+1u;
        const unsigned int xf=i+ng-1u,yf=j+ng-1u;
        float pg=m32(tem1,s32(ppd[i3(k,j,i,nj+2u,ni+2u)],ppd[i3(k,j,i-1u,nj+2u,ni+2u)]));
        pg=m32(pg,a32(thv[i3(k,j,i,nj+2u,ni+2u)],thv[i3(k,j,i-1u,nj+2u,ni+2u)]));
        float inc=m32(dts,dtu0[(unsigned long long)(j-1u)*(ni+1u)+(i-1u)]);
        inc=m32(inc,s32(uten[i3(k,yf,xf,nyu,nxu)],pg));
        u3d[i3(k,yf,xf,nyu,nxu)]=a32(u3d[i3(k,yf,xf,nyu,nxu)],inc);
        pg=m32(tem2,s32(ppd[i3(k,j,i,nj+2u,ni+2u)],ppd[i3(k,j-1u,i,nj+2u,ni+2u)]));
        pg=m32(pg,a32(thv[i3(k,j,i,nj+2u,ni+2u)],thv[i3(k,j-1u,i,nj+2u,ni+2u)]));
        inc=m32(dts,dtv0[(unsigned long long)(j-1u)*(ni+1u)+(i-1u)]);
        inc=m32(inc,s32(vten[i3(k,yf,xf,nyv,nxv)],pg));
        v3d[i3(k,yf,xf,nyv,nxv)]=a32(v3d[i3(k,yf,xf,nyv,nxv)],inc);
    }
}

extern "C" __global__
void cm1_open_bcp_we_f32_v1(float* p,unsigned int ni,unsigned int nj,unsigned int nk)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long count=(unsigned long long)nk*(nj+2u);
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int j=(unsigned int)(q%(nj+2u)),k=(unsigned int)(q/(nj+2u))+1u;
        p[i3(k,j,0u,nj+2u,ni+2u)]=p[i3(k,j,1u,nj+2u,ni+2u)];
        p[i3(k,j,ni+1u,nj+2u,ni+2u)]=p[i3(k,j,ni,nj+2u,ni+2u)];
    }
}

extern "C" __global__
void cm1_open_bcp_sn_f32_v1(float* p,unsigned int ni,unsigned int nj,unsigned int nk)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long count=(unsigned long long)nk*(ni+2u);
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i=(unsigned int)(q%(ni+2u)),k=(unsigned int)(q/(ni+2u))+1u;
        p[i3(k,0u,i,nj+2u,ni+2u)]=p[i3(k,1u,i,nj+2u,ni+2u)];
        p[i3(k,nj+1u,i,nj+2u,ni+2u)]=p[i3(k,nj,i,nj+2u,ni+2u)];
    }
}

extern "C" __global__
void cm1_open_family_we_f32_v1(float* a,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int kind)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nx=(kind==1u)?ni+1u+2u*ng:ni+2u*ng;
    const unsigned int ny=(kind==2u)?nj+1u+2u*ng:nj+2u*ng;
    const unsigned int kfirst=(kind==3u)?2u:1u,kcount=(kind==3u)?nk-1u:nk;
    const unsigned int jcount=(kind==2u)?nj+3u:nj+2u;
    const unsigned long long count=(unsigned long long)kcount*jcount*ng*2u;
    for(unsigned long long q=first;q<count;q+=step){
        unsigned long long r=q;
        const unsigned int layer=(unsigned int)(r%ng); r/=ng;
        const unsigned int side=(unsigned int)(r%2u); r/=2u;
        const unsigned int j0=(unsigned int)(r%jcount),k=(unsigned int)(r/jcount)+kfirst;
        const unsigned int j=j0+ng-1u;
        unsigned int dst,src;
        if(side==0u){dst=layer;src=ng;}
        else if(kind==1u){dst=ni+ng+1u+layer;src=ni+ng;}
        else {dst=ni+ng+layer;src=ni+ng-1u;}
        a[i3(k,j,dst,ny,nx)]=a[i3(k,j,src,ny,nx)];
    }
}

extern "C" __global__
void cm1_open_family_sn_f32_v1(float* a,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng,unsigned int kind)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nx=(kind==1u)?ni+1u+2u*ng:ni+2u*ng;
    const unsigned int ny=(kind==2u)?nj+1u+2u*ng:nj+2u*ng;
    const unsigned int kfirst=(kind==3u)?2u:1u,kcount=(kind==3u)?nk-1u:nk;
    const unsigned int icount=(kind==1u)?ni+3u:ni+2u;
    const unsigned long long count=(unsigned long long)kcount*icount*ng*2u;
    for(unsigned long long q=first;q<count;q+=step){
        unsigned long long r=q;
        const unsigned int layer=(unsigned int)(r%ng); r/=ng;
        const unsigned int side=(unsigned int)(r%2u); r/=2u;
        const unsigned int i0=(unsigned int)(r%icount),k=(unsigned int)(r/icount)+kfirst;
        const unsigned int i=i0+ng-1u;
        unsigned int dst,src;
        if(side==0u){dst=layer;src=ng;}
        else if(kind==2u){dst=nj+ng+1u+layer;src=nj+ng;}
        else {dst=nj+ng+layer;src=nj+ng-1u;}
        a[i3(k,dst,i,ny,nx)]=a[i3(k,src,i,ny,nx)];
    }
}

extern "C" __global__
void cm1_open_bcw_rigid_f32_v1(float* w,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nx=ni+2u*ng,ny=nj+2u*ng;
    const unsigned long long count=(unsigned long long)(nj+2u)*(ni+2u);
    for(unsigned long long q=first;q<count;q+=step){
        const unsigned int i0=(unsigned int)(q%(ni+2u)),j0=(unsigned int)(q/(ni+2u));
        const unsigned int i=i0+ng-1u,j=j0+ng-1u;
        w[i3(1u,j,i,ny,nx)]=0.0f; w[i3(nk+1u,j,i,ny,nx)]=0.0f;
    }
}

extern "C" __global__
void cm1_open_pack_w5_f32_v1(
    const float* ua,const float* u3d,const float* va,const float* v3d,
    const float* ruf,const float* rvf,float* pua,float* pu,float* pva,float* pv,
    float* pruf,float* prvf,unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
    const unsigned long long nu=(unsigned long long)nk*(nj+2u)*(ni+3u);
    for(unsigned long long q=first;q<nu;q+=step){
        const unsigned int i=(unsigned int)(q%(ni+3u)); const unsigned long long t=q/(ni+3u);
        const unsigned int j=(unsigned int)(t%(nj+2u)),k0=(unsigned int)(t/(nj+2u));
        const unsigned long long src=i3(k0+1u,j+ng-1u,i+ng-1u,nyu,nxu);
        pua[q]=ua[src]; pu[q]=u3d[src];
    }
    const unsigned long long nv=(unsigned long long)nk*(nj+3u)*(ni+2u);
    for(unsigned long long q=first;q<nv;q+=step){
        const unsigned int i=(unsigned int)(q%(ni+2u)); const unsigned long long t=q/(ni+2u);
        const unsigned int j=(unsigned int)(t%(nj+3u)),k0=(unsigned int)(t/(nj+3u));
        const unsigned long long src=i3(k0+1u,j+ng-1u,i+ng-1u,nyv,nxv);
        pva[q]=va[src]; pv[q]=v3d[src];
    }
    for(unsigned int i=(unsigned int)first;i<ni+3u;i+=(unsigned int)step)pruf[i]=ruf[i+ng-1u];
    for(unsigned int j=(unsigned int)first;j<nj+3u;j+=(unsigned int)step)prvf[j]=rvf[j+ng-1u];
}

extern "C" __global__
void cm1_open_unpack_radbc_f32_v1(
    const float* prw,const float* pre,const float* prs,const float* prn,
    float* rw,float* re,float* rs,float* rn,
    unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng)
{
    const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
    const unsigned long long we=(unsigned long long)nk*nj;
    for(unsigned long long q=first;q<we;q+=step){
        const unsigned int j=(unsigned int)(q%nj),k0=(unsigned int)(q/nj);
        const unsigned long long dst=(unsigned long long)(k0+1u)*(nj+2u*ng)+(j+ng);
        rw[dst]=prw[q]; re[dst]=pre[q];
    }
    const unsigned long long sn=(unsigned long long)nk*ni;
    for(unsigned long long q=first;q<sn;q+=step){
        const unsigned int i=(unsigned int)(q%ni),k0=(unsigned int)(q/ni);
        const unsigned long long dst=(unsigned long long)(k0+1u)*(ni+2u*ng)+(i+ng);
        rs[dst]=prs[q]; rn[dst]=prn[q];
    }
}
"""

CUDA_SOUND3_OPEN_SOURCE_BYTES = CUDA_SOUND3_OPEN_SOURCE.encode("utf-8")
CUDA_SOUND3_OPEN_SOURCE_SHA256 = "135689853d568f931d7cde0c59da79f37fb267a2ef2d8d124adc9299b8db4837"


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_SOUND3_OPEN_SOURCE_BYTES).hexdigest() == CUDA_SOUND3_OPEN_SOURCE_SHA256


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "", 1)
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


def _image_options(arch: str, nvrtc_major: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return common + (f"--gpu-architecture=compute_{arch}",), common + (f"--gpu-architecture=sm_{arch}",)


@dataclass(frozen=True, slots=True)
class CompiledOpenBoundaryKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_sound3_open_boundary() -> CompiledOpenBoundaryKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged open-boundary CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(CUDA_SOUND3_OPEN_SOURCE, "cm1_open_boundary_v1_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_SOUND3_OPEN_SOURCE, "cm1_open_boundary_v1_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text) for name in CUDA_SOUND3_OPEN_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks an open-boundary entry point")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled open-boundary image is not an ELF64 CUDA cubin")
    module = function.Module(); module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_SOUND3_OPEN_KERNEL_NAMES)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledOpenBoundaryKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = open_boundary_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing open-boundary device arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        dtype = cupy.dtype("float64" if name in FLOAT64_NAMES else "float32")
        if not isinstance(value, cupy.ndarray) or value.dtype != dtype or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid CUDA dtype, shape, or layout")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("open-boundary device arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaOpenBoundaryLaunch:
    ready_event: Any
    compiled: CompiledOpenBoundaryKernels
    lateral_compiled: Any
    retained_arrays: tuple[Any, ...]
    stage_snapshots: Mapping[str, Any]
    cadence: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


def launch_cuda_sound3_open_boundary_three_stage(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dt: float, nsound: int,
    dx: float, dy: float, cp: float = CM1_OPEN_CP_F32,
    selection: OpenBoundarySelection = DEFAULT_OPEN_BOUNDARY_SELECTION,
    stream: Any | None = None, compiled: CompiledOpenBoundaryKernels | None = None,
    lateral_compiled: Any | None = None,
) -> CudaOpenBoundaryLaunch:
    _validate_selection(selection)
    _validate_dimensions(ni, nj, nk)
    dt, dx, dy = _positive("dt", dt), _positive("dx", dx), _positive("dy", dy)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    compiled = compile_cuda_sound3_open_boundary() if compiled is None else compiled
    lateral_compiled = compile_cuda_lateral_closure() if lateral_compiled is None else lateral_compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    f32, u32, u64 = cupy.float32, cupy.uint32, cupy.uint64
    kprep,kss,kuv,kpwe,kpsn,kfwe,kfsn,kwrigid,kpack,kunpack = compiled.kernels
    threads=(CUDA_SOUND3_OPEN_THREADS,1,1)
    max_words=max(value.size for name,value in a.items() if name not in FLOAT64_NAMES)
    grid=((max_words+CUDA_SOUND3_OPEN_THREADS-1)//CUDA_SOUND3_OPEN_THREADS,1,1)
    edge_grid=((max(ni,nj)*nk+CUDA_SOUND3_OPEN_THREADS-1)//CUDA_SOUND3_OPEN_THREADS,1,1)
    u_shape=(nk,nj+2,ni+3); v_shape=(nk,nj+3,ni+2)
    pua=cupy.empty(u_shape,dtype=cupy.float32); pu=cupy.empty_like(pua)
    pva=cupy.empty(v_shape,dtype=cupy.float32); pv=cupy.empty_like(pva)
    pruf=cupy.empty((ni+3,),dtype=cupy.float32); prvf=cupy.empty((nj+3,),dtype=cupy.float32)
    prw=cupy.empty((nk,nj),dtype=cupy.float32); pre=cupy.empty_like(prw)
    prs=cupy.empty((nk,ni),dtype=cupy.float32); prn=cupy.empty_like(prs)
    snapshot_names=("radbcw","radbce","radbcs","radbcn","u3d","v3d","w3d","scalar","pp3d","ppd")
    snapshots={name:cupy.empty((3,*a[name].shape),dtype=cupy.float32) for name in snapshot_names}
    retained=(pua,pu,pva,pv,pruf,prvf,prw,pre,prs,prn,*snapshots.values())

    def launch_mass(suffix: str) -> None:
        layer_grid=((nk+255)//256,1,1); mass_threads=(256,1,1)
        lateral_compiled.kernels[4](layer_grid,mass_threads,(a["rho"],a["qv"],a["qc"],a["qr"],a["qi"],a["qs"],a["qg"],a["ruh"],a["rvh"],a["rmh"],a[f"dry_layer_{suffix}"],a[f"water_layer_{suffix}"],u32(ni),u32(nj),u32(nk)),stream=stream)
        lateral_compiled.kernels[5]((1,1,1),mass_threads,(a[f"dry_layer_{suffix}"],a[f"water_layer_{suffix}"],a[f"mass_totals_{suffix}"],u32(nk)),stream=stream)

    launch_mass("before")
    cadence=[]
    for nrk in (1,2,3):
        nloop,dts=sound3_stage_schedule_f32(dt,nsound,nrk)
        kprep(grid,threads,(a["ua"],a["va"],a["ppx"],a["u3d"],a["v3d"],a["ppd"],a["pp3d"],a["scalar"],a["w3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(nrk)),stream=stream)
        for n in range(1,nloop+1):
            kss(edge_grid,threads,(a["uten"],a["vten"],a["uh"],a["vh"],a["radbcw"],a["radbce"],a["radbcs"],a["radbcn"],a["u3d"],a["v3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),f32(dts),f32(1.0/dx),f32(1.0/dy)),stream=stream)
            kuv(grid,threads,(a["ppd"],a["thv"],a["uten"],a["vten"],a["dtu0"],a["dtv0"],a["u3d"],a["v3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),f32(dts),f32(1.0/dx),f32(1.0/dy),f32(cp)),stream=stream)
            if n<nloop:
                kpwe(grid,threads,(a["ppd"],u32(ni),u32(nj),u32(nk)),stream=stream)
                kpsn(grid,threads,(a["ppd"],u32(ni),u32(nj),u32(nk)),stream=stream)
        kpack(grid,threads,(a["ua"],a["u3d"],a["va"],a["v3d"],a["ruf"],a["rvf"],pua,pu,pva,pv,pruf,prvf,u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY)),stream=stream)
        dttmp=rk_stage_dt_f32(dt,nrk)
        lateral_compiled.kernels[3](edge_grid,(256,1,1),(pruf,prvf,pua,pu,pva,pv,prw,pre,prs,prn,u32(ni),u32(nj),u32(nk),f32(dx),f32(dy),f32(dttmp),f32(CM1_OPEN_SPEED_LIMIT_F32)),stream=stream)
        kunpack(edge_grid,threads,(prw,pre,prs,prn,a["radbcw"],a["radbce"],a["radbcs"],a["radbcn"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY)),stream=stream)
        # Source order: bcu WE->SN, bcv SN->WE, bcw WE->SN->rigid.
        kfwe(grid,threads,(a["u3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(1)),stream=stream)
        kfsn(grid,threads,(a["u3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(1)),stream=stream)
        kfsn(grid,threads,(a["v3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(2)),stream=stream)
        kfwe(grid,threads,(a["v3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(2)),stream=stream)
        kfwe(grid,threads,(a["w3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(3)),stream=stream)
        kfsn(grid,threads,(a["w3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(3)),stream=stream)
        kwrigid(grid,threads,(a["w3d"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY)),stream=stream)
        if nrk<3:
            kpwe(grid,threads,(a["pp3d"],u32(ni),u32(nj),u32(nk)),stream=stream)
            kpsn(grid,threads,(a["pp3d"],u32(ni),u32(nj),u32(nk)),stream=stream)
            kfwe(grid,threads,(a["scalar"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(0)),stream=stream)
            kfsn(grid,threads,(a["scalar"],u32(ni),u32(nj),u32(nk),u32(CM1_OPEN_NGXY),u32(0)),stream=stream)
        for name in snapshot_names:
            count=a[name].size
            copy_grid=((count+255)//256,1,1)
            lateral_compiled.kernels[0](copy_grid,(256,1,1),(a[name],snapshots[name][nrk-1],u64(count)),stream=stream)
        cadence.append({"nrk":nrk,"nloop":nloop,"dts_f32":float(f32(dts)),"dttmp_f32":dttmp,"radiation_consumptions":nloop,"ppd_boundary_fills":nloop-1,"radiation4_productions":1,"velocity_family_fills":3,"scalar_pressure_family_fills":2 if nrk<3 else 0})
    launch_mass("after")
    event=cupy.cuda.Event(disable_timing=True); event.record(stream)
    receipt={
        "schema":CUDA_SOUND3_OPEN_SCHEMA,
        "claim_limit":CUDA_SOUND3_OPEN_CLAIM_LIMIT,
        "source_anchors":dict(SOURCE_ANCHORS),
        "domain":{"ni":ni,"nj":nj,"nk":nk,"ngxy":CM1_OPEN_NGXY,"terrain":False,"mpi":False},
        "cadence":cadence,
        "corner_order":{"bcs_bcp_bcu_bcw":"west,east,south,north","bcv":"south,north,west,east"},
        "restart_inventory":radiation_restart_inventory(ni,nj,nk),
        "comparison_contract":{"mode":"raw-bit","max_ulp":0,"null_or_missing_is_pass":False},
        "kernel":{"abi":CUDA_SOUND3_OPEN_ABI,"names":list(CUDA_SOUND3_OPEN_KERNEL_NAMES),"source_sha256":CUDA_SOUND3_OPEN_SOURCE_SHA256,"ptx_sha256":hashlib.sha256(compiled.ptx).hexdigest(),"cubin_sha256":hashlib.sha256(compiled.cubin).hexdigest(),"target_arch":compiled.target_arch,"lateral_source_sha256":CUDA_LATERAL_CLOSURE_SOURCE_SHA256},
        "composition":{"radiation4":"reused cm1_lateral_closure kernel[3]","mass":"reused cm1_lateral_closure kernels[4:6]","history_lag":"each acoustic stage consumes prior radbc words and only then produces next-stage words"},
        "runtime":dict(compiled.runtime_identity),
    }
    return CudaOpenBoundaryLaunch(event,compiled,lateral_compiled,retained,snapshots,tuple(cadence),receipt)


__all__ = (
    "ALL_ARRAY_NAMES",
    "CM1_OPEN_NGXY",
    "CUDA_SOUND3_OPEN_ABI",
    "CUDA_SOUND3_OPEN_CLAIM_LIMIT",
    "CUDA_SOUND3_OPEN_KERNEL_NAMES",
    "CUDA_SOUND3_OPEN_SCHEMA",
    "CUDA_SOUND3_OPEN_SOURCE_SHA256",
    "DEFAULT_OPEN_BOUNDARY_SELECTION",
    "OpenBoundarySelection",
    "capture_radiation_restart_state",
    "compile_cuda_sound3_open_boundary",
    "cpu_bcp_open_inplace",
    "cpu_bcs_open_inplace",
    "cpu_bcu_open_inplace",
    "cpu_bcv_open_inplace",
    "cpu_bcw_open_inplace",
    "cpu_open_boundary_stage_inplace",
    "cpu_radiation4_composed_inplace",
    "launch_cuda_sound3_open_boundary_three_stage",
    "open_boundary_array_shapes",
    "radiation_restart_inventory",
    "restore_radiation_restart_state",
)
