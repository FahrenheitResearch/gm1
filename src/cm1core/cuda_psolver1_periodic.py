"""CM1 r21.1 flat serial-periodic ``psolver=1`` RK operator.

This is the non-time-split explicit-compressible ``soundns.F`` branch.  One
call advances exactly one CM1 RK stage from supplied live ``sten`` (the
source-ordered pressure-tendency advection result) and held momentum/
thermodynamic inputs.  It is a bounded operator family, not a full solve2 or
forecast driver.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .cuda_sound3_column import rk_stage_dt_f32
from .cuda_sound3_periodic import (
    _canonical_arch,
    _image_options,
    fill_periodic_scalar_inplace,
    fill_periodic_uv_inplace,
    scalar_shape_zyx,
    u_shape_zyx,
    v_shape_zyx,
    w_shape_zyx,
)
from .errors import refuse


CUDA_PSOLVER1_PERIODIC_SCHEMA = "gpuwm-cm1.cuda-psolver1-periodic/v1"
CUDA_PSOLVER1_PERIODIC_ABI = "gpuwm-cm1.cuda-psolver1-periodic-kernels/v1"
CUDA_PSOLVER1_PERIODIC_THREADS = 128
CM1_PSOLVER1_CP_F32 = 1005.7
CM1_PSOLVER1_SMEPS_F32 = 1.0e-30
CM1_PSOLVER1_CLAIM_LIMIT = (
    "CM1 r21.1 soundns.F non-time-split explicit-compressible psolver=1 "
    "RK-stage operator for a flat Cartesian, single-process, unstretched, "
    "four-periodic, eqtset=2, apmasscon=0, fixed-dt profile with uniform "
    "horizontal metrics, horizontally homogeneous mh/mf/c1/c2, rigid "
    "vertical w faces, and supplied live post-advs sten plus held momentum "
    "and thermodynamic inputs; exact velocity boundary closure every stage "
    "and pp3d periodic closure only for nrk<3; no pressure-tendency advs, "
    "adaptive timestep, terrain, MPI, axisymmetry, open/radiative or rigid "
    "lateral boundaries, immersed boundary, forcing, mass correction, "
    "complete solve2/solve3, stock case, coupled trajectory, forecast, "
    "meteorological product, tornado, or performance claim"
)
SOURCE_ANCHORS = {
    "rk_stage_dt": "src/solve2.F:296-300",
    "pressure_advection_seam": "src/solve2.F:1053-1071",
    "dispatch": "src/solve2.F:1085-1100",
    "stage_face_dt": "src/soundns.F:69-74",
    "flat_cartesian_uv": "src/soundns.F:140-164",
    "flat_vertical_w": "src/soundns.F:267-282",
    "flat_divergence": "src/misclibs.F:108-141",
    "pressure_update": "src/soundns.F:324-336",
    "velocity_boundary_cadence": "src/solve2.F:1662-1680",
    "pressure_boundary_cadence": "src/solve2.F:1681-1706",
    "periodic_boundaries": "src/bc.F:372-520,528-1143",
    "selection_constraints": "src/param.F:1277-1307,2052-2070,2787-2789",
}


@dataclass(frozen=True, slots=True)
class Psolver1PeriodicSelection:
    psolver: int = 1
    terrain_flag: bool = False
    axisymm: int = 0
    mpi: bool = False
    stretch_x: int = 0
    stretch_y: int = 0
    wbc: int = 1
    ebc: int = 1
    sbc: int = 1
    nbc: int = 1
    irbc: int = 4
    roflux: int = 0
    nudgeobc: int = 0
    eqtset: int = 2
    apmasscon: int = 0
    adapt_dt: int = 0
    do_ib: bool = False
    convinit: int = 0
    wnudge: int = 0


def require_psolver1_periodic_selection(
    selection: Psolver1PeriodicSelection,
) -> None:
    checks = (
        ("psolver", selection.psolver, 1, "ARWEN-PS1-PSOLVER"),
        ("terrain_flag", selection.terrain_flag, False, "ARWEN-PS1-TERRAIN"),
        ("axisymm", selection.axisymm, 0, "ARWEN-PS1-AXISYMM"),
        ("mpi", selection.mpi, False, "ARWEN-PS1-MPI"),
        ("stretch_x", selection.stretch_x, 0, "ARWEN-PS1-STRETCH-X"),
        ("stretch_y", selection.stretch_y, 0, "ARWEN-PS1-STRETCH-Y"),
        ("wbc", selection.wbc, 1, "ARWEN-PS1-WBC"),
        ("ebc", selection.ebc, 1, "ARWEN-PS1-EBC"),
        ("sbc", selection.sbc, 1, "ARWEN-PS1-SBC"),
        ("nbc", selection.nbc, 1, "ARWEN-PS1-NBC"),
        ("irbc", selection.irbc, 4, "ARWEN-PS1-IRBC"),
        ("roflux", selection.roflux, 0, "ARWEN-PS1-ROFLUX"),
        ("nudgeobc", selection.nudgeobc, 0, "ARWEN-PS1-NUDGE-OBC"),
        ("eqtset", selection.eqtset, 2, "ARWEN-PS1-EQTSET"),
        ("apmasscon", selection.apmasscon, 0, "ARWEN-PS1-APMASSCON"),
        ("adapt_dt", selection.adapt_dt, 0, "ARWEN-PS1-ADAPT-DT"),
        ("do_ib", selection.do_ib, False, "ARWEN-PS1-IMMERSED"),
        ("convinit", selection.convinit, 0, "ARWEN-PS1-CONVINIT"),
        ("wnudge", selection.wnudge, 0, "ARWEN-PS1-WNUDGE"),
    )
    for option, value, expected, code in checks:
        if value != expected:
            refuse(
                code=code,
                gate="psolver1-flat-periodic-selection",
                reason=(
                    f"the bounded psolver=1 family requires {option}={expected!r}; "
                    "the official alternate branch is not implemented here"
                ),
                remedy="select the exact flat serial-four-periodic fixed-dt family",
                option=option,
                value=value,
            )


SCALAR_NAMES = (
    "mh",
    "mf",
    "c1",
    "c2",
    "ppi",
    "pp3d",
    "sten",
    "thv",
    "ppterm",
    "div",
)
U_NAMES = ("ua", "u3d", "uten")
V_NAMES = ("va", "v3d", "vten")
W_NAMES = ("wa", "w3d", "wten")
METRIC_NAMES = ("uf", "vf", "uh", "vh")
FACE_DT_NAMES = ("dtu0", "dtv0", "dtu", "dtv")
ALL_ARRAY_NAMES = SCALAR_NAMES + U_NAMES + V_NAMES + W_NAMES + METRIC_NAMES + FACE_DT_NAMES
WRITABLE_NAMES = ("dtu", "dtv", "u3d", "v3d", "w3d", "div", "pp3d")
COMPARE_OUTPUT_NAMES = WRITABLE_NAMES


def _validate_dimensions(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")


def psolver1_periodic_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    _validate_dimensions(ni, nj, nk)
    scalar = scalar_shape_zyx(ni, nj, nk)
    result = {name: scalar for name in SCALAR_NAMES}
    result.update({name: u_shape_zyx(ni, nj, nk) for name in U_NAMES})
    result.update({name: v_shape_zyx(ni, nj, nk) for name in V_NAMES})
    result.update({name: w_shape_zyx(ni, nj, nk) for name in W_NAMES})
    result.update(
        {"uf": (ni + 2,), "uh": (ni + 2,), "vf": (nj + 2,), "vh": (nj + 2,)}
    )
    result.update({name: (nj + 2, ni + 2) for name in FACE_DT_NAMES})
    return result


def default_psolver1_periodic_face_weights(ni: int, nj: int) -> tuple[Any, Any]:
    _validate_dimensions(ni, nj, 3)
    np = importlib.import_module("numpy")
    dtu0 = np.zeros((nj + 2, ni + 2), dtype=np.float32)
    dtv0 = np.zeros((nj + 2, ni + 2), dtype=np.float32)
    # Exact serial-periodic param.F:9182-9219 initialization in scalar indexing.
    dtu0[1 : nj + 1, 1 : ni + 2] = np.float32(1.0)
    dtv0[1 : nj + 2, 1 : ni + 1] = np.float32(1.0)
    return dtu0, dtv0


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = psolver1_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver1 arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid shape or layout")
        if name in WRITABLE_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} contains a non-finite value")
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("psolver1 arrays must not alias")
    return result


def _bits_equal(np: Any, left: Any, right: Any) -> bool:
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def validate_psolver1_periodic_contract(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: Psolver1PeriodicSelection = Psolver1PeriodicSelection(),
) -> None:
    require_psolver1_periodic_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    for name in METRIC_NAMES:
        if not _bits_equal(np, a[name], np.ones_like(a[name])):
            raise ValueError(f"{name} must be exactly one for this uniform flat-grid slice")
    expected_dtu0, expected_dtv0 = default_psolver1_periodic_face_weights(ni, nj)
    if not _bits_equal(np, a["dtu0"], expected_dtu0):
        raise ValueError("dtu0 differs from CM1 serial-periodic param.F weights")
    if not _bits_equal(np, a["dtv0"], expected_dtv0):
        raise ValueError("dtv0 differs from CM1 serial-periodic param.F weights")
    for name in ("mh", "mf", "c1", "c2"):
        live = a[name][:, 1 : nj + 1, 1 : ni + 1]
        profile = a[name][:, 1:2, 1:2]
        if not _bits_equal(np, live, np.broadcast_to(profile, live.shape)):
            raise ValueError(f"{name} must be horizontally homogeneous")
    for name in ("pp3d", "thv"):
        value = a[name]
        for left, right, side in (
            (value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni], "west"),
            (value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1], "east"),
            (value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :], "south"),
            (value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :], "north"),
        ):
            if not _bits_equal(np, left, right):
                raise ValueError(f"{name} {side} periodic halo is invalid")
    for name in ("ua", "uten"):
        if not _bits_equal(
            np,
            a[name][1 : nk + 1, 1 : nj + 1, 0],
            a[name][1 : nk + 1, 1 : nj + 1, ni],
        ):
            raise ValueError(f"{name} duplicate periodic x faces differ")
    for name in ("va", "vten"):
        if not _bits_equal(
            np,
            a[name][1 : nk + 1, 0, 1 : ni + 1],
            a[name][1 : nk + 1, nj, 1 : ni + 1],
        ):
            raise ValueError(f"{name} duplicate periodic y faces differ")
    if np.any(a["wa"][[1, nk + 1]].view(np.uint32) != np.uint32(0)):
        raise ValueError("wa rigid bottom/top faces must be positive zero")


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


@dataclass(frozen=True, slots=True)
class CpuPsolver1StageDiagnostics:
    nrk: int
    dttmp_f32: float
    velocity_boundary_fills: int
    pressure_boundary_fills: int
    live_sten_consumed: bool
    boundary_trace_sha256: str


@dataclass(frozen=True, slots=True)
class CpuPsolver1ThreeStageDiagnostics:
    cadence: tuple[CpuPsolver1StageDiagnostics, ...]


def cpu_psolver1_periodic_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    cp: float = CM1_PSOLVER1_CP_F32,
    smeps: float = CM1_PSOLVER1_SMEPS_F32,
    selection: Psolver1PeriodicSelection = Psolver1PeriodicSelection(),
) -> CpuPsolver1StageDiagnostics:
    validate_psolver1_periodic_contract(
        arrays, ni=ni, nj=nj, nk=nk, selection=selection
    )
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    for name, value in (("dt", dt), ("rdx", rdx), ("rdy", rdy), ("rdz", rdz), ("cp", cp), ("smeps", smeps)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    dttmp = np.float32(rk_stage_dt_f32(dt, nrk))
    rdx, rdy, rdz, cp, smeps = map(np.float32, (rdx, rdy, rdz, cp, smeps))

    for j in range(1, nj + 2):
        for i in range(1, ni + 2):
            a["dtu"][j, i] = _mul(np, dttmp, a["dtu0"][j, i])
            a["dtv"][j, i] = _mul(np, dttmp, a["dtv0"][j, i])

    tem1 = _mul(np, _mul(np, _mul(np, dttmp, rdx), cp), np.float32(0.5))
    tem2 = _mul(np, _mul(np, _mul(np, dttmp, rdy), cp), np.float32(0.5))
    for k in range(1, nk + 1):
        for j in range(1, nj + 2):
            for i in range(1, ni + 2):
                f = i - 1
                pressure = _mul(np, tem1, _sub(np, a["pp3d"][k, j, i], a["pp3d"][k, j, i - 1]))
                pressure = _mul(np, pressure, a["uf"][i])
                pressure = _mul(np, pressure, _add(np, a["thv"][k, j, i], a["thv"][k, j, i - 1]))
                a["u3d"][k, j, f] = _sub(
                    np,
                    _add(np, a["ua"][k, j, f], _mul(np, a["dtu"][j, i], a["uten"][k, j, f])),
                    pressure,
                )
                vf = j - 1
                pressure = _mul(np, tem2, _sub(np, a["pp3d"][k, j, i], a["pp3d"][k, j - 1, i]))
                pressure = _mul(np, pressure, a["vf"][j])
                pressure = _mul(np, pressure, _add(np, a["thv"][k, j, i], a["thv"][k, j - 1, i]))
                a["v3d"][k, vf, i] = _sub(
                    np,
                    _add(np, a["va"][k, vf, i], _mul(np, a["dtv"][j, i], a["vten"][k, vf, i])),
                    pressure,
                )

    for k in range(2, nk + 1):
        temw = _mul(np, _mul(np, rdz, cp), a["mf"][k, 1, 1])
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                theta = _add(
                    np,
                    _mul(np, a["c2"][k, 1, 1], a["thv"][k, j, i]),
                    _mul(np, a["c1"][k, 1, 1], a["thv"][k - 1, j, i]),
                )
                pressure = _mul(np, temw, _sub(np, a["pp3d"][k, j, i], a["pp3d"][k - 1, j, i]))
                pressure = _mul(np, pressure, theta)
                a["w3d"][k, j0, i0] = _add(
                    np,
                    a["wa"][k, j0, i0],
                    _mul(np, dttmp, _sub(np, a["wten"][k, j0, i0], pressure)),
                )

    for k in range(1, nk + 1):
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                du = _mul(np, _mul(np, _sub(np, a["u3d"][k, j, i], a["u3d"][k, j, i - 1]), rdx), a["uh"][i])
                dv = _mul(np, _mul(np, _sub(np, a["v3d"][k, j, i], a["v3d"][k, j - 1, i]), rdy), a["vh"][j])
                horizontal = _add(np, du, dv)
                vertical = _mul(np, _mul(np, _sub(np, a["w3d"][k + 1, j0, i0], a["w3d"][k, j0, i0]), rdz), a["mh"][k, 1, 1])
                divergence = _add(np, horizontal, vertical)
                if bool(np.abs(divergence) < smeps):
                    divergence = np.float32(0.0)
                a["div"][k, j, i] = divergence
                tendency = _sub(np, a["sten"][k, j, i], _mul(np, a["ppterm"][k, j, i], divergence))
                pressure = _add(np, a["ppi"][k, j, i], _mul(np, dttmp, tendency))
                if bool(np.abs(pressure) < smeps):
                    pressure = np.float32(0.0)
                a["pp3d"][k, j, i] = pressure

    fill_periodic_uv_inplace(a["u3d"], a["v3d"], ni=ni, nj=nj, nk=nk)
    a["w3d"][[1, nk + 1], :, :] = np.float32(0.0)
    pressure_fills = 0
    if nrk < 3:
        fill_periodic_scalar_inplace(a["pp3d"], ni=ni, nj=nj, nk=nk)
        pressure_fills = 1
    trace = hashlib.sha256()
    trace.update(a["u3d"].tobytes(order="C"))
    trace.update(a["v3d"].tobytes(order="C"))
    trace.update(a["w3d"].tobytes(order="C"))
    if nrk < 3:
        trace.update(a["pp3d"].tobytes(order="C"))
    return CpuPsolver1StageDiagnostics(
        nrk=nrk,
        dttmp_f32=float(dttmp),
        velocity_boundary_fills=3,
        pressure_boundary_fills=pressure_fills,
        live_sten_consumed=True,
        boundary_trace_sha256=trace.hexdigest(),
    )


def cpu_psolver1_periodic_three_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    rdx: float,
    rdy: float,
    rdz: float,
    cp: float = CM1_PSOLVER1_CP_F32,
    smeps: float = CM1_PSOLVER1_SMEPS_F32,
    selection: Psolver1PeriodicSelection = Psolver1PeriodicSelection(),
) -> CpuPsolver1ThreeStageDiagnostics:
    cadence = tuple(
        cpu_psolver1_periodic_stage_inplace(
            arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            dt=dt,
            nrk=nrk,
            rdx=rdx,
            rdy=rdy,
            rdz=rdz,
            cp=cp,
            smeps=smeps,
            selection=selection,
        )
        for nrk in (1, 2, 3)
    )
    return CpuPsolver1ThreeStageDiagnostics(cadence)


def compare_psolver1_periodic_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    fields = {}
    for name in COMPARE_OUTPUT_NAMES:
        if name not in expected or name not in candidate:
            raise ValueError(f"missing comparison output: {name}")
        left, right = expected[name], candidate[name]
        if (
            not isinstance(left, np.ndarray)
            or not isinstance(right, np.ndarray)
            or left.dtype != np.dtype("float32")
            or right.dtype != np.dtype("float32")
            or left.shape != right.shape
        ):
            raise ValueError(f"{name} comparison dtype/shape differs")
        exact = _bits_equal(np, left, right)
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


CUDA_PSOLVER1_PERIODIC_SOURCE = r"""// gpuwm-cm1.cuda-psolver1-periodic-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);} 
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);} 
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);} 
__device__ __forceinline__ unsigned long long si(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long ui(unsigned k,unsigned j,unsigned f,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+1u)+f;}
__device__ __forceinline__ unsigned long long vi(unsigned k,unsigned f,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+1u)+f)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long wi(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*nj+j)*ni+i;}
__device__ __forceinline__ unsigned long long fi(unsigned j,unsigned i,unsigned ni){return (unsigned long long)j*(ni+2u)+i;}

extern "C" __global__ void cm1_ps1_face_dt_v1(const float* dtu0,const float* dtv0,float* dtu,float* dtv,unsigned ni,unsigned nj,float dttmp){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(ni+1u)*(nj+1u);for(unsigned long long q=first;q<n;q+=step){unsigned i=q%(ni+1u)+1u,j=q/(ni+1u)+1u;unsigned long long x=fi(j,i,ni);dtu[x]=m32(dttmp,dtu0[x]);dtv[x]=m32(dttmp,dtv0[x]);}}

extern "C" __global__ void cm1_ps1_uv_v1(const float* pp3d,const float* thv,const float* dtu,const float* dtv,const float* uf,const float* vf,const float* ua,const float* uten,const float* va,const float* vten,float* u,float* v,unsigned ni,unsigned nj,unsigned nk,float tem1,float tem2){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*(nj+1u)*(ni+1u);for(unsigned long long q=first;q<n;q+=step){unsigned i=q%(ni+1u)+1u;unsigned long long z=q/(ni+1u);unsigned j=z%(nj+1u)+1u,k=z/(nj+1u)+1u,f=i-1u;float p=m32(tem1,s32(pp3d[si(k,j,i,ni,nj)],pp3d[si(k,j,i-1u,ni,nj)]));p=m32(p,uf[i]);p=m32(p,a32(thv[si(k,j,i,ni,nj)],thv[si(k,j,i-1u,ni,nj)]));unsigned long long ux=ui(k,j,f,ni,nj);u[ux]=s32(a32(ua[ux],m32(dtu[fi(j,i,ni)],uten[ux])),p);unsigned vf0=j-1u;p=m32(tem2,s32(pp3d[si(k,j,i,ni,nj)],pp3d[si(k,j-1u,i,ni,nj)]));p=m32(p,vf[j]);p=m32(p,a32(thv[si(k,j,i,ni,nj)],thv[si(k,j-1u,i,ni,nj)]));unsigned long long vx=vi(k,vf0,i,ni,nj);v[vx]=s32(a32(va[vx],m32(dtv[fi(j,i,ni)],vten[vx])),p);}}

extern "C" __global__ void cm1_ps1_w_v1(const float* mf,const float* c1,const float* c2,const float* pp3d,const float* thv,const float* wa,const float* wten,float* w,unsigned ni,unsigned nj,unsigned nk,float dttmp,float rdz,float cp){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk-1u)*nj*ni;for(unsigned long long q=first;q<n;q+=step){unsigned i0=q%ni,j0=(q/ni)%nj,k=(unsigned)(q/(ni*nj))+2u,i=i0+1u,j=j0+1u;float tem=m32(m32(rdz,cp),mf[si(k,1u,1u,ni,nj)]);float theta=a32(m32(c2[si(k,1u,1u,ni,nj)],thv[si(k,j,i,ni,nj)]),m32(c1[si(k,1u,1u,ni,nj)],thv[si(k-1u,j,i,ni,nj)]));float p=m32(tem,s32(pp3d[si(k,j,i,ni,nj)],pp3d[si(k-1u,j,i,ni,nj)]));p=m32(p,theta);unsigned long long x=wi(k,j0,i0,ni,nj);w[x]=a32(wa[x],m32(dttmp,s32(wten[x],p)));}}

extern "C" __global__ void cm1_ps1_div_pressure_v1(const float* mh,const float* ppi,const float* sten,const float* ppterm,const float* uh,const float* vh,const float* u,const float* v,const float* w,float* div,float* pp3d,unsigned ni,unsigned nj,unsigned nk,float dttmp,float rdx,float rdy,float rdz,float smeps){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)nk*nj*ni;for(unsigned long long q=first;q<n;q+=step){unsigned i0=q%ni,j0=(q/ni)%nj,k=(unsigned)(q/(ni*nj))+1u,i=i0+1u,j=j0+1u;float du=m32(m32(s32(u[ui(k,j,i,ni,nj)],u[ui(k,j,i-1u,ni,nj)]),rdx),uh[i]);float dv=m32(m32(s32(v[vi(k,j,i,ni,nj)],v[vi(k,j-1u,i,ni,nj)]),rdy),vh[j]);float horizontal=a32(du,dv);float vertical=m32(m32(s32(w[wi(k+1u,j0,i0,ni,nj)],w[wi(k,j0,i0,ni,nj)]),rdz),mh[si(k,1u,1u,ni,nj)]);float d=a32(horizontal,vertical);if(fabsf(d)<smeps)d=0.0f;div[si(k,j,i,ni,nj)]=d;float tendency=s32(sten[si(k,j,i,ni,nj)],m32(ppterm[si(k,j,i,ni,nj)],d));float p=a32(ppi[si(k,j,i,ni,nj)],m32(dttmp,tendency));if(fabsf(p)<smeps)p=0.0f;pp3d[si(k,j,i,ni,nj)]=p;}}

extern "C" __global__ void cm1_ps1_bc_v1(float* pp3d,float* u,float* v,float* w,unsigned ni,unsigned nj,unsigned nk,unsigned nrk){if(blockIdx.x||threadIdx.x)return;for(unsigned k=1u;k<=nk;++k){for(unsigned f=0u;f<=ni;++f){u[ui(k,0u,f,ni,nj)]=u[ui(k,nj,f,ni,nj)];u[ui(k,nj+1u,f,ni,nj)]=u[ui(k,1u,f,ni,nj)];}for(unsigned f=0u;f<=nj;++f){v[vi(k,f,0u,ni,nj)]=v[vi(k,f,ni,ni,nj)];v[vi(k,f,ni+1u,ni,nj)]=v[vi(k,f,1u,ni,nj)];}if(nrk<3u){for(unsigned j=0u;j<=nj+1u;++j){pp3d[si(k,j,0u,ni,nj)]=pp3d[si(k,j,ni,ni,nj)];pp3d[si(k,j,ni+1u,ni,nj)]=pp3d[si(k,j,1u,ni,nj)];}for(unsigned i=0u;i<=ni+1u;++i){pp3d[si(k,0u,i,ni,nj)]=pp3d[si(k,nj,i,ni,nj)];pp3d[si(k,nj+1u,i,ni,nj)]=pp3d[si(k,1u,i,ni,nj)];}}}for(unsigned j=0u;j<nj;++j)for(unsigned i=0u;i<ni;++i){w[wi(1u,j,i,ni,nj)]=0.0f;w[wi(nk+1u,j,i,ni,nj)]=0.0f;}}
"""
CUDA_PSOLVER1_PERIODIC_SOURCE_BYTES = CUDA_PSOLVER1_PERIODIC_SOURCE.encode("utf-8")
CUDA_PSOLVER1_PERIODIC_SOURCE_SHA256 = (
    "6806b43a10ad4f9b582b53bc848e708e63533f4cd9c1b394ad36faac8e765dfb"
)
CUDA_PSOLVER1_PERIODIC_KERNEL_NAMES = (
    "cm1_ps1_face_dt_v1",
    "cm1_ps1_uv_v1",
    "cm1_ps1_w_v1",
    "cm1_ps1_div_pressure_v1",
    "cm1_ps1_bc_v1",
)


@dataclass(frozen=True, slots=True)
class CompiledPsolver1PeriodicKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_psolver1_periodic() -> CompiledPsolver1PeriodicKernels:
    actual = hashlib.sha256(CUDA_PSOLVER1_PERIODIC_SOURCE_BYTES).hexdigest()
    if actual != CUDA_PSOLVER1_PERIODIC_SOURCE_SHA256:
        raise RuntimeError(f"packaged psolver1 CUDA source differs from its pin: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(CUDA_PSOLVER1_PERIODIC_SOURCE, "cm1_ps1_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_PSOLVER1_PERIODIC_SOURCE, "cm1_ps1_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", text) for name in CUDA_PSOLVER1_PERIODIC_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks psolver1 entry points")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_PSOLVER1_PERIODIC_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledPsolver1PeriodicKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = psolver1_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver1 device arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid CUDA dtype, shape, or layout")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("psolver1 device arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class Psolver1PeriodicDeviceProof:
    ni: int
    nj: int
    nk: int
    device_id: int
    pointers: tuple[tuple[str, int], ...]
    contract_sha256: str


def prove_cuda_psolver1_periodic_contract(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: Psolver1PeriodicSelection = Psolver1PeriodicSelection(),
) -> Psolver1PeriodicDeviceProof:
    require_psolver1_periodic_selection(selection)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    digest = hashlib.sha256()

    def exact(label: str, left: Any, right: Any, reason: str) -> None:
        if not bool(cupy.array_equal(left.view(cupy.uint32), right.view(cupy.uint32))):
            raise ValueError(reason)
        digest.update(label.encode("ascii") + b"\0")
        digest.update(cupy.asnumpy(left).astype("<f4", copy=False).tobytes(order="C"))

    for name, value in a.items():
        if not bool(cupy.isfinite(value).all().item()):
            raise ValueError(f"{name} contains a non-finite value")
    for name in METRIC_NAMES:
        exact(name, a[name], cupy.ones_like(a[name]), f"{name} must be exactly one")
    expected_dtu0, expected_dtv0 = default_psolver1_periodic_face_weights(ni, nj)
    exact("dtu0", a["dtu0"], cupy.asarray(expected_dtu0), "dtu0 differs from CM1 serial-periodic weights")
    exact("dtv0", a["dtv0"], cupy.asarray(expected_dtv0), "dtv0 differs from CM1 serial-periodic weights")
    for name in ("mh", "mf", "c1", "c2"):
        live = a[name][:, 1 : nj + 1, 1 : ni + 1]
        profile = a[name][:, 1:2, 1:2]
        if not bool(cupy.all(live.view(cupy.uint32) == profile.view(cupy.uint32)).item()):
            raise ValueError(f"{name} must be horizontally homogeneous")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(cupy.asnumpy(profile).astype("<f4", copy=False).tobytes(order="C"))
    for name in ("pp3d", "thv"):
        value = a[name]
        for side, left, right in (
            ("west", value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni]),
            ("east", value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1]),
            ("south", value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :]),
            ("north", value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :]),
        ):
            exact(f"{name}-{side}", left, right, f"{name} {side} periodic halo is invalid")
    for name in ("ua", "uten"):
        exact(f"{name}-x", a[name][1 : nk + 1, 1 : nj + 1, 0], a[name][1 : nk + 1, 1 : nj + 1, ni], f"{name} duplicate x faces differ")
    for name in ("va", "vten"):
        exact(f"{name}-y", a[name][1 : nk + 1, 0, 1 : ni + 1], a[name][1 : nk + 1, nj, 1 : ni + 1], f"{name} duplicate y faces differ")
    exact("wa-rigid", a["wa"][[1, nk + 1]], cupy.zeros_like(a["wa"][[1, nk + 1]]), "wa rigid faces must be positive zero")
    return Psolver1PeriodicDeviceProof(
        ni,
        nj,
        nk,
        int(cupy.cuda.Device().id),
        tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES),
        digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CudaPsolver1PeriodicLaunch:
    ready_event: Any
    compiled: CompiledPsolver1PeriodicKernels
    device_proof: Psolver1PeriodicDeviceProof
    receipt: dict[str, Any]


def launch_cuda_psolver1_periodic_stage(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    device_proof: Psolver1PeriodicDeviceProof,
    cp: float = CM1_PSOLVER1_CP_F32,
    smeps: float = CM1_PSOLVER1_SMEPS_F32,
    selection: Psolver1PeriodicSelection = Psolver1PeriodicSelection(),
    stream: Any | None = None,
    compiled: CompiledPsolver1PeriodicKernels | None = None,
) -> CudaPsolver1PeriodicLaunch:
    require_psolver1_periodic_selection(selection)
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    for name, value in (("dt", dt), ("rdx", rdx), ("rdy", rdy), ("rdz", rdz), ("cp", cp), ("smeps", smeps)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    expected = tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES)
    if (
        (device_proof.ni, device_proof.nj, device_proof.nk, device_proof.device_id)
        != (ni, nj, nk, int(cupy.cuda.Device().id))
        or device_proof.pointers != expected
    ):
        raise ValueError("psolver1 device proof does not match these allocations")
    compiled = compile_cuda_psolver1_periodic() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    face_dt, uv, w, div_pressure, bc = compiled.kernels
    f32, u32 = cupy.float32, cupy.uint32
    maximum = max((ni + 1) * (nj + 1), nk * (nj + 1) * (ni + 1), (nk - 1) * nj * ni, nk * nj * ni)
    block = (CUDA_PSOLVER1_PERIODIC_THREADS, 1, 1)
    grid = ((maximum + CUDA_PSOLVER1_PERIODIC_THREADS - 1) // CUDA_PSOLVER1_PERIODIC_THREADS, 1, 1)
    dttmp = np.float32(rk_stage_dt_f32(dt, nrk))
    rdx32, rdy32, rdz32, cp32, smeps32 = map(np.float32, (rdx, rdy, rdz, cp, smeps))
    tem1 = np.float32(np.float32(np.float32(dttmp * rdx32) * cp32) * np.float32(0.5))
    tem2 = np.float32(np.float32(np.float32(dttmp * rdy32) * cp32) * np.float32(0.5))
    face_dt(grid, block, (a["dtu0"], a["dtv0"], a["dtu"], a["dtv"], u32(ni), u32(nj), f32(dttmp)), stream=stream)
    uv(grid, block, (a["pp3d"], a["thv"], a["dtu"], a["dtv"], a["uf"], a["vf"], a["ua"], a["uten"], a["va"], a["vten"], a["u3d"], a["v3d"], u32(ni), u32(nj), u32(nk), f32(tem1), f32(tem2)), stream=stream)
    w(grid, block, (a["mf"], a["c1"], a["c2"], a["pp3d"], a["thv"], a["wa"], a["wten"], a["w3d"], u32(ni), u32(nj), u32(nk), f32(dttmp), f32(rdz32), f32(cp32)), stream=stream)
    div_pressure(grid, block, (a["mh"], a["ppi"], a["sten"], a["ppterm"], a["uh"], a["vh"], a["u3d"], a["v3d"], a["w3d"], a["div"], a["pp3d"], u32(ni), u32(nj), u32(nk), f32(dttmp), f32(rdx32), f32(rdy32), f32(rdz32), f32(smeps32)), stream=stream)
    bc((1, 1, 1), (1, 1, 1), (a["pp3d"], a["u3d"], a["v3d"], a["w3d"], u32(ni), u32(nj), u32(nk), u32(nrk)), stream=stream)
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_PSOLVER1_PERIODIC_SCHEMA,
        "abi": CUDA_PSOLVER1_PERIODIC_ABI,
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "dimensions": {"ni": ni, "nj": nj, "nk": nk},
        "cadence": {
            "nrk": nrk,
            "dttmp_f32": float(dttmp),
            "velocity_boundary_fills": 3,
            "pressure_boundary_fills": 1 if nrk < 3 else 0,
            "live_sten_consumed": True,
        },
        "device_contract_proof_sha256": device_proof.contract_sha256,
        "kernel": {"source_sha256": CUDA_PSOLVER1_PERIODIC_SOURCE_SHA256, "names": list(CUDA_PSOLVER1_PERIODIC_KERNEL_NAMES), "target_arch": compiled.target_arch},
        "runtime": dict(compiled.runtime_identity),
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_PSOLVER1_CLAIM_LIMIT,
    }
    return CudaPsolver1PeriodicLaunch(event, compiled, device_proof, receipt)


__all__ = (
    "ALL_ARRAY_NAMES",
    "CM1_PSOLVER1_CLAIM_LIMIT",
    "CM1_PSOLVER1_CP_F32",
    "COMPARE_OUTPUT_NAMES",
    "CUDA_PSOLVER1_PERIODIC_ABI",
    "CUDA_PSOLVER1_PERIODIC_SCHEMA",
    "CUDA_PSOLVER1_PERIODIC_SOURCE_BYTES",
    "CUDA_PSOLVER1_PERIODIC_SOURCE_SHA256",
    "CompiledPsolver1PeriodicKernels",
    "CpuPsolver1StageDiagnostics",
    "CpuPsolver1ThreeStageDiagnostics",
    "CudaPsolver1PeriodicLaunch",
    "Psolver1PeriodicDeviceProof",
    "Psolver1PeriodicSelection",
    "SOURCE_ANCHORS",
    "compare_psolver1_periodic_raw",
    "compile_cuda_psolver1_periodic",
    "cpu_psolver1_periodic_stage_inplace",
    "cpu_psolver1_periodic_three_stage_inplace",
    "default_psolver1_periodic_face_weights",
    "fill_periodic_scalar_inplace",
    "fill_periodic_uv_inplace",
    "launch_cuda_psolver1_periodic_stage",
    "prove_cuda_psolver1_periodic_contract",
    "psolver1_periodic_shapes",
    "require_psolver1_periodic_selection",
    "validate_psolver1_periodic_contract",
)
