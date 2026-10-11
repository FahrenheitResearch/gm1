"""CM1 r21.1 flat serial-periodic ``psolver=6`` acoustic operator.

This is the compressible-Boussinesq ``soundcb.F`` branch with supplied
``nsound`` and ``csound``.  The implementation exposes one RK stage at a time
and a separate source-ordered ``solve3.F`` pressure completion.  It is a
bounded operator family, not the open-boundary/immersed-boundary stock wind
tunnel case and not a complete large step.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

from .cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32
from .cuda_sound3_periodic import (
    _canonical_arch,
    _image_options,
    default_periodic_face_weights,
    fill_periodic_scalar_inplace,
    fill_periodic_uv_inplace,
    scalar_shape_zyx,
    u_shape_zyx,
    v_shape_zyx,
    w_shape_zyx,
)
from .errors import refuse


CUDA_PSOLVER6_PERIODIC_SCHEMA = "gpuwm-cm1.cuda-psolver6-periodic/v1"
CUDA_PSOLVER6_PERIODIC_ABI = "gpuwm-cm1.cuda-psolver6-periodic-kernels/v1"
CUDA_PSOLVER6_PERIODIC_THREADS = 128
CM1_PSOLVER6_KDIV_F32 = 0.10
CM1_PSOLVER6_SMEPS_F32 = 1.0e-30
CM1_PSOLVER6_CLAIM_LIMIT = (
    "CM1 r21.1 soundcb.F compressible-Boussinesq psolver=6 acoustic/RK "
    "operator and explicit solve3 ppx completion for a flat Cartesian, "
    "single-process, four-periodic, eqtset=1, apmasscon=0 profile with "
    "supplied nsound/csound, held momentum tendencies, uniform horizontal "
    "metrics, horizontally homogeneous mh/mf, and rigid vertical w faces; "
    "no adaptive nsound derivation, terrain, MPI, axisymmetry, open/radiative "
    "or rigid lateral boundaries, immersed boundary, convergence/updraft "
    "forcing, thermodynamic/SGS composition, complete solve2/solve3, stock "
    "wind-tunnel case, coupled trajectory, forecast, product, or performance claim"
)
SOURCE_ANCHORS = {
    "dispatch": "src/solve2.F:1154-1172",
    "schedule": "src/soundcb.F:83-111",
    "stage_prepare": "src/soundcb.F:113-198",
    "flat_cartesian_uv": "src/soundcb.F:255-306",
    "flat_vertical_w": "src/soundcb.F:385-407",
    "flat_divergence": "src/misclibs.F:108-141",
    "pressure_filter": "src/soundcb.F:451-490",
    "acoustic_averages": "src/soundcb.F:493-548",
    "stage_phi_and_ppx_commit": "src/soundcb.F:552-586",
    "stage_velocity_boundaries": "src/solve2.F:1662-1689",
    "solve3_ppx_completion": "src/solve3.F:270-305",
    "periodic_boundaries": "src/bc.F:180-1143",
}


@dataclass(frozen=True, slots=True)
class Psolver6PeriodicSelection:
    psolver: int = 6
    terrain_flag: bool = False
    axisymm: int = 0
    mpi: bool = False
    wbc: int = 1
    ebc: int = 1
    sbc: int = 1
    nbc: int = 1
    irbc: int = 4
    roflux: int = 0
    nudgeobc: int = 0
    eqtset: int = 1
    apmasscon: int = 0
    do_ib: bool = False
    convinit: int = 0
    wnudge: int = 0


def require_psolver6_periodic_selection(selection: Psolver6PeriodicSelection) -> None:
    checks = (
        ("psolver", selection.psolver, 6, "ARWEN-PS6-PSOLVER"),
        ("terrain_flag", selection.terrain_flag, False, "ARWEN-PS6-TERRAIN"),
        ("axisymm", selection.axisymm, 0, "ARWEN-PS6-AXISYMM"),
        ("mpi", selection.mpi, False, "ARWEN-PS6-MPI"),
        ("wbc", selection.wbc, 1, "ARWEN-PS6-WBC"),
        ("ebc", selection.ebc, 1, "ARWEN-PS6-EBC"),
        ("sbc", selection.sbc, 1, "ARWEN-PS6-SBC"),
        ("nbc", selection.nbc, 1, "ARWEN-PS6-NBC"),
        ("irbc", selection.irbc, 4, "ARWEN-PS6-IRBC"),
        ("roflux", selection.roflux, 0, "ARWEN-PS6-ROFLUX"),
        ("nudgeobc", selection.nudgeobc, 0, "ARWEN-PS6-NUDGE-OBC"),
        ("eqtset", selection.eqtset, 1, "ARWEN-PS6-EQTSET"),
        ("apmasscon", selection.apmasscon, 0, "ARWEN-PS6-APMASSCON"),
        ("do_ib", selection.do_ib, False, "ARWEN-PS6-IMMERSED"),
        ("convinit", selection.convinit, 0, "ARWEN-PS6-CONVINIT"),
        ("wnudge", selection.wnudge, 0, "ARWEN-PS6-WNUDGE"),
    )
    for option, value, expected, code in checks:
        if value != expected:
            refuse(
                code=code,
                gate="psolver6-flat-periodic-selection",
                reason=(
                    f"the bounded psolver=6 family requires {option}={expected!r}; "
                    "the official alternate branch is not implemented here"
                ),
                remedy="select the exact flat serial-periodic family or use a separately ratified operator",
                option=option,
                value=value,
            )


SCALAR_NAMES = ("mh", "mf", "phi1", "phi2", "ppx", "ppd", "pp_filter", "div")
U_NAMES = ("ua", "u3d", "uten", "rru")
V_NAMES = ("va", "v3d", "vten", "rrv")
W_NAMES = ("wa", "w3d", "wten", "rrw")
METRIC_NAMES = ("uf", "vf", "uh", "vh")
ALL_ARRAY_NAMES = SCALAR_NAMES + U_NAMES + V_NAMES + W_NAMES + METRIC_NAMES
WRITABLE_NAMES = (
    "phi1", "phi2", "ppx", "ppd", "pp_filter", "div",
    "u3d", "rru", "v3d", "rrv", "w3d", "rrw",
)
COMPARE_OUTPUT_NAMES = WRITABLE_NAMES


def _validate_dimensions(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")


def psolver6_periodic_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_dimensions(ni, nj, nk)
    scalar = scalar_shape_zyx(ni, nj, nk)
    result = {name: scalar for name in SCALAR_NAMES}
    result.update({name: u_shape_zyx(ni, nj, nk) for name in U_NAMES})
    result.update({name: v_shape_zyx(ni, nj, nk) for name in V_NAMES})
    result.update({name: w_shape_zyx(ni, nj, nk) for name in W_NAMES})
    result.update({"uf": (ni + 2,), "vf": (nj + 2,), "uh": (ni + 2,), "vh": (nj + 2,)})
    return result


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = psolver6_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver6 arrays: {', '.join(missing)}")
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
        result[name] = value
    if len({int(value.ctypes.data) for value in result.values()}) != len(result):
        raise ValueError("psolver6 arrays must not alias")
    return result


def _bits_equal(np: Any, left: Any, right: Any) -> bool:
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def validate_psolver6_periodic_contract(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: Psolver6PeriodicSelection = Psolver6PeriodicSelection(),
) -> None:
    require_psolver6_periodic_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    for name in METRIC_NAMES:
        if not _bits_equal(np, a[name], np.ones_like(a[name])):
            raise ValueError(f"{name} must be exactly one for this uniform flat-grid slice")
    for name in ("mh", "mf"):
        live = a[name][:, 1 : nj + 1, 1 : ni + 1]
        if not _bits_equal(np, live, np.broadcast_to(a[name][:, 1:2, 1:2], live.shape)):
            raise ValueError(f"{name} must be horizontally homogeneous for flat soundcb.F")
    for name in ("phi1", "phi2", "ppx"):
        value = a[name]
        checks = (
            (value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni], "west"),
            (value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1], "east"),
            (value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :], "south"),
            (value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :], "north"),
        )
        for left, right, side in checks:
            if not _bits_equal(np, left, right):
                raise ValueError(f"{name} {side} periodic halo is invalid")
    for name in ("ua", "u3d"):
        if not _bits_equal(np, a[name][1 : nk + 1, 1 : nj + 1, 0], a[name][1 : nk + 1, 1 : nj + 1, ni]):
            raise ValueError(f"{name} duplicate periodic x faces differ")
    for name in ("va", "v3d"):
        if not _bits_equal(np, a[name][1 : nk + 1, 0, 1 : ni + 1], a[name][1 : nk + 1, nj, 1 : ni + 1]):
            raise ValueError(f"{name} duplicate periodic y faces differ")
    for name in ("wa", "w3d"):
        if np.any(a[name][[1, nk + 1], :, :].view(np.uint32) != np.uint32(0)):
            raise ValueError(f"{name} rigid bottom/top faces must be positive zero")


def _add(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) + np.float32(y))


def _sub(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) - np.float32(y))


def _mul(np: Any, x: Any, y: Any) -> Any:
    return np.float32(np.float32(x) * np.float32(y))


@dataclass(frozen=True, slots=True)
class CpuPsolver6StageDiagnostics:
    nrk: int
    nloop: int
    dts_f32: float
    dttmp_f32: float
    pressure_boundary_fills: int
    stage_phi_boundary_fills: int
    stage_velocity_boundary_fills: int
    ppx_filter_written: bool
    phi1_written: bool
    boundary_trace_sha256: str


def cpu_psolver6_periodic_stage_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dt: float,
    nsound: int,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    csound: float,
    kdiv: float = CM1_PSOLVER6_KDIV_F32,
    smeps: float = CM1_PSOLVER6_SMEPS_F32,
    selection: Psolver6PeriodicSelection = Psolver6PeriodicSelection(),
) -> CpuPsolver6StageDiagnostics:
    validate_psolver6_periodic_contract(
        arrays, ni=ni, nj=nj, nk=nk, selection=selection
    )
    if not math.isfinite(csound) or csound <= 0.0:
        raise ValueError("csound must be finite and positive")
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    nloop, dts_value = sound3_stage_schedule_f32(dt, nsound, nrk)
    dts, rdx, rdy, rdz = map(np.float32, (dts_value, rdx, rdy, rdz))
    csound, kdiv, smeps = map(np.float32, (csound, kdiv, smeps))
    pressure_scale = _mul(np, _mul(np, dts, csound), csound)
    a["ppd"][1 : nk + 1] = a["ppx"][1 : nk + 1]
    a["pp_filter"].fill(np.float32(0.0))
    a["div"].fill(np.float32(0.0))
    a["rru"].fill(np.float32(0.0))
    a["rrv"].fill(np.float32(0.0))
    a["rrw"].fill(np.float32(0.0))
    if nrk != 1:
        np.copyto(a["u3d"], a["ua"])
        np.copyto(a["v3d"], a["va"])
        np.copyto(a["w3d"], a["wa"])
        a["phi2"][1 : nk + 1] = a["phi1"][1 : nk + 1]
    trace = hashlib.sha256()
    for n in range(1, nloop + 1):
        for k in range(1, nk + 1):
            for j in range(1, nj + 1):
                for f in range(ni + 1):
                    grad = _mul(np, rdx, _sub(np, a["ppd"][k, j, f + 1], a["ppd"][k, j, f]))
                    grad = _mul(np, grad, a["uf"][f + 1])
                    a["u3d"][k, j, f] = _add(np, a["u3d"][k, j, f], _mul(np, dts, _sub(np, a["uten"][k, j, f], grad)))
            for f in range(nj + 1):
                for i in range(1, ni + 1):
                    grad = _mul(np, rdy, _sub(np, a["ppd"][k, f + 1, i], a["ppd"][k, f, i]))
                    grad = _mul(np, grad, a["vf"][f + 1])
                    a["v3d"][k, f, i] = _add(np, a["v3d"][k, f, i], _mul(np, dts, _sub(np, a["vten"][k, f, i], grad)))
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                for k in range(2, nk + 1):
                    grad = _mul(np, _mul(np, rdz, a["mf"][k, 1, 1]), _sub(np, a["ppd"][k, j, i], a["ppd"][k - 1, j, i]))
                    a["w3d"][k, j0, i0] = _add(np, a["w3d"][k, j0, i0], _mul(np, dts, _sub(np, a["wten"][k, j0, i0], grad)))
                for k in range(1, nk + 1):
                    du = _mul(np, _mul(np, _sub(np, a["u3d"][k, j, i], a["u3d"][k, j, i - 1]), rdx), a["uh"][i])
                    dv = _mul(np, _mul(np, _sub(np, a["v3d"][k, j, i], a["v3d"][k, j - 1, i]), rdy), a["vh"][j])
                    dw = _mul(np, _mul(np, _sub(np, a["w3d"][k + 1, j0, i0], a["w3d"][k, j0, i0]), rdz), a["mh"][k, 1, 1])
                    div = _add(np, _add(np, du, dv), dw)
                    if bool(np.abs(div) < smeps):
                        div = np.float32(0.0)
                    a["div"][k, j, i] = div
                    old = a["phi2"][k, j, i]
                    updated = _sub(np, old, _mul(np, pressure_scale, div))
                    if bool(np.abs(updated) < smeps):
                        updated = np.float32(0.0)
                    a["phi2"][k, j, i] = updated
                    filt = _mul(np, kdiv, _sub(np, updated, old))
                    a["pp_filter"][k, j, i] = filt
                    a["ppd"][k, j, i] = _add(np, updated, filt)
                    if nrk == 3 and n == nloop:
                        a["ppx"][k, j, i] = filt
        if n < nloop:
            fill_periodic_scalar_inplace(a["ppd"], ni=ni, nj=nj, nk=nk)
            trace.update(b"ppd\0" + n.to_bytes(4, "little"))
            trace.update(a["ppd"].tobytes(order="C"))
        tavg = np.float32(1.0 / nloop)
        for k in range(1, nk + 1):
            for j in range(1, nj + 1):
                for f in range(ni + 1):
                    total = _add(np, a["rru"][k, j, f], a["u3d"][k, j, f])
                    a["rru"][k, j, f] = total if n < nloop else _mul(np, total, tavg)
            for f in range(nj + 1):
                for i in range(1, ni + 1):
                    total = _add(np, a["rrv"][k, f, i], a["v3d"][k, f, i])
                    a["rrv"][k, f, i] = total if n < nloop else _mul(np, total, tavg)
            if k >= 2:
                for j0 in range(nj):
                    for i0 in range(ni):
                        total = _add(np, a["rrw"][k, j0, i0], a["w3d"][k, j0, i0])
                        a["rrw"][k, j0, i0] = total if n < nloop else _mul(np, total, tavg)
    fill_periodic_scalar_inplace(a["phi2"], ni=ni, nj=nj, nk=nk)
    fill_periodic_uv_inplace(a["u3d"], a["v3d"], ni=ni, nj=nj, nk=nk)
    a["w3d"][[1, nk + 1], :, :] = np.float32(0.0)
    trace.update(b"stage-boundaries\0")
    trace.update(a["phi2"].tobytes(order="C"))
    trace.update(a["u3d"].tobytes(order="C"))
    trace.update(a["v3d"].tobytes(order="C"))
    if nrk == 3:
        a["phi1"][1 : nk + 1] = a["phi2"][1 : nk + 1]
    return CpuPsolver6StageDiagnostics(
        nrk, nloop, float(dts), rk_stage_dt_f32(dt, nrk), nloop - 1,
        1, 1, nrk == 3, nrk == 3, trace.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CpuPsolver6Result:
    cadence: tuple[CpuPsolver6StageDiagnostics, ...]


def cpu_psolver6_periodic_three_stage_inplace(
    arrays: MutableMapping[str, Any], **kwargs: Any
) -> CpuPsolver6Result:
    return CpuPsolver6Result(
        tuple(cpu_psolver6_periodic_stage_inplace(arrays, nrk=nrk, **kwargs) for nrk in (1, 2, 3))
    )


def cpu_psolver6_periodic_solve3_complete_inplace(
    arrays: MutableMapping[str, Any], *, ni: int, nj: int, nk: int
) -> None:
    a = _require_host_arrays(arrays, ni, nj, nk)
    np = importlib.import_module("numpy")
    a["ppx"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1] = np.add(
        a["phi1"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1],
        a["ppx"][1 : nk + 1, 1 : nj + 1, 1 : ni + 1],
        dtype=np.float32,
    )
    fill_periodic_scalar_inplace(a["ppx"], ni=ni, nj=nj, nk=nk)


def compare_psolver6_periodic_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    missing = [name for name in COMPARE_OUTPUT_NAMES if name not in expected or name not in candidate]
    if missing:
        raise ValueError(f"missing comparison outputs: {', '.join(missing)}")
    fields = {}
    for name in COMPARE_OUTPUT_NAMES:
        left, right = expected[name], candidate[name]
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise TypeError(f"{name} comparison values must be numpy.ndarray")
        if left.dtype != np.dtype("float32") or right.dtype != np.dtype("float32") or left.shape != right.shape:
            raise ValueError(f"{name} comparison dtype/shape differs")
        exact = _bits_equal(np, left, right)
        fields[name] = {
            "exact": exact,
            "expected_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest(),
            "candidate_sha256": hashlib.sha256(right.tobytes(order="C")).hexdigest(),
        }
    return {
        "mode": "raw-bit", "max_ulp": 0, "null_or_missing_is_pass": False,
        "passed": all(item["exact"] for item in fields.values()), "fields": fields,
    }


CUDA_PSOLVER6_PERIODIC_SOURCE = r"""// gpuwm-cm1.cuda-psolver6-periodic-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);} 
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);} 
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);} 
__device__ __forceinline__ unsigned long long si(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long ui(unsigned k,unsigned j,unsigned f,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+1u)+f;}
__device__ __forceinline__ unsigned long long vi(unsigned k,unsigned f,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+1u)+f)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long wi(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*nj+j)*ni+i;}

extern "C" __global__ void cm1_ps6_prepare_v1(
 const float* phi1,const float* ppx,const float* ua,const float* va,const float* wa,
 float* phi2,float* ppd,float* filt,float* div,float* u,float* v,float* w,
 float* rru,float* rrv,float* rrw,unsigned ni,unsigned nj,unsigned nk,unsigned nrk){
 unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;
 unsigned long long ns=(unsigned long long)(nk+2u)*(nj+2u)*(ni+2u),nu=(unsigned long long)(nk+2u)*(nj+2u)*(ni+1u),nv=(unsigned long long)(nk+2u)*(nj+1u)*(ni+2u),nw=(unsigned long long)(nk+2u)*nj*ni;
 for(unsigned long long x=first;x<ns;x+=step){unsigned i=x%(ni+2u);unsigned long long q=x/(ni+2u);unsigned j=q%(nj+2u),k=q/(nj+2u);filt[x]=0.0f;div[x]=0.0f;if(k>=1u&&k<=nk){ppd[x]=ppx[x];if(nrk!=1u)phi2[x]=phi1[x];}}
 for(unsigned long long x=first;x<nu;x+=step){rru[x]=0.0f;if(nrk!=1u)u[x]=ua[x];}
 for(unsigned long long x=first;x<nv;x+=step){rrv[x]=0.0f;if(nrk!=1u)v[x]=va[x];}
 for(unsigned long long x=first;x<nw;x+=step){rrw[x]=0.0f;if(nrk!=1u)w[x]=wa[x];}
}

extern "C" __global__ void cm1_ps6_uv_v1(const float* ppd,const float* uten,const float* vten,const float* uf,const float* vf,float* u,float* v,unsigned ni,unsigned nj,unsigned nk,float dts,float rdx,float rdy){
 unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni;
 for(unsigned long long q=first;q<nu;q+=step){unsigned f=q%(ni+1u);unsigned long long z=q/(ni+1u);unsigned j=z%nj+1u,k=z/nj+1u;unsigned long long x=ui(k,j,f,ni,nj);float g=m32(m32(rdx,s32(ppd[si(k,j,f+1u,ni,nj)],ppd[si(k,j,f,ni,nj)])),uf[f+1u]);u[x]=a32(u[x],m32(dts,s32(uten[x],g)));}
 for(unsigned long long q=first;q<nv;q+=step){unsigned i=q%ni+1u;unsigned long long z=q/ni;unsigned f=z%(nj+1u),k=z/(nj+1u)+1u;unsigned long long x=vi(k,f,i,ni,nj);float g=m32(m32(rdy,s32(ppd[si(k,f+1u,i,ni,nj)],ppd[si(k,f,i,ni,nj)])),vf[f+1u]);v[x]=a32(v[x],m32(dts,s32(vten[x],g)));}
}

extern "C" __global__ void cm1_ps6_column_v1(const float* mh,const float* mf,const float* uh,const float* vh,const float* uten_w,const float* u,const float* v,float* w,float* phi2,float* ppx,float* ppd,float* filt,float* div,unsigned ni,unsigned nj,unsigned nk,unsigned nrk,unsigned n,unsigned nloop,float dts,float rdx,float rdy,float rdz,float pressure_scale,float kdiv,float smeps){
 unsigned first=blockIdx.x*blockDim.x+threadIdx.x,step=blockDim.x*gridDim.x,ncol=ni*nj;
 for(unsigned col=first;col<ncol;col+=step){unsigned j=col/ni+1u,i=col%ni+1u,j0=j-1u,i0=i-1u;
  for(unsigned k=2u;k<=nk;++k){unsigned long long x=si(k,j,i,ni,nj),b=si(k-1u,j,i,ni,nj),wx=wi(k,j0,i0,ni,nj);float g=m32(m32(rdz,mf[si(k,1u,1u,ni,nj)]),s32(ppd[x],ppd[b]));w[wx]=a32(w[wx],m32(dts,s32(uten_w[wx],g)));}
  for(unsigned k=1u;k<=nk;++k){unsigned long long x=si(k,j,i,ni,nj);float du=m32(m32(s32(u[ui(k,j,i,ni,nj)],u[ui(k,j,i-1u,ni,nj)]),rdx),uh[i]);float dv=m32(m32(s32(v[vi(k,j,i,ni,nj)],v[vi(k,j-1u,i,ni,nj)]),rdy),vh[j]);float dw=m32(m32(s32(w[wi(k+1u,j0,i0,ni,nj)],w[wi(k,j0,i0,ni,nj)]),rdz),mh[si(k,1u,1u,ni,nj)]);float d=a32(a32(du,dv),dw);if(fabsf(d)<smeps)d=0.0f;div[x]=d;float old=phi2[x],upd=s32(old,m32(pressure_scale,d));if(fabsf(upd)<smeps)upd=0.0f;phi2[x]=upd;float flt=m32(kdiv,s32(upd,old));filt[x]=flt;ppd[x]=a32(upd,flt);if(nrk==3u&&n==nloop)ppx[x]=flt;}
 }}

extern "C" __global__ void cm1_ps6_average_v1(const float* u,const float* v,const float* w,float* rru,float* rrv,float* rrw,unsigned ni,unsigned nj,unsigned nk,unsigned n,unsigned nloop){
 unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,nw=(unsigned long long)(nk-1u)*nj*ni;float t=1.0f/(float)nloop;
 for(unsigned long long q=first;q<nu;q+=step){unsigned f=q%(ni+1u);unsigned long long z=q/(ni+1u);unsigned j=z%nj+1u,k=z/nj+1u;unsigned long long x=ui(k,j,f,ni,nj);float s=a32(rru[x],u[x]);rru[x]=(n<nloop)?s:m32(s,t);}
 for(unsigned long long q=first;q<nv;q+=step){unsigned i=q%ni+1u;unsigned long long z=q/ni;unsigned f=z%(nj+1u),k=z/(nj+1u)+1u;unsigned long long x=vi(k,f,i,ni,nj);float s=a32(rrv[x],v[x]);rrv[x]=(n<nloop)?s:m32(s,t);}
 for(unsigned long long q=first;q<nw;q+=step){unsigned i=q%ni,j=(q/ni)%nj,k=(unsigned)(q/(ni*nj))+2u;unsigned long long x=wi(k,j,i,ni,nj);float s=a32(rrw[x],w[x]);rrw[x]=(n<nloop)?s:m32(s,t);}
}

extern "C" __global__ void cm1_ps6_bcp_v1(float* p,unsigned ni,unsigned nj,unsigned nk){if(blockIdx.x||threadIdx.x)return;for(unsigned k=1u;k<=nk;++k){for(unsigned j=0u;j<=nj+1u;++j){p[si(k,j,0u,ni,nj)]=p[si(k,j,ni,ni,nj)];p[si(k,j,ni+1u,ni,nj)]=p[si(k,j,1u,ni,nj)];}for(unsigned i=0u;i<=ni+1u;++i){p[si(k,0u,i,ni,nj)]=p[si(k,nj,i,ni,nj)];p[si(k,nj+1u,i,ni,nj)]=p[si(k,1u,i,ni,nj)];}}}

extern "C" __global__ void cm1_ps6_stage_bc_v1(float* phi1,float* phi2,float* u,float* v,float* w,unsigned ni,unsigned nj,unsigned nk,unsigned nrk){if(blockIdx.x||threadIdx.x)return;for(unsigned k=1u;k<=nk;++k){for(unsigned j=0u;j<=nj+1u;++j){phi2[si(k,j,0u,ni,nj)]=phi2[si(k,j,ni,ni,nj)];phi2[si(k,j,ni+1u,ni,nj)]=phi2[si(k,j,1u,ni,nj)];}for(unsigned i=0u;i<=ni+1u;++i){phi2[si(k,0u,i,ni,nj)]=phi2[si(k,nj,i,ni,nj)];phi2[si(k,nj+1u,i,ni,nj)]=phi2[si(k,1u,i,ni,nj)];}for(unsigned f=0u;f<=ni;++f){u[ui(k,0u,f,ni,nj)]=u[ui(k,nj,f,ni,nj)];u[ui(k,nj+1u,f,ni,nj)]=u[ui(k,1u,f,ni,nj)];}for(unsigned f=0u;f<=nj;++f){v[vi(k,f,0u,ni,nj)]=v[vi(k,f,ni,ni,nj)];v[vi(k,f,ni+1u,ni,nj)]=v[vi(k,f,1u,ni,nj)];}if(nrk==3u){for(unsigned j=0u;j<=nj+1u;++j)for(unsigned i=0u;i<=ni+1u;++i)phi1[si(k,j,i,ni,nj)]=phi2[si(k,j,i,ni,nj)];}}for(unsigned j=0u;j<nj;++j)for(unsigned i=0u;i<ni;++i){w[wi(1u,j,i,ni,nj)]=0.0f;w[wi(nk+1u,j,i,ni,nj)]=0.0f;}}

extern "C" __global__ void cm1_ps6_solve3_v1(const float* phi1,float* ppx,unsigned ni,unsigned nj,unsigned nk){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;for(unsigned long long q=first;q<n;q+=step){unsigned i=q%ni+1u;unsigned long long z=q/ni;unsigned j=z%nj+1u,k=z/nj+1u;unsigned long long x=si(k,j,i,ni,nj);ppx[x]=a32(phi1[x],ppx[x]);}}
"""
CUDA_PSOLVER6_PERIODIC_SOURCE_BYTES = CUDA_PSOLVER6_PERIODIC_SOURCE.encode("utf-8")
CUDA_PSOLVER6_PERIODIC_SOURCE_SHA256 = (
    "1a7df9a9520f4b8683f7b939d92dfcd6f864ef12fc24a0124e756612c79d1fb9"
)
CUDA_PSOLVER6_PERIODIC_KERNEL_NAMES = (
    "cm1_ps6_prepare_v1", "cm1_ps6_uv_v1", "cm1_ps6_column_v1",
    "cm1_ps6_average_v1", "cm1_ps6_bcp_v1", "cm1_ps6_stage_bc_v1",
    "cm1_ps6_solve3_v1",
)


@dataclass(frozen=True, slots=True)
class CompiledPsolver6PeriodicKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_psolver6_periodic() -> CompiledPsolver6PeriodicKernels:
    if hashlib.sha256(CUDA_PSOLVER6_PERIODIC_SOURCE_BYTES).hexdigest() != CUDA_PSOLVER6_PERIODIC_SOURCE_SHA256:
        raise RuntimeError("packaged psolver6 CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(CUDA_PSOLVER6_PERIODIC_SOURCE, "cm1_ps6_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_PSOLVER6_PERIODIC_SOURCE, "cm1_ps6_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", text) for name in CUDA_PSOLVER6_PERIODIC_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks psolver6 entry points")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_PSOLVER6_PERIODIC_KERNEL_NAMES)
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
    return CompiledPsolver6PeriodicKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = psolver6_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing psolver6 device arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid CUDA dtype, shape, or layout")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("psolver6 device arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class Psolver6PeriodicDeviceProof:
    ni: int
    nj: int
    nk: int
    device_id: int
    pointers: tuple[tuple[str, int], ...]
    contract_sha256: str


def prove_cuda_psolver6_periodic_contract(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int,
    selection: Psolver6PeriodicSelection = Psolver6PeriodicSelection(),
) -> Psolver6PeriodicDeviceProof:
    require_psolver6_periodic_selection(selection)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    digest = hashlib.sha256()

    def prove_exact(label: str, left: Any, right: Any, reason: str) -> None:
        if not bool(cupy.array_equal(left.view(cupy.uint32), right.view(cupy.uint32))):
            raise ValueError(reason)
        digest.update(label.encode("ascii") + b"\0")
        digest.update(
            cupy.asnumpy(left).astype("<f4", copy=False).tobytes(order="C")
        )

    for name in METRIC_NAMES:
        prove_exact(
            name,
            a[name],
            cupy.ones_like(a[name]),
            f"{name} must be exactly one for this uniform flat-grid slice",
        )
    for name in ("mh", "mf"):
        live = a[name][:, 1 : nj + 1, 1 : ni + 1]
        profile = a[name][:, 1:2, 1:2]
        if not bool(
            cupy.all(live.view(cupy.uint32) == profile.view(cupy.uint32)).item()
        ):
            raise ValueError(f"{name} must be horizontally homogeneous for flat soundcb.F")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(
            cupy.asnumpy(profile).astype("<f4", copy=False).tobytes(order="C")
        )
    for name in ("phi1", "phi2", "ppx"):
        value = a[name]
        for side, left, right in (
            ("west", value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni]),
            ("east", value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1]),
            ("south", value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :]),
            ("north", value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :]),
        ):
            prove_exact(
                f"{name}-{side}", left, right,
                f"{name} {side} periodic halo is invalid",
            )
    for name in ("ua", "u3d"):
        prove_exact(
            f"{name}-duplicate-x-face",
            a[name][1 : nk + 1, 1 : nj + 1, 0],
            a[name][1 : nk + 1, 1 : nj + 1, ni],
            f"{name} duplicate periodic x faces differ",
        )
    for name in ("va", "v3d"):
        prove_exact(
            f"{name}-duplicate-y-face",
            a[name][1 : nk + 1, 0, 1 : ni + 1],
            a[name][1 : nk + 1, nj, 1 : ni + 1],
            f"{name} duplicate periodic y faces differ",
        )
    for name in ("wa", "w3d"):
        prove_exact(
            f"{name}-rigid-faces",
            a[name][[1, nk + 1], :, :],
            cupy.zeros_like(a[name][[1, nk + 1], :, :]),
            f"{name} rigid bottom/top faces must be positive zero",
        )
    return Psolver6PeriodicDeviceProof(
        ni, nj, nk, int(cupy.cuda.Device().id),
        tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES),
        digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CudaPsolver6PeriodicLaunch:
    ready_event: Any
    compiled: CompiledPsolver6PeriodicKernels
    device_proof: Psolver6PeriodicDeviceProof
    receipt: dict[str, Any]


def _check_proof(cupy: Any, a: Mapping[str, Any], proof: Psolver6PeriodicDeviceProof, ni: int, nj: int, nk: int) -> None:
    if (proof.ni, proof.nj, proof.nk, proof.device_id) != (ni, nj, nk, int(cupy.cuda.Device().id)):
        raise ValueError("psolver6 device proof dimensions/device differ")
    if proof.pointers != tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES):
        raise ValueError("psolver6 device proof does not match these allocations")


def launch_cuda_psolver6_periodic_stage(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, dt: float,
    nsound: int, nrk: int, rdx: float, rdy: float, rdz: float, csound: float,
    device_proof: Psolver6PeriodicDeviceProof,
    kdiv: float = CM1_PSOLVER6_KDIV_F32,
    selection: Psolver6PeriodicSelection = Psolver6PeriodicSelection(),
    stream: Any | None = None,
    compiled: CompiledPsolver6PeriodicKernels | None = None,
) -> CudaPsolver6PeriodicLaunch:
    require_psolver6_periodic_selection(selection)
    if not math.isfinite(csound) or csound <= 0.0:
        raise ValueError("csound must be finite and positive")
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    _check_proof(cupy, a, device_proof, ni, nj, nk)
    compiled = compile_cuda_psolver6_periodic() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    prep, uv, column, avg, bcp, stage_bc, _ = compiled.kernels
    f32, u32 = cupy.float32, cupy.uint32
    nloop, dts = sound3_stage_schedule_f32(dt, nsound, nrk)
    pressure_scale = f32(f32(f32(dts) * f32(csound)) * f32(csound))
    maximum = max(
        (nk + 2) * (nj + 2) * (ni + 2),
        (nk + 2) * (nj + 2) * (ni + 1),
        (nk + 2) * (nj + 1) * (ni + 2),
        (nk + 2) * nj * ni,
    )
    grid = ((maximum + CUDA_PSOLVER6_PERIODIC_THREADS - 1) // CUDA_PSOLVER6_PERIODIC_THREADS, 1, 1)
    cols = ((ni * nj + CUDA_PSOLVER6_PERIODIC_THREADS - 1) // CUDA_PSOLVER6_PERIODIC_THREADS, 1, 1)
    block = (CUDA_PSOLVER6_PERIODIC_THREADS, 1, 1)
    prep(grid, block, (a["phi1"], a["ppx"], a["ua"], a["va"], a["wa"], a["phi2"], a["ppd"], a["pp_filter"], a["div"], a["u3d"], a["v3d"], a["w3d"], a["rru"], a["rrv"], a["rrw"], u32(ni), u32(nj), u32(nk), u32(nrk)), stream=stream)
    for n in range(1, nloop + 1):
        uv(grid, block, (a["ppd"], a["uten"], a["vten"], a["uf"], a["vf"], a["u3d"], a["v3d"], u32(ni), u32(nj), u32(nk), f32(dts), f32(rdx), f32(rdy)), stream=stream)
        column(cols, block, (a["mh"], a["mf"], a["uh"], a["vh"], a["wten"], a["u3d"], a["v3d"], a["w3d"], a["phi2"], a["ppx"], a["ppd"], a["pp_filter"], a["div"], u32(ni), u32(nj), u32(nk), u32(nrk), u32(n), u32(nloop), f32(dts), f32(rdx), f32(rdy), f32(rdz), pressure_scale, f32(kdiv), f32(CM1_PSOLVER6_SMEPS_F32)), stream=stream)
        if n < nloop:
            bcp((1, 1, 1), (1, 1, 1), (a["ppd"], u32(ni), u32(nj), u32(nk)), stream=stream)
        avg(grid, block, (a["u3d"], a["v3d"], a["w3d"], a["rru"], a["rrv"], a["rrw"], u32(ni), u32(nj), u32(nk), u32(n), u32(nloop)), stream=stream)
    stage_bc((1, 1, 1), (1, 1, 1), (a["phi1"], a["phi2"], a["u3d"], a["v3d"], a["w3d"], u32(ni), u32(nj), u32(nk), u32(nrk)), stream=stream)
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    cadence = {
        "nrk": nrk, "nloop": nloop, "dts_f32": float(f32(dts)),
        "dttmp_f32": rk_stage_dt_f32(dt, nrk),
        "pressure_boundary_fills": nloop - 1,
        "stage_phi_boundary_fills": 1,
        "stage_velocity_boundary_fills": 1,
        "ppx_filter_written": nrk == 3,
        "phi1_written": nrk == 3,
    }
    receipt = {
        "schema": CUDA_PSOLVER6_PERIODIC_SCHEMA,
        "abi": CUDA_PSOLVER6_PERIODIC_ABI,
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "dimensions": {"ni": ni, "nj": nj, "nk": nk},
        "cadence": cadence,
        "device_contract_proof_sha256": device_proof.contract_sha256,
        "kernel": {"source_sha256": CUDA_PSOLVER6_PERIODIC_SOURCE_SHA256, "names": list(CUDA_PSOLVER6_PERIODIC_KERNEL_NAMES), "target_arch": compiled.target_arch},
        "runtime": dict(compiled.runtime_identity),
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_PSOLVER6_CLAIM_LIMIT,
    }
    return CudaPsolver6PeriodicLaunch(event, compiled, device_proof, receipt)


def launch_cuda_psolver6_periodic_three_stage(
    arrays: Mapping[str, Any], *, device_proof: Psolver6PeriodicDeviceProof | None = None,
    selection: Psolver6PeriodicSelection = Psolver6PeriodicSelection(),
    stream: Any | None = None, compiled: CompiledPsolver6PeriodicKernels | None = None,
    **kwargs: Any,
) -> CudaPsolver6PeriodicLaunch:
    ni, nj, nk = (int(kwargs[name]) for name in ("ni", "nj", "nk"))
    if device_proof is None:
        device_proof = prove_cuda_psolver6_periodic_contract(arrays, ni=ni, nj=nj, nk=nk, selection=selection)
    last = None
    cadence = []
    for nrk in (1, 2, 3):
        last = launch_cuda_psolver6_periodic_stage(arrays, nrk=nrk, device_proof=device_proof, selection=selection, stream=stream, compiled=compiled, **kwargs)
        compiled = last.compiled
        cadence.append(last.receipt["cadence"])
    assert last is not None
    receipt = dict(last.receipt)
    receipt["cadence"] = cadence
    receipt["transition"] = "held-input three-stage wrapper; use the stage API for live RK interleaving"
    return CudaPsolver6PeriodicLaunch(last.ready_event, last.compiled, device_proof, receipt)


def launch_cuda_psolver6_periodic_solve3_complete(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int,
    device_proof: Psolver6PeriodicDeviceProof, stream: Any | None = None,
    compiled: CompiledPsolver6PeriodicKernels | None = None,
) -> CudaPsolver6PeriodicLaunch:
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    _check_proof(cupy, a, device_proof, ni, nj, nk)
    compiled = compile_cuda_psolver6_periodic() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    *_, bcp, _, complete = compiled.kernels
    u32 = cupy.uint32
    n = ni * nj * nk
    grid = ((n + CUDA_PSOLVER6_PERIODIC_THREADS - 1) // CUDA_PSOLVER6_PERIODIC_THREADS, 1, 1)
    complete(grid, (CUDA_PSOLVER6_PERIODIC_THREADS, 1, 1), (a["phi1"], a["ppx"], u32(ni), u32(nj), u32(nk)), stream=stream)
    bcp((1, 1, 1), (1, 1, 1), (a["ppx"], u32(ni), u32(nj), u32(nk)), stream=stream)
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_PSOLVER6_PERIODIC_SCHEMA,
        "transition": "source-ordered solve3.F:285-301 ppx=phi1+filter and periodic fill",
        "dimensions": {"ni": ni, "nj": nj, "nk": nk},
        "kernel_source_sha256": CUDA_PSOLVER6_PERIODIC_SOURCE_SHA256,
        "claim_limit": CM1_PSOLVER6_CLAIM_LIMIT,
    }
    return CudaPsolver6PeriodicLaunch(event, compiled, device_proof, receipt)


__all__ = (
    "ALL_ARRAY_NAMES", "CM1_PSOLVER6_CLAIM_LIMIT", "COMPARE_OUTPUT_NAMES",
    "CUDA_PSOLVER6_PERIODIC_ABI", "CUDA_PSOLVER6_PERIODIC_SCHEMA",
    "CUDA_PSOLVER6_PERIODIC_SOURCE_BYTES", "CUDA_PSOLVER6_PERIODIC_SOURCE_SHA256",
    "CpuPsolver6Result", "CpuPsolver6StageDiagnostics", "CudaPsolver6PeriodicLaunch",
    "Psolver6PeriodicDeviceProof", "Psolver6PeriodicSelection", "SOURCE_ANCHORS",
    "compare_psolver6_periodic_raw", "compile_cuda_psolver6_periodic",
    "cpu_psolver6_periodic_solve3_complete_inplace", "cpu_psolver6_periodic_stage_inplace",
    "cpu_psolver6_periodic_three_stage_inplace", "default_periodic_face_weights",
    "fill_periodic_scalar_inplace", "fill_periodic_uv_inplace",
    "launch_cuda_psolver6_periodic_solve3_complete", "launch_cuda_psolver6_periodic_stage",
    "launch_cuda_psolver6_periodic_three_stage", "prove_cuda_psolver6_periodic_contract",
    "psolver6_periodic_shapes", "require_psolver6_periodic_selection",
    "validate_psolver6_periodic_contract",
)
