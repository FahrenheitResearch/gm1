"""Bounded CM1 r21.1 psolver=4/5 flat-periodic projection framing.

CM1's deep-anelastic and incompressible selectors share ``anelp.F`` and
``poiss.F``.  This module implements the exact local Poisson RHS assembly and
the post-solve velocity projection/closure around an externally supplied
``phi2`` pressure solution.  The singleton FFT plus vertical spectral solve
between those operations is deliberately not implemented here.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

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
from .cuda_stretch_z1 import (
    StretchZ1MetricProof,
    validate_stretch_z1_metric_profiles,
)
from .errors import refuse


CUDA_ANELP_PERIODIC_SCHEMA = "gpuwm-cm1.cuda-anelp-periodic-frame/v1"
CUDA_ANELP_PERIODIC_ABI = "gpuwm-cm1.cuda-anelp-periodic-frame-kernels/v1"
CUDA_ANELP_PERIODIC_THREADS = 128
CM1_ANELP_PERIODIC_CLAIM_LIMIT = (
    "CM1 r21.1 psolver=4/5 shared anelp.F/poiss.F local Poisson-RHS and "
    "post-solve velocity-projection framing for a flat Cartesian, serial, "
    "uniform or proof-bound Wilhelmson-Chen stretch_z=1, four-periodic "
    "eqtset=1/apmasscon=0 profile, with supplied "
    "RK-stage dttmp, held tendencies/divx, horizontally homogeneous base "
    "profiles, and an externally supplied periodic phi2 pressure solution; "
    "no singleton FFT, spectral-mode coefficient field, vertical tridiagonal "
    "solve, zero-mode reconstruction, complete elliptic pressure solution, "
    "open/rigid lateral boundary, MPI, terrain, axisymmetry, immersed boundary, "
    "forcing, complete solve2, trajectory, forecast, product, or performance claim"
)
SOURCE_ANCHORS = {
    "dispatch": "src/solve2.F:1138-1152",
    "parameter_constraints": "src/param.F:721,921-965,1748-1824,2420-2439",
    "elliptic_coefficients": "src/init3d.F:1668-1720",
    "poisson_rhs": "src/poiss.F:53-77",
    "unimplemented_fft_tridiagonal": "src/poiss.F:79-209",
    "poisson_call_and_phi_boundary": "src/anelp.F:237-244",
    "velocity_projection": "src/anelp.F:247-263",
    "final_phi_commit": "src/anelp.F:266-280",
    "stage_velocity_boundaries": "src/solve2.F:1662-1689",
}


@dataclass(frozen=True, slots=True)
class AnelpPeriodicSelection:
    psolver: int = 4
    terrain_flag: bool = False
    axisymm: int = 0
    mpi: bool = False
    stretch_x: int = 0
    stretch_y: int = 0
    stretch_z: int = 0
    wbc: int = 1
    ebc: int = 1
    sbc: int = 1
    nbc: int = 1
    irbc: int = 4
    roflux: int = 1
    nudgeobc: int = 0
    eqtset: int = 1
    apmasscon: int = 0
    do_ib: bool = False
    convinit: int = 0
    wnudge: int = 0


def require_anelp_periodic_selection(selection: AnelpPeriodicSelection) -> None:
    if selection.psolver not in (4, 5):
        refuse(
            code="ARWEN-ANELP-PSOLVER",
            gate="anelp-flat-periodic-frame-selection",
            reason="this shared official operator frame admits only psolver=4 or 5",
            remedy="select psolver=4/5 or use a separately ratified pressure operator",
            option="psolver",
            value=selection.psolver,
        )
    if selection.stretch_z not in (0, 1):
        refuse(
            code="ARWEN-ANELP-STRETCH-Z",
            gate="anelp-flat-periodic-frame-selection",
            reason=(
                "the bounded psolver=4/5 frame admits only uniform stretch_z=0 "
                "or proof-bound Wilhelmson-Chen stretch_z=1"
            ),
            remedy="select stretch_z=0/1 or use a separately ratified vertical grid",
            option="stretch_z",
            value=selection.stretch_z,
        )
    checks = (
        ("terrain_flag", selection.terrain_flag, False, "ARWEN-ANELP-TERRAIN"),
        ("axisymm", selection.axisymm, 0, "ARWEN-ANELP-AXISYMM"),
        ("mpi", selection.mpi, False, "ARWEN-ANELP-MPI"),
        ("stretch_x", selection.stretch_x, 0, "ARWEN-ANELP-STRETCH-X"),
        ("stretch_y", selection.stretch_y, 0, "ARWEN-ANELP-STRETCH-Y"),
        ("wbc", selection.wbc, 1, "ARWEN-ANELP-WBC"),
        ("ebc", selection.ebc, 1, "ARWEN-ANELP-EBC"),
        ("sbc", selection.sbc, 1, "ARWEN-ANELP-SBC"),
        ("nbc", selection.nbc, 1, "ARWEN-ANELP-NBC"),
        ("irbc", selection.irbc, 4, "ARWEN-ANELP-IRBC"),
        ("roflux", selection.roflux, 1, "ARWEN-ANELP-ROFLUX"),
        ("nudgeobc", selection.nudgeobc, 0, "ARWEN-ANELP-NUDGE-OBC"),
        ("eqtset", selection.eqtset, 1, "ARWEN-ANELP-EQTSET"),
        ("apmasscon", selection.apmasscon, 0, "ARWEN-ANELP-APMASSCON"),
        ("do_ib", selection.do_ib, False, "ARWEN-ANELP-IMMERSED"),
        ("convinit", selection.convinit, 0, "ARWEN-ANELP-CONVINIT"),
        ("wnudge", selection.wnudge, 0, "ARWEN-ANELP-WNUDGE"),
    )
    for option, value, expected, code in checks:
        if value != expected:
            refuse(
                code=code,
                gate="anelp-flat-periodic-frame-selection",
                reason=(
                    f"the bounded psolver=4/5 frame requires {option}={expected!r}; "
                    "the official alternate branch is not implemented here"
                ),
                remedy="select the exact flat serial-four-periodic family",
                option=option,
                value=value,
            )


SCALAR_NAMES = (
    "mh",
    "mf",
    "rho0",
    "rf0",
    "divx",
    "def",
    "rhs_real",
    "phi1",
    "phi2",
)
U_NAMES = ("ua", "uten", "u3d")
V_NAMES = ("va", "vten", "v3d")
W_NAMES = ("wa", "wten", "w3d")
METRIC_NAMES = ("uf", "vf", "uh", "vh")
VERTICAL_NAMES = ("d1",)
ALL_ARRAY_NAMES = (
    SCALAR_NAMES + U_NAMES + V_NAMES + W_NAMES + METRIC_NAMES + VERTICAL_NAMES
)
WRITABLE_NAMES = ("def", "rhs_real", "phi1", "u3d", "v3d", "w3d")
COMPARE_OUTPUT_NAMES = WRITABLE_NAMES


def _validate_dimensions(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 3:
            raise ValueError(f"{name} must be an integer >= 3")


def anelp_periodic_shapes(
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
    result["d1"] = (nk + 2,)
    return result


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = anelp_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing anelp arrays: {', '.join(missing)}")
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
        raise ValueError("anelp arrays must not alias")
    return result


def _bits_equal(np: Any, left: Any, right: Any) -> bool:
    return bool(np.array_equal(left.view(np.uint32), right.view(np.uint32)))


def _validate_vertical_metric_admission(
    selection: AnelpPeriodicSelection,
    vertical_grid_proof: StretchZ1MetricProof | None,
    *,
    mh: Any,
    mf: Any,
    nk: int,
) -> None:
    if selection.stretch_z == 0:
        if vertical_grid_proof is not None:
            raise ValueError("stretch_z=0 must not carry a stretch_z=1 metric proof")
        return
    if vertical_grid_proof is None:
        refuse(
            code="ARWEN-ANELP-STRETCH-Z1-PROOF",
            gate="anelp-flat-periodic-vertical-metric-admission",
            reason=(
                "stretch_z=1 pressure gradients and coefficients would consume "
                "unverified mh/mf profiles"
            ),
            remedy="supply a proof built from the official stretch_z=1 grid builder",
            option="stretch_z",
            value=selection.stretch_z,
        )
    validate_stretch_z1_metric_profiles(
        vertical_grid_proof,
        mh=mh,
        mf=mf,
        nk=nk,
    )


def validate_anelp_periodic_contract(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    vertical_grid_proof: StretchZ1MetricProof | None = None,
) -> None:
    require_anelp_periodic_selection(selection)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    for name in METRIC_NAMES:
        if not _bits_equal(np, a[name], np.ones_like(a[name])):
            raise ValueError(f"{name} must be exactly one for the unstretched flat slice")
    if not _bits_equal(np, a["d1"], np.ones_like(a["d1"])):
        raise ValueError("d1 must be exactly one for official psolver=4/5 coefficients")
    for name in ("mh", "mf", "rho0", "rf0"):
        live = a[name][:, 1 : nj + 1, 1 : ni + 1]
        profile = a[name][:, 1:2, 1:2]
        if not _bits_equal(np, live, np.broadcast_to(profile, live.shape)):
            raise ValueError(f"{name} must be horizontally homogeneous")
    _validate_vertical_metric_admission(
        selection,
        vertical_grid_proof,
        mh=np.ascontiguousarray(a["mh"][:, 1, 1]),
        mf=np.ascontiguousarray(a["mf"][:, 1, 1]),
        nk=nk,
    )
    value = a["phi2"]
    for left, right, side in (
        (value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni], "west"),
        (value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1], "east"),
        (value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :], "south"),
        (value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :], "north"),
    ):
        if not _bits_equal(np, left, right):
            raise ValueError(f"phi2 {side} periodic halo is invalid")
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
    if np.any(a["wa"][[1, nk + 1], :, :].view(np.uint32) != np.uint32(0)):
        raise ValueError("wa rigid bottom/top faces must be positive zero")


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


@dataclass(frozen=True, slots=True)
class CpuAnelpPeriodicDiagnostics:
    nrk: int
    dttmp_f32: float
    rhs_cells: int
    projected_u_faces: int
    projected_v_faces: int
    projected_w_faces: int
    phi1_written: bool
    spectral_solve_status: str


def cpu_anelp_periodic_frame_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dttmp: float,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    vertical_grid_proof: StretchZ1MetricProof | None = None,
) -> CpuAnelpPeriodicDiagnostics:
    validate_anelp_periodic_contract(
        arrays,
        ni=ni,
        nj=nj,
        nk=nk,
        selection=selection,
        vertical_grid_proof=vertical_grid_proof,
    )
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    if not math.isfinite(dttmp) or dttmp <= 0.0:
        raise ValueError("dttmp must be finite and positive")
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    dttmp, rdx, rdy, rdz = map(np.float32, (dttmp, rdx, rdy, rdz))
    rdt = _div(np, np.float32(1.0), dttmp)
    a["def"].fill(np.float32(0.0))
    a["rhs_real"].fill(np.float32(0.0))
    for k in range(1, nk + 1):
        for j0 in range(nj):
            j = j0 + 1
            for i0 in range(ni):
                i = i0 + 1
                du = _mul(
                    np,
                    _mul(
                        np,
                        _sub(np, a["uten"][k, j, i], a["uten"][k, j, i - 1]),
                        rdx,
                    ),
                    a["uh"][i],
                )
                dv = _mul(
                    np,
                    _mul(
                        np,
                        _sub(np, a["vten"][k, j, i], a["vten"][k, j - 1, i]),
                        rdy,
                    ),
                    a["vh"][j],
                )
                horizontal = _mul(np, a["rho0"][k, 1, 1], _add(np, du, dv))
                vertical = _sub(
                    np,
                    _mul(np, a["rf0"][k + 1, j, i], a["wten"][k + 1, j0, i0]),
                    _mul(np, a["rf0"][k, j, i], a["wten"][k, j0, i0]),
                )
                vertical = _mul(np, _mul(np, vertical, rdz), a["mh"][k, j, i])
                value = _add(
                    np,
                    _add(np, horizontal, vertical),
                    _mul(np, a["divx"][k, j, i], rdt),
                )
                a["def"][k, j, i] = value
                a["rhs_real"][k, j, i] = _mul(np, value, a["d1"][k])
    temx, temy, temz = (
        _mul(np, dttmp, rdx),
        _mul(np, dttmp, rdy),
        _mul(np, dttmp, rdz),
    )
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            for f in range(ni + 1):
                provisional = _add(
                    np, a["ua"][k, j, f], _mul(np, dttmp, a["uten"][k, j, f])
                )
                pressure = _mul(
                    np,
                    _mul(
                        np,
                        temx,
                        _sub(np, a["phi2"][k, j, f + 1], a["phi2"][k, j, f]),
                    ),
                    a["uf"][f + 1],
                )
                a["u3d"][k, j, f] = _sub(np, provisional, pressure)
        for f in range(nj + 1):
            for i in range(1, ni + 1):
                provisional = _add(
                    np, a["va"][k, f, i], _mul(np, dttmp, a["vten"][k, f, i])
                )
                pressure = _mul(
                    np,
                    _mul(
                        np,
                        temy,
                        _sub(np, a["phi2"][k, f + 1, i], a["phi2"][k, f, i]),
                    ),
                    a["vf"][f + 1],
                )
                a["v3d"][k, f, i] = _sub(np, provisional, pressure)
    for k in range(2, nk + 1):
        for j0 in range(nj):
            for i0 in range(ni):
                provisional = _add(
                    np,
                    a["wa"][k, j0, i0],
                    _mul(np, dttmp, a["wten"][k, j0, i0]),
                )
                pressure = _mul(
                    np,
                    _mul(
                        np,
                        temz,
                        _sub(
                            np,
                            a["phi2"][k, j0 + 1, i0 + 1],
                            a["phi2"][k - 1, j0 + 1, i0 + 1],
                        ),
                    ),
                    a["mf"][k, j0 + 1, i0 + 1],
                )
                a["w3d"][k, j0, i0] = _sub(np, provisional, pressure)
    fill_periodic_uv_inplace(a["u3d"], a["v3d"], ni=ni, nj=nj, nk=nk)
    a["w3d"][[1, nk + 1], :, :] = np.float32(0.0)
    if nrk == 3:
        a["phi1"][1 : nk + 1] = a["phi2"][1 : nk + 1]
    return CpuAnelpPeriodicDiagnostics(
        nrk=nrk,
        dttmp_f32=float(dttmp),
        rhs_cells=ni * nj * nk,
        projected_u_faces=(ni + 1) * nj * nk,
        projected_v_faces=ni * (nj + 1) * nk,
        projected_w_faces=ni * nj * (nk - 1),
        phi1_written=nrk == 3,
        spectral_solve_status="external-phi2-required-not-implemented",
    )


def compare_anelp_periodic_raw(
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


CUDA_ANELP_PERIODIC_SOURCE = r"""// gpuwm-cm1.cuda-anelp-periodic-frame-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);} 
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);} 
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);} 
__device__ __forceinline__ unsigned long long si(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long ui(unsigned k,unsigned j,unsigned f,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+2u)+j)*(ni+1u)+f;}
__device__ __forceinline__ unsigned long long vi(unsigned k,unsigned f,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*(nj+1u)+f)*(ni+2u)+i;}
__device__ __forceinline__ unsigned long long wi(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)k*nj+j)*ni+i;}

extern "C" __global__ void cm1_anelp_clear_v1(float* def,float* rhs,unsigned long long words){unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,s=(unsigned long long)blockDim.x*gridDim.x;for(;x<words;x+=s){def[x]=0.0f;rhs[x]=0.0f;}}

extern "C" __global__ void cm1_anelp_rhs_v1(const float* mh,const float* rho0,const float* rf0,const float* divx,const float* d1,const float* uh,const float* vh,const float* uten,const float* vten,const float* wten,float* def,float* rhs,unsigned ni,unsigned nj,unsigned nk,float rdx,float rdy,float rdz,float rdt){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;for(unsigned long long q=first;q<n;q+=step){unsigned i=q%ni+1u;unsigned long long z=q/ni;unsigned j=z%nj+1u,k=z/nj+1u;float du=m32(m32(s32(uten[ui(k,j,i,ni,nj)],uten[ui(k,j,i-1u,ni,nj)]),rdx),uh[i]);float dv=m32(m32(s32(vten[vi(k,j,i,ni,nj)],vten[vi(k,j-1u,i,ni,nj)]),rdy),vh[j]);float horizontal=m32(rho0[si(k,1u,1u,ni,nj)],a32(du,dv));unsigned j0=j-1u,i0=i-1u;float vertical=s32(m32(rf0[si(k+1u,j,i,ni,nj)],wten[wi(k+1u,j0,i0,ni,nj)]),m32(rf0[si(k,j,i,ni,nj)],wten[wi(k,j0,i0,ni,nj)]));vertical=m32(m32(vertical,rdz),mh[si(k,j,i,ni,nj)]);float value=a32(a32(horizontal,vertical),m32(divx[si(k,j,i,ni,nj)],rdt));def[si(k,j,i,ni,nj)]=value;rhs[si(k,j,i,ni,nj)]=m32(value,d1[k]);}}

extern "C" __global__ void cm1_anelp_project_v1(const float* phi2,const float* mf,const float* uf,const float* vf,const float* ua,const float* uten,const float* va,const float* vten,const float* wa,const float* wten,float* u,float* v,float* w,unsigned ni,unsigned nj,unsigned nk,float dttmp,float temx,float temy,float temz){unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nu=(unsigned long long)nk*nj*(ni+1u),nv=(unsigned long long)nk*(nj+1u)*ni,nw=(unsigned long long)(nk-1u)*nj*ni;for(unsigned long long q=first;q<nu;q+=step){unsigned f=q%(ni+1u);unsigned long long z=q/(ni+1u);unsigned j=z%nj+1u,k=z/nj+1u;unsigned long long x=ui(k,j,f,ni,nj);float provisional=a32(ua[x],m32(dttmp,uten[x]));float pressure=m32(m32(temx,s32(phi2[si(k,j,f+1u,ni,nj)],phi2[si(k,j,f,ni,nj)])),uf[f+1u]);u[x]=s32(provisional,pressure);}for(unsigned long long q=first;q<nv;q+=step){unsigned i=q%ni+1u;unsigned long long z=q/ni;unsigned f=z%(nj+1u),k=z/(nj+1u)+1u;unsigned long long x=vi(k,f,i,ni,nj);float provisional=a32(va[x],m32(dttmp,vten[x]));float pressure=m32(m32(temy,s32(phi2[si(k,f+1u,i,ni,nj)],phi2[si(k,f,i,ni,nj)])),vf[f+1u]);v[x]=s32(provisional,pressure);}for(unsigned long long q=first;q<nw;q+=step){unsigned i=q%ni,j=(q/ni)%nj,k=(unsigned)(q/(ni*nj))+2u;unsigned long long x=wi(k,j,i,ni,nj);float provisional=a32(wa[x],m32(dttmp,wten[x]));float pressure=m32(m32(temz,s32(phi2[si(k,j+1u,i+1u,ni,nj)],phi2[si(k-1u,j+1u,i+1u,ni,nj)])),mf[si(k,j+1u,i+1u,ni,nj)]);w[x]=s32(provisional,pressure);}}

extern "C" __global__ void cm1_anelp_bc_v1(const float* phi2,float* phi1,float* u,float* v,float* w,unsigned ni,unsigned nj,unsigned nk,unsigned nrk){if(blockIdx.x||threadIdx.x)return;for(unsigned k=1u;k<=nk;++k){for(unsigned f=0u;f<=ni;++f){u[ui(k,0u,f,ni,nj)]=u[ui(k,nj,f,ni,nj)];u[ui(k,nj+1u,f,ni,nj)]=u[ui(k,1u,f,ni,nj)];}for(unsigned f=0u;f<=nj;++f){v[vi(k,f,0u,ni,nj)]=v[vi(k,f,ni,ni,nj)];v[vi(k,f,ni+1u,ni,nj)]=v[vi(k,f,1u,ni,nj)];}if(nrk==3u){for(unsigned j=0u;j<=nj+1u;++j)for(unsigned i=0u;i<=ni+1u;++i)phi1[si(k,j,i,ni,nj)]=phi2[si(k,j,i,ni,nj)];}}for(unsigned j=0u;j<nj;++j)for(unsigned i=0u;i<ni;++i){w[wi(1u,j,i,ni,nj)]=0.0f;w[wi(nk+1u,j,i,ni,nj)]=0.0f;}}
"""
CUDA_ANELP_PERIODIC_SOURCE_BYTES = CUDA_ANELP_PERIODIC_SOURCE.encode("utf-8")
CUDA_ANELP_PERIODIC_SOURCE_SHA256 = (
    "e0e989a70dd70dcfa01aa19da7cc4dfa431511a66b1ae68671e3c47a6623b698"
)
CUDA_ANELP_PERIODIC_KERNEL_NAMES = (
    "cm1_anelp_clear_v1",
    "cm1_anelp_rhs_v1",
    "cm1_anelp_project_v1",
    "cm1_anelp_bc_v1",
)


@dataclass(frozen=True, slots=True)
class CompiledAnelpPeriodicKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_anelp_periodic() -> CompiledAnelpPeriodicKernels:
    if hashlib.sha256(CUDA_ANELP_PERIODIC_SOURCE_BYTES).hexdigest() != CUDA_ANELP_PERIODIC_SOURCE_SHA256:
        raise RuntimeError("packaged anelp CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(CUDA_ANELP_PERIODIC_SOURCE, "cm1_anelp_ptx.cu", method="ptx").compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(CUDA_ANELP_PERIODIC_SOURCE, "cm1_anelp_cubin.cu", method="cubin").compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", text) for name in CUDA_ANELP_PERIODIC_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks anelp entry points")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_ANELP_PERIODIC_KERNEL_NAMES)
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
    return CompiledAnelpPeriodicKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = anelp_periodic_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing anelp device arrays: {', '.join(missing)}")
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32") or value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid CUDA dtype, shape, or layout")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("anelp device arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class AnelpPeriodicDeviceProof:
    ni: int
    nj: int
    nk: int
    device_id: int
    selection: AnelpPeriodicSelection
    vertical_grid_proof_sha256: str | None
    pointers: tuple[tuple[str, int], ...]
    contract_sha256: str


def prove_cuda_anelp_periodic_contract(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    vertical_grid_proof: StretchZ1MetricProof | None = None,
) -> AnelpPeriodicDeviceProof:
    require_anelp_periodic_selection(selection)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    digest = hashlib.sha256()

    def exact(label: str, left: Any, right: Any, reason: str) -> None:
        if not bool(cupy.array_equal(left.view(cupy.uint32), right.view(cupy.uint32))):
            raise ValueError(reason)
        digest.update(label.encode("ascii") + b"\0")
        digest.update(cupy.asnumpy(left).astype("<f4", copy=False).tobytes(order="C"))

    for name in METRIC_NAMES:
        exact(name, a[name], cupy.ones_like(a[name]), f"{name} must be exactly one")
    exact("d1", a["d1"], cupy.ones_like(a["d1"]), "d1 must be exactly one")
    for name in ("mh", "mf", "rho0", "rf0"):
        live, profile = a[name][:, 1 : nj + 1, 1 : ni + 1], a[name][:, 1:2, 1:2]
        if not bool(cupy.all(live.view(cupy.uint32) == profile.view(cupy.uint32)).item()):
            raise ValueError(f"{name} must be horizontally homogeneous")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(cupy.asnumpy(profile).astype("<f4", copy=False).tobytes(order="C"))
    _validate_vertical_metric_admission(
        selection,
        vertical_grid_proof,
        mh=cupy.asnumpy(a["mh"][:, 1, 1]),
        mf=cupy.asnumpy(a["mf"][:, 1, 1]),
        nk=nk,
    )
    if vertical_grid_proof is not None:
        digest.update(b"vertical-grid-proof\0")
        digest.update(vertical_grid_proof.proof_sha256.encode("ascii"))
    value = a["phi2"]
    for side, left, right in (
        ("west", value[1 : nk + 1, :, 0], value[1 : nk + 1, :, ni]),
        ("east", value[1 : nk + 1, :, ni + 1], value[1 : nk + 1, :, 1]),
        ("south", value[1 : nk + 1, 0, :], value[1 : nk + 1, nj, :]),
        ("north", value[1 : nk + 1, nj + 1, :], value[1 : nk + 1, 1, :]),
    ):
        exact(f"phi2-{side}", left, right, f"phi2 {side} periodic halo is invalid")
    for name in ("ua", "uten"):
        exact(f"{name}-x", a[name][1 : nk + 1, 1 : nj + 1, 0], a[name][1 : nk + 1, 1 : nj + 1, ni], f"{name} duplicate x faces differ")
    for name in ("va", "vten"):
        exact(f"{name}-y", a[name][1 : nk + 1, 0, 1 : ni + 1], a[name][1 : nk + 1, nj, 1 : ni + 1], f"{name} duplicate y faces differ")
    exact("wa-rigid", a["wa"][[1, nk + 1]], cupy.zeros_like(a["wa"][[1, nk + 1]]), "wa rigid faces must be positive zero")
    return AnelpPeriodicDeviceProof(
        ni,
        nj,
        nk,
        int(cupy.cuda.Device().id),
        selection,
        None if vertical_grid_proof is None else vertical_grid_proof.proof_sha256,
        tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES),
        digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class CudaAnelpPeriodicLaunch:
    ready_event: Any
    compiled: CompiledAnelpPeriodicKernels
    device_proof: AnelpPeriodicDeviceProof
    receipt: dict[str, Any]


def launch_cuda_anelp_periodic_frame(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dttmp: float,
    nrk: int,
    rdx: float,
    rdy: float,
    rdz: float,
    device_proof: AnelpPeriodicDeviceProof,
    selection: AnelpPeriodicSelection = AnelpPeriodicSelection(),
    stream: Any | None = None,
    compiled: CompiledAnelpPeriodicKernels | None = None,
) -> CudaAnelpPeriodicLaunch:
    require_anelp_periodic_selection(selection)
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    if not math.isfinite(dttmp) or dttmp <= 0.0:
        raise ValueError("dttmp must be finite and positive")
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    expected = tuple((name, int(a[name].data.ptr)) for name in ALL_ARRAY_NAMES)
    if (
        (device_proof.ni, device_proof.nj, device_proof.nk, device_proof.device_id)
        != (ni, nj, nk, int(cupy.cuda.Device().id))
        or device_proof.selection != selection
        or device_proof.pointers != expected
    ):
        raise ValueError("anelp device proof does not match these allocations")
    compiled = compile_cuda_anelp_periodic() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    clear, rhs, project, bc = compiled.kernels
    f32, u32, u64 = cupy.float32, cupy.uint32, cupy.uint64
    scalar_words = (nk + 2) * (nj + 2) * (ni + 2)
    maximum = max(
        scalar_words,
        ni * nj * nk,
        nk * nj * (ni + 1),
        nk * (nj + 1) * ni,
        (nk - 1) * nj * ni,
    )
    block = (CUDA_ANELP_PERIODIC_THREADS, 1, 1)
    grid = ((maximum + CUDA_ANELP_PERIODIC_THREADS - 1) // CUDA_ANELP_PERIODIC_THREADS, 1, 1)
    clear(grid, block, (a["def"], a["rhs_real"], u64(scalar_words)), stream=stream)
    dttmp32, rdx32, rdy32, rdz32 = map(np.float32, (dttmp, rdx, rdy, rdz))
    rdt32 = np.float32(np.float32(1.0) / dttmp32)
    rhs(grid, block, (a["mh"], a["rho0"], a["rf0"], a["divx"], a["d1"], a["uh"], a["vh"], a["uten"], a["vten"], a["wten"], a["def"], a["rhs_real"], u32(ni), u32(nj), u32(nk), f32(rdx32), f32(rdy32), f32(rdz32), f32(rdt32)), stream=stream)
    project(grid, block, (a["phi2"], a["mf"], a["uf"], a["vf"], a["ua"], a["uten"], a["va"], a["vten"], a["wa"], a["wten"], a["u3d"], a["v3d"], a["w3d"], u32(ni), u32(nj), u32(nk), f32(dttmp32), f32(np.float32(dttmp32 * rdx32)), f32(np.float32(dttmp32 * rdy32)), f32(np.float32(dttmp32 * rdz32))), stream=stream)
    bc((1, 1, 1), (1, 1, 1), (a["phi2"], a["phi1"], a["u3d"], a["v3d"], a["w3d"], u32(ni), u32(nj), u32(nk), u32(nrk)), stream=stream)
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_ANELP_PERIODIC_SCHEMA,
        "abi": CUDA_ANELP_PERIODIC_ABI,
        "selection": {name: getattr(selection, name) for name in selection.__dataclass_fields__},
        "dimensions": {"ni": ni, "nj": nj, "nk": nk},
        "cadence": {
            "nrk": nrk,
            "dttmp_f32": float(dttmp32),
            "phi1_written": nrk == 3,
            "spectral_solve_status": "external-phi2-required-not-implemented",
        },
        "device_contract_proof_sha256": device_proof.contract_sha256,
        "vertical_grid_proof_sha256": device_proof.vertical_grid_proof_sha256,
        "kernel": {"source_sha256": CUDA_ANELP_PERIODIC_SOURCE_SHA256, "names": list(CUDA_ANELP_PERIODIC_KERNEL_NAMES), "target_arch": compiled.target_arch},
        "runtime": dict(compiled.runtime_identity),
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_ANELP_PERIODIC_CLAIM_LIMIT,
    }
    return CudaAnelpPeriodicLaunch(event, compiled, device_proof, receipt)


__all__ = (
    "ALL_ARRAY_NAMES",
    "AnelpPeriodicDeviceProof",
    "AnelpPeriodicSelection",
    "CM1_ANELP_PERIODIC_CLAIM_LIMIT",
    "COMPARE_OUTPUT_NAMES",
    "CUDA_ANELP_PERIODIC_ABI",
    "CUDA_ANELP_PERIODIC_SCHEMA",
    "CUDA_ANELP_PERIODIC_SOURCE_BYTES",
    "CUDA_ANELP_PERIODIC_SOURCE_SHA256",
    "CompiledAnelpPeriodicKernels",
    "CpuAnelpPeriodicDiagnostics",
    "CudaAnelpPeriodicLaunch",
    "SOURCE_ANCHORS",
    "anelp_periodic_shapes",
    "compare_anelp_periodic_raw",
    "compile_cuda_anelp_periodic",
    "cpu_anelp_periodic_frame_inplace",
    "fill_periodic_scalar_inplace",
    "fill_periodic_uv_inplace",
    "launch_cuda_anelp_periodic_frame",
    "prove_cuda_anelp_periodic_contract",
    "require_anelp_periodic_selection",
    "validate_anelp_periodic_contract",
)
