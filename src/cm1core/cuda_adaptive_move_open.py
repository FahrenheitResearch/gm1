"""CM1 r21.1 flat serial adaptive moving domain with open boundaries.

This is the source-supported ``do_adapt_move``/``imove=1`` branch with all
four horizontal boundaries set to CM1 code 2 (radiative/open).  Its pressure
diagnosis differs from the periodic branch only in ``bc2d``: every smoothing
pass and the final zeta interpolation use edge-replicating open halos.

The adaptive controller does not mutate the persistent ``radbc*`` histories;
they are nevertheless admitted as exact padded carriers so a caller cannot
silently substitute periodic state.  Terrain, MPI, rigid walls, nesting,
external future-boundary interpolation, and a coupled forecast remain out of
scope and fail closed where represented by the selection.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import cuda_adaptive_move as periodic
from .cuda_sound3_periodic import _canonical_arch, _image_options
from .cuda_stretch_z1 import StretchZ1MetricProof
from .errors import refuse


CUDA_ADAPTIVE_MOVE_OPEN_SCHEMA = "gpuwm-cm1.cuda-adaptive-move-open/v1"
CUDA_ADAPTIVE_MOVE_OPEN_ABI = "gpuwm-cm1.cuda-adaptive-move-open-kernels/v1"
CM1_ADAPTIVE_MOVE_OPEN_CLAIM_LIMIT = (
    "CM1 r21.1 terrain=0, nproc=1, four-code-2 radiative/open "
    "do_adapt_move/imove=1 completed-large-step controller: lowest-level "
    "pressure diagnosis with open bc2d halos, type-2 centroid, one cadence "
    "update, moving-frame wind rebasing, typed restart state, and explicit "
    "unchanged radbc history disposition, at any uniform horizontal spacing "
    "whose reference faces and mass points are exact binary32 words and on any "
    "stretch_z=0/1/3/4 column whose lowest scalar level is named in the "
    "receipt. No terrain, MPI, rigid boundary, "
    "nesting, external LBC interpolation, coupled trajectory, or forecast claim"
)

AUTHORITY_TARBALL_SHA256 = periodic.AUTHORITY_TARBALL_SHA256
AUTHORITY_SOURCE_SHA256 = dict(periodic.AUTHORITY_SOURCE_SHA256)
AUTHORITY_CONFIG_SHA256 = {
    "run/namelist.input": "3cf17aa1ac513544d4695994013fd30ea89c2bb0f7d926e41e348635882a6ca5"
}
SOURCE_ANCHORS = {
    **periodic.SOURCE_ANCHORS,
    "official_open_imove_example": "run/namelist.input:66-69,81",
    "open_bc2d_halos": "src/bc.F:32-41,69-78,106-115,143-152",
    "open_radiation_history": "src/bc.F:1948-2172",
    "open_history_restart_write": "src/restart_write.F:1851-1970",
    "open_history_restart_read": "src/restart_read.F:2341-2474",
    "adaptive_argument_inventory": "src/azimavg.F:4376-4379",
    "supported_boundary_matrix": "src/param.F:1890-1967",
}

OPEN_HISTORY_FIELDS = ("radbcw", "radbce", "radbcs", "radbcn")


@dataclass(frozen=True, slots=True)
class AdaptiveMoveOpenSelection:
    do_adapt_move: bool = True
    imove: int = 1
    axisymm: int = 0
    terrain_flag: bool = False
    mpi: bool = False
    nproc: int = 1
    initsfc: int = 1
    wbc: int = 2
    ebc: int = 2
    sbc: int = 2
    nbc: int = 2
    stretch_x: int = 0
    stretch_y: int = 0
    stretch_z: int = 1
    output_format: int = 2
    ngxy: int = 3


DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION = AdaptiveMoveOpenSelection()


def require_adaptive_move_open_selection(
    selection: AdaptiveMoveOpenSelection = DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION,
) -> None:
    if not isinstance(selection, AdaptiveMoveOpenSelection):
        raise TypeError("selection must be an AdaptiveMoveOpenSelection")
    expected = DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION
    for name in selection.__dataclass_fields__:
        value = getattr(selection, name)
        wanted = getattr(expected, name)
        if name == "stretch_z":
            # getpsmth's zref = 0 reads kval = 1 on every admitted column;
            # the branch only names the height the receipt reports.
            if type(value) is int and value in periodic.ADAPTIVE_MOVE_STRETCH_Z:
                continue
            wanted = periodic.ADAPTIVE_MOVE_STRETCH_Z
        elif type(value) is type(wanted) and value == wanted:  # noqa: E721
            continue
        refuse(
            code="ARWEN-ADAPTIVE-MOVE-OPEN-SELECTION",
            gate="adaptive-moving-open-domain",
            option=name,
            value=value,
            reason=(
                "the bounded branch requires flat nproc=1 four-code-2 "
                "radiative/open boundaries on a vertical coordinate whose "
                "lowest scalar level can be named"
            ),
            remedy=f"select {name}={wanted!r} or add a separate source proof",
        )


def adaptive_move_open_contract() -> Mapping[str, object]:
    body = {
        "schema": CUDA_ADAPTIVE_MOVE_OPEN_SCHEMA,
        "abi": CUDA_ADAPTIVE_MOVE_OPEN_ABI,
        "selection": {
            name: getattr(DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION, name)
            for name in DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION.__dataclass_fields__
        },
        "restart": periodic.adaptive_move_restart_contract(),
        "open_history_disposition": {
            "fields": list(OPEN_HISTORY_FIELDS),
            "controller_mutation": False,
            "authority": SOURCE_ANCHORS["adaptive_argument_inventory"],
            "consumer": "separately evolved by CM1 radiative open-boundary operators",
            "restart_persisted": True,
            "restart_authority": [
                SOURCE_ANCHORS["open_history_restart_write"],
                SOURCE_ANCHORS["open_history_restart_read"],
            ],
            "transport_layout": {
                "radbcw/radbce": "C-f32[k,j] shape [nk+2,nj+2*ngxy]",
                "radbcs/radbcn": "C-f32[k,i] shape [nk+2,ni+2*ngxy]",
            },
        },
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_ADAPTIVE_MOVE_OPEN_CLAIM_LIMIT,
    }
    encoded = json.dumps(
        body, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return MappingProxyType(
        {**body, "contract_sha256": hashlib.sha256(encoded).hexdigest()}
    )


def _require_host_open_histories(
    histories: Mapping[str, Any], *, ni: int, nj: int, nk: int, ngxy: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    if not isinstance(histories, Mapping):
        raise TypeError("open_histories must be a mapping")
    if set(histories) != set(OPEN_HISTORY_FIELDS):
        raise ValueError("open-history inventory must be exactly radbcw/e/s/n")
    shapes = {
        "radbcw": (nk + 2, nj + 2 * ngxy),
        "radbce": (nk + 2, nj + 2 * ngxy),
        "radbcs": (nk + 2, ni + 2 * ngxy),
        "radbcn": (nk + 2, ni + 2 * ngxy),
    }
    result = {}
    for name in OPEN_HISTORY_FIELDS:
        value = histories[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(
                f"{name} must be writeable C-order float32 shape {shapes[name]}"
            )
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains non-finite values")
        result[name] = value
    values = tuple(result.values())
    if any(
        np.shares_memory(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    ):
        raise ValueError("open-boundary history arrays must not alias")
    return result


def cpu_open_pressure_center(
    pressure_lowest: Any,
    xh: Any,
    yh: Any,
    xfref: Any,
    yfref: Any,
    *,
    centerx: float,
    centery: float,
) -> periodic.AdaptivePressureCenter:
    """Transcribe getpsmth/getpcent with serial code-2 ``bc2d`` halos."""

    np = importlib.import_module("numpy")
    p, xh, yh, xfref, yfref, ni, nj = periodic._require_center_arrays(
        pressure_lowest, xh, yh, xfref, yfref
    )
    centerx32, centery32 = np.float32(centerx), np.float32(centery)
    if not bool(np.isfinite(centerx32)) or not bool(np.isfinite(centery32)):
        raise ValueError("domain center must be finite float32")
    work = np.ascontiguousarray(p, dtype=np.float32).copy()
    one_ninth = np.float32(0.11111111)
    for _ in range(100):
        padded = np.pad(work, ((1, 1), (1, 1)), mode="edge")
        east, west = padded[1:-1, 2:], padded[1:-1, :-2]
        north, south = padded[2:, 1:-1], padded[:-2, 1:-1]
        northeast, northwest = padded[2:, 2:], padded[2:, :-2]
        southeast, southwest = padded[:-2, 2:], padded[:-2, :-2]
        cardinal = periodic._f32_add(
            np,
            periodic._f32_add(np, east, west),
            periodic._f32_add(np, north, south),
        )
        diagonal = periodic._f32_add(
            np,
            periodic._f32_add(np, northeast, northwest),
            periodic._f32_add(np, southeast, southwest),
        )
        total = periodic._f32_add(np, work, periodic._f32_add(np, cardinal, diagonal))
        work = np.ascontiguousarray(periodic._f32_mul(np, one_ninth, total))

    padded = np.pad(work, ((1, 0), (1, 0)), mode="edge")
    west = padded[1:, :-1]
    south = padded[:-1, 1:]
    southwest = padded[:-1, :-1]
    zeta = periodic._f32_mul(
        np,
        np.float32(0.25),
        periodic._f32_add(
            np,
            periodic._f32_add(np, work, southwest),
            periodic._f32_add(np, west, south),
        ),
    )
    minimum_flat = int(np.argmin(zeta))
    minimum_j, minimum_i = divmod(minimum_flat, ni)

    ca = np.float64(0.0)
    cx = np.float64(0.0)
    cy = np.float64(0.0)
    for j in range(nj):
        dy = periodic._f32_sub(np, yh[j], centery32)
        dy2 = periodic._f32_mul(np, dy, dy)
        for i in range(ni):
            dx = periodic._f32_sub(np, xh[i], centerx32)
            radius2 = periodic._f32_add(np, periodic._f32_mul(np, dx, dx), dy2)
            if np.sqrt(radius2, dtype=np.float32) < np.float32(300000.0):
                aa = work[j, i]
                ca = np.float64(ca + np.float64(aa))
                cx = np.float64(cx + np.float64(periodic._f32_mul(np, aa, xh[i])))
                cy = np.float64(cy + np.float64(periodic._f32_mul(np, aa, yh[j])))
    if not abs(float(ca)) > float(np.float32(1.0e-30)):
        raise ValueError("CM1 adaptive open pressure centroid has zero weight")
    cx = np.float64(cx / ca)
    cy = np.float64(cy / ca)
    icenter = min(range(ni + 1), key=lambda i: abs(np.float64(xfref[i]) - cx))
    jcenter = min(range(nj + 1), key=lambda j: abs(np.float64(yfref[j]) - cy))
    return periodic.AdaptivePressureCenter(
        work,
        minimum_i + 1,
        minimum_j + 1,
        float(xfref[minimum_i]),
        float(yfref[minimum_j]),
        icenter + 1,
        jcenter + 1,
        float(np.float32(cx)),
        float(np.float32(cy)),
        float(ca),
    )


def cpu_advance_adaptive_move_open_after_large_step(
    state: periodic.AdaptiveMoveState,
    winds: Mapping[str, Any],
    open_histories: Mapping[str, Any],
    *,
    nstep: int,
    mtime_after: float,
    dbldt: float,
    adapt_move_frq: float,
    centerx: float,
    centery: float,
    pressure_lowest: Any | None = None,
    xh: Any | None = None,
    yh: Any | None = None,
    xfref: Any | None = None,
    yfref: Any | None = None,
    dowriteout: bool = False,
    dohifrqwrite: bool = False,
    doazimwrite: bool = False,
    restarted: bool = False,
    vertical_grid_proof: Any,
    selection: AdaptiveMoveOpenSelection = DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION,
) -> periodic.AdaptiveMoveTransition:
    """Apply the completed-step controller with code-2 open pressure halos."""

    require_adaptive_move_open_selection(selection)
    periodic.validate_adaptive_move_vertical_grid(vertical_grid_proof)
    periodic.pack_adaptive_move_state(state)
    a = periodic._require_host_winds(winds)
    u_shape, v_shape = a["u0"].shape, a["v0"].shape
    nk, nj, ni = u_shape[0], u_shape[1], v_shape[2]
    histories = _require_host_open_histories(
        open_histories, ni=ni, nj=nj, nk=nk, ngxy=selection.ngxy
    )
    np = importlib.import_module("numpy")
    if any(
        np.shares_memory(history, wind)
        for history in histories.values()
        for wind in a.values()
    ):
        raise ValueError("open histories must not alias adaptive wind carriers")
    if isinstance(nstep, bool) or not isinstance(nstep, int) or nstep < 0:
        raise ValueError("nstep must be a nonnegative integer")
    if not all(
        type(value) is bool
        for value in (dowriteout, dohifrqwrite, doazimwrite, restarted)
    ):
        raise TypeError("cadence and restart flags must be bool")
    frequency = np.float32(adapt_move_frq)
    centerx32, centery32 = np.float32(centerx), np.float32(centery)
    if not bool(np.isfinite(frequency)) or frequency <= np.float32(0.0):
        raise ValueError("adapt_move_frq must be a positive finite float32")
    if not bool(np.isfinite(centerx32)) or not bool(np.isfinite(centery32)):
        raise ValueError("domain center must be finite float32")
    if (
        not math.isfinite(float(mtime_after))
        or not math.isfinite(float(dbldt))
        or float(dbldt) <= 0.0
    ):
        raise ValueError("mtime_after and positive dbldt must be finite")

    old_u, old_v = np.float32(state.umove), np.float32(state.vmove)
    state.domainlocx = float(
        np.float64(state.domainlocx) + np.float64(dbldt) * np.float64(old_u)
    )
    state.domainlocy = float(
        np.float64(state.domainlocy) + np.float64(dbldt) * np.float64(old_v)
    )
    due = float(mtime_after) >= float(state.adaptmovetim)
    diagnosis = None
    diagnosed = (
        due
        and nstep >= 1
        and (dowriteout or dohifrqwrite or doazimwrite)
        and not restarted
    )
    if diagnosed:
        if any(value is None for value in (pressure_lowest, xh, yh, xfref, yfref)):
            raise ValueError(
                "due open center diagnosis requires pressure and coordinates"
            )
        if getattr(pressure_lowest, "shape", None) != (nj, ni):
            raise ValueError("open pressure domain differs from wind/history domain")
        center_values = (pressure_lowest, xh, yh, xfref, yfref)
        if any(
            np.shares_memory(center_value, other)
            for center_value in center_values
            for other in (*a.values(), *histories.values())
        ):
            raise ValueError("pressure/coordinates must not alias state carriers")
        diagnosis = cpu_open_pressure_center(
            pressure_lowest,
            xh,
            yh,
            xfref,
            yfref,
            centerx=float(centerx32),
            centery=float(centery32),
        )
        state.icenter = diagnosis.icenter
        state.jcenter = diagnosis.jcenter
        state.xcenter = diagnosis.xcenter
        state.ycenter = diagnosis.ycenter

    delta_u = np.float32(0.0)
    delta_v = np.float32(0.0)
    if due and nstep >= 1:
        term2 = np.float32(0.5)
        center_departure_u = periodic._f32_div(
            np,
            periodic._f32_mul(
                np,
                term2,
                periodic._f32_sub(np, centerx32, np.float32(state.xcenter)),
            ),
            frequency,
        )
        lag_u = periodic._f32_mul(
            np,
            periodic._f32_sub(np, np.float32(1.0), term2),
            periodic._f32_sub(np, np.float32(state.umovem2), np.float32(state.umovem1)),
        )
        new_u = periodic._f32_add(
            np, periodic._f32_sub(np, old_u, center_departure_u), lag_u
        )
        center_departure_v = periodic._f32_div(
            np,
            periodic._f32_mul(
                np,
                term2,
                periodic._f32_sub(np, centery32, np.float32(state.ycenter)),
            ),
            frequency,
        )
        lag_v = periodic._f32_mul(
            np,
            periodic._f32_sub(np, np.float32(1.0), term2),
            periodic._f32_sub(np, np.float32(state.vmovem2), np.float32(state.vmovem1)),
        )
        new_v = periodic._f32_add(
            np, periodic._f32_sub(np, old_v, center_departure_v), lag_v
        )
        state.umovem2, state.vmovem2 = (
            float(np.float32(state.umovem1)),
            float(np.float32(state.vmovem1)),
        )
        state.umovem1, state.vmovem1 = float(old_u), float(old_v)
        state.umove, state.vmove = float(new_u), float(new_v)
        delta_u = periodic._f32_sub(np, old_u, new_u)
        delta_v = periodic._f32_sub(np, old_v, new_v)
        for name in ("u0", "ua", "u3d", "ug"):
            np.add(a[name], delta_u, out=a[name], dtype=np.float32)
        for name in ("v0", "va", "v3d", "vg"):
            np.add(a[name], delta_v, out=a[name], dtype=np.float32)
    if due:
        state.nwritemv += 1
        state.adaptmovetim = float(
            np.float64(state.adaptmovetim) + np.float64(frequency)
        )
    periodic.pack_adaptive_move_state(state)
    return periodic.AdaptiveMoveTransition(
        due, diagnosed, float(delta_u), float(delta_v), diagnosis
    )


CUDA_ADAPTIVE_MOVE_OPEN_SOURCE = r"""// gpuwm-cm1.cuda-adaptive-move-open-kernels/v1
static_assert(sizeof(float)==4,"float32 required");
static_assert(sizeof(double)==8,"float64 required");
static_assert(sizeof(int)==4,"int32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
extern "C" __global__ void cm1_adapt_open_smooth9_f32_v1(const float* src,float* dst,unsigned ni,unsigned nj){unsigned n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=ni*nj)return;unsigned i=n%ni,j=n/ni,im=i?i-1:0,ip=i+1<ni?i+1:ni-1,jm=j?j-1:0,jp=j+1<nj?j+1:nj-1;float ew=a32(src[j*ni+ip],src[j*ni+im]);float ns=a32(src[jp*ni+i],src[jm*ni+i]);float card=a32(ew,ns);float dn=a32(src[jp*ni+ip],src[jp*ni+im]);float ds=a32(src[jm*ni+ip],src[jm*ni+im]);float diag=a32(dn,ds);dst[n]=m32(0.11111111f,a32(src[n],a32(card,diag)));}
extern "C" __global__ void cm1_adapt_open_center_serial_f32_v1(const float* p,const float* xh,const float* yh,const float* xf,const float* yf,float* sf,double* sd,int* si,int* status,unsigned ni,unsigned nj,float centerx,float centery,double mtime,int enabled){if(blockIdx.x||threadIdx.x||!enabled||mtime<sd[2])return;float rmin=1.0e30f;int ipmin=1,jpmin=1;for(unsigned j=0;j<nj;++j){unsigned jm=j?j-1:0;for(unsigned i=0;i<ni;++i){unsigned im=i?i-1:0;float z=m32(0.25f,a32(a32(p[j*ni+i],p[jm*ni+im]),a32(p[j*ni+im],p[jm*ni+i])));if(z<rmin){rmin=z;ipmin=(int)i+1;jpmin=(int)j+1;}}}double ca=0.0,cx=0.0,cy=0.0;for(unsigned j=0;j<nj;++j){float dy=s32(yh[j],centery);float dy2=m32(dy,dy);for(unsigned i=0;i<ni;++i){float dx=s32(xh[i],centerx);float rr=sqrtf(a32(m32(dx,dx),dy2));if(rr<300000.0f){float aa=p[j*ni+i];ca=__dadd_rn(ca,double(aa));cx=__dadd_rn(cx,double(m32(aa,xh[i])));cy=__dadd_rn(cy,double(m32(aa,yh[j])));}}}if(!(fabs(ca)>double(1.0e-30f))){status[0]=1;return;}cx=__ddiv_rn(cx,ca);cy=__ddiv_rn(cy,ca);int ic=ipmin,jc=jpmin;double minval=1.0e30;for(unsigned i=0;i<=ni;++i){double d=fabs(double(xf[i])-cx);if(d<minval){minval=d;ic=(int)i+1;}}minval=1.0e30;for(unsigned j=0;j<=nj;++j){double d=fabs(double(yf[j])-cy);if(d<minval){minval=d;jc=(int)j+1;}}si[0]=ic;si[1]=jc;sf[6]=(float)cx;sf[7]=(float)cy;}
extern "C" __global__ void cm1_adapt_open_commit_f32_f64_v1(float* sf,double* sd,int* si,float* delta,int* active,const int* status,int nstep,double mtime,double dbldt,float frq,float centerx,float centery){if(blockIdx.x||threadIdx.x)return;float oldu=sf[0],oldv=sf[3];sd[0]=__dadd_rn(sd[0],__dmul_rn(dbldt,double(oldu)));sd[1]=__dadd_rn(sd[1],__dmul_rn(dbldt,double(oldv)));delta[0]=0.0f;delta[1]=0.0f;active[0]=0;if(status[0])return;if(mtime>=sd[2]){if(nstep>=1){float term2=0.5f;float newu=a32(s32(oldu,d32(m32(term2,s32(centerx,sf[6])),frq)),m32(s32(1.0f,term2),s32(sf[2],sf[1])));float newv=a32(s32(oldv,d32(m32(term2,s32(centery,sf[7])),frq)),m32(s32(1.0f,term2),s32(sf[5],sf[4])));sf[2]=sf[1];sf[5]=sf[4];sf[1]=oldu;sf[4]=oldv;sf[0]=newu;sf[3]=newv;delta[0]=s32(oldu,newu);delta[1]=s32(oldv,newv);active[0]=1;}si[2]+=1;sd[2]=__dadd_rn(sd[2],double(frq));}}
extern "C" __global__ void cm1_adapt_open_add_delta_f32_v1(float* a,unsigned long long n,const float* delta,const int* active,unsigned which){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(active[0]&&q<n)a[q]=a32(a[q],delta[which]);}
"""
CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_BYTES = CUDA_ADAPTIVE_MOVE_OPEN_SOURCE.encode("utf-8")
CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_SHA256 = (
    "d2b64e13916314b91a1543984652ad25a31b68b116bf5164bf3fe36ce0ae6fde"
)


@dataclass(frozen=True, slots=True)
class CompiledAdaptiveMoveOpen:
    module_owner: Any
    smooth_kernel: Any
    center_kernel: Any
    commit_kernel: Any
    add_kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_adaptive_move_open() -> CompiledAdaptiveMoveOpen:
    actual = hashlib.sha256(CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_BYTES).hexdigest()
    if actual != CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_SHA256:
        raise RuntimeError(f"packaged adaptive-move-open CUDA source differs: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_ADAPTIVE_MOVE_OPEN_SOURCE, "cm1_adaptive_move_open_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_ADAPTIVE_MOVE_OPEN_SOURCE,
        "cm1_adaptive_move_open_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    names = (
        "cm1_adapt_open_smooth9_f32_v1",
        "cm1_adapt_open_center_serial_f32_v1",
        "cm1_adapt_open_commit_f32_f64_v1",
        "cm1_adapt_open_add_delta_f32_v1",
    )
    text = ptx.decode("utf-8")
    for name in names:
        if not re.search(rf"\.entry\s+{name}(?:\s|\()", text):
            raise ValueError(f"compiled PTX lacks {name}")
    module = function.Module()
    module.load(cubin)
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
    return CompiledAdaptiveMoveOpen(
        module,
        module.get_function(names[0]),
        module.get_function(names[1]),
        module.get_function(names[2]),
        module.get_function(names[3]),
        ptx,
        cubin,
        arch,
        identity,
    )


def _require_device_open_histories(
    cupy: Any,
    histories: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    ngxy: int,
) -> dict[str, Any]:
    if not isinstance(histories, Mapping) or set(histories) != set(OPEN_HISTORY_FIELDS):
        raise ValueError("device open-history inventory must be exactly radbcw/e/s/n")
    shapes = {
        "radbcw": (nk + 2, nj + 2 * ngxy),
        "radbce": (nk + 2, nj + 2 * ngxy),
        "radbcs": (nk + 2, ni + 2 * ngxy),
        "radbcn": (nk + 2, ni + 2 * ngxy),
    }
    result = {}
    for name in OPEN_HISTORY_FIELDS:
        value = histories[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"device {name} must be C-order CuPy float32 shape {shapes[name]}"
            )
        result[name] = value
    periodic._require_nonoverlapping_device_arrays(
        result, family="adaptive-move open history"
    )
    return result


@dataclass(frozen=True, slots=True)
class CudaAdaptiveMoveOpenLaunch:
    ready_event: Any
    compiled: CompiledAdaptiveMoveOpen
    state: periodic.DeviceAdaptiveMoveState
    psmth: Any
    delta: Any
    active: Any
    diagnostic_status: Any
    diagnosis_inputs_bound: bool
    wind_arrays: Mapping[str, Any]
    open_histories: Mapping[str, Any]
    receipt: Mapping[str, object]


def launch_cuda_adaptive_move_open_after_large_step(
    state: periodic.DeviceAdaptiveMoveState,
    winds: Mapping[str, Any],
    open_histories: Mapping[str, Any],
    *,
    nstep: int,
    mtime_after: float,
    dbldt: float,
    adapt_move_frq: float,
    centerx: float,
    centery: float,
    pressure_lowest: Any | None = None,
    xh: Any | None = None,
    yh: Any | None = None,
    xfref: Any | None = None,
    yfref: Any | None = None,
    dowriteout: bool = False,
    dohifrqwrite: bool = False,
    doazimwrite: bool = False,
    restarted: bool = False,
    vertical_grid_proof: Any,
    selection: AdaptiveMoveOpenSelection = DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION,
    stream: Any | None = None,
    compiled: CompiledAdaptiveMoveOpen | None = None,
) -> CudaAdaptiveMoveOpenLaunch:
    """Enqueue the exact open-halo diagnosis and adaptive controller."""

    require_adaptive_move_open_selection(selection)
    lowest_level_m = periodic.validate_adaptive_move_vertical_grid(vertical_grid_proof)
    cupy = importlib.import_module("cupy")
    periodic._require_device_state(cupy, state)
    a = periodic._require_device_winds(cupy, winds)
    u_shape, v_shape = a["u0"].shape, a["v0"].shape
    nk, nj, ni = u_shape[0], u_shape[1], v_shape[2]
    histories = _require_device_open_histories(
        cupy,
        open_histories,
        ni=ni,
        nj=nj,
        nk=nk,
        ngxy=selection.ngxy,
    )
    periodic._require_nonoverlapping_device_arrays(
        {
            "state.f32": state.f32,
            "state.f64": state.f64,
            "state.i32": state.i32,
            **{f"wind.{name}": value for name, value in a.items()},
            **{f"history.{name}": value for name, value in histories.items()},
        },
        family="adaptive-move-open state",
    )
    if isinstance(nstep, bool) or not isinstance(nstep, int) or nstep < 0:
        raise ValueError("nstep must be a nonnegative integer")
    if not all(
        type(value) is bool
        for value in (dowriteout, dohifrqwrite, doazimwrite, restarted)
    ):
        raise TypeError("cadence and restart flags must be bool")
    frequency = cupy.float32(adapt_move_frq)
    centerx32, centery32 = cupy.float32(centerx), cupy.float32(centery)
    if not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
        raise ValueError("adapt_move_frq must be positive finite float32")
    if not math.isfinite(float(centerx32)) or not math.isfinite(float(centery32)):
        raise ValueError("domain center coordinates must be finite float32")
    if (
        not math.isfinite(float(mtime_after))
        or not math.isfinite(float(dbldt))
        or float(dbldt) <= 0.0
    ):
        raise ValueError("mtime_after and positive dbldt must be finite")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    diagnose = (
        nstep >= 1 and (dowriteout or dohifrqwrite or doazimwrite) and not restarted
    )
    if diagnose:
        if any(value is None for value in (pressure_lowest, xh, yh, xfref, yfref)):
            raise ValueError(
                "requested open center diagnosis lacks pressure/coordinates"
            )
        with stream:
            actual_ni, actual_nj = periodic._require_device_center_arrays(
                cupy, pressure_lowest, xh, yh, xfref, yfref
            )
        if (actual_ni, actual_nj) != (ni, nj):
            raise ValueError("open pressure domain differs from wind/history domain")
        periodic._require_nonoverlapping_device_arrays(
            {
                "state.f32": state.f32,
                "state.f64": state.f64,
                "state.i32": state.i32,
                **{f"wind.{name}": value for name, value in a.items()},
                **{f"history.{name}": value for name, value in histories.items()},
                "pressure_lowest": pressure_lowest,
                "xh": xh,
                "yh": yh,
                "xfref": xfref,
                "yfref": yfref,
            },
            family="adaptive-move-open launch",
        )
    compiled = compile_cuda_adaptive_move_open() if compiled is None else compiled
    if not isinstance(compiled, CompiledAdaptiveMoveOpen):
        raise TypeError("compiled must be a CompiledAdaptiveMoveOpen")
    threads = 256
    with stream:
        delta = cupy.zeros(2, dtype=cupy.float32)
        active = cupy.zeros(1, dtype=cupy.int32)
        diagnostic_status = cupy.zeros(1, dtype=cupy.int32)
        if diagnose:
            first = cupy.empty_like(pressure_lowest)
            second = cupy.empty_like(pressure_lowest)
            cupy.copyto(first, pressure_lowest)
            blocks = ((ni * nj + threads - 1) // threads, 1, 1)
            for _ in range(100):
                compiled.smooth_kernel(
                    blocks,
                    (threads, 1, 1),
                    (first, second, cupy.uint32(ni), cupy.uint32(nj)),
                    stream=stream,
                )
                first, second = second, first
            psmth = first
            compiled.center_kernel(
                (1, 1, 1),
                (1, 1, 1),
                (
                    psmth,
                    xh,
                    yh,
                    xfref,
                    yfref,
                    state.f32,
                    state.f64,
                    state.i32,
                    diagnostic_status,
                    cupy.uint32(ni),
                    cupy.uint32(nj),
                    centerx32,
                    centery32,
                    cupy.float64(mtime_after),
                    cupy.int32(1),
                ),
                stream=stream,
            )
        else:
            psmth = cupy.empty((0,), dtype=cupy.float32)
        compiled.commit_kernel(
            (1, 1, 1),
            (1, 1, 1),
            (
                state.f32,
                state.f64,
                state.i32,
                delta,
                active,
                diagnostic_status,
                cupy.int32(nstep),
                cupy.float64(mtime_after),
                cupy.float64(dbldt),
                frequency,
                centerx32,
                centery32,
            ),
            stream=stream,
        )
        for which, names in (
            (0, ("u0", "ua", "u3d", "ug")),
            (1, ("v0", "va", "v3d", "vg")),
        ):
            for name in names:
                value = a[name]
                blocks = ((int(value.size) + threads - 1) // threads, 1, 1)
                compiled.add_kernel(
                    blocks,
                    (threads, 1, 1),
                    (
                        value,
                        cupy.uint64(value.size),
                        delta,
                        active,
                        cupy.uint32(which),
                    ),
                    stream=stream,
                )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = MappingProxyType(
        {
            "schema": CUDA_ADAPTIVE_MOVE_OPEN_SCHEMA,
            "abi": CUDA_ADAPTIVE_MOVE_OPEN_ABI,
            "selection": {
                name: getattr(selection, name)
                for name in selection.__dataclass_fields__
            },
            "cadence": {
                "nstep": nstep,
                "mtime_after": float(mtime_after),
                "dbldt": float(dbldt),
                "adapt_move_frq_f32": float(frequency),
                "diagnosis_inputs_bound": diagnose,
                "actual_due_and_diagnosis_device_gated": True,
                "restarted": restarted,
            },
            "vertical_grid": {
                "proof_sha256": periodic.adaptive_move_vertical_grid_identity(
                    vertical_grid_proof
                ),
                "lowest_scalar_level_m": lowest_level_m,
                "source_k_index": 1,
            },
            "open_histories": {
                name: {
                    "pointer": int(value.data.ptr),
                    "shape": list(value.shape),
                    "controller_mutation": False,
                }
                for name, value in histories.items()
            },
            "kernel": {
                "source_sha256": CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_SHA256,
                "target_arch": compiled.target_arch,
            },
            "runtime": dict(compiled.runtime_identity),
            "authority": {
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
                "config_sha256": dict(AUTHORITY_CONFIG_SHA256),
            },
            "contract_sha256": adaptive_move_open_contract()["contract_sha256"],
            "source_anchors": dict(SOURCE_ANCHORS),
            "claim_limit": CM1_ADAPTIVE_MOVE_OPEN_CLAIM_LIMIT,
        }
    )
    return CudaAdaptiveMoveOpenLaunch(
        event,
        compiled,
        state,
        psmth,
        delta,
        active,
        diagnostic_status,
        diagnose,
        MappingProxyType(a),
        MappingProxyType(histories),
        receipt,
    )


def validate_cuda_adaptive_move_open_completion(
    launch: CudaAdaptiveMoveOpenLaunch,
) -> None:
    if not isinstance(launch, CudaAdaptiveMoveOpenLaunch):
        raise TypeError("launch must be a CudaAdaptiveMoveOpenLaunch")
    cupy = importlib.import_module("cupy")
    launch.ready_event.synchronize()
    if int(cupy.asnumpy(launch.diagnostic_status)[0]) != 0:
        raise ValueError("CM1 adaptive open pressure centroid has zero weight")
    periodic.pack_adaptive_move_state(
        periodic.download_adaptive_move_state(launch.state)
    )
    for family, arrays in (
        ("wind", launch.wind_arrays),
        ("open history", launch.open_histories),
    ):
        for name, value in arrays.items():
            if not bool(cupy.all(cupy.isfinite(value)).item()):
                raise ValueError(
                    f"completed adaptive-move {family} {name} is non-finite"
                )


__all__ = (
    "AUTHORITY_CONFIG_SHA256",
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "AdaptiveMoveOpenSelection",
    "CM1_ADAPTIVE_MOVE_OPEN_CLAIM_LIMIT",
    "CUDA_ADAPTIVE_MOVE_OPEN_ABI",
    "CUDA_ADAPTIVE_MOVE_OPEN_SCHEMA",
    "CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_BYTES",
    "CUDA_ADAPTIVE_MOVE_OPEN_SOURCE_SHA256",
    "CompiledAdaptiveMoveOpen",
    "CudaAdaptiveMoveOpenLaunch",
    "DEFAULT_ADAPTIVE_MOVE_OPEN_SELECTION",
    "OPEN_HISTORY_FIELDS",
    "SOURCE_ANCHORS",
    "adaptive_move_open_contract",
    "compile_cuda_adaptive_move_open",
    "cpu_advance_adaptive_move_open_after_large_step",
    "cpu_open_pressure_center",
    "launch_cuda_adaptive_move_open_after_large_step",
    "require_adaptive_move_open_selection",
    "validate_cuda_adaptive_move_open_completion",
)
