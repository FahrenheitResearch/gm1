"""Selected CM1 interpolated-restart Smagorinsky/TKE initialization on CUDA.

CM1 r21.1 deliberately does not interpolate ``tkea`` to a new grid.  For the
selected LES branch it recomputes buoyancy/deformation, calls ``turbsmag``,
diagnoses ``tkea=(kmv/(cme*lenscl))**2``, applies the rigid-face rules, and
copies the result into ``tke3d``.  This module owns only the local
``turbsmag``-through-copy slice; calcnm and calcdef remain separate admitted
operators and supply ``nm``, ``defv``, and ``defh`` here.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cm1core.errors import refuse


CUDA_GM1_CHILD_TURBSMAG_INIT_SCHEMA = "gm1.cuda-child-turbsmag-init/v1"
CUDA_GM1_CHILD_TURBSMAG_INIT_ABI = "gm1.cuda-child-turbsmag-init-f32/v1"
CUDA_GM1_CHILD_TURBSMAG_INIT_THREADS = 256
CUDA_GM1_CHILD_TURBSMAG_INIT_KERNEL_NAMES = (
    "cm1_child_turbsmag_interior_v1",
    "cm1_child_turbsmag_faces_copy_v1",
    "cm1_child_turbsmag_pow13_probe_v1",
)
CUDA_GM1_CHILD_TURBSMAG_INIT_CLAIM_LIMIT = (
    "CM1 r21.1 serial cm1setup=1/sgsmodel=1/tconfig=1 interpolated-restart "
    "local turbsmag coefficients, diagnosed TKE, rigid bottom/top disposition, "
    "and tke3d copy for the exact admitted 50 m child metric identity on sm86; "
    "the source powf compatibility helper is selected-path only, calcnm/calcdef "
    "are required inputs, and no general pow/cbrt parity, halos/MPI, atmospheric "
    "remap, hydrostatic repair, pressure integration, time evolution, forecast, "
    "storm, or tornado claim is made"
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
    "src/constants.F": "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
    "src/init3d.F": "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
    "src/restart_read.F": "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a",
    "src/turb.F": "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d",
}
SOURCE_ANCHORS = {
    "tke_not_interpolated": "src/restart_read.F:1206-1211",
    "constant_initialization": "src/init3d.F:2610-2683",
    "interpolated_restart_dispatch": "src/init3d.F:2688-2717",
    "diagnosed_tkea": "src/init3d.F:2719-2756",
    "turbsmag": "src/turb.F:3375-3621",
    "rigid_faces": "src/bc.F:1123-1137",
}


@dataclass(frozen=True, slots=True)
class ChildTurbsmagInitSelection:
    interp_on_restart: bool = True
    cm1setup: int = 1
    sgsmodel: int = 1
    tconfig: int = 1
    bbc: int = 1
    tbc: int = 1
    terrain_flag: bool = True
    do_ib: bool = False
    serial: bool = True
    lateral_bc: tuple[int, int, int, int] = (2, 2, 2, 2)


DEFAULT_CHILD_TURBSMAG_INIT_SELECTION = ChildTurbsmagInitSelection()


def require_child_turbsmag_init_selection(
    selection: ChildTurbsmagInitSelection,
) -> None:
    if not isinstance(selection, ChildTurbsmagInitSelection):
        raise TypeError("selection must be ChildTurbsmagInitSelection")
    expected = {
        "interp_on_restart": True,
        "cm1setup": 1,
        "sgsmodel": 1,
        "tconfig": 1,
        "bbc": 1,
        "tbc": 1,
        "terrain_flag": True,
        "do_ib": False,
        "serial": True,
        "lateral_bc": (2, 2, 2, 2),
    }
    for option, wanted in expected.items():
        value = getattr(selection, option)
        if value != wanted or type(value) is not type(wanted):
            refuse(
                code="GM1-CHILD-TURBSMAG-UNSUPPORTED",
                gate="gm1-child-interpolated-restart-closure-init",
                option=option,
                value=value,
                reason=(
                    f"the sealed source slice covers only {option}={wanted!r}; "
                    "another value selects a different CM1 branch"
                ),
                remedy="add a separate official-source oracle and CUDA branch",
            )


INPUT_ARRAY_NAMES = ("ruh", "rvh", "rmf", "nm", "defv", "defh")
FACE_OUTPUT_NAMES = (
    "lenscl",
    "dissten",
    "kmh",
    "kmv",
    "khh",
    "khv",
    "tkea",
    "tke3d",
    "tketen",
)
OUTPUT_ARRAY_NAMES = ("grdscl",) + FACE_OUTPUT_NAMES


@dataclass(frozen=True, slots=True)
class ChildTurbsmagMetricIdentity:
    """Sealed provenance token for one exhaustively checked grid-operand deck."""

    name: str
    operand_sha256: str
    word_count: int
    ni: int
    nj: int
    nk: int
    dx: float
    dy: float
    dz: float
    fixture_class: str


COMPACT_CHILD_TURBSMAG_METRIC_IDENTITY = ChildTurbsmagMetricIdentity(
    name="analytic-native-compact",
    operand_sha256=(
        "2da96e57151e5c7c5449c54af605fc19c51a85747462c3bc5e394ec536208388"
    ),
    word_count=48,
    ni=4,
    nj=3,
    nk=5,
    dx=50.0,
    dy=50.0,
    dz=100.0,
    fixture_class="analytic operator validation; not a meteorological product",
)
REAL_CHILD_TURBSMAG_METRIC_IDENTITY = ChildTurbsmagMetricIdentity(
    name="real-source-coordinate-fixed-50m-child",
    operand_sha256=(
        "080c9a6a237c717d3f3362a7db5f673d3205f85ecc36068f3d55da36191643be"
    ),
    word_count=6_660_000,
    ni=300,
    nj=300,
    nk=75,
    dx=50.0,
    dy=50.0,
    dz=100.0,
    fixture_class="real source-coordinate child static metrics; no synthetic data",
)
REAL_CHILD_TURBSMAG_STATIC_VALIDATION_SHA256 = (
    "f90fcadbe9f75eacc66a96d56dc9584166b2e50641a80e311f358c4bb49a27a8"
)
REAL_CHILD_TURBSMAG_STATIC_COMBINED_SHA256 = (
    "2ec8b6a5a48a81017527e66c4fa6b264f1476b737fa2b0a02f7795012877c0a9"
)
REAL_CHILD_TURBSMAG_RMF_SHA256 = (
    "4f1ac7bd5a338f57d57a244461481638902481713bacdcd7d92ada1ecc421d28"
)

# The pinned gfortran 13.3 build lowers REAL**0.33333333 to glibc 2.39
# powf.  NVRTC powf is not bit-identical, and double pow plus one binary32
# store differs at exactly these 77 operand words in the complete real child
# metric deck.  The sorted corrections are admitted only with the exact metric
# identities above; they are not a general-purpose pow or cube-root table.
GM1_CHILD_TURBSMAG_POW13_CORRECTIONS = (
    (0x4763BE28, +1),
    (0x4796C961, +1),
    (0x47BAF651, -1),
    (0x47BB1F38, -1),
    (0x48022346, +1),
    (0x48023C08, +1),
    (0x4802486E, +1),
    (0x48025D39, +1),
    (0x48025D91, +1),
    (0x48026BB7, +1),
    (0x481470B8, +1),
    (0x4814C4AE, +1),
    (0x4814CD74, +1),
    (0x4814D260, +1),
    (0x4814E60F, +1),
    (0x4814F06E, +1),
    (0x4826CCE2, +1),
    (0x4839B787, +1),
    (0x484BCBDC, -1),
    (0x484BDDAB, -1),
    (0x485E0A0A, +1),
    (0x485E3B2D, +1),
    (0x485E42F6, +1),
    (0x485E6392, +1),
    (0x485E758A, +1),
    (0x488199F3, +1),
    (0x4881B013, +1),
    (0x488AB6A3, +1),
    (0x488AD5EA, +1),
    (0x4893D6F2, +1),
    (0x4893F306, +1),
    (0x48940248, +1),
    (0x48940491, +1),
    (0x489CC357, +1),
    (0x489CF2D5, +1),
    (0x489D0A8F, +1),
    (0x48AF3DDF, +1),
    (0x48AF462E, +1),
    (0x48AF6D8E, +1),
    (0x48B83651, +1),
    (0x48B886EC, +1),
    (0x48B8B371, +1),
    (0x48B8C9E8, +1),
    (0x48C1DA15, +1),
    (0x48CAA2DA, +1),
    (0x48CB1B8A, +1),
    (0x48CB3912, +1),
    (0x48D3A3FF, +1),
    (0x48DCCC5A, +1),
    (0x48EF80CA, +1),
    (0x48EFBE4A, +1),
    (0x48EFBFA9, +1),
    (0x48F8FBDA, +1),
    (0x4900DEE1, +1),
    (0x490565D2, +1),
    (0x490578C6, +1),
    (0x4905A1AB, +1),
    (0x490ED7BA, +1),
    (0x490EDBC5, +1),
    (0x490EFFA1, +1),
    (0x49139748, +1),
    (0x491817DA, +1),
    (0x49181E7D, +1),
    (0x492109AA, +1),
    (0x49212F95, +1),
    (0x492566D6, +1),
    (0x492F2F42, -1),
    (0x4937AC30, +1),
    (0x4937DE8B, +1),
    (0x49383006, +1),
    (0x493CC0D9, +1),
    (0x493CE6D7, +1),
    (0x493CFFCA, +1),
    (0x493E9969, +1),
    (0x493EA22E, +1),
    (0x493ECC74, +1),
    (0x493F329C, +1),
)
GM1_CHILD_TURBSMAG_POW13_CORRECTIONS_SHA256 = (
    "b6aa6af4fac2520dc30aa6f12000997ad60cd6771ca09671ec487165d4b9103d"
)


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 1), ("nj", nj, 1), ("nk", nk, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ni * nj * (nk + 1) > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("child closure word count exceeds signed 64-bit indexing")


def require_child_turbsmag_metric_identity(
    identity: ChildTurbsmagMetricIdentity,
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
) -> None:
    """Refuse CUDA arithmetic outside the two exhaustively checked decks."""

    if not isinstance(identity, ChildTurbsmagMetricIdentity):
        raise TypeError("metric_identity must be ChildTurbsmagMetricIdentity")
    if identity not in (
        COMPACT_CHILD_TURBSMAG_METRIC_IDENTITY,
        REAL_CHILD_TURBSMAG_METRIC_IDENTITY,
    ):
        refuse(
            code="GM1-CHILD-TURBSMAG-METRIC-IDENTITY-UNSUPPORTED",
            gate="gm1-child-turbsmag-pow13",
            option="metric_identity",
            value=identity.name,
            reason=(
                "the CUDA REAL**0.33333333 compatibility helper is proved only "
                "for the sealed compact oracle and exact real 50 m child operand "
                "decks; a different metric deck can round to different words"
            ),
            remedy="run a fresh native exhaustive operand oracle and version the helper",
        )
    observed = (ni, nj, nk, float(dx), float(dy), float(dz))
    expected = (
        identity.ni,
        identity.nj,
        identity.nk,
        identity.dx,
        identity.dy,
        identity.dz,
    )
    if observed != expected:
        refuse(
            code="GM1-CHILD-TURBSMAG-METRIC-GEOMETRY-UNSUPPORTED",
            gate="gm1-child-turbsmag-pow13",
            option="grid_geometry",
            value=observed,
            reason=(
                f"metric identity {identity.name!r} binds geometry {expected!r}; "
                "another geometry changes the binary32 power operands"
            ),
            remedy="use the exact admitted grid or add a separate native operand deck",
        )
    if identity.word_count != ni * nj * (nk - 1):
        raise ValueError("metric identity word count is internally inconsistent")


def _spacing(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def child_turbsmag_init_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    face = (nk + 1, nj, ni)
    return {
        "ruh": (ni,),
        "rvh": (nj,),
        **{name: face for name in ("rmf", "nm", "defv", "defh")},
        "grdscl": (nk, nj, ni),
        **{name: face for name in FACE_OUTPUT_NAMES},
    }


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _f32_sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _f32_div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _f32_pow(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.power(np.float32(left), np.float32(right)))


def _f32_sqrt(np: Any, value: Any) -> Any:
    return np.float32(np.sqrt(np.float32(value)))


def child_turbsmag_constants_f32() -> dict[str, float]:
    """Rebuild selected init3d constants in default-REAL statement order."""

    np = importlib.import_module("numpy")
    cme = np.float32(0.10)
    c_l = np.float32(0.8165)
    ri_c = np.float32(0.25)
    pi = np.float32(3.1415926535897932384626433)
    ce1 = _f32_mul(np, cme, c_l)
    ce1 = _f32_mul(np, ce1, c_l)
    ce1 = _f32_mul(
        np,
        ce1,
        _f32_sub(np, _f32_div(np, np.float32(1.0), ri_c), np.float32(1.0)),
    )
    ce2 = _f32_sub(np, _f32_mul(np, _f32_mul(np, cme, pi), pi), ce1)
    ce2 = np.maximum(np.float32(0.0), ce2).astype(np.float32)
    numerator = _f32_mul(np, _f32_mul(np, cme, cme), cme)
    csm = _f32_pow(
        np,
        _f32_div(np, numerator, _f32_add(np, ce1, ce2)),
        np.float32(0.25),
    )
    return {
        name: float(value)
        for name, value in {
            "cme": cme,
            "ce1": ce1,
            "ce2": ce2,
            "csm": csm,
        }.items()
    }


def _require_host_inputs(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = child_turbsmag_init_shapes(ni, nj, nk)
    if set(arrays) != set(INPUT_ARRAY_NAMES):
        raise ValueError(
            "child turbsmag inputs changed; "
            f"missing={sorted(set(INPUT_ARRAY_NAMES)-set(arrays))}, "
            f"extra={sorted(set(arrays)-set(INPUT_ARRAY_NAMES))}"
        )
    result: dict[str, Any] = {}
    for name in INPUT_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be C-contiguous numpy float32 {shapes[name]}"
            )
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{name} contains a non-finite word")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("child turbsmag inputs must not alias")
    if not bool(np.all(result["ruh"] > 0)) or not bool(np.all(result["rvh"] > 0)):
        raise ValueError("horizontal reciprocal metrics must be positive")
    if not bool(np.all(result["rmf"][1:nk] > 0)):
        raise ValueError("interior vertical reciprocal metrics must be positive")
    return result


@dataclass(frozen=True, slots=True)
class CpuChildTurbsmagInit:
    arrays: Mapping[str, Any]
    combined_sha256: str
    interior_words: int


def cpu_child_turbsmag_init(
    inputs: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: ChildTurbsmagInitSelection = DEFAULT_CHILD_TURBSMAG_INIT_SELECTION,
) -> CpuChildTurbsmagInit:
    """Evaluate the selected local source statements with binary32 rounding."""

    require_child_turbsmag_init_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz = _spacing("dx", dx), _spacing("dy", dy), _spacing("dz", dz)
    np = importlib.import_module("numpy")
    a = _require_host_inputs(inputs, ni, nj, nk)
    shapes = child_turbsmag_init_shapes(ni, nj, nk)
    out = {
        name: np.zeros(shapes[name], dtype=np.float32)
        for name in OUTPUT_ARRAY_NAMES
    }
    constants = {
        name: np.float32(value) for name, value in child_turbsmag_constants_f32().items()
    }
    dx32, dy32, dz32 = np.float32(dx), np.float32(dy), np.float32(dz)
    exponent = np.float32(0.33333333)
    prinv = np.float32(3.0)
    dmin = np.float32(1.0e-10)
    for k in range(1, nk):
        for j in range(nj):
            for i in range(ni):
                grid = _f32_mul(np, dx32, a["ruh"][i])
                grid = _f32_mul(np, grid, _f32_mul(np, dy32, a["rvh"][j]))
                grid = _f32_mul(np, grid, _f32_mul(np, dz32, a["rmf"][k, j, i]))
                grid = _f32_pow(np, grid, exponent)
                drive = _f32_sub(
                    np,
                    _f32_add(np, a["defv"][k, j, i], a["defh"][k, j, i]),
                    _f32_mul(np, a["nm"][k, j, i], prinv),
                )
                drive = np.maximum(drive, dmin).astype(np.float32)
                scale = _f32_mul(np, constants["csm"], grid)
                km = _f32_mul(np, _f32_mul(np, scale, scale), _f32_sqrt(np, drive))
                kh = _f32_mul(np, km, prinv)
                numerator = _f32_mul(np, _f32_mul(np, km, km), km)
                scale2 = _f32_mul(np, scale, scale)
                denominator = _f32_mul(np, scale2, scale2)
                dissipation = _f32_div(np, numerator, denominator)
                ratio = _f32_div(np, km, _f32_mul(np, constants["cme"], grid))
                tke = _f32_mul(np, ratio, ratio)
                out["grdscl"][k, j, i] = grid
                out["lenscl"][k, j, i] = grid
                out["kmh"][k, j, i] = km
                out["kmv"][k, j, i] = km
                out["khh"][k, j, i] = kh
                out["khv"][k, j, i] = kh
                out["dissten"][k, j, i] = dissipation
                out["tkea"][k, j, i] = tke
    out["tkea"][0] = out["tkea"][1]
    out["tkea"][nk].fill(np.float32(0.0))
    out["tke3d"][:] = out["tkea"]
    digest = hashlib.sha256()
    for name in OUTPUT_ARRAY_NAMES:
        value = out[name]
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"child turbsmag output {name} is non-finite")
        digest.update(value.astype("<f4", copy=False).tobytes(order="C"))
    return CpuChildTurbsmagInit(out, digest.hexdigest(), ni * nj * (nk - 1))


_CUDA_POW13_OPERAND_BITS = ",".join(
    f"0x{bits:08x}u" for bits, _ in GM1_CHILD_TURBSMAG_POW13_CORRECTIONS
)
_CUDA_POW13_ULP_DELTAS = ",".join(
    str(delta) for _, delta in GM1_CHILD_TURBSMAG_POW13_CORRECTIONS
)
_CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_TEMPLATE = r'''// gm1.cuda-child-turbsmag-init/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires binary32");
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __constant__ unsigned POW13_OPERANDS[77]={__POW13_OPERAND_BITS__};
__device__ __constant__ signed char POW13_DELTAS[77]={__POW13_ULP_DELTAS__};
__device__ __forceinline__ float cm1_pow13_selected(float x){float result=(float)pow((double)x,(double)0.33333333f);unsigned bits=__float_as_uint(x);int lo=0,hi=76;while(lo<=hi){int mid=(lo+hi)>>1;unsigned key=POW13_OPERANDS[mid];if(bits<key)hi=mid-1;else if(bits>key)lo=mid+1;else{unsigned out=__float_as_uint(result);int delta=(int)POW13_DELTAS[mid];out=delta>0?out+(unsigned)delta:out-(unsigned)(-delta);return __uint_as_float(out);}}return result;}
extern "C" __global__ void cm1_child_turbsmag_interior_v1(const float*ruh,const float*rvh,const float*rmf,const float*nm,const float*defv,const float*defh,float*grdscl,float*lenscl,float*dissten,float*kmh,float*kmv,float*khh,float*khv,float*tkea,unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nxy=(unsigned long long)ni*nj,words=nxy*(nk-1u);const float csm=0.17841240763664246f,cme=0.10f,prinv=3.0f,dmin=1.0e-10f;for(unsigned long long linear=q;linear<words;linear+=step){unsigned i=(unsigned)(linear%ni);unsigned long long jk=linear/ni;unsigned j=(unsigned)(jk%nj),k=(unsigned)(jk/nj)+1u;unsigned long long p=(unsigned long long)k*nxy+(unsigned long long)j*ni+i;float grid=M(dx,ruh[i]);grid=M(grid,M(dy,rvh[j]));grid=M(grid,M(dz,rmf[p]));grid=cm1_pow13_selected(grid);float drive=S(A(defv[p],defh[p]),M(nm[p],prinv));drive=fmaxf(drive,dmin);float scale=M(csm,grid);float km=M(M(scale,scale),sqrtf(drive));float kh=M(km,prinv);float num=M(M(km,km),km);float scale2=M(scale,scale);float den=M(scale2,scale2);float diss=D(num,den);float ratio=D(km,M(cme,grid));float tke=M(ratio,ratio);grdscl[p]=grid;lenscl[p]=grid;kmh[p]=km;kmv[p]=km;khh[p]=kh;khv[p]=kh;dissten[p]=diss;tkea[p]=tke;}}
extern "C" __global__ void cm1_child_turbsmag_faces_copy_v1(float*tkea,float*tke3d,float*tketen,unsigned ni,unsigned nj,unsigned nk){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,nxy=(unsigned long long)ni*nj,total=nxy*(nk+1u);for(unsigned long long col=q;col<nxy;col+=step){tkea[col]=tkea[nxy+col];tkea[(unsigned long long)nk*nxy+col]=0.0f;}for(unsigned long long p=q;p<total;p+=step){tke3d[p]=tkea[p];tketen[p]=0.0f;}}
extern "C" __global__ void cm1_child_turbsmag_pow13_probe_v1(const float*operand,float*result,unsigned long long words){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x;for(unsigned long long p=q;p<words;p+=step)result[p]=cm1_pow13_selected(operand[p]);}
'''
CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE = (
    _CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_TEMPLATE.replace(
        "__POW13_OPERAND_BITS__", _CUDA_POW13_OPERAND_BITS
    ).replace("__POW13_ULP_DELTAS__", _CUDA_POW13_ULP_DELTAS)
)
CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_BYTES = (
    CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE.encode("utf-8")
)
CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256 = (
    "8937435b68d61aa6c602b5f5c37796216e15a61e725b83cc0e4aab3af7eb5ac5"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(part, int) and not isinstance(part, bool) for part in value):
            return f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        major, minor = text.split(".", 1)
        text = major + minor
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


def _image_options(
    arch: str, nvrtc_major: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return common + (f"--gpu-architecture=compute_{arch}",), common + (
        f"--gpu-architecture=sm_{arch}",
    )


@dataclass(frozen=True, slots=True)
class CompiledChildTurbsmagInit:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_child_turbsmag_init() -> CompiledChildTurbsmagInit:
    correction_raw = "".join(
        f"{bits:08x}:{delta:+d}\n"
        for bits, delta in GM1_CHILD_TURBSMAG_POW13_CORRECTIONS
    ).encode("ascii")
    if (
        len(GM1_CHILD_TURBSMAG_POW13_CORRECTIONS) != 77
        or tuple(sorted(GM1_CHILD_TURBSMAG_POW13_CORRECTIONS))
        != GM1_CHILD_TURBSMAG_POW13_CORRECTIONS
        or len({bits for bits, _ in GM1_CHILD_TURBSMAG_POW13_CORRECTIONS}) != 77
        or any(delta not in (-1, 1) for _, delta in GM1_CHILD_TURBSMAG_POW13_CORRECTIONS)
        or hashlib.sha256(correction_raw).hexdigest()
        != GM1_CHILD_TURBSMAG_POW13_CORRECTIONS_SHA256
    ):
        raise RuntimeError("child turbsmag pow13 correction set differs")
    actual = hashlib.sha256(CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_BYTES).hexdigest()
    if actual != CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256:
        raise RuntimeError(f"child turbsmag CUDA source pin differs: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE,
        "gm1_child_turbsmag_init_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE,
        "gm1_child_turbsmag_init_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_GM1_CHILD_TURBSMAG_INIT_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks a child turbsmag entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled child turbsmag image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_GM1_CHILD_TURBSMAG_INIT_KERNEL_NAMES
    )
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledChildTurbsmagInit(
        module,
        kernels,
        ptx,
        cubin,
        arch,
        ptx_options,
        cubin_options,
        identity,
    )


@dataclass(frozen=True, slots=True)
class CudaChildTurbsmagPow13Launch:
    ready_event: Any
    result: Any
    compiled: CompiledChildTurbsmagInit
    receipt: Mapping[str, Any]


def launch_cuda_child_turbsmag_pow13_probe(
    operand: Any,
    *,
    metric_identity: ChildTurbsmagMetricIdentity,
    stream: Any | None = None,
    compiled: CompiledChildTurbsmagInit | None = None,
) -> CudaChildTurbsmagPow13Launch:
    """Launch the selected-path power helper over a sealed operand deck."""

    cupy = importlib.import_module("cupy")
    if (
        not isinstance(operand, cupy.ndarray)
        or operand.dtype != cupy.dtype("float32")
        or operand.ndim != 1
        or not operand.flags.c_contiguous
    ):
        raise ValueError("pow13 operand must be one-dimensional contiguous CUDA float32")
    require_child_turbsmag_metric_identity(
        metric_identity,
        ni=metric_identity.ni,
        nj=metric_identity.nj,
        nk=metric_identity.nk,
        dx=metric_identity.dx,
        dy=metric_identity.dy,
        dz=metric_identity.dz,
    )
    if int(operand.size) != metric_identity.word_count:
        raise ValueError(
            "pow13 operand count differs from the sealed metric identity"
        )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = compiled or compile_cuda_child_turbsmag_init()
    with stream:
        result = cupy.empty_like(operand)
        grid = (
            (int(operand.size) + CUDA_GM1_CHILD_TURBSMAG_INIT_THREADS - 1)
            // CUDA_GM1_CHILD_TURBSMAG_INIT_THREADS,
            1,
            1,
        )
        compiled.kernels[2](
            grid,
            (CUDA_GM1_CHILD_TURBSMAG_INIT_THREADS, 1, 1),
            (operand, result, cupy.uint64(int(operand.size))),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": "gm1.cuda-child-turbsmag-pow13-probe/v1",
        "metric_identity": {
            "name": metric_identity.name,
            "grid_operand_sha256": metric_identity.operand_sha256,
            "word_count": metric_identity.word_count,
            "fixture_class": metric_identity.fixture_class,
        },
        "arithmetic": {
            "native_lowering": "gfortran call to powf@GLIBC_2.27",
            "candidate": "CUDA double pow, one binary32 store, sealed ULP corrections",
            "correction_count": len(GM1_CHILD_TURBSMAG_POW13_CORRECTIONS),
            "corrections_sha256": GM1_CHILD_TURBSMAG_POW13_CORRECTIONS_SHA256,
            "comparison_contract": {
                "mode": "raw-bit",
                "max_ulp": 0,
                "null_or_missing_is_pass": False,
            },
            "claim": "selected-path only; not general pow or cube-root parity",
        },
        "kernel": {
            "source_sha256": CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256,
            "name": CUDA_GM1_CHILD_TURBSMAG_INIT_KERNEL_NAMES[2],
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaChildTurbsmagPow13Launch(event, result, compiled, receipt)


def _require_device_inputs(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = child_turbsmag_init_shapes(ni, nj, nk)
    if set(arrays) != set(INPUT_ARRAY_NAMES):
        raise ValueError("child turbsmag CUDA input inventory changed")
    result = {}
    for name in INPUT_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-contiguous CUDA float32 {shapes[name]}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("child turbsmag CUDA inputs must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaChildTurbsmagInitLaunch:
    ready_event: Any
    arrays: Mapping[str, Any]
    compiled: CompiledChildTurbsmagInit
    receipt: Mapping[str, Any]


def launch_cuda_child_turbsmag_init(
    inputs: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    metric_identity: ChildTurbsmagMetricIdentity,
    selection: ChildTurbsmagInitSelection = DEFAULT_CHILD_TURBSMAG_INIT_SELECTION,
    stream: Any | None = None,
    compiled: CompiledChildTurbsmagInit | None = None,
) -> CudaChildTurbsmagInitLaunch:
    """Launch the exact selected child closure initialization without a host sync."""

    require_child_turbsmag_init_selection(selection)
    _validate_grid(ni, nj, nk)
    dx, dy, dz = _spacing("dx", dx), _spacing("dy", dy), _spacing("dz", dz)
    require_child_turbsmag_metric_identity(
        metric_identity, ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz
    )
    cupy = importlib.import_module("cupy")
    a = _require_device_inputs(cupy, inputs, ni, nj, nk)
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = compiled or compile_cuda_child_turbsmag_init()
    shapes = child_turbsmag_init_shapes(ni, nj, nk)
    with stream:
        outputs = {
            name: cupy.zeros(shapes[name], dtype=cupy.float32)
            for name in OUTPUT_ARRAY_NAMES
        }
        interior_words = ni * nj * (nk - 1)
        total_words = ni * nj * (nk + 1)
        grid = (
            (max(interior_words, total_words) + CUDA_GM1_CHILD_TURBSMAG_INIT_THREADS - 1)
            // CUDA_GM1_CHILD_TURBSMAG_INIT_THREADS,
            1,
            1,
        )
        block = (CUDA_GM1_CHILD_TURBSMAG_INIT_THREADS, 1, 1)
        compiled.kernels[0](
            grid,
            block,
            tuple(a[name] for name in INPUT_ARRAY_NAMES)
            + tuple(outputs[name] for name in ("grdscl", "lenscl", "dissten", "kmh", "kmv", "khh", "khv", "tkea"))
            + (
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
                cupy.float32(dx),
                cupy.float32(dy),
                cupy.float32(dz),
            ),
            stream=stream,
        )
        compiled.kernels[1](
            grid,
            block,
            (
                outputs["tkea"],
                outputs["tke3d"],
                outputs["tketen"],
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GM1_CHILD_TURBSMAG_INIT_SCHEMA,
        "abi": CUDA_GM1_CHILD_TURBSMAG_INIT_ABI,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
        },
        "source_anchors": dict(SOURCE_ANCHORS),
        "selection": {
            name: getattr(selection, name) for name in selection.__dataclass_fields__
        },
        "shape_zyx": [nk, nj, ni],
        "spacing_m": {"dx": dx, "dy": dy, "dz": dz},
        "metric_identity": {
            "name": metric_identity.name,
            "grid_operand_sha256": metric_identity.operand_sha256,
            "word_count": metric_identity.word_count,
            "fixture_class": metric_identity.fixture_class,
            "pow13_correction_count": len(GM1_CHILD_TURBSMAG_POW13_CORRECTIONS),
            "pow13_corrections_sha256": (
                GM1_CHILD_TURBSMAG_POW13_CORRECTIONS_SHA256
            ),
        },
        "kernel": {
            "source_sha256": CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "names": list(CUDA_GM1_CHILD_TURBSMAG_INIT_KERNEL_NAMES),
        },
        "runtime": dict(compiled.runtime_identity),
        "tkea_parent_disposition": "not interpolated; diagnosed on new grid",
        "claim_limit": CUDA_GM1_CHILD_TURBSMAG_INIT_CLAIM_LIMIT,
    }
    return CudaChildTurbsmagInitLaunch(event, outputs, compiled, receipt)


def compare_child_turbsmag_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    fields = {}
    for name in OUTPUT_ARRAY_NAMES:
        left, right = expected.get(name), candidate.get(name)
        exact = bool(
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == np.dtype("float32")
            and right.dtype == np.dtype("float32")
            and left.shape == right.shape
            and np.array_equal(left.view(np.uint32), right.view(np.uint32))
        )
        fields[name] = {
            "exact": exact,
            "expected_sha256": hashlib.sha256(
                left.astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest()
            if isinstance(left, np.ndarray)
            else None,
            "candidate_sha256": hashlib.sha256(
                right.astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest()
            if isinstance(right, np.ndarray)
            else None,
        }
    return {
        "mode": "raw-bit",
        "max_ulp": 0,
        "null_or_missing_is_pass": False,
        "fields": fields,
        "passed": all(item["exact"] for item in fields.values()),
    }


__all__ = (
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "COMPACT_CHILD_TURBSMAG_METRIC_IDENTITY",
    "CUDA_GM1_CHILD_TURBSMAG_INIT_ABI",
    "CUDA_GM1_CHILD_TURBSMAG_INIT_CLAIM_LIMIT",
    "CUDA_GM1_CHILD_TURBSMAG_INIT_KERNEL_NAMES",
    "CUDA_GM1_CHILD_TURBSMAG_INIT_SCHEMA",
    "CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_BYTES",
    "CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256",
    "ChildTurbsmagInitSelection",
    "ChildTurbsmagMetricIdentity",
    "CompiledChildTurbsmagInit",
    "CpuChildTurbsmagInit",
    "CudaChildTurbsmagInitLaunch",
    "CudaChildTurbsmagPow13Launch",
    "DEFAULT_CHILD_TURBSMAG_INIT_SELECTION",
    "FACE_OUTPUT_NAMES",
    "INPUT_ARRAY_NAMES",
    "GM1_CHILD_TURBSMAG_POW13_CORRECTIONS",
    "GM1_CHILD_TURBSMAG_POW13_CORRECTIONS_SHA256",
    "OUTPUT_ARRAY_NAMES",
    "REAL_CHILD_TURBSMAG_METRIC_IDENTITY",
    "REAL_CHILD_TURBSMAG_RMF_SHA256",
    "REAL_CHILD_TURBSMAG_STATIC_COMBINED_SHA256",
    "REAL_CHILD_TURBSMAG_STATIC_VALIDATION_SHA256",
    "child_turbsmag_constants_f32",
    "child_turbsmag_init_shapes",
    "compare_child_turbsmag_raw",
    "compile_cuda_child_turbsmag_init",
    "cpu_child_turbsmag_init",
    "launch_cuda_child_turbsmag_init",
    "launch_cuda_child_turbsmag_pow13_probe",
    "require_child_turbsmag_metric_identity",
    "require_child_turbsmag_init_selection",
)
