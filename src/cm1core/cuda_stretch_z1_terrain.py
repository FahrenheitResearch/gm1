"""Fixed-terrain composition for the sealed 15 km CM1 ``stretch_z=1`` child.

The vertical profiles are the nominal terrain-free coordinate built by
``param.F``.  CM1 then applies ``init_terrain`` to those profiles and finally
rebuilds the physical three-dimensional vertical metrics in ``param.F``.
This module preserves that source ordering and admits only the serial,
four-open, fixed-domain 300 x 300 x 75 child configuration.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .cuda_sound3_periodic import _canonical_arch, _image_options
from .cuda_stretch_z1 import CudaStretchZ1Launch, launch_cuda_stretch_z1
from .errors import refuse
from .stretch_z1_child_15km import (
    STRETCH_Z1_CHILD_15KM_GRID_SHA256,
    STRETCH_Z1_CHILD_15KM_PARAMETERS,
    STRETCH_Z1_CHILD_15KM_PROOF_SHA256,
    build_stretch_z1_child_15km_grid,
    build_stretch_z1_child_15km_proof,
)


AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/param.F": "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
    "src/init_terrain.F": "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
    "src/adv_routines.F": "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
    "src/bc.F": "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
}
SOURCE_ANCHORS = {
    "fixed_terrain_moving_domain_refusal": "src/param.F:1890-1904",
    "nominal_stretch_z1_profiles": "src/param.F:7814-7953,8162-8225",
    "terrain_dispatch": "src/param.F:8283-8324",
    "terrain_coordinates_and_metrics": "src/init_terrain.F:160-297",
    "open_edge_bc2d": "src/bc.F:12-172",
    "terrain_gradient_dispatch": "src/adv_routines.F:9255-9322",
    "flx6": "src/adv_routines.F:6025-6037",
    "physical_vertical_metrics": "src/param.F:8464-8492",
    "physical_face_interpolation": "src/param.F:8603-8627",
}

CUDA_STRETCH_Z1_TERRAIN_SCHEMA = "gpuwm-cm1.cuda-stretch-z1-terrain/v1"
CUDA_STRETCH_Z1_TERRAIN_ABI = (
    "gpuwm-cm1.cuda-stretch-z1-terrain-physical-carriers-f32/v1"
)
CM1_STRETCH_Z1_TERRAIN_CLAIM_LIMIT = (
    "CM1 r21.1 fixed terrain + Wilhelmson-Chen stretch_z=1 static metric "
    "composition for the sealed 300x300x75, 50 m, 15 km-top child; serial "
    "four-open bc2d, hadvordrs=hadvordrv=5, imove=0, axisymm=0, and no "
    "immersed boundary only. This does not admit terrain-varying hydrostatic "
    "base state, parent-to-child remap, pressure integration, time evolution, "
    "adaptive motion, nesting feedback, forecast skill, storm, or tornado behavior."
)

PHYSICAL_ARRAY_NAMES = (
    "terrain_msl",
    "zh",
    "zf",
    "mh",
    "rmh",
    "mf",
    "rmf",
    "c1",
    "c2",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "gx",
    "gxu",
    "gy",
    "gyv",
    "rds",
    "rdsf",
)


@dataclass(frozen=True, slots=True)
class StretchZ1TerrainSelection:
    terrain_flag: bool = True
    stretch_z: int = 1
    stretch_x: int = 0
    stretch_y: int = 0
    axisymm: int = 0
    imove: int = 0
    serial: bool = True
    lateral_bc: tuple[int, int, int, int] = (2, 2, 2, 2)
    hadvordrs: int = 5
    hadvordrv: int = 5
    do_ib: bool = False


def require_stretch_z1_terrain_selection(
    selection: StretchZ1TerrainSelection,
) -> None:
    if not isinstance(selection, StretchZ1TerrainSelection):
        raise TypeError("selection must be StretchZ1TerrainSelection")
    checks = (
        ("terrain_flag", selection.terrain_flag, True, "ARWEN-STRETCH-Z1-TERRAIN-OFF"),
        ("stretch_z", selection.stretch_z, 1, "ARWEN-STRETCH-Z1-TERRAIN-Z"),
        ("stretch_x", selection.stretch_x, 0, "ARWEN-STRETCH-Z1-TERRAIN-X"),
        ("stretch_y", selection.stretch_y, 0, "ARWEN-STRETCH-Z1-TERRAIN-Y"),
        ("axisymm", selection.axisymm, 0, "ARWEN-STRETCH-Z1-TERRAIN-AXISYMM"),
        ("imove", selection.imove, 0, "ARWEN-STRETCH-Z1-TERRAIN-IMOVE"),
        ("serial", selection.serial, True, "ARWEN-STRETCH-Z1-TERRAIN-MPI"),
        ("lateral_bc", selection.lateral_bc, (2, 2, 2, 2), "ARWEN-STRETCH-Z1-TERRAIN-BC"),
        ("hadvordrs", selection.hadvordrs, 5, "ARWEN-STRETCH-Z1-TERRAIN-HADV-S"),
        ("hadvordrv", selection.hadvordrv, 5, "ARWEN-STRETCH-Z1-TERRAIN-HADV-V"),
        ("do_ib", selection.do_ib, False, "ARWEN-STRETCH-Z1-TERRAIN-IB"),
    )
    for option, value, expected, code in checks:
        if value != expected:
            reason = (
                "CM1 explicitly forbids terrain with imove!=0"
                if option == "imove"
                else "the bounded metric composition selects different source code for this value"
            )
            refuse(
                code=code,
                gate="stretch-z1-terrain-selection",
                option=option,
                value=value,
                reason=reason,
                remedy=f"use {option}={expected!r} or add an independent native oracle",
            )


@dataclass(frozen=True, slots=True)
class CpuStretchZ1Terrain:
    arrays: dict[str, Any]
    combined_sha256: str
    vertical_grid_sha256: str
    vertical_proof_sha256: str


@dataclass(frozen=True, slots=True)
class DeviceStretchZ1Terrain:
    arrays: dict[str, Any]
    vertical: CudaStretchZ1Launch


@dataclass(frozen=True, slots=True)
class CompiledStretchZ1Terrain:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CudaStretchZ1TerrainLaunch:
    ready_event: Any
    compiled: CompiledStretchZ1Terrain
    state: DeviceStretchZ1Terrain
    receipt: dict[str, Any]


def stretch_z1_terrain_shapes(
    ni: int, nj: int, nk: int = STRETCH_Z1_CHILD_15KM_PARAMETERS.nk
) -> dict[str, tuple[int, ...]]:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    mass = (nk, nj, ni)
    zface = (nk + 1, nj, ni)
    return {
        "terrain_msl": (nj, ni),
        "zh": mass,
        "zf": zface,
        "mh": mass,
        "rmh": mass,
        "mf": zface,
        "rmf": zface,
        "c1": zface,
        "c2": zface,
        "gz": (nj, ni),
        "rgz": (nj, ni),
        "gzu": (nj, ni + 1),
        "rgzu": (nj, ni + 1),
        "gzv": (nj + 1, ni),
        "rgzv": (nj + 1, ni),
        "dzdx": (nj, ni),
        "dzdy": (nj, ni),
        "gx": zface,
        "gxu": (nk + 1, nj, ni + 1),
        "gy": zface,
        "gyv": (nk + 1, nj + 1, ni),
        "rds": (nk + 2,),
        "rdsf": (nk + 3,),
    }


def _add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _flx6(np: Any, s1: Any, s2: Any, s3: Any, s4: Any, s5: Any, s6: Any) -> Any:
    a = _mul(np, _div(np, 37.0, 60.0), _add(np, s4, s3))
    b = _mul(np, _div(np, -8.0, 60.0), _add(np, s5, s2))
    c = _mul(np, _div(np, 1.0, 60.0), _add(np, s6, s1))
    return _add(np, _add(np, a, b), c)


def _profile_sha256(np: Any, value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype=np.float32)
        .astype("<f4", copy=False)
        .tobytes(order="C")
    ).hexdigest()


def _validate_host_terrain(
    np: Any,
    terrain_msl: Any,
    *,
    allow_compact_validation: bool,
) -> tuple[int, int]:
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    if (
        not isinstance(terrain_msl, np.ndarray)
        or terrain_msl.dtype != np.dtype("float32")
        or terrain_msl.ndim != 2
        or not terrain_msl.flags.c_contiguous
    ):
        raise ValueError("terrain_msl must be C-contiguous numpy float32 [nj,ni]")
    nj, ni = map(int, terrain_msl.shape)
    if ni < 1 or nj < 1:
        raise ValueError("terrain_msl extents must be positive")
    if not allow_compact_validation and (nj, ni) != (300, 300):
        refuse(
            code="ARWEN-STRETCH-Z1-TERRAIN-DOMAIN",
            gate="stretch-z1-terrain-geometry",
            option="shape_yx",
            value=(nj, ni),
            reason="a different horizontal extent would not be the sealed 15 km 50 m child",
            remedy="use shape_yx=(300,300) or mark a compact analytic operator validation",
        )
    finite = np.isfinite(terrain_msl)
    if not bool(np.all(finite)):
        index = int(np.flatnonzero(~finite)[0])
        raise ValueError(f"terrain_msl word {index} is non-finite")
    top = np.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.ztop)
    invalid = terrain_msl >= top
    if bool(np.any(invalid)):
        index = int(np.flatnonzero(invalid)[0])
        bits = int(terrain_msl.reshape(-1).view(np.uint32)[index])
        raise ValueError(
            f"terrain_msl word {index} reaches the 15 km model top; bits={bits:08x}"
        )
    return ni, nj


def cpu_build_stretch_z1_terrain(
    terrain_msl: Any,
    *,
    selection: StretchZ1TerrainSelection = StretchZ1TerrainSelection(),
    allow_compact_validation: bool = False,
) -> CpuStretchZ1Terrain:
    """Build source-ordered physical static metrics using binary32 operations."""

    require_stretch_z1_terrain_selection(selection)
    np = importlib.import_module("numpy")
    ni, nj = _validate_host_terrain(
        np, terrain_msl, allow_compact_validation=allow_compact_validation
    )
    vertical = build_stretch_z1_child_15km_grid()
    proof = build_stretch_z1_child_15km_proof()
    nk = STRETCH_Z1_CHILD_15KM_PARAMETERS.nk
    arrays = {
        name: np.empty(shape, dtype=np.float32)
        for name, shape in stretch_z1_terrain_shapes(ni, nj, nk).items()
    }
    arrays["terrain_msl"][:] = terrain_msl
    top = np.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.ztop)
    dz = np.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.dz)
    rdx = _div(np, 1.0, 50.0)
    rdy = _div(np, 1.0, 50.0)

    def zs_at(i: int, j: int) -> Any:
        return terrain_msl[min(max(j, 0), nj - 1), min(max(i, 0), ni - 1)]

    def terrain_height(surface: Any, sigma_value: Any) -> Any:
        return _add(
            np,
            surface,
            _div(np, _mul(np, sigma_value, _sub(np, top, surface)), top),
        )

    def rgz_at(i: int, j: int) -> Any:
        return _div(np, _sub(np, top, zs_at(i, j)), top)

    def rgzu_at(face_i: int, j: int) -> Any:
        average = _mul(np, 0.5, _add(np, zs_at(face_i - 1, j), zs_at(face_i, j)))
        return _div(np, _sub(np, top, average), top)

    def rgzv_at(i: int, face_j: int) -> Any:
        average = _mul(np, 0.5, _add(np, zs_at(i, face_j - 1), zs_at(i, face_j)))
        return _div(np, _sub(np, top, average), top)

    def x_flux(face_i: int, j: int) -> Any:
        return _flx6(np, *(zs_at(face_i + offset, j) for offset in (-3, -2, -1, 0, 1, 2)))

    def y_flux(i: int, face_j: int) -> Any:
        return _flx6(np, *(zs_at(i, face_j + offset) for offset in (-3, -2, -1, 0, 1, 2)))

    for j in range(nj):
        for i in range(ni):
            surface = terrain_msl[j, i]
            arrays["gz"][j, i] = _div(np, top, _sub(np, top, surface))
            arrays["rgz"][j, i] = _div(np, _sub(np, top, surface), top)
            arrays["dzdx"][j, i] = _mul(np, _sub(np, x_flux(i + 1, j), x_flux(i, j)), rdx)
            arrays["dzdy"][j, i] = _mul(np, _sub(np, y_flux(i, j + 1), y_flux(i, j)), rdy)
        for face_i in range(ni + 1):
            average = _mul(np, 0.5, _add(np, zs_at(face_i - 1, j), zs_at(face_i, j)))
            arrays["gzu"][j, face_i] = _div(np, top, _sub(np, top, average))
            arrays["rgzu"][j, face_i] = _div(np, _sub(np, top, average), top)
    for face_j in range(nj + 1):
        for i in range(ni):
            average = _mul(np, 0.5, _add(np, zs_at(i, face_j - 1), zs_at(i, face_j)))
            arrays["gzv"][face_j, i] = _div(np, top, _sub(np, top, average))
            arrays["rgzv"][face_j, i] = _div(np, _sub(np, top, average), top)

    for j in range(nj):
        for i in range(ni):
            surface = terrain_msl[j, i]
            for k in range(nk + 1):
                arrays["zf"][k, j, i] = terrain_height(surface, vertical.sigmaf[k + 1])
            for k in range(nk):
                arrays["zh"][k, j, i] = terrain_height(surface, vertical.sigma[k + 1])
                spacing = _sub(np, arrays["zf"][k + 1, j, i], arrays["zf"][k, j, i])
                arrays["mh"][k, j, i] = _div(np, dz, spacing)
                arrays["rmh"][k, j, i] = _div(np, 1.0, arrays["mh"][k, j, i])
            zf_bottom = _sub(
                np,
                arrays["zf"][0, j, i],
                _sub(np, arrays["zf"][1, j, i], arrays["zf"][0, j, i]),
            )
            zf_top = _add(
                np,
                arrays["zf"][nk, j, i],
                _sub(np, arrays["zf"][nk, j, i], arrays["zf"][nk - 1, j, i]),
            )
            zh_bottom = _mul(np, 0.5, _add(np, zf_bottom, arrays["zf"][0, j, i]))
            zh_top = _mul(np, 0.5, _add(np, arrays["zf"][nk, j, i], zf_top))
            for face in range(nk + 1):
                lower = zh_bottom if face == 0 else arrays["zh"][face - 1, j, i]
                upper = zh_top if face == nk else arrays["zh"][face, j, i]
                span = _sub(np, upper, lower)
                arrays["mf"][face, j, i] = _div(np, dz, span)
                arrays["rmf"][face, j, i] = _div(np, 1.0, arrays["mf"][face, j, i])
                arrays["c2"][face, j, i] = _div(
                    np, _sub(np, arrays["zf"][face, j, i], lower), span
                )
                arrays["c1"][face, j, i] = _sub(np, 1.0, arrays["c2"][face, j, i])

    for k in range(nk + 1):
        height_factor = _sub(np, top, vertical.sigmaf[k + 1])
        for j in range(nj):
            for i in range(ni):
                value = _mul(np, height_factor, arrays["gz"][j, i])
                value = _mul(np, value, _sub(np, arrays["rgzu"][j, i + 1], arrays["rgzu"][j, i]))
                arrays["gx"][k, j, i] = _mul(np, value, rdx)
                value = _mul(np, height_factor, arrays["gz"][j, i])
                value = _mul(np, value, _sub(np, arrays["rgzv"][j + 1, i], arrays["rgzv"][j, i]))
                arrays["gy"][k, j, i] = _mul(np, value, rdy)
            for face_i in range(ni + 1):
                value = _mul(np, height_factor, arrays["gzu"][j, face_i])
                value = _mul(np, value, _sub(np, rgz_at(face_i, j), rgz_at(face_i - 1, j)))
                arrays["gxu"][k, j, face_i] = _mul(np, value, rdx)
        for face_j in range(nj + 1):
            for i in range(ni):
                value = _mul(np, height_factor, arrays["gzv"][face_j, i])
                value = _mul(np, value, _sub(np, rgz_at(i, face_j), rgz_at(i, face_j - 1)))
                arrays["gyv"][k, face_j, i] = _mul(np, value, rdy)

    arrays["rds"].fill(np.float32(1.0))
    arrays["rdsf"].fill(np.float32(1.0))
    for k in range(2, nk + 1):
        arrays["rds"][k] = _div(np, 1.0, _sub(np, vertical.sigma[k], vertical.sigma[k - 1]))
    for k in range(1, nk + 1):
        arrays["rdsf"][k] = _div(np, 1.0, _sub(np, vertical.sigmaf[k + 1], vertical.sigmaf[k]))

    for name, shape in stretch_z1_terrain_shapes(ni, nj, nk).items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.dtype("float32") or not value.flags.c_contiguous:
            raise RuntimeError(f"source-ordered terrain metric layout failed for {name}")
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"source-ordered terrain metric {name} is non-finite")
    digest = hashlib.sha256()
    for name in PHYSICAL_ARRAY_NAMES:
        digest.update(arrays[name].astype("<f4", copy=False).tobytes(order="C"))
    return CpuStretchZ1Terrain(
        arrays,
        digest.hexdigest(),
        vertical.sha256,
        proof.proof_sha256,
    )


def compare_stretch_z1_terrain_raw(
    expected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    fields: dict[str, Any] = {}
    for name in PHYSICAL_ARRAY_NAMES:
        if name not in expected or name not in candidate:
            raise ValueError(f"missing terrain+stretch field {name}")
        left, right = expected[name], candidate[name]
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
            "expected_sha256": _profile_sha256(np, left),
            "candidate_sha256": _profile_sha256(np, right),
        }
    return {
        "fields": fields,
        "max_ulp": 0,
        "mode": "raw-bit",
        "null_or_missing_is_pass": False,
        "passed": all(item["exact"] for item in fields.values()),
    }


CUDA_STRETCH_Z1_TERRAIN_KERNEL_NAMES = (
    "cm1_szt_surface_v1",
    "cm1_szt_columns_v1",
    "cm1_szt_cross_v1",
    "cm1_szt_rds_v1",
)

CUDA_STRETCH_Z1_TERRAIN_SOURCE = r'''// gpuwm-cm1.cuda-stretch-z1-terrain/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long I3(unsigned k,unsigned j,unsigned i,unsigned ny,unsigned nx){return ((unsigned long long)k*ny+j)*nx+i;}__device__ __forceinline__ int CL(int v,int hi){return v<0?0:(v>hi?hi:v);}__device__ __forceinline__ float Z(const float*zs,int i,int j,unsigned ni,unsigned nj){return zs[(unsigned long long)CL(j,(int)nj-1)*ni+(unsigned)CL(i,(int)ni-1)];}
__device__ __forceinline__ float F6(float s1,float s2,float s3,float s4,float s5,float s6){float a=M(D(37.0f,60.0f),A(s4,s3)),b=M(D(-8.0f,60.0f),A(s5,s2)),c=M(D(1.0f,60.0f),A(s6,s1));return A(A(a,b),c);}__device__ __forceinline__ float XF(const float*z,int f,int j,unsigned ni,unsigned nj){return F6(Z(z,f-3,j,ni,nj),Z(z,f-2,j,ni,nj),Z(z,f-1,j,ni,nj),Z(z,f,j,ni,nj),Z(z,f+1,j,ni,nj),Z(z,f+2,j,ni,nj));}__device__ __forceinline__ float YF(const float*z,int i,int f,unsigned ni,unsigned nj){return F6(Z(z,i,f-3,ni,nj),Z(z,i,f-2,ni,nj),Z(z,i,f-1,ni,nj),Z(z,i,f,ni,nj),Z(z,i,f+1,ni,nj),Z(z,i,f+2,ni,nj));}
__device__ __forceinline__ float TH(float s,float q,float top){return A(s,D(M(q,S(top,s)),top));}__device__ __forceinline__ float RG(const float*z,int i,int j,unsigned ni,unsigned nj,float top){return D(S(top,Z(z,i,j,ni,nj)),top);}
extern "C" __global__ void cm1_szt_surface_v1(const float*zs,float*gz,float*rgz,float*gzu,float*rgzu,float*gzv,float*rgzv,float*dzdx,float*dzdy,unsigned ni,unsigned nj,float top,float rdx,float rdy){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj,nu=(unsigned long long)(ni+1u)*nj,nv=(unsigned long long)ni*(nj+1u);for(unsigned long long p=q;p<n;p+=step){unsigned i=(unsigned)(p%ni),j=(unsigned)(p/ni);float s=zs[p];gz[p]=D(top,S(top,s));rgz[p]=D(S(top,s),top);dzdx[p]=M(S(XF(zs,(int)i+1,(int)j,ni,nj),XF(zs,(int)i,(int)j,ni,nj)),rdx);dzdy[p]=M(S(YF(zs,(int)i,(int)j+1,ni,nj),YF(zs,(int)i,(int)j,ni,nj)),rdy);}for(unsigned long long p=q;p<nu;p+=step){unsigned f=(unsigned)(p%(ni+1u)),j=(unsigned)(p/(ni+1u));float s=M(0.5f,A(Z(zs,(int)f-1,(int)j,ni,nj),Z(zs,(int)f,(int)j,ni,nj)));gzu[p]=D(top,S(top,s));rgzu[p]=D(S(top,s),top);}for(unsigned long long p=q;p<nv;p+=step){unsigned i=(unsigned)(p%ni),f=(unsigned)(p/ni);float s=M(0.5f,A(Z(zs,(int)i,(int)f-1,ni,nj),Z(zs,(int)i,(int)f,ni,nj)));gzv[p]=D(top,S(top,s));rgzv[p]=D(S(top,s),top);}}
extern "C" __global__ void cm1_szt_columns_v1(const float*zs,const float*sigma,const float*sigmaf,float*zh,float*zf,float*mh,float*rmh,float*mf,float*rmf,float*c1,float*c2,unsigned ni,unsigned nj,unsigned nk,float top,float dz){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj;for(unsigned long long col=q;col<n;col+=step){float s=zs[col];for(unsigned k=0;k<=nk;++k)zf[(unsigned long long)k*n+col]=TH(s,sigmaf[k+1u],top);for(unsigned k=0;k<nk;++k){unsigned long long p=(unsigned long long)k*n+col;zh[p]=TH(s,sigma[k+1u],top);float h=D(dz,S(zf[p+n],zf[p]));mh[p]=h;rmh[p]=D(1.0f,h);}float z0=zf[col],z1=zf[n+col],zb=S(z0,S(z1,z0)),zt=zf[(unsigned long long)nk*n+col],ztm=zf[(unsigned long long)(nk-1u)*n+col],za=A(zt,S(zt,ztm)),hb=M(0.5f,A(zb,z0)),ht=M(0.5f,A(zt,za));for(unsigned k=0;k<=nk;++k){float lo=k==0u?hb:zh[(unsigned long long)(k-1u)*n+col],hi=k==nk?ht:zh[(unsigned long long)k*n+col],span=S(hi,lo);unsigned long long p=(unsigned long long)k*n+col;float f=D(dz,span);mf[p]=f;rmf[p]=D(1.0f,f);c2[p]=D(S(zf[p],lo),span);c1[p]=S(1.0f,c2[p]);}}}
extern "C" __global__ void cm1_szt_cross_v1(const float*zs,const float*sigmaf,const float*gz,const float*rgz,const float*gzu,const float*rgzu,const float*gzv,const float*rgzv,float*gx,float*gxu,float*gy,float*gyv,unsigned ni,unsigned nj,unsigned nk,float top,float rdx,float rdy){unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)(nk+1u)*nj*ni,nu=(unsigned long long)(nk+1u)*nj*(ni+1u),nv=(unsigned long long)(nk+1u)*(nj+1u)*ni;for(unsigned long long p=q;p<n;p+=step){unsigned i=(unsigned)(p%ni),j=(unsigned)((p/ni)%nj),k=(unsigned)(p/((unsigned long long)ni*nj));float h=S(top,sigmaf[k+1u]),v=M(h,gz[(unsigned long long)j*ni+i]);gx[p]=M(M(v,S(rgzu[(unsigned long long)j*(ni+1u)+i+1u],rgzu[(unsigned long long)j*(ni+1u)+i])),rdx);gy[p]=M(M(v,S(rgzv[(unsigned long long)(j+1u)*ni+i],rgzv[(unsigned long long)j*ni+i])),rdy);}for(unsigned long long p=q;p<nu;p+=step){unsigned f=(unsigned)(p%(ni+1u)),j=(unsigned)((p/(ni+1u))%nj),k=(unsigned)(p/((unsigned long long)(ni+1u)*nj));float v=M(S(top,sigmaf[k+1u]),gzu[(unsigned long long)j*(ni+1u)+f]);gxu[p]=M(M(v,S(RG(zs,(int)f,(int)j,ni,nj,top),RG(zs,(int)f-1,(int)j,ni,nj,top))),rdx);}for(unsigned long long p=q;p<nv;p+=step){unsigned i=(unsigned)(p%ni),f=(unsigned)((p/ni)%(nj+1u)),k=(unsigned)(p/((unsigned long long)ni*(nj+1u)));float v=M(S(top,sigmaf[k+1u]),gzv[(unsigned long long)f*ni+i]);gyv[p]=M(M(v,S(RG(zs,(int)i,(int)f,ni,nj,top),RG(zs,(int)i,(int)f-1,ni,nj,top))),rdy);}}
extern "C" __global__ void cm1_szt_rds_v1(const float*sigma,const float*sigmaf,float*rds,float*rdsf,unsigned nk){if(blockIdx.x||threadIdx.x)return;for(unsigned k=0;k<nk+2u;++k)rds[k]=1.0f;for(unsigned k=0;k<nk+3u;++k)rdsf[k]=1.0f;for(unsigned k=2u;k<=nk;++k)rds[k]=D(1.0f,S(sigma[k],sigma[k-1u]));for(unsigned k=1u;k<=nk;++k)rdsf[k]=D(1.0f,S(sigmaf[k+1u],sigmaf[k]));}
'''
CUDA_STRETCH_Z1_TERRAIN_SOURCE_BYTES = CUDA_STRETCH_Z1_TERRAIN_SOURCE.encode("utf-8")
CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256 = (
    "905bec56020e2bf1a73c0c854a9d5f462ccc372334ab3ceee2ad50f041be2f18"
)


def compile_cuda_stretch_z1_terrain() -> CompiledStretchZ1Terrain:
    actual = hashlib.sha256(CUDA_STRETCH_Z1_TERRAIN_SOURCE_BYTES).hexdigest()
    if actual != CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256:
        raise RuntimeError(
            f"packaged stretch_z1+terrain CUDA source differs from pin: {actual}"
        )
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_STRETCH_Z1_TERRAIN_SOURCE,
        "cm1_stretch_z1_terrain_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_STRETCH_Z1_TERRAIN_SOURCE,
        "cm1_stretch_z1_terrain_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", text)
        for name in CUDA_STRETCH_Z1_TERRAIN_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks stretch_z1+terrain entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01":
        raise ValueError("compiled stretch_z1+terrain image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_STRETCH_Z1_TERRAIN_KERNEL_NAMES)
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
    return CompiledStretchZ1Terrain(module, kernels, ptx, cubin, arch, identity)


def launch_cuda_stretch_z1_terrain(
    terrain_msl: Any,
    *,
    selection: StretchZ1TerrainSelection = StretchZ1TerrainSelection(),
    allow_compact_validation: bool = False,
    stream: Any | None = None,
    compiled: CompiledStretchZ1Terrain | None = None,
    vertical: CudaStretchZ1Launch | None = None,
) -> CudaStretchZ1TerrainLaunch:
    """Compose fixed terrain with the sealed child grid without host round trips."""

    require_stretch_z1_terrain_selection(selection)
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    cupy = importlib.import_module("cupy")
    if (
        not isinstance(terrain_msl, cupy.ndarray)
        or terrain_msl.dtype != cupy.dtype("float32")
        or terrain_msl.ndim != 2
        or not terrain_msl.flags.c_contiguous
    ):
        raise ValueError("terrain_msl must be C-contiguous CUDA float32 [nj,ni]")
    nj, ni = map(int, terrain_msl.shape)
    if not allow_compact_validation and (nj, ni) != (300, 300):
        refuse(
            code="ARWEN-STRETCH-Z1-TERRAIN-DOMAIN",
            gate="stretch-z1-terrain-geometry",
            option="shape_yx",
            value=(nj, ni),
            reason="a different horizontal extent would not be the sealed 15 km 50 m child",
            remedy="use shape_yx=(300,300) or mark a compact analytic operator validation",
        )
    if int(cupy.count_nonzero(~cupy.isfinite(terrain_msl)).get()) != 0:
        raise ValueError("terrain_msl contains a non-finite CUDA word")
    top = cupy.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.ztop)
    if bool((cupy.max(terrain_msl) >= top).get()):
        raise ValueError("terrain_msl reaches the 15 km model top")
    build_stretch_z1_child_15km_grid()
    proof = build_stretch_z1_child_15km_proof()
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    vertical = vertical or launch_cuda_stretch_z1(
        STRETCH_Z1_CHILD_15KM_PARAMETERS, stream=stream
    )
    if vertical.receipt["parameters"] != {
        name: getattr(STRETCH_Z1_CHILD_15KM_PARAMETERS, name)
        for name in STRETCH_Z1_CHILD_15KM_PARAMETERS.__dataclass_fields__
    }:
        raise ValueError("device vertical launch is not the sealed 15 km child grid")
    compiled = compiled or compile_cuda_stretch_z1_terrain()
    nk = STRETCH_Z1_CHILD_15KM_PARAMETERS.nk
    arrays = {
        name: cupy.empty(shape, dtype=cupy.float32)
        for name, shape in stretch_z1_terrain_shapes(ni, nj, nk).items()
        if name != "terrain_msl"
    }
    arrays["terrain_msl"] = terrain_msl
    surface, columns, cross, rds = compiled.kernels
    threads = (128, 1, 1)
    largest = max(value.size for value in arrays.values())
    grid = ((largest + 127) // 128, 1, 1)
    column_grid = (((ni * nj) + 127) // 128, 1, 1)
    u32 = cupy.uint32
    with stream:
        stream.wait_event(vertical.ready_event)
        surface(
            grid,
            threads,
            (
                terrain_msl,
                *(arrays[name] for name in ("gz", "rgz", "gzu", "rgzu", "gzv", "rgzv", "dzdx", "dzdy")),
                u32(ni),
                u32(nj),
                top,
                cupy.float32(1.0 / 50.0),
                cupy.float32(1.0 / 50.0),
            ),
            stream=stream,
        )
        columns(
            column_grid,
            threads,
            (
                terrain_msl,
                vertical.grid.sigma,
                vertical.grid.sigmaf,
                *(arrays[name] for name in ("zh", "zf", "mh", "rmh", "mf", "rmf", "c1", "c2")),
                u32(ni),
                u32(nj),
                u32(nk),
                top,
                cupy.float32(STRETCH_Z1_CHILD_15KM_PARAMETERS.dz),
            ),
            stream=stream,
        )
        cross(
            grid,
            threads,
            (
                terrain_msl,
                vertical.grid.sigmaf,
                *(arrays[name] for name in ("gz", "rgz", "gzu", "rgzu", "gzv", "rgzv", "gx", "gxu", "gy", "gyv")),
                u32(ni),
                u32(nj),
                u32(nk),
                top,
                cupy.float32(1.0 / 50.0),
                cupy.float32(1.0 / 50.0),
            ),
            stream=stream,
        )
        rds(
            (1, 1, 1),
            (1, 1, 1),
            (vertical.grid.sigma, vertical.grid.sigmaf, arrays["rds"], arrays["rdsf"], u32(nk)),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_STRETCH_Z1_TERRAIN_SCHEMA,
        "abi": CUDA_STRETCH_Z1_TERRAIN_ABI,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
        },
        "source_anchors": dict(SOURCE_ANCHORS),
        "selection": {
            name: getattr(selection, name) for name in selection.__dataclass_fields__
        },
        "geometry": {
            "shape_zyx": [nk, nj, ni],
            "dx_m": 50.0,
            "dy_m": 50.0,
            "ztop_m": 15000.0,
            "compact_validation": allow_compact_validation,
        },
        "vertical": {
            "grid_sha256": STRETCH_Z1_CHILD_15KM_GRID_SHA256,
            "proof_sha256": STRETCH_Z1_CHILD_15KM_PROOF_SHA256,
        },
        "kernel": {
            "source_sha256": CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
        "claim_limit": CM1_STRETCH_Z1_TERRAIN_CLAIM_LIMIT,
    }
    return CudaStretchZ1TerrainLaunch(
        event, compiled, DeviceStretchZ1Terrain(arrays, vertical), receipt
    )


__all__ = (
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_STRETCH_Z1_TERRAIN_CLAIM_LIMIT",
    "CUDA_STRETCH_Z1_TERRAIN_ABI",
    "CUDA_STRETCH_Z1_TERRAIN_KERNEL_NAMES",
    "CUDA_STRETCH_Z1_TERRAIN_SCHEMA",
    "CUDA_STRETCH_Z1_TERRAIN_SOURCE_BYTES",
    "CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256",
    "CudaStretchZ1TerrainLaunch",
    "DeviceStretchZ1Terrain",
    "PHYSICAL_ARRAY_NAMES",
    "SOURCE_ANCHORS",
    "StretchZ1TerrainSelection",
    "compare_stretch_z1_terrain_raw",
    "compile_cuda_stretch_z1_terrain",
    "cpu_build_stretch_z1_terrain",
    "launch_cuda_stretch_z1_terrain",
    "require_stretch_z1_terrain_selection",
    "stretch_z1_terrain_shapes",
)
