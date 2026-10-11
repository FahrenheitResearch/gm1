"""Live profile-aware ``calcnm``/``calcdef`` refresh for the GM1 child.

CM1 refreshes moist stability and deformation from the *current* model state
inside ``solve1`` before the local closure coefficients and held turbulence
tendencies are rebuilt.  The child ``calcnm`` kernel is already valid on a
stretched grid because the official expression is ``rdz*mf``.  ``calcdef`` is
not: its terrain-coordinate interpolations and derivatives consume the live
``sigma``, ``sigmaf``, ``rds``, and ``rdsf`` profiles.  This module preserves
the sealed operator statement order and replaces only those source-audited
profile expressions.

The public launch is deliberately a same-stream primitive.  It waits on the
sealed static-metric publication, enqueues ``calcnm`` then ``calcdef``, records
one completion event, and performs no host synchronization.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cm1core.cuda_gm1_child_nm_def import (
    CHILD_MOIST_INPUT_NAMES,
    CHILD_MOIST_OUTPUT_NAMES,
    CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
    CompiledChildMoistNmKernel,
    compile_cuda_child_moist_nm,
    launch_cuda_child_moist_nm,
)
from cm1core.cuda_gm1_child_stretched_metrics import (
    GM1_CHILD_STRETCHED_DZ_REFERENCE_F32,
    GM1_CHILD_STRETCHED_GRID_SHA256,
    GM1_CHILD_STRETCHED_METRIC_ABI,
    GM1_CHILD_STRETCHED_PROFILE_SHA256,
    GM1_CHILD_STRETCHED_PROOF_SHA256,
    GM1_CHILD_STRETCHED_ZTOP_F32,
    bind_gm1_child_stretched_metrics as _bind_sealed_stretched_metrics,
)
from cm1core.cuda_terrain_deformation import (
    CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES,
    CUDA_TERRAIN_DEFORMATION_SOURCE,
    CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
    CUDA_TERRAIN_DEFORMATION_THREADS,
    DEFAULT_TERRAIN_DEFORMATION_SELECTION,
    INPUT_NAMES as CALCDEF_INPUT_NAMES,
    TerrainDeformationSelection,
    _validate_grid,
    _validate_selection,
    terrain_deformation_scratch_shapes,
    terrain_deformation_shapes,
)
from cm1core.errors import refuse
from cm1core.nvrtc_module import compile_nvrtc_module


CUDA_GM1_CHILD_STRETCHED_NM_DEF_SCHEMA = "gm1.cuda-child-stretched-live-nm-def/v1"
CUDA_GM1_CHILD_STRETCHED_NM_DEF_ABI = "gm1.cuda-child-stretched-live-nm-def-kernels/v1"
CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256 = (
    "3114457a038061913bf760be0a3a6b24c1e56d753f249d77d224af59a7a987fd"
)
GM1_CHILD_STATIC_COMBINED_SHA256 = (
    "2ec8b6a5a48a81017527e66c4fa6b264f1476b737fa2b0a02f7795012877c0a9"
)
GM1_CHILD_STRETCHED_NM_DEF_CLAIM_LIMIT = (
    "CM1 r21.1 selected fixed-terrain stretch_z=1 live solve1 calcnm/calcdef "
    "refresh for the exact 300x300x75, 50 m horizontal, 15 km-top, four-open "
    "child metric identity. The caller supplies current live state and the "
    "official solve1 refresh index. Closure coefficients/tendencies, RK stage "
    "evolution, microphysics, forecast skill, storm, and tornado claims are "
    "excluded"
)

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = MappingProxyType(
    {
        "src/turb.F": (
            "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d"
        ),
        "src/solve1.F": (
            "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
    }
)
SOURCE_ANCHORS = MappingProxyType(
    {
        "solve1_refresh_cadence": "src/solve1.F:1050-1224",
        "calcdef_dispatch": "src/turb.F:1326-1389",
        "terrain_mass_strain": "src/turb.F:5060-5125",
        "terrain_face_strain": "src/turb.F:5134-5167",
        "four_open_boundary": "src/turb.F:5171-5290",
        "deformation_reduction": "src/turb.F:5340-5382",
        "calcnm": "src/turb.F:5432-5685",
        "metric_construction": "src/param.F:7789-7953;8985-9062",
    }
)

LIVE_INPUT_NAMES = tuple(
    dict.fromkeys((*CHILD_MOIST_INPUT_NAMES, *CALCDEF_INPUT_NAMES))
)
LIVE_OUTPUT_NAMES = ("nm", "defv", "defh")


def _replace_exact(source: str, old: str, new: str, *, count: int, label: str) -> str:
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(
            f"{label} source transform expected {count} occurrence(s), found {actual}"
        )
    return source.replace(old, new)


def _build_stretched_calcdef_source() -> str:
    """Derive the profile-aware terrain branch from the sealed base source."""

    if (
        hashlib.sha256(CUDA_TERRAIN_DEFORMATION_SOURCE.encode()).hexdigest()
        != CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256
    ):
        raise RuntimeError("base calcdef CUDA source differs from its pin")
    source = CUDA_TERRAIN_DEFORMATION_SOURCE.replace(
        "gpuwm-cm1.cuda-terrain-deformation-kernels/v1",
        CUDA_GM1_CHILD_STRETCHED_NM_DEF_ABI,
        1,
    )
    old_atwu = "__device__ __forceinline__ float atwu(const float* q,unsigned int kf,unsigned int j,unsigned int i,unsigned int nk,unsigned int nj,unsigned int ni){\n if(kf==0u)return a32(a32(m32(1.875f,q[i3(0u,j,i,nj,ni+1u)]),m32(-1.25f,q[i3(1u,j,i,nj,ni+1u)])),m32(0.375f,q[i3(2u,j,i,nj,ni+1u)]));\n if(kf==nk)return a32(a32(m32(1.875f,q[i3(nk-1u,j,i,nj,ni+1u)]),m32(-1.25f,q[i3(nk-2u,j,i,nj,ni+1u)])),m32(0.375f,q[i3(nk-3u,j,i,nj,ni+1u)]));\n return m32(0.5f,a32(q[i3(kf-1u,j,i,nj,ni+1u)],q[i3(kf,j,i,nj,ni+1u)]));}"
    new_atwu = "__device__ __forceinline__ float atwu(const float* q,unsigned int kf,unsigned int j,unsigned int i,unsigned int nk,unsigned int nj,unsigned int ni,const float* sigma,const float* sigmaf,const float* rds){\n if(kf==0u)return a32(a32(m32(1.875f,q[i3(0u,j,i,nj,ni+1u)]),m32(-1.25f,q[i3(1u,j,i,nj,ni+1u)])),m32(0.375f,q[i3(2u,j,i,nj,ni+1u)]));\n if(kf==nk)return a32(a32(m32(1.875f,q[i3(nk-1u,j,i,nj,ni+1u)]),m32(-1.25f,q[i3(nk-2u,j,i,nj,ni+1u)])),m32(0.375f,q[i3(nk-3u,j,i,nj,ni+1u)]));\n float r2=m32(s32(sigmaf[kf+1u],sigma[kf]),rds[kf+1u]),r1=s32(1.0f,r2);return a32(m32(r1,q[i3(kf-1u,j,i,nj,ni+1u)]),m32(r2,q[i3(kf,j,i,nj,ni+1u)]));}"
    old_atwv = "__device__ __forceinline__ float atwv(const float* q,unsigned int kf,unsigned int j,unsigned int i,unsigned int nk,unsigned int nj,unsigned int ni){\n if(kf==0u)return a32(a32(m32(1.875f,q[i3(0u,j,i,nj+1u,ni)]),m32(-1.25f,q[i3(1u,j,i,nj+1u,ni)])),m32(0.375f,q[i3(2u,j,i,nj+1u,ni)]));\n if(kf==nk)return a32(a32(m32(1.875f,q[i3(nk-1u,j,i,nj+1u,ni)]),m32(-1.25f,q[i3(nk-2u,j,i,nj+1u,ni)])),m32(0.375f,q[i3(nk-3u,j,i,nj+1u,ni)]));\n return m32(0.5f,a32(q[i3(kf-1u,j,i,nj+1u,ni)],q[i3(kf,j,i,nj+1u,ni)]));}"
    new_atwv = "__device__ __forceinline__ float atwv(const float* q,unsigned int kf,unsigned int j,unsigned int i,unsigned int nk,unsigned int nj,unsigned int ni,const float* sigma,const float* sigmaf,const float* rds){\n if(kf==0u)return a32(a32(m32(1.875f,q[i3(0u,j,i,nj+1u,ni)]),m32(-1.25f,q[i3(1u,j,i,nj+1u,ni)])),m32(0.375f,q[i3(2u,j,i,nj+1u,ni)]));\n if(kf==nk)return a32(a32(m32(1.875f,q[i3(nk-1u,j,i,nj+1u,ni)]),m32(-1.25f,q[i3(nk-2u,j,i,nj+1u,ni)])),m32(0.375f,q[i3(nk-3u,j,i,nj+1u,ni)]));\n float r2=m32(s32(sigmaf[kf+1u],sigma[kf]),rds[kf+1u]),r1=s32(1.0f,r2);return a32(m32(r1,q[i3(kf-1u,j,i,nj+1u,ni)]),m32(r2,q[i3(kf,j,i,nj+1u,ni)]));}"
    source = _replace_exact(source, old_atwu, new_atwu, count=1, label="calcdef atwu")
    source = _replace_exact(source, old_atwv, new_atwv, count=1, label="calcdef atwv")

    mass_marker = 'extern "C" __global__ void cm1_calcdef_terrain_mass_v1('
    face_marker = 'extern "C" __global__ void cm1_calcdef_terrain_face_v1('
    reduce_marker = 'extern "C" __global__ void cm1_calcdef_terrain_reduce_v1('
    head, tail = source.split(mass_marker, 1)
    mass, tail = tail.split(face_marker, 1)
    face, reduce = tail.split(reduce_marker, 1)

    mass = _replace_exact(
        mass,
        "float rdx,float rdy,float rds,float dz){",
        "float rdx,float rdy,const float* sigma,const float* sigmaf,const float* rds,const float* rdsf,float zt){",
        count=1,
        label="calcdef mass signature",
    )
    mass = _replace_exact(
        mass,
        "const float zt=m32((float)nk,dz);",
        "",
        count=1,
        label="calcdef mass synthesized top",
    )
    mass, atwu_count = re.subn(r"atwu\(([^()]*)\)", r"atwu(\1,sigma,sigmaf,rds)", mass)
    mass, atwv_count = re.subn(r"atwv\(([^()]*)\)", r"atwv(\1,sigma,sigmaf,rds)", mass)
    if atwu_count != 8 or atwv_count != 8:
        raise RuntimeError(
            f"calcdef interpolation call transform differs: atwu={atwu_count}, atwv={atwv_count}"
        )
    mass = _replace_exact(
        mass,
        "),0.5f),rds)",
        "),0.5f),rdsf[k+1u])",
        count=2,
        label="calcdef mass horizontal strain rdsf",
    )
    mass = _replace_exact(
        mass,
        "),rds),gz",
        "),rdsf[k+1u]),gz",
        count=1,
        label="calcdef mass vertical strain rdsf",
    )
    mass = _replace_exact(
        mass,
        "),rds),r1)",
        "),rdsf[k+1u]),r1)",
        count=2,
        label="calcdef mass terrain cross rdsf",
    )
    mass = _replace_exact(
        mass,
        "float sl=m32((float)k,dz),su=m32((float)(k+1u),dz);",
        "float sl=sigmaf[k+1u],su=sigmaf[k+2u];",
        count=1,
        label="calcdef mass sigmaf",
    )

    face = _replace_exact(
        face,
        "float rdx,float rdy,float rds,float dz){",
        "float rdx,float rdy,const float* sigma,const float* sigmaf,const float* rds,const float* rdsf,float zt){",
        count=1,
        label="calcdef face signature",
    )
    face = _replace_exact(
        face,
        "const float zt=m32((float)nk,dz);",
        "",
        count=1,
        label="calcdef face synthesized top",
    )
    face = _replace_exact(
        face,
        "float slo=m32(s32((float)kf,0.5f),dz),sup=m32(a32((float)kf,0.5f),dz);",
        "float slo=sigma[kf],sup=sigma[kf+1u];",
        count=1,
        label="calcdef face sigma",
    )
    face = _replace_exact(
        face,
        "),rds),hw=",
        "),rds[kf+1u]),hw=",
        count=2,
        label="calcdef face direct rds",
    )
    face = _replace_exact(
        face,
        "m32(0.5f,rds)",
        "m32(0.5f,rds[kf+1u])",
        count=2,
        label="calcdef face terrain cross rds",
    )

    transformed = (
        head + mass_marker + mass + face_marker + face + reduce_marker + reduce
    )
    forbidden = (
        "float rds,float dz",
        "m32((float)nk,dz)",
        "m32((float)k,dz)",
        "m32(s32((float)kf,0.5f),dz)",
    )
    if any(token in transformed for token in forbidden):
        raise RuntimeError(
            "stretched calcdef transform left a uniform-coordinate token"
        )
    return transformed


CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE = _build_stretched_calcdef_source()
_GENERATED_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE.encode()
).hexdigest()
if _GENERATED_SOURCE_SHA256 != CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256:
    raise RuntimeError("stretched calcdef generated source differs from its pin")


@dataclass(frozen=True, slots=True)
class ChildStretchedNmDefMetricView:
    arrays: Mapping[str, Any]
    ready_event: Any
    identity: Mapping[str, object]
    compact_validation: bool


@dataclass(frozen=True, slots=True)
class CompiledChildStretchedCalcdef:
    module: Any
    kernels: tuple[Any, ...]
    target_arch: str
    runtime_identity: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaChildStretchedNmDefLaunch:
    ready_event: Any
    arrays: Mapping[str, Any]
    workspace: Mapping[str, Any]
    receipt: Mapping[str, object]
    compiled_calcdef: CompiledChildStretchedCalcdef
    compiled_calcnm: CompiledChildMoistNmKernel
    calcnm_launch: Any


def bind_gm1_child_stretched_nm_def_metrics(
    metrics: Any, *, allow_compact_validation: bool = False
) -> ChildStretchedNmDefMetricView:
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    if allow_compact_validation:
        if not isinstance(metrics, Mapping):
            raise TypeError("compact stretched metrics must be a mapping")
        names = ("sigma", "sigmaf", "rds", "rdsf")
        if any(name not in metrics for name in names):
            raise ValueError("compact stretched metric mapping is incomplete")
        arrays = MappingProxyType({name: metrics[name] for name in names})
        return ChildStretchedNmDefMetricView(
            arrays,
            metrics.get("ready_event"),
            MappingProxyType(
                {
                    "metric_abi": GM1_CHILD_STRETCHED_METRIC_ABI,
                    "fixture_class": "analytic compact nonuniform operator validation",
                    "nk": int(arrays["sigma"].shape[0]) - 2,
                }
            ),
            True,
        )
    sealed = _bind_sealed_stretched_metrics(metrics)
    return ChildStretchedNmDefMetricView(
        sealed.arrays,
        metrics.ready_event,
        MappingProxyType(
            {
                "metric_abi": GM1_CHILD_STRETCHED_METRIC_ABI,
                "grid_sha256": GM1_CHILD_STRETCHED_GRID_SHA256,
                "proof_sha256": GM1_CHILD_STRETCHED_PROOF_SHA256,
                "static_combined_sha256": GM1_CHILD_STATIC_COMBINED_SHA256,
                "profile_sha256": dict(GM1_CHILD_STRETCHED_PROFILE_SHA256),
                "ztop_f32": GM1_CHILD_STRETCHED_ZTOP_F32,
                "reference_dz_f32": GM1_CHILD_STRETCHED_DZ_REFERENCE_F32,
            }
        ),
        False,
    )


def _canonical_arch(cupy: Any) -> tuple[str, Mapping[str, object]]:
    device = cupy.cuda.Device()
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    major, minor = int(props["major"]), int(props["minor"])
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return f"{major}{minor}", MappingProxyType(
        {
            "device_id": int(device.id),
            "device_name": str(name),
            "compute_capability": f"{major}.{minor}",
            "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
        }
    )


def compile_cuda_gm1_child_stretched_calcdef() -> CompiledChildStretchedCalcdef:
    if _GENERATED_SOURCE_SHA256 != CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256:
        raise RuntimeError("stretched calcdef source is not sealed to its SHA-256")
    cupy = importlib.import_module("cupy")
    target, runtime = _canonical_arch(cupy)
    options = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE,
        options=options,
        name_expressions=CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES,
        filename="gm1_child_stretched_calcdef.cu",
    )
    kernels = tuple(
        module.get_function(name) for name in CUDA_TERRAIN_DEFORMATION_KERNEL_NAMES
    )
    return CompiledChildStretchedCalcdef(module, kernels, target, runtime)


def _validate_scalar(name: str, value: float, expected: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result != expected:
        refuse(
            code="GM1-CHILD-STRETCHED-NM-DEF-GEOMETRY",
            gate="child-stretched-live-nm-def",
            option=name,
            value=result,
            reason="the live refresh is sealed to the fixed 50 m child geometry",
            remedy=f"use {name}={expected!r} or add an independent oracle",
        )
    return result


def _validate_metrics(
    cupy: Any, view: ChildStretchedNmDefMetricView, nk: int
) -> dict[str, Any]:
    shapes = {
        "sigma": (nk + 2,),
        "sigmaf": (nk + 3,),
        "rds": (nk + 2,),
        "rdsf": (nk + 3,),
    }
    result: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = view.arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 with shape {shape}")
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != 4:
        raise ValueError("stretched vertical profiles must not alias")
    return result


def _require_live_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    calcdef_shapes = terrain_deformation_shapes(ni, nj, nk)
    cell, face = (nk, nj, ni), (nk + 1, nj, ni)
    moist_shapes = {
        "c1": face,
        "c2": face,
        "mf": face,
        "pi0": cell,
        "th0": cell,
        "prs": cell,
        "pp3d": cell,
        "th3d": cell,
        "q3d": (6, nk, nj, ni),
        "nm": face,
    }
    shapes = {**calcdef_shapes, **moist_shapes}
    names = tuple(dict.fromkeys((*LIVE_INPUT_NAMES, *LIVE_OUTPUT_NAMES)))
    result: dict[str, Any] = {}
    for name in names:
        if name not in arrays:
            raise ValueError(f"missing live nm/def array: {name}")
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be C-order CUDA float32 with shape {shapes[name]}"
            )
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("live nm/def arrays must not alias")
    return result


def launch_cuda_gm1_child_stretched_nm_def_refresh(
    arrays: Mapping[str, Any],
    *,
    metrics: Any,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    refresh_index: int,
    stream: Any | None = None,
    compiled_calcdef: CompiledChildStretchedCalcdef | None = None,
    compiled_calcnm: CompiledChildMoistNmKernel | None = None,
    selection: TerrainDeformationSelection = DEFAULT_TERRAIN_DEFORMATION_SELECTION,
    allow_compact_validation: bool = False,
) -> CudaChildStretchedNmDefLaunch:
    """Enqueue live ``calcnm`` then profile-aware ``calcdef`` on one stream."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    if isinstance(refresh_index, bool) or not isinstance(refresh_index, int):
        raise TypeError("refresh_index must be an integer")
    if refresh_index < 1:
        raise ValueError("refresh_index must be >= 1")
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    if not allow_compact_validation and (ni, nj, nk) != (300, 300, 75):
        raise ValueError("real child live nm/def refresh requires 300x300x75")
    dx = _validate_scalar("dx", dx, 50.0 if not allow_compact_validation else float(dx))
    dy = _validate_scalar("dy", dy, 50.0 if not allow_compact_validation else float(dy))
    dz = _validate_scalar("dz", dz, 100.0)
    ztop = _validate_scalar(
        "ztop", ztop, 15000.0 if not allow_compact_validation else float(ztop)
    )
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    view = bind_gm1_child_stretched_nm_def_metrics(
        metrics, allow_compact_validation=allow_compact_validation
    )
    profiles = _validate_metrics(cupy, view, nk)
    live = _require_live_arrays(cupy, arrays, ni, nj, nk)
    compiled_calcdef = compiled_calcdef or compile_cuda_gm1_child_stretched_calcdef()
    compiled_calcnm = compiled_calcnm or compile_cuda_child_moist_nm()

    moist_workspace: dict[str, Any] = {"nm": live["nm"]}
    for name in CHILD_MOIST_OUTPUT_NAMES:
        if name == "nm":
            continue
        dtype = cupy.uint8 if name == "saturation_mask" else cupy.float32
        moist_workspace[name] = cupy.empty((nk, nj, ni), dtype=dtype)
    moist_arrays = {
        name: live[name] for name in CHILD_MOIST_INPUT_NAMES
    } | moist_workspace
    calcdef_scratch = {
        name: cupy.empty(shape, dtype=cupy.float32)
        for name, shape in terrain_deformation_scratch_shapes(ni, nj, nk).items()
    }

    mass_words = nk * (nj + 1) * (ni + 1)
    face_words = (nk - 1) * (nj + 1) * (ni + 1)
    reduce_words = (nk - 1) * nj * ni
    grids = tuple(
        (
            (words + CUDA_TERRAIN_DEFORMATION_THREADS - 1)
            // CUDA_TERRAIN_DEFORMATION_THREADS,
        )
        for words in (mass_words, face_words, reduce_words)
    )
    block = (CUDA_TERRAIN_DEFORMATION_THREADS,)
    u32, f32 = cupy.uint32, cupy.float32
    rdx, rdy = f32(1.0 / dx), f32(1.0 / dy)
    with stream:
        if view.ready_event is not None:
            stream.wait_event(view.ready_event)
        moist = launch_cuda_child_moist_nm(
            moist_arrays,
            ni=ni,
            nj=nj,
            nk=nk,
            dz=dz,
            stream=stream,
            compiled=compiled_calcnm,
        )
        live["defv"].fill(f32(0.0))
        live["defh"].fill(f32(0.0))
        compiled_calcdef.kernels[0](
            grids[0],
            block,
            (
                live["u3d"],
                live["v3d"],
                live["w3d"],
                live["rho"],
                live["gz"],
                live["rgzu"],
                live["rgzv"],
                live["gx"],
                live["gy"],
                calcdef_scratch["s11"],
                calcdef_scratch["s22"],
                calcdef_scratch["s33"],
                calcdef_scratch["s12"],
                u32(ni),
                u32(nj),
                u32(nk),
                rdx,
                rdy,
                profiles["sigma"],
                profiles["sigmaf"],
                profiles["rds"],
                profiles["rdsf"],
                f32(ztop),
            ),
            stream=stream,
        )
        compiled_calcdef.kernels[1](
            grids[1],
            block,
            (
                live["u3d"],
                live["v3d"],
                live["w3d"],
                live["rf"],
                live["gz"],
                live["rgz"],
                calcdef_scratch["s13"],
                calcdef_scratch["s23"],
                u32(ni),
                u32(nj),
                u32(nk),
                rdx,
                rdy,
                profiles["sigma"],
                profiles["sigmaf"],
                profiles["rds"],
                profiles["rdsf"],
                f32(ztop),
            ),
            stream=stream,
        )
        compiled_calcdef.kernels[2](
            grids[2],
            block,
            (
                live["rf"],
                live["c1"],
                live["c2"],
                calcdef_scratch["s11"],
                calcdef_scratch["s22"],
                calcdef_scratch["s33"],
                calcdef_scratch["s12"],
                calcdef_scratch["s13"],
                calcdef_scratch["s23"],
                live["defv"],
                live["defh"],
                u32(ni),
                u32(nj),
                u32(nk),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    outputs = MappingProxyType({name: live[name] for name in LIVE_OUTPUT_NAMES})
    workspace = MappingProxyType({**moist_workspace, **calcdef_scratch})
    receipt = MappingProxyType(
        {
            "schema": CUDA_GM1_CHILD_STRETCHED_NM_DEF_SCHEMA,
            "abi": CUDA_GM1_CHILD_STRETCHED_NM_DEF_ABI,
            "status": "ENQUEUED_LIVE_SOLVE1_CALCNM_CALCDEF_REFRESH",
            "refresh_index": refresh_index,
            "source_order": ["calcnm", "calcdef"],
            "same_stream_no_host_sync": True,
            "stream_pointer": int(stream.ptr),
            "domain": {"ni": ni, "nj": nj, "nk": nk},
            "spacing_m": {"dx": dx, "dy": dy, "reference_dz": dz, "ztop": ztop},
            "metrics": dict(view.identity),
            "authority": {
                "tarball_sha256": AUTHORITY_TARBALL_SHA256,
                "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
                "source_anchors": dict(SOURCE_ANCHORS),
            },
            "operator_source_sha256": {
                "calcnm": CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
                "calcdef": CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256,
                "base_calcdef": CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
            },
            "kernel_target_arch": {
                "calcnm": compiled_calcnm.target_arch,
                "calcdef": compiled_calcdef.target_arch,
            },
            "runtime": dict(compiled_calcdef.runtime_identity),
            "claim_limit": GM1_CHILD_STRETCHED_NM_DEF_CLAIM_LIMIT,
        }
    )
    return CudaChildStretchedNmDefLaunch(
        event,
        outputs,
        workspace,
        receipt,
        compiled_calcdef,
        compiled_calcnm,
        moist,
    )


def _source_pin_is_valid() -> bool:
    return (
        hashlib.sha256(CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE.encode()).hexdigest()
        == CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256
    )


__all__ = (
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE",
    "CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256",
    "CUDA_GM1_CHILD_STRETCHED_NM_DEF_ABI",
    "CUDA_GM1_CHILD_STRETCHED_NM_DEF_SCHEMA",
    "CompiledChildStretchedCalcdef",
    "CudaChildStretchedNmDefLaunch",
    "GM1_CHILD_STRETCHED_NM_DEF_CLAIM_LIMIT",
    "LIVE_INPUT_NAMES",
    "LIVE_OUTPUT_NAMES",
    "SOURCE_ANCHORS",
    "bind_gm1_child_stretched_nm_def_metrics",
    "compile_cuda_gm1_child_stretched_calcdef",
    "launch_cuda_gm1_child_stretched_nm_def_refresh",
)
