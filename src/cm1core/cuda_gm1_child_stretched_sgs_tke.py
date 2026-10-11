"""Profile-aware SGS stress and TKE evolution for the sealed GM1 child.

CM1's terrain operators use two different vertical metric families.  The
implicit vertical diffusion systems retain the namelist reference ``rdz`` and
receive stretching through ``mh``/``mf``.  Terrain cross terms and ``advw``
vertical flux divergence instead consume the live ``sigma``, ``sigmaf``,
``rds``, and ``rdsf`` profiles.  The older parent kernels only implemented the
uniform-profile specialization.  This module keeps their proven statement
order while replacing exactly those profile-dependent expressions.

The public launchers are same-stream APIs.  They accept the strict
``CudaGm1ChildTerrainMetricsLaunch`` publication (or an explicitly marked
compact analytic metric mapping), wait on its ready event, enqueue work, and
return a new ready event without synchronizing the host.
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

from cm1core.cuda_sgs_stress_tendency import (
    CUDA_SGS_STRESS_KERNEL_NAMES,
    CUDA_SGS_STRESS_MAX_NK,
    CUDA_SGS_STRESS_SOURCE,
    CUDA_SGS_STRESS_SOURCE_SHA256,
    CUDA_SGS_STRESS_THREADS,
    DEFAULT_SGS_STRESS_SELECTION,
    SgsStressSelection,
    _validate_grid as _validate_sgs_grid,
    _validate_selection as _validate_sgs_selection,
    _require_surface_binding,
    _validate_square_spacing,
    _resize_sgs_column_workspace,
    sgs_stress_shapes,
)
from cm1core.cuda_tke_evolution import (
    CUDA_TKE_EVOLUTION_KERNEL_NAMES,
    CUDA_TKE_EVOLUTION_MAX_NK,
    CUDA_TKE_EVOLUTION_NGXY,
    CUDA_TKE_EVOLUTION_SOURCE,
    CUDA_TKE_EVOLUTION_SOURCE_SHA256,
    CUDA_TKE_EVOLUTION_THREADS,
    DEFAULT_TKE_EVOLUTION_SELECTION,
    TkeEvolutionSelection,
    _validate_grid as _validate_tke_grid,
    _validate_selection as _validate_tke_selection,
    _resize_tke_column_workspace,
    _validate_column_capacity,
    tke_evolution_shapes,
)
from cm1core.cuda_surface_drag import (
    SurfaceDragBinding,
    enqueue_surface_stress,
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
from cm1core.errors import refuse
from cm1core.grid_geometry import CartesianGridGeometry


CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA = "gm1.cuda-child-stretched-sgs-tke/v1"
CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI = "gm1.cuda-child-stretched-sgs-tke-kernels/v1"
EXPECTED_STRETCHED_SGS_SOURCE_SHA256 = (
    "9787d00ed8b3309fd33df148e186d85624cb96fa1871013cc54c03516aed0d83"
)
EXPECTED_STRETCHED_TKE_SOURCE_SHA256 = (
    "471b69f2d99a5d1ed6e1b6fb37fb2aba4ba8c27c38c3fc4d40d0cb844aa97814"
)
GM1_CHILD_STATIC_COMBINED_SHA256 = (
    "2ec8b6a5a48a81017527e66c4fa6b264f1476b737fa2b0a02f7795012877c0a9"
)
# cuda_gm1_child_stretched_metrics.py as merged by d4e29a2 (sota/stretchlive's
# generated-coordinate checks, e4024cc's 3a9b1849..., beside integration's
# stretched_vertical_identity_defect), then with require_generated_vertical
# reading a binding's sigmaf(nk+1) back once per binding instead of on every
# stretched launch (2cb1e01b... before that), then with the flat column's
# mh/mf, flat_column, rdz_f32 and require_flat_column, which the Cartesian
# terrain_flag=.false. lane binds (db1da423... before that).
COMMON_STRETCHED_METRICS_MODULE_SHA256 = (
    "13484054f10eb867df52606889ef0756418fa765cc90b802e7f1feb39f8adfa9"
)
GM1_CHILD_RDS_SHA256 = (
    "b995cfc800e8b69c257d1f0e9e328c315d91652f7b0a86a748af49eadd4306fe"
)
GM1_CHILD_RDSF_SHA256 = (
    "008898d0ed25941c735f0f7a23746d3a7eea562d12d01ad7cc07b6d50ed6f416"
)
GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT = (
    "CM1 r21.1 selected fixed-terrain stretch_z=1 SGS stress divergence and "
    "SGS-TKE turbt/advw/RK/commit operators for the exact 300x300x75, 50 m "
    "horizontal, 15 km-top four-open child metric identity; supplied live "
    "stress/flow/mass-flux/tendency carriers and per-stage post-acoustic flux "
    "snapshots are required. Momentum/scalar advection, pressure, acoustic "
    "rebuild, microphysics, boundary-history production, coupled trajectory, "
    "forecast skill, storm, and tornado claims are excluded"
)

SOURCE_ANCHORS = MappingProxyType(
    {
        "vertical_grid_and_coefficients": "src/param.F:7789-7953;8985-9062",
        "turbt_profile_terms": "src/turbtend.F:700-1044",
        "turbu_profile_terms": "src/turbtend.F:1052-1361;1530-1635",
        "turbv_profile_terms": "src/turbtend.F:1816-2128;2325-2415",
        "turbw_profile_terms": "src/turbtend.F:2473-2885",
        "tke_advw_profile_terms": "src/adv.F:2778-3610",
        "tke_rk_cadence": "src/solve2.F:1708-1803",
        "tke_commit": "src/solve3.F:560-590",
    }
)

_PROFILE_SHAPES = {
    "sigma": (77,),
    "sigmaf": (78,),
    "rds": (77,),
    "rdsf": (78,),
}
_PROFILE_SHA256 = {
    **GM1_CHILD_STRETCHED_PROFILE_SHA256,
}


def _replace_exact(source: str, old: str, new: str, *, count: int, label: str) -> str:
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(
            f"{label} source transform expected {count} occurrence(s), found {actual}"
        )
    return source.replace(old, new)


def _build_stretched_sgs_source() -> str:
    """Transform the sealed uniform specialization at audited source sites."""

    if (
        hashlib.sha256(CUDA_SGS_STRESS_SOURCE.encode()).hexdigest()
        != CUDA_SGS_STRESS_SOURCE_SHA256
    ):
        raise RuntimeError("base SGS stress source differs from its pin")
    source = CUDA_SGS_STRESS_SOURCE.replace(
        "gpuwm-cm1.cuda-sgs-stress-tendency-kernels/v1",
        CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI + ":sgs",
        1,
    )
    old_atw = "__device__ __forceinline__ float ATW(const float* a,int f,int j,int i,int nk,int nj,int ni){if(f==0)return A(A(M(1.875f,CELL(a,0,j,i,nj,ni)),M(-1.25f,CELL(a,1,j,i,nj,ni))),M(0.375f,CELL(a,2,j,i,nj,ni)));if(f==nk)return A(A(M(1.875f,CELL(a,nk-1,j,i,nj,ni)),M(-1.25f,CELL(a,nk-2,j,i,nj,ni))),M(0.375f,CELL(a,nk-3,j,i,nj,ni)));return M(0.5f,A(CELL(a,f-1,j,i,nj,ni),CELL(a,f,j,i,nj,ni)));}"
    new_atw = "__device__ __forceinline__ float ATW(const float* a,int f,int j,int i,int nk,int nj,int ni,const float*sigma,const float*sigmaf,const float*rds){if(f==0)return A(A(M(1.875f,CELL(a,0,j,i,nj,ni)),M(-1.25f,CELL(a,1,j,i,nj,ni))),M(0.375f,CELL(a,2,j,i,nj,ni)));if(f==nk)return A(A(M(1.875f,CELL(a,nk-1,j,i,nj,ni)),M(-1.25f,CELL(a,nk-2,j,i,nj,ni))),M(0.375f,CELL(a,nk-3,j,i,nj,ni)));float r2=M(S(sigmaf[f+1],sigma[f]),rds[f+1]),r1=S(1.0f,r2);return A(M(r1,CELL(a,f-1,j,i,nj,ni)),M(r2,CELL(a,f,j,i,nj,ni)));}"
    old_atw12 = "__device__ __forceinline__ float ATW12(const float* a,int f,int j,int i,int nk,int nj,int ni){unsigned long long p=I3(0,C(j,nj+1),C(i,ni+1),nj+1,ni+1);unsigned long long z=(unsigned long long)(nj+1)*(ni+1);if(f==0)return A(A(M(1.875f,a[p]),M(-1.25f,a[p+z])),M(0.375f,a[p+2*z]));if(f==nk)return A(A(M(1.875f,a[p+(nk-1)*z]),M(-1.25f,a[p+(nk-2)*z])),M(0.375f,a[p+(nk-3)*z]));return M(0.5f,A(a[p+(f-1)*z],a[p+f*z]));}"
    new_atw12 = "__device__ __forceinline__ float ATW12(const float* a,int f,int j,int i,int nk,int nj,int ni,const float*sigma,const float*sigmaf,const float*rds){unsigned long long p=I3(0,C(j,nj+1),C(i,ni+1),nj+1,ni+1);unsigned long long z=(unsigned long long)(nj+1)*(ni+1);if(f==0)return A(A(M(1.875f,a[p]),M(-1.25f,a[p+z])),M(0.375f,a[p+2*z]));if(f==nk)return A(A(M(1.875f,a[p+(nk-1)*z]),M(-1.25f,a[p+(nk-2)*z])),M(0.375f,a[p+(nk-3)*z]));float r2=M(S(sigmaf[f+1],sigma[f]),rds[f+1]),r1=S(1.0f,r2);return A(M(r1,a[p+(f-1)*z]),M(r2,a[p+f*z]));}"
    source = _replace_exact(source, old_atw, new_atw, count=1, label="SGS ATW")
    source = _replace_exact(source, old_atw12, new_atw12, count=1, label="SGS ATW12")
    marker = "// COLUMN_KERNELS_V1"
    head, columns = source.split(marker, 1)
    columns, signature_count = re.subn(
        r"unsigned nj,unsigned nk,float dx,float dz,float dt\)\{",
        "unsigned nj,unsigned nk,float dx,float dz,float dt,const float*sigma,const float*sigmaf,const float*rds,const float*rdsf,float zt){",
        columns,
    )
    if signature_count != 3:
        raise RuntimeError(
            f"SGS profile signature transform found {signature_count}, expected 3"
        )
    columns, atw12_count = re.subn(
        r"ATW12\(([^()]*)\)", r"ATW12(\1,sigma,sigmaf,rds)", columns
    )
    columns, atw_count = re.subn(
        r"ATW\(([^()]*)\)", r"ATW(\1,sigma,sigmaf,rds)", columns
    )
    if atw12_count != 8 or atw_count != 8:
        raise RuntimeError(
            f"SGS interpolation call transform differs: ATW12={atw12_count}, ATW={atw_count}"
        )
    columns = _replace_exact(
        columns, ",zt=M((float)nk,dz)", "", count=3, label="SGS ztop"
    )
    # turbu/turbv mass-level horizontal divergence uses rdsf(k).
    columns = _replace_exact(
        columns,
        "M(S(up,lo),rdz)",
        "M(S(up,lo),rdsf[k+1u])",
        count=1,
        label="SGS rdsf direct terms",
    )
    columns = _replace_exact(
        columns,
        "M(S(yu,yl),rz)",
        "M(S(yu,yl),rdsf[k+1u])",
        count=1,
        label="SGS turbv rdsf direct term",
    )
    columns = _replace_exact(
        columns,
        "M(D(1.0f,dx),rdz)",
        "M(D(1.0f,dx),rdsf[k+1u])",
        count=1,
        label="SGS rdsf cross terms",
    )
    # sigmaf(k+1), sigmaf(k) at C mass index k == Fortran k-1.
    columns = columns.replace(
        "S(zt,M((float)k+1.0f,dz))", "S(zt,sigmaf[k+2u])"
    ).replace("S(zt,M((float)k,dz))", "S(zt,sigmaf[k+1u])")
    # dwdx/dwdy and turbw use sigma(k), sigma(k-1), rds(k).
    columns = columns.replace("S(zt,M((float)k+0.5f,dz))", "S(zt,sigma[k+1u])").replace(
        "S(zt,M((float)k-0.5f,dz))", "S(zt,sigma[k])"
    )
    columns = columns.replace("S(zt,M((float)f+0.5f,dz))", "S(zt,sigma[f+1u])").replace(
        "S(zt,M((float)f-0.5f,dz))", "S(zt,sigma[f])"
    )
    columns = _replace_exact(
        columns,
        "M(rdz,M(S(rr,rl),rdx))",
        "M(rds[k+1u],M(S(rr,rl),rdx))",
        count=2,
        label="SGS u rds cross",
    )
    columns = _replace_exact(
        columns,
        "M(rz,M(S(rn,rs),rd))",
        "M(rds[k+1u],M(S(rn,rs),rd))",
        count=2,
        label="SGS v rds cross",
    )
    columns = _replace_exact(
        columns,
        "M(S(r2,r1),M(rd,rz))",
        "M(S(r2,r1),M(rd,rdsf[k+1u]))",
        count=1,
        label="SGS turbv rdsf cross term",
    )
    columns = _replace_exact(
        columns,
        "M(rd,rz)",
        "M(rd,rds[f+1u])",
        count=2,
        label="SGS turbw rds terms",
    )
    forbidden = (
        "zt=M((float)nk,dz)",
        "S(zt,M((float)k+1.0f,dz))",
        "S(zt,M((float)k+0.5f,dz))",
        "S(zt,M((float)f+0.5f,dz))",
    )
    if any(token in columns for token in forbidden):
        raise RuntimeError("SGS profile transform left a uniform-coordinate token")
    return head + marker + columns


def _build_stretched_tke_source() -> str:
    """Transform turbt and advw only where the official profile is live."""

    if (
        hashlib.sha256(CUDA_TKE_EVOLUTION_SOURCE.encode()).hexdigest()
        != CUDA_TKE_EVOLUTION_SOURCE_SHA256
    ):
        raise RuntimeError("base TKE evolution source differs from its pin")
    source = CUDA_TKE_EVOLUTION_SOURCE.replace(
        "gpuwm-cm1.cuda-tke-evolution-kernels/v1",
        CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI + ":tke",
        1,
    )
    start = source.index("#if 0")
    end = source.index("#endif", start) + len("#endif")
    source = source[:start] + source[end:]
    # TX/TY/SFX/SFY are the terrain part of turbt.  Transform declarations
    # separately from calls so a broad regex cannot mutate a function ABI.
    source = _replace_exact(
        source,
        "int ni,float rd,float rz,float dz){",
        "int ni,float rd,const float*sigma,const float*rds,float zt){",
        count=2,
        label="TKE TX/TY signatures",
    )
    source = _replace_exact(
        source,
        "),kv=A(CELL(km,f,j,r,nj,ni),CELL(km,f,j,l,nj,ni)),zt=M((float)nk,dz);",
        "),kv=A(CELL(km,f,j,r,nj,ni),CELL(km,f,j,l,nj,ni));",
        count=1,
        label="TKE TX synthesized top",
    )
    source = _replace_exact(
        source,
        "),kv=A(CELL(km,f,n,i,nj,ni),CELL(km,f,s,i,nj,ni)),zt=M((float)nk,dz);",
        "),kv=A(CELL(km,f,n,i,nj,ni),CELL(km,f,s,i,nj,ni));",
        count=1,
        label="TKE TY synthesized top",
    )
    source = source.replace("S(zt,M((float)f+.5f,dz))", "S(zt,sigma[f+1])").replace(
        "S(zt,M((float)f-.5f,dz))", "S(zt,sigma[f])"
    )
    source = _replace_exact(
        source,
        "M(rz,M(S(G(rgz",
        "M(rds[f+1],M(S(G(rgz",
        count=2,
        label="TKE TX/TY rds",
    )
    source = _replace_exact(
        source,
        "int ni,float rx,float rz,float dz){return M(.5f,A(TX(t,km,rf,gz,rgz,m,j,face,nk,nj,ni,rx,rz,dz),TX(t,km,rf,gz,rgz,m+1,j,face,nk,nj,ni,rx,rz,dz)));}",
        "int ni,float rx,const float*sigma,const float*rds,float zt){return M(.5f,A(TX(t,km,rf,gz,rgz,m,j,face,nk,nj,ni,rx,sigma,rds,zt),TX(t,km,rf,gz,rgz,m+1,j,face,nk,nj,ni,rx,sigma,rds,zt)));}",
        count=1,
        label="TKE SFX signature and calls",
    )
    source = _replace_exact(
        source,
        "int ni,float ry,float rz,float dz){return M(.5f,A(TY(t,km,rf,gz,rgz,m,face,i,nk,nj,ni,ry,rz,dz),TY(t,km,rf,gz,rgz,m+1,face,i,nk,nj,ni,ry,rz,dz)));}",
        "int ni,float ry,const float*sigma,const float*rds,float zt){return M(.5f,A(TY(t,km,rf,gz,rgz,m,face,i,nk,nj,ni,ry,sigma,rds,zt),TY(t,km,rf,gz,rgz,m+1,face,i,nk,nj,ni,ry,sigma,rds,zt)));}",
        count=1,
        label="TKE SFY signature and calls",
    )
    source = _replace_exact(
        source,
        "float dz,float dt){",
        "float dz,float dt,const float*sigma,const float*rds,float zt){",
        count=1,
        label="TKE turbt signature",
    )
    source = _replace_exact(
        source,
        "float rx=D(1,dx),ry=D(1,dy),rz=D(1,dz),zt=M((float)nk,dz);",
        "float rx=D(1,dx),ry=D(1,dy),rz=D(1,dz);",
        count=1,
        label="TKE turbt synthesized top",
    )
    source = source.replace(",rx,rz,dz)", ",rx,sigma,rds,zt)")
    source = source.replace(",ry,rz,dz)", ",ry,sigma,rds,zt)")
    source = source.replace("S(zt,M((float)f+.5f,dz))", "S(zt,sigma[f+1])").replace(
        "S(zt,M((float)f-.5f,dz))", "S(zt,sigma[f])"
    )
    source = _replace_exact(
        source,
        "M(rz,M(S(rgzu",
        "M(rds[f+1],M(S(rgzu",
        count=1,
        label="TKE turbt x rds",
    )
    source = _replace_exact(
        source,
        "M(rz,M(S(rgzv",
        "M(rds[f+1],M(S(rgzv",
        count=1,
        label="TKE turbt y rds",
    )
    # advw vertical divergence at C face f == Fortran k=f+1.
    old_stage = "float dx,float dy,float dz,float dttmp,unsigned w,double ep){"
    new_stage = (
        "float dx,float dy,float dz,float dttmp,unsigned w,double ep,const float*rds){"
    )
    source = _replace_exact(
        source, old_stage, new_stage, count=1, label="TKE stage signature"
    )
    source = _replace_exact(
        source,
        "float rx=D(1,dx),ry=D(1,dy),rz=D(1,dz),mw",
        "float rx=D(1,dx),ry=D(1,dy),mw",
        count=1,
        label="TKE uniform advw rdz",
    )
    source = _replace_exact(
        source,
        "),rz)),tot=",
        "),rds[f+1])),tot=",
        count=1,
        label="TKE advw rds",
    )
    if "zt=M((float)nk,dz)" in source:
        raise RuntimeError("TKE profile transform left a synthesized model top")
    return source


CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE = _build_stretched_sgs_source()
CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE = _build_stretched_tke_source()
CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE.encode()
).hexdigest()
CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE.encode()
).hexdigest()
if CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256 != EXPECTED_STRETCHED_SGS_SOURCE_SHA256:
    raise RuntimeError("stretched SGS generated source differs from its pin")
if CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256 != EXPECTED_STRETCHED_TKE_SOURCE_SHA256:
    raise RuntimeError("stretched TKE generated source differs from its pin")


@dataclass(frozen=True, slots=True)
class ChildStretchedMetricView:
    arrays: Mapping[str, Any]
    ready_event: Any
    identity: Mapping[str, object]
    compact_validation: bool
    geometry: CartesianGridGeometry | None = None


@dataclass(frozen=True, slots=True)
class CompiledChildStretchedSgsTke:
    sgs_module: Any
    sgs_kernels: tuple[Any, ...]
    tke_module: Any
    tke_kernels: tuple[Any, ...]
    target_arch: str
    runtime_identity: Mapping[str, object]
    max_nk: int = CUDA_SGS_STRESS_MAX_NK
    sgs_source_sha256: str = CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256
    tke_source_sha256: str = CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256


@dataclass(frozen=True, slots=True)
class CudaChildStretchedLaunch:
    ready_event: Any
    receipt: Mapping[str, object]
    compiled: CompiledChildStretchedSgsTke


#: The two vertical-metric views the Cartesian closure publishes, by CM1
#: ``stretch_z`` branch.  ``cuda_cartesian_closure`` builds exactly one of
#: them per run and binds it to that run's geometry.
CARTESIAN_VERTICAL_METRIC_SCHEMA = MappingProxyType(
    {
        "uniform": "gm1.uniform-cartesian-vertical-metrics/v1",
        "stretched": "gm1.stretched-cartesian-vertical-metrics/v1",
    }
)


def bind_gm1_child_stretched_sgs_tke_metrics(
    metrics: Any, *, allow_compact_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
) -> ChildStretchedMetricView:
    """Bind the exact terrain publication without a device-to-host transfer."""

    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be boolean")
    if geometry is not None:
        # One generator builds both coordinates (cm1_vertical_grid), so the
        # view carries one of two schemas and which one is not a preference:
        # param.F:7789-7811's uniform column and param.F:7814-8139's stretched
        # one are different level sets, and binding a view that names one to a
        # geometry that declares the other would run the closure on a column
        # this grid does not have.  The two are therefore checked against each
        # other rather than one of them being hard-coded.
        expected_schema = (
            "gm1.uniform-cartesian-vertical-metrics/v1"
            if isinstance(geometry, CartesianGridGeometry) and geometry.stretch_z == 0
            else "gm1.stretched-cartesian-vertical-metrics/v1"
        )
        if (
            not isinstance(geometry, CartesianGridGeometry)
            or not isinstance(metrics, ChildStretchedMetricView)
            or metrics.geometry != geometry
            or metrics.identity.get("grid_geometry") != geometry.record()
            or metrics.identity.get("schema") != expected_schema
            or metrics.compact_validation
        ):
            raise ValueError(
                f"this grid's stretch_z={getattr(geometry, 'stretch_z', None)} "
                f"Cartesian metrics require the exact source-bound "
                f"{expected_schema} geometry view"
            )
        return metrics
    if allow_compact_validation:
        if not isinstance(metrics, Mapping):
            raise TypeError("compact stretched metrics must be a mapping")
        arrays = {name: metrics[name] for name in _PROFILE_SHAPES if name in metrics}
        if set(arrays) != set(_PROFILE_SHAPES):
            raise ValueError("compact stretched metric mapping is incomplete")
        nk = int(arrays["sigma"].shape[0]) - 2
        return ChildStretchedMetricView(
            MappingProxyType(arrays),
            metrics.get("ready_event"),
            MappingProxyType(
                {
                    "metric_abi": GM1_CHILD_STRETCHED_METRIC_ABI,
                    "fixture_class": "analytic compact nonuniform operator validation",
                    "nk": nk,
                }
            ),
            True,
        )

    sealed = _bind_sealed_stretched_metrics(metrics)
    return ChildStretchedMetricView(
        sealed.arrays,
        metrics.ready_event,
        MappingProxyType(
            {
                "metric_abi": GM1_CHILD_STRETCHED_METRIC_ABI,
                "grid_sha256": GM1_CHILD_STRETCHED_GRID_SHA256,
                "proof_sha256": GM1_CHILD_STRETCHED_PROOF_SHA256,
                "static_combined_sha256": GM1_CHILD_STATIC_COMBINED_SHA256,
                "profile_sha256": dict(_PROFILE_SHA256),
                "ztop_f32": GM1_CHILD_STRETCHED_ZTOP_F32,
                "reference_dz_f32": GM1_CHILD_STRETCHED_DZ_REFERENCE_F32,
            }
        ),
        False,
    )


def bind_uniform_cartesian_sgs_tke_metrics(
    vertical: Any, closure_launch: Any, *, geometry: CartesianGridGeometry
) -> ChildStretchedMetricView:
    """Bind profiles from the exact uniform grid and closure-static producers.

    No host field reconstruction or preset relabelling occurs.  The closure
    launch owns the CUDA rds/rdsf arrays and the event after their production.
    """

    from cm1core.cuda_gm1_tilestream_stretched_closure import (
        CudaGm1TiledStretchedClosureLaunch, _operation_token, _require_vertical,
    )

    if not isinstance(geometry, CartesianGridGeometry):
        raise TypeError("geometry must be CartesianGridGeometry")
    if not isinstance(closure_launch, CudaGm1TiledStretchedClosureLaunch):
        raise TypeError("uniform metrics require the exact closure-static launch")
    receipt = closure_launch.receipt
    if (
        receipt.get("operation_token_sha256") != _operation_token(receipt)
        or receipt.get("grid_geometry") != geometry.record()
        or receipt.get("vertical_disposition") != "source-audited-flat-uniform-cartesian"
    ):
        raise ValueError("uniform metric producer geometry/receipt differs")
    cupy = importlib.import_module("cupy")
    _require_vertical(cupy, vertical, geometry.nz, allow_uniform_idealized_validation=True, geometry=geometry)
    view = ChildStretchedMetricView(
        MappingProxyType({
            "sigma": vertical.grid.sigma, "sigmaf": vertical.grid.sigmaf,
            "rds": closure_launch.workspace["rds"], "rdsf": closure_launch.workspace["rdsf"],
        }),
        closure_launch.ready_event,
        MappingProxyType({
            "schema": "gm1.uniform-cartesian-vertical-metrics/v1",
            "metric_abi": "gm1.cuda-cartesian-metrics-f32/v1",
            "grid_geometry": geometry.record(),
            "vertical_producer": dict(vertical.receipt),
            "static_producer_operation_token_sha256": receipt["operation_token_sha256"],
            "static_source_sha256": receipt["operator_source_sha256"]["static_metrics"],
            "profile_indexing": "direct CM1 indices 0:nk+1 and 0:nk+2",
            "ztop_f32": geometry.ztop_m, "reference_dz_f32": geometry.dz_m,
        }),
        False,
        geometry,
    )
    _validate_device_metrics(cupy, view, geometry.nz)
    return view


def _validate_device_metrics(
    cupy: Any, view: ChildStretchedMetricView, nk: int
) -> dict[str, Any]:
    expected = {
        "sigma": (nk + 2,),
        "sigmaf": (nk + 3,),
        "rds": (nk + 2,),
        "rdsf": (nk + 3,),
    }
    values: dict[str, Any] = {}
    for name, shape in expected.items():
        value = view.arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 with shape {shape}")
        values[name] = value
    if len({int(value.data.ptr) for value in values.values()}) != 4:
        raise ValueError("stretched vertical profiles must not alias")
    return values


def _validate_scalar(name: str, value: float, expected: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0 or value != expected:
        refuse(
            code="GM1-CHILD-STRETCHED-SGS-TKE-GEOMETRY",
            gate="child-stretched-sgs-tke",
            option=name,
            value=value,
            reason="the operator is sealed to the exact 50 m, 15 km-top child",
            remedy=f"use {name}={expected!r} or add an independent profile oracle",
        )
    return value


def build_cuda_gm1_child_stretched_sources(
    *, max_nk: int = CUDA_SGS_STRESS_MAX_NK
) -> tuple[str, str]:
    """Extend column storage only after validating both legacy source pins."""

    _validate_column_capacity(max_nk)
    if (
        not _source_pins_are_valid()
        or hashlib.sha256(CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE.encode()).hexdigest() != EXPECTED_STRETCHED_SGS_SOURCE_SHA256
        or hashlib.sha256(CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE.encode()).hexdigest() != EXPECTED_STRETCHED_TKE_SOURCE_SHA256
    ):
        raise RuntimeError("stretched SGS/TKE base sources differ from their pins")
    return (
        _resize_sgs_column_workspace(CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE, max_nk=max_nk),
        _resize_tke_column_workspace(CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE, max_nk=max_nk, declaration_count=1),
    )


def compile_cuda_gm1_child_stretched_sgs_tke(
    *, max_nk: int = CUDA_SGS_STRESS_MAX_NK
) -> CompiledChildStretchedSgsTke:
    sgs_source, tke_source = build_cuda_gm1_child_stretched_sources(max_nk=max_nk)
    cupy = importlib.import_module("cupy")
    props = cupy.cuda.runtime.getDeviceProperties(int(cupy.cuda.Device().id))
    major, minor = int(props["major"]), int(props["minor"])
    target = f"{major}{minor}"
    options = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    from .nvrtc_module import compile_nvrtc_module

    sgs_module = compile_nvrtc_module(
        cupy,
        code=sgs_source,
        options=options,
        name_expressions=CUDA_SGS_STRESS_KERNEL_NAMES,
        filename="gm1_child_stretched_sgs.cu",
    )
    tke_module = compile_nvrtc_module(
        cupy,
        code=tke_source,
        options=options,
        name_expressions=CUDA_TKE_EVOLUTION_KERNEL_NAMES,
        filename="gm1_child_stretched_tke.cu",
    )
    sgs_kernels = tuple(
        sgs_module.get_function(name) for name in CUDA_SGS_STRESS_KERNEL_NAMES
    )
    tke_kernels = tuple(
        tke_module.get_function(name) for name in CUDA_TKE_EVOLUTION_KERNEL_NAMES
    )
    device_name = props["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    return CompiledChildStretchedSgsTke(
        sgs_module,
        sgs_kernels,
        tke_module,
        tke_kernels,
        target,
        MappingProxyType(
            {
                "device_id": int(cupy.cuda.Device().id),
                "device_name": str(device_name),
                "compute_capability": f"{major}.{minor}",
                "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
                "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
            }
        ),
        max_nk,
        hashlib.sha256(sgs_source.encode()).hexdigest(),
        hashlib.sha256(tke_source.encode()).hexdigest(),
    )


def _require_compiled_capacity(compiled: CompiledChildStretchedSgsTke | None, nk: int) -> None:
    capacity = CUDA_SGS_STRESS_MAX_NK if compiled is None else compiled.max_nk
    _validate_column_capacity(capacity)
    if nk > capacity:
        raise ValueError(f"nk={nk} exceeds compiled SGS/TKE column workspace {capacity}; compile with max_nk >= nk")
    if compiled is not None:
        sgs_source, tke_source = build_cuda_gm1_child_stretched_sources(max_nk=capacity)
        if (
            compiled.sgs_source_sha256 != hashlib.sha256(sgs_source.encode()).hexdigest()
            or compiled.tke_source_sha256 != hashlib.sha256(tke_source.encode()).hexdigest()
        ):
            raise ValueError("compiled stretched SGS/TKE source hashes differ from column capacity")


def _capacity_receipt(compiled: CompiledChildStretchedSgsTke) -> dict[str, object]:
    return {
        "column_workspace_max_nk": compiled.max_nk,
        "column_workspace_source_sha256": {
            "sgs": compiled.sgs_source_sha256, "tke": compiled.tke_source_sha256,
        },
    } if compiled.max_nk != CUDA_SGS_STRESS_MAX_NK else {}


def _require_uniform_geometry(
    geometry: CartesianGridGeometry | None, *, ni: int, nj: int, nk: int,
    dx: float, dy: float, dz: float, ztop: float, enabled: bool,
) -> None:
    if geometry is None:
        return
    if not isinstance(geometry, CartesianGridGeometry):
        raise TypeError("geometry must be CartesianGridGeometry")
    if not enabled:
        raise ValueError("explicit Cartesian geometry requires uniform idealized mode")
    if (
        ni > geometry.nx or nj > geometry.ny or nk != geometry.nz
        or (dx, dy, dz, ztop) != (*geometry.spacing_m, geometry.ztop_m)
    ):
        raise ValueError("SGS/TKE window or spacing differs from explicit uniform geometry")


def _compiled(
    compiled: CompiledChildStretchedSgsTke | None,
) -> CompiledChildStretchedSgsTke:
    return compile_cuda_gm1_child_stretched_sgs_tke() if compiled is None else compiled


def _device_arrays(
    cupy: Any,
    arrays: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...]],
    names: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in names:
        if name not in arrays:
            raise ValueError(f"missing {label} array: {name}")
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
        values[name] = value
    if len({int(value.data.ptr) for value in values.values()}) != len(values):
        raise ValueError(f"{label} arrays must not alias")
    return values


_SGS_NAMES = tuple(sgs_stress_shapes(7, 7, 8))
_TKE_NAMES = tuple(tke_evolution_shapes(7, 7, 8))


def launch_cuda_gm1_child_stretched_sgs_stress(
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
    dt: float,
    stream: Any | None = None,
    compiled: CompiledChildStretchedSgsTke | None = None,
    selection: SgsStressSelection = DEFAULT_SGS_STRESS_SELECTION,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
    surface: SurfaceDragBinding | None = None,
    two_part: Any | None = None,
) -> CudaChildStretchedLaunch:
    _validate_sgs_selection(selection)
    _require_surface_binding(selection, surface)
    _validate_sgs_grid(ni, nj, nk)
    _require_compiled_capacity(compiled, nk)
    _require_uniform_geometry(geometry, ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz, ztop=ztop, enabled=allow_uniform_idealized_validation)
    if not isinstance(allow_uniform_idealized_validation, bool):
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if geometry is None and allow_uniform_idealized_validation and (
        not allow_compact_validation
        or nk != 75
        or ni > 300
        or nj > 300
        or (float(dx), float(dy), float(dz), float(ztop))
        != (50.0, 50.0, 200.0, 15000.0)
    ):
        raise ValueError(
            "uniform idealized SGS stress requires a compact window no larger than "
            "the exact 300x300x75, 50/50/200 m, 15 km grid"
        )
    dx = _validate_scalar("dx", dx, geometry.dx_m if geometry is not None else 50.0 if not allow_compact_validation else float(dx))
    dy = _validate_scalar("dy", dy, geometry.dy_m if geometry is not None else 50.0 if not allow_compact_validation else float(dy))
    dz = _validate_scalar(
        "dz", dz, geometry.dz_m if geometry is not None else 200.0 if allow_uniform_idealized_validation else 100.0
    )
    ztop = _validate_scalar(
        "ztop", ztop, geometry.ztop_m if geometry is not None else 15000.0 if not allow_compact_validation else float(ztop)
    )
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and > 0")
    _validate_square_spacing(dx, dy)
    if geometry is None and not allow_compact_validation and (ni, nj, nk) != (300, 300, 75):
        raise ValueError("real child SGS launch requires 300x300x75")
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    view = bind_gm1_child_stretched_sgs_tke_metrics(
        metrics, allow_compact_validation=allow_compact_validation, geometry=geometry
    )
    profiles = _validate_device_metrics(cupy, view, nk)
    shapes = sgs_stress_shapes(ni, nj, nk)
    values = _device_arrays(cupy, arrays, shapes, _SGS_NAMES, "SGS stress")
    compiled = _compiled(compiled)
    totals = (
        ni * nj * nk,
        (ni + 1) * (nj + 1) * nk,
        (ni + 1) * nj * (nk + 1),
        ni * (nj + 1) * (nk + 1),
        (ni + 1) * nj,
        ni * (nj + 1),
        ni * nj,
    )
    grids = tuple(
        ((n + CUDA_SGS_STRESS_THREADS - 1) // CUDA_SGS_STRESS_THREADS,) for n in totals
    )
    block = (CUDA_SGS_STRESS_THREADS,)
    u32, f32 = cupy.uint32, cupy.float32
    kernels = compiled.sgs_kernels
    with stream:
        if view.ready_event is not None:
            stream.wait_event(view.ready_event)
        kernels[0](
            grids[0],
            block,
            (
                values["s11"],
                values["s22"],
                values["s33"],
                values["kmh"],
                values["t11"],
                values["t22"],
                values["t33"],
                u32(ni),
                u32(nj),
                u32(nk),
            ),
            stream=stream,
        )
        kernels[1](
            grids[1],
            block,
            (values["s12"], values["kmh"], values["t12"], u32(ni), u32(nj), u32(nk)),
            stream=stream,
        )
        kernels[2](
            grids[2],
            block,
            (values["s13"], values["kmv"], values["t13"], u32(ni), u32(nj), u32(nk)),
            stream=stream,
        )
        kernels[3](
            grids[3],
            block,
            (values["s23"], values["kmv"], values["t23"], u32(ni), u32(nj), u32(nk)),
            stream=stream,
        )
        # src/turb.F:4417.  gettau builds the interior stresses, then calls
        # sfcstress, which replaces t13/t23 at k=1 before the momentum stress
        # divergence reads them (src/turbtend.F:1454, src/turbtend.F:2224).
        surface_kernels: tuple[str, ...] = ()
        if surface is not None:
            surface_kernels = enqueue_surface_stress(
                surface,
                {name: values[name] for name in ("t13", "t23", "rf", "mf", "kmv", "u3d", "v3d")},
                ni=ni,
                nj=nj,
                nk=nk,
                stream=stream,
            )
        suffix = (
            u32(ni),
            u32(nj),
            u32(nk),
            f32(dx),
            f32(dz),
            f32(dt),
            profiles["sigma"],
            profiles["sigmaf"],
            profiles["rds"],
            profiles["rdsf"],
            f32(ztop),
        )
        # README.namelist:334-352.  sgsmodel=3/4 run turbuz/turbvz's dot2p
        # block (src/turbtend.F:1767-1802, :2424-2459) inside the same column;
        # without a binding these are the audited kernels and arguments.
        u_kernel, v_kernel = kernels[4], kernels[5]
        u_extra: tuple[Any, ...] = ()
        v_extra: tuple[Any, ...] = ()
        if two_part is not None:
            ntwk = int(two_part.ntwk)
            if not 2 <= ntwk < nk:
                raise ValueError("two-part ntwk must satisfy 2 <= ntwk < nk")
            for name, value, shape in (
                ("tfu", two_part.tfu, (nk + 1,)), ("tfv", two_part.tfv, (nk + 1,)),
                ("u2pt", two_part.u2pt, (ntwk, nj, ni + 1)), ("v2pt", two_part.v2pt, (ntwk, nj + 1, ni)),
            ):
                if (
                    not isinstance(value, cupy.ndarray) or value.dtype != cupy.float32
                    or value.shape != shape or not value.flags.c_contiguous
                ):
                    raise ValueError(f"two-part {name} must be C-order CUDA float32 with shape {shape}")
            u_kernel, v_kernel = two_part.kernels
            u_extra = (two_part.tfu, two_part.u2pt, u32(ntwk))
            v_extra = (two_part.tfv, two_part.v2pt, u32(ntwk))
        u_kernel(
            grids[4],
            block,
            (
                values["t11"],
                values["t12"],
                values["t13"],
                values["kmv"],
                values["rho"],
                values["rf"],
                values["mh"],
                values["mf"],
                values["gz"],
                values["rgz"],
                values["gzu"],
                values["gxu"],
                values["u3d"],
                values["w3d"],
                values["uten"],
                *suffix,
                *u_extra,
            ),
            stream=stream,
        )
        v_kernel(
            grids[5],
            block,
            (
                values["t12"],
                values["t22"],
                values["t23"],
                values["kmv"],
                values["rho"],
                values["rf"],
                values["mh"],
                values["mf"],
                values["gz"],
                values["rgz"],
                values["gzv"],
                values["gyv"],
                values["v3d"],
                values["w3d"],
                values["vten"],
                *suffix,
                *v_extra,
            ),
            stream=stream,
        )
        kernels[6](
            grids[6],
            block,
            (
                values["t13"],
                values["t23"],
                values["t33"],
                values["kmh"],
                values["rho"],
                values["rf"],
                values["mh"],
                values["mf"],
                values["gz"],
                values["rgzu"],
                values["rgzv"],
                values["w3d"],
                values["wten"],
                *suffix,
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return CudaChildStretchedLaunch(
        event,
        MappingProxyType(
            {
                "schema": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA,
                "abi": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI,
                "operator": "sgs-stress-divergence",
                "source_sha256": compiled.sgs_source_sha256,
                **_capacity_receipt(compiled),
                "base_source_sha256": CUDA_SGS_STRESS_SOURCE_SHA256,
                "metrics": dict(view.identity),
                "domain": {"ni": ni, "nj": nj, "nk": nk},
                "spacing_m": {"dx": dx, "dy": dy, "reference_dz": dz, "ztop": ztop},
                "vertical_disposition": (
                    "source-audited-flat-uniform-cartesian"
                    if geometry is not None
                    else "source-audited-flat-idealized-uniform-200m"
                    if allow_uniform_idealized_validation
                    else "sealed-stretched-child-or-compact-operator"
                ),
                "dt_seconds": float(dt),
                "same_stream_no_host_sync": True,
                "lower_boundary": (
                    {"bbc": 1, "disposition": "free-slip copy of k=2 (src/turb.F:4536-4545)"}
                    if surface is None
                    else {
                        **surface.record(),
                        "kernel_names": list(surface_kernels),
                    }
                ),
                "claim_limit": GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT,
                **({"two_part": two_part.record()} if two_part is not None else {}),
            }
        ),
        compiled,
    )


def launch_cuda_gm1_child_stretched_tke_diffusion(
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
    dt: float,
    stream: Any | None = None,
    compiled: CompiledChildStretchedSgsTke | None = None,
    selection: TkeEvolutionSelection = DEFAULT_TKE_EVOLUTION_SELECTION,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
) -> CudaChildStretchedLaunch:
    _validate_tke_selection(selection)
    _validate_tke_grid(ni, nj, nk)
    _require_compiled_capacity(compiled, nk)
    _require_uniform_geometry(geometry, ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz, ztop=ztop, enabled=allow_uniform_idealized_validation)
    dx = _validate_scalar("dx", dx, geometry.dx_m if geometry is not None else 50.0 if not allow_compact_validation else float(dx))
    dy = _validate_scalar("dy", dy, geometry.dy_m if geometry is not None else 50.0 if not allow_compact_validation else float(dy))
    dz = _validate_scalar("dz", dz, geometry.dz_m if geometry is not None else 100.0)
    ztop = _validate_scalar(
        "ztop", ztop, geometry.ztop_m if geometry is not None else 15000.0 if not allow_compact_validation else float(ztop)
    )
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if geometry is None and not allow_compact_validation and (ni, nj, nk) != (300, 300, 75):
        raise ValueError("real child TKE launch requires 300x300x75")
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    view = bind_gm1_child_stretched_sgs_tke_metrics(
        metrics, allow_compact_validation=allow_compact_validation, geometry=geometry
    )
    profiles = _validate_device_metrics(cupy, view, nk)
    values = _device_arrays(
        cupy, arrays, tke_evolution_shapes(ni, nj, nk), _TKE_NAMES, "TKE evolution"
    )
    compiled = _compiled(compiled)
    block = (CUDA_TKE_EVOLUTION_THREADS,)
    grid = ((ni * nj + CUDA_TKE_EVOLUTION_THREADS - 1) // CUDA_TKE_EVOLUTION_THREADS,)
    with stream:
        if view.ready_event is not None:
            stream.wait_event(view.ready_event)
        compiled.tke_kernels[0](
            grid,
            block,
            (
                values["tke3d"],
                values["kmh"],
                values["kmv"],
                values["rho"],
                values["rf"],
                values["mh"],
                values["mf"],
                values["gz"],
                values["rgz"],
                values["rgzu"],
                values["rgzv"],
                values["tketen"],
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
                cupy.float32(dx),
                cupy.float32(dy),
                cupy.float32(dz),
                cupy.float32(dt),
                profiles["sigma"],
                profiles["rds"],
                cupy.float32(ztop),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return CudaChildStretchedLaunch(
        event,
        MappingProxyType(
            {
                "schema": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA,
                "abi": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI,
                "operator": "turbt-diffusion",
                "source_sha256": compiled.tke_source_sha256,
                **_capacity_receipt(compiled),
                "base_source_sha256": CUDA_TKE_EVOLUTION_SOURCE_SHA256,
                "metrics": dict(view.identity),
                "same_stream_no_host_sync": True,
                "claim_limit": GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT,
            }
        ),
        compiled,
    )


def launch_cuda_gm1_child_stretched_tke_rk_stage(
    arrays: Mapping[str, Any],
    *,
    workspace: Any,
    metrics: Any,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dx: float,
    dy: float,
    dz: float,
    ztop: float,
    dt: float,
    stream: Any | None = None,
    compiled: CompiledChildStretchedSgsTke | None = None,
    selection: TkeEvolutionSelection = DEFAULT_TKE_EVOLUTION_SELECTION,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
    geometry: CartesianGridGeometry | None = None,
) -> CudaChildStretchedLaunch:
    _validate_tke_selection(selection)
    _validate_tke_grid(ni, nj, nk)
    _require_compiled_capacity(compiled, nk)
    _require_uniform_geometry(geometry, ni=ni, nj=nj, nk=nk, dx=dx, dy=dy, dz=dz, ztop=ztop, enabled=allow_uniform_idealized_validation)
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be exactly 1, 2, or 3")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if geometry is None and allow_uniform_idealized_validation and (
        not allow_compact_validation
        or nk != 75
        or ni > 300
        or nj > 300
        or (float(dx), float(dy), float(dz), float(ztop))
        != (50.0, 50.0, 200.0, 15000.0)
    ):
        raise ValueError(
            "uniform idealized TKE RK requires a compact window no larger than "
            "the exact 300x300x75, 50/50/200 m, 15 km grid"
        )
    dx = _validate_scalar("dx", dx, geometry.dx_m if geometry is not None else 50.0 if not allow_compact_validation else float(dx))
    dy = _validate_scalar("dy", dy, geometry.dy_m if geometry is not None else 50.0 if not allow_compact_validation else float(dy))
    dz = _validate_scalar(
        "dz", dz, geometry.dz_m if geometry is not None else 200.0 if allow_uniform_idealized_validation else 100.0
    )
    _validate_scalar(
        "ztop", ztop, geometry.ztop_m if geometry is not None else 15000.0 if not allow_compact_validation else float(ztop)
    )
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("dt must be finite and > 0")
    if geometry is None and not allow_compact_validation and (ni, nj, nk) != (300, 300, 75):
        raise ValueError("real child TKE launch requires 300x300x75")
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    view = bind_gm1_child_stretched_sgs_tke_metrics(
        metrics, allow_compact_validation=allow_compact_validation, geometry=geometry
    )
    profiles = _validate_device_metrics(cupy, view, nk)
    shapes = tke_evolution_shapes(ni, nj, nk)
    values = _device_arrays(cupy, arrays, shapes, _TKE_NAMES, "TKE evolution")
    if (
        not isinstance(workspace, cupy.ndarray)
        or workspace.dtype != cupy.float32
        or workspace.shape != shapes["tke3d"]
        or not workspace.flags.c_contiguous
    ):
        raise ValueError("workspace must be C-order CUDA float32 tke3d shape")
    if int(workspace.data.ptr) in {int(v.data.ptr) for v in values.values()}:
        raise ValueError("TKE stage workspace aliases a live carrier")
    compiled = _compiled(compiled)
    words = (nk + 1) * nj * ni
    block = (CUDA_TKE_EVOLUTION_THREADS,)
    grid = ((words + CUDA_TKE_EVOLUTION_THREADS - 1) // CUDA_TKE_EVOLUTION_THREADS,)
    dttmp = cupy.float32(cupy.float32(dt) / cupy.float32(4 - nrk))
    weps = float(cupy.float32(cupy.float32(100) * cupy.float32(1e-18)).item())
    with stream:
        if view.ready_event is not None:
            stream.wait_event(view.ready_event)
        compiled.tke_kernels[1](
            grid,
            block,
            (
                values["tke3d"],
                values["tkea"],
                values["tketen"],
                values["c1"],
                values["c2"],
                values["rrf0"],
                values["gz"],
                values["rru"],
                values["rrv"],
                values["rrw"],
                values["divx"],
                workspace,
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
                cupy.uint32(CUDA_TKE_EVOLUTION_NGXY),
                cupy.float32(dx),
                cupy.float32(dy),
                cupy.float32(dz),
                dttmp,
                cupy.uint32(1 if nrk == 3 else 0),
                cupy.float64(weps),
                profiles["rds"],
            ),
            stream=stream,
        )
        compiled.tke_kernels[3](
            grid, block, (workspace, values["tke3d"], cupy.uint64(words)), stream=stream
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return CudaChildStretchedLaunch(
        event,
        MappingProxyType(
            {
                "schema": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA,
                "abi": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI,
                "operator": "tke-advw-rk-stage",
                "nrk": nrk,
                "uses_weno": nrk == 3,
                "mass_flux_contract": f"rru/rrv/rrw/divx rebuilt after post-acoustic getdivx for nrk={nrk}",
                "source_sha256": compiled.tke_source_sha256,
                **_capacity_receipt(compiled),
                "metrics": dict(view.identity),
                "vertical_disposition": (
                    "source-audited-flat-uniform-cartesian"
                    if geometry is not None
                    else "source-audited-flat-idealized-uniform-200m"
                    if allow_uniform_idealized_validation
                    else "sealed-stretched-child"
                ),
                "same_stream_no_host_sync": True,
                "claim_limit": GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT,
            }
        ),
        compiled,
    )


def launch_cuda_gm1_child_stretched_tke_commit(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    completed_nrk: int,
    stream: Any | None = None,
    compiled: CompiledChildStretchedSgsTke | None = None,
    geometry: CartesianGridGeometry | None = None,
) -> CudaChildStretchedLaunch:
    if completed_nrk != 3:
        raise ValueError("TKE commit requires completed_nrk=3")
    _validate_tke_grid(ni, nj, nk)
    _require_compiled_capacity(compiled, nk)
    if geometry is not None and (
        not isinstance(geometry, CartesianGridGeometry)
        or ni > geometry.nx or nj > geometry.ny or nk != geometry.nz
    ):
        raise ValueError("TKE commit window differs from explicit geometry")
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    values = _device_arrays(
        cupy, arrays, tke_evolution_shapes(ni, nj, nk), _TKE_NAMES, "TKE evolution"
    )
    compiled = _compiled(compiled)
    words = (nk + 1) * nj * ni
    block = (CUDA_TKE_EVOLUTION_THREADS,)
    grid = ((words + CUDA_TKE_EVOLUTION_THREADS - 1) // CUDA_TKE_EVOLUTION_THREADS,)
    with stream:
        compiled.tke_kernels[2](
            grid,
            block,
            (values["tke3d"], values["tke3d"], values["tkea"], cupy.uint64(words)),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return CudaChildStretchedLaunch(
        event,
        MappingProxyType(
            {
                "schema": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA,
                "abi": CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI,
                "operator": "tke-solve3-commit",
                **_capacity_receipt(compiled),
                "completed_nrk": 3,
                "source_anchor": SOURCE_ANCHORS["tke_commit"],
                "same_stream_no_host_sync": True,
                "claim_limit": GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT,
            }
        ),
        compiled,
    )


def _source_pins_are_valid() -> bool:
    return (
        hashlib.sha256(CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE.encode()).hexdigest()
        == CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256
        and hashlib.sha256(CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE.encode()).hexdigest()
        == CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256
        and CUDA_SGS_STRESS_MAX_NK == CUDA_TKE_EVOLUTION_MAX_NK == 75
    )


__all__ = (
    "CUDA_GM1_CHILD_STRETCHED_SGS_SOURCE_SHA256",
    "CUDA_GM1_CHILD_STRETCHED_SGS_TKE_ABI",
    "CUDA_GM1_CHILD_STRETCHED_SGS_TKE_SCHEMA",
    "CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256",
    "CARTESIAN_VERTICAL_METRIC_SCHEMA",
    "COMMON_STRETCHED_METRICS_MODULE_SHA256",
    "EXPECTED_STRETCHED_SGS_SOURCE_SHA256",
    "EXPECTED_STRETCHED_TKE_SOURCE_SHA256",
    "ChildStretchedMetricView",
    "CompiledChildStretchedSgsTke",
    "CudaChildStretchedLaunch",
    "GM1_CHILD_STRETCHED_METRIC_ABI",
    "GM1_CHILD_STRETCHED_SGS_TKE_CLAIM_LIMIT",
    "SOURCE_ANCHORS",
    "bind_gm1_child_stretched_sgs_tke_metrics",
    "bind_uniform_cartesian_sgs_tke_metrics",
    "build_cuda_gm1_child_stretched_sources",
    "compile_cuda_gm1_child_stretched_sgs_tke",
    "launch_cuda_gm1_child_stretched_sgs_stress",
    "launch_cuda_gm1_child_stretched_tke_commit",
    "launch_cuda_gm1_child_stretched_tke_diffusion",
    "launch_cuda_gm1_child_stretched_tke_rk_stage",
)
