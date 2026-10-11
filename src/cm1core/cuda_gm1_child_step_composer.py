"""One complete source-ordered large step for the real 50 m GM1 child.

The existing GM1 forecast coordinator is intentionally sealed to the older
100 m initialized-product ABI.  The real 50 m child differs in three material
ways: its hydrostatic/base state is terrain varying, its Rayleigh base wind is
three-dimensional on staggered grids, and its five independently hashed Rust
publications must remain separately auditable.  This module composes the
already admitted operators without relabelling the child as that older ABI.

Python performs strict loading, device transport, launch ordering, and receipt
construction.  It does not interpolate, translate, smooth, or otherwise alter
the atmospheric product on the host.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from cm1core.arwen_wsm6_component import (
    ARWEN_WSM6_COMPONENT_ID,
    WSM6_COMPONENT_PERSISTENCE,
    create_arwen_wsm6_component,
)
from cm1core.cuda_apmasscon import compile_cuda_apmasscon, launch_cuda_apmasscon
from cm1core.cuda_gm1_child_large_step import (
    REAL_GM1_CHILD_FULL_STATE_PINS,
    Gm1ChildScalarMoistRKBinding,
    LoadedGm1ChildFullStateEnvelope,
    compile_cuda_gm1_child_terrain_rayleigh,
    create_cuda_gm1_child_scalar_moist_rk_state,
    launch_cuda_gm1_child_terrain_rayleigh,
    load_gm1_child_full_state_envelope,
)
from cm1core.cuda_gm1_child_nm_def import (
    load_gm1_child_auxiliary,
)
from cm1core.cuda_gm1_child_terrain import (
    launch_cuda_gm1_child_terrain_metrics,
    load_gm1_child_terrain_export,
)
from cm1core.cuda_gm1_child_stretched_metrics import (
    GM1_CHILD_STRETCHED_ZTOP_F32,
    Gm1ChildStretchedMetrics,
    bind_gm1_child_stretched_metrics,
)
from cm1core.cuda_gm1_child_stretched_moist import (
    compile_cuda_gm1_child_stretched_moist,
    launch_cuda_gm1_child_stretched_post_acoustic,
)
from cm1core.cuda_gm1_child_stretched_momentum import (
    compile_cuda_gm1_child_stretched_momentum,
    launch_cuda_gm1_child_stretched_momentum5,
)
from cm1core.cuda_gm1_child_stretched_scalar import (
    compile_cuda_gm1_child_stretched_scalar_advection,
    compile_cuda_gm1_child_stretched_scalar_pdef,
    launch_cuda_gm1_child_stretched_pdef_moisture,
    launch_cuda_gm1_child_stretched_scalar_advection5,
)
from cm1core.cuda_gm1_child_stretched_sgs_tke import (
    bind_gm1_child_stretched_sgs_tke_metrics,
    compile_cuda_gm1_child_stretched_sgs_tke,
    launch_cuda_gm1_child_stretched_sgs_stress,
    launch_cuda_gm1_child_stretched_tke_commit,
    launch_cuda_gm1_child_stretched_tke_diffusion,
    launch_cuda_gm1_child_stretched_tke_rk_stage,
)
from cm1core.cuda_gm1_child_post_final_rk import (
    CudaGm1ChildPostFinalRkState,
    Gm1ChildPostFinalRkCadence,
    create_cuda_gm1_child_post_final_rk_state,
    launch_cuda_gm1_child_post_final_rk,
    require_child_post_final_rk_operator_sources,
)
from cm1core.cuda_gm1_forecast import (
    GM1_FORECAST_THREADS,
    Gm1ForecastSelection,
    Gm1Solve1Carriers,
    _initial_mass,
    _seed_momentum_and_stage_boundary,
    compile_cuda_gm1_forecast,
)
from cm1core.cuda_initialized_product_terrain_open import (
    CudaInitializedTerrainOpenState,
    compile_cuda_initialized_terrain_open,
)
from cm1core.cuda_lateral_closure import compile_cuda_lateral_closure
from cm1core.cuda_moist_stage_bridge import (
    begin_cuda_moist_stage,
    compile_cuda_moist_stage_bridge,
    create_cuda_moist_stage_bridge_state,
    launch_cuda_eqtset2_stage_terms,
    launch_cuda_moist_buoyancy,
)
from cm1core.cuda_momentum_rk import compile_cuda_momentum_rk
from cm1core.cuda_gm1_child_psolver3_open import (
    compile_cuda_gm1_child_psolver3_open,
    launch_cuda_gm1_child_psolver3_open_stage,
)
from cm1core.cuda_psolver3_terrain_open import terrain_open_sound_shapes
from cm1core.cuda_scalar_moist_rk import (
    CUDA_SCALAR_MOIST_RK_THREADS,
    _fill_scalar_open_halos,
    begin_cuda_scalar_moist_rk_step,
    final_stage_moisture_advection_mode,
    final_stage_moisture_advection_record,
    moisture_weps_f64,
    scalar_moist_rk_numq,
    scalar_moist_rk_stage_timestep_f32,
    theta_weps_f64,
)
from cm1core.cuda_rayleigh_held import cpu_build_rayleigh_coefficients
from cm1core.cuda_sgs_stress_tendency import (
    sgs_stress_shapes,
)
from cm1core.cuda_sgs_tke import (
    MUTABLE_ARRAY_NAMES as SGS_TKE_MUTABLE_ARRAY_NAMES,
    compile_cuda_sgs_tke,
    launch_cuda_sgs_tke,
)
from cm1core.cuda_sound3_open_boundary import CM1_OPEN_NGXY, compile_cuda_sound3_open_boundary
from cm1core.cuda_terrain_open_boundary import compile_cuda_terrain_open_boundary
from cm1core.cuda_wsm6_pdefq import (
    compile_cuda_wsm6_pdefq,
    create_cuda_wsm6_pdefq_workspace,
)
from cm1core.eqtset2_energy_terms import Eqtset2IdissCarriers, Eqtset2IdissConsumer
from cm1core.eqtset2_wsm6_bridge import (
    EQTSET2_HELD_FIELDS,
    EQTSET2_RESTART_FIELDS,
    Eqtset2Wsm6Bridge,
    Eqtset2Wsm6Carriers,
)
from cm1core.gm1_child_atmosphere_admission import load_gm1_child_atmosphere
from cm1core.gm1_child_stretched_closure_v3 import (
    SCHEMA as GM1_CHILD_CLOSURE_SCHEMA,
    CudaChildStretchedClosure,
    LoadedChildStretchedClosure,
    load_gm1_child_stretched_closure,
    refuse_uniform_z_closure,
    upload_gm1_child_stretched_closure,
)


GM1_CHILD_STEP_SCHEMA = "gm1.cuda-real-50m-child-large-step/v1"
GM1_CHILD_STEP_ABI = "gm1.cuda-real-50m-child-large-step-f32/v1"
GM1_CHILD_FINAL_DYNAMICS_SYNC_SCHEMA = "gm1.cuda-child-final-dynamics-sync/v1"
GM1_CHILD_FINAL_DYNAMICS_SYNC_ABI = "gm1.cuda-child-final-dynamics-sync-f32/v1"
GM1_CHILD_FINAL_DYNAMICS_SYNC_OPERATOR_KEY = "child.dynamics.final-product-sync"
GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE = """gm1.cuda-child-final-dynamics-sync-f32/v1
u3d <- prepared.u3d[1:nz+1,ng:ng+ny,ng:ng+nx+1]
v3d <- prepared.v3d[1:nz+1,ng:ng+ny+1,ng:ng+nx]
w3d <- prepared.w3d[1:nz+2,ng:ng+ny,ng:ng+nx]
pp3d <- prepared.pp3d[1:nz+1,1:ny+1,1:nx+1]
same explicit CUDA stream; after completed RK3; before remove/EOS/apmasscon
"""
GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256 = (
    "90063f4944a9abe02d2eb349838e38c6656abe40e2dda3f816e76ec6e973dd85"
)
GM1_CHILD_SOLVE3_RHO_BOUNDARY_SCHEMA = "gm1.cuda-child-solve3-rho-boundary/v1"
GM1_CHILD_SOLVE3_RHO_BOUNDARY_ABI = "gm1.cuda-child-solve3-rho-boundary-f32/v1"
GM1_CHILD_SOLVE3_RHO_BOUNDARY_OPERATOR_KEY = (
    "child.dynamics.solve3-rho-boundary"
)
GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE = """gm1.cuda-child-solve3-rho-boundary-f32/v1
prepared.rho[1:nz+1,1:ny+1,1:nx+1] <- product.rho
open bcp west/east then south/north constant extension on one horizontal halo
same explicit CUDA stream; solve3.F:309-312; after ppx bcp; before held recapture
"""
GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256 = (
    "c885efc95331b7002654b53e3835614918281becb8971973a266cadab76c05c7"
)
GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SCHEMA = (
    "gm1.cuda-child-completed-closure-refresh/v1"
)
GM1_CHILD_COMPLETED_CLOSURE_REFRESH_ABI = (
    "gm1.cuda-child-completed-step-closure-refresh-f32/v1"
)
GM1_CHILD_COMPLETED_CLOSURE_REFRESH_OPERATOR_KEY = (
    "child.closure.completed-step-refresh"
)
GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE = """gm1.cuda-child-completed-step-closure-refresh-f32/v1
live stretched calcnm -> live stretched calcdef -> local persistent-tkea tkekm coefficients
clear dissten/tketen before coefficient rebuild; copy refreshed tkea to tke3d
same explicit CUDA stream; after solve3 density; before checkpoint/history publication
"""
GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256 = (
    "40e1da5228e199bdb9ef13d38b11083cd3a95cd0a20818ee6c87202a672ed934"
)
GM1_CHILD_SELECTION = Gm1ForecastSelection(
    nx=300,
    ny=300,
    nz=75,
    dx_m=50.0,
    dy_m=50.0,
    # stretch_z=1: dz_m is param.F:8170's metric reference and the top is
    # this coordinate's own sigmaf(nk+1), not dz*nk.
    dz_m=100.0,
    ztop_m=GM1_CHILD_STRETCHED_ZTOP_F32,
    dt_s=0.5,
    acoustic_substeps=6,
)

# The directory names of the first child this project composed.  They are the
# default location a standalone tool looks in; they are NOT how the composer
# finds a sibling.  The composed envelope records the directory name of every
# artifact it was built from, and `_sibling` resolves exactly that name beside
# the publication, so a child composed from another parent is found where its
# own envelope says it is instead of where the first one happened to live.
GM1_CHILD_ROOT_NAME = "gm1-t300-child-full-state-v1"
GM1_CHILD_ATMOSPHERE_NAME = "gm1-t300-child-atmosphere-v1"
GM1_CHILD_AUXILIARY_NAME = "gm1-t300-child-auxiliary-v1"
GM1_CHILD_TERRAIN_NAME = "gm1-t300-child-terrain-v1"
GM1_CHILD_STATIC_NAME = "gm1-t300-child-static-heights-v1"
# The child's fresh subgrid closure.  The superseded
# `gm1-t300-child-closure-v2` publication is NOT consumed: this project's own
# evidence (evidence/validation/gm1-t300-child-stretched-closure-v3) records its
# defv/defh carriers as byte-identical to the sealed uniform-z negative control,
# so the fresh km/kh and TKE the child would start from were diagnosed for a
# vertical grid the child does not have.  The corrected resident oracle is v3.
GM1_CHILD_CLOSURE_NAME = "gm1-t300-child-stretched-closure-v3"

# These remain explicit refusal tokens until the collaborating operator lanes
# publish immutable commits.  The source hashes alone are insufficient: an
# uncommitted shared-worktree candidate must never become production science.
GM1_CHILD_STRETCHED_NM_DEF_COMMIT = "a7d4e24eccbf8256211406680bd3030bcd45e7b3"
GM1_CHILD_STRETCHED_NM_DEF_SOURCE_SHA256 = (
    "3114457a038061913bf760be0a3a6b24c1e56d753f249d77d224af59a7a987fd"
)
# The module's compile now loads an NVRTC cubin built with ``--ftz=false``
# instead of a ``cupy.RawModule``, which CuPy compiles with ``-ftz=true``
# appended and which therefore wrote a signed zero where CM1 keeps a binary32
# sub-normal (``cm1core.nvrtc_module``, ``CM1-RISK-REGISTER.md`` R-020).  The
# module previously hashed to
# f475c4793ea9a15bd04a1d19d9c878d10d0814fa952ae81823b5196f33f33181.  The
# embedded CUDA source and its own pin above are untouched: only the compile
# route changed.
GM1_CHILD_STRETCHED_NM_DEF_MODULE_SHA256 = (
    "baccb1b16e859151e743413b0254fc0b7b41e78e73295053bf823345cd0b5f49"
)
GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_COMMIT = (
    "4c7e294bdef4774b2ac50d4f78cba2b90e2b277a"
)
GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256 = (
    "7f57b86da8aaf2f3d80a61e1f7dfada465cc2878f6d5aeaf6fccb5a6e3950506"
)
# The module gained CM1's ``numq`` on its shape function so a double-moment
# axis diffuses every scalar it transports (src/solve1.F:1213-1231); the
# embedded CUDA source and its own pin are untouched, and the default argument
# keeps every existing caller on the six-scalar inventory.  The module
# previously hashed to
# 9118a1d7cad016a40d4a2f43eca3bf0cd7e795ed53493a09419255b2322fa26a.
# 11f3ac3 then merged that numq change (4b3f004, 9951ae1f...) with 4c7e294's
# sfcmodel=1 lower scalar flux argument (d71c8a93..., whose source pin above is
# the 7f57b86d... this module still embeds) without recomputing this pin, so
# the frozen solve1 composition refused the module both parents' changes had
# produced.  That merged module (9ce1629c...) still looped range(6) in its
# launcher, so a wider axis was refused by shape; the launcher now diffuses the
# qa axis it is handed, n=1,numq, with the same kernels, arguments and order
# for six rows (e443be16...), and its receipt names each row by the caller's
# CM1 qname rather than a q3d[n] placeholder.  The pin below is that module's
# bytes; the embedded CUDA source pin is unchanged.
GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_MODULE_SHA256 = (
    "31b707b35a4a3bf5cca36964f315894f13a110ea39d1025f5f251fbc88939581"
)

SOURCE_ANCHORS = {
    "solve1": "src/solve1.F:317-383,414-454,917-1001,1115-1240,1696-1777",
    "solve2": "src/solve2.F:490-627,1277-1803,1973-2030",
    "solve3": "src/solve3.F:270-337,548-680",
    "microphysics": "src/cm1.F:3405-3427",
    "terrain_base_wind": "src/base.F:674-723",
    "scalar_turbulence": "src/turb.F selected SGS=1/doimpl=1 scalar calls",
    "live_nm_def_and_coefficients": "src/turb.F:1326-1413",
    "authority_tarball_sha256": (
        "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
    ),
}


@dataclass(frozen=True, slots=True)
class _Gm1ChildSolve1OperatorFunctions:
    launch_nm_def: Callable[..., Any]
    launch_scalar_turbulence: Callable[..., Any]


def _load_gm1_child_solve1_operator_functions() -> _Gm1ChildSolve1OperatorFunctions:
    missing = []
    if not GM1_CHILD_STRETCHED_NM_DEF_COMMIT:
        missing.append("stretched live calcnm/calcdef commit")
    if not GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_COMMIT:
        missing.append("stretched scalar-turbulence commit")
    if missing:
        raise RuntimeError(
            "real child solve1 remains fail-closed pending " + ", ".join(missing)
        )
    nm_def = importlib.import_module("cm1core.cuda_gm1_child_stretched_nm_def")
    scalar = importlib.import_module(
        "cm1core.cuda_gm1_child_stretched_scalar_turbulence"
    )
    if (
        nm_def.CUDA_GM1_CHILD_STRETCHED_CALCDEF_SOURCE_SHA256
        != GM1_CHILD_STRETCHED_NM_DEF_SOURCE_SHA256
    ):
        raise RuntimeError("installed stretched live nm/def source differs from its pin")
    if (
        scalar.CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256
        != GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256
    ):
        raise RuntimeError(
            "installed stretched scalar-turbulence source differs from its pin"
        )
    if _hash_file(Path(nm_def.__file__)) != GM1_CHILD_STRETCHED_NM_DEF_MODULE_SHA256:
        raise RuntimeError("installed stretched live nm/def module differs from its pin")
    if (
        _hash_file(Path(scalar.__file__))
        != GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_MODULE_SHA256
    ):
        raise RuntimeError("installed stretched scalar-turbulence module differs from its pin")
    return _Gm1ChildSolve1OperatorFunctions(
        nm_def.launch_cuda_gm1_child_stretched_nm_def_refresh,
        scalar.launch_cuda_gm1_child_stretched_scalar_turbulence,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{path.name} differs from its child-step SHA-256 pin")
    value = json.loads(raw)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if canonical != raw or not isinstance(value, dict):
        raise ValueError(f"{path.name} is not one canonical JSON object")
    return value


@dataclass(frozen=True, slots=True)
class Gm1ChildProductIdentity:
    manifest_sha256: str
    payload_sha256: str
    # The fresh-closure arena this child product was initialized from.  It is
    # recorded per product rather than defaulted to one historical artifact, so
    # a resume receipt names the closure the run actually consumed.
    source_artifact_sha256: str
    # The full-state envelope this product was composed from.  It is required,
    # not defaulted to one historical publication: a receipt that paired a
    # freshly uploaded closure arena with the manifest of the envelope the
    # project first composed would state that the child was initialized from
    # artifacts the run never opened, and the substitution would be invisible.
    source_contract_sha256: str
    crop_count_yx: tuple[int, int] = (300, 300)
    crop_origin_yx: tuple[int, int] = (0, 0)
    is_full_domain: bool = True


@dataclass(slots=True)
class CudaGm1ChildProduct:
    fields: dict[str, Any]
    export: Gm1ChildProductIdentity
    stream: Any
    device_id: int
    ready_event: Any

    @property
    def stream_pointer(self) -> int:
        return int(self.stream.ptr)

    def field(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError as error:
            raise KeyError(f"unknown real child CUDA carrier {name!r}") from error


@dataclass(slots=True)
class CudaGm1ChildWsm6Transition:
    bridge: Eqtset2Wsm6Bridge
    carriers: Eqtset2Wsm6Carriers
    completed_transitions: int = 0
    last_result: Any | None = None

    def apply(
        self,
        *,
        final_rk: bool = True,
        refl_10cm_due: bool = False,
        pre_component_hook: Any | None = None,
        post_component_pre_recapture_hook: Any | None = None,
    ) -> Any:
        self.last_result = self.bridge.apply(
            self.carriers,
            final_rk=final_rk,
            refl_10cm_due=refl_10cm_due,
            pre_component_hook=pre_component_hook,
            post_component_pre_recapture_hook=post_component_pre_recapture_hook,
        )
        self.completed_transitions += 1
        return self.last_result

    def receipt_dict(self) -> dict[str, Any]:
        return self.bridge.receipt_dict()


@dataclass(slots=True)
class CudaGm1ChildStretchedClosureState:
    """Live solve1/TKE carriers for the stretched child.

    This deliberately is not ``CudaGm1ClosureState``.  The generic closure
    object owns uniform-z calcdef, SGS, scalar-turbulence, and TKE launches;
    admitting it here would silently reintroduce the invalid ``dz`` fallback.
    """

    rayleigh: dict[str, Any]
    sgs: dict[str, Any]
    stress: dict[str, Any]
    tke: dict[str, Any]
    tke_workspace: Any
    idiss: Eqtset2IdissConsumer
    rayleigh_compiled: Any
    sgs_compiled: Any
    sgs_tke_compiled: Any
    sgs_tke_metrics: Any
    ready_event: Any
    refresh_index: int = 1
    completed_stage: int = 0
    retained: list[Any] = field(default_factory=list)
    last_nm_def: Any | None = None
    last_scalar_turbulence: Any | None = None
    nm_def_compiled_calcdef: Any | None = None
    nm_def_compiled_calcnm: Any | None = None
    scalar_turbulence_compiled: Any | None = None
    scalar_turbulence_workspace: Any | None = None


@dataclass(frozen=True, slots=True)
class CudaGm1ChildFinalDynamicsSync:
    """Same-stream proof that final padded RK3 dynamics reached the product."""

    ready_event: Any
    step_index: int
    stream_pointer: int
    operation_token_sha256: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CudaGm1ChildSolve3RhoBoundary:
    ready_event: Any
    step_index: int
    stream_pointer: int
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CudaGm1ChildCompletedClosureRefresh:
    ready_event: Any
    step_index: int
    completed_step_index: int
    refresh_index: int
    stream_pointer: int
    tke_commit_receipt_sha256: str
    operation_token_sha256: str
    arrays: Mapping[str, Any]
    receipt: Mapping[str, Any]


@dataclass(slots=True)
class CudaGm1ChildStepState:
    selection: Gm1ForecastSelection
    envelope: LoadedGm1ChildFullStateEnvelope
    atmosphere: Any
    auxiliary: Any
    terrain: Any
    static_metrics: Any
    vertical: Gm1ChildStretchedMetrics
    closure_export: LoadedChildStretchedClosure
    closure_upload: CudaChildStretchedClosure
    product: CudaGm1ChildProduct
    prepared: CudaInitializedTerrainOpenState
    transition: CudaGm1ChildWsm6Transition
    moist_bridge: Any
    scalar_rk: Gm1ChildScalarMoistRKBinding
    solve1: Gm1Solve1Carriers
    pi0_3d: Any
    th0_3d: Any
    compiled: Any
    momentum_compiled: Any
    momentum_advection_compiled: Any
    stretched_moist_compiled: Any
    psolver_compiled: Any
    terrain_compiled: Any
    open_compiled: Any
    lateral_compiled: Any
    scalar_advection_compiled: Any
    scalar_pdef_compiled: Any
    apmasscon_compiled: Any
    pdefq_compiled: Any
    pdefq_workspace: Any
    pdefq_asq: Any
    closure: CudaGm1ChildStretchedClosureState
    post_final: CudaGm1ChildPostFinalRkState
    child_core_compiled: Any
    mass1_f64: float
    stream: Any
    device_id: int
    ready_event: Any
    step_index: int = 0
    time_seconds: float = 0.0
    next_stage: int = 1
    solve1_ready: bool = False
    retained: list[Any] = field(default_factory=list)
    last_final_dynamics_sync: CudaGm1ChildFinalDynamicsSync | None = None
    last_completed_closure_refresh: CudaGm1ChildCompletedClosureRefresh | None = None
    last_receipt: dict[str, Any] | None = None


GM1_CHILD_CORE_KERNEL_NAMES = (
    "gm1_child_predict_and_compose_held_base3d_v1",
    "gm1_child_turbs_pack_scalar_base3d_v1",
)
GM1_CHILD_CORE_SOURCE = r"""// gm1.cuda-real-50m-child-large-step-f32/v1
static_assert(sizeof(float)==4,"GM1 default REAL requires float32");
__device__ __forceinline__ float A(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float S(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float M(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float D(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long I3(unsigned k,unsigned j,unsigned i,unsigned ny,unsigned nx){return ((unsigned long long)k*ny+j)*nx+i;}
__device__ __forceinline__ unsigned long long P3(unsigned k,unsigned j,unsigned i,unsigned ni,unsigned nj){return ((unsigned long long)(k+1u)*(nj+2u)+(j+1u))*(ni+2u)+(i+1u);}
__device__ __forceinline__ int C(int x,int n){return x<0?0:(x>=n?n-1:x);}

extern "C" __global__ void gm1_child_predict_and_compose_held_base3d_v1(
 const float*th0,const float*pi0,const float*rho,const float*ppi,const float*tha,const float*qa,
 float*thheld,float*qheld,float*ppten,const float*qpten,const float*qtten,const float*qvten,const float*qcten,
 unsigned ni,unsigned nj,unsigned nk,float dt){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
 const float rd=287.04f,rv=461.5f,rp00=1.0f/100000.0f,rddcv=287.04f/(1005.7f-287.04f),tem=0.0001f*0.0001f,qsmall=1.0e-14f;
 for(;q<n;q+=step){unsigned i=(unsigned)(q%ni),j=(unsigned)((q/ni)%nj),k=(unsigned)(q/((unsigned long long)ni*nj));float ht=thheld[q],qvt=qheld[q],pinew=0.0f;
  if(fabsf(M(dt,ht))>tem||fabsf(M(dt,qvt))>qsmall){float thnew=A(tha[q],M(dt,ht)),qvnew=A(qa[q],M(dt,qvt));float base=M(M(rho[q],A(th0[q],thnew)),A(rd,M(rv,qvnew)));base=M(base,rp00);pinew=S(powf(base,rddcv),pi0[q]);ppten[P3(k,j,i,ni,nj)]=A(D(S(pinew,ppi[q]),dt),qpten[q]);}
  else ppten[P3(k,j,i,ni,nj)]=qpten[q];
  thheld[q]=A(ht,qtten[q]);qheld[q]=A(qvt,qvten[q]);qheld[n+q]=A(qheld[n+q],qcten[q]);
 }}

extern "C" __global__ void gm1_child_turbs_pack_scalar_base3d_v1(
 const float*th0,const float*tha,const float*qa,float*sh,unsigned ni,unsigned nj,unsigned nk,unsigned species){
 unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,step=(unsigned long long)blockDim.x*gridDim.x,hx=ni+2u,hy=nj+2u,hxy=hx*hy,n=(unsigned long long)ni*nj*nk;
 for(unsigned long long z=q;z<(unsigned long long)nk*hxy;z+=step){unsigned i=(unsigned)(z%hx),j=(unsigned)((z/hx)%hy),k=(unsigned)(z/hxy);int ii=C((int)i-1,(int)ni),jj=C((int)j-1,(int)nj);unsigned long long p=I3(k,(unsigned)jj,(unsigned)ii,nj,ni);sh[z]=(species==0xffffffffu)?A(S(th0[p],300.0f),tha[p]):qa[(unsigned long long)species*n+p];}
}
"""
GM1_CHILD_CORE_SOURCE_SHA256 = (
    "ff220bd14e3f930864eabfa59961f38b107072c3888ed8a91f9f0fff50674e83"
)


@dataclass(frozen=True, slots=True)
class CompiledGm1ChildCore:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


def compile_cuda_gm1_child_core() -> CompiledGm1ChildCore:
    if hashlib.sha256(GM1_CHILD_CORE_SOURCE.encode()).hexdigest() != GM1_CHILD_CORE_SOURCE_SHA256:
        raise RuntimeError("packaged child-step CUDA source differs from its pin")
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
        GM1_CHILD_CORE_SOURCE, "gm1_child_step_core_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        GM1_CHILD_CORE_SOURCE, "gm1_child_step_core_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode()
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", decoded)
        for name in GM1_CHILD_CORE_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks child-step entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled child-step image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in GM1_CHILD_CORE_KERNEL_NAMES)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1ChildCore(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


@dataclass(frozen=True, slots=True)
class CudaGm1ChildStretchedScalarStageLaunch:
    ready_event: Any
    receipt: Mapping[str, Any]


def _launch_cuda_gm1_child_stretched_scalar_stage(
    binding: Gm1ChildScalarMoistRKBinding,
    mass_flux: Mapping[str, Any],
    metrics: Mapping[str, Any],
    divx: Any,
    theta_divergence_term: Any,
    vertical: Gm1ChildStretchedMetrics,
    *,
    nrk: int,
    dt: float,
    dx: float,
    dy: float,
    stream: Any,
    advection_compiled: Any,
    pdef_compiled: Any,
    open_compiled: Any,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1ChildStretchedScalarStageLaunch:
    """Compose the source-ordered theta/q RK stage without a uniform-z call."""

    if not isinstance(binding, Gm1ChildScalarMoistRKBinding):
        raise TypeError("binding must be Gm1ChildScalarMoistRKBinding")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    state = binding.state
    if allow_uniform_idealized_validation and (
        (state.ni, state.nj, state.nk) != (300, 300, 75)
        or vertical.identity.get("fixture_class")
        != "source-audited-flat-idealized-uniform-200m"
        or vertical.ztop_f32 != 15000.0
        or vertical.dz_reference_f32 != 200.0
    ):
        raise ValueError(
            "uniform idealized scalar RK requires the exact admitted vertical"
        )
    if nrk != state.next_nrk or state.completed or nrk not in (1, 2, 3):
        raise ValueError(
            f"stretched scalar/moist cadence requires stage {state.next_nrk}"
        )
    if int(stream.ptr) != int(state.stream.ptr):
        raise ValueError("stretched scalar/moist RK must remain on its bound stream")
    targets = {
        binding.compiled.target_arch,
        advection_compiled.target_arch,
        pdef_compiled.target_arch,
        open_compiled.target_arch,
        vertical.target_arch,
    }
    if len(targets) != 1:
        raise ValueError("stretched scalar dependencies target different architectures")
    # solve2.F:1435: every one of the axis's numq rows, number concentrations
    # included, is advected and (final stage) positive-definite corrected.
    numq = scalar_moist_rk_numq(state)
    qmag = state.qmag
    final_advection = final_stage_moisture_advection_mode()
    cupy = importlib.import_module("cupy")
    dttmp = scalar_moist_rk_stage_timestep_f32(dt, nrk)
    words = state.ni * state.nj * state.nk
    threads = (CUDA_SCALAR_MOIST_RK_THREADS, 1, 1)
    grid = (
        (words + CUDA_SCALAR_MOIST_RK_THREADS - 1)
        // CUDA_SCALAR_MOIST_RK_THREADS,
        1,
        1,
    )
    u32 = cupy.uint32
    # adv.F:214-340 and pdef.F:49-166 read the same lateral family the halo
    # fills below apply, so the sides come from those fills' own binding.
    from .cuda_boundary_families import bound_lateral_sides

    lateral_sides = bound_lateral_sides(open_compiled)
    scalar_receipts: list[Mapping[str, Any]] = []
    pdef_receipts: list[Mapping[str, Any]] = []
    with stream:
        stream.wait_event(state.ready_event)
        binding.compiled.kernels[0](
            grid,
            threads,
            (
                state.th0,
                state.theta_current,
                state.theta_padded,
                u32(state.ni),
                u32(state.nj),
                u32(state.nk),
                u32(CM1_OPEN_NGXY),
            ),
            stream=stream,
        )
        _fill_scalar_open_halos(
            state.theta_padded,
            state=state,
            stream=stream,
            open_compiled=open_compiled,
        )
        state.advection.fill(cupy.float32(0.0))
        theta_launch = launch_cuda_gm1_child_stretched_scalar_advection5(
            state.theta_padded,
            mass_flux,
            metrics,
            divx,
            state.advection,
            state.fluxes,
            vertical,
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            nrk=nrk,
            dx=dx,
            dy=dy,
            weps=theta_weps_f64(),
            stream=stream,
            compiled=advection_compiled,
            allow_compact_validation=allow_uniform_idealized_validation,
            lateral_sides=lateral_sides,
        )
        scalar_receipts.append(theta_launch.receipt)
        binding.compiled.kernels[2](
            grid,
            threads,
            (
                state.theta_a,
                state.theta_held,
                theta_divergence_term,
                state.advection,
                state.theta_current,
                cupy.uint64(words),
                cupy.float32(dttmp),
            ),
            stream=stream,
        )
        for species in range(numq):
            current = state.moisture_current[species]
            binding.compiled.kernels[1](
                grid,
                threads,
                (
                    current,
                    state.moisture_padded,
                    u32(state.ni),
                    u32(state.nj),
                    u32(state.nk),
                    u32(CM1_OPEN_NGXY),
                ),
                stream=stream,
            )
            _fill_scalar_open_halos(
                state.moisture_padded,
                state=state,
                stream=stream,
                open_compiled=open_compiled,
            )
            state.advection.fill(cupy.float32(0.0))
            weps = moisture_weps_f64(species, qmag)
            if nrk != 3 or final_advection == "launch":
                moisture_launch = launch_cuda_gm1_child_stretched_scalar_advection5(
                    state.moisture_padded,
                    mass_flux,
                    metrics,
                    divx,
                    state.advection,
                    state.fluxes,
                    vertical,
                    ni=state.ni,
                    nj=state.nj,
                    nk=state.nk,
                    nrk=nrk,
                    dx=dx,
                    dy=dy,
                    weps=weps,
                    stream=stream,
                    compiled=advection_compiled,
                    allow_compact_validation=allow_uniform_idealized_validation,
                    lateral_sides=lateral_sides,
                )
                scalar_receipts.append(moisture_launch.receipt)
            if nrk == 3:
                # adv.F:421-445 with pdef=1 (solve2.F:1451-1455) forms this
                # row's advection from the pdef chain alone, and CM1 calls
                # advs once per row.  The plain fifth-order launch the stage
                # used to enqueue first wrote only words the chain rewrites
                # before any read: its horizontal kernel's fx/fy faces are
                # exactly the pdef horizontal kernel's, its vertical kernel's
                # fz(k=2..nk) exactly the pdef vertical kernel's (the only fz
                # the directional kernel reads), and its tendency interior
                # 3<=i<ni-3, 3<=j<nj-3 exactly the one the total kernel assigns
                # with "=".  The ring stays the fill's +0.0 either way.  So it
                # is enqueued only under GM1_FINAL_STAGE_MOISTURE_ADVECTION=launch.
                pdef_launch = launch_cuda_gm1_child_stretched_pdef_moisture(
                    state.moisture_a[species],
                    state.moisture_padded,
                    mass_flux,
                    metrics,
                    divx,
                    state.fluxes,
                    state.advx,
                    state.advy,
                    state.advz,
                    state.pdef_candidate,
                    state.advection,
                    vertical,
                    ni=state.ni,
                    nj=state.nj,
                    nk=state.nk,
                    dttmp=float(dttmp),
                    dx=dx,
                    dy=dy,
                    weps=weps,
                    species=species,
                    numq=numq,
                    stream=stream,
                    compiled=pdef_compiled,
                    allow_compact_validation=allow_uniform_idealized_validation,
                    lateral_sides=lateral_sides,
                )
                pdef_receipts.append(pdef_launch.receipt)
            binding.compiled.kernels[3](
                grid,
                threads,
                (
                    state.moisture_a[species],
                    state.moisture_held[species],
                    state.advection,
                    current,
                    cupy.uint64(words),
                    cupy.float32(dttmp),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state.ready_event = event
    state.next_nrk = nrk + 1
    state.completed = nrk == 3
    receipt = MappingProxyType(
        {
            # v2: numq, and one CM1 qname and weps per row the stage advanced.
            # v3: on the final stage ``scalar_launches`` holds theta's launch
            # only, unless GM1_FINAL_STAGE_MOISTURE_ADVECTION=launch; each
            # moisture row's advection is its ``pdef_launches`` entry
            # (adv.F:421-445), as ``moisture_advection`` says.
            "schema": "gm1.cuda-child-stretched-scalar-moist-rk-stage/v3",
            "stage": nrk,
            "dttmp_f32": float(dttmp),
            "source_order": [
                "solve2.F:1310-1427 theta RK",
                "solve2.F:1430-1533 q RK",
            ],
            "moisture_advection": final_stage_moisture_advection_record(
                nrk, final_advection
            ),
            "numq": numq,
            "species": list(state.names),
            "moisture_weps_f64": [moisture_weps_f64(n, qmag) for n in range(numq)],
            "vertical_metrics": dict(vertical.identity),
            "scalar_launches": [dict(value) for value in scalar_receipts],
            "pdef_launches": [dict(value) for value in pdef_receipts],
            "uniform_idealized_validation": allow_uniform_idealized_validation,
            "legacy_uniform_z_stage_used": False,
        }
    )
    json.dumps(dict(receipt), allow_nan=False, sort_keys=True)
    return CudaGm1ChildStretchedScalarStageLaunch(event, receipt)


def _load_static_heights(directory: Path, reference: Any) -> tuple[Any, Any]:
    """Open the static heights the composed envelope names, by its digests."""

    root = directory.resolve(strict=True)
    if {entry.name for entry in root.iterdir()} != set(reference.files):
        raise ValueError(
            "child static-height inventory differs from the file list the "
            "composed child envelope records for it"
        )
    for name, (nbytes, expected) in reference.files.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != nbytes
            or _hash_file(path) != expected
        ):
            raise ValueError(
                f"child static-height {name} is not the bytes the composed child "
                "envelope was built from; the child's vertical metrics would "
                "describe a surface it does not stand on"
            )
    manifest = _canonical_json(root / "manifest.json", reference.digest("manifest.json"))
    if manifest.get("status") != "PASS_REAL_SOURCE_TERRAIN_CUDA_ZH_ZF_300X300X75":
        raise ValueError("child static-height status differs")
    np = importlib.import_module("numpy")
    zh = np.memmap(root / "zh.f32le", mode="r", dtype="<f4", shape=(75, 300, 300))
    zf = np.memmap(root / "zf.f32le", mode="r", dtype="<f4", shape=(76, 300, 300))
    return zh, zf


def _device_copy(cupy: Any, value: Any, *, stream: Any) -> Any:
    np = importlib.import_module("numpy")
    host = np.ascontiguousarray(value, dtype=np.float32)
    with stream:
        return cupy.asarray(host, dtype=cupy.float32, order="C")


def _physical_auxiliary(auxiliary: Any, name: str) -> Any:
    source = auxiliary.field("terrain_open", name)
    if name in {"pi0", "prs0", "th0", "thv0", "rth0", "qv0", "rho0", "rr0"}:
        return source[1:76, 3:303, 3:303]
    if name in {"rf0", "rrf0"}:
        return source[1:77, 3:303, 3:303]
    if name in {"ppi", "pp3d", "ppx", "piadv", "ppterm", "ppten", "thv"}:
        return source[1:76, 1:301, 1:301]
    if name in {"tha", "th3d", "thten", "thten1"}:
        return source[1:76, 3:303, 3:303]
    if name in {"ua", "u3d", "uten", "uten1"}:
        return source[1:76, 3:303, 3:304]
    if name in {"va", "v3d", "vten", "vten1"}:
        return source[1:76, 3:304, 3:303]
    if name in {"wa", "w3d", "wten", "wten1"}:
        return source[1:77, 3:303, 3:303]
    raise KeyError(f"no physical child auxiliary view is defined for {name}")


def _build_child_product_and_prepared(
    envelope: LoadedGm1ChildFullStateEnvelope,
    atmosphere: Any,
    auxiliary: Any,
    static_metrics: Any,
    *,
    closure_payload_sha256: str,
    stream: Any,
) -> tuple[CudaGm1ChildProduct, CudaInitializedTerrainOpenState]:
    cupy = importlib.import_module("cupy")
    arrays = {
        name: cupy.zeros(shape, dtype=cupy.float32)
        for name, shape in terrain_open_sound_shapes(300, 300, 75).items()
    }
    # solve3 bcr owns one horizontal scalar halo, distinct from the acoustic
    # ng=3 scalar allocation.  It is populated from live physical rho only at
    # the exact post-WSM, pre-held-recapture seam.
    arrays["rho"] = cupy.zeros((77, 302, 302), dtype=cupy.float32)
    available_padded = set(auxiliary.carriers["terrain_open"])
    with stream:
        for name in tuple(arrays):
            if name in available_padded:
                arrays[name] = _device_copy(
                    cupy, auxiliary.field("terrain_open", name), stream=stream
                )
        for name in ("uh", "vh", "ruf", "rvf"):
            arrays[name].fill(cupy.float32(1.0))

        static = dict(static_metrics.metrics.state.arrays)
        fields: dict[str, Any] = dict(static)
        for name in (
            "pi0",
            "th0",
            "qv0",
            "rho0",
            "rr0",
            "rf0",
            "rrf0",
            "ppi",
            "pp3d",
            "ppx",
            "tha",
            "th3d",
            "ua",
            "u3d",
            "va",
            "v3d",
            "wa",
            "w3d",
        ):
            fields[name] = _device_copy(
                cupy, _physical_auxiliary(auxiliary, name), stream=stream
            )
        for name in ("prs", "rho", "rr", "rf"):
            fields[name] = _device_copy(
                cupy, auxiliary.field("live_density", name), stream=stream
            )
        for name in ("qa", "q3d"):
            fields[name] = _device_copy(cupy, atmosphere.field(name), stream=stream)
        for name in EQTSET2_HELD_FIELDS:
            fields[name] = _device_copy(
                cupy, atmosphere.field(f"held.{name}"), stream=stream
            )
        for name in WSM6_COMPONENT_PERSISTENCE:
            fields[f"component.{name}"] = _device_copy(
                cupy, atmosphere.field(f"component.{name}"), stream=stream
            )
        fields["total_geopotential_zf"] = _device_copy(
            cupy,
            auxiliary.field("geopotential", "total_geopotential_zf"),
            stream=stream,
        )
        fields["u0"] = _device_copy(cupy, envelope.u0, stream=stream)
        fields["v0"] = _device_copy(cupy, envelope.v0, stream=stream)
        rho0_padded = _device_copy(
            cupy, auxiliary.field("terrain_open", "rho0"), stream=stream
        )

        metrics = dict(static)
        metrics.update(
            {
                "rho0_3d": fields["rho0"],
                # sound.F:604-613 consumes the admitted halo values at the
                # outer u/v faces.  Keep this exact padded carrier; clamping or
                # broadcasting the physical slice is not source equivalent.
                "rho0_padded": rho0_padded,
                "rr0": fields["rr0"],
                "rf0": fields["rf0"],
                "rrf0": fields["rrf0"],
            }
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)

    product = CudaGm1ChildProduct(
        fields,
        Gm1ChildProductIdentity(
            manifest_sha256=envelope.pins.manifest_sha256,
            payload_sha256=atmosphere.arena_sha256,
            source_artifact_sha256=closure_payload_sha256,
            source_contract_sha256=envelope.pins.manifest_sha256,
        ),
        stream,
        int(cupy.cuda.Device().id),
        event,
    )
    initialized_compiled = compile_cuda_initialized_terrain_open()
    open_compiled = compile_cuda_sound3_open_boundary()
    terrain_compiled = compile_cuda_terrain_open_boundary()
    prepared = CudaInitializedTerrainOpenState(
        product,
        arrays,
        metrics,
        event,
        initialized_compiled,
        open_compiled,
        terrain_compiled,
        {
            "schema": "gm1.cuda-child-padded-state-upload/v1",
            "source_manifest_sha256": envelope.pins.manifest_sha256,
            "rho_padded": {
                "shape": [77, 302, 302],
                "disposition": "solve3 post-WSM physical copy plus exact open bcr",
            },
            "python_scientific_transforms": False,
        },
    )
    return product, prepared


def _create_child_transition(
    product: CudaGm1ChildProduct,
    *,
    stream: Any,
) -> CudaGm1ChildWsm6Transition:
    cupy = importlib.import_module("cupy")
    component = create_arwen_wsm6_component(
        component_id=ARWEN_WSM6_COMPONENT_ID,
        nx=300,
        ny=300,
        nz=75,
        dx_m=50.0,
        dy_m=50.0,
        dz_m=100.0,
        # param.F:2608's dz*nk is the model top only on the uniform column.
        # This child is stretch_z=1: 100 m is the metric reference of
        # param.F:8170 and its top is sigmaf(nk+1) = 15 km, so dz_m*nz would
        # declare a 7.5 km domain under the 15 km geopotential faces the
        # component is handed.
        ztop_m=GM1_CHILD_STRETCHED_ZTOP_F32,
        dt_s=0.5,
    )
    bridge = Eqtset2Wsm6Bridge(
        component=component,
        nx=300,
        ny=300,
        nz=75,
        dt_s=0.5,
        idiss=False,
        efall=False,
        product_admission=False,
    )
    restart = {
        name: (
            product.field(name)
            if name in EQTSET2_HELD_FIELDS
            else product.field(name)
        )
        for name in EQTSET2_RESTART_FIELDS
    }
    bridge.restore_restart_arrays(restart, completed_calls=0)
    q = product.field("q3d")
    with stream:
        php = cupy.zeros_like(product.field("total_geopotential_zf"))
    carriers = Eqtset2Wsm6Carriers(
        pi0=product.field("pi0"),
        th0=product.field("th0"),
        rho=product.field("rho"),
        pp3d=product.field("pp3d"),
        th3d=product.field("th3d"),
        p=product.field("prs"),
        alt=product.field("rr"),
        phb=product.field("total_geopotential_zf"),
        php=php,
        qv=q[0],
        qc=q[1],
        qr=q[2],
        qi=q[3],
        qs=q[4],
        qg=q[5],
        qpten=product.field("qpten"),
        qtten=product.field("qtten"),
        qvten=product.field("qvten"),
        qcten=product.field("qcten"),
    )
    return CudaGm1ChildWsm6Transition(bridge, carriers)


def _create_child_closure_state(
    product: CudaGm1ChildProduct,
    prepared: CudaInitializedTerrainOpenState,
    closure_upload: CudaChildStretchedClosure,
    zh: Any,
    zf: Any,
    static_metrics: Any,
    *,
    stream: Any,
) -> CudaGm1ChildStretchedClosureState:
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    coefficients = cpu_build_rayleigh_coefficients(
        np.ascontiguousarray(np.pad(zh, ((0, 0), (1, 1), (1, 1)), mode="edge")),
        np.ascontiguousarray(np.pad(zf, ((0, 0), (1, 1), (1, 1)), mode="edge")),
    )
    closure = dict(closure_upload.arrays)
    face = (76, 300, 300)
    mass = (75, 300, 300)
    rayleigh_compiled = compile_cuda_gm1_child_terrain_rayleigh()
    sgs_compiled = compile_cuda_sgs_tke()
    sgs_tke_compiled = compile_cuda_gm1_child_stretched_sgs_tke()
    # The frozen WSM binding accepts the original terrain-metric launch, not
    # the already-normalized Gm1ChildStretchedMetrics view.  Validate once and
    # retain the launch because every public SGS/TKE call repeats that seal.
    bind_gm1_child_stretched_sgs_tke_metrics(static_metrics)
    sgs_tke_metrics = static_metrics
    with stream:
        rayleigh = {
            "tauh": cupy.asarray(coefficients.tauh),
            "taus": cupy.asarray(coefficients.taus),
            "tauf": cupy.asarray(coefficients.tauf),
            "u0": product.field("u0"),
            "v0": product.field("v0"),
            "ua": product.field("ua"),
            "va": product.field("va"),
            "wa": product.field("wa")[:75],
            "tha": product.field("tha"),
            "uten1": cupy.empty((75, 300, 301), dtype=cupy.float32),
            "vten1": cupy.empty((75, 301, 300), dtype=cupy.float32),
            "wten1": cupy.empty(mass, dtype=cupy.float32),
            "thten1": cupy.empty(mass, dtype=cupy.float32),
        }
        sgs = {
            "ruh": prepared.arrays["uh"][CM1_OPEN_NGXY : CM1_OPEN_NGXY + 300],
            "rvh": prepared.arrays["vh"][CM1_OPEN_NGXY : CM1_OPEN_NGXY + 300],
            "rmf": product.field("rmf"),
            "nm": closure["nm"],
            "defv": closure["defv"],
            "defh": closure["defh"],
        }
        for name in SGS_TKE_MUTABLE_ARRAY_NAMES:
            if name == "grdscl":
                # The durable closure ABI stores only physical k=1..nk for
                # grdscl; tkekm uses a z-face work carrier.  This is an exact
                # device view/copy, with the unused bottom face held at +0.
                value = cupy.zeros(face, dtype=cupy.float32)
                cupy.copyto(value[1:], closure[name])
                sgs[name] = value
            elif name in closure:
                sgs[name] = closure[name]
            else:
                sgs[name] = cupy.zeros(face, dtype=cupy.float32)
        stress_shapes = sgs_stress_shapes(300, 300, 75)
        stress: dict[str, Any] = {}
        for name, shape in stress_shapes.items():
            if name in {"kmh", "kmv"}:
                stress[name] = sgs[name]
            elif name in product.fields:
                stress[name] = product.field(name)
            else:
                stress[name] = cupy.zeros(shape, dtype=cupy.float32)
        tke = {
            "tkea": sgs["tkea"],
            "tke3d": closure["tke3d"],
            "tketen": sgs["tketen"],
            "kmh": sgs["kmh"],
            "kmv": sgs["kmv"],
            **{
                name: product.field(name)
                for name in (
                    "rho",
                    "rf",
                    "mh",
                    "mf",
                    "gz",
                    "rgz",
                    "rgzu",
                    "rgzv",
                    "c1",
                    "c2",
                    "rrf0",
                )
            },
            "rru": prepared.arrays["rru"],
            "rrv": prepared.arrays["rrv"],
            "rrw": prepared.arrays["rrw"],
            "divx": cupy.zeros(mass, dtype=cupy.float32),
        }
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    return CudaGm1ChildStretchedClosureState(
        rayleigh=rayleigh,
        sgs=sgs,
        stress=stress,
        tke=tke,
        tke_workspace=cupy.empty(face, dtype=cupy.float32),
        idiss=Eqtset2IdissConsumer(nx=300, ny=300, nz=75),
        rayleigh_compiled=rayleigh_compiled,
        sgs_compiled=sgs_compiled,
        sgs_tke_compiled=sgs_tke_compiled,
        sgs_tke_metrics=sgs_tke_metrics,
        ready_event=event,
    )


CUDA_CHILD_LANE_SHAPE_ZYX = (75, 300, 300)


def _require_cuda_lane_shape(envelope: Any) -> None:
    """The compiled child lane allocates one child geometry.

    Every kernel launch, workspace and carrier view below is built for
    75x300x300 at 50 m.  An envelope for another child grid is refused here,
    by value, rather than uploaded into allocations sized for this one - which
    would read and write past every arena the envelope points at.
    """

    if envelope.domain_shape_zyx != CUDA_CHILD_LANE_SHAPE_ZYX:
        raise ValueError(
            "this composed child envelope declares shape "
            f"{list(envelope.domain_shape_zyx)}, but the CUDA child lane is compiled "
            f"for {list(CUDA_CHILD_LANE_SHAPE_ZYX)}; every kernel, workspace and "
            "carrier view here is sized for that grid, so another one would be "
            "uploaded into allocations that do not describe it"
        )


def _sibling(work_root: Path, reference: Any) -> Path:
    """Resolve the sibling directory the envelope names, and only that one."""

    path = (work_root / reference.directory).resolve()
    if path.parent != work_root.resolve() or not path.is_dir() or path.is_symlink():
        raise ValueError(
            f"the composed child envelope names {reference.role} artifact "
            f"{reference.directory!r} beside it, but that is not a regular sibling "
            "directory of the publication; the child would be built from an "
            "artifact the envelope never admitted"
        )
    return path


def _require_artifact_files(loaded: dict[str, str], reference: Any) -> None:
    """Every file opened must be the byte-identical file the envelope names."""

    for name, actual in loaded.items():
        recorded = reference.digest(name)
        if actual != recorded:
            raise ValueError(
                f"the composed child envelope was built from {reference.role} "
                f"{name} {recorded}, but the artifact opened here is {actual}; "
                "every receipt this run publishes would name the artifacts the "
                "envelope records while the run consumed different bytes, and the "
                "substitution would be invisible in the receipt"
            )


def _require_auxiliary_inputs(
    auxiliary: Any, terrain_reference: Any, static_reference: Any
) -> None:
    """The estate must have been built over the terrain and heights admitted.

    The auxiliary manifest records the terrain it stands on and the static
    height payloads it was reconstructed over.  Those must be the artifacts
    this composition admits: a base state, open-boundary halo and radiation
    estate built over one surface, composed with the vertical metrics of
    another, gives the child a base state that never balanced on the terrain it
    is standing on.

    This is the binding the retired `combined_23_sha256` key stood in for.  It
    is stated in digests the Rust producer actually reads and records, instead
    of the digest of a CUDA operator output it cannot compute.
    """

    static_inputs = auxiliary.manifest["static_inputs"]
    for key, reference, file, what in (
        ("zh_sha256", static_reference, "zh.f32le", "mass heights"),
        ("zf_sha256", static_reference, "zf.f32le", "face heights"),
        (
            "terrain_manifest_sha256",
            terrain_reference,
            "manifest.json",
            "terrain manifest",
        ),
        (
            "terrain_payload_sha256",
            terrain_reference,
            "terrain-msl.f32le",
            "terrain surface",
        ),
        (
            "terrain_receipt_sha256",
            terrain_reference,
            "receipt.json",
            "terrain receipt",
        ),
    ):
        recorded = static_inputs[key]
        admitted = reference.digest(file)
        if recorded != admitted:
            raise ValueError(
                f"the child auxiliary estate was built over {what} {recorded}, but "
                f"this composition admits {admitted}; the child's base state, open "
                "boundaries and radiation histories would be applied over a surface "
                "and a set of vertical metrics they were never balanced against"
            )


def create_cuda_gm1_child_step_state(
    full_state_directory: str | Path,
    *,
    pins: Any | None = None,
    stream: Any | None = None,
) -> CudaGm1ChildStepState:
    """Strictly load and upload one complete child first-step publication.

    The envelope is admitted by the identity its own receipt records, and every
    sibling artifact is then opened at the directory the envelope names and
    required to be the exact bytes the envelope was composed from.  Nothing is
    pinned to the first child this project composed, so a child built from a
    parent at any admissible completed step is loadable; and nothing is taken
    on trust either, because a substituted sibling is refused by digest.

    ``pins`` replays one exact historical envelope byte for byte.
    """

    full_root = Path(full_state_directory).resolve(strict=True)
    envelope = load_gm1_child_full_state_envelope(full_root, pins=pins)
    _require_cuda_lane_shape(envelope)
    work_root = full_root.parent

    atmosphere_ref = envelope.artifact("atmosphere")
    auxiliary_ref = envelope.artifact("auxiliary")
    terrain_ref = envelope.artifact("terrain")
    static_ref = envelope.artifact("static_heights")
    closure_ref = envelope.artifact("closure")

    atmosphere = load_gm1_child_atmosphere(
        _sibling(work_root, atmosphere_ref),
        expected_manifest_sha256=atmosphere_ref.digest("manifest.json"),
        expected_arena_sha256=atmosphere_ref.digest("arena.f32le"),
        expected_receipt_sha256=atmosphere_ref.digest("receipt.json"),
    )
    auxiliary = load_gm1_child_auxiliary(
        _sibling(work_root, auxiliary_ref),
        expected_manifest_sha256=auxiliary_ref.digest("manifest.json"),
        expected_receipt_sha256=auxiliary_ref.digest("receipt.json"),
        expected_atmosphere_manifest_sha256=atmosphere.manifest_sha256,
        expected_atmosphere_arena_sha256=atmosphere.arena_sha256,
        expected_atmosphere_receipt_sha256=atmosphere.receipt_sha256,
    )
    _require_artifact_files(
        {
            "terrain-open.f32le": auxiliary.terrain_open_sha256,
            "live-density.f32le": auxiliary.live_density_sha256,
            "total-geopotential-zf.f32le": auxiliary.geopotential_sha256,
        },
        auxiliary_ref,
    )
    terrain = load_gm1_child_terrain_export(
        _sibling(work_root, terrain_ref),
        expected_manifest_sha256=terrain_ref.digest("manifest.json"),
        expected_payload_sha256=terrain_ref.digest("terrain-msl.f32le"),
        expected_receipt_sha256=terrain_ref.digest("receipt.json"),
    )
    closure_export = load_gm1_child_stretched_closure(_sibling(work_root, closure_ref))
    # The envelope and the closure the run actually opens must be the same
    # artifact.  Without this the envelope could textually record the refused
    # uniform-z closure while the run uploaded the corrected stretched-z arena
    # - or the reverse - and the published receipt would name a closure the
    # child was never initialized from.
    _require_artifact_files(
        {
            "manifest.json": closure_export.manifest_sha256,
            "closure.f32le": closure_export.payload_sha256,
            "receipt.json": closure_export.receipt_sha256,
        },
        closure_ref,
    )
    refuse_uniform_z_closure(
        manifest_sha256=closure_ref.digest("manifest.json"),
        payload_sha256=closure_ref.digest("closure.f32le"),
        receipt_sha256=closure_ref.digest("receipt.json"),
        carrier_sha256={},
    )
    zh, zf = _load_static_heights(_sibling(work_root, static_ref), static_ref)
    _require_auxiliary_inputs(auxiliary, terrain_ref, static_ref)
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    static_metrics = launch_cuda_gm1_child_terrain_metrics(terrain, stream=stream)
    vertical = bind_gm1_child_stretched_metrics(static_metrics)
    closure_upload = upload_gm1_child_stretched_closure(closure_export, stream=stream)
    product, prepared = _build_child_product_and_prepared(
        envelope,
        atmosphere,
        auxiliary,
        static_metrics,
        closure_payload_sha256=closure_export.payload_sha256,
        stream=stream,
    )
    transition = _create_child_transition(product, stream=stream)
    child_core = compile_cuda_gm1_child_core()
    closure_state = _create_child_closure_state(
        product, prepared, closure_upload, zh, zf, static_metrics, stream=stream
    )
    compiled = compile_cuda_gm1_forecast()
    momentum_compiled = compile_cuda_momentum_rk()
    momentum_advection_compiled = compile_cuda_gm1_child_stretched_momentum()
    stretched_moist_compiled = compile_cuda_gm1_child_stretched_moist()
    psolver_compiled = compile_cuda_gm1_child_psolver3_open()
    lateral_compiled = compile_cuda_lateral_closure()
    scalar_advection_compiled = compile_cuda_gm1_child_stretched_scalar_advection()
    scalar_pdef_compiled = compile_cuda_gm1_child_stretched_scalar_pdef()
    apmasscon_compiled = compile_cuda_apmasscon()
    pdefq_compiled = compile_cuda_wsm6_pdefq()
    pdefq_workspace = create_cuda_wsm6_pdefq_workspace(ni=300, nj=300, nk=75)
    moist_compiled = compile_cuda_moist_stage_bridge()
    moist_bridge = create_cuda_moist_stage_bridge_state(
        ni=300,
        nj=300,
        nk=75,
        base3d=True,
        stream=stream,
        compiled=moist_compiled,
        open_compiled=prepared.open_compiled,
    )
    with stream:
        solve1 = Gm1Solve1Carriers(
            {name: cupy.zeros_like(prepared.arrays[name]) for name in ("uten", "vten", "wten")},
            cupy.zeros_like(product.field("th3d")),
            cupy.zeros_like(product.field("q3d")),
        )
        scalar_rk = create_cuda_gm1_child_scalar_moist_rk_state(
            {
                "tha": product.field("tha"),
                "th3d": product.field("th3d"),
                "qa": product.field("qa"),
                "q3d": product.field("q3d"),
            },
            {"thten1": solve1.theta, "qten": solve1.moisture},
            product.field("th0"),
            ni=300,
            nj=300,
            nk=75,
            stream=stream,
        )
        pdefq_asq = cupy.zeros((6,), dtype=cupy.float64)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    post_final = create_cuda_gm1_child_post_final_rk_state(
        product=product,
        prepared=prepared,
        transition=transition,
        pdefq_asq=pdefq_asq,
        stream=stream,
        forecast_compiled=compiled,
        pdefq_compiled=pdefq_compiled,
        pdefq_workspace=pdefq_workspace,
        ni=300,
        nj=300,
        nk=75,
        dx=50.0,
        dy=50.0,
        dz=100.0,
        dt=0.5,
    )
    state = CudaGm1ChildStepState(
        selection=GM1_CHILD_SELECTION,
        envelope=envelope,
        atmosphere=atmosphere,
        auxiliary=auxiliary,
        terrain=terrain,
        static_metrics=static_metrics,
        vertical=vertical,
        closure_export=closure_export,
        closure_upload=closure_upload,
        product=product,
        prepared=prepared,
        transition=transition,
        moist_bridge=moist_bridge,
        scalar_rk=scalar_rk,
        solve1=solve1,
        pi0_3d=product.field("pi0"),
        th0_3d=product.field("th0"),
        compiled=compiled,
        momentum_compiled=momentum_compiled,
        momentum_advection_compiled=momentum_advection_compiled,
        stretched_moist_compiled=stretched_moist_compiled,
        psolver_compiled=psolver_compiled,
        terrain_compiled=prepared.terrain_compiled,
        open_compiled=prepared.open_compiled,
        lateral_compiled=lateral_compiled,
        scalar_advection_compiled=scalar_advection_compiled,
        scalar_pdef_compiled=scalar_pdef_compiled,
        apmasscon_compiled=apmasscon_compiled,
        pdefq_compiled=pdefq_compiled,
        pdefq_workspace=pdefq_workspace,
        pdefq_asq=pdefq_asq,
        closure=closure_state,
        post_final=post_final,
        child_core_compiled=child_core,
        mass1_f64=0.0,
        stream=stream,
        device_id=int(cupy.cuda.Device().id),
        ready_event=event,
    )
    target_arches = {
        vertical.target_arch,
        compiled.target_arch,
        momentum_compiled.target_arch,
        momentum_advection_compiled.target_arch,
        stretched_moist_compiled.target_arch,
        psolver_compiled.target_arch,
        prepared.terrain_compiled.target_arch,
        prepared.open_compiled.target_arch,
        lateral_compiled.target_arch,
        scalar_rk.compiled.target_arch,
        scalar_advection_compiled.target_arch,
        scalar_pdef_compiled.target_arch,
        apmasscon_compiled.target_arch,
        pdefq_compiled.target_arch,
        closure_state.rayleigh_compiled.target_arch,
        closure_state.sgs_compiled.target_arch,
        closure_state.sgs_tke_compiled.target_arch,
        child_core.target_arch,
    }
    if len(target_arches) != 1:
        raise ValueError("real child CUDA dependencies target different architectures")
    state.mass1_f64 = _initial_mass(
        product,
        prepared,
        product.field("pi0"),
        compiled,
        apmasscon_compiled,
        ni=300,
        nj=300,
        nk=75,
        stream=stream,
    )
    return state


def _gm1_child_live_nm_def_arrays(state: CudaGm1ChildStepState) -> dict[str, Any]:
    names = (
        "c1",
        "c2",
        "mf",
        "pi0",
        "th0",
        "prs",
        "pp3d",
        "th3d",
        "q3d",
        "u3d",
        "v3d",
        "w3d",
        "rho",
        "rf",
        "gz",
        "rgz",
        "gzu",
        "rgzu",
        "gzv",
        "rgzv",
        "gx",
        "gy",
    )
    return {
        **{name: state.product.field(name) for name in names},
        **{name: state.closure.sgs[name] for name in ("nm", "defv", "defh")},
    }


def _gm1_child_scalar_turbulence_arrays(
    state: CudaGm1ChildStepState,
) -> dict[str, Any]:
    p, c = state.product, state.closure
    return {
        **{
            name: p.field(name)
            for name in (
                "th0",
                "tha",
                "qa",
                "rho",
                "rr",
                "rf",
                "mh",
                "mf",
                "gz",
                "rgz",
                "rgzu",
                "rgzv",
                "gx",
                "gxu",
                "gy",
                "gyv",
            )
        },
        "khh": c.sgs["khh"],
        "khv": c.sgs["khv"],
        "thten1": state.solve1.theta,
        "qten": state.solve1.moisture,
    }


def _clear_gm1_child_stress_tendencies(
    closure: CudaGm1ChildStretchedClosureState, cupy: Any
) -> None:
    """Clear per-large-step SGS momentum accumulation targets."""

    for name in ("uten", "vten", "wten"):
        closure.stress[name].fill(cupy.float32(0.0))


def _reset_gm1_child_step_retention(state: CudaGm1ChildStepState) -> None:
    """Release prior-step launch scratch only after its event has completed."""

    state.retained.clear()
    state.closure.retained.clear()
    state.last_final_dynamics_sync = None


def _require_gm1_child_prepared_closure(
    state: CudaGm1ChildStepState,
) -> Mapping[str, Any]:
    refresh = state.last_completed_closure_refresh
    if state.step_index == 0:
        if refresh is not None:
            raise ValueError("initial child step cannot consume a runtime closure refresh")
        return MappingProxyType(
            {
                "schema": GM1_CHILD_CLOSURE_SCHEMA,
                "source": "corrected stretched-z fresh-closure initialization",
                "payload_sha256": state.closure_export.payload_sha256,
            }
        )
    if refresh is None or refresh.step_index != state.step_index - 1:
        raise ValueError(
            "child solve1 requires the immediately preceding completed-step "
            "closure refresh"
        )
    return refresh.receipt


def begin_cuda_gm1_child_large_step(
    state: CudaGm1ChildStepState,
) -> dict[str, Any]:
    """Build exact solve1 held tendencies once, or refuse before mutation."""

    if not isinstance(state, CudaGm1ChildStepState):
        raise TypeError("state must be CudaGm1ChildStepState")
    if state.next_stage != 1 or state.solve1_ready:
        raise ValueError("real child large-step cadence is already active")
    # Resolve both independently committed profile-aware operators before the
    # prior-step event is synchronized or any live tendency is cleared.
    operators = _load_gm1_child_solve1_operator_functions()
    prepared_closure = _require_gm1_child_prepared_closure(state)
    state.ready_event.synchronize()
    cupy = importlib.import_module("cupy")
    s, p, c = state.selection, state.product, state.closure
    _reset_gm1_child_step_retention(state)
    with state.stream:
        for value in state.solve1.momentum.values():
            value.fill(cupy.float32(0.0))
        state.solve1.theta.fill(cupy.float32(0.0))
        state.solve1.moisture.fill(cupy.float32(0.0))
        state.prepared.arrays["ppten"].fill(cupy.float32(0.0))

        # nm/def/km/kh/dissten/tketen are already prepared: step zero consumes
        # the sealed closure-v2 publication, and later steps consume the exact
        # post-density refresh from the immediately preceding finish.  Never
        # rediagnose them here or a completed-step checkpoint would diverge
        # from uninterrupted execution.
        cupy.copyto(c.tke["tke3d"], c.sgs["tkea"])
        tke_source = launch_cuda_gm1_child_stretched_tke_diffusion(
            c.tke,
            metrics=c.sgs_tke_metrics,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            ztop=state.vertical.ztop_f32,
            dt=s.dt_s,
            stream=state.stream,
            compiled=c.sgs_tke_compiled,
        )
        rayleigh = launch_cuda_gm1_child_terrain_rayleigh(
            c.rayleigh,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            stream=state.stream,
            compiled=c.rayleigh_compiled,
        )
        # Preserve solve1's binary32 accumulation order: Rayleigh establishes
        # the held theta tendency, idiss adds next, and scalar SGS adds last.
        cupy.copyto(state.solve1.theta, c.rayleigh["thten1"])
        q = p.field("qa")
        idiss = c.idiss.apply(
            Eqtset2IdissCarriers(
                thten1=state.solve1.theta,
                dissten=c.sgs["dissten"],
                pi0=p.field("pi0"),
                ppi=p.field("ppi"),
                qv=q[0],
                qc=q[1],
                qr=q[2],
                qi=q[3],
                qs=q[4],
                qg=q[5],
            )
        )
        _clear_gm1_child_stress_tendencies(c, cupy)
        stress = launch_cuda_gm1_child_stretched_sgs_stress(
            c.stress,
            metrics=c.sgs_tke_metrics,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            ztop=state.vertical.ztop_f32,
            dt=s.dt_s,
            stream=state.stream,
            compiled=c.sgs_tke_compiled,
        )
        scalar_turbulence = operators.launch_scalar_turbulence(
            _gm1_child_scalar_turbulence_arrays(state),
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dt=s.dt_s,
            vertical=state.vertical,
            stream=state.stream,
            compiled=c.scalar_turbulence_compiled,
            workspace=c.scalar_turbulence_workspace,
        )
        c.last_scalar_turbulence = scalar_turbulence
        c.scalar_turbulence_compiled = scalar_turbulence.compiled
        c.scalar_turbulence_workspace = scalar_turbulence.workspace

        ng = CM1_OPEN_NGXY
        uview = state.solve1.momentum["uten"][
            1 : s.nz + 1, ng : ng + s.ny, ng : ng + s.nx + 1
        ]
        vview = state.solve1.momentum["vten"][
            1 : s.nz + 1, ng : ng + s.ny + 1, ng : ng + s.nx
        ]
        wview = state.solve1.momentum["wten"][
            1 : s.nz + 2, ng : ng + s.ny, ng : ng + s.nx
        ]
        cupy.add(c.rayleigh["uten1"], c.stress["uten"], out=uview)
        cupy.add(c.rayleigh["vten1"], c.stress["vten"], out=vview)
        cupy.copyto(wview, c.stress["wten"])
        cupy.add(wview[: s.nz], c.rayleigh["wten1"], out=wview[: s.nz])
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)
    c.ready_event = event
    c.completed_stage = 0
    c.retained.extend((tke_source, rayleigh, idiss, stress, scalar_turbulence))
    state.ready_event = event
    receipt = {
        "schema": GM1_CHILD_STEP_SCHEMA,
        "phase": "solve1-held-tendencies",
        "source_order": [
            "live calcnm",
            "live calcdef",
            "closure coefficients and TKE source+turbt: solve1.F:317-383",
            "Rayleigh: solve1.F:414-454",
            "idiss: solve1.F:917-1001",
            "SGS momentum and scalar: solve1.F:1115-1240",
        ],
        "prepared_closure": dict(prepared_closure),
        "operators": {
            "tke_source": dict(tke_source.receipt),
            "rayleigh": rayleigh.receipt,
            "idiss": dict(idiss.receipt),
            "sgs_stress": dict(stress.receipt),
            "scalar_turbulence": scalar_turbulence.receipt,
        },
        "legacy_uniform_z_closure_used": False,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return receipt


def finalize_cuda_gm1_child_solve1(state: CudaGm1ChildStepState) -> Any:
    """Run the terrain-base3d held/predictor seam after exact solve1 closure."""

    if state.solve1_ready or state.next_stage != 1:
        raise ValueError("real child solve1 may be finalized exactly once")
    if not state.closure.retained:
        raise ValueError("real child solve1 closure has not run")
    cupy = importlib.import_module("cupy")
    s, p, q = state.selection, state.product, state.transition.carriers
    words = s.nx * s.ny * s.nz
    grid = ((words + GM1_FORECAST_THREADS - 1) // GM1_FORECAST_THREADS, 1, 1)
    with state.stream:
        state.child_core_compiled.kernels[0](
            grid,
            (GM1_FORECAST_THREADS, 1, 1),
            (
                p.field("th0"),
                p.field("pi0"),
                p.field("rho"),
                p.field("ppi"),
                p.field("tha"),
                p.field("qa"),
                state.solve1.theta,
                state.solve1.moisture,
                state.prepared.arrays["ppten"],
                q.qpten,
                q.qtten,
                q.qvten,
                q.qcten,
                cupy.uint32(s.nx),
                cupy.uint32(s.ny),
                cupy.uint32(s.nz),
                cupy.float32(s.dt_s),
            ),
            stream=state.stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)
    state.solve1_ready = True
    state.ready_event = event
    return event


def launch_cuda_gm1_child_rk_stage(
    state: CudaGm1ChildStepState,
    *,
    nrk: int,
    post_stage_hook: Callable[[CudaGm1ChildStepState, int], object] | None = None,
) -> dict[str, Any]:
    """Enqueue one source-ordered stretched solve2 RK stage."""

    if not state.solve1_ready or nrk != state.next_stage or nrk not in (1, 2, 3):
        raise ValueError(f"real child RK cadence requires stage {state.next_stage}")
    s, arrays, metrics, c = (
        state.selection,
        state.prepared.arrays,
        state.prepared.metrics,
        state.closure,
    )
    with state.stream:
        launch_cuda_eqtset2_stage_terms(
            state.moist_bridge,
            arrays,
            metrics,
            th0=state.product.field("th0"),
            qv0=state.product.field("qv0"),
            pi0=state.product.field("pi0"),
            th3d=state.product.field("th3d"),
            q3d=state.product.field("q3d"),
            stream=state.stream,
        )
        _seed_momentum_and_stage_boundary(state, nrk)
        momentum = launch_cuda_gm1_child_stretched_momentum5(
            arrays,
            metrics,
            state.vertical,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            nrk=nrk,
            dx=s.dx_m,
            dy=s.dy_m,
            divx=state.moist_bridge.divx,
            stream=state.stream,
            compiled=state.momentum_advection_compiled,
        )
        launch_cuda_moist_buoyancy(
            state.moist_bridge, arrays, metrics, stream=state.stream
        )
        acoustic = launch_cuda_gm1_child_psolver3_open_stage(
            arrays,
            metrics,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            nrk=nrk,
            dt=s.dt_s,
            nsound=s.acoustic_substeps,
            dx=s.dx_m,
            dy=s.dy_m,
            vertical=state.vertical,
            stream=state.stream,
            compiled=state.psolver_compiled,
            open_compiled=state.open_compiled,
            lateral_compiled=state.lateral_compiled,
        )
        post = launch_cuda_gm1_child_stretched_post_acoustic(
            state.moist_bridge,
            arrays,
            metrics,
            state.vertical,
            dx=s.dx_m,
            dy=s.dy_m,
            stream=state.stream,
            compiled=state.stretched_moist_compiled,
        )
        scalar = _launch_cuda_gm1_child_stretched_scalar_stage(
            state.scalar_rk,
            {name: arrays[name] for name in ("rru", "rrv", "rrw")},
            metrics,
            state.moist_bridge.divx,
            state.moist_bridge.theta_divergence_term,
            state.vertical,
            nrk=nrk,
            dt=s.dt_s,
            dx=s.dx_m,
            dy=s.dy_m,
            stream=state.stream,
            advection_compiled=state.scalar_advection_compiled,
            pdef_compiled=state.scalar_pdef_compiled,
            open_compiled=state.open_compiled,
        )
        c.tke["divx"] = state.moist_bridge.divx
        tke = launch_cuda_gm1_child_stretched_tke_rk_stage(
            c.tke,
            workspace=c.tke_workspace,
            metrics=c.sgs_tke_metrics,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            nrk=nrk,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            ztop=state.vertical.ztop_f32,
            dt=s.dt_s,
            stream=state.stream,
            compiled=c.sgs_tke_compiled,
        )
        hook_result = None if post_stage_hook is None else post_stage_hook(state, nrk)
        event = importlib.import_module("cupy").cuda.Event(disable_timing=True)
        event.record(state.stream)
    c.completed_stage = nrk
    c.ready_event = event
    c.retained.extend((momentum, acoustic, post, scalar, tke, hook_result))
    state.retained.extend((momentum, acoustic, post, scalar, tke, hook_result))
    begin_cuda_moist_stage(state.moist_bridge)
    state.next_stage = nrk + 1
    state.ready_event = event
    return {
        "schema": GM1_CHILD_STEP_SCHEMA,
        "phase": "solve2-rk-stage",
        "nrk": nrk,
        "source_order": [
            "solve2.F:490-627 pre-acoustic flux/divx",
            "momentum advection, buoyancy, psolver3/open",
            "solve2.F:1277-1307 post-acoustic getdiv/getdivx",
            "solve2.F:1310-1533 theta/q RK",
            "solve2.F:1708-1803 TKE RK on post-acoustic flux",
        ],
        "momentum": momentum.receipt,
        "acoustic": acoustic.receipt,
        "post_acoustic": dict(post.receipt),
        "scalar": dict(scalar.receipt),
        "tke": dict(tke.receipt),
        "post_stage_hook": post_stage_hook is not None,
        "legacy_uniform_z_operator_used": False,
    }


def _launch_cuda_gm1_child_apmasscon(state: CudaGm1ChildStepState) -> Any:
    """Launch the admitted dry-mass/EOS correction at the bridge callback seam."""

    s, p, m = state.selection, state.product, state.prepared.metrics
    ng = CM1_OPEN_NGXY
    return launch_cuda_apmasscon(
        {
            "rho": p.field("rho"),
            "ruh": state.prepared.arrays["uh"][ng : ng + s.nx],
            "rvh": state.prepared.arrays["vh"][ng : ng + s.ny],
            "rmh": m["rmh"],
            "pi0": state.pi0_3d,
            "th0": state.th0_3d,
            "th3d": p.field("th3d"),
            "qv": p.field("q3d")[0],
            "pp3d": p.field("pp3d"),
            "prs": p.field("prs"),
        },
        ni=s.nx,
        nj=s.ny,
        nk=s.nz,
        mass1=state.mass1_f64,
        stream=state.stream,
        compiled=state.apmasscon_compiled,
    )


def _carrier_pointer(value: Any) -> int:
    data = getattr(value, "data", None)
    pointer = getattr(data, "ptr", None)
    if pointer is not None:
        return int(pointer)
    ctypes_view = getattr(value, "ctypes", None)
    pointer = getattr(ctypes_view, "data", None)
    if pointer is not None:
        return int(pointer)
    raise TypeError("final-dynamics carrier exposes no stable data pointer")


def _canonical_receipt_token(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def launch_cuda_gm1_child_solve3_rho_boundary(
    state: CudaGm1ChildStepState,
    *,
    step_index: int,
    stream: Any,
) -> CudaGm1ChildSolve3RhoBoundary:
    """Publish physical rho to its exact one-halo open solve3 carrier."""

    if not isinstance(state, CudaGm1ChildStepState):
        raise TypeError("state must be CudaGm1ChildStepState")
    if isinstance(step_index, bool) or step_index != state.step_index:
        raise ValueError("solve3 rho boundary step index differs from composer cadence")
    if stream is not state.stream or int(stream.ptr) != int(state.stream.ptr):
        raise ValueError("solve3 rho boundary requires the composer CUDA stream")
    s, arrays, product = state.selection, state.prepared.arrays, state.product
    padded = arrays.get("rho")
    if padded is None or tuple(padded.shape) != (s.nz + 2, s.ny + 2, s.nx + 2):
        raise ValueError("prepared.rho must be the exact one-halo solve3 carrier")
    physical = product.field("rho")
    if tuple(physical.shape) != (s.nz, s.ny, s.nx):
        raise ValueError("product.rho differs from the exact physical child shape")
    kernels = getattr(state.open_compiled, "kernels", ())
    if len(kernels) < 5:
        raise ValueError("open-boundary compiler lacks exact bcp west/east and south/north")

    cupy = importlib.import_module("cupy")
    words = int(padded.size)
    grid = ((words + GM1_FORECAST_THREADS - 1) // GM1_FORECAST_THREADS, 1, 1)
    block = (GM1_FORECAST_THREADS, 1, 1)
    with stream:
        cupy.copyto(
            padded[1 : s.nz + 1, 1 : s.ny + 1, 1 : s.nx + 1], physical
        )
        for kernel in (kernels[3], kernels[4]):
            kernel(
                grid,
                block,
                (
                    padded,
                    cupy.uint32(s.nx),
                    cupy.uint32(s.ny),
                    cupy.uint32(s.nz),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = MappingProxyType(
        {
            "abi": GM1_CHILD_SOLVE3_RHO_BOUNDARY_ABI,
            "physical_source_pointer": _carrier_pointer(physical),
            "physical_source_shape": list(physical.shape),
            "producer_source_sha256": GM1_CHILD_SOLVE3_RHO_BOUNDARY_SOURCE_SHA256,
            "padded_destination_pointer": _carrier_pointer(padded),
            "padded_destination_shape": list(padded.shape),
            "physical_destination_slices": [
                [1, s.nz + 1, 1],
                [1, s.ny + 1, 1],
                [1, s.nx + 1, 1],
            ],
            "schema": GM1_CHILD_SOLVE3_RHO_BOUNDARY_SCHEMA,
            "source_order": (
                "solve3.F:309-312 after ppx bcp and before held recapture"
            ),
            "step_index": step_index,
            "stream_pointer": int(stream.ptr),
            "boundary_order": ["west/east", "south/north"],
            "same_stream_no_host_sync": True,
        }
    )
    result = CudaGm1ChildSolve3RhoBoundary(
        event, step_index, int(stream.ptr), receipt
    )
    state.retained.append(result)
    return result


def launch_cuda_gm1_child_completed_closure_refresh(
    state: CudaGm1ChildStepState,
    *,
    step_index: int,
    stream: Any,
    tke_commit_receipt: Mapping[str, Any],
) -> CudaGm1ChildCompletedClosureRefresh:
    """Prepare all live closure carriers after density and before publication."""

    if not isinstance(state, CudaGm1ChildStepState):
        raise TypeError("state must be CudaGm1ChildStepState")
    if (
        isinstance(step_index, bool)
        or step_index != state.step_index
    ):
        raise ValueError("closure refresh step index differs from composer cadence")
    completed_step_index = step_index + 1
    if stream is not state.stream or int(stream.ptr) != int(state.stream.ptr):
        raise ValueError("closure refresh requires the composer CUDA stream")
    if not isinstance(tke_commit_receipt, Mapping):
        raise TypeError("tke_commit_receipt must be the exact mapping from TKE commit")
    if state.last_completed_closure_refresh is not None and (
        state.last_completed_closure_refresh.step_index == step_index
    ):
        raise ValueError("closure was already refreshed for this completed step")
    if state.last_final_dynamics_sync is None or (
        state.last_final_dynamics_sync.step_index != step_index
    ):
        raise ValueError("closure refresh requires this step's final dynamics sync")
    if state.closure.completed_stage != 3:
        raise ValueError("closure refresh requires completed TKE RK stage 3")

    operators = _load_gm1_child_solve1_operator_functions()
    cupy = importlib.import_module("cupy")
    s, c = state.selection, state.closure
    refresh_index = c.refresh_index
    if refresh_index != completed_step_index:
        raise ValueError("closure refresh index differs from completed-step cadence")
    with stream:
        nm_def = operators.launch_nm_def(
            _gm1_child_live_nm_def_arrays(state),
            metrics=state.static_metrics,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            ztop=state.vertical.ztop_f32,
            refresh_index=refresh_index,
            stream=stream,
            compiled_calcdef=c.nm_def_compiled_calcdef,
            compiled_calcnm=c.nm_def_compiled_calcnm,
        )
        c.sgs.update(nm_def.arrays)
        c.stress.update(
            {
                name: nm_def.workspace[name]
                for name in ("s11", "s22", "s33", "s12", "s13", "s23")
            }
        )
        c.sgs["dissten"].fill(cupy.float32(0.0))
        c.sgs["tketen"].fill(cupy.float32(0.0))
        coefficients = launch_cuda_sgs_tke(
            c.sgs,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            stream=stream,
            compiled=c.sgs_compiled,
        )
        cupy.copyto(c.tke["tke3d"], c.sgs["tkea"])
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)

    persistent = MappingProxyType(
        {
            "nm": c.sgs["nm"],
            "defv": c.sgs["defv"],
            "defh": c.sgs["defh"],
            "grdscl": c.sgs["grdscl"][1:],
            "lenscl": c.sgs["lenscl"],
            "dissten": c.sgs["dissten"],
            "kmh": c.sgs["kmh"],
            "kmv": c.sgs["kmv"],
            "khh": c.sgs["khh"],
            "khv": c.sgs["khv"],
            "tkea": c.sgs["tkea"],
            "tke3d": c.tke["tke3d"],
            "tketen": c.sgs["tketen"],
        }
    )
    tke_commit_sha = _canonical_receipt_token(tke_commit_receipt)
    descriptor: dict[str, Any] = {
        "abi": GM1_CHILD_COMPLETED_CLOSURE_REFRESH_ABI,
        "completed_step_index": completed_step_index,
        "nm_def_receipt": dict(nm_def.receipt),
        "operation_source_sha256": GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SOURCE_SHA256,
        "persistent_ready": [
            {
                "name": name,
                "pointer": _carrier_pointer(value),
                "shape": list(value.shape),
            }
            for name, value in persistent.items()
        ],
        "refresh_index": refresh_index,
        "same_stream_no_host_sync": True,
        "schema": GM1_CHILD_COMPLETED_CLOSURE_REFRESH_SCHEMA,
        "sgs_receipt": dict(coefficients.receipt),
        "source_order": [
            "solve3 density publication",
            "cm1.F:3718-3788 live calcnm/calcdef",
            "turb.F:1326-1413 persistent-tkea tkekm coefficients",
            "checkpoint/history publication",
        ],
        "step_index": step_index,
        "stream_pointer": int(stream.ptr),
        "tke_commit_receipt_sha256": tke_commit_sha,
        "token_scope": (
            "same-stream completed-step cadence and exact device bindings; "
            "not a device-content hash"
        ),
    }
    operation_token = _canonical_receipt_token(descriptor)
    receipt = MappingProxyType(
        {**descriptor, "operation_token_sha256": operation_token}
    )
    result = CudaGm1ChildCompletedClosureRefresh(
        ready_event=event,
        step_index=step_index,
        completed_step_index=completed_step_index,
        refresh_index=refresh_index,
        stream_pointer=int(stream.ptr),
        tke_commit_receipt_sha256=tke_commit_sha,
        operation_token_sha256=operation_token,
        arrays=persistent,
        receipt=receipt,
    )
    c.last_nm_def = nm_def
    c.nm_def_compiled_calcdef = nm_def.compiled_calcdef
    c.nm_def_compiled_calcnm = nm_def.compiled_calcnm
    c.completed_stage = 0
    c.refresh_index += 1
    c.ready_event = event
    c.retained.extend((nm_def, coefficients, result))
    state.last_completed_closure_refresh = result
    state.ready_event = event
    state.retained.extend((nm_def, coefficients, result))
    return result


def sync_cuda_gm1_child_final_dynamics_to_product(
    state: CudaGm1ChildStepState,
) -> CudaGm1ChildFinalDynamicsSync:
    """Publish exact final RK3 physical slices before remove/EOS physics.

    The operation token seals cadence, stream, slice geometry, and the exact
    resident source/destination bindings.  It intentionally is not a hash of
    device words: obtaining such a hash here would introduce a host sync at an
    otherwise same-stream CM1 lifecycle seam.
    """

    if not isinstance(state, CudaGm1ChildStepState):
        raise TypeError("state must be CudaGm1ChildStepState")
    if (
        not state.solve1_ready
        or state.next_stage != 4
        or state.closure.completed_stage != 3
    ):
        raise ValueError("final dynamics sync requires completed RK stages 1, 2, and 3")
    if state.last_final_dynamics_sync is not None:
        raise ValueError("final dynamics were already synchronized for this large step")

    cupy = importlib.import_module("cupy")
    s, arrays, product = state.selection, state.prepared.arrays, state.product
    if int(state.stream.ptr) != int(product.stream.ptr):
        raise ValueError("final dynamics product and composer streams differ")
    ng = CM1_OPEN_NGXY
    copy_specs = {
        "u3d": (
            arrays["u3d"],
            (slice(1, s.nz + 1), slice(ng, ng + s.ny), slice(ng, ng + s.nx + 1)),
            product.field("u3d"),
        ),
        "v3d": (
            arrays["v3d"],
            (slice(1, s.nz + 1), slice(ng, ng + s.ny + 1), slice(ng, ng + s.nx)),
            product.field("v3d"),
        ),
        "w3d": (
            arrays["w3d"],
            (slice(1, s.nz + 2), slice(ng, ng + s.ny), slice(ng, ng + s.nx)),
            product.field("w3d"),
        ),
        "pp3d": (
            arrays["pp3d"],
            (slice(1, s.nz + 1), slice(1, s.ny + 1), slice(1, s.nx + 1)),
            product.field("pp3d"),
        ),
    }
    copies: dict[str, Any] = {}
    for name, (source, source_slice, destination) in copy_specs.items():
        view = source[source_slice]
        if tuple(view.shape) != tuple(destination.shape):
            raise ValueError(
                f"final dynamics {name} slice shape {tuple(view.shape)} differs "
                f"from product shape {tuple(destination.shape)}"
            )
        copies[name] = {
            "source_pointer": _carrier_pointer(source),
            "destination_pointer": _carrier_pointer(destination),
            "source_shape": list(source.shape),
            "source_slices": [
                [part.start, part.stop, 1 if part.step is None else part.step]
                for part in source_slice
            ],
            "destination_shape": list(destination.shape),
        }

    descriptor: dict[str, Any] = {
        "abi": GM1_CHILD_FINAL_DYNAMICS_SYNC_ABI,
        "completed_nrk": 3,
        "copies": copies,
        "producer_source_sha256": GM1_CHILD_FINAL_DYNAMICS_SYNC_SOURCE_SHA256,
        "schema": GM1_CHILD_FINAL_DYNAMICS_SYNC_SCHEMA,
        "source_order": "after completed solve2 RK3; before remove/EOS/apmasscon",
        "step_index": state.step_index,
        "stream_pointer": int(state.stream.ptr),
        "token_scope": (
            "same-stream enqueue and exact device bindings; not a device-content hash"
        ),
    }
    operation_token = _canonical_receipt_token(descriptor)
    receipt = MappingProxyType(
        {**descriptor, "operation_token_sha256": operation_token}
    )
    with state.stream:
        for source, source_slice, destination in copy_specs.values():
            cupy.copyto(destination, source[source_slice])
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)
    result = CudaGm1ChildFinalDynamicsSync(
        ready_event=event,
        step_index=state.step_index,
        stream_pointer=int(state.stream.ptr),
        operation_token_sha256=operation_token,
        receipt=receipt,
    )
    state.last_final_dynamics_sync = result
    state.ready_event = event
    state.retained.append(result)
    return result


def finish_cuda_gm1_child_large_step(
    state: CudaGm1ChildStepState,
    *,
    operator_sources: Mapping[str, str],
    reflectivity_due: bool = False,
) -> dict[str, Any]:
    """Finish one source-ordered child step and prepare the next solve1 state."""

    if not isinstance(state, CudaGm1ChildStepState):
        raise TypeError("state must be CudaGm1ChildStepState")
    if (
        state.next_stage != 4
        or not state.solve1_ready
        or state.closure.completed_stage != 3
    ):
        raise ValueError("real child finish requires completed RK stages 1, 2, and 3")
    if (
        not state.scalar_rk.state.completed
        or state.scalar_rk.state.next_nrk != 4
    ):
        raise ValueError("real child scalar/moisture RK cadence is not at completed stage 3")
    if state.post_final.next_step_index != state.step_index:
        raise ValueError("post-final transition and composer step indices differ")
    if state.closure.refresh_index != state.step_index + 1:
        raise ValueError("completed-step closure refresh index is stale or skipped")
    if type(reflectivity_due) is not bool:
        raise TypeError("reflectivity_due must be bool")

    # The sync mutates the product.  Validate the complete baseline-plus-child
    # operator identity first so any missing or stale source refuses before a
    # single live device word changes.
    operator_manifest_sha256 = require_child_post_final_rk_operator_sources(
        operator_sources
    )
    final_sync = sync_cuda_gm1_child_final_dynamics_to_product(state)

    def require_callback_cadence(step_index: int, stream: Any, label: str) -> None:
        if isinstance(step_index, bool) or step_index != state.step_index:
            raise ValueError(f"{label} callback step differs from composer cadence")
        if stream is not state.stream or int(stream.ptr) != int(state.stream.ptr):
            raise ValueError(f"{label} callback stream differs from composer stream")

    def apmasscon_hook(step_index: int, stream: Any) -> Any:
        require_callback_cadence(step_index, stream, "apmasscon")
        return _launch_cuda_gm1_child_apmasscon(state)

    def rho_boundary_hook(step_index: int, stream: Any) -> Any:
        require_callback_cadence(step_index, stream, "solve3 rho-boundary")
        return launch_cuda_gm1_child_solve3_rho_boundary(
            state, step_index=step_index, stream=stream
        )

    def tke_commit_hook(step_index: int, stream: Any) -> Any:
        require_callback_cadence(step_index, stream, "solve3 TKE commit")
        selection = state.selection
        return launch_cuda_gm1_child_stretched_tke_commit(
            state.closure.tke,
            ni=selection.nx,
            nj=selection.ny,
            nk=selection.nz,
            completed_nrk=3,
            stream=stream,
            compiled=state.closure.sgs_tke_compiled,
        )

    def closure_refresh_hook(
        step_index: int,
        stream: Any,
        tke_commit_receipt: Mapping[str, Any],
    ) -> Any:
        require_callback_cadence(step_index, stream, "completed closure refresh")
        return launch_cuda_gm1_child_completed_closure_refresh(
            state,
            step_index=step_index,
            stream=stream,
            tke_commit_receipt=tke_commit_receipt,
        )

    result = launch_cuda_gm1_child_post_final_rk(
        state.post_final,
        cadence=Gm1ChildPostFinalRkCadence(
            step_index=state.step_index,
            apmasscon_target_step_index=state.step_index,
        ),
        operator_sources=operator_sources,
        final_dynamics_sync=final_sync,
        mass1_f64=state.mass1_f64,
        apmasscon_hook=apmasscon_hook,
        rho_boundary_hook=rho_boundary_hook,
        tke_commit_hook=tke_commit_hook,
        closure_refresh_hook=closure_refresh_hook,
        reflectivity_due=reflectivity_due,
    )
    completed_step = state.step_index + 1
    if (
        result.completed_step_index != completed_step
        or result.operator_manifest_sha256 != operator_manifest_sha256
        or result.stream_pointer != int(state.stream.ptr)
    ):
        state.post_final.poisoned = True
        raise RuntimeError("post-final result differs from composer cadence or identity")
    if state.last_completed_closure_refresh is not result.closure_refresh:
        state.post_final.poisoned = True
        raise RuntimeError("post-final result did not retain the exact closure refresh")

    begin_cuda_scalar_moist_rk_step(state.scalar_rk.state)
    state.step_index = completed_step
    state.time_seconds = float(completed_step) * state.selection.dt_s
    state.next_stage = 1
    state.solve1_ready = False
    state.ready_event = result.ready_event
    state.retained.append(result)
    receipt = {
        "schema": GM1_CHILD_STEP_SCHEMA,
        "abi": GM1_CHILD_STEP_ABI,
        "phase": "completed-large-step",
        "step_index": completed_step,
        "model_time_seconds": state.time_seconds,
        "dt_seconds": state.selection.dt_s,
        "source_order": [
            "completed stretched solve2 RK3",
            "final padded dynamics to physical product sync",
            "remove/snapshot and EOS",
            "apmasscon and live pp3d/prs/rho rebuild",
            "pdefq then WSM6",
            "solve3 ppx/rho BC then held recapture",
            "momentum current-to-A then stretched TKE commit",
            "ppi/tha/qa current-to-A then rr/rf density",
            "completed-step live nm/def/coefficients refresh",
        ],
        "operator_manifest_sha256": operator_manifest_sha256,
        "final_dynamics_sync_operation_token_sha256": (
            final_sync.operation_token_sha256
        ),
        "completed_closure_refresh_operation_token_sha256": (
            result.closure_refresh.operation_token_sha256
        ),
        "post_final": dict(result.receipt),
        "next_step": {
            "next_stage": state.next_stage,
            "solve1_ready": state.solve1_ready,
            "scalar_next_nrk": state.scalar_rk.state.next_nrk,
            "closure_refresh_index": state.closure.refresh_index,
        },
        "publication": {
            "phase": "quiescent completed large-step after ready_event",
            "rust_pair_directory": f"step-{completed_step:08d}",
            "viewer_ready_signal": "atomic final pair-directory appearance",
        },
        "legacy_uniform_z_operator_used": False,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    state.last_receipt = receipt
    return receipt
