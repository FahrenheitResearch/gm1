"""GM1 single-GPU real-atmosphere forecast coordinator.

GM1 is the user-facing GPU model.  CM1 r21.1 remains the immutable numerical
authority recorded in receipts and source anchors.  This module composes the
already admitted device operators in their large-step order; Python performs
only validation, launch orchestration, and receipt construction.

The solve1 closure producer is injected as a strict device callback so the
selected SGS/TKE and upper-Rayleigh lanes can remain independently admitted.
No zero-closure product run is accepted: omission is available only behind the
explicit ``allow_incomplete_validation`` flag.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cm1core.cuda_advection5 import (
    CompiledAdvection5Kernels,
    compile_cuda_advection5,
)
from cm1core.cuda_apmasscon import (
    CompiledApMassConKernels,
    compile_cuda_apmasscon,
    launch_cuda_apmasscon,
)
from cm1core.cuda_wsm6_pdefq import (
    CompiledWsm6Pdefq,
    CudaWsm6PdefqLaunch,
    Wsm6PdefqWorkspace,
    compile_cuda_wsm6_pdefq,
    create_cuda_wsm6_pdefq_workspace,
    launch_cuda_wsm6_pdefq_pdscheme1,
)
from cm1core.cuda_initialized_product import CudaInitializedProduct
from cm1core.cuda_initialized_product_terrain_open import (
    CudaInitializedTerrainOpenState,
    prepare_cuda_initialized_product_terrain_open,
)
from cm1core.cuda_gm1_closure import (
    CudaGm1ClosureState,
    create_cuda_gm1_closure_state,
    finish_cuda_gm1_closure_step,
    launch_cuda_gm1_closure_solve1,
    launch_cuda_gm1_closure_stage,
    refresh_cuda_gm1_closure_for_next_step,
)
from cm1core.cuda_gm1_kessler_large_step import (
    CudaGm1KesslerLargeStepTransition,
    GM1_KESSLER_LARGE_STEP_SEMANTIC_ID,
)
from cm1core.cuda_lateral_closure import (
    CompiledLateralClosureKernels,
    compile_cuda_lateral_closure,
)
from cm1core.cuda_moist_stage_bridge import (
    CudaMoistStageBridgeState,
    begin_cuda_moist_stage,
    compile_cuda_moist_stage_bridge,
    create_cuda_moist_stage_bridge_state,
    launch_cuda_eqtset2_stage_terms,
    launch_cuda_moist_buoyancy,
    launch_cuda_post_acoustic_flux_bridge,
)
from cm1core.cuda_momentum_advection5 import (
    CompiledMomentumAdvection5Kernels,
    compile_cuda_momentum_advection5,
    launch_cuda_terrain_open_momentum_advection5,
)
from cm1core.cuda_momentum_rk import (
    CUDA_MOMENTUM_RK_THREADS,
    CompiledMomentumRKKernels,
    compile_cuda_momentum_rk,
    momentum_rk_stage_timestep_f32,
)
from cm1core.cuda_psolver3_terrain_open import (
    CompiledPsolver3TerrainOpenKernels,
    compile_cuda_psolver3_terrain_open,
    launch_cuda_psolver3_terrain_open_stage,
)
from cm1core.cuda_scalar_moist_rk import (
    CompiledScalarMoistRKKernels,
    CudaScalarMoistRKState,
    begin_cuda_scalar_moist_rk_step,
    compile_cuda_scalar_moist_rk,
    create_cuda_scalar_moist_rk_state,
    launch_cuda_scalar_moist_rk_stage,
)
from cm1core.cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    CompiledOpenBoundaryKernels,
)
from cm1core.cuda_terrain_open_boundary import (
    CompiledTerrainOpenBoundaryKernels,
)
from cm1core.initialized_product_wsm6_transition import (
    CudaWsm6Geopotential,
    InitializedProductWsm6Transition,
    initialize_product_wsm6_transition,
)
from cm1core.errors import refuse


GM1_FORECAST_SCHEMA = "gm1.cuda-real-forecast/v1"
GM1_FORECAST_ABI = "gm1.cuda-real-forecast-kernels/v1"
GM1_FORECAST_THREADS = 128
GM1_WSM6_MICROPHYSICS_SEMANTIC_ID = "wsm6-mp6"
GM1_HRRR_LATERAL_FORCING_SEMANTIC_ID = "hrrr-lbc-quadratic-v1"
GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID = "hrrr-lbc-davies-c2-v1"
GM1_TERRAIN_OPEN_MICROPHYSICS = frozenset(
    (GM1_WSM6_MICROPHYSICS_SEMANTIC_ID, GM1_KESSLER_LARGE_STEP_SEMANTIC_ID)
)
GM1_HRRR_INITIALIZATION_SOURCE_OFFSETS_S = {
    "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f": 0.0,
    "28e89cae538cf349c713ea42630dd8cc664a084c0c88f71eac1b3a491f14d189": 3600.0,
}

AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = {
    "src/solve1.F": "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076",
    "src/solve2.F": "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
    "src/solve3.F": "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393",
    "src/cm1.F": "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
}
SOURCE_ANCHORS = {
    "rk_stage_timestep": "src/solve2.F:299",
    "pressure_held_prediction": "src/solve1.F:1784-1838",
    "rk_tendency_restore": "src/solve2.F:459-471",
    "terrain_stage_boundary": "src/solve2.F:475-488",
    "mass_flux_and_momentum": "src/solve2.F:490-1010",
    "buoyancy_and_acoustics": "src/solve2.F:1012-1285",
    "theta_moisture_rk": "src/solve2.F:1450-1660",
    "final_microphysics_remove": "src/solve2.F:1688-1710",
    "dry_mass_correction": "src/solve2.F:1742-1796,2093-2145",
    "microphysics": "src/cm1.F:3405-3432",
    "solve3_ppx_and_held": "src/solve3.F:282-337",
    "solve3_a_and_density": "src/solve3.F:548-655",
}


@dataclass(frozen=True, slots=True)
class Gm1ForecastSelection:
    nx: int = 300
    ny: int = 300
    nz: int = 75
    dx_m: float = 100.0
    dy_m: float = 100.0
    dz_m: float = 200.0
    dt_s: float = 0.5
    acoustic_substeps: int = 6
    equation_set: int = 2
    psolver: int = 3
    terrain: bool = True
    lateral_boundaries: tuple[int, int, int, int] = (2, 2, 2, 2)
    microphysics: str = GM1_WSM6_MICROPHYSICS_SEMANTIC_ID
    dry_mass_correction: bool = True
    coriolis: int = 0
    moving_frame: bool = False
    #: CM1's model top.  ``param.F:2608`` derives it as ``dz*nk`` only on
    #: ``param.F:7789-7811``'s uniform column; on any ``param.F:7814-8139``
    #: branch ``dz_m`` is the metric reference of ``param.F:8170`` and the top
    #: is ``sigmaf(nk+1)``.  ``None`` means "take param.F:2608's derivation",
    #: which is exactly right for every uniform selection.
    ztop_m: float | None = None

    @property
    def model_top_m(self) -> float:
        """This selection's model top, derived or declared."""

        return self.nz * self.dz_m if self.ztop_m is None else float(self.ztop_m)


GM1_REAL_100M_PRESET = Gm1ForecastSelection()
DEFAULT_GM1_SELECTION = GM1_REAL_100M_PRESET


@dataclass(frozen=True, slots=True)
class Gm1ForecastSourceClock:
    """Absolute HRRR source time bound to relative GM1 integration time."""

    initialization_source_offset_s: float
    frame_start_source_offset_s: float
    frame_end_source_offset_s: float
    relative_interval_start_s: float
    relative_interval_end_s: float
    initialization_source_forecast_hour: int
    frame_start_source_forecast_hour: int
    frame_end_source_forecast_hour: int


@dataclass(frozen=True, slots=True)
class Gm1ForecastSourceClockPoint:
    relative_model_time_s: float
    absolute_source_time_s: float
    temporal_interpolation_fraction: float


@dataclass(frozen=True, slots=True)
class Gm1ForecastLateralForcing:
    """One source-clock-pinned, continuous edge-zone stage callback."""

    source_clock: Gm1ForecastSourceClock
    apply_stage: Callable[
        ["CudaGm1ForecastState", int, Gm1ForecastSourceClockPoint], object
    ]
    relaxation_width_points: int = 20
    semantic_id: str = GM1_HRRR_LATERAL_FORCING_SEMANTIC_ID
    relaxation_mode: str = "legacy-direct-quadratic"
    relaxation_timescale_s: float | None = None


GM1_HRRR_F00_F01_SOURCE_CLOCK = Gm1ForecastSourceClock(
    initialization_source_offset_s=0.0,
    frame_start_source_offset_s=0.0,
    frame_end_source_offset_s=3600.0,
    relative_interval_start_s=0.0,
    relative_interval_end_s=3600.0,
    initialization_source_forecast_hour=0,
    frame_start_source_forecast_hour=0,
    frame_end_source_forecast_hour=1,
)
GM1_HRRR_F01_F02_SOURCE_CLOCK = Gm1ForecastSourceClock(
    initialization_source_offset_s=3600.0,
    frame_start_source_offset_s=3600.0,
    frame_end_source_offset_s=7200.0,
    relative_interval_start_s=0.0,
    relative_interval_end_s=3600.0,
    initialization_source_forecast_hour=1,
    frame_start_source_forecast_hour=1,
    frame_end_source_forecast_hour=2,
)


def validate_gm1_source_clock(clock: Gm1ForecastSourceClock) -> None:
    if not isinstance(clock, Gm1ForecastSourceClock):
        raise TypeError("clock must be a Gm1ForecastSourceClock")
    numeric = {
        "initialization_source_offset_s": clock.initialization_source_offset_s,
        "frame_start_source_offset_s": clock.frame_start_source_offset_s,
        "frame_end_source_offset_s": clock.frame_end_source_offset_s,
        "relative_interval_start_s": clock.relative_interval_start_s,
        "relative_interval_end_s": clock.relative_interval_end_s,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in numeric.values()
    ):
        refuse(
            code="GM1-HRRR-LBC-SOURCE-CLOCK-NONFINITE",
            gate="boundaries-closures",
            option="source_clock",
            value=numeric,
            reason="HRRR source and relative offsets must be finite nonnegative seconds",
            remedy="bind the exact absolute offsets from the Rust forcing-series receipt",
        )
    hours = {
        "initialization": clock.initialization_source_forecast_hour,
        "frame_start": clock.frame_start_source_forecast_hour,
        "frame_end": clock.frame_end_source_forecast_hour,
    }
    if any(type(value) is not int or value < 0 for value in hours.values()):
        raise ValueError("HRRR source forecast hours must be nonnegative integers")
    expected_offsets = {
        "initialization": 3600.0 * hours["initialization"],
        "frame_start": 3600.0 * hours["frame_start"],
        "frame_end": 3600.0 * hours["frame_end"],
    }
    actual_offsets = {
        "initialization": float(clock.initialization_source_offset_s),
        "frame_start": float(clock.frame_start_source_offset_s),
        "frame_end": float(clock.frame_end_source_offset_s),
    }
    if actual_offsets != expected_offsets:
        refuse(
            code="GM1-HRRR-LBC-SOURCE-HOUR-MISMATCH",
            gate="boundaries-closures",
            option="source_clock",
            value={"hours": hours, "offsets_s": actual_offsets},
            reason="absolute source offsets do not match their sealed HRRR forecast hours",
            remedy="use f00=0 s, f01=3600 s, f02=7200 s, and f03=10800 s exactly",
        )
    source_span = float(clock.frame_end_source_offset_s) - float(
        clock.frame_start_source_offset_s
    )
    relative_span = float(clock.relative_interval_end_s) - float(
        clock.relative_interval_start_s
    )
    if source_span <= 0.0 or relative_span <= 0.0 or source_span != relative_span:
        refuse(
            code="GM1-HRRR-LBC-SOURCE-INTERVAL",
            gate="boundaries-closures",
            option="source_clock",
            value=numeric,
            reason="forcing intervals must be positive and equal in absolute and relative time",
            remedy="bind one contiguous source interval without stretching or compressing its clock",
        )
    if float(clock.frame_start_source_offset_s) != float(
        clock.initialization_source_offset_s
    ) + float(clock.relative_interval_start_s) or float(
        clock.frame_end_source_offset_s
    ) != float(clock.initialization_source_offset_s) + float(
        clock.relative_interval_end_s
    ):
        refuse(
            code="GM1-HRRR-LBC-SOURCE-OFFSET-MISMATCH",
            gate="boundaries-closures",
            option="source_clock",
            value=numeric,
            reason="relative GM1 time does not map contiguously onto the absolute HRRR source clock",
            remedy="for f01 initialization use absolute [3600,7200] with relative [0,3600]",
        )


def gm1_source_clock_point(
    clock: Gm1ForecastSourceClock, relative_model_time_s: float
) -> Gm1ForecastSourceClockPoint:
    validate_gm1_source_clock(clock)
    if (
        isinstance(relative_model_time_s, bool)
        or not isinstance(relative_model_time_s, (int, float))
        or not math.isfinite(float(relative_model_time_s))
    ):
        raise TypeError("relative_model_time_s must be finite")
    relative = float(relative_model_time_s)
    start = float(clock.relative_interval_start_s)
    end = float(clock.relative_interval_end_s)
    if relative < start or relative > end:
        refuse(
            code="GM1-HRRR-LBC-SOURCE-CLOCK-RANGE",
            gate="boundaries-closures",
            option="relative_model_time_s",
            value=relative,
            reason="the requested GM1 time lies outside the independently supplied forcing interval",
            remedy="supply the next contiguous HRRR boundary frame before advancing",
        )
    absolute = float(clock.initialization_source_offset_s) + relative
    alpha = (absolute - float(clock.frame_start_source_offset_s)) / (
        float(clock.frame_end_source_offset_s)
        - float(clock.frame_start_source_offset_s)
    )
    return Gm1ForecastSourceClockPoint(relative, absolute, alpha)


def validate_gm1_lateral_forcing(forcing: Gm1ForecastLateralForcing) -> None:
    if not isinstance(forcing, Gm1ForecastLateralForcing):
        raise TypeError("lateral_forcing must be a Gm1ForecastLateralForcing")
    validate_gm1_source_clock(forcing.source_clock)
    admitted_semantics = {
        GM1_HRRR_LATERAL_FORCING_SEMANTIC_ID: "legacy-direct-quadratic",
        GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID: ("davies-c2-stage-tendency"),
    }
    if forcing.semantic_id not in admitted_semantics:
        refuse(
            code="GM1-LATERAL-FORCING-SEMANTIC",
            gate="boundaries-closures",
            option="semantic_id",
            value=forcing.semantic_id,
            reason="the terrain/open coordinator admits only audited real-HRRR edge drivers",
            remedy=(
                "select the legacy direct-quadratic semantic or the corrected "
                "Davies C2 stage-tendency semantic"
            ),
        )
    expected_mode = admitted_semantics[forcing.semantic_id]
    if forcing.relaxation_mode != expected_mode:
        refuse(
            code="GM1-HRRR-LBC-RELAXATION-MODE",
            gate="boundaries-closures",
            option="relaxation_mode",
            value=forcing.relaxation_mode,
            reason="the forcing semantic and relaxation algorithm do not agree",
            remedy=f"bind relaxation_mode={expected_mode!r} for this semantic",
        )
    if expected_mode == "legacy-direct-quadratic":
        if forcing.relaxation_timescale_s is not None:
            raise ValueError("legacy direct relaxation may not declare a timescale")
    elif (
        isinstance(forcing.relaxation_timescale_s, bool)
        or not isinstance(forcing.relaxation_timescale_s, (int, float))
        or not math.isfinite(float(forcing.relaxation_timescale_s))
        or float(forcing.relaxation_timescale_s) <= 0.0
    ):
        refuse(
            code="GM1-HRRR-LBC-RELAXATION-TIMESCALE",
            gate="boundaries-closures",
            option="relaxation_timescale_s",
            value=forcing.relaxation_timescale_s,
            reason="Davies tendency forcing requires a finite positive timescale",
            remedy="bind a source-receipted tau no shorter than the GM1 large timestep",
        )
    if not callable(forcing.apply_stage):
        raise TypeError("lateral forcing apply_stage must be callable")
    if (
        isinstance(forcing.relaxation_width_points, bool)
        or not isinstance(forcing.relaxation_width_points, int)
        or forcing.relaxation_width_points < 2
    ):
        refuse(
            code="GM1-HRRR-LBC-BLEND-WIDTH",
            gate="boundaries-closures",
            option="relaxation_width_points",
            value=forcing.relaxation_width_points,
            reason="a one-cell or invalid edge replacement recreates a discontinuous rectangular front",
            remedy="use a multi-cell continuous relaxation zone",
        )


def validate_gm1_selection(
    selection: Gm1ForecastSelection,
    *,
    allow_cropped_validation: bool = False,
) -> None:
    """Validate the model-level configuration independently of a backend.

    A resolution is a preset choice, not a GM1 model boundary.  Concrete
    execution backends perform their own capability dispatch after this
    structural validation.
    """

    if not isinstance(selection, Gm1ForecastSelection):
        raise TypeError("selection must be a Gm1ForecastSelection")
    for name, minimum in (("nx", 1), ("ny", 1), ("nz", 3), ("acoustic_substeps", 1)):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("dx_m", "dy_m", "dz_m", "dt_s"):
        value = getattr(selection, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    if selection.equation_set not in (1, 2):
        raise ValueError("equation_set must select CM1 branch 1 or 2")
    if selection.psolver not in range(1, 8):
        raise ValueError("psolver must select a CM1 branch in 1..7")
    if (
        not isinstance(selection.lateral_boundaries, tuple)
        or len(selection.lateral_boundaries) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in selection.lateral_boundaries
        )
    ):
        raise ValueError("lateral_boundaries must be four positive CM1 branch integers")
    if (
        not isinstance(selection.microphysics, str)
        or not selection.microphysics.strip()
    ):
        raise ValueError("microphysics must be a non-empty scheme identifier")
    for name in ("terrain", "dry_mass_correction", "moving_frame"):
        if not isinstance(getattr(selection, name), bool):
            raise TypeError(f"{name} must be boolean")
    if isinstance(selection.coriolis, bool) or not isinstance(selection.coriolis, int):
        raise TypeError("coriolis must be an integer CM1 branch selector")
    if not isinstance(allow_cropped_validation, bool):
        raise TypeError("allow_cropped_validation must be boolean")


def validate_gm1_terrain_open_backend(
    selection: Gm1ForecastSelection,
    *,
    allow_cropped_validation: bool = False,
) -> None:
    """Dispatch guard for the currently selected terrain/open backend."""

    validate_gm1_selection(selection, allow_cropped_validation=allow_cropped_validation)
    expected = GM1_REAL_100M_PRESET
    actual = {name: getattr(selection, name) for name in selection.__dataclass_fields__}
    baseline = {name: getattr(expected, name) for name in expected.__dataclass_fields__}
    microphysics = actual.pop("microphysics")
    baseline.pop("microphysics")
    if microphysics not in GM1_TERRAIN_OPEN_MICROPHYSICS:
        raise ValueError(
            "configuration is valid GM1 input but its microphysics semantic is "
            "not routed to the terrain/open psolver=3 backend"
        )
    if allow_cropped_validation:
        for name in ("nx", "ny"):
            value = actual.pop(name)
            baseline.pop(name)
            if value < 10:
                raise ValueError(
                    f"{name} must be an integer >= 10 for cropped validation"
                )
        if actual == baseline:
            return
    elif actual == baseline:
        return
    raise ValueError(
        "configuration is valid GM1 input but is not routed to the terrain/open "
        "psolver=3 backend; select or implement its backend dispatch"
    )


def validate_gm1_initialized_product_microphysics(
    selection: Gm1ForecastSelection,
) -> None:
    """Validate the microphysics identity of this WSM6 product constructor."""

    validate_gm1_selection(selection)
    if selection.microphysics == GM1_WSM6_MICROPHYSICS_SEMANTIC_ID:
        return
    if selection.microphysics == GM1_KESSLER_LARGE_STEP_SEMANTIC_ID:
        refuse(
            code="GM1-KESSLER-REAL-INITIALIZATION-UNAVAILABLE",
            gate="physics-seam",
            option="microphysics",
            value=selection.microphysics,
            reason=(
                "the admitted real GM1 product is explicitly WSM6-initialized with "
                "qv/qc/qr/qi/qs/qg, while CM1 r21.1 ptype=1 initializes exactly "
                "qv/qc/qr and defines no qi/qs/qg-to-Kessler conversion"
            ),
            remedy=(
                "provide a source-attested native ptype=1 initialized export; do not "
                "relabel WSM6 state or collapse frozen species into warm rain"
            ),
        )
    refuse(
        code="GM1-MICROPHYSICS-DISPATCH",
        gate="physics-seam",
        option="microphysics",
        value=selection.microphysics,
        reason="the selected microphysics semantic has no initialized-product transition",
        remedy="select an admitted semantic or implement and independently validate its initializer",
    )


GM1_FORECAST_KERNEL_NAMES = (
    "gm1_predict_and_compose_held_v1",
    "gm1_solve3_density_v1",
    "gm1_sum_initial_mass_v1",
    "gm1_broadcast_base_profiles_v1",
)

GM1_FORECAST_CUDA_SOURCE = r"""// gm1.cuda-real-forecast-kernels/v1
static_assert(sizeof(float)==4,"GM1 default REAL requires float32");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ unsigned long long p3(unsigned int k,unsigned int j,unsigned int i,unsigned int ni,unsigned int nj){return ((unsigned long long)(k+1u)*(nj+2u)+(j+1u))*(ni+2u)+(i+1u);}
extern "C" __global__ void gm1_predict_and_compose_held_v1(
 const float* th0,const float* pi0,const float* rho,const float* ppi,const float* tha,const float* qa,
 float* thheld,float* qheld,float* ppten,const float* qpten,const float* qtten,const float* qvten,const float* qcten,
 unsigned int ni,unsigned int nj,unsigned int nk,float dt){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk;
 const float rd=287.04f,rv=461.5f,rp00=1.0f/100000.0f,rddcv=287.04f/(1005.7f-287.04f),tem=0.0001f*0.0001f,qsmall=1.0e-14f;
 for(unsigned long long z=first;z<n;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),k=(unsigned int)(z/((unsigned long long)ni*nj));
  float ht=thheld[z],qvt=qheld[z],pinew=0.0f;
  if(fabsf(m32(dt,ht))>tem||fabsf(m32(dt,qvt))>qsmall){float thnew=a32(tha[z],m32(dt,ht)),qvnew=a32(qa[z],m32(dt,qvt));float base=m32(m32(rho[z],a32(th0[k],thnew)),a32(rd,m32(rv,qvnew)));base=m32(base,rp00);pinew=s32(powf(base,rddcv),pi0[k]);ppten[p3(k,j,i,ni,nj)]=a32(d32(s32(pinew,ppi[z]),dt),qpten[z]);}
  else{ppten[p3(k,j,i,ni,nj)]=qpten[z];}
  thheld[z]=a32(ht,qtten[z]);qheld[z]=a32(qvt,qvten[z]);qheld[n+z]=a32(qheld[n+z],qcten[z]);
 }}
extern "C" __global__ void gm1_solve3_density_v1(
 const float* rho,const float* c1,const float* c2,float* rr,float* rf,unsigned int ni,unsigned int nj,unsigned int nk){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk,nf=(unsigned long long)ni*nj*(nk+1u);
 for(unsigned long long z=first;z<n;z+=step)rr[z]=d32(1.0f,rho[z]);
 for(unsigned long long z=first;z<nf;z+=step){unsigned int i=(unsigned int)(z%ni),j=(unsigned int)((z/ni)%nj),f=(unsigned int)(z/((unsigned long long)ni*nj));unsigned long long h=(unsigned long long)j*ni+i;
  if(f==0u)rf[z]=a32(a32(m32(1.875f,rho[h]),m32(-1.25f,rho[(unsigned long long)nj*ni+h])),m32(0.375f,rho[2ull*nj*ni+h]));
  else if(f==nk)rf[z]=a32(a32(m32(1.875f,rho[(unsigned long long)(nk-1u)*nj*ni+h]),m32(-1.25f,rho[(unsigned long long)(nk-2u)*nj*ni+h])),m32(0.375f,rho[(unsigned long long)(nk-3u)*nj*ni+h]));
  else{unsigned long long lo=(unsigned long long)(f-1u)*nj*ni+h,hi=(unsigned long long)f*nj*ni+h;rf[z]=a32(m32(c1[z],rho[lo]),m32(c2[z],rho[hi]));}
 }}
extern "C" __global__ void gm1_sum_initial_mass_v1(const double* layers,double* out,unsigned int nk){if(blockIdx.x==0&&threadIdx.x==0){double total=0.0;for(unsigned int k=0;k<nk;++k)total=total+layers[k];out[0]=total;}}
extern "C" __global__ void gm1_broadcast_base_profiles_v1(
 const float* pi0,const float* th0,float* pi0_3d,float* th0_3d,unsigned int ni,unsigned int nj,unsigned int nk){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x,n=(unsigned long long)ni*nj*nk,plane=(unsigned long long)ni*nj;
 for(unsigned long long z=first;z<n;z+=step){unsigned int k=(unsigned int)(z/plane);pi0_3d[z]=pi0[k];th0_3d[z]=th0[k];}}
"""
GM1_FORECAST_CUDA_SOURCE_BYTES = GM1_FORECAST_CUDA_SOURCE.encode("utf-8")
GM1_FORECAST_CUDA_SOURCE_SHA256 = (
    "22af7902616f44a0a90eeb3ab609555d7462ac12e20c5f7a93c54d6c839e4159"
)


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


@dataclass(frozen=True, slots=True)
class CompiledGm1ForecastKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_forecast() -> CompiledGm1ForecastKernels:
    digest = hashlib.sha256(GM1_FORECAST_CUDA_SOURCE_BYTES).hexdigest()
    if digest != GM1_FORECAST_CUDA_SOURCE_SHA256:
        raise RuntimeError("packaged GM1 forecast CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    options = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        options += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(
        GM1_FORECAST_CUDA_SOURCE, "gm1_forecast_v1_ptx.cu", method="ptx"
    ).compile(options + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(
        GM1_FORECAST_CUDA_SOURCE, "gm1_forecast_v1_cubin.cu", method="cubin"
    ).compile(options + (f"--gpu-architecture=sm_{arch}",))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode()
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", decoded)
        for name in GM1_FORECAST_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks GM1 forecast entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled GM1 forecast image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in GM1_FORECAST_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = props.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledGm1ForecastKernels(module, kernels, ptx, cubin, arch, identity)


@dataclass(slots=True)
class Gm1Solve1Carriers:
    momentum: dict[str, Any]
    theta: Any
    moisture: Any


@dataclass(slots=True)
class CudaGm1ForecastState:
    selection: Gm1ForecastSelection
    product: CudaInitializedProduct
    geopotential: CudaWsm6Geopotential
    prepared: CudaInitializedTerrainOpenState
    transition: InitializedProductWsm6Transition | CudaGm1KesslerLargeStepTransition
    moist_bridge: CudaMoistStageBridgeState
    scalar_rk: CudaScalarMoistRKState
    solve1: Gm1Solve1Carriers
    pi0_3d: Any
    th0_3d: Any
    compiled: CompiledGm1ForecastKernels
    momentum_compiled: CompiledMomentumRKKernels
    momentum_advection_compiled: CompiledMomentumAdvection5Kernels
    psolver_compiled: CompiledPsolver3TerrainOpenKernels
    terrain_compiled: CompiledTerrainOpenBoundaryKernels
    open_compiled: CompiledOpenBoundaryKernels
    lateral_compiled: CompiledLateralClosureKernels
    scalar_compiled: CompiledScalarMoistRKKernels
    scalar_advection_compiled: CompiledAdvection5Kernels
    apmasscon_compiled: CompiledApMassConKernels
    pdefq_compiled: CompiledWsm6Pdefq
    pdefq_workspace: Wsm6PdefqWorkspace
    pdefq_asq: Any
    closure: CudaGm1ClosureState
    mass1_f64: float
    stream: Any
    device_id: int
    ready_event: Any
    step_index: int = 0
    time_seconds: float = 0.0
    next_stage: int = 1
    solve1_ready: bool = False
    retained: list[Any] = field(default_factory=list)
    last_receipt: dict[str, Any] | None = None


def _forcing_number(receipt: Mapping[str, Any], name: str) -> float:
    value = receipt.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        refuse(
            code="GM1-HRRR-LBC-RECEIPT",
            gate="boundaries-closures",
            option=name,
            value=value,
            reason="the lateral forcing operator did not attest its exact clock/blend result",
            remedy="return the source-pinned GM1 HRRR LBC receipt from every RK-stage callback",
        )
    return float(value)


def _launch_gm1_lateral_forcing_stage(
    state: CudaGm1ForecastState,
    forcing: Gm1ForecastLateralForcing,
    nrk: int,
) -> dict[str, Any]:
    validate_gm1_lateral_forcing(forcing)
    source_artifact = getattr(
        getattr(getattr(state, "product", None), "export", None),
        "source_artifact_sha256",
        None,
    )
    source_offset = GM1_HRRR_INITIALIZATION_SOURCE_OFFSETS_S.get(source_artifact)
    if source_offset is None or source_offset != float(
        forcing.source_clock.initialization_source_offset_s
    ):
        refuse(
            code="GM1-HRRR-LBC-INITIALIZATION-CLOCK",
            gate="boundaries-closures",
            option="initialization_source",
            value={"artifact_sha256": source_artifact, "offset_s": source_offset},
            reason="the initialized real-product identity does not match the forcing source clock",
            remedy="bind f00 initialization to start=0 or sealed f01 initialization to start=3600",
        )
    stage_timestep = float(momentum_rk_stage_timestep_f32(state.selection.dt_s, nrk))
    davies = forcing.relaxation_mode == "davies-c2-stage-tendency"
    if davies:
        tau = float(forcing.relaxation_timescale_s)  # validated above
        if tau < float(state.selection.dt_s):
            refuse(
                code="GM1-HRRR-LBC-RELAXATION-TIMESCALE",
                gate="boundaries-closures",
                option="relaxation_timescale_s",
                value=tau,
                reason="tau shorter than one large step can over-relax the final RK stage",
                remedy="select tau >= dt so every CM1 dttmp/tau factor is in (0,1]",
            )
        numpy = importlib.import_module("numpy")
        expected_stage_factor = float(
            numpy.float32(numpy.float32(stage_timestep) / numpy.float32(tau))
        )
        x = 1.0 / float(forcing.relaxation_width_points)
        expected_inner_spatial = x * x * x * (x * (6.0 * x - 15.0) + 10.0)
        expected_inner = expected_inner_spatial * expected_stage_factor
    else:
        tau = None
        expected_stage_factor = 1.0
        expected_inner_spatial = 1.0 / float(
            forcing.relaxation_width_points * forcing.relaxation_width_points
        )
        expected_inner = expected_inner_spatial
    point = gm1_source_clock_point(
        forcing.source_clock, state.time_seconds + stage_timestep
    )
    raw = forcing.apply_stage(state, nrk, point)
    if not isinstance(raw, Mapping):
        refuse(
            code="GM1-HRRR-LBC-RECEIPT",
            gate="boundaries-closures",
            option="apply_stage_result",
            value=raw,
            reason="the lateral forcing callback did not return an auditable mapping",
            remedy="return the exact GM1 HRRR LBC stage receipt",
        )
    operator = dict(raw)
    target = _forcing_number(operator, "target_model_time_s")
    target_absolute = _forcing_number(operator, "target_absolute_source_time_s")
    alpha = _forcing_number(operator, "temporal_interpolation_fraction")
    width = operator.get("relaxation_width_points")
    edge = _forcing_number(operator, "edge_weight")
    inner = _forcing_number(operator, "inner_zone_weight")
    if (
        abs(target - point.relative_model_time_s) > 1.0e-9
        or abs(target_absolute - point.absolute_source_time_s) > 1.0e-9
        or abs(alpha - point.temporal_interpolation_fraction) > 2.0e-7
    ):
        refuse(
            code="GM1-HRRR-LBC-INTERPOLATION-CLOCK",
            gate="boundaries-closures",
            option="operator_receipt",
            value={
                "target_model_time_s": target,
                "target_absolute_source_time_s": target_absolute,
                "alpha": alpha,
            },
            reason="the boundary operator used a different clock from the source interval",
            remedy="compute alpha from the supplied absolute/relative source-clock point",
        )
    if davies:
        receipt_mode = operator.get("relaxation_mode")
        receipt_tau = _forcing_number(operator, "relaxation_timescale_s")
        receipt_stage_timestep = _forcing_number(operator, "rk_stage_timestep_s")
        receipt_stage_factor = _forcing_number(operator, "stage_relaxation_factor")
        receipt_inner_spatial = _forcing_number(operator, "inner_zone_spatial_weight")
        receipt_inner_effective = _forcing_number(
            operator, "inner_zone_effective_weight"
        )
        if (
            receipt_mode != forcing.relaxation_mode
            or receipt_tau != tau
            or receipt_stage_timestep != stage_timestep
            or receipt_stage_factor != expected_stage_factor
            or abs(receipt_inner_spatial - expected_inner_spatial) > 1.0e-15
            or abs(receipt_inner_effective - expected_inner) > 1.0e-15
            or inner != receipt_inner_effective
        ):
            refuse(
                code="GM1-HRRR-LBC-DAVIES-CONTRACT",
                gate="boundaries-closures",
                option="operator_receipt",
                value={
                    "mode": receipt_mode,
                    "tau_s": receipt_tau,
                    "stage_timestep_s": receipt_stage_timestep,
                    "stage_factor": receipt_stage_factor,
                    "inner_spatial": receipt_inner_spatial,
                    "inner_effective": receipt_inner_effective,
                },
                reason="the returned forcing is not the selected CM1-stage Davies C2 tendency",
                remedy="derive each stage factor from source dttmp/tau and taper exactly to zero",
            )
    if (
        type(width) is not int
        or width != forcing.relaxation_width_points
        or edge != 1.0
        or inner <= 0.0
        or abs(inner - expected_inner) > 1.0e-12
    ):
        refuse(
            code="GM1-HRRR-LBC-BLEND-CONTRACT",
            gate="boundaries-closures",
            option="operator_receipt",
            value={"width": width, "edge": edge, "inner": inner},
            reason="the returned edge blend differs from the selected continuous relaxation contract",
            remedy="retain the exact edge and source-receipted interior taper",
        )
    source_clock = {
        name: getattr(forcing.source_clock, name)
        for name in forcing.source_clock.__dataclass_fields__
    }
    result = {
        "semantic_id": forcing.semantic_id,
        "nrk": nrk,
        "source_clock": source_clock,
        "initialization_source_artifact_sha256": source_artifact,
        "clock_point": {
            name: getattr(point, name) for name in point.__dataclass_fields__
        },
        "rk_stage_timestep_s": stage_timestep,
        "rk_stage_timestep_authority": SOURCE_ANCHORS["rk_stage_timestep"],
        "cadence": "after each RK stage: 1,2,3",
        "blend": (
            "exact-source outer edge plus C2 Davies interior tendency; "
            "CM1 dttmp/tau; visualization-independent"
            if davies
            else "legacy direct quadratic edge relaxation; visualization-independent"
        ),
        "operator": operator,
    }
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


def _dependency_arches(state: CudaGm1ForecastState) -> set[str]:
    result = {
        state.compiled.target_arch,
        state.momentum_compiled.target_arch,
        state.momentum_advection_compiled.target_arch,
        state.psolver_compiled.target_arch,
        state.terrain_compiled.target_arch,
        state.open_compiled.target_arch,
        state.lateral_compiled.target_arch,
        state.scalar_compiled.target_arch,
        state.scalar_advection_compiled.target_arch,
        state.apmasscon_compiled.target_arch,
        state.pdefq_compiled.target_arch,
    } | state.closure.target_arches
    if isinstance(state.transition, CudaGm1KesslerLargeStepTransition):
        result |= state.transition.target_arches
    return result


def _initial_mass(
    product: CudaInitializedProduct,
    prepared: CudaInitializedTerrainOpenState,
    pi0_3d: Any,
    compiled: CompiledGm1ForecastKernels,
    apmasscon_compiled: CompiledApMassConKernels,
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any,
) -> float:
    cupy = importlib.import_module("cupy")
    # The sealed grid is horizontally uniform, so CM1's uh=ruh=1 and
    # vh=rvh=1.  Reuse the already device-resident physical map-factor views.
    ng = CM1_OPEN_NGXY
    ruh = prepared.arrays["uh"][ng : ng + ni]
    rvh = prepared.arrays["vh"][ng : ng + nj]
    layer_mass = cupy.empty((nk,), dtype=cupy.float64)
    layer_pressure = cupy.empty((nk,), dtype=cupy.float64)
    result = cupy.empty((1,), dtype=cupy.float64)
    apmasscon_compiled.kernels[0](
        (nk, 1, 1),
        (256, 1, 1),
        (
            product.field("rho"),
            ruh,
            rvh,
            prepared.metrics["rmh"],
            pi0_3d,
            product.field("pp3d"),
            layer_mass,
            layer_pressure,
            cupy.uint32(ni),
            cupy.uint32(nj),
            cupy.uint32(nk),
        ),
        stream=stream,
    )
    compiled.kernels[2](
        (1, 1, 1),
        (1, 1, 1),
        (layer_mass, result, cupy.uint32(nk)),
        stream=stream,
    )
    value = float(result.get(stream=stream)[0])
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("GM1 initial dry-mass ledger is not finite and positive")
    return value


def create_cuda_gm1_forecast_state(
    product: CudaInitializedProduct,
    geopotential: CudaWsm6Geopotential,
    *,
    selection: Gm1ForecastSelection = DEFAULT_GM1_SELECTION,
    allow_cropped_validation: bool = False,
) -> CudaGm1ForecastState:
    validate_gm1_terrain_open_backend(
        selection, allow_cropped_validation=allow_cropped_validation
    )
    validate_gm1_initialized_product_microphysics(selection)
    if tuple(product.export.crop_count_yx) != (selection.ny, selection.nx):
        raise ValueError("initialized product geometry differs from GM1 selection")
    cupy = importlib.import_module("cupy")
    if int(cupy.cuda.Device().id) != product.device_id:
        raise ValueError("initialized product is resident on another CUDA device")
    stream = product.stream
    prepared = prepare_cuda_initialized_product_terrain_open(
        product,
        allow_cropped_validation=allow_cropped_validation,
        stream=stream,
    )
    with stream:
        transition = initialize_product_wsm6_transition(
            product,
            geopotential,
            allow_cropped_validation=allow_cropped_validation,
        )
    compiled = compile_cuda_gm1_forecast()
    momentum_compiled = compile_cuda_momentum_rk()
    momentum_advection_compiled = compile_cuda_momentum_advection5()
    psolver_compiled = compile_cuda_psolver3_terrain_open()
    terrain_compiled = prepared.terrain_compiled
    open_compiled = prepared.open_compiled
    lateral_compiled = compile_cuda_lateral_closure()
    scalar_compiled = compile_cuda_scalar_moist_rk()
    scalar_advection_compiled = compile_cuda_advection5()
    apmasscon_compiled = compile_cuda_apmasscon()
    pdefq_compiled = compile_cuda_wsm6_pdefq()
    pdefq_workspace = create_cuda_wsm6_pdefq_workspace(
        ni=selection.nx, nj=selection.ny, nk=selection.nz
    )
    pdefq_asq = cupy.zeros((6,), dtype=cupy.float64)
    moist_compiled = compile_cuda_moist_stage_bridge()
    moist_bridge = create_cuda_moist_stage_bridge_state(
        ni=selection.nx,
        nj=selection.ny,
        nk=selection.nz,
        base3d=False,
        stream=stream,
        compiled=moist_compiled,
        open_compiled=open_compiled,
    )
    momentum = {
        name: cupy.zeros_like(prepared.arrays[name])
        for name in ("uten", "vten", "wten")
    }
    theta = cupy.zeros_like(product.field("th3d"))
    moisture = cupy.zeros_like(product.field("q3d"))
    pi0_3d = cupy.empty_like(product.field("rho"))
    th0_3d = cupy.empty_like(product.field("rho"))
    words = selection.nx * selection.ny * selection.nz
    with stream:
        compiled.kernels[3](
            ((words + GM1_FORECAST_THREADS - 1) // GM1_FORECAST_THREADS, 1, 1),
            (GM1_FORECAST_THREADS, 1, 1),
            (
                product.field("pi0"),
                product.field("th0"),
                pi0_3d,
                th0_3d,
                cupy.uint32(selection.nx),
                cupy.uint32(selection.ny),
                cupy.uint32(selection.nz),
            ),
            stream=stream,
        )
    scalar_rk = create_cuda_scalar_moist_rk_state(
        {
            "tha": product.field("tha"),
            "th3d": product.field("th3d"),
            "qa": product.field("qa"),
            "q3d": product.field("q3d"),
        },
        {"thten1": theta, "qten": moisture},
        product.field("th0"),
        ni=selection.nx,
        nj=selection.ny,
        nk=selection.nz,
        stream=stream,
    )
    closure = create_cuda_gm1_closure_state(
        product,
        prepared,
        ni=selection.nx,
        nj=selection.ny,
        nk=selection.nz,
        dx=selection.dx_m,
        dy=selection.dy_m,
        dz=selection.dz_m,
        dt=selection.dt_s,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    state = CudaGm1ForecastState(
        selection,
        product,
        geopotential,
        prepared,
        transition,
        moist_bridge,
        scalar_rk,
        Gm1Solve1Carriers(momentum, theta, moisture),
        pi0_3d,
        th0_3d,
        compiled,
        momentum_compiled,
        momentum_advection_compiled,
        psolver_compiled,
        terrain_compiled,
        open_compiled,
        lateral_compiled,
        scalar_compiled,
        scalar_advection_compiled,
        apmasscon_compiled,
        pdefq_compiled,
        pdefq_workspace,
        pdefq_asq,
        closure,
        0.0,
        stream,
        int(cupy.cuda.Device().id),
        event,
    )
    if len(_dependency_arches(state)) != 1:
        raise ValueError("GM1 dependency CUDA images target different architectures")
    state.mass1_f64 = _initial_mass(
        product,
        prepared,
        pi0_3d,
        compiled,
        apmasscon_compiled,
        ni=selection.nx,
        nj=selection.ny,
        nk=selection.nz,
        stream=stream,
    )
    return state


def begin_cuda_gm1_large_step(state: CudaGm1ForecastState) -> Gm1Solve1Carriers:
    if not isinstance(state, CudaGm1ForecastState):
        raise TypeError("state must be a CudaGm1ForecastState")
    if state.next_stage != 1 or state.solve1_ready:
        raise ValueError("GM1 large-step cadence is already active")
    state.ready_event.synchronize()
    state.retained.clear()
    cupy = importlib.import_module("cupy")
    with state.stream:
        for value in state.solve1.momentum.values():
            value.fill(cupy.float32(0.0))
        state.solve1.theta.fill(cupy.float32(0.0))
        state.solve1.moisture.fill(cupy.float32(0.0))
        state.prepared.arrays["ppten"].fill(cupy.float32(0.0))
    return state.solve1


def finalize_cuda_gm1_solve1(state: CudaGm1ForecastState) -> Any:
    if state.solve1_ready or state.next_stage != 1:
        raise ValueError("GM1 solve1 may be finalized once before RK stage 1")
    cupy = importlib.import_module("cupy")
    s = state.selection
    words = s.nx * s.ny * s.nz
    grid = ((words + GM1_FORECAST_THREADS - 1) // GM1_FORECAST_THREADS, 1, 1)
    threads = (GM1_FORECAST_THREADS, 1, 1)
    q = state.transition.carriers
    with state.stream:
        state.compiled.kernels[0](
            grid,
            threads,
            (
                state.product.field("th0"),
                state.product.field("pi0"),
                state.product.field("rho"),
                state.product.field("ppi"),
                state.product.field("tha"),
                state.product.field("qa"),
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


def _seed_momentum_and_stage_boundary(state: CudaGm1ForecastState, nrk: int) -> None:
    cupy = importlib.import_module("cupy")
    s, arrays, metrics = state.selection, state.prepared.arrays, state.prepared.metrics
    ng = CM1_OPEN_NGXY
    words = s.nz * (s.ny + 1) * (s.nx + 1)
    bc_words = max((s.ny + 2 * ng) * ng, (s.nx + 2 * ng) * ng, s.nx * s.ny)
    grid = ((words + CUDA_MOMENTUM_RK_THREADS - 1) // CUDA_MOMENTUM_RK_THREADS, 1, 1)
    bc_grid = (
        (bc_words + CUDA_MOMENTUM_RK_THREADS - 1) // CUDA_MOMENTUM_RK_THREADS,
        1,
        1,
    )
    threads = (CUDA_MOMENTUM_RK_THREADS, 1, 1)
    seed, bc2d = state.momentum_compiled.kernels
    seed(
        grid,
        threads,
        (
            state.solve1.momentum["uten"],
            state.solve1.momentum["vten"],
            state.solve1.momentum["wten"],
            arrays["uten"],
            arrays["vten"],
            arrays["wten"],
            cupy.uint32(s.nx),
            cupy.uint32(s.ny),
            cupy.uint32(s.nz),
            cupy.uint32(ng),
        ),
        stream=state.stream,
    )
    if nrk >= 2:
        state.terrain_compiled.kernels[2](
            bc_grid,
            threads,
            (
                arrays["u3d"],
                arrays["v3d"],
                metrics["gz"],
                metrics["dzdx"],
                metrics["dzdy"],
                arrays["w3d"],
                cupy.uint32(s.nx),
                cupy.uint32(s.ny),
                cupy.uint32(s.nz),
                cupy.uint32(ng),
            ),
            stream=state.stream,
        )
        for direction in range(4):
            bc2d(
                bc_grid,
                threads,
                (
                    arrays["w3d"],
                    cupy.uint32(s.nx),
                    cupy.uint32(s.ny),
                    cupy.uint32(ng),
                    cupy.uint32(direction),
                ),
                stream=state.stream,
            )


def launch_cuda_gm1_rk_stage(
    state: CudaGm1ForecastState,
    *,
    nrk: int,
    post_stage_hook: Callable[[CudaGm1ForecastState, int], object] | None = None,
) -> dict[str, Any]:
    if not state.solve1_ready or nrk != state.next_stage:
        raise ValueError(f"GM1 RK cadence requires prepared stage {state.next_stage}")
    if nrk not in (1, 2, 3):
        raise ValueError("nrk must be 1, 2, or 3")
    s, arrays, metrics = state.selection, state.prepared.arrays, state.prepared.metrics
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
        momentum = launch_cuda_terrain_open_momentum_advection5(
            arrays,
            metrics,
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            nrk=nrk,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            divx=state.moist_bridge.divx,
            stream=state.stream,
            compiled=state.momentum_advection_compiled,
        )
        launch_cuda_moist_buoyancy(
            state.moist_bridge, arrays, metrics, stream=state.stream
        )
        acoustic = launch_cuda_psolver3_terrain_open_stage(
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
            dz=s.dz_m,
            stream=state.stream,
            compiled=state.psolver_compiled,
            terrain_compiled=state.terrain_compiled,
            open_compiled=state.open_compiled,
            lateral_compiled=state.lateral_compiled,
        )
        post = launch_cuda_post_acoustic_flux_bridge(
            state.moist_bridge,
            arrays,
            metrics,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            stream=state.stream,
        )
        scalar = launch_cuda_scalar_moist_rk_stage(
            state.scalar_rk,
            {name: arrays[name] for name in ("rru", "rrv", "rrw")},
            metrics,
            state.moist_bridge.divx,
            state.moist_bridge.theta_divergence_term,
            nrk=nrk,
            dt=s.dt_s,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            stream=state.stream,
            compiled=state.scalar_compiled,
            advection_compiled=state.scalar_advection_compiled,
            open_compiled=state.open_compiled,
        )
        closure_tke = launch_cuda_gm1_closure_stage(
            state.closure,
            nrk=nrk,
            divx=state.moist_bridge.divx,
        )
        hook_result = None if post_stage_hook is None else post_stage_hook(state, nrk)
        event = importlib.import_module("cupy").cuda.Event(disable_timing=True)
        event.record(state.stream)
    state.retained.extend((momentum, acoustic, post, scalar, closure_tke, hook_result))
    begin_cuda_moist_stage(state.moist_bridge)
    state.next_stage = nrk + 1
    state.ready_event = event
    return {
        "nrk": nrk,
        "momentum": momentum.receipt,
        "acoustic": acoustic.receipt,
        "post_acoustic": post.receipt,
        "scalar": scalar.receipt,
        "closure_tke": closure_tke,
        "post_stage_hook": post_stage_hook is not None,
    }


def _sync_final_dynamics_to_product(state: CudaGm1ForecastState) -> None:
    cupy = importlib.import_module("cupy")
    s, a, p = state.selection, state.prepared.arrays, state.product
    ng = CM1_OPEN_NGXY
    cupy.copyto(
        p.field("u3d"), a["u3d"][1 : s.nz + 1, ng : ng + s.ny, ng : ng + s.nx + 1]
    )
    cupy.copyto(
        p.field("v3d"), a["v3d"][1 : s.nz + 1, ng : ng + s.ny + 1, ng : ng + s.nx]
    )
    cupy.copyto(p.field("w3d"), a["w3d"][1 : s.nz + 2, ng : ng + s.ny, ng : ng + s.nx])
    cupy.copyto(p.field("pp3d"), a["pp3d"][1 : s.nz + 1, 1 : s.ny + 1, 1 : s.nx + 1])


def _apmasscon_hook(state: CudaGm1ForecastState) -> Callable[[object], object]:
    def apply(_carriers: object) -> object:
        p, m, s = state.product, state.prepared.metrics, state.selection
        ng = CM1_OPEN_NGXY
        launch = launch_cuda_apmasscon(
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
        if state.selection.microphysics != GM1_WSM6_MICROPHYSICS_SEMANTIC_ID:
            state.retained.append(launch)
            return {"apmasscon": launch.receipt, "pdefq": None}
        pdefq = launch_cuda_wsm6_pdefq_pdscheme1(
            {
                "q3d": p.field("q3d"),
                "rho": p.field("rho"),
                "ruh": state.prepared.arrays["uh"][ng : ng + s.nx],
                "rvh": state.prepared.arrays["vh"][ng : ng + s.ny],
                "rmh": m["rmh"],
                "asq": state.pdefq_asq,
            },
            ni=s.nx,
            nj=s.ny,
            nk=s.nz,
            dx=s.dx_m,
            dy=s.dy_m,
            dz=s.dz_m,
            stream=state.stream,
            compiled=state.pdefq_compiled,
            workspace=state.pdefq_workspace,
        )
        state.retained.extend((launch, pdefq))
        return {"apmasscon": launch.receipt, "pdefq": pdefq.receipt}

    return apply


def _finish_solve3(state: CudaGm1ForecastState) -> None:
    cupy = importlib.import_module("cupy")
    s, a, p = state.selection, state.prepared.arrays, state.product
    pressure = (slice(1, s.nz + 1), slice(1, s.ny + 1), slice(1, s.nx + 1))
    ppx_tendency = a["ppx"][pressure]
    cupy.add(p.field("pp3d"), ppx_tendency, out=ppx_tendency)
    cupy.copyto(p.field("ppx"), ppx_tendency)
    cupy.copyto(a["pp3d"][pressure], p.field("pp3d"))
    cupy.copyto(a["ppi"][pressure], p.field("pp3d"))
    for name in ("ppx", "pp3d", "ppi"):
        state.open_compiled.kernels[3](
            ((a[name].size + 127) // 128, 1, 1),
            (128, 1, 1),
            (a[name], cupy.uint32(s.nx), cupy.uint32(s.ny), cupy.uint32(s.nz)),
            stream=state.stream,
        )
        state.open_compiled.kernels[4](
            ((a[name].size + 127) // 128, 1, 1),
            (128, 1, 1),
            (a[name], cupy.uint32(s.nx), cupy.uint32(s.ny), cupy.uint32(s.nz)),
            stream=state.stream,
        )
    for current, base in (("u3d", "ua"), ("v3d", "va"), ("w3d", "wa")):
        cupy.copyto(a[base], a[current])
        cupy.copyto(p.field(base), p.field(current))
    cupy.copyto(p.field("ppi"), p.field("pp3d"))
    cupy.copyto(p.field("tha"), p.field("th3d"))
    cupy.copyto(p.field("qa"), p.field("q3d"))
    words = s.nx * s.ny * (s.nz + 1)
    state.compiled.kernels[1](
        ((words + GM1_FORECAST_THREADS - 1) // GM1_FORECAST_THREADS, 1, 1),
        (GM1_FORECAST_THREADS, 1, 1),
        (
            p.field("rho"),
            p.field("c1"),
            p.field("c2"),
            p.field("rr"),
            p.field("rf"),
            cupy.uint32(s.nx),
            cupy.uint32(s.ny),
            cupy.uint32(s.nz),
        ),
        stream=state.stream,
    )


def apply_cuda_gm1_microphysics_transition(
    state: CudaGm1ForecastState,
    *,
    reflectivity_due: bool = False,
) -> Any:
    """Dispatch the selected semantic at the single native large-step seam."""

    semantic = state.selection.microphysics
    if semantic == GM1_WSM6_MICROPHYSICS_SEMANTIC_ID:
        if not isinstance(state.transition, InitializedProductWsm6Transition):
            refuse(
                code="GM1-MICROPHYSICS-STATE-MISMATCH",
                gate="physics-seam",
                option="microphysics",
                value=semantic,
                reason="the WSM6 semantic is not bound to a WSM6 transition",
                remedy="construct the forecast with the matching initialized-product transition",
            )
    elif semantic == GM1_KESSLER_LARGE_STEP_SEMANTIC_ID:
        if not isinstance(state.transition, CudaGm1KesslerLargeStepTransition):
            refuse(
                code="GM1-MICROPHYSICS-STATE-MISMATCH",
                gate="physics-seam",
                option="microphysics",
                value=semantic,
                reason="the Kessler semantic is not bound to a native ptype=1 transition",
                remedy="bind an independently initialized qv/qc/qr Kessler state",
            )
    else:
        refuse(
            code="GM1-MICROPHYSICS-DISPATCH",
            gate="physics-seam",
            option="microphysics",
            value=semantic,
            reason="no large-step transition is registered for this semantic",
            remedy="select an admitted semantic or implement and validate its source cadence",
        )
    return state.transition.apply(
        final_rk=True,
        refl_10cm_due=reflectivity_due,
        pre_component_hook=_apmasscon_hook(state),
    )


def finish_cuda_gm1_large_step(
    state: CudaGm1ForecastState,
    *,
    reflectivity_due: bool = False,
) -> dict[str, Any]:
    if state.next_stage != 4 or not state.solve1_ready:
        raise ValueError("GM1 large step requires completed RK stages 1, 2, and 3")
    cupy = importlib.import_module("cupy")
    with state.stream:
        _sync_final_dynamics_to_product(state)
        physics = apply_cuda_gm1_microphysics_transition(
            state, reflectivity_due=reflectivity_due
        )
        closure_commit = finish_cuda_gm1_closure_step(state.closure)
        _finish_solve3(state)
        closure_refresh = refresh_cuda_gm1_closure_for_next_step(state.closure)
        event = cupy.cuda.Event(disable_timing=True)
        event.record(state.stream)
    state.retained.append(physics)
    state.step_index += 1
    state.time_seconds = float(state.step_index * state.selection.dt_s)
    state.next_stage = 1
    state.solve1_ready = False
    state.ready_event = event
    begin_cuda_scalar_moist_rk_step(state.scalar_rk)
    receipt = {
        "schema": GM1_FORECAST_SCHEMA,
        "model": "GM1",
        "step": state.step_index,
        "time_seconds": state.time_seconds,
        "source_data": "real initialized atmospheric product; no synthetic atmospheric fields",
        "selection": {
            name: getattr(state.selection, name)
            for name in state.selection.__dataclass_fields__
        },
        "mass1_f64": state.mass1_f64,
        "microphysics_transition": state.transition.receipt_dict(),
        "pre_wsm6_pdefq": (
            next(
                item.receipt
                for item in reversed(state.retained)
                if isinstance(item, CudaWsm6PdefqLaunch)
            )
            if state.selection.microphysics == GM1_WSM6_MICROPHYSICS_SEMANTIC_ID
            else None
        ),
        "closure_commit": closure_commit,
        "closure_refresh": closure_refresh,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": AUTHORITY_SOURCE_SHA256,
            "anchors": SOURCE_ANCHORS,
        },
        "kernel": {
            "abi": GM1_FORECAST_ABI,
            "source_sha256": GM1_FORECAST_CUDA_SOURCE_SHA256,
            "target_arch": state.compiled.target_arch,
        },
        "ready_event_recorded": True,
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    state.last_receipt = receipt
    return receipt


def launch_cuda_gm1_large_step(
    state: CudaGm1ForecastState,
    *,
    solve1_hook: Callable[[CudaGm1ForecastState, Gm1Solve1Carriers], object]
    | None = None,
    post_stage_hook: Callable[[CudaGm1ForecastState, int], object] | None = None,
    lateral_forcing: Gm1ForecastLateralForcing | None = None,
    reflectivity_due: bool = False,
    allow_incomplete_validation: bool = False,
) -> dict[str, Any]:
    if not isinstance(state, CudaGm1ForecastState):
        raise TypeError("state must be a CudaGm1ForecastState")
    if lateral_forcing is not None:
        validate_gm1_lateral_forcing(lateral_forcing)
    forcing_receipts: list[dict[str, Any]] = []

    def apply_stage_hooks(current: CudaGm1ForecastState, nrk: int) -> object:
        forcing_result = None
        if lateral_forcing is not None:
            forcing_result = _launch_gm1_lateral_forcing_stage(
                current, lateral_forcing, nrk
            )
            forcing_receipts.append(forcing_result)
        user_result = None if post_stage_hook is None else post_stage_hook(current, nrk)
        return {"lateral_forcing": forcing_result, "user_hook": user_result}

    effective_stage_hook = (
        apply_stage_hooks
        if lateral_forcing is not None or post_stage_hook is not None
        else None
    )
    carriers = begin_cuda_gm1_large_step(state)
    builtin_closure = launch_cuda_gm1_closure_solve1(state.closure, carriers)
    solve1_result = None if solve1_hook is None else solve1_hook(state, carriers)
    state.retained.extend((builtin_closure, solve1_result))
    finalize_cuda_gm1_solve1(state)
    stages = [
        launch_cuda_gm1_rk_stage(state, nrk=nrk, post_stage_hook=effective_stage_hook)
        for nrk in (1, 2, 3)
    ]
    receipt = finish_cuda_gm1_large_step(state, reflectivity_due=reflectivity_due)
    receipt["stages"] = stages
    receipt["builtin_closure"] = builtin_closure
    receipt["solve1_hook"] = solve1_hook is not None
    receipt["post_stage_hook"] = post_stage_hook is not None
    if lateral_forcing is not None:
        if [item["nrk"] for item in forcing_receipts] != [1, 2, 3]:
            raise RuntimeError(
                "GM1 lateral forcing did not execute after all three RK stages"
            )
        receipt["lateral_forcing"] = {
            "semantic_id": lateral_forcing.semantic_id,
            "cadence": "after each RK stage: 1,2,3",
            "stage_receipts": forcing_receipts,
        }
    receipt["allow_incomplete_validation"] = allow_incomplete_validation
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    state.last_receipt = receipt
    return receipt


__all__ = [
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CompiledGm1ForecastKernels",
    "CudaGm1ForecastState",
    "CudaGm1ClosureState",
    "DEFAULT_GM1_SELECTION",
    "GM1_FORECAST_ABI",
    "GM1_FORECAST_CUDA_SOURCE",
    "GM1_FORECAST_CUDA_SOURCE_SHA256",
    "GM1_FORECAST_KERNEL_NAMES",
    "GM1_FORECAST_SCHEMA",
    "GM1_HRRR_F00_F01_SOURCE_CLOCK",
    "GM1_HRRR_F01_F02_SOURCE_CLOCK",
    "GM1_HRRR_DAVIES_LATERAL_FORCING_SEMANTIC_ID",
    "GM1_HRRR_INITIALIZATION_SOURCE_OFFSETS_S",
    "GM1_HRRR_LATERAL_FORCING_SEMANTIC_ID",
    "GM1_KESSLER_LARGE_STEP_SEMANTIC_ID",
    "GM1_REAL_100M_PRESET",
    "GM1_TERRAIN_OPEN_MICROPHYSICS",
    "GM1_WSM6_MICROPHYSICS_SEMANTIC_ID",
    "Gm1ForecastLateralForcing",
    "Gm1ForecastSelection",
    "Gm1ForecastSourceClock",
    "Gm1ForecastSourceClockPoint",
    "Gm1Solve1Carriers",
    "SOURCE_ANCHORS",
    "apply_cuda_gm1_microphysics_transition",
    "begin_cuda_gm1_large_step",
    "compile_cuda_gm1_forecast",
    "create_cuda_gm1_forecast_state",
    "finalize_cuda_gm1_solve1",
    "finish_cuda_gm1_large_step",
    "gm1_source_clock_point",
    "launch_cuda_gm1_large_step",
    "launch_cuda_gm1_rk_stage",
    "validate_gm1_selection",
    "validate_gm1_initialized_product_microphysics",
    "validate_gm1_lateral_forcing",
    "validate_gm1_source_clock",
    "validate_gm1_terrain_open_backend",
]
