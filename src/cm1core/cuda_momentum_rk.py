"""CM1 r21.1 terrain/open momentum low-storage RK stage coordinator.

This module seals a deliberately bounded portion of ``solve2``: restore the
pre-RK held momentum tendencies over CM1's exact common source rectangle,
refresh the terrain bottom boundary at stages two and three, form terrain
mass fluxes, add admitted fifth-order momentum advection, and dispatch one
terrain/all-open ``psolver=3`` acoustic stage.  The CUDA path only enqueues
device work on one bound stream.

Buoyancy, Coriolis, SGS/diffusion, damping, forcing, immersed boundaries,
post-acoustic divergence/thermodynamics, and every later solve2 operator are
outside this coordinator.  Consequently this is not a complete RK stage,
trajectory, forecast, storm, tornado, or model-validation claim.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_momentum_advection5 import (
    CompiledMomentumAdvection5Kernels,
    compile_cuda_momentum_advection5,
    cpu_terrain_open_momentum_advection5_inplace,
    launch_cuda_terrain_open_momentum_advection5,
)
from cm1core.cuda_psolver3_terrain_open import (
    CompiledPsolver3TerrainOpenKernels,
    TerrainOpenSoundAblations,
    TerrainOpenSoundStageDiagnostics,
    compile_cuda_psolver3_terrain_open,
    cpu_psolver3_terrain_open_stage_inplace,
    launch_cuda_psolver3_terrain_open_stage,
    terrain_open_sound_shapes,
)
from cm1core.cuda_sound3_column import sound3_stage_schedule_f32
from cm1core.cuda_sound3_open_boundary import (
    CM1_OPEN_NGXY,
    CompiledOpenBoundaryKernels,
    compile_cuda_sound3_open_boundary,
)
from cm1core.cuda_terrain_open_boundary import (
    CompiledTerrainOpenBoundaryKernels,
    compile_cuda_terrain_open_boundary,
    cpu_bcwsfc_terrain_physical_inplace,
    terrain_static_shapes,
)
from cm1core.cuda_lateral_closure import (
    CompiledLateralClosureKernels,
    compile_cuda_lateral_closure,
)


CUDA_MOMENTUM_RK_SCHEMA = "gpuwm-cm1.cuda-momentum-rk/v1"
CUDA_MOMENTUM_RK_ABI = "gpuwm-cm1.cuda-momentum-rk-kernels/v1"
CUDA_MOMENTUM_RK_THREADS = 128
CM1_NRKMAX = 3

CM1_SOLVE2_F_SHA256 = "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
CM1_SOUND_F_SHA256 = "5fe055e22d106e91b5bdbe35ad7c72b2bbbcfb5b01733126703fdc7a5977da54"
CM1_BC_F_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
CM1_SOLVE1_F_SHA256 = "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076"
CM1_SOLVE3_F_SHA256 = "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"

SOURCE_ANCHORS = {
    "rk_schedule": "src/solve2.F:294-299",
    "held_tendency_seed": "src/solve2.F:459-471",
    "stage_entry_terrain_boundary": "src/solve2.F:475-488",
    "mass_flux_lifetime": "src/solve2.F:490-627",
    "momentum_advection": "src/solve2.F:883-892",
    "acoustic_dispatch": "src/solve2.F:1119-1136",
    "radiation_history_production": "src/solve2.F:1258-1271",
    "stage_family_boundaries": "src/solve2.F:1662-1706",
    "acoustic_schedule": "src/sound.F:81-113",
    "acoustic_a_reset": "src/sound.F:191-248",
    "open_history_consumption": "src/sound.F:253-302",
    "terrain_bcwsfc_per_small_step": "src/sound.F:596-600",
    "implicit_w_domain": "src/sound.F:659-730",
    "acoustic_pressure_fills": "src/sound.F:791-833",
    "open_bc2d_order": "src/bc.F:12-172",
    "held_tendency_production": "src/solve1.F:414-453",
    "completed_step_a_capture": "src/solve3.F:548-558",
}

CUDA_MOMENTUM_RK_CLAIM_LIMIT = (
    "CM1 r21.1 serial terrain/all-open bounded solve2 momentum slice: exact "
    "common-rectangle held-tendency restore, dt/3-dt/2-dt RK cadence, "
    "stage-2/3 terrain bottom/open-halo refresh, admitted terrain mass flux "
    "and fifth-order momentum advection, and one admitted psolver=3 acoustic "
    "stage including 2/3/6 small steps and irbc=4 histories. Coriolis, "
    "buoyancy, SGS/diffusion, damping, forcing, immersed boundaries, MPI, "
    "post-acoustic divergence/thermodynamics, other solve2 tendencies, a "
    "complete RK stage, evolved trajectory, forecast, storm, tornado, model "
    "validation, and performance claims are excluded"
)

HELD_NAMES = ("uten1", "vten1", "wten1")
TENDENCY_NAMES = ("uten", "vten", "wten")


def _positive_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _stage(nrk: object) -> int:
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be an integer in 1..3")
    return nrk


def _strict_acoustic_ablations(
    value: TerrainOpenSoundAblations | None,
) -> TerrainOpenSoundAblations:
    selected = TerrainOpenSoundAblations() if value is None else value
    if not isinstance(selected, TerrainOpenSoundAblations):
        raise TypeError("acoustic_ablations must be TerrainOpenSoundAblations")
    if selected != TerrainOpenSoundAblations():
        raise ValueError(
            "momentum RK requires every admitted terrain/open acoustic operator"
        )
    return selected


def momentum_rk_stage_timestep_f32(dt: float, nrk: int) -> Any:
    """Return source-default-REAL ``dt/(4-nrk)``."""

    dt = _positive("dt", dt)
    nrk = _stage(nrk)
    np = importlib.import_module("numpy")
    return np.float32(np.float32(dt) / np.float32(CM1_NRKMAX + 1 - nrk))


def momentum_rk_live_regions(ni: int, nj: int, nk: int) -> dict[str, Any]:
    """Describe exact one-based inclusive source and physical domains."""

    ni = _positive_int("ni", ni, 1)
    nj = _positive_int("nj", nj, 1)
    nk = _positive_int("nk", nk, 2)
    return {
        "source_seed_common": {"i": [1, ni + 1], "j": [1, nj + 1], "k": [1, nk]},
        "u_physical": {"i": [1, ni + 1], "j": [1, nj], "k": [1, nk]},
        "v_physical": {"i": [1, ni], "j": [1, nj + 1], "k": [1, nk]},
        "w_physical": {"i": [1, ni], "j": [1, nj], "k": [1, nk + 1]},
        "u_pressure_advection_interior": {"i": [2, ni], "j": [1, nj], "k": [1, nk]},
        "v_pressure_advection_interior": {"i": [1, ni], "j": [2, nj], "k": [1, nk]},
        "w_advection_implicit_interior": {"i": [1, ni], "j": [1, nj], "k": [2, nk]},
        "u_open_normal_faces": {
            "west_i": 1,
            "east_i": ni + 1,
            "j": [1, nj],
            "k": [1, nk],
        },
        "v_open_normal_faces": {
            "south_j": 1,
            "north_j": nj + 1,
            "i": [1, ni],
            "k": [1, nk],
        },
        "w_rigid_faces": {"bottom_k": 1, "top_k": nk + 1},
        "stage_entry_bc2d": {
            "field": "w3d",
            "k": 1,
            "order": ["west", "east", "south", "north"],
            "halo_width": CM1_OPEN_NGXY,
        },
    }


def momentum_rk_boundary_cadence(nrk: int, *, dt: float, nsound: int) -> dict[str, Any]:
    """Encode the source-ordered terrain/open cadence for one stage."""

    nrk = _stage(nrk)
    dt = _positive("dt", dt)
    nloop, dts = sound3_stage_schedule_f32(dt, nsound, nrk)
    np = importlib.import_module("numpy")
    dttmp = momentum_rk_stage_timestep_f32(dt, nrk)
    acoustic_total = np.float32(np.float32(dts) * np.float32(nloop))
    return {
        "nrk": nrk,
        "large_stage_divisor": CM1_NRKMAX + 1 - nrk,
        "dttmp_f32": float(dttmp),
        "held_tendency_seeds": 1,
        "stage_entry_a_reset_in_sound": nrk >= 2,
        "stage_entry_terrain_bcwsfc": 1 if nrk >= 2 else 0,
        "stage_entry_open_bc2d_bottom": 1 if nrk >= 2 else 0,
        "mass_flux_formations": 1,
        "momentum_advection_accumulations": 1,
        "acoustic": {
            "nloop": nloop,
            "dts_f32": float(np.float32(dts)),
            "integrated_span_f32": float(acoustic_total),
            "open_history_consumptions_per_edge": nloop,
            "terrain_bcwsfc_calls": nloop,
            "between_small_step_pressure_fills": nloop - 1,
        },
        "radiation4_history_productions": 1,
        "stage_momentum_family_fills": 1,
        "stage_scalar_pressure_family_fills": 1 if nrk < 3 else 0,
    }


def _host_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    shapes = terrain_open_sound_shapes(ni, nj, nk)
    return {
        "uten": shapes["uten"],
        "vten": shapes["vten"],
        "wten": shapes["wten"],
        "uten1": shapes["uten"],
        "vten1": shapes["vten"],
        "wten1": shapes["wten"],
    }


def _require_host_f32(
    np: Any, value: Any, name: str, shape: tuple[int, ...], *, writable: bool
) -> Any:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
        raise TypeError(f"{name} must be a float32 numpy.ndarray")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def cpu_seed_momentum_rk_tendencies_inplace(
    held: Mapping[str, Any],
    tendencies: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    """Copy held A tendencies over ``i=1:ni+1,j=1:nj+1,k=1:nk``."""

    ni = _positive_int("ni", ni, 1)
    nj = _positive_int("nj", nj, 1)
    nk = _positive_int("nk", nk, 1)
    np = importlib.import_module("numpy")
    shapes = _host_shapes(ni, nj, nk)
    h = {
        name: _require_host_f32(np, held[name], name, shapes[name], writable=False)
        for name in HELD_NAMES
    }
    t = {
        name: _require_host_f32(np, tendencies[name], name, shapes[name], writable=True)
        for name in TENDENCY_NAMES
    }
    pointers = [int(value.ctypes.data) for value in (*h.values(), *t.values())]
    if len(pointers) != len(set(pointers)):
        raise ValueError("held and current momentum tendencies must not alias")
    ng = CM1_OPEN_NGXY
    region = (slice(1, nk + 1), slice(ng, ng + nj + 1), slice(ng, ng + ni + 1))
    for held_name, tendency_name in zip(HELD_NAMES, TENDENCY_NAMES, strict=True):
        np.copyto(t[tendency_name][region], h[held_name][region])


def cpu_bc2d_open_w_surface_inplace(w3d: Any, *, ni: int, nj: int, nk: int) -> None:
    """Apply the serial all-open ``bc2d`` order to ``w3d(:,:,k=1)``."""

    ni = _positive_int("ni", ni, 1)
    nj = _positive_int("nj", nj, 1)
    nk = _positive_int("nk", nk, 1)
    np = importlib.import_module("numpy")
    shape = terrain_open_sound_shapes(ni, nj, nk)["w3d"]
    w3d = _require_host_f32(np, w3d, "w3d", shape, writable=True)
    ng = CM1_OPEN_NGXY
    plane = w3d[1]
    # bc.F performs these four branches in this exact order.  The later
    # south/north writes therefore define the final corner provenance.
    plane[:, :ng] = plane[:, ng : ng + 1]
    plane[:, ng + ni : ng + ni + ng] = plane[:, ng + ni - 1 : ng + ni]
    plane[:ng, :] = plane[ng : ng + 1, :]
    plane[ng + nj : ng + nj + ng, :] = plane[ng + nj - 1 : ng + nj, :]


@dataclass(frozen=True, slots=True)
class CpuMomentumRKStageDiagnostics:
    nrk: int
    dttmp_f32: float
    stage_entry_terrain_refreshes: int
    stage_entry_bc2d_refreshes: int
    acoustic: TerrainOpenSoundStageDiagnostics
    cadence: dict[str, Any]


def cpu_momentum_rk_stage_inplace(
    arrays: MutableMapping[str, Any],
    metrics: Mapping[str, Any],
    held: Mapping[str, Any],
    divx: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    nrk: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    dz: float,
    acoustic_ablations: TerrainOpenSoundAblations | None = None,
) -> CpuMomentumRKStageDiagnostics:
    """Execute the admitted CPU momentum slice in CM1 source order."""

    nrk = _stage(nrk)
    ni = _positive_int("ni", ni, 7)
    nj = _positive_int("nj", nj, 7)
    nk = _positive_int("nk", nk, 8)
    dt, dx, dy, dz = (
        _positive(name, value)
        for name, value in (("dt", dt), ("dx", dx), ("dy", dy), ("dz", dz))
    )
    cadence = momentum_rk_boundary_cadence(nrk, dt=dt, nsound=nsound)
    acoustic_ablations = _strict_acoustic_ablations(acoustic_ablations)
    cpu_seed_momentum_rk_tendencies_inplace(held, arrays, ni=ni, nj=nj, nk=nk)
    refresh = int(nrk >= 2)
    if refresh:
        cpu_bcwsfc_terrain_physical_inplace(arrays, metrics, ni=ni, nj=nj, nk=nk)
        cpu_bc2d_open_w_surface_inplace(arrays["w3d"], ni=ni, nj=nj, nk=nk)
    cpu_terrain_open_momentum_advection5_inplace(
        arrays,
        metrics,
        divx,
        ni=ni,
        nj=nj,
        nk=nk,
        nrk=nrk,
        dx=dx,
        dy=dy,
        dz=dz,
    )
    acoustic = cpu_psolver3_terrain_open_stage_inplace(
        arrays,
        metrics,
        ni=ni,
        nj=nj,
        nk=nk,
        nrk=nrk,
        dt=dt,
        nsound=nsound,
        dx=dx,
        dy=dy,
        dz=dz,
        ablations=acoustic_ablations,
    )
    return CpuMomentumRKStageDiagnostics(
        nrk,
        float(momentum_rk_stage_timestep_f32(dt, nrk)),
        refresh,
        refresh,
        acoustic,
        cadence,
    )


@dataclass(frozen=True, slots=True)
class CpuMomentumRKResult:
    cadence: tuple[CpuMomentumRKStageDiagnostics, ...]


def cpu_momentum_rk_three_stage_inplace(
    arrays: MutableMapping[str, Any],
    metrics: Mapping[str, Any],
    held: Mapping[str, Any],
    divx: Any,
    **kwargs: Any,
) -> CpuMomentumRKResult:
    """Execute three bounded stages; omitted solve2 terms remain omitted."""

    return CpuMomentumRKResult(
        tuple(
            cpu_momentum_rk_stage_inplace(
                arrays,
                metrics,
                held,
                divx,
                nrk=nrk,
                **kwargs,
            )
            for nrk in (1, 2, 3)
        )
    )


CUDA_MOMENTUM_RK_KERNEL_NAMES = (
    "cm1_momentum_rk_seed_held_v1",
    "cm1_momentum_rk_bc2d_open_v1",
)

CUDA_MOMENTUM_RK_SOURCE = r"""// gpuwm-cm1.cuda-momentum-rk-kernels/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void cm1_momentum_rk_seed_held_v1(
 const float* u1,const float* v1,const float* w1,float* ut,float* vt,float* wt,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned long long n=(unsigned long long)nk*(nj+1u)*(ni+1u);
 const unsigned int nxu=ni+1u+2u*ng,nyu=nj+2u*ng;
 const unsigned int nxv=ni+2u*ng,nyv=nj+1u+2u*ng;
 const unsigned int nxw=ni+2u*ng,nyw=nj+2u*ng;
 for(unsigned long long q=first;q<n;q+=step){
  const unsigned int i=(unsigned int)(q%(ni+1u));
  const unsigned long long z=q/(ni+1u);
  const unsigned int j=(unsigned int)(z%(nj+1u)),k=(unsigned int)(z/(nj+1u))+1u;
  const unsigned int x=i+ng,y=j+ng;
  const unsigned long long pu=i3(k,y,x,nyu,nxu),pv=i3(k,y,x,nyv,nxv),pw=i3(k,y,x,nyw,nxw);
  ut[pu]=u1[pu];vt[pv]=v1[pv];wt[pw]=w1[pw];
 }
}
extern "C" __global__ void cm1_momentum_rk_bc2d_open_v1(
 float* w,unsigned int ni,unsigned int nj,unsigned int ng,unsigned int direction){
 const unsigned long long first=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long step=(unsigned long long)blockDim.x*gridDim.x;
 const unsigned int nx=ni+2u*ng,ny=nj+2u*ng;
 const unsigned long long n=(direction<2u)?(unsigned long long)ny*ng:(unsigned long long)nx*ng;
 for(unsigned long long q=first;q<n;q+=step){
  unsigned int x,y,sx,sy;
  if(direction<2u){
   y=(unsigned int)(q/ng);const unsigned int h=(unsigned int)(q%ng);sy=y;
   if(direction==0u){x=h;sx=ng;}else{x=ng+ni+h;sx=ng+ni-1u;}
  }else{
   x=(unsigned int)(q%nx);const unsigned int h=(unsigned int)(q/nx);sx=x;
   if(direction==2u){y=h;sy=ng;}else{y=ng+nj+h;sy=ng+nj-1u;}
  }
  w[i3(1u,y,x,ny,nx)]=w[i3(1u,sy,sx,ny,nx)];
 }
}
"""

CUDA_MOMENTUM_RK_SOURCE_BYTES = CUDA_MOMENTUM_RK_SOURCE.encode("utf-8")
CUDA_MOMENTUM_RK_SOURCE_SHA256 = (
    "130b6c79fae3e4871e26b165def1d8112a1898ada30d9d5c7e7078b61c7d52c8"
)


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
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
class CompiledMomentumRKKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_momentum_rk() -> CompiledMomentumRKKernels:
    """Compile and load the pinned seed/bottom-halo kernels."""

    if (
        hashlib.sha256(CUDA_MOMENTUM_RK_SOURCE_BYTES).hexdigest()
        != CUDA_MOMENTUM_RK_SOURCE_SHA256
    ):
        raise RuntimeError("packaged momentum-RK CUDA source differs from its pin")
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
        CUDA_MOMENTUM_RK_SOURCE,
        "cm1_momentum_rk_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_MOMENTUM_RK_SOURCE,
        "cm1_momentum_rk_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_MOMENTUM_RK_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks momentum-RK entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled momentum-RK image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_MOMENTUM_RK_KERNEL_NAMES)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledMomentumRKKernels(
        module,
        kernels,
        ptx,
        cubin,
        arch,
        ptx_options,
        cubin_options,
        identity,
    )


def _require_cuda_f32(
    cupy: Any, value: Any, name: str, shape: tuple[int, ...], *, writable: bool
) -> Any:
    if not isinstance(value, cupy.ndarray) or value.dtype != cupy.dtype("float32"):
        raise TypeError(f"{name} must be a float32 cupy.ndarray")
    if value.shape != shape or not value.flags.c_contiguous:
        raise ValueError(f"{name} must have C-order shape {shape}")
    if writable and getattr(value.flags, "writeable", True) is False:
        raise ValueError(f"{name} must be writable")
    return value


@dataclass(slots=True)
class CudaMomentumRKState:
    ni: int
    nj: int
    nk: int
    arrays: dict[str, Any]
    metrics: dict[str, Any]
    held: dict[str, Any]
    divx: Any
    ready_event: Any
    stream: Any
    device_id: int
    next_nrk: int
    completed: bool


def create_cuda_momentum_rk_state(
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    held: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    stream: Any | None = None,
) -> CudaMomentumRKState:
    """Bind complete acoustic arrays, static metrics, and held tendencies."""

    ni = _positive_int("ni", ni, 7)
    nj = _positive_int("nj", nj, 7)
    nk = _positive_int("nk", nk, 8)
    cupy = importlib.import_module("cupy")
    array_shapes = terrain_open_sound_shapes(ni, nj, nk)
    missing = [name for name in array_shapes if name not in arrays]
    if missing:
        raise ValueError(f"missing momentum-RK arrays: {', '.join(missing)}")
    bound_arrays = {
        name: _require_cuda_f32(cupy, arrays[name], name, shape, writable=True)
        for name, shape in array_shapes.items()
    }
    metric_shapes = terrain_static_shapes(ni=ni, nj=nj, nk=nk)
    missing = [name for name in metric_shapes if name not in metrics]
    if missing:
        raise ValueError(f"missing momentum-RK metrics: {', '.join(missing)}")
    bound_metrics = {
        name: _require_cuda_f32(cupy, metrics[name], name, shape, writable=False)
        for name, shape in metric_shapes.items()
    }
    held_shapes = _host_shapes(ni, nj, nk)
    bound_held = {
        name: _require_cuda_f32(
            cupy, held[name], name, held_shapes[name], writable=False
        )
        for name in HELD_NAMES
    }
    divx = cupy.empty((nk, nj, ni), dtype=cupy.float32)
    values = (
        *bound_arrays.values(),
        *bound_metrics.values(),
        *bound_held.values(),
        divx,
    )
    pointers = [int(value.data.ptr) for value in values]
    if len(pointers) != len(set(pointers)):
        raise ValueError(
            "momentum-RK arrays, metrics, held carriers, and scratch must not alias"
        )
    stream = stream or cupy.cuda.get_current_stream()
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    return CudaMomentumRKState(
        ni,
        nj,
        nk,
        bound_arrays,
        bound_metrics,
        bound_held,
        divx,
        event,
        stream,
        int(cupy.cuda.Device().id),
        1,
        False,
    )


def begin_cuda_momentum_rk_step(state: CudaMomentumRKState) -> None:
    """Reset cadence after the caller refreshes A and held carriers."""

    if not isinstance(state, CudaMomentumRKState):
        raise TypeError("state must be a CudaMomentumRKState")
    if not state.completed or state.next_nrk != 4:
        raise ValueError("cannot begin a new large step before stage 3 completes")
    state.next_nrk = 1
    state.completed = False


@dataclass(frozen=True, slots=True)
class CudaMomentumRKStageLaunch:
    state: CudaMomentumRKState
    ready_event: Any
    compiled: CompiledMomentumRKKernels
    momentum_compiled: CompiledMomentumAdvection5Kernels
    psolver_compiled: CompiledPsolver3TerrainOpenKernels
    terrain_compiled: CompiledTerrainOpenBoundaryKernels
    open_compiled: CompiledOpenBoundaryKernels
    lateral_compiled: CompiledLateralClosureKernels
    advection_launch: Any
    acoustic_launch: Any
    receipt: dict[str, Any]


def launch_cuda_momentum_rk_stage(
    state: CudaMomentumRKState,
    *,
    nrk: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    dz: float,
    acoustic_ablations: TerrainOpenSoundAblations | None = None,
    stream: Any | None = None,
    compiled: CompiledMomentumRKKernels | None = None,
    momentum_compiled: CompiledMomentumAdvection5Kernels | None = None,
    psolver_compiled: CompiledPsolver3TerrainOpenKernels | None = None,
    terrain_compiled: CompiledTerrainOpenBoundaryKernels | None = None,
    open_compiled: CompiledOpenBoundaryKernels | None = None,
    lateral_compiled: CompiledLateralClosureKernels | None = None,
) -> CudaMomentumRKStageLaunch:
    """Enqueue one bounded source-ordered momentum stage on the bound stream."""

    if not isinstance(state, CudaMomentumRKState):
        raise TypeError("state must be a CudaMomentumRKState")
    nrk = _stage(nrk)
    if nrk != state.next_nrk or state.completed:
        raise ValueError(
            f"momentum RK cadence requires stage {state.next_nrk}, received {nrk}"
        )
    dt, dx, dy, dz = (
        _positive(name, value)
        for name, value in (("dt", dt), ("dx", dx), ("dy", dy), ("dz", dz))
    )
    # Validate the complete admitted selection and cadence before any enqueue.
    cadence = momentum_rk_boundary_cadence(nrk, dt=dt, nsound=nsound)
    acoustic_ablations = _strict_acoustic_ablations(acoustic_ablations)
    cupy = importlib.import_module("cupy")
    if int(cupy.cuda.Device().id) != state.device_id:
        raise ValueError("momentum RK state is bound to another CUDA device")
    stream = stream or state.stream
    if int(stream.ptr) != int(state.stream.ptr):
        raise ValueError("momentum RK state must remain on its bound CUDA stream")
    compiled = compiled or compile_cuda_momentum_rk()
    momentum_compiled = momentum_compiled or compile_cuda_momentum_advection5()
    psolver_compiled = psolver_compiled or compile_cuda_psolver3_terrain_open()
    terrain_compiled = terrain_compiled or compile_cuda_terrain_open_boundary()
    open_compiled = open_compiled or compile_cuda_sound3_open_boundary()
    lateral_compiled = lateral_compiled or compile_cuda_lateral_closure()
    arches = {
        compiled.target_arch,
        momentum_compiled.target_arch,
        psolver_compiled.target_arch,
        terrain_compiled.target_arch,
        open_compiled.target_arch,
        lateral_compiled.target_arch,
    }
    if len(arches) != 1:
        raise ValueError(
            "momentum-RK dependency CUDA images target different architectures"
        )
    seed, bc2d = compiled.kernels
    ng = CM1_OPEN_NGXY
    seed_words = state.nk * (state.nj + 1) * (state.ni + 1)
    bc_words = max(
        (state.nj + 2 * ng) * ng,
        (state.ni + 2 * ng) * ng,
        state.ni * state.nj,
    )
    threads = (CUDA_MOMENTUM_RK_THREADS, 1, 1)
    seed_grid = (
        (seed_words + CUDA_MOMENTUM_RK_THREADS - 1) // CUDA_MOMENTUM_RK_THREADS,
        1,
        1,
    )
    bc_grid = (
        (bc_words + CUDA_MOMENTUM_RK_THREADS - 1) // CUDA_MOMENTUM_RK_THREADS,
        1,
        1,
    )
    u32 = cupy.uint32
    with stream:
        stream.wait_event(state.ready_event)
        seed(
            seed_grid,
            threads,
            (
                state.held["uten1"],
                state.held["vten1"],
                state.held["wten1"],
                state.arrays["uten"],
                state.arrays["vten"],
                state.arrays["wten"],
                u32(state.ni),
                u32(state.nj),
                u32(state.nk),
                u32(ng),
            ),
            stream=stream,
        )
        if nrk >= 2:
            terrain_compiled.kernels[2](
                bc_grid,
                threads,
                (
                    state.arrays["u3d"],
                    state.arrays["v3d"],
                    state.metrics["gz"],
                    state.metrics["dzdx"],
                    state.metrics["dzdy"],
                    state.arrays["w3d"],
                    u32(state.ni),
                    u32(state.nj),
                    u32(state.nk),
                    u32(ng),
                ),
                stream=stream,
            )
            for direction in range(4):
                bc2d(
                    bc_grid,
                    threads,
                    (
                        state.arrays["w3d"],
                        u32(state.ni),
                        u32(state.nj),
                        u32(ng),
                        u32(direction),
                    ),
                    stream=stream,
                )
        advection_launch = launch_cuda_terrain_open_momentum_advection5(
            state.arrays,
            state.metrics,
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            nrk=nrk,
            dx=dx,
            dy=dy,
            dz=dz,
            divx=state.divx,
            stream=stream,
            compiled=momentum_compiled,
        )
        acoustic_launch = launch_cuda_psolver3_terrain_open_stage(
            state.arrays,
            state.metrics,
            ni=state.ni,
            nj=state.nj,
            nk=state.nk,
            nrk=nrk,
            dt=dt,
            nsound=nsound,
            dx=dx,
            dy=dy,
            dz=dz,
            ablations=acoustic_ablations,
            stream=stream,
            compiled=psolver_compiled,
            terrain_compiled=terrain_compiled,
            open_compiled=open_compiled,
            lateral_compiled=lateral_compiled,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    state.ready_event = event
    state.next_nrk = nrk + 1
    state.completed = nrk == CM1_NRKMAX
    kernel_hashes = {
        "momentum_rk_cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
        "momentum_advection5_cubin_sha256": hashlib.sha256(
            momentum_compiled.cubin
        ).hexdigest(),
        "psolver3_terrain_open_cubin_sha256": hashlib.sha256(
            psolver_compiled.cubin
        ).hexdigest(),
        "terrain_open_boundary_cubin_sha256": hashlib.sha256(
            terrain_compiled.cubin
        ).hexdigest(),
        "sound3_open_boundary_cubin_sha256": hashlib.sha256(
            open_compiled.cubin
        ).hexdigest(),
        "lateral_closure_cubin_sha256": hashlib.sha256(
            lateral_compiled.cubin
        ).hexdigest(),
    }
    receipt = {
        "schema": CUDA_MOMENTUM_RK_SCHEMA,
        "claim_limit": CUDA_MOMENTUM_RK_CLAIM_LIMIT,
        "domain": {"ni": state.ni, "nj": state.nj, "nk": state.nk, "ngxy": ng},
        "stage": {**cadence, "dt_f32": float(cupy.float32(dt))},
        "regions": momentum_rk_live_regions(state.ni, state.nj, state.nk),
        "source_order": [
            "held_tendency_seed",
            *(["stage_entry_bcwsfc", "stage_entry_bc2d_w"] if nrk >= 2 else []),
            "terrain_mass_flux_and_divx",
            "momentum_advection_accumulation",
            "terrain_open_psolver3_acoustics",
            "radiation4_history_production",
            "stage_family_boundaries",
        ],
        "source_anchors": dict(SOURCE_ANCHORS),
        "authority_sha256": {
            "src/solve1.F": CM1_SOLVE1_F_SHA256,
            "src/solve2.F": CM1_SOLVE2_F_SHA256,
            "src/solve3.F": CM1_SOLVE3_F_SHA256,
            "src/sound.F": CM1_SOUND_F_SHA256,
            "src/bc.F": CM1_BC_F_SHA256,
        },
        "kernel": {
            "abi": CUDA_MOMENTUM_RK_ABI,
            "source_sha256": CUDA_MOMENTUM_RK_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "dependency_kernels": kernel_hashes,
        "runtime": dict(compiled.runtime_identity),
        "comparison_contract": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "null_or_missing_is_pass": False,
        },
    }
    return CudaMomentumRKStageLaunch(
        state,
        event,
        compiled,
        momentum_compiled,
        psolver_compiled,
        terrain_compiled,
        open_compiled,
        lateral_compiled,
        advection_launch,
        acoustic_launch,
        receipt,
    )


__all__ = [
    "CM1_BC_F_SHA256",
    "CM1_NRKMAX",
    "CM1_SOLVE1_F_SHA256",
    "CM1_SOLVE2_F_SHA256",
    "CM1_SOLVE3_F_SHA256",
    "CM1_SOUND_F_SHA256",
    "CUDA_MOMENTUM_RK_ABI",
    "CUDA_MOMENTUM_RK_CLAIM_LIMIT",
    "CUDA_MOMENTUM_RK_KERNEL_NAMES",
    "CUDA_MOMENTUM_RK_SCHEMA",
    "CUDA_MOMENTUM_RK_SOURCE",
    "CUDA_MOMENTUM_RK_SOURCE_BYTES",
    "CUDA_MOMENTUM_RK_SOURCE_SHA256",
    "CompiledMomentumRKKernels",
    "CpuMomentumRKResult",
    "CpuMomentumRKStageDiagnostics",
    "CudaMomentumRKStageLaunch",
    "CudaMomentumRKState",
    "SOURCE_ANCHORS",
    "begin_cuda_momentum_rk_step",
    "compile_cuda_momentum_rk",
    "cpu_bc2d_open_w_surface_inplace",
    "cpu_momentum_rk_stage_inplace",
    "cpu_momentum_rk_three_stage_inplace",
    "cpu_seed_momentum_rk_tendencies_inplace",
    "create_cuda_momentum_rk_state",
    "launch_cuda_momentum_rk_stage",
    "momentum_rk_boundary_cadence",
    "momentum_rk_live_regions",
    "momentum_rk_stage_timestep_f32",
]
