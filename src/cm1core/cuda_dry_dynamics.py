"""Bounded multi-step CUDA dry-dynamics demonstrator for CM1 r21.1.

The slice composes the existing dry vertical operators with the scalar
fifth-order horizontal flux and CM1's three-stage low-storage RK update.  It
is sealed to a flat, constant-density, periodic specialization with a
prescribed divergence-free face flow.  It evolves potential-temperature and
vertical-velocity state on the device, but does not evolve horizontal wind or
pressure and is not a forecast or native-CM1 parity result.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any

from cm1core.cuda_dry_vertical import (
    CM1_DEFAULT_CP_F32,
    CM1_DEFAULT_GRAVITY_F32,
    CUDA_DRY_VERTICAL_THREADS,
    CompiledDryVerticalKernels,
    compile_cuda_dry_vertical,
    cpu_apply_buoyancy_wten_inplace,
    cpu_dry_thermo_buoyancy_inplace,
    cpu_psolver2_vertical_w_step_inplace,
    launch_cuda_dry_vertical_slice,
    psolver2_acoustic_schedule_f32,
    rk_stage_dt_f32,
    scalar_shape_zyx,
    w_shape_zyx,
)


CUDA_DRY_DYNAMICS_SCHEMA = "gpuwm-cm1.cuda-dry-dynamics/v1"
CUDA_DRY_DYNAMICS_ABI = "gpuwm-cm1.cuda-dry-dynamics-kernels/v1"
CUDA_DRY_DYNAMICS_THREADS = CUDA_DRY_VERTICAL_THREADS
CM1_SMEPS_F32 = 1.0e-30
CUDA_DRY_DYNAMICS_CLAIM_LIMIT = (
    "CM1 r21.1 flat constant-density periodic specialization: fifth-order "
    "horizontal scalar flux, three-stage low-storage RK update, and one "
    "previously validated vertical-w acoustic substep per stage under a "
    "prescribed divergence-free rotating face flow; no horizontal-wind or "
    "pressure evolution, complete acoustic cadence, native-CM1 parity, "
    "trajectory, storm, supercell, tornado, or forecast claim"
)

SOURCE_ANCHORS = {
    "periodic_boundary": "src/bc.F:12-170",
    "scalar_hadv_dispatch": "src/adv.F:128-201",
    "scalar_hadv_tendency": "src/adv.F:270-283;630-659",
    "hadv5_flux": "src/adv_routines.F:1335-1370",
    "flx5_formula": "src/adv_routines.F:6005-6019",
    "rk_schedule": "src/solve2.F:294-299",
    "theta_source_and_update": "src/solve2.F:1310-1426",
    "stage_tendency_reset": "src/solve2.F:460-471",
    "dry_vertical_composition": "src/solve2.F:434-449;895-921;src/sounde.F:544-564",
}

CUDA_DRY_DYNAMICS_KERNEL_NAMES = (
    "cm1_copy_f32_v1",
    "cm1_reset_wten_v1",
    "cm1_hadv5_rk_stage_v1",
)

CUDA_DRY_DYNAMICS_SOURCE = r"""// gpuwm-cm1.cuda-dry-dynamics-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default-real operators require float32");
static_assert(sizeof(unsigned long long) == 8, "CM1 offsets require uint64");

__device__ __forceinline__ float cm1_add_rn(const float a, const float b)
{
    return __fadd_rn(a, b);
}

__device__ __forceinline__ float cm1_sub_rn(const float a, const float b)
{
    return __fsub_rn(a, b);
}

__device__ __forceinline__ float cm1_mul_rn(const float a, const float b)
{
    return __fmul_rn(a, b);
}

__device__ __forceinline__ int cm1_wrap(const int value, const unsigned int count)
{
    int result = value % (int)count;
    return result < 0 ? result + (int)count : result;
}

__device__ __forceinline__ unsigned long long cm1_scalar_index(
    const int i,
    const int j,
    const unsigned int k,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int x_count,
    const unsigned int y_count)
{
    const unsigned int wrapped_i = (unsigned int)cm1_wrap(i, ni);
    const unsigned int wrapped_j = (unsigned int)cm1_wrap(j, nj);
    return ((unsigned long long)(k + 1u) * (unsigned long long)y_count
            + (unsigned long long)(wrapped_j + 3u))
            * (unsigned long long)x_count
            + (unsigned long long)(wrapped_i + 3u);
}

__device__ __forceinline__ float cm1_flx5(
    const float s1,
    const float s2,
    const float s3,
    const float s4,
    const float s5)
{
    const float c1 = 2.0f / 60.0f;
    const float c2 = -13.0f / 60.0f;
    const float c3 = 47.0f / 60.0f;
    const float c4 = 27.0f / 60.0f;
    const float c5 = -3.0f / 60.0f;
    float result = cm1_mul_rn(c1, s1);
    result = cm1_add_rn(result, cm1_mul_rn(c2, s2));
    result = cm1_add_rn(result, cm1_mul_rn(c3, s3));
    result = cm1_add_rn(result, cm1_mul_rn(c4, s4));
    result = cm1_add_rn(result, cm1_mul_rn(c5, s5));
    return result;
}

__device__ __forceinline__ float cm1_x_flux(
    const float* __restrict__ state,
    const float* __restrict__ u_face,
    const unsigned int face,
    const unsigned int j,
    const unsigned int k,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int x_count,
    const unsigned int y_count)
{
    const unsigned long long velocity_index =
        ((unsigned long long)k * (unsigned long long)nj + (unsigned long long)j)
        * (unsigned long long)(ni + 1u) + (unsigned long long)face;
    const float velocity = u_face[velocity_index];
    const int f = (int)face;
    float reconstructed;
    if (velocity >= 0.0f) {
        reconstructed = cm1_flx5(
            state[cm1_scalar_index(f - 3, (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f - 2, (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f - 1, (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f,     (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f + 1, (int)j, k, ni, nj, x_count, y_count)]);
    } else {
        reconstructed = cm1_flx5(
            state[cm1_scalar_index(f + 2, (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f + 1, (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f,     (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f - 1, (int)j, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index(f - 2, (int)j, k, ni, nj, x_count, y_count)]);
    }
    return cm1_mul_rn(velocity, reconstructed);
}

__device__ __forceinline__ float cm1_y_flux(
    const float* __restrict__ state,
    const float* __restrict__ v_face,
    const unsigned int i,
    const unsigned int face,
    const unsigned int k,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int x_count,
    const unsigned int y_count)
{
    const unsigned long long velocity_index =
        ((unsigned long long)k * (unsigned long long)(nj + 1u)
         + (unsigned long long)face) * (unsigned long long)ni
        + (unsigned long long)i;
    const float velocity = v_face[velocity_index];
    const int f = (int)face;
    float reconstructed;
    if (velocity >= 0.0f) {
        reconstructed = cm1_flx5(
            state[cm1_scalar_index((int)i, f - 3, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f - 2, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f - 1, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f,     k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f + 1, k, ni, nj, x_count, y_count)]);
    } else {
        reconstructed = cm1_flx5(
            state[cm1_scalar_index((int)i, f + 2, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f + 1, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f,     k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f - 1, k, ni, nj, x_count, y_count)],
            state[cm1_scalar_index((int)i, f - 2, k, ni, nj, x_count, y_count)]);
    }
    return cm1_mul_rn(velocity, reconstructed);
}

extern "C" __global__
void cm1_copy_f32_v1(
    const float* __restrict__ source,
    float* __restrict__ destination,
    const unsigned long long count)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;
    for (unsigned long long index = first; index < count; index += stride) {
        destination[index] = source[index];
    }
}

extern "C" __global__
void cm1_reset_wten_v1(
    float* __restrict__ wten,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int nk,
    const unsigned int x_count,
    const unsigned int y_count)
{
    const unsigned long long count =
        (unsigned long long)ni * (unsigned long long)nj
        * (unsigned long long)(nk - 1u);
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;
    for (unsigned long long linear = first; linear < count; linear += stride) {
        const unsigned int i = (unsigned int)(linear % ni) + 3u;
        const unsigned long long jk = linear / ni;
        const unsigned int j = (unsigned int)(jk % nj) + 3u;
        const unsigned int k = (unsigned int)(jk / nj) + 2u;
        const unsigned long long index =
            ((unsigned long long)k * (unsigned long long)y_count
             + (unsigned long long)j) * (unsigned long long)x_count
            + (unsigned long long)i;
        wten[index] = 0.0f;
    }
}

extern "C" __global__
void cm1_hadv5_rk_stage_v1(
    const float* __restrict__ theta_a,
    const float* __restrict__ theta_current,
    float* __restrict__ theta_next,
    const float* __restrict__ u_face,
    const float* __restrict__ v_face,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int nk,
    const unsigned int x_count,
    const unsigned int y_count,
    const float rdx,
    const float rdy,
    const float dttmp,
    const float smeps)
{
    const unsigned long long count =
        (unsigned long long)ni * (unsigned long long)nj * (unsigned long long)nk;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;
    for (unsigned long long linear = first; linear < count; linear += stride) {
        const unsigned int i = (unsigned int)(linear % ni);
        const unsigned long long jk = linear / ni;
        const unsigned int j = (unsigned int)(jk % nj);
        const unsigned int k = (unsigned int)(jk / nj);
        const unsigned long long index =
            cm1_scalar_index((int)i, (int)j, k, ni, nj, x_count, y_count);
        const float west = cm1_x_flux(
            theta_current, u_face, i, j, k, ni, nj, x_count, y_count);
        const float east = cm1_x_flux(
            theta_current, u_face, i + 1u, j, k, ni, nj, x_count, y_count);
        const float south = cm1_y_flux(
            theta_current, v_face, i, j, k, ni, nj, x_count, y_count);
        const float north = cm1_y_flux(
            theta_current, v_face, i, j + 1u, k, ni, nj, x_count, y_count);
        const float advx = cm1_mul_rn(-cm1_sub_rn(east, west), rdx);
        const float advy = cm1_mul_rn(-cm1_sub_rn(north, south), rdy);
        const float tendency = cm1_add_rn(advx, advy);
        const float increment = cm1_mul_rn(dttmp, tendency);
        float updated = cm1_add_rn(theta_a[index], increment);
        if (fabsf(updated) < smeps) {
            updated = 0.0f;
        }
        theta_next[index] = updated;
    }
}
"""

CUDA_DRY_DYNAMICS_SOURCE_BYTES = CUDA_DRY_DYNAMICS_SOURCE.encode("utf-8")
CUDA_DRY_DYNAMICS_SOURCE_SHA256 = (
    "f91639542759a37e40388023046bd1d21b17bd4c057f76f1fc6d0856655cd348"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_DRY_DYNAMICS_SOURCE_BYTES).hexdigest() == (
        CUDA_DRY_DYNAMICS_SOURCE_SHA256
    )


def u_face_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    scalar_shape_zyx(ni, nj, nk)
    return nk, nj, ni + 1


def v_face_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    scalar_shape_zyx(ni, nj, nk)
    return nk, nj + 1, ni


def _require_array(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    writable: bool,
) -> Any:
    np = importlib.import_module("numpy")
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, received {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _validate_flow(u_face: Any, v_face: Any) -> None:
    np = importlib.import_module("numpy")
    if not np.array_equal(
        u_face[..., 1:].view(np.uint32), u_face[..., :-1].view(np.uint32)
    ):
        raise ValueError("v1 requires u_face to be exactly constant along x")
    if not np.array_equal(
        v_face[:, 1:, :].view(np.uint32), v_face[:, :-1, :].view(np.uint32)
    ):
        raise ValueError("v1 requires v_face to be exactly constant along y")


def flx5_f32(s1: Any, s2: Any, s3: Any, s4: Any, s5: Any) -> Any:
    """Default-real transcription of ``adv_routines.F:6005-6019``."""

    np = importlib.import_module("numpy")
    constants = tuple(
        np.float32(np.float32(numerator) / np.float32(60.0))
        for numerator in (2.0, -13.0, 47.0, 27.0, -3.0)
    )
    values = tuple(np.float32(value) for value in (s1, s2, s3, s4, s5))
    result = np.float32(constants[0] * values[0])
    for coefficient, value in zip(constants[1:], values[1:], strict=True):
        result = np.float32(result + np.float32(coefficient * value))
    return result


def _x_flux_cpu(state: Any, velocity: Any, i_face: int, j: int, k: int) -> Any:
    np = importlib.import_module("numpy")
    ni = state.shape[2]
    f = i_face
    if bool(velocity >= np.float32(0.0)):
        indices = (f - 3, f - 2, f - 1, f, f + 1)
    else:
        indices = (f + 2, f + 1, f, f - 1, f - 2)
    reconstructed = flx5_f32(*(state[k, j, index % ni] for index in indices))
    return np.float32(velocity * reconstructed)


def _y_flux_cpu(state: Any, velocity: Any, i: int, j_face: int, k: int) -> Any:
    np = importlib.import_module("numpy")
    nj = state.shape[1]
    f = j_face
    if bool(velocity >= np.float32(0.0)):
        indices = (f - 3, f - 2, f - 1, f, f + 1)
    else:
        indices = (f + 2, f + 1, f, f - 1, f - 2)
    reconstructed = flx5_f32(*(state[k, index % nj, i] for index in indices))
    return np.float32(velocity * reconstructed)


def cpu_hadv5_rk_stage_inplace(
    theta_a: Any,
    theta_current: Any,
    theta_next: Any,
    u_face: Any,
    v_face: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    rdx: float,
    rdy: float,
    dttmp: float,
    smeps: float = CM1_SMEPS_F32,
) -> None:
    """Periodic constant-density specialization of the CM1 scalar update."""

    np = importlib.import_module("numpy")
    shape = scalar_shape_zyx(ni, nj, nk)
    theta_a = _require_array("theta_a", theta_a, shape, writable=False)
    theta_current = _require_array(
        "theta_current", theta_current, shape, writable=False
    )
    theta_next = _require_array("theta_next", theta_next, shape, writable=True)
    u_face = _require_array(
        "u_face", u_face, u_face_shape_zyx(ni, nj, nk), writable=False
    )
    v_face = _require_array(
        "v_face", v_face, v_face_shape_zyx(ni, nj, nk), writable=False
    )
    _validate_flow(u_face, v_face)
    physical = theta_current[1 : nk + 1, 3 : nj + 3, 3 : ni + 3]
    rdx32 = np.float32(rdx)
    rdy32 = np.float32(rdy)
    dttmp32 = np.float32(dttmp)
    smeps32 = np.float32(smeps)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                west = _x_flux_cpu(physical, u_face[k, j, i], i, j, k)
                east = _x_flux_cpu(physical, u_face[k, j, i + 1], i + 1, j, k)
                south = _y_flux_cpu(physical, v_face[k, j, i], i, j, k)
                north = _y_flux_cpu(physical, v_face[k, j + 1, i], i, j + 1, k)
                advx = np.float32(np.float32(-(np.float32(east - west))) * rdx32)
                advy = np.float32(np.float32(-(np.float32(north - south))) * rdy32)
                tendency = np.float32(advx + advy)
                increment = np.float32(dttmp32 * tendency)
                updated = np.float32(theta_a[k + 1, j + 3, i + 3] + increment)
                if bool(np.abs(updated) < smeps32):
                    updated = np.float32(0.0)
                theta_next[k + 1, j + 3, i + 3] = updated


@dataclass(frozen=True, slots=True)
class CpuDryDynamicsResult:
    theta_final: Any
    theta_scratch: Any
    cadence: tuple[tuple[int, int, float, int, float], ...]


def cpu_dry_dynamics_inplace(
    theta: Any,
    theta_scratch: Any,
    theta_a: Any,
    dum8: Any,
    thv: Any,
    th0: Any,
    rth0: Any,
    c1: Any,
    c2: Any,
    mf: Any,
    ppd: Any,
    wten: Any,
    w3d: Any,
    u_face: Any,
    v_face: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    steps: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    rdz: float,
    gravity: float = CM1_DEFAULT_GRAVITY_F32,
    cp: float = CM1_DEFAULT_CP_F32,
) -> CpuDryDynamicsResult:
    """Execute the bounded source-ordered CPU state transition."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("dry dynamics requires at least two large steps")
    np = importlib.import_module("numpy")
    scalar_shape = scalar_shape_zyx(ni, nj, nk)
    w_shape = w_shape_zyx(ni, nj, nk)
    theta = _require_array("theta", theta, scalar_shape, writable=True)
    theta_scratch = _require_array(
        "theta_scratch", theta_scratch, scalar_shape, writable=True
    )
    theta_a = _require_array("theta_a", theta_a, scalar_shape, writable=True)
    for name, value in (
        ("dum8", dum8),
        ("thv", thv),
        ("th0", th0),
        ("rth0", rth0),
        ("c1", c1),
        ("c2", c2),
        ("ppd", ppd),
    ):
        _require_array(name, value, scalar_shape, writable=name in {"dum8", "thv"})
    _require_array("mf", mf, w_shape, writable=False)
    _require_array("wten", wten, w_shape, writable=True)
    _require_array("w3d", w3d, w_shape, writable=True)
    u_face = _require_array(
        "u_face", u_face, u_face_shape_zyx(ni, nj, nk), writable=False
    )
    v_face = _require_array(
        "v_face", v_face, v_face_shape_zyx(ni, nj, nk), writable=False
    )
    _validate_flow(u_face, v_face)
    rdx = float(np.float32(np.float32(1.0) / np.float32(dx)))
    rdy = float(np.float32(np.float32(1.0) / np.float32(dy)))
    current = theta
    scratch = theta_scratch
    cadence: list[tuple[int, int, float, int, float]] = []
    w_live = (slice(2, nk + 1), slice(3, nj + 3), slice(3, ni + 3))
    for step in range(1, steps + 1):
        np.copyto(theta_a, current)
        for nrk in (1, 2, 3):
            np.copyto(scratch, current)
            wten[w_live] = np.float32(0.0)
            dttmp = rk_stage_dt_f32(dt, nrk)
            nloop, dts = psolver2_acoustic_schedule_f32(dt, nsound, nrk)
            cpu_dry_thermo_buoyancy_inplace(
                dum8,
                thv,
                current,
                rth0,
                th0,
                ni=ni,
                nj=nj,
                nk=nk,
                gravity=gravity,
            )
            cpu_apply_buoyancy_wten_inplace(c1, c2, dum8, wten, ni=ni, nj=nj, nk=nk)
            cpu_psolver2_vertical_w_step_inplace(
                mf,
                c1,
                c2,
                thv,
                ppd,
                wten,
                w3d,
                ni=ni,
                nj=nj,
                nk=nk,
                rdz=rdz,
                cp=cp,
                dts=dts,
            )
            cpu_hadv5_rk_stage_inplace(
                theta_a,
                current,
                scratch,
                u_face,
                v_face,
                ni=ni,
                nj=nj,
                nk=nk,
                rdx=rdx,
                rdy=rdy,
                dttmp=dttmp,
            )
            cadence.append((step, nrk, dttmp, nloop, dts))
            current, scratch = scratch, current
    return CpuDryDynamicsResult(
        theta_final=current,
        theta_scratch=scratch,
        cadence=tuple(cadence),
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
    if not re.fullmatch(r"[0-9]+", text):
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
    return (
        common + (f"--gpu-architecture=compute_{arch}",),
        common + (f"--gpu-architecture=sm_{arch}",),
    )


@dataclass(frozen=True, slots=True)
class CompiledDryDynamicsKernels:
    module_owner: Any
    kernels: tuple[Any, Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_dry_dynamics() -> CompiledDryDynamicsKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged dry-dynamics CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    compiler_arch = _canonical_arch(compiler._get_arch())
    if compiler_arch != arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc_parts = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc_parts[0]))
    ptx_program = compiler._NVRTCProgram(
        CUDA_DRY_DYNAMICS_SOURCE, "cm1_dry_dynamics_v1_ptx.cu", method="ptx"
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_DRY_DYNAMICS_SOURCE, "cm1_dry_dynamics_v1_cubin.cu", method="cubin"
    )
    cubin, _ = cubin_program.compile(cubin_options)
    if isinstance(ptx, str):
        ptx = ptx.encode("utf-8")
    elif not isinstance(ptx, bytes):
        ptx = bytes(ptx)
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    elif not isinstance(cubin, bytes):
        cubin = bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_DRY_DYNAMICS_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks one or more dry-dynamics entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled dry-dynamics image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_DRY_DYNAMICS_KERNEL_NAMES
    )
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    runtime_identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc_parts[0])}.{int(nvrtc_parts[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledDryDynamicsKernels(
        module_owner=module,
        kernels=kernels,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def _require_device_array(
    cupy: Any,
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> Any:
    if not isinstance(value, cupy.ndarray):
        raise TypeError(f"{name} must be a cupy.ndarray")
    if value.dtype != cupy.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, received {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C contiguous")
    return value


@dataclass(frozen=True, slots=True)
class CudaDryDynamicsLaunch:
    ready_event: Any
    theta_final: Any
    theta_scratch: Any
    compiled: CompiledDryDynamicsKernels
    vertical_compiled: CompiledDryVerticalKernels
    retained_vertical_events: tuple[Any, ...]
    receipt: dict[str, Any]


def launch_cuda_dry_dynamics(
    theta: Any,
    theta_scratch: Any,
    theta_a: Any,
    dum8: Any,
    thv: Any,
    th0: Any,
    rth0: Any,
    c1: Any,
    c2: Any,
    mf: Any,
    ppd: Any,
    wten: Any,
    w3d: Any,
    u_face: Any,
    v_face: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    steps: int,
    dt: float,
    nsound: int,
    dx: float,
    dy: float,
    rdz: float,
    gravity: float = CM1_DEFAULT_GRAVITY_F32,
    cp: float = CM1_DEFAULT_CP_F32,
    stream: Any | None = None,
    compiled: CompiledDryDynamicsKernels | None = None,
    vertical_compiled: CompiledDryVerticalKernels | None = None,
) -> CudaDryDynamicsLaunch:
    """Enqueue the complete bounded multi-step transition without synchronization."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("dry dynamics requires at least two large steps")
    cupy = importlib.import_module("cupy")
    scalar_shape = scalar_shape_zyx(ni, nj, nk)
    w_shape = w_shape_zyx(ni, nj, nk)
    arrays = {
        "theta": _require_device_array(cupy, "theta", theta, scalar_shape),
        "theta_scratch": _require_device_array(
            cupy, "theta_scratch", theta_scratch, scalar_shape
        ),
        "theta_a": _require_device_array(cupy, "theta_a", theta_a, scalar_shape),
        "dum8": _require_device_array(cupy, "dum8", dum8, scalar_shape),
        "thv": _require_device_array(cupy, "thv", thv, scalar_shape),
        "th0": _require_device_array(cupy, "th0", th0, scalar_shape),
        "rth0": _require_device_array(cupy, "rth0", rth0, scalar_shape),
        "c1": _require_device_array(cupy, "c1", c1, scalar_shape),
        "c2": _require_device_array(cupy, "c2", c2, scalar_shape),
        "mf": _require_device_array(cupy, "mf", mf, w_shape),
        "ppd": _require_device_array(cupy, "ppd", ppd, scalar_shape),
        "wten": _require_device_array(cupy, "wten", wten, w_shape),
        "w3d": _require_device_array(cupy, "w3d", w3d, w_shape),
        "u_face": _require_device_array(
            cupy, "u_face", u_face, u_face_shape_zyx(ni, nj, nk)
        ),
        "v_face": _require_device_array(
            cupy, "v_face", v_face, v_face_shape_zyx(ni, nj, nk)
        ),
    }
    if len({int(array.data.ptr) for array in arrays.values()}) != len(arrays):
        raise ValueError("dry-dynamics v1 arrays must not alias")
    if compiled is None:
        compiled = compile_cuda_dry_dynamics()
    if vertical_compiled is None:
        vertical_compiled = compile_cuda_dry_vertical()
    if compiled.target_arch != vertical_compiled.target_arch:
        raise ValueError(
            "horizontal/RK and vertical kernels target different architectures"
        )
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    scalar_words = theta.size
    scalar_count = ni * nj * nk
    w_count = ni * nj * (nk - 1)
    grid_storage = (
        scalar_words + CUDA_DRY_DYNAMICS_THREADS - 1
    ) // CUDA_DRY_DYNAMICS_THREADS
    grid_scalar = (
        scalar_count + CUDA_DRY_DYNAMICS_THREADS - 1
    ) // CUDA_DRY_DYNAMICS_THREADS
    grid_w = (w_count + CUDA_DRY_DYNAMICS_THREADS - 1) // CUDA_DRY_DYNAMICS_THREADS
    u32 = cupy.uint32
    u64 = cupy.uint64
    f32 = cupy.float32
    current = arrays["theta"]
    scratch = arrays["theta_scratch"]
    cadence: list[dict[str, Any]] = []
    retained_events: list[Any] = []
    rdx = f32(f32(1.0) / f32(dx))
    rdy = f32(f32(1.0) / f32(dy))
    for step in range(1, steps + 1):
        compiled.kernels[0](
            (grid_storage, 1, 1),
            (CUDA_DRY_DYNAMICS_THREADS, 1, 1),
            (current, arrays["theta_a"], u64(scalar_words)),
            shared_mem=0,
            stream=stream,
        )
        for nrk in (1, 2, 3):
            compiled.kernels[0](
                (grid_storage, 1, 1),
                (CUDA_DRY_DYNAMICS_THREADS, 1, 1),
                (current, scratch, u64(scalar_words)),
                shared_mem=0,
                stream=stream,
            )
            compiled.kernels[1](
                (grid_w, 1, 1),
                (CUDA_DRY_DYNAMICS_THREADS, 1, 1),
                (
                    arrays["wten"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    u32(w_shape[2]),
                    u32(w_shape[1]),
                ),
                shared_mem=0,
                stream=stream,
            )
            dttmp = rk_stage_dt_f32(dt, nrk)
            nloop, dts = psolver2_acoustic_schedule_f32(dt, nsound, nrk)
            vertical_launch = launch_cuda_dry_vertical_slice(
                arrays["dum8"],
                arrays["thv"],
                current,
                arrays["rth0"],
                arrays["th0"],
                arrays["c1"],
                arrays["c2"],
                arrays["mf"],
                arrays["ppd"],
                arrays["wten"],
                arrays["w3d"],
                ni=ni,
                nj=nj,
                nk=nk,
                gravity=gravity,
                rdz=rdz,
                cp=cp,
                dts=dts,
                stream=stream,
                compiled=vertical_compiled,
            )
            retained_events.append(vertical_launch.ready_event)
            compiled.kernels[2](
                (grid_scalar, 1, 1),
                (CUDA_DRY_DYNAMICS_THREADS, 1, 1),
                (
                    arrays["theta_a"],
                    current,
                    scratch,
                    arrays["u_face"],
                    arrays["v_face"],
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    u32(scalar_shape[2]),
                    u32(scalar_shape[1]),
                    rdx,
                    rdy,
                    f32(dttmp),
                    f32(CM1_SMEPS_F32),
                ),
                shared_mem=0,
                stream=stream,
            )
            cadence.append(
                {
                    "step": step,
                    "nrk": nrk,
                    "dttmp_f32": float(f32(dttmp)),
                    "scheduled_acoustic_nloop": nloop,
                    "executed_vertical_substeps": 1,
                    "vertical_dts_f32": float(f32(dts)),
                }
            )
            current, scratch = scratch, current
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_DRY_DYNAMICS_SCHEMA,
        "claim_limit": CUDA_DRY_DYNAMICS_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk},
        "steps": steps,
        "cadence": cadence,
        "comparison_contract": {"mode": "raw-bit", "max_ulp": 0},
        "kernel": {
            "abi": CUDA_DRY_DYNAMICS_ABI,
            "names": list(CUDA_DRY_DYNAMICS_KERNEL_NAMES),
            "source_sha256": CUDA_DRY_DYNAMICS_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "ptx_bytes": len(compiled.ptx),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "cubin_bytes": len(compiled.cubin),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "vertical_ptx_sha256": hashlib.sha256(vertical_compiled.ptx).hexdigest(),
        "launch": {
            "block_x": CUDA_DRY_DYNAMICS_THREADS,
            "grid_storage_x": grid_storage,
            "grid_scalar_x": grid_scalar,
            "grid_w_x": grid_w,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaDryDynamicsLaunch(
        ready_event=event,
        theta_final=current,
        theta_scratch=scratch,
        compiled=compiled,
        vertical_compiled=vertical_compiled,
        retained_vertical_events=tuple(retained_events),
        receipt=receipt,
    )


__all__ = [
    "CM1_SMEPS_F32",
    "CUDA_DRY_DYNAMICS_ABI",
    "CUDA_DRY_DYNAMICS_CLAIM_LIMIT",
    "CUDA_DRY_DYNAMICS_KERNEL_NAMES",
    "CUDA_DRY_DYNAMICS_SCHEMA",
    "CUDA_DRY_DYNAMICS_SOURCE",
    "CUDA_DRY_DYNAMICS_SOURCE_SHA256",
    "CompiledDryDynamicsKernels",
    "CpuDryDynamicsResult",
    "CudaDryDynamicsLaunch",
    "SOURCE_ANCHORS",
    "compile_cuda_dry_dynamics",
    "cpu_dry_dynamics_inplace",
    "cpu_hadv5_rk_stage_inplace",
    "flx5_f32",
    "launch_cuda_dry_dynamics",
    "u_face_shape_zyx",
    "v_face_shape_zyx",
]
