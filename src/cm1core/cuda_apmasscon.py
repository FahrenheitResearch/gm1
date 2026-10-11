"""CM1 r21.1 final-stage dry-air mass correction on one CUDA device.

This module implements the Cartesian, moist, ``psolver=3`` product branch of
``apmasscon``.  The reference path preserves CM1's default-REAL expression
order and binary64 layer ledgers.  The production path keeps the reduction,
uniform Exner correction, and equation-of-state rebuild on the selected CUDA
stream.  It does not implement MPI or the axisymmetric volume factor.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from cm1core.errors import refuse


CUDA_APMASSCON_SCHEMA = "gpuwm-cm1.cuda-apmasscon/v1"
CUDA_APMASSCON_ABI = "gpuwm-cm1.cuda-apmasscon-kernels/v1"
CUDA_APMASSCON_THREADS = 256
CUDA_APMASSCON_KERNEL_NAMES = (
    "cm1_apmasscon_reduce_v1",
    "cm1_apmasscon_finish_v1",
    "cm1_apmasscon_apply_v1",
)
CUDA_APMASSCON_CLAIM_LIMIT = (
    "CM1 r21.1 single-domain Cartesian, moist, final-RK apmasscon=1 branch; "
    "default testcase constants and physical-domain arrays only; no MPI, "
    "axisymmetric volume factor, halo fill, pressure solve, or forecast claim"
)

SOURCE_ANCHORS = {
    "initial_dry_mass": "src/init3d.F:2483-2527",
    "pre_correction_eos": "src/solve2.F:1559-1595",
    "layer_reductions": "src/solve2.F:1616-1656",
    "finish_correction": "src/solve2.F:1972-2030",
    "default_constants": "src/constants.F:100-143",
}

INPUT_ARRAY_NAMES = ("ruh", "rvh", "rmh", "pi0", "th0", "th3d", "qv")
MUTABLE_ARRAY_NAMES = ("pp3d", "prs", "rho")
ALL_ARRAY_NAMES = INPUT_ARRAY_NAMES + MUTABLE_ARRAY_NAMES


@dataclass(frozen=True, slots=True)
class ApMassConSelection:
    """Exact CM1 branch values admitted by this implementation."""

    apmasscon: int = 1
    nrk: int = 3
    nrkmax: int = 3
    psolver: int = 3
    moist: int = 1
    axisymmetric: int = 0
    testcase: int = 0


DEFAULT_APMASSCON_SELECTION = ApMassConSelection()

_EXPECTED_SELECTION = {
    "dynamics.dry_mass_correction": ("apmasscon", 1),
    "runtime.nrk": ("nrk", 3),
    "runtime.nrkmax": ("nrkmax", 3),
    "dynamics.pressure_solver": ("psolver", 3),
    "physics.moist": ("moist", 1),
    "grid.axisymmetric": ("axisymmetric", 0),
    "runtime.testcase": ("testcase", 0),
}


def _validate_selection(selection: ApMassConSelection) -> None:
    if not isinstance(selection, ApMassConSelection):
        raise TypeError("selection must be an ApMassConSelection")
    for option, (attribute, expected) in _EXPECTED_SELECTION.items():
        value = getattr(selection, attribute)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            refuse(
                code="CM1-APMASSCON-UNSUPPORTED",
                gate="boundaries-closures",
                option=option,
                value=value,
                reason=(
                    f"the sealed mass correction covers only {option}={expected}; "
                    "another value selects a different CM1 equation or cadence"
                ),
                remedy="use the real-product branch or add a separate native oracle",
            )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    words = ni * nj * nk
    if words > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("grid word count exceeds signed 64-bit indexing")


def _validated_mass1(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("mass1 must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("mass1 must be finite and > 0")
    return result


def apmasscon_array_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    scalar = (nk, nj, ni)
    return {
        "ruh": (ni,),
        "rvh": (nj,),
        "rmh": scalar,
        "pi0": scalar,
        "th0": scalar,
        "th3d": scalar,
        "qv": scalar,
        "pp3d": scalar,
        "prs": scalar,
        "rho": scalar,
    }


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = apmasscon_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing apmasscon arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        if name in MUTABLE_ARRAY_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("apmasscon arrays must not alias")
    return result


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _f32_div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _f32_pow(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.power(np.float32(left), np.float32(right)))


def _default_constants_f32(np: Any) -> dict[str, Any]:
    rd = np.float32(287.04)
    cp = np.float32(1005.7)
    rv = np.float32(461.5)
    cv = np.float32(cp - rd)
    return {
        "p00": np.float32(1.0e5),
        "rd": rd,
        "rv": rv,
        "rddcv": np.float32(rd / cv),
        "cpdrd": np.float32(cp / rd),
    }


def _layer_reductions(
    arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int, include_pressure: bool
) -> tuple[Any, Any | None]:
    """Form default-REAL terms and accumulate them in binary64, source order."""

    np = importlib.import_module("numpy")
    dry = np.zeros((nk,), dtype=np.float64)
    pressure = np.zeros((nk,), dtype=np.float64) if include_pressure else None
    for k in range(nk):
        dry_sum = np.float64(0.0)
        pressure_sum = np.float64(0.0)
        for j in range(nj):
            for i in range(ni):
                term = _f32_mul(np, arrays["rho"][k, j, i], arrays["ruh"][i])
                term = _f32_mul(np, term, arrays["rvh"][j])
                term = _f32_mul(np, term, arrays["rmh"][k, j, i])
                dry_sum = np.float64(dry_sum + np.float64(term))
                if include_pressure:
                    pi = _f32_add(
                        np, arrays["pi0"][k, j, i], arrays["pp3d"][k, j, i]
                    )
                    pressure_sum = np.float64(pressure_sum + np.float64(pi))
        dry[k] = dry_sum
        if pressure is not None:
            pressure[k] = pressure_sum
    return dry, pressure


def _ordered_f64_sum(values: Any) -> float:
    np = importlib.import_module("numpy")
    result = np.float64(0.0)
    for value in values:
        result = np.float64(result + np.float64(value))
    return float(result)


def cpu_initial_dry_mass(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    selection: ApMassConSelection = DEFAULT_APMASSCON_SELECTION,
) -> float:
    """Return CM1's persisted ``mass1`` ledger for the physical domain."""

    _validate_selection(selection)
    a = _require_host_arrays(arrays, ni, nj, nk)
    layers, _ = _layer_reductions(
        a, ni=ni, nj=nj, nk=nk, include_pressure=False
    )
    result = _ordered_f64_sum(layers)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("initial dry-air mass ledger is not finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class CpuApMassConResult:
    mass1_f64: float
    mass2_before_f64: float
    pressure_sum_before_f64: float
    correction_f32: float
    mass_after_f64: float


def cpu_apmasscon_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    mass1: float,
    selection: ApMassConSelection = DEFAULT_APMASSCON_SELECTION,
) -> CpuApMassConResult:
    """Apply ``solve2.F:1618-1656,1977-2030`` in source statement order."""

    _validate_selection(selection)
    mass1_value = _validated_mass1(mass1)
    np = importlib.import_module("numpy")
    a = _require_host_arrays(arrays, ni, nj, nk)
    constants = _default_constants_f32(np)
    dry_layers, pressure_layers = _layer_reductions(
        a, ni=ni, nj=nj, nk=nk, include_pressure=True
    )
    assert pressure_layers is not None
    mass2 = _ordered_f64_sum(dry_layers)
    pressure_sum = _ordered_f64_sum(pressure_layers)
    if not math.isfinite(mass2) or mass2 <= 0.0:
        raise ValueError("current dry-air mass ledger is not finite and positive")
    ratio = np.float64(mass1_value) / np.float64(mass2)
    exponent = np.float64(constants["rddcv"])
    correction64 = np.float64(np.power(ratio, exponent) - np.float64(1.0))
    correction64 = np.float64(correction64 * np.float64(pressure_sum))
    correction64 = np.float64(
        correction64 / np.float64(ni * nj * nk)
    )
    correction = np.float32(correction64)
    if not bool(np.isfinite(correction)):
        raise ValueError("apmasscon Exner correction is not finite")

    zero = np.float32(0.0)
    for k in range(nk):
        for j in range(nj):
            for i in range(ni):
                a["pp3d"][k, j, i] = _f32_add(
                    np, a["pp3d"][k, j, i], correction
                )
                pi = _f32_add(np, a["pi0"][k, j, i], a["pp3d"][k, j, i])
                pressure = _f32_mul(
                    np, constants["p00"], _f32_pow(np, pi, constants["cpdrd"])
                )
                theta = _f32_add(np, a["th0"][k, j, i], a["th3d"][k, j, i])
                vapor = np.maximum(zero, a["qv"][k, j, i]).astype(np.float32)
                gas = _f32_add(np, constants["rd"], _f32_mul(np, vapor, constants["rv"]))
                denominator = _f32_mul(np, theta, pi)
                denominator = _f32_mul(np, denominator, gas)
                density = _f32_div(np, pressure, denominator)
                a["prs"][k, j, i] = pressure
                a["rho"][k, j, i] = density

    after_layers, _ = _layer_reductions(
        a, ni=ni, nj=nj, nk=nk, include_pressure=False
    )
    mass_after = _ordered_f64_sum(after_layers)
    return CpuApMassConResult(
        mass1_f64=mass1_value,
        mass2_before_f64=mass2,
        pressure_sum_before_f64=pressure_sum,
        correction_f32=float(correction),
        mass_after_f64=mass_after,
    )


CUDA_APMASSCON_SOURCE = r"""// gpuwm-cm1.cuda-apmasscon-kernels/v1
__device__ __forceinline__ float cm1_add(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float cm1_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float cm1_div(float a, float b) { return __fdiv_rn(a, b); }

extern "C" __global__ void cm1_apmasscon_reduce_v1(
    const float* __restrict__ rho,
    const float* __restrict__ ruh,
    const float* __restrict__ rvh,
    const float* __restrict__ rmh,
    const float* __restrict__ pi0,
    const float* __restrict__ pp3d,
    double* __restrict__ dry_layer,
    double* __restrict__ pressure_layer,
    unsigned int ni,
    unsigned int nj,
    unsigned int nk)
{
    const unsigned int k = blockIdx.x;
    if (k >= nk) return;
    const unsigned long long nxy = (unsigned long long)ni * (unsigned long long)nj;
    double dry = 0.0;
    double pressure = 0.0;
    for (unsigned long long ij = threadIdx.x; ij < nxy; ij += blockDim.x) {
        const unsigned int i = (unsigned int)(ij % (unsigned long long)ni);
        const unsigned int j = (unsigned int)(ij / (unsigned long long)ni);
        const unsigned long long index = (unsigned long long)k * nxy + ij;
        float term = cm1_mul(rho[index], ruh[i]);
        term = cm1_mul(term, rvh[j]);
        term = cm1_mul(term, rmh[index]);
        dry += (double)term;
        pressure += (double)cm1_add(pi0[index], pp3d[index]);
    }
    __shared__ double dry_shared[256];
    __shared__ double pressure_shared[256];
    dry_shared[threadIdx.x] = dry;
    pressure_shared[threadIdx.x] = pressure;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            dry_shared[threadIdx.x] += dry_shared[threadIdx.x + stride];
            pressure_shared[threadIdx.x] += pressure_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        dry_layer[k] = dry_shared[0];
        pressure_layer[k] = pressure_shared[0];
    }
}

extern "C" __global__ void cm1_apmasscon_finish_v1(
    const double* __restrict__ dry_layer,
    const double* __restrict__ pressure_layer,
    double* __restrict__ diagnostics,
    float* __restrict__ correction,
    double mass1,
    float rddcv,
    unsigned long long words,
    unsigned int nk)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    double mass2 = 0.0;
    double pressure_sum = 0.0;
    for (unsigned int k = 0u; k < nk; ++k) {
        mass2 += dry_layer[k];
        pressure_sum += pressure_layer[k];
    }
    const double value = (pow(mass1 / mass2, (double)rddcv) - 1.0)
                       * pressure_sum / (double)words;
    correction[0] = (float)value;
    diagnostics[0] = mass2;
    diagnostics[1] = pressure_sum;
    diagnostics[2] = value;
}

extern "C" __global__ void cm1_apmasscon_apply_v1(
    const float* __restrict__ pi0,
    const float* __restrict__ th0,
    const float* __restrict__ th3d,
    const float* __restrict__ qv,
    float* __restrict__ pp3d,
    float* __restrict__ prs,
    float* __restrict__ rho,
    const float* __restrict__ correction,
    float p00,
    float rd,
    float rv,
    float cpdrd,
    unsigned long long words)
{
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= words) return;
    const float pp = cm1_add(pp3d[index], correction[0]);
    const float pi = cm1_add(pi0[index], pp);
    const float pressure = cm1_mul(p00, powf(pi, cpdrd));
    const float theta = cm1_add(th0[index], th3d[index]);
    const float vapor = fmaxf(0.0f, qv[index]);
    const float gas = cm1_add(rd, cm1_mul(vapor, rv));
    float denominator = cm1_mul(theta, pi);
    denominator = cm1_mul(denominator, gas);
    pp3d[index] = pp;
    prs[index] = pressure;
    rho[index] = cm1_div(pressure, denominator);
}
"""

CUDA_APMASSCON_SOURCE_BYTES = CUDA_APMASSCON_SOURCE.encode("utf-8")
CUDA_APMASSCON_SOURCE_SHA256 = (
    "937c53c4889fe39e0564bf31fdb534718f13eb0d94c01e2b88513d41100675c4"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_APMASSCON_SOURCE_BYTES).hexdigest() == (
        CUDA_APMASSCON_SOURCE_SHA256
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
class CompiledApMassConKernels:
    module_owner: Any
    kernels: tuple[Any, Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_apmasscon() -> CompiledApMassConKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged apmasscon CUDA source differs from its pin")
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
        CUDA_APMASSCON_SOURCE, "cm1_apmasscon_v1_ptx.cu", method="ptx"
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_APMASSCON_SOURCE, "cm1_apmasscon_v1_cubin.cu", method="cubin"
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
        for name in CUDA_APMASSCON_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks one or more apmasscon entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled apmasscon image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_APMASSCON_KERNEL_NAMES
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
    return CompiledApMassConKernels(
        module_owner=module,
        kernels=kernels,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = apmasscon_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing apmasscon arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("apmasscon arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaApMassConLaunch:
    ready_event: Any
    compiled: CompiledApMassConKernels
    layer_dry_mass: Any
    layer_pressure_sum: Any
    diagnostics: Any
    correction: Any
    receipt: dict[str, Any]


def launch_cuda_apmasscon(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    mass1: float,
    selection: ApMassConSelection = DEFAULT_APMASSCON_SELECTION,
    stream: Any | None = None,
    compiled: CompiledApMassConKernels | None = None,
) -> CudaApMassConLaunch:
    """Enqueue the complete final-stage correction without host synchronization."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    mass1_value = _validated_mass1(mass1)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    if compiled is None:
        compiled = compile_cuda_apmasscon()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    np = importlib.import_module("numpy")
    constants = _default_constants_f32(np)
    layer_dry_mass = cupy.empty((nk,), dtype=cupy.float64)
    layer_pressure_sum = cupy.empty((nk,), dtype=cupy.float64)
    diagnostics = cupy.empty((3,), dtype=cupy.float64)
    correction = cupy.empty((1,), dtype=cupy.float32)
    u32 = cupy.uint32
    u64 = cupy.uint64
    f32 = cupy.float32
    f64 = cupy.float64
    words = ni * nj * nk
    compiled.kernels[0](
        (nk, 1, 1),
        (CUDA_APMASSCON_THREADS, 1, 1),
        (
            a["rho"],
            a["ruh"],
            a["rvh"],
            a["rmh"],
            a["pi0"],
            a["pp3d"],
            layer_dry_mass,
            layer_pressure_sum,
            u32(ni),
            u32(nj),
            u32(nk),
        ),
        shared_mem=0,
        stream=stream,
    )
    compiled.kernels[1](
        (1, 1, 1),
        (1, 1, 1),
        (
            layer_dry_mass,
            layer_pressure_sum,
            diagnostics,
            correction,
            f64(mass1_value),
            f32(constants["rddcv"]),
            u64(words),
            u32(nk),
        ),
        shared_mem=0,
        stream=stream,
    )
    grid_words = (words + CUDA_APMASSCON_THREADS - 1) // CUDA_APMASSCON_THREADS
    compiled.kernels[2](
        (grid_words, 1, 1),
        (CUDA_APMASSCON_THREADS, 1, 1),
        (
            a["pi0"],
            a["th0"],
            a["th3d"],
            a["qv"],
            a["pp3d"],
            a["prs"],
            a["rho"],
            correction,
            f32(constants["p00"]),
            f32(constants["rd"]),
            f32(constants["rv"]),
            f32(constants["cpdrd"]),
            u64(words),
        ),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_APMASSCON_SCHEMA,
        "claim_limit": CUDA_APMASSCON_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "selection": {
            option: getattr(selection, attribute)
            for option, (attribute, _) in _EXPECTED_SELECTION.items()
        },
        "domain": {"ni": ni, "nj": nj, "nk": nk, "words": words},
        "mass1_f64": mass1_value,
        "kernel": {
            "abi": CUDA_APMASSCON_ABI,
            "names": list(CUDA_APMASSCON_KERNEL_NAMES),
            "source_sha256": CUDA_APMASSCON_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "ptx_bytes": len(compiled.ptx),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "cubin_bytes": len(compiled.cubin),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "reduction_blocks": nk,
            "reduction_block_x": CUDA_APMASSCON_THREADS,
            "apply_grid_x": grid_words,
            "apply_block_x": CUDA_APMASSCON_THREADS,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaApMassConLaunch(
        ready_event=event,
        compiled=compiled,
        layer_dry_mass=layer_dry_mass,
        layer_pressure_sum=layer_pressure_sum,
        diagnostics=diagnostics,
        correction=correction,
        receipt=receipt,
    )


__all__ = [
    "ALL_ARRAY_NAMES",
    "ApMassConSelection",
    "CUDA_APMASSCON_ABI",
    "CUDA_APMASSCON_CLAIM_LIMIT",
    "CUDA_APMASSCON_KERNEL_NAMES",
    "CUDA_APMASSCON_SCHEMA",
    "CUDA_APMASSCON_SOURCE",
    "CUDA_APMASSCON_SOURCE_SHA256",
    "CompiledApMassConKernels",
    "CpuApMassConResult",
    "CudaApMassConLaunch",
    "DEFAULT_APMASSCON_SELECTION",
    "INPUT_ARRAY_NAMES",
    "MUTABLE_ARRAY_NAMES",
    "SOURCE_ANCHORS",
    "apmasscon_array_shapes",
    "compile_cuda_apmasscon",
    "cpu_apmasscon_inplace",
    "cpu_initial_dry_mass",
    "launch_cuda_apmasscon",
]
