"""Startup thermodynamics for a freshly initialized Cartesian grid.

CM1 r21.1 rebuilds the *current* pressure and density from the initialized
perturbation state before the model advances.  ``src/init3d.F`` does this in
``getset`` (init3d.F:2032-2872); the ``psolver.le.3`` path used by this port is
init3d.F:2386-2467.  Its moist branch takes the pressure from the total
Exner function, ``p00`` times ``(pi0 + pp3d)`` raised to ``cp/rd``, and the
density from the gas law: that pressure divided by the product of the total
potential temperature ``th0 + th3d``, the total Exner function and
``rd + max(0, qv)*rv``.  It then publishes ``rr``, the reciprocal density,
and ``rf``, the face density: ``c1``/``c2`` weights of the two neighbouring
levels inside the column, and three-level extrapolations with the ``cgs`` and
``cgt`` weights at the bottom and top faces.

The warm bubble perturbs ``th3d``.  Initializing the current words from the
base state instead leaves the initial dry-mass ledger and the first closure
computed against an atmosphere that does not exist, so this runs once, before
``CartesianDynamics._initial_dry_mass()``.

Nothing here is new arithmetic.  The pressure/density kernel is the pinned
``cuda_apmasscon`` equation of state with no mass correction applied, and the
``rr``/``rf`` publication is the existing ``gm1_solve3_density_v1`` kernel that
``CartesianFinish.solve3`` already uses at the end of every completed step.
Base profiles stay immutable and no prognostic word other than ``prs``, ``rho``,
``rr`` and ``rf`` is written.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import re
from typing import Any

from .cuda_glibc_pow13 import (
    CUDA_GLIBC_POW13_DEVICE_SOURCE,
    CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
)


CUDA_CARTESIAN_STARTUP_SCHEMA = "gm1.cuda-cartesian-startup-thermodynamics/v1"
CUDA_CARTESIAN_STARTUP_ABI = "gm1.cuda-cartesian-startup-kernels/v1"
CUDA_CARTESIAN_STARTUP_THREADS = 128
CUDA_CARTESIAN_STARTUP_KERNEL_NAMES = ("gm1_cartesian_startup_eos_v1",)
CUDA_CARTESIAN_STARTUP_AUTHORITY = "src/init3d.F:2386-2467 (getset, psolver<=3, imoist=1)"
CUDA_CARTESIAN_STARTUP_CLAIM_LIMIT = (
    "publishes initial prs/rho/rr/rf from the initialized perturbation state; "
    "it is not a trajectory, restart or native-CM1 parity claim"
)

# CM1 constants.incl default-REAL values.  cpdrd is formed once, from the same
# single-precision operands the Fortran uses, exactly as cuda_apmasscon does.
CM1_RD_F32 = 287.04
CM1_CP_F32 = 1005.7
CM1_RV_F32 = 461.5
CM1_P00_F32 = 1.0e5


CUDA_CARTESIAN_STARTUP_SOURCE = CUDA_GLIBC_POW13_DEVICE_SOURCE + r"""// gm1.cuda-cartesian-startup-kernels/v1
__device__ __forceinline__ int gm1_startup_positive_normal_f32(float x) {
  unsigned bits = __float_as_uint(x);
  return bits >= 0x00800000u && bits < 0x7f800000u;
}

// The exact glibc-2.39 powf with the cpdrd exponent pinned as a raw word, so
// this is bit-identical to gm1_rk3_eqtset2_remove_snapshot_eos_f32_v2's
// equation of state.  Using CUDA's powf here instead would make the startup
// state differ from the mid-step state by 1-2 ulp for no reason.
__device__ __forceinline__ float gm1_startup_pow_cpdrd_positive_normal(float x) {
  double logx = cm1_powf_log2_positive_normal(__float_as_uint(x));
  double ylogx = DMU((double)__uint_as_float(0x40603c81u), logx);
  return __double2float_rn(cm1_powf_exp2(ylogx));
}

// init3d.F:2386-2400, moist branch.  pp3d, th3d and q3d are read only; the
// perturbation words themselves are never rewritten here.  Constants carry the
// raw words of CM1's default-REAL p00, rd and rv.
extern "C" __global__ void gm1_cartesian_startup_eos_v1(
    const float* __restrict__ pi0,
    const float* __restrict__ pp3d,
    const float* __restrict__ th0,
    const float* __restrict__ th3d,
    const float* __restrict__ qv,
    float* __restrict__ prs,
    float* __restrict__ rho,
    unsigned int* __restrict__ invalid_words,
    unsigned long long words)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long step = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long index = first; index < words; index += step) {
        const float pi = __fadd_rn(pi0[index], pp3d[index]);
        if (!gm1_startup_positive_normal_f32(pi)) {
            prs[index] = __uint_as_float(0x7fc00000u);
            rho[index] = __uint_as_float(0x7fc00000u);
            atomicAdd(invalid_words, 1u);
            continue;
        }
        const float pressure = __fmul_rn(
            __uint_as_float(0x47c35000u),
            gm1_startup_pow_cpdrd_positive_normal(pi));
        const float theta = __fadd_rn(th0[index], th3d[index]);
        const float vapor = fmaxf(__uint_as_float(0x00000000u), qv[index]);
        const float gas = __fadd_rn(
            __uint_as_float(0x438f851fu),
            __fmul_rn(vapor, __uint_as_float(0x43e6c000u)));
        float denominator = __fmul_rn(theta, pi);
        denominator = __fmul_rn(denominator, gas);
        const float density = __fdiv_rn(pressure, denominator);
        if (!gm1_startup_positive_normal_f32(pressure) ||
            !gm1_startup_positive_normal_f32(denominator) ||
            !gm1_startup_positive_normal_f32(density)) {
            prs[index] = __uint_as_float(0x7fc00000u);
            rho[index] = __uint_as_float(0x7fc00000u);
            atomicAdd(invalid_words, 1u);
            continue;
        }
        prs[index] = pressure;
        rho[index] = density;
    }
}
"""
CUDA_CARTESIAN_STARTUP_SOURCE_BYTES = CUDA_CARTESIAN_STARTUP_SOURCE.encode("utf-8")
CUDA_CARTESIAN_STARTUP_SOURCE_SHA256 = (
    "6e2855e63b1b238b10dd20c498dbc317a7605bcc584fc2cacb5d2dabd8cd0866"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_CARTESIAN_STARTUP_SOURCE_BYTES).hexdigest() == (
        CUDA_CARTESIAN_STARTUP_SOURCE_SHA256
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


def _image_options(arch: str, nvrtc_major: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = (
        # c++17: the glibc-2.39 powf transcription uses hex-float literals.
        "--std=c++17",
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
class CompiledCartesianStartupKernels:
    module_owner: Any
    kernels: tuple[Any, ...]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_cartesian_startup() -> CompiledCartesianStartupKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged Cartesian startup CUDA source differs from its pin")
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
    ptx, _ = compiler._NVRTCProgram(
        CUDA_CARTESIAN_STARTUP_SOURCE, "gm1_cartesian_startup_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_CARTESIAN_STARTUP_SOURCE, "gm1_cartesian_startup_v1_cubin.cu", method="cubin"
    ).compile(cubin_options)
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
        for name in CUDA_CARTESIAN_STARTUP_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks the Cartesian startup entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled Cartesian startup image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_CARTESIAN_STARTUP_KERNEL_NAMES
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
    return CompiledCartesianStartupKernels(
        module_owner=module,
        kernels=kernels,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def startup_constants_f32(numpy: Any) -> dict[str, Any]:
    """The four default-REAL constants, rounded exactly once, as CM1 does."""

    rd = numpy.float32(CM1_RD_F32)
    cp = numpy.float32(CM1_CP_F32)
    return {
        "p00": numpy.float32(CM1_P00_F32),
        "rd": rd,
        "rv": numpy.float32(CM1_RV_F32),
        "cpdrd": numpy.float32(cp / rd),
    }


def _host_powf(numpy: Any, base: Any, exponent: Any) -> Any:
    """``powf`` from the host C library, so the host reference is not a
    different transcendental from the device's glibc transcription.

    numpy's float32 ``power`` computes in double and rounds once, which is not
    what either glibc's ``powf`` or the device kernel does.  Falling back to it
    is recorded rather than hidden.
    """

    import ctypes
    import ctypes.util

    name = ctypes.util.find_library("m")
    if name is not None:
        try:
            libm = ctypes.CDLL(name)
            libm.powf.restype = ctypes.c_float
            libm.powf.argtypes = (ctypes.c_float, ctypes.c_float)
        except (OSError, AttributeError):
            libm = None
        else:
            flat = numpy.asarray(base, numpy.float32).reshape(-1)
            y = float(exponent)
            out = numpy.empty(flat.shape, numpy.float32)
            for index in range(flat.size):
                out[index] = libm.powf(float(flat[index]), y)
            return out.reshape(numpy.asarray(base).shape)
    return numpy.float32(numpy.power(numpy.asarray(base, numpy.float32), numpy.float32(exponent)))


def cpu_startup_thermodynamics(
    fields: Mapping[str, Any],
    *,
    numpy: Any,
    c1: Any,
    c2: Any,
) -> dict[str, Any]:
    """Host reference for the same operation, in source order.

    ``fields`` supplies ``pi0``, ``pp3d``, ``th0``, ``th3d`` and ``qv`` as
    ``(nk, nj, ni)`` float32 arrays; ``c1``/``c2`` are ``(nk+1, nj, ni)``.  The
    return value carries ``prs``, ``rho``, ``rr`` and ``rf``.
    """

    constants = startup_constants_f32(numpy)
    f32 = numpy.float32

    def add(left: Any, right: Any) -> Any:
        return f32(f32(left) + f32(right))

    def mul(left: Any, right: Any) -> Any:
        return f32(f32(left) * f32(right))

    def div(left: Any, right: Any) -> Any:
        return f32(f32(left) / f32(right))

    pi0, pp3d = numpy.asarray(fields["pi0"], f32), numpy.asarray(fields["pp3d"], f32)
    th0, th3d = numpy.asarray(fields["th0"], f32), numpy.asarray(fields["th3d"], f32)
    qv = numpy.asarray(fields["qv"], f32)
    nk, nj, ni = pi0.shape

    pi = add(pi0, pp3d)
    prs = mul(constants["p00"], _host_powf(numpy, pi, constants["cpdrd"]))
    theta = add(th0, th3d)
    vapor = numpy.maximum(f32(0.0), qv).astype(f32)
    gas = add(constants["rd"], mul(vapor, constants["rv"]))
    rho = div(prs, mul(mul(theta, pi), gas))

    rr = div(f32(1.0), rho)
    c1a, c2a = numpy.asarray(c1, f32), numpy.asarray(c2, f32)
    rf = numpy.zeros((nk + 1, nj, ni), dtype=f32)
    for k in range(1, nk):
        rf[k] = add(mul(c1a[k], rho[k - 1]), mul(c2a[k], rho[k]))
    # cm1r17 second-order extrapolation; the uniform-grid delta() weights.
    rf[0] = add(
        add(mul(f32(1.875), rho[0]), mul(f32(-1.25), rho[1])), mul(f32(0.375), rho[2])
    )
    rf[nk] = add(
        add(mul(f32(1.875), rho[nk - 1]), mul(f32(-1.25), rho[nk - 2])),
        mul(f32(0.375), rho[nk - 3]),
    )
    return {"prs": prs, "rho": rho, "rr": rr, "rf": rf}


def launch_cuda_cartesian_startup(
    fields: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    cupy: Any,
    stream: Any,
    compiled: CompiledCartesianStartupKernels,
    density_kernel: Any,
    numq: int = 6,
) -> dict[str, Any]:
    """Publish initial ``prs``/``rho`` then ``rr``/``rf`` on ``stream``.

    ``density_kernel`` is ``gm1_solve3_density_v1`` from the already-compiled
    forecast image, so the completed-step publication and the startup
    publication are literally the same instructions.
    """

    numpy = importlib.import_module("numpy")
    if int(ni) <= 0 or int(nj) <= 0 or int(nk) < 4:
        raise ValueError("startup thermodynamics needs at least four vertical levels")
    mass_shape = (int(nk), int(nj), int(ni))
    face_shape = (int(nk) + 1, int(nj), int(ni))
    required = {
        "pi0": mass_shape,
        "pp3d": mass_shape,
        "th0": mass_shape,
        "th3d": mass_shape,
        "prs": mass_shape,
        "rho": mass_shape,
        "rr": mass_shape,
        "rf": face_shape,
    }
    resolved: dict[str, Any] = {}
    for name, shape in required.items():
        value = fields.get(name)
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, received {tuple(value.shape)}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        resolved[name] = value
    vapor = fields["q3d"]
    if not isinstance(vapor, cupy.ndarray) or vapor.dtype != cupy.dtype("float32"):
        raise TypeError("q3d must be a float32 cupy.ndarray")
    # init3d.F getset's moist density branch reads q3d(nqv) and nothing else,
    # and nqv=1 on every admitted axis, so the axis length is the component's
    # numq.  It is still checked exactly: a q3d of any other length is not the
    # axis this run allocated.
    if isinstance(numq, bool) or not isinstance(numq, int) or numq < 1:
        raise ValueError(f"numq must be a positive integer, received {numq!r}")
    if tuple(vapor.shape) != (numq, *mass_shape):
        raise ValueError(f"q3d shape must be {(numq, *mass_shape)}, received {tuple(vapor.shape)}")
    qv = vapor[0]
    if not qv.flags.c_contiguous:
        raise ValueError("the vapour slice of q3d must be C contiguous")
    for name in ("c1", "c2"):
        value = metrics.get(name)
        if not isinstance(value, cupy.ndarray) or tuple(value.shape) != face_shape:
            raise ValueError(f"metric {name} must be a cupy.ndarray with shape {face_shape}")

    written = [int(resolved[name].data.ptr) for name in ("prs", "rho", "rr", "rf")]
    read = [int(resolved[name].data.ptr) for name in ("pi0", "pp3d", "th0", "th3d")]
    read += [int(qv.data.ptr), int(metrics["c1"].data.ptr), int(metrics["c2"].data.ptr)]
    if len(set(written)) != len(written) or set(written) & set(read):
        raise ValueError("startup thermodynamics outputs must not alias their inputs")

    constants = startup_constants_f32(numpy)
    words = int(ni) * int(nj) * int(nk)
    face_words = int(ni) * int(nj) * (int(nk) + 1)
    threads = CUDA_CARTESIAN_STARTUP_THREADS
    invalid = cupy.zeros((1,), dtype=cupy.uint32)
    with stream:
        compiled.kernels[0](
            ((words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                resolved["pi0"],
                resolved["pp3d"],
                resolved["th0"],
                resolved["th3d"],
                qv,
                resolved["prs"],
                resolved["rho"],
                invalid,
                cupy.uint64(words),
            ),
            stream=stream,
        )
        density_kernel(
            ((face_words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                resolved["rho"],
                metrics["c1"],
                metrics["c2"],
                resolved["rr"],
                resolved["rf"],
                cupy.uint32(ni),
                cupy.uint32(nj),
                cupy.uint32(nk),
            ),
            stream=stream,
        )
    stream.synchronize()
    invalid_words = int(invalid.get()[0])
    if invalid_words:
        raise RuntimeError(
            f"startup thermodynamics produced {invalid_words} non-normal words; "
            "the initialized Exner/theta/vapour state is not physical"
        )
    return {
        "schema": CUDA_CARTESIAN_STARTUP_SCHEMA,
        "abi": CUDA_CARTESIAN_STARTUP_ABI,
        "authority": CUDA_CARTESIAN_STARTUP_AUTHORITY,
        "performed": True,
        "domain": {"ni": int(ni), "nj": int(nj), "nk": int(nk)},
        "published": ["prs", "rho", "rr", "rf"],
        "invalid_words": invalid_words,
        "pow_implementation": "glibc-2.39 powf transcription (gm1.cuda-glibc239-pow13/v1)",
        "glibc_pow_source_sha256": CUDA_GLIBC_POW13_DEVICE_SOURCE_SHA256,
        "base_profiles_mutated": False,
        "perturbation_words_mutated": False,
        "constants": {name: float(value) for name, value in constants.items()},
        "source_sha256": CUDA_CARTESIAN_STARTUP_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "runtime": dict(compiled.runtime_identity),
        "stream_pointer": int(stream.ptr),
        "claim_limit": CUDA_CARTESIAN_STARTUP_CLAIM_LIMIT,
    }


__all__ = (
    "CUDA_CARTESIAN_STARTUP_ABI",
    "CUDA_CARTESIAN_STARTUP_AUTHORITY",
    "CUDA_CARTESIAN_STARTUP_CLAIM_LIMIT",
    "CUDA_CARTESIAN_STARTUP_KERNEL_NAMES",
    "CUDA_CARTESIAN_STARTUP_SCHEMA",
    "CUDA_CARTESIAN_STARTUP_SOURCE",
    "CUDA_CARTESIAN_STARTUP_SOURCE_SHA256",
    "CUDA_CARTESIAN_STARTUP_THREADS",
    "CompiledCartesianStartupKernels",
    "compile_cuda_cartesian_startup",
    "cpu_startup_thermodynamics",
    "launch_cuda_cartesian_startup",
    "startup_constants_f32",
)
