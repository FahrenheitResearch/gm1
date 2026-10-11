"""CM1 r21.1 ``calccref`` and the chosen-level reflectivity plane, on device.

The three-dimensional reflectivity volume itself is not computed here.  It is
produced inside the microphysics call by the scheme that owns the size
distributions it depends on -- for this port, WRF's ``refl10cm_wsm6``
(``mp_wsm6.F90:2275-2444``) reached through the admitted public ArWen seam,
exactly as CM1 fills ``qdiag(:,:,:,qd_dbz)`` from the active ``ptype``'s own
routine (``cm1.F:3214-3237`` sets ``getdbz``; the scheme writes the field).

What this module owns is the pair of plan-view products CM1 derives from that
volume and writes beside it:

* ``cref``, the composite (column-maximum) reflectivity of
  ``misclibs.F:2259-2292``, written by ``writeout.F:1081-1091``.  CM1 seeds
  the accumulator at ``-1000.0`` and folds ``max`` up the column; the seed is
  kept because it is what makes a column of no levels report a value no
  meteorological floor can produce, rather than zero.
* the horizontal plane at one chosen model level, which CM1 leaves its users
  to slice out of the three-dimensional ``dbz`` field
  (``writeout.F:4089-4107``).  The level is selected by
  ``gm1_runtime.reflectivity_level_index``, which is CM1's own descending
  ``misclibs.F:2454-2466`` report-level scan.

Both reductions run on the device.  Nothing here derives, floors, rescales or
renders reflectivity: the words that go in are the words the microphysics
scheme produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import re
from typing import Any


CUDA_COMPOSITE_REFLECTIVITY_SCHEMA = "gm1.cuda-composite-reflectivity/v1"
CUDA_COMPOSITE_REFLECTIVITY_ABI = "gm1.cuda-composite-reflectivity-f32/v1"
CUDA_COMPOSITE_REFLECTIVITY_THREADS = 256
CUDA_COMPOSITE_REFLECTIVITY_KERNEL_NAMES = (
    "cm1_calccref_v1",
    "cm1_reflectivity_level_v1",
)
CUDA_COMPOSITE_REFLECTIVITY_CLAIM_LIMIT = (
    "CM1 r21.1 calccref column-maximum reduction and one chosen-level "
    "horizontal slice of a reflectivity volume this module did not compute; "
    "no reflectivity derivation, microphysics, unit conversion, floor, "
    "regrid, render, or forecast-skill claim"
)

SOURCE_ANCHORS = {
    "composite_reflectivity": "src/misclibs.F:2259-2292",
    "composite_reflectivity_writer": "src/writeout.F:1081-1091",
    "reflectivity_volume_writer": "src/writeout.F:4089-4107",
    "output_due_flag": "src/cm1.F:3214-3237",
    "namelist_word": "README.namelist:1366-1369",
    "level_scan": "src/misclibs.F:2454-2466",
}

#: ``misclibs.F:2267-2271``.  CM1 seeds every composite column at this value
#: before the ``k`` fold, and nothing in the fold can lower it.
CM1_CREF_SEED_DBZ = -1000.0

CUDA_COMPOSITE_REFLECTIVITY_SOURCE = r'''
// CM1 r21.1 misclibs.F:2259-2292 calccref, and the chosen-level slice of the
// reflectivity volume writeout.F:4089-4107 writes whole.
//
// Both kernels are one thread per horizontal column.  The composite fold is
// a running fmaxf over k from CM1's own -1000.0 seed; the level slice is an
// exact word copy.  Neither performs any floating-point arithmetic, so the
// result is bit-identical on any IEEE-754 binary32 host.
#define CREF_SEED (-1000.0f)

extern "C" __global__ void cm1_calccref_v1(
    const float* __restrict__ dbz, float* __restrict__ cref,
    unsigned int ni, unsigned int nj, unsigned int nk)
{
  const unsigned long long columns = (unsigned long long)ni * nj;
  const unsigned long long first =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  for (unsigned long long q = first; q < columns; q += stride) {
    float value = CREF_SEED;
    for (unsigned int k = 0u; k < nk; ++k) {
      value = fmaxf(value, dbz[(unsigned long long)k * columns + q]);
    }
    cref[q] = value;
  }
}

extern "C" __global__ void cm1_reflectivity_level_v1(
    const float* __restrict__ dbz, float* __restrict__ plane,
    unsigned int ni, unsigned int nj, unsigned int level)
{
  const unsigned long long columns = (unsigned long long)ni * nj;
  const unsigned long long first =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  for (unsigned long long q = first; q < columns; q += stride) {
    plane[q] = dbz[(unsigned long long)level * columns + q];
  }
}
'''
CUDA_COMPOSITE_REFLECTIVITY_SOURCE_BYTES = (
    CUDA_COMPOSITE_REFLECTIVITY_SOURCE.encode("utf-8")
)
CUDA_COMPOSITE_REFLECTIVITY_SOURCE_SHA256 = hashlib.sha256(
    CUDA_COMPOSITE_REFLECTIVITY_SOURCE_BYTES
).hexdigest()


def composite_reflectivity_shapes(
    ni: int, nj: int, nk: int
) -> dict[str, tuple[int, ...]]:
    """The exact array shapes this operator reads and writes."""

    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    return {
        "dbz": (nk, nj, ni),
        "cref": (nj, ni),
        "plane": (nj, ni),
    }


def cpu_composite_reflectivity(dbz: Any) -> Any:
    """``misclibs.F:2259-2292`` on the host, word for word.

    The seed and the fold order are CM1's.  ``numpy.maximum`` is IEEE
    ``fmaxf`` on binary32 finite words, and the fold runs ``k`` ascending as
    CM1's ``do k=1,nk`` does, so this returns the same words the kernel does.
    """

    np = importlib.import_module("numpy")
    if (
        not isinstance(dbz, np.ndarray)
        or dbz.dtype != np.dtype("float32")
        or dbz.ndim != 3
    ):
        raise ValueError("dbz must be a three-dimensional float32 [nk,nj,ni] array")
    nk, nj, ni = dbz.shape
    cref = np.full((nj, ni), np.float32(CM1_CREF_SEED_DBZ), dtype=np.float32)
    for k in range(nk):
        np.maximum(cref, dbz[k], out=cref)
    return cref


def cpu_reflectivity_level(dbz: Any, level: int) -> Any:
    """``writeout.F:4089-4107``'s volume, sliced at one model level."""

    np = importlib.import_module("numpy")
    if (
        not isinstance(dbz, np.ndarray)
        or dbz.dtype != np.dtype("float32")
        or dbz.ndim != 3
    ):
        raise ValueError("dbz must be a three-dimensional float32 [nk,nj,ni] array")
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError("level must be a zero-based integer model level")
    if not 0 <= level < dbz.shape[0]:
        raise ValueError(
            f"level {level} is outside the {dbz.shape[0]} model levels of this "
            "reflectivity volume; slicing it would read another column's words"
        )
    return np.ascontiguousarray(dbz[level])


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(part, int) and not isinstance(part, bool) for part in value):
            return f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower()
    for prefix in ("compute_", "sm_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        major, minor = text.split(".", 1)
        text = major + minor
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("CUDA architecture must be canonical digits")
    return text


def _image_options(arch: str, nvrtc_major: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
class CompiledCompositeReflectivityKernels:
    module_owner: Any
    kernels: tuple[Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_composite_reflectivity() -> CompiledCompositeReflectivityKernels:
    """Compile the packaged source and admit only its own entry points."""

    digest = hashlib.sha256(CUDA_COMPOSITE_REFLECTIVITY_SOURCE_BYTES).hexdigest()
    if digest != CUDA_COMPOSITE_REFLECTIVITY_SOURCE_SHA256:
        raise RuntimeError("packaged composite reflectivity source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc_parts = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc_parts[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_COMPOSITE_REFLECTIVITY_SOURCE, "cm1_calccref_v1_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_COMPOSITE_REFLECTIVITY_SOURCE, "cm1_calccref_v1_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_COMPOSITE_REFLECTIVITY_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks a composite reflectivity entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled composite reflectivity image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name)
        for name in CUDA_COMPOSITE_REFLECTIVITY_KERNEL_NAMES
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
    return CompiledCompositeReflectivityKernels(
        module_owner=module,
        kernels=kernels,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


@dataclass(frozen=True, slots=True)
class CudaCompositeReflectivityLaunch:
    ready_event: Any
    compiled: CompiledCompositeReflectivityKernels
    receipt: Mapping[str, Any]


def launch_cuda_composite_reflectivity(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    level: int | None = None,
    stream: Any | None = None,
    compiled: CompiledCompositeReflectivityKernels | None = None,
) -> CudaCompositeReflectivityLaunch:
    """Enqueue CM1's composite fold, and the level slice when one is declared."""

    shapes = composite_reflectivity_shapes(ni, nj, nk)
    cupy = importlib.import_module("cupy")
    required = ["dbz", "cref"] + ([] if level is None else ["plane"])
    values: dict[str, Any] = {}
    for name in required:
        value = arrays.get(name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or tuple(value.shape) != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"composite reflectivity {name} must be a C-order CUDA float32 "
                f"array of shape {shapes[name]}"
            )
        values[name] = value
    pointers = [int(value.data.ptr) for value in values.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("composite reflectivity arrays must not alias")
    if level is not None:
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("level must be a zero-based integer model level")
        if not 0 <= level < nk:
            raise ValueError(
                f"reflectivity level {level} is outside the {nk} model levels of "
                "this grid; the published plane would read another level's words"
            )
    compiled = compiled or compile_cuda_composite_reflectivity()
    stream = stream or cupy.cuda.get_current_stream()
    columns = ni * nj
    threads = CUDA_COMPOSITE_REFLECTIVITY_THREADS
    grid = ((columns + threads - 1) // threads, 1, 1)
    compiled.kernels[0](
        grid, (threads, 1, 1),
        (values["dbz"], values["cref"],
         cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(nk)),
        stream=stream,
    )
    if level is not None:
        compiled.kernels[1](
            grid, (threads, 1, 1),
            (values["dbz"], values["plane"],
             cupy.uint32(ni), cupy.uint32(nj), cupy.uint32(level)),
            stream=stream,
        )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_COMPOSITE_REFLECTIVITY_SCHEMA,
        "abi": CUDA_COMPOSITE_REFLECTIVITY_ABI,
        "claim_limit": CUDA_COMPOSITE_REFLECTIVITY_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "domain": {"ni": ni, "nj": nj, "nk": nk, "columns": columns},
        "composite_seed_dbz": CM1_CREF_SEED_DBZ,
        "level": level,
        "kernel_source_sha256": CUDA_COMPOSITE_REFLECTIVITY_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
    }
    return CudaCompositeReflectivityLaunch(event, compiled, receipt)


__all__ = (
    "CM1_CREF_SEED_DBZ",
    "CUDA_COMPOSITE_REFLECTIVITY_ABI",
    "CUDA_COMPOSITE_REFLECTIVITY_CLAIM_LIMIT",
    "CUDA_COMPOSITE_REFLECTIVITY_KERNEL_NAMES",
    "CUDA_COMPOSITE_REFLECTIVITY_SCHEMA",
    "CUDA_COMPOSITE_REFLECTIVITY_SOURCE",
    "CUDA_COMPOSITE_REFLECTIVITY_SOURCE_SHA256",
    "CUDA_COMPOSITE_REFLECTIVITY_THREADS",
    "CompiledCompositeReflectivityKernels",
    "CudaCompositeReflectivityLaunch",
    "SOURCE_ANCHORS",
    "composite_reflectivity_shapes",
    "compile_cuda_composite_reflectivity",
    "cpu_composite_reflectivity",
    "cpu_reflectivity_level",
    "launch_cuda_composite_reflectivity",
)
