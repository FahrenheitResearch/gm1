"""Face density at the ground and at the model top, with the column's own weights.

CM1 r21.1 extrapolates the density to the lowest and the highest w level with
a quadratic through the three nearest scalar levels: ``rf(1)`` from
``cgs1..3`` and ``rf(nk+1)`` from ``cgt1..3`` (``src/solve3.F:678-679``,
``src/init3d.F:2464-2465``).  ``src/param.F:8985-9060`` computes those six
weights from the levels themselves.  When the lowest three layers are equally
deep the ground weights are exactly 1.875, -1.25 and 0.375, and the same
holds for the highest three layers and the top weights, so the sealed kernel
``gm1_solve3_density_v1`` (``cuda_gm1_forecast``) carries those three numbers
in its text.

A column that starts stretching at the ground (``str_bot = 0``, CM1's common
setting) has other weights.  The sealed kernel is not edited.
:data:`GM1_LEVEL_DENSITY_CUDA_SOURCE` is its text with exactly one change,
made and checked here: the six weights are launch arguments instead of
literals.  Handed 1.875, -1.25 and 0.375 twice it is the sealed operator word
for word, and a run whose two ends are equally spaced never compiles it:
:func:`compile_cuda_gm1_forecast_for_levels` returns the sealed image for such
a column, so every existing grid launches exactly what it launched before.

On flat ground this is the one place the Cartesian lane's step reads the
equal-spacing numbers.  CM1's turbulence and deformation routines use
``cgs``/``cgt`` only in their terrain branches, where the result is multiplied
by the terrain slope; the surface pressure, the parcel ghosts, the nudging
ghosts and the sounding surface already take the column's own weights.  The
face density at the ground multiplies a vertical velocity that is zero there,
so under free slip it reaches no other field; with surface drag it scales the
surface stress (``cuda_surface_drag``).
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass, replace
from typing import Any

from .cm1_vertical_grid import PINNED_FORNBERG_F32
from .cuda_gm1_forecast import (
    GM1_FORECAST_CUDA_SOURCE,
    GM1_FORECAST_CUDA_SOURCE_SHA256,
    GM1_FORECAST_KERNEL_NAMES,
    CompiledGm1ForecastKernels,
    _canonical_arch,
    compile_cuda_gm1_forecast,
)

GM1_LEVEL_DENSITY_ABI = "gm1.cuda-level-density-kernels/v1"
GM1_LEVEL_DENSITY_AUTHORITY = "src/solve3.F:678-679; src/init3d.F:2464-2465; src/param.F:8985-9060"
#: The position of ``gm1_solve3_density_v1`` in the forecast image's kernel tuple.
DENSITY_KERNEL_INDEX = GM1_FORECAST_KERNEL_NAMES.index("gm1_solve3_density_v1")

_SEALED_ABI_LINE = "// gm1.cuda-real-forecast-kernels/v1\n"
_SEALED_SIGNATURE = (
    " const float* rho,const float* c1,const float* c2,float* rr,float* rf,"
    "unsigned int ni,unsigned int nj,unsigned int nk){\n"
)
_LEVEL_SIGNATURE = (
    " const float* rho,const float* c1,const float* c2,float* rr,float* rf,"
    "unsigned int ni,unsigned int nj,unsigned int nk,\n"
    " float cgs1,float cgs2,float cgs3,float cgt1,float cgt2,float cgt3){\n"
)
_SEALED_GROUND = (
    "  if(f==0u)rf[z]=a32(a32(m32(1.875f,rho[h]),m32(-1.25f,rho[(unsigned long long)nj*ni+h])),"
    "m32(0.375f,rho[2ull*nj*ni+h]));\n"
)
_LEVEL_GROUND = (
    "  if(f==0u)rf[z]=a32(a32(m32(cgs1,rho[h]),m32(cgs2,rho[(unsigned long long)nj*ni+h])),"
    "m32(cgs3,rho[2ull*nj*ni+h]));\n"
)
_SEALED_TOP = (
    "  else if(f==nk)rf[z]=a32(a32(m32(1.875f,rho[(unsigned long long)(nk-1u)*nj*ni+h]),"
    "m32(-1.25f,rho[(unsigned long long)(nk-2u)*nj*ni+h])),m32(0.375f,rho[(unsigned long long)(nk-3u)*nj*ni+h]));\n"
)
_LEVEL_TOP = (
    "  else if(f==nk)rf[z]=a32(a32(m32(cgt1,rho[(unsigned long long)(nk-1u)*nj*ni+h]),"
    "m32(cgt2,rho[(unsigned long long)(nk-2u)*nj*ni+h])),m32(cgt3,rho[(unsigned long long)(nk-3u)*nj*ni+h]));\n"
)


def _derive_level_source(sealed: str) -> str:
    if hashlib.sha256(sealed.encode("utf-8")).hexdigest() != GM1_FORECAST_CUDA_SOURCE_SHA256:
        raise RuntimeError("the sealed forecast kernel text differs from its pin")
    marker = 'extern "C" __global__ void gm1_solve3_density_v1(\n'
    following = 'extern "C" __global__ void gm1_sum_initial_mass_v1('
    if sealed.count(marker) != 1 or sealed.count(following) != 1 or not sealed.startswith(_SEALED_ABI_LINE):
        raise RuntimeError("the sealed forecast kernel text no longer has one density kernel to derive from")
    start, end = sealed.index(marker), sealed.index(following)
    head, kernel, tail = sealed[:start], sealed[start:end], sealed[end:]
    for old in (_SEALED_SIGNATURE, _SEALED_GROUND, _SEALED_TOP):
        if kernel.count(old) != 1:
            raise RuntimeError(
                "the sealed density kernel no longer has exactly one signature, one ground "
                "extrapolation and one top extrapolation; the level variant cannot be derived from it"
            )
    kernel = (
        kernel.replace(_SEALED_SIGNATURE, _LEVEL_SIGNATURE, 1)
        .replace(_SEALED_GROUND, _LEVEL_GROUND, 1)
        .replace(_SEALED_TOP, _LEVEL_TOP, 1)
    )
    if "1.875f" in kernel or "-1.25f" in kernel or "0.375f" in kernel:
        raise RuntimeError("the derived density kernel still carries an equal-spacing weight")
    return f"// {GM1_LEVEL_DENSITY_ABI}\n" + head[len(_SEALED_ABI_LINE):] + kernel + tail


GM1_LEVEL_DENSITY_CUDA_SOURCE = _derive_level_source(GM1_FORECAST_CUDA_SOURCE)
GM1_LEVEL_DENSITY_CUDA_SOURCE_SHA256 = (
    "ca90886da3df208a87c4ea8b5830b2df041b7b41322e870ac4cd35f4ec0fe670"
)


def source_pin_is_valid() -> bool:
    actual = hashlib.sha256(GM1_LEVEL_DENSITY_CUDA_SOURCE.encode("utf-8")).hexdigest()
    return actual == GM1_LEVEL_DENSITY_CUDA_SOURCE_SHA256


def boundary_density_weights(levels: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """``(ground, top)`` weights when the sealed kernel's are not this column's, else ``None``.

    An end whose three layers are equally deep keeps 1.875, -1.25 and 0.375,
    the numbers every existing grid has always run with, and the other end
    takes CM1's own ``cgs`` or ``cgt`` for these levels.  ``None`` means both
    ends are equally spaced and the sealed kernel is this column's own.
    """

    if levels is None or levels.fornberg_matches_pinned_kernels():
        return None
    ground = PINNED_FORNBERG_F32 if levels.lowest_layers_equally_spaced else tuple(levels.cgs)
    top = PINNED_FORNBERG_F32 if levels.highest_layers_equally_spaced else tuple(levels.cgt)
    return (
        (float(ground[0]), float(ground[1]), float(ground[2])),
        (float(top[0]), float(top[1]), float(top[2])),
    )


@dataclass(frozen=True, slots=True)
class LevelDensityKernel:
    """The level variant, launched with the sealed kernel's own eight arguments.

    The six weights are bound here, so the two launches of the density kernel
    (the start-up publication and the end of every step) stay as written.
    """

    kernel: Any
    ground: tuple[float, float, float]
    top: tuple[float, float, float]

    def __call__(self, grid: Any, block: Any, args: Any, **keywords: Any) -> Any:
        f32 = importlib.import_module("numpy").float32
        weights = tuple(f32(value) for value in (*self.ground, *self.top))
        return self.kernel(grid, block, tuple(args) + weights, **keywords)

    def record(self) -> dict[str, Any]:
        np = importlib.import_module("numpy")
        return {
            "abi": GM1_LEVEL_DENSITY_ABI,
            "authority": GM1_LEVEL_DENSITY_AUTHORITY,
            "source_sha256": GM1_LEVEL_DENSITY_CUDA_SOURCE_SHA256,
            "ground_weights": list(self.ground),
            "top_weights": list(self.top),
            "ground_weights_f32_bits": [int(np.float32(v).view(np.uint32)) for v in self.ground],
            "top_weights_f32_bits": [int(np.float32(v).view(np.uint32)) for v in self.top],
        }


_COMPILED: dict[tuple[int, str], Any] = {}


def _compile_level_density_kernel() -> Any:
    """The level variant's density kernel, compiled exactly as the sealed image is."""

    if not source_pin_is_valid():
        actual = hashlib.sha256(GM1_LEVEL_DENSITY_CUDA_SOURCE.encode("utf-8")).hexdigest()
        raise RuntimeError(f"derived level-density CUDA source differs from its pin: {actual}")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    key = (int(device.id), arch)
    if key in _COMPILED:
        return _COMPILED[key]
    nvrtc = tuple(compiler._get_nvrtc_version())
    # cuda_gm1_forecast.compile_cuda_gm1_forecast's options, unchanged.
    options = ("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
    if int(nvrtc[0]) >= 12:
        options += ("--device-as-default-execution-space",)
    ptx, _ = compiler._NVRTCProgram(
        GM1_LEVEL_DENSITY_CUDA_SOURCE, "gm1_level_density_v1_ptx.cu", method="ptx"
    ).compile(options + (f"--gpu-architecture=compute_{arch}",))
    cubin, _ = compiler._NVRTCProgram(
        GM1_LEVEL_DENSITY_CUDA_SOURCE, "gm1_level_density_v1_cubin.cu", method="cubin"
    ).compile(options + (f"--gpu-architecture=sm_{arch}",))
    decoded = ptx if isinstance(ptx, str) else bytes(ptx).decode()
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(r"\.entry\s+gm1_solve3_density_v1(?:\s|\()", decoded):
        raise ValueError("compiled PTX lacks the level-density entry point")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled level-density image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernel = module.get_function("gm1_solve3_density_v1")
    _COMPILED[key] = (module, kernel)
    return _COMPILED[key]


def compile_cuda_gm1_forecast_for_levels(levels: Any, *, compile_sealed: Any = None) -> CompiledGm1ForecastKernels:
    """The forecast image for one column.

    ``compile_sealed`` is the caller's own ``compile_cuda_gm1_forecast``, so
    the sealed image is compiled through the name the caller always used.

    A column whose two ends are equally spaced gets exactly what
    :func:`cuda_gm1_forecast.compile_cuda_gm1_forecast` returns, the sealed
    image.  Any other column gets that image with its density kernel replaced
    by the level variant bound to the column's own weights; the other kernels
    are the sealed image's own.
    """

    sealed = (compile_cuda_gm1_forecast if compile_sealed is None else compile_sealed)()
    weights = boundary_density_weights(levels)
    if weights is None:
        return sealed
    module, kernel = _compile_level_density_kernel()
    kernels = list(sealed.kernels)
    kernels[DENSITY_KERNEL_INDEX] = LevelDensityKernel(kernel, weights[0], weights[1])
    return replace(sealed, module_owner=(sealed.module_owner, module), kernels=tuple(kernels))


def cpu_level_density(rho: Any, c1: Any, c2: Any, ground: Any, top: Any) -> tuple[Any, Any]:
    """``rr`` and ``rf`` as ``solve3.F:668-690`` writes them, in binary32, for tests."""

    np = importlib.import_module("numpy")
    f32 = np.float32
    rho = np.asarray(rho, f32)
    nk = rho.shape[0]
    rr = (f32(1.0) / rho).astype(f32)
    rf = np.zeros((nk + 1, *rho.shape[1:]), dtype=f32)
    for k in range(1, nk):
        rf[k] = (np.asarray(c1, f32)[k] * rho[k - 1]).astype(f32) + (np.asarray(c2, f32)[k] * rho[k]).astype(f32)
    g, t = [f32(v) for v in ground], [f32(v) for v in top]
    rf[0] = ((g[0] * rho[0]).astype(f32) + (g[1] * rho[1]).astype(f32)).astype(f32) + (g[2] * rho[2]).astype(f32)
    rf[nk] = (
        ((t[0] * rho[nk - 1]).astype(f32) + (t[1] * rho[nk - 2]).astype(f32)).astype(f32)
        + (t[2] * rho[nk - 3]).astype(f32)
    )
    return rr, rf.astype(f32)


__all__ = (
    "DENSITY_KERNEL_INDEX",
    "GM1_LEVEL_DENSITY_ABI",
    "GM1_LEVEL_DENSITY_AUTHORITY",
    "GM1_LEVEL_DENSITY_CUDA_SOURCE",
    "GM1_LEVEL_DENSITY_CUDA_SOURCE_SHA256",
    "LevelDensityKernel",
    "boundary_density_weights",
    "compile_cuda_gm1_forecast_for_levels",
    "cpu_level_density",
    "source_pin_is_valid",
)
