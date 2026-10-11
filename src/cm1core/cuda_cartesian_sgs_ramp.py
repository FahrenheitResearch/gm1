"""CM1's SGS start-up ramp (``ramp_sgs=1``) for the Cartesian lane's TKE closure.

CM1 r21.1 ramps the subgrid model in over the first ``ramp_time`` seconds
(``src/turb.F:240-262``)::

    turbfac = clamp((rtime-0.0)/ramp_time, 0, 1)
    cme = max(0.001, turbfac*cm0)            where cm0 > cmemin
    ce1 = cme*c_l*c_l*(1/ri_c - 1)
    ce2 = max(0, cme*pi*pi - ce1)
    csm = (cme**3/(ce1+ce2))**0.25

with ``rtime`` the model time at the start of the step (``cm1.F:3724``
prepares step n+1 at the end of step n) and ``ramp_time`` normalized to at
least ``1e-10`` (``src/param.F:750``).  ``cm0`` is ``c_m = 0.10`` everywhere
this lane can run (``src/init3d.F:2619``; the LES-subdomain taper of
``:2640-2650`` has no GM1 key).  Every shipped CM1 deck sets ``ramp_sgs = 1``
and ``ramp_time = 1800``; the Cartesian lane's TKE kernel hard-coded
``cme = 0.10`` (``cuda_sgs_tke.py``), so a translated CM1 deck ran its first
half hour with the full subgrid mixing CM1 withholds.

The sealed kernel (``cuda_sgs_tke.CUDA_SGS_TKE_SOURCE``, pinned by its own
evidence) is not edited.  :data:`CUDA_CARTESIAN_SGS_RAMP_SOURCE` is that text
with exactly one change, made and checked here: ``cme_value`` is a launch
argument instead of the literal ``0.10f``.  Handed ``0.10`` it is the sealed
operator word for word, and the closure uses the sealed image whenever
``turbfac`` is 1, so a run without the ramp, or past it, launches exactly what
it launched before.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .cuda_sgs_tke import (
    ALL_ARRAY_NAMES,
    CUDA_SGS_TKE_KERNEL_NAMES,
    CUDA_SGS_TKE_SOURCE,
    CUDA_SGS_TKE_THREADS,
    _require_device_arrays,
)

_SEALED_SIGNATURE_END = (
    "    const float dz)\n"
    "{\n"
    "    const unsigned long long words = (unsigned long long)ni * nj * (nk - 1u);\n"
)
_RAMPED_SIGNATURE_END = (
    "    const float dz,\n"
    "    const float cme_value)\n"
    "{\n"
    "    const unsigned long long words = (unsigned long long)ni * nj * (nk - 1u);\n"
)
_SEALED_CME_LITERAL = "    const float cme_value = 0.10f;\n"


def _derive_ramped_source(sealed: str) -> str:
    if sealed.count(_SEALED_SIGNATURE_END) != 1 or sealed.count(_SEALED_CME_LITERAL) != 1:
        raise RuntimeError(
            "the sealed SGS/TKE text no longer has exactly one tkekm signature "
            "and one cme literal; the ramp variant cannot be derived from it"
        )
    if sealed.index(_SEALED_SIGNATURE_END) > sealed.index(_SEALED_CME_LITERAL):
        raise RuntimeError("the sealed cme literal is not inside the tkekm kernel")
    return sealed.replace(_SEALED_SIGNATURE_END, _RAMPED_SIGNATURE_END, 1).replace(
        _SEALED_CME_LITERAL, "", 1
    )


CUDA_CARTESIAN_SGS_RAMP_SOURCE = _derive_ramped_source(CUDA_SGS_TKE_SOURCE)
CUDA_CARTESIAN_SGS_RAMP_SOURCE_SHA256 = (
    "c9bb7ad9f6a61a73f554bc81fa4f62c0d966ea10bcca20ae027368175974671c"
)

#: src/constants.F:22,50: ``cmemin`` and ``c_m``; the floor is turb.F:251's.
CM1_CMEMIN = 1.0e-4
CM1_C_M = 0.10
CM1_CME_FLOOR = 0.001


def source_pin_is_valid() -> bool:
    return (
        hashlib.sha256(CUDA_CARTESIAN_SGS_RAMP_SOURCE.encode("utf-8")).hexdigest()
        == CUDA_CARTESIAN_SGS_RAMP_SOURCE_SHA256
    )


def cm1_turbfac(rtime_s: float, ramp_time_s: float) -> Any:
    """``src/turb.F:242-244`` in default REAL, with ``src/param.F:750``'s floor."""

    np = importlib.import_module("numpy")
    rtime = np.float32(rtime_s)
    ramp_time = np.maximum(np.float32(ramp_time_s), np.float32(1.0e-10))
    turbfac = np.float32(np.float32(rtime - np.float32(0.0)) / ramp_time)
    turbfac = np.maximum(np.float32(0.0), turbfac)
    return np.minimum(np.float32(1.0), turbfac)


def cm1_ramped_cme(rtime_s: float, ramp_time_s: float) -> Any:
    """``src/turb.F:251``: ``cme = max(0.001, turbfac*cm0)`` with ``cm0 = c_m``."""

    np = importlib.import_module("numpy")
    cm0 = np.float32(CM1_C_M)
    if not cm0 > np.float32(CM1_CMEMIN):  # turb.F:250
        raise AssertionError("c_m is above cmemin in every CM1 configuration this lane runs")
    return np.maximum(np.float32(CM1_CME_FLOOR), np.float32(cm1_turbfac(rtime_s, ramp_time_s) * cm0))


@dataclass(frozen=True, slots=True)
class CompiledCartesianSgsRamp:
    module_owner: Any
    kernels: tuple[Any, Any]
    cubin_sha256: str
    options: tuple[str, ...]


def compile_cuda_cartesian_sgs_ramp() -> CompiledCartesianSgsRamp:
    if not source_pin_is_valid():
        actual = hashlib.sha256(CUDA_CARTESIAN_SGS_RAMP_SOURCE.encode("utf-8")).hexdigest()
        raise RuntimeError(f"derived SGS ramp CUDA source differs from pin: {actual}")
    from .nvrtc_module import compile_nvrtc_module

    cupy = importlib.import_module("cupy")
    # cuda_sgs_tke._image_options' arithmetic options, through the package's
    # gradual-underflow compile path.
    module = compile_nvrtc_module(
        cupy,
        code=CUDA_CARTESIAN_SGS_RAMP_SOURCE,
        filename="gm1_cartesian_sgs_ramp.cu",
        options=("--std=c++14", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"),
        name_expressions=CUDA_SGS_TKE_KERNEL_NAMES,
    )
    return CompiledCartesianSgsRamp(
        module,
        tuple(module.get_function(name) for name in CUDA_SGS_TKE_KERNEL_NAMES),  # type: ignore[arg-type]
        hashlib.sha256(module.cubin).hexdigest(),
        tuple(module.options),
    )


def launch_cuda_cartesian_sgs_ramp(
    arrays: Mapping[str, Any],
    *,
    cme: Any,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    stream: Any,
    compiled: CompiledCartesianSgsRamp,
) -> dict[str, Any]:
    """``cuda_sgs_tke.launch_cuda_sgs_tke`` with CM1's ramped ``cme``."""

    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    interior = ni * nj * (nk - 1)
    horizontal = ni * nj
    u32, f32 = cupy.uint32, cupy.float32
    compiled.kernels[0](
        ((interior + CUDA_SGS_TKE_THREADS - 1) // CUDA_SGS_TKE_THREADS, 1, 1),
        (CUDA_SGS_TKE_THREADS, 1, 1),
        tuple(a[name] for name in ALL_ARRAY_NAMES)
        + (u32(ni), u32(nj), u32(nk), f32(dx), f32(dy), f32(dz), f32(cme)),
        stream=stream,
    )
    compiled.kernels[1](
        ((horizontal + CUDA_SGS_TKE_THREADS - 1) // CUDA_SGS_TKE_THREADS, 1, 1),
        (CUDA_SGS_TKE_THREADS, 1, 1),
        (a["tkea"], a["kmh"], a["kmv"], a["khh"], a["khv"], u32(ni), u32(nj)),
        stream=stream,
    )
    return {
        "source_sha256": CUDA_CARTESIAN_SGS_RAMP_SOURCE_SHA256,
        "cubin_sha256": compiled.cubin_sha256,
        "cme_raw_bits": int(importlib.import_module("numpy").float32(cme).view("uint32")),
    }


__all__ = [
    "CM1_CME_FLOOR",
    "CM1_CMEMIN",
    "CM1_C_M",
    "CUDA_CARTESIAN_SGS_RAMP_SOURCE",
    "CUDA_CARTESIAN_SGS_RAMP_SOURCE_SHA256",
    "CompiledCartesianSgsRamp",
    "cm1_ramped_cme",
    "cm1_turbfac",
    "compile_cuda_cartesian_sgs_ramp",
    "launch_cuda_cartesian_sgs_ramp",
    "source_pin_is_valid",
]
