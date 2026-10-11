"""NVRTC images that keep binary32 gradual underflow, loaded without CuPy's compile cache.

``cupy.RawModule``/``cupy.RawKernel`` (and CuPy's own elementwise and
reduction kernels) compile through ``cupy.cuda.compiler._compile_with_cache_cuda``,
which appends ``-ftz=true`` after the caller's options (CuPy 14.2.0,
``cupy/cuda/compiler.py:667``; CuPy's own ``cupy.nextafter`` documentation
says so: "CuPy sets ``-ftz=true``").  NVRTC's user guide (supported compile
options) documents ``--ftz`` as "when performing single-precision
floating-point operations, flush denormal values to zero or preserve denormal
values", default ``false``, and ``--use_fast_math`` as implying
``--ftz=true``; it does not say which occurrence of a repeated option wins.
Measured, the last one does: a ``--ftz=false`` handed to ``RawModule`` is
followed by CuPy's ``-ftz=true`` and the device writes the flushed word
(``evidence/oracle/nvrtc-gradual-underflow``, route ``R2``).  The hardware is
not the cause: the same text compiled by NVRTC directly with ``--ftz=false``
keeps the sub-normal on the same card, and its SASS carries no flush-to-zero
instruction modifier.  That is what NVIDIA documents too -- the compiler
"appends a modifier flag to the instruction" for adds, multiplies and
multiply-adds ("CUDA Pro Tip: Flush Denormals with Confidence") -- so there
is no run-time, driver or environment switch to reach for: the mode is fixed
when the kernel is compiled.

CM1 is compiled by gfortran for the host, where a binary32 product or a
double-to-float conversion whose exact value is sub-normal keeps that
sub-normal (``CM1-ARITHMETIC-CONTRACT.md``: flush-to-zero and
denormals-are-zero are off).  A kernel that ran through ``RawModule`` instead
wrote a zero of the same sign there, and compared, clamped and multiplied a
sub-normal operand as zero.

:func:`compile_nvrtc_module` is the one way a GM1 production kernel text is
built outside the image compile functions that already call NVRTC
themselves: one ``cupy.cuda.compiler._NVRTCProgram`` compiled to a cubin
with ``--ftz=false`` last among the arithmetic options, loaded through
``cupy.cuda.function.Module``.  The returned module answers
``get_function(name)`` as ``RawModule`` does.  Nothing is cached across
calls: each call is one NVRTC program, as it is for every image compile
function in this package.

What this does not reach: CuPy's own array operations (``a * b``,
``cupy.where``, ``cupy.sum``) are still compiled by CuPy with ``-ftz=true``,
and so are the ArWen (``gpuwm``) kernels, which GM1 does not compile.

``CUPY_COMPILE_WITH_PTX`` does not apply to a GM1 image.  CuPy reads it once
into ``cupy.cuda.compiler._use_ptx`` and its own compile path answers by
handing NVRTC ``-arch=compute_*`` and keeping the PTX for the driver to JIT
(``_get_arch_for_options_for_nvrtc``).  Every GM1 image is a cubin compiled
for ``sm_<arch>`` and loaded as one, here and in the image compile functions
that call NVRTC themselves, so setting that variable changes nothing GM1
runs.  The arithmetic is the reason: the point of these images is the exact
option tuple NVRTC receives, and a PTX route would put the driver's JIT
between that tuple and the SASS.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

NVRTC_MODULE_SCHEMA = "gm1.nvrtc-module/v1"

#: The option that keeps binary32 sub-normals.  NVRTC's default, stated
#: explicitly because CuPy's own compile path overrides the default.
GRADUAL_UNDERFLOW_OPTION = "--ftz=false"
DEVICE_DEFAULT_OPTION = "--device-as-default-execution-space"

#: Spellings that turn on flush-to-zero (``--ftz``/``-ftz`` with no value or
#: ``=true``) or fast math, which implies it (NVRTC user guide).
_FLUSHING = re.compile(r"^--?(ftz(=true)?|use_fast_math|fast-math)$")
_FTZ = re.compile(r"^--?ftz(=.*)?$")
_ARCH = re.compile(r"^(--gpu-architecture=|-arch=)")


class NvrtcModuleError(RuntimeError):
    pass


def gradual_underflow_options(options: Iterable[str]) -> tuple[str, ...]:
    """``options`` with ``--ftz=false`` in it exactly once.

    The caller's order is kept; ``--ftz=false`` is appended when absent.  A
    tuple that asks for flush-to-zero or fast math is refused rather than
    corrected: the caller said something this package's arithmetic contract
    does not allow.  An architecture option is refused too; the target is the
    active device's and is added by :func:`nvrtc_image_options`.
    """

    options = tuple(options)
    if not all(isinstance(option, str) and option for option in options):
        raise NvrtcModuleError(f"NVRTC options must be non-empty strings: {options!r}")
    flushing = [option for option in options if _FLUSHING.match(option)]
    if flushing:
        raise NvrtcModuleError(f"NVRTC options turn on flush-to-zero: {flushing}")
    other = [option for option in options if _FTZ.match(option) and option != GRADUAL_UNDERFLOW_OPTION]
    if other:
        raise NvrtcModuleError(f"unrecognised ftz option spelling: {other}")
    arch = [option for option in options if _ARCH.match(option)]
    if arch:
        raise NvrtcModuleError(f"the target architecture is the active device's, not {arch}")
    kept: list[str] = []
    for option in options:
        if option not in kept or option != GRADUAL_UNDERFLOW_OPTION:
            kept.append(option)
    if GRADUAL_UNDERFLOW_OPTION not in kept:
        kept.append(GRADUAL_UNDERFLOW_OPTION)
    return tuple(kept)


def _submodule(cupy: Any, name: str) -> Any:
    cuda = getattr(cupy, "cuda", None)
    found = getattr(cuda, name, None) if cuda is not None else None
    if found is not None:
        return found
    return importlib.import_module(f"{cupy.__name__}.cuda.{name}")


def _canonical_arch(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = f"{value[0]}{value[1]}"
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    text = text.strip().lower().replace(".", "")
    if not re.fullmatch(r"[0-9]+", text):
        raise NvrtcModuleError(f"CUDA architecture must be digits, got {value!r}")
    return text


def nvrtc_target_arch(cupy: Any) -> str:
    """The compute capability every GM1 kernel image is compiled for, as digits.

    One architecture rule for the package.  ``cupy.cuda.compiler._get_arch()``
    is the active device's compute capability clamped to the highest this
    NVRTC accepts (``_get_max_compute_capability``), which is the target CuPy
    itself compiles a ``RawModule`` for, so a GM1 image and a CuPy one name
    the same architecture.

    The image compile functions that call ``_NVRTCProgram`` themselves
    (``cuda_tke_evolution``, ``cuda_sgs_stress_tendency``,
    ``cuda_surface_drag``, ``cuda_getcape``) take their ``sm_``/``compute_``
    target from here too.  Building it from
    ``cupy.cuda.runtime.getDeviceProperties`` instead, as they once did, is
    unclamped: on a card newer than the installed NVRTC accepts those
    functions would ask for an architecture NVRTC rejects while
    :func:`compile_nvrtc_module` compiled the clamped one, and the package
    would carry two answers to the same question.
    """

    return _canonical_arch(_submodule(cupy, "compiler")._get_arch())


def nvrtc_image_options(cupy: Any, options: Iterable[str]) -> tuple[str, ...]:
    """The exact tuple NVRTC receives for ``options`` on the active device."""

    compiler = _submodule(cupy, "compiler")
    full = gradual_underflow_options(options)
    major = int(tuple(compiler._get_nvrtc_version())[0])
    if major >= 12 and DEVICE_DEFAULT_OPTION not in full:
        full += (DEVICE_DEFAULT_OPTION,)
    return full + (f"--gpu-architecture=sm_{nvrtc_target_arch(cupy)}",)


@dataclass(frozen=True, slots=True)
class NvrtcModule:
    """One loaded cubin; ``get_function`` answers as ``cupy.RawModule``'s does."""

    image: Any
    cubin: bytes
    options: tuple[str, ...]
    lowered_names: Mapping[str, str]
    source_sha256: str
    filename: str

    def get_function(self, name: str) -> Any:
        return self.image.get_function(self.lowered_names.get(name, name))


def compile_nvrtc_module(
    cupy: Any,
    *,
    code: str,
    options: Iterable[str],
    name_expressions: Iterable[str] = (),
    filename: str = "gm1_kernel.cu",
) -> NvrtcModule:
    """Compile ``code`` with gradual underflow and load it on the current device."""

    if not isinstance(code, str) or not code:
        raise NvrtcModuleError("kernel code must be a non-empty string")
    compiler = _submodule(cupy, "compiler")
    function = _submodule(cupy, "function")
    full = nvrtc_image_options(cupy, options)
    names = tuple(name_expressions or ())
    program = compiler._NVRTCProgram(code, filename, name_expressions=names or None, method="cubin")
    cubin, mapping = program.compile(full)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise NvrtcModuleError(f"{filename}: NVRTC did not return an ELF64 CUDA image")
    image = function.Module()
    image.load(cubin)
    lowered = {}
    for key, value in dict(mapping or {}).items():
        lowered[str(key)] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return NvrtcModule(
        image=image,
        cubin=cubin,
        options=full,
        lowered_names=MappingProxyType(lowered),
        source_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
        filename=filename,
    )


__all__ = [
    "DEVICE_DEFAULT_OPTION",
    "GRADUAL_UNDERFLOW_OPTION",
    "NVRTC_MODULE_SCHEMA",
    "NvrtcModule",
    "NvrtcModuleError",
    "compile_nvrtc_module",
    "gradual_underflow_options",
    "nvrtc_image_options",
    "nvrtc_target_arch",
]
