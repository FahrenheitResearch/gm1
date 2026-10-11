"""CM1 r21.1 shared vertical tail for an explicitly declared level set.

``param.F:8162-8209`` is the arithmetic every ``stretch_z`` branch runs once
its ``nk+1`` physical w faces exist.  ``cuda_stretch_z1`` fuses that tail into
the ``stretch_z=1`` analytic generator, which is correct for branches 0 and 1
because those generate their own faces.  ``stretch_z=3`` (``param.F:8005-8064``)
and ``stretch_z=4`` (``param.F:8072-8132``) instead read their levels, so this
module is the tail on its own: the caller hands over the faces and the device
produces ``zf``/``zh``/``sigmaf``/``sigma``/``mh``/``mf``/``rmh``/``rmf``.

The faces are configuration, not an atmospheric field.  Nothing here
interpolates, smooths or rescales model data.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any

from .cuda_sound3_periodic import _canonical_arch, _image_options
from .cuda_stretch_z1 import DeviceStretchZ1Grid, StretchZ1Derived


CUDA_VERTICAL_LEVELS_SCHEMA = "gm1.cuda-vertical-levels-tail/v1"
CUDA_VERTICAL_LEVELS_ABI = "gm1.cuda-vertical-levels-tail-kernel/v1"
CM1_VERTICAL_LEVELS_CLAIM_LIMIT = (
    "CM1 r21.1 param.F:8162-8209 vertical tail over caller-declared w levels "
    "for a flat Cartesian grid; no terrain composition, horizontal stretch, "
    "moving domain, coupled forecast, storm, tornado, product or performance "
    "claim"
)
SOURCE_ANCHORS = {
    "user_scalar_levels": "src/param.F:8005-8064",
    "user_w_levels": "src/param.F:8072-8132",
    "shared_tail": "src/param.F:8162-8182",
    "inverse_metrics": "src/param.F:8189-8209",
    "positive_spacing_check": "src/param.F:8212-8224",
}

# param.F:8162-8209, transcribed with the same rounded-operation helpers the
# stretch_z=1 kernel uses.  ``face`` holds the nk+1 physical w levels, CM1
# k=1..nk+1, in element 0..nk.
CUDA_VERTICAL_LEVELS_SOURCE = r"""// gm1.cuda-vertical-levels-tail-kernel/v1
static_assert(sizeof(float)==4,"float32 required");
__device__ __forceinline__ float a32(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float s32(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float m32(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float d32(float a,float b){return __fdiv_rn(a,b);}
extern "C" __global__ void cm1_vertical_tail_v1(const float* face,float* zf,float* zh,float* sigmaf,float* sigma,float* mh,float* mf,float* rmh,float* rmf,unsigned nk,float dz,int* bad){if(blockIdx.x||threadIdx.x)return;for(unsigned k=1u;k<=nk+1u;++k){zf[k]=face[k-1u];}zf[0]=-zf[2];zf[nk+2u]=a32(zf[nk+1u],s32(zf[nk+1u],zf[nk]));for(unsigned k=0u;k<nk+3u;++k)sigmaf[k]=zf[k];for(unsigned k=0u;k<=nk+1u;++k){zh[k]=m32(0.5f,a32(zf[k+1u],zf[k]));mh[k]=d32(dz,s32(zf[k+1u],zf[k]));}zh[0]=-zh[1];zh[nk+1u]=a32(zh[nk],m32(2.0f,s32(zf[nk+1u],zh[nk])));for(unsigned k=1u;k<=nk+1u;++k)mf[k]=d32(dz,s32(zh[k],zh[k-1u]));mf[0]=mf[1];mf[nk+2u]=mf[nk+1u];for(unsigned k=0u;k<=nk+1u;++k){sigma[k]=m32(0.5f,a32(sigmaf[k],sigmaf[k+1u]));rmh[k]=d32(1.0f,mh[k]);}for(unsigned k=0u;k<=nk+2u;++k)rmf[k]=d32(1.0f,mf[k]);int flag=0;for(unsigned k=1u;k<=nk;++k){if(!(zf[k+1u]>zf[k]))flag=1;}bad[0]=flag;}
"""
CUDA_VERTICAL_LEVELS_SOURCE_BYTES = CUDA_VERTICAL_LEVELS_SOURCE.encode("utf-8")
CUDA_VERTICAL_LEVELS_SOURCE_SHA256 = hashlib.sha256(
    CUDA_VERTICAL_LEVELS_SOURCE_BYTES
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledVerticalLevels:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CudaVerticalLevelsLaunch:
    ready_event: Any
    compiled: CompiledVerticalLevels
    grid: DeviceStretchZ1Grid
    receipt: dict[str, Any]


def compile_cuda_vertical_levels() -> CompiledVerticalLevels:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_VERTICAL_LEVELS_SOURCE, "cm1_vertical_tail_ptx.cu", method="ptx"
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_VERTICAL_LEVELS_SOURCE, "cm1_vertical_tail_cubin.cu", method="cubin"
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(r"\.entry\s+cm1_vertical_tail_v1(?:\s|\()", ptx.decode("utf-8")):
        raise ValueError("compiled PTX lacks the vertical-tail entry point")
    module = function.Module()
    module.load(cubin)
    kernel = module.get_function("cm1_vertical_tail_v1")
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledVerticalLevels(module, kernel, ptx, cubin, arch, identity)


def launch_cuda_vertical_levels(
    levels: Any,
    *,
    stream: Any | None = None,
    compiled: CompiledVerticalLevels | None = None,
) -> CudaVerticalLevelsLaunch:
    """Enqueue ``param.F:8162-8209`` over ``levels``' own physical faces.

    ``levels`` is a :class:`cm1core.cm1_vertical_grid.VerticalLevels`.  The
    device result is then held to that host transcription raw word for raw
    word: the faces are the only input, so any difference is a defect in one
    of the two transcriptions rather than a property of the grid.
    """

    from .cm1_vertical_grid import VerticalLevels

    if not isinstance(levels, VerticalLevels):
        raise TypeError("levels must be cm1_vertical_grid.VerticalLevels")
    cupy = importlib.import_module("cupy")
    numpy = importlib.import_module("numpy")
    compiled = compile_cuda_vertical_levels() if compiled is None else compiled
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    nk = levels.nk
    faces = numpy.ascontiguousarray(
        numpy.asarray(levels.zf, dtype=numpy.float32)[1 : nk + 2]
    )
    with stream:
        face_device = cupy.asarray(faces)
        arrays = {
            "zf": cupy.empty(nk + 3, dtype=cupy.float32),
            "zh": cupy.empty(nk + 2, dtype=cupy.float32),
            "sigmaf": cupy.empty(nk + 3, dtype=cupy.float32),
            "sigma": cupy.empty(nk + 2, dtype=cupy.float32),
            "mh": cupy.empty(nk + 2, dtype=cupy.float32),
            "mf": cupy.empty(nk + 3, dtype=cupy.float32),
            "rmh": cupy.empty(nk + 2, dtype=cupy.float32),
            "rmf": cupy.empty(nk + 3, dtype=cupy.float32),
        }
        bad = cupy.zeros(1, dtype=cupy.int32)
        compiled.kernel(
            (1, 1, 1),
            (1, 1, 1),
            (
                face_device,
                *(
                    arrays[name]
                    for name in ("zf", "zh", "sigmaf", "sigma", "mh", "mf", "rmh", "rmf")
                ),
                cupy.uint32(nk),
                cupy.float32(levels.dz_reference_m),
                bad,
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    event.synchronize()
    if int(bad.get()[0]) != 0:
        raise ValueError(
            "param.F:8212-8224 dummy check: a declared layer has non-positive depth"
        )
    for name, host in (
        ("zf", levels.zf), ("zh", levels.zh), ("sigmaf", levels.sigmaf),
        ("sigma", levels.sigma), ("mh", levels.mh), ("mf", levels.mf),
        ("rmh", levels.rmh), ("rmf", levels.rmf),
    ):
        if bytes(arrays[name].get().tobytes()) != bytes(
            numpy.asarray(host, dtype=numpy.float32).tobytes()
        ):
            raise ValueError(
                f"device vertical tail {name} differs from the param.F:8162-8209 "
                "host transcription"
            )
    derived = StretchZ1Derived(0, 0, 0, float(levels.dz_reference_m), 0.0, 0.0)
    grid = DeviceStretchZ1Grid(
        arrays["zf"], arrays["zh"], arrays["sigmaf"], arrays["sigma"],
        arrays["mh"], arrays["mf"], arrays["rmh"], arrays["rmf"], derived,
    )
    receipt = {
        "schema": CUDA_VERTICAL_LEVELS_SCHEMA,
        "stretch_z": levels.stretch_z,
        "nk": nk,
        "dz_reference_m": levels.dz_reference_m,
        "ztop_m": levels.ztop_m,
        "levels_sha256": levels.sha256,
        "kernel": {
            "source_sha256": CUDA_VERTICAL_LEVELS_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
        "source_anchors": dict(SOURCE_ANCHORS),
        "claim_limit": CM1_VERTICAL_LEVELS_CLAIM_LIMIT,
    }
    return CudaVerticalLevelsLaunch(event, compiled, grid, receipt)


__all__ = (
    "CM1_VERTICAL_LEVELS_CLAIM_LIMIT",
    "CUDA_VERTICAL_LEVELS_ABI",
    "CUDA_VERTICAL_LEVELS_SCHEMA",
    "CUDA_VERTICAL_LEVELS_SOURCE",
    "CUDA_VERTICAL_LEVELS_SOURCE_BYTES",
    "CUDA_VERTICAL_LEVELS_SOURCE_SHA256",
    "CompiledVerticalLevels",
    "CudaVerticalLevelsLaunch",
    "SOURCE_ANCHORS",
    "compile_cuda_vertical_levels",
    "launch_cuda_vertical_levels",
)
