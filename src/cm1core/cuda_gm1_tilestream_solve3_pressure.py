"""Tile-local physical ``ppx += pp3d`` from CM1 ``solve3``.

This is deliberately operation A of a split solve3 lifecycle.  For the
selected psolver=3 branch, CM1 first completes physical ``ppx`` with the
current pressure perturbation (``solve3.F:270-283``).  Boundary completion,
MPI exchange, rho boundary work, held-state recapture, momentum/TKE lifecycle,
current-to-A publication, and reciprocal-density reconstruction all occur
later and are not admitted here.

The operation is pointwise and therefore tile-streamable without halo reads.
Every physical mass cell has one owner from :class:`Gm1TilePlan`; edge padding
is transport-only and is never scattered.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TileSpec
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SCHEMA = (
    "gm1.cuda-tilestream-solve3-physical-pressure/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_ABI = (
    "gm1.cuda-tilestream-solve3-physical-ppx-f32/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_KERNEL = "gm1_tiled_solve3_physical_ppx_f32_v1"
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
AUTHORITY_ANCHORS = ("src/solve3.F:270-283",)
GM1_ACCEPTED_REAL_POST_WSM6_RECEIPT_SHA256 = (
    "5cfb967d886eac2bb6e8b6b34cc3da50d3e73e9160f1021b390159046a25d7cc"
)
GM1_ACCEPTED_REAL_POST_WSM6_SEMANTIC_SHA256 = (
    "babbf6dcc8a7d786743277c115dbb537a0762312c3596b884d826c75251c4f1b"
)
GM1_ACCEPTED_REAL_POST_WSM6_OUTPUT_SHA256 = (
    "232956d736a36c81c2013e1eb82071de9cb57c8c68f0fa1daec149dbbc4cb304"
)
GM1_ACCEPTED_REAL_PP3D_SHA256 = (
    "c8312d89a275ef3c067aec46cac22a9be45acdfbc43d2e351a0b24d536fe3013"
)
GM1_SOURCE_INITIALIZED_PHYSICAL_PPX_SHA256 = (
    "54c1a30d57a4ad0eb87655d4bbf2dffcdb8d66a79cd703b9ee7689223fbc2ab9"
)
GM1_AUXILIARY_MANIFEST_SHA256 = (
    "24764f50afc5807c37266696f9a2ea5c5cc0989819e2fa238dd9b11df4b1e8e4"
)
GM1_AUXILIARY_TERRAIN_OPEN_SHA256 = (
    "62364cf75c8cea33147f3b95ce691a783089ea3f015bef116f1d0c71df5efdb3"
)
GM1_TILESTREAM_SOLVE3_PRESSURE_CLAIM_LIMIT = (
    "CM1 r21.1 selected psolver=3 physical ppx=f32(ppx+pp3d) only; no ppx "
    "boundary completion, open-history mutation, MPI exchange, rho boundary, "
    "held recapture, momentum/TKE lifecycle, current-to-A publication, "
    "reciprocal-density rebuild, global psolver, full solve3/RK, or five-metre "
    "meteorological product claim"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def tiled_solve3_pressure_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    """Exact operation-A host/device inventory in stable source order."""

    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    mass = (nk, tile_ny, tile_nx)
    return MappingProxyType({"ppx": mass, "pp3d": mass})


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3PressureBinding:
    """Writable physical ppx and immutable accepted post-WSM6 pp3d."""

    ppx: Any
    pp3d: Any
    nx: int
    ny: int
    nk: int
    source_identity_sha256: str
    post_wsm6_receipt_sha256: str
    post_wsm6_semantic_sha256: str
    post_wsm6_output_sha256: str
    compact_validation: bool

    @classmethod
    def create(
        cls,
        *,
        ppx: Any,
        pp3d: Any,
        source_identity_sha256: str,
        post_wsm6_receipt_sha256: str,
        post_wsm6_semantic_sha256: str,
        post_wsm6_output_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledSolve3PressureBinding":
        np = importlib.import_module("numpy")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        if (
            not isinstance(ppx, np.ndarray)
            or ppx.dtype != np.dtype("<f4")
            or ppx.ndim != 3
            or not ppx.flags.c_contiguous
            or not ppx.flags.writeable
        ):
            raise ValueError("ppx must be writable host C-<f4 [nk,ny,nx]")
        nk, ny, nx = map(int, ppx.shape)
        if (
            not isinstance(pp3d, np.ndarray)
            or pp3d.dtype != np.dtype("<f4")
            or pp3d.shape != ppx.shape
            or not pp3d.flags.c_contiguous
        ):
            raise ValueError(f"pp3d must be host C-<f4 {ppx.shape}")
        if np.shares_memory(ppx, pp3d):
            raise ValueError("ppx must not alias read-only pp3d")
        if not bool(np.all(np.isfinite(ppx))) or not bool(np.all(np.isfinite(pp3d))):
            raise ValueError("solve3 pressure source contains non-finite words")
        source_identity_sha256 = _require_sha256(
            "source_identity_sha256", source_identity_sha256
        )
        post_wsm6_receipt_sha256 = _require_sha256(
            "post_wsm6_receipt_sha256", post_wsm6_receipt_sha256
        )
        post_wsm6_semantic_sha256 = _require_sha256(
            "post_wsm6_semantic_sha256", post_wsm6_semantic_sha256
        )
        post_wsm6_output_sha256 = _require_sha256(
            "post_wsm6_output_sha256", post_wsm6_output_sha256
        )
        if allow_compact_validation:
            if min(nx, ny) < 7:
                raise ValueError(
                    "compact pressure domain must retain the CM1 open band"
                )
        else:
            if (nx, ny, nk) != (300, 300, 75):
                raise ValueError("real solve3 pressure source must be [75,300,300]")
            if post_wsm6_receipt_sha256 != GM1_ACCEPTED_REAL_POST_WSM6_RECEIPT_SHA256:
                raise ValueError("post-WSM6 receipt differs from accepted real-domain bytes")
            if post_wsm6_semantic_sha256 != GM1_ACCEPTED_REAL_POST_WSM6_SEMANTIC_SHA256:
                raise ValueError("post-WSM6 semantic identity changed")
            if post_wsm6_output_sha256 != GM1_ACCEPTED_REAL_POST_WSM6_OUTPUT_SHA256:
                raise ValueError("post-WSM6 output identity changed")
            if (
                hashlib.sha256(pp3d.tobytes(order="C")).hexdigest()
                != GM1_ACCEPTED_REAL_PP3D_SHA256
            ):
                raise ValueError("pp3d differs from accepted post-WSM6 physical words")
            if (
                hashlib.sha256(ppx.tobytes(order="C")).hexdigest()
                != GM1_SOURCE_INITIALIZED_PHYSICAL_PPX_SHA256
            ):
                raise ValueError(
                    "ppx is not the source-initialized physical getset view"
                )
        return cls(
            ppx,
            pp3d,
            nx,
            ny,
            nk,
            source_identity_sha256,
            post_wsm6_receipt_sha256,
            post_wsm6_semantic_sha256,
            post_wsm6_output_sha256,
            allow_compact_validation,
        )


def fill_gm1_tiled_solve3_pressure_source(
    binding: Gm1TiledSolve3PressureBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    """Fill only a tile's physical owned rectangle; padding stays +0."""

    if not isinstance(binding, Gm1TiledSolve3PressureBinding):
        raise TypeError("binding must be Gm1TiledSolve3PressureBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec does not cover the pressure source domain")
    np = importlib.import_module("numpy")
    shapes = tiled_solve3_pressure_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if tuple(target) != tuple(shapes):
        raise ValueError("solve3 pressure target inventory or order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    for name, shape in shapes.items():
        value = target[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"{name} tile buffer must be writable C-<f4 {shape}")
        value.fill(np.float32(0.0))
        value[:, :active_ny, :active_nx] = getattr(binding, name)[:, y0:y1, x0:x1]
    if np.shares_memory(target["ppx"], target["pp3d"]):
        raise ValueError("tile ppx and pp3d buffers must not alias")
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "active_shape": [binding.nk, active_ny, active_nx],
            "padding_initialized": "+0 and excluded from scatter",
        }
    )


def scatter_gm1_tiled_solve3_pressure_output(
    binding: Gm1TiledSolve3PressureBinding,
    spec: Gm1TileSpec,
    ppx_host: Any,
) -> Mapping[str, object]:
    """Scatter exactly one physical ownership rectangle back to ppx."""

    if not isinstance(binding, Gm1TiledSolve3PressureBinding):
        raise TypeError("binding must be Gm1TiledSolve3PressureBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec does not cover the pressure source domain")
    np = importlib.import_module("numpy")
    expected = (binding.nk, spec.tile_ny, spec.tile_nx)
    if (
        not isinstance(ppx_host, np.ndarray)
        or ppx_host.dtype != np.dtype("<f4")
        or ppx_host.shape != expected
        or not ppx_host.flags.c_contiguous
    ):
        raise ValueError(f"ppx host drain must be C-<f4 {expected}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    binding.ppx[:, y0:y1, x0:x1] = ppx_host[:, :active_ny, :active_nx]
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "words_scattered": binding.nk * active_ny * active_nx,
            "padding_scattered": False,
        }
    )


def cpu_gm1_tiled_solve3_pressure_inplace(
    ppx: Any, pp3d: Any, *, active_nx: int, active_ny: int
) -> None:
    """Literal f32 operation-A oracle, excluding transport padding."""

    np = importlib.import_module("numpy")
    if (
        not isinstance(ppx, np.ndarray)
        or not isinstance(pp3d, np.ndarray)
        or ppx.dtype != np.dtype("<f4")
        or pp3d.dtype != np.dtype("<f4")
        or ppx.shape != pp3d.shape
        or ppx.ndim != 3
        or not ppx.flags.c_contiguous
        or not pp3d.flags.c_contiguous
        or not ppx.flags.writeable
    ):
        raise ValueError("ppx/pp3d must be non-aliasing C-<f4 arrays of one shape")
    if np.shares_memory(ppx, pp3d):
        raise ValueError("ppx must not alias pp3d")
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    if active_nx > ppx.shape[2] or active_ny > ppx.shape[1]:
        raise ValueError("active pressure rectangle exceeds its tile buffer")
    ppx[:, :active_ny, :active_nx] = np.add(
        ppx[:, :active_ny, :active_nx],
        pp3d[:, :active_ny, :active_nx],
        dtype=np.float32,
    )


CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE = r"""// gm1.cuda-tilestream-solve3-physical-ppx-f32/v1
extern "C" __global__ void gm1_tiled_solve3_physical_ppx_f32_v1(
    const unsigned int active_nx,
    const unsigned int active_ny,
    const unsigned int nk,
    const unsigned int tile_nx,
    const unsigned int tile_ny,
    float * const __restrict__ ppx,
    const float * const __restrict__ pp3d
) {
    const unsigned long long n =
        (unsigned long long)nk * active_ny * active_nx;
    unsigned long long q =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n) return;
    const unsigned int i = (unsigned int)(q % active_nx);
    q /= active_nx;
    const unsigned int j = (unsigned int)(q % active_ny);
    const unsigned int k = (unsigned int)(q / active_ny);
    const unsigned long long o =
        ((unsigned long long)k * tile_nx * tile_ny) +
        ((unsigned long long)j * tile_nx) + i;
    ppx[o] = __fadd_rn(ppx[o], pp3d[o]);
}
"""
CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE_BYTES = (
    CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE.encode("ascii")
)
CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE_BYTES
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledSolve3Pressure:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_solve3_pressure() -> CompiledGm1TiledSolve3Pressure:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    common = ("--std=c++14", "--fmad=false")
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE,
        "gm1_tiled_solve3_physical_pressure_ptx.cu",
        method="ptx",
    ).compile(tuple(dict.fromkeys((*ptx_options, *common))))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE,
        "gm1_tiled_solve3_physical_pressure_cubin.cu",
        method="cubin",
    ).compile(tuple(dict.fromkeys((*cubin_options, *common))))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(
        rf"\.entry\s+{CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_KERNEL}(?:\s|\()",
        ptx.decode("utf-8"),
    ):
        raise ValueError("compiled PTX lacks the solve3 physical-pressure entry")
    module = function.Module()
    module.load(cubin)
    props = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = props.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    runtime = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1TiledSolve3Pressure(
        module,
        module.get_function(CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_KERNEL),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve3PressureLaunch:
    ready_event: Any
    compiled: CompiledGm1TiledSolve3Pressure
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_solve3_pressure(
    arrays: Mapping[str, Any],
    spec: Gm1TileSpec,
    *,
    nk: int,
    source_identity_sha256: str,
    post_wsm6_receipt_sha256: str,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve3Pressure | None = None,
) -> CudaGm1TiledSolve3PressureLaunch:
    """Enqueue operation A on the exact tile stream; no host synchronization."""

    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    nk = _positive_int("nk", nk)
    source_identity_sha256 = _require_sha256(
        "source_identity_sha256", source_identity_sha256
    )
    post_wsm6_receipt_sha256 = _require_sha256(
        "post_wsm6_receipt_sha256", post_wsm6_receipt_sha256
    )
    cupy = importlib.import_module("cupy")
    shapes = tiled_solve3_pressure_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA solve3 pressure inventory or order changed")
    checked: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 {shape}")
        checked[name] = value
    if int(checked["ppx"].data.ptr) == int(checked["pp3d"].data.ptr):
        raise ValueError("CUDA ppx must not alias pp3d")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    threads = 256
    words = nk * active_ny * active_nx
    blocks = ((words + threads - 1) // threads, 1, 1)
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = (
        compile_cuda_gm1_tiled_solve3_pressure() if compiled is None else compiled
    )
    with stream:
        compiled.kernel(
            blocks,
            (threads, 1, 1),
            (
                cupy.uint32(active_nx),
                cupy.uint32(active_ny),
                cupy.uint32(nk),
                cupy.uint32(spec.tile_nx),
                cupy.uint32(spec.tile_ny),
                checked["ppx"],
                checked["pp3d"],
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_ABI,
        "status": "ENQUEUED_PHYSICAL_PPX_COMPLETION_ONLY",
        "operation": "ppx=f32(ppx+pp3d)",
        "tile": spec.index,
        "owned_yx": [y0, y1, x0, x1],
        "active_shape": [nk, active_ny, active_nx],
        "source_identity_sha256": source_identity_sha256,
        "post_wsm6_receipt_sha256": post_wsm6_receipt_sha256,
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
        "pointers": {
            "ppx": int(checked["ppx"].data.ptr),
            "pp3d": int(checked["pp3d"].data.ptr),
        },
        "kernel": {
            "source_sha256": CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
            "fmad": False,
            "addition": "__fadd_rn",
        },
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "refusals": {
            "ppx_boundary_completion": True,
            "open_history_mutation": True,
            "mpi_exchange": True,
            "rho_boundary": True,
            "held_recapture": True,
            "momentum_or_tke_lifecycle": True,
            "current_to_a_publication": True,
            "reciprocal_density_rebuild": True,
            "global_psolver": True,
            "full_solve3_or_rk": True,
            "five_meter_meteorological_product": True,
        },
        "claim_limit": GM1_TILESTREAM_SOLVE3_PRESSURE_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledSolve3PressureLaunch(event, compiled, MappingProxyType(record))


def five_meter_tiled_solve3_pressure_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192
) -> Mapping[str, object]:
    """Exact incremental operation-A memory beyond the sealed post-WSM6 seam."""

    shapes = tiled_solve3_pressure_buffer_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=75
    )
    ppx_bytes = math.prod(shapes["ppx"]) * 4
    record = {
        "schema": "gm1.hybrid-tilestream-solve3-pressure-memory/v1",
        "domain": [75, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "incremental_host_per_buffer_bytes": ppx_bytes,
        "incremental_device_per_buffer_bytes": ppx_bytes,
        "incremental_double_host_bytes": 2 * ppx_bytes,
        "incremental_double_device_bytes": 2 * ppx_bytes,
        "post_wsm6_combined_double_host_bytes": 555_319_296,
        "post_wsm6_combined_double_device_bytes": 799_211_520,
        "full_domain_ppx_duplicate_bytes": 0,
        "global_pressure_workspace_executed": False,
        "operation_b_current_to_a_available": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_ABI",
    "CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_KERNEL",
    "CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE",
    "CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE_BYTES",
    "CUDA_GM1_TILESTREAM_SOLVE3_PRESSURE_SOURCE_SHA256",
    "CompiledGm1TiledSolve3Pressure",
    "CudaGm1TiledSolve3PressureLaunch",
    "GM1_ACCEPTED_REAL_POST_WSM6_OUTPUT_SHA256",
    "GM1_ACCEPTED_REAL_POST_WSM6_RECEIPT_SHA256",
    "GM1_ACCEPTED_REAL_POST_WSM6_SEMANTIC_SHA256",
    "GM1_ACCEPTED_REAL_PP3D_SHA256",
    "GM1_SOURCE_INITIALIZED_PHYSICAL_PPX_SHA256",
    "GM1_TILESTREAM_SOLVE3_PRESSURE_CLAIM_LIMIT",
    "Gm1TiledSolve3PressureBinding",
    "compile_cuda_gm1_tiled_solve3_pressure",
    "cpu_gm1_tiled_solve3_pressure_inplace",
    "fill_gm1_tiled_solve3_pressure_source",
    "five_meter_tiled_solve3_pressure_inventory",
    "launch_cuda_gm1_tiled_solve3_pressure",
    "scatter_gm1_tiled_solve3_pressure_output",
    "tiled_solve3_pressure_buffer_shapes",
)
