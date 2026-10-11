"""Tile-streamable solve3 ppx/rho open-boundary lifecycle prefix.

CM1 orders ``bcp(ppx)`` at ``solve3.F:301`` before ``bcp(rho)`` at line 309,
and only then enters the held-tendency recapture at lines 319--337.  This
module seals that exact serial, four-open, single-rank boundary prefix after
the separately admitted physical ``ppx += pp3d`` operation A.

Only physical mass words and the one horizontal halo are published.  The
vertical halo is immutable transport state.  Held tendencies, current-to-A,
TKE, reciprocal density, global pressure, MPI, open histories, and a complete
solve3/RK step remain unavailable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SCHEMA = (
    "gm1.cuda-tilestream-solve3-ppx-rho-boundary/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_ABI = (
    "gm1.cuda-tilestream-solve3-ppx-rho-boundary-f32/v1"
)
CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_KERNEL_WE = "gm1_tiled_solve3_boundary_we_f32_v1"
CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_KERNEL_SN = "gm1_tiled_solve3_boundary_sn_f32_v1"
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
AUTHORITY_BC_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
AUTHORITY_ANCHORS = (
    "src/solve3.F:270-312",
    "src/bc.F:372-520",
)
GM1_ACCEPTED_OPERATION_A_RECEIPT_FILE_SHA256 = (
    "cd055f9bdcf46389d38cd56af42891aa07ce9f24dfb25776b355be8847a18007"
)
GM1_ACCEPTED_OPERATION_A_RECEIPT_SEMANTIC_SHA256 = (
    "c456f09b7b65e9938e17dc96d77eebec3ffdeda5f63a9ad5fc2f29583838639c"
)
GM1_ACCEPTED_OPERATION_A_OUTPUT_SHA256 = (
    "e09cc0350a09ddc3bfd0df830c6a8285a2c912f81ee8c377f99efcba1bcdd282"
)
GM1_ACCEPTED_REAL_RHO_SHA256 = (
    "3a17abd8b8c9ea400782d5cf5f96aac6824e6700177abab461782a812d9df325"
)
GM1_SOURCE_PADDED_PPX_SHA256 = (
    "70167c21d5c6f79f52704eaadff9c040c5df53f09dbab497e7abcebaa375801f"
)
GM1_CHILD_RHO_BOUNDARY_OPERATOR_SOURCE_SHA256 = (
    "c885efc95331b7002654b53e3835614918281becb8971973a266cadab76c05c7"
)
GM1_TILESTREAM_SOLVE3_BOUNDARY_CLAIM_LIMIT = (
    "CM1 r21.1 selected serial four-open ppx bcp then rho bcp on physical "
    "levels and one horizontal halo after admitted operation A; no held-rate "
    "recapture, momentum/TKE lifecycle, current-to-A, reciprocal-density "
    "rebuild, global psolver, MPI, open-history mutation, full solve3/RK, or "
    "five-metre meteorological product claim"
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


def tiled_solve3_boundary_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    padded = (nk, tile_ny + 2, tile_nx + 2)
    return MappingProxyType({"ppx": padded, "rho": padded})


def _vertical_halo_sha(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(value[0].astype("<f4", copy=False).tobytes(order="C"))
    digest.update(value[-1].astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3BoundaryBinding:
    """Physical sources plus writable one-halo transport destinations."""

    ppx: Any
    rho: Any
    ppx_padded: Any
    rho_padded: Any
    nx: int
    ny: int
    nk: int
    source_identity_sha256: str
    operation_a_receipt_file_sha256: str
    operation_a_receipt_semantic_sha256: str
    ppx_vertical_halo_sha256: str
    rho_vertical_halo_sha256: str
    compact_validation: bool

    @classmethod
    def create(
        cls,
        *,
        ppx: Any,
        rho: Any,
        ppx_padded: Any,
        rho_padded: Any,
        source_identity_sha256: str,
        operation_a_receipt_file_sha256: str,
        operation_a_receipt_semantic_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledSolve3BoundaryBinding":
        np = importlib.import_module("numpy")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        if (
            not isinstance(ppx, np.ndarray)
            or ppx.dtype != np.dtype("<f4")
            or ppx.ndim != 3
            or not ppx.flags.c_contiguous
        ):
            raise ValueError("ppx must be host C-<f4 [nk,ny,nx]")
        nk, ny, nx = map(int, ppx.shape)
        expected = {
            "rho": (nk, ny, nx),
            "ppx_padded": (nk + 2, ny + 2, nx + 2),
            "rho_padded": (nk + 2, ny + 2, nx + 2),
        }
        values = {
            "rho": rho,
            "ppx_padded": ppx_padded,
            "rho_padded": rho_padded,
        }
        for name, shape in expected.items():
            value = values[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise ValueError(f"{name} must be host C-<f4 {shape}")
        if not ppx_padded.flags.writeable or not rho_padded.flags.writeable:
            raise ValueError("padded ppx/rho destinations must be writable")
        all_values = (ppx, rho, ppx_padded, rho_padded)
        for index, left in enumerate(all_values):
            if any(np.shares_memory(left, right) for right in all_values[index + 1 :]):
                raise ValueError("solve3 boundary sources/destinations must not alias")
        if not all(bool(np.all(np.isfinite(value))) for value in all_values):
            raise ValueError("solve3 boundary source contains non-finite words")
        source_identity_sha256 = _require_sha256(
            "source_identity_sha256", source_identity_sha256
        )
        operation_a_receipt_file_sha256 = _require_sha256(
            "operation_a_receipt_file_sha256",
            operation_a_receipt_file_sha256,
        )
        operation_a_receipt_semantic_sha256 = _require_sha256(
            "operation_a_receipt_semantic_sha256",
            operation_a_receipt_semantic_sha256,
        )
        if allow_compact_validation:
            if min(nx, ny) < 7:
                raise ValueError("compact boundary domain must retain the open band")
        else:
            if (nx, ny, nk) != (300, 300, 75):
                raise ValueError("real solve3 boundary source must be [75,300,300]")
            if operation_a_receipt_file_sha256 != (
                GM1_ACCEPTED_OPERATION_A_RECEIPT_FILE_SHA256
            ) or operation_a_receipt_semantic_sha256 != (
                GM1_ACCEPTED_OPERATION_A_RECEIPT_SEMANTIC_SHA256
            ):
                raise ValueError("operation-A admission receipt differs from its seal")
            if hashlib.sha256(ppx.tobytes(order="C")).hexdigest() != (
                GM1_ACCEPTED_OPERATION_A_OUTPUT_SHA256
            ):
                raise ValueError("physical ppx differs from accepted operation A")
            if hashlib.sha256(rho.tobytes(order="C")).hexdigest() != (
                GM1_ACCEPTED_REAL_RHO_SHA256
            ):
                raise ValueError("physical rho differs from accepted live density")
            if hashlib.sha256(ppx_padded.tobytes(order="C")).hexdigest() != (
                GM1_SOURCE_PADDED_PPX_SHA256
            ):
                raise ValueError("padded ppx differs from source initialization")
            if not bool(np.all(rho_padded == np.float32(0.0))):
                raise ValueError("fresh solve3 padded rho must begin exact +0")
        return cls(
            ppx,
            rho,
            ppx_padded,
            rho_padded,
            nx,
            ny,
            nk,
            source_identity_sha256,
            operation_a_receipt_file_sha256,
            operation_a_receipt_semantic_sha256,
            _vertical_halo_sha(ppx_padded),
            _vertical_halo_sha(rho_padded),
            allow_compact_validation,
        )


def fill_gm1_tiled_solve3_boundary_source(
    binding: Gm1TiledSolve3BoundaryBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    """Fill physical tile interiors; local one-halo padding begins exact +0."""

    if not isinstance(binding, Gm1TiledSolve3BoundaryBinding):
        raise TypeError("binding must be Gm1TiledSolve3BoundaryBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec does not cover the boundary source domain")
    np = importlib.import_module("numpy")
    shapes = tiled_solve3_boundary_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(target, Mapping) or tuple(target) != tuple(shapes):
        raise ValueError("solve3 boundary tile inventory or order changed")
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
            raise ValueError(f"{name} tile must be writable C-<f4 {shape}")
        value.fill(np.float32(0.0))
        value[:, 1 : active_ny + 1, 1 : active_nx + 1] = getattr(binding, name)[
            :, y0:y1, x0:x1
        ]
    if np.shares_memory(target["ppx"], target["rho"]):
        raise ValueError("tile ppx and rho must not alias")
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "active_shape": [binding.nk, active_ny, active_nx],
            "open_edges": {
                "west": x0 == 0,
                "east": x1 == binding.nx,
                "south": y0 == 0,
                "north": y1 == binding.ny,
            },
            "local_padding_initialized": "+0",
        }
    )


def cpu_gm1_tiled_solve3_boundary_inplace(
    arrays: Mapping[str, Any],
    *,
    active_nx: int,
    active_ny: int,
    west: bool,
    east: bool,
    south: bool,
    north: bool,
) -> None:
    """Literal bcp west/east then south/north copy order."""

    np = importlib.import_module("numpy")
    if not isinstance(arrays, Mapping) or tuple(arrays) != ("ppx", "rho"):
        raise ValueError("CPU solve3 boundary inventory must be ppx,rho")
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    for name, value in arrays.items():
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.ndim != 3
            or not value.flags.c_contiguous
            or not value.flags.writeable
            or active_ny + 2 > value.shape[1]
            or active_nx + 2 > value.shape[2]
        ):
            raise ValueError(f"CPU {name} must be writable padded C-<f4")
        if west:
            value[:, : active_ny + 2, 0] = value[:, : active_ny + 2, 1]
        if east:
            value[:, : active_ny + 2, active_nx + 1] = value[
                :, : active_ny + 2, active_nx
            ]
        if south:
            value[:, 0, : active_nx + 2] = value[:, 1, : active_nx + 2]
        if north:
            value[:, active_ny + 1, : active_nx + 2] = value[
                :, active_ny, : active_nx + 2
            ]


def scatter_gm1_tiled_solve3_boundary_output(
    binding: Gm1TiledSolve3BoundaryBinding,
    spec: Gm1TileSpec,
    source: Mapping[str, Any],
) -> Mapping[str, object]:
    """Publish each physical or horizontal-halo word through one owner."""

    if not isinstance(binding, Gm1TiledSolve3BoundaryBinding):
        raise TypeError("binding must be Gm1TiledSolve3BoundaryBinding")
    np = importlib.import_module("numpy")
    shapes = tiled_solve3_boundary_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(source, Mapping) or tuple(source) != tuple(shapes):
        raise ValueError("solve3 boundary scatter inventory or order changed")
    for name, shape in shapes.items():
        value = source[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} boundary drain must be C-<f4 {shape}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    for name in shapes:
        destination = getattr(binding, f"{name}_padded")
        tile = source[name]
        destination[1 : binding.nk + 1, y0 + 1 : y1 + 1, x0 + 1 : x1 + 1] = tile[
            :, 1 : active_ny + 1, 1 : active_nx + 1
        ]
        if x0 == 0:
            destination[1 : binding.nk + 1, y0 + 1 : y1 + 1, 0] = tile[
                :, 1 : active_ny + 1, 0
            ]
        if x1 == binding.nx:
            destination[1 : binding.nk + 1, y0 + 1 : y1 + 1, -1] = tile[
                :, 1 : active_ny + 1, active_nx + 1
            ]
        if y0 == 0:
            local_x0 = 0 if x0 == 0 else 1
            local_x1 = active_nx + 2 if x1 == binding.nx else active_nx + 1
            global_x0 = 0 if x0 == 0 else x0 + 1
            global_x1 = binding.nx + 2 if x1 == binding.nx else x1 + 1
            destination[1 : binding.nk + 1, 0, global_x0:global_x1] = tile[
                :, 0, local_x0:local_x1
            ]
        if y1 == binding.ny:
            local_x0 = 0 if x0 == 0 else 1
            local_x1 = active_nx + 2 if x1 == binding.nx else active_nx + 1
            global_x0 = 0 if x0 == 0 else x0 + 1
            global_x1 = binding.nx + 2 if x1 == binding.nx else x1 + 1
            destination[1 : binding.nk + 1, -1, global_x0:global_x1] = tile[
                :, active_ny + 1, local_x0:local_x1
            ]
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "physical_words_per_carrier": binding.nk * active_ny * active_nx,
            "horizontal_halo_only": True,
            "vertical_halo_mutated": False,
        }
    )


CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE = r"""// gm1.cuda-tilestream-solve3-ppx-rho-boundary-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void gm1_tiled_solve3_boundary_we_f32_v1(
 float* p,unsigned int active_nx,unsigned int active_ny,
 unsigned int nk,unsigned int tile_nx,unsigned int tile_ny,
 unsigned int west,unsigned int east){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long n=(unsigned long long)nk*(active_ny+2u);
 if(q>=n)return;const unsigned int j=(unsigned int)(q%(active_ny+2u)),k=(unsigned int)(q/(active_ny+2u));
 const unsigned int nx=tile_nx+2u,ny=tile_ny+2u;
 if(west){p[i3(k,j,0u,ny,nx)]=p[i3(k,j,1u,ny,nx)];}
 if(east){p[i3(k,j,active_nx+1u,ny,nx)]=p[i3(k,j,active_nx,ny,nx)];}
}
extern "C" __global__ void gm1_tiled_solve3_boundary_sn_f32_v1(
 float* p,unsigned int active_nx,unsigned int active_ny,
 unsigned int nk,unsigned int tile_nx,unsigned int tile_ny,
 unsigned int south,unsigned int north){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long n=(unsigned long long)nk*(active_nx+2u);
 if(q>=n)return;const unsigned int i=(unsigned int)(q%(active_nx+2u)),k=(unsigned int)(q/(active_nx+2u));
 const unsigned int nx=tile_nx+2u,ny=tile_ny+2u;
 if(south){p[i3(k,0u,i,ny,nx)]=p[i3(k,1u,i,ny,nx)];}
 if(north){p[i3(k,active_ny+1u,i,ny,nx)]=p[i3(k,active_ny,i,ny,nx)];}
}
"""
CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE_BYTES = (
    CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE.encode("ascii")
)
CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE_BYTES
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledSolve3Boundary:
    module_owner: Any
    kernels: tuple[Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_solve3_boundary() -> CompiledGm1TiledSolve3Boundary:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    common = ("--std=c++14", "--fmad=false")
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE,
        "gm1_tiled_solve3_boundary_ptx.cu",
        method="ptx",
    ).compile(tuple(dict.fromkeys((*ptx_options, *common))))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE,
        "gm1_tiled_solve3_boundary_cubin.cu",
        method="cubin",
    ).compile(tuple(dict.fromkeys((*cubin_options, *common))))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    names = (
        CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_KERNEL_WE,
        CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_KERNEL_SN,
    )
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", text) for name in names):
        raise ValueError("compiled PTX lacks solve3 boundary entries")
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
    return CompiledGm1TiledSolve3Boundary(
        module,
        tuple(module.get_function(name) for name in names),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve3BoundaryLaunch:
    ready_event: Any
    compiled: CompiledGm1TiledSolve3Boundary
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_solve3_boundary(
    arrays: Mapping[str, Any],
    spec: Gm1TileSpec,
    *,
    nk: int,
    source_identity_sha256: str,
    operation_a_receipt_file_sha256: str,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve3Boundary | None = None,
) -> CudaGm1TiledSolve3BoundaryLaunch:
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    nk = _positive_int("nk", nk)
    source_identity_sha256 = _require_sha256(
        "source_identity_sha256", source_identity_sha256
    )
    operation_a_receipt_file_sha256 = _require_sha256(
        "operation_a_receipt_file_sha256", operation_a_receipt_file_sha256
    )
    cupy = importlib.import_module("cupy")
    shapes = tiled_solve3_boundary_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA solve3 boundary inventory or order changed")
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
    if int(checked["ppx"].data.ptr) == int(checked["rho"].data.ptr):
        raise ValueError("CUDA boundary ppx/rho must not alias")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    flags = (x0 == 0, x1 == spec.nx, y0 == 0, y1 == spec.ny)
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    compiled = (
        compile_cuda_gm1_tiled_solve3_boundary() if compiled is None else compiled
    )
    threads = 256
    we_words = nk * (active_ny + 2)
    sn_words = nk * (active_nx + 2)
    common = (
        cupy.uint32(active_nx),
        cupy.uint32(active_ny),
        cupy.uint32(nk),
        cupy.uint32(spec.tile_nx),
        cupy.uint32(spec.tile_ny),
    )
    with stream:
        for carrier in (checked["ppx"], checked["rho"]):
            compiled.kernels[0](
                ((we_words + threads - 1) // threads, 1, 1),
                (threads, 1, 1),
                (
                    carrier,
                    *common,
                    cupy.uint32(flags[0]),
                    cupy.uint32(flags[1]),
                ),
                stream=stream,
            )
            compiled.kernels[1](
                ((sn_words + threads - 1) // threads, 1, 1),
                (threads, 1, 1),
                (
                    carrier,
                    *common,
                    cupy.uint32(flags[2]),
                    cupy.uint32(flags[3]),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_ABI,
        "status": "ENQUEUED_PPX_THEN_RHO_OPEN_BOUNDARY_TILE",
        "tile": spec.index,
        "owned_yx": [y0, y1, x0, x1],
        "active_shape": [nk, active_ny, active_nx],
        "open_edges": {
            name: value
            for name, value in zip(
                ("west", "east", "south", "north"), flags, strict=True
            )
        },
        "boundary_order": ["ppx west/east then south/north", "rho same order"],
        "source_identity_sha256": source_identity_sha256,
        "operation_a_receipt_file_sha256": operation_a_receipt_file_sha256,
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
        "kernel_source_sha256": CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "refusals": {
            "held_recapture": True,
            "momentum_or_tke_lifecycle": True,
            "current_to_a": True,
            "reciprocal_density": True,
            "global_psolver": True,
            "mpi": True,
            "open_history_mutation": True,
            "full_solve3_or_rk": True,
            "five_meter_meteorological_product": True,
        },
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledSolve3BoundaryLaunch(event, compiled, MappingProxyType(record))


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3PreHeldBoundaryToken:
    """Host-publication token; permits only a later held-recapture admission."""

    receipt: Mapping[str, Any]


def finish_gm1_tiled_solve3_preheld_boundaries(
    binding: Gm1TiledSolve3BoundaryBinding,
    plan: Gm1TilePlan,
    launches: Sequence[CudaGm1TiledSolve3BoundaryLaunch],
) -> Gm1TiledSolve3PreHeldBoundaryToken:
    """Validate every tile and all published boundary dispositions."""

    if not isinstance(binding, Gm1TiledSolve3BoundaryBinding):
        raise TypeError("binding must be Gm1TiledSolve3BoundaryBinding")
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("boundary completion plan differs from source domain")
    if len(launches) != plan.ntiles:
        raise ValueError("boundary completion requires every exact tile receipt")
    tokens: list[str] = []
    streams: set[int] = set()
    for expected, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve3BoundaryLaunch):
            raise TypeError("boundary completion requires exact CUDA launch objects")
        if launch.ready_event is None:
            raise ValueError("boundary launch lost its ready event")
        receipt = launch.receipt
        if (
            receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SCHEMA
            or receipt.get("status") != "ENQUEUED_PPX_THEN_RHO_OPEN_BOUNDARY_TILE"
            or receipt.get("tile") != expected.index
            or receipt.get("source_identity_sha256") != binding.source_identity_sha256
            or receipt.get("operation_a_receipt_file_sha256")
            != binding.operation_a_receipt_file_sha256
        ):
            raise ValueError("boundary tile receipt identity/order differs")
        detached = dict(receipt)
        token = detached.pop("operation_token_sha256", None)
        if token != hashlib.sha256(_canonical(detached)).hexdigest():
            raise ValueError("boundary tile operation token changed")
        tokens.append(token)
        streams.add(int(receipt["stream_pointer"]))
    if len(streams) != 1:
        raise ValueError("all solve3 boundary tiles must use one exact stream")
    np = importlib.import_module("numpy")
    for name in ("ppx", "rho"):
        physical = getattr(binding, name)
        padded = getattr(binding, f"{name}_padded")
        if not np.array_equal(
            padded[1 : binding.nk + 1, 1 : binding.ny + 1, 1 : binding.nx + 1].view(
                np.uint32
            ),
            physical.view(np.uint32),
        ):
            raise ValueError(f"completed padded {name} physical words differ")
        if not np.array_equal(padded[1 : binding.nk + 1, 1:-1, 0], physical[:, :, 0]):
            raise ValueError(f"completed padded {name} west boundary differs")
        if not np.array_equal(padded[1 : binding.nk + 1, 1:-1, -1], physical[:, :, -1]):
            raise ValueError(f"completed padded {name} east boundary differs")
        expected_south = np.pad(physical[:, 0, :], ((0, 0), (1, 1)), mode="edge")
        expected_north = np.pad(physical[:, -1, :], ((0, 0), (1, 1)), mode="edge")
        if not np.array_equal(padded[1 : binding.nk + 1, 0], expected_south):
            raise ValueError(f"completed padded {name} south boundary differs")
        if not np.array_equal(padded[1 : binding.nk + 1, -1], expected_north):
            raise ValueError(f"completed padded {name} north boundary differs")
    if _vertical_halo_sha(binding.ppx_padded) != binding.ppx_vertical_halo_sha256:
        raise ValueError("ppx vertical halo was mutated")
    if _vertical_halo_sha(binding.rho_padded) != binding.rho_vertical_halo_sha256:
        raise ValueError("rho vertical halo was mutated")
    record = {
        "schema": "gm1.tilestream-solve3-preheld-boundary-token/v1",
        "status": "ADMITTED_PPX_RHO_BOUNDARIES_READY_FOR_HELD_RECAPTURE_ONLY",
        "source_identity_sha256": binding.source_identity_sha256,
        "operation_a": {
            "receipt_file_sha256": binding.operation_a_receipt_file_sha256,
            "receipt_semantic_sha256": binding.operation_a_receipt_semantic_sha256,
        },
        "tile_count": plan.ntiles,
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_operation_tokens": tokens,
        "stream_pointer": next(iter(streams)),
        "source_order": [
            "admitted physical ppx operation A",
            "ppx bcp west/east then south/north",
            "rho physical publication and bcp west/east then south/north",
        ],
        "outputs": {
            "ppx_padded_sha256": hashlib.sha256(
                binding.ppx_padded.tobytes(order="C")
            ).hexdigest(),
            "rho_padded_sha256": hashlib.sha256(
                binding.rho_padded.tobytes(order="C")
            ).hexdigest(),
        },
        "ready_for": "held-rate recapture admission only",
        "operation_b_current_to_a_available": False,
        "mpi_performed": False,
        "claim_limit": GM1_TILESTREAM_SOLVE3_BOUNDARY_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return Gm1TiledSolve3PreHeldBoundaryToken(MappingProxyType(record))


def five_meter_tiled_solve3_boundary_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192
) -> Mapping[str, object]:
    shapes = tiled_solve3_boundary_buffer_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=75
    )
    per_buffer = sum(math.prod(shape) * 4 for shape in shapes.values())
    record = {
        "schema": "gm1.hybrid-tilestream-solve3-boundary-memory/v1",
        "domain": [75, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "host_per_buffer_bytes": per_buffer,
        "device_per_buffer_bytes": per_buffer,
        "double_host_bytes": 2 * per_buffer,
        "double_device_bytes": 2 * per_buffer,
        "full_domain_padded_ppx_rho_duplicate_bytes": 0,
        "sequential_after_operation_a": True,
        "global_pressure_workspace_executed": False,
        "operation_b_current_to_a_available": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_BC_SHA256",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_ABI",
    "CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_KERNEL_SN",
    "CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_KERNEL_WE",
    "CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE",
    "CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE_BYTES",
    "CUDA_GM1_TILESTREAM_SOLVE3_BOUNDARY_SOURCE_SHA256",
    "CompiledGm1TiledSolve3Boundary",
    "CudaGm1TiledSolve3BoundaryLaunch",
    "GM1_TILESTREAM_SOLVE3_BOUNDARY_CLAIM_LIMIT",
    "Gm1TiledSolve3BoundaryBinding",
    "Gm1TiledSolve3PreHeldBoundaryToken",
    "compile_cuda_gm1_tiled_solve3_boundary",
    "cpu_gm1_tiled_solve3_boundary_inplace",
    "fill_gm1_tiled_solve3_boundary_source",
    "finish_gm1_tiled_solve3_preheld_boundaries",
    "five_meter_tiled_solve3_boundary_inventory",
    "launch_cuda_gm1_tiled_solve3_boundary",
    "scatter_gm1_tiled_solve3_boundary_output",
    "tiled_solve3_boundary_buffer_shapes",
)
