"""Source-anchored CUDA initialization for the first CM1 core carriers.

This module advances the CUDA state from structural sentinels to the
selected analytic initial values for the ten core carrier fields.  It is a
CUDA driver: every field update is performed by one packaged device kernel on
the same stream as structural allocation.  Importing the module does not
import CuPy or open a CUDA context.

The kernel implements only source-mapped CM1 r21.1 initialization facts:

* ``base.F:2318-2360,2661-2716`` -- analytic wind and moving-frame removal;
* ``init3d.F:445-482`` -- the selected analytic thermal perturbation;
* ``init3d.F:1733-1741,2278-2311`` -- zero pressure/vertical velocity and
  carrier copies;
* ``bc.F:180-364,528-1146`` -- the selected open-boundary result at startup.

It does not initialize the base-state, moisture, SGS/TKE, tendencies,
boundary history, clocks, dynamics, or physics.  Those omissions remain hard
gates even after this kernel passes its native post-``getset`` comparison.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass

from .config import CM1Config, Lane
from .cuda_state import (
    CUDA_STRUCTURAL_STATE_SCHEMA,
    CudaStructuralState,
    _expected_image_options,
    _validate_cubin_signature,
)


CUDA_CORE_INITIALIZATION_SCHEMA = "gpuwm-cm1.cuda-core-initialization/v1"
CUDA_CORE_INITIALIZATION_ABI = "gpuwm-cm1.cuda-core-initialization-kernel/v1"
CUDA_CORE_INITIALIZATION_KERNEL_NAME = "cm1_initialize_core_carriers_v1"
CUDA_CORE_INITIALIZATION_CLAIM_LIMIT = (
    "device-resident analytic initialization of the ten core carrier fields only; "
    "not base-state, moisture, SGS/TKE, dynamics, physics, forecast, or model parity"
)
CUDA_CORE_INITIALIZATION_THREADS = 256


CUDA_CORE_INITIALIZATION_SOURCE = r"""// gpuwm-cm1.cuda-core-initialization-kernel/v1
static_assert(sizeof(float) == 4, "CM1 carrier initialization requires float32");
static_assert(sizeof(unsigned long long) == 8, "CM1 carrier sizes require uint64");

__device__ __forceinline__ float cm1_u_ground_v1(const float z)
{
    const float pi = 3.1415926535897932384626433f;
    if (z <= 2000.0f) {
        const float angle = 90.0f * (z / 2000.0f) * (pi / 180.0f);
        return 7.0f - 7.0f * cosf(angle);
    }
    if (z <= 6000.0f) {
        return 7.0f + (z - 2000.0f) * (31.0f - 7.0f) / (6000.0f - 2000.0f);
    }
    return 31.0f;
}

__device__ __forceinline__ float cm1_v_ground_v1(const float z)
{
    const float pi = 3.1415926535897932384626433f;
    if (z <= 2000.0f) {
        const float angle = 90.0f * (z / 2000.0f) * (pi / 180.0f);
        return 7.0f * sinf(angle);
    }
    return 7.0f;
}

__device__ __forceinline__ float cm1_mass_height_v1(
    const unsigned int k,
    const float dz)
{
    return dz * ((float)k - 0.5f);
}

__device__ __forceinline__ float cm1_u_level_v1(
    const unsigned int k,
    const float dz,
    const float u_move)
{
    return cm1_u_ground_v1(cm1_mass_height_v1(k, dz)) - u_move;
}

__device__ __forceinline__ float cm1_v_level_v1(
    const unsigned int k,
    const float dz,
    const float v_move)
{
    return cm1_v_ground_v1(cm1_mass_height_v1(k, dz)) - v_move;
}

extern "C" __global__
void cm1_initialize_core_carriers_v1(
    float* __restrict__ ua,
    float* __restrict__ u3d,
    float* __restrict__ va,
    float* __restrict__ v3d,
    float* __restrict__ wa,
    float* __restrict__ w3d,
    float* __restrict__ ppi,
    float* __restrict__ pp3d,
    float* __restrict__ tha,
    float* __restrict__ th3d,
    const unsigned long long u_count,
    const unsigned long long v_count,
    const unsigned long long w_count,
    const unsigned long long scalar_count,
    const unsigned int u_x_count,
    const unsigned int v_x_count,
    const unsigned int scalar_x_count,
    const unsigned int u_y_count,
    const unsigned int v_y_count,
    const unsigned int scalar_y_count,
    const unsigned int nz,
    const unsigned int nx,
    const unsigned int ny,
    const float dx,
    const float dy,
    const float dz,
    const float x_min,
    const float y_min,
    const float u_move,
    const float v_move)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long linear = first; linear < u_count; linear += stride) {
        const unsigned long long yz = linear / u_x_count;
        const unsigned int k = (unsigned int)(yz / u_y_count);
        float value;
        if (k == 0u) {
            value = 1.875f * cm1_u_level_v1(1u, dz, u_move)
                  - 1.25f * cm1_u_level_v1(2u, dz, u_move)
                  + 0.375f * cm1_u_level_v1(3u, dz, u_move);
        } else if (k == nz + 1u) {
            value = 1.875f * cm1_u_level_v1(nz, dz, u_move)
                  - 1.25f * cm1_u_level_v1(nz - 1u, dz, u_move)
                  + 0.375f * cm1_u_level_v1(nz - 2u, dz, u_move);
        } else {
            value = cm1_u_level_v1(k, dz, u_move);
        }
        ua[linear] = value;
        u3d[linear] = value;
    }

    for (unsigned long long linear = first; linear < v_count; linear += stride) {
        const unsigned long long yz = linear / v_x_count;
        const unsigned int k = (unsigned int)(yz / v_y_count);
        float value;
        if (k == 0u) {
            value = 1.875f * cm1_v_level_v1(1u, dz, v_move)
                  - 1.25f * cm1_v_level_v1(2u, dz, v_move)
                  + 0.375f * cm1_v_level_v1(3u, dz, v_move);
        } else if (k == nz + 1u) {
            value = 1.875f * cm1_v_level_v1(nz, dz, v_move)
                  - 1.25f * cm1_v_level_v1(nz - 1u, dz, v_move)
                  + 0.375f * cm1_v_level_v1(nz - 2u, dz, v_move);
        } else {
            value = cm1_v_level_v1(k, dz, v_move);
        }
        va[linear] = value;
        v3d[linear] = value;
    }

    for (unsigned long long linear = first; linear < w_count; linear += stride) {
        wa[linear] = 0.0f;
        w3d[linear] = 0.0f;
    }

    for (unsigned long long linear = first; linear < scalar_count; linear += stride) {
        const unsigned int ax = (unsigned int)(linear % scalar_x_count);
        const unsigned long long yz = linear / scalar_x_count;
        const unsigned int ay = (unsigned int)(yz % scalar_y_count);
        const unsigned int k = (unsigned int)(yz / scalar_y_count);
        const int i = (int)ax - 2;
        const int j = (int)ay - 2;
        float theta_perturbation = 0.0f;

        if (i >= 1 && i <= (int)nx
            && j >= 1 && j <= (int)ny
            && k >= 1u && k <= nz) {
            const float x = ((float)i - 0.5f) * dx + x_min;
            const float y = ((float)j - 0.5f) * dy + y_min;
            const float z = cm1_mass_height_v1(k, dz);
            const float bx = x / 10000.0f;
            const float by = y / 10000.0f;
            const float bz = (z - 1400.0f) / 1400.0f;
            const float beta = sqrtf(bx * bx + by * by + bz * bz);
            if (beta < 1.0f) {
                const float pi = 3.1415926535897932384626433f;
                const float cosine = cosf(0.5f * pi * beta);
                theta_perturbation = cosine * cosine;
            }
        }

        ppi[linear] = 0.0f;
        pp3d[linear] = 0.0f;
        tha[linear] = theta_perturbation;
        th3d[linear] = theta_perturbation;
    }
}
"""
CUDA_CORE_INITIALIZATION_SOURCE_BYTES = CUDA_CORE_INITIALIZATION_SOURCE.encode("utf-8")
# Independent review pin: source edits must deliberately update this literal
# and the corresponding tests/evidence receipt.
CUDA_CORE_INITIALIZATION_SOURCE_SHA256 = (
    "aaf1d49f000c2d5e68b7fb900344a5a3ec739afd6234ab40e4e389aa48b9045f"
)
if hashlib.sha256(CUDA_CORE_INITIALIZATION_SOURCE_BYTES).hexdigest() != (
    CUDA_CORE_INITIALIZATION_SOURCE_SHA256
):
    raise RuntimeError(
        "packaged core-initialization CUDA source differs from its independent pin"
    )


def _canonical_arch(value: object) -> str:
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


def _validate_ptx(ptx: bytes, arch: str) -> None:
    try:
        text = ptx.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("core-initialization PTX must be UTF-8") from error
    required = (
        rf"(?m)^\s*\.target\s+sm_{re.escape(arch)}(?:,|\s|$)",
        r"(?m)^\s*\.address_size\s+64(?:\s|$)",
        rf"(?m)^\s*(?:\.visible\s+)?\.entry\s+{CUDA_CORE_INITIALIZATION_KERNEL_NAME}(?:\s|\()",
    )
    if not all(re.search(pattern, text) for pattern in required):
        raise ValueError(
            "core-initialization PTX lacks its target/address/kernel signature"
        )


@dataclass(frozen=True, slots=True)
class CudaCoreInitializationReceipt:
    source_sha256: str
    source_bytes: int
    ptx_sha256: str
    ptx_bytes: int
    cubin_sha256: str
    cubin_bytes: int
    target_arch: str
    device_id: int
    grid_x: int
    block_x: int
    event_pointer: int
    stream_pointer: int
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    config_sha256: str
    schema: str = CUDA_CORE_INITIALIZATION_SCHEMA
    claim_limit: str = CUDA_CORE_INITIALIZATION_CLAIM_LIMIT

    def __post_init__(self) -> None:
        if self.source_sha256 != CUDA_CORE_INITIALIZATION_SOURCE_SHA256:
            raise ValueError("core-initialization source receipt differs from its pin")
        if self.source_bytes != len(CUDA_CORE_INITIALIZATION_SOURCE_BYTES):
            raise ValueError(
                "core-initialization source byte count differs from its pin"
            )
        for digest in (self.ptx_sha256, self.cubin_sha256, self.config_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    "core-initialization receipt contains an invalid SHA-256"
                )
        for value in (
            self.ptx_bytes,
            self.cubin_bytes,
            self.device_id,
            self.grid_x,
            self.block_x,
            self.event_pointer,
            self.stream_pointer,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("core-initialization receipt integer is invalid")
        if self.ptx_bytes < 1 or self.cubin_bytes < 1:
            raise ValueError("core-initialization compiled images must be non-empty")
        if self.device_id < 0 or self.grid_x < 1 or self.block_x < 1:
            raise ValueError("core-initialization device/launch metadata is invalid")
        if self.event_pointer < 1:
            raise ValueError(
                "core-initialization ready event must be a real CUDA event"
            )
        if self.schema != CUDA_CORE_INITIALIZATION_SCHEMA:
            raise ValueError("core-initialization receipt schema is fixed")
        if self.claim_limit != CUDA_CORE_INITIALIZATION_CLAIM_LIMIT:
            raise ValueError("core-initialization receipt claim boundary is fixed")

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_limit": self.claim_limit,
            "compiled_images": {
                "cubin": {
                    "bytes": self.cubin_bytes,
                    "options": list(self.cubin_options),
                    "sha256": self.cubin_sha256,
                },
                "ptx": {
                    "bytes": self.ptx_bytes,
                    "options": list(self.ptx_options),
                    "sha256": self.ptx_sha256,
                },
            },
            "config_sha256": self.config_sha256,
            "device_id": self.device_id,
            "kernel": {
                "abi": CUDA_CORE_INITIALIZATION_ABI,
                "name": CUDA_CORE_INITIALIZATION_KERNEL_NAME,
                "source_bytes": self.source_bytes,
                "source_sha256": self.source_sha256,
                "target_arch": self.target_arch,
            },
            "launch": {
                "block_xyz": [self.block_x, 1, 1],
                "grid_xyz": [self.grid_x, 1, 1],
            },
            "ready_event": {
                "event_pointer": self.event_pointer,
                "stream_pointer": self.stream_pointer,
            },
            "schema": self.schema,
        }


@dataclass(frozen=True, slots=True)
class CudaCoreInitialization:
    state: CudaStructuralState
    ready_event: object
    receipt: CudaCoreInitializationReceipt
    _module_owner: object

    def __post_init__(self) -> None:
        if self.state.schema != CUDA_STRUCTURAL_STATE_SCHEMA:
            raise ValueError(
                "core initialization requires an audited production CUDA state"
            )
        if self.ready_event is None or self._module_owner is None:
            raise ValueError(
                "core initialization must retain CUDA event/module handles"
            )
        if self.receipt.device_id != self.state.arena_receipt.device_id:
            raise ValueError("core initialization ran on a different CUDA device")
        if self.receipt.stream_pointer != self.state.stream_pointer:
            raise ValueError("core initialization ran on a different CUDA stream")


def _validate_request(state: CudaStructuralState, config: CM1Config) -> None:
    if not isinstance(state, CudaStructuralState):
        raise TypeError("state must be a CudaStructuralState")
    if not isinstance(config, CM1Config):
        raise TypeError("config must be a CM1Config")
    if not state.audited_default_driver or state.schema != CUDA_STRUCTURAL_STATE_SCHEMA:
        raise ValueError(
            "scientific initialization refuses unverified CUDA test-double state"
        )
    if config.run.lane is not Lane.PRODUCT:
        raise ValueError("the analytic CUDA initializer is product-lane only")
    if config.sha256 != state.plan.config_sha256:
        raise ValueError("CUDA state and initialization configuration hashes differ")
    if (
        config.grid.nx,
        config.grid.ny,
        config.grid.nz,
        config.grid.dx_m,
        config.grid.dy_m,
        config.grid.dz_m,
    ) != (120, 120, 40, 1000.0, 1000.0, 500.0):
        raise ValueError("the analytic CUDA initializer requires the sealed grid")
    if not config.closures.moving_frame:
        raise ValueError(
            "the selected analytic initialization requires its moving frame"
        )


def initialize_cuda_core_carriers(
    state: CudaStructuralState,
    config: CM1Config,
) -> CudaCoreInitialization:
    """Enqueue source-anchored values for the ten core fields on CUDA.

    Validation finishes before CuPy is imported.  The structural ready event,
    these initialization writes, and the returned ready event are all ordered
    on the same selected stream.  The function performs no synchronization or
    host state copy.
    """

    _validate_request(state, config)
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")

    device = cupy.cuda.Device()
    if int(device.id) != state.arena_receipt.device_id:
        raise ValueError("current CUDA device differs from the allocated state device")
    device_arch = _canonical_arch(device.compute_capability)
    target_arch = _canonical_arch(compiler._get_arch())
    if target_arch != device_arch:
        raise ValueError(
            "compiled target architecture must exactly match the CUDA device"
        )
    nvrtc_parts = tuple(compiler._get_nvrtc_version())
    if len(nvrtc_parts) < 2:
        raise ValueError("NVRTC did not report a major/minor version")
    nvrtc_version = f"{int(nvrtc_parts[0])}.{int(nvrtc_parts[1])}"
    ptx_options, cubin_options = _expected_image_options(target_arch, nvrtc_version)

    ptx_program = compiler._NVRTCProgram(
        CUDA_CORE_INITIALIZATION_SOURCE,
        "cm1_core_initialization_v1_ptx.cu",
        method="ptx",
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_CORE_INITIALIZATION_SOURCE,
        "cm1_core_initialization_v1_cubin.cu",
        method="cubin",
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
    _validate_ptx(ptx, target_arch)
    _validate_cubin_signature(cubin)

    module = function.Module()
    module.load(cubin)
    kernel = module.get_function(CUDA_CORE_INITIALIZATION_KERNEL_NAME)

    fields = {
        name: state.field(name).array
        for name in (
            "ua",
            "u3d",
            "va",
            "v3d",
            "wa",
            "w3d",
            "ppi",
            "pp3d",
            "tha",
            "th3d",
        )
    }
    u_shape = state.field("ua").launch.allocated_shape_zyx
    v_shape = state.field("va").launch.allocated_shape_zyx
    scalar_shape = state.field("ppi").launch.allocated_shape_zyx
    counts = (
        state.field("ua").launch.element_count,
        state.field("va").launch.element_count,
        state.field("wa").launch.element_count,
        state.field("ppi").launch.element_count,
    )
    grid_x = (max(counts) + CUDA_CORE_INITIALIZATION_THREADS - 1) // (
        CUDA_CORE_INITIALIZATION_THREADS
    )
    x_min = -0.5 * config.grid.nx * config.grid.dx_m
    y_min = -0.5 * config.grid.ny * config.grid.dy_m
    arguments = (
        fields["ua"],
        fields["u3d"],
        fields["va"],
        fields["v3d"],
        fields["wa"],
        fields["w3d"],
        fields["ppi"],
        fields["pp3d"],
        fields["tha"],
        fields["th3d"],
        *(cupy.uint64(value) for value in counts),
        cupy.uint32(u_shape[2]),
        cupy.uint32(v_shape[2]),
        cupy.uint32(scalar_shape[2]),
        cupy.uint32(u_shape[1]),
        cupy.uint32(v_shape[1]),
        cupy.uint32(scalar_shape[1]),
        cupy.uint32(config.grid.nz),
        cupy.uint32(config.grid.nx),
        cupy.uint32(config.grid.ny),
        cupy.float32(config.grid.dx_m),
        cupy.float32(config.grid.dy_m),
        cupy.float32(config.grid.dz_m),
        cupy.float32(x_min),
        cupy.float32(y_min),
        cupy.float32(config.closures.u_move_m_s),
        cupy.float32(config.closures.v_move_m_s),
    )
    kernel(
        (grid_x, 1, 1),
        (CUDA_CORE_INITIALIZATION_THREADS, 1, 1),
        arguments,
        shared_mem=0,
        stream=state.stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(state.stream)
    receipt = CudaCoreInitializationReceipt(
        source_sha256=CUDA_CORE_INITIALIZATION_SOURCE_SHA256,
        source_bytes=len(CUDA_CORE_INITIALIZATION_SOURCE_BYTES),
        ptx_sha256=hashlib.sha256(ptx).hexdigest(),
        ptx_bytes=len(ptx),
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        cubin_bytes=len(cubin),
        target_arch=target_arch,
        device_id=int(device.id),
        grid_x=grid_x,
        block_x=CUDA_CORE_INITIALIZATION_THREADS,
        event_pointer=int(event.ptr),
        stream_pointer=state.stream_pointer,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        config_sha256=config.sha256,
    )
    return CudaCoreInitialization(
        state=state,
        ready_event=event,
        receipt=receipt,
        _module_owner=module,
    )


__all__ = [
    "CUDA_CORE_INITIALIZATION_ABI",
    "CUDA_CORE_INITIALIZATION_CLAIM_LIMIT",
    "CUDA_CORE_INITIALIZATION_KERNEL_NAME",
    "CUDA_CORE_INITIALIZATION_SCHEMA",
    "CUDA_CORE_INITIALIZATION_SOURCE",
    "CUDA_CORE_INITIALIZATION_SOURCE_SHA256",
    "CudaCoreInitialization",
    "CudaCoreInitializationReceipt",
    "initialize_cuda_core_carriers",
]
