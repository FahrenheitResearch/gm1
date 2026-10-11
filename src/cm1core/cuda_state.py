"""Device-resident structural-state allocation for the first CM1 slice.

This module is deliberately import-light.  Importing it does not import CuPy,
inspect a CUDA device, or create a CUDA context.  :func:`allocate_cuda_structural_state`
first validates the complete backend-neutral state plan and its memory/launch
request; only then does its default driver factory import CuPy.

The audited default CUDA path writes raw 32-bit words.  Confirmed physical-
interior words are ``0x00000000`` (positive zero) and every not-yet-confirmed
halo word is ``0x7fc00000`` (the fixed quiet-NaN sentinel).  That path uses no
floating-point expression, host numerical initializer, CuPy elementwise
operation, or host copy.  Dependency-injected drivers are explicitly test-
double-only.  This remains structural scaffolding, not CM1 base-state or
perturbation initialization and not a native-oracle result.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .layout import STRUCTURAL_DYCORE_NAMES
from .state import (
    CONFIRMED_INTERIOR_WORD,
    DEFAULT_STRUCTURAL_ALLOCATION_LIMIT,
    FLOAT32_ITEMSIZE,
    STRUCTURAL_STATE_PLAN_SCHEMA,
    UNCONFIRMED_HALO_WORD,
    PlannedField,
    StructuralStatePlan,
)


CUDA_STRUCTURAL_STATE_SCHEMA = "gpuwm-cm1.cuda-structural-state/v1"
CUDA_STRUCTURAL_TEST_STATE_SCHEMA = "gpuwm-cm1.cuda-structural-state-test-double/v1"
CUDA_STRUCTURAL_KERNEL_ABI = "gpuwm-cm1.cuda-structural-init/v1"
CUDA_STRUCTURAL_KERNEL_NAME = "cm1_initialize_structural_field_v1"
CUDA_STRUCTURAL_CLAIM_LIMIT = (
    "device-resident layout/interior-sentinel initialization only; not CM1 "
    "base-state or perturbation initialization, a native oracle, or forecast output"
)
CUDA_STRUCTURAL_TEST_CLAIM_LIMIT = (
    "dependency-injected orchestration test only; no CUDA residency, compiled-image "
    "provenance, stream-ordering, synchronization, or scientific claim"
)
DEFAULT_THREADS_PER_BLOCK = 256
MAX_THREADS_PER_BLOCK = 1024
MAX_GRID_X = (1 << 31) - 1
MAX_UINT32 = (1 << 32) - 1
MAX_UINT64 = (1 << 64) - 1
RUNTIME_IDENTITY_KEYS: tuple[str, ...] = (
    "compute_capability",
    "cuda_driver_version",
    "cuda_runtime_version",
    "cupy_version",
    "device_id",
    "device_name",
    "nvrtc_version",
)
PRODUCTION_VIEW_PROVENANCE = "audited-default-cupy-device-view/v1"
TEST_VIEW_PROVENANCE = "injected-driver-view-unverified/v1"
PRODUCTION_MODULE_LOAD_PROVENANCE = (
    "cupy.cuda.function.Module.load(cubin)->get_function/v1"
)
TEST_MODULE_LOAD_PROVENANCE = "injected-driver-reported-load-unverified/v1"
PRODUCTION_SYNC_STATUS = (
    "audited default CuPy driver: all ten launches enqueued on one stream, then a "
    "ready event recorded on that stream; no explicit device, stream, event, or host "
    "synchronize call"
)
TEST_SYNC_STATUS = (
    "injected driver ordering and synchronization are unverified; no production claim"
)
_PRODUCTION_ATTESTATION = object()
_DEFAULT_DRIVER_ATTESTATION = object()

# CM1-ARITHMETIC-CONTRACT.md disables FMA contraction, reciprocal/square-root
# approximation and flush-to-zero by default.  This kernel is integer-only,
# but it is compiled under the same strict profile so its receipt can be
# composed with later numerical kernels without silently changing policy.
CUDA_STRICT_ARITHMETIC_OPTIONS: tuple[str, ...] = (
    "--std=c++14",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)


CUDA_STRUCTURAL_KERNEL_SOURCE = r"""// gpuwm-cm1.cuda-structural-init/v1
static_assert(sizeof(unsigned int) == 4, "CM1 structural words require uint32");
static_assert(sizeof(unsigned long long) == 8, "CM1 offsets require uint64");

extern "C" __global__
void cm1_initialize_structural_field_v1(
    unsigned int* __restrict__ arena,
    const unsigned long long word_offset,
    const unsigned long long element_count,
    const unsigned int x_count,
    const unsigned int y_count,
    const unsigned int z_count,
    const unsigned int x_begin,
    const unsigned int x_end,
    const unsigned int y_begin,
    const unsigned int y_end,
    const unsigned int z_begin,
    const unsigned int z_end)
{
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long linear = first;
         linear < element_count;
         linear += stride) {
        const unsigned int x = (unsigned int)(linear % x_count);
        const unsigned long long yz = linear / x_count;
        const unsigned int y = (unsigned int)(yz % y_count);
        const unsigned int z = (unsigned int)(yz / y_count);
        const bool is_confirmed_interior =
            x >= x_begin && x < x_end
            && y >= y_begin && y < y_end
            && z >= z_begin && z < z_end
            && z < z_count;
        arena[word_offset + linear] = is_confirmed_interior
            ? 0x00000000u
            : 0x7fc00000u;
    }
}
"""
CUDA_STRUCTURAL_KERNEL_SOURCE_BYTES = CUDA_STRUCTURAL_KERNEL_SOURCE.encode("utf-8")
# This literal is an independent review pin, not a digest computed and then
# trusted at runtime.  Any source edit must deliberately update this value and
# the corresponding test/receipt review.
CUDA_STRUCTURAL_KERNEL_SOURCE_SHA256 = (
    "af9cb0cce62d203e065c2aff40e0237f555a30ed700934a6aa3f717ff8250a17"
)
if hashlib.sha256(CUDA_STRUCTURAL_KERNEL_SOURCE_BYTES).hexdigest() != (
    CUDA_STRUCTURAL_KERNEL_SOURCE_SHA256
):
    raise RuntimeError(
        "packaged CUDA source differs from its independently pinned SHA-256"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_bytes(label: str, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise TypeError(f"{label} must be non-empty bytes")


def _runtime_dict(identity: tuple[tuple[str, str], ...]) -> dict[str, str]:
    keys = tuple(key for key, _ in identity)
    if keys != RUNTIME_IDENTITY_KEYS:
        raise ValueError(
            "runtime identity keys/order must be exactly "
            f"{RUNTIME_IDENTITY_KEYS!r}, received {keys!r}"
        )
    values = dict(identity)
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("runtime identity values must be non-empty strings")
    if not re.fullmatch(r"[0-9]+", values["compute_capability"]):
        raise ValueError("compute_capability must be canonical digits such as '86'")
    if not re.fullmatch(r"[0-9]+", values["cuda_driver_version"]):
        raise ValueError("cuda_driver_version must be canonical decimal digits")
    if not re.fullmatch(r"[0-9]+", values["cuda_runtime_version"]):
        raise ValueError("cuda_runtime_version must be canonical decimal digits")
    if not re.fullmatch(r"[0-9]+", values["device_id"]):
        raise ValueError("device_id must be canonical decimal digits")
    if not re.fullmatch(r"[0-9]+\.[0-9]+", values["nvrtc_version"]):
        raise ValueError("nvrtc_version must be canonical major.minor")
    return values


def _expected_image_options(
    target_arch: str,
    nvrtc_version: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not re.fullmatch(r"[0-9]+", target_arch):
        raise ValueError("target_arch must be canonical compute-capability digits")
    major = int(nvrtc_version.split(".", 1)[0])
    common = CUDA_STRICT_ARITHMETIC_OPTIONS
    if major >= 12:
        common += ("--device-as-default-execution-space",)
    return (
        common + (f"--gpu-architecture=compute_{target_arch}",),
        common + (f"--gpu-architecture=sm_{target_arch}",),
    )


def _validate_compiler_options(
    target_arch: str,
    nvrtc_version: str,
    ptx_options: tuple[str, ...],
    module_options: tuple[str, ...],
) -> None:
    lowered = tuple(option.lower() for option in ptx_options + module_options)
    conflicts = (
        "--use_fast_math",
        "-use_fast_math",
        "--fmad=true",
        "--ftz=true",
        "--prec-div=false",
        "--prec-sqrt=false",
    )
    if any(conflict in lowered for conflict in conflicts):
        raise ValueError("compiled image options enable a forbidden arithmetic mode")
    expected_ptx, expected_module = _expected_image_options(
        target_arch,
        nvrtc_version,
    )
    if ptx_options != expected_ptx:
        raise ValueError("PTX option tuple differs from the exact audited profile")
    if module_options != expected_module:
        raise ValueError("cubin option tuple differs from the exact audited profile")


def _validate_ptx_signature(ptx: bytes, target_arch: str) -> None:
    try:
        text = ptx.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("PTX image must be UTF-8 text") from error
    escaped_kernel_name = re.escape(CUDA_STRUCTURAL_KERNEL_NAME)
    required = (
        r"(?m)^\s*\.version\s+[0-9]+\.[0-9]+(?:\s|$)",
        rf"(?m)^\s*\.target\s+sm_{re.escape(target_arch)}(?:,|\s|$)",
        r"(?m)^\s*\.address_size\s+64(?:\s|$)",
        rf"(?m)^\s*(?:\.visible\s+)?\.entry\s+{escaped_kernel_name}(?:\s|\()",
    )
    if not all(re.search(pattern, text) for pattern in required):
        raise ValueError(
            "PTX image lacks its version/target/address-size/kernel signature"
        )


def _validate_cubin_signature(module_image: bytes) -> None:
    # CUDA cubins are little-endian ELF64 images with the registered EM_CUDA
    # machine id (190).  Checking the full e_ident plus e_machine avoids
    # accepting an arbitrary byte string prefixed with the four ELF bytes.
    if (
        len(module_image) < 20
        or module_image[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(module_image[18:20], "little") != 190
    ):
        raise ValueError("loaded CUDA module image is not an ELF64 CUDA cubin")


@dataclass(frozen=True, slots=True)
class DriverCompilation:
    """Opaque compiled handles plus the exact images used to build them."""

    kernel: object
    module_owner: object
    ptx: bytes
    module_image: bytes
    target_arch: str
    compiler_backend: str
    module_format: str
    ptx_options: tuple[str, ...]
    module_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]
    module_load_verified: bool
    module_load_provenance: str
    loaded_kernel_name: str
    loaded_device_id: int

    def __post_init__(self) -> None:
        _require_bytes("PTX image", self.ptx)
        _require_bytes("CUDA module image", self.module_image)
        if self.kernel is None or self.module_owner is None:
            raise ValueError("compiled kernel/module handles must be non-null")
        if not re.fullmatch(r"[0-9]+", self.target_arch):
            raise ValueError("target_arch must be canonical compute-capability digits")
        if self.compiler_backend != "nvrtc":
            raise ValueError(
                "the structural CUDA v1 compiler backend is fixed to NVRTC"
            )
        if self.module_format != "cubin":
            raise ValueError("the structural CUDA v1 loaded module must be a cubin")
        runtime = _runtime_dict(self.runtime_identity)
        if self.target_arch != runtime["compute_capability"]:
            raise ValueError(
                "compiled target architecture must exactly match the selected CUDA device"
            )
        _validate_compiler_options(
            self.target_arch,
            runtime["nvrtc_version"],
            self.ptx_options,
            self.module_options,
        )
        _validate_ptx_signature(self.ptx, self.target_arch)
        _validate_cubin_signature(self.module_image)
        if self.module_load_verified is not True:
            raise ValueError("CUDA module load/function lookup was not verified")
        if self.module_load_provenance not in {
            PRODUCTION_MODULE_LOAD_PROVENANCE,
            TEST_MODULE_LOAD_PROVENANCE,
        }:
            raise ValueError("CUDA module load has unknown provenance")
        if self.loaded_kernel_name != CUDA_STRUCTURAL_KERNEL_NAME:
            raise ValueError("loaded CUDA entry point differs from the structural ABI")
        if (
            isinstance(self.loaded_device_id, bool)
            or not isinstance(self.loaded_device_id, int)
            or self.loaded_device_id < 0
        ):
            raise ValueError("loaded_device_id must be a non-negative integer")
        if self.loaded_device_id != int(runtime["device_id"]):
            raise ValueError("loaded module device differs from runtime identity")


@dataclass(frozen=True, slots=True)
class CudaKernelBuildReceipt:
    source_bytes: int
    source_sha256: str
    ptx_bytes: int
    ptx_sha256: str
    module_bytes: int
    module_sha256: str
    target_arch: str
    compiler_backend: str
    module_format: str
    requested_options: tuple[str, ...]
    ptx_options: tuple[str, ...]
    module_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]
    audited_default_driver: bool
    module_load_verified: bool
    module_load_provenance: str
    loaded_kernel_name: str
    loaded_device_id: int
    kernel_name: str = CUDA_STRUCTURAL_KERNEL_NAME
    kernel_abi: str = CUDA_STRUCTURAL_KERNEL_ABI

    @classmethod
    def from_compilation(
        cls,
        compilation: DriverCompilation,
        *,
        audited_default_driver: bool,
    ) -> "CudaKernelBuildReceipt":
        return cls(
            source_bytes=len(CUDA_STRUCTURAL_KERNEL_SOURCE_BYTES),
            source_sha256=CUDA_STRUCTURAL_KERNEL_SOURCE_SHA256,
            ptx_bytes=len(compilation.ptx),
            ptx_sha256=_sha256(compilation.ptx),
            module_bytes=len(compilation.module_image),
            module_sha256=_sha256(compilation.module_image),
            target_arch=compilation.target_arch,
            compiler_backend=compilation.compiler_backend,
            module_format=compilation.module_format,
            requested_options=CUDA_STRICT_ARITHMETIC_OPTIONS,
            ptx_options=compilation.ptx_options,
            module_options=compilation.module_options,
            runtime_identity=compilation.runtime_identity,
            audited_default_driver=audited_default_driver,
            module_load_verified=compilation.module_load_verified,
            module_load_provenance=compilation.module_load_provenance,
            loaded_kernel_name=compilation.loaded_kernel_name,
            loaded_device_id=compilation.loaded_device_id,
        )

    def __post_init__(self) -> None:
        if self.source_sha256 != CUDA_STRUCTURAL_KERNEL_SOURCE_SHA256:
            raise ValueError("kernel source receipt does not match packaged source")
        if self.source_bytes != len(CUDA_STRUCTURAL_KERNEL_SOURCE_BYTES):
            raise ValueError("kernel source byte count does not match packaged source")
        for label, payload_bytes, digest in (
            ("PTX", self.ptx_bytes, self.ptx_sha256),
            ("module", self.module_bytes, self.module_sha256),
        ):
            if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int):
                raise TypeError(f"{label} byte count must be an integer")
            if payload_bytes < 1:
                raise ValueError(f"{label} byte count must be positive")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{label} digest must be lower-case SHA-256 hex")
        if self.requested_options != CUDA_STRICT_ARITHMETIC_OPTIONS:
            raise ValueError(
                "kernel receipt arithmetic options differ from the fixed profile"
            )
        if not re.fullmatch(r"[0-9]+", self.target_arch):
            raise ValueError("kernel receipt target architecture is not canonical")
        if self.compiler_backend != "nvrtc" or self.module_format != "cubin":
            raise ValueError("kernel receipt requires an NVRTC-loaded cubin")
        if self.kernel_name != CUDA_STRUCTURAL_KERNEL_NAME:
            raise ValueError("kernel receipt names the wrong entry point")
        if self.kernel_abi != CUDA_STRUCTURAL_KERNEL_ABI:
            raise ValueError("kernel receipt names the wrong ABI")
        if not isinstance(self.audited_default_driver, bool):
            raise TypeError("audited_default_driver must be a boolean")
        runtime = _runtime_dict(self.runtime_identity)
        if self.target_arch != runtime["compute_capability"]:
            raise ValueError(
                "kernel receipt target must exactly match the selected CUDA device"
            )
        _validate_compiler_options(
            self.target_arch,
            runtime["nvrtc_version"],
            self.ptx_options,
            self.module_options,
        )
        if self.module_load_verified is not True:
            raise ValueError(
                "kernel receipt requires verified module load/function lookup"
            )
        expected_load_provenance = (
            PRODUCTION_MODULE_LOAD_PROVENANCE
            if self.audited_default_driver
            else TEST_MODULE_LOAD_PROVENANCE
        )
        if self.module_load_provenance != expected_load_provenance:
            raise ValueError("module load provenance disagrees with driver provenance")
        if self.loaded_kernel_name != self.kernel_name:
            raise ValueError("loaded entry point differs from receipt kernel name")
        if self.loaded_device_id != int(runtime["device_id"]):
            raise ValueError(
                "loaded module device differs from receipt runtime identity"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "compiler_backend": self.compiler_backend,
            "kernel_abi": self.kernel_abi,
            "kernel_name": self.kernel_name,
            "module": {
                "bytes": self.module_bytes,
                "format": self.module_format,
                "load_verified": self.module_load_verified,
                "load_provenance": self.module_load_provenance,
                "loaded_device_id": self.loaded_device_id,
                "loaded_kernel_name": self.loaded_kernel_name,
                "options": list(self.module_options),
                "sha256": self.module_sha256,
            },
            "ptx": {
                "bytes": self.ptx_bytes,
                "options": list(self.ptx_options),
                "sha256": self.ptx_sha256,
            },
            "requested_strict_arithmetic_options": list(self.requested_options),
            "provenance_status": (
                "audited-default-cupy-driver"
                if self.audited_default_driver
                else "injected-driver-artifacts-unverified"
            ),
            "runtime_identity": dict(self.runtime_identity),
            "source": {
                "bytes": self.source_bytes,
                "sha256": self.source_sha256,
            },
            "target_arch": self.target_arch,
        }


@dataclass(frozen=True, slots=True)
class CudaFieldLaunch:
    field_name: str
    word_offset: int
    element_count: int
    allocated_shape_zyx: tuple[int, int, int]
    interior_begin_zyx: tuple[int, int, int]
    interior_end_zyx: tuple[int, int, int]
    grid_xyz: tuple[int, int, int]
    block_xyz: tuple[int, int, int]

    @classmethod
    def for_field(
        cls,
        field: PlannedField,
        *,
        word_offset: int,
        threads_per_block: int,
    ) -> "CudaFieldLaunch":
        z_slice, y_slice, x_slice = field.geometry.confirmed_slices_zyx
        blocks = (field.element_count + threads_per_block - 1) // threads_per_block
        if blocks > MAX_GRID_X:
            raise OverflowError(
                f"{field.layout.native_name} launch needs {blocks} x-blocks; "
                f"the v1 limit is {MAX_GRID_X}"
            )
        return cls(
            field_name=field.layout.native_name,
            word_offset=word_offset,
            element_count=field.element_count,
            allocated_shape_zyx=field.geometry.allocated_shape_zyx,
            interior_begin_zyx=(z_slice.start, y_slice.start, x_slice.start),
            interior_end_zyx=(z_slice.stop, y_slice.stop, x_slice.stop),
            grid_xyz=(blocks, 1, 1),
            block_xyz=(threads_per_block, 1, 1),
        )

    def __post_init__(self) -> None:
        if self.field_name not in STRUCTURAL_DYCORE_NAMES:
            raise ValueError("launch names an unknown structural field")
        integer_values = (
            self.word_offset,
            self.element_count,
            *self.allocated_shape_zyx,
            *self.interior_begin_zyx,
            *self.interior_end_zyx,
            *self.grid_xyz,
            *self.block_xyz,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise TypeError("CUDA launch metadata must contain integers")
        if self.word_offset < 0 or self.element_count < 1:
            raise ValueError("CUDA field launch offset/count are invalid")
        if self.word_offset > MAX_UINT64 or self.element_count > MAX_UINT64:
            raise OverflowError("CUDA field launch offset/count exceed uint64")
        if self.word_offset + self.element_count > MAX_UINT64 + 1:
            raise OverflowError(
                "CUDA field launch end exceeds uint64 address arithmetic"
            )
        if min(self.allocated_shape_zyx) < 1:
            raise ValueError("CUDA field allocation dimensions must be positive")
        if (
            max(
                *self.allocated_shape_zyx,
                *self.interior_begin_zyx,
                *self.interior_end_zyx,
            )
            > MAX_UINT32
        ):
            raise OverflowError("CUDA field dimensions/interior bounds exceed uint32")
        if self.element_count != (
            self.allocated_shape_zyx[0]
            * self.allocated_shape_zyx[1]
            * self.allocated_shape_zyx[2]
        ):
            raise ValueError("CUDA launch element count differs from its zyx shape")
        if any(
            begin < 0 or end <= begin or end > allocated
            for begin, end, allocated in zip(
                self.interior_begin_zyx,
                self.interior_end_zyx,
                self.allocated_shape_zyx,
            )
        ):
            raise ValueError("CUDA launch interior lies outside its allocation")
        if self.grid_xyz[1:] != (1, 1) or self.block_xyz[1:] != (1, 1):
            raise ValueError("the structural CUDA v1 launch is one-dimensional")
        if not 1 <= self.grid_xyz[0] <= MAX_GRID_X:
            raise ValueError("CUDA x-grid is outside the v1 launch limit")
        if not 1 <= self.block_xyz[0] <= MAX_THREADS_PER_BLOCK:
            raise ValueError("CUDA x-block is outside the v1 launch limit")
        if (
            self.grid_xyz[0]
            != (self.element_count + self.block_xyz[0] - 1) // self.block_xyz[0]
        ):
            raise ValueError("CUDA x-grid does not exactly cover the field")

    @property
    def byte_offset(self) -> int:
        return self.word_offset * FLOAT32_ITEMSIZE

    def as_dict(self) -> dict[str, object]:
        return {
            "allocated_shape_zyx": list(self.allocated_shape_zyx),
            "block_xyz": list(self.block_xyz),
            "byte_offset": self.byte_offset,
            "element_count": self.element_count,
            "field_name": self.field_name,
            "grid_xyz": list(self.grid_xyz),
            "interior_begin_zyx": list(self.interior_begin_zyx),
            "interior_end_zyx": list(self.interior_end_zyx),
            "word_offset": self.word_offset,
        }


def _c_strides_bytes(shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    z_count, y_count, x_count = shape_zyx
    del z_count
    return (
        y_count * x_count * FLOAT32_ITEMSIZE,
        x_count * FLOAT32_ITEMSIZE,
        FLOAT32_ITEMSIZE,
    )


@dataclass(frozen=True, slots=True)
class CudaArenaReceipt:
    word_count: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    device_pointer: int
    device_id: int
    c_contiguous: bool
    provenance: str

    def __post_init__(self) -> None:
        integers = (self.word_count, self.nbytes, self.device_pointer, self.device_id)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers
        ):
            raise TypeError("CUDA arena receipt integer fields must be integers")
        if self.word_count < 1 or self.nbytes != self.word_count * FLOAT32_ITEMSIZE:
            raise ValueError("CUDA arena receipt has an invalid word/byte count")
        if self.dtype != "<u4":
            raise ValueError("CUDA arena storage must be little-endian uint32")
        if self.shape != (self.word_count,) or self.strides_bytes != (
            FLOAT32_ITEMSIZE,
        ):
            raise ValueError("CUDA arena must be one-dimensional C-contiguous uint32")
        if self.device_pointer < 1 or self.device_id < 0:
            raise ValueError(
                "CUDA arena pointer must be non-null and device non-negative"
            )
        if self.c_contiguous is not True:
            raise ValueError("CUDA arena must be C-contiguous")
        if self.provenance not in {PRODUCTION_VIEW_PROVENANCE, TEST_VIEW_PROVENANCE}:
            raise ValueError("CUDA arena receipt has unknown provenance")

    def as_dict(self) -> dict[str, object]:
        return {
            "c_contiguous": self.c_contiguous,
            "device_id": self.device_id,
            "device_pointer": self.device_pointer,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "provenance": self.provenance,
            "shape": list(self.shape),
            "strides_bytes": list(self.strides_bytes),
            "word_count": self.word_count,
        }


@dataclass(frozen=True, slots=True)
class CudaFieldViewReceipt:
    dtype: str
    shape_zyx: tuple[int, int, int]
    strides_bytes: tuple[int, int, int]
    nbytes: int
    device_pointer: int
    device_id: int
    c_contiguous: bool
    provenance: str

    def __post_init__(self) -> None:
        integers = (self.nbytes, self.device_pointer, self.device_id)
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers
        ):
            raise TypeError("CUDA view receipt integer fields must be integers")
        if self.dtype != "<f4":
            raise ValueError("CUDA structural field view must be little-endian float32")
        if min(self.shape_zyx) < 1:
            raise ValueError("CUDA structural field view shape must be positive")
        if self.strides_bytes != _c_strides_bytes(self.shape_zyx):
            raise ValueError(
                "CUDA structural field view strides are not exact C-order zyx"
            )
        expected_nbytes = (
            self.shape_zyx[0] * self.shape_zyx[1] * self.shape_zyx[2] * FLOAT32_ITEMSIZE
        )
        if self.nbytes != expected_nbytes:
            raise ValueError("CUDA structural field view byte count differs from shape")
        if self.device_pointer < 1 or self.device_id < 0:
            raise ValueError(
                "CUDA field pointer must be non-null and device non-negative"
            )
        if self.c_contiguous is not True:
            raise ValueError("CUDA structural field view must be C-contiguous")
        if self.provenance not in {PRODUCTION_VIEW_PROVENANCE, TEST_VIEW_PROVENANCE}:
            raise ValueError("CUDA field view receipt has unknown provenance")

    def as_dict(self) -> dict[str, object]:
        return {
            "c_contiguous": self.c_contiguous,
            "device_id": self.device_id,
            "device_pointer": self.device_pointer,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "provenance": self.provenance,
            "shape_zyx": list(self.shape_zyx),
            "strides_bytes": list(self.strides_bytes),
        }


@dataclass(frozen=True, slots=True)
class DriverReadyEvent:
    event: object
    event_pointer: int
    stream_pointer: int
    recorded: bool
    provenance: str

    def __post_init__(self) -> None:
        if self.event is None:
            raise ValueError("CUDA ready event handle must be non-null")
        for name, value in (
            ("event_pointer", self.event_pointer),
            ("stream_pointer", self.stream_pointer),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.recorded is not True:
            raise ValueError("CUDA ready event must be recorded after initialization")
        if self.provenance not in {PRODUCTION_VIEW_PROVENANCE, TEST_VIEW_PROVENANCE}:
            raise ValueError("CUDA ready event has unknown provenance")
        if self.provenance == PRODUCTION_VIEW_PROVENANCE and self.event_pointer == 0:
            raise ValueError("audited CUDA ready event pointer must be non-null")

    def as_dict(self) -> dict[str, object]:
        return {
            "event_pointer": self.event_pointer,
            "provenance": self.provenance,
            "recorded": self.recorded,
            "stream_pointer": self.stream_pointer,
        }


@dataclass(frozen=True, slots=True)
class CudaStructuralField:
    field: PlannedField
    array: object
    launch: CudaFieldLaunch
    view_receipt: CudaFieldViewReceipt

    def __post_init__(self) -> None:
        if self.field.layout.native_name != self.launch.field_name:
            raise ValueError("CUDA field metadata and launch name disagree")
        if self.field.element_count != self.launch.element_count:
            raise ValueError("CUDA field metadata and launch count disagree")
        if self.field.geometry.allocated_shape_zyx != self.launch.allocated_shape_zyx:
            raise ValueError("CUDA field metadata and launch shape disagree")
        if self.view_receipt.shape_zyx != self.launch.allocated_shape_zyx:
            raise ValueError("CUDA field view receipt and launch shape disagree")
        if self.view_receipt.nbytes != self.field.nbytes:
            raise ValueError("CUDA field view receipt and state-plan byte count disagree")


class CudaDriver(Protocol):
    """Small driver seam used by the real CuPy driver and CPU-only fakes."""

    def compile(
        self,
        source: str,
        options: tuple[str, ...],
        kernel_name: str,
    ) -> DriverCompilation: ...

    def current_stream(self, stream: object | None) -> object: ...

    def stream_pointer(self, stream: object) -> int: ...

    def allocate_words(self, word_count: int) -> object: ...

    def describe_arena(
        self,
        arena: object,
        word_count: int,
        *,
        audited: bool,
    ) -> CudaArenaReceipt: ...

    def field_view(
        self,
        arena: object,
        word_offset: int,
        element_count: int,
        shape_zyx: tuple[int, int, int],
    ) -> object: ...

    def describe_field_view(
        self,
        array: object,
        launch: CudaFieldLaunch,
        *,
        audited: bool,
    ) -> CudaFieldViewReceipt: ...

    def launch(
        self,
        kernel: object,
        arena: object,
        launch: CudaFieldLaunch,
        stream: object,
    ) -> None: ...

    def record_ready_event(
        self,
        stream: object,
        *,
        audited: bool,
    ) -> DriverReadyEvent: ...


DriverFactory = Callable[[], CudaDriver]


@dataclass(frozen=True, slots=True)
class CudaStructuralState:
    plan: StructuralStatePlan
    arena: object
    arena_receipt: CudaArenaReceipt
    fields: tuple[CudaStructuralField, ...]
    kernel_receipt: CudaKernelBuildReceipt
    ready_event_receipt: DriverReadyEvent
    stream: object
    stream_pointer: int
    ordering_trace: tuple[str, ...]
    synchronization_status: str
    audited_default_driver: bool
    schema: str
    claim_limit: str
    _module_owner: object
    _production_attestation: object | None

    def __post_init__(self) -> None:
        if (
            self.arena is None
            or self.ready_event_receipt.event is None
            or self._module_owner is None
        ):
            raise ValueError("CUDA state handles must be non-null")
        if not isinstance(self.audited_default_driver, bool):
            raise TypeError("audited_default_driver must be a boolean")
        if self.audited_default_driver:
            expected_identity = (
                CUDA_STRUCTURAL_STATE_SCHEMA,
                CUDA_STRUCTURAL_CLAIM_LIMIT,
                PRODUCTION_SYNC_STATUS,
                PRODUCTION_VIEW_PROVENANCE,
            )
            if self._production_attestation is not _PRODUCTION_ATTESTATION:
                raise ValueError(
                    "production CUDA state lacks private default-driver attestation"
                )
        else:
            expected_identity = (
                CUDA_STRUCTURAL_TEST_STATE_SCHEMA,
                CUDA_STRUCTURAL_TEST_CLAIM_LIMIT,
                TEST_SYNC_STATUS,
                TEST_VIEW_PROVENANCE,
            )
            if self._production_attestation is not None:
                raise ValueError(
                    "test-driver state cannot carry production attestation"
                )
        expected_schema, expected_claim, expected_sync, expected_provenance = (
            expected_identity
        )
        if (self.schema, self.claim_limit, self.synchronization_status) != (
            expected_schema,
            expected_claim,
            expected_sync,
        ):
            raise ValueError(
                "CUDA state schema/claim/synchronization do not match provenance"
            )
        if (
            self.kernel_receipt.audited_default_driver
            is not self.audited_default_driver
        ):
            raise ValueError("kernel receipt and state driver provenance disagree")
        if self.arena_receipt.provenance != expected_provenance:
            raise ValueError("CUDA arena provenance disagrees with state provenance")
        if self.ready_event_receipt.provenance != expected_provenance:
            raise ValueError(
                "CUDA ready-event provenance disagrees with state provenance"
            )
        if self.ready_event_receipt.stream_pointer != self.stream_pointer:
            raise ValueError("CUDA ready event was not recorded on the selected stream")
        if self.arena_receipt.nbytes != self.plan.total_nbytes:
            raise ValueError("CUDA arena receipt bytes differ from the state plan")
        if self.arena_receipt.device_id != self.kernel_receipt.loaded_device_id:
            raise ValueError("CUDA arena and loaded module device ids disagree")
        names = tuple(field.field.layout.native_name for field in self.fields)
        if names != STRUCTURAL_DYCORE_NAMES:
            raise ValueError("CUDA field inventory/order differs from the state plan")
        if tuple(field.field for field in self.fields) != self.plan.fields:
            raise ValueError("CUDA field metadata differs from the state plan")
        expected_offset = 0
        for field in self.fields:
            if field.launch.word_offset != expected_offset:
                raise ValueError("CUDA field arena offsets are not contiguous")
            if field.view_receipt.provenance != expected_provenance:
                raise ValueError(
                    "CUDA field view provenance disagrees with state provenance"
                )
            if field.view_receipt.device_id != self.arena_receipt.device_id:
                raise ValueError("CUDA field view and arena device ids disagree")
            if field.view_receipt.device_pointer != (
                self.arena_receipt.device_pointer + field.launch.byte_offset
            ):
                raise ValueError(
                    "CUDA field view pointer is not the exact arena offset"
                )
            expected_offset += field.launch.element_count
        if expected_offset * FLOAT32_ITEMSIZE != self.plan.total_nbytes:
            raise ValueError("CUDA arena bytes differ from the state plan")
        for name, pointer in (("stream_pointer", self.stream_pointer),):
            if isinstance(pointer, bool) or not isinstance(pointer, int) or pointer < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected_trace = (
            "launch-plan-prevalidated",
            "driver-created",
            "kernel-compiled-and-loaded",
            "stream-selected",
            "arena-allocated-and-validated",
            *(
                step
                for name in STRUCTURAL_DYCORE_NAMES
                for step in (
                    f"field-view-validated:{name}",
                    f"launch-enqueued:{name}",
                )
            ),
            "ready-event-recorded-on-selected-stream",
        )
        if self.ordering_trace != expected_trace:
            raise ValueError(
                "CUDA initialization ordering trace differs from the v1 sequence"
            )

    @property
    def total_nbytes(self) -> int:
        return self.plan.total_nbytes

    @property
    def arena_pointer(self) -> int:
        return self.arena_receipt.device_pointer

    @property
    def ready_event(self) -> object:
        return self.ready_event_receipt.event

    @property
    def ready_event_pointer(self) -> int:
        return self.ready_event_receipt.event_pointer

    def field(self, name: str) -> CudaStructuralField:
        for field in self.fields:
            if field.field.layout.native_name == name:
                return field
        raise KeyError(name)

    def receipt_dict(self) -> dict[str, object]:
        return {
            "arena": self.arena_receipt.as_dict(),
            "audited_default_driver": self.audited_default_driver,
            "claim_limit": self.claim_limit,
            "initialization_words": {
                "confirmed_interior": f"0x{CONFIRMED_INTERIOR_WORD:08x}",
                "unconfirmed_halo": f"0x{UNCONFIRMED_HALO_WORD:08x}",
            },
            "kernel": self.kernel_receipt.as_dict(),
            "launches": [
                field.launch.as_dict() | {"view": field.view_receipt.as_dict()}
                for field in self.fields
            ],
            "ordering_trace": list(self.ordering_trace),
            "plan_sha256": self.plan.sha256,
            "ready_event": self.ready_event_receipt.as_dict(),
            "schema": self.schema,
            "stream_pointer": self.stream_pointer,
            "synchronization": self.synchronization_status,
        }


class _CupyCudaDriver:
    """CuPy-backed CUDA driver, instantiated only after semantic validation."""

    def __init__(
        self,
        cupy_module: object,
        *,
        _factory_attestation: object | None = None,
    ) -> None:
        self._cupy = cupy_module
        self._compiler = importlib.import_module("cupy.cuda.compiler")
        self._function = importlib.import_module("cupy.cuda.function")
        self._factory_attestation = _factory_attestation

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @classmethod
    def _canonical_arch(cls, value: object) -> str:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            major, minor = value
            if all(
                isinstance(part, int) and not isinstance(part, bool) for part in value
            ):
                return f"{major}{minor}"
        text = cls._text(value).strip().lower()
        for prefix in ("compute_", "sm_"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
        if re.fullmatch(r"[0-9]+\.[0-9]+", text):
            major, minor = text.split(".", 1)
            text = major + minor
        if not re.fullmatch(r"[0-9]+", text):
            raise ValueError("CUDA architecture must be canonicalizable to digits")
        return text

    def _require_audited_origin(self) -> None:
        if self._factory_attestation is not _DEFAULT_DRIVER_ATTESTATION:
            raise RuntimeError(
                "CuPy production driver lacks private factory attestation"
            )

    def compile(
        self,
        source: str,
        options: tuple[str, ...],
        kernel_name: str,
    ) -> DriverCompilation:
        if source != CUDA_STRUCTURAL_KERNEL_SOURCE:
            raise ValueError("driver received unexpected structural kernel source")
        if options != CUDA_STRICT_ARITHMETIC_OPTIONS:
            raise ValueError("driver received unexpected arithmetic options")
        if kernel_name != CUDA_STRUCTURAL_KERNEL_NAME:
            raise ValueError("driver received unexpected kernel name")

        device = self._cupy.cuda.Device()
        device_arch = self._canonical_arch(device.compute_capability)
        target_arch = self._canonical_arch(self._compiler._get_arch())
        nvrtc_version_parts = tuple(self._compiler._get_nvrtc_version())
        if len(nvrtc_version_parts) < 2 or any(
            isinstance(part, bool) or not isinstance(part, int) or part < 0
            for part in nvrtc_version_parts[:2]
        ):
            raise ValueError(
                "NVRTC version must provide non-negative major/minor integers"
            )
        nvrtc_version = f"{nvrtc_version_parts[0]}.{nvrtc_version_parts[1]}"
        ptx_options, module_options = _expected_image_options(
            target_arch,
            nvrtc_version,
        )

        ptx_program = self._compiler._NVRTCProgram(
            source,
            "cm1_structural_state_v1_ptx.cu",
            method="ptx",
        )
        ptx, _ = ptx_program.compile(ptx_options)
        module_program = self._compiler._NVRTCProgram(
            source,
            "cm1_structural_state_v1_cubin.cu",
            method="cubin",
        )
        module_image, _ = module_program.compile(module_options)
        if isinstance(ptx, str):
            ptx = ptx.encode("utf-8")
        elif not isinstance(ptx, bytes):
            ptx = bytes(ptx)
        if isinstance(module_image, str):
            module_image = module_image.encode("latin-1")
        elif not isinstance(module_image, bytes):
            module_image = bytes(module_image)

        module = self._function.Module()
        module.load(module_image)
        kernel = module.get_function(kernel_name)

        runtime = self._cupy.cuda.runtime
        properties = runtime.getDeviceProperties(device.id)
        device_name = properties.get("name", properties.get(b"name", "unknown"))
        runtime_identity = (
            ("compute_capability", device_arch),
            ("cuda_driver_version", self._text(runtime.driverGetVersion())),
            ("cuda_runtime_version", self._text(runtime.runtimeGetVersion())),
            ("cupy_version", self._text(self._cupy.__version__)),
            ("device_id", self._text(device.id)),
            ("device_name", self._text(device_name)),
            ("nvrtc_version", nvrtc_version),
        )
        return DriverCompilation(
            kernel=kernel,
            module_owner=module,
            ptx=ptx,
            module_image=module_image,
            target_arch=target_arch,
            compiler_backend="nvrtc",
            module_format="cubin",
            ptx_options=ptx_options,
            module_options=module_options,
            runtime_identity=runtime_identity,
            module_load_verified=True,
            module_load_provenance=PRODUCTION_MODULE_LOAD_PROVENANCE,
            loaded_kernel_name=kernel_name,
            loaded_device_id=int(device.id),
        )

    def current_stream(self, stream: object | None) -> object:
        if stream is None:
            return self._cupy.cuda.get_current_stream()
        if not hasattr(stream, "ptr"):
            raise TypeError("CUDA stream must expose a driver pointer")
        return stream

    def stream_pointer(self, stream: object) -> int:
        pointer = int(stream.ptr)
        if pointer < 0:
            raise ValueError("CUDA stream pointer cannot be negative")
        return pointer

    def allocate_words(self, word_count: int) -> object:
        return self._cupy.empty((word_count,), dtype=self._cupy.uint32)

    def describe_arena(
        self,
        arena: object,
        word_count: int,
        *,
        audited: bool,
    ) -> CudaArenaReceipt:
        if audited:
            self._require_audited_origin()
        if not isinstance(arena, self._cupy.ndarray):
            raise TypeError("audited CUDA arena must be a CuPy ndarray")
        return CudaArenaReceipt(
            word_count=word_count,
            nbytes=int(arena.nbytes),
            dtype=str(arena.dtype.str),
            shape=tuple(int(value) for value in arena.shape),
            strides_bytes=tuple(int(value) for value in arena.strides),
            device_pointer=int(arena.data.ptr),
            device_id=int(arena.device.id),
            c_contiguous=bool(arena.flags.c_contiguous),
            provenance=(
                PRODUCTION_VIEW_PROVENANCE if audited else TEST_VIEW_PROVENANCE
            ),
        )

    def field_view(
        self,
        arena: object,
        word_offset: int,
        element_count: int,
        shape_zyx: tuple[int, int, int],
    ) -> object:
        segment = arena[word_offset : word_offset + element_count]
        return segment.view(self._cupy.float32).reshape(shape_zyx, order="C")

    def describe_field_view(
        self,
        array: object,
        launch: CudaFieldLaunch,
        *,
        audited: bool,
    ) -> CudaFieldViewReceipt:
        if audited:
            self._require_audited_origin()
        if not isinstance(array, self._cupy.ndarray):
            raise TypeError("audited CUDA structural view must be a CuPy ndarray")
        return CudaFieldViewReceipt(
            dtype=str(array.dtype.str),
            shape_zyx=tuple(int(value) for value in array.shape),
            strides_bytes=tuple(int(value) for value in array.strides),
            nbytes=int(array.nbytes),
            device_pointer=int(array.data.ptr),
            device_id=int(array.device.id),
            c_contiguous=bool(array.flags.c_contiguous),
            provenance=(
                PRODUCTION_VIEW_PROVENANCE if audited else TEST_VIEW_PROVENANCE
            ),
        )

    def launch(
        self,
        kernel: object,
        arena: object,
        launch: CudaFieldLaunch,
        stream: object,
    ) -> None:
        z_count, y_count, x_count = launch.allocated_shape_zyx
        z_begin, y_begin, x_begin = launch.interior_begin_zyx
        z_end, y_end, x_end = launch.interior_end_zyx
        arguments = (
            arena,
            self._cupy.uint64(launch.word_offset),
            self._cupy.uint64(launch.element_count),
            self._cupy.uint32(x_count),
            self._cupy.uint32(y_count),
            self._cupy.uint32(z_count),
            self._cupy.uint32(x_begin),
            self._cupy.uint32(x_end),
            self._cupy.uint32(y_begin),
            self._cupy.uint32(y_end),
            self._cupy.uint32(z_begin),
            self._cupy.uint32(z_end),
        )
        kernel(
            launch.grid_xyz,
            launch.block_xyz,
            arguments,
            shared_mem=0,
            stream=stream,
        )

    def record_ready_event(
        self,
        stream: object,
        *,
        audited: bool,
    ) -> DriverReadyEvent:
        if audited:
            self._require_audited_origin()
        event = self._cupy.cuda.Event(disable_timing=True)
        event.record(stream)
        return DriverReadyEvent(
            event=event,
            event_pointer=int(event.ptr),
            stream_pointer=self.stream_pointer(stream),
            recorded=True,
            provenance=(
                PRODUCTION_VIEW_PROVENANCE if audited else TEST_VIEW_PROVENANCE
            ),
        )


def _default_driver_factory() -> CudaDriver:
    cupy_module = importlib.import_module("cupy")
    return _CupyCudaDriver(
        cupy_module,
        _factory_attestation=_DEFAULT_DRIVER_ATTESTATION,
    )


def _validate_allocation_request(
    plan: StructuralStatePlan,
    *,
    max_bytes: int,
    threads_per_block: int,
    driver_factory: DriverFactory,
) -> None:
    if not isinstance(plan, StructuralStatePlan):
        raise TypeError("plan must be a StructuralStatePlan")
    if plan.schema != STRUCTURAL_STATE_PLAN_SCHEMA:
        raise ValueError("CUDA allocation requires the fixed state-plan schema")
    if (
        tuple(field.layout.native_name for field in plan.fields)
        != STRUCTURAL_DYCORE_NAMES
    ):
        raise ValueError("CUDA allocation requires the fixed ten-field state inventory")
    if any(
        field.dtype != "<f4" or field.itemsize != FLOAT32_ITEMSIZE
        for field in plan.fields
    ):
        raise ValueError("CUDA allocation requires little-endian float32 state fields")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if plan.total_nbytes > max_bytes:
        raise MemoryError(
            f"CUDA structural state needs {plan.total_nbytes} bytes; limit is {max_bytes}"
        )
    if plan.total_nbytes % FLOAT32_ITEMSIZE:
        raise ValueError("CUDA structural state byte count is not word aligned")
    if (
        isinstance(threads_per_block, bool)
        or not isinstance(threads_per_block, int)
        or not 1 <= threads_per_block <= MAX_THREADS_PER_BLOCK
    ):
        raise ValueError(
            f"threads_per_block must be an integer in [1, {MAX_THREADS_PER_BLOCK}]"
        )
    if not callable(driver_factory):
        raise TypeError("driver_factory must be callable")


def _precompute_launches(
    plan: StructuralStatePlan,
    *,
    threads_per_block: int,
) -> tuple[CudaFieldLaunch, ...]:
    """Seal every kernel argument and arena offset before CUDA can be imported."""

    launches: list[CudaFieldLaunch] = []
    word_offset = 0
    for field in plan.fields:
        launch = CudaFieldLaunch.for_field(
            field,
            word_offset=word_offset,
            threads_per_block=threads_per_block,
        )
        if launch.word_offset != word_offset:
            raise ValueError("precomputed CUDA field offsets are not contiguous")
        launches.append(launch)
        word_offset += launch.element_count
        if word_offset > MAX_UINT64 + 1:
            raise OverflowError("precomputed CUDA arena extent exceeds uint64")
    planned_word_count = plan.total_nbytes // FLOAT32_ITEMSIZE
    if word_offset != planned_word_count:
        raise ValueError("precomputed CUDA launch extents differ from the state plan")
    return tuple(launches)


def allocate_cuda_structural_state(
    plan: StructuralStatePlan,
    *,
    max_bytes: int = DEFAULT_STRUCTURAL_ALLOCATION_LIMIT,
    threads_per_block: int = DEFAULT_THREADS_PER_BLOCK,
    stream: object | None = None,
    driver_factory: DriverFactory = _default_driver_factory,
) -> CudaStructuralState:
    """Allocate and asynchronously initialize the ten state fields on CUDA.

    All semantic, byte-budget and launch-shape checks occur before invoking
    ``driver_factory``.  The default factory is the sole CuPy import site.  A
    single uint32 device arena backs ten C-contiguous float32 ``(z, y, x)``
    views.  The audited default driver initializes them with the explicit
    packaged CUDA C++ kernel and records a ready event after the ten launches.
    Injected drivers produce an explicitly unverified test-double state.
    """

    _validate_allocation_request(
        plan,
        max_bytes=max_bytes,
        threads_per_block=threads_per_block,
        driver_factory=driver_factory,
    )
    launches = _precompute_launches(
        plan,
        threads_per_block=threads_per_block,
    )
    ordering_trace = ["launch-plan-prevalidated"]

    # The factory is intentionally called only after the complete allocation
    # request above is admitted.  With the default factory, this is the first
    # point at which importing CuPy or opening CUDA is possible.
    selected_default_factory = driver_factory is _default_driver_factory
    driver = driver_factory()
    audited_default_driver = (
        selected_default_factory
        and type(driver) is _CupyCudaDriver
        and driver._factory_attestation is _DEFAULT_DRIVER_ATTESTATION
    )
    if selected_default_factory and not audited_default_driver:
        raise RuntimeError(
            "default CUDA factory did not return its attested CuPy driver"
        )
    ordering_trace.append("driver-created")
    compilation = driver.compile(
        CUDA_STRUCTURAL_KERNEL_SOURCE,
        CUDA_STRICT_ARITHMETIC_OPTIONS,
        CUDA_STRUCTURAL_KERNEL_NAME,
    )
    receipt = CudaKernelBuildReceipt.from_compilation(
        compilation,
        audited_default_driver=audited_default_driver,
    )
    ordering_trace.append("kernel-compiled-and-loaded")
    selected_stream = driver.current_stream(stream)
    stream_pointer = driver.stream_pointer(selected_stream)
    ordering_trace.append("stream-selected")
    word_count = plan.total_nbytes // FLOAT32_ITEMSIZE
    arena = driver.allocate_words(word_count)
    arena_receipt = driver.describe_arena(
        arena,
        word_count,
        audited=audited_default_driver,
    )
    ordering_trace.append("arena-allocated-and-validated")

    fields: list[CudaStructuralField] = []
    for field, launch in zip(plan.fields, launches, strict=True):
        array = driver.field_view(
            arena,
            launch.word_offset,
            launch.element_count,
            launch.allocated_shape_zyx,
        )
        view_receipt = driver.describe_field_view(
            array,
            launch,
            audited=audited_default_driver,
        )
        ordering_trace.append(f"field-view-validated:{launch.field_name}")
        driver.launch(compilation.kernel, arena, launch, selected_stream)
        ordering_trace.append(f"launch-enqueued:{launch.field_name}")
        fields.append(
            CudaStructuralField(
                field=field,
                array=array,
                launch=launch,
                view_receipt=view_receipt,
            )
        )

    ready_event_receipt = driver.record_ready_event(
        selected_stream,
        audited=audited_default_driver,
    )
    ordering_trace.append("ready-event-recorded-on-selected-stream")
    if ready_event_receipt.stream_pointer != stream_pointer:
        raise ValueError("driver recorded the ready event on a different stream")
    if audited_default_driver:
        schema = CUDA_STRUCTURAL_STATE_SCHEMA
        claim_limit = CUDA_STRUCTURAL_CLAIM_LIMIT
        synchronization_status = PRODUCTION_SYNC_STATUS
        production_attestation: object | None = _PRODUCTION_ATTESTATION
    else:
        schema = CUDA_STRUCTURAL_TEST_STATE_SCHEMA
        claim_limit = CUDA_STRUCTURAL_TEST_CLAIM_LIMIT
        synchronization_status = TEST_SYNC_STATUS
        production_attestation = None
    return CudaStructuralState(
        plan=plan,
        arena=arena,
        arena_receipt=arena_receipt,
        fields=tuple(fields),
        kernel_receipt=receipt,
        ready_event_receipt=ready_event_receipt,
        stream=selected_stream,
        stream_pointer=stream_pointer,
        ordering_trace=tuple(ordering_trace),
        synchronization_status=synchronization_status,
        audited_default_driver=audited_default_driver,
        schema=schema,
        claim_limit=claim_limit,
        _module_owner=compilation.module_owner,
        _production_attestation=production_attestation,
    )


__all__ = [
    "CUDA_STRICT_ARITHMETIC_OPTIONS",
    "CUDA_STRUCTURAL_CLAIM_LIMIT",
    "CUDA_STRUCTURAL_TEST_CLAIM_LIMIT",
    "CUDA_STRUCTURAL_KERNEL_ABI",
    "CUDA_STRUCTURAL_KERNEL_NAME",
    "CUDA_STRUCTURAL_KERNEL_SOURCE",
    "CUDA_STRUCTURAL_KERNEL_SOURCE_SHA256",
    "CUDA_STRUCTURAL_STATE_SCHEMA",
    "CUDA_STRUCTURAL_TEST_STATE_SCHEMA",
    "CudaArenaReceipt",
    "CudaDriver",
    "CudaFieldLaunch",
    "CudaFieldViewReceipt",
    "CudaKernelBuildReceipt",
    "CudaStructuralField",
    "CudaStructuralState",
    "DEFAULT_THREADS_PER_BLOCK",
    "DriverReadyEvent",
    "DriverCompilation",
    "DriverFactory",
    "allocate_cuda_structural_state",
]
