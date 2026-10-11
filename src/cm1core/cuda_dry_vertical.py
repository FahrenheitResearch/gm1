"""Source-ordered CUDA vertical slice for CM1 21.1 dry ``psolver=2``.

This module implements three consecutive, real CM1 operators for the flat-
terrain dry bootstrap lane:

* dry scalar buoyancy and density-potential-temperature preparation;
* scalar-to-w-face buoyancy accumulation into ``wten``;
* one explicit acoustic vertical-pressure-gradient update of ``w3d``.

The CUDA path is device-resident and asynchronous.  It deliberately does not
choose ``nsound``, advance pressure, fill boundaries, or claim a native oracle
comparison.  Those are coupled gates, not properties that can be inferred
from this isolated operator.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import dataclass
from typing import Any


CUDA_DRY_VERTICAL_SCHEMA = "gpuwm-cm1.cuda-dry-vertical/v1"
CUDA_DRY_VERTICAL_ABI = "gpuwm-cm1.cuda-dry-vertical-kernels/v1"
CUDA_DRY_VERTICAL_CLAIM_LIMIT = (
    "source-ordered CM1 21.1 dry buoyancy plus one flat-terrain psolver=2 "
    "vertical-w acoustic substep; no pressure evolution, boundary fill, "
    "native-oracle parity, coupled RK stage, trajectory, or forecast claim"
)
CUDA_DRY_VERTICAL_THREADS = 256
CM1_HORIZONTAL_HALO = 3
CM1_VERTICAL_HALO = 1
CM1_DEFAULT_GRAVITY_F32 = 9.81
CM1_DEFAULT_CP_F32 = 1005.7

CUDA_DRY_VERTICAL_KERNEL_NAMES = (
    "cm1_dry_thermo_buoyancy_v1",
    "cm1_apply_buoyancy_wten_v1",
    "cm1_psolver2_vertical_w_step_v1",
)
CUDA_STRICT_ARITHMETIC_OPTIONS = (
    "--std=c++14",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)

SOURCE_ANCHORS = {
    "allocation_bounds": "src/cm1.F:744-777",
    "scalar_and_w_allocations": "src/cm1.F:1429-1445;1615-1623;1633-1657",
    "rk_schedule": "src/solve2.F:294-299",
    "dry_thermo_buoyancy": "src/solve2.F:434-449",
    "buoyancy_application": "src/solve2.F:895-921",
    "psolver2_dispatch": "src/solve2.F:1100-1117",
    "psolver2_acoustic_schedule": "src/sounde.F:78-99",
    "psolver2_vertical_w_step": "src/sounde.F:544-564",
    "constants": "src/constants.F:100-120",
}


CUDA_DRY_VERTICAL_SOURCE = r"""// gpuwm-cm1.cuda-dry-vertical-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default-real operators require float32");
static_assert(sizeof(unsigned long long) == 8, "CM1 offsets require uint64");

__device__ __forceinline__ float cm1_add_rn(const float a, const float b)
{
    return __fadd_rn(a, b);
}

__device__ __forceinline__ float cm1_sub_rn(const float a, const float b)
{
    return __fsub_rn(a, b);
}

__device__ __forceinline__ float cm1_mul_rn(const float a, const float b)
{
    return __fmul_rn(a, b);
}

__device__ __forceinline__ unsigned long long cm1_index_zyx(
    const unsigned int i,
    const unsigned int j,
    const unsigned int k,
    const unsigned int x_count,
    const unsigned int y_count,
    const unsigned int i_zero_offset,
    const unsigned int j_zero_offset)
{
    return ((unsigned long long)k * (unsigned long long)y_count
            + (unsigned long long)(j_zero_offset + j))
            * (unsigned long long)x_count
            + (unsigned long long)(i_zero_offset + i);
}

extern "C" __global__
void cm1_dry_thermo_buoyancy_v1(
    float* __restrict__ dum8,
    float* __restrict__ thv,
    const float* __restrict__ th3d,
    const float* __restrict__ rth0,
    const float* __restrict__ th0,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int nk,
    const unsigned int x_count,
    const unsigned int y_count,
    const unsigned int i_zero_offset,
    const unsigned int j_zero_offset,
    const float gravity)
{
    const unsigned long long count =
        (unsigned long long)ni * (unsigned long long)nj * (unsigned long long)nk;
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long linear = first; linear < count; linear += stride) {
        const unsigned int i = (unsigned int)(linear % ni) + 1u;
        const unsigned long long jk = linear / ni;
        const unsigned int j = (unsigned int)(jk % nj) + 1u;
        const unsigned int k = (unsigned int)(jk / nj) + 1u;
        const unsigned long long index = cm1_index_zyx(
            i, j, k, x_count, y_count, i_zero_offset, j_zero_offset);

        // solve2.F:442 is left-associated and dum8 is an array store.
        const float gravity_theta = cm1_mul_rn(gravity, th3d[index]);
        dum8[index] = cm1_mul_rn(gravity_theta, rth0[index]);
        // solve2.F:443; t11 is the thv actual argument passed to sounde.
        thv[index] = cm1_add_rn(th0[index], th3d[index]);
    }
}

extern "C" __global__
void cm1_apply_buoyancy_wten_v1(
    const float* __restrict__ c1,
    const float* __restrict__ c2,
    const float* __restrict__ dum8,
    float* __restrict__ wten,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int nk,
    const unsigned int x_count,
    const unsigned int y_count,
    const unsigned int i_zero_offset,
    const unsigned int j_zero_offset)
{
    const unsigned long long count =
        (unsigned long long)ni * (unsigned long long)nj
        * (unsigned long long)(nk - 1u);
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long linear = first; linear < count; linear += stride) {
        const unsigned int i = (unsigned int)(linear % ni) + 1u;
        const unsigned long long jk = linear / ni;
        const unsigned int j = (unsigned int)(jk % nj) + 1u;
        const unsigned int k = (unsigned int)(jk / nj) + 2u;
        const unsigned long long index = cm1_index_zyx(
            i, j, k, x_count, y_count, i_zero_offset, j_zero_offset);
        const unsigned long long below = cm1_index_zyx(
            i, j, k - 1u, x_count, y_count, i_zero_offset, j_zero_offset);

        // Preserve the parenthesized interpolation in solve2.F:904.
        const float lower = cm1_mul_rn(c1[index], dum8[below]);
        const float upper = cm1_mul_rn(c2[index], dum8[index]);
        const float buoyancy = cm1_add_rn(lower, upper);
        wten[index] = cm1_add_rn(wten[index], buoyancy);
    }
}

extern "C" __global__
void cm1_psolver2_vertical_w_step_v1(
    const float* __restrict__ mf,
    const float* __restrict__ c1,
    const float* __restrict__ c2,
    const float* __restrict__ thv,
    const float* __restrict__ ppd,
    const float* __restrict__ wten,
    float* __restrict__ w3d,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int nk,
    const unsigned int x_count,
    const unsigned int y_count,
    const unsigned int i_zero_offset,
    const unsigned int j_zero_offset,
    const float rdz,
    const float cp,
    const float dts)
{
    const unsigned long long count =
        (unsigned long long)ni * (unsigned long long)nj
        * (unsigned long long)(nk - 1u);
    const unsigned long long first =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * (unsigned long long)gridDim.x;

    for (unsigned long long linear = first; linear < count; linear += stride) {
        const unsigned int i = (unsigned int)(linear % ni) + 1u;
        const unsigned long long jk = linear / ni;
        const unsigned int j = (unsigned int)(jk % nj) + 1u;
        const unsigned int k = (unsigned int)(jk / nj) + 2u;
        const unsigned long long index = cm1_index_zyx(
            i, j, k, x_count, y_count, i_zero_offset, j_zero_offset);
        const unsigned long long below = cm1_index_zyx(
            i, j, k - 1u, x_count, y_count, i_zero_offset, j_zero_offset);
        // sounde.F:553-555 intentionally samples the (1,1,k) metric column.
        const unsigned long long reference = cm1_index_zyx(
            1u, 1u, k, x_count, y_count, i_zero_offset, j_zero_offset);

        const float rdz_cp = cm1_mul_rn(rdz, cp);
        const float tem1 = cm1_mul_rn(rdz_cp, mf[reference]);
        const float upper_theta = cm1_mul_rn(c2[reference], thv[index]);
        const float lower_theta = cm1_mul_rn(c1[reference], thv[below]);
        const float theta_face = cm1_add_rn(upper_theta, lower_theta);
        const float delta_pp = cm1_sub_rn(ppd[index], ppd[below]);
        const float metric_delta_pp = cm1_mul_rn(tem1, delta_pp);
        const float pressure_gradient = cm1_mul_rn(metric_delta_pp, theta_face);
        const float rhs = cm1_sub_rn(wten[index], pressure_gradient);
        const float increment = cm1_mul_rn(dts, rhs);
        w3d[index] = cm1_add_rn(w3d[index], increment);
    }
}
"""
CUDA_DRY_VERTICAL_SOURCE_BYTES = CUDA_DRY_VERTICAL_SOURCE.encode("utf-8")
# Independent review pin.  Source edits must deliberately update this literal.
CUDA_DRY_VERTICAL_SOURCE_SHA256 = (
    "25c1fb11c09e4154cfa54f9f265e8b5ee6ac3cba76c72a034f2618d4828330c4"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_DRY_VERTICAL_SOURCE_BYTES).hexdigest() == (
        CUDA_DRY_VERTICAL_SOURCE_SHA256
    )


def scalar_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    """Return ``(z,y,x)`` for CM1 ``ib:ie,jb:je,kb:ke`` at ngxy=3/ngz=1."""

    _validate_grid(ni, nj, nk)
    return nk + 2, nj + 6, ni + 6


def w_shape_zyx(ni: int, nj: int, nk: int) -> tuple[int, int, int]:
    """Return ``(z,y,x)`` for CM1 ``ib:ie,jb:je,kb:ke+1``."""

    _validate_grid(ni, nj, nk)
    return nk + 3, nj + 6, ni + 6


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value in (("ni", ni), ("nj", nj), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if ni < 1 or nj < 1 or nk < 2:
        raise ValueError("dry vertical slice requires ni,nj >= 1 and nk >= 2")


def rk_stage_dt_f32(dt: float, nrk: int, nrkmax: int = 3) -> float:
    """Transcribe ``solve2.F:299`` with default-real point-of-use rounding."""

    if isinstance(nrk, bool) or not isinstance(nrk, int):
        raise TypeError("nrk must be an integer")
    if nrkmax != 3 or nrk not in (1, 2, 3):
        raise ValueError("the v1 scheduler is sealed to nrkmax=3 and nrk=1..3")
    np = importlib.import_module("numpy")
    return float(np.float32(np.float32(dt) / np.float32(nrkmax + 1 - nrk)))


def psolver2_acoustic_schedule_f32(
    dt: float,
    nsound: int,
    nrk: int,
) -> tuple[int, float]:
    """Transcribe ``sounde.F:78-99`` for a supplied diagnostic ``nsound``."""

    if isinstance(nsound, bool) or not isinstance(nsound, int) or nsound < 1:
        raise ValueError("nsound must be a positive integer")
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (1, 2, 3):
        raise ValueError("nrk must be one of 1, 2, 3")
    np = importlib.import_module("numpy")
    dt32 = np.float32(dt)
    if nrk == 1:
        # Fortran NINT is nearest integer, ties away from zero. nsound is positive.
        nloop = int(float(np.float32(nsound) / np.float32(3.0)) + 0.5)
        dts = np.float32(dt32 / np.float32(nloop * 3))
        if bool(dts > np.float32(dt32 / np.float32(nsound))):
            nloop += 1
            dts = np.float32(dt32 / np.float32(nloop * 3))
    elif nrk == 2:
        # Integer assignment of 0.5*nsound truncates toward zero.
        nloop = int(float(np.float32(0.5) * np.float32(nsound)))
        if nloop < 1:
            raise ValueError("nrk=2 requires nsound >= 2")
        dts = np.float32(dt32 / np.float32(nsound))
    else:
        nloop = nsound
        dts = np.float32(dt32 / np.float32(nsound))
    return nloop, float(dts)


def _require_host_array(
    name: str,
    value: Any,
    shape: tuple[int, int, int],
    *,
    writable: bool,
) -> Any:
    np = importlib.import_module("numpy")
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, received {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def cpu_dry_thermo_buoyancy_inplace(
    dum8: Any,
    thv: Any,
    th3d: Any,
    rth0: Any,
    th0: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    gravity: float = CM1_DEFAULT_GRAVITY_F32,
) -> None:
    """Exact scalar-loop reference for ``solve2.F:439-447``."""

    shape = scalar_shape_zyx(ni, nj, nk)
    dum8 = _require_host_array("dum8", dum8, shape, writable=True)
    thv = _require_host_array("thv", thv, shape, writable=True)
    th3d = _require_host_array("th3d", th3d, shape, writable=False)
    rth0 = _require_host_array("rth0", rth0, shape, writable=False)
    th0 = _require_host_array("th0", th0, shape, writable=False)
    np = importlib.import_module("numpy")
    gravity32 = np.float32(gravity)
    for k in range(1, nk + 1):
        for j in range(1, nj + 1):
            y = j + 2
            for i in range(1, ni + 1):
                x = i + 2
                gravity_theta = np.float32(gravity32 * th3d[k, y, x])
                dum8[k, y, x] = np.float32(gravity_theta * rth0[k, y, x])
                thv[k, y, x] = np.float32(th0[k, y, x] + th3d[k, y, x])


def cpu_apply_buoyancy_wten_inplace(
    c1: Any,
    c2: Any,
    dum8: Any,
    wten: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
) -> None:
    """Exact scalar-loop reference for ``solve2.F:899-907``."""

    scalar_shape = scalar_shape_zyx(ni, nj, nk)
    w_shape = w_shape_zyx(ni, nj, nk)
    c1 = _require_host_array("c1", c1, scalar_shape, writable=False)
    c2 = _require_host_array("c2", c2, scalar_shape, writable=False)
    dum8 = _require_host_array("dum8", dum8, scalar_shape, writable=False)
    wten = _require_host_array("wten", wten, w_shape, writable=True)
    np = importlib.import_module("numpy")
    for k in range(2, nk + 1):
        for j in range(1, nj + 1):
            y = j + 2
            for i in range(1, ni + 1):
                x = i + 2
                lower = np.float32(c1[k, y, x] * dum8[k - 1, y, x])
                upper = np.float32(c2[k, y, x] * dum8[k, y, x])
                buoyancy = np.float32(lower + upper)
                wten[k, y, x] = np.float32(wten[k, y, x] + buoyancy)


def cpu_psolver2_vertical_w_step_inplace(
    mf: Any,
    c1: Any,
    c2: Any,
    thv: Any,
    ppd: Any,
    wten: Any,
    w3d: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    rdz: float,
    cp: float,
    dts: float,
) -> None:
    """Exact scalar-loop reference for flat ``sounde.F:550-563``."""

    scalar_shape = scalar_shape_zyx(ni, nj, nk)
    w_shape = w_shape_zyx(ni, nj, nk)
    mf = _require_host_array("mf", mf, w_shape, writable=False)
    c1 = _require_host_array("c1", c1, scalar_shape, writable=False)
    c2 = _require_host_array("c2", c2, scalar_shape, writable=False)
    thv = _require_host_array("thv", thv, scalar_shape, writable=False)
    ppd = _require_host_array("ppd", ppd, scalar_shape, writable=False)
    wten = _require_host_array("wten", wten, w_shape, writable=False)
    w3d = _require_host_array("w3d", w3d, w_shape, writable=True)
    np = importlib.import_module("numpy")
    rdz32 = np.float32(rdz)
    cp32 = np.float32(cp)
    dts32 = np.float32(dts)
    ref_y = 3
    ref_x = 3
    for k in range(2, nk + 1):
        tem1 = np.float32(np.float32(rdz32 * cp32) * mf[k, ref_y, ref_x])
        c1a = c1[k, ref_y, ref_x]
        c2a = c2[k, ref_y, ref_x]
        for j in range(1, nj + 1):
            y = j + 2
            for i in range(1, ni + 1):
                x = i + 2
                upper_theta = np.float32(c2a * thv[k, y, x])
                lower_theta = np.float32(c1a * thv[k - 1, y, x])
                theta_face = np.float32(upper_theta + lower_theta)
                delta_pp = np.float32(ppd[k, y, x] - ppd[k - 1, y, x])
                metric_delta_pp = np.float32(tem1 * delta_pp)
                pressure_gradient = np.float32(metric_delta_pp * theta_face)
                rhs = np.float32(wten[k, y, x] - pressure_gradient)
                increment = np.float32(dts32 * rhs)
                w3d[k, y, x] = np.float32(w3d[k, y, x] + increment)


def cpu_dry_vertical_slice_inplace(
    dum8: Any,
    thv: Any,
    th3d: Any,
    rth0: Any,
    th0: Any,
    c1: Any,
    c2: Any,
    mf: Any,
    ppd: Any,
    wten: Any,
    w3d: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    gravity: float,
    rdz: float,
    cp: float,
    dts: float,
) -> None:
    """Run the three source-ordered CPU reference operators in sequence."""

    cpu_dry_thermo_buoyancy_inplace(
        dum8, thv, th3d, rth0, th0, ni=ni, nj=nj, nk=nk, gravity=gravity
    )
    cpu_apply_buoyancy_wten_inplace(c1, c2, dum8, wten, ni=ni, nj=nj, nk=nk)
    cpu_psolver2_vertical_w_step_inplace(
        mf,
        c1,
        c2,
        thv,
        ppd,
        wten,
        w3d,
        ni=ni,
        nj=nj,
        nk=nk,
        rdz=rdz,
        cp=cp,
        dts=dts,
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


def _image_options(
    arch: str, nvrtc_major: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    common = CUDA_STRICT_ARITHMETIC_OPTIONS
    if nvrtc_major >= 12:
        common += ("--device-as-default-execution-space",)
    return (
        common + (f"--gpu-architecture=compute_{arch}",),
        common + (f"--gpu-architecture=sm_{arch}",),
    )


@dataclass(frozen=True, slots=True)
class CompiledDryVerticalKernels:
    module_owner: Any
    kernels: tuple[Any, Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CudaDryVerticalReceipt:
    ni: int
    nj: int
    nk: int
    gravity_f32: float
    rdz_f32: float
    cp_f32: float
    dts_f32: float
    source_sha256: str
    ptx_sha256: str
    cubin_sha256: str
    ptx_bytes: int
    cubin_bytes: int
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]
    stream_pointer: int
    event_pointer: int
    grid_scalar_x: int
    grid_w_x: int
    block_x: int = CUDA_DRY_VERTICAL_THREADS
    schema: str = CUDA_DRY_VERTICAL_SCHEMA
    claim_limit: str = CUDA_DRY_VERTICAL_CLAIM_LIMIT

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_limit": self.claim_limit,
            "source_anchors": dict(SOURCE_ANCHORS),
            "domain": {"ni": self.ni, "nj": self.nj, "nk": self.nk},
            "parameters_f32": {
                "gravity": self.gravity_f32,
                "rdz": self.rdz_f32,
                "cp": self.cp_f32,
                "dts": self.dts_f32,
            },
            "kernel": {
                "abi": CUDA_DRY_VERTICAL_ABI,
                "names": list(CUDA_DRY_VERTICAL_KERNEL_NAMES),
                "source_sha256": self.source_sha256,
                "ptx": {
                    "bytes": self.ptx_bytes,
                    "sha256": self.ptx_sha256,
                    "options": list(self.ptx_options),
                },
                "cubin": {
                    "bytes": self.cubin_bytes,
                    "sha256": self.cubin_sha256,
                    "options": list(self.cubin_options),
                },
                "target_arch": self.target_arch,
            },
            "launch": {
                "block_x": self.block_x,
                "grid_scalar_x": self.grid_scalar_x,
                "grid_w_x": self.grid_w_x,
                "stream_pointer": self.stream_pointer,
                "event_pointer": self.event_pointer,
            },
            "runtime": dict(self.runtime_identity),
            "explicit_gaps": [
                "diagnostic nsound/dtsmall decision",
                "sounde horizontal momentum and pressure evolution",
                "acoustic-loop boundary exchange/fills",
                "later low-storage RK carrier updates",
                "native two-build operator oracle and negative-control deck",
                "product psolver=3 vertically implicit solve",
            ],
        }


@dataclass(frozen=True, slots=True)
class CudaDryVerticalLaunch:
    ready_event: Any
    receipt: CudaDryVerticalReceipt
    compiled: CompiledDryVerticalKernels


def compile_cuda_dry_vertical() -> CompiledDryVerticalKernels:
    """Compile and load the pinned kernels on the current CUDA device."""

    if not _source_pin_is_valid():
        raise RuntimeError("packaged dry-vertical CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    compiler_arch = _canonical_arch(compiler._get_arch())
    if compiler_arch != arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc_parts = tuple(compiler._get_nvrtc_version())
    if len(nvrtc_parts) < 2:
        raise ValueError("NVRTC did not report major/minor version")
    ptx_options, cubin_options = _image_options(arch, int(nvrtc_parts[0]))
    ptx_program = compiler._NVRTCProgram(
        CUDA_DRY_VERTICAL_SOURCE, "cm1_dry_vertical_v1_ptx.cu", method="ptx"
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_DRY_VERTICAL_SOURCE, "cm1_dry_vertical_v1_cubin.cu", method="cubin"
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
    ptx_text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text)
        for name in CUDA_DRY_VERTICAL_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks one or more dry-vertical entry points")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled dry-vertical image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name) for name in CUDA_DRY_VERTICAL_KERNEL_NAMES
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
    return CompiledDryVerticalKernels(
        module_owner=module,
        kernels=kernels,
        ptx=ptx,
        cubin=cubin,
        target_arch=arch,
        ptx_options=ptx_options,
        cubin_options=cubin_options,
        runtime_identity=runtime_identity,
    )


def _require_device_array(
    cupy: Any,
    name: str,
    value: Any,
    shape: tuple[int, int, int],
) -> Any:
    if not isinstance(value, cupy.ndarray):
        raise TypeError(f"{name} must be a cupy.ndarray")
    if value.dtype != cupy.dtype("float32"):
        raise TypeError(f"{name} must have dtype float32")
    if value.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, received {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C contiguous")
    return value


def launch_cuda_dry_vertical_slice(
    dum8: Any,
    thv: Any,
    th3d: Any,
    rth0: Any,
    th0: Any,
    c1: Any,
    c2: Any,
    mf: Any,
    ppd: Any,
    wten: Any,
    w3d: Any,
    *,
    ni: int,
    nj: int,
    nk: int,
    gravity: float,
    rdz: float,
    cp: float,
    dts: float,
    stream: Any | None = None,
    compiled: CompiledDryVerticalKernels | None = None,
) -> CudaDryVerticalLaunch:
    """Enqueue the three dry-vertical kernels without synchronizing or host copies."""

    _validate_grid(ni, nj, nk)
    cupy = importlib.import_module("cupy")
    scalar_shape = scalar_shape_zyx(ni, nj, nk)
    w_shape = w_shape_zyx(ni, nj, nk)
    arrays = {
        "dum8": _require_device_array(cupy, "dum8", dum8, scalar_shape),
        "thv": _require_device_array(cupy, "thv", thv, scalar_shape),
        "th3d": _require_device_array(cupy, "th3d", th3d, scalar_shape),
        "rth0": _require_device_array(cupy, "rth0", rth0, scalar_shape),
        "th0": _require_device_array(cupy, "th0", th0, scalar_shape),
        "c1": _require_device_array(cupy, "c1", c1, scalar_shape),
        "c2": _require_device_array(cupy, "c2", c2, scalar_shape),
        "mf": _require_device_array(cupy, "mf", mf, w_shape),
        "ppd": _require_device_array(cupy, "ppd", ppd, scalar_shape),
        "wten": _require_device_array(cupy, "wten", wten, w_shape),
        "w3d": _require_device_array(cupy, "w3d", w3d, w_shape),
    }
    pointers = [int(array.data.ptr) for array in arrays.values()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("dry-vertical v1 arrays must not alias")
    if compiled is None:
        compiled = compile_cuda_dry_vertical()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    scalar_count = ni * nj * nk
    w_count = ni * nj * (nk - 1)
    grid_scalar = (
        scalar_count + CUDA_DRY_VERTICAL_THREADS - 1
    ) // CUDA_DRY_VERTICAL_THREADS
    grid_w = (w_count + CUDA_DRY_VERTICAL_THREADS - 1) // CUDA_DRY_VERTICAL_THREADS
    u32 = cupy.uint32
    f32 = cupy.float32
    common = (
        u32(ni),
        u32(nj),
        u32(nk),
        u32(scalar_shape[2]),
        u32(scalar_shape[1]),
        u32(CM1_HORIZONTAL_HALO - 1),
        u32(CM1_HORIZONTAL_HALO - 1),
    )
    compiled.kernels[0](
        (grid_scalar, 1, 1),
        (CUDA_DRY_VERTICAL_THREADS, 1, 1),
        (
            arrays["dum8"],
            arrays["thv"],
            arrays["th3d"],
            arrays["rth0"],
            arrays["th0"],
            *common,
            f32(gravity),
        ),
        shared_mem=0,
        stream=stream,
    )
    compiled.kernels[1](
        (grid_w, 1, 1),
        (CUDA_DRY_VERTICAL_THREADS, 1, 1),
        (
            arrays["c1"],
            arrays["c2"],
            arrays["dum8"],
            arrays["wten"],
            *common,
        ),
        shared_mem=0,
        stream=stream,
    )
    compiled.kernels[2](
        (grid_w, 1, 1),
        (CUDA_DRY_VERTICAL_THREADS, 1, 1),
        (
            arrays["mf"],
            arrays["c1"],
            arrays["c2"],
            arrays["thv"],
            arrays["ppd"],
            arrays["wten"],
            arrays["w3d"],
            *common,
            f32(rdz),
            f32(cp),
            f32(dts),
        ),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = CudaDryVerticalReceipt(
        ni=ni,
        nj=nj,
        nk=nk,
        gravity_f32=float(f32(gravity)),
        rdz_f32=float(f32(rdz)),
        cp_f32=float(f32(cp)),
        dts_f32=float(f32(dts)),
        source_sha256=CUDA_DRY_VERTICAL_SOURCE_SHA256,
        ptx_sha256=hashlib.sha256(compiled.ptx).hexdigest(),
        cubin_sha256=hashlib.sha256(compiled.cubin).hexdigest(),
        ptx_bytes=len(compiled.ptx),
        cubin_bytes=len(compiled.cubin),
        target_arch=compiled.target_arch,
        ptx_options=compiled.ptx_options,
        cubin_options=compiled.cubin_options,
        runtime_identity=compiled.runtime_identity,
        stream_pointer=int(stream.ptr),
        event_pointer=int(event.ptr),
        grid_scalar_x=grid_scalar,
        grid_w_x=grid_w,
    )
    return CudaDryVerticalLaunch(
        ready_event=event,
        receipt=receipt,
        compiled=compiled,
    )


__all__ = [
    "CM1_DEFAULT_CP_F32",
    "CM1_DEFAULT_GRAVITY_F32",
    "CUDA_DRY_VERTICAL_ABI",
    "CUDA_DRY_VERTICAL_CLAIM_LIMIT",
    "CUDA_DRY_VERTICAL_KERNEL_NAMES",
    "CUDA_DRY_VERTICAL_SCHEMA",
    "CUDA_DRY_VERTICAL_SOURCE",
    "CUDA_DRY_VERTICAL_SOURCE_SHA256",
    "CompiledDryVerticalKernels",
    "CudaDryVerticalLaunch",
    "CudaDryVerticalReceipt",
    "SOURCE_ANCHORS",
    "compile_cuda_dry_vertical",
    "cpu_apply_buoyancy_wten_inplace",
    "cpu_dry_thermo_buoyancy_inplace",
    "cpu_dry_vertical_slice_inplace",
    "cpu_psolver2_vertical_w_step_inplace",
    "launch_cuda_dry_vertical_slice",
    "psolver2_acoustic_schedule_f32",
    "rk_stage_dt_f32",
    "scalar_shape_zyx",
    "w_shape_zyx",
]
