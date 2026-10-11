"""CM1 r21.1 Deardorff SGS/TKE coefficient and local-source CUDA slice.

The sealed product branch is the single-length-scale LES configuration used by
the real 100 m model: ``cm1setup=1``, ``sgsmodel=1``, ``tconfig=1``, free-slip
lower and closed/free-slip upper boundaries.  This module owns the local
``tkekm`` algebra, the selected ``tkebc`` lower face, and the pre-``turbt``
TKE source.  Deformation, buoyancy frequency, and diffusive flux divergences
remain separate operators and are deliberately not inferred here.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from cm1core.errors import refuse


CUDA_SGS_TKE_SCHEMA = "gpuwm-cm1.cuda-sgs-tke/v1"
CUDA_SGS_TKE_ABI = "gpuwm-cm1.cuda-sgs-tke-kernels/v1"
CUDA_SGS_TKE_THREADS = 256
CUDA_SGS_TKE_KERNEL_NAMES = (
    "cm1_sgs_tke_interior_v1",
    "cm1_sgs_tke_lower_free_slip_v1",
)
CUDA_SGS_TKE_CLAIM_LIMIT = (
    "CM1 r21.1 Cartesian cm1setup=1/sgsmodel=1/tconfig=1 local Deardorff "
    "coefficient, dissipation, lower free-slip face, and pre-turbt TKE-source "
    "algebra; no calcnm, calcdef, halo exchange, turbulent flux divergence, "
    "implicit vertical diffusion, RK update, microphysics, or forecast claim"
)

SOURCE_ANCHORS = {
    "constants": "src/constants.F:14-52",
    "coefficient_initialization": "src/init3d.F:2610-2683",
    "tkekm": "src/turb.F:2977-3211",
    "tkebc": "src/turb.F:3219-3367",
    "tke_source": "src/solve1.F:314-369",
}


@dataclass(frozen=True, slots=True)
class SgsTkeSelection:
    cm1setup: int = 1
    sgsmodel: int = 1
    tconfig: int = 1
    bbc: int = 1
    tbc: int = 1
    axisymmetric: int = 0
    horizturb: int = 0
    doimpl: int = 1


DEFAULT_SGS_TKE_SELECTION = SgsTkeSelection()

_EXPECTED_SELECTION = {
    "physics.cm1setup": ("cm1setup", 1),
    "physics.sgs_model": ("sgsmodel", 1),
    "physics.tke_length_scale": ("tconfig", 1),
    "boundaries.lower": ("bbc", 1),
    "boundaries.upper": ("tbc", 1),
    "grid.axisymmetric": ("axisymmetric", 0),
    "physics.horizontal_turbulence": ("horizturb", 0),
    "physics.vertical_implicit_diffusion": ("doimpl", 1),
}

INPUT_ARRAY_NAMES = ("ruh", "rvh", "rmf", "nm", "defv", "defh")
MUTABLE_ARRAY_NAMES = (
    "tkea",
    "tk",
    "grdscl",
    "rgrdscl",
    "lenscl",
    "dissten",
    "cme",
    "ce1",
    "ce2",
    "csm",
    "kmh",
    "kmv",
    "khh",
    "khv",
    "tketen",
)
ALL_ARRAY_NAMES = INPUT_ARRAY_NAMES + MUTABLE_ARRAY_NAMES


def _validate_selection(selection: SgsTkeSelection) -> None:
    if not isinstance(selection, SgsTkeSelection):
        raise TypeError("selection must be an SgsTkeSelection")
    for option, (attribute, expected) in _EXPECTED_SELECTION.items():
        value = getattr(selection, attribute)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            refuse(
                code="CM1-SGS-TKE-UNSUPPORTED",
                gate="physics-seam",
                option=option,
                value=value,
                reason=(
                    f"the sealed SGS/TKE slice covers only {option}={expected}; "
                    "another value selects different CM1 source branches"
                ),
                remedy="use the real-product branch or add a separate native oracle",
            )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 1), ("nj", nj, 1), ("nk", nk, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ni * nj * (nk + 1) > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("SGS/TKE word count exceeds signed 64-bit indexing")


def _validated_spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def sgs_tke_array_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    cell = (nk + 1, nj, ni)
    return {"ruh": (ni,), "rvh": (nj,), **{name: cell for name in ALL_ARRAY_NAMES if name not in {"ruh", "rvh"}}}


def _require_host_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = sgs_tke_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing SGS/TKE arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        if name in MUTABLE_ARRAY_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("SGS/TKE arrays must not alias")
    return result


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _f32_sub(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) - np.float32(right))


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def _f32_div(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) / np.float32(right))


def _f32_sqrt(np: Any, value: Any) -> Any:
    return np.float32(np.sqrt(np.float32(value)))


def _f32_pow(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.power(np.float32(left), np.float32(right)))


def sgs_constants_f32() -> dict[str, float]:
    """Return the selected startup constants in CM1 default-REAL order."""

    np = importlib.import_module("numpy")
    cme = np.float32(0.10)
    c_l = np.float32(0.8165)
    ri_c = np.float32(0.25)
    pi = np.float32(3.1415926535897932384626433)
    ce1 = _f32_mul(np, cme, c_l)
    ce1 = _f32_mul(np, ce1, c_l)
    ce1 = _f32_mul(
        np, ce1, _f32_sub(np, _f32_div(np, np.float32(1.0), ri_c), np.float32(1.0))
    )
    ce2 = _f32_sub(np, _f32_mul(np, _f32_mul(np, cme, pi), pi), ce1)
    ce2 = np.maximum(np.float32(0.0), ce2).astype(np.float32)
    numerator = _f32_mul(np, _f32_mul(np, cme, cme), cme)
    csm = _f32_pow(
        np,
        _f32_div(np, numerator, _f32_add(np, ce1, ce2)),
        np.float32(0.25),
    )
    return {name: float(value) for name, value in {"cme": cme, "ce1": ce1, "ce2": ce2, "csm": csm}.items()}


@dataclass(frozen=True, slots=True)
class CpuSgsTkeResult:
    interior_words: int
    minimum_grid_scale: float
    maximum_grid_scale: float
    clipped_tke_words: int
    stable_length_words: int


def cpu_sgs_tke_inplace(
    arrays: MutableMapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: SgsTkeSelection = DEFAULT_SGS_TKE_SELECTION,
) -> CpuSgsTkeResult:
    """Apply the selected local ``tkekm``/``tkebc``/TKE-source algebra."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx_value = _validated_spacing("dx", dx)
    dy_value = _validated_spacing("dy", dy)
    dz_value = _validated_spacing("dz", dz)
    a = _require_host_arrays(arrays, ni, nj, nk)
    np = importlib.import_module("numpy")
    constants = {name: np.float32(value) for name, value in sgs_constants_f32().items()}
    c_l = np.float32(0.8165)
    exponent = np.float32(0.33333333)
    tke_min = np.float32(1.0e-10)
    nm_min = np.float32(1.0e-6)
    small_len_frac = np.float32(0.001)
    dx_f32 = np.float32(dx_value)
    dy_f32 = np.float32(dy_value)
    dz_f32 = np.float32(dz_value)
    clipped = 0
    stable = 0
    minimum = math.inf
    maximum = -math.inf
    for k in range(1, nk):
        for j in range(nj):
            for i in range(ni):
                grid = _f32_mul(np, dx_f32, a["ruh"][i])
                grid = _f32_mul(np, grid, _f32_mul(np, dy_f32, a["rvh"][j]))
                grid = _f32_mul(np, grid, _f32_mul(np, dz_f32, a["rmf"][k, j, i]))
                grid = _f32_pow(np, grid, exponent)
                if not np.isfinite(grid) or grid <= np.float32(0.0):
                    raise ValueError("computed SGS grid scale must be finite and positive")
                reciprocal = _f32_div(np, np.float32(1.0), grid)
                if a["tkea"][k, j, i] <= tke_min:
                    tk = tke_min
                    length = _f32_mul(np, small_len_frac, grid)
                    clipped += 1
                else:
                    tk = np.float32(a["tkea"][k, j, i])
                    length = grid
                    if a["nm"][k, j, i] > nm_min:
                        length = _f32_mul(
                            np,
                            c_l,
                            _f32_sqrt(np, _f32_div(np, tk, a["nm"][k, j, i])),
                        )
                        length = np.minimum(length, grid).astype(np.float32)
                        length = np.maximum(
                            length, _f32_mul(np, small_len_frac, grid)
                        ).astype(np.float32)
                        stable += 1
                a["grdscl"][k, j, i] = grid
                a["rgrdscl"][k, j, i] = reciprocal
                a["tk"][k, j, i] = tk
                a["lenscl"][k, j, i] = length
                for name in ("cme", "ce1", "ce2", "csm"):
                    a[name][k, j, i] = constants[name]
                sqrt_tk = _f32_sqrt(np, tk)
                dissipation_factor = _f32_mul(np, constants["ce2"], length)
                dissipation_factor = _f32_mul(np, dissipation_factor, reciprocal)
                dissipation_factor = _f32_add(
                    np, constants["ce1"], dissipation_factor
                )
                dissipation = _f32_mul(np, dissipation_factor, tk)
                dissipation = _f32_mul(np, dissipation, sqrt_tk)
                dissipation = _f32_div(np, dissipation, length)
                a["dissten"][k, j, i] = _f32_add(
                    np, a["dissten"][k, j, i], dissipation
                )
                km = _f32_mul(np, constants["cme"], sqrt_tk)
                km = _f32_mul(np, km, length)
                a["kmh"][k, j, i] = km
                a["kmv"][k, j, i] = km
                prinv = np.float32(3.0)
                if a["nm"][k, j, i] > nm_min:
                    candidate = _f32_add(
                        np,
                        np.float32(1.0),
                        _f32_mul(
                            np,
                            np.float32(2.0),
                            _f32_mul(np, length, reciprocal),
                        ),
                    )
                    prinv = np.minimum(candidate, np.float32(3.0)).astype(np.float32)
                kh = _f32_mul(np, km, prinv)
                a["khh"][k, j, i] = kh
                a["khv"][k, j, i] = kh
                buoyancy = _f32_mul(np, -kh, a["nm"][k, j, i])
                source = _f32_sub(np, buoyancy, a["dissten"][k, j, i])
                deformation = _f32_add(
                    np, a["defv"][k, j, i], a["defh"][k, j, i]
                )
                deformation = np.maximum(np.float32(0.0), deformation).astype(
                    np.float32
                )
                source = _f32_add(
                    np, source, _f32_mul(np, km, deformation)
                )
                a["tketen"][k, j, i] = source
                minimum = min(minimum, float(grid))
                maximum = max(maximum, float(grid))
    # tkebc free-slip lower face.  Keep CM1's explicit ratio arithmetic even
    # though the selected km value makes both heat coefficients exactly zero.
    for j in range(nj):
        for i in range(ni):
            a["tkea"][0, j, i] = np.float32(0.0)
            a["kmh"][0, j, i] = np.float32(0.0)
            a["kmv"][0, j, i] = a["kmh"][0, j, i]
            ratio = _f32_div(
                np,
                a["khv"][1, j, i],
                _f32_add(np, np.float32(1.0e-10), a["kmv"][1, j, i]),
            )
            a["khv"][0, j, i] = _f32_mul(np, a["kmv"][0, j, i], ratio)
            ratio = _f32_div(
                np,
                a["khh"][1, j, i],
                _f32_add(np, np.float32(1.0e-10), a["kmh"][1, j, i]),
            )
            a["khh"][0, j, i] = _f32_mul(np, a["kmh"][0, j, i], ratio)
    return CpuSgsTkeResult(
        interior_words=ni * nj * (nk - 1),
        minimum_grid_scale=minimum,
        maximum_grid_scale=maximum,
        clipped_tke_words=clipped,
        stable_length_words=stable,
    )


CUDA_SGS_TKE_SOURCE = r"""// gpuwm-cm1.cuda-sgs-tke-kernels/v1
static_assert(sizeof(float) == 4, "CM1 default-real operators require float32");
static_assert(sizeof(unsigned long long) == 8, "CM1 offsets require uint64");

__device__ __forceinline__ float cm1_add(const float a, const float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float cm1_sub(const float a, const float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float cm1_mul(const float a, const float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float cm1_div(const float a, const float b) { return __fdiv_rn(a, b); }

extern "C" __global__ void cm1_sgs_tke_interior_v1(
    const float* __restrict__ ruh,
    const float* __restrict__ rvh,
    const float* __restrict__ rmf,
    const float* __restrict__ nm,
    const float* __restrict__ defv,
    const float* __restrict__ defh,
    float* __restrict__ tkea,
    float* __restrict__ tk,
    float* __restrict__ grdscl,
    float* __restrict__ rgrdscl,
    float* __restrict__ lenscl,
    float* __restrict__ dissten,
    float* __restrict__ cme,
    float* __restrict__ ce1,
    float* __restrict__ ce2,
    float* __restrict__ csm,
    float* __restrict__ kmh,
    float* __restrict__ kmv,
    float* __restrict__ khh,
    float* __restrict__ khv,
    float* __restrict__ tketen,
    const unsigned int ni,
    const unsigned int nj,
    const unsigned int nk,
    const float dx,
    const float dy,
    const float dz)
{
    const unsigned long long words = (unsigned long long)ni * nj * (nk - 1u);
    const unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    const unsigned long long nxy = (unsigned long long)ni * nj;
    const float c_l = 0.8165f;
    const float tke_min = 1.0e-10f;
    const float nm_min = 1.0e-6f;
    const float small_len_frac = 0.001f;
    const float cme_value = 0.10f;
    const float ri_c = 0.25f;
    const float pi = 3.1415926535897932384626433f;
    float ce1_value = cm1_mul(cme_value, c_l);
    ce1_value = cm1_mul(ce1_value, c_l);
    ce1_value = cm1_mul(ce1_value, cm1_sub(cm1_div(1.0f, ri_c), 1.0f));
    float ce2_value = cm1_sub(cm1_mul(cm1_mul(cme_value, pi), pi), ce1_value);
    ce2_value = fmaxf(0.0f, ce2_value);
    const float numerator = cm1_mul(cm1_mul(cme_value, cme_value), cme_value);
    const float csm_value = powf(cm1_div(numerator, cm1_add(ce1_value, ce2_value)), 0.25f);
    for (unsigned long long linear = first; linear < words; linear += stride) {
        const unsigned int i = (unsigned int)(linear % ni);
        const unsigned long long jk = linear / ni;
        const unsigned int j = (unsigned int)(jk % nj);
        const unsigned int k = (unsigned int)(jk / nj) + 1u;
        const unsigned long long index = (unsigned long long)k * nxy + (unsigned long long)j * ni + i;
        float grid = cm1_mul(dx, ruh[i]);
        grid = cm1_mul(grid, cm1_mul(dy, rvh[j]));
        grid = cm1_mul(grid, cm1_mul(dz, rmf[index]));
        grid = powf(grid, 0.33333333f);
        const float reciprocal = cm1_div(1.0f, grid);
        float tk_value;
        float length;
        if (tkea[index] <= tke_min) {
            tk_value = tke_min;
            length = cm1_mul(small_len_frac, grid);
        } else {
            tk_value = tkea[index];
            length = grid;
            if (nm[index] > nm_min) {
                length = cm1_mul(c_l, sqrtf(cm1_div(tk_value, nm[index])));
                length = fminf(length, grid);
                length = fmaxf(length, cm1_mul(small_len_frac, grid));
            }
        }
        grdscl[index] = grid;
        rgrdscl[index] = reciprocal;
        tk[index] = tk_value;
        lenscl[index] = length;
        cme[index] = cme_value;
        ce1[index] = ce1_value;
        ce2[index] = ce2_value;
        csm[index] = csm_value;
        const float sqrt_tk = sqrtf(tk_value);
        float factor = cm1_mul(ce2_value, length);
        factor = cm1_mul(factor, reciprocal);
        factor = cm1_add(ce1_value, factor);
        float dissipation = cm1_mul(factor, tk_value);
        dissipation = cm1_mul(dissipation, sqrt_tk);
        dissipation = cm1_div(dissipation, length);
        dissten[index] = cm1_add(dissten[index], dissipation);
        float km = cm1_mul(cme_value, sqrt_tk);
        km = cm1_mul(km, length);
        kmh[index] = km;
        kmv[index] = km;
        float prinv = 3.0f;
        if (nm[index] > nm_min) {
            const float candidate = cm1_add(1.0f, cm1_mul(2.0f, cm1_mul(length, reciprocal)));
            prinv = fminf(candidate, 3.0f);
        }
        const float kh = cm1_mul(km, prinv);
        khh[index] = kh;
        khv[index] = kh;
        const float buoyancy = cm1_mul(-kh, nm[index]);
        float source = cm1_sub(buoyancy, dissten[index]);
        const float deformation = fmaxf(0.0f, cm1_add(defv[index], defh[index]));
        source = cm1_add(source, cm1_mul(km, deformation));
        tketen[index] = source;
    }
}

extern "C" __global__ void cm1_sgs_tke_lower_free_slip_v1(
    float* __restrict__ tkea,
    float* __restrict__ kmh,
    float* __restrict__ kmv,
    float* __restrict__ khh,
    float* __restrict__ khv,
    const unsigned int ni,
    const unsigned int nj)
{
    const unsigned long long words = (unsigned long long)ni * nj;
    const unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long index = first; index < words; index += stride) {
        tkea[index] = 0.0f;
        kmh[index] = 0.0f;
        kmv[index] = kmh[index];
        const unsigned long long above = words + index;
        const float ratio_v = cm1_div(khv[above], cm1_add(1.0e-10f, kmv[above]));
        khv[index] = cm1_mul(kmv[index], ratio_v);
        const float ratio_h = cm1_div(khh[above], cm1_add(1.0e-10f, kmh[above]));
        khh[index] = cm1_mul(kmh[index], ratio_h);
    }
}
"""

CUDA_SGS_TKE_SOURCE_BYTES = CUDA_SGS_TKE_SOURCE.encode("utf-8")
CUDA_SGS_TKE_SOURCE_SHA256 = (
    "b25419495b3d51ce787a02e5d1b6258cfbfb1bd13eb485f003a7db105d438fee"
)


def _source_pin_is_valid() -> bool:
    return hashlib.sha256(CUDA_SGS_TKE_SOURCE_BYTES).hexdigest() == CUDA_SGS_TKE_SOURCE_SHA256


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
    return common + (f"--gpu-architecture=compute_{arch}",), common + (
        f"--gpu-architecture=sm_{arch}",
    )


@dataclass(frozen=True, slots=True)
class CompiledSgsTkeKernels:
    module_owner: Any
    kernels: tuple[Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_sgs_tke() -> CompiledSgsTkeKernels:
    if not _source_pin_is_valid():
        raise RuntimeError("packaged SGS/TKE CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from the current CUDA device")
    nvrtc_parts = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc_parts[0]))
    ptx_program = compiler._NVRTCProgram(
        CUDA_SGS_TKE_SOURCE, "cm1_sgs_tke_v1_ptx.cu", method="ptx"
    )
    ptx, _ = ptx_program.compile(ptx_options)
    cubin_program = compiler._NVRTCProgram(
        CUDA_SGS_TKE_SOURCE, "cm1_sgs_tke_v1_cubin.cu", method="cubin"
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
    if not all(re.search(rf"\.entry\s+{name}(?:\s|\()", ptx_text) for name in CUDA_SGS_TKE_KERNEL_NAMES):
        raise ValueError("compiled PTX lacks one or more SGS/TKE entry points")
    if len(cubin) < 20 or cubin[:7] != b"\x7fELF\x02\x01\x01" or int.from_bytes(cubin[18:20], "little") != 190:
        raise ValueError("compiled SGS/TKE image is not an ELF64 CUDA cubin")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(module.get_function(name) for name in CUDA_SGS_TKE_KERNEL_NAMES)
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
    return CompiledSgsTkeKernels(module, kernels, ptx, cubin, arch, ptx_options, cubin_options, runtime_identity)


def _require_device_arrays(cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int) -> dict[str, Any]:
    shapes = sgs_tke_array_shapes(ni, nj, nk)
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(f"missing SGS/TKE arrays: {', '.join(missing)}")
    result: dict[str, Any] = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"{name} must be a cupy.ndarray")
        if value.dtype != cupy.dtype("float32"):
            raise TypeError(f"{name} must have dtype float32")
        if value.shape != shapes[name]:
            raise ValueError(f"{name} shape must be {shapes[name]}, received {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("SGS/TKE arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaSgsTkeLaunch:
    ready_event: Any
    compiled: CompiledSgsTkeKernels
    receipt: dict[str, Any]


def launch_cuda_sgs_tke(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dz: float,
    selection: SgsTkeSelection = DEFAULT_SGS_TKE_SELECTION,
    stream: Any | None = None,
    compiled: CompiledSgsTkeKernels | None = None,
) -> CudaSgsTkeLaunch:
    """Enqueue the selected local SGS/TKE operator without host synchronization."""

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    dx_value = _validated_spacing("dx", dx)
    dy_value = _validated_spacing("dy", dy)
    dz_value = _validated_spacing("dz", dz)
    cupy = importlib.import_module("cupy")
    a = _require_device_arrays(cupy, arrays, ni, nj, nk)
    if compiled is None:
        compiled = compile_cuda_sgs_tke()
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    interior_words = ni * nj * (nk - 1)
    horizontal_words = ni * nj
    interior_grid = (interior_words + CUDA_SGS_TKE_THREADS - 1) // CUDA_SGS_TKE_THREADS
    horizontal_grid = (horizontal_words + CUDA_SGS_TKE_THREADS - 1) // CUDA_SGS_TKE_THREADS
    u32 = cupy.uint32
    f32 = cupy.float32
    compiled.kernels[0](
        (interior_grid, 1, 1),
        (CUDA_SGS_TKE_THREADS, 1, 1),
        tuple(a[name] for name in ALL_ARRAY_NAMES)
        + (u32(ni), u32(nj), u32(nk), f32(dx_value), f32(dy_value), f32(dz_value)),
        shared_mem=0,
        stream=stream,
    )
    compiled.kernels[1](
        (horizontal_grid, 1, 1),
        (CUDA_SGS_TKE_THREADS, 1, 1),
        (a["tkea"], a["kmh"], a["kmv"], a["khh"], a["khv"], u32(ni), u32(nj)),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_SGS_TKE_SCHEMA,
        "claim_limit": CUDA_SGS_TKE_CLAIM_LIMIT,
        "source_anchors": dict(SOURCE_ANCHORS),
        "selection": {option: getattr(selection, attribute) for option, (attribute, _) in _EXPECTED_SELECTION.items()},
        "domain": {"ni": ni, "nj": nj, "nk": nk, "interior_words": interior_words},
        "spacing_m": {"dx": dx_value, "dy": dy_value, "dz": dz_value},
        "kernel": {
            "abi": CUDA_SGS_TKE_ABI,
            "names": list(CUDA_SGS_TKE_KERNEL_NAMES),
            "source_sha256": CUDA_SGS_TKE_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "ptx_bytes": len(compiled.ptx),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "cubin_bytes": len(compiled.cubin),
            "target_arch": compiled.target_arch,
            "ptx_options": list(compiled.ptx_options),
            "cubin_options": list(compiled.cubin_options),
        },
        "launch": {
            "interior_grid_x": interior_grid,
            "lower_grid_x": horizontal_grid,
            "block_x": CUDA_SGS_TKE_THREADS,
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    return CudaSgsTkeLaunch(event, compiled, receipt)


__all__ = [
    "ALL_ARRAY_NAMES",
    "CUDA_SGS_TKE_ABI",
    "CUDA_SGS_TKE_CLAIM_LIMIT",
    "CUDA_SGS_TKE_KERNEL_NAMES",
    "CUDA_SGS_TKE_SCHEMA",
    "CUDA_SGS_TKE_SOURCE",
    "CUDA_SGS_TKE_SOURCE_SHA256",
    "CompiledSgsTkeKernels",
    "CpuSgsTkeResult",
    "CudaSgsTkeLaunch",
    "DEFAULT_SGS_TKE_SELECTION",
    "INPUT_ARRAY_NAMES",
    "MUTABLE_ARRAY_NAMES",
    "SOURCE_ANCHORS",
    "SgsTkeSelection",
    "compile_cuda_sgs_tke",
    "cpu_sgs_tke_inplace",
    "launch_cuda_sgs_tke",
    "sgs_constants_f32",
    "sgs_tke_array_shapes",
]
