"""Exact child ``calcnm -> calcdef -> turbsmag`` CUDA composition.

The existing :mod:`cm1core.cuda_moist_nm` operator is sealed to the parent
product ABI, where ``pi0`` and ``th0`` are horizontally uniform profiles.
CM1 r21.1 ``calcnm`` actually declares both arrays on the three-dimensional
terrain grid.  The fixed 50 m child therefore uses a separately versioned
kernel whose only mathematical change is the source-authoritative 3-D base
index.  ``calcdef`` and fresh ``turbsmag`` are then enqueued on the same CUDA
stream without a host synchronization.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cm1core.cuda_gm1_child_turbsmag_init import (
    CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256,
    REAL_CHILD_TURBSMAG_METRIC_IDENTITY,
    REAL_CHILD_TURBSMAG_RMF_SHA256,
    CudaChildTurbsmagInitLaunch,
    launch_cuda_child_turbsmag_init,
)
from cm1core.cuda_moist_nm import (
    CUDA_MOIST_NM_SOURCE,
    CUDA_MOIST_NM_SOURCE_SHA256,
    MASK_NAME,
    CpuMoistNmResult,
    MoistNmSelection,
    cpu_moist_nm_inplace,
)
from cm1core.cuda_terrain_deformation import (
    CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
    CudaTerrainDeformationLaunch,
    launch_cuda_terrain_deformation,
)
from cm1core.errors import refuse
from cm1core.gm1_child_atmosphere_admission import LoadedGm1ChildAtmosphere
from cm1core.gm1_child_auxiliary_interfaces import TERRAIN_OPEN_CARRIER_SHAPES
from cm1core.cuda_gm1_child_terrain import CudaGm1ChildTerrainMetricsLaunch


CUDA_GM1_CHILD_NM_DEF_SCHEMA = "gm1.cuda-child-nm-def-tke/v1"
CUDA_GM1_CHILD_MOIST_NM_SCHEMA = "gm1.cuda-child-moist-nm/v1"
CUDA_GM1_CHILD_MOIST_NM_ABI = "gm1.cuda-child-moist-nm-kernels/v1"
CUDA_GM1_CHILD_MOIST_NM_KERNEL_NAME = "gm1_child_moist_nm_columns_v1"
CUDA_GM1_CHILD_MOIST_NM_THREADS = 128
CUDA_GM1_CHILD_NM_DEF_CLAIM_LIMIT = (
    "exact CM1-r21.1-derived three-dimensional-base calcnm, selected serial "
    "terrain/four-open calcdef, and selected fresh child turbsmag/TKE "
    "initialization for the one sealed real fixed 300x300x75 50 m GM1 child; "
    "same-stream device-resident composition only; turbulent tendencies, "
    "microphysics, time integration, forecast skill, storm, and tornado "
    "claims are excluded"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOURCE_SHA256 = MappingProxyType(
    {
        "src/turb.F": (
            "9984ad47b7b5c8f92b8a2e98933c11474b63c7e315773117f4613620a660f04d"
        ),
        "src/cm1libs.F": (
            "830cd004f6e9abaa12b242ad7e330c5370ac4b6ed4450745f7badd2ac2d0a954"
        ),
        "src/constants.F": (
            "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94"
        ),
        "src/param.F": (
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"
        ),
    }
)
SOURCE_ANCHORS = MappingProxyType(
    {
        "driver_cadence": "src/turb.F:1326-1389",
        "calcnm": "src/turb.F:5432-5685",
        "calcnm_3d_base_declaration": "src/turb.F:5440-5444",
        "calcnm_3d_base_use": "src/turb.F:5543-5561",
        "calcdef": "src/turb.F:4711-5423",
        "uniform_horizontal_metrics": "src/param.F:6647-6688,7328-7384,7678-7722",
        "fresh_turbsmag": "src/turb.F:3375-3621; src/init3d.F:2688-2756",
    }
)

UPSTREAM_MOIST_NM_MODULE_SHA256 = (
    "b57859dd514cb25d5a55668b7c64180252fbfe442ead27a8964158799084632f"
)
UPSTREAM_TERRAIN_DEFORMATION_MODULE_SHA256 = (
    "f51053640b30cd5a475790ab175fb41df0a49ecabbc38e30531db4846a4c1ecb"
)
UPSTREAM_CHILD_TURBSMAG_MODULE_SHA256 = (
    "a25c00213b70974f1b2faf52050859a055b83eea61d935f0deb67d99eb0b42f7"
)
UPSTREAM_TERRAIN_DEFORMATION_SOURCE_SHA256 = (
    "e642efc44273461ddfc1934012aaac87bca222a0b5ce5f713438f263d3265c62"
)
UPSTREAM_CHILD_TURBSMAG_SOURCE_SHA256 = (
    "8937435b68d61aa6c602b5f5c37796216e15a61e725b83cc0e4aab3af7eb5ac5"
)
if CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256 != (
    UPSTREAM_TERRAIN_DEFORMATION_SOURCE_SHA256
) or CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256 != (
    UPSTREAM_CHILD_TURBSMAG_SOURCE_SHA256
):
    raise RuntimeError("child calcdef/fresh-TKE operator source identity changed")

GM1_CHILD_AUXILIARY_SCHEMA = "gm1.source-coordinate-50m-child-auxiliary/v1"
GM1_CHILD_AUXILIARY_MANIFEST_SCHEMA = (
    "gm1.source-coordinate-50m-child-auxiliary-manifest/v1"
)
GM1_CHILD_AUXILIARY_RECEIPT_SCHEMA = (
    "gm1.source-coordinate-50m-child-auxiliary-receipt/v1"
)
GM1_CHILD_AUXILIARY_MANIFEST_SHA256 = (
    "24764f50afc5807c37266696f9a2ea5c5cc0989819e2fa238dd9b11df4b1e8e4"
)
GM1_CHILD_AUXILIARY_RECEIPT_SHA256 = (
    "158597b8b9fdb464fad305fb5ebf5faa88326155a84f6685ee3b73bb98681717"
)
GM1_CHILD_AUXILIARY_TERRAIN_OPEN_SHA256 = (
    "62364cf75c8cea33147f3b95ce691a783089ea3f015bef116f1d0c71df5efdb3"
)
GM1_CHILD_AUXILIARY_LIVE_DENSITY_SHA256 = (
    "f1294e18b3f9c6bc3a0163f97caefccd682076f9d122f656c2257e7fad97e460"
)
GM1_CHILD_AUXILIARY_GEOPOTENTIAL_SHA256 = (
    "687bc3ebe34c0c07dc0c46418e8e6042f271e1bb1bcd119284b4a27ef80bca8c"
)
GM1_CHILD_AUXILIARY_PRODUCER_SOURCE_SHA256 = (
    "e0d1cb5a89f95c32aa17fecd0e7ea8026bc33f3bca04f3ccdece958e5de49563"
)
GM1_CHILD_AUXILIARY_PRODUCER_EXECUTABLE_SHA256 = (
    "878c5f7b7e1ce505c0a6c6adc1eac7068f1a8448753fcfd7c8697ada93b72da5"
)
GM1_CHILD_UNIFORM_HORIZONTAL_METRIC_SHA256 = (
    "9daebe2966c6a8065d691f81625f3cc8ba345f05f44b143fc2e31e5c59d72b5c"
)

_AUXILIARY_FILES = frozenset(
    {
        "terrain-open.f32le",
        "live-density.f32le",
        "total-geopotential-zf.f32le",
        "manifest.json",
        "receipt.json",
    }
)
_AUXILIARY_AUTHORITY = {
    "tarball_sha256": AUTHORITY_TARBALL_SHA256,
    "sources": [
        {
            "anchors": "175-297",
            "path": "src/init_terrain.F",
            "sha256": (
                "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c"
            ),
        },
        {
            "anchors": "7789-7812,8162-8207,8464-8492,8988-9062",
            "path": "src/param.F",
            "sha256": AUTHORITY_SOURCE_SHA256["src/param.F"],
        },
        {
            "anchors": "625-749,2044-2243",
            "path": "src/base.F",
            "sha256": (
                "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef"
            ),
        },
        {
            "anchors": "2280-2467",
            "path": "src/init3d.F",
            "sha256": (
                "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28"
            ),
        },
        {
            "anchors": "12-172,180-1206,1948-2172",
            "path": "src/bc.F",
            "sha256": (
                "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
            ),
        },
        {
            "anchors": "1559-1657",
            "path": "src/cm1.F",
            "sha256": (
                "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4"
            ),
        },
    ],
}
_AUXILIARY_MANIFEST_KEYS = {
    "artifact_schema",
    "authority",
    "boundary_initialization",
    "child_grid",
    "claim_limit",
    "closure_disposition",
    "model",
    "outputs",
    "parent",
    "producer",
    "schema",
    "source_atmosphere",
    "source_base_profiles",
    "source_sequence",
    "static_inputs",
    "status",
    "stretched_endpoint_weights",
}
_LIVE_DENSITY_SHAPES = MappingProxyType(
    {
        "prs": (75, 300, 300),
        "rf": (76, 300, 300),
        "rho": (75, 300, 300),
        "rr": (75, 300, 300),
    }
)
_GEOPOTENTIAL_SHAPES = MappingProxyType(
    {"total_geopotential_zf": (76, 300, 300)}
)
# The file name, exact byte count and carrier layout of each auxiliary arena
# for the admitted child grid.  These are the grid contract - every memmap
# below indexes them - and they stay fixed.  Which bytes occupy them is the
# artifact's own business: the digest of each arena is taken from the receipt
# the producer published beside it, which is the key of the auxiliary receipt
# named in the tuple's last slot.
_GROUP_EXPECTATIONS = MappingProxyType(
    {
        "terrain_open": (
            "terrain-open.f32le",
            953_971_136,
            "terrain_open_sha256",
            TERRAIN_OPEN_CARRIER_SHAPES,
        ),
        "live_density": (
            "live-density.f32le",
            108_360_000,
            "live_density_sha256",
            _LIVE_DENSITY_SHAPES,
        ),
        "geopotential": (
            "total-geopotential-zf.f32le",
            27_360_000,
            "geopotential_sha256",
            _GEOPOTENTIAL_SHAPES,
        ),
    }
)


def _derive_child_moist_nm_source() -> str:
    if hashlib.sha256(CUDA_MOIST_NM_SOURCE.encode()).hexdigest() != (
        CUDA_MOIST_NM_SOURCE_SHA256
    ):
        raise RuntimeError("upstream moist-Nm source differs from its source pin")
    replacements = (
        ("gpuwm-cm1.cuda-moist-nm-kernels/v1", CUDA_GM1_CHILD_MOIST_NM_ABI, 1),
        ("cm1_moist_nm_columns_v1", CUDA_GM1_CHILD_MOIST_NM_KERNEL_NAME, 1),
        ("th0[k]", "th0[z]", 1),
        ("pi0[k]", "pi0[z]", 1),
    )
    result = CUDA_MOIST_NM_SOURCE
    for old, new, count in replacements:
        if result.count(old) != count:
            raise RuntimeError(
                f"upstream moist-Nm source expected {count} instance(s) of {old!r}"
            )
        result = result.replace(old, new)
    return result


CUDA_GM1_CHILD_MOIST_NM_SOURCE = _derive_child_moist_nm_source()
CUDA_GM1_CHILD_MOIST_NM_SOURCE_BYTES = CUDA_GM1_CHILD_MOIST_NM_SOURCE.encode()
CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256 = (
    "40ddf9aa767b997f8e6d592ffcc8e7f0007f91e81bc122f0af192511b0295aac"
)
if hashlib.sha256(CUDA_GM1_CHILD_MOIST_NM_SOURCE_BYTES).hexdigest() != (
    CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256
):
    raise RuntimeError("derived child moist-Nm CUDA source differs from its pin")


@dataclass(frozen=True, slots=True)
class ChildMoistNmSelection:
    ptype: int = 5
    imoist: int = 1
    iice: int = 1
    numq: int = 6
    terrain_varying_base: bool = True


DEFAULT_CHILD_MOIST_NM_SELECTION = ChildMoistNmSelection()


def _require_selection(selection: ChildMoistNmSelection) -> None:
    if not isinstance(selection, ChildMoistNmSelection):
        raise TypeError("selection must be ChildMoistNmSelection")
    if selection != DEFAULT_CHILD_MOIST_NM_SELECTION:
        refuse(
            code="GM1-CHILD-MOIST-NM-UNSUPPORTED",
            gate="gm1-child-nm-def",
            option="child_moist_nm_selection",
            value={
                name: getattr(selection, name)
                for name in selection.__dataclass_fields__
            },
            reason=(
                "the sealed child operator covers only the selected WSM6 "
                "six-species, ice-aware calcnm branch with 3-D terrain base"
            ),
            remedy="use the exact fixed child selection or add a new native oracle",
        )


def _validate_grid(ni: int, nj: int, nk: int) -> None:
    for name, value, minimum in (("ni", ni, 1), ("nj", nj, 1), ("nk", nk, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if ni * nj * (nk + 1) > 0x7FFF_FFFF_FFFF_FFFF:
        raise ValueError("child moist-Nm word count exceeds signed 64-bit indexing")


def _spacing(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


CHILD_MOIST_INPUT_NAMES = (
    "c1",
    "c2",
    "mf",
    "pi0",
    "th0",
    "prs",
    "pp3d",
    "th3d",
    "q3d",
)
CHILD_MOIST_FLOAT_OUTPUT_NAMES = ("nm", "temperature", "qt", "thv", "qvci")
CHILD_MOIST_OUTPUT_NAMES = CHILD_MOIST_FLOAT_OUTPUT_NAMES + (MASK_NAME,)
CHILD_MOIST_ALL_NAMES = CHILD_MOIST_INPUT_NAMES + CHILD_MOIST_OUTPUT_NAMES


def child_moist_nm_shapes(ni: int, nj: int, nk: int) -> dict[str, tuple[int, ...]]:
    _validate_grid(ni, nj, nk)
    cell = (nk, nj, ni)
    face = (nk + 1, nj, ni)
    return {
        "c1": face,
        "c2": face,
        "mf": face,
        "pi0": cell,
        "th0": cell,
        "prs": cell,
        "pp3d": cell,
        "th3d": cell,
        "q3d": (6, nk, nj, ni),
        "nm": face,
        "temperature": cell,
        "qt": cell,
        "thv": cell,
        "qvci": cell,
        MASK_NAME: cell,
    }


def _require_host_child_moist_arrays(
    arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    shapes = child_moist_nm_shapes(ni, nj, nk)
    if set(arrays) != set(CHILD_MOIST_ALL_NAMES):
        raise ValueError("child moist-Nm host array inventory changed")
    result: dict[str, Any] = {}
    for name in CHILD_MOIST_ALL_NAMES:
        value = arrays[name]
        expected_dtype = np.dtype("uint8") if name == MASK_NAME else np.dtype("float32")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if value.dtype != expected_dtype:
            raise TypeError(f"{name} must have dtype {expected_dtype.name}")
        if value.shape != shapes[name] or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C contiguous with shape {shapes[name]}")
        if name in CHILD_MOIST_OUTPUT_NAMES and not value.flags.writeable:
            raise ValueError(f"{name} must be writable")
        result[name] = value
    pointers = [int(value.ctypes.data) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("child moist-Nm arrays must not alias")
    return result


def cpu_child_moist_nm_inplace(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
    selection: ChildMoistNmSelection = DEFAULT_CHILD_MOIST_NM_SELECTION,
) -> CpuMoistNmResult:
    """Evaluate the 3-D-base operator columnwise with the sealed CPU oracle.

    This deliberately slow implementation is for compact operator validation;
    production and the full real child use the CUDA path.
    """

    _require_selection(selection)
    _validate_grid(ni, nj, nk)
    dz_value = _spacing("dz", dz)
    np = importlib.import_module("numpy")
    a = _require_host_child_moist_arrays(arrays, ni, nj, nk)
    saturated = liquid = mixed = ice = 0
    minimum, maximum = math.inf, -math.inf
    for j in range(nj):
        for i in range(ni):
            one: dict[str, Any] = {
                "c1": np.ascontiguousarray(a["c1"][:, j : j + 1, i : i + 1]),
                "c2": np.ascontiguousarray(a["c2"][:, j : j + 1, i : i + 1]),
                "mf": np.ascontiguousarray(a["mf"][:, j : j + 1, i : i + 1]),
                "pi0": np.ascontiguousarray(a["pi0"][:, j, i]),
                "th0": np.ascontiguousarray(a["th0"][:, j, i]),
                "prs": np.ascontiguousarray(a["prs"][:, j : j + 1, i : i + 1]),
                "pp3d": np.ascontiguousarray(a["pp3d"][:, j : j + 1, i : i + 1]),
                "th3d": np.ascontiguousarray(a["th3d"][:, j : j + 1, i : i + 1]),
                "q3d": np.ascontiguousarray(a["q3d"][:, :, j : j + 1, i : i + 1]),
            }
            shapes = {
                "nm": (nk + 1, 1, 1),
                "temperature": (nk, 1, 1),
                "qt": (nk, 1, 1),
                "thv": (nk, 1, 1),
                "qvci": (nk, 1, 1),
                MASK_NAME: (nk, 1, 1),
            }
            for name, shape in shapes.items():
                dtype = np.uint8 if name == MASK_NAME else np.float32
                one[name] = np.empty(shape, dtype=dtype)
            result = cpu_moist_nm_inplace(
                one,
                ni=1,
                nj=1,
                nk=nk,
                dz=dz_value,
                selection=MoistNmSelection(),
            )
            for name in CHILD_MOIST_OUTPUT_NAMES:
                a[name][:, j, i] = one[name][:, 0, 0]
            saturated += result.saturated_words
            liquid += result.liquid_words
            mixed += result.mixed_words
            ice += result.ice_words
            minimum = min(minimum, result.minimum_nm)
            maximum = max(maximum, result.maximum_nm)
    return CpuMoistNmResult(
        ni * nj,
        ni * nj * (nk - 1),
        saturated,
        liquid,
        mixed,
        ice,
        minimum,
        maximum,
    )


def _canonical_arch(value: Any) -> str:
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return f"{value[0]}{value[1]}"
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip().lower().removeprefix("compute_").removeprefix("sm_")
    if re.fullmatch(r"[0-9]+\.[0-9]+", text):
        text = text.replace(".", "")
    if re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError("CUDA architecture must be canonical digits")
    return text


@dataclass(frozen=True, slots=True)
class CompiledChildMoistNmKernel:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    ptx_options: tuple[str, ...]
    cubin_options: tuple[str, ...]
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_child_moist_nm() -> CompiledChildMoistNmKernel:
    if hashlib.sha256(CUDA_GM1_CHILD_MOIST_NM_SOURCE_BYTES).hexdigest() != (
        CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256
    ):
        raise RuntimeError("packaged child moist-Nm CUDA source differs from its pin")
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_MOIST_NM_SOURCE,
        "gm1_child_moist_nm_v1_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_MOIST_NM_SOURCE,
        "gm1_child_moist_nm_v1_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if re.search(
        rf"\.entry\s+{CUDA_GM1_CHILD_MOIST_NM_KERNEL_NAME}(?:\s|\()",
        ptx.decode(),
    ) is None:
        raise ValueError("compiled PTX lacks child moist-Nm entry point")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled child moist-Nm image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    runtime = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{nvrtc[0]}.{nvrtc[1]}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(name)),
    )
    return CompiledChildMoistNmKernel(
        module,
        module.get_function(CUDA_GM1_CHILD_MOIST_NM_KERNEL_NAME),
        ptx,
        cubin,
        arch,
        ptx_options,
        cubin_options,
        runtime,
    )


def _require_device_child_moist_arrays(
    cupy: Any, arrays: Mapping[str, Any], ni: int, nj: int, nk: int
) -> dict[str, Any]:
    shapes = child_moist_nm_shapes(ni, nj, nk)
    if set(arrays) != set(CHILD_MOIST_ALL_NAMES):
        raise ValueError("child moist-Nm CUDA array inventory changed")
    result = {}
    for name in CHILD_MOIST_ALL_NAMES:
        value = arrays[name]
        expected_dtype = cupy.dtype("uint8") if name == MASK_NAME else cupy.dtype("float32")
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != expected_dtype
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be C-contiguous CUDA {expected_dtype.name} {shapes[name]}"
            )
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("child moist-Nm CUDA arrays must not alias")
    return result


@dataclass(frozen=True, slots=True)
class CudaChildMoistNmLaunch:
    ready_event: Any
    compiled: CompiledChildMoistNmKernel
    receipt: Mapping[str, Any]


def launch_cuda_child_moist_nm(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dz: float,
    selection: ChildMoistNmSelection = DEFAULT_CHILD_MOIST_NM_SELECTION,
    stream: Any | None = None,
    compiled: CompiledChildMoistNmKernel | None = None,
) -> CudaChildMoistNmLaunch:
    """Enqueue the exact 3-D-base child ``calcnm`` operator."""

    _require_selection(selection)
    _validate_grid(ni, nj, nk)
    dz_value = _spacing("dz", dz)
    cupy = importlib.import_module("cupy")
    a = _require_device_child_moist_arrays(cupy, arrays, ni, nj, nk)
    compiled = compiled or compile_cuda_child_moist_nm()
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    columns = ni * nj
    grid = (columns + CUDA_GM1_CHILD_MOIST_NM_THREADS - 1) // (
        CUDA_GM1_CHILD_MOIST_NM_THREADS
    )
    compiled.kernel(
        (grid, 1, 1),
        (CUDA_GM1_CHILD_MOIST_NM_THREADS, 1, 1),
        tuple(a[name] for name in CHILD_MOIST_ALL_NAMES)
        + (
            cupy.uint32(ni),
            cupy.uint32(nj),
            cupy.uint32(nk),
            cupy.float32(dz_value),
        ),
        shared_mem=0,
        stream=stream,
    )
    event = cupy.cuda.Event(disable_timing=True)
    event.record(stream)
    receipt = {
        "schema": CUDA_GM1_CHILD_MOIST_NM_SCHEMA,
        "abi": CUDA_GM1_CHILD_MOIST_NM_ABI,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": dict(AUTHORITY_SOURCE_SHA256),
        },
        "source_anchors": dict(SOURCE_ANCHORS),
        "selection": {
            name: getattr(selection, name) for name in selection.__dataclass_fields__
        },
        "shape_zyx": [nk, nj, ni],
        "dz_m": dz_value,
        "base_shape_zyx": [nk, nj, ni],
        "kernel": {
            "name": CUDA_GM1_CHILD_MOIST_NM_KERNEL_NAME,
            "source_sha256": CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
            "upstream_source_sha256": CUDA_MOIST_NM_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
        },
        "runtime": dict(compiled.runtime_identity),
        "claim_limit": (
            "exact selected six-species ice-aware calcnm with CM1's 3-D pi0/th0 "
            "indexing; no calcdef, closure, tendencies, or time integration"
        ),
    }
    return CudaChildMoistNmLaunch(event, compiled, receipt)


@dataclass(frozen=True, slots=True)
class Gm1ChildAuxiliaryCarrier:
    group: str
    file: str
    name: str
    shape: tuple[int, ...]
    offset_nbytes: int
    nbytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedGm1ChildAuxiliary:
    directory: Path
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    carriers: Mapping[str, Mapping[str, Gm1ChildAuxiliaryCarrier]]
    manifest_sha256: str
    receipt_sha256: str
    terrain_open_sha256: str
    live_density_sha256: str
    geopotential_sha256: str

    def field(self, group: str, name: str) -> Any:
        try:
            carrier = self.carriers[group][name]
        except KeyError as error:
            raise KeyError(f"unknown child auxiliary carrier {group}.{name}") from error
        np = importlib.import_module("numpy")
        return np.memmap(
            self.directory / carrier.file,
            dtype=np.dtype("<f4"),
            mode="r",
            offset=carrier.offset_nbytes,
            shape=carrier.shape,
            order="C",
        )

    def physical_base(self, name: str) -> Any:
        """Pack the source-defined physical [k=1:75,j/i=3:302] base window."""

        if name not in ("pi0", "th0"):
            raise KeyError("physical base packing is defined only for pi0 and th0")
        np = importlib.import_module("numpy")
        return np.ascontiguousarray(self.field("terrain_open", name)[1:76, 3:303, 3:303])


_AUXILIARY_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _auxiliary_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _AUXILIARY_HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_json(path: Path, expected_sha256: str, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 is {actual}; expected {expected_sha256}")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    canonical = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    if raw != canonical:
        raise ValueError(f"{label} is not canonical JSON")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    return value


def _validate_and_hash_group(
    root: Path,
    group_name: str,
    group: Mapping[str, Any],
    expected_file: str,
    expected_nbytes: int,
    expected_sha256: str,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> Mapping[str, Gm1ChildAuxiliaryCarrier]:
    if set(group) - {"abi", "carriers", "file", "identity_schema", "nbytes", "receipt_schema", "sha256"}:
        raise ValueError(f"child auxiliary {group_name} group has unexpected keys")
    if (
        group.get("file") != expected_file
        or group.get("nbytes") != expected_nbytes
        or group.get("sha256") != expected_sha256
    ):
        raise ValueError(f"child auxiliary {group_name} identity changed")
    raw_carriers = group.get("carriers")
    if not isinstance(raw_carriers, list):
        raise ValueError(f"child auxiliary {group_name} carriers must be a list")
    parsed: dict[str, Gm1ChildAuxiliaryCarrier] = {}
    offset = 0
    file_path = root / expected_file
    full_digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for raw, (expected_name, expected_shape) in zip(
            raw_carriers, expected_shapes.items(), strict=True
        ):
            if not isinstance(raw, Mapping) or set(raw) != {
                "dtype",
                "finite_count",
                "maximum_f32_bits",
                "minimum_f32_bits",
                "name",
                "nbytes",
                "nonzero_count",
                "offset_nbytes",
                "order",
                "sha256",
                "shape",
            }:
                raise ValueError(f"child auxiliary {group_name} carrier keys changed")
            nbytes = math.prod(expected_shape) * 4
            if (
                raw["name"] != expected_name
                or raw["shape"] != list(expected_shape)
                or raw["dtype"] != "little-endian-f32"
                or raw["order"] != "C"
                or raw["offset_nbytes"] != offset
                or raw["nbytes"] != nbytes
                or raw["finite_count"] != math.prod(expected_shape)
                or not isinstance(raw["nonzero_count"], int)
                or not 0 <= raw["nonzero_count"] <= math.prod(expected_shape)
                or re.fullmatch(r"[0-9a-f]{8}", str(raw["minimum_f32_bits"])) is None
                or re.fullmatch(r"[0-9a-f]{8}", str(raw["maximum_f32_bits"])) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(raw["sha256"])) is None
            ):
                raise ValueError(
                    f"child auxiliary {group_name} carrier {expected_name} contract changed"
                )
            digest = hashlib.sha256()
            remaining = nbytes
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"child auxiliary {group_name} arena is truncated")
                digest.update(block)
                full_digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != raw["sha256"]:
                raise ValueError(
                    f"child auxiliary {group_name} carrier {expected_name} bytes changed"
                )
            parsed[expected_name] = Gm1ChildAuxiliaryCarrier(
                group_name,
                expected_file,
                expected_name,
                expected_shape,
                offset,
                nbytes,
                raw["sha256"],
            )
            offset += nbytes
        if stream.read(1):
            raise ValueError(f"child auxiliary {group_name} arena has trailing bytes")
    if offset != expected_nbytes or file_path.stat().st_size != expected_nbytes:
        raise ValueError(f"child auxiliary {group_name} arena exhaustion changed")
    if full_digest.hexdigest() != expected_sha256:
        raise ValueError(f"child auxiliary {group_name} full SHA-256 changed")
    return MappingProxyType(parsed)


def load_gm1_child_auxiliary(
    directory: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
    expected_atmosphere_manifest_sha256: str | None = None,
    expected_atmosphere_arena_sha256: str | None = None,
    expected_atmosphere_receipt_sha256: str | None = None,
) -> LoadedGm1ChildAuxiliary:
    """Strictly admit and memory-map one five-file Rust child auxiliary estate.

    With no expected digests the estate is admitted by the identity its own
    receipt records: the receipt names the manifest and all three arenas that
    are present, and every carrier extent and offset in the manifest must
    exhaust the arena it describes.  An estate rebuilt from a parent at another
    completed step is then consumable instead of refused for not being the
    first one this project published.

    ``expected_atmosphere_*`` binds the estate to the atmosphere it must have
    been derived from.  The child-step composer passes the atmosphere it just
    admitted, so an estate whose base state, halos and radiation histories were
    reconstructed from a different atmosphere is refused rather than composed
    into one child with perturbation carriers it never saw.
    """

    requested = Path(directory)
    if requested.is_symlink():
        raise ValueError("child auxiliary directory must not be a symlink")
    root = requested.resolve(strict=True)
    entries = tuple(root.iterdir())
    if (
        not root.is_dir()
        or {entry.name for entry in entries} != _AUXILIARY_FILES
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError("child auxiliary directory inventory changed")
    receipt_raw = (root / "receipt.json").read_bytes()
    actual_receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if expected_receipt_sha256 is None:
        expected_receipt_sha256 = actual_receipt_sha256
    receipt = _canonical_json(
        root / "receipt.json", expected_receipt_sha256, "child auxiliary receipt"
    )
    # The receipt anchors this estate, so when the caller names no digests the
    # manifest expectation comes from it rather than from a module constant.
    if expected_manifest_sha256 is None:
        expected_manifest_sha256 = _auxiliary_sha(
            receipt.get("manifest_sha256"), "child auxiliary receipt manifest_sha256"
        )
    manifest = _canonical_json(
        root / "manifest.json", expected_manifest_sha256, "child auxiliary manifest"
    )
    if set(receipt) != {
        "artifact_schema",
        "claim_limit",
        "geopotential_sha256",
        "live_density_sha256",
        "manifest_sha256",
        "model",
        "producer_executable_sha256",
        "producer_source_sha256",
        "schema",
        "status",
        "terrain_open_sha256",
    }:
        raise ValueError("child auxiliary receipt contract changed")
    if (
        receipt["artifact_schema"] != GM1_CHILD_AUXILIARY_SCHEMA
        or receipt["schema"] != GM1_CHILD_AUXILIARY_RECEIPT_SCHEMA
        or receipt["model"] != "GM1"
        or receipt["status"]
        != "PASS_REAL_CHILD_BASE_DENSITY_GEOPOTENTIAL_FRESH_OPEN_BOUNDARY"
        or receipt["manifest_sha256"] != expected_manifest_sha256
        or receipt["claim_limit"] != manifest.get("claim_limit")
    ):
        raise ValueError("child auxiliary receipt contract changed")
    for key in (
        "geopotential_sha256",
        "live_density_sha256",
        "producer_executable_sha256",
        "producer_source_sha256",
        "terrain_open_sha256",
    ):
        _auxiliary_sha(receipt[key], f"child auxiliary receipt {key}")
    if (
        set(manifest) != _AUXILIARY_MANIFEST_KEYS
        or manifest.get("authority") != _AUXILIARY_AUTHORITY
        or manifest.get("schema") != GM1_CHILD_AUXILIARY_MANIFEST_SCHEMA
        or manifest.get("artifact_schema") != GM1_CHILD_AUXILIARY_SCHEMA
        or manifest.get("status")
        != "PASS_REAL_CHILD_BASE_DENSITY_GEOPOTENTIAL_FRESH_OPEN_BOUNDARY"
        or manifest.get("model") != "GM1"
        or manifest.get("producer")
        != {
            "source_sha256": receipt["producer_source_sha256"],
            "executable_sha256": receipt["producer_executable_sha256"],
        }
    ):
        raise ValueError("child auxiliary manifest identity/source binding changed")
    # The estate records the atmosphere it was derived from, the terrain it
    # stands on, the static heights it was built over and the parent all three
    # come from.  Those identities must be recordable; whether they are the
    # right ones is decided by the caller that admitted those artifacts, which
    # passes them in.  `combined_23_sha256` is deliberately not required: it is
    # the digest of a CUDA terrain-metrics output, which the Rust producer
    # cannot compute and therefore no longer claims.  The binding it stood in
    # for is the pair of static-height payload digests below, which the
    # producer does read and does record.
    # These two blocks are the producer/consumer contract, written out exactly.
    # A key the Rust producer stops emitting, or starts emitting, must fail
    # here loudly rather than pass through a `.get` that silently reads None -
    # the estate this change exists to enable would otherwise be published by
    # one side and refused deep inside the other.
    source_atmosphere = manifest.get("source_atmosphere")
    if not isinstance(source_atmosphere, Mapping) or set(source_atmosphere) != {
        "arena_sha256",
        "manifest_sha256",
        "qa_nonnegative_required",
        "receipt_sha256",
    }:
        raise ValueError("child auxiliary source-atmosphere contract changed")
    if source_atmosphere["qa_nonnegative_required"] is not True:
        raise ValueError(
            "child auxiliary was published without the qa-nonnegative requirement; "
            "a negative moisture word would enter the child's first large step"
        )
    for key in ("manifest_sha256", "arena_sha256", "receipt_sha256"):
        _auxiliary_sha(
            source_atmosphere[key], f"child auxiliary source atmosphere {key}"
        )
    # `combined_23_sha256` was retired from this block: it was the digest of a
    # CUDA terrain-metrics output the producer cannot compute, so the producer
    # stopped emitting it and the binding it stood in for is now stated in
    # digests the producer does read.  Estates published before that retirement
    # still carry the key, and they are the ones this distribution ships, so it
    # is admitted and ignored rather than making the historical estate
    # unloadable.  Exactly these two key sets; anything else is still refused.
    static_inputs = manifest.get("static_inputs")
    required = {
        "abi",
        "manifest_sha256",
        "terrain_manifest_sha256",
        "terrain_payload_sha256",
        "terrain_receipt_sha256",
        "validation_receipt_sha256",
        "zf_sha256",
        "zh_sha256",
    }
    if not isinstance(static_inputs, Mapping) or set(static_inputs) not in (
        required,
        required | {"combined_23_sha256"},
    ):
        raise ValueError("child auxiliary static-input contract changed")
    for key in (
        "manifest_sha256",
        "terrain_manifest_sha256",
        "terrain_payload_sha256",
        "terrain_receipt_sha256",
        "validation_receipt_sha256",
        "zf_sha256",
        "zh_sha256",
    ):
        _auxiliary_sha(static_inputs[key], f"child auxiliary static input {key}")
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping) or set(parent) != {
        "checkpoint_sha256",
        "plan_sha256",
    }:
        raise ValueError(
            "child auxiliary manifest does not record the parent checkpoint and "
            "validated plan it was built from; the estate could not be re-bound "
            "to any parent and could be composed with carriers from another run"
        )
    for key, value in parent.items():
        _auxiliary_sha(value, f"child auxiliary parent {key}")
    for key, expected in (
        ("manifest_sha256", expected_atmosphere_manifest_sha256),
        ("arena_sha256", expected_atmosphere_arena_sha256),
        ("receipt_sha256", expected_atmosphere_receipt_sha256),
    ):
        if expected is not None and source_atmosphere[key] != expected:
            raise ValueError(
                f"child auxiliary was derived from atmosphere {key} "
                f"{source_atmosphere[key]}, but the atmosphere admitted alongside it "
                f"is {expected}; the child's base state, halos and radiation "
                "histories would be reconstructed over one atmosphere while its "
                "perturbation carriers came from another"
            )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(_GROUP_EXPECTATIONS):
        raise ValueError("child auxiliary output group inventory changed")
    carriers = {
        name: _validate_and_hash_group(
            root,
            name,
            outputs[name],
            expected[0],
            expected[1],
            receipt[expected[2]],
            expected[3],
        )
        for name, expected in _GROUP_EXPECTATIONS.items()
    }
    loaded = LoadedGm1ChildAuxiliary(
        directory=root,
        manifest=MappingProxyType(dict(manifest)),
        receipt=MappingProxyType(dict(receipt)),
        carriers=MappingProxyType(carriers),
        manifest_sha256=expected_manifest_sha256,
        receipt_sha256=expected_receipt_sha256,
        terrain_open_sha256=receipt["terrain_open_sha256"],
        live_density_sha256=receipt["live_density_sha256"],
        geopotential_sha256=receipt["geopotential_sha256"],
    )
    # The base-state windows are already bound byte for byte by the terrain/open
    # arena digest above.  What is checked here is the physical property the
    # child integrator depends on and a digest could never state for a new
    # parent: the Exner function and the base potential temperature must be
    # finite and strictly positive everywhere, because every hydrostatic and
    # thermodynamic term below divides by or takes a power of them.
    np = importlib.import_module("numpy")
    for name in ("pi0", "th0"):
        physical = loaded.physical_base(name)
        if not bool(np.all(np.isfinite(physical))) or bool(
            np.any(physical <= np.float32(0.0))
        ):
            raise ValueError(
                f"child auxiliary physical {name} window is non-finite or not "
                "strictly positive; every hydrostatic and thermodynamic term the "
                "child evaluates divides by or takes a power of it"
            )
    return loaded


REAL_CHILD_INPUT_NAMES = (
    "c1",
    "c2",
    "mf",
    "rmf",
    "pi0",
    "th0",
    "prs",
    "pp3d",
    "th3d",
    "q3d",
    "u3d",
    "v3d",
    "w3d",
    "rho",
    "rf",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "gx",
    "gy",
)


def real_child_input_shapes() -> Mapping[str, tuple[int, ...]]:
    cell = (75, 300, 300)
    face = (76, 300, 300)
    return MappingProxyType(
        {
            "c1": face,
            "c2": face,
            "mf": face,
            "rmf": face,
            "pi0": cell,
            "th0": cell,
            "prs": cell,
            "pp3d": cell,
            "th3d": cell,
            "q3d": (6, 75, 300, 300),
            "u3d": (75, 300, 301),
            "v3d": (75, 301, 300),
            "w3d": face,
            "rho": cell,
            "rf": face,
            "gz": (300, 300),
            "rgz": (300, 300),
            "gzu": (300, 301),
            "rgzu": (300, 301),
            "gzv": (301, 300),
            "rgzv": (301, 300),
            "gx": face,
            "gy": face,
        }
    )


def _require_real_device_inputs(cupy: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    shapes = real_child_input_shapes()
    if set(inputs) != set(REAL_CHILD_INPUT_NAMES):
        raise ValueError("real child nm/def input inventory changed")
    result = {}
    for name in REAL_CHILD_INPUT_NAMES:
        value = inputs[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.dtype("float32")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-contiguous CUDA float32 {shapes[name]}")
        result[name] = value
    pointers = [int(value.data.ptr) for value in result.values()]
    if len(pointers) != len(set(pointers)):
        raise ValueError("real child nm/def inputs must not alias")
    return result


_REAL_INPUT_SEAL = object()


@dataclass(frozen=True, slots=True)
class AdmittedCudaGm1ChildNmDefInputs:
    arrays: Mapping[str, Any]
    atmosphere: LoadedGm1ChildAtmosphere
    auxiliary: LoadedGm1ChildAuxiliary
    static_metrics: CudaGm1ChildTerrainMetricsLaunch
    stream: Any
    _seal: object

    @property
    def atmosphere_manifest_sha256(self) -> str:
        return self.atmosphere.manifest_sha256

    @property
    def atmosphere_arena_sha256(self) -> str:
        return self.atmosphere.arena_sha256

    @property
    def atmosphere_receipt_sha256(self) -> str:
        return self.atmosphere.receipt_sha256

    @property
    def auxiliary_manifest_sha256(self) -> str:
        return self.auxiliary.manifest_sha256

    @property
    def auxiliary_receipt_sha256(self) -> str:
        return self.auxiliary.receipt_sha256

    @property
    def static_validation_sha256(self) -> str:
        return self.auxiliary.manifest["static_inputs"]["validation_receipt_sha256"]


def upload_admitted_gm1_child_nm_def_inputs(
    atmosphere: LoadedGm1ChildAtmosphere,
    auxiliary: LoadedGm1ChildAuxiliary,
    static_metrics: CudaGm1ChildTerrainMetricsLaunch,
    *,
    stream: Any | None = None,
) -> AdmittedCudaGm1ChildNmDefInputs:
    """Transport only the exact admitted real carriers to one CUDA stream."""

    if not isinstance(atmosphere, LoadedGm1ChildAtmosphere):
        raise TypeError("atmosphere must be LoadedGm1ChildAtmosphere")
    if not isinstance(auxiliary, LoadedGm1ChildAuxiliary):
        raise TypeError("auxiliary must be LoadedGm1ChildAuxiliary")
    if not isinstance(static_metrics, CudaGm1ChildTerrainMetricsLaunch):
        raise TypeError("static_metrics must be CudaGm1ChildTerrainMetricsLaunch")
    # The three admitted inputs must describe one child, not three artifacts
    # that happen to be the right shape.  The auxiliary estate records the
    # atmosphere it was derived from and the terrain it stands on; the static
    # metrics record the terrain they were computed from.  Requiring those
    # recorded identities to agree is what prevents the concrete wrong result -
    # deformation and mixing diagnosed over one surface applied to another
    # run's flow.  Which run it is, is not this function's business: it is
    # decided where those artifacts are admitted.
    source_atmosphere = auxiliary.manifest["source_atmosphere"]
    static_inputs = auxiliary.manifest["static_inputs"]
    metrics_source = static_metrics.receipt["source"]
    if (
        source_atmosphere["manifest_sha256"] != atmosphere.manifest_sha256
        or source_atmosphere["arena_sha256"] != atmosphere.arena_sha256
        or source_atmosphere["receipt_sha256"] != atmosphere.receipt_sha256
    ):
        raise ValueError(
            "the child auxiliary estate was derived from a different atmosphere "
            "than the one admitted here; its base state, halos and radiation "
            "histories would be applied to perturbation carriers they never saw"
        )
    if (
        metrics_source["manifest_sha256"] != static_inputs["terrain_manifest_sha256"]
        or metrics_source["payload_sha256"] != static_inputs["terrain_payload_sha256"]
        or metrics_source["receipt_sha256"] != static_inputs["terrain_receipt_sha256"]
    ):
        raise ValueError(
            "the child static metrics were computed from a different terrain than "
            "the auxiliary estate was built over; every vertical metric and "
            "deformation term would describe a surface the child does not stand on"
        )
    cupy = importlib.import_module("cupy")
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    stream.wait_event(static_metrics.ready_event)
    static = static_metrics.metrics.state.arrays
    expected_static = {
        "c1",
        "c2",
        "mf",
        "rmf",
        "gz",
        "rgz",
        "gzu",
        "rgzu",
        "gzv",
        "rgzv",
        "gx",
        "gy",
    }
    if not expected_static.issubset(static):
        raise ValueError("static metric launch lacks child nm/def carriers")
    with stream:
        arrays = {
            name: static[name] for name in expected_static
        }
        arrays.update(
            {
                "pi0": cupy.asarray(auxiliary.physical_base("pi0")),
                "th0": cupy.asarray(auxiliary.physical_base("th0")),
                "prs": cupy.asarray(auxiliary.field("live_density", "prs")),
                "rho": cupy.asarray(auxiliary.field("live_density", "rho")),
                "rf": cupy.asarray(auxiliary.field("live_density", "rf")),
                "pp3d": cupy.asarray(atmosphere.field("pp3d")),
                "th3d": cupy.asarray(atmosphere.field("th3d")),
                "q3d": cupy.asarray(atmosphere.field("q3d")),
                "u3d": cupy.asarray(atmosphere.field("u3d")),
                "v3d": cupy.asarray(atmosphere.field("v3d")),
                "w3d": cupy.asarray(atmosphere.field("w3d")),
            }
        )
    _require_real_device_inputs(cupy, arrays)
    return AdmittedCudaGm1ChildNmDefInputs(
        MappingProxyType(arrays),
        atmosphere,
        auxiliary,
        static_metrics,
        stream,
        _REAL_INPUT_SEAL,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1ChildNmDefTkeLaunch:
    ready_event: Any
    arrays: Mapping[str, Any]
    moist_nm: CudaChildMoistNmLaunch
    deformation: CudaTerrainDeformationLaunch
    turbsmag: CudaChildTurbsmagInitLaunch
    retained_inputs: Mapping[str, Any]
    receipt: Mapping[str, Any]


def launch_cuda_gm1_child_nm_def_tke(
    inputs: AdmittedCudaGm1ChildNmDefInputs,
    *,
    stream: Any | None = None,
    compiled_moist_nm: CompiledChildMoistNmKernel | None = None,
    compiled_deformation: Any | None = None,
    compiled_turbsmag: Any | None = None,
) -> CudaGm1ChildNmDefTkeLaunch:
    """Enqueue real child calcnm, calcdef, and fresh TKE on one stream."""

    if (
        not isinstance(inputs, AdmittedCudaGm1ChildNmDefInputs)
        or inputs._seal is not _REAL_INPUT_SEAL
    ):
        raise TypeError(
            "inputs must come from upload_admitted_gm1_child_nm_def_inputs"
        )
    cupy = importlib.import_module("cupy")
    a = _require_real_device_inputs(cupy, inputs.arrays)
    stream = inputs.stream if stream is None else stream
    if int(stream.ptr) != int(inputs.stream.ptr):
        raise ValueError("nm/def launch stream differs from admitted upload stream")
    moist_shapes = child_moist_nm_shapes(300, 300, 75)
    with stream:
        horizontal_metrics = {
            "ruh": cupy.ones((300,), dtype=cupy.float32),
            "rvh": cupy.ones((300,), dtype=cupy.float32),
        }
        moist_outputs = {
            name: cupy.empty(
                moist_shapes[name],
                dtype=cupy.uint8 if name == MASK_NAME else cupy.float32,
            )
            for name in CHILD_MOIST_OUTPUT_NAMES
        }
        moist_arrays = {
            name: a[name] for name in CHILD_MOIST_INPUT_NAMES
        } | moist_outputs
        moist = launch_cuda_child_moist_nm(
            moist_arrays,
            ni=300,
            nj=300,
            nk=75,
            dz=100.0,
            stream=stream,
            compiled=compiled_moist_nm,
        )
        def_outputs = {
            "defv": cupy.empty((76, 300, 300), dtype=cupy.float32),
            "defh": cupy.empty((76, 300, 300), dtype=cupy.float32),
        }
        deformation = launch_cuda_terrain_deformation(
            {
                name: a[name]
                for name in (
                    "u3d",
                    "v3d",
                    "w3d",
                    "rho",
                    "rf",
                    "c1",
                    "c2",
                    "gz",
                    "rgz",
                    "gzu",
                    "rgzu",
                    "gzv",
                    "rgzv",
                    "gx",
                    "gy",
                )
            }
            | def_outputs,
            ni=300,
            nj=300,
            nk=75,
            dx=50.0,
            dy=50.0,
            dz=100.0,
            stream=stream,
            compiled=compiled_deformation,
        )
        turbsmag = launch_cuda_child_turbsmag_init(
            {
                "ruh": horizontal_metrics["ruh"],
                "rvh": horizontal_metrics["rvh"],
                "rmf": a["rmf"],
                "nm": moist_outputs["nm"],
                "defv": def_outputs["defv"],
                "defh": def_outputs["defh"],
            },
            ni=300,
            nj=300,
            nk=75,
            dx=50.0,
            dy=50.0,
            dz=100.0,
            metric_identity=REAL_CHILD_TURBSMAG_METRIC_IDENTITY,
            stream=stream,
            compiled=compiled_turbsmag,
        )
    outputs = MappingProxyType(
        {**moist_outputs, **def_outputs, **dict(turbsmag.arrays)}
    )
    receipt = {
        "schema": CUDA_GM1_CHILD_NM_DEF_SCHEMA,
        "status": "ENQUEUED_EXACT_REAL_CHILD_NM_DEF_FRESH_TKE",
        "model": "GM1",
        "source_order": [
            "calcnm with child three-dimensional pi0/th0",
            "calcdef with four-open terrain selection",
            "fresh turbsmag/TKE initialization",
        ],
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
        "shape_zyx": [75, 300, 300],
        "spacing_m": {"dx": 50.0, "dy": 50.0, "dz": 100.0},
        # The artifacts this launch actually consumed, taken from the admitted
        # inputs rather than restated from constants: a receipt that named the
        # first child ever published while the run consumed another would make
        # the substitution invisible to every reader of this evidence.
        "upstream": {
            "atmosphere_manifest_sha256": inputs.atmosphere_manifest_sha256,
            "atmosphere_arena_sha256": inputs.atmosphere_arena_sha256,
            "atmosphere_receipt_sha256": inputs.atmosphere_receipt_sha256,
            "auxiliary_manifest_sha256": inputs.auxiliary_manifest_sha256,
            "auxiliary_receipt_sha256": inputs.auxiliary_receipt_sha256,
            "static_validation_sha256": inputs.static_validation_sha256,
            "rmf_sha256": REAL_CHILD_TURBSMAG_RMF_SHA256,
            "ruh_sha256": GM1_CHILD_UNIFORM_HORIZONTAL_METRIC_SHA256,
            "rvh_sha256": GM1_CHILD_UNIFORM_HORIZONTAL_METRIC_SHA256,
        },
        "operators": {
            "module_source_sha256": {
                "parent_profile_moist_nm": UPSTREAM_MOIST_NM_MODULE_SHA256,
                "terrain_deformation": UPSTREAM_TERRAIN_DEFORMATION_MODULE_SHA256,
                "fresh_turbsmag": UPSTREAM_CHILD_TURBSMAG_MODULE_SHA256,
            },
            "kernel_source_sha256": {
                "child_moist_nm": CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256,
                "terrain_deformation": CUDA_TERRAIN_DEFORMATION_SOURCE_SHA256,
                "fresh_turbsmag": CUDA_GM1_CHILD_TURBSMAG_INIT_SOURCE_SHA256,
            },
            "moist_nm": dict(moist.receipt),
            "terrain_deformation": dict(deformation.receipt),
            "fresh_turbsmag": dict(turbsmag.receipt),
        },
        "boundary_disposition": {
            "nm_k0_and_knk": "operator exact positive zero",
            "defv_k0_and_knk": "operator exact positive zero",
            "defh_k0_and_knk": "operator exact positive zero",
            "tkea_bottom": "copies diagnosed k=1 interior",
            "tkea_top": "operator exact positive zero",
            "tke3d": "raw-bit copy of tkea",
            "tketen": "operator exact positive zero",
        },
        "claim_limit": CUDA_GM1_CHILD_NM_DEF_CLAIM_LIMIT,
    }
    return CudaGm1ChildNmDefTkeLaunch(
        turbsmag.ready_event,
        outputs,
        moist,
        deformation,
        turbsmag,
        MappingProxyType({**a, **horizontal_metrics}),
        MappingProxyType(receipt),
    )


__all__ = (
    "AdmittedCudaGm1ChildNmDefInputs",
    "AUTHORITY_SOURCE_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CHILD_MOIST_ALL_NAMES",
    "CHILD_MOIST_FLOAT_OUTPUT_NAMES",
    "CHILD_MOIST_INPUT_NAMES",
    "CHILD_MOIST_OUTPUT_NAMES",
    "CUDA_GM1_CHILD_MOIST_NM_ABI",
    "CUDA_GM1_CHILD_MOIST_NM_KERNEL_NAME",
    "CUDA_GM1_CHILD_MOIST_NM_SCHEMA",
    "CUDA_GM1_CHILD_MOIST_NM_SOURCE",
    "CUDA_GM1_CHILD_MOIST_NM_SOURCE_SHA256",
    "CUDA_GM1_CHILD_NM_DEF_CLAIM_LIMIT",
    "CUDA_GM1_CHILD_NM_DEF_SCHEMA",
    "ChildMoistNmSelection",
    "CompiledChildMoistNmKernel",
    "CudaChildMoistNmLaunch",
    "CudaGm1ChildNmDefTkeLaunch",
    "DEFAULT_CHILD_MOIST_NM_SELECTION",
    "GM1_CHILD_AUXILIARY_GEOPOTENTIAL_SHA256",
    "GM1_CHILD_AUXILIARY_LIVE_DENSITY_SHA256",
    "GM1_CHILD_AUXILIARY_MANIFEST_SHA256",
    "GM1_CHILD_AUXILIARY_RECEIPT_SHA256",
    "GM1_CHILD_AUXILIARY_TERRAIN_OPEN_SHA256",
    "GM1_CHILD_UNIFORM_HORIZONTAL_METRIC_SHA256",
    "Gm1ChildAuxiliaryCarrier",
    "LoadedGm1ChildAuxiliary",
    "REAL_CHILD_INPUT_NAMES",
    "SOURCE_ANCHORS",
    "child_moist_nm_shapes",
    "compile_cuda_child_moist_nm",
    "cpu_child_moist_nm_inplace",
    "launch_cuda_child_moist_nm",
    "launch_cuda_gm1_child_nm_def_tke",
    "load_gm1_child_auxiliary",
    "real_child_input_shapes",
    "upload_admitted_gm1_child_nm_def_inputs",
)
