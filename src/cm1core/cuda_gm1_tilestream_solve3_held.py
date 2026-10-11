"""Tile-streamable solve3 held-rate recapture after the pre-held boundary token.

The selected CM1 r21.1 path executes the four physical mass-grid statements at
``solve3.F:319-337`` only after ``bcp(ppx)`` and ``bcp(rho)``.  This module
consumes that exact pre-held token and applies the default-REAL subtraction,
then multiplication by the precomputed reciprocal timestep, in source order.

The result authorizes only a later momentum current-to-A admission.  It does
not copy current state to A, commit TKE, rebuild reciprocal density, solve
pressure, exchange MPI halos, mutate open histories, or complete solve3/RK.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_solve3_boundaries import (
    Gm1TiledSolve3PreHeldBoundaryToken,
)
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_TILESTREAM_SOLVE3_HELD_SCHEMA = "gm1.cuda-tilestream-solve3-held/v1"
CUDA_GM1_TILESTREAM_SOLVE3_HELD_ABI = "gm1.cuda-tilestream-solve3-held-recapture-f32/v1"
CUDA_GM1_TILESTREAM_SOLVE3_HELD_KERNEL = "gm1_tiled_solve3_held_f32_v1"
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE3_SHA256 = (
    "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393"
)
AUTHORITY_ANCHOR = "src/solve3.F:319-337"
GM1_SOURCE_IDENTITY_SHA256 = (
    "d6d7588b9a2a2ecd979d4d5e609a602f52d488fd918c590bc3998a5319aebfe3"
)
GM1_ACCEPTED_PDEF_APPLY_RECEIPT_FILE_SHA256 = (
    "62673d86b63cb03dc3a534e3289dcd61392de84da3249e4bf9a37060e1169ae4"
)
GM1_ACCEPTED_PDEF_APPLY_RECEIPT_SEMANTIC_SHA256 = (
    "14611f21a925fe486b8187d58ff8add0d177e1250f099af9c1db40376ae179bc"
)
GM1_ACCEPTED_POST_WSM6_RECEIPT_FILE_SHA256 = (
    "5cfb967d886eac2bb6e8b6b34cc3da50d3e73e9160f1021b390159046a25d7cc"
)
GM1_ACCEPTED_POST_WSM6_RECEIPT_SEMANTIC_SHA256 = (
    "babbf6dcc8a7d786743277c115dbb537a0762312c3596b884d826c75251c4f1b"
)
GM1_ACCEPTED_BOUNDARY_RECEIPT_FILE_SHA256 = (
    "4d66434431f678eb2d05bcb7be705d18e1d1ec207d943978aac20310a97006be"
)
GM1_ACCEPTED_BOUNDARY_RECEIPT_SEMANTIC_SHA256 = (
    "33b362c9ce6979872bf4f0c0177287a473bbaf156331cbfe57ba26efb8ee9d93"
)
GM1_ACCEPTED_PREHELD_PPX_PADDED_SHA256 = (
    "39aeda87fb7ec443fd74bc07a6ea431d00ef54f9110e1609a48f68e84e535b44"
)
GM1_ACCEPTED_PREHELD_RHO_PADDED_SHA256 = (
    "ae8eb5006e3b24791c2ec7f68e56403c2f52ed6744c10c856f0d1e5faff43dd5"
)
GM1_ACCEPTED_CURRENT_SHA256 = MappingProxyType(
    {
        "pp3d": "c8312d89a275ef3c067aec46cac22a9be45acdfbc43d2e351a0b24d536fe3013",
        "th3d": "175f6398c5e2bd2d72e9ac7fe2d49b8d227fad32ab83cfcf32ce4d1e6a74f214",
        "q3d": "dc06ecdcde598b57f5c3fc1a63dbe9ade467adac728d6d315f22b9b3337ab5f0",
    }
)
GM1_ACCEPTED_SNAPSHOT_SHA256 = MappingProxyType(
    {
        "qpten": "54c1a30d57a4ad0eb87655d4bbf2dffcdb8d66a79cd703b9ee7689223fbc2ab9",
        "qtten": "a59f71d828c9f40702a5a87fd80930a8061e51eda1d1e16ec8a10dae3e8c080b",
        "qvten": "981eefefff70f48d821c6d31a0b5289eaa262ce3964f1c5cf24148b5b122e973",
        "qcten": "c542d08328be9e05d40373ce71a36c473dfc78d594de8396f3eb418898858e6b",
    }
)
GM1_INITIAL_HELD_POSITIVE_ZERO_SHA256 = (
    "5a4a3c8e7475e11153c3bbeab9f407aa70157d38b8dc03f1b9bf113276a7e939"
)
GM1_TILESTREAM_SOLVE3_HELD_CLAIM_LIMIT = (
    "CM1 r21.1 selected imoist=1 eqtset=2 ptype>=1 physical held-rate "
    "recapture at solve3.F:319-337 after the admitted serial ppx/rho boundary "
    "prefix; no current-to-A copy, TKE lifecycle, reciprocal-density rebuild, "
    "global psolver, MPI, open-history mutation, full solve3/RK, or five-metre "
    "meteorological product claim"
)

_CURRENT_ORDER = ("pp3d", "th3d", "qv", "qc")
_HELD_ORDER = ("qpten", "qtten", "qvten", "qcten")
_BUFFER_ORDER = (*_CURRENT_ORDER, *_HELD_ORDER)


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


def _f32_bits(value: float) -> str:
    return struct.pack("<f", value).hex()


def _sha_array(value: Any) -> str:
    return hashlib.sha256(
        value.astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()


def tiled_solve3_held_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    mass = (nk, tile_ny, tile_nx)
    return MappingProxyType({name: mass for name in _BUFFER_ORDER})


def _require_preheld_token(
    token: object,
    *,
    source_identity_sha256: str,
    real: bool,
) -> Mapping[str, Any]:
    if not isinstance(token, Gm1TiledSolve3PreHeldBoundaryToken):
        raise TypeError("preheld token must be the exact boundary-prefix token type")
    receipt = token.receipt
    if (
        receipt.get("schema") != "gm1.tilestream-solve3-preheld-boundary-token/v1"
        or receipt.get("status")
        != "ADMITTED_PPX_RHO_BOUNDARIES_READY_FOR_HELD_RECAPTURE_ONLY"
        or receipt.get("ready_for") != "held-rate recapture admission only"
        or receipt.get("operation_b_current_to_a_available") is not False
        or receipt.get("mpi_performed") is not False
        or receipt.get("source_identity_sha256") != source_identity_sha256
    ):
        raise ValueError("preheld boundary token scope or source identity changed")
    detached = dict(receipt)
    operation_token = detached.pop("operation_token_sha256", None)
    if operation_token != hashlib.sha256(_canonical(detached)).hexdigest():
        raise ValueError("preheld boundary operation token changed")
    if real and receipt.get("outputs") != {
        "ppx_padded_sha256": GM1_ACCEPTED_PREHELD_PPX_PADDED_SHA256,
        "rho_padded_sha256": GM1_ACCEPTED_PREHELD_RHO_PADDED_SHA256,
    }:
        raise ValueError("preheld boundary outputs differ from the accepted seal")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3HeldBinding:
    """Physical current state and mutable solve2 snapshots/solve3 rates."""

    pp3d: Any
    th3d: Any
    q3d: Any
    qpten: Any
    qtten: Any
    qvten: Any
    qcten: Any
    preheld_token: Gm1TiledSolve3PreHeldBoundaryToken
    nx: int
    ny: int
    nk: int
    dt: float
    rdt: float
    source_identity_sha256: str
    pdef_apply_receipt_file_sha256: str
    pdef_apply_receipt_semantic_sha256: str
    post_wsm6_receipt_file_sha256: str
    post_wsm6_receipt_semantic_sha256: str
    boundary_receipt_file_sha256: str
    boundary_receipt_semantic_sha256: str
    initial_held_positive_zero_sha256: str
    compact_validation: bool

    @classmethod
    def create(
        cls,
        *,
        pp3d: Any,
        th3d: Any,
        q3d: Any,
        qpten: Any,
        qtten: Any,
        qvten: Any,
        qcten: Any,
        preheld_token: Gm1TiledSolve3PreHeldBoundaryToken,
        dt: float,
        source_identity_sha256: str,
        pdef_apply_receipt_file_sha256: str,
        pdef_apply_receipt_semantic_sha256: str,
        post_wsm6_receipt_file_sha256: str,
        post_wsm6_receipt_semantic_sha256: str,
        boundary_receipt_file_sha256: str,
        boundary_receipt_semantic_sha256: str,
        initial_held_positive_zero_sha256: str,
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledSolve3HeldBinding":
        np = importlib.import_module("numpy")
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        if (
            not isinstance(pp3d, np.ndarray)
            or pp3d.dtype != np.dtype("<f4")
            or pp3d.ndim != 3
            or not pp3d.flags.c_contiguous
        ):
            raise ValueError("pp3d must be host C-<f4 [nk,ny,nx]")
        nk, ny, nx = map(int, pp3d.shape)
        mass = (nk, ny, nx)
        expected = {
            "th3d": mass,
            "q3d": (6, *mass),
            "qpten": mass,
            "qtten": mass,
            "qvten": mass,
            "qcten": mass,
        }
        values = {
            "th3d": th3d,
            "q3d": q3d,
            "qpten": qpten,
            "qtten": qtten,
            "qvten": qvten,
            "qcten": qcten,
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
        for name in _HELD_ORDER:
            if not values[name].flags.writeable:
                raise ValueError(f"held snapshot {name} must be writable")
        all_values = (pp3d, *values.values())
        for index, left in enumerate(all_values):
            if any(np.shares_memory(left, right) for right in all_values[index + 1 :]):
                raise ValueError("held-recapture state and snapshots must not alias")
        if not all(bool(np.all(np.isfinite(value))) for value in all_values):
            raise ValueError("held-recapture input contains non-finite words")
        try:
            dt = float(dt)
        except (TypeError, ValueError) as exc:
            raise TypeError("dt must be a finite scalar") from exc
        if not math.isfinite(dt) or _f32_bits(dt) != "0000003f":
            raise ValueError("selected held recapture requires exact float32 dt=0.5")
        rdt = float(np.float32(np.float32(1.0) / np.float32(dt)))
        source_identity_sha256 = _require_sha256(
            "source_identity_sha256", source_identity_sha256
        )
        pins = {
            "pdef_apply_receipt_file_sha256": pdef_apply_receipt_file_sha256,
            "pdef_apply_receipt_semantic_sha256": pdef_apply_receipt_semantic_sha256,
            "post_wsm6_receipt_file_sha256": post_wsm6_receipt_file_sha256,
            "post_wsm6_receipt_semantic_sha256": post_wsm6_receipt_semantic_sha256,
            "boundary_receipt_file_sha256": boundary_receipt_file_sha256,
            "boundary_receipt_semantic_sha256": boundary_receipt_semantic_sha256,
        }
        pins = {name: _require_sha256(name, value) for name, value in pins.items()}
        initial_held_positive_zero_sha256 = _require_sha256(
            "initial_held_positive_zero_sha256",
            initial_held_positive_zero_sha256,
        )
        _require_preheld_token(
            preheld_token,
            source_identity_sha256=source_identity_sha256,
            real=not allow_compact_validation,
        )
        if allow_compact_validation:
            if min(nx, ny) < 2:
                raise ValueError("compact held-recapture domain is too small")
        else:
            if (nx, ny, nk) != (300, 300, 75):
                raise ValueError("real held-recapture source must be [75,300,300]")
            if source_identity_sha256 != GM1_SOURCE_IDENTITY_SHA256:
                raise ValueError("real held-recapture source identity changed")
            expected_pins = {
                "pdef_apply_receipt_file_sha256": (
                    GM1_ACCEPTED_PDEF_APPLY_RECEIPT_FILE_SHA256
                ),
                "pdef_apply_receipt_semantic_sha256": (
                    GM1_ACCEPTED_PDEF_APPLY_RECEIPT_SEMANTIC_SHA256
                ),
                "post_wsm6_receipt_file_sha256": (
                    GM1_ACCEPTED_POST_WSM6_RECEIPT_FILE_SHA256
                ),
                "post_wsm6_receipt_semantic_sha256": (
                    GM1_ACCEPTED_POST_WSM6_RECEIPT_SEMANTIC_SHA256
                ),
                "boundary_receipt_file_sha256": (
                    GM1_ACCEPTED_BOUNDARY_RECEIPT_FILE_SHA256
                ),
                "boundary_receipt_semantic_sha256": (
                    GM1_ACCEPTED_BOUNDARY_RECEIPT_SEMANTIC_SHA256
                ),
            }
            if pins != expected_pins:
                raise ValueError("held-recapture prerequisite receipt pins changed")
            if (
                initial_held_positive_zero_sha256
                != GM1_INITIAL_HELD_POSITIVE_ZERO_SHA256
            ):
                raise ValueError("initial held +0 source disposition changed")
            current_hashes = {
                "pp3d": _sha_array(pp3d),
                "th3d": _sha_array(th3d),
                "q3d": _sha_array(q3d),
            }
            if current_hashes != dict(GM1_ACCEPTED_CURRENT_SHA256):
                raise ValueError("held-recapture current-state bytes changed")
            snapshot_hashes = {name: _sha_array(values[name]) for name in _HELD_ORDER}
            if snapshot_hashes != dict(GM1_ACCEPTED_SNAPSHOT_SHA256):
                raise ValueError(
                    "solve2 held snapshots differ from their accepted seals"
                )
        return cls(
            pp3d,
            th3d,
            q3d,
            qpten,
            qtten,
            qvten,
            qcten,
            preheld_token,
            nx,
            ny,
            nk,
            dt,
            rdt,
            source_identity_sha256,
            pins["pdef_apply_receipt_file_sha256"],
            pins["pdef_apply_receipt_semantic_sha256"],
            pins["post_wsm6_receipt_file_sha256"],
            pins["post_wsm6_receipt_semantic_sha256"],
            pins["boundary_receipt_file_sha256"],
            pins["boundary_receipt_semantic_sha256"],
            initial_held_positive_zero_sha256,
            allow_compact_validation,
        )


def fill_gm1_tiled_solve3_held_source(
    binding: Gm1TiledSolve3HeldBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    if not isinstance(binding, Gm1TiledSolve3HeldBinding):
        raise TypeError("binding must be Gm1TiledSolve3HeldBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec does not cover held-recapture source")
    np = importlib.import_module("numpy")
    shapes = tiled_solve3_held_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(target, Mapping) or tuple(target) != tuple(shapes):
        raise ValueError("held-recapture tile inventory or order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    sources = {
        "pp3d": binding.pp3d,
        "th3d": binding.th3d,
        "qv": binding.q3d[0],
        "qc": binding.q3d[1],
        **{name: getattr(binding, name) for name in _HELD_ORDER},
    }
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
        value[:, :active_ny, :active_nx] = sources[name][:, y0:y1, x0:x1]
    tile_values = tuple(target.values())
    for index, left in enumerate(tile_values):
        if any(np.shares_memory(left, right) for right in tile_values[index + 1 :]):
            raise ValueError("held-recapture tile arrays must not alias")
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "active_shape": [binding.nk, active_ny, active_nx],
            "rdt_f32_bits": _f32_bits(binding.rdt),
            "artificial_padding_initialized": "+0",
        }
    )


def cpu_gm1_tiled_solve3_held_inplace(
    arrays: Mapping[str, Any],
    *,
    active_nx: int,
    active_ny: int,
    dt: float,
) -> None:
    """Literal default-REAL subtract then multiply-by-rdt statement order."""

    np = importlib.import_module("numpy")
    if not isinstance(arrays, Mapping) or tuple(arrays) != _BUFFER_ORDER:
        raise ValueError("CPU held-recapture inventory/order changed")
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    if _f32_bits(float(dt)) != "0000003f":
        raise ValueError("selected held recapture requires exact float32 dt=0.5")
    rdt = np.float32(np.float32(1.0) / np.float32(dt))
    for name, value in arrays.items():
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.ndim != 3
            or not value.flags.c_contiguous
            or active_ny > value.shape[1]
            or active_nx > value.shape[2]
        ):
            raise ValueError(f"CPU {name} must be C-<f4 tile storage")
    for current_name, held_name in zip(_CURRENT_ORDER, _HELD_ORDER, strict=True):
        current = arrays[current_name][:, :active_ny, :active_nx]
        held = arrays[held_name][:, :active_ny, :active_nx]
        np.subtract(current, held, out=held)
        np.multiply(held, rdt, out=held)


def scatter_gm1_tiled_solve3_held_output(
    binding: Gm1TiledSolve3HeldBinding,
    spec: Gm1TileSpec,
    source: Mapping[str, Any],
) -> Mapping[str, object]:
    if not isinstance(binding, Gm1TiledSolve3HeldBinding):
        raise TypeError("binding must be Gm1TiledSolve3HeldBinding")
    np = importlib.import_module("numpy")
    shapes = tiled_solve3_held_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(source, Mapping) or tuple(source) != tuple(shapes):
        raise ValueError("held-recapture scatter inventory/order changed")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    for name in _HELD_ORDER:
        value = source[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"held drain {name} layout changed")
        getattr(binding, name)[:, y0:y1, x0:x1] = value[:, :active_ny, :active_nx]
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "published_words": len(_HELD_ORDER) * binding.nk * active_ny * active_nx,
            "physical_mass_only": True,
            "halo_or_history_mutation": False,
        }
    )


CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE = r"""// gm1.cuda-tilestream-solve3-held-recapture-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
extern "C" __global__ void gm1_tiled_solve3_held_f32_v1(
 const float*pp3d,const float*th3d,const float*qv,const float*qc,
 float*qpten,float*qtten,float*qvten,float*qcten,
 unsigned int active_nx,unsigned int active_ny,unsigned int nk,
 unsigned int tile_nx,unsigned int tile_ny,float rdt){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long n=(unsigned long long)nk*active_ny*active_nx;
 if(q>=n)return;
 const unsigned int i=(unsigned int)(q%active_nx);
 const unsigned long long t=q/active_nx;
 const unsigned int j=(unsigned int)(t%active_ny),k=(unsigned int)(t/active_ny);
 const unsigned long long z=((unsigned long long)k*tile_ny+j)*tile_nx+i;
 float d=__fsub_rn(pp3d[z],qpten[z]);qpten[z]=__fmul_rn(d,rdt);
 d=__fsub_rn(th3d[z],qtten[z]);qtten[z]=__fmul_rn(d,rdt);
 d=__fsub_rn(qv[z],qvten[z]);qvten[z]=__fmul_rn(d,rdt);
 d=__fsub_rn(qc[z],qcten[z]);qcten[z]=__fmul_rn(d,rdt);
}
"""
CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE_BYTES = (
    CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE.encode("ascii")
)
CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE_BYTES
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledSolve3Held:
    module_owner: Any
    kernel: Any
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_solve3_held() -> CompiledGm1TiledSolve3Held:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    common = ("--std=c++14", "--fmad=false")
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE,
        "gm1_tiled_solve3_held_ptx.cu",
        method="ptx",
    ).compile(tuple(dict.fromkeys((*ptx_options, *common))))
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE,
        "gm1_tiled_solve3_held_cubin.cu",
        method="cubin",
    ).compile(tuple(dict.fromkeys((*cubin_options, *common))))
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if not re.search(
        rf"\.entry\s+{CUDA_GM1_TILESTREAM_SOLVE3_HELD_KERNEL}(?:\s|\()",
        ptx.decode("utf-8"),
    ):
        raise ValueError("compiled PTX lacks held-recapture entry")
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
    return CompiledGm1TiledSolve3Held(
        module,
        module.get_function(CUDA_GM1_TILESTREAM_SOLVE3_HELD_KERNEL),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledSolve3HeldLaunch:
    ready_event: Any
    compiled: CompiledGm1TiledSolve3Held
    receipt: Mapping[str, Any]


def launch_cuda_gm1_tiled_solve3_held(
    arrays: Mapping[str, Any],
    binding: Gm1TiledSolve3HeldBinding,
    spec: Gm1TileSpec,
    *,
    stream: Any | None = None,
    compiled: CompiledGm1TiledSolve3Held | None = None,
) -> CudaGm1TiledSolve3HeldLaunch:
    if not isinstance(binding, Gm1TiledSolve3HeldBinding):
        raise TypeError("binding must be Gm1TiledSolve3HeldBinding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec does not cover held-recapture source")
    cupy = importlib.import_module("cupy")
    shapes = tiled_solve3_held_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA held-recapture inventory/order changed")
    ranges: list[tuple[int, int]] = []
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be C-order CUDA float32 {shape}")
        start = int(value.data.ptr)
        ranges.append((start, start + int(value.nbytes)))
    for index, (left_start, left_end) in enumerate(ranges):
        if any(
            left_start < right_end and right_start < left_end
            for right_start, right_end in ranges[index + 1 :]
        ):
            raise ValueError("CUDA held-recapture arrays must not alias")
    preheld = _require_preheld_token(
        binding.preheld_token,
        source_identity_sha256=binding.source_identity_sha256,
        real=not binding.compact_validation,
    )
    stream = cupy.cuda.get_current_stream() if stream is None else stream
    if int(preheld["stream_pointer"]) != int(stream.ptr):
        raise ValueError("held recapture must remain on the preheld token stream")
    compiled = compile_cuda_gm1_tiled_solve3_held() if compiled is None else compiled
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    words = binding.nk * active_ny * active_nx
    threads = 256
    with stream:
        compiled.kernel(
            ((words + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            (
                *(arrays[name] for name in _BUFFER_ORDER),
                cupy.uint32(active_nx),
                cupy.uint32(active_ny),
                cupy.uint32(binding.nk),
                cupy.uint32(spec.tile_nx),
                cupy.uint32(spec.tile_ny),
                cupy.float32(binding.rdt),
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    record = {
        "schema": CUDA_GM1_TILESTREAM_SOLVE3_HELD_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_SOLVE3_HELD_ABI,
        "status": "ENQUEUED_SOURCE_ORDERED_HELD_RECAPTURE_TILE",
        "tile": spec.index,
        "owned_yx": [y0, y1, x0, x1],
        "active_shape": [binding.nk, active_ny, active_nx],
        "statement_order": list(_HELD_ORDER),
        "arithmetic": "default REAL subtract then multiply by precomputed rdt",
        "dt_f32_bits": _f32_bits(binding.dt),
        "rdt_f32_bits": _f32_bits(binding.rdt),
        "source_identity_sha256": binding.source_identity_sha256,
        "preheld_operation_token_sha256": preheld["operation_token_sha256"],
        "boundary_receipt_file_sha256": binding.boundary_receipt_file_sha256,
        "initial_held_positive_zero_sha256": (
            binding.initial_held_positive_zero_sha256
        ),
        "same_stream_no_host_sync": True,
        "stream_pointer": int(stream.ptr),
        "kernel_source_sha256": CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE_SHA256,
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve3_sha256": AUTHORITY_SOLVE3_SHA256,
            "anchor": AUTHORITY_ANCHOR,
        },
        "refusals": {
            "current_to_a": True,
            "tke_commit": True,
            "reciprocal_density": True,
            "global_psolver": True,
            "mpi": True,
            "open_history_mutation": True,
            "full_solve3_or_rk": True,
            "five_meter_meteorological_product": True,
        },
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return CudaGm1TiledSolve3HeldLaunch(event, compiled, MappingProxyType(record))


@dataclass(frozen=True, slots=True)
class Gm1TiledSolve3HeldToken:
    """Completion token permitting only a later momentum current-to-A seam."""

    receipt: Mapping[str, Any]


def finish_gm1_tiled_solve3_held(
    binding: Gm1TiledSolve3HeldBinding,
    plan: Gm1TilePlan,
    launches: Sequence[CudaGm1TiledSolve3HeldLaunch],
) -> Gm1TiledSolve3HeldToken:
    if not isinstance(binding, Gm1TiledSolve3HeldBinding):
        raise TypeError("binding must be Gm1TiledSolve3HeldBinding")
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("held completion plan differs from source domain")
    if len(launches) != plan.ntiles:
        raise ValueError("held completion requires every exact tile receipt")
    preheld = _require_preheld_token(
        binding.preheld_token,
        source_identity_sha256=binding.source_identity_sha256,
        real=not binding.compact_validation,
    )
    tokens: list[str] = []
    streams: set[int] = set()
    for expected, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledSolve3HeldLaunch):
            raise TypeError("held completion requires exact CUDA launch objects")
        if launch.ready_event is None:
            raise ValueError("held launch lost its ready event")
        receipt = launch.receipt
        if (
            receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE3_HELD_SCHEMA
            or receipt.get("status") != "ENQUEUED_SOURCE_ORDERED_HELD_RECAPTURE_TILE"
            or receipt.get("tile") != expected.index
            or receipt.get("source_identity_sha256") != binding.source_identity_sha256
            or receipt.get("preheld_operation_token_sha256")
            != preheld["operation_token_sha256"]
            or receipt.get("boundary_receipt_file_sha256")
            != binding.boundary_receipt_file_sha256
            or receipt.get("initial_held_positive_zero_sha256")
            != binding.initial_held_positive_zero_sha256
        ):
            raise ValueError("held tile receipt identity/order differs")
        detached = dict(receipt)
        token = detached.pop("operation_token_sha256", None)
        if token != hashlib.sha256(_canonical(detached)).hexdigest():
            raise ValueError("held tile operation token changed")
        tokens.append(token)
        streams.add(int(receipt["stream_pointer"]))
    if streams != {int(preheld["stream_pointer"])}:
        raise ValueError("held tiles did not remain on the preheld token stream")
    np = importlib.import_module("numpy")
    if not all(
        bool(np.all(np.isfinite(getattr(binding, name)))) for name in _HELD_ORDER
    ):
        raise ValueError("completed held-rate output contains non-finite words")
    output_hashes = {name: _sha_array(getattr(binding, name)) for name in _HELD_ORDER}
    record = {
        "schema": "gm1.tilestream-solve3-held-token/v1",
        "status": "ADMITTED_HELD_RATES_READY_FOR_MOMENTUM_CURRENT_TO_A_ONLY",
        "source_identity_sha256": binding.source_identity_sha256,
        "preheld_operation_token_sha256": preheld["operation_token_sha256"],
        "boundary_receipt": {
            "file_sha256": binding.boundary_receipt_file_sha256,
            "semantic_sha256": binding.boundary_receipt_semantic_sha256,
        },
        "initial_held_positive_zero_sha256": (
            binding.initial_held_positive_zero_sha256
        ),
        "tile_count": plan.ntiles,
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_operation_tokens": tokens,
        "stream_pointer": next(iter(streams)),
        "source_order": [
            "accepted preheld ppx/rho boundary token",
            "qpten=(pp3d-qpten_snapshot)*rdt",
            "qtten=(th3d-qtten_snapshot)*rdt",
            "qvten=(qv-qvten_snapshot)*rdt",
            "qcten=(qc-qcten_snapshot)*rdt",
        ],
        "outputs": output_hashes,
        "ready_for": "momentum current-to-A admission only",
        "full_operation_b_available": False,
        "tke_commit_performed": False,
        "claim_limit": GM1_TILESTREAM_SOLVE3_HELD_CLAIM_LIMIT,
    }
    record["operation_token_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return Gm1TiledSolve3HeldToken(MappingProxyType(record))


def five_meter_tiled_solve3_held_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192
) -> Mapping[str, object]:
    shapes = tiled_solve3_held_buffer_shapes(tile_nx=tile_nx, tile_ny=tile_ny, nk=75)
    per_buffer = sum(math.prod(shape) * 4 for shape in shapes.values())
    record = {
        "schema": "gm1.hybrid-tilestream-solve3-held-memory/v1",
        "domain": [75, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "carriers": list(shapes),
        "host_per_buffer_bytes": per_buffer,
        "device_per_buffer_bytes": per_buffer,
        "double_host_bytes": 2 * per_buffer,
        "double_device_bytes": 2 * per_buffer,
        "full_domain_held_snapshot_duplicate_bytes": 0,
        "canonical_store_mutations": list(_HELD_ORDER),
        "sequential_after_boundary_prefix": True,
        "global_pressure_workspace_executed": False,
        "current_to_a_performed": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHOR",
    "AUTHORITY_SOLVE3_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_SOLVE3_HELD_ABI",
    "CUDA_GM1_TILESTREAM_SOLVE3_HELD_KERNEL",
    "CUDA_GM1_TILESTREAM_SOLVE3_HELD_SCHEMA",
    "CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE",
    "CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE_BYTES",
    "CUDA_GM1_TILESTREAM_SOLVE3_HELD_SOURCE_SHA256",
    "CompiledGm1TiledSolve3Held",
    "CudaGm1TiledSolve3HeldLaunch",
    "GM1_ACCEPTED_BOUNDARY_RECEIPT_FILE_SHA256",
    "GM1_ACCEPTED_BOUNDARY_RECEIPT_SEMANTIC_SHA256",
    "GM1_ACCEPTED_CURRENT_SHA256",
    "GM1_ACCEPTED_PDEF_APPLY_RECEIPT_FILE_SHA256",
    "GM1_ACCEPTED_PDEF_APPLY_RECEIPT_SEMANTIC_SHA256",
    "GM1_ACCEPTED_POST_WSM6_RECEIPT_FILE_SHA256",
    "GM1_ACCEPTED_POST_WSM6_RECEIPT_SEMANTIC_SHA256",
    "GM1_ACCEPTED_PREHELD_PPX_PADDED_SHA256",
    "GM1_ACCEPTED_PREHELD_RHO_PADDED_SHA256",
    "GM1_ACCEPTED_SNAPSHOT_SHA256",
    "GM1_INITIAL_HELD_POSITIVE_ZERO_SHA256",
    "GM1_SOURCE_IDENTITY_SHA256",
    "GM1_TILESTREAM_SOLVE3_HELD_CLAIM_LIMIT",
    "Gm1TiledSolve3HeldBinding",
    "Gm1TiledSolve3HeldToken",
    "compile_cuda_gm1_tiled_solve3_held",
    "cpu_gm1_tiled_solve3_held_inplace",
    "fill_gm1_tiled_solve3_held_source",
    "finish_gm1_tiled_solve3_held",
    "five_meter_tiled_solve3_held_inventory",
    "launch_cuda_gm1_tiled_solve3_held",
    "scatter_gm1_tiled_solve3_held_output",
    "tiled_solve3_held_buffer_shapes",
)
