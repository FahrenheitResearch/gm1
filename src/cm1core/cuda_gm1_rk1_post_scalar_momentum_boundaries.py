"""Live post-scalar RK1 momentum-boundary replay for the selected GM1 child.

The global child acoustic launcher necessarily leaves valid momentum halos,
but its family fill occurs before the scalar RK work.  CM1 r21.1 orders the
stage boundary calls after scalar/moisture and pressure/density work.  This
module therefore replays, on the same CUDA stream, the exact selected serial
order ``bcu(u3d)``, ``bcv(v3d)``, ``bcw(w3d,1)``, then terrain ``bcwsfc``.

It consumes distinct immutable checkpoints for CM1's post-sound
``rru/rrv/rrw`` averaged velocities and its stage ``u3d/v3d/w3d`` carriers,
plus the exact live scalar/moist RK1 token, so earlier numerically-equivalent
fills cannot be relabelled as source-order completion.  No MPI exchange,
scalar/pressure boundary, TKE RK, completed broader stage, tiled global
pressure solve, forecast, or five-metre product is emitted.
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

from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from .cuda_sound3_open_boundary import (
    CUDA_SOUND3_OPEN_SOURCE_SHA256,
    CompiledOpenBoundaryKernels,
    compile_cuda_sound3_open_boundary,
    cpu_bcu_open_inplace,
    cpu_bcv_open_inplace,
    cpu_bcw_open_inplace,
)
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SCHEMA = (
    "gm1.cuda-rk1-post-scalar-momentum-boundary/v1"
)
CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_ABI = (
    "gm1.cuda-rk1-post-scalar-momentum-boundary-f32/v1"
)
CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA = (
    "gm1.rk1-post-scalar-momentum-boundary-token/v1"
)
CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS = (
    "ADMITTED_LIVE_RK1_POST_SCALAR_MOMENTUM_BOUNDARIES"
)
CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_SCHEMA = (
    "gm1.rk1-post-sound-momentum-carrier-token/v1"
)
CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_STATUS = (
    "CHECKPOINTED_LIVE_RK1_POST_SOUND_MOMENTUM_CARRIERS"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA = (
    "gm1.tilestream-scalar-moist-rk1-token/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_SCALAR_MOIST_RK1"
)
CUDA_GM1_RK1_POST_SCALAR_BCWSFC_KERNEL = "gm1_rk1_post_scalar_bcwsfc_open_f32_v1"

CM1_OPEN_NGXY = 3
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_BC_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
AUTHORITY_INIT_TERRAIN_SHA256 = (
    "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1662-1680",
    "src/bc.F:528-1143",
    "src/bc.F:1151-1178",
    "src/init_terrain.F:160-230",
)
SOURCE_ORDER = (
    "unchanged post-sound stage u3d/v3d/w3d carrier token",
    "completed live scalar/moist RK1 token",
    "bcu(u3d): west,east,south,north",
    "bcv(v3d): south,north,west,east",
    "bcw(w3d,1): west,east,south,north then rigid bottom/top",
    "terrain bcwsfc replaces the rigid bottom on physical plus one-halo cells",
)
GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_CLAIM_LIMIT = (
    "one source-ordered live CM1 r21.1 selected serial post-scalar RK1 "
    "u3d/v3d/w3d open-family replay, rigid w faces, and terrain bcwsfc using "
    "source-derived open metric halos; no MPI, scalar/pressure boundary, TKE "
    "RK, completed broader stage, tiled global pressure solve, forecast, or "
    "five-metre meteorological product"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _operation_token(receipt: Mapping[str, object]) -> str:
    record = dict(receipt)
    record.pop("operation_token_sha256", None)
    return hashlib.sha256(_canonical(record)).hexdigest()


def _lower_sha(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex sha256")
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _require_velocity_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1PostSoundVelocityToken):
        raise TypeError("velocity_token must be the exact post-sound token")
    receipt = token.receipt
    hashes = receipt.get("velocity_sha256")
    shapes = receipt.get("velocity_shapes")
    if (
        receipt.get("schema") != CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT"
        or receipt.get("nrk") != 1
        or receipt.get("global_psolver_execution") is not True
        or receipt.get("global_psolver_tile_streamable") is not False
        or not isinstance(hashes, Mapping)
        or set(hashes) != {"u3d", "v3d", "w3d"}
        or not isinstance(shapes, Mapping)
        or set(shapes) != {"u3d", "v3d", "w3d"}
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-sound velocity token identity or output map changed")
    for name in hashes:
        _lower_sha(f"velocity_sha256[{name!r}]", hashes[name])
    _lower_sha("stage_identity_sha256", receipt.get("stage_identity_sha256"))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("post-sound velocity stream pointer is invalid")
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1Rk1PostSoundMomentumCarrierToken:
    """Exact host checkpoint of stage u3d/v3d/w3d immediately after sound."""

    receipt: Mapping[str, object]


def checkpoint_gm1_rk1_post_sound_momentum_carriers(
    velocity_token: Gm1PostSoundVelocityToken,
    arrays: Mapping[str, Any],
    *,
    allow_compact_validation: bool = False,
) -> Gm1Rk1PostSoundMomentumCarrierToken:
    """Bind stage momentum carriers without conflating averaged rru/rrv/rrw."""

    velocity = _require_velocity_token(velocity_token)
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be bool")
    if not isinstance(arrays, Mapping) or tuple(arrays) != ("u3d", "v3d", "w3d"):
        raise ValueError("post-sound momentum carrier inventory or order changed")
    np = importlib.import_module("numpy")
    u = arrays["u3d"]
    if (
        not isinstance(u, np.ndarray)
        or u.dtype != np.dtype("<f4")
        or u.ndim != 3
        or not u.flags.c_contiguous
    ):
        raise ValueError("post-sound u3d carrier must be host C-<f4")
    nk, ny, nx = u.shape[0], u.shape[1], u.shape[2] - 1
    expected = {
        "u3d": (nk, ny, nx + 1),
        "v3d": (nk, ny + 1, nx),
        "w3d": (nk + 1, ny, nx),
    }
    if allow_compact_validation:
        if min(nx, ny, nk) < 2:
            raise ValueError("compact post-sound momentum checkpoint is too small")
    elif (nx, ny, nk) != (300, 300, 75):
        raise ValueError("live post-sound momentum checkpoint must be 300x300x75")
    hashes: dict[str, str] = {}
    for name, shape in expected.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"post-sound {name} carrier must be finite C-<f4 {shape}")
        hashes[name] = _sha(value)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_STATUS,
        "nrk": 1,
        "stage_identity_sha256": velocity["stage_identity_sha256"],
        "post_sound_velocity_token_sha256": velocity["operation_token_sha256"],
        "stream_pointer": velocity["stream_pointer"],
        "captured_after_global_sound": True,
        "captured_before_post_acoustic_getdivx": True,
        "carrier_semantics": (
            "stage u3d/v3d/w3d; distinct from averaged rru/rrv/rrw"
        ),
        "output_shapes": {name: list(shape) for name, shape in expected.items()},
        "output_sha256": hashes,
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": ["src/solve2.F:1085-1173", "src/solve2.F:1662-1680"],
        },
        "claim_limit": (
            "host checkpoint of live RK1 post-sound stage momentum physical "
            "carriers only; no boundary, MPI, completed RK stage, or forecast"
        ),
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1Rk1PostSoundMomentumCarrierToken(MappingProxyType(receipt))


def _require_momentum_carrier_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1Rk1PostSoundMomentumCarrierToken):
        raise TypeError(
            "momentum_carrier_token must be the exact post-sound stage carrier token"
        )
    receipt = token.receipt
    shapes = receipt.get("output_shapes")
    hashes = receipt.get("output_sha256")
    if (
        receipt.get("schema")
        != CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_STATUS
        or receipt.get("nrk") != 1
        or receipt.get("captured_after_global_sound") is not True
        or receipt.get("captured_before_post_acoustic_getdivx") is not True
        or not isinstance(shapes, Mapping)
        or set(shapes) != {"u3d", "v3d", "w3d"}
        or not isinstance(hashes, Mapping)
        or set(hashes) != {"u3d", "v3d", "w3d"}
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-sound momentum carrier token changed")
    for name in hashes:
        _lower_sha(f"output_sha256[{name!r}]", hashes[name])
    return receipt


def _require_flux_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledSolve2PostAcousticFluxToken):
        raise TypeError("flux_token must be the exact post-acoustic flux token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-acoustic flux token identity changed")
    return receipt


def _require_scalar_token(token: object) -> Mapping[str, object]:
    try:
        module = importlib.import_module("cm1core.cuda_gm1_tilestream_scalar_moist_rk1")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "GM1-RK1-MOMENTUM-BOUNDARY-PREREQUISITE: exact live scalar/moist "
            "RK1 token type is unavailable"
        ) from error
    token_type = getattr(module, "Gm1TiledScalarMoistRk1Token", None)
    if token_type is None or not isinstance(token, token_type):
        raise TypeError("scalar_token must be the exact Gm1TiledScalarMoistRk1Token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS
        or receipt.get("completed_nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("scalar/moist RK1 token identity changed")
    return receipt


def _require_lineage(
    velocity_token: object,
    momentum_carrier_token: object,
    flux_token: object,
    scalar_token: object,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    velocity = _require_velocity_token(velocity_token)
    carrier = _require_momentum_carrier_token(momentum_carrier_token)
    flux = _require_flux_token(flux_token)
    scalar = _require_scalar_token(scalar_token)
    if (
        carrier.get("post_sound_velocity_token_sha256")
        != velocity.get("operation_token_sha256")
        or flux.get("post_sound_velocity_token_sha256")
        != velocity.get("operation_token_sha256")
        or carrier.get("stage_identity_sha256")
        != velocity.get("stage_identity_sha256")
        or scalar.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or flux.get("stage_identity_sha256") != velocity.get("stage_identity_sha256")
        or len(
            {
                velocity.get("stream_pointer"),
                carrier.get("stream_pointer"),
                flux.get("stream_pointer"),
                scalar.get("stream_pointer"),
            }
        )
        != 1
    ):
        raise ValueError("velocity, flux, and scalar RK lineage differ")
    return velocity, carrier, flux, scalar


def rk1_post_scalar_momentum_boundary_shapes(
    *, nx: int, ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    for name, value, minimum in (("nx", nx, 2), ("ny", ny, 2), ("nk", nk, 2)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    ng = CM1_OPEN_NGXY
    return MappingProxyType(
        {
            "u3d": (nk + 2, ny + 2 * ng, nx + 1 + 2 * ng),
            "v3d": (nk + 2, ny + 1 + 2 * ng, nx + 2 * ng),
            "w3d": (nk + 3, ny + 2 * ng, nx + 2 * ng),
            "gz": (ny, nx),
            "dzdx": (ny, nx),
            "dzdy": (ny, nx),
        }
    )


def _f32_add(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) + np.float32(right))


def _f32_mul(np: Any, left: Any, right: Any) -> Any:
    return np.float32(np.float32(left) * np.float32(right))


def cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace(
    arrays: Mapping[str, Any], *, nx: int, ny: int, nk: int
) -> None:
    """Independent default-REAL resident transcription of the replay."""

    shapes = rk1_post_scalar_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("momentum boundary inventory or order changed")
    np = importlib.import_module("numpy")
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or (name in {"u3d", "v3d", "w3d"} and not value.flags.writeable)
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"{name} must be finite host C-<f4 {shape}")
    cpu_bcu_open_inplace(arrays["u3d"], ni=nx, nj=ny, nk=nk)
    cpu_bcv_open_inplace(arrays["v3d"], ni=nx, nj=ny, nk=nk)
    cpu_bcw_open_inplace(arrays["w3d"], ni=nx, nj=ny, nk=nk, flag=1)
    ng = CM1_OPEN_NGXY
    u, v, w = arrays["u3d"], arrays["v3d"], arrays["w3d"]
    gz, dzdx, dzdy = arrays["gz"], arrays["dzdx"], arrays["dzdy"]
    half = np.float32(0.5)
    for j in range(0, ny + 2):
        metric_j = min(max(j - 1, 0), ny - 1)
        y = j + ng - 1
        for i in range(0, nx + 2):
            metric_i = min(max(i - 1, 0), nx - 1)
            x = i + ng - 1
            usum = _f32_add(np, u[1, y, x], u[1, y, x + 1])
            uterm = _f32_mul(np, usum, dzdx[metric_j, metric_i])
            vsum = _f32_add(np, v[1, y, x], v[1, y + 1, x])
            vterm = _f32_mul(np, vsum, dzdy[metric_j, metric_i])
            total = _f32_add(np, uterm, vterm)
            w[1, y, x] = _f32_mul(
                np,
                _f32_mul(np, half, total),
                gz[metric_j, metric_i],
            )


@dataclass(frozen=True, slots=True)
class Gm1Rk1PostScalarMomentumBoundaryBinding:
    arrays: Mapping[str, Any]
    nx: int
    ny: int
    nk: int
    stream_pointer: int
    stage_identity_sha256: str
    velocity_token_sha256: str
    momentum_carrier_token_sha256: str
    flux_token_sha256: str
    scalar_token_sha256: str
    input_sha256: Mapping[str, str]
    expected_output_sha256: Mapping[str, str]
    metric_sha256: Mapping[str, str]
    compact_validation: bool

    @classmethod
    def create(
        cls,
        velocity_token: Gm1PostSoundVelocityToken,
        momentum_carrier_token: Gm1Rk1PostSoundMomentumCarrierToken,
        flux_token: Gm1TiledSolve2PostAcousticFluxToken,
        scalar_token: object,
        arrays: Mapping[str, Any],
        *,
        allow_compact_validation: bool = False,
    ) -> "Gm1Rk1PostScalarMomentumBoundaryBinding":
        velocity, carrier, flux, scalar = _require_lineage(
            velocity_token, momentum_carrier_token, flux_token, scalar_token
        )
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        if not isinstance(arrays, Mapping) or tuple(arrays) != (
            "u3d",
            "v3d",
            "w3d",
            "gz",
            "dzdx",
            "dzdy",
        ):
            raise ValueError("momentum boundary inventory or order changed")
        np = importlib.import_module("numpy")
        u = arrays["u3d"]
        if (
            not isinstance(u, np.ndarray)
            or u.dtype != np.dtype("<f4")
            or u.ndim != 3
            or not u.flags.c_contiguous
        ):
            raise ValueError("u3d must be host padded C-<f4")
        nk, ny, nx = (
            u.shape[0] - 2,
            u.shape[1] - 2 * CM1_OPEN_NGXY,
            u.shape[2] - 1 - 2 * CM1_OPEN_NGXY,
        )
        shapes = rk1_post_scalar_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)
        values = []
        for name, shape in shapes.items():
            value = arrays[name]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.dtype("<f4")
                or value.shape != shape
                or not value.flags.c_contiguous
                or (name in {"u3d", "v3d", "w3d"} and not value.flags.writeable)
                or not bool(np.all(np.isfinite(value)))
            ):
                raise ValueError(f"{name} must be finite host C-<f4 {shape}")
            values.append(value)
        for index, left in enumerate(values):
            if any(np.shares_memory(left, right) for right in values[index + 1 :]):
                raise ValueError("momentum boundary arrays/metrics must not alias")
        if allow_compact_validation:
            if min(nx, ny, nk) < 2:
                raise ValueError("compact momentum boundary domain is too small")
        elif (nx, ny, nk) != (300, 300, 75):
            raise ValueError("live momentum boundary domain must be 300x300x75")
        expected_shapes = {
            "u3d": [nk, ny, nx + 1],
            "v3d": [nk, ny + 1, nx],
            "w3d": [nk + 1, ny, nx],
        }
        if carrier.get("output_shapes") != expected_shapes:
            raise ValueError("post-sound momentum carrier shape differs from padded carriers")
        ng = CM1_OPEN_NGXY
        physical = {
            "u3d": arrays["u3d"][1 : nk + 1, ng : ng + ny, ng : ng + nx + 1],
            "v3d": arrays["v3d"][1 : nk + 1, ng : ng + ny + 1, ng : ng + nx],
            "w3d": arrays["w3d"][1 : nk + 2, ng : ng + ny, ng : ng + nx],
        }
        for name, value in physical.items():
            if _sha(value) != carrier["output_sha256"][name]:  # type: ignore[index]
                raise ValueError(
                    f"padded {name} physical words differ from post-sound carrier token"
                )
        input_hashes = {name: _sha(arrays[name]) for name in ("u3d", "v3d", "w3d")}
        metric_hashes = {name: _sha(arrays[name]) for name in ("gz", "dzdx", "dzdy")}
        reference = {name: arrays[name].copy() for name in arrays}
        cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace(
            reference, nx=nx, ny=ny, nk=nk
        )
        expected_hashes = {
            name: _sha(reference[name]) for name in ("u3d", "v3d", "w3d")
        }
        return cls(
            MappingProxyType(dict(arrays)),
            nx,
            ny,
            nk,
            int(scalar["stream_pointer"]),
            str(velocity["stage_identity_sha256"]),
            str(velocity["operation_token_sha256"]),
            str(carrier["operation_token_sha256"]),
            str(flux["operation_token_sha256"]),
            str(scalar["operation_token_sha256"]),
            MappingProxyType(input_hashes),
            MappingProxyType(expected_hashes),
            MappingProxyType(metric_hashes),
            allow_compact_validation,
        )


CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE = r"""// gm1.cuda-rk1-post-scalar-momentum-boundary-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void gm1_rk1_post_scalar_bcwsfc_open_f32_v1(
 const float* u,const float* v,float* w,const float* gz,const float* dzdx,const float* dzdy,
 unsigned int ni,unsigned int nj,unsigned int nk,unsigned int ng){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
 const unsigned long long n=(unsigned long long)(nj+2u)*(ni+2u);if(q>=n)return;
 const unsigned int i=(unsigned int)(q%(ni+2u)),j=(unsigned int)(q/(ni+2u));
 const unsigned int mi=(i==0u)?0u:((i==ni+1u)?ni-1u:i-1u),mj=(j==0u)?0u:((j==nj+1u)?nj-1u:j-1u);
 const unsigned int x=i+ng-1u,y=j+ng-1u,nxu=ni+1u+2u*ng,nyu=nj+2u*ng,nxv=ni+2u*ng,nyv=nj+1u+2u*ng,nxw=ni+2u*ng,nyw=nj+2u*ng;
 const float us=__fadd_rn(u[i3(1u,y,x,nyu,nxu)],u[i3(1u,y,x+1u,nyu,nxu)]),ut=__fmul_rn(us,dzdx[(unsigned long long)mj*ni+mi]);
 const float vs=__fadd_rn(v[i3(1u,y,x,nyv,nxv)],v[i3(1u,y+1u,x,nyv,nxv)]),vt=__fmul_rn(vs,dzdy[(unsigned long long)mj*ni+mi]);
 const float total=__fadd_rn(ut,vt),half=__fmul_rn(0.5f,total);
 w[i3(1u,y,x,nyw,nxw)]=__fmul_rn(half,gz[(unsigned long long)mj*ni+mi]);
}
"""
CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE.encode("ascii")
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1Rk1PostScalarMomentumBoundary:
    module_owner: Any
    bcwsfc_kernel: Any
    open_compiled: CompiledOpenBoundaryKernels
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_rk1_post_scalar_momentum_boundary(
    *, open_compiled: CompiledOpenBoundaryKernels | None = None
) -> CompiledGm1Rk1PostScalarMomentumBoundary:
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if _canonical_arch(compiler._get_arch()) != arch:
        raise ValueError("NVRTC target differs from current CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    ptx_options, cubin_options = _image_options(arch, int(nvrtc[0]))
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE,
        "gm1_rk1_post_scalar_momentum_boundary_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE,
        "gm1_rk1_post_scalar_momentum_boundary_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    if (
        re.search(
            rf"\.entry\s+{CUDA_GM1_RK1_POST_SCALAR_BCWSFC_KERNEL}(?:\s|\()",
            ptx.decode("utf-8"),
        )
        is None
    ):
        raise ValueError("compiled PTX lacks post-scalar bcwsfc entry")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled post-scalar boundary image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    open_compiled = open_compiled or compile_cuda_sound3_open_boundary()
    if open_compiled.target_arch != arch:
        raise ValueError("open-family and bcwsfc CUDA images target different devices")
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
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
    return CompiledGm1Rk1PostScalarMomentumBoundary(
        module,
        module.get_function(CUDA_GM1_RK1_POST_SCALAR_BCWSFC_KERNEL),
        open_compiled,
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1Rk1PostScalarMomentumBoundaryLaunch:
    ready_event: Any
    compiled: CompiledGm1Rk1PostScalarMomentumBoundary
    receipt: Mapping[str, object]


def launch_cuda_gm1_rk1_post_scalar_momentum_boundaries(
    velocity_token: Gm1PostSoundVelocityToken,
    momentum_carrier_token: Gm1Rk1PostSoundMomentumCarrierToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    scalar_token: object,
    arrays: Mapping[str, Any],
    *,
    nx: int,
    ny: int,
    nk: int,
    stream: Any,
    host_sync_before_launch: bool,
    compiled: CompiledGm1Rk1PostScalarMomentumBoundary | None = None,
) -> CudaGm1Rk1PostScalarMomentumBoundaryLaunch:
    velocity, carrier, flux, scalar = _require_lineage(
        velocity_token, momentum_carrier_token, flux_token, scalar_token
    )
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    if int(stream.ptr) != scalar.get("stream_pointer"):
        raise ValueError("momentum boundary replay left the scalar RK stream")
    if not isinstance(host_sync_before_launch, bool):
        raise TypeError("host_sync_before_launch must explicitly be bool")
    shapes = rk1_post_scalar_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA momentum boundary inventory or order changed")
    cupy = importlib.import_module("cupy")
    checked: dict[str, Any] = {}
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} must be device C-f32 {shape}")
        checked[name] = value
    if len({int(value.data.ptr) for value in checked.values()}) != len(checked):
        raise ValueError("CUDA momentum boundary arrays/metrics must not alias")
    compiled = compiled or compile_cuda_gm1_rk1_post_scalar_momentum_boundary()
    _, _, _, _, _, kfwe, kfsn, kwrigid, _, _ = compiled.open_compiled.kernels
    threads = (256, 1, 1)
    words = max(value.size for value in checked.values())
    grid = ((words + threads[0] - 1) // threads[0], 1, 1)
    edge_words = (ny + 2) * (nx + 2)
    edge_grid = ((edge_words + threads[0] - 1) // threads[0], 1, 1)
    u32 = cupy.uint32
    ng = u32(CM1_OPEN_NGXY)
    common = (u32(nx), u32(ny), u32(nk), ng)
    with stream:
        kfwe(grid, threads, (checked["u3d"], *common, u32(1)), stream=stream)
        kfsn(grid, threads, (checked["u3d"], *common, u32(1)), stream=stream)
        kfsn(grid, threads, (checked["v3d"], *common, u32(2)), stream=stream)
        kfwe(grid, threads, (checked["v3d"], *common, u32(2)), stream=stream)
        kfwe(grid, threads, (checked["w3d"], *common, u32(3)), stream=stream)
        kfsn(grid, threads, (checked["w3d"], *common, u32(3)), stream=stream)
        kwrigid(grid, threads, (checked["w3d"], *common), stream=stream)
        compiled.bcwsfc_kernel(
            edge_grid,
            threads,
            (
                checked["u3d"],
                checked["v3d"],
                checked["w3d"],
                checked["gz"],
                checked["dzdx"],
                checked["dzdy"],
                *common,
            ),
            stream=stream,
        )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SCHEMA,
        "abi": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_ABI,
        "status": "ENQUEUED_LIVE_RK1_POST_SCALAR_MOMENTUM_BOUNDARIES",
        "completed_nrk": 1,
        "stage_identity_sha256": velocity["stage_identity_sha256"],
        "post_sound_velocity_token_sha256": velocity["operation_token_sha256"],
        "post_sound_momentum_carrier_token_sha256": carrier[
            "operation_token_sha256"
        ],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "scalar_moist_rk1_token_sha256": scalar["operation_token_sha256"],
        "stream_pointer": int(stream.ptr),
        "same_stream_after_scalar_rk": True,
        "host_sync_before_launch": host_sync_before_launch,
        "source_order": list(SOURCE_ORDER),
        "domain": {
            "nx": nx,
            "ny": ny,
            "nk": nk,
            "ngxy": CM1_OPEN_NGXY,
            "terrain": True,
            "mpi": False,
        },
        "metric_halo_policy": (
            "serial four-open init_terrain bc2d constant extension, evaluated "
            "by edge clamping the source-valid physical metrics"
        ),
        "kernel": {
            "bcwsfc_source_sha256": (
                CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256
            ),
            "open_family_source_sha256": CUDA_SOUND3_OPEN_SOURCE_SHA256,
            "target_arch": compiled.target_arch,
        },
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "init_terrain_sha256": AUTHORITY_INIT_TERRAIN_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1Rk1PostScalarMomentumBoundaryLaunch(
        event, compiled, MappingProxyType(receipt)
    )


@dataclass(frozen=True, slots=True)
class Gm1Rk1PostScalarMomentumBoundaryToken:
    receipt: Mapping[str, object]


def finish_cuda_gm1_rk1_post_scalar_momentum_boundaries(
    binding: Gm1Rk1PostScalarMomentumBoundaryBinding,
    launch: CudaGm1Rk1PostScalarMomentumBoundaryLaunch,
) -> Gm1Rk1PostScalarMomentumBoundaryToken:
    """Validate drained resident arrays against the independent CPU replay."""

    if not isinstance(binding, Gm1Rk1PostScalarMomentumBoundaryBinding):
        raise TypeError("binding must be the exact momentum boundary binding")
    if not isinstance(launch, CudaGm1Rk1PostScalarMomentumBoundaryLaunch):
        raise TypeError("launch must be the exact momentum boundary launch")
    receipt = launch.receipt
    if (
        launch.ready_event is None
        or receipt.get("schema") != CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SCHEMA
        or receipt.get("status") != "ENQUEUED_LIVE_RK1_POST_SCALAR_MOMENTUM_BOUNDARIES"
        or receipt.get("completed_nrk") != 1
        or receipt.get("stage_identity_sha256") != binding.stage_identity_sha256
        or receipt.get("post_sound_velocity_token_sha256")
        != binding.velocity_token_sha256
        or receipt.get("post_sound_momentum_carrier_token_sha256")
        != binding.momentum_carrier_token_sha256
        or receipt.get("post_acoustic_flux_token_sha256") != binding.flux_token_sha256
        or receipt.get("scalar_moist_rk1_token_sha256") != binding.scalar_token_sha256
        or receipt.get("stream_pointer") != binding.stream_pointer
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("momentum boundary launch identity or lineage changed")
    output_hashes = {name: _sha(binding.arrays[name]) for name in ("u3d", "v3d", "w3d")}
    if output_hashes != dict(binding.expected_output_sha256):
        raise ValueError(
            "CUDA momentum boundary output differs raw-bit from CPU replay"
        )
    for name in ("gz", "dzdx", "dzdy"):
        if _sha(binding.arrays[name]) != binding.metric_sha256[name]:
            raise ValueError(f"momentum boundary mutated static metric {name}")
    token_receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS,
        "completed_nrk": 1,
        "stage_identity_sha256": binding.stage_identity_sha256,
        "post_sound_velocity_token_sha256": binding.velocity_token_sha256,
        "post_sound_momentum_carrier_token_sha256": (
            binding.momentum_carrier_token_sha256
        ),
        "post_acoustic_flux_token_sha256": binding.flux_token_sha256,
        "scalar_moist_rk1_token_sha256": binding.scalar_token_sha256,
        "stream_pointer": binding.stream_pointer,
        "input_sha256": dict(binding.input_sha256),
        "metric_sha256": dict(binding.metric_sha256),
        "output_sha256": output_hashes,
        "comparison": {
            "mode": "raw-bit",
            "max_ulp": 0,
            "resident_cpu_reference": True,
            "mismatch_words": 0,
        },
        "source_order": list(SOURCE_ORDER),
        "component_key": "momentum_boundary",
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "mpi_performed": False,
        "scalar_pressure_boundaries_performed": False,
        "tke_rk_aggregated": False,
        "global_psolver_execution_bound": True,
        "global_psolver_tile_streamable": False,
        "full_forecast": False,
        "five_meter_meteorological_product": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "init_terrain_sha256": AUTHORITY_INIT_TERRAIN_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_CLAIM_LIMIT,
    }
    token_receipt["operation_token_sha256"] = _operation_token(token_receipt)
    return Gm1Rk1PostScalarMomentumBoundaryToken(MappingProxyType(token_receipt))


def five_meter_rk1_post_scalar_momentum_boundary_inventory(
    *, nx: int = 3000, ny: int = 3000, nk: int = 75
) -> Mapping[str, object]:
    shapes = rk1_post_scalar_momentum_boundary_shapes(nx=nx, ny=ny, nk=nk)
    resident_bytes = 4 * sum(math.prod(shapes[name]) for name in ("u3d", "v3d", "w3d"))
    metric_bytes = 4 * sum(math.prod(shapes[name]) for name in ("gz", "dzdx", "dzdy"))
    record: dict[str, object] = {
        "schema": "gm1.rk1-post-scalar-momentum-boundary-memory/v1",
        "domain": [nk, ny, nx],
        "reuses_global_sound_velocity_residency": True,
        "existing_velocity_carrier_bytes": resident_bytes,
        "existing_physical_metric_bytes": metric_bytes,
        "incremental_device_workspace_bytes": 0,
        "metric_halo_allocation_bytes": 0,
        "boundary_replay_tile_local_math": True,
        "global_psolver_tile_streamable": False,
        "live_post_sound_token_available_at_five_meters": False,
        "completed_broader_rk_stage": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_BC_SHA256",
    "AUTHORITY_INIT_TERRAIN_SHA256",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_OPEN_NGXY",
    "CUDA_GM1_RK1_POST_SCALAR_BCWSFC_KERNEL",
    "CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_ABI",
    "CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SCHEMA",
    "CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE",
    "CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_SOURCE_SHA256",
    "CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA",
    "CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS",
    "CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_SCHEMA",
    "CUDA_GM1_RK1_POST_SOUND_MOMENTUM_CARRIER_TOKEN_STATUS",
    "CompiledGm1Rk1PostScalarMomentumBoundary",
    "CudaGm1Rk1PostScalarMomentumBoundaryLaunch",
    "GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_CLAIM_LIMIT",
    "Gm1Rk1PostScalarMomentumBoundaryBinding",
    "Gm1Rk1PostScalarMomentumBoundaryToken",
    "Gm1Rk1PostSoundMomentumCarrierToken",
    "checkpoint_gm1_rk1_post_sound_momentum_carriers",
    "compile_cuda_gm1_rk1_post_scalar_momentum_boundary",
    "cpu_gm1_rk1_post_scalar_momentum_boundaries_inplace",
    "finish_cuda_gm1_rk1_post_scalar_momentum_boundaries",
    "five_meter_rk1_post_scalar_momentum_boundary_inventory",
    "launch_cuda_gm1_rk1_post_scalar_momentum_boundaries",
    "rk1_post_scalar_momentum_boundary_shapes",
)
