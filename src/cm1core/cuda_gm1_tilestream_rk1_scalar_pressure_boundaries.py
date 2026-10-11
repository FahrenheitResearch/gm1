"""Exact tiled RK1 scalar/pressure boundary transition for GM1.

CM1 r21.1 completes theta and six moisture updates before entering its
pressure/density cadence.  For the selected moist, ``psolver=3``, ``idiff=0``
profile at RK1, the pressure/density interior block is an exact no-op.  The
serial four-open source order then publishes ``bcp(rho)``, ``bcp(pp3d)``,
``bcs(th3d)``, and ``bcs(q3d(:,:,:,1:6))``.  This module implements only that
post-scalar transition and binds it to the exact live scalar/moist RK1,
post-acoustic, and source-ordered momentum-boundary lineage.

The global acoustic pressure solver remains resident and explicitly is not
claimed tile-streamable.  The immediately preceding momentum-boundary token
is consumed but not relabelled as this scalar/pressure component.  MPI
exchanges, radiation history, TKE RK, a completed broader RK stage, forecast
output, and a five-metre meteorological product are outside this token.
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
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from .cuda_gm1_rk1_post_scalar_momentum_boundaries import (
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA,
    CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS,
    Gm1Rk1PostScalarMomentumBoundaryToken,
)
from .cuda_sound3_periodic import _canonical_arch, _image_options


CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SCHEMA = (
    "gm1.cuda-tilestream-rk1-scalar-pressure-boundary/v1"
)
CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_ABI = (
    "gm1.cuda-tilestream-rk1-scalar-pressure-boundary-f32/v1"
)
CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA = (
    "gm1.tilestream-rk1-scalar-pressure-boundary-token/v1"
)
CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARIES"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA = (
    "gm1.tilestream-scalar-moist-rk1-token/v1"
)
CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_SCALAR_MOIST_RK1"
)
CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_SCHEMA = (
    "gm1.rk1-post-sound-pressure-density-checkpoint-token/v1"
)
CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_STATUS = (
    "ADMITTED_LIVE_RK1_POST_SOUND_PRESSURE_DENSITY_CHECKPOINT"
)

CUDA_GM1_TILESTREAM_RK1_BCP_WE_KERNEL = "gm1_tiled_rk1_bcp_we_f32_v1"
CUDA_GM1_TILESTREAM_RK1_BCP_SN_KERNEL = "gm1_tiled_rk1_bcp_sn_f32_v1"
CUDA_GM1_TILESTREAM_RK1_BCS_WE_KERNEL = "gm1_tiled_rk1_bcs_we_f32_v1"
CUDA_GM1_TILESTREAM_RK1_BCS_SN_KERNEL = "gm1_tiled_rk1_bcs_sn_f32_v1"
CUDA_GM1_TILESTREAM_RK1_BOUNDARY_KERNEL_NAMES = (
    CUDA_GM1_TILESTREAM_RK1_BCP_WE_KERNEL,
    CUDA_GM1_TILESTREAM_RK1_BCP_SN_KERNEL,
    CUDA_GM1_TILESTREAM_RK1_BCS_WE_KERNEL,
    CUDA_GM1_TILESTREAM_RK1_BCS_SN_KERNEL,
)

CM1_OPEN_NGXY = 3
CM1_NUMQ = 6
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_BC_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"
AUTHORITY_ANCHORS = (
    "src/solve2.F:1535-1607",
    "src/solve2.F:1616-1656",
    "src/solve2.F:1662-1706",
    "src/bc.F:180-364",
    "src/bc.F:372-520",
)
SOURCE_ORDER = (
    "selected RK1 pressure/density interior cadence is an exact no-op",
    "consumed source-ordered post-scalar momentum-boundary token",
    "bcp(rho): west,east,south,north",
    "bcp(pp3d): west,east,south,north",
    "bcs(th3d): west,east,south,north",
    "bcs(q3d[qv,qc,qr,qi,qs,qg]) in species order",
)
GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_CLAIM_LIMIT = (
    "one source-ordered live CM1 r21.1 RK1 selected-profile pressure/density "
    "no-op followed by serial four-open rho/pp3d/theta/six-moisture boundary "
    "publication on physical levels after consuming the exact preceding "
    "momentum-boundary token; no MPI exchange, radiation history, TKE RK, "
    "completed broader RK stage, tiled global pressure solve, forecast, or "
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


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _require_velocity_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1PostSoundVelocityToken):
        raise TypeError("velocity_token must be the exact post-sound token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT"
        or receipt.get("nrk") != 1
        or receipt.get("global_psolver_execution") is not True
        or receipt.get("global_psolver_tile_streamable") is not False
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-sound velocity token identity or cadence changed")
    return receipt


def _require_flux_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledSolve2PostAcousticFluxToken):
        raise TypeError("flux_token must be the exact post-acoustic flux token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or receipt.get("nrk") != 1
        or receipt.get("global_psolver_execution") is not True
        or receipt.get("global_psolver_tile_streamable") is not False
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-acoustic flux token identity or cadence changed")
    return receipt


def _require_scalar_token(token: object) -> Mapping[str, object]:
    try:
        module = importlib.import_module("cm1core.cuda_gm1_tilestream_scalar_moist_rk1")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "GM1-RK1-SCALAR-BOUNDARY-PREREQUISITE: the exact live "
            "scalar/moist RK1 token type has not landed"
        ) from error
    token_type = getattr(module, "Gm1TiledScalarMoistRk1Token", None)
    if token_type is None or not isinstance(token, token_type):
        raise TypeError("scalar_token must be the exact Gm1TiledScalarMoistRk1Token")
    receipt = token.receipt
    hashes = receipt.get("output_sha256")
    shapes = receipt.get("output_shapes")
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS
        or receipt.get("completed_nrk") != 1
        or not isinstance(hashes, Mapping)
        or set(hashes) != {"th3d", "q3d"}
        or not isinstance(shapes, Mapping)
        or set(shapes) != {"th3d", "q3d"}
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("scalar/moist RK1 token identity or output map changed")
    _lower_sha("scalar th3d output", hashes["th3d"])
    _lower_sha("scalar q3d output", hashes["q3d"])
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1Rk1PostSoundPressureDensityToken:
    """Raw host checkpoint of pressure/density immediately after RK1 sound."""

    receipt: Mapping[str, object]


def checkpoint_gm1_rk1_post_sound_pressure_density(
    velocity_token: Gm1PostSoundVelocityToken,
    arrays: Mapping[str, Any],
    *,
    allow_compact_validation: bool = False,
) -> Gm1Rk1PostSoundPressureDensityToken:
    """Bind the two carriers that the selected RK1 interior leaves unchanged."""

    velocity = _require_velocity_token(velocity_token)
    if not isinstance(allow_compact_validation, bool):
        raise TypeError("allow_compact_validation must be bool")
    if not isinstance(arrays, Mapping) or tuple(arrays) != ("rho", "pp3d"):
        raise ValueError("pressure/density checkpoint inventory must be rho,pp3d")
    np = importlib.import_module("numpy")
    rho, pp3d = arrays["rho"], arrays["pp3d"]
    if (
        not isinstance(rho, np.ndarray)
        or rho.dtype != np.dtype("<f4")
        or rho.ndim != 3
        or not rho.flags.c_contiguous
    ):
        raise ValueError("checkpoint rho must be host C-<f4 [nk,ny,nx]")
    shape = tuple(rho.shape)
    if (
        not isinstance(pp3d, np.ndarray)
        or pp3d.dtype != np.dtype("<f4")
        or pp3d.shape != shape
        or not pp3d.flags.c_contiguous
    ):
        raise ValueError(f"checkpoint pp3d must be host C-<f4 {shape}")
    if not all(bool(np.all(np.isfinite(value))) for value in (rho, pp3d)):
        raise ValueError("pressure/density checkpoint contains non-finite words")
    if bool(np.any(rho <= np.float32(0.0))):
        raise ValueError("pressure/density checkpoint rho must be positive")
    nk, ny, nx = map(int, shape)
    if allow_compact_validation:
        if min(nx, ny, nk) < 2:
            raise ValueError("compact pressure/density checkpoint is too small")
    elif (nx, ny, nk) != (300, 300, 75):
        raise ValueError("live pressure/density checkpoint must be 300x300x75")
    if np.shares_memory(rho, pp3d):
        raise ValueError("checkpoint rho and pp3d must not alias")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_STATUS,
        "nrk": 1,
        "stage_identity_sha256": velocity["stage_identity_sha256"],
        "post_sound_velocity_token_sha256": velocity["operation_token_sha256"],
        "psolver_receipt_sha256": velocity["psolver_receipt_sha256"],
        "stream_pointer": velocity["stream_pointer"],
        "output_shapes": {"rho": list(shape), "pp3d": list(shape)},
        "output_sha256": {"rho": _sha(rho), "pp3d": _sha(pp3d)},
        "captured_after_sound_before_scalar_rk": True,
        "selected_rk1_pressure_density_recompute_expected": False,
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(AUTHORITY_ANCHORS[:2]),
        },
        "claim_limit": (
            "raw checkpoint of live RK1 post-sound rho/pp3d for the selected "
            "pressure/density no-op proof only; not a pressure solver or RK stage"
        ),
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1Rk1PostSoundPressureDensityToken(MappingProxyType(receipt))


def _require_pressure_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1Rk1PostSoundPressureDensityToken):
        raise TypeError("pressure_token must be the exact pressure checkpoint token")
    receipt = token.receipt
    hashes = receipt.get("output_sha256")
    if (
        receipt.get("schema") != CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_STATUS
        or receipt.get("nrk") != 1
        or receipt.get("captured_after_sound_before_scalar_rk") is not True
        or receipt.get("selected_rk1_pressure_density_recompute_expected") is not False
        or not isinstance(hashes, Mapping)
        or set(hashes) != {"rho", "pp3d"}
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("pressure/density checkpoint identity changed")
    return receipt


def _require_momentum_token(
    token: object,
    scalar: Mapping[str, object],
    flux: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(token, Gm1Rk1PostScalarMomentumBoundaryToken):
        raise TypeError(
            "momentum_token must be the exact post-scalar momentum boundary token"
        )
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_RK1_POST_SCALAR_MOMENTUM_BOUNDARY_TOKEN_STATUS
        or receipt.get("completed_nrk") != 1
        or receipt.get("source_order_component_complete") is not True
        or receipt.get("completed_broader_rk_stage") is not False
        or receipt.get("scalar_moist_rk1_token_sha256")
        != scalar.get("operation_token_sha256")
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or not isinstance(
            receipt.get("post_sound_momentum_carrier_token_sha256"), str
        )
        or len(receipt["post_sound_momentum_carrier_token_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt["post_sound_momentum_carrier_token_sha256"]
        )
        or receipt.get("stage_identity_sha256") != flux.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != scalar.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-scalar momentum boundary identity or lineage changed")
    return receipt


def _require_lineage(
    pressure_token: object,
    scalar_token: object,
    flux_token: object,
    momentum_token: object,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    pressure = _require_pressure_token(pressure_token)
    scalar = _require_scalar_token(scalar_token)
    flux = _require_flux_token(flux_token)
    momentum = _require_momentum_token(momentum_token, scalar, flux)
    if (
        scalar.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or flux.get("post_sound_velocity_token_sha256")
        != pressure.get("post_sound_velocity_token_sha256")
        or flux.get("stage_identity_sha256") != pressure.get("stage_identity_sha256")
        or len(
            {
                pressure.get("stream_pointer"),
                scalar.get("stream_pointer"),
                flux.get("stream_pointer"),
                momentum.get("stream_pointer"),
            }
        )
        != 1
    ):
        raise ValueError(
            "pressure, scalar RK, post-acoustic, and momentum lineage differ"
        )
    return pressure, scalar, flux, momentum


def tiled_rk1_scalar_pressure_boundary_buffer_shapes(
    *, tile_nx: int, tile_ny: int, nk: int
) -> Mapping[str, tuple[int, ...]]:
    tile_nx = _positive_int("tile_nx", tile_nx)
    tile_ny = _positive_int("tile_ny", tile_ny)
    nk = _positive_int("nk", nk)
    pressure = (nk, tile_ny + 2, tile_nx + 2)
    scalar = (
        nk,
        tile_ny + 2 * CM1_OPEN_NGXY,
        tile_nx + 2 * CM1_OPEN_NGXY,
    )
    return MappingProxyType(
        {
            "rho": pressure,
            "pp3d": pressure,
            "th3d": scalar,
            "q3d": (CM1_NUMQ, *scalar),
        }
    )


def _vertical_halo_sha(value: Any) -> str:
    digest = hashlib.sha256()
    if value.ndim == 3:
        digest.update(value[0].astype("<f4", copy=False).tobytes(order="C"))
        digest.update(value[-1].astype("<f4", copy=False).tobytes(order="C"))
    else:
        digest.update(value[:, 0].astype("<f4", copy=False).tobytes(order="C"))
        digest.update(value[:, -1].astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _corner_sources(value: Any, *, ny: int, nx: int) -> tuple[Any, Any, Any, Any]:
    ng = CM1_OPEN_NGXY
    if value.ndim == 3:
        core = value[1:-1]
        return tuple(
            item.copy()
            for item in (
                core[:, ng - 1, ng],
                core[:, ng + ny, ng],
                core[:, ng - 1, ng + nx - 1],
                core[:, ng + ny, ng + nx - 1],
            )
        )  # type: ignore[return-value]
    core = value[:, 1:-1]
    return tuple(
        item.copy()
        for item in (
            core[:, :, ng - 1, ng],
            core[:, :, ng + ny, ng],
            core[:, :, ng - 1, ng + nx - 1],
            core[:, :, ng + ny, ng + nx - 1],
        )
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk1ScalarPressureBoundaryBinding:
    physical: Mapping[str, Any]
    padded: Mapping[str, Any]
    nx: int
    ny: int
    nk: int
    stream_pointer: int
    stage_identity_sha256: str
    pressure_token_sha256: str
    scalar_token_sha256: str
    flux_token_sha256: str
    momentum_boundary_token_sha256: str
    source_identity_sha256: str
    vertical_halo_sha256: Mapping[str, str]
    scalar_corner_sources: Mapping[str, tuple[Any, Any, Any, Any]]
    compact_validation: bool

    @classmethod
    def create(
        cls,
        pressure_token: Gm1Rk1PostSoundPressureDensityToken,
        scalar_token: object,
        flux_token: Gm1TiledSolve2PostAcousticFluxToken,
        momentum_token: Gm1Rk1PostScalarMomentumBoundaryToken,
        *,
        physical: Mapping[str, Any],
        padded: Mapping[str, Any],
        allow_compact_validation: bool = False,
    ) -> "Gm1TiledRk1ScalarPressureBoundaryBinding":
        pressure, scalar, flux, momentum = _require_lineage(
            pressure_token, scalar_token, flux_token, momentum_token
        )
        if not isinstance(allow_compact_validation, bool):
            raise TypeError("allow_compact_validation must be bool")
        names = ("rho", "pp3d", "th3d", "q3d")
        if not isinstance(physical, Mapping) or tuple(physical) != names:
            raise ValueError("physical inventory must be rho,pp3d,th3d,q3d")
        if not isinstance(padded, Mapping) or tuple(padded) != names:
            raise ValueError("padded inventory must be rho,pp3d,th3d,q3d")
        np = importlib.import_module("numpy")
        rho = physical["rho"]
        if (
            not isinstance(rho, np.ndarray)
            or rho.dtype != np.dtype("<f4")
            or rho.ndim != 3
            or not rho.flags.c_contiguous
        ):
            raise ValueError("physical rho must be host C-<f4 [nk,ny,nx]")
        nk, ny, nx = map(int, rho.shape)
        physical_shapes = {
            "rho": (nk, ny, nx),
            "pp3d": (nk, ny, nx),
            "th3d": (nk, ny, nx),
            "q3d": (CM1_NUMQ, nk, ny, nx),
        }
        ng = CM1_OPEN_NGXY
        padded_shapes = {
            "rho": (nk + 2, ny + 2, nx + 2),
            "pp3d": (nk + 2, ny + 2, nx + 2),
            "th3d": (nk + 2, ny + 2 * ng, nx + 2 * ng),
            "q3d": (CM1_NUMQ, nk + 2, ny + 2 * ng, nx + 2 * ng),
        }
        all_values: list[Any] = []
        for inventory, shapes, writable in (
            (physical, physical_shapes, False),
            (padded, padded_shapes, True),
        ):
            for name in names:
                value = inventory[name]
                if (
                    not isinstance(value, np.ndarray)
                    or value.dtype != np.dtype("<f4")
                    or value.shape != shapes[name]
                    or not value.flags.c_contiguous
                    or (writable and not value.flags.writeable)
                ):
                    role = "padded" if writable else "physical"
                    raise ValueError(f"{role} {name} must be host C-<f4 {shapes[name]}")
                if not bool(np.all(np.isfinite(value))):
                    raise ValueError(f"{name} contains non-finite words")
                all_values.append(value)
        for index, left in enumerate(all_values):
            if any(np.shares_memory(left, right) for right in all_values[index + 1 :]):
                raise ValueError("boundary physical/padded carriers must not alias")
        if allow_compact_validation:
            if min(nx, ny, nk) < 2:
                raise ValueError("compact boundary domain is too small")
        elif (nx, ny, nk) != (300, 300, 75):
            raise ValueError("live boundary domain must be 300x300x75")
        if pressure.get("output_shapes") != {
            "rho": [nk, ny, nx],
            "pp3d": [nk, ny, nx],
        }:
            raise ValueError("pressure checkpoint shape differs from boundary state")
        if scalar.get("output_shapes") != {
            "th3d": [nk, ny, nx],
            "q3d": [CM1_NUMQ, nk, ny, nx],
        }:
            raise ValueError("scalar RK output shape differs from boundary state")
        expected_hashes = {
            "rho": pressure["output_sha256"]["rho"],  # type: ignore[index]
            "pp3d": pressure["output_sha256"]["pp3d"],  # type: ignore[index]
            "th3d": scalar["output_sha256"]["th3d"],  # type: ignore[index]
            "q3d": scalar["output_sha256"]["q3d"],  # type: ignore[index]
        }
        for name in names:
            if _sha(physical[name]) != expected_hashes[name]:
                raise ValueError(
                    f"physical {name} differs from its live producer token"
                )
        source_record = {
            "pressure_token_sha256": pressure["operation_token_sha256"],
            "scalar_token_sha256": scalar["operation_token_sha256"],
            "flux_token_sha256": flux["operation_token_sha256"],
            "momentum_boundary_token_sha256": momentum["operation_token_sha256"],
            "physical_sha256": expected_hashes,
        }
        return cls(
            MappingProxyType(dict(physical)),
            MappingProxyType(dict(padded)),
            nx,
            ny,
            nk,
            int(scalar["stream_pointer"]),
            str(flux["stage_identity_sha256"]),
            str(pressure["operation_token_sha256"]),
            str(scalar["operation_token_sha256"]),
            str(flux["operation_token_sha256"]),
            str(momentum["operation_token_sha256"]),
            hashlib.sha256(_canonical(source_record)).hexdigest(),
            MappingProxyType(
                {name: _vertical_halo_sha(padded[name]) for name in names}
            ),
            MappingProxyType(
                {
                    name: _corner_sources(padded[name], ny=ny, nx=nx)
                    for name in ("th3d", "q3d")
                }
            ),
            allow_compact_validation,
        )


def fill_gm1_tiled_rk1_scalar_pressure_boundary_source(
    binding: Gm1TiledRk1ScalarPressureBoundaryBinding,
    spec: Gm1TileSpec,
    target: Mapping[str, Any],
) -> Mapping[str, object]:
    """Gather one owned physical tile plus its existing boundary padding."""

    if not isinstance(binding, Gm1TiledRk1ScalarPressureBoundaryBinding):
        raise TypeError("binding must be the exact RK1 boundary binding")
    if not isinstance(spec, Gm1TileSpec) or (spec.nx, spec.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("tile spec differs from boundary domain")
    shapes = tiled_rk1_scalar_pressure_boundary_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(target, Mapping) or tuple(target) != tuple(shapes):
        raise ValueError("boundary tile inventory or order changed")
    np = importlib.import_module("numpy")
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
            raise ValueError(f"{name} tile must be writable host C-<f4 {shape}")
        value.fill(np.float32(0.0))
    for name in ("rho", "pp3d"):
        target[name][:, : active_ny + 2, : active_nx + 2] = binding.padded[name][
            1 : binding.nk + 1, y0 : y1 + 2, x0 : x1 + 2
        ]
        target[name][:, 1 : active_ny + 1, 1 : active_nx + 1] = binding.physical[name][
            :, y0:y1, x0:x1
        ]
    ng = CM1_OPEN_NGXY
    target["th3d"][:, : active_ny + 2 * ng, : active_nx + 2 * ng] = binding.padded[
        "th3d"
    ][
        1 : binding.nk + 1,
        y0 : y1 + 2 * ng,
        x0 : x1 + 2 * ng,
    ]
    target["th3d"][:, ng : ng + active_ny, ng : ng + active_nx] = binding.physical[
        "th3d"
    ][:, y0:y1, x0:x1]
    target["q3d"][:, :, : active_ny + 2 * ng, : active_nx + 2 * ng] = binding.padded[
        "q3d"
    ][
        :,
        1 : binding.nk + 1,
        y0 : y1 + 2 * ng,
        x0 : x1 + 2 * ng,
    ]
    target["q3d"][:, :, ng : ng + active_ny, ng : ng + active_nx] = binding.physical[
        "q3d"
    ][:, :, y0:y1, x0:x1]
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
            "preexisting_padding_preserved_for_corner_order": True,
        }
    )


def _cpu_bcp(
    value: Any,
    *,
    active_nx: int,
    active_ny: int,
    west: bool,
    east: bool,
    south: bool,
    north: bool,
) -> None:
    if west:
        value[:, : active_ny + 2, 0] = value[:, : active_ny + 2, 1]
    if east:
        value[:, : active_ny + 2, active_nx + 1] = value[:, : active_ny + 2, active_nx]
    if south:
        value[:, 0, : active_nx + 2] = value[:, 1, : active_nx + 2]
    if north:
        value[:, active_ny + 1, : active_nx + 2] = value[:, active_ny, : active_nx + 2]


def _cpu_bcs(
    value: Any,
    *,
    active_nx: int,
    active_ny: int,
    west: bool,
    east: bool,
    south: bool,
    north: bool,
) -> None:
    ng = CM1_OPEN_NGXY
    if west:
        value[..., ng - 1 : ng + active_ny + 1, :ng] = value[
            ..., ng - 1 : ng + active_ny + 1, ng : ng + 1
        ]
    if east:
        value[
            ...,
            ng - 1 : ng + active_ny + 1,
            ng + active_nx : ng + active_nx + ng,
        ] = value[
            ...,
            ng - 1 : ng + active_ny + 1,
            ng + active_nx - 1 : ng + active_nx,
        ]
    if south:
        value[..., :ng, ng - 1 : ng + active_nx + 1] = value[
            ..., ng : ng + 1, ng - 1 : ng + active_nx + 1
        ]
    if north:
        value[
            ...,
            ng + active_ny : ng + active_ny + ng,
            ng - 1 : ng + active_nx + 1,
        ] = value[
            ...,
            ng + active_ny - 1 : ng + active_ny,
            ng - 1 : ng + active_nx + 1,
        ]


def cpu_gm1_tiled_rk1_scalar_pressure_boundaries_inplace(
    arrays: Mapping[str, Any],
    *,
    active_nx: int,
    active_ny: int,
    west: bool,
    east: bool,
    south: bool,
    north: bool,
) -> None:
    """Literal selected solve2 order for one physical-boundary tile."""

    shapes = ("rho", "pp3d", "th3d", "q3d")
    if not isinstance(arrays, Mapping) or tuple(arrays) != shapes:
        raise ValueError("CPU boundary inventory must be rho,pp3d,th3d,q3d")
    active_nx = _positive_int("active_nx", active_nx)
    active_ny = _positive_int("active_ny", active_ny)
    np = importlib.import_module("numpy")
    for name, value in arrays.items():
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or not value.flags.c_contiguous
            or not value.flags.writeable
        ):
            raise ValueError(f"CPU {name} must be writable host C-<f4")
    _cpu_bcp(
        arrays["rho"],
        active_nx=active_nx,
        active_ny=active_ny,
        west=west,
        east=east,
        south=south,
        north=north,
    )
    _cpu_bcp(
        arrays["pp3d"],
        active_nx=active_nx,
        active_ny=active_ny,
        west=west,
        east=east,
        south=south,
        north=north,
    )
    _cpu_bcs(
        arrays["th3d"],
        active_nx=active_nx,
        active_ny=active_ny,
        west=west,
        east=east,
        south=south,
        north=north,
    )
    for species in range(CM1_NUMQ):
        _cpu_bcs(
            arrays["q3d"][species],
            active_nx=active_nx,
            active_ny=active_ny,
            west=west,
            east=east,
            south=south,
            north=north,
        )


def scatter_gm1_tiled_rk1_scalar_pressure_boundary_output(
    binding: Gm1TiledRk1ScalarPressureBoundaryBinding,
    spec: Gm1TileSpec,
    source: Mapping[str, Any],
) -> Mapping[str, object]:
    """Publish every physical word and each source-written lateral halo once."""

    if not isinstance(binding, Gm1TiledRk1ScalarPressureBoundaryBinding):
        raise TypeError("binding must be the exact RK1 boundary binding")
    shapes = tiled_rk1_scalar_pressure_boundary_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=binding.nk
    )
    if not isinstance(source, Mapping) or tuple(source) != tuple(shapes):
        raise ValueError("boundary scatter inventory or order changed")
    np = importlib.import_module("numpy")
    for name, shape in shapes.items():
        value = source[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"{name} drain must be host C-<f4 {shape}")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    for name in ("rho", "pp3d"):
        destination = binding.padded[name]
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
            lx0, gx0 = (0, 0) if x0 == 0 else (1, x0 + 1)
            lx1 = active_nx + 2 if x1 == binding.nx else active_nx + 1
            gx1 = binding.nx + 2 if x1 == binding.nx else x1 + 1
            destination[1 : binding.nk + 1, 0, gx0:gx1] = tile[:, 0, lx0:lx1]
        if y1 == binding.ny:
            lx0, gx0 = (0, 0) if x0 == 0 else (1, x0 + 1)
            lx1 = active_nx + 2 if x1 == binding.nx else active_nx + 1
            gx1 = binding.nx + 2 if x1 == binding.nx else x1 + 1
            destination[1 : binding.nk + 1, -1, gx0:gx1] = tile[
                :, active_ny + 1, lx0:lx1
            ]

    ng = CM1_OPEN_NGXY
    for name in ("th3d", "q3d"):
        destination = binding.padded[name]
        tile = source[name]
        lead = (slice(None),) if name == "q3d" else ()
        destination[
            *lead,
            slice(1, binding.nk + 1),
            slice(ng + y0, ng + y1),
            slice(ng + x0, ng + x1),
        ] = tile[
            *lead,
            slice(None),
            slice(ng, ng + active_ny),
            slice(ng, ng + active_nx),
        ]
        if x0 == 0:
            destination[
                *lead,
                slice(1, binding.nk + 1),
                slice(ng + y0, ng + y1),
                slice(0, ng),
            ] = tile[
                *lead,
                slice(None),
                slice(ng, ng + active_ny),
                slice(0, ng),
            ]
        if x1 == binding.nx:
            destination[
                *lead,
                slice(1, binding.nk + 1),
                slice(ng + y0, ng + y1),
                slice(ng + binding.nx, ng + binding.nx + ng),
            ] = tile[
                *lead,
                slice(None),
                slice(ng, ng + active_ny),
                slice(ng + active_nx, ng + active_nx + ng),
            ]
        lx0 = ng - 1 if x0 == 0 else ng
        gx0 = ng - 1 if x0 == 0 else ng + x0
        lx1 = ng + active_nx + 1 if x1 == binding.nx else ng + active_nx
        gx1 = ng + binding.nx + 1 if x1 == binding.nx else ng + x1
        if y0 == 0:
            destination[
                *lead,
                slice(1, binding.nk + 1),
                slice(0, ng),
                slice(gx0, gx1),
            ] = tile[
                *lead,
                slice(None),
                slice(0, ng),
                slice(lx0, lx1),
            ]
            if x0 == 0:
                destination[
                    *lead,
                    slice(1, binding.nk + 1),
                    ng - 1,
                    slice(0, ng - 1),
                ] = tile[
                    *lead,
                    slice(None),
                    ng - 1,
                    slice(0, ng - 1),
                ]
            if x1 == binding.nx:
                destination[
                    *lead,
                    slice(1, binding.nk + 1),
                    ng - 1,
                    slice(ng + binding.nx + 1, ng + binding.nx + ng),
                ] = tile[
                    *lead,
                    slice(None),
                    ng - 1,
                    slice(ng + active_nx + 1, ng + active_nx + ng),
                ]
        if y1 == binding.ny:
            destination[
                *lead,
                slice(1, binding.nk + 1),
                slice(ng + binding.ny, ng + binding.ny + ng),
                slice(gx0, gx1),
            ] = tile[
                *lead,
                slice(None),
                slice(ng + active_ny, ng + active_ny + ng),
                slice(lx0, lx1),
            ]
            if x0 == 0:
                destination[
                    *lead,
                    slice(1, binding.nk + 1),
                    ng + binding.ny,
                    slice(0, ng - 1),
                ] = tile[
                    *lead,
                    slice(None),
                    ng + active_ny,
                    slice(0, ng - 1),
                ]
            if x1 == binding.nx:
                destination[
                    *lead,
                    slice(1, binding.nk + 1),
                    ng + binding.ny,
                    slice(ng + binding.nx + 1, ng + binding.nx + ng),
                ] = tile[
                    *lead,
                    slice(None),
                    ng + active_ny,
                    slice(ng + active_nx + 1, ng + active_nx + ng),
                ]
    return MappingProxyType(
        {
            "tile": spec.index,
            "owned_yx": [y0, y1, x0, x1],
            "physical_words": (2 + 1 + CM1_NUMQ) * binding.nk * active_ny * active_nx,
            "horizontal_halo_only": True,
            "vertical_halo_mutated": False,
        }
    )


CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE = r"""// gm1.cuda-tilestream-rk1-scalar-pressure-boundary-f32/v1
static_assert(sizeof(float)==4,"CM1 default REAL requires float32");
__device__ __forceinline__ unsigned long long i3(unsigned int k,unsigned int j,unsigned int i,unsigned int ny,unsigned int nx){return ((unsigned long long)k*ny+j)*nx+i;}
extern "C" __global__ void gm1_tiled_rk1_bcp_we_f32_v1(float* p,unsigned int ax,unsigned int ay,unsigned int nk,unsigned int tx,unsigned int ty,unsigned int west,unsigned int east){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nk*(ay+2u);if(q>=n)return;
 const unsigned int j=(unsigned int)(q%(ay+2u)),k=(unsigned int)(q/(ay+2u)),nx=tx+2u,ny=ty+2u;
 if(west)p[i3(k,j,0u,ny,nx)]=p[i3(k,j,1u,ny,nx)];if(east)p[i3(k,j,ax+1u,ny,nx)]=p[i3(k,j,ax,ny,nx)];
}
extern "C" __global__ void gm1_tiled_rk1_bcp_sn_f32_v1(float* p,unsigned int ax,unsigned int ay,unsigned int nk,unsigned int tx,unsigned int ty,unsigned int south,unsigned int north){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nk*(ax+2u);if(q>=n)return;
 const unsigned int i=(unsigned int)(q%(ax+2u)),k=(unsigned int)(q/(ax+2u)),nx=tx+2u,ny=ty+2u;
 if(south)p[i3(k,0u,i,ny,nx)]=p[i3(k,1u,i,ny,nx)];if(north)p[i3(k,ay+1u,i,ny,nx)]=p[i3(k,ay,i,ny,nx)];
}
extern "C" __global__ void gm1_tiled_rk1_bcs_we_f32_v1(float* a,unsigned int ax,unsigned int ay,unsigned int nk,unsigned int tx,unsigned int ty,unsigned int ng,unsigned int west,unsigned int east){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nk*(ay+2u)*ng;if(q>=n)return;
 unsigned long long r=q;const unsigned int layer=(unsigned int)(r%ng);r/=ng;const unsigned int j0=(unsigned int)(r%(ay+2u)),k=(unsigned int)(r/(ay+2u));
 const unsigned int j=j0+ng-1u,nx=tx+2u*ng,ny=ty+2u*ng;if(west)a[i3(k,j,layer,ny,nx)]=a[i3(k,j,ng,ny,nx)];if(east)a[i3(k,j,ng+ax+layer,ny,nx)]=a[i3(k,j,ng+ax-1u,ny,nx)];
}
extern "C" __global__ void gm1_tiled_rk1_bcs_sn_f32_v1(float* a,unsigned int ax,unsigned int ay,unsigned int nk,unsigned int tx,unsigned int ty,unsigned int ng,unsigned int south,unsigned int north){
 const unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,n=(unsigned long long)nk*(ax+2u)*ng;if(q>=n)return;
 unsigned long long r=q;const unsigned int layer=(unsigned int)(r%ng);r/=ng;const unsigned int i0=(unsigned int)(r%(ax+2u)),k=(unsigned int)(r/(ax+2u));
 const unsigned int i=i0+ng-1u,nx=tx+2u*ng,ny=ty+2u*ng;if(south)a[i3(k,layer,i,ny,nx)]=a[i3(k,ng,i,ny,nx)];if(north)a[i3(k,ng+ay+layer,i,ny,nx)]=a[i3(k,ng+ay-1u,i,ny,nx)];
}
"""
CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE_SHA256 = hashlib.sha256(
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE.encode("ascii")
).hexdigest()


@dataclass(frozen=True, slots=True)
class CompiledGm1TiledRk1ScalarPressureBoundary:
    module_owner: Any
    kernels: tuple[Any, Any, Any, Any]
    ptx: bytes
    cubin: bytes
    target_arch: str
    runtime_identity: tuple[tuple[str, str], ...]


def compile_cuda_gm1_tiled_rk1_scalar_pressure_boundary() -> (
    CompiledGm1TiledRk1ScalarPressureBoundary
):
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
        CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE,
        "gm1_tiled_rk1_scalar_pressure_boundary_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE,
        "gm1_tiled_rk1_scalar_pressure_boundary_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode() if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    text = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", text)
        for name in CUDA_GM1_TILESTREAM_RK1_BOUNDARY_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks RK1 scalar/pressure boundary entries")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled RK1 boundary image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
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
    return CompiledGm1TiledRk1ScalarPressureBoundary(
        module,
        tuple(
            module.get_function(name)
            for name in CUDA_GM1_TILESTREAM_RK1_BOUNDARY_KERNEL_NAMES
        ),
        ptx,
        cubin,
        arch,
        runtime,
    )


@dataclass(frozen=True, slots=True)
class CudaGm1TiledRk1ScalarPressureBoundaryLaunch:
    ready_event: Any
    compiled: CompiledGm1TiledRk1ScalarPressureBoundary
    receipt: Mapping[str, object]


def launch_cuda_gm1_tiled_rk1_scalar_pressure_boundary(
    pressure_token: Gm1Rk1PostSoundPressureDensityToken,
    scalar_token: object,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    momentum_token: Gm1Rk1PostScalarMomentumBoundaryToken,
    arrays: Mapping[str, Any],
    spec: Gm1TileSpec,
    *,
    nk: int,
    stream: Any,
    compiled: CompiledGm1TiledRk1ScalarPressureBoundary | None = None,
) -> CudaGm1TiledRk1ScalarPressureBoundaryLaunch:
    pressure, scalar, flux, momentum = _require_lineage(
        pressure_token, scalar_token, flux_token, momentum_token
    )
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    nk = _positive_int("nk", nk)
    if stream is None or not hasattr(stream, "ptr") or int(stream.ptr) == 0:
        raise TypeError("stream must be an explicit non-default CUDA stream")
    if int(stream.ptr) != scalar.get("stream_pointer"):
        raise ValueError("boundary launch left the scalar RK CUDA stream")
    cupy = importlib.import_module("cupy")
    shapes = tiled_rk1_scalar_pressure_boundary_buffer_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nk=nk
    )
    if not isinstance(arrays, Mapping) or tuple(arrays) != tuple(shapes):
        raise ValueError("CUDA boundary tile inventory or order changed")
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
        raise ValueError("CUDA boundary tile arrays must not alias")
    y0, y1, x0, x1 = spec.owned_rect("mass")
    active_ny, active_nx = y1 - y0, x1 - x0
    flags = (x0 == 0, x1 == spec.nx, y0 == 0, y1 == spec.ny)
    compiled = compiled or compile_cuda_gm1_tiled_rk1_scalar_pressure_boundary()
    threads = (256, 1, 1)
    u32 = cupy.uint32
    common = (
        u32(active_nx),
        u32(active_ny),
        u32(nk),
        u32(spec.tile_nx),
        u32(spec.tile_ny),
    )
    bcp_we_words = nk * (active_ny + 2)
    bcp_sn_words = nk * (active_nx + 2)
    bcs_we_words = nk * (active_ny + 2) * CM1_OPEN_NGXY
    bcs_sn_words = nk * (active_nx + 2) * CM1_OPEN_NGXY

    def grid(words: int) -> tuple[int, int, int]:
        return ((words + threads[0] - 1) // threads[0], 1, 1)

    with stream:
        for carrier in (checked["rho"], checked["pp3d"]):
            compiled.kernels[0](
                grid(bcp_we_words),
                threads,
                (carrier, *common, u32(flags[0]), u32(flags[1])),
                stream=stream,
            )
            compiled.kernels[1](
                grid(bcp_sn_words),
                threads,
                (carrier, *common, u32(flags[2]), u32(flags[3])),
                stream=stream,
            )
        for carrier in (
            checked["th3d"],
            *tuple(checked["q3d"][n] for n in range(CM1_NUMQ)),
        ):
            compiled.kernels[2](
                grid(bcs_we_words),
                threads,
                (
                    carrier,
                    *common,
                    u32(CM1_OPEN_NGXY),
                    u32(flags[0]),
                    u32(flags[1]),
                ),
                stream=stream,
            )
            compiled.kernels[3](
                grid(bcs_sn_words),
                threads,
                (
                    carrier,
                    *common,
                    u32(CM1_OPEN_NGXY),
                    u32(flags[2]),
                    u32(flags[3]),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SCHEMA,
        "abi": CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_ABI,
        "status": "ENQUEUED_LIVE_RK1_SCALAR_PRESSURE_BOUNDARY_TILE",
        "completed_nrk": 1,
        "tile": spec.record(),
        "owned_yx": [y0, y1, x0, x1],
        "active_shape": [nk, active_ny, active_nx],
        "open_edges": {
            name: value
            for name, value in zip(
                ("west", "east", "south", "north"), flags, strict=True
            )
        },
        "pressure_density_cadence": {
            "psolver": 3,
            "imoist": 1,
            "idiff": 0,
            "difforder": 6,
            "nrk": 1,
            "nrkmax": 3,
            "recompute_performed": False,
            "apmasscon_performed": False,
        },
        "boundary_order": list(SOURCE_ORDER[2:]),
        "pressure_checkpoint_token_sha256": pressure["operation_token_sha256"],
        "scalar_moist_rk1_token_sha256": scalar["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "momentum_boundary_token_sha256": momentum["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "stream_pointer": int(stream.ptr),
        "same_stream_no_host_sync": True,
        "kernel_source_sha256": (
            CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE_SHA256
        ),
        "target_arch": compiled.target_arch,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledRk1ScalarPressureBoundaryLaunch(
        event, compiled, MappingProxyType(receipt)
    )


def _verify_bcp(
    np: Any, padded: Any, physical: Any, *, nk: int, ny: int, nx: int
) -> None:
    core = padded[1 : nk + 1]
    if not np.array_equal(
        core[:, 1 : ny + 1, 1 : nx + 1].view(np.uint32), physical.view(np.uint32)
    ):
        raise ValueError("completed bcp physical words differ")
    if not np.array_equal(core[:, 1 : ny + 1, 0], physical[:, :, 0]):
        raise ValueError("completed bcp west edge differs")
    if not np.array_equal(core[:, 1 : ny + 1, -1], physical[:, :, -1]):
        raise ValueError("completed bcp east edge differs")
    south = np.pad(physical[:, 0], ((0, 0), (1, 1)), mode="edge")
    north = np.pad(physical[:, -1], ((0, 0), (1, 1)), mode="edge")
    if not np.array_equal(core[:, 0], south) or not np.array_equal(core[:, -1], north):
        raise ValueError("completed bcp south/north edge differs")


def _verify_bcs(
    np: Any,
    padded: Any,
    physical: Any,
    corners: tuple[Any, Any, Any, Any],
    *,
    nk: int,
    ny: int,
    nx: int,
) -> None:
    ng = CM1_OPEN_NGXY
    core = padded[1 : nk + 1] if physical.ndim == 3 else padded[:, 1 : nk + 1]
    actual_physical = core[..., ng : ng + ny, ng : ng + nx]
    if not np.array_equal(actual_physical.view(np.uint32), physical.view(np.uint32)):
        raise ValueError("completed bcs physical words differ")
    west = core[..., ng : ng + ny, :ng]
    east = core[..., ng : ng + ny, ng + nx : ng + nx + ng]
    expected_west = np.repeat(physical[..., :1], ng, axis=-1)
    expected_east = np.repeat(physical[..., -1:], ng, axis=-1)
    if not np.array_equal(west, expected_west) or not np.array_equal(
        east, expected_east
    ):
        raise ValueError("completed bcs west/east edge differs")
    south_line = np.pad(
        physical[..., 0, :], ((0, 0),) * (physical.ndim - 2) + ((1, 1),), mode="edge"
    )
    north_line = np.pad(
        physical[..., -1, :], ((0, 0),) * (physical.ndim - 2) + ((1, 1),), mode="edge"
    )
    expected_south = np.repeat(south_line[..., None, :], ng, axis=-2)
    expected_north = np.repeat(north_line[..., None, :], ng, axis=-2)
    south = core[..., :ng, ng - 1 : ng + nx + 1]
    north = core[..., ng + ny : ng + ny + ng, ng - 1 : ng + nx + 1]
    if not np.array_equal(south, expected_south) or not np.array_equal(
        north, expected_north
    ):
        raise ValueError("completed bcs south/north edge differs")
    sw, nw, se, ne = corners
    checks = (
        (core[..., ng - 1, : ng - 1], sw),
        (core[..., ng + ny, : ng - 1], nw),
        (
            core[..., ng - 1, ng + nx + 1 : ng + nx + ng],
            se,
        ),
        (
            core[..., ng + ny, ng + nx + 1 : ng + nx + ng],
            ne,
        ),
    )
    for actual, source in checks:
        expected = np.repeat(source[..., None], ng - 1, axis=-1)
        if not np.array_equal(actual, expected):
            raise ValueError("completed bcs source-ordered outer corner differs")


@dataclass(frozen=True, slots=True)
class Gm1TiledRk1ScalarPressureBoundaryToken:
    receipt: Mapping[str, object]


def finish_gm1_tiled_rk1_scalar_pressure_boundaries(
    binding: Gm1TiledRk1ScalarPressureBoundaryBinding,
    plan: Gm1TilePlan,
    launches: Sequence[CudaGm1TiledRk1ScalarPressureBoundaryLaunch],
) -> Gm1TiledRk1ScalarPressureBoundaryToken:
    """Admit the exact host publication after all device tiles complete."""

    if not isinstance(binding, Gm1TiledRk1ScalarPressureBoundaryBinding):
        raise TypeError("binding must be the exact RK1 boundary binding")
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (
        binding.nx,
        binding.ny,
    ):
        raise ValueError("boundary completion plan differs from its binding")
    if len(launches) != plan.ntiles:
        raise ValueError("boundary completion requires every exact tile")
    operation_tokens: list[str] = []
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledRk1ScalarPressureBoundaryLaunch):
            raise TypeError("boundary completion requires exact CUDA launch types")
        receipt = launch.receipt
        if (
            launch.ready_event is None
            or receipt.get("schema")
            != CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SCHEMA
            or receipt.get("status")
            != "ENQUEUED_LIVE_RK1_SCALAR_PRESSURE_BOUNDARY_TILE"
            or receipt.get("completed_nrk") != 1
            or receipt.get("tile") != spec.record()
            or receipt.get("pressure_checkpoint_token_sha256")
            != binding.pressure_token_sha256
            or receipt.get("scalar_moist_rk1_token_sha256")
            != binding.scalar_token_sha256
            or receipt.get("post_acoustic_flux_token_sha256")
            != binding.flux_token_sha256
            or receipt.get("momentum_boundary_token_sha256")
            != binding.momentum_boundary_token_sha256
            or receipt.get("stream_pointer") != binding.stream_pointer
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("boundary tile identity, lineage, or cadence changed")
        operation_tokens.append(str(receipt["operation_token_sha256"]))
    np = importlib.import_module("numpy")
    _verify_bcp(
        np,
        binding.padded["rho"],
        binding.physical["rho"],
        nk=binding.nk,
        ny=binding.ny,
        nx=binding.nx,
    )
    _verify_bcp(
        np,
        binding.padded["pp3d"],
        binding.physical["pp3d"],
        nk=binding.nk,
        ny=binding.ny,
        nx=binding.nx,
    )
    for name in ("th3d", "q3d"):
        _verify_bcs(
            np,
            binding.padded[name],
            binding.physical[name],
            binding.scalar_corner_sources[name],
            nk=binding.nk,
            ny=binding.ny,
            nx=binding.nx,
        )
    for name in binding.padded:
        if (
            _vertical_halo_sha(binding.padded[name])
            != binding.vertical_halo_sha256[name]
        ):
            raise ValueError(f"{name} vertical halo was mutated")
    output_hashes = {name: _sha(value) for name, value in binding.padded.items()}
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS,
        "completed_nrk": 1,
        "stage_identity_sha256": binding.stage_identity_sha256,
        "pressure_checkpoint_token_sha256": binding.pressure_token_sha256,
        "scalar_moist_rk1_token_sha256": binding.scalar_token_sha256,
        "post_acoustic_flux_token_sha256": binding.flux_token_sha256,
        "momentum_boundary_token_sha256": binding.momentum_boundary_token_sha256,
        "source_identity_sha256": binding.source_identity_sha256,
        "stream_pointer": binding.stream_pointer,
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(operation_tokens)
        ).hexdigest(),
        "host_scatter_exact_once": True,
        "source_order": list(SOURCE_ORDER),
        "pressure_density_cadence": {
            "selected": {
                "psolver": 3,
                "imoist": 1,
                "idiff": 0,
                "difforder": 6,
                "nrk": 1,
                "nrkmax": 3,
            },
            "recompute_performed": False,
            "post_sound_raw_state_preserved": True,
            "apmasscon_performed": False,
        },
        "output_shapes": {
            name: list(value.shape) for name, value in binding.padded.items()
        },
        "output_sha256": output_hashes,
        "component_key": "scalar_pressure_boundary",
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "missing_components": [
            "live TKE RK1 component aggregation",
            "broader completed-RK-stage emitter",
        ],
        "global_psolver_execution_bound": True,
        "global_psolver_tile_streamable": False,
        "mpi_performed": False,
        "open_history_mutated": False,
        "full_forecast": False,
        "five_meter_meteorological_product": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledRk1ScalarPressureBoundaryToken(MappingProxyType(receipt))


def five_meter_rk1_scalar_pressure_boundary_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    shapes = tiled_rk1_scalar_pressure_boundary_buffer_shapes(
        tile_nx=tile_nx, tile_ny=tile_ny, nk=nk
    )
    one_tile = 4 * sum(math.prod(shape) for shape in shapes.values())
    record: dict[str, object] = {
        "schema": "gm1.hybrid-rk1-scalar-pressure-boundary-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "one_host_tile_bytes": one_tile,
        "one_device_tile_bytes": one_tile,
        "double_host_tile_bytes": 2 * one_tile,
        "double_device_tile_bytes": 2 * one_tile,
        "boundary_operator_tile_streamable": True,
        "global_psolver_tile_streamable": False,
        "completed_broader_rk_stage": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_BC_SHA256",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CM1_NUMQ",
    "CM1_OPEN_NGXY",
    "CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_SCHEMA",
    "CUDA_GM1_RK1_POST_SOUND_PRESSURE_DENSITY_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK1_BOUNDARY_KERNEL_NAMES",
    "CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_ABI",
    "CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE",
    "CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_SOURCE_SHA256",
    "CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS",
    "CompiledGm1TiledRk1ScalarPressureBoundary",
    "CudaGm1TiledRk1ScalarPressureBoundaryLaunch",
    "GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_CLAIM_LIMIT",
    "Gm1Rk1PostSoundPressureDensityToken",
    "Gm1TiledRk1ScalarPressureBoundaryBinding",
    "Gm1TiledRk1ScalarPressureBoundaryToken",
    "checkpoint_gm1_rk1_post_sound_pressure_density",
    "compile_cuda_gm1_tiled_rk1_scalar_pressure_boundary",
    "cpu_gm1_tiled_rk1_scalar_pressure_boundaries_inplace",
    "fill_gm1_tiled_rk1_scalar_pressure_boundary_source",
    "finish_gm1_tiled_rk1_scalar_pressure_boundaries",
    "five_meter_rk1_scalar_pressure_boundary_inventory",
    "launch_cuda_gm1_tiled_rk1_scalar_pressure_boundary",
    "scatter_gm1_tiled_rk1_scalar_pressure_boundary_output",
    "tiled_rk1_scalar_pressure_boundary_buffer_shapes",
)
