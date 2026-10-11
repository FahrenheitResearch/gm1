"""Typed CM1 solve2 ``sound -> getdivx`` tile-stream prerequisite.

CM1 r21.1 uses ``rru/rrv/rrw`` twice during one compressible RK stage.  The
pre-acoustic copy is built near the start of ``solve2``.  ``sound`` then
overwrites those carriers with small-step-averaged velocities, and the later
``getdivx`` call converts the averaged velocities in place to terrain-aware
mass fluxes while rebuilding ``divx``.  TKE ``advw`` consumes only that latter,
coherent post-acoustic set.

This module types the narrow boundary between the already sealed full-domain
psolver=3 operator and the tile-local ``getdivx`` kernels.  It deliberately
does not tile, replace, or claim the global acoustic solve.  Consequently the
3000x3000 five-metre product remains fail-closed even though the downstream
``getdivx`` work itself is tile-streamable.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_child_psolver3_open import (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA,
    CudaGm1ChildPsolver3OpenLaunch,
    launch_cuda_gm1_child_psolver3_open_stage,
)
from .cuda_gm1_tilestream import (
    CudaGm1TileMassFluxLaunch,
    Gm1TilePlan,
    Gm1TileSpec,
    launch_cuda_gm1_tile_local_mass_flux,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
    Gm1TiledSolve2PostAcousticFluxToken,
)


CUDA_GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_SCHEMA = (
    "gm1.cuda-tilestream-post-acoustic-getdivx/v1"
)
CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA = (
    "gm1.post-sound-averaged-velocity-token/v1"
)
CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA = (
    "gm1.tilestream-post-acoustic-mass-flux-token/v1"
)
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_SOUND_SHA256 = (
    "5fe055e22d106e91b5bdbe35ad7c72b2bbbcfb5b01733126703fdc7a5977da54"
)
AUTHORITY_MISCLIBS_SHA256 = (
    "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1082-1137",
    "src/sound.F:190-250",
    "src/sound.F:806-832",
    "src/solve2.F:1303-1307",
    "src/misclibs.F:222-370",
    "src/solve2.F:1708-1803",
)
GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_CLAIM_LIMIT = (
    "one real 300x300x75 nrk=1 psolver=3 post-sound averaged-velocity snapshot "
    "followed by exact tile-local terrain/stretch_z=1 getdivx and exact-once "
    "host scatter; the full-domain acoustic operator remains global and is not "
    "a five-metre execution path. pre-acoustic momentum, complete solve2, TKE "
    "advw/RK, RK2/RK3, MPI, forecast, storm, tornado, and the five-metre "
    "meteorological product remain refused"
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
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be lowercase 64-hex sha256")
    return value


def _require_solve1_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledSolve1TkeRhsToken):
        raise TypeError("solve1_token must be the exact live TKE RHS token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS"
        or receipt.get("completed_nrk") != 0
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("solve1 TKE RHS token identity or cadence changed")
    return receipt


def _require_psolver_receipt(
    launch: object, *, nrk: int, ni: int, nj: int, nk: int
) -> Mapping[str, object]:
    if not isinstance(launch, CudaGm1ChildPsolver3OpenLaunch):
        raise TypeError("psolver launch must be the exact typed child operator")
    receipt = launch.receipt
    cadence = receipt.get("cadence")
    domain = receipt.get("domain")
    kernels = receipt.get("kernels")
    if (
        receipt.get("schema") != CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA
        or not isinstance(cadence, list)
        or len(cadence) != 1
        or cadence[0].get("nrk") != nrk
        or not isinstance(domain, Mapping)
        or (domain.get("ni"), domain.get("nj"), domain.get("nk"))
        != (ni, nj, nk)
        or domain.get("compact_validation") is not False
        or not isinstance(kernels, Mapping)
        or kernels.get("core_source_sha256")
        != CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256
    ):
        raise ValueError("psolver stage identity, cadence, or source changed")
    return receipt


@dataclass(frozen=True, slots=True)
class CudaGm1PostSoundVelocityLaunch:
    ready_event: Any
    upstream: CudaGm1ChildPsolver3OpenLaunch
    arrays: Mapping[str, Any]
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1PostSoundVelocityToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1TiledPostAcousticGetdivxLaunch:
    ready_event: Any
    mass_flux: CudaGm1TileMassFluxLaunch
    receipt: Mapping[str, object]


def launch_cuda_gm1_rk1_post_sound_velocity(
    solve1_token: Gm1TiledSolve1TkeRhsToken,
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    vertical: Any,
    stage_identity_sha256: str,
    stream: Any,
    compiled: Any | None = None,
    open_compiled: Any | None = None,
    lateral_compiled: Any | None = None,
) -> CudaGm1PostSoundVelocityLaunch:
    """Enqueue the sealed real RK1 acoustic operator after the solve1 token.

    This is an explicitly global 300x300x75 operator boundary.  No tiled
    acoustic substitution is attempted or implied.
    """

    solve1 = _require_solve1_token(solve1_token)
    stage_identity_sha256 = _lower_sha(
        "stage_identity_sha256", stage_identity_sha256
    )
    if stream is None or not hasattr(stream, "ptr"):
        raise TypeError("stream must be an explicit CUDA stream")
    stream_pointer = int(stream.ptr)
    if solve1.get("stream_pointer") != stream_pointer:
        raise ValueError("acoustic operator left the solve1 CUDA stream")
    upstream = launch_cuda_gm1_child_psolver3_open_stage(
        arrays,
        metrics,
        ni=300,
        nj=300,
        nk=75,
        nrk=1,
        dt=0.5,
        nsound=6,
        dx=50.0,
        dy=50.0,
        vertical=vertical,
        stream=stream,
        compiled=compiled,
        open_compiled=open_compiled,
        lateral_compiled=lateral_compiled,
    )
    psolver = _require_psolver_receipt(
        upstream, nrk=1, ni=300, nj=300, nk=75
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_SCHEMA,
        "status": "ENQUEUED_GLOBAL_RK1_POST_SOUND_AVERAGED_VELOCITY",
        "nrk": 1,
        "stage_identity_sha256": stage_identity_sha256,
        "solve1_tke_rhs_token_sha256": solve1["operation_token_sha256"],
        "psolver_receipt_sha256": hashlib.sha256(_canonical(psolver)).hexdigest(),
        "psolver_source_sha256": CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
        "stream_pointer": stream_pointer,
        "same_stream_after_solve1_tke_rhs": True,
        "host_sync_performed": False,
        "rru_rrv_rrw_disposition": "pending small-step-averaged velocities",
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "sound_sha256": AUTHORITY_SOUND_SHA256,
            "anchors": list(AUTHORITY_ANCHORS[:3]),
        },
        "claim_limit": GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1PostSoundVelocityLaunch(
        upstream.ready_event,
        upstream,
        arrays,
        MappingProxyType(receipt),
    )


def finish_cuda_gm1_rk1_post_sound_velocity(
    launch: CudaGm1PostSoundVelocityLaunch,
) -> tuple[Gm1PostSoundVelocityToken, Mapping[str, Any]]:
    """Synchronize once and bind the physical averaged-velocity snapshot."""

    if not isinstance(launch, CudaGm1PostSoundVelocityLaunch):
        raise TypeError("launch must be the exact typed post-sound launch")
    pending = launch.receipt
    if (
        pending.get("schema") != CUDA_GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_SCHEMA
        or pending.get("status")
        != "ENQUEUED_GLOBAL_RK1_POST_SOUND_AVERAGED_VELOCITY"
        or pending.get("nrk") != 1
        or pending.get("operation_token_sha256") != _operation_token(pending)
    ):
        raise ValueError("post-sound launch identity or cadence changed")
    _require_psolver_receipt(launch.upstream, nrk=1, ni=300, nj=300, nk=75)
    cupy = importlib.import_module("cupy")
    np = importlib.import_module("numpy")
    launch.ready_event.synchronize()
    source_slices = {
        "u3d": ("rru", (slice(1, 76), slice(3, 303), slice(3, 304))),
        "v3d": ("rrv", (slice(1, 76), slice(3, 304), slice(3, 303))),
        "w3d": ("rrw", (slice(1, 77), slice(3, 303), slice(3, 303))),
    }
    host: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    expected = {
        "u3d": (75, 300, 301),
        "v3d": (75, 301, 300),
        "w3d": (76, 300, 300),
    }
    for target, (source, slices) in source_slices.items():
        value = np.ascontiguousarray(cupy.asnumpy(launch.arrays[source][slices]))
        if (
            value.dtype != np.dtype("<f4")
            or value.shape != expected[target]
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"post-sound {target} snapshot is invalid")
        host[target] = value
        hashes[target] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
        "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
        "nrk": 1,
        "stage_identity_sha256": pending["stage_identity_sha256"],
        "solve1_tke_rhs_token_sha256": pending[
            "solve1_tke_rhs_token_sha256"
        ],
        "psolver_receipt_sha256": pending["psolver_receipt_sha256"],
        "stream_pointer": pending["stream_pointer"],
        "host_checkpoint_transport": True,
        "velocity_shapes": {name: list(shape) for name, shape in expected.items()},
        "velocity_sha256": hashes,
        "sound_output_semantics": "small-step-averaged velocities before getdivx",
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "sound_sha256": AUTHORITY_SOUND_SHA256,
            "anchors": list(AUTHORITY_ANCHORS[:3]),
        },
        "claim_limit": GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return (
        Gm1PostSoundVelocityToken(MappingProxyType(receipt)),
        MappingProxyType(host),
    )


def _require_velocity_token(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1PostSoundVelocityToken):
        raise TypeError("velocity_token must be the exact post-sound token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT"
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-sound averaged-velocity token changed")
    return receipt


def launch_cuda_gm1_tiled_post_acoustic_getdivx(
    velocity_token: Gm1PostSoundVelocityToken,
    arrays: Mapping[str, Any],
    vertical: Any,
    *,
    spec: Gm1TileSpec,
    fill_receipt: Mapping[str, object],
    stream: Any,
    compiled: Any | None = None,
) -> CudaGm1TiledPostAcousticGetdivxLaunch:
    """Convert one staged averaged-velocity tile to post-acoustic fluxes."""

    velocity = _require_velocity_token(velocity_token)
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    if stream is None or not hasattr(stream, "ptr"):
        raise TypeError("stream must be an explicit CUDA stream")
    if int(stream.ptr) != velocity.get("stream_pointer"):
        raise ValueError("post-acoustic getdivx left the acoustic CUDA stream")
    if (
        not isinstance(fill_receipt, Mapping)
        or fill_receipt.get("source_identity_sha256")
        != velocity.get("operation_token_sha256")
        or fill_receipt.get("tile") != spec.index
        or fill_receipt.get("compute_mass_yx")
        != [spec.cj0, spec.cj0 + spec.cny, spec.ci0, spec.ci0 + spec.cnx]
        or fill_receipt.get("full_domain_transient_stage_store_used") is not False
    ):
        raise ValueError("tile source is not the admitted post-sound snapshot")
    mass_flux = launch_cuda_gm1_tile_local_mass_flux(
        arrays,
        vertical,
        ni=spec.cnx,
        nj=spec.cny,
        nk=75,
        dx=50.0,
        dy=50.0,
        source_identity_sha256=str(velocity["operation_token_sha256"]),
        tile_index=spec.index,
        stream=stream,
        compiled=compiled,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_SCHEMA,
        "status": "ENQUEUED_LIVE_RK1_TILE_POST_ACOUSTIC_GETDIVX",
        "nrk": 1,
        "tile": spec.record(),
        "post_sound_velocity_token_sha256": velocity[
            "operation_token_sha256"
        ],
        "solve1_tke_rhs_token_sha256": velocity[
            "solve1_tke_rhs_token_sha256"
        ],
        "stage_identity_sha256": velocity["stage_identity_sha256"],
        "mass_flux_receipt_sha256": mass_flux.receipt["receipt_sha256"],
        "stream_pointer": int(stream.ptr),
        "same_stream_after_global_acoustic_checkpoint": True,
        "source_order": [
            "sound returns small-step-averaged rru/rrv/rrw velocities",
            "getdivx density/terrain weights rru/rrv/rrw in place",
            "getdivx rebuilds stretched divx from those same mass fluxes",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "misclibs_sha256": AUTHORITY_MISCLIBS_SHA256,
            "anchors": list(AUTHORITY_ANCHORS[3:5]),
        },
        "claim_limit": GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledPostAcousticGetdivxLaunch(
        mass_flux.ready_event,
        mass_flux,
        MappingProxyType(receipt),
    )


_OUTPUT_SHAPES = MappingProxyType(
    {
        "rru": (75, 300, 301),
        "rrv": (75, 301, 300),
        "rrw": (76, 300, 300),
        "divx": (75, 300, 300),
    }
)


def finish_cuda_gm1_tiled_post_acoustic_getdivx(
    solve1_token: Gm1TiledSolve1TkeRhsToken,
    velocity_token: Gm1PostSoundVelocityToken,
    launches: Sequence[CudaGm1TiledPostAcousticGetdivxLaunch],
    *,
    plan: Gm1TilePlan,
    outputs: Mapping[str, Any],
) -> Gm1TiledSolve2PostAcousticFluxToken:
    """Bind exact-once scattered RK1 post-acoustic mass flux/divergence."""

    solve1 = _require_solve1_token(solve1_token)
    velocity = _require_velocity_token(velocity_token)
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("post-acoustic getdivx requires the admitted 300 grid")
    if velocity.get("solve1_tke_rhs_token_sha256") != solve1.get(
        "operation_token_sha256"
    ):
        raise ValueError("post-sound velocity belongs to another solve1 token")
    if len(launches) != plan.ntiles:
        raise ValueError("post-acoustic getdivx requires every exact tile")
    tile_tokens: list[str] = []
    streams: set[int] = set()
    owned_words = 0
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledPostAcousticGetdivxLaunch):
            raise TypeError("completion requires typed post-acoustic tile launches")
        receipt = launch.receipt
        tile_words = 0
        for variant, levels in (("u", 75), ("v", 75), ("w", 76), ("mass", 75)):
            y0, y1, x0, x1 = spec.owned_rect(variant)
            tile_words += levels * (y1 - y0) * (x1 - x0)
        if (
            receipt.get("schema")
            != CUDA_GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_SCHEMA
            or receipt.get("status")
            != "ENQUEUED_LIVE_RK1_TILE_POST_ACOUSTIC_GETDIVX"
            or receipt.get("nrk") != 1
            or receipt.get("tile") != spec.record()
            or receipt.get("post_sound_velocity_token_sha256")
            != velocity.get("operation_token_sha256")
            or receipt.get("solve1_tke_rhs_token_sha256")
            != solve1.get("operation_token_sha256")
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("post-acoustic tile identity or source order changed")
        tile_tokens.append(str(receipt["operation_token_sha256"]))
        streams.add(int(receipt["stream_pointer"]))
        owned_words += tile_words
    expected_words = sum(math.prod(shape) for shape in _OUTPUT_SHAPES.values())
    if owned_words != expected_words:
        raise ValueError("post-acoustic scatter ownership is not exact once")
    if (
        len(streams) != 1
        or next(iter(streams)) != solve1.get("stream_pointer")
        or next(iter(streams)) != velocity.get("stream_pointer")
    ):
        raise ValueError("solve1, acoustic, and getdivx streams differ")
    np = importlib.import_module("numpy")
    if not isinstance(outputs, Mapping) or set(outputs) != set(_OUTPUT_SHAPES):
        raise ValueError("post-acoustic output inventory changed")
    hashes: dict[str, str] = {}
    for name, shape in _OUTPUT_SHAPES.items():
        value = outputs[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"{name} must be finite host C-<f4 {shape}")
        hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
        "nrk": 1,
        "stage_identity_sha256": velocity["stage_identity_sha256"],
        "solve1_tke_rhs_token_sha256": solve1["operation_token_sha256"],
        "post_sound_velocity_token_sha256": velocity[
            "operation_token_sha256"
        ],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(tile_tokens)
        ).hexdigest(),
        "stream_pointer": next(iter(streams)),
        "host_scatter_exact_once": True,
        "output_shapes": {name: list(shape) for name, shape in _OUTPUT_SHAPES.items()},
        "output_sha256": hashes,
        "output_words": expected_words,
        "source_order": [
            "global psolver=3 sound returns small-step-averaged velocities",
            "tile-local getdivx converts the averaged velocities to mass flux",
            "the same getdivx invocation rebuilds divx",
        ],
        "ready_for": "typed TKE advw/RK1 operator only",
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "structural_refusals": [
            "pre-acoustic or frozen flux substitution",
            "RK2/RK3 without a new same-stage sound -> getdivx token",
            "global psolver tilestream claim",
            "MPI exchange",
            "full RK or forecast",
            "five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "sound_sha256": AUTHORITY_SOUND_SHA256,
            "misclibs_sha256": AUTHORITY_MISCLIBS_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledSolve2PostAcousticFluxToken(MappingProxyType(receipt))


def five_meter_post_acoustic_getdivx_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    for name, value in (("tile_nx", tile_nx), ("tile_ny", tile_ny), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    cnx, cny, ng = tile_nx + 6, tile_ny + 6, 3
    shapes = {
        "rru": (nk + 2, cny + 2 * ng, cnx + 1 + 2 * ng),
        "rrv": (nk + 2, cny + 1 + 2 * ng, cnx + 2 * ng),
        "rrw": (nk + 3, cny + 2 * ng, cnx + 2 * ng),
        "divx": (nk, cny, cnx),
        "rho0": (nk + 2, cny + 2 * ng, cnx + 2 * ng),
        "rf0": (nk + 1, cny, cnx),
        "rgzu": (cny, cnx + 1),
        "rgzv": (cny + 1, cnx),
        "dzdx": (cny, cnx),
        "dzdy": (cny, cnx),
        "gz": (cny, cnx),
    }
    tile_bytes = 4 * sum(math.prod(shape) for shape in shapes.values())
    record = {
        "schema": "gm1.hybrid-post-acoustic-getdivx-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "getdivx_tile_device_bytes_each": tile_bytes,
        "getdivx_double_buffer_device_bytes": 2 * tile_bytes,
        "getdivx_tile_streamable": True,
        "global_psolver_tile_streamable": False,
        "post_sound_velocity_token_available_at_5m": False,
        "tke_rk1_available_at_5m": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ANCHORS",
    "AUTHORITY_MISCLIBS_SHA256",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_SOUND_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA",
    "CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_SCHEMA",
    "CudaGm1PostSoundVelocityLaunch",
    "CudaGm1TiledPostAcousticGetdivxLaunch",
    "GM1_TILESTREAM_POST_ACOUSTIC_GETDIVX_CLAIM_LIMIT",
    "Gm1PostSoundVelocityToken",
    "finish_cuda_gm1_rk1_post_sound_velocity",
    "finish_cuda_gm1_tiled_post_acoustic_getdivx",
    "five_meter_post_acoustic_getdivx_inventory",
    "launch_cuda_gm1_rk1_post_sound_velocity",
    "launch_cuda_gm1_tiled_post_acoustic_getdivx",
)
