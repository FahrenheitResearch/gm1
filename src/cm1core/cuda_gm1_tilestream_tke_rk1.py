"""Exact tile-streamed CM1 r21.1 SGS-TKE ``advw``/RK1 seam.

The source token binds the completed solve1 TKE tendency and the same-stage
post-acoustic ``rru/rrv/rrw/divx`` publication.  A fixed three-cell compute
halo lets the already sealed stretched ``advw`` kernel run unchanged on each
tile; only each tile's unique owned rectangle is admitted at completion.

This module advances RK1 only.  RK2 requires a newly evolved acoustic stage and
a new post-sound ``getdivx`` token.  The global psolver is not tile-streamed or
claimed here.
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

from .cuda_gm1_child_stretched_sgs_tke import (
    CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256,
    CudaChildStretchedLaunch,
    CompiledChildStretchedSgsTke,
    launch_cuda_gm1_child_stretched_tke_rk_stage,
)
from .cuda_gm1_tilestream import Gm1TilePlan, Gm1TileSpec
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
    Gm1TiledSolve2PostAcousticFluxToken,
    require_gm1_tiled_solve2_tke_rk1,
)
from .cuda_tke_evolution import tke_evolution_shapes


CUDA_GM1_TILESTREAM_TKE_RK1_SCHEMA = "gm1.cuda-tilestream-tke-rk1/v1"
CUDA_GM1_TILESTREAM_TKE_RK1_SOURCE_TOKEN_SCHEMA = (
    "gm1.tilestream-tke-rk1-tile-source-token/v1"
)
CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA = "gm1.tilestream-tke-rk1-token/v1"
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_ADV_SHA256 = (
    "e9985af10bdde4bfd83d34c43b641af9d073c948c472d9d5ea51ccb5f0a7a66f"
)
AUTHORITY_ADV_ROUTINES_SHA256 = (
    "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00"
)
AUTHORITY_ANCHORS = (
    "src/solve2.F:1708-1803",
    "src/adv.F:2778-3610",
    "src/adv_routines.F:204-430",
    "src/adv_routines.F:2359-2557",
    "src/bc.F:942-1143",
)
GM1_TILESTREAM_TKE_RK1_CLAIM_LIMIT = (
    "one CM1 r21.1 serial terrain/stretch_z=1 SGS-TKE advw/RK1/clip/bcw "
    "operator over exact fixed-halo tiles, consuming the live solve1 TKE RHS "
    "and same-stage post-acoustic getdivx token with exact owned-rectangle "
    "scatter. RK2/RK3, solve3 commit, global psolver tilestreaming, MPI, open "
    "history production, full RK/forecast, storm, tornado, and the five-metre "
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


def _require_tokens(
    solve1_token: object, flux_token: object
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    require_gm1_tiled_solve2_tke_rk1(solve1_token, flux_token)
    if not isinstance(solve1_token, Gm1TiledSolve1TkeRhsToken):
        raise TypeError("solve1 token type changed")
    if not isinstance(flux_token, Gm1TiledSolve2PostAcousticFluxToken):
        raise TypeError("post-acoustic flux token type changed")
    solve1, flux = solve1_token.receipt, flux_token.receipt
    if (
        solve1.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA
        or solve1.get("status") != "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS"
        or solve1.get("completed_nrk") != 0
        or solve1.get("operation_token_sha256") != _operation_token(solve1)
        or flux.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or flux.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or flux.get("nrk") != 1
        or flux.get("solve1_tke_rhs_token_sha256")
        != solve1.get("operation_token_sha256")
        or flux.get("operation_token_sha256") != _operation_token(flux)
    ):
        raise ValueError("solve1 or post-acoustic RK1 token identity changed")
    return solve1, flux


def _inventory_sha(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "little"))
        digest.update(encoded)
        digest.update(arrays[name].astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Gm1TiledTkeRk1TileSourceToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1TiledTkeRk1Launch:
    ready_event: Any
    tke3d: Any
    workspace: Any
    upstream: CudaChildStretchedLaunch
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledTkeRk1Token:
    receipt: Mapping[str, object]


def bind_gm1_tiled_tke_rk1_tile_source(
    solve1_token: Gm1TiledSolve1TkeRhsToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    device_h2d_inventory_sha256: str | None = None,
) -> Gm1TiledTkeRk1TileSourceToken:
    """Hash-bind one exact host tile before H2D transport."""

    solve1, flux = _require_tokens(solve1_token, flux_token)
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    np = importlib.import_module("numpy")
    shapes = tke_evolution_shapes(spec.cnx, spec.cny, 75)
    if not isinstance(arrays, Mapping) or set(arrays) != set(shapes):
        raise ValueError("TKE RK1 host tile inventory changed")
    ranges: list[tuple[int, int, str]] = []
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"host TKE RK1 {name} must be finite C-<f4 {shape}")
        start = int(value.ctypes.data)
        ranges.append((start, start + int(value.nbytes), name))
    for index, (start, stop, name) in enumerate(ranges):
        for other_start, other_stop, other_name in ranges[index + 1 :]:
            if start < other_stop and other_start < stop:
                raise ValueError(f"host TKE RK1 arrays alias: {name}/{other_name}")
    host_sha = _inventory_sha(arrays)
    if device_h2d_inventory_sha256 is not None:
        if (
            not isinstance(device_h2d_inventory_sha256, str)
            or len(device_h2d_inventory_sha256) != 64
            or any(c not in "0123456789abcdef" for c in device_h2d_inventory_sha256)
            or device_h2d_inventory_sha256 != host_sha
        ):
            raise ValueError("TKE RK1 H2D inventory differs from its host bytes")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_RK1_SOURCE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILE_TKE_RK1_SOURCE",
        "nrk": 1,
        "tile": spec.record(),
        "solve1_tke_rhs_token_sha256": solve1["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "host_inventory_sha256": host_sha,
        "device_h2d_inventory_sha256": device_h2d_inventory_sha256,
        "h2d_raw_bit_verified": device_h2d_inventory_sha256 is not None,
        "compute_shape_zyx": [75, spec.cny, spec.cnx],
        "source_halo": spec.halo,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "adv_sha256": AUTHORITY_ADV_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledTkeRk1TileSourceToken(MappingProxyType(receipt))


def _require_source_token(
    token: object,
    solve1: Mapping[str, object],
    flux: Mapping[str, object],
    spec: Gm1TileSpec,
) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledTkeRk1TileSourceToken):
        raise TypeError("source_token must be the exact typed RK1 tile source")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK1_SOURCE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILE_TKE_RK1_SOURCE"
        or receipt.get("nrk") != 1
        or receipt.get("tile") != spec.record()
        or receipt.get("solve1_tke_rhs_token_sha256")
        != solve1.get("operation_token_sha256")
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256") != flux.get("stage_identity_sha256")
        or receipt.get("h2d_raw_bit_verified") is not True
        or receipt.get("host_inventory_sha256")
        != receipt.get("device_h2d_inventory_sha256")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("TKE RK1 tile source identity or transport changed")
    return receipt


def launch_cuda_gm1_tiled_tke_rk1(
    solve1_token: Gm1TiledSolve1TkeRhsToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    source_token: Gm1TiledTkeRk1TileSourceToken,
    arrays: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
    metrics: Mapping[str, Any],
    stream: Any,
    compiled: CompiledChildStretchedSgsTke,
    workspace: Any | None = None,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1TiledTkeRk1Launch:
    """Enqueue exact stretched ``advw``/RK1 for one fixed-halo tile."""

    solve1, flux = _require_tokens(solve1_token, flux_token)
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    source = _require_source_token(source_token, solve1, flux, spec)
    if stream is None or not hasattr(stream, "ptr"):
        raise TypeError("stream must be an explicit CUDA stream")
    if int(stream.ptr) != solve1.get("stream_pointer") or int(stream.ptr) != flux.get(
        "stream_pointer"
    ):
        raise ValueError("TKE RK1 left the solve1/post-acoustic CUDA stream")
    cupy = importlib.import_module("cupy")
    shapes = tke_evolution_shapes(spec.cnx, spec.cny, 75)
    if not isinstance(arrays, Mapping) or set(arrays) != set(shapes):
        raise ValueError("TKE RK1 device tile inventory changed")
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"device TKE RK1 {name} must be C-f32 {shape}")
    if workspace is None:
        workspace = cupy.empty_like(arrays["tke3d"])
    upstream = launch_cuda_gm1_child_stretched_tke_rk_stage(
        arrays,
        workspace=workspace,
        metrics=metrics,
        ni=spec.cnx,
        nj=spec.cny,
        nk=75,
        nrk=1,
        dx=50.0,
        dy=50.0,
        dz=200.0 if allow_uniform_idealized_validation else 100.0,
        ztop=15000.0,
        dt=0.5,
        stream=stream,
        compiled=compiled,
        allow_compact_validation=True,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_RK1_SCHEMA,
        "status": "ENQUEUED_LIVE_TILE_TKE_ADVW_RK1",
        "nrk": 1,
        "tile": spec.record(),
        "source_token_sha256": source["operation_token_sha256"],
        "solve1_tke_rhs_token_sha256": solve1["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "stream_pointer": int(stream.ptr),
        "same_stream_after_post_acoustic_getdivx": True,
        "source_halo": spec.halo,
        "scatter_disposition": "owned mass rectangle only after event completion",
        "kernel_source_sha256": CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "upstream_receipt": dict(upstream.receipt),
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "adv_sha256": AUTHORITY_ADV_SHA256,
            "adv_routines_sha256": AUTHORITY_ADV_ROUTINES_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledTkeRk1Launch(
        upstream.ready_event,
        arrays["tke3d"],
        workspace,
        upstream,
        MappingProxyType(receipt),
    )


def finish_cuda_gm1_tiled_tke_rk1(
    solve1_token: Gm1TiledSolve1TkeRhsToken,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    launches: Sequence[CudaGm1TiledTkeRk1Launch],
    *,
    plan: Gm1TilePlan,
    output_tke3d: Any,
) -> Gm1TiledTkeRk1Token:
    """Admit exact-once RK1 scatter and bind the completed current TKE."""

    solve1, flux = _require_tokens(solve1_token, flux_token)
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("TKE RK1 completion requires the admitted 300 grid")
    if len(launches) != plan.ntiles:
        raise ValueError("TKE RK1 completion requires every exact tile")
    streams: set[int] = set()
    launch_tokens: list[str] = []
    source_tokens: list[str] = []
    uniform_idealized_dispositions: set[bool] = set()
    owned_words = 0
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledTkeRk1Launch):
            raise TypeError("TKE RK1 completion requires typed launches")
        receipt = launch.receipt
        y0, y1, x0, x1 = spec.owned_rect("mass")
        owned_words += 76 * (y1 - y0) * (x1 - x0)
        if (
            receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK1_SCHEMA
            or receipt.get("status") != "ENQUEUED_LIVE_TILE_TKE_ADVW_RK1"
            or receipt.get("nrk") != 1
            or receipt.get("tile") != spec.record()
            or receipt.get("solve1_tke_rhs_token_sha256")
            != solve1.get("operation_token_sha256")
            or receipt.get("post_acoustic_flux_token_sha256")
            != flux.get("operation_token_sha256")
            or receipt.get("stage_identity_sha256")
            != flux.get("stage_identity_sha256")
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("TKE RK1 tile identity or cadence changed")
        streams.add(int(receipt["stream_pointer"]))
        source_tokens.append(str(receipt["source_token_sha256"]))
        launch_tokens.append(str(receipt["operation_token_sha256"]))
        uniform_idealized_dispositions.add(
            receipt.get("uniform_idealized_validation") is True
        )
    expected_words = 76 * 300 * 300
    if owned_words != expected_words:
        raise ValueError("TKE RK1 scatter ownership is not exact once")
    if (
        len(streams) != 1
        or next(iter(streams)) != solve1.get("stream_pointer")
        or next(iter(streams)) != flux.get("stream_pointer")
    ):
        raise ValueError("TKE RK1 tile streams differ from prerequisite tokens")
    if len(uniform_idealized_dispositions) != 1:
        raise ValueError("TKE RK1 tiles disagree on idealized disposition")
    np = importlib.import_module("numpy")
    if (
        not isinstance(output_tke3d, np.ndarray)
        or output_tke3d.dtype != np.dtype("<f4")
        or output_tke3d.shape != (76, 300, 300)
        or not output_tke3d.flags.c_contiguous
        or not bool(np.all(np.isfinite(output_tke3d)))
    ):
        raise ValueError("output_tke3d must be finite host C-<f4 (76,300,300)")
    if np.count_nonzero(output_tke3d[[0, -1]].view(np.uint32)):
        raise ValueError("TKE RK1 did not preserve rigid zero bottom/top faces")
    output_sha = hashlib.sha256(output_tke3d.tobytes(order="C")).hexdigest()
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILESTREAM_TKE_ADVW_RK1",
        "completed_nrk": 1,
        "solve1_tke_rhs_token_sha256": solve1["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "tile_plan_sha256": plan.ownership_sha256,
        "tile_count": plan.ntiles,
        "tile_source_tokens_sha256": hashlib.sha256(
            _canonical(source_tokens)
        ).hexdigest(),
        "tile_operation_tokens_sha256": hashlib.sha256(
            _canonical(launch_tokens)
        ).hexdigest(),
        "stream_pointer": next(iter(streams)),
        "host_scatter_exact_once": True,
        "uniform_idealized_validation": next(
            iter(uniform_idealized_dispositions)
        ),
        "output_tke3d_shape": [76, 300, 300],
        "output_tke3d_words": expected_words,
        "output_tke3d_sha256": output_sha,
        "final_bcw_applied": True,
        "ready_for": (
            "RK2 only after exact evolved nrk=2 acoustic sound -> post-sound "
            "getdivx token"
        ),
        "structural_refusals": [
            "reuse of the RK1 flux token at RK2",
            "RK3 or solve3 commit",
            "global psolver tilestream claim",
            "MPI exchange",
            "full RK or forecast",
            "five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "adv_sha256": AUTHORITY_ADV_SHA256,
            "adv_routines_sha256": AUTHORITY_ADV_ROUTINES_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_RK1_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledTkeRk1Token(MappingProxyType(receipt))


def require_gm1_tiled_tke_rk2(
    rk1_token: object, nrk2_flux_token: object
) -> None:
    if not isinstance(rk1_token, Gm1TiledTkeRk1Token):
        raise TypeError("rk1_token must be the exact completed TKE RK1 token")
    receipt = rk1_token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_TKE_ADVW_RK1"
        or receipt.get("completed_nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("completed TKE RK1 token identity changed")
    if not isinstance(nrk2_flux_token, Gm1TiledSolve2PostAcousticFluxToken):
        raise RuntimeError(
            "GM1-TILESTREAM-TKE-RK2-PREREQUISITE: exact evolved nrk=2 "
            "sound -> post-sound getdivx token is not available"
        )
    flux = nrk2_flux_token.receipt
    if (
        flux.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or flux.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or flux.get("nrk") != 2
        or flux.get("stream_pointer") != receipt.get("stream_pointer")
        or flux.get("operation_token_sha256") != _operation_token(flux)
    ):
        raise ValueError("nrk=2 post-acoustic flux token cadence changed")


def five_meter_tiled_tke_rk1_inventory(
    *, tile_nx: int = 192, tile_ny: int = 192, nk: int = 75
) -> Mapping[str, object]:
    for name, value in (("tile_nx", tile_nx), ("tile_ny", tile_ny), ("nk", nk)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    cnx, cny = tile_nx + 6, tile_ny + 6
    shapes = tke_evolution_shapes(cnx, cny, nk)
    tile_bytes = 4 * sum(math.prod(shape) for shape in shapes.values())
    workspace_bytes = 4 * math.prod(shapes["tke3d"])
    record = {
        "schema": "gm1.hybrid-tilestream-tke-rk1-memory/v1",
        "domain": [nk, 3000, 3000],
        "tile": [tile_ny, tile_nx],
        "tile_count": math.ceil(3000 / tile_nx) * math.ceil(3000 / tile_ny),
        "tile_device_inventory_bytes_each": tile_bytes,
        "workspace_device_bytes_each": workspace_bytes,
        "double_buffer_device_bytes": 2 * (tile_bytes + workspace_bytes),
        "tke_advw_rk1_tile_streamable": True,
        "global_psolver_tile_streamable": False,
        "rk1_post_acoustic_flux_available_at_5m": False,
        "rk2_available": False,
        "five_meter_meteorological_product": False,
    }
    record["inventory_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
    return MappingProxyType(record)


__all__ = (
    "AUTHORITY_ADV_ROUTINES_SHA256",
    "AUTHORITY_ADV_SHA256",
    "AUTHORITY_ANCHORS",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_TKE_RK1_SCHEMA",
    "CUDA_GM1_TILESTREAM_TKE_RK1_SOURCE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_TKE_RK1_TOKEN_SCHEMA",
    "CudaGm1TiledTkeRk1Launch",
    "GM1_TILESTREAM_TKE_RK1_CLAIM_LIMIT",
    "Gm1TiledTkeRk1TileSourceToken",
    "Gm1TiledTkeRk1Token",
    "bind_gm1_tiled_tke_rk1_tile_source",
    "finish_cuda_gm1_tiled_tke_rk1",
    "five_meter_tiled_tke_rk1_inventory",
    "launch_cuda_gm1_tiled_tke_rk1",
    "require_gm1_tiled_tke_rk2",
)
