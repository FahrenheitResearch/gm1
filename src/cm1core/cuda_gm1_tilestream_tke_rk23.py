"""Exact tile-streamed CM1 r21.1 SGS-TKE RK2/RK3 seam.

The TKE tendency and ``tkea`` baseline are held across all three RK stages,
but the current ``tke3d`` and post-acoustic ``rru/rrv/rrw/divx`` belong to a
specific stage.  This module therefore requires three independent typed
lineages before it will enqueue a tile: the held TKE state, the complete
broader prior RK stage, and a fresh next-stage sound/getdivx publication.

The broader completed-stage token is owned by the whole-model source-order
aggregator.  This seam imports its frozen validator-only type, but no producer
exists here; no local substitute or duck-typed map is accepted.
"""

from __future__ import annotations

import hashlib
import importlib
import json
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
from .cuda_gm1_tilestream_completed_rk_stage import (
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
    GM1_COMPLETED_RK_STAGE_TKE_COMPONENT_KEY,
    Gm1TiledCompletedRkStageToken,
    require_gm1_tiled_completed_rk_stage_token,
)
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
)
from .cuda_gm1_rk1_source_ordered_tke import (
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA,
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS,
    Gm1Rk1SourceOrderedTkeToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from .cuda_gm1_tilestream_tke_stage_state import (
    AUTHORITY_ANCHORS,
    AUTHORITY_SOLVE2_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA,
    Gm1TiledTkeHeldStageToken,
)
from .cuda_tke_evolution import tke_evolution_shapes


CUDA_GM1_TILESTREAM_TKE_RK23_SOURCE_TOKEN_SCHEMA = (
    "gm1.tilestream-tke-rk23-tile-source-token/v1"
)
CUDA_GM1_TILESTREAM_TKE_RK23_SCHEMA = "gm1.cuda-tilestream-tke-rk23/v1"
CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA = (
    "gm1.tilestream-tke-rk23-stage-token/v1"
)
COMPLETED_RK_STAGE_STATUS = (
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS
)
GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT = (
    "one CM1 r21.1 serial terrain/stretch_z=1 SGS-TKE advw/RK2 or advw/RK3 "
    "operator over fixed three-halo tiles, after an exact broader prior-stage "
    "token and a fresh matching post-acoustic mass-flux token, with raw-bit "
    "source binding and exact owned scatter. It does not create the broader "
    "stage token, commit solve3 tkea, tile-stream the global pressure solver, "
    "provide MPI/open histories, or establish a full forecast, tornado, or "
    "five-metre meteorological product"
)

_GLOBAL_SOURCE_SHAPES = MappingProxyType(
    {
        "tkea": (76, 300, 300),
        "tke3d": (76, 300, 300),
        "tketen": (76, 300, 300),
        "c1": (76, 300, 300),
        "c2": (76, 300, 300),
        "rrf0": (76, 300, 300),
        "gz": (300, 300),
        "rru": (75, 300, 301),
        "rrv": (75, 301, 300),
        "rrw": (76, 300, 300),
        "divx": (75, 300, 300),
    }
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


def _sha(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _inventory_sha(arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(2, "little"))
        digest.update(encoded)
        digest.update(arrays[name].astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _require_held(token: object) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledTkeHeldStageToken):
        raise TypeError("held_token must be the exact held TKE stage token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_HELD_STAGE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TKE_HELD_STATE_AFTER_RK1"
        or receipt.get("completed_nrk") != 1
        or receipt.get("next_nrk") != 2
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("held TKE stage token identity changed")
    for name in (
        "closure_tkea_baseline_token_sha256",
        "fixed_tkea_sha256",
        "held_tketen_sha256",
        "current_tke3d_sha256",
        "tke_rk1_token_sha256",
    ):
        _lower_sha(name, receipt.get(name))
    return receipt


@dataclass(frozen=True, slots=True)
class Gm1TiledTkeRk23TileSourceToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1TiledTkeRk23Launch:
    ready_event: Any
    tke3d: Any
    workspace: Any
    upstream: CudaChildStretchedLaunch
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledTkeRk23Token:
    receipt: Mapping[str, object]


def _require_prior_tke(
    held: Mapping[str, object], prior_tke_token: object, *, nrk: int
) -> tuple[str, str]:
    if nrk == 2:
        if type(prior_tke_token) is not Gm1Rk1SourceOrderedTkeToken:
            raise TypeError(
                "RK2 requires the exact source-ordered RK1 TKE component token"
            )
        receipt = prior_tke_token.receipt
        if (
            receipt.get("schema")
            != CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA
            or receipt.get("status")
            != CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS
            or receipt.get("completed_nrk") != 1
            or receipt.get("next_nrk") != 2
            or receipt.get("component_key") != "tke_rk"
            or receipt.get("source_order_component_complete") is not True
            or receipt.get("tke_rk1_token_sha256")
            != held.get("tke_rk1_token_sha256")
            or receipt.get("tke_held_state_token_sha256")
            != held.get("operation_token_sha256")
            or receipt.get("output_tke3d_sha256")
            != held.get("current_tke3d_sha256")
            or receipt.get("stream_pointer") != held.get("stream_pointer")
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError(
                "source-ordered RK1 TKE identity, held state, or stream changed"
            )
        return (
            str(receipt["operation_token_sha256"]),
            str(held["current_tke3d_sha256"]),
        )
    if nrk != 3:
        raise ValueError("nrk must be exactly 2 or 3")
    if not isinstance(prior_tke_token, Gm1TiledTkeRk23Token):
        raise TypeError("RK3 requires the exact completed tiled TKE RK2 token")
    receipt = prior_tke_token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE"
        or receipt.get("completed_nrk") != 2
        or receipt.get("held_tke_state_token_sha256")
        != held.get("operation_token_sha256")
        or receipt.get("stream_pointer") != held.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("prior TKE RK2 token identity, lineage, or stream changed")
    return (
        str(receipt["operation_token_sha256"]),
        _lower_sha("prior output_tke3d_sha256", receipt.get("output_tke3d_sha256")),
    )


def _require_stage_prerequisites(
    held_token: object,
    completed_stage_token: object,
    flux_token: object,
    prior_tke_token: object,
    *,
    nrk: int,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    str,
    str,
]:
    held = _require_held(held_token)
    prior_tke_sha, current_sha = _require_prior_tke(
        held, prior_tke_token, nrk=nrk
    )
    if not isinstance(completed_stage_token, Gm1TiledCompletedRkStageToken):
        raise TypeError(
            "completed_stage_token must be the exact broader completed RK token"
        )
    completed = require_gm1_tiled_completed_rk_stage_token(
        completed_stage_token,
        completed_nrk=nrk - 1,
        stream_pointer=int(held["stream_pointer"]),
    )
    components = completed["component_operation_token_sha256"]
    if components.get(GM1_COMPLETED_RK_STAGE_TKE_COMPONENT_KEY) != prior_tke_sha:
        raise ValueError("broader completed RK-stage TKE lineage changed")
    if not isinstance(flux_token, Gm1TiledSolve2PostAcousticFluxToken):
        raise TypeError("flux_token must be the exact post-acoustic mass-flux token")
    flux = flux_token.receipt
    output_sha = flux.get("output_sha256")
    if (
        flux.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or flux.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or flux.get("nrk") != nrk
        or flux.get("stream_pointer") != held.get("stream_pointer")
        or flux.get("completed_rk_stage_token_sha256")
        != completed.get("operation_token_sha256")
        or not isinstance(output_sha, Mapping)
        or set(output_sha) != {"rru", "rrv", "rrw", "divx"}
        or flux.get("operation_token_sha256") != _operation_token(flux)
    ):
        raise ValueError("fresh post-acoustic flux identity or stage lineage changed")
    _lower_sha("flux stage identity", flux.get("stage_identity_sha256"))
    for name in ("rru", "rrv", "rrw", "divx"):
        _lower_sha(f"flux {name}", output_sha[name])
    return held, completed, flux, prior_tke_sha, current_sha


def _validate_host_sources(
    global_sources: Mapping[str, Any],
    *,
    held: Mapping[str, object],
    flux: Mapping[str, object],
    current_sha: str,
) -> Mapping[str, str]:
    np = importlib.import_module("numpy")
    if not isinstance(global_sources, Mapping) or set(global_sources) != set(
        _GLOBAL_SOURCE_SHAPES
    ):
        raise ValueError("global TKE RK2/3 source inventory changed")
    hashes: dict[str, str] = {}
    for name, shape in _GLOBAL_SOURCE_SHAPES.items():
        value = global_sources[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"global {name} must be finite host C-<f4 {shape}")
        hashes[name] = _sha(value)
    if hashes["tkea"] != held.get("fixed_tkea_sha256"):
        raise ValueError("global tkea differs from held fixed baseline")
    if hashes["tketen"] != held.get("held_tketen_sha256"):
        raise ValueError("global tketen differs from held solve1 tendency")
    if hashes["tke3d"] != current_sha:
        raise ValueError("global current tke3d differs from prior TKE stage")
    flux_hashes = flux["output_sha256"]
    for name in ("rru", "rrv", "rrw", "divx"):
        if hashes[name] != flux_hashes[name]:
            raise ValueError(f"global {name} differs from fresh flux token")
    return MappingProxyType(hashes)


def _verify_tile_against_global(
    arrays: Mapping[str, Any],
    global_sources: Mapping[str, Any],
    *,
    spec: Gm1TileSpec,
) -> None:
    np = importlib.import_module("numpy")
    y0, x0, cny, cnx = spec.cj0, spec.ci0, spec.cny, spec.cnx
    y, x = slice(y0, y0 + cny), slice(x0, x0 + cnx)
    xu, yv = slice(x0, x0 + cnx + 1), slice(y0, y0 + cny + 1)
    expected = {
        name: global_sources[name][:, y, x]
        for name in ("tkea", "tke3d", "tketen", "c1", "c2", "rrf0", "divx")
    }
    expected["gz"] = global_sources["gz"][y, x]
    for name, source in expected.items():
        if not np.array_equal(arrays[name].view(np.uint32), source.view(np.uint32)):
            raise ValueError(f"tile {name} differs raw-bit from global source window")
    ng = 3
    padded_expected = {
        "rru": np.zeros_like(arrays["rru"]),
        "rrv": np.zeros_like(arrays["rrv"]),
        "rrw": np.zeros_like(arrays["rrw"]),
    }
    padded_expected["rru"][1:76, ng : ng + cny, ng : ng + cnx + 1] = (
        global_sources["rru"][:, y, xu]
    )
    padded_expected["rrv"][1:76, ng : ng + cny + 1, ng : ng + cnx] = (
        global_sources["rrv"][:, yv, x]
    )
    padded_expected["rrw"][1:77, ng : ng + cny, ng : ng + cnx] = (
        global_sources["rrw"][:, y, x]
    )
    for name, source in padded_expected.items():
        if not np.array_equal(arrays[name].view(np.uint32), source.view(np.uint32)):
            raise ValueError(f"tile padded {name} differs from fresh global flux")


def bind_gm1_tiled_tke_rk23_tile_source(
    held_token: Gm1TiledTkeHeldStageToken,
    completed_stage_token: object,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    prior_tke_token: Gm1TiledTkeRk23Token | None,
    arrays: Mapping[str, Any],
    global_sources: Mapping[str, Any],
    *,
    nrk: int,
    spec: Gm1TileSpec,
    device_h2d_inventory_sha256: str | None = None,
) -> Gm1TiledTkeRk23TileSourceToken:
    """Hash-bind and raw-bit compare one exact RK2/RK3 tile source."""

    held, completed, flux, prior_tke_sha, current_sha = (
        _require_stage_prerequisites(
            held_token,
            completed_stage_token,
            flux_token,
            prior_tke_token,
            nrk=nrk,
        )
    )
    if not isinstance(spec, Gm1TileSpec):
        raise TypeError("spec must be Gm1TileSpec")
    np = importlib.import_module("numpy")
    shapes = tke_evolution_shapes(spec.cnx, spec.cny, 75)
    if not isinstance(arrays, Mapping) or set(arrays) != set(shapes):
        raise ValueError("TKE RK2/3 host tile inventory changed")
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
            raise ValueError(f"host TKE RK2/3 {name} must be finite C-<f4 {shape}")
        start = int(value.ctypes.data)
        ranges.append((start, start + int(value.nbytes), name))
    for index, (start, stop, name) in enumerate(ranges):
        for other_start, other_stop, other_name in ranges[index + 1 :]:
            if start < other_stop and other_start < stop:
                raise ValueError(f"host TKE RK2/3 arrays alias: {name}/{other_name}")
    global_hashes = _validate_host_sources(
        global_sources, held=held, flux=flux, current_sha=current_sha
    )
    _verify_tile_against_global(arrays, global_sources, spec=spec)
    host_sha = _inventory_sha(arrays)
    if device_h2d_inventory_sha256 is not None:
        if (
            _lower_sha("device_h2d_inventory_sha256", device_h2d_inventory_sha256)
            != host_sha
        ):
            raise ValueError("TKE RK2/3 H2D inventory differs from host bytes")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_RK23_SOURCE_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILE_TKE_RK_STAGE_SOURCE",
        "nrk": nrk,
        "tile": spec.record(),
        "held_tke_state_token_sha256": held["operation_token_sha256"],
        "prior_tke_stage_token_sha256": prior_tke_sha,
        "completed_rk_stage_token_sha256": completed["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "global_source_sha256": dict(global_hashes),
        "host_inventory_sha256": host_sha,
        "device_h2d_inventory_sha256": device_h2d_inventory_sha256,
        "h2d_raw_bit_verified": device_h2d_inventory_sha256 is not None,
        "critical_tile_windows_raw_bit_verified": True,
        "compute_shape_zyx": [75, spec.cny, spec.cnx],
        "source_halo": spec.halo,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledTkeRk23TileSourceToken(MappingProxyType(receipt))


def _require_source_token(
    token: object,
    *,
    held: Mapping[str, object],
    completed: Mapping[str, object],
    flux: Mapping[str, object],
    prior_tke_sha: str,
    nrk: int,
    spec: Gm1TileSpec,
) -> Mapping[str, object]:
    if not isinstance(token, Gm1TiledTkeRk23TileSourceToken):
        raise TypeError("source_token must be the exact typed RK2/RK3 tile source")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK23_SOURCE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILE_TKE_RK_STAGE_SOURCE"
        or receipt.get("nrk") != nrk
        or receipt.get("tile") != spec.record()
        or receipt.get("held_tke_state_token_sha256")
        != held.get("operation_token_sha256")
        or receipt.get("prior_tke_stage_token_sha256") != prior_tke_sha
        or receipt.get("completed_rk_stage_token_sha256")
        != completed.get("operation_token_sha256")
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256") != flux.get("stage_identity_sha256")
        or receipt.get("h2d_raw_bit_verified") is not True
        or receipt.get("critical_tile_windows_raw_bit_verified") is not True
        or receipt.get("host_inventory_sha256")
        != receipt.get("device_h2d_inventory_sha256")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("TKE RK2/RK3 tile source identity or transport changed")
    return receipt


def launch_cuda_gm1_tiled_tke_rk23(
    held_token: Gm1TiledTkeHeldStageToken,
    completed_stage_token: object,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    prior_tke_token: Gm1TiledTkeRk23Token | None,
    source_token: Gm1TiledTkeRk23TileSourceToken,
    arrays: Mapping[str, Any],
    *,
    nrk: int,
    spec: Gm1TileSpec,
    metrics: Mapping[str, Any],
    stream: Any,
    compiled: CompiledChildStretchedSgsTke,
    workspace: Any | None = None,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1TiledTkeRk23Launch:
    """Enqueue exact stretched ``advw`` and RK2/RK3 for one tile."""

    held, completed, flux, prior_tke_sha, _ = _require_stage_prerequisites(
        held_token,
        completed_stage_token,
        flux_token,
        prior_tke_token,
        nrk=nrk,
    )
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    source = _require_source_token(
        source_token,
        held=held,
        completed=completed,
        flux=flux,
        prior_tke_sha=prior_tke_sha,
        nrk=nrk,
        spec=spec,
    )
    if stream is None or not hasattr(stream, "ptr"):
        raise TypeError("stream must be an explicit CUDA stream")
    if int(stream.ptr) != held.get("stream_pointer"):
        raise ValueError("TKE RK2/RK3 left the admitted CUDA stream")
    cupy = importlib.import_module("cupy")
    shapes = tke_evolution_shapes(spec.cnx, spec.cny, 75)
    if not isinstance(arrays, Mapping) or set(arrays) != set(shapes):
        raise ValueError("TKE RK2/RK3 device tile inventory changed")
    for name, shape in shapes.items():
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"device TKE RK2/RK3 {name} must be C-f32 {shape}")
    if workspace is None:
        workspace = cupy.empty_like(arrays["tke3d"])
    upstream = launch_cuda_gm1_child_stretched_tke_rk_stage(
        arrays,
        workspace=workspace,
        metrics=metrics,
        ni=spec.cnx,
        nj=spec.cny,
        nk=75,
        nrk=nrk,
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
        "schema": CUDA_GM1_TILESTREAM_TKE_RK23_SCHEMA,
        "status": "ENQUEUED_LIVE_TILE_TKE_ADVW_RK_STAGE",
        "nrk": nrk,
        "tile": spec.record(),
        "source_token_sha256": source["operation_token_sha256"],
        "held_tke_state_token_sha256": held["operation_token_sha256"],
        "prior_tke_stage_token_sha256": prior_tke_sha,
        "completed_rk_stage_token_sha256": completed["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "stage_identity_sha256": flux["stage_identity_sha256"],
        "stream_pointer": int(stream.ptr),
        "same_stream_after_fresh_post_acoustic_flux": True,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "source_halo": spec.halo,
        "scatter_disposition": "owned mass rectangle only after event completion",
        "kernel_source_sha256": CUDA_GM1_CHILD_STRETCHED_TKE_SOURCE_SHA256,
        "upstream_receipt": dict(upstream.receipt),
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledTkeRk23Launch(
        upstream.ready_event,
        arrays["tke3d"],
        workspace,
        upstream,
        MappingProxyType(receipt),
    )


def finish_cuda_gm1_tiled_tke_rk23(
    held_token: Gm1TiledTkeHeldStageToken,
    completed_stage_token: object,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    prior_tke_token: Gm1TiledTkeRk23Token | None,
    launches: Sequence[CudaGm1TiledTkeRk23Launch],
    *,
    nrk: int,
    plan: Gm1TilePlan,
    output_tke3d: Any,
) -> Gm1TiledTkeRk23Token:
    """Admit exact-once RK2/RK3 scatter and bind the current TKE."""

    held, completed, flux, prior_tke_sha, _ = _require_stage_prerequisites(
        held_token,
        completed_stage_token,
        flux_token,
        prior_tke_token,
        nrk=nrk,
    )
    if not isinstance(plan, Gm1TilePlan) or (plan.nx, plan.ny) != (300, 300):
        raise ValueError("TKE RK2/RK3 completion requires the admitted 300 grid")
    if len(launches) != plan.ntiles:
        raise ValueError("TKE RK2/RK3 completion requires every exact tile")
    streams: set[int] = set()
    launch_tokens: list[str] = []
    source_tokens: list[str] = []
    owned_words = 0
    for spec, launch in zip(plan.specs, launches, strict=True):
        if not isinstance(launch, CudaGm1TiledTkeRk23Launch):
            raise TypeError("TKE RK2/RK3 completion requires typed launches")
        receipt = launch.receipt
        y0, y1, x0, x1 = spec.owned_rect("mass")
        owned_words += 76 * (y1 - y0) * (x1 - x0)
        if (
            receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK23_SCHEMA
            or receipt.get("status") != "ENQUEUED_LIVE_TILE_TKE_ADVW_RK_STAGE"
            or receipt.get("nrk") != nrk
            or receipt.get("tile") != spec.record()
            or receipt.get("held_tke_state_token_sha256")
            != held.get("operation_token_sha256")
            or receipt.get("prior_tke_stage_token_sha256") != prior_tke_sha
            or receipt.get("completed_rk_stage_token_sha256")
            != completed.get("operation_token_sha256")
            or receipt.get("post_acoustic_flux_token_sha256")
            != flux.get("operation_token_sha256")
            or receipt.get("stage_identity_sha256")
            != flux.get("stage_identity_sha256")
            or receipt.get("operation_token_sha256") != _operation_token(receipt)
        ):
            raise ValueError("TKE RK2/RK3 tile identity or cadence changed")
        streams.add(int(receipt["stream_pointer"]))
        source_tokens.append(str(receipt["source_token_sha256"]))
        launch_tokens.append(str(receipt["operation_token_sha256"]))
    expected_words = 76 * 300 * 300
    if owned_words != expected_words:
        raise ValueError("TKE RK2/RK3 scatter ownership is not exact once")
    if len(streams) != 1 or next(iter(streams)) != held.get("stream_pointer"):
        raise ValueError("TKE RK2/RK3 tile streams differ from prerequisites")
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
        raise ValueError("TKE RK2/RK3 did not preserve rigid zero bottom/top faces")
    if bool(np.any(output_tke3d[1:-1] < np.float32(0.0))):
        raise ValueError("TKE RK2/RK3 output contains negative interior TKE")
    output_sha = _sha(output_tke3d)
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE",
        "completed_nrk": nrk,
        "held_tke_state_token_sha256": held["operation_token_sha256"],
        "prior_tke_stage_token_sha256": prior_tke_sha,
        "completed_prior_rk_stage_token_sha256": completed[
            "operation_token_sha256"
        ],
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
        "output_tke3d_shape": [76, 300, 300],
        "output_tke3d_words": expected_words,
        "output_tke3d_sha256": output_sha,
        "final_bcw_applied": True,
        "ready_for": (
            "broader RK2 completion, then fresh nrk=3 sound/getdivx"
            if nrk == 2
            else "solve3 TKE commit only after exact completed broader RK3"
        ),
        "structural_refusals": [
            "TKE-only assertion of broader RK completion",
            "reuse of an earlier-stage post-acoustic flux token",
            "solve3 tkea commit before exact completed broader RK3",
            "global psolver tilestream claim",
            "MPI exchange or open-history production",
            "full forecast or five-metre meteorological product",
        ],
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(AUTHORITY_ANCHORS),
        },
        "claim_limit": GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledTkeRk23Token(MappingProxyType(receipt))


def require_gm1_tiled_tke_rk23_token(
    token: object,
    *,
    completed_nrk: int | None = None,
    stream_pointer: int | None = None,
) -> Mapping[str, object]:
    """Validate one exact admitted RK2/RK3 TKE component token."""

    if type(token) is not Gm1TiledTkeRk23Token:
        raise TypeError("token must be the exact Gm1TiledTkeRk23Token type")
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("TKE RK2/RK3 token receipt must be a mapping")
    nrk = receipt.get("completed_nrk")
    if isinstance(nrk, bool) or not isinstance(nrk, int) or nrk not in (2, 3):
        raise ValueError("TKE RK2/RK3 completed_nrk must be exactly 2 or 3")
    if completed_nrk is not None:
        if (
            isinstance(completed_nrk, bool)
            or not isinstance(completed_nrk, int)
            or completed_nrk not in (2, 3)
        ):
            raise ValueError("completed_nrk must be exactly 2 or 3")
        if nrk != completed_nrk:
            raise ValueError("TKE RK2/RK3 token belongs to another RK stage")
    expected_ready = (
        "broader RK2 completion, then fresh nrk=3 sound/getdivx"
        if nrk == 2
        else "solve3 TKE commit only after exact completed broader RK3"
    )
    expected_refusals = [
        "TKE-only assertion of broader RK completion",
        "reuse of an earlier-stage post-acoustic flux token",
        "solve3 tkea commit before exact completed broader RK3",
        "global psolver tilestream claim",
        "MPI exchange or open-history production",
        "full forecast or five-metre meteorological product",
    ]
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE"
        or receipt.get("host_scatter_exact_once") is not True
        or receipt.get("output_tke3d_shape") != [76, 300, 300]
        or receipt.get("output_tke3d_words") != 76 * 300 * 300
        or receipt.get("final_bcw_applied") is not True
        or receipt.get("ready_for") != expected_ready
        or receipt.get("structural_refusals") != expected_refusals
        or receipt.get("claim_limit") != GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("TKE RK2/RK3 token identity, output, or claim changed")
    actual_stream = receipt.get("stream_pointer")
    if (
        isinstance(actual_stream, bool)
        or not isinstance(actual_stream, int)
        or actual_stream <= 0
    ):
        raise ValueError("TKE RK2/RK3 stream pointer is invalid")
    if stream_pointer is not None:
        if (
            isinstance(stream_pointer, bool)
            or not isinstance(stream_pointer, int)
            or stream_pointer <= 0
        ):
            raise ValueError("stream_pointer must be a positive integer")
        if actual_stream != stream_pointer:
            raise ValueError("TKE RK2/RK3 token belongs to another stream")
    for name in (
        "held_tke_state_token_sha256",
        "prior_tke_stage_token_sha256",
        "completed_prior_rk_stage_token_sha256",
        "post_acoustic_flux_token_sha256",
        "stage_identity_sha256",
        "tile_plan_sha256",
        "tile_source_tokens_sha256",
        "tile_operation_tokens_sha256",
        "output_tke3d_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(name, receipt.get(name))
    tile_count = receipt.get("tile_count")
    if (
        isinstance(tile_count, bool)
        or not isinstance(tile_count, int)
        or tile_count <= 0
    ):
        raise ValueError("TKE RK2/RK3 tile count is invalid")
    return MappingProxyType(dict(receipt))


__all__ = (
    "COMPLETED_RK_STAGE_STATUS",
    "CUDA_GM1_TILESTREAM_TKE_RK23_SCHEMA",
    "CUDA_GM1_TILESTREAM_TKE_RK23_SOURCE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA",
    "CudaGm1TiledTkeRk23Launch",
    "GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT",
    "Gm1TiledTkeRk23TileSourceToken",
    "Gm1TiledTkeRk23Token",
    "bind_gm1_tiled_tke_rk23_tile_source",
    "finish_cuda_gm1_tiled_tke_rk23",
    "launch_cuda_gm1_tiled_tke_rk23",
    "require_gm1_tiled_tke_rk23_token",
)
