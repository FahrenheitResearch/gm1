"""Typed RK2/RK3 global-acoustic boundary for the real 50 m GM1 child.

CM1 may enter RK2 or RK3 only after a broader completed prior-stage token and
a fresh current-stage pre-acoustic momentum result.  This module owns the
shared psolver=3 ``sound`` family and its authenticated RK-specific cadence.
The fresh producer is implemented separately in
``cuda_gm1_tilestream_rk23_pre_acoustic_momentum``.  This module emits no
completed current-stage token, and every launch remains fail-closed until its
exact runtime prior-stage token and fresh result are supplied.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, NoReturn

from .cuda_gm1_child_psolver3_open import (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA,
    CudaGm1ChildPsolver3OpenLaunch,
    launch_cuda_gm1_child_psolver3_open_stage,
)
from .cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
from .cuda_gm1_child_terrain import CudaGm1ChildTerrainMetricsLaunch
from .cuda_gm1_tilestream_completed_rk_stage import (
    Gm1TiledCompletedRkStageToken,
    require_gm1_tiled_completed_rk_stage_token,
)
from .cuda_moist_stage_bridge import CudaMoistStageBridgeState
from .cuda_psolver3_terrain_open import TerrainOpenSoundAblations
from .cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32


CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA = (
    "gm1.tilestream-rk23-pre-acoustic-momentum-token/v1"
)
CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK23_PRE_ACOUSTIC_MOMENTUM"
)
CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_SCHEMA = (
    "gm1.cuda-tilestream-rk23-pre-acoustic-binding/v1"
)
CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_STATUS = (
    "BOUND_LIVE_TILESTREAM_RK23_PRE_ACOUSTIC_SAME_RESIDENCY"
)
CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_SCHEMA = "gm1.cuda-tilestream-rk23-global-acoustic/v1"
CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_STATUS = (
    "ADMITTED_LIVE_300X300X75_RK23_GLOBAL_ACOUSTIC"
)
CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_SCHEMA = (
    "gm1.rk23-post-sound-averaged-velocity-token/v1"
)
CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_STATUS = (
    "ADMITTED_LIVE_RK23_POST_SOUND_AVERAGED_VELOCITY"
)
CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA = (
    "gm1.tilestream-rk23-acoustic-continuation-token/v1"
)
CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK23_ACOUSTIC_CONTINUATION"
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
AUTHORITY_BC_SHA256 = "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3"

GM1_RK23_ACOUSTIC_SOURCE_ANCHORS = (
    "src/solve2.F:296-299",
    "src/solve2.F:314-1051",
    "src/solve2.F:1053-1173",
    "src/sound.F:84-105",
    "src/sound.F:107-837",
    "src/solve2.F:1254-1307",
    "src/solve2.F:1310-1706",
)
GM1_RK23_ACOUSTIC_SOURCE_ORDER = (
    "consume exact broader completed prior RK stage on the same live stream",
    "consume a fresh same-nrk pre-acoustic result after current-velocity terrain boundary replay and mass-flux rebuild",
    "solve2.F:299 uses dttmp=dt/(nrkmax+1-nrk) in default REAL arithmetic",
    "sound.F:84-100 selects RK2/RK3 nloop and binary32 dts from nsound=6",
    "sound.F:107-837 advances the global stretched terrain/four-open acoustic state",
    "solve2.F:1258-1271 produces current-stage radiation history",
    "publish fresh post-sound and acoustic continuation tokens before getdiv/getdivx",
    "defer solve2.F:1662-1706 stage-family fills to the post-scalar boundary owner",
)
GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS = (
    "uten",
    "vten",
    "wten",
    "rru",
    "rrv",
    "rrw",
    "divx",
)
GM1_RK23_PRE_ACOUSTIC_OUTPUT_SHAPES = MappingProxyType(
    {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "rru": [75, 300, 301],
        "rrv": [75, 301, 300],
        "rrw": [76, 300, 300],
        "divx": [75, 300, 300],
    }
)
GM1_RK23_ACOUSTIC_OUTPUT_KEYS = (
    "u3d",
    "v3d",
    "w3d",
    "pp3d",
    "rru",
    "rrv",
    "rrw",
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
)
GM1_RK23_ACOUSTIC_OUTPUT_SHAPES = MappingProxyType(
    {
        "u3d": [75, 300, 301],
        "v3d": [75, 301, 300],
        "w3d": [76, 300, 300],
        "pp3d": [75, 300, 300],
        "rru": [75, 300, 301],
        "rrv": [75, 301, 300],
        "rrw": [76, 300, 300],
        "radbcw": [77, 306],
        "radbce": [77, 306],
        "radbcs": [77, 306],
        "radbcn": [77, 306],
    }
)
LIVE_RK23_ACOUSTIC_ABLATIONS = TerrainOpenSoundAblations(stage_family_fills=False)
GM1_RK23_ACOUSTIC_CLAIM_LIMIT = (
    "one source-ordered real 300x300x75, 50 m RK2 or RK3 global psolver=3 "
    "acoustic continuation after an exact completed prior-stage token and a "
    "fresh same-stage pre-acoustic result. getdiv/getdivx, scalar/moisture/TKE "
    "RK, late family fills, completed current stage, solve3, checkpoint, "
    "forecast, product, tornado, and five-metre claims remain outside"
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


def _requested_nrk(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (2, 3):
        raise ValueError("requested_nrk must be exactly 2 or 3")
    return value


def _schedule(requested_nrk: int) -> Mapping[str, object]:
    requested_nrk = _requested_nrk(requested_nrk)
    nloop, dts = sound3_stage_schedule_f32(0.5, 6, requested_nrk)
    return MappingProxyType(
        {
            "rk_stage_dt_seconds_f32": rk_stage_dt_f32(0.5, requested_nrk),
            "sound_substeps": nloop,
            "sound_dt_seconds_f32": dts,
        }
    )


def audit_gm1_rk23_live_acoustic_source_contract(
    requested_nrk: int,
) -> Mapping[str, object]:
    """Return the exact RK-specific source contract without issuing a token."""

    requested_nrk = _requested_nrk(requested_nrk)
    receipt: dict[str, object] = {
        "schema": "gm1.rk23-live-acoustic-source-audit/v1",
        "status": "SOURCE_AUDITED_RUNTIME_COMPLETED_STAGE_AND_PRE_RESULT_REQUIRED",
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "selected_profile": {
            "domain_zyx": [75, 300, 300],
            "dx_m": 50.0,
            "dy_m": 50.0,
            "dt_seconds": 0.5,
            "nrkmax": 3,
            "psolver": 3,
            "nsound": 6,
            "terrain": True,
            "stretch_z": 1,
            "lateral_boundaries": [2, 2, 2, 2],
            "mpi": False,
        },
        "schedule": dict(_schedule(requested_nrk)),
        "source_order": list(GM1_RK23_ACOUSTIC_SOURCE_ORDER),
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "sound_sha256": AUTHORITY_SOUND_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(GM1_RK23_ACOUSTIC_SOURCE_ANCHORS),
        },
        "operator_source_sha256": {
            "core": CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
            "boundary": CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256,
        },
        "operator_switches": {
            "open_consumption": True,
            "terrain_pressure_gradient": True,
            "transformed_divergence": True,
            "pressure_advection": True,
            "implicit_vertical_solve": True,
            "between_step_pressure_fill": True,
            "radiation_history_production": True,
            "stage_family_fills": False,
        },
        "current_velocity_boundary_replay_required": True,
        "historical_rk1_token_admissible": False,
        "source_audit_carries_runtime_completed_stage": False,
        "source_audit_carries_runtime_pre_acoustic_result": False,
        "runtime_pre_acoustic_producer_module": (
            "cm1core.cuda_gm1_tilestream_rk23_pre_acoustic_momentum"
        ),
        "continuation_token_emitted": False,
        "completed_stage_token_emitted": False,
        "claim_limit": GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    receipt["source_contract_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return MappingProxyType(receipt)


@dataclass(frozen=True, slots=True)
class Gm1TiledRk23PreAcousticMomentumToken:
    """Exact fresh pre-acoustic token produced by the separate RK23 module."""

    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk23PreAcousticMomentumBinding:
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    vertical: Gm1ChildStretchedMetrics
    stream: Any
    bridge: CudaMoistStageBridgeState
    stage_identity_sha256: str
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FinishedGm1TiledRk23PreAcousticMomentum:
    token: Gm1TiledRk23PreAcousticMomentumToken
    binding: Gm1TiledRk23PreAcousticMomentumBinding
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken
    requested_nrk: int
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    vertical: Gm1ChildStretchedMetrics
    stream: Any
    bridge: CudaMoistStageBridgeState
    output_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk23PostSoundVelocityToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk23AcousticContinuationToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CudaGm1TiledRk23AcousticLaunch:
    ready_event: Any
    upstream: CudaGm1ChildPsolver3OpenLaunch
    requested_nrk: int
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    vertical: Gm1ChildStretchedMetrics
    stream: Any
    bridge: CudaMoistStageBridgeState
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    pre_acoustic_result: FinishedGm1TiledRk23PreAcousticMomentum
    pre_acoustic_token: Gm1TiledRk23PreAcousticMomentumToken
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk23AcousticLiveHandoff:
    ready_event: Any
    stage: CudaGm1ChildPsolver3OpenLaunch
    requested_nrk: int
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    vertical: Gm1ChildStretchedMetrics
    stream: Any
    bridge: CudaMoistStageBridgeState
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    pre_acoustic_result: FinishedGm1TiledRk23PreAcousticMomentum
    pre_acoustic_token: Gm1TiledRk23PreAcousticMomentumToken
    acoustic_token: Gm1TiledRk23AcousticContinuationToken
    post_sound_velocity_token: Gm1TiledRk23PostSoundVelocityToken


@dataclass(frozen=True, slots=True)
class FinishedGm1TiledRk23Acoustic:
    handoff: Gm1TiledRk23AcousticLiveHandoff
    output_sha256: Mapping[str, str]
    execution_receipt: Mapping[str, object]
    continuation_result: Any


def _require_pre_acoustic_result(
    result: object,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, Any],
    Mapping[str, Any],
    Gm1ChildStretchedMetrics,
    Any,
    CudaMoistStageBridgeState,
    CudaGm1ChildTerrainMetricsLaunch,
]:
    if type(result) is not FinishedGm1TiledRk23PreAcousticMomentum:
        raise TypeError(
            "pre_acoustic_result must be exact FinishedGm1TiledRk23PreAcousticMomentum"
        )
    requested_nrk = _requested_nrk(result.requested_nrk)
    if type(result.prior_completed_stage_token) is not Gm1TiledCompletedRkStageToken:
        raise TypeError("RK23 pre-acoustic result lost exact completed-stage token")
    prior = require_gm1_tiled_completed_rk_stage_token(
        result.prior_completed_stage_token,
        completed_nrk=requested_nrk - 1,
    )
    token = _require_pre_acoustic_token(
        result.token,
        requested_nrk=requested_nrk,
        prior_completed_stage_token=result.prior_completed_stage_token,
    )
    source = audit_gm1_rk23_live_acoustic_source_contract(requested_nrk)
    hashes, shapes = token.get("output_sha256"), token.get("output_shapes")
    if (
        not isinstance(hashes, Mapping)
        or tuple(hashes) != GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS
        or not isinstance(shapes, Mapping)
        or dict(shapes) != dict(GM1_RK23_PRE_ACOUSTIC_OUTPUT_SHAPES)
        or dict(result.output_sha256) != dict(hashes)
    ):
        raise ValueError("RK23 pre-acoustic output inventory changed")
    for name in GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS:
        _lower_sha(f"RK23 pre output {name}", hashes.get(name))
    binding = result.binding
    if type(binding) is not Gm1TiledRk23PreAcousticMomentumBinding:
        raise TypeError("RK23 pre-acoustic result lost exact binding")
    binding_receipt = binding.receipt
    if (
        binding_receipt.get("schema")
        != CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_SCHEMA
        or binding_receipt.get("status")
        != CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_STATUS
        or binding_receipt.get("requested_nrk") != requested_nrk
        or binding_receipt.get("prior_completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or binding_receipt.get("stream_pointer") != prior.get("stream_pointer")
        or binding_receipt.get("stage_identity_sha256")
        != token.get("stage_identity_sha256")
        or binding_receipt.get("operation_token_sha256")
        != _operation_token(binding_receipt)
    ):
        raise ValueError("RK23 pre-acoustic binding identity changed")
    arrays, metrics = result.arrays, result.metrics
    if not isinstance(arrays, Mapping) or not isinstance(metrics, Mapping):
        raise TypeError("RK23 pre-acoustic result lost resident arrays or metrics")
    if type(result.vertical) is not Gm1ChildStretchedMetrics:
        raise TypeError("RK23 pre-acoustic result lost stretched vertical binding")
    if type(result.bridge) is not CudaMoistStageBridgeState:
        raise TypeError("RK23 pre-acoustic result lost moist-stage binding")
    if type(result.terrain_metrics) is not CudaGm1ChildTerrainMetricsLaunch:
        raise TypeError("RK23 pre-acoustic result lost terrain-metric launch")
    if result.stream is None or not hasattr(result.stream, "ptr"):
        raise TypeError("RK23 pre-acoustic result lost CUDA stream")
    if int(result.stream.ptr) != prior.get("stream_pointer"):
        raise ValueError("RK23 pre-acoustic result stream differs from prior stage")
    if (
        binding.prior_completed_stage_token is not result.prior_completed_stage_token
        or binding.arrays is not arrays
        or binding.metrics is not metrics
        or binding.vertical is not result.vertical
        or binding.stream is not result.stream
        or binding.bridge is not result.bridge
        or binding.terrain_metrics is not result.terrain_metrics
        or binding.stage_identity_sha256 != token.get("stage_identity_sha256")
    ):
        raise ValueError("RK23 pre-acoustic finished result changed live residency")
    return (
        token,
        prior,
        source,
        arrays,
        metrics,
        result.vertical,
        result.stream,
        result.bridge,
        result.terrain_metrics,
    )


def _require_upstream_receipt(
    launch: CudaGm1ChildPsolver3OpenLaunch,
    *,
    requested_nrk: int,
    allow_uniform_idealized_validation: bool = False,
) -> None:
    receipt = launch.receipt
    schedule = _schedule(requested_nrk)
    cadence = [
        {
            "nrk": requested_nrk,
            "nloop": schedule["sound_substeps"],
            "dts_f32": schedule["sound_dt_seconds_f32"],
            "open_consumptions": schedule["sound_substeps"],
            "pressure_fills": int(schedule["sound_substeps"]) - 1,
            "radiation_productions": 1,
        }
    ]
    domain = receipt.get("domain")
    kernels = receipt.get("kernels")
    if (
        receipt.get("schema") != CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA
        or not isinstance(domain, Mapping)
        or dict(domain)
        != {
            "ni": 300,
            "nj": 300,
            "nk": 75,
            "dx_m": 50.0,
            "dy_m": 50.0,
            "ztop_m": 15000.0,
            "compact_validation": allow_uniform_idealized_validation,
        }
        or receipt.get("cadence") != cadence
        or not isinstance(kernels, Mapping)
        or kernels.get("core_source_sha256")
        != CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256
        or kernels.get("boundary_source_sha256")
        != CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256
        or receipt.get("source_valid_padded_rho0") is not True
        or receipt.get("scalar_dz_fallback") is not False
        or receipt.get("uniform_idealized_validation")
        is not allow_uniform_idealized_validation
    ):
        raise ValueError("RK23 global psolver source, domain, or cadence changed")


def _require_pre_acoustic_token(
    token: object,
    *,
    requested_nrk: int,
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken,
) -> Mapping[str, object]:
    requested_nrk = _requested_nrk(requested_nrk)
    if type(token) is not Gm1TiledRk23PreAcousticMomentumToken:
        raise TypeError("pre token must be exact RK23 pre-acoustic token")
    if type(prior_completed_stage_token) is not Gm1TiledCompletedRkStageToken:
        raise TypeError("prior token must be exact Gm1TiledCompletedRkStageToken")
    prior = require_gm1_tiled_completed_rk_stage_token(
        prior_completed_stage_token,
        completed_nrk=requested_nrk - 1,
    )
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("RK23 pre-acoustic token receipt must be a mapping")
    source = audit_gm1_rk23_live_acoustic_source_contract(requested_nrk)
    hashes, shapes = receipt.get("output_sha256"), receipt.get("output_shapes")
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS
        or receipt.get("requested_nrk") != requested_nrk
        or receipt.get("required_completed_nrk") != requested_nrk - 1
        or receipt.get("prior_completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or not isinstance(receipt.get("solve1_full_held_token_sha256"), str)
        or receipt.get("stream_pointer") != prior.get("stream_pointer")
        or receipt.get("source_contract_sha256") != source.get("source_contract_sha256")
        or receipt.get("current_velocity_boundary_replay") is not True
        or receipt.get("fresh_current_stage_mass_flux") is not True
        or receipt.get("historical_rk1_post_sound_token_reused") is not False
        or receipt.get("completed_rk_stage") is not False
        or receipt.get("claim_limit") != GM1_RK23_ACOUSTIC_CLAIM_LIMIT
        or not isinstance(hashes, Mapping)
        or tuple(hashes) != GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS
        or not isinstance(shapes, Mapping)
        or dict(shapes) != dict(GM1_RK23_PRE_ACOUSTIC_OUTPUT_SHAPES)
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK23 pre-acoustic token identity or chronology changed")
    _lower_sha("RK23 pre stage identity", receipt.get("stage_identity_sha256"))
    _lower_sha(
        "RK23 solve1 full-held token",
        receipt.get("solve1_full_held_token_sha256"),
    )
    for name in GM1_RK23_PRE_ACOUSTIC_OUTPUT_KEYS:
        _lower_sha(f"RK23 pre output {name}", hashes.get(name))
    return MappingProxyType(dict(receipt))


def _require_post_sound_velocity_token(
    token: object,
    *,
    requested_nrk: int,
    prior: Mapping[str, object],
    pre: Mapping[str, object],
) -> Mapping[str, object]:
    requested_nrk = _requested_nrk(requested_nrk)
    if type(token) is not Gm1TiledRk23PostSoundVelocityToken:
        raise TypeError("velocity token must be exact RK23 post-sound token")
    receipt = token.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("RK23 post-sound token receipt must be a mapping")
    expected_shapes = {
        target: GM1_RK23_ACOUSTIC_OUTPUT_SHAPES[source_name]
        for target, source_name in (
            ("u3d", "rru"),
            ("v3d", "rrv"),
            ("w3d", "rrw"),
        )
    }
    velocity_hashes = receipt.get("velocity_sha256")
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_STATUS
        or receipt.get("requested_nrk") != requested_nrk
        or receipt.get("required_completed_nrk") != requested_nrk - 1
        or receipt.get("prior_completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or receipt.get("pre_acoustic_momentum_rk_token_sha256")
        != pre.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256") != pre.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != pre.get("stream_pointer")
        or receipt.get("schedule") != _schedule(requested_nrk)
        or receipt.get("velocity_shapes") != expected_shapes
        or not isinstance(velocity_hashes, Mapping)
        or tuple(velocity_hashes) != ("u3d", "v3d", "w3d")
        or receipt.get("historical_rk1_token_reused") is not False
        or receipt.get("completed_rk_stage") is not False
        or receipt.get("claim_limit") != GM1_RK23_ACOUSTIC_CLAIM_LIMIT
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK23 post-sound velocity identity or chronology changed")
    for name in ("u3d", "v3d", "w3d"):
        _lower_sha(f"RK23 post-sound velocity {name}", velocity_hashes.get(name))
    return MappingProxyType(dict(receipt))


def launch_cuda_gm1_tilestream_rk23_acoustic(
    pre_acoustic_result: FinishedGm1TiledRk23PreAcousticMomentum,
    *,
    compiled: Any | None = None,
    open_compiled: Any | None = None,
    lateral_compiled: Any | None = None,
    allow_uniform_idealized_validation: bool = False,
) -> CudaGm1TiledRk23AcousticLaunch:
    """Enqueue RK2/RK3 sound only after exact prior and fresh-pre lineage."""

    (
        pre,
        prior,
        source,
        arrays,
        metrics,
        vertical,
        stream,
        bridge,
        terrain_metrics,
    ) = _require_pre_acoustic_result(pre_acoustic_result)
    requested_nrk = int(pre["requested_nrk"])
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    if (
        pre.get("uniform_idealized_validation") is True
    ) != allow_uniform_idealized_validation:
        raise ValueError(
            "RK23 acoustic idealized admission differs from pre-acoustic lineage"
        )
    upstream = launch_cuda_gm1_child_psolver3_open_stage(
        arrays,
        metrics,
        ni=300,
        nj=300,
        nk=75,
        nrk=requested_nrk,
        dt=0.5,
        nsound=6,
        dx=50.0,
        dy=50.0,
        vertical=vertical,
        ablations=LIVE_RK23_ACOUSTIC_ABLATIONS,
        stream=stream,
        compiled=compiled,
        open_compiled=open_compiled,
        lateral_compiled=lateral_compiled,
        allow_compact_validation=allow_uniform_idealized_validation,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    _require_upstream_receipt(
        upstream,
        requested_nrk=requested_nrk,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_SCHEMA,
        "status": "ENQUEUED_LIVE_300X300X75_RK23_GLOBAL_ACOUSTIC",
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": int(stream.ptr),
        "schedule": dict(source["schedule"]),
        "psolver_receipt_sha256": hashlib.sha256(
            _canonical(upstream.receipt)
        ).hexdigest(),
        "source_contract_sha256": source["source_contract_sha256"],
        "stage_family_fills_deferred": True,
        "getdiv_getdivx_performed": False,
        "host_sync_performed": False,
        "completed_rk_stage": False,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "claim_limit": GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return CudaGm1TiledRk23AcousticLaunch(
        upstream.ready_event,
        upstream,
        requested_nrk,
        pre_acoustic_result.prior_completed_stage_token,
        arrays,
        metrics,
        vertical,
        stream,
        bridge,
        terrain_metrics,
        pre_acoustic_result,
        pre_acoustic_result.token,
        MappingProxyType(receipt),
    )


_PHYSICAL_SLICES = MappingProxyType(
    {
        "u3d": (slice(1, 76), slice(3, 303), slice(3, 304)),
        "v3d": (slice(1, 76), slice(3, 304), slice(3, 303)),
        "w3d": (slice(1, 77), slice(3, 303), slice(3, 303)),
        "pp3d": (slice(1, 76), slice(1, 301), slice(1, 301)),
        "rru": (slice(1, 76), slice(3, 303), slice(3, 304)),
        "rrv": (slice(1, 76), slice(3, 304), slice(3, 303)),
        "rrw": (slice(1, 77), slice(3, 303), slice(3, 303)),
        "radbcw": (slice(None), slice(None)),
        "radbce": (slice(None), slice(None)),
        "radbcs": (slice(None), slice(None)),
        "radbcn": (slice(None), slice(None)),
    }
)


def _host_output(cupy: Any, value: Any, slices: tuple[slice, ...]) -> Any:
    np = importlib.import_module("numpy")
    host = np.ascontiguousarray(cupy.asnumpy(value[slices]), dtype=np.float32)
    if not bool(np.all(np.isfinite(host))):
        raise ValueError("RK23 post-sound acoustic output contains non-finite words")
    return host


def require_gm1_tiled_rk23_acoustic_continuation(
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken,
    pre_acoustic_token: Gm1TiledRk23PreAcousticMomentumToken,
    post_sound_velocity_token: Gm1TiledRk23PostSoundVelocityToken,
    acoustic_token: Gm1TiledRk23AcousticContinuationToken,
) -> Mapping[str, object]:
    """Validate exact prior -> fresh-pre -> acoustic lineage for RK2/RK3."""

    if type(acoustic_token) is not Gm1TiledRk23AcousticContinuationToken:
        raise TypeError("acoustic_token must be exact RK23 acoustic token")
    acoustic = acoustic_token.receipt
    if not isinstance(acoustic, Mapping):
        raise TypeError("RK23 acoustic token receipt must be a mapping")
    requested_nrk = _requested_nrk(acoustic.get("requested_nrk"))
    prior = require_gm1_tiled_completed_rk_stage_token(
        prior_completed_stage_token,
        completed_nrk=requested_nrk - 1,
    )
    pre = _require_pre_acoustic_token(
        pre_acoustic_token,
        requested_nrk=requested_nrk,
        prior_completed_stage_token=prior_completed_stage_token,
    )
    velocity = _require_post_sound_velocity_token(
        post_sound_velocity_token,
        requested_nrk=requested_nrk,
        prior=prior,
        pre=pre,
    )
    schedule = _schedule(requested_nrk)
    if (
        acoustic.get("schema")
        != CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA
        or acoustic.get("status")
        != CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_STATUS
        or acoustic.get("operation_token_sha256") != _operation_token(acoustic)
        or pre.get("requested_nrk") != requested_nrk
        or pre.get("prior_completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or acoustic.get("prior_completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or acoustic.get("pre_acoustic_momentum_rk_token_sha256")
        != pre.get("operation_token_sha256")
        or acoustic.get("post_sound_velocity_token_sha256")
        != velocity.get("operation_token_sha256")
        or acoustic.get("stage_identity_sha256") != pre.get("stage_identity_sha256")
        or acoustic.get("stream_pointer") != prior.get("stream_pointer")
        or acoustic.get("stream_pointer") != pre.get("stream_pointer")
        or acoustic.get("schedule") != schedule
        or acoustic.get("historical_rk1_token_reused") is not False
        or acoustic.get("completed_rk_stage") is not False
        or acoustic.get("claim_limit") != GM1_RK23_ACOUSTIC_CLAIM_LIMIT
    ):
        raise ValueError("RK23 prior/pre/acoustic lineage or cadence changed")
    hashes, shapes = acoustic.get("output_sha256"), acoustic.get("output_shapes")
    if (
        not isinstance(hashes, Mapping)
        or tuple(hashes) != GM1_RK23_ACOUSTIC_OUTPUT_KEYS
        or not isinstance(shapes, Mapping)
        or dict(shapes) != dict(GM1_RK23_ACOUSTIC_OUTPUT_SHAPES)
    ):
        raise ValueError("RK23 acoustic output inventory changed")
    for name in GM1_RK23_ACOUSTIC_OUTPUT_KEYS:
        _lower_sha(f"RK23 acoustic output {name}", hashes.get(name))
    velocity_hashes = velocity["velocity_sha256"]
    if any(
        velocity_hashes[target] != hashes[source_name]
        for target, source_name in (
            ("u3d", "rru"),
            ("v3d", "rrv"),
            ("w3d", "rrw"),
        )
    ):
        raise ValueError("RK23 post-sound velocity hashes differ from live outputs")
    audit = {
        "schema": "gm1.rk23-acoustic-lineage-audit/v1",
        "status": "VERIFIED_LIVE_RK23_ACOUSTIC_LINEAGE_ONLY",
        "requested_nrk": requested_nrk,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "post_sound_velocity_token_sha256": velocity["operation_token_sha256"],
        "acoustic_continuation_token_sha256": acoustic["operation_token_sha256"],
        "stream_pointer": acoustic["stream_pointer"],
        "completed_stage_token_emitted": False,
        "claim_limit": GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    audit["lineage_audit_sha256"] = hashlib.sha256(_canonical(audit)).hexdigest()
    return MappingProxyType(audit)


def finish_cuda_gm1_tilestream_rk23_acoustic(
    launch: CudaGm1TiledRk23AcousticLaunch,
    *,
    continuation: Callable[[Gm1TiledRk23AcousticLiveHandoff], Any] | None = None,
) -> FinishedGm1TiledRk23Acoustic:
    """Hash once, publish fresh stage tokens, then hand off live carriers."""

    if type(launch) is not CudaGm1TiledRk23AcousticLaunch:
        raise TypeError("launch must be exact CudaGm1TiledRk23AcousticLaunch")
    (
        pre,
        prior,
        source,
        arrays,
        metrics,
        vertical,
        stream,
        bridge,
        terrain_metrics,
    ) = _require_pre_acoustic_result(launch.pre_acoustic_result)
    requested_nrk = int(pre["requested_nrk"])
    allow_uniform_idealized_validation = (
        pre.get("uniform_idealized_validation") is True
    )
    pending = launch.receipt
    if (
        launch.requested_nrk != requested_nrk
        or launch.prior_completed_stage_token
        is not launch.pre_acoustic_result.prior_completed_stage_token
        or launch.pre_acoustic_token is not launch.pre_acoustic_result.token
        or launch.arrays is not arrays
        or launch.metrics is not metrics
        or launch.vertical is not vertical
        or launch.stream is not stream
        or launch.bridge is not bridge
        or launch.terrain_metrics is not terrain_metrics
        or pending.get("schema") != CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_SCHEMA
        or pending.get("status") != "ENQUEUED_LIVE_300X300X75_RK23_GLOBAL_ACOUSTIC"
        or pending.get("requested_nrk") != requested_nrk
        or pending.get("prior_completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or pending.get("pre_acoustic_momentum_rk_token_sha256")
        != pre.get("operation_token_sha256")
        or pending.get("schedule") != source.get("schedule")
        or pending.get("uniform_idealized_validation")
        is not allow_uniform_idealized_validation
        or pending.get("operation_token_sha256") != _operation_token(pending)
    ):
        raise ValueError("pending RK23 acoustic lineage or residency changed")
    _require_upstream_receipt(
        launch.upstream,
        requested_nrk=requested_nrk,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    launch.ready_event.synchronize()
    cupy = importlib.import_module("cupy")
    hashes: dict[str, str] = {}
    for name in GM1_RK23_ACOUSTIC_OUTPUT_KEYS:
        host = _host_output(cupy, launch.arrays[name], _PHYSICAL_SLICES[name])
        if list(host.shape) != GM1_RK23_ACOUSTIC_OUTPUT_SHAPES[name]:
            raise ValueError(f"RK23 post-sound {name} physical shape changed")
        hashes[name] = hashlib.sha256(host.tobytes(order="C")).hexdigest()
    schedule = dict(source["schedule"])
    velocity_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_STATUS,
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": pre["stream_pointer"],
        "schedule": schedule,
        "velocity_shapes": {
            target: GM1_RK23_ACOUSTIC_OUTPUT_SHAPES[source_name]
            for target, source_name in (
                ("u3d", "rru"),
                ("v3d", "rrv"),
                ("w3d", "rrw"),
            )
        },
        "velocity_sha256": {
            target: hashes[source_name]
            for target, source_name in (
                ("u3d", "rru"),
                ("v3d", "rrv"),
                ("w3d", "rrw"),
            )
        },
        "historical_rk1_token_reused": False,
        "completed_rk_stage": False,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "claim_limit": GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    velocity_receipt["operation_token_sha256"] = _operation_token(velocity_receipt)
    velocity_token = Gm1TiledRk23PostSoundVelocityToken(
        MappingProxyType(velocity_receipt)
    )
    acoustic_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "post_sound_velocity_token_sha256": velocity_receipt["operation_token_sha256"],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": pre["stream_pointer"],
        "schedule": schedule,
        "output_sha256": hashes,
        "output_shapes": dict(GM1_RK23_ACOUSTIC_OUTPUT_SHAPES),
        "stage_family_fills_deferred": True,
        "getdiv_getdivx_performed": False,
        "historical_rk1_token_reused": False,
        "completed_rk_stage": False,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "claim_limit": GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    acoustic_receipt["operation_token_sha256"] = _operation_token(acoustic_receipt)
    acoustic_token = Gm1TiledRk23AcousticContinuationToken(
        MappingProxyType(acoustic_receipt)
    )
    lineage = require_gm1_tiled_rk23_acoustic_continuation(
        launch.prior_completed_stage_token,
        launch.pre_acoustic_token,
        velocity_token,
        acoustic_token,
    )
    execution: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_STATUS,
        "requested_nrk": requested_nrk,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "post_sound_velocity_token_sha256": velocity_receipt["operation_token_sha256"],
        "acoustic_continuation_token_sha256": acoustic_receipt[
            "operation_token_sha256"
        ],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": pre["stream_pointer"],
        "schedule": schedule,
        "lineage_audit": dict(lineage),
        "output_sha256": hashes,
        "output_shapes": dict(GM1_RK23_ACOUSTIC_OUTPUT_SHAPES),
        "host_sync_performed": True,
        "continuation_callback_phase": (
            "after explicit acoustic hash sync; before caller teardown"
        ),
        "uninterrupted_pre_sync_enqueue_claimed": False,
        "stage_family_fills_deferred": True,
        "getdiv_getdivx_performed": False,
        "completed_rk_stage": False,
        "uniform_idealized_validation": allow_uniform_idealized_validation,
        "claim_limit": GM1_RK23_ACOUSTIC_CLAIM_LIMIT,
    }
    execution["operation_token_sha256"] = _operation_token(execution)
    handoff = Gm1TiledRk23AcousticLiveHandoff(
        launch.ready_event,
        launch.upstream,
        requested_nrk,
        launch.prior_completed_stage_token,
        launch.arrays,
        launch.metrics,
        launch.vertical,
        launch.stream,
        launch.bridge,
        launch.terrain_metrics,
        launch.pre_acoustic_result,
        launch.pre_acoustic_token,
        acoustic_token,
        velocity_token,
    )
    continuation_result = None if continuation is None else continuation(handoff)
    return FinishedGm1TiledRk23Acoustic(
        handoff,
        MappingProxyType(hashes),
        MappingProxyType(execution),
        continuation_result,
    )


def refuse_gm1_rk23_acoustic_without_fresh_pre_result(
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken,
    *,
    requested_nrk: int,
) -> NoReturn:
    """Validate exact prior cadence, then refuse the absent fresh producer."""

    requested_nrk = _requested_nrk(requested_nrk)
    if type(prior_completed_stage_token) is not Gm1TiledCompletedRkStageToken:
        raise TypeError("prior token must be exact Gm1TiledCompletedRkStageToken")
    require_gm1_tiled_completed_rk_stage_token(
        prior_completed_stage_token,
        completed_nrk=requested_nrk - 1,
    )
    raise RuntimeError(
        f"GM1-RK{requested_nrk}-ACOUSTIC-PREREQUISITE: exact fresh same-stage "
        "pre-acoustic momentum result is unavailable; no acoustic or completed "
        "stage token emitted"
    )


__all__ = (
    "AUTHORITY_BC_SHA256",
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_SOUND_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_CONTINUATION_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK23_ACOUSTIC_STATUS",
    "CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK23_POST_SOUND_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_BINDING_STATUS",
    "CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK23_PRE_ACOUSTIC_TOKEN_STATUS",
    "CudaGm1TiledRk23AcousticLaunch",
    "FinishedGm1TiledRk23Acoustic",
    "FinishedGm1TiledRk23PreAcousticMomentum",
    "GM1_RK23_ACOUSTIC_CLAIM_LIMIT",
    "GM1_RK23_ACOUSTIC_OUTPUT_KEYS",
    "GM1_RK23_ACOUSTIC_OUTPUT_SHAPES",
    "GM1_RK23_ACOUSTIC_SOURCE_ANCHORS",
    "GM1_RK23_ACOUSTIC_SOURCE_ORDER",
    "Gm1TiledRk23AcousticContinuationToken",
    "Gm1TiledRk23AcousticLiveHandoff",
    "Gm1TiledRk23PostSoundVelocityToken",
    "Gm1TiledRk23PreAcousticMomentumBinding",
    "Gm1TiledRk23PreAcousticMomentumToken",
    "LIVE_RK23_ACOUSTIC_ABLATIONS",
    "audit_gm1_rk23_live_acoustic_source_contract",
    "finish_cuda_gm1_tilestream_rk23_acoustic",
    "launch_cuda_gm1_tilestream_rk23_acoustic",
    "refuse_gm1_rk23_acoustic_without_fresh_pre_result",
    "require_gm1_tiled_rk23_acoustic_continuation",
)
