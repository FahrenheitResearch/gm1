"""Typed live RK1 psolver=3 continuation for the 50 m GM1 child.

The global acoustic solve is not tile local.  This boundary consumes the
exact resident pre-acoustic momentum token, enqueues the already sealed
300x300x75 stretched/open ``sound`` operator on the same CUDA stream, and
publishes a typed continuation at the source boundary immediately before
``getdiv``/``getdivx``.  It deliberately defers the solve2 stage-family
boundary fills, which occur only after scalar and pressure/density work.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from .cuda_gm1_child_psolver3_open import (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_BOUNDARY_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
    CUDA_GM1_CHILD_PSOLVER3_OPEN_STAGE_SCHEMA,
    CudaGm1ChildPsolver3OpenLaunch,
    launch_cuda_gm1_child_psolver3_open_stage,
)
from .cuda_gm1_child_stretched_metrics import Gm1ChildStretchedMetrics
from .cuda_gm1_child_terrain import CudaGm1ChildTerrainMetricsLaunch
from .cuda_gm1_tilestream_solve1_full_held import (
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS,
    CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA,
    Gm1TiledSolve1FullHeldToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
)
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from .cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
    GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS,
    require_gm1_rk1_pre_acoustic_acoustic_lineage,
)
from .cuda_moist_stage_bridge import CudaMoistStageBridgeState
from .cuda_momentum_advection5 import CudaMomentumAdvection5Launch
from .cuda_psolver3_terrain_open import TerrainOpenSoundAblations


CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_SCHEMA = "gm1.cuda-tilestream-rk1-global-acoustic/v1"
CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_STATUS = "ADMITTED_LIVE_300X300X75_RK1_GLOBAL_ACOUSTIC"
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
_TOKEN_MODULE = "cm1core.cuda_gm1_tilestream_pre_acoustic_momentum_rk"

GM1_RK1_LIVE_ACOUSTIC_SOURCE_ORDER = (
    "consume exact live pre-acoustic RK1 momentum token after solve2.F:314-1051",
    "solve2.F:1082 sets get_time_avg false; psolver=3 selects sound at 1119-1137",
    "sound.F:78-105 schedules RK1 nloop=2 and dts=dt/(nloop*3)",
    "sound.F:107-248 builds terrain coefficients and seeds ppd/velocity averages",
    "sound.F:253-837 consumes four-open tendencies, advances u/v/w/pressure, fills ppd between small steps, and averages velocities",
    "solve2.F:1258-1271 produces radiation history after sound returns",
    "publish exact live acoustic continuation before solve2.F:1277 getdiv and 1303 getdivx",
    "defer solve2.F:1662-1706 momentum/scalar/pressure family fills to the post-scalar boundary owner",
)
GM1_RK1_LIVE_ACOUSTIC_SOURCE_ANCHORS = (
    "src/solve2.F:1053-1173",
    "src/sound.F:78-105",
    "src/sound.F:107-248",
    "src/sound.F:253-837",
    "src/solve2.F:1258-1307",
    "src/solve2.F:1310-1660",
    "src/solve2.F:1662-1706",
)
GM1_RK1_LIVE_ACOUSTIC_OUTPUT_KEYS = (
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
GM1_RK1_LIVE_ACOUSTIC_OUTPUT_SHAPES = MappingProxyType(
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
LIVE_ACOUSTIC_ABLATIONS = TerrainOpenSoundAblations(stage_family_fills=False)
GM1_RK1_LIVE_ACOUSTIC_CLAIM_LIMIT = (
    "one source-ordered real 300x300x75, 50 m, dt=0.5 s RK1 global psolver=3 "
    "acoustic continuation on the exact pre-acoustic CUDA stream; getdiv, "
    "getdivx, scalar/moisture/TKE integration, stage-family fills, completed "
    "RK stage, RK2/RK3, solve3, forecast, five-metre scaling, and products "
    "remain outside this token"
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


def _runtime_types() -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any]]:
    try:
        module = importlib.import_module(_TOKEN_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _TOKEN_MODULE:
            raise
        raise RuntimeError(
            "GM1-RK1-ACOUSTIC: exact pre-acoustic token module has not landed"
        ) from error
    pre_type = getattr(module, "Gm1TiledPreAcousticMomentumRkToken", None)
    acoustic_type = getattr(module, "Gm1TiledRk1AcousticContinuationToken", None)
    binding_type = getattr(module, "Gm1TiledPreAcousticMomentumRkBinding", None)
    launch_type = getattr(module, "CudaGm1TiledPreAcousticMomentumRkLaunch", None)
    finished_type = getattr(module, "FinishedGm1TiledPreAcousticMomentumRk", None)
    if not all(
        isinstance(value, type)
        for value in (
            pre_type,
            acoustic_type,
            binding_type,
            launch_type,
            finished_type,
        )
    ):
        raise RuntimeError(
            "GM1-RK1-ACOUSTIC: exact token and live result classes are unavailable"
        )
    return pre_type, acoustic_type, binding_type, launch_type, finished_type


def _token_types() -> tuple[type[Any], type[Any]]:
    pre_type, acoustic_type, _, _, _ = _runtime_types()
    return pre_type, acoustic_type


def _require_pre_acoustic_token(token: object) -> Mapping[str, object]:
    pre_type, _ = _token_types()
    if type(token) is not pre_type:
        raise TypeError(
            "pre_acoustic_token must be the exact Gm1TiledPreAcousticMomentumRkToken"
        )
    receipt = getattr(token, "receipt", None)
    if not isinstance(receipt, Mapping):
        raise TypeError("pre-acoustic token does not expose an immutable receipt")
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("pre-acoustic token identity or RK cadence changed")
    for name in (
        "operation_token_sha256",
        "stage_identity_sha256",
        "solve1_full_held_token_sha256",
    ):
        _lower_sha(f"pre-acoustic {name}", receipt.get(name))
    stream_pointer = receipt.get("stream_pointer")
    if (
        isinstance(stream_pointer, bool)
        or not isinstance(stream_pointer, int)
        or stream_pointer <= 0
    ):
        raise ValueError("pre-acoustic CUDA stream pointer is invalid")
    output_sha = receipt.get("output_sha256")
    output_shapes = receipt.get("output_shapes")
    if (
        not isinstance(output_sha, Mapping)
        or tuple(output_sha) != GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS
        or not isinstance(output_shapes, Mapping)
        or tuple(output_shapes) != GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS
    ):
        raise ValueError("pre-acoustic output inventory changed")
    expected_shapes = {
        "uten": [75, 300, 301],
        "vten": [75, 301, 300],
        "wten": [76, 300, 300],
        "rru": [75, 300, 301],
        "rrv": [75, 301, 300],
        "rrw": [76, 300, 300],
        "divx": [75, 300, 300],
    }
    if dict(output_shapes) != expected_shapes:
        raise ValueError("pre-acoustic output shapes changed")
    for name in GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS:
        _lower_sha(f"pre-acoustic output {name}", output_sha.get(name))
    return receipt


def _require_pre_acoustic_result(
    result: object,
) -> tuple[
    Mapping[str, object],
    object,
    Mapping[str, Any],
    Mapping[str, Any],
    Any,
    Any,
    Any,
    Any,
    Gm1TiledSolve1FullHeldToken,
    Gm1TiledSolve1TkeRhsToken,
]:
    """Bind the exact same-residency result; token-plus-foreign arrays refuse."""

    _, _, binding_type, _, finished_type = _runtime_types()
    if type(result) is not finished_type:
        raise TypeError(
            "pre_acoustic_result must be the exact "
            "FinishedGm1TiledPreAcousticMomentumRk"
        )
    binding = getattr(result, "binding", None)
    if type(binding) is not binding_type:
        raise TypeError("pre-acoustic result no longer retains its exact binding")
    upstream = getattr(result, "upstream", None)
    if type(upstream) is not CudaMomentumAdvection5Launch:
        raise TypeError("pre-acoustic result lost its exact momentum launch")
    token = getattr(result, "token", None)
    receipt = _require_pre_acoustic_token(token)
    full_held_token = getattr(result, "full_held_token", None)
    if type(full_held_token) is not Gm1TiledSolve1FullHeldToken:
        raise TypeError("pre-acoustic result lost its exact full-held token")
    held = full_held_token.receipt
    if (
        not isinstance(held, Mapping)
        or held.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_SCHEMA
        or held.get("status") != CUDA_GM1_TILESTREAM_SOLVE1_FULL_HELD_TOKEN_STATUS
        or held.get("completed_nrk") != 0
        or held.get("operation_token_sha256") != _operation_token(held)
        or held.get("operation_token_sha256")
        != receipt.get("solve1_full_held_token_sha256")
        or held.get("stream_pointer") != receipt.get("stream_pointer")
    ):
        raise ValueError("pre-acoustic full-held lineage changed")
    _lower_sha(
        "pre-acoustic solve1 TKE RHS lineage",
        held.get("solve1_tke_operation_token_sha256"),
    )
    solve1_tke_token = getattr(result, "solve1_tke_token", None)
    if type(solve1_tke_token) is not Gm1TiledSolve1TkeRhsToken:
        raise TypeError("pre-acoustic result lost its exact solve1 TKE RHS token")
    solve1_tke = solve1_tke_token.receipt
    if (
        not isinstance(solve1_tke, Mapping)
        or solve1_tke.get("schema") != CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA
        or solve1_tke.get("status") != "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS"
        or solve1_tke.get("completed_nrk") != 0
        or solve1_tke.get("stream_pointer") != receipt.get("stream_pointer")
        or solve1_tke.get("output_tketen_shape") != [76, 300, 300]
        or solve1_tke.get("operation_token_sha256") != _operation_token(solve1_tke)
        or solve1_tke.get("operation_token_sha256")
        != held.get("solve1_tke_operation_token_sha256")
    ):
        raise ValueError("pre-acoustic solve1 TKE RHS lineage changed")
    _lower_sha(
        "pre-acoustic solve1 TKE RHS output",
        solve1_tke.get("output_tketen_sha256"),
    )
    arrays = getattr(result, "arrays", None)
    metrics = getattr(result, "metrics", None)
    vertical = getattr(result, "vertical", None)
    stream = getattr(result, "stream", None)
    bridge = getattr(result, "bridge", None)
    terrain_metrics = getattr(result, "terrain_metrics", None)
    output_sha256 = getattr(result, "output_sha256", None)
    if not isinstance(arrays, Mapping) or not isinstance(metrics, Mapping):
        raise TypeError("pre-acoustic result lost its resident arrays or metrics")
    if type(vertical) is not Gm1ChildStretchedMetrics:
        raise TypeError("pre-acoustic result lost its exact stretched vertical binding")
    if type(bridge) is not CudaMoistStageBridgeState:
        raise TypeError("pre-acoustic result lost its exact moist-stage binding")
    if type(terrain_metrics) is not CudaGm1ChildTerrainMetricsLaunch:
        raise TypeError("pre-acoustic result lost its exact terrain-metric launch")
    if stream is None or not hasattr(stream, "ptr"):
        raise TypeError("pre-acoustic result lost its explicit CUDA stream")
    if int(stream.ptr) != receipt.get("stream_pointer"):
        raise ValueError("pre-acoustic result stream differs from its token")
    if (
        not isinstance(output_sha256, Mapping)
        or tuple(output_sha256) != GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS
        or dict(output_sha256) != dict(receipt["output_sha256"])
    ):
        raise ValueError("pre-acoustic result output hashes differ from its token")
    if (
        binding.full_held_token is not full_held_token
        or binding.solve1_tke_token is not solve1_tke_token
        or binding.arrays is not arrays
        or binding.metrics is not metrics
        or binding.terrain_metrics is not terrain_metrics
        or binding.vertical is not vertical
        or binding.stream is not stream
        or binding.bridge is not bridge
        or binding.receipt.get("binding_sha256") != receipt.get("binding_sha256")
    ):
        raise ValueError("finished pre-acoustic result differs from its exact binding")
    return (
        receipt,
        token,
        arrays,
        metrics,
        vertical,
        stream,
        bridge,
        terrain_metrics,
        full_held_token,
        solve1_tke_token,
    )


def audit_gm1_rk1_live_acoustic_source_contract() -> Mapping[str, object]:
    """Return the selected source/cadence boundary without issuing a token."""

    audit: dict[str, object] = {
        "schema": "gm1.rk1-live-acoustic-source-audit/v1",
        "status": "SOURCE_AUDITED_RUNTIME_TOKEN_REQUIRED",
        "selected_profile": {
            "domain_zyx": [75, 300, 300],
            "dx_m": 50.0,
            "dy_m": 50.0,
            "dt_seconds": 0.5,
            "nrk": 1,
            "nrkmax": 3,
            "psolver": 3,
            "nsound": 6,
            "nloop": 2,
            "dts_f32": 0.0833333358168602,
            "terrain": True,
            "stretch_z": 1,
            "lateral_boundaries": [2, 2, 2, 2],
            "mpi": False,
        },
        "source_order": list(GM1_RK1_LIVE_ACOUSTIC_SOURCE_ORDER),
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "sound_sha256": AUTHORITY_SOUND_SHA256,
            "bc_sha256": AUTHORITY_BC_SHA256,
            "anchors": list(GM1_RK1_LIVE_ACOUSTIC_SOURCE_ANCHORS),
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
        "continuation_boundary": "after radiation history; before getdiv/getdivx",
        "completed_stage_token_emitted": False,
        "claim_limit": GM1_RK1_LIVE_ACOUSTIC_CLAIM_LIMIT,
    }
    audit["source_contract_sha256"] = hashlib.sha256(_canonical(audit)).hexdigest()
    return MappingProxyType(audit)


@dataclass(frozen=True, slots=True)
class CudaGm1TiledRk1AcousticLaunch:
    ready_event: Any
    upstream: CudaGm1ChildPsolver3OpenLaunch
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    vertical: Any
    stream: Any
    bridge: Any
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    full_held_token: Gm1TiledSolve1FullHeldToken
    solve1_tke_token: Gm1TiledSolve1TkeRhsToken
    pre_acoustic_result: Any
    pre_acoustic_token: Any
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk1AcousticLiveHandoff:
    """Same-residency handoff after the explicit first-gate hash sync."""

    ready_event: Any
    stage: CudaGm1ChildPsolver3OpenLaunch
    arrays: Mapping[str, Any]
    metrics: Mapping[str, Any]
    vertical: Any
    stream: Any
    bridge: Any
    terrain_metrics: CudaGm1ChildTerrainMetricsLaunch
    full_held_token: Gm1TiledSolve1FullHeldToken
    solve1_tke_token: Gm1TiledSolve1TkeRhsToken
    pre_acoustic_result: Any
    pre_acoustic_token: Any
    acoustic_token: Any
    post_sound_velocity_token: Gm1PostSoundVelocityToken


@dataclass(frozen=True, slots=True)
class FinishedGm1TiledRk1Acoustic:
    handoff: Gm1TiledRk1AcousticLiveHandoff
    host_velocity: Mapping[str, Any]
    output_sha256: Mapping[str, str]
    execution_receipt: Mapping[str, object]
    continuation_result: Any


def _require_upstream_receipt(
    launch: CudaGm1ChildPsolver3OpenLaunch,
    *,
    allow_uniform_idealized_validation: bool = False,
) -> None:
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be boolean")
    receipt = launch.receipt
    domain = receipt.get("domain")
    cadence = receipt.get("cadence")
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
        or cadence
        != [
            {
                "nrk": 1,
                "nloop": 2,
                "dts_f32": 0.0833333358168602,
                "open_consumptions": 2,
                "pressure_fills": 1,
                "radiation_productions": 1,
            }
        ]
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
        raise ValueError("global psolver3 source, domain, or RK1 cadence changed")


def launch_cuda_gm1_tilestream_rk1_acoustic(
    pre_acoustic_result: object,
    *,
    compiled: Any | None = None,
    open_compiled: Any | None = None,
    lateral_compiled: Any | None = None,
) -> CudaGm1TiledRk1AcousticLaunch:
    """Enqueue the selected global acoustic operator on the pre-token stream."""

    (
        pre,
        pre_acoustic_token,
        arrays,
        metrics,
        vertical,
        stream,
        bridge,
        terrain_metrics,
        full_held_token,
        solve1_tke_token,
    ) = _require_pre_acoustic_result(pre_acoustic_result)
    uniform_idealized_validation = (
        pre.get("uniform_idealized_validation") is True
    )
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
        ablations=LIVE_ACOUSTIC_ABLATIONS,
        stream=stream,
        compiled=compiled,
        open_compiled=open_compiled,
        lateral_compiled=lateral_compiled,
        allow_compact_validation=uniform_idealized_validation,
        allow_uniform_idealized_validation=uniform_idealized_validation,
    )
    _require_upstream_receipt(
        upstream,
        allow_uniform_idealized_validation=uniform_idealized_validation,
    )
    pending: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_SCHEMA,
        "status": "ENQUEUED_LIVE_300X300X75_RK1_GLOBAL_ACOUSTIC",
        "nrk": 1,
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "solve1_full_held_token_sha256": pre["solve1_full_held_token_sha256"],
        "solve1_tke_rhs_token_sha256": solve1_tke_token.receipt[
            "operation_token_sha256"
        ],
        "stream_pointer": int(stream.ptr),
        "psolver_receipt_sha256": hashlib.sha256(
            _canonical(upstream.receipt)
        ).hexdigest(),
        "source_contract_sha256": audit_gm1_rk1_live_acoustic_source_contract()[
            "source_contract_sha256"
        ],
        "stage_family_fills_deferred": True,
        "getdiv_getdivx_performed": False,
        "host_sync_performed": False,
        "uniform_idealized_validation": uniform_idealized_validation,
        "claim_limit": GM1_RK1_LIVE_ACOUSTIC_CLAIM_LIMIT,
    }
    pending["operation_token_sha256"] = _operation_token(pending)
    return CudaGm1TiledRk1AcousticLaunch(
        upstream.ready_event,
        upstream,
        arrays,
        metrics,
        vertical,
        stream,
        bridge,
        terrain_metrics,
        full_held_token,
        solve1_tke_token,
        pre_acoustic_result,
        pre_acoustic_token,
        MappingProxyType(pending),
    )


def _host_output(cupy: Any, value: Any, slices: tuple[slice, ...]) -> Any:
    np = importlib.import_module("numpy")
    host = np.ascontiguousarray(cupy.asnumpy(value[slices]), dtype=np.float32)
    if not bool(np.all(np.isfinite(host))):
        raise ValueError("post-sound acoustic output contains non-finite words")
    return host


def finish_cuda_gm1_tilestream_rk1_acoustic(
    launch: CudaGm1TiledRk1AcousticLaunch,
    *,
    continuation: Callable[[Gm1TiledRk1AcousticLiveHandoff], Any] | None = None,
) -> FinishedGm1TiledRk1Acoustic:
    """Hash once, then hand still-resident carriers to the continuation.

    The first complete-timestep gate intentionally keeps the historical D2H
    token semantics.  It therefore synchronizes the acoustic ready event
    before invoking ``continuation``.  The callback still receives the same
    live arrays and CUDA stream before this function returns; this is not a
    claim of uninterrupted pre-sync enqueue.
    """

    if not isinstance(launch, CudaGm1TiledRk1AcousticLaunch):
        raise TypeError("launch must be the exact live RK1 acoustic launch")
    pending = launch.receipt
    if (
        pending.get("schema") != CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_SCHEMA
        or pending.get("status") != "ENQUEUED_LIVE_300X300X75_RK1_GLOBAL_ACOUSTIC"
        or pending.get("nrk") != 1
        or pending.get("operation_token_sha256") != _operation_token(pending)
    ):
        raise ValueError("pending live acoustic identity changed")
    (
        pre,
        pre_token,
        arrays,
        metrics,
        vertical,
        stream,
        bridge,
        terrain_metrics,
        full_held_token,
        solve1_tke_token,
    ) = _require_pre_acoustic_result(launch.pre_acoustic_result)
    if (
        launch.pre_acoustic_token is not pre_token
        or launch.arrays is not arrays
        or launch.metrics is not metrics
        or launch.vertical is not vertical
        or launch.stream is not stream
        or launch.bridge is not bridge
        or launch.terrain_metrics is not terrain_metrics
        or launch.full_held_token is not full_held_token
        or launch.solve1_tke_token is not solve1_tke_token
        or pending.get("pre_acoustic_momentum_rk_token_sha256")
        != pre.get("operation_token_sha256")
        or pending.get("solve1_tke_rhs_token_sha256")
        != solve1_tke_token.receipt.get("operation_token_sha256")
        or pending.get("stage_identity_sha256") != pre.get("stage_identity_sha256")
        or pending.get("stream_pointer") != pre.get("stream_pointer")
    ):
        raise ValueError("pending acoustic lineage changed before completion")
    uniform_idealized_validation = (
        pre.get("uniform_idealized_validation") is True
    )
    if (
        pending.get("uniform_idealized_validation")
        is not uniform_idealized_validation
    ):
        raise ValueError("pending acoustic idealized disposition changed")
    _require_upstream_receipt(
        launch.upstream,
        allow_uniform_idealized_validation=uniform_idealized_validation,
    )
    launch.ready_event.synchronize()
    cupy = importlib.import_module("cupy")
    slices = {
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
    host: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for name in GM1_RK1_LIVE_ACOUSTIC_OUTPUT_KEYS:
        value = _host_output(cupy, launch.arrays[name], slices[name])
        if list(value.shape) != GM1_RK1_LIVE_ACOUSTIC_OUTPUT_SHAPES[name]:
            raise ValueError(f"post-sound {name} physical shape changed")
        host[name] = value
        hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()

    velocity_receipt: dict[str, object] = {
        "schema": CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
        "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
        "nrk": 1,
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "solve1_full_held_token_sha256": pre["solve1_full_held_token_sha256"],
        "solve1_tke_rhs_token_sha256": launch.solve1_tke_token.receipt[
            "operation_token_sha256"
        ],
        "psolver_receipt_sha256": pending["psolver_receipt_sha256"],
        "stream_pointer": pre["stream_pointer"],
        "host_checkpoint_transport": True,
        "velocity_shapes": {
            target: GM1_RK1_LIVE_ACOUSTIC_OUTPUT_SHAPES[source]
            for target, source in (
                ("u3d", "rru"),
                ("v3d", "rrv"),
                ("w3d", "rrw"),
            )
        },
        "velocity_sha256": {
            target: hashes[source]
            for target, source in (
                ("u3d", "rru"),
                ("v3d", "rrv"),
                ("w3d", "rrw"),
            )
        },
        "sound_output_semantics": "small-step-averaged velocities before getdiv/getdivx",
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "claim_limit": GM1_RK1_LIVE_ACOUSTIC_CLAIM_LIMIT,
    }
    velocity_receipt["operation_token_sha256"] = _operation_token(velocity_receipt)
    velocity_token = Gm1PostSoundVelocityToken(MappingProxyType(velocity_receipt))

    _, acoustic_type = _token_types()
    continuation_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS,
        "nrk": 1,
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "post_sound_velocity_token_sha256": velocity_receipt["operation_token_sha256"],
        "stream_pointer": pre["stream_pointer"],
    }
    continuation_receipt["operation_token_sha256"] = _operation_token(
        continuation_receipt
    )
    acoustic_token = acoustic_type(MappingProxyType(continuation_receipt))
    if getattr(acoustic_token, "receipt", None) != continuation_receipt:
        raise TypeError("acoustic token class did not preserve the exact receipt")
    lineage_audit = require_gm1_rk1_pre_acoustic_acoustic_lineage(
        launch.pre_acoustic_token, acoustic_token
    )

    execution: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_STATUS,
        "nrk": 1,
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "solve1_tke_rhs_token_sha256": launch.solve1_tke_token.receipt[
            "operation_token_sha256"
        ],
        "post_sound_velocity_token_sha256": velocity_receipt["operation_token_sha256"],
        "acoustic_continuation_token_sha256": continuation_receipt[
            "operation_token_sha256"
        ],
        "stream_pointer": pre["stream_pointer"],
        "psolver_receipt_sha256": pending["psolver_receipt_sha256"],
        "source_contract_sha256": pending["source_contract_sha256"],
        "pre_acoustic_acoustic_lineage_audit": dict(lineage_audit),
        "output_sha256": hashes,
        "output_shapes": dict(GM1_RK1_LIVE_ACOUSTIC_OUTPUT_SHAPES),
        "radiation_history_produced": True,
        "stage_family_fills_deferred": True,
        "getdiv_getdivx_performed": False,
        "host_sync_performed": True,
        "continuation_callback_phase": (
            "after explicit acoustic hash sync; before caller teardown"
        ),
        "uninterrupted_pre_sync_enqueue_claimed": False,
        "completed_rk_stage": False,
        "claim_limit": GM1_RK1_LIVE_ACOUSTIC_CLAIM_LIMIT,
    }
    execution["operation_token_sha256"] = _operation_token(execution)
    handoff = Gm1TiledRk1AcousticLiveHandoff(
        launch.ready_event,
        launch.upstream,
        launch.arrays,
        launch.metrics,
        launch.vertical,
        launch.stream,
        launch.bridge,
        launch.terrain_metrics,
        launch.full_held_token,
        launch.solve1_tke_token,
        launch.pre_acoustic_result,
        launch.pre_acoustic_token,
        acoustic_token,
        velocity_token,
    )
    continuation_result = None if continuation is None else continuation(handoff)
    return FinishedGm1TiledRk1Acoustic(
        handoff,
        MappingProxyType(
            {
                target: host[source]
                for target, source in (
                    ("u3d", "rru"),
                    ("v3d", "rrv"),
                    ("w3d", "rrw"),
                )
            }
        ),
        MappingProxyType(hashes),
        MappingProxyType(execution),
        continuation_result,
    )


__all__ = (
    "CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_STATUS",
    "CudaGm1TiledRk1AcousticLaunch",
    "FinishedGm1TiledRk1Acoustic",
    "Gm1TiledRk1AcousticLiveHandoff",
    "GM1_RK1_LIVE_ACOUSTIC_OUTPUT_KEYS",
    "GM1_RK1_LIVE_ACOUSTIC_OUTPUT_SHAPES",
    "GM1_RK1_LIVE_ACOUSTIC_SOURCE_ANCHORS",
    "GM1_RK1_LIVE_ACOUSTIC_SOURCE_ORDER",
    "LIVE_ACOUSTIC_ABLATIONS",
    "audit_gm1_rk1_live_acoustic_source_contract",
    "finish_cuda_gm1_tilestream_rk1_acoustic",
    "launch_cuda_gm1_tilestream_rk1_acoustic",
)
