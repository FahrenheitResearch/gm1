"""Exact source-order aggregator for one live tile-streamed CM1 RK1 stage.

The historical boundary/TKE evidence remains available through the partial
audit below and remains explicitly uncompleted.  The completed emitter is a
separate, strict path: it requires the exact live pre-acoustic, acoustic,
post-acoustic mass-flux, theta-divergence, scalar/moist, scalar/pressure
boundary, and source-ordered TKE tokens from one RK state and CUDA stream.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from .cuda_gm1_rk1_source_ordered_tke import (
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA,
    CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS,
    Gm1Rk1SourceOrderedTkeToken,
)
from .cuda_gm1_tilestream_completed_rk_stage import (
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
    GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    Gm1TiledCompletedRkStageToken,
    require_gm1_tiled_completed_rk_stage_token,
)
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
)
from .cuda_gm1_tilestream_pre_acoustic_momentum_rk import (
    Gm1TiledPreAcousticMomentumRkToken,
    Gm1TiledRk1AcousticContinuationToken,
)
from .cuda_gm1_tilestream_rk1_pre_acoustic_lineage import (
    require_gm1_rk1_pre_acoustic_acoustic_lineage,
)
from .cuda_gm1_tilestream_rk1_scalar_pressure_boundaries import (
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS,
    Gm1TiledRk1ScalarPressureBoundaryToken,
)
from .cuda_gm1_tilestream_scalar_moist_rk1 import (
    CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA,
    CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS,
    Gm1TiledScalarMoistRk1Token,
)
from .cuda_gm1_tilestream_scalar_post_sound_getdiv import (
    CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA,
    Gm1TiledScalarThetaDivergenceToken,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)


CUDA_GM1_RK1_STAGE_PARTIAL_AUDIT_SCHEMA = (
    "gm1.rk1-stage-partial-assembly-audit/v1"
)
CUDA_GM1_RK1_STAGE_PARTIAL_AUDIT_STATUS = (
    "VERIFIED_FAIL_CLOSED_PARTIAL_RK1_STAGE_COMPONENTS"
)
GM1_RK1_STAGE_EXISTING_COMPONENT_KEYS = (
    "post_acoustic_flux",
    "scalar_moist_rk",
    "momentum_boundary",
    "scalar_pressure_boundary",
    "tke_rk",
)
GM1_RK1_STAGE_MISSING_COMPONENT_KEYS = ("pre_acoustic_momentum_rk",)
GM1_RK1_COMPLETED_STAGE_COMPONENT_KEYS = (
    "pre_acoustic_momentum_rk",
    "acoustic_continuation",
    "post_acoustic_flux",
    "theta_divergence",
    "scalar_moist_rk",
    "momentum_boundary",
    "scalar_pressure_boundary",
    "tke_rk",
)
GM1_RK1_STAGE_PARTIAL_CLAIM_LIMIT = (
    "immutable audit of the exact live selected-profile RK1 scalar/moist, "
    "post-acoustic flux, momentum-boundary, scalar/pressure-boundary, and "
    "source-ordered TKE component lineage on one CUDA stream. It is not a "
    "completed RK-stage token because the full pre-acoustic momentum-RK "
    "component is absent; it cannot admit RK2, RK3, solve3, a tiled global "
    "pressure solve, MPI/open histories, a forecast, or a five-metre product"
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


def _require_scalar_boundary(
    token: object,
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledRk1ScalarPressureBoundaryToken:
        raise TypeError(
            "scalar_boundary_token must be the exact "
            "Gm1TiledRk1ScalarPressureBoundaryToken type"
        )
    receipt = token.receipt
    cadence = receipt.get("pressure_density_cadence")
    selected = cadence.get("selected") if isinstance(cadence, Mapping) else None
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_RK1_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS
        or receipt.get("completed_nrk") != 1
        or receipt.get("source_order_component_complete") is not True
        or receipt.get("completed_broader_rk_stage") is not False
        or receipt.get("global_psolver_execution_bound") is not True
        or receipt.get("global_psolver_tile_streamable") is not False
        or receipt.get("mpi_performed") is not False
        or not isinstance(selected, Mapping)
        or dict(selected)
        != {
            "psolver": 3,
            "imoist": 1,
            "idiff": 0,
            "difforder": 6,
            "nrk": 1,
            "nrkmax": 3,
        }
        or cadence.get("recompute_performed") is not False
        or cadence.get("apmasscon_performed") is not False
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError(
            "scalar/pressure boundary identity, selected cadence, or claim changed"
        )
    for name in (
        "pressure_checkpoint_token_sha256",
        "scalar_moist_rk1_token_sha256",
        "post_acoustic_flux_token_sha256",
        "momentum_boundary_token_sha256",
        "stage_identity_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(f"scalar boundary {name}", receipt.get(name))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("scalar boundary CUDA stream pointer is invalid")
    return receipt


def _require_source_ordered_tke(
    token: object,
) -> Mapping[str, object]:
    if type(token) is not Gm1Rk1SourceOrderedTkeToken:
        raise TypeError(
            "source_ordered_tke_token must be the exact "
            "Gm1Rk1SourceOrderedTkeToken type"
        )
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_RK1_SOURCE_ORDERED_TKE_TOKEN_STATUS
        or receipt.get("completed_nrk") != 1
        or receipt.get("next_nrk") != 2
        or receipt.get("component_key") != "tke_rk"
        or receipt.get("source_order_component_complete") is not True
        or receipt.get("completed_broader_rk_stage") is not False
        or receipt.get("global_psolver_tile_streamable") is not False
        or receipt.get("mpi_performed") is not False
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("source-ordered TKE component identity or claim changed")
    for name in (
        "scalar_pressure_boundary_token_sha256",
        "tke_rk1_token_sha256",
        "tke_held_state_token_sha256",
        "output_tke3d_sha256",
        "stage_identity_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(f"source-ordered TKE {name}", receipt.get(name))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("source-ordered TKE CUDA stream pointer is invalid")
    return receipt


def _require_post_acoustic_flux(token: object) -> Mapping[str, object]:
    if type(token) is not Gm1TiledSolve2PostAcousticFluxToken:
        raise TypeError(
            "post_acoustic_flux_token must be the exact "
            "Gm1TiledSolve2PostAcousticFluxToken type"
        )
    receipt = token.receipt
    output_shapes = receipt.get("output_shapes")
    if (
        receipt.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or receipt.get("nrk") != 1
        or receipt.get("host_scatter_exact_once") is not True
        or not isinstance(output_shapes, Mapping)
        or dict(output_shapes)
        != {
            "rru": [75, 300, 301],
            "rrv": [75, 301, 300],
            "rrw": [76, 300, 300],
            "divx": [75, 300, 300],
        }
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("post-acoustic mass-flux identity or output changed")
    for name in (
        "stage_identity_sha256",
        "solve1_tke_rhs_token_sha256",
        "post_sound_velocity_token_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(f"post-acoustic flux {name}", receipt.get(name))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("post-acoustic mass-flux stream pointer is invalid")
    return receipt


def _require_theta_divergence(token: object) -> Mapping[str, object]:
    if type(token) is not Gm1TiledScalarThetaDivergenceToken:
        raise TypeError(
            "theta_divergence_token must be the exact "
            "Gm1TiledScalarThetaDivergenceToken type"
        )
    receipt = token.receipt
    output_shapes = receipt.get("output_shapes")
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_RK1_SCALAR_THETA_DIVERGENCE"
        or receipt.get("nrk") != 1
        or receipt.get("host_scatter_exact_once") is not True
        or not isinstance(output_shapes, Mapping)
        or dict(output_shapes)
        != {
            "divergence": [75, 300, 300],
            "theta_divergence_term": [75, 300, 300],
        }
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("theta-divergence identity or output changed")
    for name in (
        "stage_identity_sha256",
        "post_sound_velocity_token_sha256",
        "thermodynamic_token_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(f"theta divergence {name}", receipt.get(name))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("theta-divergence stream pointer is invalid")
    return receipt


def _require_scalar_moist_rk1(token: object) -> Mapping[str, object]:
    if type(token) is not Gm1TiledScalarMoistRk1Token:
        raise TypeError(
            "scalar_moist_rk1_token must be the exact "
            "Gm1TiledScalarMoistRk1Token type"
        )
    receipt = token.receipt
    output_shapes = receipt.get("output_shapes")
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_SCALAR_MOIST_RK1_TOKEN_STATUS
        or receipt.get("completed_nrk") != 1
        or receipt.get("host_scatter_exact_once") is not True
        or receipt.get("boundary_disposition")
        != "physical pre-bcp/bcs th3d/q3d"
        or receipt.get("padded_halos_are_outputs") is not False
        or not isinstance(output_shapes, Mapping)
        or dict(output_shapes)
        != {"th3d": [75, 300, 300], "q3d": [6, 75, 300, 300]}
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("scalar/moist RK1 identity or output changed")
    for name in (
        "stage_identity_sha256",
        "solve1_scalar_held_token_sha256",
        "post_acoustic_flux_token_sha256",
        "theta_divergence_token_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(f"scalar/moist RK1 {name}", receipt.get(name))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("scalar/moist RK1 stream pointer is invalid")
    return receipt


def emit_gm1_tiled_completed_rk1_stage(
    pre_acoustic_token: Gm1TiledPreAcousticMomentumRkToken,
    acoustic_token: Gm1TiledRk1AcousticContinuationToken,
    post_acoustic_flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_divergence_token: Gm1TiledScalarThetaDivergenceToken,
    scalar_moist_rk1_token: Gm1TiledScalarMoistRk1Token,
    scalar_boundary_token: Gm1TiledRk1ScalarPressureBoundaryToken,
    source_ordered_tke_token: Gm1Rk1SourceOrderedTkeToken,
) -> Gm1TiledCompletedRkStageToken:
    """Emit completed RK1 only for one exact end-to-end live lineage."""

    acoustic_lineage = require_gm1_rk1_pre_acoustic_acoustic_lineage(
        pre_acoustic_token, acoustic_token
    )
    pre = pre_acoustic_token.receipt
    acoustic = acoustic_token.receipt
    flux = _require_post_acoustic_flux(post_acoustic_flux_token)
    theta = _require_theta_divergence(theta_divergence_token)
    scalar = _require_scalar_moist_rk1(scalar_moist_rk1_token)
    boundary = _require_scalar_boundary(scalar_boundary_token)
    tke = _require_source_ordered_tke(source_ordered_tke_token)

    stage = pre.get("stage_identity_sha256")
    stream = pre.get("stream_pointer")
    if (
        acoustic_lineage.get("source_order_complete_through_global_acoustic")
        is not True
        or flux.get("post_sound_velocity_token_sha256")
        != acoustic.get("post_sound_velocity_token_sha256")
        or theta.get("post_sound_velocity_token_sha256")
        != acoustic.get("post_sound_velocity_token_sha256")
        or flux.get("solve1_tke_rhs_token_sha256")
        != pre.get("solve1_tke_rhs_token_sha256")
        or scalar.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or scalar.get("theta_divergence_token_sha256")
        != theta.get("operation_token_sha256")
        or boundary.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or boundary.get("scalar_moist_rk1_token_sha256")
        != scalar.get("operation_token_sha256")
        or tke.get("scalar_pressure_boundary_token_sha256")
        != boundary.get("operation_token_sha256")
        or any(
            receipt.get("stage_identity_sha256") != stage
            for receipt in (acoustic, flux, theta, scalar, boundary, tke)
        )
        or any(
            receipt.get("stream_pointer") != stream
            for receipt in (acoustic, flux, theta, scalar, boundary, tke)
        )
    ):
        raise ValueError(
            "RK1 pre/acoustic/flux/theta/scalar/boundary/TKE lineage changed"
        )

    components = {
        "pre_acoustic_momentum_rk": pre["operation_token_sha256"],
        "acoustic_continuation": acoustic["operation_token_sha256"],
        "post_acoustic_flux": flux["operation_token_sha256"],
        "theta_divergence": theta["operation_token_sha256"],
        "scalar_moist_rk": scalar["operation_token_sha256"],
        "momentum_boundary": boundary["momentum_boundary_token_sha256"],
        "scalar_pressure_boundary": boundary["operation_token_sha256"],
        "tke_rk": tke["operation_token_sha256"],
    }
    if tuple(components) != GM1_RK1_COMPLETED_STAGE_COMPONENT_KEYS:
        raise AssertionError("internal completed RK1 component order changed")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
        "completed_nrk": 1,
        "next_nrk": 2,
        "stream_pointer": stream,
        "stage_identity_sha256": stage,
        "component_operation_token_sha256": components,
        "source_order_complete": True,
        "global_psolver_tile_streamable": False,
        "five_meter_meteorological_product": False,
        "full_forecast": False,
        "claim_limit": GM1_TILESTREAM_COMPLETED_RK_STAGE_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    token = Gm1TiledCompletedRkStageToken(MappingProxyType(receipt))
    require_gm1_tiled_completed_rk_stage_token(
        token,
        completed_nrk=1,
        stream_pointer=stream,
        stage_identity_sha256=stage,
    )
    return token


def audit_gm1_rk1_stage_partial_assembly(
    scalar_boundary_token: Gm1TiledRk1ScalarPressureBoundaryToken,
    source_ordered_tke_token: Gm1Rk1SourceOrderedTkeToken,
) -> Mapping[str, object]:
    """Audit the exact components already live without emitting a stage token."""

    boundary = _require_scalar_boundary(scalar_boundary_token)
    tke = _require_source_ordered_tke(source_ordered_tke_token)
    if (
        tke.get("scalar_pressure_boundary_token_sha256")
        != boundary.get("operation_token_sha256")
        or tke.get("stage_identity_sha256")
        != boundary.get("stage_identity_sha256")
        or tke.get("stream_pointer") != boundary.get("stream_pointer")
    ):
        raise ValueError(
            "scalar-boundary and source-ordered TKE lineage or stream changed"
        )
    components = {
        "post_acoustic_flux": boundary["post_acoustic_flux_token_sha256"],
        "scalar_moist_rk": boundary["scalar_moist_rk1_token_sha256"],
        "momentum_boundary": boundary["momentum_boundary_token_sha256"],
        "scalar_pressure_boundary": boundary["operation_token_sha256"],
        "tke_rk": tke["operation_token_sha256"],
    }
    if tuple(components) != GM1_RK1_STAGE_EXISTING_COMPONENT_KEYS:
        raise AssertionError("internal RK1 component order changed")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_STAGE_PARTIAL_AUDIT_SCHEMA,
        "status": CUDA_GM1_RK1_STAGE_PARTIAL_AUDIT_STATUS,
        "candidate_completed_nrk": 1,
        "candidate_next_nrk": 2,
        "stage_identity_sha256": boundary["stage_identity_sha256"],
        "stream_pointer": boundary["stream_pointer"],
        "verified_component_operation_token_sha256": components,
        "missing_component_keys": list(GM1_RK1_STAGE_MISSING_COMPONENT_KEYS),
        "source_order_complete": False,
        "completed_stage_token_emitted": False,
        "global_psolver_execution_bound": True,
        "global_psolver_tile_streamable": False,
        "mpi_performed": False,
        "full_forecast": False,
        "five_meter_meteorological_product": False,
        "claim_limit": GM1_RK1_STAGE_PARTIAL_CLAIM_LIMIT,
    }
    receipt["partial_audit_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return MappingProxyType(receipt)


def refuse_gm1_rk1_completed_stage_without_pre_acoustic_momentum(
    scalar_boundary_token: Gm1TiledRk1ScalarPressureBoundaryToken,
    source_ordered_tke_token: Gm1Rk1SourceOrderedTkeToken,
) -> None:
    """Validate the partial lineage, then refuse the unavailable stage emitter."""

    audit_gm1_rk1_stage_partial_assembly(
        scalar_boundary_token, source_ordered_tke_token
    )
    raise RuntimeError(
        "GM1-RK1-COMPLETED-STAGE-PREREQUISITE: exact live full pre-acoustic "
        "momentum-RK token is missing; no Gm1TiledCompletedRkStageToken emitted"
    )


__all__ = (
    "CUDA_GM1_RK1_STAGE_PARTIAL_AUDIT_SCHEMA",
    "CUDA_GM1_RK1_STAGE_PARTIAL_AUDIT_STATUS",
    "GM1_RK1_STAGE_EXISTING_COMPONENT_KEYS",
    "GM1_RK1_STAGE_MISSING_COMPONENT_KEYS",
    "GM1_RK1_STAGE_PARTIAL_CLAIM_LIMIT",
    "GM1_RK1_COMPLETED_STAGE_COMPONENT_KEYS",
    "audit_gm1_rk1_stage_partial_assembly",
    "emit_gm1_tiled_completed_rk1_stage",
    "refuse_gm1_rk1_completed_stage_without_pre_acoustic_momentum",
)
