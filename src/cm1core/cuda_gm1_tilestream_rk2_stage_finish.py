"""Exact source-ordered tail and completed-stage emitter for live GM1 RK2.

This module owns only the typed publication boundary after an admitted live
RK2 acoustic handoff.  Numerical execution stays in the same-residency runner;
the functions here hash-bind its fresh post-acoustic flux, theta divergence,
scalar/moisture update, late boundary publications, and tiled SGS-TKE result.
Only the complete chain may emit ``Gm1TiledCompletedRkStageToken`` for RK2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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
from .cuda_gm1_tilestream_rk23_acoustic import (
    Gm1TiledRk23AcousticLiveHandoff,
    require_gm1_tiled_rk23_acoustic_continuation,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from .cuda_gm1_tilestream_tke_rk23 import (
    Gm1TiledTkeRk23Token,
    require_gm1_tiled_tke_rk23_token,
)


CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_SCHEMA = (
    "gm1.tilestream-rk2-theta-divergence-token/v1"
)
CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK2_THETA_DIVERGENCE"
)
CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_SCHEMA = (
    "gm1.tilestream-scalar-moist-rk2-token/v1"
)
CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_SCALAR_MOIST_RK2"
)
CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_SCHEMA = (
    "gm1.tilestream-rk2-momentum-boundary-token/v1"
)
CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK2_MOMENTUM_BOUNDARIES"
)
CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA = (
    "gm1.tilestream-rk2-scalar-pressure-boundary-token/v1"
)
CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARIES"
)

GM1_RK2_COMPONENT_KEYS = (
    "pre_acoustic_momentum_rk",
    "acoustic_continuation",
    "post_acoustic_flux",
    "theta_divergence",
    "scalar_moist_rk",
    "momentum_boundary",
    "scalar_pressure_boundary",
    "tke_rk",
)
_FLUX_SHAPES = MappingProxyType(
    {
        "rru": (75, 300, 301),
        "rrv": (75, 301, 300),
        "rrw": (76, 300, 300),
        "divx": (75, 300, 300),
    }
)
_THETA_SHAPES = MappingProxyType(
    {
        "divergence": (75, 300, 300),
        "theta_divergence_term": (75, 300, 300),
    }
)
_SCALAR_SHAPES = MappingProxyType(
    {"th3d": (75, 300, 300), "q3d": (6, 75, 300, 300)}
)
_MOMENTUM_SHAPES = MappingProxyType(
    {
        "u3d": (75, 300, 301),
        "v3d": (75, 301, 300),
        "w3d": (76, 300, 300),
    }
)
_BOUNDARY_SHAPES = MappingProxyType(
    {
        "rho": (75, 302, 302),
        "pp3d": (75, 302, 302),
        "th3d": (75, 306, 306),
        "q3d": (6, 75, 306, 306),
    }
)
GM1_RK2_STAGE_FINISH_CLAIM_LIMIT = (
    "one exact live 300x300x75, 50 m CM1 r21.1 RK2 tail after the admitted "
    "same-stage global acoustic solve: stretched getdiv/getdivx, scalar/moist "
    "RK2, deferred momentum/scalar/pressure boundaries, and tiled SGS-TKE "
    "RK2 on one CUDA stream. It does not tile the global pressure solve, "
    "perform MPI, enter RK3 or solve3, create a forecast, or establish a "
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


def _host_hashes(
    values: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]], *, label: str
) -> Mapping[str, str]:
    import numpy as np

    if not isinstance(values, Mapping) or tuple(values) != tuple(shapes):
        raise ValueError(f"{label} output inventory or order changed")
    hashes: dict[str, str] = {}
    for name, shape in shapes.items():
        value = values[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype("<f4")
            or value.shape != shape
            or not value.flags.c_contiguous
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"{label} {name} must be finite host C-<f4 {shape}")
        hashes[name] = _sha(value)
    return MappingProxyType(hashes)


@dataclass(frozen=True, slots=True)
class Gm1TiledRk2ThetaDivergenceToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledScalarMoistRk2Token:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk2MomentumBoundaryToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk2ScalarPressureBoundaryToken:
    receipt: Mapping[str, object]


def _require_acoustic_handoff(
    handoff: object,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if type(handoff) is not Gm1TiledRk23AcousticLiveHandoff:
        raise TypeError("handoff must be exact Gm1TiledRk23AcousticLiveHandoff")
    if handoff.requested_nrk != 2:
        raise ValueError("RK2 stage finisher accepts only requested_nrk=2")
    prior = require_gm1_tiled_completed_rk_stage_token(
        handoff.prior_completed_stage_token,
        completed_nrk=1,
        stream_pointer=int(handoff.stream.ptr),
    )
    require_gm1_tiled_rk23_acoustic_continuation(
        handoff.prior_completed_stage_token,
        handoff.pre_acoustic_token,
        handoff.post_sound_velocity_token,
        handoff.acoustic_token,
    )
    pre = handoff.pre_acoustic_token.receipt
    acoustic = handoff.acoustic_token.receipt
    if (
        handoff.pre_acoustic_result.token is not handoff.pre_acoustic_token
        or handoff.pre_acoustic_result.prior_completed_stage_token
        is not handoff.prior_completed_stage_token
        or handoff.pre_acoustic_result.arrays is not handoff.arrays
        or handoff.pre_acoustic_result.metrics is not handoff.metrics
        or handoff.pre_acoustic_result.vertical is not handoff.vertical
        or handoff.pre_acoustic_result.stream is not handoff.stream
        or handoff.pre_acoustic_result.bridge is not handoff.bridge
        or handoff.pre_acoustic_result.terrain_metrics is not handoff.terrain_metrics
        or acoustic.get("prior_completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or acoustic.get("pre_acoustic_momentum_rk_token_sha256")
        != pre.get("operation_token_sha256")
        or acoustic.get("stage_family_fills_deferred") is not True
        or acoustic.get("getdiv_getdivx_performed") is not False
    ):
        raise ValueError("RK2 acoustic handoff lineage or residency changed")
    return prior, pre, acoustic


def publish_gm1_tiled_rk2_post_acoustic_tokens(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    flux_outputs: Mapping[str, Any],
    theta_outputs: Mapping[str, Any],
) -> tuple[Gm1TiledSolve2PostAcousticFluxToken, Gm1TiledRk2ThetaDivergenceToken]:
    """Hash-bind one drained same-residency getdiv/getdivx publication."""

    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    flux_hashes = _host_hashes(flux_outputs, _FLUX_SHAPES, label="RK2 flux")
    theta_hashes = _host_hashes(theta_outputs, _THETA_SHAPES, label="RK2 theta")
    stream_pointer = int(handoff.stream.ptr)
    stage_identity = _lower_sha(
        "RK2 stage identity", pre.get("stage_identity_sha256")
    )
    flux_receipt: dict[str, object] = {
        "schema": CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
        "nrk": 2,
        "completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "post_sound_velocity_token_sha256": acoustic[
            "post_sound_velocity_token_sha256"
        ],
        "acoustic_continuation_token_sha256": acoustic[
            "operation_token_sha256"
        ],
        "stage_identity_sha256": stage_identity,
        "stream_pointer": stream_pointer,
        "host_scatter_exact_once": True,
        "output_shapes": {name: list(shape) for name, shape in _FLUX_SHAPES.items()},
        "output_sha256": dict(flux_hashes),
        "source_order": [
            "global psolver=3 RK2 acoustic continuation",
            "stretched getdiv publishes divergence and theta-divergence term",
            "stretched getdivx converts current averaged velocity to mass flux",
            "stretched getdivx rebuilds divx on the same live stream",
        ],
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK2_STAGE_FINISH_CLAIM_LIMIT,
    }
    flux_receipt["operation_token_sha256"] = _operation_token(flux_receipt)
    flux_token = Gm1TiledSolve2PostAcousticFluxToken(
        MappingProxyType(flux_receipt)
    )
    theta_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_STATUS,
        "completed_nrk": 2,
        "prior_completed_rk_stage_token_sha256": prior[
            "operation_token_sha256"
        ],
        "acoustic_continuation_token_sha256": acoustic[
            "operation_token_sha256"
        ],
        "post_acoustic_flux_token_sha256": flux_receipt[
            "operation_token_sha256"
        ],
        "stage_identity_sha256": stage_identity,
        "stream_pointer": stream_pointer,
        "output_shapes": {
            name: list(shape) for name, shape in _THETA_SHAPES.items()
        },
        "output_sha256": dict(theta_hashes),
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK2_STAGE_FINISH_CLAIM_LIMIT,
    }
    theta_receipt["operation_token_sha256"] = _operation_token(theta_receipt)
    return flux_token, Gm1TiledRk2ThetaDivergenceToken(
        MappingProxyType(theta_receipt)
    )


def publish_gm1_tiled_scalar_moist_rk2_token(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledRk2ThetaDivergenceToken,
    outputs: Mapping[str, Any],
) -> Gm1TiledScalarMoistRk2Token:
    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    flux = _require_flux(flux_token, prior=prior, acoustic=acoustic)
    theta = _require_theta(theta_token, flux=flux, acoustic=acoustic)
    hashes = _host_hashes(outputs, _SCALAR_SHAPES, label="RK2 scalar/moist")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_STATUS,
        "completed_nrk": 2,
        "prior_completed_rk_stage_token_sha256": prior[
            "operation_token_sha256"
        ],
        "pre_acoustic_momentum_rk_token_sha256": pre[
            "operation_token_sha256"
        ],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "theta_divergence_token_sha256": theta["operation_token_sha256"],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": int(handoff.stream.ptr),
        "output_shapes": {
            name: list(shape) for name, shape in _SCALAR_SHAPES.items()
        },
        "output_sha256": dict(hashes),
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK2_STAGE_FINISH_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledScalarMoistRk2Token(MappingProxyType(receipt))


def publish_gm1_tiled_rk2_momentum_boundary_token(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    scalar_token: Gm1TiledScalarMoistRk2Token,
    outputs: Mapping[str, Any],
) -> Gm1TiledRk2MomentumBoundaryToken:
    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    scalar = _require_scalar(scalar_token, acoustic=acoustic)
    hashes = _host_hashes(outputs, _MOMENTUM_SHAPES, label="RK2 momentum boundary")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_STATUS,
        "completed_nrk": 2,
        "prior_completed_rk_stage_token_sha256": prior[
            "operation_token_sha256"
        ],
        "acoustic_continuation_token_sha256": acoustic[
            "operation_token_sha256"
        ],
        "scalar_moist_rk_token_sha256": scalar["operation_token_sha256"],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": int(handoff.stream.ptr),
        "output_shapes": {
            name: list(shape) for name, shape in _MOMENTUM_SHAPES.items()
        },
        "output_sha256": dict(hashes),
        "source_order": ["bcu", "bcv", "bcw", "terrain bcwsfc"],
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK2_STAGE_FINISH_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledRk2MomentumBoundaryToken(MappingProxyType(receipt))


def publish_gm1_tiled_rk2_scalar_pressure_boundary_token(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    scalar_token: Gm1TiledScalarMoistRk2Token,
    momentum_token: Gm1TiledRk2MomentumBoundaryToken,
    outputs: Mapping[str, Any],
) -> Gm1TiledRk2ScalarPressureBoundaryToken:
    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    scalar = _require_scalar(scalar_token, acoustic=acoustic)
    momentum = _require_momentum(momentum_token, scalar=scalar, acoustic=acoustic)
    hashes = _host_hashes(outputs, _BOUNDARY_SHAPES, label="RK2 scalar boundary")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS,
        "completed_nrk": 2,
        "prior_completed_rk_stage_token_sha256": prior[
            "operation_token_sha256"
        ],
        "scalar_moist_rk_token_sha256": scalar["operation_token_sha256"],
        "momentum_boundary_token_sha256": momentum["operation_token_sha256"],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": int(handoff.stream.ptr),
        "output_shapes": {
            name: list(shape) for name, shape in _BOUNDARY_SHAPES.items()
        },
        "output_sha256": dict(hashes),
        "pressure_density_cadence": {
            "selected": {"nrk": 2, "nrkmax": 3, "psolver": 3, "idiff": 0},
            "interior_recompute_performed": False,
            "apmasscon_performed": False,
        },
        "source_order": [
            "bcp(rho)",
            "bcp(pp3d)",
            "bcs(th3d)",
            "bcs(q3d[qv,qc,qr,qi,qs,qg])",
        ],
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK2_STAGE_FINISH_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledRk2ScalarPressureBoundaryToken(MappingProxyType(receipt))


def _require_flux(
    token: object,
    *,
    prior: Mapping[str, object],
    acoustic: Mapping[str, object],
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledSolve2PostAcousticFluxToken:
        raise TypeError("flux token must be exact Gm1TiledSolve2PostAcousticFluxToken")
    receipt = token.receipt
    hashes = receipt.get("output_sha256")
    if (
        receipt.get("schema") != CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA
        or receipt.get("status") != "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX"
        or receipt.get("nrk") != 2
        or receipt.get("completed_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or receipt.get("acoustic_continuation_token_sha256")
        != acoustic.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256")
        != acoustic.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != acoustic.get("stream_pointer")
        or not isinstance(hashes, Mapping)
        or tuple(hashes) != tuple(_FLUX_SHAPES)
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK2 post-acoustic flux identity or lineage changed")
    for name, value in hashes.items():
        _lower_sha(f"flux {name}", value)
    return receipt


def _require_theta(
    token: object,
    *,
    flux: Mapping[str, object],
    acoustic: Mapping[str, object],
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledRk2ThetaDivergenceToken:
        raise TypeError("theta token must be exact Gm1TiledRk2ThetaDivergenceToken")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_STATUS
        or receipt.get("completed_nrk") != 2
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or receipt.get("acoustic_continuation_token_sha256")
        != acoustic.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256") != flux.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != flux.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK2 theta divergence identity or lineage changed")
    return receipt


def _require_scalar(
    token: object, *, acoustic: Mapping[str, object]
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledScalarMoistRk2Token:
        raise TypeError("scalar token must be exact Gm1TiledScalarMoistRk2Token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_STATUS
        or receipt.get("completed_nrk") != 2
        or receipt.get("stage_identity_sha256")
        != acoustic.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != acoustic.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK2 scalar/moist identity or lineage changed")
    return receipt


def _require_momentum(
    token: object,
    *,
    scalar: Mapping[str, object],
    acoustic: Mapping[str, object],
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledRk2MomentumBoundaryToken:
        raise TypeError("momentum token must be exact Gm1TiledRk2MomentumBoundaryToken")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_STATUS
        or receipt.get("completed_nrk") != 2
        or receipt.get("scalar_moist_rk_token_sha256")
        != scalar.get("operation_token_sha256")
        or receipt.get("acoustic_continuation_token_sha256")
        != acoustic.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256")
        != acoustic.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != acoustic.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK2 momentum-boundary identity or lineage changed")
    return receipt


def _require_boundary(
    token: object,
    *,
    scalar: Mapping[str, object],
    momentum: Mapping[str, object],
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledRk2ScalarPressureBoundaryToken:
        raise TypeError(
            "boundary token must be exact Gm1TiledRk2ScalarPressureBoundaryToken"
        )
    receipt = token.receipt
    cadence = receipt.get("pressure_density_cadence")
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS
        or receipt.get("completed_nrk") != 2
        or receipt.get("scalar_moist_rk_token_sha256")
        != scalar.get("operation_token_sha256")
        or receipt.get("momentum_boundary_token_sha256")
        != momentum.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256")
        != momentum.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != momentum.get("stream_pointer")
        or not isinstance(cadence, Mapping)
        or cadence.get("interior_recompute_performed") is not False
        or cadence.get("apmasscon_performed") is not False
        or receipt.get("source_order_component_complete") is not True
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK2 scalar/pressure-boundary identity or cadence changed")
    return receipt


def emit_gm1_tiled_completed_rk2_stage(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledRk2ThetaDivergenceToken,
    scalar_token: Gm1TiledScalarMoistRk2Token,
    momentum_token: Gm1TiledRk2MomentumBoundaryToken,
    boundary_token: Gm1TiledRk2ScalarPressureBoundaryToken,
    tke_token: Gm1TiledTkeRk23Token,
) -> Gm1TiledCompletedRkStageToken:
    """Emit completed RK2 only after the exact live tail is fully bound."""

    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    flux = _require_flux(flux_token, prior=prior, acoustic=acoustic)
    theta = _require_theta(theta_token, flux=flux, acoustic=acoustic)
    scalar = _require_scalar(scalar_token, acoustic=acoustic)
    momentum = _require_momentum(momentum_token, scalar=scalar, acoustic=acoustic)
    boundary = _require_boundary(
        boundary_token, scalar=scalar, momentum=momentum
    )
    tke = require_gm1_tiled_tke_rk23_token(
        tke_token,
        completed_nrk=2,
        stream_pointer=int(handoff.stream.ptr),
    )
    if (
        scalar.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or scalar.get("theta_divergence_token_sha256")
        != theta.get("operation_token_sha256")
        or tke.get("completed_prior_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or tke.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or tke.get("stage_identity_sha256") != pre.get("stage_identity_sha256")
        or boundary.get("stage_identity_sha256") != pre.get("stage_identity_sha256")
    ):
        raise ValueError("RK2 flux/theta/scalar/boundary/TKE lineage changed")
    components = {
        "pre_acoustic_momentum_rk": pre["operation_token_sha256"],
        "acoustic_continuation": acoustic["operation_token_sha256"],
        "post_acoustic_flux": flux["operation_token_sha256"],
        "theta_divergence": theta["operation_token_sha256"],
        "scalar_moist_rk": scalar["operation_token_sha256"],
        "momentum_boundary": momentum["operation_token_sha256"],
        "scalar_pressure_boundary": boundary["operation_token_sha256"],
        "tke_rk": tke["operation_token_sha256"],
    }
    if tuple(components) != GM1_RK2_COMPONENT_KEYS:
        raise AssertionError("internal completed RK2 component order changed")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
        "completed_nrk": 2,
        "next_nrk": 3,
        "stream_pointer": int(handoff.stream.ptr),
        "stage_identity_sha256": pre["stage_identity_sha256"],
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
        completed_nrk=2,
        stream_pointer=int(handoff.stream.ptr),
        stage_identity_sha256=str(pre["stage_identity_sha256"]),
    )
    return token


__all__ = (
    "CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK2_MOMENTUM_BOUNDARY_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK2_SCALAR_MOIST_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK2_SCALAR_PRESSURE_BOUNDARY_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK2_THETA_DIVERGENCE_TOKEN_STATUS",
    "GM1_RK2_COMPONENT_KEYS",
    "GM1_RK2_STAGE_FINISH_CLAIM_LIMIT",
    "Gm1TiledRk2MomentumBoundaryToken",
    "Gm1TiledRk2ScalarPressureBoundaryToken",
    "Gm1TiledRk2ThetaDivergenceToken",
    "Gm1TiledScalarMoistRk2Token",
    "emit_gm1_tiled_completed_rk2_stage",
    "publish_gm1_tiled_rk2_momentum_boundary_token",
    "publish_gm1_tiled_rk2_post_acoustic_tokens",
    "publish_gm1_tiled_rk2_scalar_pressure_boundary_token",
    "publish_gm1_tiled_scalar_moist_rk2_token",
)
