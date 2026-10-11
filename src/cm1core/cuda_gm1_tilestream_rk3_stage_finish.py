"""Typed publication boundary for the exact live GM1 final RK stage.

The numerical runner keeps the 300x300x75 child resident and executes the
official final-stage chronology.  This module admits only the resulting
tokens and raw-byte publications: post-acoustic getdiv/getdivx, scalar/moist
RK3, the interposed eqtset2/apmasscon-reduction/final-momentum/TKE chain, and
the delayed apmasscon finish.  A common completed-stage token is emitted only
after every predecessor and corrected pressure/density output is bound.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .cuda_gm1_rk3_apmasscon_split import (
    CudaGm1Rk3ApMassConSplitFinishLaunch,
    require_gm1_rk3_apmasscon_split_token_pair,
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
from .cuda_gm1_tilestream_rk23_acoustic import (
    Gm1TiledRk23AcousticLiveHandoff,
    require_gm1_tiled_rk23_acoustic_continuation,
)
from .cuda_gm1_tilestream_rk3_interposed_prefix import (
    CudaGm1Rk3InterposedPrefixLaunch,
    require_gm1_rk3_interposed_prefix_token,
)
from .cuda_gm1_tilestream_rk3_interposed_tke import (
    Gm1Rk3InterposedPrefixTkeToken,
    require_gm1_rk3_interposed_prefix_tke_token,
)
from .cuda_gm1_tilestream_solve1_tke_rhs import (
    Gm1TiledSolve2PostAcousticFluxToken,
)
from .cuda_gm1_tilestream_tke_rk23 import (
    Gm1TiledTkeRk23Token,
    require_gm1_tiled_tke_rk23_token,
)


CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_SCHEMA = (
    "gm1.tilestream-rk3-theta-divergence-token/v1"
)
CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK3_THETA_DIVERGENCE"
)
CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_SCHEMA = (
    "gm1.tilestream-scalar-moist-rk3-token/v1"
)
CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_SCALAR_MOIST_RK3"
)
CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_SCHEMA = (
    "gm1.tilestream-rk3-interposed-mass-finish-token/v1"
)
CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK3_INTERPOSED_MASS_FINISH"
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
_MASS_OUTPUT_SHAPES = MappingProxyType(
    {
        "pp3d": (75, 300, 300),
        "prs": (75, 300, 300),
        "rho": (75, 300, 300),
    }
)
GM1_RK3_COMPONENT_KEYS = (
    "pre_acoustic_momentum_rk",
    "acoustic_continuation",
    "post_acoustic_flux",
    "theta_divergence",
    "scalar_moist_rk",
    "eqtset2_remove_eos",
    "apmasscon_reduction",
    "momentum_boundary",
    "tke_rk",
    "apmasscon_finish",
    "mass_finish",
)
GM1_RK3_STAGE_FINISH_CLAIM_LIMIT = (
    "one exact live 300x300x75, 50 m CM1 r21.1 final RK stage through fresh "
    "post-acoustic getdiv/getdivx, scalar/moist RK3, eqtset2 removal/EOS, "
    "split apmasscon reduction, final momentum boundaries, tiled SGS-TKE RK3, "
    "and delayed apmasscon correction on one CUDA stream. It does not itself "
    "perform pdefq, WSM6, solve3 publication, MPI, checkpoint, forecast, or "
    "admit a five-metre meteorological product"
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
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _host_hashes(
    values: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...]],
    *,
    label: str,
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
class Gm1TiledRk3ThetaDivergenceToken:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledScalarMoistRk3Token:
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Gm1TiledRk3MassFinishToken:
    receipt: Mapping[str, object]


def _require_acoustic_handoff(
    handoff: object,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if type(handoff) is not Gm1TiledRk23AcousticLiveHandoff:
        raise TypeError("handoff must be exact Gm1TiledRk23AcousticLiveHandoff")
    if handoff.requested_nrk != 3:
        raise ValueError("RK3 stage finisher accepts only requested_nrk=3")
    prior = require_gm1_tiled_completed_rk_stage_token(
        handoff.prior_completed_stage_token,
        completed_nrk=2,
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
        raise ValueError("RK3 acoustic handoff lineage or residency changed")
    return prior, pre, acoustic


def publish_gm1_tiled_rk3_post_acoustic_tokens(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    flux_outputs: Mapping[str, Any],
    theta_outputs: Mapping[str, Any],
) -> tuple[Gm1TiledSolve2PostAcousticFluxToken, Gm1TiledRk3ThetaDivergenceToken]:
    """Bind exact fresh RK3 getdiv/getdivx output bytes."""

    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    flux_hashes = _host_hashes(flux_outputs, _FLUX_SHAPES, label="RK3 flux")
    theta_hashes = _host_hashes(theta_outputs, _THETA_SHAPES, label="RK3 theta")
    stream_pointer = int(handoff.stream.ptr)
    stage_identity = _lower_sha("RK3 stage identity", pre.get("stage_identity_sha256"))
    flux_receipt: dict[str, object] = {
        "schema": CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
        "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
        "nrk": 3,
        "completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "post_sound_velocity_token_sha256": acoustic[
            "post_sound_velocity_token_sha256"
        ],
        "acoustic_continuation_token_sha256": acoustic["operation_token_sha256"],
        "stage_identity_sha256": stage_identity,
        "stream_pointer": stream_pointer,
        "host_scatter_exact_once": True,
        "output_shapes": {name: list(shape) for name, shape in _FLUX_SHAPES.items()},
        "output_sha256": dict(flux_hashes),
        "source_order": [
            "global psolver=3 RK3 acoustic continuation",
            "stretched getdiv publishes divergence and theta-divergence term",
            "stretched getdivx converts averaged velocity to mass flux",
            "stretched getdivx rebuilds divx on the same live stream",
        ],
        "global_psolver_execution": True,
        "global_psolver_tile_streamable": False,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK3_STAGE_FINISH_CLAIM_LIMIT,
    }
    flux_receipt["operation_token_sha256"] = _operation_token(flux_receipt)
    flux_token = Gm1TiledSolve2PostAcousticFluxToken(
        MappingProxyType(flux_receipt)
    )
    theta_receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_STATUS,
        "completed_nrk": 3,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "acoustic_continuation_token_sha256": acoustic["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux_receipt["operation_token_sha256"],
        "stage_identity_sha256": stage_identity,
        "stream_pointer": stream_pointer,
        "output_shapes": {name: list(shape) for name, shape in _THETA_SHAPES.items()},
        "output_sha256": dict(theta_hashes),
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK3_STAGE_FINISH_CLAIM_LIMIT,
    }
    theta_receipt["operation_token_sha256"] = _operation_token(theta_receipt)
    return flux_token, Gm1TiledRk3ThetaDivergenceToken(
        MappingProxyType(theta_receipt)
    )


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
        or receipt.get("nrk") != 3
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
        raise ValueError("RK3 post-acoustic flux identity or lineage changed")
    for name, value in hashes.items():
        _lower_sha(f"flux {name}", value)
    return receipt


def _require_theta(
    token: object,
    *,
    flux: Mapping[str, object],
    acoustic: Mapping[str, object],
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledRk3ThetaDivergenceToken:
        raise TypeError("theta token must be exact Gm1TiledRk3ThetaDivergenceToken")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_STATUS
        or receipt.get("completed_nrk") != 3
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or receipt.get("acoustic_continuation_token_sha256")
        != acoustic.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256") != flux.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != flux.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK3 theta divergence identity or lineage changed")
    return receipt


def publish_gm1_tiled_scalar_moist_rk3_token(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledRk3ThetaDivergenceToken,
    outputs: Mapping[str, Any],
) -> Gm1TiledScalarMoistRk3Token:
    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    flux = _require_flux(flux_token, prior=prior, acoustic=acoustic)
    theta = _require_theta(theta_token, flux=flux, acoustic=acoustic)
    hashes = _host_hashes(outputs, _SCALAR_SHAPES, label="RK3 scalar/moist")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_STATUS,
        "completed_nrk": 3,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "post_acoustic_flux_token_sha256": flux["operation_token_sha256"],
        "theta_divergence_token_sha256": theta["operation_token_sha256"],
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": int(handoff.stream.ptr),
        "output_shapes": {name: list(shape) for name, shape in _SCALAR_SHAPES.items()},
        "output_sha256": dict(hashes),
        "source_order_component_complete": True,
        "completed_broader_rk_stage": False,
        "claim_limit": GM1_RK3_STAGE_FINISH_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledScalarMoistRk3Token(MappingProxyType(receipt))


def _require_scalar(
    token: object,
    *,
    flux: Mapping[str, object],
    theta: Mapping[str, object],
    acoustic: Mapping[str, object],
) -> Mapping[str, object]:
    if type(token) is not Gm1TiledScalarMoistRk3Token:
        raise TypeError("scalar token must be exact Gm1TiledScalarMoistRk3Token")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_SCHEMA
        or receipt.get("status") != CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_STATUS
        or receipt.get("completed_nrk") != 3
        or receipt.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or receipt.get("theta_divergence_token_sha256")
        != theta.get("operation_token_sha256")
        or receipt.get("stage_identity_sha256")
        != acoustic.get("stage_identity_sha256")
        or receipt.get("stream_pointer") != acoustic.get("stream_pointer")
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK3 scalar/moist identity or lineage changed")
    return receipt


def publish_gm1_tiled_rk3_mass_finish_token(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledRk3ThetaDivergenceToken,
    scalar_token: Gm1TiledScalarMoistRk3Token,
    prefix: CudaGm1Rk3InterposedPrefixLaunch,
    interposed_tke_token: Gm1Rk3InterposedPrefixTkeToken,
    tke_token: Gm1TiledTkeRk23Token,
    finish: CudaGm1Rk3ApMassConSplitFinishLaunch,
    outputs: Mapping[str, Any],
) -> Gm1TiledRk3MassFinishToken:
    """Bind the selected delayed mass finish to its exact interposed chain."""

    prior, _pre, acoustic = _require_acoustic_handoff(handoff)
    flux = _require_flux(flux_token, prior=prior, acoustic=acoustic)
    theta = _require_theta(theta_token, flux=flux, acoustic=acoustic)
    scalar = _require_scalar(
        scalar_token, flux=flux, theta=theta, acoustic=acoustic
    )
    if type(prefix) is not CudaGm1Rk3InterposedPrefixLaunch:
        raise TypeError("prefix must be exact CudaGm1Rk3InterposedPrefixLaunch")
    if type(finish) is not CudaGm1Rk3ApMassConSplitFinishLaunch:
        raise TypeError("finish must be exact CudaGm1Rk3ApMassConSplitFinishLaunch")
    if finish.reduction is not prefix.reduction:
        raise ValueError("apmasscon finish belongs to another live prefix reduction")
    stream_pointer = int(handoff.stream.ptr)
    prefix_receipt = require_gm1_rk3_interposed_prefix_token(
        prefix.token, stream_pointer=stream_pointer
    )
    tke = require_gm1_tiled_tke_rk23_token(
        tke_token, completed_nrk=3, stream_pointer=stream_pointer
    )
    interposed = require_gm1_rk3_interposed_prefix_tke_token(
        interposed_tke_token,
        prefix_token=prefix.token,
        tke_rk3_token=tke_token,
    )
    split = require_gm1_rk3_apmasscon_split_token_pair(
        prefix.reduction.token,
        finish.token,
        stream_pointer=stream_pointer,
    )
    if (
        prefix.stream is not handoff.stream
        or finish.reduction.stream is not handoff.stream
        or interposed.get("tke_rk3_operation_token_sha256")
        != tke.get("operation_token_sha256")
        or tke.get("completed_prior_rk_stage_token_sha256")
        != prior.get("operation_token_sha256")
        or tke.get("post_acoustic_flux_token_sha256")
        != flux.get("operation_token_sha256")
        or prefix_receipt.get("stream_pointer") != stream_pointer
    ):
        raise ValueError("RK3 prefix/TKE/split-finish live lineage changed")
    hashes = _host_hashes(outputs, _MASS_OUTPUT_SHAPES, label="RK3 mass finish")
    components = prefix_receipt["component_operation_token_sha256"]
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_STATUS,
        "completed_nrk": 3,
        "prior_completed_rk_stage_token_sha256": prior["operation_token_sha256"],
        "scalar_moist_rk_token_sha256": scalar["operation_token_sha256"],
        "prefix_operation_token_sha256": prefix_receipt["operation_token_sha256"],
        "eqtset2_remove_eos_token_sha256": components["eqtset2_remove_eos"],
        "apmasscon_reduction_token_sha256": split[
            "reduction_operation_token_sha256"
        ],
        "momentum_boundary_token_sha256": components["momentum_boundary"],
        "interposed_tke_token_sha256": interposed["operation_token_sha256"],
        "tke_rk3_token_sha256": tke["operation_token_sha256"],
        "apmasscon_finish_token_sha256": split["finish_operation_token_sha256"],
        "stage_identity_sha256": acoustic["stage_identity_sha256"],
        "stream_pointer": stream_pointer,
        "mass1_f64": split["mass1_f64"],
        "output_shapes": {
            name: list(shape) for name, shape in _MASS_OUTPUT_SHAPES.items()
        },
        "output_sha256": dict(hashes),
        "source_order": [
            "RK3 scalar/moist update",
            "eqtset2 remove/snapshot and moist EOS",
            "apmasscon layer reduction begin",
            "final momentum boundaries",
            "tiled SGS-TKE RK3",
            "apmasscon correction finish",
        ],
        "source_order_complete": True,
        "corrected_pressure_density_hashes_bound": True,
        "rr_rf_rebuilt": False,
        "solve3_entered": False,
        "claim_limit": GM1_RK3_STAGE_FINISH_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = _operation_token(receipt)
    return Gm1TiledRk3MassFinishToken(MappingProxyType(receipt))


def emit_gm1_tiled_completed_rk3_stage(
    handoff: Gm1TiledRk23AcousticLiveHandoff,
    flux_token: Gm1TiledSolve2PostAcousticFluxToken,
    theta_token: Gm1TiledRk3ThetaDivergenceToken,
    scalar_token: Gm1TiledScalarMoistRk3Token,
    prefix: CudaGm1Rk3InterposedPrefixLaunch,
    interposed_tke_token: Gm1Rk3InterposedPrefixTkeToken,
    tke_token: Gm1TiledTkeRk23Token,
    finish: CudaGm1Rk3ApMassConSplitFinishLaunch,
    mass_finish_token: Gm1TiledRk3MassFinishToken,
) -> Gm1TiledCompletedRkStageToken:
    """Emit the common completed-stage token only for the full exact RK3 tail."""

    prior, pre, acoustic = _require_acoustic_handoff(handoff)
    flux = _require_flux(flux_token, prior=prior, acoustic=acoustic)
    theta = _require_theta(theta_token, flux=flux, acoustic=acoustic)
    scalar = _require_scalar(
        scalar_token, flux=flux, theta=theta, acoustic=acoustic
    )
    if type(mass_finish_token) is not Gm1TiledRk3MassFinishToken:
        raise TypeError("mass finish token must be exact Gm1TiledRk3MassFinishToken")
    mass = mass_finish_token.receipt
    prefix_receipt = require_gm1_rk3_interposed_prefix_token(
        prefix.token, stream_pointer=int(handoff.stream.ptr)
    )
    tke = require_gm1_tiled_tke_rk23_token(
        tke_token, completed_nrk=3, stream_pointer=int(handoff.stream.ptr)
    )
    interposed = require_gm1_rk3_interposed_prefix_tke_token(
        interposed_tke_token,
        prefix_token=prefix.token,
        tke_rk3_token=tke_token,
    )
    split = require_gm1_rk3_apmasscon_split_token_pair(
        prefix.reduction.token,
        finish.token,
        stream_pointer=int(handoff.stream.ptr),
    )
    if (
        mass.get("schema") != CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_SCHEMA
        or mass.get("status") != CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_STATUS
        or mass.get("completed_nrk") != 3
        or mass.get("scalar_moist_rk_token_sha256")
        != scalar.get("operation_token_sha256")
        or mass.get("prefix_operation_token_sha256")
        != prefix_receipt.get("operation_token_sha256")
        or mass.get("interposed_tke_token_sha256")
        != interposed.get("operation_token_sha256")
        or mass.get("tke_rk3_token_sha256") != tke.get("operation_token_sha256")
        or mass.get("apmasscon_finish_token_sha256")
        != split.get("finish_operation_token_sha256")
        or mass.get("stage_identity_sha256") != pre.get("stage_identity_sha256")
        or mass.get("stream_pointer") != int(handoff.stream.ptr)
        or mass.get("source_order_complete") is not True
        or mass.get("corrected_pressure_density_hashes_bound") is not True
        or mass.get("operation_token_sha256") != _operation_token(mass)
    ):
        raise ValueError("RK3 mass-finish identity or source lineage changed")
    prefix_components = prefix_receipt["component_operation_token_sha256"]
    components = {
        "pre_acoustic_momentum_rk": pre["operation_token_sha256"],
        "acoustic_continuation": acoustic["operation_token_sha256"],
        "post_acoustic_flux": flux["operation_token_sha256"],
        "theta_divergence": theta["operation_token_sha256"],
        "scalar_moist_rk": scalar["operation_token_sha256"],
        "eqtset2_remove_eos": prefix_components["eqtset2_remove_eos"],
        "apmasscon_reduction": prefix_components["apmasscon_reduction"],
        "momentum_boundary": prefix_components["momentum_boundary"],
        "tke_rk": tke["operation_token_sha256"],
        "apmasscon_finish": split["finish_operation_token_sha256"],
        "mass_finish": mass["operation_token_sha256"],
    }
    if tuple(components) != GM1_RK3_COMPONENT_KEYS:
        raise AssertionError("internal completed RK3 component order changed")
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_SCHEMA,
        "status": CUDA_GM1_TILESTREAM_COMPLETED_RK_STAGE_TOKEN_STATUS,
        "completed_nrk": 3,
        "next_nrk": None,
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
        completed_nrk=3,
        stream_pointer=int(handoff.stream.ptr),
        stage_identity_sha256=str(pre["stage_identity_sha256"]),
    )
    return token


__all__ = (
    "CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK3_MASS_FINISH_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK3_SCALAR_MOIST_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK3_THETA_DIVERGENCE_TOKEN_STATUS",
    "GM1_RK3_COMPONENT_KEYS",
    "GM1_RK3_STAGE_FINISH_CLAIM_LIMIT",
    "Gm1TiledRk3MassFinishToken",
    "Gm1TiledRk3ThetaDivergenceToken",
    "Gm1TiledScalarMoistRk3Token",
    "emit_gm1_tiled_completed_rk3_stage",
    "publish_gm1_tiled_rk3_mass_finish_token",
    "publish_gm1_tiled_rk3_post_acoustic_tokens",
    "publish_gm1_tiled_scalar_moist_rk3_token",
)
