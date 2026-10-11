"""Fail-closed RK1 pre-acoustic to acoustic lineage contract.

The original tile-streamed post-sound token predates the full solve1 held
momentum producer.  It proves a real global acoustic execution after the
solve1 TKE RHS, but it does *not* prove CM1's intervening held-momentum seed,
terrain mass flux/divergence, momentum advection, and buoyancy chronology.

This module freezes the exact selected-profile source contract and rejects
that historical token as evidence of the broader lineage.  The positive
validator remains unavailable until the independently owned full
pre-acoustic token and its acoustic continuation publish exact Python types,
schemas, statuses, and lineage fields.  No duck-typed admission is allowed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, NoReturn

from .cuda_gm1_child_psolver3_open import (
    CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
)
from .cuda_gm1_child_stretched_momentum import (
    GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256,
)
from .cuda_gm1_solve2_rotation import CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256
from .cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from .cuda_moist_stage_bridge import CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256


CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_SCHEMA = (
    "gm1.rk1-pre-acoustic-source-contract-audit/v1"
)
CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_STATUS = (
    "SOURCE_AUDITED_RUNTIME_LINEAGE_FAIL_CLOSED"
)
CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_SCHEMA = (
    "gm1.rk1-pre-acoustic-acoustic-lineage-audit/v1"
)
CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_STATUS = (
    "VERIFIED_LIVE_PRE_ACOUSTIC_ACOUSTIC_LINEAGE_ONLY"
)
CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA = (
    "gm1.tilestream-pre-acoustic-momentum-rk-token/v1"
)
CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK1"
)
CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA = (
    "gm1.tilestream-rk1-acoustic-continuation-token/v1"
)
CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS = (
    "ADMITTED_LIVE_TILESTREAM_RK1_ACOUSTIC_CONTINUATION"
)
_FUTURE_TOKEN_MODULE = "cm1core.cuda_gm1_tilestream_pre_acoustic_momentum_rk"
AUTHORITY_TARBALL_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)
AUTHORITY_SOLVE2_SHA256 = (
    "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
)
AUTHORITY_SOUND_SHA256 = (
    "5fe055e22d106e91b5bdbe35ad7c72b2bbbcfb5b01733126703fdc7a5977da54"
)
GM1_RK1_PRE_ACOUSTIC_SOURCE_ORDER = (
    "eqtset2 moist current-stage terms then bcp(t11)",
    "seed uten/vten/wten from the exact full solve1 held token",
    "RK1 skips the nrk>=2 current-velocity terrain-boundary replay",
    "terrain rru/rrv/rrw then source bcw(rrw,0) then divx",
    "selected Coriolis, open-inflow nudging, and axisymmetric terms are inactive",
    "advu then advv then advw accumulate into the held momentum tendencies",
    "vertical buoyancy accumulates into wten",
    "terrain horizontal-buoyancy correction accumulates into uten/vten",
    "piadv is pi0+pp3d before the global psolver=3 acoustic call",
)
GM1_RK1_PRE_ACOUSTIC_SOURCE_ANCHORS = (
    "src/solve2.F:314-455",
    "src/solve2.F:459-471",
    "src/solve2.F:475-488",
    "src/solve2.F:490-627",
    "src/solve2.F:630-882",
    "src/solve2.F:883-892",
    "src/solve2.F:895-1016",
    "src/solve2.F:1019-1051",
    "src/solve2.F:1119-1137",
)
GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS = (
    "uten",
    "vten",
    "wten",
    "rru",
    "rrv",
    "rrw",
    "divx",
)
GM1_RK1_PRE_ACOUSTIC_SELECTED_PROFILE = MappingProxyType(
    {
        "nrk": 1,
        "nrkmax": 3,
        "dt_seconds": 0.5,
        "dx_m": 50.0,
        "dy_m": 50.0,
        "domain": [75, 300, 300],
        "psolver": 3,
        "nsound": 6,
        "eqtset": 2,
        "imoist": 1,
        "iice": 1,
        "hadvordrv": 5,
        "vadvordrv": 5,
        "advwenov": 0,
        "terrain": True,
        "stretch_z": 1,
        "lateral_boundaries": [2, 2, 2, 2],
        "axisymmetric": 0,
        "icor": 0,
        "nudgeobc": 0,
        "mpi": False,
    }
)
GM1_RK1_PRE_ACOUSTIC_LINEAGE_CLAIM_LIMIT = (
    "source audit and structural refusal only: no full pre-acoustic runtime "
    "token, no admitted pre-acoustic-to-acoustic lineage, no completed RK "
    "stage, no RK2/RK3/solve3, no tiled global pressure solve, no MPI, no "
    "forecast, and no five-metre meteorological product"
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


def audit_gm1_rk1_pre_acoustic_source_contract(
    rotation_selection: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Return an immutable source audit that cannot masquerade as a token."""

    selected_profile = dict(GM1_RK1_PRE_ACOUSTIC_SELECTED_PROFILE)
    source_order = list(GM1_RK1_PRE_ACOUSTIC_SOURCE_ORDER)
    operator_sources = {
        "stretched_momentum": GM1_CHILD_STRETCHED_MOMENTUM_SOURCE_SHA256,
        "moist_stage_bridge": CUDA_MOIST_STAGE_BRIDGE_SOURCE_SHA256,
        "global_psolver3": CUDA_GM1_CHILD_PSOLVER3_OPEN_SOURCE_SHA256,
    }
    if rotation_selection is not None:
        if (
            not isinstance(rotation_selection, Mapping)
            or rotation_selection.get("icor") != 1
            or rotation_selection.get("beta_plane") not in (0, 1)
            or rotation_selection.get("axisymmetric") not in (0, 1)
        ):
            raise ValueError("active RK1 rotation source selection changed")
        selected_profile.update(
            {
                "icor": 1,
                "betaplane": rotation_selection["beta_plane"],
                "axisymmetric": 0,
                "rotation_axisymmetric_u_inner_term": rotation_selection[
                    "axisymmetric"
                ],
                "outer_axisymmetric_coordinate_admitted": False,
            }
        )
        source_order[4] = (
            "selected Coriolis rotation forcing follows divx before advu/advv/advw; "
            "open-inflow nudging remains inactive"
        )
        operator_sources["rotation_forcing"] = (
            CUDA_GM1_SOLVE2_ROTATION_SOURCE_SHA256
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_SCHEMA,
        "status": CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_STATUS,
        "selected_profile": selected_profile,
        "source_order": source_order,
        "required_output_sha256_keys": list(GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS),
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "sound_sha256": AUTHORITY_SOUND_SHA256,
            "anchors": list(GM1_RK1_PRE_ACOUSTIC_SOURCE_ANCHORS),
        },
        "operator_source_sha256": operator_sources,
        "required_runtime_lineage": [
            "exact Gm1TiledSolve1FullHeldToken operation hash",
            "same live stage identity and CUDA stream",
            "current-stage eqtset2 terms before momentum seed",
            "pre-sound output hashes for every required output key",
            "full pre-acoustic token hash bound by the acoustic continuation",
        ],
        "historical_post_sound_velocity_token_accepted": False,
        "positive_runtime_validator_implemented": True,
        "positive_runtime_token_types_required": True,
        "completed_stage_token_emitted": False,
        "claim_limit": GM1_RK1_PRE_ACOUSTIC_LINEAGE_CLAIM_LIMIT,
    }
    if rotation_selection is not None:
        receipt["rotation_selection"] = dict(rotation_selection)
        receipt["rotation_after_divx_before_momentum_advection"] = True
        receipt["active_rotation_cuda_raw_bit_oracle"] = False
    receipt["source_contract_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return MappingProxyType(receipt)


def refuse_historical_gm1_post_sound_velocity_as_full_lineage(
    token: object,
) -> NoReturn:
    """Validate the historical token, then reject its broader reuse."""

    if not isinstance(token, Gm1PostSoundVelocityToken):
        raise TypeError("token must be the exact historical Gm1PostSoundVelocityToken")
    receipt = token.receipt
    if (
        receipt.get("schema") != CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA
        or receipt.get("status")
        != "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT"
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("historical post-sound token identity or cadence changed")
    _lower_sha(
        "historical solve1 TKE RHS lineage",
        receipt.get("solve1_tke_rhs_token_sha256"),
    )
    _lower_sha("historical stage identity", receipt.get("stage_identity_sha256"))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("historical post-sound CUDA stream pointer is invalid")
    raise RuntimeError(
        "GM1-RK1-PRE-ACOUSTIC-LINEAGE: historical post-sound velocity token "
        "binds only solve1_tke_rhs; it does not bind the full held momentum, "
        "terrain mass-flux/divx, momentum-advection, and buoyancy chronology"
    )


def _future_token_types() -> tuple[type[Any], type[Any]]:
    try:
        module = importlib.import_module(_FUTURE_TOKEN_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _FUTURE_TOKEN_MODULE:
            raise
        raise RuntimeError(
            "GM1-RK1-PRE-ACOUSTIC-LINEAGE: reserved exact token module has not "
            "landed; runtime admission unavailable"
        ) from error
    pre = getattr(module, "Gm1TiledPreAcousticMomentumRkToken", None)
    acoustic = getattr(module, "Gm1TiledRk1AcousticContinuationToken", None)
    if not isinstance(pre, type) or not isinstance(acoustic, type):
        raise RuntimeError(
            "GM1-RK1-PRE-ACOUSTIC-LINEAGE: reserved exact token classes have "
            "not both landed; runtime admission unavailable"
        )
    return pre, acoustic


def _token_receipt(name: str, token: object) -> Mapping[str, object]:
    receipt = getattr(token, "receipt", None)
    if not isinstance(receipt, Mapping):
        raise TypeError(f"{name} must expose its exact immutable receipt")
    return receipt


def _require_pre_acoustic_token(token: object, expected_type: type[Any]) -> Mapping[str, object]:
    if type(token) is not expected_type:
        raise TypeError(
            "pre_acoustic_token must be the exact "
            "Gm1TiledPreAcousticMomentumRkToken type"
        )
    receipt = _token_receipt("pre_acoustic_token", token)
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("pre-acoustic momentum token identity or cadence changed")
    for name in (
        "stage_identity_sha256",
        "solve1_full_held_token_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(f"pre-acoustic {name}", receipt.get(name))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("pre-acoustic CUDA stream pointer is invalid")
    output_sha = receipt.get("output_sha256")
    output_shapes = receipt.get("output_shapes")
    if not isinstance(output_sha, Mapping) or tuple(output_sha) != GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS:
        raise ValueError("pre-acoustic output SHA inventory changed")
    if not isinstance(output_shapes, Mapping) or tuple(output_shapes) != GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS:
        raise ValueError("pre-acoustic output shape inventory changed")
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
        raise ValueError("pre-acoustic physical output shapes changed")
    for name in GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS:
        _lower_sha(f"pre-acoustic output {name}", output_sha.get(name))
    return receipt


def _require_acoustic_continuation_token(
    token: object, expected_type: type[Any]
) -> Mapping[str, object]:
    if type(token) is not expected_type:
        raise TypeError(
            "acoustic_token must be the exact "
            "Gm1TiledRk1AcousticContinuationToken type"
        )
    receipt = _token_receipt("acoustic_token", token)
    if (
        receipt.get("schema")
        != CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA
        or receipt.get("status")
        != CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS
        or receipt.get("nrk") != 1
        or receipt.get("operation_token_sha256") != _operation_token(receipt)
    ):
        raise ValueError("RK1 acoustic continuation token identity or cadence changed")
    for name in (
        "stage_identity_sha256",
        "pre_acoustic_momentum_rk_token_sha256",
        "post_sound_velocity_token_sha256",
        "operation_token_sha256",
    ):
        _lower_sha(f"acoustic continuation {name}", receipt.get(name))
    stream = receipt.get("stream_pointer")
    if isinstance(stream, bool) or not isinstance(stream, int) or stream <= 0:
        raise ValueError("acoustic continuation CUDA stream pointer is invalid")
    return receipt


def require_gm1_rk1_pre_acoustic_acoustic_lineage(
    pre_acoustic_token: object,
    acoustic_token: object,
) -> Mapping[str, object]:
    """Validate exact future types while rejecting the historical token alone."""

    if isinstance(acoustic_token, Gm1PostSoundVelocityToken):
        refuse_historical_gm1_post_sound_velocity_as_full_lineage(acoustic_token)
    pre_type, acoustic_type = _future_token_types()
    pre = _require_pre_acoustic_token(pre_acoustic_token, pre_type)
    acoustic = _require_acoustic_continuation_token(acoustic_token, acoustic_type)
    if (
        acoustic.get("pre_acoustic_momentum_rk_token_sha256")
        != pre.get("operation_token_sha256")
        or acoustic.get("stage_identity_sha256")
        != pre.get("stage_identity_sha256")
        or acoustic.get("stream_pointer") != pre.get("stream_pointer")
    ):
        raise ValueError(
            "pre-acoustic and acoustic continuation lineage, stage, or stream changed"
        )
    audit: dict[str, object] = {
        "schema": CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_SCHEMA,
        "status": CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_STATUS,
        "nrk": 1,
        "stage_identity_sha256": pre["stage_identity_sha256"],
        "stream_pointer": pre["stream_pointer"],
        "pre_acoustic_momentum_rk_token_sha256": pre["operation_token_sha256"],
        "acoustic_continuation_token_sha256": acoustic["operation_token_sha256"],
        "post_sound_velocity_token_sha256": acoustic[
            "post_sound_velocity_token_sha256"
        ],
        "source_order_complete_through_global_acoustic": True,
        "completed_stage_token_emitted": False,
        "global_psolver_tile_streamable": False,
        "claim_limit": GM1_RK1_PRE_ACOUSTIC_LINEAGE_CLAIM_LIMIT,
    }
    audit["lineage_audit_sha256"] = hashlib.sha256(_canonical(audit)).hexdigest()
    return MappingProxyType(audit)


__all__ = (
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_SOUND_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_SCHEMA",
    "CUDA_GM1_RK1_PRE_ACOUSTIC_SOURCE_AUDIT_STATUS",
    "CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_SCHEMA",
    "CUDA_GM1_RK1_PRE_ACOUSTIC_LINEAGE_AUDIT_STATUS",
    "CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_PRE_ACOUSTIC_MOMENTUM_RK_TOKEN_STATUS",
    "CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_SCHEMA",
    "CUDA_GM1_TILESTREAM_RK1_ACOUSTIC_CONTINUATION_TOKEN_STATUS",
    "GM1_RK1_PRE_ACOUSTIC_LINEAGE_CLAIM_LIMIT",
    "GM1_RK1_PRE_ACOUSTIC_OUTPUT_KEYS",
    "GM1_RK1_PRE_ACOUSTIC_SELECTED_PROFILE",
    "GM1_RK1_PRE_ACOUSTIC_SOURCE_ANCHORS",
    "GM1_RK1_PRE_ACOUSTIC_SOURCE_ORDER",
    "audit_gm1_rk1_pre_acoustic_source_contract",
    "refuse_historical_gm1_post_sound_velocity_as_full_lineage",
    "require_gm1_rk1_pre_acoustic_acoustic_lineage",
)
