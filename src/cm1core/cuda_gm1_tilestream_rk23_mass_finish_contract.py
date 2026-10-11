"""Fail-closed source contract for GM1 RK2/RK3 and final mass correction.

CM1 does not enter RK2 or RK3 merely because a narrower component finished.
The next stage starts only after the *broader* previous RK stage is complete,
then rebuilds its current thermodynamic terms, replays the current-velocity
terrain boundary, and produces fresh momentum, acoustic, scalar, boundary,
and TKE lineages.

The final ``apmasscon`` correction is inside RK3, not after a completed RK3
token.  Its layer reductions begin before momentum boundaries and TKE, while
the correction and moist equation-of-state rebuild finish after them.  A raw
``CudaApMassConLaunch`` therefore cannot stand in for the future live
source-order mass-finish token.  This module publishes immutable audits and
structural refusals only; it emits no continuation, mass-finish, or completed
stage token.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import NoReturn

from .cuda_apmasscon import (
    CUDA_APMASSCON_CLAIM_LIMIT,
    CUDA_APMASSCON_SCHEMA,
    CUDA_APMASSCON_SOURCE_SHA256,
    CudaApMassConLaunch,
)
from .cuda_gm1_rk3_apmasscon_split import (
    Gm1Rk3ApMassConSplitFinishToken,
    Gm1Rk3ApMassConSplitReductionToken,
    require_gm1_rk3_apmasscon_split_reduction_token,
    require_gm1_rk3_apmasscon_split_token_pair,
)
from .cuda_gm1_tilestream_rk3_eqtset2_remove_eos import (
    Gm1Rk3Eqset2RemoveEosLocalToken,
    require_gm1_rk3_eqtset2_remove_eos_local_token,
)
from .cuda_gm1_tilestream_rk3_final_momentum_boundaries import (
    Gm1Rk3FinalMomentumBoundaryLocalToken,
    require_gm1_rk3_final_momentum_boundary_local_token,
)
from .cuda_gm1_tilestream_completed_rk_stage import (
    Gm1TiledCompletedRkStageToken,
    require_gm1_tiled_completed_rk_stage_token,
)
from .cuda_sound3_column import rk_stage_dt_f32, sound3_stage_schedule_f32


CUDA_GM1_RK23_CONTINUATION_ENTRY_AUDIT_SCHEMA = (
    "gm1.rk23-continuation-entry-source-audit/v1"
)
CUDA_GM1_RK23_CONTINUATION_ENTRY_AUDIT_STATUS = (
    "SOURCE_AUDITED_RK23_CONTINUATION_RUNTIME_FAIL_CLOSED"
)
CUDA_GM1_RK3_MASS_FINISH_AUDIT_SCHEMA = (
    "gm1.rk3-mass-finish-source-audit/v1"
)
CUDA_GM1_RK3_MASS_FINISH_AUDIT_STATUS = (
    "SOURCE_AUDITED_RK3_MASS_FINISH_RUNTIME_FAIL_CLOSED"
)
CUDA_GM1_RK3_SPLIT_PAIR_AUDIT_SCHEMA = (
    "gm1.rk3-split-apmasscon-pair-source-audit/v1"
)
CUDA_GM1_RK3_SPLIT_PAIR_AUDIT_STATUS = (
    "ADMITTED_LOCAL_SPLIT_PAIR_SCIENTIFIC_LINEAGE_FAIL_CLOSED"
)
CUDA_GM1_RK3_INTERPOSED_PREFIX_AUDIT_SCHEMA = (
    "gm1.rk3-local-interposed-prefix-source-audit/v1"
)
CUDA_GM1_RK3_INTERPOSED_PREFIX_AUDIT_STATUS = (
    "ADMITTED_LOCAL_RK3_PREFIX_IDENTITIES_SOURCE_ORDER_FAIL_CLOSED"
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

GM1_RK23_CONTINUATION_SOURCE_ANCHORS = (
    "src/solve2.F:296-299",
    "src/solve2.F:314-455",
    "src/solve2.F:459-488",
    "src/solve2.F:490-1137",
    "src/solve2.F:1139-1533",
    "src/solve2.F:1535-1706",
    "src/solve2.F:1708-1803",
    "src/solve2.F:2034-2052",
    "src/sound.F:84-108",
)
GM1_RK3_MASS_FINISH_SOURCE_ANCHORS = (
    "src/solve2.F:1535-1615",
    "src/solve2.F:1616-1656",
    "src/solve2.F:1662-1706",
    "src/solve2.F:1708-1803",
    "src/solve2.F:1805-1970",
    "src/solve2.F:1972-2030",
    "src/solve2.F:2034-2052",
)

GM1_RK23_SELECTED_PROFILE = MappingProxyType(
    {
        "nrkmax": 3,
        "dt_seconds": 0.5,
        "nsound": 6,
        "domain": [75, 300, 300],
        "dx_m": 50.0,
        "dy_m": 50.0,
        "psolver": 3,
        "eqtset": 2,
        "imoist": 1,
        "iice": 1,
        "ptype": 5,
        "apmasscon": 1,
        "idiff": 0,
        "difforder": 6,
        "ipbl": 0,
        "iptra": 0,
        "axisymmetric": 0,
        "terrain": True,
        "stretch_z": 1,
        "mpi": False,
    }
)

GM1_RK23_COMMON_ENTRY_SOURCE_ORDER = (
    "require the exact broader completed prior RK stage on the live stream",
    "set dttmp=dt/(nrkmax+1-nrk) in default REAL arithmetic",
    "rebuild current eqtset2 buoyancy, theta-rho, thterm, and ppterm",
    "apply bcp to the current t11 carrier",
    "seed current momentum tendencies from the held solve1 tendencies",
    "complete prior current-velocity communication in MPI builds",
    "reapply terrain bcwsfc and bottom bc2d to current w3d",
    "build current rru, rrv, rrw, and divx",
    "accumulate momentum advection, buoyancy, and pressure-gradient terms",
    "run the global psolver=3 acoustic stage with the RK-specific schedule",
    "publish fresh post-acoustic flux and theta-divergence lineages",
    "advance current theta and moisture from their held tendencies",
)
GM1_RK2_STAGE_TAIL_SOURCE_ORDER = (
    "skip pressure-density EOS recompute because nrk<nrkmax and idiff=0",
    "skip apmasscon because it is final-RK only",
    "apply momentum bcu, bcv, bcw, and terrain bcwsfc",
    "apply rho, pp3d, th3d, and every moisture scalar boundary",
    "advance and boundary the current SGS TKE RK2 state",
    "skip reciprocal-density rebuild because idiff=0",
    "aggregate the broader RK2 stage only after every live component token",
)
GM1_RK3_STAGE_TAIL_SOURCE_ORDER = (
    "remove estimated eqtset2 diabatic terms and snapshot pre-physics fields",
    "rebuild moist pressure and density before the mass reduction",
    "begin apmasscon layer dry-mass and Exner-sum reductions",
    "apply momentum bcu, bcv, bcw, and terrain bcwsfc",
    "skip scalar and pressure boundaries because nrk equals nrkmax",
    "advance and boundary the current SGS TKE RK3 state",
    "skip selected PBL and passive-tracer branches",
    "finish apmasscon and rebuild corrected pressure and density",
    "skip reciprocal-density rebuild because idiff=0",
    "aggregate the broader RK3 stage only after exact mass-finish lineage",
)
GM1_RK3_MASS_FINISH_SOURCE_ORDER = (
    "pp3d -= dt*qpten then qpten = pp3d",
    "th3d -= dt*qtten then qtten = th3d",
    "qv -= dt*qvten then qvten = qv",
    "qc -= dt*qcten then qcten = qc",
    "rebuild moist prs and rho from the corrected RK3 state",
    "reduce layer dry mass and Exner sums before stage boundaries and TKE",
    "apply final momentum boundaries; scalar-pressure boundaries are skipped",
    "complete source-ordered SGS TKE RK3 and inactive selected optional branches",
    "sum reduction ledgers and compute the uniform Exner correction",
    "add the correction to pp3d and rebuild moist prs and rho",
    "leave rr and rf unchanged here because selected idiff=0",
)

GM1_RK23_REQUIRED_FRESH_COMPONENTS = (
    "current_velocity_terrain_boundary_replay",
    "pre_acoustic_momentum_rk",
    "acoustic_continuation",
    "post_acoustic_flux",
    "theta_divergence",
    "scalar_moist_rk",
    "momentum_boundary",
    "pressure_density_boundary_or_final_cadence",
    "tke_rk",
    "broader_stage_aggregator",
)
GM1_RK3_MASS_FINISH_REQUIRED_LINEAGE = (
    "prior_completed_rk2",
    "rk3_pre_acoustic_momentum_rk",
    "rk3_acoustic_continuation",
    "rk3_post_acoustic_flux",
    "rk3_scalar_moist_rk",
    "rk3_eqtset2_remove_snapshot_eos",
    "rk3_apmasscon_reduction_begin",
    "rk3_momentum_boundary",
    "rk3_tke_rk",
    "rk3_apmasscon_finish_checkpoint",
)
GM1_RK3_MASS_FINISH_OUTPUT_KEYS = ("pp3d", "prs", "rho")
GM1_RK3_MASS_FINISH_OUTPUT_SHAPES = MappingProxyType(
    {name: [75, 300, 300] for name in GM1_RK3_MASS_FINISH_OUTPUT_KEYS}
)

GM1_RK23_CONTINUATION_CLAIM_LIMIT = (
    "immutable CM1 r21.1 selected-profile source audit for entering RK2 or "
    "RK3 after an exact broader prior-stage token. It emits no continuation "
    "or completed-stage token, supplies no fresh current-stage component, "
    "tiles no global pressure solver, performs no MPI, and establishes no "
    "solve3 entry, forecast, tornado, or five-metre meteorological product"
)
GM1_RK3_MASS_FINISH_CLAIM_LIMIT = (
    "immutable CM1 r21.1 selected-profile audit of final-RK eqtset2 removal, "
    "snapshot, moist EOS, split apmasscon reduction/finish placement, and "
    "post-correction pp3d/prs/rho checkpoint requirements. It emits no mass-"
    "finish or completed-RK3 token, rebuilds no rr/rf, enters no solve3 or "
    "WSM6 step, and establishes no forecast or five-metre product"
)
GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT = (
    "exact-type validation of a completed-RK2 receipt plus local final-RK "
    "eqtset2/EOS, apmasscon-reduction, and momentum-boundary receipts on one "
    "CUDA stream. The local receipts carry no predecessor links or shared "
    "carrier hashes, so they do not prove enqueue order or live scientific "
    "lineage, bind RK3 TKE or apmasscon finish/output hashes, emit a mass-"
    "finish/completed-RK3 token, or enter solve3, histories, or checkpoint"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _require_prior_completed_stage(
    token: object, *, requested_nrk: int
) -> Mapping[str, object]:
    if (
        isinstance(requested_nrk, bool)
        or not isinstance(requested_nrk, int)
        or requested_nrk not in (2, 3)
    ):
        raise ValueError("requested_nrk must be exactly 2 or 3")
    if type(token) is not Gm1TiledCompletedRkStageToken:
        raise TypeError(
            "prior_completed_stage_token must be the exact "
            "Gm1TiledCompletedRkStageToken type"
        )
    return require_gm1_tiled_completed_rk_stage_token(
        token,
        completed_nrk=requested_nrk - 1,
    )


def _stage_schedule(requested_nrk: int) -> Mapping[str, object]:
    dt = float(GM1_RK23_SELECTED_PROFILE["dt_seconds"])
    nsound = int(GM1_RK23_SELECTED_PROFILE["nsound"])
    nloop, dts = sound3_stage_schedule_f32(dt, nsound, requested_nrk)
    return MappingProxyType(
        {
            "rk_stage_dt_seconds_f32": rk_stage_dt_f32(dt, requested_nrk),
            "sound_substeps": nloop,
            "sound_dt_seconds_f32": dts,
        }
    )


def audit_gm1_rk23_continuation_entry(
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken,
    *,
    requested_nrk: int,
) -> Mapping[str, object]:
    """Audit exact prior-stage cadence without publishing a continuation."""

    prior = _require_prior_completed_stage(
        prior_completed_stage_token,
        requested_nrk=requested_nrk,
    )
    schedule = _stage_schedule(requested_nrk)
    tail = (
        GM1_RK2_STAGE_TAIL_SOURCE_ORDER
        if requested_nrk == 2
        else GM1_RK3_STAGE_TAIL_SOURCE_ORDER
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK23_CONTINUATION_ENTRY_AUDIT_SCHEMA,
        "status": CUDA_GM1_RK23_CONTINUATION_ENTRY_AUDIT_STATUS,
        "requested_nrk": requested_nrk,
        "required_completed_nrk": requested_nrk - 1,
        "prior_completed_stage_token_sha256": prior["operation_token_sha256"],
        "prior_stage_identity_sha256": prior["stage_identity_sha256"],
        "stream_pointer": prior["stream_pointer"],
        "schedule": dict(schedule),
        "selected_profile": dict(GM1_RK23_SELECTED_PROFILE),
        "source_order": list(GM1_RK23_COMMON_ENTRY_SOURCE_ORDER + tail),
        "required_fresh_component_keys": list(
            GM1_RK23_REQUIRED_FRESH_COMPONENTS
        ),
        "current_velocity_boundary_replay_required": True,
        "historical_rk1_post_sound_token_admissible": False,
        "fresh_runtime_components_available": False,
        "continuation_token_emitted": False,
        "completed_stage_token_emitted": False,
        "global_psolver_tile_streamable": False,
        "mpi_performed": False,
        "solve3_entered": False,
        "full_forecast": False,
        "five_meter_meteorological_product": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "sound_sha256": AUTHORITY_SOUND_SHA256,
            "anchors": list(GM1_RK23_CONTINUATION_SOURCE_ANCHORS),
        },
        "claim_limit": GM1_RK23_CONTINUATION_CLAIM_LIMIT,
    }
    receipt["source_audit_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return MappingProxyType(receipt)


def refuse_gm1_rk23_continuation_without_fresh_runtime_lineage(
    prior_completed_stage_token: Gm1TiledCompletedRkStageToken,
    *,
    requested_nrk: int,
) -> NoReturn:
    """Validate the prior stage, then refuse an unimplemented live stage."""

    audit_gm1_rk23_continuation_entry(
        prior_completed_stage_token,
        requested_nrk=requested_nrk,
    )
    raise RuntimeError(
        f"GM1-RK{requested_nrk}-CONTINUATION-PREREQUISITE: exact fresh "
        "current-velocity boundary, momentum/acoustic, scalar, pressure, "
        "and TKE runtime lineage is missing; no continuation or completed "
        "stage token emitted"
    )


def audit_gm1_rk3_mass_finish_source_contract(
    prior_completed_rk2_token: Gm1TiledCompletedRkStageToken,
) -> Mapping[str, object]:
    """Audit final-RK mass ordering from completed RK2 without a cycle."""

    prior = _require_prior_completed_stage(
        prior_completed_rk2_token,
        requested_nrk=3,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_MASS_FINISH_AUDIT_SCHEMA,
        "status": CUDA_GM1_RK3_MASS_FINISH_AUDIT_STATUS,
        "candidate_nrk": 3,
        "required_completed_nrk": 2,
        "prior_completed_rk2_token_sha256": prior["operation_token_sha256"],
        "prior_stage_identity_sha256": prior["stage_identity_sha256"],
        "stream_pointer": prior["stream_pointer"],
        "selected_profile": dict(GM1_RK23_SELECTED_PROFILE),
        "source_order": list(GM1_RK3_MASS_FINISH_SOURCE_ORDER),
        "required_runtime_lineage_keys": list(
            GM1_RK3_MASS_FINISH_REQUIRED_LINEAGE
        ),
        "required_output_sha256_keys": list(GM1_RK3_MASS_FINISH_OUTPUT_KEYS),
        "required_output_shapes": dict(GM1_RK3_MASS_FINISH_OUTPUT_SHAPES),
        "apmasscon_operator": {
            "schema": CUDA_APMASSCON_SCHEMA,
            "source_sha256": CUDA_APMASSCON_SOURCE_SHA256,
            "claim_limit": CUDA_APMASSCON_CLAIM_LIMIT,
            "local_operator_implemented": True,
            "raw_launch_admissible_as_mass_finish_token": False,
            "source_order_split_lineage_bound": False,
        },
        "completed_rk3_token_is_prerequisite": False,
        "completed_rk3_token_would_be_circular": True,
        "runtime_lineage_available": False,
        "mass_finish_source_order_complete": False,
        "mass_finish_token_emitted": False,
        "completed_rk3_token_emitted": False,
        "rr_rf_rebuilt": False,
        "solve3_entered": False,
        "wsm6_entered": False,
        "full_forecast": False,
        "five_meter_meteorological_product": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(GM1_RK3_MASS_FINISH_SOURCE_ANCHORS),
        },
        "claim_limit": GM1_RK3_MASS_FINISH_CLAIM_LIMIT,
    }
    receipt["source_audit_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    return MappingProxyType(receipt)


def audit_gm1_rk3_split_pair_without_live_lineage(
    prior_completed_rk2_token: Gm1TiledCompletedRkStageToken,
    reduction_token: Gm1Rk3ApMassConSplitReductionToken,
    finish_token: Gm1Rk3ApMassConSplitFinishToken,
) -> Mapping[str, object]:
    """Bind the local split pair to RK2 while preserving the RK3 refusal."""

    prior = _require_prior_completed_stage(
        prior_completed_rk2_token,
        requested_nrk=3,
    )
    pair = require_gm1_rk3_apmasscon_split_token_pair(
        reduction_token,
        finish_token,
        stream_pointer=int(prior["stream_pointer"]),
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_SPLIT_PAIR_AUDIT_SCHEMA,
        "status": CUDA_GM1_RK3_SPLIT_PAIR_AUDIT_STATUS,
        "candidate_nrk": 3,
        "prior_completed_rk2_token_sha256": prior["operation_token_sha256"],
        "prior_stage_identity_sha256": prior["stage_identity_sha256"],
        "stream_pointer": prior["stream_pointer"],
        "reduction_operation_token_sha256": pair[
            "reduction_operation_token_sha256"
        ],
        "finish_operation_token_sha256": pair[
            "finish_operation_token_sha256"
        ],
        "domain": dict(pair["domain"]),
        "mass1_f64": pair["mass1_f64"],
        "pointer_inventory_sha256": pair["pointer_inventory_sha256"],
        "required_runtime_lineage_keys": list(
            GM1_RK3_MASS_FINISH_REQUIRED_LINEAGE
        ),
        "local_pair_raw_identity_bound": True,
        "scientific_interposed_lineage_bound": False,
        "eqtset2_remove_snapshot_eos_bound": False,
        "rk3_momentum_boundary_bound": False,
        "rk3_tke_bound": False,
        "post_output_sha256_bound": False,
        "mass_finish_source_order_complete": False,
        "mass_finish_token_emitted": False,
        "completed_rk3_token_emitted": False,
        "solve3_entered": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(GM1_RK3_MASS_FINISH_SOURCE_ANCHORS),
        },
        "claim_limit": GM1_RK3_MASS_FINISH_CLAIM_LIMIT,
    }
    receipt["source_audit_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    return MappingProxyType(receipt)


def audit_gm1_rk3_local_interposed_prefix_fail_closed(
    prior_completed_rk2_token: Gm1TiledCompletedRkStageToken,
    eqtset2_remove_eos_token: Gm1Rk3Eqset2RemoveEosLocalToken,
    reduction_token: Gm1Rk3ApMassConSplitReductionToken,
    momentum_boundary_token: Gm1Rk3FinalMomentumBoundaryLocalToken,
) -> Mapping[str, object]:
    """Bind the three local prefix identities without inventing chronology.

    All tokens must be exact producer types and name one CUDA stream.  Their
    current schemas intentionally contain neither a predecessor operation hash
    nor per-carrier hashes shared across all three operators.  Consequently the
    return value is an immutable audit, never an executable/live prefix token.
    """

    prior = _require_prior_completed_stage(
        prior_completed_rk2_token,
        requested_nrk=3,
    )
    stream_pointer = int(prior["stream_pointer"])
    eqtset = require_gm1_rk3_eqtset2_remove_eos_local_token(
        eqtset2_remove_eos_token,
        stream_pointer=stream_pointer,
    )
    reduction = require_gm1_rk3_apmasscon_split_reduction_token(
        reduction_token,
        stream_pointer=stream_pointer,
    )
    boundary = require_gm1_rk3_final_momentum_boundary_local_token(
        momentum_boundary_token,
        stream_pointer=stream_pointer,
    )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_INTERPOSED_PREFIX_AUDIT_SCHEMA,
        "status": CUDA_GM1_RK3_INTERPOSED_PREFIX_AUDIT_STATUS,
        "candidate_nrk": 3,
        "prior_completed_rk2_token_sha256": prior["operation_token_sha256"],
        "prior_stage_identity_sha256": prior["stage_identity_sha256"],
        "stream_pointer": stream_pointer,
        "local_operation_token_sha256": {
            "eqtset2_remove_eos": eqtset["operation_token_sha256"],
            "apmasscon_reduction": reduction["operation_token_sha256"],
            "momentum_boundary": boundary["operation_token_sha256"],
        },
        "local_pointer_inventory_sha256": {
            "eqtset2_remove_eos": eqtset["pointer_inventory_sha256"],
            "apmasscon_reduction": reduction["pointer_inventory_sha256"],
            "momentum_boundary": boundary["pointer_inventory_sha256"],
        },
        "source_order": [
            "solve2.F:1561-1595 local eqtset2 remove/snapshot and moist EOS",
            "solve2.F:1616-1656 local apmasscon layer reduction begin",
            "solve2.F:1662-1680 local final momentum boundaries",
            "solve2.F:1708-1803 missing live SGS-TKE RK3",
            "solve2.F:1972-2030 missing live apmasscon finish",
        ],
        "exact_local_token_types_bound": True,
        "same_cuda_stream_bound": True,
        "source_order_enqueue_chain_bound": False,
        "shared_live_carrier_identity_bound": False,
        "fresh_rk3_scalar_moist_lineage_bound": False,
        "prior_completed_rk2_scientific_lineage_bound_into_eqtset": False,
        "rk3_tke_bound": False,
        "apmasscon_finish_bound": False,
        "post_output_sha256_bound": False,
        "mass_finish_source_order_complete": False,
        "mass_finish_token_emitted": False,
        "completed_rk3_token_emitted": False,
        "solve3_entered": False,
        "history_checkpoint_emitted": False,
        "authority": {
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "solve2_sha256": AUTHORITY_SOLVE2_SHA256,
            "anchors": list(GM1_RK3_MASS_FINISH_SOURCE_ANCHORS),
        },
        "claim_limit": GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT,
    }
    receipt["source_audit_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    return MappingProxyType(receipt)


def refuse_gm1_rk3_local_interposed_prefix_as_completed_stage(
    prior_completed_rk2_token: Gm1TiledCompletedRkStageToken,
    eqtset2_remove_eos_token: Gm1Rk3Eqset2RemoveEosLocalToken,
    reduction_token: Gm1Rk3ApMassConSplitReductionToken,
    momentum_boundary_token: Gm1Rk3FinalMomentumBoundaryLocalToken,
) -> NoReturn:
    """Validate exact identities, then refuse promotion beyond their proof."""

    audit_gm1_rk3_local_interposed_prefix_fail_closed(
        prior_completed_rk2_token,
        eqtset2_remove_eos_token,
        reduction_token,
        momentum_boundary_token,
    )
    raise RuntimeError(
        "GM1-RK3-LOCAL-INTERPOSED-PREFIX-LINEAGE: exact local eqtset2, "
        "reduction, and final momentum-boundary identities share one stream, "
        "but their receipts do not prove predecessor order or shared live "
        "carriers; RK3 TKE, mass finish, post-output hashes, and completed "
        "RK3 remain unavailable"
    )


def refuse_gm1_rk3_split_pair_as_mass_finish_without_live_lineage(
    prior_completed_rk2_token: Gm1TiledCompletedRkStageToken,
    reduction_token: Gm1Rk3ApMassConSplitReductionToken,
    finish_token: Gm1Rk3ApMassConSplitFinishToken,
) -> NoReturn:
    """Validate the exact pair, then refuse its promotion to mass finish."""

    audit_gm1_rk3_split_pair_without_live_lineage(
        prior_completed_rk2_token,
        reduction_token,
        finish_token,
    )
    raise RuntimeError(
        "GM1-RK3-SPLIT-MASS-FINISH-LINEAGE: exact local begin/finish tokens "
        "bind one stream and pointer inventory, but do not prove that final "
        "momentum boundaries and live SGS-TKE RK3 were enqueued between them, "
        "do not bind eqtset2 remove/snapshot/EOS or post-output hashes, and "
        "cannot emit a mass-finish or completed-RK3 token"
    )


def refuse_cuda_apmasscon_launch_as_complete_rk3_mass_finish(
    prior_completed_rk2_token: Gm1TiledCompletedRkStageToken,
    launch: object,
) -> NoReturn:
    """Reject a raw local correction as broader live mass-finish evidence."""

    audit = audit_gm1_rk3_mass_finish_source_contract(
        prior_completed_rk2_token
    )
    if type(launch) is not CudaApMassConLaunch:
        raise TypeError("launch must be the exact CudaApMassConLaunch type")
    receipt = launch.receipt
    if not isinstance(receipt, Mapping):
        raise TypeError("raw apmasscon launch receipt must be a mapping")
    launch_record = receipt.get("launch")
    if (
        receipt.get("schema") != CUDA_APMASSCON_SCHEMA
        or receipt.get("claim_limit") != CUDA_APMASSCON_CLAIM_LIMIT
        or not isinstance(launch_record, Mapping)
    ):
        raise ValueError("raw apmasscon launch identity or claim changed")
    stream = launch_record.get("stream_pointer")
    if (
        isinstance(stream, bool)
        or not isinstance(stream, int)
        or stream <= 0
        or stream != audit["stream_pointer"]
    ):
        raise ValueError("raw apmasscon launch belongs to another CUDA stream")
    raise RuntimeError(
        "GM1-RK3-MASS-FINISH-LINEAGE: raw CudaApMassConLaunch proves only "
        "the local correction operator; it does not bind completed RK2, "
        "fresh RK3 dynamics, eqtset2 remove/snapshot/EOS, source-ordered "
        "reduction/finish placement, TKE RK3, or post-output hashes; no "
        "mass-finish or completed-RK3 token emitted"
    )


__all__ = (
    "AUTHORITY_SOLVE2_SHA256",
    "AUTHORITY_SOUND_SHA256",
    "AUTHORITY_TARBALL_SHA256",
    "CUDA_GM1_RK23_CONTINUATION_ENTRY_AUDIT_SCHEMA",
    "CUDA_GM1_RK23_CONTINUATION_ENTRY_AUDIT_STATUS",
    "CUDA_GM1_RK3_MASS_FINISH_AUDIT_SCHEMA",
    "CUDA_GM1_RK3_MASS_FINISH_AUDIT_STATUS",
    "CUDA_GM1_RK3_INTERPOSED_PREFIX_AUDIT_SCHEMA",
    "CUDA_GM1_RK3_INTERPOSED_PREFIX_AUDIT_STATUS",
    "CUDA_GM1_RK3_SPLIT_PAIR_AUDIT_SCHEMA",
    "CUDA_GM1_RK3_SPLIT_PAIR_AUDIT_STATUS",
    "GM1_RK23_COMMON_ENTRY_SOURCE_ORDER",
    "GM1_RK23_CONTINUATION_CLAIM_LIMIT",
    "GM1_RK23_CONTINUATION_SOURCE_ANCHORS",
    "GM1_RK23_REQUIRED_FRESH_COMPONENTS",
    "GM1_RK23_SELECTED_PROFILE",
    "GM1_RK2_STAGE_TAIL_SOURCE_ORDER",
    "GM1_RK3_MASS_FINISH_CLAIM_LIMIT",
    "GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT",
    "GM1_RK3_MASS_FINISH_OUTPUT_KEYS",
    "GM1_RK3_MASS_FINISH_OUTPUT_SHAPES",
    "GM1_RK3_MASS_FINISH_REQUIRED_LINEAGE",
    "GM1_RK3_MASS_FINISH_SOURCE_ANCHORS",
    "GM1_RK3_MASS_FINISH_SOURCE_ORDER",
    "GM1_RK3_STAGE_TAIL_SOURCE_ORDER",
    "audit_gm1_rk23_continuation_entry",
    "audit_gm1_rk3_mass_finish_source_contract",
    "audit_gm1_rk3_local_interposed_prefix_fail_closed",
    "audit_gm1_rk3_split_pair_without_live_lineage",
    "refuse_cuda_apmasscon_launch_as_complete_rk3_mass_finish",
    "refuse_gm1_rk3_split_pair_as_mass_finish_without_live_lineage",
    "refuse_gm1_rk3_local_interposed_prefix_as_completed_stage",
    "refuse_gm1_rk23_continuation_without_fresh_runtime_lineage",
)
