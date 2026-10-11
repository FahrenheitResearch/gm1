from __future__ import annotations

import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import pytest

import cm1core.cuda_gm1_tilestream_rk3_interposed_prefix as prefix_module
import cm1core.cuda_gm1_tilestream_rk3_interposed_tke as interposed
from cm1core.cuda_gm1_tilestream import build_gm1_open_tile_plan
from cm1core.cuda_gm1_tilestream_rk3_interposed_prefix import (
    CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_SCHEMA,
    CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_STATUS,
    GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT,
    SHARED_THERMODYNAMIC_ORDER,
    SOURCE_ORDER as PREFIX_SOURCE_ORDER,
    CudaGm1Rk3InterposedPrefixLaunch,
    Gm1Rk3InterposedPrefixToken,
)
from cm1core.cuda_gm1_tilestream_rk3_interposed_tke import (
    CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_SCHEMA,
    CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_SCHEMA,
    CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_STATUS,
    GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT,
    SOURCE_ORDER,
    Gm1Rk3InterposedPrefixTkeToken,
    finish_cuda_gm1_rk3_tke_after_interposed_prefix,
    launch_cuda_gm1_rk3_tke_after_interposed_prefix,
    refuse_gm1_rk3_interposed_tke_as_mass_finish_or_completed_stage,
    require_gm1_rk3_interposed_prefix_tke_token,
)
from cm1core.cuda_gm1_tilestream_tke_rk23 import (
    CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
    GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT,
    CudaGm1TiledTkeRk23Launch,
    Gm1TiledTkeRk23Token,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _tokenize(receipt: dict[str, object]) -> MappingProxyType:
    receipt["operation_token_sha256"] = hashlib.sha256(
        _canonical(receipt)
    ).hexdigest()
    return MappingProxyType(receipt)


class _Stream:
    def __init__(self, pointer: int = 17) -> None:
        self.ptr = pointer
        self.log: list[str] = []

    def wait_event(self, event: object) -> None:
        assert getattr(event, "ptr", None) == 103
        self.log.append("wait-prefix")


def _prefix(*, compact: bool = False) -> CudaGm1Rk3InterposedPrefixLaunch:
    stream = _Stream()
    ready = SimpleNamespace(ptr=103)
    components = {
        "eqtset2_remove_eos": "a" * 64,
        "apmasscon_reduction": "b" * 64,
        "momentum_boundary": "c" * 64,
    }
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_INTERPOSED_PREFIX_TOKEN_STATUS,
        "nrk": 3,
        "nrkmax": 3,
        "stream_pointer": 17,
        "domain": {"nx": 300, "ny": 300, "nk": 75},
        "source_order": list(PREFIX_SOURCE_ORDER),
        "component_operation_token_sha256": components,
        "predecessor_operation_token_sha256": {
            "apmasscon_reduction": components["eqtset2_remove_eos"],
            "momentum_boundary": components["apmasscon_reduction"],
        },
        "shared_thermodynamic_carrier_order": list(
            SHARED_THERMODYNAMIC_ORDER
        ),
        "shared_thermodynamic_pointer_inventory_sha256": "d" * 64,
        "momentum_boundary_pointer_inventory_sha256": "e" * 64,
        "event_pointer_order": [101, 102, 103],
        "exact_stream_object_identity_bound": True,
        "shared_thermodynamic_carrier_object_identity_bound": True,
        "source_order_enqueue_chain_bound": True,
        "component_negative_claims_preserved": True,
        "fresh_rk3_scalar_moist_lineage_bound": False,
        "prior_completed_rk2_scientific_lineage_bound": False,
        "invalid_counter_host_checked": False,
        "rk3_tke_bound": False,
        "apmasscon_finish_bound": False,
        "post_output_sha256_bound": False,
        "mass_finish_source_order_complete": False,
        "completed_rk3_token_emitted": False,
        "solve3_entered": False,
        "history_checkpoint_emitted": False,
        "compact_validation_only": compact,
        "claim_limit": GM1_RK3_INTERPOSED_PREFIX_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = prefix_module._operation_token(receipt)
    token = Gm1Rk3InterposedPrefixToken(MappingProxyType(receipt))
    eqtset = SimpleNamespace(stream=stream)
    reduction = SimpleNamespace(stream=stream)
    boundary = SimpleNamespace(stream=stream, ready_event=ready)
    return CudaGm1Rk3InterposedPrefixLaunch(
        ready,
        stream,
        eqtset,  # type: ignore[arg-type]
        reduction,  # type: ignore[arg-type]
        boundary,  # type: ignore[arg-type]
        token,
    )


def _finished_tke_token() -> Gm1TiledTkeRk23Token:
    return Gm1TiledTkeRk23Token(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_TKE_RK23_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_TKE_RK_STAGE",
                "completed_nrk": 3,
                "held_tke_state_token_sha256": "1" * 64,
                "prior_tke_stage_token_sha256": "2" * 64,
                "completed_prior_rk_stage_token_sha256": "3" * 64,
                "post_acoustic_flux_token_sha256": "4" * 64,
                "stage_identity_sha256": "5" * 64,
                "tile_plan_sha256": "6" * 64,
                "tile_count": 1,
                "tile_source_tokens_sha256": "7" * 64,
                "tile_operation_tokens_sha256": "8" * 64,
                "stream_pointer": 17,
                "host_scatter_exact_once": True,
                "output_tke3d_shape": [76, 300, 300],
                "output_tke3d_words": 76 * 300 * 300,
                "output_tke3d_sha256": "9" * 64,
                "final_bcw_applied": True,
                "ready_for": (
                    "solve3 TKE commit only after exact completed broader RK3"
                ),
                "structural_refusals": [
                    "TKE-only assertion of broader RK completion",
                    "reuse of an earlier-stage post-acoustic flux token",
                    "solve3 tkea commit before exact completed broader RK3",
                    "global psolver tilestream claim",
                    "MPI exchange or open-history production",
                    "full forecast or five-metre meteorological product",
                ],
                "claim_limit": GM1_TILESTREAM_TKE_RK23_CLAIM_LIMIT,
            }
        )
    )


def test_tile_wrapper_enqueues_prefix_wait_before_underlying_tke(monkeypatch) -> None:
    prefix = _prefix()
    plan = build_gm1_open_tile_plan(
        nx=300,
        ny=300,
        tile_nx=96,
        tile_ny=96,
    )
    spec = plan.specs[0]
    underlying_ready = SimpleNamespace(ptr=104)

    def fake_launch(*args, **kwargs):
        assert prefix.stream.log == ["wait-prefix"]
        prefix.stream.log.append("launch-tke")
        receipt: dict[str, object] = {
            "nrk": 3,
            "tile": spec.record(),
            "stream_pointer": 17,
        }
        receipt["operation_token_sha256"] = interposed._operation_token(receipt)
        return CudaGm1TiledTkeRk23Launch(
            underlying_ready,
            None,
            None,
            None,  # type: ignore[arg-type]
            MappingProxyType(receipt),
        )

    monkeypatch.setattr(
        interposed,
        "launch_cuda_gm1_tiled_tke_rk23",
        fake_launch,
    )
    launch = launch_cuda_gm1_rk3_tke_after_interposed_prefix(
        prefix,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        {},
        spec=spec,
        metrics={},
        compiled=object(),  # type: ignore[arg-type]
    )
    assert prefix.stream.log == ["wait-prefix", "launch-tke"]
    assert launch.token.receipt["schema"] == (
        CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_SCHEMA
    )
    assert launch.token.receipt["explicit_prefix_event_wait_enqueued"] is True
    assert launch.token.receipt["predecessor_operation_token_sha256"] == (
        prefix.token.receipt["operation_token_sha256"]
    )


def test_finish_binds_exact_tke_successor_and_keeps_mass_finish_false(
    monkeypatch,
) -> None:
    prefix = _prefix()
    plan = build_gm1_open_tile_plan(
        nx=300,
        ny=300,
        tile_nx=96,
        tile_ny=96,
    )
    wrappers: list[interposed.CudaGm1Rk3InterposedTkeTileLaunch] = []
    for index, spec in enumerate(plan.specs):
        underlying_ready = SimpleNamespace(ptr=104 + index)
        tke_receipt: dict[str, object] = {
            "nrk": 3,
            "tile": spec.record(),
            "stream_pointer": 17,
        }
        tke_receipt["operation_token_sha256"] = interposed._operation_token(
            tke_receipt
        )
        underlying = CudaGm1TiledTkeRk23Launch(
            underlying_ready,
            None,
            None,
            None,  # type: ignore[arg-type]
            MappingProxyType(tke_receipt),
        )
        tile_receipt: dict[str, object] = {
            "schema": interposed.CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_SCHEMA,
            "status": interposed.CUDA_GM1_RK3_INTERPOSED_TKE_TILE_TOKEN_STATUS,
            "nrk": 3,
            "tile": spec.record(),
            "stream_pointer": 17,
            "prefix_operation_token_sha256": prefix.token.receipt[
                "operation_token_sha256"
            ],
            "prefix_ready_event_pointer": 103,
            "tke_tile_operation_token_sha256": tke_receipt[
                "operation_token_sha256"
            ],
            "tke_tile_ready_event_pointer": 104 + index,
            "predecessor_operation_token_sha256": prefix.token.receipt[
                "operation_token_sha256"
            ],
            "explicit_prefix_event_wait_enqueued": True,
            "exact_stream_object_identity_bound": True,
            "source_order_edge_bound": True,
            "apmasscon_finish_bound": False,
            "post_output_sha256_bound": False,
            "completed_rk3_token_emitted": False,
            "claim_limit": GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT,
        }
        tile_receipt["operation_token_sha256"] = interposed._operation_token(
            tile_receipt
        )
        wrappers.append(
            interposed.CudaGm1Rk3InterposedTkeTileLaunch(
                underlying_ready,
                prefix,
                underlying,
                interposed.Gm1Rk3InterposedTkeTileToken(
                    MappingProxyType(tile_receipt)
                ),
            )
        )
    tke_token = _finished_tke_token()
    monkeypatch.setattr(
        interposed,
        "finish_cuda_gm1_tiled_tke_rk23",
        lambda *args, **kwargs: tke_token,
    )
    token = finish_cuda_gm1_rk3_tke_after_interposed_prefix(
        prefix,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        wrappers,
        plan=plan,
        output_tke3d=object(),
    )
    receipt = require_gm1_rk3_interposed_prefix_tke_token(
        token,
        prefix_token=prefix.token,
        tke_rk3_token=tke_token,
    )
    assert receipt["schema"] == CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_SCHEMA
    assert receipt["status"] == CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_STATUS
    assert receipt["source_order"] == list(SOURCE_ORDER)
    assert receipt["rk3_tke_bound"] is True
    assert receipt["apmasscon_finish_bound"] is False
    assert receipt["post_output_sha256_bound"] is False
    assert receipt["completed_rk3_token_emitted"] is False
    with pytest.raises(
        RuntimeError,
        match="GM1-RK3-INTERPOSED-TKE-NOT-MASS-FINISH",
    ):
        refuse_gm1_rk3_interposed_tke_as_mass_finish_or_completed_stage(token)


def test_compact_prefix_and_tampered_aggregate_are_refused(monkeypatch) -> None:
    compact = _prefix(compact=True)
    plan = build_gm1_open_tile_plan(
        nx=300,
        ny=300,
        tile_nx=96,
        tile_ny=96,
    )
    with pytest.raises(ValueError, match="live 300x300x75 prefix"):
        launch_cuda_gm1_rk3_tke_after_interposed_prefix(
            compact,
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            {},
            spec=plan.specs[0],
            metrics={},
            compiled=object(),  # type: ignore[arg-type]
        )
    receipt: dict[str, object] = {
        "schema": CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_SCHEMA,
        "status": CUDA_GM1_RK3_INTERPOSED_TKE_TOKEN_STATUS,
        "completed_nrk": 3,
        "stream_pointer": 17,
        "stage_identity_sha256": "1" * 64,
        "prefix_operation_token_sha256": "2" * 64,
        "tke_rk3_operation_token_sha256": "3" * 64,
        "tke_rk3_output_sha256": "4" * 64,
        "predecessor_operation_token_sha256": {"tke_rk3": "f" * 64},
        "tile_wrapper_operations_sha256": "5" * 64,
        "tile_count": 1,
        "source_order": list(SOURCE_ORDER),
        "source_order_enqueue_chain_bound": True,
        "exact_stream_object_identity_bound": True,
        "prefix_event_wait_bound_on_every_tile": True,
        "rk3_tke_bound": True,
        "apmasscon_finish_bound": False,
        "post_output_sha256_bound": False,
        "mass_finish_source_order_complete": False,
        "completed_rk3_token_emitted": False,
        "solve3_entered": False,
        "history_checkpoint_emitted": False,
        "claim_limit": GM1_RK3_INTERPOSED_TKE_CLAIM_LIMIT,
    }
    receipt["operation_token_sha256"] = interposed._operation_token(receipt)
    with pytest.raises(ValueError, match="identity, lineage, or claim"):
        require_gm1_rk3_interposed_prefix_tke_token(
            Gm1Rk3InterposedPrefixTkeToken(MappingProxyType(receipt))
        )
