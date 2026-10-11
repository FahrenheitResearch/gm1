from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import pytest

import cm1core.cuda_gm1_tilestream_post_acoustic_getdivx as cadence
from cm1core.cuda_gm1_tilestream_solve1_tke_rhs import (
    CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
    Gm1TiledSolve1TkeRhsToken,
    Gm1TiledSolve2PostAcousticFluxToken,
    require_gm1_tiled_solve2_tke_rk1,
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


def _solve1() -> Gm1TiledSolve1TkeRhsToken:
    return Gm1TiledSolve1TkeRhsToken(
        _tokenize(
            {
                "schema": CUDA_GM1_TILESTREAM_SOLVE1_TKE_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_TILESTREAM_SOLVE1_TKE_RHS",
                "completed_nrk": 0,
                "stream_pointer": 73,
            }
        )
    )


def _flux(*, nrk: int = 1) -> Gm1TiledSolve2PostAcousticFluxToken:
    return Gm1TiledSolve2PostAcousticFluxToken(
        _tokenize(
            {
                "schema": cadence.CUDA_GM1_POST_ACOUSTIC_FLUX_TOKEN_SCHEMA,
                "status": "ADMITTED_LIVE_POST_ACOUSTIC_MASS_FLUX",
                "nrk": nrk,
                "stream_pointer": 73,
                "source_order": [
                    "sound returns averaged velocity",
                    "post-sound getdivx converts it to mass flux and divx",
                ],
            }
        )
    )


def test_authority_pins_the_late_post_sound_getdivx_branch() -> None:
    assert cadence.AUTHORITY_SOLVE2_SHA256 == (
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
    )
    assert cadence.AUTHORITY_SOUND_SHA256 == (
        "5fe055e22d106e91b5bdbe35ad7c72b2bbbcfb5b01733126703fdc7a5977da54"
    )
    assert cadence.AUTHORITY_MISCLIBS_SHA256 == (
        "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1"
    )
    assert "src/solve2.F:1303-1307" in cadence.AUTHORITY_ANCHORS
    assert "src/misclibs.F:222-370" in cadence.AUTHORITY_ANCHORS


def test_exact_rk1_post_acoustic_token_unlocks_only_rk1() -> None:
    assert require_gm1_tiled_solve2_tke_rk1(_solve1(), _flux()) is None
    with pytest.raises(ValueError, match="cadence or stream"):
        require_gm1_tiled_solve2_tke_rk1(_solve1(), _flux(nrk=2))


def test_velocity_token_is_hash_bound_and_nrk1_only() -> None:
    receipt = {
        "schema": cadence.CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
        "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
        "nrk": 1,
        "stream_pointer": 73,
    }
    token = cadence.Gm1PostSoundVelocityToken(_tokenize(receipt))
    assert cadence._require_velocity_token(token)["nrk"] == 1
    changed = dict(token.receipt)
    changed["nrk"] = 2
    with pytest.raises(ValueError, match="token changed"):
        cadence._require_velocity_token(
            cadence.Gm1PostSoundVelocityToken(MappingProxyType(changed))
        )


def test_five_meter_inventory_refuses_the_global_acoustic_dependency() -> None:
    inventory = cadence.five_meter_post_acoustic_getdivx_inventory()
    assert inventory["domain"] == [75, 3000, 3000]
    assert inventory["tile_count"] == 256
    assert inventory["getdivx_tile_streamable"] is True
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["post_sound_velocity_token_available_at_5m"] is False
    assert inventory["tke_rk1_available_at_5m"] is False
    assert inventory["five_meter_meteorological_product"] is False

