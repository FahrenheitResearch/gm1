from __future__ import annotations

import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import cm1core.cuda_gm1_tilestream_scalar_post_sound_getdiv as scalar_getdiv
from cm1core.cuda_gm1_tilestream import (
    Gm1TileLocalSourceBinding,
    Gm1TileSpec,
    build_gm1_open_tile_plan,
)
from cm1core.cuda_gm1_tilestream_post_acoustic_getdivx import (
    CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
    Gm1PostSoundVelocityToken,
)
from cm1core.cuda_moist_stage_bridge import CudaMoistStageBridgeState


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


def _velocity() -> Gm1PostSoundVelocityToken:
    return Gm1PostSoundVelocityToken(
        _tokenize(
            {
                "schema": CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA,
                "status": "ADMITTED_REAL_RK1_POST_SOUND_AVERAGED_VELOCITY_SNAPSHOT",
                "nrk": 1,
                "stage_identity_sha256": "a" * 64,
                "stream_pointer": 73,
            }
        )
    )


def _thermodynamics() -> scalar_getdiv.Gm1ScalarPostSoundThermodynamicToken:
    thterm = np.zeros((75, 300, 300), dtype="<f4")
    thterm.flags.writeable = False
    receipt = {
        "schema": scalar_getdiv.CUDA_GM1_SCALAR_POST_SOUND_THERMODYNAMIC_TOKEN_SCHEMA,
        "status": "ADMITTED_REAL_RK1_POST_SOUND_THERMODYNAMIC_SNAPSHOT",
        "nrk": 1,
        "stage_identity_sha256": "a" * 64,
        "post_sound_velocity_token_sha256": _velocity().receipt[
            "operation_token_sha256"
        ],
        "stream_pointer": 73,
        "thterm_shape": [75, 300, 300],
        "thterm_sha256": hashlib.sha256(thterm.tobytes()).hexdigest(),
    }
    return scalar_getdiv.Gm1ScalarPostSoundThermodynamicToken(
        _tokenize(receipt), thterm
    )


def _small_spec() -> Gm1TileSpec:
    return Gm1TileSpec(
        index=0,
        ty=0,
        tx=0,
        nx=300,
        ny=300,
        tile_nx=6,
        tile_ny=6,
        halo=3,
        i0=0,
        i1=6,
        j0=0,
        j1=6,
        ci0=0,
        cj0=0,
        cnx=12,
        cny=12,
    )


def test_authority_and_dependency_sources_are_exact() -> None:
    assert scalar_getdiv.AUTHORITY_ANCHORS == (
        "src/solve2.F:394-412",
        "src/sound.F:806-832",
        "src/solve2.F:1277-1307",
        "src/misclibs.F:108-214",
        "src/solve2.F:1313-1343",
    )
    assert scalar_getdiv.AUTHORITY_SOLVE2_SHA256 == (
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734"
    )
    assert scalar_getdiv.AUTHORITY_MISCLIBS_SHA256 == (
        "2f8181c6de3a519fb55423911c292c56c2fb4ae0a75feeff7c0b27a73b82e7e1"
    )
    assert scalar_getdiv.GM1_CHILD_STRETCHED_MOIST_SOURCE_SHA256 == (
        "9e899ae56709ec406b836bba0bcac8941dcab3b69ef690d555a4f65ba2d3bbdf"
    )


def test_checkpoint_requires_live_terms_buoyancy_and_same_stream(monkeypatch) -> None:
    thterm = np.arange(75 * 300 * 300, dtype="<f4").reshape(75, 300, 300)
    state = CudaMoistStageBridgeState(
        300,
        300,
        75,
        True,
        object(),
        thterm,
        object(),
        object(),
        object(),
        SimpleNamespace(ptr=73),
        0,
        SimpleNamespace(target_arch="86"),
        object(),
        terms_ready=True,
        buoyancy_applied=True,
        post_acoustic_ready=False,
    )
    real_import = scalar_getdiv.importlib.import_module

    def fake_import(name: str):
        if name == "cupy":
            return SimpleNamespace(asnumpy=lambda value: value)
        return real_import(name)

    monkeypatch.setattr(scalar_getdiv.importlib, "import_module", fake_import)
    token = scalar_getdiv.checkpoint_cuda_gm1_scalar_post_sound_thermodynamics(
        _velocity(), state
    )
    assert token.receipt["status"] == (
        "ADMITTED_REAL_RK1_POST_SOUND_THERMODYNAMIC_SNAPSHOT"
    )
    assert token.receipt["stream_pointer"] == 73
    assert token.thterm.flags.writeable is False
    assert token.receipt["thterm_sha256"] == hashlib.sha256(
        thterm.tobytes()
    ).hexdigest()

    state.buoyancy_applied = False
    with pytest.raises(ValueError, match="fresh terms/buoyancy"):
        scalar_getdiv.checkpoint_cuda_gm1_scalar_post_sound_thermodynamics(
            _velocity(), state
        )


def test_thermodynamic_payload_and_velocity_hash_are_immutable() -> None:
    token = _thermodynamics()
    receipt, value = scalar_getdiv._require_thermodynamic_token(token)
    assert receipt["nrk"] == 1
    assert value.flags.writeable is False

    changed = dict(token.receipt)
    changed["thterm_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="token or payload changed"):
        scalar_getdiv._require_thermodynamic_token(
            scalar_getdiv.Gm1ScalarPostSoundThermodynamicToken(
                MappingProxyType(changed), token.thterm
            )
        )


def test_tile_fill_binds_exact_velocity_and_thterm(monkeypatch) -> None:
    token = _thermodynamics()
    spec = _small_spec()
    binding = Gm1TileLocalSourceBinding(
        SimpleNamespace(nz=75),
        MappingProxyType({}),
        "theta",
        str(token.receipt["post_sound_velocity_token_sha256"]),
    )
    shapes = scalar_getdiv.tile_scalar_post_sound_getdiv_host_shapes(
        tile_nx=spec.tile_nx, tile_ny=spec.tile_ny, nz=75
    )
    target = {name: np.empty(shape, dtype="<f4") for name, shape in shapes.items()}

    def fake_base(_binding, _spec, base):
        for value in base.values():
            value.fill(np.float32(2.0))
        return {
            "source_identity_sha256": binding.source_identity_sha256,
            "scalar": "theta",
            "tile": 0,
            "compute_mass_yx": [0, 12, 0, 12],
            "full_domain_transient_stage_store_used": False,
        }

    monkeypatch.setattr(scalar_getdiv, "fill_gm1_tile_local_source", fake_base)
    receipt = scalar_getdiv.fill_gm1_tiled_scalar_post_sound_getdiv_source(
        token, binding, spec, target
    )
    assert receipt["tile"] == 0
    assert receipt["thermodynamic_token_sha256"] == token.receipt[
        "operation_token_sha256"
    ]
    assert not np.any(target["divergence"].view(np.uint32))
    assert not np.any(target["theta_divergence_term"].view(np.uint32))
    assert receipt["thterm_tile_sha256"] == hashlib.sha256(
        target["thterm"].tobytes()
    ).hexdigest()

    wrong = Gm1TileLocalSourceBinding(
        SimpleNamespace(nz=75), MappingProxyType({}), "theta", "f" * 64
    )
    with pytest.raises(ValueError, match="another post-sound snapshot"):
        scalar_getdiv.fill_gm1_tiled_scalar_post_sound_getdiv_source(
            token, wrong, spec, target
        )


def test_finish_requires_every_tile_exact_once_and_emits_distinct_token() -> None:
    token = _thermodynamics()
    plan = build_gm1_open_tile_plan(nx=300, ny=300, tile_nx=150, tile_ny=150)
    launches = []
    for spec in plan.specs:
        receipt = {
            "schema": scalar_getdiv.CUDA_GM1_TILESTREAM_SCALAR_POST_SOUND_GETDIV_SCHEMA,
            "status": "ENQUEUED_LIVE_RK1_TILE_SCALAR_POST_SOUND_GETDIV",
            "nrk": 1,
            "tile": spec.record(),
            "thermodynamic_token_sha256": token.receipt[
                "operation_token_sha256"
            ],
            "stream_pointer": 73,
        }
        launches.append(
            scalar_getdiv.CudaGm1TiledScalarPostSoundGetdivLaunch(
                SimpleNamespace(), SimpleNamespace(), _tokenize(receipt)
            )
        )
    outputs = {
        "divergence": np.zeros((75, 300, 300), dtype="<f4"),
        "theta_divergence_term": np.ones((75, 300, 300), dtype="<f4"),
    }
    result = scalar_getdiv.finish_cuda_gm1_tiled_scalar_post_sound_getdiv(
        token, launches, plan=plan, outputs=outputs
    )
    assert result.receipt["schema"] == (
        scalar_getdiv.CUDA_GM1_TILESTREAM_SCALAR_THETA_DIVERGENCE_TOKEN_SCHEMA
    )
    assert result.receipt["host_scatter_exact_once"] is True
    assert result.receipt["post_acoustic_mass_flux_token_bound"] is False
    assert result.receipt["scalar_rk1_performed"] is False
    assert result.receipt["five_meter_meteorological_product"] is False

    with pytest.raises(ValueError, match="every exact tile"):
        scalar_getdiv.finish_cuda_gm1_tiled_scalar_post_sound_getdiv(
            token, launches[:-1], plan=plan, outputs=outputs
        )


def test_five_meter_inventory_is_streamable_but_not_executable() -> None:
    inventory = scalar_getdiv.five_meter_scalar_post_sound_getdiv_inventory()
    assert inventory["domain"] == [75, 3000, 3000]
    assert inventory["tile_count"] == 256
    assert inventory["one_device_tile_bytes"] == 74_814_576
    assert inventory["double_device_tile_bytes"] == 149_629_152
    assert inventory["getdiv_tile_streamable"] is True
    assert inventory["eqtset2_thterm_tile_streamable"] is True
    assert inventory["global_psolver_tile_streamable"] is False
    assert inventory["post_sound_velocity_token_available_at_5m"] is False
    assert inventory["scalar_rk1_available_at_5m"] is False
    assert inventory["five_meter_meteorological_product"] is False


def test_upstream_velocity_token_is_consumed_without_schema_mutation() -> None:
    token = _velocity()
    before = dict(token.receipt)
    scalar_getdiv._require_velocity_token(token)
    assert dict(token.receipt) == before
    assert token.receipt["schema"] == CUDA_GM1_POST_SOUND_VELOCITY_TOKEN_SCHEMA
