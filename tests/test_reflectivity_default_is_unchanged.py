"""The existing default configuration is byte-for-byte what it was.

Adding a carrier to a history frame and two keys to a sealed document is the
kind of change that silently moves every hash a run has already recorded: the
canonical configuration ``sha256`` that travels inside GPU-run receipts as
``plan_sha256``, the ``config_sha256`` hashed into every published frame, the
frame's own carrier table and payload size, and the device-memory plan.

Every value pinned below was measured by running this repository at commit
6c6c9c4 -- before ``physics.output_dbz`` existed -- over the shipped examples
and one representative Cartesian geometry.  They are here so that "the default
is unchanged" is a measurement and not an assurance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cm1core.cartesian_history import cartesian_history_profile
from cm1core.config import load_config
from cm1core.cuda_cartesian_finish import cartesian_wsm6_scratch_reserve
from cm1core.gm1_runtime import RuntimeConfig
from cm1core.grid_geometry import CartesianGridGeometry


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

#: ``CM1Config.sha256`` of every distributed example at 6c6c9c4.
SEALED_EXAMPLE_SHA256 = {
    "cartesian-fifty-metre-fine-step.toml":
        "39c3e8198371318b55955f26c1cf94c83d831a1945d6d132544f9cebbfa6fcef",
    "cartesian-stretched-tornado-grade-grid.toml":
        "d4f0b4a9fa9bf371a6e726067fa4ce4998925cbdd876831fbad6365212a9d811",
    "cartesian-supercell-attempt.toml":
        "f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b",
    "cartesian-supercell-surface-drag.toml":
        "5bc41131130449a709a673f780c47e844187e06d105af339391930d7bf8f0a77",
    "cartesian-supercell-translating-frame.toml":
        "77288cd92837908d70baedebc1aa914c2dc6a5bdb1f90d3d8255cfed15a545ca",
    "cartesian-uniform-grid.toml":
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48",
    "cartesian-wk-analytic-sounding-shear.toml":
        "e6ca95ed63f9555381a4ec8d04b93cf4cfff324119b0fb0df1ed9d14e696cee7",
    "dry-psolver2-bootstrap.toml":
        "0394dea1f77722e54fe6b1b5a07fe56037cda84800ef870b9a269566e6043340",
    "official-supercell-native-context.toml":
        "33761ba3ef1e35ce8dd6fa3fa126c4ed2437bdbe49d3b9e73adb2fa2bcf92b8c",
    "official-supercell-product.toml":
        "b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038",
    "real-atmosphere-100m-product.toml":
        "07c7dde701491ce71a70cec8c56de2cd50ed84574cb098250815d76eaa0c9c00",
}

#: One representative Cartesian geometry, measured at 6c6c9c4.
SEALED_GEOMETRY = (64, 64, 48, 1000.0, 1000.0, 500.0)
SEALED_TILE = (32, 32)
SEALED_RUNTIME_RECORD_SHA256 = (
    "6602a2bedc07005901b582d3ca353e1b6a3b778d07c986d89a8d123d9929b8a9"
)
#: The keys ``RuntimeConfig.record()`` had at 6c6c9c4.  Later reviewed commits
#: only ADDED keys (&param9 output_selection 59b359b, moisture_species and
#: dense_ice 4b3f004, pdefq launch 96a8e47, surface/radiation/cumulus, ...);
#: the sealed hash is therefore checked over exactly these keys, which still
#: hash to the value measured at 6c6c9c4, so no pre-existing word has moved.
SEALED_RUNTIME_RECORD_KEYS = (
    "acoustic_authority", "acoustic_small_step_limit_s", "acoustic_small_step_s",
    "acoustic_substeps", "advective_courant_authority", "advective_courant_stop",
    "advective_courant_stop_speed_m_s", "advective_courant_target",
    "advective_courant_target_speed_m_s", "bbc", "cecd",
    "cm1_diagnostic_acoustic_substeps", "cnstcd", "damping_depth_m",
    "damping_rate_s_inv", "dt_s", "grid", "isfcflx", "lower_boundary_authority",
    "memory_budget_bytes", "microphysics", "moving_frame_authority", "sfcmodel",
    "tile_shape_yx", "u_move_m_s", "v_move_m_s",
)
#: Re-sealed when gap2/outputfmt landed the reduced carrier profile.  The
#: frame the default publishes is unchanged -- same twelve carriers, same
#: payload bytes, same staging words, still no reflectivity group, all
#: asserted below -- but ``CartesianHistoryProfile.record()`` gained four
#: descriptive fields (``carrier_profile``, ``reconstructible_carriers``,
#: ``reduced_profile_reader_cost``, ``admission_host_bytes``) and
#: ``maximum_forecast_payload_bytes`` rose from the 2 GiB container guess to
#: the 16 GiB host-admission ceiling the Rust envelope enforces
#: (rust/cm1-state/src/forecast.rs:52).  The hash moved because the receipt
#: says more, not because the default frame changed.
#:
#: Re-sealed again for 59b359b/386071a: CM1's &param9 default selection
#: appends 22 carriers AFTER the twelve (the default frame grew 12 -> 34 and
#: its payload 12,689,408 -> 14,476,608 bytes), th3d's meaning string was
#: corrected to "perturbation", and each carrier record gained
#: ``horizontal_profile_column``.  The twelve original carriers keep their
#: names, order, shapes and bytes, asserted below.
SEALED_PROFILE_RECORD_SHA256 = (
    "f1af481cbeb409a7745d3a5d73bee2ed4b8ae7ad7e9ab28fa9d8a6d18f4e92cd"
)
SEALED_CARRIER_NAMES = (
    "closure.tkea", "pp3d", "prs", "q3d", "rf", "rho", "terrain_msl",
    "th3d", "u3d", "v3d", "w3d", "zf",
)
SEALED_PAYLOAD_NBYTES = 12689408
#: 386071a: the payload of the whole default frame once &param9 appends.
DEFAULT_FRAME_PAYLOAD_NBYTES = 14476608
SEALED_STAGING_WORDS = 800768
SEALED_SCRATCH_RESERVE = {
    "additional_reserve_bytes": 1585152,
    "allocator_alignment_bytes": 512,
    "block_nx": 32,
    "block_ny": 32,
    "component_scratch_reserve_bytes": 1183744,
    "expression_temporary_reserve_bytes": 401408,
    "levels": 48,
    # 4b3f004 names the component the reserve was priced for; every byte
    # count above is unchanged.
    "component_id": "wsm6-mp6",
    "component_mass_scratch_volumes": 5,
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
        ).encode("utf-8")
    ).hexdigest()


def _default_runtime_config() -> RuntimeConfig:
    nx, ny, nz, dx, dy, dz = SEALED_GEOMETRY
    return RuntimeConfig(
        CartesianGridGeometry(nx, ny, nz, dx, dy, dz),
        tile_nx=SEALED_TILE[0], tile_ny=SEALED_TILE[1],
    )


@pytest.mark.parametrize("name", sorted(SEALED_EXAMPLE_SHA256))
def test_every_shipped_example_keeps_its_sealed_canonical_identity(name):
    """A document that declares no reflectivity projects none of its keys."""

    config = load_config(EXAMPLES / name)
    assert config.sha256 == SEALED_EXAMPLE_SHA256[name]
    physics = config.canonical_dict()["physics"]
    assert set(physics) == {"component", "equation_set"}


def test_the_default_runtime_contract_record_is_unchanged():
    config = _default_runtime_config()
    assert config.output_dbz is False
    assert config.dbz_level_m is None
    assert config.reflectivity_level_index is None
    record = config.record()
    assert not [key for key in record if "dbz" in key or "reflectivity" in key]
    assert set(SEALED_RUNTIME_RECORD_KEYS) <= set(record)
    sealed = {key: record[key] for key in SEALED_RUNTIME_RECORD_KEYS}
    assert _canonical_sha256(sealed) == SEALED_RUNTIME_RECORD_SHA256


def test_the_default_history_frame_carries_exactly_the_twelve_it_always_did():
    profile = cartesian_history_profile(_default_runtime_config())
    # 386071a: &param9 carriers append after the twelve, so the twelve keep
    # their order and their bytes; the frame as a whole is larger.
    original = profile.carriers[: len(SEALED_CARRIER_NAMES)]
    assert tuple(carrier.name for carrier in original) == SEALED_CARRIER_NAMES
    assert sum(carrier.nbytes for carrier in original) == SEALED_PAYLOAD_NBYTES
    assert profile.payload_nbytes == DEFAULT_FRAME_PAYLOAD_NBYTES
    assert profile.staging_words == SEALED_STAGING_WORDS
    assert _canonical_sha256(profile.record()) == SEALED_PROFILE_RECORD_SHA256
    assert not any(carrier.group == "reflectivity" for carrier in profile.carriers)


def test_the_default_device_memory_plan_is_unchanged():
    reserve = cartesian_wsm6_scratch_reserve(_default_runtime_config())
    assert dict(reserve) == SEALED_SCRATCH_RESERVE


def test_the_default_wsm6_block_still_declines_reflectivity_in_the_same_words():
    """The tiled block's receipt is unchanged for a call that is not due.

    ``refusals.reflectivity`` is a statement about what the call produced, not
    a guard: a block that was not an output step produced no reflectivity and
    still says so in exactly the place, and with exactly the value, it always
    has, so the operation token of every non-output block is what it was.
    """

    import inspect

    from cm1core import cuda_gm1_tilestream_wsm6 as module

    source = inspect.getsource(module.launch_cuda_gm1_tiled_column_block)
    assert '"reflectivity": not refl_10cm_due' in source
    # The two reflectivity blocks are added to the record only on a due call,
    # so a not-due record hashes to exactly the bytes it did before.
    assert 'record["refl_10cm_due"] = True' in source
    assert 'if refl_10cm_due:' in source
    signature = inspect.signature(module.launch_cuda_gm1_tiled_column_block)
    assert signature.parameters["refl_10cm_due"].default is False
