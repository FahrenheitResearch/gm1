"""The lane's live admission record is held to the code that enforces it.

``records/CM1-PROFILE-MANIFEST.json`` carries two Cartesian records and they
are different on purpose.

``cartesian_runtime.admitted_envelope`` is a **sealed identity**:
``cm1core.profiles.profile_identity`` deep-copies it into every canonical
document and ``CM1Config.sha256`` is that document's digest, which recorded
GPU-run receipts carry as ``plan_sha256``.  Widening the lane must therefore
never move it -- which is why it still reads ``dt_s = 0.5`` and a five-value
``acoustic_substeps`` list long after CM1's own ``misclibs.F:818-915`` step
derivation made both free.

``cartesian_runtime_admission`` is the **live** record.  A record nothing
checks goes stale, and a stale admission record is how a capability that
exists becomes a capability no lane can find -- so every field below is
compared with the module that actually refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cm1core.arwen_morrison_component import MORRISON_NDCNST_CM3
from cm1core.column_components import COLUMN_COMPONENT_SPECS
from cm1core.config import Lane, load_config
from cm1core.dense_ice import DEFAULT_IHAIL_BY_COMPONENT, IHAIL_VALUES
from cm1core.engine_capacity import WSM6_256_CAPACITY_PROFILE, WSM6_CAPACITY_LEVEL_BOUNDS
from cm1core.moisture_species import DEFAULT_COMPONENT_ID
from cm1core.profiles import cartesian_live_admission, profile_identity

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_the_live_component_rows_are_the_bound_component_specs() -> None:
    admission = cartesian_live_admission()["column_components"]
    assert admission["default"] == DEFAULT_COMPONENT_ID
    named = {key for key in admission if key != "default"}
    assert named == set(COLUMN_COMPONENT_SPECS)

    for component_id, spec in COLUMN_COMPONENT_SPECS.items():
        row = admission[component_id]
        assert row["cm1_ptype"] == spec.profile.cm1_ptype, component_id
        assert row["arwen_mp_physics"] == spec.mp_physics, component_id
        assert row["numq"] == spec.profile.numq, component_id
        assert row["levels"] == [spec.minimum_levels, spec.maximum_levels], component_id
        if spec.capacity_profile_supported:
            assert row["capacity_profile"] == WSM6_256_CAPACITY_PROFILE, component_id
            assert row["levels_with_capacity_profile"] == list(
                WSM6_CAPACITY_LEVEL_BOUNDS
            ), component_id
        else:
            # A component whose launcher never reads the two files the overlay
            # patches must not advertise it: selecting it there is refused.
            assert row["capacity_profile"] is None, component_id
            assert row["levels_with_capacity_profile"] is None, component_id


def test_the_live_ihail_row_is_cm1s_own_admitted_range() -> None:
    ihail = cartesian_live_admission()["ihail"]
    assert ihail["values"] == list(IHAIL_VALUES)
    # Per component, because src/param.F:1567-1580 gives CM1 no default and the
    # two components have different histories; the record must not flatten that
    # into one number a reader would take for CM1's.
    assert ihail["default_by_component"] == dict(DEFAULT_IHAIL_BY_COMPONENT)
    assert "src/param.F:1567-1580" in ihail["authority"]
    assert "src/param.F:3407" in ihail["authority"]
    assert "src/param.F:3697-3706" in ihail["authority"]


def test_the_live_ndcnst_row_is_the_one_value_the_kernel_executes() -> None:
    ndcnst = cartesian_live_admission()["ndcnst"]
    assert ndcnst["values"] == [MORRISON_NDCNST_CM3]
    assert ndcnst["scope"] == "morrison-mp10"


def test_widening_the_lane_moved_no_sealed_identity() -> None:
    """The point of keeping the two records apart, asserted rather than assumed.

    The sealed envelope must still name exactly one component and must carry
    no ihail row: the moment it grows one, every Cartesian document in the
    distribution changes digest and every receipt that names one stops
    matching it.
    """
    envelope = profile_identity(Lane.CARTESIAN.value)["admitted_envelope"]
    assert envelope["physics_component"] == "wsm6-mp6"
    assert "physics_components" not in envelope
    assert "ihail" not in envelope
    assert "levels_by_component" not in envelope


@pytest.mark.parametrize(
    "name,digest",
    [
        ("cartesian-uniform-grid.toml",
         "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"),
        ("cartesian-supercell-attempt.toml",
         "f325e5e78bffd1d57052b1c1dd4e085b1eb91f619070bb83a3d85b02b22b333b"),
        ("cartesian-supercell-translating-frame.toml",
         "77288cd92837908d70baedebc1aa914c2dc6a5bdb1f90d3d8255cfed15a545ca"),
        ("cartesian-wk-analytic-sounding-shear.toml",
         "e6ca95ed63f9555381a4ec8d04b93cf4cfff324119b0fb0df1ed9d14e696cee7"),
    ],
)
def test_every_pre_existing_cartesian_document_keeps_its_digest(
    name: str, digest: str
) -> None:
    assert load_config(EXAMPLES / name).sha256 == digest


def test_the_manifest_says_why_the_two_records_are_separate() -> None:
    record = json.loads(
        (ROOT / "src/cm1core/records/CM1-PROFILE-MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )["cartesian_runtime_admission"]
    reason = record["why_this_is_not_inside_cartesian_runtime"]
    assert "plan_sha256" in reason and "SEALED IDENTITY" in reason
