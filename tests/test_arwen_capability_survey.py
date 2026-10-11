"""The estate inventory the other physics lanes consume.

`src/cm1core/records/ARWEN-CAPABILITY-SURVEY.json` is generated from the
pinned ArWen estate by `tools/survey_arwen_capabilities.py`.  These tests hold
it to three things: it describes the estate `cm1core.engine_pin` actually
admits, it regenerates byte-identically from that estate, and it does not
quietly become a GM1 support claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cm1core import engine_pin


ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "src/cm1core/records/ARWEN-CAPABILITY-SURVEY.json"
RECORD = ROOT / "src/cm1core/records/ARWEN-CAPABILITY-SURVEY.md"
ESTATE = ROOT / f"tmp/arwen-estate-gpuwm-{engine_pin.GPUWM_MEASURED_VERSION}"
#: The superseded estate, so the record can say what the bump made NEW.
BASELINE = ROOT / "tmp" / "arwen-estate-gpuwm-2.7.4"


@pytest.fixture(scope="module")
def survey() -> dict:
    return json.loads(SURVEY.read_text(encoding="utf-8"))


def test_survey_describes_the_estate_engine_pin_admits(survey: dict) -> None:
    assert survey["schema"] == "gm1.arwen-capability-survey/v1"
    assert survey["estate"]["distribution_version"] == engine_pin.GPUWM_MEASURED_VERSION
    assert survey["estate"]["source_identity"] == engine_pin.GPUWM_AUDITED_SOURCE_IDENTITY
    assert survey["estate"]["requirement"] == engine_pin.GPUWM_REQUIREMENT

    # The one unit GM1 executes must agree with the byte pin, or the survey is
    # describing a different estate than the seam admits.
    wsm6 = survey["kernels"]["wsm6"]
    assert wsm6["sha256"] == engine_pin.AUDITED_WSM6_SURFACE["gpuwm/core/kernels/wsm6.cu"]
    assert wsm6["entry_points"] == ["wsm6_column"]
    assert wsm6["python_drivers"] == ["gpuwm.core.wsm6"]
    assert "WSM6_KMAX" in wsm6["compile_time_defines"]
    assert wsm6["shared_headers"] == []


def test_every_kernel_row_is_complete_and_reachable(survey: dict) -> None:
    """A driverless kernel is the survey's one actionable defect.

    The point of the record is that a lane can find the launch path for any
    unit.  A unit with neither a launch call nor a module naming it means the
    survey cannot answer its own second question for that unit, and the lane
    would conclude a shipped kernel is unreachable.
    """
    assert survey["kernels_without_a_python_driver"] == []
    for name, row in survey["kernels"].items():
        assert row["translation_unit"] == f"gpuwm/core/kernels/{name}.cu"
        assert len(row["sha256"]) == 64
        assert row["entry_points"], f"{name} declares no __global__ entry point"
        assert row["python_drivers"] or row["python_modules_naming_it"], name


def test_every_registry_option_carries_what_a_wiring_lane_needs(survey: dict) -> None:
    for key, row in survey["registry_options"].items():
        assert key == f"{row['component']}/{row['option']}"
        assert row["implemented"] is not None
        assert row["maturity"] is not None
        # Selectors are how a lane asks for the option at all; only the one
        # unimplemented row is allowed to have none.
        assert row["selectors"] or row["implemented"] is False, key
        bounds = row["vertical_level_bounds"]
        assert bounds is None or len(bounds) == 2, key


def test_wsm6_row_agrees_with_the_seam_and_the_capacity_overlay(survey: dict) -> None:
    from cm1core.engine_capacity import WSM6_CAPACITY_LEVEL_BOUNDS

    option = survey["registry_options"]["microphysics/wsm6-mp6"]
    assert option["selectors"] == {"mp_physics": 6}
    assert option["implemented"] is True
    # Stock 2.8.0 still caps WSM6 at 80 levels.  That is exactly why the
    # capacity-256 overlay is not retired, and this assertion is what will
    # fail the day a future ArWen fixes it upstream.
    assert survey["vertical_level_bounds"]["WSM6_VERTICAL_LEVEL_BOUNDS"] == [2, 80]
    assert WSM6_CAPACITY_LEVEL_BOUNDS == (2, 256)


def test_the_survey_does_not_read_as_a_gm1_support_claim(survey: dict) -> None:
    assert survey["consumed_by_gm1_today"]["components"] == ["microphysics/wsm6-mp6"]
    assert survey["consumed_by_gm1_today"]["byte_admitted_surface_file_count"] == (
        len(engine_pin.AUDITED_WSM6_SURFACE)
    )
    assert "not a GM1 support matrix" in survey["claim_limit"]
    prose = RECORD.read_text(encoding="utf-8")
    assert "It is **not** a GM1 support claim." in prose
    assert "GM1-PHYSICS-SUPPORT.json" in prose


def test_survey_regenerates_byte_identically_from_the_pinned_estate() -> None:
    if not ESTATE.is_dir():
        pytest.skip(f"no preserved audited ArWen estate at {ESTATE}")

    if not BASELINE.is_dir():
        pytest.skip(f"no preserved superseded ArWen estate at {BASELINE}")

    from tools.survey_arwen_capabilities import _canonical, survey as generate

    assert _canonical(generate(ESTATE, BASELINE)) == SURVEY.read_bytes()


def test_the_record_says_what_2_8_0_made_newly_available(survey: dict) -> None:
    """The half of the record a version bump is actually FOR.

    A bump that ships new kernels and says nothing is how a capability sits
    unreachable for a release cycle; a bump that CLAIMS new kernels without
    measuring them is worse.  These assertions hold the delta to the bytes.
    """
    delta = survey["newly_available_since_the_superseded_estate"]
    assert delta["baseline_distribution_version"] == "2.7.4"

    units = delta["translation_units"]
    assert units["base_count"] == 94
    assert units["admitted_count"] == survey["totals"]["translation_units"] == 94
    assert units["removed"] == [], "a removed translation unit would strand any lane using it"
    assert units["added"] == []

    # The entry points 2.8.0 adds are all Thompson's: classic and aerosol-aware
    # Thompson's entry zeroing, column exit and presence handling.
    changed = delta["entry_points_changed_in_existing_units"]
    assert set(changed) == {"thompson", "thompson_aerosol_state"}
    assert all(row["removed"] == [] for row in changed.values())
    assert len(changed["thompson"]["added"]) == 10
    assert len(changed["thompson_aerosol_state"]["added"]) == 2

    # One option removed, none added; no GM1 component is among them.
    assert delta["registry_options"]["added"] == []
    assert delta["registry_options"]["removed"] == ["radiation/rte-rrtmgp-legacy-aggregate"]
    assert survey["vertical_level_bounds"]["NEW_TIEDTKE_VERTICAL_LEVEL_BOUNDS"] == [4, None]

    # WSM6's own option row did NOT move in 2.8.0; Morrison's changed only its
    # consumers.  The delta audit prices the WSM6 surface; this record is only
    # allowed to agree with it.
    fields = delta["registry_options"]["changed_fields"]
    assert "microphysics/wsm6-mp6" not in fields
    assert list(fields["microphysics/morrison-mp10"]) == ["consumers"]
    assert delta["vertical_level_bounds"]["changed"] == {}, (
        "a changed vertical bound silently re-admits or re-refuses a geometry"
    )


def test_every_kernel_row_answers_what_state_it_needs(survey: dict) -> None:
    fields = set(survey["domain_state_fields"])
    methods = set(survey["domain_state_methods"])
    external = set(survey["driver_state_attributes_bound_outside_the_state_module"])
    assert fields and methods
    for name, row in survey["kernels"].items():
        for attribute in row["driver_state_attributes"]:
            assert attribute in fields or attribute in methods or attribute in external, (
                f"{name} touches state.{attribute}, which no vocabulary in the record names"
            )

    # WSM6 is the row GM1 executes, so it is the row that has to be right: the
    # launcher's own touch set must contain every field the registry's ring
    # guard says has to survive the call.
    wsm6 = survey["kernels"]["wsm6"]
    guard = survey["registry_options"]["microphysics/wsm6-mp6"]["consumers"]["ring_guard"]
    assert set(guard["state_fields"]) <= set(wsm6["driver_state_attributes"])
    # 7,216 bytes is the 64-level tier measured in
    # ARWEN-WSM6-CAPACITY-256-AUTHORITY.md; the survey reads it from ArWen's
    # own recording, so the two agreeing is a real cross-check.
    assert set(wsm6["local_frame_bytes"].values()) == {7216}
