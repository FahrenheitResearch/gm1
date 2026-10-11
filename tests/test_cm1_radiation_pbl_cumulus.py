"""CM1's radiation, PBL and cumulus families, from TOML text to behaviour.

Every test here starts from configuration text a user could type and ends
either at a refusal that names a concrete wrong result, or at a changed
operator order in the Cartesian closure.  Nothing is asserted about a
selector that the document does not declare.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from cm1core import cuda_cartesian_closure as closure
from cm1core.arwen_physics_component import (
    ARWEN_COMPONENT_SURFACE,
    IDENTITY,
    RRTMG_LOOKUP_TABLES,
    UNCOUPLED,
    UNPORTED,
    component_inventory,
    resolve_component,
    seam_record,
    unvalidated_markers,
)
from cm1core.cartesian_door import build_runtime_config
from cm1core.cm1_physics_families import (
    CM1SETUP_LABELS,
    CUPARAM_LABELS,
    IPBL_LABELS,
    RADOPT_LABELS,
    RadiationClock,
    cm1setup_flags,
)
from cm1core.config import parse_mapping
from cm1core.cuda_gm1_solve1_idealized_radiation import (
    CUDA_GM1_RTERM_SOURCE,
    CUDA_GM1_RTERM_SOURCE_SHA256,
    cpu_gm1_rterm_theta_tendency,
)
from cm1core.errors import CM1Refusal
from cm1core.preflight import physics_component_blockers


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples" / "cartesian-uniform-grid.toml"


def _document(**tables) -> dict:
    raw = tomllib.loads(BASE.read_text())
    for section, values in tables.items():
        raw.setdefault(section, {}).update(values)
    return raw


def _refusal(**tables) -> CM1Refusal:
    with pytest.raises(CM1Refusal) as error:
        parse_mapping(_document(**tables))
    return error.value


# --------------------------------------------------------------------------
# The document keeps its sealed identity when it declares none of the keys
# --------------------------------------------------------------------------

def test_a_document_that_declares_none_of_the_new_keys_is_byte_identical():
    """Nothing new is projected, so every recorded plan_sha256 still matches."""

    config = parse_mapping(_document())
    payload = config.canonical_dict()
    assert "radiation" not in payload
    assert "rterm" not in payload["closures"]
    assert "cm1setup" not in payload["closures"] and "ipbl" not in payload["closures"]
    assert "cuparam" not in payload["physics"]
    # And the defaults it reads are CM1's own LES deck.
    assert (config.closures.cm1setup, config.closures.ipbl) == (1, 0)
    assert config.closures.rterm_value == 0
    assert config.physics.cuparam_value == 0
    assert config.radiation is None


def test_declaring_a_key_projects_exactly_that_key():
    config = parse_mapping(_document(closures={"rterm": 1}))
    payload = config.canonical_dict()
    assert payload["closures"]["rterm"] == 1
    assert "cm1setup" not in payload["closures"]


# --------------------------------------------------------------------------
# rterm: config text to a changed operator order
# --------------------------------------------------------------------------

def test_rterm_is_selectable_and_changes_the_recorded_operator_order():
    """``closures.rterm = 1`` puts src/solve1.F:1006-1053 into the chain."""

    off = parse_mapping(_document())
    on = parse_mapping(_document(closures={"rterm": 1}))
    assert build_runtime_config(off).rterm == 0
    assert build_runtime_config(on).rterm == 1
    assert closure.source_order(0) == closure.SOURCE_ORDER
    spliced = closure.source_order(1)
    assert closure.RTERM_SOURCE_ORDER_ENTRY in spliced
    index = spliced.index(closure.RTERM_SOURCE_ORDER_ENTRY)
    # src/solve1.F: idiss ends at :1001, rterm is :1006-1053, subgrid
    # turbulence starts at :1116.
    assert spliced[index - 1] == "eqtset=2 idiss theta contribution"
    assert spliced[index + 1] == "SGS momentum stress divergence"


def test_rterm_kernel_text_is_pinned_and_every_operation_is_an_rn_intrinsic():
    """The division is the one that matters, and it is measured.

    Written as a plain ``-tha / 43200.0f`` this kernel disagreed with the CPU
    operator on 3 of 11339 cells by one ULP, and all three were the signature
    of multiply-by-reciprocal: NVRTC strength-reduces a division by a
    compile-time constant even with ``--prec-div=true`` in the option list.
    CM1's Fortran does a real divide (``src/solve1.F:1028``), so the
    reciprocal form is permanently, silently wrong wherever they differ.
    """

    import hashlib

    assert (
        hashlib.sha256(CUDA_GM1_RTERM_SOURCE.encode("ascii")).hexdigest()
        == CUDA_GM1_RTERM_SOURCE_SHA256
    )
    assert "src/solve1.F:1026-1031" in CUDA_GM1_RTERM_SOURCE
    assert "__fdiv_rn(-theta_perturbation, cap_denominator)" in CUDA_GM1_RTERM_SOURCE
    assert "__fdiv_rn(-1.0f, cap_denominator)" in CUDA_GM1_RTERM_SOURCE
    assert "__fmul_rn(12.0f, 3600.0f)" in CUDA_GM1_RTERM_SOURCE
    assert "__fadd_rn(thten1[index], thrad)" in CUDA_GM1_RTERM_SOURCE
    # No bare arithmetic operator may survive in the kernel body.
    body = CUDA_GM1_RTERM_SOURCE.split("{", 1)[1]
    for line in body.splitlines():
        code = line.split("/*", 1)[0]
        assert " / " not in code, line


def test_the_rterm_cpu_oracle_is_cm1s_three_capped_regions():
    """src/solve1.F:1026-1031, including both caps and the sign flip."""

    tha = np.array([-2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5], dtype=np.float32)
    got = cpu_gm1_rterm_theta_tendency(tha.reshape(1, 1, -1)).reshape(-1)
    cap = np.float32(np.float32(1.0) / np.float32(np.float32(12.0) * np.float32(3600.0)))
    quarter = np.float32(np.float32(0.25) / np.float32(43200.0))
    expected = np.array(
        [
            +cap,      # tha < -1 takes the positive cap
            +cap,      # tha == -1 takes the uncapped arm, -(-1)/43200 = +cap
            +quarter,  # -(-0.25)/43200
            np.float32(0.0),
            -quarter,
            -cap,      # tha == 1 takes the uncapped arm, which equals -cap
            -cap,      # tha > 1 takes the negative cap
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(got, expected)


def test_the_rterm_oracle_matches_the_evidenced_cpu_operator_raw_word():
    """The live operator and the sealed CPU operator agree bit for bit."""

    from cm1core.idealized_radiation import (
        IdealizedRadiationSelection,
        cpu_idealized_radiation_inplace,
        idealized_radiation_shapes,
    )

    ni, nj, nk, numq = 5, 4, 3, 6
    rng = np.random.default_rng(20260914)
    shapes = idealized_radiation_shapes(ni, nj, nk, numq)
    arrays = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
    arrays["bud"] = np.zeros(nk, np.float64)
    arrays["qbudget"] = np.zeros(10, np.float64)
    arrays["tha"][...] = rng.uniform(-3.0, 3.0, (nk, nj, ni)).astype(np.float32)
    arrays["th0"][...] = np.float32(300.0)
    arrays["pi0"][...] = np.float32(0.9)
    arrays["rho"][...] = np.float32(1.0)
    arrays["ruh"][...] = np.float32(1.0)
    arrays["rvh"][...] = np.float32(1.0)
    arrays["rmh"][...] = np.float32(1.0)
    arrays["xf"][...] = np.float32(1.0)
    selection = IdealizedRadiationSelection(
        rterm=1, radopt=0, eqtset=2, imoist=1, iice=1,
        nqv=1, nql1=2, nql2=3, nqs1=4, nqs2=6,
        axisymmetric=0, diagnostic=False, mpi=False,
        dt_seconds=0.5, dx_m=500.0, dy_m=500.0, dz_m=500.0,
    )
    tha = arrays["tha"].copy()
    cpu_idealized_radiation_inplace(
        arrays, ni=ni, nj=nj, nk=nk, numq=numq, selection=selection
    )
    live = cpu_gm1_rterm_theta_tendency(tha)
    assert live.tobytes() == arrays["thrad"].tobytes()


# --------------------------------------------------------------------------
# Scale selection: the refusals, and that they do not fire on valid input
# --------------------------------------------------------------------------

def test_the_les_deck_is_admitted_and_is_what_an_undeclared_document_gets():
    declared = parse_mapping(_document(closures={"cm1setup": 1, "ipbl": 0}))
    assert (declared.closures.cm1setup, declared.closures.ipbl) == (1, 0)


def test_the_mesoscale_setup_refuses_because_this_closure_has_no_off_switch():
    """The honest wall: GM1's SGS chain runs every step and reads no selector."""

    detail = _refusal(closures={"cm1setup": 2, "ipbl": 1, "sgs_model": 0}).detail
    assert detail.code == "CM1-CFG-CLOSURE-HAS-NO-OFF-SWITCH"
    assert detail.option == "closures.cm1setup"
    assert "went on mixing every column" in detail.reason
    assert "held_tendencies" in detail.remedy


def test_the_inviscid_setup_refuses_for_the_same_named_reason():
    detail = _refusal(closures={"cm1setup": 0, "ipbl": 0, "sgs_model": 0}).detail
    assert detail.code == "CM1-CFG-CLOSURE-HAS-NO-OFF-SWITCH"
    assert "adiabatic, inviscid" in detail.reason


def test_a_pbl_scheme_without_the_mesoscale_setup_refuses_by_name():
    detail = _refusal(closures={"cm1setup": 1, "ipbl": 1}).detail
    assert detail.code == "CM1-CFG-PBL-WITHOUT-MESOSCALE-SETUP"
    assert detail.option == "closures.ipbl"
    assert "src/param.F:1465-1478" in detail.reason
    assert "silently inert" in detail.reason


def test_the_mesoscale_setup_without_a_pbl_scheme_refuses_by_name():
    detail = _refusal(closures={"cm1setup": 2, "ipbl": 0, "sgs_model": 0}).detail
    assert detail.code == "CM1-CFG-MESOSCALE-SETUP-WITHOUT-PBL"
    assert "no vertical mixing at all" in detail.reason


def test_a_pbl_scheme_beside_the_les_closure_refuses_the_double_count():
    """The whole point: an LES closure and a PBL scheme must not both act."""

    detail = _refusal(closures={"cm1setup": 2, "ipbl": 1, "sgs_model": 1}).detail
    assert detail.code == "CM1-CFG-DOUBLE-COUNTED-VERTICAL-MIXING"
    assert detail.option == "closures.sgs_model"
    assert "mixed twice" in detail.reason
    assert "twice as deep" in detail.reason


def test_nested_les_in_mesoscale_refuses_because_there_is_no_inner_mesh():
    detail = _refusal(closures={"cm1setup": 4, "ipbl": 1}).detail
    assert detail.code == "CM1-CFG-NESTED-LES-IN-MESOSCALE"
    assert "counted twice everywhere" in detail.reason


def test_an_out_of_range_scale_selector_refuses_by_value():
    assert _refusal(closures={"cm1setup": 7, "ipbl": 0}).detail.code == (
        "CM1-CFG-CM1SETUP-RANGE"
    )
    assert _refusal(closures={"cm1setup": 2, "ipbl": 9, "sgs_model": 0}).detail.code == (
        "CM1-CFG-IPBL-RANGE"
    )


def test_a_half_declared_scale_pair_refuses():
    detail = _refusal(closures={"cm1setup": 2}).detail
    assert detail.code == "CM1-CFG-TURBULENCE-SCALE-PARTIAL"
    assert detail.option == "closures.ipbl"


def test_cm1setup_flags_are_param_F_752_753():
    assert cm1setup_flags(0) == (False, False)
    assert cm1setup_flags(1) == (True, False)
    assert cm1setup_flags(2) == (False, True)
    assert cm1setup_flags(3) == (False, False)
    assert cm1setup_flags(4) == (True, True)


# --------------------------------------------------------------------------
# Radiation: the table, its cadence, and its sun
# --------------------------------------------------------------------------

def test_a_radiation_table_is_admitted_and_projected():
    config = parse_mapping(
        _document(
            radiation={
                "radopt": 2,
                "dtrad_s": 300.0,
                "ctrlat": 35.0,
                "ctrlon": -97.5,
                "year": 2011,
                "month": 5,
                "day": 24,
                "hour": 18,
                "minute": 0,
                "second": 0,
            }
        )
    )
    assert config.radiation.radopt == 2
    payload = config.canonical_dict()
    assert payload["radiation"]["ctrlat"] == 35.0
    runtime = build_runtime_config(config)
    record = runtime.physics_families_record()
    assert record["radiation"]["radopt"] == 2
    assert record["radiation"]["location_free"] is False
    assert record["radiation_cadence"]["large_steps_between_calls"] == 600


def test_radiation_off_with_a_declared_place_refuses_the_inert_geometry():
    detail = _refusal(radiation={"radopt": 0, "ctrlat": 35.0}).detail
    assert detail.code == "CM1-CFG-RADIATION-GEOMETRY-INERT"
    assert "no sun at all" in detail.reason


def test_radiation_and_rterm_together_refuse_with_param_F_2283():
    detail = _refusal(
        closures={"rterm": 1},
        radiation={
            "radopt": 2, "dtrad_s": 300.0, "ctrlat": 0.0, "ctrlon": 0.0,
            "year": 2011, "month": 5, "day": 24, "hour": 18,
            "minute": 0, "second": 1,
        },
    ).detail
    assert detail.code == "CM1-CFG-RADOPT-WITH-RTERM"
    assert "src/param.F:2283-2294" in detail.reason


@pytest.mark.parametrize(
    ("key", "value", "code"),
    (
        ("ctrlat", 120.0, "CM1-CFG-CTRLAT-RANGE"),
        ("ctrlon", -200.0, "CM1-CFG-CTRLON-RANGE"),
        ("month", 13, "CM1-CFG-RADIATION-CLOCK-RANGE"),
        ("hour", 24, "CM1-CFG-RADIATION-CLOCK-RANGE"),
    ),
)
def test_a_place_or_a_date_off_the_earth_refuses(key, value, code):
    table = {
        "radopt": 2, "dtrad_s": 300.0, "ctrlat": 35.0, "ctrlon": -97.5,
        "year": 2011, "month": 5, "day": 24, "hour": 18, "minute": 0, "second": 0,
    }
    table[key] = value
    assert _refusal(radiation=table).detail.code == code


def test_dtrad_must_be_positive_or_cm1s_radtim_loop_never_terminates():
    table = {
        "radopt": 2, "dtrad_s": 0.0, "ctrlat": 35.0, "ctrlon": -97.5,
        "year": 2011, "month": 5, "day": 24, "hour": 18, "minute": 0, "second": 0,
    }
    detail = _refusal(radiation=table).detail
    assert detail.code == "CM1-CFG-DTRAD-RANGE"
    assert "does not terminate" in detail.reason or "never passes it" in detail.reason


def test_the_radiation_clock_is_cm1s_own_radtim_walk():
    """src/param.F:6344 starts radtim at 0; src/cm1.F:3196-3199 and 3708-3713.

    dt=6 s, dtrad=300 s: CM1 fires on the first large step, then whenever the
    time at the END of the step reaches the next multiple of dtrad, less a
    tenth of a step.
    """

    clock = RadiationClock(dtrad_s=300.0, dt_s=6.0)
    fired = []
    for completed in range(0, 160):
        if clock.due(completed):
            fired.append(completed + 1)
            clock.advance(completed)
    assert fired[:4] == [1, 50, 100, 150]
    assert clock.steps_between_calls() == 50


def test_the_radiation_clock_fires_every_step_when_dtrad_is_the_timestep():
    clock = RadiationClock(dtrad_s=0.5, dt_s=0.5)
    fired = [n + 1 for n in range(10) if clock.due(n) or clock.advance(n) is None]
    assert fired == list(range(1, 11))


# --------------------------------------------------------------------------
# Cumulus
# --------------------------------------------------------------------------

def test_cumulus_at_storm_resolving_spacing_refuses_the_double_count():
    detail = _refusal(physics={"cuparam": 1}).detail  # the example grid is 500 m
    assert detail.code == "CM1-CFG-CUMULUS-AT-RESOLVED-SCALE"
    assert detail.option == "physics.cuparam"
    assert "same CAPE removed twice" in detail.reason
    assert "src/cu_ntiedtke.F:221-230" in detail.reason


def test_cumulus_is_admitted_on_a_mesoscale_grid():
    config = parse_mapping(
        _document(physics={"cuparam": 1}, grid={"dx_m": 12000.0, "dy_m": 12000.0})
    )
    assert config.physics.cuparam_value == 1
    from cm1core.cm1_physics_families import Cm1CumulusSelection

    assert Cm1CumulusSelection(1, 12000.0, 12000.0).grey_zone is False
    assert Cm1CumulusSelection(1, 6000.0, 6000.0).grey_zone is True


def test_an_unknown_cuparam_refuses_by_value():
    detail = _refusal(
        physics={"cuparam": 2}, grid={"dx_m": 12000.0, "dy_m": 12000.0}
    ).detail
    assert detail.code == "CM1-CFG-CUPARAM-RANGE"


# --------------------------------------------------------------------------
# The ArWen seam
# --------------------------------------------------------------------------

def test_every_cm1_selector_value_resolves_to_exactly_one_disposition():
    inventory = component_inventory()
    assert {row["cm1_value"] for row in inventory["radiation"]} == set(RADOPT_LABELS)
    assert {row["cm1_value"] for row in inventory["pbl"]} == set(IPBL_LABELS)
    assert {row["cm1_value"] for row in inventory["cumulus"]} == set(CUPARAM_LABELS)
    for rows in inventory.values():
        for row in rows:
            assert row["disposition"] in {IDENTITY, UNCOUPLED, UNPORTED}
            assert row["cm1_source"]
            if row["disposition"] != IDENTITY:
                assert row["reason"]


def test_an_uncoupled_component_names_the_carriers_it_still_needs():
    ysu = resolve_component("pbl", 1)
    assert ysu.disposition == UNCOUPLED
    assert ysu.component_id == "ysu-pbl1"
    assert ysu.entry_point == "gpuwm.core.ysu.launch_ysu"
    assert "hfx" in ysu.missing_carriers and "psim" in ysu.missing_carriers
    assert ysu.audited_files and "gpuwm/core/kernels/ysu.cu" in ysu.audited_files


def test_new_tiedtke_is_the_same_scheme_on_both_sides_and_needs_a_newer_engine():
    tiedtke = resolve_component("cumulus", 1)
    assert tiedtke.disposition == UNCOUPLED
    assert tiedtke.minimum_engine == "gpuwm>=2.7.4"
    assert "same scheme" in tiedtke.reason
    assert "scale-dependency" in tiedtke.reason


def test_the_goddard_scheme_is_unported_and_says_what_would_be_zero():
    goddard = resolve_component("radiation", 1)
    assert goddard.disposition == UNPORTED
    assert "lwten/swten/thraten at zero" in goddard.reason


def test_a_selected_uncoupled_family_blocks_preflight_and_names_the_defect():
    """The seam blocker is independent of the lane's own closure wall."""

    import dataclasses

    from cm1core.config import TurbulenceScaleConfig

    config = parse_mapping(_document())
    config = dataclasses.replace(
        config,
        closures=dataclasses.replace(
            config.closures,
            turbulence_scale=TurbulenceScaleConfig(cm1setup=2, ipbl=1),
        ),
    )
    blockers = physics_component_blockers(config)
    assert [b.code for b in blockers] == ["CM1-PHYSICS-COMPONENT-UNCOUPLED"]
    assert "closures.ipbl=1" in blockers[0].reason
    assert "hfx" in blockers[0].remedy


def test_the_default_deck_blocks_nothing_in_this_family():
    assert physics_component_blockers(parse_mapping(_document())) == ()


def test_the_rrtmg_lookup_tables_are_admitted_by_sha256_and_licence():
    record = seam_record()
    assert set(record["rrtmg_lookup_tables"]) == set(RRTMG_LOOKUP_TABLES)
    assert "RRTMG_LW_DATA" in " ".join(RRTMG_LOOKUP_TABLES)
    assert "BSD-3-Clause" in record["rrtmg_lookup_table_licence"]
    surface = ARWEN_COMPONENT_SURFACE["rrtmg-legacy-ra4"]
    for name, digest in RRTMG_LOOKUP_TABLES.items():
        assert surface[name] == digest
        assert len(digest) == 64


def test_unvalidated_markers_appear_for_every_selected_nonzero_family():
    assert unvalidated_markers(radopt=0, ipbl=0, cuparam=0, rterm=0) == []
    # rterm is the one family here with a measured oracle, so it is silent.
    assert unvalidated_markers(radopt=0, ipbl=0, cuparam=0, rterm=1) == []


def test_the_run_record_always_publishes_the_unvalidated_list():
    """An empty list is a claim, so it is published even when empty."""

    runtime = build_runtime_config(parse_mapping(_document(closures={"rterm": 1})))
    record = runtime.physics_families_record()
    assert record["unvalidated_physics"] == []
    assert record["rterm"] == 1
    assert "solve1.F:1003-1054" in record["rterm_authority"]
    assert record["turbulence_scale"]["cm1setup_label"] == CM1SETUP_LABELS[1]
    json.dumps(record)  # the whole record must be receipt-serializable


def test_doctor_names_every_family_and_calls_each_nonzero_value_unvalidated():
    from cm1core.doctor import collect

    subjects = {finding.subject: finding for finding in collect()}
    for family, selector in (
        ("radiation", "radopt"),
        ("pbl", "ipbl"),
        ("cumulus", "cuparam"),
    ):
        finding = subjects[f"CM1 {family} family ({selector})"]
        payload = json.loads(finding.detail)
        assert payload["selector"] == selector
        assert payload["runnable_values"] == [0]
        assert payload["unvalidated"]
        assert "UNVALIDATED" in finding.remedy
    assert "rterm=1" in subjects["CM1 rterm=1 idealized radiation"].subject


# --------------------------------------------------------------------------
# The offline radiation oracle, and the measured evidence beside it
# --------------------------------------------------------------------------

def test_rrtmg_tables_decode_as_the_reads_cm1_issues():
    """src/module_ra_rrtmg_lw.F:13085-13100 issues exactly sixteen reads.

    The tables are big-endian Fortran sequential records
    (``OPEN(... CONVERT="BIG_ENDIAN")`` at :13081-13082), so a truncated or
    re-endianed table fails this decode rather than failing inside a
    radiation call with a plausible heating rate.
    """

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rrtmg_table_admission", ROOT / "tools/oracle/rrtmg_table_admission.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPECTED_RECORDS == {"RRTMG_LW_DATA": 16, "RRTMG_SW_DATA": 14}

    # A hand-built two-record stream round-trips; a corrupted trailer does not.
    import struct

    payload = b"\x00" * 24
    stream = b"".join(
        struct.pack(">i", len(payload)) + payload + struct.pack(">i", len(payload))
        for _ in range(2)
    )
    assert module.sequential_records(stream) == [24, 24]
    with pytest.raises(ValueError, match="closes with"):
        module.sequential_records(stream[:-4] + struct.pack(">i", 25))
    with pytest.raises(ValueError, match="truncated"):
        module.sequential_records(stream + b"\x00\x00")


def test_the_recorded_offline_table_admission_passed():
    """The measured artifact, not a re-derivation: CM1's tables are ArWen's."""

    payload = json.loads(
        (ROOT / "evidence/radpbl/rrtmg-tables/admission.json").read_text()
    )
    assert payload["result"] == "PASS"
    assert payload["failures"] == []
    for name, expected in (("RRTMG_LW_DATA", 16), ("RRTMG_SW_DATA", 14)):
        entry = payload["tables"][name]
        assert entry["cm1_and_arwen_are_the_same_bytes"] is True
        assert entry["cm1"]["records"] == expected
        assert entry["arwen"]["records"] == expected
        assert entry["cm1"]["sha256"] == entry["pinned_sha256"]


def test_the_recorded_rterm_kernel_oracle_is_raw_word_exact():
    payload = json.loads(
        (
            ROOT / "evidence/radpbl/rterm-warm-bubble-32/kernel-raw-word-oracle.json"
        ).read_text()
    )
    assert payload["result"] == "PASS"
    assert payload["mismatch_words_vs_cpu_oracle"] == 0
    assert payload["mismatch_words_between_two_launches"] == 0
    assert payload["kernel_source_sha256"] == CUDA_GM1_RTERM_SOURCE_SHA256
    # The measured artifact must describe the kernel this tree ships.
    assert payload["kernel"] == "gm1_solve1_rterm_theta_tendency"


def test_the_recorded_pair_differential_reads_a_coefficient_of_one():
    """The operator's coefficient, measured against the real CM1 binary.

    ``slope`` is the regression of GM1's rterm increment on CM1's over the
    cells carrying the signal. One means GM1 applies CM1's tendency at CM1's
    magnitude; the reading is only meaningful while ``control_state_nrmse``
    -- the host dycore's own divergence, which this term did not cause --
    stays small.
    """

    payload = json.loads(
        (
            ROOT
            / "evidence/radpbl/rterm-warm-bubble-32/pair-differential-interior4.json"
        ).read_text()
    )
    assert payload["field"] == "th3d" and payload["cm1_field"] == "thpert"
    early = [row for row in payload["frames"] if row["model_time_s"] <= 360.0]
    assert len(early) >= 15
    for row in early:
        assert abs(row["slope_over_signal_cells"] - 1.0) < 2.0e-3
        assert row["correlation"] > 0.98
        assert row["signal_cells"] > 100
    first = payload["frames"][0]
    assert first["control_state_nrmse"] < 5.0e-3
    assert first["cm1_signal_max_abs"] > 4.0e-4


def test_the_shipped_rterm_example_is_admitted_and_selects_the_operator():
    """Config text a user can type, straight to the changed operator order."""

    from cm1core.config import load_config

    config = load_config(ROOT / "examples/cartesian-idealized-radiation-relaxation.toml")
    assert config.closures.rterm_value == 1
    runtime = build_runtime_config(config)
    assert runtime.rterm == 1
    assert closure.RTERM_SOURCE_ORDER_ENTRY in closure.source_order(runtime.rterm)


# --------------------------------------------------------------------------
# The public run door, and the harness's CM1-side vocabulary
# --------------------------------------------------------------------------

def test_the_run_door_refuses_a_selected_uncoupled_family():
    """``require_runnable`` is what ``cartesian`` calls before opening a device."""

    import dataclasses

    from cm1core.config import RadiationConfig
    from cm1core.preflight import require_runnable

    config = parse_mapping(_document())
    config = dataclasses.replace(
        config,
        radiation=RadiationConfig(
            radopt=2, dtrad_s=300.0, ctrlat=36.68, ctrlon=-98.35,
            year=2011, month=5, day=24, hour=18, minute=0, second=0,
        ),
    )
    from cm1core.preflight import blockers

    codes = [b.code for b in blockers(config)]
    assert "CM1-PHYSICS-COMPONENT-UNCOUPLED" in codes
    named = [b for b in blockers(config) if b.code == "CM1-PHYSICS-COMPONENT-UNCOUPLED"]
    assert "radiation.radopt=2" in named[0].reason
    assert "tsk" in named[0].remedy
    # require_runnable raises the first blocker; on a machine with no ArWen
    # estate installed that is the estate blocker, which is why this test
    # reads the list rather than the exception.
    with pytest.raises(CM1Refusal):
        require_runnable(config)
    # And the default deck contributes nothing to that list.
    assert not [
        b
        for b in blockers(parse_mapping(_document()))
        if b.code.startswith("CM1-PHYSICS-COMPONENT-")
    ]


def test_the_differential_harness_now_speaks_these_cm1_words():
    """The key map is what lets a case be written in CM1's own vocabulary."""

    from cm1core.oracle import keymap

    table = keymap.load()
    expected = {
        "rterm": "closures.rterm",
        "cm1setup": "closures.cm1setup",
        "ipbl": "closures.ipbl",
        "cuparam": "physics.cuparam",
        "radopt": "radiation.radopt",
        "dtrad": "radiation.dtrad_s",
        "ctrlat": "radiation.ctrlat",
        "ctrlon": "radiation.ctrlon",
        "year": "radiation.year",
        "second": "radiation.second",
    }
    for cm1_key, gm1_key in expected.items():
        entry = table.by_cm1(cm1_key)
        assert entry is not None, cm1_key
        assert entry.gm1_key == gm1_key
        assert cm1_key not in {row.cm1_key for row in table.unmapped}


def _cm1_supercell_deck_available() -> bool:
    from cm1core.oracle import cm1_run

    try:
        cm1_run.base_deck("supercell")
    except cm1_run.CM1RunError:
        return False
    return True


@pytest.mark.skipif(
    not _cm1_supercell_deck_available(),
    reason="needs CM1 r21.1's supercell deck: a built oracle (CM1_AUTHORITY_ROOT) or the pristine source tree",
)
def test_a_radiation_case_emits_a_radiation_table_the_door_admits():
    from cm1core.oracle import cm1_run
    from cm1core.oracle.spec import DifferentialCase

    base = cm1_run.base_deck("supercell")
    case = DifferentialCase(
        name="radopt-emission",
        term="radiation",
        notes="",
        cm1={
            "radopt": 2, "dtrad": 300.0, "ctrlat": 36.68, "ctrlon": -98.35,
            "year": 2011, "month": 5, "day": 24, "hour": 18,
            "minute": 0, "second": 0,
            # This lane writes no intermediate checkpoint, so the supercell
            # deck's rstfrq has to match its timax; that is a property of the
            # deck, not of radiation.
            "rstfrq": 7200.0,
        },
        gm1_only={
            "run.lane": "cartesian",
            "physics.component": "wsm6-mp6",
            # runtime.acoustic_substeps is NOT declared: misclibs.F:818-915
            # derives nsound from dtl and the grid, and the harness refuses a
            # declaration that disagrees, because handing the two codes a
            # different number of acoustic small steps per large step would
            # measure the schedule rather than the term under test.
            "runtime.tile_nx": 32,
            "runtime.tile_ny": 32,
            "runtime.warmup_steps": 0,
            "runtime.memory_budget_mib": 8192,
            "runtime.capacity_profile": "default",
        },
    )
    document = tomllib.loads(case.gm1_toml(base))
    assert document["radiation"]["radopt"] == 2
    assert document["radiation"]["ctrlat"] == 36.68
    config = parse_mapping(document)
    assert config.radiation.radopt == 2
    # The grammar admits it; the run door is what refuses it, by name.
    from cm1core.preflight import physics_component_blockers

    assert [b.code for b in physics_component_blockers(config)] == [
        "CM1-PHYSICS-COMPONENT-UNCOUPLED"
    ]


def test_the_rterm_horizon_term_exists_and_says_to_run_a_pair():
    from cm1core.oracle.horizon import Grid, compute

    grid = Grid(32, 32, 40, 1000.0, 1000.0, 500.0, 1.0)
    horizon = compute("rterm", grid)
    assert horizon.timescale_s == 43200.0
    assert horizon.horizon_s == 432.0
    assert "solve1.F:1028" in horizon.formula
    assert "rterm=0 / rterm=1 on BOTH sides" in (horizon.instead or "")
