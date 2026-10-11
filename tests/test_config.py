from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from cm1core.config import Lane, load_config, parse_mapping
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]


def product_mapping() -> dict[str, object]:
    config = load_config(ROOT / "examples" / "official-supercell-product.toml")
    payload = config.canonical_dict()
    return {section: payload[section] for section in ("run", "grid", "dynamics", "boundaries", "closures", "physics")}


def real_product_mapping() -> dict[str, object]:
    config = load_config(ROOT / "examples" / "real-atmosphere-100m-product.toml")
    payload = config.canonical_dict()
    return {
        section: payload[section]
        for section in ("run", "grid", "dynamics", "boundaries", "closures", "physics")
    }


def test_official_product_is_sealed_but_not_claimed_runnable() -> None:
    config = load_config(ROOT / "examples" / "official-supercell-product.toml")
    assert config.run.lane is Lane.PRODUCT
    assert config.sha256 == "b3dc296531782eaa166eb26cec30fa030b7405485af1cf2e3ff1af23351e4038"


def test_unknown_key_refuses_by_exact_name_and_value() -> None:
    raw = product_mapping()
    raw["dynamics"]["mystery"] = 17  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-UNKNOWN"
    assert caught.value.detail.option == "dynamics.mystery"
    assert caught.value.detail.value == 17


@pytest.mark.parametrize("bad", [True, 3.0, "3"])
def test_integer_selectors_do_not_coerce(bad: object) -> None:
    raw = product_mapping()
    raw["dynamics"]["psolver"] = bad  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-INTEGER"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "12.5"])
def test_moving_frame_velocity_must_be_finite_numeric(bad: object) -> None:
    raw = product_mapping()
    raw["closures"]["u_move_m_s"] = bad  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code.startswith("CM1-CFG-NUMERIC")


def test_product_profile_drift_is_named() -> None:
    raw = product_mapping()
    raw["grid"]["nx"] = 121  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-PROFILE-DRIFT"
    assert caught.value.detail.option == "grid.nx"


def test_real_product_is_fixed_100m_terrain_profile() -> None:
    config = load_config(ROOT / "examples" / "real-atmosphere-100m-product.toml")
    assert config.run.lane is Lane.REAL_PRODUCT
    assert (config.grid.nx, config.grid.ny, config.grid.nz) == (300, 300, 75)
    assert (config.grid.dx_m, config.grid.dy_m, config.grid.dz_m) == (
        100.0,
        100.0,
        200.0,
    )
    assert config.grid.terrain
    assert not config.closures.moving_frame
    assert config.physics.component == "wsm6-mp6"
    assert config.sha256 == (
        "07c7dde701491ce71a70cec8c56de2cd50ed84574cb098250815d76eaa0c9c00"
    )


def test_real_product_refuses_flat_or_coarser_fallback() -> None:
    for section, key, value in (
        ("grid", "terrain", False),
        ("grid", "dx_m", 1000.0),
    ):
        raw = real_product_mapping()
        raw[section][key] = value  # type: ignore[index]
        with pytest.raises(CM1Refusal) as caught:
            parse_mapping(raw)
        assert caught.value.detail.code == "CM1-CFG-PROFILE-DRIFT"
        assert caught.value.detail.option == f"{section}.{key}"


def test_derived_eqtset_one_cannot_be_called_product() -> None:
    raw = product_mapping()
    raw["physics"]["equation_set"] = 1  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.option == "physics.equation_set"


def test_dry_bootstrap_has_distinct_identity() -> None:
    config = load_config(ROOT / "examples" / "dry-psolver2-bootstrap.toml")
    assert config.run.lane is Lane.DRY_DYCORE
    assert config.dynamics.psolver == 2
    assert config.physics.component == "off"
    assert config.physics.equation_set == 1
    assert config.grid.dx_m == config.grid.dy_m == config.grid.dz_m
    assert config.grid.nx == config.grid.ny == config.grid.nz == 16
    assert (
        config.boundaries.west,
        config.boundaries.east,
        config.boundaries.south,
        config.boundaries.north,
        config.boundaries.bottom,
        config.boundaries.top,
        config.boundaries.radiative_order,
    ) == (1, 1, 1, 1, 1, 1, 1)
    assert config.closures.sgs_model == 0
    assert not config.closures.vertical_implicit_diffusion
    assert config.closures.upper_damping == 0
    assert not config.closures.moving_frame
    assert config.sha256 == "0394dea1f77722e54fe6b1b5a07fe56037cda84800ef870b9a269566e6043340"


def cartesian_mapping() -> dict[str, object]:
    payload = load_config(ROOT / "examples" / "cartesian-uniform-grid.toml").canonical_dict()
    return {
        section: copy.deepcopy(payload[section])
        for section in (
            "run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime"
        )
    }


def test_cartesian_lane_seals_its_own_runtime_table() -> None:
    config = load_config(ROOT / "examples" / "cartesian-uniform-grid.toml")
    assert config.run.lane is Lane.CARTESIAN
    assert config.runtime is not None
    assert config.runtime.acoustic_substeps == 6
    assert (config.runtime.tile_nx, config.runtime.tile_ny) == (64, 64)
    assert config.runtime.memory_budget_mib == 8192
    assert config.runtime.capacity_profile == "default"
    # Selecting a second column component did not reseal this document.  The
    # lane's admitted component catalogue lives in records/CM1-SUPPORT-MATRIX.csv
    # and records/GM1-PHYSICS-SUPPORT.json, not in the projected profile, so a
    # measured run's plan_sha256 still names the document it was run from.
    assert config.sha256 == (
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    )


def test_runtime_table_is_unknown_in_every_other_lane() -> None:
    raw = product_mapping()
    raw["runtime"] = cartesian_mapping()["runtime"]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-UNKNOWN"
    assert caught.value.detail.option == "runtime"


def test_cartesian_lane_requires_its_runtime_table() -> None:
    raw = cartesian_mapping()
    del raw["runtime"]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-MISSING-SECTION"
    assert caught.value.detail.option == "runtime"


@pytest.mark.parametrize(
    ("key", "bad", "code"),
    (
        ("mystery", 17, "CM1-CFG-UNKNOWN"),
        ("acoustic_substeps", 5, "CM1-CFG-CARTESIAN-ACOUSTIC-SUBSTEPS"),
        ("acoustic_substeps", 6.0, "CM1-CFG-INTEGER"),
        ("tile_nx", 48, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("tile_ny", 48, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("warmup_steps", -1, "CM1-CFG-INTEGER"),
        ("memory_budget_mib", 0, "CM1-CFG-INTEGER"),
        ("capacity_profile", "256", "CM1-CFG-UNSUPPORTED-VALUE"),
        ("capacity_profile", 256, "CM1-CFG-STRING"),
    ),
)
def test_runtime_keys_and_values_are_refused_by_name(
    key: str, bad: object, code: str
) -> None:
    raw = cartesian_mapping()
    raw["runtime"][key] = bad  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == code
    assert caught.value.detail.option == f"runtime.{key}"


def test_cartesian_large_step_is_a_range_bounded_by_cm1s_own_dtsmall() -> None:
    """The half second is the default, not a pin; CM1's acoustic bound is.

    ``examples/cartesian-uniform-grid.toml`` declares 500 m spacing and six
    acoustic substeps.  ``misclibs.F:867`` -- the three-dimensional psolver=3
    branch, which divides by the literal 360.0; misclibs.F:869-879 is the
    psolver=6/7 branch and divides by ``csound`` instead -- admits a small step
    up to ``0.60*500/360 = 0.8333 s`` and therefore a large step up to 5.0 s.
    """

    for dt_s, duration_s in ((0.25, 5.0), (1.0, 5.0), (2.5, 5.0), (5.0, 5.0)):
        raw = cartesian_mapping()
        raw["run"].update(  # type: ignore[union-attr]
            {
                "dt_s": dt_s,
                "duration_s": duration_s,
                "history_interval_s": duration_s,
                "restart_interval_s": duration_s,
            }
        )
        assert parse_mapping(raw).run.dt_s == dt_s


def test_cartesian_refuses_a_large_step_its_acoustic_substeps_cannot_carry() -> None:
    raw = cartesian_mapping()
    raw["run"].update(  # type: ignore[union-attr]
        {"dt_s": 6.0, "duration_s": 12.0, "history_interval_s": 12.0, "restart_interval_s": 12.0}
    )
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-CARTESIAN-ACOUSTIC-SMALL-STEP"
    assert detail.option == "run.dt_s"
    assert detail.value == 6.0
    assert "misclibs.F:867" in detail.reason
    assert "CM1's margin, not the instability" in detail.reason
    # The named remedy is the count CM1's own dtsmall would have derived.
    assert "acoustic_substeps to 8" in detail.remedy
    raw["runtime"]["acoustic_substeps"] = 8  # type: ignore[index]
    assert parse_mapping(raw).run.dt_s == 6.0


def test_cartesian_refuses_a_step_that_is_not_a_normal_binary32_word() -> None:
    raw = cartesian_mapping()
    raw["run"].update(  # type: ignore[union-attr]
        {
            "dt_s": 1e-40,
            "duration_s": 1e-39,
            "history_interval_s": 1e-39,
            "restart_interval_s": 1e-39,
        }
    )
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-CARTESIAN-TIMESTEP"
    assert detail.option == "run.dt_s"
    assert "solve2.F:1569-1576" in detail.reason


def test_cartesian_clock_must_still_divide_the_declared_events() -> None:
    """A finer step is admitted, but not one that cannot place a frame."""

    raw = cartesian_mapping()
    raw["run"].update(  # type: ignore[union-attr]
        {"dt_s": 0.3, "duration_s": 5.0, "history_interval_s": 5.0, "restart_interval_s": 5.0}
    )
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-CLOCK-ALIGNMENT"
    assert caught.value.detail.option == "run.duration_s"


def test_cartesian_admits_an_intermediate_restart_cadence() -> None:
    """The refusal that stood here said this lane writes no checkpoint.

    It does now, so the guard is gone rather than relaxed: cm1.F:3186 fires a
    restart whenever the model clock reaches rsttim and cm1.F:4247-4252
    advances rsttim by rstfrq, which is exactly a cadence shorter than the
    run.  README.namelist:103-104 is the namelist key.
    """

    raw = cartesian_mapping()
    raw["run"].update(  # type: ignore[union-attr]
        {"duration_s": 30.0, "history_interval_s": 5.0, "restart_interval_s": 10.0}
    )
    config = parse_mapping(raw)
    assert config.run.restart_interval_s == 10.0
    assert config.run.duration_s == 30.0

    # A restart between two frames is admitted as well.  CM1 compares the
    # clock against taptim and rsttim independently (cm1.F:3177, 3186), and
    # the resume schedule keeps the frames on the original absolute grid
    # (test_restart_reset_frqtim_is_reachable_and_changes_what_is_admitted
    # proves the off-grid resume), so the refusal d9509b9 pinned here named a
    # wrong result the schedule does not produce.
    raw["run"]["restart_interval_s"] = 2.5  # type: ignore[index]
    between = parse_mapping(raw)
    assert between.run.restart_interval_s == 2.5
    assert between.run.history_interval_s == 5.0


def test_cartesian_refuses_a_restart_cadence_that_misses_the_final_state() -> None:
    """The last restart must be the run's own completed state.

    A cadence that does not divide the duration leaves the newest restart up
    to one interval older than the run that wrote it, so a follow-on run would
    silently recompute model time the first run had already produced.
    """

    raw = cartesian_mapping()
    raw["run"].update(  # type: ignore[union-attr]
        {"duration_s": 30.0, "history_interval_s": 5.0, "restart_interval_s": 4.0}
    )
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-CARTESIAN-RESTART-CADENCE"
    assert caught.value.detail.option == "run.restart_interval_s"
    assert "older than the run" in caught.value.detail.reason


def test_cartesian_history_interval_must_divide_the_duration() -> None:
    """Frames land on multiples of the interval up to the duration.

    An interval that does not divide the duration would stop the frames short
    of the run's own completed state, so the state the run was asked for would
    never be published.  An interval that does divide it is now admitted: this
    lane integrates instead of emitting exactly one end-of-run frame.
    """

    raw = cartesian_mapping()
    raw["run"].update(  # type: ignore[union-attr]
        {"duration_s": 30.0, "history_interval_s": 4.0, "restart_interval_s": 30.0}
    )
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-CARTESIAN-CADENCE"
    assert caught.value.detail.option == "run.history_interval_s"

    raw = cartesian_mapping()
    raw["run"].update(  # type: ignore[union-attr]
        {"duration_s": 30.0, "history_interval_s": 5.0, "restart_interval_s": 30.0}
    )
    assert parse_mapping(raw).run.history_interval_s == 5.0


def test_absent_initialization_table_keeps_the_sealed_identity() -> None:
    """The optional table is omitted from the canonical document, not nulled."""

    config = load_config(ROOT / "examples" / "cartesian-uniform-grid.toml")
    assert config.initialization is None
    assert "initialization" not in config.canonical_dict()
    assert config.sha256 == (
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    )


def test_analytic_initialization_lane_is_sealed_by_its_cm1_selectors() -> None:
    config = load_config(
        ROOT / "examples" / "cartesian-wk-analytic-sounding-shear.toml"
    )
    assert config.run.lane is Lane.CARTESIAN
    assert config.initialization is not None
    assert (config.initialization.isnd, config.initialization.iwnd) == (5, 2)
    assert (config.run.duration_s, config.run.dt_s) == (7200.0, 0.5)
    assert config.run.history_interval_s == 60.0
    assert (config.grid.nx, config.grid.ny, config.grid.nz) == (120, 120, 40)
    assert (config.grid.dx_m, config.grid.dz_m) == (1000.0, 500.0)
    assert config.canonical_dict()["initialization"] == {"isnd": 5, "iwnd": 2}
    assert config.sha256 != load_config(
        ROOT / "examples" / "cartesian-uniform-grid.toml"
    ).sha256


@pytest.mark.parametrize(
    ("key", "bad", "code"),
    (
        ("mystery", 1, "CM1-CFG-UNKNOWN"),
        ("isnd", 7.0, "CM1-CFG-INTEGER"),
        ("isnd", True, "CM1-CFG-INTEGER"),
        # base.F:455-460 stops CM1 for isnd=6; 15 needs testcase 4/5's
        # constants (constants.F:78-98); 16 and 24 have no arm.
        ("isnd", 6, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("isnd", 15, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("isnd", 16, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("isnd", 24, "CM1-CFG-UNSUPPORTED-VALUE"),
        # base.F:2286-2640 has no iwnd=7 arm at all.
        ("iwnd", 7, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("iwnd", "2", "CM1-CFG-INTEGER"),
        # init3d.F has no iinit=6 arm; README.namelist admits ibalance 0..2.
        ("iinit", 6, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("ibalance", 3, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("testcase", 4, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("testcase", 5, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("iorigin", 3, "CM1-CFG-UNSUPPORTED-VALUE"),
        ("maintain_rh", 1, "CM1-CFG-BOOLEAN-TYPE"),
    ),
)
def test_initialization_keys_and_values_are_refused_by_name(
    key: str, bad: object, code: str
) -> None:
    raw = cartesian_mapping()
    raw["initialization"] = {"isnd": 5, "iwnd": 2}
    raw["initialization"][key] = bad  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == code
    assert caught.value.detail.option == f"initialization.{key}"


def test_every_sounding_and_every_wind_branch_are_now_admitted() -> None:
    # base.F:134-2036's sounding arms (every analytic one) and the eleven
    # base.F:2286-2640 wind branches each have a transcribed initializer.
    # A run starts from the soundings whose integration matched CM1's like for
    # like: 5, 7 and 17 (every earlier release) and 9 (15 minutes on Morrison).
    for isnd in (5, 7, 9, 17):
        raw = cartesian_mapping()
        raw["initialization"] = {"isnd": isnd, "iwnd": 0}
        assert parse_mapping(raw).initialization.isnd == isnd
    # Every other analytic sounding is built word for word but shown only, with
    # run.iconly = 1: the soundings with base-state cloud (base.F qc0/qi0) and
    # isnd = 11 did not match CM1's over 15 minutes, and the integration from
    # the rest has not been compared with CM1's.
    for isnd, code in (
        *((n, "CM1-CFG-BASE-CLOUD-INTEGRATION") for n in (4, 10)),
        *((n, "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION")
          for n in (1, 8, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23)),
    ):
        raw = cartesian_mapping()
        raw["initialization"] = {"isnd": isnd, "iwnd": 0}
        with pytest.raises(CM1Refusal) as caught:
            parse_mapping(raw)
        assert caught.value.detail.code == code, isnd
        assert caught.value.detail.option == "initialization.isnd"
        assert caught.value.detail.value == isnd
        assert "run.iconly = 1" in caught.value.detail.remedy
        assert ("no time integration from it has been compared" in caught.value.detail.reason) == (isnd not in (4, 10, 11))
        raw["run"]["iconly"] = 1
        assert parse_mapping(raw).initialization.isnd == isnd
    # isnd = 8 stays refused whatever its testcase (no integration from it was compared).
    raw = cartesian_mapping()
    raw["initialization"] = {"isnd": 8, "iwnd": 0, "testcase": 10}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-ANALYTIC-SOUNDING-INTEGRATION"
    # param.F:1501-1515: the dry isothermal and constant-lapse soundings need
    # imoist = 0, and say so by name.
    for isnd in (2, 3):
        raw = cartesian_mapping()
        raw["initialization"] = {"isnd": isnd, "iwnd": 0}
        with pytest.raises(CM1Refusal) as caught:
            parse_mapping(raw)
        assert caught.value.detail.code == "CM1-CFG-ISND-DRY"


def test_the_moist_benchmark_and_the_anelastic_balance_are_admitted() -> None:
    # ibalance = 2's integration matched CM1's over 15 minutes: admitted for a run.
    raw = cartesian_mapping()
    raw["initialization"] = {"isnd": 5, "iwnd": 2, "iinit": 1, "ibalance": 2}
    assert parse_mapping(raw).initialization.ibalance == 2
    # The moist benchmark bubble's did not: shown with run.iconly = 1, refused for a run, also on a sounding
    # without base-state cloud.
    raw["initialization"] = {"isnd": 5, "iwnd": 2, "iinit": 4}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-MOIST-BENCHMARK-INTEGRATION"
    assert caught.value.detail.option == "initialization.iinit"
    raw = cartesian_mapping()
    raw["run"]["iconly"] = 1
    raw["initialization"] = {"isnd": 4, "iwnd": 0, "iinit": 4, "ibalance": 2}
    parsed = parse_mapping(raw).initialization
    assert (parsed.isnd, parsed.iinit, parsed.ibalance) == (4, 4, 2)
    # base.F:267-268's words, CM1's values by default.
    assert (parsed.thec_mb_k, parsed.qt_mb) == (320.0, 0.020)
    # init3d.F:579-647 never sets maintain_rh.
    raw["initialization"]["maintain_rh"] = True
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-INIT-WORD-UNREAD"


@pytest.mark.parametrize(("key", "value", "isnd"), (("thec_mb_k", 330.0, 5), ("qt_mb", 0.018, 7), ("tsk0_k", 300.0, 4)))
def test_a_sounding_word_the_selected_arm_never_reads_is_refused(key: str, value: float, isnd: int) -> None:
    raw = cartesian_mapping()
    raw["initialization"] = {"isnd": isnd, "iwnd": 0, key: value}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-INIT-WORD-UNREAD"
    assert caught.value.detail.option == f"initialization.{key}"


@pytest.mark.parametrize(("testcase", "imoist"), ((1, 1), (2, 1), (3, 0), (7, 0), (8, 0)))
def test_a_testcase_cm1_runs_with_another_imoist_is_refused(testcase: int, imoist: int) -> None:
    # param.F:1044-1072 resets imoist for these test cases without stopping.
    raw = cartesian_mapping()
    raw["initialization"] = {"isnd": 5, "iwnd": 2, "testcase": testcase, "imoist": imoist}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-TESTCASE-IMOIST"
    for iwnd in (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11):
        raw = cartesian_mapping()
        raw["initialization"] = {"isnd": 5, "iwnd": iwnd}
        assert parse_mapping(raw).initialization.iwnd == iwnd


def test_initialization_table_needs_both_selectors() -> None:
    raw = cartesian_mapping()
    raw["initialization"] = {"isnd": 5}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-MISSING-KEY"
    assert caught.value.detail.option == "initialization.iwnd"


def test_single_sounding_lane_refuses_a_wind_branch_cm1_would_discard() -> None:
    """param.F:836-852 resets iwnd to 0 for isnd=7 without stopping."""

    raw = cartesian_mapping()
    raw["initialization"] = {"isnd": 7, "iwnd": 2}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-CARTESIAN-BASE-WIND"
    assert caught.value.detail.option == "initialization.iwnd"
    assert caught.value.detail.value == 2
    raw["initialization"] = {"isnd": 7, "iwnd": 0}
    assert parse_mapping(raw).initialization.isnd == 7


def test_initialization_table_is_unknown_in_every_other_lane() -> None:
    raw = product_mapping()
    raw["initialization"] = {"isnd": 5, "iwnd": 2}
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-UNKNOWN"
    assert caught.value.detail.option == "initialization"


def test_unknown_lane_is_named_before_its_optional_tables() -> None:
    raw = cartesian_mapping()
    raw["run"]["lane"] = "cartesian-experiment"  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-UNSUPPORTED-VALUE"
    assert caught.value.detail.option == "run.lane"


def test_native_context_preserves_official_signed_restart_sentinel() -> None:
    config = load_config(ROOT / "examples" / "official-supercell-native-context.toml")
    assert config.run.lane is Lane.NATIVE_CONTEXT
    assert config.run.restart_interval_s == -3600.0
    assert config.sha256 == "33761ba3ef1e35ce8dd6fa3fa126c4ed2437bdbe49d3b9e73adb2fa2bcf92b8c"


def test_disabled_dry_signed_zero_has_one_canonical_hash() -> None:
    config = load_config(ROOT / "examples" / "dry-psolver2-bootstrap.toml")
    raw = {
        section: copy.deepcopy(config.canonical_dict()[section])
        for section in ("run", "grid", "dynamics", "boundaries", "closures", "physics")
    }
    for key in (
        "upper_damping_depth_m",
        "upper_damping_rate_s_inv",
        "u_move_m_s",
        "v_move_m_s",
    ):
        raw["closures"][key] = -0.0  # type: ignore[index]
    assert parse_mapping(raw).sha256 == config.sha256


def test_acoustic_substeps_admit_cm1s_own_range_not_a_shorter_list() -> None:
    """``misclibs.F:835,883-885``: even, at least four, at most 192.

    A tornado-scale grid needs more small steps per large step than the five
    values this lane used to enumerate; CM1's own derivation admits every even
    count up to its ``max_nsound``, and so does this door.
    """

    for count in (4, 6, 14, 32, 190, 192):
        raw = cartesian_mapping()
        raw["runtime"]["acoustic_substeps"] = count  # type: ignore[index]
        assert parse_mapping(raw).runtime.acoustic_substeps == count
    for count in (3, 5, 2, 0, 191, 194):
        raw = cartesian_mapping()
        raw["runtime"]["acoustic_substeps"] = count  # type: ignore[index]
        with pytest.raises(CM1Refusal) as caught:
            parse_mapping(raw)
        detail = caught.value.detail
        assert detail.code == "CM1-CFG-CARTESIAN-ACOUSTIC-SUBSTEPS"
        assert detail.option == "runtime.acoustic_substeps"
        assert "sound.F:95" in detail.reason
        assert "misclibs.F:835" in detail.reason
    # A negative count is not an acoustic cadence question at all; the
    # grammar refuses it one layer earlier.
    raw = cartesian_mapping()
    raw["runtime"]["acoustic_substeps"] = -4  # type: ignore[index]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-INTEGER"


def test_a_tornado_scale_grid_and_step_are_admitted_end_to_end() -> None:
    """The whole point of unpinning the half second, exercised by value.

    Tens of metres horizontally and vertically, with a step small enough that
    CM1's own acoustic bound is satisfied, must parse; the same fine grid at
    the old welded half second and the old default six substeps must not.
    """

    def document(dx_m: float, dz_m: float, dt_s: float, substeps: int):
        raw = cartesian_mapping()
        raw["grid"].update(  # type: ignore[union-attr]
            {"nx": 64, "ny": 64, "nz": 64, "dx_m": dx_m, "dy_m": dx_m, "dz_m": dz_m}
        )
        raw["closures"]["upper_damping_depth_m"] = dz_m * 8  # type: ignore[index]
        raw["run"].update(  # type: ignore[union-attr]
            {
                "dt_s": dt_s,
                "duration_s": 60.0,
                "history_interval_s": 60.0,
                "restart_interval_s": 60.0,
            }
        )
        raw["runtime"]["acoustic_substeps"] = substeps  # type: ignore[index]
        return raw

    config = parse_mapping(document(50.0, 50.0, 0.05, 4))
    assert config.run.dt_s == 0.05 and config.grid.dx_m == 50.0
    config = parse_mapping(document(20.0, 20.0, 0.025, 4))
    assert config.run.dt_s == 0.025
    # 20 m cells cannot carry the historical half second on six substeps.
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(document(20.0, 20.0, 0.5, 6))
    assert caught.value.detail.code == "CM1-CFG-CARTESIAN-ACOUSTIC-SMALL-STEP"
    # CM1's own dtsmall names the count that would carry it, and it works.
    assert "acoustic_substeps to 16" in caught.value.detail.remedy
    assert parse_mapping(document(20.0, 20.0, 0.5, 16)).run.dt_s == 0.5


def test_every_shipped_example_document_is_still_admitted() -> None:
    """A profile that stops parsing would ship broken and be found by a user.

    ``examples/`` is the published surface of this configuration grammar.
    Nothing enumerated it before, so a key rename or a tightened bound could
    have left one of these documents unloadable with every test still green.
    """

    from pathlib import Path

    from cm1core.config import load_config

    root = Path(__file__).resolve().parents[1] / "examples"
    documents = sorted(root.glob("*.toml"))
    assert len(documents) >= 8, "the example set shrank without a test noticing"
    for path in documents:
        config = load_config(path)
        assert config.run.lane is not None, path.name


def test_the_fine_grid_example_is_the_step_the_welded_half_second_refused() -> None:
    """``examples/cartesian-fifty-metre-fine-step.toml`` by value.

    The profile is the reason the weld came out: fifty metres horizontally
    and vertically at a step CM1's own bounds admit comfortably, where the
    historical half second sits exactly on ``misclibs.F:867``'s acoustic
    limit and within one and a half of ``misclibs.F:2714``'s advective stop.
    """

    from pathlib import Path

    from cm1core.config import load_config
    from cm1core.gm1_runtime import (
        CM1_COURANT_STOP,
        CM1_COURANT_TARGET,
        acoustic_small_step_defect,
        cm1_acoustic_small_step_limit_s,
        cm1_courant_limited_speed_m_s,
        cm1_diagnostic_acoustic_substeps,
    )

    path = (
        Path(__file__).resolve().parents[1]
        / "examples/cartesian-fifty-metre-fine-step.toml"
    )
    config = load_config(path)
    assert config.run.dt_s == 0.05
    assert config.grid.dx_m == config.grid.dy_m == config.grid.dz_m == 50.0
    assert config.runtime is not None
    substeps = config.runtime.acoustic_substeps
    assert substeps == cm1_diagnostic_acoustic_substeps(
        dt_s=config.run.dt_s, dx_m=50.0, dy_m=50.0
    ) == 4
    assert acoustic_small_step_defect(
        dt_s=config.run.dt_s, dx_m=50.0, dy_m=50.0, acoustic_substeps=substeps
    ) is None
    limit = cm1_acoustic_small_step_limit_s(dx_m=50.0, dy_m=50.0)
    assert config.run.dt_s / substeps < limit / 2
    # The half second is not refused at 50 m, but it has no acoustic margin
    # at all and only a factor of 1.5 of advective headroom over the speed
    # CM1's own adaptive controller steers toward.
    assert 0.5 / 6 == limit
    assert acoustic_small_step_defect(
        dt_s=0.5, dx_m=50.0, dy_m=50.0, acoustic_substeps=4
    ) is not None
    assert cm1_courant_limited_speed_m_s(
        dt_s=0.5, dx_m=50.0, dy_m=50.0, dz_m=50.0, courant=CM1_COURANT_TARGET
    ) == 100.0
    assert cm1_courant_limited_speed_m_s(
        dt_s=0.5, dx_m=50.0, dy_m=50.0, dz_m=50.0, courant=CM1_COURANT_STOP
    ) == 150.0
    assert cm1_courant_limited_speed_m_s(
        dt_s=config.run.dt_s, dx_m=50.0, dy_m=50.0, dz_m=50.0,
        courant=CM1_COURANT_TARGET,
    ) == 1000.0


# ---------------------------------------------- the remedy has to work


@pytest.mark.parametrize(
    ("dx_m", "acoustic_substeps", "dt_s", "duration_s", "interval_s"),
    (
        # The two cases the adversarial review reproduced: above max_nsound,
        # and a printed maximum that was a fixed point the door rejected.
        (20.0, 6, 20.0, 600.0, 600.0),
        (125.0, 22, 10.0, 600.0, 10.0),
        (50.0, 6, 1.0, 600.0, 10.0),
        (75.0, 6, 2.0, 600.0, 60.0),
        (25.0, 4, 5.0, 300.0, 30.0),
        (1000.0, 4, 60.0, 7200.0, 60.0),
        (31.25, 8, 12.0, 480.0, 48.0),
        (250.0, 12, 30.0, 900.0, 90.0),
    ),
)
def test_the_acoustic_refusal_names_a_step_this_same_door_admits(
    dx_m: float,
    acoustic_substeps: int,
    dt_s: float,
    duration_s: float,
    interval_s: float,
) -> None:
    """A refusal whose remedy the same door refuses is worse than none.

    Both routes the message can name are followed here verbatim: the substep
    count when one exists below ``max_nsound`` (misclibs.F:835), and the large
    step, which has to satisfy the door's own ``dt/nsound > dtsm`` test and to
    place every declared event on an exact large-step boundary.
    """

    import re

    def document() -> dict[str, object]:
        raw = cartesian_mapping()
        raw["grid"].update({"dx_m": dx_m, "dy_m": dx_m, "dz_m": dx_m})  # type: ignore[union-attr]
        raw["closures"]["upper_damping_depth_m"] = min(  # type: ignore[index]
            3000.0, raw["grid"]["nz"] * dx_m * 0.25  # type: ignore[index]
        )
        raw["runtime"]["acoustic_substeps"] = acoustic_substeps  # type: ignore[index]
        raw["run"].update(  # type: ignore[union-attr]
            {
                "dt_s": dt_s,
                "duration_s": duration_s,
                "history_interval_s": interval_s,
                "restart_interval_s": duration_s,
            }
        )
        return raw

    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(document())
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-CARTESIAN-ACOUSTIC-SMALL-STEP"

    step = re.search(
        r"(?:set run\.dt_s to|lower run\.dt_s to at most) ([0-9.eE+-]+) seconds",
        detail.remedy,
    )
    assert step is not None, detail.remedy
    corrected = document()
    corrected["run"]["dt_s"] = float(step.group(1))  # type: ignore[index]
    admitted = parse_mapping(corrected)
    assert admitted.run.dt_s == float(step.group(1))

    count = re.search(r"raise runtime\.acoustic_substeps to (\d+)", detail.remedy)
    if count is None:
        # Above max_nsound there is no substep route, and the message must not
        # invent one: misclibs.F:887-911 says "Time step (dtl) needs to be
        # smaller" and that is the whole of CM1's answer.
        assert "not available here" in detail.remedy
        return
    corrected = document()
    corrected["runtime"]["acoustic_substeps"] = int(count.group(1))  # type: ignore[index]
    assert parse_mapping(corrected).run.dt_s == dt_s


def test_the_pre_existing_default_configuration_is_bit_identical() -> None:
    """Nothing in the unpinned step may move the default profile one word.

    The shipped Cartesian example is parsed and its canonical document hashed;
    the digest is the one this configuration had before the large step became a
    free parameter, and ``config.sha256`` is that document's own digest.  A
    change to any admitted default, key or value would move it.
    """

    import hashlib
    import json

    config = load_config(ROOT / "examples" / "cartesian-uniform-grid.toml")
    assert config.run.dt_s == 0.5
    assert config.runtime is not None and config.runtime.acoustic_substeps == 6
    assert config.sha256 == (
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    )
    canonical = json.dumps(
        config.canonical_dict(), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == config.sha256
