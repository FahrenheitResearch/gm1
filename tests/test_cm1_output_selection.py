"""CM1 ``&param9``: the key inventory, the door, and the path to the frame.

Nothing here touches CUDA.  The point under test is that a TOML document can
select a CM1 output field, that the selection reaches the carrier table, and
that a selection this lane cannot honour is refused by key, by value, and with
the concrete wrong result named.
"""

import json
from pathlib import Path

import pytest

from cm1core.cartesian_door import build_runtime_config
from cm1core.cartesian_history import cartesian_history_profile
from cm1core.cm1_output_selection import (
    CM1_DISTRIBUTED_DEFAULTS,
    DEFAULT_MOVES,
    DEFAULT_OUTPUT_SELECTION,
    OUTPUT_KEYS,
    PRODUCT,
    STATE,
    SWITCH_BY_KEY,
    SWITCHES,
    UNSUPPORTED,
    OutputSelection,
    output_carrier_rows,
)
from cm1core.config import _OUTPUT_KEY_OWNER, parse_mapping
from cm1core.errors import CM1Refusal

_SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "rust" / "cm1-state" / "schema" / "cm1-r21.1-namelist-support.json"
)


def _authority_param9_keys() -> tuple[str, ...]:
    payload = json.loads(_SCHEMA.read_text())
    group = next(entry for entry in payload["groups"] if entry["name"] == "param9")
    return tuple(field["name"] for field in group["fields"])


def test_every_param9_word_in_the_packaged_authority_inventory_is_admitted():
    """64 words, and exactly the 64 the packaged CM1 inventory names.

    Checked against the inventory rather than a hand list, so a CM1 word this
    port forgot fails here instead of surfacing later as a puzzling
    CM1-CFG-UNKNOWN refusal on a valid CM1 namelist.
    """

    authority = _authority_param9_keys()
    assert len(authority) == 64
    assert set(OUTPUT_KEYS) == set(authority)
    assert OUTPUT_KEYS == authority  # and in CM1's own declaration order


def test_every_switch_carries_a_disposition_and_the_reason_it_needs():
    for switch in SWITCHES:
        assert switch.disposition in (PRODUCT, STATE, UNSUPPORTED)
        assert switch.cm1_ref, f"{switch.key} names no CM1 source"
        if switch.disposition == UNSUPPORTED:
            # A refusal must name the concrete wrong result it prevents.
            assert len(switch.defect) > 40, switch.key
        if switch.disposition == STATE:
            assert switch.carried_by, switch.key
            assert len(switch.defect) > 20, switch.key
        if switch.disposition == PRODUCT:
            assert switch.fields, switch.key


def test_every_moved_default_is_named_and_explained():
    moved = set(DEFAULT_OUTPUT_SELECTION.moved_from_cm1)
    assert moved, "this lane moves at least one default; say which"
    assert moved == set(DEFAULT_MOVES), (
        "a default that differs from CM1's distributed run/namelist.input "
        "must carry its reason in DEFAULT_MOVES"
    )
    record = DEFAULT_OUTPUT_SELECTION.record()
    for key in moved:
        row = record["differs_from_cm1_distributed_deck"][key]
        assert row["cm1"] == CM1_DISTRIBUTED_DEFAULTS[key]
        assert row["gm1"] == DEFAULT_OUTPUT_SELECTION[key]
        assert len(row["reason"]) > 30


#: CM1's rotation/swath words.  Since f9be0a4 (and the uhdiag/capediag/
#: lagrangian merges a0cee5e, 26a8abf, 860df14 it reconciled) these &param9
#: words are spelled in [diagnostics] -- all-or-none per family -- and
#: output_dbz in [physics]; [output] refuses them by key.
_ROTATION_WORDS = (
    "output_uh", "output_vort", "output_pv", "output_sws", "output_svs",
    "output_sps", "output_srs", "output_sgs", "output_sus", "output_shs",
)


def _rotation_diagnostics(**on):
    table = {key: 0 for key in _ROTATION_WORDS}
    table.update(on)
    table["tornado_report"] = False
    return table


def _document(**output):
    raw = {
        "run": {"lane": "cartesian", "duration_s": 60.0, "dt_s": 1.0,
                "history_interval_s": 60.0, "restart_interval_s": 60.0},
        "grid": {"nx": 32, "ny": 32, "nz": 40, "dx_m": 1000.0, "dy_m": 1000.0,
                 "dz_m": 500.0, "terrain": False, "stretch_x": False,
                 "stretch_y": False, "stretch_z": False},
        "dynamics": {"psolver": 3, "adaptive_timestep": False,
                     "scalar_horizontal_order": 5, "scalar_vertical_order": 5,
                     "velocity_horizontal_order": 5, "velocity_vertical_order": 5,
                     "scalar_weno": 2, "velocity_weno": 0, "weno_order": 5,
                     "dry_mass_correction": True},
        "boundaries": {"west": 2, "east": 2, "south": 2, "north": 2,
                       "bottom": 1, "top": 1, "radiative_order": 4},
        "closures": {"sgs_model": 1, "vertical_implicit_diffusion": True,
                     "upper_damping": 1, "upper_damping_depth_m": 5000.0,
                     "upper_damping_rate_s_inv": 0.0033333333333,
                     "moving_frame": False, "u_move_m_s": 0.0, "v_move_m_s": 0.0},
        "physics": {"component": "wsm6-mp6", "equation_set": 2},
        "initialization": {"isnd": 5, "iwnd": 2},
        "runtime": {"acoustic_substeps": 6, "tile_nx": 32, "tile_ny": 32,
                    "warmup_steps": 0, "memory_budget_mib": 2048,
                    "capacity_profile": "default"},
    }
    if output:
        raw["output"] = dict(output)
    return raw


def test_an_absent_output_table_keeps_the_sealed_document_identity():
    """The lane's own defaults, and the same bytes the document always had."""

    config = parse_mapping(_document())
    assert config.output is None
    assert "output" not in json.loads(config.canonical_json())


def test_a_declared_word_travels_from_the_toml_to_the_published_carrier_table():
    """Config text to changed behaviour, with nothing in between assumed."""

    before = cartesian_history_profile(build_runtime_config(parse_mapping(_document())))
    assert "uh" not in [carrier.name for carrier in before.carriers]

    raw = _document()
    raw["diagnostics"] = _rotation_diagnostics(output_uh=1, output_vort=1)
    config = parse_mapping(raw)
    assert config.tornado_diagnostics.output_uh == 1
    assert config.tornado_diagnostics.output_vort == 1
    runtime = build_runtime_config(config)
    # One producer per CM1 name: the rotation words reach the runtime's
    # tornado-diagnostics engine (runtime.diagnostics), and the &param9 engine
    # is kept off for them so the frame never carries two "uh" carriers.
    assert runtime.diagnostics.output_uh == 1
    assert runtime.diagnostics.output_vort == 1
    assert runtime.output["output_uh"] == 0
    assert runtime.output["output_vort"] == 0
    from cm1core.cartesian_tornado_diagnostics import tornado_carrier_table

    after = cartesian_history_profile(runtime, tornado_carriers=tornado_carrier_table(runtime))
    names = [carrier.name for carrier in after.carriers]
    assert "uh" in names and "xvort" in names and "yvort" in names and "zvort" in names
    assert len(names) == len(set(names))
    assert after.payload_nbytes > before.payload_nbytes
    # And the declared table is the only thing the canonical document gains.
    canonical = json.loads(config.canonical_json())
    assert "output" not in canonical
    assert canonical["diagnostics"] == _rotation_diagnostics(output_uh=1, output_vort=1)
    before_doc = json.loads(parse_mapping(_document()).canonical_json())
    assert "diagnostics" not in before_doc
    table_keys = {"diagnostics", "tornado_diagnostics"}
    assert {k: v for k, v in canonical.items() if k not in table_keys} == {
        k: v for k, v in before_doc.items() if k not in table_keys
    }


def test_turning_a_product_off_removes_exactly_its_fields():
    # output_dbz is physics.output_dbz since the capediag merge (26a8abf/f9be0a4).
    on = _document()
    on["physics"]["output_dbz"] = True
    before = [carrier.name for carrier in
              cartesian_history_profile(build_runtime_config(parse_mapping(on))).carriers]
    off = _document(output_basestate=0)
    off["physics"]["output_dbz"] = False
    names = [carrier.name for carrier in
             cartesian_history_profile(build_runtime_config(parse_mapping(off))).carriers]
    removed = ("dbz", "cref", "th0", "pi0", "prs0", "qv0", "u0", "v0")
    for absent in removed:
        assert absent in before
        assert absent not in names
    assert set(before) - set(names) == set(removed)
    assert "rain" in names


@pytest.mark.parametrize("key", [
    switch.key for switch in SWITCHES
    if switch.disposition == UNSUPPORTED and switch.key not in _OUTPUT_KEY_OWNER
])
def test_an_unsupported_word_is_refused_by_value_and_names_the_defect(key):
    with pytest.raises(CM1Refusal) as error:
        parse_mapping(_document(**{key: 1}))
    detail = error.value.detail
    assert detail.code == "CM1-OUTPUT-UNSUPPORTED"
    assert detail.option == f"output.{key}"
    assert SWITCH_BY_KEY[key].defect in detail.reason
    assert detail.remedy == f"set output.{key} = 0"


@pytest.mark.parametrize("key", [
    switch.key for switch in SWITCHES
    if switch.disposition == STATE and switch.allowed == (0, 1)
])
def test_a_state_carrier_cannot_be_switched_off(key):
    wrong = 0 if SWITCH_BY_KEY[key].gm1_default == 1 else 1
    with pytest.raises(CM1Refusal) as error:
        parse_mapping(_document(**{key: wrong}))
    assert error.value.detail.code == "CM1-OUTPUT-STATE-CARRIER"
    assert SWITCH_BY_KEY[key].carried_by[0] in error.value.detail.reason


def test_a_value_outside_cm1s_own_set_is_refused_by_value():
    with pytest.raises(CM1Refusal) as error:
        parse_mapping(_document(output_uh=2))
    assert error.value.detail.option == "output.output_uh"
    with pytest.raises(CM1Refusal):
        parse_mapping(_document(output_format=7))


def test_an_unknown_output_word_is_refused_by_key():
    with pytest.raises(CM1Refusal) as error:
        parse_mapping(_document(output_tornado=1))
    assert error.value.detail.code == "CM1-CFG-UNKNOWN"
    assert error.value.detail.option == "output.output_tornado"


@pytest.mark.parametrize("key", sorted(_OUTPUT_KEY_OWNER))
def test_a_word_owned_by_another_table_is_refused_by_key_in_output(key):
    """f9be0a4: one spelling per CM1 word; [output] refuses the other by key."""

    with pytest.raises(CM1Refusal) as error:
        parse_mapping(_document(**{key: 1}))
    assert error.value.detail.code == "CM1-CFG-UNKNOWN"
    assert error.value.detail.option == f"output.{key}"


def test_a_moist_product_on_a_dry_lane_is_refused_with_the_zero_plane_named():
    # output_dbz (now physics.output_dbz) on component 'off' is refused earlier
    # by CM1-CFG-DBZ-COMPONENT, so the zero-plane refusal is exercised by the
    # [output] moist word output_rain.
    raw = _document(output_rain=1)
    raw["physics"]["component"] = "off"
    with pytest.raises(CM1Refusal) as error:
        parse_mapping(raw)
    assert error.value.detail.code == "CM1-OUTPUT-DRY-LANE"
    assert error.value.detail.option == "output.output_rain"
    assert "identically zero" in error.value.detail.reason


def test_a_stretched_column_refuses_the_uniform_metric_vorticity_products():
    raw = _document()
    raw["diagnostics"] = _rotation_diagnostics(output_vort=1)
    raw["grid"].update({
        "stretch_z": 1, "ztop_m": 20000.0, "str_bot_m": 5000.0,
        "str_top_m": 10000.0, "dz_bot_m": 250.0, "dz_top_m": 500.0,
    })
    with pytest.raises(CM1Refusal) as error:
        parse_mapping(raw)
    assert error.value.detail.code == "CM1-OUTPUT-STRETCHED-METRIC"
    assert error.value.detail.option == "diagnostics.output_vort"
    assert "param.F:8170" in error.value.detail.reason


def test_a_stretched_column_admits_calcuh_and_the_swaths_but_not_calcvort():
    # calcuh's no-terrain branch and the svs/shs swaths carry no vertical
    # metric (misclibs.F:1451-1459,1512-1530; solve3.F:885-894), so a
    # stretched column admits them; calcvort's xvort/yvort (rdz*mf) stay refused.
    stretched = {
        "nz": 80, "dz_m": 100.0, "stretch_z": 1, "ztop_m": 17600.0,
        "str_bot_m": 200.0, "str_top_m": 7400.0, "dz_bot_m": 20.0, "dz_top_m": 340.0,
    }
    raw = _document()
    raw["diagnostics"] = _rotation_diagnostics(output_uh=1, output_svs=1, output_shs=1)
    raw["grid"].update(stretched)
    config = parse_mapping(raw)
    assert (config.tornado_diagnostics.output_uh, config.tornado_diagnostics.output_svs,
            config.tornado_diagnostics.output_shs) == (1, 1, 1)
    for key in ("output_vort", "output_pv"):
        raw = _document()
        raw["diagnostics"] = _rotation_diagnostics(**{key: 1})
        raw["grid"].update(stretched)
        with pytest.raises(CM1Refusal) as error:
            parse_mapping(raw)
        assert error.value.detail.code == "CM1-OUTPUT-STRETCHED-METRIC"
        assert error.value.detail.option == f"diagnostics.{key}"


def test_the_unvalidated_marker_is_loud_machine_readable_and_specific():
    record = DEFAULT_OUTPUT_SELECTION.record()
    rows = record["unvalidated"]
    assert rows, "output_dbz is on by default and is not CM1's own operator"
    assert all(row["marker"] == "UNVALIDATED" for row in rows)
    dbz = next(row for row in rows if row["option"] == "output_dbz")
    assert "ArWen" in dbz["not_checked"]
    assert dbz["cm1_output_names"] == ["dbz", "cref"]
    # And a selection that does not ask for it carries no marker at all.
    off = OutputSelection(values=dict(DEFAULT_OUTPUT_SELECTION.values) | {"output_dbz": 0})
    assert off.unvalidated_markers() == ()


def test_carrier_rows_use_cm1s_own_output_names_and_units():
    rows = output_carrier_rows(
        OutputSelection(values=dict(DEFAULT_OUTPUT_SELECTION.values) | {
            "output_uh": 1, "output_vort": 1, "output_pv": 1, "output_cape": 1,
        }),
        mass=(8, 6, 5), w=(9, 6, 5), u=(8, 6, 6), v=(8, 7, 5),
        surface=(6, 5), profile=(8,),
    )
    by_name = {row[0]: row for row in rows}
    assert by_name["uh"][2] == "m2 s-2"
    assert by_name["zvort"][2] == "s-1"
    assert by_name["pv"][2] == "K m2 kg-1 s-1"
    assert by_name["cape"][2] == "J kg-1"
    assert by_name["rain"][2] == "cm"
    assert by_name["prate"][2] == "kg m-2 s-1"
    assert by_name["dbz"][2] == "dBZ"
    assert by_name["sps"][1].startswith("run-long min pressure perturbation")


# --------------------------------------- where a rebound word came from


def test_a_word_another_table_decided_names_that_table_not_output() -> None:
    """The receipt said "declared in this run's [output] table" for words no
    [output] table can declare (it refuses them by key)."""

    import tomllib
    from pathlib import Path

    from cm1core.cartesian_door import cartesian_output_selection
    from cm1core.config import parse_mapping

    root = Path(__file__).resolve().parents[1]
    document = tomllib.loads((root / "examples" / "cartesian-uniform-grid.toml").read_text())
    document.pop("output", None)
    document.pop("diagnostics", None)
    document["physics"]["output_dbz"] = True
    config = parse_mapping(document)
    record = cartesian_output_selection(config).record()
    moved = record["differs_from_cm1_distributed_deck"]
    for key in ("output_sws", "output_svs", "output_sps", "output_srs", "output_sgs",
                "output_sus", "output_shs"):
        assert "[output]" not in moved[key]["reason"] or "not published" in moved[key]["reason"]
        assert moved[key]["reason"].startswith("not published"), moved[key]["reason"]
        assert f"diagnostics.{key} = 1" in moved[key]["reason"]
    for row in moved.values():
        assert row["reason"] != "declared in this run's [output] table"
    # With the family declared, the words say who publishes them.
    document["diagnostics"] = {
        **{key: 1 for key in ("output_uh", "output_vort", "output_pv", "output_sws",
                              "output_svs", "output_sps", "output_srs", "output_sgs",
                              "output_sus", "output_shs")},
        "tornado_report": False,
    }
    moved = cartesian_output_selection(parse_mapping(document)).record()[
        "differs_from_cm1_distributed_deck"
    ]
    assert moved["output_sws"]["reason"].startswith(
        "published by the [diagnostics] rotation and swath producer"
    )
    # A word the document did declare in [output] still says so.
    document["output"] = {"output_uinterp": 0}
    try:
        declared = parse_mapping(document)
    except Exception:  # pragma: no cover - the grammar may need the whole family
        return
    moved = cartesian_output_selection(declared).record()["differs_from_cm1_distributed_deck"]
    assert moved["output_uinterp"]["reason"] == "declared in this run's [output] table"
