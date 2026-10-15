"""From TOML text to the words the boundary kernels receive.

Each test here starts from configuration text a user could write and ends at
something that observably changed: the bound kernel's argument list, the
runtime contract's own selection, the run receipt, or a refusal that names the
wrong result it prevents.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cm1core.boundary_validation_register import register, unvalidated_for
from cm1core.cartesian_door import build_runtime_config
from cm1core.config import load_config
from cm1core.cuda_boundary_families import (
    DEFAULT_LATERAL_BOUNDARY_SELECTION,
    KIND_SCALAR,
    LateralBoundarySelection,
    bind_lateral_families,
)
from cm1core.errors import CM1Refusal


ROOT = Path(__file__).resolve().parents[1]
#: The shipped Cartesian deck: the only lane that reaches the CUDA runtime, so
#: the only one whose boundary selection can change a kernel argument.
PRODUCT = ROOT / "examples" / "cartesian-supercell-attempt.toml"


@pytest.fixture(autouse=True)
def _scratch(tmp_path_factory):
    global _SCRATCH
    _SCRATCH = tmp_path_factory.mktemp("boundary-config")


_SCRATCH: Path | None = None


def _write(text: str) -> Path:
    assert _SCRATCH is not None
    path = _SCRATCH / f"case-{abs(hash(text)) % 10**12}.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _text(**edits: str) -> str:
    """The shipped product deck with named lines replaced or appended."""

    lines = PRODUCT.read_text().splitlines()
    remaining = dict(edits)
    out: list[str] = []
    section = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else None
        dotted = f"{section}.{key}" if key else None
        if dotted in remaining:
            out.append(f"{key} = {remaining.pop(dotted)}")
            continue
        out.append(line)
    for dotted, value in remaining.items():
        section, key = dotted.split(".", 1)
        index = out.index(f"[{section}]")
        out.insert(index + 1, f"{key} = {value}")
    return "\n".join(out) + "\n"


def _config(**edits: str):
    return load_config(_write(_text(**edits)))


class _Recorder:
    """A stand-in for a compiled CUDA kernel that only remembers its arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, grid, block, arguments, **kwargs):
        self.calls.append(tuple(arguments))


class _FakeFamilyCompiled:
    target_arch = "86"
    source_sha256 = "f" * 64

    def __init__(self) -> None:
        self.kernels = tuple(_Recorder() for _ in range(18))


class _FakeOpenCompiled:
    """Enough of ``CompiledOpenBoundaryKernels`` for ``dataclasses.replace``."""

    def __init__(self) -> None:
        from dataclasses import dataclass, field

        raise NotImplementedError


def test_a_document_that_names_periodic_laterals_binds_periodic_kernels(monkeypatch) -> None:
    from dataclasses import replace

    from cm1core.cuda_sound3_open_boundary import CompiledOpenBoundaryKernels

    config = _config(**{
        "boundaries.west": "1", "boundaries.east": "1",
        "boundaries.south": "1", "boundaries.north": "1",
    })
    runtime = build_runtime_config(config)
    selection = runtime.lateral_boundary_selection
    assert selection.sides == (1, 1, 1, 1)

    sealed = CompiledOpenBoundaryKernels(
        module_owner=None,
        kernels=tuple(_Recorder() for _ in range(10)),
        ptx=b"", cubin=b"", target_arch="86",
        ptx_options=(), cubin_options=(), runtime_identity=(),
    )
    family = _FakeFamilyCompiled()

    class _U32(int):
        pass

    monkeypatch.setattr(
        "cm1core.cuda_boundary_families.importlib.import_module",
        lambda name: type("cupy", (), {"uint32": _U32})() if name == "cupy" else __import__(name),
    )
    bound = bind_lateral_families(sealed, family, selection)
    # The call site is unchanged: it still hands the sealed six arguments.
    bound.kernels[5]((1,), (128,), (object(), 32, 32, 40, 3, KIND_SCALAR))
    arguments = family.kernels[0].calls[0]
    # ... and the two CM1 side values travel with the binding.
    assert arguments[-2:] == (1, 1)
    assert bound.kernels[0] is sealed.kernels[0]  # the sealed prepare is untouched


def test_the_sealed_open_family_binds_the_same_two_values_it_always_had() -> None:
    config = load_config(PRODUCT)
    runtime = build_runtime_config(config)
    assert runtime.lateral_boundary_selection == DEFAULT_LATERAL_BOUNDARY_SELECTION
    assert runtime.lateral_boundary_selection.all_open


def test_each_new_key_reaches_the_runtime_contract_by_cm1s_own_name() -> None:
    config = _config(**{
        "boundaries.roflux": "1",
        "boundaries.nudgeobc": "1",
        "boundaries.alphobc": "45.0",
        "boundaries.cstar": "100.0",
        "boundaries.radiative_order": "2",
        "closures.hrdamp": "1",
        "closures.xhd": "4000.0",
    })
    runtime = build_runtime_config(config)
    selection = runtime.lateral_boundary_selection
    assert (selection.roflux, selection.nudgeobc, selection.alphobc, selection.cstar) == (
        1, 1, 45.0, 100.0
    )
    assert selection.irbc == 2
    assert (runtime.hrdamp, runtime.xhd_m) == (1, 4000.0)
    record = runtime.record()
    assert record["lateral_boundaries"]["roflux"] == 1
    assert record["hrdamp"] == 1


def test_an_unset_cstar_is_rewritten_exactly_as_param_f_722_does() -> None:
    config = _config(**{
        "boundaries.roflux": "0", "boundaries.nudgeobc": "0",
        "boundaries.alphobc": "60.0", "boundaries.cstar": "0.0",
    })
    assert config.boundaries.cstar == 0.0  # the document says what it says
    assert build_runtime_config(config).lateral_boundary_selection.cstar == 30.0


def test_irdamp_zero_zeroes_the_rate_as_param_f_724_does() -> None:
    config = _config(**{"closures.upper_damping": "0"})
    assert config.closures.upper_damping_rate_s_inv > 0.0
    assert build_runtime_config(config).damping_rate_s_inv == 0.0


def test_a_document_that_declares_none_of_the_new_keys_keeps_its_sealed_identity() -> None:
    before = load_config(PRODUCT)
    assert "roflux" not in before.canonical_dict()["boundaries"]
    assert "hrdamp" not in before.canonical_dict()["closures"]
    # Every shipped example is sealed by this hash, and recorded GPU receipts
    # carry it as plan_sha256.
    assert before.sha256 == load_config(_write(_text())).sha256


def test_declaring_one_companion_without_the_rest_is_refused_by_name() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**{"boundaries.roflux": "1"})
    assert caught.value.detail.code == "CM1-CFG-BOUNDARY-COMPANION-PARTIAL"
    assert "boundaries.alphobc" in caught.value.detail.remedy


def test_a_lateral_value_cm1_does_not_have_is_refused_with_its_consequence() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**{"boundaries.west": "5"})
    assert caught.value.detail.code == "CM1-CFG-UNSUPPORTED-VALUE"
    assert "ghost planes" in caught.value.detail.reason


def test_a_rigid_wall_under_a_translating_frame_is_refused_as_param_f_1906_stops() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**{
            "boundaries.west": "3", "boundaries.east": "3",
            "closures.moving_frame": "true",
            "closures.u_move_m_s": "12.5", "closures.v_move_m_s": "3.0",
        })
    assert caught.value.detail.code == "CM1-CFG-MOVING-FRAME-RIGID-WALL"
    assert "slide the wall through" in caught.value.detail.reason
    # The remedy names both sides, so the import report's CM1 words read "wbc and ebc".
    assert "boundaries.west and boundaries.east periodic or open" in caught.value.detail.remedy


@pytest.mark.parametrize("sides", [(1, 2, 2, 2), (2, 1, 2, 2), (1, 3, 1, 1), (4, 1, 1, 1),
                                   (2, 2, 1, 2), (2, 2, 3, 1), (1, 1, 1, 4), (1, 2, 1, 3)])
def test_a_periodic_side_without_a_periodic_opposite_side_is_refused_as_cm1_stops(sides) -> None:
    """param.F:2562-2588: "Can not have periodic b.c.'s on one side only!" stops CM1 for each of these."""

    west, east, south, north = sides
    with pytest.raises(CM1Refusal) as caught:
        _config(**{
            "boundaries.west": str(west), "boundaries.east": str(east),
            "boundaries.south": str(south), "boundaries.north": str(north),
        })
    detail = caught.value.detail
    assert detail.code == "CM1-CFG-PERIODIC-ONE-SIDE"
    assert "src/param.F:2562-2588" in detail.reason and "one side only" in detail.reason
    x_bad = (west == 1) != (east == 1)
    assert detail.option == ("boundaries.west" if west == 1 else "boundaries.east") if x_bad else (
        "boundaries.south" if south == 1 else "boundaries.north")
    assert ("boundaries.west and boundaries.east" if x_bad else "boundaries.south and boundaries.north") in (
        detail.remedy)


def test_the_one_sided_periodic_refusal_reads_in_cm1_words_in_the_import_report(tmp_path) -> None:
    from cm1import_support import SUPERCELL, edited, namelist_tool
    from cm1core.cm1import.coverage import build_report, import_for_report

    report = build_report(import_for_report(edited(SUPERCELL, tmp_path, {"wbc": "1", "ebc": "2"}),
                                            tool=namelist_tool()))
    rows = [row for row in report.rows if row.code == "CM1-CFG-PERIODIC-ONE-SIDE"]
    assert len(rows) == 1 and rows[0].category == "refused" and rows[0].keys == ("wbc",)
    assert "wbc and ebc" in rows[0].fix and "boundaries." not in rows[0].fix
    assert rows[0].plain == "Repeating sides come in pairs: west with east, and south with north."
    assert report.exit_code == 2


def test_a_rigid_wall_is_admitted_once_the_frame_is_at_rest() -> None:
    config = _config(**{"boundaries.west": "3", "boundaries.east": "4"})
    assert build_runtime_config(config).lateral_boundary_selection.sides == (3, 4, 2, 2)


def test_a_sponge_over_more_than_half_the_domain_is_refused_as_param_f_8689_stops() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**{"closures.hrdamp": "1", "closures.xhd": "1.0e9"})
    assert caught.value.detail.code == "CM1-CFG-XHD-DEPTH"
    assert "more than half" in caught.value.detail.reason


def test_hrdamp_that_disagrees_with_irdamp_is_refused_rather_than_rewritten() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**{"closures.hrdamp": "2", "closures.xhd": "4000.0"})
    assert caught.value.detail.code == "CM1-CFG-HRDAMP-FOLLOWS-IRDAMP"


def test_nudgeobc_without_an_open_side_is_refused() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**{
            "boundaries.west": "1", "boundaries.east": "1",
            "boundaries.south": "1", "boundaries.north": "1",
            "boundaries.roflux": "0", "boundaries.nudgeobc": "1",
            "boundaries.alphobc": "60.0", "boundaries.cstar": "30.0",
        })
    assert caught.value.detail.code == "CM1-CFG-NUDGEOBC-WITHOUT-OPEN-SIDE"


def test_tbc_2_and_3_are_refused_with_the_free_slip_lid_they_would_produce() -> None:
    with pytest.raises(CM1Refusal) as caught:
        _config(**{"boundaries.top": "3"})
    assert caught.value.detail.code == "CM1-CFG-UNSUPPORTED-VALUE"
    assert "no drag at all" in caught.value.detail.reason


def test_the_receipt_block_names_every_unvalidated_capability_the_run_turns_on() -> None:
    config = _config(**{
        "boundaries.west": "1", "boundaries.east": "1",
        "boundaries.south": "3", "boundaries.north": "4",
        "closures.upper_damping": "2",
    })
    block = unvalidated_for(config.boundaries, config.closures)
    assert "wbc/ebc/sbc/nbc=3,4 (rigid walls)" in block["unvalidated"]
    # A periodic side is compared with CM1, and the comparison found its edge
    # column differing: it is published as not checked, with the differential.
    periodic = block["unvalidated"]["wbc/ebc/sbc/nbc=1 (periodic)"]
    assert "760 to 840 words a call" in periodic["not_checked"]
    assert periodic["differential"].startswith("evidence/") and periodic["caveat"]
    assert "wbc/ebc/sbc/nbc=1 (periodic)" not in block["validated"]
    assert "irdamp=2 (damp toward the horizontal average)" in block["validated"]
    for entry in block["validated"].values():
        assert entry["differential"].startswith("evidence/")
        assert entry["caveat"]
    for entry in block["unvalidated"].values():
        assert entry["not_checked"]
        assert entry["cm1"].startswith("src/")
    # The block is JSON, so a consumer never has to parse prose.
    json.dumps(block, allow_nan=False)


@pytest.mark.parametrize("sides", [(3, 3, 4, 4), (1, 1, 3, 3), (1, 1, 1, 1)])
def test_roflux_and_irbc_with_no_open_side_are_recorded_as_having_no_effect(sides) -> None:
    """CM1 accepts roflux = 1 and irbc = 1 or 2 with no open side, and they change nothing there."""

    from cm1core.cuda_boundary_families import LateralBoundarySelection

    west, east, south, north = sides
    config = _config(**{
        "boundaries.west": str(west), "boundaries.east": str(east),
        "boundaries.south": str(south), "boundaries.north": str(north),
        "boundaries.radiative_order": "2", "boundaries.roflux": "1", "boundaries.nudgeobc": "0",
        "boundaries.alphobc": "60.0", "boundaries.cstar": "30.0",
    })
    record = build_runtime_config(config).lateral_boundary_selection.record()
    assert set(record["no_effect"]) == {"roflux", "irbc"}
    assert "bc.F:2180-2391" in record["no_effect"]["roflux"]
    assert "sound.F:274-302" in record["no_effect"]["irbc"]
    # The receipt names no rescale the run never had.
    block = unvalidated_for(config.boundaries, config.closures)
    assert "roflux=1 (restrict outward mass flux)" not in block["capabilities"]
    # With an open side both act, and the record is what it always was.
    opened = LateralBoundarySelection(west=2, east=2, south=south, north=north, irbc=2, roflux=1)
    assert "no_effect" not in opened.record()
    assert "no_effect" not in LateralBoundarySelection().record()


def test_the_import_report_says_roflux_and_irbc_do_nothing_with_no_open_side(tmp_path) -> None:
    from cm1import_support import SUPERCELL, edited, namelist_tool
    from cm1core.cm1import.coverage import build_report, import_for_report

    report = build_report(import_for_report(edited(SUPERCELL, tmp_path, {
        "wbc": "3", "ebc": "3", "sbc": "4", "nbc": "4", "imove": "0", "roflux": "1", "irbc": "2",
    }), tool=namelist_tool()))
    rows = {row.keys: row for row in report.rows if row.category == "same" and row.text.startswith("No effect")}
    assert set(rows) == {("roflux",), ("irbc",)}
    assert "bc.F:2180-2391" in rows[("roflux",)].text and "sound.F:274-302" in rows[("irbc",)].text
    assert report.exit_code == 0
    # With an open side there is no such row.
    report = build_report(import_for_report(edited(SUPERCELL, tmp_path, {"roflux": "1", "irbc": "2"},
                                                   name="open.input"), tool=namelist_tool()))
    assert not [row for row in report.rows if row.text.startswith("No effect")]


def test_the_sealed_open_family_run_carries_no_unvalidated_marker() -> None:
    config = load_config(PRODUCT)
    block = unvalidated_for(config.boundaries, config.closures)
    assert block["unvalidated_count"] == 0
    assert block["validated"]


def test_doctor_lists_every_unvalidated_boundary_capability() -> None:
    from cm1core.doctor import UNVALIDATED, collect, render, unvalidated

    findings = collect()
    names = {item.subject for item in unvalidated(findings)}
    expected = {
        f"CM1 lateral boundary: {name}"
        for name, entry in register().items()
        if entry["differential"] is None or entry["not_checked"] is not None
    }
    assert "CM1 lateral boundary: wbc/ebc/sbc/nbc=1 (periodic)" in expected
    assert expected <= names
    for item in unvalidated(findings):
        assert item.status == UNVALIDATED
        assert "NOT checked" in item.detail
        assert item.required is False  # exposed and running, never blocking
    assert "unvalidated exposed capabilities:" in render(findings)


def test_the_register_covers_every_selectable_capability() -> None:
    # A capability that can be selected and is absent from the register would
    # run with no marker at all, which is the failure this register prevents.
    assert len(register()) == 9
    for entry in register().values():
        assert entry["cm1"].startswith("src/")
        assert entry["checked"]
        # A capability has a named CM1 differential behind it, a sentence
        # saying what was not checked, or both: a differential that found a
        # part differing from CM1 states that part, and its caveat.
        assert entry["differential"] is not None or entry["not_checked"] is not None
        if entry["differential"] is not None and entry["not_checked"] is not None:
            assert entry["caveat"]


@pytest.mark.parametrize("sides", [(1, 1, 1, 1), (2, 2, 1, 1), (3, 3, 1, 1), (2, 3, 4, 2), (1, 1, 3, 4)])
def test_every_cm1_lateral_family_is_expressible(sides) -> None:
    west, east, south, north = sides
    config = _config(**{
        "boundaries.west": str(west), "boundaries.east": str(east),
        "boundaries.south": str(south), "boundaries.north": str(north),
        "closures.moving_frame": "false",
        "closures.u_move_m_s": "0.0", "closures.v_move_m_s": "0.0",
    })
    selection = build_runtime_config(config).lateral_boundary_selection
    assert selection.sides == sides
    assert (selection.ibw, selection.ibe, selection.ibs, selection.ibn) == tuple(
        0 if value == 1 else 1 for value in sides
    )
