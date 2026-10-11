from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cm1_marks

from cm1core import engine_pin


ROOT = Path(__file__).resolve().parents[1]
#: The record that admitted the move to the currently pinned estate.
AUDIT = ROOT / "evidence" / "validation" / "arwen-wsm6-estate-delta-audit-2.8.0.json"
#: The records that admitted the previous estates.  History, not authority.
SUPERSEDED_AUDIT = ROOT / "evidence" / "validation" / "arwen-wsm6-estate-delta-audit.json"
SUPERSEDED_2_7_4_AUDIT = (
    ROOT / "evidence" / "validation" / "arwen-wsm6-estate-delta-audit-2.7.4.json")
#: The WRF v4.6.1 grading receipt.  It names the assembled NVRTC source it
#: graded, which is how a version bump can be shown not to invalidate it.
ADAPTER_ORACLE = ROOT / "evidence" / "validation" / "arwen-wsm6-adapter-oracle-v1.json"

#: The estates the delta audit compares, preserved outside the distribution:
#: each is a published wheel extracted as is.
BASE_ESTATE = ROOT / "tmp" / "arwen-estate-gpuwm-2.7.4"
ADMITTED_ESTATE = ROOT / f"tmp/arwen-estate-gpuwm-{engine_pin.GPUWM_MEASURED_VERSION}"

#: The WSM6 files whose bytes decide a WSM6 answer.  The moves to 2.7.4 and
#: 2.8.0 were taken because every one of them is byte-identical to what the
#: WRF v4.6.1 adapter oracle graded; if that ever stops being true the seam needs a new
#: oracle before it needs a new pin.
WSM6_NUMERICAL_PATHS = (
    "gpuwm/core/kernels/wsm6.cu",
    "gpuwm/core/kernels/refl.cu",
    "gpuwm/core/kernels/common.cuh",
    "gpuwm/core/wsm6.py",
    "gpuwm/core/wsm6_constants.py",
    "gpuwm/core/constants.py",
    "gpuwm/core/morrison_constants.py",
)


def _audit() -> dict:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_delta_audit_admits_exactly_the_estate_engine_pin_admits() -> None:
    audit = _audit()
    assert audit["schema"] == "gpuwm-cm1.arwen-wsm6-estate-delta-audit/v2"
    admitted = audit["admitted"]
    assert admitted["distribution_version"] == engine_pin.GPUWM_MEASURED_VERSION
    assert admitted["source_identity"] == engine_pin.GPUWM_AUDITED_SOURCE_IDENTITY
    assert admitted["distribution_sha256"] == engine_pin.GPUWM_AUDITED_DISTRIBUTION_SHA256
    assert admitted["requirement"] == engine_pin.GPUWM_REQUIREMENT
    assert audit["base"]["distribution_version"] == "2.7.4"
    assert audit["base"]["source_identity"] == (
        engine_pin.GPUWM_SUPERSEDED_2_7_4_SOURCE_IDENTITY)

    paths = audit["candidate_surface_delta"]["paths"]
    assert set(paths) == set(engine_pin.AUDITED_WSM6_SURFACE)
    for relative, expected in engine_pin.AUDITED_WSM6_SURFACE.items():
        assert paths[relative]["admitted_sha256"] == expected


def test_the_version_bump_did_not_move_wsm6() -> None:
    """The finding the whole byte pin exists to produce.

    Seven of the seventeen admitted paths differ between 2.7.4 and 2.8.0 (six
    changed, one is new), so the move is not inert by inspection.  It is inert because every file that
    decides a WSM6 answer is byte-identical, the assembled NVRTC source is
    identical at every capacity tier the seam can select, and every mechanical
    executed-path check passes.  Assert all three: any one of them alone would
    let a real change through.
    """
    audit = _audit()

    for relative in WSM6_NUMERICAL_PATHS:
        row = audit["candidate_surface_delta"]["paths"][relative]
        assert row["changed"] is False, relative
        assert row["base_sha256"] == row["admitted_sha256"]
    assert not audit["candidate_surface_delta"]["changed_numerical_or_kernel_source_paths"]

    kernel_source = audit["executed_kernel_source"]
    assert kernel_source["default_module_source"]["identical"] is True
    assert kernel_source["default_module_source"]["admitted_sha256"] == (
        engine_pin.AUDITED_WSM6_NVRTC_SOURCE_SHA256
    )
    assert set(kernel_source["capacity_tiers"]) == {"64", "80", "128", "256"}
    assert all(row["identical"] for row in kernel_source["capacity_tiers"].values())

    assert audit["semantic_audit"]["failed_checks"] == []
    assert audit["verdict"] == {
        "every_wsm6_kernel_byte_identical": True,
        "every_executed_path_check_passed": True,
        "wsm6_answers_can_move": False,
        # The wsm6-mp6 registry object is byte-identical between 2.7.4 and
        # 2.8.0.  It changed at the 2.5.8 -> 2.7.4 move, which the superseded
        # audit records.
        "wsm6_registry_object_identical": True,
    }
    # The whole dispatch body did change (the NSSL2 fallthrough only); the
    # check that decides is the WSM6 path through it, and Morrison's path is
    # recorded beside it.
    dispatch = audit["semantic_audit"]["scheme_dispatch"]
    assert dispatch["whole_body_identical"] is False
    assert dispatch["morrison_arm_identical"] is True
    assert audit["semantic_audit"]["checks"]["scheme_dispatch_wsm6_arm_identical"] is True
    morrison = audit["morrison_surface_delta"]
    assert morrison["changed_paths"] == []
    assert morrison["module_source"]["identical"] is True
    assert morrison["answers_can_move"] is False


def test_the_one_registry_refusal_2_7_4_added_cannot_reach_the_cm1_seam() -> None:
    constraints = _audit()["semantic_audit"]["registry_constraints"]
    assert constraints["unmet_required_settings"] == []
    assert constraints["forbidden_setting_values"], (
        "2.7.4 added a forbidden-value row to wsm6-mp6; if it ever disappears "
        "this assertion is what says the audit stopped pricing one"
    )
    for field, row in constraints["forbidden_setting_values"].items():
        assert row["reachable_by_default"] is False, field
        assert row["cm1_adapter_sets_it"] is False, field


def test_the_superseded_2_5_8_audit_is_kept_but_admits_nothing() -> None:
    superseded = json.loads(SUPERSEDED_AUDIT.read_text(encoding="utf-8"))
    assert superseded["admitted"]["distribution_version"] == "2.5.8"
    assert superseded["admitted"]["commit"] == engine_pin.GPUWM_SUPERSEDED_2_5_8_COMMIT
    assert superseded["admitted"]["distribution_version"] != (
        engine_pin.GPUWM_MEASURED_VERSION
    )


def test_the_superseded_2_7_4_audit_is_kept_but_admits_nothing() -> None:
    superseded = json.loads(SUPERSEDED_2_7_4_AUDIT.read_text(encoding="utf-8"))
    assert superseded["admitted"]["distribution_version"] == "2.7.4"
    assert superseded["admitted"]["source_identity"] == (
        engine_pin.GPUWM_SUPERSEDED_2_7_4_SOURCE_IDENTITY)
    assert superseded["admitted"]["distribution_version"] != (
        engine_pin.GPUWM_MEASURED_VERSION
    )
    # The base of the current audit is exactly the estate this one admitted.
    assert _audit()["base"]["source_identity"] == (
        superseded["admitted"]["source_identity"])


def test_delta_audit_regenerates_byte_identically_from_the_preserved_estates() -> None:
    """The record is derived, so it must reproduce from its inputs.

    A pinned digest that cannot be recomputed is a claim, not a measurement.
    """
    if not BASE_ESTATE.is_dir() or not ADMITTED_ESTATE.is_dir():
        pytest.skip("the preserved ArWen estates are not staged in this checkout")

    from tools.audit_arwen_wsm6_estate_delta import _canonical, audit

    assert _canonical(audit(BASE_ESTATE, ADMITTED_ESTATE)) == AUDIT.read_bytes()


def test_live_arwen_wsm6_estate_is_byte_admitted() -> None:
    """An INSTALLED ArWen must be the admitted one, byte for byte.

    Skipped only when nothing is installed -- a CPU-only checkout with no
    dependency has no estate to inspect, which is a different condition from
    an installed estate that drifted.  An installed estate still fails here,
    which is the whole point: this is the test that catches a silent
    substitution of the physics under the seam.
    """
    if engine_pin.installed_root() is None:
        pytest.skip("no installed ArWen estate to inspect")

    inspection = engine_pin.inspect()
    assert inspection.admitted, inspection.as_dict()
    assert inspection.as_dict()["cm1_public_seam_available"] is True
    assert inspection.root is not None
    for relative, expected in engine_pin.AUDITED_WSM6_SURFACE.items():
        assert (
            hashlib.sha256((inspection.root / relative).read_bytes()).hexdigest()
            == expected
        )


#: Every LIVE file that still names the superseded 2.5.8 estate, and the
#: reason it is allowed to.  This inventory is the answer to "did the re-pin
#: reach every pin site?": a file that admits, requires, installs or inspects
#: an ArWen estate must not appear here at all, and a file that legitimately
#: records history must appear here with a reason.  Sealed evidence receipts
#: under `evidence/` are excluded from the scan entirely -- they are records of
#: runs that really did execute under 2.5.8, and rewriting them would be
#: falsifying a measurement.
#:
#: The development repository's own working documents and tools that mention
#: it are listed in ``PRIVATE_MENTIONS_RECORD``, which stays there with them.
SUPERSEDED_MENTION_INVENTORY: dict[str, str] = {
    "docs/ARWEN-WSM6-CAPACITY-256-AUTHORITY.md":
        "the capacity overlay's rebase note and its superseded digest pair",
    "tests/test_arwen_morrison_component.py":
        "a drift fixture: it hands the inspector a superseded version string "
        "and asserts the component refuses",
    "tests/test_default_configuration_is_unchanged.py":
        "records which move re-digested the CM1-ARWEN-ESTATE-DRIFT blocker",
    "src/cm1core/cuda_gm1_tilestream_column.py":
        "records which audit ARWEN_WSM6_ESTATE_AUDIT_SHA256 used to name, so "
        "the repoint onto the live audit cannot be undone by accident",
    "tests/test_cuda_gm1_tilestream_wsm6.py":
        "asserts that repoint, and says which document it moved off",
    "src/cm1core/cuda_gm1_tilestream.py":
        "a frozen DESIGN CITATION of the bytes the host store was transcribed "
        "from; it is not an import path and deliberately does not move with "
        "the physics estate",
    "src/cm1core/arwen_physics_component.py":
        "the uncoupled PBL/radiation/cumulus component surface records that it "
        "was first measured on the 2.5.8 estate before the 2.8.0 re-measure",
    "src/cm1core/engine_capacity.py":
        "explains why the patched digests did and did not move across the bump",
    "src/cm1core/engine_pin.py":
        "GPUWM_SUPERSEDED_2_5_8_COMMIT, so a historical receipt stays readable",
    "src/cm1core/records/ARWEN-CAPABILITY-SURVEY.md":
        "companion prose for that delta",
    "src/cm1core/records/CM1-ARWEN-SEAM.md":
        "the recorded 2.5.8-to-2.7.4 executed-surface finding, kept as history",
    "src/cm1core/records/CM1-RISK-REGISTER.md":
        "R-014 cites the move as the worked example of how a bump is priced",
    "tests/test_cuda_gm1_tilestream.py":
        "asserts the frozen design citation above",
    "tests/test_engine_capacity.py":
        "states which move the default lane has to be bit-identical across",
    "tests/test_engine_pin.py":
        "this file: it asserts the superseded audit and this inventory",
    "tests/test_run_real_gm1_tilestream_wsm6.py":
        "asserts a sealed receipt from a real run that executed under 2.5.8",
}

#: The development repository's entries (working documents and tools that are
#: not in the public source tree), with the same meaning as the inventory above.
PRIVATE_MENTIONS_RECORD = ROOT / "docs" / "engine-pin-private-mentions.json"
if PRIVATE_MENTIONS_RECORD.is_file():
    SUPERSEDED_MENTION_INVENTORY.update(
        json.loads(PRIVATE_MENTIONS_RECORD.read_text(encoding="utf-8"))["mentions"])

#: Where a live ArWen pin can be spelled.  Each of these must name the ADMITTED
#: estate.  Scanned as text rather than imported, because several are data or
#: documentation and the failure this catches is a pin site that was missed.
_LIVE_PIN_SITES = (
    "pyproject.toml",
    "constraints/gpu-cu13-validated.txt",
    "src/cm1core/engine_pin.py",
    "src/cm1core/preflight.py",
    "src/cm1core/records/CM1-ARWEN-SEAM.md",
    "src/cm1core/records/CM1-RISK-REGISTER.md",
    "docs/ARWEN-WSM6-CAPACITY-256-AUTHORITY.md",
    "tools/inspect_artifacts.py",
)

_SCANNED_SUFFIXES = (".py", ".toml", ".md", ".json")
_SCANNED_ROOTS = ("src", "tools", "tests", "docs")


#: Records of the tree as it was, excluded like `evidence/`: the development
#: repository's scoping facts and merge records quote the 09-13 wheel's own pin
#: as history.  The public snapshot carries none.
try:
    from dev_records import ENGINE_PIN_HISTORY_ROOTS as _RECORD_ROOTS
except ModuleNotFoundError:
    _RECORD_ROOTS = ()


def _scanned_paths() -> list[Path]:
    paths = [ROOT / "pyproject.toml", ROOT / "README.md"]
    records = tuple(_RECORD_ROOTS)
    for directory in _SCANNED_ROOTS:
        for path in sorted((ROOT / directory).rglob("*")):
            if any(path.is_relative_to(record) for record in records):
                continue
            if path.is_file() and path.suffix in _SCANNED_SUFFIXES:
                paths.append(path)
    return paths


def test_every_live_pin_site_names_the_admitted_estate() -> None:
    """The re-pin reached every place a version can be spelled.

    A missed site does not fail loudly: it installs a 2.8.0 wheel against a
    2.7.4 requirement, or documents a ceiling the packaging does not enforce.
    """
    for relative in _LIVE_PIN_SITES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert engine_pin.GPUWM_MEASURED_VERSION in text, relative
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"{engine_pin.GPUWM_REQUIREMENT}"' in pyproject
    assert f'"{engine_pin.GPUWM_DATA_REQUIREMENT}"' in pyproject
    constraints = (ROOT / "constraints/gpu-cu13-validated.txt").read_text(encoding="utf-8")
    assert f"gpuwm=={engine_pin.GPUWM_MEASURED_VERSION}\n" in constraints
    assert f"gpuwm-data=={engine_pin.GPUWM_MEASURED_VERSION}\n" in constraints
    assert engine_pin.GPUWM_REQUIREMENT in (
        ROOT / "tools/inspect_artifacts.py").read_text(encoding="utf-8")
    assert engine_pin.GPUWM_AUDITED_DISTRIBUTION_SHA256 in (
        ROOT / "src/cm1core/records/CM1-ARWEN-SEAM.md").read_text(encoding="utf-8")

    # The GM1 admission ledger is a CM1-selector record and carries no ArWen
    # version at all.  Asserted rather than assumed: if it ever grows one, it
    # becomes a pin site and has to join the list above.
    ledger = (ROOT / "src/cm1core/records/GM1-PHYSICS-SUPPORT.json").read_text(
        encoding="utf-8")
    assert "gpuwm>=" not in ledger and "gpuwm==" not in ledger


def test_no_live_file_still_admits_the_superseded_estate() -> None:
    """Enumerate every live mention of 2.5.8, and make each one say why.

    Sealed receipts under `evidence/` are excluded: they record runs that
    really executed under 2.5.8 and must not be rewritten.
    """
    marker = engine_pin.GPUWM_SUPERSEDED_2_5_8_COMMIT
    found = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in _scanned_paths()
        if "2.5.8" in (text := path.read_text(encoding="utf-8", errors="replace"))
        or marker in text
    }
    declared = set(SUPERSEDED_MENTION_INVENTORY)
    if not (ROOT / "evidence").is_dir():
        # The public source tree leaves out the development repository's
        # private trees and the tests that read them; what is present must
        # still match exactly.
        declared = {path for path in declared if (ROOT / path).exists()}
    assert found == declared, {
        "undeclared": sorted(found - declared),
        "declared_but_clean": sorted(declared - found),
    }
    # Nothing may require, install or admit the superseded window.  The two
    # needles are assembled rather than written out, so this file does not
    # match its own assertion.
    superseded = "2.5.8"
    ceiling = "2.5.9"
    needles = (f"gpuwm>={superseded}", f"gpuwm<{ceiling}")
    for path in _scanned_paths():
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            assert needle not in text, (path, needle)


def test_the_seam_record_states_the_delta_audit_it_cites() -> None:
    """Prose and measurement must not drift apart.

    Every number the seam record quotes is regenerated by the audit tool, so
    the record is checkable rather than merely readable.
    """
    audit = _audit()
    seam = (ROOT / "src/cm1core/records/CM1-ARWEN-SEAM.md").read_text(encoding="utf-8")

    checks = audit["semantic_audit"]["checks"]
    assert len(checks) == 13
    assert "thirteen mechanical executed-path checks" in seam

    assert audit["executed_kernel_source"]["default_module_source"]["admitted_sha256"] in seam
    for row in audit["executed_kernel_source"]["capacity_tiers"].values():
        assert row["admitted_sha256"] in seam
    registry = audit["wsm6_registry_object"]
    assert registry["base_sha256"] in seam
    assert registry["admitted_sha256"] in seam
    assert engine_pin.GPUWM_REQUIREMENT in seam

    changed = audit["candidate_surface_delta"]["changed_paths"]
    assert len(changed) == 7
    for relative in changed:
        assert relative in seam, relative


def test_2_8_0_compiles_the_exact_source_the_wrf_oracle_graded() -> None:
    """Why the existing WSM6 evidence survives the bump.

    The adapter oracle graded ONE assembled NVRTC source string and wrote its
    digest into its own receipt.  If 2.8.0 hands NVRTC that same string, the
    grading still describes the bytes that run; if it does not, every WSM6
    claim in this repository needs a new oracle before it needs a new pin.
    Three independent records have to agree on one number, and this asserts it
    rather than trusting any of them alone.
    """
    oracle = json.loads(ADAPTER_ORACLE.read_text(encoding="utf-8"))
    graded = oracle["authority"]["assembled_nvrtc_source_sha256"]
    assert graded == engine_pin.AUDITED_WSM6_NVRTC_SOURCE_SHA256
    kernel_source = _audit()["executed_kernel_source"]["default_module_source"]
    assert kernel_source["admitted_sha256"] == graded
    assert kernel_source["base_sha256"] == graded
    assert kernel_source["identical"] is True


def test_the_audit_prices_the_configuration_the_adapter_actually_passes() -> None:
    """The delta audit's refusal pricing is only as good as its RunConfig table.

    `tools/audit_arwen_wsm6_estate_delta.py` decides whether a new registry
    refusal can reach GM1 by comparing the forbidden values against the
    settings the CM1 adapter passes.  If the adapter starts passing a setting
    the table does not list, the audit would price a refusal as unreachable
    while a real run walks into it.  So the table is read back out of the
    adapter's own call site rather than trusted.
    """
    import ast

    from cm1core.dense_ice import ARWEN_IHAIL_OPTION as ARWEN_IHAIL_OPTIONS
    from tools.audit_arwen_wsm6_estate_delta import (
        CM1_ADAPTER_RUNCONFIG_ADMITTED_VALUES,
        CM1_ADAPTER_RUNCONFIG_SETTINGS,
        adapter_admissible_values,
    )

    #: Per-run geometry and clock.  These carry no registry constraint, so the
    #: audit deliberately does not price them.
    geometry = {"nx", "ny", "nz", "dx", "dy", "ztop", "dt", "run_seconds", "clock_dt"}

    # The one call site lives in the component-generic adapter now: the WSM6
    # and Morrison bindings are ColumnComponentSpec objects, not two call sites.
    adapter = ROOT / "src/cm1core/arwen_column_component.py"
    tree = ast.parse(adapter.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "run_config"
    ]
    assert len(calls) == 1, "the adapter builds exactly one ArWen RunConfig"
    passed = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
        if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
    }
    named = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    assert named >= geometry
    # ``mp_physics`` is the selected spec's own, not a literal, so it is named
    # but not constant-folded here; the audit table states the WSM6 value.
    assert "mp_physics" not in passed
    assert {key: value for key, value in passed.items() if key not in geometry} == {
        key: value for key, value in CM1_ADAPTER_RUNCONFIG_SETTINGS.items()
        if key not in ("mp_physics", *ARWEN_IHAIL_OPTIONS.values())
    }

    # CM1's ihail travels in one ``**`` keyword whose NAME is the component's
    # own ArWen option, so it is not a named keyword in the AST at all.  That
    # is exactly the shape that could hide a setting from the audit, so the
    # starred expression is read too and its admissible keys are checked.
    starred = [keyword for keyword in calls[0].keywords if keyword.arg is None]
    assert len(starred) == 1, "exactly one ** keyword reaches the ArWen RunConfig"
    assert set(ARWEN_IHAIL_OPTIONS.values()) & set(CM1_ADAPTER_RUNCONFIG_SETTINGS)
    for option in ARWEN_IHAIL_OPTIONS.values():
        if option in CM1_ADAPTER_RUNCONFIG_SETTINGS:
            assert adapter_admissible_values(option) == (0, 1), option
            assert option in CM1_ADAPTER_RUNCONFIG_ADMITTED_VALUES

    assert named - geometry | set(ARWEN_IHAIL_OPTIONS.values()) >= set(
        CM1_ADAPTER_RUNCONFIG_SETTINGS
    )


#: CM1 r21.1 release source.  Not distributed, so this skips on a checkout
#: without it; where it IS present, the seam record's citations are checked
#: against the bytes rather than trusted.
CM1_AUTHORITY = ROOT / "work" / "native-reference" / "cm1r21.1"

#: The `file:line` claims `CM1-ARWEN-SEAM.md` makes about CM1 r21.1, and the
#: mark (tests/cm1_marks.py) of the text that must be on the first line of each
#: cited range: the moist branch, the microphysics call, WSM6's ptype=7 branches,
#: its positive-definite repair, its tendency hand-off and its reflectivity.  A citation that
#: has drifted is worse than no citation: it sends the next reader to the wrong
#: statement and looks authoritative doing it.
CM1_SEAM_CITATIONS: tuple[tuple[str, int, str], ...] = (
    ("src/cm1.F", 3405, "bbf502dc0336765f/19"),
    ("src/cm1.F", 3416, "73723a2b36bd8f0f/14"),
    ("src/param.F", 2723, "0bd8a835f9cf1b97/29"),
    ("src/param.F", 3632, "b6162766cb5c681a/22"),
    ("src/mp_driver.F", 1472, "b6162766cb5c681a/22"),
    ("src/mp_driver.F", 1474, "2155ddae4f51a84c/20"),
    ("src/mp_driver.F", 1488, "82ba67d51e784dab/25"),
    ("src/mp_driver.F", 1516, "1dae4d2dcd7647d8/20"),
    ("src/mp_driver.F", 1563, "82dacc94d8ecc9e4/24"),
    ("src/mp_driver.F", 1631, "f8a0f468a397f384/18"),
)


def test_the_seam_records_cm1_authority_lines_that_say_what_it_claims() -> None:
    if not CM1_AUTHORITY.is_dir():
        pytest.skip(f"no CM1 r21.1 release source at {CM1_AUTHORITY}")

    seam = (ROOT / "src/cm1core/records/CM1-ARWEN-SEAM.md").read_text(encoding="utf-8")
    for relative, line, expected in CM1_SEAM_CITATIONS:
        assert f"`{relative}:{line}" in seam, f"{relative}:{line} is not cited"
        text = (CM1_AUTHORITY / relative).read_text(encoding="utf-8", errors="replace")
        actual = text.splitlines()[line - 1]
        assert cm1_marks.has(actual, expected), (relative, line)
