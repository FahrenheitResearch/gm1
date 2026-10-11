"""The dynamics validation receipt describes artifacts that are still there.

A receipt whose byte identities have drifted from the tree describes a run
nobody can reproduce, so the seal is re-checked on every test run rather than
trusted.  The register it summarises is checked against the same source, so a
capability cannot be added to the configuration grammar without a line here
saying what stands behind it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from measured_source import measured_identity
from cm1core.dynamics_validation_register import (
    DYNAMICS_CAPABILITIES,
    INERT,
    UNVALIDATED,
    VALIDATED,
    dynamics_validation_record,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "evidence/validation/cm1-dynamics-cm1-words-v1/validation-receipt.json"


def _receipt() -> dict:
    return json.loads(RECEIPT.read_bytes())


def test_every_named_artifact_is_present_with_the_bytes_it_names() -> None:
    receipt = _receipt()
    for group in ("evidence", "artifacts"):
        assert receipt[group], group
        for name, item in receipt[group].items():
            path = ROOT / item["path"]
            assert path.is_file(), f"{group}.{name}: {item['path']} is gone"
            # A module reworded since the seal is compared as the receipt
            # measured it (tests/measured_source.py); every other byte counts.
            size, digest = measured_identity(path)
            assert size == item["bytes"], f"{group}.{name} bytes"
            assert digest == item["sha256"], f"{group}.{name} sha256"


def _capture() -> dict:
    return json.loads((ROOT / _receipt()["execution"]["capture_record"]).read_bytes())


def test_the_receipt_names_the_cuda_device_every_run_was_measured_on() -> None:
    """Any CUDA device will do, if the receipt names it and agrees with itself.

    The receipt claims word agreement with CM1 r21.1, which does not depend on
    a particular card.  It was first measured on an RTX 5070 Ti and measured
    again on an RTX 4090 once its source pins went stale.  So this test
    requires no particular device.  It requires that the device is named, and
    that the device every run recorded, the method text and the execution
    block all name the same one.
    """

    receipt = _receipt()
    capture = _capture()
    execution = receipt["execution"]
    device = capture["device"]["name"]
    assert device.startswith("NVIDIA ")
    assert execution["gpu_used"] is True and execution["cpu_only"] is False
    assert execution["gpu_name"] == device
    assert execution["gpu"].startswith(f"{device} at ")
    assert execution["gpu_compute_capability"] == capture["device"]["compute_capability"]
    assert device.removeprefix("NVIDIA GeForce ") in receipt["method"]
    assert receipt["captures"]["current"]["device"] == device
    runs = capture["runs"]
    assert execution["gm1_runs"] == execution["cm1_runs"] == len(runs)
    assert {run["gm1"]["device"] for run in runs} == {device}
    assert all(run["cm1"]["terminated_normally"] for run in runs)
    assert all(
        run["gm1"]["status"] == "executed-cartesian-integration-to-requested-duration"
        for run in runs
    )


def test_every_pinned_source_is_the_one_the_captured_runs_executed() -> None:
    """A receipt pin is the byte identity of what ran, not of what is on disk.

    Each GM1 run receipt carries the digest of every cm1core file it executed,
    and the capture record keeps the digests of *every* source this receipt
    pins -- the runtime files and the two harness files
    (``src/cm1core/oracle/grads.py``, ``src/cm1core/oracle/spec.py``) the
    receipt's own ``harness_defects_fixed`` block blames for two of its four
    defects, which is exactly where "the bytes the runs consumed" has to mean
    something.  A source edited after the capture is a historical pin: it is
    listed in ``pinned_sources_changed_after_capture`` with what changed, its
    live digest must differ from the executed one, and re-pinning it silently
    -- which would claim the runs read bytes that did not exist -- fails here.
    The case files are the ones the runs translated, and every run pair a
    table names is in the record.
    """

    receipt = _receipt()
    capture = _capture()
    executed = capture["source"]["pinned_sources_as_executed"]
    changed = capture["source"]["pinned_sources_changed_after_capture"]
    assert set(changed) <= set(executed)
    cases = {run["case_file"]: run["case_sha256"] for run in capture["runs"]}
    source_pins = 0
    for name, item in receipt["artifacts"].items():
        path = item["path"]
        if path in executed:
            source_pins += 1
            if path in changed:
                assert len(changed[path].split()) >= 10, name
                assert executed[path] != item["sha256"], name
            else:
                assert executed[path] == item["sha256"], name
        elif path.startswith("examples/oracle/") and path.endswith(".differential.toml"):
            assert cases[path] == item["sha256"], name
    # Every source the receipt pins is in the record: no pin gets to sit
    # outside the comparison by not being listed.
    assert source_pins == len(executed) >= 17
    assert {
        item["path"]
        for item in receipt["artifacts"].values()
        if not item["path"].endswith(".differential.toml")
    } == set(executed)
    assert set(cases) == {
        item["path"]
        for item in receipt["artifacts"].values()
        if item["path"].endswith(".differential.toml")
    }
    # The adaptive runs published every frame on the step CM1 wrote it at, and
    # raised getdbz on exactly those steps.
    for run in capture["runs"]:
        assert run["gm1"]["getdbz_steps"] == run["gm1"]["frame_steps"], run["case"]
        assert run["gm1"]["frame_steps"] == run["cm1"]["nstep"][1:], run["case"]
        assert run["gm1"]["frame_model_time_s"] == run["cm1"]["model_time_s"][1:], run["case"]
    adaptive = receipt["result"]["adaptive_timestep"]
    assert adaptive["same_frame_steps_as_cm1"] is True
    assert adaptive["getdbz_raised_on_the_frame_steps"] is True
    assert adaptive["gm1_large_steps"] == 142


def test_the_receipt_says_a_production_default_moved() -> None:
    """alph 0.5 -> 0.60 changes every psolver=3 result; it must be declared."""

    receipt = _receipt()
    assert receipt["execution"]["production_default_changed"] is True
    moved = {item["key"]: item for item in receipt["defaults_moved"]}
    assert moved["dynamics.alph"]["from"] == 0.5
    assert moved["dynamics.alph"]["to"] == 0.60


def test_every_configuration_key_in_the_grammar_is_in_the_register() -> None:
    """A selector with no line here would be reachable and undescribed."""

    from cm1core.config import _DYNAMICS_CM1_WORDS

    keys = {item.key for item in DYNAMICS_CAPABILITIES}
    assert {f"dynamics.{name}" for name in _DYNAMICS_CM1_WORDS} <= keys
    assert "dynamics.adaptive_timestep" in keys


def test_every_register_row_names_an_authority_and_a_check() -> None:
    for item in DYNAMICS_CAPABILITIES:
        assert item.status in {VALIDATED, UNVALIDATED, INERT}
        assert item.authority.strip()
        assert item.validation.strip()
        assert item.not_checked.strip()


#: Register rows admitted after the cm1-dynamics-cm1-words-v1 receipt was
#: sealed.  That receipt describes the runs it made and cannot name a key that
#: did not exist; each row here says in its own not_checked clause which
#: measurement is still owed.
ADMITTED_AFTER_RECEIPT = frozenset({"dynamics.lspgrad"})


def test_the_receipt_and_the_register_name_the_same_unvalidated_keys() -> None:
    receipt_keys = {item["key"] for item in _receipt()["unvalidated"]}
    register_keys = {item.key for item in DYNAMICS_CAPABILITIES if item.status == UNVALIDATED}
    # adapt_dt is validated as a controller and carries its own not_checked
    # clause, so the receipt names it among the caveats without the register
    # calling the whole capability unvalidated.
    assert register_keys - ADMITTED_AFTER_RECEIPT <= receipt_keys
    assert ADMITTED_AFTER_RECEIPT.isdisjoint(receipt_keys)


@pytest.mark.parametrize(
    "name", ["cartesian-uniform-grid.toml", "cartesian-supercell-translating-frame.toml"]
)
def test_a_shipped_document_reports_its_own_unvalidated_keys(name: str) -> None:
    from cm1core.config import load_config

    document = load_config(ROOT / "examples" / name)
    record = dynamics_validation_record(document.dynamics)
    # Both documents run CM1's kdiv = 0.10, the value every measured run used,
    # so the register is value-aware and flags nothing: kdiv is still
    # reported, as a capability the run depends on, with the value and the
    # measured values beside its status.
    assert document.dynamics.kdiv == 0.10
    assert record["unvalidated"] is False and record["unvalidated_keys"] == []
    rows = {item["key"]: item for item in record["capabilities"]}
    assert rows["dynamics.kdiv"]["status"] == VALIDATED
    assert rows["dynamics.kdiv"]["value"] == 0.10
    assert rows["dynamics.kdiv"]["measured_values"] == [0.10]
    # alph has been validated against CM1 at 0.5 and 0.60
    # (evidence/oracle/dynamics-cm1-words/alph-default-move.txt); it is still
    # reported, as a validated capability the run depends on.
    assert "dynamics.alph" in rows
    # An inert word must not be reported as something the run depends on.
    assert "dynamics.difforder" not in rows
    # Any other kdiv is the capability no run has measured.
    from dataclasses import replace

    moved = dynamics_validation_record(replace(document.dynamics, kdiv=0.35))
    assert moved["unvalidated"] is True and moved["unvalidated_keys"] == ["dynamics.kdiv"]
    assert {item["key"]: item for item in moved["capabilities"]}["dynamics.kdiv"][
        "status"
    ] == UNVALIDATED


def test_doctor_prints_the_unvalidated_keys_as_loudly_as_a_missing_one() -> None:
    from cm1core.doctor import UNVALIDATED_MARKER, collect, render

    findings = collect()
    rendered = render(findings, explain=True)
    unvalidated = [item for item in findings if item.status == UNVALIDATED]
    assert {item.subject for item in unvalidated} >= {"CM1 kdiv (dynamics.kdiv)"}
    assert "CM1 alph (dynamics.alph)" not in {item.subject for item in unvalidated}
    for item in unvalidated:
        # f9be0a4 made "NOT checked" (doctor.UNVALIDATED_MARKER) the one token
        # every UNVALIDATED finding carries, whatever its producer spelled.
        assert UNVALIDATED_MARKER == "NOT checked"
        assert UNVALIDATED_MARKER in item.detail
        assert f"UNVALID {item.subject}" in rendered
    # ... and it must not make the installation look broken.
    from cm1core.doctor import blocking

    assert all(not item.required for item in unvalidated)
    assert not any(item.status == UNVALIDATED for item in blocking(findings))
