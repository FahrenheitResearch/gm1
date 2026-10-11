"""Reproducible ensemble members, from TOML text to the words actually added.

CM1's ensemble switch is ``irandp`` (``README.namelist:692-695``) and
``src/init3d.F:1446-1483`` is the operation: ``tha += amplitude*(2*rand-1)``
over the whole domain, with the amplitude set in the Fortran source and the
stream taken from an unseeded ``random_number``.  These tests hold this port to
that shape, and to the one thing it adds: a member that can be re-run.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from cm1core.cartesian_door import cartesian_plan_payload, require_ensemble
from cm1core.config import load_config, parse_mapping
from cm1core.ensemble_branch import (
    CM1_IRANDP_AMPLITUDE_K,
    ENSEMBLE_BRANCH_SCHEMA,
    ENSEMBLE_PERTURBED_CARRIERS,
    EnsembleBranchError,
    apply_ensemble_branch,
    splitmix64_words,
    theta_perturbation_field,
)
from cm1core.errors import CM1Refusal

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "cartesian-uniform-grid.toml"


def cartesian_mapping() -> dict[str, Any]:
    payload = load_config(EXAMPLE).canonical_dict()
    return {
        section: copy.deepcopy(payload[section])
        for section in (
            "run", "grid", "dynamics", "boundaries", "closures", "physics", "runtime"
        )
    }


def ensemble_mapping(**overrides: Any) -> dict[str, Any]:
    raw = cartesian_mapping()
    table = {"member": 1, "seed": 20260914, "theta_amplitude_k": CM1_IRANDP_AMPLITUDE_K}
    table.update(overrides)
    raw["ensemble"] = table
    return raw


# --------------------------------------------------------------------------
# the generator, stated by value
# --------------------------------------------------------------------------


def test_splitmix64_matches_its_published_reference_stream() -> None:
    """Seed 0's first four words, so the stream is a fact and not a hope."""

    words = [int(word) for word in splitmix64_words(0, 4)]
    assert words == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
        0xF88BB8A8724C81EC,
    ]


def test_a_seed_reproduces_its_member_and_a_different_seed_does_not() -> None:
    numpy = pytest.importorskip("numpy")
    shape = (3, 4, 5)
    first = theta_perturbation_field(seed=7, amplitude_k=0.25, shape=shape)
    again = theta_perturbation_field(seed=7, amplitude_k=0.25, shape=shape)
    other = theta_perturbation_field(seed=8, amplitude_k=0.25, shape=shape)
    assert numpy.array_equal(first.view(numpy.uint32), again.view(numpy.uint32))
    assert not numpy.array_equal(first, other)


def test_the_field_is_cm1s_own_shape_and_bound() -> None:
    numpy = pytest.importorskip("numpy")
    field = theta_perturbation_field(seed=11, amplitude_k=0.5, shape=(4, 6, 8))
    assert field.dtype == numpy.dtype("float32")
    assert field.shape == (4, 6, 8)
    assert field.flags.c_contiguous, "CM1's k, j, i loop order is C order here"
    assert float(field.min()) >= -0.5 and float(field.max()) <= 0.5
    # amplitude*(2*rand-1) with rand uniform in [0,1) spans the interval.
    assert float(field.min()) < -0.4 and float(field.max()) > 0.4


def test_a_nonpositive_amplitude_is_refused_by_the_generator() -> None:
    with pytest.raises(EnsembleBranchError):
        theta_perturbation_field(seed=1, amplitude_k=0.0, shape=(2, 2, 2))


# --------------------------------------------------------------------------
# grammar
# --------------------------------------------------------------------------


def test_the_ensemble_table_binds_a_member() -> None:
    config = parse_mapping(ensemble_mapping(member=4, seed=99))
    assert config.ensemble is not None
    assert (config.ensemble.member, config.ensemble.seed) == (4, 99)
    assert config.ensemble.theta_amplitude_k == CM1_IRANDP_AMPLITUDE_K


def test_no_ensemble_table_is_the_control_and_seals_as_it_always_did() -> None:
    config = load_config(EXAMPLE)
    assert config.ensemble is None
    assert "ensemble" not in config.canonical_dict()
    assert config.sha256 == (
        "60b552c0a1cccd9174bc839d15b1258aac7f97999f8fd206be74a51975f1db48"
    )
    assert require_ensemble(config) is None


@pytest.mark.parametrize(
    ("key", "bad", "code"),
    (
        ("mystery", 1, "CM1-CFG-UNKNOWN"),
        ("member", -1, "CM1-CFG-INTEGER"),
        ("seed", -1, "CM1-CFG-INTEGER"),
        ("seed", 2**64, "CM1-CFG-ENSEMBLE-SEED"),
        ("theta_amplitude_k", 0.0, "CM1-CFG-ENSEMBLE-AMPLITUDE"),
        ("theta_amplitude_k", -0.25, "CM1-CFG-ENSEMBLE-AMPLITUDE"),
    ),
)
def test_ensemble_keys_and_values_refuse_by_name(key: str, bad: Any, code: str) -> None:
    raw = ensemble_mapping()
    raw["ensemble"][key] = bad
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == code


def test_the_zero_amplitude_refusal_names_the_control() -> None:
    raw = ensemble_mapping(theta_amplitude_k=0.0)
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert "declares no [ensemble] table" in caught.value.detail.reason
    assert "init3d.F:1466" in caught.value.detail.reason


def test_the_ensemble_table_is_declared_whole_or_not_at_all() -> None:
    raw = ensemble_mapping()
    del raw["ensemble"]["seed"]
    with pytest.raises(CM1Refusal) as caught:
        parse_mapping(raw)
    assert caught.value.detail.code == "CM1-CFG-MISSING-KEY"


def test_a_member_may_branch_from_the_controls_restart() -> None:
    """The ensemble table is outside the restart identity, deliberately."""

    control = parse_mapping(cartesian_mapping())
    member = parse_mapping(ensemble_mapping())
    assert control.sha256 != member.sha256
    assert control.restart_identity_sha256 == member.restart_identity_sha256


def test_two_members_differ_in_the_sealed_document() -> None:
    first = parse_mapping(ensemble_mapping(member=1, seed=1))
    second = parse_mapping(ensemble_mapping(member=2, seed=2))
    assert first.sha256 != second.sha256


# --------------------------------------------------------------------------
# the door
# --------------------------------------------------------------------------


def test_the_door_names_the_branch_point_and_cm1s_amplitude() -> None:
    record = require_ensemble(parse_mapping(ensemble_mapping()))
    assert record is not None
    assert record["branch_point"] == (
        "the initialized state before the first large step"
    )
    assert record["cm1_irandp_amplitude_k"] == CM1_IRANDP_AMPLITUDE_K
    assert record["perturbed_carriers"] == list(ENSEMBLE_PERTURBED_CARRIERS)
    assert "init3d.F:1446-1483" in record["authority"]


def test_the_door_names_a_restart_branch_point_when_resuming() -> None:
    raw = ensemble_mapping()
    raw["run"].update({"duration_s": 20.0, "history_interval_s": 5.0,
                       "restart_interval_s": 10.0})
    # irst/rstnum/restart_reset_frqtim are [run] words (config.py
    # _OPTIONAL_RUN_KEYS); a [restart] table is refused as unknown.
    raw["run"].update({"irst": 1, "rstnum": 2, "restart_reset_frqtim": True})
    record = require_ensemble(parse_mapping(raw))
    assert record is not None
    assert record["branch_point"] == "the restart named by run.rstnum"


def test_the_plan_prices_the_member_without_a_device() -> None:
    payload = cartesian_plan_payload(parse_mapping(ensemble_mapping(member=6)))
    assert payload["ensemble"]["member"] == 6
    assert payload["ensemble"]["cells"] == 96 * 96 * 40


# --------------------------------------------------------------------------
# what is actually added to the carriers
# --------------------------------------------------------------------------


class _Registry:
    def __init__(self, buffers: dict[str, Any]) -> None:
        self._buffers = buffers

    @property
    def buffers(self) -> dict[str, Any]:
        return self._buffers


class _Dynamics:
    def __init__(self, phase: tuple[str, int]) -> None:
        self._expected_phase = phase


class _Clock:
    completed_steps = 120


class _Runtime:
    def __init__(self, phase: tuple[str, int] = ("held", 1)) -> None:
        numpy = pytest.importorskip("numpy")
        from cm1core.cartesian_door import build_runtime_config

        raw = cartesian_mapping()
        raw["grid"].update({"nx": 16, "ny": 16, "nz": 16})
        raw["runtime"].update({"tile_nx": 16, "tile_ny": 16})
        self.config = build_runtime_config(parse_mapping(raw))
        shape = (16, 16, 16)
        self.registry = _Registry(
            {
                "field.tha": numpy.full(shape, 1.5, dtype=numpy.float32),
                "field.th3d": numpy.full(shape, 1.5, dtype=numpy.float32),
            }
        )
        self.dynamics = _Dynamics(phase)
        self.clock = _Clock()
        self.synchronized = 0

    def synchronize(self) -> None:
        self.synchronized += 1


def test_the_branch_adds_cm1s_perturbation_to_both_theta_carriers() -> None:
    numpy = pytest.importorskip("numpy")
    runtime = _Runtime()
    receipt = apply_ensemble_branch(
        runtime, member=3, seed=4242, theta_amplitude_k=0.25
    )
    field = theta_perturbation_field(seed=4242, amplitude_k=0.25, shape=(16, 16, 16))
    expected = numpy.float32(1.5) + field
    for name in ENSEMBLE_PERTURBED_CARRIERS:
        assert numpy.array_equal(
            runtime.registry.buffers[name].view(numpy.uint32),
            expected.view(numpy.uint32),
        ), f"{name} did not receive CM1's amplitude*(2*rand-1)"
    assert receipt.schema == ENSEMBLE_BRANCH_SCHEMA
    assert receipt.member == 3 and receipt.seed == 4242
    assert receipt.cells == 4096
    assert receipt.branched_from_completed_steps == 120
    assert -0.25 <= receipt.minimum_k < 0.0 < receipt.maximum_k <= 0.25
    assert runtime.synchronized == 1
    payload = receipt.as_dict()
    assert payload["field_sha256"] == receipt.field_sha256
    assert "not re-derived" in payload["claim_limit"]


def test_the_receipt_hashes_the_field_that_was_actually_added() -> None:
    import hashlib

    runtime = _Runtime()
    receipt = apply_ensemble_branch(
        runtime, member=0, seed=5, theta_amplitude_k=0.5
    )
    field = theta_perturbation_field(seed=5, amplitude_k=0.5, shape=(16, 16, 16))
    assert receipt.field_sha256 == hashlib.sha256(field.tobytes(order="C")).hexdigest()


def test_the_branch_refuses_anywhere_but_a_completed_step_boundary() -> None:
    runtime = _Runtime(phase=("pre", 2))
    with pytest.raises(EnsembleBranchError) as caught:
        apply_ensemble_branch(runtime, member=1, seed=1, theta_amplitude_k=0.25)
    assert "completed-large-step boundary" in str(caught.value)


def test_the_branch_refuses_a_runtime_without_the_theta_carriers() -> None:
    runtime = _Runtime()
    del runtime.registry.buffers["field.th3d"]
    with pytest.raises(EnsembleBranchError) as caught:
        apply_ensemble_branch(runtime, member=1, seed=1, theta_amplitude_k=0.25)
    assert "field.th3d" in str(caught.value)
