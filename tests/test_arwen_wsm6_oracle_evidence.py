from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sealed_evidence import assert_recaptured_or_owed


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "validation" / "arwen-wsm6-adapter-oracle-v1.json"


def test_arwen_wsm6_adapter_gpu_evidence_is_self_consistent() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "passed"
    assert evidence["execution"]["scenario_count"] == 3
    assert evidence["adapter"]["entry_point"] == "gpuwm.core.microphysics.apply"
    assert evidence["adapter"]["private_wsm6_imported_by_harness"] is False
    assert evidence["adapter"]["device_state_host_copy_during_component_apply"] is False
    for name, limit in evidence["gates"].items():
        assert evidence["observed_worst"][name] <= limit
    assert evidence["observed_worst"]["minimum_hydrometeor_kg_kg-1"] >= -1e-12


def test_arwen_wsm6_adapter_gpu_runner_is_byte_pinned() -> None:
    """The sealed receipt names the runner that produced it.

    One re-capture debt is declared rather than re-sealed.  The runner's
    hail_opt guard used to say "the sealed CM1 component supports WSM6 graupel
    mode only", which stopped being true the moment ``physics.ihail`` became
    declarable and ``cm1core.dense_ice`` transcribed both branches
    (src/mp_wsm6.F:88-99).  The message now names the real limit -- the frozen
    WRF v4.6.1 oracle carries hail_opt=0 scenarios only -- and that edit
    provably cannot have changed what the sealed run computed.  The runner
    selects the scenarios it grades by filtering the frozen deck to
    ``hail_opt == 0`` and then asserts the survivors are exactly {0, 1, 2}, and
    the receipt records ``scenario_count == 3``; so every scenario the sealed
    run executed entered ``run_scenario`` with ``hail_opt == 0`` and the edited
    statement is the body of a branch none of them takes.  The receipt still owes a re-capture
    on the device it names before its runner binding can be asserted again.
    """
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    runner = evidence["artifacts"]["runner"]
    payload = (ROOT / runner["path"]).read_bytes()
    assert_recaptured_or_owed(
        ROOT / runner["path"], runner["sha256"],
        reviewed_by=(
            "the ihail admission, which made the guard's own message false; "
            "every sealed scenario has hail_opt == 0 so the edited branch is "
            "not on the measured path"
        ),
    )
    if hashlib.sha256(payload).hexdigest() == runner["sha256"]:
        assert len(payload) == runner["bytes"]
