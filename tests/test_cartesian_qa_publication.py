"""qa is published once, by solve3, and the qv budget reads the step's own tendency.

CM1 r21.1 calls ``mp_driver`` before ``solve3`` (``cm1.F:3416,3432``), and only
``solve3.F:624-643`` equates ``qa`` to ``q3d``.  GM1's microphysics drain also
copied the sealed post-physics kernel's block ``qa`` (and the number rows)
into the domain ``qa``, so ``CartesianBudgetStage.after_microphysics`` -- which
forms CM1's step tendency ``(q3d - qa)/dt`` "while tha/qa still hold its
start" -- read ``qa == q3d`` and reported an actual qv tendency of exactly zero.

The microphysics-side copy is gone and solve3's stays (never both: each copy
was dead only because the other existed).  These tests drive the REAL runtime
dispatcher, ``CartesianFinish`` and budget stage through
:mod:`tools.oracle.host_cupy` (see
``tools/oracle/cartesian_qa_publication_identity.py``) and prove:

* every model carrier after solve3 -- and at every phase of the next step --
  is the word the double-copy placement produced; only ``qa`` between
  microphysics and solve3 differs, and there it is the step's start;
* with budgets on, the qv closure's actual tendency is ``(q3d - qa_start)/dt``
  over the budget interior, not zero, and budgets still write no model word.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="the host kernel emulator needs g++")

F32 = np.float32


@pytest.fixture(scope="module")
def harness():
    from tools.oracle import cartesian_qa_publication_identity

    return cartesian_qa_publication_identity


@pytest.mark.parametrize("component", ["wsm6-mp6", "morrison-mp10"])
def test_only_qa_before_solve3_differs_from_the_double_copy_placement(harness, component) -> None:
    fixed = harness.run(component=component)
    legacy = harness.run(component=component, legacy=True)
    assert fixed["blocks"] == 9 and fixed["numq"] == (6 if component == "wsm6-mp6" else 10)
    differing = harness.compare(legacy, fixed)
    assert differing == [
        f"step{step}.{phase}.fields.qa"
        for step in (1, 2, 3) for phase in ("microphysics", "post-physics-eos")
    ]
    words = fixed["words"]
    for step in (1, 2, 3):
        # solve3.F:624-643 for every n=1,numq, over the whole domain.
        assert words[f"step{step}.solve3.fields.qa"].tobytes() == words[f"step{step}.solve3.fields.q3d"].tobytes()
        # Between mp_driver and solve3, qa is the step's start ...
        start = words[f"step{step - 1}.solve3.fields.qa"] if step > 1 else words["step1.remove-held-and-moist-eos.fields.qa"]
        for phase in ("microphysics", "post-physics-eos"):
            assert words[f"step{step}.{phase}.fields.qa"].tobytes() == start.tobytes()
        # ... and the scheme did move q3d away from it, so the comparison bites.
        assert np.count_nonzero(words[f"step{step}.microphysics.fields.q3d"] != start) > start.size // 2
    # Every row the component transports reached qa at solve3, number rows too.
    assert np.array_equal(words["step3.solve3.fields.qa"], words["step3.solve3.fields.q3d"])


def test_the_qv_budget_actual_tendency_is_the_step_s_own(harness) -> None:
    fixed = harness.run(budgets=True)
    legacy = harness.run(budgets=True, legacy=True)
    interior = (slice(None), slice(3, 11 - 3), slice(3, 13 - 3))
    for step in (1, 2, 3):
        after = fixed["words"][f"step{step}.microphysics.fields.q3d"][0]
        start = fixed["qa_start"][f"step{step}"]
        # cm1_budget_operators.process_difference: (after - before) * (1/dt).
        expected = np.subtract(after, start, dtype=F32) * F32(1.0 / fixed["dt_s"])
        closure = fixed["closure"][f"step{step}"]
        assert closure["qvb_tendency_max_abs"] == float(np.abs(expected[interior]).max())
        assert closure["qvb_tendency_max_abs"] > 1.0e-5
        # The pre-fix placement read qa == q3d here: a tendency of exactly zero.
        assert legacy["closure"][f"step{step}"]["qvb_tendency_max_abs"] == 0.0
        # theta was never affected: tha is solve3's alone in both placements.
        assert closure["ptb_tendency_max_abs"] == legacy["closure"][f"step{step}"]["ptb_tendency_max_abs"]
    # The published budget terms themselves are unchanged; only the closure moved.
    budget_terms = [name for name in fixed["words"] if ".budget." in name]
    assert budget_terms and all(
        fixed["words"][name].tobytes() == legacy["words"][name].tobytes() for name in budget_terms
    )


def test_the_budget_receipt_says_its_qv_closure_changed_meaning() -> None:
    """A v1 receipt's qvb closure is 0.0 by construction; v2 marks the fix."""

    from cm1core.cartesian_budgets import CARTESIAN_BUDGET_SCHEMA

    assert CARTESIAN_BUDGET_SCHEMA == "gpuwm-cm1.cartesian-budgets/v2"


def test_budgets_write_no_model_word(harness) -> None:
    assert harness.compare(harness.run(), harness.run(budgets=True)) == []
