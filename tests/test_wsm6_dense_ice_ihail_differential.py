"""The ihail differential: CM1 r21.1's own WSM6 against GM1's, measured.

The recorded evidence lives in ``evidence/micro/wsm6-dense-ice-ihail/``.  Two
of the three checks below are live: where ``gfortran`` and the read-only CM1
r21.1 authority tree with its compiled objects are present, the CM1 half is
rebuilt and re-run and its answer must still be the recorded one.  The GM1 half
needs a CUDA device and an admitted ``gpuwm``, so it is compared against its
recorded receipt rather than re-run here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "micro" / "wsm6-dense-ice-ihail"
CM1_ROOTS = (
    ROOT / "work" / "native-reference" / "cm1r21.1",
    ROOT / "upstream" / "cm1-21.1" / "pristine",
)


def _cm1_root() -> Path | None:
    for root in CM1_ROOTS:
        if (root / "src" / "mp_wsm6.o").is_file() and (root / "src" / "mp_wsm6.mod").is_file():
            return root
    return None


def _load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_the_recorded_differential_says_both_models_move_the_same_way() -> None:
    """The measured answer, and what it is an answer to.

    CM1 src/param.F:3697-3706 hands ihail to WSM6 as hail_opt and
    src/mp_wsm6.F:86-100 turns it into n0g, deng, avtg, bvtg and lamdagmax.
    The question this receipt answers is whether gpuwm-cm1's physics.ihail
    reaches that same branch, and the measurement is a 120 s column, which is
    three crossings of one 250 m layer at the term's own fall speed.
    """

    receipt = _load("ihail-differential.json")
    assert receipt["term"] == "dense-ice sedimentation"
    cm1, gm1 = receipt["cm1"], receipt["gm1"]
    # Hail falls further than graupel in both, which is the whole physical
    # content of the selector.
    assert cm1["hail_displacement_m"] > cm1["graupel_displacement_m"]
    assert gm1["hail_displacement_m"] > gm1["graupel_displacement_m"]
    assert cm1["ratio"] > 1.5 and gm1["ratio"] > 1.5
    # The responses agree to better than 15%, which is what "the same branch"
    # means across two different implementations of WSM6.
    assert receipt["result"]["ratio_relative_difference"] < 0.15
    assert receipt["result"]["ihail_response_agrees"] is True


def test_the_receipt_names_the_divergence_it_also_found() -> None:
    """A measurement that turned up something else says so rather than hiding it."""

    receipt = _load("ihail-differential.json")
    divergence = receipt["divergence_found"]
    assert divergence["term"] == "depositional growth of the dense-ice category"
    cm1_qg = divergence["measured"]["cm1_qg_kg_m2"]
    gm1_qg = divergence["measured"]["gm1_qg_kg_m2"]
    # CM1 nearly doubles the graupel burden in 120 s; ArWen's WSM6 does not.
    assert cm1_qg[1] / cm1_qg[0] > 1.8
    assert gm1_qg[1] / gm1_qg[0] < 1.05
    assert "nothing in this pass established its cause" in divergence["statement"]


@pytest.mark.skipif(
    _cm1_root() is None or shutil.which("gfortran") is None,
    reason="CM1 r21.1 authority tree with compiled objects, or gfortran, absent",
)
def test_cm1s_own_wsm6_still_gives_the_recorded_answer() -> None:
    """Rebuild and re-run the CM1 half; the oracle must not have drifted."""

    import sys

    sys.path.insert(0, str(ROOT / "tools" / "oracle"))
    try:
        import wsm6_dense_ice_sedimentation_oracle as oracle
    finally:
        sys.path.pop(0)

    root = _cm1_root()
    assert root is not None
    import tempfile
    import subprocess

    with tempfile.TemporaryDirectory() as scratch:
        binary = oracle.build(root, Path(scratch))
        completed = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True
        )
    live = oracle.receipt(oracle.parse(completed.stdout))
    recorded = _load("cm1-wsm6-column.json")
    assert live == recorded


def test_every_bound_component_has_been_dispatched_once() -> None:
    """The recorded column smoke: what runs, and what that does and does not say.

    Two components were removed from the selectable set because this
    measurement refused them -- classic Thompson needed a column
    vertical-velocity carrier that did not exist, and ArWen's WDM6 needs WRF's
    XLAND mask, which CM1's idealized lane has no field for.  That is the whole
    value of dispatching: a binding that looks right on paper is not one.
    """

    from cm1core.column_components import COLUMN_COMPONENT_SPECS

    smoke = _load("column-component-smoke.json")
    assert "nothing about agreement with CM1" in smoke["claim_limit"]
    recorded = {row["component_id"]: row for row in smoke["components"]}
    assert set(recorded) == set(COLUMN_COMPONENT_SPECS)
    for component_id, spec in COLUMN_COMPONENT_SPECS.items():
        row = recorded[component_id]
        assert row["dispatched"] is True, component_id
        assert row["numq"] == spec.profile.numq, component_id
        assert row["finite"] is True, component_id
        assert row["negative_mass_rows"] == [], component_id
        # The seeded dense ice falls, in every scheme, which is the one
        # physical statement a single column can make.
        assert row["dense_ice_displacement_m"] > 0.0, component_id
