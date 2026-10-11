from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from cm1core.gm1_child_stretched_diagnostics import (
    GM1_CHILD_STRETCHED_DIAGNOSTICS_SCHEMA,
    SOURCE_ANCHORS,
    evaluate_gm1_child_stretched_diagnostics,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "evidence/validation/gm1-real-50m-child-step600-diagnostics-v1"
)
RECEIPT_SHA256 = (
    "23373814a8e8be6e2721794445f4012958ac1ffdf0f0b5da9f17b51de8eb0c2d"
)
PNG_SHA256 = "95d6aa945880b68831598384c2bc004f23bb8be95d11d674230a45c8ad6e5164"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stretched_diagnostic_interface_is_fail_closed_and_source_bound() -> None:
    assert GM1_CHILD_STRETCHED_DIAGNOSTICS_SCHEMA == (
        "gm1.real-50m-child-stretched-diagnostics/v1"
    )
    assert SOURCE_ANCHORS == {
        "terrain_vertical_vorticity": "src/misclibs.F:2035-2107",
        "mass_centered_w": "src/misclibs.F:1579-1587",
        "vertical_grid": "src/param.F:7814-7953,8162-8225",
    }
    source = (
        ROOT / "src/cm1core/gm1_child_stretched_diagnostics.py"
    ).read_text(encoding="utf-8")
    assert "cuda_storm_diagnostics" not in source
    assert "nearest-interior outer display closure" in source
    with pytest.raises(ValueError, match="sealed to the admitted"):
        evaluate_gm1_child_stretched_diagnostics({}, ni=301)
    with pytest.raises(ValueError, match="missing stretched diagnostic carrier u3d"):
        evaluate_gm1_child_stretched_diagnostics({})


def test_step600_scientific_receipt_seals_refusal_and_no_tornado_claim() -> None:
    receipt_path = EVIDENCE / "receipt.json"
    raw = receipt_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RECEIPT_SHA256
    receipt = json.loads(raw)
    assert raw == json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert receipt["status"] == "REFUSED_BOUNDARY_CONTINUITY_DIAGNOSTIC"
    assert receipt["edge_continuity"]["all_fields_pass"] is False
    updraft = receipt["edge_continuity"]["fields"]["low_level_updraft_ms"]
    assert updraft["outer_edge_below_or_equal_domain_p99_9"] is False
    assert updraft["outer_edge_neighbor_gradient_max"] > (
        updraft["domain_neighbor_gradient_p99_9"]
    )
    assert receipt["collocation"]["same_cell_count"] == 1601
    tornado = receipt["tornado_gate"]
    assert tornado["necessary_condition_overlap_present"] is True
    assert tornado["classification_allowed"] is False
    assert tornado["status"] == "NOT_CLASSIFIED_NO_SEALED_TORNADO_DETECTOR"
    assert receipt["moisture_gate"]["q3d_finite"] is True
    assert receipt["moisture_gate"]["q3d_nonnegative"] is True
    assert receipt["diagnostic_fields"]["cloud_condensate_qc_plus_qi"][
        "nonnegative"
    ] is True


def test_step600_classy_png_is_exact_and_visibly_refusal_labeled() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text(encoding="utf-8"))
    png_path = ROOT / receipt["png"]["path"]
    raw = png_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PNG_SHA256
    assert receipt["png"]["sha256"] == PNG_SHA256
    assert receipt["png"]["nbytes"] == len(raw)
    assert receipt["png"]["refusal_annotation"].startswith(
        "Boundary-continuity audit refused"
    )
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert raw[12:16] == b"IHDR"
    assert struct.unpack(">II", raw[16:24]) == (3072, 2472)
    sources = receipt["producer_sources"]
    assert sources["diagnostic_module_sha256"] == _sha256(
        ROOT / "src/cm1core/gm1_child_stretched_diagnostics.py"
    )
    assert sources["plot_style_module_sha256"] == _sha256(
        ROOT / "src/cm1core/gm1_scientific_plot_style.py"
    )
    assert sources["presentation_replay_sha256"] == _sha256(
        ROOT / "tools/rerender_gm1_child_step600_diagnostics.py"
    )
