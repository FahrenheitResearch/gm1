from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "render_gm1_restored_step2_science.py"
EVIDENCE = (
    ROOT
    / "evidence"
    / "validation"
    / "gm1-view-restored-step2-live-watch-v1"
)
SCIENCE_RECEIPT = EVIDENCE / "science-receipt.json"
PNG = EVIDENCE / "gm1-restored-step2-qcqi-science-classy-light.png"
EXPECTED_SCIENCE_RECEIPT_SHA256 = (
    "80b2c5946632d6d8c9067a791e6442b61c638f44440d09e903bf87329c163635"
)


def _module():
    spec = importlib.util.spec_from_file_location("restored_step2_science", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_morphology_counts_boundaries_and_full_levels_without_reconstruction() -> None:
    module = _module()
    field = np.zeros(module.SHAPE, dtype="<f4")
    field[10, :, :] = np.float32(2.0e-4)
    field[11, 20:-20, 20:-20] = np.float32(5.0e-5)
    result = module.morphology(field)
    assert result["full_horizontal_positive_levels"] == [10]
    assert result["onset_count"] == 90_000
    assert result["positive_count"] == 90_000 + 260 * 260
    assert result["peak_voxel_kji"] == [10, 0, 0]
    assert result["maximum_f32_bits"] == module._float32_bits(np.float32(2.0e-4))


def test_rendered_science_receipt_and_png_are_exact() -> None:
    raw = SCIENCE_RECEIPT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SCIENCE_RECEIPT_SHA256
    receipt = json.loads(raw)
    assert raw == json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    assert receipt["schema"] == "gm1.restored-step2-qcqi-science-figure/v1"
    assert receipt["status"] == "PASS_EXACT_QCQI_SCIENTIFIC_FIGURE_RENDER"
    assert receipt["synthetic_detail"] is False
    assert receipt["unavailable_fields"] == ["vertical_velocity", "vertical_vorticity"]
    assert hashlib.sha256(PNG.read_bytes()).hexdigest() == receipt["output"][
        "png_sha256"
    ]
    assert PNG.stat().st_size == receipt["output"]["png_nbytes"]
    assert receipt["refused_renderer"]["classification"] == (
        "REFUSED_VISUAL_SLAB_SOURCE_MORPHOLOGY"
    )


def test_real_morphology_exposes_source_wide_slab_without_renderer_inference() -> None:
    receipt = json.loads(SCIENCE_RECEIPT.read_bytes())
    morphology = receipt["morphology"]
    assert morphology["positive_count"] == 3_877_859
    assert morphology["onset_count"] == 2_721_998
    assert morphology["full_horizontal_positive_levels"] == list(range(46, 70))
    assert morphology["positive_fraction"] > 0.57
    assert morphology["interior_onset_fraction"] > 0.41
    assert morphology["edge_onset_fraction"] > 0.30
