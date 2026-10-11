from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_diffuse_material_oracle_binds_absolute_luminance_and_visual_hashes() -> None:
    base = ROOT / "evidence/validation/gm1-view-she-material-t300-v1"
    receipt = _json("evidence/validation/gm1-view-she-material-t300-v1/receipt.json")
    assert receipt["status"] == "PASS_INTERNAL_DIFFUSE_L2_APPROXIMATION"
    projection = receipt["projection"]
    assert projection["coefficients"] == 9
    assert projection["compared_rgb_words"] == 12_288
    assert projection["max_absolute_w_m2"] > 0.0
    assert projection["rms_luminance_absolute_w_m2"] > 0.0
    visual = receipt["visual"]
    assert _sha256(base / visual["file"]) == visual["sha256"]


def test_reduced_glossy_fit_and_visual_remain_explicitly_refused() -> None:
    fit_base = ROOT / "evidence/validation/gm1-view-she-glossy-fit-v1"
    fit = _json("evidence/validation/gm1-view-she-glossy-fit-v1/receipt.json")
    assert fit["status"] == "REFUSED_REDUCED_PREVIEW_FIT_ERROR"
    assert fit["upstream"]["source_exact"] is False
    assert fit["upstream"]["production_default_required"] == {
        "alpha_levels": 4,
        "collector_resolution": 256,
        "execution": "upstream WGSL-equivalent compute chain",
        "ggx_samples": 4096,
        "solver": "unregularized FP32 Cholesky",
    }
    assert _sha256(fit_base / fit["output"]["file"]) == fit["output"]["sha256"]

    visual_base = ROOT / "evidence/validation/gm1-view-she-glossy-diagnostic-v1"
    visual = _json("evidence/validation/gm1-view-she-glossy-diagnostic-v1/receipt.json")
    assert visual["status"] == "REFUSED_REDUCED_PREVIEW_FIT_ERROR"
    assert _sha256(visual_base / visual["visual"]["file"]) == visual["visual"]["sha256"]


def test_upstream_default_chain_is_exactly_configured_and_refused_on_held_error() -> None:
    base = ROOT / "evidence/validation/gm1-view-she-glossy-upstream-default-v2"
    receipt = _json(
        "evidence/validation/gm1-view-she-glossy-upstream-default-v2/receipt.json"
    )
    assert receipt["status"] == "REFUSED_UPSTREAM_DEFAULT_HELD_ERROR"
    assert receipt["upstream"]["collector_resolution"] == 256
    assert receipt["upstream"]["ggx_samples"] == 4096
    assert receipt["upstream"]["alpha_levels"] == [
        0.20000000298023224,
        0.46666666865348816,
        0.7333333492279053,
        1.0,
    ]
    assert receipt["upstream"]["solver"] == "unregularized FP32 Cholesky"
    assert receipt["implementation"]["solver_status"] == [1, 256, 4096, 4]
    assert receipt["held"]["normalized_rms_absolute_radiance"] > receipt["held"][
        "admission_max_normalized_rms_radiance"
    ]
    assert receipt["held"]["normalized_rms_luminance_absolute"] > receipt["held"][
        "admission_max_normalized_rms_luminance"
    ]
    assert _sha256(base / receipt["output"]["file"]) == receipt["output"]["sha256"]
    assert receipt["output"]["corruption_negative_control_detected"] is True
