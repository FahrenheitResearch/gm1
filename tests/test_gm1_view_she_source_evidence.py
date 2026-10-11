from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]

RECEIPT_HASHES = {
    "gm1-view-she-source-control-constant-v1": (
        "766b6ffa80578c42dc6dc70473ec90d3c43a746fea5b2a31b60a48c5887bdd8e"
    ),
    "gm1-view-she-source-control-upstream-belfast-v1": (
        "bffb018869648fce9f26b390939dc569a818286f3e5e894bf361d9cc6312ce32"
    ),
    "gm1-view-she-source-control-simsat-black-lower-v1": (
        "657786f44dacfd41f30b6033ee788a3e43b367fc874665d0c7d3a217cbef8db3"
    ),
    "gm1-view-she-gm1-full-sphere-v1": (
        "3c3c7a81212e123dd240141396307bc76df9ce9b026fdfb4f3f01a59dd2f44fb"
    ),
    "gm1-view-she-gm1-full-sphere-alpha8-v1": (
        "ff1ae152667a0d7cbab9aa50185d7ae2756d8766a5509370fcb5c401c854c2c9"
    ),
    "gm1-view-she-gm1-full-sphere-alpha16-v1": (
        "f73483ae83c76ea8444cc71afaeac8faa9a19474bfc947bd64de38ac362818b0"
    ),
    "gm1-view-she-gm1-full-sphere-alpha32-v1": (
        "b6a5fbf4bf784ff6494562092a76b2909d925121e86fca15907793bc78bfc9ea"
    ),
    "gm1-view-she-source-control-constant-alpha32-v1": (
        "09c3c6852b51ffb6423c0c2a0af5e37832db1c3de1e92962460112f8eb7f8baa"
    ),
    "gm1-view-she-source-control-upstream-belfast-alpha32-v1": (
        "f893e7768e1f90b37813eeabbdb86b909f9f7d9a73a0488d585bc311a9f7e463"
    ),
    "gm1-view-she-admitted-real-terrain-ab-v1": (
        "98c169520f373f45f22ac7f816cb2c870ef181ab534ba7b49da6b9c9a641c9fb"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(name: str) -> tuple[Path, dict[str, object]]:
    base = ROOT / "evidence" / "validation" / name
    path = base / "receipt.json"
    assert _sha256(path) == RECEIPT_HASHES[name]
    return base, json.loads(path.read_text(encoding="utf-8"))


def _require_output(base: Path, descriptor: dict[str, object]) -> None:
    path = base / str(descriptor["file"])
    assert path.stat().st_size == descriptor["nbytes"]
    assert _sha256(path) == descriptor["sha256"]


def _f32_words(path: Path) -> list[float]:
    payload = path.read_bytes()
    assert len(payload) % 4 == 0
    return [value[0] for value in struct.iter_unpack("<f", payload)]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_source_isolation_controls_preserve_pass_and_refusal_boundaries() -> None:
    _, constant = _receipt("gm1-view-she-source-control-constant-v1")
    _, belfast = _receipt("gm1-view-she-source-control-upstream-belfast-v1")
    _, black_lower = _receipt("gm1-view-she-source-control-simsat-black-lower-v1")

    assert constant["status"] == (
        "PASS_SOURCE_DIAGNOSTIC_UPSTREAM_DEFAULT_UNREGULARIZED_FP32"
    )
    assert constant["held"]["normalized_rms_absolute_radiance"] < 0.20
    assert constant["held"]["normalized_rms_luminance_absolute"] < 0.20
    assert belfast["status"] == (
        "REFUSED_SOURCE_DIAGNOSTIC_UPSTREAM_DEFAULT_HELD_ERROR"
    )
    assert belfast["held"]["normalized_rms_absolute_radiance"] > 0.20
    assert belfast["held"]["normalized_rms_luminance_absolute"] < 0.20
    assert black_lower["status"] == (
        "REFUSED_SOURCE_DIAGNOSTIC_UPSTREAM_DEFAULT_HELD_ERROR"
    )
    assert black_lower["held"]["normalized_rms_absolute_radiance"] > 0.30
    assert black_lower["held"]["normalized_rms_luminance_absolute"] > 0.30
    for receipt in (constant, belfast, black_lower):
        assert receipt["held"]["admission_max_normalized_rms_radiance"] == _f32(
            0.20
        )
        assert receipt["held"]["admission_max_normalized_rms_luminance"] == _f32(
            0.20
        )
        assert receipt["upstream"]["collector_resolution"] == 256
        assert receipt["upstream"]["ggx_samples"] == 4096


def test_gm1_alpha_ladder_stops_at_source_specific_alpha32_admission() -> None:
    _, alpha4 = _receipt("gm1-view-she-gm1-full-sphere-v1")
    _, alpha8 = _receipt("gm1-view-she-gm1-full-sphere-alpha8-v1")
    _, alpha16 = _receipt("gm1-view-she-gm1-full-sphere-alpha16-v1")
    base32, alpha32 = _receipt("gm1-view-she-gm1-full-sphere-alpha32-v1")

    assert alpha4["status"] == (
        "REFUSED_SOURCE_DIAGNOSTIC_UPSTREAM_DEFAULT_HELD_ERROR"
    )
    assert len(alpha4["upstream"]["alpha_levels"]) == 4
    for levels, receipt in ((8, alpha8), (16, alpha16)):
        assert receipt["status"] == (
            "REFUSED_UPSTREAM_MATH_ALPHA_EXTENSION_HELD_ERROR"
        )
        assert receipt["training_profile"] == "GM1_QUALITY_EXTENSION_UPSTREAM_MATH"
        assert receipt["upstream"]["matches_upstream_default"] is False
        assert receipt["upstream"]["quality_extension"] is True
        assert len(receipt["upstream"]["alpha_levels"]) == levels
        assert receipt["held"]["normalized_rms_luminance_absolute"] > 0.20

    assert alpha32["status"] == (
        "PASS_UPSTREAM_MATH_ALPHA_EXTENSION_UNREGULARIZED_FP32"
    )
    assert alpha32["training_profile"] == "GM1_QUALITY_EXTENSION_UPSTREAM_MATH"
    assert len(alpha32["upstream"]["alpha_levels"]) == 32
    assert alpha32["implementation"]["solver_status"] == [1, 256, 4096, 32]
    assert alpha32["held"]["normalized_rms_absolute_radiance"] <= 0.20
    assert alpha32["held"]["normalized_rms_luminance_absolute"] <= 0.20
    assert alpha32["admission"]["old_gate_relaxed"] is False
    assert alpha32["output"]["sha256"] == (
        "1fe4c6059a43464af304adc0a8af41ec1791104e5d6da6b7af254a12688404a2"
    )
    _require_output(base32, alpha32["output"])


def test_alpha32_controls_limit_admission_to_the_gm1_source() -> None:
    _, constant = _receipt("gm1-view-she-source-control-constant-alpha32-v1")
    _, belfast = _receipt("gm1-view-she-source-control-upstream-belfast-alpha32-v1")
    assert constant["status"] == (
        "PASS_UPSTREAM_MATH_ALPHA_EXTENSION_UNREGULARIZED_FP32"
    )
    assert constant["held"]["normalized_rms_absolute_radiance"] < 0.20
    assert constant["held"]["normalized_rms_luminance_absolute"] < 0.20
    assert belfast["status"] == (
        "REFUSED_UPSTREAM_MATH_ALPHA_EXTENSION_HELD_ERROR"
    )
    assert belfast["held"]["normalized_rms_absolute_radiance"] > 0.26
    assert belfast["held"]["normalized_rms_luminance_absolute"] > 0.25


def test_admitted_material_arrays_are_exact_and_error_is_recomputed() -> None:
    base, receipt = _receipt("gm1-view-she-gm1-full-sphere-alpha32-v1")
    material = receipt["material_ab"]
    assert material["status"] == (
        "PASS_ADMITTED_COEFFICIENTS_APPLIED_TO_EXACT_GM1_TERRAIN_NORMALS"
    )
    assert material["grid"]["source_payload_sha256"] == (
        "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91"
    )
    assert material["grid"]["sampled_shape_yx"] == [60, 60]
    assert material["material"]["microfacet_alpha"] == _f32(0.22)
    assert material["material"]["reference_ggx_samples"] == 4096
    payloads = material["payloads"]
    for descriptor in payloads.values():
        _require_output(base, descriptor)
    reference = _f32_words(
        base / payloads["reference_linear_rgb_f32le"]["file"]
    )
    she = _f32_words(base / payloads["she_linear_rgb_f32le"]["file"])
    error = _f32_words(
        base / payloads["absolute_error_rgb_f32le"]["file"]
    )
    assert len(reference) == len(she) == len(error) == 10_800
    assert error == [_f32(abs(_f32(a - b))) for a, b in zip(reference, she, strict=True)]
    assert all(math.isfinite(value) for value in reference + she)
    assert min(reference) >= 0.0 and min(she) >= 0.0


def test_classy_light_terrain_ab_binds_refusal_and_visual_hash() -> None:
    base, receipt = _receipt("gm1-view-she-admitted-real-terrain-ab-v1")
    assert receipt["status"] == (
        "PASS_DISPLAY_AB_FROM_ADMITTED_GM1_SHE_COEFFICIENTS"
    )
    assert receipt["source"]["receipt_sha256"] == RECEIPT_HASHES[
        "gm1-view-she-gm1-full-sphere-alpha32-v1"
    ]
    assert receipt["calibration_scope"]["pinned_belfast_alpha32"]["status"] == (
        "REFUSED_UPSTREAM_MATH_ALPHA_EXTENSION_HELD_ERROR"
    )
    assert "not a general operator claim" in receipt["calibration_scope"][
        "pinned_belfast_alpha32"
    ]["disposition"]
    presentation = receipt["presentation"]
    png = base / presentation["file"]
    assert _sha256(png) == presentation["sha256"] == (
        "d9c30e3d89bdcd7e9cd63cecb1624174eda78f1201d3471a5804a68b40467805"
    )
    assert presentation["style"]["theme"] == "classy-light"
    assert presentation["style"]["font_family"] == "DejaVu Sans"
    assert presentation["style"]["dpi"] >= 200
    png_bytes = png.read_bytes()
    assert png_bytes[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    width, height = struct.unpack(">II", png_bytes[16:24])
    assert (width, height) == (4032, 1584)
    figure_copy = " ".join(
        [
            "Admitted glossy SHE on exact GM1 child terrain",
            receipt["claim_limit"],
        ]
    ).casefold()
    assert "tornado" not in figure_copy
    assert "globally superior" not in figure_copy
