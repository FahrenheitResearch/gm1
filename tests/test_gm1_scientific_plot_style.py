from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cm1core.gm1_scientific_plot_style import (
    GM1_SCIENTIFIC_PLOT_AXES,
    GM1_SCIENTIFIC_PLOT_CANVAS,
    GM1_SCIENTIFIC_PLOT_DPI,
    GM1_SCIENTIFIC_PLOT_FONT_FAMILY,
    GM1_SCIENTIFIC_PLOT_RCPARAMS,
    GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA,
    gm1_scientific_plot_style_identity,
    replace_gm1_presentation_only,
    validate_gm1_scientific_figure_copy,
)


ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    (
        ROOT
        / "evidence/validation/gm1-child-stretched-sgs-tke-real-v1/receipt.json",
        "png",
    ),
    (
        ROOT
        / "evidence/validation/gm1-child-stretched-live-nm-def-real-v1/receipt.json",
        "png",
    ),
    (
        ROOT
        / "evidence/validation/gm1-child-stretched-scalar-turbulence-v1/receipt.json",
        "visualization",
    ),
)


def _rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def test_shared_style_is_light_pinned_and_uses_bundled_dejavu() -> None:
    identity = gm1_scientific_plot_style_identity()
    assert identity["schema"] == GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA
    assert identity["theme"] == "classy-light"
    assert identity["font_family"] == GM1_SCIENTIFIC_PLOT_FONT_FAMILY
    assert identity["font_file"] == "DejaVuSans.ttf"
    assert len(identity["font_sha256"]) == 64
    assert identity["dpi"] == GM1_SCIENTIFIC_PLOT_DPI >= 200
    assert GM1_SCIENTIFIC_PLOT_RCPARAMS["figure.facecolor"] == (
        GM1_SCIENTIFIC_PLOT_CANVAS
    )
    assert GM1_SCIENTIFIC_PLOT_RCPARAMS["axes.facecolor"] == (
        GM1_SCIENTIFIC_PLOT_AXES
    )


@pytest.mark.parametrize(
    "copy",
    (
        "Tornado diagnostic",
        "PASS_OPERATOR",
        "Status: complete",
        "Forecast skill",
        "Operator gate",
    ),
)
def test_shared_style_rejects_status_and_overclaim_copy(copy: str) -> None:
    with pytest.raises(ValueError, match="forbidden wording"):
        validate_gm1_scientific_figure_copy(copy)


def test_presentation_only_merge_refuses_scientific_hash_drift() -> None:
    sealed = {"output": {"sha256": "a"}, "png": {"sha256": "old"}}
    replay = {"output": {"sha256": "a"}, "png": {"sha256": "new"}}
    merged = replace_gm1_presentation_only(
        sealed,
        replay,
        presentation_key="png",
        protected_paths=(("output", "sha256"),),
    )
    assert merged == {"output": {"sha256": "a"}, "png": {"sha256": "new"}}
    replay["output"]["sha256"] = "b"
    with pytest.raises(ValueError, match="output.sha256"):
        replace_gm1_presentation_only(
            sealed,
            replay,
            presentation_key="png",
            protected_paths=(("output", "sha256"),),
        )


def test_three_sealed_child_pngs_share_the_light_style_and_bounded_copy() -> None:
    image_module = pytest.importorskip("PIL.Image")
    canvas = _rgb(GM1_SCIENTIFIC_PLOT_CANVAS)
    style_identities: list[dict[str, object]] = []
    for receipt_path, key in FIGURES:
        receipt = json.loads(receipt_path.read_bytes())
        presentation = receipt[key]
        png = ROOT / presentation["path"]
        assert hashlib.sha256(png.read_bytes()).hexdigest() == presentation["sha256"]
        style = presentation["style"]
        style_identities.append(style)
        assert style["schema"] == GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA
        assert style["theme"] == "classy-light"
        assert style["font_family"] == GM1_SCIENTIFIC_PLOT_FONT_FAMILY
        assert style["font_file"] == "DejaVuSans.ttf"
        assert style["dpi"] >= 200
        with image_module.open(png) as image:
            assert image.mode in {"RGB", "RGBA"}
            assert image.getpixel((0, 0))[:3] == canvas
            assert image.info["Software"] == GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA
            assert image.info["GM1PlotStyle"] == GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA
            assert min(image.info["dpi"]) >= 200.0
            copy = " ".join(
                str(image.info[name]) for name in ("Title", "Description")
            ).casefold()
            for forbidden in (
                "tornado",
                "forecast skill",
                "operator gate",
                "pass_",
                "status:",
            ):
                assert forbidden not in copy
    shared_keys = (
        "schema",
        "theme",
        "canvas_hex",
        "axes_hex",
        "text_hex",
        "font_family",
        "font_file",
        "font_sha256",
        "matplotlib_version",
        "dpi",
    )
    assert len({tuple(item[key] for key in shared_keys) for item in style_identities}) == 1


def test_presentation_refresh_preserves_sealed_scientific_output_hashes() -> None:
    sgs = json.loads(FIGURES[0][0].read_bytes())
    assert sgs["output_sha256"]["stress.uten"] == (
        "303234aaef58f48e04c959c36898ff94900d8efd7f51db2440704e7300e56289"
    )
    assert sgs["output_sha256"]["tke.tke3d_after_real_rk1"] == (
        "3c937066666bae46cc1961230b5b0f727860469361ce61c00014ed076165e7b6"
    )
    nm_def = json.loads(FIGURES[1][0].read_bytes())
    assert nm_def["output_sha256"] == {
        "defh": "930f8c270bdc59e8ba2d6d37301bbfaab79de9d81f3cbe490fc74b14c22f0184",
        "defv": "35c0ce245ef7b647331cf6796b492d81bd1d6da6d63bfe19f21c3e1bce5c4167",
        "nm": "1331c1feb09033a98ee5ce14cada19c3870dd9a8ded083ecb4c5d0ec3fe20802",
    }
    scalar = json.loads(FIGURES[2][0].read_bytes())
    assert scalar["output"]["theta"]["sha256_f32le"] == (
        "6fadd321526daa17a69756300f6882a14b3335e148139e5d36675252a051d28e"
    )
    assert scalar["output"]["moisture"]["sha256_f32le"] == (
        "c2398e40191b102fe21c69ce4e3ed0c7f778457037a1aa4c73ff88244e4a4489"
    )
