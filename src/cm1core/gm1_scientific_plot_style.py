"""Deterministic light styling for GM1 scientific validation figures.

The module is deliberately presentation-only: it never normalizes, clips, or
otherwise transforms scientific arrays.  Plot runners remain responsible for
choosing their displayed diagnostic and color limits.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping


GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA = "gm1.scientific-plot-style/v1"
GM1_SCIENTIFIC_PLOT_FONT_FAMILY = "DejaVu Sans"
GM1_SCIENTIFIC_PLOT_DPI = 240
GM1_SCIENTIFIC_PLOT_CANVAS = "#f7f5ef"
GM1_SCIENTIFIC_PLOT_AXES = "#fbfaf6"
GM1_SCIENTIFIC_PLOT_TEXT = "#262626"
GM1_SCIENTIFIC_PLOT_MUTED = "#66645f"
GM1_SCIENTIFIC_PLOT_SPINE = "#77736b"
GM1_SCIENTIFIC_PLOT_GRID = "#d9d5cb"

_FORBIDDEN_FIGURE_COPY = (
    "tornado",
    "pass_",
    "status:",
    "operator gate",
    "forecast skill",
)

GM1_SCIENTIFIC_PLOT_RCPARAMS: Mapping[str, object] = MappingProxyType(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [GM1_SCIENTIFIC_PLOT_FONT_FAMILY],
        "font.weight": 400,
        "font.size": 10.5,
        "text.color": GM1_SCIENTIFIC_PLOT_TEXT,
        "axes.facecolor": GM1_SCIENTIFIC_PLOT_AXES,
        "axes.edgecolor": GM1_SCIENTIFIC_PLOT_SPINE,
        "axes.labelcolor": GM1_SCIENTIFIC_PLOT_TEXT,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": 500,
        "axes.linewidth": 0.65,
        "axes.grid": False,
        "axes.axisbelow": True,
        "figure.facecolor": GM1_SCIENTIFIC_PLOT_CANVAS,
        "figure.edgecolor": GM1_SCIENTIFIC_PLOT_CANVAS,
        "figure.titlesize": 15.0,
        "figure.titleweight": 500,
        "savefig.facecolor": GM1_SCIENTIFIC_PLOT_CANVAS,
        "savefig.edgecolor": GM1_SCIENTIFIC_PLOT_CANVAS,
        "savefig.dpi": GM1_SCIENTIFIC_PLOT_DPI,
        "savefig.transparent": False,
        "xtick.color": GM1_SCIENTIFIC_PLOT_MUTED,
        "ytick.color": GM1_SCIENTIFIC_PLOT_MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "image.aspect": "equal",
        "image.interpolation": "bicubic",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_gm1_scientific_figure_copy(*values: str) -> None:
    """Reject status-like or overclaiming wording from rendered figure text."""

    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("scientific figure copy must be non-empty text")
        folded = value.casefold()
        for forbidden in _FORBIDDEN_FIGURE_COPY:
            if forbidden in folded:
                raise ValueError(
                    f"scientific figure copy contains forbidden wording: {forbidden}"
                )


def replace_gm1_presentation_only(
    sealed: Mapping[str, object],
    replay: Mapping[str, object],
    *,
    presentation_key: str,
    protected_paths: tuple[tuple[str, ...], ...],
) -> dict[str, object]:
    """Replace one presentation block after exact protected-field checks."""

    if not presentation_key or presentation_key not in sealed:
        raise KeyError(f"sealed receipt lacks presentation key {presentation_key!r}")
    if presentation_key not in replay:
        raise KeyError(f"replay receipt lacks presentation key {presentation_key!r}")

    def lookup(value: Mapping[str, object], path: tuple[str, ...]) -> object:
        current: object = value
        for name in path:
            if not isinstance(current, Mapping) or name not in current:
                raise KeyError(".".join(path))
            current = current[name]
        return current

    def plain(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    for path in protected_paths:
        if not path:
            raise ValueError("protected receipt paths must not be empty")
        if plain(lookup(sealed, path)) != plain(lookup(replay, path)):
            raise ValueError(
                "presentation replay changed protected receipt field "
                + ".".join(path)
            )
    merged = copy.deepcopy(dict(sealed))
    merged[presentation_key] = copy.deepcopy(replay[presentation_key])
    return merged


def gm1_scientific_plot_style_identity() -> dict[str, object]:
    """Resolve and pin Matplotlib's bundled DejaVu Sans face."""

    import matplotlib
    from matplotlib import font_manager

    font = Path(
        font_manager.findfont(
            font_manager.FontProperties(family=GM1_SCIENTIFIC_PLOT_FONT_FAMILY),
            fallback_to_default=False,
        )
    ).resolve(strict=True)
    if font.name != "DejaVuSans.ttf":
        raise RuntimeError(f"unexpected GM1 scientific font face: {font}")
    return {
        "schema": GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA,
        "theme": "classy-light",
        "canvas_hex": GM1_SCIENTIFIC_PLOT_CANVAS,
        "axes_hex": GM1_SCIENTIFIC_PLOT_AXES,
        "text_hex": GM1_SCIENTIFIC_PLOT_TEXT,
        "font_family": GM1_SCIENTIFIC_PLOT_FONT_FAMILY,
        "font_file": font.name,
        "font_sha256": _sha256(font),
        "matplotlib_version": matplotlib.__version__,
        "dpi": GM1_SCIENTIFIC_PLOT_DPI,
    }


@contextmanager
def gm1_scientific_plot_context() -> Iterator[dict[str, object]]:
    """Apply the shared style without leaking process-global rcParams."""

    import matplotlib

    identity = gm1_scientific_plot_style_identity()
    with matplotlib.rc_context(rc=dict(GM1_SCIENTIFIC_PLOT_RCPARAMS)):
        yield identity


def style_gm1_image_axis(axis: Any) -> None:
    """Apply restrained image-axis framing after the scientific artist exists."""

    axis.tick_params(which="both", top=False, right=False)
    for spine in axis.spines.values():
        spine.set_color(GM1_SCIENTIFIC_PLOT_SPINE)
        spine.set_linewidth(0.65)


def style_gm1_colorbar(colorbar: Any) -> None:
    """Keep colorbar framing and ticks subordinate to the diagnostic field."""

    colorbar.outline.set_edgecolor(GM1_SCIENTIFIC_PLOT_SPINE)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.tick_params(length=2.5, width=0.6, colors=GM1_SCIENTIFIC_PLOT_MUTED)
    colorbar.ax.yaxis.label.set_color(GM1_SCIENTIFIC_PLOT_TEXT)


def save_gm1_scientific_png(
    figure: Any,
    path: str | Path,
    *,
    title: str,
    description: str,
) -> dict[str, object]:
    """Save one opaque deterministic 240-DPI PNG and return style identity."""

    validate_gm1_scientific_figure_copy(title, description)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = gm1_scientific_plot_style_identity()
    figure.savefig(
        output,
        dpi=GM1_SCIENTIFIC_PLOT_DPI,
        facecolor=GM1_SCIENTIFIC_PLOT_CANVAS,
        edgecolor=GM1_SCIENTIFIC_PLOT_CANVAS,
        metadata={
            "Title": title,
            "Description": description,
            "Software": GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA,
            "GM1PlotStyle": GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA,
        },
    )
    return identity


__all__ = [
    "GM1_SCIENTIFIC_PLOT_AXES",
    "GM1_SCIENTIFIC_PLOT_CANVAS",
    "GM1_SCIENTIFIC_PLOT_DPI",
    "GM1_SCIENTIFIC_PLOT_FONT_FAMILY",
    "GM1_SCIENTIFIC_PLOT_RCPARAMS",
    "GM1_SCIENTIFIC_PLOT_STYLE_SCHEMA",
    "gm1_scientific_plot_context",
    "gm1_scientific_plot_style_identity",
    "save_gm1_scientific_png",
    "replace_gm1_presentation_only",
    "style_gm1_colorbar",
    "style_gm1_image_axis",
    "validate_gm1_scientific_figure_copy",
]
