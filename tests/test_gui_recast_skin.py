"""The GUI's chrome in the Recast design language, with every plot left as it was drawn.  CPU only.

The Recast tokens (RecastRadar console styles.css, the Recast desktop shell) are mapped onto GM1's ``--gm1-*``
variables. These tests pin that mapping, check WCAG AA contrast for every text colour the stylesheets
use on the three grounds a text can sit on (the page, a panel, a panel's lit top), hold the 10.5 px type floor, keep
the pages free of any outside URL, and keep the plots (quick-look pictures, the SVG charts, the plan map and the
hodograph) exactly as they were: no filter on a picture, the plots' own colours and faces unchanged.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/cm1core/gui/static"
SHEETS = {name: STATIC / name for name in ("css/gm1.css", "css/app.css", "guided/guided.css")}

#: The token table: Recast token (value) -> GM1 variable.
TOKENS = {
    "--gm1-void": "#0a0a0a",                       # --bg
    "--gm1-ground": "#0a0b0d",                     # --surface-1 on #0a0a0a
    "--gm1-panel": "#0e0f11",                      # --surface-2 on #0a0a0a
    "--gm1-control": "#26282c",                    # --surface-3 (--surface-solid-3)
    "--gm1-hover": "rgba(255, 255, 255, 0.055)",   # --surface-hover
    "--gm1-active": "rgba(255, 255, 255, 0.10)",   # --surface-active
    "--gm1-rule": "rgba(255, 255, 255, 0.10)",     # --line-1
    "--gm1-rule-soft": "rgba(255, 255, 255, 0.06)",  # --line-2
    "--gm1-faint": "rgba(255, 255, 255, 0.16)",    # --line-3
    "--gm1-title": "#f5f5f5",                      # --ink-1
    "--gm1-text": "#f5f5f5",                       # --ink-1
    "--gm1-data": "#d4d4d4",                       # --ink-2
    "--gm1-muted": "#a3a3a3",                      # --ink-3
    "--gm1-dim": "#858585",                        # --ink-4, lifted from #737373 for AA
    "--gm1-disabled": "#525252",                   # --ink-5
    "--gm1-accent": "#22d3ee",
    "--gm1-accent-dim": "rgba(34, 211, 238, 0.16)",
    "--gm1-accent-line": "rgba(34, 211, 238, 0.45)",
    "--gm1-new": "#a78bfa",                        # --accent-new
    "--gm1-new-dim": "rgba(167, 139, 250, 0.18)",
    "--gm1-live": "#22c55e",
    "--gm1-cold": "#60a5fa",                       # --state-archive
    "--gm1-warn": "#f59e0b",                       # --state-stale
    "--gm1-stop": "#f87171",                       # --state-error
    "--gm1-sp-1": "2px", "--gm1-sp-2": "4px", "--gm1-sp-3": "6px", "--gm1-sp-4": "8px",
    "--gm1-sp-5": "12px", "--gm1-sp-6": "16px", "--gm1-sp-7": "20px", "--gm1-sp-8": "24px",
    "--gm1-r-sm": "3px", "--gm1-r-md": "5px", "--gm1-r-lg": "8px",
    "--gm1-dur": "90ms",
    "--gm1-tracking": ".085em",
    # plot colours: unchanged since viz 2.0
    "--gm1-gold": "#ffd166", "--gm1-series-a": "#d9731f", "--gm1-series-b": "#4f86e6",
    # type sizes: GM1's readable sizes, not the console's 9-10 px
    "--fs": "13px", "--fs-small": "11.5px", "--fs-micro": "10.5px",
}

#: What the plots draw with, their own values re-declared inside every plot, so the chrome's tokens do not reach them.
PLOT_TOKENS = {"--gm1-rule": "#1d2436", "--gm1-faint": "#465066", "--gm1-muted": "#8e97a8", "--gm1-data": "#c7cfdb",
               "--gm1-text": "#e6edf7", "--gm1-panel": "#0a0e18", "--gm1-ground": "#05070d",
               # the face their unstyled labels inherit (the hodograph's heights and marks): the plots' own page face
               "font-family": "'GM1 DejaVu Sans', 'DejaVu Sans', 'Bitstream Vera Sans', Verdana, sans-serif"}

#: The plot rules, verbatim: the SVG charts, their legend, the plan map.
PLOT_RULES = (
    ".chart svg text { font-family: var(--gm1-font-condensed); font-size: 10px; fill: var(--gm1-muted); }",
    ".chart svg .grid { stroke: var(--gm1-rule); stroke-width: 1; }",
    ".chart svg .axis { stroke: var(--gm1-faint); stroke-width: 1; }",
    ".chart svg .a { stroke: var(--gm1-series-a); stroke-width: 2; fill: none; }",
    ".chart svg .b { stroke: var(--gm1-series-b); stroke-width: 2; fill: none; stroke-dasharray: 5 3; }",
    ".chart svg .dot-a { fill: var(--gm1-series-a); stroke: var(--gm1-panel); stroke-width: 2; }",
    ".chart svg .dot-b { fill: var(--gm1-series-b); stroke: var(--gm1-panel); stroke-width: 2; }",
    ".chart svg .now { stroke: var(--gm1-gold); stroke-width: 1; }",
    ".chart svg .cursor { stroke: var(--gm1-data); stroke-width: 1; stroke-dasharray: 2 2; }",
    ".chart svg text.end { fill: var(--gm1-data); }",
    ".legend .sw.a { border-color: var(--gm1-series-a); }",
    ".legend .sw.b { border-color: var(--gm1-series-b); border-top-style: dashed; }",
    ".planmap svg text { font-family: var(--gm1-font-condensed); font-size: 11px; fill: var(--gm1-muted); }",
    ".hodo .legend .sw { border-top-width: 3px; }",
)
PLOT_SELECTOR = re.compile(r"\bsvg\b|\.chart\b|\.planmap\b|\.hodo\b|\.legend\b|\.ratio-bar\b")


# ---------------------------------------------------------------- a small CSS reader

def _strip(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def rules(css: str, media: str = "") -> list[tuple[str, str, str]]:
    """(media, selector, declarations) for every style rule, descending into @media and @supports."""

    out = []
    css = _strip(css)
    i = 0
    while i < len(css):
        brace = css.find("{", i)
        if brace < 0:
            break
        selector = css[i:brace].strip()
        depth, j = 1, brace + 1
        while depth and j < len(css):
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        body = css[brace + 1:j - 1]
        if selector.startswith(("@media", "@supports")):
            out += rules(body, selector)
        elif not selector.startswith("@"):
            out.append((media, " ".join(selector.split()), body))
        i = j
    return out


def declarations(body: str) -> list[tuple[str, str]]:
    out = []
    for part in re.split(r";(?![^(]*\))", body):
        if ":" in part:
            key, value = part.split(":", 1)
            out.append((key.strip(), " ".join(value.split())))
    return out


def root_tokens() -> dict[str, str]:
    for media, selector, body in rules(SHEETS["css/gm1.css"].read_text()):
        if selector == ":root" and not media:
            return {k: v for k, v in declarations(body) if k.startswith("--")}
    raise AssertionError("gm1.css has no :root")


def resolve(value: str, tokens: dict[str, str]) -> str:
    for _ in range(8):
        match = re.search(r"var\((--[\w-]+)(?:,\s*([^)]*))?\)", value)
        if not match:
            return value
        value = value[:match.start()] + tokens.get(match.group(1), match.group(2) or "") + value[match.end():]
    return value


# ---------------------------------------------------------------- colour and contrast (WCAG 2.2, SC 1.4.3)

def rgba(value: str) -> tuple[float, float, float, float]:
    value = value.strip()
    if value.startswith("#"):
        h = value[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0
    match = re.fullmatch(r"rgba?\(([^)]*)\)", value)
    assert match, value
    parts = [float(p) for p in match.group(1).replace("/", ",").split(",")]
    return parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else 1.0


def over(top: str | tuple, ground: tuple) -> tuple:
    r, g, b, a = rgba(top) if isinstance(top, str) else top
    return tuple(a * c + (1 - a) * d for c, d in zip((r, g, b), ground[:3])) + (1.0,)


def luminance(colour: tuple) -> float:
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = colour[:3]
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: tuple, b: tuple) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def grounds(tokens: dict[str, str]) -> dict[str, tuple]:
    void = rgba(resolve("var(--gm1-void)", tokens))
    panel = over(resolve("var(--gm1-panel)", tokens), void)
    # the sheen's top stop, rgba(255,255,255,.055), over the panel: the lightest ground any chrome text sits on
    sheen = re.search(r"rgba\([^)]*\)", tokens["--gm1-sheen"]).group(0)
    return {"page": void, "panel": panel, "sheen top": over(sheen, panel),
            "rail": over("rgba(255, 255, 255, 0.045)", rgba(resolve("var(--gm1-ground)", tokens)))}


# ---------------------------------------------------------------- the tests

def test_every_recast_token_resolves_to_its_mapped_value():
    tokens = root_tokens()
    wrong = {k: (tokens.get(k), v) for k, v in TOKENS.items() if tokens.get(k, "").replace(" ", "") != v.replace(" ", "")}
    assert not wrong, wrong
    assert tokens["--gm1-sheen"].startswith("linear-gradient(180deg, rgba(255, 255, 255, 0.055)")
    assert "ui-sans-serif" in tokens["--gm1-font"] and tokens["--gm1-font"].rstrip().endswith("'GM1 DejaVu Sans', sans-serif")
    assert tokens["--gm1-font-mono"].startswith("ui-monospace") and "'GM1 DejaVu Sans Mono'" in tokens["--gm1-font-mono"]
    # every --gm1-* a stylesheet uses exists on the base, so other packages' sheets can rely on the same names
    used = set()
    for path in SHEETS.values():
        used |= set(re.findall(r"var\((--[\w-]+)", _strip(path.read_text())))
    assert not used - set(tokens), used - set(tokens)


def test_every_text_colour_meets_aa_on_every_ground():
    tokens = root_tokens()
    ground = grounds(tokens)
    failures = []
    for name, path in SHEETS.items():
        for media, selector, body in rules(path.read_text()):
            if PLOT_SELECTOR.search(selector) or selector == ":root":
                continue
            for key, value in declarations(body):
                if key != "color" or value in ("inherit", "currentColor", "transparent"):
                    continue
                colour = resolve(value, tokens)
                if colour == resolve("var(--gm1-disabled)", tokens):
                    # WCAG exempts inactive controls; the disabled ink is only for them
                    assert re.search(r":disabled|\.disabled|\[disabled\]", selector), (name, selector)
                    continue
                for where, g in ground.items():
                    ratio = contrast(over(colour, g), g)
                    if ratio < 4.5:
                        failures.append(f"{name} {selector}: {value} on {where} = {ratio:.2f}")
    assert not failures, failures


@pytest.mark.parametrize("text, fill, least", [
    # buttons, the mode switch, chips and notices: their text on their own fill, over the sheen's top
    ("var(--gm1-data)", "var(--gm1-control)", 4.5),
    ("var(--gm1-title)", "#313338", 4.5),
    ("var(--gm1-muted)", "var(--gm1-control)", 4.5),
    ("var(--gm1-accent)", "var(--gm1-accent-dim)", 4.5),
    ("var(--gm1-accent)", "rgba(34, 211, 238, 0.24)", 4.5),
    ("var(--gm1-stop)", "var(--gm1-stop-dim)", 4.5),
    ("var(--gm1-stop)", "rgba(248, 113, 113, 0.18)", 4.5),
    ("var(--gm1-live)", "var(--gm1-live-dim)", 4.5),
    ("var(--gm1-cold)", "var(--gm1-cold-dim)", 4.5),
    ("var(--gm1-warn)", "var(--gm1-warn-dim)", 4.5),
    ("var(--gm1-data)", "rgba(255, 255, 255, 0.045)", 4.5),
    ("var(--gm1-muted)", "rgba(255, 255, 255, 0.045)", 4.5),
    ("var(--gm1-text)", "var(--gm1-accent-dim)", 4.5),
    ("var(--gm1-text)", "var(--gm1-warn-dim)", 4.5),
    ("var(--gm1-text)", "var(--gm1-stop-dim)", 4.5),
    ("var(--gm1-title)", "var(--gm1-hover)", 4.5),
    ("var(--gm1-dim)", "rgba(0, 0, 0, 0.25)", 4.5),
    ("var(--gm1-new)", "var(--gm1-panel)", 3.0),   # the "new" dot is a graphic (SC 1.4.11)
    ("var(--gm1-accent)", "var(--gm1-panel)", 3.0),  # the focus ring (SC 1.4.11)
])
def test_text_on_its_own_fill_meets_aa(text, fill, least):
    tokens = root_tokens()
    for where, g in grounds(tokens).items():
        backdrop = over(resolve(fill, tokens), g)
        ratio = contrast(over(resolve(text, tokens), backdrop), backdrop)
        assert ratio >= least, (text, fill, where, round(ratio, 2))


def test_recasts_own_ink_4_would_fail_and_the_lifted_one_passes():
    tokens = root_tokens()
    top = grounds(tokens)["sheen top"]
    assert contrast(rgba("#737373"), top) < 4.5 <= contrast(rgba(tokens["--gm1-dim"]), top)


def test_no_chrome_type_below_the_readable_floor():
    tokens = root_tokens()
    small = []
    for name, path in SHEETS.items():
        for media, selector, body in rules(path.read_text()):
            if PLOT_SELECTOR.search(selector):
                continue
            for key, value in declarations(body):
                if key == "font-size":
                    size = resolve(value, tokens)
                    if size.endswith("px") and float(size[:-2]) < 10.5:
                        small.append((name, selector, size))
    assert not small, small


def test_the_pages_fetch_nothing_from_outside():
    for path in [*SHEETS.values(), STATIC / "index.html"]:
        text = _strip(path.read_text())
        urls = re.findall(r"url\(([^)]*)\)", text)
        assert all(u.strip("'\" ").startswith("data:") for u in urls), (path.name, urls)
        assert not re.search(r"(?:https?:)?//[\w.-]+\.[a-z]{2,}", text), path.name
        assert "@import" not in text, path.name
    fonts = (STATIC / "css/fonts.css").read_text()
    assert all(u.startswith("data:font/") for u in re.findall(r"url\(([^)]{0,40})", fonts))
    index = (STATIC / "index.html").read_text()
    assert '<meta name="color-scheme" content="dark">' in index


def test_plots_keep_their_colours_faces_and_pictures_their_pixels():
    css = {name: _strip(path.read_text()) for name, path in SHEETS.items()}
    gm1 = " ".join(css["css/gm1.css"].split())
    everything = " ".join(" ".join(css.values()).split())
    for rule in PLOT_RULES:
        assert rule in everything, rule
    # the plots re-declare their own values of every token they draw with
    scoped = [body for media, selector, body in rules(css["css/gm1.css"]) if selector == ".chart, .planmap, .hodo, .legend"]
    assert len(scoped) == 1
    assert {k: v for k, v in declarations(scoped[0])} == PLOT_TOKENS
    assert "--gm1-font-condensed: 'GM1 DejaVu Sans Condensed'" in gm1
    # a picture is shown as drawn: no filter, blend or mask anywhere; square corners on every image; object-fit only
    # on the 16:9 card thumbnails. The one exception: an unoffered card's picture is dimmed (the picture only, never
    # the card's words) and, in the guided gallery, grey ("a greyed card says why").
    unoffered = {(".g-card.off > .g-pic img", "opacity", ".62"), (".g-card.off > .g-pic img", "filter", "grayscale(1)"),
                 (".card.off figure img", "opacity", ".6")}
    for name, text in css.items():
        for media, selector, body in rules(text):
            for key, value in declarations(body):
                if (selector, key, value) in unoffered:
                    continue
                assert key not in ("filter", "mix-blend-mode", "mask", "-webkit-mask-image", "mask-image",
                                   "image-rendering", "backdrop-filter", "-webkit-backdrop-filter"), (name, selector)
                if re.search(r"\bimg\b", selector):
                    assert key not in ("border-radius", "opacity", "clip-path"), (name, selector, key)
                    if key == "object-fit":
                        assert (selector, value) in ((".card figure img", "cover"), (".g-pic img", "cover")), selector
    # the card's picture sits inside the rounded card with square corners, and the card does not clip it
    guided = " ".join(css["guided/guided.css"].split())
    assert ".g-card > .g-pic { margin: 6px 6px 0;" in guided and "border-radius: 0;" in guided
    for media, selector, body in rules(css["guided/guided.css"]) + rules(css["css/app.css"]):
        if re.search(r"\.(g-)?card\b", selector):
            assert "overflow" not in dict(declarations(body)), selector


def test_focus_is_a_two_pixel_accent_ring_and_nothing_removes_it():
    tokens = root_tokens()
    found = False
    for name, path in SHEETS.items():
        for media, selector, body in rules(path.read_text()):
            d = dict(declarations(body))
            if selector == ":focus-visible":
                found = d.get("outline") == "2px solid var(--gm1-accent)" and d.get("outline-offset") == "1px"
            if "outline" in d and selector != ":focus:not(:focus-visible)":
                # only the ring itself (or its offset for inset nav links) may set an outline
                assert selector.endswith(":focus-visible"), (name, selector)
    assert found and resolve("var(--gm1-accent)", tokens) == "#22d3ee"


def test_new_marker_is_styled_for_the_other_packages():
    gm1 = " ".join(_strip(SHEETS["css/gm1.css"].read_text()).split())
    assert ".is-new::after {" in gm1 and "background: var(--gm1-new)" in gm1


def test_the_mode_view_rules_are_unchanged():
    css = SHEETS["css/app.css"].read_text()
    assert 'body[data-mode="guided"] .advanced-only { display: none !important; }' in css
    assert 'body[data-mode="advanced"] .guided-only { display: none !important; }' in css
    assert "[hidden] { display: none !important; }" in SHEETS["guided/guided.css"].read_text()


def test_layered_backgrounds_are_images_over_one_colour():
    """In a layered ``background``, only the last layer may be a colour: a colour token (``--gm1-*-dim``) in an
    earlier layer makes the whole declaration invalid at computed-value time and the element's background is lost."""

    tokens = root_tokens()
    colour_tokens = {name for name, value in tokens.items() if re.fullmatch(r"#[0-9a-fA-F]{3,8}|rgba?\([^)]*\)", value)}
    checked = 0
    for name, path in SHEETS.items():
        for _media, selector, body in rules(path.read_text()):
            for prop, value in declarations(body):
                if prop != "background":
                    continue
                layers = _top_level_layers(value)
                for layer in layers[:-1]:
                    checked += 1
                    token = re.fullmatch(r"var\((--gm1-[a-z0-9-]+)\)", layer)
                    assert not (token and token.group(1) in colour_tokens), (name, selector, value)
                    assert not re.match(r"#|rgba?\(", layer), (name, selector, value)
    assert checked >= 5


def _top_level_layers(value: str) -> list[str]:
    """Split a ``background`` value at its top-level commas (not the ones inside a gradient or a var())."""

    layers, depth, start = [], 0, 0
    for i, ch in enumerate(value):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            layers.append(value[start:i].strip())
            start = i + 1
    layers.append(value[start:].strip())
    return layers


def test_nothing_with_words_in_it_is_dimmed_by_opacity():
    """Opacity dims a box's words with it: an unoffered card or an unused settings group dimmed by opacity puts its reason at
    3.2-3.3:1. Greyed boxes grey their words by colour (AA); only pictures and the running dot's pulse fade."""

    for name, path in SHEETS.items():
        for _media, selector, body in rules(path.read_text()):
            if re.fullmatch(r"[\d%, ]+", selector.strip()):
                continue  # keyframe steps (the pulse of a running status dot)
            for key, value in declarations(body):
                if key == "opacity" and float(value) < 1:
                    assert re.search(r"\bimg$", selector.strip()), (name, selector, value)
