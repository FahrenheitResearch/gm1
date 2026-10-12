"""The pages that lead beyond the GUI point only at things the public repository carries, and agree on the result.

1. Every relative link, and every link to this repository's own files on GitHub, in the README, the decks page, the
   nesting page, the two tornado criterion pages, the 4 m example's page, the runtime reference and the release
   notes names a file or folder that exists.
2. Every path under tools/, docs/ or examples/ that the first six of those pages write in code marks exists.
3. The pages that state the 4 m run's peak wind state it in the same words, and the pages that tell its story link
   each other.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DECKS = ROOT / "docs" / "GM1-DECKS.md"
NESTING = ROOT / "docs" / "NESTING.md"
CRITERION = ROOT / "docs" / "GM1-TORNADO-CRITERION-V2.md"
VERDICTS = ROOT / "docs" / "GM1-TORNADO-VERDICTS-V2.md"
EXAMPLE = ROOT / "examples" / "tornado-4m" / "README.md"
NOTES = ROOT / "docs" / "RELEASE-NOTES.md"
RUNTIME = ROOT / "docs" / "GM1-CARTESIAN-RUNTIME.md"
PAGES = (README, DECKS, NESTING, CRITERION, VERDICTS, EXAMPLE)
OWN = "https://github.com/FahrenheitResearch/gm1/"
PEAK = "about 128 m/s (129.0 m/s in the last record before the time step limit was passed)"


def _text(page: Path) -> str:
    return page.read_text(encoding="utf-8")


def _flat(page: Path) -> str:
    return " ".join(_text(page).split())


@pytest.mark.parametrize("page", PAGES + (RUNTIME, NOTES), ids=lambda p: p.name)
def test_every_link_to_the_repository_resolves(page: Path) -> None:
    missing = []
    for target in re.findall(r"\]\(([^)\s]+)\)", _text(page)):
        target = target.split("#")[0]
        if target.startswith(OWN):
            kind, _, rest = target[len(OWN):].partition("/")
            if kind not in ("blob", "tree"):
                continue
            where = ROOT / rest.partition("/")[2]
        elif not target or re.match(r"[a-z]+:", target):
            continue
        else:
            where = page.parent / target
        if not where.exists():
            missing.append(target)
    assert not missing, missing


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_named_tool_page_and_example_exists(page: Path) -> None:
    missing = []
    for path in re.findall(r"`((?:tools|docs|examples)/[A-Za-z0-9_./-]+)", _text(page)):
        path = path.rstrip(".")
        if not (ROOT / path).exists() and not (page.parent / path).exists():
            missing.append(path)
    assert not missing, missing


def test_the_peak_wind_is_stated_in_the_same_words_everywhere() -> None:
    for page in (README, EXAMPLE, VERDICTS, NOTES):
        assert PEAK in _flat(page), page.name
    # The short form on the nesting page is the same round figure.
    assert "about 128 m/s" in _flat(NESTING)
    # Only the verdicts page, which explains it, and the example's page, in a sentence that says when, give the
    # largest wind of the span that must not be cited.
    for page in (README, DECKS, NESTING, NOTES):
        assert "163" not in _flat(page), page.name
    for sentence in re.split(r"(?<=[.:])\s", _flat(EXAMPLE)):
        if "163 m/s" in sentence:
            assert "after 3312 s" in sentence, sentence


def test_the_pages_that_tell_the_story_link_each_other() -> None:
    assert "GM1-TORNADO-VERDICTS-V2.md" in _text(README) and "docs/NESTING.md" in _text(README)
    assert "examples/tornado-4m" in _text(README)
    assert "](GM1-TORNADO-VERDICTS-V2.md)" in _text(NESTING) and "](GM1-TORNADO-CRITERION-V2.md)" in _text(NESTING)
    assert "](../examples/tornado-4m/README.md)" in _text(NESTING)
    assert "](../../docs/GM1-TORNADO-VERDICTS-V2.md)" in _text(EXAMPLE) and "](../../docs/NESTING.md)" in _text(EXAMPLE)
    assert "](NESTING.md)" in _text(VERDICTS) and "examples/tornado-4m/S3.toml" in _text(VERDICTS)
    assert "](NESTING.md)" in _text(DECKS) and "](NESTING.md)" in _text(RUNTIME)
    # The headline's limit is said wherever the headline is.
    for page in (README, EXAMPLE, NOTES, VERDICTS):
        flat = _flat(page)
        assert "cannot yet be re-derived" in flat or "cannot yet re-derive" in flat or "cannot check yet" in flat, page.name


@pytest.mark.parametrize("page", PAGES + (RUNTIME, NOTES), ids=lambda p: p.name)
def test_no_page_is_left_with_a_line_waiting_to_be_filled(page: Path) -> None:
    text = _text(page)
    for marker in ("TO FILL", "TODO", "TBD", "42 table.md"):
        assert marker not in text, (page.name, marker)


def test_the_guides_say_where_pictures_land_and_which_python_runs_the_tools() -> None:
    decks, nesting = _flat(DECKS), _flat(NESTING)
    # Pictures and CM1 files go beside the frames folder, not into it.
    assert "into `run1/quicklook`" in decks and "into `run1/cm1out`" in decks
    assert "run1/frames/quicklook" not in decks and "run1/frames/cm1out" not in decks
    assert "`~/gm1-runs/nest/S1/quicklook/`" in nesting and "S1-frames" not in nesting
    for page in (NESTING, EXAMPLE):
        assert "$(uv tool dir)/gpuwm-cm1/bin/python" in _text(page), page.name
    # Every stage of the recipe writes into a folder of its own.
    for stage in ("S1", "P", "S3"):
        assert f"--history-dir ~/gm1-runs/nest/{stage}/frames" in _text(NESTING), stage

