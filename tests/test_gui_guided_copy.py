"""The copy lint over every guided string.

Every string in gui/copy/*.json and the guided pages passes viz 2.0's text lint, has no marketing or tornado
wording, and shows CM1's own names only inside the hint element.  The presets' card text and every sentence the
fit writes (the budget screen shows them) are held to the same rules.  CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from cm1core.cm1run import presets as P
from cm1core.cm1run import rates
from cm1core.cm1run.fit import BUDGETS
from cm1core.cm1run.gpuinfo import GpuInfo, HostInfo
from cm1core.gui import copy_lint as L

from cm1import_support import namelist_tool

ROOT = Path(__file__).resolve().parents[1]
GUIDED = ROOT / "src/cm1core/gui/static/guided"
COPY = ROOT / "src/cm1core/gui/copy"
MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _parser(tmp_path, monkeypatch):
    namelist_tool()
    monkeypatch.setenv(rates.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(P.PRESETS_ENV, raising=False)


def test_every_guided_file_passes_the_copy_lint():
    files = L.guided_files()
    assert {path.name for path in files} >= {"guided.json", "looks.json", "gallery.js", "budget.js", "watch.js",
                                              "results.js", "index.js", "copy.js"}
    assert L.lint_all(files) == []


def test_the_lint_finds_what_it_is_for(tmp_path):
    assert L.lint_text("Set dx to 250 m", where="t")                        # a CM1 name outside a hint
    assert not L.lint_text("dx", where="t", allow_cm1=True)                 # the hint itself
    assert not L.lint_text("Grid spacing {dx}", where="t")                  # a field filled with a number
    assert L.lint_text("A stunning storm", where="t")
    assert L.lint_text("A t" + "ornado forms", where="t")
    assert L.lint_text("Gusts reach 40 m/s near the ground", where="t")     # no frame named
    assert not L.lint_text("Gusts reach 40 m/s ground-relative", where="t")
    assert not L.lint_text("About 1 hour on your GPU", where="t")          # hour is a word as well as a key
    # the picture lint's own marketing words (quicklook/style.py) are banned here too
    for phrase in ("A breakthrough in speed", "State of the art physics", "Cutting edge numerics",
                   "Next-generation storms", "Unprecedented detail"):
        assert L.lint_text(phrase, where="t"), phrase
    bad = {"schema": "x", "strings": {"a": {"text": "Set timax", "hint": "timax"}}}
    path = tmp_path / ".copy-lint-probe.json"
    path.write_text(json.dumps(bad))
    problems = L.lint_copy(path)
    assert len(problems) == 1 and "timax" in problems[0] and "strings.a.text" in problems[0]


def test_every_copy_entry_keeps_cm1_names_in_its_hint():
    guided = json.loads((COPY / "guided.json").read_text())
    hinted = [row for row in list(guided["strings"].values()) + list(guided["terms"].values()) if "hint" in row]
    assert hinted, "the terms carry CM1's names as hints"
    for row in hinted:
        assert not L.lint_text(row["text"], where="text")
    terms = guided["terms"]
    assert terms["grid_spacing"] == {"text": "Grid spacing", "hint": "dx, dy"}
    assert terms["storm_time"]["hint"] == "timax" and terms["scheme"]["hint"] == "ptype"


def test_the_pages_draw_hints_only_from_the_copy():
    for path in GUIDED.glob("*.js"):
        source = path.read_text()
        # the one place a hint element is made, from a copy entry's hint field
        if 'class: "hint"' in source:
            assert path.name == "copy.js"


def test_the_preset_cards_pass_the_lint():
    for preset in P.load_all():
        for field in ("name", "line", "note", "gm1_settings_why"):
            text = preset.record.get(field)
            if text:
                assert L.lint_text(text, where=f"{preset.id}.{field}") == []
        why = P.examine(preset)["why_not"]
        if why:
            assert L.lint_text(why, where=f"{preset.id}.why_not") == []


def card(name: str, total_mib: int, used_bytes: float, *, wsl2: bool = False) -> GpuInfo:
    used = int(round(used_bytes / MIB)) * MIB
    return GpuInfo(0, "GPU-fake", name, total_mib * MIB, used, total_mib * MIB - used, 0, "8.6", "610", None, (), wsl2)


def test_every_sentence_the_fit_writes_passes_the_lint(tmp_path, monkeypatch):
    import shutil

    # A preset 0.1.0 withholds (its preset.json) says only why; its fit's words are linted with it offered again.
    withheld = {preset.id: P.examine(preset)["why_not"] for preset in P.load_all() if preset.record.get("offered") is False}
    for preset_id, why in withheld.items():
        assert L.lint_text(why, where=preset_id) == [], why
    offered = tmp_path / "offered-presets"
    shutil.copytree(P.presets_root(), offered, ignore=shutil.ignore_patterns("__pycache__"))
    for preset_id in withheld:
        path = offered / preset_id / "preset.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("offered", None)
        record.pop("not_offered_reason", None)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    monkeypatch.setenv(P.PRESETS_ENV, str(offered))
    host = HostInfo(64 * 10**9, 60 * 10**9, "/runs", 2000 * 10**9, 1000 * 10**9, 16)
    cards = [card("NVIDIA GeForce RTX 4060", 8188, 4.3e9), card("NVIDIA GeForce RTX 3080", 10240, 4.8e9, wsl2=True),
             card("NVIDIA GeForce RTX 4080", 16376, 0.004e9), card("NVIDIA GeForce RTX 5090", 32607, 0.002e9)]
    # and a small computer: little RAM, little free disk (the not-offered reasons of 7.4 step 6)
    small = [HostInfo(4 * 10**9, 3 * 10**9, "/runs", 100 * 10**9, 8 * 10**9, 4),
             HostInfo(64 * 10**9, 60 * 10**9, "/runs", 100 * 10**9, int(0.05 * 10**9), 16)]
    seen = set()
    for preset in P.load_all():
        for budget in BUDGETS:
            for gpu, machine in [(gpu, host) for gpu in cards] + [(cards[2], other) for other in small]:
                result = P.fit(preset, budget, gpu=gpu, host=machine)
                for text in [result.summary, result.reason, *result.sentences]:
                    if text and text not in seen:
                        seen.add(text)
                        assert L.lint_text(text, where=f"{preset.id}/{budget}") == [], text
    assert any(re.search(r"instead of", text) for text in seen)
    assert any("of RAM even at its smallest size" in text for text in seen)
    assert any("of free disk even at its smallest size" in text for text in seen)


def test_the_guided_pages_import_only_files_that_exist_and_ship():
    import fnmatch
    import tomllib

    for script in GUIDED.glob("*.js"):
        for ref in re.findall(r'from "(\.{1,2}/[^"]+)"', script.read_text()):
            assert (script.parent / ref).resolve().is_file(), (script.name, ref)
    globs = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["cm1core"]
    package = ROOT / "src/cm1core"
    wanted = [*COPY.glob("*.json"), *(path for path in (package / "presets").rglob("*") if path.is_file())]
    for path in wanted:
        name = path.relative_to(package).as_posix()
        assert any(fnmatch.fnmatch(name, glob) for glob in globs), name
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include src/cm1core/presets" in manifest and "recursive-include src/cm1core/gui/copy" in manifest
