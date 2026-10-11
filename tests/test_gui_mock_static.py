"""The GUI mock opens from disk with no network, and its bundle matches its ES-module sources.

CPU only: tools/gui_mock/check_mock.py greps every file under gui/static for a URL with a scheme, rebuilds
mock-bundle.js from js/ and data/ and compares it, checks every local reference, and scans the text for marketing
words and tornado wording.  The headless-browser half (screenshots, console, overflow, modules against bundle) is
tools/gui_mock/shoot_mock.py, run on a render node.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/cm1core/gui/static"


def _load(name: str):
    path = ROOT / "tools/gui_mock" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gui_mock_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ten_screens_exist() -> None:
    pages = {p.name for p in (STATIC / "mock").glob("*.html")}
    assert pages == {"index.html", "new-run.html", "check.html", "sounding.html", "domain.html", "settings.html",
                     "estimate.html", "run.html", "results.html", "system.html"}


def test_no_url_with_a_scheme_under_gui_static() -> None:
    offenders = [str(p.relative_to(STATIC)) for p in STATIC.rglob("*")
                 if p.is_file() and (b"http://" in p.read_bytes() or b"https://" in p.read_bytes())]
    assert offenders == []


def test_bundle_matches_sources_and_every_check_passes(capsys) -> None:
    assert _load("check_mock").main(["--static", str(STATIC)]) == 0, capsys.readouterr().out
