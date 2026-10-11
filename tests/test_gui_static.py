"""The live GUI's pages: every file they reference exists, the wheel carries them (and not the mock), and
their words pass the mock's word rules.  CPU only."""

from __future__ import annotations

import fnmatch
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/cm1core/gui/static"
BANNED = ("seamless", "powerful", "blazing", "stunning", "effortless", "magic", "supercharg", "revolutionary",
          "cutting-edge", "next-gen", "unleash", "lightning", "amazing", "incredible", "beautiful", "world-class",
          "game-chang", "state-of-the-art", "tornado", "/home")


def _live_files() -> list[Path]:
    return sorted(p for p in STATIC.rglob("*") if p.is_file() and "mock" not in p.relative_to(STATIC).parts)


def test_every_page_reference_resolves():
    index = (STATIC / "index.html").read_text()
    refs = re.findall(r'(?:href|src)="/static/([^"]+)"', index)
    assert refs and all((STATIC / ref).is_file() for ref in refs), refs
    scripts = sorted(p for p in (STATIC / "js").rglob("*.js"))
    assert any(p.parent.name == "setup" for p in scripts)
    for script in scripts:
        # `import ... from "./x.js"`, `import "./x.js"` and `../x.js`, relative to the module itself
        for ref in re.findall(r'(?:from|import)\s*"(\.{1,2}/[^"]+)"', script.read_text()):
            assert (script.parent / ref).resolve().is_file(), (script.name, ref)
    for image in re.findall(r'"/static/(img/[^"]+)"', (ROOT / "src/cm1core/gui/setup.py").read_text()):
        assert (STATIC / image).is_file(), image


def test_package_data_carries_the_pages_and_leaves_the_mock_out():
    globs = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["cm1core"]
    package = ROOT / "src/cm1core"
    shipped = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
               and any(fnmatch.fnmatch(p.relative_to(package).as_posix(), g) for g in globs)}
    live = {p.relative_to(package).as_posix() for p in _live_files()}
    assert live <= shipped
    assert not any("/mock/" in name for name in shipped)
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "prune src/cm1core/gui/static/mock" in manifest


def test_the_live_pages_use_plain_words():
    for path in _live_files():
        if path.suffix not in (".html", ".js", ".css") or path.name == "fonts.css":
            continue
        low = path.read_text(encoding="utf-8").lower()
        assert not [word for word in BANNED if word in low], path


def test_the_server_messages_use_plain_words():
    for path in sorted((ROOT / "src/cm1core/gui").glob("*.py")):
        if path.name == "copy_lint.py":
            continue  # the word rules themselves: they spell what they forbid
        low = path.read_text(encoding="utf-8").lower()
        assert not [word for word in BANNED if word in low], path.name
