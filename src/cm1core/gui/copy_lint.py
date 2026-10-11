"""The word rules for guided mode's copy, and for any page a beginner reads.

Every string in ``gui/copy/*.json``, the guided pages and the presets' card text must:

- pass viz 2.0's text lint (``viz20_style.lint``, transcribed here for text that is not a picture): no tornado
  wording, and every wind speed in m/s says which frame it is in (ground-relative, storm-relative, ...);
- contain no marketing word, no tornado wording and no damage or victim word (:data:`BANNED`; a famous day's
  card names the station and the date, never what happened that day);
- show CM1's own names only inside the hint element: a copy entry's ``hint`` field, a ``class="hint"`` element,
  or (in Markdown) a code span.  A CM1 name is any key of CM1 r21.1's namelist that is not also an everyday
  English word (:data:`ENGLISH_WORD_KEYS`).

``python -m cm1core.gui.copy_lint [FILE ...]`` checks the named files (``.json`` copy, ``.js`` pages, ``.md``
guides), or every guided file when none is named; exit 1 lists each problem.  Standard library only, plus the
importer's key inventory.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator

GUI = Path(__file__).resolve().parent
#: Marketing words (section 6: plain factual labels) and tornado wording (section 6: never "tornado"), as
#: regular expressions over lower-case text.  Kept as patterns, so this module does not itself spell the words
#: the GUI's own word test looks for.
BANNED = (
    r"\bseamless", r"\bpowerful\b", r"\bblazing", r"\bstunning\b", r"\beffortless", r"\bmagic",
    r"\bsupercharg", r"\brevolutionary\b", r"\bcutting-edge\b", r"\bnext-gen\b", r"\bunleash",
    r"\blightning-fast\b", r"\bamazing\b", r"\bincredibl", r"\bbeautiful", r"\bworld-class\b",
    r"\bgame-chang", r"\bstate-of-the-art\b", r"\bbreathtaking\b", r"\bawesome\b", r"\bepic\b",
    r"\bultimate\b", r"\bt[o]rnad", r"\btwister", r"\bfunnel cloud", r"\bwedge\b",
    # The rest of the picture lint's marketing words (quicklook/style.py), so text that passes here passes
    # there too.
    r"\bbreakthrough", r"\bcutting edge\b", r"\bstate of the art\b", r"\bnext-generation\b", r"\bunprecedented\b",
    # Damage and victim words: the famous-day presets use a day's sounding in an idealized
    # storm, so no card, note or guide says what that day's storms did.
    r"\bdamag", r"\bdestroy", r"\bkilled\b", r"\bfatal", r"\bvictim", r"\bdeadl", r"\bdeath", r"\bdied\b",
    r"\binjur", r"\bdevastat", r"\bdisaster", r"\bcatastroph", r"\boutbreak",
)
#: CM1 namelist keys that are also everyday words, so plain text may use them.
ENGLISH_WORD_KEYS = frozenset({"day", "hour", "minute", "month", "season", "second", "viscosity", "year",
                               "infall", "testcase"})
_TORNADO_OK = re.compile(r"simulated tornado \(GM1 criterion v2[^)]*\)|tornado label|GM1 criterion v2|"
                         r"tornado criterion|not a tornado|tornadic", re.I)
_WIND_NUM = re.compile(r"\d\s*m/s")
_FRAME = re.compile(r"(ground|vortex|frame|storm)-relative", re.I)
#: A wind shear (the change of the wind with height) is the same in every frame, so a text whose speeds are all
#: shears needs no frame: "wind shear (27 m/s over 6 km)", "30 m/s of shear", "changes by 20 m/s".
_SHEAR_NUM = re.compile(r"shear[^.;]{0,24}?\d+(\.\d+)?\s*m/s|\d+(\.\d+)?\s*m/s\s+(of\s+)?(wind\s+)?(shear|change)|"
                        r"chang\w*\s+(by\s+)?\d+(\.\d+)?\s*m/s", re.I)
#: Fields of the copy files that are notes for whoever edits them, not words a page shows.
NOTE_FIELDS = frozenset({"schema", "about", "id", "clause_if", "hint"})

_TERMS: list[frozenset[str]] = []


def cm1_terms() -> frozenset[str]:
    """CM1 r21.1's namelist keys that are not everyday words."""

    if not _TERMS:
        from ..cm1import import defaults

        keys = {key for group in defaults.load().fields.values() for key in group}
        _TERMS.append(frozenset(key.lower() for key in keys if key.lower() not in ENGLISH_WORD_KEYS))
    return _TERMS[0]


def viz20_text_lint(text: str) -> list[str]:
    """viz20_style.lint's rules for one block of text (a page is not a picture: 'simulated' is not required)."""

    problems = []
    low = text.lower()
    if "tornado" in low and "tornado" in _TORNADO_OK.sub("", text).lower():
        problems.append(f"'tornado' outside canonical wording: {text!r}")
    speeds = _WIND_NUM.search(_SHEAR_NUM.sub(" ", text))
    if speeds and len(text.split()) > 2 and not _FRAME.search(text):
        problems.append(f"wind speed without ground-relative/vortex-relative: {text!r}")
    return problems


def lint_text(text: str, *, where: str = "", allow_cm1: bool = False) -> list[str]:
    """Every rule, for a string a page shows.  ``allow_cm1``: the string is a hint."""

    problems = [f"{where}: {problem}" for problem in viz20_text_lint(text)]
    low = text.lower()
    for pattern in BANNED:
        found = re.search(pattern, low)
        if found:
            problems.append(f"{where}: banned word {found.group(0)!r} in {text!r}")
    if not allow_cm1:
        # {name} fields are filled with numbers before a page shows them.
        words = re.sub(r"\{[a-z_]+\}", " ", low)
        found = sorted({word for word in re.findall(r"[a-z_][a-z0-9_]*", words) if word in cm1_terms()})
        if found:
            problems.append(f"{where}: CM1 name(s) {', '.join(found)} outside a hint in {text!r}")
    return problems


def _strings(value: Any, path: str) -> Iterator[tuple[str, str, bool]]:
    """(where, text, is_hint) for every string a copy file shows."""

    if isinstance(value, str):
        yield path, value, path.rsplit(".", 1)[-1] == "hint"
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in NOTE_FIELDS and key != "hint":
                continue
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def lint_copy(path: Path) -> list[str]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for where, text, is_hint in _strings(document, ""):
        problems += lint_text(text, where=f"{Path(path).name}:{where}", allow_cm1=is_hint)
    return problems


_JS_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"|\'((?:[^\'\\\n]|\\.)*)\'|`((?:[^`\\]|\\.)*)`', re.S)


def js_strings(source: str) -> Iterator[str]:
    """The literal text of every string in a script (template literals without their ${...} parts)."""

    body = re.sub(r"//[^\n]*", "", source)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    for match in _JS_STRING.finditer(body):
        text = next(group for group in match.groups() if group is not None)
        yield re.sub(r"\$\{[^}]*\}", " ", text)


def lint_js(path: Path) -> list[str]:
    source = Path(path).read_text(encoding="utf-8")
    problems = []
    for text in js_strings(source):
        problems += lint_text(text, where=Path(path).name)
    return problems


def lint_markdown(path: Path) -> list[str]:
    """A guide's prose: CM1 names may appear only in code spans and code blocks."""

    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        prose = re.sub(r"`[^`]*`", " ", line)
        problems += lint_text(prose, where=f"{Path(path).name}:{number}")
    return problems


def guided_files() -> list[Path]:
    files = sorted((GUI / "copy").glob("*.json")) + sorted((GUI / "static" / "guided").glob("*.js"))
    return files


def lint_file(path: Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".json":
        return lint_copy(path)
    if path.suffix == ".js":
        return lint_js(path)
    if path.suffix == ".md":
        return lint_markdown(path)
    return lint_text(path.read_text(encoding="utf-8"), where=path.name)


def lint_all(paths: Iterable[Path] | None = None) -> list[str]:
    problems = []
    for path in (list(paths) if paths else guided_files()):
        problems += lint_file(Path(path))
    return problems


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (sys.argv[1:] if argv is None else argv)]
    problems = lint_all(paths or None)
    for problem in problems:
        print(problem)
    if not problems:
        print(f"copy lint: {len(paths) or len(guided_files())} file(s), no problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BANNED", "ENGLISH_WORD_KEYS", "cm1_terms", "js_strings", "lint_all", "lint_copy", "lint_file",
           "lint_js", "lint_markdown", "lint_text", "viz20_text_lint"]
