"""CM1 r21.1 statements named by a mark, so the tests that anchor GM1 to them carry no CM1 source text.

A test that checks a statement is still in CM1's source (or still in a cited line range, or in a given order) reads
the source from the pristine CM1 tree, as before, but names the statement by its mark: the first 16 hex digits of
the sha256 of the statement with every whitespace character removed, a slash, and the length of that squashed text.
``mark`` computes one; ``find``, ``has`` and ``is_line`` look one up.  Matching ignores whitespace, as Fortran does,
and is case-sensitive, as the quoted text was.
"""

from __future__ import annotations

import hashlib
import re

_SPACE = re.compile(r"\s+")


def squash(text: str) -> str:
    return _SPACE.sub("", text)


def _digest(squashed: str) -> str:
    return hashlib.sha256(squashed.encode("utf-8")).hexdigest()[:16]


def mark(text: str) -> str:
    """The mark of ``text`` (what a test writes in place of the statement)."""

    squashed = squash(text)
    return f"{_digest(squashed)}/{len(squashed)}"


def _parse(marker: str) -> tuple[str, int]:
    digest, _, length = marker.partition("/")
    return digest, int(length)


def find(text: str, marker: str) -> int:
    """Where the statement ``marker`` names first starts in ``squash(text)``, or -1."""

    digest, length = _parse(marker)
    squashed = squash(text)
    for start in range(0, len(squashed) - length + 1):
        if _digest(squashed[start:start + length]) == digest:
            return start
    return -1


def has(text: str, marker: str) -> bool:
    """True when ``text`` contains the statement ``marker`` names (whitespace aside)."""

    return find(text, marker) >= 0


def is_line(line: str, marker: str) -> bool:
    """True when ``line`` is exactly the statement ``marker`` names (whitespace aside)."""

    digest, length = _parse(marker)
    squashed = squash(line)
    return len(squashed) == length and _digest(squashed) == digest
