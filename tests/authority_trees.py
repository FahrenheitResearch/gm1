"""Skip markers for tests that read gitignored local assets.

A fresh checkout has neither the CM1 r21.1 authority tree (``upstream/``,
gitignored: CM1 source is not redistributed) nor the ``work/`` capture
directories (gitignored: gigabytes of device and native-reference output).
Tests that read them skip with a reason that says how to supply them, rather
than failing with FileNotFoundError or an "authority hash mismatch" that
reads like a regression.

Environment variables (``tests/conftest.py`` links each one into the
gitignored location the tools read, when that location is absent):

* ``CM1_R21_1_PRISTINE``  -- an extracted, unmodified cm1r21.1 tree
  (the directory holding ``src/``, ``run/`` and ``README.namelist``);
* ``CM1_R21_1_TARBALL``   -- the pinned ``cm1r21.1.tar.gz``
  (``tools/authority/fetch_cm1_21_1.py`` downloads and verifies it);
* ``CM1_AUTHORITY_ROOT``  -- the built native-reference CM1 tree
  (``work/native-reference/cm1r21.1``);
* ``GM1_WORK_ROOT``       -- a ``work/`` directory of captured evidence assets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CM1_PRISTINE = ROOT / "upstream" / "cm1-21.1" / "pristine"
CM1_TARBALL = ROOT / "upstream" / "cm1-21.1" / "cm1r21.1.tar.gz"
WORK = ROOT / "work"
NATIVE_REFERENCE = WORK / "native-reference" / "cm1r21.1"

#: Gitignored location -> the environment variable that can supply it.
LINKABLE = {
    CM1_PRISTINE: "CM1_R21_1_PRISTINE",
    CM1_TARBALL: "CM1_R21_1_TARBALL",
    NATIVE_REFERENCE: "CM1_AUTHORITY_ROOT",
    WORK: "GM1_WORK_ROOT",
}


def link_supplied_assets() -> None:
    """Link env-supplied assets into their gitignored locations, if absent."""

    import os

    for location, variable in LINKABLE.items():
        supplied = os.environ.get(variable)
        if not supplied or location.exists() or location.is_symlink():
            continue
        target = Path(supplied).expanduser().resolve()
        if not target.exists():
            continue
        location.parent.mkdir(parents=True, exist_ok=True)
        location.symlink_to(target, target_is_directory=target.is_dir())


# The markers below are formed when this module is imported, which conftest
# does before it links anything; so the links are made here first, or a fresh
# checkout given the environment variables would skip every test they gate.
link_supplied_assets()


requires_cm1_pristine = pytest.mark.skipif(
    not CM1_PRISTINE.is_dir(),
    reason=(
        "requires CM1 authority tree: set CM1_R21_1_PRISTINE to an extracted "
        "cm1r21.1 (or extract it to upstream/cm1-21.1/pristine)"
    ),
)

requires_cm1_tarball = pytest.mark.skipif(
    not CM1_TARBALL.is_file(),
    reason=(
        "requires the pinned CM1 release tarball: set CM1_R21_1_TARBALL "
        "(tools/authority/fetch_cm1_21_1.py downloads and verifies it)"
    ),
)

requires_native_reference = pytest.mark.skipif(
    not NATIVE_REFERENCE.is_dir(),
    reason=(
        "requires the built native-reference CM1 tree: set CM1_AUTHORITY_ROOT "
        "(work/native-reference/cm1r21.1 is gitignored)"
    ),
)


def requires_work(*relative: str) -> pytest.MarkDecorator:
    """Skip unless every named ``work/`` asset is present."""

    missing = [name for name in relative if not (WORK / name).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=(
            f"requires local work/ assets {missing}: set GM1_WORK_ROOT to a "
            "captured work/ directory (work/ is gitignored)"
        ),
    )


__all__ = [
    "CM1_PRISTINE",
    "CM1_TARBALL",
    "NATIVE_REFERENCE",
    "ROOT",
    "WORK",
    "link_supplied_assets",
    "requires_cm1_pristine",
    "requires_cm1_tarball",
    "requires_native_reference",
    "requires_work",
]
