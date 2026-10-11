"""Read host-specific strings out of historical receipts on any platform.

Receipts captured before the Linux port embed literal Windows absolute paths
such as ``C:\\...\\gpuwm-cm1\\evidence\\...``.  Their bytes are their
identity, so they are never rewritten; tests read the recorded string through
the helpers here instead.  ``PurePosixPath`` treats a backslash as an ordinary
name character, so ``Path(recorded).name`` returns the whole Windows string on
Linux and a basename comparison then silently fails instead of proving the
recorded artifact is the live one.

``child_environment`` exists for the same reason: the tools spawned by the
subprocess tests need the repository ``src`` tree *and* whatever the parent
interpreter already resolves imports from, because ArWen may be supplied on
``PYTHONPATH`` rather than installed.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath


def recorded_pure_path(recorded: str) -> PurePath:
    """Return a pure path for a receipt-recorded path from any capture host."""

    if "\\" in recorded:
        return PureWindowsPath(recorded)
    return PurePosixPath(recorded)


def recorded_name(recorded: str) -> str:
    """Return the final component of a receipt-recorded absolute path."""

    return recorded_pure_path(recorded).name


def recorded_relative_to(recorded: str, anchor: str) -> PurePosixPath:
    """Return the recorded path from ``anchor`` onward, in POSIX spelling.

    The capture host's drive and home prefix cannot be reproduced anywhere
    else, but everything from the repository-relative ``anchor`` component on
    is exact, so a receipt still binds to one named directory and file.
    """

    parts = recorded_pure_path(recorded).parts
    for index, part in enumerate(parts):
        if part == anchor:
            return PurePosixPath(*parts[index:])
    raise AssertionError(f"recorded path {recorded!r} has no {anchor!r} component")


def child_environment(*python_path_entries: object) -> dict[str, str]:
    """Return an environment that prepends ``python_path_entries`` to imports."""

    environment = os.environ.copy()
    entries = [str(entry) for entry in python_path_entries]
    inherited = environment.get("PYTHONPATH", "")
    if inherited:
        entries.extend(
            entry for entry in inherited.split(os.pathsep) if entry not in entries
        )
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def require_workspace_contained(root: object, *relatives: str) -> None:
    """Skip when a staged tree resolves outside the workspace root.

    The static-admission binder refuses any artifact directory whose resolved
    path escapes the workspace, which is the correct guard.  A checkout that
    stages ``work/`` as a symlink to an external tree therefore cannot exercise
    a full admission, so the case is skipped naming the resolved location.
    """

    import pytest

    base = Path(root).resolve()
    for relative in relatives:
        candidate = base / relative
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(base):
            pytest.skip(
                f"{relative} is staged outside the repository on this host "
                f"({candidate.resolve()}); cm1core.gm1_child_static_admission "
                "refuses an artifact directory whose resolved path escapes the "
                "workspace root"
            )


def require_binary32_power_is_double_rounded(base: object, exponent: object) -> None:
    """Skip when NumPy's binary32 power is not the double-rounded result.

    CM1's pinned gfortran build lowers ``REAL**REAL`` to glibc powf, whose
    result equals the binary64 power rounded once to binary32, and the CUDA
    ports transcribe that.  Some NumPy builds evaluate binary32 power in a
    vector loop that lands one ulp away, so on such a host a CPU diagnostic
    that calls ``numpy.power`` on binary32 operands is not a raw-bit reference
    for the device words and cannot be compared against them.
    """

    import numpy as np
    import pytest

    left = np.asarray(base, dtype=np.float32)
    right = np.float32(exponent)
    native = np.power(left, right)
    rounded = np.power(left.astype(np.float64), np.float64(right)).astype(np.float32)
    if np.array_equal(native.view(np.uint32), rounded.view(np.uint32)):
        return
    pytest.skip(
        f"numpy {np.__version__} evaluates binary32 x**{float(right)!r} up to one "
        "ulp away from the binary64 power rounded once to binary32, which is "
        "what glibc powf and the CUDA transcription produce; the NumPy CPU "
        "diagnostic is not a raw-bit reference on this host"
    )
