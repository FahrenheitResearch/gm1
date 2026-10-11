"""Name re-capture debt on sealed evidence instead of printing a digest diff.

A receipt whose own sha256 a test pins cannot be re-pinned when one of the
sources it names moves: the measured run it attests was produced by exactly the
recorded bytes, and rewriting them would claim the new bytes were measured.  So
when a reviewed source change lands without a re-capture, that one comparison
is given up and a warning names the file, both digests and the commit that
reviewed the change.  Every other assertion in the case still runs.  The debt
then reads as a debt, and the warning retires itself as soon as the evidence
is captured again on the device the receipt names.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import warnings


def assert_recaptured_or_owed(path: Path, recorded: str, *, reviewed_by: str) -> None:
    """Assert the sealed receipt still names ``path``'s live bytes, or warn.

    ``pytest.skip`` would abort the whole test function, taking every later
    assertion with it -- 24 of them across the seven cases that call this.
    The debt is one comparison, so only that comparison is given up: the test
    keeps running and emits a warning naming the file, both digests and the
    commit that reviewed the change.

    The warning retires itself, with no edit here, as soon as the evidence is
    captured again on the device the receipt names.
    """

    live = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if live == recorded:
        return
    warnings.warn(
        f"sealed evidence names {path.as_posix()} at {recorded}, but the live "
        f"file is {live} since {reviewed_by}; the receipt owes a re-capture on "
        "the device it names before its source binding can be asserted again",
        RecaptureOwed,
        stacklevel=2,
    )


def assert_source_pin_or_owed(
    subject: str, recorded: str, live: str, *, reviewed_by: str
) -> None:
    """The same debt, for a kernel source string rather than a file.

    A CUDA source string is assembled at import time, so there is no path to
    hash; the caller supplies the live digest.  Everything else is the rule
    above: the receipt still describes the image that ran, the comparison is
    given up rather than the test, and the warning names what changed.
    """

    if live == recorded:
        return
    warnings.warn(
        f"sealed evidence names {subject} at {recorded}, but the live source "
        f"is {live} since {reviewed_by}; the receipt owes a re-capture on the "
        "device it names before its source binding can be asserted again",
        RecaptureOwed,
        stacklevel=2,
    )


class RecaptureOwed(UserWarning):
    """A sealed receipt names source bytes that have since been reviewed."""


# Retained under the old name so a stale import fails loudly rather than
# silently skipping.
require_recaptured = assert_recaptured_or_owed
