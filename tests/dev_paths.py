"""Development-repository inputs a few tests read, by name.

The development repository keeps measurement records and working inputs that
the public source snapshot does not carry.  A test asks for one by name with
:func:`dev_path`; the table that maps names to files (``tests/dev_records.py``)
stays in the development repository too.  In the public snapshot the table is
absent, the import raises ``ModuleNotFoundError``, and
``tests/private_tree_skips.py`` reports that test as a skip because the stage
recorded the table as left out.  Anywhere else a missing table is a failure.
"""

from __future__ import annotations

from pathlib import Path


def dev_path(name: str) -> Path:
    """The development-repository file or folder recorded under ``name``."""

    from dev_records import PATHS  # absent from the public snapshot

    return PATHS[name]
