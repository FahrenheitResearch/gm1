"""The checked CM1 namelist key -> gpuwm-cm1 TOML key map.

The reader moved to :mod:`cm1core.cm1import.keymap` when the translator did;
this name stays for the harness and its tools.
"""

from __future__ import annotations

from ..cm1import.keymap import RECORD, Entry, KeyMap, load

__all__ = ["RECORD", "Entry", "KeyMap", "load"]
