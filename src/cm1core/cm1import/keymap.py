"""The checked CM1 namelist key -> gpuwm-cm1 TOML key map.

The map itself is the record ``records/CM1-GM1-KEYMAP.json``, authored and
verified by ``tools/build_cm1_gm1_keymap.py`` against the packaged authority
inventory.  This module is the read side: it is what the one translator
(:mod:`cm1core.cm1import.translate`, used by both the namelist importer and
the differential harness) and ``differential.py keymap`` consult, so there is
exactly one map and it is data, not code.  ``cm1core.oracle.keymap`` is this
module under its old name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

RECORD = Path(__file__).resolve().parents[1] / "records" / "CM1-GM1-KEYMAP.json"


@dataclass(frozen=True)
class Entry:
    cm1_key: str
    cm1_group: str
    cm1_type: str
    gm1_key: str | None
    transform: str
    note: str
    affects_state: bool = True
    #: For an unmapped key: what GM1 integrates instead -- ``gm1_runs`` (the
    #: one value it always runs) or ``read_when`` (the CM1 selector values
    #: under which CM1 reads the key at all).  See
    #: tools/build_cm1_gm1_keymap.py UNMAPPED_BEHAVIOUR.
    gm1_behaviour: Mapping[str, Any] | None = None

    @property
    def mapped(self) -> bool:
        return self.gm1_key is not None


@dataclass(frozen=True)
class KeyMap:
    schema: str
    authority_tarball_sha256: str
    counts: Mapping[str, int]
    entries: tuple[Entry, ...]
    gm1_only: tuple[Mapping[str, str], ...]
    non_physical_families: tuple[Mapping[str, Any], ...]
    harness_owned_groups: tuple[str, ...] = ()

    def by_cm1(self, key: str) -> Entry | None:
        for entry in self.entries:
            if entry.cm1_key == key:
                return entry
        return None

    def by_gm1(self, key: str) -> Entry | None:
        for entry in self.entries:
            if entry.gm1_key == key:
                return entry
        return None

    @property
    def mapped(self) -> tuple[Entry, ...]:
        return tuple(e for e in self.entries if e.mapped)

    @property
    def unmapped(self) -> tuple[Entry, ...]:
        """The derived gap register: CM1 physical keys GM1 cannot express."""

        return tuple(e for e in self.entries if not e.mapped)


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> KeyMap:
    payload = json.loads((path or RECORD).read_text(encoding="utf-8"))
    return KeyMap(
        schema=payload["schema"],
        authority_tarball_sha256=payload["authority_tarball_sha256"],
        counts=payload["counts"],
        entries=tuple(Entry(**e) for e in payload["entries"]),
        gm1_only=tuple(payload["gm1_only"]),
        non_physical_families=tuple(payload["non_physical_families"]),
        harness_owned_groups=tuple(payload.get("harness_owned_groups", ())),
    )
