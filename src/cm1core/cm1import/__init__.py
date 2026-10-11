"""Read a CM1 namelist.input the way CM1 r21.1 would, and translate it.

:mod:`.namelist`
    the native ``cm1-r21-namelist inspect --json --unknown report`` parser;
:mod:`.defaults`
    r21.1's own value for every key a file does not set;
:mod:`.aliases`
    keys from CM1 r16-r20.3, read as their r21.1 settings;
:mod:`.versions`
    keys from CM1 r22.0: their class and r21.1 equivalent;
:mod:`.translate`
    the one CM1 -> gpuwm-cm1 translator, shared with the differential harness.

No cm1.exe, CM1 source tree or ``CM1_AUTHORITY_ROOT`` is needed.
"""

from __future__ import annotations

__all__ = [
    "ComponentChoice",
    "NamelistImport",
    "PTYPE_COMPONENTS",
    "Translation",
    "TranslationError",
    "component_for",
    "import_namelist",
    "translate_parsed",
]

from .translate import (
    PTYPE_COMPONENTS,
    ComponentChoice,
    NamelistImport,
    Translation,
    TranslationError,
    component_for,
    import_namelist,
    translate_parsed,
)

# ``translate`` stays the submodule; the function is ``translate.translate``.
