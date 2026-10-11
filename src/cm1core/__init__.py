"""CM1 21.1 backend for the gpuwm/ArWen ecosystem.

The package is intentionally import-light: importing :mod:`cm1core` neither
opens a CUDA context nor imports ArWen's numerical runtime.
"""

from __future__ import annotations

from ._version import __version__

MATURITY = "port-in-progress"
CM1_RELEASE = "21.1"
CM1_AUTHORITY_SHA256 = (
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7"
)

__all__ = [
    "CM1_AUTHORITY_SHA256",
    "CM1_RELEASE",
    "MATURITY",
    "__version__",
]
