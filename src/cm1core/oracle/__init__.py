"""The CM1/GM1 differential harness.

One physical configuration is declared once (:mod:`.spec`), emitted as both a
CM1 r21.1 namelist and a gpuwm-cm1 TOML through the checked key map
(:mod:`.keymap`), run through the real CM1 binary (:mod:`.cm1_run`) and the
CUDA runtime, and compared field by field and time by time
(:mod:`.compare`).  :mod:`.horizon` computes how long the comparison must run
to be decisive for the term under test.

Nothing here imports CUDA, ArWen or NetCDF.  CM1 history is read in its
``output_format=1`` GrADS form, which is raw little-endian IEEE binary32 with
no container; GM1 history is read from the raw ``history-carriers.f32`` arena
its Rust publisher writes beside every frame.  Both sides therefore cross this
harness as raw words.
"""

from __future__ import annotations

__all__ = [
    "CASE_SCHEMA",
    "DifferentialCase",
    "load_case",
    "keymap",
]

from .spec import CASE_SCHEMA, DifferentialCase, load_case
from . import keymap
