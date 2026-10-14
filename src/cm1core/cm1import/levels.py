"""CM1's ``input_grid_z``: the level heights of ``stretch_z = 3`` and ``4``.

CM1 r21.1 opens ``input_grid_z`` in its run folder and reads one height per
record, in metres: ``nz`` scalar levels for ``stretch_z = 3``
(``param.F:8013-8026``) and ``nz + 1`` w levels, the first at the ground, for
``stretch_z = 4`` (``param.F:8083-8089``).  Each read is list-directed and
takes the first number of its record, so anything after it on the line is
ignored, and blank lines are skipped.

The importer looks for the file beside the namelist, keeps a copy in the run
folder, and hands the heights to the deck as ``grid.levels_m``.
"""

from __future__ import annotations

import re
from pathlib import Path

LEVELS_FILE = "input_grid_z"

_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?$")


class LevelsFileError(ValueError):
    """An ``input_grid_z`` CM1 itself would stop on, said in plain words."""


def locate(namelist: Path, rundir: Path | None = None) -> Path | None:
    """``input_grid_z`` beside the namelist, else the copy a run folder already keeps."""

    beside = Path(namelist).expanduser().resolve().parent / LEVELS_FILE
    if beside.is_file():
        return beside
    if rundir is not None and (Path(rundir) / LEVELS_FILE).is_file():
        return Path(rundir) / LEVELS_FILE
    return None


def expected_count(stretch_z: int, nz: int) -> int:
    return int(nz) if int(stretch_z) == 3 else int(nz) + 1


def parse(data: bytes, *, stretch_z: int, nz: int) -> tuple[float, ...]:
    """The heights CM1 would read from ``data`` for this ``stretch_z`` and ``nz``."""

    count = expected_count(stretch_z, nz)
    kind = "scalar levels" if int(stretch_z) == 3 else "w levels"
    values: list[float] = []
    for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), start=1):
        if len(values) == count:
            break
        tokens = line.replace(",", " ").split()
        if not tokens:
            continue
        token = tokens[0]
        if not _NUMBER.match(token):
            raise LevelsFileError(
                f"{LEVELS_FILE} line {number} does not start with a height in metres: {line.strip()[:60]!r}"
            )
        values.append(float(token.replace("D", "e").replace("d", "e")))
    if len(values) < count:
        raise LevelsFileError(
            f"{LEVELS_FILE} holds {len(values)} heights, and stretch_z = {stretch_z} with nz = {nz} reads "
            f"{count} {kind} (param.F:{'8013-8026' if int(stretch_z) == 3 else '8083-8089'})"
        )
    for index in range(1, count):
        if not values[index] > values[index - 1]:
            raise LevelsFileError(
                f"{LEVELS_FILE} heights must rise: height {index + 1} is {values[index]} m after {values[index - 1]} m"
            )
    if int(stretch_z) == 4 and values[0] != 0.0:
        raise LevelsFileError(f"{LEVELS_FILE} for stretch_z = 4 lists the w levels, and the first is the ground, 0 m")
    if int(stretch_z) == 3 and not values[0] > 0.0:
        raise LevelsFileError(f"{LEVELS_FILE} for stretch_z = 3 lists the scalar levels, and the first is above the ground")
    return tuple(values)


__all__ = ["LEVELS_FILE", "LevelsFileError", "expected_count", "locate", "parse"]
